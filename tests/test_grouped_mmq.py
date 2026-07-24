import contextlib
import os
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor
from transformers.integrations.moe import _grouped_linear

import torch_ggml_ops

_MODEL = Path(
    os.environ.get(
        "GGUF_MMQ_TEST_MODEL",
        os.path.expanduser("~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"),
    )
)
_PROJECTIONS = {
    "Q3_K": "blk.0.ffn_gate_exps.weight",
    "Q4_K": "blk.2.ffn_down_exps.weight",
    "Q5_K": "blk.0.ffn_down_exps.weight",
    "Q6_K": "output.weight",
    "IQ2_S": "blk.10.ffn_gate_exps.weight",
}
_PAIR_PROJECTIONS = {
    "Q3_K": ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"),
    "Q4_K": ("blk.0.attn_gate.weight", "blk.0.attn_qkv.weight"),
    "Q5_K": ("blk.0.ffn_down_exps.weight", "blk.1.ffn_down_exps.weight"),
    "Q6_K": ("output.weight", "output.weight"),
    "IQ2_S": ("blk.10.ffn_gate_exps.weight", "blk.10.ffn_up_exps.weight"),
}


def _require_runtime() -> None:
    if not _MODEL.is_file():
        pytest.skip("GGUF model is unavailable")


@pytest.fixture(scope="module")
def reader() -> gguf.GGUFReader:
    _require_runtime()
    return gguf.GGUFReader(_MODEL)


def _packed_experts(
    reader: gguf.GGUFReader,
    tensor_name: str,
    *,
    num_experts: int = 8,
    out_features: int = 37,
    row_offset: int = 0,
) -> tuple[torch.Tensor, gguf.GGMLQuantizationType, int]:
    tensor = next(t for t in reader.tensors if t.name == tensor_name)
    row_slice = slice(row_offset, row_offset + out_features)
    if tensor.data.ndim == 3:
        host = np.array(
            tensor.data[:num_experts, row_slice],
            dtype=np.uint8,
            copy=True,
            order="C",
        )
    else:
        one_expert = np.array(
            tensor.data[row_slice],
            dtype=np.uint8,
            copy=True,
            order="C",
        )
        host = np.repeat(one_expert[None, ...], num_experts, axis=0)
    return (
        torch.from_numpy(host).to("cuda"),
        tensor.tensor_type,
        int(tensor.shape[0]),
    )


def _group_metadata() -> tuple[torch.Tensor, torch.Tensor]:
    experts = torch.tensor([0, 2, 5, 7], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([1, 128, 257, 262], device="cuda", dtype=torch.int32)
    return experts, offsets


def _pair_packed_experts(
    reader: gguf.GGUFReader, qname: str, *, out_features: int = 37
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    gguf.GGMLQuantizationType,
    int,
]:
    first_name, second_name = _PAIR_PROJECTIONS[qname]
    first, quant_type, in_features = _packed_experts(
        reader, first_name, out_features=out_features
    )
    second, second_quant_type, second_in_features = _packed_experts(
        reader,
        second_name,
        out_features=out_features,
        row_offset=out_features if second_name == first_name else 0,
    )
    assert quant_type.name == qname
    assert second_quant_type == quant_type
    assert second_in_features == in_features
    return first, second, quant_type, in_features


def _logical_grouped_forward(
    input: torch.Tensor,
    packed: torch.Tensor,
    experts: torch.Tensor,
    offsets: torch.Tensor,
    quant_type: gguf.GGMLQuantizationType,
    out_features: int,
) -> torch.Tensor:
    logical = dequantize_gguf_tensor(
        packed.index_select(0, experts),
        quant_type,
        dtype=input.dtype,
        device=input.device,
    ).reshape(experts.numel(), out_features, input.shape[1])
    return _grouped_linear(input, logical, offsets)


@contextlib.contextmanager
def _accurate_bf16_reduction():
    """Stop torch from reducing bf16 matmul partial sums in bf16.

    PyTorch defaults allow_bf16_reduced_precision_reduction to True, which lets
    the backend accumulate split-K partials in bf16. On this reference that costs
    real accuracy: scored against an fp64 accumulation of the same products, the
    default reference gets 63% of its outputs correctly rounded, and 100% with
    the flag off. Without this the test compares the kernel against an oracle
    that is itself wrong on ~38% of elements.
    """
    key = "allow_bf16_reduced_precision_reduction"
    previous = getattr(torch.backends.cuda.matmul, key)
    setattr(torch.backends.cuda.matmul, key, False)
    try:
        yield
    finally:
        setattr(torch.backends.cuda.matmul, key, previous)


def _logical_grouped_input_gradient(
    grad_output: torch.Tensor,
    logical_weight: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    grad_input = torch.empty(
        grad_output.shape[0],
        logical_weight.shape[-1],
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    row_begin = 0
    for group, row_end in enumerate(offsets.cpu().tolist()):
        grad_input[row_begin:row_end] = (
            grad_output[row_begin:row_end] @ logical_weight[group]
        )
        row_begin = row_end
    return grad_input


def _logical_grouped_pair_input_gradient(
    first_grad_output: torch.Tensor,
    second_grad_output: torch.Tensor,
    first_logical_weight: torch.Tensor,
    second_logical_weight: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    grad_input = torch.empty(
        first_grad_output.shape[0],
        first_logical_weight.shape[-1],
        device=first_grad_output.device,
        dtype=first_grad_output.dtype,
    )
    row_begin = 0
    for group, row_end in enumerate(offsets.cpu().tolist()):
        combined = (
            first_grad_output[row_begin:row_end].float()
            @ first_logical_weight[group].float()
        )
        combined.addmm_(
            second_grad_output[row_begin:row_end].float(),
            second_logical_weight[group].float(),
        )
        grad_input[row_begin:row_end] = combined.to(first_grad_output.dtype)
        row_begin = row_end
    return grad_input


@pytest.mark.parametrize("qname", tuple(_PROJECTIONS))
def test_grouped_forward_matches_q8_activation_reference_error(
    reader: gguf.GGUFReader, qname: str
) -> None:
    packed, quant_type, in_features = _packed_experts(reader, _PROJECTIONS[qname])
    experts, offsets = _group_metadata()
    generator = torch.Generator(device="cuda").manual_seed(1234)
    input = torch.randn(
        262,
        in_features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    expected = _logical_grouped_forward(input, packed, experts, offsets, quant_type, 37)
    actual = torch_ggml_ops.grouped_mmq(
        input, packed, experts, offsets, int(quant_type), 37
    )
    error = actual.float() - expected.float()
    normalized_rmse = (
        error.square().mean().sqrt() / expected.float().square().mean().sqrt()
    )

    assert actual.shape == (262, 37)
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    assert normalized_rmse.item() < 0.04


@pytest.mark.parametrize("qname", tuple(_PAIR_PROJECTIONS))
def test_grouped_pair_matches_two_single_projections(
    reader: gguf.GGUFReader, qname: str
) -> None:
    first, second, quant_type, in_features = _pair_packed_experts(reader, qname)
    experts, offsets = _group_metadata()
    generator = torch.Generator(device="cuda").manual_seed(3456)
    input = torch.randn(
        262,
        in_features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    actual_first, actual_second = torch_ggml_ops.grouped_mmq_pair(
        input, first, second, experts, offsets, int(quant_type), 37
    )
    expected_first = torch_ggml_ops.grouped_mmq(
        input, first, experts, offsets, int(quant_type), 37
    )
    expected_second = torch_ggml_ops.grouped_mmq(
        input, second, experts, offsets, int(quant_type), 37
    )

    torch.testing.assert_close(actual_first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(actual_second, expected_second, rtol=0, atol=0)


def test_grouped_pair_production_row_tasks_match_dense(
    reader: gguf.GGUFReader,
) -> None:
    gate, quant_type, in_features = _packed_experts(
        reader,
        "blk.0.ffn_gate_exps.weight",
        out_features=512,
    )
    up, up_quant_type, up_in_features = _packed_experts(
        reader,
        "blk.0.ffn_up_exps.weight",
        out_features=512,
    )
    assert up_quant_type == quant_type
    assert up_in_features == in_features == 2048

    experts = torch.tensor([0, 2, 5, 7], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([65, 194, 323, 512], device="cuda", dtype=torch.int32)
    generator = torch.Generator(device="cuda").manual_seed(2468)
    input = torch.randn(
        512,
        in_features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    actual_gate, actual_up = torch_ggml_ops.grouped_mmq_pair(
        input, gate, up, experts, offsets, int(quant_type), 512
    )

    expected_gate_parts = []
    expected_up_parts = []
    row_begin = 0
    for expert, row_end in zip(experts.cpu().tolist(), offsets.cpu().tolist()):
        group_input = input[row_begin:row_end].clone()
        expected_gate_parts.append(
            torch_ggml_ops.mmq(group_input, gate[expert].clone(), int(quant_type), 512)
        )
        expected_up_parts.append(
            torch_ggml_ops.mmq(group_input, up[expert].clone(), int(quant_type), 512)
        )
        row_begin = row_end

    torch.testing.assert_close(
        actual_gate, torch.cat(expected_gate_parts), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_up, torch.cat(expected_up_parts), rtol=0, atol=0)


@pytest.mark.parametrize("qname", tuple(_PROJECTIONS))
def test_grouped_backward_is_logical_weight_jacobian(
    reader: gguf.GGUFReader, qname: str
) -> None:
    packed, quant_type, in_features = _packed_experts(reader, _PROJECTIONS[qname])
    experts = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([2, 5, 6], device="cuda", dtype=torch.int32)
    generator = torch.Generator(device="cuda").manual_seed(5678)
    input = torch.randn(
        6,
        in_features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    grad_output = torch.randn(
        6,
        37,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    output = torch_ggml_ops.grouped_mmq(
        input, packed, experts, offsets, int(quant_type), 37
    )
    output.backward(grad_output)
    logical = dequantize_gguf_tensor(
        packed.index_select(0, experts),
        quant_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(3, 37, in_features)
    expected_grad = _logical_grouped_input_gradient(grad_output, logical, offsets)

    torch.testing.assert_close(input.grad, expected_grad, rtol=0, atol=0)
    assert packed.grad is None


@pytest.mark.parametrize("qname", tuple(_PAIR_PROJECTIONS))
def test_grouped_pair_backward_sums_both_logical_jacobians(
    reader: gguf.GGUFReader, qname: str
) -> None:
    first, second, quant_type, in_features = _pair_packed_experts(reader, qname)
    experts = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([2, 5, 6], device="cuda", dtype=torch.int32)
    generator = torch.Generator(device="cuda").manual_seed(9012)
    input = torch.randn(
        6,
        in_features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    first_grad = torch.randn(
        6, 37, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    second_grad = torch.randn(
        6, 37, generator=generator, device="cuda", dtype=torch.bfloat16
    )

    first_output, second_output = torch_ggml_ops.grouped_mmq_pair(
        input, first, second, experts, offsets, int(quant_type), 37
    )
    torch.autograd.backward((first_output, second_output), (first_grad, second_grad))
    logical_first = dequantize_gguf_tensor(
        first.index_select(0, experts),
        quant_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(3, 37, in_features)
    logical_second = dequantize_gguf_tensor(
        second.index_select(0, experts),
        quant_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(3, 37, in_features)
    expected_grad = _logical_grouped_pair_input_gradient(
        first_grad,
        second_grad,
        logical_first,
        logical_second,
        offsets,
    )

    # The fused kernel accumulates both logical Jacobians in one FP32 WMMA
    # accumulator and rounds once to BF16. Torch evaluates two GEMMs before the
    # sum, so reduction and rounding order need an error-based comparison.
    input_grad = input.grad
    assert input_grad is not None
    error = input_grad.float() - expected_grad.float()
    normalized_rmse = (
        error.square().mean().sqrt() / expected_grad.float().square().mean().sqrt()
    )
    assert normalized_rmse.item() < 5e-5
    assert error.abs().max().item() <= 2**-12
    assert first.grad is None
    assert second.grad is None


def test_grouped_backward_route_group_boundaries(
    reader: gguf.GGUFReader,
) -> None:
    packed, quant_type, in_features = _packed_experts(
        reader,
        "blk.2.ffn_down_exps.weight",
        num_experts=12,
        out_features=37,
    )
    group_size_values = (1, 15, 16, 17, 63, 64, 65, 127, 128, 129)
    group_sizes = torch.tensor(
        group_size_values,
        device="cuda",
        dtype=torch.int32,
    )
    offsets = group_sizes.cumsum(0).to(torch.int32).contiguous()
    experts = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 11],
        device="cuda",
        dtype=torch.int64,
    )
    generator = torch.Generator(device="cuda").manual_seed(8642)
    grad_output = torch.randn(
        sum(group_size_values),
        37,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    actual = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
        grad_output,
        packed,
        experts,
        offsets,
        int(quant_type),
        in_features,
    )
    logical = dequantize_gguf_tensor(
        packed.index_select(0, experts),
        quant_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(experts.numel(), 37, in_features)
    with _accurate_bf16_reduction():
        expected = _logical_grouped_input_gradient(grad_output, logical, offsets)

    # One bf16 ULP, not bit-exact: two valid fp32 orderings of the same 37
    # products can land on opposite sides of a rounding boundary. Measured
    # residue on this case is 4 / 320000 elements, each exactly 1.00 ULP.
    torch.testing.assert_close(actual, expected, rtol=2**-7, atol=0)


# The tiled / row-task backward kernels are gated on the exact down-projection
# geometry -- out_features == 2048 and in_features == 512 -- plus a row-count
# threshold (use_grouped_backward_row_tasks in mmq_hip.cu, and the dispatch in
# launch_grouped_mmq_grad_input, ck/grouped_mmq_backward.cuh). Every other test in
# this file uses out_features=37, which cannot satisfy that gate, so those kernels
# -- the ones a real MoE down projection runs -- would otherwise go untested.
_DOWN_PROJECTIONS = {
    "Q4_K": "blk.2.ffn_down_exps.weight",
    "Q5_K": "blk.0.ffn_down_exps.weight",
    "IQ2_S": "blk.10.ffn_down_exps.weight",
}
_DOWN_GROUPS = 4
_DOWN_OUT_FEATURES = 2048


@pytest.mark.parametrize("qname", tuple(_DOWN_PROJECTIONS))
# Three row regimes around the dispatch thresholds. Which kernel each one selects
# differs per quant type; captured with torch.profiler on an RTX 4090, these nine
# cases cover eight distinct kernels, none of which any other test reaches:
#   rows/group  Q4_K              Q5_K            IQ2_S
#   160         q4_row_task       q5_row_task     iq2_row_task
#   100         q4_small_s2       q5_small        iq2_s2
#    40         q4_small          q5_small        iq2_tiled<true>
@pytest.mark.parametrize("rows_per_group", (160, 100, 40))
def test_grouped_backward_down_projection_selects_tiled_kernels(
    reader: gguf.GGUFReader, qname: str, rows_per_group: int
) -> None:
    packed, quant_type, in_features = _packed_experts(
        reader,
        _DOWN_PROJECTIONS[qname],
        num_experts=_DOWN_GROUPS,
        out_features=_DOWN_OUT_FEATURES,
    )
    assert quant_type.name == qname
    assert in_features == 512  # or the dispatch gate above is not being exercised

    rows = rows_per_group * _DOWN_GROUPS
    experts = torch.arange(_DOWN_GROUPS, device="cuda", dtype=torch.int64)
    offsets = torch.tensor(
        [(index + 1) * rows_per_group for index in range(_DOWN_GROUPS)],
        device="cuda",
        dtype=torch.int32,
    )
    generator = torch.Generator(device="cuda").manual_seed(97531)
    grad_output = torch.randn(
        rows,
        _DOWN_OUT_FEATURES,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    actual = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
        grad_output, packed, experts, offsets, int(quant_type), in_features
    )

    logical = dequantize_gguf_tensor(
        packed.index_select(0, experts),
        quant_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(experts.numel(), _DOWN_OUT_FEATURES, in_features)
    with _accurate_bf16_reduction():
        expected = _logical_grouped_input_gradient(grad_output, logical, offsets)

    assert actual.shape == (rows, in_features)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    # One bf16 ULP relative, plus a small absolute floor: this reduces over
    # K=2048, so a good fraction of the outputs are near-zero cancellations where
    # relative error is unbounded for any reduction order. Measured worst case
    # over the three quant types and three row regimes is 1.75e-6 absolute,
    # against an output RMS of ~0.33-0.48.
    torch.testing.assert_close(actual, expected, rtol=2**-7, atol=2**-18)


def test_grouped_grad_input_direct_ops_compose_and_reject_higher_order(
    reader: gguf.GGUFReader,
) -> None:
    gate, quant_type, in_features = _packed_experts(
        reader, "blk.10.ffn_gate_exps.weight", out_features=64
    )
    up, _, _ = _packed_experts(reader, "blk.10.ffn_up_exps.weight", out_features=64)
    experts = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
    first_grad = torch.randn(
        4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    second_grad = torch.randn(
        4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    single_result = torch.library.opcheck(
        torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default,
        (first_grad.detach(), gate, experts, offsets, int(quant_type), in_features),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in single_result.values()), single_result

    pair_result = torch.library.opcheck(
        torch.ops.torch_ggml_ops.grouped_mmq_pair_grad_input.default,
        (
            first_grad.detach(),
            second_grad.detach(),
            gate,
            up,
            experts,
            offsets,
            int(quant_type),
            in_features,
        ),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in pair_result.values()), pair_result

    single_grad_input = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
        first_grad,
        gate,
        experts,
        offsets,
        int(quant_type),
        in_features,
    )
    with pytest.raises(RuntimeError, match="does not support higher-order"):
        single_grad_input.sum().backward()

    pair_grad_input = torch.ops.torch_ggml_ops.grouped_mmq_pair_grad_input.default(
        first_grad,
        second_grad,
        gate,
        up,
        experts,
        offsets,
        int(quant_type),
        in_features,
    )
    with pytest.raises(RuntimeError, match="does not support higher-order"):
        pair_grad_input.sum().backward()


def test_grouped_grad_input_uses_current_stream_and_rejects_invalid_operands(
    reader: gguf.GGUFReader,
) -> None:
    packed, quant_type, in_features = _packed_experts(
        reader, "blk.2.ffn_down_exps.weight", out_features=37
    )
    experts = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
    grad_output = torch.randn(4, 37, device="cuda", dtype=torch.bfloat16)
    op = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream_grad = torch.ones(4, 37, device="cuda", dtype=torch.bfloat16)
        stream_result = op(
            stream_grad,
            packed,
            experts,
            offsets,
            int(quant_type),
            in_features,
        )
        checksum = stream_result.float().abs().sum()
    stream.synchronize()
    assert checksum.item() > 0

    with pytest.raises(RuntimeError, match="contiguous"):
        op(
            torch.randn(4, 74, device="cuda", dtype=torch.bfloat16)[:, ::2],
            packed,
            experts,
            offsets,
            int(quant_type),
            in_features,
        )
    with pytest.raises(RuntimeError, match="zero storage offset"):
        op(
            torch.randn(5, 37, device="cuda", dtype=torch.bfloat16)[1:],
            packed,
            experts,
            offsets,
            int(quant_type),
            in_features,
        )
    with pytest.raises(RuntimeError, match="torch.int32"):
        op(
            grad_output,
            packed,
            experts,
            offsets.long(),
            int(quant_type),
            in_features,
        )
    with pytest.raises(RuntimeError, match="bytes per row"):
        op(
            grad_output,
            packed[..., :-1].contiguous(),
            experts,
            offsets,
            int(quant_type),
            in_features,
        )


def test_grouped_pair_opcheck_and_compile(reader: gguf.GGUFReader) -> None:
    gate, quant_type, in_features = _packed_experts(
        reader, "blk.0.ffn_gate_exps.weight", out_features=64
    )
    up, _, _ = _packed_experts(reader, "blk.0.ffn_up_exps.weight", out_features=64)
    experts = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
    input = torch.randn(
        4,
        in_features,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    result = torch.library.opcheck(
        torch.ops.torch_ggml_ops.grouped_mmq_pair.default,
        (input, gate, up, experts, offsets, int(quant_type), 64),
        test_utils=(
            "test_schema",
            "test_autograd_registration",
            "test_faketensor",
            "test_aot_dispatch_dynamic",
        ),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result

    @torch.compile(fullgraph=True)
    def compiled(
        input: torch.Tensor,
        gate: torch.Tensor,
        up: torch.Tensor,
        experts: torch.Tensor,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch_ggml_ops.grouped_mmq_pair(
            input, gate, up, experts, offsets, int(quant_type), 64
        )

    expected = torch_ggml_ops.grouped_mmq_pair(
        input.detach(), gate, up, experts, offsets, int(quant_type), 64
    )
    actual = compiled(input.detach(), gate, up, experts, offsets)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_grouped_opcheck_compile_and_invalid_metadata(
    reader: gguf.GGUFReader,
) -> None:
    packed, quant_type, in_features = _packed_experts(
        reader, "blk.2.ffn_down_exps.weight"
    )
    experts = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
    input = torch.randn(
        4,
        in_features,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    result = torch.library.opcheck(
        torch.ops.torch_ggml_ops.grouped_mmq.default,
        (input, packed, experts, offsets, int(quant_type), 37),
        test_utils=(
            "test_schema",
            "test_autograd_registration",
            "test_faketensor",
            "test_aot_dispatch_dynamic",
        ),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result

    @torch.compile(fullgraph=True)
    def compiled(
        input: torch.Tensor,
        packed: torch.Tensor,
        experts: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        return torch_ggml_ops.grouped_mmq(
            input, packed, experts, offsets, int(quant_type), 37
        )

    expected = torch_ggml_ops.grouped_mmq(
        input.detach(), packed, experts, offsets, int(quant_type), 37
    )
    actual = compiled(input.detach(), packed, experts, offsets)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="torch.int32"):
        torch_ggml_ops.grouped_mmq(
            input.detach(), packed, experts, offsets.long(), int(quant_type), 37
        )
    with pytest.raises(RuntimeError, match="zero storage offset"):
        torch_ggml_ops.grouped_mmq(
            input.detach(),
            packed,
            experts,
            torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)[1:],
            int(quant_type),
            37,
        )
