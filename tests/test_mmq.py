import os
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops

_MODEL = Path(
    os.environ.get(
        "GGUF_MMQ_TEST_MODEL",
        os.path.expanduser("~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"),
    )
)
_TENSORS = {
    "IQ2_S": "blk.10.ffn_gate_exps.weight",
    "Q3_K": "blk.3.attn_q.weight",
    "Q4_K": "blk.0.attn_gate.weight",
    "Q5_K": "blk.4.attn_qkv.weight",
    "Q6_K": "output.weight",
}


def _require_runtime() -> None:
    if not _MODEL.is_file():
        pytest.skip("GGUF model is unavailable")


@pytest.fixture(scope="module")
def reader() -> gguf.GGUFReader:
    _require_runtime()
    return gguf.GGUFReader(_MODEL)


def _packed_rows(
    reader: gguf.GGUFReader, qname: str, out_features: int
) -> tuple[torch.Tensor, gguf.GGMLQuantizationType]:
    tensor = next(t for t in reader.tensors if t.name == _TENSORS[qname])
    if tensor.data.ndim == 3:
        data = tensor.data[0, :out_features]
    else:
        data = tensor.data[:out_features]
    host = np.array(data, dtype=np.uint8, copy=True, order="C")
    return torch.from_numpy(host).to("cuda"), tensor.tensor_type


@pytest.mark.parametrize("qname", tuple(_TENSORS))
def test_forward_matches_q8_activation_reference_error(
    reader: gguf.GGUFReader, qname: str
) -> None:
    packed, qtype = _packed_rows(reader, qname, out_features=37)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    input = torch.randn(
        3, 43, 2048, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    logical_weight = dequantize_gguf_tensor(
        packed, qtype, dtype=torch.bfloat16, device="cuda"
    ).reshape(37, 2048)

    expected = torch.nn.functional.linear(input, logical_weight)
    actual = torch_ggml_ops.mmq(input, packed, int(qtype), 37)
    error = actual.float() - expected.float()
    normalized_rmse = (
        error.square().mean().sqrt() / expected.float().square().mean().sqrt()
    )

    assert actual.shape == (3, 43, 37)
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    assert normalized_rmse.item() < 0.04


@pytest.mark.parametrize("qname", tuple(_TENSORS))
def test_native_backward_decodes_every_logical_weight_value(
    reader: gguf.GGUFReader, qname: str
) -> None:
    packed, qtype = _packed_rows(reader, qname, out_features=37)
    grad_output = torch.zeros(1, 37, device="cuda", dtype=torch.bfloat16)
    grad_output[0, 17] = 1

    actual = torch.ops.torch_ggml_ops.mmq_grad_input.default(
        grad_output, packed, int(qtype), 2048
    )
    logical_weight = dequantize_gguf_tensor(
        packed, qtype, dtype=torch.bfloat16, device="cuda"
    ).reshape(37, 2048)

    torch.testing.assert_close(actual[0], logical_weight[17], rtol=0, atol=0)


@pytest.mark.parametrize("qname", tuple(_TENSORS))
def test_backward_is_exact_logical_weight_jacobian(
    reader: gguf.GGUFReader, qname: str
) -> None:
    packed, qtype = _packed_rows(reader, qname, out_features=37)
    generator = torch.Generator(device="cuda").manual_seed(5678)
    input = torch.randn(
        2, 5, 2048, generator=generator, device="cuda", dtype=torch.bfloat16
    ).requires_grad_(True)
    grad_output = torch.randn(
        2, 5, 37, generator=generator, device="cuda", dtype=torch.bfloat16
    )

    output = torch_ggml_ops.mmq(input, packed, int(qtype), 37)
    output.backward(grad_output)
    logical_weight = dequantize_gguf_tensor(
        packed, qtype, dtype=torch.bfloat16, device="cuda"
    ).reshape(37, 2048)
    expected_grad = torch.mm(grad_output.reshape(-1, 37), logical_weight).reshape_as(
        input
    )

    torch.testing.assert_close(input.grad, expected_grad, rtol=0, atol=0)
    assert packed.grad is None


def test_zero_input_and_current_stream(reader: gguf.GGUFReader) -> None:
    packed, qtype = _packed_rows(reader, "Q4_K", out_features=37)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        input = torch.zeros(129, 2048, device="cuda", dtype=torch.bfloat16)
        output = torch_ggml_ops.mmq(input, packed, int(qtype), 37)
        grad_output = torch.ones(129, 37, device="cuda", dtype=torch.bfloat16)
        grad_input = torch.ops.torch_ggml_ops.mmq_grad_input.default(
            grad_output, packed, int(qtype), 2048
        )
        forward_checksum = output.float().abs().sum()
        backward_checksum = grad_input.float().abs().sum()
    stream.synchronize()

    assert forward_checksum.item() == 0.0
    assert backward_checksum.item() > 0.0
    assert torch.isfinite(grad_input).all()


def test_opcheck_and_compile(reader: gguf.GGUFReader) -> None:
    packed, qtype = _packed_rows(reader, "Q4_K", out_features=37)
    input = torch.randn(
        2, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    result = torch.library.opcheck(
        torch.ops.torch_ggml_ops.mmq.default,
        (input, packed, int(qtype), 37),
        test_utils=(
            "test_schema",
            "test_autograd_registration",
            "test_faketensor",
            "test_aot_dispatch_dynamic",
        ),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result

    grad_output = torch.randn(2, 37, device="cuda", dtype=torch.bfloat16)
    grad_result = torch.library.opcheck(
        torch.ops.torch_ggml_ops.mmq_grad_input.default,
        (grad_output, packed, int(qtype), 2048),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in grad_result.values()), grad_result

    @torch.compile(fullgraph=True)
    def compiled(input: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        return torch_ggml_ops.mmq(input, packed, int(qtype), 37)

    expected = torch_ggml_ops.mmq(input.detach(), packed, int(qtype), 37)
    actual = compiled(input.detach(), packed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_invalid_inputs_fail_without_hidden_copies(reader: gguf.GGUFReader) -> None:
    packed, qtype = _packed_rows(reader, "Q4_K", out_features=37)
    input = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="contiguous"):
        torch_ggml_ops.mmq(input[:, ::2], packed, int(qtype), 37)
    with pytest.raises(RuntimeError, match="zero storage offset"):
        torch_ggml_ops.mmq(input[1:], packed, int(qtype), 37)
    with pytest.raises(RuntimeError, match="expected"):
        torch_ggml_ops.mmq(input, packed[:-1].clone(), int(qtype), 37)
    with pytest.raises(RuntimeError, match="unsupported quant_type"):
        torch_ggml_ops.mmq(input, packed, 10, 37)
    with pytest.raises(RuntimeError, match="zero-row"):
        torch_ggml_ops.mmq(input[:0], packed, int(qtype), 37)


def test_native_grad_input_rejects_higher_order_gradients(
    reader: gguf.GGUFReader,
) -> None:
    packed, qtype = _packed_rows(reader, "Q4_K", out_features=37)
    grad_output = torch.randn(
        2, 37, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    grad_input = torch.ops.torch_ggml_ops.mmq_grad_input.default(
        grad_output, packed, int(qtype), 2048
    )

    with pytest.raises(RuntimeError, match="does not support higher-order"):
        grad_input.sum().backward()


def test_invalid_grad_input_operands_fail_without_hidden_copies(
    reader: gguf.GGUFReader,
) -> None:
    packed, qtype = _packed_rows(reader, "Q4_K", out_features=37)
    grad_output = torch.randn(4, 37, device="cuda", dtype=torch.bfloat16)
    op = torch.ops.torch_ggml_ops.mmq_grad_input.default

    with pytest.raises(RuntimeError, match="contiguous"):
        op(grad_output[:, ::2], packed, int(qtype), 2048)
    with pytest.raises(RuntimeError, match="zero storage offset"):
        op(grad_output[1:], packed, int(qtype), 2048)
    with pytest.raises(RuntimeError, match="expected"):
        op(grad_output, packed[:-1].clone(), int(qtype), 2048)
    with pytest.raises(RuntimeError, match="unsupported quant_type"):
        op(grad_output, packed, 10, 2048)
    with pytest.raises(RuntimeError, match="zero-row"):
        op(grad_output[:0], packed, int(qtype), 2048)
