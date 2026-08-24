import gguf
import pytest
import torch

import torch_ggml_ops
from tests.grouped_mmq_test_support import load_expert_weight
from tests.mmq_test_support import random_bf16
from tests.model_test_cases import QWEN_MODEL
from tests.model_test_support import model_reader

_ROWS = 16_384
_EXPERTS = 256


def _route() -> tuple[torch.Tensor, torch.Tensor]:
    experts = torch.tensor([0, 2, 5, 7], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([2048, 6144, 10240, _ROWS], device="cuda", dtype=torch.int32)
    return experts, offsets


@pytest.fixture(scope="module")
def q4_down() -> tuple[torch.Tensor, gguf.GGMLQuantizationType, int]:
    packed, quant_type, in_features = load_expert_weight(
        model_reader(QWEN_MODEL),
        "blk.2.ffn_down_exps.weight",
        num_experts=_EXPERTS,
        out_features=2048,
    )
    return packed, quant_type, in_features


@pytest.fixture(scope="module")
def iq2_s_pair() -> tuple[torch.Tensor, torch.Tensor, gguf.GGMLQuantizationType, int]:
    reader = model_reader(QWEN_MODEL)
    gate, quant_type, in_features = load_expert_weight(
        reader,
        "blk.10.ffn_gate_exps.weight",
        num_experts=_EXPERTS,
        out_features=512,
    )
    up, _, _ = load_expert_weight(
        reader,
        "blk.10.ffn_up_exps.weight",
        num_experts=_EXPERTS,
        out_features=512,
    )
    return gate, up, quant_type, in_features


def test_exact_paired_route_compiles(
    iq2_s_pair: tuple[torch.Tensor, torch.Tensor, gguf.GGMLQuantizationType, int],
) -> None:
    gate, up, quant_type, in_features = iq2_s_pair
    experts, offsets = _route()
    input = random_bf16(_ROWS, in_features, seed=20006)

    quant_id = int(quant_type)

    @torch.compile(fullgraph=True)
    def compiled(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch_ggml_ops.grouped_mmq_pair(
            input, gate, up, experts, offsets, quant_id, 512
        )

    expected = torch_ggml_ops.grouped_mmq_pair(
        input, gate, up, experts, offsets, quant_id, 512
    )
    actual = compiled(input)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_grouped_launch_rejects_non_deployed_physical_expert_count(
    q4_down: tuple[torch.Tensor, gguf.GGMLQuantizationType, int],
) -> None:
    packed, quant_type, in_features = q4_down
    experts, offsets = _route()
    input = random_bf16(_ROWS, in_features, seed=20007)
    with pytest.raises(RuntimeError, match="256 physical experts"):
        torch_ggml_ops.grouped_mmq(
            input, packed[:8].clone(), experts, offsets, int(quant_type), 2048
        )


def test_paired_launch_validates_explicit_row_task_shapes(
    iq2_s_pair: tuple[torch.Tensor, torch.Tensor, gguf.GGMLQuantizationType, int],
) -> None:
    gate, up, quant_type, in_features = iq2_s_pair
    experts, offsets = _route()
    input = random_bf16(_ROWS, in_features, seed=20008)
    first_output = torch.empty((_ROWS, 512), dtype=torch.bfloat16, device="cuda")
    second_output = torch.empty_like(first_output)
    workspace = torch.empty(
        input.numel() // 128 * 144, dtype=torch.uint8, device="cuda"
    )
    task_capacity = (_ROWS + 63) // 64 + experts.numel()
    task_count = torch.empty(1, dtype=torch.int32, device="cuda")
    task_experts = torch.empty(
        (2, task_capacity // 2), dtype=torch.int32, device="cuda"
    )
    task_row_starts = torch.empty(task_capacity, dtype=torch.int32, device="cuda")
    task_row_ends = torch.empty_like(task_row_starts)

    with pytest.raises(RuntimeError, match="exact one-dimensional shape"):
        torch.ops.torch_ggml_ops._grouped_mmq_pair_launch.default(
            input,
            gate,
            up,
            experts,
            offsets,
            int(quant_type),
            512,
            first_output,
            second_output,
            workspace,
            task_count,
            task_experts,
            task_row_starts,
            task_row_ends,
        )


def test_obsolete_grouped_dispatcher_ops_are_absent() -> None:
    for name in (
        "grouped_mmq",
        "grouped_mmq_grad_input",
        "grouped_mmq_pair",
        "grouped_mmq_pair_grad_input",
    ):
        assert not hasattr(torch.ops.torch_ggml_ops, name)
