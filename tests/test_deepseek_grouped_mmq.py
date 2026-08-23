import pytest
import torch

import torch_ggml_ops
from tests.mmq_test_support import (
    find_tensor,
    load_packed_fixed_groups,
    random_bf16,
)
from tests.model_test_cases import DEEPSEEK_MODEL
from tests.model_test_support import model_reader

_TOKENS = 2048
_GROUPS = 8
_OUT_FEATURES = 1024
_IN_FEATURES = 4096


@pytest.fixture(scope="module")
def fixed_q8() -> torch.Tensor:
    tensor = find_tensor(model_reader(DEEPSEEK_MODEL), "blk.0.attn_output_a.weight")
    packed = load_packed_fixed_groups(
        tensor,
        groups=_GROUPS,
        group_out_features=_OUT_FEATURES,
        out_features=_OUT_FEATURES,
    )
    return packed


def test_exact_fixed_q8_compiles(
    fixed_q8: torch.Tensor,
) -> None:
    packed = fixed_q8
    input = random_bf16(_TOKENS, _GROUPS, _IN_FEATURES, seed=30003)

    @torch.compile(fullgraph=True)
    def compiled(input: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        return torch_ggml_ops.fixed_grouped_mmq(input, packed)

    expected = torch_ggml_ops.fixed_grouped_mmq(input, packed)
    actual = compiled(input, packed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_unsupported_fixed_token_count_fails_at_launch(
    fixed_q8: torch.Tensor,
) -> None:
    packed = fixed_q8
    input = random_bf16(2, _GROUPS, _IN_FEATURES, seed=30004)
    with pytest.raises(RuntimeError, match="unsupported exact deployment key"):
        torch_ggml_ops.fixed_grouped_mmq(input, packed)


def test_obsolete_fixed_dispatcher_ops_are_absent() -> None:
    for name in ("fixed_grouped_mmq", "fixed_grouped_mmq_grad_input"):
        assert not hasattr(torch.ops.torch_ggml_ops, name)
