import pytest
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops
from tests.mmq_test_support import (
    assert_normalized_rmse,
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
def fixed_q8() -> tuple[torch.Tensor, torch.Tensor]:
    tensor = find_tensor(model_reader(DEEPSEEK_MODEL), "blk.0.attn_output_a.weight")
    packed = load_packed_fixed_groups(
        tensor,
        groups=_GROUPS,
        group_out_features=_OUT_FEATURES,
        out_features=_OUT_FEATURES,
    )
    logical = dequantize_gguf_tensor(
        packed[0], tensor.tensor_type, dtype=torch.bfloat16, device="cuda"
    ).reshape(_OUT_FEATURES, _IN_FEATURES)
    return packed, logical


def test_exact_fixed_q8_forward_and_autograd(
    fixed_q8: tuple[torch.Tensor, torch.Tensor],
) -> None:
    packed, logical = fixed_q8
    input = random_bf16(_TOKENS, _GROUPS, _IN_FEATURES, seed=30001, requires_grad=True)
    output = torch_ggml_ops.fixed_grouped_mmq(input, packed)
    expected = (input[:2, 0].float() @ logical.float().T).to(torch.bfloat16)
    assert output.shape == (_TOKENS, _GROUPS, _OUT_FEATURES)
    assert_normalized_rmse(output[:2, 0], expected)

    grad_output = random_bf16(_TOKENS, _GROUPS, _OUT_FEATURES, seed=30002)
    output.backward(grad_output)
    assert input.grad is not None
    expected_grad = (grad_output[:2, 0].float() @ logical.float()).to(torch.bfloat16)
    assert_normalized_rmse(input.grad[:2, 0], expected_grad, maximum=1e-4)


def test_exact_fixed_q8_compiles(
    fixed_q8: tuple[torch.Tensor, torch.Tensor],
) -> None:
    packed, _logical = fixed_q8
    input = random_bf16(_TOKENS, _GROUPS, _IN_FEATURES, seed=30003)

    @torch.compile(fullgraph=True)
    def compiled(input: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        return torch_ggml_ops.fixed_grouped_mmq(input, packed)

    expected = torch_ggml_ops.fixed_grouped_mmq(input, packed)
    actual = compiled(input, packed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_unsupported_fixed_token_count_fails_at_launch(
    fixed_q8: tuple[torch.Tensor, torch.Tensor],
) -> None:
    packed, _logical = fixed_q8
    input = random_bf16(2, _GROUPS, _IN_FEATURES, seed=30004)
    with pytest.raises(RuntimeError, match="unsupported exact deployment key"):
        torch_ggml_ops.fixed_grouped_mmq(input, packed)


def test_obsolete_fixed_dispatcher_ops_are_absent() -> None:
    for name in ("fixed_grouped_mmq", "fixed_grouped_mmq_grad_input"):
        assert not hasattr(torch.ops.torch_ggml_ops, name)
