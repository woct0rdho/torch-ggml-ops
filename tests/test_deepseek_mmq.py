import pytest
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops
from tests.deepseek_dense_cases import (
    DEEPSEEK_DENSE_TENSOR_CASES,
    DenseTensorCase,
)
from tests.mmq_test_support import (
    assert_normalized_rmse,
    find_tensor,
    load_packed_tensor,
    random_bf16,
)
from tests.model_test_cases import DEEPSEEK_MODEL
from tests.model_test_support import model_reader

_EXACT_ROWS = 2048


@pytest.mark.parametrize(
    "case",
    DEEPSEEK_DENSE_TENSOR_CASES,
    ids=tuple(case.name for case in DEEPSEEK_DENSE_TENSOR_CASES),
)
def test_deepseek_dense_q8_0_inventory(case: DenseTensorCase) -> None:
    reader = model_reader(DEEPSEEK_MODEL)
    matches = [
        tensor
        for tensor in reader.tensors
        if (
            tensor.name == case.tensor_suffix
            or tensor.name.endswith(case.tensor_suffix)
        )
        and tensor.tensor_type.name == case.quant_type
    ]

    assert len(matches) == case.tensor_count
    for tensor in matches:
        assert tuple(reversed(tuple(int(value) for value in tensor.shape))) == (
            case.out_features,
            case.in_features,
        )
        assert tuple(tensor.data.shape) == (
            case.out_features,
            case.in_features // 32 * 34,
        )
    assert find_tensor(reader, case.tensor_name) in matches


def test_deepseek_exact_q8_0_forward_and_backward() -> None:
    case = DEEPSEEK_DENSE_TENSOR_CASES[0]
    tensor = find_tensor(model_reader(DEEPSEEK_MODEL), case.tensor_name)
    packed = load_packed_tensor(tensor, case.out_features)
    logical_weight = dequantize_gguf_tensor(
        packed,
        tensor.tensor_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(case.out_features, case.in_features)
    input = random_bf16(_EXACT_ROWS, case.in_features, seed=12567, requires_grad=True)
    output = torch_ggml_ops.mmq(
        input, packed, int(tensor.tensor_type), case.out_features
    )
    expected = (input[:4].float() @ logical_weight.float().T).to(torch.bfloat16)
    torch.testing.assert_close(output[:4], expected, rtol=0.04, atol=0.04)

    grad_output = random_bf16(_EXACT_ROWS, case.out_features, seed=12568)
    output.backward(grad_output)
    assert input.grad is not None
    expected_grad = (grad_output[:4].float() @ logical_weight.float()).to(
        torch.bfloat16
    )
    assert_normalized_rmse(input.grad[:4], expected_grad, maximum=3e-4)
