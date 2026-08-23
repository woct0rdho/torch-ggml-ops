import pytest

from tests.deepseek_dense_cases import (
    DEEPSEEK_DENSE_TENSOR_CASES,
    DenseTensorCase,
)
from tests.mmq_test_support import find_tensor
from tests.model_test_cases import DEEPSEEK_MODEL
from tests.model_test_support import model_reader


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
