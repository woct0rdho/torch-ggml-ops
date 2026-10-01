"""The benchmark's routed prior default is the learned law of the matrix family.

The benchmark infers the expert prior from the matrix key and the quantization
type because the public routed inputs cannot tell learned routing from hash
routing. These tests pin that rule and the learned-layer sampling it assumes for
DeepSeek. They deliberately do not import `bench`.
"""

import re

import pytest

from tests.model_test_cases import DEEPSEEK_MODEL
from tests.model_test_support import model_reader
from tools.aiter_gmm_compat import infer_expert_prior
from tools.mmq_deployment_cases import public_deployment_cases

ROUTED_CASES = tuple(
    case for case in public_deployment_cases() if case.operation.startswith("Grouped")
)
EXPECTED_PRIORS = {"qwen": "qwen-learned", "deepseek": "deepseek-learned"}
BLOCK = re.compile(r"blk\.(\d+)\.")


def _gmm_key(case) -> tuple[int, int, int, bool]:
    """Mirror the baseline's GMM orientation: forward transposes the weight."""

    forward = "Forward" in case.operation
    return (
        case.rows,
        case.in_features if forward else case.out_features,
        case.out_features if forward else case.in_features,
        forward,
    )


def _case_id(case) -> str:
    return f"{case.operation}-{case.quant_type}-M{case.rows}"


def _inferred(case):
    m, k, n, transposed = _gmm_key(case)
    return infer_expert_prior(
        m, k, n, quant_type=case.quant_type, transposed_rhs=transposed
    )


@pytest.mark.parametrize("case", ROUTED_CASES, ids=_case_id)
def test_inferred_prior_is_the_family_learned_law(case):
    assert _inferred(case) == EXPECTED_PRIORS[case.tensor_source.model_family]


@pytest.mark.parametrize("case", ROUTED_CASES, ids=_case_id)
def test_inference_never_returns_a_hash_law(case):
    assert not _inferred(case).endswith("-hash")


def test_unknown_quant_type_is_rejected():
    with pytest.raises(ValueError, match="expert-prior family"):
        infer_expert_prior(12288, 2048, 4096, quant_type="Q8_0", transposed_rhs=True)


def test_quant_type_mismatch_with_the_tuned_inventory_is_rejected():
    with pytest.raises(ValueError, match="No tuned deepseek-learned"):
        infer_expert_prior(16384, 512, 2048, quant_type="Q2_K", transposed_rhs=True)


def test_missing_quant_type_resolves_an_unambiguous_learned_family():
    assert (
        infer_expert_prior(12288, 2048, 4096, quant_type=None, transposed_rhs=True)
        == "deepseek-learned"
    )
    assert (
        infer_expert_prior(16384, 512, 2048, quant_type=None, transposed_rhs=True)
        == "qwen-learned"
    )


def test_missing_quant_type_rejects_an_unmeasured_key():
    with pytest.raises(ValueError, match="quant_type is required"):
        infer_expert_prior(1234, 512, 2048, quant_type=None, transposed_rhs=True)


def test_deepseek_routed_cases_sample_learned_layers():
    reader = model_reader(DEEPSEEK_MODEL)
    field = reader.fields.get("deepseek4.hash_layer_count")
    if field is None:
        pytest.skip("the model does not declare hash layers")
    hash_layers = int(field.contents())
    deepseek_cases = [
        case for case in ROUTED_CASES if case.tensor_source.model_family == "deepseek"
    ]
    assert deepseek_cases, "no DeepSeek routed cases are registered"
    for case in deepseek_cases:
        for name in case.tensor_source.names:
            match = BLOCK.match(name)
            assert match is not None, name
            assert int(match.group(1)) >= hash_layers, (
                f"{case.operation} {case.quant_type} samples hash layer {name}"
            )
