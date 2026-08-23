"""External-oracle numerical coverage for every selected public route."""

import gc
from collections.abc import Iterator

import pytest
import torch

from tools.mmq_correctness import (
    PreparedCase,
    assert_changed,
    assert_external_reference,
    assert_repeat,
    external_reference,
    prepare_case,
)
from tools.mmq_deployment_cases import (
    DeploymentCase,
    hip_control_root,
    operation_counts,
    public_artifact_path,
    public_deployment_cases,
)
from tools.mmq_deployment_runner import run_implementation

CASES = public_deployment_cases()
CASE_IDS = tuple(
    f"{case.operation}-{case.identity}-{case.quant_type}-M{case.rows}" for case in CASES
)


@pytest.fixture(autouse=True)
def release_deployment_cuda_memory() -> Iterator[None]:
    yield
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def _prepare(case: DeploymentCase, *, require_artifact: bool = False) -> PreparedCase:
    if require_artifact:
        artifact = public_artifact_path(case)
        if not artifact.is_file():
            pytest.skip(f"selected GGTensile artifact is unavailable: {artifact}")
    return prepare_case(case, device=torch.device("cuda"))


def _reference(prepared: PreparedCase):
    return external_reference(prepared)


def _assert_reference(actual, expected, label: str) -> None:
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for index, (item, reference) in enumerate(zip(actual, expected, strict=True)):
            assert_external_reference(item, reference, f"{label}[{index}]")
    else:
        assert not isinstance(actual, tuple)
        assert_external_reference(actual, expected, label)


def _run(case: DeploymentCase, implementation: str, prepared: PreparedCase):
    return run_implementation(
        case,
        prepared,
        implementation,
        hip_root=hip_control_root(),
    )


def _run_hip(case: DeploymentCase, prepared: PreparedCase):
    root = hip_control_root()
    if root is None:
        pytest.skip("historical HIP-control artifacts are unavailable")
    return run_implementation(case, prepared, "hip", hip_root=root)


def test_public_inventory_has_the_complete_numerical_matrix() -> None:
    assert len(CASES) == 148
    assert operation_counts() == {
        "FixedGroupedBackward": 3,
        "FixedGroupedForward": 3,
        "GroupedBackward": 12,
        "GroupedBackwardPair": 9,
        "GroupedForward": 12,
        "GroupedForwardPair": 9,
        "OrdinaryBackward": 50,
        "OrdinaryForward": 50,
    }


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_public_api_route_matches_external_oracle(case: DeploymentCase) -> None:
    prepared = _prepare(case)
    expected = _reference(prepared)
    actual = _run(case, "public", prepared)
    _assert_reference(actual, expected, f"public/{case.identity}")


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_direct_ggtensile_route_matches_external_oracle(case: DeploymentCase) -> None:
    prepared = _prepare(case, require_artifact=True)
    expected = _reference(prepared)
    actual = _run(case, "ggtensile", prepared)
    _assert_reference(actual, expected, f"ggtensile/{case.identity}")


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_historical_hip_route_matches_external_oracle(case: DeploymentCase) -> None:
    prepared = _prepare(case)
    expected = _reference(prepared)
    actual = _run_hip(case, prepared)
    _assert_reference(actual, expected, f"hip/{case.identity}")


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_route_repeatability_and_dependency(case: DeploymentCase) -> None:
    prepared = _prepare(case)
    expected = _reference(prepared)
    actual = _run(case, "public", prepared)
    repeated = _run(case, "public", prepared)
    if isinstance(actual, tuple):
        for first, second in zip(actual, repeated, strict=True):
            assert_repeat(first, second, f"public/{case.identity}")
    else:
        assert_repeat(actual, repeated, f"public/{case.identity}")
    if case.operation.endswith("Forward") or case.operation == "GroupedForwardPair":
        assert prepared.input is not None
        saved_input = prepared.input
        prepared.input = saved_input.clone()
        prepared.input[0].neg_()
        mutated = _run(case, "public", prepared)
        if isinstance(actual, tuple):
            for index, (before, after) in enumerate(zip(actual, mutated, strict=True)):
                assert_changed(before, after, f"public/{case.identity}[{index}]")
        else:
            assert_changed(actual, mutated, f"public/{case.identity}")
        prepared.input = saved_input
    else:
        saved_grad = prepared.grad_outputs
        mutated_grad = list(saved_grad)
        mutated_grad[0] = mutated_grad[0].clone()
        mutated_grad[0][0].neg_()
        prepared.grad_outputs = tuple(mutated_grad)
        mutated = _run(case, "public", prepared)
        assert_changed(actual, mutated, f"public/{case.identity}")
        prepared.grad_outputs = saved_grad
    _assert_reference(actual, expected, f"public/{case.identity}/repeat")
