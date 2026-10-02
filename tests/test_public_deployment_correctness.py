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
    operation_counts,
    public_artifact_path,
    public_deployment_cases,
)
from tools.mmq_deployment_runner import run_implementation
from tools.mmq_hip_paths import control_root

CASES = public_deployment_cases()
CASE_IDS = tuple(
    f"{case.operation}-{case.identity}-{case.quant_type}-M{case.rows}" for case in CASES
)
ROUTES = ("public", "ggtensile", "hip", "repeat")
_REFERENCE = torch.Tensor | tuple[torch.Tensor, ...]
_CASE_STATE: dict[str, tuple[PreparedCase, _REFERENCE]] = {}


@pytest.fixture(scope="module", autouse=True)
def release_deployment_cuda_memory() -> Iterator[None]:
    yield
    _CASE_STATE.clear()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _case_state(case: DeploymentCase) -> tuple[PreparedCase, _REFERENCE]:
    """Prepare one case and its external reference, shared by every route.

    The route items are collected case-major, so only the case in flight is
    kept. The previous case is released when the next one is prepared.
    """

    state = _CASE_STATE.get(case.identity)
    if state is None:
        _CASE_STATE.clear()
        torch.cuda.empty_cache()
        prepared = prepare_case(case, device=torch.device("cuda"))
        state = (prepared, external_reference(prepared))
        _CASE_STATE[case.identity] = state
    return state


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
        hip_root=control_root(),
    )


def _assert_repeatability(
    case: DeploymentCase,
    prepared: PreparedCase,
    expected: _REFERENCE,
) -> None:
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


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=case_id, marks=pytest.mark.xdist_group(case.identity))
        for case, case_id in zip(CASES, CASE_IDS, strict=True)
    ],
)
def test_deployment_route_matches_external_oracle(
    case: DeploymentCase, route: str
) -> None:
    if route == "ggtensile":
        artifact = public_artifact_path(case)
        if not artifact.is_file():
            pytest.skip(f"selected GGTensile artifact is unavailable: {artifact}")
    if route == "hip" and control_root() is None:
        pytest.skip("HIP control artifacts are unavailable")

    prepared, expected = _case_state(case)
    if route == "repeat":
        _assert_repeatability(case, prepared, expected)
        return
    actual = _run(case, route, prepared)
    _assert_reference(actual, expected, f"{route}/{case.identity}")
