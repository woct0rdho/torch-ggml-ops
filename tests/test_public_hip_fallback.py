"""The public route serves HIP-only problems with the selected HIP control.

The public bundle resolves one kernel per exact problem: GGTensile whenever a
catalog carries the key, the deployed HIP control otherwise. These tests pin
that resolution and check the HIP-selected routes against the external oracle.
"""

import hashlib

import pytest
import torch

from tools.mmq_correctness import (
    assert_external_reference,
    external_reference,
    prepare_case,
)
from tools.mmq_deployment_cases import DeploymentCase, TensorSource
from tools.mmq_deployment_runner import run_implementation
from tools.mmq_deployment_spec import (
    IMPLEMENTATION_GGTENSILE,
    IMPLEMENTATION_HIP,
    deployments,
    hip_only_keys,
    kernels,
)
from tools.mmq_hip_deployment import select_hip_control

ITEMS = kernels()
ENTRIES = deployments(ITEMS)
HIP_ENTRIES = tuple(
    entry for entry in ENTRIES if entry.implementation == IMPLEMENTATION_HIP
)
MATRIX_KEYS = sorted({(entry.operation, entry.quant_type) for entry in HIP_ENTRIES})
# Every key of one matrix resolves to the same control family, so one
# representative per matrix covers the arithmetic of the whole family.
REPRESENTATIVES = tuple(
    next(
        entry for entry in HIP_ENTRIES if (entry.operation, entry.quant_type) == matrix
    )
    for matrix in MATRIX_KEYS
)
_SOURCE = TensorSource("synthetic", ("blk.0.weight",))


def _case_id(entry) -> str:
    return f"{entry.operation}-{entry.quant_type}-M{entry.m}"


def _case(entry) -> DeploymentCase:
    identity = hashlib.sha256(
        f"{entry.operation}|{entry.quant_type}|{entry.m}|{entry.n}|{entry.k}".encode()
    ).hexdigest()[:16]
    return DeploymentCase(
        operation=entry.operation,
        identity=identity,
        symbol=ITEMS[entry.kernel_index].symbol,
        quant_type=entry.quant_type,
        rows=entry.m,
        out_features=entry.n,
        in_features=entry.k,
        tensor_source=_SOURCE,
        instance=None,
    )


def test_resolution_prefers_ggtensile_and_falls_back_to_hip() -> None:
    """The resolved table is exactly the GGTensile keys plus the HIP-only keys."""

    ggtensile = [
        entry for entry in ENTRIES if entry.implementation == IMPLEMENTATION_GGTENSILE
    ]
    assert len(ggtensile) == 148
    assert len(HIP_ENTRIES) == len(hip_only_keys()) == 204
    covered = {
        (entry.operation, entry.quant_type, entry.m, entry.n, entry.k)
        for entry in ggtensile
    }
    assert covered.isdisjoint(
        {
            (entry.operation, entry.quant_type, entry.m, entry.n, entry.k)
            for entry in HIP_ENTRIES
        }
    )
    assert all(
        entry.operation in {"OrdinaryForward", "OrdinaryBackward"}
        for entry in HIP_ENTRIES
    )


def test_every_hip_artifact_is_a_selected_control() -> None:
    """Each HIP-selected record names the control the deployment table selects."""

    for entry in HIP_ENTRIES:
        key = (entry.operation, entry.quant_type, entry.m, entry.n, entry.k)
        assert ITEMS[entry.kernel_index].symbol == select_hip_control(*key).symbol
        spec = select_hip_control(*key).spec
        assert getattr(spec.config, "split_k", 0) == 0


@pytest.mark.parametrize("entry", REPRESENTATIVES, ids=_case_id)
def test_hip_selected_public_route_matches_the_external_oracle(entry) -> None:
    case = _case(entry)
    prepared = prepare_case(case, device=torch.device("cuda"), mode="synthetic")
    expected = external_reference(prepared)
    actual = run_implementation(case, prepared, "public", hip_root=None)
    assert_external_reference(actual, expected, f"{case.operation} {case.quant_type}")
