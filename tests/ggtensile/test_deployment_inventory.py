import json
from pathlib import Path
from typing import Any, cast

import pytest

from tools.ggtensile.deployment import DeploymentError, load_deployment_inventory
from tools.ggtensile.grouped_mmq_fwd_model import (
    GroupedForwardProblem,
    GroupedForwardSolution,
    GroupedForwardSolutionKey,
)
from tools.ggtensile.grouped_mmq_fwd_spec import DerivedGroupedForwardState


def _candidate(key: GroupedForwardSolutionKey) -> dict[str, object]:
    mapping = key.to_mapping()
    spec = cast(dict[str, object], mapping["KernelSpec"])
    geometry = cast(dict[str, object], spec["geometry"])
    work_group = geometry["work_group"]
    state = DerivedGroupedForwardState.from_solution_key(key)
    return {
        "Identity": key.hash,
        "ExactKey": key.to_mapping(),
        "Artifact": f"artifacts/{key.hash}.co",
        "Symbol": key.kernel_name,
        "ABI": "GroupedSerialRoutesV1",
        "WorkGroup": work_group,
        "Grid": list(state.grid(key.problem.max_route_entries)),
        "SharedMemoryBytes": state.physical_plan.resources.lds_bytes,
        "Ownership": "SerialRoutes",
        "RowTaskRows": None,
        "RowTaskCapacity": None,
    }


def _manifest() -> dict[str, Any]:
    problem = GroupedForwardProblem.q4_k(16_384)
    selected = GroupedForwardSolutionKey(
        problem, GroupedForwardSolution.q4_k_serial_decoded_lds()
    )
    return {
        "InventoryVersion": 1,
        "Target": {
            "ISA": [11, 5, 1],
            "WavefrontSize": 32,
            "CodeObjectVersion": 5,
        },
        "Routes": [
            {
                "Operation": "GroupedForward",
                "Kernel": _candidate(selected),
            }
        ],
    }


def _write(tmp_path: Path, manifest: object) -> Path:
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_selected_kernel_is_typed(tmp_path: Path) -> None:
    inventory = load_deployment_inventory(_write(tmp_path, _manifest()))
    assert isinstance(inventory.routes[0].kernel.exact_key, GroupedForwardSolutionKey)


def test_rejects_identity_not_derived_from_exact_key(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["Routes"][0]["Kernel"]["Identity"] = "ggsol_wrong"
    with pytest.raises(DeploymentError, match="Identity does not match"):
        load_deployment_inventory(_write(tmp_path, manifest))


def test_rejects_alternative_kernel_field(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["Routes"][0]["Candidates"] = [manifest["Routes"][0].pop("Kernel")]
    with pytest.raises(DeploymentError, match="unknown.*Candidates"):
        load_deployment_inventory(_write(tmp_path, manifest))


@pytest.mark.parametrize(
    ("field", "value"),
    (("Grid", [1, 1, 1]), ("SharedMemoryBytes", 0), ("Ownership", "SerialWrong")),
)
def test_rejects_launch_metadata_not_derived_from_key(
    tmp_path: Path, field: str, value: object
) -> None:
    manifest = _manifest()
    manifest["Routes"][0]["Kernel"][field] = value
    with pytest.raises(DeploymentError, match="launch metadata does not match"):
        load_deployment_inventory(_write(tmp_path, manifest))


def test_paired_backward_split_routes_pack_split_ownership_in_grid_y() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools/ggtensile/configs/mmq_deployment.json"
    )
    inventory = load_deployment_inventory(path)
    split_routes = [
        route.kernel
        for route in inventory.routes
        if route.operation == "GroupedBackwardPair"
        and route.kernel.ownership == "PackedSplitRoutes8"
    ]
    assert len(split_routes) == 3
    assert all(kernel.grid == (32, 2048, 1) for kernel in split_routes)


def test_rejects_row_task_bounds_for_serial_ownership(tmp_path: Path) -> None:
    manifest = _manifest()
    candidate = manifest["Routes"][0]["Kernel"]
    candidate["RowTaskRows"] = 32
    candidate["RowTaskCapacity"] = 768
    with pytest.raises(DeploymentError, match="disagree"):
        load_deployment_inventory(_write(tmp_path, manifest))
