"""Strict checked-in deployment inventory for exact GGTensile routes."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .fixed_grouped_mmq_bwd_model import FixedBackwardSolutionKey
from .fixed_grouped_mmq_bwd_spec import DerivedFixedBackwardState
from .fixed_grouped_mmq_fwd_model import FixedForwardSolutionKey
from .fixed_grouped_mmq_fwd_spec import DerivedFixedForwardState
from .grouped_mmq_bwd_pair_model import GroupedBackwardPairSolutionKey
from .grouped_mmq_bwd_pair_physical import (
    derive_grouped_backward_pair_physical_plan,
)
from .grouped_mmq_bwd_pair_spec import DerivedGroupedBackwardPairState
from .grouped_mmq_bwd_physical import derive_grouped_backward_physical_plan
from .grouped_mmq_bwd_spec import DerivedGroupedBackwardState
from .grouped_mmq_fwd_model import GroupedForwardSolutionKey
from .grouped_mmq_fwd_pair_model import GroupedForwardPairSolutionKey
from .grouped_mmq_fwd_pair_spec import DerivedGroupedForwardPairState
from .grouped_mmq_fwd_spec import DerivedGroupedForwardState
from .model import SolutionKey
from .schema import SchemaError, integer, strict_mapping


class DeploymentError(SchemaError):
    """A deployment inventory violates its exact runtime contract."""


DeploymentKey = (
    GroupedForwardSolutionKey
    | GroupedForwardPairSolutionKey
    | SolutionKey
    | GroupedBackwardPairSolutionKey
    | FixedForwardSolutionKey
    | FixedBackwardSolutionKey
)


@dataclass(frozen=True)
class _DerivedLaunchMetadata:
    grid: tuple[int, int, int]
    shared_memory_bytes: int
    ownership: str
    row_task_rows: int | None = None
    row_task_capacity: int | None = None


def _derived_launch_metadata(key: DeploymentKey) -> _DerivedLaunchMetadata:
    if isinstance(key, GroupedForwardSolutionKey):
        state = DerivedGroupedForwardState.from_solution_key(key)
        return _DerivedLaunchMetadata(
            state.grid(key.problem.max_route_entries),
            state.physical_plan.resources.lds_bytes,
            "SerialRoutes",
        )
    if isinstance(key, GroupedForwardPairSolutionKey):
        state = DerivedGroupedForwardPairState.from_solution_key(key)
        row_task_rows = state.kernel_spec.row_task_rows
        if row_task_rows is None:
            return _DerivedLaunchMetadata(
                state.grid(key.problem.max_route_entries),
                state.physical_plan.resources.lds_bytes,
                "SerialRoutes",
            )
        capacity = state.row_task_capacity(key.problem.max_route_entries)
        return _DerivedLaunchMetadata(
            state.row_task_grid(key.problem.max_route_entries),
            state.physical_plan.resources.lds_bytes,
            f"DeviceRowTasks{row_task_rows}",
            row_task_rows,
            capacity,
        )
    if isinstance(key, GroupedBackwardPairSolutionKey):
        state = DerivedGroupedBackwardPairState.from_solution_key(key)
        physical = derive_grouped_backward_pair_physical_plan(state)
        compute = key.solution.compute
        ownership = key.solution.route_ownership
        return _DerivedLaunchMetadata(
            (
                key.problem.in_features // compute.macro_tile1,
                key.problem.max_route_entries * ownership.split_factor,
                1,
            ),
            physical.ordinary.resources.lds_num_bytes,
            ownership.value,
        )
    if isinstance(key, FixedForwardSolutionKey):
        state = DerivedFixedForwardState.from_solution_key(key)
        return _DerivedLaunchMetadata(
            state.grid, state.ordinary.resources.lds_bytes, "FixedGroupZ"
        )
    if isinstance(key, FixedBackwardSolutionKey):
        state = DerivedFixedBackwardState.from_solution_key(key)
        return _DerivedLaunchMetadata(
            state.grid, state.physical.resources.lds_num_bytes, "FixedGroupZ"
        )
    if isinstance(key, SolutionKey):
        state = DerivedGroupedBackwardState.from_solution_key(key)
        physical = derive_grouped_backward_physical_plan(state)
        geometry = state.spec.compute.geometry
        return _DerivedLaunchMetadata(
            (
                state.contract.problem_size.n // geometry.macro_tile1,
                state.contract.max_route_entries,
                state.spec.ownership.split_factor,
            ),
            physical.primary.resources.lds_num_bytes,
            state.spec.ownership.solution_value,
        )
    raise TypeError(f"unsupported deployment key {type(key).__name__}")


_OPERATIONS = frozenset(
    {
        "GroupedForward",
        "GroupedForwardPair",
        "GroupedBackward",
        "GroupedBackwardPair",
        "FixedGroupedForward",
        "FixedGroupedBackward",
    }
)
_ROOT_KEYS = frozenset({"InventoryVersion", "Target", "Routes"})
_TARGET_KEYS = frozenset({"ISA", "WavefrontSize", "CodeObjectVersion"})
_ROUTE_KEYS = frozenset({"Operation", "Kernel"})
_CANDIDATE_KEYS = frozenset(
    {
        "Identity",
        "ExactKey",
        "Artifact",
        "Symbol",
        "ABI",
        "WorkGroup",
        "Grid",
        "SharedMemoryBytes",
        "Ownership",
        "RowTaskRows",
        "RowTaskCapacity",
    }
)


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise DeploymentError(f"{name} must be a nonempty string")
    return value


def _positive_triplet(value: object, name: str) -> tuple[int, int, int]:
    if type(value) is not list or len(value) != 3:
        raise DeploymentError(f"{name} must be a three-element list")
    result = tuple(
        integer(item, f"{name}[{index}]", error_type=DeploymentError)
        for index, item in enumerate(value)
    )
    if any(item <= 0 for item in result):
        raise DeploymentError(f"{name} entries must be positive")
    return result[0], result[1], result[2]


def _optional_positive(value: object, name: str) -> int | None:
    if value is None:
        return None
    result = integer(value, name, error_type=DeploymentError)
    if result <= 0:
        raise DeploymentError(f"{name} must be positive")
    return result


def _typed_key(operation: str, value: object) -> DeploymentKey:
    if operation == "GroupedForward":
        return GroupedForwardSolutionKey.from_mapping(value)
    if operation == "GroupedBackward":
        return SolutionKey.from_mapping(value)
    if operation == "GroupedForwardPair":
        return GroupedForwardPairSolutionKey.from_mapping(value)
    if operation == "GroupedBackwardPair":
        return GroupedBackwardPairSolutionKey.from_mapping(value)
    if operation == "FixedGroupedForward":
        return FixedForwardSolutionKey.from_mapping(value)
    if operation == "FixedGroupedBackward":
        return FixedBackwardSolutionKey.from_mapping(value)
    raise DeploymentError(f"unsupported operation {operation!r}")


def _problem_identity(key: DeploymentKey) -> str:
    mapping = key.to_mapping()
    return json.dumps(
        [mapping["ProblemContract"], mapping["Problem"]],
        sort_keys=True,
        separators=(",", ":"),
    )


def _declared_work_group(key: DeploymentKey) -> tuple[int, int, int]:
    spec = key.to_mapping()["KernelSpec"]
    if not isinstance(spec, Mapping):
        raise DeploymentError("ExactKey KernelSpec must be a mapping")
    geometry = spec.get("geometry")
    if geometry is None:
        compute = spec.get("compute")
        if not isinstance(compute, Mapping):
            raise DeploymentError("ExactKey compute must be a mapping")
        geometry = compute.get("geometry")
    if not isinstance(geometry, Mapping):
        raise DeploymentError("ExactKey geometry must be a mapping")
    return _positive_triplet(geometry.get("work_group"), "ExactKey work_group")


def _declared_abi(key: DeploymentKey) -> str:
    contract = key.to_mapping()["ProblemContract"]
    if not isinstance(contract, Mapping):
        raise DeploymentError("ExactKey ProblemContract must be a mapping")
    abi = contract.get("abi") or contract.get("abi_family")
    if type(abi) is not str:
        raise DeploymentError("ExactKey does not declare an ABI")
    return abi


@dataclass(frozen=True)
class DeploymentTarget:
    isa: tuple[int, int, int]
    wavefront_size: int
    code_object_version: int


@dataclass(frozen=True)
class DeploymentCandidate:
    identity: str
    exact_key: DeploymentKey
    artifact: str
    symbol: str
    abi: str
    work_group: tuple[int, int, int]
    grid: tuple[int, int, int]
    shared_memory_bytes: int
    ownership: str
    row_task_rows: int | None
    row_task_capacity: int | None


@dataclass(frozen=True)
class DeploymentRoute:
    operation: str
    kernel: DeploymentCandidate


@dataclass(frozen=True)
class DeploymentInventory:
    target: DeploymentTarget
    routes: tuple[DeploymentRoute, ...]


def load_deployment_inventory(path: Path) -> DeploymentInventory:
    root = strict_mapping(
        json.loads(path.read_text(encoding="utf-8")),
        "deployment inventory",
        _ROOT_KEYS,
        error_type=DeploymentError,
    )
    if (
        integer(
            root["InventoryVersion"], "InventoryVersion", error_type=DeploymentError
        )
        != 1
    ):
        raise DeploymentError("InventoryVersion must be 1")
    target_value = strict_mapping(
        root["Target"], "Target", _TARGET_KEYS, error_type=DeploymentError
    )
    target = DeploymentTarget(
        _positive_triplet(target_value["ISA"], "Target.ISA"),
        integer(
            target_value["WavefrontSize"],
            "Target.WavefrontSize",
            error_type=DeploymentError,
        ),
        integer(
            target_value["CodeObjectVersion"],
            "Target.CodeObjectVersion",
            error_type=DeploymentError,
        ),
    )
    if target != DeploymentTarget((11, 5, 1), 32, 5):
        raise DeploymentError("Target must be gfx1151 wave32 code object v5")
    raw_routes = root["Routes"]
    if type(raw_routes) is not list or not raw_routes:
        raise DeploymentError("Routes must be a nonempty list")
    routes: list[DeploymentRoute] = []
    route_identities: set[tuple[str, str]] = set()
    candidate_identities: set[str] = set()
    for route_index, route_value in enumerate(raw_routes):
        route = strict_mapping(
            route_value,
            f"Routes[{route_index}]",
            _ROUTE_KEYS,
            error_type=DeploymentError,
        )
        operation = _text(route["Operation"], f"Routes[{route_index}].Operation")
        if operation not in _OPERATIONS:
            raise DeploymentError(f"unsupported operation {operation!r}")
        name = f"Routes[{route_index}].Kernel"
        item = strict_mapping(
            route["Kernel"], name, _CANDIDATE_KEYS, error_type=DeploymentError
        )
        key = _typed_key(operation, item["ExactKey"])
        identity = _text(item["Identity"], f"{name}.Identity")
        if identity != key.hash:
            raise DeploymentError(f"{name}.Identity does not match ExactKey")
        symbol = _text(item["Symbol"], f"{name}.Symbol")
        if symbol != key.kernel_name:
            raise DeploymentError(f"{name}.Symbol does not match ExactKey")
        abi = _text(item["ABI"], f"{name}.ABI")
        if abi != _declared_abi(key):
            raise DeploymentError(f"{name}.ABI does not match ExactKey")
        work_group = _positive_triplet(item["WorkGroup"], f"{name}.WorkGroup")
        if work_group != _declared_work_group(key):
            raise DeploymentError(f"{name}.WorkGroup does not match ExactKey")
        row_rows = _optional_positive(item["RowTaskRows"], f"{name}.RowTaskRows")
        row_capacity = _optional_positive(
            item["RowTaskCapacity"], f"{name}.RowTaskCapacity"
        )
        ownership = _text(item["Ownership"], f"{name}.Ownership")
        if (row_rows is None) != (row_capacity is None):
            raise DeploymentError(
                f"{name} must specify both row-task bounds or neither"
            )
        if ownership.startswith("DeviceRowTasks") != (row_rows is not None):
            raise DeploymentError(f"{name} row-task bounds disagree with Ownership")
        grid = _positive_triplet(item["Grid"], f"{name}.Grid")
        shared = integer(
            item["SharedMemoryBytes"],
            f"{name}.SharedMemoryBytes",
            error_type=DeploymentError,
        )
        if shared < 0:
            raise DeploymentError(f"{name}.SharedMemoryBytes must be nonnegative")
        derived = _derived_launch_metadata(key)
        declared_metadata = (
            grid,
            shared,
            ownership,
            row_rows,
            row_capacity,
        )
        derived_metadata = (
            derived.grid,
            derived.shared_memory_bytes,
            derived.ownership,
            derived.row_task_rows,
            derived.row_task_capacity,
        )
        if declared_metadata != derived_metadata:
            raise DeploymentError(
                f"{name} launch metadata does not match its typed ExactKey: "
                f"declared={declared_metadata!r}, derived={derived_metadata!r}"
            )
        kernel = DeploymentCandidate(
            identity,
            key,
            _text(item["Artifact"], f"{name}.Artifact"),
            symbol,
            abi,
            work_group,
            grid,
            shared,
            ownership,
            row_rows,
            row_capacity,
        )
        problem_identity = _problem_identity(key)
        route_identity = (operation, problem_identity)
        if route_identity in route_identities:
            raise DeploymentError(f"duplicate exact deployment route {operation}")
        if identity in candidate_identities:
            raise DeploymentError(f"duplicate kernel identity {identity}")
        route_identities.add(route_identity)
        candidate_identities.add(identity)
        routes.append(DeploymentRoute(operation, kernel))
    return DeploymentInventory(target, tuple(routes))
