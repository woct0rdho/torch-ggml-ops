"""Launch facts derived from exact problems and canonical physical plans."""

from dataclasses import dataclass

from .family_registry import KernelAbiName, abi_for_instance
from .fixed_grouped_mmq_bwd_model import FixedBackwardProblem
from .fixed_grouped_mmq_bwd_spec import (
    DerivedFixedBackwardState,
    FixedBackwardKernelSpec,
)
from .fixed_grouped_mmq_fwd_model import FixedForwardProblem
from .fixed_grouped_mmq_fwd_spec import (
    DerivedFixedForwardState,
    FixedForwardKernelSpec,
)
from .grouped_mmq_bwd_pair_model import GroupedBackwardPairProblem
from .grouped_mmq_bwd_pair_physical import derive_grouped_backward_pair_physical_plan
from .grouped_mmq_bwd_pair_spec import (
    DerivedGroupedBackwardPairState,
    GroupedBackwardPairKernelSpec,
)
from .grouped_mmq_bwd_physical import derive_grouped_backward_physical_plan
from .grouped_mmq_bwd_spec import (
    DerivedGroupedBackwardState,
    GroupedBackwardKernelSpec,
)
from .grouped_mmq_fwd_model import GroupedForwardProblem
from .grouped_mmq_fwd_pair_model import GroupedForwardPairProblem
from .grouped_mmq_fwd_pair_spec import (
    DerivedGroupedForwardPairState,
    GroupedForwardPairKernelSpec,
)
from .grouped_mmq_fwd_spec import DerivedGroupedForwardState, GroupedForwardKernelSpec
from .identity import KernelFamily
from .kernel_instance import KernelInstance
from .mmq_bwd_physical import derive_backward_physical_plan
from .mmq_bwd_spec import BackwardKernelSpec, DerivedBackwardState
from .mmq_fwd_spec import DerivedForwardState, ForwardKernelSpec
from .model import ProblemSize
from .work_group_mapping import mapped_grid_extent


@dataclass(frozen=True)
class LaunchMetadata:
    abi: KernelAbiName
    work_group: tuple[int, int, int]
    grid: tuple[int, int, int]
    static_group_segment_bytes: int
    ownership: str
    row_task_rows: int | None = None
    row_task_capacity: int | None = None
    route_split_factor: int = 1


def _derive_launch_metadata(instance: KernelInstance) -> LaunchMetadata:
    abi, _ = abi_for_instance(instance)
    family = instance.family
    problem = instance.problem
    spec = instance.kernel_spec
    quant_type = instance.problem_type.quant_data_type
    if family is KernelFamily.GroupedForward:
        assert isinstance(problem, GroupedForwardProblem) and isinstance(
            spec, GroupedForwardKernelSpec
        )
        state = DerivedGroupedForwardState.from_problem_spec(problem, spec)
        return LaunchMetadata(
            abi,
            state.kernel_spec.geometry.work_group,
            state.grid(problem.max_route_entries),
            state.physical_plan.resources.lds_bytes,
            "SerialRoutes",
        )
    if family is KernelFamily.GroupedForwardPair:
        assert isinstance(problem, GroupedForwardPairProblem) and isinstance(
            spec, GroupedForwardPairKernelSpec
        )
        state = DerivedGroupedForwardPairState.from_problem_spec(problem, spec)
        rows = state.kernel_spec.row_task_rows
        if rows is None:
            return LaunchMetadata(
                abi,
                state.kernel_spec.geometry.work_group,
                state.grid(problem.max_route_entries),
                state.physical_plan.resources.lds_bytes,
                "SerialRoutes",
            )
        capacity = state.row_task_capacity(problem.max_route_entries)
        return LaunchMetadata(
            abi,
            state.kernel_spec.geometry.work_group,
            state.row_task_grid(problem.max_route_entries),
            state.physical_plan.resources.lds_bytes,
            "DeviceRowTasks",
            rows,
            capacity,
        )
    if family is KernelFamily.GroupedBackwardPair:
        assert isinstance(problem, GroupedBackwardPairProblem) and isinstance(
            spec, GroupedBackwardPairKernelSpec
        )
        state = DerivedGroupedBackwardPairState.from_problem_spec(problem, spec)
        physical = derive_grouped_backward_pair_physical_plan(state)
        split = state.kernel_spec.route_ownership.split_factor
        geometry = state.kernel_spec.compute.geometry
        return LaunchMetadata(
            abi,
            geometry.work_group,
            (
                mapped_grid_extent(
                    problem.in_features // geometry.macro_tile1,
                    geometry.work_group_mapping,
                ),
                problem.max_route_entries * split,
                1,
            ),
            physical.ordinary.resources.lds_bytes,
            ("SerialRoutes" if split == 1 else f"PackedSplitRoutes{split}"),
            route_split_factor=split,
        )
    if family is KernelFamily.FixedGroupedForward:
        assert isinstance(problem, FixedForwardProblem) and isinstance(
            spec, FixedForwardKernelSpec
        )
        state = DerivedFixedForwardState.from_problem_spec(problem, spec)
        return LaunchMetadata(
            abi,
            state.ordinary.kernel_spec.geometry.work_group,
            state.grid,
            state.ordinary.resources.lds_bytes,
            "FixedGroupZ",
        )
    if family is KernelFamily.FixedGroupedBackward:
        assert isinstance(problem, FixedBackwardProblem) and isinstance(
            spec, FixedBackwardKernelSpec
        )
        state = DerivedFixedBackwardState.from_problem_spec(problem, spec)
        return LaunchMetadata(
            abi,
            state.spec.compute.geometry.work_group,
            state.grid,
            state.physical.resources.lds_bytes,
            "FixedGroupZ",
        )
    if family is KernelFamily.OrdinaryForward:
        assert isinstance(problem, ProblemSize) and isinstance(spec, ForwardKernelSpec)
        state = DerivedForwardState.from_problem_spec(problem, quant_type, spec)
        return LaunchMetadata(
            abi,
            spec.geometry.work_group,
            state.grid,
            state.resources.lds_bytes,
            "Dense",
        )
    if family is KernelFamily.OrdinaryBackward:
        assert isinstance(problem, ProblemSize) and isinstance(spec, BackwardKernelSpec)
        state = DerivedBackwardState.from_problem_spec(problem, quant_type, spec)
        physical = derive_backward_physical_plan(state)
        geometry = spec.geometry
        return LaunchMetadata(
            abi,
            geometry.work_group,
            (
                mapped_grid_extent(1, geometry.work_group_mapping),
                problem.n // geometry.macro_tile1,
                problem.m // geometry.macro_tile0 // geometry.work_group_mapping,
            ),
            physical.resources.lds_bytes,
            "Dense",
        )
    if family is KernelFamily.GroupedBackward:
        assert isinstance(problem, ProblemSize) and isinstance(
            spec, GroupedBackwardKernelSpec
        )
        state = DerivedGroupedBackwardState.from_problem_spec(problem, quant_type, spec)
        physical = derive_grouped_backward_physical_plan(state)
        geometry = state.spec.compute.geometry
        split = state.spec.ownership.split_factor
        return LaunchMetadata(
            abi,
            geometry.work_group,
            (
                mapped_grid_extent(
                    state.contract.problem_size.n // geometry.macro_tile1,
                    geometry.work_group_mapping,
                ),
                state.contract.max_route_entries,
                split,
            ),
            physical.primary.resources.lds_bytes,
            (
                "SerialRoutes"
                if state.spec.ownership.kind == "Serial"
                else f"SplitRoutes{state.spec.ownership.split_factor}"
            ),
            route_split_factor=split,
        )
    raise TypeError(f"unsupported kernel family {family!r}")
