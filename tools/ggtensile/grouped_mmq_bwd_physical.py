"""Deterministic route and reusable compute resources for grouped backward."""

from dataclasses import dataclass, replace

from .grouped_mmq_bwd_spec import DerivedGroupedBackwardState
from .kernel_writer_assembly import (
    DeterministicRegisterPlan,
    RegisterLifetime,
    RegisterRole,
)
from .mmq_bwd_physical import (
    BackwardPhysicalPlan,
    derive_backward_physical_plan,
)


@dataclass(frozen=True)
class GroupedBackwardScalarPlan:
    expert_indices: int
    expert_offsets: int
    num_experts: int
    rows: int
    bytes_per_expert: int
    gemm_index: int
    row_begin: int
    row_end: int
    expert: int
    route_rows: int
    route_output_bytes: int
    saved_exec: int
    pointer_temporary: int
    tile_end: int


@dataclass(frozen=True)
class GroupedBackwardPhysicalPlan:
    primary: BackwardPhysicalPlan
    secondary: BackwardPhysicalPlan | None
    route: GroupedBackwardScalarPlan


def _roles(
    include_codebook: bool,
) -> tuple[dict[str, RegisterRole], tuple[str, ...]]:
    lifetime = RegisterLifetime(0, 3)
    widths = {
        "kernarg": 6,
        "loop_counter": 1,
        "block_offset": 1,
        "input_half": 1,
        "scalar_temporary": 2,
        "expert_indices": 2,
        "expert_offsets": 2,
        "num_experts": 1,
        "rows": 1,
        "bytes_per_expert": 2,
        "gemm_index": 1,
        "row_begin": 1,
        "row_end": 1,
        "expert": 2,
        "route_rows": 1,
        "route_output_bytes": 1,
        "saved_exec": 1,
        "pointer_temporary": 2,
        "tile_end": 1,
    }
    if include_codebook:
        widths["codebook_base"] = 2
    align_two = {
        "kernarg",
        "expert_indices",
        "expert_offsets",
        "bytes_per_expert",
        "expert",
        "pointer_temporary",
    }
    if include_codebook:
        align_two.add("codebook_base")
    order = tuple(widths)
    roles = {
        name: RegisterRole(
            name,
            width,
            lifetime,
            alignment=2 if name in align_two else 1,
            minimum_register=5,
        )
        for name, width in widths.items()
    }
    return roles, order


def derive_grouped_backward_physical_plan(
    state: DerivedGroupedBackwardState,
) -> GroupedBackwardPhysicalPlan:
    primary = derive_backward_physical_plan(state.primary)
    secondary = (
        derive_backward_physical_plan(state.secondary)
        if state.secondary is not None
        else None
    )
    roles, order = _roles(state.contract.quant_type == "IQ2_S")
    scalar = DeterministicRegisterPlan.allocate(
        roles,
        order,
        max_registers=64,
    )

    def first(name: str) -> int:
        return scalar.assignment(name).first_register

    def grouped_compute(ordinary: BackwardPhysicalPlan) -> BackwardPhysicalPlan:
        ordinary_registers = ordinary.registers
        registers = replace(
            ordinary_registers,
            kernarg=first("kernarg"),
            loop_counter=first("loop_counter"),
            block_offset=first("block_offset"),
            input_half=first("input_half"),
            scalar_temporary=first("scalar_temporary"),
            codebook_base=(
                first("codebook_base")
                if state.contract.quant_type == "IQ2_S"
                else ordinary_registers.codebook_base
            ),
            total_sgprs=scalar.register_count,
        )
        resources = replace(ordinary.resources, sgprs=scalar.register_count)
        return replace(ordinary, registers=registers, resources=resources)

    primary = grouped_compute(primary)
    secondary = grouped_compute(secondary) if secondary is not None else None
    route = GroupedBackwardScalarPlan(
        expert_indices=first("expert_indices"),
        expert_offsets=first("expert_offsets"),
        num_experts=first("num_experts"),
        rows=first("rows"),
        bytes_per_expert=first("bytes_per_expert"),
        gemm_index=first("gemm_index"),
        row_begin=first("row_begin"),
        row_end=first("row_end"),
        expert=first("expert"),
        route_rows=first("route_rows"),
        route_output_bytes=first("route_output_bytes"),
        saved_exec=first("saved_exec"),
        pointer_temporary=first("pointer_temporary"),
        tile_end=first("tile_end"),
    )
    return GroupedBackwardPhysicalPlan(
        primary=primary,
        secondary=secondary,
        route=route,
    )
