"""Deterministic physical state for paired grouped backward kernels."""

from dataclasses import dataclass, replace

from .grouped_mmq_bwd_pair_spec import DerivedGroupedBackwardPairState
from .grouped_mmq_bwd_physical import GroupedBackwardScalarPlan
from .iq2_s_grid import IQ2_S_GRID_BYTES
from .iq2_xxs_grid import IQ2_XXS_GRID_BYTES
from .kernel_writer_assembly import (
    DeterministicRegisterPlan,
    RegisterLifetime,
    RegisterRole,
)
from .mmq_bwd_physical import BackwardPhysicalPlan, derive_backward_physical_plan


@dataclass(frozen=True)
class GroupedBackwardPairScalarPlan(GroupedBackwardScalarPlan):
    second_grad_output: int
    second_packed_weight: int


@dataclass(frozen=True)
class GroupedBackwardPairPhysicalPlan:
    ordinary: BackwardPhysicalPlan
    second_projection: BackwardPhysicalPlan | None
    scalar: GroupedBackwardPairScalarPlan


def derive_grouped_backward_pair_physical_plan(
    state: DerivedGroupedBackwardPairState,
) -> GroupedBackwardPairPhysicalPlan:
    ordinary = derive_backward_physical_plan(state.ordinary)
    lifetime = RegisterLifetime(0, 3)
    widths = {
        "kernarg": 6,
        "second_grad_output": 2,
        "second_packed_weight": 2,
        "loop_counter": 1,
        "block_offset": 1,
        "input_half": 1,
        "scalar_temporary": 2,
        "codebook_base": 2,
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
    aligned = {
        "kernarg",
        "second_grad_output",
        "second_packed_weight",
        "scalar_temporary",
        "codebook_base",
        "expert_indices",
        "expert_offsets",
        "bytes_per_expert",
        "expert",
        "pointer_temporary",
    }
    roles = {
        name: RegisterRole(
            name,
            width,
            lifetime,
            alignment=2 if name in aligned else 1,
            minimum_register=5,
        )
        for name, width in widths.items()
    }
    plan = DeterministicRegisterPlan.allocate(
        roles,
        tuple(widths),
        max_registers=64,
    )

    def first(name: str) -> int:
        return plan.assignment(name).first_register

    registers = replace(
        ordinary.registers,
        kernarg=first("kernarg"),
        loop_counter=first("loop_counter"),
        block_offset=first("block_offset"),
        input_half=first("input_half"),
        scalar_temporary=first("scalar_temporary"),
        codebook_base=first("codebook_base"),
        total_sgprs=plan.register_count,
    )
    resources = replace(ordinary.resources, sgprs=plan.register_count)
    ordinary = replace(ordinary, registers=registers, resources=resources)
    scalar = GroupedBackwardPairScalarPlan(
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
        second_grad_output=first("second_grad_output"),
        second_packed_weight=first("second_packed_weight"),
    )
    policy = state.kernel_spec.projection_policy
    second_projection = None
    if policy.dual_lds:
        decoded_bytes = ordinary.lds.num_bytes
        codebook_offset = 2 * decoded_bytes
        codebook_in_lds = not policy.uses_global_codebook
        codebook_bytes = {
            "Q3_K": 0,
            "IQ2_S": IQ2_S_GRID_BYTES,
            "IQ2_XXS": IQ2_XXS_GRID_BYTES,
        }[state.contract.quant_type]
        resources = replace(
            ordinary.resources,
            lds_bytes=(
                codebook_offset + codebook_bytes if codebook_in_lds else codebook_offset
            ),
        )
        first_lds = replace(
            ordinary.lds,
            base_offset=0,
            codebook_offset=codebook_offset,
            codebook_in_lds=codebook_in_lds,
        )
        ordinary = replace(ordinary, lds=first_lds, resources=resources)
        second_projection = replace(
            ordinary,
            lds=replace(first_lds, base_offset=decoded_bytes),
        )
        if policy.direct_second_pointers and not policy.concurrent_reads:
            second_projection = replace(
                second_projection,
                registers=replace(
                    second_projection.registers,
                    kernarg=scalar.second_grad_output,
                ),
            )
        if policy.concurrent_reads:
            second_payload = (ordinary.registers.total_vgprs + 3) // 4 * 4
            second_registers = replace(
                second_projection.registers,
                kernarg=(
                    scalar.second_grad_output
                    if policy.direct_second_pointers
                    else second_projection.registers.kernarg
                ),
                global_read_b=second_payload,
                quant_dm=second_payload + 4,
                quant_scale=second_payload + 5,
                total_vgprs=second_payload + 9,
            )
            resources = replace(resources, vgprs=second_payload + 9)
            ordinary = replace(
                ordinary,
                registers=replace(
                    ordinary.registers,
                    total_vgprs=second_payload + 9,
                ),
                resources=resources,
            )
            second_projection = replace(
                second_projection,
                registers=second_registers,
                resources=resources,
            )
    return GroupedBackwardPairPhysicalPlan(ordinary, second_projection, scalar)
