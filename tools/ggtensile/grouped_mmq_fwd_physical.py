"""Deterministic register ownership for grouped MMQ forward kernels."""

from dataclasses import dataclass

from .kernel_writer_assembly import (
    DeterministicRegisterPlan,
    RegisterAssignment,
    RegisterLifetime,
    RegisterRole,
)
from .mmq_fwd_physical import DecodedWeightLdsRegisterPlan
from .mmq_fwd_spec import (
    DecodedLdsLayout,
    F16D4S4ActivationMetadata,
    ForwardResourceUsage,
)


@dataclass(frozen=True)
class GroupedDirectVectorRegisterPlan:
    c: RegisterAssignment
    sums: RegisterAssignment
    weight_payload: RegisterAssignment
    activation_payload: RegisterAssignment
    weight_metadata: RegisterAssignment
    result_weight_addresses: RegisterAssignment
    weight_address: RegisterAssignment
    activation_addresses: RegisterAssignment
    output_address: RegisterAssignment
    scale: RegisterAssignment
    minimum: RegisterAssignment
    scaled_dm: RegisterAssignment
    activation_scale_sum: RegisterAssignment
    weight_d: RegisterAssignment
    weight_minimum: RegisterAssignment
    activation_d: RegisterAssignment
    activation_sum: RegisterAssignment
    temporary: RegisterAssignment
    output_column: RegisterAssignment
    serial: RegisterAssignment
    activation_row: RegisterAssignment
    register_count: int
    declared_vgprs: int

    @classmethod
    def allocate(cls):
        roles = {
            "c": RegisterRole("c", 8, RegisterLifetime(2, 3), minimum_register=0),
            "sums": RegisterRole("sums", 8, RegisterLifetime(0, 5), minimum_register=8),
            "weight_payload": RegisterRole(
                "weight_payload", 8, RegisterLifetime(2, 3), minimum_register=16
            ),
            "activation_payload": RegisterRole(
                "activation_payload", 8, RegisterLifetime(2, 3), minimum_register=24
            ),
            "weight_metadata": RegisterRole(
                "weight_metadata", 32, RegisterLifetime(1, 4), minimum_register=32
            ),
            "result_weight_addresses": RegisterRole(
                "result_weight_addresses",
                8,
                RegisterLifetime(0, 4),
                minimum_register=64,
            ),
            "weight_address": RegisterRole(
                "weight_address", 1, RegisterLifetime(0, 4), minimum_register=72
            ),
            "activation_addresses": RegisterRole(
                "activation_addresses", 2, RegisterLifetime(0, 4), minimum_register=73
            ),
            "output_address": RegisterRole(
                "output_address", 1, RegisterLifetime(5, 5), minimum_register=75
            ),
            "scale": RegisterRole(
                "scale", 1, RegisterLifetime(2, 4), minimum_register=76
            ),
            "minimum": RegisterRole(
                "minimum", 1, RegisterLifetime(2, 4), minimum_register=77
            ),
            "scaled_dm": RegisterRole(
                "scaled_dm", 1, RegisterLifetime(4, 4), minimum_register=78
            ),
            "activation_scale_sum": RegisterRole(
                "activation_scale_sum", 1, RegisterLifetime(2, 4), minimum_register=79
            ),
            "weight_d": RegisterRole(
                "weight_d", 1, RegisterLifetime(4, 4), minimum_register=80
            ),
            "weight_minimum": RegisterRole(
                "weight_minimum", 1, RegisterLifetime(4, 4), minimum_register=81
            ),
            "activation_d": RegisterRole(
                "activation_d", 1, RegisterLifetime(4, 4), minimum_register=82
            ),
            "activation_sum": RegisterRole(
                "activation_sum", 1, RegisterLifetime(4, 4), minimum_register=83
            ),
            "temporary": RegisterRole(
                "temporary", 1, RegisterLifetime(0, 5), minimum_register=84
            ),
            "output_column": RegisterRole(
                "output_column", 1, RegisterLifetime(0, 5), minimum_register=85
            ),
            "serial": RegisterRole(
                "serial", 1, RegisterLifetime(0, 5), minimum_register=86
            ),
            "activation_row": RegisterRole(
                "activation_row", 1, RegisterLifetime(0, 5), minimum_register=87
            ),
        }
        order = tuple(roles)
        declared_vgprs = 88
        plan = DeterministicRegisterPlan.allocate(
            roles, order, max_registers=declared_vgprs
        )
        assignments = {name: plan.assignment(name) for name in order}
        return cls(
            **assignments,
            register_count=plan.register_count,
            declared_vgprs=declared_vgprs,
        )


@dataclass(frozen=True)
class GroupedDirectScalarRegisterPlan:
    kernarg: RegisterAssignment
    workgroup_tile: RegisterAssignment
    gemm_index: RegisterAssignment
    weights: RegisterAssignment
    activations: RegisterAssignment
    output: RegisterAssignment
    expert_indices: RegisterAssignment
    expert_offsets: RegisterAssignment
    num_experts: RegisterAssignment
    nrows_weight: RegisterAssignment
    nrows_activation: RegisterAssignment
    blocks_per_weight_row: RegisterAssignment
    bytes_per_expert: RegisterAssignment
    row_begin: RegisterAssignment
    row_end: RegisterAssignment
    expert: RegisterAssignment
    route_offset: RegisterAssignment
    row_start: RegisterAssignment
    activation_plane_stride: RegisterAssignment
    activation_block_stride: RegisterAssignment
    loop_counter: RegisterAssignment
    pointer_offset: RegisterAssignment
    exec_mask: RegisterAssignment
    register_count: int
    declared_sgprs: int

    @classmethod
    def allocate(cls):
        whole = RegisterLifetime(0, 5)
        roles = {
            "kernarg": RegisterRole("kernarg", 2, whole, 2, 0),
            "workgroup_tile": RegisterRole("workgroup_tile", 1, whole, 1, 2),
            "gemm_index": RegisterRole("gemm_index", 1, whole, 1, 3),
            "weights": RegisterRole("weights", 2, whole, 2, 4),
            "activations": RegisterRole("activations", 2, whole, 2, 6),
            "output": RegisterRole("output", 2, whole, 2, 8),
            "expert_indices": RegisterRole("expert_indices", 2, whole, 2, 10),
            "expert_offsets": RegisterRole("expert_offsets", 2, whole, 2, 12),
            "num_experts": RegisterRole("num_experts", 1, whole, 1, 14),
            "nrows_weight": RegisterRole("nrows_weight", 1, whole, 1, 15),
            "nrows_activation": RegisterRole("nrows_activation", 1, whole, 1, 16),
            "blocks_per_weight_row": RegisterRole(
                "blocks_per_weight_row", 1, whole, 1, 17
            ),
            "bytes_per_expert": RegisterRole("bytes_per_expert", 2, whole, 2, 18),
            "row_begin": RegisterRole("row_begin", 1, whole, 1, 20),
            "row_end": RegisterRole("row_end", 1, whole, 1, 21),
            "expert": RegisterRole("expert", 2, whole, 2, 22),
            "route_offset": RegisterRole(
                "route_offset", 1, RegisterLifetime(0, 0), 1, 24
            ),
            "row_start": RegisterRole("row_start", 1, RegisterLifetime(1, 5), 1, 24),
            "activation_plane_stride": RegisterRole(
                "activation_plane_stride", 1, RegisterLifetime(1, 5), 1, 25
            ),
            "activation_block_stride": RegisterRole(
                "activation_block_stride", 1, RegisterLifetime(1, 5), 1, 26
            ),
            "loop_counter": RegisterRole(
                "loop_counter", 1, RegisterLifetime(1, 4), 1, 27
            ),
            "pointer_offset": RegisterRole(
                "pointer_offset", 2, RegisterLifetime(0, 0), 1, 28
            ),
            "exec_mask": RegisterRole("exec_mask", 1, RegisterLifetime(2, 5), 1, 28),
        }
        order = tuple(roles)
        declared_sgprs = 32
        plan = DeterministicRegisterPlan.allocate(
            roles, order, max_registers=declared_sgprs
        )
        assignments = {name: plan.assignment(name) for name in order}
        return cls(
            **assignments,
            register_count=plan.register_count,
            declared_sgprs=declared_sgprs,
        )


@dataclass(frozen=True)
class GroupedDirectPhysicalPlan:
    activation_metadata: F16D4S4ActivationMetadata
    vector_registers: GroupedDirectVectorRegisterPlan
    scalar_registers: GroupedDirectScalarRegisterPlan
    resources: ForwardResourceUsage


@dataclass(frozen=True)
class GroupedDecodedScalarRegisterPlan:
    kernarg: RegisterAssignment
    workgroup_tile: RegisterAssignment
    gemm_index: RegisterAssignment
    weights: RegisterAssignment
    activations: RegisterAssignment
    output: RegisterAssignment
    expert_indices: RegisterAssignment
    expert_offsets: RegisterAssignment
    num_experts: RegisterAssignment
    nrows_weight: RegisterAssignment
    nrows_activation: RegisterAssignment
    blocks_per_weight_row: RegisterAssignment
    bytes_per_expert: RegisterAssignment
    row_begin: RegisterAssignment
    row_end: RegisterAssignment
    expert: RegisterAssignment
    route_offset: RegisterAssignment
    row_start: RegisterAssignment
    activation_plane_stride: RegisterAssignment
    packed_block_offset: RegisterAssignment
    loop_counter: RegisterAssignment
    pointer_offset: RegisterAssignment
    exec_mask: RegisterAssignment
    wave: RegisterAssignment
    group_loop: RegisterAssignment
    group_offset: RegisterAssignment
    activation_plane_end: RegisterAssignment
    row_tile_end: RegisterAssignment
    activation_lds_base: RegisterAssignment
    row_tile_rows: RegisterAssignment
    register_count: int
    declared_sgprs: int

    @classmethod
    def allocate(cls):
        whole = RegisterLifetime(0, 5)
        roles = {
            "kernarg": RegisterRole("kernarg", 2, whole, 2, 0),
            "workgroup_tile": RegisterRole("workgroup_tile", 1, whole, 1, 2),
            "gemm_index": RegisterRole("gemm_index", 1, whole, 1, 3),
            "weights": RegisterRole("weights", 2, whole, 2, 4),
            "activations": RegisterRole("activations", 2, whole, 2, 6),
            "output": RegisterRole("output", 2, whole, 2, 8),
            "expert_indices": RegisterRole("expert_indices", 2, whole, 2, 10),
            "expert_offsets": RegisterRole("expert_offsets", 2, whole, 2, 12),
            "num_experts": RegisterRole("num_experts", 1, whole, 1, 14),
            "nrows_weight": RegisterRole("nrows_weight", 1, whole, 1, 15),
            "nrows_activation": RegisterRole("nrows_activation", 1, whole, 1, 16),
            "blocks_per_weight_row": RegisterRole(
                "blocks_per_weight_row", 1, whole, 1, 17
            ),
            "bytes_per_expert": RegisterRole("bytes_per_expert", 2, whole, 2, 18),
            "row_begin": RegisterRole("row_begin", 1, whole, 1, 20),
            "row_end": RegisterRole("row_end", 1, whole, 1, 21),
            "expert": RegisterRole("expert", 2, whole, 2, 22),
            "route_offset": RegisterRole(
                "route_offset", 1, RegisterLifetime(0, 0), 1, 24
            ),
            "row_start": RegisterRole("row_start", 1, RegisterLifetime(1, 5), 1, 24),
            "activation_plane_stride": RegisterRole(
                "activation_plane_stride", 1, RegisterLifetime(1, 5), 1, 25
            ),
            "packed_block_offset": RegisterRole(
                "packed_block_offset", 1, RegisterLifetime(1, 4), 1, 26
            ),
            "loop_counter": RegisterRole(
                "loop_counter", 1, RegisterLifetime(1, 4), 1, 27
            ),
            "pointer_offset": RegisterRole(
                "pointer_offset", 2, RegisterLifetime(0, 0), 1, 28
            ),
            "exec_mask": RegisterRole("exec_mask", 1, RegisterLifetime(2, 5), 1, 28),
            "wave": RegisterRole("wave", 1, RegisterLifetime(1, 4), 1, 30),
            "group_loop": RegisterRole("group_loop", 1, RegisterLifetime(2, 4), 1, 31),
            "group_offset": RegisterRole(
                "group_offset", 1, RegisterLifetime(2, 4), 1, 32
            ),
            "activation_plane_end": RegisterRole(
                "activation_plane_end", 1, RegisterLifetime(1, 4), 1, 33
            ),
            "row_tile_end": RegisterRole(
                "row_tile_end", 1, RegisterLifetime(1, 5), 1, 34
            ),
            "activation_lds_base": RegisterRole(
                "activation_lds_base", 1, RegisterLifetime(1, 4), 1, 35
            ),
            "row_tile_rows": RegisterRole(
                "row_tile_rows", 1, RegisterLifetime(1, 5), 1, 36
            ),
        }
        order = tuple(roles)
        declared_sgprs = 40
        plan = DeterministicRegisterPlan.allocate(
            roles, order, max_registers=declared_sgprs
        )
        assignments = {name: plan.assignment(name) for name in order}
        return cls(
            **assignments,
            register_count=plan.register_count,
            declared_sgprs=declared_sgprs,
        )


@dataclass(frozen=True)
class GroupedDecodedPhysicalPlan:
    layout: DecodedLdsLayout
    registers: DecodedWeightLdsRegisterPlan
    scalar_registers: GroupedDecodedScalarRegisterPlan
    resources: ForwardResourceUsage


def grouped_direct_physical_plan(
    activation_block_bytes: int,
) -> GroupedDirectPhysicalPlan:
    vector = GroupedDirectVectorRegisterPlan.allocate()
    scalar = GroupedDirectScalarRegisterPlan.allocate()
    return GroupedDirectPhysicalPlan(
        activation_metadata=F16D4S4ActivationMetadata(activation_block_bytes),
        vector_registers=vector,
        scalar_registers=scalar,
        resources=ForwardResourceUsage(
            vector.declared_vgprs,
            scalar.declared_sgprs,
            0,
        ),
    )


def grouped_decoded_physical_plan(
    activation_block_bytes: int,
    macro_tile0: int,
) -> GroupedDecodedPhysicalPlan:
    if macro_tile0 not in (64, 128):
        raise ValueError("grouped decoded plan requires a 64- or 128-row tile")
    layout = DecodedLdsLayout.for_activation_block_bytes(
        activation_block_bytes, macro_tile0
    )
    vector = DecodedWeightLdsRegisterPlan.allocate(macro_tile0 // 16)
    scalar = GroupedDecodedScalarRegisterPlan.allocate()
    return GroupedDecodedPhysicalPlan(
        layout=layout,
        registers=vector,
        scalar_registers=scalar,
        resources=ForwardResourceUsage(
            vector.declared_vgprs,
            scalar.declared_sgprs,
            layout.total_bytes,
        ),
    )
