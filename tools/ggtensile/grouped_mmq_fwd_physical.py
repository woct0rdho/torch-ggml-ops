"""Deterministic register ownership for grouped MMQ forward kernels."""

from dataclasses import dataclass
from enum import Enum

from .grouped_mmq_fwd_model import GroupedActivationStaging
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
    PhysicalResourceUsage,
)
from .quant_formats import Q8_1_F32_D4_BLOCK_BYTES


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
    resources: PhysicalResourceUsage


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
    activation_staging: "GroupedActivationStagingPlan"
    resources: PhysicalResourceUsage


class GroupedActivationStageBounds(Enum):
    Exact = "Exact"
    BoundsMasked = "BoundsMasked"


@dataclass(frozen=True)
class GroupedActivationStage:
    row_tile_rows: int
    total_bytes: int
    vector_load_bytes: int
    loads_per_thread: int
    lds_write_dwords: int
    bounds: GroupedActivationStageBounds

    @property
    def bytes_per_thread(self) -> int:
        return self.loads_per_thread * self.vector_load_bytes

    @property
    def stage_dwords(self) -> int:
        return self.bytes_per_thread // 4

    @property
    def requires_bounds_mask(self) -> bool:
        return self.bounds is GroupedActivationStageBounds.BoundsMasked


@dataclass(frozen=True)
class GroupedActivationStagingPlan:
    staging: GroupedActivationStaging
    block_bytes: int
    participating_threads: int
    vector_load_bytes: int = 4
    lds_write_dwords: int = 2

    def __post_init__(self) -> None:
        assert not (self.block_bytes <= 0 or self.participating_threads <= 0)
        assert not (self.vector_load_bytes != 4 or self.lds_write_dwords != 2)

    def stage(self, row_tile_rows: int) -> GroupedActivationStage:
        assert not (row_tile_rows <= 0 or row_tile_rows % 16)
        total_bytes = row_tile_rows * self.block_bytes
        bytes_per_round = self.vector_load_bytes * self.participating_threads
        loads_per_thread = (total_bytes + bytes_per_round - 1) // bytes_per_round
        covered_bytes = loads_per_thread * bytes_per_round
        bounds = (
            GroupedActivationStageBounds.Exact
            if covered_bytes == total_bytes
            else GroupedActivationStageBounds.BoundsMasked
        )
        return GroupedActivationStage(
            row_tile_rows=row_tile_rows,
            total_bytes=total_bytes,
            vector_load_bytes=self.vector_load_bytes,
            loads_per_thread=loads_per_thread,
            lds_write_dwords=self.lds_write_dwords,
            bounds=bounds,
        )


@dataclass(frozen=True)
class GroupedIQ2SFullWeightLdsLayout:
    """Compact J64 LDS ownership for decoded IQ2_S weights and F32_D4 rows."""

    @property
    def activation_rows(self) -> int:
        return 64

    @property
    def weight_rows(self) -> int:
        return 64

    @property
    def activation_row_stride(self) -> int:
        return Q8_1_F32_D4_BLOCK_BYTES

    @property
    def half_payload_bytes(self) -> int:
        return 128

    @property
    def half_payload_stride(self) -> int:
        return 160

    @property
    def weight_scale_offset(self) -> int:
        return 128

    @property
    def weight_scale_bytes(self) -> int:
        return 32

    @property
    def weight_row_stride(self) -> int:
        return 336

    @property
    def activation_bytes(self) -> int:
        return self.activation_rows * self.activation_row_stride

    @property
    def weight_base(self) -> int:
        return self.activation_bytes

    @property
    def weight_bytes(self) -> int:
        return self.weight_rows * self.weight_row_stride

    @property
    def total_bytes(self) -> int:
        return self.activation_bytes + self.weight_bytes


@dataclass(frozen=True)
class GroupedIQ2SFullWeightRegisterPlan:
    """Fixed register ownership for distributed IQ2_S decode and J64 WMMA."""

    sums: RegisterAssignment
    activation_stage: RegisterAssignment
    producer_d: RegisterAssignment
    producer_qh: RegisterAssignment
    producer_scales: RegisterAssignment
    producer_indices: RegisterAssignment
    producer_signs: RegisterAssignment
    codebook_payload: RegisterAssignment
    decoded_payload: RegisterAssignment
    decode_auxiliary: RegisterAssignment
    producer_address: RegisterAssignment
    producer_lds_address: RegisterAssignment
    c: RegisterAssignment
    weight_payload: RegisterAssignment
    activation_payload: RegisterAssignment
    weight_scales: RegisterAssignment
    zero_accumulator: RegisterAssignment
    temporary: RegisterAssignment
    weight_scale_address: RegisterAssignment
    output_address: RegisterAssignment
    weight_address: RegisterAssignment
    activation_lds_address: RegisterAssignment
    activation_read_address: RegisterAssignment
    weight_lds_address: RegisterAssignment
    lane: RegisterAssignment
    wave: RegisterAssignment
    activation_scale: RegisterAssignment
    register_count: int
    declared_vgprs: int

    @staticmethod
    def _fixed(
        name: str,
        width: int,
        first: int,
        first_stage: int,
        last_stage: int,
    ) -> RegisterAssignment:
        return RegisterAssignment(
            RegisterRole(name, width, RegisterLifetime(first_stage, last_stage)),
            first,
        )

    @classmethod
    def allocate(cls):
        fixed = cls._fixed
        return cls(
            sums=fixed("sums", 32, 0, 0, 5),
            activation_stage=fixed("activation_stage", 18, 32, 2, 2),
            producer_d=fixed("producer_d", 1, 32, 1, 1),
            producer_qh=fixed("producer_qh", 1, 33, 1, 1),
            producer_scales=fixed("producer_scales", 1, 34, 1, 1),
            producer_indices=fixed("producer_indices", 8, 35, 1, 1),
            producer_signs=fixed("producer_signs", 8, 43, 1, 1),
            codebook_payload=fixed("codebook_payload", 32, 51, 1, 1),
            decoded_payload=fixed("decoded_payload", 4, 83, 1, 1),
            decode_auxiliary=fixed("decode_auxiliary", 5, 87, 1, 1),
            producer_address=fixed("producer_address", 1, 92, 1, 1),
            producer_lds_address=fixed("producer_lds_address", 1, 93, 1, 1),
            c=fixed("c", 32, 32, 3, 3),
            weight_payload=fixed("weight_payload", 4, 64, 3, 3),
            activation_payload=fixed("activation_payload", 16, 68, 3, 3),
            weight_scales=fixed("weight_scales", 8, 84, 3, 3),
            zero_accumulator=fixed("zero_accumulator", 8, 92, 0, 3),
            temporary=fixed("temporary", 2, 100, 0, 5),
            weight_scale_address=fixed("weight_scale_address", 4, 102, 0, 3),
            output_address=fixed("output_address", 1, 64, 5, 5),
            weight_address=fixed("weight_address", 1, 106, 0, 5),
            activation_lds_address=fixed("activation_lds_address", 1, 107, 0, 3),
            activation_read_address=fixed("activation_read_address", 1, 108, 0, 3),
            weight_lds_address=fixed("weight_lds_address", 1, 109, 0, 3),
            lane=fixed("lane", 1, 110, 0, 5),
            wave=fixed("wave", 1, 111, 0, 5),
            activation_scale=fixed("activation_scale", 4, 112, 3, 3),
            register_count=116,
            declared_vgprs=116,
        )


@dataclass(frozen=True)
class GroupedIQ2SFullWeightPhysicalPlan:
    layout: GroupedIQ2SFullWeightLdsLayout
    registers: GroupedIQ2SFullWeightRegisterPlan
    scalar_registers: GroupedDecodedScalarRegisterPlan
    resources: PhysicalResourceUsage


def grouped_direct_physical_plan(
    activation_block_bytes: int,
) -> GroupedDirectPhysicalPlan:
    vector = GroupedDirectVectorRegisterPlan.allocate()
    scalar = GroupedDirectScalarRegisterPlan.allocate()
    return GroupedDirectPhysicalPlan(
        activation_metadata=F16D4S4ActivationMetadata(activation_block_bytes),
        vector_registers=vector,
        scalar_registers=scalar,
        resources=PhysicalResourceUsage(
            vector.declared_vgprs,
            scalar.declared_sgprs,
            0,
        ),
    )


def grouped_decoded_physical_plan(
    activation_block_bytes: int,
    macro_tile0: int,
    quant_type: str,
    activation_staging: GroupedActivationStagingPlan,
) -> GroupedDecodedPhysicalPlan:
    if quant_type == "Q2_K":
        assert macro_tile0 in (32, 64, 128)
        layout = DecodedLdsLayout.for_q2_activation_block_bytes(
            activation_block_bytes, macro_tile0
        )
    else:
        assert macro_tile0 in (64, 128)
        layout = DecodedLdsLayout.for_activation_block_bytes(
            activation_block_bytes, macro_tile0
        )
    vector = DecodedWeightLdsRegisterPlan.allocate(macro_tile0 // 16)
    scalar = GroupedDecodedScalarRegisterPlan.allocate()
    assert activation_staging.block_bytes == activation_block_bytes
    return GroupedDecodedPhysicalPlan(
        layout=layout,
        registers=vector,
        scalar_registers=scalar,
        activation_staging=activation_staging,
        resources=PhysicalResourceUsage(
            vector.declared_vgprs,
            scalar.declared_sgprs,
            layout.total_bytes,
        ),
    )


def grouped_iq2_s_full_weight_physical_plan() -> GroupedIQ2SFullWeightPhysicalPlan:
    layout = GroupedIQ2SFullWeightLdsLayout()
    vector = GroupedIQ2SFullWeightRegisterPlan.allocate()
    scalar = GroupedDecodedScalarRegisterPlan.allocate()
    return GroupedIQ2SFullWeightPhysicalPlan(
        layout=layout,
        registers=vector,
        scalar_registers=scalar,
        resources=PhysicalResourceUsage(
            vector.declared_vgprs,
            scalar.declared_sgprs,
            layout.total_bytes,
        ),
    )
