"""Deterministic physical ownership for paired grouped forward."""

from dataclasses import dataclass

from .grouped_mmq_fwd_pair_model import GroupedPairRouteOwnership
from .grouped_mmq_fwd_physical import ForwardResourceUsage
from .kernel_writer_assembly import RegisterAssignment, RegisterLifetime, RegisterRole


@dataclass(frozen=True)
class GroupedIQ2SPairHalfLdsLayout:
    activation_rows: int = 64
    weight_rows: int = 64
    activation_row_stride: int = 144
    weight_row_stride: int = 160
    half_payload_bytes: int = 128
    half_payload_stride: int = 160
    weight_scale_offset: int = 128

    def __post_init__(self) -> None:
        if (
            self.activation_rows,
            self.weight_rows,
            self.activation_row_stride,
            self.weight_row_stride,
            self.half_payload_bytes,
            self.half_payload_stride,
            self.weight_scale_offset,
        ) != (64, 64, 144, 160, 128, 160, 128):
            raise ValueError("paired IQ2_S half-LDS layout has fixed dimensions")

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
class GroupedIQ2XXSPairHalfLdsLayout:
    """The Q8_0-style decoded IQ2_XXS tile used by the paired lowering."""

    activation_rows: int = 64
    weight_rows: int = 64
    activation_row_stride: int = 144
    weight_row_stride: int = 160
    half_payload_bytes: int = 128
    half_payload_stride: int = 160
    weight_scale_offset: int = 128

    def __post_init__(self) -> None:
        if self.activation_rows not in (64, 80) or (
            self.weight_rows,
            self.activation_row_stride,
            self.weight_row_stride,
            self.half_payload_bytes,
            self.half_payload_stride,
            self.weight_scale_offset,
        ) != (64, 144, 160, 128, 160, 128):
            raise ValueError(
                "paired IQ2_XXS half-LDS layout requires J64 or J80 dimensions"
            )

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
class GroupedIQ2SPairVectorRegisterPlan:
    sums_first: RegisterAssignment
    sums_second: RegisterAssignment
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
    def allocate(cls) -> "GroupedIQ2SPairVectorRegisterPlan":
        fixed = cls._fixed
        return cls(
            sums_first=fixed("sums_first", 32, 0, 0, 5),
            sums_second=fixed("sums_second", 32, 32, 0, 5),
            activation_stage=fixed("activation_stage", 18, 64, 2, 2),
            producer_d=fixed("producer_d", 1, 64, 1, 1),
            producer_qh=fixed("producer_qh", 1, 65, 1, 1),
            producer_scales=fixed("producer_scales", 1, 66, 1, 1),
            producer_indices=fixed("producer_indices", 4, 67, 1, 1),
            producer_signs=fixed("producer_signs", 4, 71, 1, 1),
            codebook_payload=fixed("codebook_payload", 16, 75, 1, 1),
            decoded_payload=fixed("decoded_payload", 4, 91, 1, 1),
            decode_auxiliary=fixed("decode_auxiliary", 5, 95, 1, 1),
            producer_address=fixed("producer_address", 1, 100, 1, 1),
            producer_lds_address=fixed("producer_lds_address", 1, 101, 1, 1),
            c=fixed("c", 32, 64, 3, 3),
            weight_payload=fixed("weight_payload", 4, 96, 3, 3),
            activation_payload=fixed("activation_payload", 16, 100, 3, 3),
            weight_scales=fixed("weight_scales", 8, 116, 3, 3),
            zero_accumulator=fixed("zero_accumulator", 8, 124, 0, 3),
            temporary=fixed("temporary", 2, 132, 0, 5),
            weight_scale_address=fixed("weight_scale_address", 4, 134, 0, 3),
            output_address=fixed("output_address", 1, 96, 5, 5),
            weight_address=fixed("weight_address", 1, 138, 0, 5),
            activation_lds_address=fixed("activation_lds_address", 1, 139, 0, 3),
            activation_read_address=fixed("activation_read_address", 1, 140, 0, 3),
            weight_lds_address=fixed("weight_lds_address", 1, 141, 0, 3),
            lane=fixed("lane", 1, 142, 0, 5),
            wave=fixed("wave", 1, 143, 0, 5),
            activation_scale=fixed("activation_scale", 4, 144, 3, 3),
            register_count=148,
            declared_vgprs=148,
        )

    @classmethod
    def allocate_iq2_xxs_j80(cls) -> "GroupedIQ2SPairVectorRegisterPlan":
        fixed = cls._fixed
        return cls(
            sums_first=fixed("sums_first", 40, 0, 0, 5),
            sums_second=fixed("sums_second", 40, 40, 0, 5),
            activation_stage=fixed("activation_stage", 24, 80, 2, 2),
            producer_d=fixed("producer_d", 1, 80, 1, 1),
            producer_qh=fixed("producer_qh", 1, 81, 1, 1),
            producer_scales=fixed("producer_scales", 1, 82, 1, 1),
            producer_indices=fixed("producer_indices", 4, 83, 1, 1),
            producer_signs=fixed("producer_signs", 4, 87, 1, 1),
            codebook_payload=fixed("codebook_payload", 16, 91, 1, 1),
            decoded_payload=fixed("decoded_payload", 4, 107, 1, 1),
            decode_auxiliary=fixed("decode_auxiliary", 5, 111, 1, 1),
            producer_address=fixed("producer_address", 1, 116, 1, 1),
            producer_lds_address=fixed("producer_lds_address", 1, 117, 1, 1),
            c=fixed("c", 40, 80, 3, 3),
            weight_payload=fixed("weight_payload", 4, 120, 3, 3),
            activation_payload=fixed("activation_payload", 20, 124, 3, 3),
            weight_scales=fixed("weight_scales", 8, 144, 3, 3),
            zero_accumulator=fixed("zero_accumulator", 8, 152, 0, 3),
            temporary=fixed("temporary", 2, 160, 0, 5),
            weight_scale_address=fixed("weight_scale_address", 4, 162, 0, 3),
            output_address=fixed("output_address", 1, 120, 5, 5),
            weight_address=fixed("weight_address", 1, 166, 0, 5),
            activation_lds_address=fixed("activation_lds_address", 1, 167, 0, 3),
            activation_read_address=fixed("activation_read_address", 1, 168, 0, 3),
            weight_lds_address=fixed("weight_lds_address", 1, 169, 0, 3),
            lane=fixed("lane", 1, 170, 0, 5),
            wave=fixed("wave", 1, 171, 0, 5),
            activation_scale=fixed("activation_scale", 5, 172, 3, 3),
            register_count=177,
            declared_vgprs=180,
        )


@dataclass(frozen=True)
class GroupedQ3KPairHalfLdsLayout:
    activation_rows: int = 64
    weight_rows: int = 64
    activation_row_stride: int = 144
    weight_row_stride: int = 160
    half_payload_bytes: int = 128
    half_payload_stride: int = 160
    weight_scale_offset: int = 128

    def __post_init__(self) -> None:
        if (
            self.activation_rows,
            self.weight_rows,
            self.activation_row_stride,
            self.weight_row_stride,
            self.half_payload_bytes,
            self.half_payload_stride,
            self.weight_scale_offset,
        ) != (64, 64, 144, 160, 128, 160, 128):
            raise ValueError("paired Q3_K half-LDS layout has fixed dimensions")

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
class GroupedQ3KPairVectorRegisterPlan:
    sums_first: RegisterAssignment
    sums_second: RegisterAssignment
    activation_stage: RegisterAssignment
    producer_low_raw: RegisterAssignment
    producer_high_raw: RegisterAssignment
    producer_metadata: RegisterAssignment
    decoded_payload: RegisterAssignment
    decode_auxiliary: RegisterAssignment
    decode_d: RegisterAssignment
    decode_scale: RegisterAssignment
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
    def allocate(cls) -> "GroupedQ3KPairVectorRegisterPlan":
        fixed = cls._fixed
        return cls(
            sums_first=fixed("sums_first", 32, 0, 0, 5),
            sums_second=fixed("sums_second", 32, 32, 0, 5),
            activation_stage=fixed("activation_stage", 18, 64, 2, 2),
            producer_low_raw=fixed("producer_low_raw", 4, 64, 1, 1),
            producer_high_raw=fixed("producer_high_raw", 4, 68, 1, 1),
            producer_metadata=fixed("producer_metadata", 4, 72, 1, 1),
            decoded_payload=fixed("decoded_payload", 4, 76, 1, 1),
            decode_auxiliary=fixed("decode_auxiliary", 2, 80, 1, 1),
            decode_d=fixed("decode_d", 1, 82, 1, 1),
            decode_scale=fixed("decode_scale", 1, 83, 1, 1),
            producer_address=fixed("producer_address", 1, 84, 1, 1),
            producer_lds_address=fixed("producer_lds_address", 1, 85, 1, 1),
            c=fixed("c", 32, 64, 3, 3),
            weight_payload=fixed("weight_payload", 4, 96, 3, 3),
            activation_payload=fixed("activation_payload", 16, 100, 3, 3),
            weight_scales=fixed("weight_scales", 8, 116, 3, 3),
            zero_accumulator=fixed("zero_accumulator", 8, 124, 0, 3),
            temporary=fixed("temporary", 2, 132, 0, 5),
            weight_scale_address=fixed("weight_scale_address", 4, 134, 0, 3),
            output_address=fixed("output_address", 1, 96, 5, 5),
            weight_address=fixed("weight_address", 1, 138, 0, 5),
            activation_lds_address=fixed("activation_lds_address", 1, 139, 0, 3),
            activation_read_address=fixed("activation_read_address", 1, 140, 0, 3),
            weight_lds_address=fixed("weight_lds_address", 1, 141, 0, 3),
            lane=fixed("lane", 1, 142, 0, 5),
            wave=fixed("wave", 1, 143, 0, 5),
            activation_scale=fixed("activation_scale", 4, 144, 3, 3),
            register_count=148,
            declared_vgprs=148,
        )


@dataclass(frozen=True)
class GroupedIQ2SPairScalarRegisterPlan:
    kernarg: RegisterAssignment
    workgroup_tile: RegisterAssignment
    gemm_index: RegisterAssignment
    weights_first: RegisterAssignment
    weights_second: RegisterAssignment
    activations: RegisterAssignment
    output_first: RegisterAssignment
    output_second: RegisterAssignment
    expert_indices: RegisterAssignment
    expert_offsets: RegisterAssignment
    task_count: RegisterAssignment
    task_experts: RegisterAssignment
    task_row_starts: RegisterAssignment
    task_row_ends: RegisterAssignment
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
    group_offset: RegisterAssignment
    activation_plane_end: RegisterAssignment
    row_tile_end: RegisterAssignment
    row_tile_rows: RegisterAssignment
    register_count: int
    declared_sgprs: int

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
    def allocate(cls) -> "GroupedIQ2SPairScalarRegisterPlan":
        whole = RegisterLifetime(0, 5)
        assignments = {
            "kernarg": RegisterAssignment(RegisterRole("kernarg", 2, whole, 2, 0), 0),
            "workgroup_tile": RegisterAssignment(
                RegisterRole("workgroup_tile", 1, whole), 2
            ),
            "gemm_index": RegisterAssignment(RegisterRole("gemm_index", 1, whole), 3),
            "weights_first": RegisterAssignment(
                RegisterRole("weights_first", 2, whole, 2, 4), 4
            ),
            "weights_second": RegisterAssignment(
                RegisterRole("weights_second", 2, whole, 2, 6), 6
            ),
            "activations": RegisterAssignment(
                RegisterRole("activations", 2, whole, 2, 8), 8
            ),
            "output_first": RegisterAssignment(
                RegisterRole("output_first", 2, whole, 2, 10), 10
            ),
            "output_second": RegisterAssignment(
                RegisterRole("output_second", 2, whole, 2, 12), 12
            ),
            "expert_indices": RegisterAssignment(
                RegisterRole("expert_indices", 2, whole, 2, 14), 14
            ),
            "expert_offsets": RegisterAssignment(
                RegisterRole("expert_offsets", 2, whole, 2, 16), 16
            ),
            "num_experts": RegisterAssignment(
                RegisterRole("num_experts", 1, whole), 18
            ),
            "nrows_weight": RegisterAssignment(
                RegisterRole("nrows_weight", 1, whole), 19
            ),
            "nrows_activation": RegisterAssignment(
                RegisterRole("nrows_activation", 1, whole), 20
            ),
            "blocks_per_weight_row": RegisterAssignment(
                RegisterRole("blocks_per_weight_row", 1, whole), 21
            ),
            "bytes_per_expert": RegisterAssignment(
                RegisterRole("bytes_per_expert", 2, whole, 2, 22), 22
            ),
            "row_begin": RegisterAssignment(RegisterRole("row_begin", 1, whole), 24),
            "row_end": RegisterAssignment(RegisterRole("row_end", 1, whole), 25),
            "expert": RegisterAssignment(RegisterRole("expert", 2, whole, 2, 26), 26),
            "route_offset": RegisterAssignment(
                RegisterRole("route_offset", 1, RegisterLifetime(0, 0)), 28
            ),
            "row_start": RegisterAssignment(
                RegisterRole("row_start", 1, RegisterLifetime(1, 5)), 28
            ),
            "activation_plane_stride": RegisterAssignment(
                RegisterRole("activation_plane_stride", 1, RegisterLifetime(1, 5)), 29
            ),
            "packed_block_offset": RegisterAssignment(
                RegisterRole("packed_block_offset", 1, RegisterLifetime(1, 4)), 30
            ),
            "loop_counter": RegisterAssignment(
                RegisterRole("loop_counter", 1, RegisterLifetime(1, 4)), 31
            ),
            "pointer_offset": RegisterAssignment(
                RegisterRole("pointer_offset", 2, RegisterLifetime(0, 0)), 32
            ),
            "exec_mask": RegisterAssignment(
                RegisterRole("exec_mask", 1, RegisterLifetime(2, 5)), 32
            ),
            "group_offset": RegisterAssignment(
                RegisterRole("group_offset", 1, RegisterLifetime(2, 4)), 34
            ),
            "activation_plane_end": RegisterAssignment(
                RegisterRole("activation_plane_end", 1, RegisterLifetime(1, 4)), 35
            ),
            "row_tile_end": RegisterAssignment(
                RegisterRole("row_tile_end", 1, RegisterLifetime(1, 5)), 36
            ),
            "row_tile_rows": RegisterAssignment(
                RegisterRole("row_tile_rows", 1, RegisterLifetime(1, 5)), 37
            ),
        }
        assignments.update(
            task_count=assignments["expert_indices"],
            task_experts=assignments["expert_offsets"],
            task_row_starts=assignments["expert_indices"],
            task_row_ends=assignments["expert_offsets"],
        )
        return cls(**assignments, register_count=44, declared_sgprs=44)

    @classmethod
    def allocate_row_tasks(cls) -> "GroupedIQ2SPairScalarRegisterPlan":
        fixed = cls._fixed
        whole = RegisterLifetime(0, 5)
        assignments = {
            "kernarg": RegisterAssignment(RegisterRole("kernarg", 2, whole, 2, 0), 0),
            "workgroup_tile": RegisterAssignment(
                RegisterRole("workgroup_tile", 1, whole), 2
            ),
            "gemm_index": RegisterAssignment(RegisterRole("task_index", 1, whole), 3),
            "weights_first": RegisterAssignment(
                RegisterRole("weights_first", 2, whole, 2, 4), 4
            ),
            "weights_second": RegisterAssignment(
                RegisterRole("weights_second", 2, whole, 2, 6), 6
            ),
            "activations": RegisterAssignment(
                RegisterRole("activations", 2, whole, 2, 8), 8
            ),
            "output_first": RegisterAssignment(
                RegisterRole("output_first", 2, whole, 2, 10), 10
            ),
            "output_second": RegisterAssignment(
                RegisterRole("output_second", 2, whole, 2, 12), 12
            ),
            "task_count": RegisterAssignment(
                RegisterRole("task_count", 2, whole, 2, 14), 14
            ),
            "task_experts": RegisterAssignment(
                RegisterRole("task_experts", 2, whole, 2, 16), 16
            ),
            "task_row_starts": RegisterAssignment(
                RegisterRole("task_row_starts", 2, whole, 2, 18), 18
            ),
            "task_row_ends": RegisterAssignment(
                RegisterRole("task_row_ends", 2, whole, 2, 20), 20
            ),
            "num_experts": fixed("num_experts", 1, 22, 0, 0),
            "nrows_weight": fixed("nrows_weight", 1, 23, 0, 0),
            "nrows_activation": fixed("nrows_activation", 1, 24, 0, 1),
            "blocks_per_weight_row": fixed("blocks_per_weight_row", 1, 25, 0, 0),
            "bytes_per_expert": RegisterAssignment(
                RegisterRole("bytes_per_expert", 2, RegisterLifetime(0, 0), 2, 26),
                26,
            ),
            "row_begin": RegisterAssignment(RegisterRole("row_begin", 1, whole), 28),
            "row_end": RegisterAssignment(RegisterRole("row_end", 1, whole), 23),
            "expert": RegisterAssignment(RegisterRole("expert", 1, whole), 30),
            "route_offset": fixed("task_offset", 1, 31, 0, 0),
            "row_start": fixed("row_start", 1, 28, 1, 5),
            "activation_plane_stride": fixed("activation_plane_stride", 1, 29, 1, 5),
            "packed_block_offset": fixed("packed_block_offset", 1, 30, 1, 4),
            "loop_counter": fixed("loop_counter", 1, 31, 1, 4),
            "pointer_offset": fixed("pointer_offset", 2, 32, 0, 0),
            "exec_mask": fixed("exec_mask", 1, 32, 2, 5),
            "group_offset": fixed("group_offset", 1, 34, 2, 4),
            "activation_plane_end": fixed("activation_plane_end", 1, 35, 1, 4),
            "row_tile_end": fixed("row_tile_end", 1, 36, 1, 5),
            "row_tile_rows": fixed("row_tile_rows", 1, 37, 1, 5),
        }
        assignments.update(
            expert_indices=assignments["task_count"],
            expert_offsets=assignments["task_experts"],
        )
        return cls(**assignments, register_count=44, declared_sgprs=44)


@dataclass(frozen=True)
class GroupedIQ2SPairPhysicalPlan:
    layout: GroupedIQ2SPairHalfLdsLayout
    registers: GroupedIQ2SPairVectorRegisterPlan
    scalar_registers: GroupedIQ2SPairScalarRegisterPlan
    resources: ForwardResourceUsage


@dataclass(frozen=True)
class GroupedIQ2XXSPairPhysicalPlan:
    layout: GroupedIQ2XXSPairHalfLdsLayout
    registers: GroupedIQ2SPairVectorRegisterPlan
    scalar_registers: GroupedIQ2SPairScalarRegisterPlan
    resources: ForwardResourceUsage


@dataclass(frozen=True)
class GroupedQ3KPairPhysicalPlan:
    layout: GroupedQ3KPairHalfLdsLayout
    registers: GroupedQ3KPairVectorRegisterPlan
    scalar_registers: GroupedIQ2SPairScalarRegisterPlan
    resources: ForwardResourceUsage


def grouped_iq2_s_pair_physical_plan(
    route_ownership: GroupedPairRouteOwnership = GroupedPairRouteOwnership.SerialRoutes,
) -> GroupedIQ2SPairPhysicalPlan:
    layout = GroupedIQ2SPairHalfLdsLayout()
    vector = GroupedIQ2SPairVectorRegisterPlan.allocate()
    scalar = (
        GroupedIQ2SPairScalarRegisterPlan.allocate_row_tasks()
        if route_ownership is GroupedPairRouteOwnership.DeviceRowTasks64
        else GroupedIQ2SPairScalarRegisterPlan.allocate()
    )
    return GroupedIQ2SPairPhysicalPlan(
        layout=layout,
        registers=vector,
        scalar_registers=scalar,
        resources=ForwardResourceUsage(
            vector.declared_vgprs,
            scalar.declared_sgprs,
            layout.total_bytes,
        ),
    )


def grouped_iq2_xxs_pair_physical_plan(
    route_ownership: GroupedPairRouteOwnership = GroupedPairRouteOwnership.SerialRoutes,
    row_tile: int = 64,
) -> GroupedIQ2XXSPairPhysicalPlan:
    layout = GroupedIQ2XXSPairHalfLdsLayout(activation_rows=row_tile)
    vector = (
        GroupedIQ2SPairVectorRegisterPlan.allocate_iq2_xxs_j80()
        if row_tile == 80
        else GroupedIQ2SPairVectorRegisterPlan.allocate()
    )
    scalar = (
        GroupedIQ2SPairScalarRegisterPlan.allocate_row_tasks()
        if route_ownership is GroupedPairRouteOwnership.DeviceRowTasks64
        else GroupedIQ2SPairScalarRegisterPlan.allocate()
    )
    return GroupedIQ2XXSPairPhysicalPlan(
        layout=layout,
        registers=vector,
        scalar_registers=scalar,
        resources=ForwardResourceUsage(
            vector.declared_vgprs,
            scalar.declared_sgprs,
            layout.total_bytes,
        ),
    )


def grouped_q3_k_pair_physical_plan(
    route_ownership: GroupedPairRouteOwnership = GroupedPairRouteOwnership.SerialRoutes,
) -> GroupedQ3KPairPhysicalPlan:
    layout = GroupedQ3KPairHalfLdsLayout()
    vector = GroupedQ3KPairVectorRegisterPlan.allocate()
    scalar = (
        GroupedIQ2SPairScalarRegisterPlan.allocate_row_tasks()
        if route_ownership is GroupedPairRouteOwnership.DeviceRowTasks64
        else GroupedIQ2SPairScalarRegisterPlan.allocate()
    )
    return GroupedQ3KPairPhysicalPlan(
        layout=layout,
        registers=vector,
        scalar_registers=scalar,
        resources=ForwardResourceUsage(
            vector.declared_vgprs,
            scalar.declared_sgprs,
            layout.total_bytes,
        ),
    )
