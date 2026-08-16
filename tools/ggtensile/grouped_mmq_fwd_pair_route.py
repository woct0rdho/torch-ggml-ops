"""Paired routed ABI emission for research grouped IQ2_S kernels."""

from dataclasses import dataclass
from typing import ClassVar

from .grouped_mmq_fwd_pair_physical import GroupedIQ2SPairScalarRegisterPlan
from .grouped_mmq_fwd_pair_spec import GroupedForwardPairRouteState
from .kernel_writer_assembly import Assembly


@dataclass(frozen=True)
class GroupedPairRouteEmitter:
    registers: GroupedIQ2SPairScalarRegisterPlan
    state: GroupedForwardPairRouteState

    EXIT_LABEL: ClassVar[str] = ".LGroupedPairIQ2SExit"

    def emit(self, asm: Assembly) -> None:
        self._emit_kernarg_loads(asm)
        self._emit_exact_shape_guard(asm)
        self._emit_route_load_and_guard(asm)
        self._emit_expert_pointer_rebase(asm)

    def _emit_kernarg_loads(self, asm: Assembly) -> None:
        scalar = self.registers
        asm.comment("Load the paired 80-byte grouped IQ2_S ABI.")
        for pointer, offset in (
            (scalar.weights_first, 0x00),
            (scalar.weights_second, 0x08),
            (scalar.activations, 0x10),
            (scalar.output_first, 0x18),
            (scalar.output_second, 0x20),
            (scalar.expert_indices, 0x28),
            (scalar.expert_offsets, 0x30),
        ):
            asm.inst(
                f"s_load_dwordx2 s[{pointer.first_register}:{pointer.first_register + 1}], "
                f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
                f"0x{offset:x}"
            )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.num_experts.first_register}:"
            f"{scalar.nrows_weight.first_register}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], 0x38"
        )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.nrows_activation.first_register}:"
            f"{scalar.blocks_per_weight_row.first_register}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], 0x40"
        )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.bytes_per_expert.first_register}:"
            f"{scalar.bytes_per_expert.first_register + 1}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], 0x48"
        )
        asm.inst("s_waitcnt lgkmcnt(0)")

    def _emit_exact_shape_guard(self, asm: Assembly) -> None:
        scalar = self.registers
        checks = (
            (scalar.num_experts.first_register, self.state.physical_experts),
            (scalar.nrows_weight.first_register, self.state.output_features),
            (scalar.nrows_activation.first_register, self.state.aggregate_rows),
            (
                scalar.blocks_per_weight_row.first_register,
                self.state.blocks_per_weight_row,
            ),
            (scalar.bytes_per_expert.first_register, self.state.bytes_per_expert),
            (scalar.bytes_per_expert.first_register + 1, 0),
        )
        asm.comment("Reject paired launch arguments outside the exact shape key.")
        for register, expected in checks:
            asm.inst(f"s_cmp_lg_u32 s{register}, {expected}")
            asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")

    def _emit_route_load_and_guard(self, asm: Assembly) -> None:
        scalar = self.registers
        route_offset = scalar.route_offset.first_register
        gemm_index = scalar.gemm_index.first_register
        asm.comment("Load one shared route entry and its cumulative row bounds.")
        asm.inst(f"s_lshl_b32 s{route_offset}, s{gemm_index}, 2")
        asm.inst(
            f"s_load_dword s{scalar.row_end.first_register}, "
            f"s[{scalar.expert_offsets.first_register}:{scalar.expert_offsets.first_register + 1}], s{route_offset}"
        )
        asm.inst(f"s_lshl_b32 s{route_offset}, s{gemm_index}, 3")
        asm.inst(
            f"s_load_dwordx2 s[{scalar.expert.first_register}:{scalar.expert.first_register + 1}], "
            f"s[{scalar.expert_indices.first_register}:{scalar.expert_indices.first_register + 1}], s{route_offset}"
        )
        asm.inst(f"s_cmp_eq_u32 s{gemm_index}, 0")
        asm.inst("s_cbranch_scc1 .LGroupedPairIQ2SFirstRoute")
        asm.inst(f"s_lshl_b32 s{route_offset}, s{gemm_index}, 2")
        asm.inst(f"s_sub_u32 s{route_offset}, s{route_offset}, 4")
        asm.inst(
            f"s_load_dword s{scalar.row_begin.first_register}, "
            f"s[{scalar.expert_offsets.first_register}:{scalar.expert_offsets.first_register + 1}], s{route_offset}"
        )
        asm.inst("s_branch .LGroupedPairIQ2SRouteLoaded")
        asm.label(".LGroupedPairIQ2SFirstRoute")
        asm.inst(f"s_mov_b32 s{scalar.row_begin.first_register}, 0")
        asm.label(".LGroupedPairIQ2SRouteLoaded")
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.comment("Invalid paired experts and cumulative ranges are inert.")
        for instruction in (
            f"s_cmp_lg_u32 s{scalar.expert.first_register + 1}, 0",
            f"s_cmp_ge_u32 s{scalar.expert.first_register}, s{scalar.num_experts.first_register}",
            f"s_cmp_lt_i32 s{scalar.row_begin.first_register}, 0",
            f"s_cmp_le_i32 s{scalar.row_end.first_register}, s{scalar.row_begin.first_register}",
            f"s_cmp_gt_u32 s{scalar.row_end.first_register}, s{scalar.nrows_activation.first_register}",
        ):
            asm.inst(instruction)
            asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")

    def _emit_expert_pointer_rebase(self, asm: Assembly) -> None:
        scalar = self.registers
        route_offset = scalar.route_offset.first_register
        pointer_offset = scalar.pointer_offset.first_register
        expert = scalar.expert.first_register
        stride = scalar.bytes_per_expert.first_register
        asm.comment("Rebase both packed banks with one shared u64 expert offset.")
        asm.inst(f"s_mul_i32 s{route_offset}, s{expert}, s{stride}")
        asm.inst(f"s_mul_hi_u32 s{pointer_offset}, s{expert}, s{stride}")
        asm.inst(f"s_mul_i32 s{pointer_offset + 1}, s{expert}, s{stride + 1}")
        asm.inst(
            f"s_add_u32 s{pointer_offset}, s{pointer_offset}, s{pointer_offset + 1}"
        )
        for pointer in (scalar.weights_first, scalar.weights_second):
            asm.inst(
                f"s_add_u32 s{pointer.first_register}, s{pointer.first_register}, s{route_offset}"
            )
            asm.inst(
                f"s_addc_u32 s{pointer.first_register + 1}, s{pointer.first_register + 1}, s{pointer_offset}"
            )


@dataclass(frozen=True)
class GroupedPairRowTaskEmitter:
    registers: GroupedIQ2SPairScalarRegisterPlan
    state: GroupedForwardPairRouteState

    EXIT_LABEL: ClassVar[str] = ".LGroupedPairIQ2SExit"

    def emit(self, asm: Assembly) -> None:
        self._emit_kernarg_loads(asm)
        self._emit_exact_shape_guard(asm)
        self._emit_task_load_and_guard(asm)
        self._emit_expert_pointer_rebase(asm)

    def _emit_kernarg_loads(self, asm: Assembly) -> None:
        scalar = self.registers
        asm.comment("Load the paired 96-byte grouped IQ2_S row-task ABI.")
        for pointer, offset in (
            (scalar.weights_first, 0x00),
            (scalar.weights_second, 0x08),
            (scalar.activations, 0x10),
            (scalar.output_first, 0x18),
            (scalar.output_second, 0x20),
            (scalar.task_count, 0x28),
            (scalar.task_experts, 0x30),
            (scalar.task_row_starts, 0x38),
            (scalar.task_row_ends, 0x40),
        ):
            asm.inst(
                f"s_load_dwordx2 s[{pointer.first_register}:{pointer.first_register + 1}], "
                f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
                f"0x{offset:x}"
            )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.num_experts.first_register}:"
            f"{scalar.nrows_weight.first_register}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], 0x48"
        )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.nrows_activation.first_register}:"
            f"{scalar.blocks_per_weight_row.first_register}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], 0x50"
        )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.bytes_per_expert.first_register}:"
            f"{scalar.bytes_per_expert.first_register + 1}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], 0x58"
        )
        asm.inst("s_waitcnt lgkmcnt(0)")

    def _emit_exact_shape_guard(self, asm: Assembly) -> None:
        scalar = self.registers
        checks = (
            (scalar.num_experts.first_register, self.state.physical_experts),
            (scalar.nrows_weight.first_register, self.state.output_features),
            (scalar.nrows_activation.first_register, self.state.aggregate_rows),
            (
                scalar.blocks_per_weight_row.first_register,
                self.state.blocks_per_weight_row,
            ),
            (scalar.bytes_per_expert.first_register, self.state.bytes_per_expert),
            (scalar.bytes_per_expert.first_register + 1, 0),
        )
        asm.comment("Reject paired row-task arguments outside the exact shape key.")
        for register, expected in checks:
            asm.inst(f"s_cmp_lg_u32 s{register}, {expected}")
            asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")

    def _emit_task_load_and_guard(self, asm: Assembly) -> None:
        scalar = self.registers
        task_offset = scalar.route_offset.first_register
        task_index = scalar.gemm_index.first_register
        asm.comment("Load one bounded device-built 64-row task.")
        asm.inst(
            f"s_load_dword s{task_offset}, "
            f"s[{scalar.task_count.first_register}:{scalar.task_count.first_register + 1}], 0x0"
        )
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst(f"s_cmp_ge_u32 s{task_index}, s{task_offset}")
        asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")
        asm.inst(f"s_lshl_b32 s{task_offset}, s{task_index}, 2")
        for destination, pointer in (
            (scalar.expert, scalar.task_experts),
            (scalar.row_begin, scalar.task_row_starts),
            (scalar.row_end, scalar.task_row_ends),
        ):
            asm.inst(
                f"s_load_dword s{destination.first_register}, "
                f"s[{pointer.first_register}:{pointer.first_register + 1}], s{task_offset}"
            )
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.comment("Invalid paired row tasks are inert.")
        for instruction in (
            f"s_cmp_ge_u32 s{scalar.expert.first_register}, s{scalar.num_experts.first_register}",
            f"s_cmp_lt_i32 s{scalar.row_begin.first_register}, 0",
            f"s_cmp_le_i32 s{scalar.row_end.first_register}, s{scalar.row_begin.first_register}",
            f"s_cmp_gt_u32 s{scalar.row_end.first_register}, s{scalar.nrows_activation.first_register}",
        ):
            asm.inst(instruction)
            asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")

    def _emit_expert_pointer_rebase(self, asm: Assembly) -> None:
        scalar = self.registers
        task_offset = scalar.route_offset.first_register
        pointer_offset = scalar.pointer_offset.first_register
        expert = scalar.expert.first_register
        stride = scalar.bytes_per_expert.first_register
        asm.comment("Rebase both packed banks with the task's shared expert.")
        asm.inst(f"s_mul_i32 s{task_offset}, s{expert}, s{stride}")
        asm.inst(f"s_mul_hi_u32 s{pointer_offset}, s{expert}, s{stride}")
        asm.inst(f"s_mul_i32 s{pointer_offset + 1}, s{expert}, s{stride + 1}")
        asm.inst(
            f"s_add_u32 s{pointer_offset}, s{pointer_offset}, s{pointer_offset + 1}"
        )
        for pointer in (scalar.weights_first, scalar.weights_second):
            asm.inst(
                f"s_add_u32 s{pointer.first_register}, s{pointer.first_register}, s{task_offset}"
            )
            asm.inst(
                f"s_addc_u32 s{pointer.first_register + 1}, s{pointer.first_register + 1}, s{pointer_offset}"
            )
