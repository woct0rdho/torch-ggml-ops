"""Paired routed ABI emission for research grouped forward kernels."""

from dataclasses import dataclass

from .grouped_mmq_fwd_pair_physical import GroupedIQ2SPairScalarRegisterPlan
from .grouped_mmq_fwd_pair_spec import GroupedForwardPairRouteState
from .kernel_abi import (
    GROUPED_FORWARD_PAIR_ABI,
    GROUPED_FORWARD_PAIR_ROW_TASK_ABI,
)
from .kernel_writer_assembly import Assembly


@dataclass(frozen=True)
class GroupedPairRouteEmitter:
    registers: GroupedIQ2SPairScalarRegisterPlan
    state: GroupedForwardPairRouteState
    quant_type: str = "IQ2_S"
    label_token: str = "IQ2S"

    @property
    def exit_label(self) -> str:
        return f".LGroupedPair{self.label_token}Exit"

    def emit(self, asm: Assembly) -> None:
        self._emit_kernarg_loads(asm)
        self._emit_exact_shape_guard(asm)
        self._emit_route_load_and_guard(asm)
        self._emit_expert_pointer_rebase(asm)

    def _emit_kernarg_loads(self, asm: Assembly) -> None:
        scalar = self.registers
        abi = GROUPED_FORWARD_PAIR_ABI
        asm.comment(
            f"Load the paired {abi.segment_size}-byte grouped {self.quant_type} ABI."
        )
        for pointer, argument_name in (
            (scalar.weights_first, "weights_first"),
            (scalar.weights_second, "weights_second"),
            (scalar.activations, "activations"),
            (scalar.output_first, "dst_first"),
            (scalar.output_second, "dst_second"),
            (scalar.expert_indices, "expert_indices"),
            (scalar.expert_offsets, "expert_offsets"),
        ):
            asm.inst(
                f"s_load_dwordx2 s[{pointer.first_register}:{pointer.first_register + 1}], "
                f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
                f"0x{abi.offset(argument_name):x}"
            )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.num_experts.first_register}:"
            f"{scalar.nrows_weight.first_register}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
            f"0x{abi.offset('num_experts'):x}"
        )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.nrows_activation.first_register}:"
            f"{scalar.blocks_per_weight_row.first_register}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
            f"0x{abi.offset('nrows_activation'):x}"
        )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.bytes_per_expert.first_register}:"
            f"{scalar.bytes_per_expert.first_register + 1}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
            f"0x{abi.offset('bytes_per_expert'):x}"
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
            asm.inst(f"s_cbranch_scc1 {self.exit_label}")

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
        asm.inst(f"s_cbranch_scc1 .LGroupedPair{self.label_token}FirstRoute")
        asm.inst(f"s_lshl_b32 s{route_offset}, s{gemm_index}, 2")
        asm.inst(f"s_sub_u32 s{route_offset}, s{route_offset}, 4")
        asm.inst(
            f"s_load_dword s{scalar.row_begin.first_register}, "
            f"s[{scalar.expert_offsets.first_register}:{scalar.expert_offsets.first_register + 1}], s{route_offset}"
        )
        asm.inst(f"s_branch .LGroupedPair{self.label_token}RouteLoaded")
        asm.label(f".LGroupedPair{self.label_token}FirstRoute")
        asm.inst(f"s_mov_b32 s{scalar.row_begin.first_register}, 0")
        asm.label(f".LGroupedPair{self.label_token}RouteLoaded")
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
            asm.inst(f"s_cbranch_scc1 {self.exit_label}")

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
    quant_type: str = "IQ2_S"
    label_token: str = "IQ2S"

    @property
    def exit_label(self) -> str:
        return f".LGroupedPair{self.label_token}Exit"

    def emit(self, asm: Assembly) -> None:
        self._emit_kernarg_loads(asm)
        self._emit_exact_shape_guard(asm)
        self._emit_task_load_and_guard(asm)
        self._emit_expert_pointer_rebase(asm)

    def _emit_kernarg_loads(self, asm: Assembly) -> None:
        scalar = self.registers
        abi = GROUPED_FORWARD_PAIR_ROW_TASK_ABI
        asm.comment(
            f"Load the paired {abi.segment_size}-byte grouped {self.quant_type} "
            "row-task ABI."
        )
        for pointer, argument_name in (
            (scalar.weights_first, "weights_first"),
            (scalar.weights_second, "weights_second"),
            (scalar.activations, "activations"),
            (scalar.output_first, "dst_first"),
            (scalar.output_second, "dst_second"),
            (scalar.task_count, "task_count"),
            (scalar.task_experts, "task_experts"),
            (scalar.task_row_starts, "task_row_starts"),
            (scalar.task_row_ends, "task_row_ends"),
        ):
            asm.inst(
                f"s_load_dwordx2 s[{pointer.first_register}:{pointer.first_register + 1}], "
                f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
                f"0x{abi.offset(argument_name):x}"
            )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.num_experts.first_register}:"
            f"{scalar.nrows_weight.first_register}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
            f"0x{abi.offset('num_experts'):x}"
        )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.nrows_activation.first_register}:"
            f"{scalar.blocks_per_weight_row.first_register}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
            f"0x{abi.offset('nrows_activation'):x}"
        )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.bytes_per_expert.first_register}:"
            f"{scalar.bytes_per_expert.first_register + 1}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
            f"0x{abi.offset('bytes_per_expert'):x}"
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
            asm.inst(f"s_cbranch_scc1 {self.exit_label}")

    def _emit_task_load_and_guard(self, asm: Assembly) -> None:
        scalar = self.registers
        task_offset = scalar.route_offset.first_register
        task_index = scalar.gemm_index.first_register
        asm.comment(
            f"Load one bounded device-built {self.state.row_task_rows}-row task."
        )
        asm.inst(
            f"s_load_dword s{task_offset}, "
            f"s[{scalar.task_count.first_register}:{scalar.task_count.first_register + 1}], 0x0"
        )
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst(f"s_cmp_ge_u32 s{task_index}, s{task_offset}")
        asm.inst(f"s_cbranch_scc1 {self.exit_label}")
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
            asm.inst(f"s_cbranch_scc1 {self.exit_label}")

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
