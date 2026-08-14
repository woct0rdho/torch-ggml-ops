"""Serial direct-global lowering for grouped Q4_K MMQ forward."""

from dataclasses import dataclass
from typing import cast

from .grouped_mmq_fwd_model import GroupedForwardSolutionKey
from .grouped_mmq_fwd_physical import GroupedDirectPhysicalPlan
from .grouped_mmq_fwd_spec import DerivedGroupedForwardState
from .kernel_writer_assembly import Assembly, emit_bf16_rne, emit_kernel_trailer
from .mmq_fwd_lowering_metadata import emit_packed_scale_minimum
from .mmq_fwd_lowering_mma import emit_signed_i8_wmma


@dataclass(frozen=True)
class GroupedForwardLoweringContext:
    solution_key: GroupedForwardSolutionKey
    state: DerivedGroupedForwardState


@dataclass(frozen=True)
class GroupedPackedScaleMinimumDirectLowering:
    """Emit one wave per `(gemmIndex, MacroTile1)` serial row owner."""

    context: GroupedForwardLoweringContext

    def _physical_plan(self) -> GroupedDirectPhysicalPlan:
        return cast(GroupedDirectPhysicalPlan, self.context.state.physical_plan)

    def body(self) -> str:
        state = self.context.state
        physical = self._physical_plan()
        vector = physical.vector_registers
        scalar = physical.scalar_registers
        asm = Assembly()
        name = self.context.solution_key.kernel_name

        self._emit_kernarg_loads(asm)
        self._emit_exact_shape_guard(asm)
        self._emit_route_load_and_guard(asm)
        self._emit_expert_pointer_rebase(asm)

        asm.inst(
            f"s_mul_i32 s{scalar.activation_plane_stride.first_register}, "
            f"s{scalar.nrows_activation.first_register}, "
            f"{self.context.solution_key.solution.activation_block_bytes}"
        )
        asm.inst(
            f"s_lshl_b32 s{scalar.activation_block_stride.first_register}, "
            f"s{scalar.activation_plane_stride.first_register}, 1"
        )
        asm.inst(
            f"s_mov_b32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_begin.first_register}"
        )

        asm.comment("Map one wave to one exact 16-column output tile.")
        asm.inst(f"v_mov_b32 v{vector.serial.first_register}, v0")
        asm.inst(
            f"v_and_b32 v{vector.output_column.first_register}, 15, "
            f"v{vector.serial.first_register}"
        )
        asm.inst(
            f"v_lshlrev_b32 v{vector.temporary.first_register}, 4, "
            f"s{scalar.workgroup_tile.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{vector.output_column.first_register}, "
            f"v{vector.temporary.first_register}, "
            f"v{vector.output_column.first_register}"
        )

        asm.label(".LGroupedQ4KRowLoop")
        self._emit_row_setup(asm)
        for register in vector.sums.registers:
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"s_mov_b32 s{scalar.loop_counter.first_register}, 0")

        asm.label(".LGroupedQ4KBlockLoop")
        asm.comment("Keep one Q4_K block's dm/scales live across its eight groups.")
        for element in range(8):
            metadata = vector.weight_metadata.first_register + 4 * element
            asm.inst(
                f"global_load_b128 v[{metadata}:{metadata + 3}], "
                f"v{vector.result_weight_addresses.first_register + element}, "
                f"s[{scalar.weights.first_register}:{scalar.weights.first_register + 1}]"
            )
        for group in range(8):
            self._emit_group(asm, group)
        for element in range(8):
            address = vector.result_weight_addresses.first_register + element
            asm.inst(
                f"v_add_nc_u32 v{address}, "
                f"{self.context.solution_key.solution.packed_weight_block_bytes}, "
                f"v{address}"
            )
        asm.inst(
            f"v_add_nc_u32 v{vector.weight_address.first_register}, "
            f"{self.context.solution_key.solution.packed_weight_block_bytes}, "
            f"v{vector.weight_address.first_register}"
        )
        for address in vector.activation_addresses.registers:
            asm.inst(
                f"v_add_nc_u32 v{address}, "
                f"s{scalar.activation_block_stride.first_register}, v{address}"
            )
        asm.inst(
            f"s_add_u32 s{scalar.loop_counter.first_register}, "
            f"s{scalar.loop_counter.first_register}, 1"
        )
        asm.inst(
            f"s_cmp_lt_u32 s{scalar.loop_counter.first_register}, "
            f"{state.blocks_per_weight_row}"
        )
        asm.inst("s_cbranch_scc1 .LGroupedQ4KBlockLoop")

        self._emit_store(asm)
        asm.inst(
            f"s_add_u32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_start.first_register}, "
            f"{self.context.solution_key.solution.macro_tile0}"
        )
        asm.inst(
            f"s_cmp_lt_u32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_end.first_register}"
        )
        asm.inst("s_cbranch_scc1 .LGroupedQ4KRowLoop")

        asm.label(".LGroupedQ4KExit")
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_kernarg_loads(self, asm: Assembly) -> None:
        scalar = self.context.state.physical_plan.scalar_registers
        asm.comment("Load the routed 64-byte grouped forward ABI.")
        for pointer, offset in (
            (scalar.weights, 0x00),
            (scalar.activations, 0x08),
            (scalar.output, 0x10),
            (scalar.expert_indices, 0x18),
            (scalar.expert_offsets, 0x20),
        ):
            asm.inst(
                f"s_load_dwordx2 s[{pointer.first_register}:{pointer.first_register + 1}], "
                f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
                f"0x{offset:x}"
            )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.num_experts.first_register}:"
            f"{scalar.num_experts.first_register + 1}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
            "0x28"
        )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.nrows_activation.first_register}:"
            f"{scalar.blocks_per_weight_row.first_register}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
            "0x30"
        )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.bytes_per_expert.first_register}:"
            f"{scalar.bytes_per_expert.first_register + 1}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
            "0x38"
        )
        asm.inst("s_waitcnt lgkmcnt(0)")

    def _emit_exact_shape_guard(self, asm: Assembly) -> None:
        state = self.context.state
        problem = self.context.solution_key.problem
        scalar = state.physical_plan.scalar_registers
        asm.comment("Reject launch arguments outside this exact grouped key.")
        checks = (
            (scalar.num_experts.first_register, problem.physical_experts),
            (scalar.nrows_weight.first_register, problem.output_features),
            (scalar.nrows_activation.first_register, problem.aggregate_rows),
            (scalar.blocks_per_weight_row.first_register, state.blocks_per_weight_row),
            (scalar.bytes_per_expert.first_register, state.bytes_per_expert),
            (scalar.bytes_per_expert.first_register + 1, 0),
        )
        for register, expected in checks:
            asm.inst(f"s_cmp_lg_u32 s{register}, {expected}")
            asm.inst("s_cbranch_scc1 .LGroupedQ4KExit")

    def _emit_route_load_and_guard(self, asm: Assembly) -> None:
        scalar = self.context.state.physical_plan.scalar_registers
        route_offset = scalar.route_offset.first_register
        gemm_index = scalar.gemm_index.first_register
        asm.comment("Load cumulative row bounds and the selected physical expert.")
        asm.inst(f"s_lshl_b32 s{route_offset}, s{gemm_index}, 2")
        asm.inst(
            f"s_load_dword s{scalar.row_end.first_register}, "
            f"s[{scalar.expert_offsets.first_register}:"
            f"{scalar.expert_offsets.first_register + 1}], s{route_offset}"
        )
        asm.inst(f"s_lshl_b32 s{route_offset}, s{gemm_index}, 3")
        asm.inst(
            f"s_load_dwordx2 s[{scalar.expert.first_register}:"
            f"{scalar.expert.first_register + 1}], "
            f"s[{scalar.expert_indices.first_register}:"
            f"{scalar.expert_indices.first_register + 1}], s{route_offset}"
        )
        asm.inst(f"s_cmp_eq_u32 s{gemm_index}, 0")
        asm.inst("s_cbranch_scc1 .LGroupedQ4KFirstRoute")
        asm.inst(f"s_lshl_b32 s{route_offset}, s{gemm_index}, 2")
        asm.inst(f"s_sub_u32 s{route_offset}, s{route_offset}, 4")
        asm.inst(
            f"s_load_dword s{scalar.row_begin.first_register}, "
            f"s[{scalar.expert_offsets.first_register}:"
            f"{scalar.expert_offsets.first_register + 1}], s{route_offset}"
        )
        asm.inst("s_branch .LGroupedQ4KRouteLoaded")
        asm.label(".LGroupedQ4KFirstRoute")
        asm.inst(f"s_mov_b32 s{scalar.row_begin.first_register}, 0")
        asm.label(".LGroupedQ4KRouteLoaded")
        asm.inst("s_waitcnt lgkmcnt(0)")

        asm.comment("Invalid experts and cumulative ranges are inert.")
        guard_instructions = (
            f"s_cmp_lg_u32 s{scalar.expert.first_register + 1}, 0",
            f"s_cmp_ge_u32 s{scalar.expert.first_register}, s{scalar.num_experts.first_register}",
            f"s_cmp_lt_i32 s{scalar.row_begin.first_register}, 0",
            f"s_cmp_le_i32 s{scalar.row_end.first_register}, s{scalar.row_begin.first_register}",
            f"s_cmp_gt_u32 s{scalar.row_end.first_register}, s{scalar.nrows_activation.first_register}",
        )
        for instruction in guard_instructions:
            asm.inst(instruction)
            asm.inst("s_cbranch_scc1 .LGroupedQ4KExit")

    def _emit_expert_pointer_rebase(self, asm: Assembly) -> None:
        scalar = self.context.state.physical_plan.scalar_registers
        route_offset = scalar.route_offset.first_register
        pointer_offset = scalar.pointer_offset.first_register
        expert = scalar.expert.first_register
        stride = scalar.bytes_per_expert.first_register
        weights = scalar.weights.first_register
        asm.comment("Rebase the shared packed bank with the full u64 expert stride.")
        asm.inst(f"s_mul_i32 s{route_offset}, s{expert}, s{stride}")
        asm.inst(f"s_mul_hi_u32 s{pointer_offset}, s{expert}, s{stride}")
        asm.inst(f"s_mul_i32 s{pointer_offset + 1}, s{expert}, s{stride + 1}")
        asm.inst(
            f"s_add_u32 s{pointer_offset}, s{pointer_offset}, s{pointer_offset + 1}"
        )
        asm.inst(f"s_add_u32 s{weights}, s{weights}, s{route_offset}")
        asm.inst(f"s_addc_u32 s{weights + 1}, s{weights + 1}, s{pointer_offset}")

    def _emit_row_setup(self, asm: Assembly) -> None:
        state = self.context.state
        physical = self._physical_plan()
        vector = physical.vector_registers
        scalar = physical.scalar_registers
        asm.comment("Build aggregate-row, packed-weight, and Q8_1 addresses.")
        asm.inst(
            f"v_and_b32 v{vector.activation_row.first_register}, 15, "
            f"v{vector.serial.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{vector.activation_row.first_register}, "
            f"s{scalar.row_start.first_register}, "
            f"v{vector.activation_row.first_register}"
        )
        asm.inst(
            f"v_lshrrev_b32 v{vector.temporary.first_register}, 4, "
            f"v{vector.serial.first_register}"
        )
        asm.inst(
            f"v_and_b32 v{vector.temporary.first_register}, 1, "
            f"v{vector.temporary.first_register}"
        )
        asm.inst(
            f"v_lshlrev_b32 v{vector.scale.first_register}, 4, "
            f"s{scalar.workgroup_tile.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{vector.temporary.first_register}, "
            f"v{vector.scale.first_register}, v{vector.temporary.first_register}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{vector.result_weight_addresses.first_register}, "
            f"{state.packed_weight_row_bytes}, v{vector.temporary.first_register}"
        )
        for element in range(1, 8):
            asm.inst(
                f"v_add_nc_u32 v{vector.result_weight_addresses.first_register + element}, "
                f"{2 * element * state.packed_weight_row_bytes}, "
                f"v{vector.result_weight_addresses.first_register}"
            )
        asm.inst(
            f"v_mul_lo_u32 v{vector.weight_address.first_register}, "
            f"{state.packed_weight_row_bytes}, "
            f"v{vector.output_column.first_register}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{vector.activation_addresses.first_register}, "
            f"{self.context.solution_key.solution.activation_block_bytes}, "
            f"v{vector.activation_row.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{vector.activation_addresses.first_register + 1}, "
            f"s{scalar.activation_plane_stride.first_register}, "
            f"v{vector.activation_addresses.first_register}"
        )

    def _emit_group(self, asm: Assembly, group: int) -> None:
        state = self.context.state
        physical = self._physical_plan()
        vector = physical.vector_registers
        scalar = physical.scalar_registers
        activation_group = physical.activation_metadata.group(group)
        weight_q_offset, weight_q_offset_high = (
            state.semantics.low_payload_group_offsets(group)
        )
        activation_address = (
            vector.activation_addresses.first_register + activation_group.plane
        )

        asm.comment(f"Q4_K group {group}: direct nibbles and masked aggregate rows.")
        asm.inst(
            f"global_load_b128 v[{vector.weight_payload.first_register}:"
            f"{vector.weight_payload.first_register + 3}], "
            f"v{vector.weight_address.first_register}, "
            f"s[{scalar.weights.first_register}:{scalar.weights.first_register + 1}] "
            f"offset:{weight_q_offset}"
        )
        asm.inst(
            f"global_load_b128 v[{vector.weight_payload.first_register + 4}:"
            f"{vector.weight_payload.first_register + 7}], "
            f"v{vector.weight_address.first_register}, "
            f"s[{scalar.weights.first_register}:{scalar.weights.first_register + 1}] "
            f"offset:{weight_q_offset_high}"
        )
        for register in vector.activation_payload.registers:
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"v_mov_b32 v{vector.activation_scale_sum.first_register}, 0")
        asm.inst(
            f"v_cmp_gt_u32_e32 vcc_lo, s{scalar.row_end.first_register}, "
            f"v{vector.activation_row.first_register}"
        )
        asm.inst(f"s_and_saveexec_b32 s{scalar.exec_mask.first_register}, vcc_lo")
        asm.inst(
            f"global_load_b128 v[{vector.activation_payload.first_register}:"
            f"{vector.activation_payload.first_register + 3}], "
            f"v{activation_address}, "
            f"s[{scalar.activations.first_register}:"
            f"{scalar.activations.first_register + 1}] "
            f"offset:{activation_group.payload_offset}"
        )
        asm.inst(
            f"global_load_b128 v[{vector.activation_payload.first_register + 4}:"
            f"{vector.activation_payload.first_register + 7}], "
            f"v{activation_address}, "
            f"s[{scalar.activations.first_register}:"
            f"{scalar.activations.first_register + 1}] "
            f"offset:{activation_group.payload_high_offset}"
        )
        asm.inst(
            f"global_load_b32 v{vector.activation_scale_sum.first_register}, "
            f"v{activation_address}, "
            f"s[{scalar.activations.first_register}:"
            f"{scalar.activations.first_register + 1}] "
            f"offset:{activation_group.scale_sum_offset}"
        )
        asm.inst(f"s_mov_b32 exec_lo, s{scalar.exec_mask.first_register}")
        asm.inst("s_waitcnt vmcnt(0)")

        for register in vector.weight_payload.registers:
            if group & 1:
                asm.inst(f"v_lshrrev_b32 v{register}, 4, v{register}")
            asm.inst(f"v_and_b32 v{register}, 0x0f0f0f0f, v{register}")
        for register in vector.c.registers:
            asm.inst(f"v_mov_b32 v{register}, 0")
        emit_signed_i8_wmma(
            asm,
            destination=vector.c.first_register,
            weight=vector.weight_payload.first_register,
            activation=vector.activation_payload.first_register,
            accumulator=vector.c.first_register,
            clamp=self.context.solution_key.solution.wmma_clamp,
        )
        emit_signed_i8_wmma(
            asm,
            destination=vector.c.first_register,
            weight=vector.weight_payload.first_register + 4,
            activation=vector.activation_payload.first_register + 4,
            accumulator=vector.c.first_register,
            clamp=self.context.solution_key.solution.wmma_clamp,
        )
        self._emit_scaled_accumulate(asm, group)

    def _emit_scaled_accumulate(self, asm: Assembly, group: int) -> None:
        state = self.context.state
        vector = self._physical_plan().vector_registers
        asm.comment("Reproduce Q4_K FP16 scale/min construction before FP32 sums.")
        asm.inst(
            f"v_cvt_f32_f16 v{vector.activation_d.first_register}, "
            f"v{vector.activation_scale_sum.first_register}"
        )
        asm.inst(
            f"v_cvt_f32_f16 v{vector.activation_sum.first_register}, "
            f"v{vector.activation_scale_sum.first_register}.h"
        )
        for element in range(8):
            metadata = vector.weight_metadata.first_register + 4 * element
            emit_packed_scale_minimum(
                asm,
                group,
                metadata,
                semantics=state.semantics,
                scale=vector.scale.first_register,
                minimum=vector.minimum.first_register,
                temporary=vector.temporary.first_register,
            )
            asm.inst(
                f"v_cvt_f32_u32 v{vector.weight_d.first_register}, "
                f"v{vector.scale.first_register}"
            )
            asm.inst(
                f"v_cvt_f32_u32 v{vector.weight_minimum.first_register}, "
                f"v{vector.minimum.first_register}"
            )
            asm.inst(
                f"v_cvt_f16_f32_e32 v{vector.scaled_dm.first_register}.l, "
                f"v{vector.weight_d.first_register}"
            )
            asm.inst(
                f"v_cvt_f16_f32_e32 v{vector.scaled_dm.first_register}.h, "
                f"v{vector.weight_minimum.first_register}"
            )
            asm.inst(
                f"v_pk_mul_f16 v{vector.scaled_dm.first_register}, 0xbc003c00, "
                f"v{vector.scaled_dm.first_register}"
            )
            asm.inst(
                f"v_pk_mul_f16 v{vector.scaled_dm.first_register}, v{metadata}, "
                f"v{vector.scaled_dm.first_register}"
            )
            asm.inst(
                f"v_cvt_f32_f16 v{vector.weight_d.first_register}, "
                f"v{vector.scaled_dm.first_register}"
            )
            asm.inst(
                f"v_cvt_f32_f16 v{vector.weight_minimum.first_register}, "
                f"v{vector.scaled_dm.first_register}.h"
            )
            c = vector.c.first_register + element
            total = vector.sums.first_register + element
            asm.inst(f"v_cvt_f32_i32 v{vector.temporary.first_register}, v{c}")
            asm.inst(
                f"v_mul_f32 v{vector.temporary.first_register}, "
                f"v{vector.weight_d.first_register}, "
                f"v{vector.temporary.first_register}"
            )
            asm.inst(
                f"v_fma_f32 v{total}, v{vector.temporary.first_register}, "
                f"v{vector.activation_d.first_register}, v{total}"
            )
            asm.inst(
                f"v_fma_f32 v{total}, v{vector.weight_minimum.first_register}, "
                f"v{vector.activation_sum.first_register}, v{total}"
            )

    def _emit_store(self, asm: Assembly) -> None:
        problem = self.context.solution_key.problem
        physical = self._physical_plan()
        vector = physical.vector_registers
        scalar = physical.scalar_registers
        asm.comment("Store valid J-major fragments into aggregate-row BF16 output.")
        asm.inst(
            f"v_mul_lo_u32 v{vector.output_address.first_register}, "
            f"{2 * problem.output_features}, v{vector.activation_row.first_register}"
        )
        asm.inst(
            f"v_lshrrev_b32 v{vector.temporary.first_register}, 4, "
            f"v{vector.serial.first_register}"
        )
        asm.inst(
            f"v_and_b32 v{vector.temporary.first_register}, 1, "
            f"v{vector.temporary.first_register}"
        )
        asm.inst(
            f"v_lshlrev_b32 v{vector.scale.first_register}, 4, "
            f"s{scalar.workgroup_tile.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{vector.temporary.first_register}, "
            f"v{vector.scale.first_register}, v{vector.temporary.first_register}"
        )
        asm.inst(
            f"v_lshlrev_b32 v{vector.temporary.first_register}, 1, "
            f"v{vector.temporary.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{vector.output_address.first_register}, "
            f"v{vector.output_address.first_register}, "
            f"v{vector.temporary.first_register}"
        )
        for element in range(8):
            emit_bf16_rne(
                asm,
                vector.sums.first_register + element,
                vector.temporary.first_register,
            )
        asm.inst(
            f"v_cmp_gt_u32_e32 vcc_lo, s{scalar.row_end.first_register}, "
            f"v{vector.activation_row.first_register}"
        )
        asm.inst(f"s_and_saveexec_b32 s{scalar.exec_mask.first_register}, vcc_lo")
        for element in range(8):
            asm.inst(
                f"global_store_d16_hi_b16 v{vector.output_address.first_register}, "
                f"v{vector.sums.first_register + element}, "
                f"s[{scalar.output.first_register}:{scalar.output.first_register + 1}]"
            )
            if element != 7:
                asm.inst(
                    f"v_add_nc_u32 v{vector.output_address.first_register}, 4, "
                    f"v{vector.output_address.first_register}"
                )
        asm.inst(f"s_mov_b32 exec_lo, s{scalar.exec_mask.first_register}")
