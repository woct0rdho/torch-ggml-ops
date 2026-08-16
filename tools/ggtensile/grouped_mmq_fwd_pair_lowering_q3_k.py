"""K128-interleaved paired Q3_K lowering for research artifacts."""

from dataclasses import dataclass
from typing import Any, cast

from .grouped_mmq_fwd_lowering import GroupedForwardLoweringResult
from .grouped_mmq_fwd_pair_lowering_iq2_s import (
    GroupedForwardPairLoweringContext,
    GroupedIQ2SPairedK128Lowering,
)
from .grouped_mmq_fwd_pair_model import GroupedPairRouteOwnership
from .grouped_mmq_fwd_pair_physical import (
    GroupedQ3KPairHalfLdsLayout,
    GroupedQ3KPairPhysicalPlan,
)
from .grouped_mmq_fwd_pair_route import (
    GroupedPairRouteEmitter,
    GroupedPairRowTaskEmitter,
)
from .kernel_writer_assembly import Assembly, emit_bf16_rne, emit_kernel_trailer


@dataclass(frozen=True)
class GroupedQ3KPairedK128Lowering:
    """Emit one J64 tile that interleaves both Q3_K projections at K128."""

    context: GroupedForwardPairLoweringContext

    def _physical_plan(self) -> GroupedQ3KPairPhysicalPlan:
        physical = self.context.state.physical_plan
        if not isinstance(physical, GroupedQ3KPairPhysicalPlan):
            raise TypeError("Q3_K paired lowering requires its Q3_K physical plan")
        return physical

    def _activation_label_token(self) -> str:
        return "PairQ3K"

    def emission(self) -> GroupedForwardLoweringResult:
        return GroupedForwardLoweringResult(self.body())

    def _emit_invariant_addresses(
        self, asm: Assembly, layout: GroupedQ3KPairHalfLdsLayout
    ) -> None:
        cast(Any, GroupedIQ2SPairedK128Lowering._emit_invariant_addresses)(
            self, asm, layout
        )

    def _emit_activation_stage(
        self,
        asm: Assembly,
        layout: GroupedQ3KPairHalfLdsLayout,
        *,
        stage_index: int,
    ) -> None:
        cast(Any, GroupedIQ2SPairedK128Lowering._emit_activation_stage)(
            self, asm, layout, stage_index=stage_index
        )

    def _emit_linear_activation_stage(
        self,
        asm: Assembly,
        layout: GroupedQ3KPairHalfLdsLayout,
        *,
        stage_index: int,
    ) -> None:
        cast(Any, GroupedIQ2SPairedK128Lowering._emit_linear_activation_stage)(
            self, asm, layout, stage_index=stage_index
        )

    def _emit_compute_projection(self, asm: Assembly, sums: int) -> None:
        cast(Any, GroupedIQ2SPairedK128Lowering._emit_compute_projection)(
            self, asm, sums
        )

    def _emit_compute_group_payload_reads(
        self, asm: Assembly, layout: GroupedQ3KPairHalfLdsLayout, group: int
    ) -> None:
        cast(Any, GroupedIQ2SPairedK128Lowering._emit_compute_group_payload_reads)(
            self, asm, layout, group
        )

    def _emit_compute_group_scale_reads(self, asm: Assembly, group: int) -> None:
        cast(Any, GroupedIQ2SPairedK128Lowering._emit_compute_group_scale_reads)(
            self, asm, group
        )

    def _emit_compute_group_wmmas(self, asm: Assembly) -> None:
        cast(Any, GroupedIQ2SPairedK128Lowering._emit_compute_group_wmmas)(self, asm)

    def _emit_fragment_correction(
        self, asm: Assembly, first: int, second: int, sums: int
    ) -> None:
        cast(Any, GroupedIQ2SPairedK128Lowering._emit_fragment_correction)(
            self, asm, first, second, sums
        )

    def body(self) -> str:
        physical = self._physical_plan()
        layout = physical.layout
        registers = physical.registers
        scalar = physical.scalar_registers
        state = self.context.state
        asm = Assembly()
        name = self.context.solution_key.kernel_name

        if state.contract.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks64:
            GroupedPairRowTaskEmitter(scalar, state.route, "Q3_K", "Q3K").emit(asm)
        else:
            GroupedPairRouteEmitter(scalar, state.route, "Q3_K", "Q3K").emit(asm)
        asm.inst(
            f"s_mul_i32 s{scalar.activation_plane_stride.first_register}, "
            f"s{scalar.nrows_activation.first_register}, "
            f"{self.context.solution_key.solution.activation_block_bytes}"
        )
        asm.inst(
            f"s_mov_b32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_begin.first_register}"
        )
        self._emit_invariant_addresses(asm, layout)

        asm.label(".LGroupedPairQ3KRowLoop")
        asm.inst(
            f"s_sub_u32 s{scalar.row_tile_rows.first_register}, "
            f"s{scalar.row_end.first_register}, s{scalar.row_start.first_register}"
        )
        asm.inst(f"s_cmp_le_u32 s{scalar.row_tile_rows.first_register}, 64")
        asm.inst("s_cbranch_scc1 .LGroupedPairQ3KRowsReady")
        asm.inst(f"s_mov_b32 s{scalar.row_tile_rows.first_register}, 64")
        asm.label(".LGroupedPairQ3KRowsReady")
        asm.inst(
            f"s_add_u32 s{scalar.row_tile_end.first_register}, "
            f"s{scalar.row_start.first_register}, s{scalar.row_tile_rows.first_register}"
        )
        asm.inst(
            f"s_mul_i32 s{scalar.packed_block_offset.first_register}, "
            f"s{scalar.row_start.first_register}, "
            f"{self.context.solution_key.solution.activation_block_bytes}"
        )
        for register in (
            *registers.sums_first.registers,
            *registers.sums_second.registers,
        ):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"s_mov_b32 s{scalar.loop_counter.first_register}, 0")

        asm.label(".LGroupedPairQ3KBlockLoop")
        for half in (0, 1):
            if half:
                asm.inst(
                    f"s_add_u32 s{scalar.packed_block_offset.first_register}, "
                    f"s{scalar.packed_block_offset.first_register}, "
                    f"s{scalar.activation_plane_stride.first_register}"
                )
            self._emit_weight_half_decode(
                asm, layout, scalar.weights_first.first_register, half
            )
            self._emit_activation_stage(asm, layout, stage_index=half)
            asm.inst("s_waitcnt lgkmcnt(0)")
            asm.inst("s_barrier")
            for register in registers.zero_accumulator.registers:
                asm.inst(f"v_mov_b32 v{register}, 0")
            self._emit_compute_projection(asm, registers.sums_first.first_register)
            asm.inst("s_barrier")

            self._emit_weight_half_decode(
                asm, layout, scalar.weights_second.first_register, half
            )
            asm.inst("s_waitcnt lgkmcnt(0)")
            asm.inst("s_barrier")
            for register in registers.zero_accumulator.registers:
                asm.inst(f"v_mov_b32 v{register}, 0")
            self._emit_compute_projection(asm, registers.sums_second.first_register)
            asm.inst("s_barrier")

        asm.inst(
            f"s_add_u32 s{scalar.packed_block_offset.first_register}, "
            f"s{scalar.packed_block_offset.first_register}, "
            f"s{scalar.activation_plane_stride.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.weight_address.first_register}, 110, "
            f"v{registers.weight_address.first_register}"
        )
        asm.inst(
            f"s_add_u32 s{scalar.loop_counter.first_register}, "
            f"s{scalar.loop_counter.first_register}, 1"
        )
        asm.inst(
            f"s_cmp_lt_u32 s{scalar.loop_counter.first_register}, "
            f"{state.blocks_per_weight_row}"
        )
        asm.inst("s_cbranch_scc1 .LGroupedPairQ3KBlockLoop")
        asm.inst(
            f"v_sub_nc_u32 v{registers.weight_address.first_register}, "
            f"v{registers.weight_address.first_register}, {state.packed_weight_row_bytes}"
        )

        self._emit_store_projection(
            asm,
            registers.sums_first.first_register,
            scalar.output_first.first_register,
            0,
        )
        self._emit_store_projection(
            asm,
            registers.sums_second.first_register,
            scalar.output_second.first_register,
            1,
        )
        asm.inst(
            f"s_add_u32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_start.first_register}, 64"
        )
        asm.inst(
            f"s_cmp_lt_u32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_end.first_register}"
        )
        asm.inst("s_cbranch_scc1 .LGroupedPairQ3KRowLoop")
        asm.label(".LGroupedPairQ3KExit")
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_weight_half_decode(
        self,
        asm: Assembly,
        layout: GroupedQ3KPairHalfLdsLayout,
        weight_scalar: int,
        half: int,
    ) -> None:
        if half not in (0, 1):
            raise ValueError("Q3_K paired decode half must be zero or one")
        registers = self._physical_plan().registers
        semantics = self.context.state.semantics
        part = registers.decode_auxiliary.first_register
        shift = part + 1
        address = registers.producer_address.first_register
        lds_address = registers.producer_lds_address.first_register
        low_base = registers.producer_low_raw.first_register
        high_base = registers.producer_high_raw.first_register
        metadata = registers.producer_metadata.first_register
        decoded = registers.decoded_payload.first_register

        asm.comment(f"Decode paired Q3_K selected K128 half {half} in two lane groups.")
        asm.inst(f"v_bfe_u32 v{part}, v{registers.lane.first_register}, 4, 1")
        asm.inst(f"v_lshlrev_b32 v{address}, 4, v{part}")
        asm.inst(
            f"v_add_nc_u32 v{address}, v{registers.weight_address.first_register}, v{address}"
        )
        asm.inst(
            f"global_load_b128 v[{low_base}:{low_base + 3}], v{address}, "
            f"s[{weight_scalar}:{weight_scalar + 1}] offset:{32 + 32 * half}"
        )
        asm.inst(
            f"global_load_b128 v[{high_base}:{high_base + 3}], v{address}, "
            f"s[{weight_scalar}:{weight_scalar + 1}] offset:0"
        )
        asm.inst(
            f"global_load_b128 v[{metadata}:{metadata + 3}], "
            f"v{registers.weight_address.first_register}, "
            f"s[{weight_scalar}:{weight_scalar + 1}] offset:94"
        )
        asm.inst("s_waitcnt vmcnt(0)")
        asm.inst(f"v_lshlrev_b32 v{address}, 4, v{part}")
        asm.inst(
            f"v_add_nc_u32 v{lds_address}, v{registers.weight_lds_address.first_register}, v{address}"
        )
        for local_group in range(4):
            logical_group = 4 * half + local_group
            low_shift = 2 * local_group
            high_shift = (logical_group + 6) % 8
            for item in range(4):
                destination = decoded + item
                asm.inst(
                    f"v_lshrrev_b32 v{destination}, {low_shift}, v{low_base + item}"
                )
                asm.inst(f"v_and_b32 v{destination}, 0x03030303, v{destination}")
                if high_shift >= 6:
                    asm.inst(
                        f"v_lshlrev_b32 v{shift}, {8 - high_shift}, v{high_base + item}"
                    )
                else:
                    asm.inst(
                        f"v_lshrrev_b32 v{shift}, {high_shift}, v{high_base + item}"
                    )
                asm.inst(
                    f"v_and_or_b32 v{destination}, v{shift}, 0x04040404, v{destination}"
                )
                asm.inst(f"v_add_nc_u32 v{destination}, 0x7c7c7c7c, v{destination}")
                asm.inst(f"v_xor_b32 v{destination}, 0x80808080, v{destination}")
            asm.inst(
                f"ds_write_b128 v{lds_address}, v[{decoded}:{decoded + 3}] "
                f"offset:{32 * local_group}"
            )

        asm.inst(
            f"v_cvt_f32_f16 v{registers.decode_d.first_register}, v{metadata + 3}.h"
        )
        asm.inst(f"v_lshlrev_b32 v{address}, 2, v{part}")
        asm.inst(
            f"v_add_nc_u32 v{lds_address}, v{registers.weight_lds_address.first_register}, v{address}"
        )
        asm.inst(f"v_lshlrev_b32 v{shift}, 3, v{part}")
        metadata_load_offset = semantics.payload_plane("scales").byte_offset - 2
        scale_plane_offset = semantics.payload_plane("scales").byte_offset
        for local_group in range(4):
            group = 8 * half + 2 * local_group
            low_field, high_field = semantics.q3_scale_fields(group)
            low_byte = scale_plane_offset - metadata_load_offset + low_field.source_byte
            high_byte = (
                scale_plane_offset - metadata_load_offset + high_field.source_byte
            )
            low_word = low_byte // 4
            high_word = high_byte // 4
            low_bit = 8 * (low_byte % 4) + low_field.bit_offset
            high_bit = 8 * (high_byte % 4) + high_field.bit_offset
            asm.inst(f"v_add_nc_u32 v{address}, {low_bit}, v{shift}")
            asm.inst(
                f"v_lshrrev_b32 v{registers.decode_scale.first_register}, v{address}, "
                f"v{metadata + low_word}"
            )
            asm.inst(
                f"v_and_b32 v{registers.decode_scale.first_register}, "
                f"{(1 << low_field.bit_count) - 1}, v{registers.decode_scale.first_register}"
            )
            asm.inst(f"v_add_nc_u32 v{address}, {high_bit}, v{shift}")
            asm.inst(f"v_lshrrev_b32 v{address}, v{address}, v{metadata + high_word}")
            asm.inst(
                f"v_and_b32 v{address}, {(1 << high_field.bit_count) - 1}, v{address}"
            )
            asm.inst(
                f"v_lshl_or_b32 v{registers.decode_scale.first_register}, v{address}, "
                f"{high_field.destination_shift}, v{registers.decode_scale.first_register}"
            )
            asm.inst(
                f"v_sub_nc_u32 v{registers.decode_scale.first_register}, "
                f"v{registers.decode_scale.first_register}, 32"
            )
            asm.inst(
                f"v_cvt_f32_i32 v{registers.decode_scale.first_register}, "
                f"v{registers.decode_scale.first_register}"
            )
            asm.inst(
                f"v_mul_f32 v{registers.decode_scale.first_register}, "
                f"v{registers.decode_d.first_register}, v{registers.decode_scale.first_register}"
            )
            asm.inst(
                f"ds_write_b32 v{lds_address}, v{registers.decode_scale.first_register} "
                f"offset:{layout.weight_scale_offset + 8 * local_group}"
            )

    def _emit_store_projection(
        self, asm: Assembly, sums: int, output: int, projection: int
    ) -> None:
        registers = self._physical_plan().registers
        scalar = self._physical_plan().scalar_registers
        temporary = registers.temporary.first_register
        column = registers.decode_auxiliary.first_register
        asm.comment(f"Store paired Q3_K projection {projection} J64 BF16 fragments.")
        asm.inst(f"v_bfe_u32 v{column}, v{registers.lane.first_register}, 4, 1")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 4, v{registers.wave.first_register}")
        asm.inst(f"v_add_nc_u32 v{column}, v{temporary}, v{column}")
        asm.inst(
            f"v_lshlrev_b32 v{temporary}, 6, s{scalar.workgroup_tile.first_register}"
        )
        asm.inst(f"v_add_nc_u32 v{column}, v{temporary}, v{column}")
        asm.inst(f"v_lshlrev_b32 v{column}, 1, v{column}")
        for m_index in range(4):
            fragment = sums + 8 * m_index
            for element in range(8):
                emit_bf16_rne(asm, fragment + element, temporary + 1)
            asm.inst(f"v_and_b32 v{temporary}, 15, v{registers.lane.first_register}")
            if m_index:
                asm.inst(f"v_add_nc_u32 v{temporary}, {16 * m_index}, v{temporary}")
            asm.inst(
                f"v_cmp_lt_u32 vcc_lo, v{temporary}, s{scalar.row_tile_rows.first_register}"
            )
            asm.inst(f"s_and_saveexec_b32 s{scalar.exec_mask.first_register}, vcc_lo")
            asm.inst(
                f"v_add_nc_u32 v{temporary}, s{scalar.row_start.first_register}, v{temporary}"
            )
            asm.inst(
                f"v_lshlrev_b32 v{registers.output_address.first_register}, 10, v{temporary}"
            )
            asm.inst(
                f"v_add_nc_u32 v{registers.output_address.first_register}, v{column}, "
                f"v{registers.output_address.first_register}"
            )
            asm.inst("s_clause 7")
            for element in range(8):
                asm.inst(
                    f"global_store_d16_hi_b16 v{registers.output_address.first_register}, "
                    f"v{fragment + element}, s[{output}:{output + 1}] offset:{4 * element}"
                )
            asm.inst(f"s_mov_b32 exec_lo, s{scalar.exec_mask.first_register}")
