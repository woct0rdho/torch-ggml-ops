"""K128-interleaved paired IQ2_XXS lowering for research artifacts."""

from dataclasses import dataclass

from .grouped_mmq_fwd_lowering import GroupedForwardLoweringResult
from .grouped_mmq_fwd_pair_lowering_common import (
    GroupedForwardPairLoweringContext,
    GroupedPairK128Mechanics,
)
from .grouped_mmq_fwd_pair_model import GroupedPairRouteOwnership
from .grouped_mmq_fwd_pair_physical import (
    GroupedIQ2XXSPairHalfLdsLayout,
    GroupedIQ2XXSPairPhysicalPlan,
)
from .grouped_mmq_fwd_pair_route import (
    GroupedPairRouteEmitter,
    GroupedPairRowTaskEmitter,
)
from .iq2_xxs_grid import iq2_xxs_grid_rodata
from .kernel_writer_assembly import Assembly, emit_bf16_rne, emit_kernel_trailer
from .mmq_fwd_lowering_mma import emit_signed_i8_wmma


@dataclass(frozen=True)
class GroupedIQ2XXSPairedK128Lowering:
    """Emit one J64 tile with explicit IQ2_XXS grid/sign reconstruction."""

    context: GroupedForwardPairLoweringContext

    GRID_SYMBOL = ".LGGTensilePairIQ2XXSGrid"
    GRID_BASE = 42

    def _physical_plan(self) -> GroupedIQ2XXSPairPhysicalPlan:
        physical = self.context.state.physical_plan
        if not isinstance(physical, GroupedIQ2XXSPairPhysicalPlan):
            raise TypeError(
                "IQ2_XXS paired lowering requires its IQ2_XXS physical plan"
            )
        return physical

    def _activation_label_token(self) -> str:
        return "PairIQ2XXS"

    def _mechanics(self) -> GroupedPairK128Mechanics:
        return GroupedPairK128Mechanics(
            self.context,
            self._physical_plan(),
            self._activation_label_token(),
        )

    def emission(self) -> GroupedForwardLoweringResult:
        return GroupedForwardLoweringResult(
            self.body(), (iq2_xxs_grid_rodata(self.GRID_SYMBOL),)
        )

    def body(self) -> str:
        physical = self._physical_plan()
        mechanics = self._mechanics()
        layout = physical.layout
        registers = physical.registers
        scalar = physical.scalar_registers
        state = self.context.state
        asm = Assembly()
        name = self.context.solution_key.kernel_name

        if (
            state.kernel_spec.route_ownership
            is GroupedPairRouteOwnership.DeviceRowTasks64
        ):
            GroupedPairRowTaskEmitter(scalar, state.route, "IQ2_XXS", "IQ2XXS").emit(
                asm
            )
        else:
            GroupedPairRouteEmitter(scalar, state.route, "IQ2_XXS", "IQ2XXS").emit(asm)

        asm.comment("Materialize the paired local IQ2_XXS codebook address.")
        asm.inst(f"s_getpc_b64 s[{self.GRID_BASE}:{self.GRID_BASE + 1}]")
        asm.inst(
            f"s_add_u32 s{self.GRID_BASE}, s{self.GRID_BASE}, "
            f"{self.GRID_SYMBOL}@rel32@lo+4"
        )
        asm.inst(
            f"s_addc_u32 s{self.GRID_BASE + 1}, s{self.GRID_BASE + 1}, "
            f"{self.GRID_SYMBOL}@rel32@hi+12"
        )
        asm.inst(
            f"s_mul_i32 s{scalar.activation_plane_stride.first_register}, "
            f"s{scalar.nrows_activation.first_register}, "
            f"{self.context.solution_key.solution.activation_block_bytes}"
        )
        asm.inst(
            f"s_mov_b32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_begin.first_register}"
        )
        mechanics.emit_invariant_addresses(asm, layout)

        asm.label(".LGroupedPairIQ2XXSRowLoop")
        asm.inst(
            f"s_sub_u32 s{scalar.row_tile_rows.first_register}, "
            f"s{scalar.row_end.first_register}, s{scalar.row_start.first_register}"
        )
        asm.inst(f"s_cmp_le_u32 s{scalar.row_tile_rows.first_register}, 64")
        asm.inst("s_cbranch_scc1 .LGroupedPairIQ2XXSRowsReady")
        asm.inst(f"s_mov_b32 s{scalar.row_tile_rows.first_register}, 64")
        asm.label(".LGroupedPairIQ2XXSRowsReady")
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

        asm.label(".LGroupedPairIQ2XXSBlockLoop")
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
            mechanics.emit_activation_stage(asm, layout, stage_index=half)
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
            f"v_add_nc_u32 v{registers.weight_address.first_register}, 66, "
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
        asm.inst("s_cbranch_scc1 .LGroupedPairIQ2XXSBlockLoop")
        asm.inst(
            f"v_sub_nc_u32 v{registers.weight_address.first_register}, "
            f"v{registers.weight_address.first_register}, "
            f"{state.packed_weight_row_bytes}"
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
        asm.inst("s_cbranch_scc1 .LGroupedPairIQ2XXSRowLoop")
        asm.label(".LGroupedPairIQ2XXSExit")
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_weight_half_decode(
        self,
        asm: Assembly,
        layout: GroupedIQ2XXSPairHalfLdsLayout,
        weight_scalar: int,
        half: int,
    ) -> None:
        if half not in (0, 1):
            raise ValueError("IQ2_XXS paired decode half must be zero or one")
        registers = self._physical_plan().registers
        d = registers.producer_d.first_register
        packed = registers.producer_indices.first_register
        part = registers.decode_auxiliary.first_register
        address = registers.producer_address.first_register
        lds_address = registers.producer_lds_address.first_register
        decoded = registers.decoded_payload.first_register

        asm.comment(
            f"Decode paired IQ2_XXS selected K128 half {half} in two lane groups."
        )
        asm.inst(
            f"v_mov_b32 v{lds_address}, v{registers.weight_lds_address.first_register}"
        )
        asm.inst(f"v_bfe_u32 v{part}, v{registers.lane.first_register}, 4, 1")
        asm.inst(f"v_lshlrev_b32 v{address}, 4, v{part}")
        asm.inst(
            f"v_add_nc_u32 v{address}, v{registers.weight_address.first_register}, "
            f"v{address}"
        )
        asm.inst(
            f"global_load_ushort v{d}, "
            f"v{registers.weight_address.first_register}, "
            f"s[{weight_scalar}:{weight_scalar + 1}] offset:0"
        )
        for local in range(2):
            asm.inst(
                f"global_load_b64 v[{packed + 2 * local}:{packed + 2 * local + 1}], "
                f"v{address}, s[{weight_scalar}:{weight_scalar + 1}] "
                f"offset:{2 + 32 * half + 8 * local}"
            )
        asm.inst("s_waitcnt vmcnt(0)")
        asm.inst(f"v_cvt_f32_f16 v{d}, v{d}.l")

        # Each 32-value group contains four 8-value codebook entries.
        for local in range(2):
            index_register = packed + 2 * local
            sign_register = index_register + 1
            for subgroup in range(4):
                index = part + 1
                asm.inst(f"v_lshrrev_b32 v{index}, {8 * subgroup}, v{index_register}")
                asm.inst(f"v_and_b32 v{index}, 0xff, v{index}")
                asm.inst(f"v_lshlrev_b32 v{address}, 3, v{index}")
                grid = registers.codebook_payload.first_register + 2 * subgroup
                asm.inst(
                    f"global_load_b64 v[{grid}:{grid + 1}], v{address}, "
                    f"s[{self.GRID_BASE}:{self.GRID_BASE + 1}]"
                )
            asm.inst("s_waitcnt vmcnt(0)")

            # Lane bit four assigns the two loaded groups to the lower/upper pair.
            asm.inst(f"v_lshlrev_b32 v{address}, 1, v{part}")
            if local:
                asm.inst(f"v_add_nc_u32 v{address}, {local}, v{address}")
            asm.inst(f"v_lshlrev_b32 v{address}, 5, v{address}")
            asm.inst(f"v_add_nc_u32 v{address}, v{lds_address}, v{address}")

            # Reconstruct the four signed grid entries in two 128-bit stores.
            for start in (0, 2):
                for subgroup in range(start, start + 2):
                    sign_byte = part + 1
                    asm.inst(
                        f"v_bfe_u32 v{sign_byte}, v{sign_register}, {7 * subgroup}, 7"
                    )
                    parity = part + 4
                    asm.inst(f"v_bcnt_u32_b32 v{parity}, v{sign_byte}, 0")
                    asm.inst(f"v_and_b32 v{parity}, 1, v{parity}")
                    asm.inst(f"v_lshlrev_b32 v{parity}, 7, v{parity}")
                    asm.inst(f"v_xor_b32 v{sign_byte}, v{sign_byte}, v{parity}")
                    grid = registers.codebook_payload.first_register + 2 * subgroup
                    self._emit_signed_grid_dword(
                        asm, grid, sign_byte, 0, decoded + 2 * (subgroup - start)
                    )
                    self._emit_signed_grid_dword(
                        asm,
                        grid + 1,
                        sign_byte,
                        4,
                        decoded + 2 * (subgroup - start) + 1,
                    )
                asm.inst(
                    f"ds_write_b128 v{address}, v[{decoded}:{decoded + 3}] "
                    f"offset:{16 * (start // 2)}"
                )

            # The sign word also carries the odd (2*scale+1) group scale.
            scale = part + 3
            asm.inst(f"v_lshrrev_b32 v{scale}, 27, v{sign_register}")
            asm.inst(f"v_or_b32 v{scale}, 1, v{scale}")
            asm.inst(f"v_cvt_f32_u32 v{scale}, v{scale}")
            asm.inst(f"v_mul_f32 v{scale}, v{d}, v{scale}")
            asm.inst(f"v_mul_f32 v{scale}, 0.125, v{scale}")
            asm.inst(f"v_lshlrev_b32 v{address}, 1, v{part}")
            if local:
                asm.inst(f"v_add_nc_u32 v{address}, {local}, v{address}")
            asm.inst(f"v_lshlrev_b32 v{address}, 3, v{address}")
            asm.inst(f"v_add_nc_u32 v{address}, v{lds_address}, v{address}")
            asm.inst(
                f"ds_write_b32 v{address}, v{scale} offset:{layout.weight_scale_offset}"
            )
            asm.inst(
                f"ds_write_b32 v{address}, v{scale} "
                f"offset:{layout.weight_scale_offset + 4}"
            )

    def _emit_signed_grid_dword(
        self,
        asm: Assembly,
        positive: int,
        sign_byte: int,
        sign_shift: int,
        destination: int,
    ) -> None:
        registers = self._physical_plan().registers
        auxiliary = registers.decode_auxiliary.first_register
        sign_bits = auxiliary + 2
        selector = auxiliary + 3
        negative = auxiliary + 4
        asm.inst(f"v_bfe_u32 v{sign_bits}, v{sign_byte}, {sign_shift}, 4")
        asm.inst(f"v_mul_lo_u32 v{selector}, 0x204081, v{sign_bits}")
        asm.inst(f"v_and_b32 v{selector}, 0x01010101, v{selector}")
        asm.inst(f"v_lshl_or_b32 v{selector}, v{selector}, 2, 0x03020100")
        asm.inst(f"v_not_b32 v{negative}, v{positive}")
        asm.inst(f"v_add_nc_u32 v{negative}, 0x01010101, v{negative}")
        asm.inst(f"v_perm_b32 v{destination}, v{negative}, v{positive}, v{selector}")

    def _emit_compute_projection(self, asm: Assembly, sums: int) -> None:
        layout = self._physical_plan().layout
        mechanics = self._mechanics()
        for group in range(0, 8, 2):
            mechanics.emit_compute_group_payload_reads(asm, layout, group)
            mechanics.emit_compute_group_scale_reads(asm, group)
            asm.inst("s_waitcnt lgkmcnt(0)")
            self._emit_compute_group_wmmas_accumulate(asm, accumulate=False)
            mechanics.emit_compute_group_payload_reads(asm, layout, group + 1)
            asm.inst("s_waitcnt lgkmcnt(0)")
            self._emit_compute_group_wmmas_accumulate(asm, accumulate=True)
            mechanics.emit_fragment_correction(asm, 0, 3, sums)
            mechanics.emit_fragment_correction(asm, 1, 2, sums)

    def _emit_compute_group_wmmas_accumulate(
        self, asm: Assembly, *, accumulate: bool
    ) -> None:
        registers = self._physical_plan().registers
        for m_index in range(4):
            destination = registers.c.first_register + 8 * m_index
            emit_signed_i8_wmma(
                asm,
                destination=destination,
                weight=registers.weight_payload.first_register,
                activation=registers.activation_payload.first_register + 4 * m_index,
                accumulator=(
                    destination
                    if accumulate
                    else registers.zero_accumulator.first_register
                ),
                clamp=False,
            )

    def _emit_store_projection(
        self, asm: Assembly, sums: int, output: int, projection: int
    ) -> None:
        registers = self._physical_plan().registers
        scalar = self._physical_plan().scalar_registers
        temporary = registers.temporary.first_register
        column = registers.decode_auxiliary.first_register
        asm.comment(f"Store paired IQ2_XXS projection {projection} J64 BF16 fragments.")
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
                f"v_cmp_lt_u32 vcc_lo, v{temporary}, "
                f"s{scalar.row_tile_rows.first_register}"
            )
            asm.inst(f"s_and_saveexec_b32 s{scalar.exec_mask.first_register}, vcc_lo")
            asm.inst(
                f"v_add_nc_u32 v{temporary}, s{scalar.row_start.first_register}, "
                f"v{temporary}"
            )
            asm.inst(
                f"v_lshlrev_b32 v{registers.output_address.first_register}, 12, "
                f"v{temporary}"
            )
            asm.inst(
                f"v_add_nc_u32 v{registers.output_address.first_register}, v{column}, "
                f"v{registers.output_address.first_register}"
            )
            asm.inst("s_clause 7")
            for element in range(8):
                asm.inst(
                    f"global_store_d16_hi_b16 v{registers.output_address.first_register}, "
                    f"v{fragment + element}, s[{output}:{output + 1}] "
                    f"offset:{4 * element}"
                )
            asm.inst(f"s_mov_b32 exec_lo, s{scalar.exec_mask.first_register}")
