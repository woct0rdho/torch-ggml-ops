"""Distributed full-weight LDS lowering for grouped IQ2_S MMQ forward."""

from dataclasses import dataclass
from typing import ClassVar, cast

from .grouped_mmq_fwd_lowering import (
    GroupedForwardLoweringContext,
)
from .grouped_mmq_fwd_model import (
    GroupedActivationStaging,
    GroupedIQ2SDecodePolicy,
)
from .grouped_mmq_fwd_physical import (
    GroupedIQ2SFullWeightLdsLayout,
    GroupedIQ2SFullWeightPhysicalPlan,
)
from .grouped_mmq_fwd_route import GroupedRouteEmitter
from .iq2_s_grid import iq2_s_grid_rodata
from .kernel_writer_assembly import (
    Assembly,
    LoweringResult,
    emit_bf16_rne,
    emit_kernel_trailer,
)
from .mmq_fwd_lowering_mma import emit_signed_i8_wmma


@dataclass(frozen=True)
class GroupedIQ2SFullWeightLdsLowering:
    """Emit a J64 routed tile with all-wave IQ2_S codebook decode."""

    context: GroupedForwardLoweringContext

    GRID_SYMBOL: ClassVar[str] = ".LGGTensileIQ2SGrid"
    GRID_BASE: ClassVar[int] = 38

    def _physical_plan(self) -> GroupedIQ2SFullWeightPhysicalPlan:
        return cast(GroupedIQ2SFullWeightPhysicalPlan, self.context.state.physical_plan)

    def _uses_payload_prefetch(self) -> bool:
        policy = self.context.state.kernel_spec.decode
        assert isinstance(policy, GroupedIQ2SDecodePolicy)
        return policy.payload_prefetch

    def _activation_label_token(self) -> str:
        return "IQ2S"

    def emission(self) -> LoweringResult:
        return LoweringResult(
            self.body(),
            (iq2_s_grid_rodata(self.GRID_SYMBOL),),
        )

    def body(self) -> str:
        physical = self._physical_plan()
        layout = physical.layout
        registers = physical.registers
        scalar = physical.scalar_registers
        state = self.context.state
        asm = Assembly()
        name = self.context.kernel_name
        GroupedRouteEmitter(scalar, state.route).emit(asm)

        asm.comment(
            "Materialize the local IQ2_S codebook address without an ABI pointer."
        )
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
            f"{state.contract.activation_block_bytes}"
        )
        asm.inst(
            f"s_mov_b32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_begin.first_register}"
        )
        self._emit_invariant_addresses(asm, layout)
        if self._uses_payload_prefetch():
            asm.inst(f"s_mov_b32 s{scalar.group_offset.first_register}, 0x03020100")

        asm.label(".LGroupedIQ2SRowLoop")
        asm.inst(
            f"s_sub_u32 s{scalar.row_tile_rows.first_register}, "
            f"s{scalar.row_end.first_register}, s{scalar.row_start.first_register}"
        )
        asm.inst(f"s_cmp_le_u32 s{scalar.row_tile_rows.first_register}, 64")
        asm.inst("s_cbranch_scc1 .LGroupedIQ2SRowsReady")
        asm.inst(f"s_mov_b32 s{scalar.row_tile_rows.first_register}, 64")
        asm.label(".LGroupedIQ2SRowsReady")
        asm.inst(
            f"s_add_u32 s{scalar.row_tile_end.first_register}, "
            f"s{scalar.row_start.first_register}, s{scalar.row_tile_rows.first_register}"
        )
        asm.inst(
            f"s_mul_i32 s{scalar.packed_block_offset.first_register}, "
            f"s{scalar.row_start.first_register}, "
            f"{state.contract.activation_block_bytes}"
        )
        for register in registers.sums.registers:
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"s_mov_b32 s{scalar.loop_counter.first_register}, 0")

        asm.label(".LGroupedIQ2SBlockLoop")
        self._emit_weight_decode(asm, layout)
        for register in registers.zero_accumulator.registers:
            asm.inst(f"v_mov_b32 v{register}, 0")
        self._emit_activation_stage(asm, layout, stage_index=0)
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        self._emit_compute_half(asm, layout, half=0)
        asm.inst("s_barrier")

        asm.inst(
            f"s_add_u32 s{scalar.packed_block_offset.first_register}, "
            f"s{scalar.packed_block_offset.first_register}, "
            f"s{scalar.activation_plane_stride.first_register}"
        )
        self._emit_activation_stage(asm, layout, stage_index=1)
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        self._emit_compute_half(asm, layout, half=1)
        asm.inst("s_barrier")
        asm.inst(
            f"s_add_u32 s{scalar.packed_block_offset.first_register}, "
            f"s{scalar.packed_block_offset.first_register}, "
            f"s{scalar.activation_plane_stride.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.weight_address.first_register}, "
            f"{state.contract.packed_weight_block_bytes}, "
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
        asm.inst("s_cbranch_scc1 .LGroupedIQ2SBlockLoop")
        asm.inst(
            f"v_sub_nc_u32 v{registers.weight_address.first_register}, "
            f"v{registers.weight_address.first_register}, {state.packed_weight_row_bytes}"
        )

        self._emit_store(asm)
        asm.inst(
            f"s_add_u32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_start.first_register}, 64"
        )
        asm.inst(
            f"s_cmp_lt_u32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_end.first_register}"
        )
        asm.inst("s_cbranch_scc1 .LGroupedIQ2SRowLoop")
        asm.label(".LGroupedQ4KExit")
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_invariant_addresses(
        self,
        asm: Assembly,
        layout: GroupedIQ2SFullWeightLdsLayout,
    ) -> None:
        registers = self._physical_plan().registers
        scalar = self._physical_plan().scalar_registers
        temporary = registers.temporary.first_register
        asm.comment(
            "Map four waves to 64 output columns and split each weight row by K half."
        )
        asm.inst(f"v_and_b32 v{registers.lane.first_register}, 31, v0")
        asm.inst(f"v_lshrrev_b32 v{registers.wave.first_register}, 5, v0")
        asm.inst(f"v_and_b32 v{temporary}, 15, v{registers.lane.first_register}")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 4, v{registers.wave.first_register}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary + 1}, v{temporary}")
        asm.inst(
            f"v_mul_lo_u32 v{registers.weight_lds_address.first_register}, "
            f"{layout.weight_row_stride}, v{temporary}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.weight_lds_address.first_register}, "
            f"{layout.weight_base}, v{registers.weight_lds_address.first_register}"
        )
        asm.inst(
            f"v_lshlrev_b32 v{temporary + 1}, 6, "
            f"s{scalar.workgroup_tile.first_register}"
        )
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary + 1}, v{temporary}")
        asm.inst(
            f"v_mul_lo_u32 v{registers.weight_address.first_register}, "
            f"{self.context.state.packed_weight_row_bytes}, v{temporary}"
        )
        asm.inst(f"v_and_b32 v{temporary}, 15, v{registers.lane.first_register}")
        asm.inst(
            f"v_mul_lo_u32 v{registers.activation_read_address.first_register}, "
            f"{layout.activation_row_stride}, v{temporary}"
        )
        asm.inst(f"v_bfe_u32 v{temporary}, v{registers.lane.first_register}, 4, 1")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 4, v{registers.wave.first_register}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary + 1}, v{temporary}")
        asm.inst(
            f"v_mul_lo_u32 v{registers.weight_scale_address.first_register}, "
            f"{layout.weight_row_stride}, v{temporary}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.weight_scale_address.first_register}, "
            f"{layout.weight_base}, "
            f"v{registers.weight_scale_address.first_register}"
        )
        for index in range(1, 4):
            asm.inst(
                f"v_add_nc_u32 v{registers.weight_scale_address.first_register + index}, "
                f"{index * 4 * layout.weight_row_stride}, "
                f"v{registers.weight_scale_address.first_register}"
            )

    def _emit_weight_decode(
        self,
        asm: Assembly,
        layout: GroupedIQ2SFullWeightLdsLayout,
    ) -> None:
        registers = self._physical_plan().registers
        semantics = self.context.state.semantics
        d_offset = semantics.payload_plane("d").byte_offset
        index_offset = semantics.payload_plane("grid_indices").byte_offset
        sign_offset = semantics.payload_plane("signs").byte_offset
        qh_offset = semantics.payload_plane("qh").byte_offset
        scale_offset = semantics.payload_plane("scales").byte_offset
        optimized_decode = self._uses_payload_prefetch()
        half = registers.decode_auxiliary.first_register
        address = registers.producer_address.first_register
        asm.comment("Cooperatively decode one IQ2_S row half per workitem.")
        asm.inst(f"v_bfe_u32 v{half}, v{registers.lane.first_register}, 4, 1")
        asm.inst(
            f"v_mul_lo_u32 v{registers.producer_lds_address.first_register}, "
            f"{layout.half_payload_stride}, v{half}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.producer_lds_address.first_register}, "
            f"v{registers.weight_lds_address.first_register}, "
            f"v{registers.producer_lds_address.first_register}"
        )
        asm.inst(f"v_lshlrev_b32 v{address}, 2, v{half}")
        asm.inst(
            f"v_add_nc_u32 v{address}, v{registers.weight_address.first_register}, "
            f"v{address}"
        )
        asm.inst(
            f"global_load_ushort v{registers.producer_d.first_register}, "
            f"v{registers.weight_address.first_register}, s[4:5] offset:{d_offset}"
        )
        asm.inst(
            f"global_load_b32 v{registers.producer_qh.first_register}, v{address}, "
            f"s[4:5] offset:{qh_offset}"
        )
        asm.inst(
            f"global_load_b32 v{registers.producer_scales.first_register}, v{address}, "
            f"s[4:5] offset:{scale_offset}"
        )
        asm.inst(f"v_lshlrev_b32 v{address}, 4, v{half}")
        asm.inst(
            f"v_add_nc_u32 v{address}, v{registers.weight_address.first_register}, v{address}"
        )
        for pair in range(8):
            asm.inst(
                f"global_load_ushort v{registers.producer_indices.first_register + pair}, "
                f"v{address}, s[4:5] offset:{index_offset + 2 * pair}"
            )
            asm.inst(
                f"global_load_ushort v{registers.producer_signs.first_register + pair}, "
                f"v{address}, s[4:5] offset:{sign_offset + 2 * pair}"
            )
        asm.inst("s_waitcnt vmcnt(0)")
        asm.inst(
            f"v_cvt_f32_f16 v{registers.producer_d.first_register}, "
            f"v{registers.producer_d.first_register}.l"
        )
        if optimized_decode:
            asm.inst(
                f"v_mul_f32 v{registers.producer_d.first_register}, 0.25, "
                f"v{registers.producer_d.first_register}"
            )

        index = registers.decode_auxiliary.first_register + 1
        high = registers.decode_auxiliary.first_register + 2
        for pair in range(8):
            packed_index = registers.producer_indices.first_register + pair
            grid = registers.codebook_payload.first_register + 4 * pair
            asm.inst(f"v_and_b32 v{index}, 0xff, v{packed_index}")
            if optimized_decode:
                asm.inst(
                    f"v_bfe_u32 v{high}, v{registers.producer_qh.first_register}, "
                    f"{4 * pair}, 2"
                )
            else:
                asm.inst(
                    f"v_lshrrev_b32 v{high}, {4 * pair}, "
                    f"v{registers.producer_qh.first_register}"
                )
                asm.inst(f"v_and_b32 v{high}, 3, v{high}")
            asm.inst(f"v_lshl_or_b32 v{index}, v{high}, 8, v{index}")
            asm.inst(f"v_lshlrev_b32 v{address}, 3, v{index}")
            asm.inst(
                f"global_load_b64 v[{grid}:{grid + 1}], v{address}, "
                f"s[{self.GRID_BASE}:{self.GRID_BASE + 1}]"
            )
            asm.inst(f"v_lshrrev_b32 v{index}, 8, v{packed_index}")
            if optimized_decode:
                asm.inst(
                    f"v_bfe_u32 v{high}, v{registers.producer_qh.first_register}, "
                    f"{4 * pair + 2}, 2"
                )
            else:
                asm.inst(
                    f"v_lshrrev_b32 v{high}, {4 * pair + 2}, "
                    f"v{registers.producer_qh.first_register}"
                )
                asm.inst(f"v_and_b32 v{high}, 3, v{high}")
            asm.inst(f"v_lshl_or_b32 v{index}, v{high}, 8, v{index}")
            asm.inst(f"v_lshlrev_b32 v{address}, 3, v{index}")
            asm.inst(
                f"global_load_b64 v[{grid + 2}:{grid + 3}], v{address}, "
                f"s[{self.GRID_BASE}:{self.GRID_BASE + 1}]"
            )
        asm.inst("s_waitcnt vmcnt(0)")

        for pair in range(8):
            packed_signs = registers.producer_signs.first_register + pair
            grid = registers.codebook_payload.first_register + 4 * pair
            decoded = registers.decoded_payload.first_register
            sign_byte = registers.decode_auxiliary.first_register
            for group in range(2):
                if group:
                    asm.inst(f"v_lshrrev_b32 v{sign_byte}, 8, v{packed_signs}")
                else:
                    asm.inst(f"v_and_b32 v{sign_byte}, 0xff, v{packed_signs}")
                self._emit_signed_grid_dword(
                    asm, grid + 2 * group, sign_byte, 0, decoded + 2 * group
                )
                self._emit_signed_grid_dword(
                    asm, grid + 2 * group + 1, sign_byte, 4, decoded + 2 * group + 1
                )
            asm.inst(
                f"ds_write_b128 v{registers.producer_lds_address.first_register}, "
                f"v[{decoded}:{decoded + 3}] offset:{16 * pair}"
            )
            scale = registers.decode_auxiliary.first_register + 1
            if optimized_decode:
                asm.inst(
                    f"v_bfe_u32 v{scale}, "
                    f"v{registers.producer_scales.first_register}, {4 * pair}, 4"
                )
            else:
                asm.inst(
                    f"v_lshrrev_b32 v{scale}, {4 * pair}, "
                    f"v{registers.producer_scales.first_register}"
                )
                asm.inst(f"v_and_b32 v{scale}, 15, v{scale}")
            asm.inst(f"v_cvt_f32_u32 v{scale}, v{scale}")
            if optimized_decode:
                asm.inst(f"v_add_f32 v{scale}, 0.5, v{scale}")
                asm.inst(
                    f"v_mul_f32 v{scale}, "
                    f"v{registers.producer_d.first_register}, v{scale}"
                )
            else:
                asm.inst(
                    f"v_mul_f32 v{scale}, "
                    f"v{registers.producer_d.first_register}, v{scale}"
                )
                asm.inst(
                    f"v_fmac_f32 v{scale}, 0.5, v{registers.producer_d.first_register}"
                )
                asm.inst(f"v_mul_f32 v{scale}, 0.25, v{scale}")
            asm.inst(
                f"ds_write_b32 v{registers.producer_lds_address.first_register}, "
                f"v{scale} offset:{layout.weight_scale_offset + 4 * pair}"
            )

    def _emit_signed_grid_dword(
        self,
        asm: Assembly,
        positive: int,
        sign_byte: int,
        sign_shift: int,
        destination: int,
    ) -> None:
        auxiliary = self._physical_plan().registers.decode_auxiliary.first_register
        sign_bits = auxiliary + 2
        selector = auxiliary + 3
        negative = auxiliary + 4
        if self._uses_payload_prefetch():
            asm.inst(f"v_bfe_u32 v{sign_bits}, v{sign_byte}, {sign_shift}, 4")
        else:
            asm.inst(f"v_lshrrev_b32 v{sign_bits}, {sign_shift}, v{sign_byte}")
            asm.inst(f"v_and_b32 v{sign_bits}, 15, v{sign_bits}")
        if self._uses_payload_prefetch():
            scalar = self._physical_plan().scalar_registers
            asm.inst(f"v_mul_lo_u32 v{selector}, 0x810204, v{sign_bits}")
            asm.inst(
                f"v_and_or_b32 v{selector}, v{selector}, 0x04040404, "
                f"s{scalar.group_offset.first_register}"
            )
        else:
            asm.inst(f"v_mul_lo_u32 v{selector}, 0x204081, v{sign_bits}")
            asm.inst(f"v_and_b32 v{selector}, 0x01010101, v{selector}")
            asm.inst(f"v_lshl_or_b32 v{selector}, v{selector}, 2, 0x03020100")
        asm.inst(f"v_not_b32 v{negative}, v{positive}")
        asm.inst(f"v_add_nc_u32 v{negative}, 0x01010101, v{negative}")
        # On gfx11 selectors 0..3 choose the second listed data operand.
        asm.inst(f"v_perm_b32 v{destination}, v{negative}, v{positive}, v{selector}")

    def _emit_activation_stage(
        self,
        asm: Assembly,
        layout: GroupedIQ2SFullWeightLdsLayout,
        *,
        stage_index: int,
    ) -> None:
        if (
            self.context.state.kernel_spec.activation.staging
            is GroupedActivationStaging.AggregateRowsTiledLinear
        ):
            self._emit_linear_activation_stage(asm, layout, stage_index=stage_index)
            return
        registers = self._physical_plan().registers
        scalar = self._physical_plan().scalar_registers
        temporary = registers.temporary.first_register
        address = registers.activation_lds_address.first_register
        stage = registers.activation_stage.first_register
        activations = scalar.activations.first_register
        asm.comment("Stage two 72-byte halves per routed F32_D4 activation row.")
        asm.inst(f"v_and_b32 v{address}, 1, v{registers.wave.first_register}")
        asm.inst(f"v_lshlrev_b32 v{address}, 5, v{address}")
        asm.inst(
            f"v_add_nc_u32 v{address}, v{registers.lane.first_register}, v{address}"
        )
        asm.inst(f"v_mul_lo_u32 v{address}, {layout.activation_row_stride}, v{address}")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 1, v{registers.wave.first_register}")
        asm.inst(f"v_mul_lo_u32 v{temporary}, 72, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{address}, v{temporary}, v{address}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{registers.wave.first_register}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 5, v{temporary}")
        asm.inst(
            f"v_add_nc_u32 v{temporary}, v{registers.lane.first_register}, v{temporary}"
        )
        asm.inst(
            f"v_add_nc_u32 v{temporary}, s{scalar.row_start.first_register}, v{temporary}"
        )
        asm.inst(
            f"v_cmp_lt_u32 vcc_lo, v{temporary}, s{scalar.row_tile_end.first_register}"
        )
        asm.inst(f"s_and_saveexec_b32 s{scalar.exec_mask.first_register}, vcc_lo")
        asm.inst(
            f"v_add_nc_u32 v{temporary}, s{scalar.packed_block_offset.first_register}, "
            f"v{address}"
        )
        for chunk in range(4):
            payload = stage + 4 * chunk
            asm.inst(
                f"global_load_b128 v[{payload}:{payload + 3}], v{temporary}, "
                f"s[{activations}:{activations + 1}] offset:{16 * chunk}"
            )
        asm.inst(
            f"global_load_b64 v[{stage + 16}:{stage + 17}], v{temporary}, "
            f"s[{activations}:{activations + 1}] offset:64"
        )
        asm.inst(f"s_mov_b32 exec_lo, s{scalar.exec_mask.first_register}")
        asm.inst("s_waitcnt vmcnt(0)")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{registers.wave.first_register}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 5, v{temporary}")
        asm.inst(
            f"v_add_nc_u32 v{temporary}, v{registers.lane.first_register}, v{temporary}"
        )
        asm.inst(
            f"v_add_nc_u32 v{temporary}, s{scalar.row_start.first_register}, v{temporary}"
        )
        asm.inst(
            f"v_cmp_lt_u32 vcc_lo, v{temporary}, s{scalar.row_tile_end.first_register}"
        )
        asm.inst(f"s_and_saveexec_b32 s{scalar.exec_mask.first_register}, vcc_lo")
        for chunk in range(4):
            payload = stage + 4 * chunk
            asm.inst(
                f"ds_write_b128 v{address}, v[{payload}:{payload + 3}] "
                f"offset:{16 * chunk}"
            )
        asm.inst(f"ds_write_b64 v{address}, v[{stage + 16}:{stage + 17}] offset:64")
        asm.inst(f"s_mov_b32 exec_lo, s{scalar.exec_mask.first_register}")

    def _emit_linear_activation_stage(
        self,
        asm: Assembly,
        layout: GroupedIQ2SFullWeightLdsLayout,
        *,
        stage_index: int,
    ) -> None:
        registers = self._physical_plan().registers
        scalar = self._physical_plan().scalar_registers
        serial = registers.temporary.first_register
        global_address = serial + 1
        local_address = registers.activation_lds_address.first_register
        payload = registers.activation_stage.first_register
        activations = scalar.activations.first_register
        label_token = self._activation_label_token()
        partial_label = f".LGrouped{label_token}ActivationPartial{stage_index}"
        store_label = f".LGrouped{label_token}ActivationStore{stage_index}"
        done_label = f".LGrouped{label_token}ActivationDone{stage_index}"

        asm.comment("Linearly stage one coalesced 9,216-byte F32_D4 activation tile.")
        asm.inst(f"v_lshlrev_b32 v{serial}, 5, v{registers.wave.first_register}")
        asm.inst(f"v_add_nc_u32 v{serial}, v{registers.lane.first_register}, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{local_address}, 4, v{serial}")
        asm.inst(f"s_cmp_eq_u32 s{scalar.row_tile_rows.first_register}, 64")
        asm.inst(f"s_cbranch_scc0 {partial_label}")
        asm.inst(
            f"v_add_nc_u32 v{global_address}, "
            f"s{scalar.packed_block_offset.first_register}, v{local_address}"
        )
        for chunk in range(4):
            destination = payload + 4 * chunk
            if chunk == 2:
                asm.inst(f"v_add_nc_u32 v{global_address}, 4096, v{global_address}")
            asm.inst(
                f"global_load_b128 v[{destination}:{destination + 3}], "
                f"v{global_address}, s[{activations}:{activations + 1}] "
                f"offset:{2048 * (chunk % 2)}"
            )
        asm.inst(f"v_lshlrev_b32 v{local_address}, 3, v{serial}")
        asm.inst(f"v_add_nc_u32 v{local_address}, 8192, v{local_address}")
        asm.inst(
            f"v_add_nc_u32 v{global_address}, "
            f"s{scalar.packed_block_offset.first_register}, v{local_address}"
        )
        asm.inst(
            f"global_load_b64 v[{payload + 16}:{payload + 17}], "
            f"v{global_address}, s[{activations}:{activations + 1}]"
        )
        asm.inst(f"s_branch {store_label}")

        asm.label(partial_label)
        asm.inst(
            f"s_mul_i32 s{scalar.activation_plane_end.first_register}, "
            f"s{scalar.row_tile_rows.first_register}, {layout.activation_row_stride}"
        )
        for chunk in range(4):
            destination = payload + 4 * chunk
            asm.inst(
                f"v_cmp_lt_u32 vcc_lo, v{local_address}, "
                f"s{scalar.activation_plane_end.first_register}"
            )
            asm.inst(f"s_and_saveexec_b32 s{scalar.exec_mask.first_register}, vcc_lo")
            asm.inst(
                f"v_add_nc_u32 v{global_address}, "
                f"s{scalar.packed_block_offset.first_register}, v{local_address}"
            )
            asm.inst(
                f"global_load_b128 v[{destination}:{destination + 3}], "
                f"v{global_address}, s[{activations}:{activations + 1}]"
            )
            asm.inst(f"s_mov_b32 exec_lo, s{scalar.exec_mask.first_register}")
            if chunk != 3:
                asm.inst(f"v_add_nc_u32 v{local_address}, 2048, v{local_address}")
        asm.inst(f"v_lshlrev_b32 v{local_address}, 3, v{serial}")
        asm.inst(f"v_add_nc_u32 v{local_address}, 8192, v{local_address}")
        asm.inst(
            f"v_cmp_lt_u32 vcc_lo, v{local_address}, "
            f"s{scalar.activation_plane_end.first_register}"
        )
        asm.inst(f"s_and_saveexec_b32 s{scalar.exec_mask.first_register}, vcc_lo")
        asm.inst(
            f"v_add_nc_u32 v{global_address}, "
            f"s{scalar.packed_block_offset.first_register}, v{local_address}"
        )
        asm.inst(
            f"global_load_b64 v[{payload + 16}:{payload + 17}], "
            f"v{global_address}, s[{activations}:{activations + 1}]"
        )
        asm.inst(f"s_mov_b32 exec_lo, s{scalar.exec_mask.first_register}")

        asm.label(store_label)
        asm.inst("s_waitcnt vmcnt(0)")
        asm.inst(f"v_lshlrev_b32 v{local_address}, 4, v{serial}")
        for chunk in range(4):
            source = payload + 4 * chunk
            asm.inst(
                f"ds_write_b128 v{local_address}, v[{source}:{source + 3}] "
                f"offset:{2048 * chunk}"
            )
        asm.inst(f"v_lshlrev_b32 v{local_address}, 3, v{serial}")
        asm.inst(f"v_add_nc_u32 v{local_address}, 8192, v{local_address}")
        asm.inst(f"ds_write_b64 v{local_address}, v[{payload + 16}:{payload + 17}]")
        asm.label(done_label)

    def _emit_compute_half(
        self,
        asm: Assembly,
        layout: GroupedIQ2SFullWeightLdsLayout,
        *,
        half: int,
    ) -> None:
        if self._uses_payload_prefetch():
            self._emit_compute_half_prefetched(asm, layout, half=half)
            return
        registers = self._physical_plan().registers
        waits_even = (9, 8, 1, 0)
        waits_odd = (7, 6, 1, 0)
        for group in range(8):
            activation_offset = 16 + 16 * group
            asm.comment(
                f"IQ2_S half {half} group {group}: signed WMMA and FP32 factors."
            )
            asm.inst(
                f"ds_read_b128 v[{registers.activation_payload.first_register}:"
                f"{registers.activation_payload.first_register + 3}], "
                f"v{registers.activation_read_address.first_register} "
                f"offset:{activation_offset}"
            )
            asm.inst(
                f"ds_read_b128 v[{registers.weight_payload.first_register}:"
                f"{registers.weight_payload.first_register + 3}], "
                f"v{registers.weight_lds_address.first_register} "
                f"offset:{layout.half_payload_stride * half + 16 * group}"
            )
            asm.inst(
                f"ds_read_b128 v[{registers.activation_payload.first_register + 4}:"
                f"{registers.activation_payload.first_register + 7}], "
                f"v{registers.activation_read_address.first_register} "
                f"offset:{16 * layout.activation_row_stride + activation_offset}"
            )
            weight_offset0 = 32 + 40 * half + group
            weight_offset1 = 200 + 40 * half + group
            for index in range(4):
                base = registers.weight_scale_address.first_register + index
                destination = registers.weight_scales.first_register + 2 * index
                asm.inst(
                    f"ds_read2_b32 v[{destination}:{destination + 1}], v{base} "
                    f"offset0:{weight_offset0} offset1:{weight_offset1}"
                )
            if group % 2 == 0:
                asm.inst(
                    f"v_add_nc_u32 v{registers.temporary.first_register}, "
                    f"{4 * (group // 2)}, "
                    f"v{registers.activation_read_address.first_register}"
                )
                for index, (offset0, offset1) in enumerate(((0, 9), (18, 27))):
                    destination = registers.activation_scale.first_register + 2 * index
                    asm.inst(
                        f"ds_read2st64_b32 v[{destination}:{destination + 1}], "
                        f"v{registers.temporary.first_register} "
                        f"offset0:{offset0} offset1:{offset1}"
                    )
            for m_index in range(2, 4):
                payload = registers.activation_payload.first_register + 4 * m_index
                asm.inst(
                    f"ds_read_b128 v[{payload}:{payload + 3}], "
                    f"v{registers.activation_read_address.first_register} "
                    f"offset:{16 * m_index * layout.activation_row_stride + activation_offset}"
                )
            for wait, m_index in zip(
                waits_even if group % 2 == 0 else waits_odd,
                range(4),
            ):
                asm.inst(f"s_waitcnt lgkmcnt({wait})")
                emit_signed_i8_wmma(
                    asm,
                    destination=registers.c.first_register + 8 * m_index,
                    weight=registers.weight_payload.first_register,
                    activation=registers.activation_payload.first_register
                    + 4 * m_index,
                    accumulator=registers.zero_accumulator.first_register,
                    clamp=False,
                )
            self._emit_fragment_correction(asm, 0, 3)
            self._emit_fragment_correction(asm, 1, 2)

    def _emit_compute_half_prefetched(
        self,
        asm: Assembly,
        layout: GroupedIQ2SFullWeightLdsLayout,
        *,
        half: int,
    ) -> None:
        self._emit_compute_group_payload_reads(asm, layout, half=half, group=0)
        self._emit_compute_group_scale_reads(asm, half=half, group=0)
        asm.inst("s_waitcnt lgkmcnt(0)")
        self._emit_compute_group_wmmas(asm)
        for group in range(8):
            if group != 7:
                self._emit_compute_group_payload_reads(
                    asm, layout, half=half, group=group + 1
                )
            self._emit_fragment_correction(asm, 0, 3)
            self._emit_fragment_correction(asm, 1, 2)
            if group == 7:
                continue
            next_group = group + 1
            self._emit_compute_group_scale_reads(asm, half=half, group=next_group)
            pending_scales = 6 if next_group % 2 == 0 else 4
            asm.inst(f"s_waitcnt lgkmcnt({pending_scales})")
            self._emit_compute_group_wmmas(asm)
            asm.inst("s_waitcnt lgkmcnt(0)")

    def _emit_compute_group_payload_reads(
        self,
        asm: Assembly,
        layout: GroupedIQ2SFullWeightLdsLayout,
        *,
        half: int,
        group: int,
    ) -> None:
        registers = self._physical_plan().registers
        activation_offset = 16 + 16 * group
        asm.comment(f"Prefetch IQ2_S half {half} group {group} payloads.")
        for m_index in range(4):
            payload = registers.activation_payload.first_register + 4 * m_index
            asm.inst(
                f"ds_read_b128 v[{payload}:{payload + 3}], "
                f"v{registers.activation_read_address.first_register} "
                f"offset:{16 * m_index * layout.activation_row_stride + activation_offset}"
            )
        asm.inst(
            f"ds_read_b128 v[{registers.weight_payload.first_register}:"
            f"{registers.weight_payload.first_register + 3}], "
            f"v{registers.weight_lds_address.first_register} "
            f"offset:{layout.half_payload_stride * half + 16 * group}"
        )

    def _emit_compute_group_scale_reads(
        self,
        asm: Assembly,
        *,
        half: int,
        group: int,
    ) -> None:
        registers = self._physical_plan().registers
        weight_offset0 = 32 + 40 * half + group
        weight_offset1 = 200 + 40 * half + group
        for index in range(4):
            base = registers.weight_scale_address.first_register + index
            destination = registers.weight_scales.first_register + 2 * index
            asm.inst(
                f"ds_read2_b32 v[{destination}:{destination + 1}], v{base} "
                f"offset0:{weight_offset0} offset1:{weight_offset1}"
            )
        if group % 2 == 0:
            asm.inst(
                f"v_add_nc_u32 v{registers.temporary.first_register}, "
                f"{4 * (group // 2)}, "
                f"v{registers.activation_read_address.first_register}"
            )
            for index, (offset0, offset1) in enumerate(((0, 9), (18, 27))):
                destination = registers.activation_scale.first_register + 2 * index
                asm.inst(
                    f"ds_read2st64_b32 v[{destination}:{destination + 1}], "
                    f"v{registers.temporary.first_register} "
                    f"offset0:{offset0} offset1:{offset1}"
                )

    def _emit_compute_group_wmmas(self, asm: Assembly) -> None:
        registers = self._physical_plan().registers
        for m_index in range(4):
            emit_signed_i8_wmma(
                asm,
                destination=registers.c.first_register + 8 * m_index,
                weight=registers.weight_payload.first_register,
                activation=registers.activation_payload.first_register + 4 * m_index,
                accumulator=registers.zero_accumulator.first_register,
                clamp=False,
            )

    def _emit_fragment_correction(
        self,
        asm: Assembly,
        first: int,
        second: int,
    ) -> None:
        registers = self._physical_plan().registers
        for fragment in (first, second):
            base = registers.c.first_register + 8 * fragment
            for element in range(8):
                asm.inst(f"v_cvt_f32_i32 v{base + element}, v{base + element}")
            for element in range(0, 8, 2):
                asm.inst(
                    f"v_dual_mul_f32 v{base + element}, "
                    f"v{registers.weight_scales.first_register + element}, "
                    f"v{base + element} :: "
                    f"v_dual_mul_f32 v{base + element + 1}, "
                    f"v{registers.weight_scales.first_register + element + 1}, "
                    f"v{base + element + 1}"
                )
        first_sum = registers.sums.first_register + 8 * first
        second_sum = registers.sums.first_register + 8 * second
        first_scale = registers.activation_scale.first_register + first
        second_scale = registers.activation_scale.first_register + second
        second_base = registers.c.first_register + 8 * second
        for element in range(8):
            asm.inst(
                f"v_dual_fmac_f32 v{first_sum + element}, v{first_scale}, "
                f"v{registers.c.first_register + 8 * first + element} :: "
                f"v_dual_fmac_f32 v{second_sum + (element ^ 3)}, v{second_scale}, "
                f"v{second_base + (element ^ 3)}"
            )

    def _emit_store(self, asm: Assembly) -> None:
        registers = self._physical_plan().registers
        scalar = self._physical_plan().scalar_registers
        temporary = registers.temporary.first_register
        column = registers.decode_auxiliary.first_register
        asm.comment("Store four route-bounds-masked J64 BF16 fragments.")
        asm.inst(f"v_bfe_u32 v{column}, v{registers.lane.first_register}, 4, 1")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 4, v{registers.wave.first_register}")
        asm.inst(f"v_add_nc_u32 v{column}, v{temporary}, v{column}")
        asm.inst(
            f"v_lshlrev_b32 v{temporary}, 6, s{scalar.workgroup_tile.first_register}"
        )
        asm.inst(f"v_add_nc_u32 v{column}, v{temporary}, v{column}")
        asm.inst(f"v_lshlrev_b32 v{column}, 1, v{column}")
        for m_index in range(4):
            fragment = registers.sums.first_register + 8 * m_index
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
                f"v_add_nc_u32 v{registers.output_address.first_register}, "
                f"v{column}, v{registers.output_address.first_register}"
            )
            asm.inst("s_clause 7")
            for element in range(8):
                asm.inst(
                    f"global_store_d16_hi_b16 v{registers.output_address.first_register}, "
                    f"v{fragment + element}, s[8:9] offset:{4 * element}"
                )
            asm.inst(f"s_mov_b32 exec_lo, s{scalar.exec_mask.first_register}")
