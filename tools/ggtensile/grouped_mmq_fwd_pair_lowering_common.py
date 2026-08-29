"""Shared mechanical emitters for paired K128 grouped forward lowerings."""

from dataclasses import dataclass

from .grouped_mmq_fwd_model import GroupedActivationStaging
from .grouped_mmq_fwd_pair_model import GroupedForwardPairProblem
from .grouped_mmq_fwd_pair_physical import (
    GroupedIQ2SPairHalfLdsLayout,
    GroupedIQ2SPairPhysicalPlan,
    GroupedIQ2XXSPairHalfLdsLayout,
    GroupedIQ2XXSPairPhysicalPlan,
    GroupedQ3KPairHalfLdsLayout,
    GroupedQ3KPairPhysicalPlan,
)
from .grouped_mmq_fwd_pair_spec import DerivedGroupedForwardPairState
from .kernel_writer_assembly import Assembly
from .mmq_fwd_lowering_mma import emit_signed_i8_wmma

GroupedPairPhysicalPlan = (
    GroupedIQ2SPairPhysicalPlan
    | GroupedIQ2XXSPairPhysicalPlan
    | GroupedQ3KPairPhysicalPlan
)
GroupedPairHalfLdsLayout = (
    GroupedIQ2SPairHalfLdsLayout
    | GroupedIQ2XXSPairHalfLdsLayout
    | GroupedQ3KPairHalfLdsLayout
)


@dataclass(frozen=True)
class GroupedForwardPairLoweringContext:
    kernel_name: str
    problem: GroupedForwardPairProblem
    state: DerivedGroupedForwardPairState


@dataclass(frozen=True)
class GroupedPairK128Mechanics:
    context: GroupedForwardPairLoweringContext
    physical: GroupedPairPhysicalPlan
    activation_label_token: str

    def emit_invariant_addresses(
        self,
        asm: Assembly,
        layout: GroupedPairHalfLdsLayout,
    ) -> None:
        registers = self.physical.registers
        scalar = self.physical.scalar_registers
        temporary = registers.temporary.first_register
        asm.comment("Map four waves to 64 rows and split each selected K half.")
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
            f"{layout.weight_base}, v{registers.weight_scale_address.first_register}"
        )
        for index in range(1, 4):
            asm.inst(
                f"v_add_nc_u32 v{registers.weight_scale_address.first_register + index}, "
                f"{index * 4 * layout.weight_row_stride}, "
                f"v{registers.weight_scale_address.first_register}"
            )

    def emit_activation_stage(
        self,
        asm: Assembly,
        layout: GroupedPairHalfLdsLayout,
        *,
        stage_index: int,
    ) -> None:
        if (
            self.context.state.kernel_spec.activation_staging
            is GroupedActivationStaging.AggregateRowsTiledLinear
        ):
            self.emit_linear_activation_stage(
                asm,
                layout,
                stage_index=stage_index,
            )
            return
        registers = self.physical.registers
        scalar = self.physical.scalar_registers
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
            f"v_add_nc_u32 v{temporary}, s{scalar.row_start.first_register}, "
            f"v{temporary}"
        )
        asm.inst(
            f"v_cmp_lt_u32 vcc_lo, v{temporary}, s{scalar.row_tile_end.first_register}"
        )
        asm.inst(f"s_and_saveexec_b32 s{scalar.exec_mask.first_register}, vcc_lo")
        asm.inst(
            f"v_add_nc_u32 v{temporary}, "
            f"s{scalar.packed_block_offset.first_register}, v{address}"
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
            f"v_add_nc_u32 v{temporary}, s{scalar.row_start.first_register}, "
            f"v{temporary}"
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

    def emit_linear_activation_stage(
        self,
        asm: Assembly,
        layout: GroupedPairHalfLdsLayout,
        *,
        stage_index: int,
    ) -> None:
        if layout.activation_rows == 80:
            self._emit_linear_activation_stage_j80(asm, layout, stage_index=stage_index)
            return
        registers = self.physical.registers
        scalar = self.physical.scalar_registers
        serial = registers.temporary.first_register
        global_address = serial + 1
        local_address = registers.activation_lds_address.first_register
        payload = registers.activation_stage.first_register
        activations = scalar.activations.first_register
        partial_label = (
            f".LGrouped{self.activation_label_token}ActivationPartial{stage_index}"
        )
        store_label = (
            f".LGrouped{self.activation_label_token}ActivationStore{stage_index}"
        )
        done_label = (
            f".LGrouped{self.activation_label_token}ActivationDone{stage_index}"
        )

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

    def _emit_linear_activation_stage_j80(
        self,
        asm: Assembly,
        layout: GroupedPairHalfLdsLayout,
        *,
        stage_index: int,
    ) -> None:
        registers = self.physical.registers
        scalar = self.physical.scalar_registers
        serial = registers.temporary.first_register
        global_address = serial + 1
        local_address = registers.activation_lds_address.first_register
        payload = registers.activation_stage.first_register
        activations = scalar.activations.first_register
        plane_end = scalar.activation_plane_end.first_register
        exec_mask = scalar.exec_mask.first_register
        partial_label = f".LGroupedPairIQ2XXSActivationPartial80_{stage_index}"
        done_label = f".LGroupedPairIQ2XXSActivationDone80_{stage_index}"

        asm.comment("Linearly stage one 11,520-byte F32_D4 activation tile for J80.")
        asm.inst(
            f"s_mul_i32 s{plane_end}, s{scalar.row_tile_rows.first_register}, "
            f"{layout.activation_row_stride}"
        )
        asm.inst(f"v_lshlrev_b32 v{serial}, 5, v{registers.wave.first_register}")
        asm.inst(f"v_add_nc_u32 v{serial}, v{registers.lane.first_register}, v{serial}")
        asm.inst(f"s_cmp_eq_u32 s{scalar.row_tile_rows.first_register}, 80")
        asm.inst(f"s_cbranch_scc0 {partial_label}")
        for band in range(6):
            self._emit_j80_activation_load(
                asm,
                band,
                serial,
                global_address,
                local_address,
                payload,
                activations,
                str(layout.activation_bytes) if band == 5 else None,
                exec_mask,
            )
        asm.inst(f"s_branch {done_label}")
        asm.label(partial_label)
        for band in range(6):
            self._emit_j80_activation_load(
                asm,
                band,
                serial,
                global_address,
                local_address,
                payload,
                activations,
                f"s{plane_end}",
                exec_mask,
            )
        asm.label(done_label)
        asm.inst("s_waitcnt vmcnt(0)")
        for band in range(6):
            asm.inst(f"v_lshlrev_b32 v{local_address}, 4, v{serial}")
            if band:
                asm.inst(
                    f"v_add_nc_u32 v{local_address}, {band * 2048}, v{local_address}"
                )
            if band == 5:
                asm.inst(f"v_cmp_lt_u32 vcc_lo, v{local_address}, s{plane_end}")
                asm.inst(f"s_and_saveexec_b32 s{exec_mask}, vcc_lo")
            source = payload + 4 * band
            asm.inst(f"ds_write_b128 v{local_address}, v[{source}:{source + 3}]")
            if band == 5:
                asm.inst(f"s_mov_b32 exec_lo, s{exec_mask}")

    def _emit_j80_activation_load(
        self,
        asm: Assembly,
        band: int,
        serial: int,
        global_address: int,
        local_address: int,
        payload: int,
        activations: int,
        limit: str | None,
        exec_mask: int,
    ) -> None:
        scalar = self.physical.scalar_registers
        asm.inst(f"v_lshlrev_b32 v{local_address}, 4, v{serial}")
        if band:
            asm.inst(f"v_add_nc_u32 v{local_address}, {band * 2048}, v{local_address}")
        if limit is not None:
            asm.inst(f"v_cmp_lt_u32 vcc_lo, v{local_address}, {limit}")
            asm.inst(f"s_and_saveexec_b32 s{exec_mask}, vcc_lo")
        asm.inst(
            f"v_add_nc_u32 v{global_address}, "
            f"s{scalar.packed_block_offset.first_register}, v{local_address}"
        )
        destination = payload + 4 * band
        asm.inst(
            f"global_load_b128 v[{destination}:{destination + 3}], "
            f"v{global_address}, s[{activations}:{activations + 1}]"
        )
        if limit is not None:
            asm.inst(f"s_mov_b32 exec_lo, s{exec_mask}")

    def emit_signed_grid_dword(
        self,
        asm: Assembly,
        positive: int,
        sign_byte: int,
        sign_shift: int,
        destination: int,
    ) -> None:
        registers = self.physical.registers
        scalar = self.physical.scalar_registers
        auxiliary = registers.decode_auxiliary.first_register
        sign_bits = auxiliary + 2
        selector = auxiliary + 3
        negative = auxiliary + 4
        asm.inst(f"v_bfe_u32 v{sign_bits}, v{sign_byte}, {sign_shift}, 4")
        asm.inst(f"v_mul_lo_u32 v{selector}, 0x810204, v{sign_bits}")
        asm.inst(
            f"v_and_or_b32 v{selector}, v{selector}, 0x04040404, "
            f"s{scalar.group_offset.first_register}"
        )
        asm.inst(f"v_not_b32 v{negative}, v{positive}")
        asm.inst(f"v_add_nc_u32 v{negative}, 0x01010101, v{negative}")
        asm.inst(f"v_perm_b32 v{destination}, v{negative}, v{positive}, v{selector}")

    def emit_compute_projection(self, asm: Assembly, sums: int) -> None:
        layout = self.physical.layout
        self.emit_compute_group_payload_reads(asm, layout, 0)
        self.emit_compute_group_scale_reads(asm, 0)
        asm.inst("s_waitcnt lgkmcnt(0)")
        self.emit_compute_group_wmmas(asm)
        for group in range(8):
            self.emit_fragment_correction(asm, 0, 3, sums)
            self.emit_fragment_correction(asm, 1, 2, sums)
            if group == 7:
                continue
            next_group = group + 1
            self.emit_compute_group_payload_reads(asm, layout, next_group)
            self.emit_compute_group_scale_reads(asm, next_group)
            pending_scales = 6 if next_group % 2 == 0 else 4
            asm.inst(f"s_waitcnt lgkmcnt({pending_scales})")
            self.emit_compute_group_wmmas(asm)
            asm.inst("s_waitcnt lgkmcnt(0)")

    def emit_compute_group_payload_reads(
        self,
        asm: Assembly,
        layout: GroupedPairHalfLdsLayout,
        group: int,
    ) -> None:
        registers = self.physical.registers
        activation_offset = 16 + 16 * group
        for m_index in range(layout.activation_rows // 16):
            payload = registers.activation_payload.first_register + 4 * m_index
            asm.inst(
                f"ds_read_b128 v[{payload}:{payload + 3}], "
                f"v{registers.activation_read_address.first_register} "
                f"offset:{16 * m_index * layout.activation_row_stride + activation_offset}"
            )
        asm.inst(
            f"ds_read_b128 v[{registers.weight_payload.first_register}:"
            f"{registers.weight_payload.first_register + 3}], "
            f"v{registers.weight_lds_address.first_register} offset:{16 * group}"
        )

    def emit_compute_group_scale_reads(self, asm: Assembly, group: int) -> None:
        registers = self.physical.registers
        layout = self.physical.layout
        first_row_offset = layout.weight_scale_offset // 4 + group
        second_row_offset = (
            2 * layout.weight_row_stride + layout.weight_scale_offset
        ) // 4 + group
        for index in range(4):
            base = registers.weight_scale_address.first_register + index
            destination = registers.weight_scales.first_register + 2 * index
            asm.inst(
                f"ds_read2_b32 v[{destination}:{destination + 1}], v{base} "
                f"offset0:{first_row_offset} offset1:{second_row_offset}"
            )
        if group % 2 == 0:
            temporary = registers.temporary.first_register
            asm.inst(
                f"v_add_nc_u32 v{temporary}, {4 * (group // 2)}, "
                f"v{registers.activation_read_address.first_register}"
            )
            for index, (offset0, offset1) in enumerate(((0, 9), (18, 27))):
                destination = registers.activation_scale.first_register + 2 * index
                asm.inst(
                    f"ds_read2st64_b32 v[{destination}:{destination + 1}], "
                    f"v{temporary} offset0:{offset0} offset1:{offset1}"
                )
            if layout.activation_rows == 80:
                destination = registers.activation_scale.first_register + 4
                asm.inst(
                    f"ds_read_b32 v{destination}, v{temporary} "
                    f"offset:{64 * layout.activation_row_stride}"
                )

    def emit_compute_group_wmmas(self, asm: Assembly) -> None:
        registers = self.physical.registers
        for m_index in range(4):
            emit_signed_i8_wmma(
                asm,
                destination=registers.c.first_register + 8 * m_index,
                weight=registers.weight_payload.first_register,
                activation=registers.activation_payload.first_register + 4 * m_index,
                accumulator=registers.zero_accumulator.first_register,
                clamp=False,
            )

    def emit_fragment_correction(
        self,
        asm: Assembly,
        first: int,
        second: int,
        sums: int,
    ) -> None:
        registers = self.physical.registers
        for fragment in (first, second):
            base = registers.c.first_register + 8 * fragment
            for element in range(8):
                asm.inst(f"v_cvt_f32_i32 v{base + element}, v{base + element}")
            for element in range(0, 8, 2):
                asm.inst(
                    f"v_dual_mul_f32 v{base + element}, "
                    f"v{registers.weight_scales.first_register + element}, "
                    f"v{base + element} :: v_dual_mul_f32 v{base + element + 1}, "
                    f"v{registers.weight_scales.first_register + element + 1}, "
                    f"v{base + element + 1}"
                )
        first_sum = sums + 8 * first
        second_sum = sums + 8 * second
        first_scale = registers.activation_scale.first_register + first
        second_scale = registers.activation_scale.first_register + second
        second_base = registers.c.first_register + 8 * second
        for element in range(8):
            asm.inst(
                f"v_dual_fmac_f32 v{first_sum + element}, v{first_scale}, "
                f"v{registers.c.first_register + 8 * first + element} :: "
                f"v_dual_fmac_f32 v{second_sum + (element ^ 3)}, "
                f"v{second_scale}, v{second_base + (element ^ 3)}"
            )
