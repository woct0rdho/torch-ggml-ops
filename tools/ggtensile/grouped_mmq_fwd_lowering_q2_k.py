"""Grouped Q2_K decoded-LDS forward lowering."""

from typing import cast

from .grouped_mmq_fwd_lowering import (
    GroupedForwardLoweringContext,
)
from .grouped_mmq_fwd_lowering_decoded_lds import GroupedDecodedWeightLdsLowering
from .grouped_mmq_fwd_lowering_row_dispatch import GroupedRowDispatchLabels
from .grouped_mmq_fwd_model import GroupedQ2DecodePolicy
from .grouped_mmq_fwd_route import GroupedRouteEmitter
from .kernel_writer_assembly import Assembly, emit_kernel_trailer
from .mmq_fwd_lowering_mma import emit_signed_i8_wmma
from .mmq_fwd_spec import F16D2S6ActivationMetadata


class GroupedQ2KDecodedWeightLdsLowering(GroupedDecodedWeightLdsLowering):
    """Emit non-paired Q2_K 32-, 64-, or 128-row decoded-LDS controls."""

    def _q2_decode_policy(self) -> GroupedQ2DecodePolicy:
        policy = self.context.state.kernel_spec.decode
        assert isinstance(policy, GroupedQ2DecodePolicy)
        return policy

    def body(self) -> str:
        spec = self.context.state.kernel_spec
        assert spec.weight_staging is self.WEIGHT_STAGING
        assert self.context.problem.quant_data_type == "Q2_K"
        return self._body_q2()

    def _body_q2(self) -> str:
        state = self.context.state
        context = self.context
        physical = self._physical_plan()
        registers = physical.registers
        scalar = physical.scalar_registers
        layout = physical.layout
        activation_metadata = cast(
            F16D2S6ActivationMetadata, layout.activation_metadata
        )
        asm = Assembly()
        name = context.kernel_name
        zero_accumulator = registers.zero_accumulator.first_register
        sum_base = registers.sums.first_register
        temporary = registers.temporary.first_register
        metadata_address = registers.metadata_lds_address.first_register
        activation_plane_address = registers.activation_plane_address.first_register
        output_column = registers.output_column.first_register
        wave_column_base = registers.wave_column_base.first_register
        wave = registers.wave.first_register
        activation_base = registers.activation_base.first_register
        lane = registers.lane.first_register
        serial = registers.serial.first_register
        GroupedRouteEmitter(scalar, state.route).emit(asm)
        asm.inst(
            f"s_mul_i32 s{scalar.activation_plane_stride.first_register}, "
            f"s{scalar.nrows_activation.first_register}, "
            f"{state.contract.activation_block_bytes}"
        )
        asm.inst(
            f"s_mov_b32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_begin.first_register}"
        )
        asm.inst(f"s_mov_b32 s{self.ACTIVATION_LDS_BASE}, {layout.activation_base}")

        asm.comment("Map four waves to one 32-row by 64-column grouped tile.")
        asm.inst(f"v_mov_b32 v{serial}, v0")
        asm.inst(f"v_and_b32 v{lane}, 15, v{serial}")
        asm.inst(f"v_lshrrev_b32 v{wave}, 5, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{wave_column_base}, 4, v{wave}")
        asm.inst(
            f"v_lshlrev_b32 v{temporary}, 6, s{scalar.workgroup_tile.first_register}"
        )
        asm.inst(f"v_add_nc_u32 v{wave_column_base}, v{temporary}, v{wave_column_base}")
        asm.inst(f"v_lshlrev_b32 v{output_column}, 4, v{wave}")
        asm.inst(f"v_add_nc_u32 v{output_column}, v{lane}, v{output_column}")
        asm.inst(
            f"v_mul_lo_u32 v{output_column}, {layout.weight_row_stride}, "
            f"v{output_column}"
        )
        asm.inst(
            f"v_add_nc_u32 v{output_column}, {layout.weight_data_base}, "
            f"v{output_column}"
        )
        asm.inst(f"v_readfirstlane_b32 s{self.WAVE_INDEX}, v{wave}")
        asm.inst(
            f"v_mad_u32_u24 v{activation_base}, {activation_metadata.block_bytes}, "
            f"v{lane}, s{self.ACTIVATION_LDS_BASE}"
        )
        for register in range(
            registers.weight_payload.first_register + 4,
            registers.weight_payload.first_register + 8,
            2,
        ):
            asm.inst(f"v_mov_b32 v{register}, 0x01010101")
            asm.inst(f"v_mov_b32 v{register + 1}, 0x01010101")

        asm.label(".LGroupedQ2KDecodedRowLoop")
        asm.inst(
            f"v_mul_lo_u32 v{activation_plane_address}, "
            f"{activation_metadata.block_bytes}, "
            f"s{scalar.row_start.first_register}"
        )
        asm.inst(f"v_lshlrev_b32 v{temporary}, 2, v{serial}")
        asm.inst(
            f"v_add_nc_u32 v{activation_plane_address}, v{temporary}, "
            f"v{activation_plane_address}"
        )
        asm.inst(
            f"s_mul_i32 s{self.ACTIVATION_PLANE_END}, "
            f"s{scalar.row_end.first_register}, {activation_metadata.block_bytes}"
        )
        asm.inst(
            f"s_sub_u32 s{scalar.row_tile_rows.first_register}, "
            f"s{scalar.row_end.first_register}, s{scalar.row_start.first_register}"
        )
        for register in range(zero_accumulator, zero_accumulator + 8, 2):
            asm.inst(f"v_mov_b32 v{register}, 0")
            asm.inst(f"v_mov_b32 v{register + 1}, 0")
        for register in range(sum_base, sum_base + 8 * self._row_tile_count(), 2):
            asm.inst(f"v_mov_b32 v{register}, v{zero_accumulator}")
            asm.inst(f"v_mov_b32 v{register + 1}, v{zero_accumulator + 1}")
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")
        asm.inst(f"s_mov_b32 s{self.SCALAR_TEMPORARY}, 0")

        asm.label(".LGroupedQ2KDecodedBlockLoop")
        self._emit_q2_weight_decode_stage(asm)
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{metadata_address}, 4, s{self.WAVE_INDEX}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary}, v{metadata_address}")
        asm.inst(
            f"v_mul_lo_u32 v{metadata_address}, {layout.weight_row_stride}, "
            f"v{temporary}"
        )
        asm.inst(
            f"v_add_nc_u32 v{metadata_address}, {layout.weight_metadata_base}, "
            f"v{metadata_address}"
        )

        self._emit_grouped_activation_stage_dispatch(asm, stage=0)
        self._emit_q2_group_dispatch(asm, group_base=0)
        asm.inst("s_barrier")
        self._emit_grouped_activation_stage_dispatch(asm, stage=1)
        self._emit_q2_group_dispatch(asm, group_base=8)
        asm.inst("s_barrier")
        asm.inst(
            f"s_add_u32 s{self.SCALAR_TEMPORARY}, s{self.SCALAR_TEMPORARY}, "
            f"{state.contract.packed_weight_block_bytes}"
        )
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {state.blocks_per_weight_row}")
        asm.inst("s_cbranch_scc1 .LGroupedQ2KDecodedBlockLoop")
        self._emit_grouped_epilogue_dispatch(asm)
        asm.inst(
            f"s_add_u32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_start.first_register}, "
            f"{state.kernel_spec.geometry.macro_tile[0]}"
        )
        asm.inst(
            f"s_cmp_lt_u32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_end.first_register}"
        )
        asm.inst("s_cbranch_scc1 .LGroupedQ2KDecodedRowLoop")
        asm.label(".LGroupedQ4KExit")
        asm.label(".LGroupedQ2KExit")
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_q2_weight_decode_stage(self, asm: Assembly) -> None:
        physical = self._physical_plan()
        layout = physical.layout
        registers = physical.registers
        weights = physical.scalar_registers.weights.first_register
        state = self.context.state
        decode = registers.decode_scratch.first_register
        staged = registers.staged_payload.first_register
        serial = registers.serial.first_register
        temporary = registers.temporary.first_register
        lds_address = registers.lds_address.first_register
        auxiliary = registers.auxiliary.first_register
        metadata = registers.metadata_lds_address.first_register
        weight_base = layout.weight_data_base
        row_stride = state.packed_weight_row_bytes
        ql_offset = 16
        scales_offset = 0
        dm_offset = 80
        policy = self._q2_decode_policy()
        if policy.distributed_producer:
            self._emit_q2_distributed_weight_decode_stage(asm)
            return
        write2_decode = policy.paired_payload_writes
        write2_metadata = policy.paired_metadata_writes
        pre_negated_dm = policy.pre_negated_dm

        asm.comment("Decode Q2_K two-bit payload and scale/minimum metadata into LDS.")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{metadata}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{metadata}, v{temporary}")
        asm.inst(f"v_mul_lo_u32 v{temporary}, {row_stride}, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{temporary}, s{self.SCALAR_TEMPORARY}, v{temporary}")
        asm.inst(f"v_and_b32 v{auxiliary}, 15, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{metadata}, 2, v{auxiliary}")
        asm.inst(f"v_add_nc_u32 v{temporary}, {ql_offset}, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{metadata}, v{temporary}")

        asm.inst(f"v_lshrrev_b32 v{lds_address}, 4, v{serial}")
        asm.inst(
            f"v_mul_lo_u32 v{lds_address}, {layout.weight_row_stride}, v{lds_address}"
        )
        asm.inst(f"v_lshrrev_b32 v{metadata}, 3, v{auxiliary}")
        asm.inst(f"v_lshlrev_b32 v{metadata}, 7, v{metadata}")
        asm.inst(f"v_and_b32 v{auxiliary}, 7, v{auxiliary}")
        asm.inst(f"v_lshlrev_b32 v{auxiliary}, 2, v{auxiliary}")
        asm.inst(
            f"v_add3_u32 v{lds_address}, v{metadata}, v{auxiliary}, v{lds_address}"
        )
        asm.inst(f"v_add_nc_u32 v{lds_address}, {weight_base}, v{lds_address}")

        for item in range(8):
            asm.inst(
                f"global_load_b32 v{staged + item}, v{temporary}, "
                f"s[{weights}:{weights + 1}]"
            )
            if item != 7:
                asm.inst(f"v_add_nc_u32 v{temporary}, {8 * row_stride}, v{temporary}")
        asm.inst(f"s_cmp_lt_u32 s{self.WAVE_INDEX}, 2")
        asm.inst("s_cbranch_scc0 .LGroupedQ2KMetadataLoadDone")
        asm.inst(f"v_lshlrev_b32 v{metadata}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{metadata}, v{metadata}, v{serial}")
        asm.inst(f"v_mul_lo_u32 v{metadata}, {row_stride}, v{metadata}")
        asm.inst(f"v_add_nc_u32 v{metadata}, s{self.SCALAR_TEMPORARY}, v{metadata}")
        asm.inst(
            f"global_load_b128 v[{staged + 12}:{staged + 15}], v{metadata}, "
            f"s[{weights}:{weights + 1}] offset:{scales_offset}"
        )
        asm.inst(
            f"global_load_b32 v{staged + 16}, v{metadata}, "
            f"s[{weights}:{weights + 1}] offset:{dm_offset}"
        )
        asm.label(".LGroupedQ2KMetadataLoadDone")
        asm.inst("s_waitcnt vmcnt(0)")

        for item in range(8):
            raw = staged + item
            decoded = staged + 8
            if item:
                asm.inst(
                    f"v_add_nc_u32 v{lds_address}, {8 * layout.weight_row_stride}, v{lds_address}"
                )
            for shift, target in (
                (0, decoded),
                (2, decoded + 1),
                (4, decoded + 2),
                (6, decoded + 3),
            ):
                if shift == 0:
                    asm.inst(f"v_and_b32 v{target}, 0x03030303, v{raw}")
                else:
                    asm.inst(f"v_lshrrev_b32 v{target}, {shift}, v{raw}")
                    asm.inst(f"v_and_b32 v{target}, 0x03030303, v{target}")
                if not write2_decode:
                    asm.inst(f"ds_write_b32 v{lds_address}, v{target}")
                    if target != decoded + 3:
                        asm.inst(f"v_add_nc_u32 v{lds_address}, 32, v{lds_address}")
            if write2_decode:
                asm.inst(
                    f"ds_write2_b32 v{lds_address}, "
                    f"v{decoded}, v{decoded + 1} offset0:0 offset1:8"
                )
                asm.inst(
                    f"ds_write2_b32 v{lds_address}, "
                    f"v{decoded + 2}, v{decoded + 3} offset0:16 offset1:24"
                )
            elif item != 7:
                asm.inst(f"v_sub_nc_u32 v{lds_address}, v{lds_address}, {32 * 3}")

        asm.inst(f"s_cmp_lt_u32 s{self.WAVE_INDEX}, 2")
        asm.inst("s_cbranch_scc0 .LGroupedQ2KMetadataDone")
        asm.inst(f"v_mul_lo_u32 v{lds_address}, {layout.weight_row_stride}, v{serial}")
        asm.inst(f"v_add_nc_u32 v{lds_address}, {weight_base + 256}, v{lds_address}")
        if pre_negated_dm:
            asm.inst(f"v_pk_mul_f16 v{staged + 16}, 0xbc003c00, v{staged + 16}")
        if write2_metadata:
            for pair in range(0, 16, 2):
                for group, destination in (
                    (pair, decode + 2),
                    (pair + 1, auxiliary),
                ):
                    word = staged + 12 + group // 4
                    bit = 8 * (group % 4)
                    asm.inst(f"v_bfe_u32 v{decode}, v{word}, {bit}, 4")
                    asm.inst(f"v_bfe_u32 v{decode + 1}, v{word}, {bit + 4}, 4")
                    asm.inst(f"v_cvt_f16_u16_e32 v{destination}.l, v{decode}.l")
                    asm.inst(f"v_cvt_f16_u16_e32 v{destination}.h, v{decode + 1}.l")
                if not pre_negated_dm:
                    asm.inst(f"v_pk_mul_f16 v{decode + 2}, 0xbc003c00, v{decode + 2}")
                    asm.inst(f"v_pk_mul_f16 v{auxiliary}, 0xbc003c00, v{auxiliary}")
                asm.inst(f"v_pk_mul_f16 v{decode + 2}, v{staged + 16}, v{decode + 2}")
                asm.inst(f"v_pk_mul_f16 v{auxiliary}, v{staged + 16}, v{auxiliary}")
                asm.inst(
                    f"ds_write2_b32 v{lds_address}, v{decode + 2}, v{auxiliary} "
                    f"offset0:{pair} offset1:{pair + 1}"
                )
        else:
            for group in range(16):
                word = staged + 12 + group // 4
                bit = 8 * (group % 4)
                asm.inst(f"v_bfe_u32 v{decode}, v{word}, {bit}, 4")
                asm.inst(f"v_bfe_u32 v{decode + 1}, v{word}, {bit + 4}, 4")
                asm.inst(f"v_cvt_f16_u16_e32 v{decode + 2}.l, v{decode}.l")
                asm.inst(f"v_cvt_f16_u16_e32 v{decode + 2}.h, v{decode + 1}.l")
                if not pre_negated_dm:
                    asm.inst(f"v_pk_mul_f16 v{decode + 2}, 0xbc003c00, v{decode + 2}")
                asm.inst(f"v_pk_mul_f16 v{decode + 2}, v{staged + 16}, v{decode + 2}")
                asm.inst(
                    f"ds_write_b32 v{lds_address}, v{decode + 2} offset:{4 * group}"
                )
        asm.label(".LGroupedQ2KMetadataDone")

    def _emit_q2_distributed_weight_decode_stage(self, asm: Assembly) -> None:
        """Distribute Q2 payload and metadata conversion across all four waves."""
        physical = self._physical_plan()
        layout = physical.layout
        registers = physical.registers
        weights = physical.scalar_registers.weights.first_register
        state = self.context.state
        staged = registers.staged_payload.first_register
        decode = registers.decode_scratch.first_register
        decoded = decode + 8
        serial = registers.serial.first_register
        payload_address = registers.temporary.first_register
        scale_address = registers.auxiliary.first_register
        dm_address = scale_address + 1
        scratch = registers.metadata_lds_address.first_register
        payload_lds = registers.lds_address.first_register
        row_stride = state.packed_weight_row_bytes
        decoded_row_stride = layout.weight_row_stride
        pair_metadata_rows = 8 * decoded_row_stride % 256 == 0
        weight_base = layout.weight_data_base

        asm.comment(
            "Queue eight distributed Q2 payload/scale/dm rows before conversion."
        )
        asm.inst(f"v_lshrrev_b32 v{payload_address}, 4, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{scratch}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{payload_address}, v{scratch}, v{payload_address}")
        asm.inst(f"v_mul_lo_u32 v{payload_address}, {row_stride}, v{payload_address}")
        asm.inst(
            f"v_add_nc_u32 v{payload_address}, s{self.SCALAR_TEMPORARY}, "
            f"v{payload_address}"
        )
        asm.inst(f"v_mov_b32 v{scale_address}, v{payload_address}")
        asm.inst(f"v_mov_b32 v{dm_address}, v{payload_address}")
        asm.inst(f"v_and_b32 v{scratch}, 15, v{serial}")
        asm.inst(f"v_add_nc_u32 v{scale_address}, v{scratch}, v{scale_address}")
        asm.inst(f"v_lshlrev_b32 v{scratch}, 2, v{scratch}")
        asm.inst(f"v_add3_u32 v{payload_address}, 16, v{scratch}, v{payload_address}")
        asm.inst(f"v_add_nc_u32 v{dm_address}, 80, v{dm_address}")

        for item in range(8):
            asm.inst(
                f"global_load_b32 v{staged + item}, v{payload_address}, "
                f"s[{weights}:{weights + 1}]"
            )
            asm.inst(
                f"global_load_u8 v{staged + 8 + item}, v{scale_address}, "
                f"s[{weights}:{weights + 1}]"
            )
            asm.inst(
                f"global_load_b32 v{staged + 16 + item}, v{dm_address}, "
                f"s[{weights}:{weights + 1}]"
            )
            if item != 7:
                advance = 8 * row_stride
                asm.inst(
                    f"v_add_nc_u32 v{payload_address}, {advance}, v{payload_address}"
                )
                asm.inst(f"v_add_nc_u32 v{scale_address}, {advance}, v{scale_address}")
                asm.inst(f"v_add_nc_u32 v{dm_address}, {advance}, v{dm_address}")

        asm.inst(f"v_lshrrev_b32 v{payload_lds}, 4, v{serial}")
        asm.inst(f"v_mul_lo_u32 v{payload_lds}, {decoded_row_stride}, v{payload_lds}")
        asm.inst(f"v_and_b32 v{scratch}, 15, v{serial}")
        asm.inst(f"v_lshrrev_b32 v{scale_address}, 3, v{scratch}")
        asm.inst(f"v_lshlrev_b32 v{scale_address}, 7, v{scale_address}")
        asm.inst(f"v_and_b32 v{dm_address}, 7, v{scratch}")
        asm.inst(f"v_lshlrev_b32 v{dm_address}, 2, v{dm_address}")
        asm.inst(
            f"v_add3_u32 v{payload_lds}, v{scale_address}, v{dm_address}, "
            f"v{payload_lds}"
        )
        asm.inst(f"v_add_nc_u32 v{payload_lds}, {weight_base}, v{payload_lds}")

        asm.inst(f"v_lshrrev_b32 v{scratch}, 4, v{serial}")
        asm.inst(f"v_mul_lo_u32 v{scratch}, {decoded_row_stride}, v{scratch}")
        asm.inst(f"v_and_b32 v{scale_address}, 15, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{scale_address}, 2, v{scale_address}")
        asm.inst(
            f"v_add3_u32 v{scratch}, {weight_base + 256}, v{scale_address}, v{scratch}"
        )

        for item in range(8):
            asm.inst(f"s_waitcnt vmcnt({21 - 3 * item})")
            if item:
                row_advance = 8 * decoded_row_stride
                asm.inst(f"v_add_nc_u32 v{payload_lds}, {row_advance}, v{payload_lds}")
                if pair_metadata_rows:
                    if item % 2 == 0:
                        asm.inst(
                            f"v_add_nc_u32 v{scratch}, {2 * row_advance}, v{scratch}"
                        )
                else:
                    asm.inst(f"v_add_nc_u32 v{scratch}, {row_advance}, v{scratch}")
            raw = staged + item
            for shift, target in (
                (0, decoded),
                (2, decoded + 1),
                (4, decoded + 2),
                (6, decoded + 3),
            ):
                if shift == 0:
                    asm.inst(f"v_and_b32 v{target}, 0x03030303, v{raw}")
                else:
                    asm.inst(f"v_lshrrev_b32 v{target}, {shift}, v{raw}")
                    asm.inst(f"v_and_b32 v{target}, 0x03030303, v{target}")
            asm.inst(
                f"ds_write2_b32 v{payload_lds}, v{decoded}, v{decoded + 1} "
                "offset0:0 offset1:8"
            )
            asm.inst(
                f"ds_write2_b32 v{payload_lds}, v{decoded + 2}, v{decoded + 3} "
                "offset0:16 offset1:24"
            )

            scale = staged + 8 + item
            dm = staged + 16 + item
            metadata_value = decoded + 6 + (item % 2 if pair_metadata_rows else 0)
            asm.inst(f"v_and_b32 v{decoded + 4}, 15, v{scale}")
            asm.inst(f"v_lshrrev_b32 v{decoded + 5}, 4, v{scale}")
            asm.inst(f"v_cvt_f16_u16_e32 v{metadata_value}.l, v{decoded + 4}.l")
            asm.inst(f"v_cvt_f16_u16_e32 v{metadata_value}.h, v{decoded + 5}.l")
            asm.inst(f"v_pk_mul_f16 v{dm}, 0xbc003c00, v{dm}")
            asm.inst(f"v_pk_mul_f16 v{metadata_value}, v{dm}, v{metadata_value}")
            if pair_metadata_rows:
                if item % 2:
                    asm.inst(
                        f"ds_write2st64_b32 v{scratch}, v{decoded + 6}, "
                        f"v{decoded + 7} offset0:0 "
                        f"offset1:{8 * decoded_row_stride // 256}"
                    )
            else:
                asm.inst(f"ds_write_b32 v{scratch}, v{metadata_value}")

    def _emit_q2_group_dispatch(self, asm: Assembly, *, group_base: int) -> None:
        policy = self.context.state.kernel_spec.row_dispatch

        def emit_body(target: Assembly, row_tile_rows: int, label_suffix: str) -> None:
            self._emit_q2_group_loop(
                target,
                group_base=group_base,
                row_tiles=row_tile_rows // 16,
                label_suffix=label_suffix,
            )

        self._row_dispatch().emit(
            asm,
            GroupedRowDispatchLabels.q2_mma(policy, group_base),
            emit_body,
        )

    def _emit_q2_group_loop(
        self,
        asm: Assembly,
        *,
        group_base: int,
        row_tiles: int | None = None,
        label_suffix: str = "",
    ) -> None:
        row_tiles = row_tiles or self._row_tile_count()
        policy = self._q2_decode_policy()
        if policy.unrolled_groups:
            for local_group in range(8):
                self._emit_q2_static_group(
                    asm,
                    group_base=group_base,
                    local_group=local_group,
                    hip_association=policy.hip_association,
                    partial_lds=policy.partial_lds,
                    row_tiles=row_tiles,
                )
            return

        physical = self._physical_plan()
        registers = physical.registers
        layout = physical.layout
        activation_metadata = cast(
            F16D2S6ActivationMetadata, layout.activation_metadata
        )
        weight_q = registers.weight_payload.first_register
        c_base = registers.c_fragments.first_register
        activation_scale_sum = registers.activation_scale_sum.first_register
        scaled_dm = registers.scaled_dm.first_register
        lds_address = registers.lds_address.first_register
        activation_base = registers.activation_base.first_register
        weight_base = registers.output_column.first_register
        metadata_base = registers.metadata_lds_address.first_register
        asm.inst(f"s_mov_b32 s{self.GROUP_LOOP}, 0")
        label = f".LGroupedQ2KGroupLoop{group_base}{label_suffix}"
        asm.label(label)
        asm.inst(f"s_lshl_b32 s{self.GROUP_OFFSET}, s{self.GROUP_LOOP}, 4")
        asm.inst(f"v_add_nc_u32 v{lds_address}, {16 * group_base}, v{weight_base}")
        asm.inst(f"v_add_nc_u32 v{lds_address}, s{self.GROUP_OFFSET}, v{lds_address}")
        asm.inst(f"ds_read_b128 v[{weight_q}:{weight_q + 3}], v{lds_address}")

        asm.inst(
            f"v_add_nc_u32 v{lds_address}, s{self.GROUP_OFFSET}, v{activation_base}"
        )
        for tile in range(row_tiles):
            activation = (
                c_base + 8 * (tile + 1)
                if tile < row_tiles - 1
                else registers.low_activation_tail.first_register
            )
            asm.inst(
                f"ds_read_b128 v[{activation}:{activation + 3}], v{lds_address} "
                f"offset:{activation_metadata.PAYLOAD_BASE + 16 * tile * activation_metadata.block_bytes}"
            )

        # Both activation scales live in the first dword. The runtime group
        # selects its half after all LDS reads have completed.
        for tile in range(row_tiles):
            scale_reg = activation_scale_sum + tile
            asm.inst(
                f"ds_read_b32 v{scale_reg}, v{activation_base} "
                f"offset:{16 * tile * activation_metadata.block_bytes}"
            )

        # Sum dwords start at byte four. Groups 6-7 intentionally read the
        # first payload dword; their branch ignores the packed high half and
        # reconstructs the sum with an all-ones WMMA.
        asm.inst(f"s_lshr_b32 s{self.GROUP_OFFSET}, s{self.GROUP_LOOP}, 1")
        asm.inst(f"s_lshl_b32 s{self.GROUP_OFFSET}, s{self.GROUP_OFFSET}, 2")
        asm.inst(f"s_add_u32 s{self.GROUP_OFFSET}, s{self.GROUP_OFFSET}, 4")
        asm.inst(
            f"v_add_nc_u32 v{lds_address}, s{self.GROUP_OFFSET}, v{activation_base}"
        )
        for tile in range(row_tiles):
            asm.inst(
                f"ds_read_b32 v{scaled_dm + tile}, v{lds_address} "
                f"offset:{16 * tile * activation_metadata.block_bytes}"
            )
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst(f"s_lshr_b32 s{self.GROUP_OFFSET}, s{self.GROUP_LOOP}, 2")
        asm.inst(f"s_lshl_b32 s{self.GROUP_OFFSET}, s{self.GROUP_OFFSET}, 4")
        for tile in range(row_tiles):
            asm.inst(
                f"v_lshrrev_b32 v{activation_scale_sum + tile}, "
                f"s{self.GROUP_OFFSET}, v{activation_scale_sum + tile}"
            )
            asm.inst(
                f"v_and_b32 v{activation_scale_sum + tile}, 0x0000ffff, "
                f"v{activation_scale_sum + tile}"
            )
        asm.inst(f"s_and_b32 s{self.GROUP_OFFSET}, s{self.GROUP_LOOP}, 1")
        asm.inst(f"s_lshl_b32 s{self.GROUP_OFFSET}, s{self.GROUP_OFFSET}, 4")
        for tile in range(row_tiles):
            asm.inst(
                f"v_lshrrev_b32 v{scaled_dm + tile}, s{self.GROUP_OFFSET}, "
                f"v{scaled_dm + tile}"
            )
            asm.inst(f"v_and_b32 v{scaled_dm + tile}, 0x0000ffff, v{scaled_dm + tile}")
            asm.inst(
                f"v_lshl_or_b32 v{activation_scale_sum + tile}, "
                f"v{scaled_dm + tile}, 16, v{activation_scale_sum + tile}"
            )

        asm.inst(f"s_lshl_b32 s{self.GROUP_OFFSET}, s{self.GROUP_LOOP}, 2")
        asm.inst(f"v_add_nc_u32 v{lds_address}, {4 * group_base}, v{metadata_base}")
        asm.inst(
            f"v_add_nc_u32 v{registers.metadata_addresses.first_register}, "
            f"s{self.GROUP_OFFSET}, v{lds_address}"
        )
        for pair in range(1, 4):
            asm.inst(
                f"v_add_nc_u32 v{registers.metadata_addresses.first_register + pair}, "
                f"{4 * layout.weight_row_stride * pair}, "
                f"v{registers.metadata_addresses.first_register}"
            )
        for element in range(0, 8, 2):
            asm.inst(
                f"ds_read2_b32 v[{scaled_dm + element}:{scaled_dm + element + 1}], "
                f"v{registers.metadata_addresses.first_register + element // 2} "
                f"offset0:0 offset1:{layout.weight_row_stride // 2}"
            )
        asm.inst("s_waitcnt lgkmcnt(0)")
        self._emit_q2_wmma_path(
            asm,
            row_tiles=row_tiles,
            label_suffix=f"{group_base}{label_suffix}",
        )
        asm.inst(f"s_add_u32 s{self.GROUP_LOOP}, s{self.GROUP_LOOP}, 1")
        asm.inst(f"s_cmp_lt_u32 s{self.GROUP_LOOP}, 8")
        asm.inst(f"s_cbranch_scc1 {label}")

    def _emit_q2_static_group(
        self,
        asm: Assembly,
        *,
        group_base: int,
        local_group: int,
        hip_association: bool,
        partial_lds: bool = False,
        row_tiles: int | None = None,
    ) -> None:
        physical = self._physical_plan()
        registers = physical.registers
        layout = physical.layout
        activation_metadata = cast(
            F16D2S6ActivationMetadata, layout.activation_metadata
        )
        weight_q = registers.weight_payload.first_register
        c_base = registers.c_fragments.first_register
        activation_scale_sum = registers.activation_scale_sum.first_register
        scaled_dm = registers.scaled_dm.first_register
        lds_address = registers.lds_address.first_register
        activation_base = registers.activation_base.first_register
        weight_base = registers.output_column.first_register
        metadata_base = registers.metadata_lds_address.first_register
        row_tiles = row_tiles or self._row_tile_count()
        group = group_base + local_group
        if partial_lds:
            self._emit_q2_static_group_partial_lds(
                asm,
                group=group,
                local_group=local_group,
                hip_association=hip_association,
                row_tiles=row_tiles,
            )
            return

        asm.comment(f"Statically lowered Q2_K group {group}.")
        asm.inst(f"v_add_nc_u32 v{lds_address}, {16 * group}, v{weight_base}")
        asm.inst(f"ds_read_b128 v[{weight_q}:{weight_q + 3}], v{lds_address}")
        for tile in range(row_tiles):
            activation = (
                c_base + 8 * (tile + 1)
                if tile < row_tiles - 1
                else registers.low_activation_tail.first_register
            )
            asm.inst(
                f"ds_read_b128 v[{activation}:{activation + 3}], "
                f"v{activation_base} offset:"
                f"{activation_metadata.payload_offset(local_group) + 16 * tile * activation_metadata.block_bytes}"
            )
            asm.inst(
                f"ds_read_b32 v{activation_scale_sum + tile}, "
                f"v{activation_base} "
                f"offset:{16 * tile * activation_metadata.block_bytes}"
            )
            if local_group < 6:
                sum_word_offset = 4 + 4 * (local_group // 2)
                asm.inst(
                    f"ds_read_b32 v{scaled_dm + tile}, v{activation_base} "
                    f"offset:{sum_word_offset + 16 * tile * activation_metadata.block_bytes}"
                )
        asm.inst("s_waitcnt lgkmcnt(0)")
        for tile in range(row_tiles):
            scale = activation_scale_sum + tile
            if local_group >= 4:
                asm.inst(f"v_lshrrev_b32 v{scale}, 16, v{scale}")
            asm.inst(f"v_and_b32 v{scale}, 0x0000ffff, v{scale}")
            if local_group < 6:
                activation_sum = scaled_dm + tile
                if local_group % 2:
                    asm.inst(f"v_lshrrev_b32 v{activation_sum}, 16, v{activation_sum}")
                asm.inst(f"v_and_b32 v{activation_sum}, 0x0000ffff, v{activation_sum}")
                asm.inst(f"v_lshl_or_b32 v{scale}, v{activation_sum}, 16, v{scale}")

        metadata_address = registers.metadata_addresses.first_register
        asm.inst(f"v_add_nc_u32 v{metadata_address}, {4 * group}, v{metadata_base}")
        for pair in range(1, 4):
            asm.inst(
                f"v_add_nc_u32 v{metadata_address + pair}, "
                f"{4 * layout.weight_row_stride * pair}, v{metadata_address}"
            )
        for element in range(0, 8, 2):
            asm.inst(
                f"ds_read2_b32 v[{scaled_dm + element}:{scaled_dm + element + 1}], "
                f"v{metadata_address + element // 2} offset0:0 "
                f"offset1:{layout.weight_row_stride // 2}"
            )
        asm.inst("s_waitcnt lgkmcnt(0)")
        self._emit_q2_static_wmma_path(
            asm,
            row_tiles=row_tiles,
            missing_sum=local_group >= 6,
            hip_association=hip_association,
        )

    def _emit_q2_static_group_partial_lds(
        self,
        asm: Assembly,
        *,
        group: int,
        local_group: int,
        hip_association: bool,
        row_tiles: int,
    ) -> None:
        physical = self._physical_plan()
        registers = physical.registers
        layout = physical.layout
        activation_metadata = cast(
            F16D2S6ActivationMetadata, layout.activation_metadata
        )
        weight_q = registers.weight_payload.first_register
        c_base = registers.c_fragments.first_register
        activation_scale_sum = registers.activation_scale_sum.first_register
        activation_sum = registers.high_activation.first_register
        scaled_dm = registers.scaled_dm.first_register
        lds_address = registers.lds_address.first_register
        activation_base = registers.activation_base.first_register
        weight_base = registers.output_column.first_register
        metadata_base = registers.metadata_lds_address.first_register

        asm.comment(f"Partial-LDS Q2_K group {group}.")
        for tile in range(row_tiles):
            asm.inst(
                f"ds_read_b32 v{activation_scale_sum + tile}, "
                f"v{activation_base} "
                f"offset:{16 * tile * activation_metadata.block_bytes}"
            )
            if local_group < 6:
                sum_word_offset = 4 + 4 * (local_group // 2)
                asm.inst(
                    f"ds_read_b32 v{activation_sum + tile}, v{activation_base} "
                    f"offset:{sum_word_offset + 16 * tile * activation_metadata.block_bytes}"
                )

        asm.inst(f"v_add_nc_u32 v{lds_address}, {16 * group}, v{weight_base}")
        asm.inst(f"ds_read_b128 v[{weight_q}:{weight_q + 3}], v{lds_address}")
        for tile in range(row_tiles):
            activation = (
                c_base + 8 * (tile + 1)
                if tile < row_tiles - 1
                else registers.low_activation_tail.first_register
            )
            asm.inst(
                f"ds_read_b128 v[{activation}:{activation + 3}], "
                f"v{activation_base} offset:"
                f"{activation_metadata.payload_offset(local_group) + 16 * tile * activation_metadata.block_bytes}"
            )

        metadata_address = registers.metadata_addresses.first_register
        asm.inst(f"v_add_nc_u32 v{metadata_address}, {4 * group}, v{metadata_base}")
        for pair in range(1, 4):
            asm.inst(
                f"v_add_nc_u32 v{metadata_address + pair}, "
                f"{4 * layout.weight_row_stride * pair}, v{metadata_address}"
            )
        for element in range(0, 8, 2):
            asm.inst(
                f"ds_read2_b32 v[{scaled_dm + element}:{scaled_dm + element + 1}], "
                f"v{metadata_address + element // 2} offset0:0 "
                f"offset1:{layout.weight_row_stride // 2}"
            )

        # The first five or seven reads provide scales, sums, and WMMA
        # payloads. Keep the four trailing d/dmin reads in flight.
        asm.inst("s_waitcnt lgkmcnt(4)")
        for tile in range(row_tiles):
            scale = activation_scale_sum + tile
            if local_group >= 4:
                asm.inst(f"v_lshrrev_b32 v{scale}, 16, v{scale}")
            asm.inst(f"v_and_b32 v{scale}, 0x0000ffff, v{scale}")
            if local_group < 6:
                total = activation_sum + tile
                if local_group % 2:
                    asm.inst(f"v_lshrrev_b32 v{total}, 16, v{total}")
                asm.inst(f"v_and_b32 v{total}, 0x0000ffff, v{total}")
                asm.inst(f"v_lshl_or_b32 v{scale}, v{total}, 16, v{scale}")

        self._emit_q2_static_wmma_path(
            asm,
            row_tiles=row_tiles,
            missing_sum=local_group >= 6,
            hip_association=hip_association,
            partial_lds=True,
        )

    def _emit_q2_static_wmma_path(
        self,
        asm: Assembly,
        *,
        row_tiles: int,
        missing_sum: bool,
        hip_association: bool,
        partial_lds: bool = False,
    ) -> None:
        registers = self._physical_plan().registers
        c_base = registers.c_fragments.first_register
        weight_q = registers.weight_payload.first_register
        zero = registers.zero_accumulator.first_register
        low_last = registers.low_activation_tail.first_register
        high = registers.high_activation.first_register
        scale_sum = registers.activation_scale_sum.first_register
        scaled_dm = registers.scaled_dm.first_register
        sums = registers.sums.first_register

        if not missing_sum:
            for tile in range(row_tiles):
                activation = (
                    c_base + 8 * (tile + 1) if tile < row_tiles - 1 else low_last
                )
                if not partial_lds:
                    asm.inst("s_waitcnt lgkmcnt(0)")
                emit_signed_i8_wmma(
                    asm,
                    destination=c_base + 8 * tile,
                    weight=weight_q,
                    activation=activation,
                    accumulator=zero,
                    clamp=True,
                )
            if partial_lds:
                asm.inst("s_waitcnt lgkmcnt(0)")
            for tile in range(row_tiles):
                c_fragment = c_base + 8 * tile
                if hip_association:
                    for element in range(8):
                        asm.inst(
                            f"v_cvt_f32_i32 v{c_fragment + element}, "
                            f"v{c_fragment + element}"
                        )
                    for element in range(8):
                        asm.inst(
                            f"v_fma_mix_f32 v{low_last + element}, "
                            f"v{c_fragment + element}, v{scaled_dm + element}, "
                            f"0 op_sel_hi:[0,1,0]"
                        )
                    for element in range(8):
                        asm.inst(
                            f"v_fma_mix_f32 v{sums + 8 * tile + element}, "
                            f"v{low_last + element}, v{scale_sum + tile}, "
                            f"v{sums + 8 * tile + element} "
                            f"op_sel_hi:[0,1,0]"
                        )
                    for element in range(8):
                        asm.inst(
                            f"v_fma_mix_f32 v{sums + 8 * tile + element}, "
                            f"v{scaled_dm + element}, v{scale_sum + tile}, "
                            f"v{sums + 8 * tile + element} op_sel:[1,1,0] "
                            f"op_sel_hi:[1,1,0]"
                        )
                    continue
                for element in range(8):
                    product = low_last + element
                    asm.inst(
                        f"v_cvt_f32_i32 v{c_fragment + element}, "
                        f"v{c_fragment + element}"
                    )
                    asm.inst(
                        f"v_fma_mix_f32 v{product}, v{scaled_dm + element}, "
                        f"v{scale_sum + tile}, 0 op_sel_hi:[1,1,0]"
                    )
                    asm.inst(
                        f"v_fma_f32 v{sums + 8 * tile + element}, "
                        f"v{product}, v{c_fragment + element}, "
                        f"v{sums + 8 * tile + element}"
                    )
                    asm.inst(
                        f"v_fma_mix_f32 v{sums + 8 * tile + element}, "
                        f"v{scaled_dm + element}, v{scale_sum + tile}, "
                        f"v{sums + 8 * tile + element} op_sel:[1,1,0] "
                        f"op_sel_hi:[1,1,0]"
                    )
            return

        correction_temporary = (
            registers.staged_payload.first_register + 42 if row_tiles <= 2 else high + 8
        )
        for tile in range(row_tiles):
            activation = c_base + 8 * (tile + 1) if tile < row_tiles - 1 else low_last
            emit_signed_i8_wmma(
                asm,
                destination=c_base + 8 * tile,
                weight=weight_q,
                activation=activation,
                accumulator=zero,
                clamp=True,
            )
            emit_signed_i8_wmma(
                asm,
                destination=high,
                weight=weight_q + 4,
                activation=activation,
                accumulator=zero,
                clamp=True,
            )
            if partial_lds and tile == 0:
                asm.inst("s_waitcnt lgkmcnt(0)")
            if hip_association:
                for element in range(8):
                    asm.inst(
                        f"v_cvt_f32_i32 v{c_base + 8 * tile + element}, "
                        f"v{c_base + 8 * tile + element}"
                    )
                for element in range(8):
                    asm.inst(f"v_cvt_f32_i32 v{high + element}, v{high + element}")
                for element in range(8):
                    asm.inst(
                        f"v_fma_mix_f32 v{correction_temporary + element}, "
                        f"v{c_base + 8 * tile + element}, "
                        f"v{scaled_dm + element}, 0 op_sel_hi:[0,1,0]"
                    )
                for element in range(8):
                    asm.inst(
                        f"v_fma_mix_f32 v{correction_temporary + element}, "
                        f"v{high + element}, v{scaled_dm + element}, "
                        f"v{correction_temporary + element} "
                        f"op_sel:[0,1,0] op_sel_hi:[0,1,0]"
                    )
                for element in range(8):
                    asm.inst(
                        f"v_fma_mix_f32 v{sums + 8 * tile + element}, "
                        f"v{correction_temporary + element}, "
                        f"v{scale_sum + tile}, "
                        f"v{sums + 8 * tile + element} op_sel_hi:[0,1,0]"
                    )
                for element in range(8):
                    asm.inst(
                        f"v_fma_mix_f32 v{sums + 8 * tile + element}, "
                        f"v{scaled_dm + element}, v{scale_sum + tile}, "
                        f"v{sums + 8 * tile + element} "
                        f"op_sel:[1,1,0] op_sel_hi:[1,1,0]"
                    )
                continue
            for element in range(8):
                c_fragment = c_base + 8 * tile + element
                ones = high + element
                temporary = registers.temporary.first_register
                asm.inst(f"v_cvt_f32_i32 v{c_fragment}, v{c_fragment}")
                asm.inst(f"v_cvt_f32_i32 v{ones}, v{ones}")
                asm.inst(
                    f"v_fma_mix_f32 v{temporary}, v{scaled_dm + element}, "
                    f"v{scale_sum + tile}, 0 op_sel:[1,0,0] "
                    f"op_sel_hi:[1,1,0]"
                )
                asm.inst(
                    f"v_fma_f32 v{sums + 8 * tile + element}, v{temporary}, "
                    f"v{ones}, v{sums + 8 * tile + element}"
                )
                asm.inst(
                    f"v_fma_mix_f32 v{temporary}, v{scaled_dm + element}, "
                    f"v{scale_sum + tile}, 0 op_sel_hi:[1,1,0]"
                )
                asm.inst(
                    f"v_fma_f32 v{sums + 8 * tile + element}, v{temporary}, "
                    f"v{c_fragment}, v{sums + 8 * tile + element}"
                )

    def _emit_q2_wmma_path(
        self, asm: Assembly, *, row_tiles: int, label_suffix: str
    ) -> None:
        registers = self._physical_plan().registers
        c_base = registers.c_fragments.first_register
        weight_q = registers.weight_payload.first_register
        zero = registers.zero_accumulator.first_register
        low_last = registers.low_activation_tail.first_register
        high = registers.high_activation.first_register
        scale_sum = registers.activation_scale_sum.first_register
        scaled_dm = registers.scaled_dm.first_register
        sums = registers.sums.first_register
        product_base = low_last
        missing = f".LGroupedQ2KMissingSum{label_suffix}"
        done = f".LGroupedQ2KStoredSum{label_suffix}"
        asm.inst(f"s_cmp_ge_u32 s{self.GROUP_LOOP}, 6")
        asm.inst(f"s_cbranch_scc1 {missing}")
        for tile in range(row_tiles):
            activation = c_base + 8 * (tile + 1) if tile < row_tiles - 1 else low_last
            asm.inst("s_waitcnt lgkmcnt(0)")
            emit_signed_i8_wmma(
                asm,
                destination=c_base + 8 * tile,
                weight=weight_q,
                activation=activation,
                accumulator=zero,
                clamp=True,
            )
        for tile in range(row_tiles):
            c_fragment = c_base + 8 * tile
            for element in range(8):
                product = product_base + element
                asm.inst(
                    f"v_cvt_f32_i32 v{c_fragment + element}, v{c_fragment + element}"
                )
                asm.inst(
                    f"v_fma_mix_f32 v{product}, v{scaled_dm + element}, "
                    f"v{scale_sum + tile}, 0 op_sel_hi:[1,1,0]"
                )
                asm.inst(
                    f"v_fma_f32 v{sums + 8 * tile + element}, v{product}, "
                    f"v{c_fragment + element}, v{sums + 8 * tile + element}"
                )
                asm.inst(
                    f"v_fma_mix_f32 v{sums + 8 * tile + element}, "
                    f"v{scaled_dm + element}, v{scale_sum + tile}, "
                    f"v{sums + 8 * tile + element} op_sel:[1,1,0] "
                    f"op_sel_hi:[1,1,0]"
                )
        asm.inst(f"s_branch {done}")
        asm.label(missing)
        for tile in range(row_tiles):
            activation = c_base + 8 * (tile + 1) if tile < row_tiles - 1 else low_last
            emit_signed_i8_wmma(
                asm,
                destination=c_base + 8 * tile,
                weight=weight_q,
                activation=activation,
                accumulator=zero,
                clamp=True,
            )
            emit_signed_i8_wmma(
                asm,
                destination=high,
                weight=weight_q + 4,
                activation=activation,
                accumulator=zero,
                clamp=True,
            )
            for element in range(8):
                c_fragment = c_base + 8 * tile + element
                ones = high + element
                asm.inst(f"v_cvt_f32_i32 v{c_fragment}, v{c_fragment}")
                asm.inst(f"v_cvt_f32_i32 v{ones}, v{ones}")
                asm.inst(
                    f"v_fma_mix_f32 v{registers.temporary.first_register}, "
                    f"v{scaled_dm + element}, v{scale_sum + tile}, 0 "
                    f"op_sel:[1,0,0] op_sel_hi:[1,1,0]"
                )
                asm.inst(
                    f"v_fma_f32 v{sums + 8 * tile + element}, "
                    f"v{registers.temporary.first_register}, v{ones}, "
                    f"v{sums + 8 * tile + element}"
                )
                asm.inst(
                    f"v_fma_mix_f32 v{registers.temporary.first_register}, "
                    f"v{scaled_dm + element}, v{scale_sum + tile}, 0 "
                    f"op_sel_hi:[1,1,0]"
                )
                asm.inst(
                    f"v_fma_f32 v{sums + 8 * tile + element}, "
                    f"v{registers.temporary.first_register}, v{c_fragment}, "
                    f"v{sums + 8 * tile + element}"
                )
        asm.label(done)


def grouped_q2_decoded_lowering(
    context: GroupedForwardLoweringContext,
) -> GroupedQ2KDecodedWeightLdsLowering:
    return GroupedQ2KDecodedWeightLdsLowering(context)
