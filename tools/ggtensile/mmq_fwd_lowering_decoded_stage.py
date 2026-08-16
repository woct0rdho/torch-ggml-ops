"""Shared decoded-weight LDS staging and MMA emission."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .kernel_writer_assembly import Assembly, RegisterAssignment
from .mmq_fwd_lowering_metadata import emit_packed_scale_minimum
from .mmq_fwd_lowering_mma import emit_signed_i8_wmma
from .mmq_fwd_physical import DecodedWeightLdsRegisterPlan
from .mmq_fwd_spec import DecodedLdsLayout, QuantForwardSemantics


class DecodedSchedulePolicy(Protocol):
    @property
    def independent_metadata_extraction(self) -> bool: ...

    @property
    def defer_metadata_reads(self) -> bool: ...


class DecodedStageScalarRegisters(Protocol):
    @property
    def weights(self) -> RegisterAssignment: ...

    @property
    def output(self) -> RegisterAssignment: ...

    @property
    def packed_block_offset(self) -> RegisterAssignment: ...

    @property
    def wave(self) -> RegisterAssignment: ...

    @property
    def group_loop(self) -> RegisterAssignment: ...

    @property
    def group_offset(self) -> RegisterAssignment: ...


@dataclass(frozen=True)
class DecodedWeightLdsStageInputs:
    quant_type: str
    packed_weight_row_bytes: int
    semantics: QuantForwardSemantics
    decode_policy: DecodedSchedulePolicy
    wmma_clamp: bool
    layout: DecodedLdsLayout
    registers: DecodedWeightLdsRegisterPlan
    scalar_registers: DecodedStageScalarRegisters
    allocated_row_tiles: int


@dataclass(frozen=True)
class DecodedWeightLdsStageEmitter:
    inputs: DecodedWeightLdsStageInputs

    def emit_weight_decode_stage(
        self,
        asm: Assembly,
    ) -> None:
        decoded_lds = self.inputs.layout
        registers = self.inputs.registers
        scalar = self.inputs.scalar_registers
        packed_block_offset = scalar.packed_block_offset.first_register
        wave = scalar.wave.first_register
        weights = scalar.weights.first_register
        decode_scratch = registers.decode_scratch.first_register
        row_stride = self.inputs.packed_weight_row_bytes
        serial = registers.serial.first_register
        temporary = registers.temporary.first_register
        lds_address = registers.lds_address.first_register
        auxiliary = registers.auxiliary.first_register
        metadata_address = registers.metadata_lds_address.first_register
        staging_base = registers.staged_payload.first_register
        weight_lds_base = decoded_lds.weight_data_base
        weight_lds_stride = decoded_lds.weight_row_stride
        quant_type = self.inputs.quant_type
        quant_label = quant_type.replace("_", "")
        semantics = self.inputs.semantics
        high_bits = semantics.high_bit_reconstruction()
        ql_offset = semantics.payload_plane("ql").byte_offset
        qh_offset = (
            semantics.payload_plane(high_bits.plane_name).byte_offset
            if high_bits is not None
            else 0
        )
        qh_address = auxiliary + 1
        payload_reads_per_slice = 1 + (high_bits is not None)

        asm.comment(f"Cooperatively decode {quant_type} payload into padded LDS rows.")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 3, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{metadata_address}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{metadata_address}, v{temporary}")
        asm.inst(f"v_mul_lo_u32 v{temporary}, {row_stride}, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{temporary}, s{packed_block_offset}, v{temporary}")
        asm.inst(f"v_and_b32 v{auxiliary}, 7, v{serial}")
        if high_bits is not None:
            asm.comment("Build Q5_K high-bit and low-nibble payload addresses.")
            asm.inst(
                f"v_and_b32 v{qh_address}, {high_bits.address_lane_mask}, v{auxiliary}"
            )
            asm.inst(
                f"v_mad_u32_u24 v{qh_address}, {high_bits.address_lane_stride}, "
                f"v{qh_address}, v{temporary}"
            )
            asm.inst(f"v_mad_u32_u24 v{temporary}, 16, v{auxiliary}, v{temporary}")
        else:
            asm.inst(f"v_lshlrev_b32 v{metadata_address}, 4, v{auxiliary}")
            asm.inst(f"v_add_nc_u32 v{temporary}, 16, v{temporary}")
            asm.inst(f"v_add_nc_u32 v{temporary}, v{metadata_address}, v{temporary}")

        asm.inst(f"v_lshrrev_b32 v{lds_address}, 3, v{serial}")
        asm.inst(f"v_mul_lo_u32 v{lds_address}, {weight_lds_stride}, v{lds_address}")
        asm.inst(f"v_lshrrev_b32 v{metadata_address}, 1, v{auxiliary}")
        asm.inst(f"v_lshlrev_b32 v{metadata_address}, 6, v{metadata_address}")
        asm.inst(f"v_and_b32 v{auxiliary}, 1, v{auxiliary}")
        asm.inst(f"v_lshlrev_b32 v{auxiliary}, 4, v{auxiliary}")
        asm.inst(
            f"v_add3_u32 v{lds_address}, v{metadata_address}, "
            f"v{auxiliary}, v{lds_address}"
        )
        asm.inst(f"v_add_nc_u32 v{lds_address}, {weight_lds_base}, v{lds_address}")

        metadata_base = staging_base + 32
        asm.inst(f"s_cmp_lt_u32 s{wave}, 2")
        asm.inst(f"s_cbranch_scc0 .LForward{quant_label}DecodedMetadataLoadDone")
        asm.inst(f"v_lshlrev_b32 v{metadata_address}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{metadata_address}, v{serial}, v{metadata_address}")
        asm.inst(f"v_mul_lo_u32 v{metadata_address}, {row_stride}, v{metadata_address}")
        asm.inst(
            f"v_add_nc_u32 v{metadata_address}, s{packed_block_offset}, "
            f"v{metadata_address}"
        )
        asm.inst(
            f"global_load_b128 v[{metadata_base}:{metadata_base + 3}], "
            f"v{metadata_address}, s[{weights}:{weights + 1}]"
        )
        asm.label(f".LForward{quant_label}DecodedMetadataLoadDone")

        if high_bits is not None:
            qh_base = staging_base + 36
            for row_slice in range(4):
                raw_base = staging_base + 8 * row_slice
                row_qh_base = qh_base + 4 * row_slice
                asm.inst(
                    f"global_load_b128 v[{row_qh_base}:{row_qh_base + 3}], "
                    f"v{qh_address}, s[{weights}:{weights + 1}] "
                    f"offset:{qh_offset}"
                )
                asm.inst(
                    f"global_load_b128 v[{raw_base}:{raw_base + 3}], "
                    f"v{temporary}, s[{weights}:{weights + 1}] "
                    f"offset:{ql_offset}"
                )
                if row_slice != 3:
                    asm.inst(
                        f"v_add_nc_u32 v{qh_address}, {16 * row_stride}, v{qh_address}"
                    )
                    asm.inst(
                        f"v_add_nc_u32 v{temporary}, {16 * row_stride}, v{temporary}"
                    )
            asm.inst(f"v_and_b32 v{qh_address}, {high_bits.lane_shift_mask}, v{serial}")
        else:
            for row_slice in range(4):
                raw_base = staging_base + 8 * row_slice
                asm.inst(
                    f"global_load_b128 v[{raw_base}:{raw_base + 3}], "
                    f"v{temporary}, s[{weights}:{weights + 1}]"
                )
                if row_slice != 3:
                    asm.inst(
                        f"v_add_nc_u32 v{temporary}, {16 * row_stride}, v{temporary}"
                    )
        for row_slice in range(4):
            raw_base = staging_base + 8 * row_slice
            asm.inst(f"s_waitcnt vmcnt({payload_reads_per_slice * (3 - row_slice)})")
            if high_bits is not None:
                qh_base = staging_base + 36 + 4 * row_slice
                for item in range(4):
                    asm.inst(
                        f"v_lshrrev_b32 v{qh_base + item}, v{qh_address}, "
                        f"v{qh_base + item}"
                    )
                for item in range(4):
                    asm.inst(
                        f"v_lshrrev_b32 v{raw_base + 4 + item}, "
                        f"{high_bits.nibble_shift}, v{raw_base + item}"
                    )
                for item in range(4):
                    asm.inst(
                        f"v_and_b32 v{raw_base + item}, "
                        f"{high_bits.nibble_mask:#010x}, v{raw_base + item}"
                    )
                    asm.inst(
                        f"v_and_b32 v{raw_base + 4 + item}, "
                        f"{high_bits.nibble_mask:#010x}, "
                        f"v{raw_base + 4 + item}"
                    )
                for item in range(4):
                    low = raw_base + item
                    high = raw_base + 4 + item
                    qh = qh_base + item
                    asm.inst(
                        f"v_and_b32 v{auxiliary}, {high_bits.low_mask:#010x}, v{qh}"
                    )
                    asm.inst(
                        f"v_lshl_or_b32 v{low}, v{auxiliary}, "
                        f"{high_bits.low_destination_shift}, v{low}"
                    )
                    asm.inst(
                        f"v_and_b32 v{auxiliary}, {high_bits.high_mask:#010x}, v{qh}"
                    )
                    asm.inst(
                        f"v_lshl_or_b32 v{high}, v{auxiliary}, "
                        f"{high_bits.high_destination_shift}, v{high}"
                    )
            else:
                for item in range(4):
                    low = raw_base + item
                    high = raw_base + 4 + item
                    asm.inst(f"v_lshrrev_b32 v{high}, 4, v{low}")
                    asm.inst(f"v_and_b32 v{low}, 0x0f0f0f0f, v{low}")
                    asm.inst(f"v_and_b32 v{high}, 0x0f0f0f0f, v{high}")
            asm.inst(f"ds_write_b128 v{lds_address}, v[{raw_base}:{raw_base + 3}]")
            asm.inst(
                f"ds_write_b128 v{lds_address}, "
                f"v[{raw_base + 4}:{raw_base + 7}] offset:32"
            )
            if row_slice != 3:
                asm.inst(
                    f"v_add_nc_u32 v{lds_address}, "
                    f"{16 * weight_lds_stride}, v{lds_address}"
                )

        asm.comment(
            f"Compute each packed {quant_type} scale/min pair once per weight row."
        )
        asm.inst(f"s_cmp_lt_u32 s{wave}, 2")
        asm.inst(f"s_cbranch_scc0 .LForward{quant_label}DecodedMetadataDone")
        asm.inst(f"v_mul_lo_u32 v{lds_address}, {weight_lds_stride}, v{serial}")
        asm.inst(
            f"v_add_nc_u32 v{lds_address}, {weight_lds_base + 256}, v{lds_address}"
        )
        decode_policy = self.inputs.decode_policy
        if decode_policy.independent_metadata_extraction:
            # Expose the eight metadata fields before conversion so the
            # conversion/product chains do not serialize on v72:v77.
            for group in range(4):
                fields = semantics.packed_scale_minimum_fields(group)
                scale_part = fields.scale[0]
                minimum_part = fields.minimum[0]
                asm.inst(
                    f"v_bfe_u32 v{decode_scratch + group}, "
                    f"v{metadata_base + scale_part.metadata_word}, "
                    f"{scale_part.bit_offset}, {scale_part.bit_count}"
                )
                asm.inst(
                    f"v_bfe_u32 v{decode_scratch + 8 + group}, "
                    f"v{metadata_base + minimum_part.metadata_word}, "
                    f"{minimum_part.bit_offset}, {minimum_part.bit_count}"
                )
            upper_fields = tuple(
                semantics.packed_scale_minimum_fields(group) for group in range(4, 8)
            )
            for packed, fields in enumerate(upper_fields):
                group = 4 + packed
                scale_low, scale_high = fields.scale
                minimum_low, minimum_high = fields.minimum
                asm.inst(
                    f"v_bfe_u32 v{decode_scratch + group}, "
                    f"v{metadata_base + scale_low.metadata_word}, "
                    f"{scale_low.bit_offset}, {scale_low.bit_count}"
                )
                asm.inst(
                    f"v_bfe_u32 v{decode_scratch + 8 + group}, "
                    f"v{metadata_base + minimum_low.metadata_word}, "
                    f"{minimum_low.bit_offset}, {minimum_low.bit_count}"
                )
                asm.inst(
                    f"v_bfe_u32 v{decode_scratch + 16 + packed}, "
                    f"v{metadata_base + scale_high.metadata_word}, "
                    f"{scale_high.bit_offset}, {scale_high.bit_count}"
                )
                asm.inst(
                    f"v_bfe_u32 v{decode_scratch + 20 + packed}, "
                    f"v{metadata_base + minimum_high.metadata_word}, "
                    f"{minimum_high.bit_offset}, {minimum_high.bit_count}"
                )
            for packed, fields in enumerate(upper_fields):
                group = 4 + packed
                scale_high = fields.scale[1]
                minimum_high = fields.minimum[1]
                asm.inst(
                    f"v_lshl_or_b32 v{decode_scratch + group}, "
                    f"v{decode_scratch + 16 + packed}, "
                    f"{scale_high.destination_shift}, v{decode_scratch + group}"
                )
                asm.inst(
                    f"v_lshl_or_b32 v{decode_scratch + 8 + group}, "
                    f"v{decode_scratch + 20 + packed}, "
                    f"{minimum_high.destination_shift}, "
                    f"v{decode_scratch + 8 + group}"
                )
            for group in range(8):
                asm.inst(
                    f"v_cvt_f16_u16_e32 v{decode_scratch + 24 + group}.l, "
                    f"v{decode_scratch + group}.l"
                )
            for group in range(8):
                asm.inst(
                    f"v_cvt_f16_u16_e32 v{decode_scratch + 24 + group}.h, "
                    f"v{decode_scratch + 8 + group}.l"
                )
            for group in range(8):
                asm.inst(
                    f"v_pk_mul_f16 v{decode_scratch + 24 + group}, "
                    f"0xbc003c00, v{decode_scratch + 24 + group}"
                )
            for group in range(8):
                asm.inst(
                    f"v_pk_mul_f16 v{decode_scratch + 24 + group}, "
                    f"v{metadata_base}, v{decode_scratch + 24 + group}"
                )
            for group in range(8):
                asm.inst(
                    f"ds_write_b32 v{lds_address}, "
                    f"v{decode_scratch + 24 + group} offset:{4 * group}"
                )
        else:
            for group in range(8):
                emit_packed_scale_minimum(
                    asm,
                    group,
                    metadata_base,
                    scale=decode_scratch,
                    minimum=decode_scratch + 1,
                    temporary=decode_scratch + 2,
                    semantics=self.inputs.semantics,
                )
                asm.inst(
                    f"v_cvt_f16_u16_e32 v{decode_scratch + 5}.l, v{decode_scratch}.l"
                )
                asm.inst(
                    f"v_cvt_f16_u16_e32 v{decode_scratch + 5}.h, "
                    f"v{decode_scratch + 1}.l"
                )
                asm.inst(
                    f"v_pk_mul_f16 v{decode_scratch + 5}, 0xbc003c00, "
                    f"v{decode_scratch + 5}"
                )
                asm.inst(
                    f"v_pk_mul_f16 v{decode_scratch + 5}, v{metadata_base}, "
                    f"v{decode_scratch + 5}"
                )
                asm.inst(
                    f"ds_write_b32 v{lds_address}, v{decode_scratch + 5} "
                    f"offset:{4 * group}"
                )
        asm.label(f".LForward{quant_label}DecodedMetadataDone")

    def emit_i8_mma_group_loop(
        self,
        asm: Assembly,
        *,
        group_base: int,
        row_tiles: int | None = None,
        label_suffix: str = "",
    ) -> None:
        decoded_lds = self.inputs.layout
        registers = self.inputs.registers
        scalar = self.inputs.scalar_registers
        group_loop = scalar.group_loop.first_register
        group_offset = scalar.group_offset.first_register
        weight_q = registers.weight_payload.first_register
        metadata = registers.metadata_addresses.first_register
        c_base = registers.c_fragments.first_register
        low_activation_last = registers.low_activation_tail.first_register
        high_activation_base = registers.high_activation.first_register
        activation_scale_sum_base = registers.activation_scale_sum.first_register
        scaled_dm_base = registers.scaled_dm.first_register
        lds_address = registers.lds_address.first_register
        activation_base = registers.activation_base.first_register
        weight_lds_base_address = registers.output_column.first_register
        metadata_lds_base_address = registers.metadata_lds_address.first_register
        activation_metadata = decoded_lds.activation_metadata
        quant_type = self.inputs.quant_type
        quant_label = quant_type.replace("_", "")
        row_tiles = row_tiles or self.inputs.allocated_row_tiles
        label = f".LForward{quant_label}DecodedGroupLoop{group_base}{label_suffix}"
        asm.comment(
            f"Roll decoded {quant_type} groups {group_base} through {group_base + 3}."
        )
        asm.inst(f"s_mov_b32 s{group_loop}, 0")
        asm.label(label)

        asm.inst(f"s_lshl_b32 s{group_offset}, s{group_loop}, 5")
        if group_base:
            asm.inst(
                f"v_add_nc_u32 v{lds_address}, {32 * group_base}, "
                f"v{weight_lds_base_address}"
            )
            asm.inst(f"v_add_nc_u32 v{lds_address}, s{group_offset}, v{lds_address}")
        else:
            asm.inst(
                f"v_add_nc_u32 v{lds_address}, s{group_offset}, "
                f"v{weight_lds_base_address}"
            )
        asm.inst(f"ds_read_b128 v[{weight_q}:{weight_q + 3}], v{lds_address}")
        asm.inst(
            f"ds_read_b128 v[{weight_q + 4}:{weight_q + 7}], v{lds_address} offset:16"
        )

        asm.inst(f"v_add_nc_u32 v{lds_address}, s{group_offset}, v{activation_base}")
        allocated_row_tiles = self.inputs.allocated_row_tiles
        for tile in range(row_tiles):
            low_activation = (
                c_base + 8 * (tile + 1)
                if tile < allocated_row_tiles - 1
                else low_activation_last
            )
            high_activation = high_activation_base + 4 * tile
            tile_offset = 16 * tile * activation_metadata.block_bytes
            asm.inst(
                f"ds_read_b128 v[{low_activation}:{low_activation + 3}], "
                f"v{lds_address} offset:{tile_offset + activation_metadata.PAYLOAD_BASE}"
            )
            asm.inst(
                f"ds_read_b128 v[{high_activation}:{high_activation + 3}], "
                f"v{lds_address} offset:{tile_offset + activation_metadata.PAYLOAD_BASE + activation_metadata.PAYLOAD_VECTOR_BYTES}"
            )

        def emit_metadata_reads() -> None:
            asm.inst(f"s_lshl_b32 s{group_offset}, s{group_loop}, 2")
            asm.inst(f"v_add_nc_u32 v{metadata}, s{group_offset}, v{activation_base}")
            for tile in range(0, row_tiles, 2):
                asm.inst(
                    f"ds_read2st64_b32 v[{activation_scale_sum_base + tile}:"
                    f"{activation_scale_sum_base + tile + 1}], v{metadata} "
                    f"offset0:{9 * tile} offset1:{9 * (tile + 1)}"
                )

            if group_base:
                asm.inst(
                    f"v_add_nc_u32 v{metadata}, {4 * group_base}, "
                    f"v{metadata_lds_base_address}"
                )
                asm.inst(f"v_add_nc_u32 v{metadata}, s{group_offset}, v{metadata}")
            else:
                asm.inst(
                    f"v_add_nc_u32 v{metadata}, s{group_offset}, "
                    f"v{metadata_lds_base_address}"
                )
            for pair in range(1, 4):
                asm.inst(
                    f"v_add_nc_u32 v{metadata + pair}, "
                    f"{metadata_group_stride * pair}, v{metadata}"
                )
            for element in range(0, 8, 2):
                asm.inst(
                    f"ds_read2_b32 v[{scaled_dm_base + element}:"
                    f"{scaled_dm_base + element + 1}], "
                    f"v{metadata + element // 2} offset0:0 offset1:152"
                )

        metadata_group_stride = decoded_lds.metadata_group_stride
        deferred_metadata = self.inputs.decode_policy.defer_metadata_reads
        if not deferred_metadata:
            emit_metadata_reads()
        self.emit_scaled_i8_mma(
            asm,
            deferred_metadata_emitter=(
                emit_metadata_reads if deferred_metadata else None
            ),
            row_tiles=row_tiles,
        )
        asm.inst(f"s_add_u32 s{group_loop}, s{group_loop}, 1")
        asm.inst(f"s_cmp_lt_u32 s{group_loop}, 4")
        asm.inst(f"s_cbranch_scc1 {label}")

    def emit_scaled_i8_mma(
        self,
        asm: Assembly,
        *,
        deferred_metadata_emitter: Callable[[], None] | None = None,
        row_tiles: int | None = None,
    ) -> None:
        registers = self.inputs.registers
        zero_accumulator = registers.zero_accumulator.first_register
        weight_q = registers.weight_payload.first_register
        c_base = registers.c_fragments.first_register
        low_activation_last = registers.low_activation_tail.first_register
        high_activation_base = registers.high_activation.first_register
        activation_scale_sum_base = registers.activation_scale_sum.first_register
        scaled_dm_base = registers.scaled_dm.first_register
        sum_base = registers.sums.first_register
        row_tiles = row_tiles or self.inputs.allocated_row_tiles
        allocated_row_tiles = self.inputs.allocated_row_tiles
        for tile in range(row_tiles):
            first_wait = (
                2 * row_tiles - 1
                if deferred_metadata_emitter is not None
                else 2 * row_tiles + row_tiles // 2 + 3
            )
            asm.inst(f"s_waitcnt lgkmcnt({first_wait - 2 * tile})")
            c_fragment = c_base + 8 * tile
            low_activation = (
                c_base + 8 * (tile + 1)
                if tile < allocated_row_tiles - 1
                else low_activation_last
            )
            emit_signed_i8_wmma(
                asm,
                destination=c_fragment,
                weight=weight_q,
                activation=low_activation,
                accumulator=zero_accumulator,
                clamp=self.inputs.wmma_clamp,
            )
        if deferred_metadata_emitter is not None:
            deferred_metadata_emitter()
        for tile in range(row_tiles):
            if tile == row_tiles - 1:
                asm.inst(f"s_waitcnt lgkmcnt({4 + row_tiles // 2})")
            c_fragment = c_base + 8 * tile
            high_activation = high_activation_base + 4 * tile
            emit_signed_i8_wmma(
                asm,
                destination=c_fragment,
                weight=weight_q + 4,
                activation=high_activation,
                accumulator=c_fragment,
                clamp=self.inputs.wmma_clamp,
            )
        asm.inst("s_waitcnt lgkmcnt(0)")

        product_base = low_activation_last
        for tile_start in range(0, row_tiles, 4):
            local_tile_count = min(4, row_tiles - tile_start)
            for local_tile in range(local_tile_count):
                tile = tile_start + local_tile
                tile_scale_sum = activation_scale_sum_base + tile
                for element in range(8):
                    product = product_base + 8 * local_tile + element
                    asm.inst(
                        f"v_fma_mix_f32 v{product}, "
                        f"v{scaled_dm_base + element}, v{tile_scale_sum}, 0 "
                        f"op_sel_hi:[1,1,0]"
                    )
            for local_tile in range(local_tile_count):
                tile = tile_start + local_tile
                c_fragment = c_base + 8 * tile
                for element in range(8):
                    asm.inst(
                        f"v_cvt_f32_i32 v{c_fragment + element}, "
                        f"v{c_fragment + element}"
                    )
            for local_tile in range(local_tile_count):
                tile = tile_start + local_tile
                c_fragment = c_base + 8 * tile
                for element in range(0, 8, 2):
                    total = sum_base + tile * 8 + element
                    product = product_base + 8 * local_tile + element
                    asm.inst(
                        f"v_dual_fmac_f32 v{total}, v{product}, "
                        f"v{c_fragment + element} :: "
                        f"v_dual_fmac_f32 v{total + 1}, v{product + 1}, "
                        f"v{c_fragment + element + 1}"
                    )
            for local_tile in range(local_tile_count):
                tile = tile_start + local_tile
                tile_scale_sum = activation_scale_sum_base + tile
                for element in range(8):
                    total = sum_base + tile * 8 + element
                    asm.inst(
                        f"v_fma_mix_f32 v{total}, "
                        f"v{scaled_dm_base + element}, v{tile_scale_sum}, v{total} "
                        f"op_sel:[1,1,0] op_sel_hi:[1,1,0]"
                    )


def emit_decoded_bf16_tile_store(
    asm: Assembly,
    *,
    registers: DecodedWeightLdsRegisterPlan,
    output: RegisterAssignment,
    tile: int,
) -> None:
    sum_base = registers.sums.first_register
    output_address = registers.output_address.first_register
    asm.inst("s_clause 7")
    for element in range(8):
        total = sum_base + tile * 8 + element
        offset = f" offset:{4 * element}" if element else ""
        asm.inst(
            f"global_store_d16_hi_b16 v{output_address}, v{total}, "
            f"s[{output.first_register}:{output.first_register + 1}]{offset}"
        )
