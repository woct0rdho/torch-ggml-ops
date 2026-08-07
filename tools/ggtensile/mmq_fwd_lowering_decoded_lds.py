"""Decoded-weight LDS and F16_D4S4 activation forward lowering."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, cast

from .kernel_writer_assembly import (
    Assembly,
    emit_bf16_rne,
    emit_kernel_trailer,
    emit_pointer_kernarg_loads,
)
from .mmq_fwd_lowering import ForwardKernelWriterError, ForwardLoweringContext
from .mmq_fwd_lowering_metadata import emit_packed_scale_minimum
from .mmq_fwd_lowering_mma import emit_signed_i8_wmma
from .mmq_fwd_physical import (
    DecodedWeightLdsPhysicalPlan,
    DecodedWeightLdsRegisterPlan,
)


@dataclass(frozen=True)
class DecodedWeightLdsLowering:
    """Emit decoded-weight LDS staging and scaled integer MMA."""

    context: ForwardLoweringContext

    OPERAND_SOURCE: ClassVar[str] = "DecodedWeightLdsBatch8"
    KERNARG: ClassVar[int] = 4
    LOOP_COUNTER: ClassVar[int] = 10
    SCALAR_TEMPORARY: ClassVar[int] = 11

    @property
    def _decoded_physical(self) -> DecodedWeightLdsPhysicalPlan:
        return cast(DecodedWeightLdsPhysicalPlan, self.context.state.physical_plan)

    @property
    def _decoded_registers(self) -> DecodedWeightLdsRegisterPlan:
        return self._decoded_physical.registers

    def body(self) -> str:
        operand_source = self.context.state.kernel_spec.global_memory.operand_source
        if operand_source != self.OPERAND_SOURCE:
            raise TypeError(
                f"unsupported decoded-weight LDS operand source {operand_source!r}"
            )
        return self._body()

    def _emit_weight_decode_stage(
        self,
        asm: Assembly,
    ) -> None:
        decoded_lds = self._decoded_physical.layout
        registers = self._decoded_registers
        decode_scratch = registers.decode_scratch.first_register
        row_stride = self.context.state.packed_weight_row_bytes
        serial = registers.serial.first_register
        wave = registers.wave.first_register
        temporary = registers.temporary.first_register
        lds_address = registers.lds_address.first_register
        auxiliary = registers.auxiliary.first_register
        metadata_address = registers.metadata_lds_address.first_register
        staging_base = registers.staged_payload.first_register
        weight_lds_base = decoded_lds.weight_data_base
        weight_lds_stride = decoded_lds.weight_row_stride
        quant_type = self.context.state.contract.quant_type
        quant_label = quant_type.replace("_", "")
        semantics = self.context.state.semantics
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
        asm.inst(f"v_add_nc_u32 v{temporary}, s{self.SCALAR_TEMPORARY}, v{temporary}")
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
        asm.inst(f"v_readfirstlane_b32 s12, v{wave}")
        asm.inst("s_cmp_lt_u32 s12, 2")
        asm.inst(f"s_cbranch_scc0 .LForward{quant_label}DecodedMetadataLoadDone")
        asm.inst(f"v_lshlrev_b32 v{metadata_address}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{metadata_address}, v{serial}, v{metadata_address}")
        asm.inst(f"v_mul_lo_u32 v{metadata_address}, {row_stride}, v{metadata_address}")
        asm.inst(
            f"v_add_nc_u32 v{metadata_address}, s{self.SCALAR_TEMPORARY}, "
            f"v{metadata_address}"
        )
        asm.inst(
            f"global_load_b128 v[{metadata_base}:{metadata_base + 3}], "
            f"v{metadata_address}, s[{self.KERNARG}:{self.KERNARG + 1}]"
        )
        asm.label(f".LForward{quant_label}DecodedMetadataLoadDone")

        if high_bits is not None:
            qh_base = staging_base + 36
            for row_slice in range(4):
                raw_base = staging_base + 8 * row_slice
                row_qh_base = qh_base + 4 * row_slice
                asm.inst(
                    f"global_load_b128 v[{row_qh_base}:{row_qh_base + 3}], "
                    f"v{qh_address}, s[{self.KERNARG}:{self.KERNARG + 1}] "
                    f"offset:{qh_offset}"
                )
                asm.inst(
                    f"global_load_b128 v[{raw_base}:{raw_base + 3}], "
                    f"v{temporary}, s[{self.KERNARG}:{self.KERNARG + 1}] "
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
                    f"v{temporary}, s[{self.KERNARG}:{self.KERNARG + 1}]"
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
        asm.inst("s_cmp_lt_u32 s12, 2")
        asm.inst(f"s_cbranch_scc0 .LForward{quant_label}DecodedMetadataDone")
        asm.inst(f"v_mul_lo_u32 v{lds_address}, {weight_lds_stride}, v{serial}")
        asm.inst(
            f"v_add_nc_u32 v{lds_address}, {weight_lds_base + 256}, v{lds_address}"
        )
        metadata_schedule = self.context.state.kernel_spec.decode.metadata_schedule
        if metadata_schedule in (
            "IndependentExtraction",
            "IndependentExtractionMetadataAfterLowWmma",
        ):
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
                    semantics=self.context.state.semantics,
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

    def _body(self) -> str:
        """Lower decoded-weight LDS staging and rolled four-group batches."""
        asm = Assembly()
        physical = self._decoded_physical
        registers = physical.registers
        name = self.context.solution_key.kernel_name
        quant_type = self.context.state.contract.quant_type
        decoded_lds = physical.layout
        activation_metadata = decoded_lds.activation_metadata
        zero_accumulator = registers.zero_accumulator.first_register
        sum_base = registers.sums.first_register
        temporary = registers.temporary.first_register
        metadata_address = registers.metadata_lds_address.first_register
        activation_plane_address = registers.activation_plane_address.first_register
        output_column = registers.output_column.first_register
        wave_column_base = registers.wave_column_base.first_register
        wave = registers.wave.first_register
        lane = registers.lane.first_register
        serial = registers.serial.first_register
        quant_label = quant_type.replace("_", "")

        asm.comment(
            "Load the exact packed-weight, Q8_1 F16_D4S4 workspace, and output pointers."
        )
        emit_pointer_kernarg_loads(asm, self.KERNARG)
        asm.inst(f"v_mov_b32 v{serial}, v0")
        asm.inst(f"v_and_b32 v{lane}, 15, v{serial}")
        asm.inst(f"v_lshrrev_b32 v{wave}, 5, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{wave_column_base}, 4, v{wave}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{wave_column_base}, v{temporary}, v{wave_column_base}")
        asm.inst(f"v_lshlrev_b32 v{output_column}, 4, v{wave}")
        asm.inst(f"v_add_nc_u32 v{output_column}, v{lane}, v{output_column}")
        asm.inst(
            f"v_mul_lo_u32 v{output_column}, {decoded_lds.weight_row_stride}, "
            f"v{output_column}"
        )
        asm.inst(
            f"v_add_nc_u32 v{output_column}, {decoded_lds.weight_data_base}, "
            f"v{output_column}"
        )
        asm.inst(f"v_lshlrev_b32 v{temporary}, 7, s3")
        asm.inst(
            f"v_mul_lo_u32 v{activation_plane_address}, "
            f"{activation_metadata.block_bytes}, v{temporary}"
        )
        asm.inst(f"v_lshlrev_b32 v{temporary}, 2, v{serial}")
        asm.inst(
            f"v_add_nc_u32 v{activation_plane_address}, v{temporary}, "
            f"v{activation_plane_address}"
        )
        if (
            self.context.state.kernel_spec.instruction_policy.accumulator_initialization
            == "VopdPair"
        ):
            for register in range(zero_accumulator, sum_base + 64, 2):
                x_source = "0" if register < sum_base else "v0"
                y_source = "0" if register < sum_base else "v1"
                asm.inst(
                    f"v_dual_mov_b32 v{register}, {x_source} :: "
                    f"v_dual_mov_b32 v{register + 1}, {y_source}"
                )
        else:
            for register in range(zero_accumulator, zero_accumulator + 8):
                asm.inst(f"v_mov_b32 v{register}, 0")
            for register in range(sum_base, sum_base + 64):
                asm.inst(f"v_mov_b32 v{register}, v0")
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")
        asm.inst(f"s_mov_b32 s{self.SCALAR_TEMPORARY}, 0")
        asm.inst("s_mov_b32 s15, 512")

        asm.label(f".LForward{quant_label}HipStagedBlockLoop")
        self._emit_weight_decode_stage(asm)

        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{metadata_address}, 4, v{wave}")
        asm.inst(f"v_add_nc_u32 v{metadata_address}, v{temporary}, v{metadata_address}")
        asm.inst(
            f"v_mul_lo_u32 v{metadata_address}, {decoded_lds.metadata_row_stride}, "
            f"v{metadata_address}"
        )
        asm.inst(
            f"v_add_nc_u32 v{metadata_address}, {decoded_lds.weight_metadata_base}, "
            f"v{metadata_address}"
        )

        self._emit_f16_d4s4_activation_stage(asm)
        self._emit_i8_mma_group_loop(asm, group_base=0)
        asm.inst("s_barrier")

        self._emit_f16_d4s4_activation_stage(asm)
        self._emit_i8_mma_group_loop(asm, group_base=4)
        asm.inst("s_barrier")
        asm.inst(
            f"s_add_u32 s{self.SCALAR_TEMPORARY}, s{self.SCALAR_TEMPORARY}, "
            f"{self.context.state.contract.packed_weight_block_bytes}"
        )
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(
            f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {self.context.state.blocks_per_weight_row}"
        )
        asm.inst(f"s_cbranch_scc1 .LForward{quant_label}HipStagedBlockLoop")

        asm.comment("Store the 128x64 row-major BF16 output tile.")
        self._emit_bf16_epilogue(asm)
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_bf16_epilogue(
        self,
        asm: Assembly,
    ) -> None:
        registers = self._decoded_registers
        size_n = self.context.state.problem_size.n
        sum_base = registers.sums.first_register
        temporary = registers.temporary.first_register
        output_address = registers.output_address.first_register
        epilogue_scratch = registers.decode_scratch.first_register
        pipeline = self.context.state.kernel_spec.epilogue.pipeline
        if (
            pipeline is None
            or pipeline.tiles_ahead is None
            or pipeline.priority is None
        ):
            raise ForwardKernelWriterError(
                "decoded-weight LDS requires a complete epilogue pipeline"
            )
        scheduled = (
            pipeline.tiles_ahead != 8
            or pipeline.dependency_width != 1
            or pipeline.priority != 0
        )

        if not scheduled:
            for total in range(sum_base, sum_base + 64):
                emit_bf16_rne(asm, total, temporary)
            self._emit_output_address(asm)
            for tile in range(8):
                if tile:
                    asm.inst(
                        f"v_add_nc_u32 v{output_address}, {32 * size_n}, "
                        f"v{output_address}"
                    )
                self._emit_bf16_tile_store(asm, tile=tile)
            return

        if pipeline.priority:
            asm.inst(f"s_setprio {pipeline.priority}")
        self._emit_output_address(asm)

        tiles_ahead = pipeline.tiles_ahead
        dependency_width = pipeline.dependency_width
        for first_tile in range(0, 8, tiles_ahead):
            tile_count = min(tiles_ahead, 8 - first_tile)
            first_element = 8 * first_tile
            element_count = 8 * tile_count
            for batch in range(
                first_element,
                first_element + element_count,
                dependency_width,
            ):
                batch_count = min(
                    dependency_width,
                    first_element + element_count - batch,
                )
                if batch_count == 1:
                    total = sum_base + batch
                    emit_bf16_rne(asm, total, temporary)
                    continue
                for item in range(batch_count):
                    total = sum_base + batch + item
                    asm.inst(f"v_bfe_u32 v{epilogue_scratch + item}, v{total}, 16, 1")
                for item in range(batch_count):
                    total = sum_base + batch + item
                    asm.inst(
                        f"v_add3_u32 v{total}, v{epilogue_scratch + item}, "
                        f"v{total}, 0x7fff"
                    )
            for relative_tile in range(tile_count):
                tile = first_tile + relative_tile
                if tile:
                    asm.inst(
                        f"v_add_nc_u32 v{output_address}, {32 * size_n}, "
                        f"v{output_address}"
                    )
                self._emit_bf16_tile_store(asm, tile=tile)

    def _emit_output_address(
        self,
        asm: Assembly,
    ) -> None:
        registers = self._decoded_registers
        size_n = self.context.state.problem_size.n
        temporary = registers.temporary.first_register
        auxiliary = registers.auxiliary.first_register
        output_address = registers.output_address.first_register
        wave_column_base = registers.wave_column_base.first_register
        lane = registers.lane.first_register
        serial = registers.serial.first_register
        asm.inst(f"v_lshlrev_b32 v{temporary}, 7, s3")
        asm.inst(f"v_add_nc_u32 v{auxiliary}, v{temporary}, v{lane}")
        asm.inst(f"v_mul_lo_u32 v{output_address}, {2 * size_n}, v{auxiliary}")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{wave_column_base}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{output_address}, v{output_address}, v{temporary}")

    def _emit_bf16_tile_store(
        self,
        asm: Assembly,
        *,
        tile: int,
    ) -> None:
        registers = self._decoded_registers
        sum_base = registers.sums.first_register
        output_address = registers.output_address.first_register
        asm.inst("s_clause 7")
        for element in range(8):
            total = sum_base + tile * 8 + element
            offset = f" offset:{4 * element}" if element else ""
            asm.inst(
                f"global_store_d16_hi_b16 v{output_address}, v{total}, "
                f"s[{self.KERNARG + 4}:{self.KERNARG + 5}]{offset}"
            )

    def _emit_f16_d4s4_activation_stage(
        self,
        asm: Assembly,
    ) -> None:
        physical = self._decoded_physical
        registers = physical.registers
        activation_plane_address = registers.activation_plane_address.first_register
        serial = registers.serial.first_register
        temporary = registers.temporary.first_register
        lds_address = registers.lds_address.first_register
        staging_base = registers.staged_payload.first_register
        activation_plane_stride = self.context.state.activation_plane_stride_bytes
        activation_lane_stride = physical.layout.activation_lane_stride
        lds_base = physical.layout.activation_base
        asm.comment("Cooperatively stage one contiguous 128-row Q8_1 F16_D4S4 plane.")
        for chunk in range(5):
            chunk_start = 8 * chunk
            chunk_count = min(8, 36 - chunk_start)
            asm.inst(
                f"v_add_nc_u32 v{temporary}, {4096 * chunk}, "
                f"v{activation_plane_address}"
            )
            for item in range(chunk_count):
                register = staging_base + chunk_start + item
                asm.inst(
                    f"global_load_b32 v{register}, v{temporary}, "
                    f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                    f"offset:{activation_lane_stride * item}"
                )
        asm.inst(f"v_lshlrev_b32 v{lds_address}, 2, v{serial}")
        asm.inst(f"v_add_nc_u32 v{lds_address}, {lds_base}, v{lds_address}")
        for item in range(0, 36, 2):
            asm.inst(f"s_waitcnt vmcnt({34 - item})")
            asm.inst(
                f"ds_write2st64_b32 v{lds_address}, v{staging_base + item}, "
                f"v{staging_base + item + 1} offset0:{2 * item} "
                f"offset1:{2 * (item + 1)}"
            )
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        asm.inst(
            f"v_add_nc_u32 v{activation_plane_address}, "
            f"{activation_plane_stride}, v{activation_plane_address}"
        )

    def _emit_i8_mma_group_loop(
        self,
        asm: Assembly,
        *,
        group_base: int,
    ) -> None:
        registers = self._decoded_registers
        weight_q = registers.weight_payload.first_register
        metadata = registers.metadata_addresses.first_register
        c_base = registers.c_fragments.first_register
        low_activation_last = registers.low_activation_tail.first_register
        high_activation_base = registers.high_activation.first_register
        activation_scale_sum_base = registers.activation_scale_sum.first_register
        scaled_dm_base = registers.scaled_dm.first_register
        lane = registers.lane.first_register
        lds_address = registers.lds_address.first_register
        weight_lds_base_address = registers.output_column.first_register
        metadata_lds_base_address = registers.metadata_lds_address.first_register
        activation_metadata = self._decoded_physical.layout.activation_metadata
        quant_type = self.context.state.contract.quant_type
        quant_label = quant_type.replace("_", "")
        label = f".LForward{quant_label}DecodedGroupLoop{group_base}"
        asm.comment(
            f"Roll decoded {quant_type} groups {group_base} through {group_base + 3}."
        )
        asm.inst("s_mov_b32 s13, 0")
        asm.label(label)

        asm.inst("s_lshl_b32 s14, s13, 5")
        if group_base:
            asm.inst(
                f"v_add_nc_u32 v{lds_address}, {32 * group_base}, "
                f"v{weight_lds_base_address}"
            )
            asm.inst(f"v_add_nc_u32 v{lds_address}, s14, v{lds_address}")
        else:
            asm.inst(f"v_add_nc_u32 v{lds_address}, s14, v{weight_lds_base_address}")
        asm.inst(f"ds_read_b128 v[{weight_q}:{weight_q + 3}], v{lds_address}")
        asm.inst(
            f"ds_read_b128 v[{weight_q + 4}:{weight_q + 7}], v{lds_address} offset:16"
        )

        asm.inst(
            f"v_mad_u32_u24 v{metadata}, {activation_metadata.block_bytes}, v{lane}, s15"
        )
        asm.inst(f"v_add_nc_u32 v{lds_address}, s14, v{metadata}")
        for tile in range(8):
            low_activation = (
                c_base + 8 * (tile + 1) if tile < 7 else low_activation_last
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
            asm.inst("s_lshl_b32 s14, s13, 2")
            asm.inst(f"v_add_nc_u32 v{metadata}, s14, v{metadata}")
            for tile in range(0, 8, 2):
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
                asm.inst(f"v_add_nc_u32 v{metadata}, s14, v{metadata}")
            else:
                asm.inst(f"v_add_nc_u32 v{metadata}, s14, v{metadata_lds_base_address}")
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

        decoded_lds = self._decoded_physical.layout
        metadata_group_stride = decoded_lds.metadata_group_stride
        metadata_schedule = self.context.state.kernel_spec.decode.metadata_schedule
        deferred_metadata = metadata_schedule in (
            "MetadataAfterLowWmma",
            "IndependentExtractionMetadataAfterLowWmma",
        )
        if not deferred_metadata:
            emit_metadata_reads()
        self._emit_scaled_i8_mma(
            asm,
            deferred_metadata_emitter=(
                emit_metadata_reads if deferred_metadata else None
            ),
        )
        asm.inst("s_add_u32 s13, s13, 1")
        asm.inst("s_cmp_lt_u32 s13, 4")
        asm.inst(f"s_cbranch_scc1 {label}")

    def _emit_scaled_i8_mma(
        self,
        asm: Assembly,
        *,
        deferred_metadata_emitter: Callable[[], None] | None = None,
    ) -> None:
        registers = self._decoded_registers
        zero_accumulator = registers.zero_accumulator.first_register
        weight_q = registers.weight_payload.first_register
        c_base = registers.c_fragments.first_register
        low_activation_last = registers.low_activation_tail.first_register
        high_activation_base = registers.high_activation.first_register
        activation_scale_sum_base = registers.activation_scale_sum.first_register
        scaled_dm_base = registers.scaled_dm.first_register
        sum_base = registers.sums.first_register
        for tile in range(8):
            first_wait = 15 if deferred_metadata_emitter is not None else 23
            asm.inst(f"s_waitcnt lgkmcnt({first_wait - 2 * tile})")
            c_fragment = c_base + 8 * tile
            low_activation = (
                c_base + 8 * (tile + 1) if tile < 7 else low_activation_last
            )
            emit_signed_i8_wmma(
                asm,
                destination=c_fragment,
                weight=weight_q,
                activation=low_activation,
                accumulator=zero_accumulator,
                clamp=self.context.state.contract.wmma_clamp,
            )
        if deferred_metadata_emitter is not None:
            deferred_metadata_emitter()
        for tile in range(8):
            if tile == 7:
                asm.inst("s_waitcnt lgkmcnt(8)")
            c_fragment = c_base + 8 * tile
            high_activation = high_activation_base + 4 * tile
            emit_signed_i8_wmma(
                asm,
                destination=c_fragment,
                weight=weight_q + 4,
                activation=high_activation,
                accumulator=c_fragment,
                clamp=self.context.state.contract.wmma_clamp,
            )
        asm.inst("s_waitcnt lgkmcnt(0)")

        product_base = low_activation_last
        for tile_start in (0, 4):
            for local_tile in range(4):
                tile = tile_start + local_tile
                tile_scale_sum = activation_scale_sum_base + tile
                for element in range(8):
                    product = product_base + 8 * local_tile + element
                    asm.inst(
                        f"v_fma_mix_f32 v{product}, "
                        f"v{scaled_dm_base + element}, v{tile_scale_sum}, 0 "
                        f"op_sel_hi:[1,1,0]"
                    )
            for local_tile in range(4):
                tile = tile_start + local_tile
                c_fragment = c_base + 8 * tile
                for element in range(8):
                    asm.inst(
                        f"v_cvt_f32_i32 v{c_fragment + element}, "
                        f"v{c_fragment + element}"
                    )
            for local_tile in range(4):
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
            for local_tile in range(4):
                tile = tile_start + local_tile
                tile_scale_sum = activation_scale_sum_base + tile
                for element in range(8):
                    total = sum_base + tile * 8 + element
                    asm.inst(
                        f"v_fma_mix_f32 v{total}, "
                        f"v{scaled_dm_base + element}, v{tile_scale_sum}, v{total} "
                        f"op_sel:[1,1,0] op_sel_hi:[1,1,0]"
                    )
