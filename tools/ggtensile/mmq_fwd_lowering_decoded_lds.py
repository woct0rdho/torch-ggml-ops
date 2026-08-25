"""Decoded-weight LDS and F16_D4S4 activation forward lowering."""

from dataclasses import dataclass
from typing import ClassVar, cast

from .kernel_abi import ORDINARY_FORWARD_ABI
from .kernel_writer_assembly import (
    Assembly,
    emit_bf16_rne,
    emit_kernel_trailer,
    emit_pointer_kernarg_loads,
)
from .mmq_fwd_lowering import ForwardKernelWriterError, ForwardLoweringContext
from .mmq_fwd_lowering_decoded_stage import (
    DecodedWeightLdsStageEmitter,
    DecodedWeightLdsStageInputs,
    emit_decoded_bf16_tile_store,
)
from .mmq_fwd_physical import (
    DecodedWeightLdsPhysicalPlan,
)


@dataclass(frozen=True)
class DecodedWeightLdsLowering:
    """Emit decoded-weight LDS staging and scaled integer MMA."""

    context: ForwardLoweringContext

    KERNARG: ClassVar[int] = 4
    LOOP_COUNTER: ClassVar[int] = 10
    SCALAR_TEMPORARY: ClassVar[int] = 11
    WAVE_INDEX: ClassVar[int] = 12
    GROUP_LOOP: ClassVar[int] = 13
    GROUP_OFFSET: ClassVar[int] = 14
    ACTIVATION_LDS_BASE: ClassVar[int] = 15

    def _row_tile_count(self) -> int:
        return 8

    def body(self) -> str:
        physical = self.context.state.physical_plan
        assert isinstance(physical, DecodedWeightLdsPhysicalPlan)
        return self._body(physical)

    def _body(self, physical: DecodedWeightLdsPhysicalPlan) -> str:
        """Lower decoded-weight LDS staging and rolled four-group batches."""
        asm = Assembly()
        registers = physical.registers
        stage = DecodedWeightLdsStageEmitter(
            DecodedWeightLdsStageInputs(
                quant_type=self.context.state.contract.quant_type,
                packed_weight_row_bytes=self.context.state.packed_weight_row_bytes,
                semantics=self.context.state.semantics,
                decode_policy=self.context.state.kernel_spec.decode,
                wmma_clamp=self.context.state.contract.wmma_clamp,
                layout=physical.layout,
                registers=registers,
                scalar_registers=physical.scalar_registers,
                allocated_row_tiles=self._row_tile_count(),
            )
        )
        name = self.context.kernel_name
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
        activation_base = registers.activation_base.first_register
        lane = registers.lane.first_register
        serial = registers.serial.first_register
        quant_label = quant_type.replace("_", "")

        asm.comment(
            "Load the exact packed-weight, Q8_1 F16_D4S4 workspace, and output pointers."
        )
        emit_pointer_kernarg_loads(asm, self.KERNARG, ORDINARY_FORWARD_ABI)
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
        asm.inst(f"s_mov_b32 s{self.ACTIVATION_LDS_BASE}, 512")
        asm.inst(f"v_readfirstlane_b32 s{self.WAVE_INDEX}, v{wave}")
        asm.inst(
            f"v_mad_u32_u24 v{activation_base}, "
            f"{activation_metadata.block_bytes}, v{lane}, "
            f"s{self.ACTIVATION_LDS_BASE}"
        )

        asm.label(f".LForward{quant_label}HipStagedBlockLoop")
        stage.emit_weight_decode_stage(asm)

        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{metadata_address}, 4, s{self.WAVE_INDEX}")
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
        stage.emit_i8_mma_group_loop(asm, group_base=0)
        asm.inst("s_barrier")

        self._emit_f16_d4s4_activation_stage(asm)
        stage.emit_i8_mma_group_loop(asm, group_base=4)
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
        physical = cast(
            DecodedWeightLdsPhysicalPlan,
            self.context.state.physical_plan,
        )
        registers = physical.registers
        output = physical.scalar_registers.output
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
                emit_decoded_bf16_tile_store(
                    asm,
                    registers=registers,
                    output=output,
                    tile=tile,
                )
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
                emit_decoded_bf16_tile_store(
                    asm,
                    registers=registers,
                    output=output,
                    tile=tile,
                )

    def _emit_output_address(
        self,
        asm: Assembly,
    ) -> None:
        registers = cast(
            DecodedWeightLdsPhysicalPlan,
            self.context.state.physical_plan,
        ).registers
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

    def _emit_f16_d4s4_activation_stage(
        self,
        asm: Assembly,
    ) -> None:
        physical = cast(
            DecodedWeightLdsPhysicalPlan,
            self.context.state.physical_plan,
        )
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
