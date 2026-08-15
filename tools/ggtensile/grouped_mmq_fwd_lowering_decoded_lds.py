"""Grouped row-tiled decoded-weight LDS lowering for Q4_K forward."""

from typing import ClassVar, cast

from .grouped_mmq_fwd_lowering import (
    GroupedForwardLoweringContext,
    GroupedPackedScaleMinimumDirectLowering,
)
from .grouped_mmq_fwd_physical import GroupedDecodedPhysicalPlan
from .grouped_mmq_fwd_spec import DerivedGroupedForwardState
from .kernel_writer_assembly import Assembly, emit_bf16_rne, emit_kernel_trailer
from .mmq_fwd_lowering import ForwardLoweringContext
from .mmq_fwd_lowering_decoded_lds import DecodedWeightLdsLowering


class GroupedDecodedWeightLdsLowering(DecodedWeightLdsLowering):
    """Reuse decoded arithmetic under routed 128-row serial ownership."""

    OPERAND_SOURCE: ClassVar[str] = "GroupedDecodedWeightLds"
    LOOP_COUNTER: ClassVar[int] = 27
    SCALAR_TEMPORARY: ClassVar[int] = 26
    WAVE_INDEX: ClassVar[int] = 30
    GROUP_LOOP: ClassVar[int] = 31
    GROUP_OFFSET: ClassVar[int] = 32
    ACTIVATION_LDS_BASE: ClassVar[int] = 35
    EXEC_MASK: ClassVar[int] = 28
    ACTIVATION_PLANE_END: ClassVar[int] = 33
    ROW_TILE_END: ClassVar[int] = 34

    def _grouped_context(self) -> GroupedForwardLoweringContext:
        return cast(GroupedForwardLoweringContext, self.context)

    def _grouped_state(self) -> DerivedGroupedForwardState:
        return self._grouped_context().state

    def _physical_plan(self) -> GroupedDecodedPhysicalPlan:
        return cast(GroupedDecodedPhysicalPlan, self._grouped_state().physical_plan)

    def _row_tile_count(self) -> int:
        return self._grouped_context().solution_key.solution.macro_tile0 // 16

    def _activation_stage_requires_mask(self, row_tile_rows: int) -> bool:
        return False

    def _activation_stage_dwords(self, row_tile_rows: int) -> int:
        solution = self._grouped_context().solution_key.solution
        total_bytes = row_tile_rows * solution.activation_block_bytes
        bytes_per_thread = 4 * solution.num_threads
        if self._activation_stage_requires_mask(row_tile_rows):
            return (total_bytes + bytes_per_thread - 1) // bytes_per_thread
        return total_bytes // bytes_per_thread

    def _has_three_row_tile_bodies(self) -> bool:
        solution = self._grouped_context().solution_key.solution
        return (
            solution.macro_tile0 == 128
            and solution.tail_macro_tile0 == 32
            and solution.output_store == "BFloat16RNEClause8Clause4Clause2MixedMasked"
        )

    def body(self) -> str:
        if (
            self._grouped_context().solution_key.solution.operand_source
            != self.OPERAND_SOURCE
        ):
            raise TypeError("grouped decoded lowering requires GroupedDecodedWeightLds")
        return self._body_grouped()

    def _body_grouped(self) -> str:
        state = self._grouped_state()
        context = self._grouped_context()
        physical = self._physical_plan()
        registers = physical.registers
        scalar = physical.scalar_registers
        layout = physical.layout
        activation_metadata = layout.activation_metadata
        asm = Assembly()
        name = context.solution_key.kernel_name
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

        route = GroupedPackedScaleMinimumDirectLowering(context)
        route._emit_kernarg_loads(asm)
        route._emit_exact_shape_guard(asm)
        route._emit_route_load_and_guard(asm)
        route._emit_expert_pointer_rebase(asm)

        asm.inst(
            f"s_mul_i32 s{scalar.activation_plane_stride.first_register}, "
            f"s{scalar.nrows_activation.first_register}, "
            f"{context.solution_key.solution.activation_block_bytes}"
        )
        asm.inst(
            f"s_mov_b32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_begin.first_register}"
        )
        asm.inst(f"s_mov_b32 s{self.ACTIVATION_LDS_BASE}, {layout.activation_base}")

        asm.comment("Map four waves to one 128-row by 64-column grouped tile.")
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

        asm.label(".LGroupedQ4KDecodedRowLoop")
        asm.inst(
            f"v_mul_lo_u32 v{activation_plane_address}, "
            f"{activation_metadata.block_bytes}, s{scalar.row_start.first_register}"
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
        for register in range(zero_accumulator, zero_accumulator + 8):
            asm.inst(f"v_mov_b32 v{register}, 0")
        for register in range(sum_base, sum_base + 8 * self._row_tile_count()):
            asm.inst(f"v_mov_b32 v{register}, v{zero_accumulator}")
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")
        asm.inst(f"s_mov_b32 s{self.SCALAR_TEMPORARY}, 0")

        asm.label(".LGroupedQ4KDecodedBlockLoop")
        self._emit_weight_decode_stage(asm)

        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{metadata_address}, 4, s{self.WAVE_INDEX}")
        asm.inst(f"v_add_nc_u32 v{metadata_address}, v{temporary}, v{metadata_address}")
        asm.inst(
            f"v_mul_lo_u32 v{metadata_address}, {layout.metadata_row_stride}, "
            f"v{metadata_address}"
        )
        asm.inst(
            f"v_add_nc_u32 v{metadata_address}, {layout.weight_metadata_base}, "
            f"v{metadata_address}"
        )

        self._emit_grouped_activation_stage_dispatch(asm, stage=0)
        self._emit_grouped_mma_dispatch(asm, group_base=0)
        asm.inst("s_barrier")

        self._emit_grouped_activation_stage_dispatch(asm, stage=1)
        self._emit_grouped_mma_dispatch(asm, group_base=4)
        asm.inst("s_barrier")
        asm.inst(
            f"s_add_u32 s{self.SCALAR_TEMPORARY}, s{self.SCALAR_TEMPORARY}, "
            f"{context.solution_key.solution.packed_weight_block_bytes}"
        )
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {state.blocks_per_weight_row}")
        asm.inst("s_cbranch_scc1 .LGroupedQ4KDecodedBlockLoop")

        self._emit_grouped_epilogue_dispatch(asm)
        asm.inst(
            f"s_add_u32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_start.first_register}, "
            f"{context.solution_key.solution.macro_tile0}"
        )
        asm.inst(
            f"s_cmp_lt_u32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_end.first_register}"
        )
        asm.inst("s_cbranch_scc1 .LGroupedQ4KDecodedRowLoop")
        asm.label(".LGroupedQ4KExit")
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_grouped_activation_stage_dispatch(
        self, asm: Assembly, *, stage: int
    ) -> None:
        solution = self._grouped_context().solution_key.solution
        scalar = self._physical_plan().scalar_registers
        if solution.tail_macro_tile0 == solution.macro_tile0:
            self._emit_grouped_activation_stage(
                asm,
                stage=stage,
                row_tile_rows=solution.macro_tile0,
                label_suffix="Macro",
            )
            self._emit_grouped_activation_stage_finalize(asm)
            return

        if self._has_three_row_tile_bodies():
            self._emit_three_way_activation_stage_dispatch(asm, stage=stage)
            return

        tail_label = f".LGroupedQ4KActivationTailDispatch{stage}"
        done_label = f".LGroupedQ4KActivationDispatchDone{stage}"
        asm.inst(
            f"s_cmp_le_u32 s{scalar.row_tile_rows.first_register}, "
            f"{solution.tail_macro_tile0}"
        )
        asm.inst(f"s_cbranch_scc1 {tail_label}")
        self._emit_grouped_activation_stage(
            asm,
            stage=stage,
            row_tile_rows=solution.macro_tile0,
            label_suffix="Macro",
        )
        asm.inst(f"s_branch {done_label}")
        asm.label(tail_label)
        self._emit_grouped_activation_stage(
            asm,
            stage=stage,
            row_tile_rows=solution.tail_macro_tile0,
            label_suffix="Tail",
        )
        asm.label(done_label)
        self._emit_grouped_activation_stage_finalize(asm)

    def _emit_three_way_activation_stage_dispatch(
        self, asm: Assembly, *, stage: int
    ) -> None:
        scalar = self._physical_plan().scalar_registers
        tail32_label = f".LGroupedQ5KActivationRows32Dispatch{stage}"
        tail64_label = f".LGroupedQ5KActivationRows64Dispatch{stage}"
        done_label = f".LGroupedQ5KActivationThreeWayDone{stage}"
        asm.inst(f"s_cmp_le_u32 s{scalar.row_tile_rows.first_register}, 32")
        asm.inst(f"s_cbranch_scc1 {tail32_label}")
        asm.inst(f"s_cmp_le_u32 s{scalar.row_tile_rows.first_register}, 64")
        asm.inst(f"s_cbranch_scc1 {tail64_label}")
        self._emit_grouped_activation_stage(
            asm, stage=stage, row_tile_rows=128, label_suffix="Rows128"
        )
        asm.inst(f"s_branch {done_label}")
        asm.label(tail64_label)
        self._emit_grouped_activation_stage(
            asm, stage=stage, row_tile_rows=64, label_suffix="Rows64"
        )
        asm.inst(f"s_branch {done_label}")
        asm.label(tail32_label)
        self._emit_grouped_activation_stage(
            asm, stage=stage, row_tile_rows=32, label_suffix="Rows32"
        )
        asm.label(done_label)
        self._emit_grouped_activation_stage_finalize(asm)

    def _emit_grouped_mma_dispatch(self, asm: Assembly, *, group_base: int) -> None:
        solution = self._grouped_context().solution_key.solution
        scalar = self._physical_plan().scalar_registers
        if solution.tail_macro_tile0 == solution.macro_tile0:
            self._emit_i8_mma_group_loop(asm, group_base=group_base)
            return

        if self._has_three_row_tile_bodies():
            self._emit_three_way_mma_dispatch(asm, group_base=group_base)
            return

        tail_label = f".LGroupedQ4KMmaTailDispatch{group_base}"
        done_label = f".LGroupedQ4KMmaDispatchDone{group_base}"
        asm.inst(
            f"s_cmp_le_u32 s{scalar.row_tile_rows.first_register}, "
            f"{solution.tail_macro_tile0}"
        )
        asm.inst(f"s_cbranch_scc1 {tail_label}")
        self._emit_i8_mma_group_loop(
            asm,
            group_base=group_base,
            row_tiles=solution.macro_tile0 // 16,
            label_suffix="Macro",
        )
        asm.inst(f"s_branch {done_label}")
        asm.label(tail_label)
        self._emit_i8_mma_group_loop(
            asm,
            group_base=group_base,
            row_tiles=solution.tail_macro_tile0 // 16,
            label_suffix="Tail",
        )
        asm.label(done_label)

    def _emit_three_way_mma_dispatch(self, asm: Assembly, *, group_base: int) -> None:
        scalar = self._physical_plan().scalar_registers
        tail32_label = f".LGroupedQ5KMmaRows32Dispatch{group_base}"
        tail64_label = f".LGroupedQ5KMmaRows64Dispatch{group_base}"
        done_label = f".LGroupedQ5KMmaThreeWayDone{group_base}"
        asm.inst(f"s_cmp_le_u32 s{scalar.row_tile_rows.first_register}, 32")
        asm.inst(f"s_cbranch_scc1 {tail32_label}")
        asm.inst(f"s_cmp_le_u32 s{scalar.row_tile_rows.first_register}, 64")
        asm.inst(f"s_cbranch_scc1 {tail64_label}")
        self._emit_i8_mma_group_loop(
            asm, group_base=group_base, row_tiles=8, label_suffix="Rows128"
        )
        asm.inst(f"s_branch {done_label}")
        asm.label(tail64_label)
        self._emit_i8_mma_group_loop(
            asm, group_base=group_base, row_tiles=4, label_suffix="Rows64"
        )
        asm.inst(f"s_branch {done_label}")
        asm.label(tail32_label)
        self._emit_i8_mma_group_loop(
            asm, group_base=group_base, row_tiles=2, label_suffix="Rows32"
        )
        asm.label(done_label)

    def _emit_grouped_epilogue_dispatch(self, asm: Assembly) -> None:
        solution = self._grouped_context().solution_key.solution
        scalar = self._physical_plan().scalar_registers
        if solution.tail_macro_tile0 == solution.macro_tile0:
            self._emit_grouped_bf16_epilogue(asm)
            return

        if self._has_three_row_tile_bodies():
            self._emit_three_way_epilogue_dispatch(asm)
            return

        tail_label = ".LGroupedQ4KEpilogueTailDispatch"
        done_label = ".LGroupedQ4KEpilogueDispatchDone"
        asm.inst(
            f"s_cmp_le_u32 s{scalar.row_tile_rows.first_register}, "
            f"{solution.tail_macro_tile0}"
        )
        asm.inst(f"s_cbranch_scc1 {tail_label}")
        self._emit_grouped_bf16_epilogue(asm, row_tiles=solution.macro_tile0 // 16)
        asm.inst(f"s_branch {done_label}")
        asm.label(tail_label)
        self._emit_grouped_bf16_epilogue(asm, row_tiles=solution.tail_macro_tile0 // 16)
        asm.label(done_label)

    def _emit_three_way_epilogue_dispatch(self, asm: Assembly) -> None:
        scalar = self._physical_plan().scalar_registers
        tail32_label = ".LGroupedQ5KEpilogueRows32Dispatch"
        tail64_label = ".LGroupedQ5KEpilogueRows64Dispatch"
        done_label = ".LGroupedQ5KEpilogueThreeWayDone"
        asm.inst(f"s_cmp_le_u32 s{scalar.row_tile_rows.first_register}, 32")
        asm.inst(f"s_cbranch_scc1 {tail32_label}")
        asm.inst(f"s_cmp_le_u32 s{scalar.row_tile_rows.first_register}, 64")
        asm.inst(f"s_cbranch_scc1 {tail64_label}")
        self._emit_grouped_bf16_epilogue(asm, row_tiles=8)
        asm.inst(f"s_branch {done_label}")
        asm.label(tail64_label)
        self._emit_grouped_bf16_epilogue(asm, row_tiles=4)
        asm.inst(f"s_branch {done_label}")
        asm.label(tail32_label)
        self._emit_grouped_bf16_epilogue(asm, row_tiles=2)
        asm.label(done_label)

    def _emit_grouped_activation_stage(
        self,
        asm: Assembly,
        *,
        stage: int,
        row_tile_rows: int,
        label_suffix: str,
    ) -> None:
        physical = self._physical_plan()
        registers = physical.registers
        scalar = physical.scalar_registers
        layout = physical.layout
        activation_plane_address = registers.activation_plane_address.first_register
        temporary = registers.temporary.first_register
        auxiliary = registers.auxiliary.first_register
        staging_base = registers.staged_payload.first_register
        lane_stride = layout.activation_lane_stride
        stage_dwords = self._activation_stage_dwords(row_tile_rows)
        force_mask = self._activation_stage_requires_mask(row_tile_rows)
        full_label = f".LGroupedQ4KActivationFull{stage}{label_suffix}"
        done_label = f".LGroupedQ4KActivationLoaded{stage}{label_suffix}"

        asm.comment("Stage one full or masked aggregate-row Q8_1 plane.")
        if not force_mask:
            asm.inst(
                f"s_add_u32 s{self.ROW_TILE_END}, s{scalar.row_start.first_register}, "
                f"{row_tile_rows}"
            )
            asm.inst(
                f"s_cmp_le_u32 s{self.ROW_TILE_END}, s{scalar.row_end.first_register}"
            )
            asm.inst(f"s_cbranch_scc1 {full_label}")

        for item in range(stage_dwords):
            asm.inst(f"v_mov_b32 v{staging_base + item}, 0")
        for chunk in range((stage_dwords + 7) // 8):
            chunk_start = 8 * chunk
            chunk_count = min(8, stage_dwords - chunk_start)
            for item in range(chunk_count):
                register = staging_base + chunk_start + item
                offset = 8 * lane_stride * chunk + lane_stride * item
                asm.inst(
                    f"v_add_nc_u32 v{auxiliary}, {offset}, v{activation_plane_address}"
                )
                asm.inst(
                    f"v_cmp_gt_u32_e32 vcc_lo, s{self.ACTIVATION_PLANE_END}, "
                    f"v{auxiliary}"
                )
                asm.inst(f"s_and_saveexec_b32 s{self.EXEC_MASK}, vcc_lo")
                asm.inst(
                    f"global_load_b32 v{register}, v{auxiliary}, "
                    f"s[{scalar.activations.first_register}:"
                    f"{scalar.activations.first_register + 1}]"
                )
                asm.inst(f"s_mov_b32 exec_lo, s{self.EXEC_MASK}")
        asm.inst("s_waitcnt vmcnt(0)")
        self._emit_activation_lds_writes(
            asm, pipelined=False, row_tile_rows=row_tile_rows
        )
        asm.inst(f"s_branch {done_label}")

        if not force_mask:
            asm.label(full_label)
        for chunk in range((stage_dwords + 7) // 8):
            chunk_start = 8 * chunk
            chunk_count = min(8, stage_dwords - chunk_start)
            asm.inst(
                f"v_add_nc_u32 v{temporary}, {8 * lane_stride * chunk}, "
                f"v{activation_plane_address}"
            )
            for item in range(chunk_count):
                register = staging_base + chunk_start + item
                asm.inst(
                    f"global_load_b32 v{register}, v{temporary}, "
                    f"s[{scalar.activations.first_register}:"
                    f"{scalar.activations.first_register + 1}] "
                    f"offset:{lane_stride * item}"
                )
        self._emit_activation_lds_writes(
            asm, pipelined=True, row_tile_rows=row_tile_rows
        )

        asm.label(done_label)
        asm.inst("s_waitcnt lgkmcnt(0)")

    def _emit_grouped_activation_stage_finalize(self, asm: Assembly) -> None:
        scalar = self._physical_plan().scalar_registers
        registers = self._physical_plan().registers
        activation_plane_address = registers.activation_plane_address.first_register
        asm.inst("s_barrier")
        asm.inst(
            f"v_add_nc_u32 v{activation_plane_address}, "
            f"s{scalar.activation_plane_stride.first_register}, "
            f"v{activation_plane_address}"
        )
        asm.inst(
            f"s_add_u32 s{self.ACTIVATION_PLANE_END}, "
            f"s{self.ACTIVATION_PLANE_END}, "
            f"s{scalar.activation_plane_stride.first_register}"
        )

    def _emit_activation_lds_writes(
        self, asm: Assembly, *, pipelined: bool, row_tile_rows: int
    ) -> None:
        physical = self._physical_plan()
        registers = physical.registers
        serial = registers.serial.first_register
        lds_address = registers.lds_address.first_register
        staging_base = registers.staged_payload.first_register
        stage_dwords = self._activation_stage_dwords(row_tile_rows)
        asm.inst(f"v_lshlrev_b32 v{lds_address}, 2, v{serial}")
        asm.inst(
            f"v_add_nc_u32 v{lds_address}, {physical.layout.activation_base}, "
            f"v{lds_address}"
        )
        paired_dwords = stage_dwords // 2 * 2
        for item in range(0, paired_dwords, 2):
            if pipelined:
                asm.inst(f"s_waitcnt vmcnt({stage_dwords - 2 - item})")
            asm.inst(
                f"ds_write2st64_b32 v{lds_address}, v{staging_base + item}, "
                f"v{staging_base + item + 1} offset0:{2 * item} "
                f"offset1:{2 * (item + 1)}"
            )
        if paired_dwords != stage_dwords:
            if pipelined:
                asm.inst("s_waitcnt vmcnt(0)")
            asm.inst(
                f"ds_write_b32 v{lds_address}, v{staging_base + paired_dwords} "
                f"offset:{physical.layout.activation_lane_stride * paired_dwords}"
            )

    def _emit_grouped_bf16_epilogue(
        self, asm: Assembly, *, row_tiles: int | None = None
    ) -> None:
        state = self._grouped_state()
        physical = self._physical_plan()
        registers = physical.registers
        sum_base = registers.sums.first_register
        temporary = registers.temporary.first_register
        auxiliary = registers.auxiliary.first_register
        output_address = registers.output_address.first_register
        epilogue_scratch = registers.decode_scratch.first_register
        solution = self._grouped_context().solution_key.solution

        row_tiles = row_tiles or self._row_tile_count()
        scheduled = (
            solution.epilogue_tiles_ahead < row_tiles
            or solution.epilogue_dependency_width != 1
            or solution.epilogue_priority != 0
        )
        asm.comment("Store the valid rows of the grouped BF16 tile.")
        if not scheduled:
            for total in range(sum_base, sum_base + 8 * row_tiles):
                emit_bf16_rne(asm, total, temporary)
            self._emit_grouped_output_address(asm)
            for tile in range(row_tiles):
                if tile:
                    asm.inst(
                        f"v_add_nc_u32 v{output_address}, "
                        f"{32 * state.problem_size.n}, v{output_address}"
                    )
                    asm.inst(f"v_add_nc_u32 v{auxiliary}, 16, v{auxiliary}")
                self._emit_grouped_masked_tile_store(asm, tile=tile)
            return

        if solution.epilogue_priority:
            asm.inst(f"s_setprio {solution.epilogue_priority}")
        self._emit_grouped_output_address(asm)
        tiles_ahead = solution.epilogue_tiles_ahead
        dependency_width = solution.epilogue_dependency_width
        for first_tile in range(0, row_tiles, tiles_ahead):
            tile_count = min(tiles_ahead, row_tiles - first_tile)
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
                    emit_bf16_rne(asm, sum_base + batch, temporary)
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
                        f"v_add_nc_u32 v{output_address}, "
                        f"{32 * state.problem_size.n}, v{output_address}"
                    )
                    asm.inst(f"v_add_nc_u32 v{auxiliary}, 16, v{auxiliary}")
                self._emit_grouped_masked_tile_store(asm, tile=tile)
        if solution.epilogue_priority:
            asm.inst("s_setprio 0")

    def _emit_grouped_output_address(self, asm: Assembly) -> None:
        state = self._grouped_state()
        physical = self._physical_plan()
        registers = physical.registers
        scalar = physical.scalar_registers
        temporary = registers.temporary.first_register
        auxiliary = registers.auxiliary.first_register
        output_address = registers.output_address.first_register
        wave_column_base = registers.wave_column_base.first_register
        lane = registers.lane.first_register
        serial = registers.serial.first_register
        asm.inst(
            f"v_add_nc_u32 v{auxiliary}, s{scalar.row_start.first_register}, v{lane}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{output_address}, {2 * state.problem_size.n}, v{auxiliary}"
        )
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{wave_column_base}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{output_address}, v{output_address}, v{temporary}")

    def _emit_grouped_masked_tile_store(self, asm: Assembly, *, tile: int) -> None:
        physical = self._physical_plan()
        registers = physical.registers
        scalar = physical.scalar_registers
        auxiliary = registers.auxiliary.first_register
        asm.inst(
            f"v_cmp_gt_u32_e32 vcc_lo, s{scalar.row_end.first_register}, v{auxiliary}"
        )
        asm.inst(f"s_and_saveexec_b32 s{self.EXEC_MASK}, vcc_lo")
        self._emit_bf16_tile_store(asm, tile=tile)
        asm.inst(f"s_mov_b32 exec_lo, s{self.EXEC_MASK}")


def grouped_decoded_lowering(
    context: GroupedForwardLoweringContext,
) -> GroupedDecodedWeightLdsLowering:
    return GroupedDecodedWeightLdsLowering(cast(ForwardLoweringContext, context))
