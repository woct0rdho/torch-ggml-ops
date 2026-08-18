"""Typed lowering boundary for the existing MMQ backward instruction stream."""

from collections.abc import Callable

from .kernel_abi import ORDINARY_BACKWARD_ABI
from .kernel_writer_assembly import (
    emit_add_pointer,
    emit_bf16_rne,
    emit_kernel_trailer,
    emit_pointer_kernarg_loads,
    emit_scale_u32,
)
from .mmq_bwd_emission import (
    BackwardDiagnosticMode,
    PendingZeroPairableOp,
    _Assembly,
)
from .mmq_bwd_lowering_quant import BackwardQuantLowering
from .mmq_bwd_physical import (
    BackwardPhysicalPlan,
    derive_backward_physical_plan,
)
from .mmq_bwd_spec import BackwardPackedRowAddress, DerivedBackwardState
from .model import SolutionKey


class BackwardKernelLowering(BackwardQuantLowering):
    """Emit one validated backward body from the current typed solution state."""

    def __init__(
        self,
        solution_key: SolutionKey,
        *,
        diagnostic_mode: BackwardDiagnosticMode | None = None,
        label_suffix: str = "",
    ) -> None:
        self.state = DerivedBackwardState.from_solution_key(solution_key)
        self.physical: BackwardPhysicalPlan = derive_backward_physical_plan(self.state)
        self.diagnostic_mode = diagnostic_mode
        self.label_suffix = label_suffix
        self.registers = self.physical.registers

    def _label(self, name: str) -> str:
        return f".L{name}{self.label_suffix}"

    def _decode_label_suffix(self, suffix: str) -> str:
        return f"{suffix}{self.label_suffix}"

    def body(self) -> str:
        asm = _Assembly()
        r = self.registers
        name = self.state.solution_key.kernel_name

        asm.comment("Flatten gfx11 packed workitem X/Y before v0 becomes C storage.")
        asm.inst(f"v_bfe_u32 v{r.serial}, v0, 10, 10")
        asm.inst(f"v_lshlrev_b32 v{r.serial}, 5, v{r.serial}")
        asm.inst(f"v_and_b32 v{r.temporary}, 0x3ff, v0")
        asm.inst(f"v_add_nc_u32 v{r.serial}, v{r.serial}, v{r.temporary}")
        group_m = self.state.solution.work_group_mapping
        asm.comment("Map grouped M launch coordinates to the logical M tile.")
        if group_m == 1:
            asm.inst("s_mov_b32 s2, s4")
        else:
            asm.inst(f"s_mul_i32 s4, s4, {group_m}")
            asm.inst("s_add_u32 s2, s4, s2")
        kernarg = r.kernarg
        emit_pointer_kernarg_loads(asm, kernarg, ORDINARY_BACKWARD_ABI)
        self._defer_accumulator_zero(asm)
        self._emit_static_packed_coordinates(asm)
        self._emit_compute_tile(asm)
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _defer_accumulator_zero(self, asm: _Assembly) -> None:
        if self.diagnostic_mode == BackwardDiagnosticMode.DECODE_FLOOR:
            return
        r = self.registers
        accumulator_count = (
            8
            * self.state.solution.matrix_instruction[5]
            * self.state.solution.matrix_instruction[6]
        )
        asm.comment("Pair accumulator zeroing with independent pre-loop address VALU.")
        asm.defer_zero_moves(range(r.accum, r.accum + accumulator_count))

    def _emit_static_packed_coordinates(self, asm: _Assembly) -> None:
        r = self.registers
        quant_format = self.state.contract.quant_format
        n_per_block = self.state.solution.macro_tile1
        if (
            self.state.contract.mechanism.decoder.row_address
            is BackwardPackedRowAddress.Block32
        ):
            blocks_per_tile = n_per_block // 32
            asm.comment("Static Q8_0 packed-row coordinates.")
            asm.inst(
                f"s_mul_i32 s{r.block_offset}, s3, "
                f"{blocks_per_tile * quant_format.block_bytes}"
            )
            asm.inst(f"s_mov_b32 s{r.input_half}, 0")
            asm.inst(f"s_mov_b32 s{r.scalar_temporary}, 0")
        else:
            tiles_per_weight_block = 256 // n_per_block
            tile_shift = tiles_per_weight_block.bit_length() - 1
            n_shift = n_per_block.bit_length() - 1
            asm.comment("Static packed-row and input-half coordinates.")
            asm.inst(f"s_lshr_b32 s{r.block_offset}, s3, {tile_shift}")
            asm.inst(
                f"s_mul_i32 s{r.block_offset}, s{r.block_offset}, "
                f"{quant_format.block_bytes}"
            )
            asm.inst(f"s_lshr_b32 s{r.input_half}, s3, {tile_shift - 1}")
            asm.inst(f"s_and_b32 s{r.input_half}, s{r.input_half}, 1")
            asm.inst(
                f"s_and_b32 s{r.scalar_temporary}, s3, {tiles_per_weight_block - 1}"
            )
            quant_tile_shift = n_shift - 1 if n_per_block == 128 else n_shift
            asm.inst(
                f"s_lshl_b32 s{r.scalar_temporary}, s{r.scalar_temporary}, "
                f"{quant_tile_shift}"
            )

    def _emit_compute_tile(self, asm: _Assembly) -> None:
        r = self.registers
        size = self.state.contract.problem_size
        asm.inst(f"s_mov_b32 s{r.loop_counter}, 0")

        self._emit_static_thread_coordinates(asm)
        asm.flush_zero_moves()
        store_output = True
        if self.diagnostic_mode == BackwardDiagnosticMode.WMMA_FLOOR:
            self._emit_wmma_floor(asm)
        elif self.diagnostic_mode == BackwardDiagnosticMode.DECODE_FLOOR:
            self._emit_decode_floor(asm)
            store_output = False
        elif self.state.spec.pipeline.decoded_b_pipeline:
            self._emit_decoded_b_pipeline(asm)
        elif self.state.spec.pipeline.prefetches_next_packed_tile:
            self._emit_packed_weight_pipeline(asm)
        else:
            asm.label(self._label("DepthULoop"))
            if self.state.solution.num_threads > 128:
                asm.comment("Only the first four waves cooperatively decode B.")
                asm.inst(f"v_readfirstlane_b32 s{r.scalar_temporary + 1}, v{r.serial}")
                asm.inst(f"s_cmp_lt_u32 s{r.scalar_temporary + 1}, 128")
                asm.inst(f"s_cbranch_scc0 {self._label('DecodeReady')}")
            schedule = self.state.spec.pipeline.schedule
            self._emit_quant_global_reads(asm, wait_for_reads=not schedule.prefetches_a)
            if schedule.prefetches_a:
                self._emit_first_a_global_reads(asm)
                a_load_count = (
                    2
                    * self.state.solution.matrix_instruction[5]
                    * self.state.spec.pipeline.global_read_prefetch
                )
                asm.inst(f"s_waitcnt vmcnt({a_load_count})")
            self._emit_packed_weight_lane_share(asm)
            self._emit_quant_decode(asm, label_suffix=self._decode_label_suffix(""))
            if self.state.solution.num_threads > 128:
                asm.label(self._label("DecodeReady"))
            asm.inst("s_waitcnt lgkmcnt(0)")
            asm.inst("s_barrier")
            self._emit_wmma(asm)
            asm.inst("s_waitcnt lgkmcnt(0)")
            asm.inst("s_barrier")
            asm.inst(
                f"s_add_u32 s{r.loop_counter}, s{r.loop_counter}, "
                f"{self.state.solution.depth_u}"
            )
            asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
            asm.inst(f"s_cbranch_scc1 {self._label('DepthULoop')}")
        if store_output:
            self._emit_store(asm)

    def _emit_wmma_floor(self, asm: _Assembly) -> None:
        r = self.registers
        size = self.state.contract.problem_size
        solution = self.state.solution
        a_load_count = (
            2
            * solution.matrix_instruction[5]
            * self.state.spec.pipeline.global_read_prefetch
        )

        asm.comment("Prime one decoded-B tile for the WMMA/A/LDS lower bound.")
        self._emit_quant_global_reads(asm, wait_for_reads=False)
        self._emit_first_a_global_reads(asm)
        asm.inst(f"s_waitcnt vmcnt({a_load_count})")
        self._emit_quant_decode(
            asm, label_suffix=self._decode_label_suffix("WmmaFloorPrime")
        )
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")

        asm.label(self._label("WmmaFloorLoop"))
        asm.inst("s_waitcnt vmcnt(0)")
        self._emit_wmma(asm, pipeline=self.state.spec.pipeline.decoded_b_pipeline)
        asm.inst(f"s_add_u32 s{r.loop_counter}, s{r.loop_counter}, {solution.depth_u}")
        asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
        asm.inst(f"s_cbranch_scc0 {self._label('WmmaFloorDone')}")
        self._emit_first_a_global_reads(asm)
        asm.inst(f"s_branch {self._label('WmmaFloorLoop')}")
        asm.label(self._label("WmmaFloorDone"))

    def _emit_decode_floor(self, asm: _Assembly) -> None:
        r = self.registers
        size = self.state.contract.problem_size
        solution = self.state.solution
        asm.comment(
            "Measure packed Q4_K reads, decode, LDS stores, and synchronization."
        )
        asm.label(self._label("DecodeFloorLoop"))
        self._emit_quant_global_reads(asm)
        self._emit_quant_decode(
            asm, label_suffix=self._decode_label_suffix("DecodeFloor")
        )
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        asm.inst(f"s_add_u32 s{r.loop_counter}, s{r.loop_counter}, {solution.depth_u}")
        asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
        asm.inst(f"s_cbranch_scc1 {self._label('DecodeFloorLoop')}")

    def _emit_packed_weight_pipeline(self, asm: _Assembly) -> None:
        r = self.registers
        size = self.state.contract.problem_size
        solution = self.state.solution
        a_load_count = (
            2
            * self.state.solution.matrix_instruction[5]
            * self.state.spec.pipeline.global_read_prefetch
        )

        asm.comment("Prime decoded B and both A fragments.")
        self._emit_quant_global_reads(asm, wait_for_reads=False)
        self._emit_first_a_global_reads(asm)
        asm.inst(f"s_waitcnt vmcnt({a_load_count})")
        self._emit_packed_weight_lane_share(asm)
        self._emit_quant_decode(asm, label_suffix=self._decode_label_suffix("Initial"))
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")

        asm.label(self._label("PackedDepthULoop"))
        asm.inst("s_waitcnt vmcnt(0)", "current A before next packed reads")
        asm.inst(f"s_add_u32 s{r.loop_counter}, s{r.loop_counter}, {solution.depth_u}")
        if solution.depth_u != 64:
            asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
            asm.inst(f"s_cbranch_scc0 {self._label('PackedNoPrefetch')}")
            self._emit_quant_global_reads(asm, wait_for_reads=False)
            asm.label(self._label("PackedNoPrefetch"))

        self._emit_wmma(asm)
        if solution.depth_u == 64:
            asm.comment("Preserve current A addresses through the second DepthU half.")
            asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
            asm.inst(f"s_cbranch_scc0 {self._label('PackedNoLatePrefetch')}")
            self._emit_quant_global_reads(asm, wait_for_reads=False)
            asm.label(self._label("PackedNoLatePrefetch"))
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
        asm.inst(f"s_cbranch_scc0 {self._label('PackedDepthUDone')}")

        self._emit_first_a_global_reads(asm)
        asm.inst(f"s_waitcnt vmcnt({a_load_count})")
        self._emit_packed_weight_lane_share(asm)
        self._emit_quant_decode(asm, label_suffix=self._decode_label_suffix("Steady"))
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        asm.inst(f"s_branch {self._label('PackedDepthULoop')}")
        asm.label(self._label("PackedDepthUDone"))

    def _emit_decoded_b_pipeline(self, asm: _Assembly) -> None:
        r = self.registers
        size = self.state.contract.problem_size
        solution = self.state.solution
        prefetch_a = self.state.spec.pipeline.schedule.prefetches_a
        a_load_count = 0
        if prefetch_a:
            a_load_count = (
                2
                * solution.matrix_instruction[5]
                * self.state.spec.pipeline.global_read_prefetch
            )
        packed_load_count = self.physical.decoder.packed_load_count
        final_pair = solution.depth_u // 16 * (solution.matrix_instruction[6] // 2) - 1

        asm.comment("Prime decoded B0 and current-tile A.")
        self._emit_quant_global_reads(asm, wait_for_reads=False)
        if prefetch_a:
            self._emit_first_a_global_reads(asm)
        asm.inst(f"s_waitcnt vmcnt({a_load_count})")
        self._emit_quant_decode(
            asm, label_suffix=self._decode_label_suffix("PipelinePrime")
        )
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        self._emit_toggle_lds_write_buffer(asm)

        if size.k > solution.depth_u:
            asm.label(self._label("DecodedBPipelineLoop"))
            asm.inst(
                f"s_add_u32 s{r.loop_counter}, s{r.loop_counter}, {solution.depth_u}"
            )

            asm.comment("Issue next packed tile behind current-tile A.")
            self._emit_quant_global_reads(asm, wait_for_reads=False)
            asm.inst(f"s_waitcnt vmcnt({packed_load_count})")

            def after_wmma_pair(k_tile: int, n_tile: int) -> None:
                pair = (k_tile // 16) * (solution.matrix_instruction[6] // 2)
                pair += n_tile // 2
                if pair == 0:
                    asm.inst("s_waitcnt vmcnt(0)")
                    self._emit_quant_decode_prepare(
                        asm, label_suffix=self._decode_label_suffix("PipelineSteady")
                    )
                    self._emit_pipeline_lds_read_addresses(asm, 0)
                    if (
                        self.state.contract.quant_type == "Q5_K"
                        and self.state.spec.decode.q5.hoists_nibble_shift
                    ):
                        self._emit_q45_nibble_shift(asm)
                self._emit_quant_decode_chunk(asm, pair)
                if pair == final_pair and prefetch_a:
                    self._emit_next_a_half(asm, 0)
                    self._emit_next_a_half(asm, 1)

            if not prefetch_a:
                asm.inst(
                    f"s_sub_u32 s{r.loop_counter}, s{r.loop_counter}, "
                    f"{solution.depth_u}"
                )
            self._emit_wmma(
                asm,
                pipeline=True,
                after_pair=after_wmma_pair,
                current_a_loop_offset=-solution.depth_u,
            )
            if not prefetch_a:
                asm.inst(
                    f"s_add_u32 s{r.loop_counter}, s{r.loop_counter}, "
                    f"{solution.depth_u}"
                )
            asm.inst("s_waitcnt lgkmcnt(0)")
            asm.inst("s_barrier")
            self._emit_swap_lds_buffers(asm)
            asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k - solution.depth_u}")
            asm.inst(f"s_cbranch_scc1 {self._label('DecodedBPipelineLoop')}")

        asm.label(self._label("DecodedBPipelineFinal"))
        asm.inst("s_waitcnt vmcnt(0)")
        self._emit_wmma(asm, pipeline=True)

    def _emit_next_a_half(self, asm: _Assembly, k_half: int) -> None:
        r = self.registers
        solution = self.state.solution
        m_tiles = solution.matrix_instruction[5]
        pointers = tuple(r.global_read_b + m_tile for m_tile in range(m_tiles))
        if k_half == 0:
            asm.comment("Reload next-tile A0 after current A0 becomes dead.")
            asm.inst(f"s_lshl_b32 s{r.scalar_temporary + 1}, s{r.loop_counter}, 1")
            for m_tile, pointer in enumerate(pointers):
                asm.inst(
                    f"v_add_nc_u32 v{pointer}, s{r.scalar_temporary + 1}, "
                    f"v{r.address + 4 + m_tile}"
                )
        else:
            asm.comment("Reload next-tile A1 after current A1 becomes dead.")
            for pointer in pointers:
                asm.inst(f"v_add_nc_u32 v{pointer}, 32, v{pointer}")
        for m_tile, pointer in enumerate(pointers):
            valu_a = r.valu_a + 8 * (k_half * m_tiles + m_tile)
            self._emit_a_global_loads(asm, valu_a, pointer, pointer)
        if k_half == 1 and solution.depth_u == 64:
            for m_tile, pointer in enumerate(pointers):
                asm.inst(f"v_mov_b32 v{r.address + m_tile}, v{pointer}")

    def _emit_toggle_lds_write_buffer(self, asm: _Assembly) -> None:
        r = self.registers
        buffer_bytes = self.physical.lds.num_bytes // 2
        for register in range(r.lds_address, r.lds_address + 4):
            asm.inst(f"v_xor_b32 v{register}, {buffer_bytes}, v{register}")

    def _emit_swap_lds_buffers(self, asm: _Assembly) -> None:
        buffer_bytes = self.physical.lds.num_bytes // 2
        lds_address = self.physical.address.lds
        asm.inst(f"v_xor_b32 v{lds_address}, {buffer_bytes}, v{lds_address}")
        self._emit_toggle_lds_write_buffer(asm)

    def _emit_pipeline_lds_read_addresses(self, asm: _Assembly, k_tile: int) -> None:
        r = self.registers
        row_stride = self.physical.lds.row_stride_bytes
        first = r.quant_dm
        second = first + 1
        temporary = (
            r.address
            if self.state.contract.quant_type == "Q6_K"
            and self.physical.decoder.rows == 1
            else r.quant_scale
        )
        lds_address = self.physical.address.lds
        asm.inst(f"v_and_b32 v{temporary}, 15, v{r.serial}")
        emit_scale_u32(asm, temporary, row_stride, temporary)
        asm.inst(f"v_add_nc_u32 v{temporary}, v{lds_address}, v{temporary}")
        asm.inst(f"v_and_b32 v{temporary + 1}, 3, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 4, v{temporary + 1}")
        asm.inst(f"v_add_nc_u32 v{first}, v{temporary}, v{temporary + 1}")
        if k_tile >= 32:
            asm.inst(f"v_add_nc_u32 v{first}, {2 * (k_tile // 32) * 32}, v{first}")
        if k_tile % 32:
            asm.inst(f"v_xor_b32 v{first}, 32, v{first}")
        asm.inst(f"v_xor_b32 v{second}, 16, v{first}")

    def _emit_static_thread_coordinates(self, asm: _Assembly) -> None:
        r = self.registers
        a = r.address
        t = r.temporary
        n_tiles = self.state.solution.matrix_instruction[6]
        k_shift = n_tiles.bit_length() - 1
        row_stride = self.physical.lds.row_stride_bytes
        segment_stride = 16 * row_stride
        lds_address = self.physical.address.lds
        quant_shift = self.physical.address.quant_shift
        asm.comment(
            "State registers after the A row pointers hold LDS and quant state."
        )
        asm.emit_pairable_with_pending_zero(
            PendingZeroPairableOp.AND_B32,
            lds_address,
            n_tiles - 1,
            r.serial,
        )
        if segment_stride > 0 and segment_stride & (segment_stride - 1) == 0:
            asm.emit_pairable_with_pending_zero(
                PendingZeroPairableOp.LSHLREV_B32,
                lds_address,
                segment_stride.bit_length() - 1,
                lds_address,
            )
        else:
            emit_scale_u32(asm, lds_address, segment_stride, lds_address)
        asm.inst(f"v_lshrrev_b32 v{t}, {k_shift}, v{r.serial}")
        asm.emit_pairable_with_pending_zero(PendingZeroPairableOp.LSHLREV_B32, t, 1, t)
        asm.emit_pairable_with_pending_zero(
            PendingZeroPairableOp.ADD_NC_U32,
            lds_address,
            f"v{lds_address}",
            t,
        )
        if self.state.contract.mechanism.extended_quant_address_state:
            asm.emit_pairable_with_pending_zero(
                PendingZeroPairableOp.AND_B32, quant_shift, 7, r.serial
            )
            asm.inst(f"v_lshrrev_b32 v{quant_shift}, 1, v{quant_shift}")
            asm.emit_pairable_with_pending_zero(
                PendingZeroPairableOp.LSHLREV_B32,
                quant_shift,
                1,
                quant_shift,
            )
        else:
            asm.emit_pairable_with_pending_zero(
                PendingZeroPairableOp.AND_B32, quant_shift, 2, r.serial
            )
            asm.emit_pairable_with_pending_zero(
                PendingZeroPairableOp.LSHLREV_B32,
                quant_shift,
                1,
                quant_shift,
            )
        if self.state.contract.quant_type == "Q5_K":
            asm.comment("Derive the Q5_K scale-group index for the high plane.")
            if n_tiles == 8:
                asm.inst(f"v_and_b32 v{t}, 1, s3")
                asm.emit_pairable_with_pending_zero(
                    PendingZeroPairableOp.LSHLREV_B32, t, 2, t
                )
                asm.inst(f"v_bfe_u32 v{t + 1}, v{r.serial}, 2, 1")
                asm.emit_pairable_with_pending_zero(
                    PendingZeroPairableOp.LSHLREV_B32, t + 1, 1, t + 1
                )
                asm.emit_pairable_with_pending_zero(
                    PendingZeroPairableOp.ADD_NC_U32, t, f"v{t}", t + 1
                )
                asm.inst(f"v_bfe_u32 v{t + 1}, v{r.serial}, 1, 1")
                asm.emit_pairable_with_pending_zero(
                    PendingZeroPairableOp.ADD_NC_U32,
                    quant_shift,
                    f"v{t}",
                    t + 1,
                )
            else:
                asm.emit_pairable_with_pending_zero(
                    PendingZeroPairableOp.AND_B32, t, n_tiles - 1, r.serial
                )
                asm.inst(f"v_lshrrev_b32 v{t}, 1, v{t}")
                asm.inst(f"v_and_b32 v{t + 1}, {n_tiles - 1}, s3")
                asm.emit_pairable_with_pending_zero(
                    PendingZeroPairableOp.LSHLREV_B32, t + 1, 1, t + 1
                )
                asm.emit_pairable_with_pending_zero(
                    PendingZeroPairableOp.ADD_NC_U32,
                    quant_shift,
                    f"v{t}",
                    t + 1,
                )
        swizzle = self.state.solution.lds_swizzle_chunk_b
        if swizzle:
            lds = r.lds_address
            residues = 32 // swizzle
            swizzle_shift = (2 * swizzle).bit_length() - 1
            asm.comment(
                f"Precompute XOR-{swizzle} LDS store bases for "
                f"N row residues 0-{residues - 1}."
            )
            asm.inst(
                f"v_lshrrev_b32 v{t}, {k_shift + swizzle.bit_length() - 1}, v{r.serial}"
            )
            asm.emit_pairable_with_pending_zero(
                PendingZeroPairableOp.LSHLREV_B32,
                t + 1,
                swizzle_shift,
                t,
            )
            asm.inst(f"v_sub_nc_u32 v{t + 1}, v{lds_address}, v{t + 1}")
            for residue in range(residues):
                asm.inst(f"v_xor_b32 v{lds + residue}, {residue}, v{t}")
                asm.emit_pairable_with_pending_zero(
                    PendingZeroPairableOp.LSHLREV_B32,
                    lds + residue,
                    swizzle_shift,
                    lds + residue,
                )
                asm.emit_pairable_with_pending_zero(
                    PendingZeroPairableOp.ADD_NC_U32,
                    lds + residue,
                    f"v{t + 1}",
                    lds + residue,
                )
            if self.state.spec.pipeline.decoded_b_pipeline:
                asm.comment("Initialize the decoded-B read buffer to LDS0.")
                asm.inst(f"v_mov_b32 v{lds_address}, 0")

        solution = self.state.solution
        if not self.state.spec.pipeline.schedule.prefetches_a:
            return
        m_tiles = solution.matrix_instruction[5]
        m_per_wave = 16 * m_tiles
        asm.comment("Precompute A row coordinates shared by every DepthU iteration.")
        asm.inst(f"v_lshrrev_b32 v{t}, 5, v{r.serial}")
        asm.emit_pairable_with_pending_zero(
            PendingZeroPairableOp.LSHLREV_B32,
            t,
            m_per_wave.bit_length() - 1,
            t,
        )
        asm.emit_pairable_with_pending_zero(
            PendingZeroPairableOp.AND_B32, t + 1, 15, r.serial
        )
        asm.emit_pairable_with_pending_zero(
            PendingZeroPairableOp.ADD_NC_U32, a + 4, f"v{t}", t + 1
        )
        asm.inst(f"v_lshlrev_b32 v{t + 2}, {solution.macro_tile0.bit_length() - 1}, s2")
        asm.emit_pairable_with_pending_zero(
            PendingZeroPairableOp.ADD_NC_U32, a + 4, f"v{a + 4}", t + 2
        )
        for m_tile in range(1, m_tiles):
            asm.emit_pairable_with_pending_zero(
                PendingZeroPairableOp.ADD_NC_U32,
                a + 4 + m_tile,
                16 * m_tile,
                a + 4,
            )
        row_stride_a = 2 * self.state.contract.problem_size.k
        for m_tile in range(m_tiles):
            pointer = a + 4 + m_tile
            if row_stride_a > 0 and row_stride_a & (row_stride_a - 1) == 0:
                asm.emit_pairable_with_pending_zero(
                    PendingZeroPairableOp.LSHLREV_B32,
                    pointer,
                    row_stride_a.bit_length() - 1,
                    pointer,
                )
            else:
                emit_scale_u32(asm, pointer, row_stride_a, pointer)

    def _emit_wmma(
        self,
        asm: _Assembly,
        *,
        pipeline: bool = False,
        after_pair: Callable[[int, int], None] | None = None,
        current_a_loop_offset: int = 0,
    ) -> None:
        r = self.registers
        solution = self.state.solution
        m_tiles = solution.matrix_instruction[5]
        n_tiles = solution.matrix_instruction[6]
        m_per_wave = 16 * m_tiles
        size = self.state.contract.problem_size
        a = r.address
        t = r.temporary
        asm.comment("Load A and issue two DepthU=16 WMMA halves.")
        if not self.state.spec.pipeline.schedule.prefetches_a:
            asm.inst(f"v_lshrrev_b32 v{t}, 5, v{r.serial}")
            asm.inst(f"v_lshlrev_b32 v{t}, {m_per_wave.bit_length() - 1}, v{t}")
            asm.inst(f"v_and_b32 v{t + 1}, 15, v{r.serial}")
            asm.inst(f"v_add_nc_u32 v{a + 4}, v{t}, v{t + 1}")
            asm.inst(
                f"v_lshlrev_b32 v{t + 2}, {solution.macro_tile0.bit_length() - 1}, s2"
            )
            asm.inst(f"v_add_nc_u32 v{a + 4}, v{a + 4}, v{t + 2}")
            if m_tiles == 2:
                asm.inst(f"v_add_nc_u32 v{a + 5}, 16, v{a + 4}")
        else:
            asm.comment("Reuse prefetched A pointers across fused B decode.")
        for k_tile in range(0, solution.depth_u, 16):
            if self.state.spec.pipeline.schedule.prefetches_a:
                if k_tile and self.state.spec.pipeline.global_read_prefetch == 1:
                    for m_tile in range(m_tiles):
                        pointer = a + m_tile
                        asm.inst(f"v_add_nc_u32 v{pointer}, {2 * k_tile}, v{pointer}")
                        valu_a = r.valu_a + 8 * m_tile
                        self._emit_a_global_loads(asm, valu_a, pointer, pointer)
                elif (
                    solution.depth_u == 64
                    and self.state.spec.pipeline.global_read_prefetch == 2
                    and k_tile == 32
                ):
                    if current_a_loop_offset:
                        asm.comment(
                            "Restore current-tile A pointers after next-B address setup."
                        )
                        asm.inst(
                            f"s_lshl_b32 s{r.scalar_temporary + 1}, "
                            f"s{r.loop_counter}, 1"
                        )
                        byte_rewind = -2 * current_a_loop_offset - 32
                        asm.inst(
                            f"s_sub_u32 s{r.scalar_temporary + 1}, "
                            f"s{r.scalar_temporary + 1}, {byte_rewind}"
                        )
                        for m_tile in range(m_tiles):
                            asm.inst(
                                f"v_add_nc_u32 v{a + m_tile}, "
                                f"s{r.scalar_temporary + 1}, v{a + 4 + m_tile}"
                            )
                    for k_half in range(2):
                        for m_tile in range(m_tiles):
                            pointer = a + m_tile
                            asm.inst(f"v_add_nc_u32 v{pointer}, 32, v{pointer}")
                            valu_a = r.valu_a + 8 * (k_half * m_tiles + m_tile)
                            self._emit_a_global_loads(asm, valu_a, pointer, pointer)
            elif m_tiles == 2:
                asm.inst(f"s_lshl_b32 s{r.scalar_temporary + 1}, s{r.loop_counter}, 1")
                for m_tile, row, pointer in (
                    (0, a + 4, a),
                    (1, a + 5, a + 2),
                ):
                    emit_scale_u32(asm, t, 2 * size.k, row)
                    asm.inst(f"v_add_nc_u32 v{t}, s{r.scalar_temporary + 1}, v{t}")
                    if k_tile:
                        asm.inst(f"v_add_nc_u32 v{t}, {2 * k_tile}, v{t}")
                    emit_add_pointer(asm, pointer, r.kernarg, t)
                for m_tile, pointer, row in (
                    (0, a, a + 4),
                    (1, a + 2, a + 5),
                ):
                    valu_a = r.valu_a + 8 * m_tile
                    self._emit_a_global_loads(
                        asm,
                        valu_a,
                        pointer,
                        row,
                        address_pair=True,
                        offset_is_bytes=False,
                    )
            else:
                asm.inst(f"s_lshl_b32 s{r.scalar_temporary + 1}, s{r.loop_counter}, 1")
                for m_tile in range(m_tiles):
                    if m_tile:
                        asm.inst(f"v_add_nc_u32 v{t}, {16 * m_tile}, v{a + 4}")
                        row = t
                    else:
                        row = a + 4
                    emit_scale_u32(asm, t, 2 * size.k, row)
                    asm.inst(f"v_add_nc_u32 v{t}, s{r.scalar_temporary + 1}, v{t}")
                    if k_tile:
                        asm.inst(f"v_add_nc_u32 v{t}, {2 * k_tile}, v{t}")
                    emit_add_pointer(asm, a, r.kernarg, t)
                    valu_a = r.valu_a + 8 * m_tile
                    self._emit_a_global_loads(
                        asm,
                        valu_a,
                        a,
                        row if row != t else t,
                        address_pair=True,
                        offset_is_bytes=row == t,
                    )
            lds_arguments = self._emit_lds_read_arguments(
                asm, k_tile, pipeline=pipeline
            )
            valu_a_base = r.valu_a
            if self.state.spec.pipeline.uses_two_global_reads:
                k_half = (k_tile // 16) % self.state.spec.pipeline.global_read_prefetch
                valu_a_base += 8 * m_tiles * k_half
            if self.state.spec.pipeline.uses_prefetched_local_read:
                self._emit_prefetched_wmma_pairs(
                    asm,
                    n_tiles,
                    lds_arguments,
                )
            else:
                for n_tile in range(0, n_tiles, 2):
                    first = r.valu_b
                    second = r.valu_b + 8
                    pair_lds_arguments = lds_arguments
                    if after_pair is not None and n_tile:
                        pair_lds_arguments = (
                            solution.lds_swizzle_chunk_b,
                            (),
                            r.quant_dm,
                            r.quant_dm + 1,
                            0,
                            self.physical.lds.row_stride_bytes,
                        )
                    load_count = self._emit_lds_pair(
                        asm,
                        n_tile,
                        first,
                        second,
                        *pair_lds_arguments,
                    )
                    if self.state.spec.pipeline.schedule.interleaves_wmma_waits:
                        trailing_a_loads = (
                            2 * m_tiles
                            if self.state.spec.pipeline.uses_two_global_reads
                            and k_tile == 0
                            else 0
                        )
                        self._emit_sia3_wmma_pair(
                            asm,
                            n_tile,
                            first,
                            second,
                            first_pair=n_tile == 0,
                            pending_second_loads=load_count // 2,
                            valu_a_base=valu_a_base,
                            trailing_a_loads=trailing_a_loads,
                        )
                    else:
                        if self.state.spec.pipeline.schedule.uses_sia4_waits:
                            needs_depth64_a_wait = (
                                (
                                    pipeline
                                    or self.state.spec.pipeline.prefetches_next_packed_tile
                                )
                                and solution.depth_u == 64
                                and self.state.spec.pipeline.uses_two_global_reads
                                and n_tile == 0
                            )
                            if needs_depth64_a_wait:
                                pending_a = 2 * m_tiles if k_tile % 32 == 0 else 0
                                asm.inst(f"s_waitcnt vmcnt({pending_a}) lgkmcnt(0)")
                            elif (
                                pipeline
                                or self.state.spec.pipeline.prefetches_next_packed_tile
                            ):
                                asm.inst("s_waitcnt lgkmcnt(0)")
                            elif n_tile == 0:
                                pending_a = (
                                    2
                                    * m_tiles
                                    * (
                                        self.state.spec.pipeline.global_read_prefetch
                                        - 1
                                    )
                                    if k_tile
                                    % (
                                        16
                                        * self.state.spec.pipeline.global_read_prefetch
                                    )
                                    == 0
                                    else 0
                                )
                                asm.inst(f"s_waitcnt vmcnt({pending_a}) lgkmcnt(0)")
                            else:
                                asm.inst("s_waitcnt lgkmcnt(0)")
                        else:
                            asm.inst("s_waitcnt vmcnt(0) lgkmcnt(0)")
                        self._emit_wmma_pair(
                            asm,
                            n_tile,
                            first,
                            valu_a_base=valu_a_base,
                        )
                        self._emit_wmma_pair(
                            asm,
                            n_tile + 1,
                            second,
                            valu_a_base=valu_a_base,
                        )
                    if after_pair is not None:
                        after_pair(k_tile, n_tile)

    def _emit_lds_read_arguments(
        self,
        asm: _Assembly,
        k_tile: int,
        *,
        pipeline: bool,
    ) -> tuple[int, tuple[int, ...], int, int, int, int]:
        r = self.registers
        solution = self.state.solution
        row_stride = self.physical.lds.row_stride_bytes
        if pipeline and k_tile:
            self._emit_pipeline_lds_read_addresses(asm, k_tile)
            return (
                solution.lds_swizzle_chunk_b,
                (),
                r.quant_dm,
                r.quant_dm + 1,
                0,
                row_stride,
            )

        t = r.temporary
        asm.inst(f"v_and_b32 v{t}, 15, v{r.serial}")
        emit_scale_u32(asm, t, row_stride, t)
        if pipeline:
            asm.inst(f"v_add_nc_u32 v{t}, v{self.physical.address.lds}, v{t}")
        swizzle = solution.lds_swizzle_chunk_b
        chunk_addresses: tuple[int, ...] = ()
        if swizzle == 4:
            asm.inst(f"v_and_b32 v{t + 1}, 7, v{r.serial}")
            chunk_addresses = tuple(t + 2 + chunk for chunk in range(4))
            for chunk, chunk_address in enumerate(chunk_addresses):
                logical_chunk = k_tile // 4 + chunk
                asm.inst(f"v_xor_b32 v{chunk_address}, {logical_chunk}, v{t + 1}")
                asm.inst(f"v_lshlrev_b32 v{chunk_address}, 3, v{chunk_address}")
                asm.inst(f"v_add_nc_u32 v{chunk_address}, v{t}, v{chunk_address}")
            first_address = -1
            second_address = -1
            second_chunk_offset = 0
        elif swizzle in (8, 16):
            mask = 32 // swizzle - 1
            byte_shift = (2 * swizzle).bit_length() - 1
            asm.inst(f"v_and_b32 v{t + 1}, {mask}, v{r.serial}")
            asm.inst(f"v_lshlrev_b32 v{t + 1}, {byte_shift}, v{t + 1}")
            asm.inst(f"v_add_nc_u32 v{t + 2}, v{t}, v{t + 1}")
            if k_tile >= 32:
                asm.inst(f"v_add_nc_u32 v{t + 2}, {2 * (k_tile // 32) * 32}, v{t + 2}")
            if k_tile % 32:
                asm.inst(f"v_xor_b32 v{t + 2}, 32, v{t + 2}")
            asm.inst(f"v_xor_b32 v{t + 3}, 16, v{t + 2}")
            first_address = t + 2
            second_address = t + 3
            second_chunk_offset = 0
        else:
            if k_tile:
                asm.inst(f"v_add_nc_u32 v{t}, {2 * k_tile}, v{t}")
            first_address = t
            second_address = t
            second_chunk_offset = 16
        return (
            swizzle,
            chunk_addresses,
            first_address,
            second_address,
            second_chunk_offset,
            row_stride,
        )

    @staticmethod
    def _emit_lds_pair(
        asm: _Assembly,
        n_tile: int,
        first: int,
        second: int,
        swizzle: int,
        chunk_addresses: tuple[int, ...],
        first_address: int,
        second_address: int,
        second_chunk_offset: int,
        row_stride: int,
    ) -> int:
        tile_stride = 16 * row_stride
        first_offset = tile_stride * n_tile
        second_offset = first_offset + tile_stride
        if swizzle == 4:
            for chunk, chunk_address in enumerate(chunk_addresses):
                destination = first + 2 * chunk
                asm.inst(
                    f"ds_load_b64 v[{destination}:{destination + 1}], "
                    f"v{chunk_address} offset:{first_offset}"
                )
            for chunk, chunk_address in enumerate(chunk_addresses):
                destination = second + 2 * chunk
                asm.inst(
                    f"ds_load_b64 v[{destination}:{destination + 1}], "
                    f"v{chunk_address} offset:{second_offset}"
                )
            return 8

        asm.inst(
            f"ds_load_b128 v[{first}:{first + 3}], "
            f"v{first_address} offset:{first_offset}"
        )
        asm.inst(
            f"ds_load_b128 v[{first + 4}:{first + 7}], "
            f"v{second_address} offset:{first_offset + second_chunk_offset}"
        )
        asm.inst(
            f"ds_load_b128 v[{second}:{second + 3}], "
            f"v{first_address} offset:{second_offset}"
        )
        asm.inst(
            f"ds_load_b128 v[{second + 4}:{second + 7}], "
            f"v{second_address} offset:{second_offset + second_chunk_offset}"
        )
        return 4

    def _emit_prefetched_wmma_pairs(
        self,
        asm: _Assembly,
        n_tiles: int,
        lds_arguments: tuple[int, tuple[int, ...], int, int, int, int],
    ) -> None:
        r = self.registers
        n_pairs = n_tiles // 2
        self._emit_lds_pair(
            asm,
            0,
            r.valu_b,
            r.valu_b + 8,
            *lds_arguments,
        )
        for pair in range(n_pairs):
            n_tile = 2 * pair
            buffer = 16 * (pair % 2)
            first = r.valu_b + buffer
            second = first + 8
            pending_loads = 0
            if pair + 1 < n_pairs:
                next_buffer = 16 * ((pair + 1) % 2)
                pending_loads = self._emit_lds_pair(
                    asm,
                    n_tile + 2,
                    r.valu_b + next_buffer,
                    r.valu_b + next_buffer + 8,
                    *lds_arguments,
                )
            self._emit_prefetched_wmma_pair(
                asm,
                n_tile,
                first,
                second,
                first_pair=pair == 0,
                pending_loads=pending_loads,
            )

    def _emit_prefetched_wmma_pair(
        self,
        asm: _Assembly,
        n_tile: int,
        first: int,
        second: int,
        *,
        first_pair: bool,
        pending_loads: int,
    ) -> None:
        r = self.registers
        m_tiles = self.state.solution.matrix_instruction[5]
        if first_pair:
            pending_a_loads = 2 * (m_tiles - 1)
            asm.inst(f"s_waitcnt vmcnt({pending_a_loads}) lgkmcnt({pending_loads + 2})")
            self._emit_wmma_instruction(asm, 0, n_tile, r.valu_a, first)
            asm.inst(f"s_waitcnt lgkmcnt({pending_loads})")
            self._emit_wmma_instruction(asm, 0, n_tile + 1, r.valu_a, second)
            for m_tile in range(1, m_tiles):
                pending_a_loads -= 2
                asm.inst(f"s_waitcnt vmcnt({pending_a_loads})")
                valu_a = r.valu_a + 8 * m_tile
                self._emit_wmma_instruction(asm, m_tile, n_tile, valu_a, first)
                self._emit_wmma_instruction(asm, m_tile, n_tile + 1, valu_a, second)
            return

        asm.inst(f"s_waitcnt lgkmcnt({pending_loads + 2})")
        self._emit_wmma_pair(asm, n_tile, first)
        asm.inst(f"s_waitcnt lgkmcnt({pending_loads})")
        self._emit_wmma_pair(asm, n_tile + 1, second)

    def _emit_sia3_wmma_pair(
        self,
        asm: _Assembly,
        n_tile: int,
        first: int,
        second: int,
        *,
        first_pair: bool,
        pending_second_loads: int,
        valu_a_base: int,
        trailing_a_loads: int,
    ) -> None:
        m_tiles = self.state.solution.matrix_instruction[5]
        if first_pair:
            pending_a_loads = 2 * (m_tiles - 1) + trailing_a_loads
            asm.inst(
                f"s_waitcnt vmcnt({pending_a_loads}) lgkmcnt({pending_second_loads})"
            )
            self._emit_wmma_instruction(asm, 0, n_tile, valu_a_base, first)
            asm.inst("s_waitcnt lgkmcnt(0)")
            self._emit_wmma_instruction(asm, 0, n_tile + 1, valu_a_base, second)
            for m_tile in range(1, m_tiles):
                pending_a_loads -= 2
                asm.inst(f"s_waitcnt vmcnt({pending_a_loads})")
                valu_a = valu_a_base + 8 * m_tile
                self._emit_wmma_instruction(asm, m_tile, n_tile, valu_a, first)
                self._emit_wmma_instruction(asm, m_tile, n_tile + 1, valu_a, second)
            return

        asm.inst(f"s_waitcnt lgkmcnt({pending_second_loads})")
        for m_tile in range(m_tiles):
            valu_a = valu_a_base + 8 * m_tile
            self._emit_wmma_instruction(asm, m_tile, n_tile, valu_a, first)
        asm.inst("s_waitcnt lgkmcnt(0)")
        for m_tile in range(m_tiles):
            valu_a = valu_a_base + 8 * m_tile
            self._emit_wmma_instruction(asm, m_tile, n_tile + 1, valu_a, second)

    def _emit_wmma_pair(
        self,
        asm: _Assembly,
        n_tile: int,
        valu_b: int,
        *,
        valu_a_base: int | None = None,
    ) -> None:
        r = self.registers
        m_tiles = self.state.solution.matrix_instruction[5]
        if valu_a_base is None:
            valu_a_base = r.valu_a
        for m_tile in range(m_tiles):
            valu_a = valu_a_base + 8 * m_tile
            self._emit_wmma_instruction(asm, m_tile, n_tile, valu_a, valu_b)

    def _emit_wmma_instruction(
        self,
        asm: _Assembly,
        m_tile: int,
        n_tile: int,
        valu_a: int,
        valu_b: int,
    ) -> None:
        r = self.registers
        n_tiles = self.state.solution.matrix_instruction[6]
        accum = r.accum + (n_tiles * m_tile + n_tile) * 8
        asm.inst(
            f"v_wmma_f32_16x16x16_bf16 v[{accum}:{accum + 7}], "
            f"v[{valu_a}:{valu_a + 7}], v[{valu_b}:{valu_b + 7}], "
            f"v[{accum}:{accum + 7}]"
        )

    def _emit_store(self, asm: _Assembly) -> None:
        r = self.registers
        solution = self.state.solution
        m_tiles = solution.matrix_instruction[5]
        n_tiles = solution.matrix_instruction[6]
        m_per_wave = 16 * m_tiles
        size = self.state.contract.problem_size
        a = r.address
        t = r.temporary
        asm.comment("Map gfx11 physical C fragments to row-major grad_input.")
        asm.inst(f"v_lshrrev_b32 v{t}, 5, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, {m_per_wave.bit_length() - 1}, v{t}")
        asm.inst(f"v_lshrrev_b32 v{t + 1}, 4, v{r.serial}")
        asm.inst(f"v_and_b32 v{t + 1}, 1, v{t + 1}")
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, {solution.macro_tile0.bit_length() - 1}, s2")
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        self._emit_store_row_begin(asm, t)
        emit_scale_u32(asm, t, 2 * size.n, t)
        asm.inst(
            f"v_lshlrev_b32 v{t + 1}, {(2 * solution.macro_tile1).bit_length() - 1}, s3"
        )
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        asm.inst(f"v_and_b32 v{t + 1}, 15, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, 1, v{t + 1}")
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        asm.inst(f"v_mov_b32 v{a}, v{t}")
        if self.state.spec.store.raises_priority:
            asm.inst("s_setprio 1")
        for m_tile in range(m_tiles):
            for element in range(8):
                self._emit_store_row_mask_begin(asm)
                for n_tile in range(n_tiles):
                    accum = r.accum + (n_tiles * m_tile + n_tile) * 8 + element
                    emit_bf16_rne(asm, accum, t + 2)
                    asm.inst(
                        f"global_store_d16_hi_b16 v{a}, v{accum}, "
                        f"s[{r.kernarg + 4}:{r.kernarg + 5}] offset:{32 * n_tile}"
                    )
                self._emit_store_row_mask_end(asm)
                if not (m_tile == m_tiles - 1 and element == 7):
                    asm.inst(f"v_add_nc_u32 v{a}, {4 * size.n}, v{a}")
                    self._emit_store_row_advance(asm)
        if self.state.spec.store.raises_priority:
            asm.inst("s_setprio 0")

    def _emit_store_row_begin(self, asm: _Assembly, row: int) -> None:
        del asm, row

    def _emit_store_row_mask_begin(self, asm: _Assembly) -> None:
        del asm

    def _emit_store_row_mask_end(self, asm: _Assembly) -> None:
        del asm

    def _emit_store_row_advance(self, asm: _Assembly) -> None:
        del asm
