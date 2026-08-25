"""Fused two-projection lowering for grouped MMQ backward."""

from .grouped_mmq_bwd_lowering import (
    GroupedBackwardTileAccess,
    GroupedBackwardTileComputeEmitter,
)
from .grouped_mmq_bwd_pair_physical import (
    GroupedBackwardPairPhysicalPlan,
    GroupedBackwardPairScalarPlan,
)
from .grouped_mmq_bwd_pair_spec import DerivedGroupedBackwardPairState
from .kernel_abi import GROUPED_BACKWARD_PAIR_ABI
from .kernel_writer_assembly import (
    LoweringResult,
    emit_kernel_trailer,
    emit_scale_sgpr_u32,
)
from .mmq_bwd_emission import _Assembly
from .mmq_bwd_lowering_quant import UnboundedBackwardTileAccess
from .work_group_mapping import mapped_route_stride, work_group_mapping_shift


class GroupedBackwardPairTileComputeEmitter(GroupedBackwardTileComputeEmitter):
    """Accumulate two interleaved projections before one final store."""

    def __init__(
        self,
        state,
        physical,
        *,
        route: GroupedBackwardPairScalarPlan,
        label_suffix: str = "",
        bounded: bool = True,
        concurrent_reads: bool = False,
        prefetch_pair_a: bool = False,
        direct_second_pointers: bool = False,
        overlap_second_read_prefetch_a: bool = False,
        k_pipeline: bool = False,
    ) -> None:
        super().__init__(
            state,
            physical,
            access=(
                GroupedBackwardTileAccess(route)
                if bounded
                else UnboundedBackwardTileAccess()
            ),
            route=route,
            enabled=bounded,
            label_suffix=label_suffix,
            clause_batch_store=(
                k_pipeline and not physical.lds.codebook_in_lds and not bounded
            ),
        )
        self.pair_route = route
        self.concurrent_reads = concurrent_reads
        self.prefetch_pair_a = prefetch_pair_a
        self.direct_second_pointers = direct_second_pointers
        self.overlap_second_read_prefetch_a = overlap_second_read_prefetch_a
        self.k_pipeline = k_pipeline
        self._pipeline_wait_index = 0
        self.second_projection: GroupedBackwardPairTileComputeEmitter | None = None

    def _emit_compute_tile(self, asm: _Assembly) -> None:
        if self.second_projection is not None:
            self._emit_dual_lds_compute_tile(asm, self.second_projection)
            return
        r = self.registers
        size = self.state.contract.problem_size
        geometry = self.state.spec.geometry
        asm.inst(f"s_mov_b32 s{r.loop_counter}, 0")
        self._emit_static_thread_coordinates(asm)
        asm.flush_zero_moves()

        asm.label(self._label("PairDepthULoop"))
        self._emit_projection(asm, "First")
        self._swap_projection_pointers(asm)
        self._emit_projection(asm, "Second")
        self._swap_projection_pointers(asm)
        asm.inst(f"s_add_u32 s{r.loop_counter}, s{r.loop_counter}, {geometry.depth_u}")
        asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
        asm.inst(f"s_cbranch_scc1 {self._label('PairDepthULoop')}")
        self._emit_store(asm)

    def _emit_dual_lds_compute_tile(
        self,
        asm: _Assembly,
        second: "GroupedBackwardPairTileComputeEmitter",
    ) -> None:
        if self.k_pipeline:
            self._emit_dual_lds_k_pipeline(asm, second)
            return
        r = self.registers
        size = self.state.contract.problem_size
        geometry = self.state.spec.geometry
        asm.inst(f"s_mov_b32 s{r.loop_counter}, 0")
        self._emit_static_thread_coordinates(asm)
        asm.flush_zero_moves()

        asm.label(self._label("PairDepthULoop"))
        if self.concurrent_reads:
            self._emit_concurrent_projection_decodes(asm, second)
        elif self.direct_second_pointers:
            self._emit_projection_decode(asm, "First")
            if self.overlap_second_read_prefetch_a:
                asm.comment(
                    "Issue the second packed projection reads alongside first A prefetch."
                )
                second._emit_quant_global_reads(asm, wait_for_reads=False)
                self._emit_first_a_global_reads(asm)
                asm.inst("s_waitcnt vmcnt(0)")
                second._emit_packed_weight_lane_share(asm)
                second._emit_quant_decode(
                    asm,
                    label_suffix=second._decode_label_suffix("PairSecond"),
                )
            else:
                second._emit_projection_decode(asm, "Second")
                if self.prefetch_pair_a:
                    self._emit_first_a_global_reads(asm)
        else:
            self._emit_projection_decode(asm, "First")
            self._swap_projection_pointers(asm)
            second._emit_projection_decode(asm, "Second")
            self._swap_projection_pointers(asm)
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        if self.prefetch_pair_a:

            def prefetch_second_a(k_tile: int) -> None:
                if self.direct_second_pointers:
                    second._emit_next_a_half(asm, k_tile // 16)
                else:
                    self._swap_grad_output_pointer(asm)
                    self._emit_next_a_half(asm, k_tile // 16)
                    self._swap_grad_output_pointer(asm)

            pending_second_a_half = 2 * self.state.spec.geometry.matrix_instruction[5]
            self._emit_wmma(
                asm,
                after_k_half=prefetch_second_a,
                pending_vmem_by_k_tile={16: pending_second_a_half},
            )
        else:
            self._emit_wmma(asm)
        if self.direct_second_pointers:
            second._emit_wmma(asm)
        elif self.concurrent_reads:
            self._swap_grad_output_pointer(asm)
            second._emit_wmma(asm)
            self._swap_grad_output_pointer(asm)
        else:
            self._swap_projection_pointers(asm)
            second._emit_wmma(asm)
            self._swap_projection_pointers(asm)
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        asm.inst(f"s_add_u32 s{r.loop_counter}, s{r.loop_counter}, {geometry.depth_u}")
        asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
        asm.inst(f"s_cbranch_scc1 {self._label('PairDepthULoop')}")
        self._emit_store(asm)

    def _emit_dual_lds_k_pipeline(
        self,
        asm: _Assembly,
        second: "GroupedBackwardPairTileComputeEmitter",
    ) -> None:
        r = self.registers
        size = self.state.contract.problem_size
        depth_u = self.state.spec.geometry.depth_u
        asm.inst(f"s_mov_b32 s{r.loop_counter}, 0")
        self._emit_static_thread_coordinates(asm)
        asm.flush_zero_moves()

        asm.comment("Prime packed weights and first-projection A for K0.")
        self._emit_concurrent_projection_decodes(asm, second)
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")

        asm.label(self._label("PairKPipelineCompute"))

        def prefetch_second_a(k_tile: int) -> None:
            second._emit_next_a_half(asm, k_tile // 16)

        pending_second_a_half = 2 * self.state.spec.geometry.matrix_instruction[5]
        self._emit_wmma(
            asm,
            after_k_half=prefetch_second_a,
            pending_vmem_by_k_tile={16: pending_second_a_half},
        )

        asm.inst(f"s_add_u32 s{r.loop_counter}, s{r.loop_counter}, {depth_u}")
        asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
        asm.inst(f"s_cbranch_scc1 {self._label('PairKPipelineNextPacked')}")
        asm.inst("s_waitcnt vmcnt(0)")
        asm.inst(f"s_branch {self._label('PairKPipelinePackedDone')}")
        asm.label(self._label("PairKPipelineNextPacked"))
        self._emit_quant_global_reads(asm, wait_for_reads=False)
        second._emit_quant_global_reads_from_current_addresses(asm)
        asm.label(self._label("PairKPipelinePackedDone"))

        def prefetch_next_a(k_tile: int) -> None:
            k_half = k_tile // 16
            asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
            asm.inst(f"s_cbranch_scc0 {self._label(f'PairKPipelineNoNextA{k_half}')}")
            if self.physical.lds.codebook_in_lds:
                self._emit_a_half_with_safe_pointers(asm, k_half)
            asm.label(self._label(f"PairKPipelineNoNextA{k_half}"))

        second._emit_wmma(
            asm,
            after_k_half=(
                prefetch_next_a if self.physical.lds.codebook_in_lds else None
            ),
            pending_vmem_by_k_tile={
                0: pending_second_a_half,
                16: pending_second_a_half,
            },
        )
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
        asm.inst(f"s_cbranch_scc0 {self._label('PairKPipelineDone')}")

        asm.comment("Decode prefetched banks after their LDS consumers retire.")
        if self.physical.lds.codebook_in_lds:
            if self.state.contract.quant_type == "IQ2_XXS":
                next_a_loads = 4 * self.state.spec.geometry.matrix_instruction[5]
                self._emit_k_pipeline_wait(
                    asm,
                    active_vmcnt=next_a_loads,
                    inactive_vmcnt=0,
                )
                self._emit_packed_weight_lane_share(asm)
                self._emit_concurrent_iq2_xxs_codebook_decodes(
                    asm,
                    second,
                    a_load_count=next_a_loads,
                )
            else:
                self._emit_k_pipeline_wait(asm, active_vmcnt=13, inactive_vmcnt=5)
                self._emit_packed_weight_lane_share(asm)
                self._emit_quant_decode(
                    asm,
                    label_suffix=self._decode_label_suffix("PairPipelineFirst"),
                )
                self._emit_k_pipeline_wait(asm, active_vmcnt=8, inactive_vmcnt=0)
                second._emit_packed_weight_lane_share(asm)
                second._emit_quant_decode(
                    asm,
                    label_suffix=second._decode_label_suffix("PairPipelineSecond"),
                )
        else:
            asm.inst("s_waitcnt vmcnt(0)")
            self._emit_batched_global_codebook_issue(asm, second)
            self._emit_a_half_with_safe_pointers(asm, 0)
            self._emit_a_half_with_safe_pointers(asm, 1)
            self._emit_k_pipeline_wait(asm, active_vmcnt=8, inactive_vmcnt=0)
            self._emit_batched_global_codebook_finish(asm, second)
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        asm.inst(f"s_branch {self._label('PairKPipelineCompute')}")

        asm.label(self._label("PairKPipelineDone"))
        self._emit_store(asm)

    def _emit_k_pipeline_wait(
        self,
        asm: _Assembly,
        *,
        active_vmcnt: int,
        inactive_vmcnt: int,
    ) -> None:
        if not self.enabled:
            asm.inst(f"s_waitcnt vmcnt({active_vmcnt})")
            return
        wait_index = self._pipeline_wait_index
        self._pipeline_wait_index += 1
        inactive = self._label(f"PairKPipelineWaitInactive{active_vmcnt}_{wait_index}")
        done = self._label(f"PairKPipelineWaitDone{active_vmcnt}_{wait_index}")
        asm.inst(f"s_cmp_eq_u32 s{self._active_m_tiles}, 0")
        asm.inst(f"s_cbranch_scc1 {inactive}")
        asm.inst(f"s_waitcnt vmcnt({active_vmcnt})")
        asm.inst(f"s_branch {done}")
        asm.label(inactive)
        asm.inst(f"s_waitcnt vmcnt({inactive_vmcnt})")
        asm.label(done)

    def _emit_a_half_with_safe_pointers(self, asm: _Assembly, k_half: int) -> None:
        r = self.registers
        m_tiles = self.state.spec.geometry.matrix_instruction[5]
        pointers = tuple(r.temporary + 4 + m_tile for m_tile in range(m_tiles))
        if k_half == 0:
            asm.inst(f"s_lshl_b32 s{r.scalar_temporary + 1}, s{r.loop_counter}, 1")
            for m_tile, pointer in enumerate(pointers):
                asm.inst(
                    f"v_add_nc_u32 v{pointer}, s{r.scalar_temporary + 1}, "
                    f"v{r.address + 4 + m_tile}"
                )
        else:
            for pointer in pointers:
                asm.inst(f"v_add_nc_u32 v{pointer}, 32, v{pointer}")
        for m_tile, pointer in enumerate(pointers):
            valu_a = r.valu_a + 8 * (k_half * m_tiles + m_tile)
            self._emit_a_global_loads(asm, valu_a, pointer, pointer)

    def _emit_concurrent_projection_decodes(
        self,
        asm: _Assembly,
        second: "GroupedBackwardPairTileComputeEmitter",
    ) -> None:
        asm.comment("Issue both packed projection reads before either decode.")
        self._emit_quant_global_reads(asm, wait_for_reads=False)
        if self.direct_second_pointers:
            second._emit_quant_global_reads_from_current_addresses(asm)
        else:
            self._swap_packed_weight_pointer(asm)
            second._emit_quant_global_reads(asm, wait_for_reads=False)
            self._swap_packed_weight_pointer(asm)
        a_load_count = 0
        if self.prefetch_pair_a:
            a_load_count = (
                2
                * self.state.spec.geometry.matrix_instruction[5]
                * self.state.spec.pipeline.global_read_prefetch
            )
        if not self.physical.lds.codebook_in_lds:
            asm.inst("s_waitcnt vmcnt(0)")
            self._emit_batched_global_codebook_issue(asm, second)
            if self.prefetch_pair_a:
                self._emit_first_a_global_reads(asm)
            self._emit_k_pipeline_wait(
                asm,
                active_vmcnt=a_load_count,
                inactive_vmcnt=0,
            )
            self._emit_batched_global_codebook_finish(asm, second)
            return
        if self.prefetch_pair_a:
            self._emit_first_a_global_reads(asm)
        asm.inst(
            f"s_waitcnt vmcnt("
            f"{second.physical.decoder.packed_load_count + a_load_count})"
        )
        self._emit_packed_weight_lane_share(asm)
        if self.state.contract.quant_type == "IQ2_XXS":
            self._emit_concurrent_iq2_xxs_codebook_decodes(
                asm,
                second,
                a_load_count=a_load_count,
            )
            return
        self._emit_quant_decode(
            asm,
            label_suffix=self._decode_label_suffix("PairFirst"),
        )
        asm.inst(f"s_waitcnt vmcnt({a_load_count})")
        second._emit_packed_weight_lane_share(asm)
        second._emit_quant_decode(
            asm,
            label_suffix=second._decode_label_suffix("PairSecond"),
        )

    def _emit_concurrent_iq2_xxs_codebook_decodes(
        self,
        asm: _Assembly,
        second: "GroupedBackwardPairTileComputeEmitter",
        *,
        a_load_count: int,
    ) -> None:
        first_lane_group = self.registers.temporary + 6
        second_lane_group = second.registers.temporary + 5
        self._emit_iq2_xxs_codebook_reads(asm, first_lane_group)
        asm.inst(f"s_waitcnt vmcnt({a_load_count})")
        second._emit_packed_weight_lane_share(asm)
        second._emit_iq2_xxs_codebook_reads(asm, second_lane_group)

        second_codebook_loads = 2 * second.physical.decoder.rows
        self._emit_iq2_xxs_decode_prepare_finish(
            asm,
            first_lane_group,
            outstanding_lgkm=second_codebook_loads,
        )
        self._emit_iq2_xxs_decode_values(asm)

        first_decoded_stores = 16 * self.physical.decoder.rows
        second._emit_iq2_xxs_decode_prepare_finish(
            asm,
            second_lane_group,
            outstanding_lgkm=first_decoded_stores,
        )
        second._emit_iq2_xxs_decode_values(asm)

    def _emit_batched_global_codebook_issue(
        self,
        asm: _Assembly,
        second: "GroupedBackwardPairTileComputeEmitter",
    ) -> None:
        self._emit_packed_weight_lane_share(asm)
        second._emit_packed_weight_lane_share(asm)
        self._emit_iq2_s_decode_prepare_addresses(asm)
        second._emit_iq2_s_decode_prepare_addresses(asm)
        load_count = 2 * (self.physical.decoder.rows + second.physical.decoder.rows)
        asm.inst(f"s_clause {load_count - 1}")
        self._emit_iq2_s_global_codebook_reads(asm, emit_clause=False)
        second._emit_iq2_s_global_codebook_reads(asm, emit_clause=False)

    def _emit_batched_global_codebook_finish(
        self,
        asm: _Assembly,
        second: "GroupedBackwardPairTileComputeEmitter",
    ) -> None:
        self._emit_iq2_s_decode_prepare_finish(asm)
        self._emit_iq2_s_decode_chunks(asm)
        second._emit_iq2_s_decode_prepare_finish(asm)
        second._emit_iq2_s_decode_chunks(asm)

    def _emit_projection_decode(self, asm: _Assembly, name: str) -> None:
        asm.comment(f"Decode the {name.lower()} pair projection into disjoint LDS.")
        self._emit_quant_global_reads(asm, wait_for_reads=True)
        self._emit_packed_weight_lane_share(asm)
        self._emit_quant_decode(
            asm,
            label_suffix=self._decode_label_suffix(f"Pair{name}"),
        )

    def _emit_projection(self, asm: _Assembly, name: str) -> None:
        asm.comment(f"Decode and accumulate the {name.lower()} pair projection.")
        self._emit_quant_global_reads(asm, wait_for_reads=True)
        self._emit_packed_weight_lane_share(asm)
        self._emit_quant_decode(
            asm,
            label_suffix=self._decode_label_suffix(f"Pair{name}"),
        )
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        self._emit_wmma(asm)
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")

    def _swap_projection_pointers(self, asm: _Assembly) -> None:
        r = self.registers
        route = self.pair_route
        temporary = route.pointer_temporary
        asm.comment("Swap the active gradient and packed-bank pointer pairs.")
        for first, second in (
            (r.kernarg, route.second_grad_output),
            (r.kernarg + 2, route.second_packed_weight),
        ):
            asm.inst(
                f"s_mov_b64 s[{temporary}:{temporary + 1}], s[{first}:{first + 1}]"
            )
            asm.inst(f"s_mov_b64 s[{first}:{first + 1}], s[{second}:{second + 1}]")
            asm.inst(
                f"s_mov_b64 s[{second}:{second + 1}], s[{temporary}:{temporary + 1}]"
            )

    def _swap_grad_output_pointer(self, asm: _Assembly) -> None:
        asm.comment("Swap only the active gradient pointer pair.")
        self._swap_pointer_pair(
            asm,
            self.registers.kernarg,
            self.pair_route.second_grad_output,
        )

    def _swap_packed_weight_pointer(self, asm: _Assembly) -> None:
        asm.comment("Swap only the active packed-bank pointer pair.")
        self._swap_pointer_pair(
            asm,
            self.registers.kernarg + 2,
            self.pair_route.second_packed_weight,
        )

    def _swap_pointer_pair(self, asm: _Assembly, first: int, second: int) -> None:
        temporary = self.pair_route.pointer_temporary
        asm.inst(f"s_mov_b64 s[{temporary}:{temporary + 1}], s[{first}:{first + 1}]")
        asm.inst(f"s_mov_b64 s[{first}:{first + 1}], s[{second}:{second + 1}]")
        asm.inst(f"s_mov_b64 s[{second}:{second + 1}], s[{temporary}:{temporary + 1}]")


class GroupedBackwardPairKernelLowering:
    """Emit one exact serial-route packed backward pair."""

    EXIT_LABEL = ".LGroupedBackwardPairExit"

    def __init__(
        self,
        kernel_name: str,
        state: DerivedGroupedBackwardPairState,
        physical: GroupedBackwardPairPhysicalPlan,
    ) -> None:
        self.kernel_name = kernel_name
        self.state = state
        self.physical = physical
        self.route_split_factor = state.kernel_spec.route_ownership.split_factor
        self.work_group_mapping = state.kernel_spec.compute.geometry.work_group_mapping
        self.effective_route_split_factor = mapped_route_stride(
            self.route_split_factor,
            self.work_group_mapping,
        )
        policy = state.kernel_spec.projection_policy
        split_full_tiles = policy.split_full_tiles
        concurrent_reads = policy.concurrent_reads
        prefetch_pair_a = policy.activation_prefetch
        direct_second_pointers = policy.direct_second_pointers
        overlap_second_read_prefetch_a = policy.overlaps_second_read
        k_pipeline = policy.pipeline_k
        self.compute = GroupedBackwardPairTileComputeEmitter(
            state.ordinary,
            physical.ordinary,
            route=physical.scalar,
            label_suffix="Tail" if split_full_tiles else "",
            concurrent_reads=concurrent_reads,
            prefetch_pair_a=prefetch_pair_a,
            direct_second_pointers=direct_second_pointers,
            overlap_second_read_prefetch_a=overlap_second_read_prefetch_a,
            k_pipeline=k_pipeline,
        )
        self.second_compute = (
            GroupedBackwardPairTileComputeEmitter(
                state.ordinary,
                physical.second_projection,
                route=physical.scalar,
                label_suffix="TailSecondLds" if split_full_tiles else "SecondLds",
                concurrent_reads=concurrent_reads,
                prefetch_pair_a=prefetch_pair_a,
                direct_second_pointers=direct_second_pointers,
                overlap_second_read_prefetch_a=overlap_second_read_prefetch_a,
                k_pipeline=k_pipeline,
            )
            if physical.second_projection is not None
            else None
        )
        self.compute.second_projection = self.second_compute
        self.full_compute = (
            GroupedBackwardPairTileComputeEmitter(
                state.ordinary,
                physical.ordinary,
                route=physical.scalar,
                label_suffix="Full",
                bounded=False,
                concurrent_reads=concurrent_reads,
                prefetch_pair_a=prefetch_pair_a,
                direct_second_pointers=direct_second_pointers,
                overlap_second_read_prefetch_a=overlap_second_read_prefetch_a,
                k_pipeline=k_pipeline,
            )
            if split_full_tiles
            else None
        )
        self.full_second_compute = (
            GroupedBackwardPairTileComputeEmitter(
                state.ordinary,
                physical.second_projection,
                route=physical.scalar,
                label_suffix="FullSecondLds",
                bounded=False,
                concurrent_reads=concurrent_reads,
                prefetch_pair_a=prefetch_pair_a,
                direct_second_pointers=direct_second_pointers,
                overlap_second_read_prefetch_a=overlap_second_read_prefetch_a,
                k_pipeline=k_pipeline,
            )
            if split_full_tiles and physical.second_projection is not None
            else None
        )
        if self.full_compute is not None:
            self.full_compute.second_projection = self.full_second_compute

    def body(self) -> str:
        asm = _Assembly()
        r = self.compute.registers
        route = self.physical.scalar

        asm.comment("Flatten gfx11 packed workitem X/Y before v0 becomes C storage.")
        asm.inst(f"v_bfe_u32 v{r.serial}, v0, 10, 10")
        asm.inst(f"v_lshlrev_b32 v{r.serial}, 5, v{r.serial}")
        asm.inst(f"v_and_b32 v{r.temporary}, 0x3ff, v0")
        asm.inst(f"v_add_nc_u32 v{r.serial}, v{r.serial}, v{r.temporary}")

        self._emit_kernarg_loads(asm)
        self._emit_exact_shape_guard(asm)
        mapping = self.work_group_mapping
        packed_split = self.route_split_factor > 1 or mapping > 1
        if self.route_split_factor > 1:
            split_shift = self.route_split_factor.bit_length() - 1
            asm.comment("Split grid Y into a route index and strided M-tile task.")
            asm.inst(f"s_and_b32 s{r.input_half}, s3, {self.route_split_factor - 1}")
            asm.inst(f"s_lshr_b32 s3, s3, {split_shift}")
        elif mapping > 1:
            asm.inst(f"s_mov_b32 s{r.input_half}, 0")
        if mapping > 1:
            mapping_shift = work_group_mapping_shift(mapping)
            asm.comment("Decode WGM-packed grid X into N and an M-task lane.")
            asm.inst(f"s_and_b32 s{route.tile_end}, s2, {mapping - 1}")
            asm.inst(f"s_lshr_b32 s2, s2, {mapping_shift}")
            asm.inst(f"s_mul_i32 s{r.input_half}, s{r.input_half}, {mapping}")
            asm.inst(f"s_add_u32 s{r.input_half}, s{r.input_half}, s{route.tile_end}")
        self._emit_route_load_and_guard(asm)
        self._emit_pointer_rebase(asm)
        self.compute.emit_quant_constants(asm)

        asm.comment("Map grid X to N and walk this grid-Y route in M tiles.")
        asm.inst(f"s_mov_b32 s{route.gemm_index}, s3")
        asm.inst("s_mov_b32 s3, s2")
        if packed_split:
            asm.inst(f"s_mov_b32 s2, s{r.input_half}")
        else:
            asm.inst("s_mov_b32 s2, 0")
        n_tiles = (
            self.state.contract.in_features
            // self.compute.state.spec.geometry.macro_tile1
        )
        asm.inst(f"s_cmp_ge_u32 s3, {n_tiles}")
        asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")
        if self.effective_route_split_factor > 1:
            emit_scale_sgpr_u32(
                asm,
                route.tile_end,
                self.compute.state.spec.geometry.macro_tile0,
                2,
            )
            asm.inst(f"s_cmp_ge_u32 s{route.tile_end}, s{route.route_rows}")
            asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")

        self.compute.emit_quant_codebook_stage(asm)
        self.compute.emit_static_coordinates(asm)
        if self.full_compute is None:
            asm.label(".LGroupedBackwardPairRowTile")
            self.compute.emit_tile(asm)
            asm.inst(f"s_add_u32 s2, s2, {self.effective_route_split_factor}")
            emit_scale_sgpr_u32(
                asm,
                route.tile_end,
                self.compute.state.spec.geometry.macro_tile0,
                2,
            )
            asm.inst(f"s_cmp_lt_u32 s{route.tile_end}, s{route.route_rows}")
            asm.inst("s_cbranch_scc1 .LGroupedBackwardPairRowTile")
        else:
            if self.effective_route_split_factor > 1:
                self._emit_split_full_and_tail_route_tiles(asm)
            else:
                self._emit_full_and_tail_route_tiles(asm)

        asm.label(self.EXIT_LABEL)
        emit_kernel_trailer(asm, self.kernel_name)
        return asm.text()

    def emission(self) -> LoweringResult:
        return LoweringResult(
            self.body(),
            self.compute.trailing_sections(),
        )

    def _emit_full_and_tail_route_tiles(self, asm: _Assembly) -> None:
        if self.full_compute is None:
            raise RuntimeError("full-tile split emitter is not configured")
        route = self.physical.scalar
        tile_rows = self.compute.state.spec.geometry.macro_tile0
        full_label = ".LGroupedBackwardPairFullRowTile"
        tail_label = ".LGroupedBackwardPairTailRowTile"
        done_label = ".LGroupedBackwardPairRowTileDone"

        asm.inst(f"s_cmp_lt_u32 s{route.route_rows}, {tile_rows}")
        asm.inst(f"s_cbranch_scc1 {tail_label}")
        asm.label(full_label)
        self.full_compute.emit_tile(asm)
        asm.inst("s_add_u32 s2, s2, 1")
        emit_scale_sgpr_u32(asm, route.tile_end, tile_rows, 2)
        asm.inst(f"s_add_u32 s{route.gemm_index}, s{route.tile_end}, {tile_rows}")
        asm.inst(f"s_cmp_le_u32 s{route.gemm_index}, s{route.route_rows}")
        asm.inst(f"s_cbranch_scc1 {full_label}")
        asm.inst(f"s_cmp_ge_u32 s{route.tile_end}, s{route.route_rows}")
        asm.inst(f"s_cbranch_scc1 {done_label}")
        asm.label(tail_label)
        self.compute.emit_tile(asm)
        asm.label(done_label)

    def _emit_split_full_and_tail_route_tiles(self, asm: _Assembly) -> None:
        if self.full_compute is None:
            raise RuntimeError("split full-tile emitter is not configured")
        route = self.physical.scalar
        tile_rows = self.compute.state.spec.geometry.macro_tile0
        full_label = ".LGroupedBackwardPairSplitFullRowTile"
        tail_label = ".LGroupedBackwardPairSplitTailRowTile"
        done_label = ".LGroupedBackwardPairSplitRowTileDone"

        emit_scale_sgpr_u32(asm, route.tile_end, tile_rows, 2)
        asm.inst(f"s_add_u32 s{route.gemm_index}, s{route.tile_end}, {tile_rows}")
        asm.inst(f"s_cmp_le_u32 s{route.gemm_index}, s{route.route_rows}")
        asm.inst(f"s_cbranch_scc0 {tail_label}")
        asm.label(full_label)
        self.full_compute.emit_tile(asm)
        asm.inst(f"s_add_u32 s2, s2, {self.effective_route_split_factor}")
        emit_scale_sgpr_u32(asm, route.tile_end, tile_rows, 2)
        asm.inst(f"s_cmp_ge_u32 s{route.tile_end}, s{route.route_rows}")
        asm.inst(f"s_cbranch_scc1 {done_label}")
        asm.inst(f"s_add_u32 s{route.gemm_index}, s{route.tile_end}, {tile_rows}")
        asm.inst(f"s_cmp_le_u32 s{route.gemm_index}, s{route.route_rows}")
        asm.inst(f"s_cbranch_scc1 {full_label}")
        asm.label(tail_label)
        self.compute.emit_tile(asm)
        asm.label(done_label)

    def _emit_kernarg_loads(self, asm: _Assembly) -> None:
        r = self.compute.registers
        route = self.physical.scalar
        abi = GROUPED_BACKWARD_PAIR_ABI
        asm.comment(f"Load the routed {abi.segment_size}-byte backward-pair ABI.")
        for register, name in (
            (r.kernarg, "first_grad_output"),
            (route.second_grad_output, "second_grad_output"),
            (r.kernarg + 2, "first_packed_weight"),
            (route.second_packed_weight, "second_packed_weight"),
            (r.kernarg + 4, "grad_input"),
            (route.expert_indices, "expert_indices"),
            (route.expert_offsets, "expert_offsets"),
        ):
            asm.inst(
                f"s_load_dwordx2 s[{register}:{register + 1}], s[0:1], "
                f"0x{abi.offset(name):x}"
            )
        asm.inst(
            f"s_load_dword s{route.num_experts}, s[0:1], "
            f"0x{abi.offset('num_experts'):x}"
        )
        asm.inst(f"s_load_dword s{route.rows}, s[0:1], 0x{abi.offset('rows'):x}")
        asm.inst(
            f"s_load_dwordx2 s[{route.bytes_per_expert}:"
            f"{route.bytes_per_expert + 1}], s[0:1], "
            f"0x{abi.offset('bytes_per_expert'):x}"
        )
        asm.inst("s_waitcnt lgkmcnt(0)")

    def _emit_exact_shape_guard(self, asm: _Assembly) -> None:
        route = self.physical.scalar
        asm.comment("Reject arguments outside this exact paired grouped key.")
        for register, expected in (
            (route.num_experts, self.state.contract.physical_experts),
            (route.rows, self.state.ordinary.contract.problem_size.m),
            (route.bytes_per_expert, self.state.bytes_per_expert),
            (route.bytes_per_expert + 1, 0),
        ):
            asm.inst(f"s_cmp_lg_u32 s{register}, {expected}")
            asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")

    def _emit_route_load_and_guard(self, asm: _Assembly) -> None:
        route = self.physical.scalar
        gemm = route.gemm_index
        offset = route.tile_end
        asm.inst(f"s_mov_b32 s{gemm}, s3")
        asm.inst(f"s_cmp_ge_u32 s{gemm}, {self.state.contract.max_route_entries}")
        asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")
        asm.comment("Load cumulative row bounds and the shared physical expert.")
        asm.inst(f"s_lshl_b32 s{offset}, s{gemm}, 2")
        asm.inst(
            f"s_load_dword s{route.row_end}, s[{route.expert_offsets}:"
            f"{route.expert_offsets + 1}], s{offset}"
        )
        asm.inst(f"s_lshl_b32 s{offset}, s{gemm}, 3")
        asm.inst(
            f"s_load_dwordx2 s[{route.expert}:{route.expert + 1}], "
            f"s[{route.expert_indices}:{route.expert_indices + 1}], s{offset}"
        )
        asm.inst(f"s_cmp_eq_u32 s{gemm}, 0")
        asm.inst("s_cbranch_scc1 .LGroupedBackwardPairFirstRoute")
        asm.inst(f"s_lshl_b32 s{offset}, s{gemm}, 2")
        asm.inst(f"s_sub_u32 s{offset}, s{offset}, 4")
        asm.inst(
            f"s_load_dword s{route.row_begin}, s[{route.expert_offsets}:"
            f"{route.expert_offsets + 1}], s{offset}"
        )
        asm.inst("s_branch .LGroupedBackwardPairRouteLoaded")
        asm.label(".LGroupedBackwardPairFirstRoute")
        asm.inst(f"s_mov_b32 s{route.row_begin}, 0")
        asm.label(".LGroupedBackwardPairRouteLoaded")
        asm.inst("s_waitcnt lgkmcnt(0)")
        for instruction in (
            f"s_cmp_lg_u32 s{route.expert + 1}, 0",
            f"s_cmp_ge_u32 s{route.expert}, s{route.num_experts}",
            f"s_cmp_lt_i32 s{route.row_begin}, 0",
            f"s_cmp_le_i32 s{route.row_end}, s{route.row_begin}",
            f"s_cmp_gt_u32 s{route.row_end}, s{route.rows}",
        ):
            asm.inst(instruction)
            asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")
        asm.inst(f"s_sub_u32 s{route.route_rows}, s{route.row_end}, s{route.row_begin}")
        row_stride = self.state.contract.out_features * 2
        asm.inst(
            f"s_lshl_b32 s{route.route_output_bytes}, s{route.route_rows}, "
            f"{row_stride.bit_length() - 1}"
        )

    def _emit_pointer_rebase(self, asm: _Assembly) -> None:
        r = self.compute.registers
        route = self.physical.scalar
        temporary = route.pointer_temporary

        asm.comment("Rebase both packed banks with one full u64 expert stride.")
        asm.inst(f"s_mul_i32 s{temporary}, s{route.expert}, s{route.bytes_per_expert}")
        asm.inst(
            f"s_mul_hi_u32 s{temporary + 1}, s{route.expert}, s{route.bytes_per_expert}"
        )
        asm.inst(
            f"s_mul_i32 s{route.tile_end}, s{route.expert}, "
            f"s{route.bytes_per_expert + 1}"
        )
        asm.inst(f"s_add_u32 s{temporary + 1}, s{temporary + 1}, s{route.tile_end}")
        for pointer in (r.kernarg + 2, route.second_packed_weight):
            asm.inst(f"s_add_u32 s{pointer}, s{pointer}, s{temporary}")
            asm.inst(f"s_addc_u32 s{pointer + 1}, s{pointer + 1}, s{temporary + 1}")

        asm.comment("Rebase both gradients and the shared destination by route start.")
        for stride, pointers in (
            (
                self.state.contract.out_features * 2,
                (r.kernarg, route.second_grad_output),
            ),
            (self.state.contract.in_features * 2, (r.kernarg + 4,)),
        ):
            asm.inst(f"s_mul_i32 s{temporary}, s{route.row_begin}, {stride}")
            asm.inst(f"s_mul_hi_u32 s{temporary + 1}, s{route.row_begin}, {stride}")
            for pointer in pointers:
                asm.inst(f"s_add_u32 s{pointer}, s{pointer}, s{temporary}")
                asm.inst(f"s_addc_u32 s{pointer + 1}, s{pointer + 1}, s{temporary + 1}")
