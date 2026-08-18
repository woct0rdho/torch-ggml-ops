"""Routed ownership around the reusable ordinary MMQ backward arithmetic."""

from dataclasses import replace

from .grouped_mmq_bwd_physical import derive_grouped_backward_physical_plan
from .grouped_mmq_bwd_spec import (
    DerivedGroupedBackwardState,
    GroupedBackwardRowTail,
)
from .kernel_abi import GROUPED_BACKWARD_ABI
from .kernel_writer_assembly import emit_kernel_trailer, emit_pointer_kernarg_loads
from .mmq_bwd_emission import BackwardDiagnosticMode, _Assembly
from .mmq_bwd_lowering import BackwardKernelLowering
from .model import SolutionKey


class GroupedBackwardKernelLowering(BackwardKernelLowering):
    """Emit routed packed backward with typed aggregate-row tails."""

    EXIT_LABEL = ".LGroupedBackwardExit"

    def __init__(self, solution_key: SolutionKey, *, label_suffix: str = "") -> None:
        self.grouped_state = DerivedGroupedBackwardState.from_solution_key(solution_key)
        self.grouped_physical = derive_grouped_backward_physical_plan(
            self.grouped_state
        )
        self.state = self.grouped_state.ordinary
        self.physical = self.grouped_physical.compute
        self.registers = self.physical.registers
        self.diagnostic_mode: BackwardDiagnosticMode | None = None
        self.label_suffix = label_suffix
        self.tail_lowering: GroupedBackwardKernelLowering | None = None
        if self.grouped_state.spec.row_tail is GroupedBackwardRowTail.Mixed128_64:
            solution = self.grouped_state.solution
            compute = solution.compute
            matrix_instruction = (
                compute.matrix_instruction[:5] + (1,) + compute.matrix_instruction[6:]
            )
            tail_compute = replace(
                compute,
                matrix_instruction=matrix_instruction,
                macro_tile0=64,
            )
            tail_solution = replace(
                solution,
                compute=tail_compute,
                row_tail="Masked",
            )
            tail_key = replace(solution_key, solution=tail_solution)
            self.tail_lowering = GroupedBackwardKernelLowering(
                tail_key,
                label_suffix=f"{label_suffix}Tail",
            )

    def body(self) -> str:
        asm = _Assembly()
        r = self.registers
        route = self.grouped_physical.route
        name = self.grouped_state.solution_key.kernel_name

        asm.comment("Flatten gfx11 packed workitem X/Y before v0 becomes C storage.")
        asm.inst(f"v_bfe_u32 v{r.serial}, v0, 10, 10")
        asm.inst(f"v_lshlrev_b32 v{r.serial}, 5, v{r.serial}")
        asm.inst(f"v_and_b32 v{r.temporary}, 0x3ff, v0")
        asm.inst(f"v_add_nc_u32 v{r.serial}, v{r.serial}, v{r.temporary}")

        self._emit_kernarg_loads(asm)
        self._emit_exact_shape_guard(asm)
        self._emit_route_load_and_guard(asm)
        self._emit_pointer_rebase(asm)

        split_factor = self.grouped_state.spec.ownership.split_factor
        asm.comment("Map grid X to N and walk this grid-Y route in M tiles.")
        asm.inst(f"s_mov_b32 s{route.gemm_index}, s3")
        asm.inst("s_mov_b32 s3, s2")
        asm.inst("s_mov_b32 s2, s4" if split_factor > 1 else "s_mov_b32 s2, 0")
        n_tiles = (
            self.grouped_state.contract.problem_size.n
            // self.state.solution.macro_tile1
        )
        asm.inst(f"s_cmp_ge_u32 s3, {n_tiles}")
        asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")
        if split_factor > 1:
            asm.inst(
                f"s_mul_i32 s{route.tile_end}, s2, {self.state.solution.macro_tile0}"
            )
            asm.inst(f"s_cmp_ge_u32 s{route.tile_end}, s{route.route_rows}")
            asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")

        if self.tail_lowering is None:
            self._emit_static_packed_coordinates(asm)
            self._emit_main_route_tiles(asm, split_factor)
        else:
            tail_label = ".LGroupedBackwardTail64"
            done_label = ".LGroupedBackwardTailDone"
            asm.inst(f"s_cmp_le_u32 s{route.route_rows}, 64")
            asm.inst(f"s_cbranch_scc1 {tail_label}")
            self._emit_static_packed_coordinates(asm)
            self._emit_main_route_tiles(asm, split_factor)
            asm.inst(f"s_branch {done_label}")
            asm.label(tail_label)
            tail = self.tail_lowering
            asm.inst(f"v_mov_b32 v{tail.registers.serial}, v{self.registers.serial}")
            tail._emit_static_packed_coordinates(asm)
            tail._defer_accumulator_zero(asm)
            tail._emit_compute_tile(asm)
            asm.label(done_label)

        asm.label(self.EXIT_LABEL)
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_main_route_tiles(self, asm: _Assembly, split_factor: int) -> None:
        route = self.grouped_physical.route
        asm.label(".LGroupedBackwardRowTile")
        self._defer_accumulator_zero(asm)
        self._emit_compute_tile(asm)
        asm.inst(f"s_add_u32 s2, s2, {split_factor}")
        asm.inst(f"s_mul_i32 s{route.tile_end}, s2, {self.state.solution.macro_tile0}")
        asm.inst(f"s_cmp_lt_u32 s{route.tile_end}, s{route.route_rows}")
        asm.inst("s_cbranch_scc1 .LGroupedBackwardRowTile")

    def _emit_kernarg_loads(self, asm: _Assembly) -> None:
        r = self.registers
        route = self.grouped_physical.route
        abi = GROUPED_BACKWARD_ABI
        asm.comment(f"Load the routed {abi.segment_size}-byte grouped backward ABI.")
        emit_pointer_kernarg_loads(asm, r.kernarg, abi)
        for first, name in (
            (route.expert_indices, "expert_indices"),
            (route.expert_offsets, "expert_offsets"),
        ):
            asm.inst(
                f"s_load_dwordx2 s[{first}:{first + 1}], s[0:1], 0x{abi.offset(name):x}"
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
        route = self.grouped_physical.route
        contract = self.grouped_state.contract
        bytes_per_expert = (
            contract.problem_size.k
            * (contract.problem_size.n // contract.quant_format.block_values)
            * contract.quant_format.block_bytes
        )
        asm.comment("Reject launch arguments outside this exact grouped key.")
        for register, expected in (
            (route.num_experts, contract.physical_experts),
            (route.rows, contract.problem_size.m),
            (route.bytes_per_expert, bytes_per_expert),
            (route.bytes_per_expert + 1, 0),
        ):
            asm.inst(f"s_cmp_lg_u32 s{register}, {expected}")
            asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")

    def _emit_route_load_and_guard(self, asm: _Assembly) -> None:
        route = self.grouped_physical.route
        gemm = route.gemm_index
        offset = route.tile_end
        asm.inst(f"s_mov_b32 s{gemm}, s3")
        asm.inst(
            f"s_cmp_ge_u32 s{gemm}, {self.grouped_state.contract.max_route_entries}"
        )
        asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")
        asm.comment("Load cumulative row bounds and the selected physical expert.")
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
        asm.inst("s_cbranch_scc1 .LGroupedBackwardFirstRoute")
        asm.inst(f"s_lshl_b32 s{offset}, s{gemm}, 2")
        asm.inst(f"s_sub_u32 s{offset}, s{offset}, 4")
        asm.inst(
            f"s_load_dword s{route.row_begin}, s[{route.expert_offsets}:"
            f"{route.expert_offsets + 1}], s{offset}"
        )
        asm.inst("s_branch .LGroupedBackwardRouteLoaded")
        asm.label(".LGroupedBackwardFirstRoute")
        asm.inst(f"s_mov_b32 s{route.row_begin}, 0")
        asm.label(".LGroupedBackwardRouteLoaded")
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
        asm.inst(f"s_lshl_b32 s{route.route_output_bytes}, s{route.route_rows}, 12")

    def _emit_pointer_rebase(self, asm: _Assembly) -> None:
        r = self.registers
        route = self.grouped_physical.route
        temporary = route.pointer_temporary

        asm.comment("Rebase the packed bank with the full u64 expert stride.")
        asm.inst(f"s_mul_i32 s{temporary}, s{route.expert}, s{route.bytes_per_expert}")
        asm.inst(
            f"s_mul_hi_u32 s{temporary + 1}, s{route.expert}, s{route.bytes_per_expert}"
        )
        asm.inst(
            f"s_mul_i32 s{route.tile_end}, s{route.expert}, "
            f"s{route.bytes_per_expert + 1}"
        )
        asm.inst(f"s_add_u32 s{temporary + 1}, s{temporary + 1}, s{route.tile_end}")
        asm.inst(f"s_add_u32 s{r.kernarg + 2}, s{r.kernarg + 2}, s{temporary}")
        asm.inst(f"s_addc_u32 s{r.kernarg + 3}, s{r.kernarg + 3}, s{temporary + 1}")

        for pointer, stride in (
            (r.kernarg, self.grouped_state.contract.problem_size.k * 2),
            (r.kernarg + 4, self.grouped_state.contract.problem_size.n * 2),
        ):
            asm.inst(f"s_mul_i32 s{temporary}, s{route.row_begin}, {stride}")
            asm.inst(f"s_mul_hi_u32 s{temporary + 1}, s{route.row_begin}, {stride}")
            asm.inst(f"s_add_u32 s{pointer}, s{pointer}, s{temporary}")
            asm.inst(f"s_addc_u32 s{pointer + 1}, s{pointer + 1}, s{temporary + 1}")

    def _emit_a_global_loads(
        self,
        asm: _Assembly,
        valu_a: int,
        address: int,
        byte_offset: int,
        *,
        address_pair: bool = False,
        offset_is_bytes: bool = True,
    ) -> None:
        route = self.grouped_physical.route
        limit = route.route_output_bytes if offset_is_bytes else route.route_rows
        for register in range(valu_a, valu_a + 8):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"v_cmp_gt_u32_e32 vcc_lo, s{limit}, v{byte_offset}")
        asm.inst(f"s_and_saveexec_b32 s{route.saved_exec}, vcc_lo")
        super()._emit_a_global_loads(
            asm,
            valu_a,
            address,
            byte_offset,
            address_pair=address_pair,
            offset_is_bytes=offset_is_bytes,
        )
        asm.inst(f"s_mov_b32 exec_lo, s{route.saved_exec}")

    def _emit_store_row_begin(self, asm: _Assembly, row: int) -> None:
        tracker = self.registers.temporary + 3
        asm.inst(f"v_mov_b32 v{tracker}, v{row}")

    def _emit_store_row_mask_begin(self, asm: _Assembly) -> None:
        route = self.grouped_physical.route
        tracker = self.registers.temporary + 3
        asm.inst(f"v_cmp_gt_u32_e32 vcc_lo, s{route.route_rows}, v{tracker}")
        asm.inst(f"s_and_saveexec_b32 s{route.saved_exec}, vcc_lo")

    def _emit_store_row_mask_end(self, asm: _Assembly) -> None:
        asm.inst(f"s_mov_b32 exec_lo, s{self.grouped_physical.route.saved_exec}")

    def _emit_store_row_advance(self, asm: _Assembly) -> None:
        tracker = self.registers.temporary + 3
        asm.inst(f"v_add_nc_u32 v{tracker}, 2, v{tracker}")
