"""Routed ownership around the reusable ordinary MMQ backward arithmetic."""

from dataclasses import dataclass

from .grouped_mmq_bwd_physical import (
    GroupedBackwardPhysicalPlan,
    GroupedBackwardScalarPlan,
)
from .grouped_mmq_bwd_spec import (
    DerivedGroupedBackwardState,
)
from .kernel_abi import GROUPED_BACKWARD_ABI
from .kernel_writer_assembly import emit_kernel_trailer, emit_pointer_kernarg_loads
from .mmq_bwd_emission import BackwardLoweringResult, _Assembly
from .mmq_bwd_lowering import BackwardTileComputeEmitter
from .mmq_bwd_lowering_quant import emit_unbounded_a_global_loads
from .mmq_bwd_physical import BackwardRegisterPlan


@dataclass(frozen=True)
class GroupedBackwardTileAccess:
    route: GroupedBackwardScalarPlan

    def emit_a_global_loads(
        self,
        asm: _Assembly,
        registers: BackwardRegisterPlan,
        valu_a: int,
        address: int,
        byte_offset: int,
        *,
        address_pair: bool,
        offset_is_bytes: bool,
    ) -> None:
        limit = (
            self.route.route_output_bytes if offset_is_bytes else self.route.route_rows
        )
        for register in range(valu_a, valu_a + 8):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"v_cmp_gt_u32_e32 vcc_lo, s{limit}, v{byte_offset}")
        asm.inst(f"s_and_saveexec_b32 s{self.route.saved_exec}, vcc_lo")
        emit_unbounded_a_global_loads(
            asm,
            registers,
            valu_a,
            address,
            address_pair=address_pair,
        )
        asm.inst(f"s_mov_b32 exec_lo, s{self.route.saved_exec}")

    def emit_store_row_begin(
        self, asm: _Assembly, registers: BackwardRegisterPlan, row: int
    ) -> None:
        asm.inst(f"v_mov_b32 v{registers.temporary + 3}, v{row}")

    def emit_store_row_mask_begin(
        self, asm: _Assembly, registers: BackwardRegisterPlan
    ) -> None:
        tracker = registers.temporary + 3
        asm.inst(f"v_cmp_gt_u32_e32 vcc_lo, s{self.route.route_rows}, v{tracker}")
        asm.inst(f"s_and_saveexec_b32 s{self.route.saved_exec}, vcc_lo")

    def emit_store_row_mask_end(
        self, asm: _Assembly, registers: BackwardRegisterPlan
    ) -> None:
        del registers
        asm.inst(f"s_mov_b32 exec_lo, s{self.route.saved_exec}")

    def emit_store_row_advance(
        self, asm: _Assembly, registers: BackwardRegisterPlan
    ) -> None:
        tracker = registers.temporary + 3
        asm.inst(f"v_add_nc_u32 v{tracker}, 2, v{tracker}")


class GroupedBackwardKernelLowering:
    """Emit routed packed backward with typed aggregate-row tails."""

    EXIT_LABEL = ".LGroupedBackwardExit"

    def __init__(
        self,
        kernel_name: str,
        state: DerivedGroupedBackwardState,
        physical: GroupedBackwardPhysicalPlan,
    ) -> None:
        self.kernel_name = kernel_name
        self.state = state
        self.physical = physical
        access = GroupedBackwardTileAccess(physical.route)
        self.primary = BackwardTileComputeEmitter(
            state.primary,
            physical.primary,
            access=access,
        )
        self.secondary = (
            BackwardTileComputeEmitter(
                state.secondary,
                physical.secondary,
                access=access,
                label_suffix="Tail",
            )
            if state.secondary is not None and physical.secondary is not None
            else None
        )

    def body(self) -> str:
        asm = _Assembly()
        r = self.primary.registers
        route = self.physical.route

        asm.comment("Flatten gfx11 packed workitem X/Y before v0 becomes C storage.")
        asm.inst(f"v_bfe_u32 v{r.serial}, v0, 10, 10")
        asm.inst(f"v_lshlrev_b32 v{r.serial}, 5, v{r.serial}")
        asm.inst(f"v_and_b32 v{r.temporary}, 0x3ff, v0")
        asm.inst(f"v_add_nc_u32 v{r.serial}, v{r.serial}, v{r.temporary}")

        self._emit_kernarg_loads(asm)
        self._emit_exact_shape_guard(asm)
        self._emit_route_load_and_guard(asm)
        self._emit_pointer_rebase(asm)
        self.primary.emit_quant_constants(asm)

        split_factor = self.state.spec.ownership.split_factor
        asm.comment("Map grid X to N and walk this grid-Y route in M tiles.")
        asm.inst(f"s_mov_b32 s{route.gemm_index}, s3")
        asm.inst("s_mov_b32 s3, s2")
        asm.inst("s_mov_b32 s2, s4" if split_factor > 1 else "s_mov_b32 s2, 0")
        n_tiles = (
            self.state.contract.problem_size.n
            // self.state.spec.compute.geometry.macro_tile1
        )
        asm.inst(f"s_cmp_ge_u32 s3, {n_tiles}")
        asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")
        if split_factor > 1:
            asm.inst(
                f"s_mul_i32 s{route.tile_end}, s2, "
                f"{self.state.spec.compute.geometry.macro_tile0}"
            )
            asm.inst(f"s_cmp_ge_u32 s{route.tile_end}, s{route.route_rows}")
            asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")

        self.primary.emit_quant_codebook_stage(asm)

        if self.secondary is None:
            self.primary.emit_static_coordinates(asm)
            self._emit_main_route_tiles(asm, split_factor)
        else:
            tail_label = f".LGroupedBackwardTail{self.state.spec.row_tail.tile_rows}"
            done_label = ".LGroupedBackwardTailDone"
            asm.inst(
                f"s_cmp_le_u32 s{route.route_rows}, "
                f"{self.state.spec.row_tail.threshold_rows}"
            )
            asm.inst(f"s_cbranch_scc1 {tail_label}")
            self.primary.emit_static_coordinates(asm)
            self._emit_main_route_tiles(asm, split_factor)
            asm.inst(f"s_branch {done_label}")
            asm.label(tail_label)
            asm.inst(f"v_mov_b32 v{self.secondary.registers.serial}, v{r.serial}")
            self.secondary.emit_static_coordinates(asm)
            self.secondary.emit_tile(asm)
            asm.label(done_label)

        asm.label(self.EXIT_LABEL)
        emit_kernel_trailer(asm, self.kernel_name)
        return asm.text()

    def emission(self) -> BackwardLoweringResult:
        return BackwardLoweringResult(self.body(), self.primary.trailing_sections())

    def _emit_main_route_tiles(self, asm: _Assembly, split_factor: int) -> None:
        route = self.physical.route
        asm.label(".LGroupedBackwardRowTile")
        self.primary.emit_tile(asm)
        asm.inst(f"s_add_u32 s2, s2, {split_factor}")
        asm.inst(
            f"s_mul_i32 s{route.tile_end}, s2, "
            f"{self.state.spec.compute.geometry.macro_tile0}"
        )
        asm.inst(f"s_cmp_lt_u32 s{route.tile_end}, s{route.route_rows}")
        asm.inst("s_cbranch_scc1 .LGroupedBackwardRowTile")

    def _emit_kernarg_loads(self, asm: _Assembly) -> None:
        r = self.primary.registers
        route = self.physical.route
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
        route = self.physical.route
        contract = self.state.contract
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
        route = self.physical.route
        gemm = route.gemm_index
        offset = route.tile_end
        asm.inst(f"s_mov_b32 s{gemm}, s3")
        asm.inst(f"s_cmp_ge_u32 s{gemm}, {self.state.contract.max_route_entries}")
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
        row_stride_bytes = self.state.contract.problem_size.k * 2
        if row_stride_bytes & (row_stride_bytes - 1):
            asm.inst(
                f"s_mul_i32 s{route.route_output_bytes}, "
                f"s{route.route_rows}, {row_stride_bytes}"
            )
        else:
            asm.inst(
                f"s_lshl_b32 s{route.route_output_bytes}, "
                f"s{route.route_rows}, {row_stride_bytes.bit_length() - 1}"
            )

    def _emit_pointer_rebase(self, asm: _Assembly) -> None:
        r = self.primary.registers
        route = self.physical.route
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
            (r.kernarg, self.state.contract.problem_size.k * 2),
            (r.kernarg + 4, self.state.contract.problem_size.n * 2),
        ):
            asm.inst(f"s_mul_i32 s{temporary}, s{route.row_begin}, {stride}")
            asm.inst(f"s_mul_hi_u32 s{temporary + 1}, s{route.row_begin}, {stride}")
            asm.inst(f"s_add_u32 s{pointer}, s{pointer}, s{temporary}")
            asm.inst(f"s_addc_u32 s{pointer + 1}, s{pointer + 1}, s{temporary + 1}")
