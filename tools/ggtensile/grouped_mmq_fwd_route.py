"""Common routed ABI emission for grouped MMQ forward kernels."""

from dataclasses import dataclass
from typing import ClassVar, Protocol

from .grouped_mmq_fwd_spec import GroupedRouteState
from .kernel_writer_assembly import Assembly, RegisterAssignment


class GroupedRouteScalarRegisters(Protocol):
    @property
    def kernarg(self) -> RegisterAssignment: ...

    @property
    def gemm_index(self) -> RegisterAssignment: ...

    @property
    def weights(self) -> RegisterAssignment: ...

    @property
    def activations(self) -> RegisterAssignment: ...

    @property
    def output(self) -> RegisterAssignment: ...

    @property
    def expert_indices(self) -> RegisterAssignment: ...

    @property
    def expert_offsets(self) -> RegisterAssignment: ...

    @property
    def num_experts(self) -> RegisterAssignment: ...

    @property
    def nrows_weight(self) -> RegisterAssignment: ...

    @property
    def nrows_activation(self) -> RegisterAssignment: ...

    @property
    def blocks_per_weight_row(self) -> RegisterAssignment: ...

    @property
    def bytes_per_expert(self) -> RegisterAssignment: ...

    @property
    def row_begin(self) -> RegisterAssignment: ...

    @property
    def row_end(self) -> RegisterAssignment: ...

    @property
    def expert(self) -> RegisterAssignment: ...

    @property
    def route_offset(self) -> RegisterAssignment: ...

    @property
    def pointer_offset(self) -> RegisterAssignment: ...


@dataclass(frozen=True)
class GroupedRouteEmitter:
    registers: GroupedRouteScalarRegisters
    state: GroupedRouteState

    EXIT_LABEL: ClassVar[str] = ".LGroupedQ4KExit"

    def emit(self, asm: Assembly) -> None:
        self._emit_kernarg_loads(asm)
        self._emit_exact_shape_guard(asm)
        self._emit_route_load_and_guard(asm)
        self._emit_expert_pointer_rebase(asm)

    def _emit_kernarg_loads(self, asm: Assembly) -> None:
        scalar = self.registers
        asm.comment("Load the routed 64-byte grouped forward ABI.")
        for pointer, offset in (
            (scalar.weights, 0x00),
            (scalar.activations, 0x08),
            (scalar.output, 0x10),
            (scalar.expert_indices, 0x18),
            (scalar.expert_offsets, 0x20),
        ):
            asm.inst(
                f"s_load_dwordx2 s[{pointer.first_register}:{pointer.first_register + 1}], "
                f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
                f"0x{offset:x}"
            )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.num_experts.first_register}:"
            f"{scalar.num_experts.first_register + 1}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
            "0x28"
        )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.nrows_activation.first_register}:"
            f"{scalar.blocks_per_weight_row.first_register}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
            "0x30"
        )
        asm.inst(
            f"s_load_dwordx2 s[{scalar.bytes_per_expert.first_register}:"
            f"{scalar.bytes_per_expert.first_register + 1}], "
            f"s[{scalar.kernarg.first_register}:{scalar.kernarg.first_register + 1}], "
            "0x38"
        )
        asm.inst("s_waitcnt lgkmcnt(0)")

    def _emit_exact_shape_guard(self, asm: Assembly) -> None:
        scalar = self.registers
        asm.comment("Reject launch arguments outside this exact grouped key.")
        checks = (
            (scalar.num_experts.first_register, self.state.physical_experts),
            (scalar.nrows_weight.first_register, self.state.output_features),
            (scalar.nrows_activation.first_register, self.state.aggregate_rows),
            (
                scalar.blocks_per_weight_row.first_register,
                self.state.blocks_per_weight_row,
            ),
            (scalar.bytes_per_expert.first_register, self.state.bytes_per_expert),
            (scalar.bytes_per_expert.first_register + 1, 0),
        )
        for register, expected in checks:
            asm.inst(f"s_cmp_lg_u32 s{register}, {expected}")
            asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")

    def _emit_route_load_and_guard(self, asm: Assembly) -> None:
        scalar = self.registers
        route_offset = scalar.route_offset.first_register
        gemm_index = scalar.gemm_index.first_register
        asm.comment("Load cumulative row bounds and the selected physical expert.")
        asm.inst(f"s_lshl_b32 s{route_offset}, s{gemm_index}, 2")
        asm.inst(
            f"s_load_dword s{scalar.row_end.first_register}, "
            f"s[{scalar.expert_offsets.first_register}:"
            f"{scalar.expert_offsets.first_register + 1}], s{route_offset}"
        )
        asm.inst(f"s_lshl_b32 s{route_offset}, s{gemm_index}, 3")
        asm.inst(
            f"s_load_dwordx2 s[{scalar.expert.first_register}:"
            f"{scalar.expert.first_register + 1}], "
            f"s[{scalar.expert_indices.first_register}:"
            f"{scalar.expert_indices.first_register + 1}], s{route_offset}"
        )
        asm.inst(f"s_cmp_eq_u32 s{gemm_index}, 0")
        asm.inst("s_cbranch_scc1 .LGroupedQ4KFirstRoute")
        asm.inst(f"s_lshl_b32 s{route_offset}, s{gemm_index}, 2")
        asm.inst(f"s_sub_u32 s{route_offset}, s{route_offset}, 4")
        asm.inst(
            f"s_load_dword s{scalar.row_begin.first_register}, "
            f"s[{scalar.expert_offsets.first_register}:"
            f"{scalar.expert_offsets.first_register + 1}], s{route_offset}"
        )
        asm.inst("s_branch .LGroupedQ4KRouteLoaded")
        asm.label(".LGroupedQ4KFirstRoute")
        asm.inst(f"s_mov_b32 s{scalar.row_begin.first_register}, 0")
        asm.label(".LGroupedQ4KRouteLoaded")
        asm.inst("s_waitcnt lgkmcnt(0)")

        asm.comment("Invalid experts and cumulative ranges are inert.")
        guard_instructions = (
            f"s_cmp_lg_u32 s{scalar.expert.first_register + 1}, 0",
            f"s_cmp_ge_u32 s{scalar.expert.first_register}, s{scalar.num_experts.first_register}",
            f"s_cmp_lt_i32 s{scalar.row_begin.first_register}, 0",
            f"s_cmp_le_i32 s{scalar.row_end.first_register}, s{scalar.row_begin.first_register}",
            f"s_cmp_gt_u32 s{scalar.row_end.first_register}, s{scalar.nrows_activation.first_register}",
        )
        for instruction in guard_instructions:
            asm.inst(instruction)
            asm.inst(f"s_cbranch_scc1 {self.EXIT_LABEL}")

    def _emit_expert_pointer_rebase(self, asm: Assembly) -> None:
        scalar = self.registers
        route_offset = scalar.route_offset.first_register
        pointer_offset = scalar.pointer_offset.first_register
        expert = scalar.expert.first_register
        stride = scalar.bytes_per_expert.first_register
        weights = scalar.weights.first_register
        asm.comment("Rebase the shared packed bank with the full u64 expert stride.")
        asm.inst(f"s_mul_i32 s{route_offset}, s{expert}, s{stride}")
        asm.inst(f"s_mul_hi_u32 s{pointer_offset}, s{expert}, s{stride}")
        asm.inst(f"s_mul_i32 s{pointer_offset + 1}, s{expert}, s{stride + 1}")
        asm.inst(
            f"s_add_u32 s{pointer_offset}, s{pointer_offset}, s{pointer_offset + 1}"
        )
        asm.inst(f"s_add_u32 s{weights}, s{weights}, s{route_offset}")
        asm.inst(f"s_addc_u32 s{weights + 1}, s{weights + 1}, s{pointer_offset}")
