"""Dedicated six-argument lowering for fixed-group Q8_0 backward."""

from dataclasses import dataclass

from .fixed_grouped_mmq_bwd_spec import DerivedFixedBackwardState
from .kernel_abi import FIXED_GROUPED_BACKWARD_ABI
from .kernel_writer_assembly import (
    emit_kernel_trailer,
    emit_pointer_kernarg_loads,
)
from .mmq_bwd_emission import BackwardLoweringResult, _Assembly
from .mmq_bwd_lowering import BackwardTileComputeEmitter
from .mmq_bwd_lowering_quant import UnboundedBackwardTileAccess


@dataclass(frozen=True)
class FixedBackwardTileAccess(UnboundedBackwardTileAccess):
    groups: int

    def activation_row_stride_bytes(self, contiguous_stride: int) -> int:
        return self.groups * contiguous_stride

    def output_row_stride_bytes(self, contiguous_stride: int) -> int:
        return self.groups * contiguous_stride


@dataclass(frozen=True)
class FixedGroupedQ8BackwardLowering:
    kernel_name: str
    state: DerivedFixedBackwardState

    def emission(self) -> BackwardLoweringResult:
        emitter = BackwardTileComputeEmitter(
            self.state.ordinary,
            self.state.physical.ordinary,
            access=FixedBackwardTileAccess(self.state.problem.groups),
            clause_batch_store=self.state.spec.store_schedule == "ClausePairs",
        )
        asm = _Assembly()
        registers = self.state.physical.ordinary.registers
        offset = self.state.physical.scalar.group_byte_offset

        asm.comment("Flatten gfx11 packed workitem X/Y into the wave32 serial id.")
        asm.inst(f"v_bfe_u32 v{registers.serial}, v0, 10, 10")
        asm.inst(f"v_lshlrev_b32 v{registers.serial}, 5, v{registers.serial}")
        asm.inst(f"v_and_b32 v{registers.temporary}, 0x3ff, v0")
        asm.inst(
            f"v_add_nc_u32 v{registers.serial}, "
            f"v{registers.serial}, v{registers.temporary}"
        )
        if self.state.spec.work_group_order == "NMajor":
            asm.comment("Map N-major grid X/Y onto the compute emitter's M/N SGPRs.")
            asm.inst(f"s_mov_b32 s{offset}, s2")
            asm.inst("s_mov_b32 s2, s3")
            asm.inst(f"s_mov_b32 s3, s{offset}")

        asm.comment("Load fixed-group pointers and the packed-bank byte stride.")
        emit_pointer_kernarg_loads(asm, registers.kernarg, FIXED_GROUPED_BACKWARD_ABI)
        asm.inst(
            f"s_load_dword s{offset}, s[0:1], "
            f"0x{FIXED_GROUPED_BACKWARD_ABI.offset('bytes_per_group'):x}"
        )
        asm.inst("s_waitcnt lgkmcnt(0)")

        asm.comment("Offset all bases to the fixed group selected by workgroup Z.")
        asm.inst(f"s_mul_i32 s{offset}, s4, s{offset}")
        self._emit_pointer_offset(asm, registers.kernarg + 2, offset)
        grad_output_shift = (2 * self.state.problem.output_features).bit_length() - 1
        asm.inst(f"s_lshl_b32 s{offset}, s4, {grad_output_shift}")
        self._emit_pointer_offset(asm, registers.kernarg, offset)
        grad_input_shift = (2 * self.state.problem.input_features).bit_length() - 1
        asm.inst(f"s_lshl_b32 s{offset}, s4, {grad_input_shift}")
        self._emit_pointer_offset(asm, registers.kernarg + 4, offset)

        emitter.emit_quant_constants(asm)
        emitter.emit_quant_codebook_stage(asm)
        emitter.emit_static_coordinates(asm)
        emitter.emit_tile(asm)
        emit_kernel_trailer(asm, self.kernel_name)
        return BackwardLoweringResult(asm.text(), emitter.trailing_sections())

    @staticmethod
    def _emit_pointer_offset(asm: _Assembly, pointer: int, offset: int) -> None:
        asm.inst(f"s_add_u32 s{pointer}, s{pointer}, s{offset}")
        asm.inst(f"s_addc_u32 s{pointer + 1}, s{pointer + 1}, 0")
