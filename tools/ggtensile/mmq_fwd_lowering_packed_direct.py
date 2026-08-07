"""Direct packed scale/minimum forward lowering."""

from dataclasses import dataclass
from typing import ClassVar, cast

from .kernel_writer_assembly import (
    Assembly,
    emit_bf16_rne,
    emit_kernel_trailer,
    emit_pointer_kernarg_loads,
)
from .mmq_fwd_lowering import ForwardLoweringContext
from .mmq_fwd_lowering_metadata import emit_packed_scale_minimum
from .mmq_fwd_lowering_mma import emit_signed_i8_wmma
from .mmq_fwd_physical import (
    PackedScaleMinimumDirectPhysicalPlan,
    PackedScaleMinimumDirectRegisterPlan,
)


@dataclass(frozen=True)
class PackedScaleMinimumDirectLowering:
    """Emit the validated direct packed scale/minimum mechanism."""

    context: ForwardLoweringContext

    OPERAND_SOURCE: ClassVar[str] = "Global"
    KERNARG: ClassVar[int] = 4
    LOOP_COUNTER: ClassVar[int] = 10

    @property
    def _direct_physical(self) -> PackedScaleMinimumDirectPhysicalPlan:
        return cast(
            PackedScaleMinimumDirectPhysicalPlan,
            self.context.state.physical_plan,
        )

    @property
    def _direct_registers(self) -> PackedScaleMinimumDirectRegisterPlan:
        return self._direct_physical.registers

    def body(self) -> str:
        operand_source = self.context.state.kernel_spec.global_memory.operand_source
        if operand_source != self.OPERAND_SOURCE:
            raise TypeError(
                f"unsupported direct packed operand source {operand_source!r}"
            )
        return self._body_global()

    def _body_global(self) -> str:
        registers = self._direct_registers
        activation_metadata = self._direct_physical.activation_metadata
        asm = Assembly()
        name = self.context.solution_key.kernel_name
        quant_type = self.context.state.contract.quant_type
        row_stride = self.context.state.packed_weight_row_bytes
        activation_plane_stride = self.context.state.activation_plane_stride_bytes
        activation_block_stride = (
            self.context.state.activation_weight_block_stride_bytes
        )

        asm.comment(
            "Load the exact packed-weight, Q8_1 F16_D4S4 workspace, and output pointers."
        )
        emit_pointer_kernarg_loads(asm, self.KERNARG)

        asm.comment("Map one wave to an exact 16x16 output tile.")
        asm.inst(f"v_mov_b32 v{registers.serial.first_register}, v0")
        asm.inst(
            f"v_and_b32 v{registers.output_column.first_register}, 15, v{registers.serial.first_register}"
        )
        asm.inst(f"v_lshlrev_b32 v{registers.temporary.first_register}, 4, s2")
        asm.inst(
            f"v_add_nc_u32 v{registers.output_column.first_register}, v{registers.temporary.first_register}, "
            f"v{registers.output_column.first_register}"
        )
        asm.inst(
            f"v_and_b32 v{registers.activation_row.first_register}, 15, v{registers.serial.first_register}"
        )
        asm.inst(f"v_lshlrev_b32 v{registers.temporary.first_register}, 4, s3")
        asm.inst(
            f"v_add_nc_u32 v{registers.activation_row.first_register}, v{registers.temporary.first_register}, "
            f"v{registers.activation_row.first_register}"
        )
        asm.comment("Build one Q payload row and eight C-fragment metadata rows.")
        asm.inst(
            f"v_lshrrev_b32 v{registers.temporary.first_register}, 4, v{registers.serial.first_register}"
        )
        asm.inst(
            f"v_and_b32 v{registers.temporary.first_register}, 1, v{registers.temporary.first_register}"
        )
        asm.inst(f"v_lshlrev_b32 v{registers.scale.first_register}, 4, s2")
        asm.inst(
            f"v_add_nc_u32 v{registers.temporary.first_register}, v{registers.scale.first_register}, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{registers.result_weight_addresses.first_register}, {row_stride}, "
            f"v{registers.temporary.first_register}"
        )
        for element in range(1, 8):
            asm.inst(
                f"v_add_nc_u32 v{registers.result_weight_addresses.first_register + element}, "
                f"{2 * element * row_stride}, v{registers.result_weight_addresses.first_register}"
            )
        asm.inst(
            f"v_mul_lo_u32 v{registers.weight_address.first_register}, {row_stride}, "
            f"v{registers.output_column.first_register}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{registers.activation_addresses.first_register}, "
            f"{activation_metadata.block_bytes}, v{registers.activation_row.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.activation_addresses.first_register + 1}, "
            f"{activation_plane_stride}, v{registers.activation_addresses.first_register}"
        )
        for register in range(
            registers.sums.first_register, registers.sums.first_register + 8
        ):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")

        quant_label = quant_type.replace("_", "")
        asm.label(f".LForward{quant_label}BlockLoop")
        asm.comment(
            f"Keep one {quant_type} block's dm/scales live across its eight groups."
        )
        for element in range(8):
            metadata = registers.weight_metadata.first_register + 4 * element
            asm.inst(
                f"global_load_b128 v[{metadata}:{metadata + 3}], "
                f"v{registers.result_weight_addresses.first_register + element}, "
                f"s[{self.KERNARG}:{self.KERNARG + 1}]"
            )
        for group in range(8):
            self._emit_group(asm, group)
        for element in range(8):
            asm.inst(
                f"v_add_nc_u32 v{registers.result_weight_addresses.first_register + element}, "
                f"{self.context.state.contract.packed_weight_block_bytes}, "
                f"v{registers.result_weight_addresses.first_register + element}"
            )
        asm.inst(
            f"v_add_nc_u32 v{registers.weight_address.first_register}, "
            f"{self.context.state.contract.packed_weight_block_bytes}, v{registers.weight_address.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.activation_addresses.first_register}, "
            f"{activation_block_stride}, v{registers.activation_addresses.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.activation_addresses.first_register + 1}, "
            f"{activation_block_stride}, v{registers.activation_addresses.first_register + 1}"
        )
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(
            f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {self.context.state.blocks_per_weight_row}"
        )
        asm.inst(f"s_cbranch_scc1 .LForward{quant_label}BlockLoop")

        self._emit_store(asm)
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_group(self, asm: Assembly, group: int) -> None:
        registers = self._direct_registers
        activation_group = self._direct_physical.activation_metadata.group(group)
        weight_q_offset, weight_q_offset_high = (
            self.context.state.semantics.low_payload_group_offsets(group)
        )
        activation_address = (
            registers.activation_addresses.first_register + activation_group.plane
        )

        asm.comment(
            f"Q4_K group {group}: direct nibbles and fixed Q8_1 F16_D4S4 block."
        )
        asm.inst(
            f"global_load_b128 v[{registers.weight_payload.first_register}:{registers.weight_payload.first_register + 3}], "
            f"v{registers.weight_address.first_register}, s[{self.KERNARG}:{self.KERNARG + 1}] "
            f"offset:{weight_q_offset}"
        )
        asm.inst(
            f"global_load_b128 v[{registers.weight_payload.first_register + 4}:{registers.weight_payload.first_register + 7}], "
            f"v{registers.weight_address.first_register}, s[{self.KERNARG}:{self.KERNARG + 1}] "
            f"offset:{weight_q_offset_high}"
        )
        asm.inst(
            f"global_load_b128 v[{registers.activation_payload.first_register}:{registers.activation_payload.first_register + 3}], "
            f"v{activation_address}, s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
            f"offset:{activation_group.payload_offset}"
        )
        asm.inst(
            f"global_load_b128 v[{registers.activation_payload.first_register + 4}:{registers.activation_payload.first_register + 7}], "
            f"v{activation_address}, s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
            f"offset:{activation_group.payload_high_offset}"
        )
        asm.inst(
            f"global_load_b32 v{registers.activation_scale_sum.first_register}, v{activation_address}, "
            f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
            f"offset:{activation_group.scale_sum_offset}"
        )
        asm.inst("s_waitcnt vmcnt(0)")

        for register in range(
            registers.weight_payload.first_register,
            registers.weight_payload.first_register + 8,
        ):
            if group & 1:
                asm.inst(f"v_lshrrev_b32 v{register}, 4, v{register}")
            asm.inst(f"v_and_b32 v{register}, 0x0f0f0f0f, v{register}")
        for register in range(
            registers.c.first_register, registers.c.first_register + 8
        ):
            asm.inst(f"v_mov_b32 v{register}, 0")
        emit_signed_i8_wmma(
            asm,
            destination=registers.c.first_register,
            weight=registers.weight_payload.first_register,
            activation=registers.activation_payload.first_register,
            accumulator=registers.c.first_register,
            clamp=self.context.state.contract.wmma_clamp,
        )
        emit_signed_i8_wmma(
            asm,
            destination=registers.c.first_register,
            weight=registers.weight_payload.first_register + 4,
            activation=registers.activation_payload.first_register + 4,
            accumulator=registers.c.first_register,
            clamp=self.context.state.contract.wmma_clamp,
        )
        self._emit_scaled_accumulate(asm, group)

    def _emit_scaled_accumulate(self, asm: Assembly, group: int) -> None:
        registers = self._direct_registers
        asm.comment("Reproduce Q4_K FP16 scale/min construction before FP32 sums.")
        asm.inst(
            f"v_cvt_f32_f16 v{registers.activation_d.first_register}, v{registers.activation_scale_sum.first_register}"
        )
        asm.inst(
            f"v_cvt_f32_f16 v{registers.activation_sum.first_register}, v{registers.activation_scale_sum.first_register}.h"
        )
        for element in range(8):
            metadata = registers.weight_metadata.first_register + 4 * element
            emit_packed_scale_minimum(
                asm,
                group,
                metadata,
                semantics=self.context.state.semantics,
                scale=registers.scale.first_register,
                minimum=registers.minimum.first_register,
                temporary=registers.temporary.first_register,
            )
            asm.inst(
                f"v_cvt_f32_u32 v{registers.weight_d.first_register}, v{registers.scale.first_register}"
            )
            asm.inst(
                f"v_cvt_f32_u32 v{registers.weight_minimum.first_register}, v{registers.minimum.first_register}"
            )
            asm.inst(
                f"v_cvt_f16_f32_e32 v{registers.scaled_dm.first_register}.l, v{registers.weight_d.first_register}"
            )
            asm.inst(
                f"v_cvt_f16_f32_e32 v{registers.scaled_dm.first_register}.h, v{registers.weight_minimum.first_register}"
            )
            asm.inst(
                f"v_pk_mul_f16 v{registers.scaled_dm.first_register}, 0xbc003c00, v{registers.scaled_dm.first_register}"
            )
            asm.inst(
                f"v_pk_mul_f16 v{registers.scaled_dm.first_register}, v{metadata}, v{registers.scaled_dm.first_register}"
            )
            asm.inst(
                f"v_cvt_f32_f16 v{registers.weight_d.first_register}, v{registers.scaled_dm.first_register}"
            )
            asm.inst(
                f"v_cvt_f32_f16 v{registers.weight_minimum.first_register}, v{registers.scaled_dm.first_register}.h"
            )
            c = registers.c.first_register + element
            total = registers.sums.first_register + element
            asm.inst(f"v_cvt_f32_i32 v{registers.temporary.first_register}, v{c}")
            asm.inst(
                f"v_mul_f32 v{registers.temporary.first_register}, v{registers.weight_d.first_register}, v{registers.temporary.first_register}"
            )
            asm.inst(
                f"v_fma_f32 v{total}, v{registers.temporary.first_register}, v{registers.activation_d.first_register}, v{total}"
            )
            asm.inst(
                f"v_fma_f32 v{total}, v{registers.weight_minimum.first_register}, "
                f"v{registers.activation_sum.first_register}, v{total}"
            )

    def _emit_store(self, asm: Assembly) -> None:
        registers = self._direct_registers
        size = self.context.state.problem_size
        asm.comment("Store the gfx11 J-major C fragments as row-major BF16 output.")
        asm.inst(
            f"v_and_b32 v{registers.temporary.first_register}, 15, v{registers.serial.first_register}"
        )
        asm.inst(f"v_lshlrev_b32 v{registers.output_address.first_register}, 4, s3")
        asm.inst(
            f"v_add_nc_u32 v{registers.output_address.first_register}, v{registers.output_address.first_register}, "
            f"v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{registers.output_address.first_register}, {2 * size.n}, v{registers.output_address.first_register}"
        )
        asm.inst(
            f"v_lshrrev_b32 v{registers.temporary.first_register}, 4, v{registers.serial.first_register}"
        )
        asm.inst(
            f"v_and_b32 v{registers.temporary.first_register}, 1, v{registers.temporary.first_register}"
        )
        asm.inst(f"v_lshlrev_b32 v{registers.temporary.first_register + 1}, 4, s2")
        asm.inst(
            f"v_add_nc_u32 v{registers.temporary.first_register}, v{registers.temporary.first_register + 1}, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_lshlrev_b32 v{registers.temporary.first_register}, 1, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.output_address.first_register}, v{registers.output_address.first_register}, "
            f"v{registers.temporary.first_register}"
        )
        for element in range(8):
            total = registers.sums.first_register + element
            emit_bf16_rne(asm, total, registers.temporary.first_register)
            asm.inst(
                f"global_store_d16_hi_b16 v{registers.output_address.first_register}, v{total}, "
                f"s[{self.KERNARG + 4}:{self.KERNARG + 5}]"
            )
            if element != 7:
                asm.inst(
                    f"v_add_nc_u32 v{registers.output_address.first_register}, 4, v{registers.output_address.first_register}"
                )
