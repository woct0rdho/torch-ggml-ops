"""Dedicated six-argument lowering for fixed-group Q8_0 forward."""

from dataclasses import dataclass
from typing import ClassVar, cast

from .fixed_grouped_mmq_fwd_model import FixedForwardSolutionKey
from .fixed_grouped_mmq_fwd_spec import DerivedFixedForwardState
from .kernel_writer_assembly import (
    Assembly,
    emit_bf16_rne,
    emit_kernel_trailer,
    emit_pointer_kernarg_loads,
)
from .mmq_fwd_lowering_signed_i8 import SignedInt8ForwardLowering
from .mmq_fwd_physical import (
    SignedInt8SmallMTiledLdsPhysicalPlan,
    SignedInt8TiledLdsRegisters,
    SignedInt8TiledLdsScaleLayout,
)
from .model import ProblemSize


@dataclass(frozen=True)
class FixedForwardLoweringContext:
    solution_key: FixedForwardSolutionKey
    state: DerivedFixedForwardState


@dataclass(frozen=True)
class FixedGroupedQ8ForwardLowering(SignedInt8ForwardLowering):
    """Reuse signed-int8 staging and WMMA helpers under the fixed ABI."""

    context: FixedForwardLoweringContext
    KERNARG: ClassVar[int] = 6
    LOOP_COUNTER: ClassVar[int] = 5

    def body(self) -> str:
        return self._body_fixed_grouped_small_m_tiled_lds()

    def _body_fixed_grouped_small_m_tiled_lds(self) -> str:
        asm = Assembly()
        state = self.context.state
        macro_tile_m = state.kernel_spec.macro_tile[0]
        m_fragments = macro_tile_m // 16
        activation_row_share = 128 // macro_tile_m
        groups_per_lane = 4 // activation_row_share
        physical = cast(SignedInt8SmallMTiledLdsPhysicalPlan, state.physical_plan)
        registers = physical.registers
        name = self.context.solution_key.kernel_name
        size = state.problem_size
        row_stride = state.packed_weight_row_bytes
        iteration_count = state.activation_blocks_per_row
        sums = registers.sums.first_register
        zero_accumulator = registers.zero_accumulator.first_register
        weight_scale_address = registers.weight_scale_address.first_register
        weight_address = registers.weight_address.first_register
        activation_address = registers.activation_address.first_register
        activation_scale_stage_address = (
            registers.activation_scale_stage_address.first_register
        )
        activation_lds_address = registers.activation_lds_address.first_register
        activation_scale_lds_address = (
            registers.activation_scale_lds_address.first_register
        )
        weight_lds_address = registers.weight_lds_address.first_register
        temporary = registers.temporary.first_register
        lane = registers.lane.first_register
        wave = registers.wave.first_register
        activation_row = registers.activation_row.first_register
        activation_group = registers.activation_group.first_register
        activation_read_address = registers.activation_read_address.first_register
        weight_row = registers.weight_row.first_register

        layout = physical.layout
        policy = physical.policy
        if policy.stage_order != "WeightThenActivation":
            raise ValueError("fixed Q8 lowering requires weight-first staging")
        tiled_registers: SignedInt8TiledLdsRegisters = registers
        tiled_scale_layout: SignedInt8TiledLdsScaleLayout = layout
        activation_lds_row_stride = layout.activation_row_stride
        weight_lds_base = layout.weight_base
        weight_lds_row_stride = layout.weight_row_stride
        paired_scale_reads = policy.scale_read == "PairedHoistedSecondBase"

        asm.comment("Load the fixed-group Q8_0 pointers and scalar dimensions.")
        emit_pointer_kernarg_loads(asm, self.KERNARG)
        asm.inst(f"s_load_dword s{self.KERNARG + 6}, s[0:1], 0x18")
        asm.inst(f"s_load_dword s{self.KERNARG + 7}, s[0:1], 0x1c")
        asm.inst(
            f"s_load_dwordx2 s[{self.KERNARG + 8}:{self.KERNARG + 9}], s[0:1], 0x20"
        )
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.comment("Map four waves to sixteen output-feature rows each.")
        asm.inst(f"v_bfe_u32 v{wave}, v0, 10, 10")
        asm.inst(f"v_and_b32 v{lane}, 0x3ff, v0")
        asm.inst(f"v_and_b32 v{temporary}, 15, v{lane}")
        asm.inst(f"v_lshlrev_b32 v{weight_row}, 4, v{wave}")
        asm.inst(f"v_add_nc_u32 v{weight_row}, v{weight_row}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary}, v{weight_row}")
        asm.inst(f"v_mul_lo_u32 v{weight_address}, {row_stride}, v{temporary}")
        asm.comment("Select the packed weight group with workgroup z.")
        asm.inst(f"v_mul_lo_u32 v{temporary}, s4, s{self.KERNARG + 8}")
        asm.inst(f"v_add_nc_u32 v{weight_address}, v{temporary}, v{weight_address}")

        asm.comment("Map the activation tile to token rows within one fixed group.")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 5, v{wave}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary}, v{lane}")
        asm.inst(
            f"v_and_b32 v{activation_group}, {activation_row_share - 1}, v{temporary}"
        )
        asm.inst(
            f"v_lshrrev_b32 v{activation_row}, "
            f"{activation_row_share.bit_length() - 1}, v{temporary}"
        )
        if groups_per_lane > 1:
            asm.inst(
                f"v_lshlrev_b32 v{activation_group}, "
                f"{groups_per_lane.bit_length() - 1}, v{activation_group}"
            )
        asm.inst(f"v_lshlrev_b32 v{temporary}, 3, v{activation_row}")
        asm.inst(f"v_add_nc_u32 v{temporary}, s4, v{temporary}")
        asm.inst(
            f"v_mul_lo_u32 v{activation_address}, "
            f"{activation_lds_row_stride}, v{temporary}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{temporary}, "
            f"{activation_lds_row_stride * macro_tile_m * 8}, s3"
        )
        asm.inst(
            f"v_add_nc_u32 v{activation_address}, v{temporary}, v{activation_address}"
        )
        asm.inst(
            f"v_lshlrev_b32 v{activation_scale_stage_address}, 2, v{activation_group}"
        )
        asm.inst(
            f"v_add_nc_u32 v{activation_scale_stage_address}, "
            f"v{activation_address}, v{activation_scale_stage_address}"
        )
        asm.inst(f"v_lshlrev_b32 v{temporary}, 5, v{activation_group}")
        asm.inst(
            f"v_add_nc_u32 v{activation_address}, v{activation_address}, v{temporary}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{activation_lds_address}, "
            f"{activation_lds_row_stride}, v{activation_row}"
        )
        asm.inst(
            f"v_lshlrev_b32 v{activation_scale_lds_address}, 2, v{activation_group}"
        )
        asm.inst(
            f"v_add_nc_u32 v{activation_scale_lds_address}, "
            f"v{activation_lds_address}, v{activation_scale_lds_address}"
        )
        asm.inst(f"v_lshlrev_b32 v{temporary}, 5, v{activation_group}")
        asm.inst(
            f"v_add_nc_u32 v{activation_lds_address}, "
            f"v{activation_lds_address}, v{temporary}"
        )
        asm.inst(f"v_and_b32 v{activation_read_address}, 15, v{lane}")
        asm.inst(
            f"v_mul_lo_u32 v{activation_read_address}, "
            f"{activation_lds_row_stride}, v{activation_read_address}"
        )
        asm.inst(f"v_mul_lo_u32 v{temporary}, {weight_lds_row_stride}, v{weight_row}")
        asm.inst(f"v_add_nc_u32 v{weight_lds_address}, {weight_lds_base}, v{temporary}")

        for register in range(sums, sums + 8 * m_fragments, 2):
            asm.inst(
                f"v_dual_mov_b32 v{register}, 0 :: v_dual_mov_b32 v{register + 1}, 0"
            )
        for register in range(zero_accumulator, zero_accumulator + 8, 2):
            asm.inst(
                f"v_dual_mov_b32 v{register}, 0 :: v_dual_mov_b32 v{register + 1}, 0"
            )
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")

        asm.label(".LFixedGroupedQ80BlockLoop")
        self._emit_signed_int8_tiled_weight_stage(
            asm,
            tiled_registers,
            tiled_scale_layout,
            group_base=0,
        )
        self._emit_signed_int8_tiled_activation_loads(
            asm,
            tiled_registers,
            registers.activation_address,
            registers.activation_scale_stage_address,
            groups_per_lane,
        )
        self._emit_signed_int8_tiled_weight_writes(
            asm,
            tiled_registers,
            group_base=0,
            wait_counts=(6, 0),
        )
        self._emit_signed_int8_tiled_activation_writes(
            asm,
            tiled_registers,
            registers.activation_lds_address,
            registers.activation_scale_lds_address,
            groups_per_lane,
            trailing_vmem=0,
        )
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{lane}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{weight_scale_address}, 4, v{wave}")
        asm.inst(
            f"v_add_nc_u32 v{weight_scale_address}, "
            f"v{weight_scale_address}, v{temporary}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{weight_scale_address}, "
            f"{weight_lds_row_stride}, v{weight_scale_address}"
        )
        asm.inst(
            f"v_add_nc_u32 v{weight_scale_address}, {weight_lds_base}, "
            f"v{weight_scale_address}"
        )
        if paired_scale_reads:
            weight_scale_pair_base_delta = layout.weight_scale_pair_base_delta
            if weight_scale_pair_base_delta is None:
                raise ValueError("paired Q8 scale reads require a second-base delta")
            asm.inst(
                f"v_add_nc_u32 v{temporary}, {weight_scale_pair_base_delta}, "
                f"v{weight_scale_address}"
            )
        for group in range(4):
            self._emit_signed_int8_tiled_group(
                asm,
                group,
                tiled_registers,
                tiled_scale_layout,
                policy,
                m_fragments=m_fragments,
            )
        asm.inst("s_barrier")
        asm.inst(
            f"v_add_nc_u32 v{weight_address}, "
            f"{4 * state.contract.packed_weight_block_bytes}, v{weight_address}"
        )
        asm.inst(f"v_mul_lo_u32 v{temporary}, 1152, s{self.KERNARG + 6}")
        for address in (activation_address, activation_scale_stage_address):
            asm.inst(f"v_add_nc_u32 v{address}, v{temporary}, v{address}")
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {iteration_count}")
        asm.inst("s_cbranch_scc1 .LFixedGroupedQ80BlockLoop")

        self._emit_fixed_grouped_tiled_store(asm, tiled_registers, size, m_fragments)
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_fixed_grouped_tiled_store(
        self,
        asm: Assembly,
        registers: SignedInt8TiledLdsRegisters,
        size: ProblemSize,
        m_fragments: int,
    ) -> None:
        """Store the tile through the flattened [tokens*groups, N] view."""
        sums = registers.sums.first_register
        output_address = registers.output_address.first_register
        store_auxiliary = registers.store_auxiliary.first_register
        lane = registers.lane.first_register
        wave = registers.wave.first_register
        temporary = registers.temporary.first_register
        asm.comment("Store fixed-group rows as BF16 RNE in [tokens, groups, N].")
        asm.inst(f"v_and_b32 v{temporary}, 15, v{lane}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 3, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{output_address}, 9, s3")
        asm.inst(f"v_add_nc_u32 v{output_address}, s4, v{output_address}")
        asm.inst(f"v_add_nc_u32 v{output_address}, v{output_address}, v{temporary}")
        asm.inst(
            f"v_mul_lo_u32 v{output_address}, s{self.KERNARG + 7}, v{output_address}"
        )
        asm.inst(f"v_lshlrev_b32 v{output_address}, 1, v{output_address}")
        asm.inst(f"v_lshlrev_b32 v{store_auxiliary}, 6, s2")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 4, v{wave}")
        asm.inst(f"v_add_nc_u32 v{store_auxiliary}, v{store_auxiliary}, v{temporary}")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{lane}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{store_auxiliary}, v{store_auxiliary}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{store_auxiliary}, 1, v{store_auxiliary}")
        asm.inst(
            f"v_add_nc_u32 v{output_address}, v{output_address}, v{store_auxiliary}"
        )
        for m_index in range(m_fragments):
            address_base = output_address + 8 * m_index
            if m_index:
                asm.inst(f"v_lshlrev_b32 v{temporary}, 8, s{self.KERNARG + 7}")
                asm.inst(
                    f"v_add_nc_u32 v{address_base}, v{temporary}, v{address_base - 8}"
                )
            for element in range(1, 8):
                asm.inst(
                    f"v_add_nc_u32 v{address_base + element}, 4, "
                    f"v{address_base + element - 1}"
                )
            fragment = sums + 8 * m_index
            for element in range(8):
                emit_bf16_rne(asm, fragment + element, temporary)
        asm.inst(f"s_clause {8 * m_fragments - 1}")
        for m_index in range(m_fragments):
            address_base = output_address + 8 * m_index
            fragment = sums + 8 * m_index
            for element in range(8):
                asm.inst(
                    f"global_store_d16_hi_b16 v{address_base + element}, "
                    f"v{fragment + element}, s[{self.KERNARG + 4}:{self.KERNARG + 5}]"
                )
