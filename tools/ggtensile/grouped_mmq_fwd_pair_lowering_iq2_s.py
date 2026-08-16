"""K128-interleaved paired IQ2_S lowering for research artifacts."""

from dataclasses import dataclass
from typing import Any, ClassVar, cast

from .grouped_mmq_fwd_lowering import GroupedForwardLoweringResult
from .grouped_mmq_fwd_lowering_iq2_s import GroupedIQ2SFullWeightLdsLowering
from .grouped_mmq_fwd_pair_model import GroupedPairRouteOwnership
from .grouped_mmq_fwd_pair_physical import (
    GroupedIQ2SPairHalfLdsLayout,
    GroupedIQ2SPairPhysicalPlan,
)
from .grouped_mmq_fwd_pair_route import (
    GroupedPairRouteEmitter,
    GroupedPairRowTaskEmitter,
)
from .grouped_mmq_fwd_pair_spec import (
    DerivedGroupedForwardPairState,
    GroupedForwardPairSolutionKey,
)
from .iq2_s_grid import iq2_s_grid_rodata
from .kernel_writer_assembly import Assembly, emit_bf16_rne, emit_kernel_trailer
from .mmq_fwd_lowering_mma import emit_signed_i8_wmma


@dataclass(frozen=True)
class GroupedForwardPairLoweringContext:
    solution_key: GroupedForwardPairSolutionKey
    state: DerivedGroupedForwardPairState


@dataclass(frozen=True)
class GroupedIQ2SPairedK128Lowering:
    """Emit one J64 tile that interleaves both projections at K128."""

    context: GroupedForwardPairLoweringContext

    GRID_SYMBOL: ClassVar[str] = ".LGGTensilePairIQ2SGrid"
    GRID_BASE: ClassVar[int] = 42

    def _physical_plan(self) -> GroupedIQ2SPairPhysicalPlan:
        return self.context.state.physical_plan

    def _uses_payload_prefetch(self) -> bool:
        return True

    def emission(self) -> GroupedForwardLoweringResult:
        return GroupedForwardLoweringResult(
            self.body(),
            (iq2_s_grid_rodata(self.GRID_SYMBOL),),
        )

    def body(self) -> str:
        physical = self._physical_plan()
        layout = physical.layout
        registers = physical.registers
        scalar = physical.scalar_registers
        state = self.context.state
        asm = Assembly()
        name = self.context.solution_key.kernel_name

        if state.contract.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks64:
            GroupedPairRowTaskEmitter(scalar, state.route).emit(asm)
        else:
            GroupedPairRouteEmitter(scalar, state.route).emit(asm)
        asm.comment("Materialize the paired local IQ2_S codebook address.")
        asm.inst(f"s_getpc_b64 s[{self.GRID_BASE}:{self.GRID_BASE + 1}]")
        asm.inst(
            f"s_add_u32 s{self.GRID_BASE}, s{self.GRID_BASE}, "
            f"{self.GRID_SYMBOL}@rel32@lo+4"
        )
        asm.inst(
            f"s_addc_u32 s{self.GRID_BASE + 1}, s{self.GRID_BASE + 1}, "
            f"{self.GRID_SYMBOL}@rel32@hi+12"
        )
        asm.inst(
            f"s_mul_i32 s{scalar.activation_plane_stride.first_register}, "
            f"s{scalar.nrows_activation.first_register}, "
            f"{self.context.solution_key.solution.activation_block_bytes}"
        )
        asm.inst(
            f"s_mov_b32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_begin.first_register}"
        )
        asm.inst(f"s_mov_b32 s{scalar.group_offset.first_register}, 0x03020100")
        self._emit_invariant_addresses(asm, layout)

        asm.label(".LGroupedPairIQ2SRowLoop")
        asm.inst(
            f"s_sub_u32 s{scalar.row_tile_rows.first_register}, "
            f"s{scalar.row_end.first_register}, s{scalar.row_start.first_register}"
        )
        asm.inst(f"s_cmp_le_u32 s{scalar.row_tile_rows.first_register}, 64")
        asm.inst("s_cbranch_scc1 .LGroupedPairIQ2SRowsReady")
        asm.inst(f"s_mov_b32 s{scalar.row_tile_rows.first_register}, 64")
        asm.label(".LGroupedPairIQ2SRowsReady")
        asm.inst(
            f"s_add_u32 s{scalar.row_tile_end.first_register}, "
            f"s{scalar.row_start.first_register}, s{scalar.row_tile_rows.first_register}"
        )
        asm.inst(
            f"s_mul_i32 s{scalar.packed_block_offset.first_register}, "
            f"s{scalar.row_start.first_register}, "
            f"{self.context.solution_key.solution.activation_block_bytes}"
        )
        for register in (
            *registers.sums_first.registers,
            *registers.sums_second.registers,
        ):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"s_mov_b32 s{scalar.loop_counter.first_register}, 0")

        asm.label(".LGroupedPairIQ2SBlockLoop")
        for half in (0, 1):
            if half:
                asm.inst(
                    f"s_add_u32 s{scalar.packed_block_offset.first_register}, "
                    f"s{scalar.packed_block_offset.first_register}, "
                    f"s{scalar.activation_plane_stride.first_register}"
                )
            self._emit_weight_half_decode(
                asm, layout, scalar.weights_first.first_register, half
            )
            if not half:
                self._emit_activation_stage(asm, layout, stage_index=0)
            else:
                self._emit_activation_stage(asm, layout, stage_index=1)
            asm.inst("s_waitcnt lgkmcnt(0)")
            asm.inst("s_barrier")
            for register in registers.zero_accumulator.registers:
                asm.inst(f"v_mov_b32 v{register}, 0")
            self._emit_compute_projection(asm, registers.sums_first.first_register)
            asm.inst("s_barrier")

            self._emit_weight_half_decode(
                asm, layout, scalar.weights_second.first_register, half
            )
            asm.inst("s_waitcnt lgkmcnt(0)")
            asm.inst("s_barrier")
            for register in registers.zero_accumulator.registers:
                asm.inst(f"v_mov_b32 v{register}, 0")
            self._emit_compute_projection(asm, registers.sums_second.first_register)
            asm.inst("s_barrier")

        asm.inst(
            f"s_add_u32 s{scalar.packed_block_offset.first_register}, "
            f"s{scalar.packed_block_offset.first_register}, "
            f"s{scalar.activation_plane_stride.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.weight_address.first_register}, 82, "
            f"v{registers.weight_address.first_register}"
        )
        asm.inst(
            f"s_add_u32 s{scalar.loop_counter.first_register}, "
            f"s{scalar.loop_counter.first_register}, 1"
        )
        asm.inst(
            f"s_cmp_lt_u32 s{scalar.loop_counter.first_register}, "
            f"{state.blocks_per_weight_row}"
        )
        asm.inst("s_cbranch_scc1 .LGroupedPairIQ2SBlockLoop")
        asm.inst(
            f"v_sub_nc_u32 v{registers.weight_address.first_register}, "
            f"v{registers.weight_address.first_register}, {state.packed_weight_row_bytes}"
        )

        self._emit_store_projection(
            asm,
            registers.sums_first.first_register,
            scalar.output_first.first_register,
            0,
        )
        self._emit_store_projection(
            asm,
            registers.sums_second.first_register,
            scalar.output_second.first_register,
            1,
        )
        asm.inst(
            f"s_add_u32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_start.first_register}, 64"
        )
        asm.inst(
            f"s_cmp_lt_u32 s{scalar.row_start.first_register}, "
            f"s{scalar.row_end.first_register}"
        )
        asm.inst("s_cbranch_scc1 .LGroupedPairIQ2SRowLoop")
        asm.label(".LGroupedPairIQ2SExit")
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_invariant_addresses(
        self, asm: Assembly, layout: GroupedIQ2SPairHalfLdsLayout
    ) -> None:
        registers = self._physical_plan().registers
        scalar = self._physical_plan().scalar_registers
        temporary = registers.temporary.first_register
        asm.comment("Map four waves to 64 rows and split each selected K half.")
        asm.inst(f"v_and_b32 v{registers.lane.first_register}, 31, v0")
        asm.inst(f"v_lshrrev_b32 v{registers.wave.first_register}, 5, v0")
        asm.inst(f"v_and_b32 v{temporary}, 15, v{registers.lane.first_register}")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 4, v{registers.wave.first_register}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary + 1}, v{temporary}")
        asm.inst(
            f"v_mul_lo_u32 v{registers.weight_lds_address.first_register}, "
            f"{layout.weight_row_stride}, v{temporary}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.weight_lds_address.first_register}, "
            f"{layout.weight_base}, v{registers.weight_lds_address.first_register}"
        )
        asm.inst(
            f"v_lshlrev_b32 v{temporary + 1}, 6, s{scalar.workgroup_tile.first_register}"
        )
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary + 1}, v{temporary}")
        asm.inst(
            f"v_mul_lo_u32 v{registers.weight_address.first_register}, "
            f"{self.context.state.packed_weight_row_bytes}, v{temporary}"
        )
        asm.inst(f"v_and_b32 v{temporary}, 15, v{registers.lane.first_register}")
        asm.inst(
            f"v_mul_lo_u32 v{registers.activation_read_address.first_register}, "
            f"{layout.activation_row_stride}, v{temporary}"
        )
        asm.inst(f"v_bfe_u32 v{temporary}, v{registers.lane.first_register}, 4, 1")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 4, v{registers.wave.first_register}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary + 1}, v{temporary}")
        asm.inst(
            f"v_mul_lo_u32 v{registers.weight_scale_address.first_register}, "
            f"{layout.weight_row_stride}, v{temporary}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.weight_scale_address.first_register}, "
            f"{layout.weight_base}, v{registers.weight_scale_address.first_register}"
        )
        for index in range(1, 4):
            asm.inst(
                f"v_add_nc_u32 v{registers.weight_scale_address.first_register + index}, "
                f"{index * 4 * layout.weight_row_stride}, "
                f"v{registers.weight_scale_address.first_register}"
            )

    def _emit_weight_half_decode(
        self,
        asm: Assembly,
        layout: GroupedIQ2SPairHalfLdsLayout,
        weight_scalar: int,
        half: int,
    ) -> None:
        registers = self._physical_plan().registers
        semantics = self.context.state.semantics
        d_offset = semantics.payload_plane("d").byte_offset
        index_offset = semantics.payload_plane("grid_indices").byte_offset
        sign_offset = semantics.payload_plane("signs").byte_offset
        qh_offset = semantics.payload_plane("qh").byte_offset
        scale_offset = semantics.payload_plane("scales").byte_offset
        part = registers.decode_auxiliary.first_register
        address = registers.producer_address.first_register
        asm.comment(
            f"Decode paired IQ2_S selected K128 half {half} in two lane groups."
        )
        asm.inst(
            f"v_mov_b32 v{registers.producer_lds_address.first_register}, "
            f"v{registers.weight_lds_address.first_register}"
        )
        asm.inst(f"v_bfe_u32 v{part}, v{registers.lane.first_register}, 4, 1")
        # The invariant address already names this workitem's output row.
        asm.inst(f"v_lshlrev_b32 v{address}, 3, v{part}")
        asm.inst(
            f"v_add_nc_u32 v{address}, v{registers.weight_address.first_register}, v{address}"
        )
        asm.inst(
            f"global_load_ushort v{registers.producer_d.first_register}, "
            f"v{registers.weight_address.first_register}, "
            f"s[{weight_scalar}:{weight_scalar + 1}] offset:{d_offset}"
        )
        asm.inst(
            f"global_load_b32 v{registers.producer_qh.first_register}, v{registers.weight_address.first_register}, "
            f"s[{weight_scalar}:{weight_scalar + 1}] offset:{qh_offset + 4 * half}"
        )
        asm.inst(
            f"global_load_b32 v{registers.producer_scales.first_register}, v{registers.weight_address.first_register}, "
            f"s[{weight_scalar}:{weight_scalar + 1}] offset:{scale_offset + 4 * half}"
        )
        for local in range(4):
            asm.inst(
                f"global_load_ushort v{registers.producer_indices.first_register + local}, "
                f"v{address}, s[{weight_scalar}:{weight_scalar + 1}] "
                f"offset:{index_offset + 16 * half + 2 * local}"
            )
            asm.inst(
                f"global_load_ushort v{registers.producer_signs.first_register + local}, "
                f"v{address}, s[{weight_scalar}:{weight_scalar + 1}] "
                f"offset:{sign_offset + 16 * half + 2 * local}"
            )
        asm.inst("s_waitcnt vmcnt(0)")
        asm.inst(
            f"v_cvt_f32_f16 v{registers.producer_d.first_register}, "
            f"v{registers.producer_d.first_register}.l"
        )
        asm.inst(
            f"v_mul_f32 v{registers.producer_d.first_register}, 0.25, "
            f"v{registers.producer_d.first_register}"
        )
        index = registers.decode_auxiliary.first_register + 1
        high = registers.decode_auxiliary.first_register + 2
        for local in range(4):
            # The lane's bit-four selects the lower or upper four groups.
            asm.inst(f"v_lshlrev_b32 v{index}, 2, v{part}")
            if local:
                asm.inst(f"v_add_nc_u32 v{index}, {local}, v{index}")
            asm.inst(f"v_lshlrev_b32 v{address}, 2, v{index}")
            asm.inst(
                f"v_bfe_u32 v{high}, v{registers.producer_qh.first_register}, "
                f"v{address}, 2"
            )
            asm.inst(
                f"v_and_b32 v{address}, 0xff, "
                f"v{registers.producer_indices.first_register + local}"
            )
            asm.inst(f"v_lshl_or_b32 v{address}, v{high}, 8, v{address}")
            asm.inst(f"v_lshlrev_b32 v{address}, 3, v{address}")
            asm.inst(
                f"global_load_b64 v[{registers.codebook_payload.first_register + 4 * local}:"
                f"{registers.codebook_payload.first_register + 4 * local + 1}], "
                f"v{address}, s[{self.GRID_BASE}:{self.GRID_BASE + 1}]"
            )
            asm.inst(
                f"v_lshrrev_b32 v{address}, 8, "
                f"v{registers.producer_indices.first_register + local}"
            )
            asm.inst(f"v_lshlrev_b32 v{index}, 2, v{part}")
            if local:
                asm.inst(f"v_add_nc_u32 v{index}, {local}, v{index}")
            asm.inst(f"v_lshlrev_b32 v{high}, 2, v{index}")
            asm.inst(f"v_add_nc_u32 v{high}, 2, v{high}")
            asm.inst(
                f"v_bfe_u32 v{index}, v{registers.producer_qh.first_register}, "
                f"v{high}, 2"
            )
            asm.inst(f"v_lshl_or_b32 v{address}, v{index}, 8, v{address}")
            asm.inst(f"v_lshlrev_b32 v{address}, 3, v{address}")
            asm.inst(
                f"global_load_b64 v[{registers.codebook_payload.first_register + 4 * local + 2}:"
                f"{registers.codebook_payload.first_register + 4 * local + 3}], "
                f"v{address}, s[{self.GRID_BASE}:{self.GRID_BASE + 1}]"
            )
        asm.inst("s_waitcnt vmcnt(0)")
        for local in range(4):
            signs = registers.producer_signs.first_register + local
            grid = registers.codebook_payload.first_register + 4 * local
            decoded = registers.decoded_payload.first_register
            sign_byte = registers.decode_auxiliary.first_register
            for group in range(2):
                if group:
                    asm.inst(f"v_lshrrev_b32 v{sign_byte}, 8, v{signs}")
                else:
                    asm.inst(f"v_and_b32 v{sign_byte}, 0xff, v{signs}")
                self._emit_signed_grid_dword(
                    asm, grid + 2 * group, sign_byte, 0, decoded + 2 * group
                )
                self._emit_signed_grid_dword(
                    asm, grid + 2 * group + 1, sign_byte, 4, decoded + 2 * group + 1
                )
            global_pair = registers.decode_auxiliary.first_register
            asm.inst(
                f"v_bfe_u32 v{global_pair}, v{registers.lane.first_register}, 4, 1"
            )
            asm.inst(f"v_lshlrev_b32 v{global_pair}, 2, v{global_pair}")
            if local:
                asm.inst(f"v_add_nc_u32 v{global_pair}, {local}, v{global_pair}")
            asm.inst(f"v_lshlrev_b32 v{address}, 4, v{global_pair}")
            asm.inst(
                f"v_add_nc_u32 v{address}, v{registers.producer_lds_address.first_register}, v{address}"
            )
            asm.inst(f"ds_write_b128 v{address}, v[{decoded}:{decoded + 3}]")
            scale = registers.decode_auxiliary.first_register + 1
            asm.inst(f"v_lshlrev_b32 v{address}, 2, v{global_pair}")
            asm.inst(
                f"v_add_nc_u32 v{address}, {layout.weight_scale_offset}, v{address}"
            )
            asm.inst(
                f"v_add_nc_u32 v{address}, v{registers.producer_lds_address.first_register}, v{address}"
            )
            asm.inst(f"v_lshlrev_b32 v{scale}, 2, v{global_pair}")
            asm.inst(
                f"v_bfe_u32 v{scale}, v{registers.producer_scales.first_register}, "
                f"v{scale}, 4"
            )
            asm.inst(f"v_cvt_f32_u32 v{scale}, v{scale}")
            asm.inst(f"v_add_f32 v{scale}, 0.5, v{scale}")
            asm.inst(
                f"v_mul_f32 v{scale}, v{registers.producer_d.first_register}, v{scale}"
            )
            asm.inst(f"ds_write_b32 v{address}, v{scale}")

    def _emit_activation_stage(
        self, asm: Assembly, layout: GroupedIQ2SPairHalfLdsLayout, *, stage_index: int
    ) -> None:
        cast(Any, GroupedIQ2SFullWeightLdsLowering._emit_activation_stage)(
            self, asm, layout, stage_index=stage_index
        )

    def _emit_linear_activation_stage(
        self, asm: Assembly, layout: GroupedIQ2SPairHalfLdsLayout, *, stage_index: int
    ) -> None:
        cast(Any, GroupedIQ2SFullWeightLdsLowering._emit_linear_activation_stage)(
            self, asm, layout, stage_index=stage_index
        )

    def _emit_signed_grid_dword(
        self,
        asm: Assembly,
        positive: int,
        sign_byte: int,
        sign_shift: int,
        destination: int,
    ) -> None:
        cast(Any, GroupedIQ2SFullWeightLdsLowering._emit_signed_grid_dword)(
            self, asm, positive, sign_byte, sign_shift, destination
        )

    def _emit_compute_projection(self, asm: Assembly, sums: int) -> None:
        layout = self._physical_plan().layout
        self._emit_compute_group_payload_reads(asm, layout, 0)
        self._emit_compute_group_scale_reads(asm, 0)
        asm.inst("s_waitcnt lgkmcnt(0)")
        self._emit_compute_group_wmmas(asm)
        for group in range(8):
            self._emit_fragment_correction(asm, 0, 3, sums)
            self._emit_fragment_correction(asm, 1, 2, sums)
            if group == 7:
                continue
            next_group = group + 1
            self._emit_compute_group_payload_reads(asm, layout, next_group)
            self._emit_compute_group_scale_reads(asm, next_group)
            pending_scales = 6 if next_group % 2 == 0 else 4
            asm.inst(f"s_waitcnt lgkmcnt({pending_scales})")
            self._emit_compute_group_wmmas(asm)
            asm.inst("s_waitcnt lgkmcnt(0)")

    def _emit_compute_group_payload_reads(
        self, asm: Assembly, layout: GroupedIQ2SPairHalfLdsLayout, group: int
    ) -> None:
        registers = self._physical_plan().registers
        activation_offset = 16 + 16 * group
        for m_index in range(4):
            payload = registers.activation_payload.first_register + 4 * m_index
            asm.inst(
                f"ds_read_b128 v[{payload}:{payload + 3}], v{registers.activation_read_address.first_register} "
                f"offset:{16 * m_index * layout.activation_row_stride + activation_offset}"
            )
        asm.inst(
            f"ds_read_b128 v[{registers.weight_payload.first_register}:{registers.weight_payload.first_register + 3}], "
            f"v{registers.weight_lds_address.first_register} offset:{16 * group}"
        )

    def _emit_compute_group_scale_reads(self, asm: Assembly, group: int) -> None:
        registers = self._physical_plan().registers
        layout = self._physical_plan().layout
        first_row_offset = layout.weight_scale_offset // 4 + group
        second_row_offset = (
            2 * layout.weight_row_stride + layout.weight_scale_offset
        ) // 4 + group
        for index in range(4):
            base = registers.weight_scale_address.first_register + index
            destination = registers.weight_scales.first_register + 2 * index
            asm.inst(
                f"ds_read2_b32 v[{destination}:{destination + 1}], v{base} "
                f"offset0:{first_row_offset} offset1:{second_row_offset}"
            )
        if group % 2 == 0:
            temporary = registers.temporary.first_register
            asm.inst(
                f"v_add_nc_u32 v{temporary}, {4 * (group // 2)}, "
                f"v{registers.activation_read_address.first_register}"
            )
            for index, (offset0, offset1) in enumerate(((0, 9), (18, 27))):
                destination = registers.activation_scale.first_register + 2 * index
                asm.inst(
                    f"ds_read2st64_b32 v[{destination}:{destination + 1}], "
                    f"v{temporary} offset0:{offset0} offset1:{offset1}"
                )

    def _emit_compute_group_wmmas(self, asm: Assembly) -> None:
        registers = self._physical_plan().registers
        for m_index in range(4):
            emit_signed_i8_wmma(
                asm,
                destination=registers.c.first_register + 8 * m_index,
                weight=registers.weight_payload.first_register,
                activation=registers.activation_payload.first_register + 4 * m_index,
                accumulator=registers.zero_accumulator.first_register,
                clamp=False,
            )

    def _emit_fragment_correction(
        self, asm: Assembly, first: int, second: int, sums: int
    ) -> None:
        registers = self._physical_plan().registers
        for fragment in (first, second):
            base = registers.c.first_register + 8 * fragment
            for element in range(8):
                asm.inst(f"v_cvt_f32_i32 v{base + element}, v{base + element}")
            for element in range(0, 8, 2):
                asm.inst(
                    f"v_dual_mul_f32 v{base + element}, v{registers.weight_scales.first_register + element}, v{base + element} :: "
                    f"v_dual_mul_f32 v{base + element + 1}, v{registers.weight_scales.first_register + element + 1}, v{base + element + 1}"
                )
        first_sum = sums + 8 * first
        second_sum = sums + 8 * second
        first_scale = registers.activation_scale.first_register + first
        second_scale = registers.activation_scale.first_register + second
        second_base = registers.c.first_register + 8 * second
        for element in range(8):
            asm.inst(
                f"v_dual_fmac_f32 v{first_sum + element}, v{first_scale}, v{registers.c.first_register + 8 * first + element} :: "
                f"v_dual_fmac_f32 v{second_sum + (element ^ 3)}, v{second_scale}, v{second_base + (element ^ 3)}"
            )

    def _emit_store_projection(
        self, asm: Assembly, sums: int, output: int, projection: int
    ) -> None:
        registers = self._physical_plan().registers
        scalar = self._physical_plan().scalar_registers
        temporary = registers.temporary.first_register
        column = registers.decode_auxiliary.first_register
        asm.comment(f"Store paired IQ2_S projection {projection} J64 BF16 fragments.")
        asm.inst(f"v_bfe_u32 v{column}, v{registers.lane.first_register}, 4, 1")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 4, v{registers.wave.first_register}")
        asm.inst(f"v_add_nc_u32 v{column}, v{temporary}, v{column}")
        asm.inst(
            f"v_lshlrev_b32 v{temporary}, 6, s{scalar.workgroup_tile.first_register}"
        )
        asm.inst(f"v_add_nc_u32 v{column}, v{temporary}, v{column}")
        asm.inst(f"v_lshlrev_b32 v{column}, 1, v{column}")
        for m_index in range(4):
            fragment = sums + 8 * m_index
            for element in range(8):
                emit_bf16_rne(asm, fragment + element, temporary + 1)
            asm.inst(f"v_and_b32 v{temporary}, 15, v{registers.lane.first_register}")
            if m_index:
                asm.inst(f"v_add_nc_u32 v{temporary}, {16 * m_index}, v{temporary}")
            asm.inst(
                f"v_cmp_lt_u32 vcc_lo, v{temporary}, s{scalar.row_tile_rows.first_register}"
            )
            asm.inst(f"s_and_saveexec_b32 s{scalar.exec_mask.first_register}, vcc_lo")
            asm.inst(
                f"v_add_nc_u32 v{temporary}, s{scalar.row_start.first_register}, v{temporary}"
            )
            asm.inst(
                f"v_lshlrev_b32 v{registers.output_address.first_register}, 10, v{temporary}"
            )
            asm.inst(
                f"v_add_nc_u32 v{registers.output_address.first_register}, v{column}, v{registers.output_address.first_register}"
            )
            asm.inst("s_clause 7")
            for element in range(8):
                asm.inst(
                    f"global_store_d16_hi_b16 v{registers.output_address.first_register}, "
                    f"v{fragment + element}, s[{output}:{output + 1}] offset:{4 * element}"
                )
            asm.inst(f"s_mov_b32 exec_lo, s{scalar.exec_mask.first_register}")
