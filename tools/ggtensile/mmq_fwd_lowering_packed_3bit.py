"""Packed three-bit half-tile MMQ forward lowering."""

from dataclasses import dataclass
from typing import ClassVar, cast

from .kernel_abi import ORDINARY_FORWARD_ABI
from .kernel_writer_assembly import (
    Assembly,
    emit_bf16_rne,
    emit_kernel_trailer,
    emit_pointer_kernarg_loads,
)
from .mmq_fwd_lowering import ForwardLoweringContext
from .mmq_fwd_lowering_mma import emit_signed_i8_wmma
from .mmq_fwd_physical import (
    Packed3BitTiledLdsPhysicalPlan,
    Packed3BitTiledLdsRegisterPlan,
)
from .mmq_fwd_spec import Packed3BitTiledLdsLayout


@dataclass(frozen=True)
class Packed3BitTiledLdsLowering:
    """Emit the validated packed-Q3 half-tile mechanism."""

    context: ForwardLoweringContext

    KERNARG: ClassVar[int] = 4
    LOOP_COUNTER: ClassVar[int] = 10

    def body(self) -> str:
        """Lower the isolated wave-N 128x64 Q3_K LDS research control."""
        asm = Assembly()
        physical = cast(
            Packed3BitTiledLdsPhysicalPlan, self.context.state.physical_plan
        )
        registers = physical.registers
        layout = physical.layout
        name = self.context.solution_key.kernel_name
        row_stride = self.context.state.packed_weight_row_bytes
        activation_plane_stride = self.context.state.activation_plane_stride_bytes
        sums = registers.sums.first_register
        weight_address = registers.weight_address.first_register
        activation_address = registers.activation_address.first_register
        activation_lds_address = registers.activation_lds_address.first_register
        activation_read_address = registers.activation_read_address.first_register
        weight_lds_address = registers.weight_lds_address.first_register
        zero_accumulator = registers.zero_accumulator.first_register
        temporary = registers.temporary.first_register
        lane = registers.lane.first_register
        wave = registers.wave.first_register

        asm.comment("Load pointers for the four-wave Q3_K half-tile control.")
        emit_pointer_kernarg_loads(asm, self.KERNARG, ORDINARY_FORWARD_ABI)
        asm.comment("Map each wave to sixteen output-feature rows.")
        asm.inst(f"v_bfe_u32 v{wave}, v0, 10, 10")
        asm.inst(f"v_and_b32 v{lane}, 0x3ff, v0")
        asm.comment("Hoist the invariant activation LDS lane base.")
        asm.inst(f"v_and_b32 v{activation_read_address}, 15, v{lane}")
        asm.inst(
            f"v_mul_lo_u32 v{activation_read_address}, "
            f"{layout.activation_row_stride}, v{activation_read_address}"
        )
        asm.inst(f"v_and_b32 v{temporary}, 15, v{lane}")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 4, v{wave}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary + 1}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary + 1}, v{temporary}")
        asm.inst(f"v_mul_lo_u32 v{weight_address}, {row_stride}, v{temporary}")

        asm.comment("Build global and LDS addresses for all 128 activation rows.")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 5, v{wave}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{lane}, v{temporary}")
        asm.inst(
            f"v_mul_lo_u32 v{activation_lds_address}, "
            f"{layout.activation_row_stride}, v{temporary}"
        )
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 7, s3")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary + 1}, v{temporary}")
        asm.inst(
            f"v_mul_lo_u32 v{activation_address}, "
            f"{layout.activation_row_stride}, v{temporary}"
        )

        asm.comment("Build one decoded-weight LDS row per output feature.")
        asm.inst(f"v_and_b32 v{temporary}, 15, v{lane}")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 4, v{wave}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary + 1}, v{temporary}")
        asm.inst(
            f"v_mul_lo_u32 v{weight_lds_address}, "
            f"{layout.weight_row_stride}, v{temporary}"
        )
        asm.inst(
            f"v_add_nc_u32 v{weight_lds_address}, {layout.weight_base}, "
            f"v{weight_lds_address}"
        )

        for register in range(sums, sums + 64):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.comment("Initialize one persistent zero source for all Q3 WMMAs.")
        for element in range(0, 8, 2):
            asm.inst(
                f"v_dual_mov_b32 v{zero_accumulator + element}, 0 :: "
                f"v_dual_mov_b32 v{zero_accumulator + element + 1}, 0"
            )
        self._emit_packed_3bit_weight_prefetch(asm, registers, half=0)
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")

        asm.label(".LForwardQ3KHipTiledLdsBlockLoop")
        for half in range(2):
            self._emit_packed_3bit_activation_stage(asm, registers, layout)
            self._emit_packed_3bit_weight_stage(
                asm,
                registers,
                layout,
                half,
                preloaded=half == 0,
                store_activation=True,
            )
            asm.inst("s_waitcnt lgkmcnt(0)")
            asm.inst("s_barrier")
            self._emit_packed_3bit_half_accumulate(asm, registers, layout)
            asm.inst("s_barrier")
            asm.inst(
                f"v_add_nc_u32 v{activation_address}, "
                f"{activation_plane_stride}, v{activation_address}"
            )
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(
            f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {self.context.state.blocks_per_weight_row}"
        )
        asm.inst("s_cbranch_scc0 .LForwardQ3KHipTiledLdsEnd")
        asm.inst(
            f"v_add_nc_u32 v{weight_address}, "
            f"{self.context.state.contract.packed_weight_block_bytes}, v{weight_address}"
        )
        self._emit_packed_3bit_weight_prefetch(asm, registers, half=0)
        asm.inst("s_branch .LForwardQ3KHipTiledLdsBlockLoop")
        asm.label(".LForwardQ3KHipTiledLdsEnd")

        self._emit_packed_3bit_store(asm, registers)
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_packed_3bit_activation_stage(
        self,
        asm: Assembly,
        registers: Packed3BitTiledLdsRegisterPlan,
        layout: Packed3BitTiledLdsLayout,
    ) -> None:
        """Issue one 128-value Q8_1 activation block stage."""
        activation_stage = registers.activation_stage.first_register
        activation_address = registers.activation_address.first_register
        asm.comment("Stage 128 contiguous Q8_1 F32_D4 rows into LDS.")
        for chunk in range(layout.activation_row_stride // 16):
            offset = 16 * chunk
            payload = activation_stage + 4 * chunk
            asm.inst(
                f"global_load_b128 v[{payload}:{payload + 3}], "
                f"v{activation_address}, "
                f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] offset:{offset}"
            )

    def _emit_packed_3bit_activation_stores(
        self,
        asm: Assembly,
        registers: Packed3BitTiledLdsRegisterPlan,
        layout: Packed3BitTiledLdsLayout,
    ) -> None:
        """Store the ready activation stage into its cooperative LDS tile."""
        activation_stage = registers.activation_stage.first_register
        activation_lds_address = registers.activation_lds_address.first_register
        for chunk in range(layout.activation_row_stride // 16):
            offset = 16 * chunk
            payload = activation_stage + 4 * chunk
            asm.inst(
                f"ds_write_b128 v{activation_lds_address}, "
                f"v[{payload}:{payload + 3}] offset:{offset}"
            )

    def _emit_packed_3bit_weight_prefetch(
        self,
        asm: Assembly,
        registers: Packed3BitTiledLdsRegisterPlan,
        *,
        half: int,
    ) -> None:
        """Issue one Q3 half's raw VMEM reads for the following block."""
        semantics = self.context.state.semantics
        first_group = semantics.q3_payload_group(8 * half)
        low_raw = registers.weight_low_raw.first_register
        high_raw = registers.weight_high_raw.first_register
        metadata = registers.weight_metadata.first_register
        stage_address = registers.weight_stage_address.first_register
        weight_address = registers.weight_address.first_register
        temporary = registers.temporary.first_register
        lane = registers.lane.first_register
        metadata_load_offset = semantics.payload_plane("scales").byte_offset - 2

        asm.comment(f"Prefetch Q3_K half {half} raw operands for the next block.")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{lane}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 4, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{stage_address}, v{temporary + 1}, v{weight_address}")
        asm.inst(
            f"global_load_b128 v[{low_raw}:{low_raw + 3}], v{stage_address}, "
            f"s[{self.KERNARG}:{self.KERNARG + 1}] "
            f"offset:{first_group.low_payload_offset}"
        )
        asm.inst(
            f"global_load_b128 v[{high_raw}:{high_raw + 3}], v{stage_address}, "
            f"s[{self.KERNARG}:{self.KERNARG + 1}] "
            f"offset:{first_group.high_payload_offset}"
        )
        asm.inst(
            f"global_load_b128 v[{metadata}:{metadata + 3}], v{weight_address}, "
            f"s[{self.KERNARG}:{self.KERNARG + 1}] "
            f"offset:{metadata_load_offset}"
        )

    def _emit_packed_3bit_weight_stage(
        self,
        asm: Assembly,
        registers: Packed3BitTiledLdsRegisterPlan,
        layout: Packed3BitTiledLdsLayout,
        half: int,
        *,
        preloaded: bool = False,
        store_activation: bool = False,
    ) -> None:
        """Decode four interleaved Q3 groups per thread into one LDS half row."""
        semantics = self.context.state.semantics
        decode = semantics.q3_signed_decode()
        first_group = semantics.q3_payload_group(8 * half)
        low_raw = registers.weight_low_raw.first_register
        high_raw = registers.weight_high_raw.first_register
        metadata = registers.weight_metadata.first_register
        decoded = registers.decoded_payload.first_register
        stage_d = registers.stage_d.first_register
        stage_scale = registers.stage_scale.first_register
        auxiliary = registers.stage_auxiliary.first_register
        stage_address = registers.weight_stage_address.first_register
        scale_shift = registers.scale_shift.first_register
        weight_address = registers.weight_address.first_register
        weight_lds_address = registers.weight_lds_address.first_register
        temporary = registers.temporary.first_register
        lane = registers.lane.first_register
        metadata_load_offset = semantics.payload_plane("scales").byte_offset - 2

        asm.comment(
            f"Decode Q3_K half {half} into signed int8 payloads and FP32 scales."
        )
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{lane}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 4, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{stage_address}, v{temporary + 1}, v{weight_address}")
        if not preloaded:
            asm.inst(
                f"global_load_b128 v[{low_raw}:{low_raw + 3}], v{stage_address}, "
                f"s[{self.KERNARG}:{self.KERNARG + 1}] "
                f"offset:{first_group.low_payload_offset}"
            )
            asm.inst(
                f"global_load_b128 v[{high_raw}:{high_raw + 3}], v{stage_address}, "
                f"s[{self.KERNARG}:{self.KERNARG + 1}] "
                f"offset:{first_group.high_payload_offset}"
            )
            asm.inst(
                f"global_load_b128 v[{metadata}:{metadata + 3}], v{weight_address}, "
                f"s[{self.KERNARG}:{self.KERNARG + 1}] "
                f"offset:{metadata_load_offset}"
            )
        asm.inst(f"v_lshlrev_b32 v{scale_shift}, 3, v{temporary}")
        asm.inst("s_waitcnt vmcnt(0)")
        if store_activation:
            self._emit_packed_3bit_activation_stores(asm, registers, layout)

        asm.inst(
            f"v_add_nc_u32 v{stage_address}, v{temporary + 1}, v{weight_lds_address}"
        )
        for local_group in range(4):
            group = 8 * half + 2 * local_group
            group_spec = semantics.q3_payload_group(group)
            for item in range(4):
                asm.inst(
                    f"v_lshrrev_b32 v{decoded + item}, {group_spec.low_shift}, "
                    f"v{low_raw + item}"
                )
                asm.inst(
                    f"v_and_b32 v{decoded + item}, {decode.low_mask:#010x}, "
                    f"v{decoded + item}"
                )
                asm.inst(
                    f"v_lshrrev_b32 v{auxiliary}, {group_spec.high_shift}, "
                    f"v{high_raw + item}"
                )
                asm.inst(
                    f"v_and_b32 v{auxiliary}, {decode.high_mask:#010x}, v{auxiliary}"
                )
                asm.inst(
                    f"v_lshl_or_b32 v{decoded + item}, v{auxiliary}, "
                    f"{decode.high_destination_shift}, v{decoded + item}"
                )
                asm.inst(
                    f"v_add_nc_u32 v{decoded + item}, {decode.signed_add:#010x}, "
                    f"v{decoded + item}"
                )
                asm.inst(
                    f"v_xor_b32 v{decoded + item}, {decode.signed_xor:#010x}, "
                    f"v{decoded + item}"
                )
            asm.inst(
                f"ds_write_b128 v{stage_address}, v[{decoded}:{decoded + 3}] "
                f"offset:{32 * local_group}"
            )

        asm.inst(f"v_cvt_f32_f16 v{stage_d}, v{metadata + 3}.h")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 2, v{temporary}")
        asm.inst(
            f"v_add_nc_u32 v{stage_address}, v{temporary + 1}, v{weight_lds_address}"
        )
        scale_plane_offset = semantics.payload_plane("scales").byte_offset
        for local_group in range(4):
            group = 8 * half + 2 * local_group
            low_field, high_field = semantics.q3_scale_fields(group)
            low_byte = scale_plane_offset - metadata_load_offset + low_field.source_byte
            high_byte = (
                scale_plane_offset - metadata_load_offset + high_field.source_byte
            )
            low_word = low_byte // 4
            high_word = high_byte // 4
            low_bit = 8 * (low_byte % 4) + low_field.bit_offset
            high_bit = 8 * (high_byte % 4) + high_field.bit_offset
            asm.inst(f"v_add_nc_u32 v{auxiliary}, {low_bit}, v{scale_shift}")
            asm.inst(
                f"v_lshrrev_b32 v{stage_scale}, v{auxiliary}, v{metadata + low_word}"
            )
            asm.inst(
                f"v_and_b32 v{stage_scale}, {(1 << low_field.bit_count) - 1}, "
                f"v{stage_scale}"
            )
            asm.inst(f"v_add_nc_u32 v{auxiliary}, {high_bit}, v{scale_shift}")
            asm.inst(
                f"v_lshrrev_b32 v{auxiliary}, v{auxiliary}, v{metadata + high_word}"
            )
            asm.inst(
                f"v_and_b32 v{auxiliary}, {(1 << high_field.bit_count) - 1}, "
                f"v{auxiliary}"
            )
            asm.inst(
                f"v_lshl_or_b32 v{stage_scale}, v{auxiliary}, "
                f"{high_field.destination_shift}, v{stage_scale}"
            )
            asm.inst(f"v_sub_nc_u32 v{stage_scale}, v{stage_scale}, 32")
            asm.inst(f"v_cvt_f32_i32 v{stage_scale}, v{stage_scale}")
            asm.inst(f"v_mul_f32 v{stage_scale}, v{stage_d}, v{stage_scale}")
            asm.inst(
                f"ds_write_b32 v{stage_address}, v{stage_scale} "
                f"offset:{layout.weight_payload_bytes + 8 * local_group}"
            )

    def _emit_packed_3bit_half_accumulate(
        self,
        asm: Assembly,
        registers: Packed3BitTiledLdsRegisterPlan,
        layout: Packed3BitTiledLdsLayout,
    ) -> None:
        """Accumulate eight Q3 scale groups over all eight M fragments."""
        c = registers.c.first_register
        zero_accumulator = registers.zero_accumulator.first_register
        sums = registers.sums.first_register
        weight_payload = registers.weight_payload.first_register
        activation_payload = registers.activation_payload.first_register
        weight_scales = registers.weight_scales.first_register
        activation_scale = registers.activation_scale.first_register
        activation_read_address = registers.activation_read_address.first_register
        weight_scale_address = registers.weight_scale_address.first_register
        weight_lds_address = registers.weight_lds_address.first_register
        temporary = registers.temporary.first_register
        lane = registers.lane.first_register
        wave = registers.wave.first_register

        asm.comment("Build the first C-fragment weight-scale row in LDS.")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{lane}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 4, v{wave}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary + 1}, v{temporary}")
        asm.inst(
            f"v_mul_lo_u32 v{weight_scale_address}, "
            f"{layout.weight_row_stride}, v{temporary}"
        )
        asm.inst(
            f"v_add_nc_u32 v{weight_scale_address}, {layout.weight_base}, "
            f"v{weight_scale_address}"
        )
        for group in range(layout.scale_count):
            group_spec = self.context.state.semantics.q3_payload_group(group)
            asm.comment(f"Q3_K half group {group}: signed WMMA and FP32 correction.")
            asm.inst(
                f"ds_read_b128 v[{weight_payload}:{weight_payload + 3}], "
                f"v{weight_lds_address} offset:{16 * group}"
            )
            for element in range(8):
                asm.inst(
                    f"ds_read_b32 v{weight_scales + element}, "
                    f"v{weight_scale_address} "
                    f"offset:{layout.weight_payload_bytes + 2 * layout.weight_row_stride * element + 4 * group}"
                )
            for m_index in range(8):
                activation_row_offset = 16 * m_index * layout.activation_row_stride
                payload = activation_payload + 4 * m_index
                scale = activation_scale + m_index
                asm.inst(
                    f"ds_read_b128 v[{payload}:{payload + 3}], "
                    f"v{activation_read_address} "
                    f"offset:{activation_row_offset + group_spec.activation_payload_offset}"
                )
                if group % 2 == 0:
                    asm.inst(
                        f"ds_read_b32 v{scale}, v{activation_read_address} "
                        f"offset:{activation_row_offset + group_spec.activation_scale_offset}"
                    )
            asm.inst("s_waitcnt lgkmcnt(0)")
            for m_index in range(8):
                payload = activation_payload + 4 * m_index
                scale = activation_scale + m_index
                activation_scale_copy = temporary + ((temporary ^ scale ^ 1) & 1)
                asm.inst(f"v_mov_b32 v{activation_scale_copy}, v{scale}")
                emit_signed_i8_wmma(
                    asm,
                    destination=c,
                    weight=weight_payload,
                    activation=payload,
                    accumulator=zero_accumulator,
                    clamp=False,
                )
                sum_fragment = sums + 8 * m_index
                for element in range(8):
                    asm.inst(f"v_cvt_f32_i32 v{c + element}, v{c + element}")
                for element in range(0, 8, 2):
                    asm.inst(
                        f"v_dual_mul_f32 v{c + element}, "
                        f"v{weight_scales + element}, v{c + element} :: "
                        f"v_dual_mul_f32 v{c + element + 1}, "
                        f"v{weight_scales + element + 1}, "
                        f"v{c + element + 1}"
                    )
                for element in range(0, 8, 2):
                    asm.inst(
                        f"v_dual_fmac_f32 v{sum_fragment + element}, "
                        f"v{scale}, v{c + element} :: "
                        f"v_dual_fmac_f32 v{sum_fragment + element + 1}, "
                        f"v{activation_scale_copy}, v{c + element + 1}"
                    )

    def _emit_packed_3bit_store(
        self,
        asm: Assembly,
        registers: Packed3BitTiledLdsRegisterPlan,
    ) -> None:
        """Store eight wave-N fragments as row-major BF16 with RNE."""
        size = self.context.state.problem_size
        sums = registers.sums.first_register
        output_address = registers.output_address.first_register
        temporary = registers.temporary.first_register
        lane = registers.lane.first_register
        wave = registers.wave.first_register

        asm.comment("Store the Q3 wave-N fragments as row-major BF16.")
        asm.inst(f"v_and_b32 v{temporary}, 15, v{lane}")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 7, s3")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary + 1}, v{temporary}")
        asm.inst(f"v_mul_lo_u32 v{output_address}, {2 * size.n}, v{temporary}")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{lane}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 4, v{wave}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary + 1}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary + 1}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{output_address}, v{temporary}, v{output_address}")
        for m_index in range(8):
            fragment = sums + 8 * m_index
            for element in range(8):
                emit_bf16_rne(asm, fragment + element, temporary)
            asm.inst("s_clause 7")
            for element in range(8):
                asm.inst(
                    f"global_store_d16_hi_b16 v{output_address}, "
                    f"v{fragment + element}, "
                    f"s[{self.KERNARG + 4}:{self.KERNARG + 5}] "
                    f"offset:{4 * element}"
                )
            if m_index != 7:
                asm.inst(
                    f"v_add_nc_u32 v{output_address}, {32 * size.n}, v{output_address}"
                )
