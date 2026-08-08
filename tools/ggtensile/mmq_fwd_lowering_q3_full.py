"""Full-weight Q3_K forward lowering with typed LDS ownership."""

from dataclasses import dataclass
from typing import ClassVar, cast

from .kernel_writer_assembly import (
    Assembly,
    emit_bf16_rne,
    emit_kernel_trailer,
    emit_pointer_kernarg_loads,
)
from .mmq_fwd_lowering import ForwardLoweringContext
from .mmq_fwd_lowering_mma import emit_signed_i8_wmma
from .mmq_fwd_physical import (
    Q3FullWeightTiledLdsPhysicalPlan,
    Q3FullWeightTiledLdsRegisterPlan,
)
from .mmq_fwd_spec import Q3FullWeightTiledLdsLayout


@dataclass(frozen=True)
class FullWeightQ3TiledLdsLowering:
    """Emit the qualified full-block Q3 dataflow from typed ownership facts."""

    context: ForwardLoweringContext

    KERNARG: ClassVar[int] = 4
    LOOP_COUNTER: ClassVar[int] = 10
    ACTIVATION_PLANE_SGPR: ClassVar[int] = 11

    @property
    def _registers(self) -> Q3FullWeightTiledLdsRegisterPlan:
        physical = cast(
            Q3FullWeightTiledLdsPhysicalPlan,
            self.context.state.physical_plan,
        )
        return physical.registers

    def _register(self, role: str) -> int:
        return getattr(self._registers, role).first_register

    @property
    def WEIGHT_RAW_BASE(self) -> int:
        return self._register("weight_metadata")

    @property
    def WEIGHT_LOW0(self) -> int:
        return self._register("weight_low_raw")

    @property
    def WEIGHT_HIGH(self) -> int:
        return self._register("weight_high_raw")

    @property
    def WEIGHT_LOW1(self) -> int:
        return self._register("weight_low1_raw")

    @property
    def DECODED(self) -> int:
        return self._register("decoded_payload")

    @property
    def DECODE_AUX(self) -> int:
        return self._register("decode_auxiliary")

    @property
    def DECODE_D(self) -> int:
        return self._register("decode_d")

    @property
    def DECODE_SCALE(self) -> int:
        return self._register("decode_scale")

    @property
    def WEIGHT_STAGE_ADDRESS(self) -> int:
        return self._register("weight_stage_address")

    @property
    def HALF_SHIFT(self) -> int:
        return self._register("half_shift")

    @property
    def WEIGHT_LDS_ADDRESS(self) -> int:
        return self._register("weight_lds_address")

    @property
    def WEIGHT_SCALE_BASE(self) -> int:
        return self._register("weight_scale_address")

    @property
    def TEMPORARY(self) -> int:
        return self._register("temporary")

    @property
    def LANE(self) -> int:
        return self._register("lane")

    @property
    def WAVE(self) -> int:
        return self._register("wave")

    @property
    def ACTIVATION_LDS_ADDRESS(self) -> int:
        return self._register("activation_lds_address")

    @property
    def ACTIVATION_READ_ADDRESS(self) -> int:
        return self._register("activation_read_address")

    @property
    def WEIGHT_ADDRESS(self) -> int:
        return self._register("weight_address")

    @property
    def SUMS(self) -> int:
        return self._register("sums")

    @property
    def ACTIVATION_STAGE(self) -> int:
        return self._register("activation_stage")

    @property
    def C(self) -> int:
        return self._register("c")

    @property
    def WEIGHT_PAYLOAD(self) -> int:
        return self._register("weight_payload")

    @property
    def WEIGHT_SCALES(self) -> int:
        return self._register("weight_scales")

    @property
    def ACTIVATION_SCALES(self) -> int:
        return self._register("activation_scale")

    @property
    def ZERO(self) -> int:
        return self._register("zero_accumulator")

    def body(self) -> str:
        physical = cast(
            Q3FullWeightTiledLdsPhysicalPlan,
            self.context.state.physical_plan,
        )
        layout = physical.layout
        state = self.context.state
        asm = Assembly()
        name = self.context.solution_key.kernel_name
        row_stride = state.packed_weight_row_bytes
        activation_plane_stride = state.activation_plane_stride_bytes
        blocks = state.blocks_per_weight_row

        asm.comment("Load pointers for the typed full-weight Q3 tile.")
        emit_pointer_kernarg_loads(asm, self.KERNARG)
        asm.comment("Map each wave to sixteen output-feature rows.")
        asm.inst(f"v_bfe_u32 v{self.WAVE}, v0, 10, 10")
        asm.inst(f"v_and_b32 v{self.LANE}, 0x3ff, v0")
        asm.comment("Hoist the invariant activation LDS lane base.")
        asm.inst(f"v_and_b32 v{self.ACTIVATION_READ_ADDRESS}, 15, v{self.LANE}")
        asm.inst(
            f"v_mul_lo_u32 v{self.ACTIVATION_READ_ADDRESS}, "
            f"{layout.activation_row_stride}, v{self.ACTIVATION_READ_ADDRESS}"
        )
        asm.inst(f"v_and_b32 v{self.TEMPORARY}, 15, v{self.LANE}")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY + 1}, 4, v{self.WAVE}")
        asm.inst(
            f"v_add_nc_u32 v{self.TEMPORARY}, v{self.TEMPORARY + 1}, v{self.TEMPORARY}"
        )
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY + 1}, 6, s2")
        asm.inst(
            f"v_add_nc_u32 v{self.TEMPORARY}, v{self.TEMPORARY + 1}, v{self.TEMPORARY}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{self.WEIGHT_ADDRESS}, {row_stride}, v{self.TEMPORARY}"
        )

        asm.comment("Build global and LDS addresses for all activation rows.")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY}, 5, v{self.WAVE}")
        asm.inst(f"v_add_nc_u32 v{self.TEMPORARY}, v{self.LANE}, v{self.TEMPORARY}")
        asm.inst(
            f"v_mul_lo_u32 v{self.ACTIVATION_LDS_ADDRESS}, "
            f"{layout.activation_row_stride}, v{self.TEMPORARY}"
        )
        asm.inst(
            f"s_mul_i32 s{self.ACTIVATION_PLANE_SGPR}, {layout.activation_bytes}, s3"
        )

        asm.comment("Build one decoded-weight LDS row per output feature.")
        asm.inst(f"v_and_b32 v{self.TEMPORARY}, 15, v{self.LANE}")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY + 1}, 4, v{self.WAVE}")
        asm.inst(
            f"v_add_nc_u32 v{self.TEMPORARY}, v{self.TEMPORARY + 1}, v{self.TEMPORARY}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{self.WEIGHT_LDS_ADDRESS}, "
            f"{layout.weight_row_stride}, v{self.TEMPORARY}"
        )
        asm.inst(
            f"v_add_nc_u32 v{self.WEIGHT_LDS_ADDRESS}, {layout.weight_base}, "
            f"v{self.WEIGHT_LDS_ADDRESS}"
        )

        for register in range(self.SUMS, self.SUMS + 64):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.comment("Initialize one persistent zero source for all Q3 WMMAs.")
        for element in range(0, 8, 2):
            asm.inst(
                f"v_dual_mov_b32 v{self.ZERO + element}, 0 :: "
                f"v_dual_mov_b32 v{self.ZERO + element + 1}, 0"
            )
        self._emit_weight_prefetch(asm, half=0)
        self._emit_weight_scale_bases(asm, layout)
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")
        asm.label(".LForwardQ3KFullWeightBlockLoop")

        self._emit_activation_stage(asm, layout)
        self._emit_weight_decode(asm, layout, half=0, wait_for_vmem=True)
        self._emit_weight_decode(asm, layout, half=1, wait_for_vmem=False)
        asm.inst("s_waitcnt vmcnt(0)")
        self._emit_activation_stores(asm, layout)
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        self._emit_compute_half(asm, layout, half=0)
        asm.inst("s_barrier")

        asm.inst(
            f"s_add_u32 s{self.ACTIVATION_PLANE_SGPR}, "
            f"s{self.ACTIVATION_PLANE_SGPR}, {activation_plane_stride}"
        )
        self._emit_activation_stage(asm, layout)
        asm.inst("s_waitcnt vmcnt(0)")
        self._emit_activation_stores(asm, layout)
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        self._emit_compute_half(asm, layout, half=1)
        asm.inst("s_barrier")
        asm.inst(
            f"s_add_u32 s{self.ACTIVATION_PLANE_SGPR}, "
            f"s{self.ACTIVATION_PLANE_SGPR}, {activation_plane_stride}"
        )

        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {blocks}")
        asm.inst("s_cbranch_scc0 .LForwardQ3KFullWeightEnd")
        asm.inst(
            f"v_add_nc_u32 v{self.WEIGHT_ADDRESS}, "
            f"{self.context.state.contract.packed_weight_block_bytes}, "
            f"v{self.WEIGHT_ADDRESS}"
        )
        self._emit_weight_prefetch(asm, half=0)
        asm.inst("s_branch .LForwardQ3KFullWeightBlockLoop")
        asm.label(".LForwardQ3KFullWeightEnd")
        self._emit_store(asm)
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_activation_stage(
        self,
        asm: Assembly,
        layout: Q3FullWeightTiledLdsLayout,
    ) -> None:
        asm.comment("Stage 128 contiguous Q8_1 F32_D4 rows into LDS.")
        base = self.ACTIVATION_STAGE
        asm.inst(
            f"v_add_nc_u32 v{self.TEMPORARY}, s{self.ACTIVATION_PLANE_SGPR}, "
            f"v{self.ACTIVATION_LDS_ADDRESS}"
        )
        for chunk in range(layout.activation_row_stride // 16):
            payload = base + 4 * chunk
            asm.inst(
                f"global_load_b128 v[{payload}:{payload + 3}], "
                f"v{self.TEMPORARY}, s[6:7] offset:{16 * chunk}"
            )

    def _emit_activation_stores(
        self,
        asm: Assembly,
        layout: Q3FullWeightTiledLdsLayout,
    ) -> None:
        base = self.ACTIVATION_STAGE
        for chunk in range(layout.activation_row_stride // 16):
            payload = base + 4 * chunk
            asm.inst(
                f"ds_write_b128 v{self.ACTIVATION_LDS_ADDRESS}, "
                f"v[{payload}:{payload + 3}] offset:{16 * chunk}"
            )

    def _emit_weight_prefetch(self, asm: Assembly, *, half: int) -> None:
        if half != 0:
            raise ValueError("full-weight prefetch currently starts at half zero")
        asm.comment("Prefetch Q3_K half 0 raw operands for the next block.")
        asm.inst(f"v_bfe_u32 v{self.TEMPORARY}, v{self.LANE}, 4, 1")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY + 1}, 4, v{self.TEMPORARY}")
        asm.inst(
            f"v_add_nc_u32 v{self.WEIGHT_STAGE_ADDRESS}, "
            f"v{self.TEMPORARY + 1}, v{self.WEIGHT_ADDRESS}"
        )
        asm.inst(
            f"global_load_b128 v[{self.WEIGHT_LOW0}:{self.WEIGHT_LOW0 + 3}], "
            f"v{self.WEIGHT_STAGE_ADDRESS}, s[4:5] offset:32"
        )
        asm.inst(
            f"global_load_b128 v[{self.WEIGHT_HIGH}:{self.WEIGHT_HIGH + 3}], "
            f"v{self.WEIGHT_STAGE_ADDRESS}, s[4:5] offset:0"
        )
        asm.inst(
            f"global_load_b128 v[{self.WEIGHT_RAW_BASE}:{self.WEIGHT_RAW_BASE + 3}], "
            f"v{self.WEIGHT_ADDRESS}, s[4:5] offset:94"
        )
        asm.inst(
            f"global_load_b128 v[{self.WEIGHT_LOW1}:{self.WEIGHT_LOW1 + 3}], "
            f"v{self.WEIGHT_STAGE_ADDRESS}, s[4:5] offset:64"
        )

    def _emit_weight_scale_bases(
        self,
        asm: Assembly,
        layout: Q3FullWeightTiledLdsLayout,
    ) -> None:
        asm.comment(
            "Hoist invariant weight-scale LDS row bases across both halves and all K blocks."
        )
        asm.inst(f"v_bfe_u32 v{self.TEMPORARY}, v{self.LANE}, 4, 1")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY + 1}, 4, v{self.WAVE}")
        asm.inst(
            f"v_add_nc_u32 v{self.TEMPORARY}, v{self.TEMPORARY + 1}, v{self.TEMPORARY}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{self.WEIGHT_SCALE_BASE}, "
            f"{layout.weight_row_stride}, v{self.TEMPORARY}"
        )
        asm.inst(
            f"v_add_nc_u32 v{self.WEIGHT_SCALE_BASE}, {layout.weight_base}, "
            f"v{self.WEIGHT_SCALE_BASE}"
        )
        for index in range(1, 4):
            asm.inst(
                f"v_add_nc_u32 v{self.WEIGHT_SCALE_BASE + index}, "
                f"{index * 4 * layout.weight_row_stride}, "
                f"v{self.WEIGHT_SCALE_BASE}"
            )

    def _emit_weight_decode(
        self,
        asm: Assembly,
        layout: Q3FullWeightTiledLdsLayout,
        *,
        half: int,
        wait_for_vmem: bool,
    ) -> None:
        asm.comment(
            f"Decode Q3_K half {half} into signed int8 payloads and FP32 scales."
        )
        asm.inst(f"v_bfe_u32 v{self.TEMPORARY}, v{self.LANE}, 4, 1")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY + 1}, 4, v{self.TEMPORARY}")
        asm.inst(
            f"v_add_nc_u32 v{self.WEIGHT_STAGE_ADDRESS}, "
            f"v{self.TEMPORARY + 1}, v{self.WEIGHT_ADDRESS}"
        )
        asm.inst(f"v_lshlrev_b32 v{self.HALF_SHIFT}, 3, v{self.TEMPORARY}")
        if wait_for_vmem:
            asm.inst("s_waitcnt vmcnt(9)")
        asm.inst(
            f"v_add_nc_u32 v{self.WEIGHT_STAGE_ADDRESS}, "
            f"v{self.TEMPORARY + 1}, v{self.WEIGHT_LDS_ADDRESS}"
        )
        if half:
            asm.inst(
                f"v_add_nc_u32 v{self.WEIGHT_STAGE_ADDRESS}, "
                f"{layout.half_payload_stride}, v{self.WEIGHT_STAGE_ADDRESS}"
            )
        low_base = self.WEIGHT_LOW0 if half == 0 else self.WEIGHT_LOW1
        high_base = self.WEIGHT_HIGH
        for local_group in range(4):
            # Q3 payload groups are interleaved by two logical groups per lane.
            # The top bit wraps after groups 0 and 2, so derive its position from
            # the logical group rather than from the half-local loop index.
            logical_group = 4 * half + local_group
            low_shift = 2 * local_group
            high_shift = (logical_group + 6) % 8
            for item in range(4):
                destination = self.DECODED + item
                asm.inst(
                    f"v_lshrrev_b32 v{destination}, {low_shift}, v{low_base + item}"
                )
                asm.inst(f"v_and_b32 v{destination}, 0x03030303, v{destination}")
                if high_shift >= 6:
                    asm.inst(
                        f"v_lshlrev_b32 v{self.DECODE_AUX}, "
                        f"{8 - high_shift}, v{high_base + item}"
                    )
                else:
                    asm.inst(
                        f"v_lshrrev_b32 v{self.DECODE_AUX}, {high_shift}, "
                        f"v{high_base + item}"
                    )
                asm.inst(
                    f"v_and_or_b32 v{destination}, v{self.DECODE_AUX}, "
                    f"0x04040404, v{destination}"
                )
                asm.inst(f"v_add_nc_u32 v{destination}, 0x7c7c7c7c, v{destination}")
                asm.inst(f"v_xor_b32 v{destination}, 0x80808080, v{destination}")
            asm.inst(
                f"ds_write_b128 v{self.WEIGHT_STAGE_ADDRESS}, "
                f"v[{self.DECODED}:{self.DECODED + 3}] "
                f"offset:{32 * local_group}"
            )

        asm.inst(f"v_cvt_f32_f16 v{self.DECODE_D}, v{self.WEIGHT_RAW_BASE + 3}.h")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY + 1}, 2, v{self.TEMPORARY}")
        asm.inst(
            f"v_add_nc_u32 v{self.WEIGHT_STAGE_ADDRESS}, "
            f"v{self.TEMPORARY + 1}, v{self.WEIGHT_LDS_ADDRESS}"
        )
        if half:
            asm.inst(
                f"v_add_nc_u32 v{self.WEIGHT_STAGE_ADDRESS}, "
                f"{layout.half_payload_stride}, v{self.WEIGHT_STAGE_ADDRESS}"
            )
        semantics = self.context.state.semantics
        metadata_load_offset = semantics.payload_plane("scales").byte_offset - 2
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
            asm.inst(f"v_add_nc_u32 v{self.DECODE_AUX}, {low_bit}, v{self.HALF_SHIFT}")
            asm.inst(
                f"v_lshrrev_b32 v{self.DECODE_SCALE}, v{self.DECODE_AUX}, "
                f"v{self.WEIGHT_RAW_BASE + low_word}"
            )
            asm.inst(
                f"v_and_b32 v{self.DECODE_SCALE}, "
                f"{(1 << low_field.bit_count) - 1}, v{self.DECODE_SCALE}"
            )
            asm.inst(f"v_add_nc_u32 v{self.DECODE_AUX}, {high_bit}, v{self.HALF_SHIFT}")
            asm.inst(
                f"v_lshrrev_b32 v{self.DECODE_AUX}, v{self.DECODE_AUX}, "
                f"v{self.WEIGHT_RAW_BASE + high_word}"
            )
            asm.inst(
                f"v_and_b32 v{self.DECODE_AUX}, "
                f"{(1 << high_field.bit_count) - 1}, v{self.DECODE_AUX}"
            )
            asm.inst(
                f"v_lshl_or_b32 v{self.DECODE_SCALE}, v{self.DECODE_AUX}, "
                f"{high_field.destination_shift}, v{self.DECODE_SCALE}"
            )
            asm.inst(f"v_sub_nc_u32 v{self.DECODE_SCALE}, v{self.DECODE_SCALE}, 32")
            asm.inst(f"v_cvt_f32_i32 v{self.DECODE_SCALE}, v{self.DECODE_SCALE}")
            asm.inst(
                f"v_mul_f32 v{self.DECODE_SCALE}, v{self.DECODE_D}, "
                f"v{self.DECODE_SCALE}"
            )
            asm.inst(
                f"ds_write_b32 v{self.WEIGHT_STAGE_ADDRESS}, "
                f"v{self.DECODE_SCALE} offset:{layout.weight_scale_offset + 8 * local_group}"
            )

    def _emit_compute_half(
        self,
        asm: Assembly,
        layout: Q3FullWeightTiledLdsLayout,
        *,
        half: int,
    ) -> None:
        waits_even = (15, 14, 5, 4, 3, 2, 1, 0)
        waits_odd = (11, 10, 5, 4, 3, 2, 1, 0)
        pair_order = ((0, 3), (1, 2), (4, 7), (5, 6))
        for group in range(8):
            group_spec = self.context.state.semantics.q3_payload_group(group)
            asm.comment(f"Q3_K half group {group}: signed WMMA and FP32 correction.")
            asm.inst(
                f"ds_read_b128 v[{self.ACTIVATION_STAGE}:{self.ACTIVATION_STAGE + 3}], "
                f"v{self.ACTIVATION_READ_ADDRESS} "
                f"offset:{group_spec.activation_payload_offset}"
            )
            asm.inst(
                f"ds_read_b128 v[{self.WEIGHT_PAYLOAD}:{self.WEIGHT_PAYLOAD + 3}], "
                f"v{self.WEIGHT_LDS_ADDRESS} offset:{layout.half_payload_stride * half + 16 * group}"
            )
            asm.inst(
                f"ds_read_b128 v[{self.ACTIVATION_STAGE + 4}:{self.ACTIVATION_STAGE + 7}], "
                f"v{self.ACTIVATION_READ_ADDRESS} "
                f"offset:{16 * layout.activation_row_stride + group_spec.activation_payload_offset}"
            )
            weight_offset0 = 32 + 40 * half + group
            weight_offset1 = 200 + 40 * half + group
            for index in range(4):
                base = self.WEIGHT_SCALE_BASE + index
                destination = self.WEIGHT_SCALES + 2 * index
                asm.inst(
                    f"ds_read2_b32 v[{destination}:{destination + 1}], v{base} "
                    f"offset0:{weight_offset0} offset1:{weight_offset1}"
                )
            if group % 2 == 0:
                asm.inst(
                    f"v_add_nc_u32 v{self.TEMPORARY}, {4 * (group // 2)}, "
                    f"v{self.ACTIVATION_READ_ADDRESS}"
                )
                for index, (offset0, offset1) in enumerate(
                    ((0, 9), (18, 27), (36, 45), (54, 63))
                ):
                    destination = self.ACTIVATION_SCALES + 2 * index
                    asm.inst(
                        f"ds_read2st64_b32 v[{destination}:{destination + 1}], "
                        f"v{self.TEMPORARY} offset0:{offset0} offset1:{offset1}"
                    )
            for m_index in range(2, 8):
                payload = self.ACTIVATION_STAGE + 4 * m_index
                asm.inst(
                    f"ds_read_b128 v[{payload}:{payload + 3}], "
                    f"v{self.ACTIVATION_READ_ADDRESS} "
                    f"offset:{16 * m_index * layout.activation_row_stride + group_spec.activation_payload_offset}"
                )
            for wait, m_index in zip(
                waits_even if group % 2 == 0 else waits_odd,
                range(8),
            ):
                asm.inst(f"s_waitcnt lgkmcnt({wait})")
                emit_signed_i8_wmma(
                    asm,
                    destination=self.C + 8 * m_index,
                    weight=self.WEIGHT_PAYLOAD,
                    activation=self.ACTIVATION_STAGE + 4 * m_index,
                    accumulator=self.ZERO,
                    clamp=False,
                )
            for first, second in pair_order:
                self._emit_fragment_correction(asm, first, second)

    def _emit_fragment_correction(
        self,
        asm: Assembly,
        first: int,
        second: int,
    ) -> None:
        for fragment in (first, second):
            base = self.C + 8 * fragment
            for element in range(8):
                asm.inst(f"v_cvt_f32_i32 v{base + element}, v{base + element}")
            for element in range(0, 8, 2):
                asm.inst(
                    f"v_dual_mul_f32 v{base + element}, "
                    f"v{self.WEIGHT_SCALES + element}, v{base + element} :: "
                    f"v_dual_mul_f32 v{base + element + 1}, "
                    f"v{self.WEIGHT_SCALES + element + 1}, "
                    f"v{base + element + 1}"
                )
        first_sum = self.SUMS + 8 * first
        second_sum = self.SUMS + 8 * second
        first_scale = self.ACTIVATION_SCALES + first
        second_scale = self.ACTIVATION_SCALES + second
        second_base = self.C + 8 * second
        for element in range(8):
            asm.inst(
                f"v_dual_fmac_f32 v{first_sum + element}, v{first_scale}, "
                f"v{self.C + 8 * first + element} :: "
                f"v_dual_fmac_f32 v{second_sum + (element ^ 3)}, v{second_scale}, "
                f"v{second_base + (element ^ 3)}"
            )

    def _emit_store(self, asm: Assembly) -> None:
        size = self.context.state.problem_size
        asm.comment("Store the Q3 wave-N fragments as row-major BF16.")
        asm.inst(f"v_and_b32 v{self.TEMPORARY}, 15, v{self.LANE}")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY + 1}, 7, s3")
        asm.inst(
            f"v_add_nc_u32 v{self.TEMPORARY}, v{self.TEMPORARY + 1}, v{self.TEMPORARY}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{self.ACTIVATION_STAGE}, {2 * size.n}, v{self.TEMPORARY}"
        )
        asm.inst(f"v_bfe_u32 v{self.TEMPORARY}, v{self.LANE}, 4, 1")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY + 1}, 4, v{self.WAVE}")
        asm.inst(
            f"v_add_nc_u32 v{self.TEMPORARY}, v{self.TEMPORARY + 1}, v{self.TEMPORARY}"
        )
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY + 1}, 6, s2")
        asm.inst(
            f"v_add_nc_u32 v{self.TEMPORARY}, v{self.TEMPORARY + 1}, v{self.TEMPORARY}"
        )
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY}, 1, v{self.TEMPORARY}")
        asm.inst(
            f"v_add_nc_u32 v{self.ACTIVATION_STAGE}, v{self.TEMPORARY}, "
            f"v{self.ACTIVATION_STAGE}"
        )
        for m_index in range(8):
            fragment = self.SUMS + 8 * m_index
            for element in range(8):
                emit_bf16_rne(asm, fragment + element, self.TEMPORARY)
            asm.inst("s_clause 7")
            for element in range(8):
                asm.inst(
                    f"global_store_d16_hi_b16 v{self.ACTIVATION_STAGE}, "
                    f"v{fragment + element}, s[8:9] offset:{4 * element}"
                )
            if m_index != 7:
                asm.inst(
                    f"v_add_nc_u32 v{self.ACTIVATION_STAGE}, {32 * size.n}, "
                    f"v{self.ACTIVATION_STAGE}"
                )
