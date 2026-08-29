"""Full-weight Q3_K forward lowering with typed LDS ownership."""

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
    Q3FullWeightTiledLdsPhysicalPlan,
    Q3FullWeightTiledLdsRegisterPlan,
)
from .mmq_fwd_spec import (
    ForwardDataMovementPolicy,
    ForwardDecodeProducerPlan,
    Q3FullForwardDecodePolicy,
    Q3FullWeightTiledLdsLayout,
    Q3PackedFieldPart,
)
from .tuning_policy import PipelineStage


@dataclass(frozen=True)
class FullWeightQ3TiledLdsLowering:
    """Emit the qualified full-block Q3 dataflow from typed ownership facts."""

    context: ForwardLoweringContext

    KERNARG: ClassVar[int] = 4
    LOOP_COUNTER: ClassVar[int] = 10
    ACTIVATION_PLANE_SGPR: ClassVar[int] = 11

    @property
    def _movement(self) -> ForwardDataMovementPolicy:
        physical = cast(
            Q3FullWeightTiledLdsPhysicalPlan, self.context.state.physical_plan
        )
        return physical.data_movement

    @property
    def _prefetch_global_read(self) -> bool:
        physical = cast(
            Q3FullWeightTiledLdsPhysicalPlan, self.context.state.physical_plan
        )
        return physical.pipeline.prefetch_global_read is PipelineStage.DoubleStage

    @property
    def _producer_plan(self) -> ForwardDecodeProducerPlan:
        physical = cast(
            Q3FullWeightTiledLdsPhysicalPlan, self.context.state.physical_plan
        )
        return physical.producer_plan

    def _global_load_bytes(
        self, asm: Assembly, destination: int, address: int, offset: int
    ) -> None:
        width = self._movement.payload_global_read_vector_width
        for chunk in range(0, 16, width):
            first = destination + chunk // 4
            last = first + width // 4 - 1
            source_offset = offset + chunk
            asm.inst(
                f"global_load_b{width * 8} v[{first}:{last}], "
                f"v{address}, s[4:5] offset:{source_offset}"
            )

    def _lds_write_bytes(
        self, asm: Assembly, address: int, source: int, offset: int
    ) -> None:
        width = self._movement.payload_lds_write_vector_width
        for chunk in range(0, 16, width):
            first = source + chunk // 4
            last = first + width // 4 - 1
            asm.inst(
                f"ds_write_b{width * 8} v{address}, v[{first}:{last}] "
                f"offset:{offset + chunk}"
            )

    def _metadata_load_bytes(
        self, asm: Assembly, destination: int, address: int
    ) -> None:
        width = self._movement.metadata_load_vector_width
        for chunk in range(0, 16, width):
            first = destination + chunk // 4
            last = first + width // 4 - 1
            asm.inst(
                f"global_load_b{width * 8} v[{first}:{last}], v{address}, "
                f"s[4:5] offset:{94 + chunk}"
            )

    @property
    def _registers(self) -> Q3FullWeightTiledLdsRegisterPlan:
        physical = cast(
            Q3FullWeightTiledLdsPhysicalPlan,
            self.context.state.physical_plan,
        )
        return physical.registers

    def body(self) -> str:
        physical = self.context.state.physical_plan
        assert isinstance(physical, Q3FullWeightTiledLdsPhysicalPlan)
        layout = physical.layout
        registers = physical.registers
        state = self.context.state
        asm = Assembly()
        name = self.context.kernel_name
        row_stride = state.packed_weight_row_bytes
        activation_plane_stride = state.activation_plane_stride_bytes
        blocks = state.blocks_per_weight_row
        producer_plan = self._producer_plan
        assert producer_plan.producer_wave_ids == (0, 1)

        asm.comment("Load pointers for the typed full-weight Q3 tile.")
        emit_pointer_kernarg_loads(asm, self.KERNARG, ORDINARY_FORWARD_ABI)
        asm.comment("Map each wave to sixteen output-feature rows.")
        asm.inst(f"v_bfe_u32 v{registers.wave.first_register}, v0, 10, 10")
        asm.inst(f"v_and_b32 v{registers.lane.first_register}, 0x3ff, v0")
        asm.comment("Hoist the invariant activation LDS lane base.")
        asm.inst(
            f"v_and_b32 v{registers.activation_read_address.first_register}, 15, v{registers.lane.first_register}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{registers.activation_read_address.first_register}, "
            f"{layout.activation_row_stride}, v{registers.activation_read_address.first_register}"
        )
        asm.inst(
            f"v_and_b32 v{registers.temporary.first_register}, 15, v{registers.lane.first_register}"
        )
        asm.inst(
            f"v_lshlrev_b32 v{registers.temporary.first_register + 1}, 4, v{registers.wave.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.temporary.first_register}, v{registers.temporary.first_register + 1}, v{registers.temporary.first_register}"
        )
        asm.inst(f"v_lshlrev_b32 v{registers.temporary.first_register + 1}, 6, s2")
        asm.inst(
            f"v_add_nc_u32 v{registers.temporary.first_register}, v{registers.temporary.first_register + 1}, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{registers.weight_address.first_register}, {row_stride}, v{registers.temporary.first_register}"
        )

        asm.comment("Build global and LDS addresses for all activation rows.")
        asm.inst(
            f"v_lshlrev_b32 v{registers.temporary.first_register}, 5, v{registers.wave.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.temporary.first_register}, v{registers.lane.first_register}, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{registers.activation_lds_address.first_register}, "
            f"{layout.activation_row_stride}, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"s_mul_i32 s{self.ACTIVATION_PLANE_SGPR}, {layout.activation_bytes}, s3"
        )

        asm.comment("Build one decoded-weight LDS row per output feature.")
        asm.inst(
            f"v_and_b32 v{registers.temporary.first_register}, 15, v{registers.lane.first_register}"
        )
        asm.inst(
            f"v_lshlrev_b32 v{registers.temporary.first_register + 1}, 4, v{registers.wave.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.temporary.first_register}, v{registers.temporary.first_register + 1}, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{registers.weight_lds_address.first_register}, "
            f"{layout.weight_row_stride}, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.weight_lds_address.first_register}, {layout.weight_base}, "
            f"v{registers.weight_lds_address.first_register}"
        )

        for register in range(
            registers.sums.first_register, registers.sums.first_register + 64
        ):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.comment("Initialize one persistent zero source for all Q3 WMMAs.")
        for element in range(0, 8, 2):
            asm.inst(
                f"v_dual_mov_b32 v{registers.zero_accumulator.first_register + element}, 0 :: "
                f"v_dual_mov_b32 v{registers.zero_accumulator.first_register + element + 1}, 0"
            )
        if self._prefetch_global_read:
            self._emit_weight_prefetch(asm, half=0)
        self._emit_weight_scale_bases(asm, layout)
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")
        asm.label(".LForwardQ3KFullWeightBlockLoop")

        self._emit_activation_stage(asm, layout)
        self._emit_weight_decode(
            asm,
            layout,
            half=0,
            wait_for_vmem=self._prefetch_global_read,
            preloaded=self._prefetch_global_read,
        )
        self._emit_weight_decode(
            asm,
            layout,
            half=1,
            wait_for_vmem=False,
            preloaded=True,
        )
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
            f"v_add_nc_u32 v{registers.weight_address.first_register}, "
            f"{self.context.state.contract.packed_weight_block_bytes}, "
            f"v{registers.weight_address.first_register}"
        )
        if self._prefetch_global_read:
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
        registers = self._registers
        asm.comment("Stage 128 contiguous Q8_1 F32_D4 rows into LDS.")
        base = registers.activation_stage.first_register
        asm.inst(
            f"v_add_nc_u32 v{registers.temporary.first_register}, s{self.ACTIVATION_PLANE_SGPR}, "
            f"v{registers.activation_lds_address.first_register}"
        )
        for chunk in range(self.context.state.contract.activation_block_bytes // 16):
            payload = base + 4 * chunk
            asm.inst(
                f"global_load_b128 v[{payload}:{payload + 3}], "
                f"v{registers.temporary.first_register}, s[6:7] offset:{16 * chunk}"
            )

    def _emit_activation_stores(
        self,
        asm: Assembly,
        layout: Q3FullWeightTiledLdsLayout,
    ) -> None:
        registers = self._registers
        base = registers.activation_stage.first_register
        for chunk in range(self.context.state.contract.activation_block_bytes // 16):
            payload = base + 4 * chunk
            self._lds_write_bytes(
                asm,
                registers.activation_lds_address.first_register,
                payload,
                16 * chunk,
            )

    def _emit_weight_prefetch(self, asm: Assembly, *, half: int) -> None:
        registers = self._registers
        assert half == 0
        asm.comment("Prefetch Q3_K half 0 raw operands for the next block.")
        asm.inst(
            f"v_bfe_u32 v{registers.temporary.first_register}, v{registers.lane.first_register}, 4, 1"
        )
        asm.inst(
            f"v_lshlrev_b32 v{registers.temporary.first_register + 1}, 4, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.weight_stage_address.first_register}, "
            f"v{registers.temporary.first_register + 1}, v{registers.weight_address.first_register}"
        )
        self._global_load_bytes(
            asm,
            registers.weight_low_raw.first_register,
            registers.weight_stage_address.first_register,
            32,
        )
        self._global_load_bytes(
            asm,
            registers.weight_high_raw.first_register,
            registers.weight_stage_address.first_register,
            0,
        )
        self._metadata_load_bytes(
            asm,
            registers.weight_metadata.first_register,
            registers.weight_address.first_register,
        )
        self._global_load_bytes(
            asm,
            registers.weight_low1_raw.first_register,
            registers.weight_stage_address.first_register,
            64,
        )

    def _emit_weight_scale_bases(
        self,
        asm: Assembly,
        layout: Q3FullWeightTiledLdsLayout,
    ) -> None:
        registers = self._registers
        asm.comment(
            "Hoist invariant weight-scale LDS row bases across both halves and all K blocks."
        )
        asm.inst(
            f"v_bfe_u32 v{registers.temporary.first_register}, v{registers.lane.first_register}, 4, 1"
        )
        asm.inst(
            f"v_lshlrev_b32 v{registers.temporary.first_register + 1}, 4, v{registers.wave.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.temporary.first_register}, v{registers.temporary.first_register + 1}, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{registers.weight_scale_address.first_register}, "
            f"{layout.weight_row_stride}, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.weight_scale_address.first_register}, {layout.weight_base}, "
            f"v{registers.weight_scale_address.first_register}"
        )
        for index in range(1, 4):
            asm.inst(
                f"v_add_nc_u32 v{registers.weight_scale_address.first_register + index}, "
                f"{index * 4 * layout.weight_row_stride}, "
                f"v{registers.weight_scale_address.first_register}"
            )

    def _emit_weight_decode(
        self,
        asm: Assembly,
        layout: Q3FullWeightTiledLdsLayout,
        *,
        half: int,
        wait_for_vmem: bool,
        preloaded: bool,
    ) -> None:
        registers = self._registers
        asm.comment(
            f"Decode Q3_K half {half} into signed int8 payloads and FP32 scales."
        )
        asm.inst(
            f"v_bfe_u32 v{registers.temporary.first_register}, v{registers.lane.first_register}, 4, 1"
        )
        asm.inst(
            f"v_lshlrev_b32 v{registers.temporary.first_register + 1}, 4, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.weight_stage_address.first_register}, "
            f"v{registers.temporary.first_register + 1}, v{registers.weight_address.first_register}"
        )
        asm.inst(
            f"v_lshlrev_b32 v{registers.half_shift.first_register}, 3, v{registers.temporary.first_register}"
        )
        if not preloaded:
            self._emit_weight_prefetch(asm, half=half)
        if wait_for_vmem:
            asm.inst("s_waitcnt vmcnt(9)")
        elif not preloaded:
            asm.inst("s_waitcnt vmcnt(0)")
        asm.inst(
            f"v_add_nc_u32 v{registers.weight_stage_address.first_register}, "
            f"v{registers.temporary.first_register + 1}, v{registers.weight_lds_address.first_register}"
        )
        if half:
            asm.inst(
                f"v_add_nc_u32 v{registers.weight_stage_address.first_register}, "
                f"{layout.half_payload_stride}, v{registers.weight_stage_address.first_register}"
            )
        low_base = (
            registers.weight_low_raw.first_register
            if half == 0
            else registers.weight_low1_raw.first_register
        )
        high_base = registers.weight_high_raw.first_register
        decode_policy = cast(
            Q3FullForwardDecodePolicy,
            self.context.state.kernel_spec.decode.policy,
        )
        decode_ready_frontier = decode_policy.decode_ready_frontier
        for local_group in range(4):
            # Q3 payload groups are interleaved by two logical groups per lane.
            # The top bit wraps after groups 0 and 2, so derive its position from
            # the logical group rather than from the half-local loop index.
            logical_group = 4 * half + local_group
            low_shift = 2 * local_group
            high_shift = (logical_group + 6) % 8
            if decode_ready_frontier:
                self._emit_payload_decode_ready_frontier(
                    asm,
                    low_base=low_base,
                    high_base=high_base,
                    low_shift=low_shift,
                    high_shift=high_shift,
                )
            else:
                for item in range(4):
                    destination = registers.decoded_payload.first_register + item
                    asm.inst(
                        f"v_lshrrev_b32 v{destination}, {low_shift}, v{low_base + item}"
                    )
                    asm.inst(f"v_and_b32 v{destination}, 0x03030303, v{destination}")
                    if high_shift >= 6:
                        asm.inst(
                            f"v_lshlrev_b32 v{registers.decode_auxiliary.first_register}, "
                            f"{8 - high_shift}, v{high_base + item}"
                        )
                    else:
                        asm.inst(
                            f"v_lshrrev_b32 v{registers.decode_auxiliary.first_register}, {high_shift}, "
                            f"v{high_base + item}"
                        )
                    asm.inst(
                        f"v_and_or_b32 v{destination}, v{registers.decode_auxiliary.first_register}, "
                        f"0x04040404, v{destination}"
                    )
                    asm.inst(f"v_add_nc_u32 v{destination}, 0x7c7c7c7c, v{destination}")
                    asm.inst(f"v_xor_b32 v{destination}, 0x80808080, v{destination}")
            self._lds_write_bytes(
                asm,
                registers.weight_stage_address.first_register,
                registers.decoded_payload.first_register,
                32 * local_group,
            )

        asm.inst(
            f"v_cvt_f32_f16 v{registers.decode_d.first_register}, v{registers.weight_metadata.first_register + 3}.h"
        )
        asm.inst(
            f"v_lshlrev_b32 v{registers.temporary.first_register + 1}, 2, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.weight_stage_address.first_register}, "
            f"v{registers.temporary.first_register + 1}, v{registers.weight_lds_address.first_register}"
        )
        if half:
            asm.inst(
                f"v_add_nc_u32 v{registers.weight_stage_address.first_register}, "
                f"{layout.half_payload_stride}, v{registers.weight_stage_address.first_register}"
            )
        semantics = self.context.state.semantics
        metadata_load_offset = semantics.payload_plane("scales").byte_offset - 2
        scale_plane_offset = semantics.payload_plane("scales").byte_offset
        if decode_ready_frontier:
            self._emit_scale_decode_ready_frontier(
                asm,
                layout,
                half=half,
                metadata_load_offset=metadata_load_offset,
                scale_plane_offset=scale_plane_offset,
            )
        else:
            self._emit_serial_scale_decode(
                asm,
                layout,
                half=half,
                metadata_load_offset=metadata_load_offset,
                scale_plane_offset=scale_plane_offset,
            )

    def _emit_payload_decode_ready_frontier(
        self,
        asm: Assembly,
        *,
        low_base: int,
        high_base: int,
        low_shift: int,
        high_shift: int,
    ) -> None:
        registers = self._registers
        payload = registers.decoded_payload.first_register
        high = registers.decode_payload_high.first_register
        for item in range(4):
            asm.inst(
                f"v_lshrrev_b32 v{payload + item}, {low_shift}, v{low_base + item}"
            )
        for item in range(4):
            operation = "v_lshlrev_b32" if high_shift >= 6 else "v_lshrrev_b32"
            shift = 8 - high_shift if high_shift >= 6 else high_shift
            asm.inst(f"{operation} v{high + item}, {shift}, v{high_base + item}")
        for item in range(4):
            asm.inst(f"v_and_b32 v{payload + item}, 0x03030303, v{payload + item}")
        for item in range(4):
            asm.inst(
                f"v_and_or_b32 v{payload + item}, v{high + item}, "
                f"0x04040404, v{payload + item}"
            )
        for item in range(4):
            asm.inst(f"v_add_nc_u32 v{payload + item}, 0x7c7c7c7c, v{payload + item}")
        for item in range(4):
            asm.inst(f"v_xor_b32 v{payload + item}, 0x80808080, v{payload + item}")

    def _emit_scale_decode_ready_frontier(
        self,
        asm: Assembly,
        layout: Q3FullWeightTiledLdsLayout,
        *,
        half: int,
        metadata_load_offset: int,
        scale_plane_offset: int,
    ) -> None:
        registers = self._registers
        semantics = self.context.state.semantics
        scales = registers.decode_scale_frontier.first_register
        auxiliary = registers.decode_scale_auxiliary_frontier.first_register
        fields: list[
            tuple[Q3PackedFieldPart, Q3PackedFieldPart, int, int, int, int]
        ] = []
        for local_group in range(4):
            low_field, high_field = semantics.q3_scale_fields(
                8 * half + 2 * local_group
            )
            low_byte = scale_plane_offset - metadata_load_offset + low_field.source_byte
            high_byte = (
                scale_plane_offset - metadata_load_offset + high_field.source_byte
            )
            fields.append(
                (
                    low_field,
                    high_field,
                    low_byte // 4,
                    high_byte // 4,
                    8 * (low_byte % 4) + low_field.bit_offset,
                    8 * (high_byte % 4) + high_field.bit_offset,
                )
            )
        for item, (*_, low_bit, _high_bit) in enumerate(fields):
            asm.inst(
                f"v_add_nc_u32 v{auxiliary + item}, {low_bit}, "
                f"v{registers.half_shift.first_register}"
            )
        for item, (_low, _high, low_word, _high_word, _low_bit, _high_bit) in enumerate(
            fields
        ):
            asm.inst(
                f"v_lshrrev_b32 v{scales + item}, v{auxiliary + item}, "
                f"v{registers.weight_metadata.first_register + low_word}"
            )
        for item, (*_, high_bit) in enumerate(fields):
            asm.inst(
                f"v_add_nc_u32 v{auxiliary + item}, {high_bit}, "
                f"v{registers.half_shift.first_register}"
            )
        for item, (low_field, *_rest) in enumerate(fields):
            asm.inst(
                f"v_and_b32 v{scales + item}, {(1 << low_field.bit_count) - 1}, "
                f"v{scales + item}"
            )
        for item, (_low, _high, _low_word, high_word, _low_bit, _high_bit) in enumerate(
            fields
        ):
            asm.inst(
                f"v_lshrrev_b32 v{auxiliary + item}, v{auxiliary + item}, "
                f"v{registers.weight_metadata.first_register + high_word}"
            )
        for item, (_low, high_field, *_rest) in enumerate(fields):
            asm.inst(
                f"v_and_b32 v{auxiliary + item}, "
                f"{(1 << high_field.bit_count) - 1}, v{auxiliary + item}"
            )
        for item, (_low, high_field, *_rest) in enumerate(fields):
            asm.inst(
                f"v_lshl_or_b32 v{scales + item}, v{auxiliary + item}, "
                f"{high_field.destination_shift}, v{scales + item}"
            )
        for item in range(4):
            asm.inst(f"v_sub_nc_u32 v{scales + item}, v{scales + item}, 32")
        for item in range(4):
            asm.inst(f"v_cvt_f32_i32 v{scales + item}, v{scales + item}")
        for item in range(4):
            asm.inst(
                f"v_mul_f32 v{scales + item}, v{registers.decode_d.first_register}, "
                f"v{scales + item}"
            )
        for item in range(4):
            asm.inst(
                f"ds_write_b32 v{registers.weight_stage_address.first_register}, "
                f"v{scales + item} offset:{layout.weight_scale_offset + 8 * item}"
            )

    def _emit_serial_scale_decode(
        self,
        asm: Assembly,
        layout: Q3FullWeightTiledLdsLayout,
        *,
        half: int,
        metadata_load_offset: int,
        scale_plane_offset: int,
    ) -> None:
        registers = self._registers
        semantics = self.context.state.semantics
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
            asm.inst(
                f"v_add_nc_u32 v{registers.decode_auxiliary.first_register}, {low_bit}, v{registers.half_shift.first_register}"
            )
            asm.inst(
                f"v_lshrrev_b32 v{registers.decode_scale.first_register}, v{registers.decode_auxiliary.first_register}, "
                f"v{registers.weight_metadata.first_register + low_word}"
            )
            asm.inst(
                f"v_and_b32 v{registers.decode_scale.first_register}, "
                f"{(1 << low_field.bit_count) - 1}, v{registers.decode_scale.first_register}"
            )
            asm.inst(
                f"v_add_nc_u32 v{registers.decode_auxiliary.first_register}, {high_bit}, v{registers.half_shift.first_register}"
            )
            asm.inst(
                f"v_lshrrev_b32 v{registers.decode_auxiliary.first_register}, v{registers.decode_auxiliary.first_register}, "
                f"v{registers.weight_metadata.first_register + high_word}"
            )
            asm.inst(
                f"v_and_b32 v{registers.decode_auxiliary.first_register}, "
                f"{(1 << high_field.bit_count) - 1}, v{registers.decode_auxiliary.first_register}"
            )
            asm.inst(
                f"v_lshl_or_b32 v{registers.decode_scale.first_register}, v{registers.decode_auxiliary.first_register}, "
                f"{high_field.destination_shift}, v{registers.decode_scale.first_register}"
            )
            asm.inst(
                f"v_sub_nc_u32 v{registers.decode_scale.first_register}, v{registers.decode_scale.first_register}, 32"
            )
            asm.inst(
                f"v_cvt_f32_i32 v{registers.decode_scale.first_register}, v{registers.decode_scale.first_register}"
            )
            asm.inst(
                f"v_mul_f32 v{registers.decode_scale.first_register}, v{registers.decode_d.first_register}, "
                f"v{registers.decode_scale.first_register}"
            )
            asm.inst(
                f"ds_write_b32 v{registers.weight_stage_address.first_register}, "
                f"v{registers.decode_scale.first_register} offset:{layout.weight_scale_offset + 8 * local_group}"
            )

    def _emit_compute_half(
        self,
        asm: Assembly,
        layout: Q3FullWeightTiledLdsLayout,
        *,
        half: int,
    ) -> None:
        registers = self._registers
        waits_even = (15, 14, 5, 4, 3, 2, 1, 0)
        waits_odd = (11, 10, 5, 4, 3, 2, 1, 0)
        pair_order = ((0, 3), (1, 2), (4, 7), (5, 6))
        for group in range(8):
            group_spec = self.context.state.semantics.q3_payload_group(group)
            asm.comment(f"Q3_K half group {group}: signed WMMA and FP32 correction.")
            asm.inst(
                f"ds_read_b128 v[{registers.activation_stage.first_register}:{registers.activation_stage.first_register + 3}], "
                f"v{registers.activation_read_address.first_register} "
                f"offset:{group_spec.activation_payload_offset}"
            )
            asm.inst(
                f"ds_read_b128 v[{registers.weight_payload.first_register}:{registers.weight_payload.first_register + 3}], "
                f"v{registers.weight_lds_address.first_register} offset:{layout.half_payload_stride * half + 16 * group}"
            )
            asm.inst(
                f"ds_read_b128 v[{registers.activation_stage.first_register + 4}:{registers.activation_stage.first_register + 7}], "
                f"v{registers.activation_read_address.first_register} "
                f"offset:{16 * layout.activation_row_stride + group_spec.activation_payload_offset}"
            )
            weight_offset0 = 32 + 40 * half + group
            weight_offset1 = 200 + 40 * half + group
            for index in range(4):
                base = registers.weight_scale_address.first_register + index
                destination = registers.weight_scales.first_register + 2 * index
                asm.inst(
                    f"ds_read2_b32 v[{destination}:{destination + 1}], v{base} "
                    f"offset0:{weight_offset0} offset1:{weight_offset1}"
                )
            if group % 2 == 0:
                asm.inst(
                    f"v_add_nc_u32 v{registers.temporary.first_register}, {4 * (group // 2)}, "
                    f"v{registers.activation_read_address.first_register}"
                )
                for index, (offset0, offset1) in enumerate(
                    ((0, 9), (18, 27), (36, 45), (54, 63))
                ):
                    destination = registers.activation_scale.first_register + 2 * index
                    asm.inst(
                        f"ds_read2st64_b32 v[{destination}:{destination + 1}], "
                        f"v{registers.temporary.first_register} offset0:{offset0} offset1:{offset1}"
                    )
            for m_index in range(2, 8):
                payload = registers.activation_stage.first_register + 4 * m_index
                asm.inst(
                    f"ds_read_b128 v[{payload}:{payload + 3}], "
                    f"v{registers.activation_read_address.first_register} "
                    f"offset:{16 * m_index * layout.activation_row_stride + group_spec.activation_payload_offset}"
                )
            for wait, m_index in zip(
                waits_even if group % 2 == 0 else waits_odd,
                range(8),
            ):
                asm.inst(f"s_waitcnt lgkmcnt({wait})")
                emit_signed_i8_wmma(
                    asm,
                    destination=registers.c.first_register + 8 * m_index,
                    weight=registers.weight_payload.first_register,
                    activation=registers.activation_stage.first_register + 4 * m_index,
                    accumulator=registers.zero_accumulator.first_register,
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
        registers = self._registers
        for fragment in (first, second):
            base = registers.c.first_register + 8 * fragment
            for element in range(8):
                asm.inst(f"v_cvt_f32_i32 v{base + element}, v{base + element}")
            for element in range(0, 8, 2):
                asm.inst(
                    f"v_dual_mul_f32 v{base + element}, "
                    f"v{registers.weight_scales.first_register + element}, v{base + element} :: "
                    f"v_dual_mul_f32 v{base + element + 1}, "
                    f"v{registers.weight_scales.first_register + element + 1}, "
                    f"v{base + element + 1}"
                )
        first_sum = registers.sums.first_register + 8 * first
        second_sum = registers.sums.first_register + 8 * second
        first_scale = registers.activation_scale.first_register + first
        second_scale = registers.activation_scale.first_register + second
        second_base = registers.c.first_register + 8 * second
        for element in range(8):
            asm.inst(
                f"v_dual_fmac_f32 v{first_sum + element}, v{first_scale}, "
                f"v{registers.c.first_register + 8 * first + element} :: "
                f"v_dual_fmac_f32 v{second_sum + (element ^ 3)}, v{second_scale}, "
                f"v{second_base + (element ^ 3)}"
            )

    def _emit_store(self, asm: Assembly) -> None:
        registers = self._registers
        size = self.context.state.problem_size
        asm.comment("Store the Q3 wave-N fragments as row-major BF16.")
        asm.inst(
            f"v_and_b32 v{registers.temporary.first_register}, 15, v{registers.lane.first_register}"
        )
        asm.inst(f"v_lshlrev_b32 v{registers.temporary.first_register + 1}, 7, s3")
        asm.inst(
            f"v_add_nc_u32 v{registers.temporary.first_register}, v{registers.temporary.first_register + 1}, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{registers.output_address.first_register}, {2 * size.n}, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_bfe_u32 v{registers.temporary.first_register}, v{registers.lane.first_register}, 4, 1"
        )
        asm.inst(
            f"v_lshlrev_b32 v{registers.temporary.first_register + 1}, 4, v{registers.wave.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.temporary.first_register}, v{registers.temporary.first_register + 1}, v{registers.temporary.first_register}"
        )
        asm.inst(f"v_lshlrev_b32 v{registers.temporary.first_register + 1}, 6, s2")
        asm.inst(
            f"v_add_nc_u32 v{registers.temporary.first_register}, v{registers.temporary.first_register + 1}, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_lshlrev_b32 v{registers.temporary.first_register}, 1, v{registers.temporary.first_register}"
        )
        asm.inst(
            f"v_add_nc_u32 v{registers.output_address.first_register}, v{registers.temporary.first_register}, "
            f"v{registers.output_address.first_register}"
        )
        for m_index in range(8):
            fragment = registers.sums.first_register + 8 * m_index
            for element in range(8):
                emit_bf16_rne(
                    asm, fragment + element, registers.temporary.first_register
                )
            asm.inst("s_clause 7")
            for element in range(8):
                asm.inst(
                    f"global_store_d16_hi_b16 v{registers.output_address.first_register}, "
                    f"v{fragment + element}, s[8:9] offset:{4 * element}"
                )
            if m_index != 7:
                asm.inst(
                    f"v_add_nc_u32 v{registers.output_address.first_register}, {32 * size.n}, "
                    f"v{registers.output_address.first_register}"
                )
