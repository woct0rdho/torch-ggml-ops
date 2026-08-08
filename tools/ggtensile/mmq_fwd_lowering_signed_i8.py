"""Signed-int8 direct, register-tiled, and tiled-LDS forward lowerings."""

from dataclasses import dataclass
from typing import ClassVar, cast

from .kernel_writer_assembly import (
    Assembly,
    RegisterAssignment,
    emit_bf16_rne,
    emit_kernel_trailer,
    emit_pointer_kernarg_loads,
)
from .mmq_fwd_lowering import ForwardLoweringContext
from .mmq_fwd_lowering_mma import emit_signed_i8_wmma
from .mmq_fwd_physical import (
    SignedInt8DirectPhysicalPlan,
    SignedInt8DirectRegisterPlan,
    SignedInt8MmaGroupRole,
    SignedInt8RegisterTiledPhysicalPlan,
    SignedInt8RegisterTiledRegisterPlan,
    SignedInt8RegisterTileRole,
    SignedInt8SmallMTiledLdsPhysicalPlan,
    SignedInt8TiledLdsRegisters,
    SignedInt8TiledLdsScaleLayout,
    SignedInt8WaveNTiledLdsPhysicalPlan,
)
from .model import ProblemSize


@dataclass(frozen=True)
class SignedInt8ForwardLowering:
    """Emit one validated signed-int8 mechanism without exact-key selection."""

    context: ForwardLoweringContext

    KERNARG: ClassVar[int] = 4
    LOOP_COUNTER: ClassVar[int] = 10
    OPERAND_SOURCES: ClassVar[frozenset[str]] = frozenset(
        {
            "Q8DirectGlobal",
            "Q8RegisterTiled",
            "Q8HipTiledLds",
            "Q8SmallMTiledLds",
        }
    )

    def body(self) -> str:
        physical = self.context.state.physical_plan
        if isinstance(physical, SignedInt8DirectPhysicalPlan):
            return self._body_signed_int8_direct_global()
        if isinstance(physical, SignedInt8RegisterTiledPhysicalPlan):
            return self._body_signed_int8_register_tiled()
        if isinstance(physical, SignedInt8WaveNTiledLdsPhysicalPlan):
            return self._body_signed_int8_wave_n_tiled_lds()
        if isinstance(physical, SignedInt8SmallMTiledLdsPhysicalPlan):
            return self._body_signed_int8_small_m_tiled_lds()
        raise TypeError(f"unsupported Q8 physical plan {type(physical).__name__}")

    def _body_signed_int8_direct_global(self) -> str:
        """Lower the isolated one-wave signed-int8 direct-global control."""
        asm = Assembly()
        physical = cast(SignedInt8DirectPhysicalPlan, self.context.state.physical_plan)
        registers = physical.registers
        name = self.context.solution_key.kernel_name
        row_stride = self.context.state.packed_weight_row_bytes
        activation_plane_stride = self.context.state.activation_plane_stride_bytes
        sums = registers.sums.first_register
        result_addresses = registers.result_addresses.first_register
        temporary = registers.temporary.first_register
        output_column = registers.output_column.first_register
        serial = registers.serial.first_register
        activation_row = registers.activation_row.first_register

        asm.comment("Load packed Q8_0, the Q8_1 F32_D4 workspace, and output pointers.")
        emit_pointer_kernarg_loads(asm, self.KERNARG)

        asm.comment("Map one wave to an exact 16x16 output tile.")
        asm.inst(f"v_mov_b32 v{serial}, v0")
        asm.inst(f"v_and_b32 v{output_column}, 15, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 4, s2")
        asm.inst(f"v_add_nc_u32 v{output_column}, v{temporary}, v{output_column}")
        asm.inst(f"v_and_b32 v{activation_row}, 15, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 4, s3")
        asm.inst(f"v_add_nc_u32 v{activation_row}, v{temporary}, v{activation_row}")

        asm.comment("Build Q8 payload, scale, and activation row addresses.")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{registers.activation_scale.first_register}, 4, s2")
        asm.inst(
            f"v_add_nc_u32 v{temporary}, "
            f"v{registers.activation_scale.first_register}, v{temporary}"
        )
        asm.inst(f"v_mul_lo_u32 v{result_addresses}, {row_stride}, v{temporary}")
        for element in range(1, 8):
            asm.inst(
                f"v_add_nc_u32 v{result_addresses + element}, "
                f"{2 * element * row_stride}, v{result_addresses}"
            )
        asm.inst(
            f"v_mul_lo_u32 v{registers.weight_address.first_register}, "
            f"{row_stride}, v{output_column}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{registers.activation_address.first_register}, "
            f"{self.context.state.contract.activation_block_bytes}, v{activation_row}"
        )
        for register in range(sums, sums + 8):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")

        asm.label(".LForwardQ80ActivationBlockLoop")
        for group in range(4):
            self._emit_signed_int8_direct_group(
                asm,
                SignedInt8MmaGroupRole.from_semantics(
                    self.context.state.semantics, group
                ),
                registers,
            )
            for element in range(8):
                asm.inst(
                    f"v_add_nc_u32 v{result_addresses + element}, "
                    f"{self.context.state.contract.packed_weight_block_bytes}, "
                    f"v{result_addresses + element}"
                )
            asm.inst(
                f"v_add_nc_u32 v{registers.weight_address.first_register}, "
                f"{self.context.state.contract.packed_weight_block_bytes}, "
                f"v{registers.weight_address.first_register}"
            )
        asm.inst(
            f"v_add_nc_u32 v{registers.activation_address.first_register}, "
            f"{activation_plane_stride}, "
            f"v{registers.activation_address.first_register}"
        )
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(
            f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {self.context.state.activation_blocks_per_row}"
        )
        asm.inst("s_cbranch_scc1 .LForwardQ80ActivationBlockLoop")

        self._emit_signed_int8_direct_store(asm, registers)
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_signed_int8_direct_group(
        self,
        asm: Assembly,
        group: SignedInt8MmaGroupRole,
        registers: SignedInt8DirectRegisterPlan,
    ) -> None:
        c = registers.c.first_register
        sums = registers.sums.first_register
        weight_payload = registers.weight_payload.first_register
        activation_payload = registers.activation_payload.first_register
        weight_scales = registers.weight_scales.first_register
        activation_scale = registers.activation_scale.first_register
        result_addresses = registers.result_addresses.first_register
        weight_address = registers.weight_address.first_register
        activation_address = registers.activation_address.first_register

        asm.comment(
            f"Q8_0 group {group.index}: signed payload and FP32 scale correction."
        )
        asm.inst(
            f"global_load_b128 v[{weight_payload}:{weight_payload + 3}], "
            f"v{weight_address}, s[{self.KERNARG}:{self.KERNARG + 1}] "
            f"offset:{group.weight_payload_offset}"
        )
        asm.inst(
            f"global_load_b128 v[{weight_payload + 4}:{weight_payload + 7}], "
            f"v{weight_address}, s[{self.KERNARG}:{self.KERNARG + 1}] "
            f"offset:{group.weight_payload_offset + 16}"
        )
        asm.inst(
            f"global_load_b128 v[{activation_payload}:{activation_payload + 3}], "
            f"v{activation_address}, s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
            f"offset:{group.activation_payload_offset}"
        )
        asm.inst(
            f"global_load_b128 v[{activation_payload + 4}:"
            f"{activation_payload + 7}], v{activation_address}, "
            f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
            f"offset:{group.activation_payload_offset + 16}"
        )
        asm.inst(
            f"global_load_b32 v{activation_scale}, v{activation_address}, "
            f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
            f"offset:{group.activation_scale_offset}"
        )
        for element in range(8):
            asm.inst(
                f"global_load_d16_b16 v{weight_scales + element}, "
                f"v{result_addresses + element}, "
                f"s[{self.KERNARG}:{self.KERNARG + 1}]"
            )
        asm.inst("s_waitcnt vmcnt(0)")

        for register in range(c, c + 8):
            asm.inst(f"v_mov_b32 v{register}, 0")
        emit_signed_i8_wmma(
            asm,
            destination=c,
            weight=weight_payload,
            activation=activation_payload,
            accumulator=c,
            clamp=False,
        )
        emit_signed_i8_wmma(
            asm,
            destination=c,
            weight=weight_payload + 4,
            activation=activation_payload + 4,
            accumulator=c,
            clamp=False,
        )
        for element in range(8):
            asm.inst(
                f"v_cvt_f32_f16 v{weight_scales + element}, v{weight_scales + element}"
            )
            asm.inst(f"v_cvt_f32_i32 v{c + element}, v{c + element}")
            asm.inst(
                f"v_mul_f32 v{c + element}, v{weight_scales + element}, v{c + element}"
            )
            asm.inst(
                f"v_fmac_f32 v{sums + element}, v{activation_scale}, v{c + element}"
            )

    def _emit_signed_int8_direct_store(
        self,
        asm: Assembly,
        registers: SignedInt8DirectRegisterPlan,
    ) -> None:
        size = self.context.state.problem_size
        sums = registers.sums.first_register
        output_address = registers.output_address.first_register
        temporary = registers.temporary.first_register
        store_auxiliary = registers.store_auxiliary.first_register
        serial = registers.serial.first_register

        asm.comment("Store the Q8 direct J-major fragments as row-major BF16.")
        asm.inst(f"v_and_b32 v{temporary}, 15, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{output_address}, 4, s3")
        asm.inst(f"v_add_nc_u32 v{output_address}, v{output_address}, v{temporary}")
        asm.inst(f"v_mul_lo_u32 v{output_address}, {2 * size.n}, v{output_address}")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{store_auxiliary}, 4, s2")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{store_auxiliary}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{output_address}, v{output_address}, v{temporary}")
        for element in range(8):
            total = sums + element
            emit_bf16_rne(asm, total, temporary)
            asm.inst(
                f"global_store_d16_hi_b16 v{output_address}, v{total}, "
                f"s[{self.KERNARG + 4}:{self.KERNARG + 5}]"
            )
            if element != 7:
                asm.inst(f"v_add_nc_u32 v{output_address}, 4, v{output_address}")

    def _body_signed_int8_register_tiled(self) -> str:
        """Lower four signed-int8 fragments per wave in a four-wave group."""
        asm = Assembly()
        wave_tile_m, wave_tile_n = self.context.state.mi_wave_tile
        physical = cast(
            SignedInt8RegisterTiledPhysicalPlan, self.context.state.physical_plan
        )
        registers = physical.registers
        name = self.context.solution_key.kernel_name
        row_stride = self.context.state.packed_weight_row_bytes
        activation_plane_stride = self.context.state.activation_plane_stride_bytes
        sums = registers.sums.first_register
        result_addresses = registers.result_addresses.first_register
        weight_addresses = registers.weight_addresses.first_register
        activation_addresses = registers.activation_addresses.first_register
        temporary = registers.temporary.first_register
        lane = registers.lane.first_register
        wave = registers.wave.first_register
        output_column = registers.output_column.first_register
        activation_row = registers.activation_row.first_register

        asm.comment("Load the Q8_0, Q8_1 F32_D4, and output pointers.")
        emit_pointer_kernarg_loads(asm, self.KERNARG)
        asm.comment("Flatten gfx11 packed workitem X/Y and retain wave ownership.")
        asm.inst(f"v_bfe_u32 v{wave}, v0, 10, 10")
        asm.inst(f"v_and_b32 v{lane}, 0x3ff, v0")

        asm.comment("Build two output-column weight and scale address groups.")
        for n_index in range(wave_tile_n):
            asm.inst(f"v_and_b32 v{output_column}, 15, v{lane}")
            asm.inst(
                f"v_lshlrev_b32 v{temporary}, "
                f"{self.context.solution_key.solution.macro_tile1.bit_length() - 1}, s2"
            )
            asm.inst(f"v_add_nc_u32 v{output_column}, v{temporary}, v{output_column}")
            if n_index:
                asm.inst(
                    f"v_add_nc_u32 v{output_column}, {16 * n_index}, v{output_column}"
                )
            asm.inst(
                f"v_mul_lo_u32 v{weight_addresses + n_index}, {row_stride}, "
                f"v{output_column}"
            )

            asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{lane}")
            asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
            asm.inst(
                f"v_lshlrev_b32 v{output_column}, "
                f"{self.context.solution_key.solution.macro_tile1.bit_length() - 1}, s2"
            )
            asm.inst(f"v_add_nc_u32 v{temporary}, v{output_column}, v{temporary}")
            if n_index:
                asm.inst(f"v_add_nc_u32 v{temporary}, {16 * n_index}, v{temporary}")
            address_base = result_addresses + 8 * n_index
            asm.inst(f"v_mul_lo_u32 v{address_base}, {row_stride}, v{temporary}")
            for element in range(1, 8):
                asm.inst(
                    f"v_add_nc_u32 v{address_base + element}, "
                    f"{2 * element * row_stride}, v{address_base}"
                )

        asm.comment("Build two activation-row addresses owned by each wave.")
        for m_index in range(wave_tile_m):
            asm.inst(f"v_and_b32 v{activation_row}, 15, v{lane}")
            asm.inst(
                f"v_lshlrev_b32 v{temporary}, "
                f"{self.context.solution_key.solution.macro_tile0.bit_length() - 1}, s3"
            )
            asm.inst(f"v_add_nc_u32 v{activation_row}, v{temporary}, v{activation_row}")
            asm.inst(
                f"v_lshlrev_b32 v{temporary}, "
                f"{(16 * wave_tile_m).bit_length() - 1}, v{wave}"
            )
            asm.inst(f"v_add_nc_u32 v{activation_row}, v{temporary}, v{activation_row}")
            if m_index:
                asm.inst(
                    f"v_add_nc_u32 v{activation_row}, {16 * m_index}, v{activation_row}"
                )
            asm.inst(
                f"v_mul_lo_u32 v{activation_addresses + m_index}, "
                f"{self.context.state.contract.activation_block_bytes}, v{activation_row}"
            )

        for register in range(sums, sums + 32):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")

        asm.label(".LForwardQ80RegisterTiledActivationBlockLoop")
        for group in range(4):
            group_role = SignedInt8MmaGroupRole.from_semantics(
                self.context.state.semantics, group
            )
            self._emit_signed_int8_register_tiled_group(asm, group_role, registers)
        packed_activation_block_bytes = (
            4 * self.context.state.contract.packed_weight_block_bytes
        )
        for element in range(8 * wave_tile_n):
            asm.inst(
                f"v_add_nc_u32 v{result_addresses + element}, "
                f"{packed_activation_block_bytes}, v{result_addresses + element}"
            )
        for element in range(wave_tile_n):
            asm.inst(
                f"v_add_nc_u32 v{weight_addresses + element}, "
                f"{packed_activation_block_bytes}, v{weight_addresses + element}"
            )
        for element in range(wave_tile_m):
            asm.inst(
                f"v_add_nc_u32 v{activation_addresses + element}, "
                f"{activation_plane_stride}, v{activation_addresses + element}"
            )
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(
            f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {self.context.state.activation_blocks_per_row}"
        )
        asm.inst("s_cbranch_scc1 .LForwardQ80RegisterTiledActivationBlockLoop")

        self._emit_signed_int8_register_tiled_store(asm, registers)
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_signed_int8_register_tiled_group(
        self,
        asm: Assembly,
        group: SignedInt8MmaGroupRole,
        registers: SignedInt8RegisterTiledRegisterPlan,
    ) -> None:
        wave_tile_m, wave_tile_n = self.context.state.mi_wave_tile
        weight_payloads = registers.weight_payloads.first_register
        activation_payloads = registers.activation_payloads.first_register
        weight_scales = registers.weight_scales.first_register
        activation_scales = registers.activation_scales.first_register
        result_addresses = registers.result_addresses.first_register
        weight_addresses = registers.weight_addresses.first_register
        activation_addresses = registers.activation_addresses.first_register

        asm.comment(f"Q8_0 group {group.index}: load register-tile operand fragments.")
        asm.inst(f"s_clause {10 * wave_tile_n + 3 * wave_tile_m - 1}")
        for n_index in range(wave_tile_n):
            payload = weight_payloads + 8 * n_index
            address = weight_addresses + n_index
            asm.inst(
                f"global_load_b128 v[{payload}:{payload + 3}], v{address}, "
                f"s[{self.KERNARG}:{self.KERNARG + 1}] "
                f"offset:{group.weight_block_offset + group.weight_payload_offset}"
            )
            asm.inst(
                f"global_load_b128 v[{payload + 4}:{payload + 7}], v{address}, "
                f"s[{self.KERNARG}:{self.KERNARG + 1}] "
                f"offset:{group.weight_block_offset + group.weight_payload_offset + 16}"
            )
        for m_index in range(wave_tile_m):
            payload = activation_payloads + 8 * m_index
            address = activation_addresses + m_index
            asm.inst(
                f"global_load_b128 v[{payload}:{payload + 3}], v{address}, "
                f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                f"offset:{group.activation_payload_offset}"
            )
            asm.inst(
                f"global_load_b128 v[{payload + 4}:{payload + 7}], v{address}, "
                f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                f"offset:{group.activation_payload_offset + 16}"
            )
            asm.inst(
                f"global_load_b32 v{activation_scales + m_index}, v{address}, "
                f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                f"offset:{group.activation_scale_offset}"
            )
        for element in range(8 * wave_tile_n):
            asm.inst(
                f"global_load_d16_b16 v{weight_scales + element}, "
                f"v{result_addresses + element}, "
                f"s[{self.KERNARG}:{self.KERNARG + 1}] "
                f"offset:{group.weight_block_offset}"
            )
        asm.inst("s_waitcnt vmcnt(0)")
        self._emit_signed_int8_register_tile_dot(asm, registers)

    def _emit_signed_int8_register_tile_dot(
        self,
        asm: Assembly,
        registers: SignedInt8RegisterTiledRegisterPlan,
    ) -> None:
        wave_tile_m, wave_tile_n = self.context.state.mi_wave_tile
        c = registers.c.first_register
        sums = registers.sums.first_register
        weight_payloads = registers.weight_payloads.first_register
        activation_payloads = registers.activation_payloads.first_register
        weight_scales = registers.weight_scales.first_register
        activation_scales = registers.activation_scales.first_register
        for element in range(8 * wave_tile_n):
            asm.inst(
                f"v_cvt_f32_f16 v{weight_scales + element}, v{weight_scales + element}"
            )

        for tile in SignedInt8RegisterTileRole.all(wave_tile_m, wave_tile_n):
            fragment = 8 * tile.fragment_index
            c_fragment = c + fragment
            sum_fragment = sums + fragment
            weight_payload = weight_payloads + 8 * tile.n_index
            activation_payload = activation_payloads + 8 * tile.m_index
            weight_scale = weight_scales + 8 * tile.n_index
            activation_scale = activation_scales + tile.m_index
            for register in range(c_fragment, c_fragment + 8):
                asm.inst(f"v_mov_b32 v{register}, 0")
            emit_signed_i8_wmma(
                asm,
                destination=c_fragment,
                weight=weight_payload,
                activation=activation_payload,
                accumulator=c_fragment,
                clamp=False,
            )
            emit_signed_i8_wmma(
                asm,
                destination=c_fragment,
                weight=weight_payload + 4,
                activation=activation_payload + 4,
                accumulator=c_fragment,
                clamp=False,
            )
            for element in range(8):
                asm.inst(
                    f"v_cvt_f32_i32 v{c_fragment + element}, v{c_fragment + element}"
                )
                asm.inst(
                    f"v_mul_f32 v{c_fragment + element}, "
                    f"v{weight_scale + element}, v{c_fragment + element}"
                )
                asm.inst(
                    f"v_fmac_f32 v{sum_fragment + element}, "
                    f"v{activation_scale}, v{c_fragment + element}"
                )

    def _emit_signed_int8_register_tiled_store(
        self,
        asm: Assembly,
        registers: SignedInt8RegisterTiledRegisterPlan,
    ) -> None:
        wave_tile_m, wave_tile_n = self.context.state.mi_wave_tile
        size = self.context.state.problem_size
        sums = registers.sums.first_register
        temporary = registers.temporary.first_register
        output_address = registers.output_address.first_register
        lane = registers.lane.first_register
        wave = registers.wave.first_register
        store_column = registers.store_column.first_register
        store_row = registers.store_row.first_register

        asm.comment("Store four 16x16 J-major fragments as row-major BF16.")
        asm.inst(f"v_and_b32 v{store_row}, 15, v{lane}")
        asm.inst(
            f"v_lshlrev_b32 v{temporary}, "
            f"{self.context.solution_key.solution.macro_tile0.bit_length() - 1}, s3"
        )
        asm.inst(f"v_add_nc_u32 v{store_row}, v{temporary}, v{store_row}")
        asm.inst(
            f"v_lshlrev_b32 v{temporary}, "
            f"{(16 * wave_tile_m).bit_length() - 1}, v{wave}"
        )
        asm.inst(f"v_add_nc_u32 v{store_row}, v{temporary}, v{store_row}")
        asm.inst(f"v_mul_lo_u32 v{output_address}, {2 * size.n}, v{store_row}")

        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{lane}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(
            f"v_lshlrev_b32 v{store_column}, "
            f"{self.context.solution_key.solution.macro_tile1.bit_length() - 1}, s2"
        )
        asm.inst(f"v_add_nc_u32 v{store_column}, v{store_column}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{store_column}, 1, v{store_column}")
        asm.inst(f"v_add_nc_u32 v{output_address}, v{output_address}, v{store_column}")

        for tile_index, tile in enumerate(
            SignedInt8RegisterTileRole.all(wave_tile_m, wave_tile_n)
        ):
            address_base = output_address + 8 * tile_index
            if tile_index:
                address_delta = (
                    32 if tile.n_index else 32 * size.n - 32 * (wave_tile_n - 1)
                )
                asm.inst(
                    f"v_add_nc_u32 v{address_base}, {address_delta}, "
                    f"v{address_base - 8}"
                )

            for element in range(1, 8):
                asm.inst(
                    f"v_add_nc_u32 v{address_base + element}, 4, "
                    f"v{address_base + element - 1}"
                )

            fragment = sums + 8 * tile.fragment_index
            for element in range(8):
                total = fragment + element
                emit_bf16_rne(asm, total, temporary)
        asm.inst("s_clause 31")
        for tile in SignedInt8RegisterTileRole.all(wave_tile_m, wave_tile_n):
            address_base = output_address + 8 * tile.fragment_index
            fragment = sums + 8 * tile.fragment_index
            for element in range(8):
                total = fragment + element
                asm.inst(
                    f"global_store_d16_hi_b16 v{address_base + element}, v{total}, "
                    f"s[{self.KERNARG + 4}:{self.KERNARG + 5}]"
                )

    def _body_signed_int8_small_m_tiled_lds(self) -> str:
        """Lower an exact M32/M64 wave-N Q8 tile with split-row staging."""
        asm = Assembly()
        macro_tile_m = self.context.state.kernel_spec.macro_tile[0]
        m_fragments = macro_tile_m // 16
        activation_row_share = 128 // macro_tile_m
        groups_per_lane = 4 // activation_row_share
        physical = cast(
            SignedInt8SmallMTiledLdsPhysicalPlan, self.context.state.physical_plan
        )
        registers = physical.registers
        name = self.context.solution_key.kernel_name
        size = self.context.state.problem_size
        row_stride = self.context.state.packed_weight_row_bytes
        activation_plane_stride = self.context.state.activation_plane_stride_bytes
        iteration_count = self.context.state.activation_blocks_per_row
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
        tiled_registers = SignedInt8TiledLdsRegisters.from_plan(registers)
        tiled_scale_layout = SignedInt8TiledLdsScaleLayout.from_layout(layout)
        activation_lds_row_stride = layout.activation_row_stride
        weight_lds_base = layout.weight_base
        weight_lds_row_stride = layout.weight_row_stride
        weight_scale_pair_base_delta = layout.weight_scale_pair_base_delta

        asm.comment("Load pointers for an exact small-M wave-N Q8 tile.")
        emit_pointer_kernarg_loads(asm, self.KERNARG)
        asm.comment("Map four waves to sixteen output-feature rows each.")
        asm.inst(f"v_bfe_u32 v{wave}, v0, 10, 10")
        asm.inst(f"v_and_b32 v{lane}, 0x3ff, v0")
        asm.inst(f"v_and_b32 v{temporary}, 15, v{lane}")
        asm.inst(f"v_lshlrev_b32 v{weight_row}, 4, v{wave}")
        asm.inst(f"v_add_nc_u32 v{weight_row}, v{weight_row}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary}, v{weight_row}")
        asm.inst(f"v_mul_lo_u32 v{weight_address}, {row_stride}, v{temporary}")

        asm.comment("Split each activation row across all 128 workitems.")
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
        asm.inst(
            f"v_mul_lo_u32 v{activation_address}, "
            f"{activation_lds_row_stride}, v{activation_row}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{temporary}, {activation_lds_row_stride * macro_tile_m}, s3"
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

        asm.label(".LForwardQ80SmallMTiledLdsBlockLoop")
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
        if weight_scale_pair_base_delta is not None:
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
                m_fragments=m_fragments,
            )
        asm.inst("s_barrier")
        asm.inst(
            f"v_add_nc_u32 v{weight_address}, "
            f"{4 * self.context.state.contract.packed_weight_block_bytes}, "
            f"v{weight_address}"
        )
        for address in (activation_address, activation_scale_stage_address):
            asm.inst(f"v_add_nc_u32 v{address}, {activation_plane_stride}, v{address}")
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {iteration_count}")
        asm.inst("s_cbranch_scc1 .LForwardQ80SmallMTiledLdsBlockLoop")

        self._emit_signed_int8_tiled_store(
            asm,
            tiled_registers,
            size,
            m_fragments=m_fragments,
        )
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _body_signed_int8_wave_n_tiled_lds(self) -> str:
        """Lower a wave-N 128x64 Q8 tile with cooperative LDS operands."""
        asm = Assembly()
        physical = cast(
            SignedInt8WaveNTiledLdsPhysicalPlan, self.context.state.physical_plan
        )
        registers = physical.registers
        layout = physical.layout
        tiled_registers = SignedInt8TiledLdsRegisters.from_plan(registers)
        tiled_scale_layout = SignedInt8TiledLdsScaleLayout.from_layout(layout)
        name = self.context.solution_key.kernel_name
        size = self.context.state.problem_size
        row_stride = self.context.state.packed_weight_row_bytes
        activation_plane_stride = self.context.state.activation_plane_stride_bytes
        depth_u = self.context.state.kernel_spec.geometry.depth_u
        groups_per_iteration = depth_u // 8
        activation_planes_per_iteration = groups_per_iteration // 4
        iteration_count = (
            self.context.state.activation_blocks_per_row
            // activation_planes_per_iteration
        )
        sums = registers.sums.first_register
        zero_accumulator = registers.zero_accumulator.first_register
        weight_scale_address = registers.weight_scale_address.first_register
        weight_address = registers.weight_address.first_register
        activation_address = registers.activation_address.first_register
        activation_lds_address = registers.activation_lds_address.first_register
        weight_lds_address = registers.weight_lds_address.first_register
        temporary = registers.temporary.first_register
        lane = registers.lane.first_register
        wave = registers.wave.first_register
        activation_row = registers.activation_row.first_register
        activation_read_address = registers.activation_read_address.first_register
        weight_row = registers.weight_row.first_register

        activation_lds_row_stride = layout.activation_row_stride
        weight_lds_base = layout.weight_base
        weight_lds_row_stride = layout.weight_row_stride
        compact_depth32 = layout.weight_scale_pair_base_delta is not None

        asm.comment("Load pointers for the HIP-shaped wave-N Q8 tile.")
        emit_pointer_kernarg_loads(asm, self.KERNARG)
        asm.comment("Map each wave to sixteen output-feature rows.")
        asm.inst(f"v_bfe_u32 v{wave}, v0, 10, 10")
        asm.inst(f"v_and_b32 v{lane}, 0x3ff, v0")
        asm.inst(f"v_and_b32 v{temporary}, 15, v{lane}")
        asm.inst(f"v_lshlrev_b32 v{weight_row}, 4, v{wave}")
        asm.inst(f"v_add_nc_u32 v{weight_row}, v{weight_row}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary}, v{weight_row}")
        asm.inst(f"v_mul_lo_u32 v{weight_address}, {row_stride}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{activation_row}, 4, v{wave}")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{lane}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{activation_row}, v{activation_row}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 5, v{wave}")
        asm.inst(f"v_add_nc_u32 v{activation_row}, v{temporary}, v{lane}")
        asm.inst(
            f"v_mul_lo_u32 v{activation_address}, {activation_lds_row_stride}, v{activation_row}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{temporary}, "
            f"{activation_lds_row_stride * self.context.state.kernel_spec.macro_tile[0]}, s3"
        )
        asm.inst(
            f"v_add_nc_u32 v{activation_address}, v{temporary}, v{activation_address}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{activation_lds_address}, "
            f"{activation_lds_row_stride}, v{activation_row}"
        )
        asm.comment("Hoist the lane-local activation LDS row base across all groups.")
        asm.inst(f"v_and_b32 v{activation_read_address}, 15, v{lane}")
        asm.inst(
            f"v_mul_lo_u32 v{activation_read_address}, "
            f"{activation_lds_row_stride}, v{activation_read_address}"
        )
        asm.inst(f"v_mul_lo_u32 v{temporary}, {weight_lds_row_stride}, v{weight_row}")
        asm.inst(f"v_add_nc_u32 v{weight_lds_address}, {weight_lds_base}, v{temporary}")

        for register in range(sums, sums + 64, 2):
            asm.inst(
                f"v_dual_mov_b32 v{register}, 0 :: v_dual_mov_b32 v{register + 1}, 0"
            )
        asm.comment("Keep one zero WMMA fragment live across all reduction loops.")
        for register in range(zero_accumulator, zero_accumulator + 8, 2):
            asm.inst(
                f"v_dual_mov_b32 v{register}, 0 :: v_dual_mov_b32 v{register + 1}, 0"
            )
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")

        asm.label(".LForwardQ80HipTiledLdsBlockLoop")
        if compact_depth32:
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
                registers.activation_address,
                4,
            )
            self._emit_signed_int8_tiled_weight_writes(
                asm,
                tiled_registers,
                group_base=0,
                wait_counts=(12, 0),
            )
            self._emit_signed_int8_tiled_activation_writes(
                asm,
                tiled_registers,
                registers.activation_lds_address,
                registers.activation_lds_address,
                4,
                trailing_vmem=0,
            )
        else:
            self._emit_signed_int8_tiled_activation_loads(
                asm,
                tiled_registers,
                registers.activation_address,
                registers.activation_address,
                4,
            )
            self._emit_signed_int8_tiled_weight_stage(
                asm,
                tiled_registers,
                tiled_scale_layout,
                group_base=0,
            )
            self._emit_signed_int8_tiled_activation_writes(
                asm,
                tiled_registers,
                registers.activation_lds_address,
                registers.activation_lds_address,
                4,
            )
            self._emit_signed_int8_tiled_weight_writes(
                asm,
                tiled_registers,
                group_base=0,
            )
        if groups_per_iteration == 8:
            self._emit_signed_int8_tiled_weight_stage(
                asm,
                tiled_registers,
                tiled_scale_layout,
                group_base=4,
            )
            self._emit_signed_int8_tiled_weight_writes(
                asm,
                tiled_registers,
                group_base=4,
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
        if compact_depth32:
            asm.inst(
                f"v_add_nc_u32 v{temporary}, "
                f"{layout.weight_scale_pair_base_delta}, v{weight_scale_address}"
            )
        for group in range(4):
            self._emit_signed_int8_tiled_group(
                asm,
                group,
                tiled_registers,
                tiled_scale_layout,
            )
        if groups_per_iteration == 8:
            asm.inst(
                f"v_add_nc_u32 v{temporary}, {activation_plane_stride}, "
                f"v{activation_address}"
            )
            self._emit_signed_int8_tiled_activation_loads(
                asm,
                tiled_registers,
                registers.temporary,
                registers.temporary,
                4,
            )
            asm.inst("s_barrier")
            self._emit_signed_int8_tiled_activation_writes(
                asm,
                tiled_registers,
                registers.activation_lds_address,
                registers.activation_lds_address,
                4,
                trailing_vmem=0,
            )
            asm.inst("s_waitcnt lgkmcnt(0)")
            asm.inst("s_barrier")
            for group in range(4, 8):
                self._emit_signed_int8_tiled_group(
                    asm,
                    group,
                    tiled_registers,
                    tiled_scale_layout,
                    activation_group=group - 4,
                )
        asm.inst("s_barrier")
        asm.inst(
            f"v_add_nc_u32 v{weight_address}, "
            f"{groups_per_iteration * self.context.state.contract.packed_weight_block_bytes}, "
            f"v{weight_address}"
        )
        asm.inst(
            f"v_add_nc_u32 v{activation_address}, "
            f"{activation_planes_per_iteration * activation_plane_stride}, "
            f"v{activation_address}"
        )
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {iteration_count}")
        asm.inst("s_cbranch_scc1 .LForwardQ80HipTiledLdsBlockLoop")

        self._emit_signed_int8_tiled_store(
            asm,
            tiled_registers,
            size,
        )
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_signed_int8_tiled_activation_loads(
        self,
        asm: Assembly,
        registers: SignedInt8TiledLdsRegisters,
        payload_address_role: RegisterAssignment,
        scale_address_role: RegisterAssignment,
        groups_per_lane: int,
    ) -> None:
        """Load one lane's formula-derived share of an activation stage."""
        payload_address = payload_address_role.first_register
        scale_address = scale_address_role.first_register
        activation_payloads = registers.activation_payloads.first_register
        activation_scales = registers.activation_scales.first_register
        asm.inst(f"s_clause {3 * groups_per_lane - 1}")
        for group in range(groups_per_lane):
            payload = activation_payloads + 8 * group
            asm.inst(
                f"global_load_b128 v[{payload}:{payload + 3}], "
                f"v{payload_address}, "
                f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                f"offset:{16 + 32 * group}"
            )
            asm.inst(
                f"global_load_b128 v[{payload + 4}:{payload + 7}], "
                f"v{payload_address}, "
                f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                f"offset:{32 + 32 * group}"
            )
            asm.inst(
                f"global_load_b32 v{activation_scales + group}, "
                f"v{scale_address}, "
                f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] offset:{4 * group}"
            )

    def _emit_signed_int8_tiled_activation_writes(
        self,
        asm: Assembly,
        registers: SignedInt8TiledLdsRegisters,
        payload_lds_address_role: RegisterAssignment,
        scale_lds_address_role: RegisterAssignment,
        groups_per_lane: int,
        *,
        trailing_vmem: int = 6,
    ) -> None:
        """Commit ready activation loads at their producer-derived waits."""
        payload_lds_address = payload_lds_address_role.first_register
        scale_lds_address = scale_lds_address_role.first_register
        activation_payloads = registers.activation_payloads.first_register
        activation_scales = registers.activation_scales.first_register
        for group in range(groups_per_lane):
            wait_count = 3 * (groups_per_lane - group - 1) + trailing_vmem
            asm.inst(f"s_waitcnt vmcnt({wait_count})")
            payload = activation_payloads + 8 * group
            asm.inst(
                f"ds_write_b128 v{payload_lds_address}, "
                f"v[{payload}:{payload + 3}] offset:{16 + 32 * group}"
            )
            asm.inst(
                f"ds_write_b128 v{payload_lds_address}, "
                f"v[{payload + 4}:{payload + 7}] offset:{32 + 32 * group}"
            )
            asm.inst(
                f"ds_write_b32 v{scale_lds_address}, "
                f"v{activation_scales + group} offset:{4 * group}"
            )

    def _emit_signed_int8_tiled_weight_stage(
        self,
        asm: Assembly,
        registers: SignedInt8TiledLdsRegisters,
        layout: SignedInt8TiledLdsScaleLayout,
        *,
        group_base: int,
    ) -> None:
        """Stage four signed-int8 groups and their FP32 scales in LDS."""
        weight_address = registers.weight_address.first_register
        weight_lds_address = registers.weight_lds_address.first_register
        weight_stage_address = registers.weight_stage_address.first_register
        weight_scale_stage_address = registers.weight_scale_stage_address.first_register
        weight_payload = registers.weight_payload.first_register
        weight_stage_payload = registers.weight_stage_payload.first_register
        weight_scales = registers.weight_scales.first_register
        lane = registers.lane.first_register
        temporary = registers.temporary.first_register
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{lane}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_mul_lo_u32 v{weight_stage_address}, 68, v{temporary}")
        asm.inst(
            f"v_add_nc_u32 v{weight_stage_address}, v{weight_address}, "
            f"v{weight_stage_address}"
        )
        asm.inst(f"v_lshlrev_b32 v{temporary}, 6, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{weight_lds_address}, v{temporary}")
        asm.inst(f"v_lshrrev_b32 v{weight_scale_stage_address}, 4, v{lane}")
        asm.inst(
            f"v_and_b32 v{weight_scale_stage_address}, 1, v{weight_scale_stage_address}"
        )
        asm.inst(
            f"v_lshlrev_b32 v{weight_scale_stage_address}, 3, "
            f"v{weight_scale_stage_address}"
        )
        asm.inst(
            f"v_add_nc_u32 v{weight_scale_stage_address}, "
            f"{layout.weight_scale_offset}, v{weight_scale_stage_address}"
        )
        asm.inst(
            f"v_add_nc_u32 v{weight_scale_stage_address}, v{weight_lds_address}, "
            f"v{weight_scale_stage_address}"
        )
        asm.inst("s_clause 5")
        for group in range(2):
            payload = weight_payload if group == 0 else weight_stage_payload
            asm.inst(
                f"global_load_b128 v[{payload}:{payload + 3}], "
                f"v{weight_stage_address}, s[{self.KERNARG}:{self.KERNARG + 1}] "
                f"offset:{2 + 34 * (group_base + group)}"
            )
            asm.inst(
                f"global_load_b128 v[{payload + 4}:{payload + 7}], "
                f"v{weight_stage_address}, s[{self.KERNARG}:{self.KERNARG + 1}] "
                f"offset:{18 + 34 * (group_base + group)}"
            )
            asm.inst(
                f"global_load_d16_b16 v{weight_scales + group}, "
                f"v{weight_stage_address}, "
                f"s[{self.KERNARG}:{self.KERNARG + 1}] "
                f"offset:{34 * (group_base + group)}"
            )

    def _emit_signed_int8_tiled_weight_writes(
        self,
        asm: Assembly,
        registers: SignedInt8TiledLdsRegisters,
        *,
        group_base: int,
        wait_counts: tuple[int, int] = (3, 0),
    ) -> None:
        """Commit the packed-weight loads after their VMEM dependencies mature."""
        weight_payload = registers.weight_payload.first_register
        weight_stage_payload = registers.weight_stage_payload.first_register
        weight_scales = registers.weight_scales.first_register
        weight_scale_stage_address = registers.weight_scale_stage_address.first_register
        lds_address = registers.temporary.first_register
        for group, wait_count in enumerate(wait_counts):
            asm.inst(f"s_waitcnt vmcnt({wait_count})")
            payload = weight_payload if group == 0 else weight_stage_payload
            asm.inst(
                f"ds_write_b128 v{lds_address}, v[{payload}:{payload + 3}] "
                f"offset:{32 * (group_base + group)}"
            )
            asm.inst(
                f"ds_write_b128 v{lds_address}, v[{payload + 4}:{payload + 7}] "
                f"offset:{16 + 32 * (group_base + group)}"
            )
            asm.inst(
                f"v_cvt_f32_f16 v{weight_scales + group}, v{weight_scales + group}"
            )
            asm.inst(
                f"ds_write_b32 v{weight_scale_stage_address}, "
                f"v{weight_scales + group} "
                f"offset:{4 * (group_base + group)}"
            )

    def _emit_signed_int8_tiled_group(
        self,
        asm: Assembly,
        group: int,
        registers: SignedInt8TiledLdsRegisters,
        layout: SignedInt8TiledLdsScaleLayout,
        *,
        activation_group: int | None = None,
        m_fragments: int = 8,
    ) -> None:
        """Read one staged Q8 group and accumulate eight activation fragments."""
        if m_fragments not in (2, 4, 8):
            raise ValueError("Q8 HIP-shaped group requires 2, 4, or 8 M fragments")
        c = registers.c.first_register
        sums = registers.sums.first_register
        weight_payload = registers.weight_payload.first_register
        activation_payloads = registers.activation_payloads.first_register
        weight_scales = registers.weight_scales.first_register
        activation_scales = registers.activation_scales.first_register
        activation_scale_copies = registers.activation_scale_copies.first_register
        weight_scale_address = registers.weight_scale_address.first_register
        weight_lds_address = registers.weight_lds_address.first_register
        activation_read_address = registers.activation_read_address.first_register
        zero_accumulator = registers.zero_accumulator.first_register
        temporary = registers.temporary.first_register
        activation_group = group if activation_group is None else activation_group
        weight_offset = 32 * group
        asm.inst(
            f"ds_read_b128 v[{weight_payload}:{weight_payload + 3}], "
            f"v{weight_lds_address} "
            f"offset:{weight_offset}"
        )
        asm.inst(
            f"ds_read_b128 v[{weight_payload + 4}:{weight_payload + 7}], "
            f"v{weight_lds_address} "
            f"offset:{weight_offset + 16}"
        )
        if layout.weight_scale_pair_base_delta is not None:
            for pair in range(4):
                relative_element = 2 * (pair % 2)
                address = weight_scale_address if pair < 2 else temporary
                offset0 = (
                    layout.weight_scale_offset
                    + layout.weight_scale_element_stride * relative_element
                    + 4 * group
                ) // 4
                offset1 = offset0 + layout.weight_scale_element_stride // 4
                asm.inst(
                    f"ds_read2_b32 "
                    f"v[{weight_scales + 2 * pair}:{weight_scales + 2 * pair + 1}], "
                    f"v{address} offset0:{offset0} offset1:{offset1}"
                )
        else:
            for element in range(8):
                asm.inst(
                    f"ds_read_b32 v{weight_scales + element}, "
                    f"v{weight_scale_address} "
                    f"offset:{layout.weight_scale_offset + layout.weight_scale_element_stride * element + 4 * group}"
                )
        for m_index in range(m_fragments):
            payload = activation_payloads + 8 * m_index
            activation_row_offset = (
                16 * m_index * self.context.state.contract.activation_block_bytes
            )
            asm.inst(
                f"ds_read_b128 v[{payload}:{payload + 3}], "
                f"v{activation_read_address} "
                f"offset:{activation_row_offset + 16 + 32 * activation_group}"
            )
            asm.inst(
                f"ds_read_b128 v[{payload + 4}:{payload + 7}], "
                f"v{activation_read_address} "
                f"offset:{activation_row_offset + 32 + 32 * activation_group}"
            )
            asm.inst(
                f"ds_read_b32 v{activation_scales + m_index}, "
                f"v{activation_read_address} "
                f"offset:{activation_row_offset + 4 * activation_group}"
            )
        for m_index in range(m_fragments):
            scale_pair_tail = 1 if m_index % 2 == 0 else 0
            consumed_fragments = m_index + 1 + scale_pair_tail
            asm.inst(f"s_waitcnt lgkmcnt({3 * (m_fragments - consumed_fragments)})")
            if m_index % 2 == 0:
                asm.inst(
                    f"v_dual_mov_b32 v{activation_scale_copies}, "
                    f"v{activation_scales + m_index} :: "
                    f"v_dual_mov_b32 v{activation_scale_copies + 1}, "
                    f"v{activation_scales + m_index + 1}"
                )
            fragment = 8 * m_index
            c_fragment = c + fragment
            sum_fragment = sums + fragment
            payload = activation_payloads + 8 * m_index
            emit_signed_i8_wmma(
                asm,
                destination=c_fragment,
                weight=weight_payload,
                activation=payload,
                accumulator=zero_accumulator,
                clamp=False,
            )
            emit_signed_i8_wmma(
                asm,
                destination=c_fragment,
                weight=weight_payload + 4,
                activation=payload + 4,
                accumulator=c_fragment,
                clamp=False,
            )
            for element in range(8):
                asm.inst(
                    f"v_cvt_f32_i32 v{c_fragment + element}, v{c_fragment + element}"
                )
            for element in range(0, 8, 2):
                asm.inst(
                    f"v_dual_mul_f32 v{c_fragment + element}, "
                    f"v{weight_scales + element}, v{c_fragment + element} :: "
                    f"v_dual_mul_f32 v{c_fragment + element + 1}, "
                    f"v{weight_scales + element + 1}, "
                    f"v{c_fragment + element + 1}"
                )
            for element in range(0, 8, 2):
                asm.inst(
                    f"v_dual_fmac_f32 v{sum_fragment + element}, "
                    f"v{activation_scales + m_index}, v{c_fragment + element} :: "
                    f"v_dual_fmac_f32 v{sum_fragment + element + 1}, "
                    f"v{activation_scale_copies + (m_index & 1)}, "
                    f"v{c_fragment + element + 1}"
                )

    def _emit_signed_int8_tiled_store(
        self,
        asm: Assembly,
        registers: SignedInt8TiledLdsRegisters,
        size: ProblemSize,
        *,
        m_fragments: int = 8,
    ) -> None:
        """Store eight wave-N fragments with one legal global clause."""
        if m_fragments not in (2, 4, 8):
            raise ValueError("Q8 HIP-shaped store requires 2, 4, or 8 M fragments")
        sums = registers.sums.first_register
        output_address = registers.output_address.first_register
        store_auxiliary = registers.store_auxiliary.first_register
        lane = registers.lane.first_register
        wave = registers.wave.first_register
        temporary = registers.temporary.first_register
        asm.comment("Store eight wave-N fragments as row-major BF16.")
        asm.inst(f"v_and_b32 v{temporary}, 15, v{lane}")
        asm.inst(
            f"v_lshlrev_b32 v{output_address}, "
            f"{self.context.state.kernel_spec.macro_tile[0].bit_length() - 1}, s3"
        )
        asm.inst(f"v_add_nc_u32 v{output_address}, v{output_address}, v{temporary}")
        asm.inst(f"v_mul_lo_u32 v{output_address}, {2 * size.n}, v{output_address}")
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
                asm.inst(
                    f"v_add_nc_u32 v{address_base}, {32 * size.n}, v{address_base - 8}"
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
