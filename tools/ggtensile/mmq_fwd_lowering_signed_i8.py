"""Signed-int8 direct, register-tiled, and tiled-LDS forward lowerings."""

from dataclasses import dataclass
from typing import ClassVar

from .kernel_abi import ORDINARY_FORWARD_ABI
from .kernel_writer_assembly import (
    Assembly,
    emit_bf16_rne,
    emit_kernel_trailer,
    emit_pointer_kernarg_loads,
)
from .mmq_fwd_lowering import ForwardLoweringContext
from .mmq_fwd_lowering_mma import emit_signed_i8_wmma
from .mmq_fwd_lowering_signed_i8_tiled import SignedInt8TiledLdsMechanics
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

    def body(self) -> str:
        physical = self.context.state.physical_plan
        if isinstance(physical, SignedInt8DirectPhysicalPlan):
            return self._body_signed_int8_direct_global(physical)
        if isinstance(physical, SignedInt8RegisterTiledPhysicalPlan):
            return self._body_signed_int8_register_tiled(physical)
        if isinstance(physical, SignedInt8WaveNTiledLdsPhysicalPlan):
            return self._body_signed_int8_wave_n_tiled_lds(physical)
        if isinstance(physical, SignedInt8SmallMTiledLdsPhysicalPlan):
            return self._body_signed_int8_small_m_tiled_lds(physical)
        raise AssertionError

    def _body_signed_int8_direct_global(
        self, physical: SignedInt8DirectPhysicalPlan
    ) -> str:
        """Lower the isolated one-wave signed-int8 direct-global control."""
        asm = Assembly()
        registers = physical.registers
        name = self.context.kernel_name
        row_stride = self.context.state.packed_weight_row_bytes
        activation_plane_stride = self.context.state.activation_plane_stride_bytes
        sums = registers.sums.first_register
        result_addresses = registers.result_addresses.first_register
        temporary = registers.temporary.first_register
        output_column = registers.output_column.first_register
        serial = registers.serial.first_register
        activation_row = registers.activation_row.first_register

        asm.comment("Load packed Q8_0, the Q8_1 F32_D4 workspace, and output pointers.")
        emit_pointer_kernarg_loads(asm, self.KERNARG, ORDINARY_FORWARD_ABI)

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

    def _body_signed_int8_register_tiled(
        self, physical: SignedInt8RegisterTiledPhysicalPlan
    ) -> str:
        """Lower four signed-int8 fragments per wave in a four-wave group."""
        asm = Assembly()
        wave_tile_m, wave_tile_n = self.context.state.mi_wave_tile
        registers = physical.registers
        name = self.context.kernel_name
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
        emit_pointer_kernarg_loads(asm, self.KERNARG, ORDINARY_FORWARD_ABI)
        asm.comment("Flatten gfx11 packed workitem X/Y and retain wave ownership.")
        asm.inst(f"v_bfe_u32 v{wave}, v0, 10, 10")
        asm.inst(f"v_and_b32 v{lane}, 0x3ff, v0")

        asm.comment("Build two output-column weight and scale address groups.")
        for n_index in range(wave_tile_n):
            asm.inst(f"v_and_b32 v{output_column}, 15, v{lane}")
            asm.inst(
                f"v_lshlrev_b32 v{temporary}, "
                f"{self.context.state.kernel_spec.macro_tile[1].bit_length() - 1}, s2"
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
                f"{self.context.state.kernel_spec.macro_tile[1].bit_length() - 1}, s2"
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
                f"{self.context.state.kernel_spec.macro_tile[0].bit_length() - 1}, s3"
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
            f"{self.context.state.kernel_spec.macro_tile[0].bit_length() - 1}, s3"
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
            f"{self.context.state.kernel_spec.macro_tile[1].bit_length() - 1}, s2"
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

    def _body_signed_int8_small_m_tiled_lds(
        self, physical: SignedInt8SmallMTiledLdsPhysicalPlan
    ) -> str:
        """Lower an exact M32/M64 wave-N Q8 tile with split-row staging."""
        asm = Assembly()
        mechanics = SignedInt8TiledLdsMechanics(
            self.KERNARG,
            self.context.state.contract.activation_block_bytes,
            physical.layout.activation_row_stride,
            physical.data_movement,
        )
        macro_tile_m = self.context.state.kernel_spec.macro_tile[0]
        m_fragments = macro_tile_m // 16
        activation_row_share = 128 // macro_tile_m
        groups_per_lane = 4 // activation_row_share
        registers = physical.registers
        name = self.context.kernel_name
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
        policy = physical.policy
        assert policy.stage_order == "WeightThenActivation"
        tiled_registers: SignedInt8TiledLdsRegisters = registers
        tiled_scale_layout: SignedInt8TiledLdsScaleLayout = layout
        activation_global_row_stride = (
            self.context.state.contract.activation_block_bytes
        )
        activation_lds_row_stride = layout.activation_row_stride
        weight_lds_base = layout.weight_base
        weight_lds_row_stride = layout.weight_row_stride
        paired_scale_reads = policy.scale_read == "PairedHoistedSecondBase"

        asm.comment("Load pointers for an exact small-M wave-N Q8 tile.")
        emit_pointer_kernarg_loads(asm, self.KERNARG, ORDINARY_FORWARD_ABI)
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
            f"{activation_global_row_stride}, v{activation_row}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{temporary}, "
            f"{activation_global_row_stride * macro_tile_m}, s3"
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
        mechanics.emit_weight_stage(
            asm,
            tiled_registers,
            tiled_scale_layout,
            group_base=0,
        )
        mechanics.emit_activation_loads(
            asm,
            tiled_registers,
            registers.activation_address,
            registers.activation_scale_stage_address,
            groups_per_lane,
        )
        mechanics.emit_weight_writes(
            asm,
            tiled_registers,
            group_base=0,
            wait_counts=(6, 0),
        )
        mechanics.emit_activation_writes(
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
            assert weight_scale_pair_base_delta is not None
            asm.inst(
                f"v_add_nc_u32 v{temporary}, {weight_scale_pair_base_delta}, "
                f"v{weight_scale_address}"
            )
        for group in range(4):
            mechanics.emit_group(
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

    def _body_signed_int8_wave_n_tiled_lds(
        self, physical: SignedInt8WaveNTiledLdsPhysicalPlan
    ) -> str:
        """Lower a wave-N 128x64 Q8 tile with cooperative LDS operands."""
        asm = Assembly()
        layout = physical.layout
        mechanics = SignedInt8TiledLdsMechanics(
            self.KERNARG,
            self.context.state.contract.activation_block_bytes,
            layout.activation_row_stride,
            physical.data_movement,
        )
        registers = physical.registers
        policy = physical.policy
        tiled_registers: SignedInt8TiledLdsRegisters = registers
        tiled_scale_layout: SignedInt8TiledLdsScaleLayout = layout
        name = self.context.kernel_name
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

        activation_global_row_stride = (
            self.context.state.contract.activation_block_bytes
        )
        activation_lds_row_stride = layout.activation_row_stride
        weight_lds_base = layout.weight_base
        weight_lds_row_stride = layout.weight_row_stride
        weight_first = policy.stage_order == "WeightThenActivation"

        asm.comment("Load pointers for the HIP-shaped wave-N Q8 tile.")
        emit_pointer_kernarg_loads(asm, self.KERNARG, ORDINARY_FORWARD_ABI)
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
            f"v_mul_lo_u32 v{activation_address}, "
            f"{activation_global_row_stride}, v{activation_row}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{temporary}, "
            f"{activation_global_row_stride * self.context.state.kernel_spec.macro_tile[0]}, s3"
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
        if weight_first:
            mechanics.emit_weight_stage(
                asm,
                tiled_registers,
                tiled_scale_layout,
                group_base=0,
            )
            mechanics.emit_activation_loads(
                asm,
                tiled_registers,
                registers.activation_address,
                registers.activation_address,
                4,
            )
            mechanics.emit_weight_writes(
                asm,
                tiled_registers,
                group_base=0,
                wait_counts=(12, 0),
            )
            mechanics.emit_activation_writes(
                asm,
                tiled_registers,
                registers.activation_lds_address,
                registers.activation_lds_address,
                4,
                trailing_vmem=0,
            )
        elif policy.stage_order == "Interleaved":
            mechanics.emit_activation_loads(
                asm,
                tiled_registers,
                registers.activation_address,
                registers.activation_address,
                4,
            )
            mechanics.emit_weight_stage(
                asm,
                tiled_registers,
                tiled_scale_layout,
                group_base=0,
            )
            mechanics.emit_activation_writes(
                asm,
                tiled_registers,
                registers.activation_lds_address,
                registers.activation_lds_address,
                4,
            )
            mechanics.emit_weight_writes(
                asm,
                tiled_registers,
                group_base=0,
            )
        else:
            raise AssertionError
        if groups_per_iteration == 8:
            mechanics.emit_weight_stage(
                asm,
                tiled_registers,
                tiled_scale_layout,
                group_base=4,
            )
            mechanics.emit_weight_writes(
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
        if policy.scale_read == "PairedHoistedSecondBase":
            weight_scale_pair_base_delta = layout.weight_scale_pair_base_delta
            assert weight_scale_pair_base_delta is not None
            asm.inst(
                f"v_add_nc_u32 v{temporary}, "
                f"{weight_scale_pair_base_delta}, v{weight_scale_address}"
            )
        for group in range(4):
            mechanics.emit_group(
                asm,
                group,
                tiled_registers,
                tiled_scale_layout,
                policy,
            )
        if groups_per_iteration == 8:
            asm.inst(
                f"v_add_nc_u32 v{temporary}, {activation_plane_stride}, "
                f"v{activation_address}"
            )
            mechanics.emit_activation_loads(
                asm,
                tiled_registers,
                registers.temporary,
                registers.temporary,
                4,
            )
            asm.inst("s_barrier")
            mechanics.emit_activation_writes(
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
                mechanics.emit_group(
                    asm,
                    group,
                    tiled_registers,
                    tiled_scale_layout,
                    policy,
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

    def _emit_signed_int8_tiled_store(
        self,
        asm: Assembly,
        registers: SignedInt8TiledLdsRegisters,
        size: ProblemSize,
        *,
        m_fragments: int = 8,
    ) -> None:
        """Store eight wave-N fragments with one legal global clause."""
        assert m_fragments in (2, 4, 8)
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
