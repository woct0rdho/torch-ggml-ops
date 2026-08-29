"""Shared signed-int8 tiled-LDS staging and MMA mechanics."""

from dataclasses import dataclass

from .kernel_writer_assembly import Assembly, RegisterAssignment
from .mmq_fwd_lowering_mma import emit_signed_i8_wmma
from .mmq_fwd_physical import (
    SignedInt8TiledLdsPolicy,
    SignedInt8TiledLdsRegisters,
    SignedInt8TiledLdsScaleLayout,
)
from .mmq_fwd_spec import ForwardDataMovementPolicy


@dataclass(frozen=True)
class SignedInt8TiledLdsMechanics:
    """Emit tiled-LDS operations independent of kernel ownership and ABI shape."""

    kernarg: int
    activation_block_bytes: int
    activation_row_stride: int
    data_movement: ForwardDataMovementPolicy

    def emit_activation_loads(
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
                f"s[{self.kernarg + 2}:{self.kernarg + 3}] "
                f"offset:{16 + 32 * group}"
            )
            asm.inst(
                f"global_load_b128 v[{payload + 4}:{payload + 7}], "
                f"v{payload_address}, "
                f"s[{self.kernarg + 2}:{self.kernarg + 3}] "
                f"offset:{32 + 32 * group}"
            )
            asm.inst(
                f"global_load_b32 v{activation_scales + group}, "
                f"v{scale_address}, "
                f"s[{self.kernarg + 2}:{self.kernarg + 3}] offset:{4 * group}"
            )

    def emit_activation_writes(
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

    def emit_weight_stage(
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
        self.emit_weight_loads(asm, registers, group_base=group_base)

    def emit_weight_loads(
        self,
        asm: Assembly,
        registers: SignedInt8TiledLdsRegisters,
        *,
        group_base: int,
    ) -> None:
        """Load two packed signed-int8 groups from a ready global address."""
        weight_stage_address = registers.weight_stage_address.first_register
        weight_payload = registers.weight_payload.first_register
        weight_stage_payload = registers.weight_stage_payload.first_register
        weight_scales = registers.weight_scales.first_register
        movement = self.data_movement
        payload_width = movement.payload_global_read_vector_width
        metadata_width = movement.metadata_load_vector_width
        assert metadata_width == 2
        load_count = 2 * (2 * (16 // payload_width) + 2 // metadata_width)
        asm.inst(f"s_clause {load_count - 1}")
        for group in range(2):
            payload = weight_payload if group == 0 else weight_stage_payload
            for part, offset in enumerate((2, 18)):
                for chunk in range(0, 16, payload_width):
                    first = payload + part * 4 + chunk // 4
                    last = first + payload_width // 4 - 1
                    asm.inst(
                        f"global_load_b{payload_width * 8} v[{first}:{last}], "
                        f"v{weight_stage_address}, s[{self.kernarg}:{self.kernarg + 1}] "
                        f"offset:{offset + chunk + 34 * (group_base + group)}"
                    )
            for chunk in range(0, 2, metadata_width):
                first = weight_scales + group + chunk // 2
                asm.inst(
                    f"global_load_d16_b16 v{first}, "
                    f"v{weight_stage_address}, s[{self.kernarg}:{self.kernarg + 1}] "
                    f"offset:{34 * (group_base + group) + chunk}"
                )

    def emit_weight_writes(
        self,
        asm: Assembly,
        registers: SignedInt8TiledLdsRegisters,
        *,
        group_base: int,
        wait_counts: tuple[int, int] = (3, 0),
    ) -> None:
        """Commit the packed-weight loads after their VMEM dependencies mature."""
        weight_scale_stage_address = registers.weight_scale_stage_address.first_register
        lds_address = registers.temporary.first_register
        self.emit_weight_writes_to_addresses(
            asm,
            registers,
            group_base=group_base,
            wait_counts=wait_counts,
            lds_address=lds_address,
            scale_lds_address=weight_scale_stage_address,
        )

    def emit_weight_writes_to_addresses(
        self,
        asm: Assembly,
        registers: SignedInt8TiledLdsRegisters,
        *,
        group_base: int,
        wait_counts: tuple[int, int],
        lds_address: int,
        scale_lds_address: int,
    ) -> None:
        """Commit packed-weight loads to ready payload and scale LDS addresses."""
        weight_payload = registers.weight_payload.first_register
        weight_stage_payload = registers.weight_stage_payload.first_register
        weight_scales = registers.weight_scales.first_register
        movement = self.data_movement
        payload_width = movement.payload_lds_write_vector_width
        metadata_width = movement.metadata_lds_write_vector_width
        for group, wait_count in enumerate(wait_counts):
            asm.inst(f"s_waitcnt vmcnt({wait_count})")
            payload = weight_payload if group == 0 else weight_stage_payload
            for part, offset in enumerate((0, 16)):
                for chunk in range(0, 16, payload_width):
                    first = payload + part * 4 + chunk // 4
                    last = first + payload_width // 4 - 1
                    asm.inst(
                        f"ds_write_b{payload_width * 8} v{lds_address}, "
                        f"v[{first}:{last}] "
                        f"offset:{offset + chunk + 32 * (group_base + group)}"
                    )
            asm.inst(
                f"v_cvt_f32_f16 v{weight_scales + group}, v{weight_scales + group}"
            )
            if metadata_width == 4:
                asm.inst(
                    f"ds_write_b32 v{scale_lds_address}, "
                    f"v{weight_scales + group} offset:{4 * (group_base + group)}"
                )
            else:
                raise AssertionError(
                    "signed-int8 scale writes require 32-bit transactions"
                )

    def emit_group(
        self,
        asm: Assembly,
        group: int,
        registers: SignedInt8TiledLdsRegisters,
        layout: SignedInt8TiledLdsScaleLayout,
        policy: SignedInt8TiledLdsPolicy,
        *,
        activation_group: int | None = None,
        m_fragments: int = 8,
        paired_weight_scale_address: int | None = None,
    ) -> None:
        """Read one staged Q8 group and accumulate activation fragments."""
        assert m_fragments in (2, 4, 8)
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
        if policy.scale_read == "PairedHoistedSecondBase":
            assert layout.weight_scale_pair_base_delta is not None
            second_scale_address = paired_weight_scale_address
            if second_scale_address is None:
                second_scale_address = temporary
            for pair in range(4):
                relative_element = 2 * (pair % 2)
                address = weight_scale_address if pair < 2 else second_scale_address
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
        elif policy.scale_read == "Scalar":
            for element in range(8):
                asm.inst(
                    f"ds_read_b32 v{weight_scales + element}, "
                    f"v{weight_scale_address} "
                    f"offset:{layout.weight_scale_offset + layout.weight_scale_element_stride * element + 4 * group}"
                )
        else:
            raise AssertionError
        for m_index in range(m_fragments):
            payload = activation_payloads + 8 * m_index
            row_stride = self.activation_row_stride or self.activation_block_bytes
            activation_row_offset = 16 * m_index * row_stride
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
