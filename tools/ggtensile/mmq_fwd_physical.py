"""Pure physical planning for MMQ forward assembly lowerings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

from .kernel_writer_assembly import (
    DeterministicRegisterPlan,
    DeterministicRegisterPool,
    RegisterAssignment,
    RegisterLifetime,
    RegisterRole,
)
from .mmq_fwd_spec import (
    DecodedLdsLayout,
    F16D4S4ActivationMetadata,
    ForwardKernelSpec,
    ForwardResourceUsage,
    Packed3BitTiledLdsLayout,
    Q3FullWeightTiledLdsLayout,
    Q6LdsLayout,
    QuantForwardSemantics,
    SignedInt8CompactDepth32TiledLdsLayout,
    SignedInt8SmallMTiledLdsLayout,
    forward_mechanism_contract,
)
from .quant_formats import Q8_1_F32_D4_BLOCK_BYTES


@dataclass(frozen=True)
class SignedInt8MmaGroupRole:
    """One 32-value signed payload and activation-scale MMA group."""

    index: int
    weight_block_offset: int
    weight_payload_offset: int
    activation_payload_offset: int
    activation_scale_offset: int

    @classmethod
    def from_semantics(
        cls,
        semantics: QuantForwardSemantics,
        index: int,
    ) -> SignedInt8MmaGroupRole:
        if index not in range(4):
            raise ValueError("signed-int8 direct group requires index 0..3")
        if (
            semantics.weight_bits != 8
            or semantics.payload_plane("qs").encoding != "SignedInt8"
            or semantics.payload_plane("d").encoding != "Float16"
            or semantics.post_wmma_correction != "SignedScaleTimesActivationScale"
        ):
            raise ValueError(
                "signed-int8 direct group requires signed payload semantics"
            )
        weight_block_bytes = sum(plane.byte_count for plane in semantics.payload_planes)
        return cls(
            index=index,
            weight_block_offset=weight_block_bytes * index,
            weight_payload_offset=semantics.payload_plane("qs").byte_offset,
            activation_payload_offset=16 + 32 * index,
            activation_scale_offset=4 * index,
        )


@dataclass(frozen=True)
class SignedInt8DirectRegisterPlan:
    """Typed deterministic ownership for direct signed-int8 lowering."""

    c: RegisterAssignment
    sums: RegisterAssignment
    weight_payload: RegisterAssignment
    activation_payload: RegisterAssignment
    weight_scales: RegisterAssignment
    activation_scale: RegisterAssignment
    result_addresses: RegisterAssignment
    weight_address: RegisterAssignment
    activation_address: RegisterAssignment
    output_address: RegisterAssignment
    temporary: RegisterAssignment
    output_column: RegisterAssignment
    store_auxiliary: RegisterAssignment
    serial: RegisterAssignment
    activation_row: RegisterAssignment
    register_count: int
    declared_vgprs: int

    @classmethod
    def allocate(cls) -> SignedInt8DirectRegisterPlan:
        roles = {
            "c": RegisterRole("c", 8, RegisterLifetime(2, 3), minimum_register=0),
            "sums": RegisterRole("sums", 8, RegisterLifetime(0, 5), minimum_register=8),
            "weight_payload": RegisterRole(
                "weight_payload", 8, RegisterLifetime(2, 3), minimum_register=16
            ),
            "activation_payload": RegisterRole(
                "activation_payload", 8, RegisterLifetime(2, 3), minimum_register=24
            ),
            "weight_scales": RegisterRole(
                "weight_scales", 8, RegisterLifetime(2, 4), minimum_register=32
            ),
            "activation_scale": RegisterRole(
                "activation_scale", 1, RegisterLifetime(2, 4), minimum_register=40
            ),
            "result_addresses": RegisterRole(
                "result_addresses", 8, RegisterLifetime(0, 4), minimum_register=64
            ),
            "weight_address": RegisterRole(
                "weight_address", 1, RegisterLifetime(0, 4), minimum_register=72
            ),
            "activation_address": RegisterRole(
                "activation_address", 1, RegisterLifetime(0, 4), minimum_register=73
            ),
            "output_address": RegisterRole(
                "output_address", 1, RegisterLifetime(5, 5), minimum_register=75
            ),
            "temporary": RegisterRole(
                "temporary", 1, RegisterLifetime(0, 5), minimum_register=84
            ),
            "output_column": RegisterRole(
                "output_column", 1, RegisterLifetime(0, 0), minimum_register=85
            ),
            "store_auxiliary": RegisterRole(
                "store_auxiliary", 1, RegisterLifetime(5, 5), minimum_register=85
            ),
            "serial": RegisterRole(
                "serial", 1, RegisterLifetime(0, 5), minimum_register=86
            ),
            "activation_row": RegisterRole(
                "activation_row", 1, RegisterLifetime(0, 0), minimum_register=87
            ),
        }
        order = tuple(roles)
        declared_vgprs = 88
        plan = DeterministicRegisterPlan.allocate(
            roles,
            order,
            max_registers=declared_vgprs,
        )
        assignments = {name: plan.assignment(name) for name in order}
        return cls(
            **assignments,
            register_count=plan.register_count,
            declared_vgprs=declared_vgprs,
        )


@dataclass(frozen=True)
class SignedInt8RegisterTileRole:
    m_index: int
    n_index: int
    fragment_index: int

    @classmethod
    def all(
        cls,
        wave_tile_m: int,
        wave_tile_n: int,
    ) -> tuple[SignedInt8RegisterTileRole, ...]:
        return tuple(
            cls(m_index, n_index, wave_tile_n * m_index + n_index)
            for m_index in range(wave_tile_m)
            for n_index in range(wave_tile_n)
        )


@dataclass(frozen=True)
class SignedInt8RegisterTiledRegisterPlan:
    """Deterministic ownership for a four-wave signed-int8 register tile."""

    c: RegisterAssignment
    sums: RegisterAssignment
    weight_payloads: RegisterAssignment
    activation_payloads: RegisterAssignment
    weight_scales: RegisterAssignment
    activation_scales: RegisterAssignment
    result_addresses: RegisterAssignment
    weight_addresses: RegisterAssignment
    activation_addresses: RegisterAssignment
    temporary: RegisterAssignment
    output_address: RegisterAssignment
    lane: RegisterAssignment
    wave: RegisterAssignment
    output_column: RegisterAssignment
    store_column: RegisterAssignment
    activation_row: RegisterAssignment
    store_row: RegisterAssignment
    register_count: int
    declared_vgprs: int

    @classmethod
    def allocate(
        cls,
        wave_tile_m: int,
        wave_tile_n: int,
    ) -> SignedInt8RegisterTiledRegisterPlan:
        if wave_tile_m * wave_tile_n != 4:
            raise ValueError("Q8 register tile must own four 16x16 fragments per wave")
        weight_payloads = 8 * wave_tile_n
        activation_payloads = 8 * wave_tile_m
        weight_scales = 8 * wave_tile_n
        result_addresses = 8 * wave_tile_n
        roles = {
            "c": RegisterRole("c", 32, RegisterLifetime(2, 3), minimum_register=0),
            "sums": RegisterRole(
                "sums", 32, RegisterLifetime(0, 5), minimum_register=32
            ),
            "weight_payloads": RegisterRole(
                "weight_payloads",
                weight_payloads,
                RegisterLifetime(2, 3),
                minimum_register=64,
            ),
            "activation_payloads": RegisterRole(
                "activation_payloads",
                activation_payloads,
                RegisterLifetime(2, 3),
                minimum_register=64 + weight_payloads,
            ),
            "weight_scales": RegisterRole(
                "weight_scales",
                weight_scales,
                RegisterLifetime(2, 4),
                minimum_register=64 + weight_payloads + activation_payloads,
            ),
            "activation_scales": RegisterRole(
                "activation_scales",
                wave_tile_m,
                RegisterLifetime(2, 4),
                minimum_register=(
                    64 + weight_payloads + activation_payloads + weight_scales
                ),
            ),
            "result_addresses": RegisterRole(
                "result_addresses",
                result_addresses,
                RegisterLifetime(0, 4),
                minimum_register=(
                    64
                    + weight_payloads
                    + activation_payloads
                    + weight_scales
                    + wave_tile_m
                ),
            ),
            "weight_addresses": RegisterRole(
                "weight_addresses",
                wave_tile_n,
                RegisterLifetime(0, 4),
                minimum_register=(
                    64
                    + weight_payloads
                    + activation_payloads
                    + weight_scales
                    + wave_tile_m
                    + result_addresses
                ),
            ),
            "activation_addresses": RegisterRole(
                "activation_addresses",
                wave_tile_m,
                RegisterLifetime(0, 4),
                minimum_register=(
                    64
                    + weight_payloads
                    + activation_payloads
                    + weight_scales
                    + wave_tile_m
                    + result_addresses
                    + wave_tile_n
                ),
            ),
            "temporary": RegisterRole("temporary", 1, RegisterLifetime(0, 5)),
            "output_address": RegisterRole(
                "output_address", 32, RegisterLifetime(5, 5)
            ),
            "lane": RegisterRole("lane", 1, RegisterLifetime(0, 5)),
            "wave": RegisterRole("wave", 1, RegisterLifetime(0, 5)),
            "output_column": RegisterRole("output_column", 1, RegisterLifetime(0, 0)),
            "store_column": RegisterRole("store_column", 1, RegisterLifetime(5, 5)),
            "activation_row": RegisterRole("activation_row", 1, RegisterLifetime(0, 0)),
            "store_row": RegisterRole("store_row", 1, RegisterLifetime(5, 5)),
        }
        order = tuple(roles)
        max_registers = 67 + 25 * wave_tile_n + 10 * wave_tile_m
        plan = DeterministicRegisterPlan.allocate(
            roles,
            order,
            max_registers=max_registers,
        )
        assignments = {name: plan.assignment(name) for name in order}
        return cls(
            **assignments,
            register_count=plan.register_count,
            declared_vgprs=max_registers,
        )


@dataclass(frozen=True)
class SignedInt8WaveNTiledLdsRegisterPlan:
    """Deterministic registers for the wave-N 128x64 signed-int8 LDS tile."""

    c: RegisterAssignment
    sums: RegisterAssignment
    weight_payload: RegisterAssignment
    activation_payloads: RegisterAssignment
    weight_scales: RegisterAssignment
    activation_scales: RegisterAssignment
    activation_scale_copies: RegisterAssignment
    zero_accumulator: RegisterAssignment
    weight_stage_payload: RegisterAssignment
    weight_scale_address: RegisterAssignment
    weight_address: RegisterAssignment
    activation_address: RegisterAssignment
    activation_lds_address: RegisterAssignment
    weight_lds_address: RegisterAssignment
    weight_stage_address: RegisterAssignment
    weight_scale_stage_address: RegisterAssignment
    temporary: RegisterAssignment
    lane: RegisterAssignment
    wave: RegisterAssignment
    activation_row: RegisterAssignment
    activation_read_address: RegisterAssignment
    weight_row: RegisterAssignment
    output_address: RegisterAssignment
    store_auxiliary: RegisterAssignment
    register_count: int
    declared_vgprs: int

    @classmethod
    def allocate(cls) -> SignedInt8WaveNTiledLdsRegisterPlan:
        roles = {
            "c": RegisterRole("c", 64, RegisterLifetime(2, 3), minimum_register=0),
            "sums": RegisterRole(
                "sums", 64, RegisterLifetime(0, 5), minimum_register=64
            ),
            "weight_payload": RegisterRole(
                "weight_payload", 8, RegisterLifetime(1, 3), minimum_register=128
            ),
            "activation_payloads": RegisterRole(
                "activation_payloads",
                64,
                RegisterLifetime(1, 3),
                minimum_register=136,
            ),
            "weight_scales": RegisterRole(
                "weight_scales", 8, RegisterLifetime(1, 4), minimum_register=200
            ),
            "activation_scales": RegisterRole(
                "activation_scales", 8, RegisterLifetime(1, 3), minimum_register=208
            ),
            "activation_scale_copies": RegisterRole(
                "activation_scale_copies",
                2,
                RegisterLifetime(2, 3),
                minimum_register=225,
            ),
            "zero_accumulator": RegisterRole(
                "zero_accumulator",
                8,
                RegisterLifetime(2, 4),
                minimum_register=227,
            ),
            "weight_stage_payload": RegisterRole(
                "weight_stage_payload", 8, RegisterLifetime(1, 1), minimum_register=0
            ),
            "weight_scale_address": RegisterRole(
                "weight_scale_address",
                1,
                RegisterLifetime(2, 4),
                minimum_register=224,
            ),
            "weight_address": RegisterRole(
                "weight_address", 1, RegisterLifetime(0, 4), minimum_register=216
            ),
            "activation_address": RegisterRole(
                "activation_address", 1, RegisterLifetime(0, 4), minimum_register=217
            ),
            "activation_lds_address": RegisterRole(
                "activation_lds_address",
                1,
                RegisterLifetime(0, 4),
                minimum_register=218,
            ),
            "weight_lds_address": RegisterRole(
                "weight_lds_address",
                1,
                RegisterLifetime(0, 4),
                minimum_register=219,
            ),
            "weight_stage_address": RegisterRole(
                "weight_stage_address",
                1,
                RegisterLifetime(1, 1),
                minimum_register=8,
            ),
            "weight_scale_stage_address": RegisterRole(
                "weight_scale_stage_address",
                1,
                RegisterLifetime(1, 1),
                minimum_register=9,
            ),
            "temporary": RegisterRole(
                "temporary", 1, RegisterLifetime(0, 5), minimum_register=220
            ),
            "lane": RegisterRole(
                "lane", 1, RegisterLifetime(0, 5), minimum_register=221
            ),
            "wave": RegisterRole(
                "wave", 1, RegisterLifetime(0, 5), minimum_register=222
            ),
            "activation_row": RegisterRole(
                "activation_row", 1, RegisterLifetime(0, 0), minimum_register=223
            ),
            "activation_read_address": RegisterRole(
                "activation_read_address",
                1,
                RegisterLifetime(1, 4),
                minimum_register=223,
            ),
            "weight_row": RegisterRole(
                "weight_row", 1, RegisterLifetime(0, 0), minimum_register=10
            ),
            "output_address": RegisterRole(
                "output_address", 64, RegisterLifetime(5, 5), minimum_register=0
            ),
            "store_auxiliary": RegisterRole(
                "store_auxiliary", 1, RegisterLifetime(5, 5), minimum_register=216
            ),
        }
        order = tuple(roles)
        declared_vgprs = 240
        plan = DeterministicRegisterPlan.allocate(
            roles,
            order,
            max_registers=declared_vgprs,
        )
        assignments = {name: plan.assignment(name) for name in order}
        return cls(
            **assignments,
            register_count=plan.register_count,
            declared_vgprs=declared_vgprs,
        )


@dataclass(frozen=True)
class SignedInt8SmallMTiledLdsRegisterPlan:
    """Deterministic registers for exact M32/M64 signed-int8 LDS tiles."""

    c: RegisterAssignment
    sums: RegisterAssignment
    weight_payload: RegisterAssignment
    activation_payloads: RegisterAssignment
    weight_scales: RegisterAssignment
    activation_scales: RegisterAssignment
    activation_scale_copies: RegisterAssignment
    zero_accumulator: RegisterAssignment
    weight_stage_payload: RegisterAssignment
    weight_scale_address: RegisterAssignment
    weight_address: RegisterAssignment
    activation_address: RegisterAssignment
    activation_scale_stage_address: RegisterAssignment
    activation_lds_address: RegisterAssignment
    activation_scale_lds_address: RegisterAssignment
    weight_lds_address: RegisterAssignment
    weight_stage_address: RegisterAssignment
    weight_scale_stage_address: RegisterAssignment
    temporary: RegisterAssignment
    lane: RegisterAssignment
    wave: RegisterAssignment
    activation_row: RegisterAssignment
    activation_group: RegisterAssignment
    activation_read_address: RegisterAssignment
    weight_row: RegisterAssignment
    output_address: RegisterAssignment
    store_auxiliary: RegisterAssignment
    register_count: int
    declared_vgprs: int

    @classmethod
    def allocate(cls, m_fragments: int) -> SignedInt8SmallMTiledLdsRegisterPlan:
        if m_fragments not in (2, 4):
            raise ValueError("Q8 small-M register plan requires two or four fragments")
        c_width = 8 * m_fragments
        sums_base = c_width
        weight_payload_base = 2 * c_width
        activation_payload_base = weight_payload_base + 8
        weight_scales_base = activation_payload_base + c_width
        activation_scales_base = weight_scales_base + 8
        scale_copies_base = activation_scales_base + m_fragments + 1
        zero_base = (scale_copies_base + 2 + 7) // 8 * 8
        address_base = zero_base + 8
        max_registers = 96 if m_fragments == 2 else 144
        roles = {
            "c": RegisterRole("c", c_width, RegisterLifetime(2, 3), minimum_register=0),
            "sums": RegisterRole(
                "sums",
                c_width,
                RegisterLifetime(0, 5),
                minimum_register=sums_base,
            ),
            "weight_payload": RegisterRole(
                "weight_payload",
                8,
                RegisterLifetime(1, 3),
                minimum_register=weight_payload_base,
            ),
            "activation_payloads": RegisterRole(
                "activation_payloads",
                c_width,
                RegisterLifetime(1, 3),
                minimum_register=activation_payload_base,
            ),
            "weight_scales": RegisterRole(
                "weight_scales",
                8,
                RegisterLifetime(1, 4),
                minimum_register=weight_scales_base,
            ),
            "activation_scales": RegisterRole(
                "activation_scales",
                m_fragments,
                RegisterLifetime(1, 3),
                minimum_register=activation_scales_base,
            ),
            "activation_scale_copies": RegisterRole(
                "activation_scale_copies",
                2,
                RegisterLifetime(2, 3),
                minimum_register=scale_copies_base,
            ),
            "zero_accumulator": RegisterRole(
                "zero_accumulator",
                8,
                RegisterLifetime(2, 4),
                alignment=8,
                minimum_register=zero_base,
            ),
            "weight_stage_payload": RegisterRole(
                "weight_stage_payload", 8, RegisterLifetime(1, 1)
            ),
            "weight_scale_address": RegisterRole(
                "weight_scale_address",
                1,
                RegisterLifetime(2, 4),
                minimum_register=address_base,
            ),
            "weight_address": RegisterRole(
                "weight_address",
                1,
                RegisterLifetime(0, 4),
                minimum_register=address_base,
            ),
            "activation_address": RegisterRole(
                "activation_address",
                1,
                RegisterLifetime(0, 4),
                minimum_register=address_base,
            ),
            "activation_scale_stage_address": RegisterRole(
                "activation_scale_stage_address",
                1,
                RegisterLifetime(0, 4),
                minimum_register=address_base,
            ),
            "activation_lds_address": RegisterRole(
                "activation_lds_address",
                1,
                RegisterLifetime(0, 4),
                minimum_register=address_base,
            ),
            "activation_scale_lds_address": RegisterRole(
                "activation_scale_lds_address",
                1,
                RegisterLifetime(0, 4),
                minimum_register=address_base,
            ),
            "weight_lds_address": RegisterRole(
                "weight_lds_address",
                1,
                RegisterLifetime(0, 4),
                minimum_register=address_base,
            ),
            "weight_stage_address": RegisterRole(
                "weight_stage_address",
                1,
                RegisterLifetime(1, 1),
                minimum_register=8,
            ),
            "weight_scale_stage_address": RegisterRole(
                "weight_scale_stage_address",
                1,
                RegisterLifetime(1, 1),
                minimum_register=9,
            ),
            "temporary": RegisterRole(
                "temporary",
                1,
                RegisterLifetime(0, 5),
                minimum_register=address_base,
            ),
            "lane": RegisterRole(
                "lane",
                1,
                RegisterLifetime(0, 5),
                minimum_register=address_base,
            ),
            "wave": RegisterRole(
                "wave",
                1,
                RegisterLifetime(0, 5),
                minimum_register=address_base,
            ),
            "activation_row": RegisterRole(
                "activation_row",
                1,
                RegisterLifetime(0, 0),
                minimum_register=address_base,
            ),
            "activation_group": RegisterRole(
                "activation_group", 1, RegisterLifetime(0, 0), minimum_register=8
            ),
            "activation_read_address": RegisterRole(
                "activation_read_address",
                1,
                RegisterLifetime(1, 4),
                minimum_register=address_base,
            ),
            "weight_row": RegisterRole(
                "weight_row", 1, RegisterLifetime(0, 0), minimum_register=10
            ),
            "output_address": RegisterRole(
                "output_address", c_width, RegisterLifetime(5, 5)
            ),
            "store_auxiliary": RegisterRole(
                "store_auxiliary",
                1,
                RegisterLifetime(5, 5),
                minimum_register=address_base,
            ),
        }
        order = tuple(roles)
        plan = DeterministicRegisterPlan.allocate(
            roles,
            order,
            max_registers=max_registers,
        )
        assignments = {name: plan.assignment(name) for name in order}
        return cls(
            **assignments,
            register_count=plan.register_count,
            declared_vgprs=max_registers,
        )


class SignedInt8TiledLdsRegisters(Protocol):
    """Register roles required by shared signed-int8 tiled-LDS emitters."""

    @property
    def c(self) -> RegisterAssignment: ...

    @property
    def sums(self) -> RegisterAssignment: ...

    @property
    def weight_payload(self) -> RegisterAssignment: ...

    @property
    def activation_payloads(self) -> RegisterAssignment: ...

    @property
    def weight_scales(self) -> RegisterAssignment: ...

    @property
    def activation_scales(self) -> RegisterAssignment: ...

    @property
    def activation_scale_copies(self) -> RegisterAssignment: ...

    @property
    def zero_accumulator(self) -> RegisterAssignment: ...

    @property
    def weight_stage_payload(self) -> RegisterAssignment: ...

    @property
    def weight_scale_address(self) -> RegisterAssignment: ...

    @property
    def weight_address(self) -> RegisterAssignment: ...

    @property
    def activation_address(self) -> RegisterAssignment: ...

    @property
    def activation_lds_address(self) -> RegisterAssignment: ...

    @property
    def weight_lds_address(self) -> RegisterAssignment: ...

    @property
    def weight_stage_address(self) -> RegisterAssignment: ...

    @property
    def weight_scale_stage_address(self) -> RegisterAssignment: ...

    @property
    def temporary(self) -> RegisterAssignment: ...

    @property
    def lane(self) -> RegisterAssignment: ...

    @property
    def wave(self) -> RegisterAssignment: ...

    @property
    def activation_read_address(self) -> RegisterAssignment: ...

    @property
    def output_address(self) -> RegisterAssignment: ...

    @property
    def store_auxiliary(self) -> RegisterAssignment: ...


@dataclass(frozen=True)
class Packed3BitTiledLdsRegisterPlan:
    """Deterministic registers for the wave-N 128x64 Q3 half-tile."""

    sums: RegisterAssignment
    activation_stage: RegisterAssignment
    weight_low_raw: RegisterAssignment
    weight_high_raw: RegisterAssignment
    weight_metadata: RegisterAssignment
    decoded_payload: RegisterAssignment
    stage_d: RegisterAssignment
    stage_scale: RegisterAssignment
    stage_auxiliary: RegisterAssignment
    weight_stage_address: RegisterAssignment
    scale_shift: RegisterAssignment
    c: RegisterAssignment
    zero_accumulator: RegisterAssignment
    weight_payload: RegisterAssignment
    activation_payload: RegisterAssignment
    weight_scales: RegisterAssignment
    activation_scale: RegisterAssignment
    weight_scale_address: RegisterAssignment
    output_address: RegisterAssignment
    weight_address: RegisterAssignment
    activation_address: RegisterAssignment
    activation_lds_address: RegisterAssignment
    activation_read_address: RegisterAssignment
    weight_lds_address: RegisterAssignment
    temporary: RegisterAssignment
    lane: RegisterAssignment
    wave: RegisterAssignment
    register_count: int
    declared_vgprs: int

    @classmethod
    def allocate(cls) -> Packed3BitTiledLdsRegisterPlan:
        roles = {
            "sums": RegisterRole("sums", 64, RegisterLifetime(0, 5)),
            "activation_stage": RegisterRole(
                "activation_stage", 36, RegisterLifetime(1, 1)
            ),
            "weight_low_raw": RegisterRole("weight_low_raw", 4, RegisterLifetime(1, 2)),
            "weight_high_raw": RegisterRole(
                "weight_high_raw", 4, RegisterLifetime(1, 2)
            ),
            "weight_metadata": RegisterRole(
                "weight_metadata", 4, RegisterLifetime(1, 2)
            ),
            "decoded_payload": RegisterRole(
                "decoded_payload", 4, RegisterLifetime(2, 2)
            ),
            "stage_d": RegisterRole("stage_d", 1, RegisterLifetime(2, 2)),
            "stage_scale": RegisterRole("stage_scale", 1, RegisterLifetime(2, 2)),
            "stage_auxiliary": RegisterRole(
                "stage_auxiliary", 1, RegisterLifetime(2, 2)
            ),
            "weight_stage_address": RegisterRole(
                "weight_stage_address", 1, RegisterLifetime(1, 2)
            ),
            "scale_shift": RegisterRole("scale_shift", 1, RegisterLifetime(1, 2)),
            "c": RegisterRole("c", 8, RegisterLifetime(3, 3)),
            "zero_accumulator": RegisterRole(
                "zero_accumulator", 8, RegisterLifetime(0, 3)
            ),
            "weight_payload": RegisterRole("weight_payload", 4, RegisterLifetime(3, 3)),
            "activation_payload": RegisterRole(
                "activation_payload", 32, RegisterLifetime(3, 3)
            ),
            "weight_scales": RegisterRole("weight_scales", 8, RegisterLifetime(3, 3)),
            "activation_scale": RegisterRole(
                "activation_scale", 8, RegisterLifetime(3, 3)
            ),
            "weight_scale_address": RegisterRole(
                "weight_scale_address", 1, RegisterLifetime(3, 3)
            ),
            "output_address": RegisterRole("output_address", 1, RegisterLifetime(5, 5)),
            "weight_address": RegisterRole("weight_address", 1, RegisterLifetime(0, 5)),
            "activation_address": RegisterRole(
                "activation_address", 1, RegisterLifetime(0, 5)
            ),
            "activation_lds_address": RegisterRole(
                "activation_lds_address", 1, RegisterLifetime(0, 5)
            ),
            "activation_read_address": RegisterRole(
                "activation_read_address", 1, RegisterLifetime(0, 3)
            ),
            "weight_lds_address": RegisterRole(
                "weight_lds_address", 1, RegisterLifetime(0, 5)
            ),
            "temporary": RegisterRole("temporary", 2, RegisterLifetime(0, 5)),
            "lane": RegisterRole("lane", 1, RegisterLifetime(0, 5)),
            "wave": RegisterRole("wave", 1, RegisterLifetime(0, 5)),
        }
        order = (
            "sums",
            "activation_stage",
            "activation_payload",
            "decoded_payload",
            "c",
            "weight_metadata",
            "output_address",
            "stage_d",
            "wave",
            "activation_read_address",
            "scale_shift",
            "activation_lds_address",
            "lane",
            "weight_payload",
            "weight_low_raw",
            "weight_scales",
            "temporary",
            "weight_lds_address",
            "stage_auxiliary",
            "weight_scale_address",
            "zero_accumulator",
            "stage_scale",
            "weight_stage_address",
            "weight_address",
            "activation_scale",
            "activation_address",
            "weight_high_raw",
        )
        declared_vgprs = 144
        plan = DeterministicRegisterPlan.allocate(
            roles, order, max_registers=declared_vgprs
        )
        assignments = {name: plan.assignment(name) for name in order}
        return cls(
            **assignments,
            register_count=plan.register_count,
            declared_vgprs=declared_vgprs,
        )


@dataclass(frozen=True)
class Q3FullWeightTiledLdsRegisterPlan:
    """Explicit typed register ownership for the full-weight Q3 tile."""

    sums: RegisterAssignment
    activation_stage: RegisterAssignment
    weight_low_raw: RegisterAssignment
    weight_high_raw: RegisterAssignment
    weight_low1_raw: RegisterAssignment
    weight_metadata: RegisterAssignment
    decoded_payload: RegisterAssignment
    decode_auxiliary: RegisterAssignment
    decode_d: RegisterAssignment
    decode_scale: RegisterAssignment
    weight_stage_address: RegisterAssignment
    half_shift: RegisterAssignment
    c: RegisterAssignment
    zero_accumulator: RegisterAssignment
    weight_payload: RegisterAssignment
    activation_payload: RegisterAssignment
    weight_scales: RegisterAssignment
    activation_scale: RegisterAssignment
    weight_scale_address: RegisterAssignment
    output_address: RegisterAssignment
    weight_address: RegisterAssignment
    activation_address: RegisterAssignment
    activation_lds_address: RegisterAssignment
    activation_read_address: RegisterAssignment
    weight_lds_address: RegisterAssignment
    temporary: RegisterAssignment
    lane: RegisterAssignment
    wave: RegisterAssignment
    register_count: int
    declared_vgprs: int

    def __post_init__(self) -> None:
        assignments = (
            self.sums,
            self.activation_stage,
            self.weight_low_raw,
            self.weight_high_raw,
            self.weight_low1_raw,
            self.weight_metadata,
            self.decoded_payload,
            self.decode_auxiliary,
            self.decode_d,
            self.decode_scale,
            self.weight_stage_address,
            self.half_shift,
            self.c,
            self.zero_accumulator,
            self.weight_payload,
            self.activation_payload,
            self.weight_scales,
            self.activation_scale,
            self.weight_scale_address,
            self.output_address,
            self.weight_address,
            self.activation_address,
            self.activation_lds_address,
            self.activation_read_address,
            self.weight_lds_address,
            self.temporary,
            self.lane,
            self.wave,
        )
        if self.register_count != self.declared_vgprs or self.register_count <= 0:
            raise ValueError("Q3 full-weight register count is inconsistent")
        if any(
            assignment.first_register < 0
            or assignment.first_register + assignment.role.width > self.register_count
            for assignment in assignments
        ):
            raise ValueError("Q3 full-weight register assignment exceeds the plan")

    @staticmethod
    def _fixed(
        name: str,
        width: int,
        first: int,
        first_stage: int,
        last_stage: int,
    ) -> RegisterAssignment:
        return RegisterAssignment(
            RegisterRole(name, width, RegisterLifetime(first_stage, last_stage)),
            first,
        )

    @classmethod
    def allocate(cls) -> Q3FullWeightTiledLdsRegisterPlan:
        fixed = cls._fixed
        return cls(
            sums=fixed("sums", 64, 0, 0, 5),
            activation_stage=fixed("activation_stage", 36, 64, 1, 1),
            weight_low_raw=fixed("weight_low_raw", 4, 105, 1, 2),
            weight_high_raw=fixed("weight_high_raw", 4, 110, 1, 2),
            weight_low1_raw=fixed("weight_low1_raw", 4, 114, 1, 2),
            weight_metadata=fixed("weight_metadata", 4, 100, 1, 2),
            decoded_payload=fixed("decoded_payload", 4, 126, 2, 2),
            decode_auxiliary=fixed("decode_auxiliary", 2, 131, 2, 2),
            decode_d=fixed("decode_d", 1, 130, 2, 2),
            decode_scale=fixed("decode_scale", 1, 132, 2, 2),
            weight_stage_address=fixed("weight_stage_address", 1, 109, 1, 2),
            half_shift=fixed("half_shift", 1, 104, 1, 2),
            c=fixed("c", 8, 96, 3, 3),
            zero_accumulator=fixed("zero_accumulator", 8, 183, 0, 3),
            weight_payload=fixed("weight_payload", 4, 164, 3, 3),
            activation_payload=fixed("activation_payload", 32, 64, 3, 3),
            weight_scales=fixed("weight_scales", 8, 168, 3, 3),
            activation_scale=fixed("activation_scale", 8, 192, 3, 3),
            weight_scale_address=fixed("weight_scale_address", 1, 179, 3, 3),
            output_address=fixed("output_address", 1, 64, 5, 5),
            weight_address=fixed("weight_address", 1, 191, 0, 5),
            activation_address=fixed("activation_address", 1, 162, 0, 5),
            activation_lds_address=fixed("activation_lds_address", 1, 162, 0, 5),
            activation_read_address=fixed("activation_read_address", 1, 161, 0, 3),
            weight_lds_address=fixed("weight_lds_address", 1, 178, 0, 5),
            temporary=fixed("temporary", 2, 176, 0, 5),
            lane=fixed("lane", 1, 163, 0, 5),
            wave=fixed("wave", 1, 160, 0, 5),
            register_count=200,
            declared_vgprs=200,
        )


@dataclass(frozen=True)
class Q3FullWeightTiledLdsPhysicalPlan:
    layout: Q3FullWeightTiledLdsLayout
    registers: Q3FullWeightTiledLdsRegisterPlan
    resources: ForwardResourceUsage


@dataclass(frozen=True)
class SignedInt8WaveNTiledLdsLayout:
    """Formula-derived ordinary signed-int8 wave-N LDS planes."""

    activation_rows: int = 128
    weight_rows: int = 64
    activation_row_stride: int = Q8_1_F32_D4_BLOCK_BYTES
    weight_row_stride: int = 304
    weight_scale_offset: int = 256
    allocation_padding_bytes: int = 512

    def __post_init__(self) -> None:
        if (
            self.activation_rows,
            self.weight_rows,
            self.activation_row_stride,
            self.weight_row_stride,
            self.weight_scale_offset,
            self.allocation_padding_bytes,
        ) != (128, 64, Q8_1_F32_D4_BLOCK_BYTES, 304, 256, 512):
            raise ValueError("ordinary Q8 LDS layout has fixed HIP-shaped dimensions")

    @property
    def activation_bytes(self) -> int:
        return self.activation_rows * self.activation_row_stride

    @property
    def weight_base(self) -> int:
        return self.activation_bytes

    @property
    def weight_bytes(self) -> int:
        return self.weight_rows * self.weight_row_stride

    @property
    def weight_scale_element_stride(self) -> int:
        return 2 * self.weight_row_stride

    @property
    def weight_scale_pair_base_delta(self) -> int | None:
        return None

    @property
    def total_bytes(self) -> int:
        return self.activation_bytes + self.weight_bytes + self.allocation_padding_bytes


class SignedInt8TiledLdsScaleLayout(Protocol):
    """Scale-plane facts shared by ordinary, small-M, and compact-KV LDS."""

    @property
    def weight_scale_offset(self) -> int: ...

    @property
    def weight_scale_element_stride(self) -> int: ...

    @property
    def weight_scale_pair_base_delta(self) -> int | None: ...


@dataclass(frozen=True)
class SignedInt8TiledLdsPolicy:
    """Independent staging and scale-read policies for a tiled Q8 layout."""

    stage_order: Literal["Interleaved", "WeightThenActivation"]
    scale_read: Literal["Scalar", "PairedHoistedSecondBase"]


_Q6_PHYSICAL_SEMANTICS = QuantForwardSemantics.for_quant_type("Q6_K")


@dataclass(frozen=True)
class Q6Vgpr:
    """One typed physical VGPR operand at the final Q6 emission boundary."""

    register: int

    def __post_init__(self) -> None:
        if self.register < 0:
            raise ValueError("Q6 VGPR must be nonnegative")

    def __str__(self) -> str:
        return f"v{self.register}"


@dataclass(frozen=True)
class Q6Sgpr:
    """One typed physical SGPR operand at the final Q6 emission boundary."""

    register: int

    def __post_init__(self) -> None:
        if self.register < 0:
            raise ValueError("Q6 SGPR must be nonnegative")

    def __str__(self) -> str:
        return f"s{self.register}"


@dataclass(frozen=True)
class Q6Immediate:
    """One integer address operand with explicit decimal or hexadecimal spelling."""

    value: int
    hexadecimal: bool = False

    def __str__(self) -> str:
        return hex(self.value) if self.hexadecimal else str(self.value)


Q6AddressOperand = Q6Vgpr | Q6Sgpr | Q6Immediate
Q6DelayKind = Literal["VALU_DEP", "SALU_CYCLE"]
Q6DelaySkip = Literal["NEXT", "SKIP_1", "SKIP_2", "SKIP_3"]


@dataclass(frozen=True)
class Q6DependencyDelay:
    """One typed single- or dual-dependency s_delay_alu expression."""

    kind: Q6DelayKind
    first_distance: int
    skip: Q6DelaySkip | None = None
    second_distance: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("VALU_DEP", "SALU_CYCLE"):
            raise ValueError(f"unsupported Q6 dependency kind: {self.kind}")
        if self.first_distance <= 0:
            raise ValueError("Q6 dependency distance must be positive")
        if self.skip is not None and self.skip not in (
            "NEXT",
            "SKIP_1",
            "SKIP_2",
            "SKIP_3",
        ):
            raise ValueError(f"unsupported Q6 dependency skip: {self.skip}")
        if (self.skip is None) != (self.second_distance is None):
            raise ValueError("Q6 paired dependency delay requires skip and second")
        if self.second_distance is not None and self.second_distance <= 0:
            raise ValueError("Q6 dependency distance must be positive")

    def __str__(self) -> str:
        expression = f"instid0({self.kind}_{self.first_distance})"
        if self.skip is not None and self.second_distance is not None:
            expression += (
                f" | instskip({self.skip}) | "
                f"instid1({self.kind}_{self.second_distance})"
            )
        return expression


@dataclass(frozen=True)
class Q6OffsetAddress:
    """One derived byte-offset address based on a physical VGPR pair."""

    destination: int
    byte_offset: int
    base_register: int
    delay_after_low: bool


Q6AddressBatch = tuple[Q6OffsetAddress, ...]


Q6HalfName = Literal["l", "h"]
Q6PackedPayloadPlane = Literal["ql", "qh"]


@dataclass(frozen=True)
class Q6HalfRegister:
    """One typed 16-bit half of a physical Q6 VGPR."""

    register: int
    half: Q6HalfName

    def __post_init__(self) -> None:
        if self.register < 0:
            raise ValueError("Q6 half-register VGPR must be nonnegative")
        if self.half not in ("l", "h"):
            raise ValueError("Q6 half-register selector must be 'l' or 'h'")

    def __str__(self) -> str:
        return f"v{self.register}.{self.half}"


@dataclass(frozen=True)
class Q6OwnershipRegisterPlan:
    """Typed payload ownership shared by setup, decode, LDS, and refill."""

    output_tile_rows: int
    packed_payloads: tuple[RegisterAssignment, ...]
    activation_payloads: tuple[RegisterAssignment, ...]
    refill_payloads: tuple[RegisterAssignment, ...]

    @classmethod
    def for_output_tile_rows(cls, output_tile_rows: int) -> Q6OwnershipRegisterPlan:
        source_pairs = {
            1: (
                (10, 122),
                (8, 79),
                (7, 9),
                (6, 80),
                (84, 88),
                (83, 86),
                (82, 85),
                (81, 87),
                (109, 113),
                (108, 111),
                (107, 110),
                (106, 112),
                (121, 120),
                (116, 119),
                (118, 117),
                (114, 115),
            ),
            2: (
                (112, 172),
                (9, 10),
                (7, 8),
                (6, 111),
                (119, 120),
                (116, 117),
                (114, 115),
                (113, 118),
                (162, 163),
                (159, 160),
                (157, 158),
                (156, 161),
                (170, 171),
                (168, 169),
                (166, 167),
                (164, 165),
            ),
        }.get(output_tile_rows)
        if source_pairs is None:
            raise ValueError(f"unsupported Q6 ownership rows: {output_tile_rows}")

        packed_payloads = tuple(
            RegisterAssignment(
                RegisterRole(
                    f"packed_payload.{atom}.{plane}",
                    1,
                    RegisterLifetime(0, 2 * atom + (plane == "qh")),
                    minimum_register=register,
                ),
                register,
            )
            for atom, (low, high) in enumerate(source_pairs)
            for plane, register in (("ql", low), ("qh", high))
        )
        activation_registers = (
            (5, 89, *range(90, 96), *range(97, 105))
            if output_tile_rows == 1
            else (
                5,
                *range(121, 128),
                153,
                *range(128, 135),
                146,
                *range(135, 142),
                145,
                142,
                143,
                147,
                *range(148, 152),
                144,
                152,
            )
        )
        refill_registers = (
            (8, 9, 10, 79, 80, 81, 82, 83, 84, 7, 85, 86, 87, 88, 89, 4, 6, 2)
            if output_tile_rows == 1
            else (
                10,
                113,
                114,
                115,
                116,
                117,
                118,
                119,
                134,
                120,
                121,
                122,
                123,
                124,
                125,
                126,
                6,
                127,
                128,
                129,
                130,
                131,
                132,
                8,
                7,
                9,
                133,
                112,
                135,
                136,
                137,
                4,
                111,
                5,
                138,
                2,
            )
        )
        activation_payloads = tuple(
            RegisterAssignment(
                RegisterRole(
                    f"activation_payload.{slot}",
                    1,
                    RegisterLifetime(1, 3),
                    minimum_register=register,
                ),
                register,
            )
            for slot, register in enumerate(activation_registers)
        )
        refill_payloads = tuple(
            RegisterAssignment(
                RegisterRole(
                    f"refill_payload.{slot}",
                    1,
                    RegisterLifetime(6, 7),
                    minimum_register=register,
                ),
                register,
            )
            for slot, register in enumerate(refill_registers)
        )
        return cls(
            output_tile_rows,
            packed_payloads,
            activation_payloads,
            refill_payloads,
        )

    def packed_payload(
        self, atom: int, plane: Q6PackedPayloadPlane
    ) -> RegisterAssignment:
        if atom not in range(16):
            raise ValueError(f"unsupported Q6 payload atom: {atom}")
        plane_index = 0 if plane == "ql" else 1
        return self.packed_payloads[2 * atom + plane_index]


@dataclass(frozen=True)
class Q6PackedDecodeSourceRole:
    """One logical QL/QH source pair consumed by a Q6 decode atom."""

    atom: int
    low_payload: RegisterAssignment
    high_payload: RegisterAssignment


@dataclass(frozen=True)
class Q6DecodedInt8PairRole:
    """The low/high signed int8 dwords produced by one Q6 decode atom."""

    atom: int
    low: RegisterAssignment
    high: RegisterAssignment


@dataclass(frozen=True)
class Q6DecodeRegisterPlan:
    """Typed Q6 decode operands and deterministic source-to-output reuse."""

    output_tile_rows: int
    sources: tuple[Q6PackedDecodeSourceRole, ...]
    outputs: tuple[Q6DecodedInt8PairRole, ...]
    lane_shift_register: int
    factor_source_register: int
    factor_register: int
    initial_scratch_register: int
    activation_payload_registers: tuple[int, ...]

    @classmethod
    def for_output_tile_rows(cls, output_tile_rows: int) -> Q6DecodeRegisterPlan:
        ownership = Q6OwnershipRegisterPlan.for_output_tile_rows(output_tile_rows)
        source_assignments = tuple(
            (
                ownership.packed_payload(atom, "ql"),
                ownership.packed_payload(atom, "qh"),
            )
            for atom in range(16)
        )
        if output_tile_rows == 1:
            lane_shift_register = 42
            factor_source_register = 124
            factor_register = 142
            initial_scratch_register = 123
        else:
            lane_shift_register = 69
            factor_source_register = 174
            factor_register = 205
            initial_scratch_register = 173

        atom_count = len(source_assignments)
        output_last_stage = 2 * atom_count + 1
        pool = DeterministicRegisterPool(
            tuple(
                sorted(
                    {
                        initial_scratch_register,
                        *(
                            assignment.first_register
                            for pair in source_assignments
                            for assignment in pair
                        ),
                    }
                )
            )
        )
        sources = []
        for atom, (low_assignment, high_assignment) in enumerate(source_assignments):
            sources.append(
                Q6PackedDecodeSourceRole(
                    atom,
                    pool.checkout(
                        low_assignment.role,
                        preferred_register=low_assignment.first_register,
                    ),
                    pool.checkout(
                        high_assignment.role,
                        preferred_register=high_assignment.first_register,
                    ),
                )
            )

        outputs = []
        for source in sources:
            atom = source.atom
            pool.checkin(source.low_payload.role.name)
            low_role = RegisterRole(
                f"decoded_int8.{atom}.low",
                1,
                RegisterLifetime(2 * atom + 1, output_last_stage),
            )
            high_role = RegisterRole(
                f"decoded_int8.{atom}.high",
                1,
                RegisterLifetime(2 * atom + 1, output_last_stage),
            )
            low = pool.checkout(
                low_role,
                preferred_register=source.low_payload.first_register,
            )
            high = pool.checkout(high_role)
            pool.checkin(source.high_payload.role.name)
            outputs.append(Q6DecodedInt8PairRole(atom, low, high))

        return cls(
            output_tile_rows=output_tile_rows,
            sources=tuple(sources),
            outputs=tuple(outputs),
            lane_shift_register=lane_shift_register,
            factor_source_register=factor_source_register,
            factor_register=factor_register,
            initial_scratch_register=initial_scratch_register,
            activation_payload_registers=tuple(
                assignment.first_register
                for assignment in ownership.activation_payloads
            ),
        )

    @property
    def factor_write_address(self) -> int:
        return 28 + 27 * self.output_tile_rows

    @property
    def scale_write_address_base(self) -> int:
        return 29 + 28 * self.output_tile_rows

    @property
    def cooperative_write_address(self) -> int:
        return 25 + 27 * self.output_tile_rows


@dataclass(frozen=True)
class Q6AddressAdd:
    destination: int
    left: Q6AddressOperand
    right: Q6AddressOperand
    high: Q6AddressOperand
    high_left: Q6AddressOperand = Q6Immediate(0)
    delay_after_low: Q6DependencyDelay | None = None
    delay_after_high: Q6DependencyDelay | None = None


@dataclass(frozen=True)
class Q6GlobalRead:
    destination_register: int
    address_register: int
    offset: int = 0
    width_bits: int = 32
    local_write_slot: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.destination_register, int)
            or self.destination_register < 0
        ):
            destination = (
                f"v{self.destination_register}"
                if isinstance(self.destination_register, int)
                else str(self.destination_register)
            )
            raise ValueError(
                f"Q6 global-read destination must be one VGPR: {destination}"
            )
        if not isinstance(self.address_register, int) or self.address_register < 0:
            raise ValueError("Q6 global-read address must be one VGPR pair")
        if self.width_bits not in (16, 32):
            raise ValueError(f"unsupported Q6 global-read width: {self.width_bits}")
        if self.local_write_slot is not None and self.local_write_slot < 0:
            raise ValueError("Q6 local-write slot must be nonnegative")

    @property
    def payload_assignment(self) -> RegisterAssignment:
        lifetime = (
            RegisterLifetime(6, 7)
            if self.local_write_slot is not None
            else RegisterLifetime(1, 4)
        )
        role_name = (
            f"refill_payload.{self.local_write_slot}"
            if self.local_write_slot is not None
            else f"packed_payload.v{self.destination_register}"
        )
        role = RegisterRole(
            role_name,
            1,
            lifetime,
            minimum_register=self.destination_register,
        )
        return RegisterAssignment(role, self.destination_register)

    @property
    def address_assignment(self) -> RegisterAssignment:
        lifetime = (
            RegisterLifetime(5, 6)
            if self.local_write_slot is not None
            else RegisterLifetime(0, 1)
        )
        role = RegisterRole(
            f"global_address.v{self.address_register}",
            2,
            lifetime,
            minimum_register=self.address_register,
        )
        return RegisterAssignment(role, self.address_register)


@dataclass(frozen=True)
class Q6AccumulatorOutputRole:
    """One semantic output pair carried through the Q6 dot epilogue."""

    group: int
    element: int
    left_register: int
    right_register: int
    left_product_index: int
    right_product_index: int
    lifetime: RegisterLifetime


@dataclass(frozen=True)
class Q6PhysicalRegisterMap:
    """Typed physical roles for one wave's retained Q6 lowering."""

    assignments: tuple[RegisterAssignment, ...]
    output_roles: tuple[Q6AccumulatorOutputRole, ...]

    @classmethod
    def for_output_tile_rows(cls, output_tile_rows: int) -> Q6PhysicalRegisterMap:
        if output_tile_rows == 1:
            accumulator_registers = (
                32,
                46,
                44,
                43,
                40,
                39,
                38,
                37,
                36,
                35,
                33,
                31,
                *range(30, 10, -1),
            )
            right_products = (9, 10, 11, 8, 13, 12, 15, 14)
        elif output_tile_rows == 2:
            accumulator_registers = (
                110,
                105,
                97,
                91,
                84,
                76,
                72,
                71,
                67,
                66,
                65,
                64,
                63,
                62,
                60,
                59,
                *range(58, 10, -1),
            )
            right_products = (10, 11, 9, 13, 15, 8, 12, 14)
        else:
            raise ValueError(f"unsupported Q6 physical output rows: {output_tile_rows}")
        assignments = [
            RegisterAssignment(
                RegisterRole(
                    f"accumulator.{index}",
                    1,
                    RegisterLifetime(5, 8),
                    minimum_register=register,
                ),
                register,
            )
            for index, register in enumerate(accumulator_registers)
        ]
        base = 48 + 32 * output_tile_rows
        roles = (
            ("row_pointer", base - 1, RegisterLifetime(0, 8)),
            ("weight_address", base, RegisterLifetime(0, 8)),
            ("activation_address", base + 1, RegisterLifetime(0, 8)),
            ("product", base + 2, RegisterLifetime(5, 7)),
            ("first_fragment", base + 18, RegisterLifetime(5, 7)),
            ("scale", base + 18 + 36 * output_tile_rows, RegisterLifetime(5, 7)),
            ("factor", base + 18 + 32 * output_tile_rows, RegisterLifetime(5, 7)),
            ("output_index", 7 + 27 * output_tile_rows, RegisterLifetime(0, 8)),
            (
                "first_stage_weight.0",
                38 + 58 * output_tile_rows,
                RegisterLifetime(1, 4),
            ),
            (
                "first_stage_weight.1",
                55 + 50 * output_tile_rows,
                RegisterLifetime(1, 4),
            ),
            (
                "first_stage_write_address",
                25 + 27 * output_tile_rows,
                RegisterLifetime(1, 4),
            ),
            (
                "first_stage_read_address",
                44 + 31 * output_tile_rows,
                RegisterLifetime(4, 5),
            ),
        )
        assignments.extend(
            RegisterAssignment(
                RegisterRole(name, 1, lifetime, minimum_register=register), register
            )
            for name, register, lifetime in roles
        )
        output_roles = []
        accumulator_lifetime = RegisterLifetime(5, 8)
        standard_right_products = (9, 8, 11, 10, 13, 12, 15, 14)
        for group in range(2 * output_tile_rows):
            group_right_products = (
                right_products if group == 0 else standard_right_products
            )
            group_base = 16 * group
            output_roles.extend(
                Q6AccumulatorOutputRole(
                    group=group,
                    element=element,
                    left_register=accumulator_registers[group_base + element],
                    right_register=accumulator_registers[
                        group_base + group_right_products[element]
                    ],
                    left_product_index=group_base + element,
                    right_product_index=group_base + group_right_products[element],
                    lifetime=accumulator_lifetime,
                )
                for element in range(8)
            )
        return cls(tuple(assignments), tuple(output_roles))

    def assignment(self, role: str) -> RegisterAssignment:
        for assignment in self.assignments:
            if assignment.role.name == role:
                return assignment
        raise KeyError(role)

    def register(self, role: str) -> int:
        return self.assignment(role).first_register

    @property
    def accumulator_registers(self) -> tuple[int, ...]:
        return tuple(
            assignment.first_register
            for assignment in self.assignments
            if assignment.role.name.startswith("accumulator.")
        )

    def lifetime(self, role: str) -> RegisterLifetime:
        return self.assignment(role).role.lifetime

    def group_output_roles(self, group: int) -> tuple[Q6AccumulatorOutputRole, ...]:
        roles = tuple(role for role in self.output_roles if role.group == group)
        if len(roles) != 8:
            raise ValueError(f"Q6 accumulator group has {len(roles)} output roles")
        return roles


@dataclass(frozen=True)
class Q6PhysicalLayout:
    """Selected physical role layout derived from per-wave output ownership."""

    output_tile_rows: int

    @property
    def registers(self) -> Q6PhysicalRegisterMap:
        return Q6PhysicalRegisterMap.for_output_tile_rows(self.output_tile_rows)

    @property
    def tile_count(self) -> int:
        return 4 * self.output_tile_rows

    @property
    def lds(self) -> Q6LdsLayout:
        return Q6LdsLayout(self.output_tile_rows)

    @property
    def ownership(self) -> Q6OwnershipRegisterPlan:
        return Q6OwnershipRegisterPlan.for_output_tile_rows(self.output_tile_rows)

    @property
    def decode(self) -> Q6DecodeRegisterPlan:
        return Q6DecodeRegisterPlan.for_output_tile_rows(self.output_tile_rows)

    def packed_read(
        self,
        atom: int,
        plane: Q6PackedPayloadPlane,
        address: int,
    ) -> Q6GlobalRead:
        return Q6GlobalRead(
            self.ownership.packed_payload(atom, plane).first_register,
            address,
            _Q6_PHYSICAL_SEMANTICS.q6_packed_payload_offset(atom, plane),
        )

    def refill_read(
        self,
        slot: int,
        address: int,
        offset: int = 0,
    ) -> Q6GlobalRead:
        if slot not in range(len(self.ownership.refill_payloads)):
            raise ValueError(f"unsupported Q6 refill slot: {slot}")
        destination = self.ownership.refill_payloads[slot].first_register
        return Q6GlobalRead(
            destination,
            address,
            offset,
            local_write_slot=slot,
        )

    def decoded_write_address(self, atom: int) -> int:
        if atom not in range(16):
            raise ValueError(f"unsupported Q6 decoded write atom: {atom}")
        base = 31 + 28 * self.output_tile_rows
        if self.output_tile_rows == 1:
            return base + atom
        return base + atom + int(atom >= 4) + int(atom >= 9)

    def decoded_write_offsets(self, atom: int) -> tuple[int, int]:
        if atom not in range(16):
            raise ValueError(f"unsupported Q6 decoded write atom: {atom}")
        canonical_atom = atom if self.output_tile_rows == 1 else (atom + 12) % 16
        if canonical_atom < 9:
            offset0 = (64 + 48 * canonical_atom) % 256
        elif canonical_atom == 9:
            offset0 = 112
        else:
            offset0 = (32 + 48 * (canonical_atom - 10)) % 256
        return offset0, offset0 + 16

    @property
    def weight_address_base(self) -> int:
        return self.registers.register("weight_address")

    @property
    def activation_address_base(self) -> int:
        return self.registers.register("activation_address")

    @property
    def row_pointer_base(self) -> int:
        return self.registers.register("row_pointer")

    @property
    def product_base(self) -> int:
        return self.registers.register("product")

    @property
    def scale_base(self) -> int:
        return self.registers.register("scale")

    @property
    def factor_base(self) -> int:
        return self.registers.register("factor")

    @property
    def first_fragment_base(self) -> int:
        return self.registers.register("first_fragment")

    @property
    def accumulator_registers(self) -> tuple[int, ...]:
        return self.registers.accumulator_registers

    @property
    def output_index_register(self) -> int:
        return self.registers.register("output_index")

    @property
    def scratch_base(self) -> int:
        return max((*self.accumulator_registers, self.output_index_register)) + 1

    @property
    def declared_vgprs(self) -> int:
        return 106 + 52 * self.output_tile_rows

    @property
    def declared_sgprs(self) -> int:
        return 27

    @property
    def first_stage_weight_values(self) -> tuple[int, int]:
        return (
            self.registers.register("first_stage_weight.0"),
            self.registers.register("first_stage_weight.1"),
        )

    @property
    def first_stage_vmem_wait(self) -> int | None:
        return 0 if self.output_tile_rows == 1 else None

    @property
    def loop_exit_annotation(self) -> str:
        macro_tile_m = 64 * self.output_tile_rows
        exit_index = 68 + 88 * self.output_tile_rows
        return (
            "; %bb.5:                                ; "
            f"%_ZL18mmq_vec_dot_targetIL9ggml_type14ELi{macro_tile_m}"
            f"ELb1ELb0EEvPKiS2_Pfi.exit{exit_index}.i"
        )

    @property
    def first_stage_write_address(self) -> int:
        return self.registers.register("first_stage_write_address")

    @property
    def first_stage_read_address_base(self) -> int:
        return self.registers.register("first_stage_read_address")

    @property
    def stage_pointer_sources(self) -> tuple[int, int, int]:
        return (
            self.first_stage_write_address - 1,
            self.first_stage_write_address - self.output_tile_rows - 2,
            self.first_stage_write_address - 2,
        )

    def stage_reads(
        self, destination_base: int
    ) -> tuple[tuple[tuple[int, int], int, int, int], ...]:
        return tuple(
            (
                (
                    destination_base + 2 * role.pair,
                    destination_base + 2 * role.pair + 1,
                ),
                self.first_stage_read_address_base + role.pair,
                role.offset0,
                role.offset1,
            )
            for role in self.lds.stage_read_roles
        )


@dataclass(frozen=True)
class PackedScaleMinimumDirectRegisterPlan:
    """Deterministic ownership for direct packed scale/minimum lowering."""

    c: RegisterAssignment
    sums: RegisterAssignment
    weight_payload: RegisterAssignment
    activation_payload: RegisterAssignment
    weight_metadata: RegisterAssignment
    result_weight_addresses: RegisterAssignment
    weight_address: RegisterAssignment
    activation_addresses: RegisterAssignment
    output_address: RegisterAssignment
    scale: RegisterAssignment
    minimum: RegisterAssignment
    scaled_dm: RegisterAssignment
    activation_scale_sum: RegisterAssignment
    weight_d: RegisterAssignment
    weight_minimum: RegisterAssignment
    activation_d: RegisterAssignment
    activation_sum: RegisterAssignment
    temporary: RegisterAssignment
    output_column: RegisterAssignment
    store_auxiliary: RegisterAssignment
    serial: RegisterAssignment
    activation_row: RegisterAssignment
    register_count: int
    declared_vgprs: int

    @classmethod
    def allocate(cls) -> PackedScaleMinimumDirectRegisterPlan:
        roles = {
            "c": RegisterRole("c", 8, RegisterLifetime(2, 3), minimum_register=0),
            "sums": RegisterRole("sums", 8, RegisterLifetime(0, 5), minimum_register=8),
            "weight_payload": RegisterRole(
                "weight_payload", 8, RegisterLifetime(2, 3), minimum_register=16
            ),
            "activation_payload": RegisterRole(
                "activation_payload", 8, RegisterLifetime(2, 3), minimum_register=24
            ),
            "weight_metadata": RegisterRole(
                "weight_metadata", 32, RegisterLifetime(1, 4), minimum_register=32
            ),
            "result_weight_addresses": RegisterRole(
                "result_weight_addresses",
                8,
                RegisterLifetime(0, 4),
                minimum_register=64,
            ),
            "weight_address": RegisterRole(
                "weight_address", 1, RegisterLifetime(0, 4), minimum_register=72
            ),
            "activation_addresses": RegisterRole(
                "activation_addresses",
                2,
                RegisterLifetime(0, 4),
                minimum_register=73,
            ),
            "output_address": RegisterRole(
                "output_address", 1, RegisterLifetime(5, 5), minimum_register=75
            ),
            "scale": RegisterRole(
                "scale", 1, RegisterLifetime(2, 4), minimum_register=76
            ),
            "minimum": RegisterRole(
                "minimum", 1, RegisterLifetime(2, 4), minimum_register=77
            ),
            "scaled_dm": RegisterRole(
                "scaled_dm", 1, RegisterLifetime(4, 4), minimum_register=78
            ),
            "activation_scale_sum": RegisterRole(
                "activation_scale_sum",
                1,
                RegisterLifetime(2, 4),
                minimum_register=79,
            ),
            "weight_d": RegisterRole(
                "weight_d", 1, RegisterLifetime(4, 4), minimum_register=80
            ),
            "weight_minimum": RegisterRole(
                "weight_minimum", 1, RegisterLifetime(4, 4), minimum_register=81
            ),
            "activation_d": RegisterRole(
                "activation_d", 1, RegisterLifetime(4, 4), minimum_register=82
            ),
            "activation_sum": RegisterRole(
                "activation_sum", 1, RegisterLifetime(4, 4), minimum_register=83
            ),
            "temporary": RegisterRole(
                "temporary", 1, RegisterLifetime(0, 5), minimum_register=84
            ),
            "output_column": RegisterRole(
                "output_column", 1, RegisterLifetime(0, 0), minimum_register=85
            ),
            "store_auxiliary": RegisterRole(
                "store_auxiliary", 1, RegisterLifetime(5, 5), minimum_register=85
            ),
            "serial": RegisterRole(
                "serial", 1, RegisterLifetime(0, 5), minimum_register=86
            ),
            "activation_row": RegisterRole(
                "activation_row", 1, RegisterLifetime(0, 0), minimum_register=87
            ),
        }
        order = tuple(roles)
        declared_vgprs = 88
        plan = DeterministicRegisterPlan.allocate(
            roles,
            order,
            max_registers=declared_vgprs,
        )
        assignments = {name: plan.assignment(name) for name in order}
        return cls(
            **assignments,
            register_count=plan.register_count,
            declared_vgprs=declared_vgprs,
        )


@dataclass(frozen=True)
class DecodedWeightLdsRegisterPlan:
    """Stage-aware ownership for the retained decoded 128x64 path."""

    zero_accumulator: RegisterAssignment
    sums: RegisterAssignment
    decode_scratch: RegisterAssignment
    weight_payload: RegisterAssignment
    metadata_addresses: RegisterAssignment
    staged_payload: RegisterAssignment
    c_fragments: RegisterAssignment
    low_activation_tail: RegisterAssignment
    high_activation: RegisterAssignment
    activation_scale_sum: RegisterAssignment
    scaled_dm: RegisterAssignment
    temporary: RegisterAssignment
    lds_address: RegisterAssignment
    auxiliary: RegisterAssignment
    metadata_lds_address: RegisterAssignment
    output_address: RegisterAssignment
    activation_plane_address: RegisterAssignment
    output_column: RegisterAssignment
    wave_column_base: RegisterAssignment
    wave: RegisterAssignment
    activation_base: RegisterAssignment
    lane: RegisterAssignment
    serial: RegisterAssignment
    register_count: int
    declared_vgprs: int

    @classmethod
    def allocate(cls) -> DecodedWeightLdsRegisterPlan:
        roles = {
            "zero_accumulator": RegisterRole(
                "zero_accumulator", 8, RegisterLifetime(0, 3), minimum_register=0
            ),
            "sums": RegisterRole(
                "sums", 64, RegisterLifetime(0, 5), minimum_register=8
            ),
            "decode_scratch": RegisterRole(
                "decode_scratch", 32, RegisterLifetime(1, 1), minimum_register=72
            ),
            "weight_payload": RegisterRole(
                "weight_payload", 8, RegisterLifetime(2, 3), minimum_register=72
            ),
            "metadata_addresses": RegisterRole(
                "metadata_addresses", 4, RegisterLifetime(2, 3), minimum_register=80
            ),
            "staged_payload": RegisterRole(
                "staged_payload", 64, RegisterLifetime(1, 1), minimum_register=112
            ),
            "c_fragments": RegisterRole(
                "c_fragments", 64, RegisterLifetime(2, 3), minimum_register=112
            ),
            "low_activation_tail": RegisterRole(
                "low_activation_tail",
                4,
                RegisterLifetime(2, 3),
                minimum_register=176,
            ),
            "high_activation": RegisterRole(
                "high_activation", 32, RegisterLifetime(2, 3), minimum_register=180
            ),
            "activation_scale_sum": RegisterRole(
                "activation_scale_sum",
                8,
                RegisterLifetime(2, 3),
                minimum_register=212,
            ),
            "scaled_dm": RegisterRole(
                "scaled_dm", 8, RegisterLifetime(2, 3), minimum_register=220
            ),
            "temporary": RegisterRole(
                "temporary", 1, RegisterLifetime(0, 5), minimum_register=228
            ),
            "lds_address": RegisterRole(
                "lds_address", 1, RegisterLifetime(0, 4), minimum_register=229
            ),
            "auxiliary": RegisterRole(
                "auxiliary", 2, RegisterLifetime(0, 5), minimum_register=230
            ),
            "metadata_lds_address": RegisterRole(
                "metadata_lds_address",
                1,
                RegisterLifetime(0, 4),
                minimum_register=232,
            ),
            "output_address": RegisterRole(
                "output_address", 1, RegisterLifetime(5, 5), minimum_register=232
            ),
            "activation_plane_address": RegisterRole(
                "activation_plane_address",
                1,
                RegisterLifetime(0, 4),
                minimum_register=233,
            ),
            "output_column": RegisterRole(
                "output_column", 1, RegisterLifetime(0, 4), minimum_register=234
            ),
            "wave_column_base": RegisterRole(
                "wave_column_base", 1, RegisterLifetime(0, 5), minimum_register=235
            ),
            "wave": RegisterRole(
                "wave", 1, RegisterLifetime(0, 0), minimum_register=236
            ),
            "activation_base": RegisterRole(
                "activation_base", 1, RegisterLifetime(1, 4), minimum_register=236
            ),
            "lane": RegisterRole(
                "lane", 1, RegisterLifetime(0, 5), minimum_register=237
            ),
            "serial": RegisterRole(
                "serial", 1, RegisterLifetime(0, 5), minimum_register=238
            ),
        }
        order = tuple(roles)
        declared_vgprs = 239
        plan = DeterministicRegisterPlan.allocate(
            roles,
            order,
            max_registers=declared_vgprs,
        )
        assignments = {name: plan.assignment(name) for name in order}
        return cls(
            **assignments,
            register_count=plan.register_count,
            declared_vgprs=declared_vgprs,
        )


@dataclass(frozen=True)
class PackedScaleMinimumDirectPhysicalPlan:
    activation_metadata: F16D4S4ActivationMetadata
    registers: PackedScaleMinimumDirectRegisterPlan
    resources: ForwardResourceUsage


@dataclass(frozen=True)
class DecodedWeightLdsPhysicalPlan:
    layout: DecodedLdsLayout
    registers: DecodedWeightLdsRegisterPlan
    resources: ForwardResourceUsage


@dataclass(frozen=True)
class Q6StructuredPhysicalPlan:
    layout: Q6PhysicalLayout
    resources: ForwardResourceUsage


@dataclass(frozen=True)
class Packed3BitTiledLdsPhysicalPlan:
    layout: Packed3BitTiledLdsLayout
    registers: Packed3BitTiledLdsRegisterPlan
    resources: ForwardResourceUsage


@dataclass(frozen=True)
class SignedInt8DirectPhysicalPlan:
    registers: SignedInt8DirectRegisterPlan
    resources: ForwardResourceUsage


@dataclass(frozen=True)
class SignedInt8RegisterTiledPhysicalPlan:
    registers: SignedInt8RegisterTiledRegisterPlan
    resources: ForwardResourceUsage


@dataclass(frozen=True)
class SignedInt8WaveNTiledLdsPhysicalPlan:
    layout: SignedInt8WaveNTiledLdsLayout | SignedInt8CompactDepth32TiledLdsLayout
    registers: SignedInt8WaveNTiledLdsRegisterPlan
    policy: SignedInt8TiledLdsPolicy
    resources: ForwardResourceUsage


SignedInt8SmallLdsLayout: TypeAlias = (
    SignedInt8SmallMTiledLdsLayout | SignedInt8CompactDepth32TiledLdsLayout
)


@dataclass(frozen=True)
class SignedInt8SmallMTiledLdsPhysicalPlan:
    layout: SignedInt8SmallLdsLayout
    registers: SignedInt8SmallMTiledLdsRegisterPlan
    policy: SignedInt8TiledLdsPolicy
    resources: ForwardResourceUsage


ForwardPhysicalPlan: TypeAlias = (
    PackedScaleMinimumDirectPhysicalPlan
    | DecodedWeightLdsPhysicalPlan
    | Q6StructuredPhysicalPlan
    | Packed3BitTiledLdsPhysicalPlan
    | Q3FullWeightTiledLdsPhysicalPlan
    | SignedInt8DirectPhysicalPlan
    | SignedInt8RegisterTiledPhysicalPlan
    | SignedInt8WaveNTiledLdsPhysicalPlan
    | SignedInt8SmallMTiledLdsPhysicalPlan
)


def q6_structured_physical_plan(output_tile_rows: int) -> Q6StructuredPhysicalPlan:
    if output_tile_rows not in (1, 2):
        raise ValueError("structured Q6 implements one or two output rows per wave")
    layout = Q6PhysicalLayout(output_tile_rows)
    return Q6StructuredPhysicalPlan(
        layout=layout,
        resources=ForwardResourceUsage(
            vgprs=layout.declared_vgprs,
            sgprs=layout.declared_sgprs,
            lds_bytes=layout.lds.total_bytes,
        ),
    )


def packed_3bit_tiled_lds_physical_plan() -> Packed3BitTiledLdsPhysicalPlan:
    layout = Packed3BitTiledLdsLayout()
    registers = Packed3BitTiledLdsRegisterPlan.allocate()
    return Packed3BitTiledLdsPhysicalPlan(
        layout=layout,
        registers=registers,
        resources=ForwardResourceUsage(
            vgprs=registers.declared_vgprs,
            sgprs=16,
            lds_bytes=layout.total_bytes,
        ),
    )


def q3_full_weight_tiled_lds_physical_plan() -> Q3FullWeightTiledLdsPhysicalPlan:
    layout = Q3FullWeightTiledLdsLayout()
    registers = Q3FullWeightTiledLdsRegisterPlan.allocate()
    return Q3FullWeightTiledLdsPhysicalPlan(
        layout=layout,
        registers=registers,
        resources=ForwardResourceUsage(
            vgprs=registers.declared_vgprs,
            sgprs=16,
            lds_bytes=layout.total_bytes,
        ),
    )


def derive_forward_physical_plan(spec: ForwardKernelSpec) -> ForwardPhysicalPlan:
    """Derive one complete mechanism plan without emitting instructions."""
    operand_source = spec.global_memory.operand_source
    mechanism = forward_mechanism_contract(operand_source)
    plan_kind = mechanism.physical_plan
    if plan_kind == "StructuredQ6":
        return q6_structured_physical_plan(spec.ownership.mi_wave_tile[0])
    if plan_kind == "Packed3BitTiledLds":
        if spec.geometry.work_group != (32, 4, 1) or spec.macro_tile != (128, 64):
            raise ValueError("Q3 HIP-shaped LDS control requires a 128x64 tile")
        return packed_3bit_tiled_lds_physical_plan()
    if plan_kind == "Packed3BitFullWeightTiledLds":
        if spec.geometry.work_group != (32, 4, 1) or spec.macro_tile != (128, 64):
            raise ValueError("Q3 full-weight LDS control requires a 128x64 tile")
        return q3_full_weight_tiled_lds_physical_plan()
    if plan_kind == "DecodedWeightLds":
        layout = DecodedLdsLayout.for_activation_block_bytes(
            mechanism.activation_block_bytes
        )
        registers = DecodedWeightLdsRegisterPlan.allocate()
        return DecodedWeightLdsPhysicalPlan(
            layout,
            registers,
            ForwardResourceUsage(
                registers.declared_vgprs,
                16,
                layout.total_bytes,
            ),
        )
    if plan_kind == "PackedScaleMinimumDirect":
        registers = PackedScaleMinimumDirectRegisterPlan.allocate()
        return PackedScaleMinimumDirectPhysicalPlan(
            F16D4S4ActivationMetadata(mechanism.activation_block_bytes),
            registers,
            ForwardResourceUsage(registers.declared_vgprs, 16, 0),
        )
    if plan_kind == "SignedInt8Direct":
        registers = SignedInt8DirectRegisterPlan.allocate()
        return SignedInt8DirectPhysicalPlan(
            registers,
            ForwardResourceUsage(registers.declared_vgprs, 16, 0),
        )
    if plan_kind == "SignedInt8RegisterTiled":
        wave_tile_m, wave_tile_n = spec.ownership.mi_wave_tile
        registers = SignedInt8RegisterTiledRegisterPlan.allocate(
            wave_tile_m, wave_tile_n
        )
        return SignedInt8RegisterTiledPhysicalPlan(
            registers,
            ForwardResourceUsage(registers.declared_vgprs, 16, 0),
        )
    if plan_kind == "SignedInt8WaveNTiledLds":
        if spec.geometry.depth_u not in (32, 64):
            raise ValueError("Q8 HIP-shaped LDS controls require DepthU 32 or 64")
        if spec.lds.address_hoist == "CompactDepth32WeightRows":
            layout = SignedInt8CompactDepth32TiledLdsLayout(
                activation_rows=spec.macro_tile[0],
                activation_row_stride=mechanism.activation_block_bytes,
            )
            policy = SignedInt8TiledLdsPolicy(
                "WeightThenActivation", "PairedHoistedSecondBase"
            )
        else:
            layout = SignedInt8WaveNTiledLdsLayout(
                activation_row_stride=mechanism.activation_block_bytes
            )
            policy = SignedInt8TiledLdsPolicy("Interleaved", "Scalar")
        registers = SignedInt8WaveNTiledLdsRegisterPlan.allocate()
        return SignedInt8WaveNTiledLdsPhysicalPlan(
            layout,
            registers,
            policy,
            ForwardResourceUsage(
                registers.declared_vgprs,
                16,
                layout.total_bytes,
            ),
        )
    if plan_kind != "SignedInt8SmallMTiledLds":
        raise AssertionError(f"unhandled forward physical plan {plan_kind!r}")
    macro_tile_m, macro_tile_n = spec.macro_tile
    if (
        spec.geometry.work_group != (32, 4, 1)
        or macro_tile_m not in (32, 64)
        or macro_tile_n != 64
        or spec.geometry.depth_u != 32
    ):
        raise ValueError("Q8 small-M LDS control requires MT32/MT64 x N64, DepthU 32")
    if spec.lds.address_hoist == "SmallMTile":
        layout: SignedInt8SmallLdsLayout = SignedInt8SmallMTiledLdsLayout(
            macro_tile_m,
            activation_row_stride=mechanism.activation_block_bytes,
        )
        policy = SignedInt8TiledLdsPolicy("WeightThenActivation", "Scalar")
    elif spec.lds.address_hoist == "CompactDepth32WeightRows":
        layout = SignedInt8CompactDepth32TiledLdsLayout(
            activation_rows=macro_tile_m,
            activation_row_stride=mechanism.activation_block_bytes,
        )
        policy = SignedInt8TiledLdsPolicy(
            "WeightThenActivation", "PairedHoistedSecondBase"
        )
    else:
        raise ValueError("Q8 small-M LDS control has an unsupported layout")
    registers = SignedInt8SmallMTiledLdsRegisterPlan.allocate(macro_tile_m // 16)
    return SignedInt8SmallMTiledLdsPhysicalPlan(
        layout,
        registers,
        policy,
        ForwardResourceUsage(
            registers.declared_vgprs,
            16,
            layout.total_bytes,
        ),
    )
