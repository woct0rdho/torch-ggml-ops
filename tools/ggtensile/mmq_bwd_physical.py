"""Pure physical derivation for MMQ backward assembly kernels."""

from dataclasses import dataclass

from .mmq_bwd_spec import (
    BackwardQ3Pairing,
    BackwardQ4DecodeSchedule,
    DerivedBackwardState,
)


@dataclass(frozen=True)
class BackwardRegisterPlan:
    accum: int
    valu_a: int
    valu_b: int
    global_read_b: int
    quant_dm: int
    quant_scale: int
    lds_address: int
    address: int
    temporary: int
    serial: int
    total_vgprs: int
    kernarg: int
    loop_counter: int
    block_offset: int
    input_half: int
    scalar_temporary: int
    total_sgprs: int


@dataclass(frozen=True)
class BackwardDecoderPlan:
    threads: int
    rows: int
    payload_register_count: int
    packed_load_count: int
    q3_full_vopd: bool


@dataclass(frozen=True)
class BackwardQ4DecodePlan:
    schedule: BackwardQ4DecodeSchedule
    value_registers: tuple[int, ...]
    rounding_registers: tuple[int, ...]

    @property
    def dependency_width(self) -> int:
        return self.schedule.dependency_width


@dataclass(frozen=True)
class BackwardAddressPlan:
    uses_extended_a_pointer_state: bool
    register_count: int
    lds: int
    quant_shift: int
    q3_low_shift: int


@dataclass(frozen=True)
class BackwardLdsPlan:
    num_bytes: int
    row_stride_bytes: int
    swizzle_chunk: int

    def decoded_store_location(
        self,
        registers: BackwardRegisterPlan,
        address: BackwardAddressPlan,
        element: int,
        row: int,
        k_span: int,
    ) -> tuple[int, int]:
        lds_offset = self.row_stride_bytes * element + 2 * k_span * row
        if not self.swizzle_chunk:
            return address.lds, lds_offset

        residues = 32 // self.swizzle_chunk
        residue = element % residues
        logical_k = row * k_span
        lds_offset = self.row_stride_bytes * element + 64 * (logical_k // 32)
        if logical_k % 32:
            lds_offset += 32 if residue < residues // 2 else -32
        return registers.lds_address + residue, lds_offset


@dataclass(frozen=True)
class BackwardResourcePlan:
    total_vgprs: int
    total_sgprs: int
    lds_num_bytes: int
    private_segment_bytes: int = 0


@dataclass(frozen=True)
class BackwardPhysicalPlan:
    registers: BackwardRegisterPlan
    decoder: BackwardDecoderPlan
    q4_decode: BackwardQ4DecodePlan
    address: BackwardAddressPlan
    lds: BackwardLdsPlan
    resources: BackwardResourcePlan


class _FirstFitRegisters:
    def __init__(self, first: int, last: int) -> None:
        self._first = first
        self._last = last
        self._used: set[int] = set()

    def allocate(self, count: int, *, alignment: int = 1) -> int:
        for start in range(self._first, self._last - count + 2):
            if start % alignment:
                continue
            registers = range(start, start + count)
            if not any(register in self._used for register in registers):
                self._used.update(registers)
                return start
        raise ValueError(f"cannot allocate {count} registers aligned to {alignment}")


def _derive_decoder(state: DerivedBackwardState) -> BackwardDecoderPlan:
    spec = state.spec
    geometry = spec.geometry
    decode = spec.decode
    decoder_threads = min(geometry.num_threads, 128)
    rows = (
        geometry.depth_u
        * geometry.macro_tile1
        // (decoder_threads * decode.decoder_width)
    )
    decoder_capability = state.contract.mechanism.decoder
    payload_register_count = decoder_capability.payload_register_count
    packed_load_count = decoder_capability.packed_load_count(decode, rows)

    q3_full_vopd = decode.q3.pairing is BackwardQ3Pairing.Full
    return BackwardDecoderPlan(
        threads=decoder_threads,
        rows=rows,
        payload_register_count=payload_register_count,
        packed_load_count=packed_load_count,
        q3_full_vopd=q3_full_vopd,
    )


def _derive_lds_num_bytes(state: DerivedBackwardState) -> int:
    geometry = state.spec.geometry
    memory = state.spec.memory
    if memory.lds_block_size_per_pad_b:
        bytes_unpadded = 2 * geometry.macro_tile1 * geometry.depth_u
        pad_periods = bytes_unpadded // memory.lds_block_size_per_pad_b
        single_buffer = bytes_unpadded + 2 * memory.lds_pad_b * pad_periods
    else:
        single_buffer = 2 * (geometry.depth_u + memory.lds_pad_b) * geometry.macro_tile1
    return single_buffer * state.spec.pipeline.lds_buffering.buffer_count


def derive_backward_physical_plan(
    state: DerivedBackwardState,
) -> BackwardPhysicalPlan:
    spec = state.spec
    geometry = spec.geometry
    decoder = _derive_decoder(state)
    mechanism = state.contract.mechanism
    decoder_capability = mechanism.decoder
    extended_a = mechanism.address.uses_extended_a(
        spec.pipeline.schedule, geometry.matrix_instruction[5]
    )
    address_register_count = mechanism.address.register_count(
        spec.pipeline.schedule, geometry.matrix_instruction[5]
    )

    vgprs = _FirstFitRegisters(0, 255)
    m_tiles = geometry.matrix_instruction[5]
    n_tiles = geometry.matrix_instruction[6]
    accum = vgprs.allocate(8 * m_tiles * n_tiles, alignment=8)
    valu_a = vgprs.allocate(
        8 * m_tiles * spec.pipeline.global_read_prefetch,
        alignment=4,
    )
    valu_b = vgprs.allocate(
        16 * spec.pipeline.local_read_prefetch,
        alignment=4,
    )
    global_read_b = vgprs.allocate(
        decoder.payload_register_count * decoder.rows,
        alignment=4,
    )
    quant_shape = mechanism.quant_register_shape.for_decode(spec.decode)
    quant_dm = vgprs.allocate(quant_shape.dm_registers_per_row * decoder.rows)
    if quant_shape.scale_alias_offset is not None:
        quant_scale = quant_dm + quant_shape.scale_alias_offset
    else:
        quant_scale = vgprs.allocate(quant_shape.scale_registers_per_row * decoder.rows)
    lds_address = -1
    if spec.memory.lds_swizzle_chunk_b:
        lds_address = vgprs.allocate(32 // spec.memory.lds_swizzle_chunk_b)
    address = vgprs.allocate(address_register_count, alignment=2)
    temporary_count = decoder_capability.temporary_register_count(
        spec.decode, decoder.rows
    )
    temporary = vgprs.allocate(temporary_count)
    serial = vgprs.allocate(1)

    sgprs = _FirstFitRegisters(5, 63)
    kernarg = sgprs.allocate(6, alignment=2)
    loop_counter = sgprs.allocate(1)
    block_offset = sgprs.allocate(1)
    input_half = sgprs.allocate(1)
    scalar_temporary = sgprs.allocate(2)
    registers = BackwardRegisterPlan(
        accum=accum,
        valu_a=valu_a,
        valu_b=valu_b,
        global_read_b=global_read_b,
        quant_dm=quant_dm,
        quant_scale=quant_scale,
        lds_address=lds_address,
        address=address,
        temporary=temporary,
        serial=serial,
        total_vgprs=max(
            serial + 1,
            temporary + temporary_count,
            address + address_register_count,
        ),
        kernarg=kernarg,
        loop_counter=loop_counter,
        block_offset=block_offset,
        input_half=input_half,
        scalar_temporary=scalar_temporary,
        total_sgprs=scalar_temporary + 2,
    )
    q4_schedule = spec.decode.q4.schedule
    if q4_schedule is BackwardQ4DecodeSchedule.DependencyBatch4:
        q4_values = tuple(registers.valu_b + 2 * slot for slot in range(4))
    else:
        q4_values = (registers.temporary + 1 + 2 * decoder.rows,)
    q4_decode = BackwardQ4DecodePlan(
        schedule=q4_schedule,
        value_registers=q4_values,
        rounding_registers=tuple(value + 1 for value in q4_values),
    )
    lds_address_register = (
        address + 4 + geometry.matrix_instruction[5] if extended_a else address + 6
    )
    quant_shift = lds_address_register + 1
    q3_low_shift = address + 2 if not extended_a else quant_shift + 1
    address_plan = BackwardAddressPlan(
        uses_extended_a_pointer_state=extended_a,
        register_count=address_register_count,
        lds=lds_address_register,
        quant_shift=quant_shift,
        q3_low_shift=q3_low_shift,
    )
    lds_num_bytes = _derive_lds_num_bytes(state)
    lds = BackwardLdsPlan(
        num_bytes=lds_num_bytes,
        row_stride_bytes=2 * (geometry.depth_u + spec.memory.lds_pad_b),
        swizzle_chunk=spec.memory.lds_swizzle_chunk_b,
    )
    return BackwardPhysicalPlan(
        registers=registers,
        decoder=decoder,
        q4_decode=q4_decode,
        address=address_plan,
        lds=lds,
        resources=BackwardResourcePlan(
            total_vgprs=registers.total_vgprs,
            total_sgprs=registers.total_sgprs,
            lds_num_bytes=lds_num_bytes,
        ),
    )
