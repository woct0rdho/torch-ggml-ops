"""Pure physical derivation for MMQ backward assembly kernels."""

from dataclasses import dataclass

from .mmq_bwd_spec import BackwardQ3Pairing, DerivedBackwardState


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
    quant_type = state.contract.quant_type
    payload_register_count = 4 if quant_type in ("Q4_K", "Q8_0") else 8
    if quant_type in ("Q3_K", "Q4_K"):
        packed_load_count = 5 * rows
    elif quant_type == "Q6_K":
        packed_load_count = 4 * rows
    elif quant_type == "Q8_0":
        packed_load_count = (5 if decode.q8_0_extraction == "scalar" else 2) * rows
    else:
        packed_load_count = (3 if decode.q5_k_metadata_vector_load else 6) * rows

    q3_full_vopd = decode.q3_k_pairing is BackwardQ3Pairing.FULL
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
    return single_buffer * (2 if state.spec.pipeline.one_lds_buffer == 0 else 1)


def derive_backward_physical_plan(
    state: DerivedBackwardState,
) -> BackwardPhysicalPlan:
    spec = state.spec
    geometry = spec.geometry
    decoder = _derive_decoder(state)
    quant_type = state.contract.quant_type
    extended_a = (
        spec.pipeline.schedule_iter_alg in (4, 5) and geometry.matrix_instruction[5] > 2
    )
    address_register_count = (
        8
        if not extended_a
        else 6
        + geometry.matrix_instruction[5]
        + 2 * int(quant_type in ("Q3_K", "Q6_K"))
    )

    vgprs = _FirstFitRegisters(0, 255)
    m_tiles = geometry.matrix_instruction[5]
    n_tiles = geometry.matrix_instruction[6]
    accum = vgprs.allocate(8 * m_tiles * n_tiles, alignment=8)
    valu_a = vgprs.allocate(
        8 * m_tiles * spec.pipeline.prefetch_global_read,
        alignment=4,
    )
    valu_b = vgprs.allocate(
        16 * spec.pipeline.prefetch_local_read,
        alignment=4,
    )
    global_read_b = vgprs.allocate(
        decoder.payload_register_count * decoder.rows,
        alignment=4,
    )
    if quant_type == "Q3_K":
        quant_dm = vgprs.allocate(decoder.rows)
        quant_scale = vgprs.allocate(2 * decoder.rows)
    elif quant_type == "Q6_K":
        quant_dm = vgprs.allocate(decoder.rows)
        quant_scale = vgprs.allocate(decoder.rows)
    elif quant_type == "Q8_0":
        quant_dm = vgprs.allocate(decoder.rows)
        quant_scale = quant_dm
    elif quant_type == "Q5_K" and spec.decode.q5_k_metadata_vector_load:
        quant_dm = vgprs.allocate(4 * decoder.rows)
        quant_scale = quant_dm + 1
    else:
        quant_dm = vgprs.allocate(decoder.rows)
        quant_scale = vgprs.allocate(3 * decoder.rows)
    lds_address = -1
    if spec.memory.lds_swizzle_chunk_b:
        lds_address = vgprs.allocate(32 // spec.memory.lds_swizzle_chunk_b)
    address = vgprs.allocate(address_register_count, alignment=2)
    temporary_count = max(
        11
        if quant_type == "Q3_K" and spec.decode.q3_k_extraction == "packed"
        else 7 + 2 * decoder.rows
        if quant_type == "Q6_K" and spec.decode.q6_k_extraction == "packed_vopd"
        else 7,
        (5 if spec.decode.q8_0_extraction == "packed_vopd" else 3) + 2 * decoder.rows,
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
        address=address_plan,
        lds=lds,
        resources=BackwardResourcePlan(
            total_vgprs=registers.total_vgprs,
            total_sgprs=registers.total_sgprs,
            lds_num_bytes=lds_num_bytes,
        ),
    )
