"""Structured Q6_K semantic schedule and forward body lowering."""

from dataclasses import dataclass
from typing import ClassVar

from rocisa import code  # ty: ignore[unresolved-import]

from .kernel_writer_assembly import (
    Assembly,
    DeterministicRegisterPlan,
    RegisterAssignment,
    RegisterLifetime,
    RegisterRole,
)
from .mmq_fwd_lowering import ForwardLoweringContext
from .mmq_fwd_physical import (
    Q6AddressAdd,
    Q6AddressBatch,
    Q6DecodeRegisterPlan,
    Q6DependencyDelay,
    Q6GlobalRead,
    Q6HalfRegister,
    Q6Immediate,
    Q6OffsetAddress,
    Q6PhysicalLayout,
    Q6Sgpr,
    Q6Vgpr,
    q6_structured_physical_plan,
)
from .mmq_fwd_spec import (
    Q6DotPhase,
    Q6ForwardSchedule,
    Q6LdsLayout,
    Q6SignedDecodeSpec,
    QuantForwardSemantics,
    q6_schedule_from_kernel_spec,
)

_Q6_SEMANTICS = QuantForwardSemantics.for_quant_type("Q6_K")
_Q6_PACKED_BLOCK_BYTES = max(
    plane.byte_offset + plane.byte_count for plane in _Q6_SEMANTICS.payload_planes
)
_Q6_SCALE_OFFSET = _Q6_SEMANTICS.payload_plane("scales").byte_offset
_Q6_FACTOR_OFFSET = _Q6_SEMANTICS.payload_plane("d").byte_offset


class Q6ScheduleEmitter:
    """Semantic lowering for physically ordered Q6 schedule components."""

    def __init__(self, schedule: "Q6ForwardSchedule", name: str) -> None:
        if schedule.semantic_policy.latency not in {
            "SerializedDependencyDistance",
            "WavefrontDependencyDistance",
        }:
            raise ValueError("unsupported Q6 latency-lowering policy")
        if schedule.semantic_policy.pairing != "DependencyCompatibleDualIssue":
            raise ValueError("unsupported Q6 dual-issue pairing policy")
        if schedule.semantic_policy.wait != "ProducerFirstUse":
            raise ValueError("unsupported Q6 wait-lowering policy")
        self.schedule = schedule
        self.name = name
        self.assembly = Assembly(indent="\t")
        self._global_read_indices: dict[int, int] = {}
        self._global_read_count = 0
        self._vmem_wait_count: int | None = None
        self._local_write_sources: dict[int, int] = {}

    def annotation(self, text: str) -> None:
        self.assembly.line(text)

    def instruction(self, text: str) -> None:
        self.assembly.inst(text)

    def inst(self, opcode: str, operands: str = "") -> None:
        suffix = f" {operands}" if operands else ""
        self.assembly.inst(f"{opcode}{suffix}")

    def dual(
        self,
        left_opcode: str,
        left_operands: str,
        right_opcode: str,
        right_operands: str,
    ) -> None:
        self.inst(
            left_opcode,
            f"{left_operands} :: {right_opcode} {right_operands}",
        )

    def dual_zero(self, register: int, opcode: str, operands: str) -> None:
        self.dual("v_dual_mov_b32", f"v{register}, 0", opcode, operands)

    def scalar_argument_loads(self) -> None:
        self.inst("s_clause", "0x2")
        self.inst("s_load_b128", "s[12:15], s[0:1], 0x0")
        self.inst("s_load_b64", "s[16:17], s[0:1], 0x10")
        self.inst("s_load_b32", "s22, s[0:1], 0x20")

    def add_u64(self, address: Q6AddressAdd) -> None:
        self.inst(
            "v_add_co_u32",
            f"v{address.destination}, vcc_lo, {address.left}, {address.right}",
        )
        if address.delay_after_low is not None:
            self.dependency_delay(address.delay_after_low)
        self.inst(
            "v_add_co_ci_u32_e64",
            f"v{address.destination + 1}, null, {address.high_left}, "
            f"{address.high}, vcc_lo",
        )
        if address.delay_after_high is not None:
            self.dependency_delay(address.delay_after_high)

    def add_u64_batch(self, addresses: tuple[Q6AddressAdd, ...]) -> None:
        for address in addresses:
            self.add_u64(address)

    def add_offset_addresses(
        self,
        addresses: Q6AddressBatch,
        dependency_delay: Q6DependencyDelay,
    ) -> None:
        self.add_u64_batch(
            tuple(
                Q6AddressAdd(
                    address.destination,
                    Q6Immediate(address.byte_offset, hexadecimal=True),
                    Q6Vgpr(address.base_register),
                    Q6Vgpr(address.base_register + 1),
                    delay_after_low=(
                        dependency_delay if address.delay_after_low else None
                    ),
                )
                for address in addresses
            )
        )

    def global_read_clause(self, reads: tuple[Q6GlobalRead, ...]) -> None:
        self.inst("s_clause", hex(len(reads) - 1))
        for read in reads:
            opcode = (
                "global_load_d16_b16" if read.width_bits == 16 else "global_load_b32"
            )
            offset = f" offset:{read.offset}" if read.offset else ""
            self.inst(
                opcode,
                f"v{read.destination_register}, "
                f"v[{read.address_register}:{read.address_register + 1}], off{offset}",
            )
            destination = read.destination_register
            self._global_read_indices[destination] = self._global_read_count
            if read.local_write_slot is not None:
                if read.local_write_slot in self._local_write_sources:
                    raise ValueError(
                        f"duplicate Q6 local-write slot: {read.local_write_slot}"
                    )
                self._local_write_sources[read.local_write_slot] = destination
            self._global_read_count += 1
            self._vmem_wait_count = None

    def wait_for_global_sources(self, sources: tuple[int, ...]) -> None:
        if not sources:
            raise ValueError("Q6 LDS write requires at least one global source")
        missing_source = next(
            (source for source in sources if source not in self._global_read_indices),
            None,
        )
        if missing_source is not None:
            raise ValueError(f"Q6 LDS source has no global producer: v{missing_source}")
        latest = max(self._global_read_indices[source] for source in sources)
        required = self._global_read_count - latest - 1
        if self._vmem_wait_count is None or required < self._vmem_wait_count:
            self.wait_vmem(required)

    def wait_vmem(self, count: int) -> None:
        self.inst("s_waitcnt", f"vmcnt({count})")
        self._vmem_wait_count = count

    def dependency_delay(self, delay: Q6DependencyDelay) -> None:
        if self.schedule.dependency_delay_mode == "Explicit":
            self.inst("s_delay_alu", str(delay))

    def invalidate_global_cache(self) -> None:
        if self.schedule.global_read_cache_policy == "InvalidateL0":
            self.inst("buffer_gl0_inv")

    def shift_right_dword(
        self,
        destination: int,
        shift: int | Q6Vgpr,
        source: int,
        *,
        arithmetic: bool = False,
    ) -> None:
        opcode = "v_ashrrev_i32_e32" if arithmetic else "v_lshrrev_b32_e32"
        self.inst(opcode, f"v{destination}, {shift}, v{source}")

    def shift_left_dword(self, destination: int, shift: int, source: int) -> None:
        self.inst("v_lshlrev_b32_e32", f"v{destination}, {shift}, v{source}")

    def mask_dword(self, destination: int, mask: str, source: int) -> None:
        self.inst("v_and_b32_e32", f"v{destination}, {mask}, v{source}")

    def merge_q6_dword(self, destination: int, high: int, low: int) -> None:
        self.inst(
            "v_and_or_b32",
            f"v{destination}, 0x30303030, v{high}, v{low}",
        )

    def store_bf16_clause(self, address: str, registers: tuple[int, ...]) -> None:
        if len(registers) != self.schedule.store_vector_width:
            raise ValueError(
                "Q6 store clause does not match StoreVectorWidth: "
                f"{len(registers)} != {self.schedule.store_vector_width}"
            )
        self.inst("s_clause", hex(self.schedule.store_vector_width - 1))
        for index, register in enumerate(registers):
            offset = f" offset:{4 * index}" if index else ""
            self.inst(
                "global_store_d16_hi_b16",
                f"{address}, v{register}, off{offset}",
            )

    def local_write_pair(
        self,
        address: int,
        first: int,
        second: int,
        offset0: int,
        offset1: int,
        *,
        stride64: bool = True,
    ) -> None:
        opcode = "ds_store_2addr_stride64_b32" if stride64 else "ds_store_2addr_b32"
        first_offset = f" offset0:{offset0}" if offset0 else ""
        second_offset = f" offset1:{offset1}" if offset1 else ""
        self.inst(
            opcode,
            f"v{address}, v{first}, v{second}{first_offset}{second_offset}",
        )

    def local_write(self, address: int, value: int, offset: int = 0) -> None:
        suffix = f" offset:{offset}" if offset else ""
        self.inst("ds_store_b32", f"v{address}, v{value}{suffix}")

    def local_write_cooperative_pair(
        self,
        lds: Q6LdsLayout,
        address: int,
        first: int,
        second: int,
        pair: int,
    ) -> None:
        role = lds.cooperative_write_role(pair)
        self.local_write_pair(address, first, second, role.offset0, role.offset1)

    def commit_cooperative_global_reads(
        self,
        lds: Q6LdsLayout,
        address: int,
    ) -> None:
        slots = tuple(sorted(self._local_write_sources))
        if slots != tuple(range(len(slots))) or len(slots) % 2:
            raise ValueError(
                "Q6 cooperative local-write slots must be contiguous pairs"
            )
        for pair in range(len(slots) // 2):
            first = self._local_write_sources[2 * pair]
            second = self._local_write_sources[2 * pair + 1]
            self.wait_for_global_sources((first, second))
            self.local_write_cooperative_pair(
                lds,
                address,
                first,
                second,
                pair,
            )

    def local_read_pair(
        self,
        destination: tuple[int, int],
        address: int,
        offset0: int,
        offset1: int,
    ) -> None:
        first_offset = f" offset0:{offset0}" if offset0 else ""
        second_offset = f" offset1:{offset1}" if offset1 else ""
        address_count = self.schedule.local_read_vector_width
        self.inst(
            f"ds_load_{address_count}addr_b32",
            f"v[{destination[0]}:{destination[1]}], v{address}"
            f"{first_offset}{second_offset}",
        )

    def commit_local_stage(
        self,
        dual_moves: tuple[tuple[int, int], tuple[int, int]],
        move: tuple[int, int],
        reads: tuple[tuple[tuple[int, int], int, int, int], ...],
    ) -> None:
        self.dual(
            "v_dual_mov_b32",
            f"v{dual_moves[0][0]}, v{dual_moves[0][1]}",
            "v_dual_mov_b32",
            f"v{dual_moves[1][0]}, v{dual_moves[1][1]}",
        )
        self.inst("v_mov_b32_e32", f"v{move[0]}, v{move[1]}")
        self.inst("s_waitcnt", "lgkmcnt(0)")
        self.inst("s_barrier")
        self.invalidate_global_cache()
        for destination, address, offset0, offset1 in reads:
            self.local_read_pair(destination, address, offset0, offset1)

    def module(self) -> code.Module:
        module = code.Module(self.name)
        module.add(code.TextBlock(self.assembly.text()))
        return module


def _q6_epilogue_scratch_base(
    layout: Q6PhysicalLayout,
    dependency_width: int,
    vgpr_count: int,
) -> int:
    lifetime = RegisterLifetime(0, 0)
    persistent = RegisterRole("persistent", layout.scratch_base, lifetime)
    scratch = RegisterRole("bf16_scratch", 2 * dependency_width, lifetime)
    plan = DeterministicRegisterPlan.allocate(
        {scratch.name: scratch},
        (scratch.name,),
        max_registers=vgpr_count,
        reserved=(RegisterAssignment(persistent, 0),),
    )
    return plan.assignment(scratch.name).first_register


def _q6_physical_layout(schedule: Q6ForwardSchedule) -> Q6PhysicalLayout:
    if schedule.semantic_policy.traversal not in {
        "OutputRoleGroupMajor",
        "OutputRoleWavefront",
    }:
        raise ValueError("unsupported Q6 output traversal policy")
    if schedule.semantic_policy.clustering not in {
        "StageDependencyOrder",
        "RowBatchedDecodeOrder",
    }:
        raise ValueError("unsupported Q6 semantic-stage clustering policy")
    if schedule.semantic_policy.pressure != "ExplicitRoleLifetime":
        raise ValueError("unsupported Q6 register-pressure policy")
    mi_wave_tile_m = schedule.mi_wave_tile[0]
    if mi_wave_tile_m not in (1, 2):
        raise ValueError(
            f"unsupported Q6 physical layout for MIWaveTileM={mi_wave_tile_m}"
        )
    return q6_structured_physical_plan(mi_wave_tile_m).layout


def _q6_dot_fragment_register(layout: Q6PhysicalLayout, tile: int) -> tuple[int, int]:
    start = layout.first_fragment_base + 8 * tile
    if tile == layout.tile_count - 1:
        start += 4
    return start, start + 3


def _q6_emit_dot_fragment_reads(
    emitter: Q6ScheduleEmitter,
    phase: Q6DotPhase,
    layout: Q6PhysicalLayout,
) -> None:
    shift = phase.register_shift
    weight_address = layout.weight_address_base - shift
    activation_address = layout.activation_address_base - shift
    product_read_address = layout.product_base + 8 - shift
    emitter.instruction(
        f"v_dual_mov_b32 v{layout.product_base + 7 - shift}, s11 :: "
        f"v_dual_add_nc_u32 v{product_read_address}, "
        f"{hex(layout.lds.stage_stride_bytes + 128 * shift)}, v{weight_address}"
    )
    first_address_offset = layout.lds.first_address_offset
    emitter.instruction(
        f"v_dual_mov_b32 v{layout.product_base - shift}, s4 :: "
        f"v_dual_add_nc_u32 v{product_read_address + 1}, "
        f"{hex(first_address_offset)}, v{activation_address}"
    )
    direct_read_after = min(3, layout.tile_count - 2)
    for tile in range(1, layout.tile_count - 1):
        emitter.instruction(
            f"v_add_nc_u32_e32 v{product_read_address + 1 + tile}, "
            f"{hex(first_address_offset + 2304 * tile)}, v{activation_address}"
        )
        if tile == direct_read_after:
            direct_offset = layout.lds.direct_read_offset
            emitter.instruction(
                f"ds_load_2addr_b64 v[{layout.first_fragment_base - shift}:"
                f"{layout.first_fragment_base + 3 - shift}], "
                f"v{activation_address} offset0:{direct_offset} "
                f"offset1:{direct_offset + 1}"
            )
    weight_fragment_base = layout.product_base + 8 * (layout.tile_count + 1)
    emitter.instruction(
        f"ds_load_2addr_b64 v[{weight_fragment_base - shift}:"
        f"{weight_fragment_base + 3 - shift}], v{product_read_address} offset1:1"
    )
    for tile in range(1, layout.tile_count):
        destination, end = _q6_dot_fragment_register(layout, tile)
        emitter.instruction(
            f"ds_load_2addr_b64 v[{destination - shift}:{end - shift}], "
            f"v{product_read_address + tile} offset1:1"
        )


def _q6_emit_dot_scale_and_loop_state(
    emitter: Q6ScheduleEmitter,
    phase: Q6DotPhase,
    layout: Q6PhysicalLayout,
) -> int:
    shift = phase.register_shift
    row_pointer = layout.row_pointer_base if shift == 0 else 10
    scale_offset = layout.lds.scale_read_base
    for element in range(8):
        emitter.instruction(
            f"ds_load_i8 v{layout.scale_base + element - shift}, v{row_pointer} "
            f"offset:{scale_offset + 8 * shift + 608 * element}"
        )
    emitter.instruction("s_add_i32 s18, s18, 4")
    for pair_index, (first, second, first_s, second_s) in enumerate(
        (
            (6, 5, 10, 9),
            (4, 3, 8, 7),
            (2, 1, 6, 5),
        )
    ):
        if pair_index == 1:
            emitter.instruction("s_lshr_b32 s19, s18, 1")
        if pair_index == 2:
            emitter.instruction("s_and_b32 s19, s19, 0x7ffffffc")
        emitter.instruction(
            f"v_dual_mov_b32 v{layout.product_base + first - shift}, s{first_s} :: "
            f"v_dual_mov_b32 v{layout.product_base + second - shift}, s{second_s}"
        )
    factor_address = layout.product_base + 8 - shift
    factor_base_address = 23 + 27 * layout.output_tile_rows
    emitter.instruction(
        f"v_add_nc_u32_e32 v{factor_address}, s19, v{factor_base_address}"
    )
    for pair in range(layout.tile_count // 2):
        role = layout.lds.factor_role(pair)
        emitter.instruction(
            f"ds_load_2addr_stride64_b32 v[{layout.factor_base + 2 * pair - shift}:"
            f"{layout.factor_base + 2 * pair + 1 - shift}], v{factor_address} "
            f"offset0:{role.offset0} offset1:{role.offset1}"
        )
    if layout.output_tile_rows == 1:
        emitter.instruction("s_cmp_lt_u32 s18, 28")
    return row_pointer


def _q6_emit_dot_wmma_and_integer_products(
    emitter: Q6ScheduleEmitter,
    phase: Q6DotPhase,
    layout: Q6PhysicalLayout,
) -> None:
    shift = phase.register_shift
    wmma_output_base = layout.product_base + 8 - shift
    weight_fragment_base = layout.product_base + 8 * (layout.tile_count + 1) - shift
    wait_start = 7 + 2 * layout.output_tile_rows + layout.tile_count
    for tile in range(layout.tile_count):
        emitter.instruction(f"s_waitcnt lgkmcnt({wait_start - tile})")
        fragment_start, fragment_end = _q6_dot_fragment_register(layout, tile)
        emitter.instruction(
            f"{phase.schedule.wmma_opcode} "
            f"v[{wmma_output_base + 8 * tile}:{wmma_output_base + 8 * tile + 7}], "
            f"v[{weight_fragment_base}:{weight_fragment_base + 3}], "
            f"v[{fragment_start - shift}:{fragment_end - shift}], "
            f"v[{layout.product_base - shift}:{layout.product_base + 7 - shift}] "
            "neg_lo:[1,1,0]"
        )
    product_base = layout.product_base - shift
    for element in range(8):
        emitter.instruction(
            f"s_waitcnt lgkmcnt({7 + layout.output_tile_rows * 2 - element})"
        )
        emitter.instruction(
            f"v_mul_lo_u32 v{product_base + element}, "
            f"v{wmma_output_base + element}, v{layout.scale_base + element - shift}"
        )
    for tile in range(1, layout.tile_count):
        for element in range(8):
            destination = product_base + 8 * tile + element
            emitter.instruction(
                f"v_mul_lo_u32 v{destination}, v{wmma_output_base + 8 * tile + element}, "
                f"v{layout.scale_base + element - shift}"
            )
    for register in range(product_base, product_base + 8 * layout.tile_count):
        emitter.instruction(f"v_cvt_f32_i32_e32 v{register}, v{register}")


def _q6_dot_product_scale(
    phase: Q6DotPhase, layout: Q6PhysicalLayout, local: int
) -> int:
    return 3 + local % 8 - phase.register_shift


def _q6_emit_dot_product_scale(
    emitter: Q6ScheduleEmitter,
    phase: Q6DotPhase,
    layout: Q6PhysicalLayout,
    local: int,
    row_pointer: int,
    pointer: str | None = None,
) -> None:
    register = layout.product_base - phase.register_shift + local
    operands = (
        f"v{register}, v{_q6_dot_product_scale(phase, layout, local)}, v{register}"
    )
    if pointer is None:
        emitter.inst("v_mul_f32_e32", operands)
    else:
        emitter.inst("v_dual_mul_f32", f"{operands} :: {pointer}")


def _q6_emit_dot_product_scales(
    emitter: Q6ScheduleEmitter,
    phase: Q6DotPhase,
    layout: Q6PhysicalLayout,
    row_pointer: int,
) -> None:
    shift = phase.register_shift
    product_base = layout.product_base - shift
    product_count = 8 * layout.tile_count
    weight_pointer = layout.weight_address_base - shift
    activation_pointer = layout.activation_address_base - shift
    if layout.output_tile_rows == 1:
        pointer_locals = (2, 3, 4)
    else:
        pointer_locals = (6, 3, 8)
    pointers = (
        f"v_dual_add_nc_u32 v{activation_pointer}, 16, v{activation_pointer}",
        f"v_dual_add_nc_u32 v{weight_pointer}, 16, v{weight_pointer}",
        f"v_dual_add_nc_u32 v{row_pointer}, 1, v{row_pointer}",
    )
    for local, pointer in zip(pointer_locals, pointers, strict=True):
        _q6_emit_dot_product_scale(emitter, phase, layout, local, row_pointer, pointer)
    emitter.instruction(
        f"v_dual_mul_f32 v{product_base}, v{_q6_dot_product_scale(phase, layout, 0)}, v{product_base} :: "
        f"v_dual_mul_f32 v{product_base + 1}, v{_q6_dot_product_scale(phase, layout, 1)}, v{product_base + 1}"
    )
    if layout.output_tile_rows == 1:
        pair_start = 5
    else:
        _q6_emit_dot_product_scale(emitter, phase, layout, 2, row_pointer)
        emitter.instruction(
            f"v_dual_mul_f32 v{product_base + 4}, v{_q6_dot_product_scale(phase, layout, 4)}, v{product_base + 4} :: "
            f"v_dual_mul_f32 v{product_base + 5}, v{_q6_dot_product_scale(phase, layout, 5)}, v{product_base + 5}"
        )
        if shift == 0:
            emitter.instruction(
                f"v_dual_mul_f32 v{product_base + 7}, v{_q6_dot_product_scale(phase, layout, 7)}, v{product_base + 7} :: "
                f"v_dual_mul_f32 v{product_base + 10}, v{_q6_dot_product_scale(phase, layout, 10)}, v{product_base + 10}"
            )
            _q6_emit_dot_product_scale(emitter, phase, layout, 9, row_pointer)
            pair_start = 11
        else:
            _q6_emit_dot_product_scale(emitter, phase, layout, 7, row_pointer)
            pair_start = 9
    for local in range(pair_start, product_count - 1, 2):
        left = product_base + local
        right = left + 1
        emitter.instruction(
            f"v_dual_mul_f32 v{left}, v{_q6_dot_product_scale(phase, layout, local)}, v{left} :: "
            f"v_dual_mul_f32 v{right}, v{_q6_dot_product_scale(phase, layout, local + 1)}, v{right}"
        )
    final = product_base + product_count - 1
    emitter.instruction(
        f"v_mul_f32_e32 v{final}, v{_q6_dot_product_scale(phase, layout, product_count - 1)}, v{final}"
    )


def _q6_emit_dot_accumulation(
    emitter: Q6ScheduleEmitter,
    phase: Q6DotPhase,
    layout: Q6PhysicalLayout,
) -> None:
    shift = phase.register_shift
    product_base = layout.product_base - shift
    group_count = layout.tile_count // 2
    if layout.output_tile_rows == 1:
        emitter.instruction("s_waitcnt lgkmcnt(1)")
        wait_counts = (None, 0)
    else:
        wait_counts = tuple(3 - group for group in range(group_count))
    for group in range(group_count):
        wait = wait_counts[group]
        if wait is not None:
            emitter.instruction(f"s_waitcnt lgkmcnt({wait})")
        factor_left = layout.factor_base + 2 * group - shift
        factor_right = factor_left + 1
        for role in layout.registers.group_output_roles(group):
            emitter.instruction(
                f"v_dual_fmac_f32 v{role.left_register}, v{factor_left}, "
                f"v{product_base + role.left_product_index} :: "
                f"v_dual_fmac_f32 v{role.right_register}, v{factor_right}, "
                f"v{product_base + role.right_product_index}"
            )
    if layout.output_tile_rows == 2:
        emitter.instruction("s_cmp_lt_u32 s18, 28")


def _emit_q6_dot_phase(schedule: Q6ForwardSchedule, phase_index: int) -> code.Module:
    phase = schedule.phase(phase_index)
    layout = _q6_physical_layout(schedule)
    shift = phase.register_shift
    label = 2 + 2 * shift
    suffix = (
        ".i" if shift == 0 else ("89.i" if layout.output_tile_rows == 1 else "125.i")
    )
    emitter = Q6ScheduleEmitter(schedule, f"q6_dot_phase_{phase.phase}")
    emitter.annotation(
        f".LBB0_{label}:                                ; %.preheader8.i.i{suffix}"
    )
    emitter.annotation(
        "                                        ;   Parent Loop BB0_1 Depth=1"
    )
    emitter.annotation(
        "                                        ; =>  This Inner Loop Header: Depth=2"
    )
    _q6_emit_dot_fragment_reads(emitter, phase, layout)
    row_pointer = _q6_emit_dot_scale_and_loop_state(emitter, phase, layout)
    _q6_emit_dot_wmma_and_integer_products(emitter, phase, layout)
    _q6_emit_dot_product_scales(emitter, phase, layout, row_pointer)
    _q6_emit_dot_accumulation(emitter, phase, layout)
    emitter.instruction(f"s_cbranch_scc1 .LBB0_{label}")
    return emitter.module()


def _emit_q6_scheduled_body(
    schedule: Q6ForwardSchedule,
    blocks_per_weight_row: int = 8,
) -> code.Module:
    if blocks_per_weight_row <= 0:
        raise ValueError("Q6 blocks-per-weight-row must be positive")
    if len(schedule.dot_register_shifts) != 2:
        raise ValueError("structured Q6 currently requires exactly two dot phases")
    body = code.Module("Q6StructuredDecoded")
    body.add(_q6_lane_and_address_setup(schedule))
    body.add(_q6_cooperative_global_loads(schedule))
    body.add(_q6_packed_decode(schedule))
    body.add(_q6_decoded_lds_writes(schedule))
    body.add(_q6_first_stage_barrier(schedule))
    body.add(_emit_q6_dot_phase(schedule, 0))
    body.add(_q6_second_stage_loads(schedule))
    body.add(_emit_q6_dot_phase(schedule, 1))
    body.add(_q6_bf16_epilogue(schedule, blocks_per_weight_row))
    return body


def _q6_emit_lane_setup_annotations(
    emitter: Q6ScheduleEmitter,
    layout: Q6PhysicalLayout,
) -> None:
    delay_annotation = (
        "// J64 omits dependency-delay hints selected by the J128 policy."
        if layout.output_tile_rows == 1
        else "// J128 retains dependency-delay hints selected by the schedule policy."
    )
    for annotation in (
        "// Deterministic Q6_K lowering from the project-owned MMQ semantics.",
        "// ScheduleIterAlg=3 uses explicit prefetch, decode, dot, and BF16 pipelines.",
        delay_annotation,
        "; %bb.0:",
    ):
        emitter.annotation(annotation)


def _q6_emit_scalar_setup_tail(
    emitter: Q6ScheduleEmitter,
    schedule: Q6ForwardSchedule,
) -> None:
    emitter.inst("s_mov_b32", "s4, 0")
    emitter.inst("s_lshl_b32", f"s3, s3, {schedule.macro_tile0.bit_length() - 1}")
    emitter.inst("s_lshl_b32", "s20, s2, 9")
    emitter.inst("s_waitcnt", "lgkmcnt(0)")
    emitter.inst("s_lshl_b32", "s21, s22, 1")
    emitter.inst("s_mul_i32", "s22, s22, 36")
    for register in (*range(5, 12), 23):
        emitter.inst("s_mov_b32", f"s{register}, s4")


def _q6_lane_and_address_setup(schedule: Q6ForwardSchedule) -> code.Module:
    emitter = Q6ScheduleEmitter(schedule, "lane_and_address_setup")
    layout = _q6_physical_layout(schedule)
    _q6_emit_lane_setup_annotations(emitter, layout)
    if layout.output_tile_rows == 1:
        _q6_emit_single_row_lane_and_address_setup(emitter)
    else:
        _q6_emit_dual_row_lane_and_address_setup(emitter)
    _q6_emit_scalar_setup_tail(emitter, schedule)
    return emitter.module()


def _q6_emit_single_row_lane_and_address_setup(
    emitter: Q6ScheduleEmitter,
) -> None:
    emitter.inst("v_bfe_u32", "v4, v0, 10, 10")
    emitter.dual_zero(32, "v_dual_and_b32", "v1, 0x3ff, v0")
    emitter.inst("v_bfe_u32", "v5, v0, 2, 8")
    emitter.dual_zero(46, "v_dual_and_b32", "v3, 0x70, v0")
    emitter.dual_zero(44, "v_dual_lshlrev_b32", "v41, 3, v4")
    emitter.dual_zero(43, "v_dual_lshlrev_b32", "v2, 1, v1")
    emitter.dual_zero(39, "v_dual_and_b32", "v34, 15, v0")
    emitter.dual_zero(37, "v_dual_and_b32", "v42, 2, v5")
    emitter.dual_zero(38, "v_dual_add_nc_u32", "v5, v5, v41")
    emitter.dual_zero(36, "v_dual_and_b32", "v9, 3, v0")
    emitter.inst("v_bfe_u32", "v0, v0, 4, 6")
    emitter.scalar_argument_loads()

    emitter.inst("v_lshl_add_u32", "v1, v4, 5, v1")
    emitter.inst("v_sub_nc_u32_e32", "v6, v2, v34")
    emitter.dual_zero(28, "v_dual_and_b32", "v5, 63, v5")
    emitter.inst("v_lshl_or_b32", "v0, v4, 4, v0")
    emitter.dual_zero(40, "v_dual_and_b32", "v7, 63, v1")
    emitter.inst("v_lshl_add_u32", "v6, v6, 2, 0")
    emitter.inst("v_mul_u32_u24_e32", "v8, 0x130, v4")
    emitter.inst("v_xor_b32_e32", "v11, 32, v5")
    emitter.inst("v_mul_u32_u24_e32", "v12, 0x4c, v0")
    emitter.inst("v_and_or_b32", "v3, v2, 14, v3")
    emitter.dual_zero(30, "v_dual_lshlrev_b32", "v45, 3, v7")
    emitter.inst("v_mul_u32_u24_e32", "v7, 0x130, v7")
    emitter.dual_zero(35, "v_dual_lshlrev_b32", "v10, 1, v9")
    emitter.inst("v_lshl_add_u32", "v9, v9, 2, 0")
    emitter.dual_zero(26, "v_dual_lshlrev_b32", "v47, 3, v5")
    emitter.inst("v_mul_u32_u24_e32", "v5, 0x130, v5")
    emitter.inst("v_mad_u32_u24", "v4, 0x1300, v4, 0")
    emitter.dual_zero(33, "v_dual_lshlrev_b32", "v48, 3, v11")
    emitter.inst("v_mul_u32_u24_e32", "v11, 0x130, v11")
    emitter.inst("v_lshl_add_u32", "v51, v12, 2, 0")
    emitter.dual_zero(24, "v_dual_lshlrev_b32", "v53, 1, v2")
    emitter.dual_zero(29, "v_dual_add_nc_u32", "v2, v6, v8")
    emitter.inst("v_mad_u32_u24", "v49, 0x130, v34, v4")
    emitter.inst("v_mad_u32_u24", "v50, 0x90, v34, 0")
    emitter.inst("v_lshl_add_u32", "v52, v1, 2, 0")
    emitter.dual_zero(31, "v_dual_lshlrev_b32", "v54, 1, v3")
    emitter.dual_zero(22, "v_dual_add_nc_u32", "v55, 0, v7")
    emitter.dual_zero(27, "v_dual_lshlrev_b32", "v56, 1, v10")
    emitter.dual_zero(20, "v_dual_add_nc_u32", "v57, v9, v5")
    emitter.dual_zero(25, "v_dual_add_nc_u32", "v58, v9, v11")

    for zero, destination, offset in (
        (18, 59, 0x2400),
        (23, 60, 0x2800),
        (16, 61, 0x2C00),
        (21, 62, 0x3000),
        (14, 63, 0x3800),
        (19, 64, 0x3C00),
        (12, 65, 0x4000),
        (17, 66, 0x4400),
    ):
        emitter.dual_zero(
            zero,
            "v_dual_add_nc_u32",
            f"v{destination}, {hex(offset)}, v2",
        )
    emitter.inst("v_add_nc_u32_e32", "v67, 0x4800, v2")
    emitter.dual_zero(15, "v_dual_add_nc_u32", "v68, 0x4e00, v2")
    emitter.inst("v_add_nc_u32_e32", "v69, 0x5400, v2")
    emitter.dual_zero(13, "v_dual_add_nc_u32", "v70, 0x5800, v2")
    emitter.inst("v_add_nc_u32_e32", "v71, 0x5c00, v2")
    emitter.dual_zero(11, "v_dual_add_nc_u32", "v72, 0x6000, v2")
    emitter.inst("v_add_nc_u32_e32", "v73, 0x6400, v2")
    emitter.inst("v_add_nc_u32_e32", "v74, 0x6c00, v2")
    for destination, offset in zip(range(75, 79), (0x2600, 0x2A00, 0x2E00, 0x3400)):
        emitter.inst("v_add_nc_u32_e32", f"v{destination}, {hex(offset)}, v51")


def _q6_cooperative_global_loads(schedule: Q6ForwardSchedule) -> code.Module:
    emitter = Q6ScheduleEmitter(schedule, "cooperative_global_loads")
    layout = _q6_physical_layout(schedule)
    if layout.output_tile_rows == 1:
        _q6_emit_single_row_near_weight_reads(emitter, layout)
        _q6_emit_single_row_far_weight_reads(emitter, layout)
        _q6_emit_single_row_activation_reads(emitter)
    else:
        dependency_delay = Q6DependencyDelay("VALU_DEP", 1, "SKIP_1", 1)
        _q6_emit_dual_row_near_weight_reads(emitter, layout, dependency_delay)
        _q6_emit_dual_row_far_weight_reads(emitter, layout, dependency_delay)
        _q6_emit_dual_row_activation_reads(emitter, layout, dependency_delay)
    return emitter.module()


def _q6_emit_single_row_near_weight_reads(
    emitter: Q6ScheduleEmitter,
    layout: Q6PhysicalLayout,
) -> None:
    packed_block_bytes = hex(_Q6_PACKED_BLOCK_BYTES)
    emitter.annotation(
        ".LBB0_1:                                ; =>This Loop Header: Depth=1"
    )
    emitter.annotation(
        "                                        ;     Child Loop BB0_2 Depth 2"
    )
    emitter.annotation(
        "                                        ;     Child Loop BB0_4 Depth 2"
    )
    emitter.inst("s_add_i32", "s18, s23, s20")
    emitter.inst("s_mul_i32", f"s19, s18, {packed_block_bytes}")
    emitter.inst("s_mul_hi_i32", f"s24, s18, {packed_block_bytes}")
    emitter.inst("s_add_u32", "s18, s12, s19")
    emitter.inst("s_addc_u32", "s19, s13, s24")
    emitter.inst("s_mul_i32", "s24, s21, s23")
    emitter.inst("v_mad_u64_u32", f"v[2:3], null, {packed_block_bytes}, v41, s[18:19]")
    emitter.inst("s_add_i32", "s24, s24, s3")
    emitter.inst(
        "v_mad_u64_u32",
        f"v[114:115], null, {packed_block_bytes}, v48, s[18:19]",
    )
    emitter.inst(
        "v_mad_u64_u32",
        f"v[123:124], null, {packed_block_bytes}, v45, s[18:19]",
    )
    emitter.add_u64_batch(
        (
            Q6AddressAdd(4, Q6Vgpr(2), Q6Vgpr(53), Q6Vgpr(3)),
            Q6AddressAdd(2, Q6Vgpr(2), Q6Vgpr(54), Q6Vgpr(3)),
            Q6AddressAdd(
                6, Q6Immediate(0x1000, hexadecimal=True), Q6Vgpr(4), Q6Vgpr(5)
            ),
            Q6AddressAdd(
                79, Q6Immediate(0x1000, hexadecimal=True), Q6Vgpr(2), Q6Vgpr(3)
            ),
            Q6AddressAdd(
                81, Q6Immediate(0x3000, hexadecimal=True), Q6Vgpr(4), Q6Vgpr(5)
            ),
            Q6AddressAdd(
                83, Q6Immediate(0x3000, hexadecimal=True), Q6Vgpr(2), Q6Vgpr(3)
            ),
            Q6AddressAdd(
                85, Q6Immediate(0x4000, hexadecimal=True), Q6Vgpr(4), Q6Vgpr(5)
            ),
            Q6AddressAdd(
                87, Q6Immediate(0x4000, hexadecimal=True), Q6Vgpr(2), Q6Vgpr(3)
            ),
        )
    )
    emitter.global_read_clause(
        (
            layout.packed_read(0, "ql", 4),
            layout.packed_read(0, "qh", 2),
            layout.packed_read(1, "ql", 6),
            layout.packed_read(1, "qh", 79),
            layout.packed_read(2, "ql", 81),
            layout.packed_read(2, "qh", 83),
            layout.packed_read(3, "ql", 85),
            layout.packed_read(3, "qh", 87),
        )
    )
    emitter.add_u64_batch(
        tuple(
            Q6AddressAdd(
                destination,
                Q6Immediate(offset, hexadecimal=True),
                Q6Vgpr(base),
                Q6Vgpr(base + 1),
            )
            for destination, offset, base in (
                (81, 0x6000, 4),
                (85, 0x6000, 2),
                (89, 0x8000, 4),
                (91, 0x8000, 2),
                (93, 0x9000, 4),
                (95, 0x9000, 2),
                (97, 0xB000, 4),
                (99, 0xB000, 2),
            )
        )
    )
    emitter.global_read_clause(
        (
            layout.packed_read(4, "ql", 81),
            layout.packed_read(4, "qh", 85),
            layout.packed_read(5, "ql", 89),
            layout.packed_read(5, "qh", 91),
            layout.packed_read(6, "ql", 93),
            layout.packed_read(6, "qh", 95),
            layout.packed_read(7, "ql", 97),
            layout.packed_read(7, "qh", 99),
        )
    )


def _q6_emit_single_row_far_weight_reads(
    emitter: Q6ScheduleEmitter,
    layout: Q6PhysicalLayout,
) -> None:
    packed_block_bytes = hex(_Q6_PACKED_BLOCK_BYTES)
    emitter.add_u64_batch(
        tuple(
            Q6AddressAdd(
                destination,
                Q6Immediate(offset, hexadecimal=True),
                Q6Vgpr(base),
                Q6Vgpr(base + 1),
            )
            for destination, offset, base in (
                (89, 0xD000, 4),
                (91, 0xD000, 2),
                (93, 0xE000, 4),
                (95, 0xE000, 2),
                (97, 0x10000, 4),
                (99, 0x10000, 2),
                (101, 0x12000, 4),
                (103, 0x12000, 2),
            )
        )
    )
    emitter.global_read_clause(
        (
            layout.packed_read(8, "ql", 89),
            layout.packed_read(8, "qh", 91),
            layout.packed_read(9, "ql", 93),
            layout.packed_read(9, "qh", 95),
            layout.packed_read(10, "ql", 97),
            layout.packed_read(10, "qh", 99),
            layout.packed_read(11, "ql", 101),
            layout.packed_read(11, "qh", 103),
        )
    )
    emitter.add_u64_batch(
        tuple(
            Q6AddressAdd(
                destination,
                Q6Immediate(offset, hexadecimal=True),
                Q6Vgpr(base),
                Q6Vgpr(base + 1),
            )
            for destination, offset, base in (
                (89, 0x13000, 4),
                (91, 0x13000, 2),
                (93, 0x15000, 4),
                (95, 0x15000, 2),
                (97, 0x16000, 4),
                (99, 0x17000, 2),
                (101, 0x18000, 2),
            )
        )
    )
    emitter.inst("v_mad_u64_u32", "v[2:3], null, s24, 36, v[1:2]")
    emitter.inst(
        "v_mad_u64_u32",
        f"v[103:104], null, {packed_block_bytes}, v47, s[18:19]",
    )
    emitter.add_u64(
        Q6AddressAdd(4, Q6Immediate(0x18000, hexadecimal=True), Q6Vgpr(4), Q6Vgpr(5))
    )
    emitter.inst("s_mov_b32", "s18, -4")
    emitter.inst("v_ashrrev_i32_e32", "v3, 31, v2")
    emitter.add_u64(Q6AddressAdd(103, Q6Vgpr(103), Q6Vgpr(56), Q6Vgpr(104)))
    emitter.inst("v_lshlrev_b64", "v[116:117], 2, v[2:3]")
    emitter.add_u64(Q6AddressAdd(125, Q6Vgpr(114), Q6Vgpr(56), Q6Vgpr(115)))
    emitter.add_u64(
        Q6AddressAdd(
            127,
            Q6Sgpr(14),
            Q6Vgpr(116),
            Q6Vgpr(117),
            high_left=Q6Sgpr(15),
        )
    )
    emitter.global_read_clause(
        (
            layout.packed_read(12, "qh", 91),
            layout.packed_read(13, "ql", 93),
            layout.packed_read(13, "qh", 95),
            layout.packed_read(14, "qh", 99),
            layout.packed_read(15, "qh", 101),
            layout.packed_read(15, "ql", 4),
            layout.packed_read(14, "ql", 97),
            layout.packed_read(12, "ql", 89),
            Q6GlobalRead(124, 123, _Q6_FACTOR_OFFSET, 16),
            Q6GlobalRead(3, 103, _Q6_SCALE_OFFSET),
            Q6GlobalRead(4, 125, _Q6_SCALE_OFFSET),
        )
    )


def _q6_emit_single_row_activation_reads(emitter: Q6ScheduleEmitter) -> None:
    emitter.global_read_clause(
        tuple(
            Q6GlobalRead(destination, 127, 512 * index)
            for index, destination in enumerate((5, 89, 90, 91, 92, 93, 94, 95))
        )
    )
    emitter.add_u64_batch(
        (
            Q6AddressAdd(
                98, Q6Vgpr(127), Q6Immediate(0x2000, hexadecimal=True), Q6Vgpr(128)
            ),
            Q6AddressAdd(
                104, Q6Immediate(0x1000, hexadecimal=True), Q6Vgpr(127), Q6Vgpr(128)
            ),
            Q6AddressAdd(
                125, Q6Immediate(0x2000, hexadecimal=True), Q6Vgpr(127), Q6Vgpr(128)
            ),
        )
    )
    emitter.global_read_clause(
        (
            Q6GlobalRead(97, 98, -4096),
            Q6GlobalRead(96, 98),
            *tuple(
                Q6GlobalRead(98 + index, 104, 512 * (index + 1)) for index in range(7)
            ),
            Q6GlobalRead(105, 125, 512),
        )
    )


def _q6_packed_decode(schedule: Q6ForwardSchedule) -> code.Module:
    layout = _q6_physical_layout(schedule)
    plan = layout.decode
    semantics = _Q6_SEMANTICS.q6_signed_decode()
    emitter = Q6ScheduleEmitter(schedule, "packed_decode")
    emitter.wait_vmem(0)
    emitter.inst(
        "v_cvt_f32_f16_e64",
        f"v{plan.factor_register}, {Q6HalfRegister(plan.factor_source_register, 'l')}",
    )
    lane_shift = Q6Vgpr(plan.lane_shift_register)
    if schedule.semantic_policy.traversal == "OutputRoleWavefront":
        _q6_emit_decode_wavefront(emitter, plan, semantics, lane_shift)
        return emitter.module()
    for source, output in zip(plan.sources, plan.outputs, strict=True):
        ql = source.low_payload.first_register
        qh = source.high_payload.first_register
        low = output.low.first_register
        high = output.high.first_register
        emitter.shift_right_dword(high, semantics.high_bits_shift, ql)
        emitter.mask_dword(high, hex(semantics.low_nibble_mask), high)
        emitter.mask_dword(low, hex(semantics.low_nibble_mask), ql)
        emitter.shift_right_dword(qh, lane_shift, qh)
        emitter.merge_q6_dword(high, qh, high)
        emitter.shift_left_dword(qh, semantics.high_bits_shift, qh)
        emitter.merge_q6_dword(low, qh, low)
        for register in (low, high):
            emitter.inst(
                "v_add_nc_u32_e32",
                f"v{register}, {hex(semantics.signed_add)}, v{register}",
            )
        for register in (low, high):
            emitter.inst(
                "v_xor_b32_e32",
                f"v{register}, {hex(semantics.signed_xor)}, v{register}",
            )
    return emitter.module()


def _q6_emit_decode_wavefront(
    emitter: Q6ScheduleEmitter,
    plan: Q6DecodeRegisterPlan,
    semantics: Q6SignedDecodeSpec,
    lane_shift: Q6Vgpr,
) -> None:
    """Emit the fixed hazard-aware row/role decode frontier.

    The physical plan intentionally reuses each atom's packed QH register for the
    next output high role.  All destructive QH shifts therefore form the first
    frontier, the high-role handoff is walked in atom order, and the independent
    low merge and signed normalization frontiers are deferred until their inputs
    are no longer needed.
    """
    # Keep the helper's inputs at the semantic boundary without introducing an
    # instruction scheduler or a register-repair fallback.
    sources = plan.sources
    outputs = plan.outputs
    signed = semantics
    for source in sources:
        qh = source.high_payload.first_register
        emitter.shift_right_dword(qh, lane_shift, qh)

    for source, output in zip(sources, outputs, strict=True):
        ql = source.low_payload.first_register
        qh = source.high_payload.first_register
        low = output.low.first_register
        high = output.high.first_register
        emitter.shift_right_dword(high, signed.high_bits_shift, ql)
        emitter.mask_dword(high, hex(signed.low_nibble_mask), high)
        emitter.mask_dword(low, hex(signed.low_nibble_mask), ql)
        emitter.merge_q6_dword(high, qh, high)
        emitter.shift_left_dword(qh, signed.high_bits_shift, qh)
        # This low merge must precede the next role's high-result handoff,
        # which reuses the current role's packed QH register.
        emitter.merge_q6_dword(low, qh, low)

    for output in outputs:
        emitter.inst(
            "v_add_nc_u32_e32",
            f"v{output.low.first_register}, {hex(signed.signed_add)}, "
            f"v{output.low.first_register}",
        )
    for output in outputs:
        emitter.inst(
            "v_add_nc_u32_e32",
            f"v{output.high.first_register}, {hex(signed.signed_add)}, "
            f"v{output.high.first_register}",
        )
    for output in outputs:
        emitter.inst(
            "v_xor_b32_e32",
            f"v{output.low.first_register}, {hex(signed.signed_xor)}, "
            f"v{output.low.first_register}",
        )
    for output in outputs:
        emitter.inst(
            "v_xor_b32_e32",
            f"v{output.high.first_register}, {hex(signed.signed_xor)}, "
            f"v{output.high.first_register}",
        )


def _q6_decoded_lds_writes(schedule: Q6ForwardSchedule) -> code.Module:
    layout = _q6_physical_layout(schedule)
    plan = layout.decode
    lds = layout.lds
    emitter = Q6ScheduleEmitter(schedule, "decoded_lds_writes")
    for output in plan.outputs:
        offset0, offset1 = layout.decoded_write_offsets(output.atom)
        emitter.local_write_pair(
            layout.decoded_write_address(output.atom),
            output.low.first_register,
            output.high.first_register,
            offset0,
            offset1,
            stride64=False,
        )
    emitter.local_write(
        plan.factor_write_address,
        plan.factor_register,
        lds.decoded_plane_base,
    )
    emitter.local_write(plan.scale_write_address_base, 3, lds.scale_read_base)
    emitter.local_write(plan.scale_write_address_base + 1, 4, lds.scale_read_base)
    payloads = plan.activation_payload_registers
    for pair in range(len(payloads) // 2):
        emitter.local_write_cooperative_pair(
            lds,
            plan.cooperative_write_address,
            payloads[2 * pair],
            payloads[2 * pair + 1],
            pair,
        )
    return emitter.module()


def _q6_commit_stage(
    emitter: Q6ScheduleEmitter,
    layout: Q6PhysicalLayout,
    pointer_destinations: tuple[int, int, int],
    read_destination_base: int,
) -> None:
    sources = layout.stage_pointer_sources
    emitter.commit_local_stage(
        (
            (pointer_destinations[0], sources[0]),
            (pointer_destinations[1], sources[1]),
        ),
        (pointer_destinations[2], sources[2]),
        layout.stage_reads(read_destination_base),
    )


def _q6_first_stage_barrier(schedule: Q6ForwardSchedule) -> code.Module:
    layout = _q6_physical_layout(schedule)
    emitter = Q6ScheduleEmitter(schedule, "first_stage_barrier")
    if layout.first_stage_vmem_wait is not None:
        emitter.wait_vmem(layout.first_stage_vmem_wait)
    write_offset = layout.lds.first_stage_write_offset
    emitter.local_write_pair(
        layout.first_stage_write_address,
        *layout.first_stage_weight_values,
        write_offset,
        write_offset + 2,
    )
    _q6_commit_stage(
        emitter,
        layout,
        (
            layout.row_pointer_base,
            layout.weight_address_base,
            layout.activation_address_base,
        ),
        3,
    )
    return emitter.module()


def _q6_begin_second_stage(
    emitter: Q6ScheduleEmitter,
    layout: Q6PhysicalLayout,
) -> None:
    macro_tile_m = 64 * layout.output_tile_rows
    emitter.annotation(
        "; %bb.3:                                ; "
        f"%_ZL18mmq_vec_dot_targetIL9ggml_type14ELi{macro_tile_m}"
        "ELb1ELb0EEvPKiS2_Pfi.exit.i"
    )
    emitter.annotation(
        "                                        ;   in Loop: Header=BB0_1 Depth=1"
    )
    emitter.inst("v_add_nc_u32_e32", "v2, s22, v2")
    emitter.inst("s_barrier")
    emitter.invalidate_global_cache()
    emitter.inst("s_mov_b32", "s18, -4")
    emitter.inst("v_ashrrev_i32_e32", "v3, 31, v2")
    explicit_delays = emitter.schedule.dependency_delay_mode == "Explicit"
    if explicit_delays:
        emitter.dependency_delay(Q6DependencyDelay("VALU_DEP", 1, "NEXT", 1))
    emitter.inst("v_lshlrev_b64", "v[2:3], 2, v[2:3]")
    emitter.add_u64(
        Q6AddressAdd(
            2,
            Q6Sgpr(14),
            Q6Vgpr(2),
            Q6Vgpr(3),
            high_left=Q6Sgpr(15),
            delay_after_low=Q6DependencyDelay("VALU_DEP", 1)
            if explicit_delays
            else None,
        )
    )


def _q6_second_stage_loads(schedule: Q6ForwardSchedule) -> code.Module:
    emitter = Q6ScheduleEmitter(schedule, "second_stage_loads")
    layout = _q6_physical_layout(schedule)
    _q6_begin_second_stage(emitter, layout)
    emitter.global_read_clause(
        tuple(layout.refill_read(slot, 2, 512 * slot) for slot in range(8))
    )
    if layout.output_tile_rows == 1:
        _q6_emit_single_row_refill_reads(emitter, layout)
    else:
        _q6_emit_dual_row_refill_reads(emitter, layout)
    emitter.commit_cooperative_global_reads(
        layout.lds,
        layout.first_stage_write_address,
    )
    _q6_commit_stage(
        emitter,
        layout,
        (10, layout.row_pointer_base, layout.weight_address_base),
        2,
    )
    return emitter.module()


def _q6_emit_single_row_refill_reads(
    emitter: Q6ScheduleEmitter,
    layout: Q6PhysicalLayout,
) -> None:
    emitter.add_u64_batch(
        (
            Q6AddressAdd(
                4, Q6Immediate(0x1000, hexadecimal=True), Q6Vgpr(2), Q6Vgpr(3)
            ),
            Q6AddressAdd(
                6, Q6Vgpr(2), Q6Immediate(0x2000, hexadecimal=True), Q6Vgpr(3)
            ),
            Q6AddressAdd(
                2, Q6Immediate(0x2000, hexadecimal=True), Q6Vgpr(2), Q6Vgpr(3)
            ),
        )
    )
    emitter.global_read_clause(
        (
            layout.refill_read(8, 6, -4096),
            layout.refill_read(16, 6),
            *tuple(
                layout.refill_read(slot, 4, 512 * (slot - 8)) for slot in range(9, 16)
            ),
            layout.refill_read(17, 2, 512),
        )
    )


def _q6_emit_epilogue_control_flow(
    emitter: Q6ScheduleEmitter,
    layout: Q6PhysicalLayout,
    blocks_per_weight_row: int,
) -> None:
    emitter.annotation(layout.loop_exit_annotation)
    emitter.annotation(
        "                                        ;   in Loop: Header=BB0_1 Depth=1"
    )
    emitter.inst("s_add_i32", "s23, s23, 1")
    if emitter.schedule.dependency_delay_mode == "Explicit":
        emitter.dependency_delay(Q6DependencyDelay("SALU_CYCLE", 1))
    emitter.inst("s_cmp_lg_u32", f"s23, {blocks_per_weight_row}")
    emitter.inst("s_barrier")
    emitter.invalidate_global_cache()
    emitter.inst("s_cbranch_scc1", ".LBB0_1")
    emitter.annotation(
        "; %bb.6:                                ; semantic BF16 output epilogue"
    )


def _q6_initialize_epilogue_address_cursor(
    emitter: Q6ScheduleEmitter,
    layout: Q6PhysicalLayout,
) -> None:
    emitter.inst("s_load_b32", "s4, s[0:1], 0x18")
    emitter.inst("s_lshl_b32", "s0, s2, 6")
    emitter.inst("s_waitcnt", "lgkmcnt(0)")
    emitter.inst(
        "v_mad_u64_u32",
        f"v[0:1], null, s4, v{layout.output_index_register}, v[0:1]",
    )
    emitter.inst("s_mul_i32", "s2, s4, s3")
    emitter.inst("s_ashr_i32", "s3, s2, 31")
    emitter.inst("s_lshl_b64", "s[2:3], s[2:3], 1")
    emitter.inst("s_add_u32", "s2, s16, s2")
    emitter.inst("s_addc_u32", "s3, s17, s3")
    emitter.inst("s_ashr_i32", "s1, s0, 31")
    emitter.inst("s_lshl_b64", "s[0:1], s[0:1], 1")
    emitter.inst("s_add_u32", "s0, s2, s0")
    emitter.inst("s_addc_u32", "s1, s3, s1")
    emitter.inst("s_lshl_b32", "s2, s4, 4")


def _q6_emit_epilogue_store_address(emitter: Q6ScheduleEmitter) -> None:
    emitter.inst("v_ashrrev_i32_e32", "v1, 31, v0")
    emitter.inst("v_lshlrev_b64", "v[1:2], 1, v[0:1]")
    emitter.add_u64(
        Q6AddressAdd(
            1,
            Q6Sgpr(0),
            Q6Vgpr(1),
            Q6Vgpr(2),
            high_left=Q6Sgpr(1),
            delay_after_low=(
                Q6DependencyDelay("VALU_DEP", 1)
                if emitter.schedule.dependency_delay_mode == "Explicit"
                else None
            ),
        )
    )


def _q6_emit_bf16_pipeline(
    emitter: Q6ScheduleEmitter,
    layout: Q6PhysicalLayout,
) -> None:
    all_sources = layout.accumulator_registers
    store_width = emitter.schedule.store_vector_width
    scratch_base = _q6_epilogue_scratch_base(
        layout,
        emitter.schedule.epilogue_dependency_width,
        emitter.schedule.resource_usage.vgprs,
    )
    full_tile = emitter.schedule.epilogue_pipeline_scope == "FullTile"
    segments = (
        (all_sources,)
        if full_tile
        else tuple(
            all_sources[start : start + store_width]
            for start in range(0, len(all_sources), store_width)
        )
    )
    for segment_index, sources in enumerate(segments):
        width = emitter.schedule.epilogue_dependency_width
        if not full_tile:
            _q6_emit_epilogue_store_address(emitter)
        for index, source in enumerate(sources[:width]):
            emitter.inst(
                "v_bfe_u32",
                f"v{scratch_base + index}, v{source}, 16, 1",
            )
            emitter.inst(
                "v_or_b32_e32",
                f"v{scratch_base + width + index}, 0x400000, v{source}",
            )
        if full_tile and segment_index == 0:
            _q6_emit_epilogue_store_address(emitter)
        for index, source in enumerate(sources):
            slot = index % width
            future = index + width
            emitter.inst("v_cmp_u_f32_e32", f"vcc_lo, v{source}, v{source}")
            emitter.inst(
                "v_add3_u32",
                f"v{source}, v{scratch_base + slot}, v{source}, 0x7fff",
            )
            if future < len(sources):
                emitter.inst(
                    "v_bfe_u32",
                    f"v{scratch_base + slot}, v{sources[future]}, 16, 1",
                )
            if emitter.schedule.dependency_delay_mode == "Explicit":
                dependency_distance = 2 if future < len(sources) else 1
                emitter.dependency_delay(
                    Q6DependencyDelay("VALU_DEP", dependency_distance)
                )
            emitter.inst(
                "v_cndmask_b32_e32",
                f"v{source}, v{source}, v{scratch_base + width + slot}, vcc_lo",
            )
            if future < len(sources):
                emitter.inst(
                    "v_or_b32_e32",
                    f"v{scratch_base + width + slot}, 0x400000, v{sources[future]}",
                )
            global_index = segment_index * store_width + index
            if full_tile and (global_index + 1) % store_width == 0:
                batch = all_sources[global_index + 1 - store_width : global_index + 1]
                emitter.store_bf16_clause("v[1:2]", batch)
                if future < len(sources):
                    emitter.inst("v_add_nc_u32_e32", "v0, s2, v0")
                    _q6_emit_epilogue_store_address(emitter)
        if not full_tile:
            emitter.store_bf16_clause("v[1:2]", sources)
            if segment_index + 1 < len(segments):
                emitter.inst("v_add_nc_u32_e32", "v0, s2, v0")


def _q6_bf16_epilogue(
    schedule: Q6ForwardSchedule,
    blocks_per_weight_row: int = 8,
) -> code.Module:
    emitter = Q6ScheduleEmitter(schedule, "bf16_epilogue")
    layout = _q6_physical_layout(schedule)
    _q6_emit_epilogue_control_flow(emitter, layout, blocks_per_weight_row)
    _q6_initialize_epilogue_address_cursor(emitter, layout)
    _q6_emit_bf16_pipeline(emitter, layout)
    emitter.inst("s_nop", "0")
    emitter.inst("s_sendmsg", "sendmsg(MSG_DEALLOC_VGPRS)")
    emitter.inst("s_endpgm")
    return emitter.module()


def _q6_emit_dual_row_lane_and_address_setup(
    emitter: Q6ScheduleEmitter,
) -> None:
    emitter.inst("v_and_b32_e32", "v1, 0x3ff, v0")
    emitter.dual_zero(14, "v_dual_and_b32", "v3, 0x70, v0")
    emitter.inst("v_bfe_u32", "v4, v0, 10, 10")
    emitter.dual_zero(16, "v_dual_and_b32", "v61, 15, v0")
    emitter.inst("v_bfe_u32", "v5, v0, 2, 8")
    emitter.dual_zero(24, "v_dual_and_b32", "v9, 3, v0")
    emitter.inst("v_bfe_u32", "v0, v0, 4, 6")
    emitter.inst("v_lshlrev_b32_e32", "v68, 3, v4")
    emitter.dependency_delay(Q6DependencyDelay("VALU_DEP", 4, "NEXT", 4))
    emitter.dual_zero(18, "v_dual_and_b32", "v69, 2, v5")
    emitter.dual_zero(17, "v_dual_lshlrev_b32", "v10, 1, v9")
    emitter.dependency_delay(Q6DependencyDelay("VALU_DEP", 4))
    emitter.inst("v_lshl_or_b32", "v0, v4, 4, v0")
    emitter.inst("v_mov_b32_e32", "v22, 0")
    emitter.inst("v_lshl_add_u32", "v9, v9, 2, 0")
    emitter.scalar_argument_loads()
    emitter.inst("v_mul_u32_u24_e32", "v8, 0x130, v4")
    emitter.inst("v_mul_u32_u24_e32", "v13, 0x4c, v0")
    emitter.inst("v_mov_b32_e32", "v26, 0")
    emitter.inst("v_mad_u32_u24", "v77, 0x90, v61, 0")
    emitter.dual_zero(32, "v_dual_lshlrev_b32", "v83, 1, v10")
    emitter.dependency_delay(Q6DependencyDelay("VALU_DEP", 4, "SKIP_2", 3))
    emitter.inst("v_lshl_add_u32", "v78, v13, 2, 0")
    emitter.inst("v_mov_b32_e32", "v13, 0")
    emitter.dual_zero(28, "v_dual_add_nc_u32", "v5, v5, v68")
    emitter.dual_zero(47, "v_dual_add_nc_u32", "v106, 0x4a00, v78")
    emitter.dual_zero(50, "v_dual_add_nc_u32", "v107, 0x4f00, v78")
    emitter.dependency_delay(Q6DependencyDelay("VALU_DEP", 3, "SKIP_3", 4))
    emitter.inst("v_and_b32_e32", "v5, 63, v5")
    emitter.dual_zero(49, "v_dual_add_nc_u32", "v108, 0x5400, v78")
    emitter.inst("v_mov_b32_e32", "v11, 0")
    emitter.dual_zero(52, "v_dual_add_nc_u32", "v109, 0x5800, v78")
    emitter.inst("v_xor_b32_e32", "v12, 32, v5")
    emitter.inst("v_lshlrev_b32_e32", "v73, 3, v5")
    emitter.inst("v_mul_u32_u24_e32", "v5, 0x130, v5")
    emitter.dual("v_dual_mov_b32", "v20, 0", "v_dual_mov_b32", "v19, 0")
    emitter.dependency_delay(Q6DependencyDelay("VALU_DEP", 4, "SKIP_1", 4))
    emitter.inst("v_lshlrev_b32_e32", "v74, 3, v12")
    emitter.inst("v_mul_u32_u24_e32", "v12, 0x130, v12")
    emitter.dual_zero(34, "v_dual_add_nc_u32", "v85, v9, v5")
    emitter.inst("v_mov_b32_e32", "v15, 0")
    emitter.dependency_delay(Q6DependencyDelay("VALU_DEP", 3))
    emitter.dual_zero(21, "v_dual_add_nc_u32", "v86, v9, v12")
    emitter.dual("v_dual_mov_b32", "v27, 0", "v_dual_mov_b32", "v12, 0")
    emitter.inst("v_lshlrev_b32_e32", "v2, 1, v1")
    emitter.inst("v_lshl_add_u32", "v1, v4, 5, v1")
    emitter.inst("v_mad_u32_u24", "v4, 0x1300, v4, 0")
    emitter.inst("v_mov_b32_e32", "v23, 0")
    emitter.inst("v_mov_b32_e32", "v25, 0")
    emitter.inst("v_sub_nc_u32_e32", "v6, v2, v61")
    emitter.inst("v_and_b32_e32", "v7, 63, v1")
    emitter.inst("v_and_or_b32", "v3, v2, 14, v3")
    emitter.inst("v_lshlrev_b32_e32", "v80, 1, v2")
    emitter.inst("v_mad_u32_u24", "v75, 0x130, v61, v4")
    emitter.inst("v_lshl_add_u32", "v6, v6, 2, 0")
    emitter.inst("v_lshlrev_b32_e32", "v70, 3, v7")
    emitter.inst("v_mul_u32_u24_e32", "v7, 0x130, v7")
    emitter.inst("v_lshl_add_u32", "v79, v1, 2, 0")
    emitter.dual_zero(30, "v_dual_lshlrev_b32", "v81, 1, v3")
    emitter.inst("v_add_nc_u32_e32", "v2, v6, v8")
    emitter.dependency_delay(Q6DependencyDelay("VALU_DEP", 4, "SKIP_1", 3))
    emitter.dual_zero(29, "v_dual_add_nc_u32", "v82, 0, v7")
    emitter.inst("v_mov_b32_e32", "v31, 0")

    for zero, destination, offset in (
        (33, 92, 0x5C00),
        (36, 87, 0x4800),
    ):
        emitter.dual_zero(
            zero,
            "v_dual_add_nc_u32",
            f"v{destination}, {hex(offset)}, v2",
        )
    emitter.inst("v_add_nc_u32_e32", "v88, 0x4c00, v2")
    for zero, destination, offset in ((38, 89, 0x5000),):
        emitter.dual_zero(
            zero,
            "v_dual_add_nc_u32",
            f"v{destination}, {hex(offset)}, v2",
        )
    emitter.inst("v_add_nc_u32_e32", "v90, 0x5800, v2")
    for zero, destination, offset in (
        (40, 93, 0x6000),
        (35, 94, 0x6400),
        (42, 95, 0x6800),
        (37, 96, 0x7000),
        (39, 98, 0x7400),
        (44, 99, 0x7800),
        (41, 100, 0x7C00),
        (46, 101, 0x8000),
        (43, 102, 0x8600),
        (48, 103, 0x8C00),
        (45, 104, 0x9000),
    ):
        emitter.dual_zero(
            zero,
            "v_dual_add_nc_u32",
            f"v{destination}, {hex(offset)}, v2",
        )
    for left, right in (
        (51, 54),
        (53, 56),
        (55, 58),
        (57, 60),
        (59, 62),
        (63, 64),
        (65, 66),
        (67, 72),
        (71, 76),
        (84, 91),
        (97, 110),
    ):
        emitter.dual(
            "v_dual_mov_b32",
            f"v{left}, 0",
            "v_dual_mov_b32",
            f"v{right}, 0",
        )
    emitter.inst("v_mov_b32_e32", "v105, 0")


def _q6_emit_dual_row_near_weight_reads(
    emitter: Q6ScheduleEmitter,
    layout: Q6PhysicalLayout,
    dependency_delay: Q6DependencyDelay,
) -> None:
    packed_block_bytes = hex(_Q6_PACKED_BLOCK_BYTES)
    emitter.annotation(
        ".LBB0_1:                                ; =>This Loop Header: Depth=1"
    )
    emitter.annotation(
        "                                        ;     Child Loop BB0_2 Depth 2"
    )
    emitter.annotation(
        "                                        ;     Child Loop BB0_4 Depth 2"
    )
    emitter.dependency_delay(Q6DependencyDelay("SALU_CYCLE", 1, "NEXT", 1))
    emitter.inst("s_add_i32", "s18, s23, s20")
    emitter.inst("s_mul_i32", f"s19, s18, {packed_block_bytes}")
    emitter.inst("s_mul_hi_i32", f"s24, s18, {packed_block_bytes}")
    emitter.inst("s_add_u32", "s18, s12, s19")
    emitter.inst("s_addc_u32", "s19, s13, s24")
    emitter.inst("s_mul_i32", "s24, s21, s23")
    emitter.inst("v_mad_u64_u32", f"v[4:5], null, {packed_block_bytes}, v68, s[18:19]")
    emitter.inst("s_add_i32", "s24, s24, s3")
    emitter.inst(
        "v_mad_u64_u32",
        f"v[137:138], null, {packed_block_bytes}, v74, s[18:19]",
    )
    emitter.inst(
        "v_mad_u64_u32",
        f"v[139:140], null, {packed_block_bytes}, v70, s[18:19]",
    )
    emitter.add_u64_batch(
        (
            Q6AddressAdd(
                2,
                Q6Vgpr(4),
                Q6Vgpr(80),
                Q6Vgpr(5),
                delay_after_low=dependency_delay,
            ),
            Q6AddressAdd(
                4,
                Q6Vgpr(4),
                Q6Vgpr(81),
                Q6Vgpr(5),
                delay_after_high=Q6DependencyDelay("VALU_DEP", 4, "NEXT", 1),
            ),
            Q6AddressAdd(
                6,
                Q6Immediate(0x1000, hexadecimal=True),
                Q6Vgpr(2),
                Q6Vgpr(3),
                delay_after_high=Q6DependencyDelay("VALU_DEP", 4, "NEXT", 1),
            ),
            Q6AddressAdd(
                113, Q6Immediate(0x1000, hexadecimal=True), Q6Vgpr(4), Q6Vgpr(5)
            ),
            Q6AddressAdd(
                115,
                Q6Immediate(0x3000, hexadecimal=True),
                Q6Vgpr(2),
                Q6Vgpr(3),
                delay_after_low=dependency_delay,
            ),
            Q6AddressAdd(
                117, Q6Immediate(0x3000, hexadecimal=True), Q6Vgpr(4), Q6Vgpr(5)
            ),
            Q6AddressAdd(
                119,
                Q6Immediate(0x4000, hexadecimal=True),
                Q6Vgpr(2),
                Q6Vgpr(3),
                delay_after_low=dependency_delay,
            ),
            Q6AddressAdd(
                121, Q6Immediate(0x4000, hexadecimal=True), Q6Vgpr(4), Q6Vgpr(5)
            ),
        )
    )
    emitter.global_read_clause(
        (
            layout.packed_read(0, "ql", 2),
            layout.packed_read(0, "qh", 4),
            layout.packed_read(1, "ql", 6),
            layout.packed_read(1, "qh", 113),
            layout.packed_read(2, "ql", 115),
            layout.packed_read(2, "qh", 117),
            layout.packed_read(3, "ql", 119),
            layout.packed_read(3, "qh", 121),
        )
    )
    emitter.add_offset_addresses(
        (
            Q6OffsetAddress(113, 0x6000, 2, True),
            Q6OffsetAddress(115, 0x6000, 4, False),
            Q6OffsetAddress(117, 0x8000, 2, True),
            Q6OffsetAddress(121, 0x8000, 4, False),
            Q6OffsetAddress(123, 0x9000, 2, True),
            Q6OffsetAddress(125, 0x9000, 4, False),
            Q6OffsetAddress(127, 0xB000, 2, True),
            Q6OffsetAddress(129, 0xB000, 4, False),
        ),
        dependency_delay,
    )


def _q6_emit_dual_row_far_weight_reads(
    emitter: Q6ScheduleEmitter,
    layout: Q6PhysicalLayout,
    dependency_delay: Q6DependencyDelay,
) -> None:
    packed_block_bytes = hex(_Q6_PACKED_BLOCK_BYTES)
    emitter.global_read_clause(
        (
            layout.packed_read(4, "ql", 113),
            layout.packed_read(4, "qh", 115),
            layout.packed_read(5, "ql", 117),
            layout.packed_read(5, "qh", 121),
            layout.packed_read(6, "ql", 123),
            layout.packed_read(6, "qh", 125),
            layout.packed_read(7, "ql", 127),
            layout.packed_read(7, "qh", 129),
        )
    )
    emitter.add_offset_addresses(
        (
            Q6OffsetAddress(121, 0xD000, 2, True),
            Q6OffsetAddress(123, 0xD000, 4, False),
            Q6OffsetAddress(125, 0xE000, 2, True),
            Q6OffsetAddress(127, 0xE000, 4, False),
            Q6OffsetAddress(129, 0x10000, 2, True),
            Q6OffsetAddress(131, 0x10000, 4, False),
            Q6OffsetAddress(133, 0x12000, 2, True),
            Q6OffsetAddress(135, 0x12000, 4, False),
        ),
        dependency_delay,
    )
    emitter.global_read_clause(
        (
            layout.packed_read(8, "ql", 121),
            layout.packed_read(8, "qh", 123),
            layout.packed_read(9, "ql", 125),
            layout.packed_read(9, "qh", 127),
            layout.packed_read(10, "ql", 129),
            layout.packed_read(10, "qh", 131),
            layout.packed_read(11, "ql", 133),
            layout.packed_read(11, "qh", 135),
        )
    )
    emitter.add_offset_addresses(
        (
            Q6OffsetAddress(121, 0x13000, 2, True),
            Q6OffsetAddress(123, 0x13000, 4, False),
            Q6OffsetAddress(125, 0x15000, 2, True),
            Q6OffsetAddress(127, 0x15000, 4, False),
            Q6OffsetAddress(129, 0x16000, 2, True),
            Q6OffsetAddress(131, 0x17000, 4, False),
        ),
        dependency_delay,
    )
    emitter.add_u64(
        Q6AddressAdd(
            133,
            Q6Immediate(0x18000, hexadecimal=True),
            Q6Vgpr(2),
            Q6Vgpr(3),
            delay_after_low=Q6DependencyDelay("VALU_DEP", 1, "SKIP_3", 1),
        )
    )
    emitter.inst("v_mad_u64_u32", "v[2:3], null, s24, 36, v[1:2]")
    emitter.inst(
        "v_mad_u64_u32",
        f"v[135:136], null, {packed_block_bytes}, v73, s[18:19]",
    )
    emitter.add_u64(
        Q6AddressAdd(4, Q6Immediate(0x18000, hexadecimal=True), Q6Vgpr(4), Q6Vgpr(5))
    )
    emitter.inst("s_mov_b32", "s18, -4")
    emitter.inst("v_ashrrev_i32_e32", "v3, 31, v2")
    emitter.add_u64(
        Q6AddressAdd(
            135,
            Q6Vgpr(135),
            Q6Vgpr(83),
            Q6Vgpr(136),
            delay_after_low=Q6DependencyDelay("VALU_DEP", 1, "NEXT", 3),
        )
    )


def _q6_emit_dual_row_activation_reads(
    emitter: Q6ScheduleEmitter,
    layout: Q6PhysicalLayout,
    dependency_delay: Q6DependencyDelay,
) -> None:
    emitter.inst("v_lshlrev_b64", "v[141:142], 2, v[2:3]")
    emitter.add_u64(
        Q6AddressAdd(
            137,
            Q6Vgpr(137),
            Q6Vgpr(83),
            Q6Vgpr(138),
            delay_after_low=Q6DependencyDelay("VALU_DEP", 1, "NEXT", 3),
        )
    )
    emitter.add_u64(
        Q6AddressAdd(
            141,
            Q6Sgpr(14),
            Q6Vgpr(141),
            Q6Vgpr(142),
            high_left=Q6Sgpr(15),
            delay_after_low=Q6DependencyDelay("VALU_DEP", 1),
        )
    )
    emitter.global_read_clause(
        (
            layout.packed_read(12, "ql", 121),
            layout.packed_read(12, "qh", 123),
            layout.packed_read(13, "ql", 125),
            layout.packed_read(13, "qh", 127),
            layout.packed_read(14, "ql", 129),
            layout.packed_read(14, "qh", 131),
            layout.packed_read(15, "ql", 133),
            layout.packed_read(15, "qh", 4),
            Q6GlobalRead(174, 139, _Q6_FACTOR_OFFSET, 16),
            Q6GlobalRead(3, 135, _Q6_SCALE_OFFSET),
            Q6GlobalRead(4, 137, _Q6_SCALE_OFFSET),
        )
    )
    emitter.global_read_clause(
        tuple(
            Q6GlobalRead(destination, 141, 512 * index)
            for index, destination in enumerate((5, 121, 122, 123, 124, 125, 126, 127))
        )
    )
    emitter.add_u64_batch(
        (
            Q6AddressAdd(
                134,
                Q6Immediate(0x1000, hexadecimal=True),
                Q6Vgpr(141),
                Q6Vgpr(142),
                delay_after_low=dependency_delay,
            ),
            Q6AddressAdd(
                144, Q6Vgpr(141), Q6Immediate(0x2000, hexadecimal=True), Q6Vgpr(142)
            ),
            Q6AddressAdd(
                146,
                Q6Immediate(0x2000, hexadecimal=True),
                Q6Vgpr(141),
                Q6Vgpr(142),
                delay_after_low=dependency_delay,
            ),
            Q6AddressAdd(
                151, Q6Immediate(0x3000, hexadecimal=True), Q6Vgpr(141), Q6Vgpr(142)
            ),
            Q6AddressAdd(
                148,
                Q6Vgpr(141),
                Q6Immediate(0x4000, hexadecimal=True),
                Q6Vgpr(142),
                delay_after_low=dependency_delay,
            ),
            Q6AddressAdd(
                175, Q6Immediate(0x4000, hexadecimal=True), Q6Vgpr(141), Q6Vgpr(142)
            ),
        )
    )
    emitter.global_read_clause(
        (
            *tuple(
                Q6GlobalRead(128 + index, 134, 512 * (index + 1)) for index in range(7)
            ),
            *tuple(
                Q6GlobalRead(135 + index, 146, 512 * (index + 1)) for index in range(7)
            ),
            Q6GlobalRead(142, 151, 512),
            Q6GlobalRead(143, 151, 1024),
            Q6GlobalRead(153, 144, -4096),
            Q6GlobalRead(146, 144),
            Q6GlobalRead(145, 148, -4096),
            Q6GlobalRead(144, 148),
            *tuple(
                Q6GlobalRead(147 + index, 151, 512 * (index + 3)) for index in range(5)
            ),
            Q6GlobalRead(152, 175, 512),
            Q6GlobalRead(154, 175, 1024),
            Q6GlobalRead(155, 175, 1536),
        )
    )


def _q6_emit_dual_row_refill_reads(
    emitter: Q6ScheduleEmitter,
    layout: Q6PhysicalLayout,
) -> None:
    dep1 = Q6DependencyDelay("VALU_DEP", 1, "SKIP_1", 1)
    emitter.add_u64_batch(
        (
            Q6AddressAdd(
                4,
                Q6Immediate(0x1000, hexadecimal=True),
                Q6Vgpr(2),
                Q6Vgpr(3),
                delay_after_low=dep1,
            ),
            Q6AddressAdd(
                6, Q6Vgpr(2), Q6Immediate(0x2000, hexadecimal=True), Q6Vgpr(3)
            ),
            Q6AddressAdd(
                8,
                Q6Immediate(0x2000, hexadecimal=True),
                Q6Vgpr(2),
                Q6Vgpr(3),
                delay_after_low=Q6DependencyDelay("VALU_DEP", 1),
            ),
        )
    )
    emitter.global_read_clause(
        (
            *tuple(
                layout.refill_read(slot, 4, 512 * (slot - 8)) for slot in range(9, 16)
            ),
            layout.refill_read(17, 8, 512),
        )
    )
    emitter.add_u64_batch(
        (
            Q6AddressAdd(
                4,
                Q6Immediate(0x3000, hexadecimal=True),
                Q6Vgpr(2),
                Q6Vgpr(3),
                delay_after_low=dep1,
            ),
            Q6AddressAdd(
                111, Q6Vgpr(2), Q6Immediate(0x4000, hexadecimal=True), Q6Vgpr(3)
            ),
        )
    )
    emitter.inst("v_add_co_u32", "v2, vcc_lo, 0x4000, v2")
    emitter.global_read_clause(
        (
            *tuple(
                layout.refill_read(slot, 8, 512 * (slot - 16)) for slot in range(18, 24)
            ),
            layout.refill_read(25, 4, 512),
            layout.refill_read(26, 4, 1024),
        )
    )
    emitter.inst("v_add_co_ci_u32_e64", "v3, null, 0, v3, vcc_lo")
    emitter.global_read_clause(
        (
            layout.refill_read(8, 6, -4096),
            layout.refill_read(16, 6),
            layout.refill_read(24, 111, -4096),
            layout.refill_read(32, 111),
            *tuple(
                layout.refill_read(slot, 4, 512 * (slot - 24)) for slot in range(27, 32)
            ),
            layout.refill_read(33, 2, 512),
            layout.refill_read(34, 2, 1024),
            layout.refill_read(35, 2, 1536),
        )
    )


@dataclass(frozen=True)
class Q6StructuredLowering:
    """Emit the validated structured Q6 semantic schedule."""

    context: ForwardLoweringContext

    OPERAND_SOURCES: ClassVar[frozenset[str]] = frozenset({"Q6StructuredDecoded"})

    def body(self) -> str:
        schedule = q6_schedule_from_kernel_spec(self.context.state.kernel_spec)
        body = _emit_q6_scheduled_body(
            schedule, self.context.state.blocks_per_weight_row
        )
        return (
            str(body)
            + f".L{self.context.solution_key.kernel_name}_end:\n"
            + f".size {self.context.solution_key.kernel_name}, "
            + f".L{self.context.solution_key.kernel_name}_end - "
            + f"{self.context.solution_key.kernel_name}\n"
        )
