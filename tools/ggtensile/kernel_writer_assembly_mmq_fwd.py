from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rocisa import code  # ty: ignore[unresolved-import]
from rocisa.enum import SignatureValueKind as SVK  # ty: ignore[unresolved-import]

from .kernel_writer_assembly import (
    Assembly,
    DeterministicRegisterPlan,
    DeterministicRegisterPool,
    RegisterAssignment,
    RegisterLifetime,
    RegisterRole,
    emit_bf16_rne,
    emit_kernel_trailer,
    emit_pointer_kernarg_loads,
    initialize_rocisa,
    write_assembly_source,
)
from .mmq_fwd_spec import (
    DerivedForwardState,
    Q3HipTiledLdsLayout,
    Q6DotPhase,
    Q6ForwardSchedule,
    Q6LdsLayout,
    Q6SemanticPlan,
    Q6SemanticStage,
    QuantForwardSemantics,
    q6_schedule_from_kernel_spec,
)
from .model import ForwardSolution, ProblemSize, SolutionKey
from .quant_formats import Q8_1_F16_D4S4_BLOCK_BYTES
from .toolchain import Toolchain
from .validation import validate_solution

_Q6_SEMANTICS = QuantForwardSemantics.for_quant_type("Q6_K")
_Q6_PACKED_BLOCK_BYTES = max(
    plane.byte_offset + plane.byte_count for plane in _Q6_SEMANTICS.payload_planes
)
_Q6_SCALE_OFFSET = _Q6_SEMANTICS.payload_plane("scales").byte_offset
_Q6_FACTOR_OFFSET = _Q6_SEMANTICS.payload_plane("d").byte_offset


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
    def for_output_tile_rows(cls, output_tile_rows: int) -> "Q6OwnershipRegisterPlan":
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
    def for_output_tile_rows(cls, output_tile_rows: int) -> "Q6DecodeRegisterPlan":
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
        elif output_tile_rows == 2:
            lane_shift_register = 69
            factor_source_register = 174
            factor_register = 205
            initial_scratch_register = 173
        else:
            raise ValueError(f"unsupported Q6 decode output rows: {output_tile_rows}")

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

        if len(pool.checked_out) != 2 * atom_count:
            raise ValueError("Q6 decode pool did not retain every output role")
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


class Q6ScheduleEmitter:
    """Semantic lowering for physically ordered Q6 schedule components."""

    def __init__(self, schedule: "Q6ForwardSchedule", name: str) -> None:
        if schedule.semantic_policy.latency != "SerializedDependencyDistance":
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

    def convert_f32_f16(self, destination: int, source: Q6HalfRegister) -> None:
        self.inst("v_cvt_f32_f16_e64", f"v{destination}, {source}")

    def extract_bf16_round_bit(self, destination: int, source: int) -> None:
        self.inst("v_bfe_u32", f"v{destination}, v{source}, 16, 1")

    def bf16_nan_payload(self, destination: int, source: int) -> None:
        self.inst("v_or_b32_e32", f"v{destination}, 0x400000, v{source}")

    def compare_unordered(self, source: int) -> None:
        self.inst("v_cmp_u_f32_e32", f"vcc_lo, v{source}, v{source}")

    def round_bf16(
        self,
        destination: int,
        round_bit: int,
        source: int,
    ) -> None:
        self.inst(
            "v_add3_u32",
            f"v{destination}, v{round_bit}, v{source}, 0x7fff",
        )

    def select_bf16(
        self,
        destination: int,
        rounded: int,
        nan_payload: int,
    ) -> None:
        self.inst(
            "v_cndmask_b32_e32",
            f"v{destination}, v{rounded}, v{nan_payload}, vcc_lo",
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
    def for_output_tile_rows(cls, output_tile_rows: int) -> "Q6PhysicalRegisterMap":
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
            _Q6_SEMANTICS.q6_packed_payload_offset(atom, plane),
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


_Q6_PHYSICAL_LAYOUTS = {
    output_tile_rows: Q6PhysicalLayout(output_tile_rows) for output_tile_rows in (1, 2)
}


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
    if schedule.semantic_policy.traversal != "OutputRoleGroupMajor":
        raise ValueError("unsupported Q6 output traversal policy")
    if schedule.semantic_policy.pressure != "ExplicitRoleLifetime":
        raise ValueError("unsupported Q6 register-pressure policy")
    mi_wave_tile_m = schedule.mi_wave_tile[0]
    if mi_wave_tile_m not in _Q6_PHYSICAL_LAYOUTS:
        raise ValueError(
            f"unsupported Q6 physical layout for MIWaveTileM={mi_wave_tile_m}"
        )
    return _Q6_PHYSICAL_LAYOUTS[mi_wave_tile_m]


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
    multiply = f"v_dual_mul_f32 v{register}, v{_q6_dot_product_scale(phase, layout, local)}, v{register}"
    if pointer is None:
        emitter.instruction(multiply.replace("v_dual_mul_f32", "v_mul_f32_e32", 1))
    else:
        emitter.instruction(f"{multiply} :: {pointer}")


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


def _q6_emit_semantic_stage(
    stage: Q6SemanticStage,
    schedule: Q6ForwardSchedule,
    blocks_per_weight_row: int,
) -> code.Module:
    if stage.kind == "Setup":
        return _q6_lane_and_address_setup(schedule)
    if stage.kind == "GlobalRead":
        return _q6_cooperative_global_loads(schedule)
    if stage.kind == "Decode":
        return _q6_packed_decode(schedule)
    if stage.kind == "LocalWrite":
        return _q6_decoded_lds_writes(schedule)
    if stage.kind == "BarrierLocalRead":
        return _q6_first_stage_barrier(schedule)
    if stage.kind == "Dot":
        assert stage.phase is not None
        return _emit_q6_dot_phase(schedule, stage.phase)
    if stage.kind == "Refill":
        return _q6_second_stage_loads(schedule)
    if stage.kind == "Epilogue":
        return _q6_bf16_epilogue(schedule, blocks_per_weight_row)
    raise AssertionError(f"unhandled Q6 semantic stage {stage.kind}")


def _emit_q6_scheduled_body(
    schedule: Q6ForwardSchedule,
    blocks_per_weight_row: int = 8,
) -> code.Module:
    if blocks_per_weight_row <= 0:
        raise ValueError("Q6 blocks-per-weight-row must be positive")
    plan = Q6SemanticPlan.from_schedule(schedule)
    body = code.Module("Q6StructuredDecoded")
    for stage in plan.stages:
        body.add(_q6_emit_semantic_stage(stage, schedule, blocks_per_weight_row))
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
    emitter.convert_f32_f16(
        plan.factor_register,
        Q6HalfRegister(plan.factor_source_register, "l"),
    )
    lane_shift = Q6Vgpr(plan.lane_shift_register)
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
            emitter.extract_bf16_round_bit(scratch_base + index, source)
            emitter.bf16_nan_payload(scratch_base + width + index, source)
        if full_tile and segment_index == 0:
            _q6_emit_epilogue_store_address(emitter)
        for index, source in enumerate(sources):
            slot = index % width
            future = index + width
            emitter.compare_unordered(source)
            emitter.round_bf16(source, scratch_base + slot, source)
            if future < len(sources):
                emitter.extract_bf16_round_bit(
                    scratch_base + slot,
                    sources[future],
                )
            if emitter.schedule.dependency_delay_mode == "Explicit":
                dependency_distance = 2 if future < len(sources) else 1
                emitter.dependency_delay(
                    Q6DependencyDelay("VALU_DEP", dependency_distance)
                )
            emitter.select_bf16(
                source,
                source,
                scratch_base + width + slot,
            )
            if future < len(sources):
                emitter.bf16_nan_payload(
                    scratch_base + width + slot,
                    sources[future],
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


class ForwardKernelWriterError(RuntimeError):
    pass


@dataclass(frozen=True)
class Q8DirectGroupRole:
    """One 32-value Q8_0/Q8_1 scale and payload group."""

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
    ) -> "Q8DirectGroupRole":
        if semantics.quant_type != "Q8_0" or index not in range(4):
            raise ValueError("Q8 direct group requires Q8_0 index 0..3")
        weight_block_bytes = sum(plane.byte_count for plane in semantics.payload_planes)
        return cls(
            index=index,
            weight_block_offset=weight_block_bytes * index,
            weight_payload_offset=semantics.payload_plane("qs").byte_offset,
            activation_payload_offset=16 + 32 * index,
            activation_scale_offset=4 * index,
        )


@dataclass(frozen=True)
class Q8DirectRegisterPlan:
    """Typed deterministic register ownership for the direct Q8 control."""

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

    @classmethod
    def allocate(cls) -> "Q8DirectRegisterPlan":
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
        plan = DeterministicRegisterPlan.allocate(
            roles,
            order,
            max_registers=88,
        )
        assignments = {name: plan.assignment(name) for name in order}
        return cls(
            **assignments,
            register_count=plan.register_count,
        )


@dataclass(frozen=True)
class Q8RegisterTileRole:
    m_index: int
    n_index: int
    fragment_index: int

    @classmethod
    def all(
        cls,
        wave_tile_m: int,
        wave_tile_n: int,
    ) -> tuple["Q8RegisterTileRole", ...]:
        return tuple(
            cls(m_index, n_index, wave_tile_n * m_index + n_index)
            for m_index in range(wave_tile_m)
            for n_index in range(wave_tile_n)
        )


@dataclass(frozen=True)
class Q8RegisterTiledRegisterPlan:
    """Deterministic ownership for a four-wave Q8 register tile."""

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

    @classmethod
    def allocate(
        cls,
        wave_tile_m: int,
        wave_tile_n: int,
    ) -> "Q8RegisterTiledRegisterPlan":
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
        )


@dataclass(frozen=True)
class Q8HipTiledLdsRegisterPlan:
    """Deterministic registers for the wave-N 128x64 Q8 LDS tile."""

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

    @classmethod
    def allocate(cls) -> "Q8HipTiledLdsRegisterPlan":
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
        plan = DeterministicRegisterPlan.allocate(
            roles,
            order,
            max_registers=240,
        )
        assignments = {name: plan.assignment(name) for name in order}
        return cls(
            **assignments,
            register_count=plan.register_count,
        )


@dataclass(frozen=True)
class Q3HipTiledLdsRegisterPlan:
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

    @classmethod
    def allocate(cls) -> "Q3HipTiledLdsRegisterPlan":
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
        plan = DeterministicRegisterPlan.allocate(roles, order, max_registers=144)
        assignments = {name: plan.assignment(name) for name in order}
        return cls(**assignments, register_count=plan.register_count)


class ForwardKernelWriterAssembly:
    """Emit strict packed K-quant/Q8_1 forward controls."""

    C = 0
    SUM = 8
    WEIGHT_Q = 16
    ACTIVATION_Q = 24
    WEIGHT_METADATA = 32
    RESULT_WEIGHT_ADDRESS = 64
    WEIGHT_Q_ADDRESS = 72
    ACTIVATION_ADDRESS_0 = 73
    ACTIVATION_ADDRESS_1 = 74
    OUTPUT_ADDRESS = 75
    SCALE = 76
    MINIMUM = 77
    SCALED_DM = 78
    ACTIVATION_SCALE_SUM = 79
    WEIGHT_D = 80
    WEIGHT_MIN = 81
    ACTIVATION_D = 82
    ACTIVATION_SUM = 83
    TEMPORARY = 84
    OUTPUT_COLUMN = 85
    SERIAL = 86
    ACTIVATION_ROW = 87

    KERNARG = 4
    LOOP_COUNTER = 10
    SCALAR_TEMPORARY = 11

    def __init__(self, solution_key: SolutionKey, toolchain: Toolchain) -> None:
        reasons = validate_solution(solution_key)
        if reasons:
            details = "; ".join(
                f"{reason.rule_id}: {reason.message}" for reason in reasons
            )
            raise ForwardKernelWriterError(f"solution rejected: {details}")
        if not isinstance(solution_key.solution, ForwardSolution):
            raise ForwardKernelWriterError("forward writer requires ForwardSolution")
        self.solution_key = solution_key
        self.state = DerivedForwardState.from_solution_key(solution_key)
        self.toolchain = toolchain

    def write(self, output: Path) -> str:
        return write_assembly_source(output, self.source())

    def source(self) -> str:
        initialize_rocisa(
            self.state.contract.isa,
            self.state.contract.wavefront_size,
            self.toolchain.assembler,
            temporary_prefix="ggtensile-forward-rocisa-",
        )

        signature = code.SignatureBase(
            kernelName=self.solution_key.kernel_name,
            kernArgsVersion=0,
            codeObjectVersion="5",
            groupSegmentSize=self.state.resources.lds_bytes,
            sgprWorkGroup=(1, 1, 0),
            vgprWorkItem=(
                1
                if self.state.kernel_spec.global_memory.operand_source
                in {
                    "Q6StructuredDecoded",
                    "Q3HipTiledLds",
                    "Q8RegisterTiled",
                    "Q8HipTiledLds",
                }
                else 0
            ),
            flatWorkGroupSize=self.state.num_threads,
            totalVgprs=self.state.resources.vgprs,
            totalAgprs=0,
            totalSgprs=self.state.resources.sgprs,
        )
        signature.addDescriptionTopic(
            f"GGTensile {self.state.contract.quant_type} MMQ forward, fixed Q8_1 "
            f"{self.state.contract.activation_layout} producer"
        )
        signature.addArg("packed_weight", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("activations", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("output", SVK.SIG_GLOBALBUFFER, "bf16", "generic")
        signature.addArg("nrows_weight", SVK.SIG_VALUE, "u32")
        signature.addArg("nrows_activation", SVK.SIG_VALUE, "u32")
        signature.addArg("nrows_activation_padded", SVK.SIG_VALUE, "u32")
        signature.addArg("blocks_per_weight_row", SVK.SIG_VALUE, "u32")

        module = code.Module("GGTensileForwardKernel")
        module.add(signature)
        module.add(code.TextBlock(self._body()))
        return str(module)

    def _body(self) -> str:
        operand_source = self.state.kernel_spec.global_memory.operand_source
        if operand_source == "Q6StructuredDecoded":
            return self._body_q6_structured_decoded()
        if operand_source == "Q3HipTiledLds":
            return self._body_q3_hip_tiled_lds()
        if operand_source == "DecodedWeightLdsBatch8":
            return self._body_decoded_weight_lds()
        if operand_source == "Q8DirectGlobal":
            return self._body_q8_direct_global()
        if operand_source == "Q8RegisterTiled":
            return self._body_q8_register_tiled()
        if operand_source == "Q8HipTiledLds":
            return self._body_q8_hip_tiled_lds()
        asm = Assembly()
        name = self.solution_key.kernel_name
        quant_type = self.state.contract.quant_type
        row_stride = self.state.packed_weight_row_bytes
        activation_plane_stride = self.state.activation_plane_stride_bytes
        activation_block_stride = self.state.activation_weight_block_stride_bytes

        asm.comment(
            "Load the exact packed-weight, Q8_1 F16_D4S4 workspace, and output pointers."
        )
        emit_pointer_kernarg_loads(asm, self.KERNARG)

        asm.comment("Map one wave to an exact 16x16 output tile.")
        asm.inst(f"v_mov_b32 v{self.SERIAL}, v0")
        asm.inst(f"v_and_b32 v{self.OUTPUT_COLUMN}, 15, v{self.SERIAL}")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY}, 4, s2")
        asm.inst(
            f"v_add_nc_u32 v{self.OUTPUT_COLUMN}, v{self.TEMPORARY}, "
            f"v{self.OUTPUT_COLUMN}"
        )
        asm.inst(f"v_and_b32 v{self.ACTIVATION_ROW}, 15, v{self.SERIAL}")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY}, 4, s3")
        asm.inst(
            f"v_add_nc_u32 v{self.ACTIVATION_ROW}, v{self.TEMPORARY}, "
            f"v{self.ACTIVATION_ROW}"
        )
        asm.comment("Build one Q payload row and eight C-fragment metadata rows.")
        asm.inst(f"v_lshrrev_b32 v{self.TEMPORARY}, 4, v{self.SERIAL}")
        asm.inst(f"v_and_b32 v{self.TEMPORARY}, 1, v{self.TEMPORARY}")
        asm.inst(f"v_lshlrev_b32 v{self.SCALE}, 4, s2")
        asm.inst(f"v_add_nc_u32 v{self.TEMPORARY}, v{self.SCALE}, v{self.TEMPORARY}")
        asm.inst(
            f"v_mul_lo_u32 v{self.RESULT_WEIGHT_ADDRESS}, {row_stride}, "
            f"v{self.TEMPORARY}"
        )
        for element in range(1, 8):
            asm.inst(
                f"v_add_nc_u32 v{self.RESULT_WEIGHT_ADDRESS + element}, "
                f"{2 * element * row_stride}, v{self.RESULT_WEIGHT_ADDRESS}"
            )
        asm.inst(
            f"v_mul_lo_u32 v{self.WEIGHT_Q_ADDRESS}, {row_stride}, "
            f"v{self.OUTPUT_COLUMN}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{self.ACTIVATION_ADDRESS_0}, "
            f"{Q8_1_F16_D4S4_BLOCK_BYTES}, v{self.ACTIVATION_ROW}"
        )
        asm.inst(
            f"v_add_nc_u32 v{self.ACTIVATION_ADDRESS_1}, "
            f"{activation_plane_stride}, v{self.ACTIVATION_ADDRESS_0}"
        )
        for register in range(self.SUM, self.SUM + 8):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")

        quant_label = quant_type.replace("_", "")
        asm.label(f".LForward{quant_label}BlockLoop")
        asm.comment(
            f"Keep one {quant_type} block's dm/scales live across its eight groups."
        )
        for element in range(8):
            metadata = self.WEIGHT_METADATA + 4 * element
            asm.inst(
                f"global_load_b128 v[{metadata}:{metadata + 3}], "
                f"v{self.RESULT_WEIGHT_ADDRESS + element}, "
                f"s[{self.KERNARG}:{self.KERNARG + 1}]"
            )
        for group in range(8):
            self._emit_group(asm, group)
        for element in range(8):
            asm.inst(
                f"v_add_nc_u32 v{self.RESULT_WEIGHT_ADDRESS + element}, "
                f"{self.state.contract.packed_weight_block_bytes}, "
                f"v{self.RESULT_WEIGHT_ADDRESS + element}"
            )
        asm.inst(
            f"v_add_nc_u32 v{self.WEIGHT_Q_ADDRESS}, "
            f"{self.state.contract.packed_weight_block_bytes}, v{self.WEIGHT_Q_ADDRESS}"
        )
        asm.inst(
            f"v_add_nc_u32 v{self.ACTIVATION_ADDRESS_0}, "
            f"{activation_block_stride}, v{self.ACTIVATION_ADDRESS_0}"
        )
        asm.inst(
            f"v_add_nc_u32 v{self.ACTIVATION_ADDRESS_1}, "
            f"{activation_block_stride}, v{self.ACTIVATION_ADDRESS_1}"
        )
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(
            f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {self.state.blocks_per_weight_row}"
        )
        asm.inst(f"s_cbranch_scc1 .LForward{quant_label}BlockLoop")

        self._emit_store(asm)
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _body_q8_direct_global(self) -> str:
        """Lower the isolated one-wave Q8_0 direct-global correctness control."""
        asm = Assembly()
        registers = Q8DirectRegisterPlan.allocate()
        name = self.solution_key.kernel_name
        row_stride = self.state.packed_weight_row_bytes
        activation_plane_stride = self.state.activation_plane_stride_bytes
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
            f"{self.state.contract.activation_block_bytes}, v{activation_row}"
        )
        for register in range(sums, sums + 8):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")

        asm.label(".LForwardQ80ActivationBlockLoop")
        for group in range(4):
            self._emit_q8_direct_group(
                asm,
                Q8DirectGroupRole.from_semantics(self.state.semantics, group),
                registers,
            )
            for element in range(8):
                asm.inst(
                    f"v_add_nc_u32 v{result_addresses + element}, "
                    f"{self.state.contract.packed_weight_block_bytes}, "
                    f"v{result_addresses + element}"
                )
            asm.inst(
                f"v_add_nc_u32 v{registers.weight_address.first_register}, "
                f"{self.state.contract.packed_weight_block_bytes}, "
                f"v{registers.weight_address.first_register}"
            )
        asm.inst(
            f"v_add_nc_u32 v{registers.activation_address.first_register}, "
            f"{activation_plane_stride}, "
            f"v{registers.activation_address.first_register}"
        )
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(
            f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {self.state.activation_blocks_per_row}"
        )
        asm.inst("s_cbranch_scc1 .LForwardQ80ActivationBlockLoop")

        self._emit_q8_direct_store(asm, registers)
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_q8_direct_group(
        self,
        asm: Assembly,
        group: Q8DirectGroupRole,
        registers: Q8DirectRegisterPlan,
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
        asm.inst(
            f"v_wmma_i32_16x16x16_iu8 v[{c}:{c + 7}], "
            f"v[{weight_payload}:{weight_payload + 3}], "
            f"v[{activation_payload}:{activation_payload + 3}], "
            f"v[{c}:{c + 7}] neg_lo:[1,1,0]"
        )
        asm.inst(
            f"v_wmma_i32_16x16x16_iu8 v[{c}:{c + 7}], "
            f"v[{weight_payload + 4}:{weight_payload + 7}], "
            f"v[{activation_payload + 4}:{activation_payload + 7}], "
            f"v[{c}:{c + 7}] neg_lo:[1,1,0]"
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

    def _emit_q8_direct_store(
        self,
        asm: Assembly,
        registers: Q8DirectRegisterPlan,
    ) -> None:
        size = self.state.problem_size
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

    def _body_q8_register_tiled(self) -> str:
        """Lower a four-wave Q8_0 workgroup with four fragments per wave."""
        asm = Assembly()
        wave_tile_m, wave_tile_n = self.state.mi_wave_tile
        registers = Q8RegisterTiledRegisterPlan.allocate(wave_tile_m, wave_tile_n)
        name = self.solution_key.kernel_name
        row_stride = self.state.packed_weight_row_bytes
        activation_plane_stride = self.state.activation_plane_stride_bytes
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
                f"{self.solution_key.solution.macro_tile1.bit_length() - 1}, s2"
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
                f"{self.solution_key.solution.macro_tile1.bit_length() - 1}, s2"
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
                f"{self.solution_key.solution.macro_tile0.bit_length() - 1}, s3"
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
                f"{self.state.contract.activation_block_bytes}, v{activation_row}"
            )

        for register in range(sums, sums + 32):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")

        asm.label(".LForwardQ80RegisterTiledActivationBlockLoop")
        for group in range(4):
            group_role = Q8DirectGroupRole.from_semantics(self.state.semantics, group)
            self._emit_q8_register_tiled_group(asm, group_role, registers)
        packed_activation_block_bytes = (
            4 * self.state.contract.packed_weight_block_bytes
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
            f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {self.state.activation_blocks_per_row}"
        )
        asm.inst("s_cbranch_scc1 .LForwardQ80RegisterTiledActivationBlockLoop")

        self._emit_q8_register_tiled_store(asm, registers)
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_q8_register_tiled_group(
        self,
        asm: Assembly,
        group: Q8DirectGroupRole,
        registers: Q8RegisterTiledRegisterPlan,
    ) -> None:
        wave_tile_m, wave_tile_n = self.state.mi_wave_tile
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
        self._emit_q8_register_tile_dot(asm, registers)

    def _emit_q8_register_tile_dot(
        self,
        asm: Assembly,
        registers: Q8RegisterTiledRegisterPlan,
    ) -> None:
        wave_tile_m, wave_tile_n = self.state.mi_wave_tile
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

        for tile in Q8RegisterTileRole.all(wave_tile_m, wave_tile_n):
            fragment = 8 * tile.fragment_index
            c_fragment = c + fragment
            sum_fragment = sums + fragment
            weight_payload = weight_payloads + 8 * tile.n_index
            activation_payload = activation_payloads + 8 * tile.m_index
            weight_scale = weight_scales + 8 * tile.n_index
            activation_scale = activation_scales + tile.m_index
            for register in range(c_fragment, c_fragment + 8):
                asm.inst(f"v_mov_b32 v{register}, 0")
            asm.inst(
                f"v_wmma_i32_16x16x16_iu8 v[{c_fragment}:{c_fragment + 7}], "
                f"v[{weight_payload}:{weight_payload + 3}], "
                f"v[{activation_payload}:{activation_payload + 3}], "
                f"v[{c_fragment}:{c_fragment + 7}] neg_lo:[1,1,0]"
            )
            asm.inst(
                f"v_wmma_i32_16x16x16_iu8 v[{c_fragment}:{c_fragment + 7}], "
                f"v[{weight_payload + 4}:{weight_payload + 7}], "
                f"v[{activation_payload + 4}:{activation_payload + 7}], "
                f"v[{c_fragment}:{c_fragment + 7}] neg_lo:[1,1,0]"
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

    def _emit_q8_register_tiled_store(
        self,
        asm: Assembly,
        registers: Q8RegisterTiledRegisterPlan,
    ) -> None:
        wave_tile_m, wave_tile_n = self.state.mi_wave_tile
        size = self.state.problem_size
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
            f"{self.solution_key.solution.macro_tile0.bit_length() - 1}, s3"
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
            f"{self.solution_key.solution.macro_tile1.bit_length() - 1}, s2"
        )
        asm.inst(f"v_add_nc_u32 v{store_column}, v{store_column}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{store_column}, 1, v{store_column}")
        asm.inst(f"v_add_nc_u32 v{output_address}, v{output_address}, v{store_column}")

        for tile_index, tile in enumerate(
            Q8RegisterTileRole.all(wave_tile_m, wave_tile_n)
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
        for tile in Q8RegisterTileRole.all(wave_tile_m, wave_tile_n):
            address_base = output_address + 8 * tile.fragment_index
            fragment = sums + 8 * tile.fragment_index
            for element in range(8):
                total = fragment + element
                asm.inst(
                    f"global_store_d16_hi_b16 v{address_base + element}, v{total}, "
                    f"s[{self.KERNARG + 4}:{self.KERNARG + 5}]"
                )

    def _body_q3_hip_tiled_lds(self) -> str:
        """Lower the isolated wave-N 128x64 Q3_K LDS research control."""
        asm = Assembly()
        registers = Q3HipTiledLdsRegisterPlan.allocate()
        layout = Q3HipTiledLdsLayout()
        name = self.solution_key.kernel_name
        row_stride = self.state.packed_weight_row_bytes
        activation_plane_stride = self.state.activation_plane_stride_bytes
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
        emit_pointer_kernarg_loads(asm, self.KERNARG)
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
        self._emit_q3_weight_prefetch(asm, registers, half=0)
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")

        asm.label(".LForwardQ3KHipTiledLdsBlockLoop")
        for half in range(2):
            self._emit_q3_activation_stage(asm, registers, layout)
            self._emit_q3_weight_stage(
                asm,
                registers,
                layout,
                half,
                preloaded=half == 0,
                store_activation=True,
            )
            asm.inst("s_waitcnt lgkmcnt(0)")
            asm.inst("s_barrier")
            self._emit_q3_half_accumulate(asm, registers, layout)
            asm.inst("s_barrier")
            asm.inst(
                f"v_add_nc_u32 v{activation_address}, "
                f"{activation_plane_stride}, v{activation_address}"
            )
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(
            f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {self.state.blocks_per_weight_row}"
        )
        asm.inst("s_cbranch_scc0 .LForwardQ3KHipTiledLdsEnd")
        asm.inst(
            f"v_add_nc_u32 v{weight_address}, "
            f"{self.state.contract.packed_weight_block_bytes}, v{weight_address}"
        )
        self._emit_q3_weight_prefetch(asm, registers, half=0)
        asm.inst("s_branch .LForwardQ3KHipTiledLdsBlockLoop")
        asm.label(".LForwardQ3KHipTiledLdsEnd")

        self._emit_q3_store(asm, registers)
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_q3_activation_stage(
        self,
        asm: Assembly,
        registers: Q3HipTiledLdsRegisterPlan,
        layout: Q3HipTiledLdsLayout,
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

    def _emit_q3_activation_stores(
        self,
        asm: Assembly,
        registers: Q3HipTiledLdsRegisterPlan,
        layout: Q3HipTiledLdsLayout,
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

    def _emit_q3_weight_prefetch(
        self,
        asm: Assembly,
        registers: Q3HipTiledLdsRegisterPlan,
        *,
        half: int,
    ) -> None:
        """Issue one Q3 half's raw VMEM reads for the following block."""
        semantics = self.state.semantics
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

    def _emit_q3_weight_stage(
        self,
        asm: Assembly,
        registers: Q3HipTiledLdsRegisterPlan,
        layout: Q3HipTiledLdsLayout,
        half: int,
        *,
        preloaded: bool = False,
        store_activation: bool = False,
    ) -> None:
        """Decode four interleaved Q3 groups per thread into one LDS half row."""
        semantics = self.state.semantics
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
            self._emit_q3_activation_stores(asm, registers, layout)

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

    def _emit_q3_half_accumulate(
        self,
        asm: Assembly,
        registers: Q3HipTiledLdsRegisterPlan,
        layout: Q3HipTiledLdsLayout,
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
            group_spec = self.state.semantics.q3_payload_group(group)
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
                asm.inst(
                    f"v_wmma_i32_16x16x16_iu8 v[{c}:{c + 7}], "
                    f"v[{weight_payload}:{weight_payload + 3}], "
                    f"v[{payload}:{payload + 3}], "
                    f"v[{zero_accumulator}:{zero_accumulator + 7}] "
                    "neg_lo:[1,1,0]"
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

    def _emit_q3_store(
        self,
        asm: Assembly,
        registers: Q3HipTiledLdsRegisterPlan,
    ) -> None:
        """Store eight wave-N fragments as row-major BF16 with RNE."""
        size = self.state.problem_size
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

    def _body_q8_hip_tiled_lds(self) -> str:
        """Lower a wave-N 128x64 Q8 tile with cooperative LDS operands."""
        asm = Assembly()
        registers = Q8HipTiledLdsRegisterPlan.allocate()
        name = self.solution_key.kernel_name
        size = self.state.problem_size
        row_stride = self.state.packed_weight_row_bytes
        activation_plane_stride = self.state.activation_plane_stride_bytes
        depth_u = self.state.kernel_spec.geometry.depth_u
        groups_per_iteration = depth_u // 8
        activation_planes_per_iteration = groups_per_iteration // 4
        iteration_count = (
            self.state.activation_blocks_per_row // activation_planes_per_iteration
        )
        c = registers.c.first_register
        sums = registers.sums.first_register
        weight_payload = registers.weight_payload.first_register
        activation_payloads = registers.activation_payloads.first_register
        weight_scales = registers.weight_scales.first_register
        activation_scales = registers.activation_scales.first_register
        activation_scale_copies = registers.activation_scale_copies.first_register
        zero_accumulator = registers.zero_accumulator.first_register
        weight_stage_payload = registers.weight_stage_payload.first_register
        weight_scale_address = registers.weight_scale_address.first_register
        weight_address = registers.weight_address.first_register
        activation_address = registers.activation_address.first_register
        activation_lds_address = registers.activation_lds_address.first_register
        weight_lds_address = registers.weight_lds_address.first_register
        weight_stage_address = registers.weight_stage_address.first_register
        weight_scale_stage_address = registers.weight_scale_stage_address.first_register
        temporary = registers.temporary.first_register
        lane = registers.lane.first_register
        wave = registers.wave.first_register
        activation_row = registers.activation_row.first_register
        activation_read_address = registers.activation_read_address.first_register
        weight_row = registers.weight_row.first_register
        output_address = registers.output_address.first_register
        store_auxiliary = registers.store_auxiliary.first_register

        activation_lds_row_stride = self.state.contract.activation_block_bytes
        weight_lds_base = 18_432
        weight_lds_row_stride = 304

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
            f"{activation_lds_row_stride * self.state.kernel_spec.macro_tile[0]}, s3"
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
        self._emit_q8_hip_activation_loads(
            asm,
            activation_address,
            activation_payloads,
            activation_scales,
        )
        self._emit_q8_hip_weight_stage(
            asm,
            weight_address,
            weight_lds_address,
            weight_stage_address,
            weight_scale_stage_address,
            weight_payload,
            weight_stage_payload,
            weight_scales,
            lane,
            temporary,
            group_base=0,
        )
        self._emit_q8_hip_activation_writes(
            asm,
            activation_lds_address,
            activation_payloads,
            activation_scales,
        )
        self._emit_q8_hip_weight_writes(
            asm,
            weight_payload,
            weight_stage_payload,
            weight_scales,
            weight_scale_stage_address,
            temporary,
            group_base=0,
        )
        if groups_per_iteration == 8:
            self._emit_q8_hip_weight_stage(
                asm,
                weight_address,
                weight_lds_address,
                weight_stage_address,
                weight_scale_stage_address,
                weight_payload,
                weight_stage_payload,
                weight_scales,
                lane,
                temporary,
                group_base=4,
            )
            self._emit_q8_hip_weight_writes(
                asm,
                weight_payload,
                weight_stage_payload,
                weight_scales,
                weight_scale_stage_address,
                temporary,
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
        for group in range(4):
            self._emit_q8_hip_group(
                asm,
                group,
                c,
                sums,
                weight_payload,
                activation_payloads,
                weight_scales,
                activation_scales,
                activation_scale_copies,
                weight_scale_address,
                weight_lds_address,
                activation_read_address,
                zero_accumulator=zero_accumulator,
            )
        if groups_per_iteration == 8:
            asm.inst(
                f"v_add_nc_u32 v{temporary}, {activation_plane_stride}, "
                f"v{activation_address}"
            )
            self._emit_q8_hip_activation_loads(
                asm,
                temporary,
                activation_payloads,
                activation_scales,
            )
            asm.inst("s_barrier")
            self._emit_q8_hip_activation_writes(
                asm,
                activation_lds_address,
                activation_payloads,
                activation_scales,
                trailing_vmem=0,
            )
            asm.inst("s_waitcnt lgkmcnt(0)")
            asm.inst("s_barrier")
            for group in range(4, 8):
                self._emit_q8_hip_group(
                    asm,
                    group,
                    c,
                    sums,
                    weight_payload,
                    activation_payloads,
                    weight_scales,
                    activation_scales,
                    activation_scale_copies,
                    weight_scale_address,
                    weight_lds_address,
                    activation_read_address,
                    activation_group=group - 4,
                    zero_accumulator=zero_accumulator,
                )
        asm.inst("s_barrier")
        asm.inst(
            f"v_add_nc_u32 v{weight_address}, "
            f"{groups_per_iteration * self.state.contract.packed_weight_block_bytes}, "
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

        self._emit_q8_hip_store(
            asm,
            sums,
            output_address,
            store_auxiliary,
            lane,
            wave,
            temporary,
            size,
        )
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_q8_hip_activation_loads(
        self,
        asm: Assembly,
        activation_address: int,
        activation_payloads: int,
        activation_scales: int,
    ) -> None:
        """Issue the cooperative activation loads before the weight loads."""
        asm.inst("s_clause 11")
        for group in range(4):
            payload = activation_payloads + 8 * group
            scale = activation_scales + group
            asm.inst(
                f"global_load_b128 v[{payload}:{payload + 3}], v{activation_address}, "
                f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] offset:{16 + 32 * group}"
            )
            asm.inst(
                f"global_load_b128 v[{payload + 4}:{payload + 7}], v{activation_address}, "
                f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] offset:{32 + 32 * group}"
            )
            asm.inst(
                f"global_load_b32 v{scale}, v{activation_address}, "
                f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] offset:{4 * group}"
            )

    def _emit_q8_hip_activation_writes(
        self,
        asm: Assembly,
        lds_address: int,
        activation_payloads: int,
        activation_scales: int,
        *,
        trailing_vmem: int = 6,
    ) -> None:
        """Commit the completed activation loads to their LDS rows."""
        for group in range(4):
            asm.inst(f"s_waitcnt vmcnt({9 + trailing_vmem - 3 * group})")
            payload = activation_payloads + 8 * group
            scale = activation_scales + group
            asm.inst(
                f"ds_write_b128 v{lds_address}, v[{payload}:{payload + 3}] "
                f"offset:{16 + 32 * group}"
            )
            asm.inst(
                f"ds_write_b128 v{lds_address}, v[{payload + 4}:{payload + 7}] "
                f"offset:{32 + 32 * group}"
            )
            asm.inst(f"ds_write_b32 v{lds_address}, v{scale} offset:{4 * group}")

    def _emit_q8_hip_weight_stage(
        self,
        asm: Assembly,
        weight_address: int,
        weight_lds_address: int,
        weight_stage_address: int,
        weight_scale_stage_address: int,
        weight_payload: int,
        weight_stage_payload: int,
        weight_scales: int,
        lane: int,
        temporary: int,
        *,
        group_base: int,
    ) -> None:
        """Stage four packed Q8_0 groups and their FP32 scales in LDS."""
        weight_lds_scale_offset = 256
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
            f"{weight_lds_scale_offset}, v{weight_scale_stage_address}"
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

    def _emit_q8_hip_weight_writes(
        self,
        asm: Assembly,
        weight_payload: int,
        weight_stage_payload: int,
        weight_scales: int,
        weight_scale_stage_address: int,
        lds_address: int,
        *,
        group_base: int,
    ) -> None:
        """Commit the packed-weight loads after their VMEM dependencies mature."""
        for group, wait_count in enumerate((3, 0)):
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

    def _emit_q8_hip_group(
        self,
        asm: Assembly,
        group: int,
        c: int,
        sums: int,
        weight_payload: int,
        activation_payloads: int,
        weight_scales: int,
        activation_scales: int,
        activation_scale_copies: int,
        weight_scale_address: int,
        weight_lds_address: int,
        activation_read_address: int,
        *,
        activation_group: int | None = None,
        zero_accumulator: int,
    ) -> None:
        """Read one staged Q8 group and accumulate eight activation fragments."""
        activation_group = group if activation_group is None else activation_group
        weight_lds_scale_offset = 256
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
        for element in range(8):
            asm.inst(
                f"ds_read_b32 v{weight_scales + element}, "
                f"v{weight_scale_address} "
                f"offset:{weight_lds_scale_offset + 608 * element + 4 * group}"
            )
        for m_index in range(8):
            payload = activation_payloads + 8 * m_index
            activation_row_offset = (
                16 * m_index * self.state.contract.activation_block_bytes
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
        for m_index in range(8):
            scale_pair_tail = 1 if m_index % 2 == 0 else 0
            asm.inst(f"s_waitcnt lgkmcnt({21 - 3 * (m_index + scale_pair_tail)})")
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
            asm.inst(
                f"v_wmma_i32_16x16x16_iu8 v[{c_fragment}:{c_fragment + 7}], "
                f"v[{weight_payload}:{weight_payload + 3}], "
                f"v[{payload}:{payload + 3}], "
                f"v[{zero_accumulator}:{zero_accumulator + 7}] neg_lo:[1,1,0]"
            )
            asm.inst(
                f"v_wmma_i32_16x16x16_iu8 v[{c_fragment}:{c_fragment + 7}], "
                f"v[{weight_payload + 4}:{weight_payload + 7}], "
                f"v[{payload + 4}:{payload + 7}], "
                f"v[{c_fragment}:{c_fragment + 7}] neg_lo:[1,1,0]"
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

    def _emit_q8_hip_store(
        self,
        asm: Assembly,
        sums: int,
        output_address: int,
        store_auxiliary: int,
        lane: int,
        wave: int,
        temporary: int,
        size: ProblemSize,
    ) -> None:
        """Store eight wave-N fragments with one legal global clause."""
        asm.comment("Store eight wave-N fragments as row-major BF16.")
        asm.inst(f"v_and_b32 v{temporary}, 15, v{lane}")
        asm.inst(f"v_lshlrev_b32 v{output_address}, 7, s3")
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
        for m_index in range(8):
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
        asm.inst("s_clause 63")
        for m_index in range(8):
            address_base = output_address + 8 * m_index
            fragment = sums + 8 * m_index
            for element in range(8):
                asm.inst(
                    f"global_store_d16_hi_b16 v{address_base + element}, "
                    f"v{fragment + element}, s[{self.KERNARG + 4}:{self.KERNARG + 5}]"
                )

    def _emit_decoded_weight_lds_stage(
        self,
        asm: Assembly,
        *,
        row_stride: int,
        serial: int,
        wave: int,
        temporary: int,
        lds_address: int,
        auxiliary: int,
        metadata_address: int,
        staging_base: int,
    ) -> None:
        decoded_lds = self.state.decoded_lds
        if decoded_lds is None:
            raise ForwardKernelWriterError("decoded LDS layout is unavailable")
        weight_lds_base = decoded_lds.weight_data_base
        weight_lds_stride = decoded_lds.weight_row_stride
        quant_type = self.state.contract.quant_type
        quant_label = quant_type.replace("_", "")
        semantics = self.state.semantics
        high_bits = semantics.high_bit_reconstruction()
        ql_offset = semantics.payload_plane("ql").byte_offset
        qh_offset = (
            semantics.payload_plane(high_bits.plane_name).byte_offset
            if high_bits is not None
            else 0
        )
        qh_address = auxiliary + 1
        payload_reads_per_slice = 1 + (high_bits is not None)

        asm.comment(f"Cooperatively decode {quant_type} payload into padded LDS rows.")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 3, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{metadata_address}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{metadata_address}, v{temporary}")
        asm.inst(f"v_mul_lo_u32 v{temporary}, {row_stride}, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{temporary}, s{self.SCALAR_TEMPORARY}, v{temporary}")
        asm.inst(f"v_and_b32 v{auxiliary}, 7, v{serial}")
        if high_bits is not None:
            asm.comment("Build Q5_K high-bit and low-nibble payload addresses.")
            asm.inst(
                f"v_and_b32 v{qh_address}, {high_bits.address_lane_mask}, v{auxiliary}"
            )
            asm.inst(
                f"v_mad_u32_u24 v{qh_address}, {high_bits.address_lane_stride}, "
                f"v{qh_address}, v{temporary}"
            )
            asm.inst(f"v_mad_u32_u24 v{temporary}, 16, v{auxiliary}, v{temporary}")
        else:
            asm.inst(f"v_lshlrev_b32 v{metadata_address}, 4, v{auxiliary}")
            asm.inst(f"v_add_nc_u32 v{temporary}, 16, v{temporary}")
            asm.inst(f"v_add_nc_u32 v{temporary}, v{metadata_address}, v{temporary}")

        asm.inst(f"v_lshrrev_b32 v{lds_address}, 3, v{serial}")
        asm.inst(f"v_mul_lo_u32 v{lds_address}, {weight_lds_stride}, v{lds_address}")
        asm.inst(f"v_lshrrev_b32 v{metadata_address}, 1, v{auxiliary}")
        asm.inst(f"v_lshlrev_b32 v{metadata_address}, 6, v{metadata_address}")
        asm.inst(f"v_and_b32 v{auxiliary}, 1, v{auxiliary}")
        asm.inst(f"v_lshlrev_b32 v{auxiliary}, 4, v{auxiliary}")
        asm.inst(
            f"v_add3_u32 v{lds_address}, v{metadata_address}, "
            f"v{auxiliary}, v{lds_address}"
        )
        asm.inst(f"v_add_nc_u32 v{lds_address}, {weight_lds_base}, v{lds_address}")

        metadata_base = staging_base + 32
        asm.inst(f"v_readfirstlane_b32 s12, v{wave}")
        asm.inst("s_cmp_lt_u32 s12, 2")
        asm.inst(f"s_cbranch_scc0 .LForward{quant_label}DecodedMetadataLoadDone")
        asm.inst(f"v_lshlrev_b32 v{metadata_address}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{metadata_address}, v{serial}, v{metadata_address}")
        asm.inst(f"v_mul_lo_u32 v{metadata_address}, {row_stride}, v{metadata_address}")
        asm.inst(
            f"v_add_nc_u32 v{metadata_address}, s{self.SCALAR_TEMPORARY}, "
            f"v{metadata_address}"
        )
        asm.inst(
            f"global_load_b128 v[{metadata_base}:{metadata_base + 3}], "
            f"v{metadata_address}, s[{self.KERNARG}:{self.KERNARG + 1}]"
        )
        asm.label(f".LForward{quant_label}DecodedMetadataLoadDone")

        if high_bits is not None:
            qh_base = staging_base + 36
            for row_slice in range(4):
                raw_base = staging_base + 8 * row_slice
                row_qh_base = qh_base + 4 * row_slice
                asm.inst(
                    f"global_load_b128 v[{row_qh_base}:{row_qh_base + 3}], "
                    f"v{qh_address}, s[{self.KERNARG}:{self.KERNARG + 1}] "
                    f"offset:{qh_offset}"
                )
                asm.inst(
                    f"global_load_b128 v[{raw_base}:{raw_base + 3}], "
                    f"v{temporary}, s[{self.KERNARG}:{self.KERNARG + 1}] "
                    f"offset:{ql_offset}"
                )
                if row_slice != 3:
                    asm.inst(
                        f"v_add_nc_u32 v{qh_address}, {16 * row_stride}, v{qh_address}"
                    )
                    asm.inst(
                        f"v_add_nc_u32 v{temporary}, {16 * row_stride}, v{temporary}"
                    )
            asm.inst(f"v_and_b32 v{qh_address}, {high_bits.lane_shift_mask}, v{serial}")
        else:
            for row_slice in range(4):
                raw_base = staging_base + 8 * row_slice
                asm.inst(
                    f"global_load_b128 v[{raw_base}:{raw_base + 3}], "
                    f"v{temporary}, s[{self.KERNARG}:{self.KERNARG + 1}]"
                )
                if row_slice != 3:
                    asm.inst(
                        f"v_add_nc_u32 v{temporary}, {16 * row_stride}, v{temporary}"
                    )
        for row_slice in range(4):
            raw_base = staging_base + 8 * row_slice
            asm.inst(f"s_waitcnt vmcnt({payload_reads_per_slice * (3 - row_slice)})")
            if high_bits is not None:
                qh_base = staging_base + 36 + 4 * row_slice
                for item in range(4):
                    asm.inst(
                        f"v_lshrrev_b32 v{qh_base + item}, v{qh_address}, "
                        f"v{qh_base + item}"
                    )
                for item in range(4):
                    asm.inst(
                        f"v_lshrrev_b32 v{raw_base + 4 + item}, "
                        f"{high_bits.nibble_shift}, v{raw_base + item}"
                    )
                for item in range(4):
                    asm.inst(
                        f"v_and_b32 v{raw_base + item}, "
                        f"{high_bits.nibble_mask:#010x}, v{raw_base + item}"
                    )
                    asm.inst(
                        f"v_and_b32 v{raw_base + 4 + item}, "
                        f"{high_bits.nibble_mask:#010x}, "
                        f"v{raw_base + 4 + item}"
                    )
                for item in range(4):
                    low = raw_base + item
                    high = raw_base + 4 + item
                    qh = qh_base + item
                    asm.inst(
                        f"v_and_b32 v{auxiliary}, {high_bits.low_mask:#010x}, v{qh}"
                    )
                    asm.inst(
                        f"v_lshl_or_b32 v{low}, v{auxiliary}, "
                        f"{high_bits.low_destination_shift}, v{low}"
                    )
                    asm.inst(
                        f"v_and_b32 v{auxiliary}, {high_bits.high_mask:#010x}, v{qh}"
                    )
                    asm.inst(
                        f"v_lshl_or_b32 v{high}, v{auxiliary}, "
                        f"{high_bits.high_destination_shift}, v{high}"
                    )
            else:
                for item in range(4):
                    low = raw_base + item
                    high = raw_base + 4 + item
                    asm.inst(f"v_lshrrev_b32 v{high}, 4, v{low}")
                    asm.inst(f"v_and_b32 v{low}, 0x0f0f0f0f, v{low}")
                    asm.inst(f"v_and_b32 v{high}, 0x0f0f0f0f, v{high}")
            asm.inst(f"ds_write_b128 v{lds_address}, v[{raw_base}:{raw_base + 3}]")
            asm.inst(
                f"ds_write_b128 v{lds_address}, "
                f"v[{raw_base + 4}:{raw_base + 7}] offset:32"
            )
            if row_slice != 3:
                asm.inst(
                    f"v_add_nc_u32 v{lds_address}, "
                    f"{16 * weight_lds_stride}, v{lds_address}"
                )

        asm.comment(
            f"Compute each packed {quant_type} scale/min pair once per weight row."
        )
        asm.inst("s_cmp_lt_u32 s12, 2")
        asm.inst(f"s_cbranch_scc0 .LForward{quant_label}DecodedMetadataDone")
        asm.inst(f"v_mul_lo_u32 v{lds_address}, {weight_lds_stride}, v{serial}")
        asm.inst(
            f"v_add_nc_u32 v{lds_address}, {weight_lds_base + 256}, v{lds_address}"
        )
        metadata_schedule = self.state.kernel_spec.decode.metadata_schedule
        if metadata_schedule in (
            "IndependentExtraction",
            "IndependentExtractionMetadataAfterLowWmma",
        ):
            # Expose the eight metadata fields before conversion so the
            # conversion/product chains do not serialize on v72:v77.
            for group in range(4):
                fields = semantics.packed_scale_minimum_fields(group)
                scale_part = fields.scale[0]
                minimum_part = fields.minimum[0]
                asm.inst(
                    f"v_bfe_u32 v{72 + group}, "
                    f"v{metadata_base + scale_part.metadata_word}, "
                    f"{scale_part.bit_offset}, {scale_part.bit_count}"
                )
                asm.inst(
                    f"v_bfe_u32 v{80 + group}, "
                    f"v{metadata_base + minimum_part.metadata_word}, "
                    f"{minimum_part.bit_offset}, {minimum_part.bit_count}"
                )
            upper_fields = tuple(
                semantics.packed_scale_minimum_fields(group) for group in range(4, 8)
            )
            for packed, fields in enumerate(upper_fields):
                group = 4 + packed
                scale_low, scale_high = fields.scale
                minimum_low, minimum_high = fields.minimum
                asm.inst(
                    f"v_bfe_u32 v{72 + group}, "
                    f"v{metadata_base + scale_low.metadata_word}, "
                    f"{scale_low.bit_offset}, {scale_low.bit_count}"
                )
                asm.inst(
                    f"v_bfe_u32 v{80 + group}, "
                    f"v{metadata_base + minimum_low.metadata_word}, "
                    f"{minimum_low.bit_offset}, {minimum_low.bit_count}"
                )
                asm.inst(
                    f"v_bfe_u32 v{88 + packed}, "
                    f"v{metadata_base + scale_high.metadata_word}, "
                    f"{scale_high.bit_offset}, {scale_high.bit_count}"
                )
                asm.inst(
                    f"v_bfe_u32 v{92 + packed}, "
                    f"v{metadata_base + minimum_high.metadata_word}, "
                    f"{minimum_high.bit_offset}, {minimum_high.bit_count}"
                )
            for packed, fields in enumerate(upper_fields):
                group = 4 + packed
                scale_high = fields.scale[1]
                minimum_high = fields.minimum[1]
                asm.inst(
                    f"v_lshl_or_b32 v{72 + group}, v{88 + packed}, "
                    f"{scale_high.destination_shift}, v{72 + group}"
                )
                asm.inst(
                    f"v_lshl_or_b32 v{80 + group}, v{92 + packed}, "
                    f"{minimum_high.destination_shift}, v{80 + group}"
                )
            for group in range(8):
                asm.inst(f"v_cvt_f16_u16_e32 v{96 + group}.l, v{72 + group}.l")
            for group in range(8):
                asm.inst(f"v_cvt_f16_u16_e32 v{96 + group}.h, v{80 + group}.l")
            for group in range(8):
                asm.inst(f"v_pk_mul_f16 v{96 + group}, 0xbc003c00, v{96 + group}")
            for group in range(8):
                asm.inst(f"v_pk_mul_f16 v{96 + group}, v{metadata_base}, v{96 + group}")
            for group in range(8):
                asm.inst(
                    f"ds_write_b32 v{lds_address}, v{96 + group} offset:{4 * group}"
                )
        else:
            for group in range(8):
                self._emit_scale_and_minimum(
                    asm,
                    group,
                    metadata_base,
                    scale=72,
                    minimum=73,
                    temporary=74,
                )
                asm.inst("v_cvt_f16_u16_e32 v77.l, v72.l")
                asm.inst("v_cvt_f16_u16_e32 v77.h, v73.l")
                asm.inst("v_pk_mul_f16 v77, 0xbc003c00, v77")
                asm.inst(f"v_pk_mul_f16 v77, v{metadata_base}, v77")
                asm.inst(f"ds_write_b32 v{lds_address}, v77 offset:{4 * group}")
        asm.label(f".LForward{quant_label}DecodedMetadataDone")

    def _body_q6_structured_decoded(self) -> str:
        schedule = q6_schedule_from_kernel_spec(self.state.kernel_spec)
        body = _emit_q6_scheduled_body(schedule, self.state.blocks_per_weight_row)
        return (
            str(body)
            + f".L{self.solution_key.kernel_name}_end:\n"
            + f".size {self.solution_key.kernel_name}, "
            + f".L{self.solution_key.kernel_name}_end - "
            + f"{self.solution_key.kernel_name}\n"
        )

    def _body_decoded_weight_lds(self) -> str:
        """Lower decoded-weight LDS staging and rolled four-group batches."""
        asm = Assembly()
        size = self.state.problem_size
        name = self.solution_key.kernel_name
        quant_type = self.state.contract.quant_type
        row_stride = self.state.packed_weight_row_bytes
        activation_plane_stride = self.state.activation_plane_stride_bytes
        decoded_lds = self.state.decoded_lds
        if decoded_lds is None:
            raise ForwardKernelWriterError("decoded LDS layout is unavailable")
        activation_lds_base = decoded_lds.activation_base
        activation_lane_stride = decoded_lds.activation_lane_stride
        sum_base = 8
        weight_q = 72
        metadata = 80
        c_base = 112
        low_activation_last = 176
        high_activation_base = 180
        activation_scale_sum_base = 212
        scaled_dm_base = 220
        temporary = 228
        lds_address = 229
        auxiliary = 230
        metadata_address = 232
        activation_plane_address = 233
        output_column = 234
        wave_column_base = 235
        wave = 236
        lane = 237
        serial = 238
        quant_label = quant_type.replace("_", "")

        asm.comment(
            "Load the exact packed-weight, Q8_1 F16_D4S4 workspace, and output pointers."
        )
        emit_pointer_kernarg_loads(asm, self.KERNARG)
        asm.inst(f"v_mov_b32 v{serial}, v0")
        asm.inst(f"v_and_b32 v{lane}, 15, v{serial}")
        asm.inst(f"v_lshrrev_b32 v{wave}, 5, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{wave_column_base}, 4, v{wave}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{wave_column_base}, v{temporary}, v{wave_column_base}")
        asm.inst(f"v_lshlrev_b32 v{output_column}, 4, v{wave}")
        asm.inst(f"v_add_nc_u32 v{output_column}, v{lane}, v{output_column}")
        asm.inst(
            f"v_mul_lo_u32 v{output_column}, {decoded_lds.weight_row_stride}, "
            f"v{output_column}"
        )
        asm.inst(
            f"v_add_nc_u32 v{output_column}, {decoded_lds.weight_data_base}, "
            f"v{output_column}"
        )
        asm.inst(f"v_lshlrev_b32 v{temporary}, 7, s3")
        asm.inst(
            f"v_mul_lo_u32 v{activation_plane_address}, "
            f"{Q8_1_F16_D4S4_BLOCK_BYTES}, v{temporary}"
        )
        asm.inst(f"v_lshlrev_b32 v{temporary}, 2, v{serial}")
        asm.inst(
            f"v_add_nc_u32 v{activation_plane_address}, v{temporary}, "
            f"v{activation_plane_address}"
        )
        if (
            self.state.kernel_spec.instruction_policy.accumulator_initialization
            == "VopdPair"
        ):
            for register in range(self.C, sum_base + 64, 2):
                x_source = "0" if register < sum_base else "v0"
                y_source = "0" if register < sum_base else "v1"
                asm.inst(
                    f"v_dual_mov_b32 v{register}, {x_source} :: "
                    f"v_dual_mov_b32 v{register + 1}, {y_source}"
                )
        else:
            for register in range(self.C, self.C + 8):
                asm.inst(f"v_mov_b32 v{register}, 0")
            for register in range(sum_base, sum_base + 64):
                asm.inst(f"v_mov_b32 v{register}, v0")
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")
        asm.inst(f"s_mov_b32 s{self.SCALAR_TEMPORARY}, 0")
        asm.inst("s_mov_b32 s15, 512")

        asm.label(f".LForward{quant_label}HipStagedBlockLoop")
        self._emit_decoded_weight_lds_stage(
            asm,
            row_stride=row_stride,
            serial=serial,
            wave=wave,
            temporary=temporary,
            lds_address=lds_address,
            auxiliary=auxiliary,
            metadata_address=metadata_address,
            staging_base=c_base,
        )

        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{metadata_address}, 4, v{wave}")
        asm.inst(f"v_add_nc_u32 v{metadata_address}, v{temporary}, v{metadata_address}")
        asm.inst(
            f"v_mul_lo_u32 v{metadata_address}, {decoded_lds.metadata_row_stride}, "
            f"v{metadata_address}"
        )
        asm.inst(
            f"v_add_nc_u32 v{metadata_address}, {decoded_lds.weight_metadata_base}, "
            f"v{metadata_address}"
        )

        self._emit_activation_lds_stage(
            asm,
            activation_plane_address=activation_plane_address,
            serial=serial,
            temporary=temporary,
            lds_address=lds_address,
            activation_plane_stride=activation_plane_stride,
            activation_lane_stride=activation_lane_stride,
            staging_base=c_base,
            lds_base=activation_lds_base,
        )
        self._emit_decoded_group_loop(
            asm,
            group_base=0,
            weight_q=weight_q,
            metadata=metadata,
            c_base=c_base,
            low_activation_last=low_activation_last,
            high_activation_base=high_activation_base,
            activation_scale_sum_base=activation_scale_sum_base,
            scaled_dm_base=scaled_dm_base,
            sum_base=sum_base,
            lane=lane,
            lds_address=lds_address,
            weight_lds_base_address=output_column,
            metadata_lds_base_address=metadata_address,
        )
        asm.inst("s_barrier")

        self._emit_activation_lds_stage(
            asm,
            activation_plane_address=activation_plane_address,
            serial=serial,
            temporary=temporary,
            lds_address=lds_address,
            activation_plane_stride=activation_plane_stride,
            activation_lane_stride=activation_lane_stride,
            staging_base=c_base,
            lds_base=activation_lds_base,
        )
        self._emit_decoded_group_loop(
            asm,
            group_base=4,
            weight_q=weight_q,
            metadata=metadata,
            c_base=c_base,
            low_activation_last=low_activation_last,
            high_activation_base=high_activation_base,
            activation_scale_sum_base=activation_scale_sum_base,
            scaled_dm_base=scaled_dm_base,
            sum_base=sum_base,
            lane=lane,
            lds_address=lds_address,
            weight_lds_base_address=output_column,
            metadata_lds_base_address=metadata_address,
        )
        asm.inst("s_barrier")
        asm.inst(
            f"s_add_u32 s{self.SCALAR_TEMPORARY}, s{self.SCALAR_TEMPORARY}, "
            f"{self.state.contract.packed_weight_block_bytes}"
        )
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(
            f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {self.state.blocks_per_weight_row}"
        )
        asm.inst(f"s_cbranch_scc1 .LForward{quant_label}HipStagedBlockLoop")

        asm.comment("Store the 128x64 row-major BF16 output tile.")
        self._emit_decoded_bf16_store(
            asm,
            size_n=size.n,
            sum_base=sum_base,
            temporary=temporary,
            auxiliary=auxiliary,
            metadata_address=metadata_address,
            wave_column_base=wave_column_base,
            lane=lane,
            serial=serial,
        )
        emit_kernel_trailer(asm, name)
        return asm.text()

    def _emit_decoded_bf16_store(
        self,
        asm: Assembly,
        *,
        size_n: int,
        sum_base: int,
        temporary: int,
        auxiliary: int,
        metadata_address: int,
        wave_column_base: int,
        lane: int,
        serial: int,
    ) -> None:
        pipeline = self.state.kernel_spec.epilogue.pipeline
        if (
            pipeline is None
            or pipeline.tiles_ahead is None
            or pipeline.priority is None
        ):
            raise ForwardKernelWriterError(
                "decoded-weight LDS requires a complete epilogue pipeline"
            )
        scheduled = (
            pipeline.tiles_ahead != 8
            or pipeline.dependency_width != 1
            or pipeline.priority != 0
        )

        if not scheduled:
            for total in range(sum_base, sum_base + 64):
                emit_bf16_rne(asm, total, temporary)
            self._emit_decoded_store_address(
                asm,
                size_n=size_n,
                temporary=temporary,
                auxiliary=auxiliary,
                metadata_address=metadata_address,
                wave_column_base=wave_column_base,
                lane=lane,
                serial=serial,
            )
            for tile in range(8):
                if tile:
                    asm.inst(
                        f"v_add_nc_u32 v{metadata_address}, {32 * size_n}, "
                        f"v{metadata_address}"
                    )
                self._emit_decoded_store_tile(
                    asm,
                    tile=tile,
                    sum_base=sum_base,
                    metadata_address=metadata_address,
                )
            return

        if pipeline.priority:
            asm.inst(f"s_setprio {pipeline.priority}")
        self._emit_decoded_store_address(
            asm,
            size_n=size_n,
            temporary=temporary,
            auxiliary=auxiliary,
            metadata_address=metadata_address,
            wave_column_base=wave_column_base,
            lane=lane,
            serial=serial,
        )

        tiles_ahead = pipeline.tiles_ahead
        dependency_width = pipeline.dependency_width
        for first_tile in range(0, 8, tiles_ahead):
            tile_count = min(tiles_ahead, 8 - first_tile)
            first_element = 8 * first_tile
            element_count = 8 * tile_count
            for batch in range(
                first_element,
                first_element + element_count,
                dependency_width,
            ):
                batch_count = min(
                    dependency_width,
                    first_element + element_count - batch,
                )
                if batch_count == 1:
                    total = sum_base + batch
                    emit_bf16_rne(asm, total, temporary)
                    continue
                for item in range(batch_count):
                    total = sum_base + batch + item
                    asm.inst(f"v_bfe_u32 v{72 + item}, v{total}, 16, 1")
                for item in range(batch_count):
                    total = sum_base + batch + item
                    asm.inst(f"v_add3_u32 v{total}, v{72 + item}, v{total}, 0x7fff")
            for relative_tile in range(tile_count):
                tile = first_tile + relative_tile
                if tile:
                    asm.inst(
                        f"v_add_nc_u32 v{metadata_address}, {32 * size_n}, "
                        f"v{metadata_address}"
                    )
                self._emit_decoded_store_tile(
                    asm,
                    tile=tile,
                    sum_base=sum_base,
                    metadata_address=metadata_address,
                )

    def _emit_decoded_store_address(
        self,
        asm: Assembly,
        *,
        size_n: int,
        temporary: int,
        auxiliary: int,
        metadata_address: int,
        wave_column_base: int,
        lane: int,
        serial: int,
    ) -> None:
        asm.inst(f"v_lshlrev_b32 v{temporary}, 7, s3")
        asm.inst(f"v_add_nc_u32 v{auxiliary}, v{temporary}, v{lane}")
        asm.inst(f"v_mul_lo_u32 v{metadata_address}, {2 * size_n}, v{auxiliary}")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{wave_column_base}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{metadata_address}, v{metadata_address}, v{temporary}")

    def _emit_decoded_store_tile(
        self,
        asm: Assembly,
        *,
        tile: int,
        sum_base: int,
        metadata_address: int,
    ) -> None:
        asm.inst("s_clause 7")
        for element in range(8):
            total = sum_base + tile * 8 + element
            offset = f" offset:{4 * element}" if element else ""
            asm.inst(
                f"global_store_d16_hi_b16 v{metadata_address}, v{total}, "
                f"s[{self.KERNARG + 4}:{self.KERNARG + 5}]{offset}"
            )

    def _emit_activation_lds_stage(
        self,
        asm: Assembly,
        *,
        activation_plane_address: int,
        serial: int,
        temporary: int,
        lds_address: int,
        activation_plane_stride: int,
        activation_lane_stride: int,
        staging_base: int,
        lds_base: int,
    ) -> None:
        asm.comment("Cooperatively stage one contiguous 128-row Q8_1 F16_D4S4 plane.")
        for chunk in range(5):
            chunk_start = 8 * chunk
            chunk_count = min(8, 36 - chunk_start)
            asm.inst(
                f"v_add_nc_u32 v{temporary}, {4096 * chunk}, "
                f"v{activation_plane_address}"
            )
            for item in range(chunk_count):
                register = staging_base + chunk_start + item
                asm.inst(
                    f"global_load_b32 v{register}, v{temporary}, "
                    f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                    f"offset:{activation_lane_stride * item}"
                )
        asm.inst(f"v_lshlrev_b32 v{lds_address}, 2, v{serial}")
        asm.inst(f"v_add_nc_u32 v{lds_address}, {lds_base}, v{lds_address}")
        for item in range(0, 36, 2):
            asm.inst(f"s_waitcnt vmcnt({34 - item})")
            asm.inst(
                f"ds_write2st64_b32 v{lds_address}, v{staging_base + item}, "
                f"v{staging_base + item + 1} offset0:{2 * item} "
                f"offset1:{2 * (item + 1)}"
            )
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        asm.inst(
            f"v_add_nc_u32 v{activation_plane_address}, "
            f"{activation_plane_stride}, v{activation_plane_address}"
        )

    def _emit_decoded_group_loop(
        self,
        asm: Assembly,
        *,
        group_base: int,
        weight_q: int,
        metadata: int,
        c_base: int,
        low_activation_last: int,
        high_activation_base: int,
        activation_scale_sum_base: int,
        scaled_dm_base: int,
        sum_base: int,
        lane: int,
        lds_address: int,
        weight_lds_base_address: int,
        metadata_lds_base_address: int,
    ) -> None:
        quant_type = self.state.contract.quant_type
        quant_label = quant_type.replace("_", "")
        label = f".LForward{quant_label}DecodedGroupLoop{group_base}"
        asm.comment(
            f"Roll decoded {quant_type} groups {group_base} through {group_base + 3}."
        )
        asm.inst("s_mov_b32 s13, 0")
        asm.label(label)

        asm.inst("s_lshl_b32 s14, s13, 5")
        if group_base:
            asm.inst(
                f"v_add_nc_u32 v{lds_address}, {32 * group_base}, "
                f"v{weight_lds_base_address}"
            )
            asm.inst(f"v_add_nc_u32 v{lds_address}, s14, v{lds_address}")
        else:
            asm.inst(f"v_add_nc_u32 v{lds_address}, s14, v{weight_lds_base_address}")
        asm.inst(f"ds_read_b128 v[{weight_q}:{weight_q + 3}], v{lds_address}")
        asm.inst(
            f"ds_read_b128 v[{weight_q + 4}:{weight_q + 7}], v{lds_address} offset:16"
        )

        asm.inst(
            f"v_mad_u32_u24 v{metadata}, {Q8_1_F16_D4S4_BLOCK_BYTES}, v{lane}, s15"
        )
        asm.inst(f"v_add_nc_u32 v{lds_address}, s14, v{metadata}")
        for tile in range(8):
            low_activation = (
                c_base + 8 * (tile + 1) if tile < 7 else low_activation_last
            )
            high_activation = high_activation_base + 4 * tile
            tile_offset = 16 * tile * Q8_1_F16_D4S4_BLOCK_BYTES
            asm.inst(
                f"ds_read_b128 v[{low_activation}:{low_activation + 3}], "
                f"v{lds_address} offset:{tile_offset + 16}"
            )
            asm.inst(
                f"ds_read_b128 v[{high_activation}:{high_activation + 3}], "
                f"v{lds_address} offset:{tile_offset + 32}"
            )

        def emit_metadata_reads() -> None:
            asm.inst("s_lshl_b32 s14, s13, 2")
            asm.inst(f"v_add_nc_u32 v{metadata}, s14, v{metadata}")
            for tile in range(0, 8, 2):
                asm.inst(
                    f"ds_read2st64_b32 v[{activation_scale_sum_base + tile}:"
                    f"{activation_scale_sum_base + tile + 1}], v{metadata} "
                    f"offset0:{9 * tile} offset1:{9 * (tile + 1)}"
                )

            if group_base:
                asm.inst(
                    f"v_add_nc_u32 v{metadata}, {4 * group_base}, "
                    f"v{metadata_lds_base_address}"
                )
                asm.inst(f"v_add_nc_u32 v{metadata}, s14, v{metadata}")
            else:
                asm.inst(f"v_add_nc_u32 v{metadata}, s14, v{metadata_lds_base_address}")
            for pair in range(1, 4):
                asm.inst(
                    f"v_add_nc_u32 v{metadata + pair}, "
                    f"{metadata_group_stride * pair}, v{metadata}"
                )
            for element in range(0, 8, 2):
                asm.inst(
                    f"ds_read2_b32 v[{scaled_dm_base + element}:"
                    f"{scaled_dm_base + element + 1}], "
                    f"v{metadata + element // 2} offset0:0 offset1:152"
                )

        decoded_lds = self.state.decoded_lds
        if decoded_lds is None:
            raise ForwardKernelWriterError("decoded LDS layout is unavailable")
        metadata_group_stride = decoded_lds.metadata_group_stride
        metadata_schedule = self.state.kernel_spec.decode.metadata_schedule
        deferred_metadata = metadata_schedule in (
            "MetadataAfterLowWmma",
            "IndependentExtractionMetadataAfterLowWmma",
        )
        if not deferred_metadata:
            emit_metadata_reads()
        self._emit_decoded_group_accumulate(
            asm,
            weight_q=weight_q,
            c_base=c_base,
            low_activation_last=low_activation_last,
            high_activation_base=high_activation_base,
            activation_scale_sum_base=activation_scale_sum_base,
            scaled_dm_base=scaled_dm_base,
            sum_base=sum_base,
            overlap_lds=True,
            deferred_metadata_emitter=(
                emit_metadata_reads if deferred_metadata else None
            ),
        )
        asm.inst("s_add_u32 s13, s13, 1")
        asm.inst("s_cmp_lt_u32 s13, 4")
        asm.inst(f"s_cbranch_scc1 {label}")

    def _emit_decoded_group_accumulate(
        self,
        asm: Assembly,
        *,
        weight_q: int,
        c_base: int,
        low_activation_last: int,
        high_activation_base: int,
        activation_scale_sum_base: int,
        scaled_dm_base: int,
        sum_base: int,
        overlap_lds: bool,
        deferred_metadata_emitter: Callable[[], None] | None = None,
    ) -> None:
        clamp = " clamp" if self.state.contract.wmma_clamp else ""
        for tile in range(8):
            if overlap_lds:
                first_wait = 15 if deferred_metadata_emitter is not None else 23
                asm.inst(f"s_waitcnt lgkmcnt({first_wait - 2 * tile})")
            c_fragment = c_base + 8 * tile
            low_activation = (
                c_base + 8 * (tile + 1) if tile < 7 else low_activation_last
            )
            asm.inst(
                f"v_wmma_i32_16x16x16_iu8 v[{c_fragment}:{c_fragment + 7}], "
                f"v[{weight_q}:{weight_q + 3}], "
                f"v[{low_activation}:{low_activation + 3}], "
                f"v[{self.C}:{self.C + 7}] neg_lo:[1,1,0]{clamp}"
            )
        if deferred_metadata_emitter is not None:
            deferred_metadata_emitter()
        for tile in range(8):
            if overlap_lds and tile == 7:
                asm.inst("s_waitcnt lgkmcnt(8)")
            c_fragment = c_base + 8 * tile
            high_activation = high_activation_base + 4 * tile
            asm.inst(
                f"v_wmma_i32_16x16x16_iu8 v[{c_fragment}:{c_fragment + 7}], "
                f"v[{weight_q + 4}:{weight_q + 7}], "
                f"v[{high_activation}:{high_activation + 3}], "
                f"v[{c_fragment}:{c_fragment + 7}] neg_lo:[1,1,0]{clamp}"
            )
        if overlap_lds:
            asm.inst("s_waitcnt lgkmcnt(0)")

        product_base = low_activation_last
        for tile_start in (0, 4):
            for local_tile in range(4):
                tile = tile_start + local_tile
                tile_scale_sum = activation_scale_sum_base + tile
                for element in range(8):
                    product = product_base + 8 * local_tile + element
                    asm.inst(
                        f"v_fma_mix_f32 v{product}, "
                        f"v{scaled_dm_base + element}, v{tile_scale_sum}, 0 "
                        f"op_sel_hi:[1,1,0]"
                    )
            for local_tile in range(4):
                tile = tile_start + local_tile
                c_fragment = c_base + 8 * tile
                for element in range(8):
                    asm.inst(
                        f"v_cvt_f32_i32 v{c_fragment + element}, "
                        f"v{c_fragment + element}"
                    )
            for local_tile in range(4):
                tile = tile_start + local_tile
                c_fragment = c_base + 8 * tile
                for element in range(0, 8, 2):
                    total = sum_base + tile * 8 + element
                    product = product_base + 8 * local_tile + element
                    asm.inst(
                        f"v_dual_fmac_f32 v{total}, v{product}, "
                        f"v{c_fragment + element} :: "
                        f"v_dual_fmac_f32 v{total + 1}, v{product + 1}, "
                        f"v{c_fragment + element + 1}"
                    )
            for local_tile in range(4):
                tile = tile_start + local_tile
                tile_scale_sum = activation_scale_sum_base + tile
                for element in range(8):
                    total = sum_base + tile * 8 + element
                    asm.inst(
                        f"v_fma_mix_f32 v{total}, "
                        f"v{scaled_dm_base + element}, v{tile_scale_sum}, v{total} "
                        f"op_sel:[1,1,0] op_sel_hi:[1,1,0]"
                    )

    def _emit_group(self, asm: Assembly, group: int) -> None:
        weight_q_offset, weight_q_offset_high = (
            self.state.semantics.low_payload_group_offsets(group)
        )
        activation_q_offset = 16 + 32 * (group % 4)
        activation_scale_sum_offset = 4 * (group % 4)
        activation_address = (
            self.ACTIVATION_ADDRESS_0 if group < 4 else self.ACTIVATION_ADDRESS_1
        )

        asm.comment(
            f"Q4_K group {group}: direct nibbles and fixed Q8_1 F16_D4S4 block."
        )
        asm.inst(
            f"global_load_b128 v[{self.WEIGHT_Q}:{self.WEIGHT_Q + 3}], "
            f"v{self.WEIGHT_Q_ADDRESS}, s[{self.KERNARG}:{self.KERNARG + 1}] "
            f"offset:{weight_q_offset}"
        )
        asm.inst(
            f"global_load_b128 v[{self.WEIGHT_Q + 4}:{self.WEIGHT_Q + 7}], "
            f"v{self.WEIGHT_Q_ADDRESS}, s[{self.KERNARG}:{self.KERNARG + 1}] "
            f"offset:{weight_q_offset_high}"
        )
        asm.inst(
            f"global_load_b128 v[{self.ACTIVATION_Q}:{self.ACTIVATION_Q + 3}], "
            f"v{activation_address}, s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
            f"offset:{activation_q_offset}"
        )
        asm.inst(
            f"global_load_b128 v[{self.ACTIVATION_Q + 4}:{self.ACTIVATION_Q + 7}], "
            f"v{activation_address}, s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
            f"offset:{activation_q_offset + 16}"
        )
        asm.inst(
            f"global_load_b32 v{self.ACTIVATION_SCALE_SUM}, v{activation_address}, "
            f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
            f"offset:{activation_scale_sum_offset}"
        )
        asm.inst("s_waitcnt vmcnt(0)")

        for register in range(self.WEIGHT_Q, self.WEIGHT_Q + 8):
            if group & 1:
                asm.inst(f"v_lshrrev_b32 v{register}, 4, v{register}")
            asm.inst(f"v_and_b32 v{register}, 0x0f0f0f0f, v{register}")
        for register in range(self.C, self.C + 8):
            asm.inst(f"v_mov_b32 v{register}, 0")
        clamp = " clamp" if self.state.contract.wmma_clamp else ""
        asm.inst(
            f"v_wmma_i32_16x16x16_iu8 v[{self.C}:{self.C + 7}], "
            f"v[{self.WEIGHT_Q}:{self.WEIGHT_Q + 3}], "
            f"v[{self.ACTIVATION_Q}:{self.ACTIVATION_Q + 3}], "
            f"v[{self.C}:{self.C + 7}] neg_lo:[1,1,0]{clamp}"
        )
        asm.inst(
            f"v_wmma_i32_16x16x16_iu8 v[{self.C}:{self.C + 7}], "
            f"v[{self.WEIGHT_Q + 4}:{self.WEIGHT_Q + 7}], "
            f"v[{self.ACTIVATION_Q + 4}:{self.ACTIVATION_Q + 7}], "
            f"v[{self.C}:{self.C + 7}] neg_lo:[1,1,0]{clamp}"
        )
        self._emit_scaled_accumulate(asm, group)

    def _emit_scale_and_minimum(
        self,
        asm: Assembly,
        group: int,
        metadata: int,
        *,
        scale: int | None = None,
        minimum: int | None = None,
        temporary: int | None = None,
    ) -> None:
        scale = self.SCALE if scale is None else scale
        minimum = self.MINIMUM if minimum is None else minimum
        temporary = self.TEMPORARY if temporary is None else temporary
        fields = self.state.semantics.packed_scale_minimum_fields(group)
        for destination, parts in ((scale, fields.scale), (minimum, fields.minimum)):
            for index, part in enumerate(parts):
                target = destination if index == 0 else temporary
                asm.inst(
                    f"v_bfe_u32 v{target}, v{metadata + part.metadata_word}, "
                    f"{part.bit_offset}, {part.bit_count}"
                )
                if index:
                    asm.inst(
                        f"v_lshl_or_b32 v{destination}, v{temporary}, "
                        f"{part.destination_shift}, v{destination}"
                    )

    def _emit_scaled_accumulate(self, asm: Assembly, group: int) -> None:
        asm.comment("Reproduce Q4_K FP16 scale/min construction before FP32 sums.")
        asm.inst(f"v_cvt_f32_f16 v{self.ACTIVATION_D}, v{self.ACTIVATION_SCALE_SUM}")
        asm.inst(
            f"v_cvt_f32_f16 v{self.ACTIVATION_SUM}, v{self.ACTIVATION_SCALE_SUM}.h"
        )
        for element in range(8):
            metadata = self.WEIGHT_METADATA + 4 * element
            self._emit_scale_and_minimum(asm, group, metadata)
            asm.inst(f"v_cvt_f32_u32 v{self.WEIGHT_D}, v{self.SCALE}")
            asm.inst(f"v_cvt_f32_u32 v{self.WEIGHT_MIN}, v{self.MINIMUM}")
            asm.inst(f"v_cvt_f16_f32_e32 v{self.SCALED_DM}.l, v{self.WEIGHT_D}")
            asm.inst(f"v_cvt_f16_f32_e32 v{self.SCALED_DM}.h, v{self.WEIGHT_MIN}")
            asm.inst(f"v_pk_mul_f16 v{self.SCALED_DM}, 0xbc003c00, v{self.SCALED_DM}")
            asm.inst(f"v_pk_mul_f16 v{self.SCALED_DM}, v{metadata}, v{self.SCALED_DM}")
            asm.inst(f"v_cvt_f32_f16 v{self.WEIGHT_D}, v{self.SCALED_DM}")
            asm.inst(f"v_cvt_f32_f16 v{self.WEIGHT_MIN}, v{self.SCALED_DM}.h")
            c = self.C + element
            total = self.SUM + element
            asm.inst(f"v_cvt_f32_i32 v{self.TEMPORARY}, v{c}")
            asm.inst(
                f"v_mul_f32 v{self.TEMPORARY}, v{self.WEIGHT_D}, v{self.TEMPORARY}"
            )
            asm.inst(
                f"v_fma_f32 v{total}, v{self.TEMPORARY}, v{self.ACTIVATION_D}, v{total}"
            )
            asm.inst(
                f"v_fma_f32 v{total}, v{self.WEIGHT_MIN}, "
                f"v{self.ACTIVATION_SUM}, v{total}"
            )

    def _emit_store(self, asm: Assembly) -> None:
        size = self.state.problem_size
        asm.comment("Store the gfx11 J-major C fragments as row-major BF16 output.")
        asm.inst(f"v_and_b32 v{self.TEMPORARY}, 15, v{self.SERIAL}")
        asm.inst(f"v_lshlrev_b32 v{self.OUTPUT_ADDRESS}, 4, s3")
        asm.inst(
            f"v_add_nc_u32 v{self.OUTPUT_ADDRESS}, v{self.OUTPUT_ADDRESS}, "
            f"v{self.TEMPORARY}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{self.OUTPUT_ADDRESS}, {2 * size.n}, v{self.OUTPUT_ADDRESS}"
        )
        asm.inst(f"v_lshrrev_b32 v{self.TEMPORARY}, 4, v{self.SERIAL}")
        asm.inst(f"v_and_b32 v{self.TEMPORARY}, 1, v{self.TEMPORARY}")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY + 1}, 4, s2")
        asm.inst(
            f"v_add_nc_u32 v{self.TEMPORARY}, v{self.TEMPORARY + 1}, v{self.TEMPORARY}"
        )
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY}, 1, v{self.TEMPORARY}")
        asm.inst(
            f"v_add_nc_u32 v{self.OUTPUT_ADDRESS}, v{self.OUTPUT_ADDRESS}, "
            f"v{self.TEMPORARY}"
        )
        for element in range(8):
            total = self.SUM + element
            emit_bf16_rne(asm, total, self.TEMPORARY)
            asm.inst(
                f"global_store_d16_hi_b16 v{self.OUTPUT_ADDRESS}, v{total}, "
                f"s[{self.KERNARG + 4}:{self.KERNARG + 5}]"
            )
            if element != 7:
                asm.inst(
                    f"v_add_nc_u32 v{self.OUTPUT_ADDRESS}, 4, v{self.OUTPUT_ADDRESS}"
                )
