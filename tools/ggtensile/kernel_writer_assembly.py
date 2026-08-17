import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import rocisa  # ty: ignore[unresolved-import]
from rocisa import code  # ty: ignore[unresolved-import]
from rocisa.enum import SignatureValueKind as SVK  # ty: ignore[unresolved-import]

from .kernel_abi import KernelAbi, KernelArgumentKind


@dataclass(frozen=True)
class RegisterLifetime:
    first_stage: int
    last_stage: int

    def __post_init__(self) -> None:
        if self.first_stage < 0 or self.last_stage < self.first_stage:
            raise ValueError("invalid register lifetime")

    def overlaps(self, other: "RegisterLifetime") -> bool:
        return not (
            self.last_stage < other.first_stage or other.last_stage < self.first_stage
        )


@dataclass(frozen=True)
class RegisterRole:
    name: str
    width: int
    lifetime: RegisterLifetime
    alignment: int = 1
    minimum_register: int = 0

    def __post_init__(self) -> None:
        if not self.name or self.width <= 0:
            raise ValueError("register role requires a name and positive width")
        if self.alignment <= 0 or self.minimum_register < 0:
            raise ValueError("invalid register role alignment or minimum")


@dataclass(frozen=True)
class RegisterAssignment:
    role: RegisterRole
    first_register: int

    @property
    def registers(self) -> range:
        return range(self.first_register, self.first_register + self.role.width)


@dataclass(frozen=True)
class DeterministicRegisterPlan:
    """First-fit allocation over an explicit semantic role order."""

    assignments: tuple[RegisterAssignment, ...]
    register_count: int

    @classmethod
    def allocate(
        cls,
        roles: Mapping[str, RegisterRole],
        role_order: tuple[str, ...],
        *,
        max_registers: int,
        reserved: tuple[RegisterAssignment, ...] = (),
    ) -> "DeterministicRegisterPlan":
        if max_registers <= 0:
            raise ValueError("max_registers must be positive")
        if len(role_order) != len(set(role_order)) or set(role_order) != set(roles):
            raise ValueError("role_order must name every register role exactly once")
        assigned = list(reserved)
        for name in role_order:
            role = roles[name]
            if role.name != name:
                raise ValueError("register role mapping key does not match role name")
            candidate = _align_register(role.minimum_register, role.alignment)
            while candidate + role.width <= max_registers:
                proposed = range(candidate, candidate + role.width)
                conflict = any(
                    role.lifetime.overlaps(item.role.lifetime)
                    and _register_ranges_overlap(proposed, item.registers)
                    for item in assigned
                )
                if not conflict:
                    assigned.append(RegisterAssignment(role, candidate))
                    break
                candidate = _align_register(candidate + 1, role.alignment)
            else:
                raise ValueError(f"register role {name!r} exceeds the register limit")
        register_count = max(
            (assignment.registers.stop for assignment in assigned),
            default=0,
        )
        return cls(tuple(assigned), register_count)

    def assignment(self, role_name: str) -> RegisterAssignment:
        for assignment in self.assignments:
            if assignment.role.name == role_name:
                return assignment
        raise KeyError(role_name)


class DeterministicRegisterPool:
    """Explicit first-fit checkout/checkin pool over a fixed register set."""

    def __init__(self, registers: tuple[int, ...]) -> None:
        if not registers or any(register < 0 for register in registers):
            raise ValueError("register pool requires nonnegative registers")
        if len(registers) != len(set(registers)):
            raise ValueError("register pool entries must be unique")
        self._registers = tuple(sorted(registers))
        self._assignments: dict[str, RegisterAssignment] = {}
        self._owners: dict[int, str] = {}

    def checkout(
        self,
        role: RegisterRole,
        *,
        preferred_register: int | None = None,
    ) -> RegisterAssignment:
        if role.name in self._assignments:
            raise ValueError(f"register role {role.name!r} is already checked out")
        candidates = self._registers
        if preferred_register is not None:
            candidates = (preferred_register,)
        for first in candidates:
            if first % role.alignment or first < role.minimum_register:
                continue
            registers = range(first, first + role.width)
            if all(
                register in self._registers and register not in self._owners
                for register in registers
            ):
                assignment = RegisterAssignment(role, first)
                self._assignments[role.name] = assignment
                for register in registers:
                    self._owners[register] = role.name
                return assignment
        preference = (
            f" at v{preferred_register}" if preferred_register is not None else ""
        )
        raise ValueError(
            f"register role {role.name!r} cannot be checked out{preference}"
        )

    def checkin(self, role_name: str) -> RegisterAssignment:
        if role_name not in self._assignments:
            raise ValueError(f"register role {role_name!r} is not checked out")
        assignment = self._assignments.pop(role_name)
        for register in assignment.registers:
            if self._owners.pop(register, None) != role_name:
                raise ValueError(f"register pool ownership mismatch for {role_name!r}")
        return assignment

    def assignment(self, role_name: str) -> RegisterAssignment:
        return self._assignments[role_name]

    @property
    def checked_out(self) -> tuple[RegisterAssignment, ...]:
        return tuple(self._assignments[name] for name in sorted(self._assignments))


def _align_register(register: int, alignment: int) -> int:
    return (register + alignment - 1) // alignment * alignment


def _register_ranges_overlap(left: range, right: range) -> bool:
    return left.start < right.stop and right.start < left.stop


class Assembly:
    def __init__(self, *, indent: str = "  ") -> None:
        self.indent = indent
        self.lines: list[str] = []

    def line(self, text: str) -> None:
        self.lines.append(text)

    def comment(self, text: str) -> None:
        self.lines.append(f"// {text}")

    def label(self, name: str) -> None:
        self.lines.append(f"{name}:")

    def inst(self, text: str, comment: str = "") -> None:
        suffix = f" // {comment}" if comment else ""
        self.lines.append(f"{self.indent}{text}{suffix}")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


class _Signature(Protocol):
    def addArg(
        self,
        name: str,
        value_kind: Any,
        value_type: str,
        address_space: str | None = None,
    ) -> None: ...


def add_kernel_abi_arguments(signature: _Signature, abi: KernelAbi) -> None:
    for argument in abi.arguments:
        if argument.kind is KernelArgumentKind.GlobalBuffer:
            signature.addArg(
                argument.name,
                SVK.SIG_GLOBALBUFFER,
                argument.value_type.value,
                "generic",
            )
        else:
            signature.addArg(
                argument.name,
                SVK.SIG_VALUE,
                argument.value_type.value,
            )


@dataclass(frozen=True)
class KernelEnvelope:
    module_name: str
    kernel_name: str
    isa: tuple[int, int, int]
    wavefront_size: int
    assembler: Path
    temporary_prefix: str
    code_object_version: int
    group_segment_size: int
    sgpr_work_group: tuple[int, int, int]
    vgpr_work_item: int
    flat_workgroup_size: int
    total_vgprs: int
    total_sgprs: int
    abi: KernelAbi
    description: str

    def initialize(self) -> None:
        initialize_rocisa(
            self.isa,
            self.wavefront_size,
            self.assembler,
            temporary_prefix=self.temporary_prefix,
        )

    def render(
        self,
        body: str,
        *,
        trailing_sections: tuple[str, ...] = (),
    ) -> str:
        signature = code.SignatureBase(
            kernelName=self.kernel_name,
            kernArgsVersion=0,
            codeObjectVersion=str(self.code_object_version),
            groupSegmentSize=self.group_segment_size,
            sgprWorkGroup=self.sgpr_work_group,
            vgprWorkItem=self.vgpr_work_item,
            flatWorkGroupSize=self.flat_workgroup_size,
            totalVgprs=self.total_vgprs,
            totalAgprs=0,
            totalSgprs=self.total_sgprs,
        )
        signature.addDescriptionTopic(self.description)
        add_kernel_abi_arguments(signature, self.abi)
        module = code.Module(self.module_name)
        module.add(signature)
        module.add(code.TextBlock(body))
        source = str(module)
        for section in trailing_sections:
            source += "\n" + section
        return source


def initialize_rocisa(
    isa: tuple[int, int, int],
    wavefront_size: int,
    assembler: Path,
    *,
    temporary_prefix: str,
) -> None:
    global_isa = rocisa.rocIsa.getInstance()
    original_directory = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=temporary_prefix) as temporary:
        os.chdir(temporary)
        try:
            global_isa.init(isa, str(assembler), False)
        finally:
            os.chdir(original_directory)
    global_isa.setKernel(isa, wavefront_size)


def write_assembly_source(output: Path, source: str) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(source, encoding="utf-8")
    temporary.replace(output)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def emit_pointer_kernarg_loads(
    assembly: Assembly,
    kernarg: int,
    abi: KernelAbi,
) -> None:
    pointers = tuple(
        item
        for item in abi.layout
        if item.argument.kind is KernelArgumentKind.GlobalBuffer
    )
    if len(pointers) < 3:
        raise ValueError("kernel ABI requires three leading pointer arguments")
    for index, pointer in enumerate(pointers[:3]):
        first = kernarg + 2 * index
        assembly.inst(
            f"s_load_dwordx2 s[{first}:{first + 1}], s[0:1], 0x{pointer.offset:x}"
        )
    assembly.inst("s_waitcnt lgkmcnt(0)")


def emit_kernel_trailer(assembly: Assembly, kernel_name: str) -> None:
    assembly.inst("s_endpgm")
    assembly.lines.append(f".L{kernel_name}_end:")
    assembly.lines.append(f".size {kernel_name}, .L{kernel_name}_end - {kernel_name}")


def emit_bf16_rne(
    assembly: Assembly,
    value_register: int,
    temporary_register: int,
) -> None:
    assembly.inst(f"v_bfe_u32 v{temporary_register}, v{value_register}, 16, 1")
    assembly.inst(
        f"v_add3_u32 v{value_register}, v{temporary_register}, "
        f"v{value_register}, 0x7fff"
    )


def emit_scale_u32(
    assembly: Assembly,
    destination: int,
    scale: int,
    source: int,
) -> None:
    if scale > 0 and scale & (scale - 1) == 0:
        shift = scale.bit_length() - 1
        assembly.inst(f"v_lshlrev_b32 v{destination}, {shift}, v{source}")
    else:
        assembly.inst(f"v_mul_lo_u32 v{destination}, {scale}, v{source}")


def emit_add_pointer(
    assembly: Assembly,
    destination: int,
    scalar_pointer: int,
    offset: int,
) -> None:
    assembly.inst(f"v_add_co_u32 v{destination}, vcc_lo, s{scalar_pointer}, v{offset}")
    assembly.inst(
        f"v_add_co_ci_u32_e64 v{destination + 1}, null, "
        f"s{scalar_pointer + 1}, 0, vcc_lo"
    )
