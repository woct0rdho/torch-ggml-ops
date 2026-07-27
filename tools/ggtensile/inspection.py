import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .model import SolutionKey
from .toolchain import Toolchain


class InspectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactInspection:
    kernel_name: str
    code_object_sha256: str
    normalized_assembly_sha256: str
    code_object_version: int
    target: str
    kernarg_segment_size: int
    wavefront_size: int
    max_flat_workgroup_size: int
    lds_num_bytes: int
    vgpr_count: int
    sgpr_count: int
    private_segment_bytes: int
    vgpr_spill_count: int
    sgpr_spill_count: int
    max_vgpr_index: int
    max_sgpr_index: int
    wmma_count: int
    barrier_count: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "KernelName": self.kernel_name,
            "CodeObjectSHA256": self.code_object_sha256,
            "NormalizedAssemblySHA256": self.normalized_assembly_sha256,
            "CodeObjectVersion": self.code_object_version,
            "Target": self.target,
            "KernargSegmentSize": self.kernarg_segment_size,
            "WavefrontSize": self.wavefront_size,
            "MaxFlatWorkgroupSize": self.max_flat_workgroup_size,
            "LdsNumBytes": self.lds_num_bytes,
            "NumVgpr": self.vgpr_count,
            "NumSgpr": self.sgpr_count,
            "PrivateSegmentBytes": self.private_segment_bytes,
            "VgprSpillCount": self.vgpr_spill_count,
            "SgprSpillCount": self.sgpr_spill_count,
            "MaxVgprIndex": self.max_vgpr_index,
            "MaxSgprIndex": self.max_sgpr_index,
            "StaticWmmaCount": self.wmma_count,
            "StaticBarrierCount": self.barrier_count,
        }


_EXPECTED_ARGS = (
    ("grad_output", 0, 8, "global_buffer", "bf16"),
    ("packed_weight", 8, 8, "global_buffer", "struct"),
    ("grad_input", 16, 8, "global_buffer", "bf16"),
    ("rows", 24, 4, "by_value", "u32"),
    ("out_features", 28, 4, "by_value", "u32"),
    ("in_features", 32, 4, "by_value", "u32"),
    ("blocks_per_weight_row", 36, 4, "by_value", "u32"),
)


def inspect_artifact(
    solution_key: SolutionKey,
    code_object: Path,
    toolchain: Toolchain,
) -> ArtifactInspection:
    if not code_object.is_file():
        raise InspectionError(f"code object does not exist: {code_object}")
    readelf = toolchain.readelf_output(code_object)
    disassembly = toolchain.disassembly_output(code_object)
    metadata = _metadata(readelf)
    kernel = _kernel_metadata(metadata, solution_key.kernel_name)
    instructions = _instructions(disassembly)
    errors: list[str] = []

    _require(_elf_value(readelf, "ABI Version") == "3", "code object is not v5", errors)
    _require(
        _elf_value(readelf, "Flags").endswith("gfx1151"),
        "ELF target is not gfx1151",
        errors,
    )
    functions = _global_function_symbols(readelf)
    _require(
        functions == {solution_key.kernel_name},
        f"global kernel symbols are {sorted(functions)!r}",
        errors,
    )
    _validate_metadata(kernel, solution_key, errors)

    mnemonics = tuple(instruction.split(None, 1)[0] for instruction in instructions)
    wmma_count = mnemonics.count("v_wmma_f32_16x16x16_bf16")
    barrier_count = mnemonics.count("s_barrier")
    _require(wmma_count == 32, f"expected 32 static WMMAs, found {wmma_count}", errors)
    _require(
        barrier_count == 2, f"expected two barriers, found {barrier_count}", errors
    )
    _require(
        not any(mnemonic.startswith("scratch_") for mnemonic in mnemonics),
        "scratch instruction found",
        errors,
    )
    call_mnemonics = {
        mnemonic
        for mnemonic in mnemonics
        if "call" in mnemonic
        or mnemonic in {"s_getpc_b64", "s_setpc_b64", "s_swappc_b64"}
    }
    _require(
        not call_mnemonics, f"call instruction found: {sorted(call_mnemonics)}", errors
    )

    max_vgpr = _max_register_index(disassembly, "v")
    max_sgpr = _max_register_index(disassembly, "s")
    vgpr_count = _integer(kernel, ".vgpr_count")
    sgpr_count = _integer(kernel, ".sgpr_count")
    _require(max_vgpr < vgpr_count, "VGPR index exceeds metadata declaration", errors)
    _require(max_sgpr < sgpr_count, "SGPR index exceeds metadata declaration", errors)

    if errors:
        raise InspectionError("artifact rejected: " + "; ".join(errors))

    normalized = "\n".join(instructions) + "\n"
    return ArtifactInspection(
        kernel_name=solution_key.kernel_name,
        code_object_sha256=_sha256_file(code_object),
        normalized_assembly_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
        code_object_version=5,
        target="gfx1151",
        kernarg_segment_size=_integer(kernel, ".kernarg_segment_size"),
        wavefront_size=_integer(kernel, ".wavefront_size"),
        max_flat_workgroup_size=_integer(kernel, ".max_flat_workgroup_size"),
        lds_num_bytes=_integer(kernel, ".group_segment_fixed_size"),
        vgpr_count=vgpr_count,
        sgpr_count=sgpr_count,
        private_segment_bytes=_integer(kernel, ".private_segment_fixed_size"),
        vgpr_spill_count=_integer(kernel, ".vgpr_spill_count"),
        sgpr_spill_count=_integer(kernel, ".sgpr_spill_count"),
        max_vgpr_index=max_vgpr,
        max_sgpr_index=max_sgpr,
        wmma_count=wmma_count,
        barrier_count=barrier_count,
    )


def _metadata(readelf: str) -> Mapping[str, Any]:
    marker = "AMDGPU Metadata:\n        ---\n"
    try:
        start = readelf.index(marker) + len("AMDGPU Metadata:\n        ")
        end = readelf.index("\n...\n", start) + len("\n...")
    except ValueError as error:
        raise InspectionError("cannot find AMDGPU metadata note") from error
    try:
        import yaml
    except ImportError as error:
        raise InspectionError("PyYAML is required for artifact inspection") from error
    parsed = yaml.safe_load(readelf[start:end])
    if not isinstance(parsed, Mapping):
        raise InspectionError("AMDGPU metadata is not a mapping")
    return parsed


def _kernel_metadata(
    metadata: Mapping[str, Any], kernel_name: str
) -> Mapping[str, Any]:
    kernels = metadata.get("amdhsa.kernels")
    if not isinstance(kernels, list) or len(kernels) != 1:
        raise InspectionError("metadata must contain exactly one kernel")
    kernel = kernels[0]
    if not isinstance(kernel, Mapping) or kernel.get(".name") != kernel_name:
        raise InspectionError("metadata kernel name does not match SolutionKey")
    return kernel


def _validate_metadata(
    kernel: Mapping[str, Any],
    solution_key: SolutionKey,
    errors: list[str],
) -> None:
    solution = solution_key.solution
    m_tiles = solution.matrix_instruction[5]
    decoder_rows = solution.matrix_instruction[6] // 4
    swizzle_vgprs = (
        32 // solution.lds_swizzle_chunk_b if solution.lds_swizzle_chunk_b else 0
    )
    global_prefetch_vgprs = 8 * m_tiles * (solution.prefetch_global_read - 1)
    local_prefetch_vgprs = 16 * (solution.prefetch_local_read - 1)
    expected_vgprs = (
        164
        + 8 * m_tiles
        + 8 * decoder_rows
        + swizzle_vgprs
        + global_prefetch_vgprs
        + local_prefetch_vgprs
    )
    expected = {
        ".kernarg_segment_size": 40,
        ".kernarg_segment_align": 8,
        ".group_segment_fixed_size": solution.lds_num_bytes,
        ".private_segment_fixed_size": 0,
        ".max_flat_workgroup_size": solution.num_threads,
        ".wavefront_size": solution.wavefront_size,
        ".vgpr_count": expected_vgprs,
        ".sgpr_count": 20,
        ".vgpr_spill_count": 0,
        ".sgpr_spill_count": 0,
    }
    for field, value in expected.items():
        actual = kernel.get(field)
        _require(actual == value, f"{field} is {actual!r}, expected {value}", errors)
    _require(
        kernel.get(".uses_dynamic_stack", False) is False,
        "dynamic stack is enabled",
        errors,
    )
    arguments = kernel.get(".args")
    actual_args = ()
    if isinstance(arguments, list):
        actual_args = tuple(
            (
                argument.get(".name"),
                argument.get(".offset"),
                argument.get(".size"),
                argument.get(".value_kind"),
                argument.get(".value_type"),
            )
            for argument in arguments
            if isinstance(argument, Mapping)
        )
    _require(actual_args == _EXPECTED_ARGS, "kernarg ABI does not match", errors)


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if type(value) is not int:
        raise InspectionError(f"metadata field {key} is not an integer")
    return value


def _elf_value(readelf: str, field: str) -> str:
    match = re.search(rf"^\s*{re.escape(field)}:\s*(.+?)\s*$", readelf, re.MULTILINE)
    if match is None:
        raise InspectionError(f"cannot find ELF field {field}")
    return match.group(1)


def _global_function_symbols(readelf: str) -> set[str]:
    return set(
        re.findall(
            r"^\s*\d+:\s+[0-9a-f]+\s+\d+\s+FUNC\s+GLOBAL\s+PROTECTED\s+\d+\s+(\S+)\s*$",
            readelf,
            re.MULTILINE,
        )
    )


def _instructions(disassembly: str) -> tuple[str, ...]:
    result = []
    for line in disassembly.splitlines():
        if not line.startswith("\t") or "//" not in line:
            continue
        result.append(line.split("//", 1)[0].strip())
    if not result:
        raise InspectionError("artifact disassembly contains no instructions")
    return tuple(result)


def _max_register_index(disassembly: str, prefix: str) -> int:
    values: list[int] = []
    pattern = re.compile(rf"\b{prefix}(?:\[(\d+):(\d+)\]|(\d+))\b")
    for match in pattern.finditer(disassembly):
        values.extend(int(value) for value in match.groups() if value is not None)
    return max(values, default=-1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)
