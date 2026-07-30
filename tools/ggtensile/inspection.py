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
    valu_issue_count: int
    valu_operation_count: int
    vopd_count: int
    vmem_count: int
    lds_count: int
    wait_count: int
    clause_count: int
    delay_alu_count: int
    buffer_gl0_inv_count: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "KernelName": self.kernel_name,
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
            "StaticValuIssueCount": self.valu_issue_count,
            "StaticValuOperationCount": self.valu_operation_count,
            "StaticVopdCount": self.vopd_count,
            "StaticVmemCount": self.vmem_count,
            "StaticLdsCount": self.lds_count,
            "StaticWaitCount": self.wait_count,
            "StaticClauseCount": self.clause_count,
            "StaticDelayAluCount": self.delay_alu_count,
            "StaticBufferGl0InvCount": self.buffer_gl0_inv_count,
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
    *,
    expected_wmma_count: int | None = None,
    expected_barrier_count: int | None = None,
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
    vopd_count = sum(" :: " in instruction for instruction in instructions)
    valu_issue_count = sum(
        mnemonic.startswith("v_") and not mnemonic.startswith("v_wmma_")
        for mnemonic in mnemonics
    )
    valu_operation_count = valu_issue_count + vopd_count
    vmem_count = sum(
        mnemonic.startswith(("global_", "flat_", "buffer_", "scratch_"))
        and mnemonic != "buffer_gl0_inv"
        for mnemonic in mnemonics
    )
    lds_count = sum(mnemonic.startswith("ds_") for mnemonic in mnemonics)
    wait_count = sum(mnemonic.startswith("s_waitcnt") for mnemonic in mnemonics)
    clause_count = mnemonics.count("s_clause")
    delay_alu_count = mnemonics.count("s_delay_alu")
    buffer_gl0_inv_count = mnemonics.count("buffer_gl0_inv")
    solution = solution_key.solution
    expected_wmmas = expected_wmma_count
    if expected_wmmas is None:
        expected_wmmas = (
            solution.matrix_instruction[5]
            * solution.matrix_instruction[6]
            * solution.depth_u
            // 16
        )
        if solution.one_lds_buffer == 0:
            expected_wmmas *= 1 + int(solution_key.problem_size.k > solution.depth_u)
    _require(
        wmma_count == expected_wmmas,
        f"expected {expected_wmmas} static WMMAs, found {wmma_count}",
        errors,
    )
    expected_barriers = expected_barrier_count
    if expected_barriers is None:
        if solution.one_lds_buffer == 0:
            expected_barriers = 1 + int(solution_key.problem_size.k > solution.depth_u)
        else:
            expected_barriers = 3 if solution.prefetch_packed_weight_next else 2
    _require(
        barrier_count == expected_barriers,
        f"expected {expected_barriers} barriers, found {barrier_count}",
        errors,
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

    return ArtifactInspection(
        kernel_name=solution_key.kernel_name,
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
        valu_issue_count=valu_issue_count,
        valu_operation_count=valu_operation_count,
        vopd_count=vopd_count,
        vmem_count=vmem_count,
        lds_count=lds_count,
        wait_count=wait_count,
        clause_count=clause_count,
        delay_alu_count=delay_alu_count,
        buffer_gl0_inv_count=buffer_gl0_inv_count,
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
    decoder_threads = min(solution.num_threads, 128)
    decoder_rows = (
        solution.depth_u
        * solution.macro_tile1
        // (decoder_threads * solution.decoder_width)
    )
    n_tiles = solution.matrix_instruction[6]

    def allocate(cursor: int, count: int, alignment: int = 1) -> int:
        aligned = (cursor + alignment - 1) // alignment * alignment
        return aligned + count

    expected_vgprs = allocate(0, 8 * m_tiles * n_tiles, 8)
    expected_vgprs = allocate(
        expected_vgprs,
        8 * m_tiles * solution.prefetch_global_read,
        4,
    )
    expected_vgprs = allocate(expected_vgprs, 16 * solution.prefetch_local_read, 4)
    quant_type = solution_key.problem_type.quant_data_type
    payload_registers = 4 if quant_type == "Q4_K" else 8
    expected_vgprs = allocate(expected_vgprs, payload_registers * decoder_rows, 4)
    expected_vgprs = allocate(expected_vgprs, decoder_rows)
    expected_vgprs = allocate(
        expected_vgprs, (2 if quant_type == "Q3_K" else 3) * decoder_rows
    )
    if solution.lds_swizzle_chunk_b:
        expected_vgprs = allocate(expected_vgprs, 32 // solution.lds_swizzle_chunk_b)
    expected_vgprs = allocate(expected_vgprs, 8, 2)
    expected_vgprs = allocate(expected_vgprs, max(7, 3 + 2 * decoder_rows))
    if quant_type == "Q3_K" and n_tiles == 4:
        # RegisterPool reuses the single hole before the temporary range.
        expected_vgprs -= 1
    else:
        expected_vgprs = allocate(expected_vgprs, 1)
    expected = {
        ".kernarg_segment_size": 40,
        ".kernarg_segment_align": 8,
        ".group_segment_fixed_size": solution.lds_num_bytes,
        ".private_segment_fixed_size": 0,
        ".max_flat_workgroup_size": solution.num_threads,
        ".wavefront_size": solution.wavefront_size,
        ".vgpr_count": expected_vgprs,
        ".sgpr_count": 16,
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


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)
