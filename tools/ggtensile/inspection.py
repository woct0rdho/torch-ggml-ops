import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .fixed_grouped_mmq_bwd_model import FixedBackwardSolutionKey
from .fixed_grouped_mmq_bwd_spec import DerivedFixedBackwardState
from .fixed_grouped_mmq_fwd_model import FixedForwardSolutionKey
from .fixed_grouped_mmq_fwd_spec import DerivedFixedForwardState
from .grouped_mmq_bwd_physical import derive_grouped_backward_physical_plan
from .grouped_mmq_bwd_spec import (
    DerivedGroupedBackwardState,
)
from .kernel_abi import (
    FIXED_GROUPED_BACKWARD_ABI,
    FIXED_GROUPED_FORWARD_ABI,
    GROUPED_BACKWARD_ABI,
    ORDINARY_BACKWARD_ABI,
    ORDINARY_FORWARD_ABI,
)
from .mmq_bwd_physical import derive_backward_physical_plan
from .mmq_bwd_spec import DerivedBackwardState
from .mmq_fwd_physical import (
    DecodedWeightLdsPhysicalPlan,
    Packed3BitTiledLdsPhysicalPlan,
    Q3FullWeightTiledLdsPhysicalPlan,
    Q6StructuredPhysicalPlan,
    SignedInt8DirectPhysicalPlan,
    SignedInt8RegisterTiledPhysicalPlan,
    SignedInt8SmallMTiledLdsPhysicalPlan,
    SignedInt8WaveNTiledLdsPhysicalPlan,
    derive_forward_physical_plan,
)
from .mmq_fwd_spec import ForwardKernelSpec, derive_forward_resource_usage
from .model import (
    BackwardSolution,
    ForwardSolution,
    GroupedBackwardSolution,
    SolutionKey,
)
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


def _backward_static_wmma_count(state: DerivedBackwardState) -> int:
    geometry = state.spec.geometry
    count = (
        geometry.matrix_instruction[5]
        * geometry.matrix_instruction[6]
        * geometry.depth_u
        // 16
    )
    if state.spec.pipeline.decoded_b_pipeline:
        count *= 1 + int(state.contract.problem_size.k > geometry.depth_u)
    return count


def _backward_static_barrier_count(state: DerivedBackwardState) -> int:
    pipeline = state.spec.pipeline
    if pipeline.decoded_b_pipeline:
        return 1 + int(state.contract.problem_size.k > state.spec.geometry.depth_u)
    return 3 if pipeline.prefetches_next_packed_tile else 2


def inspect_artifact(
    solution_key: SolutionKey | FixedForwardSolutionKey | FixedBackwardSolutionKey,
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
    solution = solution_key.solution
    grouped_state = (
        DerivedGroupedBackwardState.from_solution_key(solution_key)
        if isinstance(solution_key, SolutionKey)
        and isinstance(solution, GroupedBackwardSolution)
        else None
    )
    backward_state = (
        DerivedBackwardState.from_solution_key(solution_key)
        if isinstance(solution_key, SolutionKey)
        and isinstance(solution, BackwardSolution)
        else None
    )
    fixed_backward_state = (
        DerivedFixedBackwardState.from_solution_key(solution_key)
        if isinstance(solution_key, FixedBackwardSolutionKey)
        else None
    )
    _validate_metadata(
        kernel,
        solution_key,
        errors,
        backward_state=backward_state,
        grouped_backward_state=grouped_state,
        fixed_backward_state=fixed_backward_state,
    )

    mnemonics = tuple(instruction.split(None, 1)[0] for instruction in instructions)
    wmma_count = sum(
        mnemonic in {"v_wmma_f32_16x16x16_bf16", "v_wmma_i32_16x16x16_iu8"}
        for mnemonic in mnemonics
    )
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
    expected_wmmas = expected_wmma_count
    if expected_wmmas is None:
        if fixed_backward_state is not None:
            expected_wmmas = _backward_static_wmma_count(fixed_backward_state.ordinary)
        elif isinstance(solution_key, FixedForwardSolutionKey):
            expected_wmmas = solution_key.solution.macro_tile_tokens // 2
        elif isinstance(solution, ForwardSolution):
            physical = derive_forward_physical_plan(
                ForwardKernelSpec.from_solution(solution)
            )
            expected_wmmas = (
                8 * physical.layout.output_tile_rows
                if isinstance(physical, Q6StructuredPhysicalPlan)
                else 128
                if isinstance(
                    physical,
                    (Packed3BitTiledLdsPhysicalPlan, Q3FullWeightTiledLdsPhysicalPlan),
                )
                else 32
                if isinstance(physical, DecodedWeightLdsPhysicalPlan)
                else 8
                if isinstance(physical, SignedInt8DirectPhysicalPlan)
                else 32
                if isinstance(physical, SignedInt8RegisterTiledPhysicalPlan)
                else 2 * solution.depth_u
                if isinstance(physical, SignedInt8WaveNTiledLdsPhysicalPlan)
                else solution.macro_tile0 // 2
                if isinstance(physical, SignedInt8SmallMTiledLdsPhysicalPlan)
                else 16
            )
        elif isinstance(solution, BackwardSolution | GroupedBackwardSolution):
            primary_state = (
                grouped_state.primary if grouped_state is not None else backward_state
            )
            if primary_state is None:
                raise TypeError("backward inspection requires derived state")
            expected_wmmas = _backward_static_wmma_count(primary_state)
            if grouped_state is not None and grouped_state.secondary is not None:
                expected_wmmas += _backward_static_wmma_count(grouped_state.secondary)
    _require(
        wmma_count == expected_wmmas,
        f"expected {expected_wmmas} static WMMAs, found {wmma_count}",
        errors,
    )
    expected_barriers = expected_barrier_count
    if expected_barriers is None:
        if fixed_backward_state is not None:
            expected_barriers = _backward_static_barrier_count(
                fixed_backward_state.ordinary
            )
        elif isinstance(solution_key, FixedForwardSolutionKey):
            expected_barriers = 2
        elif isinstance(solution, ForwardSolution):
            physical = derive_forward_physical_plan(
                ForwardKernelSpec.from_solution(solution)
            )
            expected_barriers = (
                4
                if isinstance(
                    physical,
                    (
                        Q6StructuredPhysicalPlan,
                        DecodedWeightLdsPhysicalPlan,
                        Packed3BitTiledLdsPhysicalPlan,
                        Q3FullWeightTiledLdsPhysicalPlan,
                    ),
                )
                else 2 * solution.depth_u // 32
                if isinstance(physical, SignedInt8WaveNTiledLdsPhysicalPlan)
                else 2
                if isinstance(physical, SignedInt8SmallMTiledLdsPhysicalPlan)
                else 0
            )
        elif isinstance(solution, BackwardSolution | GroupedBackwardSolution):
            primary_state = (
                grouped_state.primary if grouped_state is not None else backward_state
            )
            if primary_state is None:
                raise TypeError("backward inspection requires derived state")
            expected_barriers = _backward_static_barrier_count(primary_state)
            if grouped_state is not None and grouped_state.secondary is not None:
                expected_barriers += _backward_static_barrier_count(
                    grouped_state.secondary
                )
            if (
                isinstance(solution_key, SolutionKey)
                and solution_key.problem_type.quant_data_type == "IQ2_S"
            ):
                expected_barriers += 1
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
    if (
        isinstance(solution_key, SolutionKey)
        and solution_key.problem_type.quant_data_type == "IQ2_S"
        and "s_getpc_b64" in call_mnemonics
    ):
        call_mnemonics.remove("s_getpc_b64")
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
    start = readelf.index(marker) + len("AMDGPU Metadata:\n        ")
    end = readelf.index("\n...\n", start) + len("\n...")
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
    solution_key: SolutionKey | FixedForwardSolutionKey | FixedBackwardSolutionKey,
    errors: list[str],
    *,
    backward_state: DerivedBackwardState | None,
    grouped_backward_state: DerivedGroupedBackwardState | None,
    fixed_backward_state: DerivedFixedBackwardState | None,
) -> None:
    if isinstance(solution_key, FixedBackwardSolutionKey):
        if fixed_backward_state is None:
            raise TypeError("fixed backward metadata requires derived state")
        _validate_fixed_backward_metadata(kernel, fixed_backward_state, errors)
        return
    if isinstance(solution_key, FixedForwardSolutionKey):
        _validate_fixed_forward_metadata(kernel, solution_key, errors)
        return
    solution = solution_key.solution
    if isinstance(solution, ForwardSolution):
        _validate_forward_metadata(kernel, solution, errors)
        return
    if isinstance(solution, GroupedBackwardSolution):
        if grouped_backward_state is None:
            raise TypeError("grouped backward metadata requires derived state")
        _validate_grouped_backward_metadata(kernel, grouped_backward_state, errors)
        return
    if backward_state is None:
        raise TypeError("backward metadata requires derived state")
    physical = derive_backward_physical_plan(backward_state)
    expected = {
        ".kernarg_segment_size": ORDINARY_BACKWARD_ABI.segment_size,
        ".kernarg_segment_align": ORDINARY_BACKWARD_ABI.segment_alignment,
        ".group_segment_fixed_size": physical.resources.lds_num_bytes,
        ".private_segment_fixed_size": physical.resources.private_segment_bytes,
        ".max_flat_workgroup_size": backward_state.spec.geometry.num_threads,
        ".wavefront_size": backward_state.spec.geometry.wavefront_size,
        ".vgpr_count": physical.resources.total_vgprs,
        ".sgpr_count": physical.resources.total_sgprs,
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
    _require(
        _metadata_arguments(kernel) == ORDINARY_BACKWARD_ABI.metadata_arguments,
        "kernarg ABI does not match",
        errors,
    )


def _validate_fixed_backward_metadata(
    kernel: Mapping[str, Any],
    state: DerivedFixedBackwardState,
    errors: list[str],
) -> None:
    resources = state.physical.resources
    geometry = state.spec.compute.geometry
    expected = {
        ".kernarg_segment_size": FIXED_GROUPED_BACKWARD_ABI.segment_size,
        ".kernarg_segment_align": FIXED_GROUPED_BACKWARD_ABI.segment_alignment,
        ".group_segment_fixed_size": resources.lds_num_bytes,
        ".private_segment_fixed_size": resources.private_segment_bytes,
        ".max_flat_workgroup_size": geometry.num_threads,
        ".wavefront_size": geometry.wavefront_size,
        ".vgpr_count": resources.total_vgprs,
        ".sgpr_count": resources.total_sgprs,
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
    _require(
        _metadata_arguments(kernel) == FIXED_GROUPED_BACKWARD_ABI.metadata_arguments,
        "fixed backward kernarg ABI does not match",
        errors,
    )


def _validate_grouped_backward_metadata(
    kernel: Mapping[str, Any],
    state: DerivedGroupedBackwardState,
    errors: list[str],
) -> None:
    physical = derive_grouped_backward_physical_plan(state)
    expected = {
        ".kernarg_segment_size": GROUPED_BACKWARD_ABI.segment_size,
        ".kernarg_segment_align": GROUPED_BACKWARD_ABI.segment_alignment,
        ".group_segment_fixed_size": physical.primary.resources.lds_num_bytes,
        ".private_segment_fixed_size": physical.primary.resources.private_segment_bytes,
        ".max_flat_workgroup_size": state.spec.compute.geometry.num_threads,
        ".wavefront_size": state.spec.compute.geometry.wavefront_size,
        ".vgpr_count": physical.primary.resources.total_vgprs,
        ".sgpr_count": physical.primary.resources.total_sgprs,
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
    _require(
        _metadata_arguments(kernel) == GROUPED_BACKWARD_ABI.metadata_arguments,
        "grouped backward kernarg ABI does not match",
        errors,
    )


def _validate_forward_metadata(
    kernel: Mapping[str, Any],
    solution: ForwardSolution,
    errors: list[str],
) -> None:
    kernel_spec = ForwardKernelSpec.from_solution(solution)
    resources = derive_forward_resource_usage(kernel_spec)
    expected = {
        ".kernarg_segment_size": ORDINARY_FORWARD_ABI.segment_size,
        ".kernarg_segment_align": ORDINARY_FORWARD_ABI.segment_alignment,
        ".group_segment_fixed_size": resources.lds_bytes,
        ".private_segment_fixed_size": 0,
        ".max_flat_workgroup_size": (
            kernel_spec.geometry.work_group[0]
            * kernel_spec.geometry.work_group[1]
            * kernel_spec.geometry.work_group[2]
        ),
        ".wavefront_size": solution.wavefront_size,
        ".vgpr_count": resources.vgprs,
        ".sgpr_count": resources.sgprs,
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
    _require(
        _metadata_arguments(kernel) == ORDINARY_FORWARD_ABI.metadata_arguments,
        "forward kernarg ABI does not match",
        errors,
    )


def _validate_fixed_forward_metadata(
    kernel: Mapping[str, Any],
    solution_key: FixedForwardSolutionKey,
    errors: list[str],
) -> None:
    state = DerivedFixedForwardState.from_solution_key(solution_key)
    resources = state.physical.resources
    expected = {
        ".kernarg_segment_size": FIXED_GROUPED_FORWARD_ABI.segment_size,
        ".kernarg_segment_align": FIXED_GROUPED_FORWARD_ABI.segment_alignment,
        ".group_segment_fixed_size": resources.lds_bytes,
        ".private_segment_fixed_size": 0,
        ".max_flat_workgroup_size": state.ordinary.num_threads,
        ".wavefront_size": solution_key.solution.wavefront_size,
        ".vgpr_count": resources.vgprs,
        ".sgpr_count": resources.sgprs,
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
    _require(
        _metadata_arguments(kernel) == FIXED_GROUPED_FORWARD_ABI.metadata_arguments,
        "fixed forward kernarg ABI does not match",
        errors,
    )


def _metadata_arguments(
    kernel: Mapping[str, Any],
) -> tuple[tuple[object, object, object, object, object], ...]:
    arguments = kernel.get(".args")
    if not isinstance(arguments, list):
        return ()
    return tuple(
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
