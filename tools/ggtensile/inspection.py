import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .family_registry import instance_name
from .fixed_grouped_mmq_bwd_model import FixedBackwardProblem
from .fixed_grouped_mmq_bwd_spec import (
    DerivedFixedBackwardState,
    FixedBackwardKernelSpec,
)
from .fixed_grouped_mmq_fwd_model import FixedForwardProblem
from .fixed_grouped_mmq_fwd_spec import (
    DerivedFixedForwardState,
    FixedForwardKernelSpec,
)
from .grouped_mmq_bwd_physical import derive_grouped_backward_physical_plan
from .grouped_mmq_bwd_spec import (
    DerivedGroupedBackwardState,
    GroupedBackwardKernelSpec,
)
from .identity import KernelFamily
from .kernel_abi import (
    FIXED_GROUPED_BACKWARD_ABI,
    FIXED_GROUPED_FORWARD_ABI,
    GROUPED_BACKWARD_ABI,
    ORDINARY_BACKWARD_ABI,
    ORDINARY_FORWARD_ABI,
)
from .kernel_instance import KernelInstance
from .mmq_bwd_physical import derive_backward_physical_plan
from .mmq_bwd_spec import BackwardKernelSpec, DerivedBackwardState
from .mmq_fwd_physical import (
    DecodedWeightLdsPhysicalPlan,
    Packed3BitTiledLdsPhysicalPlan,
    Q3FullWeightTiledLdsPhysicalPlan,
    Q6StructuredPhysicalPlan,
    SignedInt8DirectPhysicalPlan,
    SignedInt8RegisterTiledPhysicalPlan,
    SignedInt8SmallMTiledLdsPhysicalPlan,
    SignedInt8WaveNTiledLdsPhysicalPlan,
)
from .mmq_fwd_spec import DerivedForwardState, ForwardKernelSpec
from .model import ProblemSize
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


def _forward_static_wmma_count(state: DerivedForwardState) -> int:
    physical = state.physical_plan
    if isinstance(physical, Q6StructuredPhysicalPlan):
        return 8 * physical.layout.output_tile_rows
    if isinstance(
        physical,
        (Packed3BitTiledLdsPhysicalPlan, Q3FullWeightTiledLdsPhysicalPlan),
    ):
        return 128
    if isinstance(physical, DecodedWeightLdsPhysicalPlan):
        return 32
    if isinstance(physical, SignedInt8DirectPhysicalPlan):
        return 8
    if isinstance(physical, SignedInt8RegisterTiledPhysicalPlan):
        return 32
    if isinstance(physical, SignedInt8WaveNTiledLdsPhysicalPlan):
        return 2 * state.kernel_spec.geometry.depth_u
    if isinstance(physical, SignedInt8SmallMTiledLdsPhysicalPlan):
        return physical.layout.activation_rows // 2
    return 16


def _backward_static_barrier_count(state: DerivedBackwardState) -> int:
    pipeline = state.spec.pipeline
    if pipeline.decoded_b_pipeline:
        return 1 + int(state.contract.problem_size.k > state.spec.geometry.depth_u)
    return 3 if pipeline.prefetches_next_packed_tile else 2


def inspect_artifact(
    instance: KernelInstance,
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
    kernel_name = instance_name(instance)
    kernel = _kernel_metadata(metadata, kernel_name)
    instructions = _instructions(disassembly)

    assert _elf_value(readelf, "ABI Version") == "3"
    assert _elf_value(readelf, "Flags").endswith("gfx1151")
    functions = _global_function_symbols(readelf)
    assert functions == {kernel_name}
    family = instance.family
    problem = instance.problem
    spec = instance.kernel_spec
    quant_type = instance.problem_type.quant_data_type
    forward_state = None
    grouped_state = None
    backward_state = None
    fixed_backward_state = None
    fixed_forward_state = None
    if family is KernelFamily.OrdinaryForward:
        assert isinstance(problem, ProblemSize) and isinstance(spec, ForwardKernelSpec)
        forward_state = DerivedForwardState.from_problem_spec(problem, quant_type, spec)
    elif family is KernelFamily.GroupedBackward:
        assert isinstance(problem, ProblemSize) and isinstance(
            spec, GroupedBackwardKernelSpec
        )
        grouped_state = DerivedGroupedBackwardState.from_problem_spec(
            problem, quant_type, spec
        )
    elif family is KernelFamily.OrdinaryBackward:
        assert isinstance(problem, ProblemSize) and isinstance(spec, BackwardKernelSpec)
        backward_state = DerivedBackwardState.from_problem_spec(
            problem, quant_type, spec
        )
    elif family is KernelFamily.FixedGroupedBackward:
        assert isinstance(problem, FixedBackwardProblem) and isinstance(
            spec, FixedBackwardKernelSpec
        )
        fixed_backward_state = DerivedFixedBackwardState.from_problem_spec(
            problem, spec
        )
    elif family is KernelFamily.FixedGroupedForward:
        assert isinstance(problem, FixedForwardProblem) and isinstance(
            spec, FixedForwardKernelSpec
        )
        fixed_forward_state = DerivedFixedForwardState.from_problem_spec(problem, spec)
    _validate_metadata(
        kernel,
        family,
        forward_state=forward_state,
        backward_state=backward_state,
        grouped_backward_state=grouped_state,
        fixed_backward_state=fixed_backward_state,
        fixed_forward_state=fixed_forward_state,
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
        elif fixed_forward_state is not None:
            expected_wmmas = fixed_forward_state.spec.macro_tile_tokens // 2
        elif forward_state is not None:
            expected_wmmas = _forward_static_wmma_count(forward_state)
        elif backward_state is not None or grouped_state is not None:
            primary_state = (
                grouped_state.primary if grouped_state is not None else backward_state
            )
            if primary_state is None:
                raise TypeError("backward inspection requires derived state")
            expected_wmmas = _backward_static_wmma_count(primary_state)
            if grouped_state is not None and grouped_state.secondary is not None:
                expected_wmmas += _backward_static_wmma_count(grouped_state.secondary)
    assert wmma_count == expected_wmmas
    expected_barriers = expected_barrier_count
    if expected_barriers is None:
        if fixed_backward_state is not None:
            expected_barriers = _backward_static_barrier_count(
                fixed_backward_state.ordinary
            )
        elif fixed_forward_state is not None:
            expected_barriers = 2
        elif forward_state is not None:
            physical = forward_state.physical_plan
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
                else 2 * forward_state.kernel_spec.geometry.depth_u // 32
                if isinstance(physical, SignedInt8WaveNTiledLdsPhysicalPlan)
                else 2
                if isinstance(physical, SignedInt8SmallMTiledLdsPhysicalPlan)
                else 0
            )
        elif backward_state is not None or grouped_state is not None:
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
            if quant_type == "IQ2_S":
                expected_barriers += 1
    assert barrier_count == expected_barriers
    assert not any(mnemonic.startswith("scratch_") for mnemonic in mnemonics)
    call_mnemonics = {
        mnemonic
        for mnemonic in mnemonics
        if "call" in mnemonic
        or mnemonic in {"s_getpc_b64", "s_setpc_b64", "s_swappc_b64"}
    }
    if quant_type == "IQ2_S" and "s_getpc_b64" in call_mnemonics:
        call_mnemonics.remove("s_getpc_b64")
    assert not call_mnemonics

    max_vgpr = _max_register_index(disassembly, "v")
    max_sgpr = _max_register_index(disassembly, "s")
    vgpr_count = _integer(kernel, ".vgpr_count")
    sgpr_count = _integer(kernel, ".sgpr_count")
    assert max_vgpr < vgpr_count
    assert max_sgpr < sgpr_count

    return ArtifactInspection(
        kernel_name=kernel_name,
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
        raise InspectionError("metadata kernel name does not match KernelSpecKey")
    return kernel


def _validate_metadata(
    kernel: Mapping[str, Any],
    family: KernelFamily,
    *,
    forward_state: DerivedForwardState | None,
    backward_state: DerivedBackwardState | None,
    grouped_backward_state: DerivedGroupedBackwardState | None,
    fixed_backward_state: DerivedFixedBackwardState | None,
    fixed_forward_state: DerivedFixedForwardState | None,
) -> None:
    if family is KernelFamily.FixedGroupedBackward:
        if fixed_backward_state is None:
            raise TypeError("fixed backward metadata requires derived state")
        _validate_fixed_backward_metadata(kernel, fixed_backward_state)
        return
    if family is KernelFamily.FixedGroupedForward:
        if fixed_forward_state is None:
            raise TypeError("fixed forward metadata requires derived state")
        _validate_fixed_forward_metadata(kernel, fixed_forward_state)
        return
    if forward_state is not None:
        _validate_forward_metadata(kernel, forward_state)
        return
    if grouped_backward_state is not None:
        _validate_grouped_backward_metadata(kernel, grouped_backward_state)
        return
    if backward_state is None:
        raise TypeError("backward metadata requires derived state")
    physical = derive_backward_physical_plan(backward_state)
    expected = {
        ".kernarg_segment_size": ORDINARY_BACKWARD_ABI.segment_size,
        ".kernarg_segment_align": ORDINARY_BACKWARD_ABI.segment_alignment,
        ".group_segment_fixed_size": physical.resources.lds_bytes,
        ".private_segment_fixed_size": physical.resources.private_bytes,
        ".max_flat_workgroup_size": backward_state.spec.geometry.num_threads,
        ".wavefront_size": backward_state.spec.geometry.wavefront_size,
        ".vgpr_count": physical.resources.vgprs,
        ".sgpr_count": physical.resources.sgprs,
        ".vgpr_spill_count": physical.resources.vgpr_spills,
        ".sgpr_spill_count": physical.resources.sgpr_spills,
    }
    for field, value in expected.items():
        actual = kernel.get(field)
        assert actual == value
    assert kernel.get(".uses_dynamic_stack", False) is False
    assert _metadata_arguments(kernel) == ORDINARY_BACKWARD_ABI.metadata_arguments


def _validate_fixed_backward_metadata(
    kernel: Mapping[str, Any],
    state: DerivedFixedBackwardState,
) -> None:
    resources = state.physical.resources
    geometry = state.spec.compute.geometry
    expected = {
        ".kernarg_segment_size": FIXED_GROUPED_BACKWARD_ABI.segment_size,
        ".kernarg_segment_align": FIXED_GROUPED_BACKWARD_ABI.segment_alignment,
        ".group_segment_fixed_size": resources.lds_bytes,
        ".private_segment_fixed_size": resources.private_bytes,
        ".max_flat_workgroup_size": geometry.num_threads,
        ".wavefront_size": geometry.wavefront_size,
        ".vgpr_count": resources.vgprs,
        ".sgpr_count": resources.sgprs,
        ".vgpr_spill_count": resources.vgpr_spills,
        ".sgpr_spill_count": resources.sgpr_spills,
    }
    for field, value in expected.items():
        actual = kernel.get(field)
        assert actual == value
    assert kernel.get(".uses_dynamic_stack", False) is False
    assert _metadata_arguments(kernel) == FIXED_GROUPED_BACKWARD_ABI.metadata_arguments


def _validate_grouped_backward_metadata(
    kernel: Mapping[str, Any],
    state: DerivedGroupedBackwardState,
) -> None:
    physical = derive_grouped_backward_physical_plan(state)
    expected = {
        ".kernarg_segment_size": GROUPED_BACKWARD_ABI.segment_size,
        ".kernarg_segment_align": GROUPED_BACKWARD_ABI.segment_alignment,
        ".group_segment_fixed_size": physical.primary.resources.lds_bytes,
        ".private_segment_fixed_size": physical.primary.resources.private_bytes,
        ".max_flat_workgroup_size": state.spec.compute.geometry.num_threads,
        ".wavefront_size": state.spec.compute.geometry.wavefront_size,
        ".vgpr_count": physical.primary.resources.vgprs,
        ".sgpr_count": physical.primary.resources.sgprs,
        ".vgpr_spill_count": physical.primary.resources.vgpr_spills,
        ".sgpr_spill_count": physical.primary.resources.sgpr_spills,
    }
    for field, value in expected.items():
        actual = kernel.get(field)
        assert actual == value
    assert kernel.get(".uses_dynamic_stack", False) is False
    assert _metadata_arguments(kernel) == GROUPED_BACKWARD_ABI.metadata_arguments


def _validate_forward_metadata(
    kernel: Mapping[str, Any],
    state: DerivedForwardState,
) -> None:
    kernel_spec = state.kernel_spec
    resources = state.resources
    expected = {
        ".kernarg_segment_size": ORDINARY_FORWARD_ABI.segment_size,
        ".kernarg_segment_align": ORDINARY_FORWARD_ABI.segment_alignment,
        ".group_segment_fixed_size": resources.lds_bytes,
        ".private_segment_fixed_size": resources.private_bytes,
        ".max_flat_workgroup_size": (
            kernel_spec.geometry.work_group[0]
            * kernel_spec.geometry.work_group[1]
            * kernel_spec.geometry.work_group[2]
        ),
        ".wavefront_size": state.contract.wavefront_size,
        ".vgpr_count": resources.vgprs,
        ".sgpr_count": resources.sgprs,
        ".vgpr_spill_count": resources.vgpr_spills,
        ".sgpr_spill_count": resources.sgpr_spills,
    }
    for field, value in expected.items():
        actual = kernel.get(field)
        assert actual == value
    assert kernel.get(".uses_dynamic_stack", False) is False
    assert _metadata_arguments(kernel) == ORDINARY_FORWARD_ABI.metadata_arguments


def _validate_fixed_forward_metadata(
    kernel: Mapping[str, Any],
    state: DerivedFixedForwardState,
) -> None:
    resources = state.physical.resources
    expected = {
        ".kernarg_segment_size": FIXED_GROUPED_FORWARD_ABI.segment_size,
        ".kernarg_segment_align": FIXED_GROUPED_FORWARD_ABI.segment_alignment,
        ".group_segment_fixed_size": resources.lds_bytes,
        ".private_segment_fixed_size": resources.private_bytes,
        ".max_flat_workgroup_size": state.ordinary.num_threads,
        ".wavefront_size": state.contract.wavefront_size,
        ".vgpr_count": resources.vgprs,
        ".sgpr_count": resources.sgprs,
        ".vgpr_spill_count": resources.vgpr_spills,
        ".sgpr_spill_count": resources.sgpr_spills,
    }
    for field, value in expected.items():
        actual = kernel.get(field)
        assert actual == value
    assert kernel.get(".uses_dynamic_stack", False) is False
    assert _metadata_arguments(kernel) == FIXED_GROUPED_FORWARD_ABI.metadata_arguments


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
