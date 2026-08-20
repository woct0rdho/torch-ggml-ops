"""Strict artifact inspection for paired grouped backward kernels."""

from pathlib import Path

from .grouped_mmq_bwd_pair_model import (
    GroupedBackwardPairProjectionSchedule,
    GroupedBackwardPairSolutionKey,
)
from .grouped_mmq_bwd_pair_physical import (
    derive_grouped_backward_pair_physical_plan,
)
from .grouped_mmq_bwd_pair_spec import DerivedGroupedBackwardPairState
from .grouped_mmq_bwd_pair_validation import validate_grouped_backward_pair_solution
from .inspection import (
    ArtifactInspection,
    InspectionError,
    _elf_value,
    _global_function_symbols,
    _instructions,
    _integer,
    _kernel_metadata,
    _max_register_index,
    _metadata,
    _metadata_arguments,
    _require,
)
from .kernel_abi import GROUPED_BACKWARD_PAIR_ABI
from .toolchain import Toolchain


def inspect_grouped_backward_pair_artifact(
    key: GroupedBackwardPairSolutionKey,
    code_object: Path,
    toolchain: Toolchain,
) -> ArtifactInspection:
    reasons = validate_grouped_backward_pair_solution(key)
    if reasons:
        raise InspectionError(
            "cannot inspect rejected backward pair: "
            + "; ".join(reason.rule_id for reason in reasons)
        )
    if not code_object.is_file():
        raise InspectionError(f"code object does not exist: {code_object}")
    state = DerivedGroupedBackwardPairState.from_solution_key(key)
    physical = derive_grouped_backward_pair_physical_plan(state)
    resources = physical.ordinary.resources
    readelf = toolchain.readelf_output(code_object)
    disassembly = toolchain.disassembly_output(code_object)
    metadata = _metadata(readelf)
    kernel = _kernel_metadata(metadata, key.kernel_name)
    instructions = _instructions(disassembly)
    errors: list[str] = []

    _require(_elf_value(readelf, "ABI Version") == "3", "code object is not v5", errors)
    _require(
        _elf_value(readelf, "Flags").endswith("gfx1151"),
        "ELF target is not gfx1151",
        errors,
    )
    _require(
        _global_function_symbols(readelf) == {key.kernel_name},
        "global symbols do not match the backward-pair key",
        errors,
    )
    expected_metadata = {
        ".kernarg_segment_size": GROUPED_BACKWARD_PAIR_ABI.segment_size,
        ".kernarg_segment_align": GROUPED_BACKWARD_PAIR_ABI.segment_alignment,
        ".group_segment_fixed_size": resources.lds_num_bytes,
        ".private_segment_fixed_size": 0,
        ".max_flat_workgroup_size": key.solution.num_threads,
        ".wavefront_size": key.solution.compute.wavefront_size,
        ".vgpr_count": resources.total_vgprs,
        ".sgpr_count": resources.total_sgprs,
        ".vgpr_spill_count": 0,
        ".sgpr_spill_count": 0,
    }
    for field, value in expected_metadata.items():
        _require(
            kernel.get(field) == value,
            f"{field} is {kernel.get(field)!r}, expected {value}",
            errors,
        )
    _require(
        kernel.get(".uses_dynamic_stack", False) is False,
        "dynamic stack is enabled",
        errors,
    )
    _require(
        _metadata_arguments(kernel) == GROUPED_BACKWARD_PAIR_ABI.metadata_arguments,
        "backward-pair kernarg ABI does not match",
        errors,
    )

    mnemonics = tuple(instruction.split(None, 1)[0] for instruction in instructions)
    wmma_count = mnemonics.count("v_wmma_f32_16x16x16_bf16")
    barrier_count = mnemonics.count("s_barrier")
    expected_wmmas = (
        2
        * key.solution.compute.matrix_instruction[5]
        * key.solution.compute.matrix_instruction[6]
        * key.solution.compute.depth_u
        // 16
    )
    if key.solution.projection_schedule in (
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchASerialReadsInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersOverlapSecondReadPrefetchAInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsPrefetchAInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchAInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineGlobalCodebookInterleaveWmmaWaitsDepthU,
    ):
        expected_wmmas *= 2
    _require(
        wmma_count == expected_wmmas,
        f"expected {expected_wmmas} static WMMAs, found {wmma_count}",
        errors,
    )
    expected_barriers = {
        GroupedBackwardPairProjectionSchedule.InterleavedDepthU: 4,
        GroupedBackwardPairProjectionSchedule.DualLdsInterleavedDepthU: 2,
        GroupedBackwardPairProjectionSchedule.DualLdsGlobalCodebookInterleavedDepthU: 2,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitInterleavedDepthU: 4,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersInterleavedDepthU: 4,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchASerialReadsInterleavedDepthU: 4,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersOverlapSecondReadPrefetchAInterleavedDepthU: 4,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsInterleavedDepthU: 4,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsPrefetchAInterleavedDepthU: 4,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchAInterleavedDepthU: 4,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineInterleavedDepthU: 6,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineGlobalCodebookInterleaveWmmaWaitsDepthU: 6,
    }[key.solution.projection_schedule]
    if (
        state.contract.quant_type in ("IQ2_S", "IQ2_XXS")
        and physical.ordinary.lds.codebook_in_lds
    ):
        expected_barriers += 1
    _require(
        barrier_count == expected_barriers,
        f"expected {expected_barriers} static barriers, found {barrier_count}",
        errors,
    )
    _require(
        not any(mnemonic.startswith("scratch_") for mnemonic in mnemonics),
        "scratch instruction found",
        errors,
    )
    calls = {
        mnemonic
        for mnemonic in mnemonics
        if "call" in mnemonic or mnemonic in {"s_setpc_b64", "s_swappc_b64"}
    }
    _require(not calls, f"call instruction found: {sorted(calls)}", errors)
    max_vgpr = _max_register_index(disassembly, "v")
    max_sgpr = _max_register_index(disassembly, "s")
    _require(
        max_vgpr < resources.total_vgprs,
        "VGPR index exceeds metadata declaration",
        errors,
    )
    _require(
        max_sgpr < resources.total_sgprs,
        "SGPR index exceeds metadata declaration",
        errors,
    )
    if errors:
        raise InspectionError("backward-pair artifact rejected: " + "; ".join(errors))

    vopd_count = sum(" :: " in instruction for instruction in instructions)
    valu_issue_count = sum(
        mnemonic.startswith("v_") and not mnemonic.startswith("v_wmma_")
        for mnemonic in mnemonics
    )
    vmem_count = sum(
        mnemonic.startswith(("global_", "flat_", "buffer_", "scratch_"))
        and mnemonic != "buffer_gl0_inv"
        for mnemonic in mnemonics
    )
    return ArtifactInspection(
        kernel_name=key.kernel_name,
        code_object_version=5,
        target="gfx1151",
        kernarg_segment_size=_integer(kernel, ".kernarg_segment_size"),
        wavefront_size=_integer(kernel, ".wavefront_size"),
        max_flat_workgroup_size=_integer(kernel, ".max_flat_workgroup_size"),
        lds_num_bytes=_integer(kernel, ".group_segment_fixed_size"),
        vgpr_count=_integer(kernel, ".vgpr_count"),
        sgpr_count=_integer(kernel, ".sgpr_count"),
        private_segment_bytes=_integer(kernel, ".private_segment_fixed_size"),
        vgpr_spill_count=_integer(kernel, ".vgpr_spill_count"),
        sgpr_spill_count=_integer(kernel, ".sgpr_spill_count"),
        max_vgpr_index=max_vgpr,
        max_sgpr_index=max_sgpr,
        wmma_count=wmma_count,
        barrier_count=barrier_count,
        valu_issue_count=valu_issue_count,
        valu_operation_count=valu_issue_count + vopd_count,
        vopd_count=vopd_count,
        vmem_count=vmem_count,
        lds_count=sum(mnemonic.startswith("ds_") for mnemonic in mnemonics),
        wait_count=sum(mnemonic.startswith("s_waitcnt") for mnemonic in mnemonics),
        clause_count=mnemonics.count("s_clause"),
        delay_alu_count=mnemonics.count("s_delay_alu"),
        buffer_gl0_inv_count=mnemonics.count("buffer_gl0_inv"),
    )
