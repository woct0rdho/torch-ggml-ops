"""Artifact inspection for isolated grouped MMQ forward kernels."""

from pathlib import Path

from .grouped_mmq_fwd_model import (
    GroupedForwardProblem,
    GroupedOperandSource,
    GroupedQ2DecodePolicy,
)
from .grouped_mmq_fwd_spec import DerivedGroupedForwardState, GroupedForwardKernelSpec
from .grouped_mmq_fwd_validation import validate_grouped_forward_solution
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
)
from .kernel_abi import GROUPED_FORWARD_ABI
from .toolchain import Toolchain


def inspect_grouped_forward_artifact(
    problem: GroupedForwardProblem,
    kernel_spec: GroupedForwardKernelSpec,
    kernel_name: str,
    code_object: Path,
    toolchain: Toolchain,
) -> ArtifactInspection:
    validate_grouped_forward_solution(problem, kernel_spec)
    if not code_object.is_file():
        raise InspectionError(f"code object does not exist: {code_object}")

    state = DerivedGroupedForwardState.from_problem_spec(problem, kernel_spec)
    readelf = toolchain.readelf_output(code_object)
    disassembly = toolchain.disassembly_output(code_object)
    metadata = _metadata(readelf)
    kernel = _kernel_metadata(metadata, kernel_name)
    instructions = _instructions(disassembly)
    assert _elf_value(readelf, "ABI Version") == "3"
    assert _elf_value(readelf, "Flags").endswith("gfx1151")
    assert _global_function_symbols(readelf) == {kernel_name}

    expected_metadata = {
        ".kernarg_segment_size": GROUPED_FORWARD_ABI.segment_size,
        ".kernarg_segment_align": GROUPED_FORWARD_ABI.segment_alignment,
        ".group_segment_fixed_size": state.physical_plan.resources.lds_bytes,
        ".private_segment_fixed_size": state.physical_plan.resources.private_bytes,
        ".max_flat_workgroup_size": (
            state.kernel_spec.geometry.work_group[0]
            * state.kernel_spec.geometry.work_group[1]
            * state.kernel_spec.geometry.work_group[2]
        ),
        ".wavefront_size": state.contract.wavefront_size,
        ".vgpr_count": state.physical_plan.resources.vgprs,
        ".sgpr_count": state.physical_plan.resources.sgprs,
        ".vgpr_spill_count": state.physical_plan.resources.vgpr_spills,
        ".sgpr_spill_count": state.physical_plan.resources.sgpr_spills,
    }
    for field, value in expected_metadata.items():
        assert kernel.get(field) == value
    assert kernel.get(".uses_dynamic_stack", False) is False
    assert _metadata_arguments(kernel) == GROUPED_FORWARD_ABI.metadata_arguments

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
    decoded_lds = (
        state.kernel_spec.operand_source is GroupedOperandSource.GroupedDecodedWeightLds
    )
    iq2_s_full_weight = (
        state.kernel_spec.operand_source
        is GroupedOperandSource.GroupedIQ2SFullWeightLds
    )
    if iq2_s_full_weight:
        expected_wmmas = 64
    elif decoded_lds:
        row_tiles = sum(state.kernel_spec.row_dispatch.body_row_tiles)
        if (
            problem.quant_data_type == "Q2_K"
            and isinstance(state.kernel_spec.decode, GroupedQ2DecodePolicy)
            and state.kernel_spec.decode.unrolled_groups
        ):
            expected_wmmas = 20 * row_tiles
        elif problem.quant_data_type == "Q2_K":
            expected_wmmas = 6 * row_tiles
        else:
            expected_wmmas = 4 * row_tiles
    else:
        expected_wmmas = 16
    expected_barriers = 4 if decoded_lds or iq2_s_full_weight else 0
    assert wmma_count == expected_wmmas
    assert barrier_count == expected_barriers
    assert not any(mnemonic.startswith("scratch_") for mnemonic in mnemonics)
    call_mnemonics = {
        mnemonic
        for mnemonic in mnemonics
        if "call" in mnemonic or mnemonic in {"s_setpc_b64", "s_swappc_b64"}
    }
    if not iq2_s_full_weight and "s_getpc_b64" in mnemonics:
        call_mnemonics.add("s_getpc_b64")
    assert not call_mnemonics
    max_vgpr = _max_register_index(disassembly, "v")
    max_sgpr = _max_register_index(disassembly, "s")
    assert max_vgpr < state.physical_plan.resources.vgprs
    assert max_sgpr < state.physical_plan.resources.sgprs
    return ArtifactInspection(
        kernel_name=kernel_name,
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
        lds_count=lds_count,
        wait_count=wait_count,
        clause_count=clause_count,
        delay_alu_count=delay_alu_count,
        buffer_gl0_inv_count=buffer_gl0_inv_count,
    )
