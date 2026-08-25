"""Strict artifact inspection for paired grouped backward kernels."""

from pathlib import Path

from .grouped_mmq_bwd_pair_model import GroupedBackwardPairProblem
from .grouped_mmq_bwd_pair_physical import (
    derive_grouped_backward_pair_physical_plan,
)
from .grouped_mmq_bwd_pair_spec import (
    DerivedGroupedBackwardPairState,
    GroupedBackwardPairKernelSpec,
)
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
)
from .kernel_abi import GROUPED_BACKWARD_PAIR_ABI
from .toolchain import Toolchain


def inspect_grouped_backward_pair_artifact(
    problem: GroupedBackwardPairProblem,
    kernel_spec: GroupedBackwardPairKernelSpec,
    kernel_name: str,
    code_object: Path,
    toolchain: Toolchain,
) -> ArtifactInspection:
    validate_grouped_backward_pair_solution(problem, kernel_spec)
    if not code_object.is_file():
        raise InspectionError(f"code object does not exist: {code_object}")
    state = DerivedGroupedBackwardPairState.from_problem_spec(problem, kernel_spec)
    physical = derive_grouped_backward_pair_physical_plan(state)
    resources = physical.ordinary.resources
    readelf = toolchain.readelf_output(code_object)
    disassembly = toolchain.disassembly_output(code_object)
    metadata = _metadata(readelf)
    kernel = _kernel_metadata(metadata, kernel_name)
    instructions = _instructions(disassembly)

    assert _elf_value(readelf, "ABI Version") == "3"
    assert _elf_value(readelf, "Flags").endswith("gfx1151")
    assert _global_function_symbols(readelf) == {kernel_name}
    expected_metadata = {
        ".kernarg_segment_size": GROUPED_BACKWARD_PAIR_ABI.segment_size,
        ".kernarg_segment_align": GROUPED_BACKWARD_PAIR_ABI.segment_alignment,
        ".group_segment_fixed_size": resources.lds_bytes,
        ".private_segment_fixed_size": resources.private_bytes,
        ".max_flat_workgroup_size": state.kernel_spec.compute.geometry.num_threads,
        ".wavefront_size": state.kernel_spec.compute.geometry.wavefront_size,
        ".vgpr_count": resources.vgprs,
        ".sgpr_count": resources.sgprs,
        ".vgpr_spill_count": resources.vgpr_spills,
        ".sgpr_spill_count": resources.sgpr_spills,
    }
    for field, value in expected_metadata.items():
        assert kernel.get(field) == value
    assert kernel.get(".uses_dynamic_stack", False) is False
    assert _metadata_arguments(kernel) == GROUPED_BACKWARD_PAIR_ABI.metadata_arguments

    mnemonics = tuple(instruction.split(None, 1)[0] for instruction in instructions)
    wmma_count = mnemonics.count("v_wmma_f32_16x16x16_bf16")
    barrier_count = mnemonics.count("s_barrier")
    expected_wmmas = (
        2
        * state.kernel_spec.compute.geometry.matrix_instruction[5]
        * state.kernel_spec.compute.geometry.matrix_instruction[6]
        * state.kernel_spec.compute.geometry.depth_u
        // 16
    )
    if state.kernel_spec.projection_policy.split_full_tiles:
        expected_wmmas *= 2
    assert wmma_count == expected_wmmas
    policy = state.kernel_spec.projection_policy
    expected_barriers = (
        2
        + 2 * (not policy.dual_lds)
        + 2 * policy.split_full_tiles
        + 2 * policy.pipeline_k
    )
    if (
        state.contract.quant_type in ("IQ2_S", "IQ2_XXS")
        and physical.ordinary.lds.codebook_in_lds
    ):
        expected_barriers += 1
    assert barrier_count == expected_barriers
    assert not any(mnemonic.startswith("scratch_") for mnemonic in mnemonics)
    calls = {
        mnemonic
        for mnemonic in mnemonics
        if "call" in mnemonic or mnemonic in {"s_setpc_b64", "s_swappc_b64"}
    }
    assert not calls
    max_vgpr = _max_register_index(disassembly, "v")
    max_sgpr = _max_register_index(disassembly, "s")
    assert max_vgpr < resources.vgprs
    assert max_sgpr < resources.sgprs

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
        lds_count=sum(mnemonic.startswith("ds_") for mnemonic in mnemonics),
        wait_count=sum(mnemonic.startswith("s_waitcnt") for mnemonic in mnemonics),
        clause_count=mnemonics.count("s_clause"),
        delay_alu_count=mnemonics.count("s_delay_alu"),
        buffer_gl0_inv_count=mnemonics.count("buffer_gl0_inv"),
    )
