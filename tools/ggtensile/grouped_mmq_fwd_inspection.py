"""Artifact inspection for isolated grouped MMQ forward kernels."""

from collections.abc import Mapping
from pathlib import Path

from .grouped_mmq_fwd_model import GroupedForwardSolutionKey
from .grouped_mmq_fwd_spec import DerivedGroupedForwardState
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
    _require,
)
from .toolchain import Toolchain

_EXPECTED_GROUPED_ARGS = (
    ("weights", 0, 8, "global_buffer", "struct"),
    ("activations", 8, 8, "global_buffer", "struct"),
    ("dst", 16, 8, "global_buffer", "bf16"),
    ("expert_indices", 24, 8, "global_buffer", "i64"),
    ("expert_offsets", 32, 8, "global_buffer", "i32"),
    ("num_experts", 40, 4, "by_value", "u32"),
    ("nrows_weight", 44, 4, "by_value", "u32"),
    ("nrows_activation", 48, 4, "by_value", "u32"),
    ("blocks_per_weight_row", 52, 4, "by_value", "u32"),
    ("bytes_per_expert", 56, 8, "by_value", "u64"),
)


def inspect_grouped_forward_artifact(
    solution_key: GroupedForwardSolutionKey,
    code_object: Path,
    toolchain: Toolchain,
) -> ArtifactInspection:
    reasons = validate_grouped_forward_solution(solution_key)
    if reasons:
        details = "; ".join(reason.rule_id for reason in reasons)
        raise InspectionError(f"cannot inspect rejected grouped solution: {details}")
    if not code_object.is_file():
        raise InspectionError(f"code object does not exist: {code_object}")

    state = DerivedGroupedForwardState.from_solution_key(solution_key)
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
    _require(
        _global_function_symbols(readelf) == {solution_key.kernel_name},
        "global kernel symbols do not match grouped solution",
        errors,
    )

    expected_metadata = {
        ".kernarg_segment_size": 64,
        ".kernarg_segment_align": 8,
        ".group_segment_fixed_size": state.physical_plan.resources.lds_bytes,
        ".private_segment_fixed_size": 0,
        ".max_flat_workgroup_size": solution_key.solution.num_threads,
        ".wavefront_size": solution_key.solution.wavefront_size,
        ".vgpr_count": state.physical_plan.resources.vgprs,
        ".sgpr_count": state.physical_plan.resources.sgprs,
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
    _require(
        actual_args == _EXPECTED_GROUPED_ARGS,
        "grouped kernarg ABI does not match",
        errors,
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
    decoded_lds = solution_key.solution.operand_source == "GroupedDecodedWeightLds"
    if decoded_lds:
        row_tiles = solution_key.solution.macro_tile0 // 16
        if solution_key.solution.tail_macro_tile0 < solution_key.solution.macro_tile0:
            row_tiles += solution_key.solution.tail_macro_tile0 // 16
        expected_wmmas = 4 * row_tiles
    else:
        expected_wmmas = 16
    expected_barriers = 4 if decoded_lds else 0
    _require(
        wmma_count == expected_wmmas,
        f"expected {expected_wmmas} static WMMAs, found {wmma_count}",
        errors,
    )
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
    _require(
        max_vgpr < state.physical_plan.resources.vgprs,
        "VGPR index exceeds metadata declaration",
        errors,
    )
    _require(
        max_sgpr < state.physical_plan.resources.sgprs,
        "SGPR index exceeds metadata declaration",
        errors,
    )
    if errors:
        raise InspectionError("grouped artifact rejected: " + "; ".join(errors))

    return ArtifactInspection(
        kernel_name=solution_key.kernel_name,
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
