from dataclasses import replace

import pytest

from tools.ggtensile.fixed_grouped_mmq_fwd_model import (
    FixedForwardProblem,
    FixedForwardSolution,
    FixedForwardSolutionKey,
)
from tools.ggtensile.fixed_grouped_mmq_fwd_spec import DerivedFixedForwardState
from tools.ggtensile.fixed_grouped_mmq_fwd_validation import (
    fixed_forward_rejection_reason,
    validate_fixed_forward_solution_key,
)
from tools.ggtensile.inspection import inspect_artifact
from tools.ggtensile.kernel_writer_assembly_fixed_grouped_mmq_fwd import (
    FixedGroupedForwardKernelWriterAssembly,
)
from tools.ggtensile.model import SchemaError
from tools.ggtensile.toolchain import Toolchain


def _key(tokens: int = 2048) -> FixedForwardSolutionKey:
    return FixedForwardSolutionKey(
        FixedForwardProblem.deepseek_q8_0(tokens),
        FixedForwardSolution.q8_0_small_m_tiled_lds(),
    )


def _compact_key(tokens: int = 2048) -> FixedForwardSolutionKey:
    return FixedForwardSolutionKey(
        FixedForwardProblem.deepseek_q8_0(tokens),
        FixedForwardSolution.q8_0_compact_depth32_tiled_lds(),
    )


def _hoisted_key(tokens: int = 2048) -> FixedForwardSolutionKey:
    return FixedForwardSolutionKey(
        FixedForwardProblem.deepseek_q8_0(tokens),
        FixedForwardSolution.q8_0_compact_depth32_tiled_lds_hoisted(),
    )


def _weight_hoisted_key(tokens: int = 2048) -> FixedForwardSolutionKey:
    return FixedForwardSolutionKey(
        FixedForwardProblem.deepseek_q8_0(tokens),
        FixedForwardSolution.q8_0_compact_depth32_tiled_lds_weight_hoisted(),
    )


@pytest.mark.parametrize(
    "field", ("unknown", "SchemaVersion", "ArtifactKind", "KernelFamily")
)
def test_fixed_forward_key_rejects_unknown_root_fields(field: str) -> None:
    mapping = _key().to_mapping()
    mapping[field] = 1
    with pytest.raises(SchemaError, match="unknown"):
        FixedForwardSolutionKey.from_mapping(mapping)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("quant_type", "Q4_K"),
        ("groups", 4),
        ("block_values", 256),
        ("packed_weight_block_bytes", 144),
        ("activation_layout", "F16_D4S4"),
        ("activation_block_bytes", 128),
        ("arithmetic_contract", "Unknown"),
        ("kernel_language", "Source"),
        ("isa", [11, 0, 0]),
        ("wavefront_size", 64),
        ("weight_decode", "Prepared"),
        ("activation_addressing", "Routed"),
        ("scale_arithmetic", "FP16"),
        ("signed_weight", False),
        ("signed_activation", False),
        ("wmma_clamp", True),
        ("destination_type", "Float32"),
        ("bf16_rounding", "Truncate"),
        ("abi", "Unknown"),
    ),
)
def test_fixed_forward_key_rejects_noncanonical_contract_fields(
    field: str, value: object
) -> None:
    mapping = _key().to_mapping()
    contract = mapping["ProblemContract"]
    assert isinstance(contract, dict)
    contract[field] = value
    with pytest.raises(SchemaError):
        FixedForwardSolutionKey.from_mapping(mapping)


def test_fixed_forward_key_rejects_invalid_problem_and_enum() -> None:
    mapping = _key().to_mapping()
    problem = mapping["Problem"]
    assert isinstance(problem, dict)
    problem["tokens"] = 0
    with pytest.raises(ValueError, match="positive u32"):
        FixedForwardSolutionKey.from_mapping(mapping)

    mapping = _key().to_mapping()
    kernel_spec = mapping["KernelSpec"]
    assert isinstance(kernel_spec, dict)
    lowering = kernel_spec["lowering"]
    assert isinstance(lowering, dict)
    lowering["operand_source"] = 1
    with pytest.raises(SchemaError, match="must be str"):
        FixedForwardSolutionKey.from_mapping(mapping)


def test_fixed_forward_identity_roundtrip_and_derived_shapes() -> None:
    key = _key()
    assert FixedForwardSolutionKey.from_mapping(key.to_mapping()) == key
    validate_fixed_forward_solution_key(key)
    state = DerivedFixedForwardState.from_solution_key(key)
    assert state.grid == (16, 32, 8)
    assert state.ordinary.activation_plane_stride_bytes == 8 * 2048 * 144
    assert state.expected_packed_weight_shape == (8, 1024, 4352)
    assert state.expected_activation_shape == (32, 16_384, 144)
    assert state.expected_output_shape == (2048, 8, 1024)
    assert state.physical.resources.lds_bytes == 28_672
    assert state.physical.resources.vgprs == 144
    assert state.physical.resources.sgprs == 16
    compact_key = _compact_key()
    assert FixedForwardSolutionKey.from_mapping(compact_key.to_mapping()) == compact_key
    compact_state = DerivedFixedForwardState.from_solution_key(compact_key)
    assert compact_state.physical.resources.lds_bytes == 18_432
    assert compact_state.physical.resources.vgprs == 144
    assert compact_state.physical.resources.sgprs == 16
    hoisted_key = _hoisted_key()
    assert FixedForwardSolutionKey.from_mapping(hoisted_key.to_mapping()) == hoisted_key
    hoisted_state = DerivedFixedForwardState.from_solution_key(hoisted_key)
    assert hoisted_state.physical.paired_weight_scale_address_vgpr == 139
    assert hoisted_state.physical.activation_plane_stride_sgpr == 15
    weight_hoisted_state = DerivedFixedForwardState.from_solution_key(
        _weight_hoisted_key()
    )
    assert weight_hoisted_state.physical.weight_lane_offset_vgpr == 140
    assert weight_hoisted_state.physical.weight_payload_lds_address_vgpr == 141
    assert weight_hoisted_state.physical.weight_scale_lds_address_vgpr == 142


def test_fixed_forward_accepts_formula_compatible_noncatalog_shape() -> None:
    mapping = _key().to_mapping()
    contract = mapping["ProblemContract"]
    problem = mapping["Problem"]
    assert isinstance(contract, dict)
    assert isinstance(problem, dict)
    contract["output_features"] = 512
    contract["input_features"] = 2048
    problem["tokens"] = 4096

    key = FixedForwardSolutionKey.from_mapping(mapping)
    validate_fixed_forward_solution_key(key)
    state = DerivedFixedForwardState.from_solution_key(key)
    assert state.grid == (8, 64, 8)
    assert state.expected_packed_weight_shape == (8, 512, 2176)
    assert state.expected_activation_shape == (16, 32_768, 144)
    assert state.expected_output_shape == (4096, 8, 512)


@pytest.mark.parametrize(
    "bad_key, message",
    [
        (
            replace(
                _key(),
                problem=replace(_key().problem, groups=4),
            ),
            "eight groups",
        ),
        (
            replace(
                _key(),
                solution=replace(_key().solution, work_group=(64, 2, 1)),
            ),
            "workgroup",
        ),
        (
            replace(
                _key(),
                problem=replace(_key().problem, tokens=32),
            ),
            "64-row tile",
        ),
        (
            replace(
                _key(),
                solution=replace(_key().solution, lds_address_hoist="Unknown"),
            ),
            "LDS addressing policy",
        ),
        (
            replace(
                _key(),
                solution=replace(_key().solution, fixed_address_hoist="ReductionLoop"),
            ),
            "requires compact DepthU32 LDS",
        ),
        (
            replace(
                _compact_key(),
                solution=replace(
                    _compact_key().solution, fixed_address_hoist="Unknown"
                ),
            ),
            "address hoist",
        ),
    ],
)
def test_fixed_forward_rejects_cross_contract_values(
    bad_key: FixedForwardSolutionKey,
    message: str,
) -> None:
    assert fixed_forward_rejection_reason(bad_key) is not None
    with pytest.raises(ValueError, match=message):
        validate_fixed_forward_solution_key(bad_key)


def test_fixed_forward_source_loads_scalars_and_flattens_group_rows() -> None:
    key = _key()
    source = FixedGroupedForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    state = DerivedFixedForwardState.from_solution_key(key)
    output_address = state.physical.ordinary.registers.output_address.first_register
    temporary = state.physical.ordinary.registers.temporary.first_register
    assert "s_load_dword s12, s[0:1], 0x18" in source
    assert "s_load_dword s13, s[0:1], 0x1c" in source
    assert "s_load_dwordx2 s[14:15], s[0:1], 0x20" in source
    assert f"v_mul_lo_u32 v{output_address}, s13, v{output_address}" in source
    assert f"v_lshlrev_b32 v{temporary}, 8, s13" in source
    assert "s[10:11]" in source
    assert "s[8:9]" in source


def test_fixed_forward_compact_source_uses_paired_scale_reads() -> None:
    key = _compact_key()
    source = FixedGroupedForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert source.count("ds_read2_b32") == 16
    assert source.count("ds_read_b32") == 16
    assert "offset0:179 offset1:251" in source
    assert (
        DerivedFixedForwardState.from_solution_key(key).physical.resources.lds_bytes
        == 18_432
    )


def test_fixed_forward_hoists_reduction_invariants() -> None:
    source = FixedGroupedForwardKernelWriterAssembly(
        _hoisted_key(), Toolchain.discover()
    ).source()
    loop = source.index(".LFixedGroupedQ80BlockLoop:")
    assert source.index("v_add_nc_u32 v139, 1152, v128") < loop
    assert source.index("s_mul_i32 s15, s12, 1152") < loop
    assert "v_mul_lo_u32 v135, 1152, s12" not in source
    assert source.count("v139 offset0:") == 8


def test_fixed_forward_hoists_weight_stage_addresses() -> None:
    source = FixedGroupedForwardKernelWriterAssembly(
        _weight_hoisted_key(), Toolchain.discover()
    ).source()
    loop = source.index(".LFixedGroupedQ80BlockLoop:")
    assert source.index("v_mul_lo_u32 v140, 68, v140") < loop
    assert source.index("v_add_nc_u32 v141, v134, v141") < loop
    assert source.index("v_add_nc_u32 v142, v134, v142") < loop
    assert source.count("v_add_nc_u32 v8, v129, v140") == 1
    assert source.count("ds_write_b128 v141") == 4
    assert source.count("ds_write_b32 v142") == 2


def test_fixed_forward_build_is_deterministic_and_inspectable(tmp_path) -> None:
    key = _weight_hoisted_key()
    toolchain = Toolchain.discover()
    first = FixedGroupedForwardKernelWriterAssembly(key, toolchain)
    second = FixedGroupedForwardKernelWriterAssembly(key, toolchain)
    first_source = first.source()
    second_source = second.source()
    assert first_source == second_source

    first_assembly = tmp_path / "first.s"
    first_object = tmp_path / "first.o"
    first_code_object = tmp_path / "first.hsaco"
    first.write(first_assembly)
    toolchain.assemble(first_assembly, first_object)
    toolchain.link(first_object, first_code_object)
    inspection = inspect_artifact(key, first_code_object, toolchain)
    assert inspection.kernarg_segment_size == 40
    assert inspection.lds_num_bytes == 18_432
    assert inspection.wmma_count == 32
    assert inspection.barrier_count == 2
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0
    assert inspection.max_vgpr_index == 142
    assert inspection.max_sgpr_index == 15
