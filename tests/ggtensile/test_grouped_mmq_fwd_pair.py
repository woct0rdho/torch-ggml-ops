from dataclasses import replace
from pathlib import Path

import pytest

from tools.ggtensile.grouped_mmq_fwd_pair_inspection import (
    inspect_grouped_forward_pair_artifact,
)
from tools.ggtensile.grouped_mmq_fwd_pair_model import (
    GroupedForwardPairProblem,
    GroupedForwardPairSolution,
    GroupedForwardPairSolutionKey,
    GroupedPairRouteOwnership,
)
from tools.ggtensile.grouped_mmq_fwd_pair_runtime import (
    InstalledGroupedForwardPairIQ2XXSSerialControl,
)
from tools.ggtensile.grouped_mmq_fwd_pair_spec import (
    DerivedGroupedForwardPairState,
)
from tools.ggtensile.grouped_mmq_fwd_pair_validation import (
    validate_grouped_forward_pair_solution,
)
from tools.ggtensile.iq2_xxs_grid import (
    iq2_xxs_grid_rodata,
    iq2_xxs_grid_values,
)
from tools.ggtensile.kernel_writer_assembly_grouped_mmq_fwd_pair import (
    GroupedForwardPairKernelWriterAssembly,
)
from tools.ggtensile.model import SchemaError
from tools.ggtensile.runtime import HIPRuntimeError
from tools.ggtensile.toolchain import Toolchain


def _key(aggregate_rows: int = 16_384) -> GroupedForwardPairSolutionKey:
    return GroupedForwardPairSolutionKey(
        GroupedForwardPairProblem.iq2_s(aggregate_rows),
        GroupedForwardPairSolution.iq2_s_k128_interleaved(),
    )


def _row_task_key(aggregate_rows: int = 16_384) -> GroupedForwardPairSolutionKey:
    return GroupedForwardPairSolutionKey(
        GroupedForwardPairProblem.iq2_s(aggregate_rows),
        GroupedForwardPairSolution.iq2_s_k128_interleaved_row_tasks(),
    )


def _q3_key(aggregate_rows: int = 16_384) -> GroupedForwardPairSolutionKey:
    return GroupedForwardPairSolutionKey(
        GroupedForwardPairProblem.q3_k(aggregate_rows),
        GroupedForwardPairSolution.q3_k_k128_interleaved(),
    )


def _q3_row_task_key(
    aggregate_rows: int = 16_384,
) -> GroupedForwardPairSolutionKey:
    return GroupedForwardPairSolutionKey(
        GroupedForwardPairProblem.q3_k(aggregate_rows),
        GroupedForwardPairSolution.q3_k_k128_interleaved_row_tasks(),
    )


def _iq2_xxs_key(aggregate_rows: int = 12_288) -> GroupedForwardPairSolutionKey:
    return GroupedForwardPairSolutionKey(
        GroupedForwardPairProblem.iq2_xxs(aggregate_rows),
        GroupedForwardPairSolution.iq2_xxs_k128_interleaved(),
    )


@pytest.mark.parametrize("field", ("unknown", "SchemaVersion"))
def test_grouped_pair_key_rejects_unknown_root_fields(field: str) -> None:
    mapping = _key().to_mapping()
    mapping[field] = 1
    with pytest.raises(SchemaError, match="unknown"):
        GroupedForwardPairSolutionKey.from_mapping(mapping)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("quant_type", "Q3_K"),
        ("output_features", 1),
        ("input_features", 1),
        ("physical_experts", 1),
        ("max_route_entries", 1),
        ("projection_count", 1),
        ("block_values", 128),
        ("activation_layout", "F16_D4S4"),
        ("activation_block_bytes", 128),
        ("packed_weight_block_bytes", 1),
        ("kernel_language", "Source"),
        ("isa", [11, 0, 0]),
        ("wavefront_size", 64),
        ("arithmetic_contract", "Unknown"),
        ("weight_decode", "Prepared"),
        ("metadata_conversion", "Unknown"),
        ("scale_arithmetic", "FP16"),
        ("signed_weight", False),
        ("signed_activation", False),
        ("wmma_clamp", True),
        ("destination_type", "Float32"),
        ("bf16_rounding", "Truncate"),
        ("abi_family", "Unknown"),
    ),
)
def test_grouped_pair_key_rejects_noncanonical_contract_fields(
    field: str, value: object
) -> None:
    mapping = _key().to_mapping()
    contract = mapping["ProblemContract"]
    assert isinstance(contract, dict)
    contract[field] = value
    with pytest.raises(SchemaError, match="canonical|unsupported"):
        GroupedForwardPairSolutionKey.from_mapping(mapping)


def test_grouped_pair_key_rejects_invalid_problem_and_route_enum() -> None:
    mapping = _key().to_mapping()
    problem = mapping["Problem"]
    assert isinstance(problem, dict)
    problem["aggregate_rows"] = 0
    with pytest.raises(SchemaError, match="positive u32"):
        GroupedForwardPairSolutionKey.from_mapping(mapping)

    mapping = _key().to_mapping()
    kernel_spec = mapping["KernelSpec"]
    assert isinstance(kernel_spec, dict)
    lowering = kernel_spec["lowering"]
    assert isinstance(lowering, dict)
    lowering["route_ownership"] = 1
    with pytest.raises(SchemaError, match="must be str"):
        GroupedForwardPairSolutionKey.from_mapping(mapping)


def test_grouped_pair_route_ownership_belongs_only_to_the_kernel_spec() -> None:
    mapping = _row_task_key().to_mapping()
    contract = mapping["ProblemContract"]
    kernel_spec = mapping["KernelSpec"]
    assert isinstance(contract, dict)
    assert isinstance(kernel_spec, dict)
    lowering = kernel_spec["lowering"]
    assert isinstance(lowering, dict)
    assert "route_ownership" not in contract
    assert lowering["route_ownership"] == "DeviceRowTasks64"


def test_iq2_xxs_pair_key_rejects_unimplemented_row_task_ownership() -> None:
    mapping = _iq2_xxs_key().to_mapping()
    kernel_spec = mapping["KernelSpec"]
    assert isinstance(kernel_spec, dict)
    lowering = kernel_spec["lowering"]
    assert isinstance(lowering, dict)
    lowering["route_ownership"] = "DeviceRowTasks64"
    with pytest.raises(SchemaError, match="unavailable"):
        GroupedForwardPairSolutionKey.from_mapping(mapping)


@pytest.mark.parametrize("aggregate_rows", (16_384, 65_536, 262_144))
def test_grouped_iq2_s_pair_production_keys_derive(aggregate_rows: int) -> None:
    key = _key(aggregate_rows)
    assert GroupedForwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert validate_grouped_forward_pair_solution(key) == ()
    state = DerivedGroupedForwardPairState.from_solution_key(key)
    assert state.expected_packed_weight_shape == (256, 512, 656)
    assert state.expected_activation_shape == (16, aggregate_rows, 144)
    assert state.expected_output_shape == (aggregate_rows, 512)
    assert state.grid(256) == (8, 256, 1)
    assert state.blocks_per_weight_row == 8
    assert state.bytes_per_expert == 335_872
    assert state.physical_plan.resources.vgprs == 148
    assert state.physical_plan.resources.sgprs == 44
    assert state.physical_plan.resources.lds_bytes == 19_456
    assert state.physical_plan.layout.weight_row_stride == 160
    assert state.physical_plan.layout.weight_row_stride % 16 == 0


def test_grouped_iq2_s_pair_serialization_rejects_unknown_enum() -> None:
    mapping = _key().to_mapping()
    kernel_spec = mapping["KernelSpec"]
    assert isinstance(kernel_spec, dict)
    projection = kernel_spec["projection"]
    assert isinstance(projection, dict)
    projection["schedule"] = "projection_schedule"
    with pytest.raises(ValueError, match="ProjectionSchedule"):
        GroupedForwardPairSolutionKey.from_mapping(mapping)


def test_grouped_iq2_s_pair_sources_remain_byte_stable() -> None:
    toolchain = Toolchain.discover()
    serial_writer = GroupedForwardPairKernelWriterAssembly(_key(35), toolchain)
    row_task_writer = GroupedForwardPairKernelWriterAssembly(
        _row_task_key(35), toolchain
    )
    assert serial_writer.source() == serial_writer.source()
    assert row_task_writer.source() == row_task_writer.source()
    assert serial_writer.source() != row_task_writer.source()


@pytest.mark.parametrize(
    ("aggregate_rows", "capacity"),
    ((16_384, 512), (65_536, 1_280), (262_144, 4_352)),
)
def test_grouped_iq2_s_pair_row_task_keys_derive(
    aggregate_rows: int, capacity: int
) -> None:
    key = _row_task_key(aggregate_rows)
    assert GroupedForwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert key.hash != _key(aggregate_rows).hash
    assert key.solution.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks64
    assert validate_grouped_forward_pair_solution(key) == ()
    state = DerivedGroupedForwardPairState.from_solution_key(key)
    assert state.row_task_capacity(256) == capacity
    assert state.row_task_grid(256) == (8, capacity, 1)
    assert state.physical_plan.scalar_registers.row_end.first_register == 23
    assert (
        state.physical_plan.scalar_registers.activation_plane_stride.first_register
        == 29
    )
    assert state.physical_plan.resources.vgprs == 148
    assert state.physical_plan.resources.sgprs == 44
    assert state.physical_plan.resources.lds_bytes == 19_456


@pytest.mark.parametrize("aggregate_rows", (16_384, 65_536, 262_144))
def test_grouped_q3_k_pair_production_keys_derive(aggregate_rows: int) -> None:
    serial = _q3_key(aggregate_rows)
    row_tasks = _q3_row_task_key(aggregate_rows)
    assert GroupedForwardPairSolutionKey.from_mapping(serial.to_mapping()) == serial
    assert (
        GroupedForwardPairSolutionKey.from_mapping(row_tasks.to_mapping()) == row_tasks
    )
    assert serial.hash != row_tasks.hash
    assert validate_grouped_forward_pair_solution(serial) == ()
    assert validate_grouped_forward_pair_solution(row_tasks) == ()
    for key in (serial, row_tasks):
        state = DerivedGroupedForwardPairState.from_solution_key(key)
        assert state.expected_packed_weight_shape == (256, 512, 880)
        assert state.expected_activation_shape == (16, aggregate_rows, 144)
        assert state.expected_output_shape == (aggregate_rows, 512)
        assert state.blocks_per_weight_row == 8
        assert state.bytes_per_expert == 450_560
        if key.solution.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks64:
            capacity = aggregate_rows // 64 + 256
            assert state.row_task_capacity(256) == capacity
            assert state.row_task_grid(256) == (8, capacity, 1)
            assert state.physical_plan.scalar_registers.row_end.first_register == 23
        assert state.physical_plan.resources.vgprs == 148
        assert state.physical_plan.resources.sgprs == 44
        assert state.physical_plan.resources.lds_bytes == 19_456
        assert state.physical_plan.layout.weight_row_stride == 160


def test_grouped_q3_k_pair_r35_identities_are_distinct_and_round_trip() -> None:
    serial = _q3_key(35)
    row_task = _q3_row_task_key(35)
    assert serial.kernel_name != row_task.kernel_name
    assert GroupedForwardPairSolutionKey.from_mapping(serial.to_mapping()) == serial
    assert GroupedForwardPairSolutionKey.from_mapping(row_task.to_mapping()) == row_task


@pytest.mark.parametrize("aggregate_rows", (12_288, 49_152, 196_608))
def test_grouped_iq2_xxs_pair_production_keys_derive(aggregate_rows: int) -> None:
    key = _iq2_xxs_key(aggregate_rows)
    assert GroupedForwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert validate_grouped_forward_pair_solution(key) == ()
    state = DerivedGroupedForwardPairState.from_solution_key(key)
    assert state.expected_packed_weight_shape == (256, 2048, 1056)
    assert state.expected_activation_shape == (32, aggregate_rows, 144)
    assert state.expected_output_shape == (aggregate_rows, 2048)
    assert state.grid(256) == (32, 256, 1)
    assert state.blocks_per_weight_row == 16
    assert state.bytes_per_expert == 2_162_688
    assert state.physical_plan.resources.vgprs == 148
    assert state.physical_plan.resources.sgprs == 44
    assert state.physical_plan.resources.lds_bytes == 19_456
    assert state.physical_plan.layout.weight_row_stride == 160
    assert state.semantics.payload_plane("d").byte_offset == 0
    assert state.semantics.payload_plane("grid_indices_and_signs").byte_offset == 2


def test_grouped_iq2_xxs_pair_r35_identity_and_control_abi_are_stable() -> None:
    assert InstalledGroupedForwardPairIQ2XXSSerialControl._CONFIGS == {
        64: (
            "torch_ggml_ops_mmq_gfx1151_v1_grouped_fwd_serial_iq2_xxs_n2048_k4096_j64",
            28_928,
        ),
        80: (
            "torch_ggml_ops_mmq_gfx1151_v1_grouped_fwd_serial_iq2_xxs_n2048_k4096_j80",
            31_552,
        ),
    }
    assert InstalledGroupedForwardPairIQ2XXSSerialControl.BYTES_PER_EXPERT == 2_162_688
    with pytest.raises(HIPRuntimeError, match="requires J64 or J80"):
        InstalledGroupedForwardPairIQ2XXSSerialControl(96)


def test_grouped_iq2_xxs_grid_is_extracted_from_the_vendor_authority() -> None:
    values = iq2_xxs_grid_values()
    assert len(values) == 256
    assert values[0] == 0x0808080808080808
    assert values[-1] == 0x2B2B2B1908081908
    rodata = iq2_xxs_grid_rodata(".LTestIQ2XXSGrid")
    assert rodata.count(".quad ") == 64
    assert ".size .LTestIQ2XXSGrid, 2048" in rodata
    with pytest.raises(ValueError, match="assembly-local"):
        iq2_xxs_grid_rodata("IQ2XXSGrid")


def test_grouped_iq2_s_pair_validation_rejects_other_geometry_and_solution() -> None:
    key = _key()
    invalid_problem = replace(key.problem, output_features=1024)
    invalid_solution = replace(key.solution, depth_u=256)
    assert {
        reason.rule_id
        for reason in validate_grouped_forward_pair_solution(
            replace(key, problem=invalid_problem, solution=invalid_solution)
        )
    } == {
        "grouped_forward_pair.problem.unsupported",
        "grouped_forward_pair.solution.unimplemented",
    }


def test_grouped_pair_validation_rejects_cross_quant_solution() -> None:
    key = _q3_key()
    reasons = validate_grouped_forward_pair_solution(
        replace(key, solution=GroupedForwardPairSolution.iq2_s_k128_interleaved())
    )
    assert tuple(reason.rule_id for reason in reasons) == (
        "grouped_forward_pair.solution.unimplemented",
    )

    iq2_xxs = _iq2_xxs_key()
    reasons = validate_grouped_forward_pair_solution(
        replace(iq2_xxs, solution=GroupedForwardPairSolution.q3_k_k128_interleaved())
    )
    assert tuple(reason.rule_id for reason in reasons) == (
        "grouped_forward_pair.solution.unimplemented",
    )


def test_grouped_iq2_s_pair_writer_has_shared_k128_dataflow() -> None:
    source = GroupedForwardPairKernelWriterAssembly(
        _key(35), Toolchain.discover()
    ).source()
    assert "paired grouped IQ2_S MMQ forward, K128 interleaved" in source
    assert source.count("Decode paired IQ2_S selected K128 half 0") == 2
    assert source.count("Decode paired IQ2_S selected K128 half 1") == 2
    assert source.count("Linearly stage one coalesced 9,216-byte") == 2
    assert source.count("v_wmma_i32_16x16x16_iu8") == 128
    assert source.count("s_barrier") == 8
    assert source.count(".quad ") == 256
    assert source.count('.section .rodata,"a",@progbits') == 1
    assert "s_load_dwordx2 s[4:5], s[0:1], 0x0" in source
    assert "s_load_dwordx2 s[6:7], s[0:1], 0x8" in source
    assert "s_load_dwordx2 s[8:9], s[0:1], 0x10" in source
    assert "s_load_dwordx2 s[10:11], s[0:1], 0x18" in source
    assert "s_load_dwordx2 s[12:13], s[0:1], 0x20" in source
    assert "v_mul_lo_u32 v141, 160, v132" in source
    assert "offset0:32 offset1:112" in source
    assert "v_lshlrev_b32 v95, 2, v95\n  v_add_nc_u32 v95, 1, v95" in source
    assert "s_add_u32 s30, s30, s29" in source
    assert "global_store_d16_hi_b16" in source
    assert "s[10:11]" in source
    assert "s[12:13]" in source


def test_grouped_iq2_s_pair_row_task_writer_uses_device_descriptors() -> None:
    source = GroupedForwardPairKernelWriterAssembly(
        _row_task_key(35), Toolchain.discover()
    ).source()
    assert "device 64-row task ownership" in source
    assert ".kernarg_segment_size:       96" in source
    assert "s_load_dwordx2 s[14:15], s[0:1], 0x28" in source
    assert "s_load_dwordx2 s[18:19], s[0:1], 0x38" in source
    assert "s_load_dwordx2 s[20:21], s[0:1], 0x40" in source
    assert "s_load_dword s31, s[14:15], 0x0" in source
    assert "s_load_dword s23, s[20:21], s31" in source
    assert "s_sub_u32 s37, s23, s28" in source
    assert "s_mul_i32 s29, s24, 144" in source
    assert source.count("v_wmma_i32_16x16x16_iu8") == 128
    assert source.count("s_barrier") == 8


def test_grouped_q3_k_pair_writer_has_shared_k128_dataflow() -> None:
    source = GroupedForwardPairKernelWriterAssembly(
        _q3_key(35), Toolchain.discover()
    ).source()
    assert "paired grouped Q3_K MMQ forward, K128 interleaved" in source
    assert source.count("Decode paired Q3_K selected K128 half 0") == 2
    assert source.count("Decode paired Q3_K selected K128 half 1") == 2
    assert source.count("Linearly stage one coalesced 9,216-byte") == 2
    assert source.count("v_wmma_i32_16x16x16_iu8") == 128
    assert source.count("s_barrier") == 8
    assert '.section .rodata,"a",@progbits' not in source
    assert "global_load_b128 v[64:67]" in source
    assert "global_load_b128 v[68:71]" in source
    assert "global_load_b128 v[72:75]" in source
    assert "offset:94" in source
    assert "v_sub_nc_u32 v83, v83, 32" in source
    assert "v_add_nc_u32 v138, 110, v138" in source
    assert "Store paired Q3_K projection 0" in source
    assert "Store paired Q3_K projection 1" in source


def test_grouped_q3_k_pair_row_task_writer_uses_device_descriptors() -> None:
    source = GroupedForwardPairKernelWriterAssembly(
        _q3_row_task_key(35), Toolchain.discover()
    ).source()
    assert "device 64-row task ownership" in source
    assert "Load the paired 96-byte grouped Q3_K row-task ABI" in source
    assert ".kernarg_segment_size:       96" in source
    assert "s_load_dword s23, s[20:21], s31" in source
    assert source.count("v_wmma_i32_16x16x16_iu8") == 128
    assert source.count("s_barrier") == 8


def test_grouped_iq2_xxs_pair_writer_has_q8_style_k32_dataflow() -> None:
    source = GroupedForwardPairKernelWriterAssembly(
        _iq2_xxs_key(35), Toolchain.discover()
    ).source()
    assert "paired grouped IQ2_XXS MMQ forward, K128 interleaved" in source
    assert source.count("Decode paired IQ2_XXS selected K128 half 0") == 2
    assert source.count("Decode paired IQ2_XXS selected K128 half 1") == 2
    assert source.count("Linearly stage one coalesced 9,216-byte") == 2
    assert source.count("v_wmma_i32_16x16x16_iu8") == 128
    assert source.count("v[124:131] neg_lo:[1,1,0]") == 64
    assert source.count("v_cvt_f32_i32") == 512
    assert source.count("s_barrier") == 8
    assert source.count("v_bcnt_u32_b32") == 32
    assert source.count("v_perm_b32") == 64
    assert source.count(".quad ") == 64
    assert source.count('.section .rodata,"a",@progbits') == 1
    assert "global_load_ushort v64, v138, s[4:5] offset:0" in source
    assert "global_load_b64 v[67:68], v100, s[4:5] offset:2" in source
    assert "global_load_b64 v[69:70], v100, s[4:5] offset:10" in source
    assert "v_xor_b32 v96, v96, v99" in source
    assert (
        "v_wmma_i32_16x16x16_iu8 v[64:71], v[96:99], v[100:103], "
        "v[64:71] neg_lo:[1,1,0]"
    ) in source
    assert source.count("v_lshlrev_b32 v96, 12") == 8
    assert "v_lshlrev_b32 v96, 10" not in source
    assert "Store paired IQ2_XXS projection 0" in source
    assert "Store paired IQ2_XXS projection 1" in source


def _build(
    directory: Path,
    key: GroupedForwardPairSolutionKey,
    toolchain: Toolchain,
) -> tuple[bytes, bytes, Path]:
    directory.mkdir(parents=True)
    assembly = directory / "kernel.s"
    obj = directory / "kernel.o"
    code_object = directory / "kernel.hsaco"
    GroupedForwardPairKernelWriterAssembly(key, toolchain).write(assembly)
    toolchain.assemble(assembly, obj)
    toolchain.link(obj, code_object)
    return assembly.read_bytes(), code_object.read_bytes(), code_object


def test_grouped_iq2_s_pair_artifact_is_deterministic_and_resource_clean(
    tmp_path: Path,
) -> None:
    key = _key(35)
    toolchain = Toolchain.discover()
    first_source, first_code, first_path = _build(tmp_path / "first", key, toolchain)
    second_source, second_code, _ = _build(tmp_path / "second", key, toolchain)
    assert first_source == second_source
    assert first_code == second_code
    inspection = inspect_grouped_forward_pair_artifact(key, first_path, toolchain)
    assert inspection.code_object_version == 5
    assert inspection.target == "gfx1151"
    assert inspection.kernarg_segment_size == 80
    assert inspection.wavefront_size == 32
    assert inspection.max_flat_workgroup_size == 128
    assert inspection.vgpr_count == 148
    assert inspection.sgpr_count == 44
    assert inspection.lds_num_bytes == 19_456
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0
    assert inspection.wmma_count == 128
    assert inspection.barrier_count == 8


def test_grouped_iq2_s_pair_row_task_artifact_is_resource_clean(
    tmp_path: Path,
) -> None:
    key = _row_task_key(35)
    toolchain = Toolchain.discover()
    _, _, artifact = _build(tmp_path / "row-task", key, toolchain)
    inspection = inspect_grouped_forward_pair_artifact(key, artifact, toolchain)
    assert inspection.kernarg_segment_size == 96
    assert inspection.wavefront_size == 32
    assert inspection.max_flat_workgroup_size == 128
    assert inspection.vgpr_count == 148
    assert inspection.sgpr_count == 44
    assert inspection.lds_num_bytes == 19_456
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0
    assert inspection.wmma_count == 128
    assert inspection.barrier_count == 8


@pytest.mark.parametrize(
    "key_factory",
    (_q3_key, _q3_row_task_key),
    ids=("serial", "row-tasks"),
)
def test_grouped_q3_k_pair_artifact_is_deterministic_and_resource_clean(
    tmp_path: Path,
    key_factory,
) -> None:
    key = key_factory(35)
    toolchain = Toolchain.discover()
    first_source, first_code, first_path = _build(tmp_path / "first", key, toolchain)
    second_source, second_code, _ = _build(tmp_path / "second", key, toolchain)
    assert first_source == second_source
    assert first_code == second_code
    inspection = inspect_grouped_forward_pair_artifact(key, first_path, toolchain)
    assert inspection.code_object_version == 5
    assert inspection.target == "gfx1151"
    assert inspection.kernarg_segment_size == (
        96
        if key.solution.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks64
        else 80
    )
    assert inspection.wavefront_size == 32
    assert inspection.max_flat_workgroup_size == 128
    assert inspection.vgpr_count == 148
    assert inspection.sgpr_count == 44
    assert inspection.lds_num_bytes == 19_456
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0
    assert inspection.wmma_count == 128
    assert inspection.barrier_count == 8


def test_grouped_iq2_xxs_pair_artifact_is_deterministic_and_resource_clean(
    tmp_path: Path,
) -> None:
    key = _iq2_xxs_key(35)
    toolchain = Toolchain.discover()
    first_source, first_code, first_path = _build(tmp_path / "first", key, toolchain)
    second_source, second_code, _ = _build(tmp_path / "second", key, toolchain)
    assert first_source == second_source
    assert first_code == second_code
    inspection = inspect_grouped_forward_pair_artifact(key, first_path, toolchain)
    assert inspection.code_object_version == 5
    assert inspection.target == "gfx1151"
    assert inspection.kernarg_segment_size == 80
    assert inspection.wavefront_size == 32
    assert inspection.max_flat_workgroup_size == 128
    assert inspection.vgpr_count == 148
    assert inspection.sgpr_count == 44
    assert inspection.lds_num_bytes == 19_456
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0
    assert inspection.wmma_count == 128
    assert inspection.barrier_count == 8
