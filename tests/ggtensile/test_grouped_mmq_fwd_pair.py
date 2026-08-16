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
from tools.ggtensile.grouped_mmq_fwd_pair_spec import (
    DerivedGroupedForwardPairState,
)
from tools.ggtensile.grouped_mmq_fwd_pair_validation import (
    validate_grouped_forward_pair_solution,
)
from tools.ggtensile.kernel_writer_assembly_grouped_mmq_fwd_pair import (
    GroupedForwardPairKernelWriterAssembly,
)
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
    solution = mapping["Solution"]
    assert isinstance(solution, dict)
    solution["ProjectionSchedule"] = "projection_schedule"
    with pytest.raises(ValueError, match="ProjectionSchedule"):
        GroupedForwardPairSolutionKey.from_mapping(mapping)


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
