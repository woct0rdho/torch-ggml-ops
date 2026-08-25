from dataclasses import replace
from pathlib import Path

import pytest

from tools.ggtensile.campaign import load_catalog
from tools.ggtensile.family_registry import (
    launch_for_instance,
    mapping_for_instance,
    parse_instance,
    writer_for_instance,
)
from tools.ggtensile.fixed_grouped_mmq_bwd_model import FixedBackwardProblem
from tools.ggtensile.fixed_grouped_mmq_bwd_spec import (
    DerivedFixedBackwardState,
    FixedBackwardKernelSpec,
)
from tools.ggtensile.fixed_grouped_mmq_bwd_validation import (
    validate_fixed_backward_solution,
)
from tools.ggtensile.inspection import inspect_artifact
from tools.ggtensile.kernel_instance import KernelInstance
from tools.ggtensile.schema import SchemaError
from tools.ggtensile.toolchain import Toolchain

_CONFIG = Path(__file__).resolve().parents[2] / "tools/ggtensile/configs"


def _instances() -> tuple[KernelInstance, ...]:
    catalog = load_catalog(_CONFIG / "mmq_fixed_grouped_bwd_q8_0_catalog.json")
    return tuple(entry.instance for entry in catalog.entries)


def _problem_spec(
    instance: KernelInstance,
) -> tuple[FixedBackwardProblem, FixedBackwardKernelSpec]:
    assert isinstance(instance.problem, FixedBackwardProblem)
    assert isinstance(instance.kernel_spec, FixedBackwardKernelSpec)
    return instance.problem, instance.kernel_spec


def test_fixed_backward_catalog_keys_round_trip_and_derive() -> None:
    instances = _instances()
    assert len(instances) == 3
    for instance in instances:
        assert parse_instance(mapping_for_instance(instance)) == instance
        problem, spec = _problem_spec(instance)
        validate_fixed_backward_solution(problem, spec)
        state = DerivedFixedBackwardState.from_problem_spec(problem, spec)
        assert state.expected_grad_output_shape == (
            problem.tokens,
            problem.groups,
            problem.output_features,
        )
        assert state.expected_packed_weight_shape[0] == 8
        assert state.expected_grad_input_shape[2] == problem.input_features
        assert state.physical.resources.private_bytes == 0


def test_fixed_backward_spec_is_the_compute_authority() -> None:
    spec = _problem_spec(_instances()[0])[1]
    assert isinstance(spec, FixedBackwardKernelSpec)
    assert spec.compute.geometry.depth_u == 64
    assert spec.work_group_order == "NMajor"
    assert spec.store_schedule == "ElementSerial"
    assert FixedBackwardKernelSpec.from_mapping(spec.to_mapping()) == spec


def test_fixed_backward_writer_is_deterministic_and_inspectable(tmp_path: Path) -> None:
    instance = _instances()[0]
    toolchain = Toolchain.discover()
    writer = writer_for_instance(instance, toolchain)
    assert writer.source() == writer.source()
    assembly = tmp_path / "kernel.s"
    obj = tmp_path / "kernel.o"
    code_object = tmp_path / "kernel.hsaco"
    writer.write(assembly)
    toolchain.assemble(assembly, obj)
    toolchain.link(obj, code_object)
    inspection = inspect_artifact(instance, code_object, toolchain)
    resources = DerivedFixedBackwardState.from_problem_spec(
        *_problem_spec(instance)
    ).physical.resources
    assert inspection.vgpr_count == resources.vgprs
    assert inspection.sgpr_count == resources.sgprs
    assert inspection.lds_num_bytes == resources.lds_bytes
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


def test_fixed_backward_mapping_rejects_unknown_fields() -> None:
    mapping = mapping_for_instance(_instances()[0])
    mapping["Unknown"] = 1
    with pytest.raises(SchemaError):
        parse_instance(mapping)


def test_fixed_backward_problem_axes_are_not_repaired() -> None:
    instance = _instances()[0]
    problem, spec = _problem_spec(instance)
    invalid = type(problem)(
        problem.quant_data_type,
        problem.tokens,
        problem.output_features,
        problem.input_features,
        4,
    )
    with pytest.raises(AssertionError):
        validate_fixed_backward_solution(invalid, spec)


def test_fixed_backward_work_group_mapping_preserves_group_z_and_order() -> None:
    base = _instances()[0]
    _, spec = _problem_spec(base)
    mapped = replace(
        base,
        kernel_spec=replace(
            spec,
            compute=replace(
                spec.compute,
                geometry=replace(spec.compute.geometry, work_group_mapping=2),
            ),
        ),
    )
    validate_fixed_backward_solution(*_problem_spec(mapped))
    assert launch_for_instance(base).grid == (32, 16, 8)
    assert launch_for_instance(mapped).grid == (64, 8, 8)
    source = writer_for_instance(mapped, Toolchain.discover()).source()
    assert "Decode the WGM-packed fixed-group M/N grid." in source
    assert "s_and_b32 s13, s2, 1" in source
    assert "s_lshr_b32 s2, s2, 1" in source

    mapped_spec = _problem_spec(mapped)[1]
    mm_major = replace(
        mapped,
        kernel_spec=replace(mapped_spec, work_group_order="MMajor"),
    )
    validate_fixed_backward_solution(*_problem_spec(mm_major))
    assert launch_for_instance(mm_major).grid == (8, 64, 8)
