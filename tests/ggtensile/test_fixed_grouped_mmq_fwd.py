from pathlib import Path

import pytest

from tools.ggtensile.campaign import load_catalog
from tools.ggtensile.family_registry import (
    mapping_for_instance,
    parse_instance,
    writer_for_instance,
)
from tools.ggtensile.fixed_grouped_mmq_fwd_model import FixedForwardProblem
from tools.ggtensile.fixed_grouped_mmq_fwd_spec import (
    DerivedFixedForwardState,
    FixedForwardKernelSpec,
)
from tools.ggtensile.fixed_grouped_mmq_fwd_validation import (
    validate_fixed_forward_solution,
)
from tools.ggtensile.inspection import inspect_artifact
from tools.ggtensile.kernel_instance import KernelInstance
from tools.ggtensile.schema import SchemaError
from tools.ggtensile.toolchain import Toolchain

_CONFIG = Path(__file__).resolve().parents[2] / "tools/ggtensile/configs"


def _instances() -> tuple[KernelInstance, ...]:
    catalog = load_catalog(_CONFIG / "mmq_fixed_grouped_fwd_q8_0_catalog.json")
    return tuple(entry.instance for entry in catalog.entries)


def test_fixed_forward_catalog_keys_round_trip_and_derive() -> None:
    instances = _instances()
    assert len(instances) == 3
    for instance in instances:
        assert parse_instance(mapping_for_instance(instance)) == instance
        assert isinstance(instance.problem, FixedForwardProblem)
        assert isinstance(instance.kernel_spec, FixedForwardKernelSpec)
        validate_fixed_forward_solution(instance.problem, instance.kernel_spec)
        state = DerivedFixedForwardState.from_problem_spec(
            instance.problem, instance.kernel_spec
        )
        assert state.expected_packed_weight_shape[0] == 8
        assert (
            state.expected_activation_shape[1] == instance.problem.total_activation_rows
        )
        assert state.expected_output_shape == (
            instance.problem.tokens,
            instance.problem.groups,
            instance.problem.output_features,
        )
        assert state.physical.resources.private_bytes == 0


def test_fixed_forward_spec_owns_geometry_and_addressing() -> None:
    instance = _instances()[0]
    spec = instance.kernel_spec
    assert isinstance(spec, FixedForwardKernelSpec)
    assert FixedForwardKernelSpec.from_mapping(spec.to_mapping()) == spec
    assert spec.weight_staging.value == "Q8SmallMTiledLds"
    assert spec.depth_u == 32
    assert spec.macro_tile_tokens > 0
    assert spec.macro_tile_features > 0


def test_fixed_forward_writer_is_deterministic_and_inspectable(tmp_path: Path) -> None:
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
    problem = instance.problem
    spec = instance.kernel_spec
    assert isinstance(problem, FixedForwardProblem)
    assert isinstance(spec, FixedForwardKernelSpec)
    resources = DerivedFixedForwardState.from_problem_spec(
        problem, spec
    ).physical.resources
    assert inspection.vgpr_count == resources.vgprs
    assert inspection.sgpr_count == resources.sgprs
    assert inspection.lds_num_bytes == resources.lds_bytes
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


def test_fixed_forward_mapping_rejects_unknown_fields() -> None:
    mapping = mapping_for_instance(_instances()[0])
    mapping["Unknown"] = 1
    with pytest.raises(SchemaError):
        parse_instance(mapping)


def test_fixed_forward_problem_axes_are_not_repaired() -> None:
    instance = _instances()[0]
    problem = instance.problem
    assert isinstance(problem, FixedForwardProblem)
    invalid = type(problem)(
        problem.quant_data_type,
        32,
        problem.output_features,
        problem.input_features,
        problem.groups,
    )
    with pytest.raises(AssertionError):
        assert isinstance(instance.kernel_spec, FixedForwardKernelSpec)
        validate_fixed_forward_solution(invalid, instance.kernel_spec)
