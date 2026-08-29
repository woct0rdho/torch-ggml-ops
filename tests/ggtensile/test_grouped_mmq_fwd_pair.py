from dataclasses import replace
from pathlib import Path

import pytest

from tools.ggtensile.campaign import load_catalog
from tools.ggtensile.family_registry import (
    instance_name,
    mapping_for_instance,
    parse_instance,
    writer_for_instance,
)
from tools.ggtensile.grouped_mmq_fwd_pair_inspection import (
    inspect_grouped_forward_pair_artifact,
)
from tools.ggtensile.grouped_mmq_fwd_pair_model import (
    GroupedForwardPairProblem,
    GroupedPairRouteOwnership,
    GroupedPairWeightStaging,
)
from tools.ggtensile.grouped_mmq_fwd_pair_spec import (
    DerivedGroupedForwardPairState,
    GroupedForwardPairKernelSpec,
)
from tools.ggtensile.grouped_mmq_fwd_pair_validation import (
    validate_grouped_forward_pair_solution,
)
from tools.ggtensile.kernel_instance import KernelInstance
from tools.ggtensile.model import SchemaError
from tools.ggtensile.toolchain import Toolchain

_CONFIG = Path(__file__).resolve().parents[2] / "tools/ggtensile/configs"


def _problem_spec(
    instance: KernelInstance,
) -> tuple[GroupedForwardPairProblem, GroupedForwardPairKernelSpec]:
    assert isinstance(instance.problem, GroupedForwardPairProblem)
    assert isinstance(instance.kernel_spec, GroupedForwardPairKernelSpec)
    return instance.problem, instance.kernel_spec


def _lowering_mapping(instance: KernelInstance) -> dict[str, object]:
    mapping = mapping_for_instance(instance)
    kernel_spec = mapping["KernelSpec"]
    assert isinstance(kernel_spec, dict)
    lowering = kernel_spec["Lowering"]
    assert isinstance(lowering, dict)
    return lowering


def _catalog_instances() -> tuple[KernelInstance, ...]:
    instances = []
    for path in sorted(_CONFIG.glob("mmq_grouped_fwd_pair_*_catalog.json")):
        catalog = load_catalog(path)
        for entry in catalog.entries:
            instances.append(entry.instance)
    return tuple(instances)


def test_grouped_pair_catalog_keys_are_typed_and_round_trip() -> None:
    instances = _catalog_instances()
    assert instances
    for instance in instances:
        assert parse_instance(mapping_for_instance(instance)) == instance
        problem, spec = _problem_spec(instance)
        validate_grouped_forward_pair_solution(problem, spec)
        state = DerivedGroupedForwardPairState.from_problem_spec(problem, spec)
        assert state.expected_output_shape == (
            problem.aggregate_rows,
            problem.output_features,
        )
        assert state.kernel_spec.geometry.depth_u == 128
        assert state.physical_plan.resources.private_bytes == 0


def test_grouped_pair_route_ownership_is_spec_data() -> None:
    for instance in _catalog_instances():
        _, spec = _problem_spec(instance)
        mapping = mapping_for_instance(instance)
        problem = mapping["Problem"]
        lowering = _lowering_mapping(instance)
        assert isinstance(problem, dict)
        assert "RouteOwnership" not in problem
        assert lowering["RouteOwnership"] == spec.route_ownership.value
        if spec.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks:
            assert spec.row_task_rows is not None


def test_grouped_pair_capability_rejects_a_cross_format_weight_staging() -> None:
    for instance in _catalog_instances():
        problem, spec = _problem_spec(instance)
        for source in GroupedPairWeightStaging:
            if source is spec.weight_staging:
                continue
            invalid_spec = replace(spec, weight_staging=source)
            with pytest.raises(AssertionError):
                validate_grouped_forward_pair_solution(problem, invalid_spec)


def test_grouped_pair_writer_sources_are_deterministic() -> None:
    toolchain = Toolchain.discover()
    for instance in _catalog_instances():
        writer = writer_for_instance(instance, toolchain)
        source = writer.source()
        assert source == writer.source()
        assert instance_name(instance) in source
        assert ".amdhsa_kernel" in source
        assert "s_barrier" in source


def test_grouped_pair_inspection_follows_the_physical_plan(tmp_path: Path) -> None:
    instance = _catalog_instances()[0]
    problem, spec = _problem_spec(instance)
    toolchain = Toolchain.discover()
    assembly = tmp_path / "kernel.s"
    obj = tmp_path / "kernel.o"
    code_object = tmp_path / "kernel.hsaco"
    writer_for_instance(instance, toolchain).write(assembly)
    toolchain.assemble(assembly, obj)
    toolchain.link(obj, code_object)
    inspection = inspect_grouped_forward_pair_artifact(
        problem, spec, instance_name(instance), code_object, toolchain
    )
    resources = DerivedGroupedForwardPairState.from_problem_spec(
        problem, spec
    ).physical_plan.resources
    assert inspection.vgpr_count == resources.vgprs
    assert inspection.sgpr_count == resources.sgprs
    assert inspection.lds_num_bytes == resources.lds_bytes
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


def test_grouped_pair_schema_rejects_unknown_root_field() -> None:
    mapping = mapping_for_instance(_catalog_instances()[0])
    mapping["Unknown"] = 1
    with pytest.raises(SchemaError):
        parse_instance(mapping)


def test_grouped_pair_schema_rejects_missing_row_task_dimension() -> None:
    instance = next(
        instance
        for instance in _catalog_instances()
        if _problem_spec(instance)[1].route_ownership
        is GroupedPairRouteOwnership.DeviceRowTasks
    )
    mapping = mapping_for_instance(instance)
    kernel_spec = mapping["KernelSpec"]
    assert isinstance(kernel_spec, dict)
    lowering = kernel_spec["Lowering"]
    assert isinstance(lowering, dict)
    lowering.pop("RowTaskRows")
    with pytest.raises(SchemaError):
        parse_instance(mapping)
