from dataclasses import replace
from pathlib import Path

import pytest

from tools.ggtensile.campaign import load_catalog
from tools.ggtensile.family_registry import instance_name
from tools.ggtensile.identity import KernelFamily
from tools.ggtensile.kernel_instance import KernelInstance
from tools.ggtensile.kernel_writer_assembly_mmq_bwd import BackwardKernelWriterAssembly
from tools.ggtensile.mmq_bwd_physical import derive_backward_physical_plan
from tools.ggtensile.mmq_bwd_spec import (
    BackwardIterationPolicy,
    BackwardKernelSpec,
    DerivedBackwardState,
    LdsBuffering,
)
from tools.ggtensile.model import ProblemSize, ProblemType
from tools.ggtensile.schema import SchemaError
from tools.ggtensile.toolchain import Toolchain
from tools.ggtensile.validation import validate_backward_solution, validate_instance

_CONFIG = Path(__file__).resolve().parents[2] / "tools/ggtensile/configs"


def _instances() -> tuple[KernelInstance, ...]:
    return tuple(
        load_catalog(_CONFIG / name).entries[0].instance
        for name in (
            "mmq_bwd_q3_k_catalog.json",
            "mmq_bwd_q4_k_catalog.json",
            "mmq_bwd_q5_k_catalog.json",
            "mmq_bwd_q6_k_catalog.json",
            "mmq_bwd_q8_0_catalog.json",
        )
    )


def _writer(
    instance: KernelInstance, toolchain: Toolchain
) -> BackwardKernelWriterAssembly:
    assert isinstance(instance.problem, ProblemSize)
    assert isinstance(instance.kernel_spec, BackwardKernelSpec)
    return BackwardKernelWriterAssembly(
        instance.problem,
        instance.problem_type.quant_data_type,
        instance.kernel_spec,
        instance_name(instance),
        toolchain,
    )


def test_backward_pipeline_policy_round_trips_canonically() -> None:
    instance = _instances()[1]
    spec = instance.kernel_spec
    assert isinstance(spec, BackwardKernelSpec)
    pipeline = spec.pipeline
    assert pipeline.iteration.prefetch_activation
    assert pipeline.iteration.interleave_wmma_waits
    assert pipeline.prefetch_packed_weight
    assert not pipeline.prefetch_next_packed_weight


def test_q6_compact_pipeline_has_explicit_unswizzled_buffer_state() -> None:
    instance = _instances()[3]
    assert isinstance(instance.problem, ProblemSize)
    spec = instance.kernel_spec
    assert isinstance(spec, BackwardKernelSpec)
    candidate = replace(
        spec,
        pipeline=replace(spec.pipeline, lds_buffering=LdsBuffering.Double),
    )
    state = DerivedBackwardState.from_problem_spec(
        instance.problem, instance.problem_type.quant_data_type, candidate
    )
    physical = derive_backward_physical_plan(state)
    assert physical.resources.vgprs == 77
    assert physical.resources.lds_bytes == 9216
    assert physical.address.pipeline_read_lds is not None

    source = _writer(
        replace(instance, kernel_spec=candidate), Toolchain.discover()
    ).source()
    read_lds = physical.address.pipeline_read_lds
    write_lds = physical.address.lds
    first = physical.registers.quant_dm
    second = first + 1
    assert f"v_mov_b32 v{read_lds}, 0" in source
    assert f"v_add_nc_u32 v{physical.registers.temporary}, v{read_lds}" in source
    assert f"v_xor_b32 v{read_lds}, 4608, v{read_lds}" in source
    assert f"v_xor_b32 v{write_lds}, 4608, v{write_lds}" in source
    assert f"v_add_nc_u32 v{first}, 32, v{physical.registers.temporary}" in source
    assert f"v_xor_b32 v{first}, 32, v{first}" not in source
    assert f"v_xor_b32 v{second}, 16, v{first}" not in source
    assert f"ds_load_b128 v[36:39], v{first} offset:16" in source
    assert "v-1" not in source


def test_backward_kernel_spec_round_trips_pascal_case_mapping() -> None:
    instance = _instances()[0]
    spec = instance.kernel_spec
    assert isinstance(spec, BackwardKernelSpec)
    quant_type = instance.problem_type.quant_data_type
    mapping = spec.to_mapping(quant_type)
    assert set(mapping) == {"Geometry", "Memory", "Pipeline", "Store", "Decode"}
    assert BackwardKernelSpec.from_mapping(mapping, quant_type) == spec
    noncanonical = dict(mapping)
    noncanonical["pipeline"] = mapping["Pipeline"]
    with pytest.raises(SchemaError):
        BackwardKernelSpec.from_mapping(noncanonical, quant_type)


def test_backward_physical_plan_is_derived_from_explicit_values() -> None:
    instance = _instances()[2]
    assert isinstance(instance.problem, ProblemSize)
    assert isinstance(instance.kernel_spec, BackwardKernelSpec)
    state = DerivedBackwardState.from_problem_spec(
        instance.problem,
        instance.problem_type.quant_data_type,
        instance.kernel_spec,
    )
    physical = derive_backward_physical_plan(state)
    assert physical.resources.lds_bytes > 0
    assert physical.resources.vgprs > 0
    assert physical.resources.sgprs > 0


def test_backward_writer_is_deterministic_for_each_catalog_family() -> None:
    toolchain = Toolchain.discover()
    for instance in _instances():
        writer = _writer(instance, toolchain)
        assert writer.source() == _writer(instance, toolchain).source()
        assert instance_name(instance) in writer.source()


def test_backward_validation_rejects_a_forward_spec() -> None:
    forward = load_catalog(_CONFIG / "mmq_fwd_q4_k_catalog.json").entries[0].instance
    invalid = replace(
        forward,
        family=KernelFamily.OrdinaryBackward,
        problem_type=ProblemType.mmq_backward("Q4_K"),
    )
    with pytest.raises(AssertionError):
        validate_instance(invalid)


def test_invalid_backward_iteration_policy_is_reported_by_general_validation() -> None:
    instance = _instances()[1]
    assert isinstance(instance.problem, ProblemSize)
    spec = instance.kernel_spec
    assert isinstance(spec, BackwardKernelSpec)
    invalid = replace(
        spec,
        pipeline=replace(
            spec.pipeline,
            global_read_prefetch=2,
            iteration=BackwardIterationPolicy(False, False),
        ),
    )
    with pytest.raises(AssertionError):
        validate_backward_solution(
            instance.problem, instance.problem_type.quant_data_type, invalid
        )
    with pytest.raises(AssertionError):
        _writer(replace(instance, kernel_spec=invalid), Toolchain.discover())
