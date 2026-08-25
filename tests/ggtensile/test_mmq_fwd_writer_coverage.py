from dataclasses import replace

import pytest

from tests.ggtensile.ordinary_forward_fixtures import (
    q3_full_weight_tiled_lds_kernel_spec,
    q3_hip_tiled_lds_kernel_spec,
    q4_decoded_weight_lds_kernel_spec,
    q5_decoded_weight_lds_kernel_spec,
    q6_structured_kernel_spec,
    q8_direct_global_kernel_spec,
    q8_hip_tiled_lds_kernel_spec,
    q8_small_m_tiled_lds_kernel_spec,
)
from tests.ggtensile.support import ordinary_forward_instance
from tools.ggtensile.family_registry import instance_name, mapping_for_instance
from tools.ggtensile.inspection import _forward_static_wmma_count
from tools.ggtensile.kernel_writer_assembly_mmq_fwd import ForwardKernelWriterAssembly
from tools.ggtensile.mmq_fwd_physical import derive_forward_physical_plan
from tools.ggtensile.mmq_fwd_search import (
    q6_kernel_spec_with_schedule,
    q6_schedule_from_kernel_spec,
)
from tools.ggtensile.mmq_fwd_spec import (
    DecodedLdsForwardDecodePolicy,
    DerivedForwardState,
    ForwardKernelSpec,
    SemanticSchedulePolicy,
)
from tools.ggtensile.model import ProblemSize
from tools.ggtensile.toolchain import Toolchain
from tools.ggtensile.validation import validate_forward_solution


def _writer(
    quant_type: str,
    size: ProblemSize,
    spec: ForwardKernelSpec,
    toolchain: Toolchain,
) -> ForwardKernelWriterAssembly:
    instance = ordinary_forward_instance(quant_type, size, spec)
    return ForwardKernelWriterAssembly(
        size, quant_type, spec, instance_name(instance), toolchain
    )


def test_writer_dispatches_each_typed_forward_mechanism() -> None:
    cases = (
        ("Q3_K", ProblemSize(2048, 2048, 2048), q3_hip_tiled_lds_kernel_spec(), "Q3_K"),
        (
            "Q3_K",
            ProblemSize(32768, 4096, 2048),
            q3_full_weight_tiled_lds_kernel_spec(),
            "Q3_K",
        ),
        (
            "Q4_K",
            ProblemSize(8192, 2048, 512),
            q4_decoded_weight_lds_kernel_spec(
                epilogue_tiles_ahead=1, epilogue_dependency_width=1, epilogue_priority=0
            ),
            "Q4_K",
        ),
        (
            "Q5_K",
            ProblemSize(8192, 2048, 512),
            q5_decoded_weight_lds_kernel_spec(
                epilogue_tiles_ahead=1, epilogue_dependency_width=2, epilogue_priority=2
            ),
            "Q5_K",
        ),
        (
            "Q6_K",
            ProblemSize(128, 248320, 2048),
            q6_structured_kernel_spec(128),
            "Q6_K",
        ),
        ("Q8_0", ProblemSize(2048, 1024, 4096), q8_direct_global_kernel_spec(), "Q8_0"),
        (
            "Q8_0",
            ProblemSize(128, 129280, 4096),
            q8_hip_tiled_lds_kernel_spec(),
            "Q8_0",
        ),
        (
            "Q8_0",
            ProblemSize(64, 129280, 4096),
            q8_small_m_tiled_lds_kernel_spec(64),
            "Q8_0",
        ),
    )
    toolchain = Toolchain.discover()
    for quant_type, size, spec, marker in cases:
        instance = ordinary_forward_instance(quant_type, size, spec)
        source = _writer(quant_type, size, spec, toolchain).source()
        assert marker in source
        assert instance_name(instance) in source
        assert ".amdhsa_kernel" in source


def test_writer_source_is_deterministic() -> None:
    spec = q4_decoded_weight_lds_kernel_spec(
        epilogue_tiles_ahead=2, epilogue_dependency_width=4, epilogue_priority=2
    )
    writer = _writer("Q4_K", ProblemSize(8192, 2048, 512), spec, Toolchain.discover())
    assert writer.source() == writer.source()


def test_validation_rejects_dataflow_mutation() -> None:
    spec = q4_decoded_weight_lds_kernel_spec(
        epilogue_tiles_ahead=1, epilogue_dependency_width=1, epilogue_priority=0
    )
    invalid = replace(
        spec,
        global_memory=replace(spec.global_memory, activation_addressing="Unknown"),
    )
    size = ProblemSize(8192, 2048, 512)
    with pytest.raises(AssertionError):
        validate_forward_solution(size, "Q4_K", invalid)
    with pytest.raises(AssertionError):
        _writer("Q4_K", size, invalid, Toolchain.discover())


def test_decoded_policy_fields_are_active() -> None:
    spec = q4_decoded_weight_lds_kernel_spec(
        epilogue_tiles_ahead=1, epilogue_dependency_width=1, epilogue_priority=0
    )
    changed = replace(
        spec,
        decode=replace(
            spec.decode,
            policy=DecodedLdsForwardDecodePolicy(False, True),
        ),
    )
    assert changed != spec
    size = ProblemSize(8192, 2048, 512)
    validate_forward_solution(size, "Q4_K", changed)
    instance = ordinary_forward_instance("Q4_K", size, changed)
    kernel_spec_mapping = mapping_for_instance(instance)["KernelSpec"]
    assert isinstance(kernel_spec_mapping, dict)
    decode_mapping = kernel_spec_mapping["Decode"]
    assert isinstance(decode_mapping, dict)
    assert "DeferMetadataReads" in decode_mapping


def test_q6_schedule_mutation_flows_through_writer() -> None:
    base = q6_structured_kernel_spec(64)
    schedule = replace(
        q6_schedule_from_kernel_spec(base),
        semantic_policy=SemanticSchedulePolicy.structured_q6_wavefront(),
        global_read_cache_policy="Default",
    )
    spec = q6_kernel_spec_with_schedule(base, schedule)
    size = ProblemSize(64, 248320, 2048)
    validate_forward_solution(size, "Q6_K", spec)
    source = _writer("Q6_K", size, spec, Toolchain.discover()).source()
    assert "buffer_gl0_inv" not in source
    assert spec.semantic_schedule == schedule.semantic_policy


def test_small_m_inspection_wmma_count_follows_typed_layout() -> None:
    toolchain = Toolchain.discover()
    for macro_tile_m in (32, 64):
        spec = q8_small_m_tiled_lds_kernel_spec(macro_tile_m)
        size = ProblemSize(macro_tile_m, 129280, 4096)
        source = _writer("Q8_0", size, spec, toolchain).source()
        state = DerivedForwardState.from_problem_spec(size, "Q8_0", spec)
        assert source.count("v_wmma_i32_16x16x16_iu8") == macro_tile_m // 2
        assert _forward_static_wmma_count(state) == macro_tile_m // 2


def test_physical_plan_is_the_resource_authority() -> None:
    cases = (
        ("Q3_K", q3_hip_tiled_lds_kernel_spec()),
        (
            "Q4_K",
            q4_decoded_weight_lds_kernel_spec(
                epilogue_tiles_ahead=1, epilogue_dependency_width=1, epilogue_priority=0
            ),
        ),
        ("Q6_K", q6_structured_kernel_spec(64)),
        ("Q8_0", q8_direct_global_kernel_spec()),
    )
    for quant_type, spec in cases:
        plan = derive_forward_physical_plan(spec)
        assert plan.resources.private_bytes == 0
        assert plan.resources.vgprs > 0
        assert plan.resources.sgprs > 0
