"""Targeted branch and complete executable-line coverage for the backward writer."""

import copy
from dataclasses import asdict, fields, replace

import pytest

from tests.ggtensile.support import (
    BWD_IMPLEMENTATION_SOURCE_PATHS,
    BWD_WRITER_SOURCE_PATH,
    MMQ_BWD_INVENTORY_CASE_IDS,
    MMQ_BWD_INVENTORY_CASES,
    GGTensileInventoryCase,
    assert_writer_methods_have_complete_line_coverage,
    load_inventory_case,
    selected_solution_keys,
)
from tools.ggtensile import kernel_writer_assembly_mmq_bwd as bwd_writer_module
from tools.ggtensile.kernel_writer_assembly_mmq_bwd import (
    BackwardDiagnosticMode,
    BackwardKernelWriterAssembly,
    BackwardKernelWriterError,
)
from tools.ggtensile.mmq_bwd_emission import _Assembly
from tools.ggtensile.mmq_bwd_physical import (
    _FirstFitRegisters,
    derive_backward_physical_plan,
)
from tools.ggtensile.mmq_bwd_spec import (
    BackwardExtraction,
    BackwardKernelSpec,
    BackwardPipelineSpec,
    BackwardProblemContract,
    BackwardQ3Pairing,
    BackwardQ4DecodeSchedule,
    DerivedBackwardState,
    backward_mechanism_contract,
)
from tools.ggtensile.model import (
    BackwardSolution,
    ForwardSolution,
    ProblemSize,
    ProblemType,
    SchemaError,
    SolutionKey,
)
from tools.ggtensile.toolchain import Toolchain
from tools.ggtensile.validation import validate_solution


def _key(
    quant: str, size: tuple[int, int, int], solution: BackwardSolution
) -> SolutionKey:
    return SolutionKey(
        ProblemType.mmq_backward(quant),
        ProblemSize(*size),
        solution,
    )


def test_backward_pipeline_policy_rejects_unparseable_modes() -> None:
    pilot = BackwardSolution.pilot()
    assert BackwardPipelineSpec.try_from_solution(pilot) is not None
    assert (
        BackwardPipelineSpec.try_from_solution(replace(pilot, schedule_iter_alg=99))
        is None
    )
    assert (
        BackwardPipelineSpec.try_from_solution(replace(pilot, one_lds_buffer=99))
        is None
    )
    assert (
        BackwardPipelineSpec.from_solution(
            replace(
                pilot, prefetch_packed_weight=False, prefetch_packed_weight_next=True
            )
        ).packed_weight_prefetch
        is not None
    )
    assert BackwardQ3Pairing.try_from_serialized("invalid") is None
    assert BackwardQ4DecodeSchedule.try_from_serialized("invalid") is None
    assert BackwardExtraction.try_from_serialized("invalid") is None


def test_backward_canonical_schema_rejects_every_noncanonical_boundary() -> None:
    size = ProblemSize(128, 256, 128)
    pilot = BackwardSolution.pilot()
    key = SolutionKey(ProblemType.mmq_backward("Q4_K"), size, pilot)
    contract = BackwardProblemContract.from_solution_key(key)
    spec = BackwardKernelSpec.from_solution(pilot)
    mapping = spec.to_mapping("Q4_K")

    invalid_contract = contract.to_mapping()
    invalid_contract["quant_type"] = "Q2_K"
    with pytest.raises(SchemaError, match="unsupported backward quant"):
        BackwardProblemContract.from_mapping(invalid_contract, size)

    invalid_contract = contract.to_mapping()
    invalid_contract["activation_type"] = "Float16"
    with pytest.raises(SchemaError, match="not canonical"):
        BackwardProblemContract.from_mapping(invalid_contract, size)

    with pytest.raises(ValueError, match="not canonically representable"):
        BackwardKernelSpec.from_solution(
            replace(pilot, lds_pad_b=8, lds_swizzle_chunk_b=8)
        ).to_mapping("Q4_K")
    with pytest.raises(ValueError, match="unsupported backward quant type"):
        spec.to_mapping("Q2_K")

    invalid = copy.deepcopy(mapping)
    invalid["decode"] = {"extraction": "packed"}
    with pytest.raises(SchemaError, match="invalid BackwardKernelSpec.decode"):
        BackwardKernelSpec.from_mapping(invalid, "Q4_K")

    invalid = copy.deepcopy(mapping)
    invalid["decode"] = {"schedule": "Serial"}
    with pytest.raises(SchemaError, match="canonically represented by absent"):
        BackwardKernelSpec.from_mapping(invalid, "Q4_K")

    batched = replace(pilot, q4_k_decode_schedule="DependencyBatch4")
    batched_mapping = BackwardKernelSpec.from_solution(batched).to_mapping("Q4_K")
    assert batched_mapping["decode"] == {"schedule": "DependencyBatch4"}
    assert BackwardKernelSpec.from_mapping(batched_mapping, "Q4_K") == (
        BackwardKernelSpec.from_solution(batched)
    )

    invalid = copy.deepcopy(mapping)
    invalid["geometry"]["work_group"] = [0, 4, 1]
    with pytest.raises(SchemaError, match="must be positive"):
        BackwardKernelSpec.from_mapping(invalid, "Q4_K")

    invalid = copy.deepcopy(mapping)
    invalid["geometry"]["work_group"] = [33, 1, 1]
    with pytest.raises(SchemaError, match="whole wave32"):
        BackwardKernelSpec.from_mapping(invalid, "Q4_K")

    invalid = copy.deepcopy(mapping)
    invalid["memory"]["lds_layout"] = "Padded"
    with pytest.raises(SchemaError, match="lds_layout must be a mapping"):
        BackwardKernelSpec.from_mapping(invalid, "Q4_K")

    invalid = copy.deepcopy(mapping)
    invalid["memory"]["lds_layout"] = {"kind": "Unknown"}
    with pytest.raises(SchemaError, match="invalid backward LDS layout kind"):
        BackwardKernelSpec.from_mapping(invalid, "Q4_K")

    invalid = copy.deepcopy(mapping)
    invalid["pipeline"]["schedule"] = "SIA99"
    with pytest.raises(SchemaError, match="invalid schedule"):
        BackwardKernelSpec.from_mapping(invalid, "Q4_K")

    q3 = replace(pilot, q3_k_pairing="Full")
    invalid = BackwardKernelSpec.from_solution(q3).to_mapping("Q3_K")
    invalid["decode"]["pairing"] = "Inactive"
    with pytest.raises(SchemaError, match="must be an active policy"):
        BackwardKernelSpec.from_mapping(invalid, "Q3_K")

    with pytest.raises(SchemaError, match="unsupported backward quant type"):
        BackwardKernelSpec.from_mapping(
            BackwardKernelSpec.from_solution(q3).to_mapping("Q3_K"), "Q2_K"
        )

    invalid_spec = replace(spec, geometry=replace(spec.geometry, isa=(11, 0, 0)))
    with pytest.raises(ValueError, match="ISA and wavefront"):
        invalid_spec.to_solution(contract)


def test_backward_pipeline_validation_keeps_field_errors_independent() -> None:
    pilot = BackwardSolution.pilot()
    invalid_schedule = _key(
        "Q4_K", (128, 256, 128), replace(pilot, schedule_iter_alg=99)
    )
    schedule_rules = {reason.rule_id for reason in validate_solution(invalid_schedule)}
    assert "solution.scheduleiteralg.unimplemented" in schedule_rules
    assert "solution.1ldsbuffer.unimplemented" not in schedule_rules

    invalid_lds = _key("Q4_K", (128, 256, 128), replace(pilot, one_lds_buffer=99))
    lds_rules = {reason.rule_id for reason in validate_solution(invalid_lds)}
    assert "solution.1ldsbuffer.unimplemented" in lds_rules
    assert "solution.scheduleiteralg.unimplemented" not in lds_rules

    inactive_q4 = _key(
        "Q8_0", (128, 256, 128), replace(pilot, q4_k_decode_schedule="DependencyBatch4")
    )
    inactive_q4_rules = {reason.rule_id for reason in validate_solution(inactive_q4)}
    assert "solution.q4.controls.inert" in inactive_q4_rules


@pytest.mark.parametrize(
    "case",
    MMQ_BWD_INVENTORY_CASES,
    ids=MMQ_BWD_INVENTORY_CASE_IDS,
)
def test_writer_emits_every_selected_production_kernel(
    case: GGTensileInventoryCase,
) -> None:
    catalog = load_inventory_case(case)
    emitted: set[str] = set()
    for key in selected_solution_keys(catalog):
        source = BackwardKernelWriterAssembly(key, Toolchain.discover()).source()
        assert f".globl {key.kernel_name}" in source
        emitted.add(key.hash)
    assert len(emitted) == case.entry_count


def _targeted_writer_keys() -> tuple[SolutionKey, ...]:
    pilot = BackwardSolution.pilot()
    q5_sia3 = replace(
        pilot,
        one_lds_buffer=0,
        schedule_iter_alg=3,
        prefetch_global_read=1,
        lds_swizzle_chunk_b=8,
    )
    q5_depth64_pipeline = replace(
        pilot,
        depth_u=64,
        one_lds_buffer=0,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
    )
    q6_256x64_share = replace(
        pilot,
        matrix_instruction=(16, 16, 16, 1, 1, 4, 4, 4, 1),
        macro_tile0=256,
        macro_tile1=64,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        packed_weight_lane_share=2,
        lds_swizzle_chunk_b=8,
    )
    q6_64x64_depth64 = replace(
        pilot,
        matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
        macro_tile0=64,
        macro_tile1=64,
        depth_u=64,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
    )
    q6_128x64_pipeline = replace(
        pilot,
        matrix_instruction=(16, 16, 16, 1, 1, 2, 4, 4, 1),
        macro_tile1=64,
        one_lds_buffer=0,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
    )
    return (
        _key("Q3_K", (128, 2048, 512), replace(pilot, q3_k_extraction="scalar")),
        _key("Q5_K", (128, 2048, 512), replace(pilot, q5_k_extraction="scalar")),
        _key(
            "Q5_K",
            (128, 2048, 512),
            replace(pilot, q5_k_metadata_vector_load=True),
        ),
        _key(
            "Q5_K",
            (128, 2048, 512),
            replace(pilot, q5_k_nibble_shift_hoist=True),
        ),
        _key(
            "Q5_K",
            (128, 2048, 512),
            replace(pilot, packed_weight_lane_share=2),
        ),
        _key("Q5_K", (128, 2048, 512), q5_sia3),
        _key("Q5_K", (128, 2048, 512), q5_depth64_pipeline),
        _key("Q6_K", (128, 2048, 248320), replace(pilot, q6_k_extraction="scalar")),
        _key(
            "Q6_K",
            (128, 2048, 248320),
            replace(pilot, q6_k_extraction="packed_vopd"),
        ),
        _key("Q6_K", (256, 2048, 248320), q6_256x64_share),
        _key("Q6_K", (64, 2048, 248320), q6_64x64_depth64),
        _key("Q6_K", (128, 2048, 248320), q6_128x64_pipeline),
        _key("Q8_0", (128, 4096, 1024), replace(pilot, q8_0_extraction="scalar")),
        _key(
            "Q8_0",
            (128, 4096, 1024),
            replace(
                pilot,
                q8_0_extraction="scalar",
                one_lds_buffer=0,
                schedule_iter_alg=4,
                prefetch_global_read=2,
                lds_swizzle_chunk_b=8,
            ),
        ),
        _key(
            "Q3_K",
            (128, 2048, 512),
            replace(
                pilot,
                one_lds_buffer=0,
                schedule_iter_alg=4,
                prefetch_global_read=2,
                lds_swizzle_chunk_b=8,
                q3_k_pairing="Partial",
            ),
        ),
        _key("Q3_K", (128, 2048, 8192), replace(pilot, q3_k_pairing="Full")),
        _key(
            "Q4_K",
            (128, 2048, 512),
            replace(
                pilot,
                depth_u=64,
                schedule_iter_alg=4,
                prefetch_global_read=2,
                lds_swizzle_chunk_b=8,
            ),
        ),
        _key(
            "Q5_K",
            (128, 2048, 512),
            replace(
                pilot,
                depth_u=64,
                schedule_iter_alg=4,
                prefetch_global_read=2,
                lds_swizzle_chunk_b=8,
            ),
        ),
        _key(
            "Q6_K",
            (64, 2048, 248320),
            replace(q6_64x64_depth64, prefetch_packed_weight_next=True),
        ),
        _key(
            "Q6_K",
            (64, 2048, 248320),
            replace(
                pilot,
                matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
                macro_tile0=64,
                macro_tile1=64,
            ),
        ),
        _key(
            "Q4_K",
            (256, 2048, 512),
            replace(
                pilot,
                work_group=(32, 8, 1),
                matrix_instruction=(16, 16, 16, 1, 1, 2, 8, 8, 1),
                macro_tile0=256,
            ),
        ),
        _key(
            "Q4_K",
            (256, 2048, 512),
            replace(
                pilot,
                matrix_instruction=(16, 16, 16, 1, 1, 4, 4, 4, 1),
                macro_tile0=256,
                macro_tile1=64,
            ),
        ),
    )


@pytest.mark.parametrize("key", _targeted_writer_keys(), ids=lambda key: key.hash)
def test_writer_emits_targeted_quant_and_pipeline_paths(key: SolutionKey) -> None:
    source = BackwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert source.endswith(
        f".size {key.kernel_name}, .L{key.kernel_name}_end - {key.kernel_name}\n"
    )


def test_writer_emits_repaired_q6_lane_share_and_depth64_layouts() -> None:
    share_key = _targeted_writer_keys()[9]
    source = BackwardKernelWriterAssembly(share_key, Toolchain.discover()).source()
    assert source.count("v_mov_b32_dpp") == 4
    assert source.index("Load unique Q6_K low planes on every lane.") < source.index(
        "Load shared Q6_K high planes on lane-pair owners."
    )
    assert source.index(
        "Load shared Q6_K high planes on lane-pair owners."
    ) < source.index("s_and_saveexec_b32")

    depth64_key = _targeted_writer_keys()[10]
    writer = BackwardKernelWriterAssembly(depth64_key, Toolchain.discover())
    row_stride = 2 * depth64_key.solution.depth_u
    plan = writer.physical
    assert (
        plan.lds.decoded_store_location(plan.registers, plan.address, 0, 1, 32)[1] == 64
    )
    assert (
        plan.lds.decoded_store_location(plan.registers, plan.address, 2, 1, 32)[1]
        == 2 * row_stride + 64
    )


def test_writer_emits_repaired_q5_sia3_and_depth64_pipeline_state() -> None:
    sia3_source = BackwardKernelWriterAssembly(
        _targeted_writer_keys()[5], Toolchain.discover()
    ).source()
    assert "s_sub_u32 s5, s5, 32" in sia3_source
    assert "s_add_u32 s5, s5, 32" in sia3_source

    depth64_source = BackwardKernelWriterAssembly(
        _targeted_writer_keys()[6], Toolchain.discover()
    ).source()
    assert (
        "Restore current-tile A pointers after next-B address setup." in depth64_source
    )
    assert "Reload next-tile A1 after current A1 becomes dead." in depth64_source
    assert depth64_source.count("v_add_nc_u32 v208, 64, v208") >= 2


def test_writer_emits_both_lower_bound_diagnostics() -> None:
    key = _key("Q4_K", (128, 2048, 512), BackwardSolution.pilot())
    wmma = BackwardKernelWriterAssembly(
        key,
        Toolchain.discover(),
        diagnostic_mode=BackwardDiagnosticMode.WMMA_FLOOR,
    ).source()
    decode = BackwardKernelWriterAssembly(
        key,
        Toolchain.discover(),
        diagnostic_mode=BackwardDiagnosticMode.DECODE_FLOOR,
    ).source()
    assert ".LWmmaFloorLoop:" in wmma
    assert ".LDecodeFloorLoop:" in decode


def test_writer_emits_non_direct_multirow_decode_floors() -> None:
    keys = _targeted_writer_keys()
    for key in (keys[16], keys[17]):
        source = BackwardKernelWriterAssembly(
            key,
            Toolchain.discover(),
            diagnostic_mode=BackwardDiagnosticMode.DECODE_FLOOR,
        ).source()
        assert ".LDecodeFloorLoop:" in source
        assert source.count("global_load_d16_u8") >= 12


def test_assembly_rejects_nested_deferred_zero_fills() -> None:
    assembly = _Assembly()
    assembly.defer_zero_moves(range(2))
    with pytest.raises(BackwardKernelWriterError, match="cannot nest"):
        assembly.defer_zero_moves(range(2, 4))


def test_depth64_next_packed_prefetch_remains_q6_specific() -> None:
    q4 = _key(
        "Q4_K",
        (128, 2048, 512),
        replace(
            BackwardSolution.pilot(),
            depth_u=64,
            schedule_iter_alg=4,
            prefetch_global_read=2,
            prefetch_packed_weight_next=True,
            lds_swizzle_chunk_b=8,
        ),
    )
    rule_ids = {reason.rule_id for reason in validate_solution(q4)}
    assert "solution.depthu64.prefetchpacked.quant" in rule_ids


def test_writer_rejects_invalid_solution() -> None:
    invalid = _key(
        "Q4_K",
        (128, 2048, 512),
        replace(BackwardSolution.pilot(), depth_u=48),
    )
    with pytest.raises(BackwardKernelWriterError, match="solution rejected"):
        BackwardKernelWriterAssembly(invalid, Toolchain.discover())


def test_writer_rejects_forward_solution_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(128, 2048, 512),
        ForwardSolution.q4_k_pilot(),
    )
    monkeypatch.setattr(bwd_writer_module, "validate_solution", lambda _: ())
    with pytest.raises(BackwardKernelWriterError, match="requires BackwardSolution"):
        BackwardKernelWriterAssembly(key, Toolchain.discover())
    with pytest.raises(TypeError, match="requires BackwardSolution"):
        DerivedBackwardState.from_solution_key(key)


def test_physical_plan_covers_allocator_and_periodic_lds_padding() -> None:
    with pytest.raises(ValueError, match="unknown backward quant mechanism"):
        backward_mechanism_contract("Q2_K")

    allocator = _FirstFitRegisters(0, 0)
    assert allocator.allocate(1) == 0
    with pytest.raises(ValueError, match="cannot allocate"):
        allocator.allocate(1)

    key = _targeted_writer_keys()[0]
    assert isinstance(key.solution, BackwardSolution)
    solution = replace(
        key.solution,
        lds_pad_b=1,
        lds_block_size_per_pad_b=128,
    )
    padded_key = replace(key, solution=solution)
    plan = derive_backward_physical_plan(
        DerivedBackwardState.from_solution_key(padded_key)
    )
    bytes_unpadded = 2 * solution.macro_tile1 * solution.depth_u
    single_buffer = bytes_unpadded + 2 * (bytes_unpadded // 128)
    expected = single_buffer * (2 if solution.one_lds_buffer == 0 else 1)
    assert plan.resources.lds_num_bytes == expected


def test_backward_solution_fields_project_into_kernel_spec_or_reject() -> None:
    baseline = BackwardSolution.pilot()
    alternatives: dict[str, object] = {
        "kernel_language": "HIP",
        "isa": (11, 5, 0),
        "wavefront_size": 64,
        "work_group": (32, 2, 1),
        "matrix_instruction": (16, 16, 16, 1, 1, 1, 4, 2, 1),
        "macro_tile0": 32,
        "macro_tile1": 64,
        "depth_u": 64,
        "global_read_vector_width_a": 8,
        "global_read_vector_width_b": 8,
        "local_read_vector_width": 8,
        "prefetch_global_read": 2,
        "prefetch_local_read": 2,
        "one_lds_buffer": 0,
        "schedule_iter_alg": 3,
        "store_priority_opt": False,
        "num_elements_per_batch_store": 4,
        "store_vector_width": 2,
        "work_group_mapping": 2,
        "transpose_lds": 1,
        "lds_pad_b": 8,
        "lds_block_size_per_pad_b": 128,
        "lds_swizzle_chunk_b": 8,
        "decoder_width": 8,
        "prefetch_packed_weight": False,
        "prefetch_packed_weight_next": True,
        "packed_weight_lane_share": 2,
        "q3_k_extraction": "scalar",
        "q3_k_pairing": "Partial",
        "q4_k_decode_schedule": "DependencyBatch4",
        "q5_k_extraction": "scalar",
        "q5_k_nibble_shift_hoist": True,
        "q5_k_metadata_vector_load": True,
        "q6_k_extraction": "scalar",
        "q8_0_extraction": "scalar",
    }
    assert set(alternatives) == {field.name for field in fields(BackwardSolution)}

    baseline_spec = asdict(BackwardKernelSpec.from_solution(baseline))
    for field, value in alternatives.items():
        candidate = replace(baseline, **{field: value})
        projection = asdict(BackwardKernelSpec.from_solution(candidate))
        if field == "kernel_language":
            assert projection == baseline_spec
            reasons = validate_solution(
                SolutionKey(
                    ProblemType.mmq_backward("Q4_K"),
                    ProblemSize(128, 256, 128),
                    candidate,
                )
            )
            assert "solution.kernellanguage.unimplemented" in {
                reason.rule_id for reason in reasons
            }
        else:
            assert projection != baseline_spec, field


def test_writer_methods_have_complete_line_coverage() -> None:
    for source_path in BWD_IMPLEMENTATION_SOURCE_PATHS:
        assert_writer_methods_have_complete_line_coverage(source_path)
    assert_writer_methods_have_complete_line_coverage(BWD_WRITER_SOURCE_PATH)
