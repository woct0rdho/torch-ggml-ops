import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import cast

import pytest

from tests.ggtensile.support import (
    MMQ_FWD_INVENTORY_CASE_IDS,
    MMQ_FWD_INVENTORY_CASES,
    GGTensileInventoryCase,
    assert_resource_clean,
    build_and_inspect,
    load_inventory_case,
    selected_solution_keys,
)
from tools.build_mmq_bundle import kernel_specs
from tools.ggtensile.kernel_writer_assembly_mmq_fwd import (
    ForwardKernelWriterAssembly,
    ForwardKernelWriterError,
)
from tools.ggtensile.mmq_fwd_physical import (
    SignedInt8SmallMTiledLdsRegisterPlan,
    SignedInt8WaveNTiledLdsRegisterPlan,
)
from tools.ggtensile.model import (
    ForwardSolution,
    ProblemSize,
    ProblemType,
    SchemaError,
    SolutionKey,
)
from tools.ggtensile.runtime import FixedHipForwardModule, ForwardModule
from tools.ggtensile.toolchain import Toolchain
from tools.ggtensile.validation import validate_solution

_FWD_INVENTORY_CASES = {case.quant_type: case for case in MMQ_FWD_INVENTORY_CASES}
_Q3_INVENTORY_CASE = _FWD_INVENTORY_CASES["Q3_K"]
_Q5_INVENTORY_CASE = _FWD_INVENTORY_CASES["Q5_K"]
_Q6_INVENTORY_CASE = _FWD_INVENTORY_CASES["Q6_K"]

_Q6_SCHEDULE_ARTIFACT_SHA256 = {
    64: {
        "source": "c14dd8e63a6ed0ef68c1f7000b2c7905b3c97018fe38f24d7b970505d230cbc7",
        "text": "cce9eace941433c87ed1676c60de16483862839bc8b88ef74e6309ce76962427",
        "hsaco": "6843f95b4e5372e00992a09ece234d134386a83844cc687ff605b00df99583f7",
    },
    128: {
        "source": "98218bc2300f59f6fbe2de09d304f3b74c44e04178231763d06efe3d7b530ae9",
        "text": "655c33ae070ff214998a32c7075d629aa9635c3deb90b0581deef19f3e8f8956",
        "hsaco": "b85503c7fe93d1f69b5a89be528910b79439c560d952fe2d9570b6b1769f3eb6",
    },
    256: {
        "source": "e46cf7f0c91e9af5dd4801ce5394e510b587f8d02fc61e58b253a08b6e09ad16",
        "text": "655c33ae070ff214998a32c7075d629aa9635c3deb90b0581deef19f3e8f8956",
        "hsaco": "13f6621814702ef832f67597f49fc506b3e9fbd9235711dd596f427a2f89cf4a",
    },
}
_Q6_WAVEFRONT_ARTIFACT_SHA256 = {
    64: {
        "source": "b6ea810db285e2db70932e44ee83f572f7eaab64bbf99b1fb6ac61ecdf5935ad",
        "text": "acba1aef19be4ebbbf0e7f0930a42b12a9597bfcd2630acf7916f40b58b487ef",
        "hsaco": "986e78b7e065845b8902d32f67b7691e642fae1136ee62386f1b79a7f3a55af6",
    },
    128: {
        "source": "98652abf23a9fb794353bfec40c2a374198d084917c92b107760aece655c612a",
        "text": "997bfabcac5a8674aad5dfbdb223a74f76b787191573c72df258afa314ae8fde",
        "hsaco": "d724079f31a1a341b3aa25969451a4a8d43b2538ba5c244f611e3eaa57cad2a6",
    },
}


def _q4_extraction(
    tiles_ahead: int,
    dependency_width: int,
    *,
    priority: int = 2,
    metadata_after_low_wmma: bool = False,
) -> ForwardSolution:
    return ForwardSolution.q4_k_decoded_weight_lds_extraction(
        epilogue_tiles_ahead=tiles_ahead,
        epilogue_dependency_width=dependency_width,
        epilogue_priority=priority,
        metadata_after_low_wmma=metadata_after_low_wmma,
    )


def _q5_extraction(
    tiles_ahead: int = 8,
    dependency_width: int = 1,
    *,
    priority: int = 0,
    accumulator_initialization: str = "ScalarCopy",
) -> ForwardSolution:
    return ForwardSolution.q5_k_decoded_weight_lds_extraction(
        epilogue_tiles_ahead=tiles_ahead,
        epilogue_dependency_width=dependency_width,
        epilogue_priority=priority,
        accumulator_initialization=accumulator_initialization,
    )


def _key(
    quant_type: str = "Q4_K",
    size: ProblemSize | None = None,
    solution: ForwardSolution | None = None,
) -> SolutionKey:
    if solution is None:
        default_solutions = {
            "Q3_K": ForwardSolution.q3_k_hip_tiled_lds(),
            "Q4_K": ForwardSolution.q4_k_pilot(),
            "Q5_K": _q5_extraction(),
            "Q6_K": ForwardSolution.q6_k_structured_decoded(macro_tile0=64),
            "Q8_0": ForwardSolution.q8_0_direct_global(),
        }
        solution = default_solutions[quant_type]
    return SolutionKey(
        ProblemType.mmq_forward(quant_type),
        size or ProblemSize(2048, 512, 2048),
        solution,
    )


@pytest.mark.parametrize(
    "case",
    MMQ_FWD_INVENTORY_CASES,
    ids=MMQ_FWD_INVENTORY_CASE_IDS,
)
def test_forward_campaign_inventory_is_exact_and_versionless(
    case: GGTensileInventoryCase,
) -> None:
    catalog = load_inventory_case(case)
    assert catalog.problem_type == case.problem_type
    assert len(catalog.entries) == case.entry_count
    assert len(catalog.solutions) == case.solution_count
    assert {entry.problem_size.m for entry in catalog.entries} == case.m_values
    assert all(entry.quant_data_type == case.quant_type for entry in catalog.entries)
    assert all(isinstance(entry.solution, ForwardSolution) for entry in catalog.entries)
    solutions = tuple(
        cast(ForwardSolution, entry.solution) for entry in catalog.entries
    )
    if case.quant_type == "Q8_0":
        assert (
            sum(solution.operand_source == "Q8HipTiledLds" for solution in solutions)
            == 21
        )
        assert (
            sum(
                solution.lds_address_hoist == "CompactDepth32WeightRows"
                for solution in solutions
            )
            == 20
        )
        assert (
            sum(solution.lds_address_hoist == "SmallMTile" for solution in solutions)
            == 2
        )
        assert (
            sum(solution.lds_address_hoist == "HipTile" for solution in solutions) == 1
        )
        assert (
            sum(solution.lds_address_hoist == "KvCompactTile" for solution in solutions)
            == 0
        )
    assert all(validate_solution(key) == () for key in selected_solution_keys(catalog))
    narrow = catalog.entries[0]
    if case.quant_type == "Q3_K":
        assert narrow.expected_logical_weight_shape == (512, 2048)
        assert narrow.expected_physical_weight_shape == (512, 880)
    elif case.quant_type == "Q6_K":
        assert narrow.expected_logical_weight_shape == (248320, 2048)
        assert narrow.expected_physical_weight_shape == (248320, 1680)
    elif case.quant_type == "Q8_0":
        assert narrow.expected_logical_weight_shape == (1024, 4096)
        assert narrow.expected_physical_weight_shape == (1024, 4352)
    else:
        assert narrow.expected_logical_weight_shape == (512, 2048)
        expected_row_bytes = {"Q4_K": 1152, "Q5_K": 1408}[case.quant_type]
        assert narrow.expected_physical_weight_shape == (512, expected_row_bytes)
    raw = json.loads(case.catalog_path.read_text(encoding="utf-8"))
    assert catalog.to_mapping() == raw
    assert set(raw) == {"ProblemType", "Solutions", "ExactLogic"}
    forbidden = {
        "Family",
        "RepresentativeTensor",
        "CallCount",
        "HistoricalHipMedianMs",
        "CurrentStatus",
        "SelectedSolution",
    }
    serialized = json.dumps(raw)
    assert not any(field in serialized for field in forbidden)


def test_q3_forward_inventory_selects_all_qualified_exact_keys() -> None:
    catalog = load_inventory_case(_Q3_INVENTORY_CASE)
    assert len(catalog.entries) == 12
    assert len(catalog.solutions) == 1
    assert catalog.solutions[0] == ForwardSolution.q3_k_full_weight_tiled_lds()


def test_q3_forward_public_bundle_remains_unwired() -> None:
    assert all(
        not spec.cpp_id.startswith("GGTensileDenseFwdQ3K") for spec in kernel_specs()
    )


def test_q6_forward_catalog_selects_wavefront_by_exact_shape() -> None:
    catalog = load_inventory_case(_Q6_INVENTORY_CASE)
    by_m = {
        entry.problem_size.m: cast(ForwardSolution, entry.solution)
        for entry in catalog.entries
    }
    assert by_m[64].q6_output_traversal == "OutputRoleWavefront"
    assert by_m[128].q6_output_traversal == "OutputRoleWavefront"
    assert by_m[256].q6_output_traversal == "OutputRoleWavefront"
    assert by_m[64].q6_stage_clustering == "RowBatchedDecodeOrder"
    assert by_m[128].q6_latency_policy == "WavefrontDependencyDistance"
    assert by_m[256].q6_latency_policy == "WavefrontDependencyDistance"


def test_q5_forward_inventory_selects_retained_vopd_epilogue() -> None:
    catalog = load_inventory_case(_Q5_INVENTORY_CASE)
    selected = next(
        entry.solution
        for entry in catalog.entries
        if entry.problem_size == ProblemSize(32768, 512, 2048)
    )
    assert isinstance(selected, ForwardSolution)
    assert selected.epilogue_tiles_ahead == 7
    assert selected.epilogue_dependency_width == 3
    assert selected.epilogue_priority == 3
    assert selected.accumulator_initialization == "VopdPair"


@pytest.mark.parametrize(
    "quant_type", ("Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"), ids=str.lower
)
def test_forward_solution_key_is_strict_and_round_trips(quant_type: str) -> None:
    key = _key(quant_type)
    assert SolutionKey.from_mapping(key.to_mapping()) == key
    assert f"mmq_fwd_{quant_type.lower()}" in key.kernel_name
    mapping = key.to_mapping()
    solution_mapping = mapping["Solution"]
    assert isinstance(solution_mapping, dict)
    solution = dict(solution_mapping)
    solution["Unknown"] = 1
    mapping["Solution"] = solution
    with pytest.raises(SchemaError, match="invalid Solution"):
        SolutionKey.from_mapping(mapping)


def test_forward_quant_types_have_distinct_problem_and_solution_identity() -> None:
    q4_key = _key(
        "Q4_K",
        solution=ForwardSolution.q4_k_decoded_weight_lds_retained(),
    )
    q5_key = _key(
        "Q5_K",
        solution=ForwardSolution.q5_k_decoded_weight_lds_retained(),
    )
    assert q4_key.problem_type != q5_key.problem_type
    assert q4_key.hash != q5_key.hash
    assert isinstance(q5_key.solution, ForwardSolution)
    assert q5_key.solution.packed_weight_block_bytes == 176
    assert q5_key.solution.weight_decode == "DirectNibbleHighBit"


def test_forward_solution_identity_normalizes_scalar_accumulator_initialization() -> (
    None
):
    mapping = ForwardSolution.q4_k_pilot().to_mapping()
    assert len(mapping) == len(fields(ForwardSolution)) - 11
    assert "AccumulatorInitialization" not in mapping
    assert "Q6EpiloguePipelineScope" not in mapping
    assert "Q6DependencyDelayMode" not in mapping
    assert "Q6GlobalReadCachePolicy" not in mapping
    assert ForwardSolution.from_mapping(mapping).to_mapping() == mapping

    vopd = replace(
        _q5_extraction(),
        accumulator_initialization="VopdPair",
    )
    vopd_mapping = vopd.to_mapping()
    assert len(vopd_mapping) == len(fields(ForwardSolution)) - 10
    assert vopd_mapping["AccumulatorInitialization"] == "VopdPair"
    assert ForwardSolution.from_mapping(vopd_mapping) == vopd

    structured = ForwardSolution.q6_k_structured_decoded(macro_tile0=128)
    structured_mapping = structured.to_mapping()
    assert structured_mapping["Q6EpiloguePipelineScope"] == "FullTile"
    assert structured_mapping["Q6DependencyDelayMode"] == "Explicit"
    assert structured_mapping["Q6GlobalReadCachePolicy"] == "InvalidateL0"
    assert structured_mapping["Q6OutputTraversal"] == "OutputRoleGroupMajor"
    assert structured_mapping["Q6StageClustering"] == "StageDependencyOrder"
    assert structured_mapping["Q6LatencyPolicy"] == "SerializedDependencyDistance"
    assert structured_mapping["Q6PressurePolicy"] == "ExplicitRoleLifetime"
    assert structured_mapping["Q6WaitPolicy"] == "ProducerFirstUse"
    assert structured_mapping["Q6PairingPolicy"] == "DependencyCompatibleDualIssue"
    assert len(structured_mapping) == len(fields(ForwardSolution)) - 2
    assert ForwardSolution.from_mapping(structured_mapping) == structured

    for policy in (
        "Q6OutputTraversal",
        "Q6StageClustering",
        "Q6LatencyPolicy",
        "Q6PressurePolicy",
        "Q6WaitPolicy",
        "Q6PairingPolicy",
    ):
        incomplete = dict(structured_mapping)
        del incomplete[policy]
        with pytest.raises(SchemaError, match="invalid Solution"):
            ForwardSolution.from_mapping(incomplete)


@pytest.mark.parametrize(
    ("attribute", "value"),
    (
        ("kernel_language", "Source"),
        ("isa", (11, 0, 0)),
        ("wavefront_size", 64),
        ("work_group", (64, 1, 1)),
        ("matrix_instruction", (16, 16, 16, 1, 1, 2, 1, 1, 1)),
        ("macro_tile0", 32),
        ("macro_tile1", 32),
        ("depth_u", 64),
        ("activation_layout", "F32_D4"),
        ("activation_block_bytes", 136),
        ("packed_weight_block_bytes", 136),
        ("operand_source", "LDS"),
        ("weight_decode", "Prepared"),
        ("lds_address_hoist", "All"),
        ("activation_addressing", "MadU32"),
        ("metadata_conversion", "PackedFloat16"),
        ("scale_arithmetic", "FP32"),
        ("output_store", "BFloat16Truncate"),
        ("signed_weight", False),
        ("signed_activation", False),
        ("wmma_clamp", False),
        ("metadata_schedule", "EarlyWeightOverlap"),
        ("epilogue_tiles_ahead", 2),
        ("epilogue_dependency_width", 2),
        ("epilogue_priority", 1),
    ),
)
def test_forward_validation_rejects_unimplemented_mechanisms(
    attribute: str, value: object
) -> None:
    solution = replace(ForwardSolution.q4_k_pilot(), **{attribute: value})
    assert validate_solution(_key(solution=solution))


@pytest.mark.parametrize(
    ("quant_type", "size"),
    (
        ("Q4_K", ProblemSize(128, 512, 2048)),
        ("Q4_K", ProblemSize(2048, 1024, 2048)),
        ("Q4_K", ProblemSize(2048, 512, 2304)),
        ("Q5_K", ProblemSize(2048, 2048, 4096)),
    ),
)
def test_forward_validation_accepts_noncatalog_tile_aligned_sizes(
    quant_type: str,
    size: ProblemSize,
) -> None:
    assert validate_solution(_key(quant_type, size=size)) == ()


@pytest.mark.parametrize(
    "size",
    (
        ProblemSize(127, 512, 2048),
        ProblemSize(2048, 1000, 2048),
        ProblemSize(2048, 512, 2305),
    ),
)
def test_forward_validation_rejects_nondivisible_sizes(size: ProblemSize) -> None:
    assert validate_solution(_key(size=size))


def test_q6_forward_validation_accepts_wavefront_policy() -> None:
    solution = replace(
        ForwardSolution.q6_k_structured_decoded(macro_tile0=64),
        q6_output_traversal="OutputRoleWavefront",
        q6_stage_clustering="RowBatchedDecodeOrder",
        q6_latency_policy="WavefrontDependencyDistance",
    )
    assert (
        validate_solution(_key("Q6_K", ProblemSize(64, 248320, 2048), solution)) == ()
    )


def test_q6_forward_validation_rejects_unknown_semantic_policy() -> None:
    solution = replace(
        ForwardSolution.q6_k_structured_decoded(macro_tile0=64),
        q6_output_traversal="UnknownTraversal",
    )
    reasons = validate_solution(_key("Q6_K", ProblemSize(64, 248320, 2048), solution))
    assert [reason.rule_id for reason in reasons] == ["solution.forward.kernel_spec"]


def test_q6_forward_validation_rejects_q4_control() -> None:
    problem_type = ProblemType(
        operation_type="MMQForward",
        quant_data_type="Q6_K",
        data_type_a="Q8_1",
        data_type_b="Q6_K",
        dest_data_type="BFloat16",
        compute_data_type="Float",
        transpose_a=False,
        transpose_b=True,
    )
    key = SolutionKey(
        problem_type,
        ProblemSize(2048, 512, 2048),
        ForwardSolution.q4_k_pilot(),
    )
    assert {reason.rule_id for reason in validate_solution(key)} == {
        "solution.forward.problem_contract"
    }


def test_q5_forward_validation_rejects_q4_control() -> None:
    key = _key(
        "Q5_K",
        solution=ForwardSolution.q4_k_decoded_weight_lds_retained(),
    )
    assert {reason.rule_id for reason in validate_solution(key)} == {
        "solution.forward.problem_contract"
    }


def test_q8_forward_validation_rejects_unimplemented_control_variants() -> None:
    key = _key(
        "Q8_0",
        solution=replace(ForwardSolution.q8_0_direct_global(), epilogue_tiles_ahead=7),
    )
    reasons = validate_solution(key)
    assert [(reason.rule_id, reason.message) for reason in reasons] == [
        (
            "solution.forward.q8.control.unimplemented",
            "Q8_0 forward currently implements direct, register-tiled, and typed LDS controls",
        )
    ]


def test_forward_reduction_multiple_follows_the_lowering_mechanism() -> None:
    signed_int8 = _key(
        "Q8_0",
        ProblemSize(16, 16, 128),
        ForwardSolution.q8_0_direct_global(),
    )
    assert validate_solution(signed_int8) == ()

    packed_scale_minimum = _key(
        "Q4_K",
        ProblemSize(16, 16, 128),
        ForwardSolution.q4_k_pilot(),
    )
    reasons = validate_solution(packed_scale_minimum)
    assert [(reason.rule_id, reason.message) for reason in reasons] == [
        ("problem_size.k.forward_tile_multiple", "K must be a positive multiple of 256")
    ]


def test_q3_forward_validation_rejects_unimplemented_control_variants() -> None:
    key = _key(
        "Q3_K",
        solution=replace(
            ForwardSolution.q3_k_hip_tiled_lds(),
            output_store="BFloat16RNE",
        ),
    )
    reasons = validate_solution(key)
    assert {reason.rule_id for reason in reasons} == {
        "solution.forward.q3.control.unimplemented"
    }


def test_q3_full_weight_control_accepts_formula_compatible_shapes() -> None:
    solution = ForwardSolution.q3_k_full_weight_tiled_lds()
    supported = (
        ProblemSize(m, n, k)
        for m in (2048, 8192, 32768)
        for n, k in ((512, 2048), (8192, 2048), (4096, 2048), (2048, 4096))
    )
    assert all(
        validate_solution(_key("Q3_K", size=size, solution=solution)) == ()
        for size in supported
    )
    assert validate_solution(_key("Q3_K", ProblemSize(640, 704, 768), solution)) == ()
    rejected = validate_solution(_key("Q3_K", ProblemSize(641, 704, 768), solution))
    assert {reason.rule_id for reason in rejected} == {
        "problem_size.m.forward_tile_multiple"
    }


def test_forward_writer_emits_direct_q8_1_f16_d4s4_q4_k_control(tmp_path: Path) -> None:
    key = _key()
    writer = ForwardKernelWriterAssembly(key, Toolchain.discover())
    source = writer.source()
    assembly = tmp_path / "kernel.s"
    digest = writer.write(assembly)
    assert len(digest) == 64
    assert assembly.read_text(encoding="utf-8") == source
    assert source.count("v_wmma_i32_16x16x16_iu8") == 16
    assert source.count("neg_lo:[1,1,0] clamp") == 16
    assert "offset:112" in source
    assert "offset:128" in source
    assert "Reproduce Q4_K FP16 scale/min construction" in source
    assert "global_store_d16_hi_b16" in source
    assert "s_barrier" not in source
    assert "ds_" not in source


def test_forward_writer_emits_direct_q8_0_f32_d4_control(tmp_path: Path) -> None:
    key = _key("Q8_0", ProblemSize(2048, 1024, 4096))
    writer = ForwardKernelWriterAssembly(key, Toolchain.discover())
    source = writer.source()
    assembly = tmp_path / "q8_0.s"
    assert writer.write(assembly) == hashlib.sha256(source.encode()).hexdigest()
    assert source.count("v_wmma_i32_16x16x16_iu8") == 8
    assert source.count("neg_lo:[1,1,0]") == 8
    assert " clamp" not in source
    assert source.count("global_load_d16_b16 v3") == 32
    assert "offset:2" in source
    assert "offset:128" in source
    assert "s_cmp_lt_u32 s10, 32" in source
    assert "Q8_1 F32_D4 workspace" in source
    assert "s_barrier" not in source
    assert "ds_" not in source


def test_forward_writer_emits_q8_0_register_tiled_candidate(tmp_path: Path) -> None:
    key = _key(
        "Q8_0",
        ProblemSize(2048, 1024, 4096),
        ForwardSolution.q8_0_register_tiled(),
    )
    writer = ForwardKernelWriterAssembly(key, Toolchain.discover())
    source = writer.source()
    assembly = tmp_path / "q8_0_register_tiled.s"
    assert writer.write(assembly) == hashlib.sha256(source.encode()).hexdigest()
    assert source.count("v_wmma_i32_16x16x16_iu8") == 32
    assert source.count("neg_lo:[1,1,0]") == 32
    assert source.count("global_load_b128") == 32
    assert source.count("global_load_d16_b16") == 64
    assert source.count("global_store_d16_hi_b16") == 32
    assert source.count("s_clause 31") == 1
    assert ".amdhsa_system_vgpr_workitem_id 1" in source
    assert "s_cmp_lt_u32 s10, 32" in source
    assert "s_barrier" not in source
    assert "ds_" not in source


def test_forward_writer_emits_q8_0_hip_tiled_lds_control(tmp_path: Path) -> None:
    registers = SignedInt8WaveNTiledLdsRegisterPlan.allocate()
    assert registers.activation_read_address.first_register == (
        registers.activation_row.first_register
    )
    assert registers.register_count == 235
    key = _key(
        "Q8_0",
        ProblemSize(2048, 1024, 4096),
        ForwardSolution.q8_0_hip_tiled_lds(),
    )
    writer = ForwardKernelWriterAssembly(key, Toolchain.discover())
    source = writer.source()
    assembly = tmp_path / "q8_0_hip_tiled_lds.s"
    assert writer.write(assembly) == hashlib.sha256(source.encode()).hexdigest()
    assert source.count("v_wmma_i32_16x16x16_iu8") == 64
    assert source.count("global_load_b128") == 12
    assert source.count("ds_read_b128") == 72
    assert source.count("global_store_d16_hi_b16") == 64
    assert source.count("v_dual_fmac_f32") == 256
    assert source.count("s_barrier") == 2
    assert source.count("s_clause 63") == 1
    assert source.count("v_mul_lo_u32 v223, 144, v223") == 1
    assert source.count("v_mul_lo_u32 v220, 144, v220") == 0
    assert source.count("s_waitcnt lgkmcnt(18)") == 8
    assert source.count("s_waitcnt lgkmcnt(0)") == 10
    assert "HIP-shaped wave-N Q8 tile" in source
    assert "s_waitcnt lgkmcnt(0)" in source


def test_forward_writer_emits_q8_0_hip_tiled_lds_depth64_control(
    tmp_path: Path,
) -> None:
    key = _key(
        "Q8_0",
        ProblemSize(2048, 1024, 4096),
        ForwardSolution.q8_0_hip_tiled_lds_depth64(),
    )
    writer = ForwardKernelWriterAssembly(key, Toolchain.discover())
    source = writer.source()
    assembly = tmp_path / "q8_0_hip_tiled_lds_depth64.s"
    assert writer.write(assembly) == hashlib.sha256(source.encode()).hexdigest()
    assert source.count("v_wmma_i32_16x16x16_iu8") == 128
    assert source.count("global_load_b128") == 24
    assert source.count("ds_read_b128") == 144
    assert source.count("global_store_d16_hi_b16") == 64
    assert source.count("v_dual_fmac_f32") == 512
    assert source.count("s_barrier") == 4
    assert source.count("s_clause 63") == 1
    assert source.count("v_mul_lo_u32 v223, 144, v223") == 1
    assert source.count("v_mul_lo_u32 v220, 144, v220") == 0
    assert "s_cmp_lt_u32 s10, 16" in source


@pytest.mark.parametrize(
    ("macro_tile_m", "register_count", "wmma_count", "load_count", "read_count"),
    (
        (32, 91, 16, 6, 24),
        (64, 139, 32, 8, 40),
    ),
)
def test_forward_writer_emits_q8_0_small_m_tiled_lds_control(
    tmp_path: Path,
    macro_tile_m: int,
    register_count: int,
    wmma_count: int,
    load_count: int,
    read_count: int,
) -> None:
    registers = SignedInt8SmallMTiledLdsRegisterPlan.allocate(macro_tile_m // 16)
    assert registers.register_count == register_count
    key = _key(
        "Q8_0",
        ProblemSize(macro_tile_m, 129280, 4096),
        ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=macro_tile_m),
    )
    assert validate_solution(key) == ()
    writer = ForwardKernelWriterAssembly(key, Toolchain.discover())
    source = writer.source()
    assembly = tmp_path / f"q8_0_small_m{macro_tile_m}_tiled_lds.s"
    assert writer.write(assembly) == hashlib.sha256(source.encode()).hexdigest()
    assert source.count("v_wmma_i32_16x16x16_iu8") == wmma_count
    assert source.count("global_load_b128") == load_count
    assert source.count("ds_read_b128") == read_count
    assert source.count("global_store_d16_hi_b16") == wmma_count
    assert source.count("v_dual_fmac_f32") == 4 * wmma_count
    assert source.count("s_barrier") == 2
    assert source.count(f"s_clause {8 * (macro_tile_m // 16) - 1}") == 1
    assert "Split each activation row across all 128 workitems" in source
    assert "s_cmp_lt_u32 s10, 32" in source


def test_q8_0_small_m_tiled_lds_accepts_formula_compatible_shapes() -> None:
    solution = ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=32)
    assert validate_solution(_key("Q8_0", ProblemSize(96, 704, 384), solution)) == ()


def test_forward_writer_emits_q8_0_kv_compact_m64_tiled_lds_control(
    tmp_path: Path,
) -> None:
    key = _key(
        "Q8_0",
        ProblemSize(2048, 512, 4096),
        ForwardSolution.q8_0_kv_tiled_lds(),
    )
    assert validate_solution(key) == ()
    writer = ForwardKernelWriterAssembly(key, Toolchain.discover())
    source = writer.source()
    assembly = tmp_path / "q8_0_kv_compact_m64_tiled_lds.s"
    assert writer.write(assembly) == hashlib.sha256(source.encode()).hexdigest()
    assert source.count("v_wmma_i32_16x16x16_iu8") == 32
    assert source.count("global_load_b128") == 8
    assert source.count("ds_read_b128") == 40
    assert source.count("global_store_d16_hi_b16") == 32
    assert source.count("s_barrier") == 2
    assert source.count("ds_read2_b32") == 16
    assert source.count("ds_read2st64_b32") == 0
    assert source.count(", 1152, v128") == 1
    assert "offset0:179 offset1:251" in source
    assert "offset:128" in source
    assert "offset:608" not in source
    first_weight_load = source.index("global_load_b128 v[64:67], v8, s[4:5] offset:2")
    first_activation_load = source.index(
        "global_load_b128 v[72:75], v130, s[6:7] offset:16"
    )
    assert first_weight_load < first_activation_load
    assert source.index("s_waitcnt vmcnt(6)") < source.index("s_waitcnt vmcnt(3)")


def test_forward_writer_emits_q8_0_compact_depth32_weight_first_control(
    tmp_path: Path,
) -> None:
    key = _key(
        "Q8_0",
        ProblemSize(8192, 4096, 8192),
        ForwardSolution.q8_0_compact_depth32_tiled_lds(),
    )
    assert validate_solution(key) == ()
    writer = ForwardKernelWriterAssembly(key, Toolchain.discover())
    source = writer.source()
    assembly = tmp_path / "q8_0_compact_depth32_weight_first.s"
    assert writer.write(assembly) == hashlib.sha256(source.encode()).hexdigest()
    assert source.count("v_wmma_i32_16x16x16_iu8") == 64
    assert source.count("global_load_b128") == 12
    assert source.count("ds_read2_b32") == 16
    assert source.count("s_barrier") == 2
    assert source.index("global_load_b128 v[128:131]") < source.index(
        "global_load_b128 v[136:139]"
    )
    assert "1152, v" in source


def test_forward_writer_emits_q8_0_compact_depth32_small_m_control(
    tmp_path: Path,
) -> None:
    key = _key(
        "Q8_0",
        ProblemSize(64, 129280, 4096),
        ForwardSolution.q8_0_compact_depth32_tiled_lds(macro_tile0=64),
    )
    assert validate_solution(key) == ()
    writer = ForwardKernelWriterAssembly(key, Toolchain.discover())
    source = writer.source()
    assembly = tmp_path / "q8_0_compact_depth32_small_m.s"
    assert writer.write(assembly) == hashlib.sha256(source.encode()).hexdigest()
    assert source.count("v_wmma_i32_16x16x16_iu8") == 32
    assert source.count("global_load_b128") == 8
    assert source.count("ds_read2_b32") == 16
    assert source.count("global_store_d16_hi_b16") == 32
    assert source.count("s_barrier") == 2
    assert source.count("s_clause 5") == 2
    assert "1152, v" in source


def test_forward_writer_emits_q3_k_hip_tiled_lds_control(tmp_path: Path) -> None:
    key = _key(
        "Q3_K",
        ProblemSize(2048, 2048, 2048),
        ForwardSolution.q3_k_hip_tiled_lds(),
    )
    assert validate_solution(key) == ()
    writer = ForwardKernelWriterAssembly(key, Toolchain.discover())
    source = writer.source()
    assembly = tmp_path / "q3_k_hip_tiled_lds.s"
    assert writer.write(assembly) == hashlib.sha256(source.encode()).hexdigest()
    assert source.count("v_wmma_i32_16x16x16_iu8") == 128
    assert source.count("neg_lo:[1,1,0]") == 128
    assert source.count("global_load_b128") == 27
    assert source.count("ds_write_b128") == 26
    assert source.count("ds_read_b128") == 144
    assert source.count("global_store_d16_hi_b16") == 64
    assert source.count("v_dual_mov_b32") == 8
    assert "persistent zero source for all Q3 WMMAs" in source
    assert source.count("v_sub_nc_u32 v") >= 8
    assert source.count("s_barrier") == 4
    assert source.count("s_clause 7") == 8
    assert "0x7c7c7c7c" in source
    assert "0x80808080" in source
    assert "s_cmp_lt_u32 s10, 8" in source
    assert ".amdhsa_system_vgpr_workitem_id 1" in source


def test_forward_writer_emits_retained_decoded_staged_control() -> None:
    key = _key(solution=ForwardSolution.q4_k_decoded_weight_lds_retained())
    source = ForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert source.count("v_wmma_i32_16x16x16_iu8") == 32
    assert source.count("s_barrier") == 4
    assert source.count("v_cvt_f16_u16_e32") == 16
    assert source.count("v_mad_u32_u24 v80, 144, v237, s15") == 0
    assert source.count("v_mad_u32_u24 v236, 144, v237, s15") == 1
    assert source.count("v_add_nc_u32 v229, s14, v236") == 2
    assert source.count("v_add_nc_u32 v80, s14, v236") == 2
    assert source.count("s_clause 7") == 8
    assert source.count("v_mul_lo_u32 v232, 1024") == 1
    assert source.count("v_add_nc_u32 v232, 16384, v232") == 7
    assert "v_add_nc_u32 v229, s14, v234" in source
    assert "v_lshlrev_b32 v232, 4, s12" in source
    assert "v_add_nc_u32 v80, s14, v232" in source


def test_forward_writer_emits_metadata_after_low_wmma_schedule() -> None:
    solution = ForwardSolution.q4_k_decoded_weight_lds_metadata_after_low_wmma()
    key = _key(solution=solution)
    assert validate_solution(key) == ()
    source = ForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert source.count("s_waitcnt lgkmcnt(23)") == 0
    assert source.count("s_waitcnt lgkmcnt(15)") == 2
    assert source.count("s_waitcnt lgkmcnt(1)") == 2
    first_loop = source[
        source.index(".LForwardQ4KDecodedGroupLoop0:") : source.index(
            ".LForwardQ4KDecodedGroupLoop4:"
        )
    ]
    low_wmma = "v_wmma_i32_16x16x16_iu8 v[168:175], v[72:75], v[176:179], v[0:7]"
    metadata_read = "s_lshl_b32 s14, s13, 2"
    high_wmma = "v_wmma_i32_16x16x16_iu8 v[112:119], v[76:79]"
    assert first_loop.index(low_wmma) < first_loop.index(metadata_read)
    assert first_loop.index(metadata_read) < first_loop.index(high_wmma)


@pytest.mark.parametrize(
    "size",
    (
        ProblemSize(2048, 512, 2048),
        ProblemSize(8192, 2048, 512),
        ProblemSize(32768, 2048, 4096),
        ProblemSize(32768, 8192, 2048),
    ),
)
def test_forward_independent_extraction_metadata_after_low_is_production_wide(
    size: ProblemSize,
) -> None:
    solution = _q4_extraction(8, 1, priority=0, metadata_after_low_wmma=True)
    key = _key(size=size, solution=solution)
    assert validate_solution(key) == ()
    source = ForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert source.count("v_cvt_f16_u16_e32") == 16
    assert source.count("ds_write_b32 v229, v9") == 4
    assert source.count("ds_write_b32 v229, v10") == 4
    assert source.count("s_waitcnt lgkmcnt(15)") == 2


@pytest.mark.parametrize(
    ("size", "solution", "dependency_width"),
    (
        (
            ProblemSize(8192, 2048, 512),
            _q4_extraction(1, 4),
            4,
        ),
        (
            ProblemSize(32768, 2048, 512),
            _q4_extraction(1, 2),
            2,
        ),
        (
            ProblemSize(2048, 2048, 512),
            _q4_extraction(1, 2, metadata_after_low_wmma=True),
            2,
        ),
        (
            ProblemSize(8192, 2048, 512),
            _q4_extraction(1, 4, metadata_after_low_wmma=True),
            4,
        ),
        (
            ProblemSize(32768, 2048, 512),
            _q4_extraction(1, 2, metadata_after_low_wmma=True),
            2,
        ),
    ),
)
def test_forward_writer_emits_selected_shared_down_extraction(
    size: ProblemSize,
    solution: ForwardSolution,
    dependency_width: int,
) -> None:
    key = _key(size=size, solution=solution)
    assert validate_solution(key) == ()
    source = ForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert source.count("v_cvt_f16_u16_e32") == 16
    assert source.count("ds_write_b32 v229, v9") == 4
    assert source.count("ds_write_b32 v229, v10") == 4
    assert source.count("s_clause 7") == 8
    assert source.count("s_setprio 2") == 1
    assert "s_setprio 0" not in source
    assert source.index("v_mul_lo_u32 v232, 4096, v230") < source.index(
        "v_bfe_u32 v72, v8, 16, 1"
    )
    assert source.index("v_bfe_u32 v72, v8, 16, 1") < source.index(
        f"v_bfe_u32 v{72 + dependency_width - 1}, v{8 + dependency_width - 1}, 16, 1"
    )


@pytest.mark.parametrize(
    ("size", "solution", "dependency_width"),
    (
        (
            ProblemSize(32768, 512, 2048),
            _q4_extraction(4, 4, metadata_after_low_wmma=True),
            4,
        ),
        (
            ProblemSize(2048, 8192, 2048),
            _q4_extraction(1, 2, metadata_after_low_wmma=True),
            2,
        ),
        (
            ProblemSize(8192, 8192, 2048),
            _q4_extraction(4, 4, metadata_after_low_wmma=True),
            4,
        ),
        (
            ProblemSize(32768, 8192, 2048),
            _q4_extraction(2, 2, metadata_after_low_wmma=True),
            2,
        ),
    ),
)
def test_forward_writer_emits_selected_exact_epilogue(
    size: ProblemSize,
    solution: ForwardSolution,
    dependency_width: int,
) -> None:
    key = _key(size=size, solution=solution)
    assert validate_solution(key) == ()
    source = ForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert source.count("s_setprio 2") == 1
    assert "s_setprio 0" not in source
    assert source.count("s_clause 7") == 8
    epilogue = source[source.index("// Store the 128x64 row-major BF16 output tile.") :]
    for offset in range(dependency_width):
        assert epilogue.count(f"v_bfe_u32 v{72 + offset},") == 64 // dependency_width


@pytest.mark.parametrize(
    ("size", "solution"),
    (
        (
            ProblemSize(2048, 2048, 512),
            _q4_extraction(1, 4),
        ),
        (
            ProblemSize(32768, 2048, 512),
            _q4_extraction(1, 4),
        ),
        (
            ProblemSize(8192, 2048, 512),
            _q4_extraction(1, 2),
        ),
        (
            ProblemSize(8192, 8192, 2048),
            _q4_extraction(2, 2, metadata_after_low_wmma=True),
        ),
    ),
)
def test_forward_extraction_accepts_unselected_capability_candidate(
    size: ProblemSize,
    solution: ForwardSolution,
) -> None:
    key = _key(size=size, solution=solution)
    assert validate_solution(key) == ()
    source = ForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert f"s_setprio {solution.epilogue_priority}" in source


@pytest.mark.parametrize(
    "solution",
    (
        ForwardSolution.q5_k_decoded_weight_lds_retained(),
        ForwardSolution.q5_k_decoded_weight_lds_metadata_after_low_wmma(),
        _q5_extraction(),
    ),
)
def test_q5_forward_writer_covers_high_bit_decode_and_schedule(
    solution: ForwardSolution,
) -> None:
    key = _key("Q5_K", solution=solution)
    source = ForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "GGTensile Q5_K MMQ forward" in source
    assert "Cooperatively decode Q5_K payload into padded LDS rows." in source
    assert "Build Q5_K high-bit and low-nibble payload addresses." in source
    assert "v_mad_u32_u24 v231, 16, v231, v228" in source
    assert "v_mad_u32_u24 v228, 16, v230, v228" in source
    assert "global_load_b128 v[148:151], v231, s[4:5] offset:16" in source
    assert "global_load_b128 v[112:115], v228, s[4:5] offset:48" in source
    assert source.count("v_and_b32 v230, 0x01010101") == 16
    assert source.count("v_and_b32 v230, 0x02020202") == 16
    assert source.count("v_lshl_or_b32") == 40
    assert "s_add_u32 s11, s11, 176" in source
    assert ".LForwardQ5KHipStagedBlockLoop:" in source
    assert ".LForwardQ5KDecodedGroupLoop0:" in source
    assert ".LForwardQ5KDecodedGroupLoop4:" in source
    assert "Compute each packed Q5_K scale/min pair once per weight row." in source
    assert source.endswith(
        f".size {key.kernel_name}, .L{key.kernel_name}_end - {key.kernel_name}\n"
    )


def test_q5_forward_writer_emits_vopd_accumulator_initialization() -> None:
    solution = _q5_extraction(
        7,
        3,
        priority=3,
        accumulator_initialization="VopdPair",
    )
    key = _key("Q5_K", ProblemSize(32768, 512, 2048), solution)
    assert validate_solution(key) == ()
    source = ForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0" in source
    assert "v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0" in source
    assert "v_dual_mov_b32 v8, v0 :: v_dual_mov_b32 v9, v1" in source
    assert "v_dual_mov_b32 v70, v0 :: v_dual_mov_b32 v71, v1" in source
    assert "v_mov_b32 v8, v0" not in source


def test_forward_runtime_uses_exact_candidate_launch_geometry() -> None:
    direct = ForwardModule.__new__(ForwardModule)
    direct.solution_key = _key(
        solution=ForwardSolution.q4_k_decoded_weight_lds_retained()
    )
    assert direct._launch_configuration() == ((8, 16, 1), (128, 1, 1), 0)


def test_q3_forward_runtime_uses_exact_candidate_and_hip_geometry() -> None:
    key = _key(
        "Q3_K",
        ProblemSize(2048, 2048, 2048),
        ForwardSolution.q3_k_hip_tiled_lds(),
    )
    candidate = ForwardModule.__new__(ForwardModule)
    candidate.solution_key = key
    assert candidate._launch_configuration() == ((32, 16, 1), (32, 4, 1), 0)
    hip = FixedHipForwardModule.__new__(FixedHipForwardModule)
    hip.solution_key = key
    assert hip._launch_configuration() == ((32, 16, 1), (32, 4, 1), 40_448)


@pytest.mark.parametrize(
    ("k", "suffix"),
    ((2048, "k2048_j128_full"), (4096, "j128")),
)
def test_q3_fixed_hip_runtime_selects_exact_or_generic_installed_kernel(
    monkeypatch: pytest.MonkeyPatch,
    k: int,
    suffix: str,
) -> None:
    key = _key(
        "Q3_K",
        ProblemSize(2048, 2048, k),
        ForwardSolution.q3_k_hip_tiled_lds(),
    )
    selected: dict[str, object] = {}

    def capture_init(
        self: ForwardModule,
        solution_key: SolutionKey,
        code_object: Path,
        hip_library: Path | None = None,
        *,
        kernel_name: str | None = None,
    ) -> None:
        selected.update(
            solution_key=solution_key,
            code_object=code_object,
            hip_library=hip_library,
            kernel_name=kernel_name,
        )

    monkeypatch.setattr(ForwardModule, "__init__", capture_init)
    code_object = Path("q3-hip-oracle.hsaco")
    FixedHipForwardModule(key, code_object=code_object)
    assert selected == {
        "solution_key": key,
        "code_object": code_object,
        "hip_library": None,
        "kernel_name": ("torch_ggml_ops_mmq_gfx1151_v1_dense_fwd_q3_k_" + suffix),
    }


@pytest.mark.parametrize(
    ("m", "macro_tile0", "grid_y"),
    (
        (64, 64, 1),
        (128, 128, 1),
        (192, 64, 3),
        (256, 128, 2),
        (384, 128, 3),
    ),
)
def test_q6_structured_decoded_runtime_uses_exact_geometry(
    m: int,
    macro_tile0: int,
    grid_y: int,
) -> None:
    module = ForwardModule.__new__(ForwardModule)
    module.solution_key = _key(
        "Q6_K",
        ProblemSize(m, 248320, 2048),
        ForwardSolution.q6_k_structured_decoded(macro_tile0=macro_tile0),
    )
    assert validate_solution(module.solution_key) == ()
    assert module._launch_configuration() == (
        (3880, grid_y, 1),
        (32, 4, 1),
        0,
    )


@pytest.mark.parametrize(
    ("size", "macro_tile0"),
    (
        (ProblemSize(64, 64, 256), 64),
        (ProblemSize(320, 192, 768), 64),
        (ProblemSize(128, 64, 256), 128),
        (ProblemSize(384, 320, 1024), 128),
        (ProblemSize(512, 256, 2304), 128),
    ),
)
def test_q6_structured_candidate_generates_for_blind_divisible_shape(
    size: ProblemSize,
    macro_tile0: int,
) -> None:
    key = _key(
        "Q6_K",
        size,
        ForwardSolution.q6_k_structured_decoded(macro_tile0=macro_tile0),
    )
    assert validate_solution(key) == ()
    source = ForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    blocks_per_weight_row = size.k // 256
    assert key.kernel_name in source
    assert "blocks_per_weight_row" in source
    assert f"s_cmp_lg_u32 s23, {blocks_per_weight_row}" in source
    module = ForwardModule.__new__(ForwardModule)
    module.solution_key = key
    assert module._launch_configuration() == (
        (size.n // 64, size.m // macro_tile0, 1),
        (32, 4, 1),
        0,
    )


@pytest.mark.parametrize("quant_type", ("Q4_K", "Q5_K"), ids=str.lower)
def test_forward_runtime_uses_exact_hip_launch_geometry(quant_type: str) -> None:
    hip = FixedHipForwardModule.__new__(FixedHipForwardModule)
    hip.solution_key = _key(quant_type)
    assert hip._launch_configuration() == ((8, 16, 1), (32, 4, 1), 38_400)


@pytest.mark.parametrize(
    ("m", "grid_y", "lds_bytes"),
    ((32, 1, 28_928), (64, 1, 28_928), (128, 1, 38_400), (512, 4, 38_400)),
)
def test_q8_forward_runtime_uses_exact_hip_launch_geometry(
    m: int,
    grid_y: int,
    lds_bytes: int,
) -> None:
    hip = FixedHipForwardModule.__new__(FixedHipForwardModule)
    hip.solution_key = _key(
        "Q8_0",
        ProblemSize(m, 129280, 4096),
        ForwardSolution.q8_0_direct_global(),
    )
    assert hip._launch_configuration() == (
        (2020, grid_y, 1),
        (32, 4, 1),
        lds_bytes,
    )


@pytest.mark.parametrize(
    (
        "key",
        "wmma_count",
        "vgpr_count",
        "barrier_count",
        "lds_num_bytes",
        "clause_count",
    ),
    (
        pytest.param(
            _key(
                "Q3_K",
                ProblemSize(2048, 2048, 2048),
                ForwardSolution.q3_k_hip_tiled_lds(),
            ),
            128,
            144,
            4,
            28_672,
            8,
            id="q3-hip-tiled-lds-control",
        ),
        pytest.param(
            _key(
                "Q3_K",
                ProblemSize(32768, 4096, 2048),
                ForwardSolution.q3_k_full_weight_tiled_lds(),
            ),
            128,
            200,
            4,
            39_936,
            8,
            id="q3-full-weight-typed-qualified",
        ),
        pytest.param(
            _key(solution=ForwardSolution.q4_k_pilot()),
            16,
            88,
            0,
            0,
            None,
            id="q4-pilot",
        ),
        pytest.param(
            _key(solution=ForwardSolution.q4_k_decoded_weight_lds_retained()),
            32,
            239,
            4,
            38_400,
            None,
            id="q4-retained",
        ),
        pytest.param(
            _key(
                solution=ForwardSolution.q4_k_decoded_weight_lds_metadata_after_low_wmma()
            ),
            32,
            239,
            4,
            38_400,
            None,
            id="q4-metadata-after-low",
        ),
        pytest.param(
            _key(
                solution=_q4_extraction(8, 1, priority=0, metadata_after_low_wmma=True)
            ),
            32,
            239,
            4,
            38_400,
            None,
            id="q4-independent-extraction",
        ),
        pytest.param(
            _key(
                size=ProblemSize(8192, 2048, 512),
                solution=_q4_extraction(1, 4),
            ),
            32,
            239,
            4,
            38_400,
            8,
            id="q4-selected-shared-down-m8192",
        ),
        pytest.param(
            _key(
                size=ProblemSize(32768, 2048, 512),
                solution=_q4_extraction(1, 2),
            ),
            32,
            239,
            4,
            38_400,
            8,
            id="q4-selected-shared-down-m32768",
        ),
        pytest.param(_key("Q5_K"), 32, 239, 4, 38_400, 8, id="q5-selected"),
        pytest.param(
            _key(
                "Q8_0",
                ProblemSize(2048, 1024, 4096),
                ForwardSolution.q8_0_direct_global(),
            ),
            8,
            88,
            0,
            0,
            None,
            id="q8-direct-global-control",
        ),
        pytest.param(
            _key(
                "Q8_0",
                ProblemSize(2048, 1024, 4096),
                ForwardSolution.q8_0_register_tiled(),
            ),
            32,
            137,
            0,
            0,
            None,
            id="q8-register-tiled-128x32",
        ),
        pytest.param(
            _key(
                "Q8_0",
                ProblemSize(32, 129280, 4096),
                ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=32),
            ),
            16,
            96,
            2,
            24_064,
            3,
            id="q8-small-m32-tiled-lds",
        ),
        pytest.param(
            _key(
                "Q8_0",
                ProblemSize(64, 129280, 4096),
                ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=64),
            ),
            32,
            144,
            2,
            28_672,
            3,
            id="q8-small-m64-tiled-lds",
        ),
        pytest.param(
            _key(
                "Q6_K",
                ProblemSize(64, 248320, 2048),
                ForwardSolution.q6_k_structured_decoded(macro_tile0=64),
            ),
            8,
            158,
            4,
            28_928,
            13,
            id="q6-structured-decoded-m64",
        ),
        pytest.param(
            _key(
                "Q6_K",
                ProblemSize(128, 248320, 2048),
                ForwardSolution.q6_k_structured_decoded(macro_tile0=128),
            ),
            16,
            210,
            4,
            38_400,
            19,
            id="q6-structured-decoded-m128",
        ),
        pytest.param(
            _key(
                "Q6_K",
                ProblemSize(64, 248320, 2048),
                replace(
                    ForwardSolution.q6_k_structured_decoded(macro_tile0=64),
                    q6_output_traversal="OutputRoleWavefront",
                    q6_stage_clustering="RowBatchedDecodeOrder",
                    q6_latency_policy="WavefrontDependencyDistance",
                ),
            ),
            8,
            158,
            4,
            28_928,
            13,
            id="q6-wavefront-m64",
        ),
        pytest.param(
            _key(
                "Q6_K",
                ProblemSize(128, 248320, 2048),
                replace(
                    ForwardSolution.q6_k_structured_decoded(macro_tile0=128),
                    q6_output_traversal="OutputRoleWavefront",
                    q6_stage_clustering="RowBatchedDecodeOrder",
                    q6_latency_policy="WavefrontDependencyDistance",
                ),
            ),
            16,
            210,
            4,
            38_400,
            19,
            id="q6-wavefront-m128",
        ),
        pytest.param(
            _key(
                "Q6_K",
                ProblemSize(256, 248320, 2048),
                ForwardSolution.q6_k_structured_decoded(macro_tile0=128),
            ),
            16,
            210,
            4,
            38_400,
            19,
            id="q6-structured-decoded-m256",
        ),
        pytest.param(
            _key(
                "Q6_K",
                ProblemSize(320, 192, 768),
                ForwardSolution.q6_k_structured_decoded(macro_tile0=64),
            ),
            8,
            158,
            4,
            28_928,
            13,
            id="q6-structured-decoded-blind-mt64",
        ),
        pytest.param(
            _key(
                "Q6_K",
                ProblemSize(384, 320, 1024),
                ForwardSolution.q6_k_structured_decoded(macro_tile0=128),
            ),
            16,
            210,
            4,
            38_400,
            19,
            id="q6-structured-decoded-blind-mt128",
        ),
    ),
)
def test_forward_artifact_passes_strict_inspection(
    tmp_path: Path,
    key: SolutionKey,
    wmma_count: int,
    vgpr_count: int,
    barrier_count: int,
    lds_num_bytes: int,
    clause_count: int | None,
) -> None:
    toolchain = Toolchain.discover()
    artifact = build_and_inspect(
        key,
        ForwardKernelWriterAssembly(key, toolchain),
        toolchain,
        tmp_path,
    )
    inspection = artifact.inspection
    assert inspection.wmma_count == wmma_count
    assert inspection.barrier_count == barrier_count
    assert inspection.vgpr_count == vgpr_count
    assert inspection.lds_num_bytes == lds_num_bytes
    if clause_count is not None:
        assert inspection.clause_count == clause_count
    assert_resource_clean(
        inspection,
        sgpr_count=(
            27
            if isinstance(key.solution, ForwardSolution)
            and key.solution.operand_source == "Q6StructuredDecoded"
            else 16
        ),
    )
    if (
        isinstance(key.solution, ForwardSolution)
        and key.solution.operand_source == "Q6StructuredDecoded"
    ):
        expected_hashes = (
            _Q6_WAVEFRONT_ARTIFACT_SHA256
            if key.solution.q6_output_traversal == "OutputRoleWavefront"
            else _Q6_SCHEDULE_ARTIFACT_SHA256
        ).get(key.problem_size.m)
        if expected_hashes is not None:
            code_object = tmp_path / "kernel.hsaco"
            text_section = tmp_path / "kernel.text"
            toolchain.extract_section(code_object, ".text", text_section)
            assert artifact.source_hash == expected_hashes["source"]
            assert (
                hashlib.sha256(text_section.read_bytes()).hexdigest()
                == expected_hashes["text"]
            )
            assert (
                hashlib.sha256(code_object.read_bytes()).hexdigest()
                == expected_hashes["hsaco"]
            )


def test_forward_writer_rejects_invalid_solution() -> None:
    key = _key(solution=replace(ForwardSolution.q4_k_pilot(), wmma_clamp=False))
    with pytest.raises(ForwardKernelWriterError, match="solution rejected"):
        ForwardKernelWriterAssembly(key, Toolchain.discover())
