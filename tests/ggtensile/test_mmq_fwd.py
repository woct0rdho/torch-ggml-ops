import builtins
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from pathlib import Path
from types import ModuleType

import pytest

from tests.ggtensile.support import (
    MMQ_FWD_INVENTORY_CASE_IDS,
    MMQ_FWD_INVENTORY_CASES,
    GGTensileInventoryCase,
    assert_resource_clean,
    build_and_inspect,
    ggtensile_toolchain,
    load_inventory_case,
    selected_solution_keys,
)
from tools.ggtensile.kernel_writer_assembly_mmq_fwd import (
    ForwardKernelWriterAssembly,
    ForwardKernelWriterError,
)
from tools.ggtensile.model import (
    ForwardSolution,
    ProblemSize,
    ProblemType,
    SchemaError,
    SolutionKey,
)
from tools.ggtensile.runtime import FixedHipForwardModule, ForwardModule
from tools.ggtensile.validation import validate_solution

_Q5_INVENTORY_CASE = next(
    case for case in MMQ_FWD_INVENTORY_CASES if case.quant_type == "Q5_K"
)


def _q4_extraction(
    tiles_ahead: int,
    dependency_width: int,
    *,
    priority: int = 2,
    metadata_after_low_wmma: bool = False,
) -> ForwardSolution:
    return ForwardSolution.q4_k_hip_decoded_staged_extraction(
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
    return ForwardSolution.q5_k_hip_decoded_staged_extraction(
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
            "Q4_K": ForwardSolution.q4_k_pilot(),
            "Q5_K": _q5_extraction(),
        }
        solution = default_solutions[quant_type]
    return SolutionKey(
        ProblemType.mmq_forward(quant_type),
        size or ProblemSize(2048, 512, 2048),
        solution,
    )


def _source(key: SolutionKey) -> str:
    return ForwardKernelWriterAssembly(key, ggtensile_toolchain()).source()


@pytest.mark.parametrize(
    "case",
    MMQ_FWD_INVENTORY_CASES,
    ids=MMQ_FWD_INVENTORY_CASE_IDS,
)
def test_forward_campaign_inventory_is_exact_and_versionless(
    case: GGTensileInventoryCase,
) -> None:
    inventory, catalog = load_inventory_case(case)
    assert inventory.problem_type == case.problem_type
    assert len(inventory.entries) == case.entry_count
    assert {entry.family for entry in inventory.entries} == case.families
    assert {entry.problem_size.m for entry in inventory.entries} == case.m_values
    assert Counter(entry.current_status for entry in inventory.entries) == dict(
        case.status_counts
    )
    assert set(catalog) == case.solution_names
    assert all(entry.quant_data_type == case.quant_type for entry in inventory.entries)
    assert all(
        validate_solution(key) == ()
        for key in selected_solution_keys(inventory, catalog)
    )
    narrow = inventory.entries[0]
    assert narrow.expected_logical_weight_shape == (512, 2048)
    expected_row_bytes = {"Q4_K": 1152, "Q5_K": 1408}[case.quant_type]
    assert narrow.expected_physical_weight_shape == (512, expected_row_bytes)
    raw = json.loads(case.inventory_path.read_text(encoding="utf-8"))
    assert "Version" not in raw and "SchemaVersion" not in raw


def test_q5_forward_inventory_selects_retained_vopd_epilogue() -> None:
    _, catalog = load_inventory_case(_Q5_INVENTORY_CASE)
    selected = catalog["selected_narrow_m32768_a7d3_p3_vopd_init"]
    assert isinstance(selected, ForwardSolution)
    assert selected.epilogue_tiles_ahead == 7
    assert selected.epilogue_dependency_width == 3
    assert selected.epilogue_priority == 3
    assert selected.accumulator_initialization == "VopdPair"


@pytest.mark.parametrize("quant_type", ("Q4_K", "Q5_K"), ids=str.lower)
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
        solution=ForwardSolution.q4_k_hip_decoded_staged_retained(),
    )
    q5_key = _key(
        "Q5_K",
        solution=ForwardSolution.q5_k_hip_decoded_staged_retained(),
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
    assert len(mapping) == len(fields(ForwardSolution)) - 1
    assert "AccumulatorInitialization" not in mapping
    assert ForwardSolution.from_mapping(mapping).to_mapping() == mapping

    vopd = replace(
        _q5_extraction(),
        accumulator_initialization="VopdPair",
    )
    vopd_mapping = vopd.to_mapping()
    assert len(vopd_mapping) == len(fields(ForwardSolution))
    assert vopd_mapping["AccumulatorInitialization"] == "VopdPair"
    assert ForwardSolution.from_mapping(vopd_mapping) == vopd


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
def test_forward_validation_rejects_nonproduction_sizes(
    quant_type: str,
    size: ProblemSize,
) -> None:
    assert validate_solution(_key(quant_type, size=size))


def test_forward_validation_rejects_mismatched_problem_type() -> None:
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
    assert {reason.rule_id for reason in validate_solution(key)} >= {
        "problem_type.forward.unsupported"
    }


def test_q5_forward_validation_rejects_q4_control() -> None:
    key = _key(
        "Q5_K",
        solution=ForwardSolution.q4_k_hip_decoded_staged_retained(),
    )
    assert {reason.rule_id for reason in validate_solution(key)} == {
        "solution.forward.control.unimplemented"
    }


def test_forward_writer_emits_direct_q8_1_f16_d4s4_q4_k_control(tmp_path: Path) -> None:
    key = _key()
    writer = ForwardKernelWriterAssembly(key, ggtensile_toolchain())
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


def test_forward_writer_emits_flat_wave_reuse_control() -> None:
    key = _key(solution=ForwardSolution.q4_k_wave_reuse())
    source = _source(key)
    assert source.count("v_wmma_i32_16x16x16_iu8") == 128
    assert "v_lshrrev_b32 v159, 5, v157" in source
    assert "Reused Q4_K group 7 across eight activation tiles." in source
    assert "global_store_d16_hi_b16" in source
    assert "v_fma_mix_f32" in source
    assert "v_cvt_f32_f16 v120" not in source
    assert "s_barrier" not in source


def test_forward_writer_emits_wave_batch_control() -> None:
    key = _key(solution=ForwardSolution.q4_k_wave_batch4())
    source = _source(key)
    assert source.count("v_wmma_i32_16x16x16_iu8") == 128
    assert "Batch Q4_K group 7 across four activation tiles at a time." in source
    assert "v_wmma_i32_16x16x16_iu8 v[112:119]" in source
    assert "v_wmma_i32_16x16x16_iu8 v[136:143]" in source
    assert "v_mov_b32 v175, v2" in source
    assert "v_mov_b32 v176, v2" not in source
    assert "v_fma_mix_f32 v148" in source
    assert "s_barrier" not in source


def test_forward_writer_emits_hip_shaped_staged_control() -> None:
    key = _key(solution=ForwardSolution.q4_k_hip_staged())
    source = _source(key)
    assert source.count("v_wmma_i32_16x16x16_iu8") == 128
    assert source.count("s_barrier") == 4
    assert "Cooperatively stage the raw packed Q4_K payload." in source
    assert "Cooperatively stage one contiguous 128-row Q8_1 F16_D4S4 plane." in source
    assert "ds_write_b128" in source
    assert "ds_read_b128" in source
    assert source.count("v_dual_fmac_f32") == 512
    assert "v_fmac_f32_e32" not in source


def test_forward_writer_emits_hip_decoded_staged_control() -> None:
    key = _key(solution=ForwardSolution.q4_k_hip_decoded_staged())
    source = _source(key)
    assert source.count("v_wmma_i32_16x16x16_iu8") == 32
    assert source.count("s_barrier") == 4
    assert "Cooperatively decode Q4_K payload into HIP's padded LDS rows." in source
    assert "Compute each packed Q4_K scale/min pair once per weight row." in source
    assert "Roll decoded Q4_K groups 0 through 3." in source
    assert "Roll decoded Q4_K groups 4 through 7." in source
    assert "ds_write2st64_b32" in source
    assert "ds_read2st64_b32" in source
    assert source.count("v_dual_fmac_f32") == 128
    assert "s_clause" not in source


def test_forward_writer_emits_retained_decoded_staged_control() -> None:
    key = _key(solution=ForwardSolution.q4_k_hip_decoded_staged_retained())
    source = _source(key)
    assert source.count("v_wmma_i32_16x16x16_iu8") == 32
    assert source.count("s_barrier") == 4
    assert source.count("v_cvt_f16_u16_e32") == 16
    assert source.count("v_mad_u32_u24 v80, 144, v237, s15") == 2
    assert source.count("s_clause 7") == 8
    assert source.count("v_mul_lo_u32 v232, 1024") == 1
    assert source.count("v_add_nc_u32 v232, 16384, v232") == 7
    assert "v_add_nc_u32 v229, s14, v234" in source
    assert "v_add_nc_u32 v80, s14, v232" in source


def test_forward_writer_emits_metadata_after_low_wmma_schedule() -> None:
    solution = ForwardSolution.q4_k_hip_decoded_staged_metadata_after_low_wmma()
    key = _key(solution=solution)
    assert validate_solution(key) == ()
    source = _source(key)
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
    source = _source(key)
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
    source = _source(key)
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
    source = _source(key)
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
def test_forward_extraction_rejects_unmeasured_exact_key(
    size: ProblemSize,
    solution: ForwardSolution,
) -> None:
    key = _key(size=size, solution=solution)
    assert {reason.rule_id for reason in validate_solution(key)} >= {
        "solution.forward.metadata_schedule.key"
    }


@pytest.mark.parametrize(
    "solution",
    (
        ForwardSolution.q5_k_hip_decoded_staged_retained(),
        ForwardSolution.q5_k_hip_decoded_staged_metadata_after_low_wmma(),
        _q5_extraction(),
    ),
)
def test_q5_forward_writer_covers_high_bit_decode_and_schedule(
    solution: ForwardSolution,
) -> None:
    key = _key("Q5_K", solution=solution)
    source = _source(key)
    assert "GGTensile Q5_K MMQ forward" in source
    assert "Cooperatively decode Q5_K payload into HIP's padded LDS rows." in source
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
    source = _source(key)
    assert "v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0" in source
    assert "v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0" in source
    assert "v_dual_mov_b32 v8, v0 :: v_dual_mov_b32 v9, v1" in source
    assert "v_dual_mov_b32 v70, v0 :: v_dual_mov_b32 v71, v1" in source
    assert "v_mov_b32 v8, v0" not in source


def test_forward_runtime_uses_exact_candidate_launch_geometry() -> None:
    direct = ForwardModule.__new__(ForwardModule)
    direct.solution_key = _key(solution=ForwardSolution.q4_k_wave_reuse())
    assert direct._launch_configuration() == ((8, 16, 1), (128, 1, 1), 0)


@pytest.mark.parametrize("quant_type", ("Q4_K", "Q5_K"), ids=str.lower)
def test_forward_runtime_uses_exact_hip_launch_geometry(quant_type: str) -> None:
    hip = FixedHipForwardModule.__new__(FixedHipForwardModule)
    hip.solution_key = _key(quant_type)
    assert hip._launch_configuration() == ((8, 16, 1), (32, 4, 1), 38_400)


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
            _key(solution=ForwardSolution.q4_k_pilot()),
            16,
            88,
            0,
            0,
            None,
            id="q4-pilot",
        ),
        pytest.param(
            _key(solution=ForwardSolution.q4_k_wave_reuse()),
            128,
            164,
            0,
            0,
            None,
            id="q4-wave-reuse",
        ),
        pytest.param(
            _key(solution=ForwardSolution.q4_k_wave_batch4()),
            128,
            194,
            0,
            0,
            None,
            id="q4-wave-batch4",
        ),
        pytest.param(
            _key(solution=ForwardSolution.q4_k_hip_staged()),
            128,
            239,
            4,
            26_624,
            None,
            id="q4-hip-staged",
        ),
        pytest.param(
            _key(solution=ForwardSolution.q4_k_hip_decoded_staged()),
            32,
            239,
            4,
            38_400,
            None,
            id="q4-decoded-staged",
        ),
        pytest.param(
            _key(solution=ForwardSolution.q4_k_hip_decoded_staged_retained()),
            32,
            239,
            4,
            38_400,
            None,
            id="q4-retained",
        ),
        pytest.param(
            _key(
                solution=ForwardSolution.q4_k_hip_decoded_staged_metadata_after_low_wmma()
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
    toolchain = ggtensile_toolchain()
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
    assert_resource_clean(inspection)


def test_forward_writer_rejects_invalid_solution() -> None:
    key = _key(solution=replace(ForwardSolution.q4_k_pilot(), wmma_clamp=False))
    with pytest.raises(ForwardKernelWriterError, match="solution rejected"):
        ForwardKernelWriterAssembly(key, ggtensile_toolchain())


def test_forward_writer_reports_missing_rocisa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = ForwardKernelWriterAssembly(_key(), ggtensile_toolchain())
    original_import = builtins.__import__

    def reject_rocisa(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "rocisa" or name.startswith("rocisa."):
            raise ImportError("missing rocisa")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_rocisa)
    with pytest.raises(ForwardKernelWriterError, match="rocisa is required"):
        writer.source()
