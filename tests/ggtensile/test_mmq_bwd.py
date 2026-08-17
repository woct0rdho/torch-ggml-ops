import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.ggtensile.support import (
    MMQ_BWD_INVENTORY_CASE_IDS,
    MMQ_BWD_INVENTORY_CASES,
    GGTensileInventoryCase,
    assert_resource_clean,
    build_and_inspect,
    load_inventory_case,
    selected_solution_keys,
)
from tools.ggtensile.campaign import (
    CatalogError,
    load_catalog,
)
from tools.ggtensile.cli import ManifestError
from tools.ggtensile.cli import main as ggtensile_cli_main
from tools.ggtensile.inspection import inspect_artifact
from tools.ggtensile.kernel_writer_assembly_mmq_bwd import (
    BackwardDiagnosticMode,
    BackwardKernelWriterAssembly,
)
from tools.ggtensile.mmq_bwd_search import backward_candidate_mapping
from tools.ggtensile.model import (
    BackwardSolution,
    ProblemSize,
    ProblemType,
    SchemaError,
    SolutionKey,
)
from tools.ggtensile.toolchain import Toolchain
from tools.ggtensile.validation import validate_solution
from tools.run_ggtensile_mmq_bwd_campaign import main as campaign_main

_BWD_INVENTORY_CASES = {case.quant_type: case for case in MMQ_BWD_INVENTORY_CASES}
_Q4_CATALOG = _BWD_INVENTORY_CASES["Q4_K"].catalog_path


def _pilot_key() -> SolutionKey:
    return SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(128, 2048, 512),
        BackwardSolution.pilot(),
    )


def _selected_solution(quant_type: str, problem_size: ProblemSize) -> BackwardSolution:
    catalog = load_inventory_case(_BWD_INVENTORY_CASES[quant_type])
    solution = catalog.entry_for(problem_size).solution
    assert isinstance(solution, BackwardSolution)
    assert catalog.problem_type == ProblemType.mmq_backward(quant_type)
    return solution


@pytest.mark.parametrize(
    "case",
    MMQ_BWD_INVENTORY_CASES,
    ids=MMQ_BWD_INVENTORY_CASE_IDS,
)
def test_backward_campaign_inventory_is_exact_and_versionless(
    case: GGTensileInventoryCase,
) -> None:
    catalog = load_inventory_case(case)
    assert catalog.problem_type == case.problem_type
    assert len(catalog.entries) == case.entry_count
    assert len(catalog.solutions) == case.solution_count
    assert {entry.problem_size.m for entry in catalog.entries} == case.m_values
    assert all(entry.quant_data_type == case.quant_type for entry in catalog.entries)
    assert all(validate_solution(key) == () for key in selected_solution_keys(catalog))
    raw = json.loads(case.catalog_path.read_text(encoding="utf-8"))
    assert catalog.to_mapping() == raw
    assert set(raw) == {
        "ArtifactKind",
        "KernelFamily",
        "ProblemContract",
        "KernelSpecs",
        "ExactLogic",
    }
    assert raw["ArtifactKind"] == "DeploymentCatalog"
    assert raw["KernelFamily"] == "OrdinaryBackward"
    forbidden = {
        "Family",
        "RepresentativeTensor",
        "CallCount",
        "HistoricalHipMedianMs",
        "CurrentStatus",
        "SelectedSolution",
    }
    serialized = json.dumps(raw)
    assert not any(f'"{field}":' in serialized for field in forbidden)


@pytest.mark.parametrize(
    ("quant_type", "size", "logical_shape", "physical_shape"),
    (
        ("Q3_K", ProblemSize(2048, 2048, 512), (512, 2048), (512, 880)),
        ("Q4_K", ProblemSize(2048, 512, 2048), (2048, 512), (2048, 288)),
        ("Q5_K", ProblemSize(2048, 2048, 512), (512, 2048), (512, 1408)),
        ("Q5_K", ProblemSize(2048, 512, 2048), (2048, 512), (2048, 352)),
    ),
)
def test_backward_inventory_reports_expected_packed_shapes(
    quant_type: str,
    size: ProblemSize,
    logical_shape: tuple[int, int],
    physical_shape: tuple[int, int],
) -> None:
    catalog = load_inventory_case(_BWD_INVENTORY_CASES[quant_type])
    entry = catalog.entry_for(size)
    assert entry.expected_logical_weight_shape == logical_shape
    assert entry.expected_physical_weight_shape == physical_shape


def test_q6_k_campaign_inventory_selects_exact_lm_head_solutions() -> None:
    catalog = load_inventory_case(_BWD_INVENTORY_CASES["Q6_K"])
    assert all(
        entry.expected_logical_weight_shape == (248320, 2048)
        for entry in catalog.entries
    )
    assert all(
        entry.expected_physical_weight_shape == (248320, 1680)
        for entry in catalog.entries
    )
    retained_m64 = catalog.entry_for(ProblemSize(64, 2048, 248320)).solution
    retained_m128 = catalog.entry_for(ProblemSize(128, 2048, 248320)).solution
    retained_m256 = catalog.entry_for(ProblemSize(256, 2048, 248320)).solution
    assert isinstance(retained_m64, BackwardSolution)
    assert isinstance(retained_m128, BackwardSolution)
    assert isinstance(retained_m256, BackwardSolution)
    assert retained_m64.q6_k_extraction == "packed_vopd"
    assert retained_m128.macro_tile1 == 32
    assert retained_m128.depth_u == 64
    assert retained_m256.prefetch_packed_weight_next is True


def test_q8_0_campaign_inventory_covers_ordinary_and_lm_head_keys() -> None:
    catalog = load_inventory_case(_BWD_INVENTORY_CASES["Q8_0"])
    lm_head = tuple(
        entry
        for entry in catalog.entries
        if (entry.problem_size.n, entry.problem_size.k) == (4096, 129280)
    )
    ordinary = tuple(entry for entry in catalog.entries if entry not in lm_head)
    assert len(ordinary) == 18
    assert len(lm_head) == 5
    q_a = catalog.entry_for(ProblemSize(2048, 4096, 1024))
    assert q_a.expected_physical_weight_shape == (1024, 4352)
    assert lm_head[0].expected_physical_weight_shape == (129280, 4352)


def test_q3_k_packed_decoder_uses_wave32_vopd_scale_pairs() -> None:
    key = SolutionKey(
        ProblemType.mmq_backward("Q3_K"),
        ProblemSize(8192, 2048, 512),
        replace(
            BackwardSolution.pilot(),
            one_lds_buffer=1,
            schedule_iter_alg=5,
            store_priority_opt=True,
            q3_k_extraction="packed",
            q3_k_pairing="Full",
        ),
    )
    assert validate_solution(key) == ()
    source = BackwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "v_dual_mul_f32" in source
    assert "v_dual_sub_f32" in source
    assert source.count("v_dual_mul_f32") >= 4
    assert "v_lshl_or_b32" in source
    assert "v_mul_f32" not in source


def test_q6_k_packed_vopd_decoder_pairs_adjacent_values() -> None:
    catalog = load_inventory_case(_BWD_INVENTORY_CASES["Q6_K"])
    key = catalog.entry_for(ProblemSize(64, 2048, 248320)).solution_key
    assert isinstance(key.solution, BackwardSolution)
    assert key.solution.q6_k_extraction == "packed_vopd"
    assert validate_solution(key) == ()
    source = BackwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert source.count("v_dual_sub_f32") >= 16
    assert source.count("v_dual_mul_f32") >= 16
    assert "v_lshl_or_b32" in source


def test_q6_k_rejects_geometry_with_no_decoder_rows() -> None:
    solution = replace(
        BackwardSolution.pilot(),
        matrix_instruction=(16, 16, 16, 1, 1, 1, 2, 4, 1),
        macro_tile0=64,
        macro_tile1=32,
        depth_u=32,
        prefetch_global_read=2,
        schedule_iter_alg=4,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q6_K"),
        ProblemSize(64, 2048, 248320),
        solution,
    )
    assert any(
        reason.rule_id == "problem_size.decoder_rows.empty"
        for reason in validate_solution(key)
    )


def test_lds_row_padding_is_strict_and_quant_aware() -> None:
    padded = replace(
        BackwardSolution.pilot(),
        lds_pad_b=8,
        lds_swizzle_chunk_b=0,
        q3_k_pairing="Partial",
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q3_K"),
        ProblemSize(2048, 2048, 512),
        padded,
    )
    assert padded.lds_num_bytes == 10240
    assert validate_solution(key) == ()

    conflicting = replace(padded, lds_swizzle_chunk_b=8)
    reasons = validate_solution(
        SolutionKey(key.problem_type, key.problem_size, conflicting)
    )
    assert any(reason.rule_id == "solution.ldspadb.swizzle" for reason in reasons)


def test_backward_quant_types_have_distinct_problem_and_solution_identity() -> None:
    sizes = {
        "Q3_K": ProblemSize(128, 2048, 512),
        "Q4_K": ProblemSize(128, 2048, 512),
        "Q5_K": ProblemSize(128, 2048, 512),
        "Q6_K": ProblemSize(128, 2048, 248320),
        "Q8_0": ProblemSize(128, 4096, 1024),
    }
    keys = tuple(
        SolutionKey(
            ProblemType.mmq_backward(quant_type),
            size,
            BackwardSolution.pilot(),
        )
        for quant_type, size in sizes.items()
    )
    assert len({key.problem_type for key in keys}) == len(sizes)
    assert len({key.hash for key in keys}) == len(sizes)
    for quant_type, key in zip(sizes, keys, strict=True):
        assert f"mmq_bwd_{quant_type.lower()}_" in key.kernel_name

    q4 = ProblemType.mmq_backward("Q4_K")
    q5 = ProblemType.mmq_backward("Q5_K")
    q5_scalar = SolutionKey(
        q5,
        ProblemSize(128, 2048, 512),
        replace(BackwardSolution.pilot(), q5_k_extraction="scalar"),
    )
    assert validate_solution(q5_scalar) == ()
    assert validate_solution(
        SolutionKey(
            q4,
            ProblemSize(128, 2048, 512),
            replace(BackwardSolution.pilot(), q5_k_extraction="scalar"),
        )
    )


@pytest.mark.parametrize(
    ("legacy_root", "canonical_root"),
    (
        ("ProblemType", "ProblemContract"),
        ("Solutions", "KernelSpecs"),
    ),
)
def test_q4_k_deployment_catalog_rejects_legacy_roots(
    tmp_path: Path, legacy_root: str, canonical_root: str
) -> None:
    catalog_path = tmp_path / "catalog.json"
    value = json.loads(_Q4_CATALOG.read_text())
    value[legacy_root] = value.pop(canonical_root)
    catalog_path.write_text(json.dumps(value))
    with pytest.raises(CatalogError, match="invalid deployment catalog keys"):
        load_catalog(catalog_path)


def test_ordinary_exact_key_rejects_nonpositive_problem_dimensions() -> None:
    mapping = _pilot_key().to_mapping()
    problem = mapping["Problem"]
    assert isinstance(problem, dict)
    problem["n"] = 0
    with pytest.raises(SchemaError, match="dimensions must be positive"):
        SolutionKey.from_mapping(mapping)


def test_q4_k_campaign_prepare_is_serial_and_immutable(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    arguments = [
        "prepare",
        "--artifact-root",
        str(root),
        "--key",
        "2048,512,2048",
    ]
    assert campaign_main(arguments) == 0
    summary = json.loads((root / "prepare.json").read_text())
    assert summary["Phase"] == "Prepare"
    assert summary["Status"] == "Accepted"
    assert summary["Catalog"] == str(_Q4_CATALOG.resolve())
    assert "Solution" not in summary
    assert len(summary["Entries"]) == 1
    selected_key = (
        load_catalog(_Q4_CATALOG).entry_for(ProblemSize(2048, 512, 2048)).solution_key
    )
    assert summary["Entries"][0]["SolutionHash"] == selected_key.hash
    assert summary["Entries"][0]["KernelName"] == selected_key.kernel_name
    artifact = root / "m2048_n512_k2048"
    assert (artifact / "generate.json").is_file()
    assert (artifact / "build.json").is_file()
    assert (artifact / "inspect.json").is_file()
    with pytest.raises(CatalogError, match="refusing to overwrite artifact root"):
        campaign_main(arguments)


def test_q4_k_campaign_prepare_accepts_explicit_solution(tmp_path: Path) -> None:
    solution_path = tmp_path / "solution.json"
    solution = (
        load_catalog(_Q4_CATALOG).entry_for(ProblemSize(2048, 512, 2048)).solution
    )
    assert isinstance(solution, BackwardSolution)
    solution_path.write_text(json.dumps(backward_candidate_mapping(solution, "Q4_K")))
    root = tmp_path / "campaign"
    assert (
        campaign_main(
            [
                "prepare",
                "--artifact-root",
                str(root),
                "--key",
                "2048,512,2048",
                "--solution",
                str(solution_path),
            ]
        )
        == 0
    )
    summary = json.loads((root / "prepare.json").read_text())
    assert summary["Solution"] == str(solution_path.resolve())
    assert summary["Catalog"] == str(_Q4_CATALOG.resolve())


def test_pilot_solution_identity_and_round_trip() -> None:
    key = _pilot_key()
    assert validate_solution(key) == ()
    assert key.hash.startswith("ggsol_")
    assert SolutionKey.from_mapping(key.to_mapping()) == key


@pytest.mark.parametrize(
    ("n", "packed_row_bytes"), ((512, 288), (2048, 1152), (4096, 2304))
)
def test_writer_specializes_production_q4_k_row_stride(
    n: int, packed_row_bytes: int
) -> None:
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(128, n, 512),
        BackwardSolution.pilot(),
    )
    assert validate_solution(key) == ()
    writer = BackwardKernelWriterAssembly(key, Toolchain.discover())
    source = writer.source()
    temporary = writer.registers.temporary
    assert f"v_mul_lo_u32 v{temporary}, {packed_row_bytes}, v{temporary}" in source


def test_writer_strength_reduces_power_of_two_row_strides() -> None:
    toolchain = Toolchain.discover()
    pipeline = replace(
        BackwardSolution.pilot(),
        one_lds_buffer=0,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
        store_priority_opt=False,
    )
    production_key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(128, 2048, 512),
        pipeline,
    )
    production_writer = BackwardKernelWriterAssembly(production_key, toolchain)
    production_source = production_writer.source()
    registers = production_writer.registers
    assert (
        f"lshlrev_b32 v{registers.address + 4}, 10, "
        f"v{registers.address + 4}" in production_source
    )
    assert (
        f"v_lshlrev_b32 v{registers.temporary}, 12, "
        f"v{registers.temporary}" in production_source
    )

    reduced_key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(128, 2048, 96),
        pipeline,
    )
    reduced_writer = BackwardKernelWriterAssembly(reduced_key, toolchain)
    reduced_source = reduced_writer.source()
    assert (
        f"v_mul_lo_u32 v{reduced_writer.registers.address + 4}, 192, "
        f"v{reduced_writer.registers.address + 4}" in reduced_source
    )


def test_validation_accepts_formula_n_and_rejects_partial_quant_blocks() -> None:
    compatible = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(128, 1024, 512),
        BackwardSolution.pilot(),
    )
    assert validate_solution(compatible) == ()

    partial = replace(compatible, problem_size=ProblemSize(128, 1000, 512))
    assert "problem_size.n.quant_block" in {
        reason.rule_id for reason in validate_solution(partial)
    }


def test_q3_pairing_is_explicit_shape_independent_and_strict() -> None:
    catalog = load_inventory_case(_BWD_INVENTORY_CASES["Q3_K"])
    selected = {}
    for entry in catalog.entries:
        assert isinstance(entry.solution, BackwardSolution)
        selected[(entry.problem_size.m, entry.problem_size.k)] = (
            entry.solution.q3_k_pairing
        )
    assert selected == {
        (2048, 512): "Full",
        (8192, 512): "Partial",
        (32768, 512): "Full",
        (2048, 8192): "Full",
        (8192, 8192): "Full",
        (32768, 8192): "Full",
    }

    full = _selected_solution("Q3_K", ProblemSize(32768, 2048, 512))
    first = SolutionKey(
        ProblemType.mmq_backward("Q3_K"), ProblemSize(8192, 2048, 512), full
    )
    second = replace(first, problem_size=ProblemSize(32768, 2048, 512))
    assert validate_solution(first) == validate_solution(second) == ()
    first_source = BackwardKernelWriterAssembly(first, Toolchain.discover()).source()
    second_source = BackwardKernelWriterAssembly(second, Toolchain.discover()).source()
    assert first_source.replace(first.kernel_name, "<KERNEL>") == second_source.replace(
        second.kernel_name, "<KERNEL>"
    )

    inactive = replace(full, q3_k_pairing="Inactive")
    assert "solution.q3.packed.pairing" in {
        reason.rule_id
        for reason in validate_solution(replace(first, solution=inactive))
    }
    scalar_full = replace(full, q3_k_extraction="scalar", q3_k_pairing="Full")
    assert "solution.q3.scalar.pairing" in {
        reason.rule_id
        for reason in validate_solution(replace(first, solution=scalar_full))
    }
    q4_with_q3_policy = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(128, 256, 128),
        replace(BackwardSolution.pilot(), q3_k_pairing="Partial"),
    )
    assert "solution.q3.controls.inert" in {
        reason.rule_id for reason in validate_solution(q4_with_q3_policy)
    }


def test_backward_decoder_capabilities_generalize_compact_geometry_only() -> None:
    compact = replace(
        BackwardSolution.pilot(),
        work_group=(32, 2, 1),
        matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 2, 1),
        macro_tile0=32,
        macro_tile1=64,
        prefetch_global_read=2,
        schedule_iter_alg=4,
        lds_pad_b=8,
    )
    shape = ProblemSize(32, 256, 128)
    for quant_type in ("Q4_K", "Q5_K"):
        assert (
            validate_solution(
                SolutionKey(ProblemType.mmq_backward(quant_type), shape, compact)
            )
            == ()
        )

    padded_depth64 = replace(compact, depth_u=64)
    for quant_type in ("Q4_K", "Q5_K"):
        reasons = validate_solution(
            SolutionKey(ProblemType.mmq_backward(quant_type), shape, padded_depth64)
        )
        assert "solution.depthu64.padded.decoder" in {
            reason.rule_id for reason in reasons
        }

    q3_full_rows4 = replace(padded_depth64, q3_k_pairing="Full")
    reasons = validate_solution(
        SolutionKey(ProblemType.mmq_backward("Q3_K"), shape, q3_full_rows4)
    )
    assert "solution.q3.full_pairing.decoder_rows" in {
        reason.rule_id for reason in reasons
    }
    q3_lane_share = replace(compact, q3_k_pairing="Partial", packed_weight_lane_share=2)
    reasons = validate_solution(
        SolutionKey(ProblemType.mmq_backward("Q3_K"), shape, q3_lane_share)
    )
    assert "solution.decoder.packedweightlaneshare" in {
        reason.rule_id for reason in reasons
    }


def test_backward_deep_pipeline_geometry_is_formula_derived() -> None:
    compact = replace(
        BackwardSolution.pilot(),
        work_group=(32, 2, 1),
        matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 2, 1),
        macro_tile0=32,
        macro_tile1=64,
        prefetch_global_read=2,
        schedule_iter_alg=4,
        lds_pad_b=8,
    )
    compact_key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32, 256, 128),
        compact,
    )
    assert validate_solution(compact_key) == ()

    too_narrow = replace(
        compact,
        matrix_instruction=(16, 16, 16, 1, 1, 1, 2, 2, 1),
        macro_tile1=32,
    )
    assert "solution.scheduleiteralg.geometry" in {
        reason.rule_id
        for reason in validate_solution(replace(compact_key, solution=too_narrow))
    }

    four_wave = replace(
        compact,
        work_group=(32, 4, 1),
        matrix_instruction=(16, 16, 16, 1, 1, 4, 4, 4, 1),
        macro_tile0=256,
        macro_tile1=64,
        lds_pad_b=0,
        lds_swizzle_chunk_b=8,
    )
    four_wave_key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(256, 256, 128),
        four_wave,
    )
    assert validate_solution(four_wave_key) == ()

    over_capacity = replace(
        four_wave,
        matrix_instruction=(16, 16, 16, 1, 1, 4, 8, 4, 1),
        macro_tile1=128,
    )
    assert "solution.scheduleiteralg.geometry" in {
        reason.rule_id
        for reason in validate_solution(replace(four_wave_key, solution=over_capacity))
    }


def test_writer_enables_and_flattens_packed_workitem_xy() -> None:
    writer = BackwardKernelWriterAssembly(_pilot_key(), Toolchain.discover())
    source = writer.source()
    registers = writer.registers
    assert ".amdhsa_system_vgpr_workitem_id 1" in source
    assert f"v_bfe_u32 v{registers.serial}, v0, 10, 10" in source
    assert (
        f"v_add_nc_u32 v{registers.serial}, v{registers.serial}, "
        f"v{registers.temporary}" in source
    )
    assert "s_load_dwordx4" not in source
    assert registers.total_sgprs == 16


@pytest.mark.parametrize(
    ("key", "vgpr_count", "lds_num_bytes", "wmma_count", "source_markers"),
    (
        pytest.param(
            SolutionKey(
                ProblemType.mmq_backward("Q3_K"),
                ProblemSize(2048, 2048, 512),
                replace(
                    _selected_solution("Q3_K", ProblemSize(8192, 2048, 512)),
                    q3_k_pairing="Full",
                ),
            ),
            243,
            5120,
            32,
            (
                "Precompute A row coordinates shared by every DepthU iteration.",
                "ds_store_b16_d16_hi",
                "ds_load_b128",
            ),
            id="q3-k",
        ),
        pytest.param(
            _pilot_key(),
            192,
            8192,
            32,
            ("GGTensile Q4_K MMQ backward",),
            id="q4-k",
        ),
        pytest.param(
            SolutionKey(
                ProblemType.mmq_backward("Q5_K"),
                ProblemSize(128, 2048, 512),
                BackwardSolution.pilot(),
            ),
            200,
            8192,
            32,
            (
                "Build Q5_K payload",
                "Decode Q5_K low nibbles and high payload bits into LDS.",
                "offset:48",
                "0x01010101",
                "v_lshl_or_b32",
            ),
            id="q5-k",
        ),
        pytest.param(
            SolutionKey(
                ProblemType.mmq_backward("Q6_K"),
                ProblemSize(128, 2048, 248320),
                replace(
                    BackwardSolution.pilot(),
                    matrix_instruction=(16, 16, 16, 1, 1, 2, 4, 4, 1),
                    macro_tile0=128,
                    macro_tile1=64,
                    prefetch_global_read=2,
                    schedule_iter_alg=5,
                    lds_swizzle_chunk_b=8,
                ),
            ),
            142,
            4096,
            16,
            ("Build Q6_K block addresses", "global_load_b128", "v_lshl_or_b32"),
            id="q6-k",
        ),
        pytest.param(
            SolutionKey(
                ProblemType.mmq_backward("Q8_0"),
                ProblemSize(128, 4096, 1024),
                BackwardSolution.pilot(),
            ),
            186,
            8192,
            32,
            (
                "Build Q8_0 block addresses",
                "Decode Q8_0 signed int8 payload and scale",
                "global_load_d16_b16",
                "global_load_b128",
                "v_cvt_f32_i32",
            ),
            id="q8-0",
        ),
    ),
)
def test_build_and_inspect_quant_backend(
    tmp_path: Path,
    key: SolutionKey,
    vgpr_count: int,
    lds_num_bytes: int,
    wmma_count: int,
    source_markers: tuple[str, ...],
) -> None:
    assert validate_solution(key) == ()
    toolchain = Toolchain.discover()
    artifact = build_and_inspect(
        key,
        BackwardKernelWriterAssembly(key, toolchain),
        toolchain,
        tmp_path,
    )
    assert len(artifact.source_hash) == 64
    assert all(marker in artifact.source for marker in source_markers)
    inspection = artifact.inspection
    assert inspection.target == "gfx1151"
    assert inspection.code_object_version == 5
    assert inspection.vgpr_count == vgpr_count
    assert inspection.lds_num_bytes == lds_num_bytes
    assert inspection.wmma_count == wmma_count
    assert inspection.valu_issue_count > 0
    assert inspection.vmem_count > 0
    assert inspection.lds_count > 0
    assert inspection.wait_count > 0
    assert_resource_clean(inspection)
    if key.problem_type.quant_data_type == "Q3_K":
        assert artifact.source.count("v_wmma_f32_16x16x16_bf16") == 32
        assert artifact.source.count("ds_store_b16_d16_hi") == 16
        assert artifact.source.count("ds_load_b128") == 16
    if key.problem_type.quant_data_type == "Q4_K":
        assert inspection.max_vgpr_index == 191
        assert inspection.max_sgpr_index == 15
        assert inspection.valu_operation_count > inspection.valu_issue_count
        assert inspection.vopd_count > 0
        assert inspection.barrier_count == 2
        assert inspection.clause_count == 0
        assert inspection.delay_alu_count == 0
        assert inspection.buffer_gl0_inv_count == 0


def test_build_and_inspect_q8_0_depth_u64_backend(tmp_path: Path) -> None:
    solution = replace(
        BackwardSolution.pilot(),
        depth_u=64,
        prefetch_global_read=2,
        schedule_iter_alg=4,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q8_0"),
        ProblemSize(128, 4096, 1024),
        solution,
    )
    assert validate_solution(key) == ()
    toolchain = Toolchain.discover()
    artifact = build_and_inspect(
        key,
        BackwardKernelWriterAssembly(key, toolchain),
        toolchain,
        tmp_path,
        stem="q8_0_depth_u64",
    )
    text = artifact.source
    assert text.count("global_load_d16_b16") == 4
    assert text.count("ds_store_b16_d16_hi") == 64
    assert "Precompute XOR-8 LDS store bases" in text
    inspection = artifact.inspection
    assert inspection.vgpr_count == 220
    assert inspection.lds_num_bytes == 16384
    assert inspection.wmma_count == 64
    assert_resource_clean(inspection)


def test_build_and_inspect_q8_0_compact_geometry(tmp_path: Path) -> None:
    solution = replace(
        BackwardSolution.pilot(),
        matrix_instruction=(16, 16, 16, 1, 1, 4, 4, 4, 1),
        macro_tile0=256,
        macro_tile1=64,
        prefetch_global_read=2,
        schedule_iter_alg=5,
        lds_pad_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q8_0"),
        ProblemSize(2048, 4096, 1024),
        solution,
    )
    assert validate_solution(key) == ()
    toolchain = Toolchain.discover()
    artifact = build_and_inspect(
        key,
        BackwardKernelWriterAssembly(key, toolchain),
        toolchain,
        tmp_path,
        stem="q8_0_compact",
    )
    inspection = artifact.inspection
    assert inspection.vgpr_count == 231
    assert inspection.lds_num_bytes == 5120
    assert inspection.wmma_count == 32
    assert_resource_clean(inspection)


def test_build_and_inspect_q8_0_depth_u64_compact_geometry(tmp_path: Path) -> None:
    solution = replace(
        BackwardSolution.pilot(),
        matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
        macro_tile0=64,
        macro_tile1=64,
        depth_u=64,
        prefetch_global_read=2,
        schedule_iter_alg=4,
        lds_pad_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q8_0"),
        ProblemSize(64, 4096, 129280),
        solution,
    )
    assert validate_solution(key) == ()
    q4_control = validate_solution(
        SolutionKey(
            ProblemType.mmq_backward("Q4_K"),
            ProblemSize(64, 4096, 129280),
            solution,
        )
    )
    assert {reason.rule_id for reason in q4_control} >= {
        "solution.depthu64.padded.decoder"
    }
    toolchain = Toolchain.discover()
    artifact = build_and_inspect(
        key,
        BackwardKernelWriterAssembly(key, toolchain),
        toolchain,
        tmp_path,
        stem="q8_0_m64_depth_u64",
    )
    inspection = artifact.inspection
    assert inspection.vgpr_count == 90
    assert inspection.lds_num_bytes == 9216
    assert inspection.wmma_count == 16
    assert_resource_clean(inspection)


def test_build_and_inspect_q8_0_two_wave_m32_geometry(tmp_path: Path) -> None:
    solution = replace(
        BackwardSolution.pilot(),
        work_group=(32, 2, 1),
        matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 2, 1),
        macro_tile0=32,
        macro_tile1=64,
        prefetch_global_read=2,
        schedule_iter_alg=5,
        lds_pad_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q8_0"),
        ProblemSize(32, 4096, 129280),
        solution,
    )
    assert validate_solution(key) == ()
    assert (
        validate_solution(
            SolutionKey(
                ProblemType.mmq_backward("Q4_K"),
                ProblemSize(32, 4096, 129280),
                solution,
            )
        )
        == ()
    )
    toolchain = Toolchain.discover()
    artifact = build_and_inspect(
        key,
        BackwardKernelWriterAssembly(key, toolchain),
        toolchain,
        tmp_path,
        stem="q8_0_m32",
    )
    inspection = artifact.inspection
    assert inspection.max_flat_workgroup_size == 64
    assert inspection.vgpr_count == 90
    assert inspection.lds_num_bytes == 5120
    assert inspection.wmma_count == 8
    assert_resource_clean(inspection)


def test_build_and_inspect_q8_0_packed_vopd_decode(tmp_path: Path) -> None:
    solution = replace(
        BackwardSolution.pilot(),
        work_group=(32, 2, 1),
        matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 2, 1),
        macro_tile0=32,
        macro_tile1=64,
        depth_u=64,
        prefetch_global_read=2,
        schedule_iter_alg=4,
        lds_pad_b=8,
        q8_0_extraction="packed_vopd",
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q8_0"),
        ProblemSize(32, 4096, 129280),
        solution,
    )
    assert validate_solution(key) == ()
    toolchain = Toolchain.discover()
    artifact = build_and_inspect(
        key,
        BackwardKernelWriterAssembly(key, toolchain),
        toolchain,
        tmp_path,
        stem="q8_0_m32_vopd",
    )
    assert "v_dual_mul_f32" in artifact.source
    assert "global_load_b128" in artifact.source
    inspection = artifact.inspection
    assert inspection.vgpr_count == 106
    assert inspection.lds_num_bytes == 9216
    assert inspection.vopd_count > 40
    assert_resource_clean(inspection)


def test_cli_generate_build_and_inspect_manifests(tmp_path: Path) -> None:
    solution_path = tmp_path / "requested.json"
    solution_path.write_text(json.dumps(_pilot_key().to_mapping()))
    artifact_dir = tmp_path / "artifact"

    assert (
        ggtensile_cli_main(
            [
                "generate",
                "--solution-key",
                str(solution_path),
                "--output-dir",
                str(artifact_dir),
            ]
        )
        == 0
    )
    generate = json.loads((artifact_dir / "generate.json").read_text())
    assert generate["Phase"] == "Generate"
    assert generate["Status"] == "Accepted"
    assert len(generate["AssemblySHA256"]) == 64

    assert (
        ggtensile_cli_main(
            [
                "build",
                "--generate-manifest",
                str(artifact_dir / "generate.json"),
            ]
        )
        == 0
    )
    build = json.loads((artifact_dir / "build.json").read_text())
    assert build["Phase"] == "Build"
    assert build["Status"] == "Accepted"
    assert "ObjectSHA256" not in build
    assert "CodeObjectSHA256" not in build

    assert (
        ggtensile_cli_main(
            [
                "inspect",
                "--build-manifest",
                str(artifact_dir / "build.json"),
            ]
        )
        == 0
    )
    inspection = json.loads((artifact_dir / "inspect.json").read_text())
    assert inspection["Phase"] == "Inspect"
    assert inspection["Status"] == "Accepted"
    assert inspection["Inspection"]["NumVgpr"] == 192
    assert inspection["Inspection"]["StaticValuIssueCount"] > 0
    assert inspection["Inspection"]["StaticVopdCount"] > 0
    assert "CodeObjectSHA256" not in inspection["Inspection"]
    assert "NormalizedAssemblySHA256" not in inspection["Inspection"]

    with pytest.raises(ManifestError, match="refusing to overwrite"):
        ggtensile_cli_main(
            [
                "generate",
                "--solution-key",
                str(solution_path),
                "--output-dir",
                str(artifact_dir),
            ]
        )


def test_cli_records_rejected_solution_manifest(tmp_path: Path) -> None:
    rejected = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32768, 2048, 8192),
        replace(BackwardSolution.pilot(), depth_u=16),
    )
    solution_path = tmp_path / "rejected.json"
    solution_path.write_text(json.dumps(rejected.to_mapping()))
    artifact_dir = tmp_path / "rejected"

    assert (
        ggtensile_cli_main(
            [
                "generate",
                "--solution-key",
                str(solution_path),
                "--output-dir",
                str(artifact_dir),
            ]
        )
        == 2
    )
    manifest = json.loads((artifact_dir / "generate.json").read_text())
    assert manifest["Status"] == "Rejected"
    assert manifest["RejectReasons"]
    assert not (artifact_dir / "kernel.s").exists()


def test_writer_emits_distinct_xor8_lds_layout() -> None:
    pilot = BackwardSolution.pilot()
    swizzled = replace(pilot, lds_swizzle_chunk_b=8)
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(128, 2048, 512),
        swizzled,
    )
    assert validate_solution(key) == ()

    source = BackwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Precompute XOR-8 LDS store bases" in source
    assert "v_xor_b32" in source
    assert source.count("ds_load_b128") == 32
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32


def test_writer_emits_distinct_xor16_lds_layout() -> None:
    pilot = BackwardSolution.pilot()
    swizzled = replace(pilot, lds_swizzle_chunk_b=16)
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(128, 2048, 512),
        swizzled,
    )
    assert validate_solution(key) == ()

    source = BackwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Precompute XOR-16 LDS store bases" in source
    assert source.count("ds_load_b128") == 32
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32


def test_writer_emits_distinct_xor4_lds_layout() -> None:
    pilot = BackwardSolution.pilot()
    swizzled = replace(pilot, lds_swizzle_chunk_b=4, schedule_iter_alg=3)
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(128, 2048, 512),
        swizzled,
    )
    assert validate_solution(key) == ()

    source = BackwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Precompute XOR-4 LDS store bases" in source
    assert source.count("ds_load_b64") == 64
    assert source.count("ds_load_b128") == 0
    assert source.count("s_waitcnt vmcnt(2) lgkmcnt(4)") == 2
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32


def test_writer_emits_sia3_partial_wait_schedule() -> None:
    pilot = BackwardSolution.pilot()
    scheduled = replace(pilot, schedule_iter_alg=3)
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(128, 2048, 512),
        scheduled,
    )
    assert validate_solution(key) == ()

    source = BackwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert source.count("s_waitcnt vmcnt(2) lgkmcnt(2)") == 2
    assert source.count("s_waitcnt lgkmcnt(2)") == 6
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32


def test_writer_overlaps_first_a_half_with_decode(tmp_path: Path) -> None:
    pilot = BackwardSolution.pilot()
    scheduled = replace(
        pilot,
        schedule_iter_alg=4,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32768, 2048, 8192),
        scheduled,
    )
    assert validate_solution(key) == ()

    toolchain = Toolchain.discover()
    assembly = tmp_path / "schedule4.s"
    object_path = tmp_path / "schedule4.o"
    code_object = tmp_path / "schedule4.hsaco"
    source = BackwardKernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    prefetch = source.index("Prefetch A fragments")
    q4_decode = source.index("Unpack Q4_K six-bit scale/min fields")
    assert prefetch < q4_decode
    assert "Reuse prefetched A pointers across fused B decode." in source
    assert "0x0f0f0f0f" in source
    assert "v_cvt_f32_ubyte3_e32" in source
    assert "s_nop" not in source
    assert source[prefetch:q4_decode].count("s_waitcnt vmcnt(4)") == 1
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 196
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 8192


def test_writer_prefetches_both_a_halves(tmp_path: Path) -> None:
    pilot = BackwardSolution.pilot()
    prefetched = replace(
        pilot,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32768, 2048, 8192),
        prefetched,
    )
    assert validate_solution(key) == ()

    toolchain = Toolchain.discover()
    assembly = tmp_path / "prefetch_global_read.s"
    object_path = tmp_path / "prefetch_global_read.o"
    code_object = tmp_path / "prefetch_global_read.hsaco"
    source = BackwardKernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert source.count("s_waitcnt vmcnt(8)") == 1
    assert source.count("s_waitcnt vmcnt(4) lgkmcnt(0)") == 1
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 212
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 8192


def test_writer_shares_packed_weight_across_nibble_lanes(
    tmp_path: Path,
) -> None:
    pilot = BackwardSolution.pilot()
    shared = replace(
        pilot,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        packed_weight_lane_share=2,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32768, 2048, 8192),
        shared,
    )
    assert validate_solution(key) == ()

    toolchain = Toolchain.discover()
    assembly = tmp_path / "packed_weight_lane_share.s"
    object_path = tmp_path / "packed_weight_lane_share.o"
    code_object = tmp_path / "packed_weight_lane_share.hsaco"
    source = BackwardKernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert "s_and_saveexec_b32" in source
    assert source.count("v_mov_b32_dpp") == 8
    assert "ds_bpermute_b32" not in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 212
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 8192


def test_writer_pipelines_packed_weight_reads(tmp_path: Path) -> None:
    pilot = BackwardSolution.pilot()
    pipelined = replace(
        pilot,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        prefetch_packed_weight_next=True,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32768, 2048, 8192),
        pipelined,
    )
    assert validate_solution(key) == ()

    toolchain = Toolchain.discover()
    assembly = tmp_path / "prefetch_packed_weight.s"
    object_path = tmp_path / "prefetch_packed_weight.o"
    code_object = tmp_path / "prefetch_packed_weight.hsaco"
    source = BackwardKernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert ".LPackedDepthULoop:" in source
    assert "current A before next packed reads" in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 212
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 8192
    assert inspection.barrier_count == 3


def test_writer_combines_global_prefetch_with_partial_waits(
    tmp_path: Path,
) -> None:
    pilot = BackwardSolution.pilot()
    scheduled = replace(
        pilot,
        schedule_iter_alg=5,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32768, 2048, 8192),
        scheduled,
    )
    assert validate_solution(key) == ()

    toolchain = Toolchain.discover()
    assembly = tmp_path / "schedule5.s"
    object_path = tmp_path / "schedule5.o"
    code_object = tmp_path / "schedule5.hsaco"
    source = BackwardKernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert source.count("s_waitcnt vmcnt(6) lgkmcnt(2)") == 1
    assert source.count("s_waitcnt vmcnt(2) lgkmcnt(2)") == 1
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 212
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 8192


def test_writer_builds_128x64_geometry(tmp_path: Path) -> None:
    pilot = BackwardSolution.pilot()
    tile_128x64 = replace(
        pilot,
        matrix_instruction=(16, 16, 16, 1, 1, 2, 4, 4, 1),
        macro_tile1=64,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32768, 2048, 8192),
        tile_128x64,
    )
    assert validate_solution(key) == ()

    toolchain = Toolchain.discover()
    assembly = tmp_path / "tile_128x64.s"
    object_path = tmp_path / "tile_128x64.o"
    code_object = tmp_path / "tile_128x64.hsaco"
    source = BackwardKernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert source.count("v_wmma_f32_16x16x16_bf16") == 16
    assert source.count("ds_store_b16_d16_hi") == 16
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 140
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 4096
    assert inspection.barrier_count == 2


def test_writer_builds_64x128_geometry(tmp_path: Path) -> None:
    pilot = BackwardSolution.pilot()
    tile_64x128 = replace(
        pilot,
        matrix_instruction=(16, 16, 16, 1, 1, 1, 8, 4, 1),
        macro_tile0=64,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32768, 2048, 8192),
        tile_64x128,
    )
    assert validate_solution(key) == ()

    toolchain = Toolchain.discover()
    assembly = tmp_path / "tile_64x128.s"
    object_path = tmp_path / "tile_64x128.o"
    code_object = tmp_path / "tile_64x128.hsaco"
    source = BackwardKernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert source.count("v_wmma_f32_16x16x16_bf16") == 16
    assert source.count("ds_store_b16_d16_hi") == 32
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 132
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 8192
    assert inspection.barrier_count == 2


def test_writer_builds_depth_u64_geometry(tmp_path: Path) -> None:
    pilot = BackwardSolution.pilot()
    depth_u64 = replace(
        pilot,
        depth_u=64,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32768, 2048, 8192),
        depth_u64,
    )
    assert validate_solution(key) == ()

    toolchain = Toolchain.discover()
    assembly = tmp_path / "depth_u64.s"
    object_path = tmp_path / "depth_u64.o"
    code_object = tmp_path / "depth_u64.hsaco"
    source = BackwardKernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert source.count("v_wmma_f32_16x16x16_bf16") == 64
    assert source.count("ds_store_b16_d16_hi") == 64
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 232
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 16384
    assert inspection.barrier_count == 2


def test_writer_prefetches_next_local_read_pair(tmp_path: Path) -> None:
    pilot = BackwardSolution.pilot()
    prefetched = replace(
        pilot,
        schedule_iter_alg=3,
        prefetch_local_read=2,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32768, 2048, 8192),
        prefetched,
    )
    assert validate_solution(key) == ()

    toolchain = Toolchain.discover()
    assembly = tmp_path / "prefetch_local_read.s"
    object_path = tmp_path / "prefetch_local_read.o"
    code_object = tmp_path / "prefetch_local_read.hsaco"
    source = BackwardKernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert source.count("s_waitcnt vmcnt(2) lgkmcnt(6)") == 2
    assert source.count("s_waitcnt lgkmcnt(6)") == 4
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 212
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 8192


def test_writer_maps_grouped_m_launch_coordinates() -> None:
    pilot = BackwardSolution.pilot()
    all_m = replace(pilot, work_group_mapping=256)
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32768, 2048, 8192),
        all_m,
    )
    assert validate_solution(key) == ()

    source = BackwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert ".amdhsa_system_sgpr_workgroup_id_z 1" in source
    assert "s_mul_i32 s4, s4, 256" in source
    assert "s_add_u32 s2, s4, s2" in source


def test_validation_rejects_unsupported_256x128_sia5_geometry() -> None:
    pilot = BackwardSolution.pilot()
    geometry = replace(
        pilot,
        work_group=(32, 8, 1),
        matrix_instruction=(16, 16, 16, 1, 1, 2, 8, 8, 1),
        macro_tile0=256,
        schedule_iter_alg=5,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32768, 2048, 8192),
        geometry,
    )
    reasons = validate_solution(key)
    assert any(
        reason.rule_id == "solution.scheduleiteralg.geometry" for reason in reasons
    )


def test_writer_builds_true_decoded_b_pipeline(tmp_path: Path) -> None:
    pipeline = replace(
        BackwardSolution.pilot(),
        one_lds_buffer=0,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
        store_priority_opt=False,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32768, 2048, 512),
        pipeline,
    )
    assert validate_solution(key) == ()
    assert pipeline.lds_num_bytes == 16384

    toolchain = Toolchain.discover()
    assembly = tmp_path / "decoded_b_pipeline.s"
    object_path = tmp_path / "decoded_b_pipeline.o"
    code_object = tmp_path / "decoded_b_pipeline.hsaco"
    source = BackwardKernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert "Prime decoded B0 and current-tile A." in source
    assert "Issue next packed tile behind current-tile A." in source
    assert "Reload next-tile A0 after current A0 becomes dead." in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 64
    assert source.count("s_barrier") == 2
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 212
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 16384
    assert inspection.wmma_count == 64
    assert inspection.barrier_count == 2


def test_writer_specializes_single_tile_decoded_b_pipeline(tmp_path: Path) -> None:
    pipeline = replace(
        BackwardSolution.pilot(),
        one_lds_buffer=0,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
        store_priority_opt=False,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(128, 2048, 32),
        pipeline,
    )
    toolchain = Toolchain.discover()
    assembly = tmp_path / "single_tile_pipeline.s"
    object_path = tmp_path / "single_tile_pipeline.o"
    code_object = tmp_path / "single_tile_pipeline.hsaco"
    source = BackwardKernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert ".LDecodedBPipelineLoop:" not in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    assert source.count("s_barrier") == 1
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.wmma_count == 32
    assert inspection.barrier_count == 1


def test_two_lds_buffers_reject_unsupported_schedule() -> None:
    unsupported = replace(BackwardSolution.pilot(), one_lds_buffer=0)
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32768, 2048, 512),
        unsupported,
    )
    rule_ids = {reason.rule_id for reason in validate_solution(key)}
    assert "solution.1ldsbuffer.pipeline" in rule_ids


@pytest.mark.parametrize(
    ("mode", "wmma_count"),
    ((BackwardDiagnosticMode.WMMA_FLOOR, 32), (BackwardDiagnosticMode.DECODE_FLOOR, 0)),
)
def test_writer_builds_lower_bound_diagnostics(
    tmp_path: Path,
    mode: BackwardDiagnosticMode,
    wmma_count: int,
) -> None:
    solution = replace(
        BackwardSolution.pilot(),
        one_lds_buffer=0,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
        store_priority_opt=False,
    )
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(32768, 2048, 512),
        solution,
    )
    toolchain = Toolchain.discover()
    assembly = tmp_path / f"{mode.value}.s"
    object_path = tmp_path / f"{mode.value}.o"
    code_object = tmp_path / f"{mode.value}.hsaco"
    source = BackwardKernelWriterAssembly(
        key,
        toolchain,
        diagnostic_mode=mode,
    ).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    inspection = inspect_artifact(
        key,
        code_object,
        toolchain,
        expected_wmma_count=wmma_count,
        expected_barrier_count=1,
    )
    assert inspection.vgpr_count == 212
    assert inspection.lds_num_bytes == 16384
    assert inspection.wmma_count == wmma_count
    assert inspection.barrier_count == 1


@pytest.mark.parametrize(
    ("mode", "wmma_count"),
    ((BackwardDiagnosticMode.WMMA_FLOOR, 32), (BackwardDiagnosticMode.DECODE_FLOOR, 0)),
)
def test_q8_one_buffer_lower_bound_diagnostics(
    tmp_path: Path,
    mode: BackwardDiagnosticMode,
    wmma_count: int,
) -> None:
    key = SolutionKey(
        ProblemType.mmq_backward("Q8_0"),
        ProblemSize(128, 4096, 1024),
        replace(
            BackwardSolution.pilot(),
            prefetch_global_read=2,
            schedule_iter_alg=5,
            lds_pad_b=8,
        ),
    )
    toolchain = Toolchain.discover()
    assembly = tmp_path / f"q8_{mode.value}.s"
    object_path = tmp_path / f"q8_{mode.value}.o"
    code_object = tmp_path / f"q8_{mode.value}.hsaco"
    BackwardKernelWriterAssembly(key, toolchain, diagnostic_mode=mode).write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    inspection = inspect_artifact(
        key,
        code_object,
        toolchain,
        expected_wmma_count=wmma_count,
        expected_barrier_count=1,
    )
    assert inspection.vgpr_count == 202
    assert inspection.lds_num_bytes == 10240
    assert inspection.wmma_count == wmma_count
    assert inspection.barrier_count == 1


def test_writer_can_disable_store_priority() -> None:
    pilot = BackwardSolution.pilot()
    no_store_priority = replace(pilot, store_priority_opt=False)
    key = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(128, 2048, 512),
        no_store_priority,
    )
    assert validate_solution(key) == ()

    source = BackwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "s_setprio" not in source
