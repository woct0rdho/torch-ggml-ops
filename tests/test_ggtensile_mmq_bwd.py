import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools.ggtensile.campaign import (
    CampaignError,
    load_inventory,
    load_solution_catalog,
)
from tools.ggtensile.cli import main as ggtensile_cli_main
from tools.ggtensile.inspection import inspect_artifact
from tools.ggtensile.kernel_writer_assembly_mmq_bwd import (
    BackwardDiagnosticMode,
    BackwardKernelWriterAssembly,
)
from tools.ggtensile.model import (
    BackwardSolution,
    ProblemSize,
    ProblemType,
    SolutionKey,
)
from tools.ggtensile.toolchain import Toolchain, ToolchainError
from tools.ggtensile.validation import validate_solution
from tools.run_ggtensile_mmq_bwd_campaign import main as campaign_main

_CONFIG_DIR = Path("tools/ggtensile/configs")
_Q4_INVENTORY = _CONFIG_DIR / "mmq_bwd_q4_k_inventory.json"
_Q4_CATALOG = _CONFIG_DIR / "mmq_bwd_q4_k_solutions.json"


def _pilot_key() -> SolutionKey:
    return SolutionKey(
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(128, 2048, 512),
        BackwardSolution.pilot(),
    )


def _toolchain() -> Toolchain:
    try:
        return Toolchain.discover()
    except ToolchainError as error:
        pytest.skip(str(error))


def test_q4_k_campaign_inventory_is_exact_and_versionless() -> None:
    inventory = load_inventory(_Q4_INVENTORY)
    catalog = load_solution_catalog(
        _Q4_CATALOG,
        problem_type=inventory.problem_type,
    )
    assert len(inventory.entries) == 12
    assert {entry.family for entry in inventory.entries} == {
        "narrow",
        "shared_down",
        "attention_output",
        "query",
    }
    assert {entry.problem_size.m for entry in inventory.entries} == {
        2048,
        8192,
        32768,
    }
    assert all(entry.current_status == "selected" for entry in inventory.entries)
    assert {entry.selected_solution for entry in inventory.entries} == set(catalog)
    assert all(
        validate_solution(
            entry.solution_key(inventory.problem_type, catalog[entry.selected_solution])
        )
        == ()
        for entry in inventory.entries
    )
    shared_down = next(
        entry
        for entry in inventory.entries
        if entry.problem_size == ProblemSize(2048, 512, 2048)
    )
    assert shared_down.expected_logical_weight_shape == (2048, 512)
    assert shared_down.expected_physical_weight_shape == (2048, 288)


def test_q5_k_campaign_inventory_is_exact_and_quant_aware() -> None:
    inventory = load_inventory(
        Path("tools/ggtensile/configs/mmq_bwd_q5_k_inventory.json")
    )
    catalog = load_solution_catalog(
        Path("tools/ggtensile/configs/mmq_bwd_q5_k_solutions.json"),
        problem_type=inventory.problem_type,
    )
    assert inventory.problem_type == ProblemType.mmq_backward_q5_k()
    assert len(inventory.entries) == 6
    assert {entry.family for entry in inventory.entries} == {"narrow", "shared_down"}
    assert {entry.problem_size.m for entry in inventory.entries} == {
        2048,
        8192,
        32768,
    }
    assert all(entry.quant_data_type == "Q5_K" for entry in inventory.entries)
    assert all(entry.current_status == "selected" for entry in inventory.entries)
    assert {entry.selected_solution for entry in inventory.entries} == set(catalog)
    assert all(
        validate_solution(
            entry.solution_key(inventory.problem_type, catalog[entry.selected_solution])
        )
        == ()
        for entry in inventory.entries
    )
    narrow = next(entry for entry in inventory.entries if entry.family == "narrow")
    shared_down = next(
        entry for entry in inventory.entries if entry.family == "shared_down"
    )
    assert narrow.expected_physical_weight_shape == (512, 1408)
    assert shared_down.expected_physical_weight_shape == (2048, 352)


def test_q6_k_campaign_inventory_covers_exact_lm_head_chunks() -> None:
    inventory = load_inventory(
        Path("tools/ggtensile/configs/mmq_bwd_q6_k_inventory.json")
    )
    catalog = load_solution_catalog(
        Path("tools/ggtensile/configs/mmq_bwd_q6_k_solutions.json"),
        problem_type=inventory.problem_type,
    )
    assert inventory.problem_type == ProblemType.mmq_backward_q6_k()
    assert len(inventory.entries) == 3
    assert {entry.family for entry in inventory.entries} == {"lm_head"}
    assert {entry.problem_size.m for entry in inventory.entries} == {64, 128, 256}
    assert all(entry.quant_data_type == "Q6_K" for entry in inventory.entries)
    assert all(entry.current_status == "selected" for entry in inventory.entries)
    assert {entry.selected_solution for entry in inventory.entries} == set(catalog)
    assert all(
        validate_solution(
            entry.solution_key(inventory.problem_type, catalog[entry.selected_solution])
        )
        == ()
        for entry in inventory.entries
    )
    assert all(
        entry.expected_logical_weight_shape == (248320, 2048)
        for entry in inventory.entries
    )
    assert all(
        entry.expected_physical_weight_shape == (248320, 1680)
        for entry in inventory.entries
    )
    retained_m64 = catalog["retained_m64x32x64_pad8_vopd"]
    retained_m128 = catalog["retained_m128x32x64_pad8_vopd"]
    retained_m256 = catalog["retained_m256x64x32_next_pad8_vopd"]
    assert isinstance(retained_m64, BackwardSolution)
    assert isinstance(retained_m128, BackwardSolution)
    assert isinstance(retained_m256, BackwardSolution)
    assert retained_m64.q6_k_extraction == "packed_vopd"
    assert retained_m128.macro_tile1 == 32
    assert retained_m128.depth_u == 64
    assert retained_m256.prefetch_packed_weight_next is True


def test_q3_k_campaign_inventory_selects_exact_padded_geometries() -> None:
    inventory = load_inventory(
        Path("tools/ggtensile/configs/mmq_bwd_q3_k_inventory.json")
    )
    catalog = load_solution_catalog(
        Path("tools/ggtensile/configs/mmq_bwd_q3_k_solutions.json"),
        problem_type=inventory.problem_type,
    )
    assert inventory.problem_type == ProblemType.mmq_backward("Q3_K")
    assert len(inventory.entries) == 6
    assert all(entry.current_status == "selected" for entry in inventory.entries)
    assert {entry.selected_solution for entry in inventory.entries} == set(catalog)
    assert all(
        validate_solution(
            entry.solution_key(inventory.problem_type, catalog[entry.selected_solution])
        )
        == ()
        for entry in inventory.entries
    )
    narrow = next(entry for entry in inventory.entries if entry.family == "narrow")
    assert narrow.expected_physical_weight_shape == (512, 880)


def test_q8_0_campaign_inventory_covers_ordinary_and_lm_head_keys() -> None:
    inventory = load_inventory(
        Path("tools/ggtensile/configs/mmq_bwd_q8_0_inventory.json")
    )
    catalog = load_solution_catalog(
        Path("tools/ggtensile/configs/mmq_bwd_q8_0_solutions.json"),
        problem_type=inventory.problem_type,
    )
    assert inventory.problem_type == ProblemType.mmq_backward_q8_0()
    assert len(inventory.entries) == 23
    assert {entry.problem_size.m for entry in inventory.entries} == {
        32,
        64,
        128,
        256,
        512,
        2048,
        8192,
        32768,
    }
    ordinary = tuple(entry for entry in inventory.entries if entry.family != "lm_head")
    lm_head = tuple(entry for entry in inventory.entries if entry.family == "lm_head")
    assert len(ordinary) == 18
    assert len(lm_head) == 5
    assert {
        sum(entry.call_count for entry in ordinary if entry.problem_size.m == m)
        for m in (2048, 8192, 32768)
    } == {301}
    assert all(entry.current_status == "selected" for entry in ordinary)
    assert all(entry.current_status == "selected" for entry in lm_head)
    assert {entry.selected_solution for entry in inventory.entries} == set(catalog)
    assert all(
        validate_solution(
            entry.solution_key(inventory.problem_type, catalog[entry.selected_solution])
        )
        == ()
        for entry in inventory.entries
    )
    q_a = next(entry for entry in ordinary if entry.family == "attention_q_a")
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
        ),
    )
    assert validate_solution(key) == ()
    source = BackwardKernelWriterAssembly(key, _toolchain()).source()
    assert "v_dual_mul_f32" in source
    assert "v_dual_sub_f32" in source
    assert source.count("v_dual_mul_f32") >= 4
    assert "v_lshl_or_b32" in source
    assert "v_mul_f32" not in source


def test_q6_k_packed_vopd_decoder_pairs_adjacent_values() -> None:
    inventory = load_inventory(
        Path("tools/ggtensile/configs/mmq_bwd_q6_k_inventory.json")
    )
    catalog = load_solution_catalog(
        Path("tools/ggtensile/configs/mmq_bwd_q6_k_solutions.json"),
        problem_type=inventory.problem_type,
    )
    entry = next(item for item in inventory.entries if item.problem_size.m == 64)
    key = entry.solution_key(inventory.problem_type, catalog[entry.selected_solution])
    assert isinstance(key.solution, BackwardSolution)
    assert key.solution.q6_k_extraction == "packed_vopd"
    assert validate_solution(key) == ()
    source = BackwardKernelWriterAssembly(key, _toolchain()).source()
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
        ProblemType.mmq_backward_q6_k(),
        ProblemSize(64, 2048, 248320),
        solution,
    )
    assert any(
        reason.rule_id == "problem_size.decoder_rows.empty"
        for reason in validate_solution(key)
    )


def test_lds_row_padding_is_strict_and_quant_aware() -> None:
    padded = replace(BackwardSolution.pilot(), lds_pad_b=8, lds_swizzle_chunk_b=0)
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


def test_quant_types_have_distinct_problem_identity() -> None:
    q4 = ProblemType.mmq_backward_q4_k()
    q5 = ProblemType.mmq_backward_q5_k()
    q6 = ProblemType.mmq_backward_q6_k()
    q8 = ProblemType.mmq_backward_q8_0()
    assert len({q4, q5, q6, q8}) == 4
    q4_key = SolutionKey(q4, ProblemSize(128, 2048, 512), BackwardSolution.pilot())
    q5_key = SolutionKey(q5, ProblemSize(128, 2048, 512), BackwardSolution.pilot())
    q6_key = SolutionKey(q6, ProblemSize(128, 2048, 248320), BackwardSolution.pilot())
    q8_key = SolutionKey(q8, ProblemSize(128, 4096, 1024), BackwardSolution.pilot())
    assert len({q4_key.hash, q5_key.hash, q6_key.hash, q8_key.hash}) == 4
    assert "mmq_bwd_q4_k_" in q4_key.kernel_name
    assert "mmq_bwd_q5_k_" in q5_key.kernel_name
    assert "mmq_bwd_q6_k_" in q6_key.kernel_name
    assert "mmq_bwd_q8_0_" in q8_key.kernel_name
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


def test_q4_k_campaign_inventory_rejects_schema_version(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.json"
    value = json.loads(
        Path("tools/ggtensile/configs/mmq_bwd_q4_k_inventory.json").read_text()
    )
    value["SchemaVersion"] = 1
    inventory_path.write_text(json.dumps(value))
    with pytest.raises(CampaignError, match="invalid inventory keys"):
        load_inventory(inventory_path)


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
    assert summary["SolutionCatalog"] == str(
        Path("tools/ggtensile/configs/mmq_bwd_q4_k_solutions.json").resolve()
    )
    assert "Solution" not in summary
    assert len(summary["Entries"]) == 1
    assert summary["Entries"][0]["SelectedSolution"] == ("retained_128x128_pipeline")
    artifact = root / "m2048_n512_k2048"
    assert (artifact / "generate.json").is_file()
    assert (artifact / "build.json").is_file()
    assert (artifact / "inspect.json").is_file()
    assert campaign_main(arguments) == 2


def test_q4_k_campaign_prepare_accepts_explicit_solution(tmp_path: Path) -> None:
    solution_path = tmp_path / "solution.json"
    solution = load_solution_catalog(
        _Q4_CATALOG,
        problem_type=ProblemType.mmq_backward_q4_k(),
    )["retained_128x128_pipeline"]
    solution_path.write_text(json.dumps(solution.to_mapping()))
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
    assert "SolutionCatalog" not in summary


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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(128, n, 512),
        BackwardSolution.pilot(),
    )
    assert validate_solution(key) == ()
    writer = BackwardKernelWriterAssembly(key, _toolchain())
    source = writer.source()
    temporary = writer.registers.temporary
    assert f"v_mul_lo_u32 v{temporary}, {packed_row_bytes}, v{temporary}" in source


def test_writer_strength_reduces_power_of_two_row_strides() -> None:
    toolchain = _toolchain()
    pipeline = replace(
        BackwardSolution.pilot(),
        one_lds_buffer=0,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
        store_priority_opt=False,
    )
    production_key = SolutionKey(
        ProblemType.mmq_backward_q4_k(),
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(128, 2048, 96),
        pipeline,
    )
    reduced_writer = BackwardKernelWriterAssembly(reduced_key, toolchain)
    reduced_source = reduced_writer.source()
    assert (
        f"v_mul_lo_u32 v{reduced_writer.registers.address + 4}, 192, "
        f"v{reduced_writer.registers.address + 4}" in reduced_source
    )


def test_validation_rejects_nonproduction_n() -> None:
    key = SolutionKey(
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(128, 1024, 512),
        BackwardSolution.pilot(),
    )
    assert {reason.rule_id for reason in validate_solution(key)} == {
        "problem_size.n.production"
    }


def test_writer_enables_and_flattens_packed_workitem_xy() -> None:
    writer = BackwardKernelWriterAssembly(_pilot_key(), _toolchain())
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


def test_build_and_inspect_pilot(tmp_path: Path) -> None:
    key = _pilot_key()
    toolchain = _toolchain()
    assembly = tmp_path / "pilot.s"
    object_path = tmp_path / "pilot.o"
    code_object = tmp_path / "pilot.hsaco"
    BackwardKernelWriterAssembly(key, toolchain).write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.target == "gfx1151"
    assert inspection.code_object_version == 5
    assert inspection.vgpr_count == 192
    assert inspection.sgpr_count == 16
    assert inspection.max_vgpr_index == 191
    assert inspection.max_sgpr_index == 15
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0
    assert inspection.wmma_count == 32
    assert inspection.barrier_count == 2
    assert inspection.valu_issue_count > 0
    assert inspection.valu_operation_count > inspection.valu_issue_count
    assert inspection.vopd_count > 0
    assert inspection.vmem_count > 0
    assert inspection.lds_count > 0
    assert inspection.wait_count > 0
    assert inspection.clause_count == 0
    assert inspection.delay_alu_count == 0
    assert inspection.buffer_gl0_inv_count == 0


def test_build_and_inspect_q5_k_payload_backend(tmp_path: Path) -> None:
    key = SolutionKey(
        ProblemType.mmq_backward_q5_k(),
        ProblemSize(128, 2048, 512),
        BackwardSolution.pilot(),
    )
    toolchain = _toolchain()
    assembly = tmp_path / "q5_k.s"
    object_path = tmp_path / "q5_k.o"
    code_object = tmp_path / "q5_k.hsaco"
    source = BackwardKernelWriterAssembly(key, toolchain).write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert len(source) == 64
    text = assembly.read_text()
    assert "Build Q5_K payload" in text
    assert "Decode Q5_K low nibbles and high payload bits into LDS." in text
    assert "offset:48" in text
    assert "0x01010101" in text
    assert "v_lshl_or_b32" in text
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 200
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 8192
    assert inspection.wmma_count == 32
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


def test_build_and_inspect_q8_0_payload_backend(tmp_path: Path) -> None:
    key = SolutionKey(
        ProblemType.mmq_backward_q8_0(),
        ProblemSize(128, 4096, 1024),
        BackwardSolution.pilot(),
    )
    toolchain = _toolchain()
    assembly = tmp_path / "q8_0.s"
    object_path = tmp_path / "q8_0.o"
    code_object = tmp_path / "q8_0.hsaco"
    BackwardKernelWriterAssembly(key, toolchain).write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    text = assembly.read_text()
    assert "Build Q8_0 block addresses" in text
    assert "Decode Q8_0 signed int8 payload and scale" in text
    assert "global_load_d16_b16" in text
    assert "global_load_b128" in text
    assert "v_cvt_f32_i32" in text
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 186
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 8192
    assert inspection.wmma_count == 32
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


def test_build_and_inspect_q8_0_depth_u64_backend(tmp_path: Path) -> None:
    solution = replace(
        BackwardSolution.pilot(),
        depth_u=64,
        prefetch_global_read=2,
        schedule_iter_alg=4,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward_q8_0(),
        ProblemSize(128, 4096, 1024),
        solution,
    )
    assert validate_solution(key) == ()
    toolchain = _toolchain()
    assembly = tmp_path / "q8_0_depth_u64.s"
    object_path = tmp_path / "q8_0_depth_u64.o"
    code_object = tmp_path / "q8_0_depth_u64.hsaco"
    writer = BackwardKernelWriterAssembly(key, toolchain)
    writer.write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    text = assembly.read_text()
    assert text.count("global_load_d16_b16") == 4
    assert text.count("ds_store_b16_d16_hi") == 64
    assert "Precompute XOR-8 LDS store bases" in text
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 220
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 16384
    assert inspection.wmma_count == 64
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


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
        ProblemType.mmq_backward_q8_0(),
        ProblemSize(2048, 4096, 1024),
        solution,
    )
    assert validate_solution(key) == ()
    toolchain = _toolchain()
    assembly = tmp_path / "q8_0_compact.s"
    object_path = tmp_path / "q8_0_compact.o"
    code_object = tmp_path / "q8_0_compact.hsaco"
    BackwardKernelWriterAssembly(key, toolchain).write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 231
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 5120
    assert inspection.wmma_count == 32
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


def test_build_and_inspect_q6_k_backend(tmp_path: Path) -> None:
    solution = replace(
        BackwardSolution.pilot(),
        matrix_instruction=(16, 16, 16, 1, 1, 2, 4, 4, 1),
        macro_tile0=128,
        macro_tile1=64,
        prefetch_global_read=2,
        schedule_iter_alg=5,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.mmq_backward_q6_k(),
        ProblemSize(128, 2048, 248320),
        solution,
    )
    assert validate_solution(key) == ()
    toolchain = _toolchain()
    assembly = tmp_path / "q6_k_m128.s"
    object_path = tmp_path / "q6_k_m128.o"
    code_object = tmp_path / "q6_k_m128.hsaco"
    BackwardKernelWriterAssembly(key, toolchain).write(assembly)
    source = assembly.read_text()
    assert "Build Q6_K block addresses" in source
    assert "global_load_b128" in source
    assert "v_lshl_or_b32" in source
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 142
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 4096
    assert inspection.wmma_count == 16
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


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
        ProblemType.mmq_backward_q8_0(),
        ProblemSize(64, 4096, 129280),
        solution,
    )
    assert validate_solution(key) == ()
    q4_reasons = validate_solution(
        SolutionKey(
            ProblemType.mmq_backward_q4_k(),
            ProblemSize(64, 4096, 129280),
            solution,
        )
    )
    assert any(reason.rule_id == "solution.depthu64.q8_pad8" for reason in q4_reasons)
    toolchain = _toolchain()
    assembly = tmp_path / "q8_0_m64_depth_u64.s"
    object_path = tmp_path / "q8_0_m64_depth_u64.o"
    code_object = tmp_path / "q8_0_m64_depth_u64.hsaco"
    BackwardKernelWriterAssembly(key, toolchain).write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 90
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 9216
    assert inspection.wmma_count == 16
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


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
        ProblemType.mmq_backward_q8_0(),
        ProblemSize(32, 4096, 129280),
        solution,
    )
    assert validate_solution(key) == ()
    assert validate_solution(
        SolutionKey(
            ProblemType.mmq_backward_q4_k(),
            ProblemSize(32, 4096, 129280),
            solution,
        )
    )
    toolchain = _toolchain()
    assembly = tmp_path / "q8_0_m32.s"
    object_path = tmp_path / "q8_0_m32.o"
    code_object = tmp_path / "q8_0_m32.hsaco"
    BackwardKernelWriterAssembly(key, toolchain).write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.max_flat_workgroup_size == 64
    assert inspection.vgpr_count == 90
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 5120
    assert inspection.wmma_count == 8
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


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
        ProblemType.mmq_backward_q8_0(),
        ProblemSize(32, 4096, 129280),
        solution,
    )
    assert validate_solution(key) == ()
    toolchain = _toolchain()
    assembly = tmp_path / "q8_0_m32_vopd.s"
    object_path = tmp_path / "q8_0_m32_vopd.o"
    code_object = tmp_path / "q8_0_m32_vopd.hsaco"
    BackwardKernelWriterAssembly(key, toolchain).write(assembly)
    source = assembly.read_text()
    assert "v_dual_mul_f32" in source
    assert "global_load_b128" in source
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 106
    assert inspection.lds_num_bytes == 9216
    assert inspection.vopd_count > 40
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


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
    assert "SchemaVersion" not in generate
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


def test_cli_records_rejected_solution_manifest(tmp_path: Path) -> None:
    rejected = SolutionKey(
        ProblemType.mmq_backward_q4_k(),
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(128, 2048, 512),
        swizzled,
    )
    assert validate_solution(key) == ()

    source = BackwardKernelWriterAssembly(key, _toolchain()).source()
    assert "Precompute XOR-8 LDS store bases" in source
    assert "v_xor_b32" in source
    assert source.count("ds_load_b128") == 32
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32


def test_writer_emits_distinct_xor16_lds_layout() -> None:
    pilot = BackwardSolution.pilot()
    swizzled = replace(pilot, lds_swizzle_chunk_b=16)
    key = SolutionKey(
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(128, 2048, 512),
        swizzled,
    )
    assert validate_solution(key) == ()

    source = BackwardKernelWriterAssembly(key, _toolchain()).source()
    assert "Precompute XOR-16 LDS store bases" in source
    assert source.count("ds_load_b128") == 32
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32


def test_writer_emits_distinct_xor4_lds_layout() -> None:
    pilot = BackwardSolution.pilot()
    swizzled = replace(pilot, lds_swizzle_chunk_b=4, schedule_iter_alg=3)
    key = SolutionKey(
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(128, 2048, 512),
        swizzled,
    )
    assert validate_solution(key) == ()

    source = BackwardKernelWriterAssembly(key, _toolchain()).source()
    assert "Precompute XOR-4 LDS store bases" in source
    assert source.count("ds_load_b64") == 64
    assert source.count("ds_load_b128") == 0
    assert source.count("s_waitcnt vmcnt(2) lgkmcnt(4)") == 2
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32


def test_writer_emits_sia3_partial_wait_schedule() -> None:
    pilot = BackwardSolution.pilot()
    scheduled = replace(pilot, schedule_iter_alg=3)
    key = SolutionKey(
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(128, 2048, 512),
        scheduled,
    )
    assert validate_solution(key) == ()

    source = BackwardKernelWriterAssembly(key, _toolchain()).source()
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        scheduled,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        prefetched,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        shared,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        pipelined,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        scheduled,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        tile_128x64,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        tile_64x128,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        depth_u64,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        prefetched,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        all_m,
    )
    assert validate_solution(key) == ()

    source = BackwardKernelWriterAssembly(key, _toolchain()).source()
    assert ".amdhsa_system_sgpr_workgroup_id_z 1" in source
    assert "s_mul_i32 s4, s4, 256" in source
    assert "s_add_u32 s2, s4, s2" in source


def test_writer_builds_q3_k_padded_256x64_geometry(tmp_path: Path) -> None:
    catalog = load_solution_catalog(
        Path("tools/ggtensile/configs/mmq_bwd_q3_k_solutions.json"),
        problem_type=ProblemType.mmq_backward("Q3_K"),
    )
    geometry = catalog["retained_256x64_pad8_sia5"]
    key = SolutionKey(
        ProblemType.mmq_backward("Q3_K"),
        ProblemSize(2048, 2048, 512),
        geometry,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
    assembly = tmp_path / "geometry.s"
    object_path = tmp_path / "geometry.o"
    code_object = tmp_path / "geometry.hsaco"
    source = BackwardKernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert "Precompute A row coordinates shared by every DepthU iteration." in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    assert source.count("ds_store_b16_d16_hi") == 16
    assert source.count("ds_load_b128") == 16
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 243
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 5120


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
        ProblemType.mmq_backward_q4_k(),
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 512),
        pipeline,
    )
    assert validate_solution(key) == ()
    assert pipeline.lds_num_bytes == 16384

    toolchain = _toolchain()
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(128, 2048, 32),
        pipeline,
    )
    toolchain = _toolchain()
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
        ProblemType.mmq_backward_q4_k(),
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 512),
        solution,
    )
    toolchain = _toolchain()
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
        ProblemType.mmq_backward_q8_0(),
        ProblemSize(128, 4096, 1024),
        replace(
            BackwardSolution.pilot(),
            prefetch_global_read=2,
            schedule_iter_alg=5,
            lds_pad_b=8,
        ),
    )
    toolchain = _toolchain()
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
        ProblemType.mmq_backward_q4_k(),
        ProblemSize(128, 2048, 512),
        no_store_priority,
    )
    assert validate_solution(key) == ()

    source = BackwardKernelWriterAssembly(key, _toolchain()).source()
    assert "s_setprio" not in source
