import json
from pathlib import Path

import pytest

from tools.ggtensile.campaign import load_inventory, load_solution_catalog
from tools.ggtensile.inspection import inspect_artifact
from tools.ggtensile.kernel_writer_assembly_mmq_fwd import (
    DenseForwardKernelWriterAssembly,
)
from tools.ggtensile.model import (
    DenseForwardSolution,
    ProblemSize,
    ProblemType,
    SolutionKey,
)
from tools.ggtensile.runtime import FixedHipDenseForwardModule
from tools.ggtensile.toolchain import Toolchain, ToolchainError
from tools.ggtensile.validation import validate_solution

_CONFIG_DIR = Path("tools/ggtensile/configs")
_INVENTORY = _CONFIG_DIR / "mmq_fwd_q5_k_dense_inventory.json"
_CATALOG = _CONFIG_DIR / "mmq_fwd_q5_k_open_solutions.json"


def _toolchain() -> Toolchain:
    try:
        return Toolchain.discover()
    except ToolchainError as error:
        pytest.skip(str(error))


def _key(
    size: ProblemSize | None = None,
    solution: DenseForwardSolution | None = None,
) -> SolutionKey:
    return SolutionKey(
        ProblemType.dense_mmq_forward_q5_k(),
        size or ProblemSize(2048, 512, 2048),
        solution
        or DenseForwardSolution.q5_k_hip_decoded_staged_independent_extraction_metadata_after_low_wmma(),
    )


def test_q5_forward_inventory_is_exact_selected_and_versionless() -> None:
    inventory = load_inventory(_INVENTORY)
    catalog = load_solution_catalog(_CATALOG, problem_type=inventory.problem_type)
    assert inventory.problem_type == ProblemType.dense_mmq_forward_q5_k()
    assert len(inventory.entries) == 6
    assert {entry.family for entry in inventory.entries} == {
        "narrow",
        "shared_down",
    }
    assert {entry.problem_size.m for entry in inventory.entries} == {
        2048,
        8192,
        32768,
    }
    assert all(entry.current_status == "selected" for entry in inventory.entries)
    assert set(catalog) == {
        "retained_decoded_staged",
        "retained_metadata_after_low_wmma",
        "retained_independent_extraction_metadata_after_low_wmma",
        "selected_narrow_m2048_a1d8_p0",
        "selected_narrow_m8192_a8d1_p0",
        "selected_narrow_m32768_a8d4_p3_vopd_init",
        "selected_shared_down_m2048_a1d2_p2",
        "selected_shared_down_m8192_a1d4_p2_vopd_init",
        "selected_shared_down_m32768_a1d2_p2",
    }
    assert all(
        validate_solution(
            entry.solution_key(
                inventory.problem_type,
                catalog[entry.selected_solution],
            )
        )
        == ()
        for entry in inventory.entries
    )
    narrow = inventory.entries[0]
    assert narrow.expected_logical_weight_shape == (512, 2048)
    assert narrow.expected_physical_weight_shape == (512, 1408)
    raw = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    assert "Version" not in raw and "SchemaVersion" not in raw


def test_q5_forward_problem_and_solution_identity_are_quant_aware() -> None:
    q4_key = SolutionKey(
        ProblemType.dense_mmq_forward_q4_k(),
        ProblemSize(2048, 512, 2048),
        DenseForwardSolution.q4_k_hip_decoded_staged_retained(),
    )
    q5_key = _key(solution=DenseForwardSolution.q5_k_hip_decoded_staged_retained())
    assert q4_key.hash != q5_key.hash
    assert "dense_fwd_q4_k" in q4_key.kernel_name
    assert "dense_fwd_q5_k" in q5_key.kernel_name
    assert isinstance(q5_key.solution, DenseForwardSolution)
    assert q5_key.solution.packed_weight_block_bytes == 176
    assert q5_key.solution.weight_decode == "DirectNibbleHighBit"


@pytest.mark.parametrize(
    "solution",
    (
        DenseForwardSolution.q5_k_hip_decoded_staged_retained(),
        DenseForwardSolution.q5_k_hip_decoded_staged_metadata_after_low_wmma(),
        DenseForwardSolution.q5_k_hip_decoded_staged_independent_extraction_metadata_after_low_wmma(),
    ),
)
def test_q5_forward_writer_covers_high_bit_decode_and_schedule(
    solution: DenseForwardSolution,
) -> None:
    key = _key(solution=solution)
    source = DenseForwardKernelWriterAssembly(key, _toolchain()).source()
    assert "GGTensile Q5_K dense MMQ forward" in source
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
    solution = DenseForwardSolution.q5_k_hip_decoded_staged_extraction(
        epilogue_tiles_ahead=8,
        epilogue_dependency_width=4,
        epilogue_priority=3,
        accumulator_initialization="VopdPair",
    )
    key = _key(size=ProblemSize(32768, 512, 2048), solution=solution)
    assert validate_solution(key) == ()
    source = DenseForwardKernelWriterAssembly(key, _toolchain()).source()
    assert "v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, 0" in source
    assert "v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0" in source
    assert "v_dual_mov_b32 v8, v0 :: v_dual_mov_b32 v9, v1" in source
    assert "v_dual_mov_b32 v70, v0 :: v_dual_mov_b32 v71, v1" in source
    assert "v_mov_b32 v8, v0" not in source


def test_q5_forward_validation_rejects_q4_control_and_nonproduction_shape() -> None:
    q4_control = SolutionKey(
        ProblemType.dense_mmq_forward_q5_k(),
        ProblemSize(2048, 512, 2048),
        DenseForwardSolution.q4_k_hip_decoded_staged_retained(),
    )
    assert {reason.rule_id for reason in validate_solution(q4_control)} == {
        "solution.forward.control.unimplemented"
    }
    wrong_shape = _key(size=ProblemSize(2048, 2048, 4096))
    assert {reason.rule_id for reason in validate_solution(wrong_shape)} >= {
        "problem_size.nk.forward_production"
    }


def test_q5_forward_runtime_uses_exact_hip_geometry() -> None:
    hip = FixedHipDenseForwardModule.__new__(FixedHipDenseForwardModule)
    hip.solution_key = _key()
    assert hip._launch_configuration() == ((8, 16, 1), (32, 4, 1), 38_400)


def test_q5_forward_artifact_passes_strict_inspection(tmp_path: Path) -> None:
    key = _key()
    toolchain = _toolchain()
    assembly = tmp_path / "kernel.s"
    object_path = tmp_path / "kernel.o"
    code_object = tmp_path / "kernel.hsaco"
    DenseForwardKernelWriterAssembly(key, toolchain).write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.wmma_count == 32
    assert inspection.barrier_count == 4
    assert inspection.clause_count == 8
    assert inspection.vgpr_count == 239
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == 38_400
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0
