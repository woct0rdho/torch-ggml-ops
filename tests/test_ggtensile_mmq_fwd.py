import builtins
import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from pathlib import Path
from types import ModuleType

import pytest

from tools.ggtensile.campaign import load_inventory, load_solution_catalog
from tools.ggtensile.inspection import inspect_artifact
from tools.ggtensile.kernel_writer_assembly_mmq_fwd import (
    DenseForwardKernelWriterAssembly,
    ForwardKernelWriterError,
)
from tools.ggtensile.model import (
    DenseForwardSolution,
    ProblemSize,
    ProblemType,
    SchemaError,
    SolutionKey,
)
from tools.ggtensile.runtime import DenseForwardModule, FixedHipDenseForwardModule
from tools.ggtensile.toolchain import Toolchain, ToolchainError
from tools.ggtensile.validation import validate_solution

_CONFIG_DIR = Path("tools/ggtensile/configs")
_INVENTORY = _CONFIG_DIR / "mmq_fwd_q4_k_dense_inventory.json"
_CATALOG = _CONFIG_DIR / "mmq_fwd_q4_k_open_solutions.json"


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
        ProblemType.dense_mmq_forward_q4_k(),
        size or ProblemSize(2048, 512, 2048),
        solution or DenseForwardSolution.q4_k_pilot(),
    )


def test_forward_inventory_is_exact_open_and_versionless() -> None:
    inventory = load_inventory(_INVENTORY)
    catalog = load_solution_catalog(_CATALOG, problem_type=inventory.problem_type)
    assert inventory.problem_type == ProblemType.dense_mmq_forward_q4_k()
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
    assert sum(entry.current_status == "selected" for entry in inventory.entries) == 2
    assert sum(entry.current_status == "open" for entry in inventory.entries) == 10
    assert set(catalog) == {
        "pilot_direct_global",
        "retained_metadata_after_low_wmma",
        "retained_independent_extraction_metadata_after_low_wmma",
        "retained_shared_down_m8192_extract_a1d4_p2",
        "retained_shared_down_m32768_extract_a1d2_p2",
        "composed_shared_down_m2048_extract_a1d2_p2",
        "composed_shared_down_m8192_extract_a1d4_p2",
        "composed_shared_down_m32768_extract_a1d2_p2",
        "composed_narrow_m32768_extract_a4d4_p2",
        "composed_query_m2048_extract_a1d2_p2",
        "composed_query_m8192_extract_a4d4_p2",
        "composed_query_m32768_extract_a2d2_p2",
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
    assert narrow.expected_physical_weight_shape == (512, 1152)
    raw = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    assert "Version" not in raw and "SchemaVersion" not in raw


def test_forward_solution_key_is_strict_and_round_trips() -> None:
    key = _key()
    assert SolutionKey.from_mapping(key.to_mapping()) == key
    assert "dense_fwd_q4_k" in key.kernel_name
    mapping = key.to_mapping()
    solution_mapping = mapping["Solution"]
    assert isinstance(solution_mapping, dict)
    solution = dict(solution_mapping)
    solution["Unknown"] = 1
    mapping["Solution"] = solution
    with pytest.raises(SchemaError, match="invalid Solution"):
        SolutionKey.from_mapping(mapping)


def test_forward_solution_identity_contains_every_dataclass_field() -> None:
    mapping = DenseForwardSolution.q4_k_pilot().to_mapping()
    assert len(mapping) == len(fields(DenseForwardSolution))
    assert DenseForwardSolution.from_mapping(mapping).to_mapping() == mapping


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
        ("activation_layout", "Q8_1_D4"),
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
    solution = replace(DenseForwardSolution.q4_k_pilot(), **{attribute: value})
    assert validate_solution(_key(solution=solution))


@pytest.mark.parametrize(
    "size",
    (
        ProblemSize(128, 512, 2048),
        ProblemSize(2048, 1024, 2048),
        ProblemSize(2048, 512, 2304),
    ),
)
def test_forward_validation_rejects_nonproduction_sizes(size: ProblemSize) -> None:
    assert validate_solution(_key(size=size))


def test_forward_validation_rejects_mismatched_problem_type() -> None:
    problem_type = ProblemType(
        operation_type="DenseMMQForward",
        quant_data_type="Q5_K",
        data_type_a="Q8_1_DS4",
        data_type_b="Q5_K",
        dest_data_type="BFloat16",
        compute_data_type="Float",
        transpose_a=False,
        transpose_b=True,
    )
    key = SolutionKey(
        problem_type,
        ProblemSize(2048, 512, 2048),
        DenseForwardSolution.q4_k_pilot(),
    )
    assert {reason.rule_id for reason in validate_solution(key)} >= {
        "problem_type.forward.unsupported"
    }


def test_forward_writer_emits_direct_ds4_q4_k_control(tmp_path: Path) -> None:
    key = _key()
    writer = DenseForwardKernelWriterAssembly(key, _toolchain())
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
    key = _key(solution=DenseForwardSolution.q4_k_wave_reuse())
    source = DenseForwardKernelWriterAssembly(key, _toolchain()).source()
    assert source.count("v_wmma_i32_16x16x16_iu8") == 128
    assert "v_lshrrev_b32 v159, 5, v157" in source
    assert "Reused Q4_K group 7 across eight activation tiles." in source
    assert "global_store_d16_hi_b16" in source
    assert "v_fma_mix_f32" in source
    assert "v_cvt_f32_f16 v120" not in source
    assert "s_barrier" not in source


def test_forward_writer_emits_wave_batch_control() -> None:
    key = _key(solution=DenseForwardSolution.q4_k_wave_batch4())
    source = DenseForwardKernelWriterAssembly(key, _toolchain()).source()
    assert source.count("v_wmma_i32_16x16x16_iu8") == 128
    assert "Batch Q4_K group 7 across four activation tiles at a time." in source
    assert "v_wmma_i32_16x16x16_iu8 v[112:119]" in source
    assert "v_wmma_i32_16x16x16_iu8 v[136:143]" in source
    assert "v_mov_b32 v175, v2" in source
    assert "v_mov_b32 v176, v2" not in source
    assert "v_fma_mix_f32 v148" in source
    assert "s_barrier" not in source


def test_forward_writer_emits_hip_shaped_staged_control() -> None:
    key = _key(solution=DenseForwardSolution.q4_k_hip_staged())
    source = DenseForwardKernelWriterAssembly(key, _toolchain()).source()
    assert source.count("v_wmma_i32_16x16x16_iu8") == 128
    assert source.count("s_barrier") == 4
    assert "Cooperatively stage the raw packed Q4_K payload." in source
    assert "Cooperatively stage one contiguous 128-row DS4 plane." in source
    assert "ds_write_b128" in source
    assert "ds_read_b128" in source
    assert source.count("v_dual_fmac_f32") == 512
    assert "v_fmac_f32_e32" not in source


def test_forward_writer_emits_hip_decoded_staged_control() -> None:
    key = _key(solution=DenseForwardSolution.q4_k_hip_decoded_staged())
    source = DenseForwardKernelWriterAssembly(key, _toolchain()).source()
    assert source.count("v_wmma_i32_16x16x16_iu8") == 32
    assert source.count("s_barrier") == 4
    assert "Cooperatively decode Q4_K nibbles into HIP's padded LDS rows." in source
    assert "Compute each packed Q4_K scale/min pair once per weight row." in source
    assert "Roll decoded Q4_K groups 0 through 3." in source
    assert "Roll decoded Q4_K groups 4 through 7." in source
    assert "ds_write2st64_b32" in source
    assert "ds_read2st64_b32" in source
    assert source.count("v_dual_fmac_f32") == 128
    assert "s_clause" not in source


def test_forward_writer_emits_retained_decoded_staged_control() -> None:
    key = _key(solution=DenseForwardSolution.q4_k_hip_decoded_staged_retained())
    source = DenseForwardKernelWriterAssembly(key, _toolchain()).source()
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
    solution = DenseForwardSolution.q4_k_hip_decoded_staged_metadata_after_low_wmma()
    key = _key(solution=solution)
    assert validate_solution(key) == ()
    source = DenseForwardKernelWriterAssembly(key, _toolchain()).source()
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
    solution = DenseForwardSolution.q4_k_hip_decoded_staged_independent_extraction_metadata_after_low_wmma()
    key = _key(size=size, solution=solution)
    assert validate_solution(key) == ()
    source = DenseForwardKernelWriterAssembly(key, _toolchain()).source()
    assert source.count("v_cvt_f16_u16_e32") == 16
    assert source.count("ds_write_b32 v229, v9") == 4
    assert source.count("ds_write_b32 v229, v10") == 4
    assert source.count("s_waitcnt lgkmcnt(15)") == 2


@pytest.mark.parametrize(
    ("size", "solution", "dependency_width"),
    (
        (
            ProblemSize(8192, 2048, 512),
            DenseForwardSolution.q4_k_hip_decoded_staged_shared_down_m8192(),
            4,
        ),
        (
            ProblemSize(32768, 2048, 512),
            DenseForwardSolution.q4_k_hip_decoded_staged_shared_down_m32768(),
            2,
        ),
        (
            ProblemSize(2048, 2048, 512),
            DenseForwardSolution.q4_k_hip_decoded_staged_shared_down_m2048_metadata_after_low_wmma(),
            2,
        ),
        (
            ProblemSize(8192, 2048, 512),
            DenseForwardSolution.q4_k_hip_decoded_staged_shared_down_m8192_metadata_after_low_wmma(),
            4,
        ),
        (
            ProblemSize(32768, 2048, 512),
            DenseForwardSolution.q4_k_hip_decoded_staged_shared_down_m32768_metadata_after_low_wmma(),
            2,
        ),
    ),
)
def test_forward_writer_emits_selected_shared_down_extraction(
    size: ProblemSize,
    solution: DenseForwardSolution,
    dependency_width: int,
) -> None:
    key = _key(size=size, solution=solution)
    assert validate_solution(key) == ()
    source = DenseForwardKernelWriterAssembly(key, _toolchain()).source()
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
            DenseForwardSolution.q4_k_hip_decoded_staged_narrow_m32768_metadata_after_low_wmma(),
            4,
        ),
        (
            ProblemSize(2048, 8192, 2048),
            DenseForwardSolution.q4_k_hip_decoded_staged_query_m2048_metadata_after_low_wmma(),
            2,
        ),
        (
            ProblemSize(8192, 8192, 2048),
            DenseForwardSolution.q4_k_hip_decoded_staged_query_m8192_metadata_after_low_wmma(),
            4,
        ),
        (
            ProblemSize(32768, 8192, 2048),
            DenseForwardSolution.q4_k_hip_decoded_staged_query_m32768_metadata_after_low_wmma(),
            2,
        ),
    ),
)
def test_forward_writer_emits_selected_exact_epilogue(
    size: ProblemSize,
    solution: DenseForwardSolution,
    dependency_width: int,
) -> None:
    key = _key(size=size, solution=solution)
    assert validate_solution(key) == ()
    source = DenseForwardKernelWriterAssembly(key, _toolchain()).source()
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
            DenseForwardSolution.q4_k_hip_decoded_staged_shared_down_m8192(),
        ),
        (
            ProblemSize(32768, 2048, 512),
            DenseForwardSolution.q4_k_hip_decoded_staged_shared_down_m8192(),
        ),
        (
            ProblemSize(8192, 2048, 512),
            DenseForwardSolution.q4_k_hip_decoded_staged_shared_down_m32768(),
        ),
        (
            ProblemSize(8192, 8192, 2048),
            DenseForwardSolution.q4_k_hip_decoded_staged_query_m32768_metadata_after_low_wmma(),
        ),
    ),
)
def test_forward_extraction_rejects_unmeasured_exact_key(
    size: ProblemSize,
    solution: DenseForwardSolution,
) -> None:
    key = _key(size=size, solution=solution)
    assert {reason.rule_id for reason in validate_solution(key)} >= {
        "solution.forward.metadata_schedule.key"
    }


def test_forward_runtime_uses_exact_candidate_and_hip_launch_geometry() -> None:
    direct = DenseForwardModule.__new__(DenseForwardModule)
    direct.solution_key = _key(solution=DenseForwardSolution.q4_k_wave_reuse())
    assert direct._launch_configuration() == ((8, 16, 1), (128, 1, 1), 0)
    hip = FixedHipDenseForwardModule.__new__(FixedHipDenseForwardModule)
    hip.solution_key = direct.solution_key
    assert hip._launch_configuration() == ((8, 16, 1), (32, 4, 1), 38_400)


@pytest.mark.parametrize(
    ("solution", "wmma_count", "vgpr_count", "barrier_count", "lds_num_bytes"),
    (
        (DenseForwardSolution.q4_k_pilot(), 16, 88, 0, 0),
        (DenseForwardSolution.q4_k_wave_reuse(), 128, 164, 0, 0),
        (DenseForwardSolution.q4_k_wave_batch4(), 128, 194, 0, 0),
        (DenseForwardSolution.q4_k_hip_staged(), 128, 239, 4, 26_624),
        (
            DenseForwardSolution.q4_k_hip_decoded_staged(),
            32,
            239,
            4,
            38_400,
        ),
        (
            DenseForwardSolution.q4_k_hip_decoded_staged_retained(),
            32,
            239,
            4,
            38_400,
        ),
        (
            DenseForwardSolution.q4_k_hip_decoded_staged_metadata_after_low_wmma(),
            32,
            239,
            4,
            38_400,
        ),
        (
            DenseForwardSolution.q4_k_hip_decoded_staged_independent_extraction_metadata_after_low_wmma(),
            32,
            239,
            4,
            38_400,
        ),
    ),
)
def test_forward_artifact_passes_strict_inspection(
    tmp_path: Path,
    solution: DenseForwardSolution,
    wmma_count: int,
    vgpr_count: int,
    barrier_count: int,
    lds_num_bytes: int,
) -> None:
    key = _key(solution=solution)
    toolchain = _toolchain()
    assembly = tmp_path / "kernel.s"
    object_path = tmp_path / "kernel.o"
    code_object = tmp_path / "kernel.hsaco"
    DenseForwardKernelWriterAssembly(key, toolchain).write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.wmma_count == wmma_count
    assert inspection.barrier_count == barrier_count
    assert inspection.vgpr_count == vgpr_count
    assert inspection.sgpr_count == 16
    assert inspection.lds_num_bytes == lds_num_bytes
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


@pytest.mark.parametrize(
    ("size", "solution"),
    (
        (
            ProblemSize(8192, 2048, 512),
            DenseForwardSolution.q4_k_hip_decoded_staged_shared_down_m8192(),
        ),
        (
            ProblemSize(32768, 2048, 512),
            DenseForwardSolution.q4_k_hip_decoded_staged_shared_down_m32768(),
        ),
    ),
)
def test_selected_forward_artifact_passes_strict_inspection(
    tmp_path: Path,
    size: ProblemSize,
    solution: DenseForwardSolution,
) -> None:
    key = _key(size=size, solution=solution)
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


def test_forward_writer_rejects_invalid_solution() -> None:
    key = _key(solution=replace(DenseForwardSolution.q4_k_pilot(), wmma_clamp=False))
    with pytest.raises(ForwardKernelWriterError, match="solution rejected"):
        DenseForwardKernelWriterAssembly(key, _toolchain())


def test_forward_writer_reports_missing_rocisa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = DenseForwardKernelWriterAssembly(_key(), _toolchain())
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
