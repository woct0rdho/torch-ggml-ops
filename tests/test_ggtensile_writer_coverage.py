import builtins
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from tools.ggtensile.campaign import load_inventory, load_solution_catalog
from tools.ggtensile.kernel_writer_assembly_mmq_bwd import (
    DiagnosticMode,
    KernelWriterAssembly,
    KernelWriterError,
    _Assembly,
)
from tools.ggtensile.model import ProblemSize, ProblemType, Solution, SolutionKey
from tools.ggtensile.toolchain import Toolchain, ToolchainError
from tools.ggtensile.validation import validate_solution


def _toolchain() -> Toolchain:
    try:
        return Toolchain.discover()
    except ToolchainError as error:
        pytest.skip(str(error))


def _key(quant: str, size: tuple[int, int, int], solution: Solution) -> SolutionKey:
    return SolutionKey(
        ProblemType.dense_mmq_backward(quant),
        ProblemSize(*size),
        solution,
    )


def _source(key: SolutionKey, diagnostic_mode: DiagnosticMode | None = None) -> str:
    assert validate_solution(key) == ()
    return KernelWriterAssembly(
        key,
        _toolchain(),
        diagnostic_mode=diagnostic_mode,
    ).source()


def test_writer_emits_every_selected_production_kernel() -> None:
    catalogs = (
        ("q3_k_dense_inventory.json", "q3_k_selected_solutions.json"),
        ("q4_k_dense_inventory.json", "q4_k_selected_solutions.json"),
        ("q5_k_dense_inventory.json", "q5_k_selected_solutions.json"),
        ("q6_k_dense_inventory.json", "q6_k_selected_solutions.json"),
        ("q8_0_dense_inventory.json", "q8_0_selected_solutions.json"),
    )
    emitted: set[str] = set()
    for inventory_name, catalog_name in catalogs:
        inventory = load_inventory(Path("tools/ggtensile/configs") / inventory_name)
        catalog = load_solution_catalog(Path("tools/ggtensile/configs") / catalog_name)
        for entry in inventory.entries:
            key = entry.solution_key(
                inventory.problem_type,
                catalog[entry.selected_solution],
            )
            source = _source(key)
            assert f".globl {key.kernel_name}" in source
            emitted.add(key.hash)
    assert len(emitted) == 50


def _targeted_writer_keys() -> tuple[SolutionKey, ...]:
    pilot = Solution.pilot()
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
            ),
        ),
        _key("Q3_K", (128, 2048, 8192), pilot),
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
    source = _source(key)
    assert source.endswith(
        f".size {key.kernel_name}, .L{key.kernel_name}_end - {key.kernel_name}\n"
    )


def test_writer_emits_repaired_q6_lane_share_and_depth64_layouts() -> None:
    share_key = _targeted_writer_keys()[9]
    source = _source(share_key)
    assert source.count("v_mov_b32_dpp") == 4
    assert source.index("Load unique Q6_K low planes on every lane.") < source.index(
        "Load shared Q6_K high planes on lane-pair owners."
    )
    assert source.index(
        "Load shared Q6_K high planes on lane-pair owners."
    ) < source.index("s_and_saveexec_b32")

    depth64_key = _targeted_writer_keys()[10]
    writer = KernelWriterAssembly(depth64_key, _toolchain())
    row_stride = 2 * depth64_key.solution.depth_u
    assert writer._decoded_lds_store_location(0, 1, 32)[1] == 64
    assert writer._decoded_lds_store_location(2, 1, 32)[1] == 2 * row_stride + 64


def test_writer_emits_repaired_q5_sia3_and_depth64_pipeline_state() -> None:
    sia3_source = _source(_targeted_writer_keys()[5])
    assert "s_sub_u32 s5, s5, 32" in sia3_source
    assert "s_add_u32 s5, s5, 32" in sia3_source

    depth64_source = _source(_targeted_writer_keys()[6])
    assert (
        "Restore current-tile A pointers after next-B address setup." in depth64_source
    )
    assert "Reload next-tile A1 after current A1 becomes dead." in depth64_source
    assert depth64_source.count("v_add_nc_u32 v208, 64, v208") >= 2


def test_writer_emits_both_lower_bound_diagnostics() -> None:
    key = _key("Q4_K", (128, 2048, 512), Solution.pilot())
    wmma = _source(key, DiagnosticMode.WMMA_FLOOR)
    decode = _source(key, DiagnosticMode.DECODE_FLOOR)
    assert ".LWmmaFloorLoop:" in wmma
    assert ".LDecodeFloorLoop:" in decode


def test_writer_emits_non_direct_multirow_decode_floors() -> None:
    keys = _targeted_writer_keys()
    for key in (keys[16], keys[17]):
        source = _source(key, DiagnosticMode.DECODE_FLOOR)
        assert ".LDecodeFloorLoop:" in source
        assert source.count("global_load_d16_u8") >= 12


def test_assembly_rejects_nested_deferred_zero_fills() -> None:
    assembly = _Assembly()
    assembly.defer_zero_moves(range(2))
    with pytest.raises(KernelWriterError, match="cannot nest"):
        assembly.defer_zero_moves(range(2, 4))


def test_depth64_next_packed_prefetch_remains_q6_specific() -> None:
    q4 = _key(
        "Q4_K",
        (128, 2048, 512),
        replace(
            Solution.pilot(),
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
        replace(Solution.pilot(), depth_u=48),
    )
    with pytest.raises(KernelWriterError, match="solution rejected"):
        KernelWriterAssembly(invalid, _toolchain())


def test_writer_reports_missing_rocisa(monkeypatch: pytest.MonkeyPatch) -> None:
    key = _key("Q4_K", (128, 2048, 512), Solution.pilot())
    writer = KernelWriterAssembly(key, _toolchain())
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
    with pytest.raises(KernelWriterError, match="rocisa is required"):
        writer.source()
