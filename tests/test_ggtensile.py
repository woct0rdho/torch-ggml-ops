from dataclasses import replace
from pathlib import Path

import pytest

from tools.ggtensile.inspection import inspect_artifact
from tools.ggtensile.kernel_writer_assembly import KernelWriterAssembly
from tools.ggtensile.model import ProblemSize, ProblemType, Solution, SolutionKey
from tools.ggtensile.toolchain import Toolchain, ToolchainError
from tools.ggtensile.validation import validate_solution


def _pilot_key() -> SolutionKey:
    return SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(128, 2048, 512),
        Solution.pilot(),
    )


def _toolchain() -> Toolchain:
    try:
        return Toolchain.discover()
    except ToolchainError as error:
        pytest.skip(str(error))


def test_pilot_solution_identity_and_round_trip() -> None:
    key = _pilot_key()
    assert validate_solution(key) == ()
    assert key.hash == "ggsol_493082f972ef96fc"
    assert SolutionKey.from_mapping(key.to_mapping()) == key


def test_writer_enables_and_flattens_packed_workitem_xy() -> None:
    source = KernelWriterAssembly(_pilot_key(), _toolchain()).source()
    assert ".amdhsa_system_vgpr_workitem_id 1" in source
    assert "v_bfe_u32 v195, v0, 10, 10" in source
    assert "v_add_nc_u32 v195, v195, v188" in source


def test_build_and_inspect_pilot(tmp_path: Path) -> None:
    key = _pilot_key()
    toolchain = _toolchain()
    assembly = tmp_path / "pilot.s"
    object_path = tmp_path / "pilot.o"
    code_object = tmp_path / "pilot.hsaco"
    KernelWriterAssembly(key, toolchain).write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.target == "gfx1151"
    assert inspection.code_object_version == 5
    assert inspection.vgpr_count == 196
    assert inspection.sgpr_count == 20
    assert inspection.max_vgpr_index == 195
    assert inspection.max_sgpr_index == 19
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0
    assert inspection.wmma_count == 32
    assert inspection.barrier_count == 2


def test_writer_emits_distinct_xor8_lds_layout() -> None:
    pilot = Solution.pilot()
    swizzled = replace(pilot, lds_swizzle_chunk_b=8)
    key = SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(128, 2048, 512),
        swizzled,
    )
    assert validate_solution(key) == ()

    source = KernelWriterAssembly(key, _toolchain()).source()
    assert "Precompute XOR-8 LDS store bases" in source
    assert "v_xor_b32" in source
    assert source.count("ds_load_b128") == 32
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32


def test_writer_emits_sia3_partial_wait_schedule() -> None:
    pilot = Solution.pilot()
    scheduled = replace(pilot, schedule_iter_alg=3)
    key = SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(128, 2048, 512),
        scheduled,
    )
    assert validate_solution(key) == ()

    source = KernelWriterAssembly(key, _toolchain()).source()
    assert source.count("s_waitcnt vmcnt(2) lgkmcnt(2)") == 2
    assert source.count("s_waitcnt lgkmcnt(2)") == 6
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32


def test_writer_maps_grouped_m_launch_coordinates() -> None:
    pilot = Solution.pilot()
    all_m = replace(pilot, work_group_mapping=256)
    key = SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        all_m,
    )
    assert validate_solution(key) == ()

    source = KernelWriterAssembly(key, _toolchain()).source()
    assert ".amdhsa_system_sgpr_workgroup_id_z 1" in source
    assert "s_mul_i32 s4, s4, 256" in source
    assert "s_add_u32 s2, s4, s2" in source


def test_writer_can_disable_store_priority() -> None:
    pilot = Solution.pilot()
    no_store_priority = replace(pilot, store_priority_opt=False)
    key = SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(128, 2048, 512),
        no_store_priority,
    )
    assert validate_solution(key) == ()

    source = KernelWriterAssembly(key, _toolchain()).source()
    assert "s_setprio" not in source
