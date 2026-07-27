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
    assert key.hash.startswith("ggsol_")
    assert SolutionKey.from_mapping(key.to_mapping()) == key


def test_writer_enables_and_flattens_packed_workitem_xy() -> None:
    writer = KernelWriterAssembly(_pilot_key(), _toolchain())
    source = writer.source()
    registers = writer.registers
    assert ".amdhsa_system_vgpr_workitem_id 1" in source
    assert f"v_bfe_u32 v{registers.serial}, v0, 10, 10" in source
    assert (
        f"v_add_nc_u32 v{registers.serial}, v{registers.serial}, "
        f"v{registers.temporary}" in source
    )


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


def test_writer_emits_distinct_xor16_lds_layout() -> None:
    pilot = Solution.pilot()
    swizzled = replace(pilot, lds_swizzle_chunk_b=16)
    key = SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(128, 2048, 512),
        swizzled,
    )
    assert validate_solution(key) == ()

    source = KernelWriterAssembly(key, _toolchain()).source()
    assert "Precompute XOR-16 LDS store bases" in source
    assert source.count("ds_load_b128") == 32
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32


def test_writer_emits_distinct_xor4_lds_layout() -> None:
    pilot = Solution.pilot()
    swizzled = replace(pilot, lds_swizzle_chunk_b=4, schedule_iter_alg=3)
    key = SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(128, 2048, 512),
        swizzled,
    )
    assert validate_solution(key) == ()

    source = KernelWriterAssembly(key, _toolchain()).source()
    assert "Precompute XOR-4 LDS store bases" in source
    assert source.count("ds_load_b64") == 64
    assert source.count("ds_load_b128") == 0
    assert source.count("s_waitcnt vmcnt(2) lgkmcnt(4)") == 2
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


def test_writer_overlaps_first_a_half_with_decode(tmp_path: Path) -> None:
    pilot = Solution.pilot()
    scheduled = replace(
        pilot,
        schedule_iter_alg=4,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        scheduled,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
    assembly = tmp_path / "schedule4.s"
    object_path = tmp_path / "schedule4.o"
    code_object = tmp_path / "schedule4.hsaco"
    source = KernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    prefetch = source.index("Prefetch A fragments")
    q4_decode = source.index("Unpack Q4_K six-bit scale/min fields")
    assert prefetch < q4_decode
    assert "Reuse prefetched A pointers across fused B decode." in source
    assert "Prepare direct packed-nibble bit offsets." in source
    assert source[prefetch:q4_decode].count("s_waitcnt vmcnt(4)") == 1
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 200
    assert inspection.sgpr_count == 20
    assert inspection.lds_num_bytes == 8192


def test_writer_prefetches_both_a_halves(tmp_path: Path) -> None:
    pilot = Solution.pilot()
    prefetched = replace(
        pilot,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        prefetched,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
    assembly = tmp_path / "prefetch_global_read.s"
    object_path = tmp_path / "prefetch_global_read.o"
    code_object = tmp_path / "prefetch_global_read.hsaco"
    source = KernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert source.count("s_waitcnt vmcnt(8)") == 1
    assert source.count("s_waitcnt vmcnt(4) lgkmcnt(0)") == 1
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 216
    assert inspection.sgpr_count == 20
    assert inspection.lds_num_bytes == 8192


def test_writer_shares_packed_weight_across_nibble_lanes(
    tmp_path: Path,
) -> None:
    pilot = Solution.pilot()
    shared = replace(
        pilot,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        packed_weight_lane_share=2,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        shared,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
    assembly = tmp_path / "packed_weight_lane_share.s"
    object_path = tmp_path / "packed_weight_lane_share.o"
    code_object = tmp_path / "packed_weight_lane_share.hsaco"
    source = KernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert "s_and_saveexec_b32" in source
    assert source.count("v_mov_b32_dpp") == 8
    assert "ds_bpermute_b32" not in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 216
    assert inspection.sgpr_count == 20
    assert inspection.lds_num_bytes == 8192


def test_writer_pipelines_packed_weight_reads(tmp_path: Path) -> None:
    pilot = Solution.pilot()
    pipelined = replace(
        pilot,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        prefetch_packed_weight_next=True,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        pipelined,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
    assembly = tmp_path / "prefetch_packed_weight.s"
    object_path = tmp_path / "prefetch_packed_weight.o"
    code_object = tmp_path / "prefetch_packed_weight.hsaco"
    source = KernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert ".LPackedDepthULoop:" in source
    assert "current A before next packed reads" in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 216
    assert inspection.sgpr_count == 20
    assert inspection.lds_num_bytes == 8192
    assert inspection.barrier_count == 3


def test_writer_combines_global_prefetch_with_partial_waits(
    tmp_path: Path,
) -> None:
    pilot = Solution.pilot()
    scheduled = replace(
        pilot,
        schedule_iter_alg=5,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        scheduled,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
    assembly = tmp_path / "schedule5.s"
    object_path = tmp_path / "schedule5.o"
    code_object = tmp_path / "schedule5.hsaco"
    source = KernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert source.count("s_waitcnt vmcnt(6) lgkmcnt(2)") == 1
    assert source.count("s_waitcnt vmcnt(2) lgkmcnt(2)") == 1
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 216
    assert inspection.sgpr_count == 20
    assert inspection.lds_num_bytes == 8192


def test_writer_builds_depth_u64_geometry(tmp_path: Path) -> None:
    pilot = Solution.pilot()
    depth_u64 = replace(
        pilot,
        depth_u=64,
        schedule_iter_alg=4,
        prefetch_global_read=2,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        depth_u64,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
    assembly = tmp_path / "depth_u64.s"
    object_path = tmp_path / "depth_u64.o"
    code_object = tmp_path / "depth_u64.hsaco"
    source = KernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert source.count("v_wmma_f32_16x16x16_bf16") == 64
    assert source.count("ds_store_b16_d16_hi") == 64
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 236
    assert inspection.sgpr_count == 20
    assert inspection.lds_num_bytes == 16384
    assert inspection.barrier_count == 2


def test_writer_prefetches_next_local_read_pair(tmp_path: Path) -> None:
    pilot = Solution.pilot()
    prefetched = replace(
        pilot,
        schedule_iter_alg=3,
        prefetch_local_read=2,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        prefetched,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
    assembly = tmp_path / "prefetch_local_read.s"
    object_path = tmp_path / "prefetch_local_read.o"
    code_object = tmp_path / "prefetch_local_read.hsaco"
    source = KernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert source.count("s_waitcnt vmcnt(2) lgkmcnt(6)") == 2
    assert source.count("s_waitcnt lgkmcnt(6)") == 4
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 216
    assert inspection.sgpr_count == 20
    assert inspection.lds_num_bytes == 8192


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


def test_writer_builds_256x64_geometry(tmp_path: Path) -> None:
    pilot = Solution.pilot()
    geometry = replace(
        pilot,
        matrix_instruction=(16, 16, 16, 1, 1, 4, 4, 4, 1),
        macro_tile0=256,
        macro_tile1=64,
        lds_swizzle_chunk_b=8,
    )
    key = SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        geometry,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
    assembly = tmp_path / "geometry.s"
    object_path = tmp_path / "geometry.o"
    code_object = tmp_path / "geometry.hsaco"
    source = KernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    assert source.count("ds_store_b16_d16_hi") == 16
    assert source.count("ds_load_b128") == 16
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 208
    assert inspection.sgpr_count == 20
    assert inspection.lds_num_bytes == 4096


def test_writer_builds_256x128_cooperative_decode_geometry(
    tmp_path: Path,
) -> None:
    pilot = Solution.pilot()
    geometry = replace(
        pilot,
        work_group=(32, 8, 1),
        matrix_instruction=(16, 16, 16, 1, 1, 2, 8, 8, 1),
        macro_tile0=256,
    )
    key = SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(32768, 2048, 8192),
        geometry,
    )
    assert validate_solution(key) == ()

    toolchain = _toolchain()
    assembly = tmp_path / "cooperative_decode.s"
    object_path = tmp_path / "cooperative_decode.o"
    code_object = tmp_path / "cooperative_decode.hsaco"
    source = KernelWriterAssembly(key, toolchain).source()
    assembly.write_text(source)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    assert "Only the first four waves cooperatively decode B." in source
    assert "s_cbranch_scc0 .LDecodeReady" in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.vgpr_count == 196
    assert inspection.sgpr_count == 20
    assert inspection.max_flat_workgroup_size == 256
    assert inspection.lds_num_bytes == 8192


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
