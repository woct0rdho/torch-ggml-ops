from dataclasses import replace

import pytest

from tools.ggtensile.grouped_mmq_bwd_physical import (
    derive_grouped_backward_physical_plan,
)
from tools.ggtensile.grouped_mmq_bwd_spec import DerivedGroupedBackwardState
from tools.ggtensile.inspection import inspect_artifact
from tools.ggtensile.kernel_abi import GROUPED_BACKWARD_ABI
from tools.ggtensile.kernel_writer_assembly_grouped_mmq_bwd import (
    GroupedBackwardKernelWriterAssembly,
)
from tools.ggtensile.model import (
    GroupedBackwardSolution,
    ProblemSize,
    ProblemType,
    SchemaError,
    SolutionKey,
)
from tools.ggtensile.toolchain import Toolchain
from tools.ggtensile.validation import validate_solution


def _key(rows: int | None = None, quant_type: str = "Q4_K") -> SolutionKey:
    if quant_type == "Q2_K":
        size = ProblemSize(12_288 if rows is None else rows, 2048, 4096)
    else:
        size = ProblemSize(16_384 if rows is None else rows, 512, 2048)
    return SolutionKey(
        ProblemType.grouped_mmq_backward(quant_type),
        size,
        GroupedBackwardSolution.pilot(),
    )


def _mixed_key(rows: int = 16_384) -> SolutionKey:
    base = _key(rows)
    solution = base.solution
    assert isinstance(solution, GroupedBackwardSolution)
    compute = replace(
        solution.compute,
        prefetch_global_read=2,
        schedule_iter_alg=5,
        lds_pad_b=8,
        q4_k_decode_schedule="DependencyBatch4",
    )
    return replace(
        base,
        solution=replace(solution, compute=compute, row_tail="Mixed128_64"),
    )


def test_grouped_backward_identity_roundtrip_and_exact_rows() -> None:
    first = _key()
    assert SolutionKey.from_mapping(first.to_mapping()) == first
    assert not validate_solution(first)
    assert first.kernel_name.startswith(
        "torch_ggml_ops_ggtensile_gfx1151_v1_grouped_mmq_bwd_q4_k_"
    )
    for rows in (16_384, 65_536, 262_144):
        assert not validate_solution(_key(rows))
    for rows in (16_383, 32_768, 262_145):
        with pytest.raises(SchemaError, match="exact"):
            _key(rows).to_mapping()


def test_grouped_backward_q2_identity_and_source() -> None:
    for rows in (12_288, 49_152, 196_608):
        assert not validate_solution(_key(rows, "Q2_K"))
    for rows in (12_287, 16_384, 196_609):
        with pytest.raises(SchemaError, match="exact"):
            _key(rows, "Q2_K").to_mapping()

    q2 = _key(quant_type="Q2_K")
    assert SolutionKey.from_mapping(q2.to_mapping()) == q2
    assert not validate_solution(q2)
    assert "grouped_mmq_bwd_q2_k" in q2.kernel_name
    contract = q2.to_mapping()["ProblemContract"]
    assert isinstance(contract, dict)
    assert contract["packed_weight_block_bytes"] == 84

    source = GroupedBackwardKernelWriterAssembly(q2, Toolchain.discover()).source()
    assert "GGTensile Q2_K grouped MMQ backward" in source
    assert "Decode Q2_K two-bit payload" in source
    assert "s_cmp_lg_u32 s22, 2752512" in source
    assert "s_lshl_b32 s30, s27, 13" in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    assert source.count("s_barrier") == 2


def test_grouped_backward_q2_dependency_batch_identity_and_source() -> None:
    key = _key(quant_type="Q2_K")
    solution = key.solution
    assert isinstance(solution, GroupedBackwardSolution)
    candidate = replace(
        key,
        solution=replace(
            solution,
            compute=replace(
                solution.compute,
                q2_k_decode_schedule="DependencyBatch4",
            ),
        ),
    )
    assert SolutionKey.from_mapping(candidate.to_mapping()) == candidate
    assert not validate_solution(candidate)
    spec = candidate.to_mapping()["KernelSpec"]
    assert isinstance(spec, dict)
    assert spec["decode"] == {"schedule": "DependencyBatch4"}

    physical = derive_grouped_backward_physical_plan(
        DerivedGroupedBackwardState.from_solution_key(candidate)
    ).compute
    assert physical.q2_decode.dependency_width == 4
    values = physical.q2_decode.value_registers
    assert values == tuple(physical.registers.valu_b + 2 * slot for slot in range(4))

    source = GroupedBackwardKernelWriterAssembly(
        candidate, Toolchain.discover()
    ).source()
    conversions = [
        source.index(f"v_cvt_f32_ubyte{slot}_e32 v{value}")
        for slot, value in enumerate(values)
    ]
    first_fma = source.index(f"v_fma_f32 v{values[0]}", conversions[-1])
    assert max(conversions) < first_fma


def test_grouped_backward_q5_identity_and_source() -> None:
    q4 = _key()
    q5 = _key(quant_type="Q5_K")
    assert SolutionKey.from_mapping(q5.to_mapping()) == q5
    assert not validate_solution(q5)
    assert q4.hash != q5.hash
    assert "grouped_mmq_bwd_q5_k" in q5.kernel_name
    contract = q5.to_mapping()["ProblemContract"]
    assert isinstance(contract, dict)
    assert contract["packed_weight_block_bytes"] == 176

    source = GroupedBackwardKernelWriterAssembly(q5, Toolchain.discover()).source()
    assert "GGTensile Q5_K grouped MMQ backward" in source
    assert "Build Q5_K payload and scale-byte addresses." in source
    assert "Decode Q5_K low nibbles and high payload bits into LDS." in source
    assert "s_cmp_lg_u32 s22, 720896" in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    assert source.count("s_barrier") == 2


def test_grouped_backward_iq2_s_identity_source_and_inspection(tmp_path) -> None:
    for rows in (16_384, 65_536, 262_144):
        assert not validate_solution(_key(rows, "IQ2_S"))
    for rows in (16_383, 32_768, 262_145):
        with pytest.raises(SchemaError, match="exact"):
            _key(rows, "IQ2_S").to_mapping()

    key = _key(quant_type="IQ2_S")
    assert SolutionKey.from_mapping(key.to_mapping()) == key
    assert "grouped_mmq_bwd_iq2_s" in key.kernel_name
    contract = key.to_mapping()["ProblemContract"]
    assert isinstance(contract, dict)
    assert contract["packed_weight_block_bytes"] == 82

    toolchain = Toolchain.discover()
    writer = GroupedBackwardKernelWriterAssembly(key, toolchain)
    source = writer.source()
    assert "GGTensile IQ2_S grouped MMQ backward" in source
    assert "Map each lane to one IQ2_S aligned 16-value group." in source
    assert "Decode IQ2_S codebook values and signed scales into LDS." in source
    assert "s_getpc_b64 s[36:37]" in source
    assert "s_mov_b32 s13, 0x03020100" in source
    assert "s_cmp_lg_u32 s22, 335872" in source
    assert source.count("global_load_b64") == 0
    assert source.count("ds_load_b64") == 4
    assert "Stage the IQ2_S codebook once in disjoint LDS." in source
    assert source.count("v_perm_b32") == 8
    assert source.count(".quad") == 256
    assert ".size .LGGTensileIQ2SGrid, 8192" in source

    assembly = tmp_path / "iq2_s.s"
    object_path = tmp_path / "iq2_s.o"
    code_object = tmp_path / "iq2_s.hsaco"
    writer.write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    result = inspect_artifact(key, code_object, toolchain)
    assert result.vgpr_count == 194
    assert result.sgpr_count == 38
    assert result.lds_num_bytes == 16384
    assert result.private_segment_bytes == 0
    assert result.vgpr_spill_count == 0
    assert result.sgpr_spill_count == 0
    assert result.wmma_count == 32
    assert result.barrier_count == 3


@pytest.mark.parametrize(
    ("field", "value", "rule_id"),
    (
        (
            "q5_k_metadata_vector_load",
            True,
            "solution.q5kmetadata.n64",
        ),
        (
            "packed_weight_lane_share",
            2,
            "solution.packedweightlaneshare.q5n64",
        ),
    ),
)
def test_grouped_backward_q5_n64_rejects_unimplemented_sharing(
    field: str, value: object, rule_id: str
) -> None:
    key = _key(quant_type="Q5_K")
    solution = key.solution
    assert isinstance(solution, GroupedBackwardSolution)
    n64 = replace(
        solution.compute,
        matrix_instruction=(16, 16, 16, 1, 1, 2, 4, 4, 1),
        macro_tile1=64,
        **{field: value},
    )
    invalid = replace(key, solution=replace(solution, compute=n64))
    assert rule_id in {reason.rule_id for reason in validate_solution(invalid)}

    n128 = replace(solution.compute, **{field: value})
    valid = replace(key, solution=replace(solution, compute=n128))
    assert rule_id not in {reason.rule_id for reason in validate_solution(valid)}


def test_grouped_backward_rejects_unsupported_quant_type() -> None:
    with pytest.raises(ValueError, match="unsupported grouped"):
        ProblemType.grouped_mmq_backward("Q3_K")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("abi", "BackwardOutputA"),
        ("physical_experts", 255),
        ("max_route_entries", 257),
        ("wavefront_size", 64),
        ("packed_weight_block_bytes", 176),
    ),
)
def test_grouped_backward_rejects_noncanonical_contract(
    field: str, value: object
) -> None:
    mapping = _key().to_mapping()
    contract = mapping["ProblemContract"]
    assert isinstance(contract, dict)
    contract[field] = value
    with pytest.raises(SchemaError, match="canonical"):
        SolutionKey.from_mapping(mapping)


def test_grouped_backward_mixed_tail_roundtrip_and_schema() -> None:
    key = _mixed_key()
    assert SolutionKey.from_mapping(key.to_mapping()) == key
    mapping = key.to_mapping()
    spec = mapping["KernelSpec"]
    assert isinstance(spec, dict)
    ownership = spec["ownership"]
    assert isinstance(ownership, dict)
    ownership["row_tail"] = "Mixed96_48"
    with pytest.raises(SchemaError, match="ownership"):
        SolutionKey.from_mapping(mapping)


def test_grouped_backward_mixed_tail_source_is_namespaced() -> None:
    key = _mixed_key()
    source = GroupedBackwardKernelWriterAssembly(key, Toolchain.discover()).source()
    labels = tuple(
        line.strip() for line in source.splitlines() if line.startswith(".L")
    )
    assert len(labels) == len(set(labels))
    assert source.count("v_wmma_f32_16x16x16_bf16") == 48
    assert source.count("s_barrier") == 4
    assert source.count(".LDepthULoop:") == 1
    assert source.count(".LDepthULoopTail:") == 1
    assert source.count(".LScaleOdd:") == 1
    assert source.count(".LScaleOddTail:") == 1
    assert "v_mov_b32 v127, v207" in source


def test_grouped_backward_mixed_tail_rejects_invalid_primary_geometry() -> None:
    pilot = GroupedBackwardSolution.pilot()
    invalid = replace(
        _key(),
        solution=replace(
            pilot,
            row_tail="Mixed128_64",
            compute=replace(pilot.compute, macro_tile0=64),
        ),
    )
    assert any(
        reason.rule_id == "solution.grouped_backward.row_tail"
        for reason in validate_solution(invalid)
    )


def test_grouped_backward_mixed_tail_rejects_unqualified_pipeline() -> None:
    key = _mixed_key()
    solution = key.solution
    assert isinstance(solution, GroupedBackwardSolution)
    invalid = replace(
        key,
        solution=replace(
            solution,
            compute=replace(
                solution.compute,
                schedule_iter_alg=4,
                prefetch_packed_weight_next=True,
            ),
        ),
    )
    assert any(
        reason.rule_id == "solution.grouped_backward.row_tail"
        for reason in validate_solution(invalid)
    )


def test_grouped_backward_rejects_unknown_ownership_and_mapping() -> None:
    mapping = _key().to_mapping()
    spec = mapping["KernelSpec"]
    assert isinstance(spec, dict)
    ownership = spec["ownership"]
    assert isinstance(ownership, dict)
    ownership["kind"] = "ImplicitTasks"
    with pytest.raises(SchemaError, match="ownership"):
        SolutionKey.from_mapping(mapping)

    solution = _key().solution
    assert isinstance(solution, GroupedBackwardSolution)
    invalid = replace(
        _key(),
        solution=replace(
            solution,
            compute=replace(solution.compute, work_group_mapping=2),
        ),
    )
    assert any(
        reason.rule_id == "solution.grouped_backward.work_group_mapping"
        for reason in validate_solution(invalid)
    )


@pytest.mark.parametrize(
    ("ownership", "split_factor"),
    (
        ("SplitRoutes2", 2),
        ("SplitRoutes4", 4),
        ("SplitRoutes8", 8),
        ("SplitRoutes16", 16),
        ("SplitRoutes32", 32),
    ),
)
def test_grouped_backward_split_route_identity_and_source(
    ownership: str, split_factor: int
) -> None:
    pilot = GroupedBackwardSolution.pilot()
    key = replace(_key(), solution=replace(pilot, route_ownership=ownership))
    assert SolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_solution(key)

    source = GroupedBackwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert ".amdhsa_system_sgpr_workgroup_id_z 1" in source
    assert "s_mov_b32 s2, s4" in source
    assert f"s_add_u32 s2, s2, {split_factor}" in source
    assert "s_cmp_ge_u32 s34, s27" in source


def test_grouped_iq2_s_split64_is_format_specific() -> None:
    iq2 = _key(quant_type="IQ2_S")
    solution = iq2.solution
    assert isinstance(solution, GroupedBackwardSolution)
    split64 = replace(iq2, solution=replace(solution, route_ownership="SplitRoutes64"))
    assert not validate_solution(split64)
    assert SolutionKey.from_mapping(split64.to_mapping()) == split64
    source = GroupedBackwardKernelWriterAssembly(split64, Toolchain.discover()).source()
    assert ".amdhsa_system_sgpr_workgroup_id_z 1" in source
    assert "s_add_u32 s2, s2, 64" in source

    q4 = _key()
    q4_solution = q4.solution
    assert isinstance(q4_solution, GroupedBackwardSolution)
    invalid = replace(
        q4,
        solution=replace(q4_solution, route_ownership="SplitRoutes64"),
    )
    assert any(
        reason.rule_id == "solution.grouped_backward.split64.quant"
        for reason in validate_solution(invalid)
    )


def test_grouped_backward_physical_plan_is_deterministic() -> None:
    state = DerivedGroupedBackwardState.from_solution_key(_key())
    first = derive_grouped_backward_physical_plan(state)
    second = derive_grouped_backward_physical_plan(state)
    assert first == second
    assert first.resources.total_vgprs == 192
    assert first.resources.total_sgprs == 35
    assert first.resources.lds_num_bytes == 8192
    assert first.resources.private_segment_bytes == 0
    assert first.route.saved_exec == 31
    assert first.route.pointer_temporary == 32


def test_grouped_backward_abi_matches_specialized_launcher() -> None:
    assert GROUPED_BACKWARD_ABI.segment_size == 56
    assert GROUPED_BACKWARD_ABI.metadata_arguments == (
        ("grad_output", 0, 8, "global_buffer", "bf16"),
        ("packed_weight", 8, 8, "global_buffer", "struct"),
        ("grad_input", 16, 8, "global_buffer", "bf16"),
        ("expert_indices", 24, 8, "global_buffer", "i64"),
        ("expert_offsets", 32, 8, "global_buffer", "i32"),
        ("num_experts", 40, 4, "by_value", "i32"),
        ("rows", 44, 4, "by_value", "i32"),
        ("bytes_per_expert", 48, 8, "by_value", "i64"),
    )


def test_grouped_backward_source_has_route_and_tail_guards() -> None:
    source = GroupedBackwardKernelWriterAssembly(_key(), Toolchain.discover()).source()
    assert "Load the routed 56-byte grouped backward ABI" in source
    assert "s_cmp_lg_u32 s20, 256" in source
    assert "s_cmp_lg_u32 s21, 16384" in source
    assert "s_cmp_lg_u32 s22, 589824" in source
    assert "s_load_dwordx2 s[28:29], s[16:17]" in source
    assert "s_mul_hi_u32 s33, s28, s22" in source
    assert "s_and_saveexec_b32 s31, vcc_lo" in source
    assert "s_mov_b32 exec_lo, s31" in source
    assert source.count(".LGroupedBackwardRowTile:") == 1
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32


def test_grouped_backward_build_is_deterministic_and_inspectable(tmp_path) -> None:
    key = _key()
    toolchain = Toolchain.discover()
    first = GroupedBackwardKernelWriterAssembly(key, toolchain)
    second = GroupedBackwardKernelWriterAssembly(key, toolchain)
    assert first.source() == second.source()

    assembly = tmp_path / "kernel.s"
    object_path = tmp_path / "kernel.o"
    code_object = tmp_path / "kernel.hsaco"
    first.write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    result = inspect_artifact(key, code_object, toolchain)
    assert result.kernarg_segment_size == 56
    assert result.max_flat_workgroup_size == 128
    assert result.vgpr_count == 192
    assert result.sgpr_count == 35
    assert result.lds_num_bytes == 8192
    assert result.private_segment_bytes == 0
    assert result.vgpr_spill_count == 0
    assert result.sgpr_spill_count == 0
    assert result.wmma_count == 32
    assert result.barrier_count == 2


def test_grouped_backward_mixed_tail_inspection_counts(tmp_path) -> None:
    key = _mixed_key()
    toolchain = Toolchain.discover()
    assembly = tmp_path / "mixed.s"
    object_path = tmp_path / "mixed.o"
    code_object = tmp_path / "mixed.hsaco"
    writer = GroupedBackwardKernelWriterAssembly(key, toolchain)
    writer.write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    result = inspect_artifact(key, code_object, toolchain)
    assert result.vgpr_count == 208
    assert result.sgpr_count == 35
    assert result.lds_num_bytes == 10240
    assert result.private_segment_bytes == 0
    assert result.vgpr_spill_count == 0
    assert result.sgpr_spill_count == 0
    assert result.wmma_count == 48
    assert result.barrier_count == 4
