from dataclasses import replace

import pytest

from tools.ggtensile.grouped_mmq_bwd_pair_inspection import (
    inspect_grouped_backward_pair_artifact,
)
from tools.ggtensile.grouped_mmq_bwd_pair_model import (
    GroupedBackwardPairProblem,
    GroupedBackwardPairSolution,
    GroupedBackwardPairSolutionKey,
)
from tools.ggtensile.grouped_mmq_bwd_pair_physical import (
    derive_grouped_backward_pair_physical_plan,
)
from tools.ggtensile.grouped_mmq_bwd_pair_runtime import (
    InstalledGroupedBackwardPairIQ2SControl,
    InstalledGroupedBackwardPairIQ2XXSControl,
)
from tools.ggtensile.grouped_mmq_bwd_pair_spec import (
    DerivedGroupedBackwardPairState,
)
from tools.ggtensile.grouped_mmq_bwd_pair_validation import (
    validate_grouped_backward_pair_solution,
)
from tools.ggtensile.iq2_s_grid import iq2_s_grid_values
from tools.ggtensile.kernel_abi import GROUPED_BACKWARD_PAIR_ABI
from tools.ggtensile.kernel_writer_assembly_grouped_mmq_bwd_pair import (
    GroupedBackwardPairKernelWriterAssembly,
)
from tools.ggtensile.schema import SchemaError
from tools.ggtensile.toolchain import Toolchain


def _key(rows: int = 16_384, tile: int = 64) -> GroupedBackwardPairSolutionKey:
    solution = (
        GroupedBackwardPairSolution.iq2_s_m64_n64()
        if tile == 64
        else GroupedBackwardPairSolution.iq2_s_m128_n64()
    )
    return GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_s(rows), solution
    )


def _iq2_xxs_key(
    rows: int = 12_288, tile: int = 64
) -> GroupedBackwardPairSolutionKey:
    solution = (
        GroupedBackwardPairSolution.iq2_xxs_m64_n64()
        if tile == 64
        else GroupedBackwardPairSolution.iq2_xxs_m128_n64()
    )
    return GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_xxs(rows), solution
    )


@pytest.mark.parametrize("rows", (35, 16_384, 65_536, 262_144))
@pytest.mark.parametrize("tile", (64, 128))
def test_backward_pair_identity_roundtrip(rows: int, tile: int) -> None:
    key = _key(rows, tile)
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)
    assert "grouped_mmq_bwd_pair_iq2_s" in key.kernel_name


@pytest.mark.parametrize("rows", (35, 12_288, 49_152, 196_608))
@pytest.mark.parametrize("tile", (64, 128))
def test_iq2_xxs_backward_pair_identity_roundtrip(rows: int, tile: int) -> None:
    key = _iq2_xxs_key(rows, tile)
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)
    assert "grouped_mmq_bwd_pair_iq2_xxs" in key.kernel_name


def test_iq2_s_packed_negative_payload_identity() -> None:
    mask = 0xFFFFFFFF
    for grid_value in iq2_s_grid_values():
        for payload in (grid_value & mask, grid_value >> 32):
            software_negation = ((~payload & mask) + 0x01010101) & mask
            packed_subtract = (0x01010100 - payload) & mask
            assert software_negation == packed_subtract


def test_backward_pair_identity_rejects_noncanonical_contract() -> None:
    mapping = _key().to_mapping()
    contract = mapping["ProblemContract"]
    assert isinstance(contract, dict)
    contract["projection_count"] = 1
    with pytest.raises(SchemaError, match="canonical"):
        GroupedBackwardPairSolutionKey.from_mapping(mapping)


def test_backward_pair_rejects_nonfused_projection_count() -> None:
    key = _key()
    invalid = replace(key, solution=replace(key.solution, projection_count=1))
    reasons = validate_grouped_backward_pair_solution(invalid)
    assert {reason.rule_id for reason in reasons} == {
        "grouped_backward_pair.solution.unimplemented"
    }


def test_backward_pair_abi_matches_specialized_launcher() -> None:
    assert GROUPED_BACKWARD_PAIR_ABI.segment_size == 72
    assert GROUPED_BACKWARD_PAIR_ABI.metadata_arguments == (
        ("first_grad_output", 0, 8, "global_buffer", "bf16"),
        ("second_grad_output", 8, 8, "global_buffer", "bf16"),
        ("first_packed_weight", 16, 8, "global_buffer", "struct"),
        ("second_packed_weight", 24, 8, "global_buffer", "struct"),
        ("grad_input", 32, 8, "global_buffer", "bf16"),
        ("expert_indices", 40, 8, "global_buffer", "i64"),
        ("expert_offsets", 48, 8, "global_buffer", "i32"),
        ("num_experts", 56, 4, "by_value", "i32"),
        ("rows", 60, 4, "by_value", "i32"),
        ("bytes_per_expert", 64, 8, "by_value", "i64"),
    )


@pytest.mark.parametrize(("tile", "vgprs", "wmmas"), ((64, 81, 16), (128, 121, 32)))
def test_backward_pair_physical_and_source(tile: int, vgprs: int, wmmas: int) -> None:
    key = _key(tile=tile)
    state = DerivedGroupedBackwardPairState.from_solution_key(key)
    physical = derive_grouped_backward_pair_physical_plan(state)
    assert physical.ordinary.resources.total_vgprs == vgprs
    assert physical.ordinary.resources.total_sgprs == 41
    assert physical.ordinary.resources.lds_num_bytes == 12_288
    assert state.expected_packed_weight_shape == (256, 512, 656)
    assert state.bytes_per_expert == 335_872

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Load the routed 72-byte backward-pair ABI" in source
    assert "Decode and accumulate the first pair projection" in source
    assert "Decode and accumulate the second pair projection" in source
    assert source.count("Swap the active gradient and packed-bank pointer pairs") == 2
    assert source.count("v_wmma_f32_16x16x16_bf16") == wmmas
    assert source.count("s_barrier") == 5
    assert source.count("global_store_d16_hi_b16") == wmmas * 2


@pytest.mark.parametrize(("tile", "vgprs", "wmmas"), ((64, 87, 16), (128, 127, 32)))
def test_iq2_xxs_backward_pair_physical_and_source(
    tile: int, vgprs: int, wmmas: int
) -> None:
    key = _iq2_xxs_key(rows=35, tile=tile)
    state = DerivedGroupedBackwardPairState.from_solution_key(key)
    physical = derive_grouped_backward_pair_physical_plan(state)
    assert physical.ordinary.resources.total_vgprs == vgprs
    assert physical.ordinary.resources.total_sgprs == 41
    assert physical.ordinary.resources.lds_num_bytes == 6_144
    assert state.expected_packed_weight_shape == (256, 2048, 1056)
    assert state.bytes_per_expert == 2_162_688

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Stage the IQ2_XXS codebook once in disjoint LDS" in source
    assert "Map each lane to one IQ2_XXS aligned 16-value group" in source
    assert "Load and sign the two IQ2_XXS grids owned by each lane" in source
    registers = physical.ordinary.registers
    assert (
        f"v_and_b32 v{registers.temporary}, 1, v{registers.serial}" in source
    )
    assert source.count("v_wmma_f32_16x16x16_bf16") == wmmas
    assert source.count("s_barrier") == 5


def test_backward_pair_k_pipeline_identity_and_synchronized_tail() -> None:
    key = GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_s(35),
        GroupedBackwardPairSolution.iq2_s_m128_n64_dual_lds_full_tile_split_k_pipeline(),
    )
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)

    state = DerivedGroupedBackwardPairState.from_solution_key(key)
    physical = derive_grouped_backward_pair_physical_plan(state)
    assert physical.ordinary.resources.total_vgprs == 149
    assert physical.ordinary.resources.total_sgprs == 41
    assert physical.ordinary.resources.lds_num_bytes == 16_384
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert source.count("v_wmma_f32_16x16x16_bf16") == 64
    assert source.count("s_barrier") == 7
    assert source.count("s_add_u32 s5, s5, 32") == 2
    assert "s_waitcnt vmcnt(14) lgkmcnt(0)" in source
    assert "s_waitcnt vmcnt(5)" in source

    tail = source[source.index(".LPairKPipelineComputeTail:") :]
    advance = tail.index("s_add_u32 s5, s5, 32")
    inactive_second = tail.index(
        "s_cbranch_scc1 .LGroupedBackwardInactiveWave0TailSecondLds"
    )
    assert advance < inactive_second


def test_backward_pair_sia5_global_codebook_packed_split_routes_8() -> None:
    key = GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_s(257),
        GroupedBackwardPairSolution.iq2_s_m128_n64_sia5_global_codebook_packed_split_routes_8(),
    )
    assert not validate_grouped_backward_pair_solution(key)

    state = DerivedGroupedBackwardPairState.from_solution_key(key)
    physical = derive_grouped_backward_pair_physical_plan(state)
    assert physical.ordinary.resources.lds_num_bytes == 8_192
    assert not physical.ordinary.lds.codebook_in_lds

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Stage the IQ2_S codebook once in disjoint LDS" not in source
    assert "global_load_b64" in source
    assert source.count("s_barrier") == 6
    assert "s_and_b32 s17, s3, 7" in source
    assert "s_lshr_b32 s3, s3, 3" in source
    assert "s_mov_b32 s2, s17" in source


def test_backward_pair_global_codebook_source() -> None:
    key = GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_s(16_384),
        GroupedBackwardPairSolution.iq2_s_m128_n64_dual_lds_global_codebook(),
    )
    state = DerivedGroupedBackwardPairState.from_solution_key(key)
    physical = derive_grouped_backward_pair_physical_plan(state)
    assert physical.ordinary.resources.lds_num_bytes == 8_192
    assert not physical.ordinary.lds.codebook_in_lds

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert source.count("global_load_b64") == 4
    assert source.count("s_barrier") == 2


def test_installed_backward_pair_dispatch_threshold() -> None:
    assert InstalledGroupedBackwardPairIQ2SControl.dispatched_macro_tile(35, 4) == 64
    assert (
        InstalledGroupedBackwardPairIQ2SControl.dispatched_macro_tile(16_384, 249) == 64
    )
    assert (
        InstalledGroupedBackwardPairIQ2SControl.dispatched_macro_tile(16_384, 125)
        == 128
    )


def test_installed_iq2_xxs_backward_pair_dispatch_rows() -> None:
    control = InstalledGroupedBackwardPairIQ2XXSControl
    assert control.dispatched_macro_tile(35) == 64
    assert control.dispatched_macro_tile(12_288) == 64
    assert control.dispatched_macro_tile(49_152) == 128
    assert control.dispatched_macro_tile(196_608) == 128
    assert control.BYTES_PER_EXPERT == 2_162_688


def test_backward_pair_build_and_inspection(tmp_path) -> None:
    key = _key(rows=35)
    toolchain = Toolchain.discover()
    writer = GroupedBackwardPairKernelWriterAssembly(key, toolchain)
    assert (
        writer.source()
        == GroupedBackwardPairKernelWriterAssembly(key, toolchain).source()
    )
    assembly = tmp_path / "pair.s"
    object_path = tmp_path / "pair.o"
    code_object = tmp_path / "pair.hsaco"
    writer.write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    result = inspect_grouped_backward_pair_artifact(key, code_object, toolchain)
    assert result.kernarg_segment_size == 72
    assert result.vgpr_count == 81
    assert result.sgpr_count == 41
    assert result.lds_num_bytes == 12_288
    assert result.private_segment_bytes == 0
    assert result.vgpr_spill_count == 0
    assert result.sgpr_spill_count == 0
    assert result.wmma_count == 16
    assert result.barrier_count == 5


def test_iq2_xxs_backward_pair_build_and_inspection(tmp_path) -> None:
    key = _iq2_xxs_key(rows=35)
    toolchain = Toolchain.discover()
    writer = GroupedBackwardPairKernelWriterAssembly(key, toolchain)
    assert writer.source() == GroupedBackwardPairKernelWriterAssembly(
        key, toolchain
    ).source()
    assembly = tmp_path / "iq2_xxs_pair.s"
    object_path = tmp_path / "iq2_xxs_pair.o"
    code_object = tmp_path / "iq2_xxs_pair.hsaco"
    writer.write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    result = inspect_grouped_backward_pair_artifact(key, code_object, toolchain)
    assert result.kernarg_segment_size == 72
    assert result.vgpr_count == 87
    assert result.sgpr_count == 41
    assert result.lds_num_bytes == 6_144
    assert result.private_segment_bytes == 0
    assert result.vgpr_spill_count == 0
    assert result.sgpr_spill_count == 0
    assert result.wmma_count == 16
    assert result.barrier_count == 5
