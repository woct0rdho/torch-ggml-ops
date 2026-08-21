import hashlib
from dataclasses import replace

import pytest

from tools.ggtensile.grouped_mmq_bwd_pair_inspection import (
    inspect_grouped_backward_pair_artifact,
)
from tools.ggtensile.grouped_mmq_bwd_pair_model import (
    GroupedBackwardPairProblem,
    GroupedBackwardPairProjectionSchedule,
    GroupedBackwardPairSolution,
    GroupedBackwardPairSolutionKey,
)
from tools.ggtensile.grouped_mmq_bwd_pair_physical import (
    derive_grouped_backward_pair_physical_plan,
)
from tools.ggtensile.grouped_mmq_bwd_pair_runtime import (
    InstalledGroupedBackwardPairIQ2SControl,
    InstalledGroupedBackwardPairIQ2XXSControl,
    InstalledGroupedBackwardPairQ3KControl,
)
from tools.ggtensile.grouped_mmq_bwd_pair_selection import (
    ResearchGroupedBackwardPairQ3KSelector,
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


def _iq2_xxs_key(rows: int = 12_288, tile: int = 64) -> GroupedBackwardPairSolutionKey:
    solution = (
        GroupedBackwardPairSolution.iq2_xxs_m64_n64()
        if tile == 64
        else GroupedBackwardPairSolution.iq2_xxs_m128_n64()
    )
    return GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_xxs(rows), solution
    )


def _q3_k_key(rows: int = 16_384, tile: int = 64) -> GroupedBackwardPairSolutionKey:
    solution = (
        GroupedBackwardPairSolution.q3_k_m64_n64()
        if tile == 64
        else GroupedBackwardPairSolution.q3_k_m128_n64()
    )
    return GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.q3_k(rows), solution
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


@pytest.mark.parametrize("rows", (35, 16_384, 65_536, 262_144))
@pytest.mark.parametrize("tile", (64, 128))
def test_q3_k_backward_pair_identity_roundtrip(rows: int, tile: int) -> None:
    key = _q3_k_key(rows, tile)
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)
    assert "grouped_mmq_bwd_pair_q3_k" in key.kernel_name


@pytest.mark.parametrize(
    ("constructor", "expected_vgprs"),
    (
        ("iq2_xxs_m128_n64_dual_lds", 127),
        ("iq2_xxs_m128_n64_dual_lds_full_tile_split", 127),
        (
            "iq2_xxs_m128_n64_dual_lds_full_tile_split_concurrent_reads",
            137,
        ),
        (
            "iq2_xxs_m128_n64_dual_lds_full_tile_split_concurrent_reads_prefetch_a",
            153,
        ),
    ),
)
def test_iq2_xxs_staged_schedule_identity_and_physical_plan(
    constructor: str, expected_vgprs: int
) -> None:
    solution = getattr(GroupedBackwardPairSolution, constructor)()
    key = GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_xxs(35), solution
    )
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)

    physical = derive_grouped_backward_pair_physical_plan(
        DerivedGroupedBackwardPairState.from_solution_key(key)
    )
    assert physical.ordinary.resources.total_vgprs == expected_vgprs
    assert physical.ordinary.resources.lds_num_bytes == 10_240
    assert physical.ordinary.lds.codebook_offset == 8_192
    assert physical.ordinary.lds.codebook_in_lds
    assert physical.second_projection is not None
    assert physical.second_projection.lds.base_offset == 4_096


@pytest.mark.parametrize(
    ("schedule", "prefetch_a", "expected_vgprs"),
    (
        (
            GroupedBackwardPairProjectionSchedule.DualLdsInterleavedDepthU,
            False,
            87,
        ),
        (
            GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitInterleavedDepthU,
            False,
            87,
        ),
        (
            GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsInterleavedDepthU,
            False,
            97,
        ),
        (
            GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsPrefetchAInterleavedDepthU,
            True,
            105,
        ),
    ),
)
def test_iq2_xxs_m64_staged_schedule_identity_and_physical_plan(
    schedule: GroupedBackwardPairProjectionSchedule,
    prefetch_a: bool,
    expected_vgprs: int,
) -> None:
    solution = GroupedBackwardPairSolution.iq2_xxs_m64_n64()
    solution = replace(
        solution,
        compute=replace(
            solution.compute,
            prefetch_global_read=2 if prefetch_a else 1,
            schedule_iter_alg=4 if prefetch_a else 2,
        ),
        projection_schedule=schedule,
    )
    key = GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_xxs(35), solution
    )
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)

    physical = derive_grouped_backward_pair_physical_plan(
        DerivedGroupedBackwardPairState.from_solution_key(key)
    )
    assert physical.ordinary.resources.total_vgprs == expected_vgprs
    assert physical.ordinary.resources.lds_num_bytes == 10_240
    assert physical.second_projection is not None


def test_iq2_xxs_m64_prefetch_a_swizzle8_selected_physical_plan() -> None:
    solution = GroupedBackwardPairSolution.iq2_xxs_m64_n64_dual_lds_full_tile_split_concurrent_reads_prefetch_a()
    assert solution.compute.prefetch_global_read == 2
    assert solution.compute.schedule_iter_alg == 4
    assert solution.compute.lds_swizzle_chunk_b == 8

    key = GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_xxs(35), solution
    )
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)

    physical = derive_grouped_backward_pair_physical_plan(
        DerivedGroupedBackwardPairState.from_solution_key(key)
    )
    assert physical.ordinary.resources.total_vgprs == 101
    assert physical.ordinary.resources.lds_num_bytes == 10_240
    assert physical.second_projection is not None

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert source.count("ds_load_b128") == 64
    assert source.count("ds_load_b64") == 8


def test_iq2_xxs_m64_direct_pointers_reuse_addresses_without_swaps() -> None:
    solution = GroupedBackwardPairSolution.iq2_xxs_m64_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a()
    key = GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_xxs(35), solution
    )
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)

    physical = derive_grouped_backward_pair_physical_plan(
        DerivedGroupedBackwardPairState.from_solution_key(key)
    )
    assert physical.ordinary.resources.total_vgprs == 101
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )
    assert (
        physical.scalar.second_packed_weight
        == physical.second_projection.registers.kernarg + 2
    )

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Swap only the active packed-bank pointer pair" not in source
    assert "Swap only the active gradient pointer pair" not in source
    second_packed = (
        f"s[{physical.scalar.second_packed_weight}:"
        f"{physical.scalar.second_packed_weight + 1}]"
    )
    assert source.count(f"{second_packed} offset:2") == 2
    assert source.count(second_packed) == 5


def test_iq2_xxs_m128_direct_pointers_swizzle8_identity_and_source() -> None:
    solution = GroupedBackwardPairSolution.iq2_xxs_m128_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a()
    compute = solution.compute
    assert compute.matrix_instruction == (16, 16, 16, 1, 1, 2, 4, 4, 1)
    assert (compute.macro_tile0, compute.macro_tile1, compute.depth_u) == (128, 64, 32)
    assert (compute.prefetch_global_read, compute.schedule_iter_alg) == (2, 4)
    assert compute.lds_swizzle_chunk_b == 8

    key = GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_xxs(35), solution
    )
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)

    physical = derive_grouped_backward_pair_physical_plan(
        DerivedGroupedBackwardPairState.from_solution_key(key)
    )
    resources = physical.ordinary.resources
    assert (resources.total_vgprs, resources.total_sgprs) == (149, 41)
    assert resources.lds_num_bytes == 10_240
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Swap only the active packed-bank pointer pair" not in source
    assert "Swap only the active gradient pointer pair" not in source
    assert source.count("ds_load_b128") == 64
    assert source.count("v_wmma_f32_16x16x16_bf16") == 64


def test_iq2_xxs_m192_lower_state_identity_physical_plan_and_source() -> None:
    solution = GroupedBackwardPairSolution.iq2_xxs_m192_n64_dual_lds_full_tile_split_direct_pointers_overlap_second_read_prefetch_a()
    compute = solution.compute
    assert compute.matrix_instruction == (16, 16, 16, 1, 1, 3, 4, 4, 1)
    assert (compute.macro_tile0, compute.macro_tile1, compute.depth_u) == (192, 64, 32)
    assert (compute.prefetch_global_read, compute.schedule_iter_alg) == (2, 4)
    assert compute.lds_swizzle_chunk_b == 8

    key = GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_xxs(257), solution
    )
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)

    physical = derive_grouped_backward_pair_physical_plan(
        DerivedGroupedBackwardPairState.from_solution_key(key)
    )
    resources = physical.ordinary.resources
    assert (resources.total_vgprs, resources.total_sgprs) == (188, 41)
    assert resources.lds_num_bytes == 10_240
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Issue the second packed projection reads alongside first A prefetch" in source
    assert "s_mul_i32" in source
    assert "v_mul_lo_u32" in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 96
    assert source.count("s_waitcnt vmcnt(6) lgkmcnt(0)") == 6
    assert "s_waitcnt vmcnt(14) lgkmcnt(0)" not in source


def test_iq2_xxs_m192_requires_exact_identity() -> None:
    solution = GroupedBackwardPairSolution.iq2_xxs_m192_n64_dual_lds_full_tile_split_direct_pointers_overlap_second_read_prefetch_a()
    synthetic = replace(
        solution,
        projection_schedule=(
            GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchAInterleavedDepthU
        ),
    )
    key = GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_xxs(257), synthetic
    )
    reasons = validate_grouped_backward_pair_solution(key)
    assert [reason.message for reason in reasons] == [
        "paired backward M tile must be 64, 128, or the exact IQ2_XXS M192 identity"
    ]
    with pytest.raises(SchemaError, match="exact IQ2_XXS M192 identity"):
        GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping())


def test_iq2_xxs_m64_sia5_k_pipeline_identity_and_source() -> None:
    solution = GroupedBackwardPairSolution.iq2_xxs_m64_n64_sia5_dual_lds_full_tile_split_k_pipeline()
    compute = solution.compute
    assert compute.matrix_instruction == (16, 16, 16, 1, 1, 1, 4, 4, 1)
    assert (compute.macro_tile0, compute.macro_tile1, compute.depth_u) == (64, 64, 32)
    assert (compute.prefetch_global_read, compute.schedule_iter_alg) == (2, 5)
    assert compute.prefetch_packed_weight
    assert compute.prefetch_packed_weight_next
    assert compute.decoder_width == 16
    assert compute.lds_swizzle_chunk_b == 8
    assert solution.projection_schedule is (
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineInterleavedDepthU
    )

    key = GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_xxs(35), solution
    )
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)

    physical = derive_grouped_backward_pair_physical_plan(
        DerivedGroupedBackwardPairState.from_solution_key(key)
    )
    resources = physical.ordinary.resources
    assert (resources.total_vgprs, resources.total_sgprs) == (101, 41)
    assert resources.lds_num_bytes == 10_240
    assert resources.private_segment_bytes == 0
    assert physical.ordinary.lds.base_offset == 0
    assert physical.ordinary.lds.num_bytes == 4_096
    assert physical.ordinary.lds.codebook_offset == 8_192
    assert physical.ordinary.lds.codebook_in_lds
    assert physical.second_projection is not None
    assert physical.second_projection.lds.base_offset == 4_096
    assert physical.second_projection.lds.codebook_offset == 8_192
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )
    assert (
        physical.scalar.second_packed_weight
        == physical.second_projection.registers.kernarg + 2
    )

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Swap only the active packed-bank pointer pair" not in source
    assert "Swap only the active gradient pointer pair" not in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    assert source.count("s_barrier") == 7
    assert source.count("s_add_u32 s5, s5, 32") == 2
    assert source.count("s_waitcnt vmcnt(2) lgkmcnt(2)") == 4
    assert source.count("s_waitcnt vmcnt(0) lgkmcnt(2)") == 4
    assert "s_waitcnt vmcnt(4) lgkmcnt(2)" not in source

    first_payload = physical.ordinary.registers.global_read_b
    second_payload = physical.second_projection.registers.global_read_b
    for body in ("Full", "Tail"):
        compute_label = f".LPairKPipelineCompute{body}:"
        next_packed_label = f".LPairKPipelineNextPacked{body}:"
        packed_done_label = f".LPairKPipelinePackedDone{body}:"
        done_label = f".LPairKPipelineDone{body}:"
        assert compute_label in source
        assert next_packed_label in source
        assert packed_done_label in source
        assert done_label in source

        overlap_start = source.index(
            "// Decode prefetched banks after their LDS consumers retire.",
            source.index(compute_label),
        )
        overlap_end = source.index(
            f"s_branch .LPairKPipelineCompute{body}", overlap_start
        )
        overlap = source[overlap_start:overlap_end]
        first_read = overlap.index(
            f"ds_load_b64 v[{first_payload}:{first_payload + 1}]"
        )
        second_read = overlap.index(
            f"ds_load_b64 v[{second_payload}:{second_payload + 1}]"
        )
        first_sign = overlap.index(
            "// Sign the two IQ2_XXS grids and reconstruct their scale."
        )
        first_decode = overlap.index(
            "// Decode IQ2_XXS signed codebook values into LDS."
        )
        second_sign = overlap.index(
            "// Sign the two IQ2_XXS grids and reconstruct their scale.",
            first_decode,
        )
        second_decode = overlap.index(
            "// Decode IQ2_XXS signed codebook values into LDS.",
            second_sign,
        )
        assert first_read < second_read < first_sign < first_decode
        assert first_decode < second_sign < second_decode
        assert first_sign < overlap.index("s_waitcnt lgkmcnt(2)") < first_decode
        assert second_sign < overlap.index("s_waitcnt lgkmcnt(16)") < second_decode

    tail = source[source.index(".LPairKPipelineComputeTail:") :]
    assert tail.index("s_add_u32 s5, s5, 32") < tail.index(
        "s_cbranch_scc1 .LGroupedBackwardInactiveWave0TailSecondLds"
    )


def test_iq2_xxs_m64_sia5_k_pipeline_requires_exact_identity() -> None:
    solution = GroupedBackwardPairSolution.iq2_xxs_m64_n64_sia5_dual_lds_full_tile_split_k_pipeline()
    synthetic = replace(
        solution,
        compute=replace(solution.compute, schedule_iter_alg=4),
    )
    key = GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_xxs(35), synthetic
    )
    reasons = validate_grouped_backward_pair_solution(key)
    assert [reason.message for reason in reasons] == [
        "paired IQ2_XXS schedule requires its staged-codebook lowering"
    ]
    with pytest.raises(SchemaError, match="staged-codebook lowering"):
        GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping())


def test_iq2_s_m64_still_rejects_dual_lds() -> None:
    solution = replace(
        GroupedBackwardPairSolution.iq2_s_m64_n64(),
        projection_schedule=(
            GroupedBackwardPairProjectionSchedule.DualLdsInterleavedDepthU
        ),
    )
    key = GroupedBackwardPairSolutionKey(GroupedBackwardPairProblem.iq2_s(35), solution)
    reasons = validate_grouped_backward_pair_solution(key)
    assert [reason.message for reason in reasons] == [
        "paired backward dual LDS requires M128 or a qualified staged M64 identity"
    ]


def test_iq2_xxs_prefetch_a_wait_frontier_tracks_m_tiles() -> None:
    toolchain = Toolchain.discover()
    m64 = GroupedBackwardPairSolution.iq2_xxs_m64_n64_dual_lds_full_tile_split_concurrent_reads_prefetch_a()
    sources = {
        64: GroupedBackwardPairKernelWriterAssembly(
            GroupedBackwardPairSolutionKey(GroupedBackwardPairProblem.iq2_xxs(35), m64),
            toolchain,
        ).source(),
        128: GroupedBackwardPairKernelWriterAssembly(
            GroupedBackwardPairSolutionKey(
                GroupedBackwardPairProblem.iq2_xxs(35),
                GroupedBackwardPairSolution.iq2_xxs_m128_n64_dual_lds_full_tile_split_concurrent_reads_prefetch_a(),
            ),
            toolchain,
        ).source(),
    }
    assert sources[64].count("s_waitcnt vmcnt(2) lgkmcnt(0)") == 6
    assert "s_waitcnt vmcnt(4) lgkmcnt(0)" not in sources[64]
    assert sources[128].count("s_waitcnt vmcnt(4) lgkmcnt(0)") == 6
    assert "s_waitcnt vmcnt(2) lgkmcnt(0)" not in sources[128]


@pytest.mark.parametrize(
    "schedule",
    (
        GroupedBackwardPairProjectionSchedule.DualLdsGlobalCodebookInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineGlobalCodebookInterleaveWmmaWaitsDepthU,
    ),
)
def test_iq2_xxs_rejects_nonstaged_schedule(
    schedule: GroupedBackwardPairProjectionSchedule,
) -> None:
    solution = replace(
        GroupedBackwardPairSolution.iq2_xxs_m128_n64(),
        projection_schedule=schedule,
    )
    key = GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_xxs(35), solution
    )
    reasons = validate_grouped_backward_pair_solution(key)
    assert [reason.message for reason in reasons] == [
        "paired IQ2_XXS schedule requires its staged-codebook lowering"
    ]
    with pytest.raises(SchemaError, match="staged-codebook lowering"):
        GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping())


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
    assert f"v_and_b32 v{registers.temporary}, 1, v{registers.serial}" in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == wmmas
    assert source.count("s_barrier") == 5


@pytest.mark.parametrize(("tile", "vgprs", "wmmas"), ((64, 87, 16), (128, 127, 32)))
def test_q3_k_backward_pair_physical_and_source(
    tile: int, vgprs: int, wmmas: int
) -> None:
    key = _q3_k_key(rows=35, tile=tile)
    state = DerivedGroupedBackwardPairState.from_solution_key(key)
    physical = derive_grouped_backward_pair_physical_plan(state)
    assert physical.ordinary.resources.total_vgprs == vgprs
    assert physical.ordinary.resources.total_sgprs == 41
    assert physical.ordinary.resources.lds_num_bytes == 5_120
    assert state.expected_packed_weight_shape == (256, 512, 880)
    assert state.bytes_per_expert == 450_560

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Build Q3_K block addresses for decoder-owned output rows" in source
    assert "Decode Q3_K 2-bit payload and high mask into LDS" in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == wmmas
    assert source.count("s_barrier") == 4


def test_q3_k_backward_pair_dual_lds_physical_and_source() -> None:
    key = GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.q3_k(35),
        GroupedBackwardPairSolution.q3_k_m128_n64_dual_lds(),
    )
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)

    state = DerivedGroupedBackwardPairState.from_solution_key(key)
    physical = derive_grouped_backward_pair_physical_plan(state)
    assert physical.ordinary.resources.total_vgprs == 127
    assert physical.ordinary.resources.lds_num_bytes == 10_240
    assert physical.second_projection is not None
    assert physical.second_projection.lds.base_offset == 5_120

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert source.count("Decode the first pair projection into disjoint LDS") == 1
    assert source.count("Decode the second pair projection into disjoint LDS") == 1
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    assert source.count("s_barrier") == 2


def test_q3_k_backward_pair_dual_lds_full_tile_split() -> None:
    solution = GroupedBackwardPairSolution.q3_k_m128_n64_dual_lds_full_tile_split()
    key = GroupedBackwardPairSolutionKey(GroupedBackwardPairProblem.q3_k(257), solution)
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)

    physical = derive_grouped_backward_pair_physical_plan(
        DerivedGroupedBackwardPairState.from_solution_key(key)
    )
    assert physical.ordinary.resources.total_vgprs == 127
    assert physical.ordinary.resources.lds_num_bytes == 10_240
    assert physical.second_projection is not None
    assert physical.second_projection.lds.base_offset == 5_120

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Decode the first pair projection into disjoint LDS" in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 64
    assert source.count("s_barrier") == 4


def test_q3_k_backward_pair_full_tile_direct_pointers() -> None:
    solution = GroupedBackwardPairSolution.q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers()
    key = GroupedBackwardPairSolutionKey(GroupedBackwardPairProblem.q3_k(257), solution)
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)

    physical = derive_grouped_backward_pair_physical_plan(
        DerivedGroupedBackwardPairState.from_solution_key(key)
    )
    assert physical.ordinary.resources.total_vgprs == 127
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Swap the active gradient and packed-bank pointer pairs" not in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 64
    assert source.count("s_barrier") == 4


def test_q3_k_backward_pair_serial_reads_prefetch_a() -> None:
    solution = GroupedBackwardPairSolution.q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a()
    key = GroupedBackwardPairSolutionKey(GroupedBackwardPairProblem.q3_k(257), solution)
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)

    physical = derive_grouped_backward_pair_physical_plan(
        DerivedGroupedBackwardPairState.from_solution_key(key)
    )
    assert physical.ordinary.resources.total_vgprs == 143
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Prefetch A fragments" in source
    assert "Issue both packed projection reads" not in source
    assert "Swap the active gradient and packed-bank pointer pairs" not in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 64
    assert source.count("s_barrier") == 4


def test_q3_k_backward_pair_m64_serial_reads_prefetch_a() -> None:
    solution = GroupedBackwardPairSolution.q3_k_m64_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a()
    key = GroupedBackwardPairSolutionKey(GroupedBackwardPairProblem.q3_k(257), solution)
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)

    physical = derive_grouped_backward_pair_physical_plan(
        DerivedGroupedBackwardPairState.from_solution_key(key)
    )
    assert physical.ordinary.resources.total_vgprs == 95
    assert physical.ordinary.resources.lds_num_bytes == 10_240
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Prefetch A fragments" in source
    assert "Issue both packed projection reads" not in source
    assert "Swap the active gradient and packed-bank pointer pairs" not in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    assert source.count("s_barrier") == 4


def test_q3_k_backward_pair_overlap_second_read_prefetch_a() -> None:
    solution = GroupedBackwardPairSolution.q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers_overlap_second_read_prefetch_a()
    key = GroupedBackwardPairSolutionKey(GroupedBackwardPairProblem.q3_k(257), solution)
    assert GroupedBackwardPairSolutionKey.from_mapping(key.to_mapping()) == key
    assert not validate_grouped_backward_pair_solution(key)

    physical = derive_grouped_backward_pair_physical_plan(
        DerivedGroupedBackwardPairState.from_solution_key(key)
    )
    assert physical.ordinary.resources.total_vgprs == 143
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )

    source = GroupedBackwardPairKernelWriterAssembly(key, Toolchain.discover()).source()
    overlap = source.index("Issue the second packed projection reads")
    first_a = source.index("Prefetch A fragments", overlap)
    wait = source.index("s_waitcnt vmcnt(0)", first_a)
    second_decode = source.index("Decode Q3_K", wait)
    assert overlap < first_a < wait < second_decode
    assert "Issue both packed projection reads" not in source
    assert "Swap the active gradient and packed-bank pointer pairs" not in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 64
    assert source.count("s_barrier") == 4


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
    assert source.count("s_waitcnt vmcnt(4) lgkmcnt(0)") == 8
    assert "s_waitcnt vmcnt(14) lgkmcnt(0)" not in source
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


def test_installed_q3_k_backward_pair_dispatch_threshold() -> None:
    control = InstalledGroupedBackwardPairQ3KControl
    assert control.dispatched_macro_tile(35, 4) == 64
    assert control.dispatched_macro_tile(16_384, 249) == 64
    assert control.dispatched_macro_tile(16_384, 125) == 128
    assert control.BYTES_PER_EXPERT == 450_560


@pytest.mark.parametrize(
    ("rows", "route_entries", "constructor"),
    (
        (
            16_384,
            249,
            "q3_k_m64_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a",
        ),
        (
            16_384,
            128,
            "q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a",
        ),
        (
            65_536,
            125,
            "q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a",
        ),
        (
            262_144,
            125,
            "q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers_overlap_second_read_prefetch_a",
        ),
    ),
)
def test_research_q3_k_selector_uses_confirmed_mixed_dispatch(
    rows: int, route_entries: int, constructor: str
) -> None:
    key = ResearchGroupedBackwardPairQ3KSelector.select_solution_key(
        rows, route_entries
    )
    expected = getattr(GroupedBackwardPairSolution, constructor)()
    assert key.problem == GroupedBackwardPairProblem.q3_k(rows)
    assert key.solution == expected
    assert key.solution.macro_tile0 == (64 if "m64" in constructor else 128)
    assert not validate_grouped_backward_pair_solution(key)


@pytest.mark.parametrize(
    ("rows", "route_entries"),
    ((16_384, 129), (16_384, 128), (65_536, 256)),
)
def test_research_q3_k_selector_matches_threshold_at_boundary(
    rows: int, route_entries: int
) -> None:
    expected = InstalledGroupedBackwardPairQ3KControl.dispatched_macro_tile(
        rows, route_entries
    )
    assert (
        ResearchGroupedBackwardPairQ3KSelector.selected_macro_tile(rows, route_entries)
        == expected
    )


@pytest.mark.parametrize(
    ("rows", "route_entries"),
    (
        (35, 4),
        (0, 4),
        (16_384, 0),
        (16_384, 257),
        (True, 4),
        (16_384.0, 4),
        (16_384, 4.0),
    ),
)
def test_research_q3_k_selector_rejects_unqualified_dimensions(
    rows: int, route_entries: int
) -> None:
    with pytest.raises(ValueError):
        ResearchGroupedBackwardPairQ3KSelector.select_solution_key(rows, route_entries)


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


def test_iq2_xxs_m64_sia5_pipeline_build_is_deterministic_and_inspectable(
    tmp_path,
) -> None:
    key = GroupedBackwardPairSolutionKey(
        GroupedBackwardPairProblem.iq2_xxs(35),
        GroupedBackwardPairSolution.iq2_xxs_m64_n64_sia5_dual_lds_full_tile_split_k_pipeline(),
    )
    toolchain = Toolchain.discover()
    build_hashes = []
    for directory_name in ("first", "second"):
        directory = tmp_path / directory_name
        directory.mkdir()
        assembly = directory / "kernel.s"
        object_path = directory / "kernel.o"
        code_object = directory / "kernel.hsaco"
        GroupedBackwardPairKernelWriterAssembly(key, toolchain).write(assembly)
        toolchain.assemble(assembly, object_path)
        toolchain.link(object_path, code_object)
        build_hashes.append(
            tuple(
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (assembly, object_path, code_object)
            )
        )

    assert build_hashes[0] == build_hashes[1]
    result = inspect_grouped_backward_pair_artifact(
        key, tmp_path / "second" / "kernel.hsaco", toolchain
    )
    assert result.kernarg_segment_size == 72
    assert result.max_flat_workgroup_size == 128
    assert result.vgpr_count == 101
    assert result.sgpr_count == 41
    assert result.lds_num_bytes == 10_240
    assert result.private_segment_bytes == 0
    assert result.vgpr_spill_count == 0
    assert result.sgpr_spill_count == 0
    assert result.wmma_count == 32
    assert result.barrier_count == 7


def test_iq2_xxs_backward_pair_build_and_inspection(tmp_path) -> None:
    key = _iq2_xxs_key(rows=35)
    toolchain = Toolchain.discover()
    writer = GroupedBackwardPairKernelWriterAssembly(key, toolchain)
    assert (
        writer.source()
        == GroupedBackwardPairKernelWriterAssembly(key, toolchain).source()
    )
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


def test_q3_k_backward_pair_build_and_inspection(tmp_path) -> None:
    key = _q3_k_key(rows=35)
    toolchain = Toolchain.discover()
    writer = GroupedBackwardPairKernelWriterAssembly(key, toolchain)
    assembly = tmp_path / "q3_k_pair.s"
    object_path = tmp_path / "q3_k_pair.o"
    code_object = tmp_path / "q3_k_pair.hsaco"
    writer.write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    result = inspect_grouped_backward_pair_artifact(key, code_object, toolchain)
    assert result.kernarg_segment_size == 72
    assert result.vgpr_count == 87
    assert result.sgpr_count == 41
    assert result.lds_num_bytes == 5_120
    assert result.private_segment_bytes == 0
    assert result.vgpr_spill_count == 0
    assert result.sgpr_spill_count == 0
    assert result.wmma_count == 16
    assert result.barrier_count == 4
