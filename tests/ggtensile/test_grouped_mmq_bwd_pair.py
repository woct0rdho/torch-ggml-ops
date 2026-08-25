import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from tests.ggtensile.grouped_backward_pair_fixtures import (
    GroupedBackwardPairTestSolutions as _Solutions,
)
from tests.ggtensile.grouped_backward_pair_fixtures import (
    grouped_backward_pair_problem,
)
from tools.ggtensile.family_registry import (
    instance_name,
    launch_for_instance,
    mapping_for_instance,
    parse_instance,
    problem_type_for_family,
    writer_for_instance,
)
from tools.ggtensile.grouped_mmq_bwd_pair_inspection import (
    inspect_grouped_backward_pair_artifact,
)
from tools.ggtensile.grouped_mmq_bwd_pair_model import (
    GroupedBackwardPairProblem,
    GroupedBackwardPairProjectionPolicy,
    PairCodebookStorage,
    PairPointerMode,
    PairReadConcurrency,
)
from tools.ggtensile.grouped_mmq_bwd_pair_physical import (
    derive_grouped_backward_pair_physical_plan,
)
from tools.ggtensile.grouped_mmq_bwd_pair_spec import (
    DerivedGroupedBackwardPairState,
    GroupedBackwardPairKernelSpec,
)
from tools.ggtensile.grouped_mmq_bwd_pair_validation import (
    validate_grouped_backward_pair_solution,
)
from tools.ggtensile.identity import KernelFamily
from tools.ggtensile.iq2_s_grid import iq2_s_grid_values
from tools.ggtensile.kernel_abi import GROUPED_BACKWARD_PAIR_ABI
from tools.ggtensile.kernel_instance import KernelInstance
from tools.ggtensile.toolchain import Toolchain


def _projection_policy(
    *,
    dual_lds: bool = True,
    codebook_storage: PairCodebookStorage = PairCodebookStorage.Lds,
    split_full_tiles: bool = False,
    pointer_mode: PairPointerMode = PairPointerMode.Swapped,
    read_concurrency: PairReadConcurrency = PairReadConcurrency.Serial,
    activation_prefetch: bool = False,
    pipeline_k: bool = False,
) -> GroupedBackwardPairProjectionPolicy:
    return GroupedBackwardPairProjectionPolicy(
        dual_lds,
        codebook_storage,
        split_full_tiles,
        pointer_mode,
        read_concurrency,
        activation_prefetch,
        pipeline_k,
    )


def _instance(
    problem: GroupedBackwardPairProblem,
    spec: GroupedBackwardPairKernelSpec,
) -> KernelInstance:
    return KernelInstance.for_gfx1151(
        KernelFamily.GroupedBackwardPair,
        problem_type_for_family(
            KernelFamily.GroupedBackwardPair, problem.quant_data_type
        ),
        problem,
        spec,
    )


def _problem_spec(
    instance: KernelInstance,
) -> tuple[GroupedBackwardPairProblem, GroupedBackwardPairKernelSpec]:
    assert isinstance(instance.problem, GroupedBackwardPairProblem)
    assert isinstance(instance.kernel_spec, GroupedBackwardPairKernelSpec)
    return instance.problem, instance.kernel_spec


def _validate(instance: KernelInstance) -> None:
    validate_grouped_backward_pair_solution(*_problem_spec(instance))


def _state(instance: KernelInstance) -> DerivedGroupedBackwardPairState:
    return DerivedGroupedBackwardPairState.from_problem_spec(*_problem_spec(instance))


def _writer(instance: KernelInstance, toolchain: Toolchain):
    return writer_for_instance(instance, toolchain)


def _inspect(instance: KernelInstance, code_object: Path, toolchain: Toolchain):
    return inspect_grouped_backward_pair_artifact(
        *_problem_spec(instance),
        instance_name(instance),
        code_object,
        toolchain,
    )


def _key(rows: int = 16_384, tile: int = 64) -> KernelInstance:
    solution = _Solutions.iq2_s_m64_n64() if tile == 64 else _Solutions.iq2_s_m128_n64()
    return _instance(grouped_backward_pair_problem("IQ2_S", rows), solution)


def _iq2_xxs_key(rows: int = 12_288, tile: int = 64) -> KernelInstance:
    solution = (
        _Solutions.iq2_xxs_m64_n64() if tile == 64 else _Solutions.iq2_xxs_m128_n64()
    )
    return _instance(grouped_backward_pair_problem("IQ2_XXS", rows), solution)


def _q3_k_key(rows: int = 16_384, tile: int = 64) -> KernelInstance:
    solution = _Solutions.q3_k_m64_n64() if tile == 64 else _Solutions.q3_k_m128_n64()
    return _instance(grouped_backward_pair_problem("Q3_K", rows), solution)


@pytest.mark.parametrize("rows", (35, 16_384, 65_536, 262_144))
@pytest.mark.parametrize("tile", (64, 128))
def test_backward_pair_identity_roundtrip(rows: int, tile: int) -> None:
    key = _key(rows, tile)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)
    assert "grouped_mmq_bwd_pair_iq2_s" in instance_name(key)


@pytest.mark.parametrize("rows", (35, 12_288, 49_152, 196_608))
@pytest.mark.parametrize("tile", (64, 128))
def test_iq2_xxs_backward_pair_identity_roundtrip(rows: int, tile: int) -> None:
    key = _iq2_xxs_key(rows, tile)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)
    assert "grouped_mmq_bwd_pair_iq2_xxs" in instance_name(key)


@pytest.mark.parametrize("rows", (35, 16_384, 65_536, 262_144))
@pytest.mark.parametrize("tile", (64, 128))
def test_q3_k_backward_pair_identity_roundtrip(rows: int, tile: int) -> None:
    key = _q3_k_key(rows, tile)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)
    assert "grouped_mmq_bwd_pair_q3_k" in instance_name(key)


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
    solution = getattr(_Solutions, constructor)()
    key = _instance(grouped_backward_pair_problem("IQ2_XXS", 35), solution)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)

    physical = derive_grouped_backward_pair_physical_plan(_state(key))
    assert physical.ordinary.resources.vgprs == expected_vgprs
    assert physical.ordinary.resources.lds_bytes == 10_240
    assert physical.ordinary.lds.codebook_offset == 8_192
    assert physical.ordinary.lds.codebook_in_lds
    assert physical.second_projection is not None
    assert physical.second_projection.lds.base_offset == 4_096


@pytest.mark.parametrize(
    ("policy", "prefetch_a", "expected_vgprs"),
    (
        (_projection_policy(), False, 87),
        (
            _projection_policy(split_full_tiles=True),
            False,
            87,
        ),
        (
            _projection_policy(
                split_full_tiles=True,
                read_concurrency=PairReadConcurrency.Concurrent,
            ),
            False,
            97,
        ),
        (
            _projection_policy(
                split_full_tiles=True,
                read_concurrency=PairReadConcurrency.Concurrent,
                activation_prefetch=True,
            ),
            True,
            105,
        ),
    ),
)
def test_iq2_xxs_m64_staged_schedule_identity_and_physical_plan(
    policy: GroupedBackwardPairProjectionPolicy,
    prefetch_a: bool,
    expected_vgprs: int,
) -> None:
    solution = _Solutions.iq2_xxs_m64_n64()
    solution = replace(
        solution,
        compute=replace(
            solution.compute,
            pipeline=replace(
                solution.compute.pipeline,
                global_read_prefetch=2 if prefetch_a else 1,
                iteration=replace(
                    solution.compute.pipeline.iteration,
                    prefetch_activation=prefetch_a,
                    interleave_wmma_waits=False,
                ),
            ),
        ),
        projection_policy=policy,
    )
    key = _instance(grouped_backward_pair_problem("IQ2_XXS", 35), solution)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)

    physical = derive_grouped_backward_pair_physical_plan(_state(key))
    assert physical.ordinary.resources.vgprs == expected_vgprs
    assert physical.ordinary.resources.lds_bytes == 10_240
    assert physical.second_projection is not None


def test_iq2_xxs_m64_prefetch_a_swizzle8_selected_physical_plan() -> None:
    solution = _Solutions.iq2_xxs_m64_n64_dual_lds_full_tile_split_concurrent_reads_prefetch_a()
    assert solution.compute.pipeline.global_read_prefetch == 2
    assert solution.compute.pipeline.iteration.prefetch_activation
    assert not solution.compute.pipeline.iteration.interleave_wmma_waits
    assert solution.compute.memory.lds_swizzle_chunk_b == 8

    key = _instance(grouped_backward_pair_problem("IQ2_XXS", 35), solution)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)

    physical = derive_grouped_backward_pair_physical_plan(_state(key))
    assert physical.ordinary.resources.vgprs == 101
    assert physical.ordinary.resources.lds_bytes == 10_240
    assert physical.second_projection is not None

    source = _writer(key, Toolchain.discover()).source()
    assert source.count("ds_load_b128") == 64
    assert source.count("ds_load_b64") == 8


def test_iq2_xxs_m64_direct_pointers_reuse_addresses_without_swaps() -> None:
    solution = (
        _Solutions.iq2_xxs_m64_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a()
    )
    key = _instance(grouped_backward_pair_problem("IQ2_XXS", 35), solution)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)

    physical = derive_grouped_backward_pair_physical_plan(_state(key))
    assert physical.ordinary.resources.vgprs == 101
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )
    assert (
        physical.scalar.second_packed_weight
        == physical.second_projection.registers.kernarg + 2
    )

    source = _writer(key, Toolchain.discover()).source()
    assert "Swap only the active packed-bank pointer pair" not in source
    assert "Swap only the active gradient pointer pair" not in source
    second_packed = (
        f"s[{physical.scalar.second_packed_weight}:"
        f"{physical.scalar.second_packed_weight + 1}]"
    )
    assert source.count(f"{second_packed} offset:2") == 2
    assert source.count(second_packed) == 5


def test_iq2_xxs_m128_direct_pointers_swizzle8_identity_and_source() -> None:
    solution = _Solutions.iq2_xxs_m128_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a()
    compute = solution.compute
    assert compute.geometry.matrix_instruction == (16, 16, 16, 1, 1, 2, 4, 4, 1)
    assert (
        compute.geometry.macro_tile0,
        compute.geometry.macro_tile1,
        compute.geometry.depth_u,
    ) == (128, 64, 32)
    assert compute.pipeline.global_read_prefetch == 2
    assert compute.pipeline.iteration.prefetch_activation
    assert not compute.pipeline.iteration.interleave_wmma_waits
    assert compute.memory.lds_swizzle_chunk_b == 8

    key = _instance(grouped_backward_pair_problem("IQ2_XXS", 35), solution)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)

    physical = derive_grouped_backward_pair_physical_plan(_state(key))
    resources = physical.ordinary.resources
    assert (resources.vgprs, resources.sgprs) == (149, 41)
    assert resources.lds_bytes == 10_240
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )

    source = _writer(key, Toolchain.discover()).source()
    assert "Swap only the active packed-bank pointer pair" not in source
    assert "Swap only the active gradient pointer pair" not in source
    assert source.count("ds_load_b128") == 64
    assert source.count("v_wmma_f32_16x16x16_bf16") == 64


def test_iq2_xxs_m192_lower_state_identity_physical_plan_and_source() -> None:
    solution = _Solutions.iq2_xxs_m192_n64_dual_lds_full_tile_split_direct_pointers_overlap_second_read_prefetch_a()
    compute = solution.compute
    assert compute.geometry.matrix_instruction == (16, 16, 16, 1, 1, 3, 4, 4, 1)
    assert (
        compute.geometry.macro_tile0,
        compute.geometry.macro_tile1,
        compute.geometry.depth_u,
    ) == (192, 64, 32)
    assert compute.pipeline.global_read_prefetch == 2
    assert compute.pipeline.iteration.prefetch_activation
    assert not compute.pipeline.iteration.interleave_wmma_waits
    assert compute.memory.lds_swizzle_chunk_b == 8

    key = _instance(grouped_backward_pair_problem("IQ2_XXS", 257), solution)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)

    physical = derive_grouped_backward_pair_physical_plan(_state(key))
    resources = physical.ordinary.resources
    assert (resources.vgprs, resources.sgprs) == (188, 41)
    assert resources.lds_bytes == 10_240
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )

    source = _writer(key, Toolchain.discover()).source()
    assert (
        "Issue the second packed projection reads alongside first A prefetch" in source
    )
    assert "s_mul_i32" in source
    assert "v_mul_lo_u32" in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 96
    assert source.count("s_waitcnt vmcnt(6) lgkmcnt(0)") == 6
    assert "s_waitcnt vmcnt(14) lgkmcnt(0)" not in source


def test_iq2_xxs_m64_sia5_k_pipeline_identity_and_source() -> None:
    solution = _Solutions.iq2_xxs_m64_n64_sia5_dual_lds_full_tile_split_k_pipeline()
    compute = solution.compute
    assert compute.geometry.matrix_instruction == (16, 16, 16, 1, 1, 1, 4, 4, 1)
    assert (
        compute.geometry.macro_tile0,
        compute.geometry.macro_tile1,
        compute.geometry.depth_u,
    ) == (64, 64, 32)
    assert compute.pipeline.global_read_prefetch == 2
    assert compute.pipeline.iteration.prefetch_activation
    assert compute.pipeline.iteration.interleave_wmma_waits
    assert compute.pipeline.prefetch_packed_weight
    assert compute.pipeline.prefetch_next_packed_weight
    assert compute.decode.decoder_width == 16
    assert compute.memory.lds_swizzle_chunk_b == 8
    assert solution.projection_policy == _projection_policy(
        split_full_tiles=True,
        pointer_mode=PairPointerMode.Direct,
        read_concurrency=PairReadConcurrency.Concurrent,
        activation_prefetch=True,
        pipeline_k=True,
    )

    key = _instance(grouped_backward_pair_problem("IQ2_XXS", 35), solution)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)

    physical = derive_grouped_backward_pair_physical_plan(_state(key))
    resources = physical.ordinary.resources
    assert (resources.vgprs, resources.sgprs) == (101, 41)
    assert resources.lds_bytes == 10_240
    assert resources.private_bytes == 0
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

    source = _writer(key, Toolchain.discover()).source()
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


def test_iq2_xxs_prefetch_a_wait_frontier_tracks_m_tiles() -> None:
    toolchain = Toolchain.discover()
    m64 = _Solutions.iq2_xxs_m64_n64_dual_lds_full_tile_split_concurrent_reads_prefetch_a()
    sources = {
        64: _writer(
            _instance(grouped_backward_pair_problem("IQ2_XXS", 35), m64),
            toolchain,
        ).source(),
        128: _writer(
            _instance(
                grouped_backward_pair_problem("IQ2_XXS", 35),
                _Solutions.iq2_xxs_m128_n64_dual_lds_full_tile_split_concurrent_reads_prefetch_a(),
            ),
            toolchain,
        ).source(),
    }
    assert sources[64].count("s_waitcnt vmcnt(2) lgkmcnt(0)") == 6
    assert "s_waitcnt vmcnt(4) lgkmcnt(0)" not in sources[64]
    assert sources[128].count("s_waitcnt vmcnt(4) lgkmcnt(0)") == 6
    assert "s_waitcnt vmcnt(2) lgkmcnt(0)" not in sources[128]


def test_iq2_s_packed_negative_payload_identity() -> None:
    mask = 0xFFFFFFFF
    for grid_value in iq2_s_grid_values():
        for payload in (grid_value & mask, grid_value >> 32):
            software_negation = ((~payload & mask) + 0x01010101) & mask
            packed_subtract = (0x01010100 - payload) & mask
            assert software_negation == packed_subtract


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
    state = _state(key)
    physical = derive_grouped_backward_pair_physical_plan(state)
    assert physical.ordinary.resources.vgprs == vgprs
    assert physical.ordinary.resources.sgprs == 41
    assert physical.ordinary.resources.lds_bytes == 12_288
    assert state.expected_packed_weight_shape == (256, 512, 656)
    assert state.bytes_per_expert == 335_872

    source = _writer(key, Toolchain.discover()).source()
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
    state = _state(key)
    physical = derive_grouped_backward_pair_physical_plan(state)
    assert physical.ordinary.resources.vgprs == vgprs
    assert physical.ordinary.resources.sgprs == 41
    assert physical.ordinary.resources.lds_bytes == 6_144
    assert state.expected_packed_weight_shape == (256, 2048, 1056)
    assert state.bytes_per_expert == 2_162_688

    source = _writer(key, Toolchain.discover()).source()
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
    state = _state(key)
    physical = derive_grouped_backward_pair_physical_plan(state)
    assert physical.ordinary.resources.vgprs == vgprs
    assert physical.ordinary.resources.sgprs == 41
    assert physical.ordinary.resources.lds_bytes == 5_120
    assert state.expected_packed_weight_shape == (256, 512, 880)
    assert state.bytes_per_expert == 450_560

    source = _writer(key, Toolchain.discover()).source()
    assert "Build Q3_K block addresses for decoder-owned output rows" in source
    assert "Decode Q3_K 2-bit payload and high mask into LDS" in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == wmmas
    assert source.count("s_barrier") == 4


def test_q3_k_backward_pair_dual_lds_physical_and_source() -> None:
    key = _instance(
        grouped_backward_pair_problem("Q3_K", 35),
        _Solutions.q3_k_m128_n64_dual_lds(),
    )
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)

    state = _state(key)
    physical = derive_grouped_backward_pair_physical_plan(state)
    assert physical.ordinary.resources.vgprs == 127
    assert physical.ordinary.resources.lds_bytes == 10_240
    assert physical.second_projection is not None
    assert physical.second_projection.lds.base_offset == 5_120

    source = _writer(key, Toolchain.discover()).source()
    assert source.count("Decode the first pair projection into disjoint LDS") == 1
    assert source.count("Decode the second pair projection into disjoint LDS") == 1
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    assert source.count("s_barrier") == 2


def test_q3_k_backward_pair_dual_lds_full_tile_split() -> None:
    solution = _Solutions.q3_k_m128_n64_dual_lds_full_tile_split()
    key = _instance(grouped_backward_pair_problem("Q3_K", 257), solution)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)

    physical = derive_grouped_backward_pair_physical_plan(_state(key))
    assert physical.ordinary.resources.vgprs == 127
    assert physical.ordinary.resources.lds_bytes == 10_240
    assert physical.second_projection is not None
    assert physical.second_projection.lds.base_offset == 5_120

    source = _writer(key, Toolchain.discover()).source()
    assert "Decode the first pair projection into disjoint LDS" in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 64
    assert source.count("s_barrier") == 4


def test_q3_k_backward_pair_full_tile_direct_pointers() -> None:
    solution = _Solutions.q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers()
    key = _instance(grouped_backward_pair_problem("Q3_K", 257), solution)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)

    physical = derive_grouped_backward_pair_physical_plan(_state(key))
    assert physical.ordinary.resources.vgprs == 127
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )

    source = _writer(key, Toolchain.discover()).source()
    assert "Swap the active gradient and packed-bank pointer pairs" not in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 64
    assert source.count("s_barrier") == 4


def test_q3_k_backward_pair_serial_reads_prefetch_a() -> None:
    solution = (
        _Solutions.q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a()
    )
    key = _instance(grouped_backward_pair_problem("Q3_K", 257), solution)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)

    physical = derive_grouped_backward_pair_physical_plan(_state(key))
    assert physical.ordinary.resources.vgprs == 143
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )

    source = _writer(key, Toolchain.discover()).source()
    assert "Prefetch A fragments" in source
    assert "Issue both packed projection reads" not in source
    assert "Swap the active gradient and packed-bank pointer pairs" not in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 64
    assert source.count("s_barrier") == 4


def test_q3_k_backward_pair_m64_serial_reads_prefetch_a() -> None:
    solution = (
        _Solutions.q3_k_m64_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a()
    )
    key = _instance(grouped_backward_pair_problem("Q3_K", 257), solution)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)

    physical = derive_grouped_backward_pair_physical_plan(_state(key))
    assert physical.ordinary.resources.vgprs == 95
    assert physical.ordinary.resources.lds_bytes == 10_240
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )

    source = _writer(key, Toolchain.discover()).source()
    assert "Prefetch A fragments" in source
    assert "Issue both packed projection reads" not in source
    assert "Swap the active gradient and packed-bank pointer pairs" not in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    assert source.count("s_barrier") == 4


def test_q3_k_backward_pair_overlap_second_read_prefetch_a() -> None:
    solution = _Solutions.q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers_overlap_second_read_prefetch_a()
    key = _instance(grouped_backward_pair_problem("Q3_K", 257), solution)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)

    physical = derive_grouped_backward_pair_physical_plan(_state(key))
    assert physical.ordinary.resources.vgprs == 143
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )

    source = _writer(key, Toolchain.discover()).source()
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
    key = _instance(
        grouped_backward_pair_problem("IQ2_S", 35),
        _Solutions.iq2_s_m128_n64_dual_lds_full_tile_split_k_pipeline(),
    )
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)

    state = _state(key)
    physical = derive_grouped_backward_pair_physical_plan(state)
    assert physical.ordinary.resources.vgprs == 149
    assert physical.ordinary.resources.sgprs == 41
    assert physical.ordinary.resources.lds_bytes == 16_384
    assert physical.second_projection is not None
    assert (
        physical.second_projection.registers.kernarg
        == physical.scalar.second_grad_output
    )

    source = _writer(key, Toolchain.discover()).source()
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
    key = _instance(
        grouped_backward_pair_problem("IQ2_S", 257),
        _Solutions.iq2_s_m128_n64_sia5_global_codebook_packed_split_routes_8(),
    )
    _validate(key)

    state = _state(key)
    physical = derive_grouped_backward_pair_physical_plan(state)
    assert physical.ordinary.resources.lds_bytes == 8_192
    assert not physical.ordinary.lds.codebook_in_lds

    source = _writer(key, Toolchain.discover()).source()
    assert "Stage the IQ2_S codebook once in disjoint LDS" not in source
    assert "global_load_b64" in source
    assert source.count("s_barrier") == 6
    assert "s_and_b32 s17, s3, 7" in source
    assert "s_lshr_b32 s3, s3, 3" in source
    assert "s_mov_b32 s2, s17" in source


def test_backward_pair_global_codebook_source() -> None:
    key = _instance(
        grouped_backward_pair_problem("IQ2_S", 16_384),
        _Solutions.iq2_s_m128_n64_dual_lds_global_codebook(),
    )
    state = _state(key)
    physical = derive_grouped_backward_pair_physical_plan(state)
    assert physical.ordinary.resources.lds_bytes == 8_192
    assert not physical.ordinary.lds.codebook_in_lds

    source = _writer(key, Toolchain.discover()).source()
    assert source.count("global_load_b64") == 4
    assert source.count("s_barrier") == 2


def test_backward_pair_work_group_mapping_changes_grid_and_source() -> None:
    base = _key(rows=16_384, tile=64)
    base_spec = _problem_spec(base)[1]
    base_geometry = base_spec.compute.geometry
    mapped = replace(
        base,
        kernel_spec=replace(
            base_spec,
            compute=replace(
                base_spec.compute,
                geometry=replace(base_geometry, work_group_mapping=2),
            ),
        ),
    )
    _validate(mapped)
    assert launch_for_instance(base).grid == (32, 256, 1)
    assert launch_for_instance(mapped).grid == (64, 256, 1)
    base_source = _writer(base, Toolchain.discover()).source()
    mapped_source = _writer(mapped, Toolchain.discover()).source()
    assert mapped_source != base_source
    assert "Decode WGM-packed grid X into N and an M-task lane." in mapped_source
    assert "s_and_b32 s40, s2, 1" in mapped_source
    assert "s_lshr_b32 s2, s2, 1" in mapped_source
    assert "s_add_u32 s2, s2, 2" in mapped_source


def test_backward_pair_build_and_inspection(tmp_path) -> None:
    key = _key(rows=35)
    toolchain = Toolchain.discover()
    writer = _writer(key, toolchain)
    assert writer.source() == _writer(key, toolchain).source()
    assembly = tmp_path / "pair.s"
    object_path = tmp_path / "pair.o"
    code_object = tmp_path / "pair.hsaco"
    writer.write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    result = _inspect(key, code_object, toolchain)
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
    key = _instance(
        grouped_backward_pair_problem("IQ2_XXS", 35),
        _Solutions.iq2_xxs_m64_n64_sia5_dual_lds_full_tile_split_k_pipeline(),
    )
    toolchain = Toolchain.discover()
    build_hashes = []
    for directory_name in ("first", "second"):
        directory = tmp_path / directory_name
        directory.mkdir()
        assembly = directory / "kernel.s"
        object_path = directory / "kernel.o"
        code_object = directory / "kernel.hsaco"
        _writer(key, toolchain).write(assembly)
        toolchain.assemble(assembly, object_path)
        toolchain.link(object_path, code_object)
        build_hashes.append(
            tuple(
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (assembly, object_path, code_object)
            )
        )

    assert build_hashes[0] == build_hashes[1]
    result = _inspect(key, tmp_path / "second" / "kernel.hsaco", toolchain)
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
    writer = _writer(key, toolchain)
    assert writer.source() == _writer(key, toolchain).source()
    assembly = tmp_path / "iq2_xxs_pair.s"
    object_path = tmp_path / "iq2_xxs_pair.o"
    code_object = tmp_path / "iq2_xxs_pair.hsaco"
    writer.write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    result = _inspect(key, code_object, toolchain)
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
    writer = _writer(key, toolchain)
    assembly = tmp_path / "q3_k_pair.s"
    object_path = tmp_path / "q3_k_pair.o"
    code_object = tmp_path / "q3_k_pair.hsaco"
    writer.write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    result = _inspect(key, code_object, toolchain)
    assert result.kernarg_segment_size == 72
    assert result.vgpr_count == 87
    assert result.sgpr_count == 41
    assert result.lds_num_bytes == 5_120
    assert result.private_segment_bytes == 0
    assert result.vgpr_spill_count == 0
    assert result.sgpr_spill_count == 0
    assert result.wmma_count == 16
    assert result.barrier_count == 4
