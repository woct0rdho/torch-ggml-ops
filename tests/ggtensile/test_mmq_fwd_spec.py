from dataclasses import fields, replace
from typing import Any, cast

import pytest

from tools.ggtensile.mmq_fwd_spec import (
    DerivedForwardState,
    F16D4S4ActivationMetadata,
    ForwardFormatTraits,
    ForwardKernelCandidate,
    ForwardKernelSpec,
    ForwardProblemContract,
    ForwardResourceUsage,
    Packed3BitTiledLdsLayout,
    Q3FullWeightTiledLdsLayout,
    Q6LdsLayout,
    Q6SemanticPlan,
    Q6SemanticStage,
    QuantForwardSemantics,
    ResourceLimits,
    SemanticSchedulePolicy,
    SignedInt8CompactDepth32TiledLdsLayout,
    SignedInt8KvTiledLdsLayout,
    SignedInt8SmallMTiledLdsLayout,
    derive_forward_resource_usage,
    forward_kernel_spec_rejection_reason,
    forward_mechanism_contract,
    q6_schedule_from_kernel_spec,
    q6_schedule_from_solution,
)
from tools.ggtensile.model import (
    BackwardSolution,
    ForwardSolution,
    ProblemSize,
    ProblemType,
    SolutionKey,
)


def _key(
    quant_type: str,
    size: ProblemSize,
    solution: ForwardSolution,
) -> SolutionKey:
    return SolutionKey(ProblemType.mmq_forward(quant_type), size, solution)


@pytest.mark.parametrize(
    ("quant_type", "layout", "block_bytes"),
    (
        ("Q3_K", "F32_D4", 144),
        ("Q4_K", "F16_D4S4", 144),
        ("Q5_K", "F16_D4S4", 144),
        ("Q6_K", "F32_D4", 144),
        ("Q8_0", "F32_D4", 144),
    ),
)
def test_forward_format_traits_are_fixed_contracts(
    quant_type: str,
    layout: str,
    block_bytes: int,
) -> None:
    traits = ForwardFormatTraits.for_quant_type(quant_type)
    assert traits.activation_layout == layout
    assert traits.activation_block_bytes == block_bytes
    assert traits.block_values == (32 if quant_type == "Q8_0" else 256)


def test_f16_d4s4_activation_metadata_owns_group_offsets() -> None:
    metadata = F16D4S4ActivationMetadata(144)
    assert metadata.group(0).payload_offset == 16
    assert metadata.group(3).payload_high_offset == 128
    assert metadata.group(4).plane == 1
    assert metadata.group(4).payload_offset == 16
    assert metadata.group(7).scale_sum_offset == 12
    with pytest.raises(ValueError, match="group must be in range"):
        metadata.group(8)


def test_forward_mechanism_contracts_are_data_driven() -> None:
    decoded = forward_mechanism_contract("DecodedWeightLdsBatch8")
    direct = forward_mechanism_contract("Global")
    signed_int8 = forward_mechanism_contract("Q8DirectGlobal")
    assert decoded.lowering == "DecodedWeightLds"
    assert decoded.weight_decodes == ("DirectNibble", "DirectNibbleHighBit")
    assert direct.lowering == "PackedScaleMinimumDirect"
    assert direct.weight_decodes == ("DirectNibble",)
    assert signed_int8 == forward_mechanism_contract("Q8HipTiledLds")
    assert signed_int8.weight_block_values == 32
    assert signed_int8.reduction_values == 128
    with pytest.raises(ValueError, match="unsupported forward operand source"):
        forward_mechanism_contract("Unknown")


def test_quant_forward_semantics_describe_packed_planes() -> None:
    q3 = QuantForwardSemantics.for_quant_type("Q3_K")
    q4 = QuantForwardSemantics.for_quant_type("Q4_K")
    q5 = QuantForwardSemantics.for_quant_type("Q5_K")
    q6 = QuantForwardSemantics.for_quant_type("Q6_K")
    q8 = QuantForwardSemantics.for_quant_type("Q8_0")
    assert q3.payload_plane("hmask").byte_count == 32
    assert q3.payload_plane("qs").byte_offset == 32
    assert q3.payload_plane("scales").byte_offset == 96
    assert q3.payload_plane("d").byte_offset == 108
    assert q4.payload_plane("ql").byte_offset == 16
    assert q5.payload_plane("qh").byte_count == 32
    assert q6.payload_plane("scales").encoding == "SignedInt8"
    assert q6.payload_plane("d").byte_offset == 208
    assert q8.payload_plane("d").byte_count == 2
    assert q8.payload_plane("qs").byte_offset == 2
    assert q8.payload_plane("qs").encoding == "SignedInt8"
    assert q8.post_wmma_correction == "SignedScaleTimesActivationScale"
    with pytest.raises(ValueError, match="no payload plane"):
        q4.payload_plane("qh")


def test_q6_packed_payload_offsets_are_formula_derived() -> None:
    q6 = QuantForwardSemantics.for_quant_type("Q6_K")
    offsets = tuple(
        (
            q6.q6_packed_payload_offset(atom, "ql"),
            q6.q6_packed_payload_offset(atom, "qh"),
        )
        for atom in range(16)
    )
    assert len(set(offsets)) == 16
    assert len({offset for pair in offsets for offset in pair}) == 30
    assert all(high == (low + 128) % 4096 for low, high in offsets)
    assert offsets[11] == (192, 320)
    assert offsets[14] == (3968, 0)
    with pytest.raises(ValueError, match="unsupported Q6 payload atom"):
        q6.q6_packed_payload_offset(16, "ql")
    with pytest.raises(ValueError, match="unsupported Q6 payload plane"):
        q6.q6_packed_payload_offset(0, cast(Any, "bad"))
    with pytest.raises(ValueError, match="has no Q6 packed payload"):
        QuantForwardSemantics.for_quant_type("Q4_K").q6_packed_payload_offset(0, "ql")


def test_quant_forward_semantics_describe_metadata_reconstruction() -> None:
    q4 = QuantForwardSemantics.for_quant_type("Q4_K")
    q5 = QuantForwardSemantics.for_quant_type("Q5_K")
    q6 = QuantForwardSemantics.for_quant_type("Q6_K")
    lower = q4.packed_scale_minimum_fields(2)
    assert lower.scale == (lower.scale[0],)
    assert lower.scale[0].metadata_word == 1
    assert lower.scale[0].bit_offset == 16
    assert lower.minimum[0].metadata_word == 2
    upper = q5.packed_scale_minimum_fields(7)
    assert upper.scale[0].metadata_word == 3
    assert upper.scale[0].bit_offset == 24
    assert upper.scale[1].metadata_word == 1
    assert upper.scale[1].bit_offset == 30
    assert upper.scale[1].destination_shift == 4
    assert upper.minimum[0].bit_offset == 28
    assert upper.minimum[1].metadata_word == 2
    assert q4.high_bit_reconstruction() is None
    high_bits = q5.high_bit_reconstruction()
    assert high_bits is not None
    assert high_bits.plane_name == "qh"
    assert high_bits.low_mask == 0x01010101
    assert high_bits.high_mask == 0x02020202
    assert high_bits.high_destination_shift == 3
    with pytest.raises(ValueError, match="group must be in range"):
        q4.packed_scale_minimum_fields(8)
    with pytest.raises(ValueError, match="no packed scale/minimum"):
        q6.packed_scale_minimum_fields(0)


def test_forward_derived_state_owns_shape_formulas() -> None:
    key = _key(
        "Q6_K",
        ProblemSize(192, 248320, 2048),
        ForwardSolution.q6_k_structured_decoded(macro_tile0=64),
    )
    state = DerivedForwardState.from_solution_key(key)
    assert state.num_threads == 128
    assert state.waves_per_workgroup == 4
    assert state.mi_wave_group == (4, 1)
    assert state.mi_wave_tile == (1, 4)
    assert state.accumulator_count == 32
    assert state.k_phases_per_iteration == 2
    assert state.lds_bytes == 28_928
    assert state.blocks_per_weight_row == 8
    assert state.activation_blocks_per_row == 16
    assert state.packed_weight_row_bytes == 1680
    assert state.activation_plane_stride_bytes == 192 * 144
    assert state.activation_weight_block_stride_bytes == 2 * 192 * 144
    assert state.grid == (3880, 3, 1)
    assert state.expected_packed_weight_bytes == 248320 * 1680
    assert state.expected_activation_shape == (16, 192, 144)
    assert state.expected_output_shape == (192, 248320)


def test_signed_int8_reduction_domain_uses_one_d4_activation_block() -> None:
    key = _key(
        "Q8_0",
        ProblemSize(16, 16, 128),
        ForwardSolution.q8_0_direct_global(),
    )
    state = DerivedForwardState.from_solution_key(key)
    assert state.blocks_per_weight_row == 4
    assert state.activation_blocks_per_row == 1
    assert state.expected_activation_shape == (1, 16, 144)


def test_complete_candidate_round_trips_to_the_normal_build_solution() -> None:
    candidates = (
        ("Q4_K", ForwardSolution.q4_k_pilot()),
        ("Q3_K", ForwardSolution.q3_k_hip_tiled_lds()),
        ("Q3_K", ForwardSolution.q3_k_full_weight_tiled_lds()),
        ("Q4_K", ForwardSolution.q4_k_decoded_weight_lds_retained()),
        (
            "Q4_K",
            ForwardSolution.q4_k_decoded_weight_lds_extraction(
                epilogue_tiles_ahead=2,
                epilogue_dependency_width=4,
                epilogue_priority=3,
                metadata_after_low_wmma=True,
            ),
        ),
        (
            "Q5_K",
            ForwardSolution.q5_k_decoded_weight_lds_extraction(
                epilogue_tiles_ahead=3,
                epilogue_dependency_width=5,
                epilogue_priority=2,
                accumulator_initialization="VopdPair",
            ),
        ),
        ("Q6_K", ForwardSolution.q6_k_structured_decoded(macro_tile0=64)),
        ("Q6_K", ForwardSolution.q6_k_structured_decoded(macro_tile0=128)),
        (
            "Q6_K",
            replace(
                ForwardSolution.q6_k_structured_decoded(macro_tile0=64),
                q6_output_traversal="OutputRoleWavefront",
                q6_stage_clustering="RowBatchedDecodeOrder",
                q6_latency_policy="WavefrontDependencyDistance",
            ),
        ),
        ("Q8_0", ForwardSolution.q8_0_direct_global()),
        ("Q8_0", ForwardSolution.q8_0_register_tiled()),
        ("Q8_0", ForwardSolution.q8_0_hip_tiled_lds()),
        ("Q8_0", ForwardSolution.q8_0_hip_tiled_lds_depth64()),
        ("Q8_0", ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=32)),
        ("Q8_0", ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=64)),
    )
    for quant_type, solution in candidates:
        candidate = ForwardKernelCandidate.from_solution(quant_type, solution)
        assert candidate.to_solution() == solution

    candidate = ForwardKernelCandidate.from_solution(
        "Q6_K", ForwardSolution.q6_k_structured_decoded(macro_tile0=64)
    )
    assert candidate.kernel_spec.semantic_schedule == (
        SemanticSchedulePolicy.structured_q6()
    )
    assert candidate.kernel_spec.to_mapping()["semantic_schedule"] == {
        "traversal": "OutputRoleGroupMajor",
        "clustering": "StageDependencyOrder",
        "latency": "SerializedDependencyDistance",
        "pressure": "ExplicitRoleLifetime",
        "wait": "ProducerFirstUse",
        "pairing": "DependencyCompatibleDualIssue",
    }
    with pytest.raises(ValueError, match="resource limits are a fixed"):
        replace(
            candidate.kernel_spec,
            resource_limits=ResourceLimits(max_vgprs=200),
        ).to_solution(candidate.problem_contract)
    wavefront = ForwardKernelCandidate.from_solution(
        "Q6_K",
        replace(
            ForwardSolution.q6_k_structured_decoded(macro_tile0=64),
            q6_output_traversal="OutputRoleWavefront",
            q6_stage_clustering="RowBatchedDecodeOrder",
            q6_latency_policy="WavefrontDependencyDistance",
        ),
    )
    assert wavefront.kernel_spec.semantic_schedule == (
        SemanticSchedulePolicy.structured_q6_wavefront()
    )
    assert wavefront.to_solution() == replace(
        ForwardSolution.q6_k_structured_decoded(macro_tile0=64),
        q6_output_traversal="OutputRoleWavefront",
        q6_stage_clustering="RowBatchedDecodeOrder",
        q6_latency_policy="WavefrontDependencyDistance",
    )
    with pytest.raises(ValueError, match="semantic schedule does not match"):
        replace(
            candidate.kernel_spec,
            semantic_schedule=replace(
                candidate.kernel_spec.semantic_schedule,
                pairing="OpaqueIssueTable",
            ),
        ).to_solution(candidate.problem_contract)


@pytest.mark.parametrize(
    ("solution", "expected"),
    (
        (ForwardSolution.q4_k_pilot(), (88, 16, 0)),
        (ForwardSolution.q3_k_hip_tiled_lds(), (144, 16, 28_672)),
        (ForwardSolution.q3_k_full_weight_tiled_lds(), (200, 16, 39_936)),
        (
            ForwardSolution.q4_k_decoded_weight_lds_retained(),
            (239, 16, 38_400),
        ),
        (
            ForwardSolution.q6_k_structured_decoded(macro_tile0=64),
            (158, 27, 28_928),
        ),
        (
            ForwardSolution.q6_k_structured_decoded(macro_tile0=128),
            (210, 27, 38_400),
        ),
        (ForwardSolution.q8_0_direct_global(), (88, 16, 0)),
        (ForwardSolution.q8_0_register_tiled(), (137, 16, 0)),
        (ForwardSolution.q8_0_hip_tiled_lds(), (240, 16, 38_400)),
        (
            ForwardSolution.q8_0_compact_depth32_tiled_lds(),
            (240, 16, 27_648),
        ),
        (ForwardSolution.q8_0_hip_tiled_lds_depth64(), (240, 16, 38_400)),
        (
            ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=32),
            (96, 16, 24_064),
        ),
        (
            ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=64),
            (144, 16, 28_672),
        ),
        (ForwardSolution.q8_0_kv_tiled_lds(), (144, 16, 18_432)),
    ),
)
def test_forward_resources_are_derived_from_the_kernel_spec(
    solution: ForwardSolution,
    expected: tuple[int, int, int],
) -> None:
    usage = derive_forward_resource_usage(ForwardKernelSpec.from_solution(solution))
    assert (usage.vgprs, usage.sgprs, usage.lds_bytes) == expected
    usage.admit(ResourceLimits())


def test_q8_small_m_layout_derives_exact_lds_planes() -> None:
    m32 = SignedInt8SmallMTiledLdsLayout(32)
    m64 = SignedInt8SmallMTiledLdsLayout(64)
    assert (m32.activation_bytes, m32.weight_base, m32.weight_bytes) == (
        4_608,
        4_608,
        19_456,
    )
    assert m32.total_bytes == 24_064
    assert m64.activation_bytes == m64.weight_base == 9_216
    assert m64.weight_bytes == 19_456
    assert m64.total_bytes == 28_672
    with pytest.raises(ValueError, match="requires 32 or 64 rows"):
        SignedInt8SmallMTiledLdsLayout(128)


def test_q8_kv_layout_derives_compact_lds_planes() -> None:
    layout = SignedInt8KvTiledLdsLayout()
    assert layout.activation_bytes == layout.weight_base == 9_216
    assert layout.weight_row_stride == 144
    assert layout.weight_scale_offset == 128
    assert layout.weight_scale_element_stride == 288
    assert layout.weight_scale_pair_base_delta == 1_152
    assert layout.weight_bytes == 9_216
    assert layout.total_bytes == 18_432


def test_q8_compact_depth32_layout_derives_wave_n_lds_planes() -> None:
    layout = SignedInt8CompactDepth32TiledLdsLayout(128)
    assert layout.activation_bytes == layout.weight_base == 18_432
    assert layout.weight_bytes == 9_216
    assert layout.weight_scale_element_stride == 288
    assert layout.weight_scale_pair_base_delta == 1_152
    assert layout.total_bytes == 27_648


def test_q6_lds_layout_derives_selected_plane_offsets() -> None:
    narrow = Q6LdsLayout(1)
    wide = Q6LdsLayout(2)
    assert narrow.stage_stride_bytes == 9_472
    assert narrow.total_bytes == 28_928
    assert narrow.decoded_plane_base == 9_728
    assert narrow.scale_read_base == 9_732
    assert narrow.first_address_offset == 2_576
    assert narrow.direct_read_offset == 34
    assert narrow.first_stage_write_offset == 33
    assert narrow.stage_read_offsets == (0, 48, 96, 16)
    assert narrow.read_pair_plane_stride == 152
    assert tuple(
        (role.offset0, role.offset1, role.first_stage, role.last_stage)
        for role in narrow.stage_read_roles
    ) == (
        (0, 152, 4, 5),
        (48, 200, 4, 5),
        (96, 248, 4, 5),
        (16, 168, 4, 5),
    )
    narrow_write = narrow.cooperative_write_role(8)
    assert (narrow_write.offset0, narrow_write.offset1) == (33, 35)
    assert (
        narrow.cooperative_write_role(8).first_stage,
        narrow.cooperative_write_role(8).last_stage,
    ) == (6, 7)
    narrow_factor = narrow.factor_role(0)
    assert (narrow_factor.offset0, narrow_factor.offset1) == (1, 10)
    assert (
        narrow.factor_role(0).first_stage,
        narrow.factor_role(0).last_stage,
    ) == (5, 7)
    assert wide.stage_stride_bytes == 18_944
    assert wide.total_bytes == 38_400
    assert wide.decoded_plane_base == 19_200
    assert wide.scale_read_base == 19_204
    assert wide.first_address_offset == 2_832
    assert wide.direct_read_offset == 66
    assert wide.first_stage_write_offset == 70
    assert wide.stage_read_offsets == (64, 48, 32, 80)
    assert wide.read_pair_plane_stride == 152
    wide_write = wide.cooperative_write_role(17)
    assert (wide_write.offset0, wide_write.offset1) == (70, 72)
    wide_factor = wide.factor_role(3)
    assert (wide_factor.offset0, wide_factor.offset1) == (56, 65)
    with pytest.raises(ValueError, match="rows per wave"):
        Q6LdsLayout(0)
    with pytest.raises(ValueError, match="cooperative-write pair"):
        narrow.cooperative_write_role(-1)
    with pytest.raises(ValueError, match="factor pair"):
        narrow.factor_role(-1)


def test_q3_half_tile_lds_and_packed_groups_are_formula_derived() -> None:
    layout = Packed3BitTiledLdsLayout()
    assert layout.activation_bytes == 18_432
    assert layout.weight_payload_bytes == 128
    assert layout.weight_scale_bytes == 32
    assert layout.weight_row_stride == 160
    assert layout.weight_base == 18_432
    assert layout.weight_bytes == 10_240
    assert layout.total_bytes == 28_672

    full = Q3FullWeightTiledLdsLayout()
    assert full.activation_bytes == 18_432
    assert full.weight_base == 18_432
    assert full.weight_payload_bytes == 256
    assert full.weight_scale_total_bytes == 64
    assert full.weight_padding_bytes == 16
    assert full.weight_bytes == 21_504
    assert full.total_bytes == 39_936

    semantics = QuantForwardSemantics.for_quant_type("Q3_K")
    first = semantics.q3_payload_group(0)
    assert (
        first.half,
        first.low_payload_offset,
        first.low_shift,
        first.high_payload_offset,
        first.high_shift,
        first.activation_payload_offset,
        first.activation_scale_offset,
    ) == (0, 32, 0, 0, 0, 16, 0)
    last = semantics.q3_payload_group(15)
    assert (
        last.half,
        last.low_payload_offset,
        last.low_shift,
        last.high_payload_offset,
        last.high_shift,
        last.activation_payload_offset,
        last.activation_scale_offset,
    ) == (1, 80, 6, 16, 7, 128, 12)
    assert semantics.q3_scale_fields(0)[0].source_byte == 0
    assert semantics.q3_scale_fields(15)[0].bit_offset == 4
    assert semantics.q3_scale_fields(15)[1].source_byte == 11
    assert semantics.q3_scale_fields(15)[1].bit_offset == 6
    signed = semantics.q3_signed_decode()
    assert signed.signed_add == 0x7C7C7C7C
    assert signed.signed_xor == 0x80808080
    with pytest.raises(ValueError, match="payload group"):
        semantics.q3_payload_group(16)
    with pytest.raises(ValueError, match="scale group"):
        semantics.q3_scale_fields(-1)
    with pytest.raises(ValueError, match="has no signed Q3"):
        QuantForwardSemantics.for_quant_type("Q4_K").q3_signed_decode()
    with pytest.raises(ValueError, match="dimensions must be positive"):
        Packed3BitTiledLdsLayout(activation_rows=0)
    with pytest.raises(ValueError, match="fixed dimensions"):
        Q3FullWeightTiledLdsLayout(weight_row_stride=320)


def test_forward_resource_admission_rejects_each_fixed_limit() -> None:
    with pytest.raises(ValueError, match="VGPR"):
        ForwardResourceUsage(257, 1, 0).admit(ResourceLimits())
    with pytest.raises(ValueError, match="SGPR"):
        ForwardResourceUsage(1, 107, 0).admit(ResourceLimits())
    with pytest.raises(ValueError, match="LDS"):
        ForwardResourceUsage(1, 1, 65_537).admit(ResourceLimits())
    with pytest.raises(ValueError, match="private storage or spills"):
        ForwardResourceUsage(1, 1, 0, vgpr_spills=1).admit(ResourceLimits())


def test_every_forward_solution_field_is_projected_or_rejected() -> None:
    representatives = (
        ("Q3_K", ForwardSolution.q3_k_hip_tiled_lds()),
        ("Q3_K", ForwardSolution.q3_k_full_weight_tiled_lds()),
        ("Q4_K", ForwardSolution.q4_k_pilot()),
        ("Q4_K", ForwardSolution.q4_k_decoded_weight_lds_retained()),
        ("Q5_K", ForwardSolution.q5_k_decoded_weight_lds_retained()),
        ("Q6_K", ForwardSolution.q6_k_structured_decoded(macro_tile0=64)),
        ("Q8_0", ForwardSolution.q8_0_direct_global()),
        ("Q8_0", ForwardSolution.q8_0_register_tiled()),
        ("Q8_0", ForwardSolution.q8_0_hip_tiled_lds()),
        ("Q8_0", ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=32)),
        ("Q8_0", ForwardSolution.q8_0_kv_tiled_lds()),
    )

    def different(value: object) -> object:
        if isinstance(value, bool):
            return not value
        if isinstance(value, int):
            return value + 1
        if isinstance(value, str):
            return f"{value}Invalid"
        if isinstance(value, tuple):
            first = value[0]
            assert isinstance(first, int)
            return (first + 1, *value[1:])
        raise AssertionError(f"unsupported ForwardSolution field value: {value!r}")

    for quant_type, solution in representatives:
        baseline = (
            ForwardProblemContract.from_solution(quant_type, solution),
            ForwardKernelSpec.from_solution(solution),
        )
        for field in fields(ForwardSolution):
            candidate = replace(
                solution,
                **{field.name: different(getattr(solution, field.name))},
            )
            if (
                ForwardProblemContract.rejection_reason(quant_type, candidate)
                is not None
                or forward_kernel_spec_rejection_reason(candidate) is not None
            ):
                continue
            projection = (
                ForwardProblemContract.from_solution(quant_type, candidate),
                ForwardKernelSpec.from_solution(candidate),
            )
            assert projection != baseline, (
                f"accepted {quant_type} ForwardSolution field is ignored: {field.name}"
            )


def test_forward_kernel_spec_rejects_inactive_legacy_fields() -> None:
    q6 = ForwardSolution.q6_k_structured_decoded(macro_tile0=64)
    with pytest.raises(ValueError, match="canonical sentinel"):
        ForwardKernelSpec.from_solution(
            replace(q6, matrix_instruction=(*q6.matrix_instruction[:8], 2))
        )
    with pytest.raises(ValueError, match="metadata schedule is inactive"):
        ForwardKernelSpec.from_solution(
            replace(q6, metadata_schedule="MetadataAfterLowWmma")
        )

    decoded = ForwardSolution.q4_k_decoded_weight_lds_retained()
    with pytest.raises(ValueError, match="Q6 delay mode is inactive"):
        ForwardKernelSpec.from_solution(
            replace(decoded, q6_dependency_delay_mode="Explicit")
        )

    direct = ForwardSolution.q4_k_pilot()
    with pytest.raises(ValueError, match="epilogue pipeline is inactive"):
        ForwardKernelSpec.from_solution(replace(direct, epilogue_priority=1))


def test_q6_semantic_plan_has_explicit_ordered_dependencies() -> None:
    schedule = q6_schedule_from_solution(
        ForwardSolution.q6_k_structured_decoded(macro_tile0=128)
    )
    plan = Q6SemanticPlan.from_schedule(schedule)
    assert tuple(stage.kind for stage in plan.stages) == (
        "Setup",
        "GlobalRead",
        "Decode",
        "LocalWrite",
        "BarrierLocalRead",
        "Dot",
        "Refill",
        "Dot",
        "Epilogue",
    )
    assert tuple(stage.phase for stage in plan.stages if stage.kind == "Dot") == (0, 1)
    with pytest.raises(ValueError, match="must precede"):
        Q6SemanticPlan((Q6SemanticStage("Setup", (0,)),)).validate()
    with pytest.raises(ValueError, match="requires a nonnegative phase"):
        Q6SemanticStage("Dot", ())
    with pytest.raises(ValueError, match="only Q6 dot stages"):
        Q6SemanticStage("Setup", (), phase=0)
    with pytest.raises(ValueError, match="exactly two dot phases"):
        Q6SemanticPlan.from_schedule(replace(schedule, dot_register_shifts=(0,)))


def test_q6_schedule_is_derived_from_the_canonical_kernel_spec() -> None:
    solution = ForwardSolution.q6_k_structured_decoded(macro_tile0=128)
    spec = ForwardKernelSpec.from_solution(solution)
    assert q6_schedule_from_kernel_spec(spec) == q6_schedule_from_solution(solution)
    schedule = q6_schedule_from_kernel_spec(spec)
    assert schedule.macro_tile == spec.macro_tile == (128, 64)
    assert schedule.depth_u == spec.geometry.depth_u == 32
    with pytest.raises(ValueError, match="structured decoded operands"):
        q6_schedule_from_kernel_spec(
            ForwardKernelSpec.from_solution(
                ForwardSolution.q4_k_decoded_weight_lds_retained()
            )
        )


def test_forward_derived_state_rejects_contract_and_geometry_mismatches() -> None:
    base = ForwardSolution.q4_k_pilot()
    key = _key("Q4_K", ProblemSize(128, 512, 2048), base)
    with pytest.raises(ValueError, match="activation layout"):
        DerivedForwardState.from_solution_key(
            replace(key, solution=replace(base, activation_layout="F32_D4"))
        )
    with pytest.raises(ValueError, match="packed-weight bytes"):
        DerivedForwardState.from_solution_key(
            replace(key, solution=replace(base, packed_weight_block_bytes=1))
        )
    with pytest.raises(ValueError, match="positive multiple"):
        DerivedForwardState.from_solution_key(
            replace(key, problem_size=ProblemSize(127, 512, 2048))
        )
    backward = SolutionKey(
        ProblemType.mmq_backward("Q4_K"),
        ProblemSize(128, 512, 2048),
        BackwardSolution.pilot(),
    )
    with pytest.raises(TypeError, match="requires ForwardSolution"):
        DerivedForwardState.from_solution_key(backward)
