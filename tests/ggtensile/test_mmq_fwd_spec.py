from dataclasses import replace

import pytest

from tests.ggtensile.ordinary_forward_fixtures import (
    q3_full_weight_tiled_lds_kernel_spec,
    q3_hip_tiled_lds_kernel_spec,
    q4_decoded_weight_lds_kernel_spec,
    q5_decoded_weight_lds_kernel_spec,
    q6_structured_kernel_spec,
    q8_compact_depth32_tiled_lds_kernel_spec,
    q8_direct_global_kernel_spec,
    q8_small_m_tiled_lds_kernel_spec,
)
from tools.ggtensile.mmq_fwd_spec import (
    DerivedForwardState,
    ForwardKernelCandidate,
    ForwardKernelSpec,
    ForwardProblemContract,
    ForwardWeightStaging,
    Packed3BitTiledLdsLayout,
    Q3FullWeightTiledLdsLayout,
    Q6LdsLayout,
    QuantForwardSemantics,
    SemanticSchedulePolicy,
    SignedInt8CompactDepth32TiledLdsLayout,
    SignedInt8SmallMTiledLdsLayout,
    StructuredQ6ScheduleVariant,
    derive_forward_resource_usage,
    forward_mechanism_contract,
    q6_schedule_from_kernel_spec,
)
from tools.ggtensile.model import ProblemSize, SchemaError
from tools.ggtensile.physical_resources import (
    HardwareResourceCapacity,
    PhysicalResourceUsage,
)


def test_quant_semantics_are_formula_driven() -> None:
    q3 = QuantForwardSemantics.for_quant_type("Q3_K")
    q5 = QuantForwardSemantics.for_quant_type("Q5_K")
    q6 = QuantForwardSemantics.for_quant_type("Q6_K")
    assert q3.payload_plane("qs").byte_offset == 32
    assert q3.payload_plane("d").byte_offset == 108
    high_bit_reconstruction = q5.high_bit_reconstruction()
    assert high_bit_reconstruction is not None
    assert high_bit_reconstruction.high_destination_shift == 3
    assert q6.q6_packed_payload_offset(11, "ql") == 192
    assert q6.q6_packed_payload_offset(11, "qh") == 320
    with pytest.raises(AssertionError):
        q3.payload_plane("missing")


def test_mechanism_contracts_own_dataflow_and_reduction_domains() -> None:
    decoded = forward_mechanism_contract(ForwardWeightStaging.DecodedWeightLdsBatch8)
    signed = forward_mechanism_contract(ForwardWeightStaging.Q8DirectGlobal)
    assert decoded.lowering == "DecodedWeightLds"
    assert decoded.reduction_values == 256
    assert signed.physical_plan == "SignedInt8Direct"
    assert signed.weight_block_values == 32
    assert signed.reduction_values == 128


@pytest.mark.parametrize(
    ("quant_type", "spec", "size"),
    (
        ("Q3_K", q3_hip_tiled_lds_kernel_spec(), ProblemSize(2048, 2048, 2048)),
        (
            "Q3_K",
            q3_full_weight_tiled_lds_kernel_spec(),
            ProblemSize(32768, 4096, 2048),
        ),
        (
            "Q4_K",
            q4_decoded_weight_lds_kernel_spec(
                epilogue_tiles_ahead=1, epilogue_dependency_width=1, epilogue_priority=0
            ),
            ProblemSize(8192, 2048, 512),
        ),
        (
            "Q5_K",
            q5_decoded_weight_lds_kernel_spec(
                epilogue_tiles_ahead=1, epilogue_dependency_width=2, epilogue_priority=2
            ),
            ProblemSize(8192, 2048, 512),
        ),
        ("Q6_K", q6_structured_kernel_spec(64), ProblemSize(128, 248320, 2048)),
        ("Q8_0", q8_direct_global_kernel_spec(), ProblemSize(2048, 1024, 4096)),
    ),
)
def test_forward_state_and_resources_are_derived(
    quant_type: str, spec: ForwardKernelSpec, size: ProblemSize
) -> None:
    state = DerivedForwardState.from_problem_spec(size, quant_type, spec)
    assert state.expected_output_shape == (size.m, size.n)
    assert state.expected_activation_shape[1] == size.m
    usage = derive_forward_resource_usage(spec)
    usage.admit(HardwareResourceCapacity())
    assert usage.private_bytes == 0


def test_forward_spec_mapping_is_strict_and_typed() -> None:
    spec = q4_decoded_weight_lds_kernel_spec(
        epilogue_tiles_ahead=2, epilogue_dependency_width=4, epilogue_priority=2
    )
    assert ForwardKernelSpec.from_mapping(spec.to_mapping()) == spec
    mapping = spec.to_mapping()
    mapping["Unexpected"] = 1
    with pytest.raises(SchemaError):
        ForwardKernelSpec.from_mapping(mapping)


def test_forward_schema_requires_applicable_policy_fields() -> None:
    spec = q4_decoded_weight_lds_kernel_spec(
        epilogue_tiles_ahead=2, epilogue_dependency_width=1, epilogue_priority=0
    )
    mapping = spec.to_mapping()
    assert isinstance(mapping["Lds"], dict)
    mapping["Lds"].pop("LdsPadA")
    with pytest.raises(SchemaError):
        ForwardKernelSpec.from_mapping(mapping)

    mapping = spec.to_mapping()
    mapping.pop("Staging")
    with pytest.raises(SchemaError):
        ForwardKernelSpec.from_mapping(mapping)

    mapping = spec.to_mapping()
    assert isinstance(mapping["DataMovement"], dict)
    mapping["DataMovement"].pop("DecodeProducerCount")
    with pytest.raises(SchemaError):
        ForwardKernelSpec.from_mapping(mapping)


def test_forward_schema_keeps_semantically_inactive_fields_optional() -> None:
    signed = q8_small_m_tiled_lds_kernel_spec(32)
    parsed = ForwardKernelSpec.from_mapping(signed.to_mapping())
    assert parsed == signed
    assert parsed.data_movement is not None
    assert parsed.data_movement.decode_producer_count is None

    direct = q8_direct_global_kernel_spec().to_mapping()
    assert isinstance(direct["Lds"], dict)
    direct["Lds"]["LdsLayout"] = "Canonical"
    with pytest.raises(SchemaError):
        ForwardKernelSpec.from_mapping(direct)


def test_forward_candidate_round_trip_contains_only_kernel_spec_data() -> None:
    spec = q6_structured_kernel_spec(64)
    candidate = ForwardKernelCandidate(
        ForwardProblemContract.for_quant_type("Q6_K"), spec
    )
    parsed = ForwardKernelCandidate.from_mapping(candidate.to_mapping())
    assert parsed == candidate
    kernel_spec_mapping = candidate.to_mapping()["KernelSpec"]
    assert isinstance(kernel_spec_mapping, dict)
    assert "HardwareResourceCapacity" not in kernel_spec_mapping
    assert spec.semantic_schedule.variant is StructuredQ6ScheduleVariant.Wavefront
    assert (
        q6_schedule_from_kernel_spec(spec).semantic_policy
        == SemanticSchedulePolicy.structured_q6_wavefront()
    )


def test_q6_schedule_variants_are_explicit() -> None:
    wavefront = replace(
        q6_structured_kernel_spec(64),
        semantic_schedule=SemanticSchedulePolicy.structured_q6_wavefront(),
        instruction_policy=replace(
            q6_structured_kernel_spec(64).instruction_policy,
            physical_plan="WideScalarCarryFrontier",
        ),
    )
    assert wavefront.semantic_schedule.variant is StructuredQ6ScheduleVariant.Wavefront
    assert (
        q6_schedule_from_kernel_spec(wavefront).physical_plan
        == "WideScalarCarryFrontier"
    )


def test_lds_layouts_derive_stable_sizes() -> None:
    assert SignedInt8SmallMTiledLdsLayout(32).total_bytes == 24_064
    assert SignedInt8CompactDepth32TiledLdsLayout(64).total_bytes == 18_432
    assert Q6LdsLayout(1).total_bytes == 28_928
    assert Packed3BitTiledLdsLayout().total_bytes == 28_672
    assert Q3FullWeightTiledLdsLayout().total_bytes == 39_936
    with pytest.raises(AssertionError):
        Q3FullWeightTiledLdsLayout(16, 16)
    with pytest.raises(AssertionError):
        Q3FullWeightTiledLdsLayout(4, 0)
    with pytest.raises(AssertionError):
        Q3FullWeightTiledLdsLayout(0, 4)
    with pytest.raises(AssertionError):
        SignedInt8SmallMTiledLdsLayout(128)


def test_resource_admission_covers_fixed_limits() -> None:
    capacity = HardwareResourceCapacity()
    with pytest.raises(AssertionError):
        PhysicalResourceUsage(257, 1, 0).admit(capacity)
    with pytest.raises(AssertionError):
        PhysicalResourceUsage(1, 107, 0).admit(capacity)
    with pytest.raises(AssertionError):
        PhysicalResourceUsage(1, 1, 65_537).admit(capacity)
    with pytest.raises(AssertionError):
        PhysicalResourceUsage(1, 1, 0, vgpr_spills=1).admit(capacity)


def test_compact_q8_factory_preserves_depth_contract() -> None:
    spec = q8_compact_depth32_tiled_lds_kernel_spec(64)
    assert spec.geometry.depth_u == 32
    assert spec.macro_tile == (64, 64)
    assert spec.lds.address_hoist == "CompactDepth32WeightRows"
