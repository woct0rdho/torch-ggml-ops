from dataclasses import dataclass, replace

from .mmq_bwd_physical import derive_backward_physical_plan
from .mmq_bwd_spec import (
    BackwardExtraction,
    BackwardLdsBuffering,
    BackwardPackedWeightPrefetch,
    BackwardQ3Pairing,
    BackwardQ4DecodeSchedule,
    BackwardQ5MetadataLoad,
    BackwardQ5NibbleShift,
    BackwardScheduleIterAlg,
    DerivedBackwardState,
    backward_mechanism_contract,
)
from .mmq_fwd_spec import (
    ForwardMechanismContract,
    ForwardProblemContract,
    forward_kernel_spec_rejection_reason,
    forward_mechanism_contract,
)
from .model import (
    BackwardSolution,
    ForwardSolution,
    GroupedBackwardSolution,
    ProblemSize,
    ProblemType,
    SolutionKey,
)
from .quant_formats import BACKWARD_QUANT_FORMATS, QUANT_FORMATS


@dataclass(frozen=True)
class RejectReason:
    rule_id: str
    message: str
    parameters: tuple[str, ...]
    source: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "RuleId": self.rule_id,
            "Message": self.message,
            "Parameters": list(self.parameters),
            "Source": self.source,
        }


def _reject(
    reasons: list[RejectReason],
    rule_id: str,
    message: str,
    *parameters: str,
    source: str = "KernelWriter",
) -> None:
    reasons.append(RejectReason(rule_id, message, tuple(parameters), source))


def _validate_backward_problem_type(
    problem_type: ProblemType, reasons: list[RejectReason]
) -> None:
    if problem_type.quant_data_type not in BACKWARD_QUANT_FORMATS:
        _reject(
            reasons,
            "problem_type.quant_data_type.unsupported",
            "MMQ backward supports only Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and Q8_0",
            "QuantDataType",
            source="ProblemType",
        )
        return
    expected = ProblemType.mmq_backward(problem_type.quant_data_type)
    for attribute, parameter in (
        ("operation_type", "OperationType"),
        ("quant_data_type", "QuantDataType"),
        ("data_type_a", "DataTypeA"),
        ("data_type_b", "DataTypeB"),
        ("dest_data_type", "DestDataType"),
        ("compute_data_type", "ComputeDataType"),
        ("transpose_a", "TransposeA"),
        ("transpose_b", "TransposeB"),
    ):
        if getattr(problem_type, attribute) != getattr(expected, attribute):
            _reject(
                reasons,
                f"problem_type.{parameter.lower()}.unsupported",
                f"MMQ backward requires {parameter}={getattr(expected, attribute)!r}",
                parameter,
                source="ProblemType",
            )


def _validate_forward_tile_multiples(
    problem_size: ProblemSize,
    solution: ForwardSolution,
    reasons: list[RejectReason],
) -> None:
    mechanism = forward_mechanism_contract(solution.operand_source)
    for parameter, value, divisor in (
        ("M", problem_size.m, solution.macro_tile0),
        ("N", problem_size.n, solution.macro_tile1),
        ("K", problem_size.k, mechanism.reduction_values),
    ):
        if value <= 0 or divisor <= 0 or value % divisor:
            _reject(
                reasons,
                f"problem_size.{parameter.lower()}.forward_tile_multiple",
                f"{parameter} must be a positive multiple of {divisor}",
                parameter,
                source="ProblemSize",
            )


def _is_forward_extraction(
    solution: ForwardSolution,
    base: ForwardSolution,
    metadata_schedules: tuple[str, ...],
) -> bool:
    if (
        solution.metadata_schedule not in metadata_schedules
        or solution.epilogue_tiles_ahead not in range(1, 9)
        or solution.epilogue_dependency_width not in range(1, 9)
        or solution.epilogue_priority not in range(4)
        or solution.accumulator_initialization not in ("ScalarCopy", "VopdPair")
    ):
        return False
    return replace(
        solution,
        metadata_schedule="Serialized",
        epilogue_tiles_ahead=8,
        epilogue_dependency_width=1,
        epilogue_priority=0,
        accumulator_initialization="ScalarCopy",
    ) == replace(base, metadata_schedule="Serialized")


def _validate_structured_q6_forward_solution(
    problem_size: ProblemSize,
    solution: ForwardSolution,
    reasons: list[RejectReason],
) -> None:
    if solution.macro_tile0 not in (64, 128, 256):
        _reject(
            reasons,
            "solution.forward.q6.control.unimplemented",
            "Q6_K forward implements structured MT64, MT128, or MT256 controls",
            "Solution",
        )
        return
    structured = ForwardSolution.q6_k_structured_decoded(
        macro_tile0=solution.macro_tile0
    )
    structured_candidate = (
        replace(
            solution,
            epilogue_dependency_width=structured.epilogue_dependency_width,
            q6_epilogue_pipeline_scope=structured.q6_epilogue_pipeline_scope,
            q6_dependency_delay_mode=structured.q6_dependency_delay_mode,
            q6_global_read_cache_policy=structured.q6_global_read_cache_policy,
            q6_output_traversal=structured.q6_output_traversal,
            q6_stage_clustering=structured.q6_stage_clustering,
            q6_latency_policy=structured.q6_latency_policy,
            q6_pressure_policy=structured.q6_pressure_policy,
            q6_wait_policy=structured.q6_wait_policy,
            q6_pairing_policy=structured.q6_pairing_policy,
            q6_physical_plan=structured.q6_physical_plan,
        )
        == structured
    )
    if not structured_candidate:
        _reject(
            reasons,
            "solution.forward.q6.control.unimplemented",
            "Q6_K forward implements structured MT64, MT128, or MT256 controls",
            "Solution",
        )
        return

    _validate_forward_tile_multiples(problem_size, solution, reasons)
    if (
        solution.epilogue_dependency_width not in (1, 2, 4, 8)
        or solution.q6_epilogue_pipeline_scope not in ("StoreBatch", "FullTile")
        or solution.q6_dependency_delay_mode not in ("None", "Explicit")
        or solution.q6_global_read_cache_policy not in ("Default", "InvalidateL0")
        or (solution.macro_tile0 == 64 and solution.q6_dependency_delay_mode != "None")
        or solution.q6_physical_plan
        not in {"CanonicalRegisterRoles", "WideScalarCarryFrontier"}
        or (
            solution.q6_physical_plan == "WideScalarCarryFrontier"
            and (
                solution.macro_tile0 != 64
                or solution.q6_output_traversal != "OutputRoleWavefront"
            )
        )
    ):
        _reject(
            reasons,
            "solution.forward.q6.schedule.unimplemented",
            "structured Q6 requires an implemented epilogue, delay, and cache policy",
            "EpilogueDependencyWidth",
            "Q6EpiloguePipelineScope",
            "Q6DependencyDelayMode",
            "Q6GlobalReadCachePolicy",
            "Q6PhysicalPlan",
        )


def _validate_packed_3bit_forward_solution(
    problem_size: ProblemSize,
    solution: ForwardSolution,
    reasons: list[RejectReason],
) -> None:
    if solution != ForwardSolution.q3_k_hip_tiled_lds():
        _reject(
            reasons,
            "solution.forward.q3.control.unimplemented",
            "Q3_K forward currently implements one HIP-shaped LDS research control",
            "Solution",
        )
        return
    _validate_forward_tile_multiples(problem_size, solution, reasons)


def _validate_q3_full_weight_forward_solution(
    problem_size: ProblemSize,
    solution: ForwardSolution,
    reasons: list[RejectReason],
) -> None:
    if solution != ForwardSolution.q3_k_full_weight_tiled_lds():
        _reject(
            reasons,
            "solution.forward.q3.full_weight.control.unimplemented",
            "Q3_K full-weight forward implements one typed ownership schedule",
            "Solution",
        )
        return
    _validate_forward_tile_multiples(problem_size, solution, reasons)


def _validate_signed_int8_forward_solution(
    problem_size: ProblemSize,
    solution: ForwardSolution,
    reasons: list[RejectReason],
) -> None:
    if solution not in {
        ForwardSolution.q8_0_direct_global(),
        ForwardSolution.q8_0_register_tiled(wave_tile_m=1, wave_tile_n=4),
        ForwardSolution.q8_0_register_tiled(wave_tile_m=2, wave_tile_n=2),
        ForwardSolution.q8_0_register_tiled(wave_tile_m=4, wave_tile_n=1),
        ForwardSolution.q8_0_hip_tiled_lds(),
        ForwardSolution.q8_0_compact_depth32_tiled_lds(macro_tile0=32),
        ForwardSolution.q8_0_compact_depth32_tiled_lds(macro_tile0=64),
        ForwardSolution.q8_0_compact_depth32_tiled_lds(macro_tile0=128),
        ForwardSolution.q8_0_hip_tiled_lds_depth64(),
        ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=32),
        ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=64),
        ForwardSolution.q8_0_kv_tiled_lds(),
    }:
        _reject(
            reasons,
            "solution.forward.q8.control.unimplemented",
            "Q8_0 forward currently implements direct, register-tiled, and typed LDS controls",
            "Solution",
        )
        return
    _validate_forward_tile_multiples(problem_size, solution, reasons)


def _with_forward_contract(
    solution: ForwardSolution,
    contract: ForwardProblemContract,
) -> ForwardSolution:
    """Project one mechanism control onto an already validated data contract."""
    return replace(
        solution,
        kernel_language=contract.kernel_language,
        isa=contract.isa,
        wavefront_size=contract.wavefront_size,
        activation_layout=contract.activation_layout,
        activation_block_bytes=contract.activation_block_bytes,
        packed_weight_block_bytes=contract.packed_weight_block_bytes,
        weight_decode=contract.weight_decode,
        scale_arithmetic=contract.scale_arithmetic,
        signed_weight=contract.signed_weight,
        signed_activation=contract.signed_activation,
        wmma_clamp=contract.wmma_clamp,
    )


def _validate_decoded_weight_lds_forward_solution(
    problem_size: ProblemSize,
    solution: ForwardSolution,
    contract: ForwardProblemContract,
    reasons: list[RejectReason],
) -> None:
    retained = _with_forward_contract(
        ForwardSolution.q4_k_decoded_weight_lds_retained(),
        contract,
    )
    fixed = (
        retained,
        replace(retained, metadata_schedule="MetadataAfterLowWmma"),
    )
    has_high_bit_plane = contract.weight_decode == "DirectNibbleHighBit"
    metadata_schedules = (
        ("IndependentExtractionMetadataAfterLowWmma",)
        if has_high_bit_plane
        else ("IndependentExtraction", "IndependentExtractionMetadataAfterLowWmma")
    )
    extraction = _is_forward_extraction(
        solution,
        retained,
        metadata_schedules,
    )
    if solution not in fixed and not extraction:
        message = (
            "Q5_K forward implements retained decoded staging and explicit "
            "extraction/epilogue policies"
            if has_high_bit_plane
            else "Q4_K forward implements direct global or decoded-staged "
            "extraction/epilogue policies"
        )
        _reject(
            reasons,
            "solution.forward.control.unimplemented",
            message,
            "Solution",
        )
        return
    _validate_forward_tile_multiples(problem_size, solution, reasons)


def _validate_packed_scale_minimum_direct_forward_solution(
    problem_size: ProblemSize,
    solution: ForwardSolution,
    contract: ForwardProblemContract,
    reasons: list[RejectReason],
) -> None:
    expected = _with_forward_contract(ForwardSolution.q4_k_pilot(), contract)
    if solution != expected:
        _reject(
            reasons,
            "solution.forward.control.unimplemented",
            "Q4_K forward implements direct global or decoded-staged extraction/epilogue policies",
            "Solution",
        )
        return
    _validate_forward_tile_multiples(problem_size, solution, reasons)


def _validate_forward_mechanism_control(
    mechanism: ForwardMechanismContract,
    contract: ForwardProblemContract,
    problem_size: ProblemSize,
    solution: ForwardSolution,
    reasons: list[RejectReason],
) -> None:
    lowering = mechanism.lowering
    if lowering == "Packed3BitTiledLds":
        _validate_packed_3bit_forward_solution(problem_size, solution, reasons)
    elif lowering == "Packed3BitFullWeightTiledLds":
        _validate_q3_full_weight_forward_solution(problem_size, solution, reasons)
    elif lowering == "StructuredQ6":
        _validate_structured_q6_forward_solution(problem_size, solution, reasons)
    elif lowering == "DecodedWeightLds":
        _validate_decoded_weight_lds_forward_solution(
            problem_size,
            solution,
            contract,
            reasons,
        )
    elif lowering == "PackedScaleMinimumDirect":
        _validate_packed_scale_minimum_direct_forward_solution(
            problem_size,
            solution,
            contract,
            reasons,
        )
    elif lowering == "SignedInt8":
        _validate_signed_int8_forward_solution(
            problem_size,
            solution,
            reasons,
        )
    else:
        raise AssertionError(f"unknown forward lowering {lowering!r}")


def _validate_forward_solution(
    solution_key: SolutionKey, reasons: list[RejectReason]
) -> None:
    problem_type = solution_key.problem_type
    solution = solution_key.solution
    supported_problem_types = {
        ProblemType.mmq_forward(quant_type) for quant_type in QUANT_FORMATS
    }
    if problem_type not in supported_problem_types:
        _reject(
            reasons,
            "problem_type.forward.unsupported",
            "MMQ forward requires an exact Q3_K, Q4_K, Q5_K, Q6_K, or Q8_0 Q8_1 problem type",
            "ProblemType",
            source="ProblemType",
        )
    if not isinstance(solution, ForwardSolution):
        _reject(
            reasons,
            "solution.forward.schema",
            "MMQ forward requires ForwardSolution",
            "Solution",
            source="SolutionStructs",
        )
        return
    if problem_type not in supported_problem_types:
        return

    contract_rejection = ForwardProblemContract.rejection_reason(
        problem_type.quant_data_type,
        solution,
    )
    if contract_rejection is not None:
        _reject(
            reasons,
            "solution.forward.problem_contract",
            contract_rejection,
            "Solution",
            source="ForwardProblemContract",
        )
        return
    contract = ForwardProblemContract.from_solution(
        problem_type.quant_data_type,
        solution,
    )
    try:
        mechanism = forward_mechanism_contract(solution.operand_source)
    except ValueError as error:
        _reject(
            reasons,
            "solution.forward.kernel_spec",
            str(error),
            "OperandSource",
            source="ForwardMechanismContract",
        )
        return
    mechanism_rejection = mechanism.rejection_reason(contract)
    if mechanism_rejection is not None:
        _reject(
            reasons,
            "solution.forward.problem_contract",
            mechanism_rejection,
            "Solution",
            source="ForwardMechanismContract",
        )
        return

    problem_size = solution_key.problem_size
    if mechanism.lowering == "SignedInt8":
        _validate_forward_mechanism_control(
            mechanism,
            contract,
            problem_size,
            solution,
            reasons,
        )
        if reasons:
            return

    spec_rejection = forward_kernel_spec_rejection_reason(solution)
    if spec_rejection is not None:
        _reject(
            reasons,
            "solution.forward.kernel_spec",
            spec_rejection,
            "Solution",
            source="ForwardKernelSpec",
        )
        return
    if mechanism.lowering != "SignedInt8":
        _validate_forward_mechanism_control(
            mechanism,
            contract,
            problem_size,
            solution,
            reasons,
        )


def _validate_backward_problem_size(
    problem_type: ProblemType,
    problem_size: ProblemSize,
    solution: BackwardSolution,
    reasons: list[RejectReason],
) -> None:
    for parameter, value in (
        ("M", problem_size.m),
        ("N", problem_size.n),
        ("K", problem_size.k),
    ):
        if value <= 0:
            _reject(
                reasons,
                f"problem_size.{parameter.lower()}.positive",
                f"{parameter} must be positive",
                parameter,
                source="ProblemSize",
            )
    quant_format = BACKWARD_QUANT_FORMATS[problem_type.quant_data_type]
    if problem_size.n % quant_format.block_values:
        _reject(
            reasons,
            "problem_size.n.quant_block",
            "N must contain complete packed-weight quant blocks",
            "N",
            "QuantDataType",
            source="ProblemSize",
        )
    if problem_type.quant_data_type == "Q8_0":
        compatible_n_tile = solution.macro_tile1 % quant_format.block_values == 0
    else:
        compatible_n_tile = quant_format.block_values % solution.macro_tile1 == 0
    if solution.macro_tile1 > 0 and not compatible_n_tile:
        _reject(
            reasons,
            "problem_size.n.decoder_tile",
            "MacroTile1 must preserve complete decoder ownership within a quant block",
            "N",
            "MacroTile1",
            "QuantDataType",
            source="ProblemSize",
        )
    for parameter, value, divisor, divisor_name in (
        ("M", problem_size.m, solution.macro_tile0, "MacroTile0"),
        ("N", problem_size.n, solution.macro_tile1, "MacroTile1"),
        ("K", problem_size.k, solution.depth_u, "DepthU"),
    ):
        if divisor > 0 and value % divisor:
            _reject(
                reasons,
                f"problem_size.{parameter.lower()}.tile_multiple",
                f"{parameter} must be divisible by {divisor_name}",
                parameter,
                divisor_name,
                source="ProblemSize",
            )
    decoder_threads = min(solution.num_threads, 128)
    if (
        decoder_threads > 0
        and solution.decoder_width > 0
        and solution.depth_u
        * solution.macro_tile1
        // (decoder_threads * solution.decoder_width)
        == 0
    ):
        _reject(
            reasons,
            "problem_size.decoder_rows.empty",
            "DepthU and MacroTile1 must provide at least one decoder row",
            "DepthU",
            "MacroTile1",
            "DecoderWidth",
            "WorkGroup",
            source="ProblemSize",
        )
    if solution.macro_tile0 > 0 and solution.work_group_mapping > 0:
        m_blocks = problem_size.m // solution.macro_tile0
        if m_blocks % solution.work_group_mapping:
            _reject(
                reasons,
                "problem_size.m.work_group_mapping",
                "M tile count must be divisible by WorkGroupMapping",
                "M",
                "MacroTile0",
                "WorkGroupMapping",
                source="ProblemSize",
            )


def _validate_backward_solution_parameters(
    solution: BackwardSolution, reasons: list[RejectReason]
) -> None:
    pilot = BackwardSolution.pilot()
    implemented = (
        ("kernel_language", "KernelLanguage"),
        ("isa", "ISA"),
        ("wavefront_size", "WavefrontSize"),
        ("global_read_vector_width_a", "GlobalReadVectorWidthA"),
        ("global_read_vector_width_b", "GlobalReadVectorWidthB"),
        ("local_read_vector_width", "LocalReadVectorWidth"),
        ("num_elements_per_batch_store", "NumElementsPerBatchStore"),
        ("store_vector_width", "StoreVectorWidth"),
        ("transpose_lds", "TransposeLDS"),
        ("lds_block_size_per_pad_b", "LdsBlockSizePerPadB"),
        ("decoder_width", "DecoderWidth"),
        ("prefetch_packed_weight", "PrefetchPackedWeight"),
    )
    for attribute, parameter in implemented:
        if getattr(solution, attribute) != getattr(pilot, attribute):
            _reject(
                reasons,
                f"solution.{parameter.lower()}.unimplemented",
                f"pilot MMQ backward writer implements only {parameter}={getattr(pilot, attribute)!r}",
                parameter,
            )

    if solution.lds_swizzle_chunk_b not in (0, 4, 8, 16):
        _reject(
            reasons,
            "solution.ldsswizzlechunkb.unimplemented",
            "MMQ backward writer implements only LdsSwizzleChunkB=0, 4, 8, or 16",
            "LdsSwizzleChunkB",
        )
    if solution.lds_pad_b not in (0, 8, 16, 24):
        _reject(
            reasons,
            "solution.ldspadb.unimplemented",
            "MMQ backward writer implements only LdsPadB=0, 8, 16, or 24",
            "LdsPadB",
        )
    if solution.lds_pad_b and solution.lds_swizzle_chunk_b:
        _reject(
            reasons,
            "solution.ldspadb.swizzle",
            "LdsPadB requires an unswizzled LDS layout",
            "LdsPadB",
            "LdsSwizzleChunkB",
        )
    lds_buffering = BackwardLdsBuffering.try_from_serialized(solution.one_lds_buffer)
    schedule = BackwardScheduleIterAlg.try_from_serialized(solution.schedule_iter_alg)
    packed_prefetch = BackwardPackedWeightPrefetch.from_solution(solution)
    q3_extraction = BackwardExtraction.try_from_serialized(solution.q3_k_extraction)
    q3_pairing = BackwardQ3Pairing.try_from_serialized(solution.q3_k_pairing)
    q4_decode_schedule = BackwardQ4DecodeSchedule.try_from_serialized(
        solution.q4_k_decode_schedule
    )
    q5_extraction = BackwardExtraction.try_from_serialized(solution.q5_k_extraction)
    q6_extraction = BackwardExtraction.try_from_serialized(solution.q6_k_extraction)
    q8_extraction = BackwardExtraction.try_from_serialized(solution.q8_0_extraction)
    if lds_buffering is None:
        _reject(
            reasons,
            "solution.1ldsbuffer.unimplemented",
            "MMQ backward writer implements only 1LDSBuffer=0 or 1",
            "1LDSBuffer",
        )
    pipeline_schedule = (
        schedule is BackwardScheduleIterAlg.SIA3
        and schedule.supports_global_read_prefetch(solution.prefetch_global_read)
        and solution.prefetch_global_read == 1
        and solution.depth_u == 32
    ) or (
        schedule is not None
        and schedule.supports_global_read_prefetch(solution.prefetch_global_read)
        and solution.prefetch_global_read == 2
    )
    if lds_buffering is BackwardLdsBuffering.Double and (
        solution.macro_tile0 != 128
        or solution.macro_tile1 not in (64, 128)
        or solution.depth_u not in (32, 64)
        or solution.num_threads != 128
        or not pipeline_schedule
        or solution.prefetch_local_read != 1
        or solution.lds_swizzle_chunk_b != 8
        or solution.packed_weight_lane_share != 1
        or packed_prefetch.includes_next_tile
    ):
        _reject(
            reasons,
            "solution.1ldsbuffer.pipeline",
            "1LDSBuffer=0 requires supported 128x64/128x128 ownership, SIA3/PGR1 or SIA4-5/PGR2, PLR1, XOR-8, and independent packed reads",
            "1LDSBuffer",
            "MacroTile0",
            "MacroTile1",
            "DepthU",
            "ScheduleIterAlg",
            "PrefetchGlobalRead",
            "PrefetchLocalRead",
            "LdsSwizzleChunkB",
            "PackedWeightLaneShare",
            "PrefetchPackedWeightNext",
        )
    if schedule is None:
        _reject(
            reasons,
            "solution.scheduleiteralg.unimplemented",
            "MMQ backward writer implements only ScheduleIterAlg=2, 3, 4, or 5",
            "ScheduleIterAlg",
        )
    if solution.prefetch_global_read not in (1, 2):
        _reject(
            reasons,
            "solution.prefetchglobalread.unimplemented",
            "MMQ backward writer implements only PrefetchGlobalRead=1 or 2",
            "PrefetchGlobalRead",
        )
    if solution.prefetch_global_read == 2 and (
        schedule is None
        or not schedule.supports_global_read_prefetch(solution.prefetch_global_read)
    ):
        _reject(
            reasons,
            "solution.prefetchglobalread.schedule",
            "PrefetchGlobalRead=2 requires ScheduleIterAlg=4 or 5",
            "PrefetchGlobalRead",
            "ScheduleIterAlg",
        )
    if solution.prefetch_local_read not in (1, 2):
        _reject(
            reasons,
            "solution.prefetchlocalread.unimplemented",
            "MMQ backward writer implements only PrefetchLocalRead=1 or 2",
            "PrefetchLocalRead",
        )
    if solution.prefetch_local_read == 2 and (
        schedule is None
        or not schedule.supports_local_read_prefetch(solution.prefetch_local_read)
    ):
        _reject(
            reasons,
            "solution.prefetchlocalread.schedule",
            "PrefetchLocalRead=2 requires ScheduleIterAlg=3",
            "PrefetchLocalRead",
            "ScheduleIterAlg",
        )
    if solution.packed_weight_lane_share not in (1, 2):
        _reject(
            reasons,
            "solution.packedweightlaneshare.unimplemented",
            "MMQ backward writer implements PackedWeightLaneShare=1 or 2",
            "PackedWeightLaneShare",
        )
    if q3_extraction not in (BackwardExtraction.packed, BackwardExtraction.scalar):
        _reject(
            reasons,
            "solution.q3kextraction.unimplemented",
            "Q3KExtraction must be 'packed' or 'scalar'",
            "Q3KExtraction",
        )
    if q3_pairing is None:
        _reject(
            reasons,
            "solution.q3kpairing.unimplemented",
            "Q3KPairing must be 'Inactive', 'Partial', or 'Full'",
            "Q3KPairing",
        )
    if q4_decode_schedule is None:
        _reject(
            reasons,
            "solution.q4kdecodeschedule.unimplemented",
            "Q4KDecodeSchedule must be 'Serial' or 'DependencyBatch4'",
            "Q4KDecodeSchedule",
        )
    if q5_extraction not in (BackwardExtraction.packed, BackwardExtraction.scalar):
        _reject(
            reasons,
            "solution.q5kextraction.unimplemented",
            "Q5KExtraction must be 'packed' or 'scalar'",
            "Q5KExtraction",
        )
    if q6_extraction is None:
        _reject(
            reasons,
            "solution.q6kextraction.unimplemented",
            "Q6KExtraction must be 'packed', 'packed_vopd', or 'scalar'",
            "Q6KExtraction",
        )
    if q8_extraction is None:
        _reject(
            reasons,
            "solution.q8kextraction.unimplemented",
            "Q8KExtraction must be 'packed', 'packed_vopd', or 'scalar'",
            "Q8KExtraction",
        )
    if packed_prefetch.includes_next_tile and (
        schedule is not BackwardScheduleIterAlg.SIA4
        or solution.prefetch_global_read != 2
        or solution.prefetch_local_read != 1
    ):
        _reject(
            reasons,
            "solution.prefetchpackedweightnext.schedule",
            "PrefetchPackedWeightNext requires SIA4, PGR2, and PLR1",
            "PrefetchPackedWeightNext",
            "ScheduleIterAlg",
            "PrefetchGlobalRead",
            "PrefetchLocalRead",
        )
    depth_u64_layout = solution.lds_swizzle_chunk_b in (8, 16) or (
        solution.lds_swizzle_chunk_b == 0 and solution.lds_pad_b in (8, 16, 24)
    )
    if solution.depth_u == 64 and (
        schedule is not BackwardScheduleIterAlg.SIA4
        or solution.prefetch_global_read != 2
        or solution.prefetch_local_read != 1
        or not depth_u64_layout
        or solution.packed_weight_lane_share != 1
    ):
        _reject(
            reasons,
            "solution.depthu64.schedule",
            "DepthU=64 requires SIA4, PGR2, PLR1, XOR-8 or row padding, and independent packed reads",
            "DepthU",
            "ScheduleIterAlg",
            "PrefetchGlobalRead",
            "PrefetchLocalRead",
            "LdsSwizzleChunkB",
            "PackedWeightLaneShare",
            "PrefetchPackedWeightNext",
        )
    if (
        schedule is not None
        and schedule.prefetches_a
        and len(solution.matrix_instruction) == 9
    ):
        m_repeats = solution.matrix_instruction[5]
        n_repeats = solution.matrix_instruction[6]
        wave_count = solution.matrix_instruction[7]
        compact_geometry = wave_count == 2 and m_repeats == 1 and n_repeats >= 4
        four_wave_geometry = wave_count == 4 and m_repeats * n_repeats <= 16
        if not compact_geometry and not four_wave_geometry:
            _reject(
                reasons,
                "solution.scheduleiteralg.geometry",
                "ScheduleIterAlg=4 or 5 requires a supported compact or four-wave geometry",
                "ScheduleIterAlg",
                "MacroTile0",
                "MacroTile1",
                "WorkGroup",
            )
    if solution.work_group_mapping not in (1, 2, 4, 8, 128, 256):
        _reject(
            reasons,
            "solution.workgroupmapping.unimplemented",
            "MMQ backward writer implements WorkGroupMapping=1, 2, 4, 8, 128, or 256",
            "WorkGroupMapping",
        )

    instruction = solution.matrix_instruction
    geometry_supported = (
        len(instruction) == 9
        and instruction[:5] == (16, 16, 16, 1, 1)
        and instruction[5] in (1, 2, 4)
        and instruction[6] in (2, 4, 8)
        and instruction[7] in (2, 4, 8)
        and instruction[8] == 1
        and solution.work_group == (32, instruction[7], 1)
        and solution.depth_u in (32, 64)
    )
    if not geometry_supported:
        _reject(
            reasons,
            "solution.geometry.unimplemented",
            "MMQ backward requires wave32 BF16 WMMA tiles with power-of-two 1/2/4 M repeats, 2/4/8 N repeats, 2/4/8 waves, and DepthU 32 or 64",
            "MatrixInstruction",
            "DepthU",
            "WorkGroup",
        )

    instruction = solution.matrix_instruction
    if len(instruction) == 9:
        macro_tile0 = instruction[0] * instruction[5] * instruction[7]
        macro_tile1 = instruction[1] * instruction[6] * instruction[8]
        if (macro_tile0, macro_tile1) != (
            solution.macro_tile0,
            solution.macro_tile1,
        ):
            _reject(
                reasons,
                "solution.matrix_instruction.macro_tile",
                "MatrixInstruction does not derive MacroTile0/1",
                "MatrixInstruction",
                "MacroTile0",
                "MacroTile1",
                source="SolutionStructs",
            )
    if solution.num_threads not in (64, 128, 256):
        _reject(
            reasons,
            "solution.work_group.num_threads",
            "MMQ backward writer requires NumThreads=64, 128, or 256",
            "WorkGroup",
            source="SolutionStructs",
        )


def validate_solution(solution_key: SolutionKey) -> tuple[RejectReason, ...]:
    reasons: list[RejectReason] = []
    if solution_key.problem_type.operation_type == "MMQForward":
        _validate_forward_solution(solution_key, reasons)
        return tuple(reasons)
    if isinstance(solution_key.solution, GroupedBackwardSolution):
        try:
            from .grouped_mmq_bwd_spec import DerivedGroupedBackwardState

            state = DerivedGroupedBackwardState.from_solution_key(solution_key)
        except (TypeError, ValueError) as error:
            _reject(
                reasons,
                "solution.grouped_backward.schema",
                str(error),
                "Problem",
                "Solution",
                source="GroupedBackwardContract",
            )
            return tuple(reasons)
        reasons.extend(validate_solution(state.ordinary.solution_key))
        compute = solution_key.solution.compute
        if compute.work_group_mapping != 1:
            _reject(
                reasons,
                "solution.grouped_backward.work_group_mapping",
                "routed backward requires WorkGroupMapping=1",
                "WorkGroupMapping",
                source="GroupedBackwardContract",
            )
        if state.spec.row_tail.value == "Mixed128_64":
            mixed_tail_valid = (
                compute.macro_tile0 == 128
                and compute.matrix_instruction[5] == 2
                and compute.matrix_instruction[7] == 4
                and compute.num_threads == 128
                and compute.one_lds_buffer == 1
                and compute.schedule_iter_alg == 5
                and compute.prefetch_global_read == 2
                and compute.prefetch_local_read == 1
                and not compute.prefetch_packed_weight_next
            )
            if not mixed_tail_valid:
                _reject(
                    reasons,
                    "solution.grouped_backward.row_tail",
                    "Mixed128_64 requires the supported 128xN, two-M-tile, 128-thread, single-LDS primary geometry",
                    "MacroTile0",
                    "MatrixInstruction",
                    "NumThreads",
                    "1LDSBuffer",
                    "ScheduleIterAlg",
                    "PrefetchGlobalRead",
                    "PrefetchLocalRead",
                    "PrefetchPackedWeightNext",
                    source="GroupedBackwardContract",
                )
        if state.contract.problem_size.n % compute.macro_tile1:
            _reject(
                reasons,
                "solution.grouped_backward.n_tile",
                "grouped backward N must be exactly divisible by MacroTile1",
                "MacroTile1",
                source="GroupedBackwardContract",
            )
        return tuple(reasons)
    if not isinstance(solution_key.solution, BackwardSolution):
        _reject(
            reasons,
            "solution.backward.schema",
            "MMQ backward requires BackwardSolution",
            "Solution",
            source="SolutionStructs",
        )
        return tuple(reasons)
    _validate_backward_problem_type(solution_key.problem_type, reasons)
    _validate_backward_solution_parameters(solution_key.solution, reasons)
    quant_type = solution_key.problem_type.quant_data_type
    if quant_type not in BACKWARD_QUANT_FORMATS:
        return tuple(reasons)
    mechanism = backward_mechanism_contract(quant_type)
    lds_buffering = BackwardLdsBuffering.try_from_serialized(
        solution_key.solution.one_lds_buffer
    )
    schedule = BackwardScheduleIterAlg.try_from_serialized(
        solution_key.solution.schedule_iter_alg
    )
    packed_prefetch = BackwardPackedWeightPrefetch.from_solution(solution_key.solution)
    q3_extraction = BackwardExtraction.try_from_serialized(
        solution_key.solution.q3_k_extraction
    )
    q3_pairing = BackwardQ3Pairing.try_from_serialized(
        solution_key.solution.q3_k_pairing
    )
    q4_decode_schedule = BackwardQ4DecodeSchedule.try_from_serialized(
        solution_key.solution.q4_k_decode_schedule
    )
    q5_extraction = BackwardExtraction.try_from_serialized(
        solution_key.solution.q5_k_extraction
    )
    q5_nibble_shift = (
        BackwardQ5NibbleShift.Hoisted
        if solution_key.solution.q5_k_nibble_shift_hoist
        else BackwardQ5NibbleShift.Inline
    )
    q5_metadata_load = (
        BackwardQ5MetadataLoad.Vector
        if solution_key.solution.q5_k_metadata_vector_load
        else BackwardQ5MetadataLoad.Scalar
    )
    q6_extraction = BackwardExtraction.try_from_serialized(
        solution_key.solution.q6_k_extraction
    )
    q8_extraction = BackwardExtraction.try_from_serialized(
        solution_key.solution.q8_0_extraction
    )
    if (
        solution_key.problem_type.quant_data_type == "Q5_K"
        and solution_key.solution.macro_tile1 == 64
        and solution_key.solution.q5_k_metadata_vector_load
    ):
        _reject(
            reasons,
            "solution.q5kmetadata.n64",
            "Q5_K N64 does not implement vector metadata loading",
            "MacroTile1",
            "Q5KMetadataVectorLoad",
            source="ProblemType",
        )
    if (
        solution_key.problem_type.quant_data_type == "Q5_K"
        and solution_key.solution.macro_tile1 == 64
        and solution_key.solution.packed_weight_lane_share != 1
    ):
        _reject(
            reasons,
            "solution.packedweightlaneshare.q5n64",
            "Q5_K N64 requires independent packed-weight loads",
            "MacroTile1",
            "PackedWeightLaneShare",
            source="ProblemType",
        )
    if (
        solution_key.solution.depth_u == 64
        and solution_key.solution.lds_pad_b in (8, 16, 24)
        and not mechanism.supports_padded_depth(solution_key.solution.depth_u)
    ):
        _reject(
            reasons,
            "solution.depthu64.padded.decoder",
            "the selected decoder does not implement DepthU64 row padding",
            "DepthU",
            "LdsPadB",
            source="ProblemType",
        )
    if solution_key.solution.packed_weight_lane_share not in (
        mechanism.lane_share_values
    ):
        _reject(
            reasons,
            "solution.decoder.packedweightlaneshare",
            "PackedWeightLaneShare is not implemented by the selected decoder",
            "PackedWeightLaneShare",
            source="ProblemType",
        )
    if (
        lds_buffering is BackwardLdsBuffering.Double
        and solution_key.solution.macro_tile1 == 64
        and not mechanism.supports_pipeline_n(solution_key.solution.macro_tile1)
    ):
        _reject(
            reasons,
            "solution.pipeline.n64.quant",
            "the selected decoder does not implement N64 decoded-B pipelining",
            "1LDSBuffer",
            "MacroTile1",
            source="ProblemType",
        )
    if (
        lds_buffering is BackwardLdsBuffering.Double
        and schedule is BackwardScheduleIterAlg.SIA3
        and not mechanism.supports_pipeline_schedule(schedule)
    ):
        _reject(
            reasons,
            "solution.pipeline.sia3.quant",
            "the selected decoder does not implement SIA3 decoded-B pipelining",
            "1LDSBuffer",
            "ScheduleIterAlg",
            source="ProblemType",
        )
    if (
        lds_buffering is BackwardLdsBuffering.Double
        and solution_key.solution.depth_u == 64
        and not mechanism.supports_pipeline_depth(solution_key.solution.depth_u)
    ):
        _reject(
            reasons,
            "solution.pipeline.depthu64.quant",
            "the selected decoder does not implement DepthU64 decoded-B pipelining",
            "1LDSBuffer",
            "DepthU",
            source="ProblemType",
        )
    if (
        solution_key.solution.depth_u == 64
        and packed_prefetch.includes_next_tile
        and not mechanism.supports_next_packed_depth(solution_key.solution.depth_u)
    ):
        _reject(
            reasons,
            "solution.depthu64.prefetchpacked.quant",
            "the selected decoder does not implement DepthU64 next-packed-tile prefetch",
            "DepthU",
            "PrefetchPackedWeightNext",
            source="ProblemType",
        )
    if (
        solution_key.problem_type.quant_data_type == "Q4_K"
        and q5_extraction is not BackwardExtraction.packed
    ):
        _reject(
            reasons,
            "solution.q5kextraction.q4_inert",
            "Q5KExtraction='scalar' is valid only for Q5_K",
            "Q5KExtraction",
            source="ProblemType",
        )
    if solution_key.problem_type.quant_data_type != "Q6_K" and (
        q6_extraction is not BackwardExtraction.packed
    ):
        _reject(
            reasons,
            "solution.q6.controls.inert",
            "Q6KExtraction='scalar' is valid only for Q6_K",
            "Q6KExtraction",
            source="ProblemType",
        )
    if solution_key.problem_type.quant_data_type != "Q8_0" and (
        q8_extraction is not BackwardExtraction.packed
    ):
        _reject(
            reasons,
            "solution.q8.controls.inert",
            "Q8KExtraction='packed_vopd' or 'scalar' is valid only for Q8_0",
            "Q8KExtraction",
            source="ProblemType",
        )
    if quant_type != "Q4_K" and (
        q4_decode_schedule is not BackwardQ4DecodeSchedule.Serial
    ):
        _reject(
            reasons,
            "solution.q4.controls.inert",
            "Q4-specific decode scheduling is valid only for Q4_K",
            "Q4KDecodeSchedule",
            source="ProblemType",
        )
    if solution_key.problem_type.quant_data_type != "Q5_K" and (
        q5_extraction is not BackwardExtraction.packed
        or q5_nibble_shift is BackwardQ5NibbleShift.Hoisted
        or q5_metadata_load is BackwardQ5MetadataLoad.Vector
    ):
        _reject(
            reasons,
            "solution.q5.controls.inert",
            "Q5-specific controls are valid only for Q5_K",
            "Q5KExtraction",
            "Q5KNibbleShiftHoist",
            "Q5KMetadataVectorLoad",
            source="ProblemType",
        )
    if quant_type == "Q3_K":
        pairing = q3_pairing
        extraction = q3_extraction
        if extraction is BackwardExtraction.packed and pairing not in (
            BackwardQ3Pairing.Partial,
            BackwardQ3Pairing.Full,
        ):
            _reject(
                reasons,
                "solution.q3.packed.pairing",
                "packed Q3_K extraction requires explicit Partial or Full pairing",
                "Q3KExtraction",
                "Q3KPairing",
                source="ProblemType",
            )
        if (
            extraction is BackwardExtraction.scalar
            and pairing is not BackwardQ3Pairing.Inactive
        ):
            _reject(
                reasons,
                "solution.q3.scalar.pairing",
                "scalar Q3_K extraction requires Q3KPairing='Inactive'",
                "Q3KExtraction",
                "Q3KPairing",
                source="ProblemType",
            )
        decoder_threads = min(solution_key.solution.num_threads, 128)
        decoder_rows = (
            solution_key.solution.depth_u
            * solution_key.solution.macro_tile1
            // (decoder_threads * solution_key.solution.decoder_width)
        )
        if pairing is BackwardQ3Pairing.Full and decoder_rows > 2:
            _reject(
                reasons,
                "solution.q3.full_pairing.decoder_rows",
                "Full Q3_K pairing supports at most two decoder rows",
                "Q3KPairing",
                "DecoderWidth",
                "DepthU",
                "MacroTile1",
                "WorkGroup",
                source="BackwardMechanismContract",
            )
    elif (
        q3_extraction is not BackwardExtraction.packed
        or q3_pairing is not BackwardQ3Pairing.Inactive
    ):
        _reject(
            reasons,
            "solution.q3.controls.inert",
            "Q3-specific controls are valid only for Q3_K",
            "Q3KExtraction",
            "Q3KPairing",
            source="ProblemType",
        )
    _validate_backward_problem_size(
        solution_key.problem_type,
        solution_key.problem_size,
        solution_key.solution,
        reasons,
    )
    try:
        physical = derive_backward_physical_plan(
            DerivedBackwardState.from_solution_key(solution_key)
        )
    except (IndexError, ValueError, ZeroDivisionError) as error:
        _reject(
            reasons,
            "solution.physical.capacity",
            f"backward physical planning failed: {error}",
            "Solution",
            source="BackwardPhysicalPlan",
        )
    else:
        if physical.resources.lds_num_bytes > 65536:
            _reject(
                reasons,
                "solution.lds.capacity",
                "backward physical LDS use exceeds 64 KiB",
                "DepthU",
                "MacroTile1",
                "LdsPadB",
                "1LDSBuffer",
                source="BackwardPhysicalPlan",
            )
    return tuple(reasons)
