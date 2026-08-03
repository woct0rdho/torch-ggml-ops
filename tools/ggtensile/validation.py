from dataclasses import dataclass, replace

from .model import (
    BackwardSolution,
    ForwardSolution,
    ProblemSize,
    ProblemType,
    SolutionKey,
)


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
    if problem_type.quant_data_type not in {"Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"}:
        _reject(
            reasons,
            "problem_type.quant_data_type.unsupported",
            "MMQ backward supports only Q3_K, Q4_K, Q5_K, Q6_K, and Q8_0",
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


def _validate_forward_solution(
    solution_key: SolutionKey, reasons: list[RejectReason]
) -> None:
    problem_type = solution_key.problem_type
    problem_size = solution_key.problem_size
    solution = solution_key.solution
    supported_problem_types = {
        ProblemType.mmq_forward_q4_k(),
        ProblemType.mmq_forward_q5_k(),
    }
    if problem_type not in supported_problem_types:
        _reject(
            reasons,
            "problem_type.forward.unsupported",
            "MMQ forward requires an exact Q4_K/Q8_1 or Q5_K/Q8_1 problem type",
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
    if problem_type == ProblemType.mmq_forward_q5_k():
        q5_retained = ForwardSolution.q5_k_hip_decoded_staged_retained()
        q5_metadata_after_low = (
            ForwardSolution.q5_k_hip_decoded_staged_metadata_after_low_wmma()
        )
        q5_independent = ForwardSolution.q5_k_hip_decoded_staged_independent_extraction_metadata_after_low_wmma()
        q5_extraction = (
            replace(
                solution,
                epilogue_tiles_ahead=8,
                epilogue_dependency_width=1,
                epilogue_priority=0,
                accumulator_initialization="ScalarCopy",
            )
            == q5_independent
            and solution.epilogue_tiles_ahead in range(1, 9)
            and solution.epilogue_dependency_width in range(1, 9)
            and solution.epilogue_priority in range(4)
            and solution.accumulator_initialization in ("ScalarCopy", "VopdPair")
        )
        if solution not in (q5_retained, q5_metadata_after_low) and not q5_extraction:
            _reject(
                reasons,
                "solution.forward.control.unimplemented",
                "Q5_K forward implements only the retained decoded-staged controls",
                "Solution",
            )
            return
        if problem_size.m not in (2048, 8192, 32768):
            _reject(
                reasons,
                "problem_size.m.forward_production",
                "Q5_K forward requires M=2048, 8192, or 32768",
                "M",
                source="ProblemSize",
            )
        if (problem_size.n, problem_size.k) not in {(512, 2048), (2048, 512)}:
            _reject(
                reasons,
                "problem_size.nk.forward_production",
                "Q5_K forward requires an exact production (N,K) pair",
                "N",
                "K",
                source="ProblemSize",
            )
        for parameter, value, divisor in (
            ("M", problem_size.m, solution.macro_tile0),
            ("N", problem_size.n, solution.macro_tile1),
            ("K", problem_size.k, 256),
        ):
            if value <= 0 or value % divisor:
                _reject(
                    reasons,
                    f"problem_size.{parameter.lower()}.forward_tile_multiple",
                    f"{parameter} must be a positive multiple of {divisor}",
                    parameter,
                    source="ProblemSize",
                )
        return
    pilot = ForwardSolution.q4_k_pilot()
    wave_reuse = ForwardSolution.q4_k_wave_reuse()
    wave_batch = ForwardSolution.q4_k_wave_batch4()
    hip_staged = ForwardSolution.q4_k_hip_staged()
    hip_decoded_staged = ForwardSolution.q4_k_hip_decoded_staged()
    hip_decoded_staged_retained = ForwardSolution.q4_k_hip_decoded_staged_retained()
    hip_decoded_staged_metadata_after_low_wmma = (
        ForwardSolution.q4_k_hip_decoded_staged_metadata_after_low_wmma()
    )
    hip_decoded_staged_independent_extraction_metadata_after_low_wmma = ForwardSolution.q4_k_hip_decoded_staged_independent_extraction_metadata_after_low_wmma()
    hip_decoded_staged_shared_down_m8192 = (
        ForwardSolution.q4_k_hip_decoded_staged_shared_down_m8192()
    )
    hip_decoded_staged_shared_down_m32768 = (
        ForwardSolution.q4_k_hip_decoded_staged_shared_down_m32768()
    )
    hip_decoded_staged_shared_down_m2048_metadata_after_low_wmma = ForwardSolution.q4_k_hip_decoded_staged_shared_down_m2048_metadata_after_low_wmma()
    hip_decoded_staged_shared_down_m8192_metadata_after_low_wmma = ForwardSolution.q4_k_hip_decoded_staged_shared_down_m8192_metadata_after_low_wmma()
    hip_decoded_staged_shared_down_m32768_metadata_after_low_wmma = ForwardSolution.q4_k_hip_decoded_staged_shared_down_m32768_metadata_after_low_wmma()
    hip_decoded_staged_narrow_m32768_metadata_after_low_wmma = (
        ForwardSolution.q4_k_hip_decoded_staged_narrow_m32768_metadata_after_low_wmma()
    )
    hip_decoded_staged_query_m2048_metadata_after_low_wmma = (
        ForwardSolution.q4_k_hip_decoded_staged_query_m2048_metadata_after_low_wmma()
    )
    hip_decoded_staged_query_m8192_metadata_after_low_wmma = (
        ForwardSolution.q4_k_hip_decoded_staged_query_m8192_metadata_after_low_wmma()
    )
    hip_decoded_staged_query_m32768_metadata_after_low_wmma = (
        ForwardSolution.q4_k_hip_decoded_staged_query_m32768_metadata_after_low_wmma()
    )
    implemented = (
        pilot,
        wave_reuse,
        wave_batch,
        hip_staged,
        hip_decoded_staged,
        hip_decoded_staged_retained,
        hip_decoded_staged_metadata_after_low_wmma,
        hip_decoded_staged_independent_extraction_metadata_after_low_wmma,
        hip_decoded_staged_shared_down_m8192,
        hip_decoded_staged_shared_down_m32768,
        hip_decoded_staged_shared_down_m2048_metadata_after_low_wmma,
        hip_decoded_staged_shared_down_m8192_metadata_after_low_wmma,
        hip_decoded_staged_shared_down_m32768_metadata_after_low_wmma,
        hip_decoded_staged_narrow_m32768_metadata_after_low_wmma,
        hip_decoded_staged_query_m2048_metadata_after_low_wmma,
        hip_decoded_staged_query_m8192_metadata_after_low_wmma,
        hip_decoded_staged_query_m32768_metadata_after_low_wmma,
    )
    if solution not in implemented:
        _reject(
            reasons,
            "solution.forward.control.unimplemented",
            "forward implements only the direct, wave-reuse, four-tile-wave-batch, raw staged, decoded staged, and retained decoded-staged controls",
            "Solution",
        )
        return
    allowed_pairs = {(512, 2048), (2048, 512), (2048, 4096), (8192, 2048)}
    if problem_size.m not in (2048, 8192, 32768):
        _reject(
            reasons,
            "problem_size.m.forward_production",
            "Q4_K forward requires M=2048, 8192, or 32768",
            "M",
            source="ProblemSize",
        )
    if (problem_size.n, problem_size.k) not in allowed_pairs:
        _reject(
            reasons,
            "problem_size.nk.forward_production",
            "Q4_K forward requires an exact production (N,K) pair",
            "N",
            "K",
            source="ProblemSize",
        )
    if solution.metadata_schedule in (
        "IndependentExtraction",
        "IndependentExtractionMetadataAfterLowWmma",
    ):
        if (
            solution
            == hip_decoded_staged_independent_extraction_metadata_after_low_wmma
        ):
            expected = solution
        else:
            if solution.metadata_schedule == "IndependentExtraction":
                selected_by_size = {
                    (8192, 2048, 512): hip_decoded_staged_shared_down_m8192,
                    (32768, 2048, 512): hip_decoded_staged_shared_down_m32768,
                }
            else:
                selected_by_size = {
                    (2048, 2048, 512): (
                        hip_decoded_staged_shared_down_m2048_metadata_after_low_wmma
                    ),
                    (8192, 2048, 512): (
                        hip_decoded_staged_shared_down_m8192_metadata_after_low_wmma
                    ),
                    (32768, 2048, 512): (
                        hip_decoded_staged_shared_down_m32768_metadata_after_low_wmma
                    ),
                    (32768, 512, 2048): (
                        hip_decoded_staged_narrow_m32768_metadata_after_low_wmma
                    ),
                    (2048, 8192, 2048): (
                        hip_decoded_staged_query_m2048_metadata_after_low_wmma
                    ),
                    (8192, 8192, 2048): (
                        hip_decoded_staged_query_m8192_metadata_after_low_wmma
                    ),
                    (32768, 8192, 2048): (
                        hip_decoded_staged_query_m32768_metadata_after_low_wmma
                    ),
                }
            expected = selected_by_size.get(
                (problem_size.m, problem_size.n, problem_size.k)
            )
        if solution != expected:
            _reject(
                reasons,
                "solution.forward.metadata_schedule.key",
                "independent metadata extraction and epilogue scheduling require their measured exact shared-down key",
                "MetadataSchedule",
                "EpilogueTilesAhead",
                "EpilogueDependencyWidth",
                "EpiloguePriority",
                "M",
                "N",
                "K",
                source="SolutionStructs",
            )
    for parameter, value, divisor in (
        ("M", problem_size.m, solution.macro_tile0),
        ("N", problem_size.n, solution.macro_tile1),
        ("K", problem_size.k, 256),
    ):
        if value <= 0 or value % divisor:
            _reject(
                reasons,
                f"problem_size.{parameter.lower()}.forward_tile_multiple",
                f"{parameter} must be a positive multiple of {divisor}",
                parameter,
                source="ProblemSize",
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
    allowed_n = (512, 2048, 4096)
    if problem_type.quant_data_type == "Q3_K":
        allowed_n = (2048,)
    elif problem_type.quant_data_type == "Q5_K":
        allowed_n = (512, 2048)
    elif problem_type.quant_data_type == "Q6_K":
        allowed_n = (2048,)
    elif problem_type.quant_data_type == "Q8_0":
        allowed_n = (1024, 2048, 4096, 8192)
    if problem_size.n not in allowed_n:
        _reject(
            reasons,
            "problem_size.n.production",
            f"{problem_type.quant_data_type} MMQ campaign requires N=in_features in {allowed_n}",
            "N",
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
    if solution.one_lds_buffer not in (0, 1):
        _reject(
            reasons,
            "solution.1ldsbuffer.unimplemented",
            "MMQ backward writer implements only 1LDSBuffer=0 or 1",
            "1LDSBuffer",
        )
    pipeline_schedule = (
        solution.schedule_iter_alg == 3
        and solution.prefetch_global_read == 1
        and solution.depth_u == 32
    ) or (solution.schedule_iter_alg in (4, 5) and solution.prefetch_global_read == 2)
    if solution.one_lds_buffer == 0 and (
        solution.macro_tile0 != 128
        or solution.macro_tile1 not in (64, 128)
        or solution.depth_u not in (32, 64)
        or solution.num_threads != 128
        or not pipeline_schedule
        or solution.prefetch_local_read != 1
        or solution.lds_swizzle_chunk_b != 8
        or solution.packed_weight_lane_share != 1
        or solution.prefetch_packed_weight_next
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
    if solution.schedule_iter_alg not in (2, 3, 4, 5):
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
    if solution.prefetch_global_read == 2 and solution.schedule_iter_alg not in (4, 5):
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
    if solution.prefetch_local_read == 2 and solution.schedule_iter_alg != 3:
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
    if solution.q3_k_extraction not in ("packed", "scalar"):
        _reject(
            reasons,
            "solution.q3kextraction.unimplemented",
            "Q3KExtraction must be 'packed' or 'scalar'",
            "Q3KExtraction",
        )
    if solution.q5_k_extraction not in ("packed", "scalar"):
        _reject(
            reasons,
            "solution.q5kextraction.unimplemented",
            "Q5KExtraction must be 'packed' or 'scalar'",
            "Q5KExtraction",
        )
    if solution.q6_k_extraction not in ("packed", "packed_vopd", "scalar"):
        _reject(
            reasons,
            "solution.q6kextraction.unimplemented",
            "Q6KExtraction must be 'packed', 'packed_vopd', or 'scalar'",
            "Q6KExtraction",
        )
    if solution.q8_0_extraction not in ("packed", "packed_vopd", "scalar"):
        _reject(
            reasons,
            "solution.q8kextraction.unimplemented",
            "Q8KExtraction must be 'packed', 'packed_vopd', or 'scalar'",
            "Q8KExtraction",
        )
    if solution.prefetch_packed_weight_next and (
        solution.schedule_iter_alg != 4
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
        solution.schedule_iter_alg != 4
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
    if solution.schedule_iter_alg in (4, 5) and (
        solution.macro_tile0,
        solution.macro_tile1,
        solution.num_threads,
    ) not in (
        (32, 64, 64),
        (32, 128, 64),
        (64, 32, 128),
        (64, 64, 128),
        (64, 128, 128),
        (128, 32, 128),
        (128, 64, 128),
        (128, 128, 128),
        (256, 32, 128),
        (256, 64, 128),
    ):
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

    allowed_geometries = {
        ((16, 16, 16, 1, 1, 1, 4, 2, 1), 32, 64, 32, (32, 2, 1)),
        ((16, 16, 16, 1, 1, 1, 4, 2, 1), 32, 64, 64, (32, 2, 1)),
        ((16, 16, 16, 1, 1, 1, 8, 2, 1), 32, 128, 32, (32, 2, 1)),
        ((16, 16, 16, 1, 1, 1, 2, 4, 1), 64, 32, 32, (32, 4, 1)),
        ((16, 16, 16, 1, 1, 1, 2, 4, 1), 64, 32, 64, (32, 4, 1)),
        ((16, 16, 16, 1, 1, 1, 4, 4, 1), 64, 64, 32, (32, 4, 1)),
        ((16, 16, 16, 1, 1, 1, 4, 4, 1), 64, 64, 64, (32, 4, 1)),
        ((16, 16, 16, 1, 1, 1, 8, 4, 1), 64, 128, 32, (32, 4, 1)),
        ((16, 16, 16, 1, 1, 2, 2, 4, 1), 128, 32, 64, (32, 4, 1)),
        ((16, 16, 16, 1, 1, 2, 4, 4, 1), 128, 64, 32, (32, 4, 1)),
        ((16, 16, 16, 1, 1, 2, 4, 4, 1), 128, 64, 64, (32, 4, 1)),
        ((16, 16, 16, 1, 1, 2, 8, 4, 1), 128, 128, 32, (32, 4, 1)),
        ((16, 16, 16, 1, 1, 2, 8, 4, 1), 128, 128, 64, (32, 4, 1)),
        ((16, 16, 16, 1, 1, 4, 2, 4, 1), 256, 32, 64, (32, 4, 1)),
        ((16, 16, 16, 1, 1, 4, 4, 4, 1), 256, 64, 32, (32, 4, 1)),
        ((16, 16, 16, 1, 1, 4, 4, 4, 1), 256, 64, 64, (32, 4, 1)),
        ((16, 16, 16, 1, 1, 2, 8, 8, 1), 256, 128, 32, (32, 8, 1)),
    }
    geometry = (
        solution.matrix_instruction,
        solution.macro_tile0,
        solution.macro_tile1,
        solution.depth_u,
        solution.work_group,
    )
    if geometry not in allowed_geometries:
        _reject(
            reasons,
            "solution.geometry.unimplemented",
            "MMQ backward writer implements the selected 32x64, 32x128, 64x32, 64x64, 64x128, 128x32, 128x64, 128x128, 256x32, 256x64, and 256x128 DepthU32/64 geometries",
            "MatrixInstruction",
            "MacroTile0",
            "MacroTile1",
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
    expected_lds = (
        2
        * (solution.depth_u + solution.lds_pad_b)
        * solution.macro_tile1
        * (2 if solution.one_lds_buffer == 0 else 1)
    )
    if solution.lds_num_bytes != expected_lds:
        _reject(
            reasons,
            "solution.lds_num_bytes",
            "LdsNumBytes must hold the selected decoded-B buffer count",
            "MacroTile1",
            "DepthU",
            "1LDSBuffer",
            "LdsPadB",
            "LdsBlockSizePerPadB",
            source="SolutionStructs",
        )


def validate_solution(solution_key: SolutionKey) -> tuple[RejectReason, ...]:
    reasons: list[RejectReason] = []
    if solution_key.problem_type.operation_type == "MMQForward":
        _validate_forward_solution(solution_key, reasons)
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
    if (
        solution_key.solution.num_threads == 64
        or (
            solution_key.solution.macro_tile0 == 64
            and solution_key.solution.macro_tile1 == 64
        )
    ) and solution_key.problem_type.quant_data_type not in ("Q6_K", "Q8_0"):
        _reject(
            reasons,
            "solution.work_group.small_m_quant",
            "small-M 32x64, 32x128, and 64x64 geometries are implemented only for Q6_K and Q8_0",
            "WorkGroup",
            source="ProblemType",
        )
    if (
        solution_key.solution.depth_u == 64
        and solution_key.solution.lds_pad_b in (8, 16, 24)
        and solution_key.problem_type.quant_data_type not in ("Q6_K", "Q8_0")
    ):
        _reject(
            reasons,
            "solution.depthu64.q8_pad8",
            "DepthU64 with unswizzled row padding is implemented only for Q6_K and Q8_0",
            "DepthU",
            "LdsPadB",
            source="ProblemType",
        )
    if (
        solution_key.problem_type.quant_data_type == "Q8_0"
        and solution_key.solution.packed_weight_lane_share != 1
    ):
        _reject(
            reasons,
            "solution.q8.packedweightlaneshare",
            "Q8_0 currently requires independent packed payload loads",
            "PackedWeightLaneShare",
            source="ProblemType",
        )
    if (
        solution_key.solution.one_lds_buffer == 0
        and solution_key.solution.macro_tile1 == 64
        and solution_key.problem_type.quant_data_type not in ("Q4_K", "Q6_K")
    ):
        _reject(
            reasons,
            "solution.pipeline.n64.quant",
            "128x64 decoded-B pipelining is implemented only for Q4_K and Q6_K",
            "1LDSBuffer",
            "MacroTile1",
            source="ProblemType",
        )
    if (
        solution_key.solution.one_lds_buffer == 0
        and solution_key.solution.schedule_iter_alg == 3
        and solution_key.problem_type.quant_data_type != "Q5_K"
    ):
        _reject(
            reasons,
            "solution.pipeline.sia3.quant",
            "SIA3 decoded-B pipelining is implemented only for Q5_K",
            "1LDSBuffer",
            "ScheduleIterAlg",
            source="ProblemType",
        )
    if (
        solution_key.solution.one_lds_buffer == 0
        and solution_key.solution.depth_u == 64
        and solution_key.problem_type.quant_data_type != "Q5_K"
    ):
        _reject(
            reasons,
            "solution.pipeline.depthu64.quant",
            "DepthU64 decoded-B pipelining is implemented only for Q5_K",
            "1LDSBuffer",
            "DepthU",
            source="ProblemType",
        )
    if (
        solution_key.solution.depth_u == 64
        and solution_key.solution.prefetch_packed_weight_next
        and solution_key.problem_type.quant_data_type != "Q6_K"
    ):
        _reject(
            reasons,
            "solution.depthu64.prefetchpacked.quant",
            "DepthU64 next-packed-tile prefetch is implemented only for Q6_K",
            "DepthU",
            "PrefetchPackedWeightNext",
            source="ProblemType",
        )
    if (
        solution_key.problem_type.quant_data_type == "Q4_K"
        and solution_key.solution.q5_k_extraction != "packed"
    ):
        _reject(
            reasons,
            "solution.q5kextraction.q4_inert",
            "Q5KExtraction='scalar' is valid only for Q5_K",
            "Q5KExtraction",
            source="ProblemType",
        )
    if solution_key.problem_type.quant_data_type != "Q6_K" and (
        solution_key.solution.q6_k_extraction != "packed"
    ):
        _reject(
            reasons,
            "solution.q6.controls.inert",
            "Q6KExtraction='scalar' is valid only for Q6_K",
            "Q6KExtraction",
            source="ProblemType",
        )
    if solution_key.problem_type.quant_data_type != "Q8_0" and (
        solution_key.solution.q8_0_extraction != "packed"
    ):
        _reject(
            reasons,
            "solution.q8.controls.inert",
            "Q8KExtraction='packed_vopd' or 'scalar' is valid only for Q8_0",
            "Q8KExtraction",
            source="ProblemType",
        )
    if solution_key.problem_type.quant_data_type != "Q5_K" and (
        solution_key.solution.q5_k_extraction != "packed"
        or solution_key.solution.q5_k_nibble_shift_hoist
        or solution_key.solution.q5_k_metadata_vector_load
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
    if solution_key.problem_type.quant_data_type != "Q3_K" and (
        solution_key.solution.q3_k_extraction != "packed"
    ):
        _reject(
            reasons,
            "solution.q3.controls.inert",
            "Q3-specific controls are valid only for Q3_K",
            "Q3KExtraction",
            source="ProblemType",
        )
    _validate_backward_problem_size(
        solution_key.problem_type,
        solution_key.problem_size,
        solution_key.solution,
        reasons,
    )
    return tuple(reasons)
