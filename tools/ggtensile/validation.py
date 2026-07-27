from dataclasses import dataclass

from .model import ProblemSize, ProblemType, Solution, SolutionKey


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
    source: str = "KernelWriterAssembly",
) -> None:
    reasons.append(RejectReason(rule_id, message, tuple(parameters), source))


def _validate_problem_type(
    problem_type: ProblemType, reasons: list[RejectReason]
) -> None:
    expected = ProblemType.dense_mmq_backward_q4_k()
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
                f"pilot requires {parameter}={getattr(expected, attribute)!r}",
                parameter,
                source="ProblemType",
            )


def _validate_problem_size(
    problem_size: ProblemSize, solution: Solution, reasons: list[RejectReason]
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
    if problem_size.n != 2048:
        _reject(
            reasons,
            "problem_size.n.pilot",
            "Q4_K pilot requires N=in_features=2048",
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


def _validate_solution_parameters(
    solution: Solution, reasons: list[RejectReason]
) -> None:
    pilot = Solution.pilot()
    implemented = (
        ("kernel_language", "KernelLanguage"),
        ("isa", "ISA"),
        ("wavefront_size", "WavefrontSize"),
        ("global_read_vector_width_a", "GlobalReadVectorWidthA"),
        ("global_read_vector_width_b", "GlobalReadVectorWidthB"),
        ("local_read_vector_width", "LocalReadVectorWidth"),
        ("one_lds_buffer", "1LDSBuffer"),
        ("num_elements_per_batch_store", "NumElementsPerBatchStore"),
        ("store_vector_width", "StoreVectorWidth"),
        ("transpose_lds", "TransposeLDS"),
        ("lds_pad_b", "LdsPadB"),
        ("lds_block_size_per_pad_b", "LdsBlockSizePerPadB"),
        ("decoder_width", "DecoderWidth"),
        ("prefetch_packed_weight", "PrefetchPackedWeight"),
    )
    for attribute, parameter in implemented:
        if getattr(solution, attribute) != getattr(pilot, attribute):
            _reject(
                reasons,
                f"solution.{parameter.lower()}.unimplemented",
                f"pilot KernelWriterAssembly implements only {parameter}={getattr(pilot, attribute)!r}",
                parameter,
            )

    if solution.lds_swizzle_chunk_b not in (0, 4, 8, 16):
        _reject(
            reasons,
            "solution.ldsswizzlechunkb.unimplemented",
            "KernelWriterAssembly implements only LdsSwizzleChunkB=0, 4, 8, or 16",
            "LdsSwizzleChunkB",
        )
    if solution.schedule_iter_alg not in (2, 3, 4, 5):
        _reject(
            reasons,
            "solution.scheduleiteralg.unimplemented",
            "KernelWriterAssembly implements only ScheduleIterAlg=2, 3, 4, or 5",
            "ScheduleIterAlg",
        )
    if solution.prefetch_global_read not in (1, 2):
        _reject(
            reasons,
            "solution.prefetchglobalread.unimplemented",
            "KernelWriterAssembly implements only PrefetchGlobalRead=1 or 2",
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
            "KernelWriterAssembly implements only PrefetchLocalRead=1 or 2",
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
            "KernelWriterAssembly implements PackedWeightLaneShare=1 or 2",
            "PackedWeightLaneShare",
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
    if solution.depth_u == 64 and (
        solution.schedule_iter_alg != 4
        or solution.prefetch_global_read != 2
        or solution.prefetch_local_read != 1
        or solution.lds_swizzle_chunk_b != 8
        or solution.packed_weight_lane_share != 1
        or solution.prefetch_packed_weight_next
    ):
        _reject(
            reasons,
            "solution.depthu64.schedule",
            "DepthU=64 requires SIA4, PGR2, PLR1, XOR-8, and independent current-tile packed reads",
            "DepthU",
            "ScheduleIterAlg",
            "PrefetchGlobalRead",
            "PrefetchLocalRead",
            "LdsSwizzleChunkB",
            "PackedWeightLaneShare",
            "PrefetchPackedWeightNext",
        )
    if solution.schedule_iter_alg in (4, 5) and (
        solution.macro_tile0 != 128
        or solution.macro_tile1 != 128
        or solution.num_threads != 128
    ):
        _reject(
            reasons,
            "solution.scheduleiteralg.geometry",
            "ScheduleIterAlg=4 or 5 requires the 128x128 four-wave geometry",
            "ScheduleIterAlg",
            "MacroTile0",
            "MacroTile1",
            "WorkGroup",
        )
    if solution.work_group_mapping not in (1, 2, 4, 8, 128, 256):
        _reject(
            reasons,
            "solution.workgroupmapping.unimplemented",
            "KernelWriterAssembly implements WorkGroupMapping=1, 2, 4, 8, 128, or 256",
            "WorkGroupMapping",
        )

    allowed_geometries = {
        ((16, 16, 16, 1, 1, 2, 8, 4, 1), 128, 128, 32, (32, 4, 1)),
        ((16, 16, 16, 1, 1, 2, 8, 4, 1), 128, 128, 64, (32, 4, 1)),
        ((16, 16, 16, 1, 1, 4, 4, 4, 1), 256, 64, 32, (32, 4, 1)),
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
            "KernelWriterAssembly implements only 128x128x32, 128x128x64, 256x64x32, and 256x128x32 geometry",
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
    if solution.num_threads not in (128, 256):
        _reject(
            reasons,
            "solution.work_group.num_threads",
            "KernelWriterAssembly requires NumThreads=128 or 256",
            "WorkGroup",
            source="SolutionStructs",
        )
    expected_lds = 2 * solution.depth_u * solution.macro_tile1
    if solution.lds_num_bytes != expected_lds:
        _reject(
            reasons,
            "solution.lds_num_bytes",
            "LdsNumBytes must hold one decoded B tile",
            "MacroTile1",
            "DepthU",
            "LdsPadB",
            "LdsBlockSizePerPadB",
            source="SolutionStructs",
        )


def validate_solution(solution_key: SolutionKey) -> tuple[RejectReason, ...]:
    reasons: list[RejectReason] = []
    _validate_problem_type(solution_key.problem_type, reasons)
    _validate_solution_parameters(solution_key.solution, reasons)
    _validate_problem_size(solution_key.problem_size, solution_key.solution, reasons)
    return tuple(reasons)
