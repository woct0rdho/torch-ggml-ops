"""Derived authorities for research-only paired grouped backward kernels."""

from dataclasses import dataclass, replace

from .grouped_mmq_bwd_pair_model import (
    GroupedBackwardPairProblem,
    GroupedBackwardPairProjectionSchedule,
    GroupedBackwardPairRouteOwnership,
    GroupedBackwardPairSolution,
    GroupedBackwardPairSolutionKey,
)
from .mmq_bwd_spec import (
    BackwardKernelSpec,
    BackwardProblemContract,
    DerivedBackwardState,
    backward_mechanism_contract,
)
from .model import ProblemSize
from .quant_formats import BACKWARD_QUANT_FORMATS
from .schema import SchemaError
from .schema import integer as _integer
from .schema import integer_tuple as _integer_tuple
from .schema import strict_mapping as _mapping
from .schema import string as _string

_U32_MAX = 0xFFFFFFFF


@dataclass(frozen=True)
class GroupedBackwardPairContract:
    quant_type: str
    out_features: int
    in_features: int
    physical_experts: int
    max_route_entries: int
    block_values: int
    packed_weight_block_bytes: int
    kernel_language: str
    isa: tuple[int, int, int]
    wavefront_size: int
    projection_count: int
    arithmetic_contract: str
    destination_type: str
    bf16_rounding: str
    abi_family: str

    @classmethod
    def q3_k(cls) -> "GroupedBackwardPairContract":
        return cls(
            quant_type="Q3_K",
            out_features=512,
            in_features=2048,
            physical_experts=256,
            max_route_entries=256,
            block_values=256,
            packed_weight_block_bytes=110,
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            projection_count=2,
            arithmetic_contract="InterleavedPairFP32Accumulator",
            destination_type="BFloat16",
            bf16_rounding="RNEPreserveNaN",
            abi_family="GroupedBackwardPairV1",
        )

    @classmethod
    def iq2_s(cls) -> "GroupedBackwardPairContract":
        return cls(
            quant_type="IQ2_S",
            out_features=512,
            in_features=2048,
            physical_experts=256,
            max_route_entries=256,
            block_values=256,
            packed_weight_block_bytes=82,
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            projection_count=2,
            arithmetic_contract="InterleavedPairFP32Accumulator",
            destination_type="BFloat16",
            bf16_rounding="RNEPreserveNaN",
            abi_family="GroupedBackwardPairV1",
        )

    @classmethod
    def iq2_xxs(cls) -> "GroupedBackwardPairContract":
        return cls(
            quant_type="IQ2_XXS",
            out_features=2048,
            in_features=4096,
            physical_experts=256,
            max_route_entries=256,
            block_values=256,
            packed_weight_block_bytes=66,
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            projection_count=2,
            arithmetic_contract="InterleavedPairFP32Accumulator",
            destination_type="BFloat16",
            bf16_rounding="RNEPreserveNaN",
            abi_family="GroupedBackwardPairV1",
        )

    @classmethod
    def for_quant_type(cls, quant_type: str) -> "GroupedBackwardPairContract":
        if quant_type == "Q3_K":
            return cls.q3_k()
        if quant_type == "IQ2_S":
            return cls.iq2_s()
        if quant_type == "IQ2_XXS":
            return cls.iq2_xxs()
        raise SchemaError(f"unsupported paired backward quant type {quant_type!r}")

    @classmethod
    def from_solution(
        cls,
        problem: GroupedBackwardPairProblem,
        solution: GroupedBackwardPairSolution,
    ) -> "GroupedBackwardPairContract":
        expected = cls.for_quant_type(problem.quant_data_type)
        return cls(
            quant_type=problem.quant_data_type,
            out_features=problem.out_features,
            in_features=problem.in_features,
            physical_experts=problem.physical_experts,
            max_route_entries=problem.max_route_entries,
            block_values=expected.block_values,
            packed_weight_block_bytes=expected.packed_weight_block_bytes,
            kernel_language=solution.compute.kernel_language,
            isa=solution.compute.isa,
            wavefront_size=solution.compute.wavefront_size,
            projection_count=solution.projection_count,
            arithmetic_contract=(
                "InterleavedPairFP32Accumulator"
                if solution.projection_schedule
                in (
                    GroupedBackwardPairProjectionSchedule.InterleavedDepthU,
                    GroupedBackwardPairProjectionSchedule.DualLdsInterleavedDepthU,
                    GroupedBackwardPairProjectionSchedule.DualLdsGlobalCodebookInterleavedDepthU,
                    GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitInterleavedDepthU,
                    GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersInterleavedDepthU,
                    GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchASerialReadsInterleavedDepthU,
                    GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersOverlapSecondReadPrefetchAInterleavedDepthU,
                    GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsInterleavedDepthU,
                    GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsPrefetchAInterleavedDepthU,
                    GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchAInterleavedDepthU,
                    GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineInterleavedDepthU,
                    GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineGlobalCodebookInterleaveWmmaWaitsDepthU,
                )
                else solution.projection_schedule.value
            ),
            destination_type=expected.destination_type,
            bf16_rounding=expected.bf16_rounding,
            abi_family=expected.abi_family,
        )

    @classmethod
    def from_mapping(cls, value: object) -> "GroupedBackwardPairContract":
        keys = frozenset(cls.iq2_s().to_mapping())
        item = _mapping(value, "GroupedBackwardPairContract", keys)
        actual = cls(
            quant_type=_string(item["quant_type"], "quant_type"),
            out_features=_integer(item["out_features"], "out_features"),
            in_features=_integer(item["in_features"], "in_features"),
            physical_experts=_integer(item["physical_experts"], "physical_experts"),
            max_route_entries=_integer(item["max_route_entries"], "max_route_entries"),
            block_values=_integer(item["block_values"], "block_values"),
            packed_weight_block_bytes=_integer(
                item["packed_weight_block_bytes"], "packed_weight_block_bytes"
            ),
            kernel_language=_string(item["kernel_language"], "kernel_language"),
            isa=_integer_tuple(item["isa"], "isa", 3),
            wavefront_size=_integer(item["wavefront_size"], "wavefront_size"),
            projection_count=_integer(item["projection_count"], "projection_count"),
            arithmetic_contract=_string(
                item["arithmetic_contract"], "arithmetic_contract"
            ),
            destination_type=_string(item["destination_type"], "destination_type"),
            bf16_rounding=_string(item["bf16_rounding"], "bf16_rounding"),
            abi_family=_string(item["abi_family"], "abi_family"),
        )
        if actual != cls.for_quant_type(actual.quant_type):
            raise SchemaError("GroupedBackwardPairContract is not canonical")
        return actual

    def to_mapping(self) -> dict[str, object]:
        return {
            "quant_type": self.quant_type,
            "out_features": self.out_features,
            "in_features": self.in_features,
            "physical_experts": self.physical_experts,
            "max_route_entries": self.max_route_entries,
            "block_values": self.block_values,
            "packed_weight_block_bytes": self.packed_weight_block_bytes,
            "kernel_language": self.kernel_language,
            "isa": list(self.isa),
            "wavefront_size": self.wavefront_size,
            "projection_count": self.projection_count,
            "arithmetic_contract": self.arithmetic_contract,
            "destination_type": self.destination_type,
            "bf16_rounding": self.bf16_rounding,
            "abi_family": self.abi_family,
        }

    def problem(self, aggregate_rows: int) -> GroupedBackwardPairProblem:
        return GroupedBackwardPairProblem(
            self.quant_type,
            aggregate_rows,
            self.out_features,
            self.in_features,
            self.physical_experts,
            self.max_route_entries,
            self.projection_count,
        )

    def ordinary(self, aggregate_rows: int) -> BackwardProblemContract:
        problem_size = ProblemSize(aggregate_rows, self.in_features, self.out_features)
        quant_format = BACKWARD_QUANT_FORMATS[self.quant_type]
        return BackwardProblemContract(
            problem_size,
            self.quant_type,
            quant_format,
            backward_mechanism_contract(self.quant_type),
        )


@dataclass(frozen=True)
class GroupedBackwardPairKernelSpec:
    compute: BackwardKernelSpec
    projection_schedule: GroupedBackwardPairProjectionSchedule
    route_ownership: GroupedBackwardPairRouteOwnership

    @classmethod
    def from_solution(
        cls, solution: GroupedBackwardPairSolution
    ) -> "GroupedBackwardPairKernelSpec":
        return cls(
            BackwardKernelSpec.from_solution(solution.compute),
            solution.projection_schedule,
            solution.route_ownership,
        )

    @classmethod
    def from_mapping(
        cls, value: object, contract: GroupedBackwardPairContract
    ) -> "GroupedBackwardPairKernelSpec":
        item = _mapping(
            value,
            "GroupedBackwardPairKernelSpec",
            frozenset({"compute", "projection_schedule", "route_ownership"}),
        )
        projection_schedule = GroupedBackwardPairProjectionSchedule(
            _string(item["projection_schedule"], "projection_schedule")
        )
        route_ownership = GroupedBackwardPairRouteOwnership(
            _string(item["route_ownership"], "route_ownership")
        )
        return cls(
            BackwardKernelSpec.from_mapping(item["compute"], contract.quant_type),
            projection_schedule,
            route_ownership,
        )

    def to_mapping(self, contract: GroupedBackwardPairContract) -> dict[str, object]:
        return {
            "compute": self.compute.to_mapping(contract.quant_type),
            "projection_schedule": self.projection_schedule.value,
            "route_ownership": self.route_ownership.value,
        }

    def to_solution(
        self, contract: GroupedBackwardPairContract
    ) -> GroupedBackwardPairSolution:
        return GroupedBackwardPairSolution(
            compute=self.compute.to_solution(contract.ordinary(64)),
            projection_schedule=self.projection_schedule,
            route_ownership=self.route_ownership,
            projection_count=contract.projection_count,
        )


def grouped_backward_pair_problem_rejection_reason(
    problem: GroupedBackwardPairProblem,
) -> str | None:
    expected = {
        "Q3_K": GroupedBackwardPairProblem.q3_k,
        "IQ2_S": GroupedBackwardPairProblem.iq2_s,
        "IQ2_XXS": GroupedBackwardPairProblem.iq2_xxs,
    }.get(problem.quant_data_type)
    if expected is None or problem != expected(problem.aggregate_rows):
        return "paired backward implements only the Qwen Q3_K/IQ2_S and DeepSeek IQ2_XXS gate/up geometries"
    if not 0 < problem.aggregate_rows <= _U32_MAX:
        return "paired backward aggregate rows must fit in a positive u32"
    quant_format = BACKWARD_QUANT_FORMATS[problem.quant_data_type]
    packed_row_bytes = (
        problem.in_features // quant_format.block_values * quant_format.block_bytes
    )
    bytes_per_expert = problem.out_features * packed_row_bytes
    if bytes_per_expert > _U32_MAX:
        return "paired backward expert stride must fit in a u32"
    if problem.aggregate_rows * problem.out_features * 2 > _U32_MAX:
        return "paired backward gradient output must fit in a u32 offset"
    if problem.aggregate_rows * problem.in_features * 2 > _U32_MAX:
        return "paired backward gradient input must fit in a u32 offset"
    return None


def grouped_backward_pair_capability_rejection_reason(
    problem: GroupedBackwardPairProblem,
    solution: GroupedBackwardPairSolution,
) -> str | None:
    problem_rejection = grouped_backward_pair_problem_rejection_reason(problem)
    if problem_rejection is not None:
        return problem_rejection
    compute = solution.compute
    expected_m_tiles = compute.macro_tile0 // 64
    prefetches_pair_a = solution.projection_schedule in (
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchASerialReadsInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersOverlapSecondReadPrefetchAInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsPrefetchAInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchAInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineGlobalCodebookInterleaveWmmaWaitsDepthU,
    )
    pipelines_pair_k = solution.projection_schedule in (
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineGlobalCodebookInterleaveWmmaWaitsDepthU,
    )
    interleaves_wmma_waits = compute.schedule_iter_alg == 5
    q3_k = problem.quant_data_type == "Q3_K"
    q3_k_m64_prefetch = (
        q3_k
        and solution
        == GroupedBackwardPairSolution.q3_k_m64_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a()
    )
    iq2_xxs = problem.quant_data_type == "IQ2_XXS"
    iq2_xxs_pipeline = (
        iq2_xxs
        and solution.projection_schedule
        is GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineInterleavedDepthU
    )
    iq2_xxs_final_pipeline = (
        iq2_xxs_pipeline
        and solution
        == GroupedBackwardPairSolution.iq2_xxs_m64_n64_sia5_dual_lds_full_tile_split_k_pipeline()
    )
    iq2_xxs_m192 = (
        iq2_xxs
        and solution
        == GroupedBackwardPairSolution.iq2_xxs_m192_n64_dual_lds_full_tile_split_direct_pointers_overlap_second_read_prefetch_a()
    )
    iq2_xxs_staged_schedules = (
        GroupedBackwardPairProjectionSchedule.InterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsPrefetchAInterleavedDepthU,
        GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchAInterleavedDepthU,
    )
    iq2_xxs_staged = (
        solution.projection_schedule in iq2_xxs_staged_schedules
        or iq2_xxs_final_pipeline
        or iq2_xxs_m192
    )
    checks = (
        (solution.projection_count == 2, "paired backward requires two projections"),
        (
            solution.projection_schedule
            in (
                GroupedBackwardPairProjectionSchedule.InterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsGlobalCodebookInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchASerialReadsInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersOverlapSecondReadPrefetchAInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsPrefetchAInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchAInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineGlobalCodebookInterleaveWmmaWaitsDepthU,
            ),
            "paired backward requires interleaved DepthU projection order",
        ),
        (
            solution.route_ownership
            in (
                GroupedBackwardPairRouteOwnership.SerialRoutes,
                GroupedBackwardPairRouteOwnership.PackedSplitRoutes8,
            ),
            "paired backward route ownership is not implemented",
        ),
        (
            not iq2_xxs
            or solution.route_ownership
            is GroupedBackwardPairRouteOwnership.SerialRoutes,
            "paired IQ2_XXS currently requires serial-route ownership",
        ),
        (
            not q3_k
            or solution
            in (
                GroupedBackwardPairSolution.q3_k_m64_n64(),
                GroupedBackwardPairSolution.q3_k_m128_n64(),
                GroupedBackwardPairSolution.q3_k_m128_n64_dual_lds(),
                GroupedBackwardPairSolution.q3_k_m128_n64_dual_lds_full_tile_split(),
                GroupedBackwardPairSolution.q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers(),
                GroupedBackwardPairSolution.q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a(),
                GroupedBackwardPairSolution.q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers_overlap_second_read_prefetch_a(),
                GroupedBackwardPairSolution.q3_k_m64_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a(),
            ),
            "paired Q3_K currently requires its padded serial anchor schedule",
        ),
        (
            not iq2_xxs or iq2_xxs_staged,
            "paired IQ2_XXS schedule requires its staged-codebook lowering",
        ),
        (
            solution.route_ownership is GroupedBackwardPairRouteOwnership.SerialRoutes
            or interleaves_wmma_waits,
            "paired backward split routes require the SIA5 K pipeline",
        ),
        (compute.kernel_language == "Assembly", "kernel language must be Assembly"),
        (compute.isa == (11, 5, 1), "paired backward ISA must be gfx1151"),
        (compute.wavefront_size == 32, "paired backward requires wave32"),
        (compute.work_group == (32, 4, 1), "paired backward needs four waves"),
        (
            compute.macro_tile0 in (64, 128) or iq2_xxs_m192,
            "paired backward M tile must be 64, 128, or the exact IQ2_XXS M192 identity",
        ),
        (compute.macro_tile1 == 64, "paired backward N tile must be 64"),
        (compute.depth_u == 32, "paired backward reduction tile must be K32"),
        (
            compute.matrix_instruction == (16, 16, 16, 1, 1, expected_m_tiles, 4, 4, 1),
            "paired backward matrix instruction geometry is inconsistent",
        ),
        (compute.one_lds_buffer == 1, "paired backward first body uses one LDS tile"),
        (
            (compute.schedule_iter_alg, compute.prefetch_global_read)
            == (
                (5, 2)
                if interleaves_wmma_waits
                else (4, 2)
                if prefetches_pair_a
                else (2, 1)
            ),
            "paired backward A-read schedule is not implemented",
        ),
        (compute.prefetch_local_read == 1, "paired backward needs one LDS read stage"),
        (compute.prefetch_packed_weight, "paired backward requires current B prefetch"),
        (
            compute.prefetch_packed_weight_next == pipelines_pair_k,
            "paired backward next-B setting does not match its schedule",
        ),
        (
            compute.packed_weight_lane_share == 1,
            "paired backward lane share must be one",
        ),
        (compute.work_group_mapping == 1, "paired backward route mapping must be one"),
        (compute.decoder_width == 16, "paired backward decoder width must be 16"),
        (
            solution.projection_schedule
            not in (
                GroupedBackwardPairProjectionSchedule.DualLdsInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsGlobalCodebookInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchASerialReadsInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersOverlapSecondReadPrefetchAInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsPrefetchAInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchAInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineInterleavedDepthU,
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineGlobalCodebookInterleaveWmmaWaitsDepthU,
            )
            or compute.macro_tile0 == 128
            or q3_k_m64_prefetch
            or (iq2_xxs and compute.macro_tile0 == 64 and iq2_xxs_staged)
            or iq2_xxs_m192,
            "paired backward dual LDS requires M128 or a qualified staged M64 identity",
        ),
    )
    for valid, message in checks:
        if not valid:
            return message

    from .model import ProblemType, SolutionKey
    from .validation import validate_solution

    if iq2_xxs_m192:
        ordinary_compute = replace(
            compute,
            matrix_instruction=(16, 16, 16, 1, 1, 2, 4, 4, 1),
            macro_tile0=128,
        )
    elif interleaves_wmma_waits:
        ordinary_compute = replace(compute, prefetch_packed_weight_next=False)
    else:
        ordinary_compute = compute
    ordinary_tile = ordinary_compute.macro_tile0
    padded_rows = (
        (problem.aggregate_rows + ordinary_tile - 1)
        // ordinary_tile
        * ordinary_tile
    )
    ordinary = SolutionKey(
        ProblemType.mmq_backward(problem.quant_data_type),
        ProblemSize(padded_rows, problem.in_features, problem.out_features),
        ordinary_compute,
    )
    reasons = validate_solution(ordinary)
    if reasons:
        return f"ordinary backward capability rejected: {reasons[0].message}"
    return None


@dataclass(frozen=True)
class DerivedGroupedBackwardPairState:
    contract: GroupedBackwardPairContract
    kernel_spec: GroupedBackwardPairKernelSpec
    ordinary: DerivedBackwardState

    @classmethod
    def from_solution_key(
        cls, key: GroupedBackwardPairSolutionKey
    ) -> "DerivedGroupedBackwardPairState":
        contract = GroupedBackwardPairContract.from_solution(key.problem, key.solution)
        kernel_spec = GroupedBackwardPairKernelSpec.from_solution(key.solution)
        ordinary_contract = contract.ordinary(key.problem.aggregate_rows)
        ordinary = DerivedBackwardState.from_contract_spec(
            ordinary_contract, kernel_spec.compute
        )
        return cls(contract, kernel_spec, ordinary)

    @property
    def packed_row_bytes(self) -> int:
        return (
            self.contract.in_features
            // self.contract.block_values
            * self.contract.packed_weight_block_bytes
        )

    @property
    def bytes_per_expert(self) -> int:
        return self.contract.out_features * self.packed_row_bytes

    @property
    def expected_packed_weight_shape(self) -> tuple[int, int, int]:
        return (
            self.contract.physical_experts,
            self.contract.out_features,
            self.packed_row_bytes,
        )
