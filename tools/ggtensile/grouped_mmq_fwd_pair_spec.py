"""Derived authorities for research-only paired grouped kernels."""

from dataclasses import dataclass

from .grouped_mmq_fwd_model import GroupedActivationAddressing, GroupedOutputStore
from .grouped_mmq_fwd_pair_model import (
    GroupedForwardPairProblem,
    GroupedForwardPairSolution,
    GroupedForwardPairSolutionKey,
    GroupedPairDecodeSchedule,
    GroupedPairOperandSource,
    GroupedPairProjectionSchedule,
    GroupedPairRouteOwnership,
)
from .grouped_mmq_fwd_pair_physical import (
    GroupedIQ2SPairPhysicalPlan,
    GroupedQ3KPairPhysicalPlan,
    grouped_iq2_s_pair_physical_plan,
    grouped_q3_k_pair_physical_plan,
)
from .mmq_fwd_spec import QuantForwardSemantics
from .model import ProblemSize
from .quant_formats import GROUPED_QUANT_FORMATS, Q8_1_D4_BLOCK_VALUES


@dataclass(frozen=True)
class GroupedForwardPairContract:
    quant_type: str
    block_values: int
    activation_layout: str
    activation_block_bytes: int
    packed_weight_block_bytes: int
    kernel_language: str
    isa: tuple[int, int, int]
    wavefront_size: int
    projection_count: int
    arithmetic_contract: str
    route_ownership: GroupedPairRouteOwnership

    @classmethod
    def from_solution(
        cls,
        problem: GroupedForwardPairProblem,
        solution: GroupedForwardPairSolution,
    ) -> "GroupedForwardPairContract":
        quant_format = GROUPED_QUANT_FORMATS.get(problem.quant_data_type)
        if quant_format is None:
            raise ValueError(
                f"unsupported paired quant type {problem.quant_data_type!r}"
            )
        route_ownership = solution.route_ownership
        expected_mechanism = {
            "IQ2_S": (
                GroupedPairOperandSource.IQ2SHalfWeightLds,
                GroupedPairDecodeSchedule.TwoLaneSelectedHalfPayloadPrefetch,
            ),
            "Q3_K": (
                GroupedPairOperandSource.Q3KHalfWeightLds,
                GroupedPairDecodeSchedule.TwoLaneSelectedHalfQ3,
            ),
        }.get(problem.quant_data_type)
        if expected_mechanism is None:
            raise ValueError(
                f"unsupported paired quant type {problem.quant_data_type!r}"
            )
        expected_operand_source, expected_decode_schedule = expected_mechanism
        checks = (
            (problem.projection_count == 2, "paired problem requires two projections"),
            (
                solution.projection_count == 2,
                "paired solution requires two projections",
            ),
            (
                solution.kernel_language == "Assembly",
                "paired kernel language must be Assembly",
            ),
            (solution.isa == (11, 5, 1), "paired ISA must be gfx1151"),
            (solution.wavefront_size == 32, "paired wavefront size must be 32"),
            (
                solution.work_group == (128, 1, 1),
                "paired workgroup must be 128 threads",
            ),
            (solution.macro_tile0 == 64, "paired row tile must be 64"),
            (solution.macro_tile1 == 64, "paired output tile must be 64"),
            (solution.depth_u == 128, "paired depth must be K128"),
            (
                solution.activation_layout == quant_format.activation_layout,
                "paired activation layout mismatch",
            ),
            (
                solution.activation_block_bytes == quant_format.activation_block_bytes,
                "paired activation bytes mismatch",
            ),
            (
                solution.packed_weight_block_bytes == quant_format.block_bytes,
                "paired weight bytes mismatch",
            ),
            (
                solution.operand_source is expected_operand_source,
                "paired operand source mismatch",
            ),
            (
                solution.projection_schedule
                is GroupedPairProjectionSchedule.K128Interleaved,
                "paired projection schedule mismatch",
            ),
            (
                solution.metadata_schedule is expected_decode_schedule,
                "paired decode schedule mismatch",
            ),
            (
                solution.activation_addressing
                is GroupedActivationAddressing.AggregateRowsTiledLinear,
                "paired activation addressing mismatch",
            ),
            (
                solution.output_store is GroupedOutputStore.BFloat16RNEClause4Masked,
                "paired output store mismatch",
            ),
            (
                solution.signed_weight and solution.signed_activation,
                "paired dot operands must be signed",
            ),
            (not solution.wmma_clamp, "paired signed WMMA must not clamp"),
        )
        rejection = next(
            (message for accepted, message in checks if not accepted), None
        )
        if rejection is not None:
            raise ValueError(rejection)
        return cls(
            quant_type=problem.quant_data_type,
            block_values=quant_format.block_values,
            activation_layout=quant_format.activation_layout,
            activation_block_bytes=quant_format.activation_block_bytes,
            packed_weight_block_bytes=quant_format.block_bytes,
            kernel_language=solution.kernel_language,
            isa=solution.isa,
            wavefront_size=solution.wavefront_size,
            projection_count=solution.projection_count,
            arithmetic_contract=quant_format.arithmetic_contract,
            route_ownership=route_ownership,
        )


@dataclass(frozen=True)
class GroupedForwardPairGeometry:
    work_group: tuple[int, int, int]
    matrix_instruction: tuple[int, int, int, int]
    macro_tile: tuple[int, int]
    depth_u: int


@dataclass(frozen=True)
class GroupedForwardPairKernelSpec:
    geometry: GroupedForwardPairGeometry
    operand_source: GroupedPairOperandSource
    activation_addressing: GroupedActivationAddressing
    output_store: GroupedOutputStore

    @classmethod
    def from_solution(
        cls, solution: GroupedForwardPairSolution
    ) -> "GroupedForwardPairKernelSpec":
        return cls(
            geometry=GroupedForwardPairGeometry(
                solution.work_group,
                (
                    solution.matrix_instruction[0],
                    solution.matrix_instruction[1],
                    solution.matrix_instruction[2],
                    solution.matrix_instruction[3],
                ),
                (solution.macro_tile0, solution.macro_tile1),
                solution.depth_u,
            ),
            operand_source=solution.operand_source,
            activation_addressing=solution.activation_addressing,
            output_store=solution.output_store,
        )


@dataclass(frozen=True)
class GroupedForwardPairRouteState:
    physical_experts: int
    output_features: int
    aggregate_rows: int
    blocks_per_weight_row: int
    bytes_per_expert: int
    projection_count: int


@dataclass(frozen=True)
class DerivedGroupedForwardPairState:
    key: GroupedForwardPairSolutionKey
    semantics: QuantForwardSemantics
    physical_plan: GroupedIQ2SPairPhysicalPlan | GroupedQ3KPairPhysicalPlan
    contract: GroupedForwardPairContract
    kernel_spec: GroupedForwardPairKernelSpec
    route: GroupedForwardPairRouteState
    problem_size: ProblemSize
    blocks_per_weight_row: int
    activation_blocks_per_row: int
    packed_weight_row_bytes: int
    bytes_per_expert: int
    activation_plane_stride_bytes: int
    output_column_tiles: int

    @classmethod
    def from_solution_key(
        cls, key: GroupedForwardPairSolutionKey
    ) -> "DerivedGroupedForwardPairState":
        problem = key.problem
        solution = key.solution
        contract = GroupedForwardPairContract.from_solution(problem, solution)
        if problem.quant_data_type not in {"IQ2_S", "Q3_K"}:
            raise ValueError("paired research supports IQ2_S and Q3_K")
        if problem.output_features != 512 or problem.input_features != 2048:
            raise ValueError("paired grouped forward requires N512 K2048")
        physical = (
            grouped_iq2_s_pair_physical_plan(solution.route_ownership)
            if problem.quant_data_type == "IQ2_S"
            else grouped_q3_k_pair_physical_plan(solution.route_ownership)
        )
        semantics = QuantForwardSemantics.for_quant_type(problem.quant_data_type)
        blocks_per_weight_row = problem.input_features // contract.block_values
        activation_blocks_per_row = problem.input_features // Q8_1_D4_BLOCK_VALUES
        packed_weight_row_bytes = (
            blocks_per_weight_row * contract.packed_weight_block_bytes
        )
        bytes_per_expert = problem.output_features * packed_weight_row_bytes
        return cls(
            key=key,
            semantics=semantics,
            physical_plan=physical,
            contract=contract,
            kernel_spec=GroupedForwardPairKernelSpec.from_solution(solution),
            route=GroupedForwardPairRouteState(
                physical_experts=problem.physical_experts,
                output_features=problem.output_features,
                aggregate_rows=problem.aggregate_rows,
                blocks_per_weight_row=blocks_per_weight_row,
                bytes_per_expert=bytes_per_expert,
                projection_count=problem.projection_count,
            ),
            problem_size=ProblemSize(
                problem.aggregate_rows, problem.output_features, problem.input_features
            ),
            blocks_per_weight_row=blocks_per_weight_row,
            activation_blocks_per_row=activation_blocks_per_row,
            packed_weight_row_bytes=packed_weight_row_bytes,
            bytes_per_expert=bytes_per_expert,
            activation_plane_stride_bytes=problem.aggregate_rows
            * solution.activation_block_bytes,
            output_column_tiles=problem.output_features // solution.macro_tile1,
        )

    @property
    def expected_packed_weight_shape(self) -> tuple[int, int, int]:
        return (
            self.key.problem.physical_experts,
            self.key.problem.output_features,
            self.packed_weight_row_bytes,
        )

    @property
    def expected_activation_shape(self) -> tuple[int, int, int]:
        return (
            self.activation_blocks_per_row,
            self.key.problem.aggregate_rows,
            self.key.solution.activation_block_bytes,
        )

    @property
    def expected_output_shape(self) -> tuple[int, int]:
        return (self.key.problem.aggregate_rows, self.key.problem.output_features)

    def grid(self, route_entries: int) -> tuple[int, int, int]:
        if self.contract.route_ownership is not GroupedPairRouteOwnership.SerialRoutes:
            raise ValueError("serial route grid requested for a row-task solution")
        if route_entries <= 0 or route_entries > self.key.problem.max_route_entries:
            raise ValueError("paired route entry count is outside the contract")
        return (self.output_column_tiles, route_entries, 1)

    def row_task_capacity(self, route_entries: int) -> int:
        if route_entries <= 0 or route_entries > self.key.problem.max_route_entries:
            raise ValueError("paired route entry count is outside the contract")
        row_tile = self.key.solution.macro_tile0
        return (
            self.key.problem.aggregate_rows + row_tile - 1
        ) // row_tile + route_entries

    def row_task_grid(self, route_entries: int) -> tuple[int, int, int]:
        if (
            self.contract.route_ownership
            is not GroupedPairRouteOwnership.DeviceRowTasks64
        ):
            raise ValueError("row-task grid requested for a serial route solution")
        return (self.output_column_tiles, self.row_task_capacity(route_entries), 1)
