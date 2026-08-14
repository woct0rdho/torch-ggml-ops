"""Derived state for isolated grouped MMQ forward kernels."""

from dataclasses import dataclass

from .grouped_mmq_fwd_model import GroupedForwardSolutionKey
from .grouped_mmq_fwd_physical import (
    GroupedDecodedPhysicalPlan,
    GroupedDirectPhysicalPlan,
    grouped_decoded_physical_plan,
    grouped_direct_physical_plan,
)
from .mmq_fwd_spec import QuantForwardSemantics
from .model import ProblemSize
from .quant_formats import GROUPED_QUANT_FORMATS, Q8_1_D4_BLOCK_VALUES


@dataclass(frozen=True)
class GroupedForwardCompatibilityContract:
    quant_type: str
    packed_weight_block_bytes: int
    wmma_clamp: bool


@dataclass(frozen=True)
class GroupedForwardCompatibilityDecode:
    metadata_schedule: str


@dataclass(frozen=True)
class GroupedForwardCompatibilityKernelSpec:
    decode: GroupedForwardCompatibilityDecode


@dataclass(frozen=True)
class DerivedGroupedForwardState:
    key: GroupedForwardSolutionKey
    semantics: QuantForwardSemantics
    physical_plan: GroupedDirectPhysicalPlan | GroupedDecodedPhysicalPlan
    contract: GroupedForwardCompatibilityContract
    kernel_spec: GroupedForwardCompatibilityKernelSpec
    problem_size: ProblemSize
    blocks_per_weight_row: int
    activation_blocks_per_row: int
    packed_weight_row_bytes: int
    bytes_per_expert: int
    activation_plane_stride_bytes: int
    activation_weight_block_stride_bytes: int
    output_column_tiles: int

    @classmethod
    def from_solution_key(
        cls, key: GroupedForwardSolutionKey
    ) -> "DerivedGroupedForwardState":
        problem = key.problem
        solution = key.solution
        quant_format = GROUPED_QUANT_FORMATS[problem.quant_data_type]
        semantics = QuantForwardSemantics.for_quant_type(problem.quant_data_type)
        blocks_per_weight_row = problem.input_features // quant_format.block_values
        activation_blocks_per_row = problem.input_features // Q8_1_D4_BLOCK_VALUES
        packed_weight_row_bytes = blocks_per_weight_row * quant_format.block_bytes
        activation_plane_stride_bytes = (
            problem.aggregate_rows * solution.activation_block_bytes
        )
        if solution.operand_source == "GroupedDirectGlobal":
            physical_plan: GroupedDirectPhysicalPlan | GroupedDecodedPhysicalPlan = (
                grouped_direct_physical_plan(solution.activation_block_bytes)
            )
        elif solution.operand_source == "GroupedDecodedWeightLds":
            physical_plan = grouped_decoded_physical_plan(
                solution.activation_block_bytes,
                solution.macro_tile0,
                problem.quant_data_type,
            )
        else:
            raise ValueError(
                f"unsupported grouped operand source {solution.operand_source!r}"
            )
        return cls(
            key=key,
            semantics=semantics,
            physical_plan=physical_plan,
            contract=GroupedForwardCompatibilityContract(
                problem.quant_data_type,
                solution.packed_weight_block_bytes,
                solution.wmma_clamp,
            ),
            kernel_spec=GroupedForwardCompatibilityKernelSpec(
                GroupedForwardCompatibilityDecode(solution.metadata_schedule)
            ),
            problem_size=ProblemSize(
                problem.aggregate_rows,
                problem.output_features,
                problem.input_features,
            ),
            blocks_per_weight_row=blocks_per_weight_row,
            activation_blocks_per_row=activation_blocks_per_row,
            packed_weight_row_bytes=packed_weight_row_bytes,
            bytes_per_expert=problem.output_features * packed_weight_row_bytes,
            activation_plane_stride_bytes=activation_plane_stride_bytes,
            activation_weight_block_stride_bytes=2 * activation_plane_stride_bytes,
            output_column_tiles=problem.output_features // solution.macro_tile1,
        )

    @property
    def expected_packed_weight_shape(self) -> tuple[int, int, int]:
        problem = self.key.problem
        return (
            problem.physical_experts,
            problem.output_features,
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
        problem = self.key.problem
        return (problem.aggregate_rows, problem.output_features)

    def grid(self, route_entries: int) -> tuple[int, int, int]:
        if route_entries <= 0 or route_entries > self.key.problem.max_route_entries:
            raise ValueError(
                "route entry count is outside the grouped problem contract"
            )
        return (self.output_column_tiles, route_entries, 1)
