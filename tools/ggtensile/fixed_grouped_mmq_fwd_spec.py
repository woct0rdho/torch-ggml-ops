"""Typed contract and derived state for fixed-group Q8_0 forward."""

from dataclasses import dataclass, replace

from .fixed_grouped_mmq_fwd_model import (
    FixedForwardOperandSource,
    FixedForwardProblem,
    FixedForwardSolution,
    FixedForwardSolutionKey,
)
from .fixed_grouped_mmq_fwd_physical import (
    FixedQ8ForwardPhysicalPlan,
    fixed_q8_forward_physical_plan,
)
from .mmq_fwd_spec import (
    DerivedForwardState,
    ForwardKernelSpec,
    ForwardResourceUsage,
)
from .quant_formats import QUANT_FORMATS


@dataclass(frozen=True)
class FixedForwardProblemContract:
    """Non-tunable fixed-group data, arithmetic, and ABI contract."""

    quant_type: str
    groups: int
    block_values: int
    packed_weight_block_bytes: int
    activation_layout: str
    activation_block_bytes: int
    arithmetic_contract: str
    abi: str = "FixedGroupedQ8ForwardV1"

    @classmethod
    def from_problem(
        cls,
        problem: FixedForwardProblem,
        solution: FixedForwardSolution,
    ) -> "FixedForwardProblemContract":
        try:
            quant = QUANT_FORMATS[problem.quant_data_type]
        except KeyError:
            raise ValueError(
                f"unsupported fixed forward quant type {problem.quant_data_type!r}"
            ) from None
        checks = (
            (problem.quant_data_type == "Q8_0", "fixed forward requires Q8_0"),
            (problem.groups == 8, "fixed forward requires eight groups"),
            (problem.input_features == 4096, "fixed forward requires K=4096"),
            (problem.output_features == 1024, "fixed forward requires N=1024"),
            (
                problem.tokens in (2048, 8192, 32768),
                "fixed forward requires a production token count",
            ),
            (
                solution.kernel_language == "Assembly",
                "fixed forward language must be Assembly",
            ),
            (solution.isa == (11, 5, 1), "fixed forward ISA must be gfx1151"),
            (solution.wavefront_size == 32, "fixed forward wavefront must be 32"),
            (
                solution.activation_layout == quant.activation_layout,
                "fixed activation layout does not match Q8_0",
            ),
            (
                solution.activation_block_bytes == quant.activation_block_bytes,
                "fixed activation block bytes do not match Q8_0",
            ),
            (
                solution.packed_weight_block_bytes == quant.block_bytes,
                "fixed packed weight bytes do not match Q8_0",
            ),
            (
                solution.operand_source is FixedForwardOperandSource.Q8SmallMTiledLds,
                "unsupported fixed Q8 physical dataflow",
            ),
            (
                solution.weight_decode == "DirectSignedInt8",
                "fixed Q8 weight decode must be signed int8",
            ),
            (
                solution.scale_arithmetic == "Int32ScaleF32",
                "fixed Q8 scale arithmetic must be integer-WMMA FP32 correction",
            ),
            (
                solution.signed_weight and solution.signed_activation,
                "fixed Q8 operands must be signed",
            ),
            (not solution.wmma_clamp, "fixed Q8 WMMA must not clamp"),
        )
        rejection = next(
            (message for accepted, message in checks if not accepted), None
        )
        if rejection is not None:
            raise ValueError(rejection)
        return cls(
            quant_type=problem.quant_data_type,
            groups=problem.groups,
            block_values=quant.block_values,
            packed_weight_block_bytes=quant.block_bytes,
            activation_layout=quant.activation_layout,
            activation_block_bytes=quant.activation_block_bytes,
            arithmetic_contract=quant.arithmetic_contract,
        )


@dataclass(frozen=True)
class DerivedFixedForwardState:
    """All fixed-group values consumed by validation, lowering, and runtime."""

    problem: FixedForwardProblem
    solution: FixedForwardSolution
    contract: FixedForwardProblemContract
    ordinary_state: DerivedForwardState
    physical_plan: object
    fixed_physical_plan: FixedQ8ForwardPhysicalPlan
    kernel_spec: ForwardKernelSpec
    activation_plane_stride_bytes: int
    grid: tuple[int, int, int]

    @classmethod
    def from_solution_key(
        cls, key: FixedForwardSolutionKey
    ) -> "DerivedFixedForwardState":
        contract = FixedForwardProblemContract.from_problem(key.problem, key.solution)
        ordinary_key = key.to_standard_solution_key()
        ordinary_state = DerivedForwardState.from_solution_key(ordinary_key)
        kernel_spec = ordinary_state.kernel_spec
        fixed_plan = fixed_q8_forward_physical_plan(kernel_spec)
        fixed_stride = (
            key.problem.total_activation_rows * contract.activation_block_bytes
        )
        ordinary_state = replace(
            ordinary_state,
            activation_plane_stride_bytes=fixed_stride,
        )
        solution = key.solution
        for name, value, divisor in (
            ("tokens", key.problem.tokens, solution.macro_tile_tokens),
            (
                "output features",
                key.problem.output_features,
                solution.macro_tile_features,
            ),
            ("input features", key.problem.input_features, 4 * 32),
        ):
            if value <= 0 or divisor <= 0 or value % divisor:
                raise ValueError(
                    f"fixed {name} must be a positive multiple of {divisor}"
                )
        return cls(
            problem=key.problem,
            solution=solution,
            contract=contract,
            ordinary_state=ordinary_state,
            physical_plan=ordinary_state.physical_plan,
            fixed_physical_plan=fixed_plan,
            kernel_spec=kernel_spec,
            activation_plane_stride_bytes=fixed_stride,
            grid=(
                key.problem.output_features // solution.macro_tile_features,
                key.problem.tokens // solution.macro_tile_tokens,
                key.problem.groups,
            ),
        )

    @property
    def problem_size(self):
        return self.ordinary_state.problem_size

    @property
    def semantics(self):
        return self.ordinary_state.semantics

    @property
    def resources(self) -> ForwardResourceUsage:
        return self.fixed_physical_plan.resources

    @property
    def num_threads(self) -> int:
        return self.solution.num_threads

    @property
    def activation_blocks_per_row(self) -> int:
        return self.ordinary_state.activation_blocks_per_row

    @property
    def packed_weight_row_bytes(self) -> int:
        return self.ordinary_state.packed_weight_row_bytes

    @property
    def blocks_per_weight_row(self) -> int:
        return self.ordinary_state.blocks_per_weight_row

    @property
    def expected_packed_weight_shape(self) -> tuple[int, int, int]:
        return (
            self.problem.groups,
            self.problem.output_features,
            self.problem.packed_row_bytes,
        )

    @property
    def expected_activation_shape(self) -> tuple[int, int, int]:
        return (
            self.activation_blocks_per_row,
            self.problem.total_activation_rows,
            self.contract.activation_block_bytes,
        )

    @property
    def expected_output_shape(self) -> tuple[int, int, int]:
        return (self.problem.tokens, self.problem.groups, self.problem.output_features)
