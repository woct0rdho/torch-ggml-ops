"""Typed contract and derived state for fixed-group Q8_0 forward."""

from dataclasses import dataclass

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
    ForwardProblemContract,
    signed_int8_small_m_tiled_kernel_spec,
)
from .model import ProblemSize
from .quant_formats import Q8_1_D4_BLOCK_VALUES, QUANT_FORMATS
from .schema import SchemaError
from .schema import boolean as _boolean
from .schema import enum_value as _enum
from .schema import integer as _integer
from .schema import integer_triple as _integer_triple
from .schema import integer_tuple as _integer_tuple
from .schema import strict_mapping as _mapping
from .schema import string as _string

_U32_MAX = 0xFFFFFFFF


def fixed_forward_problem_rejection_reason(
    problem: FixedForwardProblem,
) -> str | None:
    """Return a formula-backed rejection for the fixed eight-group layout."""
    quant = QUANT_FORMATS.get(problem.quant_data_type)
    if quant is None or problem.quant_data_type != "Q8_0":
        return "fixed forward requires Q8_0"
    for name, value in (
        ("tokens", problem.tokens),
        ("output features", problem.output_features),
        ("input features", problem.input_features),
    ):
        if not 0 < value <= _U32_MAX:
            return f"fixed forward {name} must fit in a positive u32"
    if problem.groups != 8:
        return "fixed forward requires eight groups"
    if problem.tokens % 64:
        return "fixed forward tokens must be divisible by the 64-row tile"
    if problem.output_features % 64:
        return "fixed forward output features must be divisible by the 64-column tile"
    if problem.input_features % quant.block_values:
        return "fixed forward input features must contain complete quant blocks"
    if problem.input_features % Q8_1_D4_BLOCK_VALUES:
        return "fixed forward input features must contain complete activation blocks"

    activation_blocks = problem.input_features // Q8_1_D4_BLOCK_VALUES
    activation_bytes = (
        activation_blocks * problem.total_activation_rows * quant.activation_block_bytes
    )
    output_bytes = problem.tokens * problem.groups * problem.output_features * 2
    if problem.bytes_per_group > _U32_MAX:
        return "fixed forward packed bytes per group must fit in a u32 offset"
    if activation_bytes > _U32_MAX:
        return "fixed forward activation workspace must fit in a u32 offset"
    if output_bytes > _U32_MAX:
        return "fixed forward output workspace must fit in a u32 offset"
    return None


@dataclass(frozen=True)
class FixedForwardProblemContract:
    """Non-tunable fixed-group data, arithmetic, and ABI contract."""

    quant_type: str
    output_features: int
    input_features: int
    groups: int
    block_values: int
    packed_weight_block_bytes: int
    activation_layout: str
    activation_block_bytes: int
    arithmetic_contract: str
    kernel_language: str
    isa: tuple[int, int, int]
    wavefront_size: int
    weight_decode: str
    activation_addressing: str
    scale_arithmetic: str
    signed_weight: bool
    signed_activation: bool
    wmma_clamp: bool
    destination_type: str = "BFloat16"
    bf16_rounding: str = "RNEPreserveNaN"
    abi: str = "FixedGroupedQ8OutputV1"

    @classmethod
    def rejection_reason(
        cls,
        problem: FixedForwardProblem,
        solution: FixedForwardSolution,
    ) -> str | None:
        quant = QUANT_FORMATS.get(problem.quant_data_type)
        if quant is None:
            return f"unsupported fixed forward quant type {problem.quant_data_type!r}"
        problem_rejection = fixed_forward_problem_rejection_reason(problem)
        if problem_rejection is not None:
            return problem_rejection
        checks = (
            (problem.quant_data_type == "Q8_0", "fixed forward requires Q8_0"),
            (problem.groups == 8, "fixed forward requires eight groups"),
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
                solution.lds_address_hoist
                in ("SmallMTile", "CompactDepth32WeightRows"),
                "unsupported fixed Q8 LDS addressing policy",
            ),
            (
                solution.fixed_address_hoist
                in ("None", "ReductionLoop", "ReductionLoopAndWeightStage"),
                "unsupported fixed Q8 address hoist",
            ),
            (
                solution.fixed_address_hoist == "None"
                or solution.lds_address_hoist == "CompactDepth32WeightRows",
                "fixed Q8 reduction-loop hoist requires compact DepthU32 LDS",
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
        return next((message for accepted, message in checks if not accepted), None)

    @classmethod
    def from_problem(
        cls,
        problem: FixedForwardProblem,
        solution: FixedForwardSolution,
    ) -> "FixedForwardProblemContract":
        quant = QUANT_FORMATS.get(problem.quant_data_type)
        if quant is None:
            raise ValueError(
                f"unsupported fixed forward quant type {problem.quant_data_type!r}"
            ) from None
        rejection = cls.rejection_reason(problem, solution)
        if rejection is not None:
            raise ValueError(rejection)
        return cls(
            quant_type=problem.quant_data_type,
            output_features=problem.output_features,
            input_features=problem.input_features,
            groups=problem.groups,
            block_values=quant.block_values,
            packed_weight_block_bytes=quant.block_bytes,
            activation_layout=quant.activation_layout,
            activation_block_bytes=quant.activation_block_bytes,
            arithmetic_contract=quant.arithmetic_contract,
            kernel_language=solution.kernel_language,
            isa=solution.isa,
            wavefront_size=solution.wavefront_size,
            weight_decode=solution.weight_decode,
            activation_addressing=solution.activation_addressing,
            scale_arithmetic=solution.scale_arithmetic,
            signed_weight=solution.signed_weight,
            signed_activation=solution.signed_activation,
            wmma_clamp=solution.wmma_clamp,
        )

    @classmethod
    def from_mapping(cls, value: object) -> "FixedForwardProblemContract":
        item = _mapping(
            value,
            "FixedForwardProblemContract",
            frozenset(
                {
                    "quant_type",
                    "output_features",
                    "input_features",
                    "groups",
                    "block_values",
                    "packed_weight_block_bytes",
                    "activation_layout",
                    "activation_block_bytes",
                    "arithmetic_contract",
                    "kernel_language",
                    "isa",
                    "wavefront_size",
                    "weight_decode",
                    "activation_addressing",
                    "scale_arithmetic",
                    "signed_weight",
                    "signed_activation",
                    "wmma_clamp",
                    "destination_type",
                    "bf16_rounding",
                    "abi",
                }
            ),
        )
        quant_type = _string(item["quant_type"], "quant_type")
        quant = QUANT_FORMATS.get(quant_type)
        if quant is None:
            raise SchemaError(f"unsupported fixed forward quant type {quant_type!r}")
        contract = cls(
            quant_type=quant_type,
            output_features=_integer(item["output_features"], "output_features"),
            input_features=_integer(item["input_features"], "input_features"),
            groups=_integer(item["groups"], "groups"),
            block_values=_integer(item["block_values"], "block_values"),
            packed_weight_block_bytes=_integer(
                item["packed_weight_block_bytes"], "packed_weight_block_bytes"
            ),
            activation_layout=_string(item["activation_layout"], "activation_layout"),
            activation_block_bytes=_integer(
                item["activation_block_bytes"], "activation_block_bytes"
            ),
            arithmetic_contract=_string(
                item["arithmetic_contract"], "arithmetic_contract"
            ),
            kernel_language=_string(item["kernel_language"], "kernel_language"),
            isa=_integer_triple(item["isa"], "isa"),
            wavefront_size=_integer(item["wavefront_size"], "wavefront_size"),
            weight_decode=_string(item["weight_decode"], "weight_decode"),
            activation_addressing=_string(
                item["activation_addressing"], "activation_addressing"
            ),
            scale_arithmetic=_string(item["scale_arithmetic"], "scale_arithmetic"),
            signed_weight=_boolean(item["signed_weight"], "signed_weight"),
            signed_activation=_boolean(item["signed_activation"], "signed_activation"),
            wmma_clamp=_boolean(item["wmma_clamp"], "wmma_clamp"),
            destination_type=_string(item["destination_type"], "destination_type"),
            bf16_rounding=_string(item["bf16_rounding"], "bf16_rounding"),
            abi=_string(item["abi"], "abi"),
        )
        contract_problem_rejection = fixed_forward_problem_rejection_reason(
            contract.problem(64)
        )
        if contract_problem_rejection is not None:
            raise SchemaError(contract_problem_rejection)
        fixed = (
            contract.quant_type == "Q8_0"
            and contract.groups == 8
            and contract.block_values == quant.block_values
            and contract.packed_weight_block_bytes == quant.block_bytes
            and contract.activation_layout == quant.activation_layout
            and contract.activation_block_bytes == quant.activation_block_bytes
            and contract.arithmetic_contract == quant.arithmetic_contract
            and contract.kernel_language == "Assembly"
            and contract.isa == (11, 5, 1)
            and contract.wavefront_size == 32
            and contract.weight_decode == "DirectSignedInt8"
            and contract.activation_addressing == "FixedGroupRows"
            and contract.scale_arithmetic == "Int32ScaleF32"
            and contract.signed_weight
            and contract.signed_activation
            and not contract.wmma_clamp
            and contract.destination_type == "BFloat16"
            and contract.bf16_rounding == "RNEPreserveNaN"
            and contract.abi == "FixedGroupedQ8OutputV1"
        )
        if not fixed:
            raise SchemaError("FixedForwardProblemContract is not canonical")
        return contract

    def to_mapping(self) -> dict[str, object]:
        return {
            "quant_type": self.quant_type,
            "output_features": self.output_features,
            "input_features": self.input_features,
            "groups": self.groups,
            "block_values": self.block_values,
            "packed_weight_block_bytes": self.packed_weight_block_bytes,
            "activation_layout": self.activation_layout,
            "activation_block_bytes": self.activation_block_bytes,
            "arithmetic_contract": self.arithmetic_contract,
            "kernel_language": self.kernel_language,
            "isa": list(self.isa),
            "wavefront_size": self.wavefront_size,
            "weight_decode": self.weight_decode,
            "activation_addressing": self.activation_addressing,
            "scale_arithmetic": self.scale_arithmetic,
            "signed_weight": self.signed_weight,
            "signed_activation": self.signed_activation,
            "wmma_clamp": self.wmma_clamp,
            "destination_type": self.destination_type,
            "bf16_rounding": self.bf16_rounding,
            "abi": self.abi,
        }

    def problem(self, tokens: int) -> FixedForwardProblem:
        return FixedForwardProblem(
            self.quant_type,
            tokens,
            self.output_features,
            self.input_features,
            self.groups,
        )

    def ordinary(self) -> ForwardProblemContract:
        return ForwardProblemContract(
            quant_type=self.quant_type,
            block_values=self.block_values,
            packed_weight_block_bytes=self.packed_weight_block_bytes,
            activation_layout=self.activation_layout,
            activation_block_bytes=self.activation_block_bytes,
            kernel_language=self.kernel_language,
            isa=self.isa,
            wavefront_size=self.wavefront_size,
            signed_weight=self.signed_weight,
            signed_activation=self.signed_activation,
            wmma_clamp=self.wmma_clamp,
            weight_decode=self.weight_decode,
            scale_arithmetic=self.scale_arithmetic,
            arithmetic_contract=self.arithmetic_contract,
            destination_type=self.destination_type,
            bf16_rounding=self.bf16_rounding,
        )


@dataclass(frozen=True)
class FixedForwardKernelSpec:
    work_group: tuple[int, int, int]
    matrix_instruction: tuple[int, ...]
    macro_tile_tokens: int
    macro_tile_features: int
    depth_u: int
    operand_source: FixedForwardOperandSource
    lds_address_hoist: str
    fixed_address_hoist: str
    output_store: str

    @classmethod
    def from_solution(cls, solution: FixedForwardSolution) -> "FixedForwardKernelSpec":
        return cls(
            work_group=solution.work_group,
            matrix_instruction=solution.matrix_instruction,
            macro_tile_tokens=solution.macro_tile_tokens,
            macro_tile_features=solution.macro_tile_features,
            depth_u=solution.depth_u,
            operand_source=solution.operand_source,
            lds_address_hoist=solution.lds_address_hoist,
            fixed_address_hoist=solution.fixed_address_hoist,
            output_store=solution.output_store,
        )

    @classmethod
    def from_mapping(cls, value: object) -> "FixedForwardKernelSpec":
        item = _mapping(
            value,
            "FixedForwardKernelSpec",
            frozenset({"geometry", "lowering", "epilogue"}),
        )
        geometry = _mapping(
            item["geometry"],
            "FixedForwardKernelSpec.geometry",
            frozenset(
                {
                    "work_group",
                    "matrix_instruction",
                    "macro_tile_tokens",
                    "macro_tile_features",
                    "depth_u",
                }
            ),
        )
        lowering = _mapping(
            item["lowering"],
            "FixedForwardKernelSpec.lowering",
            frozenset({"operand_source", "lds_address_hoist", "fixed_address_hoist"}),
        )
        epilogue = _mapping(
            item["epilogue"],
            "FixedForwardKernelSpec.epilogue",
            frozenset({"output_store"}),
        )
        return cls(
            work_group=_integer_triple(geometry["work_group"], "work_group"),
            matrix_instruction=_integer_tuple(
                geometry["matrix_instruction"], "matrix_instruction", 9
            ),
            macro_tile_tokens=_integer(
                geometry["macro_tile_tokens"], "macro_tile_tokens"
            ),
            macro_tile_features=_integer(
                geometry["macro_tile_features"], "macro_tile_features"
            ),
            depth_u=_integer(geometry["depth_u"], "depth_u"),
            operand_source=_enum(
                lowering["operand_source"],
                "operand_source",
                FixedForwardOperandSource,
            ),
            lds_address_hoist=_string(
                lowering["lds_address_hoist"], "lds_address_hoist"
            ),
            fixed_address_hoist=_string(
                lowering["fixed_address_hoist"], "fixed_address_hoist"
            ),
            output_store=_string(epilogue["output_store"], "output_store"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "geometry": {
                "work_group": list(self.work_group),
                "matrix_instruction": list(self.matrix_instruction),
                "macro_tile_tokens": self.macro_tile_tokens,
                "macro_tile_features": self.macro_tile_features,
                "depth_u": self.depth_u,
            },
            "lowering": {
                "operand_source": self.operand_source.value,
                "lds_address_hoist": self.lds_address_hoist,
                "fixed_address_hoist": self.fixed_address_hoist,
            },
            "epilogue": {"output_store": self.output_store},
        }

    def forward_kernel_spec(
        self, contract: FixedForwardProblemContract
    ) -> ForwardKernelSpec:
        return signed_int8_small_m_tiled_kernel_spec(
            work_group=self.work_group,
            matrix_instruction=self.matrix_instruction,
            macro_tile=(self.macro_tile_tokens, self.macro_tile_features),
            depth_u=self.depth_u,
            operand_source=self.operand_source.value,
            activation_addressing=contract.activation_addressing,
            lds_address_hoist=self.lds_address_hoist,
            output_store=self.output_store,
        )

    def to_solution(
        self,
        contract: FixedForwardProblemContract,
    ) -> FixedForwardSolution:
        return FixedForwardSolution(
            kernel_language=contract.kernel_language,
            isa=contract.isa,
            wavefront_size=contract.wavefront_size,
            work_group=self.work_group,
            matrix_instruction=self.matrix_instruction,
            macro_tile_tokens=self.macro_tile_tokens,
            macro_tile_features=self.macro_tile_features,
            depth_u=self.depth_u,
            activation_layout=contract.activation_layout,
            activation_block_bytes=contract.activation_block_bytes,
            packed_weight_block_bytes=contract.packed_weight_block_bytes,
            operand_source=self.operand_source,
            lds_address_hoist=self.lds_address_hoist,
            fixed_address_hoist=self.fixed_address_hoist,
            weight_decode=contract.weight_decode,
            activation_addressing=contract.activation_addressing,
            scale_arithmetic=contract.scale_arithmetic,
            output_store=self.output_store,
            signed_weight=contract.signed_weight,
            signed_activation=contract.signed_activation,
            wmma_clamp=contract.wmma_clamp,
        )


@dataclass(frozen=True)
class DerivedFixedForwardState:
    """All fixed-group values consumed by validation, lowering, and runtime."""

    problem: FixedForwardProblem
    contract: FixedForwardProblemContract
    ordinary: DerivedForwardState
    physical: FixedQ8ForwardPhysicalPlan
    grid: tuple[int, int, int]

    @classmethod
    def from_solution_key(
        cls, key: FixedForwardSolutionKey
    ) -> "DerivedFixedForwardState":
        contract = FixedForwardProblemContract.from_problem(key.problem, key.solution)
        fixed_spec = FixedForwardKernelSpec.from_solution(key.solution)
        kernel_spec = fixed_spec.forward_kernel_spec(contract)
        ordinary_state = DerivedForwardState.from_contract_spec(
            ProblemSize(
                key.problem.tokens,
                key.problem.output_features,
                key.problem.input_features,
            ),
            contract.ordinary(),
            kernel_spec,
            activation_rows=key.problem.total_activation_rows,
        )
        fixed_plan = fixed_q8_forward_physical_plan(
            kernel_spec, fixed_spec.fixed_address_hoist
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
            contract=contract,
            ordinary=ordinary_state,
            physical=fixed_plan,
            grid=(
                key.problem.output_features // solution.macro_tile_features,
                key.problem.tokens // solution.macro_tile_tokens,
                key.problem.groups,
            ),
        )

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
            self.ordinary.activation_blocks_per_row,
            self.problem.total_activation_rows,
            self.contract.activation_block_bytes,
        )

    @property
    def expected_output_shape(self) -> tuple[int, int, int]:
        return (self.problem.tokens, self.problem.groups, self.problem.output_features)
