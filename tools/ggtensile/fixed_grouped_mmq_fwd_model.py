"""Strict identities for the fixed-group Q8_0 forward experiment."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar, TypeVar

from typing_extensions import Self

from .model import ForwardSolution, ProblemSize, ProblemType, SchemaError, SolutionKey
from .quant_formats import Q8_1_F32_D4_BLOCK_BYTES, QUANT_FORMATS


def _mapping(value: object, name: str, keys: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise SchemaError(f"{name} keys must be strings")
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise SchemaError(f"invalid {name}: {', '.join(details)}")
    return value


def _string(value: object, name: str) -> str:
    if type(value) is not str:
        raise SchemaError(f"{name} must be str, not {type(value).__name__}")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise SchemaError(f"{name} must be int, not {type(value).__name__}")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise SchemaError(f"{name} must be bool, not {type(value).__name__}")
    return value


def _integer_tuple(value: object, name: str, length: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise SchemaError(f"{name} must be a {length}-element list")
    return tuple(_integer(item, f"{name}[{index}]") for index, item in enumerate(value))


def _integer_triple(value: object, name: str) -> tuple[int, int, int]:
    values = _integer_tuple(value, name, 3)
    return values[0], values[1], values[2]


_EnumT = TypeVar("_EnumT", bound=Enum)


def _enum(value: object, name: str, enum_type: type[_EnumT]) -> _EnumT:
    serialized = _string(value, name)
    try:
        return enum_type(serialized)
    except ValueError:
        raise SchemaError(f"{name} has unsupported value {serialized!r}") from None


class FixedForwardOperandSource(str, Enum):
    """Physical Q8_0 dataflow families owned by this experiment."""

    Q8SmallMTiledLds = "Q8SmallMTiledLds"


@dataclass(frozen=True)
class FixedForwardProblem:
    """One exact fixed-group problem, including its non-routed group axis."""

    quant_data_type: str
    tokens: int
    output_features: int
    input_features: int
    groups: int

    @classmethod
    def deepseek_q8_0(cls, tokens: int) -> Self:
        return cls("Q8_0", tokens, 1024, 4096, 8)

    @property
    def packed_row_bytes(self) -> int:
        quant = QUANT_FORMATS[self.quant_data_type]
        if self.input_features % quant.block_values:
            raise ValueError("fixed input features are not block divisible")
        return self.input_features // quant.block_values * quant.block_bytes

    @property
    def bytes_per_group(self) -> int:
        return self.output_features * self.packed_row_bytes

    @property
    def total_activation_rows(self) -> int:
        return self.tokens * self.groups


@dataclass(frozen=True)
class FixedForwardSolution:
    """Complete mechanism identity for one fixed-group forward lowering."""

    kernel_language: str
    isa: tuple[int, int, int]
    wavefront_size: int
    work_group: tuple[int, int, int]
    matrix_instruction: tuple[int, ...]
    macro_tile_tokens: int
    macro_tile_features: int
    depth_u: int
    activation_layout: str
    activation_block_bytes: int
    packed_weight_block_bytes: int
    operand_source: FixedForwardOperandSource
    lds_address_hoist: str
    fixed_address_hoist: str
    weight_decode: str
    activation_addressing: str
    scale_arithmetic: str
    output_store: str
    signed_weight: bool
    signed_activation: bool
    wmma_clamp: bool

    @classmethod
    def q8_0_small_m_tiled_lds(cls) -> Self:
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(32, 4, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile_tokens=64,
            macro_tile_features=64,
            depth_u=32,
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            packed_weight_block_bytes=QUANT_FORMATS["Q8_0"].block_bytes,
            operand_source=FixedForwardOperandSource.Q8SmallMTiledLds,
            lds_address_hoist="SmallMTile",
            fixed_address_hoist="None",
            weight_decode="DirectSignedInt8",
            activation_addressing="FixedGroupRows",
            scale_arithmetic="Int32ScaleF32",
            output_store="BFloat16RNEClauseTile",
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=False,
        )

    @classmethod
    def q8_0_compact_depth32_tiled_lds(cls) -> Self:
        return replace(
            cls.q8_0_small_m_tiled_lds(),
            lds_address_hoist="CompactDepth32WeightRows",
        )

    @classmethod
    def q8_0_compact_depth32_tiled_lds_hoisted(cls) -> Self:
        return replace(
            cls.q8_0_compact_depth32_tiled_lds(),
            fixed_address_hoist="ReductionLoop",
        )

    @classmethod
    def q8_0_compact_depth32_tiled_lds_weight_hoisted(cls) -> Self:
        return replace(
            cls.q8_0_compact_depth32_tiled_lds(),
            fixed_address_hoist="ReductionLoopAndWeightStage",
        )

    @property
    def num_threads(self) -> int:
        return self.work_group[0] * self.work_group[1] * self.work_group[2]


@dataclass(frozen=True)
class FixedForwardSolutionKey:
    problem: FixedForwardProblem
    solution: FixedForwardSolution

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {"ArtifactKind", "KernelFamily", "ProblemContract", "Problem", "KernelSpec"}
    )

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _mapping(value, "FixedForwardSolutionKey", cls._KEYS)
        if item["ArtifactKind"] != "ExactKernel":
            raise SchemaError("fixed key ArtifactKind must be ExactKernel")
        if item["KernelFamily"] != "FixedGroupedForward":
            raise SchemaError("fixed key KernelFamily must be FixedGroupedForward")
        from .fixed_grouped_mmq_fwd_spec import (
            FixedForwardKernelSpec,
            FixedForwardProblemContract,
        )

        contract = FixedForwardProblemContract.from_mapping(item["ProblemContract"])
        problem_item = _mapping(
            item["Problem"], "FixedForwardProblem", frozenset({"tokens"})
        )
        problem = contract.problem(_integer(problem_item["tokens"], "tokens"))
        spec = FixedForwardKernelSpec.from_mapping(item["KernelSpec"])
        solution = spec.to_solution(contract)
        if FixedForwardProblemContract.from_problem(problem, solution) != contract:
            raise SchemaError(
                "FixedForwardProblemContract does not round-trip canonically"
            )
        from .fixed_grouped_mmq_fwd_validation import fixed_forward_rejection_reason

        rejection = fixed_forward_rejection_reason(cls(problem, solution))
        if rejection is not None:
            raise SchemaError(
                f"fixed kernel specification is not canonical: {rejection}"
            )
        return cls(problem, solution)

    def to_mapping(self) -> dict[str, object]:
        from .fixed_grouped_mmq_fwd_spec import (
            FixedForwardKernelSpec,
            FixedForwardProblemContract,
        )

        contract = FixedForwardProblemContract.from_problem(self.problem, self.solution)
        spec = FixedForwardKernelSpec.from_solution(self.solution)
        return {
            "ArtifactKind": "ExactKernel",
            "KernelFamily": "FixedGroupedForward",
            "ProblemContract": contract.to_mapping(),
            "Problem": {"tokens": self.problem.tokens},
            "KernelSpec": spec.to_mapping(),
        }

    def to_standard_solution_key(self) -> SolutionKey:
        """Project the compatible Q8 mechanics onto the ordinary typed state."""
        base = ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=64)
        solution = replace(
            base,
            isa=self.solution.isa,
            wavefront_size=self.solution.wavefront_size,
            work_group=self.solution.work_group,
            matrix_instruction=self.solution.matrix_instruction,
            macro_tile0=self.solution.macro_tile_tokens,
            macro_tile1=self.solution.macro_tile_features,
            depth_u=self.solution.depth_u,
            activation_layout=self.solution.activation_layout,
            activation_block_bytes=self.solution.activation_block_bytes,
            packed_weight_block_bytes=self.solution.packed_weight_block_bytes,
            lds_address_hoist=self.solution.lds_address_hoist,
            output_store=self.solution.output_store,
            signed_weight=self.solution.signed_weight,
            signed_activation=self.solution.signed_activation,
            wmma_clamp=self.solution.wmma_clamp,
        )
        return SolutionKey(
            ProblemType.mmq_forward(self.problem.quant_data_type),
            ProblemSize(
                self.problem.tokens,
                self.problem.output_features,
                self.problem.input_features,
            ),
            solution,
        )

    @property
    def hash(self) -> str:
        canonical = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))
        return f"ggsol_{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"

    @property
    def kernel_name(self) -> str:
        problem = self.problem
        return (
            "torch_ggml_ops_ggtensile_gfx1151_v1_fixed_grouped_mmq_fwd_q8_0_"
            f"t{problem.tokens}_n{problem.output_features}_k{problem.input_features}_"
            f"{self.hash[6:]}"
        )
