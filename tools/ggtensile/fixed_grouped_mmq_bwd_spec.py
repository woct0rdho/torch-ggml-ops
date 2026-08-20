"""Typed contract and derived state for fixed-group Q8_0 backward."""

from dataclasses import dataclass
from typing import ClassVar

from .fixed_grouped_mmq_bwd_model import (
    FixedBackwardProblem,
    FixedBackwardSolution,
    FixedBackwardSolutionKey,
)
from .fixed_grouped_mmq_bwd_physical import (
    FixedBackwardPhysicalPlan,
    derive_fixed_backward_physical_plan,
)
from .mmq_bwd_physical import derive_backward_physical_plan
from .mmq_bwd_spec import (
    BackwardKernelSpec,
    BackwardProblemContract,
    DerivedBackwardState,
    backward_mechanism_contract,
)
from .model import ProblemSize, ProblemType, SolutionKey
from .quant_formats import BACKWARD_QUANT_FORMATS
from .schema import SchemaError
from .schema import integer as _integer
from .schema import integer_tuple as _integer_tuple
from .schema import strict_mapping as _mapping
from .schema import string as _string

_PRODUCTION_TOKENS = frozenset({2048, 8192, 32768})
_U32_MAX = 0xFFFFFFFF


def fixed_backward_problem_rejection_reason(
    problem: FixedBackwardProblem,
) -> str | None:
    if problem.quant_data_type != "Q8_0":
        return "fixed backward requires Q8_0"
    if problem.tokens not in _PRODUCTION_TOKENS:
        return "fixed backward tokens must be one of 2048, 8192, or 32768"
    if problem.output_features != 1024 or problem.input_features != 4096:
        return "fixed backward requires the DeepSeek 1024x4096 projection"
    if problem.groups != 8:
        return "fixed backward requires eight groups"
    quant = BACKWARD_QUANT_FORMATS["Q8_0"]
    if problem.input_features % quant.block_values:
        return "fixed backward input features require complete Q8_0 blocks"
    byte_counts = (
        problem.bytes_per_group,
        problem.tokens * problem.groups * problem.output_features * 2,
        problem.tokens * problem.groups * problem.input_features * 2,
    )
    if any(value > _U32_MAX for value in byte_counts):
        return "fixed backward workspaces must use u32 vector offsets"
    return None


@dataclass(frozen=True)
class FixedBackwardProblemContract:
    quant_type: str = "Q8_0"
    output_features: int = 1024
    input_features: int = 4096
    groups: int = 8

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "quant_type",
            "output_features",
            "input_features",
            "groups",
            "block_values",
            "packed_weight_block_bytes",
            "kernel_language",
            "isa",
            "wavefront_size",
            "activation_type",
            "destination_type",
            "compute_type",
            "row_layout",
            "packed_weight_layout",
            "abi",
            "code_object_version",
        }
    )

    @classmethod
    def from_problem(
        cls, problem: FixedBackwardProblem, solution: FixedBackwardSolution
    ) -> "FixedBackwardProblemContract":
        rejection = fixed_backward_problem_rejection_reason(problem)
        if rejection is not None:
            raise ValueError(rejection)
        if (
            solution.group_axis != "WorkGroupZ"
            or solution.row_layout != "TokenMajorGroupInterleaved"
            or solution.packed_weight_layout != "GroupMajorRows"
        ):
            raise ValueError("fixed backward addressing policy is not canonical")
        return cls(
            problem.quant_data_type,
            problem.output_features,
            problem.input_features,
            problem.groups,
        )

    @classmethod
    def from_mapping(cls, value: object) -> "FixedBackwardProblemContract":
        item = _mapping(value, "FixedBackwardProblemContract", cls._KEYS)
        contract = cls(
            _string(item["quant_type"], "quant_type"),
            _integer(item["output_features"], "output_features"),
            _integer(item["input_features"], "input_features"),
            _integer(item["groups"], "groups"),
        )
        actual = dict(item)
        actual["isa"] = list(_integer_tuple(item["isa"], "isa", 3))
        for name in cls._KEYS - {"isa"}:
            if name in {
                "quant_type",
                "kernel_language",
                "activation_type",
                "destination_type",
                "compute_type",
                "row_layout",
                "packed_weight_layout",
                "abi",
            }:
                actual[name] = _string(item[name], name)
            else:
                actual[name] = _integer(item[name], name)
        if actual != contract.to_mapping():
            raise SchemaError("FixedBackwardProblemContract is not canonical")
        if fixed_backward_problem_rejection_reason(contract.problem(2048)):
            raise SchemaError("fixed backward problem contract is invalid")
        return contract

    def to_mapping(self) -> dict[str, object]:
        quant = BACKWARD_QUANT_FORMATS["Q8_0"]
        return {
            "quant_type": self.quant_type,
            "output_features": self.output_features,
            "input_features": self.input_features,
            "groups": self.groups,
            "block_values": quant.block_values,
            "packed_weight_block_bytes": quant.block_bytes,
            "kernel_language": "Assembly",
            "isa": [11, 5, 1],
            "wavefront_size": 32,
            "activation_type": "BFloat16",
            "destination_type": "BFloat16",
            "compute_type": "Float32",
            "row_layout": "TokenMajorGroupInterleaved",
            "packed_weight_layout": "GroupMajorRows",
            "abi": "FixedGroupedBackwardQ8V1",
            "code_object_version": 5,
        }

    def problem(self, tokens: int) -> FixedBackwardProblem:
        return FixedBackwardProblem(
            self.quant_type,
            tokens,
            self.output_features,
            self.input_features,
            self.groups,
        )

    def ordinary(self, tokens: int) -> BackwardProblemContract:
        quant = BACKWARD_QUANT_FORMATS[self.quant_type]
        return BackwardProblemContract(
            ProblemSize(tokens, self.input_features, self.output_features),
            self.quant_type,
            quant,
            backward_mechanism_contract(self.quant_type),
        )


@dataclass(frozen=True)
class FixedBackwardKernelSpec:
    compute: BackwardKernelSpec
    work_group_order: str
    store_schedule: str
    group_axis: str
    row_layout: str
    packed_weight_layout: str

    @classmethod
    def from_solution(cls, solution: FixedBackwardSolution):
        return cls(
            BackwardKernelSpec.from_solution(solution.compute),
            solution.work_group_order,
            solution.store_schedule,
            solution.group_axis,
            solution.row_layout,
            solution.packed_weight_layout,
        )

    @classmethod
    def from_mapping(cls, value: object):
        item = _mapping(
            value,
            "FixedBackwardKernelSpec",
            frozenset({"compute", "addressing", "epilogue"}),
        )
        addressing = _mapping(
            item["addressing"],
            "addressing",
            frozenset(
                {
                    "work_group_order",
                    "group_axis",
                    "row_layout",
                    "packed_weight_layout",
                }
            ),
        )
        epilogue = _mapping(item["epilogue"], "epilogue", frozenset({"store_schedule"}))
        return cls(
            BackwardKernelSpec.from_mapping(item["compute"], "Q8_0"),
            _string(addressing["work_group_order"], "work_group_order"),
            _string(epilogue["store_schedule"], "store_schedule"),
            _string(addressing["group_axis"], "group_axis"),
            _string(addressing["row_layout"], "row_layout"),
            _string(addressing["packed_weight_layout"], "packed_weight_layout"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "compute": self.compute.to_mapping("Q8_0"),
            "epilogue": {"store_schedule": self.store_schedule},
            "addressing": {
                "work_group_order": self.work_group_order,
                "group_axis": self.group_axis,
                "row_layout": self.row_layout,
                "packed_weight_layout": self.packed_weight_layout,
            },
        }

    def to_solution(
        self, contract: FixedBackwardProblemContract, tokens: int
    ) -> FixedBackwardSolution:
        return FixedBackwardSolution(
            self.compute.to_solution(contract.ordinary(tokens)),
            self.work_group_order,
            self.store_schedule,
            self.group_axis,
            self.row_layout,
            self.packed_weight_layout,
        )


@dataclass(frozen=True)
class DerivedFixedBackwardState:
    problem: FixedBackwardProblem
    contract: FixedBackwardProblemContract
    spec: FixedBackwardKernelSpec
    ordinary: DerivedBackwardState
    physical: FixedBackwardPhysicalPlan

    @classmethod
    def from_solution_key(cls, key: FixedBackwardSolutionKey):
        from .fixed_grouped_mmq_bwd_validation import (
            validate_fixed_backward_solution_key,
        )

        validate_fixed_backward_solution_key(key)
        contract = FixedBackwardProblemContract.from_problem(key.problem, key.solution)
        spec = FixedBackwardKernelSpec.from_solution(key.solution)
        ordinary_key = SolutionKey(
            ProblemType.mmq_backward("Q8_0"),
            ProblemSize(
                key.problem.tokens,
                key.problem.input_features,
                key.problem.output_features,
            ),
            key.solution.compute,
        )
        ordinary = DerivedBackwardState.from_solution_key(ordinary_key)
        physical = derive_fixed_backward_physical_plan(
            derive_backward_physical_plan(ordinary)
        )
        return cls(key.problem, contract, spec, ordinary, physical)

    @property
    def grid(self) -> tuple[int, int, int]:
        geometry = self.spec.compute.geometry
        m_tiles = self.problem.tokens // geometry.macro_tile0
        n_tiles = self.problem.input_features // geometry.macro_tile1
        if self.spec.work_group_order == "NMajor":
            return (n_tiles, m_tiles, self.problem.groups)
        return (
            m_tiles,
            n_tiles,
            self.problem.groups,
        )

    @property
    def expected_grad_output_shape(self) -> tuple[int, int, int]:
        return (self.problem.tokens, self.problem.groups, self.problem.output_features)

    @property
    def expected_packed_weight_shape(self) -> tuple[int, int, int]:
        return (
            self.problem.groups,
            self.problem.output_features,
            self.problem.packed_row_bytes,
        )

    @property
    def expected_grad_input_shape(self) -> tuple[int, int, int]:
        return (self.problem.tokens, self.problem.groups, self.problem.input_features)
