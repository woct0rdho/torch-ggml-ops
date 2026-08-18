"""Strict problem and solution state for routed MMQ backward kernels."""

from dataclasses import dataclass
from enum import Enum

from .mmq_bwd_spec import (
    BackwardKernelSpec,
    BackwardProblemContract,
    DerivedBackwardState,
    backward_mechanism_contract,
)
from .model import (
    GroupedBackwardSolution,
    ProblemSize,
    ProblemType,
    SchemaError,
    SolutionKey,
    _integer,
    _strict_mapping,
    _strict_mapping_optional,
    _string,
)
from .quant_formats import BACKWARD_QUANT_FORMATS, QuantFormat

GROUPED_BACKWARD_QWEN_ROWS = frozenset({16_384, 65_536, 262_144})
GROUPED_BACKWARD_DEEPSEEK_ROWS = frozenset({12_288, 49_152, 196_608})


class GroupedBackwardOwnership(str, Enum):
    SerialRoutes = "SerialRoutes"
    SplitRoutes2 = "SplitRoutes2"
    SplitRoutes4 = "SplitRoutes4"
    SplitRoutes8 = "SplitRoutes8"
    SplitRoutes16 = "SplitRoutes16"

    @property
    def split_factor(self) -> int:
        return {
            GroupedBackwardOwnership.SerialRoutes: 1,
            GroupedBackwardOwnership.SplitRoutes2: 2,
            GroupedBackwardOwnership.SplitRoutes4: 4,
            GroupedBackwardOwnership.SplitRoutes8: 8,
            GroupedBackwardOwnership.SplitRoutes16: 16,
        }[self]


class GroupedBackwardRowTail(str, Enum):
    Masked = "Masked"
    Mixed128_64 = "Mixed128_64"


@dataclass(frozen=True)
class GroupedBackwardProblemContract:
    problem_size: ProblemSize
    quant_type: str
    quant_format: QuantFormat
    physical_experts: int = 256
    max_route_entries: int = 256
    code_object_version: int = 5

    @classmethod
    def from_solution_key(cls, solution_key: SolutionKey):
        quant_type = solution_key.problem_type.quant_data_type
        if solution_key.problem_type != ProblemType.grouped_mmq_backward(quant_type):
            raise ValueError("invalid grouped backward problem type")
        cls._validate_problem(solution_key.problem_size, quant_type)
        return cls(
            problem_size=solution_key.problem_size,
            quant_type=quant_type,
            quant_format=BACKWARD_QUANT_FORMATS[quant_type],
        )

    @classmethod
    def from_mapping(cls, value: object, problem_size: ProblemSize):
        item = _strict_mapping(
            value,
            name="GroupedBackwardProblemContract",
            keys=frozenset(
                {
                    "quant_type",
                    "block_values",
                    "packed_weight_block_bytes",
                    "kernel_language",
                    "isa",
                    "wavefront_size",
                    "activation_type",
                    "destination_type",
                    "compute_type",
                    "abi",
                    "physical_experts",
                    "max_route_entries",
                    "code_object_version",
                }
            ),
        )
        quant_type = _string(item["quant_type"], "quant_type")
        if quant_type not in {"Q2_K", "Q4_K", "Q5_K"}:
            raise SchemaError(f"unsupported grouped backward quant type {quant_type!r}")
        try:
            expected = cls(
                problem_size,
                quant_type,
                BACKWARD_QUANT_FORMATS[quant_type],
            ).to_mapping()
        except (KeyError, ValueError):
            raise SchemaError(
                f"unsupported grouped backward quant type {quant_type!r}"
            ) from None
        cls._validate_problem(problem_size, quant_type)
        actual = {
            "quant_type": quant_type,
            "block_values": _integer(item["block_values"], "block_values"),
            "packed_weight_block_bytes": _integer(
                item["packed_weight_block_bytes"], "packed_weight_block_bytes"
            ),
            "kernel_language": _string(item["kernel_language"], "kernel_language"),
            "isa": item["isa"],
            "wavefront_size": _integer(item["wavefront_size"], "wavefront_size"),
            "activation_type": _string(item["activation_type"], "activation_type"),
            "destination_type": _string(item["destination_type"], "destination_type"),
            "compute_type": _string(item["compute_type"], "compute_type"),
            "abi": _string(item["abi"], "abi"),
            "physical_experts": _integer(item["physical_experts"], "physical_experts"),
            "max_route_entries": _integer(
                item["max_route_entries"], "max_route_entries"
            ),
            "code_object_version": _integer(
                item["code_object_version"], "code_object_version"
            ),
        }
        if actual != expected:
            raise SchemaError("GroupedBackwardProblemContract is not canonical")
        return cls(problem_size, quant_type, BACKWARD_QUANT_FORMATS[quant_type])

    @staticmethod
    def _validate_problem(problem_size: ProblemSize, quant_type: str) -> None:
        if quant_type == "Q2_K":
            valid = (
                problem_size.m in GROUPED_BACKWARD_DEEPSEEK_ROWS
                and problem_size.n == 2048
                and problem_size.k == 4096
            )
            message = (
                "grouped Q2_K backward requires exact (R,2048,4096) with "
                "R in {12288,49152,196608}"
            )
        else:
            valid = (
                problem_size.m in GROUPED_BACKWARD_QWEN_ROWS
                and problem_size.n == 512
                and problem_size.k == 2048
            )
            message = (
                "grouped Q4_K/Q5_K backward requires exact (R,512,2048) with "
                "R in {16384,65536,262144}"
            )
        if not valid:
            raise SchemaError(message)

    def to_mapping(self) -> dict[str, object]:
        return {
            "quant_type": self.quant_type,
            "block_values": self.quant_format.block_values,
            "packed_weight_block_bytes": self.quant_format.block_bytes,
            "kernel_language": "Assembly",
            "isa": [11, 5, 1],
            "wavefront_size": 32,
            "activation_type": "BFloat16",
            "destination_type": "BFloat16",
            "compute_type": "Float32",
            "abi": "GroupedBackwardSerialRoutesV1",
            "physical_experts": self.physical_experts,
            "max_route_entries": self.max_route_entries,
            "code_object_version": self.code_object_version,
        }

    def ordinary(self) -> BackwardProblemContract:
        return BackwardProblemContract(
            problem_size=self.problem_size,
            quant_type=self.quant_type,
            quant_format=self.quant_format,
            mechanism=backward_mechanism_contract(self.quant_type),
        )


@dataclass(frozen=True)
class GroupedBackwardKernelSpec:
    compute: BackwardKernelSpec
    ownership: GroupedBackwardOwnership
    row_tail: GroupedBackwardRowTail

    @classmethod
    def from_solution(cls, solution: GroupedBackwardSolution):
        try:
            ownership = GroupedBackwardOwnership(solution.route_ownership)
            row_tail = GroupedBackwardRowTail(solution.row_tail)
        except ValueError as error:
            raise SchemaError("invalid grouped backward ownership policy") from error
        return cls(
            BackwardKernelSpec.from_solution(solution.compute), ownership, row_tail
        )

    @classmethod
    def from_mapping(cls, value: object, quant_type: str):
        item = _strict_mapping_optional(
            value,
            name="GroupedBackwardKernelSpec",
            required=frozenset(
                {"geometry", "memory", "pipeline", "store", "ownership"}
            ),
            optional=frozenset({"decode"}),
        )
        ownership_item = _strict_mapping(
            item["ownership"],
            name="GroupedBackwardKernelSpec.ownership",
            keys=frozenset({"kind", "row_tail"}),
        )
        compute_mapping = {
            key: item[key] for key in ("geometry", "memory", "pipeline", "store")
        }
        if "decode" in item:
            compute_mapping["decode"] = item["decode"]
        try:
            ownership = GroupedBackwardOwnership(
                _string(ownership_item["kind"], "ownership.kind")
            )
            row_tail = GroupedBackwardRowTail(
                _string(ownership_item["row_tail"], "ownership.row_tail")
            )
        except ValueError as error:
            raise SchemaError("invalid grouped backward ownership policy") from error
        return cls(
            BackwardKernelSpec.from_mapping(compute_mapping, quant_type),
            ownership,
            row_tail,
        )

    def to_mapping(self, quant_type: str) -> dict[str, object]:
        mapping = self.compute.to_mapping(quant_type)
        mapping["ownership"] = {
            "kind": self.ownership.value,
            "row_tail": self.row_tail.value,
        }
        return mapping

    def to_solution(
        self, contract: GroupedBackwardProblemContract
    ) -> GroupedBackwardSolution:
        return GroupedBackwardSolution(
            compute=self.compute.to_solution(contract.ordinary()),
            route_ownership=self.ownership.value,
            row_tail=self.row_tail.value,
        )


@dataclass(frozen=True)
class DerivedGroupedBackwardState:
    solution_key: SolutionKey
    solution: GroupedBackwardSolution
    contract: GroupedBackwardProblemContract
    spec: GroupedBackwardKernelSpec
    ordinary: DerivedBackwardState

    @classmethod
    def from_solution_key(cls, solution_key: SolutionKey):
        solution = solution_key.solution
        if not isinstance(solution, GroupedBackwardSolution):
            raise TypeError("grouped backward state requires GroupedBackwardSolution")
        contract = GroupedBackwardProblemContract.from_solution_key(solution_key)
        spec = GroupedBackwardKernelSpec.from_solution(solution)
        ordinary_key = SolutionKey(
            ProblemType.mmq_backward(contract.quant_type),
            contract.problem_size,
            solution.compute,
        )
        return cls(
            solution_key=solution_key,
            solution=solution,
            contract=contract,
            spec=spec,
            ordinary=DerivedBackwardState.from_solution_key(ordinary_key),
        )
