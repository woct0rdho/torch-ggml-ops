"""Strict problem and solution state for routed MMQ backward kernels."""

from dataclasses import dataclass, replace

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
    SolutionKey,
)
from .quant_formats import BACKWARD_QUANT_FORMATS, QuantFormat
from .schema import SchemaError
from .schema import integer as _integer
from .schema import strict_mapping as _strict_mapping
from .schema import strict_mapping_optional as _strict_mapping_optional
from .schema import string as _string

_U32_MAX = 0xFFFFFFFF


@dataclass(frozen=True)
class GroupedBackwardOwnership:
    kind: str
    split_factor: int

    @classmethod
    def from_solution_value(cls, value: str) -> "GroupedBackwardOwnership":
        if type(value) is not str:
            raise SchemaError("grouped backward route ownership must be a string")
        if value == "SerialRoutes":
            return cls("Serial", 1)
        if value.startswith("SplitRoutes"):
            factor = int(value.removeprefix("SplitRoutes"))
            if factor > 1 and factor & (factor - 1) == 0:
                return cls("RouteSplit", factor)
        raise SchemaError("invalid grouped backward route ownership")

    @classmethod
    def from_mapping(cls, value: object) -> "GroupedBackwardOwnership":
        item = _strict_mapping_optional(
            value,
            name="GroupedBackwardKernelSpec.ownership",
            required=frozenset({"kind"}),
            optional=frozenset({"split_factor"}),
        )
        kind = _string(item["kind"], "ownership.kind")
        if kind == "Serial":
            if "split_factor" in item:
                raise SchemaError("serial ownership has no split_factor")
            return cls(kind, 1)
        if kind != "RouteSplit" or "split_factor" not in item:
            raise SchemaError("route split ownership requires split_factor")
        factor = _integer(item["split_factor"], "ownership.split_factor")
        if factor <= 1 or factor & (factor - 1):
            raise SchemaError("ownership.split_factor must be a power of two above one")
        return cls(kind, factor)

    def to_mapping(self) -> dict[str, object]:
        if self.kind == "Serial":
            return {"kind": self.kind}
        return {"kind": self.kind, "split_factor": self.split_factor}

    @property
    def solution_value(self) -> str:
        return (
            "SerialRoutes"
            if self.kind == "Serial"
            else f"SplitRoutes{self.split_factor}"
        )


@dataclass(frozen=True)
class GroupedBackwardRowTail:
    kind: str
    threshold_rows: int | None = None
    tile_rows: int | None = None

    @classmethod
    def from_solution(
        cls, solution: GroupedBackwardSolution
    ) -> "GroupedBackwardRowTail":
        value = solution.row_tail
        threshold = solution.row_tail_threshold_rows
        tile_rows = solution.row_tail_tile_rows
        if value == "Masked":
            if threshold is not None or tile_rows is not None:
                raise SchemaError("masked row tail has no numeric parameters")
            return cls("Masked")
        if value == "Mixed128_64":
            if (threshold, tile_rows) not in ((None, None), (64, 64)):
                raise SchemaError("Mixed128_64 has fixed numeric parameters")
            return cls("SecondaryTile", 64, 64)
        if value != "SecondaryTile" or threshold is None or tile_rows is None:
            raise SchemaError("parameterized row tail requires numeric parameters")
        threshold = _integer(threshold, "row_tail.threshold_rows")
        tile_rows = _integer(tile_rows, "row_tail.tile_rows")
        if threshold <= 0 or tile_rows <= 0:
            raise SchemaError("row-tail dimensions must be positive")
        if threshold > _U32_MAX or tile_rows > _U32_MAX:
            raise SchemaError("row-tail dimensions must fit in a u32")
        return cls("SecondaryTile", threshold, tile_rows)

    @classmethod
    def from_mapping(cls, value: object) -> "GroupedBackwardRowTail":
        item = _strict_mapping_optional(
            value,
            name="GroupedBackwardKernelSpec.row_tail",
            required=frozenset({"kind"}),
            optional=frozenset({"threshold_rows", "tile_rows"}),
        )
        kind = _string(item["kind"], "row_tail.kind")
        if kind == "Masked":
            if any(name in item for name in ("threshold_rows", "tile_rows")):
                raise SchemaError("masked row tail has no secondary tile parameters")
            return cls(kind)
        if kind != "SecondaryTile":
            raise SchemaError(f"unsupported row_tail.kind {kind!r}")
        threshold = _integer(item.get("threshold_rows"), "row_tail.threshold_rows")
        tile_rows = _integer(item.get("tile_rows"), "row_tail.tile_rows")
        if threshold <= 0 or tile_rows <= 0:
            raise SchemaError("row-tail dimensions must be positive")
        if threshold > _U32_MAX or tile_rows > _U32_MAX:
            raise SchemaError("row-tail dimensions must fit in a u32")
        return cls(kind, threshold, tile_rows)

    @property
    def is_secondary(self) -> bool:
        return self.kind == "SecondaryTile"

    def to_mapping(self) -> dict[str, object]:
        if not self.is_secondary:
            return {"kind": self.kind}
        return {
            "kind": self.kind,
            "threshold_rows": self.threshold_rows,
            "tile_rows": self.tile_rows,
        }

    @property
    def legacy_hash_value(self) -> object:
        if not self.is_secondary:
            return "Masked"
        if (self.threshold_rows, self.tile_rows) == (64, 64):
            return "Mixed128_64"
        return self.to_mapping()

    @property
    def solution_value(self) -> str:
        if not self.is_secondary:
            return "Masked"
        if (self.threshold_rows, self.tile_rows) == (64, 64):
            return "Mixed128_64"
        return "SecondaryTile"


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
        if quant_type not in {"Q2_K", "Q4_K", "Q5_K", "IQ2_S"}:
            raise SchemaError(f"unsupported grouped backward quant type {quant_type!r}")
        expected = cls(
            problem_size,
            quant_type,
            BACKWARD_QUANT_FORMATS[quant_type],
        ).to_mapping()
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
        if quant_type not in {"Q2_K", "Q4_K", "Q5_K", "IQ2_S"}:
            raise SchemaError(f"unsupported grouped backward quant type {quant_type!r}")
        for name, value in (
            ("M", problem_size.m),
            ("N", problem_size.n),
            ("K", problem_size.k),
        ):
            if not 0 < value <= _U32_MAX:
                raise SchemaError(f"grouped backward {name} must fit in a positive u32")

        quant_format = BACKWARD_QUANT_FORMATS[quant_type]
        if problem_size.n % quant_format.block_values:
            raise SchemaError(
                "grouped backward N must contain complete packed-weight quant blocks"
            )
        if problem_size.k % 32:
            raise SchemaError("grouped backward K must contain complete DepthU32 tiles")

        packed_row_bytes = (
            problem_size.n // quant_format.block_values * quant_format.block_bytes
        )
        if problem_size.k * packed_row_bytes > _U32_MAX:
            raise SchemaError(
                "grouped backward packed bytes per expert must fit in a u32 offset"
            )
        if problem_size.m * problem_size.k * 2 > _U32_MAX:
            raise SchemaError(
                "grouped backward grad-output workspace must fit in a u32 offset"
            )
        if problem_size.m * problem_size.n * 2 > _U32_MAX:
            raise SchemaError(
                "grouped backward grad-input workspace must fit in a u32 offset"
            )

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
        ownership = GroupedBackwardOwnership.from_solution_value(
            solution.route_ownership
        )
        row_tail = GroupedBackwardRowTail.from_solution(solution)
        return cls(
            BackwardKernelSpec.from_solution(solution.compute), ownership, row_tail
        )

    @classmethod
    def from_mapping(cls, value: object, quant_type: str):
        item = _strict_mapping_optional(
            value,
            name="GroupedBackwardKernelSpec",
            required=frozenset(
                {"geometry", "memory", "pipeline", "store", "ownership", "row_tail"}
            ),
            optional=frozenset({"decode"}),
        )
        ownership_item = _strict_mapping_optional(
            item["ownership"],
            name="GroupedBackwardKernelSpec.ownership",
            required=frozenset({"kind"}),
            optional=frozenset({"split_factor"}),
        )
        compute_mapping = {
            key: item[key] for key in ("geometry", "memory", "pipeline", "store")
        }
        if "decode" in item:
            compute_mapping["decode"] = item["decode"]
        return cls(
            BackwardKernelSpec.from_mapping(compute_mapping, quant_type),
            GroupedBackwardOwnership.from_mapping(ownership_item),
            GroupedBackwardRowTail.from_mapping(item["row_tail"]),
        )

    def to_mapping(self, quant_type: str) -> dict[str, object]:
        mapping = self.compute.to_mapping(quant_type)
        mapping["ownership"] = self.ownership.to_mapping()
        mapping["row_tail"] = self.row_tail.to_mapping()
        return mapping

    def to_legacy_hash_mapping(self, quant_type: str) -> dict[str, object]:
        mapping = self.compute.to_legacy_hash_mapping(quant_type)
        mapping["ownership"] = {
            "kind": self.ownership.solution_value,
            "row_tail": self.row_tail.legacy_hash_value,
        }
        return mapping

    def secondary_compute_spec(self) -> BackwardKernelSpec | None:
        if not self.row_tail.is_secondary:
            return None
        if (
            self.row_tail.threshold_rows is None
            or self.row_tail.tile_rows is None
            or self.row_tail.threshold_rows <= 0
            or self.row_tail.tile_rows <= 0
            or self.row_tail.threshold_rows > _U32_MAX
            or self.row_tail.tile_rows > _U32_MAX
            or self.row_tail.threshold_rows > self.row_tail.tile_rows
            or self.row_tail.tile_rows % 64
        ):
            raise ValueError("secondary row tail has incompatible numeric geometry")
        geometry = self.compute.geometry
        matrix_instruction = (
            geometry.matrix_instruction[:5]
            + (self.row_tail.tile_rows // 64,)
            + geometry.matrix_instruction[6:]
        )
        return replace(
            self.compute,
            geometry=replace(
                geometry,
                matrix_instruction=matrix_instruction,
                macro_tile0=self.row_tail.tile_rows,
            ),
        )

    def to_solution(
        self, contract: GroupedBackwardProblemContract
    ) -> GroupedBackwardSolution:
        threshold_rows = None
        tile_rows = None
        if self.row_tail.solution_value == "SecondaryTile":
            threshold_rows = self.row_tail.threshold_rows
            tile_rows = self.row_tail.tile_rows
        return GroupedBackwardSolution(
            compute=self.compute.to_solution(contract.ordinary()),
            route_ownership=self.ownership.solution_value,
            row_tail=self.row_tail.solution_value,
            row_tail_threshold_rows=threshold_rows,
            row_tail_tile_rows=tile_rows,
        )


@dataclass(frozen=True)
class DerivedGroupedBackwardState:
    contract: GroupedBackwardProblemContract
    spec: GroupedBackwardKernelSpec
    primary: DerivedBackwardState
    secondary: DerivedBackwardState | None

    @classmethod
    def from_solution_key(cls, solution_key: SolutionKey):
        solution = solution_key.solution
        if not isinstance(solution, GroupedBackwardSolution):
            raise TypeError("grouped backward state requires GroupedBackwardSolution")
        contract = GroupedBackwardProblemContract.from_solution_key(solution_key)
        spec = GroupedBackwardKernelSpec.from_solution(solution)
        return cls.from_contract_spec(contract, spec)

    @classmethod
    def from_contract_spec(
        cls,
        contract: GroupedBackwardProblemContract,
        spec: GroupedBackwardKernelSpec,
    ) -> "DerivedGroupedBackwardState":
        ordinary_contract = contract.ordinary()
        secondary_spec = spec.secondary_compute_spec()
        return cls(
            contract=contract,
            spec=spec,
            primary=DerivedBackwardState.from_contract_spec(
                ordinary_contract, spec.compute
            ),
            secondary=(
                DerivedBackwardState.from_contract_spec(
                    ordinary_contract, secondary_spec
                )
                if secondary_spec is not None
                else None
            ),
        )
