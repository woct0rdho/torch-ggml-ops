"""Strict problem and solution state for routed MMQ backward kernels."""

from dataclasses import dataclass, replace
from enum import Enum

from .mmq_bwd_spec import (
    BackwardKernelSpec,
    BackwardProblemContract,
    DerivedBackwardState,
    backward_mechanism_contract,
)
from .model import ProblemSize
from .quant_formats import BACKWARD_QUANT_FORMATS, QuantFormat
from .schema import SchemaError
from .schema import enum_value as _enum
from .schema import integer as _integer
from .schema import strict_mapping_optional as _strict_mapping_optional
from .schema import string as _string

_U32_MAX = 0xFFFFFFFF


class GroupedBackwardEmptyTilePolicy(str, Enum):
    Suppress = "Suppress"
    Execute = "Execute"


@dataclass(frozen=True)
class GroupedBackwardOwnership:
    kind: str
    split_factor: int
    empty_tile_policy: GroupedBackwardEmptyTilePolicy

    @classmethod
    def from_mapping(cls, value: object) -> "GroupedBackwardOwnership":
        item = _strict_mapping_optional(
            value,
            name="GroupedBackwardKernelSpec.Ownership",
            required=frozenset({"Kind", "EmptyTilePolicy"}),
            optional=frozenset({"RouteSplitFactor"}),
        )
        kind = _string(item, "Kind")
        policy = _enum(
            item,
            "EmptyTilePolicy",
            GroupedBackwardEmptyTilePolicy,
        )
        if kind == "Serial":
            if "RouteSplitFactor" in item:
                raise SchemaError("serial ownership has no RouteSplitFactor")
            return cls(kind, 1, policy)
        if kind != "RouteSplit" or "RouteSplitFactor" not in item:
            raise SchemaError("route split ownership requires RouteSplitFactor")
        factor = _integer(item, "RouteSplitFactor")
        if factor <= 1 or factor & (factor - 1):
            raise SchemaError("ownership split factor must be a power of two above one")
        return cls(kind, factor, policy)

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "Kind": self.kind,
            "EmptyTilePolicy": self.empty_tile_policy.value,
        }
        if self.kind != "Serial":
            result["RouteSplitFactor"] = self.split_factor
        return result


@dataclass(frozen=True)
class GroupedBackwardRowTail:
    kind: str
    threshold_rows: int | None = None
    tile_rows: int | None = None

    @classmethod
    def from_mapping(cls, value: object) -> "GroupedBackwardRowTail":
        item = _strict_mapping_optional(
            value,
            name="GroupedBackwardKernelSpec.RowTail",
            required=frozenset({"Kind"}),
            optional=frozenset({"ThresholdRows", "TileRows"}),
        )
        kind = _string(item, "Kind")
        if kind == "Masked":
            if any(name in item for name in ("ThresholdRows", "TileRows")):
                raise SchemaError("masked row tail has no secondary tile parameters")
            return cls(kind)
        if kind != "SecondaryTile":
            raise SchemaError(f"unsupported RowTail.Kind {kind!r}")
        threshold = _integer(item.get("ThresholdRows"), "RowTail.ThresholdRows")
        tile_rows = _integer(item.get("TileRows"), "RowTail.TileRows")
        if threshold <= 0 or tile_rows <= 0:
            raise SchemaError("row-tail dimensions must be positive")
        if threshold > _U32_MAX or tile_rows > _U32_MAX:
            raise SchemaError("row-tail dimensions must fit in a u32")
        if threshold > tile_rows:
            raise SchemaError("row-tail threshold cannot exceed tile rows")
        if tile_rows % 64:
            raise SchemaError("row-tail tile rows must be a multiple of 64")
        return cls(kind, threshold, tile_rows)

    @property
    def is_secondary(self) -> bool:
        return self.kind == "SecondaryTile"

    def to_mapping(self) -> dict[str, object]:
        if not self.is_secondary:
            return {"Kind": self.kind}
        return {
            "Kind": self.kind,
            "ThresholdRows": self.threshold_rows,
            "TileRows": self.tile_rows,
        }


@dataclass(frozen=True)
class GroupedBackwardProblemContract:
    problem_size: ProblemSize
    quant_type: str
    quant_format: QuantFormat
    physical_experts: int = 256
    max_route_entries: int = 256
    code_object_version: int = 5

    @classmethod
    def for_problem(
        cls, problem_size: ProblemSize, quant_type: str
    ) -> "GroupedBackwardProblemContract":
        cls._validate_problem(problem_size, quant_type)
        return cls(
            problem_size=problem_size,
            quant_type=quant_type,
            quant_format=BACKWARD_QUANT_FORMATS[quant_type],
        )

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
    def from_mapping(cls, value: object, quant_type: str):
        item = _strict_mapping_optional(
            value,
            name="GroupedBackwardKernelSpec",
            required=frozenset(
                {"Geometry", "Memory", "Pipeline", "Store", "Ownership", "RowTail"}
            ),
            optional=frozenset({"Decode"}),
        )
        ownership_item = item["Ownership"]
        compute_mapping = {
            key: item[key] for key in ("Geometry", "Memory", "Pipeline", "Store")
        }
        if "Decode" in item:
            compute_mapping["Decode"] = item["Decode"]
        return cls(
            BackwardKernelSpec.from_mapping(compute_mapping, quant_type),
            GroupedBackwardOwnership.from_mapping(ownership_item),
            GroupedBackwardRowTail.from_mapping(item["RowTail"]),
        )

    def to_mapping(self, quant_type: str) -> dict[str, object]:
        mapping = self.compute.to_mapping(quant_type)
        mapping["Ownership"] = self.ownership.to_mapping()
        mapping["RowTail"] = self.row_tail.to_mapping()
        return mapping

    def secondary_compute_spec(self) -> BackwardKernelSpec | None:
        if not self.row_tail.is_secondary:
            return None
        assert self.row_tail.threshold_rows is not None
        assert self.row_tail.tile_rows is not None
        assert 0 < self.row_tail.threshold_rows <= _U32_MAX
        assert 0 < self.row_tail.tile_rows <= _U32_MAX
        assert self.row_tail.threshold_rows <= self.row_tail.tile_rows
        assert self.row_tail.tile_rows % 64 == 0
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


@dataclass(frozen=True)
class DerivedGroupedBackwardState:
    contract: GroupedBackwardProblemContract
    spec: GroupedBackwardKernelSpec
    primary: DerivedBackwardState
    secondary: DerivedBackwardState | None

    @classmethod
    def from_problem_spec(
        cls,
        problem_size: ProblemSize,
        quant_type: str,
        spec: GroupedBackwardKernelSpec,
    ) -> "DerivedGroupedBackwardState":
        contract = GroupedBackwardProblemContract.for_problem(problem_size, quant_type)
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
