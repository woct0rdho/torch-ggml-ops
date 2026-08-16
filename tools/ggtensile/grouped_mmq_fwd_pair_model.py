"""Strict identities for research-only paired grouped forward paths."""

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar

from typing_extensions import Self

from .grouped_mmq_fwd_model import (
    GroupedActivationAddressing,
    GroupedOutputStore,
    _boolean,
    _enum,
    _integer,
    _integer_triple,
    _integer_tuple,
    _mapping,
    _string,
)
from .quant_formats import Q8_1_F32_D4_BLOCK_BYTES


class GroupedPairOperandSource(str, Enum):
    IQ2SHalfWeightLds = "IQ2SHalfWeightLds"
    IQ2XXSHalfWeightLds = "IQ2XXSHalfWeightLds"
    Q3KHalfWeightLds = "Q3KHalfWeightLds"


class GroupedPairProjectionSchedule(str, Enum):
    K128Interleaved = "K128Interleaved"


class GroupedPairDecodeSchedule(str, Enum):
    TwoLaneSelectedHalfPayloadPrefetch = "TwoLaneSelectedHalfPayloadPrefetch"
    TwoLaneSelectedHalfIQ2XXS = "TwoLaneSelectedHalfIQ2XXS"
    TwoLaneSelectedHalfQ3 = "TwoLaneSelectedHalfQ3"


class GroupedPairRouteOwnership(str, Enum):
    SerialRoutes = "SerialRoutes"
    DeviceRowTasks64 = "DeviceRowTasks64"


@dataclass(frozen=True)
class GroupedForwardPairProblem:
    """One exact two-projection aggregate routed problem."""

    quant_data_type: str
    aggregate_rows: int
    output_features: int
    input_features: int
    physical_experts: int
    max_route_entries: int
    projection_count: int

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "QuantDataType",
            "AggregateRows",
            "OutputFeatures",
            "InputFeatures",
            "PhysicalExperts",
            "MaxRouteEntries",
            "ProjectionCount",
        }
    )

    @classmethod
    def iq2_s(cls, aggregate_rows: int) -> Self:
        return cls("IQ2_S", aggregate_rows, 512, 2048, 256, 256, 2)

    @classmethod
    def iq2_xxs(cls, aggregate_rows: int) -> Self:
        return cls("IQ2_XXS", aggregate_rows, 2048, 4096, 256, 256, 2)

    @classmethod
    def q3_k(cls, aggregate_rows: int) -> Self:
        return cls("Q3_K", aggregate_rows, 512, 2048, 256, 256, 2)

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _mapping(value, "GroupedForwardPairProblem", cls._KEYS)
        return cls(
            quant_data_type=_string(item["QuantDataType"], "QuantDataType"),
            aggregate_rows=_integer(item["AggregateRows"], "AggregateRows"),
            output_features=_integer(item["OutputFeatures"], "OutputFeatures"),
            input_features=_integer(item["InputFeatures"], "InputFeatures"),
            physical_experts=_integer(item["PhysicalExperts"], "PhysicalExperts"),
            max_route_entries=_integer(item["MaxRouteEntries"], "MaxRouteEntries"),
            projection_count=_integer(item["ProjectionCount"], "ProjectionCount"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "QuantDataType": self.quant_data_type,
            "AggregateRows": self.aggregate_rows,
            "OutputFeatures": self.output_features,
            "InputFeatures": self.input_features,
            "PhysicalExperts": self.physical_experts,
            "MaxRouteEntries": self.max_route_entries,
            "ProjectionCount": self.projection_count,
        }


@dataclass(frozen=True)
class GroupedForwardPairSolution:
    """Complete mechanism identity for one paired lowering."""

    kernel_language: str
    isa: tuple[int, int, int]
    wavefront_size: int
    work_group: tuple[int, int, int]
    matrix_instruction: tuple[int, ...]
    macro_tile0: int
    macro_tile1: int
    depth_u: int
    activation_layout: str
    activation_block_bytes: int
    packed_weight_block_bytes: int
    operand_source: GroupedPairOperandSource
    projection_schedule: GroupedPairProjectionSchedule
    weight_decode: str
    group_mapping: str
    route_layout: str
    activation_addressing: GroupedActivationAddressing
    metadata_conversion: str
    metadata_schedule: GroupedPairDecodeSchedule
    output_store: GroupedOutputStore
    scale_arithmetic: str
    projection_count: int
    signed_weight: bool
    signed_activation: bool
    wmma_clamp: bool

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "KernelLanguage",
            "ISA",
            "WavefrontSize",
            "WorkGroup",
            "MatrixInstruction",
            "MacroTile0",
            "MacroTile1",
            "DepthU",
            "ActivationLayout",
            "ActivationBlockBytes",
            "PackedWeightBlockBytes",
            "OperandSource",
            "ProjectionSchedule",
            "WeightDecode",
            "GroupMapping",
            "RouteLayout",
            "ActivationAddressing",
            "MetadataConversion",
            "MetadataSchedule",
            "OutputStore",
            "ScaleArithmetic",
            "ProjectionCount",
            "SignedWeight",
            "SignedActivation",
            "WmmaClamp",
        }
    )

    @classmethod
    def iq2_s_k128_interleaved(cls) -> Self:
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(128, 1, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=64,
            macro_tile1=64,
            depth_u=128,
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            packed_weight_block_bytes=82,
            operand_source=GroupedPairOperandSource.IQ2SHalfWeightLds,
            projection_schedule=GroupedPairProjectionSchedule.K128Interleaved,
            weight_decode="TwoLaneSelectedHalfIQ2SGridSigned",
            group_mapping="SerialGemmPair",
            route_layout="CumulativeOffsetsExpertIndices",
            activation_addressing=GroupedActivationAddressing.AggregateRowsTiledLinear,
            metadata_conversion="Float16DUnsignedNibbleScaleToFloat32",
            metadata_schedule=(
                GroupedPairDecodeSchedule.TwoLaneSelectedHalfPayloadPrefetch
            ),
            output_store=GroupedOutputStore.BFloat16RNEClause4Masked,
            scale_arithmetic="Int32ScaleF32",
            projection_count=2,
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=False,
        )

    @classmethod
    def iq2_s_k128_interleaved_row_tasks(cls) -> Self:
        return replace(
            cls.iq2_s_k128_interleaved(),
            group_mapping="RowTaskGemmPair",
            route_layout="DeviceRowTasks64",
        )

    @classmethod
    def iq2_xxs_k128_interleaved(cls) -> Self:
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(128, 1, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=64,
            macro_tile1=64,
            depth_u=128,
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            packed_weight_block_bytes=66,
            operand_source=GroupedPairOperandSource.IQ2XXSHalfWeightLds,
            projection_schedule=GroupedPairProjectionSchedule.K128Interleaved,
            weight_decode="TwoLaneSelectedHalfIQ2XXSGridParitySigned",
            group_mapping="SerialGemmPair",
            route_layout="CumulativeOffsetsExpertIndices",
            activation_addressing=GroupedActivationAddressing.AggregateRowsTiledLinear,
            metadata_conversion="Float16DParitySignsOddScaleToFloat32",
            metadata_schedule=GroupedPairDecodeSchedule.TwoLaneSelectedHalfIQ2XXS,
            output_store=GroupedOutputStore.BFloat16RNEClause4Masked,
            scale_arithmetic="Int32ScaleF32",
            projection_count=2,
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=False,
        )

    @classmethod
    def q3_k_k128_interleaved(cls) -> Self:
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(128, 1, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=64,
            macro_tile1=64,
            depth_u=128,
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            packed_weight_block_bytes=110,
            operand_source=GroupedPairOperandSource.Q3KHalfWeightLds,
            projection_schedule=GroupedPairProjectionSchedule.K128Interleaved,
            weight_decode="TwoLaneSelectedHalfQ3Signed",
            group_mapping="SerialGemmPair",
            route_layout="CumulativeOffsetsExpertIndices",
            activation_addressing=GroupedActivationAddressing.AggregateRowsTiledLinear,
            metadata_conversion="Float16DSignedSixBitScaleToFloat32",
            metadata_schedule=GroupedPairDecodeSchedule.TwoLaneSelectedHalfQ3,
            output_store=GroupedOutputStore.BFloat16RNEClause4Masked,
            scale_arithmetic="Int32ScaleF32",
            projection_count=2,
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=False,
        )

    @classmethod
    def q3_k_k128_interleaved_row_tasks(cls) -> Self:
        return replace(
            cls.q3_k_k128_interleaved(),
            group_mapping="RowTaskGemmPair",
            route_layout="DeviceRowTasks64",
        )

    @property
    def route_ownership(self) -> GroupedPairRouteOwnership:
        identity = (self.group_mapping, self.route_layout)
        if identity == ("SerialGemmPair", "CumulativeOffsetsExpertIndices"):
            return GroupedPairRouteOwnership.SerialRoutes
        if identity == ("RowTaskGemmPair", "DeviceRowTasks64"):
            return GroupedPairRouteOwnership.DeviceRowTasks64
        raise ValueError("paired route ownership identity is unsupported")

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _mapping(value, "GroupedForwardPairSolution", cls._KEYS)
        return cls(
            kernel_language=_string(item["KernelLanguage"], "KernelLanguage"),
            isa=_integer_triple(item["ISA"], "ISA"),
            wavefront_size=_integer(item["WavefrontSize"], "WavefrontSize"),
            work_group=_integer_triple(item["WorkGroup"], "WorkGroup"),
            matrix_instruction=_integer_tuple(
                item["MatrixInstruction"], "MatrixInstruction", 9
            ),
            macro_tile0=_integer(item["MacroTile0"], "MacroTile0"),
            macro_tile1=_integer(item["MacroTile1"], "MacroTile1"),
            depth_u=_integer(item["DepthU"], "DepthU"),
            activation_layout=_string(item["ActivationLayout"], "ActivationLayout"),
            activation_block_bytes=_integer(
                item["ActivationBlockBytes"], "ActivationBlockBytes"
            ),
            packed_weight_block_bytes=_integer(
                item["PackedWeightBlockBytes"], "PackedWeightBlockBytes"
            ),
            operand_source=_enum(
                item["OperandSource"], "OperandSource", GroupedPairOperandSource
            ),
            projection_schedule=_enum(
                item["ProjectionSchedule"],
                "ProjectionSchedule",
                GroupedPairProjectionSchedule,
            ),
            weight_decode=_string(item["WeightDecode"], "WeightDecode"),
            group_mapping=_string(item["GroupMapping"], "GroupMapping"),
            route_layout=_string(item["RouteLayout"], "RouteLayout"),
            activation_addressing=_enum(
                item["ActivationAddressing"],
                "ActivationAddressing",
                GroupedActivationAddressing,
            ),
            metadata_conversion=_string(
                item["MetadataConversion"], "MetadataConversion"
            ),
            metadata_schedule=_enum(
                item["MetadataSchedule"],
                "MetadataSchedule",
                GroupedPairDecodeSchedule,
            ),
            output_store=_enum(item["OutputStore"], "OutputStore", GroupedOutputStore),
            scale_arithmetic=_string(item["ScaleArithmetic"], "ScaleArithmetic"),
            projection_count=_integer(item["ProjectionCount"], "ProjectionCount"),
            signed_weight=_boolean(item["SignedWeight"], "SignedWeight"),
            signed_activation=_boolean(item["SignedActivation"], "SignedActivation"),
            wmma_clamp=_boolean(item["WmmaClamp"], "WmmaClamp"),
        )

    @property
    def num_threads(self) -> int:
        return self.work_group[0] * self.work_group[1] * self.work_group[2]

    def to_mapping(self) -> dict[str, object]:
        return {
            "KernelLanguage": self.kernel_language,
            "ISA": list(self.isa),
            "WavefrontSize": self.wavefront_size,
            "WorkGroup": list(self.work_group),
            "MatrixInstruction": list(self.matrix_instruction),
            "MacroTile0": self.macro_tile0,
            "MacroTile1": self.macro_tile1,
            "DepthU": self.depth_u,
            "ActivationLayout": self.activation_layout,
            "ActivationBlockBytes": self.activation_block_bytes,
            "PackedWeightBlockBytes": self.packed_weight_block_bytes,
            "OperandSource": self.operand_source.value,
            "ProjectionSchedule": self.projection_schedule.value,
            "WeightDecode": self.weight_decode,
            "GroupMapping": self.group_mapping,
            "RouteLayout": self.route_layout,
            "ActivationAddressing": self.activation_addressing.value,
            "MetadataConversion": self.metadata_conversion,
            "MetadataSchedule": self.metadata_schedule.value,
            "OutputStore": self.output_store.value,
            "ScaleArithmetic": self.scale_arithmetic,
            "ProjectionCount": self.projection_count,
            "SignedWeight": self.signed_weight,
            "SignedActivation": self.signed_activation,
            "WmmaClamp": self.wmma_clamp,
        }


@dataclass(frozen=True)
class GroupedForwardPairSolutionKey:
    problem: GroupedForwardPairProblem
    solution: GroupedForwardPairSolution

    _KEYS: ClassVar[frozenset[str]] = frozenset({"Problem", "Solution"})

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _mapping(value, "GroupedForwardPairSolutionKey", cls._KEYS)
        return cls(
            GroupedForwardPairProblem.from_mapping(item["Problem"]),
            GroupedForwardPairSolution.from_mapping(item["Solution"]),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "Problem": self.problem.to_mapping(),
            "Solution": self.solution.to_mapping(),
        }

    @property
    def hash(self) -> str:
        canonical = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"ggpair_{digest[:16]}"

    @property
    def kernel_name(self) -> str:
        problem = self.problem
        return (
            "torch_ggml_ops_ggtensile_gfx1151_v1_grouped_mmq_fwd_pair_"
            f"{problem.quant_data_type.lower()}_r{problem.aggregate_rows}_"
            f"n{problem.output_features}_k{problem.input_features}_{self.hash[7:]}"
        )
