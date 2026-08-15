"""Strict identities for isolated grouped MMQ forward kernels."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import ClassVar

from typing_extensions import Self

from .model import SchemaError
from .quant_formats import (
    GROUPED_QUANT_FORMATS,
    Q8_1_F16_D2S6_BLOCK_BYTES,
    Q8_1_F16_D4S4_BLOCK_BYTES,
    Q8_1_F32_D4_BLOCK_BYTES,
)


def _mapping(value: object, name: str, keys: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{name} must be a mapping")
    normalized = {key: item for key, item in value.items() if isinstance(key, str)}
    if len(normalized) != len(value):
        raise SchemaError(f"{name} keys must be strings")
    missing = sorted(keys - set(normalized))
    unknown = sorted(set(normalized) - keys)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise SchemaError(f"invalid {name}: {', '.join(details)}")
    return normalized


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
    items = _integer_tuple(value, name, 3)
    return (items[0], items[1], items[2])


@dataclass(frozen=True)
class GroupedForwardProblem:
    """One aggregate routed problem over the fixed packed expert bank."""

    quant_data_type: str
    aggregate_rows: int
    output_features: int
    input_features: int
    physical_experts: int
    max_route_entries: int

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "QuantDataType",
            "AggregateRows",
            "OutputFeatures",
            "InputFeatures",
            "PhysicalExperts",
            "MaxRouteEntries",
        }
    )

    @classmethod
    def q2_k(cls, aggregate_rows: int) -> Self:
        return cls("Q2_K", aggregate_rows, 4096, 2048, 256, 256)

    @classmethod
    def q4_k(cls, aggregate_rows: int) -> Self:
        return cls("Q4_K", aggregate_rows, 2048, 512, 256, 256)

    @classmethod
    def q5_k(cls, aggregate_rows: int) -> Self:
        return cls("Q5_K", aggregate_rows, 2048, 512, 256, 256)

    @classmethod
    def iq2_s(cls, aggregate_rows: int) -> Self:
        return cls("IQ2_S", aggregate_rows, 2048, 512, 256, 256)

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _mapping(value, "GroupedForwardProblem", cls._KEYS)
        return cls(
            quant_data_type=_string(item["QuantDataType"], "QuantDataType"),
            aggregate_rows=_integer(item["AggregateRows"], "AggregateRows"),
            output_features=_integer(item["OutputFeatures"], "OutputFeatures"),
            input_features=_integer(item["InputFeatures"], "InputFeatures"),
            physical_experts=_integer(item["PhysicalExperts"], "PhysicalExperts"),
            max_route_entries=_integer(item["MaxRouteEntries"], "MaxRouteEntries"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "QuantDataType": self.quant_data_type,
            "AggregateRows": self.aggregate_rows,
            "OutputFeatures": self.output_features,
            "InputFeatures": self.input_features,
            "PhysicalExperts": self.physical_experts,
            "MaxRouteEntries": self.max_route_entries,
        }


@dataclass(frozen=True)
class GroupedForwardSolution:
    """Complete mechanism identity for one grouped forward lowering."""

    kernel_language: str
    isa: tuple[int, int, int]
    wavefront_size: int
    work_group: tuple[int, int, int]
    matrix_instruction: tuple[int, ...]
    macro_tile0: int
    tail_macro_tile0: int
    macro_tile1: int
    depth_u: int
    activation_layout: str
    activation_block_bytes: int
    packed_weight_block_bytes: int
    operand_source: str
    weight_decode: str
    group_mapping: str
    route_layout: str
    activation_addressing: str
    metadata_conversion: str
    metadata_schedule: str
    epilogue_tiles_ahead: int
    epilogue_dependency_width: int
    epilogue_priority: int
    scale_arithmetic: str
    output_store: str
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
            "TailMacroTile0",
            "MacroTile1",
            "DepthU",
            "ActivationLayout",
            "ActivationBlockBytes",
            "PackedWeightBlockBytes",
            "OperandSource",
            "WeightDecode",
            "GroupMapping",
            "RouteLayout",
            "ActivationAddressing",
            "MetadataConversion",
            "MetadataSchedule",
            "EpilogueTilesAhead",
            "EpilogueDependencyWidth",
            "EpiloguePriority",
            "ScaleArithmetic",
            "OutputStore",
            "SignedWeight",
            "SignedActivation",
            "WmmaClamp",
        }
    )

    @classmethod
    def q4_k_serial_direct(cls) -> Self:
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(32, 1, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 1, 1, 1),
            macro_tile0=16,
            tail_macro_tile0=16,
            macro_tile1=16,
            depth_u=32,
            activation_layout="F16_D4S4",
            activation_block_bytes=Q8_1_F16_D4S4_BLOCK_BYTES,
            packed_weight_block_bytes=GROUPED_QUANT_FORMATS["Q4_K"].block_bytes,
            operand_source="GroupedDirectGlobal",
            weight_decode="DirectNibble",
            group_mapping="SerialGemm",
            route_layout="CumulativeOffsetsExpertIndices",
            activation_addressing="AggregateRows",
            metadata_conversion="Float32ThenFloat16",
            metadata_schedule="Serialized",
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=1,
            epilogue_priority=0,
            scale_arithmetic="FP16",
            output_store="BFloat16RNEMasked",
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=True,
        )

    @classmethod
    def q4_k_serial_decoded_lds(cls) -> Self:
        return replace(
            cls.q4_k_serial_direct(),
            work_group=(128, 1, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=128,
            tail_macro_tile0=128,
            macro_tile1=64,
            operand_source="GroupedDecodedWeightLds",
            activation_addressing="AggregateRowsTiled",
            metadata_conversion="DirectFloat16Unsigned16",
            epilogue_tiles_ahead=8,
            output_store="BFloat16RNEClause8Masked",
        )

    @classmethod
    def q4_k_serial_decoded_lds_64(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds(),
            macro_tile0=64,
            tail_macro_tile0=64,
            epilogue_tiles_ahead=4,
            output_store="BFloat16RNEClause4Masked",
        )

    @classmethod
    def q4_k_serial_decoded_lds_scheduled(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds(),
            metadata_schedule="IndependentExtractionMetadataAfterLowWmma",
        )

    @classmethod
    def q4_k_serial_decoded_lds_scheduled_a1d2p2(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds_scheduled(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
        )

    @classmethod
    def q4_k_serial_decoded_lds_scheduled_a1d4p2(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds_scheduled(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=4,
            epilogue_priority=2,
        )

    @classmethod
    def q4_k_serial_decoded_lds_64_scheduled(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds_64(),
            metadata_schedule="IndependentExtractionMetadataAfterLowWmma",
        )

    @classmethod
    def q4_k_serial_decoded_lds_64_scheduled_mixed32(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds_64_scheduled(),
            tail_macro_tile0=32,
            output_store="BFloat16RNEClause4Clause2MixedMasked",
        )

    @classmethod
    def q4_k_serial_decoded_lds_64_scheduled_mixed32_a1d2p2(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds_64_scheduled_mixed32(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
        )

    @classmethod
    def q4_k_serial_decoded_lds_64_scheduled_mixed32_a1d4p2(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds_64_scheduled_mixed32(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=4,
            epilogue_priority=2,
        )

    @classmethod
    def iq2_s_serial_full_weight_lds_64(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds_64(),
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            packed_weight_block_bytes=GROUPED_QUANT_FORMATS["IQ2_S"].block_bytes,
            operand_source="GroupedIQ2SFullWeightLds",
            weight_decode="DirectIQ2SGridSigned",
            metadata_conversion="Float16DUnsignedNibbleScaleToFloat32",
            metadata_schedule="IQ2SDistributedFullWeightDecode",
            scale_arithmetic="Int32ScaleF32",
            output_store="BFloat16RNEClause4Masked",
            wmma_clamp=False,
        )

    @classmethod
    def iq2_s_serial_full_weight_lds_64_linear_activation(cls) -> Self:
        return replace(
            cls.iq2_s_serial_full_weight_lds_64(),
            activation_addressing="AggregateRowsTiledLinear",
        )

    @classmethod
    def q2_k_serial_decoded_lds_32(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds(),
            macro_tile0=32,
            tail_macro_tile0=32,
            activation_layout="F16_D2S6",
            activation_block_bytes=Q8_1_F16_D2S6_BLOCK_BYTES,
            packed_weight_block_bytes=GROUPED_QUANT_FORMATS["Q2_K"].block_bytes,
            weight_decode="DirectTwoBitNibbleScaleMinimum",
            metadata_conversion="DirectQ2Float16NibblePairs",
            metadata_schedule="Q2ScaleMinimumNibble",
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
            output_store="BFloat16RNEClause2Masked",
        )

    @classmethod
    def q2_k_serial_decoded_lds_64(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32(),
            macro_tile0=64,
            tail_macro_tile0=64,
            epilogue_tiles_ahead=4,
            output_store="BFloat16RNEClause4Masked",
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_unrolled(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32(),
            metadata_schedule="Q2ScaleMinimumNibbleUnrolled",
        )

    @classmethod
    def q2_k_serial_decoded_lds_64_unrolled(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_64(),
            metadata_schedule="Q2ScaleMinimumNibbleUnrolled",
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_association(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_unrolled(),
            metadata_schedule="Q2ScaleMinimumNibbleUnrolledHipAssociation",
        )

    @classmethod
    def q2_k_serial_decoded_lds_64_hip_association(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_64_unrolled(),
            metadata_schedule="Q2ScaleMinimumNibbleUnrolledHipAssociation",
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_partial_lds(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_hip_association(),
            metadata_schedule="Q2HipAssociationPartialLds",
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_pre_negated_dm(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_hip_partial_lds(),
            metadata_schedule="Q2HipAssociationPartialLdsPreNegatedDm",
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_hip_pre_negated_dm(),
            metadata_schedule="Q2HipAssociationPartialLdsPreNegatedDmWrite2",
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2(),
            metadata_schedule="Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2",
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2_distributed(
        cls,
    ) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2(),
            metadata_schedule=(
                "Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2DistributedProducer"
            ),
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2_mixed16(
        cls,
    ) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2(),
            tail_macro_tile0=16,
            output_store="BFloat16RNEClause2Clause1MixedMasked",
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2_distributed_mixed16(
        cls,
    ) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2_distributed(),
            tail_macro_tile0=16,
            output_store="BFloat16RNEClause2Clause1MixedMasked",
        )

    @classmethod
    def q2_k_serial_decoded_lds_64_hip_distributed(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_64_hip_association(),
            metadata_schedule=(
                "Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2DistributedProducer"
            ),
        )

    @classmethod
    def q2_k_serial_decoded_lds_128_unrolled(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_64_unrolled(),
            macro_tile0=128,
            tail_macro_tile0=128,
            epilogue_tiles_ahead=8,
            output_store="BFloat16RNEClause8Masked",
        )

    @classmethod
    def q5_k_serial_decoded_lds(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds(),
            packed_weight_block_bytes=GROUPED_QUANT_FORMATS["Q5_K"].block_bytes,
            weight_decode="DirectNibbleHighBit",
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds(),
            metadata_schedule="IndependentExtractionMetadataAfterLowWmma",
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_a1d2p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_scheduled(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_mixed64(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_scheduled(),
            tail_macro_tile0=64,
            output_store="BFloat16RNEClause8Clause4MixedMasked",
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_mixed64_a1d2p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_scheduled_mixed64(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_mixed64_a1d4p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_scheduled_mixed64(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=4,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_mixed64_mixed32(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_scheduled(),
            tail_macro_tile0=32,
            output_store="BFloat16RNEClause8Clause4Clause2MixedMasked",
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_mixed64_mixed32_a1d2p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_scheduled_mixed64_mixed32(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_mixed64_mixed32_a1d4p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_scheduled_mixed64_mixed32(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=4,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_64(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds(),
            macro_tile0=64,
            tail_macro_tile0=64,
            epilogue_tiles_ahead=4,
            output_store="BFloat16RNEClause4Masked",
        )

    @classmethod
    def q5_k_serial_decoded_lds_64_scheduled(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_64(),
            metadata_schedule="IndependentExtractionMetadataAfterLowWmma",
        )

    @classmethod
    def q5_k_serial_decoded_lds_64_scheduled_a1d2p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_64_scheduled(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_64_scheduled_a1d4p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_64_scheduled(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=4,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_64_scheduled_mixed32(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_64_scheduled(),
            tail_macro_tile0=32,
            output_store="BFloat16RNEClause4Clause2MixedMasked",
        )

    @classmethod
    def q5_k_serial_decoded_lds_64_scheduled_mixed32_a1d2p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_64_scheduled_mixed32(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_64_scheduled_mixed32_a1d4p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_64_scheduled_mixed32(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=4,
            epilogue_priority=2,
        )

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _mapping(value, "GroupedForwardSolution", cls._KEYS)
        return cls(
            kernel_language=_string(item["KernelLanguage"], "KernelLanguage"),
            isa=_integer_triple(item["ISA"], "ISA"),
            wavefront_size=_integer(item["WavefrontSize"], "WavefrontSize"),
            work_group=_integer_triple(item["WorkGroup"], "WorkGroup"),
            matrix_instruction=_integer_tuple(
                item["MatrixInstruction"], "MatrixInstruction", 9
            ),
            macro_tile0=_integer(item["MacroTile0"], "MacroTile0"),
            tail_macro_tile0=_integer(item["TailMacroTile0"], "TailMacroTile0"),
            macro_tile1=_integer(item["MacroTile1"], "MacroTile1"),
            depth_u=_integer(item["DepthU"], "DepthU"),
            activation_layout=_string(item["ActivationLayout"], "ActivationLayout"),
            activation_block_bytes=_integer(
                item["ActivationBlockBytes"], "ActivationBlockBytes"
            ),
            packed_weight_block_bytes=_integer(
                item["PackedWeightBlockBytes"], "PackedWeightBlockBytes"
            ),
            operand_source=_string(item["OperandSource"], "OperandSource"),
            weight_decode=_string(item["WeightDecode"], "WeightDecode"),
            group_mapping=_string(item["GroupMapping"], "GroupMapping"),
            route_layout=_string(item["RouteLayout"], "RouteLayout"),
            activation_addressing=_string(
                item["ActivationAddressing"], "ActivationAddressing"
            ),
            metadata_conversion=_string(
                item["MetadataConversion"], "MetadataConversion"
            ),
            metadata_schedule=_string(item["MetadataSchedule"], "MetadataSchedule"),
            epilogue_tiles_ahead=_integer(
                item["EpilogueTilesAhead"], "EpilogueTilesAhead"
            ),
            epilogue_dependency_width=_integer(
                item["EpilogueDependencyWidth"], "EpilogueDependencyWidth"
            ),
            epilogue_priority=_integer(item["EpiloguePriority"], "EpiloguePriority"),
            scale_arithmetic=_string(item["ScaleArithmetic"], "ScaleArithmetic"),
            output_store=_string(item["OutputStore"], "OutputStore"),
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
            "TailMacroTile0": self.tail_macro_tile0,
            "MacroTile1": self.macro_tile1,
            "DepthU": self.depth_u,
            "ActivationLayout": self.activation_layout,
            "ActivationBlockBytes": self.activation_block_bytes,
            "PackedWeightBlockBytes": self.packed_weight_block_bytes,
            "OperandSource": self.operand_source,
            "WeightDecode": self.weight_decode,
            "GroupMapping": self.group_mapping,
            "RouteLayout": self.route_layout,
            "ActivationAddressing": self.activation_addressing,
            "MetadataConversion": self.metadata_conversion,
            "MetadataSchedule": self.metadata_schedule,
            "EpilogueTilesAhead": self.epilogue_tiles_ahead,
            "EpilogueDependencyWidth": self.epilogue_dependency_width,
            "EpiloguePriority": self.epilogue_priority,
            "ScaleArithmetic": self.scale_arithmetic,
            "OutputStore": self.output_store,
            "SignedWeight": self.signed_weight,
            "SignedActivation": self.signed_activation,
            "WmmaClamp": self.wmma_clamp,
        }


@dataclass(frozen=True)
class GroupedForwardSolutionKey:
    problem: GroupedForwardProblem
    solution: GroupedForwardSolution

    _KEYS: ClassVar[frozenset[str]] = frozenset({"Problem", "Solution"})

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _mapping(value, "GroupedForwardSolutionKey", cls._KEYS)
        return cls(
            GroupedForwardProblem.from_mapping(item["Problem"]),
            GroupedForwardSolution.from_mapping(item["Solution"]),
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
        return f"ggsol_{digest[:16]}"

    @property
    def kernel_name(self) -> str:
        problem = self.problem
        return (
            "torch_ggml_ops_ggtensile_gfx1151_v1_grouped_mmq_fwd_"
            f"{problem.quant_data_type.lower()}_r{problem.aggregate_rows}_"
            f"n{problem.output_features}_k{problem.input_features}_{self.hash[6:]}"
        )
