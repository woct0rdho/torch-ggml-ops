import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Literal, overload

from typing_extensions import Self

from .quant_formats import (
    Q8_1_F16_D4S4_BLOCK_BYTES,
    Q8_1_F32_D4_BLOCK_BYTES,
    QUANT_FORMATS,
)


class SchemaError(ValueError):
    """A GGTensile input does not match its strict schema."""


def _strict_mapping(
    value: object,
    *,
    name: str,
    keys: frozenset[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{name} must be a mapping")
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise SchemaError(f"{name} keys must be strings")
        normalized[key] = item
    actual = set(normalized)
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
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


@overload
def _integer_tuple(
    value: object, name: str, length: Literal[3]
) -> tuple[int, int, int]: ...


@overload
def _integer_tuple(value: object, name: str, length: int) -> tuple[int, ...]: ...


def _integer_tuple(value: object, name: str, length: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise SchemaError(f"{name} must be a {length}-element list")
    return tuple(_integer(item, f"{name}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class ProblemType:
    operation_type: str
    quant_data_type: str
    data_type_a: str
    data_type_b: str
    dest_data_type: str
    compute_data_type: str
    transpose_a: bool
    transpose_b: bool

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "OperationType",
            "QuantDataType",
            "DataTypeA",
            "DataTypeB",
            "DestDataType",
            "ComputeDataType",
            "TransposeA",
            "TransposeB",
        }
    )

    @classmethod
    def mmq_backward(cls, quant_data_type: str) -> Self:
        if quant_data_type not in {"Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"}:
            raise ValueError(f"unsupported MMQ backward quant type {quant_data_type!r}")
        return cls(
            operation_type="MMQBackward",
            quant_data_type=quant_data_type,
            data_type_a="BFloat16",
            data_type_b=quant_data_type,
            dest_data_type="BFloat16",
            compute_data_type="Float",
            transpose_a=False,
            transpose_b=False,
        )

    @classmethod
    def mmq_forward(cls, quant_data_type: str) -> Self:
        if quant_data_type not in {"Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"}:
            raise ValueError(f"unsupported MMQ forward quant type {quant_data_type!r}")
        return cls(
            operation_type="MMQForward",
            quant_data_type=quant_data_type,
            data_type_a="Q8_1",
            data_type_b=quant_data_type,
            dest_data_type="BFloat16",
            compute_data_type="Float",
            transpose_a=False,
            transpose_b=True,
        )

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _strict_mapping(value, name="ProblemType", keys=cls._KEYS)
        return cls(
            operation_type=_string(item["OperationType"], "OperationType"),
            quant_data_type=_string(item["QuantDataType"], "QuantDataType"),
            data_type_a=_string(item["DataTypeA"], "DataTypeA"),
            data_type_b=_string(item["DataTypeB"], "DataTypeB"),
            dest_data_type=_string(item["DestDataType"], "DestDataType"),
            compute_data_type=_string(item["ComputeDataType"], "ComputeDataType"),
            transpose_a=_boolean(item["TransposeA"], "TransposeA"),
            transpose_b=_boolean(item["TransposeB"], "TransposeB"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "OperationType": self.operation_type,
            "QuantDataType": self.quant_data_type,
            "DataTypeA": self.data_type_a,
            "DataTypeB": self.data_type_b,
            "DestDataType": self.dest_data_type,
            "ComputeDataType": self.compute_data_type,
            "TransposeA": self.transpose_a,
            "TransposeB": self.transpose_b,
        }


@dataclass(frozen=True)
class ProblemSize:
    """Exact GEMM coordinates interpreted by the selected ProblemType."""

    m: int
    n: int
    k: int

    _KEYS: ClassVar[frozenset[str]] = frozenset({"M", "N", "K"})

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _strict_mapping(value, name="ProblemSize", keys=cls._KEYS)
        return cls(
            m=_integer(item["M"], "M"),
            n=_integer(item["N"], "N"),
            k=_integer(item["K"], "K"),
        )

    def to_mapping(self) -> dict[str, int]:
        return {"M": self.m, "N": self.n, "K": self.k}


@dataclass(frozen=True)
class BackwardSolution:
    """Strict MMQ backward control. Fields name only implemented mechanisms."""

    kernel_language: str
    isa: tuple[int, int, int]
    wavefront_size: int
    work_group: tuple[int, int, int]
    matrix_instruction: tuple[int, ...]
    macro_tile0: int
    macro_tile1: int
    depth_u: int
    global_read_vector_width_a: int
    global_read_vector_width_b: int
    local_read_vector_width: int
    prefetch_global_read: int
    prefetch_local_read: int
    one_lds_buffer: int
    schedule_iter_alg: int
    store_priority_opt: bool
    num_elements_per_batch_store: int
    store_vector_width: int
    work_group_mapping: int
    transpose_lds: int
    lds_pad_b: int
    lds_block_size_per_pad_b: int
    lds_swizzle_chunk_b: int
    decoder_width: int
    prefetch_packed_weight: bool
    prefetch_packed_weight_next: bool
    packed_weight_lane_share: int
    q3_k_extraction: str
    q5_k_extraction: str
    q5_k_nibble_shift_hoist: bool
    q5_k_metadata_vector_load: bool
    q6_k_extraction: str
    q8_0_extraction: str

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
            "GlobalReadVectorWidthA",
            "GlobalReadVectorWidthB",
            "LocalReadVectorWidth",
            "PrefetchGlobalRead",
            "PrefetchLocalRead",
            "1LDSBuffer",
            "ScheduleIterAlg",
            "StorePriorityOpt",
            "NumElementsPerBatchStore",
            "StoreVectorWidth",
            "WorkGroupMapping",
            "TransposeLDS",
            "LdsPadB",
            "LdsBlockSizePerPadB",
            "LdsSwizzleChunkB",
            "DecoderWidth",
            "PrefetchPackedWeight",
            "PrefetchPackedWeightNext",
            "PackedWeightLaneShare",
            "Q3KExtraction",
            "Q5KExtraction",
            "Q5KNibbleShiftHoist",
            "Q5KMetadataVectorLoad",
            "Q6KExtraction",
            "Q8KExtraction",
        }
    )

    @classmethod
    def pilot(cls) -> Self:
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(32, 4, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 2, 8, 4, 1),
            macro_tile0=128,
            macro_tile1=128,
            depth_u=32,
            global_read_vector_width_a=16,
            global_read_vector_width_b=16,
            local_read_vector_width=16,
            prefetch_global_read=1,
            prefetch_local_read=1,
            one_lds_buffer=1,
            schedule_iter_alg=2,
            store_priority_opt=True,
            num_elements_per_batch_store=8,
            store_vector_width=1,
            work_group_mapping=1,
            transpose_lds=0,
            lds_pad_b=0,
            lds_block_size_per_pad_b=0,
            lds_swizzle_chunk_b=0,
            decoder_width=16,
            prefetch_packed_weight=True,
            prefetch_packed_weight_next=False,
            packed_weight_lane_share=1,
            q3_k_extraction="packed",
            q5_k_extraction="packed",
            q5_k_nibble_shift_hoist=False,
            q5_k_metadata_vector_load=False,
            q6_k_extraction="packed",
            q8_0_extraction="packed",
        )

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _strict_mapping(value, name="Solution", keys=cls._KEYS)
        return cls(
            kernel_language=_string(item["KernelLanguage"], "KernelLanguage"),
            isa=_integer_tuple(item["ISA"], "ISA", 3),
            wavefront_size=_integer(item["WavefrontSize"], "WavefrontSize"),
            work_group=_integer_tuple(item["WorkGroup"], "WorkGroup", 3),
            matrix_instruction=_integer_tuple(
                item["MatrixInstruction"], "MatrixInstruction", 9
            ),
            macro_tile0=_integer(item["MacroTile0"], "MacroTile0"),
            macro_tile1=_integer(item["MacroTile1"], "MacroTile1"),
            depth_u=_integer(item["DepthU"], "DepthU"),
            global_read_vector_width_a=_integer(
                item["GlobalReadVectorWidthA"], "GlobalReadVectorWidthA"
            ),
            global_read_vector_width_b=_integer(
                item["GlobalReadVectorWidthB"], "GlobalReadVectorWidthB"
            ),
            local_read_vector_width=_integer(
                item["LocalReadVectorWidth"], "LocalReadVectorWidth"
            ),
            prefetch_global_read=_integer(
                item["PrefetchGlobalRead"], "PrefetchGlobalRead"
            ),
            prefetch_local_read=_integer(
                item["PrefetchLocalRead"], "PrefetchLocalRead"
            ),
            one_lds_buffer=_integer(item["1LDSBuffer"], "1LDSBuffer"),
            schedule_iter_alg=_integer(item["ScheduleIterAlg"], "ScheduleIterAlg"),
            store_priority_opt=_boolean(item["StorePriorityOpt"], "StorePriorityOpt"),
            num_elements_per_batch_store=_integer(
                item["NumElementsPerBatchStore"], "NumElementsPerBatchStore"
            ),
            store_vector_width=_integer(item["StoreVectorWidth"], "StoreVectorWidth"),
            work_group_mapping=_integer(item["WorkGroupMapping"], "WorkGroupMapping"),
            transpose_lds=_integer(item["TransposeLDS"], "TransposeLDS"),
            lds_pad_b=_integer(item["LdsPadB"], "LdsPadB"),
            lds_block_size_per_pad_b=_integer(
                item["LdsBlockSizePerPadB"], "LdsBlockSizePerPadB"
            ),
            lds_swizzle_chunk_b=_integer(item["LdsSwizzleChunkB"], "LdsSwizzleChunkB"),
            decoder_width=_integer(item["DecoderWidth"], "DecoderWidth"),
            prefetch_packed_weight=_boolean(
                item["PrefetchPackedWeight"], "PrefetchPackedWeight"
            ),
            prefetch_packed_weight_next=_boolean(
                item["PrefetchPackedWeightNext"], "PrefetchPackedWeightNext"
            ),
            packed_weight_lane_share=_integer(
                item["PackedWeightLaneShare"], "PackedWeightLaneShare"
            ),
            q3_k_extraction=_string(item["Q3KExtraction"], "Q3KExtraction"),
            q5_k_extraction=_string(item["Q5KExtraction"], "Q5KExtraction"),
            q5_k_nibble_shift_hoist=_boolean(
                item["Q5KNibbleShiftHoist"], "Q5KNibbleShiftHoist"
            ),
            q5_k_metadata_vector_load=_boolean(
                item["Q5KMetadataVectorLoad"], "Q5KMetadataVectorLoad"
            ),
            q6_k_extraction=_string(item["Q6KExtraction"], "Q6KExtraction"),
            q8_0_extraction=_string(item["Q8KExtraction"], "Q8KExtraction"),
        )

    @property
    def num_threads(self) -> int:
        return self.work_group[0] * self.work_group[1] * self.work_group[2]

    @property
    def lds_num_bytes(self) -> int:
        if self.lds_block_size_per_pad_b:
            elements = self.macro_tile1 * self.depth_u
            bytes_unpadded = 2 * elements
            pad_periods = bytes_unpadded // self.lds_block_size_per_pad_b
            single_buffer = bytes_unpadded + 2 * self.lds_pad_b * pad_periods
        else:
            single_buffer = 2 * (self.depth_u + self.lds_pad_b) * self.macro_tile1
        return single_buffer * (2 if self.one_lds_buffer == 0 else 1)

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
            "GlobalReadVectorWidthA": self.global_read_vector_width_a,
            "GlobalReadVectorWidthB": self.global_read_vector_width_b,
            "LocalReadVectorWidth": self.local_read_vector_width,
            "PrefetchGlobalRead": self.prefetch_global_read,
            "PrefetchLocalRead": self.prefetch_local_read,
            "1LDSBuffer": self.one_lds_buffer,
            "ScheduleIterAlg": self.schedule_iter_alg,
            "StorePriorityOpt": self.store_priority_opt,
            "NumElementsPerBatchStore": self.num_elements_per_batch_store,
            "StoreVectorWidth": self.store_vector_width,
            "WorkGroupMapping": self.work_group_mapping,
            "TransposeLDS": self.transpose_lds,
            "LdsPadB": self.lds_pad_b,
            "LdsBlockSizePerPadB": self.lds_block_size_per_pad_b,
            "LdsSwizzleChunkB": self.lds_swizzle_chunk_b,
            "DecoderWidth": self.decoder_width,
            "PrefetchPackedWeight": self.prefetch_packed_weight,
            "PrefetchPackedWeightNext": self.prefetch_packed_weight_next,
            "PackedWeightLaneShare": self.packed_weight_lane_share,
            "Q3KExtraction": self.q3_k_extraction,
            "Q5KExtraction": self.q5_k_extraction,
            "Q5KNibbleShiftHoist": self.q5_k_nibble_shift_hoist,
            "Q5KMetadataVectorLoad": self.q5_k_metadata_vector_load,
            "Q6KExtraction": self.q6_k_extraction,
            "Q8KExtraction": self.q8_0_extraction,
        }


@dataclass(frozen=True)
class ForwardSolution:
    """Strict MMQ forward control. Fields name only implemented mechanisms."""

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
    operand_source: str
    weight_decode: str
    lds_address_hoist: str
    activation_addressing: str
    metadata_conversion: str
    scale_arithmetic: str
    output_store: str
    signed_weight: bool
    signed_activation: bool
    wmma_clamp: bool
    metadata_schedule: str = "Serialized"
    epilogue_tiles_ahead: int = 8
    epilogue_dependency_width: int = 1
    epilogue_priority: int = 0
    accumulator_initialization: str = "ScalarCopy"
    q6_epilogue_pipeline_scope: str = "StoreBatch"
    q6_dependency_delay_mode: str = "None"
    q6_global_read_cache_policy: str = "Default"
    q6_output_traversal: str = "OutputRoleGroupMajor"
    q6_stage_clustering: str = "StageDependencyOrder"
    q6_latency_policy: str = "SerializedDependencyDistance"
    q6_pressure_policy: str = "ExplicitRoleLifetime"
    q6_wait_policy: str = "ProducerFirstUse"
    q6_pairing_policy: str = "DependencyCompatibleDualIssue"

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
            "WeightDecode",
            "LdsAddressHoist",
            "ActivationAddressing",
            "MetadataConversion",
            "ScaleArithmetic",
            "OutputStore",
            "SignedWeight",
            "MetadataSchedule",
            "EpilogueTilesAhead",
            "EpilogueDependencyWidth",
            "EpiloguePriority",
            "AccumulatorInitialization",
            "SignedActivation",
            "WmmaClamp",
            "Q6EpiloguePipelineScope",
            "Q6DependencyDelayMode",
            "Q6GlobalReadCachePolicy",
            "Q6OutputTraversal",
            "Q6StageClustering",
            "Q6LatencyPolicy",
            "Q6PressurePolicy",
            "Q6WaitPolicy",
            "Q6PairingPolicy",
        }
    )

    @classmethod
    def q4_k_pilot(cls) -> Self:
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(32, 1, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 1, 1, 1),
            macro_tile0=16,
            macro_tile1=16,
            depth_u=32,
            activation_layout="F16_D4S4",
            activation_block_bytes=Q8_1_F16_D4S4_BLOCK_BYTES,
            packed_weight_block_bytes=QUANT_FORMATS["Q4_K"].block_bytes,
            operand_source="Global",
            weight_decode="DirectNibble",
            lds_address_hoist="None",
            activation_addressing="MultiplyAdd",
            metadata_conversion="Float32ThenFloat16",
            scale_arithmetic="FP16",
            output_store="BFloat16RNE",
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=True,
        )

    @classmethod
    def q3_k_hip_tiled_lds(cls) -> Self:
        """Return the first four-wave Q3_K forward research control."""
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(32, 4, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=128,
            macro_tile1=64,
            depth_u=16,
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            packed_weight_block_bytes=QUANT_FORMATS["Q3_K"].block_bytes,
            operand_source="Q3HipTiledLds",
            weight_decode="DirectQ3Signed",
            lds_address_hoist="Q3HalfTile",
            activation_addressing="MadU24",
            metadata_conversion="Float16DToFloat32Signed6Scale",
            scale_arithmetic="Int32ScaleF32",
            output_store="BFloat16RNEClause8",
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=False,
        )

    @classmethod
    def q4_k_decoded_weight_lds_retained(cls) -> Self:
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(128, 1, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=128,
            macro_tile1=64,
            depth_u=32,
            activation_layout="F16_D4S4",
            activation_block_bytes=Q8_1_F16_D4S4_BLOCK_BYTES,
            packed_weight_block_bytes=QUANT_FORMATS["Q4_K"].block_bytes,
            operand_source="DecodedWeightLdsBatch8",
            weight_decode="DirectNibble",
            lds_address_hoist="WeightMetadata",
            activation_addressing="MadU24",
            metadata_conversion="DirectFloat16Unsigned16",
            scale_arithmetic="FP16",
            output_store="BFloat16RNEClauseBatch8IncrementRows",
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=True,
        )

    @classmethod
    def q4_k_decoded_weight_lds_metadata_after_low_wmma(cls) -> Self:
        return replace(
            cls.q4_k_decoded_weight_lds_retained(),
            metadata_schedule="MetadataAfterLowWmma",
        )

    @classmethod
    def q5_k_decoded_weight_lds_retained(cls) -> Self:
        return replace(
            cls.q4_k_decoded_weight_lds_retained(),
            packed_weight_block_bytes=QUANT_FORMATS["Q5_K"].block_bytes,
            weight_decode="DirectNibbleHighBit",
        )

    @classmethod
    def q5_k_decoded_weight_lds_metadata_after_low_wmma(cls) -> Self:
        return replace(
            cls.q5_k_decoded_weight_lds_retained(),
            metadata_schedule="MetadataAfterLowWmma",
        )

    @classmethod
    def q5_k_decoded_weight_lds_extraction(
        cls,
        *,
        epilogue_tiles_ahead: int,
        epilogue_dependency_width: int,
        epilogue_priority: int,
        accumulator_initialization: str = "ScalarCopy",
    ) -> Self:
        if epilogue_tiles_ahead not in range(1, 9):
            raise ValueError("unsupported forward epilogue tiles-ahead")
        if epilogue_dependency_width not in range(1, 9):
            raise ValueError("unsupported forward epilogue dependency width")
        if epilogue_priority not in (0, 1, 2, 3):
            raise ValueError("unsupported forward epilogue priority")
        if accumulator_initialization not in ("ScalarCopy", "VopdPair"):
            raise ValueError("unsupported forward accumulator initialization")
        return replace(
            cls.q5_k_decoded_weight_lds_retained(),
            metadata_schedule="IndependentExtractionMetadataAfterLowWmma",
            epilogue_tiles_ahead=epilogue_tiles_ahead,
            epilogue_dependency_width=epilogue_dependency_width,
            epilogue_priority=epilogue_priority,
            accumulator_initialization=accumulator_initialization,
        )

    @classmethod
    def q6_k_structured_decoded(cls, *, macro_tile0: int) -> Self:
        if macro_tile0 not in (64, 128):
            raise ValueError("structured Q6_K forward implements MT64 or MT128")
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(32, 4, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=macro_tile0,
            macro_tile1=64,
            depth_u=32,
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            packed_weight_block_bytes=QUANT_FORMATS["Q6_K"].block_bytes,
            operand_source="Q6StructuredDecoded",
            weight_decode="DirectQ6Signed",
            lds_address_hoist="StructuredDecodeDot",
            activation_addressing="MadU24",
            metadata_conversion="Float16DToFloat32Signed8Scale",
            scale_arithmetic="Int32ScaleF32",
            output_store="BFloat16RNEClauseBatch8IncrementRows",
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=False,
            epilogue_dependency_width=2,
            q6_epilogue_pipeline_scope=(
                "StoreBatch" if macro_tile0 == 64 else "FullTile"
            ),
            q6_dependency_delay_mode="None" if macro_tile0 == 64 else "Explicit",
            q6_global_read_cache_policy="InvalidateL0",
        )

    @classmethod
    def q8_0_direct_global(cls) -> Self:
        """Return the fixed semantic Q8_0 direct-global control."""
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(32, 1, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 1, 1, 1),
            macro_tile0=16,
            macro_tile1=16,
            depth_u=32,
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            packed_weight_block_bytes=QUANT_FORMATS["Q8_0"].block_bytes,
            operand_source="Q8DirectGlobal",
            weight_decode="DirectSignedInt8",
            lds_address_hoist="None",
            activation_addressing="MultiplyAdd",
            metadata_conversion="Float16DToFloat32",
            scale_arithmetic="Int32ScaleF32",
            output_store="BFloat16RNE",
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=False,
        )

    @classmethod
    def q8_0_register_tiled(
        cls,
        *,
        wave_tile_m: int = 2,
        wave_tile_n: int = 2,
    ) -> Self:
        """Return a four-wave Q8_0 four-fragment register tile."""
        if wave_tile_m * wave_tile_n != 4:
            raise ValueError("Q8_0 register tile must own four fragments per wave")
        return replace(
            cls.q8_0_direct_global(),
            work_group=(32, 4, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=64 * wave_tile_m,
            macro_tile1=16 * wave_tile_n,
            operand_source="Q8RegisterTiled",
        )

    @classmethod
    def q8_0_hip_tiled_lds(cls) -> Self:
        """Return the HIP-shaped wave-N Q8_0 LDS research control."""
        return replace(
            cls.q8_0_direct_global(),
            work_group=(32, 4, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=128,
            macro_tile1=64,
            operand_source="Q8HipTiledLds",
            lds_address_hoist="HipTile",
            activation_addressing="MadU24",
            output_store="BFloat16RNEClause64",
        )

    @classmethod
    def q8_0_hip_tiled_lds_depth64(cls) -> Self:
        """Return the two-activation-plane Q8_0 LDS research control."""
        return replace(cls.q8_0_hip_tiled_lds(), depth_u=64)

    @classmethod
    def q8_0_small_m_tiled_lds(cls, *, macro_tile0: int) -> Self:
        """Return an exact M32 or M64 wave-N Q8_0 LDS research control."""
        if macro_tile0 not in (32, 64):
            raise ValueError("Q8_0 small-M LDS control requires MT32 or MT64")
        return replace(
            cls.q8_0_direct_global(),
            work_group=(32, 4, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=macro_tile0,
            macro_tile1=64,
            operand_source="Q8SmallMTiledLds",
            lds_address_hoist="SmallMTile",
            activation_addressing="MadU24",
            output_store="BFloat16RNEClauseTile",
        )

    @classmethod
    def q8_0_kv_tiled_lds(cls) -> Self:
        """Return the exact KV M2048 compact depth-32 LDS control."""
        return replace(
            cls.q8_0_small_m_tiled_lds(macro_tile0=64),
            lds_address_hoist="KvCompactTile",
        )

    @classmethod
    def q4_k_decoded_weight_lds_extraction(
        cls,
        *,
        epilogue_tiles_ahead: int,
        epilogue_dependency_width: int,
        epilogue_priority: int,
        metadata_after_low_wmma: bool = False,
    ) -> Self:
        if epilogue_tiles_ahead not in (1, 2, 4, 8):
            raise ValueError("unsupported forward epilogue tiles-ahead")
        if epilogue_dependency_width not in (1, 2, 4, 8):
            raise ValueError("unsupported forward epilogue dependency width")
        if epilogue_priority not in (0, 1, 2, 3):
            raise ValueError("unsupported forward epilogue priority")
        return replace(
            cls.q4_k_decoded_weight_lds_retained(),
            metadata_schedule=(
                "IndependentExtractionMetadataAfterLowWmma"
                if metadata_after_low_wmma
                else "IndependentExtraction"
            ),
            epilogue_tiles_ahead=epilogue_tiles_ahead,
            epilogue_dependency_width=epilogue_dependency_width,
            epilogue_priority=epilogue_priority,
        )

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        if isinstance(value, Mapping):
            defaults: dict[str, object] = {
                "AccumulatorInitialization": "ScalarCopy",
                "Q6EpiloguePipelineScope": "StoreBatch",
                "Q6DependencyDelayMode": "None",
                "Q6GlobalReadCachePolicy": "Default",
            }
            if value.get("OperandSource") != "Q6StructuredDecoded":
                defaults.update(
                    {
                        "Q6OutputTraversal": "OutputRoleGroupMajor",
                        "Q6StageClustering": "StageDependencyOrder",
                        "Q6LatencyPolicy": "SerializedDependencyDistance",
                        "Q6PressurePolicy": "ExplicitRoleLifetime",
                        "Q6WaitPolicy": "ProducerFirstUse",
                        "Q6PairingPolicy": "DependencyCompatibleDualIssue",
                    }
                )
            value = {**defaults, **value}
        item = _strict_mapping(value, name="Solution", keys=cls._KEYS)
        return cls(
            kernel_language=_string(item["KernelLanguage"], "KernelLanguage"),
            isa=_integer_tuple(item["ISA"], "ISA", 3),
            wavefront_size=_integer(item["WavefrontSize"], "WavefrontSize"),
            work_group=_integer_tuple(item["WorkGroup"], "WorkGroup", 3),
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
            operand_source=_string(item["OperandSource"], "OperandSource"),
            weight_decode=_string(item["WeightDecode"], "WeightDecode"),
            lds_address_hoist=_string(item["LdsAddressHoist"], "LdsAddressHoist"),
            activation_addressing=_string(
                item["ActivationAddressing"], "ActivationAddressing"
            ),
            metadata_conversion=_string(
                item["MetadataConversion"], "MetadataConversion"
            ),
            scale_arithmetic=_string(item["ScaleArithmetic"], "ScaleArithmetic"),
            output_store=_string(item["OutputStore"], "OutputStore"),
            signed_weight=_boolean(item["SignedWeight"], "SignedWeight"),
            signed_activation=_boolean(item["SignedActivation"], "SignedActivation"),
            wmma_clamp=_boolean(item["WmmaClamp"], "WmmaClamp"),
            metadata_schedule=_string(item["MetadataSchedule"], "MetadataSchedule"),
            epilogue_tiles_ahead=_integer(
                item["EpilogueTilesAhead"], "EpilogueTilesAhead"
            ),
            epilogue_dependency_width=_integer(
                item["EpilogueDependencyWidth"], "EpilogueDependencyWidth"
            ),
            epilogue_priority=_integer(item["EpiloguePriority"], "EpiloguePriority"),
            accumulator_initialization=_string(
                item["AccumulatorInitialization"], "AccumulatorInitialization"
            ),
            q6_epilogue_pipeline_scope=_string(
                item["Q6EpiloguePipelineScope"], "Q6EpiloguePipelineScope"
            ),
            q6_dependency_delay_mode=_string(
                item["Q6DependencyDelayMode"], "Q6DependencyDelayMode"
            ),
            q6_global_read_cache_policy=_string(
                item["Q6GlobalReadCachePolicy"], "Q6GlobalReadCachePolicy"
            ),
            q6_output_traversal=_string(item["Q6OutputTraversal"], "Q6OutputTraversal"),
            q6_stage_clustering=_string(item["Q6StageClustering"], "Q6StageClustering"),
            q6_latency_policy=_string(item["Q6LatencyPolicy"], "Q6LatencyPolicy"),
            q6_pressure_policy=_string(item["Q6PressurePolicy"], "Q6PressurePolicy"),
            q6_wait_policy=_string(item["Q6WaitPolicy"], "Q6WaitPolicy"),
            q6_pairing_policy=_string(item["Q6PairingPolicy"], "Q6PairingPolicy"),
        )

    @property
    def num_threads(self) -> int:
        return self.work_group[0] * self.work_group[1] * self.work_group[2]

    @property
    def lds_num_bytes(self) -> int:
        if self.operand_source == "Q6StructuredDecoded":
            output_rows_per_wave = self.macro_tile0 // 64
            return 19_456 + 9_472 * output_rows_per_wave
        if self.operand_source == "DecodedWeightLdsBatch8":
            return 38_400
        return 0

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
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
            "OperandSource": self.operand_source,
            "WeightDecode": self.weight_decode,
            "LdsAddressHoist": self.lds_address_hoist,
            "ActivationAddressing": self.activation_addressing,
            "MetadataConversion": self.metadata_conversion,
            "ScaleArithmetic": self.scale_arithmetic,
            "OutputStore": self.output_store,
            "SignedWeight": self.signed_weight,
            "SignedActivation": self.signed_activation,
            "WmmaClamp": self.wmma_clamp,
            "MetadataSchedule": self.metadata_schedule,
            "EpilogueTilesAhead": self.epilogue_tiles_ahead,
            "EpilogueDependencyWidth": self.epilogue_dependency_width,
            "EpiloguePriority": self.epilogue_priority,
        }
        if self.accumulator_initialization != "ScalarCopy":
            mapping["AccumulatorInitialization"] = self.accumulator_initialization
        if self.q6_epilogue_pipeline_scope != "StoreBatch":
            mapping["Q6EpiloguePipelineScope"] = self.q6_epilogue_pipeline_scope
        if self.q6_dependency_delay_mode != "None":
            mapping["Q6DependencyDelayMode"] = self.q6_dependency_delay_mode
        if self.q6_global_read_cache_policy != "Default":
            mapping["Q6GlobalReadCachePolicy"] = self.q6_global_read_cache_policy
        q6_policy_fields = (
            (
                "Q6OutputTraversal",
                self.q6_output_traversal,
                "OutputRoleGroupMajor",
            ),
            (
                "Q6StageClustering",
                self.q6_stage_clustering,
                "StageDependencyOrder",
            ),
            (
                "Q6LatencyPolicy",
                self.q6_latency_policy,
                "SerializedDependencyDistance",
            ),
            (
                "Q6PressurePolicy",
                self.q6_pressure_policy,
                "ExplicitRoleLifetime",
            ),
            ("Q6WaitPolicy", self.q6_wait_policy, "ProducerFirstUse"),
            (
                "Q6PairingPolicy",
                self.q6_pairing_policy,
                "DependencyCompatibleDualIssue",
            ),
        )
        for key, value, default in q6_policy_fields:
            if self.operand_source == "Q6StructuredDecoded" or value != default:
                mapping[key] = value
        return mapping


@dataclass(frozen=True)
class SolutionKey:
    problem_type: ProblemType
    problem_size: ProblemSize
    solution: BackwardSolution | ForwardSolution

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {"ProblemType", "ProblemSize", "Solution"}
    )

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _strict_mapping(value, name="SolutionKey", keys=cls._KEYS)
        problem_type = ProblemType.from_mapping(item["ProblemType"])
        solution_type = (
            ForwardSolution
            if problem_type.operation_type == "MMQForward"
            else BackwardSolution
        )
        return cls(
            problem_type=problem_type,
            problem_size=ProblemSize.from_mapping(item["ProblemSize"]),
            solution=solution_type.from_mapping(item["Solution"]),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> Self:
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_mapping(value)

    def to_mapping(self) -> dict[str, object]:
        return {
            "ProblemType": self.problem_type.to_mapping(),
            "ProblemSize": self.problem_size.to_mapping(),
            "Solution": self.solution.to_mapping(),
        }

    @property
    def hash(self) -> str:
        canonical = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"ggsol_{digest[:16]}"

    @property
    def kernel_name(self) -> str:
        size = self.problem_size
        quant_type = self.problem_type.quant_data_type.lower()
        operation = (
            "mmq_fwd" if self.problem_type.operation_type == "MMQForward" else "mmq_bwd"
        )
        return (
            f"torch_ggml_ops_ggtensile_gfx1151_v1_{operation}_"
            f"{quant_type}_"
            f"m{size.m}_n{size.n}_k{size.k}_{self.hash[6:]}"
        )


@dataclass(frozen=True)
class KernelArtifact:
    solution_key: SolutionKey
    kernel_name: str
    assembly_path: Path
    object_path: Path | None
    code_object_path: Path | None
    assembly_sha256: str
    vgpr_count: int
    sgpr_count: int
    lds_num_bytes: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "SolutionKey": self.solution_key.to_mapping(),
            "SolutionHash": self.solution_key.hash,
            "KernelName": self.kernel_name,
            "AssemblyPath": str(self.assembly_path),
            "ObjectPath": str(self.object_path) if self.object_path else None,
            "CodeObjectPath": (
                str(self.code_object_path) if self.code_object_path else None
            ),
            "AssemblySHA256": self.assembly_sha256,
            "NumVgpr": self.vgpr_count,
            "NumSgpr": self.sgpr_count,
            "LdsNumBytes": self.lds_num_bytes,
        }
