import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Mapping, Self


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
    actual = {str(key) for key in value}
    missing = sorted(keys - actual)
    unknown = sorted(actual - keys)
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
    if type(value) is not list or len(value) != length:
        raise SchemaError(f"{name} must be a {length}-element list")
    for index, item in enumerate(value):
        if type(item) is not int:
            raise SchemaError(f"{name}[{index}] must be int, not {type(item).__name__}")
    return tuple(value)


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
    def dense_mmq_backward(cls, quant_data_type: str) -> Self:
        if quant_data_type not in {"Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"}:
            raise ValueError(
                f"unsupported dense MMQ backward quant type {quant_data_type!r}"
            )
        return cls(
            operation_type="DenseMMQBackward",
            quant_data_type=quant_data_type,
            data_type_a="BFloat16",
            data_type_b=quant_data_type,
            dest_data_type="BFloat16",
            compute_data_type="Float",
            transpose_a=False,
            transpose_b=False,
        )

    @classmethod
    def dense_mmq_backward_q4_k(cls) -> Self:
        return cls.dense_mmq_backward("Q4_K")

    @classmethod
    def dense_mmq_backward_q5_k(cls) -> Self:
        return cls.dense_mmq_backward("Q5_K")

    @classmethod
    def dense_mmq_backward_q6_k(cls) -> Self:
        return cls.dense_mmq_backward("Q6_K")

    @classmethod
    def dense_mmq_backward_q8_0(cls) -> Self:
        return cls.dense_mmq_backward("Q8_0")

    @classmethod
    def dense_mmq_forward_q4_k(cls) -> Self:
        return cls(
            operation_type="DenseMMQForward",
            quant_data_type="Q4_K",
            data_type_a="Q8_1_DS4",
            data_type_b="Q4_K",
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
class Solution:
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
class DenseForwardSolution:
    """Strict Q4_K forward control; fields name only implemented mechanisms."""

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
            "SignedActivation",
            "WmmaClamp",
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
            activation_layout="Q8_1_DS4",
            activation_block_bytes=144,
            packed_weight_block_bytes=144,
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
    def q4_k_wave_reuse(cls) -> Self:
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(128, 1, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=128,
            macro_tile1=64,
            depth_u=32,
            activation_layout="Q8_1_DS4",
            activation_block_bytes=144,
            packed_weight_block_bytes=144,
            operand_source="GlobalWaveReuse",
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
    def q4_k_wave_batch4(cls) -> Self:
        return replace(cls.q4_k_wave_reuse(), operand_source="GlobalWaveBatch4")

    @classmethod
    def q4_k_hip_staged(cls) -> Self:
        return replace(cls.q4_k_wave_reuse(), operand_source="HipStagedBatch8")

    @classmethod
    def q4_k_hip_decoded_staged(cls) -> Self:
        return replace(cls.q4_k_wave_reuse(), operand_source="HipDecodedStagedBatch8")

    @classmethod
    def q4_k_hip_decoded_staged_retained(cls) -> Self:
        return replace(
            cls.q4_k_hip_decoded_staged(),
            lds_address_hoist="WeightMetadata",
            activation_addressing="MadU24",
            metadata_conversion="DirectFloat16Unsigned16",
            output_store="BFloat16RNEClauseBatch8IncrementRows",
        )

    @classmethod
    def q4_k_hip_decoded_staged_metadata_after_low_wmma(cls) -> Self:
        return replace(
            cls.q4_k_hip_decoded_staged_retained(),
            metadata_schedule="MetadataAfterLowWmma",
        )

    @classmethod
    def q4_k_hip_decoded_staged_independent_extraction_metadata_after_low_wmma(
        cls,
    ) -> Self:
        return cls.q4_k_hip_decoded_staged_extraction(
            epilogue_tiles_ahead=8,
            epilogue_dependency_width=1,
            epilogue_priority=0,
            metadata_after_low_wmma=True,
        )

    @classmethod
    def q4_k_hip_decoded_staged_extraction(
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
            cls.q4_k_hip_decoded_staged_retained(),
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
    def q4_k_hip_decoded_staged_shared_down_m8192(cls) -> Self:
        return cls.q4_k_hip_decoded_staged_extraction(
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=4,
            epilogue_priority=2,
        )

    @classmethod
    def q4_k_hip_decoded_staged_shared_down_m32768(cls) -> Self:
        return cls.q4_k_hip_decoded_staged_extraction(
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
        )

    @classmethod
    def q4_k_hip_decoded_staged_shared_down_m8192_metadata_after_low_wmma(
        cls,
    ) -> Self:
        return cls.q4_k_hip_decoded_staged_extraction(
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=4,
            epilogue_priority=2,
            metadata_after_low_wmma=True,
        )

    @classmethod
    def q4_k_hip_decoded_staged_shared_down_m32768_metadata_after_low_wmma(
        cls,
    ) -> Self:
        return cls.q4_k_hip_decoded_staged_extraction(
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
            metadata_after_low_wmma=True,
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
        )

    @property
    def num_threads(self) -> int:
        return self.work_group[0] * self.work_group[1] * self.work_group[2]

    @property
    def lds_num_bytes(self) -> int:
        if self.operand_source == "HipDecodedStagedBatch8":
            return 38_400
        if self.operand_source == "HipStagedBatch8":
            return 18_432 + 8_192
        return 0

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


@dataclass(frozen=True)
class SolutionKey:
    problem_type: ProblemType
    problem_size: ProblemSize
    solution: Solution | DenseForwardSolution

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {"ProblemType", "ProblemSize", "Solution"}
    )

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _strict_mapping(value, name="SolutionKey", keys=cls._KEYS)
        problem_type = ProblemType.from_mapping(item["ProblemType"])
        solution_type = (
            DenseForwardSolution
            if problem_type.operation_type == "DenseMMQForward"
            else Solution
        )
        return cls(
            problem_type=problem_type,
            problem_size=ProblemSize.from_mapping(item["ProblemSize"]),
            solution=solution_type.from_mapping(item["Solution"]),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> Self:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SchemaError(f"invalid JSON in {path}: {error}") from error
        return cls.from_mapping(value)

    def to_mapping(self) -> dict[str, object]:
        return {
            "ProblemType": self.problem_type.to_mapping(),
            "ProblemSize": self.problem_size.to_mapping(),
            "Solution": self.solution.to_mapping(),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))

    @property
    def hash(self) -> str:
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"ggsol_{digest[:16]}"

    @property
    def kernel_name(self) -> str:
        size = self.problem_size
        quant_type = self.problem_type.quant_data_type.lower()
        operation = (
            "dense_fwd"
            if self.problem_type.operation_type == "DenseMMQForward"
            else "dense_bwd"
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

    def to_mapping(self) -> dict[str, Any]:
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
