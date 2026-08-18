import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
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


def _strict_mapping_optional(
    value: object,
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{name} must be a mapping")
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise SchemaError(f"{name} keys must be strings")
        normalized[key] = item
    actual = set(normalized)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise SchemaError(f"invalid {name}: {', '.join(details)}")
    return normalized


def _canonical_value(value: object) -> object:
    """Project typed records without null inactive-policy sentinels."""
    if isinstance(value, Enum):
        return value.value if isinstance(value.value, str) else value.name
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, object] = {}
        for field in fields(value):
            item = getattr(value, field.name)
            if item is None:
                continue
            projected = _canonical_value(item)
            if isinstance(projected, dict) and not projected:
                continue
            result[field.name] = projected
        return result
    if isinstance(value, tuple | list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
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
    def grouped_mmq_backward(cls, quant_data_type: str) -> Self:
        if quant_data_type != "Q4_K":
            raise ValueError(
                f"unsupported grouped MMQ backward quant type {quant_data_type!r}"
            )
        ordinary = cls.mmq_backward(quant_data_type)
        return replace(ordinary, operation_type="GroupedMMQBackward")

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


@dataclass(frozen=True)
class ProblemSize:
    """Exact GEMM coordinates interpreted by the selected ProblemType."""

    m: int
    n: int
    k: int

    def to_mapping(self) -> dict[str, int]:
        return {"M": self.m, "N": self.n, "K": self.k}

    @classmethod
    def from_canonical_mapping(cls, value: object) -> Self:
        item = _strict_mapping(
            value,
            name="Problem",
            keys=frozenset({"m", "n", "k"}),
        )
        problem_size = cls(
            m=_integer(item["m"], "Problem.m"),
            n=_integer(item["n"], "Problem.n"),
            k=_integer(item["k"], "Problem.k"),
        )
        if min(problem_size.m, problem_size.n, problem_size.k) <= 0:
            raise SchemaError("Problem dimensions must be positive")
        return problem_size

    def to_canonical_mapping(self) -> dict[str, int]:
        return {"m": self.m, "n": self.n, "k": self.k}


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
    q3_k_pairing: str
    q4_k_decode_schedule: str
    q5_k_extraction: str
    q5_k_nibble_shift_hoist: bool
    q5_k_metadata_vector_load: bool
    q6_k_extraction: str
    q8_0_extraction: str

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
            q3_k_pairing="Inactive",
            q4_k_decode_schedule="Serial",
            q5_k_extraction="packed",
            q5_k_nibble_shift_hoist=False,
            q5_k_metadata_vector_load=False,
            q6_k_extraction="packed",
            q8_0_extraction="packed",
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


@dataclass(frozen=True)
class GroupedBackwardSolution:
    """Grouped ownership wrapped around one reusable backward compute spec."""

    compute: BackwardSolution
    route_ownership: str
    row_tail: str

    @classmethod
    def pilot(cls) -> Self:
        return cls(
            compute=BackwardSolution.pilot(),
            route_ownership="SerialRoutes",
            row_tail="Masked",
        )

    @property
    def num_threads(self) -> int:
        return self.compute.num_threads

    @property
    def lds_num_bytes(self) -> int:
        return self.compute.lds_num_bytes

    @property
    def matrix_instruction(self) -> tuple[int, ...]:
        return self.compute.matrix_instruction

    @property
    def macro_tile0(self) -> int:
        return self.compute.macro_tile0

    @property
    def macro_tile1(self) -> int:
        return self.compute.macro_tile1

    @property
    def depth_u(self) -> int:
        return self.compute.depth_u


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
    q6_physical_plan: str = "CanonicalRegisterRoles"

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
    def q3_k_full_weight_tiled_lds(cls) -> Self:
        """Return the typed full-256-value Q3_K LDS mechanism."""
        return replace(
            cls.q3_k_hip_tiled_lds(),
            operand_source="Q3FullWeightTiledLds",
            lds_address_hoist="Q3FullTile336",
            activation_addressing="ScalarPlaneBase",
            metadata_conversion="Float16DToFloat32Signed6ScaleShared",
            scale_arithmetic="Int32ScaleF32",
            metadata_schedule="Q3FullTileSharedDecode",
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
        if macro_tile0 not in (64, 128, 256):
            raise ValueError("structured Q6_K forward implements MT64, MT128, or MT256")
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(32, 8 if macro_tile0 == 256 else 4, 1),
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
    def q8_0_compact_depth32_tiled_lds(cls, *, macro_tile0: int = 128) -> Self:
        """Return the typed compact depth-32 wave-N LDS control."""
        if macro_tile0 == 128:
            base = cls.q8_0_hip_tiled_lds()
        elif macro_tile0 in (32, 64):
            base = cls.q8_0_small_m_tiled_lds(macro_tile0=macro_tile0)
        else:
            raise ValueError("Q8 compact depth-32 control requires MT32/MT64/MT128")
        return replace(base, lds_address_hoist="CompactDepth32WeightRows")

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
        """Return the canonical M64 compact depth-32 LDS control."""
        return cls.q8_0_compact_depth32_tiled_lds(
            macro_tile0=64,
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
        if self.operand_source == "Q3FullWeightTiledLds":
            return 39_936
        return 0


@dataclass(frozen=True)
class SolutionKey:
    problem_type: ProblemType
    problem_size: ProblemSize
    solution: BackwardSolution | ForwardSolution | GroupedBackwardSolution

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {"ArtifactKind", "KernelFamily", "ProblemContract", "Problem", "KernelSpec"}
    )

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _strict_mapping(value, name="SolutionKey", keys=cls._KEYS)
        if item["ArtifactKind"] != "ExactKernel":
            raise SchemaError("SolutionKey ArtifactKind must be ExactKernel")
        family = _string(item["KernelFamily"], "KernelFamily")
        problem_size = ProblemSize.from_canonical_mapping(item["Problem"])
        if family == "OrdinaryForward":
            from .mmq_fwd_spec import ForwardKernelSpec, ForwardProblemContract

            contract = ForwardProblemContract.from_mapping(item["ProblemContract"])
            spec = ForwardKernelSpec.from_mapping(item["KernelSpec"])
            solution = spec.to_solution(contract)
            if ForwardKernelSpec.from_solution(solution) != spec:
                raise SchemaError("ForwardKernelSpec does not round-trip canonically")
            problem_type = ProblemType.mmq_forward(contract.quant_type)
        elif family == "OrdinaryBackward":
            from .mmq_bwd_spec import BackwardKernelSpec, BackwardProblemContract

            contract = BackwardProblemContract.from_mapping(
                item["ProblemContract"], problem_size
            )
            spec = BackwardKernelSpec.from_mapping(
                item["KernelSpec"], contract.quant_type
            )
            solution = spec.to_solution(contract)
            if BackwardKernelSpec.from_solution(solution) != spec:
                raise SchemaError("BackwardKernelSpec does not round-trip canonically")
            problem_type = ProblemType.mmq_backward(contract.quant_type)
        elif family == "GroupedBackward":
            from .grouped_mmq_bwd_spec import (
                GroupedBackwardKernelSpec,
                GroupedBackwardProblemContract,
            )

            contract = GroupedBackwardProblemContract.from_mapping(
                item["ProblemContract"], problem_size
            )
            spec = GroupedBackwardKernelSpec.from_mapping(
                item["KernelSpec"], contract.quant_type
            )
            solution = spec.to_solution(contract)
            if GroupedBackwardKernelSpec.from_solution(solution) != spec:
                raise SchemaError(
                    "GroupedBackwardKernelSpec does not round-trip canonically"
                )
            problem_type = ProblemType.grouped_mmq_backward(contract.quant_type)
        else:
            raise SchemaError(f"unsupported KernelFamily {family!r}")
        return cls(problem_type, problem_size, solution)

    @classmethod
    def from_json_file(cls, path: Path) -> Self:
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_mapping(value)

    def to_mapping(self) -> dict[str, object]:
        if isinstance(self.solution, GroupedBackwardSolution):
            from .grouped_mmq_bwd_spec import (
                GroupedBackwardKernelSpec,
                GroupedBackwardProblemContract,
            )

            family = "GroupedBackward"
            contract = GroupedBackwardProblemContract.from_solution_key(self)
            spec_mapping = GroupedBackwardKernelSpec.from_solution(
                self.solution
            ).to_mapping(contract.quant_type)
        elif isinstance(self.solution, ForwardSolution):
            from .mmq_fwd_spec import ForwardKernelSpec, ForwardProblemContract

            family = "OrdinaryForward"
            contract = ForwardProblemContract.from_solution(
                self.problem_type.quant_data_type, self.solution
            )
            spec_mapping = ForwardKernelSpec.from_solution(self.solution).to_mapping()
        else:
            from .mmq_bwd_spec import BackwardKernelSpec, BackwardProblemContract

            family = "OrdinaryBackward"
            contract = BackwardProblemContract.from_solution_key(self)
            spec_mapping = BackwardKernelSpec.from_solution(self.solution).to_mapping(
                contract.quant_type
            )
        return {
            "ArtifactKind": "ExactKernel",
            "KernelFamily": family,
            "ProblemContract": contract.to_mapping(),
            "Problem": self.problem_size.to_canonical_mapping(),
            "KernelSpec": spec_mapping,
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
        operation = {
            "MMQForward": "mmq_fwd",
            "MMQBackward": "mmq_bwd",
            "GroupedMMQBackward": "grouped_mmq_bwd",
        }[self.problem_type.operation_type]
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
