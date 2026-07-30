import hashlib
import json
from dataclasses import dataclass
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
        if quant_data_type not in {"Q3_K", "Q4_K", "Q5_K"}:
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
    """Exact GEMM coordinates: M=rows, N=in_features, K=out_features."""

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
    q3_k_metadata_vector_load: bool
    q5_k_extraction: str
    q5_k_nibble_shift_hoist: bool
    q5_k_metadata_vector_load: bool

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
            "Q3KMetadataVectorLoad",
            "Q5KExtraction",
            "Q5KNibbleShiftHoist",
            "Q5KMetadataVectorLoad",
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
            q3_k_metadata_vector_load=False,
            q5_k_extraction="packed",
            q5_k_nibble_shift_hoist=False,
            q5_k_metadata_vector_load=False,
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
            q3_k_metadata_vector_load=_boolean(
                item["Q3KMetadataVectorLoad"], "Q3KMetadataVectorLoad"
            ),
            q5_k_extraction=_string(item["Q5KExtraction"], "Q5KExtraction"),
            q5_k_nibble_shift_hoist=_boolean(
                item["Q5KNibbleShiftHoist"], "Q5KNibbleShiftHoist"
            ),
            q5_k_metadata_vector_load=_boolean(
                item["Q5KMetadataVectorLoad"], "Q5KMetadataVectorLoad"
            ),
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
            single_buffer = 2 * self.depth_u * (self.macro_tile1 + self.lds_pad_b)
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
            "Q3KMetadataVectorLoad": self.q3_k_metadata_vector_load,
            "Q5KExtraction": self.q5_k_extraction,
            "Q5KNibbleShiftHoist": self.q5_k_nibble_shift_hoist,
            "Q5KMetadataVectorLoad": self.q5_k_metadata_vector_load,
        }


@dataclass(frozen=True)
class SolutionKey:
    problem_type: ProblemType
    problem_size: ProblemSize
    solution: Solution

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {"ProblemType", "ProblemSize", "Solution"}
    )

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _strict_mapping(value, name="SolutionKey", keys=cls._KEYS)
        return cls(
            problem_type=ProblemType.from_mapping(item["ProblemType"]),
            problem_size=ProblemSize.from_mapping(item["ProblemSize"]),
            solution=Solution.from_mapping(item["Solution"]),
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
        return (
            "torch_ggml_ops_ggtensile_gfx1151_v1_dense_bwd_"
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
