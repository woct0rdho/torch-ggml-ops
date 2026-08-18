"""Typed problem and solution state for MMQ backward assembly lowering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TypeVar, cast

from .model import (
    BackwardSolution,
    ProblemSize,
    SchemaError,
    SolutionKey,
    _integer,
    _integer_tuple,
    _strict_mapping,
    _strict_mapping_optional,
    _string,
)
from .quant_formats import BACKWARD_QUANT_FORMATS, QuantFormat

EnumT = TypeVar("EnumT", bound=Enum)


def _serialized_enum(enum_type: type[EnumT], value: object, name: str) -> EnumT:
    serialized = _string(value, name)
    try:
        if issubclass(enum_type, IntEnum):
            return enum_type[serialized]
        return enum_type(serialized)
    except (KeyError, ValueError):
        raise SchemaError(f"invalid {name}: {serialized!r}") from None


class BackwardQ2DecodeSchedule(str, Enum):
    Serial = "Serial"
    DependencyBatch4 = "DependencyBatch4"

    @classmethod
    def try_from_serialized(cls, value: str) -> BackwardQ2DecodeSchedule | None:
        try:
            return cls(value)
        except ValueError:
            return None

    @property
    def dependency_width(self) -> int:
        return 4 if self is BackwardQ2DecodeSchedule.DependencyBatch4 else 1


class BackwardQ3Pairing(str, Enum):
    Inactive = "Inactive"
    Partial = "Partial"
    Full = "Full"

    @classmethod
    def try_from_serialized(cls, value: str) -> BackwardQ3Pairing | None:
        try:
            return cls(value)
        except ValueError:
            return None


class BackwardQ4DecodeSchedule(str, Enum):
    Serial = "Serial"
    DependencyBatch4 = "DependencyBatch4"

    @classmethod
    def try_from_serialized(cls, value: str) -> BackwardQ4DecodeSchedule | None:
        try:
            return cls(value)
        except ValueError:
            return None

    @property
    def dependency_width(self) -> int:
        return 4 if self is BackwardQ4DecodeSchedule.DependencyBatch4 else 1


class BackwardExtraction(str, Enum):
    packed = "packed"
    packed_vopd = "packed_vopd"
    scalar = "scalar"

    @classmethod
    def try_from_serialized(cls, value: str) -> BackwardExtraction | None:
        try:
            return cls(value)
        except ValueError:
            return None


class BackwardScheduleIterAlg(IntEnum):
    SIA2 = 2
    SIA3 = 3
    SIA4 = 4
    SIA5 = 5

    @classmethod
    def try_from_serialized(cls, value: int) -> BackwardScheduleIterAlg | None:
        try:
            return cls(value)
        except ValueError:
            return None

    @property
    def prefetches_a(self) -> bool:
        return self in (BackwardScheduleIterAlg.SIA4, BackwardScheduleIterAlg.SIA5)

    @property
    def interleaves_wmma_waits(self) -> bool:
        return self in (BackwardScheduleIterAlg.SIA3, BackwardScheduleIterAlg.SIA5)

    @property
    def uses_sia4_waits(self) -> bool:
        return self is BackwardScheduleIterAlg.SIA4

    def supports_global_read_prefetch(self, count: int) -> bool:
        return count == 1 or (count == 2 and self.prefetches_a)

    def supports_local_read_prefetch(self, count: int) -> bool:
        return count == 1 or (count == 2 and self is BackwardScheduleIterAlg.SIA3)


class BackwardLdsBuffering(IntEnum):
    Double = 0
    Single = 1

    @classmethod
    def try_from_serialized(cls, value: int) -> BackwardLdsBuffering | None:
        try:
            return cls(value)
        except ValueError:
            return None

    @property
    def buffer_count(self) -> int:
        return 2 if self is BackwardLdsBuffering.Double else 1


class BackwardPackedWeightPrefetch(str, Enum):
    Disabled = "Disabled"
    CurrentTile = "CurrentTile"
    NextTile = "NextTile"
    CurrentAndNextTile = "CurrentAndNextTile"

    @classmethod
    def from_solution(cls, solution: BackwardSolution) -> BackwardPackedWeightPrefetch:
        if solution.prefetch_packed_weight:
            return (
                cls.CurrentAndNextTile
                if solution.prefetch_packed_weight_next
                else cls.CurrentTile
            )
        return cls.NextTile if solution.prefetch_packed_weight_next else cls.Disabled

    @property
    def includes_next_tile(self) -> bool:
        return self in (
            BackwardPackedWeightPrefetch.NextTile,
            BackwardPackedWeightPrefetch.CurrentAndNextTile,
        )


class BackwardQ5NibbleShift(str, Enum):
    Inline = "Inline"
    Hoisted = "Hoisted"


class BackwardQ5MetadataLoad(str, Enum):
    Scalar = "Scalar"
    Vector = "Vector"


class BackwardStorePriority(str, Enum):
    Default = "Default"
    Raised = "Raised"


class BackwardPackedRowAddress(str, Enum):
    Block32 = "Block32"
    SuperBlock256 = "SuperBlock256"


@dataclass(frozen=True)
class BackwardDecoderCapability:
    payload_register_count: int
    packed_loads_per_row: int
    scalar_packed_loads_per_row: int | None
    vector_metadata_packed_loads_per_row: int | None
    q3_packed_temporary_count: int | None
    q6_packed_vopd_temporary_base: int | None
    q8_packed_vopd_temporary_base: int | None
    row_address: BackwardPackedRowAddress

    def packed_load_count(self, decode: BackwardDecodeSpec, rows: int) -> int:
        per_row = self.packed_loads_per_row
        if (
            self.row_address is BackwardPackedRowAddress.Block32
            and self.scalar_packed_loads_per_row is not None
            and decode.q8.extraction is BackwardExtraction.scalar
        ):
            per_row = self.scalar_packed_loads_per_row
        elif (
            self.vector_metadata_packed_loads_per_row is not None
            and decode.q5.vector_metadata_load
        ):
            per_row = self.vector_metadata_packed_loads_per_row
        return per_row * rows

    def temporary_register_count(self, decode: BackwardDecodeSpec, rows: int) -> int:
        candidates = [7, 3 + 2 * rows]
        if (
            self.q3_packed_temporary_count is not None
            and decode.q3.extraction is BackwardExtraction.packed
        ):
            candidates.append(self.q3_packed_temporary_count)
        if (
            self.q6_packed_vopd_temporary_base is not None
            and decode.q6.extraction is BackwardExtraction.packed_vopd
        ):
            candidates.append(self.q6_packed_vopd_temporary_base + 2 * rows)
        if (
            self.q8_packed_vopd_temporary_base is not None
            and decode.q8.extraction is BackwardExtraction.packed_vopd
        ):
            candidates.append(self.q8_packed_vopd_temporary_base + 2 * rows)
        return max(candidates)


@dataclass(frozen=True)
class BackwardAddressCapability:
    extended_quant_address_register_count: int

    def uses_extended_a(self, schedule: BackwardScheduleIterAlg, m_tiles: int) -> bool:
        return schedule.prefetches_a and m_tiles > 2

    def register_count(self, schedule: BackwardScheduleIterAlg, m_tiles: int) -> int:
        if not self.uses_extended_a(schedule, m_tiles):
            return 8
        return 6 + m_tiles + self.extended_quant_address_register_count


@dataclass(frozen=True)
class BackwardQuantRegisterShape:
    dm_registers_per_row: int
    scale_registers_per_row: int
    scale_alias_offset: int | None
    vector_metadata_shape: BackwardQuantRegisterShape | None = None

    def for_decode(self, decode: BackwardDecodeSpec) -> BackwardQuantRegisterShape:
        if decode.q5.vector_metadata_load and self.vector_metadata_shape is not None:
            return self.vector_metadata_shape
        return self


@dataclass(frozen=True)
class BackwardMechanismContract:
    lane_share_values: frozenset[int]
    padded_depth_values: frozenset[int]
    pipeline_n_values: frozenset[int]
    pipeline_schedule_iter_algs: frozenset[BackwardScheduleIterAlg]
    pipeline_depth_values: frozenset[int]
    next_packed_depth_values: frozenset[int]
    decoder: BackwardDecoderCapability
    address: BackwardAddressCapability
    quant_register_shape: BackwardQuantRegisterShape

    def supports_padded_depth(self, depth: int) -> bool:
        return depth in self.padded_depth_values

    def supports_pipeline_n(self, macro_tile1: int) -> bool:
        return macro_tile1 in self.pipeline_n_values

    def supports_pipeline_schedule(
        self, schedule: BackwardScheduleIterAlg | None
    ) -> bool:
        return schedule is not None and schedule in self.pipeline_schedule_iter_algs

    def supports_pipeline_depth(self, depth: int) -> bool:
        return depth in self.pipeline_depth_values

    def supports_next_packed_depth(self, depth: int) -> bool:
        return depth in self.next_packed_depth_values

    @property
    def extended_quant_address_state(self) -> bool:
        return self.address.extended_quant_address_register_count > 0


def backward_mechanism_contract(quant_type: str) -> BackwardMechanismContract:
    if quant_type not in BACKWARD_QUANT_FORMATS:
        raise ValueError(f"unknown backward quant mechanism: {quant_type}") from None
    return BackwardMechanismContract(
        lane_share_values=(
            frozenset({1, 2})
            if quant_type in ("Q4_K", "Q5_K", "Q6_K")
            else frozenset({1})
        ),
        padded_depth_values=(
            frozenset({32, 64})
            if quant_type in ("Q3_K", "Q6_K", "Q8_0")
            else frozenset({32})
        ),
        pipeline_n_values=(
            frozenset({64, 128})
            if quant_type in ("Q2_K", "Q4_K", "Q6_K")
            else frozenset({128})
        ),
        pipeline_schedule_iter_algs=(
            frozenset(BackwardScheduleIterAlg)
            if quant_type == "Q5_K"
            else frozenset(
                {
                    BackwardScheduleIterAlg.SIA2,
                    BackwardScheduleIterAlg.SIA4,
                    BackwardScheduleIterAlg.SIA5,
                }
            )
        ),
        pipeline_depth_values=(
            frozenset({32, 64}) if quant_type == "Q5_K" else frozenset({32})
        ),
        next_packed_depth_values=(
            frozenset({32, 64}) if quant_type == "Q6_K" else frozenset({32})
        ),
        decoder=BackwardDecoderCapability(
            payload_register_count=(
                4 if quant_type in ("Q2_K", "Q4_K", "Q8_0", "IQ2_S") else 8
            ),
            packed_loads_per_row=(
                3
                if quant_type == "Q2_K"
                else 5
                if quant_type in ("Q3_K", "Q4_K")
                else 4
                if quant_type == "Q6_K"
                else 2
                if quant_type == "Q8_0"
                else 5
                if quant_type == "IQ2_S"
                else 6
            ),
            scalar_packed_loads_per_row=5 if quant_type == "Q8_0" else None,
            vector_metadata_packed_loads_per_row=3 if quant_type == "Q5_K" else None,
            q3_packed_temporary_count=11 if quant_type == "Q3_K" else None,
            q6_packed_vopd_temporary_base=7 if quant_type == "Q6_K" else None,
            q8_packed_vopd_temporary_base=5 if quant_type == "Q8_0" else None,
            row_address=(
                BackwardPackedRowAddress.Block32
                if quant_type == "Q8_0"
                else BackwardPackedRowAddress.SuperBlock256
            ),
        ),
        address=BackwardAddressCapability(
            extended_quant_address_register_count=(
                2 if quant_type in ("Q3_K", "Q6_K") else 0
            )
        ),
        quant_register_shape=(
            BackwardQuantRegisterShape(1, 2, None)
            if quant_type == "Q2_K"
            else BackwardQuantRegisterShape(1, 2, None)
            if quant_type == "Q3_K"
            else BackwardQuantRegisterShape(1, 3, None)
            if quant_type == "Q4_K"
            else BackwardQuantRegisterShape(
                1,
                3,
                None,
                vector_metadata_shape=BackwardQuantRegisterShape(4, 0, 1),
            )
            if quant_type == "Q5_K"
            else BackwardQuantRegisterShape(1, 1, None)
            if quant_type == "Q6_K"
            else BackwardQuantRegisterShape(1, 4, None)
            if quant_type == "IQ2_S"
            else BackwardQuantRegisterShape(1, 0, 0)
        ),
    )


@dataclass(frozen=True)
class BackwardProblemContract:
    problem_size: ProblemSize
    quant_type: str
    quant_format: QuantFormat
    mechanism: BackwardMechanismContract
    code_object_version: int = 5

    @classmethod
    def from_solution_key(cls, solution_key: SolutionKey):
        return cls(
            problem_size=solution_key.problem_size,
            quant_type=solution_key.problem_type.quant_data_type,
            quant_format=BACKWARD_QUANT_FORMATS[
                solution_key.problem_type.quant_data_type
            ],
            mechanism=backward_mechanism_contract(
                solution_key.problem_type.quant_data_type
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        problem_size: ProblemSize,
    ) -> BackwardProblemContract:
        item = _strict_mapping(
            value,
            name="BackwardProblemContract",
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
                    "code_object_version",
                }
            ),
        )
        quant_type = _string(item["quant_type"], "quant_type")
        if quant_type not in BACKWARD_QUANT_FORMATS:
            raise SchemaError(f"unsupported backward quant type {quant_type!r}")
        quant_format = BACKWARD_QUANT_FORMATS[quant_type]
        expected: dict[str, object] = {
            "quant_type": quant_type,
            "block_values": quant_format.block_values,
            "packed_weight_block_bytes": quant_format.block_bytes,
            "kernel_language": "Assembly",
            "isa": [11, 5, 1],
            "wavefront_size": 32,
            "activation_type": "BFloat16",
            "destination_type": "BFloat16",
            "compute_type": "Float32",
            "abi": "BackwardOutputA",
            "code_object_version": 5,
        }
        actual = {
            "quant_type": quant_type,
            "block_values": _integer(item["block_values"], "block_values"),
            "packed_weight_block_bytes": _integer(
                item["packed_weight_block_bytes"], "packed_weight_block_bytes"
            ),
            "kernel_language": _string(item["kernel_language"], "kernel_language"),
            "isa": list(_integer_tuple(item["isa"], "isa", 3)),
            "wavefront_size": _integer(item["wavefront_size"], "wavefront_size"),
            "activation_type": _string(item["activation_type"], "activation_type"),
            "destination_type": _string(item["destination_type"], "destination_type"),
            "compute_type": _string(item["compute_type"], "compute_type"),
            "abi": _string(item["abi"], "abi"),
            "code_object_version": _integer(
                item["code_object_version"], "code_object_version"
            ),
        }
        if actual != expected:
            raise SchemaError("BackwardProblemContract is not canonical")
        return cls(
            problem_size=problem_size,
            quant_type=quant_type,
            quant_format=quant_format,
            mechanism=backward_mechanism_contract(quant_type),
            code_object_version=5,
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
            "abi": "BackwardOutputA",
            "code_object_version": self.code_object_version,
        }


@dataclass(frozen=True)
class BackwardGeometrySpec:
    isa: tuple[int, int, int]
    wavefront_size: int
    work_group: tuple[int, int, int]
    matrix_instruction: tuple[int, ...]
    macro_tile0: int
    macro_tile1: int
    depth_u: int
    work_group_mapping: int

    @property
    def num_threads(self) -> int:
        return self.work_group[0] * self.work_group[1] * self.work_group[2]


@dataclass(frozen=True)
class BackwardMemorySpec:
    global_read_vector_width_a: int
    global_read_vector_width_b: int
    local_read_vector_width: int
    transpose_lds: int
    lds_pad_b: int
    lds_block_size_per_pad_b: int
    lds_swizzle_chunk_b: int


@dataclass(frozen=True)
class BackwardPipelineSpec:
    global_read_prefetch: int
    local_read_prefetch: int
    lds_buffering: BackwardLdsBuffering
    schedule: BackwardScheduleIterAlg
    packed_weight_prefetch: BackwardPackedWeightPrefetch
    packed_weight_lane_share: int

    @classmethod
    def from_solution(cls, solution: BackwardSolution) -> BackwardPipelineSpec:
        return cls(
            global_read_prefetch=solution.prefetch_global_read,
            local_read_prefetch=solution.prefetch_local_read,
            lds_buffering=BackwardLdsBuffering(solution.one_lds_buffer),
            schedule=BackwardScheduleIterAlg(solution.schedule_iter_alg),
            packed_weight_prefetch=BackwardPackedWeightPrefetch.from_solution(solution),
            packed_weight_lane_share=solution.packed_weight_lane_share,
        )

    @classmethod
    def try_from_solution(cls, solution: BackwardSolution):
        lds_buffering = BackwardLdsBuffering.try_from_serialized(
            solution.one_lds_buffer
        )
        schedule = BackwardScheduleIterAlg.try_from_serialized(
            solution.schedule_iter_alg
        )
        if lds_buffering is None or schedule is None:
            return None
        return cls(
            global_read_prefetch=solution.prefetch_global_read,
            local_read_prefetch=solution.prefetch_local_read,
            lds_buffering=lds_buffering,
            schedule=schedule,
            packed_weight_prefetch=BackwardPackedWeightPrefetch.from_solution(solution),
            packed_weight_lane_share=solution.packed_weight_lane_share,
        )

    @property
    def decoded_b_pipeline(self) -> bool:
        return self.lds_buffering is BackwardLdsBuffering.Double

    @property
    def prefetches_next_packed_tile(self) -> bool:
        return self.packed_weight_prefetch.includes_next_tile

    @property
    def uses_prefetched_local_read(self) -> bool:
        return self.local_read_prefetch == 2

    @property
    def uses_two_global_reads(self) -> bool:
        return self.global_read_prefetch == 2


@dataclass(frozen=True)
class BackwardQ2DecodePolicy:
    schedule: BackwardQ2DecodeSchedule


@dataclass(frozen=True)
class BackwardQ3DecodePolicy:
    extraction: BackwardExtraction
    pairing: BackwardQ3Pairing


@dataclass(frozen=True)
class BackwardQ4DecodePolicy:
    schedule: BackwardQ4DecodeSchedule


@dataclass(frozen=True)
class BackwardQ5DecodePolicy:
    extraction: BackwardExtraction
    nibble_shift: BackwardQ5NibbleShift
    metadata_load: BackwardQ5MetadataLoad

    @property
    def hoists_nibble_shift(self) -> bool:
        return self.nibble_shift is BackwardQ5NibbleShift.Hoisted

    @property
    def vector_metadata_load(self) -> bool:
        return self.metadata_load is BackwardQ5MetadataLoad.Vector


@dataclass(frozen=True)
class BackwardQ6DecodePolicy:
    extraction: BackwardExtraction


@dataclass(frozen=True)
class BackwardQ8DecodePolicy:
    extraction: BackwardExtraction


@dataclass(frozen=True)
class BackwardDecodeSpec:
    decoder_width: int
    q2: BackwardQ2DecodePolicy
    q3: BackwardQ3DecodePolicy
    q4: BackwardQ4DecodePolicy
    q5: BackwardQ5DecodePolicy
    q6: BackwardQ6DecodePolicy
    q8: BackwardQ8DecodePolicy

    @classmethod
    def from_solution(cls, solution: BackwardSolution) -> BackwardDecodeSpec:
        return cls(
            decoder_width=solution.decoder_width,
            q2=BackwardQ2DecodePolicy(
                BackwardQ2DecodeSchedule(solution.q2_k_decode_schedule)
            ),
            q3=BackwardQ3DecodePolicy(
                BackwardExtraction(solution.q3_k_extraction),
                BackwardQ3Pairing(solution.q3_k_pairing),
            ),
            q4=BackwardQ4DecodePolicy(
                BackwardQ4DecodeSchedule(solution.q4_k_decode_schedule)
            ),
            q5=BackwardQ5DecodePolicy(
                BackwardExtraction(solution.q5_k_extraction),
                BackwardQ5NibbleShift.Hoisted
                if solution.q5_k_nibble_shift_hoist
                else BackwardQ5NibbleShift.Inline,
                BackwardQ5MetadataLoad.Vector
                if solution.q5_k_metadata_vector_load
                else BackwardQ5MetadataLoad.Scalar,
            ),
            q6=BackwardQ6DecodePolicy(BackwardExtraction(solution.q6_k_extraction)),
            q8=BackwardQ8DecodePolicy(BackwardExtraction(solution.q8_0_extraction)),
        )


@dataclass(frozen=True)
class BackwardStoreSpec:
    priority: BackwardStorePriority
    num_elements_per_batch_store: int
    store_vector_width: int

    @property
    def raises_priority(self) -> bool:
        return self.priority is BackwardStorePriority.Raised


@dataclass(frozen=True)
class BackwardKernelSpec:
    geometry: BackwardGeometrySpec
    memory: BackwardMemorySpec
    pipeline: BackwardPipelineSpec
    decode: BackwardDecodeSpec
    store: BackwardStoreSpec

    @classmethod
    def from_solution(cls, solution: BackwardSolution):
        return cls(
            geometry=BackwardGeometrySpec(
                isa=solution.isa,
                wavefront_size=solution.wavefront_size,
                work_group=solution.work_group,
                matrix_instruction=solution.matrix_instruction,
                macro_tile0=solution.macro_tile0,
                macro_tile1=solution.macro_tile1,
                depth_u=solution.depth_u,
                work_group_mapping=solution.work_group_mapping,
            ),
            memory=BackwardMemorySpec(
                global_read_vector_width_a=solution.global_read_vector_width_a,
                global_read_vector_width_b=solution.global_read_vector_width_b,
                local_read_vector_width=solution.local_read_vector_width,
                transpose_lds=solution.transpose_lds,
                lds_pad_b=solution.lds_pad_b,
                lds_block_size_per_pad_b=solution.lds_block_size_per_pad_b,
                lds_swizzle_chunk_b=solution.lds_swizzle_chunk_b,
            ),
            pipeline=BackwardPipelineSpec.from_solution(solution),
            decode=BackwardDecodeSpec.from_solution(solution),
            store=BackwardStoreSpec(
                priority=(
                    BackwardStorePriority.Raised
                    if solution.store_priority_opt
                    else BackwardStorePriority.Default
                ),
                num_elements_per_batch_store=solution.num_elements_per_batch_store,
                store_vector_width=solution.store_vector_width,
            ),
        )

    @property
    def mi_wave_tile(self) -> tuple[int, int]:
        return (
            self.geometry.matrix_instruction[5],
            self.geometry.matrix_instruction[6],
        )

    def to_mapping(self, quant_type: str) -> dict[str, object]:
        memory = self.memory
        if memory.lds_pad_b > 0 and memory.lds_swizzle_chunk_b == 0:
            lds_layout: dict[str, object] = {
                "kind": "Padded",
                "pad_b": memory.lds_pad_b,
            }
        elif memory.lds_pad_b == 0 and memory.lds_swizzle_chunk_b > 0:
            lds_layout = {
                "kind": "Swizzled",
                "chunk_b": memory.lds_swizzle_chunk_b,
            }
        elif memory.lds_pad_b == 0 and memory.lds_swizzle_chunk_b == 0:
            lds_layout = {"kind": "Linear"}
        else:
            raise ValueError("backward LDS layout is not canonically representable")
        mapping: dict[str, object] = {
            "geometry": {
                "work_group": list(self.geometry.work_group),
                "mi_wave_tile": list(self.mi_wave_tile),
                "depth_u": self.geometry.depth_u,
                "work_group_mapping": self.geometry.work_group_mapping,
            },
            "memory": {"lds_layout": lds_layout},
            "pipeline": {
                "global_read_prefetch": self.pipeline.global_read_prefetch,
                "local_read_prefetch": self.pipeline.local_read_prefetch,
                "lds_buffering": self.pipeline.lds_buffering.name,
                "schedule": self.pipeline.schedule.name,
                "packed_weight_prefetch": self.pipeline.packed_weight_prefetch.value,
                "packed_weight_lane_share": self.pipeline.packed_weight_lane_share,
            },
            "store": {
                "priority": self.store.priority.value,
            },
        }
        if quant_type == "Q2_K":
            if self.decode.q2.schedule is not BackwardQ2DecodeSchedule.Serial:
                mapping["decode"] = {"schedule": self.decode.q2.schedule.value}
        elif quant_type == "Q3_K":
            mapping["decode"] = {
                "extraction": self.decode.q3.extraction.value,
                "pairing": self.decode.q3.pairing.value,
            }
        elif quant_type == "Q4_K":
            if self.decode.q4.schedule is not BackwardQ4DecodeSchedule.Serial:
                mapping["decode"] = {"schedule": self.decode.q4.schedule.value}
        elif quant_type == "Q5_K":
            mapping["decode"] = {
                "extraction": self.decode.q5.extraction.value,
                "nibble_shift": self.decode.q5.nibble_shift.value,
                "metadata_load": self.decode.q5.metadata_load.value,
            }
        elif quant_type == "Q6_K":
            mapping["decode"] = {"extraction": self.decode.q6.extraction.value}
        elif quant_type == "Q8_0":
            mapping["decode"] = {"extraction": self.decode.q8.extraction.value}
        elif quant_type == "IQ2_S":
            pass
        else:
            raise ValueError(f"unsupported backward quant type {quant_type!r}")
        return mapping

    @classmethod
    def from_mapping(cls, value: object, quant_type: str) -> BackwardKernelSpec:
        decode_required = quant_type not in ("Q2_K", "Q4_K", "IQ2_S")
        decode_optional = quant_type in ("Q2_K", "Q4_K")
        item = _strict_mapping_optional(
            value,
            name="BackwardKernelSpec",
            required=frozenset(
                {"geometry", "memory", "pipeline", "store"}
                | ({"decode"} if decode_required else set())
            ),
            optional=frozenset({"decode"}) if decode_optional else frozenset(),
        )
        geometry = _strict_mapping(
            item["geometry"],
            name="BackwardKernelSpec.geometry",
            keys=frozenset(
                {"work_group", "mi_wave_tile", "depth_u", "work_group_mapping"}
            ),
        )
        work_group = _integer_tuple(geometry["work_group"], "work_group", 3)
        mi_wave_tile = cast(
            tuple[int, int],
            _integer_tuple(geometry["mi_wave_tile"], "mi_wave_tile", 2),
        )
        if min(*work_group, *mi_wave_tile) <= 0:
            raise SchemaError("backward workgroup and wave tile must be positive")
        thread_count = work_group[0] * work_group[1] * work_group[2]
        if thread_count % 32:
            raise SchemaError("backward workgroup must contain whole wave32 waves")
        wave_count = thread_count // 32
        matrix_instruction = (
            16,
            16,
            16,
            1,
            1,
            mi_wave_tile[0],
            mi_wave_tile[1],
            wave_count,
            1,
        )
        memory_item = _strict_mapping(
            item["memory"],
            name="BackwardKernelSpec.memory",
            keys=frozenset({"lds_layout"}),
        )
        lds_layout_value = memory_item["lds_layout"]
        if not isinstance(lds_layout_value, dict):
            raise SchemaError("BackwardKernelSpec.memory.lds_layout must be a mapping")
        kind = _string(lds_layout_value.get("kind"), "lds_layout.kind")
        if kind == "Padded":
            layout_item = _strict_mapping(
                lds_layout_value,
                name="BackwardKernelSpec.memory.lds_layout",
                keys=frozenset({"kind", "pad_b"}),
            )
            lds_pad_b = _integer(layout_item["pad_b"], "lds_layout.pad_b")
            lds_swizzle_chunk_b = 0
        elif kind == "Swizzled":
            layout_item = _strict_mapping(
                lds_layout_value,
                name="BackwardKernelSpec.memory.lds_layout",
                keys=frozenset({"kind", "chunk_b"}),
            )
            lds_pad_b = 0
            lds_swizzle_chunk_b = _integer(layout_item["chunk_b"], "lds_layout.chunk_b")
        elif kind == "Linear":
            _strict_mapping(
                lds_layout_value,
                name="BackwardKernelSpec.memory.lds_layout",
                keys=frozenset({"kind"}),
            )
            lds_pad_b = 0
            lds_swizzle_chunk_b = 0
        else:
            raise SchemaError(f"invalid backward LDS layout kind {kind!r}")
        pipeline_item = _strict_mapping(
            item["pipeline"],
            name="BackwardKernelSpec.pipeline",
            keys=frozenset(
                {
                    "global_read_prefetch",
                    "local_read_prefetch",
                    "lds_buffering",
                    "schedule",
                    "packed_weight_prefetch",
                    "packed_weight_lane_share",
                }
            ),
        )
        pipeline = BackwardPipelineSpec(
            global_read_prefetch=_integer(
                pipeline_item["global_read_prefetch"], "global_read_prefetch"
            ),
            local_read_prefetch=_integer(
                pipeline_item["local_read_prefetch"], "local_read_prefetch"
            ),
            lds_buffering=_serialized_enum(
                BackwardLdsBuffering,
                pipeline_item["lds_buffering"],
                "lds_buffering",
            ),
            schedule=_serialized_enum(
                BackwardScheduleIterAlg, pipeline_item["schedule"], "schedule"
            ),
            packed_weight_prefetch=_serialized_enum(
                BackwardPackedWeightPrefetch,
                pipeline_item["packed_weight_prefetch"],
                "packed_weight_prefetch",
            ),
            packed_weight_lane_share=_integer(
                pipeline_item["packed_weight_lane_share"],
                "packed_weight_lane_share",
            ),
        )
        q2 = BackwardQ2DecodePolicy(BackwardQ2DecodeSchedule.Serial)
        q3 = BackwardQ3DecodePolicy(
            BackwardExtraction.packed, BackwardQ3Pairing.Inactive
        )
        q4 = BackwardQ4DecodePolicy(BackwardQ4DecodeSchedule.Serial)
        q5 = BackwardQ5DecodePolicy(
            BackwardExtraction.packed,
            BackwardQ5NibbleShift.Inline,
            BackwardQ5MetadataLoad.Scalar,
        )
        q6 = BackwardQ6DecodePolicy(BackwardExtraction.packed)
        q8 = BackwardQ8DecodePolicy(BackwardExtraction.packed)
        if quant_type == "Q2_K" and "decode" in item:
            decode_item = _strict_mapping(
                item["decode"],
                name="BackwardKernelSpec.decode",
                keys=frozenset({"schedule"}),
            )
            q2 = BackwardQ2DecodePolicy(
                _serialized_enum(
                    BackwardQ2DecodeSchedule,
                    decode_item["schedule"],
                    "decode.schedule",
                )
            )
            if q2.schedule is BackwardQ2DecodeSchedule.Serial:
                raise SchemaError(
                    "serial Q2_K decode is canonically represented by absent decode"
                )
        if quant_type == "Q4_K" and "decode" in item:
            decode_item = _strict_mapping(
                item["decode"],
                name="BackwardKernelSpec.decode",
                keys=frozenset({"schedule"}),
            )
            q4 = BackwardQ4DecodePolicy(
                _serialized_enum(
                    BackwardQ4DecodeSchedule,
                    decode_item["schedule"],
                    "decode.schedule",
                )
            )
            if q4.schedule is BackwardQ4DecodeSchedule.Serial:
                raise SchemaError(
                    "serial Q4_K decode is canonically represented by absent decode"
                )
        elif decode_required:
            if quant_type == "Q3_K":
                decode_item = _strict_mapping(
                    item["decode"],
                    name="BackwardKernelSpec.decode",
                    keys=frozenset({"extraction", "pairing"}),
                )
                q3 = BackwardQ3DecodePolicy(
                    _serialized_enum(
                        BackwardExtraction,
                        decode_item["extraction"],
                        "decode.extraction",
                    ),
                    _serialized_enum(
                        BackwardQ3Pairing,
                        decode_item["pairing"],
                        "decode.pairing",
                    ),
                )
                if q3.pairing is BackwardQ3Pairing.Inactive:
                    raise SchemaError("Q3_K pairing must be an active policy")
            elif quant_type == "Q5_K":
                decode_item = _strict_mapping(
                    item["decode"],
                    name="BackwardKernelSpec.decode",
                    keys=frozenset({"extraction", "nibble_shift", "metadata_load"}),
                )
                q5 = BackwardQ5DecodePolicy(
                    _serialized_enum(
                        BackwardExtraction,
                        decode_item["extraction"],
                        "decode.extraction",
                    ),
                    _serialized_enum(
                        BackwardQ5NibbleShift,
                        decode_item["nibble_shift"],
                        "decode.nibble_shift",
                    ),
                    _serialized_enum(
                        BackwardQ5MetadataLoad,
                        decode_item["metadata_load"],
                        "decode.metadata_load",
                    ),
                )
            elif quant_type in ("Q6_K", "Q8_0"):
                decode_item = _strict_mapping(
                    item["decode"],
                    name="BackwardKernelSpec.decode",
                    keys=frozenset({"extraction"}),
                )
                extraction = _serialized_enum(
                    BackwardExtraction,
                    decode_item["extraction"],
                    "decode.extraction",
                )
                if quant_type == "Q6_K":
                    q6 = BackwardQ6DecodePolicy(extraction)
                else:
                    q8 = BackwardQ8DecodePolicy(extraction)
            else:
                raise SchemaError(f"unsupported backward quant type {quant_type!r}")
        store_item = _strict_mapping(
            item["store"],
            name="BackwardKernelSpec.store",
            keys=frozenset({"priority"}),
        )
        return cls(
            geometry=BackwardGeometrySpec(
                isa=(11, 5, 1),
                wavefront_size=32,
                work_group=work_group,
                matrix_instruction=matrix_instruction,
                macro_tile0=16 * wave_count * mi_wave_tile[0],
                macro_tile1=16 * mi_wave_tile[1],
                depth_u=_integer(geometry["depth_u"], "depth_u"),
                work_group_mapping=_integer(
                    geometry["work_group_mapping"], "work_group_mapping"
                ),
            ),
            memory=BackwardMemorySpec(
                global_read_vector_width_a=16,
                global_read_vector_width_b=16,
                local_read_vector_width=16,
                transpose_lds=0,
                lds_pad_b=lds_pad_b,
                lds_block_size_per_pad_b=0,
                lds_swizzle_chunk_b=lds_swizzle_chunk_b,
            ),
            pipeline=pipeline,
            decode=BackwardDecodeSpec(
                decoder_width=16,
                q2=q2,
                q3=q3,
                q4=q4,
                q5=q5,
                q6=q6,
                q8=q8,
            ),
            store=BackwardStoreSpec(
                priority=_serialized_enum(
                    BackwardStorePriority, store_item["priority"], "store.priority"
                ),
                num_elements_per_batch_store=8,
                store_vector_width=1,
            ),
        )

    def to_solution(self, contract: BackwardProblemContract) -> BackwardSolution:
        if self.geometry.isa != (11, 5, 1) or self.geometry.wavefront_size != 32:
            raise ValueError("backward ISA and wavefront are fixed contracts")
        prefetch = self.pipeline.packed_weight_prefetch
        return BackwardSolution(
            kernel_language="Assembly",
            isa=self.geometry.isa,
            wavefront_size=self.geometry.wavefront_size,
            work_group=self.geometry.work_group,
            matrix_instruction=self.geometry.matrix_instruction,
            macro_tile0=self.geometry.macro_tile0,
            macro_tile1=self.geometry.macro_tile1,
            depth_u=self.geometry.depth_u,
            global_read_vector_width_a=self.memory.global_read_vector_width_a,
            global_read_vector_width_b=self.memory.global_read_vector_width_b,
            local_read_vector_width=self.memory.local_read_vector_width,
            prefetch_global_read=self.pipeline.global_read_prefetch,
            prefetch_local_read=self.pipeline.local_read_prefetch,
            one_lds_buffer=self.pipeline.lds_buffering.value,
            schedule_iter_alg=self.pipeline.schedule.value,
            store_priority_opt=self.store.priority is BackwardStorePriority.Raised,
            num_elements_per_batch_store=self.store.num_elements_per_batch_store,
            store_vector_width=self.store.store_vector_width,
            work_group_mapping=self.geometry.work_group_mapping,
            transpose_lds=self.memory.transpose_lds,
            lds_pad_b=self.memory.lds_pad_b,
            lds_block_size_per_pad_b=self.memory.lds_block_size_per_pad_b,
            lds_swizzle_chunk_b=self.memory.lds_swizzle_chunk_b,
            decoder_width=self.decode.decoder_width,
            prefetch_packed_weight=prefetch
            in (
                BackwardPackedWeightPrefetch.CurrentTile,
                BackwardPackedWeightPrefetch.CurrentAndNextTile,
            ),
            prefetch_packed_weight_next=prefetch.includes_next_tile,
            packed_weight_lane_share=self.pipeline.packed_weight_lane_share,
            q2_k_decode_schedule=self.decode.q2.schedule.value,
            q3_k_extraction=self.decode.q3.extraction.value,
            q3_k_pairing=self.decode.q3.pairing.value,
            q4_k_decode_schedule=self.decode.q4.schedule.value,
            q5_k_extraction=self.decode.q5.extraction.value,
            q5_k_nibble_shift_hoist=(
                self.decode.q5.nibble_shift is BackwardQ5NibbleShift.Hoisted
            ),
            q5_k_metadata_vector_load=(
                self.decode.q5.metadata_load is BackwardQ5MetadataLoad.Vector
            ),
            q6_k_extraction=self.decode.q6.extraction.value,
            q8_0_extraction=self.decode.q8.extraction.value,
        )


@dataclass(frozen=True)
class DerivedBackwardState:
    solution_key: SolutionKey
    solution: BackwardSolution
    contract: BackwardProblemContract
    spec: BackwardKernelSpec

    @classmethod
    def from_solution_key(cls, solution_key: SolutionKey):
        solution = solution_key.solution
        if not isinstance(solution, BackwardSolution):
            raise TypeError("MMQ backward state requires BackwardSolution")
        return cls(
            solution_key=solution_key,
            solution=solution,
            contract=BackwardProblemContract.from_solution_key(solution_key),
            spec=BackwardKernelSpec.from_solution(solution),
        )
