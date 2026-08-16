"""Typed problem and solution state for MMQ backward assembly lowering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

from .model import BackwardSolution, ProblemSize, SolutionKey
from .quant_formats import QUANT_FORMATS, QuantFormat


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
    if quant_type not in QUANT_FORMATS:
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
            frozenset({64, 128}) if quant_type in ("Q4_K", "Q6_K") else frozenset({128})
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
            payload_register_count=(4 if quant_type in ("Q4_K", "Q8_0") else 8),
            packed_loads_per_row=(
                5
                if quant_type in ("Q3_K", "Q4_K")
                else 4
                if quant_type == "Q6_K"
                else 2
                if quant_type == "Q8_0"
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
            else BackwardQuantRegisterShape(1, 0, 0)
        ),
    )


@dataclass(frozen=True)
class BackwardProblemContract:
    problem_size: ProblemSize
    quant_type: str
    quant_format: QuantFormat
    mechanism: BackwardMechanismContract
    kernarg_segment_size: int = 40
    code_object_version: int = 5

    @classmethod
    def from_solution_key(cls, solution_key: SolutionKey):
        return cls(
            problem_size=solution_key.problem_size,
            quant_type=solution_key.problem_type.quant_data_type,
            quant_format=QUANT_FORMATS[solution_key.problem_type.quant_data_type],
            mechanism=backward_mechanism_contract(
                solution_key.problem_type.quant_data_type
            ),
        )


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
class BackwardQ3DecodePolicy:
    extraction: BackwardExtraction
    pairing: BackwardQ3Pairing


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
    q3: BackwardQ3DecodePolicy
    q5: BackwardQ5DecodePolicy
    q6: BackwardQ6DecodePolicy
    q8: BackwardQ8DecodePolicy

    @classmethod
    def from_solution(cls, solution: BackwardSolution) -> BackwardDecodeSpec:
        return cls(
            decoder_width=solution.decoder_width,
            q3=BackwardQ3DecodePolicy(
                BackwardExtraction(solution.q3_k_extraction),
                BackwardQ3Pairing(solution.q3_k_pairing),
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
