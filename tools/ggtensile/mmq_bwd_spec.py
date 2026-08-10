"""Typed problem and solution state for MMQ backward assembly lowering."""

from dataclasses import dataclass
from enum import Enum

from .model import BackwardSolution, ProblemSize, SolutionKey
from .quant_formats import QUANT_FORMATS, QuantFormat


class BackwardQ3Pairing(str, Enum):
    INACTIVE = "Inactive"
    PARTIAL = "Partial"
    FULL = "Full"


@dataclass(frozen=True)
class BackwardMechanismContract:
    lane_share_values: frozenset[int]
    padded_depth64: bool
    pipeline_n64: bool
    pipeline_sia3: bool
    pipeline_depth64: bool
    next_packed_depth64: bool


def backward_mechanism_contract(quant_type: str) -> BackwardMechanismContract:
    if quant_type not in QUANT_FORMATS:
        raise ValueError(f"unknown backward quant mechanism: {quant_type}")
    return BackwardMechanismContract(
        lane_share_values=(
            frozenset({1, 2})
            if quant_type in ("Q4_K", "Q5_K", "Q6_K")
            else frozenset({1})
        ),
        padded_depth64=quant_type in ("Q3_K", "Q6_K", "Q8_0"),
        pipeline_n64=quant_type in ("Q4_K", "Q6_K"),
        pipeline_sia3=quant_type == "Q5_K",
        pipeline_depth64=quant_type == "Q5_K",
        next_packed_depth64=quant_type == "Q6_K",
    )


@dataclass(frozen=True)
class BackwardProblemContract:
    problem_size: ProblemSize
    quant_type: str
    quant_format: QuantFormat
    kernarg_segment_size: int = 40
    code_object_version: int = 5

    @classmethod
    def from_solution_key(cls, solution_key: SolutionKey):
        return cls(
            problem_size=solution_key.problem_size,
            quant_type=solution_key.problem_type.quant_data_type,
            quant_format=QUANT_FORMATS[solution_key.problem_type.quant_data_type],
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
    prefetch_global_read: int
    prefetch_local_read: int
    one_lds_buffer: int
    schedule_iter_alg: int
    prefetch_packed_weight: bool
    prefetch_packed_weight_next: bool
    packed_weight_lane_share: int


@dataclass(frozen=True)
class BackwardDecodeSpec:
    decoder_width: int
    q3_k_extraction: str
    q3_k_pairing: BackwardQ3Pairing
    q5_k_extraction: str
    q5_k_nibble_shift_hoist: bool
    q5_k_metadata_vector_load: bool
    q6_k_extraction: str
    q8_0_extraction: str


@dataclass(frozen=True)
class BackwardStoreSpec:
    store_priority_opt: bool
    num_elements_per_batch_store: int
    store_vector_width: int


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
            pipeline=BackwardPipelineSpec(
                prefetch_global_read=solution.prefetch_global_read,
                prefetch_local_read=solution.prefetch_local_read,
                one_lds_buffer=solution.one_lds_buffer,
                schedule_iter_alg=solution.schedule_iter_alg,
                prefetch_packed_weight=solution.prefetch_packed_weight,
                prefetch_packed_weight_next=solution.prefetch_packed_weight_next,
                packed_weight_lane_share=solution.packed_weight_lane_share,
            ),
            decode=BackwardDecodeSpec(
                decoder_width=solution.decoder_width,
                q3_k_extraction=solution.q3_k_extraction,
                q3_k_pairing=BackwardQ3Pairing(solution.q3_k_pairing),
                q5_k_extraction=solution.q5_k_extraction,
                q5_k_nibble_shift_hoist=solution.q5_k_nibble_shift_hoist,
                q5_k_metadata_vector_load=solution.q5_k_metadata_vector_load,
                q6_k_extraction=solution.q6_k_extraction,
                q8_0_extraction=solution.q8_0_extraction,
            ),
            store=BackwardStoreSpec(
                store_priority_opt=solution.store_priority_opt,
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
