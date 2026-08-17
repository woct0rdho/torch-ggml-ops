"""Deterministic linked candidate neighborhoods for MMQ backward assembly."""

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Literal

from .mmq_bwd_spec import BackwardKernelSpec, BackwardProblemContract
from .model import BackwardSolution, ProblemSize, ProblemType, SolutionKey
from .quant_formats import QUANT_FORMATS
from .validation import RejectReason, validate_solution

BackwardSearchKnobGroup = Literal["Decoder", "Pipeline", "LdsLayout"]


@dataclass(frozen=True)
class BackwardCandidateDomain:
    quant_type: str
    problem_size: ProblemSize
    seed: BackwardSolution
    knob_groups: tuple[BackwardSearchKnobGroup, ...]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def backward_candidate_mapping(
    candidate: BackwardSolution, quant_type: str
) -> dict[str, object]:
    key = SolutionKey(
        ProblemType.mmq_backward(quant_type), ProblemSize(1, 1, 1), candidate
    )
    contract = BackwardProblemContract.from_solution_key(key)
    return {
        "ArtifactKind": "KernelCandidate",
        "KernelFamily": "OrdinaryBackward",
        "ProblemContract": contract.to_mapping(),
        "KernelSpec": BackwardKernelSpec.from_solution(candidate).to_mapping(
            quant_type
        ),
    }


def backward_candidate_hash(candidate: BackwardSolution, quant_type: str) -> str:
    return hashlib.sha256(
        _canonical_json(backward_candidate_mapping(candidate, quant_type)).encode()
    ).hexdigest()


def explain_invalid(
    candidate: BackwardSolution,
    quant_type: str,
    shape: ProblemSize,
) -> tuple[RejectReason, ...]:
    return validate_solution(
        SolutionKey(ProblemType.mmq_backward(quant_type), shape, candidate)
    )


def _decoder_neighbors(
    seed: BackwardSolution,
    quant_type: str,
) -> tuple[BackwardSolution, ...]:
    if quant_type == "Q3_K":
        return (
            replace(seed, q3_k_extraction="packed", q3_k_pairing="Partial"),
            replace(seed, q3_k_extraction="packed", q3_k_pairing="Full"),
            replace(seed, q3_k_extraction="scalar", q3_k_pairing="Inactive"),
        )
    if quant_type == "Q4_K":
        return (
            replace(seed, packed_weight_lane_share=1),
            replace(seed, packed_weight_lane_share=2),
        )
    if quant_type == "Q5_K":
        return (
            replace(seed, q5_k_extraction="packed"),
            replace(seed, q5_k_extraction="scalar"),
            replace(seed, q5_k_nibble_shift_hoist=True),
            replace(seed, q5_k_metadata_vector_load=True),
            replace(seed, packed_weight_lane_share=2),
        )
    if quant_type == "Q6_K":
        return (
            replace(seed, q6_k_extraction="packed"),
            replace(seed, q6_k_extraction="packed_vopd"),
            replace(seed, q6_k_extraction="scalar"),
            replace(seed, packed_weight_lane_share=2),
        )
    if quant_type == "Q8_0":
        return tuple(
            replace(seed, q8_0_extraction=extraction)
            for extraction in ("packed", "packed_vopd", "scalar")
        )
    raise ValueError(f"unsupported backward search quant type: {quant_type}")


def _pipeline_neighbors(seed: BackwardSolution) -> tuple[BackwardSolution, ...]:
    candidates = [
        replace(
            seed,
            one_lds_buffer=1,
            schedule_iter_alg=2,
            prefetch_global_read=1,
            prefetch_local_read=1,
            prefetch_packed_weight_next=False,
        ),
        replace(
            seed,
            one_lds_buffer=1,
            schedule_iter_alg=4,
            prefetch_global_read=2,
            prefetch_local_read=1,
            prefetch_packed_weight_next=False,
        ),
        replace(
            seed,
            one_lds_buffer=0,
            schedule_iter_alg=4,
            prefetch_global_read=2,
            prefetch_local_read=1,
            lds_pad_b=0,
            lds_swizzle_chunk_b=8,
            packed_weight_lane_share=1,
            prefetch_packed_weight_next=False,
        ),
    ]
    if seed.depth_u == 32:
        candidates.append(
            replace(
                seed,
                one_lds_buffer=1,
                schedule_iter_alg=3,
                prefetch_global_read=1,
                prefetch_local_read=2,
                prefetch_packed_weight_next=False,
            )
        )
    return tuple(candidates)


def _lds_layout_neighbors(seed: BackwardSolution) -> tuple[BackwardSolution, ...]:
    return (
        replace(seed, lds_pad_b=0, lds_swizzle_chunk_b=0),
        replace(seed, lds_pad_b=8, lds_swizzle_chunk_b=0),
        replace(seed, lds_pad_b=0, lds_swizzle_chunk_b=8),
    )


def candidate_neighbors(
    seed: BackwardSolution,
    quant_type: str,
    shape: ProblemSize,
    knob_groups: tuple[BackwardSearchKnobGroup, ...],
) -> tuple[BackwardSolution, ...]:
    unknown = sorted(set(knob_groups) - {"Decoder", "Pipeline", "LdsLayout"})
    if unknown:
        raise ValueError(f"unsupported backward search knob groups: {unknown}")

    proposed = [seed]
    if "Decoder" in knob_groups:
        proposed.extend(_decoder_neighbors(seed, quant_type))
    if "Pipeline" in knob_groups:
        proposed.extend(_pipeline_neighbors(seed))
    if "LdsLayout" in knob_groups:
        proposed.extend(_lds_layout_neighbors(seed))
    valid = (
        candidate
        for candidate in proposed
        if not explain_invalid(candidate, quant_type, shape)
    )
    unique = {
        backward_candidate_hash(candidate, quant_type): candidate for candidate in valid
    }
    return tuple(unique[digest] for digest in sorted(unique))


def candidate_domains(
    quant_type: str,
    shape: ProblemSize,
) -> tuple[BackwardCandidateDomain, ...]:
    if quant_type not in QUANT_FORMATS:
        raise ValueError(f"unsupported backward search quant type: {quant_type}")
    seed = BackwardSolution.pilot()
    if quant_type == "Q3_K":
        seed = replace(seed, q3_k_pairing="Full")
    if explain_invalid(seed, quant_type, shape):
        return ()
    return (
        BackwardCandidateDomain(
            quant_type=quant_type,
            problem_size=shape,
            seed=seed,
            knob_groups=("Decoder", "Pipeline", "LdsLayout"),
        ),
    )
