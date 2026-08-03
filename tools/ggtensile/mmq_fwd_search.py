"""Deterministic manual-search support for MMQ forward assembly candidates.

This module proposes complete candidates and records exact-pair manifests. It does not
select winners, benchmark kernels, repair invalid candidates, or participate in
production generation.
"""

import hashlib
import json
from dataclasses import dataclass, replace
from itertools import product
from typing import Literal

from .mmq_fwd_spec import (
    ForwardKernelCandidate,
    ForwardKernelSpec,
    Q6ForwardSchedule,
    q6_schedule_from_solution,
)
from .model import (
    ForwardSolution,
    ProblemSize,
    ProblemType,
    SolutionKey,
)
from .validation import RejectReason, validate_solution

Q6SearchKnobGroup = Literal["InstructionPolicy", "Epilogue"]
ForwardSearchKnobGroup = Literal["InstructionPolicy", "Epilogue", "Metadata"]


@dataclass(frozen=True)
class ForwardCandidateDomain:
    """One linkage-safe geometry domain with one complete canonical seed."""

    quant_type: str
    problem_size: ProblemSize
    kernel_spec: ForwardKernelSpec
    seed: ForwardSolution
    knob_groups: tuple[ForwardSearchKnobGroup, ...]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def q6_candidate_mapping(schedule: Q6ForwardSchedule) -> dict[str, object]:
    """Return the complete parameter-only identity of one Q6 candidate."""
    base = ForwardSolution.q6_k_structured_decoded(macro_tile0=schedule.macro_tile0)
    solution = q6_solution_with_schedule(base, schedule)
    candidate = ForwardKernelCandidate.from_solution("Q6_K", solution)
    return {
        "SchemaVersion": 2,
        "KernelFamily": "Q6StructuredDecoded",
        "ProblemContract": candidate.problem_contract.to_mapping(),
        "KernelSpec": candidate.kernel_spec.to_mapping(),
    }


def q6_candidate_hash(schedule: Q6ForwardSchedule) -> str:
    """Return a stable hash over every assembly-affecting Q6 candidate field."""
    return hashlib.sha256(
        _canonical_json(q6_candidate_mapping(schedule)).encode()
    ).hexdigest()


def q6_solution_with_schedule(
    base: ForwardSolution,
    schedule: Q6ForwardSchedule,
) -> ForwardSolution:
    """Serialize one complete candidate through the normal ForwardSolution path."""
    solution = replace(
        base,
        epilogue_dependency_width=schedule.epilogue_dependency_width,
        q6_epilogue_pipeline_scope=schedule.epilogue_pipeline_scope,
        q6_dependency_delay_mode=schedule.dependency_delay_mode,
        q6_global_read_cache_policy=schedule.global_read_cache_policy,
    )
    if q6_schedule_from_solution(solution) != schedule:
        raise ValueError("Q6 schedule geometry does not match the base solution")
    return solution


@dataclass(frozen=True)
class ForwardExactPairManifest:
    """One exact problem/candidate evaluation unit for any forward family."""

    quant_type: str
    problem_size: ProblemSize
    candidate: ForwardSolution

    @property
    def candidate_hash(self) -> str:
        return forward_candidate_hash(self.candidate, self.quant_type)

    @property
    def solution_key(self) -> SolutionKey:
        return SolutionKey(
            ProblemType.mmq_forward(self.quant_type),
            self.problem_size,
            self.candidate,
        )

    @property
    def exact_pair_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_mapping()).encode()).hexdigest()

    def to_mapping(self) -> dict[str, object]:
        solution_key = self.solution_key
        return {
            "SchemaVersion": 1,
            "QuantType": self.quant_type,
            "ProblemSize": self.problem_size.to_mapping(),
            "CandidateHash": self.candidate_hash,
            "Candidate": canonical_candidate(self.candidate, self.quant_type),
            "SolutionKey": solution_key.to_mapping(),
            "SolutionHash": solution_key.hash,
            "KernelName": solution_key.kernel_name,
        }


@dataclass(frozen=True)
class Q6ExactPairManifest:
    """One exact problem/candidate evaluation unit for manual campaigns."""

    problem_size: ProblemSize
    schedule: Q6ForwardSchedule

    @property
    def candidate_hash(self) -> str:
        return q6_candidate_hash(self.schedule)

    @property
    def solution_key(self) -> SolutionKey:
        base = ForwardSolution.q6_k_structured_decoded(
            macro_tile0=self.schedule.macro_tile0
        )
        solution = q6_solution_with_schedule(base, self.schedule)
        return SolutionKey(
            ProblemType.mmq_forward("Q6_K"),
            self.problem_size,
            solution,
        )

    @property
    def exact_pair_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_mapping()).encode()).hexdigest()

    def to_mapping(self) -> dict[str, object]:
        solution_key = self.solution_key
        return {
            "SchemaVersion": 1,
            "ProblemSize": self.problem_size.to_mapping(),
            "CandidateHash": self.candidate_hash,
            "Candidate": q6_candidate_mapping(self.schedule),
            "SolutionKey": solution_key.to_mapping(),
            "SolutionHash": solution_key.hash,
            "KernelName": solution_key.kernel_name,
        }


def q6_schedule_neighbors(
    seed: Q6ForwardSchedule,
    knob_groups: tuple[Q6SearchKnobGroup, ...],
) -> tuple[Q6ForwardSchedule, ...]:
    """Enumerate a deterministic linked neighborhood around one complete seed.

    Only implemented domains are exposed. The result consists of complete schedules;
    code generation receives one result and performs no search or repair.
    """
    unknown = sorted(set(knob_groups) - {"InstructionPolicy", "Epilogue"})
    if unknown:
        raise ValueError(f"unsupported Q6 search knob groups: {unknown}")

    dependency_delay_modes = (seed.dependency_delay_mode,)
    cache_policies = (seed.global_read_cache_policy,)
    dependency_widths = (seed.epilogue_dependency_width,)
    pipeline_scopes = (seed.epilogue_pipeline_scope,)
    if "InstructionPolicy" in knob_groups:
        dependency_delay_modes = (
            ("None", "Explicit") if seed.macro_tile0 == 128 else ("None",)
        )
        cache_policies = ("Default", "InvalidateL0")
    if "Epilogue" in knob_groups:
        dependency_widths = (1, 2, 4, 8)
        pipeline_scopes = ("StoreBatch", "FullTile")

    candidates = (
        replace(
            seed,
            dependency_delay_mode=dependency_delay_mode,
            global_read_cache_policy=cache_policy,
            epilogue_dependency_width=dependency_width,
            epilogue_pipeline_scope=pipeline_scope,
        )
        for dependency_delay_mode, cache_policy, dependency_width, pipeline_scope in product(
            dependency_delay_modes,
            cache_policies,
            dependency_widths,
            pipeline_scopes,
        )
    )
    unique = {q6_candidate_hash(candidate): candidate for candidate in candidates}
    return tuple(unique[digest] for digest in sorted(unique))


def q6_schedule_candidates(macro_tile0: int) -> tuple[Q6ForwardSchedule, ...]:
    """Return the complete currently implemented manual neighborhood for a geometry."""
    selected = ForwardSolution.q6_k_structured_decoded(macro_tile0=macro_tile0)
    return q6_schedule_neighbors(
        q6_schedule_from_solution(selected),
        ("InstructionPolicy", "Epilogue"),
    )


def canonical_candidate(
    candidate: ForwardSolution,
    quant_type: str,
) -> dict[str, object]:
    """Canonicalize a complete serialized candidate without problem-shape data."""
    return ForwardKernelCandidate.from_solution(quant_type, candidate).to_mapping()


def forward_candidate_hash(candidate: ForwardSolution, quant_type: str) -> str:
    """Hash a complete forward candidate independently of exact problem shape."""
    return hashlib.sha256(
        _canonical_json(canonical_candidate(candidate, quant_type)).encode()
    ).hexdigest()


def explain_invalid(
    candidate: ForwardSolution,
    quant_type: str,
    shape: ProblemSize,
) -> tuple[RejectReason, ...]:
    """Return formula/capability rejection reasons for one exact pair."""
    return validate_solution(
        SolutionKey(ProblemType.mmq_forward(quant_type), shape, candidate)
    )


def _candidate_quant_type(candidate: ForwardSolution) -> str:
    if candidate.operand_source == "Q6StructuredDecoded":
        return "Q6_K"
    if candidate.operand_source == "DecodedWeightLdsBatch8":
        if candidate.weight_decode == "DirectNibble":
            return "Q4_K"
        if candidate.weight_decode == "DirectNibbleHighBit":
            return "Q5_K"
    raise ValueError(
        "candidate does not belong to an implemented forward search family"
    )


def candidate_neighbors(
    seed: ForwardSolution,
    knob_groups: tuple[ForwardSearchKnobGroup, ...],
) -> tuple[ForwardSolution, ...]:
    """Build complete linked neighbors outside production code generation."""
    quant_type = _candidate_quant_type(seed)
    if quant_type == "Q6_K":
        unknown = sorted(set(knob_groups) - {"InstructionPolicy", "Epilogue"})
        if unknown:
            raise ValueError(f"unsupported Q6 search knob groups: {unknown}")
        q6_groups = tuple(
            group for group in knob_groups if group in ("InstructionPolicy", "Epilogue")
        )
        schedules = q6_schedule_neighbors(
            q6_schedule_from_solution(seed),
            q6_groups,
        )
        candidates = tuple(q6_solution_with_schedule(seed, item) for item in schedules)
    else:
        unknown = sorted(
            set(knob_groups) - {"InstructionPolicy", "Epilogue", "Metadata"}
        )
        if unknown:
            raise ValueError(
                f"unsupported decoded-weight LDS search knob groups: {unknown}"
            )
        metadata_schedules = (seed.metadata_schedule,)
        tiles_ahead = (seed.epilogue_tiles_ahead,)
        dependency_widths = (seed.epilogue_dependency_width,)
        priorities = (seed.epilogue_priority,)
        accumulator_initializations = (seed.accumulator_initialization,)
        if "Metadata" in knob_groups:
            metadata_schedules = (
                (
                    "Serialized",
                    "MetadataAfterLowWmma",
                    "IndependentExtraction",
                    "IndependentExtractionMetadataAfterLowWmma",
                )
                if quant_type == "Q4_K"
                else (
                    "Serialized",
                    "MetadataAfterLowWmma",
                    "IndependentExtractionMetadataAfterLowWmma",
                )
            )
        if "Epilogue" in knob_groups:
            tiles_ahead = tuple(range(1, 9))
            dependency_widths = tuple(range(1, 9))
            priorities = tuple(range(4))
        if "InstructionPolicy" in knob_groups:
            accumulator_initializations = ("ScalarCopy", "VopdPair")
        proposed = (
            replace(
                seed,
                metadata_schedule=metadata_schedule,
                epilogue_tiles_ahead=tiles,
                epilogue_dependency_width=width,
                epilogue_priority=priority,
                accumulator_initialization=initialization,
            )
            for metadata_schedule, tiles, width, priority, initialization in product(
                metadata_schedules,
                tiles_ahead,
                dependency_widths,
                priorities,
                accumulator_initializations,
            )
        )
        capability_shape = ProblemSize(seed.macro_tile0, seed.macro_tile1, 256)
        candidates = tuple(
            candidate
            for candidate in proposed
            if not explain_invalid(candidate, quant_type, capability_shape)
        )
    unique = {
        forward_candidate_hash(candidate, quant_type): candidate
        for candidate in candidates
    }
    return tuple(unique[digest] for digest in sorted(unique))


def candidate_domains(
    quant_type: str,
    shape: ProblemSize,
) -> tuple[ForwardCandidateDomain, ...]:
    """Return linkage-safe complete-candidate domains valid for an exact shape."""
    domains: list[ForwardCandidateDomain] = []
    if quant_type == "Q6_K":
        seeds = tuple(
            ForwardSolution.q6_k_structured_decoded(macro_tile0=macro_tile0)
            for macro_tile0 in (64, 128)
        )
        knob_groups: tuple[ForwardSearchKnobGroup, ...] = (
            "InstructionPolicy",
            "Epilogue",
        )
    elif quant_type == "Q4_K":
        seeds = (
            ForwardSolution.q4_k_decoded_weight_lds_extraction(
                epilogue_tiles_ahead=8,
                epilogue_dependency_width=1,
                epilogue_priority=0,
                metadata_after_low_wmma=True,
            ),
        )
        knob_groups = ("Metadata", "Epilogue", "InstructionPolicy")
    elif quant_type == "Q5_K":
        seeds = (
            ForwardSolution.q5_k_decoded_weight_lds_extraction(
                epilogue_tiles_ahead=8,
                epilogue_dependency_width=1,
                epilogue_priority=0,
            ),
        )
        knob_groups = ("Metadata", "Epilogue", "InstructionPolicy")
    else:
        raise ValueError("manual forward domains implement Q4_K, Q5_K, and Q6_K")
    for seed in seeds:
        if explain_invalid(seed, quant_type, shape):
            continue
        domains.append(
            ForwardCandidateDomain(
                quant_type=quant_type,
                problem_size=shape,
                kernel_spec=ForwardKernelSpec.from_solution(seed),
                seed=seed,
                knob_groups=knob_groups,
            )
        )
    return tuple(domains)
