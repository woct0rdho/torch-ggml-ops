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
    SemanticSchedulePolicy,
    forward_mechanism_contract,
    q6_schedule_from_solution,
)
from .model import (
    ForwardSolution,
    ProblemSize,
    ProblemType,
    SolutionKey,
)
from .validation import RejectReason, validate_solution

Q6SearchKnobGroup = Literal["InstructionPolicy", "Epilogue", "PhysicalPlan"]
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
    return ForwardKernelCandidate.from_solution("Q6_K", solution).to_mapping()


def q6_candidate_hash(schedule: Q6ForwardSchedule) -> str:
    """Return a stable hash over every assembly-affecting Q6 candidate field."""
    identity = {
        "ArtifactKind": "KernelCandidate",
        "KernelFamily": "OrdinaryForward",
        **q6_candidate_mapping(schedule),
    }
    return hashlib.sha256(_canonical_json(identity).encode()).hexdigest()


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
        q6_output_traversal=schedule.semantic_policy.traversal,
        q6_stage_clustering=schedule.semantic_policy.clustering,
        q6_latency_policy=schedule.semantic_policy.latency,
        q6_pressure_policy=schedule.semantic_policy.pressure,
        q6_wait_policy=schedule.semantic_policy.wait,
        q6_pairing_policy=schedule.semantic_policy.pairing,
        q6_physical_plan=schedule.physical_plan,
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
    unknown = sorted(
        set(knob_groups) - {"InstructionPolicy", "Epilogue", "PhysicalPlan"}
    )
    if unknown:
        raise ValueError(f"unsupported Q6 search knob groups: {unknown}")

    dependency_delay_modes = (seed.dependency_delay_mode,)
    cache_policies = (seed.global_read_cache_policy,)
    dependency_widths = (seed.epilogue_dependency_width,)
    pipeline_scopes = (seed.epilogue_pipeline_scope,)
    semantic_policies = (seed.semantic_policy,)
    physical_plans = (seed.physical_plan,)
    if "InstructionPolicy" in knob_groups:
        dependency_delay_modes = (
            ("None", "Explicit") if seed.macro_tile0 == 128 else ("None",)
        )
        cache_policies = ("Default", "InvalidateL0")
        semantic_policies = SemanticSchedulePolicy.supported_structured_q6()
    if "Epilogue" in knob_groups:
        dependency_widths = (1, 2, 4, 8)
        pipeline_scopes = ("StoreBatch", "FullTile")
    if "PhysicalPlan" in knob_groups and seed.macro_tile0 == 64:
        physical_plans = ("CanonicalRegisterRoles", "WideScalarCarryFrontier")

    candidates = (
        replace(
            seed,
            semantic_policy=semantic_policy,
            physical_plan=physical_plan,
            dependency_delay_mode=dependency_delay_mode,
            global_read_cache_policy=cache_policy,
            epilogue_dependency_width=dependency_width,
            epilogue_pipeline_scope=pipeline_scope,
        )
        for semantic_policy, physical_plan, dependency_delay_mode, cache_policy, dependency_width, pipeline_scope in product(
            semantic_policies,
            physical_plans,
            dependency_delay_modes,
            cache_policies,
            dependency_widths,
            pipeline_scopes,
        )
        if physical_plan == "CanonicalRegisterRoles"
        or semantic_policy == SemanticSchedulePolicy.structured_q6_wavefront()
    )
    unique = {q6_candidate_hash(candidate): candidate for candidate in candidates}
    return tuple(unique[digest] for digest in sorted(unique))


def q6_schedule_candidates(macro_tile0: int) -> tuple[Q6ForwardSchedule, ...]:
    """Return the complete currently implemented manual neighborhood for a geometry."""
    selected = ForwardSolution.q6_k_structured_decoded(macro_tile0=macro_tile0)
    return q6_schedule_neighbors(
        q6_schedule_from_solution(selected),
        ("InstructionPolicy", "Epilogue", "PhysicalPlan"),
    )


def canonical_candidate(
    candidate: ForwardSolution,
    quant_type: str,
) -> dict[str, object]:
    """Canonicalize a complete serialized candidate without problem-shape data."""
    return ForwardKernelCandidate.from_solution(quant_type, candidate).to_mapping()


def forward_candidate_hash(candidate: ForwardSolution, quant_type: str) -> str:
    """Hash a complete forward candidate independently of exact problem shape."""
    identity = {
        "ArtifactKind": "KernelCandidate",
        "KernelFamily": "OrdinaryForward",
        **canonical_candidate(candidate, quant_type),
    }
    return hashlib.sha256(_canonical_json(identity).encode()).hexdigest()


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
    lowering = forward_mechanism_contract(candidate.operand_source).lowering
    if lowering == "StructuredQ6":
        return "Q6_K"
    if lowering in ("Packed3BitTiledLds", "Packed3BitFullWeightTiledLds"):
        return "Q3_K"
    if lowering == "SignedInt8":
        return "Q8_0"
    if lowering == "DecodedWeightLds":
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
    elif quant_type in ("Q3_K", "Q8_0"):
        if knob_groups:
            raise ValueError(
                f"{quant_type} exposes complete implemented policies without free knob groups"
            )
        candidates = (seed,)
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
    elif quant_type == "Q3_K":
        seeds = (
            ForwardSolution.q3_k_hip_tiled_lds(),
            ForwardSolution.q3_k_full_weight_tiled_lds(),
            ForwardSolution.q3_k_full_weight_decode_ready_frontier(),
        )
        knob_groups = ()
    elif quant_type == "Q8_0":
        seeds = (
            ForwardSolution.q8_0_direct_global(),
            ForwardSolution.q8_0_register_tiled(),
            ForwardSolution.q8_0_hip_tiled_lds(),
            ForwardSolution.q8_0_hip_tiled_lds_depth64(),
            ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=32),
            ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=64),
            ForwardSolution.q8_0_compact_depth32_tiled_lds(macro_tile0=32),
            ForwardSolution.q8_0_compact_depth32_tiled_lds(macro_tile0=64),
            ForwardSolution.q8_0_compact_depth32_tiled_lds(macro_tile0=128),
        )
        knob_groups = ()
    else:
        raise ValueError(
            "manual forward domains implement Q3_K, Q4_K, Q5_K, Q6_K, and Q8_0"
        )
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
