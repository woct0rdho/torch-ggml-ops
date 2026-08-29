"""Deterministic manual-search support for MMQ forward assembly candidates.

This module proposes complete candidates and records exact-pair manifests. It does not
select winners, benchmark kernels, repair invalid candidates, or participate in
production generation.
"""

from dataclasses import dataclass, replace
from functools import cache
from itertools import product
from pathlib import Path
from typing import Literal

from .campaign import load_catalog, unique_kernel_specs
from .family_registry import (
    instance_hash,
    instance_name,
    mapping_for_instance,
    problem_type_for_family,
)
from .identity import KernelFamily, canonical_sha256
from .kernel_instance import KernelInstance
from .mmq_fwd_spec import (
    DecodedLdsForwardDecodePolicy,
    ForwardDataMovementPolicy,
    ForwardKernelCandidate,
    ForwardKernelSpec,
    ForwardPipelinePolicy,
    ForwardProblemContract,
    LdsSpec,
    Q6ForwardSchedule,
    SemanticSchedulePolicy,
    StructuredQ6ScheduleVariant,
    q6_schedule_from_kernel_spec,
)
from .model import ProblemSize
from .tuning_policy import (
    IterationSchedule,
    LdsLayout,
    PipelineStage,
)
from .validation import validate_forward_solution

Q6SearchKnobGroup = Literal["InstructionPolicy", "Epilogue", "PhysicalPlan"]
ForwardSearchKnobGroup = Literal[
    "InstructionPolicy",
    "Epilogue",
    "Metadata",
    "Staging",
    "DataMovement",
    "LdsLayout",
]

_CONFIG_DIR = Path(__file__).with_name("configs")
_CATALOG_FILENAMES = {
    "Q3_K": "mmq_fwd_q3_k_catalog.json",
    "Q4_K": "mmq_fwd_q4_k_catalog.json",
    "Q5_K": "mmq_fwd_q5_k_catalog.json",
    "Q6_K": "mmq_fwd_q6_k_catalog.json",
    "Q8_0": "mmq_fwd_q8_0_catalog.json",
}


@cache
def _catalog_forward_specs(quant_type: str) -> tuple[ForwardKernelSpec, ...]:
    catalog = load_catalog(_CONFIG_DIR / _CATALOG_FILENAMES[quant_type])
    specs = unique_kernel_specs(catalog)
    assert all(isinstance(spec, ForwardKernelSpec) for spec in specs)
    return tuple(spec for spec in specs if isinstance(spec, ForwardKernelSpec))


def _catalog_q6_spec(macro_tile0: int) -> ForwardKernelSpec:
    for spec in _catalog_forward_specs("Q6_K"):
        if spec.macro_tile == (macro_tile0, 64):
            return spec
    raise ValueError(f"no catalog Q6_K specification for macro tile {macro_tile0}")


def q6_schedule_seed(macro_tile0: int) -> Q6ForwardSchedule:
    """Load the canonical Q6 geometry seed used by manual schedule search."""
    return q6_schedule_from_kernel_spec(_catalog_q6_spec(macro_tile0))


@dataclass(frozen=True)
class ForwardCandidateDomain:
    """One linkage-safe geometry domain with one complete canonical seed."""

    quant_type: str
    problem_size: ProblemSize
    kernel_spec: ForwardKernelSpec
    knob_groups: tuple[ForwardSearchKnobGroup, ...]


def q6_candidate_mapping(schedule: Q6ForwardSchedule) -> dict[str, object]:
    """Return the complete parameter-only identity of one Q6 candidate."""
    base = _catalog_q6_spec(schedule.macro_tile0)
    spec = q6_kernel_spec_with_schedule(base, schedule)
    return ForwardKernelCandidate(
        ForwardProblemContract.for_quant_type("Q6_K"), spec
    ).to_mapping()


def q6_candidate_hash(schedule: Q6ForwardSchedule) -> str:
    """Return a stable hash over every assembly-affecting Q6 candidate field."""
    return canonical_sha256(q6_candidate_mapping(schedule))


def q6_kernel_spec_with_schedule(
    base: ForwardKernelSpec,
    schedule: Q6ForwardSchedule,
) -> ForwardKernelSpec:
    """Apply one complete Q6 schedule to a typed base specification."""
    pipeline = base.epilogue.pipeline
    assert pipeline is not None
    spec = replace(
        base,
        epilogue=replace(
            base.epilogue,
            pipeline=replace(
                pipeline,
                dependency_width=schedule.epilogue_dependency_width,
                scope=schedule.epilogue_pipeline_scope,
            ),
        ),
        global_memory=replace(
            base.global_memory,
            global_read_cache_policy=schedule.global_read_cache_policy,
        ),
        instruction_policy=replace(
            base.instruction_policy,
            dependency_delay_mode=schedule.dependency_delay_mode,
            physical_plan=schedule.physical_plan,
        ),
        semantic_schedule=schedule.semantic_policy,
    )
    if q6_schedule_from_kernel_spec(spec) != schedule:
        raise ValueError("Q6 schedule geometry does not match the base specification")
    return spec


@dataclass(frozen=True)
class ForwardExactPairManifest:
    """One exact problem/candidate evaluation unit for any forward family."""

    quant_type: str
    problem_size: ProblemSize
    candidate: ForwardKernelSpec

    @property
    def candidate_hash(self) -> str:
        return forward_candidate_hash(self.candidate, self.quant_type)

    @property
    def kernel_instance(self) -> KernelInstance:
        return KernelInstance.for_gfx1151(
            KernelFamily.OrdinaryForward,
            problem_type_for_family(KernelFamily.OrdinaryForward, self.quant_type),
            self.problem_size,
            self.candidate,
        )

    @property
    def exact_pair_hash(self) -> str:
        return canonical_sha256(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        instance = self.kernel_instance
        return {
            "QuantType": self.quant_type,
            "ProblemSize": self.problem_size.to_mapping(),
            "CandidateHash": self.candidate_hash,
            "Candidate": canonical_candidate(self.candidate, self.quant_type),
            "KernelSpecKey": mapping_for_instance(instance),
            "KernelSpecHash": instance_hash(instance),
            "KernelName": instance_name(instance),
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
    def kernel_instance(self) -> KernelInstance:
        base = _catalog_q6_spec(self.schedule.macro_tile0)
        spec = q6_kernel_spec_with_schedule(base, self.schedule)
        return KernelInstance.for_gfx1151(
            KernelFamily.OrdinaryForward,
            problem_type_for_family(KernelFamily.OrdinaryForward, "Q6_K"),
            self.problem_size,
            spec,
        )

    @property
    def exact_pair_hash(self) -> str:
        return canonical_sha256(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        instance = self.kernel_instance
        return {
            "ProblemSize": self.problem_size.to_mapping(),
            "CandidateHash": self.candidate_hash,
            "Candidate": q6_candidate_mapping(self.schedule),
            "KernelSpecKey": mapping_for_instance(instance),
            "KernelSpecHash": instance_hash(instance),
            "KernelName": instance_name(instance),
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
            ("None", "Explicit") if seed.macro_tile[0] == 128 else ("None",)
        )
        cache_policies = ("Default", "InvalidateL0")
        semantic_policies = SemanticSchedulePolicy.structured_q6_variants()
    if "Epilogue" in knob_groups:
        dependency_widths = (1, 2, 4, 8)
        pipeline_scopes = ("StoreBatch", "FullTile")
    if "PhysicalPlan" in knob_groups and seed.macro_tile[0] == 64:
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
        or semantic_policy.variant is StructuredQ6ScheduleVariant.Wavefront
    )
    unique = {q6_candidate_hash(candidate): candidate for candidate in candidates}
    return tuple(unique[digest] for digest in sorted(unique))


def q6_schedule_candidates(macro_tile0: int) -> tuple[Q6ForwardSchedule, ...]:
    """Return the complete currently implemented manual neighborhood for a geometry."""
    return q6_schedule_neighbors(
        q6_schedule_seed(macro_tile0),
        ("InstructionPolicy", "Epilogue", "PhysicalPlan"),
    )


def canonical_candidate(
    candidate: ForwardKernelSpec,
    quant_type: str,
) -> dict[str, object]:
    """Canonicalize a complete typed candidate without problem-shape data."""
    return ForwardKernelCandidate(
        ForwardProblemContract.for_quant_type(quant_type), candidate
    ).to_mapping()


def forward_candidate_hash(candidate: ForwardKernelSpec, quant_type: str) -> str:
    """Hash a complete forward candidate independently of exact problem shape."""
    return canonical_sha256(canonical_candidate(candidate, quant_type))


def is_valid_candidate(
    candidate: ForwardKernelSpec,
    quant_type: str,
    shape: ProblemSize,
) -> bool:
    """Return whether one exact candidate/shape pair satisfies codegen assertions."""
    try:
        validate_forward_solution(shape, quant_type, candidate)
    except AssertionError:
        return False
    return True


def candidate_neighbors(
    seed: ForwardKernelSpec,
    quant_type: str,
    knob_groups: tuple[ForwardSearchKnobGroup, ...],
) -> tuple[ForwardKernelSpec, ...]:
    """Build complete linked neighbors outside production code generation."""
    if quant_type == "Q6_K":
        unknown = sorted(set(knob_groups) - {"InstructionPolicy", "Epilogue"})
        if unknown:
            raise ValueError(f"unsupported Q6 search knob groups: {unknown}")
        q6_groups = tuple(
            group for group in knob_groups if group in ("InstructionPolicy", "Epilogue")
        )
        schedules = q6_schedule_neighbors(
            q6_schedule_from_kernel_spec(seed),
            q6_groups,
        )
        candidates = tuple(
            q6_kernel_spec_with_schedule(seed, item) for item in schedules
        )
    elif quant_type in ("Q3_K", "Q8_0"):
        if knob_groups:
            raise ValueError(
                f"{quant_type} exposes complete implemented policies without free knob groups"
            )
        candidates = (seed,)
    else:
        unknown = sorted(
            set(knob_groups)
            - {
                "InstructionPolicy",
                "Epilogue",
                "Metadata",
                "Staging",
                "DataMovement",
                "LdsLayout",
            }
        )
        if unknown:
            raise ValueError(
                f"unsupported decoded-weight LDS search knob groups: {unknown}"
            )
        pipeline = seed.epilogue.pipeline
        assert pipeline is not None
        new_policy_groups = {"Staging", "DataMovement"} & set(knob_groups)
        working_seed = seed
        if new_policy_groups:
            working_seed = replace(
                seed,
                pipeline=seed.pipeline or ForwardPipelinePolicy.canonical(),
                data_movement=(
                    seed.data_movement or ForwardDataMovementPolicy.canonical()
                ),
            )
        staging_policies = (working_seed.pipeline,)
        movement_policies = (working_seed.data_movement,)
        lds_policies = (working_seed.lds,)
        if "Staging" in knob_groups:
            staging_policies = (
                ForwardPipelinePolicy.canonical(),
                replace(
                    ForwardPipelinePolicy.canonical(),
                    prefetch_global_read=PipelineStage.DoubleStage,
                    schedule_iter_alg=IterationSchedule.ExplicitPipeline,
                ),
            )
        if "DataMovement" in knob_groups:
            movement_policies = tuple(
                ForwardDataMovementPolicy(
                    payload_global_read_vector_width=payload_width,
                    metadata_load_vector_width=metadata_width,
                    payload_lds_write_vector_width=lds_width,
                    metadata_lds_write_vector_width=4,
                    decode_producer_count=2,
                )
                for payload_width, metadata_width, lds_width in product(
                    (4, 8, 16),
                    (4, 8, 16),
                    (4, 8, 16),
                )
            )
        if "LdsLayout" in knob_groups:
            address_hoist = seed.lds.address_hoist
            lds_policies = (
                seed.lds,
                *(
                    LdsSpec(address_hoist, LdsLayout.PaddedRows, pad_a, pad_b, 64)
                    for pad_a, pad_b in (
                        (4, 0),
                        (8, 0),
                        (16, 0),
                        (0, 4),
                        (0, 8),
                        (0, 16),
                        (4, 16),
                    )
                ),
            )
        decode_policies = (seed.decode.policy,)
        tiles_ahead = (pipeline.tiles_ahead,)
        dependency_widths = (pipeline.dependency_width,)
        priorities = (pipeline.priority,)
        accumulator_initializations = (
            seed.instruction_policy.accumulator_initialization,
        )
        if "Metadata" in knob_groups:
            decode_policies = (
                (
                    DecodedLdsForwardDecodePolicy(False, False),
                    DecodedLdsForwardDecodePolicy(False, True),
                    DecodedLdsForwardDecodePolicy(True, False),
                    DecodedLdsForwardDecodePolicy(True, True),
                )
                if quant_type == "Q4_K"
                else (
                    DecodedLdsForwardDecodePolicy(False, False),
                    DecodedLdsForwardDecodePolicy(False, True),
                    DecodedLdsForwardDecodePolicy(True, True),
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
                working_seed,
                pipeline=staging_policy,
                data_movement=movement_policy,
                lds=lds_policy,
                decode=replace(seed.decode, policy=decode_policy),
                epilogue=replace(
                    seed.epilogue,
                    pipeline=replace(
                        pipeline,
                        tiles_ahead=tiles,
                        dependency_width=width,
                        priority=priority,
                    ),
                ),
                instruction_policy=replace(
                    seed.instruction_policy,
                    accumulator_initialization=initialization,
                ),
            )
            for (
                decode_policy,
                tiles,
                width,
                priority,
                initialization,
                staging_policy,
                movement_policy,
                lds_policy,
            ) in product(
                decode_policies,
                tiles_ahead,
                dependency_widths,
                priorities,
                accumulator_initializations,
                staging_policies,
                movement_policies,
                lds_policies,
            )
        )
        capability_shape = ProblemSize(
            seed.macro_tile[0],
            seed.macro_tile[1],
            256,
        )
        candidates = tuple(
            candidate
            for candidate in proposed
            if is_valid_candidate(candidate, quant_type, capability_shape)
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
    if quant_type not in _CATALOG_FILENAMES:
        raise ValueError(
            "manual forward domains implement Q3_K, Q4_K, Q5_K, Q6_K, and Q8_0"
        )
    if quant_type == "Q6_K":
        knob_groups: tuple[ForwardSearchKnobGroup, ...] = (
            "InstructionPolicy",
            "Epilogue",
        )
        seeds = _catalog_forward_specs(quant_type)
    elif quant_type in {"Q4_K", "Q5_K"}:
        knob_groups = (
            "Metadata",
            "Epilogue",
            "InstructionPolicy",
            "Staging",
            "DataMovement",
            "LdsLayout",
        )
        valid_seeds = tuple(
            seed
            for seed in _catalog_forward_specs(quant_type)
            if is_valid_candidate(seed, quant_type, shape)
        )
        seeds = valid_seeds[:1]
    else:
        knob_groups = ()
        seeds = _catalog_forward_specs(quant_type)
    return tuple(
        ForwardCandidateDomain(
            quant_type=quant_type,
            problem_size=shape,
            kernel_spec=seed,
            knob_groups=knob_groups,
        )
        for seed in seeds
        if is_valid_candidate(seed, quant_type, shape)
    )
