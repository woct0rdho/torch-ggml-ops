"""Deterministic linked candidate neighborhoods for MMQ backward assembly."""

from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Literal

from .campaign import load_catalog, unique_kernel_specs
from .identity import (
    GFX1151_TARGET,
    KernelFamily,
    canonical_sha256,
    problem_type_mapping,
)
from .mmq_bwd_spec import (
    BackwardBitfieldShiftPlacement,
    BackwardExtraction,
    BackwardIterationPolicy,
    BackwardKernelSpec,
    BackwardMetadataLoad,
    BackwardPairing,
    LdsBuffering,
    PackedLoadGrouping,
)
from .model import ProblemSize
from .validation import validate_backward_solution

BackwardSearchKnobGroup = Literal["Decoder", "Pipeline", "LdsLayout"]

_CONFIG_DIR = Path(__file__).with_name("configs")
_CATALOG_FILENAMES = {
    "Q3_K": "mmq_bwd_q3_k_catalog.json",
    "Q4_K": "mmq_bwd_q4_k_catalog.json",
    "Q5_K": "mmq_bwd_q5_k_catalog.json",
    "Q6_K": "mmq_bwd_q6_k_catalog.json",
    "Q8_0": "mmq_bwd_q8_0_catalog.json",
}


@cache
def _catalog_backward_specs(quant_type: str) -> tuple[BackwardKernelSpec, ...]:
    catalog = load_catalog(_CONFIG_DIR / _CATALOG_FILENAMES[quant_type])
    specs = unique_kernel_specs(catalog)
    assert all(isinstance(spec, BackwardKernelSpec) for spec in specs)
    return tuple(spec for spec in specs if isinstance(spec, BackwardKernelSpec))


def _catalog_seed_for_shape(
    quant_type: str, shape: ProblemSize
) -> BackwardKernelSpec | None:
    valid = tuple(
        spec
        for spec in _catalog_backward_specs(quant_type)
        if is_valid_candidate(spec, quant_type, shape)
    )
    if not valid:
        return None
    if quant_type == "Q3_K":
        full_pairing = next(
            (spec for spec in valid if spec.decode.pairing is BackwardPairing.Full),
            None,
        )
        if full_pairing is not None:
            return full_pairing
    return valid[0]


@dataclass(frozen=True)
class BackwardCandidateDomain:
    quant_type: str
    problem_size: ProblemSize
    seed: BackwardKernelSpec
    knob_groups: tuple[BackwardSearchKnobGroup, ...]


def backward_candidate_mapping(
    candidate: BackwardKernelSpec, quant_type: str
) -> dict[str, object]:
    family = KernelFamily.OrdinaryBackward
    return {
        "KernelFamily": family.value,
        "Target": GFX1151_TARGET.to_mapping(),
        "ProblemType": problem_type_mapping(family, quant_type),
        "KernelSpec": candidate.to_mapping(quant_type),
    }


def backward_candidate_hash(candidate: BackwardKernelSpec, quant_type: str) -> str:
    return canonical_sha256(backward_candidate_mapping(candidate, quant_type))


def is_valid_candidate(
    candidate: BackwardKernelSpec,
    quant_type: str,
    shape: ProblemSize,
) -> bool:
    try:
        validate_backward_solution(shape, quant_type, candidate)
    except AssertionError:
        return False
    return True


def _decoder_neighbors(
    seed: BackwardKernelSpec,
    quant_type: str,
) -> tuple[BackwardKernelSpec, ...]:
    if quant_type == "Q3_K":
        return (
            replace(
                seed,
                decode=replace(
                    seed.decode,
                    extraction=BackwardExtraction.Packed,
                    pairing=BackwardPairing.Partial,
                ),
            ),
            replace(
                seed,
                decode=replace(
                    seed.decode,
                    extraction=BackwardExtraction.Packed,
                    pairing=BackwardPairing.Full,
                ),
            ),
            replace(
                seed,
                decode=replace(
                    seed.decode,
                    extraction=BackwardExtraction.Scalar,
                    pairing=None,
                ),
            ),
        )
    if quant_type == "Q4_K":
        return (
            replace(
                seed,
                pipeline=replace(
                    seed.pipeline, packed_load_grouping=PackedLoadGrouping.PerLane
                ),
            ),
            replace(
                seed,
                pipeline=replace(
                    seed.pipeline, packed_load_grouping=PackedLoadGrouping.LanePair
                ),
            ),
        )
    if quant_type == "Q5_K":
        return (
            replace(
                seed,
                decode=replace(seed.decode, extraction=BackwardExtraction.Packed),
            ),
            replace(
                seed,
                decode=replace(seed.decode, extraction=BackwardExtraction.Scalar),
            ),
            replace(
                seed,
                decode=replace(
                    seed.decode,
                    bitfield_shift_placement=BackwardBitfieldShiftPlacement.Hoisted,
                ),
            ),
            replace(
                seed,
                decode=replace(
                    seed.decode,
                    metadata_load=BackwardMetadataLoad.Vector,
                ),
            ),
            replace(
                seed,
                pipeline=replace(
                    seed.pipeline, packed_load_grouping=PackedLoadGrouping.LanePair
                ),
            ),
        )
    if quant_type == "Q6_K":
        return (
            replace(
                seed,
                decode=replace(seed.decode, extraction=BackwardExtraction.Packed),
            ),
            replace(
                seed,
                decode=replace(seed.decode, extraction=BackwardExtraction.PackedVopd),
            ),
            replace(
                seed,
                decode=replace(seed.decode, extraction=BackwardExtraction.Scalar),
            ),
            replace(
                seed,
                pipeline=replace(
                    seed.pipeline, packed_load_grouping=PackedLoadGrouping.LanePair
                ),
            ),
        )
    if quant_type == "Q8_0":
        return tuple(
            replace(
                seed,
                decode=replace(seed.decode, extraction=BackwardExtraction(extraction)),
            )
            for extraction in BackwardExtraction
        )
    raise ValueError(f"unsupported backward search quant type: {quant_type}")


def _pipeline_neighbors(seed: BackwardKernelSpec) -> tuple[BackwardKernelSpec, ...]:
    candidates = [
        replace(
            seed,
            pipeline=replace(
                seed.pipeline,
                lds_buffering=LdsBuffering.Single,
                iteration=BackwardIterationPolicy(False, False),
                global_read_prefetch=1,
                local_read_prefetch=1,
                prefetch_next_packed_weight=False,
            ),
        ),
        replace(
            seed,
            pipeline=replace(
                seed.pipeline,
                lds_buffering=LdsBuffering.Single,
                iteration=BackwardIterationPolicy(True, False),
                global_read_prefetch=2,
                local_read_prefetch=1,
                prefetch_next_packed_weight=False,
            ),
        ),
        replace(
            seed,
            memory=replace(seed.memory, lds_pad_b=0, lds_swizzle_chunk_b=8),
            pipeline=replace(
                seed.pipeline,
                lds_buffering=LdsBuffering.Double,
                iteration=BackwardIterationPolicy(True, False),
                global_read_prefetch=2,
                local_read_prefetch=1,
                packed_load_grouping=PackedLoadGrouping.PerLane,
                prefetch_next_packed_weight=False,
            ),
        ),
    ]
    if seed.geometry.depth_u == 32:
        candidates.append(
            replace(
                seed,
                pipeline=replace(
                    seed.pipeline,
                    lds_buffering=LdsBuffering.Single,
                    iteration=BackwardIterationPolicy(False, True),
                    global_read_prefetch=1,
                    local_read_prefetch=2,
                    prefetch_next_packed_weight=False,
                ),
            )
        )
    return tuple(candidates)


def _lds_layout_neighbors(seed: BackwardKernelSpec) -> tuple[BackwardKernelSpec, ...]:
    return (
        replace(
            seed,
            memory=replace(seed.memory, lds_pad_b=0, lds_swizzle_chunk_b=0),
        ),
        replace(
            seed,
            memory=replace(seed.memory, lds_pad_b=8, lds_swizzle_chunk_b=0),
        ),
        replace(
            seed,
            memory=replace(seed.memory, lds_pad_b=0, lds_swizzle_chunk_b=8),
        ),
    )


def candidate_neighbors(
    seed: BackwardKernelSpec,
    quant_type: str,
    shape: ProblemSize,
    knob_groups: tuple[BackwardSearchKnobGroup, ...],
) -> tuple[BackwardKernelSpec, ...]:
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
        if is_valid_candidate(candidate, quant_type, shape)
    )
    unique = {
        backward_candidate_hash(candidate, quant_type): candidate for candidate in valid
    }
    return tuple(unique[digest] for digest in sorted(unique))


def candidate_domains(
    quant_type: str,
    shape: ProblemSize,
) -> tuple[BackwardCandidateDomain, ...]:
    if quant_type not in _CATALOG_FILENAMES:
        raise ValueError(f"unsupported backward search quant type: {quant_type}")
    seed = _catalog_seed_for_shape(quant_type, shape)
    if seed is None:
        return ()
    return (
        BackwardCandidateDomain(
            quant_type=quant_type,
            problem_size=shape,
            seed=seed,
            knob_groups=("Decoder", "Pipeline", "LdsLayout"),
        ),
    )
