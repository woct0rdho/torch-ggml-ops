from dataclasses import replace
from typing import Any, cast

import pytest

from tools.ggtensile.mmq_bwd_search import (
    backward_candidate_hash,
    backward_candidate_mapping,
    candidate_domains,
    candidate_neighbors,
    is_valid_candidate,
)
from tools.ggtensile.mmq_bwd_spec import (
    BackwardExtraction,
    BackwardKernelSpec,
    BackwardPairing,
)
from tools.ggtensile.model import ProblemSize


def test_backward_domains_are_linked_deterministic_and_typed() -> None:
    shape = ProblemSize(128, 256, 128)
    for quant_type in ("Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"):
        domain = candidate_domains(quant_type, shape)[0]
        candidates = candidate_neighbors(
            domain.seed, quant_type, shape, domain.knob_groups
        )
        assert candidates == candidate_neighbors(
            domain.seed, quant_type, shape, domain.knob_groups
        )
        assert len(
            {backward_candidate_hash(item, quant_type) for item in candidates}
        ) == len(candidates)
        assert all(isinstance(item, BackwardKernelSpec) for item in candidates)
        assert all(is_valid_candidate(item, quant_type, shape) for item in candidates)


def test_q3_neighbors_serialize_pairing_as_complete_policy() -> None:
    shape = ProblemSize(128, 256, 128)
    domain = candidate_domains("Q3_K", shape)[0]
    candidates = candidate_neighbors(domain.seed, "Q3_K", shape, ("Decoder",))
    policies = {
        (candidate.decode.extraction, candidate.decode.pairing)
        for candidate in candidates
    }
    assert policies == {
        (BackwardExtraction.Packed, BackwardPairing.Partial),
        (BackwardExtraction.Packed, BackwardPairing.Full),
        (BackwardExtraction.Scalar, None),
    }
    for candidate in candidates:
        kernel_spec = backward_candidate_mapping(candidate, "Q3_K")["KernelSpec"]
        assert isinstance(kernel_spec, dict)
        decode = kernel_spec.get("Decode")
        assert isinstance(decode, dict)


def test_backward_search_rejects_unknown_domains_and_groups() -> None:
    shape = ProblemSize(128, 256, 128)
    seed = candidate_domains("Q4_K", shape)[0].seed
    with pytest.raises(ValueError, match="unsupported backward search quant type"):
        candidate_domains("Q2_K", shape)
    with pytest.raises(ValueError, match="unsupported backward search quant type"):
        candidate_neighbors(seed, "Q2_K", shape, ("Decoder",))
    with pytest.raises(ValueError, match="unsupported backward search knob groups"):
        candidate_neighbors(
            seed,
            "Q4_K",
            shape,
            cast(Any, ("InstructionOrder",)),
        )


def test_backward_candidate_identity_covers_serialized_policy() -> None:
    seed = candidate_domains("Q8_0", ProblemSize(128, 256, 128))[0].seed
    changed = replace(
        seed,
        decode=replace(seed.decode, extraction=BackwardExtraction.PackedVopd),
    )
    assert backward_candidate_hash(seed, "Q8_0") != backward_candidate_hash(
        changed, "Q8_0"
    )
