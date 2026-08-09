from dataclasses import replace
from typing import Any, cast

import pytest

from tools.ggtensile.mmq_bwd_search import (
    backward_candidate_hash,
    candidate_domains,
    candidate_neighbors,
    explain_invalid,
)
from tools.ggtensile.model import BackwardSolution, ProblemSize


def test_backward_domains_are_linked_deterministic_and_capability_valid() -> None:
    shape = ProblemSize(128, 256, 128)
    for quant_type in ("Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"):
        domain = candidate_domains(quant_type, shape)[0]
        candidates = candidate_neighbors(
            domain.seed,
            quant_type,
            shape,
            domain.knob_groups,
        )
        assert 2 <= len(candidates) < 16
        assert candidates == candidate_neighbors(
            domain.seed,
            quant_type,
            shape,
            domain.knob_groups,
        )
        assert len({backward_candidate_hash(item) for item in candidates}) == len(
            candidates
        )
        assert all(not explain_invalid(item, quant_type, shape) for item in candidates)


def test_q3_neighbors_serialize_pairing_as_complete_policy() -> None:
    shape = ProblemSize(128, 256, 128)
    domain = candidate_domains("Q3_K", shape)[0]
    candidates = candidate_neighbors(domain.seed, "Q3_K", shape, ("Decoder",))
    policies = {
        (candidate.q3_k_extraction, candidate.q3_k_pairing) for candidate in candidates
    }
    assert policies == {
        ("packed", "Partial"),
        ("packed", "Full"),
        ("scalar", "Inactive"),
    }
    assert all("Q3KPairing" in candidate.to_mapping() for candidate in candidates)


def test_backward_search_rejects_unknown_domains_and_groups() -> None:
    shape = ProblemSize(128, 256, 128)
    with pytest.raises(ValueError, match="unsupported backward search quant type"):
        candidate_domains("Q2_K", shape)
    with pytest.raises(ValueError, match="unsupported backward search quant type"):
        candidate_neighbors(BackwardSolution.pilot(), "Q2_K", shape, ("Decoder",))
    with pytest.raises(ValueError, match="unsupported backward search knob groups"):
        candidate_neighbors(
            BackwardSolution.pilot(),
            "Q4_K",
            shape,
            cast(Any, ("InstructionOrder",)),
        )


def test_backward_candidate_identity_covers_every_serialized_policy() -> None:
    seed = BackwardSolution.pilot()
    assert backward_candidate_hash(seed) == backward_candidate_hash(seed)
    assert backward_candidate_hash(seed) != backward_candidate_hash(
        replace(seed, q8_0_extraction="packed_vopd")
    )
