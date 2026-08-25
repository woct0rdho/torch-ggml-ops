import json
from dataclasses import replace

import pytest

from tests.ggtensile.ordinary_forward_fixtures import q6_structured_kernel_spec
from tools.ggtensile.mmq_fwd_search import (
    Q6ExactPairManifest,
    candidate_domains,
    candidate_neighbors,
    canonical_candidate,
    forward_candidate_hash,
    is_valid_candidate,
    q6_candidate_hash,
    q6_schedule_candidates,
    q6_schedule_from_kernel_spec,
    q6_schedule_neighbors,
)
from tools.ggtensile.mmq_fwd_spec import ForwardKernelSpec
from tools.ggtensile.model import ProblemSize


def test_q6_candidate_hash_covers_complete_schedule() -> None:
    selected = q6_schedule_from_kernel_spec(q6_structured_kernel_spec(64))
    changed = replace(selected, global_read_cache_policy="Default")
    assert q6_candidate_hash(selected) != q6_candidate_hash(changed)
    mapping = canonical_candidate(q6_structured_kernel_spec(64), "Q6_K")
    problem_type = mapping["ProblemType"]
    kernel_spec = mapping["KernelSpec"]
    assert isinstance(problem_type, dict)
    assert isinstance(kernel_spec, dict)
    ownership = kernel_spec["Ownership"]
    assert isinstance(ownership, dict)
    assert problem_type["QuantDataType"] == "Q6_K"
    assert ownership["MIWaveTile"] == [1, 4]
    assert "HardwareResourceCapacity" not in kernel_spec


def test_forward_domains_are_linked_and_shape_validated() -> None:
    shape = ProblemSize(192, 248320, 2048)
    domains = candidate_domains("Q6_K", shape)
    assert domains
    for domain in domains:
        candidates = candidate_neighbors(domain.kernel_spec, "Q6_K", domain.knob_groups)
        assert candidates == candidate_neighbors(
            domain.kernel_spec, "Q6_K", domain.knob_groups
        )
        assert all(is_valid_candidate(item, "Q6_K", shape) for item in candidates)
        assert all(isinstance(item, ForwardKernelSpec) for item in candidates)
    seed = domains[0].kernel_spec
    assert forward_candidate_hash(seed, "Q6_K") == forward_candidate_hash(seed, "Q6_K")
    assert not is_valid_candidate(seed, "Q6_K", ProblemSize(65, 248320, 2048))
    with pytest.raises(ValueError, match="Q3_K, Q4_K, Q5_K, Q6_K, and Q8_0"):
        candidate_domains("Q2_K", shape)


def test_q3_and_q8_domains_have_no_free_knobs() -> None:
    q3_domains = candidate_domains("Q3_K", ProblemSize(128, 64, 256))
    assert q3_domains
    assert all(domain.knob_groups == () for domain in q3_domains)
    for domain in q3_domains:
        assert candidate_neighbors(domain.kernel_spec, "Q3_K", ()) == (
            domain.kernel_spec,
        )
    q8_domains = candidate_domains("Q8_0", ProblemSize(64, 64, 128))
    assert q8_domains
    with pytest.raises(ValueError, match="without free knob groups"):
        candidate_neighbors(q8_domains[0].kernel_spec, "Q8_0", ("Epilogue",))


def test_decoded_weight_domains_expose_typed_policy_neighbors() -> None:
    shape = ProblemSize(2048, 512, 2048)
    for quant_type in ("Q4_K", "Q5_K"):
        domain = candidate_domains(quant_type, shape)[0]
        candidates = candidate_neighbors(domain.kernel_spec, quant_type, ("Metadata",))
        assert candidates
        assert all(is_valid_candidate(item, quant_type, shape) for item in candidates)
        assert all(isinstance(item, ForwardKernelSpec) for item in candidates)


def test_q6_manual_neighborhoods_are_deterministic() -> None:
    j64 = q6_schedule_from_kernel_spec(q6_structured_kernel_spec(64))
    assert q6_schedule_neighbors(j64, ()) == (j64,)
    candidates = q6_schedule_candidates(64)
    assert candidates == q6_schedule_candidates(64)
    assert {candidate.semantic_policy.traversal for candidate in candidates} == {
        "OutputRoleGroupMajor",
        "OutputRoleWavefront",
    }


def test_q6_exact_pair_manifest_separates_candidate_and_shape_identity() -> None:
    schedule = q6_schedule_from_kernel_spec(q6_structured_kernel_spec(128))
    first = Q6ExactPairManifest(ProblemSize(128, 248320, 2048), schedule)
    second = Q6ExactPairManifest(ProblemSize(256, 248320, 2048), schedule)
    assert first.candidate_hash == second.candidate_hash
    assert first.exact_pair_hash != second.exact_pair_hash
    mapping = json.loads(json.dumps(first.to_mapping()))
    assert mapping["CandidateHash"] == first.candidate_hash
    assert (
        mapping["KernelSpecKey"]["KernelSpec"]["Epilogue"]["Pipeline"]["Scope"]
        == "FullTile"
    )
