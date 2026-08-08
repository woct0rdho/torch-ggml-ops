import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools.ggtensile.cli import ManifestError
from tools.ggtensile.cli import main as ggtensile_cli_main
from tools.ggtensile.kernel_writer_assembly_mmq_fwd import (
    ForwardKernelWriterAssembly,
)
from tools.ggtensile.mmq_fwd_search import (
    Q6ExactPairManifest,
    candidate_domains,
    candidate_neighbors,
    canonical_candidate,
    explain_invalid,
    forward_candidate_hash,
    q6_candidate_hash,
    q6_schedule_candidates,
    q6_schedule_neighbors,
    q6_solution_with_schedule,
)
from tools.ggtensile.mmq_fwd_spec import Q6ForwardSchedule, q6_schedule_from_solution
from tools.ggtensile.model import (
    ForwardSolution,
    ProblemSize,
    ProblemType,
    SolutionKey,
)
from tools.ggtensile.toolchain import Toolchain
from tools.ggtensile.validation import validate_solution


def _q6_schedule(macro_tile0: int) -> Q6ForwardSchedule:
    return q6_schedule_from_solution(
        ForwardSolution.q6_k_structured_decoded(macro_tile0=macro_tile0)
    )


def test_q6_candidate_hash_covers_complete_schedule() -> None:
    selected = _q6_schedule(64)
    assert q6_candidate_hash(selected) == q6_candidate_hash(selected)
    assert q6_candidate_hash(selected) != q6_candidate_hash(
        replace(selected, global_read_cache_policy="Default")
    )
    mapping = json.loads(
        json.dumps(
            canonical_candidate(
                ForwardSolution.q6_k_structured_decoded(macro_tile0=64),
                "Q6_K",
            )
        )
    )
    assert mapping["ProblemContract"]["quant_type"] == "Q6_K"
    assert mapping["KernelSpec"]["ownership"]["mi_wave_tile"] == [1, 4]
    assert mapping["KernelSpec"]["epilogue"]["pipeline"]["scope"] == "StoreBatch"
    assert mapping["KernelSpec"]["resource_limits"]["max_lds_bytes"] == 65536


def test_generic_manual_profile_keeps_shape_and_linked_domains_explicit() -> None:
    shape = ProblemSize(192, 248320, 2048)
    domains = candidate_domains("Q6_K", shape)
    assert len(domains) == 1
    assert domains[0].kernel_spec.macro_tile == (64, 64)
    candidates = candidate_neighbors(domains[0].seed, domains[0].knob_groups)
    assert len(candidates) == 32
    assert all(not explain_invalid(item, "Q6_K", shape) for item in candidates)
    seed = ForwardSolution.q6_k_structured_decoded(macro_tile0=64)
    assert len(candidate_neighbors(seed, ("InstructionPolicy",))) == 4
    assert forward_candidate_hash(seed, "Q6_K") == forward_candidate_hash(seed, "Q6_K")
    assert explain_invalid(seed, "Q6_K", ProblemSize(65, 248320, 2048))
    with pytest.raises(ValueError, match="Q4_K, Q5_K, and Q6_K"):
        candidate_domains("Q8_0", shape)


def test_decoded_weight_lds_domains_expose_only_implemented_policies() -> None:
    shape = ProblemSize(2048, 512, 2048)
    q4_domain = candidate_domains("Q4_K", shape)[0]
    q5_domain = candidate_domains("Q5_K", shape)[0]
    assert (
        q4_domain.kernel_spec.global_memory.operand_source == "DecodedWeightLdsBatch8"
    )
    assert len(candidate_neighbors(q4_domain.seed, ("Metadata",))) == 4
    assert len(candidate_neighbors(q5_domain.seed, ("Metadata",))) == 3
    assert len(candidate_neighbors(q4_domain.seed, ("InstructionPolicy",))) == 2
    assert len(candidate_neighbors(q5_domain.seed, ("Epilogue",))) == 256
    assert all(
        not explain_invalid(item, "Q5_K", shape)
        for item in candidate_neighbors(q5_domain.seed, ("Metadata",))
    )


def test_q6_manual_neighborhoods_are_linked_and_deterministic() -> None:
    j64 = _q6_schedule(64)
    j128 = _q6_schedule(128)
    assert q6_schedule_neighbors(j64, ()) == (j64,)
    assert len(q6_schedule_neighbors(j64, ("InstructionPolicy",))) == 4
    assert len(q6_schedule_neighbors(j128, ("InstructionPolicy",))) == 8
    assert len(q6_schedule_neighbors(j64, ("Epilogue",))) == 8
    assert len(q6_schedule_candidates(64)) == 32
    assert len(q6_schedule_candidates(128)) == 64
    assert q6_schedule_candidates(64) == q6_schedule_candidates(64)
    assert {
        candidate.semantic_policy.traversal for candidate in q6_schedule_candidates(64)
    } == {"OutputRoleGroupMajor", "OutputRoleWavefront"}


def test_q6_manual_candidate_uses_normal_solution_and_writer_path() -> None:
    base = ForwardSolution.q6_k_structured_decoded(macro_tile0=64)
    schedule = replace(
        q6_schedule_from_solution(base),
        global_read_cache_policy="Default",
        epilogue_dependency_width=4,
    )
    solution = q6_solution_with_schedule(base, schedule)
    assert q6_schedule_from_solution(solution) == schedule
    key = SolutionKey(
        ProblemType.mmq_forward("Q6_K"),
        ProblemSize(64, 248320, 2048),
        solution,
    )
    assert validate_solution(key) == ()
    source = ForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "buffer_gl0_inv" not in source
    assert solution.to_mapping()["EpilogueDependencyWidth"] == 4


def test_generic_manual_enumeration_writes_q4_q5_pair_manifests(tmp_path: Path) -> None:
    for quant_type, expected_count in (("Q4_K", 4), ("Q5_K", 3)):
        root = tmp_path / quant_type
        arguments = [
            "enumerate-forward",
            "--quant-type",
            quant_type,
            "--m",
            "2048",
            "--n",
            "512",
            "--k",
            "2048",
            "--knob-group",
            "Metadata",
            "--output-dir",
            str(root),
        ]
        assert ggtensile_cli_main(arguments) == 0
        index = json.loads((root / "index.json").read_text(encoding="utf-8"))
        assert index["CandidateCount"] == expected_count
        for entry in index["Candidates"]:
            candidate_dir = root / entry["Directory"]
            key = SolutionKey.from_json_file(candidate_dir / "solution-key.json")
            assert key.problem_type.quant_data_type == quant_type
            assert validate_solution(key) == ()


def test_q6_manual_enumeration_writes_normal_solution_keys(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    arguments = [
        "enumerate-forward-q6",
        "--macro-tile",
        "64",
        "--m",
        "192",
        "--n",
        "248320",
        "--k",
        "2048",
        "--knob-group",
        "InstructionPolicy",
        "--output-dir",
        str(root),
    ]
    assert ggtensile_cli_main(arguments) == 0
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    assert index["CandidateCount"] == 4
    for entry in index["Candidates"]:
        candidate_dir = root / entry["Directory"]
        key = SolutionKey.from_json_file(candidate_dir / "solution-key.json")
        assert validate_solution(key) == ()
        assert key.problem_size == ProblemSize(192, 248320, 2048)
    with pytest.raises(ManifestError, match="nonempty"):
        ggtensile_cli_main(arguments)


def test_q6_exact_pair_manifest_separates_candidate_and_shape_identity() -> None:
    schedule = _q6_schedule(128)
    first = Q6ExactPairManifest(ProblemSize(128, 248320, 2048), schedule)
    second = Q6ExactPairManifest(ProblemSize(256, 248320, 2048), schedule)
    assert first.candidate_hash == second.candidate_hash
    assert first.exact_pair_hash != second.exact_pair_hash
    mapping = json.loads(json.dumps(first.to_mapping()))
    assert mapping["CandidateHash"] == first.candidate_hash
    assert mapping["SolutionKey"]["Solution"]["Q6EpiloguePipelineScope"] == "FullTile"
