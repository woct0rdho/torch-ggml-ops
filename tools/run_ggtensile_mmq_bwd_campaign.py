#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ggtensile.campaign import (
    CatalogEntry,
    CatalogError,
    DeploymentCatalog,
    load_catalog,
    load_solution,
)
from tools.ggtensile.cli import main as ggtensile_cli_main
from tools.ggtensile.model import BackwardSolution, ProblemSize, SolutionKey
from tools.ggtensile.validation import validate_solution

BENCHMARK = REPO_ROOT / "tools" / "benchmark_ggtensile_mmq_bwd.py"
CONFIG_DIR = REPO_ROOT / "tools" / "ggtensile" / "configs"
DEFAULT_CATALOG = CONFIG_DIR / "mmq_bwd_q4_k_catalog.json"
DEFAULT_MODEL = Path.home() / "models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"
_PHASE_PROTOCOL = {
    "correctness": ("Correctness", 2, 1),
    "screen": ("Screen", 10, 9),
    "confirmation": ("Confirmation", 10, 25),
}


CampaignError = CatalogError


@dataclass(frozen=True)
class CampaignEntry:
    """Benchmark context kept outside the deployment catalog."""

    catalog_entry: CatalogEntry
    family: str
    representative_tensor: str
    call_count: int

    @property
    def problem_size(self) -> ProblemSize:
        return self.catalog_entry.problem_size

    @property
    def solution(self) -> BackwardSolution:
        solution = self.catalog_entry.solution
        if not isinstance(solution, BackwardSolution):
            raise CampaignError("MMQ backward campaign requires backward solutions")
        return solution

    @property
    def slug(self) -> str:
        size = self.problem_size
        return f"m{size.m}_n{size.n}_k{size.k}"


_BWD_WORKLOADS: dict[
    str, tuple[tuple[str, tuple[int, int], str, int, tuple[int, ...]], ...]
] = {
    "Q3_K": (
        ("narrow", (2048, 512), "blk.3.attn_k.weight", 9, (2048, 8192, 32768)),
        ("query", (2048, 8192), "blk.3.attn_q.weight", 9, (2048, 8192, 32768)),
    ),
    "Q4_K": (
        ("narrow", (2048, 512), "blk.5.ffn_gate_shexp.weight", 70, (2048, 8192, 32768)),
        (
            "shared_down",
            (512, 2048),
            "blk.5.ffn_down_shexp.weight",
            30,
            (2048, 8192, 32768),
        ),
        (
            "attention_output",
            (4096, 2048),
            "blk.3.attn_output.weight",
            10,
            (2048, 8192, 32768),
        ),
        ("query", (2048, 8192), "blk.39.attn_q.weight", 1, (2048, 8192, 32768)),
    ),
    "Q5_K": (
        ("narrow", (2048, 512), "blk.0.ffn_gate_shexp.weight", 21, (2048, 8192, 32768)),
        (
            "shared_down",
            (512, 2048),
            "blk.0.ffn_down_shexp.weight",
            10,
            (2048, 8192, 32768),
        ),
    ),
    "Q6_K": (("lm_head", (2048, 248320), "output.weight", 1, (64, 128, 256)),),
    "Q8_0": (
        (
            "attention_q_a",
            (4096, 1024),
            "blk.0.attn_q_a.weight",
            43,
            (2048, 8192, 32768),
        ),
        (
            "attention_q_b",
            (1024, 32768),
            "blk.0.attn_q_b.weight",
            43,
            (2048, 8192, 32768),
        ),
        ("attention_kv", (4096, 512), "blk.0.attn_kv.weight", 43, (2048, 8192, 32768)),
        (
            "attention_output_b",
            (8192, 4096),
            "blk.0.attn_output_b.weight",
            43,
            (2048, 8192, 32768),
        ),
        (
            "shared_gate_up",
            (4096, 2048),
            "blk.0.ffn_gate_shexp.weight",
            86,
            (2048, 8192, 32768),
        ),
        (
            "shared_down",
            (2048, 4096),
            "blk.0.ffn_down_shexp.weight",
            43,
            (2048, 8192, 32768),
        ),
        ("lm_head", (4096, 129280), "output.weight", 1, (32, 64, 128, 256, 512)),
    ),
}


def _campaign_entries(catalog: DeploymentCatalog) -> tuple[CampaignEntry, ...]:
    workloads = _BWD_WORKLOADS.get(catalog.problem_type.quant_data_type)
    if workloads is None:
        raise CampaignError(
            f"no backward campaign workload is defined for "
            f"{catalog.problem_type.quant_data_type}"
        )
    by_shape = {
        (n, k): (family, tensor, calls, m_values)
        for family, (n, k), tensor, calls, m_values in workloads
    }
    entries: list[CampaignEntry] = []
    for catalog_entry in catalog.entries:
        size = catalog_entry.problem_size
        workload = by_shape.get((size.n, size.k))
        if workload is None or size.m not in workload[3]:
            raise CampaignError(
                f"catalog key has no backward workload descriptor: {size.to_mapping()}"
            )
        family, tensor, calls, _ = workload
        entries.append(CampaignEntry(catalog_entry, family, tensor, calls))
    return tuple(entries)


def _problem_size(value: str) -> ProblemSize:
    fields = tuple(int(field) for field in value.split(","))
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("key must be M,N,K")
    return ProblemSize(*fields)


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--key", type=_problem_size, action="append", default=[])
    parser.add_argument("--artifact-root", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run serial immutable phases over exact MMQ backward production keys"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    _add_selection_arguments(prepare)
    solution_source = prepare.add_mutually_exclusive_group()
    solution_source.add_argument("--solution", type=Path)

    for command in _PHASE_PROTOCOL:
        phase = subparsers.add_parser(command)
        _add_selection_arguments(phase)
        phase.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        phase.add_argument("--assembly-control-root", type=Path)
        phase.add_argument("--with-reference", action="store_true")
    return parser


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _prepare(
    arguments: argparse.Namespace,
    catalog: DeploymentCatalog,
    entries: tuple[CampaignEntry, ...],
) -> int:
    root = arguments.artifact_root.resolve()
    if root.exists():
        raise CampaignError(f"refusing to overwrite artifact root {root}")
    source_mapping: dict[str, object]
    if arguments.solution is None:
        solutions = [entry.solution for entry in entries]
        source_mapping = {"Catalog": str(arguments.catalog.resolve())}
    else:
        solution_path = arguments.solution
        solution = load_solution(
            solution_path,
            problem_type=catalog.problem_type,
        )
        if not isinstance(solution, BackwardSolution):
            raise CampaignError("MMQ backward campaign requires a backward solution")
        solutions = [solution] * len(entries)
        source_mapping = {"Solution": str(solution_path.resolve())}
    keys = [
        SolutionKey(catalog.problem_type, entry.problem_size, solution)
        for entry, solution in zip(entries, solutions, strict=True)
    ]
    rejected = [
        (entry.slug, tuple(reason.to_mapping() for reason in validate_solution(key)))
        for entry, key in zip(entries, keys, strict=True)
        if validate_solution(key)
    ]
    if rejected:
        raise CampaignError(f"solution is invalid for selected keys: {rejected}")
    root.mkdir(parents=True)

    prepared: list[dict[str, object]] = []
    for entry, key in zip(entries, keys, strict=True):
        artifact = root / entry.slug
        request = artifact / "requested.json"
        _write_json_exclusive(request, key.to_mapping())
        phases = (
            (
                "generate",
                "--solution-key",
                str(request),
                "--output-dir",
                str(artifact),
            ),
            ("build", "--generate-manifest", str(artifact / "generate.json")),
            ("inspect", "--build-manifest", str(artifact / "build.json")),
        )
        for phase in phases:
            result = ggtensile_cli_main(list(phase))
            if result:
                raise CampaignError(
                    f"{phase[0]} rejected {entry.slug}; inspect {artifact}"
                )
        prepared.append(
            {
                "Family": entry.family,
                "ProblemSize": entry.problem_size.to_mapping(),
                "RepresentativeTensor": entry.representative_tensor,
                "CallCount": entry.call_count,
                "SolutionHash": key.hash,
                "KernelName": key.kernel_name,
                "ArtifactDirectory": str(artifact),
                "GenerateManifest": str(artifact / "generate.json"),
                "BuildManifest": str(artifact / "build.json"),
                "InspectManifest": str(artifact / "inspect.json"),
            }
        )

    _write_json_exclusive(
        root / "prepare.json",
        {
            "Phase": "Prepare",
            "Status": "Accepted",
            "Catalog": str(arguments.catalog.resolve()),
            **source_mapping,
            "Entries": prepared,
        },
    )
    return 0


def _load_report(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CampaignError(f"benchmark report {path} is not a JSON object")
    return {str(key): item for key, item in value.items()}


def _timing_median(report: Mapping[str, object], name: str) -> float:
    timing = report.get(name)
    if not isinstance(timing, Mapping):
        raise CampaignError(f"benchmark report lacks {name}")
    value = timing.get("median_ms")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CampaignError(f"benchmark report has invalid {name}.median_ms")
    return float(value)


def _check_correctness(report: Mapping[str, object], *, has_control: bool) -> None:
    correctness_value = report.get("Correctness")
    if not isinstance(correctness_value, dict) or not all(
        isinstance(key, str) for key in correctness_value
    ):
        raise CampaignError("benchmark report lacks Correctness")
    correctness = {str(key): value for key, value in correctness_value.items()}
    required = {
        "candidate_vs_hip",
        "candidate_after_grad_output_update_vs_hip",
        "candidate_after_packed_weight_update_vs_hip",
    }
    if has_control:
        required.add("assembly_control_vs_hip")
    missing = required - correctness.keys()
    if missing:
        raise CampaignError(
            f"benchmark report lacks correctness rows {sorted(missing)}"
        )
    for name in required:
        metrics = correctness[name]
        if not isinstance(metrics, Mapping):
            raise CampaignError(f"invalid correctness row {name}")
        if (
            metrics.get("different_bf16_elements") != 0
            or metrics.get("finite") is not True
        ):
            raise CampaignError(f"correctness row {name} is not bit-exact and finite")


def _run_benchmark(command: list[str], *, entry: CampaignEntry) -> None:
    result = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode:
        details = result.stderr.strip() or result.stdout.strip()
        raise CampaignError(
            f"benchmark failed for {entry.slug} ({result.returncode}): {details[-2000:]}"
        )


def _measure(
    arguments: argparse.Namespace,
    catalog: DeploymentCatalog,
    entries: tuple[CampaignEntry, ...],
) -> int:
    phase_name, warmup, repeats = _PHASE_PROTOCOL[arguments.command]
    root = arguments.artifact_root.resolve()
    if not root.is_dir():
        raise CampaignError(f"artifact root does not exist: {root}")
    summary_path = root / f"{arguments.command}.json"
    if summary_path.exists():
        raise CampaignError(f"refusing to overwrite {summary_path}")
    control_root = (
        arguments.assembly_control_root.resolve()
        if arguments.assembly_control_root is not None
        else None
    )

    observations: list[dict[str, object]] = []
    weighted_hip = 0.0
    weighted_candidate = 0.0
    weighted_control = 0.0
    for entry in entries:
        artifact = root / entry.slug
        solution_path = artifact / "solution.json"
        code_object = artifact / "kernel.hsaco"
        inspect_manifest = artifact / "inspect.json"
        for required in (solution_path, code_object, inspect_manifest):
            if not required.is_file():
                raise CampaignError(f"missing prepared artifact {required}")
        report_path = artifact / f"{arguments.command}.json"
        if report_path.exists():
            raise CampaignError(f"refusing to overwrite {report_path}")
        command = [
            sys.executable,
            str(BENCHMARK),
            "--solution-key",
            str(solution_path),
            "--code-object",
            str(code_object),
            "--output",
            str(report_path),
            "--model",
            str(arguments.model),
            "--tensor",
            entry.representative_tensor,
            "--warmup",
            str(warmup),
            "--repeats",
            str(repeats),
        ]
        if not arguments.with_reference:
            command.append("--skip-reference")
        if control_root is not None:
            control = control_root / entry.slug
            command.extend(
                (
                    "--assembly-control-solution-key",
                    str(control / "solution.json"),
                    "--assembly-control-code-object",
                    str(control / "kernel.hsaco"),
                )
            )
        _run_benchmark(command, entry=entry)
        report = _load_report(report_path)
        expected_key = SolutionKey.from_json_file(solution_path)
        reported_key = SolutionKey.from_mapping(report["SolutionKey"])
        if (
            reported_key != expected_key
            or report.get("Tensor") != entry.representative_tensor
        ):
            raise CampaignError(f"benchmark identity mismatch in {report_path}")
        _check_correctness(report, has_control=control_root is not None)
        hip_ms = _timing_median(report, "HIP")
        candidate_ms = _timing_median(report, "GGTensile")
        control_ms = (
            _timing_median(report, "AssemblyControl")
            if control_root is not None
            else None
        )
        weighted_hip += entry.call_count * hip_ms
        weighted_candidate += entry.call_count * candidate_ms
        if control_ms is not None:
            weighted_control += entry.call_count * control_ms
        observation: dict[str, object] = {
            "Family": entry.family,
            "ProblemSize": entry.problem_size.to_mapping(),
            "RepresentativeTensor": entry.representative_tensor,
            "CallCount": entry.call_count,
            "HipMedianMs": hip_ms,
            "CandidateMedianMs": candidate_ms,
            "CandidateToHipLatency": candidate_ms / hip_ms,
            "Report": str(report_path),
        }
        if control_ms is not None:
            observation["AssemblyControlMedianMs"] = control_ms
            observation["CandidateToAssemblyControlLatency"] = candidate_ms / control_ms
        observations.append(observation)

    summary: dict[str, object] = {
        "Phase": phase_name,
        "Status": "Accepted",
        "Catalog": str(arguments.catalog.resolve()),
        "Model": str(arguments.model.resolve()),
        "Protocol": {
            "Warmup": warmup,
            "Repeats": repeats,
            "Serial": True,
            "RotatingControls": True,
            "IndependentReference": bool(arguments.with_reference),
        },
        "WeightedHipMedianMs": weighted_hip,
        "WeightedCandidateMedianMs": weighted_candidate,
        "WeightedCandidateToHipLatency": weighted_candidate / weighted_hip,
        "Entries": observations,
    }
    if control_root is not None:
        summary["AssemblyControlRoot"] = str(control_root)
        summary["WeightedAssemblyControlMedianMs"] = weighted_control
        summary["WeightedCandidateToAssemblyControlLatency"] = (
            weighted_candidate / weighted_control
        )
    _write_json_exclusive(summary_path, summary)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    catalog = load_catalog(arguments.catalog)
    campaign_entries = _campaign_entries(catalog)
    family_filter = set(arguments.family)
    unknown_families = family_filter - {entry.family for entry in campaign_entries}
    if unknown_families:
        raise CampaignError(f"unknown families: {sorted(unknown_families)}")
    size_filter = set(arguments.key)
    unknown_sizes = [
        size.to_mapping()
        for size in arguments.key
        if size not in {entry.problem_size for entry in campaign_entries}
    ]
    if unknown_sizes:
        raise CampaignError(f"sizes are not in the deployment catalog: {unknown_sizes}")
    entries = tuple(
        entry
        for entry in campaign_entries
        if (not family_filter or entry.family in family_filter)
        and (not size_filter or entry.problem_size in size_filter)
    )
    if not entries:
        raise CampaignError("campaign selection is empty")
    if arguments.command == "prepare":
        return _prepare(arguments, catalog, entries)
    return _measure(arguments, catalog, entries)


if __name__ == "__main__":
    raise SystemExit(main())
