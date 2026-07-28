#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ggtensile.campaign import (  # noqa: E402
    DEFAULT_INVENTORY,
    DEFAULT_RETAINED_SOLUTION,
    CampaignEntry,
    CampaignError,
    CampaignInventory,
    load_inventory,
    load_solution,
)
from tools.ggtensile.cli import main as ggtensile_cli_main  # noqa: E402
from tools.ggtensile.model import ProblemSize, Solution, SolutionKey  # noqa: E402
from tools.ggtensile.validation import validate_solution  # noqa: E402

BENCHMARK = REPO_ROOT / "tools" / "benchmark_ggtensile.py"
DEFAULT_MODEL = Path("/home/wd/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf")
_PHASE_PROTOCOL = {
    "correctness": ("Correctness", 2, 1),
    "screen": ("Screen", 10, 9),
    "confirmation": ("Confirmation", 10, 25),
}


def _problem_size(value: str) -> ProblemSize:
    try:
        fields = tuple(int(field) for field in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("key must be M,N,K") from error
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("key must be M,N,K")
    return ProblemSize(*fields)


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--key", type=_problem_size, action="append", default=[])
    parser.add_argument("--artifact-root", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run serial immutable phases over exact dense Q4_K production keys"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    _add_selection_arguments(prepare)
    prepare.add_argument("--solution", type=Path, default=DEFAULT_RETAINED_SOLUTION)

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


def _selected(
    inventory: CampaignInventory, arguments: argparse.Namespace
) -> tuple[CampaignEntry, ...]:
    return inventory.selected(
        families=tuple(arguments.family), sizes=tuple(arguments.key)
    )


def _artifact_dir(root: Path, entry: CampaignEntry) -> Path:
    return root / entry.slug


def _prepare(
    arguments: argparse.Namespace,
    inventory: CampaignInventory,
    entries: tuple[CampaignEntry, ...],
) -> int:
    root = arguments.artifact_root.resolve()
    if root.exists():
        raise CampaignError(f"refusing to overwrite artifact root {root}")
    solution: Solution = load_solution(arguments.solution)
    keys = [
        entry.solution_key(inventory.problem_type, solution) for entry in entries
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
        artifact = _artifact_dir(root, entry)
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
            "Inventory": str(arguments.inventory.resolve()),
            "Solution": str(arguments.solution.resolve()),
            "Entries": prepared,
        },
    )
    return 0


def _load_report(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read benchmark report {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise CampaignError(f"benchmark report {path} is not a JSON object")
    return value


def _timing_median(report: Mapping[str, object], name: str) -> float:
    timing = report.get(name)
    if not isinstance(timing, Mapping):
        raise CampaignError(f"benchmark report lacks {name}")
    value = timing.get("median_ms")
    if type(value) not in (int, float):
        raise CampaignError(f"benchmark report has invalid {name}.median_ms")
    return float(value)


def _check_correctness(report: Mapping[str, object], *, has_control: bool) -> None:
    correctness = report.get("Correctness")
    if not isinstance(correctness, Mapping):
        raise CampaignError("benchmark report lacks Correctness")
    required = {
        "candidate_vs_hip",
        "candidate_after_grad_output_update_vs_hip",
        "candidate_after_packed_weight_update_vs_hip",
    }
    if has_control:
        required.add("assembly_control_vs_hip")
    missing = required - correctness.keys()
    if missing:
        raise CampaignError(f"benchmark report lacks correctness rows {sorted(missing)}")
    for name in required:
        metrics = correctness[name]
        if not isinstance(metrics, Mapping):
            raise CampaignError(f"invalid correctness row {name}")
        if metrics.get("different_bf16_elements") != 0 or metrics.get("finite") is not True:
            raise CampaignError(f"correctness row {name} is not bit-exact and finite")


def _run_benchmark(command: list[str], *, entry: CampaignEntry) -> None:
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode:
        details = result.stderr.strip() or result.stdout.strip()
        raise CampaignError(
            f"benchmark failed for {entry.slug} ({result.returncode}): {details[-2000:]}"
        )


def _measure(
    arguments: argparse.Namespace,
    inventory: CampaignInventory,
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
        artifact = _artifact_dir(root, entry)
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
            control = _artifact_dir(control_root, entry)
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
        try:
            reported_key = SolutionKey.from_mapping(report["SolutionKey"])
        except (KeyError, TypeError, ValueError) as error:
            raise CampaignError(f"invalid SolutionKey in {report_path}: {error}") from error
        if reported_key != expected_key or report.get("Tensor") != entry.representative_tensor:
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
            "HistoricalHipMedianMs": entry.historical_hip_median_ms,
            "HipMedianMs": hip_ms,
            "CandidateMedianMs": candidate_ms,
            "CandidateToHipLatency": candidate_ms / hip_ms,
            "Report": str(report_path),
        }
        if control_ms is not None:
            observation["AssemblyControlMedianMs"] = control_ms
            observation["CandidateToAssemblyControlLatency"] = (
                candidate_ms / control_ms
            )
        observations.append(observation)

    summary: dict[str, object] = {
        "Phase": phase_name,
        "Status": "Accepted",
        "Inventory": str(arguments.inventory.resolve()),
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
    try:
        inventory = load_inventory(arguments.inventory)
        entries = _selected(inventory, arguments)
        if arguments.command == "prepare":
            return _prepare(arguments, inventory, entries)
        return _measure(arguments, inventory, entries)
    except (CampaignError, FileExistsError) as error:
        print(f"ggtensile campaign {arguments.command}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
