"""Shared protocol, CLI, and reporting for MMQ benchmarks."""

import argparse
import gc
import json
import statistics
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch

BENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCH_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mmq_deployment_cases import DeploymentCase, public_deployment_cases


class TimingSummary(TypedDict):
    samples_ms: list[float]
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    median_tflops: float


@dataclass(frozen=True)
class OperationSpec:
    operation: str
    slug: str
    public_label: str
    baseline_label: str
    projections: int


OPERATIONS = {
    "OrdinaryForward": OperationSpec(
        "OrdinaryForward", "mmq_fwd", "torch_ggml_ops.mmq", "torch.mm", 1
    ),
    "OrdinaryBackward": OperationSpec(
        "OrdinaryBackward",
        "mmq_bwd",
        "torch.autograd.grad(torch_ggml_ops.mmq)",
        "torch.mm",
        1,
    ),
    "GroupedForward": OperationSpec(
        "GroupedForward",
        "grouped_mmq_fwd",
        "torch_ggml_ops.grouped_mmq",
        "AITER gmm",
        1,
    ),
    "GroupedBackward": OperationSpec(
        "GroupedBackward",
        "grouped_mmq_bwd",
        "torch.autograd.grad(torch_ggml_ops.grouped_mmq)",
        "AITER gmm",
        1,
    ),
    "GroupedForwardPair": OperationSpec(
        "GroupedForwardPair",
        "grouped_mmq_pair_fwd",
        "torch_ggml_ops.grouped_mmq_pair",
        "two AITER gmm calls",
        2,
    ),
    "GroupedBackwardPair": OperationSpec(
        "GroupedBackwardPair",
        "grouped_mmq_pair_bwd",
        "torch.autograd.grad(torch_ggml_ops.grouped_mmq_pair)",
        "two AITER gmm calls plus add",
        2,
    ),
    "FixedGroupedForward": OperationSpec(
        "FixedGroupedForward",
        "fixed_grouped_mmq_fwd",
        "torch_ggml_ops.fixed_grouped_mmq",
        "torch.bmm",
        8,
    ),
    "FixedGroupedBackward": OperationSpec(
        "FixedGroupedBackward",
        "fixed_grouped_mmq_bwd",
        "torch.autograd.grad(torch_ggml_ops.fixed_grouped_mmq)",
        "torch.bmm",
        8,
    ),
}


def parse_args(description: str, *, routed: bool, kernels: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-family", choices=("qwen", "deepseek"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="exact identity or unique identity prefix; repeat to select multiple cases",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--launches-per-sample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260824)
    if routed:
        from workload_prior import EXPERT_PRIOR_NAMES

        parser.add_argument("--expert-prior", choices=EXPERT_PRIOR_NAMES, required=True)
    else:
        parser.set_defaults(expert_prior=None)
    if kernels:
        parser.add_argument("--ggtensile-root", type=Path)
        parser.add_argument("--hip-root", type=Path)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"GGUF model not found: {args.model}")
    if args.warmup < 0 or args.repeats <= 0 or args.launches_per_sample <= 0:
        parser.error("warmup must be nonnegative and repeats/launches must be positive")
    if routed and not args.expert_prior.startswith(f"{args.model_family}-"):
        parser.error("--expert-prior must match --model-family")
    return args


def select_cases(
    operation: str, family: str, selectors: list[str]
) -> tuple[DeploymentCase, ...]:
    cases = tuple(
        case
        for case in public_deployment_cases()
        if case.operation == operation and case.tensor_source.model_family == family
    )
    if not selectors:
        return cases
    selected: list[DeploymentCase] = []
    for selector in selectors:
        matches = [case for case in cases if case.identity.startswith(selector)]
        if len(matches) != 1:
            raise ValueError(
                f"case selector {selector!r} matched {len(matches)} {operation} cases"
            )
        if matches[0] not in selected:
            selected.append(matches[0])
    return tuple(selected)


def logical_flops(case: DeploymentCase, spec: OperationSpec) -> int:
    return 2 * spec.projections * case.rows * case.out_features * case.in_features


def _event_time(function: Callable[[], object], launches: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(launches):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / launches


def rotating_timings(
    functions: Mapping[str, Callable[[], object]],
    *,
    warmup: int,
    repeats: int,
    launches_per_sample: int,
    flops: int,
) -> tuple[dict[str, TimingSummary], list[list[str]]]:
    names = list(functions)
    if len(names) != 2:
        raise ValueError("benchmark comparisons require exactly two implementations")
    for repeat in range(warmup):
        order = names[repeat % 2 :] + names[: repeat % 2]
        for name in order:
            for _ in range(launches_per_sample):
                functions[name]()
    torch.cuda.synchronize()
    samples = {name: [] for name in names}
    orders = []
    for repeat in range(repeats):
        order = names[repeat % 2 :] + names[: repeat % 2]
        orders.append(order)
        for name in order:
            samples[name].append(_event_time(functions[name], launches_per_sample))
    summaries: dict[str, TimingSummary] = {}
    for name, values in samples.items():
        median = statistics.median(values)
        summaries[name] = {
            "samples_ms": values,
            "median_ms": median,
            "mean_ms": statistics.fmean(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "stdev_ms": statistics.pstdev(values),
            "median_tflops": flops / (median * 1.0e9),
        }
    return summaries, orders


def device_info() -> dict[str, object]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "name": properties.name,
        "architecture": getattr(properties, "gcnArchName", None),
        "torch": torch.__version__,
        "hip": torch.version.hip,
    }


def write_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def release_cuda() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def print_result(
    case: DeploymentCase, timing: Mapping[str, TimingSummary], speedup: float
) -> None:
    names = list(timing)
    print(
        f"{case.quant_type:<8} M={case.rows:>6} N={case.out_features:>6} K={case.in_features:>5} "
        f"{names[0]}={timing[names[0]]['median_ms']:.3f} ms "
        f"{names[1]}={timing[names[1]]['median_ms']:.3f} ms speedup={speedup:.3f}x",
        flush=True,
    )
