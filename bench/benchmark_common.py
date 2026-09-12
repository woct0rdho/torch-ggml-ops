"""Shared CLI, timing, and reporting helpers for kernel benchmarks."""

import argparse
import gc
import json
import math
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from bench.workload_prior import EXPERT_PRIOR_NAMES
from tools.mmq_deployment_cases import DeploymentCase, public_deployment_cases


@dataclass(frozen=True)
class OperationSpec:
    operation: str
    projections: int
    routed: bool
    baseline: str


OPERATIONS = {
    "OrdinaryForward": OperationSpec("OrdinaryForward", 1, False, "torch.mm"),
    "OrdinaryBackward": OperationSpec("OrdinaryBackward", 1, False, "torch.mm"),
    "GroupedForward": OperationSpec("GroupedForward", 1, True, "AITER gmm"),
    "GroupedBackward": OperationSpec("GroupedBackward", 1, True, "AITER gmm"),
    "GroupedForwardPair": OperationSpec(
        "GroupedForwardPair", 2, True, "two AITER gmm calls"
    ),
    "GroupedBackwardPair": OperationSpec(
        "GroupedBackwardPair", 2, True, "two AITER gmm calls plus add"
    ),
    "FixedGroupedForward": OperationSpec("FixedGroupedForward", 8, False, "torch.bmm"),
    "FixedGroupedBackward": OperationSpec(
        "FixedGroupedBackward", 8, False, "torch.bmm"
    ),
}
IMPLEMENTATIONS = ("baseline", "hip", "ggtensile")


@dataclass(frozen=True)
class Implementation:
    name: str
    launch: Callable[[], object]
    metadata: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark two prepared MMQ implementations."
    )
    parser.add_argument("--operation", choices=tuple(OPERATIONS), required=True)
    parser.add_argument(
        "--implementation",
        dest="implementations",
        action="append",
        choices=IMPLEMENTATIONS,
        required=True,
        help="repeat twice to choose the two implementations to benchmark",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-family", choices=("qwen", "deepseek"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--expert-prior", choices=EXPERT_PRIOR_NAMES)
    parser.add_argument("--ggtensile-root", type=Path)
    parser.add_argument("--hip-root", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--route-vectors", type=int)
    parser.add_argument("--launches-per-sample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    spec = OPERATIONS[args.operation]
    if len(args.implementations) != 2 or len(set(args.implementations)) != 2:
        parser.error(
            "--implementation must be given exactly twice with distinct values"
        )
    if not args.model.is_file():
        parser.error(f"GGUF model not found: {args.model}")
    if spec.routed and args.expert_prior is None:
        parser.error("--expert-prior is required for grouped operations")
    if spec.routed and not args.expert_prior.startswith(f"{args.model_family}-"):
        parser.error("--expert-prior must match --model-family")
    if args.warmup < 0 or args.repeats <= 0 or args.launches_per_sample <= 0:
        parser.error(
            "warmup must be nonnegative, and repeats and launches must be positive"
        )
    args.route_vectors = (
        args.repeats if args.route_vectors is None else args.route_vectors
    )
    if args.route_vectors <= 0 or (spec.routed and args.route_vectors < args.repeats):
        parser.error(
            "--route-vectors must be positive and at least --repeats for grouped operations"
        )
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


def _summary(samples: list[float], flops: int) -> dict[str, object]:
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("timing samples must be finite and positive")
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.pstdev(samples),
        "median_tflops": flops / (statistics.median(samples) * 1.0e9),
    }


def measure(
    functions: Mapping[str, Callable[[], object]],
    *,
    warmup: int,
    repeats: int,
    launches: int,
    flops: int,
    select_sample: Callable[[int], None] | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    names = list(functions)
    if len(names) != 2:
        raise ValueError("a benchmark must contain exactly two implementations")

    def order(index: int) -> list[str]:
        return names[index % 2 :] + names[: index % 2]

    if select_sample is None:
        for repeat in range(warmup):
            for name in order(repeat):
                for _ in range(launches):
                    functions[name]()
    else:
        for index in range(repeats):
            select_sample(index)
            for repeat in range(warmup):
                for name in order(index + repeat):
                    for _ in range(launches):
                        functions[name]()
    torch.cuda.synchronize()

    samples = {name: [] for name in names}
    sample_orders = []
    for index in range(repeats):
        if select_sample is not None:
            select_sample(index)
        current_order = order(index)
        sample_orders.append(current_order)
        for name in current_order:
            samples[name].append(_event_time(functions[name], launches))
    return (
        {name: _summary(values, flops) for name, values in samples.items()},
        {
            "sample_count": repeats,
            "sample_unit": "route_vector" if select_sample else "launch",
            "warmup": warmup,
            "launches_per_sample": launches,
            "sample_order": sample_orders,
        },
    )


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
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def release_cuda() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
