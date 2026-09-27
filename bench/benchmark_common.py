"""Shared CLI, timing, and reporting helpers for kernel benchmarks."""

import argparse
import gc
import json
import math
import statistics
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="minimum warmup iterations per implementation before sampling",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=0.5,
        help="warmup runs until this much device time has elapsed as well",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="samples per implementation per block; must be even",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=2,
        help="repeat the sample loop, flipping the starting order per block",
    )
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
    if args.warmup < 0 or args.warmup_seconds < 0 or args.launches_per_sample <= 0:
        parser.error(
            "warmup and warmup-seconds must be nonnegative, and launches must be positive"
        )
    if args.repeats < 2 or args.repeats % 2:
        parser.error("--repeats must be an even count of at least 2")
    if args.blocks <= 0:
        parser.error("--blocks must be positive")
    args.route_vectors = args.repeats if spec.routed else 1
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


def _alternating_order(names: list[str], index: int) -> list[str]:
    return names[index % 2 :] + names[: index % 2]


def _event_time(function: Callable[[], object], launches: int) -> float:
    """Return the mean device time of ``launches`` calls of one implementation.

    One unmeasured launch runs first, so the device is still busy while the
    measured window is enqueued. Without it the host-side cost of dispatching the
    first measured launch appears as device idle time inside the window, which
    penalizes any implementation whose launcher is slower than the kernel.
    """

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    function()
    start.record()
    for _ in range(launches):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / launches


def _warm_up(
    functions: Mapping[str, Callable[[], object]],
    names: list[str],
    iterations_min: int,
    launches: int,
    seconds_min: float,
) -> tuple[int, float]:
    """Warm up until ``iterations_min`` iterations and ``seconds_min`` device time.

    The budget is device time, not host time: a fast launcher would otherwise
    submit many seconds of queued work while the elapsed host clock is still
    inside the budget.
    """

    def run_batch(index: int) -> None:
        for name in _alternating_order(names, index):
            for _ in range(launches):
                functions[name]()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iterations = 0
    total = 0.0
    rate = 0.0
    count = max(iterations_min, 1)
    for _ in range(4):
        start.record()
        for index in range(iterations, iterations + count):
            run_batch(index)
        iterations += count
        end.record()
        end.synchronize()
        seconds = float(start.elapsed_time(end)) / 1.0e3
        total += seconds
        rate = seconds / count
        if total >= seconds_min or rate <= 0:
            break
        count = int(
            min(
                _WARMUP_MAX_ITERATIONS,
                max(1, math.ceil((seconds_min - total) / rate)),
            )
        )
    return iterations, total


_HOST_PROBE_SECONDS = 0.1
_HOST_PROBE_CALLS_MIN = 8
_HOST_PROBE_CALLS_MAX = 128
_WARMUP_MAX_ITERATIONS = 4096
_POSITION_GAP_WARN_PCT = 2.0
_HOST_BOUND_WARN_RATIO = 2.0
_BLOCK_SPREAD_WARN_PCT = 3.0


def _host_launch_us(function: Callable[[], object], device_ms: float) -> float:
    """Return the host-side cost of one launch closure call.

    Submitted work keeps the device busy, so this is the enqueue rate rather than
    device execution time. The call count is capped by ``_HOST_PROBE_SECONDS`` of
    device work so the probe stays cheap on large kernels.
    """

    device_s = max(device_ms, 1.0e-6) / 1.0e3
    calls = int(
        min(
            _HOST_PROBE_CALLS_MAX,
            max(_HOST_PROBE_CALLS_MIN, _HOST_PROBE_SECONDS / device_s),
        )
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(calls):
        function()
    elapsed = time.perf_counter() - started
    torch.cuda.synchronize()
    return elapsed / calls * 1.0e6


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
    warmup_seconds: float,
    repeats: int,
    launches: int,
    flops: int,
    blocks: int = 1,
    select_sample: Callable[[int], None] | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Time two implementations on paired samples.

    Each block runs the same sample indices with the starting order flipped, so
    every implementation occupies the first and second alternation position
    equally often. Samples stay balanced only when ``repeats`` is even.
    """

    names = list(functions)
    if len(names) != 2:
        raise ValueError("a benchmark must contain exactly two implementations")
    if repeats < 2 or repeats % 2:
        raise ValueError("measure requires an even sample count for position balance")
    if blocks < 1:
        raise ValueError("measure requires at least one block")

    if select_sample is not None:
        select_sample(0)
    warmup_iterations, warmup_elapsed = _warm_up(
        functions, names, warmup, launches, warmup_seconds
    )

    samples: dict[str, list[float]] = {name: [] for name in names}
    positions: dict[str, dict[str, list[float]]] = {
        name: {"first": [], "second": []} for name in names
    }
    sample_orders: list[list[str]] = []
    block_medians: list[dict[str, object]] = []
    for block in range(blocks):
        block_samples: dict[str, list[float]] = {name: [] for name in names}
        for index in range(repeats):
            if select_sample is not None:
                select_sample(index)
                for iteration in range(warmup):
                    for name in _alternating_order(names, index + iteration):
                        for _ in range(launches):
                            functions[name]()
            current_order = _alternating_order(names, index + block)
            sample_orders.append(current_order)
            for position, name in enumerate(current_order):
                value = _event_time(functions[name], launches)
                samples[name].append(value)
                block_samples[name].append(value)
                positions[name]["first" if position == 0 else "second"].append(value)
        block_medians.append(
            {
                "block": block + 1,
                "starting_implementation": _alternating_order(names, block)[0],
                "median_ms": {
                    name: statistics.median(block_samples[name]) for name in names
                },
            }
        )

    summaries = {name: _summary(values, flops) for name, values in samples.items()}
    host_launch_us = {
        name: _host_launch_us(
            functions[name], cast(float, summaries[name]["median_ms"])
        )
        for name in names
    }
    device_over_host = {
        name: cast(float, summaries[name]["median_ms"]) * 1.0e3 / host_launch_us[name]
        for name in names
    }
    position_medians = {
        name: {
            position: statistics.median(values) if values else None
            for position, values in positions[name].items()
        }
        for name in names
    }
    position_gap = {
        name: (
            None
            if position_medians[name]["first"] is None
            or position_medians[name]["second"] is None
            else (
                cast(float, position_medians[name]["second"])
                / cast(float, position_medians[name]["first"])
                - 1.0
            )
            * 100.0
        )
        for name in names
    }
    return (
        summaries,
        {
            "sample_count": repeats * blocks,
            "samples_per_block": repeats,
            "blocks": blocks,
            "sample_unit": "route_vector" if select_sample else "launch",
            "warmup": warmup,
            "warmup_iterations": warmup_iterations,
            "warmup_seconds": warmup_elapsed,
            "launches_per_sample": launches,
            "pipelined_first_launch": True,
            "sample_order": sample_orders,
            "host_launch_us": host_launch_us,
            "device_over_host_ratio": device_over_host,
            "host_bound": {
                name: device_over_host[name] < _HOST_BOUND_WARN_RATIO for name in names
            },
            "position_counts": {
                name: {
                    position: len(values)
                    for position, values in positions[name].items()
                }
                for name in names
            },
            "position_medians_ms": position_medians,
            "position_gap_pct": position_gap,
            "block_medians": block_medians,
        },
    )


def stability_warnings(
    names: Iterable[str],
    protocol: Mapping[str, object],
    block_spread_pct: float | None,
) -> list[str]:
    """Return warnings for position-imbalanced or host-bound measurements."""

    warnings = []
    gaps = cast(Mapping[str, float | None], protocol["position_gap_pct"])
    for name in names:
        gap = gaps[name]
        if gap is not None and abs(gap) > _POSITION_GAP_WARN_PCT:
            warnings.append(f"{name} alternation-position gap {gap:+.1f}%")
    bound = cast(Mapping[str, bool], protocol["host_bound"])
    ratios = cast(Mapping[str, float], protocol["device_over_host_ratio"])
    for name in names:
        if bound[name]:
            warnings.append(f"{name} host-bound at {ratios[name]:.2f}x device/host")
    if block_spread_pct is not None and block_spread_pct > _BLOCK_SPREAD_WARN_PCT:
        warnings.append(f"block ratio spread {block_spread_pct:.1f}%")
    return warnings


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
