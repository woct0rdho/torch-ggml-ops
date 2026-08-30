"""Shared protocol, CLI, and reporting for MMQ benchmarks."""

import argparse
import gc
import json
import math
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
    mean_log_ms: float
    median_log_ms: float
    stddev_log_ms: float
    robust_sigma_log: float
    stderr_median_log: float
    mad_log: float
    iqr_log: float
    p10_ms: float
    p90_ms: float
    outlier_count: int


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
    parser.add_argument(
        "--repeats",
        type=int,
        default=16,
        help="initial timing samples; routed benchmarks use one sample per route vector",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="maximum route vectors or ordinary timing samples",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=16,
        help="new route vectors or ordinary samples added per adaptive round",
    )
    parser.add_argument("--adaptive-confidence", type=float, default=0.90)
    parser.add_argument("--adaptive-epsilon-pct", type=float, default=2.0)
    parser.add_argument("--adaptive-stable-rounds", type=int, default=2)
    parser.add_argument("--adaptive-noise-floor-pct", type=float, default=0.5)
    parser.add_argument(
        "--launches-per-sample",
        type=int,
        default=1,
        help="launches averaged inside each timing sample",
    )
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
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.launches_per_sample <= 0:
        parser.error("--launches-per-sample must be positive")
    if args.max_samples < args.repeats:
        parser.error("--max-samples must be at least --repeats")
    if args.sample_step <= 0:
        parser.error("--sample-step must be positive")
    if not math.isfinite(args.adaptive_confidence) or not (
        0.0 < args.adaptive_confidence < 1.0
    ):
        parser.error("--adaptive-confidence must be between 0 and 1")
    if not math.isfinite(args.adaptive_epsilon_pct) or args.adaptive_epsilon_pct <= 0.0:
        parser.error("--adaptive-epsilon-pct must be positive")
    if args.adaptive_stable_rounds <= 0:
        parser.error("--adaptive-stable-rounds must be positive")
    if (
        not math.isfinite(args.adaptive_noise_floor_pct)
        or args.adaptive_noise_floor_pct < 0.0
    ):
        parser.error("--adaptive-noise-floor-pct must be nonnegative")
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


MEDIAN_SE_FACTOR = 1.2533141373155001


@dataclass(frozen=True)
class AdaptiveTimingPolicy:
    """Policy for expanding route-vector or ordinary timing samples."""

    min_samples: int
    max_samples: int
    sample_step: int
    confidence: float
    epsilon_pct: float
    stable_rounds: int
    noise_floor_pct: float

    def __post_init__(self) -> None:
        if self.min_samples <= 0:
            raise ValueError("minimum timing samples must be positive")
        if self.max_samples < self.min_samples:
            raise ValueError("maximum timing samples must be at least the minimum")
        if self.sample_step <= 0:
            raise ValueError("timing sample step must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("timing confidence must be in (0, 1)")
        if self.epsilon_pct <= 0.0:
            raise ValueError("timing epsilon must be positive")
        if self.stable_rounds <= 0:
            raise ValueError("stable timing rounds must be positive")
        if self.noise_floor_pct < 0.0:
            raise ValueError("timing noise floor must be nonnegative")

    @property
    def epsilon_log(self) -> float:
        return math.log1p(self.epsilon_pct / 100.0)

    @property
    def noise_floor_log(self) -> float:
        return math.log1p(self.noise_floor_pct / 100.0)

    @property
    def z_value(self) -> float:
        return statistics.NormalDist().inv_cdf(0.5 + self.confidence / 2.0)

    def to_mapping(self) -> dict[str, object]:
        return {
            "min_samples": self.min_samples,
            "max_samples": self.max_samples,
            "sample_step": self.sample_step,
            "confidence": self.confidence,
            "epsilon_pct": self.epsilon_pct,
            "stable_rounds": self.stable_rounds,
            "noise_floor_pct": self.noise_floor_pct,
        }


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of empty timing values")
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] * (upper - position) + sorted_values[upper] * (
        position - lower
    )


def _timing_summary(
    samples_ms: list[float], flops: int, *, noise_floor_log: float = 0.0
) -> TimingSummary:
    values = [float(value) for value in samples_ms]
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("timing samples must be finite and positive")
    sorted_times = sorted(values)
    logs = [math.log(value) for value in values]
    sorted_logs = sorted(logs)
    median_log = statistics.median(logs)
    mad_log = statistics.median(abs(value - median_log) for value in logs)
    q25 = _quantile(sorted_logs, 0.25)
    q75 = _quantile(sorted_logs, 0.75)
    iqr_log = q75 - q25
    stddev_log = statistics.stdev(logs) if len(logs) >= 2 else 0.0
    robust_sigma = max(
        stddev_log,
        1.4826 * mad_log,
        iqr_log / 1.349,
        noise_floor_log,
    )
    stderr = MEDIAN_SE_FACTOR * robust_sigma / math.sqrt(len(values))
    high_fence = q75 + 1.5 * iqr_log if iqr_log > 0.0 else float("inf")
    return {
        "samples_ms": values,
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.pstdev(values),
        "median_tflops": flops / (statistics.median(values) * 1.0e9),
        "mean_log_ms": statistics.fmean(logs),
        "median_log_ms": median_log,
        "stddev_log_ms": stddev_log,
        "robust_sigma_log": robust_sigma,
        "stderr_median_log": stderr,
        "mad_log": mad_log,
        "iqr_log": iqr_log,
        "p10_ms": _quantile(sorted_times, 0.10),
        "p90_ms": _quantile(sorted_times, 0.90),
        "outlier_count": sum(value > high_fence for value in logs),
    }


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
    summaries = {
        name: _timing_summary(values, flops) for name, values in samples.items()
    }
    return summaries, orders


def adaptive_timings(
    functions: Mapping[str, Callable[[], object]],
    *,
    policy: AdaptiveTimingPolicy,
    warmup: int,
    launches_per_sample: int,
    flops: int,
    select_sample: Callable[[int], None] | None = None,
    sample_capacity: int | None = None,
) -> tuple[dict[str, TimingSummary], dict[str, object]]:
    """Measure matched implementation samples until the estimate stabilizes.

    With ``select_sample`` set, each sample index names a prebuilt route vector.
    Without it, each index is an ordinary repeated launch measurement.
    """
    names = list(functions)
    if len(names) != 2:
        raise ValueError("benchmark comparisons require exactly two implementations")
    if warmup < 0 or launches_per_sample <= 0:
        raise ValueError("warmup must be nonnegative and launches must be positive")
    capacity = policy.max_samples
    if sample_capacity is not None:
        if sample_capacity < policy.min_samples:
            raise ValueError("sample capacity is below the adaptive minimum")
        capacity = min(capacity, sample_capacity)

    def order_for(index: int) -> list[str]:
        return names[index % 2 :] + names[: index % 2]

    samples = {name: [] for name in names}
    sample_records: list[dict[str, object]] = []
    rounds: list[dict[str, object]] = []
    orders: list[list[str]] = []
    measured = 0
    warmed = 0
    target = min(policy.min_samples, capacity)
    previous_speedup: float | None = None
    stable_streak = 0
    stop_reason = "max_samples"

    while True:
        if select_sample is not None:
            for index in range(warmed, target):
                select_sample(index)
                order = order_for(index)
                for _ in range(warmup):
                    for name in order:
                        for _ in range(launches_per_sample):
                            functions[name]()
            warmed = target
        elif warmed == 0:
            for repeat in range(warmup):
                order = order_for(repeat)
                for name in order:
                    for _ in range(launches_per_sample):
                        functions[name]()
            warmed = target
        torch.cuda.synchronize()

        for index in range(measured, target):
            if select_sample is not None:
                select_sample(index)
            order = order_for(index)
            values: dict[str, float] = {}
            for name in order:
                values[name] = _event_time(functions[name], launches_per_sample)
                samples[name].append(values[name])
            orders.append(order)
            sample_records.append({"sample_index": index, **values, "order": order})
        measured = target

        summaries = {
            name: _timing_summary(
                values,
                flops,
                noise_floor_log=policy.noise_floor_log,
            )
            for name, values in samples.items()
        }
        log_gap = float(summaries[names[1]]["median_log_ms"]) - float(
            summaries[names[0]]["median_log_ms"]
        )
        gap_se = math.sqrt(
            float(summaries[names[0]]["stderr_median_log"]) ** 2
            + float(summaries[names[1]]["stderr_median_log"]) ** 2
        )
        ci_half_log = policy.z_value * gap_se
        ci_low_log = log_gap - ci_half_log
        ci_high_log = log_gap + ci_half_log
        speedup = math.exp(log_gap)
        half_width_pct = (math.exp(ci_half_log) - 1.0) * 100.0
        change_pct = (
            None
            if previous_speedup is None
            else abs(speedup / previous_speedup - 1.0) * 100.0
        )
        change_log = (
            None
            if previous_speedup is None
            else abs(math.log(speedup) - math.log(previous_speedup))
        )
        precise = ci_half_log <= policy.epsilon_log
        stable = precise and change_log is not None and change_log <= policy.epsilon_log
        stable_streak = stable_streak + 1 if stable else 0
        rounds.append(
            {
                "sample_count": measured,
                "speedup": speedup,
                "log_speedup": log_gap,
                "ci_low_speedup": math.exp(ci_low_log),
                "ci_high_speedup": math.exp(ci_high_log),
                "ci_half_width_pct": half_width_pct,
                "estimate_change_pct": change_pct,
                "precise": precise,
                "stable": stable,
                "stable_streak": stable_streak,
            }
        )
        if stable_streak >= policy.stable_rounds:
            stop_reason = "stable_confidence_and_prefix"
            break
        if target >= capacity:
            stop_reason = "max_samples"
            break
        previous_speedup = speedup
        target = min(capacity, target + policy.sample_step)

    adaptive = {
        "sample_unit": "route_vector"
        if select_sample is not None
        else "ordinary_measurement",
        "sample_count": measured,
        "sample_capacity": capacity,
        "policy": policy.to_mapping(),
        "stop_reason": stop_reason,
        "stable": stop_reason == "stable_confidence_and_prefix",
        "sample_order": orders,
        "samples": sample_records,
        "rounds": rounds,
    }
    return summaries, adaptive


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
