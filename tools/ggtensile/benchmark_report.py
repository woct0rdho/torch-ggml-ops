"""Shared machine-readable timing and correctness records for GGTensile tools."""

import statistics
from collections.abc import Callable, Mapping
from typing import TypedDict

import torch


class ErrorMetrics(TypedDict):
    different_bf16_elements: int
    elements: int
    finite: bool
    max_absolute_error: float | None
    error_rms: float | None
    reference_rms: float
    normalized_rmse: float | None


class TimingSummary(TypedDict):
    samples_ms: list[float]
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    median_tflops: float


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> ErrorMetrics:
    difference = actual.float() - expected.float()
    finite = bool(torch.isfinite(difference).all())
    reference_rms = float(expected.float().square().mean().sqrt())
    error_rms = float(difference.square().mean().sqrt()) if finite else None
    return {
        "different_bf16_elements": int(torch.count_nonzero(actual != expected)),
        "elements": actual.numel(),
        "finite": finite,
        "max_absolute_error": float(difference.abs().max()) if finite else None,
        "error_rms": error_rms,
        "reference_rms": reference_rms,
        "normalized_rmse": (
            error_rms / reference_rms
            if error_rms is not None and reference_rms
            else None
        ),
    }


def event_time(function: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def timing_summary(samples_ms: list[float], logical_flops: int) -> TimingSummary:
    median_ms = statistics.median(samples_ms)
    return {
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "median_tflops": logical_flops / (median_ms * 1.0e9),
    }


def rotating_timings(
    functions: Mapping[str, Callable[[], None]],
    *,
    warmup: int,
    repeats: int,
    logical_flops: int,
) -> dict[str, TimingSummary]:
    for _ in range(warmup):
        for function in functions.values():
            function()
    torch.cuda.synchronize()
    names = list(functions)
    samples = {name: [] for name in names}
    for repeat in range(repeats):
        offset = repeat % len(names)
        for name in names[offset:] + names[:offset]:
            samples[name].append(event_time(functions[name]))
    return {
        name: timing_summary(values, logical_flops) for name, values in samples.items()
    }
