"""Shared production cases, routing distributions, and timing helpers for grouped MMQ benchmarks."""

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from mmq_benchmark_common import cuda_event_times_ms, incremental_peak_bytes

DEFAULT_AITER_HEURISTIC_DIR = Path.home() / "test_no_unsloth"


@dataclass(frozen=True)
class GroupedMMQCase:
    name: str
    kind: str
    tensor_names: tuple[str, ...]
    expected_out_features: int
    expected_in_features: int
    model_layer_count: int
    priority: str
    description: str

    @property
    def projections(self) -> int:
        return len(self.tensor_names)


CASES = (
    GroupedMMQCase(
        "gate_up_q3_k",
        "pair",
        ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"),
        512,
        2048,
        20,
        "primary",
        "paired routed gate/up for layers 0-9 and 30-39",
    ),
    GroupedMMQCase(
        "gate_up_iq2_s",
        "pair",
        ("blk.10.ffn_gate_exps.weight", "blk.10.ffn_up_exps.weight"),
        512,
        2048,
        20,
        "primary",
        "paired routed gate/up for layers 10-29",
    ),
    GroupedMMQCase(
        "down_iq2_s",
        "single",
        ("blk.10.ffn_down_exps.weight",),
        2048,
        512,
        20,
        "primary",
        "routed down projection for layers 10-29",
    ),
    GroupedMMQCase(
        "down_q4_k",
        "single",
        ("blk.2.ffn_down_exps.weight",),
        2048,
        512,
        18,
        "primary",
        "routed down projection for layers 2-9 and 30-39",
    ),
    GroupedMMQCase(
        "down_q5_k",
        "single",
        ("blk.0.ffn_down_exps.weight",),
        2048,
        512,
        2,
        "secondary",
        "routed down projection for layers 0-1",
    ),
)


@dataclass(frozen=True)
class RouteDistribution:
    name: str
    expert_indices_cpu: tuple[int, ...]
    group_sizes_cpu: tuple[int, ...]

    @property
    def rows(self) -> int:
        return sum(self.group_sizes_cpu)


def parse_name_list(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated nonempty list")
    return result


def select_cases(case_names: str, primary_only: bool) -> tuple[GroupedMMQCase, ...]:
    by_name = {case.name: case for case in CASES}
    if case_names:
        names = tuple(name.strip() for name in case_names.split(",") if name.strip())
        unknown = sorted(set(names) - set(by_name))
        if unknown:
            raise ValueError(
                f"unknown cases {unknown}; available cases are {sorted(by_name)}"
            )
        selected = tuple(by_name[name] for name in names)
    else:
        selected = CASES
    if primary_only:
        selected = tuple(case for case in selected if case.priority == "primary")
    if not selected:
        raise ValueError("no grouped benchmark cases selected")
    return selected


def load_gmm_config(heuristic_dir: Path) -> Callable[[int, int], dict[str, int]]:
    sys.path.insert(0, str(heuristic_dir))
    try:
        from fast_moe_lora import _gmm_config  # ty: ignore[unresolved-import]
    finally:
        sys.path.pop(0)
    return _gmm_config


def adjust_positive_sizes(values: list[int], total: int) -> tuple[int, ...]:
    if not values or total < len(values):
        raise ValueError("cannot construct positive grouped sizes")
    values = [max(1, int(value)) for value in values]
    difference = total - sum(values)
    index = 0
    while difference != 0:
        slot = index % len(values)
        if difference > 0:
            values[slot] += 1
            difference -= 1
        elif values[slot] > 1:
            values[slot] -= 1
            difference += 1
        index += 1
    return tuple(values)


def centered_sizes(total: int, groups: int, amplitude: int) -> tuple[int, ...]:
    center = total / groups
    values = [
        round(center + (((index * 37) % 129) - 64) * amplitude / 64)
        for index in range(groups)
    ]
    return adjust_positive_sizes(values, total)


def route_distributions(rows: int, batch: int) -> dict[str, RouteDistribution]:
    all_experts = tuple(range(256))
    uniform = adjust_positive_sizes([rows // 256] * 256, rows)
    amplitude = {1: 22, 4: 64, 16: 256}.get(batch, max(1, rows // 1024))
    skewed = centered_sizes(rows, 256, amplitude)

    sparse_groups = {1: 192, 4: 224, 16: 240}.get(batch, 224)
    sparse_experts = tuple(
        int(value) for value in np.linspace(0, 255, sparse_groups, dtype=np.int64)
    )
    sparse_amplitude = max(1, round((rows / sparse_groups) * 0.25))
    sparse_sizes = centered_sizes(rows, sparse_groups, sparse_amplitude)

    boundary_prefix = (1, 15, 16, 17, 63, 64, 65, 127, 128, 129)
    tail_groups = 256 - len(boundary_prefix)
    tail_total = rows - sum(boundary_prefix)
    if tail_total < tail_groups:
        raise ValueError(f"rows={rows} is too small for boundary distribution")
    boundary_tail = centered_sizes(
        tail_total,
        tail_groups,
        max(1, round((tail_total / tail_groups) * 0.10)),
    )
    boundary_sizes = boundary_prefix + boundary_tail

    return {
        "uniform": RouteDistribution("uniform", all_experts, uniform),
        "skewed": RouteDistribution("skewed", all_experts, skewed),
        "sparse": RouteDistribution("sparse", sparse_experts, sparse_sizes),
        "boundary": RouteDistribution("boundary", all_experts, boundary_sizes),
    }


def distribution_summary(distribution: RouteDistribution) -> dict:
    sizes = distribution.group_sizes_cpu
    return {
        "active_experts": len(sizes),
        "min_rows": min(sizes),
        "max_rows": max(sizes),
        "mean_rows": statistics.mean(sizes),
        "median_rows": statistics.median(sizes),
        "stdev_rows": statistics.pstdev(sizes),
        "non_multiple_16_groups": sum(size % 16 != 0 for size in sizes),
        "non_multiple_64_groups": sum(size % 64 != 0 for size in sizes),
        "non_multiple_128_groups": sum(size % 128 != 0 for size in sizes),
    }


def device_metadata(
    distribution: RouteDistribution,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expert_indices = torch.tensor(
        distribution.expert_indices_cpu, device="cuda", dtype=torch.int64
    )
    group_sizes = torch.tensor(
        distribution.group_sizes_cpu, device="cuda", dtype=torch.int32
    )
    expert_offsets = group_sizes.cumsum(0).to(torch.int32).contiguous()
    return expert_indices, expert_offsets, group_sizes


def truncate_distribution(
    distribution: RouteDistribution, max_rows: int
) -> RouteDistribution:
    experts = []
    sizes = []
    remaining = min(max_rows, distribution.rows)
    for expert, size in zip(
        distribution.expert_indices_cpu, distribution.group_sizes_cpu, strict=True
    ):
        if remaining <= 0:
            break
        take = min(size, remaining)
        experts.append(expert)
        sizes.append(take)
        remaining -= take
    return RouteDistribution(
        f"{distribution.name}_correctness", tuple(experts), tuple(sizes)
    )


def timing_summary(
    times_ms: list[float], rows: int, n: int, k: int, projections: int
) -> dict:
    median_ms = statistics.median(times_ms)
    logical_flops = 2 * projections * rows * n * k
    return {
        "samples_ms": times_ms,
        "median_ms": median_ms,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "logical_tflops": logical_flops / (median_ms * 1.0e9),
    }


def benchmark_function(
    function: Callable[[], object],
    rows: int,
    n: int,
    k: int,
    projections: int,
    warmup: int,
    repeats: int,
) -> dict:
    times = cuda_event_times_ms(function, warmup, repeats)
    allocated, reserved = incremental_peak_bytes(function)
    result = timing_summary(times, rows, n, k, projections)
    result.update(
        {
            "incremental_peak_allocated_bytes": allocated,
            "incremental_peak_reserved_bytes": reserved,
        }
    )
    return result


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = actual.float() - expected.float()
    reference_rms = expected.float().square().mean().sqrt()
    error_rms = difference.square().mean().sqrt()
    return {
        "reference_rms": float(reference_rms),
        "error_rms": float(error_rms),
        "normalized_rmse": float(error_rms / reference_rms),
        "max_absolute_error": float(difference.abs().max()),
        "different_bf16_elements": int(torch.count_nonzero(actual != expected)),
        "elements": actual.numel(),
    }
