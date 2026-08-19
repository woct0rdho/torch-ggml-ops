"""Deterministic route controls shared by grouped benchmark entry points."""

import statistics
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch

DISTRIBUTION_NAMES = ("uniform", "skewed", "sparse", "boundary")


@dataclass(frozen=True)
class RouteDistribution:
    name: str
    expert_indices_cpu: tuple[int, ...]
    group_sizes_cpu: tuple[int, ...]

    @property
    def rows(self) -> int:
        return sum(self.group_sizes_cpu)


def fixed_group_distribution(rows: int, groups: int) -> RouteDistribution:
    return RouteDistribution("fixed", tuple(range(groups)), (rows,) * groups)


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


class GroupSummary(TypedDict):
    active_experts: int
    min_rows: int
    max_rows: int
    mean_rows: float
    median_rows: float
    stdev_rows: float
    non_multiple_16_groups: int
    non_multiple_64_groups: int
    non_multiple_128_groups: int


def distribution_summary(distribution: RouteDistribution) -> GroupSummary:
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


def make_route_tensors(
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
        distribution.expert_indices_cpu,
        distribution.group_sizes_cpu,
        strict=True,
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
