"""Route tensors backed by the canonical fitted benchmark workload laws."""

import statistics
from dataclasses import dataclass
from typing import TypedDict

import torch

if __package__:
    from .workload_prior import (
        ExpertPrior,
        ExpertProfile,
        profile_for_routed_rows,
        sample_expert_profile,
    )
else:
    from workload_prior import (
        ExpertPrior,
        ExpertProfile,
        profile_for_routed_rows,
        sample_expert_profile,
    )


@dataclass(frozen=True)
class RouteDistribution:
    name: str
    expert_indices_cpu: tuple[int, ...]
    group_sizes_cpu: tuple[int, ...]
    profile: ExpertProfile | None = None

    @property
    def rows(self) -> int:
        return sum(self.group_sizes_cpu)

    @property
    def rows_per_expert(self) -> tuple[int, ...]:
        rows = [0] * 256
        for expert, size in zip(
            self.expert_indices_cpu, self.group_sizes_cpu, strict=True
        ):
            rows[expert] = size
        return tuple(rows)


def fitted_prior_distribution(
    prior: str | ExpertPrior, tokens: int
) -> RouteDistribution:
    return _distribution_from_profile(sample_expert_profile(prior, tokens))


def fitted_prior_distribution_for_rows(
    prior: str | ExpertPrior, aggregate_rows: int
) -> RouteDistribution:
    return _distribution_from_profile(profile_for_routed_rows(prior, aggregate_rows))


def _distribution_from_profile(profile: ExpertProfile) -> RouteDistribution:
    expert_indices = tuple(
        expert for expert, rows in enumerate(profile.rows_per_expert) if rows
    )
    group_sizes = tuple(profile.rows_per_expert[expert] for expert in expert_indices)
    return RouteDistribution(
        profile.profile_id,
        expert_indices,
        group_sizes,
        profile,
    )


def fixed_group_distribution(rows: int, groups: int) -> RouteDistribution:
    return RouteDistribution("fixed", tuple(range(groups)), (rows,) * groups)


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
        f"{distribution.name}_correctness",
        tuple(experts),
        tuple(sizes),
        distribution.profile,
    )
