"""Route tensors backed by the canonical fitted benchmark workload laws."""

from dataclasses import dataclass

import torch

if __package__:
    from .workload_prior import ExpertPrior, ExpertProfile, profile_for_routed_rows
else:
    from workload_prior import ExpertPrior, ExpertProfile, profile_for_routed_rows


@dataclass(frozen=True)
class RouteDistribution:
    expert_indices_cpu: tuple[int, ...]
    group_sizes_cpu: tuple[int, ...]
    profile: ExpertProfile


def fitted_prior_distribution_for_rows(
    prior: str | ExpertPrior, aggregate_rows: int
) -> RouteDistribution:
    return _distribution_from_profile(profile_for_routed_rows(prior, aggregate_rows))


def _distribution_from_profile(profile: ExpertProfile) -> RouteDistribution:
    expert_indices = tuple(
        expert for expert, rows in enumerate(profile.rows_per_expert) if rows
    )
    group_sizes = tuple(profile.rows_per_expert[expert] for expert in expert_indices)
    return RouteDistribution(expert_indices, group_sizes, profile)


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
