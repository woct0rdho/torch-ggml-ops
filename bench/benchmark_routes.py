"""Route tensors backed by the canonical fitted benchmark workload laws.

Every vector is materialized in the production caller's contract: one group per
physical expert, zero rows for inactive experts, and cumulative int32 offsets
over every group.
"""

from dataclasses import dataclass

import torch

from bench.workload_prior import (
    ExpertPrior,
    ExpertProfile,
    profiles_for_routed_rows,
)


@dataclass(frozen=True)
class RouteDistribution:
    """One complete expert-route vector in the production call contract.

    `group_sizes_cpu` covers every physical expert. `active_expert_indices_cpu`
    is the active support, which is report provenance rather than dispatch input.
    """

    active_expert_indices_cpu: tuple[int, ...]
    group_sizes_cpu: tuple[int, ...]
    profile: ExpertProfile


def fitted_prior_distribution_for_rows(
    prior: str | ExpertPrior, aggregate_rows: int
) -> RouteDistribution:
    return _distribution_from_profile(
        profiles_for_routed_rows(prior, aggregate_rows, 1)[0]
    )


def fitted_prior_distributions_for_rows(
    prior: str | ExpertPrior,
    aggregate_rows: int,
    count: int,
    *,
    seed: int | None = None,
) -> tuple[RouteDistribution, ...]:
    return tuple(
        _distribution_from_profile(profile)
        for profile in profiles_for_routed_rows(prior, aggregate_rows, count, seed=seed)
    )


def _distribution_from_profile(profile: ExpertProfile) -> RouteDistribution:
    support = tuple(
        expert for expert, rows in enumerate(profile.rows_per_expert) if rows
    )
    return RouteDistribution(support, tuple(profile.rows_per_expert), profile)


def make_route_tensors(
    distribution: RouteDistribution,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return int64 expert ids, int32 offsets, and int32 group sizes.

    This is the production caller's contract: every physical expert owns a group,
    inactive experts carry zero rows, and the offsets are a prefix sum over all
    groups rather than over the active support.
    """

    group_sizes = torch.tensor(
        distribution.group_sizes_cpu, device="cuda", dtype=torch.int32
    )
    expert_indices = torch.arange(group_sizes.numel(), device="cuda", dtype=torch.int64)
    expert_offsets = group_sizes.cumsum(0, dtype=torch.int32)
    return expert_indices, expert_offsets, group_sizes
