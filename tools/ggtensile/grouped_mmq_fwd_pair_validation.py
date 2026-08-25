"""Strict validation for research-only paired grouped forward kernels."""

from .grouped_mmq_fwd_pair_model import (
    GroupedForwardPairProblem,
)
from .grouped_mmq_fwd_pair_spec import (
    GroupedForwardPairKernelSpec,
    validate_grouped_forward_pair_capability,
    validate_grouped_forward_pair_problem,
)


def validate_grouped_forward_pair_solution(
    problem: GroupedForwardPairProblem,
    spec: GroupedForwardPairKernelSpec,
) -> None:
    validate_grouped_forward_pair_problem(problem)
    validate_grouped_forward_pair_capability(problem, spec)
