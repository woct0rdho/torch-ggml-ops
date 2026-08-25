"""Strict validation for paired grouped backward kernels."""

from .grouped_mmq_bwd_pair_model import GroupedBackwardPairProblem
from .grouped_mmq_bwd_pair_spec import (
    GroupedBackwardPairKernelSpec,
    validate_grouped_backward_pair_capability,
    validate_grouped_backward_pair_problem,
)


def validate_grouped_backward_pair_solution(
    problem: GroupedBackwardPairProblem,
    spec: GroupedBackwardPairKernelSpec,
) -> None:
    validate_grouped_backward_pair_problem(problem)
    validate_grouped_backward_pair_capability(problem, spec)
