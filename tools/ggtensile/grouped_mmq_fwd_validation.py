"""Validation for isolated grouped MMQ forward problems and specifications."""

from .grouped_mmq_fwd_model import (
    GroupedForwardProblem,
)
from .grouped_mmq_fwd_spec import (
    GroupedForwardKernelSpec,
    validate_grouped_forward_capability,
    validate_grouped_forward_problem,
)


def validate_grouped_forward_solution(
    problem: GroupedForwardProblem,
    spec: GroupedForwardKernelSpec,
) -> None:
    validate_grouped_forward_problem(problem)
    validate_grouped_forward_capability(problem, spec)
