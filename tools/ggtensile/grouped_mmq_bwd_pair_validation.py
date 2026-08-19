"""Strict validation for paired grouped backward identities."""

from .grouped_mmq_bwd_pair_model import GroupedBackwardPairSolutionKey
from .grouped_mmq_bwd_pair_spec import (
    grouped_backward_pair_capability_rejection_reason,
    grouped_backward_pair_problem_rejection_reason,
)
from .validation import RejectReason


def validate_grouped_backward_pair_solution(
    key: GroupedBackwardPairSolutionKey,
) -> tuple[RejectReason, ...]:
    reasons: list[RejectReason] = []
    problem_rejection = grouped_backward_pair_problem_rejection_reason(key.problem)
    if problem_rejection is not None:
        reasons.append(
            RejectReason(
                "grouped_backward_pair.problem.unsupported",
                problem_rejection,
                ("Problem",),
                "GroupedBackwardPairProblem",
            )
        )
    capability_rejection = grouped_backward_pair_capability_rejection_reason(
        key.problem, key.solution
    )
    if capability_rejection is not None and capability_rejection != problem_rejection:
        reasons.append(
            RejectReason(
                "grouped_backward_pair.solution.unimplemented",
                capability_rejection,
                ("Solution",),
                "GroupedBackwardPairSolution",
            )
        )
    return tuple(reasons)
