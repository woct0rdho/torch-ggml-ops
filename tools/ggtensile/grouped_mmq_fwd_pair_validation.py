"""Strict validation for research-only paired grouped forward identities."""

from .grouped_mmq_fwd_pair_model import (
    GroupedForwardPairSolutionKey,
)
from .grouped_mmq_fwd_pair_spec import (
    grouped_forward_pair_capability_rejection_reason,
    grouped_forward_pair_problem_rejection_reason,
)
from .validation import RejectReason


def validate_grouped_forward_pair_solution(
    key: GroupedForwardPairSolutionKey,
) -> tuple[RejectReason, ...]:
    reasons: list[RejectReason] = []
    problem_rejection = grouped_forward_pair_problem_rejection_reason(key.problem)
    aggregate_rows_valid = 0 < key.problem.aggregate_rows <= 0xFFFFFFFF
    if problem_rejection is not None and aggregate_rows_valid:
        reasons.append(
            RejectReason(
                "grouped_forward_pair.problem.unsupported",
                problem_rejection,
                ("Problem",),
                "GroupedForwardPairProblem",
            )
        )
    if not aggregate_rows_valid:
        reasons.append(
            RejectReason(
                "grouped_forward_pair.aggregate_rows.range",
                "paired aggregate rows must fit in a positive u32",
                ("AggregateRows",),
                "GroupedForwardPairProblem",
            )
        )
    rejection = grouped_forward_pair_capability_rejection_reason(
        key.problem,
        key.solution,
    )
    if rejection is not None and rejection != problem_rejection:
        reasons.append(
            RejectReason(
                "grouped_forward_pair.solution.unimplemented",
                rejection,
                ("Solution",),
                "GroupedForwardPairSolution",
            )
        )
    return tuple(reasons)
