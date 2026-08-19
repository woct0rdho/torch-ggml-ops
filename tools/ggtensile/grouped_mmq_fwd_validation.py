"""Validation for isolated grouped MMQ forward solution keys."""

from .grouped_mmq_fwd_model import (
    GroupedForwardSolutionKey,
)
from .grouped_mmq_fwd_spec import (
    grouped_forward_capability_rejection_reason,
    grouped_forward_problem_rejection_reason,
)
from .validation import RejectReason


def validate_grouped_forward_solution(
    key: GroupedForwardSolutionKey,
) -> tuple[RejectReason, ...]:
    reasons: list[RejectReason] = []
    problem_rejection = grouped_forward_problem_rejection_reason(key.problem)
    aggregate_rows_valid = 0 < key.problem.aggregate_rows <= 0xFFFFFFFF
    if problem_rejection is not None and aggregate_rows_valid:
        reasons.append(
            RejectReason(
                "grouped_forward.problem.unsupported",
                problem_rejection,
                ("Problem",),
                "GroupedForwardProblem",
            )
        )
    if not aggregate_rows_valid:
        reasons.append(
            RejectReason(
                "grouped_forward.aggregate_rows.range",
                "aggregate rows must fit in a positive u32",
                ("AggregateRows",),
                "GroupedForwardProblem",
            )
        )
    rejection = grouped_forward_capability_rejection_reason(
        key.problem,
        key.solution,
    )
    if rejection is not None and rejection != problem_rejection:
        reasons.append(
            RejectReason(
                "grouped_forward.solution.unimplemented",
                rejection,
                ("Solution",),
                "GroupedForwardSolution",
            )
        )
    return tuple(reasons)
