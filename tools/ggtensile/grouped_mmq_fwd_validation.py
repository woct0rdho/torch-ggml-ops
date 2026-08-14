"""Validation for isolated grouped MMQ forward solution keys."""

from .grouped_mmq_fwd_model import (
    GroupedForwardProblem,
    GroupedForwardSolution,
    GroupedForwardSolutionKey,
)
from .validation import RejectReason


def validate_grouped_forward_solution(
    key: GroupedForwardSolutionKey,
) -> tuple[RejectReason, ...]:
    reasons: list[RejectReason] = []
    expected_problem = GroupedForwardProblem.q4_k(key.problem.aggregate_rows)
    if key.problem != expected_problem:
        reasons.append(
            RejectReason(
                "grouped_forward.problem.unsupported",
                "grouped Q4_K forward requires N=2048, K=512, 256 physical experts, and at most 256 route entries",
                ("Problem",),
                "GroupedForwardProblem",
            )
        )
    if key.problem.aggregate_rows <= 0 or key.problem.aggregate_rows > 0xFFFFFFFF:
        reasons.append(
            RejectReason(
                "grouped_forward.aggregate_rows.range",
                "aggregate rows must fit in a positive u32",
                ("AggregateRows",),
                "GroupedForwardProblem",
            )
        )
    if key.solution != GroupedForwardSolution.q4_k_serial_direct():
        reasons.append(
            RejectReason(
                "grouped_forward.solution.unimplemented",
                "grouped Q4_K forward currently implements the serial direct-global control",
                ("Solution",),
                "GroupedForwardSolution",
            )
        )
    return tuple(reasons)
