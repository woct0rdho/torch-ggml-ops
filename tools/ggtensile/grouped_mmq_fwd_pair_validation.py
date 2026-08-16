"""Strict validation for the research-only paired IQ2_S forward identity."""

from .grouped_mmq_fwd_pair_model import (
    GroupedForwardPairProblem,
    GroupedForwardPairSolution,
    GroupedForwardPairSolutionKey,
)
from .validation import RejectReason


def validate_grouped_forward_pair_solution(
    key: GroupedForwardPairSolutionKey,
) -> tuple[RejectReason, ...]:
    reasons: list[RejectReason] = []
    expected = GroupedForwardPairProblem.iq2_s(key.problem.aggregate_rows)
    if key.problem != expected:
        reasons.append(
            RejectReason(
                "grouped_forward_pair.problem.unsupported",
                "paired grouped IQ2_S requires exact N512/K2048 geometry, two projections, 256 experts, and at most 256 routes",
                ("Problem",),
                "GroupedForwardPairProblem",
            )
        )
    if key.problem.aggregate_rows <= 0 or key.problem.aggregate_rows > 0xFFFFFFFF:
        reasons.append(
            RejectReason(
                "grouped_forward_pair.aggregate_rows.range",
                "paired aggregate rows must fit in a positive u32",
                ("AggregateRows",),
                "GroupedForwardPairProblem",
            )
        )
    supported_solutions = (
        GroupedForwardPairSolution.iq2_s_k128_interleaved(),
        GroupedForwardPairSolution.iq2_s_k128_interleaved_row_tasks(),
    )
    if key.solution not in supported_solutions:
        reasons.append(
            RejectReason(
                "grouped_forward_pair.solution.unimplemented",
                "paired grouped forward implements IQ2_S K128 interleaving with serial-route or device-row-task ownership",
                ("Solution",),
                "GroupedForwardPairSolution",
            )
        )
    return tuple(reasons)
