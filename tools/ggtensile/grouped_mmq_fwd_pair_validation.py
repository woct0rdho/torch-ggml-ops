"""Strict validation for research-only paired grouped forward identities."""

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
    expected = {
        "IQ2_S": GroupedForwardPairProblem.iq2_s,
        "IQ2_XXS": GroupedForwardPairProblem.iq2_xxs,
        "Q3_K": GroupedForwardPairProblem.q3_k,
    }.get(key.problem.quant_data_type)
    if expected is None or key.problem != expected(key.problem.aggregate_rows):
        reasons.append(
            RejectReason(
                "grouped_forward_pair.problem.unsupported",
                "paired grouped forward requires a supported quant type, its exact N/K geometry, two projections, 256 experts, and at most 256 routes",
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
    supported_solutions = {
        "IQ2_S": (
            GroupedForwardPairSolution.iq2_s_k128_interleaved(),
            GroupedForwardPairSolution.iq2_s_k128_interleaved_row_tasks(),
        ),
        "IQ2_XXS": (GroupedForwardPairSolution.iq2_xxs_k128_interleaved(),),
        "Q3_K": (
            GroupedForwardPairSolution.q3_k_k128_interleaved(),
            GroupedForwardPairSolution.q3_k_k128_interleaved_row_tasks(),
        ),
    }.get(key.problem.quant_data_type, ())
    if key.solution not in supported_solutions:
        reasons.append(
            RejectReason(
                "grouped_forward_pair.solution.unimplemented",
                "paired grouped forward implements quant-specific K128 interleaving with serial-route or device-row-task ownership",
                ("Solution",),
                "GroupedForwardPairSolution",
            )
        )
    return tuple(reasons)
