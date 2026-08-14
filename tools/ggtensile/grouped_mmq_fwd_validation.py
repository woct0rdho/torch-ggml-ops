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
    if key.solution not in {
        GroupedForwardSolution.q4_k_serial_direct(),
        GroupedForwardSolution.q4_k_serial_decoded_lds(),
        GroupedForwardSolution.q4_k_serial_decoded_lds_64(),
        GroupedForwardSolution.q4_k_serial_decoded_lds_scheduled(),
        GroupedForwardSolution.q4_k_serial_decoded_lds_scheduled_a1d2p2(),
        GroupedForwardSolution.q4_k_serial_decoded_lds_scheduled_a1d4p2(),
        GroupedForwardSolution.q4_k_serial_decoded_lds_64_scheduled(),
        GroupedForwardSolution.q4_k_serial_decoded_lds_64_scheduled_mixed32(),
        GroupedForwardSolution.q4_k_serial_decoded_lds_64_scheduled_mixed32_a1d2p2(),
        GroupedForwardSolution.q4_k_serial_decoded_lds_64_scheduled_mixed32_a1d4p2(),
    }:
        reasons.append(
            RejectReason(
                "grouped_forward.solution.unimplemented",
                "grouped Q4_K forward currently implements serial direct-global and decoded-weight LDS controls",
                ("Solution",),
                "GroupedForwardSolution",
            )
        )
    return tuple(reasons)
