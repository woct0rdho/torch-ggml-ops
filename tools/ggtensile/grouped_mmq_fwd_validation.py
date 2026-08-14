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
    expected_problem = {
        "Q4_K": GroupedForwardProblem.q4_k,
        "Q5_K": GroupedForwardProblem.q5_k,
    }.get(key.problem.quant_data_type)
    if expected_problem is None or key.problem != expected_problem(
        key.problem.aggregate_rows
    ):
        reasons.append(
            RejectReason(
                "grouped_forward.problem.unsupported",
                "grouped Q4_K/Q5_K forward requires N=2048, K=512, 256 physical experts, and at most 256 route entries",
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
    q4_solutions = {
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
    }
    q5_solutions = {
        GroupedForwardSolution.q5_k_serial_decoded_lds(),
        GroupedForwardSolution.q5_k_serial_decoded_lds_scheduled(),
        GroupedForwardSolution.q5_k_serial_decoded_lds_scheduled_a1d2p2(),
        GroupedForwardSolution.q5_k_serial_decoded_lds_scheduled_mixed64(),
        GroupedForwardSolution.q5_k_serial_decoded_lds_scheduled_mixed64_a1d2p2(),
        GroupedForwardSolution.q5_k_serial_decoded_lds_scheduled_mixed64_a1d4p2(),
        GroupedForwardSolution.q5_k_serial_decoded_lds_scheduled_mixed64_mixed32(),
        GroupedForwardSolution.q5_k_serial_decoded_lds_scheduled_mixed64_mixed32_a1d2p2(),
        GroupedForwardSolution.q5_k_serial_decoded_lds_scheduled_mixed64_mixed32_a1d4p2(),
        GroupedForwardSolution.q5_k_serial_decoded_lds_64(),
        GroupedForwardSolution.q5_k_serial_decoded_lds_64_scheduled(),
        GroupedForwardSolution.q5_k_serial_decoded_lds_64_scheduled_a1d2p2(),
        GroupedForwardSolution.q5_k_serial_decoded_lds_64_scheduled_a1d4p2(),
        GroupedForwardSolution.q5_k_serial_decoded_lds_64_scheduled_mixed32(),
        GroupedForwardSolution.q5_k_serial_decoded_lds_64_scheduled_mixed32_a1d2p2(),
        GroupedForwardSolution.q5_k_serial_decoded_lds_64_scheduled_mixed32_a1d4p2(),
    }
    expected_solutions = {
        "Q4_K": q4_solutions,
        "Q5_K": q5_solutions,
    }.get(key.problem.quant_data_type, set())
    if key.solution not in expected_solutions:
        reasons.append(
            RejectReason(
                "grouped_forward.solution.unimplemented",
                "grouped forward currently implements Q4_K direct/decoded controls and Q5_K decoded controls",
                ("Solution",),
                "GroupedForwardSolution",
            )
        )
    return tuple(reasons)
