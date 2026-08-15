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
        "Q2_K": GroupedForwardProblem.q2_k,
        "Q4_K": GroupedForwardProblem.q4_k,
        "Q5_K": GroupedForwardProblem.q5_k,
    }.get(key.problem.quant_data_type)
    if expected_problem is None or key.problem != expected_problem(
        key.problem.aggregate_rows
    ):
        reasons.append(
            RejectReason(
                "grouped_forward.problem.unsupported",
                "grouped Q2_K/Q4_K/Q5_K forward requires its exact production N/K shape, 256 physical experts, and at most 256 route entries",
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
    q2_solutions = {
        GroupedForwardSolution.q2_k_serial_decoded_lds_32(),
        GroupedForwardSolution.q2_k_serial_decoded_lds_64(),
        GroupedForwardSolution.q2_k_serial_decoded_lds_32_unrolled(),
        GroupedForwardSolution.q2_k_serial_decoded_lds_64_unrolled(),
        GroupedForwardSolution.q2_k_serial_decoded_lds_128_unrolled(),
        GroupedForwardSolution.q2_k_serial_decoded_lds_32_hip_association(),
        GroupedForwardSolution.q2_k_serial_decoded_lds_64_hip_association(),
        GroupedForwardSolution.q2_k_serial_decoded_lds_32_hip_partial_lds(),
        GroupedForwardSolution.q2_k_serial_decoded_lds_32_hip_pre_negated_dm(),
        GroupedForwardSolution.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2(),
        GroupedForwardSolution.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2(),
        GroupedForwardSolution.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2_distributed(),
        GroupedForwardSolution.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2_mixed16(),
        GroupedForwardSolution.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2_distributed_mixed16(),
        GroupedForwardSolution.q2_k_serial_decoded_lds_64_hip_distributed(),
    }
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
        "Q2_K": q2_solutions,
        "Q4_K": q4_solutions,
        "Q5_K": q5_solutions,
    }.get(key.problem.quant_data_type, set())
    if key.solution not in expected_solutions:
        reasons.append(
            RejectReason(
                "grouped_forward.solution.unimplemented",
                "grouped forward currently implements Q2_K/Q4_K/Q5_K decoded controls and the Q4_K direct control",
                ("Solution",),
                "GroupedForwardSolution",
            )
        )
    return tuple(reasons)
