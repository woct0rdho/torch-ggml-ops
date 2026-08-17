"""Validation for isolated grouped MMQ forward solution keys."""

from .grouped_mmq_fwd_model import (
    GroupedForwardSolutionKey,
)
from .grouped_mmq_fwd_spec import grouped_forward_capability_rejection_reason
from .validation import RejectReason


def validate_grouped_forward_solution(
    key: GroupedForwardSolutionKey,
) -> tuple[RejectReason, ...]:
    reasons: list[RejectReason] = []
    expected_shape = {
        "Q2_K": (4096, 2048),
        "Q4_K": (2048, 512),
        "Q5_K": (2048, 512),
        "IQ2_S": (2048, 512),
    }.get(key.problem.quant_data_type)
    if (
        expected_shape is None
        or (key.problem.output_features, key.problem.input_features) != expected_shape
        or key.problem.physical_experts != 256
        or key.problem.max_route_entries != 256
    ):
        reasons.append(
            RejectReason(
                "grouped_forward.problem.unsupported",
                "grouped Q2_K/Q4_K/Q5_K/IQ2_S forward requires its exact production N/K shape, 256 physical experts, and at most 256 route entries",
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
    rejection = grouped_forward_capability_rejection_reason(
        key.problem,
        key.solution,
    )
    if rejection is not None:
        reasons.append(
            RejectReason(
                "grouped_forward.solution.unimplemented",
                rejection,
                ("Solution",),
                "GroupedForwardSolution",
            )
        )
    return tuple(reasons)
