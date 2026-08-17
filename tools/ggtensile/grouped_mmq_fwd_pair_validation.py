"""Strict validation for research-only paired grouped forward identities."""

from .grouped_mmq_fwd_pair_model import (
    GroupedForwardPairSolutionKey,
)
from .grouped_mmq_fwd_pair_spec import (
    grouped_forward_pair_capability_rejection_reason,
)
from .validation import RejectReason


def validate_grouped_forward_pair_solution(
    key: GroupedForwardPairSolutionKey,
) -> tuple[RejectReason, ...]:
    reasons: list[RejectReason] = []
    expected_shape = {
        "IQ2_S": (512, 2048),
        "IQ2_XXS": (2048, 4096),
        "Q3_K": (512, 2048),
    }.get(key.problem.quant_data_type)
    if (
        expected_shape is None
        or (key.problem.output_features, key.problem.input_features) != expected_shape
        or key.problem.projection_count != 2
        or key.problem.physical_experts != 256
        or key.problem.max_route_entries != 256
    ):
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
    rejection = grouped_forward_pair_capability_rejection_reason(
        key.problem,
        key.solution,
    )
    if rejection is not None:
        reasons.append(
            RejectReason(
                "grouped_forward_pair.solution.unimplemented",
                rejection,
                ("Solution",),
                "GroupedForwardPairSolution",
            )
        )
    return tuple(reasons)
