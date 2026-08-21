"""Validation gates for fixed-group Q8_0 backward."""

from .fixed_grouped_mmq_bwd_model import (
    FixedBackwardSolution,
    FixedBackwardSolutionKey,
)
from .fixed_grouped_mmq_bwd_spec import fixed_backward_problem_rejection_reason
from .model import ProblemSize, ProblemType, SolutionKey
from .validation import validate_solution


def fixed_backward_rejection_reason(
    key: FixedBackwardSolutionKey,
) -> str | None:
    rejection = fixed_backward_problem_rejection_reason(key.problem)
    if rejection is not None:
        return rejection
    solution = key.solution
    compute = solution.compute
    checks = (
        (
            solution.work_group_order in ("MMajor", "NMajor"),
            "fixed backward workgroup order must be MMajor or NMajor",
        ),
        (
            solution.store_schedule in ("ElementSerial", "ClausePairs"),
            "fixed backward store schedule is not retained",
        ),
        (
            solution.group_axis == "WorkGroupZ",
            "fixed backward groups must use workgroup Z",
        ),
        (
            solution.row_layout == "TokenMajorGroupInterleaved",
            "fixed backward requires token-major interleaved groups",
        ),
        (
            solution.packed_weight_layout == "GroupMajorRows",
            "fixed backward requires group-major packed weights",
        ),
        (
            compute.work_group_mapping == 1,
            "fixed backward requires direct M workgroup mapping",
        ),
        (
            key.problem.tokens % compute.macro_tile0 == 0,
            "fixed backward tokens must contain complete M tiles",
        ),
        (
            key.problem.input_features % compute.macro_tile1 == 0,
            "fixed backward input features must contain complete N tiles",
        ),
    )
    policy_rejection = next(
        (message for accepted, message in checks if not accepted), None
    )
    if policy_rejection is not None:
        return policy_rejection
    ordinary_key = SolutionKey(
        ProblemType.mmq_backward("Q8_0"),
        ProblemSize(
            key.problem.tokens,
            key.problem.input_features,
            key.problem.output_features,
        ),
        compute,
    )
    reasons = validate_solution(ordinary_key)
    if key.solution.compute == FixedBackwardSolution.q8_0_m64_n256_k64().compute:
        # This complete fixed identity is the sole exception to the ordinary
        # writer's intentionally narrower N-repeat capability.
        reasons = tuple(
            reason
            for reason in reasons
            if reason.rule_id != "solution.geometry.unimplemented"
        )
    if reasons:
        return "; ".join(f"{reason.rule_id}: {reason.message}" for reason in reasons)
    return None


def validate_fixed_backward_solution_key(key: FixedBackwardSolutionKey) -> None:
    rejection = fixed_backward_rejection_reason(key)
    if rejection is not None:
        raise ValueError(rejection)
