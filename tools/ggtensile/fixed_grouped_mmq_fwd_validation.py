"""Validation gates for the fixed-group Q8_0 forward experiment."""

from .fixed_grouped_mmq_fwd_model import (
    FixedForwardOperandSource,
    FixedForwardSolutionKey,
)
from .fixed_grouped_mmq_fwd_spec import FixedForwardProblemContract


def fixed_forward_rejection_reason(key: FixedForwardSolutionKey) -> str | None:
    """Return the first contract violation without deriving or emitting state."""
    try:
        FixedForwardProblemContract.from_problem(key.problem, key.solution)
    except ValueError as error:
        return str(error)
    problem = key.problem
    solution = key.solution
    checks = (
        (
            solution.work_group == (32, 4, 1),
            "fixed Q8 forward requires workgroup (32, 4, 1)",
        ),
        (
            solution.matrix_instruction == (16, 16, 16, 1, 1, 1, 4, 4, 1),
            "fixed Q8 forward requires the signed-int8 WMMA instruction",
        ),
        (
            solution.macro_tile_tokens == 64,
            "fixed Q8 forward requires a 64-token tile",
        ),
        (
            solution.macro_tile_features == 64,
            "fixed Q8 forward requires a 64-feature tile",
        ),
        (solution.depth_u == 32, "fixed Q8 forward requires DepthU=32"),
        (
            solution.activation_addressing == "FixedGroupRows",
            "fixed Q8 forward requires fixed-group activation addressing",
        ),
        (
            solution.output_store == "BFloat16RNEClauseTile",
            "fixed Q8 forward requires the BF16 RNE tile store",
        ),
        (
            solution.num_threads == 128,
            "fixed Q8 forward requires 128 work-items",
        ),
        (
            key.problem.packed_row_bytes == 4352,
            "fixed Q8 forward requires 4352 packed bytes per weight row",
        ),
        (
            key.problem.bytes_per_group == problem.output_features * 4352,
            "fixed Q8 forward bytes_per_group is inconsistent",
        ),
        (
            solution.operand_source is FixedForwardOperandSource.Q8SmallMTiledLds,
            "fixed Q8 forward operand source is not retained",
        ),
        (
            solution.lds_address_hoist in ("SmallMTile", "CompactDepth32WeightRows"),
            "fixed Q8 forward LDS addressing policy is not retained",
        ),
        (
            solution.fixed_address_hoist
            in ("None", "ReductionLoop", "ReductionLoopAndWeightStage"),
            "fixed Q8 forward address hoist is not retained",
        ),
        (
            solution.fixed_address_hoist == "None"
            or solution.lds_address_hoist == "CompactDepth32WeightRows",
            "fixed Q8 forward reduction-loop hoist requires compact DepthU32 LDS",
        ),
    )
    return next((message for accepted, message in checks if not accepted), None)


def validate_fixed_forward_solution_key(key: FixedForwardSolutionKey) -> None:
    rejection = fixed_forward_rejection_reason(key)
    if rejection is not None:
        raise ValueError(rejection)
