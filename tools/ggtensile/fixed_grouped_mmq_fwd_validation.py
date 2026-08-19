"""Validation gates for the fixed-group Q8_0 forward experiment."""

from .fixed_grouped_mmq_fwd_model import (
    FixedForwardOperandSource,
    FixedForwardSolutionKey,
)
from .fixed_grouped_mmq_fwd_spec import FixedForwardProblemContract


def fixed_forward_rejection_reason(key: FixedForwardSolutionKey) -> str | None:
    """Return the first contract violation without deriving or emitting state."""
    contract_rejection = FixedForwardProblemContract.rejection_reason(
        key.problem, key.solution
    )
    if contract_rejection is not None:
        return contract_rejection
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
