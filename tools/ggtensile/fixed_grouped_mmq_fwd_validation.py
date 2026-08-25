"""Validation gates for the fixed-group Q8_0 forward experiment."""

from .fixed_grouped_mmq_fwd_model import FixedForwardProblem
from .fixed_grouped_mmq_fwd_physical import fixed_q8_forward_physical_plan
from .fixed_grouped_mmq_fwd_spec import (
    FixedForwardKernelSpec,
    FixedForwardProblemContract,
)
from .physical_resources import GFX1151_RESOURCE_CAPACITY


def validate_fixed_forward_solution(
    problem: FixedForwardProblem,
    spec: FixedForwardKernelSpec,
) -> None:
    contract = FixedForwardProblemContract.for_problem(problem)
    assert spec.macro_tile_tokens == 64
    assert spec.macro_tile_features == 64
    fixed_q8_forward_physical_plan(
        spec.forward_kernel_spec(contract), spec.fixed_address_hoist
    ).resources.admit(GFX1151_RESOURCE_CAPACITY)
