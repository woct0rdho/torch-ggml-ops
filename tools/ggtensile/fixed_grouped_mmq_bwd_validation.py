"""Validation gates for fixed-group Q8_0 backward."""

from .fixed_grouped_mmq_bwd_model import FixedBackwardProblem
from .fixed_grouped_mmq_bwd_physical import derive_fixed_backward_physical_plan
from .fixed_grouped_mmq_bwd_spec import (
    FixedBackwardKernelSpec,
    FixedBackwardProblemContract,
    validate_fixed_backward_problem,
)
from .mmq_bwd_physical import derive_backward_physical_plan
from .mmq_bwd_spec import DerivedBackwardState
from .model import ProblemSize, ProblemType
from .physical_resources import GFX1151_RESOURCE_CAPACITY
from .validation import _validate_backward_spec_record
from .work_group_mapping import mapped_grid_extent, mapped_m_tile_count


def validate_fixed_backward_solution(
    problem: FixedBackwardProblem,
    spec: FixedBackwardKernelSpec,
) -> None:
    validate_fixed_backward_problem(problem)
    compute = spec.compute
    assert spec.work_group_order in ("MMajor", "NMajor")
    assert spec.store_schedule in ("ElementSerial", "ClausePairs")
    assert spec.group_axis == "WorkGroupZ"
    assert spec.row_layout == "TokenMajorGroupInterleaved"
    assert spec.packed_weight_layout == "GroupMajorRows"
    _validate_backward_spec_record(
        ProblemType.mmq_backward("Q8_0"),
        ProblemSize(
            problem.tokens,
            problem.input_features,
            problem.output_features,
        ),
        compute,
    )
    assert problem.tokens % compute.geometry.macro_tile0 == 0
    assert problem.input_features % compute.geometry.macro_tile1 == 0
    m_tiles = problem.tokens // compute.geometry.macro_tile0
    n_tiles = problem.input_features // compute.geometry.macro_tile1
    mapping = compute.geometry.work_group_mapping
    mapped_m_tile_count(m_tiles, mapping)
    mapped_grid_extent(n_tiles, mapping)
    contract = FixedBackwardProblemContract.for_problem(problem)
    ordinary = DerivedBackwardState.from_contract_spec(
        contract.ordinary(problem.tokens), compute
    )
    derive_fixed_backward_physical_plan(
        derive_backward_physical_plan(ordinary)
    ).resources.admit(GFX1151_RESOURCE_CAPACITY)
