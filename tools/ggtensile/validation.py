from .grouped_mmq_bwd_spec import (
    DerivedGroupedBackwardState,
    GroupedBackwardKernelSpec,
    GroupedBackwardProblemContract,
)
from .identity import KernelFamily
from .kernel_instance import KernelInstance
from .mmq_bwd_physical import derive_backward_physical_plan
from .mmq_bwd_spec import (
    BackwardKernelSpec,
    BackwardProblemContract,
    DerivedBackwardState,
    backward_mechanism_contract,
)
from .mmq_fwd_spec import (
    ForwardKernelSpec,
    ForwardProblemContract,
    validate_forward_kernel_spec_record,
)
from .model import ProblemSize, ProblemType
from .physical_resources import GFX1151_RESOURCE_CAPACITY
from .quant_formats import BACKWARD_QUANT_FORMATS, QUANT_FORMATS
from .work_group_mapping import (
    mapped_grid_extent,
    mapped_m_tile_count,
    mapped_route_stride,
)


def _validate_backward_problem_type(problem_type: ProblemType) -> None:
    assert problem_type.quant_data_type in BACKWARD_QUANT_FORMATS
    expected = ProblemType.mmq_backward(problem_type.quant_data_type)
    for attribute in (
        "operation_type",
        "quant_data_type",
        "data_type_a",
        "data_type_b",
        "dest_data_type",
        "compute_data_type",
        "transpose_a",
        "transpose_b",
    ):
        assert getattr(problem_type, attribute) == getattr(expected, attribute)


def validate_forward_solution(
    problem_size: ProblemSize,
    quant_type: str,
    spec: ForwardKernelSpec,
) -> None:
    problem_type = ProblemType.mmq_forward(quant_type)
    supported_problem_types = {
        ProblemType.mmq_forward(quant_type) for quant_type in QUANT_FORMATS
    }
    assert problem_type in supported_problem_types
    validate_forward_kernel_spec_record(
        problem_size,
        ForwardProblemContract.for_quant_type(problem_type.quant_data_type),
        spec,
    )


def _validate_backward_spec_record(
    problem_type: ProblemType,
    problem_size: ProblemSize,
    spec: BackwardKernelSpec,
    *,
    require_row_tiles: bool = True,
    require_mapping_tiles: bool = True,
) -> BackwardKernelSpec:
    quant_type = problem_type.quant_data_type
    contract = BackwardProblemContract(
        problem_size,
        quant_type,
        BACKWARD_QUANT_FORMATS[quant_type],
        backward_mechanism_contract(quant_type),
    )
    spec.validate(contract)
    for value in (problem_size.m, problem_size.n, problem_size.k):
        assert value > 0
    assert problem_size.n % contract.quant_format.block_values == 0
    if quant_type == "Q8_0":
        assert spec.geometry.macro_tile1 % contract.quant_format.block_values == 0
    else:
        assert contract.quant_format.block_values % spec.geometry.macro_tile1 == 0
    if require_row_tiles:
        assert problem_size.m % spec.geometry.macro_tile0 == 0
    assert problem_size.n % spec.geometry.macro_tile1 == 0
    assert problem_size.k % spec.geometry.depth_u == 0
    decoder_threads = min(spec.geometry.num_threads, 128)
    assert (
        spec.geometry.depth_u
        * spec.geometry.macro_tile1
        // (decoder_threads * spec.decode.decoder_width)
    ) > 0
    if require_mapping_tiles:
        mapped_m_tile_count(
            problem_size.m // spec.geometry.macro_tile0,
            spec.geometry.work_group_mapping,
        )
    return spec


def _validate_backward_state_problem_size(
    problem_type: ProblemType,
    problem_size: ProblemSize,
    state: DerivedBackwardState,
    *,
    require_mapping_tiles: bool = True,
) -> None:
    _validate_backward_spec_record(
        problem_type,
        problem_size,
        state.spec,
        require_mapping_tiles=require_mapping_tiles,
    )


def _validate_backward_physical_state(state: DerivedBackwardState) -> None:
    physical = derive_backward_physical_plan(state)
    physical.resources.admit(GFX1151_RESOURCE_CAPACITY)


def validate_backward_solution(
    problem_size: ProblemSize,
    quant_type: str,
    spec: BackwardKernelSpec,
) -> None:
    problem_type = ProblemType.mmq_backward(quant_type)
    _validate_backward_problem_type(problem_type)
    _validate_backward_spec_record(
        problem_type,
        problem_size,
        spec,
    )
    state = DerivedBackwardState.from_problem_spec(problem_size, quant_type, spec)
    _validate_backward_physical_state(state)


def validate_grouped_backward_solution(
    problem_size: ProblemSize,
    quant_type: str,
    spec: GroupedBackwardKernelSpec,
) -> None:
    GroupedBackwardProblemContract.for_problem(problem_size, quant_type)
    compute = spec.compute
    ordinary_problem_type = ProblemType.mmq_backward(quant_type)
    _validate_backward_problem_type(ordinary_problem_type)
    m_tile = compute.geometry.macro_tile0
    assert m_tile > 0
    padded_m = (problem_size.m + m_tile - 1) // m_tile * m_tile
    padded_problem_size = ProblemSize(padded_m, problem_size.n, problem_size.k)
    _validate_backward_spec_record(
        ordinary_problem_type,
        padded_problem_size,
        compute,
        require_mapping_tiles=False,
    )
    state = DerivedGroupedBackwardState.from_problem_spec(
        problem_size, quant_type, spec
    )
    _validate_backward_physical_state(state.primary)
    split_factor = state.spec.ownership.split_factor
    mapping = compute.geometry.work_group_mapping
    mapped_grid_extent(problem_size.n // compute.geometry.macro_tile1, mapping)
    effective_split = mapped_route_stride(split_factor, mapping)
    assert effective_split * compute.geometry.macro_tile0 <= 0xFFFFFFFF
    row_tail = state.spec.row_tail
    if row_tail.is_secondary:
        # The secondary emitter has no decomposed M-task traversal yet.
        assert mapping == 1
        assert state.secondary is not None
        secondary_tile = state.secondary.spec.geometry.macro_tile0
        padded_secondary_m = (
            (problem_size.m + secondary_tile - 1) // secondary_tile * secondary_tile
        )
        _validate_backward_state_problem_size(
            ordinary_problem_type,
            ProblemSize(
                padded_secondary_m,
                state.contract.problem_size.n,
                state.contract.problem_size.k,
            ),
            state.secondary,
        )
        _validate_backward_physical_state(state.secondary)


def validate_instance(instance: KernelInstance) -> None:
    from .fixed_grouped_mmq_bwd_model import FixedBackwardProblem
    from .fixed_grouped_mmq_bwd_spec import FixedBackwardKernelSpec
    from .fixed_grouped_mmq_bwd_validation import validate_fixed_backward_solution
    from .fixed_grouped_mmq_fwd_model import FixedForwardProblem
    from .fixed_grouped_mmq_fwd_spec import FixedForwardKernelSpec
    from .fixed_grouped_mmq_fwd_validation import validate_fixed_forward_solution
    from .grouped_mmq_bwd_pair_model import GroupedBackwardPairProblem
    from .grouped_mmq_bwd_pair_spec import GroupedBackwardPairKernelSpec
    from .grouped_mmq_bwd_pair_validation import (
        validate_grouped_backward_pair_solution,
    )
    from .grouped_mmq_fwd_model import GroupedForwardProblem
    from .grouped_mmq_fwd_pair_model import GroupedForwardPairProblem
    from .grouped_mmq_fwd_pair_spec import GroupedForwardPairKernelSpec
    from .grouped_mmq_fwd_pair_validation import validate_grouped_forward_pair_solution
    from .grouped_mmq_fwd_spec import GroupedForwardKernelSpec
    from .grouped_mmq_fwd_validation import validate_grouped_forward_solution

    family = instance.family
    problem = instance.problem
    spec = instance.kernel_spec
    quant_type = instance.problem_type.quant_data_type
    if family is KernelFamily.OrdinaryForward:
        assert isinstance(problem, ProblemSize) and isinstance(spec, ForwardKernelSpec)
        validate_forward_solution(problem, quant_type, spec)
    elif family is KernelFamily.OrdinaryBackward:
        assert isinstance(problem, ProblemSize) and isinstance(spec, BackwardKernelSpec)
        validate_backward_solution(problem, quant_type, spec)
    elif family is KernelFamily.GroupedBackward:
        assert isinstance(problem, ProblemSize) and isinstance(
            spec, GroupedBackwardKernelSpec
        )
        validate_grouped_backward_solution(problem, quant_type, spec)
    elif family is KernelFamily.GroupedForward:
        assert isinstance(problem, GroupedForwardProblem) and isinstance(
            spec, GroupedForwardKernelSpec
        )
        validate_grouped_forward_solution(problem, spec)
    elif family is KernelFamily.GroupedForwardPair:
        assert isinstance(problem, GroupedForwardPairProblem) and isinstance(
            spec, GroupedForwardPairKernelSpec
        )
        validate_grouped_forward_pair_solution(problem, spec)
    elif family is KernelFamily.GroupedBackwardPair:
        assert isinstance(problem, GroupedBackwardPairProblem) and isinstance(
            spec, GroupedBackwardPairKernelSpec
        )
        validate_grouped_backward_pair_solution(problem, spec)
    elif family is KernelFamily.FixedGroupedForward:
        assert isinstance(problem, FixedForwardProblem) and isinstance(
            spec, FixedForwardKernelSpec
        )
        validate_fixed_forward_solution(problem, spec)
    elif family is KernelFamily.FixedGroupedBackward:
        assert isinstance(problem, FixedBackwardProblem) and isinstance(
            spec, FixedBackwardKernelSpec
        )
        validate_fixed_backward_solution(problem, spec)
    else:
        raise TypeError(f"unsupported kernel family {family!r}")
