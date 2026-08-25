"""Derived authorities for research-only paired grouped backward kernels."""

from dataclasses import dataclass

from .grouped_mmq_bwd_pair_model import (
    GroupedBackwardPairProblem,
    GroupedBackwardPairProjectionPolicy,
    GroupedBackwardPairRouteOwnership,
    PairCodebookStorage,
    PairPointerMode,
    PairReadConcurrency,
)
from .mmq_bwd_spec import (
    BackwardKernelSpec,
    BackwardProblemContract,
    DerivedBackwardState,
    backward_mechanism_contract,
)
from .model import ProblemSize
from .physical_resources import GFX1151_RESOURCE_CAPACITY
from .quant_formats import BACKWARD_QUANT_FORMATS
from .schema import SchemaError
from .schema import boolean as _boolean
from .schema import enum_value as _enum
from .schema import integer as _integer
from .schema import strict_mapping as _mapping
from .schema import strict_mapping_optional as _mapping_optional
from .schema import string as _string
from .work_group_mapping import (
    mapped_grid_extent,
    mapped_route_stride,
)

_U32_MAX = 0xFFFFFFFF


@dataclass(frozen=True)
class GroupedBackwardPairContract:
    quant_type: str
    out_features: int
    in_features: int
    physical_experts: int
    max_route_entries: int
    block_values: int
    packed_weight_block_bytes: int
    kernel_language: str
    isa: tuple[int, int, int]
    wavefront_size: int
    projection_count: int
    arithmetic_contract: str
    destination_type: str
    bf16_rounding: str
    abi_family: str

    @classmethod
    def for_problem(
        cls, problem: GroupedBackwardPairProblem
    ) -> "GroupedBackwardPairContract":
        quant_format = BACKWARD_QUANT_FORMATS.get(problem.quant_data_type)
        assert problem.quant_data_type in {"Q3_K", "IQ2_S", "IQ2_XXS"}
        assert quant_format is not None
        validate_grouped_backward_pair_problem(problem)
        return cls(
            quant_type=problem.quant_data_type,
            out_features=problem.out_features,
            in_features=problem.in_features,
            physical_experts=problem.physical_experts,
            max_route_entries=problem.max_route_entries,
            block_values=quant_format.block_values,
            packed_weight_block_bytes=quant_format.block_bytes,
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            projection_count=problem.projection_count,
            arithmetic_contract="InterleavedPairFP32Accumulator",
            destination_type="BFloat16",
            bf16_rounding="RNEPreserveNaN",
            abi_family="GroupedBackwardPairV1",
        )

    def problem(self, aggregate_rows: int) -> GroupedBackwardPairProblem:
        return GroupedBackwardPairProblem(
            self.quant_type,
            aggregate_rows,
            self.out_features,
            self.in_features,
            self.physical_experts,
            self.max_route_entries,
            self.projection_count,
        )

    def ordinary(self, aggregate_rows: int) -> BackwardProblemContract:
        problem_size = ProblemSize(aggregate_rows, self.in_features, self.out_features)
        quant_format = BACKWARD_QUANT_FORMATS[self.quant_type]
        return BackwardProblemContract(
            problem_size,
            self.quant_type,
            quant_format,
            backward_mechanism_contract(self.quant_type),
        )


def _projection_policy_from_mapping(
    value: object,
) -> GroupedBackwardPairProjectionPolicy:
    item = _mapping(
        value,
        "GroupedBackwardPairKernelSpec.Projection",
        frozenset(
            {
                "DualLds",
                "CodebookStorage",
                "SplitFullTiles",
                "PointerMode",
                "ReadConcurrency",
                "ActivationPrefetch",
                "PipelineK",
            }
        ),
    )
    return GroupedBackwardPairProjectionPolicy(
        _boolean(item, "DualLds"),
        _enum(item, "CodebookStorage", PairCodebookStorage),
        _boolean(item, "SplitFullTiles"),
        _enum(item, "PointerMode", PairPointerMode),
        _enum(item, "ReadConcurrency", PairReadConcurrency),
        _boolean(item, "ActivationPrefetch"),
        _boolean(item, "PipelineK"),
    )


def _projection_policy_to_mapping(
    policy: GroupedBackwardPairProjectionPolicy,
) -> dict[str, object]:
    return {
        "DualLds": policy.dual_lds,
        "CodebookStorage": policy.codebook_storage.value,
        "SplitFullTiles": policy.split_full_tiles,
        "PointerMode": policy.pointer_mode.value,
        "ReadConcurrency": policy.read_concurrency.value,
        "ActivationPrefetch": policy.activation_prefetch,
        "PipelineK": policy.pipeline_k,
    }


def _route_ownership_from_mapping(
    value: object,
) -> GroupedBackwardPairRouteOwnership:
    item = _mapping_optional(
        value,
        "GroupedBackwardPairKernelSpec.Ownership",
        frozenset({"Kind"}),
        frozenset({"RouteSplitFactor"}),
    )
    kind = _string(item, "Kind")
    if kind == "Serial":
        if "RouteSplitFactor" in item:
            raise SchemaError("serial paired ownership has no route split factor")
        return GroupedBackwardPairRouteOwnership.serial()
    if kind != "PackedSplit" or "RouteSplitFactor" not in item:
        raise SchemaError("packed-split ownership requires RouteSplitFactor")
    factor = _integer(item, "RouteSplitFactor")
    if factor <= 1 or factor & (factor - 1):
        raise SchemaError("paired route split factor must be a power of two")
    return GroupedBackwardPairRouteOwnership.packed_split(factor)


def _route_ownership_to_mapping(
    ownership: GroupedBackwardPairRouteOwnership,
) -> dict[str, object]:
    result: dict[str, object] = {"Kind": ownership.kind}
    if ownership.kind != "Serial":
        result["RouteSplitFactor"] = ownership.split_factor
    return result


@dataclass(frozen=True)
class GroupedBackwardPairKernelSpec:
    compute: BackwardKernelSpec
    projection_policy: GroupedBackwardPairProjectionPolicy
    route_ownership: GroupedBackwardPairRouteOwnership

    @classmethod
    def from_mapping(
        cls, value: object, contract: GroupedBackwardPairContract
    ) -> "GroupedBackwardPairKernelSpec":
        item = _mapping(
            value,
            "GroupedBackwardPairKernelSpec",
            frozenset({"Compute", "Projection", "Ownership"}),
        )
        return cls(
            BackwardKernelSpec.from_mapping(item["Compute"], contract.quant_type),
            _projection_policy_from_mapping(item["Projection"]),
            _route_ownership_from_mapping(item["Ownership"]),
        )

    def to_mapping(self, contract: GroupedBackwardPairContract) -> dict[str, object]:
        return {
            "Compute": self.compute.to_mapping(contract.quant_type),
            "Projection": _projection_policy_to_mapping(self.projection_policy),
            "Ownership": _route_ownership_to_mapping(self.route_ownership),
        }


def validate_grouped_backward_pair_problem(problem: GroupedBackwardPairProblem) -> None:
    assert problem.quant_data_type in {"Q3_K", "IQ2_S", "IQ2_XXS"}
    assert problem.physical_experts == 256
    assert problem.max_route_entries == 256
    assert problem.projection_count == 2
    for value in (
        problem.aggregate_rows,
        problem.out_features,
        problem.in_features,
    ):
        assert 0 < value <= _U32_MAX
    quant_format = BACKWARD_QUANT_FORMATS[problem.quant_data_type]
    assert problem.in_features % quant_format.block_values == 0
    assert problem.out_features % 16 == 0
    packed_row_bytes = (
        problem.in_features // quant_format.block_values * quant_format.block_bytes
    )
    bytes_per_expert = problem.out_features * packed_row_bytes
    assert bytes_per_expert <= _U32_MAX
    assert problem.aggregate_rows * problem.out_features * 2 <= _U32_MAX
    assert problem.aggregate_rows * problem.in_features * 2 <= _U32_MAX


def validate_grouped_backward_pair_capability(
    problem: GroupedBackwardPairProblem,
    kernel_spec: GroupedBackwardPairKernelSpec,
) -> None:
    contract = GroupedBackwardPairContract.for_problem(problem)
    compute = kernel_spec.compute
    ordinary_contract = contract.ordinary(problem.aggregate_rows)

    assert problem.projection_count == 2
    compute.validate(ordinary_contract)
    assert problem.in_features % compute.geometry.macro_tile1 == 0
    assert problem.out_features % compute.geometry.depth_u == 0
    kernel_spec.projection_policy.validate(problem.quant_data_type, compute)
    kernel_spec.route_ownership.validate(compute.geometry.macro_tile0)
    mapping = compute.geometry.work_group_mapping
    mapped_grid_extent(problem.in_features // compute.geometry.macro_tile1, mapping)
    effective_split = mapped_route_stride(
        kernel_spec.route_ownership.split_factor,
        mapping,
    )
    assert effective_split * compute.geometry.macro_tile0 <= _U32_MAX

    state = DerivedGroupedBackwardPairState.from_problem_spec(problem, kernel_spec)
    from .grouped_mmq_bwd_pair_physical import (
        derive_grouped_backward_pair_physical_plan,
    )

    physical = derive_grouped_backward_pair_physical_plan(state)
    for plan in (physical.ordinary, physical.second_projection):
        if plan is not None:
            plan.resources.admit(GFX1151_RESOURCE_CAPACITY)


@dataclass(frozen=True)
class DerivedGroupedBackwardPairState:
    contract: GroupedBackwardPairContract
    kernel_spec: GroupedBackwardPairKernelSpec
    ordinary: DerivedBackwardState

    @classmethod
    def from_problem_spec(
        cls,
        problem: GroupedBackwardPairProblem,
        kernel_spec: GroupedBackwardPairKernelSpec,
    ) -> "DerivedGroupedBackwardPairState":
        contract = GroupedBackwardPairContract.for_problem(problem)
        ordinary_contract = contract.ordinary(problem.aggregate_rows)
        ordinary = DerivedBackwardState.from_contract_spec(
            ordinary_contract, kernel_spec.compute
        )
        return cls(contract, kernel_spec, ordinary)

    @property
    def packed_row_bytes(self) -> int:
        return (
            self.contract.in_features
            // self.contract.block_values
            * self.contract.packed_weight_block_bytes
        )

    @property
    def bytes_per_expert(self) -> int:
        return self.contract.out_features * self.packed_row_bytes

    @property
    def expected_packed_weight_shape(self) -> tuple[int, int, int]:
        return (
            self.contract.physical_experts,
            self.contract.out_features,
            self.packed_row_bytes,
        )
