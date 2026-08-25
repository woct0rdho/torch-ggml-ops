"""Typed contract and derived state for fixed-group Q8_0 backward."""

from dataclasses import dataclass

from .fixed_grouped_mmq_bwd_model import FixedBackwardProblem
from .fixed_grouped_mmq_bwd_physical import (
    FixedBackwardPhysicalPlan,
    derive_fixed_backward_physical_plan,
)
from .mmq_bwd_physical import derive_backward_physical_plan
from .mmq_bwd_spec import (
    BackwardKernelSpec,
    BackwardProblemContract,
    DerivedBackwardState,
    backward_mechanism_contract,
)
from .model import ProblemSize
from .quant_formats import BACKWARD_QUANT_FORMATS
from .schema import strict_mapping as _mapping
from .schema import string as _string
from .work_group_mapping import (
    mapped_grid_extent,
    mapped_m_tile_count,
)

_U32_MAX = 0xFFFFFFFF


def validate_fixed_backward_problem(problem: FixedBackwardProblem) -> None:
    assert problem.quant_data_type == "Q8_0"
    for value in (
        problem.tokens,
        problem.output_features,
        problem.input_features,
    ):
        assert 0 < value <= _U32_MAX
    assert problem.groups == 8
    quant = BACKWARD_QUANT_FORMATS["Q8_0"]
    assert problem.input_features % quant.block_values == 0
    assert problem.output_features % 16 == 0
    byte_counts = (
        problem.bytes_per_group,
        problem.tokens * problem.groups * problem.output_features * 2,
        problem.tokens * problem.groups * problem.input_features * 2,
    )
    assert all(value <= _U32_MAX for value in byte_counts)


@dataclass(frozen=True)
class FixedBackwardProblemContract:
    quant_type: str = "Q8_0"
    output_features: int = 1024
    input_features: int = 4096
    groups: int = 8

    @classmethod
    def for_problem(
        cls, problem: FixedBackwardProblem
    ) -> "FixedBackwardProblemContract":
        validate_fixed_backward_problem(problem)
        return cls(
            problem.quant_data_type,
            problem.output_features,
            problem.input_features,
            problem.groups,
        )

    def problem(self, tokens: int) -> FixedBackwardProblem:
        return FixedBackwardProblem(
            self.quant_type,
            tokens,
            self.output_features,
            self.input_features,
            self.groups,
        )

    def ordinary(self, tokens: int) -> BackwardProblemContract:
        quant = BACKWARD_QUANT_FORMATS[self.quant_type]
        return BackwardProblemContract(
            ProblemSize(tokens, self.input_features, self.output_features),
            self.quant_type,
            quant,
            backward_mechanism_contract(self.quant_type),
        )


@dataclass(frozen=True)
class FixedBackwardKernelSpec:
    compute: BackwardKernelSpec
    work_group_order: str
    store_schedule: str
    group_axis: str
    row_layout: str
    packed_weight_layout: str

    @classmethod
    def from_mapping(cls, value: object):
        item = _mapping(
            value,
            "FixedBackwardKernelSpec",
            frozenset({"Compute", "Addressing", "Epilogue"}),
        )
        addressing = _mapping(
            item["Addressing"],
            "Addressing",
            frozenset(
                {
                    "WorkGroupOrder",
                    "GroupAxis",
                    "RowLayout",
                    "PackedWeightLayout",
                }
            ),
        )
        epilogue = _mapping(item["Epilogue"], "Epilogue", frozenset({"StoreSchedule"}))
        return cls(
            BackwardKernelSpec.from_mapping(item["Compute"], "Q8_0"),
            _string(addressing, "WorkGroupOrder"),
            _string(epilogue, "StoreSchedule"),
            _string(addressing, "GroupAxis"),
            _string(addressing, "RowLayout"),
            _string(addressing, "PackedWeightLayout"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "Compute": self.compute.to_mapping("Q8_0"),
            "Epilogue": {"StoreSchedule": self.store_schedule},
            "Addressing": {
                "WorkGroupOrder": self.work_group_order,
                "GroupAxis": self.group_axis,
                "RowLayout": self.row_layout,
                "PackedWeightLayout": self.packed_weight_layout,
            },
        }


@dataclass(frozen=True)
class DerivedFixedBackwardState:
    problem: FixedBackwardProblem
    contract: FixedBackwardProblemContract
    spec: FixedBackwardKernelSpec
    ordinary: DerivedBackwardState
    physical: FixedBackwardPhysicalPlan

    @classmethod
    def from_problem_spec(
        cls,
        problem: FixedBackwardProblem,
        spec: FixedBackwardKernelSpec,
    ) -> "DerivedFixedBackwardState":
        from .fixed_grouped_mmq_bwd_validation import (
            validate_fixed_backward_solution,
        )

        validate_fixed_backward_solution(problem, spec)
        contract = FixedBackwardProblemContract.for_problem(problem)
        ordinary = DerivedBackwardState.from_contract_spec(
            contract.ordinary(problem.tokens), spec.compute
        )
        physical = derive_fixed_backward_physical_plan(
            derive_backward_physical_plan(ordinary)
        )
        return cls(problem, contract, spec, ordinary, physical)

    @property
    def grid(self) -> tuple[int, int, int]:
        geometry = self.spec.compute.geometry
        m_tiles = self.problem.tokens // geometry.macro_tile0
        n_tiles = self.problem.input_features // geometry.macro_tile1
        mapping = geometry.work_group_mapping
        m_groups = mapped_m_tile_count(m_tiles, mapping)
        if self.spec.work_group_order == "NMajor":
            if mapping == 1:
                return (n_tiles, m_tiles, self.problem.groups)
            return (
                mapped_grid_extent(n_tiles, mapping),
                m_groups,
                self.problem.groups,
            )
        if mapping == 1:
            return (
                m_tiles,
                n_tiles,
                self.problem.groups,
            )
        return (
            m_groups,
            mapped_grid_extent(n_tiles, mapping),
            self.problem.groups,
        )

    @property
    def expected_grad_output_shape(self) -> tuple[int, int, int]:
        return (self.problem.tokens, self.problem.groups, self.problem.output_features)

    @property
    def expected_packed_weight_shape(self) -> tuple[int, int, int]:
        return (
            self.problem.groups,
            self.problem.output_features,
            self.problem.packed_row_bytes,
        )

    @property
    def expected_grad_input_shape(self) -> tuple[int, int, int]:
        return (self.problem.tokens, self.problem.groups, self.problem.input_features)
