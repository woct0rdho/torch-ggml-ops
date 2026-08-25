"""Derived authorities for research-only paired grouped kernels."""

from dataclasses import dataclass

from .grouped_mmq_fwd_model import GroupedActivationAddressing
from .grouped_mmq_fwd_pair_model import (
    GroupedForwardPairProblem,
    GroupedPairDecodePolicy,
    GroupedPairFixedDecodePolicy,
    GroupedPairIQ2XXSDecodePolicy,
    GroupedPairOperandSource,
    GroupedPairProjectionInterleave,
    GroupedPairQ3DecodePolicy,
    GroupedPairRouteOwnership,
)
from .grouped_mmq_fwd_pair_physical import (
    GroupedIQ2SPairPhysicalPlan,
    GroupedIQ2XXSPairPhysicalPlan,
    GroupedQ3KPairPhysicalPlan,
    grouped_iq2_s_pair_physical_plan,
    grouped_iq2_xxs_pair_physical_plan,
    grouped_q3_k_pair_physical_plan,
)
from .mmq_fwd_spec import QuantForwardSemantics
from .model import ProblemSize
from .physical_resources import GFX1151_RESOURCE_CAPACITY
from .quant_formats import GROUPED_QUANT_FORMATS, Q8_1_D4_BLOCK_VALUES
from .schema import SchemaError
from .schema import boolean as _boolean
from .schema import enum_value as _enum
from .schema import integer as _integer
from .schema import integer_tuple as _integer_tuple
from .schema import strict_mapping as _mapping
from .schema import strict_mapping_optional as _mapping_optional

_U32_MAX = 0xFFFFFFFF


def validate_grouped_forward_pair_problem(problem: GroupedForwardPairProblem) -> None:
    quant_format = GROUPED_QUANT_FORMATS.get(problem.quant_data_type)
    assert quant_format is not None
    assert _paired_mechanism(problem.quant_data_type) is not None
    for value in (
        problem.aggregate_rows,
        problem.output_features,
        problem.input_features,
    ):
        assert 0 < value <= _U32_MAX
    assert problem.physical_experts == 256
    assert problem.max_route_entries == 256
    assert problem.projection_count == 2
    assert problem.output_features % 64 == 0
    assert problem.input_features % quant_format.block_values == 0
    assert problem.input_features % Q8_1_D4_BLOCK_VALUES == 0

    blocks_per_weight_row = problem.input_features // quant_format.block_values
    packed_weight_row_bytes = blocks_per_weight_row * quant_format.block_bytes
    bytes_per_expert = problem.output_features * packed_weight_row_bytes
    activation_blocks = problem.input_features // Q8_1_D4_BLOCK_VALUES
    activation_bytes = (
        activation_blocks * problem.aggregate_rows * quant_format.activation_block_bytes
    )
    output_bytes = problem.aggregate_rows * problem.output_features * 2
    assert bytes_per_expert <= _U32_MAX
    assert activation_bytes <= _U32_MAX
    assert output_bytes <= _U32_MAX


@dataclass(frozen=True)
class GroupedPairMechanism:
    operand_source: GroupedPairOperandSource
    decode_type: type[object]
    weight_decode: str
    metadata_conversion: str


def _paired_mechanism(quant_type: str) -> GroupedPairMechanism | None:
    return {
        "IQ2_S": GroupedPairMechanism(
            GroupedPairOperandSource.GridHalfWeightLds,
            GroupedPairFixedDecodePolicy,
            "TwoLaneSelectedHalfIQ2SGridSigned",
            "Float16DUnsignedNibbleScaleToFloat32",
        ),
        "IQ2_XXS": GroupedPairMechanism(
            GroupedPairOperandSource.ParityGridHalfWeightLds,
            GroupedPairIQ2XXSDecodePolicy,
            "TwoLaneSelectedHalfParityGridGridParitySigned",
            "Float16DParitySignsOddScaleToFloat32",
        ),
        "Q3_K": GroupedPairMechanism(
            GroupedPairOperandSource.SignedThreeBitHalfWeightLds,
            GroupedPairQ3DecodePolicy,
            "TwoLaneSelectedHalfSignedThreeBitSigned",
            "Float16DSignedSixBitScaleToFloat32",
        ),
    }.get(quant_type)


@dataclass(frozen=True)
class GroupedForwardPairContract:
    quant_type: str
    output_features: int
    input_features: int
    physical_experts: int
    max_route_entries: int
    block_values: int
    activation_layout: str
    activation_block_bytes: int
    packed_weight_block_bytes: int
    kernel_language: str
    isa: tuple[int, int, int]
    wavefront_size: int
    projection_count: int
    arithmetic_contract: str
    weight_decode: str
    metadata_conversion: str
    scale_arithmetic: str
    signed_weight: bool
    signed_activation: bool
    wmma_clamp: bool
    destination_type: str = "BFloat16"
    bf16_rounding: str = "RNEPreserveNaN"
    abi_family: str = "GroupedPairV1"

    @classmethod
    def for_problem(
        cls, problem: GroupedForwardPairProblem
    ) -> "GroupedForwardPairContract":
        quant_format = GROUPED_QUANT_FORMATS.get(problem.quant_data_type)
        mechanism = _paired_mechanism(problem.quant_data_type)
        assert quant_format is not None
        assert mechanism is not None
        validate_grouped_forward_pair_problem(problem)
        return cls(
            quant_type=problem.quant_data_type,
            output_features=problem.output_features,
            input_features=problem.input_features,
            physical_experts=problem.physical_experts,
            max_route_entries=problem.max_route_entries,
            block_values=quant_format.block_values,
            activation_layout=quant_format.activation_layout,
            activation_block_bytes=quant_format.activation_block_bytes,
            packed_weight_block_bytes=quant_format.block_bytes,
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            projection_count=problem.projection_count,
            arithmetic_contract=quant_format.arithmetic_contract,
            weight_decode=mechanism.weight_decode,
            metadata_conversion=mechanism.metadata_conversion,
            scale_arithmetic="Int32ScaleF32",
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=False,
        )

    def problem(self, aggregate_rows: int) -> GroupedForwardPairProblem:
        return GroupedForwardPairProblem(
            self.quant_type,
            aggregate_rows,
            self.output_features,
            self.input_features,
            self.physical_experts,
            self.max_route_entries,
            self.projection_count,
        )


@dataclass(frozen=True)
class GroupedForwardPairGeometry:
    work_group: tuple[int, int, int]
    matrix_instruction: tuple[int, ...]
    macro_tile: tuple[int, int]
    depth_u: int


@dataclass(frozen=True)
class GroupedForwardPairKernelSpec:
    geometry: GroupedForwardPairGeometry
    operand_source: GroupedPairOperandSource
    projection_interleave: GroupedPairProjectionInterleave
    route_ownership: GroupedPairRouteOwnership
    row_task_rows: int | None
    activation_addressing: GroupedActivationAddressing
    decode_policy: GroupedPairDecodePolicy

    @classmethod
    def from_mapping(
        cls,
        value: object,
        contract: GroupedForwardPairContract,
    ) -> "GroupedForwardPairKernelSpec":
        keys = {"Geometry", "Lowering", "Projection"}
        if contract.quant_type != "IQ2_S":
            keys.add("Decode")
        item = _mapping(
            value,
            "GroupedForwardPairKernelSpec",
            frozenset(keys),
        )
        geometry_item = _mapping(
            item["Geometry"],
            "GroupedForwardPairKernelSpec.Geometry",
            frozenset({"WorkGroup", "MatrixInstruction", "MacroTile", "DepthU"}),
        )
        macro_tile = _integer_tuple(geometry_item, "MacroTile", 2)
        geometry = GroupedForwardPairGeometry(
            _integer_tuple(geometry_item, "WorkGroup", 3),
            _integer_tuple(geometry_item, "MatrixInstruction", 9),
            (macro_tile[0], macro_tile[1]),
            _integer(geometry_item, "DepthU"),
        )
        lowering = _mapping_optional(
            item["Lowering"],
            name="GroupedForwardPairKernelSpec.Lowering",
            required=frozenset(
                {"OperandSource", "RouteOwnership", "ActivationAddressing"}
            ),
            optional=frozenset({"RowTaskRows"}),
        )
        projection = _mapping(
            item["Projection"],
            "GroupedForwardPairKernelSpec.Projection",
            frozenset({"Interleave"}),
        )
        if contract.quant_type == "IQ2_S":
            decode_policy: GroupedPairDecodePolicy = GroupedPairFixedDecodePolicy()
        else:
            decode_key, decode_type = {
                "IQ2_XXS": ("FusedGridSelector", GroupedPairIQ2XXSDecodePolicy),
                "Q3_K": ("VariableBitfieldExtraction", GroupedPairQ3DecodePolicy),
            }[contract.quant_type]
            decode = _mapping(
                item["Decode"],
                "GroupedForwardPairKernelSpec.Decode",
                frozenset({decode_key}),
            )
            decode_policy = decode_type(_boolean(decode, decode_key))
        route_ownership = _enum(
            lowering,
            "RouteOwnership",
            GroupedPairRouteOwnership,
        )
        row_task_rows = (
            _integer(lowering, "RowTaskRows") if "RowTaskRows" in lowering else None
        )
        if route_ownership is GroupedPairRouteOwnership.SerialRoutes:
            if row_task_rows is not None:
                raise SchemaError("serial paired routes cannot specify row_task_rows")
        elif row_task_rows is None:
            raise SchemaError("paired device row tasks require row_task_rows")
        return cls(
            geometry=geometry,
            operand_source=_enum(lowering, "OperandSource", GroupedPairOperandSource),
            projection_interleave=_enum(
                projection,
                "Interleave",
                GroupedPairProjectionInterleave,
            ),
            route_ownership=route_ownership,
            row_task_rows=row_task_rows,
            activation_addressing=_enum(
                lowering,
                "ActivationAddressing",
                GroupedActivationAddressing,
            ),
            decode_policy=decode_policy,
        )

    def to_mapping(self) -> dict[str, object]:
        decode = (
            {"FusedGridSelector": self.decode_policy.fused_grid_selector}
            if isinstance(self.decode_policy, GroupedPairIQ2XXSDecodePolicy)
            else {
                "VariableBitfieldExtraction": (
                    self.decode_policy.variable_bitfield_extraction
                )
            }
            if isinstance(self.decode_policy, GroupedPairQ3DecodePolicy)
            else None
        )
        return {
            "Geometry": {
                "WorkGroup": list(self.geometry.work_group),
                "MatrixInstruction": list(self.geometry.matrix_instruction),
                "MacroTile": list(self.geometry.macro_tile),
                "DepthU": self.geometry.depth_u,
            },
            "Lowering": {
                "OperandSource": self.operand_source.value,
                "RouteOwnership": self.route_ownership.value,
                "ActivationAddressing": self.activation_addressing.value,
                **(
                    {"RowTaskRows": self.row_task_rows}
                    if self.row_task_rows is not None
                    else {}
                ),
            },
            "Projection": {"Interleave": self.projection_interleave.value},
            **({"Decode": decode} if decode is not None else {}),
        }


def validate_grouped_forward_pair_capability(
    problem: GroupedForwardPairProblem,
    kernel_spec: GroupedForwardPairKernelSpec,
) -> None:
    contract = GroupedForwardPairContract.for_problem(problem)
    mechanism = _paired_mechanism(problem.quant_data_type)
    assert mechanism is not None
    geometry = kernel_spec.geometry
    assert contract.projection_count == 2
    assert geometry.work_group == (128, 1, 1)
    assert geometry.matrix_instruction == (16, 16, 16, 1, 1, 1, 4, 4, 1)
    assert geometry.macro_tile[0] in (64, 80)
    assert geometry.macro_tile[1] == 64
    assert geometry.depth_u == 128
    assert (
        kernel_spec.projection_interleave is GroupedPairProjectionInterleave.Interleaved
    )
    assert (
        kernel_spec.activation_addressing
        is GroupedActivationAddressing.AggregateRowsTiledLinear
    )
    assert kernel_spec.operand_source is mechanism.operand_source
    assert isinstance(kernel_spec.decode_policy, mechanism.decode_type)
    supported_ownership = {
        "IQ2_S": {
            GroupedPairRouteOwnership.SerialRoutes,
            GroupedPairRouteOwnership.DeviceRowTasks,
        },
        "IQ2_XXS": {GroupedPairRouteOwnership.SerialRoutes},
        "Q3_K": {
            GroupedPairRouteOwnership.SerialRoutes,
            GroupedPairRouteOwnership.DeviceRowTasks,
        },
    }.get(problem.quant_data_type, set())
    assert kernel_spec.route_ownership in supported_ownership
    decode = kernel_spec.decode_policy
    if geometry.macro_tile[0] == 80:
        assert isinstance(decode, GroupedPairIQ2XXSDecodePolicy)
        assert decode.fused_grid_selector
        assert kernel_spec.route_ownership is GroupedPairRouteOwnership.SerialRoutes
    if (
        isinstance(decode, GroupedPairQ3DecodePolicy)
        and decode.variable_bitfield_extraction
    ):
        assert kernel_spec.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks
    if isinstance(decode, GroupedPairIQ2XXSDecodePolicy) and decode.fused_grid_selector:
        assert kernel_spec.route_ownership is GroupedPairRouteOwnership.SerialRoutes
    if kernel_spec.route_ownership is GroupedPairRouteOwnership.SerialRoutes:
        assert kernel_spec.row_task_rows is None
    else:
        assert kernel_spec.row_task_rows is not None
        assert kernel_spec.row_task_rows > 0
        assert kernel_spec.row_task_rows <= kernel_spec.geometry.macro_tile[0]

    physical = (
        grouped_iq2_s_pair_physical_plan(kernel_spec.route_ownership)
        if problem.quant_data_type == "IQ2_S"
        else (
            grouped_iq2_xxs_pair_physical_plan(
                kernel_spec.route_ownership, kernel_spec.geometry.macro_tile[0]
            )
            if problem.quant_data_type == "IQ2_XXS"
            else grouped_q3_k_pair_physical_plan(kernel_spec.route_ownership)
        )
    )
    physical.resources.admit(GFX1151_RESOURCE_CAPACITY)


@dataclass(frozen=True)
class GroupedForwardPairRouteState:
    physical_experts: int
    output_features: int
    aggregate_rows: int
    blocks_per_weight_row: int
    bytes_per_expert: int
    projection_count: int
    row_task_rows: int | None


@dataclass(frozen=True)
class DerivedGroupedForwardPairState:
    problem: GroupedForwardPairProblem
    semantics: QuantForwardSemantics
    physical_plan: (
        GroupedIQ2SPairPhysicalPlan
        | GroupedIQ2XXSPairPhysicalPlan
        | GroupedQ3KPairPhysicalPlan
    )
    contract: GroupedForwardPairContract
    kernel_spec: GroupedForwardPairKernelSpec
    route: GroupedForwardPairRouteState
    problem_size: ProblemSize
    blocks_per_weight_row: int
    activation_blocks_per_row: int
    packed_weight_row_bytes: int
    bytes_per_expert: int
    activation_plane_stride_bytes: int
    output_column_tiles: int

    @classmethod
    def from_problem_spec(
        cls,
        problem: GroupedForwardPairProblem,
        kernel_spec: GroupedForwardPairKernelSpec,
    ) -> "DerivedGroupedForwardPairState":
        contract = GroupedForwardPairContract.for_problem(problem)
        assert problem.quant_data_type in {"IQ2_S", "IQ2_XXS", "Q3_K"}
        physical = (
            grouped_iq2_s_pair_physical_plan(kernel_spec.route_ownership)
            if problem.quant_data_type == "IQ2_S"
            else (
                grouped_iq2_xxs_pair_physical_plan(
                    kernel_spec.route_ownership, kernel_spec.geometry.macro_tile[0]
                )
                if problem.quant_data_type == "IQ2_XXS"
                else grouped_q3_k_pair_physical_plan(kernel_spec.route_ownership)
            )
        )
        semantics = QuantForwardSemantics.for_quant_type(problem.quant_data_type)
        blocks_per_weight_row = problem.input_features // contract.block_values
        activation_blocks_per_row = problem.input_features // Q8_1_D4_BLOCK_VALUES
        packed_weight_row_bytes = (
            blocks_per_weight_row * contract.packed_weight_block_bytes
        )
        bytes_per_expert = problem.output_features * packed_weight_row_bytes
        return cls(
            problem=problem,
            semantics=semantics,
            physical_plan=physical,
            contract=contract,
            kernel_spec=kernel_spec,
            route=GroupedForwardPairRouteState(
                physical_experts=problem.physical_experts,
                output_features=problem.output_features,
                aggregate_rows=problem.aggregate_rows,
                blocks_per_weight_row=blocks_per_weight_row,
                bytes_per_expert=bytes_per_expert,
                projection_count=problem.projection_count,
                row_task_rows=kernel_spec.row_task_rows,
            ),
            problem_size=ProblemSize(
                problem.aggregate_rows, problem.output_features, problem.input_features
            ),
            blocks_per_weight_row=blocks_per_weight_row,
            activation_blocks_per_row=activation_blocks_per_row,
            packed_weight_row_bytes=packed_weight_row_bytes,
            bytes_per_expert=bytes_per_expert,
            activation_plane_stride_bytes=problem.aggregate_rows
            * contract.activation_block_bytes,
            output_column_tiles=problem.output_features
            // kernel_spec.geometry.macro_tile[1],
        )

    @property
    def expected_packed_weight_shape(self) -> tuple[int, int, int]:
        return (
            self.problem.physical_experts,
            self.problem.output_features,
            self.packed_weight_row_bytes,
        )

    @property
    def expected_activation_shape(self) -> tuple[int, int, int]:
        return (
            self.activation_blocks_per_row,
            self.problem.aggregate_rows,
            self.contract.activation_block_bytes,
        )

    @property
    def expected_output_shape(self) -> tuple[int, int]:
        return (self.problem.aggregate_rows, self.problem.output_features)

    def grid(self, route_entries: int) -> tuple[int, int, int]:
        assert (
            self.kernel_spec.route_ownership is GroupedPairRouteOwnership.SerialRoutes
        )
        assert not (
            route_entries <= 0 or route_entries > self.problem.max_route_entries
        )
        return (self.output_column_tiles, route_entries, 1)

    def row_task_capacity(self, route_entries: int) -> int:
        assert not (
            route_entries <= 0 or route_entries > self.problem.max_route_entries
        )
        row_tile = self.kernel_spec.row_task_rows
        assert row_tile is not None
        return (self.problem.aggregate_rows + row_tile - 1) // row_tile + route_entries

    def row_task_grid(self, route_entries: int) -> tuple[int, int, int]:
        assert (
            self.kernel_spec.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks
        )
        return (self.output_column_tiles, self.row_task_capacity(route_entries), 1)
