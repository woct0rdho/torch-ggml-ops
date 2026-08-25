"""Derived state for isolated grouped MMQ forward kernels."""

from dataclasses import dataclass

from .grouped_mmq_fwd_model import (
    GroupedActivationAddressing,
    GroupedDecodedPolicy,
    GroupedDecodePolicy,
    GroupedForwardProblem,
    GroupedIQ2SDecodePolicy,
    GroupedOperandSource,
    GroupedQ2DecodePolicy,
)
from .grouped_mmq_fwd_physical import (
    GroupedActivationStagingPlan,
    GroupedDecodedPhysicalPlan,
    GroupedDirectPhysicalPlan,
    GroupedIQ2SFullWeightPhysicalPlan,
    grouped_decoded_physical_plan,
    grouped_direct_physical_plan,
    grouped_iq2_s_full_weight_physical_plan,
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
from .schema import string as _string

_U32_MAX = 0xFFFFFFFF


def validate_grouped_forward_problem(problem: GroupedForwardProblem) -> None:
    quant_format = GROUPED_QUANT_FORMATS.get(problem.quant_data_type)
    assert quant_format is not None
    for value in (
        problem.aggregate_rows,
        problem.output_features,
        problem.input_features,
    ):
        assert 0 < value <= _U32_MAX
    assert problem.physical_experts == 256
    assert problem.max_route_entries == 256
    assert problem.output_features % 16 == 0
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
class GroupedForwardProblemContract:
    """Non-tunable grouped data, arithmetic, ISA, and destination contract."""

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
    signed_weight: bool
    signed_activation: bool
    wmma_clamp: bool
    weight_decode: str
    scale_arithmetic: str
    arithmetic_contract: str
    destination_type: str = "BFloat16"
    bf16_rounding: str = "RNEPreserveNaN"
    abi: str = "GroupedSerialRoutesV1"

    @classmethod
    def for_problem(
        cls, problem: GroupedForwardProblem
    ) -> "GroupedForwardProblemContract":
        quant_format = GROUPED_QUANT_FORMATS.get(problem.quant_data_type)
        assert quant_format is not None
        validate_grouped_forward_problem(problem)
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
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=quant_format.wmma_clamp,
            weight_decode=quant_format.weight_decode,
            scale_arithmetic=quant_format.scale_arithmetic,
            arithmetic_contract=quant_format.arithmetic_contract,
        )

    def problem(self, aggregate_rows: int) -> GroupedForwardProblem:
        return GroupedForwardProblem(
            self.quant_type,
            aggregate_rows,
            self.output_features,
            self.input_features,
            self.physical_experts,
            self.max_route_entries,
        )


@dataclass(frozen=True)
class GroupedActivationPolicy:
    addressing: GroupedActivationAddressing
    block_bytes: int


@dataclass(frozen=True)
class GroupedRowTileDispatchPolicy:
    body_rows: tuple[int, ...]

    @classmethod
    def from_geometry(
        cls,
        macro_tile0: int,
        tail_macro_tile0: int,
    ) -> "GroupedRowTileDispatchPolicy":
        assert not (
            macro_tile0 < tail_macro_tile0 or macro_tile0 % 16 or tail_macro_tile0 % 16
        )
        body_rows: list[int] = []
        rows = macro_tile0
        while rows > tail_macro_tile0:
            body_rows.append(rows)
            if rows % 2:
                break
            rows //= 2
        assert rows == tail_macro_tile0
        body_rows.append(rows)
        return cls(tuple(body_rows))

    @property
    def body_row_tiles(self) -> tuple[int, ...]:
        return tuple(rows // 16 for rows in self.body_rows)


def _grouped_decode_mapping(policy: GroupedDecodePolicy) -> dict[str, object]:
    if isinstance(policy, GroupedQ2DecodePolicy):
        return {
            "Kind": "ScaleMinimumBitfield",
            "MetadataConversion": policy.metadata_conversion,
            "UnrolledGroups": policy.unrolled_groups,
            "HipAssociation": policy.hip_association,
            "PartialLds": policy.partial_lds,
            "PreNegatedDm": policy.pre_negated_dm,
            "PairedPayloadWrites": policy.paired_payload_writes,
            "PairedMetadataWrites": policy.paired_metadata_writes,
            "DistributedProducer": policy.distributed_producer,
        }
    if isinstance(policy, GroupedIQ2SDecodePolicy):
        return {
            "Kind": "GridCodebook",
            "MetadataConversion": policy.metadata_conversion,
            "PayloadPrefetch": policy.payload_prefetch,
        }
    return {
        "Kind": "ScaleMinimum",
        "MetadataConversion": policy.metadata_conversion,
        "IndependentMetadataExtraction": policy.independent_metadata_extraction,
        "DeferMetadataReads": policy.defer_metadata_reads,
    }


def _mapping_for_grouped_decode(
    value: object,
    quant_type: str,
) -> GroupedDecodePolicy:
    if quant_type == "Q2_K":
        item = _mapping(
            value,
            "GroupedForwardKernelSpec.Decode",
            frozenset(
                {
                    "Kind",
                    "MetadataConversion",
                    "UnrolledGroups",
                    "HipAssociation",
                    "PartialLds",
                    "PreNegatedDm",
                    "PairedPayloadWrites",
                    "PairedMetadataWrites",
                    "DistributedProducer",
                }
            ),
        )
        if item["Kind"] != "ScaleMinimumBitfield":
            raise SchemaError("Q2_K grouped decode Kind must be Q2ScaleMinimum")
        return GroupedQ2DecodePolicy(
            metadata_conversion=_string(item, "MetadataConversion"),
            unrolled_groups=_boolean(item, "UnrolledGroups"),
            hip_association=_boolean(item, "HipAssociation"),
            partial_lds=_boolean(item, "PartialLds"),
            pre_negated_dm=_boolean(item, "PreNegatedDm"),
            paired_payload_writes=_boolean(item, "PairedPayloadWrites"),
            paired_metadata_writes=_boolean(item, "PairedMetadataWrites"),
            distributed_producer=_boolean(item, "DistributedProducer"),
        )
    if quant_type == "IQ2_S":
        item = _mapping(
            value,
            "GroupedForwardKernelSpec.Decode",
            frozenset({"Kind", "MetadataConversion", "PayloadPrefetch"}),
        )
        if item["Kind"] != "GridCodebook":
            raise SchemaError("IQ2_S grouped decode Kind must be IQ2SGrid")
        return GroupedIQ2SDecodePolicy(
            _string(item, "MetadataConversion"),
            _boolean(item, "PayloadPrefetch"),
        )
    if quant_type not in {"Q4_K", "Q5_K"}:
        raise SchemaError(f"unsupported grouped forward quant type {quant_type!r}")
    item = _mapping(
        value,
        "GroupedForwardKernelSpec.Decode",
        frozenset(
            {
                "Kind",
                "MetadataConversion",
                "IndependentMetadataExtraction",
                "DeferMetadataReads",
            }
        ),
    )
    if item["Kind"] != "ScaleMinimum":
        raise SchemaError("Q4_K/Q5_K grouped decode Kind must be ScaleMinimum")
    return GroupedDecodedPolicy(
        _string(item, "MetadataConversion"),
        _boolean(item, "IndependentMetadataExtraction"),
        _boolean(item, "DeferMetadataReads"),
    )


@dataclass(frozen=True)
class GroupedEpiloguePolicy:
    tiles_ahead: int
    dependency_width: int
    priority: int

    @classmethod
    def from_parameters(
        cls,
        tiles_ahead: int,
        dependency_width: int,
        priority: int,
    ) -> "GroupedEpiloguePolicy":
        return cls(
            tiles_ahead=tiles_ahead,
            dependency_width=dependency_width,
            priority=priority,
        )

    def scheduled_for(self, row_tiles: int) -> bool:
        return (
            self.tiles_ahead < row_tiles
            or self.dependency_width != 1
            or self.priority != 0
        )


@dataclass(frozen=True)
class GroupedGeometrySpec:
    work_group: tuple[int, int, int]
    matrix_instruction: tuple[int, ...]
    macro_tile: tuple[int, int]
    tail_macro_tile0: int
    depth_u: int


@dataclass(frozen=True)
class GroupedForwardKernelSpec:
    geometry: GroupedGeometrySpec
    operand_source: GroupedOperandSource
    activation: GroupedActivationPolicy
    decode: GroupedDecodePolicy
    epilogue: GroupedEpiloguePolicy

    @property
    def row_dispatch(self) -> GroupedRowTileDispatchPolicy:
        return GroupedRowTileDispatchPolicy.from_geometry(
            self.geometry.macro_tile[0], self.geometry.tail_macro_tile0
        )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        contract: GroupedForwardProblemContract,
    ) -> "GroupedForwardKernelSpec":
        item = _mapping(
            value,
            "GroupedForwardKernelSpec",
            frozenset({"Geometry", "Lowering", "Decode", "Epilogue"}),
        )
        geometry_item = _mapping(
            item["Geometry"],
            "GroupedForwardKernelSpec.Geometry",
            frozenset(
                {
                    "WorkGroup",
                    "MatrixInstruction",
                    "MacroTile",
                    "TailMacroTileM",
                    "DepthU",
                }
            ),
        )
        macro_tile = _integer_tuple(geometry_item, "MacroTile", 2)
        geometry = GroupedGeometrySpec(
            work_group=_integer_tuple(geometry_item, "WorkGroup", 3),
            matrix_instruction=_integer_tuple(geometry_item, "MatrixInstruction", 9),
            macro_tile=(macro_tile[0], macro_tile[1]),
            tail_macro_tile0=_integer(geometry_item, "TailMacroTileM"),
            depth_u=_integer(geometry_item, "DepthU"),
        )
        lowering = _mapping(
            item["Lowering"],
            "GroupedForwardKernelSpec.Lowering",
            frozenset({"OperandSource", "ActivationAddressing"}),
        )
        decode_item = _mapping_for_grouped_decode(item["Decode"], contract.quant_type)
        epilogue_item = _mapping(
            item["Epilogue"],
            "GroupedForwardKernelSpec.Epilogue",
            frozenset({"TilesAhead", "DependencyWidth", "Priority"}),
        )
        return cls(
            geometry=geometry,
            operand_source=_enum(lowering, "OperandSource", GroupedOperandSource),
            activation=GroupedActivationPolicy(
                _enum(
                    lowering,
                    "ActivationAddressing",
                    GroupedActivationAddressing,
                ),
                contract.activation_block_bytes,
            ),
            decode=decode_item,
            epilogue=GroupedEpiloguePolicy.from_parameters(
                _integer(epilogue_item, "TilesAhead"),
                _integer(epilogue_item, "DependencyWidth"),
                _integer(epilogue_item, "Priority"),
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "Geometry": {
                "WorkGroup": list(self.geometry.work_group),
                "MatrixInstruction": list(self.geometry.matrix_instruction),
                "MacroTile": list(self.geometry.macro_tile),
                "TailMacroTileM": self.geometry.tail_macro_tile0,
                "DepthU": self.geometry.depth_u,
            },
            "Lowering": {
                "OperandSource": self.operand_source.value,
                "ActivationAddressing": self.activation.addressing.value,
            },
            "Decode": _grouped_decode_mapping(self.decode),
            "Epilogue": {
                "TilesAhead": self.epilogue.tiles_ahead,
                "DependencyWidth": self.epilogue.dependency_width,
                "Priority": self.epilogue.priority,
            },
        }


def validate_grouped_forward_capability(
    problem: GroupedForwardProblem,
    kernel_spec: GroupedForwardKernelSpec,
) -> None:
    GroupedForwardProblemContract.for_problem(problem)
    geometry = kernel_spec.geometry
    activation = kernel_spec.activation
    epilogue = kernel_spec.epilogue
    assert geometry.depth_u == 32
    assert geometry.macro_tile[1] > 0
    assert problem.output_features % geometry.macro_tile[1] == 0

    source = kernel_spec.operand_source
    decode = kernel_spec.decode
    supported_sources = {
        "Q2_K": {GroupedOperandSource.GroupedDecodedWeightLds},
        "Q4_K": {
            GroupedOperandSource.GroupedDirectGlobal,
            GroupedOperandSource.GroupedDecodedWeightLds,
        },
        "Q5_K": {GroupedOperandSource.GroupedDecodedWeightLds},
        "IQ2_S": {GroupedOperandSource.GroupedIQ2SFullWeightLds},
    }.get(problem.quant_data_type, set())
    assert source in supported_sources

    if source is GroupedOperandSource.GroupedDirectGlobal:
        assert geometry.work_group == (32, 1, 1)
        assert geometry.matrix_instruction == (16, 16, 16, 1, 1, 1, 1, 1, 1)
        assert (
            geometry.macro_tile[0],
            geometry.tail_macro_tile0,
            geometry.macro_tile[1],
        ) == (16, 16, 16)
        assert activation.addressing is GroupedActivationAddressing.AggregateRows
        assert isinstance(decode, GroupedDecodedPolicy)
        assert not decode.independent_metadata_extraction
        assert not decode.defer_metadata_reads
        assert decode.metadata_conversion == "Float32ThenFloat16"
        assert (
            epilogue.tiles_ahead,
            epilogue.dependency_width,
            epilogue.priority,
        ) == (1, 1, 0)
        grouped_direct_physical_plan(activation.block_bytes).resources.admit(
            GFX1151_RESOURCE_CAPACITY
        )
        return

    if source is GroupedOperandSource.GroupedIQ2SFullWeightLds:
        assert geometry.work_group == (128, 1, 1)
        assert geometry.matrix_instruction == (16, 16, 16, 1, 1, 1, 4, 4, 1)
        assert (
            geometry.macro_tile[0],
            geometry.tail_macro_tile0,
            geometry.macro_tile[1],
        ) == (64, 64, 64)
        assert activation.addressing in {
            GroupedActivationAddressing.AggregateRowsTiled,
            GroupedActivationAddressing.AggregateRowsTiledLinear,
        }
        assert isinstance(decode, GroupedIQ2SDecodePolicy)
        assert decode.metadata_conversion == "Float16DUnsignedNibbleScaleToFloat32"
        assert (
            epilogue.tiles_ahead,
            epilogue.dependency_width,
            epilogue.priority,
        ) == (4, 1, 0)
        grouped_iq2_s_full_weight_physical_plan().resources.admit(
            GFX1151_RESOURCE_CAPACITY
        )
        return

    assert source is GroupedOperandSource.GroupedDecodedWeightLds
    row_tiles = geometry.macro_tile[0] // 16
    assert geometry.work_group == (128, 1, 1)
    assert geometry.matrix_instruction == (16, 16, 16, 1, 1, 1, 4, 4, 1)
    assert geometry.macro_tile[1] == 64
    assert activation.addressing is GroupedActivationAddressing.AggregateRowsTiled
    assert isinstance(
        decode,
        GroupedQ2DecodePolicy
        if problem.quant_data_type == "Q2_K"
        else GroupedDecodedPolicy,
    )
    assert decode.metadata_conversion == (
        "DirectQ2Float16NibblePairs"
        if problem.quant_data_type == "Q2_K"
        else "DirectFloat16Unsigned16"
    )
    assert epilogue.tiles_ahead in {1, row_tiles}
    assert epilogue.dependency_width in {1, 2, 4}
    assert epilogue.dependency_width <= row_tiles
    assert epilogue.priority in {0, 2}

    if problem.quant_data_type == "Q2_K":
        assert isinstance(decode, GroupedQ2DecodePolicy)
        max_rows = (
            128
            if decode.unrolled_groups and not decode.hip_association
            else 32
            if decode.partial_lds and not decode.distributed_producer
            else 64
        )
        rows = geometry.macro_tile[0]
        assert 32 <= rows <= max_rows
        assert rows & (rows - 1) == 0
    else:
        assert geometry.macro_tile[0] in {64, 128}

    activation_staging = GroupedActivationStagingPlan(
        addressing=kernel_spec.activation.addressing,
        block_bytes=kernel_spec.activation.block_bytes,
        participating_threads=(
            geometry.work_group[0] * geometry.work_group[1] * geometry.work_group[2]
        ),
    )
    physical = grouped_decoded_physical_plan(
        activation.block_bytes,
        geometry.macro_tile[0],
        problem.quant_data_type,
        activation_staging,
    )
    physical.resources.admit(GFX1151_RESOURCE_CAPACITY)


@dataclass(frozen=True)
class GroupedRouteState:
    physical_experts: int
    output_features: int
    aggregate_rows: int
    blocks_per_weight_row: int
    bytes_per_expert: int


@dataclass(frozen=True)
class DerivedGroupedForwardState:
    problem: GroupedForwardProblem
    semantics: QuantForwardSemantics
    physical_plan: (
        GroupedDirectPhysicalPlan
        | GroupedDecodedPhysicalPlan
        | GroupedIQ2SFullWeightPhysicalPlan
    )
    contract: GroupedForwardProblemContract
    kernel_spec: GroupedForwardKernelSpec
    route: GroupedRouteState
    problem_size: ProblemSize
    blocks_per_weight_row: int
    activation_blocks_per_row: int
    packed_weight_row_bytes: int
    bytes_per_expert: int
    activation_plane_stride_bytes: int
    activation_weight_block_stride_bytes: int
    output_column_tiles: int

    @classmethod
    def from_problem_spec(
        cls,
        problem: GroupedForwardProblem,
        kernel_spec: GroupedForwardKernelSpec,
    ) -> "DerivedGroupedForwardState":
        semantics = QuantForwardSemantics.for_quant_type(problem.quant_data_type)
        contract = GroupedForwardProblemContract.for_problem(problem)
        blocks_per_weight_row = problem.input_features // contract.block_values
        activation_blocks_per_row = problem.input_features // Q8_1_D4_BLOCK_VALUES
        packed_weight_row_bytes = (
            blocks_per_weight_row * contract.packed_weight_block_bytes
        )
        activation_plane_stride_bytes = (
            problem.aggregate_rows * kernel_spec.activation.block_bytes
        )
        if kernel_spec.operand_source is GroupedOperandSource.GroupedDirectGlobal:
            physical_plan: (
                GroupedDirectPhysicalPlan
                | GroupedDecodedPhysicalPlan
                | GroupedIQ2SFullWeightPhysicalPlan
            ) = grouped_direct_physical_plan(kernel_spec.activation.block_bytes)
        elif kernel_spec.operand_source is GroupedOperandSource.GroupedDecodedWeightLds:
            activation_staging = GroupedActivationStagingPlan(
                addressing=kernel_spec.activation.addressing,
                block_bytes=kernel_spec.activation.block_bytes,
                participating_threads=(
                    kernel_spec.geometry.work_group[0]
                    * kernel_spec.geometry.work_group[1]
                    * kernel_spec.geometry.work_group[2]
                ),
            )
            physical_plan = grouped_decoded_physical_plan(
                kernel_spec.activation.block_bytes,
                kernel_spec.geometry.macro_tile[0],
                problem.quant_data_type,
                activation_staging,
            )
        elif (
            kernel_spec.operand_source is GroupedOperandSource.GroupedIQ2SFullWeightLds
        ):
            physical_plan = grouped_iq2_s_full_weight_physical_plan()
        else:
            raise AssertionError
        return cls(
            problem=problem,
            semantics=semantics,
            physical_plan=physical_plan,
            contract=contract,
            kernel_spec=kernel_spec,
            route=GroupedRouteState(
                physical_experts=problem.physical_experts,
                output_features=problem.output_features,
                aggregate_rows=problem.aggregate_rows,
                blocks_per_weight_row=blocks_per_weight_row,
                bytes_per_expert=problem.output_features * packed_weight_row_bytes,
            ),
            problem_size=ProblemSize(
                problem.aggregate_rows,
                problem.output_features,
                problem.input_features,
            ),
            blocks_per_weight_row=blocks_per_weight_row,
            activation_blocks_per_row=activation_blocks_per_row,
            packed_weight_row_bytes=packed_weight_row_bytes,
            bytes_per_expert=problem.output_features * packed_weight_row_bytes,
            activation_plane_stride_bytes=activation_plane_stride_bytes,
            activation_weight_block_stride_bytes=2 * activation_plane_stride_bytes,
            output_column_tiles=problem.output_features
            // kernel_spec.geometry.macro_tile[1],
        )

    @property
    def expected_packed_weight_shape(self) -> tuple[int, int, int]:
        problem = self.problem
        return (
            problem.physical_experts,
            problem.output_features,
            self.packed_weight_row_bytes,
        )

    @property
    def expected_activation_shape(self) -> tuple[int, int, int]:
        return (
            self.activation_blocks_per_row,
            self.problem.aggregate_rows,
            self.kernel_spec.activation.block_bytes,
        )

    @property
    def expected_output_shape(self) -> tuple[int, int]:
        problem = self.problem
        return (problem.aggregate_rows, problem.output_features)

    def grid(self, route_entries: int) -> tuple[int, int, int]:
        assert not (
            route_entries <= 0 or route_entries > self.problem.max_route_entries
        )
        return (self.output_column_tiles, route_entries, 1)
