"""Derived authorities for research-only paired grouped kernels."""

from dataclasses import dataclass

from .grouped_mmq_fwd_model import (
    GroupedActivationAddressing,
    GroupedOutputStore,
)
from .grouped_mmq_fwd_pair_model import (
    GroupedForwardPairProblem,
    GroupedForwardPairSolution,
    GroupedForwardPairSolutionKey,
    GroupedPairDecodeSchedule,
    GroupedPairOperandSource,
    GroupedPairProjectionSchedule,
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
from .quant_formats import GROUPED_QUANT_FORMATS, Q8_1_D4_BLOCK_VALUES
from .schema import SchemaError
from .schema import boolean as _boolean
from .schema import enum_value as _enum
from .schema import integer as _integer
from .schema import integer_triple as _integer_triple
from .schema import integer_tuple as _integer_tuple
from .schema import strict_mapping as _mapping
from .schema import strict_mapping_optional as _mapping_optional
from .schema import string as _string

_U32_MAX = 0xFFFFFFFF


def grouped_forward_pair_problem_rejection_reason(
    problem: GroupedForwardPairProblem,
) -> str | None:
    """Return a formula-backed rejection for a paired routed problem."""
    quant_format = GROUPED_QUANT_FORMATS.get(problem.quant_data_type)
    if quant_format is None or _paired_mechanism(problem.quant_data_type) is None:
        return "unsupported paired quant type"
    for name, value in (
        ("aggregate rows", problem.aggregate_rows),
        ("output features", problem.output_features),
        ("input features", problem.input_features),
    ):
        if not 0 < value <= _U32_MAX:
            return f"paired {name} must fit in a positive u32"
    if problem.physical_experts != 256:
        return "paired forward requires 256 physical experts"
    if problem.max_route_entries != 256:
        return "paired forward requires at most 256 route entries"
    if problem.projection_count != 2:
        return "paired forward requires two projections"
    if problem.output_features % 64:
        return "paired output features must be divisible by the 64-column tile"
    if problem.input_features % quant_format.block_values:
        return "paired input features must contain complete quant blocks"
    if problem.input_features % Q8_1_D4_BLOCK_VALUES:
        return "paired input features must contain complete activation blocks"

    blocks_per_weight_row = problem.input_features // quant_format.block_values
    packed_weight_row_bytes = blocks_per_weight_row * quant_format.block_bytes
    bytes_per_expert = problem.output_features * packed_weight_row_bytes
    activation_blocks = problem.input_features // Q8_1_D4_BLOCK_VALUES
    activation_bytes = (
        activation_blocks * problem.aggregate_rows * quant_format.activation_block_bytes
    )
    output_bytes = problem.aggregate_rows * problem.output_features * 2
    if bytes_per_expert > _U32_MAX:
        return "paired packed bytes per expert must fit in a u32 offset"
    if activation_bytes > _U32_MAX:
        return "paired activation workspace must fit in a u32 offset"
    if output_bytes > _U32_MAX:
        return "paired output workspace must fit in a u32 offset"
    return None


def _paired_mechanism(
    quant_type: str,
) -> (
    tuple[
        GroupedPairOperandSource,
        GroupedPairDecodeSchedule,
        str,
        str,
    ]
    | None
):
    return {
        "IQ2_S": (
            GroupedPairOperandSource.IQ2SHalfWeightLds,
            GroupedPairDecodeSchedule.TwoLaneSelectedHalfPayloadPrefetch,
            "TwoLaneSelectedHalfIQ2SGridSigned",
            "Float16DUnsignedNibbleScaleToFloat32",
        ),
        "IQ2_XXS": (
            GroupedPairOperandSource.IQ2XXSHalfWeightLds,
            GroupedPairDecodeSchedule.TwoLaneSelectedHalfIQ2XXS,
            "TwoLaneSelectedHalfIQ2XXSGridParitySigned",
            "Float16DParitySignsOddScaleToFloat32",
        ),
        "Q3_K": (
            GroupedPairOperandSource.Q3KHalfWeightLds,
            GroupedPairDecodeSchedule.TwoLaneSelectedHalfQ3,
            "TwoLaneSelectedHalfQ3Signed",
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
    def rejection_reason(
        cls,
        problem: GroupedForwardPairProblem,
        solution: GroupedForwardPairSolution,
    ) -> str | None:
        quant_format = GROUPED_QUANT_FORMATS.get(problem.quant_data_type)
        mechanism = _paired_mechanism(problem.quant_data_type)
        if quant_format is None or mechanism is None:
            return f"unsupported paired quant type {problem.quant_data_type!r}"
        operand_source, decode_schedule, weight_decode, metadata_conversion = mechanism
        decode_schedules = {decode_schedule}
        if problem.quant_data_type == "Q3_K":
            decode_schedules.add(
                GroupedPairDecodeSchedule.TwoLaneSelectedHalfQ3VariableBFE
            )
        elif problem.quant_data_type == "IQ2_XXS":
            decode_schedules.add(
                GroupedPairDecodeSchedule.TwoLaneSelectedHalfIQ2XXSFusedSelector
            )
        checks = (
            (problem.projection_count == 2, "paired problem requires two projections"),
            (
                solution.projection_count == 2,
                "paired solution requires two projections",
            ),
            (
                solution.kernel_language == "Assembly",
                "paired kernel language must be Assembly",
            ),
            (solution.isa == (11, 5, 1), "paired ISA must be gfx1151"),
            (solution.wavefront_size == 32, "paired wavefront size must be 32"),
            (
                solution.work_group == (128, 1, 1),
                "paired workgroup must be 128 threads",
            ),
            (
                solution.matrix_instruction == (16, 16, 16, 1, 1, 1, 4, 4, 1),
                "paired matrix instruction mismatch",
            ),
            (solution.macro_tile0 in (64, 80), "paired row tile must be 64 or 80"),
            (solution.macro_tile1 == 64, "paired output tile must be 64"),
            (solution.depth_u == 128, "paired depth must be K128"),
            (
                solution.activation_layout == quant_format.activation_layout,
                "paired activation layout mismatch",
            ),
            (
                solution.activation_block_bytes == quant_format.activation_block_bytes,
                "paired activation bytes mismatch",
            ),
            (
                solution.packed_weight_block_bytes == quant_format.block_bytes,
                "paired weight bytes mismatch",
            ),
            (
                solution.operand_source is operand_source,
                "paired operand source mismatch",
            ),
            (
                solution.projection_schedule
                is GroupedPairProjectionSchedule.Interleaved,
                "paired projection schedule mismatch",
            ),
            (
                solution.metadata_schedule in decode_schedules,
                "paired decode schedule mismatch",
            ),
            (solution.weight_decode == weight_decode, "paired weight decode mismatch"),
            (
                solution.metadata_conversion == metadata_conversion,
                "paired metadata conversion mismatch",
            ),
            (
                solution.activation_addressing
                is GroupedActivationAddressing.AggregateRowsTiledLinear,
                "paired activation addressing mismatch",
            ),
            (
                solution.output_store is GroupedOutputStore.BFloat16RNEClause4Masked,
                "paired output store mismatch",
            ),
            (
                solution.scale_arithmetic == "Int32ScaleF32",
                "paired scale arithmetic mismatch",
            ),
            (
                solution.signed_weight and solution.signed_activation,
                "paired dot operands must be signed",
            ),
            (not solution.wmma_clamp, "paired signed WMMA must not clamp"),
        )
        rejection = next(
            (message for accepted, message in checks if not accepted), None
        )
        return rejection or grouped_forward_pair_problem_rejection_reason(problem)

    @classmethod
    def from_solution(
        cls,
        problem: GroupedForwardPairProblem,
        solution: GroupedForwardPairSolution,
    ) -> "GroupedForwardPairContract":
        quant_format = GROUPED_QUANT_FORMATS.get(problem.quant_data_type)
        if quant_format is None:
            raise ValueError(
                f"unsupported paired quant type {problem.quant_data_type!r}"
            )
        rejection = cls.rejection_reason(problem, solution)
        if rejection is not None:
            raise ValueError(rejection)
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
            kernel_language=solution.kernel_language,
            isa=solution.isa,
            wavefront_size=solution.wavefront_size,
            projection_count=solution.projection_count,
            arithmetic_contract=quant_format.arithmetic_contract,
            weight_decode=solution.weight_decode,
            metadata_conversion=solution.metadata_conversion,
            scale_arithmetic=solution.scale_arithmetic,
            signed_weight=solution.signed_weight,
            signed_activation=solution.signed_activation,
            wmma_clamp=solution.wmma_clamp,
        )

    @classmethod
    def from_mapping(cls, value: object) -> "GroupedForwardPairContract":
        item = _mapping(
            value,
            "GroupedForwardPairContract",
            frozenset(
                {
                    "quant_type",
                    "output_features",
                    "input_features",
                    "physical_experts",
                    "max_route_entries",
                    "projection_count",
                    "block_values",
                    "activation_layout",
                    "activation_block_bytes",
                    "packed_weight_block_bytes",
                    "kernel_language",
                    "isa",
                    "wavefront_size",
                    "arithmetic_contract",
                    "weight_decode",
                    "metadata_conversion",
                    "scale_arithmetic",
                    "signed_weight",
                    "signed_activation",
                    "wmma_clamp",
                    "destination_type",
                    "bf16_rounding",
                    "abi_family",
                }
            ),
        )
        quant_type = _string(item["quant_type"], "quant_type")
        quant_format = GROUPED_QUANT_FORMATS.get(quant_type)
        if quant_format is None:
            raise SchemaError(f"unsupported paired quant type {quant_type!r}")
        mechanism = _paired_mechanism(quant_type)
        if mechanism is None:
            raise SchemaError(f"unsupported paired quant type {quant_type!r}")
        contract = cls(
            quant_type=quant_type,
            output_features=_integer(item["output_features"], "output_features"),
            input_features=_integer(item["input_features"], "input_features"),
            physical_experts=_integer(item["physical_experts"], "physical_experts"),
            max_route_entries=_integer(item["max_route_entries"], "max_route_entries"),
            projection_count=_integer(item["projection_count"], "projection_count"),
            block_values=_integer(item["block_values"], "block_values"),
            activation_layout=_string(item["activation_layout"], "activation_layout"),
            activation_block_bytes=_integer(
                item["activation_block_bytes"], "activation_block_bytes"
            ),
            packed_weight_block_bytes=_integer(
                item["packed_weight_block_bytes"], "packed_weight_block_bytes"
            ),
            kernel_language=_string(item["kernel_language"], "kernel_language"),
            isa=_integer_triple(item["isa"], "isa"),
            wavefront_size=_integer(item["wavefront_size"], "wavefront_size"),
            arithmetic_contract=_string(
                item["arithmetic_contract"], "arithmetic_contract"
            ),
            weight_decode=_string(item["weight_decode"], "weight_decode"),
            metadata_conversion=_string(
                item["metadata_conversion"], "metadata_conversion"
            ),
            scale_arithmetic=_string(item["scale_arithmetic"], "scale_arithmetic"),
            signed_weight=_boolean(item["signed_weight"], "signed_weight"),
            signed_activation=_boolean(item["signed_activation"], "signed_activation"),
            wmma_clamp=_boolean(item["wmma_clamp"], "wmma_clamp"),
            destination_type=_string(item["destination_type"], "destination_type"),
            bf16_rounding=_string(item["bf16_rounding"], "bf16_rounding"),
            abi_family=_string(item["abi_family"], "abi_family"),
        )
        problem_rejection = grouped_forward_pair_problem_rejection_reason(
            contract.problem(1)
        )
        if problem_rejection is not None:
            raise SchemaError(problem_rejection)
        fixed = (
            contract.physical_experts == 256
            and contract.max_route_entries == 256
            and contract.projection_count == 2
            and contract.block_values == quant_format.block_values
            and contract.activation_layout == quant_format.activation_layout
            and contract.activation_block_bytes == quant_format.activation_block_bytes
            and contract.packed_weight_block_bytes == quant_format.block_bytes
            and contract.kernel_language == "Assembly"
            and contract.isa == (11, 5, 1)
            and contract.wavefront_size == 32
            and contract.arithmetic_contract == quant_format.arithmetic_contract
            and contract.weight_decode == mechanism[2]
            and contract.metadata_conversion == mechanism[3]
            and contract.scale_arithmetic == "Int32ScaleF32"
            and contract.signed_weight
            and contract.signed_activation
            and not contract.wmma_clamp
            and contract.destination_type == "BFloat16"
            and contract.bf16_rounding == "RNEPreserveNaN"
            and contract.abi_family == "GroupedPairV1"
        )
        if not fixed:
            raise SchemaError("GroupedForwardPairContract is not canonical")
        return contract

    def to_mapping(self) -> dict[str, object]:
        return {
            "quant_type": self.quant_type,
            "output_features": self.output_features,
            "input_features": self.input_features,
            "physical_experts": self.physical_experts,
            "max_route_entries": self.max_route_entries,
            "projection_count": self.projection_count,
            "block_values": self.block_values,
            "activation_layout": self.activation_layout,
            "activation_block_bytes": self.activation_block_bytes,
            "packed_weight_block_bytes": self.packed_weight_block_bytes,
            "kernel_language": self.kernel_language,
            "isa": list(self.isa),
            "wavefront_size": self.wavefront_size,
            "arithmetic_contract": self.arithmetic_contract,
            "weight_decode": self.weight_decode,
            "metadata_conversion": self.metadata_conversion,
            "scale_arithmetic": self.scale_arithmetic,
            "signed_weight": self.signed_weight,
            "signed_activation": self.signed_activation,
            "wmma_clamp": self.wmma_clamp,
            "destination_type": self.destination_type,
            "bf16_rounding": self.bf16_rounding,
            "abi_family": self.abi_family,
        }

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
    projection_schedule: GroupedPairProjectionSchedule
    route_ownership: GroupedPairRouteOwnership
    row_task_rows: int | None
    activation_addressing: GroupedActivationAddressing
    metadata_schedule: GroupedPairDecodeSchedule
    output_store: GroupedOutputStore

    @classmethod
    def from_solution(
        cls, solution: GroupedForwardPairSolution
    ) -> "GroupedForwardPairKernelSpec":
        return cls(
            geometry=GroupedForwardPairGeometry(
                solution.work_group,
                solution.matrix_instruction,
                (solution.macro_tile0, solution.macro_tile1),
                solution.depth_u,
            ),
            operand_source=solution.operand_source,
            projection_schedule=solution.projection_schedule,
            route_ownership=solution.route_ownership,
            row_task_rows=solution.row_task_rows,
            activation_addressing=solution.activation_addressing,
            metadata_schedule=solution.metadata_schedule,
            output_store=solution.output_store,
        )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        contract: GroupedForwardPairContract,
    ) -> "GroupedForwardPairKernelSpec":
        item = _mapping(
            value,
            "GroupedForwardPairKernelSpec",
            frozenset({"geometry", "lowering", "projection", "decode", "epilogue"}),
        )
        geometry_item = _mapping(
            item["geometry"],
            "GroupedForwardPairKernelSpec.geometry",
            frozenset({"work_group", "matrix_instruction", "macro_tile", "depth_u"}),
        )
        macro_tile = _integer_tuple(geometry_item["macro_tile"], "macro_tile", 2)
        geometry = GroupedForwardPairGeometry(
            _integer_triple(geometry_item["work_group"], "work_group"),
            _integer_tuple(
                geometry_item["matrix_instruction"], "matrix_instruction", 9
            ),
            (macro_tile[0], macro_tile[1]),
            _integer(geometry_item["depth_u"], "depth_u"),
        )
        lowering = _mapping_optional(
            item["lowering"],
            name="GroupedForwardPairKernelSpec.lowering",
            required=frozenset(
                {"operand_source", "route_ownership", "activation_addressing"}
            ),
            optional=frozenset({"row_task_rows"}),
        )
        projection = _mapping(
            item["projection"],
            "GroupedForwardPairKernelSpec.projection",
            frozenset({"schedule"}),
        )
        decode = _mapping(
            item["decode"],
            "GroupedForwardPairKernelSpec.decode",
            frozenset({"schedule"}),
        )
        epilogue = _mapping(
            item["epilogue"],
            "GroupedForwardPairKernelSpec.epilogue",
            frozenset({"output_store"}),
        )
        route_ownership = _enum(
            lowering["route_ownership"],
            "route_ownership",
            GroupedPairRouteOwnership,
        )
        row_task_rows = (
            _integer(lowering["row_task_rows"], "row_task_rows")
            if "row_task_rows" in lowering
            else None
        )
        if route_ownership is GroupedPairRouteOwnership.SerialRoutes:
            if row_task_rows is not None:
                raise SchemaError("serial paired routes cannot specify row_task_rows")
        elif row_task_rows is None:
            raise SchemaError("paired device row tasks require row_task_rows")
        return cls(
            geometry=geometry,
            operand_source=_enum(
                lowering["operand_source"], "operand_source", GroupedPairOperandSource
            ),
            projection_schedule=_enum(
                projection["schedule"],
                "ProjectionSchedule",
                GroupedPairProjectionSchedule,
            ),
            route_ownership=route_ownership,
            row_task_rows=row_task_rows,
            activation_addressing=_enum(
                lowering["activation_addressing"],
                "activation_addressing",
                GroupedActivationAddressing,
            ),
            metadata_schedule=_enum(
                decode["schedule"], "decode.schedule", GroupedPairDecodeSchedule
            ),
            output_store=_enum(
                epilogue["output_store"], "output_store", GroupedOutputStore
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "geometry": {
                "work_group": list(self.geometry.work_group),
                "matrix_instruction": list(self.geometry.matrix_instruction),
                "macro_tile": list(self.geometry.macro_tile),
                "depth_u": self.geometry.depth_u,
            },
            "lowering": {
                "operand_source": self.operand_source.value,
                "route_ownership": self.route_ownership.value,
                "activation_addressing": self.activation_addressing.value,
                **(
                    {"row_task_rows": self.row_task_rows}
                    if self.row_task_rows is not None
                    else {}
                ),
            },
            "projection": {"schedule": self.projection_schedule.value},
            "decode": {"schedule": self.metadata_schedule.value},
            "epilogue": {"output_store": self.output_store.value},
        }

    def to_legacy_hash_mapping(self) -> dict[str, object]:
        """Project numeric row-task sizing onto the frozen route identity."""
        mapping = self.to_mapping()
        projection = mapping["projection"]
        assert isinstance(projection, dict)
        projection["schedule"] = "K128Interleaved"
        if (
            self.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks
            and self.row_task_rows == 64
        ):
            lowering = mapping["lowering"]
            assert isinstance(lowering, dict)
            lowering["route_ownership"] = "DeviceRowTasks64"
            del lowering["row_task_rows"]
        return mapping

    def to_solution(
        self,
        contract: GroupedForwardPairContract,
    ) -> GroupedForwardPairSolution:
        route = {
            GroupedPairRouteOwnership.SerialRoutes: (
                "SerialGemmPair",
                "CumulativeOffsetsExpertIndices",
            ),
            GroupedPairRouteOwnership.DeviceRowTasks: (
                "RowTaskGemmPair",
                "DeviceRowTasks",
            ),
        }[self.route_ownership]
        return GroupedForwardPairSolution(
            kernel_language=contract.kernel_language,
            isa=contract.isa,
            wavefront_size=contract.wavefront_size,
            work_group=self.geometry.work_group,
            matrix_instruction=self.geometry.matrix_instruction,
            macro_tile0=self.geometry.macro_tile[0],
            macro_tile1=self.geometry.macro_tile[1],
            depth_u=self.geometry.depth_u,
            activation_layout=contract.activation_layout,
            activation_block_bytes=contract.activation_block_bytes,
            packed_weight_block_bytes=contract.packed_weight_block_bytes,
            operand_source=self.operand_source,
            projection_schedule=self.projection_schedule,
            weight_decode=contract.weight_decode,
            group_mapping=route[0],
            route_layout=route[1],
            row_task_rows=self.row_task_rows,
            activation_addressing=self.activation_addressing,
            metadata_conversion=contract.metadata_conversion,
            metadata_schedule=self.metadata_schedule,
            output_store=self.output_store,
            scale_arithmetic=contract.scale_arithmetic,
            projection_count=contract.projection_count,
            signed_weight=contract.signed_weight,
            signed_activation=contract.signed_activation,
            wmma_clamp=contract.wmma_clamp,
        )


def grouped_forward_pair_capability_rejection_reason(
    problem: GroupedForwardPairProblem,
    solution: GroupedForwardPairSolution,
) -> str | None:
    contract_rejection = GroupedForwardPairContract.rejection_reason(problem, solution)
    if contract_rejection is not None:
        return contract_rejection
    kernel_spec = GroupedForwardPairKernelSpec.from_solution(solution)
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
    if kernel_spec.route_ownership not in supported_ownership:
        return "paired route ownership is unavailable for the quant format"
    if solution.macro_tile0 == 80 and (
        problem.quant_data_type != "IQ2_XXS"
        or kernel_spec.metadata_schedule
        is not GroupedPairDecodeSchedule.TwoLaneSelectedHalfIQ2XXSFusedSelector
        or kernel_spec.route_ownership is not GroupedPairRouteOwnership.SerialRoutes
    ):
        return "paired J80 requires IQ2_XXS fused-selector serial-route ownership"
    if (
        kernel_spec.metadata_schedule
        is GroupedPairDecodeSchedule.TwoLaneSelectedHalfQ3VariableBFE
        and kernel_spec.route_ownership is not GroupedPairRouteOwnership.DeviceRowTasks
    ):
        return "paired Q3_K variable-BFE decode requires device row-task ownership"
    if (
        kernel_spec.metadata_schedule
        is GroupedPairDecodeSchedule.TwoLaneSelectedHalfIQ2XXSFusedSelector
        and kernel_spec.route_ownership is not GroupedPairRouteOwnership.SerialRoutes
    ):
        return "paired IQ2_XXS fused selector requires serial-route ownership"
    if kernel_spec.route_ownership is GroupedPairRouteOwnership.SerialRoutes:
        if kernel_spec.row_task_rows is not None:
            return "paired serial-route ownership cannot specify row-task rows"
    elif (
        kernel_spec.row_task_rows is None
        or kernel_spec.row_task_rows <= 0
        or kernel_spec.row_task_rows > kernel_spec.geometry.macro_tile[0]
    ):
        return "paired row-task rows must fit in the compute row tile"
    return None


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
    key: GroupedForwardPairSolutionKey
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
    def from_solution_key(
        cls, key: GroupedForwardPairSolutionKey
    ) -> "DerivedGroupedForwardPairState":
        problem = key.problem
        solution = key.solution
        contract = GroupedForwardPairContract.from_solution(problem, solution)
        kernel_spec = GroupedForwardPairKernelSpec.from_solution(solution)
        if problem.quant_data_type not in {"IQ2_S", "IQ2_XXS", "Q3_K"}:
            raise ValueError("paired research supports IQ2_S, IQ2_XXS, and Q3_K")
        physical = (
            grouped_iq2_s_pair_physical_plan(solution.route_ownership)
            if problem.quant_data_type == "IQ2_S"
            else (
                grouped_iq2_xxs_pair_physical_plan(
                    solution.route_ownership, solution.macro_tile0
                )
                if problem.quant_data_type == "IQ2_XXS"
                else grouped_q3_k_pair_physical_plan(solution.route_ownership)
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
            key=key,
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
            * solution.activation_block_bytes,
            output_column_tiles=problem.output_features // solution.macro_tile1,
        )

    @property
    def expected_packed_weight_shape(self) -> tuple[int, int, int]:
        return (
            self.key.problem.physical_experts,
            self.key.problem.output_features,
            self.packed_weight_row_bytes,
        )

    @property
    def expected_activation_shape(self) -> tuple[int, int, int]:
        return (
            self.activation_blocks_per_row,
            self.key.problem.aggregate_rows,
            self.key.solution.activation_block_bytes,
        )

    @property
    def expected_output_shape(self) -> tuple[int, int]:
        return (self.key.problem.aggregate_rows, self.key.problem.output_features)

    def grid(self, route_entries: int) -> tuple[int, int, int]:
        if (
            self.kernel_spec.route_ownership
            is not GroupedPairRouteOwnership.SerialRoutes
        ):
            raise ValueError("serial route grid requested for a row-task solution")
        if route_entries <= 0 or route_entries > self.key.problem.max_route_entries:
            raise ValueError("paired route entry count is outside the contract")
        return (self.output_column_tiles, route_entries, 1)

    def row_task_capacity(self, route_entries: int) -> int:
        if route_entries <= 0 or route_entries > self.key.problem.max_route_entries:
            raise ValueError("paired route entry count is outside the contract")
        row_tile = self.kernel_spec.row_task_rows
        if row_tile is None:
            raise ValueError("row-task capacity requested for a serial solution")
        return (
            self.key.problem.aggregate_rows + row_tile - 1
        ) // row_tile + route_entries

    def row_task_grid(self, route_entries: int) -> tuple[int, int, int]:
        if (
            self.kernel_spec.route_ownership
            is not GroupedPairRouteOwnership.DeviceRowTasks
        ):
            raise ValueError("row-task grid requested for a serial route solution")
        return (self.output_column_tiles, self.row_task_capacity(route_entries), 1)
