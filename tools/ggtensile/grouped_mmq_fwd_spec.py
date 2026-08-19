"""Derived state for isolated grouped MMQ forward kernels."""

from dataclasses import dataclass

from .grouped_mmq_fwd_model import (
    GroupedActivationAddressing,
    GroupedForwardProblem,
    GroupedForwardSolution,
    GroupedForwardSolutionKey,
    GroupedMetadataSchedule,
    GroupedOperandSource,
    GroupedOutputStore,
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
from .quant_formats import GROUPED_QUANT_FORMATS, Q8_1_D4_BLOCK_VALUES
from .schema import (
    SchemaError,
)
from .schema import (
    boolean as _boolean,
)
from .schema import (
    enum_value as _enum,
)
from .schema import (
    integer as _integer,
)
from .schema import (
    integer_triple as _integer_triple,
)
from .schema import (
    integer_tuple as _integer_tuple,
)
from .schema import (
    strict_mapping as _mapping,
)
from .schema import (
    string as _string,
)

_U32_MAX = 0xFFFFFFFF


def grouped_forward_problem_rejection_reason(
    problem: GroupedForwardProblem,
) -> str | None:
    """Return a formula-backed rejection for the routed forward problem."""
    quant_format = GROUPED_QUANT_FORMATS.get(problem.quant_data_type)
    if quant_format is None:
        return "unsupported grouped forward quant type"
    for name, value in (
        ("aggregate rows", problem.aggregate_rows),
        ("output features", problem.output_features),
        ("input features", problem.input_features),
    ):
        if not 0 < value <= _U32_MAX:
            return f"grouped {name} must fit in a positive u32"
    if problem.physical_experts != 256:
        return "grouped forward requires 256 physical experts"
    if problem.max_route_entries != 256:
        return "grouped forward requires at most 256 route entries"
    if problem.output_features % 16:
        return "grouped output features must contain complete WMMA columns"
    if problem.input_features % quant_format.block_values:
        return "grouped input features must contain complete quant blocks"
    if problem.input_features % Q8_1_D4_BLOCK_VALUES:
        return "grouped input features must contain complete activation blocks"

    blocks_per_weight_row = problem.input_features // quant_format.block_values
    packed_weight_row_bytes = blocks_per_weight_row * quant_format.block_bytes
    bytes_per_expert = problem.output_features * packed_weight_row_bytes
    activation_blocks = problem.input_features // Q8_1_D4_BLOCK_VALUES
    activation_bytes = (
        activation_blocks * problem.aggregate_rows * quant_format.activation_block_bytes
    )
    output_bytes = problem.aggregate_rows * problem.output_features * 2
    if bytes_per_expert > _U32_MAX:
        return "grouped packed bytes per expert must fit in a u32 offset"
    if activation_bytes > _U32_MAX:
        return "grouped activation workspace must fit in a u32 offset"
    if output_bytes > _U32_MAX:
        return "grouped output workspace must fit in a u32 offset"
    return None


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
    def rejection_reason(
        cls,
        problem: GroupedForwardProblem,
        solution: GroupedForwardSolution,
    ) -> str | None:
        quant_format = GROUPED_QUANT_FORMATS.get(problem.quant_data_type)
        if quant_format is None:
            return f"unsupported grouped forward quant type {problem.quant_data_type!r}"
        checks = (
            (
                solution.activation_layout == quant_format.activation_layout,
                "grouped activation layout does not match the quant-format contract",
            ),
            (
                solution.activation_block_bytes == quant_format.activation_block_bytes,
                "grouped activation block bytes do not match the quant-format contract",
            ),
            (
                solution.packed_weight_block_bytes == quant_format.block_bytes,
                "grouped packed-weight bytes do not match the quant-format contract",
            ),
            (
                solution.kernel_language == "Assembly",
                "grouped kernel language must be Assembly",
            ),
            (solution.isa == (11, 5, 1), "grouped ISA must be gfx1151"),
            (solution.wavefront_size == 32, "grouped wavefront size must be 32"),
            (
                solution.signed_weight and solution.signed_activation,
                "grouped dot operands must be signed",
            ),
            (
                solution.wmma_clamp == quant_format.wmma_clamp,
                "grouped WMMA clamp does not match the arithmetic contract",
            ),
            (
                solution.weight_decode == quant_format.weight_decode,
                "grouped weight decode does not match the quant-format contract",
            ),
            (
                solution.scale_arithmetic == quant_format.scale_arithmetic,
                "grouped scale arithmetic does not match the quant-format contract",
            ),
            (
                solution.group_mapping == "SerialGemm",
                "grouped routing requires serial GEMM ownership",
            ),
            (
                solution.route_layout == "CumulativeOffsetsExpertIndices",
                "grouped routing requires cumulative offsets and expert indices",
            ),
        )
        rejection = next(
            (message for accepted, message in checks if not accepted), None
        )
        return rejection or grouped_forward_problem_rejection_reason(problem)

    @classmethod
    def from_solution(
        cls,
        problem: GroupedForwardProblem,
        solution: GroupedForwardSolution,
    ) -> "GroupedForwardProblemContract":
        quant_format = GROUPED_QUANT_FORMATS.get(problem.quant_data_type)
        if quant_format is None:
            raise ValueError(
                f"unsupported grouped forward quant type {problem.quant_data_type!r}"
            ) from None
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
            signed_weight=solution.signed_weight,
            signed_activation=solution.signed_activation,
            wmma_clamp=solution.wmma_clamp,
            weight_decode=solution.weight_decode,
            scale_arithmetic=solution.scale_arithmetic,
            arithmetic_contract=quant_format.arithmetic_contract,
        )

    @classmethod
    def from_mapping(cls, value: object) -> "GroupedForwardProblemContract":
        item = _mapping(
            value,
            "GroupedForwardProblemContract",
            frozenset(
                {
                    "quant_type",
                    "output_features",
                    "input_features",
                    "physical_experts",
                    "max_route_entries",
                    "block_values",
                    "activation_layout",
                    "activation_block_bytes",
                    "packed_weight_block_bytes",
                    "kernel_language",
                    "isa",
                    "wavefront_size",
                    "signed_weight",
                    "signed_activation",
                    "wmma_clamp",
                    "weight_decode",
                    "scale_arithmetic",
                    "arithmetic_contract",
                    "destination_type",
                    "bf16_rounding",
                    "abi",
                }
            ),
        )
        quant_type = _string(item["quant_type"], "quant_type")
        quant_format = GROUPED_QUANT_FORMATS.get(quant_type)
        if quant_format is None:
            raise SchemaError(
                f"unsupported grouped forward quant type {quant_type!r}"
            ) from None
        contract = cls(
            quant_type=quant_type,
            output_features=_integer(item["output_features"], "output_features"),
            input_features=_integer(item["input_features"], "input_features"),
            physical_experts=_integer(item["physical_experts"], "physical_experts"),
            max_route_entries=_integer(item["max_route_entries"], "max_route_entries"),
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
            signed_weight=_boolean(item["signed_weight"], "signed_weight"),
            signed_activation=_boolean(item["signed_activation"], "signed_activation"),
            wmma_clamp=_boolean(item["wmma_clamp"], "wmma_clamp"),
            weight_decode=_string(item["weight_decode"], "weight_decode"),
            scale_arithmetic=_string(item["scale_arithmetic"], "scale_arithmetic"),
            arithmetic_contract=_string(
                item["arithmetic_contract"], "arithmetic_contract"
            ),
            destination_type=_string(item["destination_type"], "destination_type"),
            bf16_rounding=_string(item["bf16_rounding"], "bf16_rounding"),
            abi=_string(item["abi"], "abi"),
        )
        problem_rejection = grouped_forward_problem_rejection_reason(
            contract.problem(1)
        )
        if problem_rejection is not None:
            raise SchemaError(problem_rejection)
        fixed = (
            contract.physical_experts == 256
            and contract.max_route_entries == 256
            and contract.block_values == quant_format.block_values
            and contract.activation_layout == quant_format.activation_layout
            and contract.activation_block_bytes == quant_format.activation_block_bytes
            and contract.packed_weight_block_bytes == quant_format.block_bytes
            and contract.kernel_language == "Assembly"
            and contract.isa == (11, 5, 1)
            and contract.wavefront_size == 32
            and contract.signed_weight
            and contract.signed_activation
            and contract.wmma_clamp == quant_format.wmma_clamp
            and contract.weight_decode == quant_format.weight_decode
            and contract.scale_arithmetic == quant_format.scale_arithmetic
            and contract.arithmetic_contract == quant_format.arithmetic_contract
            and contract.destination_type == "BFloat16"
            and contract.bf16_rounding == "RNEPreserveNaN"
            and contract.abi == "GroupedSerialRoutesV1"
        )
        if not fixed:
            raise SchemaError("GroupedForwardProblemContract is not canonical")
        return contract

    def to_mapping(self) -> dict[str, object]:
        return {
            "quant_type": self.quant_type,
            "output_features": self.output_features,
            "input_features": self.input_features,
            "physical_experts": self.physical_experts,
            "max_route_entries": self.max_route_entries,
            "block_values": self.block_values,
            "activation_layout": self.activation_layout,
            "activation_block_bytes": self.activation_block_bytes,
            "packed_weight_block_bytes": self.packed_weight_block_bytes,
            "kernel_language": self.kernel_language,
            "isa": list(self.isa),
            "wavefront_size": self.wavefront_size,
            "signed_weight": self.signed_weight,
            "signed_activation": self.signed_activation,
            "wmma_clamp": self.wmma_clamp,
            "weight_decode": self.weight_decode,
            "scale_arithmetic": self.scale_arithmetic,
            "arithmetic_contract": self.arithmetic_contract,
            "destination_type": self.destination_type,
            "bf16_rounding": self.bf16_rounding,
            "abi": self.abi,
        }

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
        if macro_tile0 < tail_macro_tile0 or macro_tile0 % 16 or tail_macro_tile0 % 16:
            raise ValueError(
                "grouped row dispatch requires ordered 16-row macro/tail ownership"
            )
        body_rows: list[int] = []
        rows = macro_tile0
        while rows > tail_macro_tile0:
            body_rows.append(rows)
            if rows % 2:
                break
            rows //= 2
        if rows != tail_macro_tile0:
            raise ValueError(
                "grouped row dispatch requires a halving path to the tail body"
            )
        body_rows.append(rows)
        return cls(tuple(body_rows))

    @property
    def body_row_tiles(self) -> tuple[int, ...]:
        return tuple(rows // 16 for rows in self.body_rows)


@dataclass(frozen=True)
class GroupedDecodedSchedulePolicy:
    schedule: GroupedMetadataSchedule
    metadata_conversion: str

    @property
    def independent_metadata_extraction(self) -> bool:
        return (
            self.schedule
            is GroupedMetadataSchedule.IndependentExtractionMetadataAfterLowWmma
        )

    @property
    def defer_metadata_reads(self) -> bool:
        return (
            self.schedule
            is GroupedMetadataSchedule.IndependentExtractionMetadataAfterLowWmma
        )


@dataclass(frozen=True)
class GroupedQ2SchedulePolicy:
    metadata_conversion: str
    unrolled_groups: bool
    hip_association: bool
    partial_lds: bool
    pre_negated_dm: bool
    paired_payload_writes: bool
    paired_metadata_writes: bool
    distributed_producer: bool

    def __post_init__(self) -> None:
        dependencies = (
            (
                self.hip_association,
                self.unrolled_groups,
                "HIP association requires unrolling",
            ),
            (
                self.partial_lds,
                self.hip_association,
                "partial LDS requires HIP association",
            ),
            (
                self.pre_negated_dm,
                self.partial_lds,
                "pre-negated dm requires partial LDS",
            ),
            (
                self.paired_payload_writes,
                self.pre_negated_dm,
                "paired payload writes require pre-negated dm",
            ),
            (
                self.paired_metadata_writes,
                self.paired_payload_writes,
                "paired metadata writes require paired payload writes",
            ),
            (
                self.distributed_producer,
                self.paired_metadata_writes,
                "distributed production requires paired metadata writes",
            ),
        )
        rejection = next(
            (
                message
                for enabled, required, message in dependencies
                if enabled and not required
            ),
            None,
        )
        if rejection is not None:
            raise ValueError(rejection)

    @property
    def independent_metadata_extraction(self) -> bool:
        return False

    @property
    def defer_metadata_reads(self) -> bool:
        return False


@dataclass(frozen=True)
class GroupedIQ2SSchedulePolicy:
    schedule: GroupedMetadataSchedule
    metadata_conversion: str

    @property
    def payload_prefetch(self) -> bool:
        return self.schedule is GroupedMetadataSchedule.IQ2SPayloadPrefetch

    @property
    def independent_metadata_extraction(self) -> bool:
        return False

    @property
    def defer_metadata_reads(self) -> bool:
        return False


GroupedDecodePolicy = (
    GroupedDecodedSchedulePolicy | GroupedQ2SchedulePolicy | GroupedIQ2SSchedulePolicy
)


def _grouped_decode_policy(
    quant_type: str,
    schedule: GroupedMetadataSchedule,
    metadata_conversion: str,
) -> GroupedDecodePolicy:
    if quant_type == "Q2_K":
        supported = frozenset(
            {
                GroupedMetadataSchedule.Q2ScaleMinimumNibble,
                GroupedMetadataSchedule.Q2ScaleMinimumNibbleUnrolled,
                GroupedMetadataSchedule.Q2ScaleMinimumNibbleUnrolledHipAssociation,
                GroupedMetadataSchedule.Q2HipAssociationPartialLds,
                GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDm,
                GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2,
                GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2,
                GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2DistributedProducer,
            }
        )
        if schedule not in supported:
            raise ValueError(
                f"unsupported {quant_type} grouped decode schedule {schedule.value!r}"
            )
        flags = {
            GroupedMetadataSchedule.Q2ScaleMinimumNibble: (
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ),
            GroupedMetadataSchedule.Q2ScaleMinimumNibbleUnrolled: (
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ),
            GroupedMetadataSchedule.Q2ScaleMinimumNibbleUnrolledHipAssociation: (
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ),
            GroupedMetadataSchedule.Q2HipAssociationPartialLds: (
                True,
                True,
                True,
                False,
                False,
                False,
                False,
            ),
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDm: (
                True,
                True,
                True,
                True,
                False,
                False,
                False,
            ),
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2: (
                True,
                True,
                True,
                True,
                True,
                False,
                False,
            ),
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2: (
                True,
                True,
                True,
                True,
                True,
                True,
                False,
            ),
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2DistributedProducer: (
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ),
        }[schedule]
        return GroupedQ2SchedulePolicy(metadata_conversion, *flags)
    elif quant_type == "IQ2_S":
        supported = frozenset(
            {
                GroupedMetadataSchedule.IQ2SDistributedFullWeightDecode,
                GroupedMetadataSchedule.IQ2SPayloadPrefetch,
            }
        )
        if schedule not in supported:
            raise ValueError(
                f"unsupported {quant_type} grouped decode schedule {schedule.value!r}"
            )
        return GroupedIQ2SSchedulePolicy(schedule, metadata_conversion)
    else:
        supported = frozenset(
            {
                GroupedMetadataSchedule.Serialized,
                GroupedMetadataSchedule.IndependentExtractionMetadataAfterLowWmma,
            }
        )
        if schedule not in supported:
            raise ValueError(
                f"unsupported {quant_type} grouped decode schedule {schedule.value!r}"
            )
        return GroupedDecodedSchedulePolicy(schedule, metadata_conversion)


def _grouped_metadata_schedule(policy: GroupedDecodePolicy) -> GroupedMetadataSchedule:
    if isinstance(policy, GroupedDecodedSchedulePolicy | GroupedIQ2SSchedulePolicy):
        return policy.schedule
    flags = (
        policy.unrolled_groups,
        policy.hip_association,
        policy.partial_lds,
        policy.pre_negated_dm,
        policy.paired_payload_writes,
        policy.paired_metadata_writes,
        policy.distributed_producer,
    )
    schedules = {
        (
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ): GroupedMetadataSchedule.Q2ScaleMinimumNibble,
        (
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ): GroupedMetadataSchedule.Q2ScaleMinimumNibbleUnrolled,
        (
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ): GroupedMetadataSchedule.Q2ScaleMinimumNibbleUnrolledHipAssociation,
        (
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ): GroupedMetadataSchedule.Q2HipAssociationPartialLds,
        (
            True,
            True,
            True,
            True,
            False,
            False,
            False,
        ): GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDm,
        (
            True,
            True,
            True,
            True,
            True,
            False,
            False,
        ): GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2,
        (
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ): GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2,
        (
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ): GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2DistributedProducer,
    }
    schedule = schedules.get(flags)
    if schedule is None:
        raise ValueError("unsupported orthogonal Q2 grouped decode policy") from None
    return schedule


def _grouped_decode_mapping(policy: GroupedDecodePolicy) -> dict[str, object]:
    if isinstance(policy, GroupedQ2SchedulePolicy):
        return {
            "kind": "Q2ScaleMinimum",
            "metadata_conversion": policy.metadata_conversion,
            "unrolled_groups": policy.unrolled_groups,
            "hip_association": policy.hip_association,
            "partial_lds": policy.partial_lds,
            "pre_negated_dm": policy.pre_negated_dm,
            "paired_payload_writes": policy.paired_payload_writes,
            "paired_metadata_writes": policy.paired_metadata_writes,
            "distributed_producer": policy.distributed_producer,
        }
    if isinstance(policy, GroupedIQ2SSchedulePolicy):
        return {
            "kind": "IQ2SGrid",
            "metadata_conversion": policy.metadata_conversion,
            "payload_prefetch": policy.payload_prefetch,
        }
    return {
        "kind": "ScaleMinimum",
        "metadata_conversion": policy.metadata_conversion,
        "schedule": policy.schedule.value,
    }


def _mapping_for_grouped_decode(
    value: object,
    quant_type: str,
) -> GroupedDecodePolicy:
    if quant_type == "Q2_K":
        item = _mapping(
            value,
            "GroupedForwardKernelSpec.decode",
            frozenset(
                {
                    "kind",
                    "metadata_conversion",
                    "unrolled_groups",
                    "hip_association",
                    "partial_lds",
                    "pre_negated_dm",
                    "paired_payload_writes",
                    "paired_metadata_writes",
                    "distributed_producer",
                }
            ),
        )
        if item["kind"] != "Q2ScaleMinimum":
            raise SchemaError("Q2_K grouped decode kind must be Q2ScaleMinimum")
        return GroupedQ2SchedulePolicy(
            metadata_conversion=_string(
                item["metadata_conversion"], "metadata_conversion"
            ),
            unrolled_groups=_boolean(item["unrolled_groups"], "unrolled_groups"),
            hip_association=_boolean(item["hip_association"], "hip_association"),
            partial_lds=_boolean(item["partial_lds"], "partial_lds"),
            pre_negated_dm=_boolean(item["pre_negated_dm"], "pre_negated_dm"),
            paired_payload_writes=_boolean(
                item["paired_payload_writes"], "paired_payload_writes"
            ),
            paired_metadata_writes=_boolean(
                item["paired_metadata_writes"], "paired_metadata_writes"
            ),
            distributed_producer=_boolean(
                item["distributed_producer"], "distributed_producer"
            ),
        )
    if quant_type == "IQ2_S":
        item = _mapping(
            value,
            "GroupedForwardKernelSpec.decode",
            frozenset({"kind", "metadata_conversion", "payload_prefetch"}),
        )
        if item["kind"] != "IQ2SGrid":
            raise SchemaError("IQ2_S grouped decode kind must be IQ2SGrid")
        schedule = (
            GroupedMetadataSchedule.IQ2SPayloadPrefetch
            if _boolean(item["payload_prefetch"], "payload_prefetch")
            else GroupedMetadataSchedule.IQ2SDistributedFullWeightDecode
        )
        return GroupedIQ2SSchedulePolicy(
            schedule,
            _string(item["metadata_conversion"], "metadata_conversion"),
        )
    if quant_type not in {"Q4_K", "Q5_K"}:
        raise SchemaError(f"unsupported grouped forward quant type {quant_type!r}")
    item = _mapping(
        value,
        "GroupedForwardKernelSpec.decode",
        frozenset({"kind", "metadata_conversion", "schedule"}),
    )
    if item["kind"] != "ScaleMinimum":
        raise SchemaError("Q4_K/Q5_K grouped decode kind must be ScaleMinimum")
    schedule = _enum(item["schedule"], "schedule", GroupedMetadataSchedule)
    return _grouped_decode_policy(
        quant_type,
        schedule,
        _string(item["metadata_conversion"], "metadata_conversion"),
    )


@dataclass(frozen=True)
class GroupedEpiloguePolicy:
    output_store: GroupedOutputStore
    tiles_ahead: int
    dependency_width: int
    priority: int

    @classmethod
    def from_solution(
        cls,
        solution: GroupedForwardSolution,
        row_dispatch: GroupedRowTileDispatchPolicy,
    ) -> "GroupedEpiloguePolicy":
        return cls.from_parameters(
            solution.output_store,
            solution.epilogue_tiles_ahead,
            solution.epilogue_dependency_width,
            solution.epilogue_priority,
            row_dispatch,
        )

    @classmethod
    def from_parameters(
        cls,
        output_store: GroupedOutputStore,
        tiles_ahead: int,
        dependency_width: int,
        priority: int,
        row_dispatch: GroupedRowTileDispatchPolicy,
    ) -> "GroupedEpiloguePolicy":
        stores_by_body_rows = {
            (16,): (
                GroupedOutputStore.BFloat16RNEMasked,
                GroupedOutputStore.BFloat16RNEClause1Masked,
            ),
            (32,): (GroupedOutputStore.BFloat16RNEClause2Masked,),
            (64,): (GroupedOutputStore.BFloat16RNEClause4Masked,),
            (128,): (GroupedOutputStore.BFloat16RNEClause8Masked,),
            (32, 16): (GroupedOutputStore.BFloat16RNEClause2Clause1MixedMasked,),
            (64, 32): (GroupedOutputStore.BFloat16RNEClause4Clause2MixedMasked,),
            (128, 64): (GroupedOutputStore.BFloat16RNEClause8Clause4MixedMasked,),
            (128, 64, 32): (
                GroupedOutputStore.BFloat16RNEClause8Clause4Clause2MixedMasked,
            ),
        }
        supported_stores = stores_by_body_rows.get(row_dispatch.body_rows, ())
        if output_store not in supported_stores:
            raise ValueError(
                "grouped output-store policy does not match row-body ownership"
            )
        return cls(
            output_store=output_store,
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
    row_dispatch: GroupedRowTileDispatchPolicy
    decode: GroupedDecodePolicy
    epilogue: GroupedEpiloguePolicy

    @classmethod
    def from_solution(
        cls,
        problem: GroupedForwardProblem,
        solution: GroupedForwardSolution,
    ) -> "GroupedForwardKernelSpec":
        decode = _grouped_decode_policy(
            problem.quant_data_type,
            solution.metadata_schedule,
            solution.metadata_conversion,
        )
        row_dispatch = GroupedRowTileDispatchPolicy.from_geometry(
            solution.macro_tile0, solution.tail_macro_tile0
        )
        return cls(
            geometry=GroupedGeometrySpec(
                work_group=solution.work_group,
                matrix_instruction=solution.matrix_instruction,
                macro_tile=(solution.macro_tile0, solution.macro_tile1),
                tail_macro_tile0=solution.tail_macro_tile0,
                depth_u=solution.depth_u,
            ),
            operand_source=solution.operand_source,
            activation=GroupedActivationPolicy(
                solution.activation_addressing, solution.activation_block_bytes
            ),
            row_dispatch=row_dispatch,
            decode=decode,
            epilogue=GroupedEpiloguePolicy.from_solution(solution, row_dispatch),
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
            frozenset({"geometry", "lowering", "decode", "epilogue"}),
        )
        geometry_item = _mapping(
            item["geometry"],
            "GroupedForwardKernelSpec.geometry",
            frozenset(
                {
                    "work_group",
                    "matrix_instruction",
                    "macro_tile",
                    "tail_macro_tile0",
                    "depth_u",
                }
            ),
        )
        macro_tile = _integer_tuple(geometry_item["macro_tile"], "macro_tile", 2)
        geometry = GroupedGeometrySpec(
            work_group=_integer_triple(geometry_item["work_group"], "work_group"),
            matrix_instruction=_integer_tuple(
                geometry_item["matrix_instruction"], "matrix_instruction", 9
            ),
            macro_tile=(macro_tile[0], macro_tile[1]),
            tail_macro_tile0=_integer(
                geometry_item["tail_macro_tile0"], "tail_macro_tile0"
            ),
            depth_u=_integer(geometry_item["depth_u"], "depth_u"),
        )
        row_dispatch = GroupedRowTileDispatchPolicy.from_geometry(
            geometry.macro_tile[0], geometry.tail_macro_tile0
        )
        lowering = _mapping(
            item["lowering"],
            "GroupedForwardKernelSpec.lowering",
            frozenset({"operand_source", "activation_addressing"}),
        )
        decode_item = _mapping_for_grouped_decode(item["decode"], contract.quant_type)
        epilogue_item = _mapping(
            item["epilogue"],
            "GroupedForwardKernelSpec.epilogue",
            frozenset({"tiles_ahead", "dependency_width", "priority", "output_store"}),
        )
        return cls(
            geometry=geometry,
            operand_source=_enum(
                lowering["operand_source"], "operand_source", GroupedOperandSource
            ),
            activation=GroupedActivationPolicy(
                _enum(
                    lowering["activation_addressing"],
                    "activation_addressing",
                    GroupedActivationAddressing,
                ),
                contract.activation_block_bytes,
            ),
            row_dispatch=row_dispatch,
            decode=decode_item,
            epilogue=GroupedEpiloguePolicy.from_parameters(
                _enum(
                    epilogue_item["output_store"],
                    "output_store",
                    GroupedOutputStore,
                ),
                _integer(epilogue_item["tiles_ahead"], "tiles_ahead"),
                _integer(epilogue_item["dependency_width"], "dependency_width"),
                _integer(epilogue_item["priority"], "priority"),
                row_dispatch,
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "geometry": {
                "work_group": list(self.geometry.work_group),
                "matrix_instruction": list(self.geometry.matrix_instruction),
                "macro_tile": list(self.geometry.macro_tile),
                "tail_macro_tile0": self.geometry.tail_macro_tile0,
                "depth_u": self.geometry.depth_u,
            },
            "lowering": {
                "operand_source": self.operand_source.value,
                "activation_addressing": self.activation.addressing.value,
            },
            "decode": _grouped_decode_mapping(self.decode),
            "epilogue": {
                "tiles_ahead": self.epilogue.tiles_ahead,
                "dependency_width": self.epilogue.dependency_width,
                "priority": self.epilogue.priority,
                "output_store": self.epilogue.output_store.value,
            },
        }

    def to_solution(
        self,
        contract: GroupedForwardProblemContract,
    ) -> GroupedForwardSolution:
        return GroupedForwardSolution(
            kernel_language=contract.kernel_language,
            isa=contract.isa,
            wavefront_size=contract.wavefront_size,
            work_group=self.geometry.work_group,
            matrix_instruction=self.geometry.matrix_instruction,
            macro_tile0=self.geometry.macro_tile[0],
            tail_macro_tile0=self.geometry.tail_macro_tile0,
            macro_tile1=self.geometry.macro_tile[1],
            depth_u=self.geometry.depth_u,
            activation_layout=contract.activation_layout,
            activation_block_bytes=contract.activation_block_bytes,
            packed_weight_block_bytes=contract.packed_weight_block_bytes,
            operand_source=self.operand_source,
            weight_decode=contract.weight_decode,
            group_mapping="SerialGemm",
            route_layout="CumulativeOffsetsExpertIndices",
            activation_addressing=self.activation.addressing,
            metadata_conversion=self.decode.metadata_conversion,
            metadata_schedule=_grouped_metadata_schedule(self.decode),
            epilogue_tiles_ahead=self.epilogue.tiles_ahead,
            epilogue_dependency_width=self.epilogue.dependency_width,
            epilogue_priority=self.epilogue.priority,
            scale_arithmetic=contract.scale_arithmetic,
            output_store=self.epilogue.output_store,
            signed_weight=contract.signed_weight,
            signed_activation=contract.signed_activation,
            wmma_clamp=contract.wmma_clamp,
        )


def grouped_forward_capability_rejection_reason(
    problem: GroupedForwardProblem,
    solution: GroupedForwardSolution,
) -> str | None:
    contract_rejection = GroupedForwardProblemContract.rejection_reason(
        problem, solution
    )
    if contract_rejection is not None:
        return contract_rejection
    kernel_spec = GroupedForwardKernelSpec.from_solution(problem, solution)

    common_checks = (
        (solution.depth_u == 32, "grouped forward requires DepthU=32"),
        (
            solution.macro_tile1 > 0
            and problem.output_features % solution.macro_tile1 == 0,
            "grouped output features must be divisible by the output tile",
        ),
    )
    rejection = next(
        (message for accepted, message in common_checks if not accepted), None
    )
    if rejection is not None:
        return rejection

    source = kernel_spec.operand_source
    supported_sources = {
        "Q2_K": {GroupedOperandSource.GroupedDecodedWeightLds},
        "Q4_K": {
            GroupedOperandSource.GroupedDirectGlobal,
            GroupedOperandSource.GroupedDecodedWeightLds,
        },
        "Q5_K": {GroupedOperandSource.GroupedDecodedWeightLds},
        "IQ2_S": {GroupedOperandSource.GroupedIQ2SFullWeightLds},
    }.get(problem.quant_data_type, set())
    if source not in supported_sources:
        return "grouped operand source is unavailable for the quant format"

    if source is GroupedOperandSource.GroupedDirectGlobal:
        checks = (
            (solution.work_group == (32, 1, 1), "direct grouped workgroup mismatch"),
            (
                solution.matrix_instruction == (16, 16, 16, 1, 1, 1, 1, 1, 1),
                "direct grouped matrix instruction mismatch",
            ),
            (
                (solution.macro_tile0, solution.tail_macro_tile0, solution.macro_tile1)
                == (16, 16, 16),
                "direct grouped tile geometry mismatch",
            ),
            (
                solution.activation_addressing
                is GroupedActivationAddressing.AggregateRows,
                "direct grouped activation addressing mismatch",
            ),
            (
                solution.metadata_schedule is GroupedMetadataSchedule.Serialized,
                "direct grouped metadata schedule mismatch",
            ),
            (
                solution.metadata_conversion == "Float32ThenFloat16",
                "direct grouped metadata conversion mismatch",
            ),
            (
                (
                    solution.epilogue_tiles_ahead,
                    solution.epilogue_dependency_width,
                    solution.epilogue_priority,
                )
                == (1, 1, 0),
                "direct grouped epilogue policy mismatch",
            ),
        )
        return next((message for accepted, message in checks if not accepted), None)

    if source is GroupedOperandSource.GroupedIQ2SFullWeightLds:
        checks = (
            (solution.work_group == (128, 1, 1), "IQ2_S workgroup mismatch"),
            (
                solution.matrix_instruction == (16, 16, 16, 1, 1, 1, 4, 4, 1),
                "IQ2_S matrix instruction mismatch",
            ),
            (
                (solution.macro_tile0, solution.tail_macro_tile0, solution.macro_tile1)
                == (64, 64, 64),
                "IQ2_S tile geometry mismatch",
            ),
            (
                solution.activation_addressing
                in {
                    GroupedActivationAddressing.AggregateRowsTiled,
                    GroupedActivationAddressing.AggregateRowsTiledLinear,
                },
                "IQ2_S activation addressing mismatch",
            ),
            (
                solution.metadata_conversion == "Float16DUnsignedNibbleScaleToFloat32",
                "IQ2_S metadata conversion mismatch",
            ),
            (
                (
                    solution.epilogue_tiles_ahead,
                    solution.epilogue_dependency_width,
                    solution.epilogue_priority,
                )
                == (4, 1, 0),
                "IQ2_S epilogue policy mismatch",
            ),
        )
        return next((message for accepted, message in checks if not accepted), None)

    row_tiles = solution.macro_tile0 // 16
    checks = (
        (solution.work_group == (128, 1, 1), "decoded grouped workgroup mismatch"),
        (
            solution.matrix_instruction == (16, 16, 16, 1, 1, 1, 4, 4, 1),
            "decoded grouped matrix instruction mismatch",
        ),
        (solution.macro_tile1 == 64, "decoded grouped output tile mismatch"),
        (
            solution.activation_addressing
            is GroupedActivationAddressing.AggregateRowsTiled,
            "decoded grouped activation addressing mismatch",
        ),
        (
            solution.metadata_conversion
            == (
                "DirectQ2Float16NibblePairs"
                if problem.quant_data_type == "Q2_K"
                else "DirectFloat16Unsigned16"
            ),
            "decoded grouped metadata conversion mismatch",
        ),
        (
            solution.epilogue_tiles_ahead in {1, row_tiles},
            "decoded grouped epilogue distance is unsupported",
        ),
        (
            solution.epilogue_dependency_width in {1, 2, 4}
            and solution.epilogue_dependency_width <= row_tiles,
            "decoded grouped epilogue dependency width is unsupported",
        ),
        (
            solution.epilogue_priority in {0, 2},
            "decoded grouped epilogue priority is unsupported",
        ),
    )
    rejection = next((message for accepted, message in checks if not accepted), None)
    if rejection is not None:
        return rejection

    if problem.quant_data_type == "Q2_K":
        q2_rows = {
            GroupedMetadataSchedule.Q2ScaleMinimumNibble: {32, 64},
            GroupedMetadataSchedule.Q2ScaleMinimumNibbleUnrolled: {32, 64, 128},
            GroupedMetadataSchedule.Q2ScaleMinimumNibbleUnrolledHipAssociation: {
                32,
                64,
            },
            GroupedMetadataSchedule.Q2HipAssociationPartialLds: {32},
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDm: {32},
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2: {32},
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2: {
                32
            },
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2DistributedProducer: {
                32,
                64,
            },
        }
        if solution.macro_tile0 not in q2_rows.get(solution.metadata_schedule, set()):
            return "Q2 decode schedule does not support the requested row tile"
    elif solution.macro_tile0 not in {64, 128}:
        return "Q4/Q5 decoded lowering requires a 64- or 128-row tile"

    activation_staging = GroupedActivationStagingPlan(
        addressing=kernel_spec.activation.addressing,
        block_bytes=kernel_spec.activation.block_bytes,
        participating_threads=solution.num_threads,
    )
    grouped_decoded_physical_plan(
        solution.activation_block_bytes,
        solution.macro_tile0,
        problem.quant_data_type,
        activation_staging,
    )
    return None


@dataclass(frozen=True)
class GroupedRouteState:
    physical_experts: int
    output_features: int
    aggregate_rows: int
    blocks_per_weight_row: int
    bytes_per_expert: int


@dataclass(frozen=True)
class DerivedGroupedForwardState:
    key: GroupedForwardSolutionKey
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
    def from_solution_key(
        cls, key: GroupedForwardSolutionKey
    ) -> "DerivedGroupedForwardState":
        problem = key.problem
        solution = key.solution
        semantics = QuantForwardSemantics.for_quant_type(problem.quant_data_type)
        contract = GroupedForwardProblemContract.from_solution(problem, solution)
        kernel_spec = GroupedForwardKernelSpec.from_solution(problem, solution)
        blocks_per_weight_row = problem.input_features // contract.block_values
        activation_blocks_per_row = problem.input_features // Q8_1_D4_BLOCK_VALUES
        packed_weight_row_bytes = (
            blocks_per_weight_row * contract.packed_weight_block_bytes
        )
        activation_plane_stride_bytes = (
            problem.aggregate_rows * solution.activation_block_bytes
        )
        if solution.operand_source is GroupedOperandSource.GroupedDirectGlobal:
            physical_plan: (
                GroupedDirectPhysicalPlan
                | GroupedDecodedPhysicalPlan
                | GroupedIQ2SFullWeightPhysicalPlan
            ) = grouped_direct_physical_plan(solution.activation_block_bytes)
        elif solution.operand_source is GroupedOperandSource.GroupedDecodedWeightLds:
            activation_staging = GroupedActivationStagingPlan(
                addressing=kernel_spec.activation.addressing,
                block_bytes=kernel_spec.activation.block_bytes,
                participating_threads=solution.num_threads,
            )
            physical_plan = grouped_decoded_physical_plan(
                solution.activation_block_bytes,
                solution.macro_tile0,
                problem.quant_data_type,
                activation_staging,
            )
        elif solution.operand_source is GroupedOperandSource.GroupedIQ2SFullWeightLds:
            physical_plan = grouped_iq2_s_full_weight_physical_plan()
        else:
            raise ValueError(
                f"unsupported grouped operand source {solution.operand_source!r}"
            )
        return cls(
            key=key,
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
            output_column_tiles=problem.output_features // solution.macro_tile1,
        )

    @property
    def expected_packed_weight_shape(self) -> tuple[int, int, int]:
        problem = self.key.problem
        return (
            problem.physical_experts,
            problem.output_features,
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
        problem = self.key.problem
        return (problem.aggregate_rows, problem.output_features)

    def grid(self, route_entries: int) -> tuple[int, int, int]:
        if route_entries <= 0 or route_entries > self.key.problem.max_route_entries:
            raise ValueError(
                "route entry count is outside the grouped problem contract"
            )
        return (self.output_column_tiles, route_entries, 1)
