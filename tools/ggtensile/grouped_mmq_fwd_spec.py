"""Derived state for isolated grouped MMQ forward kernels."""

from dataclasses import dataclass
from typing import cast

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


@dataclass(frozen=True)
class GroupedForwardProblemContract:
    """Non-tunable grouped data, arithmetic, ISA, and destination contract."""

    quant_type: str
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

    @classmethod
    def from_solution(
        cls,
        problem: GroupedForwardProblem,
        solution: GroupedForwardSolution,
    ) -> "GroupedForwardProblemContract":
        try:
            quant_format = GROUPED_QUANT_FORMATS[problem.quant_data_type]
        except KeyError:
            raise ValueError(
                f"unsupported grouped forward quant type {problem.quant_data_type!r}"
            ) from None
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
        if rejection is not None:
            raise ValueError(rejection)
        return cls(
            quant_type=problem.quant_data_type,
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
    schedule: GroupedMetadataSchedule

    @property
    def unrolled_groups(self) -> bool:
        return self.schedule is not GroupedMetadataSchedule.Q2ScaleMinimumNibble

    @property
    def hip_association(self) -> bool:
        return self.schedule not in (
            GroupedMetadataSchedule.Q2ScaleMinimumNibble,
            GroupedMetadataSchedule.Q2ScaleMinimumNibbleUnrolled,
        )

    @property
    def partial_lds(self) -> bool:
        return self.schedule in (
            GroupedMetadataSchedule.Q2HipAssociationPartialLds,
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDm,
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2,
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2,
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2DistributedProducer,
        )

    @property
    def pre_negated_dm(self) -> bool:
        return self.schedule in (
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDm,
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2,
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2,
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2DistributedProducer,
        )

    @property
    def paired_payload_writes(self) -> bool:
        return self.schedule in (
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2,
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2,
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2DistributedProducer,
        )

    @property
    def paired_metadata_writes(self) -> bool:
        return self.schedule in (
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2,
            GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2DistributedProducer,
        )

    @property
    def distributed_producer(self) -> bool:
        return (
            self.schedule
            is GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2DistributedProducer
        )

    @property
    def independent_metadata_extraction(self) -> bool:
        return False

    @property
    def defer_metadata_reads(self) -> bool:
        return False


@dataclass(frozen=True)
class GroupedIQ2SSchedulePolicy:
    schedule: GroupedMetadataSchedule

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
        return GroupedQ2SchedulePolicy(schedule)
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
        return GroupedIQ2SSchedulePolicy(schedule)
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
        return GroupedDecodedSchedulePolicy(schedule)


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
        if solution.output_store not in supported_stores:
            raise ValueError(
                "grouped output-store policy does not match row-body ownership"
            )
        return cls(
            output_store=solution.output_store,
            tiles_ahead=solution.epilogue_tiles_ahead,
            dependency_width=solution.epilogue_dependency_width,
            priority=solution.epilogue_priority,
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
    matrix_instruction: tuple[int, int, int, int]
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
            problem.quant_data_type, solution.metadata_schedule
        )
        row_dispatch = GroupedRowTileDispatchPolicy.from_geometry(
            solution.macro_tile0, solution.tail_macro_tile0
        )
        return cls(
            geometry=GroupedGeometrySpec(
                work_group=solution.work_group,
                matrix_instruction=cast(
                    tuple[int, int, int, int], solution.matrix_instruction[:4]
                ),
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
