"""Strict identities for research-only paired grouped forward paths."""

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar

from typing_extensions import Self

from .grouped_mmq_fwd_model import (
    GroupedActivationAddressing,
    GroupedOutputStore,
    _integer,
    _mapping,
)
from .model import SchemaError
from .quant_formats import Q8_1_F32_D4_BLOCK_BYTES


class GroupedPairOperandSource(str, Enum):
    IQ2SHalfWeightLds = "IQ2SHalfWeightLds"
    IQ2XXSHalfWeightLds = "IQ2XXSHalfWeightLds"
    Q3KHalfWeightLds = "Q3KHalfWeightLds"


class GroupedPairProjectionSchedule(str, Enum):
    K128Interleaved = "K128Interleaved"


class GroupedPairDecodeSchedule(str, Enum):
    TwoLaneSelectedHalfPayloadPrefetch = "TwoLaneSelectedHalfPayloadPrefetch"
    TwoLaneSelectedHalfIQ2XXS = "TwoLaneSelectedHalfIQ2XXS"
    TwoLaneSelectedHalfIQ2XXSFusedSelector = "TwoLaneSelectedHalfIQ2XXSFusedSelector"
    TwoLaneSelectedHalfQ3 = "TwoLaneSelectedHalfQ3"
    TwoLaneSelectedHalfQ3VariableBFE = "TwoLaneSelectedHalfQ3VariableBFE"


class GroupedPairRouteOwnership(str, Enum):
    SerialRoutes = "SerialRoutes"
    DeviceRowTasks64 = "DeviceRowTasks64"


@dataclass(frozen=True)
class GroupedForwardPairProblem:
    """One exact two-projection aggregate routed problem."""

    quant_data_type: str
    aggregate_rows: int
    output_features: int
    input_features: int
    physical_experts: int
    max_route_entries: int
    projection_count: int

    @classmethod
    def iq2_s(cls, aggregate_rows: int) -> Self:
        return cls("IQ2_S", aggregate_rows, 512, 2048, 256, 256, 2)

    @classmethod
    def iq2_xxs(cls, aggregate_rows: int) -> Self:
        return cls("IQ2_XXS", aggregate_rows, 2048, 4096, 256, 256, 2)

    @classmethod
    def q3_k(cls, aggregate_rows: int) -> Self:
        return cls("Q3_K", aggregate_rows, 512, 2048, 256, 256, 2)


@dataclass(frozen=True)
class GroupedForwardPairSolution:
    """Complete mechanism identity for one paired lowering."""

    kernel_language: str
    isa: tuple[int, int, int]
    wavefront_size: int
    work_group: tuple[int, int, int]
    matrix_instruction: tuple[int, ...]
    macro_tile0: int
    macro_tile1: int
    depth_u: int
    activation_layout: str
    activation_block_bytes: int
    packed_weight_block_bytes: int
    operand_source: GroupedPairOperandSource
    projection_schedule: GroupedPairProjectionSchedule
    weight_decode: str
    group_mapping: str
    route_layout: str
    activation_addressing: GroupedActivationAddressing
    metadata_conversion: str
    metadata_schedule: GroupedPairDecodeSchedule
    output_store: GroupedOutputStore
    scale_arithmetic: str
    projection_count: int
    signed_weight: bool
    signed_activation: bool
    wmma_clamp: bool

    @classmethod
    def iq2_s_k128_interleaved(cls) -> Self:
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(128, 1, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=64,
            macro_tile1=64,
            depth_u=128,
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            packed_weight_block_bytes=82,
            operand_source=GroupedPairOperandSource.IQ2SHalfWeightLds,
            projection_schedule=GroupedPairProjectionSchedule.K128Interleaved,
            weight_decode="TwoLaneSelectedHalfIQ2SGridSigned",
            group_mapping="SerialGemmPair",
            route_layout="CumulativeOffsetsExpertIndices",
            activation_addressing=GroupedActivationAddressing.AggregateRowsTiledLinear,
            metadata_conversion="Float16DUnsignedNibbleScaleToFloat32",
            metadata_schedule=(
                GroupedPairDecodeSchedule.TwoLaneSelectedHalfPayloadPrefetch
            ),
            output_store=GroupedOutputStore.BFloat16RNEClause4Masked,
            scale_arithmetic="Int32ScaleF32",
            projection_count=2,
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=False,
        )

    @classmethod
    def iq2_s_k128_interleaved_row_tasks(cls) -> Self:
        return replace(
            cls.iq2_s_k128_interleaved(),
            group_mapping="RowTaskGemmPair",
            route_layout="DeviceRowTasks64",
        )

    @classmethod
    def iq2_xxs_k128_interleaved(cls) -> Self:
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(128, 1, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=64,
            macro_tile1=64,
            depth_u=128,
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            packed_weight_block_bytes=66,
            operand_source=GroupedPairOperandSource.IQ2XXSHalfWeightLds,
            projection_schedule=GroupedPairProjectionSchedule.K128Interleaved,
            weight_decode="TwoLaneSelectedHalfIQ2XXSGridParitySigned",
            group_mapping="SerialGemmPair",
            route_layout="CumulativeOffsetsExpertIndices",
            activation_addressing=GroupedActivationAddressing.AggregateRowsTiledLinear,
            metadata_conversion="Float16DParitySignsOddScaleToFloat32",
            metadata_schedule=GroupedPairDecodeSchedule.TwoLaneSelectedHalfIQ2XXS,
            output_store=GroupedOutputStore.BFloat16RNEClause4Masked,
            scale_arithmetic="Int32ScaleF32",
            projection_count=2,
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=False,
        )

    @classmethod
    def iq2_xxs_k128_interleaved_fused_selector(cls) -> Self:
        return replace(
            cls.iq2_xxs_k128_interleaved(),
            metadata_schedule=(
                GroupedPairDecodeSchedule.TwoLaneSelectedHalfIQ2XXSFusedSelector
            ),
        )

    @classmethod
    def iq2_xxs_k128_interleaved_fused_selector_j80(cls) -> Self:
        return replace(
            cls.iq2_xxs_k128_interleaved_fused_selector(),
            macro_tile0=80,
        )

    @classmethod
    def q3_k_k128_interleaved(cls) -> Self:
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(128, 1, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=64,
            macro_tile1=64,
            depth_u=128,
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            packed_weight_block_bytes=110,
            operand_source=GroupedPairOperandSource.Q3KHalfWeightLds,
            projection_schedule=GroupedPairProjectionSchedule.K128Interleaved,
            weight_decode="TwoLaneSelectedHalfQ3Signed",
            group_mapping="SerialGemmPair",
            route_layout="CumulativeOffsetsExpertIndices",
            activation_addressing=GroupedActivationAddressing.AggregateRowsTiledLinear,
            metadata_conversion="Float16DSignedSixBitScaleToFloat32",
            metadata_schedule=GroupedPairDecodeSchedule.TwoLaneSelectedHalfQ3,
            output_store=GroupedOutputStore.BFloat16RNEClause4Masked,
            scale_arithmetic="Int32ScaleF32",
            projection_count=2,
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=False,
        )

    @classmethod
    def q3_k_k128_interleaved_row_tasks(cls) -> Self:
        return replace(
            cls.q3_k_k128_interleaved(),
            group_mapping="RowTaskGemmPair",
            route_layout="DeviceRowTasks64",
        )

    @classmethod
    def q3_k_k128_interleaved_row_tasks_variable_bfe(cls) -> Self:
        return replace(
            cls.q3_k_k128_interleaved_row_tasks(),
            metadata_schedule=(
                GroupedPairDecodeSchedule.TwoLaneSelectedHalfQ3VariableBFE
            ),
        )

    @property
    def route_ownership(self) -> GroupedPairRouteOwnership:
        identity = (self.group_mapping, self.route_layout)
        if identity == ("SerialGemmPair", "CumulativeOffsetsExpertIndices"):
            return GroupedPairRouteOwnership.SerialRoutes
        if identity == ("RowTaskGemmPair", "DeviceRowTasks64"):
            return GroupedPairRouteOwnership.DeviceRowTasks64
        raise ValueError("paired route ownership identity is unsupported")

    @property
    def num_threads(self) -> int:
        return self.work_group[0] * self.work_group[1] * self.work_group[2]


@dataclass(frozen=True)
class GroupedForwardPairSolutionKey:
    problem: GroupedForwardPairProblem
    solution: GroupedForwardPairSolution

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {"ArtifactKind", "KernelFamily", "ProblemContract", "Problem", "KernelSpec"}
    )

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _mapping(value, "GroupedForwardPairSolutionKey", cls._KEYS)
        if item["ArtifactKind"] != "ExactKernel":
            raise SchemaError("paired key ArtifactKind must be ExactKernel")
        if item["KernelFamily"] != "GroupedForwardPair":
            raise SchemaError("paired key KernelFamily must be GroupedForwardPair")
        from .grouped_mmq_fwd_pair_spec import (
            GroupedForwardPairContract,
            GroupedForwardPairKernelSpec,
            grouped_forward_pair_capability_rejection_reason,
        )

        contract = GroupedForwardPairContract.from_mapping(item["ProblemContract"])
        problem_item = _mapping(
            item["Problem"], "GroupedForwardPairProblem", frozenset({"aggregate_rows"})
        )
        problem = contract.problem(
            _integer(problem_item["aggregate_rows"], "aggregate_rows")
        )
        if not 0 < problem.aggregate_rows <= 0xFFFFFFFF:
            raise SchemaError("paired aggregate_rows must fit in a positive u32")
        spec = GroupedForwardPairKernelSpec.from_mapping(item["KernelSpec"], contract)
        solution = spec.to_solution(contract)
        if GroupedForwardPairContract.from_solution(problem, solution) != contract:
            raise SchemaError(
                "GroupedForwardPairContract does not round-trip canonically"
            )
        rejection = grouped_forward_pair_capability_rejection_reason(problem, solution)
        if rejection is not None:
            raise SchemaError(
                f"paired kernel specification is not canonical: {rejection}"
            )
        return cls(problem, solution)

    def to_mapping(self) -> dict[str, object]:
        from .grouped_mmq_fwd_pair_spec import (
            GroupedForwardPairContract,
            GroupedForwardPairKernelSpec,
        )

        contract = GroupedForwardPairContract.from_solution(self.problem, self.solution)
        spec = GroupedForwardPairKernelSpec.from_solution(self.solution)
        return {
            "ArtifactKind": "ExactKernel",
            "KernelFamily": "GroupedForwardPair",
            "ProblemContract": contract.to_mapping(),
            "Problem": {"aggregate_rows": self.problem.aggregate_rows},
            "KernelSpec": spec.to_mapping(),
        }

    @property
    def hash(self) -> str:
        canonical = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"ggpair_{digest[:16]}"

    @property
    def kernel_name(self) -> str:
        problem = self.problem
        return (
            "torch_ggml_ops_ggtensile_gfx1151_v1_grouped_mmq_fwd_pair_"
            f"{problem.quant_data_type.lower()}_r{problem.aggregate_rows}_"
            f"n{problem.output_features}_k{problem.input_features}_{self.hash[7:]}"
        )
