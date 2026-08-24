"""Strict identities for isolated grouped MMQ forward kernels."""

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar

from typing_extensions import Self

from .quant_formats import (
    GROUPED_QUANT_FORMATS,
    Q8_1_F16_D2S6_BLOCK_BYTES,
    Q8_1_F16_D4S4_BLOCK_BYTES,
    Q8_1_F32_D4_BLOCK_BYTES,
)
from .schema import SchemaError
from .schema import integer as _integer
from .schema import strict_mapping as _mapping


class GroupedOperandSource(str, Enum):
    GroupedDirectGlobal = "GroupedDirectGlobal"
    GroupedDecodedWeightLds = "GroupedDecodedWeightLds"
    GroupedIQ2SFullWeightLds = "GroupedIQ2SFullWeightLds"


class GroupedActivationAddressing(str, Enum):
    AggregateRows = "AggregateRows"
    AggregateRowsTiled = "AggregateRowsTiled"
    AggregateRowsTiledLinear = "AggregateRowsTiledLinear"


class GroupedMetadataSchedule(str, Enum):
    Serialized = "Serialized"
    IndependentExtractionMetadataAfterLowWmma = (
        "IndependentExtractionMetadataAfterLowWmma"
    )
    IQ2SDistributedFullWeightDecode = "IQ2SDistributedFullWeightDecode"
    IQ2SPayloadPrefetch = "IQ2SPayloadPrefetch"
    Q2ScaleMinimumNibble = "Q2ScaleMinimumNibble"
    Q2ScaleMinimumNibbleUnrolled = "Q2ScaleMinimumNibbleUnrolled"
    Q2ScaleMinimumNibbleUnrolledHipAssociation = (
        "Q2ScaleMinimumNibbleUnrolledHipAssociation"
    )
    Q2HipAssociationPartialLds = "Q2HipAssociationPartialLds"
    Q2HipAssociationPartialLdsPreNegatedDm = "Q2HipAssociationPartialLdsPreNegatedDm"
    Q2HipAssociationPartialLdsPreNegatedDmWrite2 = (
        "Q2HipAssociationPartialLdsPreNegatedDmWrite2"
    )
    Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2 = (
        "Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2"
    )
    Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2DistributedProducer = (
        "Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2DistributedProducer"
    )


class GroupedOutputStore(str, Enum):
    BFloat16RNEMasked = "BFloat16RNEMasked"
    BFloat16RNEClause1Masked = "BFloat16RNEClause1Masked"
    BFloat16RNEClause2Masked = "BFloat16RNEClause2Masked"
    BFloat16RNEClause4Masked = "BFloat16RNEClause4Masked"
    BFloat16RNEClause8Masked = "BFloat16RNEClause8Masked"
    BFloat16RNEClause2Clause1MixedMasked = "BFloat16RNEClause2Clause1MixedMasked"
    BFloat16RNEClause4Clause2MixedMasked = "BFloat16RNEClause4Clause2MixedMasked"
    BFloat16RNEClause8Clause4MixedMasked = "BFloat16RNEClause8Clause4MixedMasked"
    BFloat16RNEClause8Clause4Clause2MixedMasked = (
        "BFloat16RNEClause8Clause4Clause2MixedMasked"
    )


@dataclass(frozen=True)
class GroupedForwardProblem:
    """One aggregate routed problem over the fixed packed expert bank."""

    quant_data_type: str
    aggregate_rows: int
    output_features: int
    input_features: int
    physical_experts: int
    max_route_entries: int

    @classmethod
    def q2_k(cls, aggregate_rows: int) -> Self:
        return cls("Q2_K", aggregate_rows, 4096, 2048, 256, 256)

    @classmethod
    def q4_k(cls, aggregate_rows: int) -> Self:
        return cls("Q4_K", aggregate_rows, 2048, 512, 256, 256)

    @classmethod
    def q5_k(cls, aggregate_rows: int) -> Self:
        return cls("Q5_K", aggregate_rows, 2048, 512, 256, 256)

    @classmethod
    def iq2_s(cls, aggregate_rows: int) -> Self:
        return cls("IQ2_S", aggregate_rows, 2048, 512, 256, 256)


@dataclass(frozen=True)
class GroupedForwardSolution:
    """Complete mechanism identity for one grouped forward lowering."""

    kernel_language: str
    isa: tuple[int, int, int]
    wavefront_size: int
    work_group: tuple[int, int, int]
    matrix_instruction: tuple[int, ...]
    macro_tile0: int
    tail_macro_tile0: int
    macro_tile1: int
    depth_u: int
    activation_layout: str
    activation_block_bytes: int
    packed_weight_block_bytes: int
    operand_source: GroupedOperandSource
    weight_decode: str
    group_mapping: str
    route_layout: str
    activation_addressing: GroupedActivationAddressing
    metadata_conversion: str
    metadata_schedule: GroupedMetadataSchedule
    epilogue_tiles_ahead: int
    epilogue_dependency_width: int
    epilogue_priority: int
    scale_arithmetic: str
    output_store: GroupedOutputStore
    signed_weight: bool
    signed_activation: bool
    wmma_clamp: bool

    @classmethod
    def q4_k_serial_direct(cls) -> Self:
        return cls(
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            work_group=(32, 1, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 1, 1, 1),
            macro_tile0=16,
            tail_macro_tile0=16,
            macro_tile1=16,
            depth_u=32,
            activation_layout="F16_D4S4",
            activation_block_bytes=Q8_1_F16_D4S4_BLOCK_BYTES,
            packed_weight_block_bytes=GROUPED_QUANT_FORMATS["Q4_K"].block_bytes,
            operand_source=GroupedOperandSource.GroupedDirectGlobal,
            weight_decode="DirectNibble",
            group_mapping="SerialGemm",
            route_layout="CumulativeOffsetsExpertIndices",
            activation_addressing=GroupedActivationAddressing.AggregateRows,
            metadata_conversion="Float32ThenFloat16",
            metadata_schedule=GroupedMetadataSchedule.Serialized,
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=1,
            epilogue_priority=0,
            scale_arithmetic="FP16",
            output_store=GroupedOutputStore.BFloat16RNEMasked,
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=True,
        )

    @classmethod
    def q4_k_serial_decoded_lds(cls) -> Self:
        return replace(
            cls.q4_k_serial_direct(),
            work_group=(128, 1, 1),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=128,
            tail_macro_tile0=128,
            macro_tile1=64,
            operand_source=GroupedOperandSource.GroupedDecodedWeightLds,
            activation_addressing=GroupedActivationAddressing.AggregateRowsTiled,
            metadata_conversion="DirectFloat16Unsigned16",
            epilogue_tiles_ahead=8,
            output_store=GroupedOutputStore.BFloat16RNEClause8Masked,
        )

    @classmethod
    def q4_k_serial_decoded_lds_64(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds(),
            macro_tile0=64,
            tail_macro_tile0=64,
            epilogue_tiles_ahead=4,
            output_store=GroupedOutputStore.BFloat16RNEClause4Masked,
        )

    @classmethod
    def q4_k_serial_decoded_lds_scheduled(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds(),
            metadata_schedule=(
                GroupedMetadataSchedule.IndependentExtractionMetadataAfterLowWmma
            ),
        )

    @classmethod
    def q4_k_serial_decoded_lds_scheduled_a1d2p2(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds_scheduled(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
        )

    @classmethod
    def q4_k_serial_decoded_lds_scheduled_a1d4p2(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds_scheduled(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=4,
            epilogue_priority=2,
        )

    @classmethod
    def q4_k_serial_decoded_lds_64_scheduled(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds_64(),
            metadata_schedule=(
                GroupedMetadataSchedule.IndependentExtractionMetadataAfterLowWmma
            ),
        )

    @classmethod
    def q4_k_serial_decoded_lds_64_scheduled_mixed32(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds_64_scheduled(),
            tail_macro_tile0=32,
            output_store=GroupedOutputStore.BFloat16RNEClause4Clause2MixedMasked,
        )

    @classmethod
    def q4_k_serial_decoded_lds_64_scheduled_mixed32_a1d2p2(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds_64_scheduled_mixed32(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
        )

    @classmethod
    def q4_k_serial_decoded_lds_64_scheduled_mixed32_a1d4p2(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds_64_scheduled_mixed32(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=4,
            epilogue_priority=2,
        )

    @classmethod
    def iq2_s_serial_full_weight_lds_64(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds_64(),
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            packed_weight_block_bytes=GROUPED_QUANT_FORMATS["IQ2_S"].block_bytes,
            operand_source=GroupedOperandSource.GroupedIQ2SFullWeightLds,
            weight_decode="DirectIQ2SGridSigned",
            metadata_conversion="Float16DUnsignedNibbleScaleToFloat32",
            metadata_schedule=GroupedMetadataSchedule.IQ2SDistributedFullWeightDecode,
            scale_arithmetic="Int32ScaleF32",
            output_store=GroupedOutputStore.BFloat16RNEClause4Masked,
            wmma_clamp=False,
        )

    @classmethod
    def iq2_s_serial_full_weight_lds_64_linear_activation(cls) -> Self:
        return replace(
            cls.iq2_s_serial_full_weight_lds_64(),
            activation_addressing=(
                GroupedActivationAddressing.AggregateRowsTiledLinear
            ),
        )

    @classmethod
    def iq2_s_serial_full_weight_lds_64_linear_payload_prefetch(cls) -> Self:
        return replace(
            cls.iq2_s_serial_full_weight_lds_64_linear_activation(),
            metadata_schedule=GroupedMetadataSchedule.IQ2SPayloadPrefetch,
        )

    @classmethod
    def q2_k_serial_decoded_lds_32(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds(),
            macro_tile0=32,
            tail_macro_tile0=32,
            activation_layout="F16_D2S6",
            activation_block_bytes=Q8_1_F16_D2S6_BLOCK_BYTES,
            packed_weight_block_bytes=GROUPED_QUANT_FORMATS["Q2_K"].block_bytes,
            weight_decode="DirectTwoBitNibbleScaleMinimum",
            metadata_conversion="DirectQ2Float16NibblePairs",
            metadata_schedule=GroupedMetadataSchedule.Q2ScaleMinimumNibble,
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
            output_store=GroupedOutputStore.BFloat16RNEClause2Masked,
        )

    @classmethod
    def q2_k_serial_decoded_lds_64(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32(),
            macro_tile0=64,
            tail_macro_tile0=64,
            epilogue_tiles_ahead=4,
            output_store=GroupedOutputStore.BFloat16RNEClause4Masked,
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_unrolled(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32(),
            metadata_schedule=GroupedMetadataSchedule.Q2ScaleMinimumNibbleUnrolled,
        )

    @classmethod
    def q2_k_serial_decoded_lds_64_unrolled(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_64(),
            metadata_schedule=GroupedMetadataSchedule.Q2ScaleMinimumNibbleUnrolled,
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_association(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_unrolled(),
            metadata_schedule=(
                GroupedMetadataSchedule.Q2ScaleMinimumNibbleUnrolledHipAssociation
            ),
        )

    @classmethod
    def q2_k_serial_decoded_lds_64_hip_association(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_64_unrolled(),
            metadata_schedule=(
                GroupedMetadataSchedule.Q2ScaleMinimumNibbleUnrolledHipAssociation
            ),
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_partial_lds(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_hip_association(),
            metadata_schedule=GroupedMetadataSchedule.Q2HipAssociationPartialLds,
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_pre_negated_dm(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_hip_partial_lds(),
            metadata_schedule=(
                GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDm
            ),
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_hip_pre_negated_dm(),
            metadata_schedule=(
                GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2
            ),
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2(),
            metadata_schedule=(
                GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2
            ),
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2_distributed(
        cls,
    ) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2(),
            metadata_schedule=(
                GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2DistributedProducer
            ),
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2_mixed16(
        cls,
    ) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2(),
            tail_macro_tile0=16,
            output_store=(GroupedOutputStore.BFloat16RNEClause2Clause1MixedMasked),
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2_distributed_mixed16(
        cls,
    ) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2_distributed(),
            tail_macro_tile0=16,
            output_store=(GroupedOutputStore.BFloat16RNEClause2Clause1MixedMasked),
        )

    @classmethod
    def q2_k_serial_decoded_lds_64_hip_distributed(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_64_hip_association(),
            metadata_schedule=(
                GroupedMetadataSchedule.Q2HipAssociationPartialLdsPreNegatedDmWrite2Meta2DistributedProducer
            ),
        )

    @classmethod
    def q2_k_serial_decoded_lds_128_unrolled(cls) -> Self:
        return replace(
            cls.q2_k_serial_decoded_lds_64_unrolled(),
            macro_tile0=128,
            tail_macro_tile0=128,
            epilogue_tiles_ahead=8,
            output_store=GroupedOutputStore.BFloat16RNEClause8Masked,
        )

    @classmethod
    def q5_k_serial_decoded_lds(cls) -> Self:
        return replace(
            cls.q4_k_serial_decoded_lds(),
            packed_weight_block_bytes=GROUPED_QUANT_FORMATS["Q5_K"].block_bytes,
            weight_decode="DirectNibbleHighBit",
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds(),
            metadata_schedule=(
                GroupedMetadataSchedule.IndependentExtractionMetadataAfterLowWmma
            ),
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_a1d2p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_scheduled(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_mixed64(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_scheduled(),
            tail_macro_tile0=64,
            output_store=GroupedOutputStore.BFloat16RNEClause8Clause4MixedMasked,
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_mixed64_a1d2p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_scheduled_mixed64(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_mixed64_a1d4p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_scheduled_mixed64(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=4,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_mixed64_mixed32(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_scheduled(),
            tail_macro_tile0=32,
            output_store=(
                GroupedOutputStore.BFloat16RNEClause8Clause4Clause2MixedMasked
            ),
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_mixed64_mixed32_a1d2p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_scheduled_mixed64_mixed32(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_mixed64_mixed32_a1d4p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_scheduled_mixed64_mixed32(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=4,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_64(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds(),
            macro_tile0=64,
            tail_macro_tile0=64,
            epilogue_tiles_ahead=4,
            output_store=GroupedOutputStore.BFloat16RNEClause4Masked,
        )

    @classmethod
    def q5_k_serial_decoded_lds_64_scheduled(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_64(),
            metadata_schedule=(
                GroupedMetadataSchedule.IndependentExtractionMetadataAfterLowWmma
            ),
        )

    @classmethod
    def q5_k_serial_decoded_lds_64_scheduled_a1d2p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_64_scheduled(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_64_scheduled_a1d4p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_64_scheduled(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=4,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_64_scheduled_mixed32(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_64_scheduled(),
            tail_macro_tile0=32,
            output_store=GroupedOutputStore.BFloat16RNEClause4Clause2MixedMasked,
        )

    @classmethod
    def q5_k_serial_decoded_lds_64_scheduled_mixed32_a1d2p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_64_scheduled_mixed32(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=2,
            epilogue_priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_64_scheduled_mixed32_a1d4p2(cls) -> Self:
        return replace(
            cls.q5_k_serial_decoded_lds_64_scheduled_mixed32(),
            epilogue_tiles_ahead=1,
            epilogue_dependency_width=4,
            epilogue_priority=2,
        )

    @property
    def num_threads(self) -> int:
        return self.work_group[0] * self.work_group[1] * self.work_group[2]


@dataclass(frozen=True)
class GroupedForwardSolutionKey:
    problem: GroupedForwardProblem
    solution: GroupedForwardSolution

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {"ProblemContract", "Problem", "KernelSpec"}
    )

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _mapping(value, "GroupedForwardSolutionKey", cls._KEYS)
        from .grouped_mmq_fwd_spec import (
            GroupedForwardKernelSpec,
            GroupedForwardProblemContract,
            grouped_forward_capability_rejection_reason,
        )

        contract = GroupedForwardProblemContract.from_mapping(item["ProblemContract"])
        problem_item = _mapping(
            item["Problem"], "GroupedForwardProblem", frozenset({"aggregate_rows"})
        )
        problem = contract.problem(
            _integer(problem_item["aggregate_rows"], "aggregate_rows")
        )
        if not 0 < problem.aggregate_rows <= 0xFFFFFFFF:
            raise SchemaError("grouped aggregate_rows must fit in a positive u32")
        spec = GroupedForwardKernelSpec.from_mapping(item["KernelSpec"], contract)
        solution = spec.to_solution(contract)
        if GroupedForwardProblemContract.from_solution(problem, solution) != contract:
            raise SchemaError(
                "GroupedForwardProblemContract does not round-trip canonically"
            )
        rejection = grouped_forward_capability_rejection_reason(problem, solution)
        if rejection is not None:
            raise SchemaError(
                f"grouped kernel specification is not canonical: {rejection}"
            )
        return cls(problem, solution)

    def to_mapping(self) -> dict[str, object]:
        from .grouped_mmq_fwd_spec import (
            GroupedForwardKernelSpec,
            GroupedForwardProblemContract,
        )

        contract = GroupedForwardProblemContract.from_solution(
            self.problem, self.solution
        )
        spec = GroupedForwardKernelSpec.from_solution(self.problem, self.solution)
        return {
            "ProblemContract": contract.to_mapping(),
            "Problem": {"aggregate_rows": self.problem.aggregate_rows},
            "KernelSpec": spec.to_mapping(),
        }

    @property
    def hash(self) -> str:
        identity = {
            "ArtifactKind": "ExactKernel",
            "KernelFamily": "GroupedForward",
            **self.to_mapping(),
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"ggsol_{digest[:16]}"

    @property
    def kernel_name(self) -> str:
        problem = self.problem
        return (
            "torch_ggml_ops_ggtensile_gfx1151_v1_grouped_mmq_fwd_"
            f"{problem.quant_data_type.lower()}_r{problem.aggregate_rows}_"
            f"n{problem.output_features}_k{problem.input_features}_{self.hash[6:]}"
        )
