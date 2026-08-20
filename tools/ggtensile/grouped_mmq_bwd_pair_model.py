"""Strict identities for research-only paired grouped backward kernels."""

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar

from typing_extensions import Self

from .model import BackwardSolution
from .schema import SchemaError
from .schema import integer as _integer
from .schema import strict_mapping as _mapping


class GroupedBackwardPairProjectionSchedule(str, Enum):
    InterleavedDepthU = "InterleavedDepthU"
    DualLdsInterleavedDepthU = "DualLdsInterleavedDepthU"
    DualLdsGlobalCodebookInterleavedDepthU = "DualLdsGlobalCodebookInterleavedDepthU"
    DualLdsFullTileSplitInterleavedDepthU = "DualLdsFullTileSplitInterleavedDepthU"
    DualLdsFullTileSplitConcurrentReadsInterleavedDepthU = (
        "DualLdsFullTileSplitConcurrentReadsInterleavedDepthU"
    )
    DualLdsFullTileSplitConcurrentReadsPrefetchAInterleavedDepthU = (
        "DualLdsFullTileSplitConcurrentReadsPrefetchAInterleavedDepthU"
    )
    DualLdsFullTileSplitDirectPointersPrefetchAInterleavedDepthU = (
        "DualLdsFullTileSplitDirectPointersPrefetchAInterleavedDepthU"
    )
    DualLdsFullTileSplitKPipelineInterleavedDepthU = (
        "DualLdsFullTileSplitKPipelineInterleavedDepthU"
    )
    DualLdsFullTileSplitKPipelineGlobalCodebookInterleaveWmmaWaitsDepthU = (
        "DualLdsFullTileSplitKPipelineGlobalCodebookInterleaveWmmaWaitsDepthU"
    )


class GroupedBackwardPairRouteOwnership(str, Enum):
    SerialRoutes = "SerialRoutes"
    PackedSplitRoutes8 = "PackedSplitRoutes8"

    @property
    def split_factor(self) -> int:
        return {
            GroupedBackwardPairRouteOwnership.SerialRoutes: 1,
            GroupedBackwardPairRouteOwnership.PackedSplitRoutes8: 8,
        }[self]


@dataclass(frozen=True)
class GroupedBackwardPairProblem:
    """One exact two-projection routed backward sum."""

    quant_data_type: str
    aggregate_rows: int
    out_features: int
    in_features: int
    physical_experts: int
    max_route_entries: int
    projection_count: int

    @classmethod
    def iq2_s(cls, aggregate_rows: int) -> Self:
        return cls("IQ2_S", aggregate_rows, 512, 2048, 256, 256, 2)

    @classmethod
    def iq2_xxs(cls, aggregate_rows: int) -> Self:
        return cls("IQ2_XXS", aggregate_rows, 2048, 4096, 256, 256, 2)


@dataclass(frozen=True)
class GroupedBackwardPairSolution:
    """Compute and ownership identity for one fused backward pair."""

    compute: BackwardSolution
    projection_schedule: GroupedBackwardPairProjectionSchedule
    route_ownership: GroupedBackwardPairRouteOwnership
    projection_count: int

    @classmethod
    def iq2_s_m64_n64(cls) -> Self:
        compute = replace(
            BackwardSolution.pilot(),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=64,
            macro_tile1=64,
        )
        return cls(
            compute,
            GroupedBackwardPairProjectionSchedule.InterleavedDepthU,
            GroupedBackwardPairRouteOwnership.SerialRoutes,
            2,
        )

    @classmethod
    def iq2_s_m128_n64(cls) -> Self:
        return replace(
            cls.iq2_s_m64_n64(),
            compute=replace(
                cls.iq2_s_m64_n64().compute,
                matrix_instruction=(16, 16, 16, 1, 1, 2, 4, 4, 1),
                macro_tile0=128,
            ),
        )

    @classmethod
    def iq2_xxs_m64_n64(cls) -> Self:
        compute = replace(
            BackwardSolution.pilot(),
            matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
            macro_tile0=64,
            macro_tile1=64,
            lds_swizzle_chunk_b=4,
        )
        return cls(
            compute,
            GroupedBackwardPairProjectionSchedule.InterleavedDepthU,
            GroupedBackwardPairRouteOwnership.SerialRoutes,
            2,
        )

    @classmethod
    def iq2_xxs_m64_n64_dual_lds_full_tile_split_concurrent_reads_prefetch_a(
        cls,
    ) -> Self:
        parent = cls.iq2_xxs_m64_n64()
        return replace(
            parent,
            compute=replace(
                parent.compute,
                prefetch_global_read=2,
                schedule_iter_alg=4,
                lds_swizzle_chunk_b=8,
            ),
            projection_schedule=(
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsPrefetchAInterleavedDepthU
            ),
        )

    @classmethod
    def iq2_xxs_m64_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a(
        cls,
    ) -> Self:
        parent = (
            cls.iq2_xxs_m64_n64_dual_lds_full_tile_split_concurrent_reads_prefetch_a()
        )
        return replace(
            parent,
            projection_schedule=(
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchAInterleavedDepthU
            ),
        )

    @classmethod
    def iq2_xxs_m64_n64_sia5_dual_lds_full_tile_split_k_pipeline(cls) -> Self:
        parent = (
            cls.iq2_xxs_m64_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a()
        )
        return replace(
            parent,
            compute=replace(
                parent.compute,
                schedule_iter_alg=5,
                prefetch_packed_weight_next=True,
            ),
            projection_schedule=(
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineInterleavedDepthU
            ),
        )

    @classmethod
    def iq2_xxs_m128_n64(cls) -> Self:
        parent = cls.iq2_xxs_m64_n64()
        return replace(
            parent,
            compute=replace(
                parent.compute,
                matrix_instruction=(16, 16, 16, 1, 1, 2, 4, 4, 1),
                macro_tile0=128,
            ),
        )

    @classmethod
    def iq2_xxs_m128_n64_dual_lds(cls) -> Self:
        return replace(
            cls.iq2_xxs_m128_n64(),
            projection_schedule=(
                GroupedBackwardPairProjectionSchedule.DualLdsInterleavedDepthU
            ),
        )

    @classmethod
    def iq2_xxs_m128_n64_dual_lds_full_tile_split(cls) -> Self:
        return replace(
            cls.iq2_xxs_m128_n64(),
            projection_schedule=(
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitInterleavedDepthU
            ),
        )

    @classmethod
    def iq2_xxs_m128_n64_dual_lds_full_tile_split_concurrent_reads(cls) -> Self:
        return replace(
            cls.iq2_xxs_m128_n64(),
            projection_schedule=(
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsInterleavedDepthU
            ),
        )

    @classmethod
    def iq2_xxs_m128_n64_dual_lds_full_tile_split_concurrent_reads_prefetch_a(
        cls,
    ) -> Self:
        parent = cls.iq2_xxs_m128_n64()
        return replace(
            parent,
            compute=replace(
                parent.compute,
                prefetch_global_read=2,
                schedule_iter_alg=4,
            ),
            projection_schedule=(
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsPrefetchAInterleavedDepthU
            ),
        )

    @classmethod
    def iq2_s_m128_n64_dual_lds(cls) -> Self:
        return replace(
            cls.iq2_s_m128_n64(),
            projection_schedule=(
                GroupedBackwardPairProjectionSchedule.DualLdsInterleavedDepthU
            ),
        )

    @classmethod
    def iq2_s_m128_n64_dual_lds_global_codebook(cls) -> Self:
        return replace(
            cls.iq2_s_m128_n64(),
            projection_schedule=(
                GroupedBackwardPairProjectionSchedule.DualLdsGlobalCodebookInterleavedDepthU
            ),
        )

    @classmethod
    def iq2_s_m128_n64_dual_lds_full_tile_split(cls) -> Self:
        return replace(
            cls.iq2_s_m128_n64(),
            projection_schedule=(
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitInterleavedDepthU
            ),
        )

    @classmethod
    def iq2_s_m128_n64_dual_lds_full_tile_split_concurrent_reads(cls) -> Self:
        return replace(
            cls.iq2_s_m128_n64(),
            projection_schedule=(
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsInterleavedDepthU
            ),
        )

    @classmethod
    def iq2_s_m128_n64_dual_lds_full_tile_split_concurrent_reads_prefetch_a(
        cls,
    ) -> Self:
        parent = cls.iq2_s_m128_n64()
        return replace(
            parent,
            compute=replace(
                parent.compute,
                prefetch_global_read=2,
                schedule_iter_alg=4,
            ),
            projection_schedule=(
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitConcurrentReadsPrefetchAInterleavedDepthU
            ),
        )

    @classmethod
    def iq2_s_m128_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a(
        cls,
    ) -> Self:
        parent = cls.iq2_s_m128_n64()
        return replace(
            parent,
            compute=replace(
                parent.compute,
                prefetch_global_read=2,
                schedule_iter_alg=4,
            ),
            projection_schedule=(
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitDirectPointersPrefetchAInterleavedDepthU
            ),
        )

    @classmethod
    def iq2_s_m128_n64_dual_lds_full_tile_split_k_pipeline(cls) -> Self:
        parent = cls.iq2_s_m128_n64()
        return replace(
            parent,
            compute=replace(
                parent.compute,
                prefetch_global_read=2,
                schedule_iter_alg=4,
                prefetch_packed_weight_next=True,
            ),
            projection_schedule=(
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineInterleavedDepthU
            ),
        )

    @classmethod
    def iq2_s_m128_n64_sia5_global_codebook_packed_split_routes_8(cls) -> Self:
        parent = cls.iq2_s_m128_n64_dual_lds_full_tile_split_k_pipeline()
        return replace(
            parent,
            compute=replace(parent.compute, schedule_iter_alg=5),
            projection_schedule=(
                GroupedBackwardPairProjectionSchedule.DualLdsFullTileSplitKPipelineGlobalCodebookInterleaveWmmaWaitsDepthU
            ),
            route_ownership=GroupedBackwardPairRouteOwnership.PackedSplitRoutes8,
        )

    @property
    def num_threads(self) -> int:
        return self.compute.num_threads

    @property
    def macro_tile0(self) -> int:
        return self.compute.macro_tile0

    @property
    def macro_tile1(self) -> int:
        return self.compute.macro_tile1


@dataclass(frozen=True)
class GroupedBackwardPairSolutionKey:
    problem: GroupedBackwardPairProblem
    solution: GroupedBackwardPairSolution

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {"ProblemContract", "Problem", "KernelSpec"}
    )

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _mapping(value, "GroupedBackwardPairSolutionKey", cls._KEYS)
        from .grouped_mmq_bwd_pair_spec import (
            GroupedBackwardPairContract,
            GroupedBackwardPairKernelSpec,
            grouped_backward_pair_capability_rejection_reason,
        )

        contract = GroupedBackwardPairContract.from_mapping(item["ProblemContract"])
        problem_item = _mapping(
            item["Problem"],
            "GroupedBackwardPairProblem",
            frozenset({"aggregate_rows"}),
        )
        problem = contract.problem(
            _integer(problem_item["aggregate_rows"], "aggregate_rows")
        )
        if not 0 < problem.aggregate_rows <= 0xFFFFFFFF:
            raise SchemaError("paired backward aggregate_rows must fit in a u32")
        spec = GroupedBackwardPairKernelSpec.from_mapping(item["KernelSpec"], contract)
        solution = spec.to_solution(contract)
        if GroupedBackwardPairContract.from_solution(problem, solution) != contract:
            raise SchemaError("paired backward contract does not round-trip")
        rejection = grouped_backward_pair_capability_rejection_reason(problem, solution)
        if rejection is not None:
            raise SchemaError(
                f"paired backward kernel specification is not canonical: {rejection}"
            )
        return cls(problem, solution)

    def to_mapping(self) -> dict[str, object]:
        from .grouped_mmq_bwd_pair_spec import (
            GroupedBackwardPairContract,
            GroupedBackwardPairKernelSpec,
        )

        contract = GroupedBackwardPairContract.from_solution(
            self.problem, self.solution
        )
        spec = GroupedBackwardPairKernelSpec.from_solution(self.solution)
        return {
            "ProblemContract": contract.to_mapping(),
            "Problem": {"aggregate_rows": self.problem.aggregate_rows},
            "KernelSpec": spec.to_mapping(contract),
        }

    @property
    def hash(self) -> str:
        identity = {
            "ArtifactKind": "ExactKernel",
            "KernelFamily": "GroupedBackwardPair",
            **self.to_mapping(),
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"ggbpair_{digest[:16]}"

    @property
    def kernel_name(self) -> str:
        problem = self.problem
        return (
            "torch_ggml_ops_ggtensile_gfx1151_v1_grouped_mmq_bwd_pair_"
            f"{problem.quant_data_type.lower()}_r{problem.aggregate_rows}_"
            f"n{problem.in_features}_k{problem.out_features}_{self.hash[8:]}"
        )
