from dataclasses import replace
from pathlib import Path

from tools.ggtensile.campaign import DeploymentCatalog, load_catalog
from tools.ggtensile.grouped_mmq_bwd_pair_model import (
    GroupedBackwardPairProblem,
    GroupedBackwardPairProjectionPolicy,
    GroupedBackwardPairRouteOwnership,
    PairCodebookStorage,
    PairPointerMode,
    PairReadConcurrency,
)
from tools.ggtensile.grouped_mmq_bwd_pair_spec import GroupedBackwardPairKernelSpec

_CONFIG = Path(__file__).resolve().parents[2] / "tools/ggtensile/configs"


def _catalog(quant_type: str) -> DeploymentCatalog:
    return load_catalog(
        _CONFIG / f"mmq_grouped_bwd_pair_{quant_type.lower()}_catalog.json"
    )


def _specs(catalog: DeploymentCatalog) -> tuple[GroupedBackwardPairKernelSpec, ...]:
    result = []
    seen = set()
    for entry in catalog.entries:
        instance = entry.instance
        assert isinstance(instance.problem, GroupedBackwardPairProblem)
        assert isinstance(instance.kernel_spec, GroupedBackwardPairKernelSpec)
        if instance.kernel_spec in seen:
            continue
        seen.add(instance.kernel_spec)
        result.append(instance.kernel_spec)
    return tuple(result)


_CATALOGS = {
    quant_type: _catalog(quant_type) for quant_type in ("IQ2_S", "IQ2_XXS", "Q3_K")
}

(_IQ2_S_SELECTED,) = _specs(_CATALOGS["IQ2_S"])
_IQ2_XXS_M64_SELECTED, _IQ2_XXS_M128_SELECTED = _specs(_CATALOGS["IQ2_XXS"])
_Q3_M64_SELECTED, _Q3_M128_SELECTED, _Q3_M128_OVERLAP_SELECTED = _specs(
    _CATALOGS["Q3_K"]
)


def _policy(
    *,
    dual_lds: bool = True,
    codebook_storage: PairCodebookStorage = PairCodebookStorage.Lds,
    split_full_tiles: bool = False,
    pointer_mode: PairPointerMode = PairPointerMode.Swapped,
    read_concurrency: PairReadConcurrency = PairReadConcurrency.Serial,
    activation_prefetch: bool = False,
    pipeline_k: bool = False,
) -> GroupedBackwardPairProjectionPolicy:
    return GroupedBackwardPairProjectionPolicy(
        dual_lds,
        codebook_storage,
        split_full_tiles,
        pointer_mode,
        read_concurrency,
        activation_prefetch,
        pipeline_k,
    )


_SHARED = _policy(dual_lds=False)
_DUAL = _policy()
_DUAL_GLOBAL = _policy(codebook_storage=PairCodebookStorage.Global)
_FULL = _policy(split_full_tiles=True)
_DIRECT = _policy(split_full_tiles=True, pointer_mode=PairPointerMode.Direct)
_CONCURRENT = _policy(
    split_full_tiles=True, read_concurrency=PairReadConcurrency.Concurrent
)
_CONCURRENT_PREFETCH = _policy(
    split_full_tiles=True,
    read_concurrency=PairReadConcurrency.Concurrent,
    activation_prefetch=True,
)
_DIRECT_PREFETCH = _policy(
    split_full_tiles=True,
    pointer_mode=PairPointerMode.Direct,
    read_concurrency=PairReadConcurrency.Concurrent,
    activation_prefetch=True,
)
_K_PIPELINE = _policy(
    split_full_tiles=True,
    pointer_mode=PairPointerMode.Direct,
    read_concurrency=PairReadConcurrency.Concurrent,
    activation_prefetch=True,
    pipeline_k=True,
)


def grouped_backward_pair_problem(
    quant_type: str, aggregate_rows: int
) -> GroupedBackwardPairProblem:
    problem = _CATALOGS[quant_type].entries[0].instance.problem
    assert isinstance(problem, GroupedBackwardPairProblem)
    return replace(problem, aggregate_rows=aggregate_rows)


def _compute(
    spec: GroupedBackwardPairKernelSpec, **changes: object
) -> GroupedBackwardPairKernelSpec:
    return replace(spec, compute=replace(spec.compute, **changes))


def _pipeline(
    spec: GroupedBackwardPairKernelSpec, **changes: object
) -> GroupedBackwardPairKernelSpec:
    return _compute(spec, pipeline=replace(spec.compute.pipeline, **changes))


def _memory(
    spec: GroupedBackwardPairKernelSpec, **changes: object
) -> GroupedBackwardPairKernelSpec:
    return _compute(spec, memory=replace(spec.compute.memory, **changes))


class GroupedBackwardPairTestSolutions:
    @staticmethod
    def _baseline(
        selected: GroupedBackwardPairKernelSpec,
    ) -> GroupedBackwardPairKernelSpec:
        return replace(
            _pipeline(
                selected,
                global_read_prefetch=1,
                iteration=replace(
                    selected.compute.pipeline.iteration,
                    prefetch_activation=False,
                    interleave_wmma_waits=False,
                ),
                prefetch_next_packed_weight=False,
            ),
            projection_policy=_SHARED,
            route_ownership=GroupedBackwardPairRouteOwnership.serial(),
        )

    @classmethod
    def q3_k_m64_n64(cls) -> GroupedBackwardPairKernelSpec:
        return cls._baseline(_Q3_M64_SELECTED)

    @classmethod
    def q3_k_m128_n64(cls) -> GroupedBackwardPairKernelSpec:
        return cls._baseline(_Q3_M128_SELECTED)

    @classmethod
    def q3_k_m128_n64_dual_lds(cls) -> GroupedBackwardPairKernelSpec:
        return replace(cls.q3_k_m128_n64(), projection_policy=_DUAL)

    @classmethod
    def q3_k_m128_n64_dual_lds_full_tile_split(cls) -> GroupedBackwardPairKernelSpec:
        return replace(cls.q3_k_m128_n64(), projection_policy=_FULL)

    @classmethod
    def q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers(
        cls,
    ) -> GroupedBackwardPairKernelSpec:
        return replace(cls.q3_k_m128_n64(), projection_policy=_DIRECT)

    @staticmethod
    def q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a() -> (
        GroupedBackwardPairKernelSpec
    ):
        return _Q3_M128_SELECTED

    @staticmethod
    def q3_k_m64_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a() -> (
        GroupedBackwardPairKernelSpec
    ):
        return _Q3_M64_SELECTED

    @staticmethod
    def q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers_overlap_second_read_prefetch_a() -> (
        GroupedBackwardPairKernelSpec
    ):
        return _Q3_M128_OVERLAP_SELECTED

    @classmethod
    def iq2_s_m128_n64(cls) -> GroupedBackwardPairKernelSpec:
        return cls._baseline(_IQ2_S_SELECTED)

    @classmethod
    def iq2_s_m64_n64(cls) -> GroupedBackwardPairKernelSpec:
        return _compute(
            cls.iq2_s_m128_n64(),
            geometry=replace(
                cls.iq2_s_m128_n64().compute.geometry,
                matrix_instruction=(16, 16, 16, 1, 1, 1, 4, 4, 1),
                macro_tile0=64,
            ),
        )

    @classmethod
    def iq2_s_m128_n64_dual_lds_global_codebook(cls) -> GroupedBackwardPairKernelSpec:
        return replace(cls.iq2_s_m128_n64(), projection_policy=_DUAL_GLOBAL)

    @classmethod
    def iq2_s_m128_n64_dual_lds_full_tile_split_k_pipeline(
        cls,
    ) -> GroupedBackwardPairKernelSpec:
        return replace(
            _pipeline(
                cls.iq2_s_m128_n64(),
                global_read_prefetch=2,
                iteration=replace(
                    cls.iq2_s_m128_n64().compute.pipeline.iteration,
                    prefetch_activation=True,
                    interleave_wmma_waits=False,
                ),
                prefetch_next_packed_weight=True,
            ),
            projection_policy=_K_PIPELINE,
        )

    @staticmethod
    def iq2_s_m128_n64_sia5_global_codebook_packed_split_routes_8() -> (
        GroupedBackwardPairKernelSpec
    ):
        return _IQ2_S_SELECTED

    @classmethod
    def iq2_xxs_m64_n64(cls) -> GroupedBackwardPairKernelSpec:
        return _memory(cls._baseline(_IQ2_XXS_M64_SELECTED), lds_swizzle_chunk_b=4)

    @classmethod
    def iq2_xxs_m128_n64(cls) -> GroupedBackwardPairKernelSpec:
        return _memory(cls._baseline(_IQ2_XXS_M128_SELECTED), lds_swizzle_chunk_b=4)

    @classmethod
    def iq2_xxs_m64_n64_dual_lds_full_tile_split_concurrent_reads_prefetch_a(
        cls,
    ) -> GroupedBackwardPairKernelSpec:
        return replace(
            _memory(
                _pipeline(
                    cls.iq2_xxs_m64_n64(),
                    global_read_prefetch=2,
                    iteration=replace(
                        cls.iq2_xxs_m64_n64().compute.pipeline.iteration,
                        prefetch_activation=True,
                        interleave_wmma_waits=False,
                    ),
                ),
                lds_swizzle_chunk_b=8,
            ),
            projection_policy=_CONCURRENT_PREFETCH,
        )

    @classmethod
    def iq2_xxs_m64_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a(
        cls,
    ) -> GroupedBackwardPairKernelSpec:
        return replace(
            cls.iq2_xxs_m64_n64_dual_lds_full_tile_split_concurrent_reads_prefetch_a(),
            projection_policy=_DIRECT_PREFETCH,
        )

    @staticmethod
    def iq2_xxs_m64_n64_sia5_dual_lds_full_tile_split_k_pipeline() -> (
        GroupedBackwardPairKernelSpec
    ):
        return _IQ2_XXS_M64_SELECTED

    @classmethod
    def iq2_xxs_m128_n64_dual_lds(cls) -> GroupedBackwardPairKernelSpec:
        return replace(cls.iq2_xxs_m128_n64(), projection_policy=_DUAL)

    @classmethod
    def iq2_xxs_m128_n64_dual_lds_full_tile_split(cls) -> GroupedBackwardPairKernelSpec:
        return replace(cls.iq2_xxs_m128_n64(), projection_policy=_FULL)

    @classmethod
    def iq2_xxs_m128_n64_dual_lds_full_tile_split_concurrent_reads(
        cls,
    ) -> GroupedBackwardPairKernelSpec:
        return replace(cls.iq2_xxs_m128_n64(), projection_policy=_CONCURRENT)

    @classmethod
    def iq2_xxs_m128_n64_dual_lds_full_tile_split_concurrent_reads_prefetch_a(
        cls,
    ) -> GroupedBackwardPairKernelSpec:
        return replace(
            _pipeline(
                cls.iq2_xxs_m128_n64(),
                global_read_prefetch=2,
                iteration=replace(
                    cls.iq2_xxs_m128_n64().compute.pipeline.iteration,
                    prefetch_activation=True,
                    interleave_wmma_waits=False,
                ),
            ),
            projection_policy=_CONCURRENT_PREFETCH,
        )

    @staticmethod
    def iq2_xxs_m128_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a() -> (
        GroupedBackwardPairKernelSpec
    ):
        return _IQ2_XXS_M128_SELECTED

    @staticmethod
    def iq2_xxs_m192_n64_dual_lds_full_tile_split_direct_pointers_overlap_second_read_prefetch_a() -> (
        GroupedBackwardPairKernelSpec
    ):
        parent = _IQ2_XXS_M128_SELECTED
        return replace(
            _compute(
                parent,
                geometry=replace(
                    parent.compute.geometry,
                    matrix_instruction=(16, 16, 16, 1, 1, 3, 4, 4, 1),
                    macro_tile0=192,
                ),
            ),
            projection_policy=replace(
                parent.projection_policy,
                read_concurrency=PairReadConcurrency.OverlapSecond,
            ),
        )
