from collections.abc import Callable
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import cast

from tools.ggtensile.campaign import load_catalog, unique_kernel_specs
from tools.ggtensile.mmq_fwd_spec import (
    DecodedLdsForwardDecodePolicy,
    DecodeSpec,
    EpiloguePipelineSpec,
    FixedForwardDecodePolicy,
    ForwardActivationStaging,
    ForwardKernelSpec,
    ForwardWeightStaging,
    GeometrySpec,
    GlobalMemorySpec,
    LdsSpec,
    OwnershipSpec,
    Q3FullForwardDecodePolicy,
    SemanticSchedulePolicy,
)

_CONFIG = Path(__file__).resolve().parents[2] / "tools/ggtensile/configs"


@cache
def _catalog_specs(quant_type: str) -> tuple[ForwardKernelSpec, ...]:
    catalog = load_catalog(_CONFIG / f"mmq_fwd_{quant_type.lower()}_catalog.json")
    specs = unique_kernel_specs(catalog)
    assert all(isinstance(spec, ForwardKernelSpec) for spec in specs)
    return tuple(cast(ForwardKernelSpec, spec) for spec in specs)


def _find_spec(
    quant_type: str, predicate: Callable[[ForwardKernelSpec], bool]
) -> ForwardKernelSpec:
    for spec in _catalog_specs(quant_type):
        if predicate(spec):
            return spec
    raise AssertionError(f"no {quant_type} catalog specification matched")


def _q3_full_catalog_spec() -> ForwardKernelSpec:
    return _find_spec(
        "Q3_K",
        lambda spec: spec.global_memory.weight_staging == "Q3FullWeightTiledLds",
    )


def q3_hip_tiled_lds_kernel_spec() -> ForwardKernelSpec:
    base = _q3_full_catalog_spec()
    return replace(
        base,
        global_memory=GlobalMemorySpec(
            ForwardWeightStaging.Q3HipTiledLds,
            ForwardActivationStaging.MadU24,
            None,
        ),
        lds=LdsSpec("Q3HalfTile"),
        decode=DecodeSpec("Float16DToFloat32Signed6Scale", FixedForwardDecodePolicy()),
    )


def q3_full_weight_tiled_lds_kernel_spec(
    *, decode_ready_frontier: bool = True
) -> ForwardKernelSpec:
    base = _q3_full_catalog_spec()
    if decode_ready_frontier:
        return base
    return replace(
        base,
        decode=DecodeSpec(
            base.decode.metadata_conversion,
            Q3FullForwardDecodePolicy(False),
        ),
    )


def _decoded_weight_spec(
    quant_type: str,
    *,
    epilogue_tiles_ahead: int,
    epilogue_dependency_width: int,
    epilogue_priority: int,
    defer_metadata_reads: bool,
    accumulator_initialization: str,
) -> ForwardKernelSpec:
    base = _find_spec(
        quant_type,
        lambda spec: spec.global_memory.weight_staging == "DecodedWeightLdsBatch8",
    )
    for spec in _catalog_specs(quant_type):
        pipeline = spec.epilogue.pipeline
        policy = spec.decode.policy
        if (
            pipeline is not None
            and pipeline.tiles_ahead == epilogue_tiles_ahead
            and pipeline.dependency_width == epilogue_dependency_width
            and pipeline.priority == epilogue_priority
            and isinstance(policy, DecodedLdsForwardDecodePolicy)
            and policy.defer_metadata_reads == defer_metadata_reads
            and spec.instruction_policy.accumulator_initialization
            == accumulator_initialization
        ):
            return spec
    policy = base.decode.policy
    assert isinstance(policy, DecodedLdsForwardDecodePolicy)
    return replace(
        base,
        decode=replace(
            base.decode,
            policy=DecodedLdsForwardDecodePolicy(
                policy.independent_metadata_extraction,
                defer_metadata_reads,
            ),
        ),
        epilogue=replace(
            base.epilogue,
            pipeline=EpiloguePipelineSpec(
                epilogue_tiles_ahead,
                epilogue_dependency_width,
                epilogue_priority,
                None,
            ),
        ),
        instruction_policy=replace(
            base.instruction_policy,
            accumulator_initialization=accumulator_initialization,
        ),
    )


def q4_decoded_weight_lds_kernel_spec(
    *,
    epilogue_tiles_ahead: int,
    epilogue_dependency_width: int,
    epilogue_priority: int,
    defer_metadata_reads: bool = False,
) -> ForwardKernelSpec:
    return _decoded_weight_spec(
        "Q4_K",
        epilogue_tiles_ahead=epilogue_tiles_ahead,
        epilogue_dependency_width=epilogue_dependency_width,
        epilogue_priority=epilogue_priority,
        defer_metadata_reads=defer_metadata_reads,
        accumulator_initialization="ScalarCopy",
    )


def q5_decoded_weight_lds_kernel_spec(
    *,
    epilogue_tiles_ahead: int,
    epilogue_dependency_width: int,
    epilogue_priority: int,
    accumulator_initialization: str = "ScalarCopy",
) -> ForwardKernelSpec:
    return _decoded_weight_spec(
        "Q5_K",
        epilogue_tiles_ahead=epilogue_tiles_ahead,
        epilogue_dependency_width=epilogue_dependency_width,
        epilogue_priority=epilogue_priority,
        defer_metadata_reads=True,
        accumulator_initialization=accumulator_initialization,
    )


def q6_structured_kernel_spec(macro_tile0: int) -> ForwardKernelSpec:
    if macro_tile0 in (64, 128):
        return _find_spec("Q6_K", lambda spec: spec.macro_tile == (macro_tile0, 64))
    if macro_tile0 == 256:
        base = q6_structured_kernel_spec(128)
        return replace(
            base,
            geometry=replace(base.geometry, work_group=(32, 8, 1)),
            ownership=OwnershipSpec((8, 1), (2, 4)),
            instruction_policy=replace(
                base.instruction_policy, dependency_delay_mode="Explicit"
            ),
            semantic_schedule=SemanticSchedulePolicy.structured_q6(),
        )
    raise AssertionError


def q8_direct_global_kernel_spec() -> ForwardKernelSpec:
    base = _find_spec(
        "Q8_0",
        lambda spec: spec.global_memory.weight_staging == "Q8HipTiledLds",
    )
    return replace(
        base,
        geometry=GeometrySpec((32, 1, 1), (16, 16, 16, 1), 32),
        ownership=OwnershipSpec((1, 1), (1, 1)),
        global_memory=GlobalMemorySpec(
            ForwardWeightStaging.Q8DirectGlobal,
            ForwardActivationStaging.MultiplyAdd,
            None,
        ),
        lds=LdsSpec("None"),
        decode=DecodeSpec("Float16DToFloat32", FixedForwardDecodePolicy()),
    )


def q8_register_tiled_kernel_spec(
    *, wave_tile_m: int = 2, wave_tile_n: int = 2
) -> ForwardKernelSpec:
    assert wave_tile_m * wave_tile_n == 4
    base = _find_spec(
        "Q8_0",
        lambda spec: spec.global_memory.weight_staging == "Q8HipTiledLds",
    )
    return replace(
        base,
        geometry=GeometrySpec((32, 4, 1), (16, 16, 16, 1), 32),
        ownership=OwnershipSpec((4, 1), (wave_tile_m, wave_tile_n)),
        global_memory=GlobalMemorySpec(
            ForwardWeightStaging.Q8RegisterTiled,
            ForwardActivationStaging.MultiplyAdd,
            None,
        ),
        lds=LdsSpec("None"),
        decode=DecodeSpec("Float16DToFloat32", FixedForwardDecodePolicy()),
    )


def q8_hip_tiled_lds_kernel_spec(depth_u: int = 32) -> ForwardKernelSpec:
    base = _find_spec(
        "Q8_0",
        lambda spec: (
            spec.global_memory.weight_staging == "Q8HipTiledLds"
            and spec.lds.address_hoist == "HipTile"
        ),
    )
    assert depth_u in (32, 64)
    return replace(base, geometry=replace(base.geometry, depth_u=depth_u))


def q8_small_m_tiled_lds_kernel_spec(macro_tile0: int) -> ForwardKernelSpec:
    assert macro_tile0 in (32, 64)
    return _find_spec(
        "Q8_0",
        lambda spec: (
            spec.macro_tile == (macro_tile0, 64)
            and spec.lds.address_hoist == "SmallMTile"
        ),
    )


def q8_compact_depth32_tiled_lds_kernel_spec(
    macro_tile0: int = 128,
) -> ForwardKernelSpec:
    if macro_tile0 == 128:
        return _find_spec(
            "Q8_0",
            lambda spec: spec.lds.address_hoist == "CompactDepth32WeightRows",
        )
    base = q8_small_m_tiled_lds_kernel_spec(macro_tile0)
    return replace(base, lds=LdsSpec("CompactDepth32WeightRows"))


class OrdinaryForwardTestSolutions:
    @staticmethod
    def q4_k_pilot() -> ForwardKernelSpec:
        base = q8_direct_global_kernel_spec()
        return replace(
            base,
            global_memory=GlobalMemorySpec(
                ForwardWeightStaging.Global,
                ForwardActivationStaging.MultiplyAdd,
                None,
            ),
            lds=LdsSpec("None"),
            decode=DecodeSpec("Float32ThenFloat16", base.decode.policy),
        )

    @staticmethod
    def q4_k_decoded_weight_lds_metadata_after_low_wmma() -> ForwardKernelSpec:
        base = q4_decoded_weight_lds_kernel_spec(
            epilogue_tiles_ahead=8,
            epilogue_dependency_width=1,
            epilogue_priority=0,
            defer_metadata_reads=True,
        )
        return replace(
            base,
            decode=replace(
                base.decode,
                policy=DecodedLdsForwardDecodePolicy(False, True),
            ),
        )

    @staticmethod
    def q5_k_decoded_weight_lds_metadata_after_low_wmma() -> ForwardKernelSpec:
        base = q5_decoded_weight_lds_kernel_spec(
            epilogue_tiles_ahead=8,
            epilogue_dependency_width=1,
            epilogue_priority=0,
        )
        return replace(
            base,
            decode=replace(
                base.decode,
                policy=DecodedLdsForwardDecodePolicy(False, True),
            ),
        )

    @staticmethod
    def q8_0_kv_tiled_lds() -> ForwardKernelSpec:
        return q8_compact_depth32_tiled_lds_kernel_spec(64)
