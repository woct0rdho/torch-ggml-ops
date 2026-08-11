from pathlib import Path
from typing import cast

import pytest

from tools.build_mmq_bundle import kernel_specs, render_wrapper
from tools.mmq_bundle_wrapper_source import (
    ForwardConfig,
    ForwardKind,
    GroupedBackwardConfig,
    GroupedBackwardKind,
    QuantType,
)

ROOT = Path(__file__).resolve().parents[1]


def _forward_config(cpp_id: str) -> ForwardConfig:
    spec = next(spec for spec in kernel_specs() if spec.cpp_id == cpp_id)
    return cast(ForwardConfig, spec.config)


def _grouped_backward_config(cpp_id: str) -> GroupedBackwardConfig:
    spec = next(spec for spec in kernel_specs() if spec.cpp_id == cpp_id)
    return cast(GroupedBackwardConfig, spec.config)


def test_q4_mixed_tail_config_is_bounded_to_qualified_rows() -> None:
    config = _forward_config("GroupedFwdSerialQ4KN2048K512J64")
    assert config.mixed_j32_tails
    assert config.mixed_j32_rows == (16384, 65536)
    source = render_wrapper("qualified_q4", config)
    assert (
        "true,\n        false,\n        false,\n        16384,\n        65536>"
        in source
    )


def test_iq2_s_mixed_tail_config_remains_unbounded() -> None:
    config = _forward_config("GroupedFwdSerialIQ2SN2048K512J64J32")
    assert config.mixed_j32_tails
    assert config.mixed_j32_rows == (0, 0)
    source = render_wrapper("qualified_iq2_s", config)
    assert "true,\n        false,\n        false>" in source


@pytest.mark.parametrize(
    "mixed_j32_tails,j,mixed_j32_rows",
    (
        (False, 64, (16384, 65536)),
        (True, 64, (0, 65536)),
        (True, 64, (65536, 16384)),
        (True, 32, (16384, 65536)),
    ),
)
def test_mixed_j32_tail_config_rejects_inactive_or_invalid_values(
    mixed_j32_tails: bool,
    j: int,
    mixed_j32_rows: tuple[int, int],
) -> None:
    with pytest.raises(ValueError):
        ForwardConfig(
            kind=ForwardKind.GROUPED_SERIAL,
            quant_type=QuantType.Q4_K,
            j=j,
            nrows_weight=2048,
            blocks_per_weight_row=2,
            mixed_j32_tails=mixed_j32_tails,
            mixed_j32_rows=mixed_j32_rows,
        )


def test_nonaligned_full_activation_load_has_a_guarded_final_iteration() -> None:
    source = (ROOT / "csrc/mmq_core.cuh").read_text()
    assert "static_assert(tile_ints % MMQ_NTHREADS != 0);" in source
    assert "tile_y[l] = l < tile_ints ? activation[l] : 0;" in source
    assert source.count("load_grouped_nonaligned_full_activation_tile<J>(") == 4


def test_deepseek_q2_b4_uses_the_existing_mixed_tail_kernel() -> None:
    source = (ROOT / "csrc/mmq_bundle.cpp").read_text()
    assert "constexpr int kQualifiedDeepSeekQ2KDownB4Rows = 49152;" in source
    assert "rows == kQualifiedDeepSeekQ2KDownB4Rows || rows < num_groups * 64" in source


def test_deepseek_iq2_xxs_pair_uses_m128_at_qualified_route_rows() -> None:
    config = _grouped_backward_config("GroupedBwdTunedPairIQ2XXSN2048K4096M128N64")
    assert (
        config.kind.value,
        config.quant_type.value if config.quant_type is not None else None,
        config.n_tiles,
        config.m_tiles_per_wave,
        config.reduction_unroll,
    ) == (
        GroupedBackwardKind.TUNED_DEEPSEEK_PAIR.value,
        QuantType.IQ2_XXS.value,
        4,
        2,
        1,
    )
    source = render_wrapper("qualified_iq2_xxs_pair", config)
    assert "GGML_TYPE_IQ2_XXS, 2048, 4096, 16, 2, true>" in source
    dispatch = (ROOT / "csrc/mmq_bundle.cpp").read_text()
    assert "constexpr int kQualifiedDeepSeekIQ2XXSPairB4Rows = 49152;" in dispatch
    assert "constexpr int kQualifiedDeepSeekIQ2XXSPairB16Rows = 196608;" in dispatch
    assert "rows == kQualifiedDeepSeekIQ2XXSPairB4Rows ||" in dispatch
    assert "rows == kQualifiedDeepSeekIQ2XXSPairB16Rows" in dispatch


def test_deepseek_fixed_q8_uses_m192_at_b16() -> None:
    config = _grouped_backward_config("GroupedBwdTunedFixedQ80G8K4096M192N64")
    assert (
        config.kind.value,
        config.quant_type,
        config.n_tiles,
        config.m_tiles_per_wave,
        config.reduction_unroll,
    ) == (
        GroupedBackwardKind.TUNED_FIXED_Q8_0.value,
        None,
        4,
        3,
        1,
    )
    source = render_wrapper("qualified_fixed_q8", config)
    assert "grad_input_tiled_body<\n        3, 16, 0>" in source
    dispatch = (ROOT / "csrc/mmq_bundle.cpp").read_text()
    assert "constexpr int kQualifiedDeepSeekFixedB16Tokens = 32768;" in dispatch
    assert "tokens == kQualifiedDeepSeekFixedB16Tokens" in dispatch
    assert "MMQKernelId::GroupedBwdTunedFixedQ80G8K4096M192N64" in dispatch


@pytest.mark.parametrize(
    "kind,quant_type,n_tiles,m_tiles_per_wave,reduction_unroll",
    (
        (GroupedBackwardKind.IQ2_XXS_PAIR_M64, None, 4, 2, 1),
        (GroupedBackwardKind.TUNED_DEEPSEEK_PAIR, QuantType.IQ2_XXS, 4, 3, 1),
        (GroupedBackwardKind.TUNED_DEEPSEEK_PAIR, QuantType.Q2_K, 4, 2, 1),
        (GroupedBackwardKind.TUNED_FIXED_Q8_0, None, 4, 2, 1),
        (GroupedBackwardKind.TUNED_FIXED_Q8_0, QuantType.Q8_0, 4, 3, 1),
    ),
)
def test_grouped_backward_geometry_rejects_invalid_state(
    kind: GroupedBackwardKind,
    quant_type: QuantType | None,
    n_tiles: int,
    m_tiles_per_wave: int,
    reduction_unroll: int,
) -> None:
    with pytest.raises(ValueError):
        GroupedBackwardConfig(
            kind=kind,
            quant_type=quant_type,
            n_tiles=n_tiles,
            m_tiles_per_wave=m_tiles_per_wave,
            reduction_unroll=reduction_unroll,
        )


def test_production_bundle_contains_only_the_two_qualified_bwd_additions() -> None:
    specs = kernel_specs()
    assert len(specs) == 181
    assert [spec.cpp_id for spec in specs[-2:]] == [
        "GroupedBwdTunedPairIQ2XXSN2048K4096M128N64",
        "GroupedBwdTunedFixedQ80G8K4096M192N64",
    ]
    tuned_ids = {spec.cpp_id for spec in specs if "GroupedBwdTuned" in spec.cpp_id}
    assert tuned_ids == {
        "GroupedBwdTunedPairIQ2XXSN2048K4096M128N64",
        "GroupedBwdTunedFixedQ80G8K4096M192N64",
    }
