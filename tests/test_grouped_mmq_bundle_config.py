from pathlib import Path
from typing import cast

import pytest

from tools.build_mmq_bundle import kernel_specs, render_wrapper
from tools.mmq_bundle_wrapper_source import ForwardConfig, ForwardKind, QuantType

ROOT = Path(__file__).resolve().parents[1]


def _forward_config(cpp_id: str) -> ForwardConfig:
    spec = next(spec for spec in kernel_specs() if spec.cpp_id == cpp_id)
    return cast(ForwardConfig, spec.config)


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
