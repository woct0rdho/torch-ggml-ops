from pathlib import Path

import pytest
import torch

from tools.ggtensile.grouped_mmq_bwd_runtime import (
    InstalledGroupedBackwardControl,
    _control_path,
)
from tools.ggtensile.runtime import HIPRuntimeError, _resolve_code_object
from tools.mmq_hip_control_spec import hip_control_specs


@pytest.mark.parametrize(
    ("quant_type", "rows", "route_entries", "name"),
    (
        ("Q2_K", 1_000, 8, "mt64_nt64"),
        ("Q2_K", 2_000, 8, "mt128_nt64_u2"),
        ("Q2_K", 5_000, 8, "mt128_nt64"),
        ("Q4_K", 1_000, 16, "mt64_nt64"),
        ("Q4_K", 1_000, 8, "mt128_nt64"),
        ("Q5_K", 100_000, 1, "mt64_nt64"),
        ("IQ2_S", 500, 8, "mt64_nt64"),
        ("IQ2_S", 1_000, 8, "mt128_nt64"),
    ),
)
def test_standalone_backward_control_dispatch(
    quant_type: str, rows: int, route_entries: int, name: str
) -> None:
    spec = InstalledGroupedBackwardControl.select_spec(quant_type, rows, route_entries)
    assert spec.out_features > 0
    assert spec.in_features > 0
    assert spec.packed_row_bytes > 0
    assert spec.symbol.endswith(name)


@pytest.mark.parametrize(
    ("quant_type", "rows", "route_entries"),
    (("Q2_K", 0, 1), ("Q4_K", 1, 0), ("Q6_K", 1, 1)),
)
def test_standalone_backward_control_rejects_invalid_dispatch(
    quant_type: str, rows: int, route_entries: int
) -> None:
    with pytest.raises(HIPRuntimeError):
        InstalledGroupedBackwardControl.select_spec(quant_type, rows, route_entries)


def test_standalone_backward_control_resolves_directory_and_file(
    tmp_path: Path,
) -> None:
    symbol = "control_symbol"
    artifact = tmp_path / "hip_controls" / f"{symbol}.hsaco"
    artifact.parent.mkdir()
    artifact.write_bytes(b"test")

    assert _control_path(symbol, tmp_path) == artifact
    assert _control_path(symbol, artifact) == artifact
    assert _resolve_code_object(tmp_path, symbol) == artifact

    with pytest.raises(HIPRuntimeError, match="does not contain"):
        _resolve_code_object(tmp_path, "missing_symbol")


def test_historical_hip_control_spec_is_separate_and_complete() -> None:
    specs = hip_control_specs()
    assert len(specs) == 181
    assert len({spec.symbol for spec in specs}) == len(specs)
    assert all(
        spec.symbol.startswith("torch_ggml_ops_mmq_gfx1151_v1_") for spec in specs
    )
    assert not any("ggtensile" in spec.symbol for spec in specs)


def test_standalone_backward_control_rejects_cpu_launch() -> None:
    control = InstalledGroupedBackwardControl.__new__(InstalledGroupedBackwardControl)
    control.spec = InstalledGroupedBackwardControl.select_spec("Q4_K", 16_384, 8)
    tensors = [
        torch.empty((16, control.spec.out_features), dtype=torch.bfloat16),
        torch.empty(
            (256, control.spec.out_features, control.spec.packed_row_bytes),
            dtype=torch.uint8,
        ),
        torch.empty((16, control.spec.in_features), dtype=torch.bfloat16),
        torch.empty(8, dtype=torch.int64),
        torch.empty(8, dtype=torch.int32),
    ]

    with pytest.raises(HIPRuntimeError, match="contiguous HIP"):
        control.launch(*tensors, stream=0)
