from pathlib import Path

import pytest
import torch

from tools.mmq_hip_control_spec import hip_control_specs
from tools.mmq_hip_deployment import select_routed_control
from tools.mmq_hip_grouped_bwd import (
    _SYMBOL_SPECS,
    InstalledGroupedBackwardControl,
    InstalledGroupedBackwardRowTaskControl,
)
from tools.mmq_hip_launchers import is_row_task_body
from tools.mmq_runtime import HIPRuntimeError, _resolve_code_object, find_control


@pytest.mark.parametrize(
    ("quant_type", "out_features", "in_features", "rows", "route_entries", "symbol"),
    (
        (
            "Q2_K",
            4096,
            2048,
            12_288,
            256,
            "grouped_bwd_single_q2_k_n4096_k2048_mt64_nt64",
        ),
        (
            "Q2_K",
            4096,
            2048,
            12_288,
            8,
            "grouped_bwd_single_q2_k_n4096_k2048_mt128_nt64",
        ),
        (
            "Q2_K",
            4096,
            2048,
            49_152,
            256,
            "grouped_bwd_single_q2_k_n4096_k2048_mt128_nt64_u2",
        ),
        (
            "Q2_K",
            4096,
            2048,
            196_608,
            8,
            "grouped_bwd_single_q2_k_n4096_k2048_mt128_nt64",
        ),
        ("Q4_K", 2048, 512, 16_384, 8, "grouped_bwd_single_q4_k_n2048_k512_mt128_nt64"),
        (
            "Q4_K",
            2048,
            512,
            16_384,
            256,
            "grouped_bwd_single_q4_k_n2048_k512_mt128_nt64",
        ),
        (
            "Q4_K",
            2048,
            512,
            65_536,
            8,
            "grouped_bwd_row_task_q4_k_n2048_k512_mt128_nt128",
        ),
        (
            "Q4_K",
            2048,
            512,
            262_144,
            8,
            "grouped_bwd_row_task_q4_k_n2048_k512_mt128_nt128",
        ),
        (
            "Q5_K",
            2048,
            512,
            16_384,
            8,
            "grouped_bwd_row_task_q5_k_n2048_k512_mt128_nt128",
        ),
        (
            "Q5_K",
            2048,
            512,
            262_144,
            8,
            "grouped_bwd_row_task_q5_k_n2048_k512_mt128_nt128",
        ),
        (
            "IQ2_S",
            2048,
            512,
            16_384,
            8,
            "grouped_bwd_single_iq2_s_n2048_k512_mt128_nt64",
        ),
        (
            "IQ2_S",
            2048,
            512,
            16_384,
            256,
            "grouped_bwd_single_iq2_s_n2048_k512_mt128_nt64",
        ),
        (
            "IQ2_S",
            2048,
            512,
            65_536,
            8,
            "grouped_bwd_row_task_iq2_s_n2048_k512_mt128_nt128",
        ),
        (
            "IQ2_S",
            2048,
            512,
            262_144,
            8,
            "grouped_bwd_row_task_iq2_s_n2048_k512_mt128_nt128",
        ),
    ),
)
def test_standalone_backward_control_follows_the_deployed_rules(
    quant_type: str,
    out_features: int,
    in_features: int,
    rows: int,
    route_entries: int,
    symbol: str,
) -> None:
    choice = select_routed_control(
        "GroupedBackward",
        quant_type,
        out_features,
        in_features,
        rows,
        route_entries,
    )
    assert choice.symbol == symbol
    if is_row_task_body(symbol):
        control = InstalledGroupedBackwardRowTaskControl(symbol)
        assert control.spec.symbol == symbol
        assert (control.spec.out_features, control.spec.in_features) == (
            out_features,
            in_features,
        )
        return
    spec = _SYMBOL_SPECS[symbol]
    assert (spec.out_features, spec.in_features) == (out_features, in_features)
    assert spec.packed_row_bytes > 0


@pytest.mark.parametrize(
    ("quant_type", "out_features", "in_features", "rows"),
    (
        ("Q6_K", 2048, 512, 16_384),
        ("Q4_K", 1024, 512, 16_384),
        ("Q2_K", 4096, 1024, 12_288),
    ),
)
def test_standalone_backward_control_rejects_invalid_dispatch(
    quant_type: str, out_features: int, in_features: int, rows: int
) -> None:
    with pytest.raises(ValueError, match="no deployed HIP routed control"):
        select_routed_control(
            "GroupedBackward", quant_type, out_features, in_features, rows, 256
        )
    with pytest.raises(HIPRuntimeError, match="no standalone grouped-backward control"):
        InstalledGroupedBackwardControl("not_a_control_symbol")


def test_standalone_backward_control_resolves_directory_and_file(
    tmp_path: Path,
) -> None:
    symbol = "control_symbol"
    artifact = tmp_path / "hip_controls" / f"{symbol}.hsaco"
    artifact.parent.mkdir()
    artifact.write_bytes(b"test")

    assert find_control(symbol, tmp_path) == artifact
    assert find_control(symbol, artifact) == artifact
    assert _resolve_code_object(tmp_path, symbol) == artifact

    with pytest.raises(HIPRuntimeError, match="does not contain"):
        _resolve_code_object(tmp_path, "missing_symbol")


def test_hip_control_spec_is_separate_and_complete() -> None:
    specs = hip_control_specs()
    assert len(specs) == 182
    assert len({spec.symbol for spec in specs}) == len(specs)


def test_standalone_backward_control_rejects_cpu_launch() -> None:
    control = InstalledGroupedBackwardControl.__new__(InstalledGroupedBackwardControl)
    control.spec = _SYMBOL_SPECS["grouped_bwd_single_q4_k_n2048_k512_mt64_nt64"]
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
