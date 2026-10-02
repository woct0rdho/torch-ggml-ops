"""Construct deployed HIP controls from catalog choices.

`tools.mmq_hip_deployment` owns *which* body serves a problem. This module owns
*how* one named body is constructed, so the benchmark, the deployment runner,
and the tests share one construction path for every problem type.
"""

from pathlib import Path

from tools.mmq_hip_deployment import RoutedControl
from tools.mmq_hip_grouped_bwd import (
    InstalledGroupedBackwardControl,
    InstalledGroupedBackwardRowTaskControl,
)
from tools.mmq_hip_grouped_pair_bwd import (
    InstalledGroupedBackwardPairIQ2SControl,
    InstalledGroupedBackwardPairIQ2XXSControl,
    InstalledGroupedBackwardPairQ3KControl,
)
from tools.mmq_hip_grouped_pair_fwd import (
    InstalledGroupedForwardPairIQ2XXSSerialControl,
    InstalledGroupedForwardPairQ3RowTaskControl,
    InstalledGroupedForwardPairQ3SerialControl,
    InstalledGroupedForwardPairRowTaskControl,
    InstalledGroupedForwardPairSerialControl,
)
from tools.mmq_runtime import (
    FixedQ81F16D2S6QuantizerModule,
    FixedQ81F16D4S4QuantizerModule,
    FixedQ81F32D4QuantizerModule,
)

ROW_TASK_PREFIXES = ("grouped_bwd_row_task_", "grouped_fwd_row_task_")

_PAIR_FORWARD_CONTROLS = {
    ("IQ2_S", True): InstalledGroupedForwardPairRowTaskControl,
    ("IQ2_S", False): InstalledGroupedForwardPairSerialControl,
    ("Q3_K", True): InstalledGroupedForwardPairQ3RowTaskControl,
    ("Q3_K", False): InstalledGroupedForwardPairQ3SerialControl,
}

_PAIR_BACKWARD_CONTROLS = {
    "Q3_K": InstalledGroupedBackwardPairQ3KControl,
    "IQ2_S": InstalledGroupedBackwardPairIQ2SControl,
    "IQ2_XXS": InstalledGroupedBackwardPairIQ2XXSControl,
}


def quantizer_module(operation: str, quant_type: str):
    """Return the installed quantizer module that feeds one deployed case."""

    if quant_type == "Q2_K":
        return FixedQ81F16D2S6QuantizerModule
    if quant_type in {"Q4_K", "Q5_K"} and operation in {
        "OrdinaryForward",
        "GroupedForward",
    }:
        return FixedQ81F16D4S4QuantizerModule
    return FixedQ81F32D4QuantizerModule


def is_row_task_body(symbol: str) -> bool:
    """Return whether a catalogued symbol names a device row-task body."""

    return symbol.startswith(ROW_TASK_PREFIXES)


def build_forward_pair_control(choice: RoutedControl, root: Path | None):
    """Build the paired grouped-forward control named by one catalog choice."""

    if choice.quant_type == "IQ2_XXS":
        if choice.tile is None:
            raise ValueError(
                f"paired grouped-forward control {choice.symbol!r} needs a J tile"
            )
        return InstalledGroupedForwardPairIQ2XXSSerialControl(choice.tile, root)
    control = _PAIR_FORWARD_CONTROLS.get(
        (choice.quant_type, is_row_task_body(choice.symbol))
    )
    if control is None:
        raise ValueError(f"no paired grouped-forward HIP control for {choice.symbol!r}")
    return control(root)


def build_backward_control(choice: RoutedControl, root: Path | None):
    """Build the routed single-projection grouped-backward control."""

    if is_row_task_body(choice.symbol):
        return InstalledGroupedBackwardRowTaskControl(choice.symbol, root)
    return InstalledGroupedBackwardControl(choice.symbol, root)


def build_backward_pair_control(choice: RoutedControl, root: Path | None):
    """Build the paired grouped-backward control named by one catalog choice."""

    control = _PAIR_BACKWARD_CONTROLS.get(choice.quant_type)
    if control is None:
        raise ValueError(
            f"no paired grouped-backward HIP control for {choice.quant_type}"
        )
    if choice.tile is None:
        raise ValueError(
            f"paired grouped-backward control {choice.symbol!r} needs an M tile"
        )
    return control(choice.tile, root)
