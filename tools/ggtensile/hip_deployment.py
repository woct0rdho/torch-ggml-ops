"""Deployed HIP control selection for the exact deployment keys.

`configs/hip_deployment.json` is the single source of truth for which
installed HIP control serves each exact key. The control's build configuration
in `tools.mmq_hip_control_spec` remains the authority for its geometry, so
a deployment entry only names a symbol and the launch configuration is derived
from the same record that built the artifact. The GGTensile catalogs under the
same `configs/` directory play the same role for the public bundle.

Keys use the module problem axes:
- `OrdinaryForward` / `OrdinaryBackward`: the `ProblemSize(m, n, k)` of the
  operation. For a backward input gradient that is `(rows, in_features,
  out_features)` because the op contracts over the forward output axis.
- `FixedGroupedForward` / `FixedGroupedBackward`: `(tokens,
  output_features, input_features)` of the fixed problem.

Only the non-routed families are keyed here. Routed grouped controls are chosen
per route bank by the grouped runtime modules, because their tile selection also
depends on the number of active route entries.
"""

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tools.mmq_bundle_wrapper_source import (
    DenseBackwardConfig,
    ForwardConfig,
    ForwardKind,
)
from tools.mmq_hip_control_spec import HIPControlSpec, hip_control_specs

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "hip_deployment.json"
SCHEMA = "hip-control-deployment-v1"
_DENSE_FORWARD_BLOCK = (32, 4, 1)
_DENSE_BACKWARD_BLOCK = (128, 1, 1)
_Q3_K_LDS_BYTES = 40_448
_J64_LDS_BYTES = 28_928
_J128_LDS_BYTES = 38_400


@lru_cache(maxsize=1)
def control_inventory() -> dict[str, HIPControlSpec]:
    """Return every built HIP control indexed by symbol."""

    return {spec.symbol: spec for spec in hip_control_specs()}


@lru_cache(maxsize=1)
def deployment_table() -> dict[tuple[str, str, int, int, int], str]:
    """Return the deployed symbol for every catalogued exact key."""

    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(
            f"{CONFIG_PATH.name} declares schema {payload.get('schema')!r}, "
            f"expected {SCHEMA!r}"
        )
    table: dict[tuple[str, str, int, int, int], str] = {}
    for family in payload["families"]:
        operation = family["operation"]
        quant_type = family["quant_type"]
        for entry in family["controls"]:
            key = (
                operation,
                quant_type,
                int(entry["m"]),
                int(entry["n"]),
                int(entry["k"]),
            )
            if key in table:
                raise ValueError(f"duplicate HIP deployment key {key}")
            table[key] = entry["symbol"]
    return table


@dataclass(frozen=True)
class HipControl:
    """One deployed HIP control and the build record behind it."""

    symbol: str
    spec: HIPControlSpec

    def launch_configuration(
        self, m: int, n: int, k: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        """Return `(grid, block, shared_bytes)` for this control."""

        config = self.spec.config
        if isinstance(config, ForwardConfig) and config.kind is ForwardKind.DENSE:
            j = config.j
            quant_name = getattr(config.quant_type, "name", None)
            if quant_name == "Q3_K":
                shared_bytes = _Q3_K_LDS_BYTES
            else:
                shared_bytes = _J64_LDS_BYTES if j == 64 else _J128_LDS_BYTES
            return ((n // 64, math.ceil(m / j), 1), _DENSE_FORWARD_BLOCK, shared_bytes)
        if isinstance(config, DenseBackwardConfig):
            m_per_block = 16 * config.m_tiles_per_wave * config.active_waves
            n_per_block = 16 * config.n_tiles
            m_blocks = math.ceil(m / m_per_block)
            n_blocks = math.ceil(n / n_per_block)
            if config.group_m:
                grid = (
                    config.group_m,
                    n_blocks,
                    math.ceil(m_blocks / config.group_m),
                )
            else:
                grid = (m_blocks, n_blocks, 1)
            return (grid, _DENSE_BACKWARD_BLOCK, 0)
        raise ValueError(
            f"{self.symbol} has no derived launch configuration; its family "
            "derives geometry from the problem and kernel spec"
        )

    def fixed_m_tile(self) -> int:
        """Return the M tile of a fixed-grouped backward control."""

        return 192 if "tuned" in self.symbol else 256


def select_hip_control(
    operation: str, quant_type: str, m: int, n: int, k: int
) -> HipControl:
    """Return the deployed control for one exact key, failing closed."""

    symbol = deployment_table().get((operation, quant_type, m, n, k))
    if symbol is None:
        raise ValueError(
            f"no deployed HIP control for {operation} {quant_type} M={m}, N={n}, K={k}"
        )
    spec = control_inventory().get(symbol)
    if spec is None:
        raise ValueError(
            f"deployed HIP control {symbol!r} is not a built control. "
            f"Rebuild the HIP control bundle or fix {CONFIG_PATH.name}"
        )
    return HipControl(symbol, spec)
