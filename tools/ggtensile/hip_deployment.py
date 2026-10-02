"""Deployed HIP control selection for the exact deployment keys.

`tools/configs/hip_deployment.json` is the single source of truth for which
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

The non-routed families are keyed by their exact `(operation, quant, m, n, k)`
in the catalog. The routed single-projection grouped-forward controls cannot be
keyed that way because their tile selection also depends on the number of route
entries in the bank, so their rule table lives here as well and every caller
resolves them through `select_grouped_forward_control`.
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

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "hip_deployment.json"
_DENSE_FORWARD_BLOCK = (32, 4, 1)
_DENSE_BACKWARD_BLOCK = (128, 1, 1)
_Q3_K_LDS_BYTES = 40_448
_J64_LDS_BYTES = 28_928
_J128_LDS_BYTES = 38_400
# Grouped-forward launch geometry. The activation tile holds one Q8_1 plane per
# J-tile row (32 quant ints plus four metadata ints), the weight tile holds
# MMQ_I rows of the packed SRAM layout, and the kernel reserves a J-int prefix.
_GROUPED_FORWARD_BLOCK = (32, 4, 1)
_GROUPED_FORWARD_TILE_I = 64
_GROUPED_TILE_Y_K = 36
_MMQ_SRAM_STRIDE = {
    "Q2_K": 100,
    "Q3_K": 84,
    "IQ2_S": 84,
    "Q6_K": 91,
    "Q8_0": 76,
    "IQ2_XXS": 76,
    "Q4_K": 76,
    "Q5_K": 76,
}


@lru_cache(maxsize=1)
def control_inventory() -> dict[str, HIPControlSpec]:
    """Return every built HIP control indexed by symbol."""

    return {spec.symbol: spec for spec in hip_control_specs()}


@lru_cache(maxsize=1)
def deployment_table() -> dict[tuple[str, str, int, int, int], str]:
    """Return the deployed symbol for every catalogued exact key."""

    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
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
            f"{self.symbol} has no derived launch configuration. Its family "
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


def _grouped_forward_lds_bytes(spec: HIPControlSpec) -> int:
    """Return the dynamic LDS bytes of one grouped-forward control.

    Mirrors the tile layout in `csrc/mmq_core.cuh`: a J-int prefix, a padded
    `J x 36` activation tile, and a `64 x stride` weight tile.
    """

    config = spec.config
    if (
        not isinstance(config, ForwardConfig)
        or config.kind is not ForwardKind.GROUPED_SERIAL
    ):
        raise ValueError(f"{spec.symbol} is not a routed grouped-forward control")
    if config.quant_type is None:
        raise ValueError(f"{spec.symbol} has no quant type")
    stride = _MMQ_SRAM_STRIDE[config.quant_type.name]
    tile_y = -(-config.j * _GROUPED_TILE_Y_K // 128) * 128
    return 4 * (config.j + tile_y + _GROUPED_FORWARD_TILE_I * stride)


@dataclass(frozen=True)
class GroupedForwardControl:
    """One deployed routed grouped-forward control and its launch geometry."""

    symbol: str
    spec: HIPControlSpec

    @property
    def lds_bytes(self) -> int:
        return _grouped_forward_lds_bytes(self.spec)

    def launch_configuration(
        self, out_features: int, route_entries: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        """Return `(grid, block, shared_bytes)` for one route bank."""

        if route_entries <= 0:
            raise ValueError("grouped-forward launch needs at least one route entry")
        return (
            (out_features // _GROUPED_FORWARD_TILE_I, route_entries, 1),
            _GROUPED_FORWARD_BLOCK,
            self.lds_bytes,
        )


@dataclass(frozen=True)
class _RoutedRule:
    """One ordered choice of the routed rule table."""

    when: str
    value: int
    symbol: str


def _rows_equal(rows: int, route_entries: int, value: int) -> bool:
    return rows == value


def _rows_below_per_entry(rows: int, route_entries: int, value: int) -> bool:
    return rows < value * route_entries


def _always(rows: int, route_entries: int, value: int) -> bool:
    return True


_ROUTED_PREDICATES = {
    "rows_eq": _rows_equal,
    "rows_lt_per_entry": _rows_below_per_entry,
    "always": _always,
}

# First match wins, and every family ends with the unconditional default.
_GROUPED_FORWARD_RULES: dict[tuple[str, int, int], tuple[_RoutedRule, ...]] = {
    ("Q4_K", 2048, 512): (
        _RoutedRule("rows_lt_per_entry", 128, "grouped_fwd_serial_q4_k_n2048_k512_j32"),
        _RoutedRule("always", 0, "grouped_fwd_serial_q4_k_n2048_k512_j64"),
    ),
    ("Q5_K", 2048, 512): (
        _RoutedRule("rows_lt_per_entry", 128, "grouped_fwd_serial_q5_k_n2048_k512_j32"),
        _RoutedRule("always", 0, "grouped_fwd_serial_q5_k_n2048_k512_j64"),
    ),
    ("Q2_K", 4096, 2048): (
        _RoutedRule("rows_eq", 49_152, "grouped_fwd_serial_q2_k_n4096_k2048_j32_j16"),
        _RoutedRule(
            "rows_lt_per_entry", 64, "grouped_fwd_serial_q2_k_n4096_k2048_j32_j16"
        ),
        _RoutedRule("always", 0, "grouped_fwd_serial_q2_k_n4096_k2048_j32"),
    ),
    ("IQ2_S", 2048, 512): (
        _RoutedRule("rows_eq", 65_536, "grouped_fwd_serial_iq2_s_n2048_k512_j64_j32"),
        _RoutedRule(
            "rows_lt_per_entry", 128, "grouped_fwd_serial_iq2_s_n2048_k512_j64_j32"
        ),
        _RoutedRule("always", 0, "grouped_fwd_serial_iq2_s_n2048_k512_j64"),
    ),
}


def routed_grouped_forward_rules() -> tuple[tuple[str, int, int, int, str], ...]:
    """Return the routed rule table as flat `(quant, n, k, value, symbol)` rows."""

    return tuple(
        (quant_type, n, k, rule.value, rule.symbol)
        for (quant_type, n, k), rules in sorted(_GROUPED_FORWARD_RULES.items())
        for rule in rules
    )


def select_grouped_forward_control(
    quant_type: str,
    out_features: int,
    in_features: int,
    rows: int,
    route_entries: int,
) -> GroupedForwardControl:
    """Return the deployed HIP control for one routed single-projection key.

    The key is the quant type and output shape. The tile choice then follows the
    ordered rules for that key, which depend on the aggregate rows and on the
    number of route entries in the bank. Unknown keys and unbuilt symbols fail
    closed.
    """

    rules = _GROUPED_FORWARD_RULES.get((quant_type, out_features, in_features))
    if rules is None:
        raise ValueError(
            "no deployed HIP grouped-forward control for "
            f"{quant_type} N={out_features}, K={in_features}"
        )
    if route_entries <= 0:
        raise ValueError(
            "routed grouped-forward selection needs at least one route entry"
        )
    inventory = control_inventory()
    for rule in rules:
        if rule.when not in _ROUTED_PREDICATES:
            raise ValueError(f"unknown routed predicate {rule.when!r}")
        if not _ROUTED_PREDICATES[rule.when](rows, route_entries, rule.value):
            continue
        spec = inventory.get(rule.symbol)
        if spec is None:
            raise ValueError(
                f"deployed HIP control {rule.symbol!r} is not a built control. "
                "Rebuild the HIP control bundle or fix the routed rule table"
            )
        return GroupedForwardControl(spec.symbol, spec)
    raise ValueError(
        f"routed grouped-forward rule table for {quant_type} N={out_features}, "
        f"K={in_features} has no matching entry for M={rows}"
    )
