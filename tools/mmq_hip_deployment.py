"""Deployed HIP control selection for the exact deployment keys.

`tools/configs/hip_deployment.json` is the single source of truth for which
installed HIP control serves each problem. The control's build configuration in
`tools.mmq_hip_control_spec` remains the authority for its geometry, so a
deployment entry only names a symbol and the launch configuration is derived
from the same record that built the artifact. The GGTensile catalogs under the
same `configs/` directory play the same role for the public bundle.

Keys use the module problem axes:
- `OrdinaryForward` / `OrdinaryBackward`: the `ProblemSize(m, n, k)` of the
  operation, `(rows, out_features, in_features)` in both directions, because
  `n` is the weight's row count and `k` the values per packed row.
- `FixedGroupedForward` / `FixedGroupedBackward`: `(tokens,
  out_features, in_features)` of the fixed problem.

The `families` table keys the non-routed families by their exact
`(operation, quant, m, n, k)`. The routed families cannot be keyed that way
because the deployed body also depends on the aggregate routed rows and, for
the grouped-forward tile choice, on the route entries in the bank. Their
ordered rules live in the `routed` table of the same catalog, and every caller
resolves them through `select_routed_control`.

A control symbol is `<kind>_<quant>_<body tags>`. A control that serves one
shape spells a family key: a routed control spells
`n<out_features>_k<in_features>`, the weight's own letters, so the forward and
backward symbols of one family agree, and a fixed-group control spells
`g<group count>_k<in_features>`. A dense control serves several shapes and
keeps only its tags, sometimes with a partial shape annotation of its own. The
body tags describe geometry:
`mt<row block>`, `nt<result columns>` (`16 * n_tiles`), `ki<contraction
stage>` (the `k_iteration` value), `j<token tile>`, `g<group count>`,
`pad<LDS padding words>`, `sw<LDS swizzle chunk>`, `mw<m_tiles_per_wave>`,
`s<split or stage count>` and `mb<min resident blocks>`.

`nt` and `ki` are the matrix-instruction letters rather than the problem's:
the instruction computes `D[M,N] = A[M,K] * B[K,N]` and the tile's `N` is the
dimension it writes while its `K` is the contraction. A backward control
writes `in_features` and contracts over `out_features`, so its `nt`/`ki` are
the transpose of the family key's `n`/`k`. A forward control's agree. The
letters are kept from colliding by never spelling the contraction stage as a
bare `k`, so `k<value>` in a symbol is `in_features`.
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
# Split-contraction slices step by the 32-wide contraction stage, so the
# contraction bound must be a multiple of it for the unguarded fast path.
SPLIT_K_STEP = 32
SPLIT_K_REDUCE_SYMBOL = "dense_bwd_split_k_reduce"
# Dense-forward LDS geometry. A J-row activation half tile holds
# `J * MMQ_TILE_Y_K` ints (32 values plus their Q8_1 metadata per row), and the
# packed weight tile holds `MMQ_I` rows of the type's SRAM stride. Compact
# strides are the ones the vendored loader uses when `MMQ_COMPACT_TILE` is set.
# Only Q4_K and Q5_K have a smaller compact stride.
_FORWARD_TILE_Y_K = 36
_FORWARD_MMQ_I = 64
_FORWARD_COMPACT_STRIDES = {"Q4_K": 36, "Q5_K": 36, "Q2_K": 40}
_FORWARD_MMQ_I_STRIDE_HALF = 40
_FORWARD_META_STRIDE = 4
_FORWARD_STRIDES = {
    "Q2_K": 100,
    "Q3_K": 84,
    "Q6_K": 76,
    "Q8_0": 76,
    "Q2_0": 76,
    "Q4_0": 76,
    "Q5_0": 76,
    "IQ4_NL": 76,
    "IQ4_XS": 76,
    "Q4_K": 76,
    "Q5_K": 76,
}


def forward_lds_bytes(
    j: int,
    quant_name: str,
    compact_tile: bool,
    table_bytes: int,
    half_stage: bool = False,
    fragment_activation: bool = False,
) -> int:
    """Return the dynamic LDS a dense-forward body addresses."""

    stride = _FORWARD_STRIDES.get(quant_name)
    if stride is None:
        raise ValueError(f"no dense-forward LDS stride is known for {quant_name}")
    if compact_tile:
        stride = _FORWARD_COMPACT_STRIDES.get(quant_name, stride)
    if half_stage:
        # A stage holds one 128-value half, so the weight row is one quant region
        # plus one scale region instead of two of each.
        stride = _FORWARD_MMQ_I_STRIDE_HALF
    # A control that reads its activation fragments from global memory keeps only
    # the four per-group scales of each token, which is what frees the workgroups.
    if fragment_activation:
        activation = j * _FORWARD_META_STRIDE
    else:
        activation = -(-(j * _FORWARD_TILE_Y_K) // 128) * 128
    return 4 * (j + activation + _FORWARD_MMQ_I * stride) + (1024 if table_bytes else 0)


_DENSE_FORWARD_BLOCK = (32, 4, 1)
_WIDE_FORWARD_BLOCK = (32, 8, 1)
_DENSE_BACKWARD_BLOCK = (128, 1, 1)
_Q3_K_LDS_BYTES = 40_448
# A wide dense tile holds twice the weight rows, so its LDS is the J-row
# activation tile plus 128 rows of that type's packed SRAM layout.
_WIDE_LDS_BYTES = {"Q3_K": 61_952, "Q4_K": 57_856, "Q5_K": 57_856, "IQ4_XS": 57_856}
_J64_LDS_BYTES = 28_928
# The Q2_0 level table holds 256 expanded payload bytes behind the weight tile.
_Q2_0_LEVEL_TABLE_BYTES = 1_024
# The Q5_0 table holds the 16 fifth-bit spreads of a group.
_Q5_0_LEVEL_TABLE_BYTES = 64
_J128_LDS_BYTES = 38_400
# Grouped-forward launch geometry. The activation tile holds one Q8_1 plane per
# J-tile row (32 quant ints plus four metadata ints), the weight tile holds
# MMQ_I rows of the packed SRAM layout, and the kernel reserves a J-int prefix.
_GROUPED_FORWARD_BLOCK = (32, 4, 1)
_GROUPED_FORWARD_TILE_I = 64
_GROUPED_FORWARD_THREADS = 128
_GROUPED_TILE_Y_K = 36
# A compact grouped weight tile halves the stride. Only Q2_K deploys it.
_GROUPED_COMPACT_STRIDES = {"Q2_K": 40}
_MMQ_SRAM_STRIDE = {
    "Q2_K": 100,
    "Q3_K": 84,
    "IQ2_S": 84,
    "Q6_K": 91,
    "Q8_0": 76,
    "Q2_0": 76,
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
        self, m: int, out_features: int, in_features: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        """Return `(grid, block, shared_bytes)` for this control.

        `out_features` is the weight's row count and `in_features` the values
        per packed row. A forward control writes `out_features` rows and reduces
        over `in_features`. A backward control writes `in_features` and reduces
        over `out_features`.
        """

        config = self.spec.config
        if isinstance(config, ForwardConfig) and config.kind is ForwardKind.DENSE:
            j = config.j
            quant_name = str(getattr(config.quant_type, "name", ""))
            if config.wide_tile:
                if quant_name not in _WIDE_LDS_BYTES:
                    raise ValueError(f"no wide dense tile is built for {quant_name}")
                if out_features % 128:
                    raise ValueError("a wide dense tile needs whole 128-row tiles")
                return (
                    (out_features // 128, math.ceil(m / j), 1),
                    _WIDE_FORWARD_BLOCK,
                    _WIDE_LDS_BYTES[quant_name],
                )
            shared_bytes = forward_lds_bytes(
                j,
                quant_name,
                bool(config.compact_tile),
                (
                    _Q5_0_LEVEL_TABLE_BYTES
                    if quant_name == "Q5_0"
                    else _Q2_0_LEVEL_TABLE_BYTES
                )
                if config.table_decode
                else 0,
                False,
                bool(getattr(config, "fragment_activation", False)),
            )
            return (
                (out_features // 64, math.ceil(m / j), 1),
                _DENSE_FORWARD_BLOCK,
                shared_bytes,
            )
        if isinstance(config, DenseBackwardConfig):
            if config.m_tiles_per_wave < 1:
                raise ValueError("a dense backward tile needs a positive row tile")
            m_per_block = 16 * config.m_tiles_per_wave * config.active_waves
            # `config.n_tiles` counts the written sixteen-column tiles, which a
            # backward body lays over `in_features`, and `config.k_iteration`
            # steps the contraction over `out_features`.
            n_per_block = 16 * config.n_tiles
            m_blocks = math.ceil(m / m_per_block)
            n_blocks = math.ceil(in_features / n_per_block)
            if config.full_tiles and (
                m % m_per_block
                or in_features % n_per_block
                or out_features % config.k_iteration
            ):
                # The unguarded tile writes whole result tiles, so a key that
                # does not divide evenly faults rather than clipping. That is
                # how an `m_tiles_per_wave = 3` body failed: its 192-row block
                # does not divide 2048.
                raise ValueError(
                    f"{self.symbol} stages whole tiles only: the rows {m} must be "
                    f"a multiple of {m_per_block}, the written width "
                    f"(in_features) {in_features} a multiple of the "
                    f"{n_per_block}-wide result tile, and the contraction "
                    f"(out_features) {out_features} a multiple of the "
                    f"{config.k_iteration}-deep contraction stage"
                )
            if config.split_k:
                if config.group_m:
                    raise ValueError("split-contraction controls cannot group in M")
                step = split_k_step(config.k_iteration)
                if config.exact_out_features % step:
                    raise ValueError(
                        "split-contraction controls need a contraction bound that "
                        "is a multiple of the decode step"
                    )
                # The slice width is rounded up to the stage width, so every
                # slice but the last is exactly that wide and the last one, at
                # most that wide, is a multiple of the stage width too because
                # both the contraction bound and the width are. No stage is
                # therefore truncated and the unguarded full-tile path is safe.
                return (
                    (m_blocks, n_blocks, config.split_k),
                    _DENSE_BACKWARD_BLOCK,
                    0,
                )
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


def split_k_step(k_iteration: int) -> int:
    """Return the slice quantum: the stage width, at least the decode step."""

    return max(SPLIT_K_STEP, k_iteration)


def split_k_chunk(out_features: int, slices: int, k_iteration: int = 0) -> int:
    """Return the slice width, rounded up to the contraction step."""

    step = split_k_step(k_iteration)
    chunk = -(-out_features // slices)
    return -(-chunk // step) * step


def split_k_reduce_configuration(
    rows: int, in_features: int
) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
    """Return `(grid, block, shared_bytes)` for the partial reduction."""

    count = rows * in_features
    blocks = min(1024, max(1, math.ceil(count / 256)))
    return ((blocks, 1, 1), (256, 1, 1), 0)


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


def grouped_forward_geometry(config: ForwardConfig) -> tuple[int, int]:
    """Return the `(row tile, workgroup width)` of one grouped-forward control.

    The vector dot gives each warp sixteen weight rows, so the row tile is the
    warp count times sixteen: the inherited configuration stages 64 rows with
    four warps and the wide tile stages 128 rows with eight. A wider tile halves
    the number of times the packed weights of an expert are streamed.
    """

    if (
        not isinstance(config, ForwardConfig)
        or config.kind is not ForwardKind.GROUPED_SERIAL
    ):
        raise ValueError("the geometry of a grouped-forward control needs a spec")
    if config.quant_type is None:
        raise ValueError("a grouped-forward control needs a quant type")
    if config.wide_tile:
        return 128, 256
    return _GROUPED_FORWARD_TILE_I, _GROUPED_FORWARD_THREADS


def _grouped_forward_lds_bytes(spec: HIPControlSpec) -> int:
    """Return the dynamic LDS bytes of one grouped-forward control.

    Mirrors the tile layout in `csrc/mmq_core.cuh`: a J-int prefix, a `J x 36`
    activation tile padded to the workgroup width, and a `row tile x stride`
    weight tile.
    """

    config = spec.config
    if (
        not isinstance(config, ForwardConfig)
        or config.kind is not ForwardKind.GROUPED_SERIAL
    ):
        raise ValueError(f"{spec.symbol} is not a routed grouped-forward control")
    if config.quant_type is None:
        raise ValueError(f"{spec.symbol} has no quant type")
    tile_i, threads = grouped_forward_geometry(config)
    stride = _MMQ_SRAM_STRIDE[config.quant_type.name]
    if config.compact_tile:
        # A compact weight tile holds one 128-value half per row, and Q2_K's
        # half is forty ints rather than the seventy-six of its full layout.
        stride = _GROUPED_COMPACT_STRIDES.get(config.quant_type.name, stride)
    if getattr(config, "fragment_activation", False):
        # Only the sixteen-byte metadata header of each token stays in LDS.
        tile_y = config.j * _FORWARD_META_STRIDE
    else:
        tile_y = -(-config.j * _GROUPED_TILE_Y_K // threads) * threads
    # A control that decodes through the level table keeps its 256 expansions
    # behind the weight tile.
    table_bytes = (
        _Q2_0_LEVEL_TABLE_BYTES if getattr(config, "table_decode", False) else 0
    )
    return 4 * (config.j + tile_y + tile_i * stride) + table_bytes


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
        config = self.spec.config
        if (
            not isinstance(config, ForwardConfig)
            or config.kind is not ForwardKind.GROUPED_SERIAL
        ):
            raise ValueError(
                f"{self.spec.symbol} is not a routed grouped-forward control"
            )
        tile_i, threads = grouped_forward_geometry(config)
        if out_features % tile_i:
            raise ValueError("grouped-forward out features must divide by the tile")
        return (
            (out_features // tile_i, route_entries, 1),
            (
                _GROUPED_FORWARD_BLOCK[0],
                _GROUPED_FORWARD_BLOCK[1] * threads // _GROUPED_FORWARD_THREADS,
                1,
            ),
            self.lds_bytes,
        )


@dataclass(frozen=True)
class RoutedControl:
    """One deployed routed control choice for a family.

    The routed families cannot be keyed by an exact `(m, n, k)` because the
    body also depends on the aggregate routed rows and, for the routed
    grouped-forward tile choice, on the route entries in the bank. Their
    ordered rules live in the same catalog as the exact families and every
    caller resolves them through `select_routed_control`.
    """

    operation: str
    quant_type: str
    out_features: int
    in_features: int
    when: str
    value: int | None
    symbol: str
    tile: int | None = None
    # Family-level deployed geometry. `rows` is the set of aggregate routed row
    # counts the generation materializes, `experts` the route capacity used to
    # size the device task bank, and `pair_single` marks a paired operation that
    # the deployed family serves as one single-projection body per projection.
    experts: int = 0
    rows: tuple[int, ...] = ()
    pair_single: bool = False

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.operation, self.quant_type, self.out_features, self.in_features)


def _rows_equal(rows: int, route_entries: int, value: int) -> bool:
    return rows == value


def _rows_below_per_entry(rows: int, route_entries: int, value: int) -> bool:
    return rows < value * route_entries


def _rows_at_least(rows: int, route_entries: int, value: int) -> bool:
    return rows >= value


def _always(rows: int, route_entries: int, value: int) -> bool:
    return True


_ROUTED_PREDICATES = {
    "rows_eq": _rows_equal,
    "rows_lt_per_entry": _rows_below_per_entry,
    "rows_at_least": _rows_at_least,
    "always": _always,
}


@lru_cache(maxsize=1)
def routed_table() -> tuple[RoutedControl, ...]:
    """Return every catalogued routed rule in evaluation order."""

    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    controls: list[RoutedControl] = []
    for family in payload.get("routed", []):
        operation = family["operation"]
        quant_type = family["quant_type"]
        out_features = int(family["n"])
        in_features = int(family["k"])
        family_experts = int(family.get("experts", 0))
        family_rows = tuple(int(row) for row in family.get("rows", ()))
        pair_single = bool(family.get("pair_single", False))
        for rule in family["rules"]:
            when = rule["when"]
            if when not in _ROUTED_PREDICATES:
                raise ValueError(f"unknown routed predicate {when!r}")
            value = rule.get("value")
            controls.append(
                RoutedControl(
                    operation=operation,
                    quant_type=quant_type,
                    out_features=out_features,
                    in_features=in_features,
                    when=when,
                    value=None if value is None else int(value),
                    symbol=rule["symbol"],
                    tile=None if rule.get("tile") is None else int(rule["tile"]),
                    experts=family_experts,
                    rows=family_rows,
                    pair_single=pair_single,
                )
            )
    return tuple(controls)


def select_routed_control(
    operation: str,
    quant_type: str,
    out_features: int,
    in_features: int,
    rows: int,
    route_entries: int,
) -> RoutedControl:
    """Return the deployed routed control for one family key and route bank.

    The key is the operation, quant type, and output shape. The body then
    follows the ordered rules for that key, which depend on the aggregate rows
    and, for the routed grouped-forward tile choice, on the route entries in
    the bank. Unknown keys, unmatched shapes, and unbuilt symbols fail closed.
    """

    key = (operation, quant_type, out_features, in_features)
    rules = [control for control in routed_table() if control.key == key]
    if not rules:
        raise ValueError(
            f"no deployed HIP routed control for {operation} {quant_type} "
            f"N={out_features}, K={in_features}"
        )
    inventory = control_inventory()
    for rule in rules:
        predicate = _ROUTED_PREDICATES[rule.when]
        if not predicate(rows, route_entries, rule.value or 0):
            continue
        if rule.symbol not in inventory:
            raise ValueError(
                f"deployed HIP control {rule.symbol!r} is not a built control. "
                f"Rebuild the HIP control bundle or fix {CONFIG_PATH.name}"
            )
        return rule
    raise ValueError(
        f"no deployed HIP routed control for {operation} {quant_type} "
        f"N={out_features}, K={in_features} at rows={rows}"
    )


def select_grouped_forward_control(
    quant_type: str,
    out_features: int,
    in_features: int,
    rows: int,
    route_entries: int,
) -> GroupedForwardControl:
    """Return the deployed HIP control for one routed single-projection key."""

    if route_entries <= 0:
        raise ValueError(
            "routed grouped-forward selection needs at least one route entry"
        )
    choice = select_routed_control(
        "GroupedForward", quant_type, out_features, in_features, rows, route_entries
    )
    spec = control_inventory()[choice.symbol]
    return GroupedForwardControl(choice.symbol, spec)
