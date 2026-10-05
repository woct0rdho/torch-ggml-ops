"""The HIP control catalog: which body serves which key, and how it is built.

`tools/configs/hip_deployment.json` is the single source of truth for the
deployed matrix: its families and routed tables name the body each exact key
selects, and its `helpers` list names the bodies the direct launchers use
outside a key. `tools/configs/hip_control_catalog.json` holds the build
parameters of those bodies and of the candidates a running screen needs, one
record per symbol.

Both files are data. The schema is the config dataclasses in
`tools/mmq_bundle_wrapper_source.py`, and the kernel source is generated from
them by `render_wrapper`. An experiment that screens a new candidate adds its
record to the catalog, so the catalog is a superset of the deployment while a
screen is running. A screened-out candidate is dropped again, and its
measurement lives in the experiment record.
"""

import dataclasses
import json
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, get_args, get_type_hints

from tools.mmq_bundle_wrapper_source import (
    DenseBackwardConfig,
    ForwardConfig,
    GroupedBackwardConfig,
    SplitKReduceConfig,
    render_wrapper,
)

_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEPLOYMENT_PATH = _CONFIG_DIR / "hip_deployment.json"
CATALOG_PATH = _CONFIG_DIR / "hip_control_catalog.json"

_CONFIG_TYPES: dict[str, Any] = {
    "dense_backward": DenseBackwardConfig,
    "forward": ForwardConfig,
    "grouped_backward": GroupedBackwardConfig,
    "split_k_reduce": SplitKReduceConfig,
}


@dataclass(frozen=True)
class HIPControlSpec:
    symbol: str
    config: (
        ForwardConfig | DenseBackwardConfig | GroupedBackwardConfig | SplitKReduceConfig
    )

    @property
    def filename(self) -> str:
        return control_filename(self.symbol)


def control_filename(symbol: str) -> str:
    return f"{symbol}.hsaco"


def _payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path.name} must hold a JSON object")
    return data


def _enum_of(hint: Any) -> type[Enum] | None:
    """Return the enumeration a config field names, if it names one."""

    for candidate in get_args(hint) or (hint,):
        if isinstance(candidate, type) and issubclass(candidate, Enum):
            return candidate
    return None


@lru_cache(maxsize=1)
def retained_symbols() -> frozenset[str]:
    """Return every symbol the bundle has to carry."""

    payload = _payload(DEPLOYMENT_PATH)
    symbols = set(payload.get("helpers", ()))
    for family in payload.get("families", ()):
        for control in family.get("controls", ()):
            symbols.add(control["symbol"])
    for family in payload.get("routed", ()):
        for rule in family.get("rules", ()):
            symbols.add(rule["symbol"])
    return frozenset(symbols)


def _config(record: dict[str, Any]) -> Any:
    kind = record.get("config")
    if kind not in _CONFIG_TYPES:
        raise ValueError(f"unknown config kind {kind!r} in {CATALOG_PATH.name}")
    config_type = _CONFIG_TYPES[kind]
    declared = dataclasses.fields(config_type)
    fields = {field.name for field in declared}
    unknown = set(record) - fields - {"symbol", "config"}
    if unknown:
        raise ValueError(
            f"{record['symbol']} ({kind}) has unknown fields {sorted(unknown)}"
        )
    values = {name: value for name, value in record.items() if name in fields}
    hints = get_type_hints(config_type)
    for field in declared:
        if field.name not in values:
            continue
        hint = hints[field.name]
        if isinstance(field.default, tuple):
            values[field.name] = tuple(values[field.name])
        else:
            enumeration = _enum_of(hint)
            if enumeration is not None and values[field.name] is not None:
                values[field.name] = enumeration[values[field.name]]
    return config_type(**values)


@lru_cache(maxsize=1)
def hip_control_specs() -> tuple[HIPControlSpec, ...]:
    """Return the catalog entries the bundle compiles."""

    records = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    specs = tuple(
        HIPControlSpec(record["symbol"], _config(record)) for record in records
    )
    symbols = [spec.symbol for spec in specs]
    if len(symbols) != len(set(symbols)):
        raise ValueError("HIP control symbols must be unique")
    missing = sorted(retained_symbols() - set(symbols))
    if missing:
        raise ValueError(
            f"{DEPLOYMENT_PATH.name} names bodies {CATALOG_PATH.name} does not "
            f"build: {missing[:4]}"
        )
    return specs


def render_control(spec: HIPControlSpec) -> str:
    return render_wrapper(spec.symbol, spec.config)
