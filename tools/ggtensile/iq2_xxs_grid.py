"""Authoritative IQ2_XXS codebook extraction and assembly emission."""

import re
from pathlib import Path

_GRID_COUNT = 256
IQ2_XXS_GRID_BYTES = 8 * _GRID_COUNT
_GRID_DECLARATION = re.compile(
    r"static const __device__ uint64_t iq2xxs_grid\[256\] = \{(?P<body>.*?)\};",
    re.DOTALL,
)
_GRID_VALUE = re.compile(r"0x([0-9a-fA-F]{16})")


def iq2_xxs_grid_values() -> tuple[int, ...]:
    header = (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "vendor"
        / "llama_cpp"
        / "iq2_xxs_grid.cuh"
    )
    text = header.read_text(encoding="utf-8")
    declaration = _GRID_DECLARATION.search(text)
    if declaration is None:
        raise ValueError("cannot find the authoritative iq2xxs_grid declaration")
    values = tuple(
        int(value, 16) for value in _GRID_VALUE.findall(declaration.group("body"))
    )
    if len(values) != _GRID_COUNT:
        raise ValueError(
            f"authoritative iq2xxs_grid has {len(values)} entries, expected {_GRID_COUNT}"
        )
    return values


def iq2_xxs_grid_rodata(symbol: str) -> str:
    if not symbol.startswith(".L"):
        raise ValueError("IQ2_XXS codebook symbol must be assembly-local")
    values = iq2_xxs_grid_values()
    lines = [
        '.section .rodata,"a",@progbits',
        ".p2align 3",
        f".type {symbol},@object",
        f"{symbol}:",
    ]
    for start in range(0, len(values), 4):
        entries = ", ".join(f"0x{value:016x}" for value in values[start : start + 4])
        lines.append(f".quad {entries}")
    lines.append(f".size {symbol}, {IQ2_XXS_GRID_BYTES}")
    return "\n".join(lines) + "\n"
