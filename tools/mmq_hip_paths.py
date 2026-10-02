"""One lookup policy for the installed HIP control artifacts.

`GGTENSILE_HIP_CONTROL_ROOT` overrides the search, the in-repo control bundle
comes next, and the wheel-installed bundle is the last resort. Every consumer
(benchmark, deployment runner, control launchers) resolves artifacts through
this module so a control is found or missed the same way everywhere.
"""

import os
import sysconfig
from pathlib import Path

_IN_REPO = (
    "build/mmq_hip_controls/gfx1151",
    "torch_ggml_ops/kernels/gfx1151",
    "torch_ggml_ops/kernels/gfx1151/hip_controls",
)
_IN_SITE_PACKAGES = (
    "torch_ggml_ops/kernels/gfx1151",
    "torch_ggml_ops/kernels/gfx1151/hip_controls",
)


def control_roots() -> tuple[Path, ...]:
    """Return every directory that may hold control artifacts, in priority order."""

    roots: list[Path] = []
    configured = os.environ.get("GGTENSILE_HIP_CONTROL_ROOT")
    if configured:
        path = Path(configured)
        roots.extend(
            (
                path,
                path / "gfx1151",
                path / "hip_controls",
                path / "gfx1151" / "hip_controls",
            )
        )
    repo_root = Path(__file__).resolve().parents[1]
    purelib = Path(sysconfig.get_paths()["purelib"])
    roots.extend(repo_root / relative for relative in _IN_REPO)
    roots.extend(purelib / relative for relative in _IN_SITE_PACKAGES)
    return tuple(roots)


def control_root() -> Path | None:
    """Return the first existing control root, or None when none is installed."""

    for root in control_roots():
        if root.is_dir():
            return root
    return None


def locate_control(symbol: str, code_object: Path | None = None) -> Path | None:
    """Return the artifact path of one control, or None when it is absent.

    A supplied path may be the artifact itself or one of the control
    directories. The accepted layouts mirror `control_roots`.
    """

    if code_object is not None:
        if code_object.is_file():
            return code_object
        if code_object.is_dir():
            for directory in (
                code_object,
                code_object / "hip_controls",
                code_object / "gfx1151",
                code_object / "gfx1151" / "hip_controls",
            ):
                candidate = directory / f"{symbol}.hsaco"
                if candidate.is_file():
                    return candidate
        return None
    for root in control_roots():
        candidate = root / f"{symbol}.hsaco"
        if candidate.is_file():
            return candidate
    return None
