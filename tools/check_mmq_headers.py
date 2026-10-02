"""Verify that every checked-in C++ header under `csrc/` stands on its own.

Each header is compiled as its own translation unit, so a missing include, a
missing `#pragma once`, or a dependency on an earlier include in the same
translation unit fails the check. Include order therefore never matters for the
headers that pass, and any translation unit may include them freely.

Three vendored llama.cpp files are configuration fragments rather than headers:
`mmq-load-targets.cuh`, `mmq-vec-dot-targets.cuh`, and
`mmq-vec-dot-q2-k-rolled.cuh` expand against the `MMQ_*` settings and helpers
that `mmq_core.cuh` defines before including them, so `mmq_core.cuh` is the only
place they can be included from. They are listed in `CONFIGURATION_FRAGMENTS`
and reported separately instead of being compiled on their own. The three
translation units under `csrc/` are compiled from their own text, so the check
also covers the extension build.
"""

import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc"
ARCH = "gfx1151"
STABLE_DEFINES = (
    "-DUSE_ROCM",
    "-DTORCH_TARGET_VERSION=0x020A000000000000",
    "-DTORCH_STABLE_ONLY",
)
# Headers that define device code only, and the translation units that provide
# the torch stable ABI surface.
DEVICE_PREFIXES = ("ck/", "vendor/", "generated/")
DEVICE_HEADERS = ("mmq_core.cuh",)
CONFIGURATION_FRAGMENTS = (
    "vendor/llama_cpp/mmq-load-targets.cuh",
    "vendor/llama_cpp/mmq-vec-dot-targets.cuh",
    "vendor/llama_cpp/mmq-vec-dot-q2-k-rolled.cuh",
)
INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)


def _sources() -> list[Path]:
    paths = [
        path
        for pattern in ("*.h", "*.cuh", "*.cu", "*.cpp")
        for path in sorted(CSRC.rglob(pattern))
    ]
    return [path for path in paths if path.name != "mmq_bundle_table.cuh"]


def _relative(path: Path) -> str:
    return path.relative_to(CSRC).as_posix()


def _is_device(relative: str) -> bool:
    return relative.startswith(DEVICE_PREFIXES) or relative in DEVICE_HEADERS


def _torch_includes() -> list[str]:
    import torch.utils.cpp_extension

    include_dir = Path(sysconfig.get_paths()["include"])
    return [*torch.utils.cpp_extension.include_paths(), str(include_dir)]


def _command(relative: str, source: Path, hipcc: Path) -> list[str]:
    if _is_device(relative):
        flags = [
            "-x",
            "hip",
            "--cuda-device-only",
            f"--offload-arch={ARCH}",
        ]
    else:
        flags = ["-x", "hip", *(f"-I{path}" for path in _torch_includes())]
    return [
        str(hipcc),
        "-fsyntax-only",
        *flags,
        "-std=c++17",
        "-fuse-cuid=none",
        *STABLE_DEFINES,
        f"-I{CSRC}",
        str(source),
    ]


def _compile(path: Path, hipcc: Path, staging: Path) -> tuple[str, str | None]:
    relative = _relative(path)
    text = (
        path.read_text(encoding="utf-8")
        if path.suffix in (".cu", ".cpp")
        else f'#include "{relative}"\n'
    )
    source = staging / (path.stem + "_standalone.cpp")
    source.write_text(text, encoding="utf-8")
    result = subprocess.run(
        _command(relative, source, hipcc), capture_output=True, text=True, check=False
    )
    if result.returncode == 0:
        return relative, None
    errors = [
        line
        for line in result.stderr.splitlines()
        if " error: " in line or line.startswith("fatal error")
    ]
    return relative, "\n".join(errors[:8]) or result.stderr[-400:]


def _unresolved_includes() -> list[str]:
    problems = []
    for path in _sources():
        for name in INCLUDE_PATTERN.findall(path.read_text(encoding="utf-8")):
            if not (path.parent / name).is_file() and not (CSRC / name).is_file():
                problems.append(f"{_relative(path)}: missing include {name}")
    return problems


def _guarded(path: Path) -> bool:
    """A header is guarded when its first directive is `#pragma once`."""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        return stripped == "#pragma once"
    return False


def _missing_include_guard() -> list[str]:
    return [
        _relative(path)
        for path in _sources()
        if path.suffix in (".h", ".cuh") and not _guarded(path)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hipcc", type=Path)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    hipcc = (args.hipcc or Path(shutil.which("hipcc") or "")).resolve()
    if not hipcc.is_file():
        raise SystemExit("hipcc is required to check the headers")

    problems = [*_unresolved_includes(), *_missing_include_guard()]
    checked = [
        path for path in _sources() if _relative(path) not in CONFIGURATION_FRAGMENTS
    ]
    with tempfile.TemporaryDirectory() as directory:
        staging = Path(directory)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for relative, error in pool.map(
                lambda path: _compile(path, hipcc, staging), checked
            ):
                if error:
                    problems.append(f"{relative}:\n{error}")

    if problems:
        raise SystemExit("header layout problems:\n" + "\n".join(problems))
    print(
        f"{len(checked)} self-contained translation units, "
        f"{len(CONFIGURATION_FRAGMENTS)} configuration fragments skipped"
    )


if __name__ == "__main__":
    main()
