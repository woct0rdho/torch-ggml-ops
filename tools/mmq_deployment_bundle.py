"""Build the exact-key gfx1151 deployment bundle.

HIP compilation is limited to Q8_1 quantizers and grouped row-task setup.
Every multiply artifact is emitted by a typed GGTensile writer.
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ggtensile.toolchain import Toolchain
from tools.mmq_bundle_wrapper_source import render_wrapper
from tools.mmq_deployment_spec import (
    BundleKernel,
    header_text,
    kernels,
    writer_for,
)

CSRC = ROOT / "csrc"
PACKAGE_DIR = ROOT / "torch_ggml_ops/kernels/gfx1151"
HEADER = CSRC / "generated/mmq_bundle_table.cuh"
SOURCE_DIR = ROOT / "build/mmq_bundle_sources/gfx1151"
ARCH = "gfx1151"


def _verify(path: Path, symbol: str, readelf: Path) -> None:
    data = path.read_bytes()
    if not data.startswith(b"\x7fELF"):
        raise RuntimeError(f"invalid ELF code object {path}")
    output = subprocess.run(
        [str(readelf), "--file-header", "--symbols", "--wide", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    flags = next(
        (line for line in output.splitlines() if line.lstrip().startswith("Flags:")),
        "",
    )
    if ARCH not in flags:
        raise RuntimeError(f"code object is not {ARCH}: {path}")
    exported = {
        line.split()[-1]
        for line in output.splitlines()
        if " FUNC " in line and " GLOBAL " in line
    }
    if exported != {symbol}:
        raise RuntimeError(f"{path} exports {sorted(exported)}, expected {symbol}")


def _compile(
    kernel: BundleKernel,
    staging: Path,
    hipcc: Path,
    toolchain: Toolchain,
    readelf: Path,
) -> None:
    destination = staging / f"{kernel.symbol}.hsaco"
    if kernel.hip_config is not None:
        source_text = render_wrapper(kernel.symbol, kernel.hip_config)
        source = SOURCE_DIR / f"{kernel.cpp_id}.cu"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(source_text, encoding="utf-8")
        command = [
            str(hipcc),
            "--genco",
            "--no-gpu-bundle-output",
            "-c",
            "-O3",
            "-std=c++17",
            "-fuse-cuid=none",
            "-mcode-object-version=5",
            f"--offload-arch={ARCH}",
            f"-I{CSRC}",
            str(source),
            "-o",
            str(destination),
        ]
        subprocess.run(command, check=True)
    else:
        assembly = SOURCE_DIR / f"{kernel.cpp_id}-{kernel.symbol}.s"
        obj = staging / f"{kernel.cpp_id}.o"
        toolchain.assemble(assembly, obj)
        toolchain.link(obj, destination)
        obj.unlink()
    _verify(destination, kernel.symbol, readelf)


def _build(
    items: tuple[BundleKernel, ...],
    hipcc: Path,
    toolchain: Toolchain,
    jobs: int,
) -> Path:
    PACKAGE_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mmq-gfx1151-", dir=PACKAGE_DIR.parent))
    try:
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        for item in items:
            if item.hip_config is None:
                writer_for(item, toolchain).write(
                    SOURCE_DIR / f"{item.cpp_id}-{item.symbol}.s"
                )
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    _compile, item, staging, hipcc, toolchain, toolchain.readelf
                ): item
                for item in items
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
                print(f"built {futures[future].cpp_id}", flush=True)
        return staging
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _install(staging: Path, items: tuple[BundleKernel, ...]) -> None:
    HEADER.parent.mkdir(parents=True, exist_ok=True)
    HEADER.write_text(header_text(items), encoding="utf-8")
    old = PACKAGE_DIR.with_name(PACKAGE_DIR.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if PACKAGE_DIR.exists():
        PACKAGE_DIR.rename(old)
    staging.rename(PACKAGE_DIR)
    shutil.rmtree(old, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hipcc", type=Path)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    hipcc = (args.hipcc or Path(shutil.which("hipcc") or "")).resolve()
    if not hipcc.is_file():
        raise FileNotFoundError("hipcc is required to build Q8_1/setup artifacts")
    toolchain = Toolchain.discover()
    items = kernels()
    staging = _build(items, hipcc, toolchain, args.jobs)
    _install(staging, items)
    print(f"installed {len(items)} exact MMQ kernels in {PACKAGE_DIR}")


if __name__ == "__main__":
    main()
