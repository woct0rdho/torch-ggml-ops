"""Build the exact-key gfx1151 deployment bundle.

HIP compilation is limited to Q8_1 quantizers and grouped row-task setup.
Every multiply artifact is emitted by a typed GGTensile writer.
"""

import argparse
import concurrent.futures
import hashlib
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
CONFIG = ROOT / "tools/ggtensile/configs"
PACKAGE_DIR = ROOT / "torch_ggml_ops/kernels/gfx1151"
HEADER = CSRC / "generated/mmq_bundle_table.cuh"
SOURCE_DIR = ROOT / "build/mmq_bundle_sources/gfx1151"
STAMP = ".mmq-build-input"
ARCH = "gfx1151"


def _digest(items: tuple[BundleKernel, ...], hipcc: Path, toolchain: Toolchain) -> str:
    digest = hashlib.sha256(
        subprocess.run(
            [str(hipcc), "--version"], check=True, capture_output=True
        ).stdout
    )
    digest.update(str(toolchain.assembler).encode())
    digest.update(
        subprocess.run(
            [str(toolchain.assembler), "--version"], check=True, capture_output=True
        ).stdout
    )
    for path in (
        Path(__file__),
        ROOT / "tools/mmq_deployment_spec.py",
        ROOT / "tools/mmq_bundle_wrapper_source.py",
        ROOT / "tools/ggtensile/configs/mmq_deployment.json",
        *sorted(CONFIG.glob("mmq_*_catalog.json")),
        *sorted((ROOT / "tools/ggtensile").glob("*.py")),
        CSRC / "mmq_core.cuh",
        *sorted((CSRC / "vendor/llama_cpp").glob("*.cuh")),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    digest.update(header_text(items).encode())
    return digest.hexdigest()


def _verify(path: Path, symbol: str, readelf: Path) -> bytes:
    data = path.read_bytes()
    if not data.startswith(b"\x7fELF") or ARCH.encode() not in data:
        raise RuntimeError(f"invalid gfx1151 code object {path}")
    output = subprocess.run(
        [str(readelf), "--symbols", "--wide", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    exported = {
        line.split()[-1]
        for line in output.splitlines()
        if " FUNC " in line and " GLOBAL " in line
    }
    if exported != {symbol}:
        raise RuntimeError(f"{path} exports {sorted(exported)}, expected {symbol}")
    return data


def _compile(
    kernel: BundleKernel,
    staging: Path,
    hipcc: Path,
    toolchain: Toolchain,
    readelf: Path,
) -> tuple[str, bytes]:
    destination = staging / kernel.filename
    if kernel.hip_config is not None:
        source_text = render_wrapper(kernel.symbol, kernel.hip_config)
        source = (
            SOURCE_DIR
            / f"{kernel.cpp_id}-{hashlib.sha256(source_text.encode()).hexdigest()[:16]}.cu"
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(source_text)
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
    return kernel.cpp_id, _verify(destination, kernel.symbol, readelf)


def _current(items: tuple[BundleKernel, ...], digest: str) -> bool:
    expected = {item.filename for item in items}
    actual = (
        {path.name for path in PACKAGE_DIR.iterdir() if path.name != STAMP}
        if PACKAGE_DIR.is_dir()
        else set()
    )
    return (
        PACKAGE_DIR.is_dir()
        and HEADER.is_file()
        and (PACKAGE_DIR / STAMP).is_file()
        and expected == actual
        and HEADER.read_text() == header_text(items)
        and (PACKAGE_DIR / STAMP).read_text() == digest + "\n"
    )


def _build(
    items: tuple[BundleKernel, ...], hipcc: Path, toolchain: Toolchain, jobs: int
) -> tuple[Path, list[bytes]]:
    PACKAGE_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mmq-gfx1151-", dir=PACKAGE_DIR.parent))
    try:
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        for item in items:
            if item.hip_config is None:
                writer_for(item, toolchain).write(
                    SOURCE_DIR / f"{item.cpp_id}-{item.symbol}.s"
                )
        images: dict[str, bytes] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    _compile, item, staging, hipcc, toolchain, toolchain.readelf
                ): item
                for item in items
            }
            for future in concurrent.futures.as_completed(futures):
                name, image = future.result()
                images[name] = image
                print(f"built {name}", flush=True)
        return staging, [images[item.cpp_id] for item in items]
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _install(staging: Path, items: tuple[BundleKernel, ...], digest: str) -> None:
    (staging / STAMP).write_text(digest + "\n")
    HEADER.parent.mkdir(parents=True, exist_ok=True)
    HEADER.write_text(header_text(items))
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify-reproducible", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    hipcc = (args.hipcc or Path(shutil.which("hipcc") or "")).resolve()
    if not hipcc.is_file():
        raise FileNotFoundError("hipcc is required to build Q8_1/setup artifacts")
    toolchain = Toolchain.discover()
    items = kernels()
    digest = _digest(items, hipcc, toolchain)
    if args.check:
        if not _current(items, digest):
            raise SystemExit("MMQ gfx1151 bundle is stale")
        return
    if not args.force and not args.verify_reproducible and _current(items, digest):
        return
    first, images = _build(items, hipcc, toolchain, args.jobs)
    if args.verify_reproducible:
        second, second_images = _build(items, hipcc, toolchain, args.jobs)
        shutil.rmtree(second, ignore_errors=True)
        if images != second_images:
            shutil.rmtree(first, ignore_errors=True)
            raise RuntimeError("non-reproducible MMQ deployment bundle")
    _install(first, items, digest)
    print(f"installed {len(items)} exact MMQ kernels in {PACKAGE_DIR}")


if __name__ == "__main__":
    main()
