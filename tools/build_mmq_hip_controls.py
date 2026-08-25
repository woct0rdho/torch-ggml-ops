"""Build the historical HIP controls outside the public GGTensile bundle."""

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
from tools.mmq_hip_control_spec import (
    HIPControlSpec,
    control_filename,
    hip_control_specs,
    render_control,
)

ARCH = "gfx1151"
OUTPUT_DIR = ROOT / "build/mmq_hip_controls/gfx1151"
SOURCE_DIR = ROOT / "build/mmq_hip_control_sources/gfx1151"
STAMP = ".mmq-hip-controls-build-input"


def _hipcc(value: Path | None) -> Path:
    selected = value or Path(shutil.which("hipcc") or "")
    selected = selected.resolve()
    if not selected.is_file():
        raise FileNotFoundError("hipcc is required to build HIP controls")
    return selected


def _digest(
    specs: tuple[HIPControlSpec, ...], hipcc: Path, toolchain: Toolchain
) -> str:
    digest = hashlib.sha256()
    digest.update(str(hipcc).encode())
    digest.update(
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
        ROOT / "tools/mmq_hip_control_spec.py",
        ROOT / "tools/mmq_bundle_wrapper_source.py",
        ROOT / "csrc/mmq_core.cuh",
        *sorted((ROOT / "csrc/ck").rglob("*.cuh")),
        *sorted((ROOT / "csrc/vendor/llama_cpp").rglob("*.cuh")),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    for spec in specs:
        digest.update(spec.symbol.encode())
        digest.update(render_control(spec).encode())
    return digest.hexdigest()


def _verify(path: Path, symbol: str, readelf: Path) -> bytes:
    data = path.read_bytes()
    if not data.startswith(b"\x7fELF") or ARCH.encode() not in data:
        raise RuntimeError(f"invalid gfx1151 HIP control {path}")
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
    spec: HIPControlSpec,
    staging: Path,
    hipcc: Path,
    readelf: Path,
) -> tuple[str, bytes]:
    source_text = render_control(spec)
    source = (
        SOURCE_DIR
        / f"{spec.symbol}-{hashlib.sha256(source_text.encode()).hexdigest()[:16]}.cu"
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(source_text, encoding="utf-8")
    destination = staging / control_filename(spec.symbol)
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
        f"-I{ROOT / 'csrc'}",
        str(source),
        "-o",
        str(destination),
    ]
    subprocess.run(command, check=True)
    return spec.symbol, _verify(destination, spec.symbol, readelf)


def _current(specs: tuple[HIPControlSpec, ...], digest: str) -> bool:
    expected = {control_filename(spec.symbol) for spec in specs}
    actual = {path.name for path in OUTPUT_DIR.glob("*.hsaco")}
    return (
        OUTPUT_DIR.is_dir()
        and actual == expected
        and (OUTPUT_DIR / STAMP).is_file()
        and (OUTPUT_DIR / STAMP).read_text(encoding="utf-8") == digest + "\n"
    )


def _build(
    specs: tuple[HIPControlSpec, ...], hipcc: Path, readelf: Path, jobs: int
) -> tuple[Path, list[bytes]]:
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mmq-hip-controls-", dir=OUTPUT_DIR.parent))
    keep_staging = False
    try:
        images: dict[str, bytes] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(_compile, spec, staging, hipcc, readelf): spec
                for spec in specs
            }
            for future in concurrent.futures.as_completed(futures):
                symbol, image = future.result()
                images[symbol] = image
                print(f"built {symbol}", flush=True)
        keep_staging = True
        return staging, [images[spec.symbol] for spec in specs]
    finally:
        if not keep_staging:
            shutil.rmtree(staging, ignore_errors=True)


def _install(staging: Path, digest: str) -> None:
    (staging / STAMP).write_text(digest + "\n", encoding="utf-8")
    old = OUTPUT_DIR.with_name(OUTPUT_DIR.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if OUTPUT_DIR.exists():
        OUTPUT_DIR.rename(old)
    staging.rename(OUTPUT_DIR)
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
    hipcc = _hipcc(args.hipcc)
    toolchain = Toolchain.discover()
    specs = hip_control_specs()
    digest = _digest(specs, hipcc, toolchain)
    if args.check:
        if not _current(specs, digest):
            raise SystemExit("HIP control bundle is stale")
        print(f"HIP control bundle is current ({len(specs)} kernels)")
        return
    if not args.force and not args.verify_reproducible and _current(specs, digest):
        return
    first, images = _build(specs, hipcc, toolchain.readelf, args.jobs)
    if args.verify_reproducible:
        second, second_images = _build(specs, hipcc, toolchain.readelf, args.jobs)
        shutil.rmtree(second, ignore_errors=True)
        if images != second_images:
            shutil.rmtree(first, ignore_errors=True)
            raise RuntimeError("non-reproducible HIP control bundle")
    _install(first, digest)
    print(f"installed {len(specs)} HIP controls in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
