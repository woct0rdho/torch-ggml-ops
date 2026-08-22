import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
PACKAGE_KERNEL_DIR = ROOT / "torch_ggml_ops" / "kernels" / "gfx1151"
SOURCES = [
    "csrc/mmq_hip.cu",
    "csrc/mmq_bundle.cpp",
    "csrc/mmq_bundle_loader.cpp",
]
HEADER_DEPENDENCIES = [
    path.relative_to(ROOT).as_posix()
    for pattern in ("*.h", "*.cuh")
    for path in sorted(CSRC.rglob(pattern))
    if not path.name.endswith("_hip.cuh")
]
CUDAExtension = cpp_extension.CUDAExtension


def _enable_ccache() -> None:
    ccache = shutil.which("ccache")
    hipcc = shutil.which("hipcc")
    if ccache is None or hipcc is None:
        return

    os.environ.setdefault(
        "PYTORCH_NVCC",
        f"{shlex.quote(ccache)} {shlex.quote(hipcc)}",
    )
    if "CXX" not in os.environ:
        for candidate in (Path("/usr/lib/ccache/c++"), Path("/usr/lib64/ccache/c++")):
            if candidate.is_file():
                os.environ["CXX"] = str(candidate)
                break


_enable_ccache()


class BuildExtension(cpp_extension.BuildExtension):
    def run(self) -> None:
        subprocess.run(
            [sys.executable, "tools/build_mmq_bundle.py"],
            cwd=ROOT,
            check=True,
        )
        super().run()
        built_kernel_dir = (
            Path(self.build_lib) / "torch_ggml_ops" / "kernels" / "gfx1151"
        )
        built_kernel_dir.mkdir(parents=True, exist_ok=True)
        expected = {path.name for path in PACKAGE_KERNEL_DIR.glob("*.hsaco")}
        for artifact in built_kernel_dir.glob("*.hsaco"):
            if artifact.name not in expected:
                artifact.unlink()
        for artifact in PACKAGE_KERNEL_DIR.glob("*.hsaco"):
            shutil.copy2(artifact, built_kernel_dir / artifact.name)


stable_defines = [
    "-DTORCH_TARGET_VERSION=0x020A000000000000",
    "-DTORCH_STABLE_ONLY",
]

setup(
    packages=find_packages(exclude=("tests", "tests.*")),
    ext_modules=[
        CUDAExtension(
            name="torch_ggml_ops._C",
            sources=SOURCES,
            include_dirs=[str(CSRC)],
            depends=HEADER_DEPENDENCIES,
            extra_compile_args={
                "cxx": ["-O3", *stable_defines],
                "nvcc": ["-O3", *stable_defines],
            },
            extra_link_args=["-ldl"],
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
