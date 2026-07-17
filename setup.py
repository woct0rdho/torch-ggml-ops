from pathlib import Path

from setuptools import find_packages, setup
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
SOURCE = "csrc/mmq_hip.cu"
HEADER_DEPENDENCIES = [
    path.relative_to(ROOT).as_posix() for path in sorted(CSRC.rglob("*.cuh"))
]

CUDAExtension = cpp_extension.CUDAExtension


class BuildExtension(cpp_extension.BuildExtension):
    def get_source_files(self) -> list[str]:
        # CUDAExtension eagerly rewrites ext.sources to hipify-generated files
        # on ROCm. Source distributions should contain only the canonical input.
        return [SOURCE, *HEADER_DEPENDENCIES]


stable_defines = [
    "-DTORCH_TARGET_VERSION=0x020A000000000000",
    "-DTORCH_STABLE_ONLY",
]

setup(
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="torch_ggml_ops._C",
            sources=[SOURCE],
            include_dirs=[str(CSRC)],
            depends=HEADER_DEPENDENCIES,
            extra_compile_args={
                "cxx": ["-O3", *stable_defines],
                "nvcc": ["-O3", *stable_defines],
            },
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
