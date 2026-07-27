import os
import shutil
import subprocess
import sysconfig
from dataclasses import dataclass
from pathlib import Path


class ToolchainError(RuntimeError):
    pass


@dataclass(frozen=True)
class Toolchain:
    assembler: Path
    readelf: Path
    objdump: Path

    @classmethod
    def discover(cls, assembler: Path | None = None):
        selected = assembler or _find_assembler()
        if not selected.is_file() or not os.access(selected, os.X_OK):
            raise ToolchainError(f"assembler is not executable: {selected}")
        tool_dir = selected.resolve().parent
        return cls(
            assembler=selected.resolve(),
            readelf=_find_sibling_tool(tool_dir, "llvm-readelf"),
            objdump=_find_sibling_tool(tool_dir, "llvm-objdump"),
        )

    def identity(self) -> str:
        return _run([str(self.assembler), "--version"]).stdout.strip()

    def assemble(self, source: Path, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                str(self.assembler),
                "-x",
                "assembler",
                "--target=amdgcn-amd-amdhsa",
                "-mcode-object-version=5",
                "-c",
                "-mcpu=gfx1151",
                "-mno-wavefrontsize64",
                str(source),
                "-o",
                str(output),
            ]
        )

    def link(self, source: Path, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                str(self.assembler),
                "--target=amdgcn-amd-amdhsa",
                "-Xlinker",
                "--build-id=sha1",
                str(source),
                "-o",
                str(output),
            ]
        )

    def readelf_output(self, code_object: Path) -> str:
        return _run(
            [
                str(self.readelf),
                "--wide",
                "--file-header",
                "--symbols",
                "--notes",
                str(code_object),
            ]
        ).stdout

    def disassembly_output(self, code_object: Path) -> str:
        return _run(
            [
                str(self.objdump),
                "--disassemble",
                "--mcpu=gfx1151",
                str(code_object),
            ]
        ).stdout


def _find_assembler() -> Path:
    override = os.environ.get("GGTENSILE_AMDCLANGXX")
    if override:
        return Path(override)
    found = shutil.which("amdclang++")
    if found:
        return Path(found)
    purelib = Path(sysconfig.get_paths()["purelib"])
    sdk = purelib / "_rocm_sdk_devel"
    candidates = (
        sdk / "lib" / "llvm" / "bin" / "amdclang++",
        sdk / "bin" / "amdclang++",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ToolchainError(
        "cannot find amdclang++; set GGTENSILE_AMDCLANGXX or add it to PATH"
    )


def _find_sibling_tool(tool_dir: Path, name: str) -> Path:
    candidate = tool_dir / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which(name)
    if found:
        return Path(found).resolve()
    raise ToolchainError(f"cannot find {name} next to {tool_dir} or on PATH")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise ToolchainError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result
