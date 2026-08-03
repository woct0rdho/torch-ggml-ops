import hashlib
import os
import tempfile
from pathlib import Path

import rocisa


class Assembly:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def comment(self, text: str) -> None:
        self.lines.append(f"// {text}")

    def label(self, name: str) -> None:
        self.lines.append(f"{name}:")

    def inst(self, text: str, comment: str = "") -> None:
        suffix = f" // {comment}" if comment else ""
        self.lines.append(f"  {text}{suffix}")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def initialize_rocisa(
    isa: tuple[int, int, int],
    wavefront_size: int,
    assembler: Path,
    *,
    temporary_prefix: str,
) -> None:
    global_isa = rocisa.rocIsa.getInstance()  # ty: ignore[unresolved-attribute]
    original_directory = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=temporary_prefix) as temporary:
        os.chdir(temporary)
        try:
            global_isa.init(isa, str(assembler), False)
        finally:
            os.chdir(original_directory)
    global_isa.setKernel(isa, wavefront_size)


def write_assembly_source(output: Path, source: str) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(source, encoding="utf-8")
    temporary.replace(output)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def emit_pointer_kernarg_loads(assembly: Assembly, kernarg: int) -> None:
    assembly.inst(f"s_load_dwordx2 s[{kernarg}:{kernarg + 1}], s[0:1], 0x0")
    assembly.inst(f"s_load_dwordx2 s[{kernarg + 2}:{kernarg + 3}], s[0:1], 0x8")
    assembly.inst(f"s_load_dwordx2 s[{kernarg + 4}:{kernarg + 5}], s[0:1], 0x10")
    assembly.inst("s_waitcnt lgkmcnt(0)")


def emit_kernel_trailer(assembly: Assembly, kernel_name: str) -> None:
    assembly.inst("s_endpgm")
    assembly.lines.append(f".L{kernel_name}_end:")
    assembly.lines.append(f".size {kernel_name}, .L{kernel_name}_end - {kernel_name}")


def emit_bf16_rne(
    assembly: Assembly,
    value_register: int,
    temporary_register: int,
) -> None:
    assembly.inst(f"v_bfe_u32 v{temporary_register}, v{value_register}, 16, 1")
    assembly.inst(
        f"v_add3_u32 v{value_register}, v{temporary_register}, "
        f"v{value_register}, 0x7fff"
    )


def emit_scale_u32(
    assembly: Assembly,
    destination: int,
    scale: int,
    source: int,
) -> None:
    if scale > 0 and scale & (scale - 1) == 0:
        shift = scale.bit_length() - 1
        assembly.inst(f"v_lshlrev_b32 v{destination}, {shift}, v{source}")
    else:
        assembly.inst(f"v_mul_lo_u32 v{destination}, {scale}, v{source}")


def emit_add_pointer(
    assembly: Assembly,
    destination: int,
    scalar_pointer: int,
    offset: int,
) -> None:
    assembly.inst(f"v_add_co_u32 v{destination}, vcc_lo, s{scalar_pointer}, v{offset}")
    assembly.inst(
        f"v_add_co_ci_u32_e64 v{destination + 1}, null, "
        f"s{scalar_pointer + 1}, 0, vcc_lo"
    )
