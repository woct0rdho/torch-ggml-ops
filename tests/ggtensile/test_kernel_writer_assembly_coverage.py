"""Targeted branch and complete executable-line coverage for shared assembly support."""

import hashlib
from pathlib import Path

from tests.ggtensile.support import (
    SHARED_WRITER_SOURCE_PATH,
    assert_writer_methods_have_complete_line_coverage,
)
from tools.ggtensile.kernel_writer_assembly import (
    Assembly,
    emit_add_pointer,
    emit_bf16_rne,
    emit_kernel_trailer,
    emit_pointer_kernarg_loads,
    emit_scale_u32,
    write_assembly_source,
)


def test_shared_assembly_primitives_and_source_write(tmp_path: Path) -> None:
    assembly = Assembly()
    assembly.comment("shared")
    assembly.label(".LShared")
    assembly.inst("s_nop 0", "comment")
    emit_pointer_kernarg_loads(assembly, 4)
    emit_scale_u32(assembly, 0, 8, 1)
    emit_scale_u32(assembly, 2, 3, 4)
    emit_add_pointer(assembly, 6, 8, 10)
    emit_bf16_rne(assembly, 12, 13)
    emit_kernel_trailer(assembly, "shared")
    source = assembly.text()
    output = tmp_path / "shared.s"
    digest = write_assembly_source(output, source)
    assert output.read_text(encoding="utf-8") == source
    assert digest == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert "v_lshlrev_b32 v0, 3, v1" in source
    assert "v_mul_lo_u32 v2, 3, v4" in source


def test_writer_methods_have_complete_line_coverage() -> None:
    assert_writer_methods_have_complete_line_coverage(SHARED_WRITER_SOURCE_PATH)
