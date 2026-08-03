"""Targeted branch and complete executable-line coverage for the forward writer."""

import pytest

from tests.ggtensile.support import (
    FWD_WRITER_SOURCE_PATH,
    assert_writer_methods_have_complete_line_coverage,
)
from tools.ggtensile import kernel_writer_assembly_mmq_fwd as fwd_writer_module
from tools.ggtensile.kernel_writer_assembly_mmq_fwd import (
    ForwardKernelWriterAssembly,
    ForwardKernelWriterError,
)
from tools.ggtensile.model import (
    BackwardSolution,
    ForwardSolution,
    ProblemSize,
    ProblemType,
    SolutionKey,
)
from tools.ggtensile.toolchain import Toolchain
from tools.ggtensile.validation import validate_solution


def test_writer_rejects_backward_solution_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = SolutionKey(
        ProblemType.mmq_forward("Q4_K"),
        ProblemSize(2048, 512, 2048),
        BackwardSolution.pilot(),
    )
    monkeypatch.setattr(fwd_writer_module, "validate_solution", lambda _: ())
    with pytest.raises(ForwardKernelWriterError, match="requires ForwardSolution"):
        ForwardKernelWriterAssembly(key, Toolchain.discover())


@pytest.mark.parametrize("macro_tile0", (64, 128, 256))
def test_writer_emits_q6_forward_controls(macro_tile0: int) -> None:
    solution = ForwardSolution.q6_k_decoded_staged(macro_tile0=macro_tile0)
    key = SolutionKey(
        ProblemType.mmq_forward("Q6_K"),
        ProblemSize(macro_tile0, 248320, 2048),
        solution,
    )
    assert validate_solution(key) == ()
    source = ForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Decode low/high Q6_K planes" in source
    assert "v_wmma_i32_16x16x16_iu8" in source
    assert "v_dual_fmac_f32" in source
    assert "F32_D4" in source
    if macro_tile0 == 256:
        assert "v_and_b32 v198, 3, v205" in source
    elif macro_tile0 == 128:
        assert "v_lshlrev_b32 v198, 4, v205" in source
    else:
        assert "v_lshlrev_b32" in source


def test_writer_emits_single_dependency_scheduled_epilogue() -> None:
    solution = ForwardSolution.q5_k_hip_decoded_staged_extraction(
        epilogue_tiles_ahead=1,
        epilogue_dependency_width=1,
        epilogue_priority=0,
    )
    key = SolutionKey(
        ProblemType.mmq_forward("Q5_K"),
        ProblemSize(2048, 512, 2048),
        solution,
    )
    assert validate_solution(key) == ()
    source = ForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert source.count("v_bfe_u32 v228,") == 64


def test_writer_methods_have_complete_line_coverage() -> None:
    assert_writer_methods_have_complete_line_coverage(FWD_WRITER_SOURCE_PATH)
