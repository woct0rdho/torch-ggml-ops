from pathlib import Path

import pytest

from tools.ggtensile.inspection import inspect_artifact
from tools.ggtensile.kernel_writer_assembly import KernelWriterAssembly
from tools.ggtensile.model import ProblemSize, ProblemType, Solution, SolutionKey
from tools.ggtensile.toolchain import Toolchain, ToolchainError
from tools.ggtensile.validation import validate_solution


def _pilot_key() -> SolutionKey:
    return SolutionKey(
        ProblemType.dense_mmq_backward_q4_k(),
        ProblemSize(128, 2048, 512),
        Solution.pilot(),
    )


def _toolchain() -> Toolchain:
    try:
        return Toolchain.discover()
    except ToolchainError as error:
        pytest.skip(str(error))


def test_pilot_solution_identity_and_round_trip() -> None:
    key = _pilot_key()
    assert validate_solution(key) == ()
    assert key.hash == "ggsol_e2cace7280dd3cb8"
    assert SolutionKey.from_mapping(key.to_mapping()) == key


def test_writer_enables_and_flattens_packed_workitem_xy() -> None:
    source = KernelWriterAssembly(_pilot_key(), _toolchain()).source()
    assert ".amdhsa_system_vgpr_workitem_id 1" in source
    assert "v_bfe_u32 v195, v0, 10, 10" in source
    assert "v_add_nc_u32 v195, v195, v188" in source


def test_build_and_inspect_pilot(tmp_path: Path) -> None:
    key = _pilot_key()
    toolchain = _toolchain()
    assembly = tmp_path / "pilot.s"
    object_path = tmp_path / "pilot.o"
    code_object = tmp_path / "pilot.hsaco"
    KernelWriterAssembly(key, toolchain).write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)

    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.target == "gfx1151"
    assert inspection.code_object_version == 5
    assert inspection.vgpr_count == 196
    assert inspection.sgpr_count == 19
    assert inspection.max_vgpr_index == 195
    assert inspection.max_sgpr_index == 18
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0
    assert inspection.wmma_count == 32
    assert inspection.barrier_count == 2
