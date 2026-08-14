import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from tools.ggtensile.grouped_mmq_fwd_inspection import (
    inspect_grouped_forward_artifact,
)
from tools.ggtensile.grouped_mmq_fwd_model import (
    GroupedForwardProblem,
    GroupedForwardSolution,
    GroupedForwardSolutionKey,
)
from tools.ggtensile.grouped_mmq_fwd_spec import DerivedGroupedForwardState
from tools.ggtensile.grouped_mmq_fwd_validation import (
    validate_grouped_forward_solution,
)
from tools.ggtensile.kernel_writer_assembly_grouped_mmq_fwd import (
    GroupedForwardKernelWriterAssembly,
)
from tools.ggtensile.toolchain import Toolchain


def _key(aggregate_rows: int = 16384) -> GroupedForwardSolutionKey:
    return GroupedForwardSolutionKey(
        GroupedForwardProblem.q4_k(aggregate_rows),
        GroupedForwardSolution.q4_k_serial_direct(),
    )


@pytest.mark.parametrize("aggregate_rows", (16384, 65536, 262144))
def test_grouped_q4_k_exact_production_keys_derive(
    aggregate_rows: int,
) -> None:
    key = _key(aggregate_rows)
    assert validate_grouped_forward_solution(key) == ()
    state = DerivedGroupedForwardState.from_solution_key(key)
    assert state.expected_packed_weight_shape == (256, 2048, 288)
    assert state.expected_activation_shape == (4, aggregate_rows, 144)
    assert state.expected_output_shape == (aggregate_rows, 2048)
    assert state.grid(256) == (128, 256, 1)


def test_grouped_q4_k_identity_round_trips_strict_mapping() -> None:
    key = _key(35)
    assert GroupedForwardSolutionKey.from_mapping(key.to_mapping()) == key
    assert key.kernel_name.endswith(key.hash[6:])


def test_grouped_q4_k_rejects_inactive_solution_fields() -> None:
    solution = GroupedForwardSolution.q4_k_serial_direct()
    key = GroupedForwardSolutionKey(
        GroupedForwardProblem.q4_k(35),
        replace(solution, group_mapping="Flattened"),
    )
    reasons = validate_grouped_forward_solution(key)
    assert [reason.rule_id for reason in reasons] == [
        "grouped_forward.solution.unimplemented"
    ]


def test_grouped_q4_k_writer_emits_routed_abi_and_masks() -> None:
    source = GroupedForwardKernelWriterAssembly(_key(35), Toolchain.discover()).source()
    assert "s_load_dwordx2 s[10:11], s[0:1], 0x18" in source
    assert "s_load_dwordx2 s[16:17], s[0:1], 0x30" in source
    assert "s_load_dwordx2 s[18:19], s[0:1], 0x38" in source
    assert "s_load_dwordx2" in source
    assert source.count("s_and_saveexec_b32 s28, vcc_lo") == 9
    assert source.count("s_mov_b32 exec_lo, s28") == 9
    assert "s_load_dwordx2 s[22:23], s[10:11], s24" in source
    assert "s_mul_i32 s24, s22, s18" in source
    assert "s_cmp_lg_u32 s23, 0" in source
    assert ".LGroupedQ4KRowLoop:" in source
    assert ".LGroupedQ4KBlockLoop:" in source


def test_grouped_q4_k_artifact_passes_strict_inspection(tmp_path: Path) -> None:
    key = _key(35)
    toolchain = Toolchain.discover()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_writer = GroupedForwardKernelWriterAssembly(key, toolchain)
    second_writer = GroupedForwardKernelWriterAssembly(key, toolchain)
    first_assembly = first_dir / "kernel.s"
    first_object = first_dir / "kernel.o"
    first_code_object = first_dir / "kernel.hsaco"
    second_assembly = second_dir / "kernel.s"
    second_object = second_dir / "kernel.o"
    second_code_object = second_dir / "kernel.hsaco"
    first_hash = first_writer.write(first_assembly)
    second_hash = second_writer.write(second_assembly)
    assert first_hash == second_hash
    assert first_assembly.read_bytes() == second_assembly.read_bytes()
    toolchain.assemble(first_assembly, first_object)
    toolchain.link(first_object, first_code_object)
    toolchain.assemble(second_assembly, second_object)
    toolchain.link(second_object, second_code_object)
    assert (
        hashlib.sha256(first_code_object.read_bytes()).digest()
        == hashlib.sha256(second_code_object.read_bytes()).digest()
    )
    inspection = inspect_grouped_forward_artifact(key, first_code_object, toolchain)
    assert inspection.kernarg_segment_size == 64
    assert inspection.vgpr_count == 88
    assert inspection.sgpr_count == 32
    assert inspection.wmma_count == 16
    assert inspection.barrier_count == 0
    assert inspection.lds_num_bytes == 0
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0
