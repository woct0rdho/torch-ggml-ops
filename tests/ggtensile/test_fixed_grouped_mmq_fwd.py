from dataclasses import replace

import pytest

from tools.ggtensile.fixed_grouped_mmq_fwd_model import (
    FixedForwardProblem,
    FixedForwardSolution,
    FixedForwardSolutionKey,
)
from tools.ggtensile.fixed_grouped_mmq_fwd_spec import DerivedFixedForwardState
from tools.ggtensile.fixed_grouped_mmq_fwd_validation import (
    fixed_forward_rejection_reason,
    validate_fixed_forward_solution_key,
)
from tools.ggtensile.inspection import inspect_artifact
from tools.ggtensile.kernel_writer_assembly_fixed_grouped_mmq_fwd import (
    FixedGroupedForwardKernelWriterAssembly,
)
from tools.ggtensile.toolchain import Toolchain


def _key(tokens: int = 2048) -> FixedForwardSolutionKey:
    return FixedForwardSolutionKey(
        FixedForwardProblem.deepseek_q8_0(tokens),
        FixedForwardSolution.q8_0_small_m_tiled_lds(),
    )


def test_fixed_forward_identity_roundtrip_and_derived_shapes() -> None:
    key = _key()
    assert FixedForwardSolutionKey.from_mapping(key.to_mapping()) == key
    validate_fixed_forward_solution_key(key)
    state = DerivedFixedForwardState.from_solution_key(key)
    assert state.grid == (16, 32, 8)
    assert state.activation_plane_stride_bytes == 8 * 2048 * 144
    assert state.expected_packed_weight_shape == (8, 1024, 4352)
    assert state.expected_activation_shape == (32, 16_384, 144)
    assert state.expected_output_shape == (2048, 8, 1024)
    assert state.resources.lds_bytes == 28_672
    assert state.resources.vgprs == 144
    assert state.resources.sgprs == 16


@pytest.mark.parametrize(
    "bad_key, message",
    [
        (
            replace(
                _key(),
                problem=replace(_key().problem, groups=4),
            ),
            "eight groups",
        ),
        (
            replace(
                _key(),
                solution=replace(_key().solution, work_group=(64, 2, 1)),
            ),
            "workgroup",
        ),
        (
            replace(
                _key(),
                problem=replace(_key().problem, tokens=4096),
            ),
            "production token count",
        ),
    ],
)
def test_fixed_forward_rejects_cross_contract_values(
    bad_key: FixedForwardSolutionKey,
    message: str,
) -> None:
    assert fixed_forward_rejection_reason(bad_key) is not None
    with pytest.raises(ValueError, match=message):
        validate_fixed_forward_solution_key(bad_key)


def test_fixed_forward_source_loads_scalars_and_flattens_group_rows() -> None:
    key = _key()
    source = FixedGroupedForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    state = DerivedFixedForwardState.from_solution_key(key)
    output_address = state.fixed_physical_plan.registers.output_address.first_register
    temporary = state.fixed_physical_plan.registers.temporary.first_register
    assert "s_load_dword s12, s[0:1], 0x18" in source
    assert "s_load_dword s13, s[0:1], 0x1c" in source
    assert "s_load_dwordx2 s[14:15], s[0:1], 0x20" in source
    assert f"v_mul_lo_u32 v{output_address}, s13, v{output_address}" in source
    assert f"v_lshlrev_b32 v{temporary}, 8, s13" in source
    assert "s[10:11]" in source
    assert "s[8:9]" in source


def test_fixed_forward_build_is_deterministic_and_inspectable(tmp_path) -> None:
    key = _key()
    toolchain = Toolchain.discover()
    first = FixedGroupedForwardKernelWriterAssembly(key, toolchain)
    second = FixedGroupedForwardKernelWriterAssembly(key, toolchain)
    first_source = first.source()
    second_source = second.source()
    assert first_source == second_source

    first_assembly = tmp_path / "first.s"
    first_object = tmp_path / "first.o"
    first_code_object = tmp_path / "first.hsaco"
    first.write(first_assembly)
    toolchain.assemble(first_assembly, first_object)
    toolchain.link(first_object, first_code_object)
    inspection = inspect_artifact(key, first_code_object, toolchain)
    assert inspection.kernarg_segment_size == 40
    assert inspection.lds_num_bytes == 28_672
    assert inspection.wmma_count == 32
    assert inspection.barrier_count == 2
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0
