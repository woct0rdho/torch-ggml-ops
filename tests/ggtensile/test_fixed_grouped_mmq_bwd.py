from dataclasses import replace

import pytest

from tools.ggtensile.fixed_grouped_mmq_bwd_model import (
    FixedBackwardProblem,
    FixedBackwardSolution,
    FixedBackwardSolutionKey,
)
from tools.ggtensile.fixed_grouped_mmq_bwd_spec import DerivedFixedBackwardState
from tools.ggtensile.fixed_grouped_mmq_bwd_validation import (
    fixed_backward_rejection_reason,
    validate_fixed_backward_solution_key,
)
from tools.ggtensile.inspection import inspect_artifact
from tools.ggtensile.kernel_writer_assembly_fixed_grouped_mmq_bwd import (
    FixedGroupedBackwardKernelWriterAssembly,
)
from tools.ggtensile.model import ProblemSize, ProblemType, SolutionKey
from tools.ggtensile.runtime import (
    FixedGroupedQ8BackwardModule,
    InstalledFixedGroupedQ8BackwardModule,
)
from tools.ggtensile.schema import SchemaError
from tools.ggtensile.toolchain import Toolchain
from tools.ggtensile.validation import validate_solution


def _key(tokens: int = 2048) -> FixedBackwardSolutionKey:
    return FixedBackwardSolutionKey(
        FixedBackwardProblem.deepseek_q8_0(tokens),
        FixedBackwardSolution.q8_0_m256_n64_k32(),
    )


def _selected_key(tokens: int = 2048) -> FixedBackwardSolutionKey:
    return FixedBackwardSolutionKey(
        FixedBackwardProblem.deepseek_q8_0(tokens),
        FixedBackwardSolution.selected_q8_0(),
    )


def _e9_key(tokens: int = 32768) -> FixedBackwardSolutionKey:
    return FixedBackwardSolutionKey(
        FixedBackwardProblem.deepseek_q8_0(tokens),
        FixedBackwardSolution.q8_0_m64_n256_k64(),
    )


def test_fixed_backward_identity_roundtrip_and_derived_state() -> None:
    key = _key()
    assert FixedBackwardSolutionKey.from_mapping(key.to_mapping()) == key
    validate_fixed_backward_solution_key(key)
    state = DerivedFixedBackwardState.from_solution_key(key)
    assert state.grid == (64, 8, 8)
    assert state.expected_grad_output_shape == (2048, 8, 1024)
    assert state.expected_packed_weight_shape == (8, 1024, 4352)
    assert state.expected_grad_input_shape == (2048, 8, 4096)
    assert state.physical.resources.total_vgprs == 231
    assert state.physical.resources.total_sgprs == 17
    assert state.physical.resources.lds_num_bytes == 5120


@pytest.mark.parametrize("tokens", (2048, 8192, 32768))
def test_fixed_backward_accepts_only_production_shapes(tokens: int) -> None:
    state = DerivedFixedBackwardState.from_solution_key(_selected_key(tokens))
    assert state.grid == (32, tokens // 128, 8)
    assert state.physical.resources.total_vgprs == 216


def test_fixed_backward_square_review_geometry_is_strict_and_derived() -> None:
    key = FixedBackwardSolutionKey(
        FixedBackwardProblem.deepseek_q8_0(32768),
        FixedBackwardSolution.q8_0_m128_n128_k32(),
    )
    assert FixedBackwardSolutionKey.from_mapping(key.to_mapping()) == key
    state = DerivedFixedBackwardState.from_solution_key(key)
    assert state.grid == (32, 256, 8)
    assert state.spec.compute.geometry.macro_tile0 == 128
    assert state.spec.compute.geometry.macro_tile1 == 128


def test_fixed_backward_square_vopd_review_identity_is_derived() -> None:
    key = FixedBackwardSolutionKey(
        FixedBackwardProblem.deepseek_q8_0(32768),
        FixedBackwardSolution.q8_0_m128_n128_k32_packed_vopd(),
    )
    assert FixedBackwardSolutionKey.from_mapping(key.to_mapping()) == key
    state = DerivedFixedBackwardState.from_solution_key(key)
    assert state.spec.compute.decode.q8.extraction.value == "packed_vopd"


def test_fixed_backward_m64_review_crosses_vgpr_boundary() -> None:
    key = FixedBackwardSolutionKey(
        FixedBackwardProblem.deepseek_q8_0(32768),
        FixedBackwardSolution.q8_0_m64_n128_k32(),
    )
    state = DerivedFixedBackwardState.from_solution_key(key)
    assert state.grid == (32, 512, 8)
    assert state.physical.resources.total_vgprs == 122


def test_fixed_backward_clause_store_is_an_exact_epilogue_identity() -> None:
    key = FixedBackwardSolutionKey(
        FixedBackwardProblem.deepseek_q8_0(32768),
        FixedBackwardSolution.q8_0_m128_n128_k32_clause_store(),
    )
    assert FixedBackwardSolutionKey.from_mapping(key.to_mapping()) == key
    source = FixedGroupedBackwardKernelWriterAssembly(
        key, Toolchain.discover()
    ).source()
    assert source.count("s_clause 15") == 8


def test_fixed_backward_depth64_review_identity_is_derived() -> None:
    key = FixedBackwardSolutionKey(
        FixedBackwardProblem.deepseek_q8_0(32768),
        FixedBackwardSolution.q8_0_m128_n128_k64(),
    )
    state = DerivedFixedBackwardState.from_solution_key(key)
    assert state.spec.compute.geometry.depth_u == 64
    assert state.physical.resources.lds_num_bytes == 18432
    assert FixedBackwardSolution.selected_q8_0() == key.solution


def test_fixed_backward_depth64_vopd_review_identity_is_derived() -> None:
    key = FixedBackwardSolutionKey(
        FixedBackwardProblem.deepseek_q8_0(32768),
        FixedBackwardSolution.q8_0_m128_n128_k64_packed_vopd(),
    )
    state = DerivedFixedBackwardState.from_solution_key(key)
    assert state.spec.compute.decode.q8.extraction.value == "packed_vopd"


def test_fixed_backward_depth64_next_prefetch_is_rejected() -> None:
    key = FixedBackwardSolutionKey(
        FixedBackwardProblem.deepseek_q8_0(32768),
        FixedBackwardSolution.q8_0_m128_n128_k64_next_prefetch(),
    )
    with pytest.raises(ValueError, match="DepthU64 next-packed-tile prefetch"):
        DerivedFixedBackwardState.from_solution_key(key)


def test_fixed_backward_e9_m64_n256_depth64_is_fixed_only() -> None:
    key = _e9_key()
    assert FixedBackwardSolutionKey.from_mapping(key.to_mapping()) == key
    validate_fixed_backward_solution_key(key)
    state = DerivedFixedBackwardState.from_solution_key(key)
    assert state.grid == (16, 512, 8)
    assert state.spec.compute.geometry.matrix_instruction[6] == 16
    assert state.physical.ordinary.decoder.rows == 8
    assert state.physical.ordinary.address.lds == 208
    assert state.physical.ordinary.address.quant_shift == 209
    assert state.physical.resources.total_vgprs == 230
    assert state.physical.resources.total_sgprs == 17
    assert state.physical.resources.lds_num_bytes == 36864

    ordinary = SolutionKey(
        ProblemType.mmq_backward("Q8_0"),
        ProblemSize(32768, 4096, 1024),
        key.solution.compute,
    )
    assert any(
        reason.rule_id == "solution.geometry.unimplemented"
        for reason in validate_solution(ordinary)
    )


@pytest.mark.parametrize(
    ("key", "message"),
    (
        (
            replace(_key(), problem=replace(_key().problem, tokens=4096)),
            "tokens must be one of",
        ),
        (
            replace(_key(), problem=replace(_key().problem, groups=4)),
            "eight groups",
        ),
        (
            replace(
                _key(),
                solution=replace(_key().solution, group_axis="SyntheticRoutes"),
            ),
            "workgroup Z",
        ),
        (
            replace(
                _key(),
                solution=replace(
                    _key().solution,
                    compute=replace(_key().solution.compute, macro_tile0=192),
                ),
            ),
            "complete M tiles",
        ),
    ),
)
def test_fixed_backward_rejects_cross_contract_values(
    key: FixedBackwardSolutionKey, message: str
) -> None:
    assert fixed_backward_rejection_reason(key) is not None
    with pytest.raises(ValueError, match=message):
        validate_fixed_backward_solution_key(key)


def test_fixed_backward_mapping_rejects_unknown_and_noncanonical_fields() -> None:
    mapping = _key().to_mapping()
    mapping["unknown"] = 1
    with pytest.raises(SchemaError, match="unknown"):
        FixedBackwardSolutionKey.from_mapping(mapping)

    mapping = _key().to_mapping()
    contract = mapping["ProblemContract"]
    assert isinstance(contract, dict)
    contract["abi"] = "Routed"
    with pytest.raises(SchemaError, match="not canonical"):
        FixedBackwardSolutionKey.from_mapping(mapping)


def test_fixed_backward_source_uses_group_bases_and_interleaved_rows() -> None:
    key = _key()
    source = FixedGroupedBackwardKernelWriterAssembly(
        key, Toolchain.discover()
    ).source()
    assert "s_load_dword s16, s[0:1], 0x20" in source
    assert "s_mov_b32 s16, s2" in source
    assert "s_mov_b32 s2, s3" in source
    assert "s_mov_b32 s3, s16" in source
    assert "s_mul_i32 s16, s4, s16" in source
    assert "s_add_u32 s8, s8, s16" in source
    assert "s_add_u32 s6, s6, s16" in source
    assert "s_add_u32 s10, s10, s16" in source
    assert source.count("v_dual_lshlrev_b32 v218, 14, v218") == 1
    assert "v_lshlrev_b32 v224, 16, v224" in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32


def test_fixed_backward_build_is_deterministic_and_inspectable(tmp_path) -> None:
    key = _selected_key()
    toolchain = Toolchain.discover()
    writer = FixedGroupedBackwardKernelWriterAssembly(key, toolchain)
    assert (
        writer.source()
        == FixedGroupedBackwardKernelWriterAssembly(key, toolchain).source()
    )
    assembly = tmp_path / "kernel.s"
    obj = tmp_path / "kernel.o"
    code_object = tmp_path / "kernel.hsaco"
    writer.write(assembly)
    toolchain.assemble(assembly, obj)
    toolchain.link(obj, code_object)
    inspection = inspect_artifact(key, code_object, toolchain)
    assert inspection.kernarg_segment_size == 40
    assert inspection.lds_num_bytes == 18432
    assert inspection.vgpr_count == 216
    assert inspection.sgpr_count == 17
    assert inspection.max_sgpr_index == 16
    assert inspection.wmma_count == 64
    assert inspection.barrier_count == 2
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


def test_fixed_backward_runtime_configurations_match_candidate_and_controls() -> None:
    candidate = object.__new__(FixedGroupedQ8BackwardModule)
    candidate.state = DerivedFixedBackwardState.from_solution_key(_selected_key())
    assert candidate._launch_configuration() == (32, 16, 8, 32, 4, 1, 0)

    control = object.__new__(InstalledFixedGroupedQ8BackwardModule)
    control.state = candidate.state
    assert control._launch_configuration() == (64, 8, 8, 128, 1, 1, 0)
    control.state = DerivedFixedBackwardState.from_solution_key(_selected_key(32768))
    assert control._launch_configuration() == (64, 171, 8, 128, 1, 1, 0)
    candidate.state = DerivedFixedBackwardState.from_solution_key(_e9_key(32768))
    assert candidate._launch_configuration() == (16, 512, 8, 32, 4, 1, 0)
