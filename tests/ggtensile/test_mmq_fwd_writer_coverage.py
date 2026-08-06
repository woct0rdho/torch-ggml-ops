"""Targeted branch and complete executable-line coverage for the forward writer."""

import ast
import hashlib
import re
from dataclasses import fields, replace
from typing import Any, cast

import pytest

from tests.ggtensile.support import (
    FWD_WRITER_SOURCE_PATH,
    assert_writer_methods_have_complete_line_coverage,
)
from tools.ggtensile import kernel_writer_assembly_mmq_fwd as fwd_writer_module
from tools.ggtensile.kernel_writer_assembly_mmq_fwd import (
    ForwardKernelWriterAssembly,
    ForwardKernelWriterError,
    Q8SmallMTiledLdsRegisterPlan,
    _emit_q6_dot_phase,
    _emit_q6_scheduled_body,
)
from tools.ggtensile.mmq_fwd_search import q6_schedule_candidates
from tools.ggtensile.mmq_fwd_spec import Q6ForwardSchedule, q6_schedule_from_solution
from tools.ggtensile.model import (
    BackwardSolution,
    ForwardSolution,
    ProblemSize,
    ProblemType,
    SolutionKey,
)
from tools.ggtensile.toolchain import Toolchain
from tools.ggtensile.validation import validate_solution


def _q6_schedule(macro_tile0: int) -> Q6ForwardSchedule:
    return q6_schedule_from_solution(
        ForwardSolution.q6_k_structured_decoded(macro_tile0=macro_tile0)
    )


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


@pytest.mark.parametrize(
    ("m", "macro_tile0", "wmma_count", "delay_count"),
    ((64, 64, 8, 0), (128, 128, 16, 109), (256, 128, 16, 109)),
)
def test_writer_emits_q6_structured_decoded_controls(
    m: int,
    macro_tile0: int,
    wmma_count: int,
    delay_count: int,
) -> None:
    solution = ForwardSolution.q6_k_structured_decoded(macro_tile0=macro_tile0)
    key = SolutionKey(
        ProblemType.mmq_forward("Q6_K"),
        ProblemSize(m, 248320, 2048),
        solution,
    )
    assert validate_solution(key) == ()
    source = ForwardKernelWriterAssembly(key, Toolchain.discover()).source()
    assert "Deterministic Q6_K lowering" in source
    assert "ScheduleIterAlg=3" in source
    assert not any(
        line.strip().startswith(
            ("DenseFwdQ6K", ".globl DenseFwdQ6K", ".type DenseFwdQ6K")
        )
        for line in source.splitlines()
    )
    assert source.count("v_wmma_i32_16x16x16_iu8") == wmma_count
    assert (
        sum(line.lstrip().startswith("s_delay_alu") for line in source.splitlines())
        == delay_count
    )
    assert source.count("s_barrier") == 4
    assert source.count("buffer_gl0_inv") == 4
    assert source.count("s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)") == 1


@pytest.mark.parametrize(
    ("macro_tile0", "expected_sha256"),
    (
        (64, "c36e63b542ae7a9894fd02d85c1bd0e4ad6bef447329b831a65901f6689a7dfa"),
        (128, "33f468f4eeb7a99ad31948bdcc3157411707098cf4dbaacd1e8bb1bac88661c9"),
    ),
)
def test_q6_scheduled_body_matches_promoted_instruction_stream(
    macro_tile0: int,
    expected_sha256: str,
) -> None:
    source = str(_emit_q6_scheduled_body(_q6_schedule(macro_tile0)))
    assert hashlib.sha256(source.encode()).hexdigest() == expected_sha256


@pytest.mark.parametrize("macro_tile0", (64, 128))
def test_q6_schedule_exposes_semantic_components(macro_tile0: int) -> None:
    module = _emit_q6_scheduled_body(_q6_schedule(macro_tile0))
    assert tuple(item.name for item in module.items()) == (
        "lane_and_address_setup",
        "cooperative_global_loads",
        "packed_decode",
        "decoded_lds_writes",
        "first_stage_barrier",
        "q6_dot_phase_0",
        "second_stage_loads",
        "q6_dot_phase_1",
        "bf16_epilogue",
    )


@pytest.mark.parametrize(("macro_tile0", "group_count"), ((64, 2), (128, 4)))
def test_q6_physical_register_map_uses_complete_output_roles(
    macro_tile0: int,
    group_count: int,
) -> None:
    layout = fwd_writer_module._q6_physical_layout(_q6_schedule(macro_tile0))
    register_map = layout.registers
    output_roles = tuple(
        role
        for group in range(group_count)
        for role in register_map.group_output_roles(group)
    )
    assert len(output_roles) == 16 * group_count // 2
    assert {
        register
        for role in output_roles
        for register in (role.left_register, role.right_register)
    } == set(layout.accumulator_registers)
    assert {
        product
        for role in output_roles
        for product in (role.left_product_index, role.right_product_index)
    } == set(range(16 * group_count))
    assert all(
        (role.lifetime.first_stage, role.lifetime.last_stage) == (5, 8)
        for role in output_roles
    )
    assert register_map.lifetime(
        "weight_address"
    ) == fwd_writer_module.RegisterLifetime(0, 8)
    assert register_map.lifetime("product") == fwd_writer_module.RegisterLifetime(5, 7)
    assert register_map.lifetime(
        "first_stage_write_address"
    ) == fwd_writer_module.RegisterLifetime(1, 4)
    assert register_map.lifetime(
        "first_stage_read_address"
    ) == fwd_writer_module.RegisterLifetime(4, 5)
    assert not hasattr(layout, "accumulator_transfer_prefix")


def test_q6_schedule_policy_knobs_are_explicit() -> None:
    selected = _q6_schedule(128)
    assert selected.macro_tile == (128, 64)
    assert selected.work_group == (32, 4, 1)
    assert selected.matrix_instruction == (16, 16, 16, 1)
    assert selected.wmma_opcode == "v_wmma_i32_16x16x16_iu8"
    assert selected.mi_wave_group == (4, 1)
    assert selected.mi_wave_tile == (2, 4)
    assert selected.depth_u == 32
    assert selected.local_read_vector_width == 2
    assert selected.store_vector_width == 8
    assert selected.epilogue_dependency_width == 2
    assert selected.epilogue_pipeline_scope == "FullTile"
    assert {field.name for field in fields(selected)} == {
        "matrix_instruction",
        "mi_wave_group",
        "mi_wave_tile",
        "semantic_policy",
        "epilogue_dependency_width",
        "epilogue_pipeline_scope",
        "dot_register_shifts",
        "dependency_delay_mode",
        "global_read_cache_policy",
    }
    assert (
        selected.resource_usage.vgprs,
        selected.resource_usage.sgprs,
        selected.resource_usage.lds_bytes,
    ) == (210, 27, 38_400)
    assert selected.semantic_policy.traversal == "OutputRoleGroupMajor"
    assert selected.semantic_policy.clustering == "StageDependencyOrder"
    assert selected.semantic_policy.latency == "SerializedDependencyDistance"
    assert selected.semantic_policy.pressure == "ExplicitRoleLifetime"
    assert selected.semantic_policy.wait == "ProducerFirstUse"
    assert selected.semantic_policy.pairing == "DependencyCompatibleDualIssue"
    with pytest.raises(ValueError, match="semantic schedule policy"):
        replace(
            selected,
            semantic_policy=replace(
                selected.semantic_policy,
                pairing="OpaqueIssueTable",
            ),
        )

    without_delays = str(
        _emit_q6_scheduled_body(replace(selected, dependency_delay_mode="None"))
    )
    without_invalidation = str(
        _emit_q6_scheduled_body(replace(selected, global_read_cache_policy="Default"))
    )
    assert "s_delay_alu" not in without_delays
    assert "buffer_gl0_inv" not in without_invalidation

    j64_candidates = q6_schedule_candidates(64)
    j128_candidates = q6_schedule_candidates(128)
    assert len(j64_candidates) == 16
    assert len(j128_candidates) == 32
    assert {candidate.global_read_cache_policy for candidate in j64_candidates} == {
        "Default",
        "InvalidateL0",
    }
    assert {candidate.dependency_delay_mode for candidate in j128_candidates} == {
        "None",
        "Explicit",
    }
    assert {candidate.epilogue_dependency_width for candidate in j64_candidates} == {
        1,
        2,
        4,
        8,
    }
    assert {candidate.epilogue_dependency_width for candidate in j128_candidates} == {
        1,
        2,
        4,
        8,
    }
    assert {candidate.epilogue_pipeline_scope for candidate in j64_candidates} == {
        "StoreBatch",
        "FullTile",
    }
    assert {candidate.epilogue_pipeline_scope for candidate in j128_candidates} == {
        "StoreBatch",
        "FullTile",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("latency", "OpaqueLatencyTable", "latency-lowering"),
        ("pairing", "OpaqueIssueTable", "dual-issue pairing"),
        ("traversal", "PhysicalRegisterOrder", "output traversal"),
        ("pressure", "SourceOrderFallback", "register-pressure"),
    ),
)
def test_q6_lowering_rejects_unknown_second_level_policy(
    field: str,
    value: str,
    message: str,
) -> None:
    schedule = _q6_schedule(64)
    object.__setattr__(
        schedule,
        "semantic_policy",
        replace(schedule.semantic_policy, **{field: value}),
    )
    with pytest.raises(ValueError, match=message):
        if field in {"latency", "pairing"}:
            fwd_writer_module.Q6ScheduleEmitter(schedule, "invalid")
        else:
            fwd_writer_module._q6_physical_layout(schedule)


def test_q6_solution_schedule_derivation_rejects_invalid_structure() -> None:
    structured = ForwardSolution.q6_k_structured_decoded(macro_tile0=64)
    with pytest.raises(ValueError, match="requires structured decoded operands"):
        q6_schedule_from_solution(ForwardSolution.q4_k_pilot())
    with pytest.raises(ValueError, match=r"WorkGroup=\(32,waves,1\)"):
        q6_schedule_from_solution(replace(structured, work_group=(64, 2, 1)))
    with pytest.raises(ValueError, match="at least one wave"):
        q6_schedule_from_solution(replace(structured, work_group=(32, 0, 1)))
    with pytest.raises(
        ValueError, match="not divisible by matrix-instruction ownership"
    ):
        q6_schedule_from_solution(replace(structured, macro_tile0=96))


def test_q6_schedule_rejects_unsupported_geometry_and_phase() -> None:
    with pytest.raises(ValueError, match="implements MT64 or MT128"):
        _q6_schedule(256)
    with pytest.raises(ValueError, match="unsupported Q6 dot phase: 2"):
        _q6_schedule(64).phase(2)
    selected = _q6_schedule(64)
    unsupported = replace(selected, mi_wave_tile=(4, 4))
    with pytest.raises(ValueError, match="unsupported Q6 physical layout"):
        _emit_q6_dot_phase(unsupported, 0)
    with pytest.raises(ValueError, match="unsupported Q6 physical layout"):
        _emit_q6_scheduled_body(unsupported)
    with pytest.raises(ValueError, match="blocks-per-weight-row must be positive"):
        _emit_q6_scheduled_body(selected, 0)
    with pytest.raises(ValueError, match="unsupported Q6 physical layout"):
        fwd_writer_module._q6_physical_layout(unsupported)
    opaque_stage = fwd_writer_module.Q6SemanticStage(
        "Opaque",  # ty: ignore[invalid-argument-type]
        (),
    )
    with pytest.raises(AssertionError, match="unhandled Q6 semantic stage"):
        fwd_writer_module._q6_emit_semantic_stage(opaque_stage, selected, 8)
    with pytest.raises(ValueError, match="one input block"):
        replace(selected, matrix_instruction=(16, 16, 16, 2))
    with pytest.raises(ValueError, match="at least one dot phase"):
        replace(selected, dot_register_shifts=())
    with pytest.raises(ValueError, match="epilogue dependency width"):
        replace(selected, epilogue_dependency_width=3)
    with pytest.raises(ValueError, match="epilogue pipeline scope"):
        replace(selected, epilogue_pipeline_scope="Opaque")
    with pytest.raises(ValueError, match="unsupported Q6 dependency-delay mode"):
        replace(selected, dependency_delay_mode="Automatic")
    with pytest.raises(ValueError, match="unsupported Q6 global-read cache policy"):
        replace(selected, global_read_cache_policy="Streaming")


def test_q6_semantic_emitter_rejects_unsupported_memory_operations() -> None:
    selected = _q6_schedule(64)
    emitter = fwd_writer_module.Q6ScheduleEmitter(selected, "invalid")
    assert str(fwd_writer_module.Q6Vgpr(2)) == "v2"
    assert str(fwd_writer_module.Q6Sgpr(3)) == "s3"
    assert str(fwd_writer_module.Q6Immediate(16)) == "16"
    assert str(fwd_writer_module.Q6Immediate(16, hexadecimal=True)) == "0x10"
    valu = fwd_writer_module.Q6DependencyDelay("VALU_DEP", 4)
    assert str(valu) == "instid0(VALU_DEP_4)"
    assert str(fwd_writer_module.Q6DependencyDelay("VALU_DEP", 4, "NEXT", 1)) == (
        "instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)"
    )
    with pytest.raises(ValueError, match="VGPR must be nonnegative"):
        fwd_writer_module.Q6Vgpr(-1)
    with pytest.raises(ValueError, match="SGPR must be nonnegative"):
        fwd_writer_module.Q6Sgpr(-1)
    with pytest.raises(ValueError, match="unsupported Q6 dependency kind"):
        fwd_writer_module.Q6DependencyDelay(cast(Any, "OPAQUE"), 1)
    with pytest.raises(ValueError, match="distance must be positive"):
        fwd_writer_module.Q6DependencyDelay("VALU_DEP", 0)
    with pytest.raises(ValueError, match="unsupported Q6 dependency skip"):
        fwd_writer_module.Q6DependencyDelay("VALU_DEP", 1, cast(Any, "SKIP_ALL"), 1)
    with pytest.raises(ValueError, match="requires skip and second"):
        fwd_writer_module.Q6DependencyDelay("VALU_DEP", 1, "NEXT")
    with pytest.raises(ValueError, match="requires skip and second"):
        fwd_writer_module.Q6DependencyDelay("VALU_DEP", 1, second_distance=1)
    with pytest.raises(ValueError, match="unsupported Q6 global-read width: 64"):
        fwd_writer_module.Q6GlobalRead(1, 2, width_bits=64)
    with pytest.raises(ValueError, match="destination must be one VGPR"):
        emitter.global_read_clause(
            (fwd_writer_module.Q6GlobalRead(cast(int, "v[1:2]"), 2),)
        )
    with pytest.raises(ValueError, match="address must be one VGPR pair"):
        fwd_writer_module.Q6GlobalRead(1, cast(int, "v[2:3]"))
    with pytest.raises(ValueError, match="does not match StoreVectorWidth"):
        emitter.store_bf16_clause("v[0:1]", (1, 2))


def test_q6_semantic_emitter_derives_refill_pair_order_from_slots() -> None:
    selected = _q6_schedule(64)
    emitter = fwd_writer_module.Q6ScheduleEmitter(selected, "refill")
    first = fwd_writer_module.Q6GlobalRead(8, 2, local_write_slot=0)
    second = fwd_writer_module.Q6GlobalRead(9, 2, local_write_slot=1)
    assert first.payload_assignment.first_register == 8
    assert first.payload_assignment.role.lifetime == fwd_writer_module.RegisterLifetime(
        6, 7
    )
    assert first.address_assignment.first_register == 2
    assert first.address_assignment.role.width == 2
    assert first.address_assignment.role.lifetime == fwd_writer_module.RegisterLifetime(
        5, 6
    )
    emitter.global_read_clause((first, second))
    emitter.commit_cooperative_global_reads(fwd_writer_module.Q6LdsLayout(1), 52)
    emitted = str(emitter.module())
    assert "ds_store_2addr_stride64_b32 v52, v8, v9 offset0:1 offset1:3" in emitted
    with pytest.raises(ValueError, match="nonnegative"):
        fwd_writer_module.Q6GlobalRead(1, 2, local_write_slot=-1)

    duplicate = fwd_writer_module.Q6ScheduleEmitter(selected, "duplicate")
    with pytest.raises(ValueError, match="duplicate Q6 local-write slot"):
        duplicate.global_read_clause(
            (
                fwd_writer_module.Q6GlobalRead(8, 2, local_write_slot=0),
                fwd_writer_module.Q6GlobalRead(9, 2, local_write_slot=0),
            )
        )

    incomplete = fwd_writer_module.Q6ScheduleEmitter(selected, "incomplete")
    incomplete.global_read_clause(
        (fwd_writer_module.Q6GlobalRead(8, 2, local_write_slot=1),)
    )
    with pytest.raises(ValueError, match="contiguous pairs"):
        incomplete.commit_cooperative_global_reads(fwd_writer_module.Q6LdsLayout(1), 52)


def test_q6_semantic_emitter_derives_vmem_waits_from_producers() -> None:
    emitter = fwd_writer_module.Q6ScheduleEmitter(_q6_schedule(64), "waits")
    reads = tuple(
        fwd_writer_module.Q6GlobalRead(register, 2) for register in (8, 9, 10)
    )
    assert reads[0].payload_assignment.role.lifetime == (
        fwd_writer_module.RegisterLifetime(1, 4)
    )
    assert reads[0].address_assignment.role.lifetime == (
        fwd_writer_module.RegisterLifetime(0, 1)
    )
    emitter.global_read_clause(reads)
    emitter.wait_for_global_sources((8, 9))
    emitter.wait_for_global_sources((8,))
    emitted = str(emitter.module())
    assert "s_waitcnt vmcnt(1)" in emitted
    assert emitted.count("s_waitcnt vmcnt") == 1
    with pytest.raises(ValueError, match="at least one global source"):
        emitter.wait_for_global_sources(())
    with pytest.raises(ValueError, match="has no global producer: v99"):
        emitter.wait_for_global_sources((99,))


def test_forward_writer_lowers_from_derived_state_after_boundary() -> None:
    tree = ast.parse(FWD_WRITER_SOURCE_PATH.read_text(encoding="utf-8"))
    legacy_reads = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == "solution"
        and isinstance(node.ctx, ast.Load)
    ]
    assert legacy_reads == []


def test_forward_writer_has_no_large_inline_assembly_collections() -> None:
    tree = ast.parse(FWD_WRITER_SOURCE_PATH.read_text(encoding="utf-8"))
    inline_assembly_collections = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.List, ast.Tuple))
        and sum(
            isinstance(element, ast.Constant)
            and isinstance(element.value, str)
            and element.value.startswith("\t")
            for element in node.elts
        )
        >= 16
    ]
    assert inline_assembly_collections == []

    opaque_instruction_tables = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and [
            element.id for element in node.target.elts if isinstance(element, ast.Name)
        ]
        == ["opcode", "operands"]
    ]
    assert opaque_instruction_tables == []
    assert "_q6_ordered_module" not in FWD_WRITER_SOURCE_PATH.read_text(
        encoding="utf-8"
    )


def test_q6_decode_operands_are_typed_at_emission_boundary() -> None:
    source = FWD_WRITER_SOURCE_PATH.read_text(encoding="utf-8")
    assert re.findall(r'"v\d+\.[lh]"', source) == []
    assert "_q6_narrow_decode_dwords" not in source
    assert "_q6_wide_decode_dwords" not in source
    assert "_q6_narrow_pack_signed_halfwords" not in source
    assert "_q6_wide_pack_signed_halfwords" not in source
    assert "_source_register_pairs" not in source

    tree = ast.parse(source)
    definitions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert {
        "Q6DelayDependency",
        "_lo",
        "_q6_hex",
        "_q6_salu_delay",
        "_q6_salu_pair_delay",
        "_q6_schedule",
        "_q6_sgpr",
        "_q6_valu_delay",
        "_q6_valu_pair_delay",
        "_q6_vgpr",
        "_total_sgprs",
        "_total_vgprs",
        "packed_payload_register",
        "refill_payload_register",
    }.isdisjoint(definitions)
    address_calls = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "Q6AddressAdd"
    ]
    raw_address_operands = [
        call.lineno
        for call in address_calls
        if (
            any(
                isinstance(operand, ast.Constant) and isinstance(operand.value, str)
                for operand in call.args[1:]
            )
            or any(
                keyword.arg == "high_left"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
                for keyword in call.keywords
            )
        )
    ]
    raw_delay_operands = [
        call.lineno
        for call in address_calls
        for keyword in call.keywords
        if keyword.arg in ("delay_after_low", "delay_after_high")
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, str)
    ]
    raw_delay_calls = [
        call.lineno
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "dependency_delay"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    ]
    assert raw_address_operands == []
    assert raw_delay_operands == []
    assert raw_delay_calls == []


@pytest.mark.parametrize("macro_tile0", (64, 128))
def test_q6_decode_plan_uses_semantic_atoms_and_pool_reuse(
    macro_tile0: int,
) -> None:
    layout = fwd_writer_module._q6_physical_layout(_q6_schedule(macro_tile0))
    plan = layout.decode
    assert len(plan.sources) == len(plan.outputs) == 16
    assert tuple(source.atom for source in plan.sources) == tuple(range(16))
    assert tuple(output.atom for output in plan.outputs) == tuple(range(16))
    assert tuple(output.low.first_register for output in plan.outputs) == tuple(
        source.low_payload.first_register for source in plan.sources
    )
    assert plan.outputs[0].high.first_register == plan.initial_scratch_register
    assert tuple(output.high.first_register for output in plan.outputs[1:]) == tuple(
        source.high_payload.first_register for source in plan.sources[:-1]
    )
    assert all(
        output.low.role.lifetime.first_stage == 2 * output.atom + 1
        and output.high.role.lifetime.first_stage == 2 * output.atom + 1
        for output in plan.outputs
    )
    assert len(plan.activation_payload_registers) == 18 * layout.output_tile_rows - 2

    decode = str(fwd_writer_module._q6_packed_decode(_q6_schedule(macro_tile0)))
    assert decode.count("v_add_nc_u32_e32") == 32
    assert decode.count("v_xor_b32_e32") == 32
    assert "0x60606060" in decode
    assert "0x80808080" in decode


@pytest.mark.parametrize("macro_tile0", (64, 128))
def test_q6_ownership_plan_is_shared_by_decode_and_refill(macro_tile0: int) -> None:
    layout = fwd_writer_module._q6_physical_layout(_q6_schedule(macro_tile0))
    ownership = layout.ownership
    decode = layout.decode
    assert tuple(
        (source.low_payload.first_register, source.high_payload.first_register)
        for source in decode.sources
    ) == tuple(
        (
            ownership.packed_payload(atom, "ql").first_register,
            ownership.packed_payload(atom, "qh").first_register,
        )
        for atom in range(16)
    )
    assert tuple(role.first_register for role in ownership.activation_payloads) == (
        decode.activation_payload_registers
    )
    assert len(ownership.refill_payloads) == 18 * layout.output_tile_rows
    assert all(
        role.role.lifetime.first_stage == 6 and role.role.lifetime.last_stage == 7
        for role in ownership.refill_payloads
    )


def test_q6_bf16_pipeline_is_semantic_and_knob_driven() -> None:
    selected = _q6_schedule(64)
    epilogue = str(fwd_writer_module._q6_bf16_epilogue(selected))
    assert epilogue.count("v_bfe_u32") == 32
    assert epilogue.count("v_or_b32_e32") == 32
    assert epilogue.count("v_cmp_u_f32_e32") == 32
    assert epilogue.count("v_add3_u32") == 32
    assert epilogue.count("v_cndmask_b32_e32") == 32
    assert epilogue.count("global_store_d16_hi_b16") == 32
    assert "v_bfe_u32 v47, v32, 16, 1" in epilogue
    assert "v_bfe_u32 v48, v46, 16, 1" in epilogue
    assert "v_bfe_u32 v47, v44, 16, 1" in epilogue

    width_one = str(
        fwd_writer_module._q6_bf16_epilogue(
            replace(selected, epilogue_dependency_width=1)
        )
    )
    width_four = str(
        fwd_writer_module._q6_bf16_epilogue(
            replace(selected, epilogue_dependency_width=4)
        )
    )
    full_tile = str(
        fwd_writer_module._q6_bf16_epilogue(
            replace(selected, epilogue_pipeline_scope="FullTile")
        )
    )
    assert selected.epilogue_pipeline_scope == "StoreBatch"
    assert width_one != epilogue != width_four
    assert full_tile != epilogue


def test_forward_writer_has_no_numbered_schedule_fragments() -> None:
    source = FWD_WRITER_SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    q6_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (node.name.startswith("_q6_") or node.name.startswith("_emit_q6_"))
    ]
    fragmented = [node.name for node in q6_functions if re.search(r"_\\d+$", node.name)]
    oversized = [
        (node.name, node.end_lineno - node.lineno + 1)
        for node in q6_functions
        if node.end_lineno is not None and node.end_lineno - node.lineno + 1 > 200
    ]
    emitter_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "emitter"
    ]
    concentrated = [
        (function.name, count)
        for function in q6_functions
        if (
            count := sum(
                call in emitter_calls
                for call in ast.walk(function)
                if isinstance(call, ast.Call)
            )
        )
        > 190
    ]
    assert fragmented == []
    assert oversized == []
    assert len(emitter_calls) <= 1_300
    assert concentrated == []
    assert "# fmt:" not in source
    assert "issue_slot" not in source


def test_writer_rejects_missing_decoded_lds_layout() -> None:
    solution = ForwardSolution.q5_k_decoded_weight_lds_retained()
    key = SolutionKey(
        ProblemType.mmq_forward("Q5_K"),
        ProblemSize(2048, 512, 2048),
        solution,
    )
    state = fwd_writer_module.DerivedForwardState.from_solution_key(key)
    writer = ForwardKernelWriterAssembly.__new__(ForwardKernelWriterAssembly)
    writer.solution_key = key
    writer.state = replace(state, decoded_lds=None)
    with pytest.raises(ForwardKernelWriterError, match="decoded LDS layout"):
        writer._body_decoded_weight_lds()
    with pytest.raises(ForwardKernelWriterError, match="decoded LDS layout"):
        writer._emit_decoded_weight_lds_stage(
            fwd_writer_module.Assembly(),
            row_stride=176,
            serial=238,
            wave=236,
            temporary=228,
            lds_address=229,
            auxiliary=230,
            metadata_address=232,
            staging_base=112,
        )
    with pytest.raises(ForwardKernelWriterError, match="decoded LDS layout"):
        writer._emit_decoded_group_loop(
            fwd_writer_module.Assembly(),
            group_base=0,
            weight_q=72,
            metadata=80,
            c_base=112,
            low_activation_last=176,
            high_activation_base=180,
            activation_scale_sum_base=212,
            scaled_dm_base=220,
            sum_base=8,
            lane=237,
            lds_address=229,
            weight_lds_base_address=234,
            metadata_lds_base_address=232,
        )


def test_writer_rejects_incomplete_decoded_epilogue_pipeline() -> None:
    solution = ForwardSolution.q5_k_decoded_weight_lds_retained()
    key = SolutionKey(
        ProblemType.mmq_forward("Q5_K"),
        ProblemSize(2048, 512, 2048),
        solution,
    )
    writer = ForwardKernelWriterAssembly.__new__(ForwardKernelWriterAssembly)
    writer.solution_key = key
    state = fwd_writer_module.DerivedForwardState.from_solution_key(key)
    writer.state = replace(
        state,
        kernel_spec=replace(
            state.kernel_spec,
            epilogue=replace(state.kernel_spec.epilogue, pipeline=None),
        ),
    )
    writer.toolchain = Toolchain.discover()
    with pytest.raises(
        ForwardKernelWriterError,
        match="complete epilogue pipeline",
    ):
        writer._emit_decoded_bf16_store(
            fwd_writer_module.Assembly(),
            size_n=512,
            sum_base=8,
            temporary=228,
            auxiliary=230,
            metadata_address=232,
            wave_column_base=235,
            lane=237,
            serial=238,
        )


def test_writer_emits_single_dependency_scheduled_epilogue() -> None:
    solution = ForwardSolution.q5_k_decoded_weight_lds_extraction(
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


def test_small_m_q8_helpers_reject_invalid_fragment_counts() -> None:
    with pytest.raises(ValueError, match="two or four"):
        Q8SmallMTiledLdsRegisterPlan.allocate(8)
    writer = ForwardKernelWriterAssembly(
        SolutionKey(
            ProblemType.mmq_forward("Q8_0"),
            ProblemSize(32, 129280, 4096),
            ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=32),
        ),
        Toolchain.discover(),
    )
    with pytest.raises(ValueError, match="2, 4, or 8"):
        writer._emit_q8_hip_group(
            fwd_writer_module.Assembly(),
            0,
            0,
            8,
            16,
            24,
            32,
            40,
            44,
            52,
            53,
            54,
            m_fragments=3,
            zero_accumulator=60,
        )
    with pytest.raises(ValueError, match="2, 4, or 8"):
        writer._emit_q8_hip_store(
            fwd_writer_module.Assembly(),
            8,
            70,
            78,
            79,
            80,
            81,
            ProblemSize(32, 129280, 4096),
            m_fragments=3,
        )


def test_writer_methods_have_complete_line_coverage() -> None:
    assert_writer_methods_have_complete_line_coverage(FWD_WRITER_SOURCE_PATH)
