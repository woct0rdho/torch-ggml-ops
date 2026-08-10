"""Targeted branch and complete executable-line coverage for the forward writer."""

import ast
import hashlib
import re
from dataclasses import fields, replace
from typing import Any, cast

import pytest

from tests.ggtensile.support import (
    FWD_LOWERING_SOURCE_PATHS,
    FWD_PHYSICAL_SOURCE_PATH,
    FWD_WRITER_SOURCE_PATH,
    assert_writer_methods_have_complete_line_coverage,
)
from tools.ggtensile import kernel_writer_assembly_mmq_fwd as fwd_writer_module
from tools.ggtensile import mmq_fwd_lowering_q6 as q6_lowering_module
from tools.ggtensile import mmq_fwd_physical as q6_physical_module
from tools.ggtensile.kernel_writer_assembly import Assembly, RegisterLifetime
from tools.ggtensile.kernel_writer_assembly_mmq_fwd import (
    ForwardKernelWriterAssembly,
    ForwardKernelWriterError,
)
from tools.ggtensile.mmq_fwd_lowering_decoded_lds import DecodedWeightLdsLowering
from tools.ggtensile.mmq_fwd_lowering_packed_direct import (
    PackedScaleMinimumDirectLowering,
)
from tools.ggtensile.mmq_fwd_lowering_q3_full import FullWeightQ3TiledLdsLowering
from tools.ggtensile.mmq_fwd_lowering_q6 import (
    Q6ScheduleEmitter,
    _emit_q6_dot_phase,
    _emit_q6_scheduled_body,
)
from tools.ggtensile.mmq_fwd_lowering_signed_i8 import SignedInt8ForwardLowering
from tools.ggtensile.mmq_fwd_physical import (
    DecodedWeightLdsPhysicalPlan,
    Q3FullWeightTiledLdsRegisterPlan,
    Q6AddressAdd,
    Q6DependencyDelay,
    Q6Immediate,
    Q6PhysicalLayout,
    Q6Vgpr,
    SignedInt8MmaGroupRole,
    SignedInt8RegisterTiledRegisterPlan,
    SignedInt8SmallMTiledLdsPhysicalPlan,
    SignedInt8SmallMTiledLdsRegisterPlan,
    SignedInt8TiledLdsPolicy,
    SignedInt8TiledLdsRegisters,
    SignedInt8TiledLdsScaleLayout,
    SignedInt8WaveNTiledLdsLayout,
    SignedInt8WaveNTiledLdsPhysicalPlan,
    derive_forward_physical_plan,
    q6_structured_physical_plan,
)
from tools.ggtensile.mmq_fwd_search import q6_schedule_candidates
from tools.ggtensile.mmq_fwd_spec import (
    ForwardKernelSpec,
    Q6ForwardSchedule,
    Q6LdsLayout,
    QuantForwardSemantics,
    q6_schedule_from_solution,
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

Q6_LOWERING_SOURCE_PATH = next(
    path for path in FWD_LOWERING_SOURCE_PATHS if path.name == "mmq_fwd_lowering_q6.py"
)


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
    output_tile_rows = macro_tile0 // 64
    physical = q6_structured_physical_plan(output_tile_rows)
    layout = q6_lowering_module._q6_physical_layout(_q6_schedule(macro_tile0))
    assert physical.layout.output_tile_rows == output_tile_rows
    assert physical.layout.lds == layout.lds
    assert physical.resources.vgprs == layout.declared_vgprs
    assert physical.resources.sgprs == layout.declared_sgprs
    assert physical.resources.lds_bytes == layout.lds.total_bytes
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
    assert register_map.lifetime("weight_address") == RegisterLifetime(0, 8)
    assert register_map.lifetime("product") == RegisterLifetime(5, 7)
    assert register_map.lifetime("first_stage_write_address") == RegisterLifetime(1, 4)
    assert register_map.lifetime("first_stage_read_address") == RegisterLifetime(4, 5)
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
        "physical_plan",
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
    assert len(j64_candidates) == 48
    assert len(j128_candidates) == 64
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


def test_q6_shared_mt256_and_wide_scalar_carry_plans_are_emitted() -> None:
    mt256 = _q6_schedule(256)
    shared_source = str(_emit_q6_scheduled_body(mt256, 8))
    assert "s_and_b32 s25, s25, 0x1000" in shared_source
    assert "s_cbranch_scc0 .LQ6SharedDecodeDone" in shared_source
    shared_wavefront = replace(
        mt256,
        semantic_policy=mt256.semantic_policy.structured_q6_wavefront(),
    )
    assert ".LQ6SharedDecodeDone:" in str(_emit_q6_scheduled_body(shared_wavefront, 8))

    wide = replace(
        _q6_schedule(64),
        semantic_policy=mt256.semantic_policy.structured_q6_wavefront(),
        physical_plan="WideScalarCarryFrontier",
    )
    wide_source = str(_emit_q6_scheduled_body(wide, 8))
    assert "v_add_co_u32 v4, s25, v2, v53" in wide_source
    assert "v_add_co_ci_u32_e64 v5, null, 0, v3, s25" in wide_source


def test_q6_scalar_carry_frontier_rejects_an_oversized_address_batch() -> None:
    schedule = replace(
        _q6_schedule(64),
        semantic_policy=_q6_schedule(256).semantic_policy.structured_q6_wavefront(),
        physical_plan="WideScalarCarryFrontier",
    )
    emitter = Q6ScheduleEmitter(schedule, "oversized")
    addresses = tuple(
        Q6AddressAdd(index * 2, Q6Immediate(index), Q6Vgpr(0), Q6Vgpr(1))
        for index in range(9)
    )
    with pytest.raises(ValueError, match="at most eight addresses"):
        emitter.add_u64_batch(addresses)
    delayed = (
        Q6AddressAdd(
            0,
            Q6Immediate(1),
            Q6Vgpr(0),
            Q6Vgpr(1),
            delay_after_low=Q6DependencyDelay("VALU_DEP", 1),
        ),
        Q6AddressAdd(
            2,
            Q6Immediate(2),
            Q6Vgpr(0),
            Q6Vgpr(1),
            delay_after_high=Q6DependencyDelay("VALU_DEP", 1),
        ),
    )
    explicit = replace(schedule, dependency_delay_mode="Explicit")
    delayed_emitter = Q6ScheduleEmitter(explicit, "delayed")
    delayed_emitter.add_u64_batch(delayed)
    assert str(delayed_emitter.module()).count("s_delay_alu") == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("latency", "OpaqueLatencyTable", "latency-lowering"),
        ("pairing", "OpaqueIssueTable", "dual-issue pairing"),
        ("wait", "OpaqueWaitPolicy", "wait-lowering"),
        ("traversal", "PhysicalRegisterOrder", "output traversal"),
        ("clustering", "OpaqueClustering", "semantic-stage clustering"),
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
        if field in {"latency", "pairing", "wait"}:
            q6_lowering_module.Q6ScheduleEmitter(schedule, "invalid")
        else:
            q6_lowering_module._q6_physical_layout(schedule)


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
    shared_m256 = _q6_schedule(256)
    assert shared_m256.work_group == (32, 8, 1)
    assert shared_m256.macro_tile == (256, 64)
    assert shared_m256.resource_usage.lds_bytes == 57_344
    with pytest.raises(ValueError, match="implements MT64, MT128, or MT256"):
        _q6_schedule(192)
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
        q6_lowering_module._q6_physical_layout(unsupported)
    invalid_wave_group = replace(selected, mi_wave_group=(2, 2))
    with pytest.raises(ValueError, match="unsupported Q6 M-wave ownership"):
        q6_lowering_module._q6_physical_layout(invalid_wave_group)
    with pytest.raises(ValueError, match="unsupported Q6 physical output rows"):
        Q6PhysicalLayout(3)
    with pytest.raises(ValueError, match="unsupported Q6 M-wave groups"):
        Q6PhysicalLayout(2, 3)
    with pytest.raises(ValueError, match="requires two rows per wave"):
        Q6PhysicalLayout(1, 2)
    with pytest.raises(ValueError, match="four or eight M-owned waves"):
        q6_structured_physical_plan(1, 2)
    with pytest.raises(ValueError, match="unsupported structured Q6 physical plan"):
        q6_structured_physical_plan(1, 4, "OpaquePlan")
    with pytest.raises(ValueError, match="requires four-wave MT64"):
        q6_structured_physical_plan(2, 4, "WideScalarCarryFrontier")
    with pytest.raises(ValueError, match="unsupported Q6 ownership rows"):
        q6_physical_module.Q6OwnershipRegisterPlan.for_output_tile_rows(3)
    ownership = q6_physical_module.Q6OwnershipRegisterPlan.for_output_tile_rows(1)
    with pytest.raises(ValueError, match="unsupported Q6 payload atom"):
        ownership.packed_payload(16, "ql")
    with pytest.raises(ValueError, match="unsupported Q6 physical output rows"):
        q6_physical_module.Q6PhysicalRegisterMap.for_output_tile_rows(3)
    layout = q6_physical_module.Q6PhysicalLayout(1)
    with pytest.raises(KeyError, match="missing"):
        layout.registers.assignment("missing")
    with pytest.raises(ValueError, match="has 0 output roles"):
        layout.registers.group_output_roles(2)
    with pytest.raises(ValueError, match="unsupported Q6 refill slot"):
        layout.refill_read(-1, 0)
    with pytest.raises(ValueError, match="unsupported Q6 decoded write atom"):
        layout.decoded_write_address(16)
    with pytest.raises(ValueError, match="unsupported Q6 decoded write atom"):
        layout.decoded_write_offsets(16)
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
    emitter = q6_lowering_module.Q6ScheduleEmitter(selected, "invalid")
    assert str(q6_physical_module.Q6Vgpr(2)) == "v2"
    assert str(q6_physical_module.Q6Sgpr(3)) == "s3"
    assert str(q6_physical_module.Q6Immediate(16)) == "16"
    assert str(q6_physical_module.Q6Immediate(16, hexadecimal=True)) == "0x10"
    valu = q6_physical_module.Q6DependencyDelay("VALU_DEP", 4)
    assert str(valu) == "instid0(VALU_DEP_4)"
    assert str(q6_physical_module.Q6DependencyDelay("VALU_DEP", 4, "NEXT", 1)) == (
        "instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)"
    )
    with pytest.raises(ValueError, match="VGPR must be nonnegative"):
        q6_physical_module.Q6Vgpr(-1)
    with pytest.raises(ValueError, match="SGPR must be nonnegative"):
        q6_physical_module.Q6Sgpr(-1)
    with pytest.raises(ValueError, match="unsupported Q6 dependency kind"):
        q6_physical_module.Q6DependencyDelay(cast(Any, "OPAQUE"), 1)
    with pytest.raises(ValueError, match="distance must be positive"):
        q6_physical_module.Q6DependencyDelay("VALU_DEP", 0)
    with pytest.raises(ValueError, match="distance must be positive"):
        q6_physical_module.Q6DependencyDelay("VALU_DEP", 1, "NEXT", 0)
    with pytest.raises(ValueError, match="unsupported Q6 dependency skip"):
        q6_physical_module.Q6DependencyDelay("VALU_DEP", 1, cast(Any, "SKIP_ALL"), 1)
    with pytest.raises(ValueError, match="requires skip and second"):
        q6_physical_module.Q6DependencyDelay("VALU_DEP", 1, "NEXT")
    with pytest.raises(ValueError, match="requires skip and second"):
        q6_physical_module.Q6DependencyDelay("VALU_DEP", 1, second_distance=1)
    with pytest.raises(ValueError, match="VGPR must be nonnegative"):
        q6_physical_module.Q6HalfRegister(-1, "l")
    with pytest.raises(ValueError, match="selector must be 'l' or 'h'"):
        q6_physical_module.Q6HalfRegister(1, cast(Any, "x"))
    with pytest.raises(ValueError, match="unsupported Q6 global-read width: 64"):
        q6_physical_module.Q6GlobalRead(1, 2, width_bits=64)
    with pytest.raises(ValueError, match="destination must be one VGPR"):
        q6_physical_module.Q6GlobalRead(-1, 2)
    with pytest.raises(ValueError, match="destination must be one VGPR"):
        emitter.global_read_clause(
            (q6_physical_module.Q6GlobalRead(cast(int, "v[1:2]"), 2),)
        )
    with pytest.raises(ValueError, match="address must be one VGPR pair"):
        q6_physical_module.Q6GlobalRead(1, cast(int, "v[2:3]"))
    with pytest.raises(ValueError, match="does not match StoreVectorWidth"):
        emitter.store_bf16_clause("v[0:1]", (1, 2))


def test_q6_semantic_emitter_derives_refill_pair_order_from_slots() -> None:
    selected = _q6_schedule(64)
    emitter = q6_lowering_module.Q6ScheduleEmitter(selected, "refill")
    first = q6_physical_module.Q6GlobalRead(8, 2, local_write_slot=0)
    second = q6_physical_module.Q6GlobalRead(9, 2, local_write_slot=1)
    assert first.payload_assignment.first_register == 8
    assert first.payload_assignment.role.lifetime == RegisterLifetime(6, 7)
    assert first.address_assignment.first_register == 2
    assert first.address_assignment.role.width == 2
    assert first.address_assignment.role.lifetime == RegisterLifetime(5, 6)
    emitter.global_read_clause((first, second))
    emitter.commit_cooperative_global_reads(Q6LdsLayout(1), 52)
    emitted = str(emitter.module())
    assert "ds_store_2addr_stride64_b32 v52, v8, v9 offset0:1 offset1:3" in emitted
    with pytest.raises(ValueError, match="nonnegative"):
        q6_physical_module.Q6GlobalRead(1, 2, local_write_slot=-1)

    duplicate = q6_lowering_module.Q6ScheduleEmitter(selected, "duplicate")
    with pytest.raises(ValueError, match="duplicate Q6 local-write slot"):
        duplicate.global_read_clause(
            (
                q6_physical_module.Q6GlobalRead(8, 2, local_write_slot=0),
                q6_physical_module.Q6GlobalRead(9, 2, local_write_slot=0),
            )
        )

    incomplete = q6_lowering_module.Q6ScheduleEmitter(selected, "incomplete")
    incomplete.global_read_clause(
        (q6_physical_module.Q6GlobalRead(8, 2, local_write_slot=1),)
    )
    with pytest.raises(ValueError, match="contiguous pairs"):
        incomplete.commit_cooperative_global_reads(Q6LdsLayout(1), 52)


def test_q6_semantic_emitter_derives_vmem_waits_from_producers() -> None:
    emitter = q6_lowering_module.Q6ScheduleEmitter(_q6_schedule(64), "waits")
    reads = tuple(
        q6_physical_module.Q6GlobalRead(register, 2) for register in (8, 9, 10)
    )
    assert reads[0].payload_assignment.role.lifetime == (RegisterLifetime(1, 4))
    assert reads[0].address_assignment.role.lifetime == (RegisterLifetime(0, 1))
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
    assert "_q6_ordered_module" not in Q6_LOWERING_SOURCE_PATH.read_text(
        encoding="utf-8"
    )


def test_q6_decode_operands_are_typed_at_emission_boundary() -> None:
    source = Q6_LOWERING_SOURCE_PATH.read_text(encoding="utf-8")
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
    layout = q6_lowering_module._q6_physical_layout(_q6_schedule(macro_tile0))
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

    decode = str(q6_lowering_module._q6_packed_decode(_q6_schedule(macro_tile0)))
    assert decode.count("v_add_nc_u32_e32") == 32
    assert decode.count("v_xor_b32_e32") == 32
    assert "0x60606060" in decode
    assert "0x80808080" in decode


@pytest.mark.parametrize("macro_tile0", (64, 128))
def test_q6_wavefront_decode_preserves_register_handoffs(
    macro_tile0: int,
) -> None:
    schedule = replace(
        _q6_schedule(macro_tile0),
        semantic_policy=replace(
            _q6_schedule(macro_tile0).semantic_policy,
            traversal="OutputRoleWavefront",
            clustering="RowBatchedDecodeOrder",
            latency="WavefrontDependencyDistance",
        ),
    )
    decode = str(q6_lowering_module._q6_packed_decode(schedule))
    lines = [line.strip() for line in decode.splitlines()]
    qh_shift_positions = [
        index
        for index, line in enumerate(lines)
        if line.startswith("v_lshrrev_b32_e32") and ("v42" in line or "v69" in line)
    ]
    assert len(qh_shift_positions) == 16
    first_high_extract = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("v_lshrrev_b32_e32")
        and ("v123, 4, v10" in line or "v173, 4, v112" in line)
    )
    assert max(qh_shift_positions) < first_high_extract
    assert decode.count("v_add_nc_u32_e32") == 32
    assert decode.count("v_xor_b32_e32") == 32


@pytest.mark.parametrize("macro_tile0", (64, 128))
def test_q6_ownership_plan_is_shared_by_decode_and_refill(macro_tile0: int) -> None:
    layout = q6_lowering_module._q6_physical_layout(_q6_schedule(macro_tile0))
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
    epilogue = str(q6_lowering_module._q6_bf16_epilogue(selected))
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
        q6_lowering_module._q6_bf16_epilogue(
            replace(selected, epilogue_dependency_width=1)
        )
    )
    width_four = str(
        q6_lowering_module._q6_bf16_epilogue(
            replace(selected, epilogue_dependency_width=4)
        )
    )
    full_tile = str(
        q6_lowering_module._q6_bf16_epilogue(
            replace(selected, epilogue_pipeline_scope="FullTile")
        )
    )
    assert selected.epilogue_pipeline_scope == "StoreBatch"
    assert width_one != epilogue != width_four
    assert full_tile != epilogue


def test_forward_writer_has_no_numbered_schedule_fragments() -> None:
    source = Q6_LOWERING_SOURCE_PATH.read_text(encoding="utf-8")
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


def test_decoded_lowering_uses_plan_owned_lds_layout() -> None:
    solution = ForwardSolution.q5_k_decoded_weight_lds_retained()
    key = SolutionKey(
        ProblemType.mmq_forward("Q5_K"),
        ProblemSize(2048, 512, 2048),
        solution,
    )
    state = fwd_writer_module.DerivedForwardState.from_solution_key(key)
    assert isinstance(state.physical_plan, DecodedWeightLdsPhysicalPlan)
    physical = state.physical_plan
    assert physical.layout.weight_row_stride == 304
    assert physical.layout.total_bytes == 38_400
    assert physical.resources.lds_bytes == physical.layout.total_bytes
    assert physical.resources.vgprs == physical.registers.declared_vgprs == 239


def test_writer_rejects_incomplete_decoded_epilogue_pipeline() -> None:
    solution = ForwardSolution.q5_k_decoded_weight_lds_retained()
    key = SolutionKey(
        ProblemType.mmq_forward("Q5_K"),
        ProblemSize(2048, 512, 2048),
        solution,
    )
    writer = ForwardKernelWriterAssembly(key, Toolchain.discover())
    state = fwd_writer_module.DerivedForwardState.from_solution_key(key)
    state = replace(
        state,
        kernel_spec=replace(
            state.kernel_spec,
            epilogue=replace(state.kernel_spec.epilogue, pipeline=None),
        ),
    )
    lowering = DecodedWeightLdsLowering(replace(writer.context, state=state))
    with pytest.raises(
        ForwardKernelWriterError,
        match="complete epilogue pipeline",
    ):
        lowering._emit_bf16_epilogue(Assembly())


def test_mechanism_lowerers_and_facade_reject_unknown_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = SolutionKey(
        ProblemType.mmq_forward("Q4_K"),
        ProblemSize(2048, 512, 2048),
        ForwardSolution.q4_k_pilot(),
    )
    writer = ForwardKernelWriterAssembly(key, Toolchain.discover())
    invalid_state = replace(
        writer.state,
        kernel_spec=replace(
            writer.state.kernel_spec,
            global_memory=replace(
                writer.state.kernel_spec.global_memory,
                operand_source="Unknown",
            ),
        ),
    )
    invalid_context = replace(writer.context, state=invalid_state)
    with pytest.raises(TypeError, match="unsupported direct packed operand source"):
        PackedScaleMinimumDirectLowering(invalid_context).body()
    with pytest.raises(
        TypeError,
        match="unsupported decoded-weight LDS operand source",
    ):
        DecodedWeightLdsLowering(invalid_context).body()
    writer.state = invalid_state
    writer.context = replace(writer.context, state=invalid_state)
    with pytest.raises(TypeError, match="unsupported forward operand source"):
        writer._body()
    mechanism = fwd_writer_module.forward_mechanism_contract("Global")
    monkeypatch.setattr(
        fwd_writer_module,
        "forward_mechanism_contract",
        lambda _: replace(
            mechanism,
            lowering="Opaque",
        ),
    )
    with pytest.raises(TypeError, match="unsupported forward operand source"):
        writer._body()


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
        SignedInt8SmallMTiledLdsRegisterPlan.allocate(8)
    writer = ForwardKernelWriterAssembly(
        SolutionKey(
            ProblemType.mmq_forward("Q8_0"),
            ProblemSize(32, 129280, 4096),
            ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=32),
        ),
        Toolchain.discover(),
    )
    lowering = SignedInt8ForwardLowering(writer.context)
    physical = cast(
        SignedInt8SmallMTiledLdsPhysicalPlan,
        writer.context.state.physical_plan,
    )
    tiled_registers: SignedInt8TiledLdsRegisters = physical.registers
    tiled_scale_layout: SignedInt8TiledLdsScaleLayout = physical.layout
    with pytest.raises(ValueError, match="2, 4, or 8"):
        lowering._emit_signed_int8_tiled_group(
            Assembly(),
            0,
            tiled_registers,
            tiled_scale_layout,
            physical.policy,
            m_fragments=3,
        )
    with pytest.raises(ValueError, match="2, 4, or 8"):
        lowering._emit_signed_int8_tiled_store(
            Assembly(),
            tiled_registers,
            ProblemSize(32, 129280, 4096),
            m_fragments=3,
        )

    q3_writer = ForwardKernelWriterAssembly(
        SolutionKey(
            ProblemType.mmq_forward("Q3_K"),
            ProblemSize(2048, 4096, 2048),
            ForwardSolution.q3_k_hip_tiled_lds(),
        ),
        Toolchain.discover(),
    )
    with pytest.raises(TypeError, match="unsupported Q8 physical plan"):
        SignedInt8ForwardLowering(q3_writer.context).body()


def test_signed_int8_tiled_policies_reject_inconsistent_internal_state() -> None:
    small_writer = ForwardKernelWriterAssembly(
        SolutionKey(
            ProblemType.mmq_forward("Q8_0"),
            ProblemSize(32, 129280, 4096),
            ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=32),
        ),
        Toolchain.discover(),
    )
    small = cast(
        SignedInt8SmallMTiledLdsPhysicalPlan,
        small_writer.context.state.physical_plan,
    )

    def small_lowering(policy: SignedInt8TiledLdsPolicy) -> SignedInt8ForwardLowering:
        state = replace(
            small_writer.context.state,
            physical_plan=replace(small, policy=policy),
        )
        return SignedInt8ForwardLowering(replace(small_writer.context, state=state))

    with pytest.raises(ValueError, match="requires weight-first staging"):
        small_lowering(
            SignedInt8TiledLdsPolicy(
                "Opaque",  # ty: ignore[invalid-argument-type]
                "Scalar",
            )
        ).body()
    with pytest.raises(ValueError, match="require a second-base delta"):
        small_lowering(
            SignedInt8TiledLdsPolicy(
                "WeightThenActivation",
                "PairedHoistedSecondBase",
            )
        ).body()

    wave_writer = ForwardKernelWriterAssembly(
        SolutionKey(
            ProblemType.mmq_forward("Q8_0"),
            ProblemSize(2048, 1024, 4096),
            ForwardSolution.q8_0_hip_tiled_lds(),
        ),
        Toolchain.discover(),
    )
    wave = cast(
        SignedInt8WaveNTiledLdsPhysicalPlan,
        wave_writer.context.state.physical_plan,
    )

    def wave_lowering(policy: SignedInt8TiledLdsPolicy) -> SignedInt8ForwardLowering:
        state = replace(
            wave_writer.context.state,
            physical_plan=replace(wave, policy=policy),
        )
        return SignedInt8ForwardLowering(replace(wave_writer.context, state=state))

    with pytest.raises(ValueError, match="unsupported Q8 stage order"):
        wave_lowering(
            SignedInt8TiledLdsPolicy(
                "Opaque",  # ty: ignore[invalid-argument-type]
                "Scalar",
            )
        ).body()
    with pytest.raises(ValueError, match="require a second-base delta"):
        wave_lowering(
            SignedInt8TiledLdsPolicy(
                "Interleaved",
                "PairedHoistedSecondBase",
            )
        ).body()

    lowering = SignedInt8ForwardLowering(wave_writer.context)
    registers: SignedInt8TiledLdsRegisters = wave.registers
    layout: SignedInt8TiledLdsScaleLayout = wave.layout
    with pytest.raises(ValueError, match="require a second-base delta"):
        lowering._emit_signed_int8_tiled_group(
            Assembly(),
            0,
            registers,
            layout,
            SignedInt8TiledLdsPolicy(
                "Interleaved",
                "PairedHoistedSecondBase",
            ),
        )
    with pytest.raises(ValueError, match="unsupported Q8 scale-read policy"):
        lowering._emit_signed_int8_tiled_group(
            Assembly(),
            0,
            registers,
            layout,
            SignedInt8TiledLdsPolicy(
                "Interleaved",
                "Opaque",  # ty: ignore[invalid-argument-type]
            ),
        )


def test_forward_physical_planner_rejects_unknown_descriptor_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ForwardKernelSpec.from_solution(ForwardSolution.q8_0_direct_global())
    mechanism = q6_physical_module.forward_mechanism_contract("Q8DirectGlobal")
    monkeypatch.setattr(
        q6_physical_module,
        "forward_mechanism_contract",
        lambda _: replace(
            mechanism,
            physical_plan="Opaque",
        ),
    )
    with pytest.raises(AssertionError, match="unhandled forward physical plan"):
        derive_forward_physical_plan(spec)


def test_writer_emits_typed_q3_full_weight_control() -> None:
    solution = ForwardSolution.q3_k_full_weight_tiled_lds()
    key = SolutionKey(
        ProblemType.mmq_forward("Q3_K"),
        ProblemSize(32768, 4096, 2048),
        solution,
    )
    assert validate_solution(key) == ()
    writer = ForwardKernelWriterAssembly(key, Toolchain.discover())
    source = writer.source()
    assert source.count("v_wmma_i32_16x16x16_iu8") == 128
    assert source.count("s_barrier") == 4
    assert "s_mul_i32 s11, 18432, s3" in source
    assert "v_mul_lo_u32 v179, 336, v176" in source
    with pytest.raises(ValueError, match="starts at half zero"):
        FullWeightQ3TiledLdsLowering(writer.context)._emit_weight_prefetch(
            Assembly(), half=1
        )


def test_forward_physical_plans_reject_invalid_domains() -> None:
    physical_source = FWD_PHYSICAL_SOURCE_PATH.read_text(encoding="utf-8")
    physical_tree = ast.parse(physical_source)
    imported_modules = {
        node.module
        for node in ast.walk(physical_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        module == "rocisa" or module.endswith(("toolchain", "inspection"))
        for module in imported_modules
    )
    assert "Assembly(" not in physical_source

    with pytest.raises(ValueError, match="index 0..3"):
        SignedInt8MmaGroupRole.from_semantics(
            QuantForwardSemantics.for_quant_type("Q8_0"), 4
        )
    invalid_signed_int8 = replace(
        QuantForwardSemantics.for_quant_type("Q8_0"),
        weight_bits=7,
    )
    with pytest.raises(ValueError, match="signed-int8 direct group"):
        SignedInt8MmaGroupRole.from_semantics(invalid_signed_int8, 0)
    with pytest.raises(ValueError, match="four 16x16 fragments"):
        SignedInt8RegisterTiledRegisterPlan.allocate(1, 1)
    with pytest.raises(ValueError, match="fixed HIP-shaped dimensions"):
        SignedInt8WaveNTiledLdsLayout(allocation_padding_bytes=0)
    with pytest.raises(ValueError, match="one or two output rows"):
        q6_structured_physical_plan(3)

    q3_registers = Q3FullWeightTiledLdsRegisterPlan.allocate()
    with pytest.raises(ValueError, match="count is inconsistent"):
        replace(q3_registers, register_count=199)
    with pytest.raises(ValueError, match="exceeds the plan"):
        replace(
            q3_registers,
            wave=replace(q3_registers.wave, first_register=200),
        )

    q3 = ForwardKernelSpec.from_solution(ForwardSolution.q3_k_hip_tiled_lds())
    with pytest.raises(ValueError, match="Q3 HIP-shaped"):
        derive_forward_physical_plan(
            replace(q3, geometry=replace(q3.geometry, work_group=(32, 2, 1)))
        )

    q3_full = ForwardKernelSpec.from_solution(
        ForwardSolution.q3_k_full_weight_tiled_lds()
    )
    with pytest.raises(ValueError, match="Q3 full-weight"):
        derive_forward_physical_plan(
            replace(q3_full, geometry=replace(q3_full.geometry, work_group=(32, 2, 1)))
        )

    q8_hip = ForwardKernelSpec.from_solution(ForwardSolution.q8_0_hip_tiled_lds())
    with pytest.raises(ValueError, match="DepthU 32 or 64"):
        derive_forward_physical_plan(
            replace(q8_hip, geometry=replace(q8_hip.geometry, depth_u=48))
        )

    q8_small = ForwardKernelSpec.from_solution(
        ForwardSolution.q8_0_small_m_tiled_lds(macro_tile0=32)
    )
    with pytest.raises(ValueError, match="MT32/MT64"):
        derive_forward_physical_plan(
            replace(
                q8_small,
                geometry=replace(q8_small.geometry, work_group=(32, 2, 1)),
            )
        )
    with pytest.raises(ValueError, match="unsupported layout"):
        derive_forward_physical_plan(
            replace(q8_small, lds=replace(q8_small.lds, address_hoist="Unknown"))
        )
    with pytest.raises(ValueError, match="unsupported forward operand source"):
        derive_forward_physical_plan(
            replace(
                q8_small,
                global_memory=replace(q8_small.global_memory, operand_source="Unknown"),
            )
        )


def test_writer_methods_have_complete_line_coverage() -> None:
    for source_path in FWD_LOWERING_SOURCE_PATHS:
        assert_writer_methods_have_complete_line_coverage(source_path)
    assert_writer_methods_have_complete_line_coverage(FWD_PHYSICAL_SOURCE_PATH)
    assert_writer_methods_have_complete_line_coverage(FWD_WRITER_SOURCE_PATH)
