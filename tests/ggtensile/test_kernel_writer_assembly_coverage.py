"""Targeted branch and complete executable-line coverage for shared assembly support."""

import hashlib
from pathlib import Path

import pytest

from tests.ggtensile.support import (
    SHARED_WRITER_SOURCE_PATH,
    assert_writer_methods_have_complete_line_coverage,
)
from tools.ggtensile.kernel_abi import (
    ORDINARY_FORWARD_ABI,
    KernelAbi,
    KernelArgument,
    KernelArgumentKind,
    KernelValueType,
)
from tools.ggtensile.kernel_writer_assembly import (
    Assembly,
    DeterministicRegisterPlan,
    DeterministicRegisterPool,
    RegisterAssignment,
    RegisterLifetime,
    RegisterRole,
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
    emit_pointer_kernarg_loads(assembly, 4, ORDINARY_FORWARD_ABI)
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

    short_abi = KernelAbi(
        (
            KernelArgument(
                "only_pointer",
                KernelArgumentKind.GlobalBuffer,
                KernelValueType.Struct,
            ),
        )
    )
    with pytest.raises(AssertionError):
        emit_pointer_kernarg_loads(Assembly(), 4, short_abi)


def test_deterministic_register_plan_uses_explicit_order_and_lifetimes() -> None:
    active = RegisterLifetime(0, 2)
    retired = RegisterLifetime(3, 4)
    roles = {
        "accumulator": RegisterRole("accumulator", 2, active),
        "operand": RegisterRole("operand", 2, active),
        "epilogue": RegisterRole("epilogue", 2, retired),
        "aligned": RegisterRole("aligned", 2, active, alignment=4, minimum_register=1),
    }
    plan = DeterministicRegisterPlan.allocate(
        roles,
        ("accumulator", "operand", "epilogue", "aligned"),
        max_registers=8,
    )
    assert tuple(plan.assignment("accumulator").registers) == (0, 1)
    assert tuple(plan.assignment("operand").registers) == (2, 3)
    assert tuple(plan.assignment("epilogue").registers) == (0, 1)
    assert tuple(plan.assignment("aligned").registers) == (4, 5)
    assert plan.register_count == 6
    with pytest.raises(KeyError):
        plan.assignment("missing")


def test_deterministic_register_plan_rejects_incomplete_or_impossible_plans() -> None:
    with pytest.raises(AssertionError):
        RegisterLifetime(2, 1)
    with pytest.raises(AssertionError):
        RegisterRole("", 0, RegisterLifetime(0, 0))
    with pytest.raises(AssertionError):
        RegisterRole("bad", 1, RegisterLifetime(0, 0), alignment=0)
    role = RegisterRole("value", 2, RegisterLifetime(0, 1))
    with pytest.raises(AssertionError):
        DeterministicRegisterPlan.allocate({}, (), max_registers=0)
    with pytest.raises(AssertionError):
        DeterministicRegisterPlan.allocate({"value": role}, (), max_registers=2)
    with pytest.raises(AssertionError):
        DeterministicRegisterPlan.allocate({"other": role}, ("other",), max_registers=2)
    reserved_role = RegisterRole("reserved", 2, RegisterLifetime(0, 1))
    reserved = RegisterAssignment(reserved_role, 0)
    with pytest.raises(AssertionError):
        DeterministicRegisterPlan.allocate(
            {"value": role},
            ("value",),
            max_registers=2,
            reserved=(reserved,),
        )


def test_deterministic_register_pool_uses_explicit_checkout_and_reuse() -> None:
    pool = DeterministicRegisterPool((3, 1, 2))
    lifetime = RegisterLifetime(0, 1)
    preferred = pool.checkout(
        RegisterRole("preferred", 1, lifetime),
        preferred_register=2,
    )
    first_fit = pool.checkout(RegisterRole("first_fit", 1, lifetime))
    assert preferred.first_register == 2
    assert first_fit.first_register == 1
    assert pool.assignment("preferred") == preferred
    assert tuple(item.role.name for item in pool.checked_out) == (
        "first_fit",
        "preferred",
    )
    assert pool.checkin("first_fit") == first_fit
    reused = pool.checkout(RegisterRole("reused", 1, lifetime))
    assert reused.first_register == 1

    with pytest.raises(AssertionError):
        pool.checkout(RegisterRole("reused", 1, lifetime))
    with pytest.raises(AssertionError):
        pool.checkout(
            RegisterRole("too_wide", 2, lifetime),
            preferred_register=3,
        )
    with pytest.raises(AssertionError):
        pool.checkout(RegisterRole("too_high", 1, lifetime, minimum_register=4))
    with pytest.raises(AssertionError):
        pool.checkin("missing")
    with pytest.raises(KeyError):
        pool.assignment("missing")


def test_deterministic_register_pool_rejects_invalid_state() -> None:
    with pytest.raises(AssertionError):
        DeterministicRegisterPool(())
    with pytest.raises(AssertionError):
        DeterministicRegisterPool((-1,))
    with pytest.raises(AssertionError):
        DeterministicRegisterPool((1, 1))

    pool = DeterministicRegisterPool((0,))
    assignment = pool.checkout(RegisterRole("owned", 1, RegisterLifetime(0, 0)))
    pool._owners[assignment.first_register] = "other"
    with pytest.raises(AssertionError):
        pool.checkin("owned")


def test_writer_methods_have_complete_line_coverage() -> None:
    assert_writer_methods_have_complete_line_coverage(SHARED_WRITER_SOURCE_PATH)
