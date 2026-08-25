import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from tests.ggtensile.grouped_forward_fixtures import (
    GroupedForwardTestSolutions as _Solutions,
)
from tests.ggtensile.grouped_forward_fixtures import grouped_forward_problem
from tools.ggtensile.family_registry import (
    instance_hash,
    instance_name,
    mapping_for_instance,
    parse_instance,
    problem_type_for_family,
    writer_for_instance,
)
from tools.ggtensile.grouped_mmq_fwd_inspection import (
    inspect_grouped_forward_artifact,
)
from tools.ggtensile.grouped_mmq_fwd_model import (
    GroupedActivationAddressing,
    GroupedForwardProblem,
)
from tools.ggtensile.grouped_mmq_fwd_physical import (
    GroupedActivationStageBounds,
    GroupedActivationStagingPlan,
)
from tools.ggtensile.grouped_mmq_fwd_spec import (
    DerivedGroupedForwardState,
    GroupedForwardKernelSpec,
    GroupedRowTileDispatchPolicy,
)
from tools.ggtensile.grouped_mmq_fwd_validation import (
    validate_grouped_forward_solution,
)
from tools.ggtensile.identity import KernelFamily
from tools.ggtensile.kernel_instance import KernelInstance
from tools.ggtensile.model import SchemaError
from tools.ggtensile.runtime import (
    GroupedForwardModule,
    HIPRuntimeError,
    InstalledGroupedForwardIQ2SJ64J32Module,
    InstalledGroupedForwardIQ2SJ64Module,
    InstalledGroupedForwardModule,
    InstalledGroupedForwardQ2J32J16Module,
    InstalledGroupedForwardQ2J32Module,
    InstalledGroupedForwardQ5J32Module,
    InstalledGroupedForwardQ5Module,
)
from tools.ggtensile.toolchain import Toolchain


def _instance(
    problem: GroupedForwardProblem, spec: GroupedForwardKernelSpec
) -> KernelInstance:
    return KernelInstance.for_gfx1151(
        KernelFamily.GroupedForward,
        problem_type_for_family(KernelFamily.GroupedForward, problem.quant_data_type),
        problem,
        spec,
    )


def _key(aggregate_rows: int = 16384) -> KernelInstance:
    return _instance(
        grouped_forward_problem("Q4_K", aggregate_rows),
        _Solutions.q4_k_serial_direct(),
    )


def _decoded_key(
    solution: GroupedForwardKernelSpec | None = None,
    aggregate_rows: int = 16384,
) -> KernelInstance:
    return _instance(
        grouped_forward_problem("Q4_K", aggregate_rows),
        solution or _Solutions.q4_k_serial_decoded_lds(),
    )


def _q5_key(
    solution: GroupedForwardKernelSpec | None = None,
    aggregate_rows: int = 16384,
) -> KernelInstance:
    return _instance(
        grouped_forward_problem("Q5_K", aggregate_rows),
        solution or _Solutions.q5_k_serial_decoded_lds(),
    )


def _q2_key(
    solution: GroupedForwardKernelSpec | None = None,
    aggregate_rows: int = 12_288,
) -> KernelInstance:
    return _instance(
        grouped_forward_problem("Q2_K", aggregate_rows),
        solution or _q2_selected_solution(aggregate_rows),
    )


def _q2_selected_solution(aggregate_rows: int) -> GroupedForwardKernelSpec:
    if aggregate_rows == 196_608:
        return _Solutions.q2_k_serial_decoded_lds_64_hip_distributed()
    return _Solutions.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2_distributed_mixed16()


def _iq2_s_key(aggregate_rows: int = 16_384) -> KernelInstance:
    return _instance(
        grouped_forward_problem("IQ2_S", aggregate_rows),
        _Solutions.iq2_s_serial_full_weight_lds_64(),
    )


def _iq2_s_linear_key(aggregate_rows: int = 16_384) -> KernelInstance:
    return _instance(
        grouped_forward_problem("IQ2_S", aggregate_rows),
        _Solutions.iq2_s_serial_full_weight_lds_64_linear_activation(),
    )


def _iq2_s_payload_prefetch_key(
    aggregate_rows: int = 16_384,
) -> KernelInstance:
    return _instance(
        grouped_forward_problem("IQ2_S", aggregate_rows),
        _Solutions.iq2_s_serial_full_weight_lds_64_linear_payload_prefetch(),
    )


def _problem_spec(
    instance: KernelInstance,
) -> tuple[GroupedForwardProblem, GroupedForwardKernelSpec]:
    assert isinstance(instance.problem, GroupedForwardProblem)
    assert isinstance(instance.kernel_spec, GroupedForwardKernelSpec)
    return instance.problem, instance.kernel_spec


def _validate(instance: KernelInstance) -> None:
    validate_grouped_forward_solution(*_problem_spec(instance))


def _state(instance: KernelInstance) -> DerivedGroupedForwardState:
    return DerivedGroupedForwardState.from_problem_spec(*_problem_spec(instance))


def _writer(instance: KernelInstance, toolchain: Toolchain):
    return writer_for_instance(instance, toolchain)


def _inspect(instance: KernelInstance, code_object: Path, toolchain: Toolchain):
    problem, spec = _problem_spec(instance)
    return inspect_grouped_forward_artifact(
        problem, spec, instance_name(instance), code_object, toolchain
    )


def test_grouped_key_rejects_unknown_root_field() -> None:
    mapping = mapping_for_instance(_key())
    mapping["Unknown"] = 1
    with pytest.raises(SchemaError, match="unknown"):
        parse_instance(mapping)


@pytest.mark.parametrize("aggregate_rows", (16_384, 65_536, 262_144))
def test_grouped_iq2_s_exact_production_keys_derive(aggregate_rows: int) -> None:
    instance = _iq2_s_key(aggregate_rows)
    assert parse_instance(mapping_for_instance(instance)) == instance
    assert isinstance(instance.problem, GroupedForwardProblem)
    assert isinstance(instance.kernel_spec, GroupedForwardKernelSpec)
    validate_grouped_forward_solution(instance.problem, instance.kernel_spec)
    state = DerivedGroupedForwardState.from_problem_spec(
        instance.problem, instance.kernel_spec
    )
    assert state.expected_packed_weight_shape == (256, 2048, 164)
    assert state.expected_activation_shape == (4, aggregate_rows, 144)
    assert state.expected_output_shape == (aggregate_rows, 2048)
    assert state.grid(256) == (32, 256, 1)
    assert state.physical_plan.resources.lds_bytes == 30_720


def test_grouped_iq2_s_writer_embeds_distributed_codebook_decode() -> None:
    source = writer_for_instance(_iq2_s_key(35), Toolchain.discover()).source()
    assert "GGTensile grouped IQ2_S MMQ forward" in source
    assert "Cooperatively decode one IQ2_S row half per workitem." in source
    assert "v_and_b32 v107, 1, v111" in source
    assert "v_and_b32 v107, 63, v0" not in source
    assert (
        sum(
            "global_load_b64" in line and "s[38:39]" in line
            for line in source.splitlines()
        )
        == 16
    )
    assert source.count("s_barrier") == 4
    assert '.section .rodata,"a",@progbits' in source
    assert source.count(".quad ") == 256


def test_grouped_activation_staging_derives_exact_and_ceil_masked_stages() -> None:
    staging = GroupedActivationStagingPlan(
        GroupedActivationAddressing.AggregateRowsTiled,
        block_bytes=144,
        participating_threads=128,
    )
    exact = staging.stage(64)
    assert exact.total_bytes == 9_216
    assert exact.loads_per_thread == 18
    assert exact.stage_dwords == 18
    assert exact.bounds is GroupedActivationStageBounds.Exact
    assert not exact.requires_bounds_mask

    masked = staging.stage(16)
    assert masked.total_bytes == 2_304
    assert masked.loads_per_thread == 5
    assert masked.stage_dwords == 5
    assert masked.bounds is GroupedActivationStageBounds.BoundsMasked
    assert masked.requires_bounds_mask


@pytest.mark.parametrize(
    ("macro_rows", "tail_rows", "expected"),
    (
        (16, 16, (16,)),
        (32, 16, (32, 16)),
        (64, 32, (64, 32)),
        (128, 64, (128, 64)),
        (128, 32, (128, 64, 32)),
        (128, 16, (128, 64, 32, 16)),
    ),
)
def test_grouped_row_dispatch_policy_derives_body_order(
    macro_rows: int, tail_rows: int, expected: tuple[int, ...]
) -> None:
    policy = GroupedRowTileDispatchPolicy.from_geometry(macro_rows, tail_rows)
    assert policy.body_rows == expected
    assert policy.body_row_tiles == tuple(rows // 16 for rows in expected)


def test_grouped_decoded_writer_emits_an_arbitrary_row_partition() -> None:
    base = _Solutions.q4_k_serial_decoded_lds()
    spec = replace(
        base,
        geometry=replace(base.geometry, tail_macro_tile0=16),
    )
    source = _writer(
        _decoded_key(spec, aggregate_rows=35), Toolchain.discover()
    ).source()
    assert ".LGroupedQ5KActivationRows16Dispatch0:" in source
    assert ".LGroupedQ5KMmaRows16Dispatch0:" in source
    assert ".LGroupedQ5KEpilogueRows16Dispatch:" in source


def test_grouped_iq2_s_linear_activation_stage_is_coalesced() -> None:
    source = _writer(_iq2_s_linear_key(35), Toolchain.discover()).source()
    assert "Linearly stage one coalesced 9,216-byte" in source
    assert "v_add_nc_u32 v101, 4096, v101" in source
    assert "v_add_nc_u32 v107, 8192, v107" in source
    assert source.count("s_barrier") == 4


def test_grouped_iq2_s_payload_prefetch_overlaps_fragment_correction() -> None:
    source = _writer(_iq2_s_payload_prefetch_key(35), Toolchain.discover()).source()
    assert "v_bfe_u32 v89, v87, 4, 4" in source
    assert "v_bfe_u32 v89, v33, 30, 2" in source
    assert "v_bfe_u32 v88, v34, 28, 4" in source
    assert "s_mov_b32 s32, 0x03020100" in source
    assert "v_mul_lo_u32 v90, 0x810204, v89" in source
    assert "v_and_or_b32 v90, v90, 0x04040404, s32" in source
    assert source.count("v_mul_f32 v32, 0.25, v32") == 1
    assert source.count("v_add_f32 v88, 0.5, v88") == 8
    assert "v_fmac_f32 v88, 0.5, v32" not in source
    prefetch = source.index("Prefetch IQ2_S half 0 group 1 payloads.")
    correction = source.index("v_cvt_f32_i32 v32, v32", prefetch)
    scale_read = source.index("ds_read2_b32 v[84:85]", correction)
    assert prefetch < correction < scale_read
    assert "s_waitcnt lgkmcnt(4)" in source
    assert "s_waitcnt lgkmcnt(6)" in source


def test_grouped_iq2_s_artifact_passes_strict_inspection(tmp_path: Path) -> None:
    key = _iq2_s_payload_prefetch_key(35)
    toolchain = Toolchain.discover()
    assembly = tmp_path / "kernel.s"
    obj = tmp_path / "kernel.o"
    code_object = tmp_path / "kernel.hsaco"
    _writer(key, toolchain).write(assembly)
    toolchain.assemble(assembly, obj)
    toolchain.link(obj, code_object)
    inspection = _inspect(key, code_object, toolchain)
    assert inspection.vgpr_count == 116
    assert inspection.sgpr_count == 40
    assert inspection.lds_num_bytes == 30_720
    assert inspection.wmma_count == 64
    assert inspection.barrier_count == 4
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


def test_grouped_iq2_s_rebuild_is_deterministic(tmp_path: Path) -> None:
    key = _iq2_s_payload_prefetch_key(65_536)
    toolchain = Toolchain.discover()
    sources = []
    code_objects = []
    for name in ("first", "second"):
        directory = tmp_path / name
        assembly = directory / "kernel.s"
        obj = directory / "kernel.o"
        code_object = directory / "kernel.hsaco"
        _writer(key, toolchain).write(assembly)
        toolchain.assemble(assembly, obj)
        toolchain.link(obj, code_object)
        sources.append(assembly.read_bytes())
        code_objects.append(code_object.read_bytes())
    assert sources[0] == sources[1]
    assert code_objects[0] == code_objects[1]


def test_installed_grouped_iq2_s_dispatch_preserves_b4_exception() -> None:
    pure = InstalledGroupedForwardIQ2SJ64Module.__new__(
        InstalledGroupedForwardIQ2SJ64Module
    )
    pure.problem = _problem_spec(_iq2_s_key(65_536))[0]
    with pytest.raises(HIPRuntimeError, match="mixed"):
        pure._launch_configuration(256)
    pure.problem = _problem_spec(_iq2_s_key(262_144))[0]
    assert pure._launch_configuration(256) == (
        (32, 256, 1),
        (32, 4, 1),
        30_976,
    )

    mixed = InstalledGroupedForwardIQ2SJ64J32Module.__new__(
        InstalledGroupedForwardIQ2SJ64J32Module
    )
    mixed.problem = _problem_spec(_iq2_s_key(65_536))[0]
    assert mixed._launch_configuration(256) == (
        (32, 256, 1),
        (32, 4, 1),
        30_976,
    )
    mixed.problem = _problem_spec(_iq2_s_key(262_144))[0]
    with pytest.raises(HIPRuntimeError, match="pure"):
        mixed._launch_configuration(256)


@pytest.mark.parametrize("aggregate_rows", (16384, 65536, 262144))
def test_grouped_q4_k_exact_production_keys_derive(
    aggregate_rows: int,
) -> None:
    key = _key(aggregate_rows)
    _validate(key)
    state = _state(key)
    assert state.expected_packed_weight_shape == (256, 2048, 288)
    assert state.expected_activation_shape == (4, aggregate_rows, 144)
    assert state.expected_output_shape == (aggregate_rows, 2048)
    assert state.grid(256) == (128, 256, 1)


@pytest.mark.parametrize("aggregate_rows", (12_288, 49_152, 196_608))
def test_grouped_q2_k_exact_production_keys_derive(
    aggregate_rows: int,
) -> None:
    key = _q2_key(aggregate_rows=aggregate_rows)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)
    state = _state(key)
    assert state.expected_packed_weight_shape == (256, 4096, 672)
    assert state.expected_activation_shape == (16, aggregate_rows, 144)
    assert state.expected_output_shape == (aggregate_rows, 4096)
    assert state.grid(256) == (64, 256, 1)
    assert "grouped_mmq_fwd_q2_k" in instance_name(key)
    assert key.kernel_spec == _q2_selected_solution(aggregate_rows)


def test_grouped_q2_k_writer_emits_f16_d2s6_unrolled_groups() -> None:
    solution = _Solutions.q2_k_serial_decoded_lds_32_unrolled()
    source = _writer(
        _q2_key(solution, aggregate_rows=35), Toolchain.discover()
    ).source()
    assert "GGTensile grouped Q2_K MMQ forward" in source
    assert "F16_D2S6" in source
    assert "Decode Q2_K two-bit payload" in source
    assert source.count("Statically lowered Q2_K group") > 0
    assert "s_cmp_ge_u32 s31, 6" not in source
    assert source.count("s_barrier") == 4


def test_grouped_q2_k_writer_emits_distributed_mixed_tail() -> None:
    solution = _Solutions.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2_distributed_mixed16()
    source = _writer(
        _q2_key(solution, aggregate_rows=35), Toolchain.discover()
    ).source()
    assert (
        "Queue eight distributed Q2 payload/scale/dm rows before conversion." in source
    )
    assert "s_waitcnt vmcnt(21)" in source
    assert "ds_write2st64_b32" in source
    assert "offset1:10" in source
    assert ".LGroupedQ2KMmaTailDispatch0" in source
    assert ".LGroupedQ4KEpilogueTailDispatch" in source
    assert ".LGroupedQ4KActivationFull0Tail" not in source


@pytest.mark.parametrize(
    "solution, expected_vgprs, expected_lds, expected_wmmas",
    (
        (
            _Solutions.q2_k_serial_decoded_lds_32(),
            135,
            25_600,
            12,
        ),
        (
            _Solutions.q2_k_serial_decoded_lds_32_unrolled(),
            135,
            25_600,
            40,
        ),
        (
            _Solutions.q2_k_serial_decoded_lds_64_unrolled(),
            159,
            30_208,
            80,
        ),
        (
            _Solutions.q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2_distributed_mixed16(),
            135,
            25_600,
            60,
        ),
        (
            _Solutions.q2_k_serial_decoded_lds_64_hip_distributed(),
            159,
            30_208,
            80,
        ),
    ),
)
def test_grouped_q2_k_artifact_passes_strict_inspection(
    tmp_path: Path,
    solution: GroupedForwardKernelSpec,
    expected_vgprs: int,
    expected_lds: int,
    expected_wmmas: int,
) -> None:
    key = _q2_key(solution, aggregate_rows=35)
    toolchain = Toolchain.discover()
    assembly = tmp_path / "kernel.s"
    obj = tmp_path / "kernel.o"
    code_object = tmp_path / "kernel.hsaco"
    _writer(key, toolchain).write(assembly)
    toolchain.assemble(assembly, obj)
    toolchain.link(obj, code_object)
    inspection = _inspect(key, code_object, toolchain)
    assert inspection.vgpr_count == expected_vgprs
    assert inspection.sgpr_count == 40
    assert inspection.lds_num_bytes == expected_lds
    assert inspection.wmma_count > 0
    assert inspection.barrier_count == 4
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


@pytest.mark.parametrize("aggregate_rows", (49_152, 196_608))
def test_grouped_q2_k_selected_rebuild_is_deterministic(
    tmp_path: Path, aggregate_rows: int
) -> None:
    key = _q2_key(aggregate_rows=aggregate_rows)
    toolchain = Toolchain.discover()
    sources = []
    code_objects = []
    for name in ("first", "second"):
        directory = tmp_path / name
        assembly = directory / "kernel.s"
        obj = directory / "kernel.o"
        code_object = directory / "kernel.hsaco"
        _writer(key, toolchain).write(assembly)
        toolchain.assemble(assembly, obj)
        toolchain.link(obj, code_object)
        sources.append(assembly)
        code_objects.append(code_object)
    assert sources[0].read_bytes() == sources[1].read_bytes()
    assert code_objects[0].read_bytes() == code_objects[1].read_bytes()


def test_installed_grouped_q2_k_dispatch_preserves_exact_exception() -> None:
    pure = InstalledGroupedForwardQ2J32Module.__new__(
        InstalledGroupedForwardQ2J32Module
    )
    pure.problem = _problem_spec(_q2_key(aggregate_rows=49_152))[0]
    with pytest.raises(HIPRuntimeError, match="mixed"):
        pure._launch_configuration(256)
    pure.problem = _problem_spec(_q2_key(aggregate_rows=12_288))[0]
    assert pure._launch_configuration(128) == ((64, 128, 1), (32, 4, 1), 30_336)

    mixed = InstalledGroupedForwardQ2J32J16Module.__new__(
        InstalledGroupedForwardQ2J32J16Module
    )
    mixed.problem = _problem_spec(_q2_key(aggregate_rows=49_152))[0]
    assert mixed._launch_configuration(256) == (
        (64, 256, 1),
        (32, 4, 1),
        30_336,
    )
    mixed.problem = _problem_spec(_q2_key(aggregate_rows=12_288))[0]
    with pytest.raises(HIPRuntimeError, match="pure"):
        mixed._launch_configuration(128)


def test_grouped_q2_k_rejects_cross_format_solution() -> None:
    key = _instance(
        grouped_forward_problem("Q2_K", 35),
        _Solutions.q4_k_serial_decoded_lds(),
    )
    with pytest.raises(AssertionError):
        _validate(key)


@pytest.mark.parametrize("aggregate_rows", (16384, 65536, 262144))
def test_grouped_q5_k_exact_production_keys_derive(
    aggregate_rows: int,
) -> None:
    key = _q5_key(aggregate_rows=aggregate_rows)
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)
    state = _state(key)
    assert state.expected_packed_weight_shape == (256, 2048, 352)
    assert state.expected_activation_shape == (4, aggregate_rows, 144)
    assert state.expected_output_shape == (aggregate_rows, 2048)
    assert state.grid(256) == (32, 256, 1)
    assert "grouped_mmq_fwd_q5_k" in instance_name(key)


def test_grouped_q5_k_writer_emits_high_bit_decode() -> None:
    solution = _Solutions.q5_k_serial_decoded_lds_scheduled_a1d2p2()
    source = _writer(
        _q5_key(solution, aggregate_rows=35), Toolchain.discover()
    ).source()
    assert "GGTensile grouped Q5_K MMQ forward" in source
    assert "Cooperatively decode Q5_K payload into padded LDS rows." in source
    assert "Build Q5_K high-bit and low-nibble payload addresses." in source
    assert "offset:16" in source
    assert "offset:48" in source
    assert "v_lshl_or_b32" in source
    assert source.count("s_barrier") == 4


def test_grouped_q5_k_writer_emits_three_way_row_dispatch() -> None:
    solution = _Solutions.q5_k_serial_decoded_lds_scheduled_mixed64_mixed32_a1d4p2()
    source = _writer(
        _q5_key(solution, aggregate_rows=388), Toolchain.discover()
    ).source()
    assert source.count("s_cmp_le_u32 s36, 32") == 5
    assert source.count("s_cmp_le_u32 s36, 64") == 5
    assert ".LGroupedQ5KActivationRows32Dispatch0:" in source
    assert ".LGroupedQ5KActivationRows64Dispatch0:" in source
    assert ".LGroupedQ5KMmaRows32Dispatch0:" in source
    assert ".LGroupedQ5KMmaRows64Dispatch0:" in source
    assert ".LGroupedQ5KEpilogueRows32Dispatch:" in source
    assert ".LGroupedQ5KEpilogueRows64Dispatch:" in source


@pytest.mark.parametrize(
    "solution, expected_vgprs, expected_lds, expected_wmmas",
    (
        (
            _Solutions.q5_k_serial_decoded_lds_scheduled_a1d2p2(),
            239,
            38_400,
            32,
        ),
        (
            _Solutions.q5_k_serial_decoded_lds_scheduled_mixed64_a1d2p2(),
            239,
            38_400,
            48,
        ),
        (
            _Solutions.q5_k_serial_decoded_lds_scheduled_mixed64_mixed32_a1d4p2(),
            239,
            38_400,
            56,
        ),
        (
            _Solutions.q5_k_serial_decoded_lds_64_scheduled_a1d4p2(),
            159,
            29_184,
            16,
        ),
        (
            _Solutions.q5_k_serial_decoded_lds_64_scheduled_mixed32_a1d4p2(),
            159,
            29_184,
            24,
        ),
    ),
)
def test_grouped_q5_k_artifact_passes_strict_inspection(
    tmp_path: Path,
    solution: GroupedForwardKernelSpec,
    expected_vgprs: int,
    expected_lds: int,
    expected_wmmas: int,
) -> None:
    key = _q5_key(solution, aggregate_rows=35)
    toolchain = Toolchain.discover()
    assembly = tmp_path / "kernel.s"
    obj = tmp_path / "kernel.o"
    code_object = tmp_path / "kernel.hsaco"
    _writer(key, toolchain).write(assembly)
    toolchain.assemble(assembly, obj)
    toolchain.link(obj, code_object)
    inspection = _inspect(key, code_object, toolchain)
    assert inspection.vgpr_count == expected_vgprs
    assert inspection.sgpr_count == 40
    assert inspection.lds_num_bytes == expected_lds
    assert inspection.wmma_count > 0
    assert inspection.barrier_count == 4
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


def test_grouped_q5_k_selected_rebuild_is_deterministic(tmp_path: Path) -> None:
    key = _q5_key(
        _Solutions.q5_k_serial_decoded_lds_scheduled_mixed64_mixed32_a1d4p2(),
        aggregate_rows=65536,
    )
    toolchain = Toolchain.discover()
    sources = []
    code_objects = []
    for name in ("first", "second"):
        directory = tmp_path / name
        assembly = directory / "kernel.s"
        obj = directory / "kernel.o"
        code_object = directory / "kernel.hsaco"
        _writer(key, toolchain).write(assembly)
        toolchain.assemble(assembly, obj)
        toolchain.link(obj, code_object)
        sources.append(assembly)
        code_objects.append(code_object)
    assert sources[0].read_bytes() == sources[1].read_bytes()
    assert code_objects[0].read_bytes() == code_objects[1].read_bytes()


def test_grouped_q5_k_accepts_shared_decoded_spec() -> None:
    key = _instance(
        grouped_forward_problem("Q5_K", 35),
        _Solutions.q4_k_serial_decoded_lds(),
    )
    _validate(key)
    assert parse_instance(mapping_for_instance(key)) == key


def test_installed_grouped_q5_k_launch_geometries_follow_dispatch() -> None:
    j64 = InstalledGroupedForwardQ5Module.__new__(InstalledGroupedForwardQ5Module)
    j64.problem = _problem_spec(_q5_key(aggregate_rows=65536))[0]
    assert j64._launch_configuration(256) == (
        (32, 256, 1),
        (32, 4, 1),
        28_928,
    )
    small_route = InstalledGroupedForwardQ5Module.__new__(
        InstalledGroupedForwardQ5Module
    )
    small_route.problem = _problem_spec(_q5_key(aggregate_rows=16384))[0]
    with pytest.raises(HIPRuntimeError, match="dedicated module"):
        small_route._launch_configuration(256)

    j32 = InstalledGroupedForwardQ5J32Module.__new__(InstalledGroupedForwardQ5J32Module)
    j32.problem = _problem_spec(_q5_key(aggregate_rows=16384))[0]
    assert j32._launch_configuration(256) == (
        (32, 256, 1),
        (32, 4, 1),
        24_192,
    )
    with pytest.raises(HIPRuntimeError, match="selects J64"):
        j32._launch_configuration(128)


def test_grouped_q4_k_decoded_plans_cover_row_tiles() -> None:
    cases = (
        (_Solutions.q4_k_serial_decoded_lds(), 239, 38_400, 32),
        (_Solutions.q4_k_serial_decoded_lds_64(), 159, 29_184, 16),
        (
            _Solutions.q4_k_serial_decoded_lds_64_scheduled_mixed32(),
            159,
            29_184,
            24,
        ),
    )
    for solution, vgprs, lds_bytes, wmmas in cases:
        key = _decoded_key(solution, aggregate_rows=35)
        assert parse_instance(mapping_for_instance(key)) == key
        _validate(key)
        state = _state(key)
        assert state.physical_plan.resources.vgprs == vgprs
        assert state.physical_plan.resources.sgprs == 40
        assert state.physical_plan.resources.lds_bytes == lds_bytes
        assert state.problem_size.m == 35
        assert (
            state.kernel_spec.decode.independent_metadata_extraction
            is solution.decode.independent_metadata_extraction
        )
        assert (
            state.kernel_spec.decode.defer_metadata_reads
            is solution.decode.defer_metadata_reads
        )
        assert solution.geometry.tail_macro_tile0 <= solution.geometry.macro_tile[0]
        row_tiles = sum(state.kernel_spec.row_dispatch.body_row_tiles)
        assert 4 * row_tiles == wmmas


def test_grouped_q4_k_decoded_writer_emits_tail_and_schedule_controls() -> None:
    solution = _Solutions.q4_k_serial_decoded_lds_64_scheduled_mixed32_a1d4p2()
    source = _writer(
        _decoded_key(solution, aggregate_rows=35), Toolchain.discover()
    ).source()
    assert "s_cmp_le_u32 s36, 32" in source
    assert "ds_write_b32" in source
    assert "s_waitcnt lgkmcnt(7)" in source
    assert "s_setprio 2" in source
    assert "s_setprio 0" in source
    assert source.count("s_barrier") == 4


@pytest.mark.parametrize(
    "solution, expected_vgprs, expected_lds, expected_wmmas",
    (
        (
            _Solutions.q4_k_serial_decoded_lds_scheduled_a1d2p2(),
            239,
            38_400,
            32,
        ),
        (
            _Solutions.q4_k_serial_decoded_lds_64_scheduled_mixed32_a1d4p2(),
            159,
            29_184,
            24,
        ),
    ),
)
def test_grouped_q4_k_decoded_artifact_passes_strict_inspection(
    tmp_path: Path,
    solution: GroupedForwardKernelSpec,
    expected_vgprs: int,
    expected_lds: int,
    expected_wmmas: int,
) -> None:
    key = _decoded_key(solution, aggregate_rows=35)
    toolchain = Toolchain.discover()
    assembly = tmp_path / "kernel.s"
    obj = tmp_path / "kernel.o"
    code_object = tmp_path / "kernel.hsaco"
    _writer(key, toolchain).write(assembly)
    toolchain.assemble(assembly, obj)
    toolchain.link(obj, code_object)
    inspection = _inspect(key, code_object, toolchain)
    assert inspection.vgpr_count == expected_vgprs
    assert inspection.sgpr_count == 40
    assert inspection.lds_num_bytes == expected_lds
    assert inspection.wmma_count > 0
    assert inspection.barrier_count == 4
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0


def test_grouped_q4_k_mixed_artifact_rebuild_is_deterministic(tmp_path: Path) -> None:
    key = _decoded_key(
        _Solutions.q4_k_serial_decoded_lds_64_scheduled_mixed32_a1d4p2(),
        aggregate_rows=35,
    )
    toolchain = Toolchain.discover()
    code_objects = []
    for index in ("first", "second"):
        directory = tmp_path / index
        assembly = directory / "kernel.s"
        obj = directory / "kernel.o"
        code_object = directory / "kernel.hsaco"
        writer = _writer(key, toolchain)
        assert writer.write(assembly) == writer.write(directory / "repeat.s")
        toolchain.assemble(assembly, obj)
        toolchain.link(obj, code_object)
        code_objects.append(code_object)
    assert code_objects[0].read_bytes() == code_objects[1].read_bytes()


def test_grouped_q4_k_identity_round_trips_strict_mapping() -> None:
    key = _key(35)
    assert parse_instance(mapping_for_instance(key)) == key
    assert instance_name(key).endswith(instance_hash(key)[6:])


def test_grouped_q4_k_rejects_inactive_solution_fields() -> None:
    solution = _Solutions.q4_k_serial_direct()
    key = _instance(
        grouped_forward_problem("Q4_K", 35),
        replace(
            solution,
            geometry=replace(solution.geometry, work_group=(64, 1, 1)),
        ),
    )
    with pytest.raises(AssertionError):
        _validate(key)


def test_grouped_q4_k_launch_geometries_match_each_mechanism() -> None:
    candidate = GroupedForwardModule.__new__(GroupedForwardModule)
    candidate.state = _state(_key(35))
    assert candidate._launch_configuration(4) == ((128, 4, 1), (32, 1, 1), 0)
    installed = InstalledGroupedForwardModule.__new__(InstalledGroupedForwardModule)
    assert installed._launch_configuration(4) == (
        (32, 4, 1),
        (32, 4, 1),
        28_928,
    )
    decoded = GroupedForwardModule.__new__(GroupedForwardModule)
    decoded.state = _state(
        _decoded_key(
            _Solutions.q4_k_serial_decoded_lds_64_scheduled_mixed32_a1d4p2(),
            aggregate_rows=35,
        )
    )
    assert decoded._launch_configuration(4) == ((32, 4, 1), (128, 1, 1), 0)


def test_grouped_q4_k_writer_emits_routed_abi_and_masks() -> None:
    source = _writer(_key(35), Toolchain.discover()).source()
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
    first_writer = _writer(key, toolchain)
    second_writer = _writer(key, toolchain)
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
    inspection = _inspect(key, first_code_object, toolchain)
    assert inspection.kernarg_segment_size == 64
    assert inspection.vgpr_count == 88
    assert inspection.sgpr_count == 32
    assert inspection.wmma_count == 16
    assert inspection.barrier_count == 0
    assert inspection.lds_num_bytes == 0
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0
