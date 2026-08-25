from dataclasses import replace
from pathlib import Path

import pytest

from tools.ggtensile.campaign import load_catalog
from tools.ggtensile.family_registry import (
    instance_hash,
    instance_name,
    launch_for_instance,
    mapping_for_instance,
    parse_instance,
    writer_for_instance,
)
from tools.ggtensile.grouped_mmq_bwd_physical import (
    derive_grouped_backward_physical_plan,
)
from tools.ggtensile.grouped_mmq_bwd_spec import (
    DerivedGroupedBackwardState,
    GroupedBackwardKernelSpec,
    GroupedBackwardOwnership,
    GroupedBackwardRowTail,
)
from tools.ggtensile.identity import KernelFamily
from tools.ggtensile.inspection import inspect_artifact
from tools.ggtensile.kernel_abi import GROUPED_BACKWARD_ABI
from tools.ggtensile.kernel_instance import KernelInstance
from tools.ggtensile.mmq_bwd_spec import (
    BackwardBitfieldShiftPlacement,
    BackwardExtraction,
    BackwardMetadataLoad,
)
from tools.ggtensile.model import ProblemSize, ProblemType, SchemaError
from tools.ggtensile.runtime import GroupedBackwardModule
from tools.ggtensile.toolchain import Toolchain
from tools.ggtensile.validation import validate_grouped_backward_solution

_CONFIG = Path(__file__).resolve().parents[2] / "tools/ggtensile/configs"
_CATALOG_INSTANCE = (
    load_catalog(_CONFIG / "mmq_grouped_bwd_q4_k_catalog.json").entries[1].instance
)
assert isinstance(_CATALOG_INSTANCE.kernel_spec, GroupedBackwardKernelSpec)
_PILOT_SPEC = replace(
    _CATALOG_INSTANCE.kernel_spec,
    compute=replace(
        _CATALOG_INSTANCE.kernel_spec.compute,
        pipeline=replace(
            _CATALOG_INSTANCE.kernel_spec.compute.pipeline,
            global_read_prefetch=1,
            iteration=replace(
                _CATALOG_INSTANCE.kernel_spec.compute.pipeline.iteration,
                prefetch_activation=False,
                interleave_wmma_waits=False,
            ),
        ),
        memory=replace(
            _CATALOG_INSTANCE.kernel_spec.compute.memory,
            lds_pad_b=0,
        ),
        decode=replace(
            _CATALOG_INSTANCE.kernel_spec.compute.decode,
            dependency_width=1,
        ),
    ),
    ownership=GroupedBackwardOwnership.from_mapping(
        {"Kind": "Serial", "EmptyTilePolicy": "Suppress"}
    ),
    row_tail=GroupedBackwardRowTail.from_mapping({"Kind": "Masked"}),
)


def _key(rows: int | None = None, quant_type: str = "Q4_K") -> KernelInstance:
    if quant_type == "Q2_K":
        size = ProblemSize(12_288 if rows is None else rows, 2048, 4096)
    else:
        size = ProblemSize(16_384 if rows is None else rows, 512, 2048)
    spec = _PILOT_SPEC
    if quant_type == "Q5_K":
        spec = replace(
            spec,
            compute=replace(
                spec.compute,
                decode=replace(
                    spec.compute.decode,
                    extraction=BackwardExtraction.Packed,
                    bitfield_shift_placement=BackwardBitfieldShiftPlacement.Inline,
                    metadata_load=BackwardMetadataLoad.Scalar,
                ),
            ),
        )
    return KernelInstance.for_gfx1151(
        KernelFamily.GroupedBackward,
        ProblemType.grouped_mmq_backward(quant_type),
        size,
        spec,
    )


def _spec(key: KernelInstance) -> GroupedBackwardKernelSpec:
    assert isinstance(key.kernel_spec, GroupedBackwardKernelSpec)
    return key.kernel_spec


def _with_spec(key: KernelInstance, spec: GroupedBackwardKernelSpec) -> KernelInstance:
    return replace(key, kernel_spec=spec)


def _mixed_key(rows: int = 16_384) -> KernelInstance:
    base = _key(rows)
    spec = _spec(base)
    compute = replace(
        spec.compute,
        pipeline=replace(
            spec.compute.pipeline,
            global_read_prefetch=2,
            iteration=replace(
                spec.compute.pipeline.iteration,
                prefetch_activation=True,
                interleave_wmma_waits=True,
            ),
        ),
        memory=replace(spec.compute.memory, lds_pad_b=8),
        decode=replace(spec.compute.decode, dependency_width=4),
    )
    return _with_spec(
        base,
        replace(
            spec,
            compute=compute,
            row_tail=GroupedBackwardRowTail.from_mapping(
                {"Kind": "SecondaryTile", "ThresholdRows": 64, "TileRows": 64}
            ),
        ),
    )


def _problem(key: KernelInstance) -> ProblemSize:
    assert isinstance(key.problem, ProblemSize)
    return key.problem


def _validate(key: KernelInstance) -> None:
    validate_grouped_backward_solution(
        _problem(key), key.problem_type.quant_data_type, _spec(key)
    )


def _state(key: KernelInstance) -> DerivedGroupedBackwardState:
    return DerivedGroupedBackwardState.from_problem_spec(
        _problem(key), key.problem_type.quant_data_type, _spec(key)
    )


def _writer(key: KernelInstance, toolchain: Toolchain):
    return writer_for_instance(key, toolchain)


def test_grouped_backward_identity_roundtrip_and_formula_compatible_rows() -> None:
    first = _key()
    assert parse_instance(mapping_for_instance(first)) == first
    _validate(first)
    assert instance_name(first).startswith("grouped_mmq_bwd_q4_k_")
    for rows in (16_384, 65_536, 262_144):
        _validate(_key(rows))
    for rows in (16_383, 32_768, 262_145):
        key = _key(rows)
        assert parse_instance(mapping_for_instance(key)) == key
        _validate(key)


def test_grouped_backward_q2_identity_and_source() -> None:
    for rows in (12_288, 49_152, 196_608):
        _validate(_key(rows, "Q2_K"))
    for rows in (12_287, 16_384, 196_609):
        key = _key(rows, "Q2_K")
        assert parse_instance(mapping_for_instance(key)) == key
        _validate(key)

    q2 = _key(quant_type="Q2_K")
    assert parse_instance(mapping_for_instance(q2)) == q2
    _validate(q2)
    assert "grouped_mmq_bwd_q2_k" in instance_name(q2)
    state = _state(q2)
    assert state.contract.quant_format.block_bytes == 84

    source = _writer(q2, Toolchain.discover()).source()
    assert "GGTensile Q2_K grouped MMQ backward" in source
    assert "Decode Q2_K two-bit payload" in source
    assert "s_cmp_lg_u32 s22, 2752512" in source
    assert "s_lshl_b32 s30, s27, 13" in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    assert source.count("s_barrier") == 2


def test_grouped_backward_q2_dependency_batch_identity_and_source() -> None:
    key = _key(quant_type="Q2_K")
    spec = _spec(key)
    candidate = replace(
        key,
        kernel_spec=replace(
            spec,
            compute=replace(
                spec.compute,
                decode=replace(spec.compute.decode, dependency_width=4),
            ),
        ),
    )
    assert parse_instance(mapping_for_instance(candidate)) == candidate
    _validate(candidate)
    spec = mapping_for_instance(candidate)["KernelSpec"]
    assert isinstance(spec, dict)
    assert spec["Decode"] == {"DecodeDependencyWidth": 4}

    physical = derive_grouped_backward_physical_plan(_state(candidate)).primary
    assert physical.dependency_decode.dependency_width == 4
    values = physical.dependency_decode.value_registers
    assert values == tuple(physical.registers.valu_b + 2 * slot for slot in range(4))

    source = _writer(candidate, Toolchain.discover()).source()
    conversions = [
        source.index(f"v_cvt_f32_ubyte{slot}_e32 v{value}")
        for slot, value in enumerate(values)
    ]
    first_fma = source.index(f"v_fma_f32 v{values[0]}", conversions[-1])
    assert max(conversions) < first_fma


def test_grouped_backward_q5_identity_and_source() -> None:
    q4 = _key()
    q5 = _key(quant_type="Q5_K")
    assert parse_instance(mapping_for_instance(q5)) == q5
    _validate(q5)
    assert instance_hash(q4) != instance_hash(q5)
    assert "grouped_mmq_bwd_q5_k" in instance_name(q5)
    state = _state(q5)
    assert state.contract.quant_format.block_bytes == 176

    source = _writer(q5, Toolchain.discover()).source()
    assert "GGTensile Q5_K grouped MMQ backward" in source
    assert "Build Q5_K payload and scale-byte addresses." in source
    assert "Decode Q5_K low nibbles and high payload bits into LDS." in source
    assert "s_cmp_lg_u32 s22, 720896" in source
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32
    assert source.count("s_barrier") == 2


def test_grouped_backward_accepts_formula_compatible_noncatalog_shape() -> None:
    key = replace(_key(12_345), problem=ProblemSize(12_345, 1024, 1024))
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)
    state = _state(key)
    assert state.contract.problem_size == ProblemSize(12_345, 1024, 1024)


def test_grouped_backward_rejects_incomplete_formula_dimensions() -> None:
    with pytest.raises(SchemaError, match="quant blocks"):
        _state(replace(_key(), problem=ProblemSize(16_384, 513, 2048)))
    with pytest.raises(SchemaError, match="DepthU32"):
        _state(replace(_key(), problem=ProblemSize(16_384, 512, 2049)))


def test_grouped_backward_iq2_s_identity_source_and_inspection(tmp_path) -> None:
    for rows in (16_384, 65_536, 262_144):
        _validate(_key(rows, "IQ2_S"))
    for rows in (16_383, 32_768, 262_145):
        key = _key(rows, "IQ2_S")
        assert parse_instance(mapping_for_instance(key)) == key
        _validate(key)

    key = _key(quant_type="IQ2_S")
    assert parse_instance(mapping_for_instance(key)) == key
    assert "grouped_mmq_bwd_iq2_s" in instance_name(key)
    state = _state(key)
    assert state.contract.quant_format.block_bytes == 82

    toolchain = Toolchain.discover()
    writer = _writer(key, toolchain)
    source = writer.source()
    assert "GGTensile IQ2_S grouped MMQ backward" in source
    assert "Map each lane to one IQ2_S aligned 16-value group." in source
    assert "Decode IQ2_S codebook values and signed scales into LDS." in source
    assert "s_getpc_b64 s[36:37]" in source
    assert "s_mov_b32 s13, 0x03020100" in source
    assert "s_cmp_lg_u32 s22, 335872" in source
    assert source.count("global_load_b64") == 0
    assert source.count("ds_load_b64") == 4
    assert "Stage the IQ2_S codebook once in disjoint LDS." in source
    assert source.count("v_perm_b32") == 8
    assert source.count(".quad") == 256
    assert ".size .LGGTensileIQ2SGrid, 8192" in source

    assembly = tmp_path / "iq2_s.s"
    object_path = tmp_path / "iq2_s.o"
    code_object = tmp_path / "iq2_s.hsaco"
    writer.write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    result = inspect_artifact(key, code_object, toolchain)
    assert result.vgpr_count == 194
    assert result.sgpr_count == 38
    assert result.lds_num_bytes == 16384
    assert result.private_segment_bytes == 0
    assert result.vgpr_spill_count == 0
    assert result.sgpr_spill_count == 0
    assert result.wmma_count == 32
    assert result.barrier_count == 3


def test_grouped_backward_rejects_unsupported_quant_type() -> None:
    with pytest.raises(ValueError, match="unsupported grouped"):
        ProblemType.grouped_mmq_backward("Q3_K")


def test_grouped_backward_secondary_tail_roundtrip_and_schema() -> None:
    key = _mixed_key()
    assert parse_instance(mapping_for_instance(key)) == key
    mapping = mapping_for_instance(key)
    spec = mapping["KernelSpec"]
    assert isinstance(spec, dict)
    ownership = spec["Ownership"]
    assert isinstance(ownership, dict)
    assert ownership == {"Kind": "Serial", "EmptyTilePolicy": "Suppress"}
    spec["RowTail"] = {
        "Kind": "SecondaryTile",
        "ThresholdRows": "96",
        "TileRows": 48,
    }
    with pytest.raises(SchemaError, match="RowTail"):
        parse_instance(mapping)


def _parameterized_key() -> KernelInstance:
    base = _key()
    spec = _spec(base)
    compute = replace(
        spec.compute,
        geometry=replace(
            spec.compute.geometry,
            matrix_instruction=(16, 16, 16, 1, 1, 4, 4, 4, 1),
            macro_tile0=256,
            macro_tile1=64,
        ),
        pipeline=replace(
            spec.compute.pipeline,
            global_read_prefetch=2,
            iteration=replace(
                spec.compute.pipeline.iteration,
                prefetch_activation=True,
                interleave_wmma_waits=True,
            ),
        ),
        memory=replace(spec.compute.memory, lds_pad_b=8),
        decode=replace(spec.compute.decode, dependency_width=4),
    )
    return replace(
        base,
        kernel_spec=replace(
            spec,
            compute=compute,
            row_tail=GroupedBackwardRowTail.from_mapping(
                {"Kind": "SecondaryTile", "ThresholdRows": 96, "TileRows": 128}
            ),
        ),
    )


def test_grouped_backward_parameterized_tail_roundtrip() -> None:
    key = _parameterized_key()
    mapping = mapping_for_instance(key)
    assert parse_instance(mapping) == key
    _validate(key)
    spec = mapping["KernelSpec"]
    assert isinstance(spec, dict)
    row_tail = spec["RowTail"]
    assert row_tail == {
        "Kind": "SecondaryTile",
        "ThresholdRows": 96,
        "TileRows": 128,
    }


def test_grouped_backward_parameterized_tail_build_and_inspect(tmp_path) -> None:
    key = _parameterized_key()
    toolchain = Toolchain.discover()
    assembly = tmp_path / "parameterized.s"
    object_path = tmp_path / "parameterized.o"
    code_object = tmp_path / "parameterized.hsaco"
    writer = _writer(key, toolchain)
    writer.write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    result = inspect_artifact(key, code_object, toolchain)
    assert result.vgpr_count == 234
    assert result.sgpr_count == 35
    assert result.lds_num_bytes == 5120
    assert result.wmma_count == 48
    assert result.barrier_count == 4


@pytest.mark.parametrize(
    ("threshold_rows", "tile_rows"),
    ((0, 64), (65, 64), (96, 48)),
)
def test_grouped_backward_parameterized_tail_rejects_incompatible_geometry(
    threshold_rows: int, tile_rows: int
) -> None:
    base = _key()
    spec = _spec(base)
    mapping = mapping_for_instance(base)
    spec = mapping["KernelSpec"]
    assert isinstance(spec, dict)
    spec["RowTail"] = {
        "Kind": "SecondaryTile",
        "ThresholdRows": threshold_rows,
        "TileRows": tile_rows,
    }
    with pytest.raises(SchemaError):
        parse_instance(mapping)


def test_grouped_backward_mixed_tail_source_is_namespaced() -> None:
    key = _mixed_key()
    source = _writer(key, Toolchain.discover()).source()
    labels = tuple(
        line.strip() for line in source.splitlines() if line.startswith(".L")
    )
    assert len(labels) == len(set(labels))
    assert source.count("v_wmma_f32_16x16x16_bf16") == 48
    assert source.count("s_barrier") == 4
    assert source.count(".LDepthULoop:") == 1
    assert source.count(".LDepthULoopTail:") == 1
    assert source.count(".LScaleOdd:") == 1
    assert source.count(".LScaleOddTail:") == 1
    assert "v_mov_b32 v127, v207" in source


def test_grouped_backward_mixed_tail_rejects_invalid_primary_geometry() -> None:
    pilot = _PILOT_SPEC
    invalid = replace(
        _key(),
        kernel_spec=replace(
            pilot,
            row_tail=GroupedBackwardRowTail.from_mapping(
                {"Kind": "SecondaryTile", "ThresholdRows": 64, "TileRows": 64}
            ),
            compute=replace(
                pilot.compute,
                geometry=replace(pilot.compute.geometry, macro_tile0=64),
            ),
        ),
    )
    with pytest.raises(AssertionError):
        _validate(invalid)


def test_grouped_backward_secondary_tail_rejects_work_group_mapping() -> None:
    key = _mixed_key()
    spec = _spec(key)
    mapped = replace(
        key,
        kernel_spec=replace(
            spec,
            compute=replace(
                spec.compute,
                geometry=replace(spec.compute.geometry, work_group_mapping=2),
            ),
        ),
    )
    with pytest.raises(AssertionError):
        _validate(mapped)


def test_grouped_backward_rejects_unknown_ownership_and_decodes_mapping() -> None:
    mapping = mapping_for_instance(_key())
    spec = mapping["KernelSpec"]
    assert isinstance(spec, dict)
    ownership = spec["Ownership"]
    assert isinstance(ownership, dict)
    ownership["Kind"] = "ImplicitTasks"
    with pytest.raises(SchemaError, match="ownership"):
        parse_instance(mapping)

    spec = _spec(_key())
    mapped = replace(
        _key(),
        kernel_spec=replace(
            spec,
            compute=replace(
                spec.compute,
                geometry=replace(spec.compute.geometry, work_group_mapping=2),
            ),
        ),
    )
    _validate(mapped)
    assert launch_for_instance(mapped).grid == (8, 256, 1)
    source = _writer(mapped, Toolchain.discover()).source()
    assert "Decode WGM-packed grid X into N and an M-task lane." in source
    assert "s_and_b32 s13, s2, 1" in source
    assert "s_lshr_b32 s2, s2, 1" in source
    assert "s_add_u32 s2, s2, 2" in source
    assert "s_mul_i32 s34, s4, 2" not in source

    invalid = replace(
        mapped,
        kernel_spec=replace(
            _spec(mapped),
            compute=replace(
                _spec(mapped).compute,
                geometry=replace(
                    _spec(mapped).compute.geometry,
                    work_group_mapping=3,
                ),
            ),
        ),
    )
    with pytest.raises(AssertionError):
        _validate(invalid)


@pytest.mark.parametrize(
    ("ownership", "split_factor"),
    (
        ("SplitRoutes2", 2),
        ("SplitRoutes4", 4),
        ("SplitRoutes8", 8),
        ("SplitRoutes16", 16),
        ("SplitRoutes32", 32),
    ),
)
def test_grouped_backward_split_route_identity_and_source(
    ownership: str, split_factor: int
) -> None:
    pilot = _PILOT_SPEC
    key = _with_spec(
        _key(),
        replace(
            pilot,
            ownership=GroupedBackwardOwnership.from_mapping(
                {
                    "Kind": "RouteSplit",
                    "RouteSplitFactor": split_factor,
                    "EmptyTilePolicy": "Suppress",
                }
            ),
        ),
    )
    assert parse_instance(mapping_for_instance(key)) == key
    _validate(key)

    source = _writer(key, Toolchain.discover()).source()
    assert ".amdhsa_system_sgpr_workgroup_id_z 1" in source
    assert "s_mov_b32 s2, s4" in source
    assert f"s_add_u32 s2, s2, {split_factor}" in source
    assert "s_cmp_ge_u32 s34, s27" in source


def test_grouped_split64_route_ownership_is_format_neutral() -> None:
    iq2 = _key(quant_type="IQ2_S")
    split64 = _with_spec(
        iq2,
        replace(
            _spec(iq2),
            ownership=GroupedBackwardOwnership.from_mapping(
                {
                    "Kind": "RouteSplit",
                    "RouteSplitFactor": 64,
                    "EmptyTilePolicy": "Suppress",
                }
            ),
        ),
    )
    _validate(split64)
    assert parse_instance(mapping_for_instance(split64)) == split64
    source = _writer(split64, Toolchain.discover()).source()
    assert ".amdhsa_system_sgpr_workgroup_id_z 1" in source
    assert "s_add_u32 s2, s2, 64" in source

    q4 = _key()
    q4_split64 = _with_spec(
        q4,
        replace(
            _spec(q4),
            ownership=GroupedBackwardOwnership.from_mapping(
                {
                    "Kind": "RouteSplit",
                    "RouteSplitFactor": 64,
                    "EmptyTilePolicy": "Suppress",
                }
            ),
        ),
    )
    _validate(q4_split64)


def test_grouped_backward_runtime_launch_configuration_uses_typed_geometry() -> None:
    key = _key()
    state = _state(key)
    assert GroupedBackwardModule._launch_configuration_for_state(state, 7) == (
        (4, 7, 1),
        (32, 4, 1),
        0,
    )


def test_grouped_backward_physical_plan_is_deterministic() -> None:
    state = _state(_key())
    first = derive_grouped_backward_physical_plan(state)
    second = derive_grouped_backward_physical_plan(state)
    assert first == second
    assert first.primary.resources.vgprs == 192
    assert first.primary.resources.sgprs == 35
    assert first.primary.resources.lds_bytes == 8192
    assert first.primary.resources.private_bytes == 0
    assert first.route.saved_exec == 31
    assert first.route.pointer_temporary == 32


def test_grouped_backward_abi_matches_specialized_launcher() -> None:
    assert GROUPED_BACKWARD_ABI.segment_size == 56
    assert GROUPED_BACKWARD_ABI.metadata_arguments == (
        ("grad_output", 0, 8, "global_buffer", "bf16"),
        ("packed_weight", 8, 8, "global_buffer", "struct"),
        ("grad_input", 16, 8, "global_buffer", "bf16"),
        ("expert_indices", 24, 8, "global_buffer", "i64"),
        ("expert_offsets", 32, 8, "global_buffer", "i32"),
        ("num_experts", 40, 4, "by_value", "i32"),
        ("rows", 44, 4, "by_value", "i32"),
        ("bytes_per_expert", 48, 8, "by_value", "i64"),
    )


def test_grouped_backward_source_has_route_and_tail_guards() -> None:
    source = _writer(_key(), Toolchain.discover()).source()
    assert "Load the routed 56-byte grouped backward ABI" in source
    assert "s_cmp_lg_u32 s20, 256" in source
    assert "s_cmp_lg_u32 s21, 16384" in source
    assert "s_cmp_lg_u32 s22, 589824" in source
    assert "s_load_dwordx2 s[28:29], s[16:17]" in source
    assert "s_mul_hi_u32 s33, s28, s22" in source
    assert "s_and_saveexec_b32 s31, vcc_lo" in source
    assert "s_mov_b32 exec_lo, s31" in source
    assert "Count active 16-row M consumers for this wave and route tile." in source
    assert "s_cbranch_scc1 .LGroupedBackwardInactiveWave0" in source
    assert "s_cbranch_scc1 .LGroupedBackwardInactiveM1" in source
    assert "s_cbranch_scc1 .LGroupedBackwardInactiveStoreWave" in source
    assert source.count(".LGroupedBackwardRowTile:") == 1
    assert source.count("v_wmma_f32_16x16x16_bf16") == 32


def test_grouped_backward_build_is_deterministic_and_inspectable(tmp_path) -> None:
    key = _key()
    toolchain = Toolchain.discover()
    first = _writer(key, toolchain)
    second = _writer(key, toolchain)
    assert first.source() == second.source()

    assembly = tmp_path / "kernel.s"
    object_path = tmp_path / "kernel.o"
    code_object = tmp_path / "kernel.hsaco"
    first.write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    result = inspect_artifact(key, code_object, toolchain)
    assert result.kernarg_segment_size == 56
    assert result.max_flat_workgroup_size == 128
    assert result.vgpr_count == 192
    assert result.sgpr_count == 35
    assert result.lds_num_bytes == 8192
    assert result.private_segment_bytes == 0
    assert result.vgpr_spill_count == 0
    assert result.sgpr_spill_count == 0
    assert result.wmma_count == 32
    assert result.barrier_count == 2


def test_grouped_backward_mixed_tail_inspection_counts(tmp_path) -> None:
    key = _mixed_key()
    toolchain = Toolchain.discover()
    assembly = tmp_path / "mixed.s"
    object_path = tmp_path / "mixed.o"
    code_object = tmp_path / "mixed.hsaco"
    writer = _writer(key, toolchain)
    writer.write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    result = inspect_artifact(key, code_object, toolchain)
    assert result.vgpr_count == 208
    assert result.sgpr_count == 35
    assert result.lds_num_bytes == 10240
    assert result.private_segment_bytes == 0
    assert result.vgpr_spill_count == 0
    assert result.sgpr_spill_count == 0
    assert result.wmma_count == 48
    assert result.barrier_count == 4
