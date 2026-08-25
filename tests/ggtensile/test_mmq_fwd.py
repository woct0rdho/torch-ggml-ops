import json
from pathlib import Path

import pytest

from tests.ggtensile.ordinary_forward_fixtures import (
    q3_full_weight_tiled_lds_kernel_spec,
    q3_hip_tiled_lds_kernel_spec,
    q4_decoded_weight_lds_kernel_spec,
    q5_decoded_weight_lds_kernel_spec,
    q6_structured_kernel_spec,
    q8_direct_global_kernel_spec,
    q8_register_tiled_kernel_spec,
    q8_small_m_tiled_lds_kernel_spec,
)
from tests.ggtensile.support import (
    MMQ_FWD_INVENTORY_CASES,
    GGTensileInventoryCase,
    load_inventory_case,
    ordinary_forward_instance,
)
from tools.ggtensile.campaign import load_catalog
from tools.ggtensile.family_registry import (
    instance_hash,
    instance_name,
    mapping_for_instance,
    parse_instance,
    writer_for_instance,
)
from tools.ggtensile.mmq_fwd_spec import (
    DerivedForwardState,
    ForwardKernelCandidate,
    ForwardKernelSpec,
    ForwardProblemContract,
    derive_forward_resource_usage,
)
from tools.ggtensile.model import ProblemSize, SchemaError
from tools.ggtensile.physical_resources import HardwareResourceCapacity
from tools.ggtensile.toolchain import Toolchain
from tools.ggtensile.validation import validate_instance

_CONFIG = Path(__file__).resolve().parents[2] / "tools/ggtensile/configs"
_CASES = {case.quant_type: case for case in MMQ_FWD_INVENTORY_CASES}


def _instance(
    quant_type: str,
    size: ProblemSize,
    spec: ForwardKernelSpec,
):
    return ordinary_forward_instance(quant_type, size, spec)


def _representatives() -> tuple[tuple[str, ProblemSize, ForwardKernelSpec], ...]:
    return (
        ("Q3_K", ProblemSize(2048, 2048, 2048), q3_hip_tiled_lds_kernel_spec()),
        (
            "Q3_K",
            ProblemSize(32768, 4096, 2048),
            q3_full_weight_tiled_lds_kernel_spec(decode_ready_frontier=True),
        ),
        (
            "Q4_K",
            ProblemSize(8192, 2048, 512),
            q4_decoded_weight_lds_kernel_spec(
                epilogue_tiles_ahead=1,
                epilogue_dependency_width=4,
                epilogue_priority=0,
                defer_metadata_reads=True,
            ),
        ),
        (
            "Q5_K",
            ProblemSize(8192, 2048, 512),
            q5_decoded_weight_lds_kernel_spec(
                epilogue_tiles_ahead=1,
                epilogue_dependency_width=2,
                epilogue_priority=2,
                accumulator_initialization="VopdPair",
            ),
        ),
        (
            "Q6_K",
            ProblemSize(128, 248320, 2048),
            q6_structured_kernel_spec(128),
        ),
        ("Q8_0", ProblemSize(2048, 1024, 4096), q8_direct_global_kernel_spec()),
        ("Q8_0", ProblemSize(2048, 1024, 4096), q8_register_tiled_kernel_spec()),
        ("Q8_0", ProblemSize(64, 129280, 4096), q8_small_m_tiled_lds_kernel_spec(64)),
    )


@pytest.mark.parametrize("case", MMQ_FWD_INVENTORY_CASES, ids=lambda c: c.quant_type)
def test_forward_catalog_is_typed_and_versionless(case: GGTensileInventoryCase) -> None:
    catalog = load_inventory_case(case)
    assert catalog.problem_type == case.problem_type
    assert len(catalog.entries) == case.entry_count
    assert len(catalog.instances) == case.entry_count
    assert {entry.problem_size.m for entry in catalog.entries} == case.m_values
    assert all(
        isinstance(entry.instance.kernel_spec, ForwardKernelSpec)
        for entry in catalog.entries
    )
    for entry in catalog.entries:
        validate_instance(entry.instance)
    raw = json.loads(case.catalog_path.read_text(encoding="utf-8"))
    assert catalog.to_mapping() == raw
    assert set(raw) == {
        "KernelFamily",
        "Target",
        "ProblemType",
        "KernelSpecs",
        "ExactLogic",
    }
    assert not any(
        field in json.dumps(raw)
        for field in ("SelectedSolution", "HistoricalHipMedianMs", "CurrentStatus")
    )


def test_forward_specs_round_trip() -> None:
    for quant_type, size, spec in _representatives():
        instance = _instance(quant_type, size, spec)
        assert parse_instance(mapping_for_instance(instance)) == instance
        assert ForwardKernelSpec.from_mapping(spec.to_mapping()) == spec
        state = DerivedForwardState.from_problem_spec(size, quant_type, spec)
        assert state.expected_output_shape == (size.m, size.n)
        assert state.expected_activation_shape[1] == size.m
        derive_forward_resource_usage(spec).admit(HardwareResourceCapacity())


def test_forward_candidate_mapping_is_self_contained() -> None:
    for quant_type, _, spec in _representatives():
        candidate = ForwardKernelCandidate(
            ForwardProblemContract.for_quant_type(quant_type), spec
        )
        parsed = ForwardKernelCandidate.from_mapping(candidate.to_mapping())
        assert parsed == candidate
        candidate_mapping = candidate.to_mapping()
    kernel_spec_mapping = candidate_mapping["KernelSpec"]
    assert isinstance(kernel_spec_mapping, dict)
    assert "HardwareResourceCapacity" not in kernel_spec_mapping


def test_forward_validation_rejects_incompatible_geometry() -> None:
    quant_type, size, spec = _representatives()[0]
    mapping = mapping_for_instance(_instance(quant_type, size, spec))
    kernel_spec_mapping = mapping["KernelSpec"]
    assert isinstance(kernel_spec_mapping, dict)
    geometry = kernel_spec_mapping["Geometry"]
    assert isinstance(geometry, dict)
    geometry["DepthU"] = 15
    with pytest.raises((SchemaError, AssertionError)):
        parse_instance(mapping)


def test_forward_writer_source_is_available_for_every_catalog_entry() -> None:
    toolchain = Toolchain.discover()
    for path in sorted(_CONFIG.glob("mmq_fwd_*_catalog.json")):
        catalog = load_catalog(path)
        for entry in catalog.entries:
            source = writer_for_instance(entry.instance, toolchain).source()
            assert instance_name(entry.instance) in source
            assert ".amdhsa_kernel" in source


def test_forward_catalog_specs_have_distinct_quant_identity() -> None:
    instances = [
        entry.instance
        for case in _CASES.values()
        for entry in load_inventory_case(case).entries[:1]
    ]
    assert len({instance.problem_type for instance in instances}) == len(instances)
    assert len({instance_hash(instance) for instance in instances}) == len(instances)
