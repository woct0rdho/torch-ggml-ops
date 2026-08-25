import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.ggtensile.support import (
    MMQ_BWD_INVENTORY_CASES,
    GGTensileInventoryCase,
    load_inventory_case,
    ordinary_backward_instance,
)
from tools.ggtensile.campaign import load_catalog
from tools.ggtensile.family_registry import (
    instance_name,
    mapping_for_instance,
    parse_instance,
    writer_for_instance,
)
from tools.ggtensile.kernel_instance import KernelInstance
from tools.ggtensile.mmq_bwd_search import candidate_domains, candidate_neighbors
from tools.ggtensile.mmq_bwd_spec import (
    BackwardDecodeSpec,
    BackwardExtraction,
    BackwardKernelSpec,
    DerivedBackwardState,
)
from tools.ggtensile.model import ProblemSize, SchemaError
from tools.ggtensile.physical_resources import HardwareResourceCapacity
from tools.ggtensile.toolchain import Toolchain
from tools.ggtensile.validation import validate_backward_solution, validate_instance

_CONFIG = Path(__file__).resolve().parents[2] / "tools/ggtensile/configs"
_CASES = {case.quant_type: case for case in MMQ_BWD_INVENTORY_CASES}


def _instance(
    quant_type: str, size: ProblemSize, spec: BackwardKernelSpec
) -> KernelInstance:
    return ordinary_backward_instance(quant_type, size, spec)


def _selected(quant_type: str, size: ProblemSize) -> KernelInstance:
    return load_inventory_case(_CASES[quant_type]).entry_for(size).instance


@pytest.mark.parametrize("case", MMQ_BWD_INVENTORY_CASES, ids=lambda c: c.quant_type)
def test_backward_catalog_is_typed_and_versionless(
    case: GGTensileInventoryCase,
) -> None:
    catalog = load_inventory_case(case)
    assert catalog.problem_type == case.problem_type
    assert len(catalog.entries) == case.entry_count
    assert all(
        isinstance(entry.instance.kernel_spec, BackwardKernelSpec)
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


def test_backward_catalog_keys_round_trip_exactly() -> None:
    for case in MMQ_BWD_INVENTORY_CASES:
        catalog = load_inventory_case(case)
        for entry in catalog.entries:
            instance = entry.instance
            assert isinstance(instance.kernel_spec, BackwardKernelSpec)
            assert parse_instance(mapping_for_instance(instance)) == instance
            assert (
                BackwardKernelSpec.from_mapping(
                    instance.kernel_spec.to_mapping(case.quant_type), case.quant_type
                )
                == instance.kernel_spec
            )


def test_backward_decode_policy_is_explicit() -> None:
    q3 = _selected("Q3_K", ProblemSize(2048, 2048, 512)).kernel_spec
    assert isinstance(q3, BackwardKernelSpec)
    assert q3.decode.extraction is BackwardExtraction.Packed
    assert q3.decode.pairing is not None
    q4 = _selected("Q4_K", ProblemSize(2048, 2048, 512)).kernel_spec
    assert isinstance(q4, BackwardKernelSpec)
    assert isinstance(q4.decode, BackwardDecodeSpec)
    assert q4.decode.extraction is None
    with pytest.raises(AssertionError):
        replace_decode = replace(
            q4,
            decode=BackwardDecodeSpec(16, extraction=BackwardExtraction.Packed),
        )
        validate_backward_solution(ProblemSize(2048, 2048, 512), "Q4_K", replace_decode)


def test_backward_pipeline_rejects_unimplemented_packed_weight_mode() -> None:
    instance = _selected("Q4_K", ProblemSize(2048, 2048, 512))
    assert isinstance(instance.problem, ProblemSize)
    assert isinstance(instance.kernel_spec, BackwardKernelSpec)
    invalid = replace(
        instance.kernel_spec,
        pipeline=replace(instance.kernel_spec.pipeline, prefetch_packed_weight=False),
    )
    with pytest.raises(AssertionError):
        validate_backward_solution(instance.problem, "Q4_K", invalid)


def test_backward_search_returns_only_typed_valid_candidates() -> None:
    shape = ProblemSize(128, 2048, 512)
    for quant_type in ("Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"):
        domains = candidate_domains(quant_type, shape)
        assert len(domains) == 1
        domain = domains[0]
        assert isinstance(domain.seed, BackwardKernelSpec)
        candidates = candidate_neighbors(
            domain.seed, quant_type, shape, domain.knob_groups
        )
        assert candidates
        assert all(
            isinstance(candidate, BackwardKernelSpec) for candidate in candidates
        )
        assert len(
            {candidate.to_mapping(quant_type).__repr__() for candidate in candidates}
        ) == len(candidates)


def test_backward_writer_source_is_available_for_every_catalog_entry() -> None:
    toolchain = Toolchain.discover()
    for path in sorted(_CONFIG.glob("mmq_bwd_*_catalog.json")):
        catalog = load_catalog(path)
        for entry in catalog.entries:
            source = writer_for_instance(entry.instance, toolchain).source()
            assert instance_name(entry.instance) in source
            assert ".amdhsa_kernel" in source


def test_backward_validation_rejects_malformed_mapping() -> None:
    instance = _selected("Q4_K", ProblemSize(2048, 2048, 512))
    mapping = mapping_for_instance(instance)
    kernel_spec = mapping["KernelSpec"]
    assert isinstance(kernel_spec, dict)
    geometry = kernel_spec["Geometry"]
    assert isinstance(geometry, dict)
    geometry["DepthU"] = 31
    with pytest.raises((SchemaError, AssertionError)):
        parse_instance(mapping)


def test_backward_seed_resources_are_admitted() -> None:
    shape = ProblemSize(128, 256, 128)
    for quant_type in ("Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"):
        domain = candidate_domains(quant_type, shape)[0]
        instance = _instance(quant_type, shape, domain.seed)
        state = DerivedBackwardState.from_problem_spec(shape, quant_type, domain.seed)
        state.spec.validate(state.contract)
        writer_for_instance(instance, Toolchain.discover()).source()
    assert HardwareResourceCapacity().max_vgprs == 256
