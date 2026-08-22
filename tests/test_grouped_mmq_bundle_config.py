from pathlib import Path

from tools.ggtensile.campaign import load_catalog
from tools.ggtensile.deployment import load_deployment_inventory
from tools.mmq_deployment_bundle import kernels

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tools/ggtensile/configs"


def test_checked_in_configs_contain_only_selected_kernels() -> None:
    for path in CONFIG.glob("mmq_*_catalog.json"):
        catalog = load_catalog(path)
        assert all(
            any(entry.solution == solution for entry in catalog.entries)
            for solution in catalog.solutions
        )

    inventory = load_deployment_inventory(CONFIG / "mmq_deployment.json")
    assert all(route.kernel is not None for route in inventory.routes)


def test_public_bundle_contains_no_hip_compute_artifacts() -> None:
    bundle = kernels()
    assert [kernel.cpp_id for kernel in bundle if kernel.hip_config is not None] == [
        "QuantizeQ81F32D4",
        "QuantizeQ81F16D4S4",
        "QuantizeQ81F16D2S6",
        "GroupedRowTaskSetup",
    ]
    assert all(kernel.key is not None for kernel in bundle[4:])


def test_dispatch_uses_exact_records_without_legacy_route_selection() -> None:
    source = (ROOT / "csrc/mmq_bundle.cpp").read_text()
    assert "exact_record(" in source
    for forbidden in (
        "kQualified",
        "select_",
        "fallback",
        "heuristic",
        "nearest",
    ):
        assert forbidden not in source
