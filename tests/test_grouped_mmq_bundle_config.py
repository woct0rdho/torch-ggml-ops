from pathlib import Path

from tools.ggtensile.campaign import load_catalog
from tools.mmq_deployment_bundle import kernels

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tools/ggtensile/configs"


def test_checked_in_configs_contain_only_selected_kernels() -> None:
    for path in CONFIG.glob("mmq_*_catalog.json"):
        catalog = load_catalog(path)
        assert catalog.entries
        assert len(catalog.entries) == len(
            {entry.problem_size for entry in catalog.entries}
        )


def test_public_bundle_contains_no_hip_compute_artifacts() -> None:
    bundle = kernels()
    assert [kernel.cpp_id for kernel in bundle if kernel.hip_config is not None] == [
        "QuantizeQ81F32D4",
        "QuantizeQ81F16D4S4",
        "QuantizeQ81F16D2S6",
        "GroupedRowTaskSetup",
    ]
    assert all(kernel.instance is not None for kernel in bundle[4:])
