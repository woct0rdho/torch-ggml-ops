from pathlib import Path

from tools.ggtensile.campaign import load_catalog
from tools.mmq_deployment_bundle import kernels
from tools.mmq_deployment_spec import deployments

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tools/ggtensile/configs"


def test_checked_in_configs_contain_only_selected_kernels() -> None:
    for path in CONFIG.glob("mmq_*_catalog.json"):
        catalog = load_catalog(path)
        assert catalog.entries
        assert len(catalog.entries) == len(
            {entry.problem_size for entry in catalog.entries}
        )


def test_public_bundle_ships_only_the_selected_hip_compute_artifacts() -> None:
    """Every shipped HIP artifact is selected, and every HIP record has one.

    The public launch resolves GGTensile first and the HIP deployment table
    second, so the bundle carries the support artifacts plus exactly the HIP
    controls that resolution picks.
    """

    bundle = kernels()
    support = [
        kernel.cpp_id
        for kernel in bundle
        if kernel.hip_config is not None
        and kernel.symbol.startswith("quantize")
        or kernel.hip_config is not None
        and kernel.symbol == "grouped_row_task_setup"
    ]
    assert support == [
        "QuantizeQ81F32D4",
        "QuantizeQ81F16D4S4",
        "QuantizeQ81F16D2S6",
        "GroupedRowTaskSetup",
        # The grouped producers are the same quantization entry points with a
        # thread-per-half-group body. The dispatch picks between them by size.
        "QuantizeQ81GroupedF32D4",
        "QuantizeQ81GroupedF16D4S4",
    ]
    assert all(
        kernel.instance is not None
        for kernel in bundle[6:]
        if kernel.hip_config is None
    )
    hip_records = [entry for entry in deployments(bundle) if entry.implementation == 1]
    shipped = {
        index
        for index, kernel in enumerate(bundle)
        if kernel.instance is None and kernel.cpp_id not in support
    }
    assert shipped == {entry.kernel_index for entry in hip_records}
    assert len(hip_records) == 204
