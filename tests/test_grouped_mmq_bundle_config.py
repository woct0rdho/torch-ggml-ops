from pathlib import Path

from tools.ggtensile.campaign import load_catalog
from tools.mmq_deployment_bundle import kernels
from tools.mmq_deployment_spec import deployments

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tools/ggtensile/configs"
# Support artifacts ship for every deployment: the Q8_1 activation producers,
# the shared device row-task setup, and the dense split-contraction reduction.
SUPPORT_SYMBOLS = frozenset({"grouped_row_task_setup", "dense_bwd_split_k_reduce"})


def _is_support(kernel) -> bool:
    """Whether one bundle item is a support artifact rather than a compute kernel."""

    return kernel.symbol.startswith("quantize") or kernel.symbol in SUPPORT_SYMBOLS


def _is_hip_control(kernel, support: set[str]) -> bool:
    """Whether one bundle item is a selected HIP compute control."""

    return (
        kernel.instance is None
        and kernel.hip_config is not None
        and kernel.cpp_id not in support
        and not _is_support(kernel)
    )


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
    support = {kernel.cpp_id for kernel in bundle if _is_support(kernel)}
    # Every bundle item is one of three things: a GGTensile kernel, a support
    # artifact, or a HIP compute control the resolution selects. The support
    # artifacts are the quantizers, the shared row-task setup, and the
    # split-contraction reduction.
    assert all(
        kernel.instance is not None
        or _is_support(kernel)
        or _is_hip_control(kernel, support)
        for kernel in bundle
    )
    hip_records = [entry for entry in deployments(bundle) if entry.implementation == 1]
    shipped = {
        index for index, kernel in enumerate(bundle) if _is_hip_control(kernel, support)
    }
    assert hip_records
    assert shipped == {entry.kernel_index for entry in hip_records}
