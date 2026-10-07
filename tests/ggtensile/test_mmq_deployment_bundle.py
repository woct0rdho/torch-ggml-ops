from tools.ggtensile.family_registry import instance_name
from tools.ggtensile.grouped_mmq_bwd_pair_model import (
    GroupedBackwardPairProblem,
)
from tools.ggtensile.grouped_mmq_fwd_pair_model import (
    GroupedForwardPairProblem,
)
from tools.ggtensile.kernel_instance import KernelInstance
from tools.mmq_deployment_bundle import kernels
from tools.mmq_deployment_spec import deployments, header_text, hip_only_keys
from tools.mmq_deployment_spec import kernels as public_kernels


def _hip_selected_symbols() -> set[str]:
    """Return the HIP control symbols the resolution selects."""

    from tools.mmq_deployment_spec import _routed_hip_deployments
    from tools.mmq_hip_deployment import select_hip_control

    symbols = {entry.symbol for entry in _routed_hip_deployments()}
    for key in hip_only_keys():
        if key[:1] in {("OrdinaryForward",), ("OrdinaryBackward",)}:
            symbols.add(select_hip_control(*key).symbol)
    return symbols


def test_bundle_retains_hip_only_for_quantization_and_task_setup() -> None:
    bundle = kernels()
    assert [kernel.cpp_id for kernel in bundle[:7]] == [
        "QuantizeQ81F32D4",
        "QuantizeQ81F16D4S4",
        "QuantizeQ81F16D2S6",
        "GroupedRowTaskSetup",
        "QuantizeQ81GroupedF32D4",
        "QuantizeQ81GroupedF16D4S4",
        "SplitKReduce",
    ]
    ggtensile = [kernel for kernel in bundle[7:] if kernel.instance is not None]
    assert ggtensile
    assert all(isinstance(kernel.instance, KernelInstance) for kernel in ggtensile)
    # The remaining artifacts are exactly the HIP controls the resolution
    # selects, one artifact per control symbol.
    support = {kernel.cpp_id for kernel in bundle[:7]}
    symbols = {
        kernel.symbol
        for kernel in bundle
        if kernel.instance is None and kernel.cpp_id not in support
    }
    assert symbols == _hip_selected_symbols()
    assert hip_only_keys()


def test_paired_backward_split_factor_is_preserved_in_host_records() -> None:
    bundle = kernels()
    entries = [
        entry
        for entry in deployments(bundle)
        if entry.operation == "GroupedBackwardPair" and entry.route_split_factor == 8
    ]
    assert entries
    assert all(entry.grid == (32, 2048, 1) for entry in entries)
    header = header_text(bundle)
    assert "int route_split_factor;" in header
    assert "kMMQKernelFilenames" not in header
    assert "kQuantQ4_K = 12" in header


def test_public_bundle_operation_counts() -> None:
    bundle = public_kernels()
    operations = [
        kernel.operation
        for kernel in bundle
        if kernel.instance is not None and kernel.operation is not None
    ]
    assert {
        operation: operations.count(operation) for operation in sorted(set(operations))
    } == {
        "OrdinaryForward": 50,
        "OrdinaryBackward": 50,
        "GroupedForward": 12,
        "GroupedBackward": 12,
        "GroupedForwardPair": 9,
        "GroupedBackwardPair": 9,
        "FixedGroupedForward": 3,
        "FixedGroupedBackward": 3,
    }


def test_public_pair_inventory_keeps_promoted_iq2_xxs_forward_route() -> None:
    bundle = public_kernels()
    forward_rows = [
        kernel.instance.problem.aggregate_rows
        for kernel in bundle
        if kernel.operation == "GroupedForwardPair"
        and kernel.instance is not None
        and isinstance(kernel.instance.problem, GroupedForwardPairProblem)
        and kernel.instance.problem.quant_data_type == "IQ2_XXS"
    ]
    backward_rows = [
        kernel.instance.problem.aggregate_rows
        for kernel in bundle
        if kernel.operation == "GroupedBackwardPair"
        and kernel.instance is not None
        and isinstance(kernel.instance.problem, GroupedBackwardPairProblem)
        and kernel.instance.problem.quant_data_type == "IQ2_XXS"
    ]
    assert sorted(forward_rows) == [12288, 49152, 196608]
    assert sorted(backward_rows) == [12288, 49152, 196608]


def test_bundle_symbols_are_exact_and_unique() -> None:
    bundle = kernels()
    symbols = [kernel.symbol for kernel in bundle]
    assert len(symbols) == len(set(symbols))
    assert all(
        kernel.instance is None or kernel.symbol == instance_name(kernel.instance)
        for kernel in bundle
    )
