from tools.ggtensile.family_registry import instance_name
from tools.ggtensile.grouped_mmq_bwd_pair_model import (
    GroupedBackwardPairProblem,
)
from tools.ggtensile.grouped_mmq_bwd_pair_spec import (
    GroupedBackwardPairKernelSpec,
)
from tools.ggtensile.grouped_mmq_fwd_pair_model import (
    GroupedForwardPairProblem,
)
from tools.ggtensile.kernel_instance import KernelInstance
from tools.mmq_deployment_bundle import kernels
from tools.mmq_deployment_spec import _record, header_text
from tools.mmq_deployment_spec import kernels as public_kernels


def test_bundle_retains_hip_only_for_quantization_and_task_setup() -> None:
    bundle = kernels()
    hip_kernels = [kernel for kernel in bundle if kernel.hip_config is not None]
    assert [kernel.cpp_id for kernel in hip_kernels] == [
        "QuantizeQ81F32D4",
        "QuantizeQ81F16D4S4",
        "QuantizeQ81F16D2S6",
        "GroupedRowTaskSetup",
    ]
    assert all(isinstance(kernel.instance, KernelInstance) for kernel in bundle[4:])


def test_paired_backward_split_factor_is_preserved_in_host_records() -> None:
    bundle = kernels()
    records = [
        _record(item, index)
        for index, item in enumerate(bundle)
        if item.operation == "GroupedBackwardPair"
        and item.instance is not None
        and isinstance(item.instance.kernel_spec, GroupedBackwardPairKernelSpec)
        and item.instance.kernel_spec.route_ownership.split_factor == 8
    ]
    assert len(records) == 3
    assert all(record is not None for record in records)
    assert all(record[6:9] == (32, 2048, 1) for record in records if record)
    assert all(record[-1] == 8 for record in records if record)
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
