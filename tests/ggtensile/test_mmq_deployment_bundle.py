from tools.ggtensile.deployment import DeploymentKey
from tools.ggtensile.grouped_mmq_bwd_pair_model import (
    GroupedBackwardPairSolutionKey,
)
from tools.ggtensile.grouped_mmq_fwd_pair_model import (
    GroupedForwardPairSolutionKey,
)
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
    assert all(isinstance(kernel.key, DeploymentKey) for kernel in bundle[4:])


def test_paired_backward_split_factor_is_preserved_in_host_records() -> None:
    bundle = kernels()
    records = [
        _record(kernel, index)
        for index, kernel in enumerate(bundle)
        if kernel.operation == "GroupedBackwardPair"
        and kernel.candidate is not None
        and kernel.candidate.ownership == "PackedSplitRoutes8"
    ]
    assert len(records) == 3
    assert all(record is not None for record in records)
    assert all(record[6:9] == (32, 2048, 1) for record in records if record)
    assert all(record[-1] == 8 for record in records if record)
    assert "int route_split_factor;" in header_text(bundle)


def test_public_bundle_operation_counts() -> None:
    bundle = public_kernels()
    operations = [
        kernel.operation
        for kernel in bundle
        if kernel.key is not None and kernel.operation is not None
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
        kernel.key.problem.aggregate_rows
        for kernel in bundle
        if kernel.operation == "GroupedForwardPair"
        and isinstance(kernel.key, GroupedForwardPairSolutionKey)
        and kernel.key.problem.quant_data_type == "IQ2_XXS"
    ]
    backward_rows = [
        kernel.key.problem.aggregate_rows
        for kernel in bundle
        if kernel.operation == "GroupedBackwardPair"
        and isinstance(kernel.key, GroupedBackwardPairSolutionKey)
        and kernel.key.problem.quant_data_type == "IQ2_XXS"
    ]
    assert sorted(forward_rows) == [12288, 49152, 196608]
    assert sorted(backward_rows) == [12288, 49152, 196608]


def test_bundle_symbols_are_exact_and_unique() -> None:
    bundle = kernels()
    symbols = [kernel.symbol for kernel in bundle]
    assert len(symbols) == len(set(symbols))
    assert all(
        kernel.key is None or kernel.symbol == kernel.key.kernel_name
        for kernel in bundle
    )
