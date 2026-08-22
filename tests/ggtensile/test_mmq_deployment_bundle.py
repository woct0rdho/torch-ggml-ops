from tools.ggtensile.deployment import DeploymentKey
from tools.mmq_deployment_bundle import kernels
from tools.mmq_deployment_spec import _record, header_text


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


def test_bundle_symbols_are_exact_and_unique() -> None:
    bundle = kernels()
    symbols = [kernel.symbol for kernel in bundle]
    assert len(symbols) == len(set(symbols))
    assert all(
        kernel.key is None or kernel.symbol == kernel.key.kernel_name
        for kernel in bundle
    )
