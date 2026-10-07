"""Typed kernel inventory and generated host records for the MMQ bundle."""

from dataclasses import dataclass
from pathlib import Path

from tools.ggtensile.campaign import load_catalog
from tools.ggtensile.family_registry import (
    family_for_instance,
    instance_name,
    launch_for_instance,
    problem_size_for_instance,
    writer_for_instance,
)
from tools.ggtensile.identity import KernelFamily
from tools.ggtensile.kernel_instance import KernelInstance
from tools.ggtensile.toolchain import Toolchain
from tools.mmq_bundle_wrapper_source import (
    ForwardConfig,
    ForwardKind,
    KernelConfig,
    QuantType,
)
from tools.mmq_hip_deployment import deployment_table, select_hip_control

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tools/ggtensile/configs"

# The public launch resolves one kernel per exact problem: GGTensile whenever
# its catalogs carry the key, the HIP control otherwise. The choice is made
# here, at generation time, and baked into the host table, so a launch is one
# table lookup with no catalog or inventory arbitration at runtime.
IMPLEMENTATION_GGTENSILE = 0
IMPLEMENTATION_HIP = 1


@dataclass(frozen=True)
class BundleKernel:
    cpp_id: str
    symbol: str
    instance: KernelInstance | None
    hip_config: KernelConfig | None = None

    @property
    def operation(self) -> str | None:
        return (
            family_for_instance(self.instance).value
            if self.instance is not None
            else None
        )


def _hip_kernels() -> list[BundleKernel]:
    values = (
        (
            "QuantizeQ81F32D4",
            "quantize_bf16_q8_1_f32_d4",
            QuantType.Q8_0,
        ),
        (
            "QuantizeQ81F16D4S4",
            "quantize_bf16_q8_1_f16_d4s4",
            QuantType.Q4_K,
        ),
        (
            "QuantizeQ81F16D2S6",
            "quantize_bf16_q8_1_f16_d2s6",
            QuantType.Q2_K,
        ),
        (
            "GroupedRowTaskSetup",
            "grouped_row_task_setup",
            None,
        ),
    )
    result = [
        BundleKernel(
            name,
            symbol,
            None,
            ForwardConfig(
                kind=ForwardKind.QUANTIZE if quant else ForwardKind.ROW_TASK_SETUP,
                quant_type=quant,
            ),
        )
        for name, symbol, quant in values
    ]
    # The grouped producer wins while the activation row set is cache resident
    # and loses a few percent past it, so it ships beside the row producer and
    # the dispatch picks between them by size.
    for name, symbol, quant in (
        (
            "QuantizeQ81GroupedF32D4",
            "quantize_bf16_q8_1_f32_d4_grouped",
            QuantType.Q8_0,
        ),
        (
            "QuantizeQ81GroupedF16D4S4",
            "quantize_bf16_q8_1_f16_d4s4_grouped",
            QuantType.Q4_K,
        ),
    ):
        result.append(
            BundleKernel(
                name,
                symbol,
                None,
                ForwardConfig(
                    kind=ForwardKind.QUANTIZE,
                    quant_type=quant,
                    grouped_producer=True,
                ),
            )
        )
    return result


def _gt_catalog_instances() -> tuple[KernelInstance, ...]:
    """Return the GGTensile catalog instances in catalog order."""

    result: list[KernelInstance] = []
    for path in sorted(CONFIG.glob("mmq_*_catalog.json")):
        catalog = load_catalog(path)
        result.extend(entry.instance for entry in catalog.entries)
    return tuple(result)


def _gt_key(instance: KernelInstance) -> tuple[str, str, int, int, int]:
    problem = problem_size_for_instance(instance)
    return (
        family_for_instance(instance).value,
        instance.problem_type.quant_data_type,
        problem.m,
        problem.n,
        problem.k,
    )


def hip_only_keys() -> tuple[tuple[str, str, int, int, int], ...]:
    """Return the deployed problems that only the HIP catalog serves.

    Every key here is an ordinary dense problem. A routed or fixed-group key
    without a GGTensile kernel would mean the two inventories disagree about
    the deployed families, and a split-contraction control needs a slice
    buffer plus a reduction launch that the public signatures do not carry, so
    both are rejected instead of silently dropped.
    """

    covered = {_gt_key(instance) for instance in _gt_catalog_instances()}
    result: list[tuple[str, str, int, int, int]] = []
    for key in sorted(deployment_table()):
        if key in covered:
            continue
        operation, quant_type, m, n, k = key
        if operation not in {"OrdinaryForward", "OrdinaryBackward"}:
            raise ValueError(f"deployed {operation} key {key} has no GGTensile kernel")
        control = select_hip_control(operation, quant_type, m, n, k)
        if getattr(control.spec.config, "split_k", 0):
            continue
        result.append(key)
    return tuple(result)


def kernels() -> tuple[BundleKernel, ...]:
    result = _hip_kernels()
    for instance in _gt_catalog_instances():
        result.append(
            BundleKernel(
                f"Kernel{len(result):03d}",
                instance_name(instance),
                instance,
            )
        )
    # Only the control artifacts the resolution above actually selects enter
    # the public bundle, and each is built once no matter how many keys use it.
    selected: dict[str, BundleKernel] = {}
    for key in hip_only_keys():
        control = select_hip_control(*key)
        if control.symbol in selected:
            continue
        selected[control.symbol] = BundleKernel(
            f"Kernel{len(result) + len(selected):03d}",
            control.symbol,
            None,
            control.spec.config,
        )
    result.extend(selected.values())
    symbols = [item.symbol for item in result]
    if len(symbols) != len(set(symbols)):
        raise ValueError("deployment bundle contains duplicate symbols")
    return tuple(result)


def activation_producer(operation: str, quant_type: str) -> str:
    """Return the Q8_1 producer that feeds one deployed forward problem.

    The producer has to match the weight quant type's blocking: Q2_K reads a
    two-block layer, Q4_K and Q5_K a four-block layer, and every other
    deployed quant type reads the generic four-block layer. A backward problem
    consumes the BF16 gradient and quantizes nothing, so its recorded producer
    documents the family default only.
    """

    if quant_type == "Q2_K":
        return "QuantizeQ81F16D2S6"
    if quant_type in {"Q4_K", "Q5_K"} and operation in {
        "OrdinaryForward",
        "GroupedForward",
    }:
        return "QuantizeQ81F16D4S4"
    return "QuantizeQ81F32D4"


def writer_for(kernel: BundleKernel, toolchain: Toolchain):
    if kernel.instance is None:
        raise TypeError("HIP support artifacts do not have a GGTensile writer")
    return writer_for_instance(kernel.instance, toolchain)


@dataclass(frozen=True)
class DeploymentEntry:
    """One resolved public problem and the kernel the public launch selects."""

    operation: str
    quant_type: str
    m: int
    n: int
    k: int
    kernel_index: int
    implementation: int
    producer_index: int
    grid: tuple[int, int, int]
    work_group: tuple[int, int, int]
    dynamic_shared_bytes: int
    ownership: str
    row_task_rows: int
    row_task_capacity: int
    route_split_factor: int


def deployments(items: tuple[BundleKernel, ...]) -> tuple[DeploymentEntry, ...]:
    """Resolve every public problem to one kernel: GGTensile first, else HIP."""

    indices = {item.symbol: index for index, item in enumerate(items)}
    producers = {item.cpp_id: index for index, item in enumerate(items)}
    result: list[DeploymentEntry] = []
    for instance in _gt_catalog_instances():
        problem = problem_size_for_instance(instance)
        quant_name = instance.problem_type.quant_data_type
        launch = launch_for_instance(instance)
        operation = family_for_instance(instance).value
        result.append(
            DeploymentEntry(
                operation=operation,
                quant_type=quant_name,
                m=problem.m,
                n=problem.n,
                k=problem.k,
                kernel_index=indices[instance_name(instance)],
                implementation=IMPLEMENTATION_GGTENSILE,
                producer_index=producers[activation_producer(operation, quant_name)],
                grid=launch.grid,
                work_group=launch.work_group,
                # A GGTensile artifact declares its LDS in the code object's
                # group segment, so the launch requests no dynamic bytes.
                dynamic_shared_bytes=0,
                ownership=launch.ownership,
                row_task_rows=launch.row_task_rows or 0,
                row_task_capacity=launch.row_task_capacity or 0,
                route_split_factor=launch.route_split_factor,
            )
        )
    for operation, quant_name, m, n, k in hip_only_keys():
        control = select_hip_control(operation, quant_name, m, n, k)
        grid, work_group, shared_memory = control.launch_configuration(m, n, k)
        result.append(
            DeploymentEntry(
                operation=operation,
                quant_type=quant_name,
                m=m,
                n=n,
                k=k,
                kernel_index=indices[control.symbol],
                implementation=IMPLEMENTATION_HIP,
                producer_index=producers[activation_producer(operation, quant_name)],
                grid=grid,
                work_group=work_group,
                dynamic_shared_bytes=shared_memory,
                ownership="Serial",
                row_task_rows=0,
                row_task_capacity=0,
                route_split_factor=1,
            )
        )
    return tuple(result)


def _record(entry: DeploymentEntry) -> tuple[int, ...]:
    return (
        list(KernelFamily).index(KernelFamily[entry.operation]),
        QuantType[entry.quant_type].value,
        entry.m,
        entry.n,
        entry.k,
        entry.kernel_index,
        *entry.grid,
        *entry.work_group,
        entry.dynamic_shared_bytes,
        int(entry.ownership == "DeviceRowTasks"),
        entry.row_task_rows,
        entry.row_task_capacity,
        entry.route_split_factor,
        entry.implementation,
        entry.producer_index,
    )


def header_text(items: tuple[BundleKernel, ...]) -> str:
    symbols = "\n".join(f'    "{item.symbol}",' for item in items)
    records = [_record(entry) for entry in deployments(items)]
    rows = "\n".join("    {" + ", ".join(map(str, record)) + "}," for record in records)
    indices = {item.cpp_id: index for index, item in enumerate(items)}
    operation_constants = "\n".join(
        f"inline constexpr int k{family.value} = {index};"
        for index, family in enumerate(KernelFamily)
    )
    quant_constants = "\n".join(
        f"inline constexpr int kQuant{quant.name} = {quant.value};"
        for quant in QuantType
    )
    return f"""// Generated by tools/mmq_deployment_bundle.py. Do not edit directly.
#pragma once
#include <array>
#include <cstdint>
namespace torch_ggml_ops::mmq_bundle {{
using MMQKernelIndex = std::uint16_t;
inline constexpr int kImplementationGGTensile = {IMPLEMENTATION_GGTENSILE};
inline constexpr int kImplementationHip = {IMPLEMENTATION_HIP};
{operation_constants}
{quant_constants}
inline constexpr MMQKernelIndex kQuantizeQ81F32D4 = {indices["QuantizeQ81F32D4"]};
inline constexpr MMQKernelIndex kQuantizeQ81F16D4S4 = {indices["QuantizeQ81F16D4S4"]};
inline constexpr MMQKernelIndex kQuantizeQ81F16D2S6 = {indices["QuantizeQ81F16D2S6"]};
inline constexpr MMQKernelIndex kGroupedRowTaskSetup = {indices["GroupedRowTaskSetup"]};
inline constexpr std::array<const char *, {len(items)}> kMMQKernelSymbols{{{{
{symbols}
}}}};
inline constexpr const char * mmq_kernel_symbol(MMQKernelIndex index) {{
    return kMMQKernelSymbols[index];
}}
struct MMQDeploymentRecord {{
    int operation;
    int quant_type;
    int m;
    int n;
    int k;
    MMQKernelIndex kernel;
    unsigned int grid_x;
    unsigned int grid_y;
    unsigned int grid_z;
    unsigned int block_x;
    unsigned int block_y;
    unsigned int block_z;
    unsigned int dynamic_shared_bytes;
    int ownership;
    int row_task_rows;
    int row_task_capacity;
    int route_split_factor;
    int implementation;
    MMQKernelIndex producer;
}};
inline constexpr std::array<MMQDeploymentRecord, {len(records)}> kMMQDeployments{{{{
{rows}
}}}};
}} // namespace torch_ggml_ops::mmq_bundle
"""
