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
    DenseBackwardConfig,
    ForwardConfig,
    ForwardKind,
    KernelConfig,
    QuantType,
    SplitKReduceConfig,
)
from tools.mmq_hip_deployment import (
    control_inventory,
    deployment_table,
    routed_table,
    select_hip_control,
    select_routed_control,
    split_k_chunk,
)
from tools.mmq_hip_grouped_bwd import InstalledGroupedBackwardRowTaskControl
from tools.mmq_hip_grouped_pair_bwd import InstalledGroupedBackwardPairRowTaskControl
from torch_ggml_ops.runtime_contract import paired_row_task_capacity

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
    result.append(
        BundleKernel(
            "SplitKReduce",
            "dense_bwd_split_k_reduce",
            None,
            SplitKReduceConfig(),
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
    routed = {
        (entry.operation, entry.quant_type, entry.m, entry.n, entry.k): entry
        for entry in _routed_hip_deployments()
    }
    for key in hip_only_keys():
        routed_entry = routed.get(key)
        if routed_entry is not None:
            symbol = routed_entry.symbol
            config = control_inventory()[symbol].config
        else:
            control = select_hip_control(*key)
            symbol = control.symbol
            config = control.spec.config
        if symbol in selected:
            continue
        selected[symbol] = BundleKernel(
            f"Kernel{len(result) + len(selected):03d}",
            symbol,
            None,
            config,
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
    split_slices: int = 0
    reduce_kernel: int = 0
    split_chunk: int = 0


@dataclass(frozen=True)
class HipDeployment:
    """One HIP-only deployed problem with the geometry its record carries."""

    operation: str
    quant_type: str
    m: int
    n: int
    k: int
    symbol: str
    grid: tuple[int, int, int]
    work_group: tuple[int, int, int]
    dynamic_shared_bytes: int
    ownership: str
    row_task_rows: int
    row_task_capacity: int
    split_slices: int = 0


def _routed_hip_deployments() -> tuple[HipDeployment, ...]:
    """Materialize one record per declared routed row count.

    A routed family declares the aggregate row counts it is deployed for (the
    physical batches of its model), and the ordered rules pick the control for
    each of them. The device grid's Y dimension is the route or task count,
    which every launch replaces with its own bank's, so the record stores zero
    there and the launch supplies the solved value.
    """

    inventory = control_inventory()
    result: list[HipDeployment] = []
    emitted: set[tuple[str, str, int, int]] = set()
    for control in routed_table():
        family = (
            control.operation,
            control.quant_type,
            control.out_features,
            control.in_features,
        )
        if family in emitted:
            continue
        emitted.add(family)
        if not control.rows:
            continue
        for rows in control.rows:
            selected = select_routed_control(
                control.operation,
                control.quant_type,
                control.out_features,
                control.in_features,
                rows,
                control.experts,
            )
            spec = inventory[selected.symbol]
            row_task_rows = 0
            row_task_capacity = 0
            if selected.operation in {"GroupedForward", "GroupedForwardPair"}:
                if (
                    selected.operation == "GroupedForwardPair"
                    and not selected.pair_single
                ):
                    raise ValueError(
                        f"routed paired forward {selected.symbol!r} must declare pair_single"
                    )
                from tools.mmq_hip_deployment import GroupedForwardControl

                grid, work_group, shared = GroupedForwardControl(
                    selected.symbol, spec
                ).launch_configuration(selected.out_features, 1)
                ownership = (
                    "PairSingle"
                    if selected.operation == "GroupedForwardPair"
                    else "Serial"
                )
            elif selected.operation == "GroupedBackward":
                grid, work_group, shared, row_task_rows = (
                    InstalledGroupedBackwardRowTaskControl.record_geometry(
                        selected.symbol,
                        selected.out_features,
                        selected.in_features,
                    )
                )
                row_task_capacity = paired_row_task_capacity(
                    rows, selected.experts, row_task_rows
                )
                ownership = "DeviceRowTasks"
            elif selected.operation == "GroupedBackwardPair":
                grid, work_group, shared, row_task_rows = (
                    InstalledGroupedBackwardPairRowTaskControl.record_geometry(
                        selected.symbol,
                        selected.out_features,
                        selected.in_features,
                    )
                )
                row_task_capacity = paired_row_task_capacity(
                    rows, selected.experts, row_task_rows
                )
                ownership = "DeviceRowTasks"
            else:
                raise ValueError(
                    f"routed operation {selected.operation!r} has no record geometry"
                )
            result.append(
                HipDeployment(
                    operation=selected.operation,
                    quant_type=selected.quant_type,
                    m=rows,
                    n=selected.out_features,
                    k=selected.in_features,
                    symbol=selected.symbol,
                    grid=(grid[0], 0, grid[2]),
                    work_group=work_group,
                    dynamic_shared_bytes=shared,
                    ownership=ownership,
                    row_task_rows=row_task_rows,
                    row_task_capacity=row_task_capacity,
                )
            )
    return tuple(result)


def hip_only_keys() -> tuple[tuple[str, str, int, int, int], ...]:
    """Return the deployed problems that only the HIP catalog serves.

    These are the ordinary dense problems with no GGTensile key, including the
    split-contraction controls the public launch drives with a Python-owned
    partial buffer and a reduction, and the routed families that declare their
    aggregate row counts in the catalog. A routed key is admitted only through
    a declared family. An undeclared one still fails closed.
    """

    covered = {_gt_key(instance) for instance in _gt_catalog_instances()}
    result: list[tuple[str, str, int, int, int]] = []
    for key in sorted(deployment_table()):
        if key in covered:
            continue
        operation = key[0]
        if operation not in {"OrdinaryForward", "OrdinaryBackward"}:
            raise ValueError(f"deployed {operation} key {key} has no GGTensile kernel")
        result.append(key)
    for entry in _routed_hip_deployments():
        key = (entry.operation, entry.quant_type, entry.m, entry.n, entry.k)
        if key in covered:
            continue
        result.append(key)
    return tuple(result)


def deployments(items: tuple[BundleKernel, ...]) -> tuple[DeploymentEntry, ...]:
    """Resolve every public problem to one kernel: GGTensile first, else HIP."""

    indices = {item.symbol: index for index, item in enumerate(items)}
    producers = {item.cpp_id: index for index, item in enumerate(items)}
    reduce_index = producers["SplitKReduce"]
    inventory = control_inventory()
    routed = {
        (entry.operation, entry.quant_type, entry.m, entry.n, entry.k): entry
        for entry in _routed_hip_deployments()
    }
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
        key = (operation, quant_name, m, n, k)
        routed_entry = routed.get(key)
        if routed_entry is not None:
            result.append(
                DeploymentEntry(
                    operation=operation,
                    quant_type=quant_name,
                    m=m,
                    n=n,
                    k=k,
                    kernel_index=indices[routed_entry.symbol],
                    implementation=IMPLEMENTATION_HIP,
                    producer_index=producers[
                        activation_producer(operation, quant_name)
                    ],
                    grid=routed_entry.grid,
                    work_group=routed_entry.work_group,
                    dynamic_shared_bytes=routed_entry.dynamic_shared_bytes,
                    ownership=routed_entry.ownership,
                    row_task_rows=routed_entry.row_task_rows,
                    row_task_capacity=routed_entry.row_task_capacity,
                    route_split_factor=1,
                )
            )
            continue
        control = select_hip_control(operation, quant_name, m, n, k)
        grid, work_group, shared_memory = control.launch_configuration(m, n, k)
        config = inventory[control.symbol].config
        split_slices = (
            int(config.split_k) if isinstance(config, DenseBackwardConfig) else 0
        )
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
                split_slices=split_slices,
                reduce_kernel=reduce_index if split_slices else 0,
                split_chunk=split_k_chunk(
                    int(config.exact_out_features),
                    split_slices,
                    config.k_iteration,
                )
                if isinstance(config, DenseBackwardConfig) and split_slices
                else 0,
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
        int(entry.ownership == "DeviceRowTasks")
        + (2 if entry.ownership == "PairSingle" else 0),
        entry.row_task_rows,
        entry.row_task_capacity,
        entry.route_split_factor,
        entry.implementation,
        entry.producer_index,
        entry.split_slices,
        entry.reduce_kernel,
        entry.split_chunk,
    )


def python_contract_text(items: tuple[BundleKernel, ...]) -> str:
    """Return the generated Python counterpart of the host record table.

    The Python layer has to size its own partial and task buffers before it
    calls a launch, and it must agree with the record the native side resolves.
    Both come from this one resolution, so the generated module cannot drift
    from the C++ table.
    """

    row_tasks: dict[tuple[str, int, int, int, int], int] = {}
    split_slices: dict[tuple[str, int, int, int, int], int] = {}
    for entry in deployments(items):
        key = (
            entry.operation,
            QuantType[entry.quant_type].value,
            entry.m,
            entry.n,
            entry.k,
        )
        if entry.row_task_rows:
            row_tasks[key] = entry.row_task_rows
        if entry.split_slices:
            split_slices[key] = entry.split_slices

    def literal(key: tuple[object, ...]) -> str:
        parts = ", ".join(
            f'"{part}"' if isinstance(part, str) else str(part) for part in key
        )
        return f"({parts})"

    row_lines = "\n".join(
        f"    {literal(key)}: {value}," for key, value in sorted(row_tasks.items())
    )
    split_lines = "\n".join(
        f"    {literal(key)}: {value}," for key, value in sorted(split_slices.items())
    )
    return f'''"""Generated by tools/mmq_deployment_bundle.py. Do not edit directly.

Device row-task tiles and split-contraction slice counts of the exact public
records, keyed by `(operation, quant_type, rows, out_features, in_features)`.
A missing key means the record runs as one launch over the route bank.
"""

# KernelFamily member names, spelled exactly as the generated host table.
ROW_TASK_TILES: dict[tuple[str, int, int, int, int], int] = {{
{row_lines}
}}

SPLIT_SLICES: dict[tuple[str, int, int, int, int], int] = {{
{split_lines}
}}
'''


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
inline constexpr MMQKernelIndex kDenseBackwardSplitKReduce = {indices["SplitKReduce"]};
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
    int split_slices;
    MMQKernelIndex reduce_kernel;
    int split_chunk;
}};
inline constexpr std::array<MMQDeploymentRecord, {len(records)}> kMMQDeployments{{{{
{rows}
}}}};
}} // namespace torch_ggml_ops::mmq_bundle
"""
