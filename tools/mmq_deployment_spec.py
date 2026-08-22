"""Typed kernel inventory and generated host records for the MMQ bundle."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tools.ggtensile.campaign import load_catalog
from tools.ggtensile.deployment import (
    DeploymentCandidate,
    DeploymentKey,
    load_deployment_inventory,
)
from tools.ggtensile.fixed_grouped_mmq_bwd_model import FixedBackwardSolutionKey
from tools.ggtensile.fixed_grouped_mmq_fwd_model import FixedForwardSolutionKey
from tools.ggtensile.grouped_mmq_bwd_pair_model import GroupedBackwardPairSolutionKey
from tools.ggtensile.grouped_mmq_fwd_model import GroupedForwardSolutionKey
from tools.ggtensile.grouped_mmq_fwd_pair_model import GroupedForwardPairSolutionKey
from tools.ggtensile.kernel_writer_assembly_fixed_grouped_mmq_bwd import (
    FixedGroupedBackwardKernelWriterAssembly,
)
from tools.ggtensile.kernel_writer_assembly_fixed_grouped_mmq_fwd import (
    FixedGroupedForwardKernelWriterAssembly,
)
from tools.ggtensile.kernel_writer_assembly_grouped_mmq_bwd import (
    GroupedBackwardKernelWriterAssembly,
)
from tools.ggtensile.kernel_writer_assembly_grouped_mmq_bwd_pair import (
    GroupedBackwardPairKernelWriterAssembly,
)
from tools.ggtensile.kernel_writer_assembly_grouped_mmq_fwd import (
    GroupedForwardKernelWriterAssembly,
)
from tools.ggtensile.kernel_writer_assembly_grouped_mmq_fwd_pair import (
    GroupedForwardPairKernelWriterAssembly,
)
from tools.ggtensile.kernel_writer_assembly_mmq_bwd import BackwardKernelWriterAssembly
from tools.ggtensile.kernel_writer_assembly_mmq_fwd import ForwardKernelWriterAssembly
from tools.ggtensile.mmq_fwd_spec import DerivedForwardState
from tools.ggtensile.model import (
    BackwardSolution,
    ForwardSolution,
    GroupedBackwardSolution,
    SolutionKey,
)
from tools.ggtensile.schema import integer
from tools.ggtensile.toolchain import Toolchain
from tools.mmq_bundle_wrapper_source import (
    ForwardConfig,
    ForwardKind,
    QuantType,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tools/ggtensile/configs"


@dataclass(frozen=True)
class BundleKernel:
    cpp_id: str
    symbol: str
    key: DeploymentKey | None
    hip_config: ForwardConfig | None = None
    operation: str | None = None
    candidate: DeploymentCandidate | None = None

    @property
    def filename(self) -> str:
        return f"{self.symbol}.hsaco"


def _hip_kernels() -> list[BundleKernel]:
    values = (
        (
            "QuantizeQ81F32D4",
            "torch_ggml_ops_mmq_gfx1151_v1_quantize_bf16_q8_1_f32_d4",
            QuantType.Q8_0,
        ),
        (
            "QuantizeQ81F16D4S4",
            "torch_ggml_ops_mmq_gfx1151_v1_quantize_bf16_q8_1_f16_d4s4",
            QuantType.Q4_K,
        ),
        (
            "QuantizeQ81F16D2S6",
            "torch_ggml_ops_mmq_gfx1151_v1_quantize_bf16_q8_1_f16_d2s6",
            QuantType.Q2_K,
        ),
        (
            "GroupedRowTaskSetup",
            "torch_ggml_ops_mmq_gfx1151_v1_grouped_row_task_setup",
            None,
        ),
    )
    return [
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


_NON_PUBLIC_FORWARD_LAYOUTS = frozenset(
    {
        ("Q3_K", 4096, 2048),
        ("Q3_K", 2048, 4096),
    }
)


def _is_public_kernel(operation: str, key: DeploymentKey) -> bool:
    if operation == "OrdinaryForward":
        contract = key.to_mapping()["ProblemContract"]
        problem = key.to_mapping()["Problem"]
        if (
            isinstance(contract, Mapping)
            and isinstance(problem, Mapping)
            and (contract.get("quant_type"), problem.get("n"), problem.get("k"))
            in _NON_PUBLIC_FORWARD_LAYOUTS
        ):
            return False
    return True


def kernels() -> tuple[BundleKernel, ...]:
    result = _hip_kernels()
    for path in sorted(CONFIG.glob("mmq_*_catalog.json")):
        catalog = load_catalog(path)
        operation = "OrdinaryForward" if "_fwd_" in path.name else "OrdinaryBackward"
        for entry in catalog.entries:
            key = entry.solution_key
            if not _is_public_kernel(operation, key):
                continue
            result.append(
                BundleKernel(
                    f"Kernel{len(result):03d}",
                    key.kernel_name,
                    key,
                    operation=operation,
                )
            )
    inventory = load_deployment_inventory(CONFIG / "mmq_deployment.json")
    for route in inventory.routes:
        candidate = route.kernel
        if not _is_public_kernel(route.operation, candidate.exact_key):
            continue
        result.append(
            BundleKernel(
                f"Kernel{len(result):03d}",
                candidate.symbol,
                candidate.exact_key,
                operation=route.operation,
                candidate=candidate,
            )
        )
    symbols = [item.symbol for item in result]
    if len(symbols) != len(set(symbols)):
        raise ValueError("deployment bundle contains duplicate symbols")
    return tuple(result)


def writer_for(kernel: BundleKernel, toolchain: Toolchain):
    key = kernel.key
    if isinstance(key, GroupedForwardSolutionKey):
        return GroupedForwardKernelWriterAssembly(key, toolchain)
    if isinstance(key, GroupedForwardPairSolutionKey):
        return GroupedForwardPairKernelWriterAssembly(key, toolchain)
    if isinstance(key, GroupedBackwardPairSolutionKey):
        return GroupedBackwardPairKernelWriterAssembly(key, toolchain)
    if isinstance(key, FixedForwardSolutionKey):
        return FixedGroupedForwardKernelWriterAssembly(key, toolchain)
    if isinstance(key, FixedBackwardSolutionKey):
        return FixedGroupedBackwardKernelWriterAssembly(key, toolchain)
    if isinstance(key, SolutionKey) and isinstance(key.solution, ForwardSolution):
        return ForwardKernelWriterAssembly(key, toolchain)
    if isinstance(key, SolutionKey) and isinstance(key.solution, BackwardSolution):
        return BackwardKernelWriterAssembly(key, toolchain)
    if isinstance(key, SolutionKey):
        return GroupedBackwardKernelWriterAssembly(key, toolchain)
    raise TypeError(f"no GGTensile writer for {type(key).__name__}")


def _quant_value(name: str) -> int:
    return {
        "Q8_0": 8,
        "Q2_K": 10,
        "Q3_K": 11,
        "Q4_K": 12,
        "Q5_K": 13,
        "Q6_K": 14,
        "IQ2_XXS": 16,
        "IQ2_S": 22,
    }[name]


def _record(kernel: BundleKernel, index: int) -> tuple[int, ...] | None:
    key = kernel.key
    if key is None:
        return None
    mapping = key.to_mapping()
    contract = mapping["ProblemContract"]
    problem = mapping["Problem"]
    if not isinstance(contract, Mapping) or not isinstance(problem, Mapping):
        raise TypeError("deployment exact key has invalid problem mappings")
    quant_name = contract.get("quant_type")
    if type(quant_name) is not str:
        raise ValueError("deployment exact key has invalid quant_type")
    quant = _quant_value(quant_name)
    if kernel.operation == "OrdinaryForward":
        if not isinstance(key, SolutionKey):
            raise ValueError("ordinary forward deployment requires SolutionKey")
        operation = 0
        m = integer(problem["m"], "Problem.m")
        n = integer(problem["n"], "Problem.n")
        k = integer(problem["k"], "Problem.k")
        state = DerivedForwardState.from_solution_key(key)
        grid, block, shared = (
            state.grid,
            state.kernel_spec.geometry.work_group,
            state.resources.lds_bytes,
        )
    elif kernel.operation == "OrdinaryBackward":
        operation = 1
        m = integer(problem["m"], "Problem.m")
        n = integer(problem["n"], "Problem.n")
        k = integer(problem["k"], "Problem.k")
        solution = key.solution
        if not isinstance(solution, BackwardSolution):
            raise ValueError("ordinary backward deployment requires BackwardSolution")
        grid = (
            solution.work_group_mapping,
            n // solution.macro_tile1,
            m // solution.macro_tile0 // solution.work_group_mapping,
        )
        block, shared = solution.work_group, 0
    else:
        if kernel.operation is None:
            raise ValueError("deployment kernel lacks an operation")
        operation = {
            "GroupedForward": 2,
            "GroupedForwardPair": 3,
            "GroupedBackward": 4,
            "GroupedBackwardPair": 5,
            "FixedGroupedForward": 6,
            "FixedGroupedBackward": 7,
        }[kernel.operation]
        m_value = problem.get("aggregate_rows", problem.get("m", problem.get("tokens")))
        output_value = contract.get(
            "output_features", contract.get("out_features", problem.get("n"))
        )
        input_value = contract.get(
            "input_features", contract.get("in_features", problem.get("k"))
        )
        m = integer(m_value, "deployment m")
        output_features = integer(output_value, "deployment output features")
        input_features = integer(input_value, "deployment input features")
        if kernel.operation == "GroupedBackward":
            n = integer(problem["n"], "Problem.n")
            k = integer(problem["k"], "Problem.k")
        elif kernel.operation in {"GroupedBackwardPair", "FixedGroupedBackward"}:
            n, k = input_features, output_features
        else:
            n, k = output_features, input_features
        candidate = kernel.candidate
        if candidate is None:
            raise ValueError("deployment kernel lacks candidate metadata")
        grid, block, shared = (
            candidate.grid,
            candidate.work_group,
            candidate.shared_memory_bytes,
        )
    ownership = 0
    row_rows = row_capacity = 0
    if kernel.candidate is not None:
        ownership = 1 if kernel.candidate.ownership.startswith("DeviceRowTasks") else 0
        row_rows = kernel.candidate.row_task_rows or 0
        row_capacity = kernel.candidate.row_task_capacity or 0
    route_split_factor = 1
    if isinstance(key, GroupedBackwardPairSolutionKey):
        route_split_factor = key.solution.route_ownership.split_factor
    elif isinstance(key, SolutionKey) and isinstance(
        key.solution, GroupedBackwardSolution
    ):
        route_ownership = key.solution.route_ownership
        if route_ownership.startswith("SplitRoutes"):
            route_split_factor = int(route_ownership.removeprefix("SplitRoutes"))
    return (
        operation,
        quant,
        m,
        n,
        k,
        index,
        *grid,
        *block,
        shared,
        ownership,
        row_rows,
        row_capacity,
        route_split_factor,
    )


def header_text(items: tuple[BundleKernel, ...]) -> str:
    symbols = "\n".join(f'    "{item.symbol}",' for item in items)
    records = [
        record
        for index, item in enumerate(items)
        if (record := _record(item, index)) is not None
    ]
    rows = "\n".join("    {" + ", ".join(map(str, record)) + "}," for record in records)
    return f"""// Generated by tools/mmq_deployment_bundle.py. Do not edit directly.
#pragma once
#include <array>
#include <cstdint>
namespace torch_ggml_ops::mmq_bundle {{
using MMQKernelIndex = std::uint16_t;
inline constexpr MMQKernelIndex kQuantizeQ81F32D4 = 0;
inline constexpr MMQKernelIndex kQuantizeQ81F16D4S4 = 1;
inline constexpr MMQKernelIndex kQuantizeQ81F16D2S6 = 2;
inline constexpr MMQKernelIndex kGroupedRowTaskSetup = 3;
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
    unsigned int shared_memory;
    int ownership;
    int row_task_rows;
    int row_task_capacity;
    int route_split_factor;
}};
inline constexpr std::array<MMQDeploymentRecord, {len(records)}> kMMQDeployments{{{{
{rows}
}}}};
}} // namespace torch_ggml_ops::mmq_bundle
"""
