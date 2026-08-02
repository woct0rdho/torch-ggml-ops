#!/usr/bin/env python3
"""Benchmark production routed and fixed-group MMQ input gradients.

For each routed expert group, a forward projection is
``Y[M,N] = X[M,K] @ W[N,K].T`` and the frozen-base input gradient is
``dX[M,K] = dY[M,N] @ W[N,K]``. The packed path times the public grouped
input-gradient operators on real GGUF expert weights. Paired gate/up backward
uses one fused kernel that accumulates both logical Jacobians into one FP32
accumulator. Routed BF16 references use AITER GMM with the benchmark-owned
production gfx1151 heuristic. Fixed DeepSeek output-A uses the dedicated
eight-group operator and a BF16 strided-batched GEMM reference with the public
layout conversion included.
"""

from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

import gguf
import torch
from aiter.ops.triton.gmm import gmm
from aiter_gmm_heuristics import gmm_config as aiter_gmm_config
from grouped_mmq_benchmark_common import (
    GroupedMMQCase,
    GroupSummary,
    RouteDistribution,
    RoutedWeights,
    bf16_fixed_grad_input_reference,
    fixed_group_distribution,
    grouped_result_metadata,
    load_fixed_weight,
    load_routed_weights,
    make_route_tensors,
    parse_grouped_benchmark_args,
    route_distributions,
    select_cases,
    truncate_distribution,
)
from mmq_benchmark_common import (
    BenchmarkTiming,
    ErrorMetrics,
    benchmark_callable,
    clear_cuda_cache,
    cuda_device_info,
    error_metrics,
    load_gguf_tensors,
    make_bf16_input,
    performance_comparison,
    print_benchmark_header,
    write_json_report,
)
from typing_extensions import NotRequired

import torch_ggml_ops  # noqa: F401 Register native operators before torch.ops use.

DEFAULT_OUTPUT = Path("/tmp/torch_ggml_ops_grouped_mmq_bwd_benchmark.json")
DENSE_PACKED_REFERENCE_QUANT_TYPES = frozenset({"Q3_K", "Q4_K", "Q5_K", "Q6_K"})


class BackwardCorrectness(TypedDict):
    rows: int
    dense_reference_kind: NotRequired[str]
    dense_reference: NotRequired[ErrorMetrics]
    bf16_reference: ErrorMetrics


def aiter_grouped_pair(
    grad_outputs: tuple[torch.Tensor, ...],
    selected_logical_weights: tuple[torch.Tensor, ...],
    group_sizes: torch.Tensor,
    config: dict[str, int],
) -> torch.Tensor:
    first = gmm(
        grad_outputs[0],
        selected_logical_weights[0],
        group_sizes,
        preferred_element_type=grad_outputs[0].dtype,
        config=config,
    )
    second = gmm(
        grad_outputs[1],
        selected_logical_weights[1],
        group_sizes,
        preferred_element_type=grad_outputs[1].dtype,
        config=config,
    )
    return torch.add(first, second)


def transient_bf16_materialization_floor(
    grad_outputs: tuple[torch.Tensor, ...],
    selected_logical_weights: tuple[torch.Tensor, ...],
    group_sizes: torch.Tensor,
    config: dict[str, int],
) -> torch.Tensor:
    materialized = tuple(
        torch.empty_like(weight).zero_() for weight in selected_logical_weights
    )
    if len(materialized) == 1:
        return gmm(
            grad_outputs[0],
            materialized[0],
            group_sizes,
            preferred_element_type=grad_outputs[0].dtype,
            config=config,
        )
    if len(materialized) == 2:
        return aiter_grouped_pair(grad_outputs, materialized, group_sizes, config)
    raise ValueError("transient BF16 control supports one or two projections")


def dense_grouped_grad_input_reference(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    distribution: RouteDistribution,
    quant_type: int,
    in_features: int,
) -> torch.Tensor:
    outputs = []
    row_begin = 0
    for expert, size in zip(
        distribution.expert_indices_cpu, distribution.group_sizes_cpu, strict=True
    ):
        row_end = row_begin + size
        outputs.append(
            torch.ops.torch_ggml_ops.mmq_grad_input.default(
                grad_output[row_begin:row_end].clone(),
                packed_weight[expert].clone(),
                quant_type,
                in_features,
            )
        )
        row_begin = row_end
    return torch.cat(outputs, dim=0)


def bf16_grouped_grad_input_reference(
    grad_output: torch.Tensor,
    selected_logical_weight: torch.Tensor,
    distribution: RouteDistribution,
) -> torch.Tensor:
    outputs = []
    row_begin = 0
    for group, size in enumerate(distribution.group_sizes_cpu):
        row_end = row_begin + size
        outputs.append(grad_output[row_begin:row_end] @ selected_logical_weight[group])
        row_begin = row_end
    return torch.cat(outputs, dim=0)


def dense_correctness_reference(
    case: GroupedMMQCase,
    grad_outputs: tuple[torch.Tensor, ...],
    weights: RoutedWeights,
    selected_logical: tuple[torch.Tensor, ...],
    distribution: RouteDistribution,
) -> tuple[torch.Tensor, str]:
    dense_packed_supported = case.quant_type in DENSE_PACKED_REFERENCE_QUANT_TYPES
    dense_parts = tuple(
        dense_grouped_grad_input_reference(
            grad_output,
            packed_weight,
            distribution,
            weights.quant_type,
            case.in_features,
        )
        if dense_packed_supported
        else bf16_grouped_grad_input_reference(
            grad_output,
            selected_weight,
            distribution,
        )
        for grad_output, packed_weight, selected_weight in zip(
            grad_outputs,
            weights.packed,
            selected_logical,
            strict=True,
        )
    )
    reference = (
        torch.add(dense_parts[0], dense_parts[1])
        if case.kind == "pair"
        else dense_parts[0]
    )
    reference_kind = (
        "packed dense MMQ backward"
        if dense_packed_supported
        else "independently dequantized per-expert BF16 matmul"
    )
    return reference, reference_kind


def correctness_metrics(
    case: GroupedMMQCase,
    grad_outputs: tuple[torch.Tensor, ...],
    weights: RoutedWeights,
    distribution: RouteDistribution,
    aiter_config: dict[str, int],
) -> BackwardCorrectness:
    checked_distribution = truncate_distribution(distribution, grad_outputs[0].shape[0])
    expert_indices, expert_offsets, group_sizes = make_route_tensors(
        checked_distribution
    )
    selected_logical = tuple(
        weight.index_select(0, expert_indices).contiguous()
        for weight in weights.logical
    )
    packed_function, reference_function = make_routed_functions(
        case,
        grad_outputs,
        weights,
        expert_indices,
        expert_offsets,
        selected_logical,
        group_sizes,
        aiter_config,
    )
    with torch.inference_mode():
        actual = packed_function()
        dense_reference, dense_reference_kind = dense_correctness_reference(
            case,
            grad_outputs,
            weights,
            selected_logical,
            checked_distribution,
        )
        bf16_reference = reference_function()
    return {
        "rows": grad_outputs[0].shape[0],
        "dense_reference_kind": dense_reference_kind,
        "dense_reference": error_metrics(actual, dense_reference),
        "bf16_reference": error_metrics(actual, bf16_reference),
    }


def fixed_correctness_metrics(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    logical_weight: torch.Tensor,
) -> BackwardCorrectness:
    with torch.inference_mode():
        actual = torch.ops.torch_ggml_ops.fixed_grouped_mmq_grad_input.default(
            grad_output,
            packed_weight,
        )
        expected = bf16_fixed_grad_input_reference(grad_output, logical_weight)
    return {
        "rows": grad_output.shape[0],
        "bf16_reference": error_metrics(actual, expected),
    }


def print_result(result: dict[str, object]) -> None:
    packed = cast(BenchmarkTiming, result["packed"])
    reference = cast(BenchmarkTiming, result["bf16_reference"])
    correctness = cast(BackwardCorrectness, result["correctness"])
    group_summary = cast(GroupSummary, result["group_summary"])
    reference_label = "BMM" if result["kind"] == "fixed" else "AITER"
    row_label = "M" if result["kind"] == "fixed" else "R"
    dense_metrics = correctness.get("dense_reference", correctness["bf16_reference"])
    normalized_rmse = correctness["bf16_reference"]["normalized_rmse"]
    assert normalized_rmse is not None
    print(
        f"{result['case']:<24} B={result['batch']:>2} "
        f"{result['distribution']:<8} {row_label}={result['rows']:>6} "
        f"G={group_summary['active_experts']:>3} "
        f"{result['quant_type']:<8} "
        f"PACKED={packed['median_ms']:>8.3f} ms "
        f"{packed['logical_tflops']:>6.2f} TF "
        f"{reference_label}={reference['median_ms']:>8.3f} ms "
        f"{reference['logical_tflops']:>6.2f} TF "
        f"ratio={result['packed_to_reference_tflops_ratio']:>5.2f}x "
        f"diff={dense_metrics['different_bf16_elements']:>7} "
        f"NRMSE={normalized_rmse:.3e}",
        flush=True,
    )


def benchmark_fixed_case(
    args: Namespace,
    case: GroupedMMQCase,
    case_index: int,
    tensor: gguf.ReaderTensor,
) -> list[dict[str, object]]:
    weight = load_fixed_weight(case, tensor)
    physical_shapes = (tuple(weight.packed.shape),)
    results = []
    for batch_index, batch in enumerate(args.batches):
        rows = batch * args.sequence_length
        grad_output = make_bf16_input(
            rows * case.fixed_groups,
            case.out_features,
            args.seed + case_index * 10000 + batch_index * 100,
        ).view(rows, case.fixed_groups, case.out_features)
        packed = benchmark_callable(
            lambda grad_output=grad_output, packed_weight=weight.packed: (
                torch.ops.torch_ggml_ops.fixed_grouped_mmq_grad_input.default(
                    grad_output,
                    packed_weight,
                )
            ),
            rows,
            case.out_features,
            case.in_features,
            args.warmup,
            args.repeats,
            projections=case.projections,
        )
        reference = benchmark_callable(
            lambda grad_output=grad_output, logical_weight=weight.logical: (
                bf16_fixed_grad_input_reference(grad_output, logical_weight)
            ),
            rows,
            case.out_features,
            case.in_features,
            args.warmup,
            args.repeats,
            projections=case.projections,
        )
        correctness_grad_output = grad_output[
            : min(rows, args.correctness_rows)
        ].clone()
        distribution = fixed_group_distribution(rows, case.fixed_groups)
        correctness = fixed_correctness_metrics(
            correctness_grad_output,
            weight.packed,
            weight.logical,
        )
        result = {
            **grouped_result_metadata(
                case,
                batch,
                rows,
                weight.quant_name,
                weight.quant_type,
                distribution,
                physical_shapes,
            ),
            "logical_group_rows": rows * case.fixed_groups,
            "grad_output_shapes": [[rows, case.fixed_groups, case.out_features]],
            "grad_input_shape": [rows, case.fixed_groups, case.in_features],
            "reference_kind": "BF16 torch.bmm with public-layout conversion",
            "grad_input_bytes": (
                rows * case.fixed_groups * case.in_features * torch.bfloat16.itemsize
            ),
            "pair_fuses_two_fp32_accumulations": False,
            "packed": packed,
            "bf16_reference": reference,
            **performance_comparison(packed, reference, case.model_calls),
            "correctness": correctness,
        }
        results.append(result)
        print_result(result)
        del grad_output, correctness_grad_output
        clear_cuda_cache()

    del weight
    clear_cuda_cache()
    return results


def make_routed_functions(
    case: GroupedMMQCase,
    grad_outputs: tuple[torch.Tensor, ...],
    weights: RoutedWeights,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    selected_logical: tuple[torch.Tensor, ...],
    group_sizes: torch.Tensor,
    aiter_config: dict[str, int],
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    if case.kind == "pair":

        def packed_function() -> torch.Tensor:
            return torch.ops.torch_ggml_ops.grouped_mmq_pair_grad_input.default(
                grad_outputs[0],
                grad_outputs[1],
                weights.packed[0],
                weights.packed[1],
                expert_indices,
                expert_offsets,
                weights.quant_type,
                case.in_features,
            )

        def reference_function() -> torch.Tensor:
            return aiter_grouped_pair(
                grad_outputs,
                selected_logical,
                group_sizes,
                aiter_config,
            )

    else:

        def packed_function() -> torch.Tensor:
            return torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
                grad_outputs[0],
                weights.packed[0],
                expert_indices,
                expert_offsets,
                weights.quant_type,
                case.in_features,
            )

        def reference_function() -> torch.Tensor:
            return gmm(
                grad_outputs[0],
                selected_logical[0],
                group_sizes,
                preferred_element_type=grad_outputs[0].dtype,
                config=aiter_config,
            )

    return packed_function, reference_function


def benchmark_transient_control(
    args: Namespace,
    case: GroupedMMQCase,
    rows: int,
    grad_outputs: tuple[torch.Tensor, ...],
    selected_logical: tuple[torch.Tensor, ...],
    group_sizes: torch.Tensor,
    aiter_config: dict[str, int],
) -> BenchmarkTiming | None:
    if not args.transient_bf16_control:
        return None
    return benchmark_callable(
        lambda: transient_bf16_materialization_floor(
            grad_outputs,
            selected_logical,
            group_sizes,
            aiter_config,
        ),
        rows,
        case.out_features,
        case.in_features,
        args.warmup,
        args.repeats,
        projections=case.projections,
    )


def make_routed_result(
    case: GroupedMMQCase,
    batch: int,
    rows: int,
    weights: RoutedWeights,
    distribution: RouteDistribution,
    top_k: int,
    aiter_config: dict[str, int],
    packed: BenchmarkTiming,
    reference: BenchmarkTiming,
    transient: BenchmarkTiming | None,
    transient_workspace_bytes: int,
    correctness: BackwardCorrectness,
) -> dict[str, object]:
    result = {
        **grouped_result_metadata(
            case,
            batch,
            rows,
            weights.quant_name,
            weights.quant_type,
            distribution,
            weights.physical_shapes,
            top_k=top_k,
        ),
        "grad_output_shapes": [
            [rows, case.out_features] for _ in range(case.projections)
        ],
        "grad_input_shape": [rows, case.in_features],
        "reference_kind": "BF16 AITER gmm input gradient",
        "aiter_config": dict(aiter_config),
        "grad_input_bytes": rows * case.in_features * torch.bfloat16.itemsize,
        "pair_fuses_two_fp32_accumulations": case.kind == "pair",
        "packed": packed,
        "bf16_reference": reference,
        **performance_comparison(packed, reference, case.model_calls),
        "correctness": correctness,
    }
    if transient is not None:
        result.update(
            {
                "transient_bf16_materialization_floor": transient,
                "transient_bf16_workspace_bytes": transient_workspace_bytes,
                "packed_to_transient_bf16_floor_latency_ratio": (
                    packed["median_ms"] / transient["median_ms"]
                ),
            }
        )
    return result


def benchmark_routed_distribution(
    args: Namespace,
    case: GroupedMMQCase,
    case_index: int,
    batch: int,
    batch_index: int,
    distribution_name: str,
    distribution_index: int,
    weights: RoutedWeights,
    top_k: int,
) -> dict[str, object]:
    rows = batch * args.sequence_length * top_k
    distribution = route_distributions(rows, batch)[distribution_name]
    expert_indices, expert_offsets, group_sizes = make_route_tensors(distribution)
    selected_logical = tuple(
        weight.index_select(0, expert_indices).contiguous()
        for weight in weights.logical
    )
    aiter_config = aiter_gmm_config(
        rows,
        case.out_features,
        case.in_features,
        selected_logical[0].stride(1) == 1,
    )
    grad_outputs = tuple(
        make_bf16_input(
            rows,
            case.out_features,
            args.seed
            + case_index * 10000
            + batch_index * 100
            + distribution_index * 10
            + projection,
        )
        for projection in range(case.projections)
    )
    packed_function, reference_function = make_routed_functions(
        case,
        grad_outputs,
        weights,
        expert_indices,
        expert_offsets,
        selected_logical,
        group_sizes,
        aiter_config,
    )

    packed = benchmark_callable(
        packed_function,
        rows,
        case.out_features,
        case.in_features,
        args.warmup,
        args.repeats,
        projections=case.projections,
    )
    reference = benchmark_callable(
        reference_function,
        rows,
        case.out_features,
        case.in_features,
        args.warmup,
        args.repeats,
        projections=case.projections,
    )
    transient = benchmark_transient_control(
        args,
        case,
        rows,
        grad_outputs,
        selected_logical,
        group_sizes,
        aiter_config,
    )

    checked_distribution = truncate_distribution(distribution, args.correctness_rows)
    correctness_grad_outputs = tuple(
        grad[: checked_distribution.rows].clone() for grad in grad_outputs
    )
    correctness = correctness_metrics(
        case,
        correctness_grad_outputs,
        weights,
        checked_distribution,
        aiter_config,
    )
    transient_workspace_bytes = (
        sum(weight.numel() * weight.element_size() for weight in selected_logical)
        if transient is not None
        else 0
    )
    result = make_routed_result(
        case,
        batch,
        rows,
        weights,
        distribution,
        top_k,
        aiter_config,
        packed,
        reference,
        transient,
        transient_workspace_bytes,
        correctness,
    )
    print_result(result)
    del packed_function, reference_function
    del grad_outputs, correctness_grad_outputs, selected_logical
    del expert_indices, expert_offsets, group_sizes
    clear_cuda_cache()
    return result


def benchmark_routed_case(
    args: Namespace,
    case: GroupedMMQCase,
    case_index: int,
    tensors: tuple[gguf.ReaderTensor, ...],
) -> list[dict[str, object]]:
    weights = load_routed_weights(case, tensors)
    top_k = args.top_k if args.top_k is not None else case.top_k
    results = [
        benchmark_routed_distribution(
            args,
            case,
            case_index,
            batch,
            batch_index,
            distribution_name,
            distribution_index,
            weights,
            top_k,
        )
        for batch_index, batch in enumerate(args.batches)
        for distribution_index, distribution_name in enumerate(args.distributions)
    ]
    del weights
    clear_cuda_cache()
    return results


def main() -> None:
    args = parse_grouped_benchmark_args(
        __doc__,
        DEFAULT_OUTPUT,
        20260711,
        transient_control_help=(
            "time an optimistic transient representation floor: allocate and "
            "write active-expert BF16 workspace, then run AITER, without decode"
        ),
    )
    device = cuda_device_info()
    cases = select_cases(args.cases, args.primary_only, args.model_family)
    if args.transient_bf16_control and any(not case.routed for case in cases):
        raise ValueError("--transient-bf16-control requires routed cases")
    reader, tensors = load_gguf_tensors(
        args.model,
        tuple(name for case in cases for name in case.tensor_names),
    )
    routed_top_ks = {
        args.top_k if args.top_k is not None else case.top_k
        for case in cases
        if case.routed
    }
    report = {
        "model": str(args.model),
        "model_family": args.model_family,
        "operation": "backward",
        "device": device,
        "configuration": {
            "sequence_length": args.sequence_length,
            "top_k": next(iter(routed_top_ks)) if len(routed_top_ks) == 1 else None,
            "top_k_override": args.top_k,
            "batches": list(args.batches),
            "cases": [case.name for case in cases],
            "distributions": list(args.distributions),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "correctness_rows": args.correctness_rows,
            "aiter_heuristic": "bench/aiter_gmm_heuristics.py:gmm_config",
            "reference": (
                "routed: BF16 AITER gmm input gradient; fixed: BF16 torch.bmm"
            ),
            "pair_reference": "two AITER gmm calls plus torch.add",
            "aiter_work_stealing": False,
            "grad_output_dtype": str(torch.bfloat16),
            "packed_storage_dtype": str(torch.uint8),
            "grad_input_dtype": str(torch.bfloat16),
            "grad_output_quantization": None,
            "transient_bf16_control": args.transient_bf16_control,
            "transient_bf16_control_scope": (
                "active-expert BF16 allocation and stores plus AITER; excludes "
                "packed reads and decode arithmetic"
                if args.transient_bf16_control
                else None
            ),
            "metadata_device_resident": True,
            "host_group_descriptor_build_in_timed_path": False,
        },
        "results": [],
    }
    print_benchmark_header(
        device,
        args.model,
        args.model_family,
        tuple(case.name for case in cases),
    )

    with torch.inference_mode():
        for case_index, case in enumerate(cases):
            case_tensors = tuple(tensors[name] for name in case.tensor_names)
            if case.kind == "fixed":
                results = benchmark_fixed_case(
                    args,
                    case,
                    case_index,
                    case_tensors[0],
                )
            else:
                results = benchmark_routed_case(
                    args,
                    case,
                    case_index,
                    case_tensors,
                )
            report["results"].extend(results)

    del reader
    write_json_report(args.output, report)
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
