#!/usr/bin/env python3
"""Benchmark production routed and fixed-group MMQ forward.

The packed path measures the complete public operator, including Q8_1 activation
quantization and grouped multiplication. Routed gate/up uses grouped_mmq_pair so
both projections share one Q8_1 workspace and compares with BF16 AITER GMM.
Fixed output-A uses fixed_grouped_mmq and a BF16 strided-batched GEMM reference.
"""

from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

import gguf
import torch
from aiter.ops.triton.gmm import gmm
from benchmark_routes import (
    GroupSummary,
    RouteDistribution,
    fitted_prior_distribution,
    fixed_group_distribution,
    make_route_tensors,
    truncate_distribution,
)
from grouped_mmq_benchmark_common import (
    GroupedMMQCase,
    RoutedWeights,
    bf16_fixed_mmq_reference,
    grouped_result_metadata,
    load_active_routed_logical_weights,
    load_fixed_weight,
    load_routed_weights,
    parse_grouped_benchmark_args,
    select_cases,
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
from workload_prior import expert_prior_metadata, expert_prior_top_k

import torch_ggml_ops
from tools.aiter_gmm_heuristics import gmm_config as aiter_gmm_config

DEFAULT_OUTPUT = Path("/tmp/torch_ggml_ops_grouped_mmq_fwd_benchmark.json")


class ForwardCorrectness(TypedDict):
    rows: int
    packed_dense_reference: ErrorMetrics | list[ErrorMetrics]
    bf16_reference: ErrorMetrics | list[ErrorMetrics]


def dense_grouped_mmq_reference(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    distribution: RouteDistribution,
    quant_type: int,
    out_features: int,
) -> torch.Tensor:
    outputs = []
    row_begin = 0
    for expert, size in zip(
        distribution.expert_indices_cpu, distribution.group_sizes_cpu, strict=True
    ):
        row_end = row_begin + size
        outputs.append(
            torch_ggml_ops.mmq(
                input[row_begin:row_end].clone(),
                packed_weight[expert].clone(),
                quant_type,
                out_features,
            )
        )
        row_begin = row_end
    return torch.cat(outputs, dim=0)


def dense_fixed_mmq_reference(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_type: int,
    out_features: int,
) -> torch.Tensor:
    return torch.stack(
        tuple(
            torch_ggml_ops.mmq(
                input[:, group].clone(),
                packed_weight[group].clone(),
                quant_type,
                out_features,
            )
            for group in range(input.shape[1])
        ),
        dim=1,
    )


def fixed_correctness_metrics(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    logical_weight: torch.Tensor,
    quant_type: int,
    out_features: int,
) -> ForwardCorrectness:
    with torch.inference_mode():
        actual = torch_ggml_ops.fixed_grouped_mmq(input, packed_weight)
        packed_reference = dense_fixed_mmq_reference(
            input,
            packed_weight,
            quant_type,
            out_features,
        )
        bf16_reference = bf16_fixed_mmq_reference(input, logical_weight)
    return {
        "rows": input.shape[0],
        "packed_dense_reference": error_metrics(actual, packed_reference),
        "bf16_reference": error_metrics(actual, bf16_reference),
    }


def correctness_metrics(
    case: GroupedMMQCase,
    input: torch.Tensor,
    weights: RoutedWeights,
    distribution: RouteDistribution,
    gmm_config: dict[str, int],
) -> ForwardCorrectness:
    checked_distribution = truncate_distribution(distribution, input.shape[0])
    expert_indices, expert_offsets, group_sizes = make_route_tensors(
        checked_distribution
    )
    selected_logical = tuple(
        weight.transpose(1, 2)
        for weight in load_active_routed_logical_weights(case, weights, expert_indices)
    )
    packed_function, reference_function = make_routed_functions(
        case,
        input,
        weights,
        expert_indices,
        expert_offsets,
        selected_logical,
        group_sizes,
        gmm_config,
    )
    with torch.inference_mode():
        actual = packed_function()
        packed_reference = tuple(
            dense_grouped_mmq_reference(
                input,
                weight,
                checked_distribution,
                weights.quant_type,
                case.out_features,
            )
            for weight in weights.packed
        )
        bf16_reference = reference_function()
    if case.kind == "pair":
        assert isinstance(actual, tuple)
        assert isinstance(bf16_reference, tuple)
        return {
            "rows": input.shape[0],
            "packed_dense_reference": [
                error_metrics(actual[index], packed_reference[index])
                for index in range(2)
            ],
            "bf16_reference": [
                error_metrics(actual[index], bf16_reference[index])
                for index in range(2)
            ],
        }
    assert isinstance(actual, torch.Tensor)
    assert isinstance(bf16_reference, torch.Tensor)
    return {
        "rows": input.shape[0],
        "packed_dense_reference": error_metrics(actual, packed_reference[0]),
        "bf16_reference": error_metrics(actual, bf16_reference),
    }


def max_normalized_rmse(correctness: ForwardCorrectness) -> float:
    metrics = correctness["bf16_reference"]
    if isinstance(metrics, list):
        values = [metric["normalized_rmse"] for metric in metrics]
    else:
        values = [metrics["normalized_rmse"]]
    assert all(value is not None for value in values)
    return max(value for value in values if value is not None)


def print_result(result: dict[str, object]) -> None:
    packed = cast(BenchmarkTiming, result["packed"])
    reference = cast(BenchmarkTiming, result["bf16_reference"])
    correctness = cast(ForwardCorrectness, result["correctness"])
    group_summary = cast(GroupSummary, result["group_summary"])
    reference_label = "BMM" if result["kind"] == "fixed" else "AITER"
    row_label = "M" if result["kind"] == "fixed" else "R"
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
        f"NRMSE={max_normalized_rmse(correctness):.3e}",
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
        input = make_bf16_input(
            rows * case.fixed_groups,
            case.in_features,
            args.seed + case_index * 10000 + batch_index * 100,
        ).view(rows, case.fixed_groups, case.in_features)
        packed = benchmark_callable(
            lambda input=input, packed_weight=weight.packed: (
                torch_ggml_ops.fixed_grouped_mmq(input, packed_weight)
            ),
            rows,
            case.out_features,
            case.in_features,
            args.warmup,
            args.repeats,
            projections=case.projections,
        )
        reference = benchmark_callable(
            lambda input=input, logical_weight=weight.logical: bf16_fixed_mmq_reference(
                input, logical_weight
            ),
            rows,
            case.out_features,
            case.in_features,
            args.warmup,
            args.repeats,
            projections=case.projections,
        )
        correctness_input = input[: min(rows, args.correctness_rows)].clone()
        distribution = fixed_group_distribution(rows, case.fixed_groups)
        correctness = fixed_correctness_metrics(
            correctness_input,
            weight.packed,
            weight.logical,
            weight.quant_type,
            case.out_features,
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
            "input_shape": [rows, case.fixed_groups, case.in_features],
            "output_shape": [rows, case.fixed_groups, case.out_features],
            "reference_kind": "BF16 torch.bmm with public-layout conversion",
            "q8_workspace_bytes": (
                rows * case.fixed_groups * (case.in_features // (4 * 32)) * 144
            ),
            "output_bytes": (
                rows * case.fixed_groups * case.out_features * torch.bfloat16.itemsize
            ),
            "pair_shares_one_q8_workspace": False,
            "fixed_groups_share_one_q8_workspace": True,
            "packed": packed,
            "bf16_reference": reference,
            **performance_comparison(packed, reference, case.model_calls),
            "correctness": correctness,
        }
        results.append(result)
        print_result(result)
        del input, correctness_input
        clear_cuda_cache()

    del weight
    clear_cuda_cache()
    return results


def make_routed_functions(
    case: GroupedMMQCase,
    input: torch.Tensor,
    weights: RoutedWeights,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    selected_logical: tuple[torch.Tensor, ...],
    group_sizes: torch.Tensor,
    gmm_config: dict[str, int],
) -> tuple[
    Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
]:
    if case.kind == "pair":

        def packed_function() -> tuple[torch.Tensor, torch.Tensor]:
            return torch_ggml_ops.grouped_mmq_pair(
                input,
                weights.packed[0],
                weights.packed[1],
                expert_indices,
                expert_offsets,
                weights.quant_type,
                case.out_features,
            )

        def reference_function() -> tuple[torch.Tensor, ...]:
            return tuple(
                gmm(
                    input,
                    weight,
                    group_sizes,
                    preferred_element_type=input.dtype,
                    config=gmm_config,
                )
                for weight in selected_logical
            )

    else:

        def packed_function() -> torch.Tensor:
            return torch_ggml_ops.grouped_mmq(
                input,
                weights.packed[0],
                expert_indices,
                expert_offsets,
                weights.quant_type,
                case.out_features,
            )

        def reference_function() -> torch.Tensor:
            return gmm(
                input,
                selected_logical[0],
                group_sizes,
                preferred_element_type=input.dtype,
                config=gmm_config,
            )

    return packed_function, reference_function


def benchmark_routed_profile(
    args: Namespace,
    case: GroupedMMQCase,
    case_index: int,
    batch: int,
    batch_index: int,
    weights: RoutedWeights,
    top_k: int,
) -> dict[str, object]:
    rows = batch * args.sequence_length * top_k
    distribution = fitted_prior_distribution(
        args.expert_prior, batch * args.sequence_length
    )
    expert_indices, expert_offsets, group_sizes = make_route_tensors(distribution)
    selected_logical = tuple(
        weight.transpose(1, 2)
        for weight in load_active_routed_logical_weights(case, weights, expert_indices)
    )
    gmm_config = aiter_gmm_config(
        rows,
        case.in_features,
        case.out_features,
        selected_logical[0].stride(1) == 1,
    )
    input = make_bf16_input(
        rows,
        case.in_features,
        args.seed + case_index * 10000 + batch_index * 100,
    )
    packed_function, reference_function = make_routed_functions(
        case,
        input,
        weights,
        expert_indices,
        expert_offsets,
        selected_logical,
        group_sizes,
        gmm_config,
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
    del packed_function, reference_function, selected_logical
    checked_distribution = truncate_distribution(distribution, args.correctness_rows)
    correctness_input = input[: checked_distribution.rows].clone()
    correctness = correctness_metrics(
        case,
        correctness_input,
        weights,
        checked_distribution,
        gmm_config,
    )
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
        "input_shape": [rows, case.in_features],
        "output_shapes": [[rows, case.out_features] for _ in range(case.projections)],
        "reference_kind": "BF16 AITER gmm",
        "aiter_config": dict(gmm_config),
        "q8_workspace_bytes": rows * (case.in_features // (4 * 32)) * 144,
        "output_bytes": (
            case.projections * rows * case.out_features * torch.bfloat16.itemsize
        ),
        "pair_shares_one_q8_workspace": case.kind == "pair",
        "packed": packed,
        "bf16_reference": reference,
        **performance_comparison(packed, reference, case.model_calls),
        "correctness": correctness,
    }
    print_result(result)
    del input, correctness_input
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
    top_k = expert_prior_top_k(args.expert_prior)
    if top_k != case.top_k:
        raise ValueError(
            f"{args.expert_prior} top-k {top_k} does not match {case.name} "
            f"top-k {case.top_k}"
        )
    results = [
        benchmark_routed_profile(
            args,
            case,
            case_index,
            batch,
            batch_index,
            weights,
            top_k,
        )
        for batch_index, batch in enumerate(args.batches)
    ]
    del weights
    clear_cuda_cache()
    return results


def main() -> None:
    args = parse_grouped_benchmark_args(__doc__, DEFAULT_OUTPUT, 20260709)
    device = cuda_device_info()
    cases = select_cases(args.cases, args.primary_only, args.model_family)
    reader, tensors = load_gguf_tensors(
        args.model,
        tuple(name for case in cases for name in case.tensor_names),
    )
    routed_top_ks = {case.top_k for case in cases if case.routed}
    report = {
        "model": str(args.model),
        "model_family": args.model_family,
        "operation": "forward",
        "device": device,
        "configuration": {
            "sequence_length": args.sequence_length,
            "top_k": next(iter(routed_top_ks)) if len(routed_top_ks) == 1 else None,
            "expert_prior": expert_prior_metadata(args.expert_prior),
            "batches": list(args.batches),
            "cases": [case.name for case in cases],
            "warmup": args.warmup,
            "repeats": args.repeats,
            "correctness_rows": args.correctness_rows,
            "aiter_heuristic": "bench/aiter_gmm_heuristics.py:gmm_config",
            "reference": "routed: BF16 AITER gmm; fixed: BF16 torch.bmm",
            "aiter_work_stealing": False,
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
