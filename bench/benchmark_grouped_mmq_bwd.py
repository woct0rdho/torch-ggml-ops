#!/usr/bin/env python3
"""Benchmark production grouped MMQ input gradients against BF16 AITER GMM.

For each routed expert group, a forward projection is
``Y[M,N] = X[M,K] @ W[N,K].T`` and the frozen-base input gradient is
``dX[M,K] = dY[M,N] @ W[N,K]``. The packed path times the public grouped
input-gradient operators on real GGUF expert weights. Paired gate/up backward
uses one fused kernel that accumulates both logical Jacobians into one FP32
accumulator. The BF16 reference uses AITER GMM with the project-owned
production gfx1151 heuristic. Paired backward includes two AITER GMM calls and
the BF16 gradient sum.
"""

import argparse
import gc
import json
from pathlib import Path

import gguf
import torch
from aiter.ops.triton.gmm import gmm
from grouped_mmq_benchmark_common import (
    GroupedMMQCase,
    RouteDistribution,
    benchmark_function,
    device_metadata,
    distribution_summary,
    error_metrics,
    parse_name_list,
    route_distributions,
    select_cases,
    truncate_distribution,
)
from mmq_benchmark_common import (
    DEFAULT_MODEL,
    load_packed_tensor,
    make_bf16_input,
    parse_int_list,
    synchronize,
)
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops  # noqa: F401 Register native operators before torch.ops use.
from torch_ggml_ops.aiter_gmm_heuristics import gmm_config as aiter_gmm_config

DEFAULT_OUTPUT = Path("/tmp/torch_ggml_ops_grouped_mmq_bwd_benchmark.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batches", type=parse_int_list, default=(1, 4, 16))
    parser.add_argument(
        "--distributions",
        type=parse_name_list,
        default=("uniform", "skewed", "sparse", "boundary"),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--correctness-rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="comma-separated case names; empty selects every production case",
    )
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"GGUF model not found: {args.model}")
    if args.sequence_length <= 0 or args.top_k <= 0:
        parser.error("--sequence-length and --top-k must be positive")
    if args.warmup < 0 or args.repeats <= 0 or args.correctness_rows <= 0:
        parser.error(
            "warmup must be nonnegative; repeats/correctness rows must be positive"
        )
    known_distributions = {"uniform", "skewed", "sparse", "boundary"}
    unknown = sorted(set(args.distributions) - known_distributions)
    if unknown:
        parser.error(
            f"unknown distributions {unknown}; expected {sorted(known_distributions)}"
        )
    return args


def packed_grouped_single(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    quant_type: int,
    in_features: int,
) -> torch.Tensor:
    return torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
        grad_output,
        packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        in_features,
    )


def packed_grouped_pair(
    first_grad_output: torch.Tensor,
    second_grad_output: torch.Tensor,
    first_packed_weight: torch.Tensor,
    second_packed_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    quant_type: int,
    in_features: int,
) -> torch.Tensor:
    return torch.ops.torch_ggml_ops.grouped_mmq_pair_grad_input.default(
        first_grad_output,
        second_grad_output,
        first_packed_weight,
        second_packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        in_features,
    )


def aiter_grouped_single(
    grad_output: torch.Tensor,
    selected_logical_weight: torch.Tensor,
    group_sizes: torch.Tensor,
    config: dict[str, int],
) -> torch.Tensor:
    return gmm(
        grad_output,
        selected_logical_weight,
        group_sizes,
        preferred_element_type=grad_output.dtype,
        config=config,
    )


def aiter_grouped_pair(
    first_grad_output: torch.Tensor,
    second_grad_output: torch.Tensor,
    first_selected_logical_weight: torch.Tensor,
    second_selected_logical_weight: torch.Tensor,
    group_sizes: torch.Tensor,
    config: dict[str, int],
) -> torch.Tensor:
    first_grad_input = aiter_grouped_single(
        first_grad_output, first_selected_logical_weight, group_sizes, config
    )
    second_grad_input = aiter_grouped_single(
        second_grad_output, second_selected_logical_weight, group_sizes, config
    )
    return torch.add(first_grad_input, second_grad_input)


def dense_grouped_single_reference(
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


def correctness_metrics(
    case: GroupedMMQCase,
    grad_outputs: tuple[torch.Tensor, ...],
    packed_weights: tuple[torch.Tensor, ...],
    logical_weights: tuple[torch.Tensor, ...],
    distribution: RouteDistribution,
    quant_type: int,
    aiter_config: dict[str, int],
) -> dict:
    checked_distribution = truncate_distribution(distribution, grad_outputs[0].shape[0])
    expert_indices, expert_offsets, group_sizes = device_metadata(checked_distribution)
    selected_logical = tuple(
        weight.index_select(0, expert_indices).contiguous()
        for weight in logical_weights
    )

    with torch.inference_mode():
        if case.kind == "pair":
            actual = packed_grouped_pair(
                grad_outputs[0],
                grad_outputs[1],
                packed_weights[0],
                packed_weights[1],
                expert_indices,
                expert_offsets,
                quant_type,
                case.expected_in_features,
            )
            first_dense = dense_grouped_single_reference(
                grad_outputs[0],
                packed_weights[0],
                checked_distribution,
                quant_type,
                case.expected_in_features,
            )
            second_dense = dense_grouped_single_reference(
                grad_outputs[1],
                packed_weights[1],
                checked_distribution,
                quant_type,
                case.expected_in_features,
            )
            dense_reference = torch.add(first_dense, second_dense)
            aiter_reference = aiter_grouped_pair(
                grad_outputs[0],
                grad_outputs[1],
                selected_logical[0],
                selected_logical[1],
                group_sizes,
                aiter_config,
            )
        else:
            actual = packed_grouped_single(
                grad_outputs[0],
                packed_weights[0],
                expert_indices,
                expert_offsets,
                quant_type,
                case.expected_in_features,
            )
            dense_reference = dense_grouped_single_reference(
                grad_outputs[0],
                packed_weights[0],
                checked_distribution,
                quant_type,
                case.expected_in_features,
            )
            aiter_reference = aiter_grouped_single(
                grad_outputs[0],
                selected_logical[0],
                group_sizes,
                aiter_config,
            )

        return {
            "rows": grad_outputs[0].shape[0],
            "packed_dense_bf16_sum": error_metrics(actual, dense_reference),
            "bf16_aiter": error_metrics(actual, aiter_reference),
        }


def print_result(result: dict) -> None:
    packed = result["grouped_mmq_grad_input"]
    aiter = result["aiter_bf16_grad_input"]
    correctness = result["correctness"]
    print(
        f"{result['case']:<18} B={result['batch']:>2} "
        f"{result['distribution']:<8} R={result['rows']:>6} "
        f"G={result['group_summary']['active_experts']:>3} "
        f"{result['quant_type']:<5} "
        f"PACKED={packed['median_ms']:>8.3f} ms {packed['logical_tflops']:>6.2f} TF "
        f"AITER={aiter['median_ms']:>8.3f} ms {aiter['logical_tflops']:>6.2f} TF "
        f"ratio={result['packed_to_aiter_tflops_ratio']:>5.2f}x "
        f"dense_diff={correctness['packed_dense_bf16_sum']['different_bf16_elements']:>7} "
        f"NRMSE={correctness['bf16_aiter']['normalized_rmse']:.3e}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a HIP/CUDA device is required")

    cases = select_cases(args.cases, args.primary_only)
    reader = gguf.GGUFReader(args.model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    quant_names = {int(value): value.name for value in gguf.GGMLQuantizationType}
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())

    missing = [
        name for case in cases for name in case.tensor_names if name not in tensors
    ]
    if missing:
        raise RuntimeError(f"checkpoint is missing benchmark tensors: {missing}")

    report = {
        "model": str(args.model),
        "device": {
            "name": properties.name,
            "gcn_arch_name": getattr(properties, "gcnArchName", None),
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
        },
        "configuration": {
            "sequence_length": args.sequence_length,
            "top_k": args.top_k,
            "batches": list(args.batches),
            "distributions": list(args.distributions),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "correctness_rows": args.correctness_rows,
            "aiter_heuristic": "torch_ggml_ops.aiter_gmm_heuristics.gmm_config",
            "reference": "BF16 AITER gmm input gradient with project-owned gmm_config",
            "pair_reference": "two AITER gmm calls plus torch.add",
            "aiter_work_stealing": False,
            "cotangent_dtype": str(torch.bfloat16),
            "packed_storage_dtype": str(torch.uint8),
            "grad_input_dtype": str(torch.bfloat16),
            "cotangent_quantization": None,
        },
        "results": [],
    }

    print(
        f"device={properties.name} arch={getattr(properties, 'gcnArchName', None)} "
        f"torch={torch.__version__} hip={torch.version.hip}",
        flush=True,
    )
    print(f"model={args.model}", flush=True)

    with torch.inference_mode():
        for case_index, case in enumerate(cases):
            case_tensors = tuple(tensors[name] for name in case.tensor_names)
            quant_types = {int(tensor.tensor_type) for tensor in case_tensors}
            if len(quant_types) != 1:
                raise RuntimeError(f"{case.name} paired tensors have different qtypes")
            quant_type = quant_types.pop()
            quant_name = quant_names.get(quant_type, str(quant_type))

            packed_weights = tuple(
                load_packed_tensor(tensor) for tensor in case_tensors
            )
            physical_shapes = []
            for tensor, packed in zip(case_tensors, packed_weights, strict=True):
                logical_shape = tuple(
                    int(value) for value in reversed(tensor.shape[:-1])
                )
                if logical_shape != (
                    case.expected_out_features,
                    case.expected_in_features,
                ):
                    raise RuntimeError(
                        f"{tensor.name} has logical per-expert shape {logical_shape}, "
                        f"expected {(case.expected_out_features, case.expected_in_features)}"
                    )
                if packed.shape[0] != 256:
                    raise RuntimeError(
                        f"{tensor.name} has {packed.shape[0]} experts, expected 256"
                    )
                physical_shapes.append(list(packed.shape))

            logical_weights = tuple(
                dequantize_gguf_tensor(
                    packed,
                    tensor.tensor_type,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                .reshape(256, case.expected_out_features, case.expected_in_features)
                .contiguous()
                for tensor, packed in zip(case_tensors, packed_weights, strict=True)
            )
            aiter_config = aiter_gmm_config(
                case.expected_out_features, case.expected_in_features
            )

            for batch_index, batch in enumerate(args.batches):
                rows = batch * args.sequence_length * args.top_k
                available_distributions = route_distributions(rows, batch)
                for distribution_index, distribution_name in enumerate(
                    args.distributions
                ):
                    distribution = available_distributions[distribution_name]
                    expert_indices, expert_offsets, group_sizes = device_metadata(
                        distribution
                    )
                    selected_logical = tuple(
                        weight.index_select(0, expert_indices).contiguous()
                        for weight in logical_weights
                    )
                    grad_outputs = tuple(
                        make_bf16_input(
                            rows,
                            case.expected_out_features,
                            args.seed
                            + case_index * 10000
                            + batch_index * 100
                            + distribution_index * 10
                            + projection,
                        )
                        for projection in range(case.projections)
                    )

                    if case.kind == "pair":

                        def packed_function(
                            grad_outputs=grad_outputs,
                            packed_weights=packed_weights,
                            expert_indices=expert_indices,
                            expert_offsets=expert_offsets,
                            quant_type=quant_type,
                            in_features=case.expected_in_features,
                        ):
                            return packed_grouped_pair(
                                grad_outputs[0],
                                grad_outputs[1],
                                packed_weights[0],
                                packed_weights[1],
                                expert_indices,
                                expert_offsets,
                                quant_type,
                                in_features,
                            )

                        def aiter_function(
                            grad_outputs=grad_outputs,
                            selected_logical=selected_logical,
                            group_sizes=group_sizes,
                            aiter_config=aiter_config,
                        ):
                            return aiter_grouped_pair(
                                grad_outputs[0],
                                grad_outputs[1],
                                selected_logical[0],
                                selected_logical[1],
                                group_sizes,
                                aiter_config,
                            )
                    else:

                        def packed_function(
                            grad_outputs=grad_outputs,
                            packed_weights=packed_weights,
                            expert_indices=expert_indices,
                            expert_offsets=expert_offsets,
                            quant_type=quant_type,
                            in_features=case.expected_in_features,
                        ):
                            return packed_grouped_single(
                                grad_outputs[0],
                                packed_weights[0],
                                expert_indices,
                                expert_offsets,
                                quant_type,
                                in_features,
                            )

                        def aiter_function(
                            grad_outputs=grad_outputs,
                            selected_logical=selected_logical,
                            group_sizes=group_sizes,
                            aiter_config=aiter_config,
                        ):
                            return aiter_grouped_single(
                                grad_outputs[0],
                                selected_logical[0],
                                group_sizes,
                                aiter_config,
                            )

                    packed_result = benchmark_function(
                        packed_function,
                        rows,
                        case.expected_out_features,
                        case.expected_in_features,
                        case.projections,
                        args.warmup,
                        args.repeats,
                    )
                    aiter_result = benchmark_function(
                        aiter_function,
                        rows,
                        case.expected_out_features,
                        case.expected_in_features,
                        case.projections,
                        args.warmup,
                        args.repeats,
                    )

                    checked_distribution = truncate_distribution(
                        distribution, args.correctness_rows
                    )
                    correctness_grad_outputs = tuple(
                        grad[: checked_distribution.rows].clone()
                        for grad in grad_outputs
                    )
                    correctness = correctness_metrics(
                        case,
                        correctness_grad_outputs,
                        packed_weights,
                        logical_weights,
                        checked_distribution,
                        quant_type,
                        aiter_config,
                    )

                    output_bytes = (
                        rows * case.expected_in_features * torch.bfloat16.itemsize
                    )
                    result = {
                        "case": case.name,
                        "kind": case.kind,
                        "description": case.description,
                        "priority": case.priority,
                        "batch": batch,
                        "rows": rows,
                        "out_features": case.expected_out_features,
                        "in_features": case.expected_in_features,
                        "projections": case.projections,
                        "grad_output_shapes": [
                            [rows, case.expected_out_features]
                            for _ in range(case.projections)
                        ],
                        "grad_input_shape": [rows, case.expected_in_features],
                        "physical_weight_shapes": physical_shapes,
                        "quant_type": quant_name,
                        "quant_type_id": quant_type,
                        "distribution": distribution.name,
                        "group_summary": distribution_summary(distribution),
                        "expert_indices": list(distribution.expert_indices_cpu),
                        "group_sizes": list(distribution.group_sizes_cpu),
                        "aiter_config": dict(aiter_config),
                        "expected_output_bytes": output_bytes,
                        "pair_fuses_two_fp32_accumulations": case.kind == "pair",
                        "model_calls_per_backward": case.model_layer_count,
                        "grouped_mmq_grad_input": packed_result,
                        "aiter_bf16_grad_input": aiter_result,
                        "packed_to_aiter_tflops_ratio": (
                            packed_result["logical_tflops"]
                            / aiter_result["logical_tflops"]
                        ),
                        "estimated_packed_model_backward_ms": (
                            case.model_layer_count * packed_result["median_ms"]
                        ),
                        "estimated_aiter_model_backward_ms": (
                            case.model_layer_count * aiter_result["median_ms"]
                        ),
                        "correctness": correctness,
                        "metadata_device_resident": True,
                        "host_group_descriptor_build_in_timed_path": False,
                        "current_stream_operator_contract_tested": True,
                    }
                    report["results"].append(result)
                    print_result(result)

                    del packed_function, aiter_function
                    del (
                        grad_outputs,
                        correctness_grad_outputs,
                        selected_logical,
                        expert_indices,
                        expert_offsets,
                        group_sizes,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    synchronize()

            del logical_weights, packed_weights
            gc.collect()
            torch.cuda.empty_cache()
            synchronize()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
