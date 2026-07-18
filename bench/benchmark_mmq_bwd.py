#!/usr/bin/env python3
"""Benchmark production dense MMQ input-gradient shapes against BF16 hipBLASLt.

For a forward projection Y[M,N] = X[M,K] @ W[N,K].T, the frozen-base
input gradient is dX[M,K] = dY[M,N] @ W[N,K]. This benchmark reads the real
packed GGUF weights, times torch_ggml_ops::mmq_grad_input, and compares it with
torch.mm using the same logical weight dequantized to BF16.

Ordinary projections use sequence length 2048 and batch sizes 1, 4, and 16.
The packed LM head uses bounded token chunks because full-sequence logits and
cotangents are never retained by production training.
"""

import argparse
import gc
import json
import math
from pathlib import Path

import gguf
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops  # noqa: F401 Register native operators before torch.ops use.
from mmq_benchmark_common import (
    DEFAULT_MODEL,
    cuda_event_times_ms,
    incremental_peak_bytes,
    load_packed_tensor,
    make_bf16_input,
    parse_int_list,
    select_cases,
    summarize_timing,
    synchronize,
)


DEFAULT_OUTPUT = Path("/tmp/torch_ggml_ops_mmq_bwd_benchmark.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batches", type=parse_int_list, default=(1, 4, 16))
    parser.add_argument(
        "--lm-head-chunks", type=parse_int_list, default=(64, 128, 256)
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--correctness-rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="comma-separated case names; empty selects every production case",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="benchmark only cases marked primary",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"GGUF model not found: {args.model}")
    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.correctness_rows <= 0:
        parser.error("--correctness-rows must be positive")
    return args


def correctness_metrics(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    logical_weight: torch.Tensor,
    quant_type: int,
    in_features: int,
) -> dict:
    with torch.inference_mode():
        packed_grad_input = torch.ops.torch_ggml_ops.mmq_grad_input.default(
            grad_output,
            packed_weight,
            quant_type,
            in_features,
        )
        bf16_grad_input = torch.mm(grad_output, logical_weight)
        difference = packed_grad_input.float() - bf16_grad_input.float()
        reference_rms = bf16_grad_input.float().square().mean().sqrt()
        error_rms = difference.square().mean().sqrt()
        result = {
            "rows": grad_output.shape[0],
            "reference_rms": float(reference_rms),
            "error_rms": float(error_rms),
            "normalized_rmse": float(error_rms / reference_rms),
            "max_absolute_error": float(difference.abs().max()),
            "different_bf16_elements": int(
                torch.count_nonzero(packed_grad_input != bf16_grad_input)
            ),
            "elements": packed_grad_input.numel(),
        }
        del packed_grad_input, bf16_grad_input, difference
        return result


def benchmark_mmq_grad_input(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_type: int,
    in_features: int,
    warmup: int,
    repeats: int,
) -> dict:
    def function() -> torch.Tensor:
        return torch.ops.torch_ggml_ops.mmq_grad_input.default(
            grad_output,
            packed_weight,
            quant_type,
            in_features,
        )

    times = cuda_event_times_ms(function, warmup, repeats)
    allocated, reserved = incremental_peak_bytes(function)
    result = summarize_timing(
        times,
        grad_output.shape[0],
        grad_output.shape[1],
        in_features,
    )
    result.update(
        {
            "incremental_peak_allocated_bytes": allocated,
            "incremental_peak_reserved_bytes": reserved,
        }
    )
    return result


def benchmark_bf16_grad_input(
    grad_output: torch.Tensor,
    logical_weight: torch.Tensor,
    warmup: int,
    repeats: int,
) -> dict:
    def function() -> torch.Tensor:
        return torch.mm(grad_output, logical_weight)

    times = cuda_event_times_ms(function, warmup, repeats)
    allocated, reserved = incremental_peak_bytes(function)
    result = summarize_timing(
        times,
        grad_output.shape[0],
        logical_weight.shape[0],
        logical_weight.shape[1],
    )
    result.update(
        {
            "incremental_peak_allocated_bytes": allocated,
            "incremental_peak_reserved_bytes": reserved,
        }
    )
    return result


def print_result(row: dict) -> None:
    packed = row["mmq_grad_input"]
    bf16 = row["torch_bf16"]
    print(
        f"{row['case']:<23} "
        f"B={row['batch']:>2} calls={row['model_invocations_per_backward']:>3} "
        f"M={row['m']:>6} N={row['n']:>6} K={row['k']:>4} "
        f"{row['quant_type']:<5} "
        f"PACKED={packed['median_ms']:>8.3f} ms "
        f"{packed['logical_tflops']:>6.2f} TF "
        f"BF16={bf16['median_ms']:>8.3f} ms {bf16['logical_tflops']:>6.2f} TF "
        f"ratio={row['packed_to_bf16_tflops_ratio']:>5.2f}x "
        f"NRMSE={row['correctness']['normalized_rmse']:.3e}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a HIP/CUDA device is required")

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    reader = gguf.GGUFReader(args.model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    quant_names = {int(value): value.name for value in gguf.GGMLQuantizationType}

    cases = select_cases(args.cases, args.primary_only)
    missing = [case.tensor_name for case in cases if case.tensor_name not in tensors]
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
            "batches": list(args.batches),
            "lm_head_chunks": list(args.lm_head_chunks),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "correctness_rows": args.correctness_rows,
            "grad_output_dtype": str(torch.bfloat16),
            "packed_storage_dtype": str(torch.uint8),
            "grad_input_dtype": str(torch.bfloat16),
            "cotangent_quantization": None,
            "torch_reference": (
                "torch.mm(BF16_grad_output, dequantized_BF16_weight)"
            ),
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
            tensor = tensors[case.tensor_name]
            logical_shape = tuple(int(value) for value in reversed(tensor.shape))
            if logical_shape != (
                case.expected_out_features,
                case.expected_in_features,
            ):
                raise RuntimeError(
                    f"{case.tensor_name} has logical shape {logical_shape}, expected "
                    f"{(case.expected_out_features, case.expected_in_features)}"
                )

            quant_type = int(tensor.tensor_type)
            quant_name = quant_names.get(quant_type, str(quant_type))
            physical_shape = tuple(int(value) for value in tensor.data.shape)
            packed_weight = load_packed_tensor(tensor)
            if packed_weight.dtype != torch.uint8 or not packed_weight.is_contiguous():
                raise RuntimeError("packed benchmark weight is not contiguous uint8")

            if case.lm_head:
                row_specs = [
                    {
                        "batch": batch,
                        "m": chunk,
                        "model_rows": batch * args.sequence_length,
                        "calls": math.ceil(batch * args.sequence_length / chunk),
                    }
                    for chunk in args.lm_head_chunks
                    for batch in args.batches
                ]
                unique_m = args.lm_head_chunks
            else:
                row_specs = [
                    {
                        "batch": batch,
                        "m": batch * args.sequence_length,
                        "model_rows": batch * args.sequence_length,
                        "calls": 1,
                    }
                    for batch in args.batches
                ]
                unique_m = tuple(spec["m"] for spec in row_specs)

            packed_by_m = {}
            for m_index, rows in enumerate(unique_m):
                grad_seed = args.seed + case_index * 1000 + m_index
                grad_output = make_bf16_input(
                    rows,
                    case.expected_out_features,
                    grad_seed,
                )
                packed_by_m[rows] = benchmark_mmq_grad_input(
                    grad_output,
                    packed_weight,
                    quant_type,
                    case.expected_in_features,
                    args.warmup,
                    args.repeats,
                )
                del grad_output

            synchronize()
            logical_weight = dequantize_gguf_tensor(
                packed_weight,
                tensor.tensor_type,
                dtype=torch.bfloat16,
                device="cuda",
            ).reshape(case.expected_out_features, case.expected_in_features)
            logical_weight = logical_weight.contiguous()

            bf16_by_m = {}
            correctness_by_m = {}
            for m_index, rows in enumerate(unique_m):
                grad_seed = args.seed + case_index * 1000 + m_index
                grad_output = make_bf16_input(
                    rows,
                    case.expected_out_features,
                    grad_seed,
                )
                bf16_by_m[rows] = benchmark_bf16_grad_input(
                    grad_output,
                    logical_weight,
                    args.warmup,
                    args.repeats,
                )
                checked_rows = min(args.correctness_rows, rows)
                correctness_grad_output = grad_output[:checked_rows].clone()
                correctness_by_m[rows] = correctness_metrics(
                    correctness_grad_output,
                    packed_weight,
                    logical_weight,
                    quant_type,
                    case.expected_in_features,
                )
                del grad_output, correctness_grad_output

            for spec in row_specs:
                rows = spec["m"]
                packed_result = packed_by_m[rows]
                bf16_result = bf16_by_m[rows]
                model_invocations = spec["calls"] * case.model_tensor_count
                result = {
                    "case": case.name,
                    "description": case.description,
                    "priority": case.priority,
                    "tensor_name": case.tensor_name,
                    "model_tensor_count": case.model_tensor_count,
                    "batch": spec["batch"],
                    "sequence_length": args.sequence_length,
                    "m": rows,
                    "n": case.expected_out_features,
                    "k": case.expected_in_features,
                    "grad_output_shape": [rows, case.expected_out_features],
                    "logical_weight_shape": [
                        case.expected_out_features,
                        case.expected_in_features,
                    ],
                    "physical_weight_shape": list(physical_shape),
                    "grad_input_shape": [rows, case.expected_in_features],
                    "grad_output_dtype": str(torch.bfloat16),
                    "packed_weight_dtype": str(torch.uint8),
                    "grad_input_dtype": str(torch.bfloat16),
                    "quant_type": quant_name,
                    "quant_type_id": quant_type,
                    "model_rows": spec["model_rows"],
                    "calls_per_weight_per_model_backward": spec["calls"],
                    "model_invocations_per_backward": model_invocations,
                    "mmq_grad_input": packed_result,
                    "torch_bf16": bf16_result,
                    "packed_to_bf16_tflops_ratio": (
                        packed_result["logical_tflops"]
                        / bf16_result["logical_tflops"]
                    ),
                    "estimated_packed_model_backward_ms": (
                        model_invocations * packed_result["median_ms"]
                    ),
                    "estimated_torch_bf16_model_backward_ms": (
                        model_invocations * bf16_result["median_ms"]
                    ),
                    "correctness": correctness_by_m[rows],
                }
                report["results"].append(result)
                print_result(result)

            del logical_weight, packed_weight
            gc.collect()
            torch.cuda.empty_cache()
            synchronize()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
