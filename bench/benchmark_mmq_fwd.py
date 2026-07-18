#!/usr/bin/env python3
"""Benchmark production dense MMQ forward shapes against BF16 hipBLASLt.

The benchmark reads representative packed tensors directly from the Qwen3.6
GGUF checkpoint. Ordinary projections are measured at sequence length 2048 and
batch sizes 1, 4, and 16. The packed LM head is measured at bounded token chunk
sizes because production training never retains full-sequence logits.

MMQ uses BF16 input, an internal Q8_1 activation workspace, packed uint8 GGUF
weights, and BF16 output. The reference uses the same BF16 input and the same
logical GGUF weight dequantized to BF16, evaluated by torch.mm. On ROCm this is
normally dispatched to hipBLASLt.
"""

import argparse
import gc
import json
import math
from pathlib import Path

import gguf
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops
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


DEFAULT_OUTPUT = Path("/tmp/torch_ggml_ops_mmq_fwd_benchmark.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batches", type=parse_int_list, default=(1, 4, 16))
    parser.add_argument("--lm-head-chunks", type=parse_int_list, default=(64, 128, 256))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--correctness-rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260705)
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
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    logical_weight: torch.Tensor,
    quant_type: int,
    out_features: int,
) -> dict:
    with torch.inference_mode():
        mmq_output = torch_ggml_ops.mmq(input, packed_weight, quant_type, out_features)
        bf16_output = torch.mm(input, logical_weight.transpose(0, 1))
        difference = mmq_output.float() - bf16_output.float()
        reference_rms = bf16_output.float().square().mean().sqrt()
        error_rms = difference.square().mean().sqrt()
        result = {
            "rows": input.shape[0],
            "reference_rms": float(reference_rms),
            "error_rms": float(error_rms),
            "normalized_rmse": float(error_rms / reference_rms),
            "max_absolute_error": float(difference.abs().max()),
            "different_bf16_elements": int(
                torch.count_nonzero(mmq_output != bf16_output)
            ),
            "elements": mmq_output.numel(),
        }
        del mmq_output, bf16_output, difference
        return result


def benchmark_mmq(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_type: int,
    out_features: int,
    warmup: int,
    repeats: int,
) -> dict:
    def function() -> torch.Tensor:
        return torch_ggml_ops.mmq(input, packed_weight, quant_type, out_features)

    times = cuda_event_times_ms(function, warmup, repeats)
    allocated, reserved = incremental_peak_bytes(function)
    result = summarize_timing(times, input.shape[0], out_features, input.shape[1])
    result.update(
        {
            "incremental_peak_allocated_bytes": allocated,
            "incremental_peak_reserved_bytes": reserved,
        }
    )
    return result


def benchmark_bf16(
    input: torch.Tensor,
    logical_weight: torch.Tensor,
    warmup: int,
    repeats: int,
) -> dict:
    transposed_weight = logical_weight.transpose(0, 1)

    def function() -> torch.Tensor:
        return torch.mm(input, transposed_weight)

    times = cuda_event_times_ms(function, warmup, repeats)
    allocated, reserved = incremental_peak_bytes(function)
    result = summarize_timing(
        times,
        input.shape[0],
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
    mmq = row["mmq"]
    bf16 = row["torch_bf16"]
    print(
        f"{row['case']:<23} "
        f"B={row['batch']:>2} calls={row['model_invocations_per_forward']:>3} "
        f"M={row['m']:>6} N={row['n']:>6} K={row['k']:>4} "
        f"{row['quant_type']:<5} "
        f"MMQ={mmq['median_ms']:>8.3f} ms {mmq['logical_tflops']:>6.2f} TF "
        f"BF16={bf16['median_ms']:>8.3f} ms {bf16['logical_tflops']:>6.2f} TF "
        f"ratio={row['mmq_to_bf16_tflops_ratio']:>5.2f}x "
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
            "input_dtype": str(torch.bfloat16),
            "packed_storage_dtype": str(torch.uint8),
            "output_dtype": str(torch.bfloat16),
            "activation_quantization": "Q8_1",
            "torch_reference": "torch.mm(BF16, dequantized_BF16_weight.T)",
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

            mmq_by_m = {}
            for m_index, rows in enumerate(unique_m):
                input_seed = args.seed + case_index * 1000 + m_index
                input = make_bf16_input(rows, case.expected_in_features, input_seed)
                mmq_by_m[rows] = benchmark_mmq(
                    input,
                    packed_weight,
                    quant_type,
                    case.expected_out_features,
                    args.warmup,
                    args.repeats,
                )
                del input

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
                input_seed = args.seed + case_index * 1000 + m_index
                input = make_bf16_input(rows, case.expected_in_features, input_seed)
                bf16_by_m[rows] = benchmark_bf16(
                    input, logical_weight, args.warmup, args.repeats
                )
                checked_rows = min(args.correctness_rows, rows)
                correctness_input = input[:checked_rows].clone()
                correctness_by_m[rows] = correctness_metrics(
                    correctness_input,
                    packed_weight,
                    logical_weight,
                    quant_type,
                    case.expected_out_features,
                )
                del input, correctness_input

            for spec in row_specs:
                rows = spec["m"]
                mmq_result = mmq_by_m[rows]
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
                    "input_shape": [rows, case.expected_in_features],
                    "logical_weight_shape": [
                        case.expected_out_features,
                        case.expected_in_features,
                    ],
                    "physical_weight_shape": list(physical_shape),
                    "output_shape": [rows, case.expected_out_features],
                    "input_dtype": str(torch.bfloat16),
                    "packed_weight_dtype": str(torch.uint8),
                    "output_dtype": str(torch.bfloat16),
                    "quant_type": quant_name,
                    "quant_type_id": quant_type,
                    "model_rows": spec["model_rows"],
                    "calls_per_weight_per_model_forward": spec["calls"],
                    "model_invocations_per_forward": model_invocations,
                    "mmq": mmq_result,
                    "torch_bf16": bf16_result,
                    "mmq_to_bf16_tflops_ratio": (
                        mmq_result["logical_tflops"] / bf16_result["logical_tflops"]
                    ),
                    "estimated_mmq_model_forward_ms": (
                        model_invocations * mmq_result["median_ms"]
                    ),
                    "estimated_torch_bf16_model_forward_ms": (
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
