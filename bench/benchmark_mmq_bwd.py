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

from argparse import Namespace
from pathlib import Path

import gguf
import torch
from mmq_benchmark_common import (
    BenchmarkTiming,
    DenseMMQCase,
    benchmark_callable,
    clear_cuda_cache,
    cuda_device_info,
    dense_result_metadata,
    error_metrics,
    load_gguf_tensors,
    make_bf16_input,
    make_row_specs,
    parse_dense_benchmark_args,
    performance_comparison,
    print_benchmark_header,
    print_dense_result,
    resolve_lm_head_chunks,
    select_cases,
    validate_weight_case,
    write_json_report,
)
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops  # noqa: F401 Register native operators before torch.ops use.
from tests.mmq_test_support import load_packed_tensor

DEFAULT_OUTPUT = Path("/tmp/torch_ggml_ops_mmq_bwd_benchmark.json")


def correctness_metrics(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    logical_weight: torch.Tensor,
    quant_type: int,
    in_features: int,
) -> dict[str, object]:
    with torch.inference_mode():
        actual = torch.ops.torch_ggml_ops.mmq_grad_input.default(
            grad_output,
            packed_weight,
            quant_type,
            in_features,
        )
        expected = torch.mm(grad_output, logical_weight)
    return {"rows": grad_output.shape[0], **error_metrics(actual, expected)}


def benchmark_packed_rows(
    args: Namespace,
    case: DenseMMQCase,
    case_index: int,
    rows_to_measure: tuple[int, ...],
    packed_weight: torch.Tensor,
    quant_type: int,
) -> dict[int, BenchmarkTiming]:
    measurements: dict[int, BenchmarkTiming] = {}
    for row_index, rows in enumerate(rows_to_measure):
        grad_output = make_bf16_input(
            rows,
            case.out_features,
            args.seed + case_index * 1000 + row_index,
        )
        measurements[rows] = benchmark_callable(
            lambda grad_output=grad_output, packed_weight=packed_weight, quant_type=quant_type, in_features=case.in_features: (
                torch.ops.torch_ggml_ops.mmq_grad_input.default(
                    grad_output,
                    packed_weight,
                    quant_type,
                    in_features,
                )
            ),
            rows,
            case.out_features,
            case.in_features,
            args.warmup,
            args.repeats,
        )
        del grad_output
    return measurements


def benchmark_reference_rows(
    args: Namespace,
    case: DenseMMQCase,
    case_index: int,
    rows_to_measure: tuple[int, ...],
    packed_weight: torch.Tensor,
    logical_weight: torch.Tensor,
    quant_type: int,
) -> tuple[dict[int, BenchmarkTiming], dict[int, dict[str, object]]]:
    reference_measurements: dict[int, BenchmarkTiming] = {}
    correctness: dict[int, dict[str, object]] = {}
    for row_index, rows in enumerate(rows_to_measure):
        grad_output = make_bf16_input(
            rows,
            case.out_features,
            args.seed + case_index * 1000 + row_index,
        )
        reference_measurements[rows] = benchmark_callable(
            lambda grad_output=grad_output, logical_weight=logical_weight: torch.mm(
                grad_output, logical_weight
            ),
            rows,
            case.out_features,
            case.in_features,
            args.warmup,
            args.repeats,
        )
        correctness_grad_output = grad_output[
            : min(args.correctness_rows, rows)
        ].clone()
        correctness[rows] = correctness_metrics(
            correctness_grad_output,
            packed_weight,
            logical_weight,
            quant_type,
            case.in_features,
        )
        del grad_output, correctness_grad_output
    return reference_measurements, correctness


def benchmark_case(
    args: Namespace,
    case: DenseMMQCase,
    case_index: int,
    tensor: gguf.ReaderTensor,
    lm_head_chunks: tuple[int, ...],
) -> list[dict[str, object]]:
    quant_type, quant_name, physical_shape = validate_weight_case(tensor, case)
    packed_weight = load_packed_tensor(tensor)
    if packed_weight.dtype != torch.uint8 or not packed_weight.is_contiguous():
        raise RuntimeError("packed benchmark weight is not contiguous uint8")

    row_specs, unique_rows = make_row_specs(
        case,
        args.batches,
        args.sequence_length,
        lm_head_chunks,
    )
    packed_by_rows = benchmark_packed_rows(
        args,
        case,
        case_index,
        unique_rows,
        packed_weight,
        quant_type,
    )

    torch.cuda.synchronize()
    logical_weight = dequantize_gguf_tensor(
        packed_weight,
        tensor.tensor_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(case.out_features, case.in_features)
    logical_weight = logical_weight.contiguous()
    reference_by_rows, correctness_by_rows = benchmark_reference_rows(
        args,
        case,
        case_index,
        unique_rows,
        packed_weight,
        logical_weight,
        quant_type,
    )

    results = []
    for row_spec in row_specs:
        rows = row_spec["m"]
        packed = packed_by_rows[rows]
        reference = reference_by_rows[rows]
        model_calls = row_spec["calls"] * case.model_calls
        result = {
            **dense_result_metadata(
                case,
                row_spec,
                args.sequence_length,
                physical_shape,
                quant_name,
                quant_type,
            ),
            "grad_output_shape": [rows, case.out_features],
            "grad_input_shape": [rows, case.in_features],
            "grad_output_dtype": str(torch.bfloat16),
            "packed_weight_dtype": str(torch.uint8),
            "grad_input_dtype": str(torch.bfloat16),
            "packed": packed,
            "bf16_reference": reference,
            **performance_comparison(packed, reference, model_calls),
            "correctness": correctness_by_rows[rows],
        }
        results.append(result)
        print_dense_result(result)

    del logical_weight, packed_weight
    clear_cuda_cache()
    return results


def main() -> None:
    args = parse_dense_benchmark_args(__doc__, DEFAULT_OUTPUT, 20260706)
    device = cuda_device_info()
    cases = select_cases(args.cases, args.primary_only, args.model_family)
    reader, tensors = load_gguf_tensors(
        args.model,
        tuple(case.tensor_name for case in cases),
    )
    lm_head_chunks = resolve_lm_head_chunks(args.lm_head_chunks, args.model_family)
    report = {
        "model": str(args.model),
        "model_family": args.model_family,
        "operation": "backward",
        "device": device,
        "configuration": {
            "sequence_length": args.sequence_length,
            "batches": list(args.batches),
            "lm_head_chunks": list(lm_head_chunks),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "correctness_rows": args.correctness_rows,
            "grad_output_dtype": str(torch.bfloat16),
            "packed_storage_dtype": str(torch.uint8),
            "grad_input_dtype": str(torch.bfloat16),
            "grad_output_quantization": None,
            "reference": "torch.mm(BF16_grad_output, dequantized_BF16_weight)",
        },
        "results": [],
    }
    print_benchmark_header(device, args.model, args.model_family)

    with torch.inference_mode():
        for case_index, case in enumerate(cases):
            report["results"].extend(
                benchmark_case(
                    args,
                    case,
                    case_index,
                    tensors[case.tensor_name],
                    lm_head_chunks,
                )
            )

    del reader
    write_json_report(args.output, report)
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
