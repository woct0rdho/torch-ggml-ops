#!/usr/bin/env python3
"""Benchmark production dense MMQ forward shapes against BF16 hipBLASLt.

The benchmark reads representative packed tensors directly from the Qwen3.6
or DeepSeek-V4-Flash GGUF checkpoint. Ordinary projections are measured at
sequence length 2048 and batch sizes 1, 4, and 16. The packed LM head is
measured at bounded token chunk sizes because production training never retains
full-sequence logits.

MMQ uses BF16 input, an internal Q8_1 activation workspace, packed uint8 GGUF
weights, and BF16 output. The reference uses the same BF16 input and the same
logical GGUF weight dequantized to BF16, evaluated by torch.mm. On ROCm this is
normally dispatched to hipBLASLt.
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
    synchronize,
    validate_weight_case,
    write_json_report,
)
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops
from tests.mmq_test_support import load_packed_tensor

DEFAULT_OUTPUT = Path("/tmp/torch_ggml_ops_mmq_fwd_benchmark.json")


class TransientTiming(BenchmarkTiming):
    workspace_bytes: int
    decode_compute_included: bool


def correctness_metrics(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    logical_weight: torch.Tensor,
    quant_type: int,
    out_features: int,
) -> dict[str, object]:
    with torch.inference_mode():
        actual = torch_ggml_ops.mmq(input, packed_weight, quant_type, out_features)
        expected = torch.mm(input, logical_weight.transpose(0, 1))
    return {"rows": input.shape[0], **error_metrics(actual, expected)}


def benchmark_transient_bf16_floor(
    input: torch.Tensor,
    logical_weight: torch.Tensor,
    warmup: int,
    repeats: int,
) -> TransientTiming:
    def function() -> torch.Tensor:
        transient_weight = torch.empty_like(logical_weight)
        transient_weight.copy_(logical_weight)
        return torch.mm(input, transient_weight.transpose(0, 1))

    return {
        **benchmark_callable(
            function,
            input.shape[0],
            logical_weight.shape[0],
            logical_weight.shape[1],
            warmup,
            repeats,
        ),
        "workspace_bytes": logical_weight.numel() * logical_weight.element_size(),
        "decode_compute_included": False,
    }


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
        input = make_bf16_input(
            rows,
            case.in_features,
            args.seed + case_index * 1000 + row_index,
        )
        measurements[rows] = benchmark_callable(
            lambda input=input, packed_weight=packed_weight, quant_type=quant_type, out_features=case.out_features: (
                torch_ggml_ops.mmq(input, packed_weight, quant_type, out_features)
            ),
            rows,
            case.out_features,
            case.in_features,
            args.warmup,
            args.repeats,
        )
        del input
    return measurements


def benchmark_reference_rows(
    args: Namespace,
    case: DenseMMQCase,
    case_index: int,
    rows_to_measure: tuple[int, ...],
    packed_weight: torch.Tensor,
    logical_weight: torch.Tensor,
    quant_type: int,
) -> tuple[
    dict[int, BenchmarkTiming],
    dict[int, TransientTiming],
    dict[int, dict[str, object]],
]:
    reference_measurements: dict[int, BenchmarkTiming] = {}
    transient_measurements: dict[int, TransientTiming] = {}
    correctness: dict[int, dict[str, object]] = {}
    transposed_weight = logical_weight.transpose(0, 1)
    for row_index, rows in enumerate(rows_to_measure):
        input = make_bf16_input(
            rows,
            case.in_features,
            args.seed + case_index * 1000 + row_index,
        )
        reference_measurements[rows] = benchmark_callable(
            lambda input=input, transposed_weight=transposed_weight: torch.mm(
                input, transposed_weight
            ),
            rows,
            case.out_features,
            case.in_features,
            args.warmup,
            args.repeats,
        )
        if args.transient_bf16_control:
            transient_measurements[rows] = benchmark_transient_bf16_floor(
                input,
                logical_weight,
                args.warmup,
                args.repeats,
            )
        correctness_input = input[: min(args.correctness_rows, rows)].clone()
        correctness[rows] = correctness_metrics(
            correctness_input,
            packed_weight,
            logical_weight,
            quant_type,
            case.out_features,
        )
        del input, correctness_input
    return reference_measurements, transient_measurements, correctness


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

    synchronize()
    logical_weight = dequantize_gguf_tensor(
        packed_weight,
        tensor.tensor_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(case.out_features, case.in_features)
    logical_weight = logical_weight.contiguous()
    reference_by_rows, transient_by_rows, correctness_by_rows = (
        benchmark_reference_rows(
            args,
            case,
            case_index,
            unique_rows,
            packed_weight,
            logical_weight,
            quant_type,
        )
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
            "input_shape": [rows, case.in_features],
            "output_shape": [rows, case.out_features],
            "input_dtype": str(torch.bfloat16),
            "packed_weight_dtype": str(torch.uint8),
            "output_dtype": str(torch.bfloat16),
            "packed": packed,
            "bf16_reference": reference,
            **performance_comparison(packed, reference, model_calls),
            "correctness": correctness_by_rows[rows],
        }
        if args.transient_bf16_control:
            transient = transient_by_rows[rows]
            result.update(
                {
                    "transient_bf16_materialization_floor": transient,
                    "transient_bf16_workspace_bytes": transient["workspace_bytes"],
                    "packed_to_transient_bf16_floor_latency_ratio": (
                        packed["median_ms"] / transient["median_ms"]
                    ),
                }
            )
        results.append(result)
        print_dense_result(result)

    del logical_weight, packed_weight
    clear_cuda_cache()
    return results


def main() -> None:
    args = parse_dense_benchmark_args(
        __doc__,
        DEFAULT_OUTPUT,
        20260705,
        transient_bf16_control=True,
    )
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
        "operation": "forward",
        "device": device,
        "configuration": {
            "sequence_length": args.sequence_length,
            "batches": list(args.batches),
            "lm_head_chunks": list(lm_head_chunks),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "correctness_rows": args.correctness_rows,
            "input_dtype": str(torch.bfloat16),
            "packed_storage_dtype": str(torch.uint8),
            "output_dtype": str(torch.bfloat16),
            "activation_quantization": "Q8_1",
            "reference": "torch.mm(BF16, dequantized_BF16_weight.T)",
            "transient_bf16_control": args.transient_bf16_control,
            "transient_bf16_control_scope": (
                "allocate + copy predecoded BF16 weight + torch.mm; decode excluded"
                if args.transient_bf16_control
                else None
            ),
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
