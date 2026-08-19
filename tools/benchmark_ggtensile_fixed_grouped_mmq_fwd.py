#!/usr/bin/env python3

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import cast

import gguf
import numpy as np
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ggtensile.benchmark_report import (
    ErrorMetrics,
    error_metrics,
    rotating_timings,
)
from tools.ggtensile.fixed_grouped_mmq_fwd_model import FixedForwardSolutionKey
from tools.ggtensile.runtime import (
    FixedGroupedQ8ForwardModule,
    FixedQ81F32D4QuantizerModule,
    InstalledFixedGroupedQ8ForwardModule,
)

MAX_CONTROL_NORMALIZED_RMSE = 5e-4
MAX_CONTROL_ABSOLUTE_ERROR = 0.015625
MAX_REFERENCE_NORMALIZED_RMSE = 0.04


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one exact fixed-group GGTensile forward artifact"
    )
    parser.add_argument("--solution-key", type=Path, required=True)
    parser.add_argument("--code-object", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--skip-reference", action="store_true")
    return parser


def _load_key(path: Path) -> FixedForwardSolutionKey:
    return FixedForwardSolutionKey.from_mapping(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _load_weight(
    model: Path, tensor_name: str, key: FixedForwardSolutionKey
) -> tuple[gguf.ReaderTensor, torch.Tensor]:
    reader = gguf.GGUFReader(model)
    tensor = next((item for item in reader.tensors if item.name == tensor_name), None)
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {tensor_name}")
    problem = key.problem
    if tensor.tensor_type.name != problem.quant_data_type:
        raise ValueError("fixed tensor quant type does not match the exact key")
    logical_shape = tuple(int(value) for value in reversed(tensor.shape))
    expected_logical = (
        problem.groups * problem.output_features,
        problem.input_features,
    )
    if logical_shape != expected_logical:
        raise ValueError(
            f"fixed logical shape {logical_shape} does not match {expected_logical}"
        )
    host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    expected_packed = (
        problem.groups,
        problem.output_features,
        problem.packed_row_bytes,
    )
    packed = torch.from_numpy(host).view(expected_packed).cuda()
    return tensor, packed


def _require_correctness(correctness: dict[str, object], reference: bool) -> None:
    names = ("candidate_vs_hip_kernel", "candidate_vs_public_complete")
    for name in names:
        metrics = cast(ErrorMetrics, correctness[name])
        if (
            not metrics["finite"]
            or metrics["normalized_rmse"] is None
            or metrics["normalized_rmse"] > MAX_CONTROL_NORMALIZED_RMSE
            or metrics["max_absolute_error"] is None
            or metrics["max_absolute_error"] > MAX_CONTROL_ABSOLUTE_ERROR
        ):
            raise RuntimeError(f"fixed correctness failed: {name}")
    if reference:
        metrics = cast(ErrorMetrics, correctness["candidate_vs_bf16_reference"])
        if (
            metrics["normalized_rmse"] is None
            or metrics["normalized_rmse"] > MAX_REFERENCE_NORMALIZED_RMSE
        ):
            raise RuntimeError("fixed BF16 reference tolerance failed")
    producer = cast(dict[str, int], correctness["producer_repeat"])
    if producer["different_bytes"]:
        raise RuntimeError("fixed Q8_1 producer is not deterministic")


def main() -> None:
    args = _parser().parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    key = _load_key(args.solution_key)
    problem = key.problem
    tensor, packed_weight = _load_weight(args.model, args.tensor, key)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    input_tensor = torch.randn(
        (problem.tokens, problem.groups, problem.input_features),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    flat_input = input_tensor.view(
        problem.total_activation_rows, problem.input_features
    )
    candidate_output = torch.empty(
        (problem.tokens, problem.groups, problem.output_features),
        dtype=torch.bfloat16,
        device="cuda",
    )
    hip_output = torch.empty_like(candidate_output)
    stream = torch.cuda.current_stream().cuda_stream

    with contextlib.ExitStack() as stack:
        quantizer = stack.enter_context(FixedQ81F32D4QuantizerModule())
        workspace = quantizer.allocate(flat_input)
        candidate = stack.enter_context(
            FixedGroupedQ8ForwardModule(key, args.code_object)
        )
        hip = stack.enter_context(InstalledFixedGroupedQ8ForwardModule(key))

        def quantize() -> None:
            quantizer.launch(flat_input, workspace, stream=stream)

        def candidate_kernel() -> None:
            candidate.launch(packed_weight, workspace, candidate_output, stream=stream)

        def hip_kernel() -> None:
            hip.launch(packed_weight, workspace, hip_output, stream=stream)

        def candidate_complete() -> None:
            quantize()
            candidate_kernel()

        def hip_complete() -> None:
            quantize()
            hip_kernel()

        quantize()
        workspace_control = workspace.clone()
        quantize()
        hip_kernel()
        candidate_kernel()
        public_output = torch_ggml_ops.fixed_grouped_mmq(input_tensor, packed_weight)
        torch.cuda.synchronize()
        correctness: dict[str, object] = {
            "producer_repeat": {
                "different_bytes": int(
                    torch.count_nonzero(workspace != workspace_control)
                ),
                "bytes": workspace.numel(),
            },
            "candidate_vs_hip_kernel": error_metrics(candidate_output, hip_output),
            "candidate_vs_public_complete": error_metrics(
                candidate_output, public_output
            ),
        }
        if not args.skip_reference:
            logical_weight = dequantize_gguf_tensor(
                packed_weight,
                tensor.tensor_type,
                dtype=torch.bfloat16,
                device="cuda",
            ).reshape(problem.groups, problem.output_features, problem.input_features)
            reference = (
                torch.bmm(input_tensor.permute(1, 0, 2), logical_weight.transpose(1, 2))
                .permute(1, 0, 2)
                .contiguous()
            )
            correctness["candidate_vs_bf16_reference"] = error_metrics(
                candidate_output, reference
            )

        logical_flops = (
            2
            * problem.tokens
            * problem.groups
            * problem.output_features
            * problem.input_features
        )
        timing = rotating_timings(
            {
                "hip_complete": hip_complete,
                "candidate_complete": candidate_complete,
                "hip_kernel": hip_kernel,
                "candidate_kernel": candidate_kernel,
            },
            warmup=args.warmup,
            repeats=args.repeats,
            logical_flops=logical_flops,
        )

    report = {
        "solution_key": key.to_mapping(),
        "solution_hash": key.hash,
        "kernel_name": key.kernel_name,
        "code_object": str(args.code_object),
        "model": str(args.model),
        "tensor": args.tensor,
        "input_shape": list(input_tensor.shape),
        "packed_weight_shape": list(packed_weight.shape),
        "output_shape": list(candidate_output.shape),
        "logical_flops": logical_flops,
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "rotating_order": True,
            "complete_includes_quantization": True,
            "complete_includes_output_allocation": False,
            "kernel_uses_preallocated_output": True,
        },
        "correctness_thresholds": {
            "max_control_normalized_rmse": MAX_CONTROL_NORMALIZED_RMSE,
            "max_control_absolute_error": MAX_CONTROL_ABSOLUTE_ERROR,
            "max_reference_normalized_rmse": MAX_REFERENCE_NORMALIZED_RMSE,
        },
        "correctness": correctness,
        "timing": timing,
        "candidate_complete_to_hip_latency": (
            timing["candidate_complete"]["median_ms"]
            / timing["hip_complete"]["median_ms"]
        ),
        "candidate_complete_to_hip_throughput": (
            timing["candidate_complete"]["median_tflops"]
            / timing["hip_complete"]["median_tflops"]
        ),
        "candidate_kernel_to_hip_latency": (
            timing["candidate_kernel"]["median_ms"] / timing["hip_kernel"]["median_ms"]
        ),
        "candidate_kernel_to_hip_throughput": (
            timing["candidate_kernel"]["median_tflops"]
            / timing["hip_kernel"]["median_tflops"]
        ),
    }
    _require_correctness(correctness, not args.skip_reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
