#!/usr/bin/env python3

import argparse
import contextlib
import json
import statistics
import sys
from pathlib import Path
from typing import TypedDict, cast

import gguf
import numpy as np
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ggtensile.model import SolutionKey
from tools.ggtensile.runtime import (
    FixedHipForwardModule,
    FixedQ81F16D4S4QuantizerModule,
    FixedQ81F32D4QuantizerModule,
    ForwardModule,
)

DEFAULT_MODEL = Path.home() / "models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"
MAX_CANDIDATE_TO_HIP_NORMALIZED_RMSE = 5e-4
MAX_CANDIDATE_TO_HIP_ABSOLUTE_ERROR = 0.015625
MAX_INDEPENDENT_NORMALIZED_RMSE = 0.04


class Metrics(TypedDict):
    different_bf16_elements: int
    elements: int
    finite: bool
    max_absolute_error: float | None
    error_rms: float | None
    reference_rms: float
    normalized_rmse: float | None


class Timing(TypedDict):
    samples_ms: list[float]
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    median_tflops: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and benchmark one exact K-quant GGTensile forward artifact"
    )
    parser.add_argument("--solution-key", type=Path, required=True)
    parser.add_argument("--code-object", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--skip-reference", action="store_true")
    return parser


def _event_time(function) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> Metrics:
    difference = actual.float() - expected.float()
    finite = bool(torch.isfinite(difference).all())
    reference_rms = float(expected.float().square().mean().sqrt())
    error_rms = float(difference.square().mean().sqrt()) if finite else None
    return {
        "different_bf16_elements": int(torch.count_nonzero(actual != expected)),
        "elements": actual.numel(),
        "finite": finite,
        "max_absolute_error": float(difference.abs().max()) if finite else None,
        "error_rms": error_rms,
        "reference_rms": reference_rms,
        "normalized_rmse": (
            error_rms / reference_rms
            if error_rms is not None and reference_rms
            else None
        ),
    }


def _timing(samples: list[float], logical_flops: int) -> Timing:
    median_ms = statistics.median(samples)
    return {
        "samples_ms": samples,
        "median_ms": median_ms,
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "median_tflops": logical_flops / (median_ms * 1.0e9),
    }


def _load_weight(
    model: Path,
    tensor_name: str,
    key: SolutionKey,
) -> tuple[gguf.ReaderTensor, torch.Tensor]:
    reader = gguf.GGUFReader(model)
    tensor = next((item for item in reader.tensors if item.name == tensor_name), None)
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {tensor_name}")
    expected_quant_type = key.problem_type.quant_data_type
    if tensor.tensor_type.name != expected_quant_type:
        raise ValueError(
            f"expected {expected_quant_type} tensor, found {tensor.tensor_type.name}"
        )
    size = key.problem_size
    logical_shape = tuple(int(value) for value in reversed(tensor.shape))
    if logical_shape != (size.n, size.k):
        raise ValueError(
            f"tensor shape {logical_shape} does not match N,K={(size.n, size.k)}"
        )
    packed_host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    packed_weight = torch.from_numpy(packed_host).cuda()
    return tensor, packed_weight


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.warmup < 0 or arguments.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    key = SolutionKey.from_json_file(arguments.solution_key)
    if key.problem_type.operation_type != "MMQForward":
        raise ValueError("solution key is not MMQ forward")
    size = key.problem_size
    tensor, packed_weight = _load_weight(arguments.model, arguments.tensor, key)
    generator = torch.Generator(device="cuda").manual_seed(arguments.seed)
    input_tensor = torch.randn(
        size.m,
        size.k,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    candidate_output = torch.empty(
        (size.m, size.n), dtype=torch.bfloat16, device="cuda"
    )
    hip_output = torch.empty_like(candidate_output)
    stream = torch.cuda.current_stream().cuda_stream
    quant_type = int(tensor.tensor_type)

    with contextlib.ExitStack() as stack:
        quantizer_type = (
            FixedQ81F32D4QuantizerModule
            if key.problem_type.quant_data_type in ("Q3_K", "Q6_K", "Q8_0")
            else FixedQ81F16D4S4QuantizerModule
        )
        quantizer = stack.enter_context(quantizer_type())
        workspace = quantizer.allocate(input_tensor)
        candidate = stack.enter_context(ForwardModule(key, arguments.code_object))
        hip_multiply = stack.enter_context(FixedHipForwardModule(key))

        def quantize() -> None:
            quantizer.launch(input_tensor, workspace, stream=stream)

        def launch_candidate() -> None:
            candidate.launch(
                packed_weight,
                workspace,
                candidate_output,
                stream=stream,
            )

        def launch_hip() -> None:
            hip_multiply.launch(
                packed_weight,
                workspace,
                hip_output,
                stream=stream,
            )

        def candidate_complete() -> None:
            quantize()
            launch_candidate()

        def hip_complete() -> None:
            quantize()
            launch_hip()

        quantize()
        workspace_control = workspace.clone()
        quantize()
        torch.cuda.synchronize()
        correctness: dict[str, object] = {
            "producer_repeat": {
                "different_bytes": int(
                    torch.count_nonzero(workspace != workspace_control)
                ),
                "bytes": workspace.numel(),
            }
        }
        launch_hip()
        launch_candidate()
        public_output = torch_ggml_ops.mmq(
            input_tensor,
            packed_weight,
            quant_type,
            size.n,
        )
        torch.cuda.synchronize()
        baseline_candidate = candidate_output.clone()
        correctness["candidate_vs_hip_kernel"] = _metrics(candidate_output, hip_output)
        correctness["candidate_vs_public_complete"] = _metrics(
            candidate_output, public_output
        )

        input_tensor.neg_()
        quantize()
        launch_hip()
        launch_candidate()
        updated_public = torch_ggml_ops.mmq(
            input_tensor,
            packed_weight,
            quant_type,
            size.n,
        )
        torch.cuda.synchronize()
        correctness["input_mutation_vs_hip_kernel"] = _metrics(
            candidate_output, hip_output
        )
        correctness["input_mutation_vs_public_complete"] = _metrics(
            candidate_output, updated_public
        )
        correctness["input_mutation_changed_elements"] = int(
            torch.count_nonzero(candidate_output != baseline_candidate)
        )
        input_tensor.neg_()

        packed_bytes = packed_weight.view(-1)
        packed_original = packed_bytes[16].clone()
        packed_bytes[16].bitwise_xor_(1)
        quantize()
        launch_hip()
        launch_candidate()
        torch.cuda.synchronize()
        correctness["packed_weight_mutation_vs_hip_kernel"] = _metrics(
            candidate_output, hip_output
        )
        correctness["packed_weight_mutation_changed_elements"] = int(
            torch.count_nonzero(candidate_output != baseline_candidate)
        )
        packed_bytes[16].copy_(packed_original)

        quantize()
        workspace_bytes = workspace.view(-1)
        workspace_index = 16
        workspace_original = workspace_bytes[workspace_index].clone()
        workspace_bytes[workspace_index].bitwise_xor_(1)
        launch_hip()
        launch_candidate()
        torch.cuda.synchronize()
        correctness["workspace_mutation_vs_hip_kernel"] = _metrics(
            candidate_output, hip_output
        )
        correctness["workspace_mutation_changed_elements"] = int(
            torch.count_nonzero(candidate_output != baseline_candidate)
        )
        workspace_bytes[workspace_index].copy_(workspace_original)

        quantize()
        launch_candidate()
        torch.cuda.synchronize()
        if not arguments.skip_reference:
            logical_weight = dequantize_gguf_tensor(
                packed_weight,
                tensor.tensor_type,
                dtype=torch.bfloat16,
                device="cuda",
            ).reshape(size.n, size.k)
            reference = torch.mm(input_tensor, logical_weight.transpose(0, 1))
            correctness["candidate_vs_independent_reference"] = _metrics(
                candidate_output,
                reference,
            )
            correctness["public_vs_independent_reference"] = _metrics(
                public_output,
                reference,
            )
            del logical_weight, reference

        timing_functions = {
            "hip_complete": hip_complete,
            "candidate_complete": candidate_complete,
            "hip_kernel": launch_hip,
            "candidate_kernel": launch_candidate,
        }
        for _ in range(arguments.warmup):
            for function in timing_functions.values():
                function()
        torch.cuda.synchronize()
        samples = {name: [] for name in timing_functions}
        names = list(timing_functions)
        for repeat in range(arguments.repeats):
            offset = repeat % len(names)
            for name in names[offset:] + names[:offset]:
                samples[name].append(_event_time(timing_functions[name]))

    logical_flops = 2 * size.m * size.n * size.k
    timing = {name: _timing(values, logical_flops) for name, values in samples.items()}
    report = {
        "solution_key": key.to_mapping(),
        "solution_hash": key.hash,
        "kernel_name": key.kernel_name,
        "code_object": str(arguments.code_object),
        "model": str(arguments.model),
        "tensor": arguments.tensor,
        "logical_weight_shape": [size.n, size.k],
        "physical_weight_shape": list(tensor.data.shape),
        "input_shape": [size.m, size.k],
        "output_shape": [size.m, size.n],
        "workspace_shape": [size.k // 128, size.m, 144],
        "logical_flops": logical_flops,
        "protocol": {
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
            "rotating_order": True,
        },
        "correctness_thresholds": {
            "max_candidate_to_hip_normalized_rmse": (
                MAX_CANDIDATE_TO_HIP_NORMALIZED_RMSE
            ),
            "max_candidate_to_hip_absolute_error": (
                MAX_CANDIDATE_TO_HIP_ABSOLUTE_ERROR
            ),
            "max_independent_normalized_rmse": MAX_INDEPENDENT_NORMALIZED_RMSE,
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
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))

    hip_agreement_names = (
        "candidate_vs_hip_kernel",
        "candidate_vs_public_complete",
        "input_mutation_vs_hip_kernel",
        "input_mutation_vs_public_complete",
        "packed_weight_mutation_vs_hip_kernel",
        "workspace_mutation_vs_hip_kernel",
    )
    hip_agreement = tuple(
        cast(Metrics, correctness[name]) for name in hip_agreement_names
    )
    if any(
        not metrics["finite"]
        or metrics["normalized_rmse"] is None
        or metrics["normalized_rmse"] > MAX_CANDIDATE_TO_HIP_NORMALIZED_RMSE
        or metrics["max_absolute_error"] is None
        or metrics["max_absolute_error"] > MAX_CANDIDATE_TO_HIP_ABSOLUTE_ERROR
        for metrics in hip_agreement
    ):
        raise SystemExit("candidate failed forward correctness tolerance")
    if not arguments.skip_reference:
        independent_metrics = cast(
            Metrics, correctness["candidate_vs_independent_reference"]
        )
        independent_nrmse = independent_metrics["normalized_rmse"]
        if (
            independent_nrmse is None
            or independent_nrmse > MAX_INDEPENDENT_NORMALIZED_RMSE
        ):
            raise SystemExit("candidate failed independent-reference tolerance")
    producer_repeat = correctness["producer_repeat"]
    if producer_repeat["different_bytes"] != 0:
        raise SystemExit("fixed Q8_1 F16_D4S4 producer is not deterministic")
    if correctness["input_mutation_changed_elements"] == 0:
        raise SystemExit("input mutation did not affect output")
    if correctness["packed_weight_mutation_changed_elements"] == 0:
        raise SystemExit("packed-weight mutation did not affect output")
    if correctness["workspace_mutation_changed_elements"] == 0:
        raise SystemExit("workspace mutation did not affect output")


if __name__ == "__main__":
    main()
