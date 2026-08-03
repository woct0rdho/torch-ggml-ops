#!/usr/bin/env python3


import argparse
import contextlib
import json
import statistics
import sys
from pathlib import Path
from typing import TypedDict

import gguf
import numpy as np
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops  # noqa: F401 Register the installed HIP control.

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ggtensile.model import SolutionKey
from tools.ggtensile.runtime import BackwardModule

DEFAULT_MODEL = Path.home() / "models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"
DEFAULT_TENSOR = "blk.39.attn_q.weight"
BF16_WMMA_ROOFLINE_TFLOPS = 59.4


class Metrics(TypedDict):
    different_bf16_elements: int
    elements: int
    finite: bool
    max_absolute_error: float | None
    error_rms: float | None
    reference_rms: float
    normalized_rmse: float | None


class TimingSummary(TypedDict):
    samples_ms: list[float]
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    median_tflops: float
    wmma_roofline_fraction: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one exact GGTensile MMQ backward artifact against HIP"
    )
    parser.add_argument("--solution-key", type=Path, required=True)
    parser.add_argument("--code-object", type=Path, required=True)
    parser.add_argument("--assembly-control-solution-key", type=Path)
    parser.add_argument("--assembly-control-code-object", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tensor", default=DEFAULT_TENSOR)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--skip-reference", action="store_true", help="skip independent BF16 matmul"
    )
    return parser


def _event_time(function) -> tuple[float, torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), output


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


def _timing_summary(samples_ms: list[float], logical_flops: int) -> TimingSummary:
    median_ms = statistics.median(samples_ms)
    median_tflops = logical_flops / (median_ms * 1.0e9)
    return {
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "median_tflops": median_tflops,
        "wmma_roofline_fraction": median_tflops / BF16_WMMA_ROOFLINE_TFLOPS,
    }


def main() -> None:
    args = _parser().parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    if (args.assembly_control_solution_key is None) != (
        args.assembly_control_code_object is None
    ):
        raise ValueError("both assembly-control paths must be provided together")
    key = SolutionKey.from_json_file(args.solution_key)
    size = key.problem_size
    assembly_control_key = None
    if args.assembly_control_solution_key is not None:
        assembly_control_key = SolutionKey.from_json_file(
            args.assembly_control_solution_key
        )
        if assembly_control_key.problem_size != size:
            raise ValueError("assembly control and candidate sizes differ")
    reader = gguf.GGUFReader(args.model)
    tensor = next((item for item in reader.tensors if item.name == args.tensor), None)
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {args.tensor}")
    expected_quant_type = key.problem_type.quant_data_type
    if tensor.tensor_type.name != expected_quant_type:
        raise ValueError(
            f"expected {expected_quant_type} tensor, found {tensor.tensor_type.name}"
        )
    logical_shape = tuple(int(value) for value in reversed(tensor.shape))
    if logical_shape != (size.k, size.n):
        raise ValueError(
            f"tensor shape {logical_shape} does not match K,N={(size.k, size.n)}"
        )

    packed_host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    packed_weight = torch.from_numpy(packed_host).cuda()
    del packed_host
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    grad_output = torch.randn(
        size.m,
        size.k,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    candidate_output = torch.empty(
        (size.m, size.n), dtype=torch.bfloat16, device="cuda"
    )
    assembly_control_output = None
    if assembly_control_key is not None:
        assembly_control_output = torch.empty_like(candidate_output)
    quant_type = int(tensor.tensor_type)

    def control():
        return torch.ops.torch_ggml_ops.mmq_grad_input.default(
            grad_output, packed_weight, quant_type, size.n
        )

    with contextlib.ExitStack() as stack:
        candidate = stack.enter_context(BackwardModule(key, args.code_object))
        assembly_control = None
        if assembly_control_key is not None:
            assembly_control = stack.enter_context(
                BackwardModule(assembly_control_key, args.assembly_control_code_object)
            )

        def launch_candidate():
            candidate.launch(
                grad_output,
                packed_weight,
                candidate_output,
                stream=torch.cuda.current_stream().cuda_stream,
            )
            return candidate_output

        def launch_assembly_control():
            assert assembly_control is not None
            assert assembly_control_output is not None
            assembly_control.launch(
                grad_output,
                packed_weight,
                assembly_control_output,
                stream=torch.cuda.current_stream().cuda_stream,
            )
            return assembly_control_output

        functions = {"hip": control, "candidate": launch_candidate}
        if assembly_control is not None:
            functions["assembly_control"] = launch_assembly_control

        for _ in range(args.warmup):
            for function in functions.values():
                function()
        torch.cuda.synchronize()

        control_output = control()
        launch_candidate()
        if assembly_control is not None:
            launch_assembly_control()
        torch.cuda.synchronize()
        correctness: dict[str, Metrics] = {
            "candidate_vs_hip": _metrics(candidate_output, control_output)
        }
        if assembly_control_output is not None:
            correctness["assembly_control_vs_hip"] = _metrics(
                assembly_control_output, control_output
            )

        grad_output.neg_()
        updated_control_output = control()
        launch_candidate()
        torch.cuda.synchronize()
        correctness["candidate_after_grad_output_update_vs_hip"] = _metrics(
            candidate_output, updated_control_output
        )
        grad_output.neg_()
        del updated_control_output

        packed_bytes = packed_weight.view(-1)
        original_packed_byte = packed_bytes[16].clone()
        packed_bytes[16].bitwise_xor_(1)
        updated_control_output = control()
        launch_candidate()
        torch.cuda.synchronize()
        correctness["candidate_after_packed_weight_update_vs_hip"] = _metrics(
            candidate_output, updated_control_output
        )
        packed_bytes[16].copy_(original_packed_byte)
        del original_packed_byte, updated_control_output
        launch_candidate()
        torch.cuda.synchronize()

        if not args.skip_reference:
            logical_weight = (
                dequantize_gguf_tensor(
                    packed_weight,
                    tensor.tensor_type,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                .reshape(size.k, size.n)
                .contiguous()
            )
            reference = torch.mm(grad_output, logical_weight)
            correctness["candidate_vs_reference"] = _metrics(
                candidate_output, reference
            )
            correctness["hip_vs_reference"] = _metrics(control_output, reference)
            del logical_weight, reference

        samples = {name: [] for name in functions}
        names = list(functions)
        for repeat in range(args.repeats):
            offset = repeat % len(names)
            order = names[offset:] + names[:offset]
            for name in order:
                elapsed, output = _event_time(functions[name])
                samples[name].append(elapsed)
                if name == "hip":
                    del output

    logical_flops = 2 * size.m * size.n * size.k
    control_timing = _timing_summary(samples["hip"], logical_flops)
    candidate_timing = _timing_summary(samples["candidate"], logical_flops)
    report = {
        "SolutionKey": key.to_mapping(),
        "SolutionHash": key.hash,
        "KernelName": key.kernel_name,
        "CodeObject": str(args.code_object),
        "Model": str(args.model),
        "Tensor": args.tensor,
        "LogicalWeightShape": [size.k, size.n],
        "PhysicalWeightShape": list(tensor.data.shape),
        "GradOutputShape": [size.m, size.k],
        "GradInputShape": [size.m, size.n],
        "LogicalFlops": logical_flops,
        "Bf16WmmaRooflineTflops": BF16_WMMA_ROOFLINE_TFLOPS,
        "Correctness": correctness,
        "HIP": control_timing,
        "GGTensile": candidate_timing,
        "CandidateToHipLatency": (
            candidate_timing["median_ms"] / control_timing["median_ms"]
        ),
        "CandidateToHipThroughput": (
            candidate_timing["median_tflops"] / control_timing["median_tflops"]
        ),
    }
    if assembly_control_key is not None:
        assembly_control_timing = _timing_summary(
            samples["assembly_control"], logical_flops
        )
        report["AssemblyControl"] = {
            "SolutionKey": assembly_control_key.to_mapping(),
            "SolutionHash": assembly_control_key.hash,
            "KernelName": assembly_control_key.kernel_name,
            "CodeObject": str(args.assembly_control_code_object),
            **assembly_control_timing,
        }
        report["CandidateToAssemblyControlLatency"] = (
            candidate_timing["median_ms"] / assembly_control_timing["median_ms"]
        )
        report["CandidateToAssemblyControlThroughput"] = (
            candidate_timing["median_tflops"] / assembly_control_timing["median_tflops"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    mismatches = {
        name: metrics["different_bf16_elements"]
        for name, metrics in correctness.items()
        if name.endswith("_vs_hip") and metrics["different_bf16_elements"]
    }
    if mismatches:
        raise SystemExit(f"candidate does not match HIP: {mismatches}")


if __name__ == "__main__":
    main()
