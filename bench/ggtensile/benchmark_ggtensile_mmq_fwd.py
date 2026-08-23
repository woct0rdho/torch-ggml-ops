#!/usr/bin/env python3
"""Benchmark one exact ordinary-forward deployment key."""

import argparse
import contextlib
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "bench"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_report import error_metrics, rotating_timings
from benchmark_support import require_case

import torch_ggml_ops
from tools.ggtensile.model import SolutionKey
from tools.ggtensile.runtime import (
    FixedHipForwardModule,
    FixedQ81F16D2S6QuantizerModule,
    FixedQ81F16D4S4QuantizerModule,
    FixedQ81F32D4QuantizerModule,
    ForwardModule,
)
from tools.mmq_correctness import bf16_forward_reference


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solution-key", type=Path, required=True)
    p.add_argument("--code-object", type=Path, required=True)
    p.add_argument("--hip-code-object", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tensor", required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=9)
    p.add_argument("--seed", type=int, default=20260802)
    p.add_argument("--skip-reference", action="store_true")
    p.add_argument("--skip-timing", action="store_true")
    return p


def load_weight(model: Path, name: str, key: SolutionKey):
    reader = gguf.GGUFReader(model)
    tensor = next((item for item in reader.tensors if item.name == name), None)
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {name}")
    size = key.problem_size
    if tensor.tensor_type.name != key.problem_type.quant_data_type:
        raise ValueError("tensor quant type does not match the exact key")
    logical = tuple(int(value) for value in reversed(tensor.shape))
    if logical != (size.n, size.k):
        raise ValueError(f"tensor shape {logical} does not match {(size.n, size.k)}")
    host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    return tensor, torch.from_numpy(host).cuda()


def quantizer_type(quant_type: str):
    if quant_type == "Q2_K":
        return FixedQ81F16D2S6QuantizerModule
    if quant_type in {"Q4_K", "Q5_K"}:
        return FixedQ81F16D4S4QuantizerModule
    return FixedQ81F32D4QuantizerModule


def main() -> None:
    args = parser().parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    key = SolutionKey.from_json_file(args.solution_key)
    if key.problem_type.operation_type != "MMQForward":
        raise ValueError("solution key is not ordinary forward")
    case = require_case("OrdinaryForward", key, args.code_object, (args.tensor,))
    tensor, packed = load_weight(args.model, args.tensor, key)
    size = key.problem_size
    rng = torch.Generator(device="cuda").manual_seed(args.seed)
    input_tensor = torch.randn(
        size.m, size.k, dtype=torch.bfloat16, device="cuda", generator=rng
    )
    candidate_output = torch.empty(
        (size.m, size.n), dtype=torch.bfloat16, device="cuda"
    )
    hip_output = torch.empty_like(candidate_output)
    stream = torch.cuda.current_stream().cuda_stream
    quant_type = int(tensor.tensor_type)
    logical = None
    hip_reason = None

    with contextlib.ExitStack() as stack:
        quantizer = stack.enter_context(
            quantizer_type(key.problem_type.quant_data_type)()
        )
        workspace = quantizer.allocate(input_tensor)
        candidate = stack.enter_context(ForwardModule(key, args.code_object))
        hip = stack.enter_context(FixedHipForwardModule(key, args.hip_code_object))

        def quantize() -> None:
            quantizer.launch(input_tensor, workspace, stream=stream)

        def ggtensile_kernel() -> None:
            candidate.launch(packed, workspace, candidate_output, stream=stream)

        def ggtensile_complete() -> None:
            quantize()
            ggtensile_kernel()

        def hip_kernel() -> None:
            if hip is not None:
                hip.launch(packed, workspace, hip_output, stream=stream)

        def hip_complete() -> None:
            quantize()
            hip_kernel()

        def public_api() -> torch.Tensor:
            return torch_ggml_ops.mmq(input_tensor, packed, quant_type, size.n)

        quantize()
        correctness: dict[str, object] = {}
        ggtensile_kernel()
        public_output = public_api()
        if hip is not None:
            hip_kernel()
        if not args.skip_reference:
            logical = (
                dequantize_gguf_tensor(
                    packed, tensor.tensor_type, dtype=torch.bfloat16, device="cuda"
                )
                .reshape(size.n, size.k)
                .contiguous()
            )
            baseline_output = bf16_forward_reference(input_tensor, logical)
        else:
            baseline_output = None
        torch.cuda.synchronize()
        correctness["ggtensile_vs_public_api"] = error_metrics(
            candidate_output, public_output
        )
        if baseline_output is not None:
            correctness["public_api_vs_bf16_baseline"] = error_metrics(
                public_output, baseline_output
            )
            correctness["ggtensile_vs_bf16_baseline"] = error_metrics(
                candidate_output, baseline_output
            )
        if hip is not None:
            correctness["ggtensile_vs_hip"] = error_metrics(
                candidate_output, hip_output
            )
            correctness["public_api_vs_hip"] = error_metrics(public_output, hip_output)
        has_baseline = baseline_output is not None
        del public_output, baseline_output
        timing = {}
        if not args.skip_timing:
            quantize()
            functions = {
                "public_api": public_api,
                "ggtensile_complete": ggtensile_complete,
                "ggtensile_kernel": ggtensile_kernel,
            }
            if has_baseline:
                assert logical is not None
                baseline_logical = logical
                functions["bf16_baseline"] = lambda: bf16_forward_reference(
                    input_tensor, baseline_logical
                )
            if hip is not None:
                functions["hip_complete"] = hip_complete
                functions["hip_kernel"] = hip_kernel
            timing = rotating_timings(
                functions,
                warmup=args.warmup,
                repeats=args.repeats,
                logical_flops=2 * size.m * size.n * size.k,
            )

    report = {
        **case.to_mapping(),
        "solution_key": key.to_mapping(),
        "solution_hash": key.hash,
        "kernel_name": key.kernel_name,
        "model": str(args.model),
        "tensor": args.tensor,
        "input_shape": [size.m, size.k],
        "output_shape": [size.m, size.n],
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "correctness_before_timing": True,
        },
        "implementations": {
            "public_api": {"available": True, "label": "torch_ggml_ops.mmq"},
            "ggtensile": {"available": True, "artifact": str(args.code_object)},
            "hip": {"available": hip is not None, "reason": hip_reason},
            "bf16_baseline": {
                "available": has_baseline,
                "label": "torch.mm on dequantized BF16 weight",
            },
        },
        "correctness": correctness,
        "timing": timing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
