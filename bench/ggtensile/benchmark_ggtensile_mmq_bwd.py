#!/usr/bin/env python3
"""Benchmark one exact ordinary-backward deployment key."""

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
from benchmark_support import public_dense_backward, require_case

from tools.ggtensile.dense_mmq_bwd_runtime import InstalledDenseBackwardModule
from tools.ggtensile.model import SolutionKey
from tools.ggtensile.runtime import BackwardModule
from tools.mmq_correctness import bf16_backward_reference


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
    tensor = next(
        (item for item in gguf.GGUFReader(model).tensors if item.name == name), None
    )
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {name}")
    size = key.problem_size
    if tensor.tensor_type.name != key.problem_type.quant_data_type:
        raise ValueError("tensor quant type does not match the exact key")
    logical = tuple(int(value) for value in reversed(tensor.shape))
    if logical != (size.k, size.n):
        raise ValueError(f"tensor shape {logical} does not match {(size.k, size.n)}")
    host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    return tensor, torch.from_numpy(host).cuda()


def main() -> None:
    args = parser().parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    key = SolutionKey.from_json_file(args.solution_key)
    if key.problem_type.operation_type != "MMQBackward":
        raise ValueError("solution key is not ordinary backward")
    case = require_case("OrdinaryBackward", key, args.code_object, (args.tensor,))
    tensor, packed = load_weight(args.model, args.tensor, key)
    size = key.problem_size
    rng = torch.Generator(device="cuda").manual_seed(args.seed)
    grad_output = torch.randn(
        size.m, size.k, dtype=torch.bfloat16, device="cuda", generator=rng
    )
    candidate_output = torch.empty(
        (size.m, size.n), dtype=torch.bfloat16, device="cuda"
    )
    hip_output = torch.empty_like(candidate_output)
    stream = torch.cuda.current_stream().cuda_stream
    quant_type = int(tensor.tensor_type)
    hip = None
    hip_reason = "no legacy HIP control artifact was supplied"

    with contextlib.ExitStack() as stack:
        candidate = stack.enter_context(BackwardModule(key, args.code_object))
        if args.hip_code_object is not None:
            hip = stack.enter_context(
                InstalledDenseBackwardModule(key, args.hip_code_object)
            )
            hip_reason = None

        def ggtensile() -> torch.Tensor:
            candidate.launch(grad_output, packed, candidate_output, stream=stream)
            return candidate_output

        def hip_launch() -> torch.Tensor:
            if hip is None:
                return hip_output
            hip.launch(grad_output, packed, hip_output, stream=stream)
            return hip_output

        def public_api() -> torch.Tensor:
            return public_dense_backward(grad_output, packed, quant_type, size.n)

        if not args.skip_reference:
            logical = (
                dequantize_gguf_tensor(
                    packed, tensor.tensor_type, dtype=torch.bfloat16, device="cuda"
                )
                .reshape(size.k, size.n)
                .contiguous()
            )
            baseline = bf16_backward_reference(grad_output, logical)
        else:
            logical = baseline = None
        public_output = public_api()
        ggtensile()
        if hip is not None:
            hip_launch()
        torch.cuda.synchronize()
        correctness: dict[str, object] = {
            "ggtensile_vs_public_api": error_metrics(candidate_output, public_output)
        }
        if baseline is not None:
            correctness["public_api_vs_bf16_baseline"] = error_metrics(
                public_output, baseline
            )
            correctness["ggtensile_vs_bf16_baseline"] = error_metrics(
                candidate_output, baseline
            )
        if hip is not None:
            correctness["ggtensile_vs_hip"] = error_metrics(
                candidate_output, hip_output
            )
            correctness["public_api_vs_hip"] = error_metrics(public_output, hip_output)
        has_baseline = baseline is not None
        del public_output, baseline
        timing = {}
        if not args.skip_timing:
            functions = {"public_api": public_api, "ggtensile": ggtensile}
            if has_baseline:
                assert logical is not None
                baseline_logical = logical
                functions["bf16_baseline"] = lambda: bf16_backward_reference(
                    grad_output, baseline_logical
                )
            if hip is not None:
                functions["hip"] = hip_launch
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
            "public_api": {"available": True, "label": "autograd(torch_ggml_ops.mmq)"},
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
