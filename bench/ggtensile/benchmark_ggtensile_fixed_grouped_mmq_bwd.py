#!/usr/bin/env python3
"""Benchmark one exact fixed-group backward deployment key."""

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

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "bench"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_report import ErrorMetrics, error_metrics, rotating_timings
from benchmark_support import public_fixed_backward, require_case

from tools.ggtensile.fixed_grouped_mmq_bwd_model import FixedBackwardSolutionKey
from tools.ggtensile.runtime import (
    FixedGroupedQ8BackwardModule,
    HIPRuntimeError,
    InstalledFixedGroupedQ8BackwardModule,
)


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
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--skip-reference", action="store_true")
    p.add_argument("--skip-mutations", action="store_true")
    p.add_argument("--skip-timing", action="store_true")
    return p


def load_weight(model: Path, name: str, key: FixedBackwardSolutionKey):
    tensor = next(
        (item for item in gguf.GGUFReader(model).tensors if item.name == name), None
    )
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {name}")
    problem = key.problem
    logical = tuple(int(value) for value in reversed(tensor.shape))
    expected = (problem.groups * problem.output_features, problem.input_features)
    if tensor.tensor_type.name != "Q8_0" or logical != expected:
        raise ValueError("GGUF tensor does not match the fixed Q8_0 problem")
    host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    packed = (
        torch.from_numpy(host)
        .view(problem.groups, problem.output_features, problem.packed_row_bytes)
        .cuda()
    )
    return tensor, packed


def mutation_checks(launch, grad_output, packed_weight, output) -> dict[str, list[int]]:
    result = {"grad_output": [], "packed_weight": []}
    baseline = output.clone()
    for group in range(grad_output.shape[1]):
        saved = grad_output[0, group].clone()
        grad_output[0, group].zero_()
        launch()
        torch.cuda.synchronize()
        changed = int(torch.count_nonzero(output != baseline))
        grad_output[0, group].copy_(saved)
        if not changed:
            raise RuntimeError(f"grad-output mutation did not affect group {group}")
        result["grad_output"].append(changed)
    launch()
    torch.cuda.synchronize()
    for group in range(packed_weight.shape[0]):
        saved = packed_weight[group, 0].clone()
        packed_weight[group, 0].zero_()
        launch()
        torch.cuda.synchronize()
        changed = int(torch.count_nonzero(output != baseline))
        packed_weight[group, 0].copy_(saved)
        if not changed:
            raise RuntimeError(f"packed-weight mutation did not affect group {group}")
        result["packed_weight"].append(changed)
    return result


def require(metrics: object, label: str, limit: float) -> None:
    typed = cast(ErrorMetrics, metrics)
    if (
        typed["finite"] is not True
        or not isinstance(typed["normalized_rmse"], float)
        or typed["normalized_rmse"] > limit
    ):
        raise RuntimeError(f"{label} correctness failed: {metrics}")


def main() -> None:
    args = parser().parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    key = FixedBackwardSolutionKey.from_mapping(
        json.loads(args.solution_key.read_text(encoding="utf-8"))
    )
    case = require_case("FixedGroupedBackward", key, args.code_object, (args.tensor,))
    tensor, packed = load_weight(args.model, args.tensor, key)
    problem = key.problem
    rng = torch.Generator(device="cuda").manual_seed(args.seed)
    grad_output = torch.randn(
        problem.tokens,
        problem.groups,
        problem.output_features,
        dtype=torch.bfloat16,
        device="cuda",
        generator=rng,
    )
    candidate_output = torch.empty(
        problem.tokens,
        problem.groups,
        problem.input_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    hip_output = torch.empty_like(candidate_output)
    stream = torch.cuda.current_stream().cuda_stream
    hip = None
    hip_reason = "no legacy HIP control artifact was supplied"

    with contextlib.ExitStack() as stack:
        candidate = stack.enter_context(
            FixedGroupedQ8BackwardModule(key, args.code_object)
        )
        if args.hip_code_object is not None:
            try:
                hip = stack.enter_context(
                    InstalledFixedGroupedQ8BackwardModule(key, args.hip_code_object)
                )
                hip_reason = None
            except HIPRuntimeError as error:
                hip_reason = str(error)

        def ggtensile() -> torch.Tensor:
            candidate.launch(grad_output, packed, candidate_output, stream=stream)
            return candidate_output

        def hip_launch() -> torch.Tensor:
            if hip is not None:
                hip.launch(grad_output, packed, hip_output, stream=stream)
            return hip_output

        def public_api() -> torch.Tensor:
            return public_fixed_backward(grad_output, packed, problem.input_features)

        ggtensile()
        public_output = public_api()
        if hip is not None:
            hip_launch()
        torch.cuda.synchronize()
        correctness: dict[str, object] = {
            "ggtensile_vs_public_api": error_metrics(candidate_output, public_output)
        }
        if hip is not None:
            correctness["ggtensile_vs_hip"] = error_metrics(
                candidate_output, hip_output
            )
        repeat = candidate_output.clone()
        ggtensile()
        torch.cuda.synchronize()
        correctness["repeat"] = {
            "different_elements": int(torch.count_nonzero(candidate_output != repeat))
        }
        if correctness["repeat"]["different_elements"]:
            raise RuntimeError("fixed backward output is not deterministic")
        mutations = (
            None
            if args.skip_mutations
            else mutation_checks(ggtensile, grad_output, packed, candidate_output)
        )
        if not args.skip_reference:
            logical = dequantize_gguf_tensor(
                packed, tensor.tensor_type, dtype=torch.bfloat16, device="cuda"
            ).reshape(problem.groups, problem.output_features, problem.input_features)
            rows = min(problem.tokens, 16)
            baseline = (
                torch.bmm(grad_output[:rows].permute(1, 0, 2), logical)
                .permute(1, 0, 2)
                .contiguous()
            )
            correctness["ggtensile_vs_bf16_baseline"] = error_metrics(
                candidate_output[:rows], baseline
            )
        else:
            logical = baseline = None
        require(correctness["ggtensile_vs_public_api"], "GGTensile/public API", 5e-4)
        if hip is not None:
            require(correctness["ggtensile_vs_hip"], "GGTensile/HIP", 5e-4)
        if baseline is not None:
            require(correctness["ggtensile_vs_bf16_baseline"], "GGTensile/BF16", 0.04)

        timing = {}
        if not args.skip_timing:
            functions = {"public_api": public_api, "ggtensile": ggtensile}
            if baseline is not None:
                assert logical is not None
                baseline_logical = logical
                functions["bf16_baseline"] = lambda: (
                    torch.bmm(grad_output[:rows].permute(1, 0, 2), baseline_logical)
                    .permute(1, 0, 2)
                    .contiguous()
                )
            if hip is not None:
                functions["hip"] = hip_launch
            timing = rotating_timings(
                functions,
                warmup=args.warmup,
                repeats=args.repeats,
                logical_flops=2
                * problem.tokens
                * problem.groups
                * problem.output_features
                * problem.input_features,
            )

    report = {
        **case.to_mapping(),
        "solution_key": key.to_mapping(),
        "solution_hash": key.hash,
        "kernel_name": key.kernel_name,
        "model": str(args.model),
        "tensor": args.tensor,
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "correctness_before_timing": True,
        },
        "implementations": {
            "public_api": {
                "available": True,
                "label": "autograd(torch_ggml_ops.fixed_grouped_mmq)",
            },
            "ggtensile": {"available": True, "artifact": str(args.code_object)},
            "hip": {"available": hip is not None, "reason": hip_reason},
            "bf16_baseline": {
                "available": baseline is not None,
                "label": "sampled torch.bmm on dequantized BF16 banks",
            },
        },
        "correctness": {**correctness, "mutations": mutations},
        "timing": timing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
