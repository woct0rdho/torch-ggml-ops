#!/usr/bin/env python3

import argparse
import contextlib
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ggtensile.benchmark_report import (
    ErrorMetrics,
    error_metrics,
    rotating_timings,
)
from tools.ggtensile.fixed_grouped_mmq_bwd_model import FixedBackwardSolutionKey
from tools.ggtensile.runtime import (
    FixedGroupedQ8BackwardModule,
    InstalledFixedGroupedQ8BackwardModule,
)

MAX_CONTROL_NRMSE = 5e-4
MAX_CONTROL_ABS = 0.015625
MAX_REFERENCE_NRMSE = 0.04


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualify and benchmark one fixed-group Q8_0 backward artifact"
    )
    parser.add_argument("--solution-key", type=Path, required=True)
    parser.add_argument("--code-object", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-mutations", action="store_true")
    return parser


def _load_weight(
    path: Path, name: str, key: FixedBackwardSolutionKey
) -> tuple[gguf.ReaderTensor, torch.Tensor]:
    reader = gguf.GGUFReader(path)
    tensor = next((item for item in reader.tensors if item.name == name), None)
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {name}")
    problem = key.problem
    logical = tuple(int(value) for value in reversed(tensor.shape))
    expected = (problem.groups * problem.output_features, problem.input_features)
    if tensor.tensor_type.name != "Q8_0" or logical != expected:
        raise ValueError("GGUF tensor does not match the fixed Q8_0 problem")
    host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    return tensor, torch.from_numpy(host).view(
        problem.groups, problem.output_features, problem.packed_row_bytes
    ).cuda()


def _changed_groups(actual: torch.Tensor, baseline: torch.Tensor) -> list[int]:
    return [
        int(torch.count_nonzero(actual[:, group] != baseline[:, group]))
        for group in range(actual.shape[1])
    ]


def _mutation_checks(
    launch,
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    output: torch.Tensor,
    baseline: torch.Tensor,
) -> dict[str, list[list[int]]]:
    result: dict[str, list[list[int]]] = {"grad_output": [], "packed_weight": []}
    for group in range(8):
        saved = grad_output[0, group].clone()
        grad_output[0, group].zero_()
        launch(output)
        grad_output[0, group].copy_(saved)
        changed = _changed_groups(output, baseline)
        if changed[group] == 0 or any(
            changed[index] for index in range(8) if index != group
        ):
            raise RuntimeError(
                f"grad_output group-isolation mutation failed for {group}"
            )
        result["grad_output"].append(changed)
    for group in range(8):
        saved = packed_weight[group, 0].clone()
        packed_weight[group, 0].zero_()
        launch(output)
        packed_weight[group, 0].copy_(saved)
        changed = _changed_groups(output, baseline)
        if changed[group] == 0 or any(
            changed[index] for index in range(8) if index != group
        ):
            raise RuntimeError(
                f"packed_weight group-isolation mutation failed for {group}"
            )
        result["packed_weight"].append(changed)
    return result


def _require_metrics(metrics: ErrorMetrics, *, reference: bool = False) -> None:
    limit = MAX_REFERENCE_NRMSE if reference else MAX_CONTROL_NRMSE
    nrmse = metrics["normalized_rmse"]
    if not metrics["finite"] or nrmse is None or nrmse > limit:
        raise RuntimeError("fixed backward correctness threshold failed")
    absolute = metrics["max_absolute_error"]
    if not reference and (absolute is None or absolute > MAX_CONTROL_ABS):
        raise RuntimeError("fixed backward absolute-error threshold failed")


def main() -> None:
    args = _parser().parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    key = FixedBackwardSolutionKey.from_mapping(
        json.loads(args.solution_key.read_text(encoding="utf-8"))
    )
    problem = key.problem
    tensor, packed_weight = _load_weight(args.model, args.tensor, key)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    grad_output = torch.randn(
        problem.tokens,
        problem.groups,
        problem.output_features,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    candidate_output = torch.full(
        (problem.tokens, problem.groups, problem.input_features),
        torch.nan,
        dtype=torch.bfloat16,
        device="cuda",
    )
    control_output = torch.empty_like(candidate_output)
    stream = int(torch.cuda.current_stream().cuda_stream)
    with contextlib.ExitStack() as stack:
        candidate = stack.enter_context(
            FixedGroupedQ8BackwardModule(key, args.code_object)
        )
        control = stack.enter_context(InstalledFixedGroupedQ8BackwardModule(key))

        def candidate_kernel(output: torch.Tensor = candidate_output) -> None:
            candidate.launch(grad_output, packed_weight, output, stream=stream)

        def control_kernel() -> None:
            control.launch(grad_output, packed_weight, control_output, stream=stream)

        candidate_kernel()
        control_kernel()
        torch.cuda.synchronize()
        if int(torch.count_nonzero(~torch.isfinite(candidate_output))):
            raise RuntimeError("candidate left non-finite or poisoned destination rows")
        baseline = candidate_output.clone()
        candidate_kernel()
        torch.cuda.synchronize()
        repeat_differences = int(torch.count_nonzero(candidate_output != baseline))
        if repeat_differences:
            raise RuntimeError("candidate output is not deterministic")
        control_metrics = error_metrics(candidate_output, control_output)
        _require_metrics(control_metrics)
        mutations = (
            None
            if args.skip_mutations
            else _mutation_checks(
                candidate_kernel, grad_output, packed_weight, candidate_output, baseline
            )
        )
        candidate_kernel()
        reference_metrics = None
        if not args.skip_reference:
            weight = dequantize_gguf_tensor(
                packed_weight,
                tensor.tensor_type,
                dtype=torch.bfloat16,
                device="cuda",
            ).reshape(8, 1024, 4096)
            rows = min(problem.tokens, 16)
            reference = (
                torch.bmm(grad_output[:rows].permute(1, 0, 2), weight)
                .permute(1, 0, 2)
                .contiguous()
            )
            reference_metrics = error_metrics(candidate_output[:rows], reference)
            _require_metrics(reference_metrics, reference=True)
        logical_flops = 2 * problem.tokens * 8 * 1024 * 4096
        timing = rotating_timings(
            {"candidate_kernel": candidate_kernel, "hip_kernel": control_kernel},
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
        "grad_output_shape": list(grad_output.shape),
        "packed_weight_shape": list(packed_weight.shape),
        "grad_input_shape": list(candidate_output.shape),
        "logical_flops": logical_flops,
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "rotating_order": True,
        },
        "correctness": {
            "candidate_vs_hip": control_metrics,
            "candidate_vs_reference_sample": reference_metrics,
            "repeat_differences": repeat_differences,
            "group_mutations": mutations,
        },
        "timing": timing,
        "candidate_to_hip_latency": timing["candidate_kernel"]["median_ms"]
        / timing["hip_kernel"]["median_ms"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
