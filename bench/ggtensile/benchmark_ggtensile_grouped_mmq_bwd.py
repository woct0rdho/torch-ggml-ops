#!/usr/bin/env python3
"""Benchmark one exact standalone grouped-backward deployment route."""

import argparse
import contextlib
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "bench"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_report import error_metrics, rotating_timings
from benchmark_routes import fitted_prior_distribution_for_rows, make_route_tensors
from benchmark_support import public_grouped_backward, require_case
from workload_prior import EXPERT_PRIOR_NAMES, expert_prior_metadata

from tools.ggtensile.grouped_mmq_bwd_runtime import (
    InstalledGroupedBackwardControl,
)
from tools.ggtensile.model import SolutionKey
from tools.ggtensile.quant_formats import BACKWARD_QUANT_FORMATS
from tools.ggtensile.runtime import GroupedBackwardModule
from tools.mmq_correctness import (
    dequantize_active_routed_weight,
    routed_backward_reference,
)

DEFAULT_TENSORS = {
    "Q4_K": "blk.2.ffn_down_exps.weight",
    "Q5_K": "blk.0.ffn_down_exps.weight",
    "IQ2_S": "blk.10.ffn_down_exps.weight",
    "Q2_K": "blk.0.ffn_down_exps.weight",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solution-key", type=Path, required=True)
    p.add_argument("--code-object", type=Path, required=True)
    p.add_argument("--hip-code-object", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tensor")
    p.add_argument("--expert-prior", choices=EXPERT_PRIOR_NAMES, required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=9)
    p.add_argument("--seed", type=int, default=20260818)
    p.add_argument("--skip-reference", action="store_true")
    p.add_argument("--skip-timing", action="store_true")
    return p


def load_weight(model: Path, name: str, key: SolutionKey):
    tensor = next(
        (item for item in gguf.GGUFReader(model).tensors if item.name == name), None
    )
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {name}")
    quant = key.problem_type.quant_data_type
    if tensor.tensor_type.name != quant:
        raise ValueError("tensor quant type does not match the exact key")
    fmt = BACKWARD_QUANT_FORMATS[quant]
    size = key.problem_size
    expected = (256, size.k, size.n // fmt.block_values * fmt.block_bytes)
    if tuple(int(value) for value in tensor.data.shape) != expected:
        raise ValueError(f"packed tensor shape does not match {expected}")
    host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    return tensor, torch.from_numpy(host).cuda()


def reference(
    grad_output, packed, quant_type, expert_indices, group_sizes, output_features
):
    logical = dequantize_active_routed_weight(
        packed,
        gguf.GGMLQuantizationType(quant_type),
        expert_indices,
        output_features,
        grad_output.shape[1],
    )
    active = torch.arange(group_sizes.numel(), device=grad_output.device)
    return routed_backward_reference(grad_output, logical, active, group_sizes)


def main() -> None:
    args = parser().parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    key = SolutionKey.from_json_file(args.solution_key)
    if key.problem_type.operation_type != "GroupedMMQBackward":
        raise ValueError("solution key is not grouped backward")
    tensor_name = args.tensor or DEFAULT_TENSORS.get(key.problem_type.quant_data_type)
    if tensor_name is None:
        raise ValueError("--tensor is required for this quant type")
    case = require_case("GroupedBackward", key, args.code_object, (tensor_name,))
    tensor, packed = load_weight(args.model, tensor_name, key)
    size = key.problem_size
    distribution = fitted_prior_distribution_for_rows(args.expert_prior, size.m)
    expert_indices, expert_offsets, group_sizes = make_route_tensors(distribution)
    rng = torch.Generator(device="cuda").manual_seed(args.seed)
    grad_output = torch.randn(
        size.m, size.k, dtype=torch.bfloat16, device="cuda", generator=rng
    )
    candidate_output = torch.empty(size.m, size.n, dtype=torch.bfloat16, device="cuda")
    hip_output = torch.empty_like(candidate_output)
    stream = torch.cuda.current_stream().cuda_stream
    hip = None
    hip_reason = "no HIP control artifact was found"

    with contextlib.ExitStack() as stack:
        candidate = stack.enter_context(GroupedBackwardModule(key, args.code_object))
        if args.hip_code_object is not None:
            hip = stack.enter_context(
                InstalledGroupedBackwardControl(
                    key.problem_type.quant_data_type,
                    size.m,
                    expert_indices.numel(),
                    args.hip_code_object,
                )
            )
            hip_reason = None
        else:
            hip = stack.enter_context(
                InstalledGroupedBackwardControl(
                    key.problem_type.quant_data_type,
                    size.m,
                    expert_indices.numel(),
                )
            )
            hip_reason = None

        def ggtensile() -> torch.Tensor:
            candidate.launch(
                grad_output,
                packed,
                candidate_output,
                expert_indices,
                expert_offsets,
                stream=stream,
            )
            return candidate_output

        def hip_launch() -> torch.Tensor:
            if hip is not None:
                hip.launch(
                    grad_output,
                    packed,
                    hip_output,
                    expert_indices,
                    expert_offsets,
                    stream=stream,
                )
            return hip_output

        def public_api() -> torch.Tensor:
            return public_grouped_backward(
                grad_output,
                packed,
                expert_indices,
                expert_offsets,
                int(tensor.tensor_type),
                size.n,
            )

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
        if not args.skip_reference:
            baseline = reference(
                grad_output,
                packed,
                int(tensor.tensor_type),
                expert_indices,
                group_sizes,
                size.n,
            )
            correctness["ggtensile_vs_bf16_baseline"] = error_metrics(
                candidate_output, baseline
            )
        else:
            baseline = None
        has_baseline = baseline is not None
        del public_output, baseline
        timing = {}
        if not args.skip_timing:
            functions = {"public_api": public_api, "ggtensile": ggtensile}
            if has_baseline:
                functions["bf16_baseline"] = lambda: reference(
                    grad_output,
                    packed,
                    int(tensor.tensor_type),
                    expert_indices,
                    group_sizes,
                    size.n,
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
        "tensor": tensor_name,
        "expert_prior": expert_prior_metadata(args.expert_prior),
        "expert_prior_profile": distribution.profile.to_mapping()
        if distribution.profile
        else None,
        "route": {
            "expert_indices": list(distribution.expert_indices_cpu),
            "group_sizes": list(distribution.group_sizes_cpu),
        },
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "correctness_before_timing": True,
        },
        "implementations": {
            "public_api": {
                "available": True,
                "label": "autograd(torch_ggml_ops.grouped_mmq)",
            },
            "ggtensile": {"available": True, "artifact": str(args.code_object)},
            "hip": {
                "available": hip is not None,
                "artifact": str(hip.code_object) if hip is not None else None,
                "kernel_name": hip.spec.symbol if hip is not None else None,
                "reason": hip_reason,
            },
            "bf16_baseline": {
                "available": has_baseline,
                "label": "routed torch.mm on dequantized BF16 banks",
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
