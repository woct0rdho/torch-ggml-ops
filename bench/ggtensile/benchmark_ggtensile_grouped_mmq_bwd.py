#!/usr/bin/env python3
"""Benchmark one exact standalone grouped-backward deployment route."""

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
from benchmark_routes import fitted_prior_distribution_for_rows, make_route_tensors
from benchmark_support import public_grouped_backward, require_case
from workload_prior import EXPERT_PRIOR_NAMES, expert_prior_metadata

from tools.ggtensile.grouped_mmq_bwd_runtime import (
    InstalledGroupedBackwardControl,
)
from tools.ggtensile.model import SolutionKey
from tools.ggtensile.quant_formats import BACKWARD_QUANT_FORMATS
from tools.ggtensile.runtime import GroupedBackwardModule, HIPRuntimeError

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
    p.add_argument("--skip-mutations", action="store_true")
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
    selected = packed.index_select(0, expert_indices).contiguous()
    logical = dequantize_gguf_tensor(
        selected,
        gguf.GGMLQuantizationType(quant_type),
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(len(group_sizes), grad_output.shape[1], output_features)
    outputs = []
    begin = 0
    for group, rows in enumerate(group_sizes):
        end = begin + rows
        outputs.append(grad_output[begin:end] @ logical[group])
        begin = end
    return torch.cat(outputs)


def require(metrics: object, label: str, limit: float) -> None:
    typed = cast(ErrorMetrics, metrics)
    nrmse = typed["normalized_rmse"]
    if typed["finite"] is not True or not isinstance(nrmse, float) or nrmse > limit:
        raise RuntimeError(f"{label} correctness failed: {metrics}")


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
    expert_indices, expert_offsets, _ = make_route_tensors(distribution)
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
            try:
                hip = stack.enter_context(
                    InstalledGroupedBackwardControl(
                        key.problem_type.quant_data_type,
                        size.m,
                        expert_indices.numel(),
                    )
                )
                hip_reason = None
            except HIPRuntimeError as error:
                hip_reason = str(error)

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

        def run_mutation() -> torch.Tensor:
            mutated_public = public_api()
            ggtensile()
            if hip is not None:
                hip_launch()
            torch.cuda.synchronize()
            return mutated_public

        ggtensile()
        public_output = public_api()
        if hip is not None:
            hip_launch()
        torch.cuda.synchronize()
        baseline_candidate = candidate_output.clone()
        baseline_public = public_output.clone()
        baseline_hip = hip_output.clone() if hip is not None else None
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
            raise RuntimeError("grouped backward output is not deterministic")
        if not args.skip_reference:
            baseline = reference(
                grad_output,
                packed,
                int(tensor.tensor_type),
                expert_indices,
                distribution.group_sizes_cpu,
                size.n,
            )
            correctness["ggtensile_vs_bf16_baseline"] = error_metrics(
                candidate_output, baseline
            )
        else:
            baseline = None
        require(correctness["ggtensile_vs_public_api"], "GGTensile/public API", 5e-4)
        if hip is not None:
            require(correctness["ggtensile_vs_hip"], "GGTensile/HIP", 5e-4)
        if baseline is not None:
            require(correctness["ggtensile_vs_bf16_baseline"], "GGTensile/BF16", 0.04)
        mutations = None
        if not args.skip_mutations:
            mutations = {}

            def record_mutation(name: str, mutated_public: torch.Tensor) -> None:
                item: dict[str, object] = {
                    "public_changed": int(
                        torch.count_nonzero(mutated_public != baseline_public)
                    ),
                    "ggtensile_changed": int(
                        torch.count_nonzero(candidate_output != baseline_candidate)
                    ),
                    "ggtensile_vs_public": error_metrics(
                        candidate_output, mutated_public
                    ),
                }
                if hip is not None and baseline_hip is not None:
                    item["hip_changed"] = int(
                        torch.count_nonzero(hip_output != baseline_hip)
                    )
                    item["ggtensile_vs_hip"] = error_metrics(
                        candidate_output, hip_output
                    )
                    require(item["ggtensile_vs_hip"], "GGTensile/HIP mutation", 5e-4)
                require(
                    item["ggtensile_vs_public"],
                    "GGTensile/public mutation",
                    5e-4,
                )
                mutations[name] = item

            saved = grad_output[0].clone()
            grad_output[0].neg_()
            record_mutation("grad_output", run_mutation())
            grad_output[0].copy_(saved)
            if not mutations["grad_output"]["public_changed"]:
                raise RuntimeError("grouped backward gradient mutation was ineffective")

            saved_expert = expert_indices[0].clone()
            expert_indices[0] = (int(saved_expert) + 1) % 256
            record_mutation("expert_indices", run_mutation())
            expert_indices[0].copy_(saved_expert)
            if not mutations["expert_indices"]["public_changed"]:
                raise RuntimeError("grouped backward route mutation was ineffective")

            active_expert = int(expert_indices[0])
            replacement_expert = (active_expert + 1) % 256
            saved_weight = packed[active_expert].clone()
            packed[active_expert].copy_(packed[replacement_expert])
            record_mutation("active_weight", run_mutation())
            packed[active_expert].copy_(saved_weight)
            if not mutations["active_weight"]["public_changed"]:
                raise RuntimeError("grouped backward weight mutation was ineffective")

            active_ids = set(distribution.expert_indices_cpu)
            inactive_expert = next(
                expert for expert in range(256) if expert not in active_ids
            )
            saved_weight = packed[inactive_expert].clone()
            packed[inactive_expert].bitwise_xor_(0x55)
            record_mutation("inactive_weight", run_mutation())
            packed[inactive_expert].copy_(saved_weight)
            if mutations["inactive_weight"]["public_changed"]:
                raise RuntimeError("inactive grouped weight affected the output")

        timing = {}
        if not args.skip_timing:
            functions = {"public_api": public_api, "ggtensile": ggtensile}
            if baseline is not None:
                functions["bf16_baseline"] = lambda: reference(
                    grad_output,
                    packed,
                    int(tensor.tensor_type),
                    expert_indices,
                    distribution.group_sizes_cpu,
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
                "available": baseline is not None,
                "label": "routed torch.mm on dequantized BF16 banks",
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
