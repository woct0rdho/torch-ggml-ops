#!/usr/bin/env python3
"""Benchmark one exact paired grouped-backward deployment route."""

import argparse
import contextlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import gguf
import numpy as np
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "bench"
for path in (REPO_ROOT, BENCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_report import ErrorMetrics, error_metrics, rotating_timings
from benchmark_routes import fitted_prior_distribution_for_rows, make_route_tensors
from benchmark_support import public_grouped_pair_backward, require_case
from workload_prior import EXPERT_PRIOR_NAMES, expert_prior_metadata

from tools.ggtensile.grouped_mmq_bwd_pair_model import GroupedBackwardPairSolutionKey
from tools.ggtensile.grouped_mmq_bwd_pair_runtime import (
    GroupedBackwardPairModule,
    InstalledGroupedBackwardPairIQ2SControl,
    InstalledGroupedBackwardPairIQ2XXSControl,
    InstalledGroupedBackwardPairQ3KControl,
)
from tools.ggtensile.grouped_mmq_bwd_pair_spec import DerivedGroupedBackwardPairState
from tools.ggtensile.runtime import HIPRuntimeError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-key", type=Path, required=True)
    parser.add_argument("--code-object", type=Path, required=True)
    parser.add_argument("--hip-code-object", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--first-tensor", required=True)
    parser.add_argument("--second-tensor", required=True)
    parser.add_argument("--expert-prior", choices=EXPERT_PRIOR_NAMES, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--skip-timing", action="store_true")
    return parser


def _load_weight(
    reader: gguf.GGUFReader,
    name: str,
    key: GroupedBackwardPairSolutionKey,
    state: DerivedGroupedBackwardPairState,
) -> tuple[gguf.ReaderTensor, torch.Tensor]:
    tensor = next((item for item in reader.tensors if item.name == name), None)
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {name}")
    if tensor.tensor_type.name != key.problem.quant_data_type:
        raise ValueError(f"paired tensor {name} has the wrong quant type")
    host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    return tensor, torch.from_numpy(host).view(
        state.expected_packed_weight_shape
    ).cuda()


def _control(key: GroupedBackwardPairSolutionKey, code_object: Path):
    macro_tile = key.solution.compute.macro_tile0
    quant = key.problem.quant_data_type
    if quant == "Q3_K":
        return InstalledGroupedBackwardPairQ3KControl(macro_tile, code_object)
    if quant == "IQ2_S":
        return InstalledGroupedBackwardPairIQ2SControl(macro_tile, code_object)
    if quant == "IQ2_XXS":
        return InstalledGroupedBackwardPairIQ2XXSControl(macro_tile, code_object)
    raise HIPRuntimeError(f"no legacy HIP control for {quant}")


def _reference(
    first_grad: torch.Tensor,
    second_grad: torch.Tensor,
    first_weight: torch.Tensor,
    second_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    group_sizes: tuple[int, ...],
) -> torch.Tensor:
    first_selected = first_weight.index_select(0, expert_indices)
    second_selected = second_weight.index_select(0, expert_indices)
    outputs = []
    begin = 0
    for index, rows in enumerate(group_sizes):
        end = begin + rows
        outputs.append(
            first_grad[begin:end] @ first_selected[index]
            + second_grad[begin:end] @ second_selected[index]
        )
        begin = end
    return torch.cat(outputs)


def _require_correctness(correctness: Mapping[str, object], hip: bool) -> None:
    names = ["ggtensile_vs_public_api", "ggtensile_vs_bf16_baseline"]
    if hip:
        names.append("ggtensile_vs_hip")
    for name in names:
        metrics = cast(ErrorMetrics, correctness[name])
        if (
            not metrics["finite"]
            or metrics["normalized_rmse"] is None
            or metrics["normalized_rmse"] > 0.04
        ):
            raise RuntimeError(f"paired backward correctness failed: {name}: {metrics}")


def main() -> None:
    args = _parser().parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    key = GroupedBackwardPairSolutionKey.from_mapping(
        json.loads(args.solution_key.read_text(encoding="utf-8"))
    )
    state = DerivedGroupedBackwardPairState.from_solution_key(key)
    case = require_case(
        "GroupedBackwardPair",
        key,
        args.code_object,
        (args.first_tensor, args.second_tensor),
    )
    reader = gguf.GGUFReader(args.model)
    first_tensor, first_weight = _load_weight(reader, args.first_tensor, key, state)
    second_tensor, second_weight = _load_weight(reader, args.second_tensor, key, state)
    distribution = fitted_prior_distribution_for_rows(
        args.expert_prior, key.problem.aggregate_rows
    )
    expert_indices, expert_offsets, _ = make_route_tensors(distribution)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    first_grad = torch.randn(
        key.problem.aggregate_rows,
        key.problem.out_features,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    second_grad = torch.randn_like(first_grad)
    candidate_output = torch.empty(
        key.problem.aggregate_rows,
        key.problem.in_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    hip_output = torch.empty_like(candidate_output)
    stream = torch.cuda.current_stream().cuda_stream

    def public_api() -> torch.Tensor:
        return public_grouped_pair_backward(
            first_grad,
            second_grad,
            first_weight,
            second_weight,
            expert_indices,
            expert_offsets,
            int(first_tensor.tensor_type),
            key.problem.in_features,
        )

    hip = None
    hip_reason = "no legacy HIP control artifact was supplied"
    with contextlib.ExitStack() as stack:
        candidate = stack.enter_context(
            GroupedBackwardPairModule(key, args.code_object)
        )
        if args.hip_code_object is not None:
            try:
                hip = stack.enter_context(_control(key, args.hip_code_object))
                hip_reason = None
            except HIPRuntimeError as error:
                hip_reason = str(error)

        def ggtensile() -> None:
            candidate.launch(
                first_grad,
                second_grad,
                first_weight,
                second_weight,
                candidate_output,
                expert_indices,
                expert_offsets,
                stream=stream,
            )

        def hip_kernel() -> None:
            if hip is not None:
                hip.launch(
                    first_grad,
                    second_grad,
                    first_weight,
                    second_weight,
                    hip_output,
                    expert_indices,
                    expert_offsets,
                    stream=stream,
                )

        logical_first = (
            dequantize_gguf_tensor(
                first_weight,
                first_tensor.tensor_type,
                dtype=torch.bfloat16,
                device="cuda",
            )
            .reshape(256, key.problem.out_features, key.problem.in_features)
            .contiguous()
        )
        logical_second = (
            dequantize_gguf_tensor(
                second_weight,
                second_tensor.tensor_type,
                dtype=torch.bfloat16,
                device="cuda",
            )
            .reshape(256, key.problem.out_features, key.problem.in_features)
            .contiguous()
        )
        ggtensile()
        public_output = public_api()
        baseline_output = _reference(
            first_grad,
            second_grad,
            logical_first,
            logical_second,
            expert_indices,
            distribution.group_sizes_cpu,
        )
        if hip is not None:
            hip_kernel()
        torch.cuda.synchronize()
        correctness = {
            "ggtensile_vs_public_api": error_metrics(candidate_output, public_output),
            "public_api_vs_bf16_baseline": error_metrics(
                public_output, baseline_output
            ),
            "ggtensile_vs_bf16_baseline": error_metrics(
                candidate_output, baseline_output
            ),
        }
        if hip is not None:
            correctness["ggtensile_vs_hip"] = error_metrics(
                candidate_output, hip_output
            )
            correctness["public_api_vs_hip"] = error_metrics(public_output, hip_output)

        _require_correctness(correctness, hip is not None)
        timing = {}
        if not args.skip_timing:
            functions = {
                "public_api": public_api,
                "ggtensile": ggtensile,
                "bf16_baseline": lambda: _reference(
                    first_grad,
                    second_grad,
                    logical_first,
                    logical_second,
                    expert_indices,
                    distribution.group_sizes_cpu,
                ),
            }
            if hip is not None:
                functions["hip"] = hip_kernel
            timing = rotating_timings(
                functions,
                warmup=args.warmup,
                repeats=args.repeats,
                logical_flops=4
                * key.problem.aggregate_rows
                * key.problem.out_features
                * key.problem.in_features,
            )

    report = {
        **case.to_mapping(),
        "solution_key": key.to_mapping(),
        "solution_hash": key.hash,
        "kernel_name": key.kernel_name,
        "model": str(args.model),
        "tensors": [args.first_tensor, args.second_tensor],
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
                "label": "torch_ggml_ops.grouped_mmq_pair autograd",
            },
            "ggtensile": {"available": True, "artifact": str(args.code_object)},
            "hip": {
                "available": hip is not None,
                "reason": None if hip is not None else hip_reason,
            },
            "bf16_baseline": {
                "available": True,
                "label": "independently dequantized routed matmul pair",
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
