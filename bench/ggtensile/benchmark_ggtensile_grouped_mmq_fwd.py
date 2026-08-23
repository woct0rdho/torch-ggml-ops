#!/usr/bin/env python3
"""Benchmark one exact standalone grouped-forward deployment route."""

import argparse
import contextlib
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import torch

import torch_ggml_ops

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "bench"
for path in (REPO_ROOT, BENCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_inventory import validate_direct_artifact
from benchmark_report import error_metrics, rotating_timings
from benchmark_routes import fitted_prior_distribution_for_rows, make_route_tensors
from workload_prior import EXPERT_PRIOR_NAMES, expert_prior_metadata

from tools.ggtensile.grouped_mmq_fwd_model import GroupedForwardSolutionKey
from tools.ggtensile.grouped_mmq_fwd_spec import DerivedGroupedForwardState
from tools.ggtensile.runtime import (
    FixedQ81F32D4QuantizerModule,
    GroupedForwardModule,
    HIPRuntimeError,
    InstalledGroupedForwardIQ2SJ64J32Module,
    InstalledGroupedForwardIQ2SJ64Module,
    InstalledGroupedForwardModule,
    InstalledGroupedForwardQ2J32J16Module,
    InstalledGroupedForwardQ2J32Module,
    InstalledGroupedForwardQ5J32Module,
    InstalledGroupedForwardQ5Module,
)
from tools.mmq_correctness import (
    dequantize_active_routed_weight,
    routed_forward_reference,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-key", type=Path, required=True)
    parser.add_argument("--code-object", type=Path, required=True)
    parser.add_argument("--hip-code-object", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--expert-prior", choices=EXPERT_PRIOR_NAMES, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--skip-timing", action="store_true")
    return parser


def _load_weight(
    model: Path, tensor_name: str, state: DerivedGroupedForwardState
) -> tuple[gguf.ReaderTensor, torch.Tensor]:
    reader = gguf.GGUFReader(model)
    tensor = next((item for item in reader.tensors if item.name == tensor_name), None)
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {tensor_name}")
    problem = state.key.problem
    if tensor.tensor_type.name != problem.quant_data_type:
        raise ValueError("grouped tensor quant type does not match the exact key")
    host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    packed = torch.from_numpy(host).view(state.expected_packed_weight_shape).cuda()
    return tensor, packed


def _control(
    key: GroupedForwardSolutionKey,
    route_entries: int,
    code_object: Path | None,
):
    rows = key.problem.aggregate_rows
    quant = key.problem.quant_data_type
    if quant == "Q4_K":
        return InstalledGroupedForwardModule(key, code_object)
    if quant == "Q5_K":
        control = (
            InstalledGroupedForwardQ5J32Module
            if rows < 128 * route_entries
            else InstalledGroupedForwardQ5Module
        )
        return control(key, code_object)
    if quant == "Q2_K":
        control = (
            InstalledGroupedForwardQ2J32J16Module
            if rows == 49_152 or rows < 64 * route_entries
            else InstalledGroupedForwardQ2J32Module
        )
        return control(key, code_object)
    if quant == "IQ2_S":
        control = (
            InstalledGroupedForwardIQ2SJ64J32Module
            if rows == 65_536 or rows < 128 * route_entries
            else InstalledGroupedForwardIQ2SJ64Module
        )
        return control(key, code_object)
    raise HIPRuntimeError(f"no legacy HIP control for {quant}")


def main() -> None:
    args = _parser().parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    key = GroupedForwardSolutionKey.from_mapping(
        json.loads(args.solution_key.read_text(encoding="utf-8"))
    )
    case = validate_direct_artifact("GroupedForward", key, args.code_object)
    state = DerivedGroupedForwardState.from_solution_key(key)
    tensor, packed_weight = _load_weight(args.model, args.tensor, state)
    distribution = fitted_prior_distribution_for_rows(
        args.expert_prior, key.problem.aggregate_rows
    )
    expert_indices, expert_offsets, group_sizes = make_route_tensors(distribution)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    input_tensor = torch.randn(
        key.problem.aggregate_rows,
        key.problem.input_features,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    candidate_output = torch.empty(
        state.expected_output_shape, device="cuda", dtype=torch.bfloat16
    )
    hip_output = torch.empty_like(candidate_output)
    stream = torch.cuda.current_stream().cuda_stream

    hip_reason = "no legacy HIP control artifact was supplied"
    with contextlib.ExitStack() as stack:
        quantizer = stack.enter_context(FixedQ81F32D4QuantizerModule())
        workspace = quantizer.allocate(input_tensor)
        candidate = stack.enter_context(GroupedForwardModule(key, args.code_object))
        hip = stack.enter_context(
            _control(key, expert_indices.numel(), args.hip_code_object)
        )
        hip_reason = None

        def quantize() -> None:
            quantizer.launch(input_tensor, workspace, stream=stream)

        def ggtensile() -> None:
            quantize()
            candidate.launch(
                packed_weight,
                workspace,
                candidate_output,
                expert_indices,
                expert_offsets,
                stream=stream,
            )

        def hip_kernel() -> None:
            if hip is None:
                return
            quantize()
            hip.launch(
                packed_weight,
                workspace,
                hip_output,
                expert_indices,
                expert_offsets,
                stream=stream,
            )

        def public_api() -> torch.Tensor:
            return torch_ggml_ops.grouped_mmq(
                input_tensor,
                packed_weight,
                expert_indices,
                expert_offsets,
                int(tensor.tensor_type),
                key.problem.output_features,
            )

        logical_weight = dequantize_active_routed_weight(
            packed_weight,
            tensor.tensor_type,
            expert_indices,
            key.problem.output_features,
            key.problem.input_features,
        )
        quantize()
        candidate.launch(
            packed_weight,
            workspace,
            candidate_output,
            expert_indices,
            expert_offsets,
            stream=stream,
        )
        public_output = public_api()
        baseline_output = routed_forward_reference(
            input_tensor, logical_weight, expert_indices, group_sizes
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

        del public_output, baseline_output
        timing = {}
        if not args.skip_timing:
            functions = {
                "public_api": public_api,
                "ggtensile": ggtensile,
                "bf16_baseline": lambda: routed_forward_reference(
                    input_tensor, logical_weight, expert_indices, group_sizes
                ),
            }
            if hip is not None:
                functions["hip"] = hip_kernel
            timing = rotating_timings(
                functions,
                warmup=args.warmup,
                repeats=args.repeats,
                logical_flops=2
                * key.problem.aggregate_rows
                * key.problem.output_features
                * key.problem.input_features,
            )

    report = {
        **case.to_mapping(),
        "solution_key": key.to_mapping(),
        "solution_hash": key.hash,
        "kernel_name": key.kernel_name,
        "model": str(args.model),
        "tensor": args.tensor,
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
            "public_api": {"available": True, "label": "torch_ggml_ops.grouped_mmq"},
            "ggtensile": {"available": True, "artifact": str(args.code_object)},
            "hip": {
                "available": hip is not None,
                "reason": None if hip is not None else hip_reason,
            },
            "bf16_baseline": {
                "available": True,
                "label": "independently dequantized routed matmul",
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
