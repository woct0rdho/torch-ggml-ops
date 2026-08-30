"""Driver for preallocated GGTensile versus HIP-control benchmarks."""

import math
from typing import Any

import gguf
import torch
from benchmark_common import (
    OPERATIONS,
    AdaptiveTimingPolicy,
    adaptive_timings,
    device_info,
    logical_flops,
    parse_args,
    print_result,
    release_cuda,
    select_cases,
    write_report,
)
from benchmark_data import input_mapping, prepare_input
from benchmark_kernel_modules import prepare_kernel_comparison
from workload_prior import expert_prior_metadata

from tools.ggtensile.family_registry import mapping_for_instance
from tools.mmq_correctness import assert_implementation_equivalence
from tools.mmq_deployment_cases import hip_control_root


def _correctness(actual: Any, expected: Any) -> Any:
    if isinstance(actual, tuple):
        if not isinstance(expected, tuple) or len(actual) != len(expected):
            raise ValueError("direct outputs have different structures")
        return [
            assert_implementation_equivalence(a, e, f"direct output {index}")
            for index, (a, e) in enumerate(zip(actual, expected, strict=True))
        ]
    if isinstance(expected, tuple):
        raise TypeError("direct outputs have different structures")
    return assert_implementation_equivalence(actual, expected, "direct output")


def run_kernels(operation: str, *, routed: bool) -> None:
    spec = OPERATIONS[operation]
    args = parse_args(
        f"Benchmark direct GGTensile and HIP-control {spec.slug} kernels.",
        routed=routed,
        kernels=True,
    )
    cases = select_cases(operation, args.model_family, args.case)
    if not cases:
        raise ValueError(f"no {operation} cases exist for {args.model_family}")
    adaptive_policy = AdaptiveTimingPolicy(
        min_samples=args.repeats,
        max_samples=args.max_samples,
        sample_step=args.sample_step,
        confidence=args.adaptive_confidence,
        epsilon_pct=args.adaptive_epsilon_pct,
        stable_rounds=args.adaptive_stable_rounds,
        noise_floor_pct=args.adaptive_noise_floor_pct,
    )
    selected_hip_root = args.hip_root or hip_control_root()
    if selected_hip_root is None or not selected_hip_root.exists():
        raise FileNotFoundError(
            "HIP controls are unavailable; run python tools/build_mmq_hip_controls.py"
        )
    backward = "Backward" in operation
    report = {
        "comparison": "ggtensile_vs_hip",
        "operation": operation,
        "device": device_info(),
        "model": str(args.model),
        "model_family": args.model_family,
        "configuration": {
            "expert_prior": args.expert_prior,
            "expert_prior_fit": (
                expert_prior_metadata(args.expert_prior)
                if args.expert_prior is not None
                else None
            ),
            "case_selectors": args.case,
            "input_seed": args.seed,
            "adaptive_timing_policy": adaptive_policy.to_mapping(),
            "ggtensile_root": str(args.ggtensile_root) if args.ggtensile_root else None,
            "hip_root": str(selected_hip_root),
        },
        "results": [],
    }
    reader = gguf.GGUFReader(args.model)
    write_report(args.output, report)
    for case in cases:
        benchmark_input = prepare_input(
            case,
            reader,
            expert_prior=args.expert_prior,
            seed=args.seed,
            route_vectors=args.max_samples if routed else 1,
            route_seed=args.seed,
        )
        prepared = benchmark_input.prepared
        with prepare_kernel_comparison(
            case,
            prepared,
            ggtensile_root=args.ggtensile_root,
            hip_root=selected_hip_root,
            route_bank=benchmark_input.route_bank,
        ) as comparison:
            if comparison.select_sample is not None:
                if benchmark_input.route_bank is None:
                    raise RuntimeError("routed comparison is missing its route bank")
                correctness_samples = []
                for route_index in range(len(benchmark_input.route_bank)):
                    comparison.select_sample(route_index)
                    comparison.launch_ggtensile()
                    comparison.launch_hip()
                    torch.cuda.synchronize()
                    correctness_samples.append(
                        {
                            "sample_index": route_index,
                            "metrics": _correctness(
                                comparison.ggtensile_output,
                                comparison.hip_output,
                            ),
                        }
                    )
                correctness = {
                    "route_vectors_checked": len(correctness_samples),
                    "samples": correctness_samples,
                }
            else:
                comparison.launch_ggtensile()
                comparison.launch_hip()
                torch.cuda.synchronize()
                correctness = _correctness(
                    comparison.ggtensile_output, comparison.hip_output
                )
            flops = logical_flops(case, spec)
            timing, adaptive = adaptive_timings(
                {
                    "ggtensile": comparison.launch_ggtensile,
                    "hip": comparison.launch_hip,
                },
                policy=adaptive_policy,
                warmup=args.warmup,
                launches_per_sample=args.launches_per_sample,
                flops=flops,
                select_sample=comparison.select_sample if routed else None,
                sample_capacity=(
                    len(benchmark_input.route_bank)
                    if routed and benchmark_input.route_bank is not None
                    else None
                ),
            )
            speedup = math.exp(
                float(timing["hip"]["median_log_ms"])
                - float(timing["ggtensile"]["median_log_ms"])
            )
            case_mapping = case.to_mapping()
            case_mapping.pop("ggtensile", None)
            case_mapping.pop("hip", None)
            result = {
                **case_mapping,
                "exact_key": mapping_for_instance(case.instance),
                "problem": {
                    "rows": case.rows,
                    "out_features": case.out_features,
                    "in_features": case.in_features,
                    "projections": spec.projections,
                    "logical_flops": flops,
                },
                "inputs": input_mapping(benchmark_input, args.model),
                "protocol": {
                    "surface": (
                        "preallocated_direct_backward_kernel"
                        if backward
                        else "prequantized_direct_forward_kernel"
                    ),
                    "warmup": args.warmup,
                    "initial_samples": args.repeats,
                    "max_samples": args.max_samples,
                    "sample_step": args.sample_step,
                    "launches_per_sample": args.launches_per_sample,
                    "correctness_before_timing": True,
                    "correctness_route_vectors": (
                        len(benchmark_input.route_bank)
                        if routed and benchmark_input.route_bank is not None
                        else 1
                    ),
                    "activation_quantization_in_timing": False
                    if not backward
                    else None,
                    "row_task_setup_in_timing": False if not backward else None,
                    "output_allocation_in_timing": False,
                    "torch_autograd_in_timing": False,
                    "adaptive": adaptive,
                },
                "implementations": comparison.metadata,
                "correctness": correctness,
                "timing": timing,
                "speedup": speedup,
            }
            report["results"].append(result)
            print_result(case, timing, speedup)
        write_report(args.output, report)
        del comparison, prepared, benchmark_input
        release_cuda()
    print(f"report={args.output}", flush=True)
