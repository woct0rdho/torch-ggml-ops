"""Driver for preallocated GGTensile versus HIP-control benchmarks."""

from typing import Any

import gguf
import torch
from benchmark_common import (
    OPERATIONS,
    device_info,
    logical_flops,
    parse_args,
    print_result,
    release_cuda,
    rotating_timings,
    select_cases,
    write_report,
)
from benchmark_data import input_mapping, prepare_input
from benchmark_kernel_modules import prepare_kernel_comparison
from workload_prior import expert_prior_metadata

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
        )
        prepared = benchmark_input.prepared
        with prepare_kernel_comparison(
            case,
            prepared,
            ggtensile_root=args.ggtensile_root,
            hip_root=selected_hip_root,
        ) as comparison:
            comparison.launch_ggtensile()
            comparison.launch_hip()
            torch.cuda.synchronize()
            correctness = _correctness(
                comparison.ggtensile_output, comparison.hip_output
            )
            flops = logical_flops(case, spec)
            timing, order = rotating_timings(
                {
                    "ggtensile": comparison.launch_ggtensile,
                    "hip": comparison.launch_hip,
                },
                warmup=args.warmup,
                repeats=args.repeats,
                launches_per_sample=args.launches_per_sample,
                flops=flops,
            )
            speedup = float(timing["hip"]["median_ms"]) / float(
                timing["ggtensile"]["median_ms"]
            )
            case_mapping = case.to_mapping()
            case_mapping.pop("ggtensile", None)
            case_mapping.pop("hip", None)
            result = {
                **case_mapping,
                "exact_key": case.key.to_mapping(),
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
                    "repeats": args.repeats,
                    "launches_per_sample": args.launches_per_sample,
                    "sample_order": order,
                    "correctness_before_timing": True,
                    "activation_quantization_in_timing": False
                    if not backward
                    else None,
                    "row_task_setup_in_timing": False if not backward else None,
                    "output_allocation_in_timing": False,
                    "torch_autograd_in_timing": False,
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
