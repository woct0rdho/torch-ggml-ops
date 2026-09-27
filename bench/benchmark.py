"""Benchmark two prepared implementations of one MMQ operation."""

from contextlib import ExitStack
from typing import cast

import gguf
import torch

from bench.benchmark_baseline import prepare_baseline
from bench.benchmark_common import (
    OPERATIONS,
    device_info,
    logical_flops,
    measure,
    parse_args,
    release_cuda,
    select_cases,
    write_report,
)
from bench.benchmark_data import RouteSelection, input_mapping, prepare_input
from bench.benchmark_kernels import prepare_direct_implementations
from tools.mmq_deployment_cases import hip_control_root


def run() -> None:
    args = parse_args()
    spec = OPERATIONS[args.operation]
    cases = select_cases(args.operation, args.model_family, args.case)
    if not cases:
        raise ValueError(f"no {args.operation} cases exist for {args.model_family}")
    names = tuple(args.implementations)
    hip_root = args.hip_root
    if "hip" in names:
        hip_root = hip_root or hip_control_root()
        if hip_root is None or not hip_root.exists():
            raise FileNotFoundError("HIP controls are unavailable")
    report = {
        "comparison": {"first": names[0], "second": names[1]},
        "operation": args.operation,
        "device": device_info(),
        "model": str(args.model),
        "model_family": args.model_family,
        "implementations": list(names),
        "configuration": {
            "expert_prior": args.expert_prior,
            "case_selectors": args.case,
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "route_vectors": args.route_vectors,
            "launches_per_sample": args.launches_per_sample,
            "ggtensile_root": str(args.ggtensile_root) if args.ggtensile_root else None,
            "hip_root": str(hip_root) if hip_root else None,
        },
        "results": [],
    }
    reader = gguf.GGUFReader(args.model)
    write_report(args.output, report)

    with torch.inference_mode():
        for case in cases:
            benchmark_input = prepare_input(
                case,
                reader,
                expert_prior=args.expert_prior,
                seed=args.seed,
                route_vectors=args.route_vectors,
                materialize_baseline="baseline" in names,
            )
            prepared = benchmark_input.prepared
            selection = RouteSelection(benchmark_input.route_bank or ())
            with ExitStack() as stack:
                implementations = {}
                if "baseline" in names:
                    implementations["baseline"] = prepare_baseline(
                        prepared, selection, args.expert_prior, spec.baseline
                    )
                direct = tuple(name for name in names if name != "baseline")
                implementations.update(
                    prepare_direct_implementations(
                        stack,
                        case,
                        prepared,
                        selection,
                        direct,
                        ggtensile_root=args.ggtensile_root,
                        hip_root=hip_root,
                    )
                )
                torch.cuda.synchronize()
                timing, timing_protocol = measure(
                    {name: implementations[name].launch for name in names},
                    warmup=args.warmup,
                    repeats=args.repeats,
                    launches=args.launches_per_sample,
                    flops=logical_flops(case, spec),
                    select_sample=selection.select if spec.routed else None,
                )
            first, second = names
            ratio = cast(float, timing[second]["median_tflops"]) / cast(
                float, timing[first]["median_tflops"]
            )
            result = {
                **case.to_mapping(),
                "problem": {
                    "rows": case.rows,
                    "out_features": case.out_features,
                    "in_features": case.in_features,
                    "projections": spec.projections,
                    "logical_flops": logical_flops(case, spec),
                },
                "inputs": input_mapping(benchmark_input, args.model),
                "protocol": {
                    "surface": "prepared_kernel",
                    "weight_dequantization_in_timing": False,
                    "activation_quantization_in_timing": False,
                    "route_selection_in_timing": False,
                    "output_allocation_in_timing": False,
                    "correctness_in_benchmark": False,
                    **timing_protocol,
                },
                "implementations": {
                    name: implementations[name].metadata for name in names
                },
                "timing": timing,
                "throughput_ratio": {
                    "numerator": second,
                    "denominator": first,
                    "value": ratio,
                },
                "speedup": ratio,
            }
            report["results"].append(result)
            print(
                f"{case.quant_type:<8} M={case.rows:>7} "
                f"N={case.out_features:>6} K={case.in_features:>5} "
                f"{first}={timing[first]['median_ms']:.3f} ms "
                f"{second}={timing[second]['median_ms']:.3f} ms "
                f"{second}/{first}={ratio:.3f}x",
                flush=True,
            )
            write_report(args.output, report)
            del implementations, prepared, benchmark_input
            release_cuda()
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    run()
