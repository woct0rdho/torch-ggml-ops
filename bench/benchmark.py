"""Benchmark two prepared implementations of one MMQ operation."""

import statistics
import sys
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
    stability_warnings,
    write_report,
)
from bench.benchmark_data import RouteSelection, input_mapping, prepare_input
from bench.benchmark_kernels import prepare_direct_implementations
from tools.aiter_gmm_compat import infer_expert_prior
from tools.mmq_deployment_cases import hip_control_root


def _inferred_expert_prior(case) -> str:
    """Infer the default routed law from the case's matrix key and quant type."""

    forward = "Forward" in case.operation
    return infer_expert_prior(
        case.rows,
        case.in_features if forward else case.out_features,
        case.out_features if forward else case.in_features,
        quant_type=case.quant_type,
        transposed_rhs=forward,
    )


def _resolve_expert_prior(args, spec, cases) -> None:
    """Fill in the benchmark's default routed prior so no caller has to."""

    if not spec.routed:
        args.prior_source = "explicit" if args.expert_prior else None
        return
    if args.expert_prior is not None:
        args.prior_source = "explicit"
        return
    inferred = {_inferred_expert_prior(case) for case in cases}
    if len(inferred) != 1:
        raise ValueError(
            f"cannot infer one expert prior for this run: {sorted(inferred)}"
        )
    args.expert_prior = inferred.pop()
    args.prior_source = "inferred"


def run() -> None:
    args = parse_args()
    spec = OPERATIONS[args.operation]
    cases = select_cases(args.operation, args.model_family, args.case)
    if not cases:
        raise ValueError(f"no {args.operation} cases exist for {args.model_family}")
    _resolve_expert_prior(args, spec, cases)
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
            "prior_source": args.prior_source,
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
                    warmup_seconds=args.warmup_seconds,
                    repeats=args.repeats,
                    launches=args.launches_per_sample,
                    flops=logical_flops(case, spec),
                    blocks=args.blocks,
                    select_sample=selection.select if spec.routed else None,
                )
            first, second = names
            ratio = cast(float, timing[second]["median_tflops"]) / cast(
                float, timing[first]["median_tflops"]
            )
            first_samples = cast(list[float], timing[first]["samples_ms"])
            second_samples = cast(list[float], timing[second]["samples_ms"])
            paired_ratio = statistics.median(
                [
                    left / right
                    for left, right in zip(first_samples, second_samples, strict=True)
                ]
            )
            block_medians = cast(
                list[dict[str, object]], timing_protocol["block_medians"]
            )
            block_ratios = [
                cast(dict[str, float], block["median_ms"])[first]
                / cast(dict[str, float], block["median_ms"])[second]
                for block in block_medians
            ]
            block_spread_pct = (
                (max(block_ratios) - min(block_ratios))
                / statistics.median(block_ratios)
                * 100.0
                if len(block_ratios) > 1
                else None
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
                "stability": {
                    "paired_ratio": paired_ratio,
                    "block_ratios": block_ratios,
                    "block_spread_pct": block_spread_pct,
                    "position_gap_pct": timing_protocol["position_gap_pct"],
                    "position_counts": timing_protocol["position_counts"],
                    "host_launch_us": timing_protocol["host_launch_us"],
                    "device_over_host_ratio": timing_protocol["device_over_host_ratio"],
                    "host_bound": timing_protocol["host_bound"],
                },
                "implementations": {
                    name: implementations[name].metadata for name in names
                },
                "timing": timing,
                "throughput_ratio": {
                    "numerator": second,
                    "denominator": first,
                    "value": ratio,
                    "paired": paired_ratio,
                },
                "speedup": ratio,
            }
            for warning in stability_warnings(names, timing_protocol, block_spread_pct):
                print(
                    f"WARNING {case.identity}: {warning}", file=sys.stderr, flush=True
                )
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
