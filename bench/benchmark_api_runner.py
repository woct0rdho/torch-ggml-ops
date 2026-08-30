"""Driver for complete public-API versus logical-baseline benchmarks."""

import math
from collections.abc import Callable
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
from workload_prior import expert_prior_metadata

import torch_ggml_ops
from tools.ggtensile.family_registry import mapping_for_instance
from tools.mmq_correctness import (
    PreparedCase,
    RouteData,
    assert_external_reference,
    external_reference,
)


def _correctness(actual: Any, expected: Any) -> Any:
    if isinstance(actual, tuple):
        if not isinstance(expected, tuple) or len(actual) != len(expected):
            raise ValueError("public outputs have different structures")
        return [
            assert_external_reference(a, e, f"public output {index}")
            for index, (a, e) in enumerate(zip(actual, expected, strict=True))
        ]
    if isinstance(expected, tuple):
        raise TypeError("public outputs have different structures")
    return assert_external_reference(actual, expected, "public output")


def _public_forward(prepared: PreparedCase, input_tensor: torch.Tensor) -> Any:
    case = prepared.case
    quant_type = int(prepared.tensor_types[0])
    if case.operation in {"OrdinaryForward", "OrdinaryBackward"}:
        return torch_ggml_ops.mmq(
            input_tensor,
            prepared.packed_weights[0],
            quant_type,
            case.out_features,
        )
    if case.operation in {"GroupedForward", "GroupedBackward"}:
        if prepared.route is None:
            raise ValueError("grouped case is missing route metadata")
        return torch_ggml_ops.grouped_mmq(
            input_tensor,
            prepared.packed_weights[0],
            prepared.route.expert_indices,
            prepared.route.expert_offsets,
            quant_type,
            case.out_features,
        )
    if case.operation in {"GroupedForwardPair", "GroupedBackwardPair"}:
        if prepared.route is None:
            raise ValueError("grouped pair is missing route metadata")
        return torch_ggml_ops.grouped_mmq_pair(
            input_tensor,
            prepared.packed_weights[0],
            prepared.packed_weights[1],
            prepared.route.expert_indices,
            prepared.route.expert_offsets,
            quant_type,
            case.out_features,
        )
    return torch_ggml_ops.fixed_grouped_mmq(input_tensor, prepared.packed_weights[0])


def _public_callable(
    prepared: PreparedCase,
    route_bank: tuple[RouteData, ...] | None = None,
) -> tuple[Callable[[], Any], str, Callable[[int], None] | None]:
    case = prepared.case
    input_value = prepared.input
    if input_value is None:
        raise ValueError("public benchmark input is missing")
    if route_bank is not None and not route_bank:
        raise ValueError("public routed benchmark has an empty route bank")

    if case.operation.endswith("Forward") or case.operation == "GroupedForwardPair":

        def select_sample(index: int) -> None:
            if route_bank is None or not 0 <= index < len(route_bank):
                raise IndexError(
                    f"route vector index {index} is outside the route bank"
                )
            prepared.route = route_bank[index]

        return (
            lambda: _public_forward(prepared, input_value),
            "complete_public_call",
            select_sample if route_bank is not None else None,
        )

    input_tensor = input_value.detach().requires_grad_(True)
    graph_state: dict[str, Any] = {}

    def rebuild_graph() -> None:
        outputs = _public_forward(prepared, input_tensor)
        graph_state["outputs"] = outputs
        graph_state["grad_outputs"] = (
            prepared.grad_outputs
            if isinstance(outputs, tuple)
            else prepared.grad_outputs[0]
        )

    rebuild_graph()

    def backward() -> torch.Tensor:
        return torch.autograd.grad(
            graph_state["outputs"],
            input_tensor,
            graph_state["grad_outputs"],
            retain_graph=True,
        )[0]

    def select_sample(index: int) -> None:
        if route_bank is None or not 0 <= index < len(route_bank):
            raise IndexError(f"route vector index {index} is outside the route bank")
        prepared.route = route_bank[index]
        rebuild_graph()

    return (
        backward,
        "autograd_backward_only",
        select_sample if route_bank is not None else None,
    )


def _result(
    prepared: PreparedCase,
    benchmark_input,
    args,
    adaptive_policy: AdaptiveTimingPolicy,
    public_call: Callable[[], Any],
    public_surface: str,
    select_sample: Callable[[int], None] | None,
) -> dict[str, object]:
    case = prepared.case
    spec = OPERATIONS[case.operation]
    baseline_call = lambda: external_reference(prepared)
    if select_sample is not None:
        if benchmark_input.route_bank is None:
            raise RuntimeError("public routed comparison is missing its route bank")
        correctness_samples = []
        for route_index in range(len(benchmark_input.route_bank)):
            select_sample(route_index)
            public_output = public_call()
            baseline_output = baseline_call()
            torch.cuda.synchronize()
            correctness_samples.append(
                {
                    "sample_index": route_index,
                    "metrics": _correctness(public_output, baseline_output),
                }
            )
            del public_output, baseline_output
        correctness = {
            "route_vectors_checked": len(correctness_samples),
            "samples": correctness_samples,
        }
    else:
        public_output = public_call()
        baseline_output = baseline_call()
        torch.cuda.synchronize()
        correctness = _correctness(public_output, baseline_output)
        del public_output, baseline_output
    flops = logical_flops(case, spec)
    timing, adaptive = adaptive_timings(
        {"public_api": public_call, "baseline": baseline_call},
        policy=adaptive_policy,
        warmup=args.warmup,
        launches_per_sample=args.launches_per_sample,
        flops=flops,
        select_sample=select_sample,
        sample_capacity=(
            len(benchmark_input.route_bank)
            if select_sample is not None and benchmark_input.route_bank is not None
            else None
        ),
    )
    speedup = math.exp(
        float(timing["baseline"]["median_log_ms"])
        - float(timing["public_api"]["median_log_ms"])
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
            "surface": public_surface,
            "warmup": args.warmup,
            "initial_samples": args.repeats,
            "max_samples": args.max_samples,
            "sample_step": args.sample_step,
            "launches_per_sample": args.launches_per_sample,
            "correctness_before_timing": True,
            "correctness_route_vectors": (
                len(benchmark_input.route_bank)
                if select_sample is not None and benchmark_input.route_bank is not None
                else 1
            ),
            "weight_dequantization_in_timing": False,
            "public_forward_graph_construction_in_backward_timing": False,
            "public_autograd_in_backward_timing": case.operation.endswith("Backward")
            or case.operation == "GroupedBackwardPair",
            "adaptive": adaptive,
        },
        "implementations": {
            "public_api": {"label": spec.public_label},
            "baseline": {"label": spec.baseline_label},
        },
        "correctness": correctness,
        "timing": timing,
        "speedup": speedup,
    }
    print_result(case, timing, speedup)
    return result


def run_api(operation: str, *, routed: bool) -> None:
    spec = OPERATIONS[operation]
    args = parse_args(
        f"Benchmark {spec.public_label} against {spec.baseline_label}.",
        routed=routed,
        kernels=False,
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
    report = {
        "comparison": "public_api_vs_baseline",
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
            full_logical_weights=routed,
        )
        prepared = benchmark_input.prepared
        public_call, public_surface, select_sample = _public_callable(
            prepared,
            benchmark_input.route_bank if routed else None,
        )
        report["results"].append(
            _result(
                prepared,
                benchmark_input,
                args,
                adaptive_policy,
                public_call,
                public_surface,
                select_sample,
            )
        )
        write_report(args.output, report)
        del public_call, prepared, benchmark_input
        release_cuda()
    print(f"report={args.output}", flush=True)
