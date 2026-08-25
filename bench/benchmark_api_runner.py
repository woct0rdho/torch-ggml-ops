"""Driver for complete public-API versus logical-baseline benchmarks."""

from collections.abc import Callable
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
from workload_prior import expert_prior_metadata

import torch_ggml_ops
from tools.ggtensile.family_registry import mapping_for_instance
from tools.mmq_correctness import (
    PreparedCase,
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


def _public_callable(prepared: PreparedCase) -> tuple[Callable[[], Any], str]:
    case = prepared.case
    input_value = prepared.input
    if input_value is None:
        raise ValueError("public benchmark input is missing")
    if case.operation.endswith("Forward") or case.operation == "GroupedForwardPair":
        return lambda: _public_forward(prepared, input_value), "complete_public_call"

    input_tensor = input_value.detach().requires_grad_(True)
    outputs = _public_forward(prepared, input_tensor)
    grad_outputs: Any = (
        prepared.grad_outputs
        if isinstance(outputs, tuple)
        else prepared.grad_outputs[0]
    )

    def backward() -> torch.Tensor:
        return torch.autograd.grad(
            outputs,
            input_tensor,
            grad_outputs,
            retain_graph=True,
        )[0]

    return backward, "autograd_backward_only"


def _result(
    prepared: PreparedCase,
    benchmark_input,
    args,
    public_call: Callable[[], Any],
    public_surface: str,
) -> dict[str, object]:
    case = prepared.case
    spec = OPERATIONS[case.operation]
    baseline_call = lambda: external_reference(prepared)
    public_output = public_call()
    baseline_output = baseline_call()
    torch.cuda.synchronize()
    correctness = _correctness(public_output, baseline_output)
    del public_output, baseline_output
    flops = logical_flops(case, spec)
    timing, order = rotating_timings(
        {"public_api": public_call, "baseline": baseline_call},
        warmup=args.warmup,
        repeats=args.repeats,
        launches_per_sample=args.launches_per_sample,
        flops=flops,
    )
    speedup = float(timing["baseline"]["median_ms"]) / float(
        timing["public_api"]["median_ms"]
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
            "repeats": args.repeats,
            "launches_per_sample": args.launches_per_sample,
            "sample_order": order,
            "correctness_before_timing": True,
            "weight_dequantization_in_timing": False,
            "public_forward_graph_construction_in_backward_timing": False,
            "public_autograd_in_backward_timing": case.operation.endswith("Backward")
            or case.operation == "GroupedBackwardPair",
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
        public_call, public_surface = _public_callable(prepared)
        report["results"].append(
            _result(
                prepared,
                benchmark_input,
                args,
                public_call,
                public_surface,
            )
        )
        write_report(args.output, report)
        del public_call, prepared, benchmark_input
        release_cuda()
    print(f"report={args.output}", flush=True)
