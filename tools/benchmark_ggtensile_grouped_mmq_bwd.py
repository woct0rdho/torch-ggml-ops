#!/usr/bin/env python3

import argparse
import contextlib
import json
import statistics
import sys
from pathlib import Path
from typing import TypedDict, cast

import gguf
import numpy as np
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops  # noqa: F401 Register the installed packed control.

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ggtensile.benchmark_routes import (
    RouteDistribution,
    distribution_summary,
    make_route_tensors,
    route_distributions,
    truncate_distribution,
)
from tools.ggtensile.model import SolutionKey
from tools.ggtensile.quant_formats import BACKWARD_QUANT_FORMATS
from tools.ggtensile.runtime import GroupedBackwardModule

DEFAULT_MODEL = Path.home() / "models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"
DEFAULT_TENSORS = {
    "Q4_K": "blk.2.ffn_down_exps.weight",
    "Q5_K": "blk.0.ffn_down_exps.weight",
    "IQ2_S": "blk.10.ffn_down_exps.weight",
}
BF16_WMMA_ROOFLINE_TFLOPS = 59.4


class Metrics(TypedDict):
    different_bf16_elements: int
    elements: int
    finite: bool
    max_absolute_error: float | None
    error_rms: float | None
    reference_rms: float
    normalized_rmse: float | None


class TimingSummary(TypedDict):
    samples_ms: list[float]
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    median_tflops: float
    wmma_roofline_fraction: float


class TimingReport(TypedDict, total=False):
    distribution: str
    group_summary: dict[str, object]
    logical_flops: int
    hip_complete: TimingSummary
    candidate_complete: TimingSummary
    candidate_kernel: TimingSummary
    candidate_to_hip_latency: float
    candidate_to_hip_throughput: float
    medoid_weight: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark exact grouped GGTensile packed backward against HIP"
    )
    parser.add_argument("--solution-key", type=Path, required=True)
    parser.add_argument("--code-object", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--tensor",
        help="GGUF tensor override; defaults to the quant-specific Qwen down tensor",
    )
    parser.add_argument(
        "--distributions",
        default="uniform,skewed,sparse,boundary",
        help="comma-separated production distribution controls",
    )
    parser.add_argument("--routing-prior", type=Path)
    parser.add_argument(
        "--prior-family", choices=("auto", "qwen", "deepseek"), default="auto"
    )
    parser.add_argument(
        "--prior-bank", choices=("search", "confirmation"), default="search"
    )
    parser.add_argument("--prior-only", action="store_true")
    parser.add_argument("--correctness-rows", type=int, default=625)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    return parser


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> Metrics:
    difference = actual.float() - expected.float()
    finite = bool(torch.isfinite(difference).all())
    reference_rms = float(expected.float().square().mean().sqrt())
    error_rms = float(difference.square().mean().sqrt()) if finite else None
    return {
        "different_bf16_elements": int(torch.count_nonzero(actual != expected)),
        "elements": actual.numel(),
        "finite": finite,
        "max_absolute_error": float(difference.abs().max()) if finite else None,
        "error_rms": error_rms,
        "reference_rms": reference_rms,
        "normalized_rmse": (
            error_rms / reference_rms
            if error_rms is not None and reference_rms
            else None
        ),
    }


def _event_time(function) -> tuple[float, torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), output


def _timing_summary(samples_ms: list[float], logical_flops: int) -> TimingSummary:
    median_ms = statistics.median(samples_ms)
    median_tflops = logical_flops / (median_ms * 1.0e9)
    return {
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "median_tflops": median_tflops,
        "wmma_roofline_fraction": median_tflops / BF16_WMMA_ROOFLINE_TFLOPS,
    }


def _distribution_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(names) - {"uniform", "skewed", "sparse", "boundary"}
    if not names or unknown:
        raise ValueError(f"invalid distributions: {sorted(unknown)}")
    return names


def _prior_distributions(
    path: Path,
    bank: str,
    batch: int,
    family: str,
) -> tuple[tuple[RouteDistribution, ...], dict[str, float]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if family == "qwen":
        components = (("qwen_learned", 1.0),)
        expected_rows = batch * 16_384
    else:
        reporting_weights = document["deepseek_reporting_weights"]
        components = (
            ("deepseek_learned", float(reporting_weights["learned"])),
            ("deepseek_hash", float(reporting_weights["hash"])),
        )
        expected_rows = batch * 12_288

    distributions = []
    weights = {}
    for component, component_weight in components:
        profiles = document["banks"][bank][f"{component}_b{batch}"]["profiles"]
        for profile in profiles:
            distribution = RouteDistribution(
                profile["profile_id"],
                tuple(profile["expert_ids"]),
                tuple(profile["group_sizes"]),
            )
            if distribution.rows != expected_rows:
                raise ValueError(
                    f"prior profile {distribution.name} has {distribution.rows} rows"
                )
            distributions.append(distribution)
            weights[distribution.name] = component_weight * float(
                profile["medoid_weight"]
            )
    if abs(sum(weights.values()) - 1.0) > 1.0e-12:
        raise ValueError("prior medoid weights do not sum to one")
    return tuple(distributions), weights


def _load_packed(
    model: Path,
    tensor_name: str,
    key: SolutionKey,
) -> tuple[torch.Tensor, int, list[int]]:
    reader = gguf.GGUFReader(model)
    tensor = next((item for item in reader.tensors if item.name == tensor_name), None)
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {tensor_name}")
    quant_type = key.problem_type.quant_data_type
    if tensor.tensor_type.name != quant_type:
        raise ValueError(
            f"expected {quant_type} tensor, found {tensor.tensor_type.name}"
        )
    quant_format = BACKWARD_QUANT_FORMATS[quant_type]
    packed_row_bytes = (
        key.problem_size.n // quant_format.block_values * quant_format.block_bytes
    )
    physical_shape = [int(value) for value in tensor.data.shape]
    expected_shape = [256, key.problem_size.k, packed_row_bytes]
    if physical_shape != expected_shape:
        raise ValueError(
            f"unexpected packed expert shape {physical_shape}, expected {expected_shape}"
        )
    host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    packed = torch.from_numpy(host).cuda()
    del host
    return packed, int(tensor.tensor_type), physical_shape


def _launch_candidate(
    module: GroupedBackwardModule,
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    output: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    module.launch(
        grad_output,
        packed_weight,
        output,
        expert_indices,
        expert_offsets,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    return output


def _bf16_route_reference(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_type: int,
    distribution: RouteDistribution,
    expert_indices: torch.Tensor,
    output_features: int,
) -> torch.Tensor:
    selected_packed = packed_weight.index_select(0, expert_indices).contiguous()
    logical = dequantize_gguf_tensor(
        selected_packed,
        gguf.GGMLQuantizationType(quant_type),
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(len(distribution.group_sizes_cpu), grad_output.shape[1], output_features)
    outputs = []
    row_begin = 0
    for group, size in enumerate(distribution.group_sizes_cpu):
        row_end = row_begin + size
        outputs.append(grad_output[row_begin:row_end] @ logical[group])
        row_begin = row_end
    result = torch.cat(outputs)
    del selected_packed, logical, outputs
    return result


def _correctness(
    module: GroupedBackwardModule,
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_type: int,
    distribution: RouteDistribution,
    max_rows: int,
    output_features: int,
) -> dict[str, object]:
    checked = truncate_distribution(distribution, max_rows)
    expert_indices, expert_offsets, _ = make_route_tensors(checked)
    active_rows = checked.rows
    sentinel = 19.0
    actual = torch.full(
        (grad_output.shape[0], output_features),
        sentinel,
        device="cuda",
        dtype=torch.bfloat16,
    )

    def control() -> torch.Tensor:
        return torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
            grad_output,
            packed_weight,
            expert_indices,
            expert_offsets,
            quant_type,
            output_features,
        )

    def candidate() -> torch.Tensor:
        actual.fill_(sentinel)
        return _launch_candidate(
            module,
            grad_output,
            packed_weight,
            actual,
            expert_indices,
            expert_offsets,
        )

    reference = control()
    candidate()
    torch.cuda.synchronize()
    baseline = actual[:active_rows].clone()
    report: dict[str, object] = {
        "distribution": checked.name,
        "rows": active_rows,
        "group_summary": distribution_summary(checked),
        "candidate_vs_hip": _metrics(baseline, reference[:active_rows]),
        "tail_changed_elements": int(
            torch.count_nonzero(actual[active_rows:] != sentinel)
        ),
    }
    bf16_reference = _bf16_route_reference(
        grad_output,
        packed_weight,
        quant_type,
        checked,
        expert_indices,
        output_features,
    )
    report["candidate_vs_independent_bf16"] = _metrics(baseline, bf16_reference)
    del bf16_reference

    candidate()
    torch.cuda.synchronize()
    report["deterministic_rerun"] = _metrics(actual[:active_rows], baseline)

    grad_output[:active_rows].neg_()
    updated_reference = control()
    candidate()
    torch.cuda.synchronize()
    report["grad_output_mutation_vs_hip"] = _metrics(
        actual[:active_rows], updated_reference[:active_rows]
    )
    report["grad_output_mutation_changed_elements"] = int(
        torch.count_nonzero(updated_reference[:active_rows] != reference[:active_rows])
    )
    grad_output[:active_rows].neg_()

    original_expert = expert_indices[0].clone()
    expert_indices[0] = (int(original_expert) + 1) % 256
    updated_reference = control()
    candidate()
    torch.cuda.synchronize()
    report["route_mutation_vs_hip"] = _metrics(
        actual[:active_rows], updated_reference[:active_rows]
    )
    report["route_mutation_changed_elements"] = int(
        torch.count_nonzero(updated_reference[:active_rows] != reference[:active_rows])
    )
    expert_indices[0].copy_(original_expert)

    active_expert = int(expert_indices[0])
    replacement_expert = (active_expert + 1) % 256
    original_row = packed_weight[active_expert, 0].clone()
    packed_weight[active_expert, 0].copy_(packed_weight[replacement_expert, 0])
    updated_reference = control()
    candidate()
    torch.cuda.synchronize()
    report["active_weight_mutation_vs_hip"] = _metrics(
        actual[:active_rows], updated_reference[:active_rows]
    )
    report["active_weight_mutation_changed_elements"] = int(
        torch.count_nonzero(updated_reference[:active_rows] != reference[:active_rows])
    )
    packed_weight[active_expert, 0].copy_(original_row)

    active_ids = set(checked.expert_indices_cpu)
    inactive_expert = next(
        (expert for expert in range(256) if expert not in active_ids), None
    )
    if inactive_expert is not None:
        inactive_row = packed_weight[inactive_expert, 0].clone()
        packed_weight[inactive_expert, 0].bitwise_xor_(0x55)
        candidate()
        torch.cuda.synchronize()
        report["inactive_weight_mutation_vs_baseline"] = _metrics(
            actual[:active_rows], baseline
        )
        packed_weight[inactive_expert, 0].copy_(inactive_row)
    else:
        report["inactive_weight_mutation_vs_baseline"] = {"applicable": False}

    invalid_indices = expert_indices.clone()
    invalid_indices[0] = -1
    actual.fill_(sentinel)
    _launch_candidate(
        module,
        grad_output,
        packed_weight,
        actual,
        invalid_indices,
        expert_offsets,
    )
    torch.cuda.synchronize()
    first_end = checked.group_sizes_cpu[0]
    report["invalid_expert_first_route_changed_elements"] = int(
        torch.count_nonzero(actual[:first_end] != sentinel)
    )
    report["invalid_expert_later_route_written_elements"] = int(
        torch.count_nonzero(actual[first_end:active_rows] != sentinel)
    )

    invalid_offsets = expert_offsets.clone()
    invalid_offsets[0] = -1
    actual.fill_(sentinel)
    _launch_candidate(
        module,
        grad_output,
        packed_weight,
        actual,
        expert_indices,
        invalid_offsets,
    )
    torch.cuda.synchronize()
    report["invalid_offset_first_route_changed_elements"] = int(
        torch.count_nonzero(actual[:first_end] != sentinel)
    )
    report["invalid_offset_later_route_written_elements"] = int(
        torch.count_nonzero(actual[first_end:active_rows] != sentinel)
    )

    invalid_final_offsets = expert_offsets.clone()
    invalid_final_offsets[-1] = grad_output.shape[0] + 1
    actual.fill_(sentinel)
    _launch_candidate(
        module,
        grad_output,
        packed_weight,
        actual,
        expert_indices,
        invalid_final_offsets,
    )
    torch.cuda.synchronize()
    last_begin = active_rows - checked.group_sizes_cpu[-1]
    report["invalid_final_offset_last_route_changed_elements"] = int(
        torch.count_nonzero(actual[last_begin:active_rows] != sentinel)
    )
    report["invalid_final_offset_prior_routes_written_elements"] = int(
        torch.count_nonzero(actual[:last_begin] != sentinel)
    )
    return report


def _timings(
    module: GroupedBackwardModule,
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_type: int,
    distribution: RouteDistribution,
    output_features: int,
    warmup: int,
    repeats: int,
    weight: float | None = None,
    reverse_order: bool = False,
) -> TimingReport:
    expert_indices, expert_offsets, _ = make_route_tensors(distribution)
    kernel_output = torch.empty(
        (grad_output.shape[0], output_features), device="cuda", dtype=torch.bfloat16
    )

    def hip() -> torch.Tensor:
        return torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
            grad_output,
            packed_weight,
            expert_indices,
            expert_offsets,
            quant_type,
            output_features,
        )

    def candidate_kernel() -> torch.Tensor:
        return _launch_candidate(
            module,
            grad_output,
            packed_weight,
            kernel_output,
            expert_indices,
            expert_offsets,
        )

    def candidate_complete() -> torch.Tensor:
        output = torch.empty_like(kernel_output)
        return _launch_candidate(
            module,
            grad_output,
            packed_weight,
            output,
            expert_indices,
            expert_offsets,
        )

    functions = {
        "hip_complete": hip,
        "candidate_complete": candidate_complete,
        "candidate_kernel": candidate_kernel,
    }
    names = list(functions)
    if reverse_order:
        names.reverse()
    for _ in range(warmup):
        for name in names:
            functions[name]()
    torch.cuda.synchronize()

    samples = {name: [] for name in functions}
    for repeat in range(repeats):
        offset = repeat % len(names)
        for name in names[offset:] + names[:offset]:
            elapsed, output = _event_time(functions[name])
            samples[name].append(elapsed)
            del output

    logical_flops = 2 * distribution.rows * output_features * grad_output.shape[1]
    summaries = {
        name: _timing_summary(values, logical_flops) for name, values in samples.items()
    }
    hip_ms = summaries["hip_complete"]["median_ms"]
    complete_ms = summaries["candidate_complete"]["median_ms"]
    report = cast(
        TimingReport,
        {
            "distribution": distribution.name,
            "group_summary": distribution_summary(distribution),
            "logical_flops": logical_flops,
            **summaries,
            "candidate_to_hip_latency": complete_ms / hip_ms,
            "candidate_to_hip_throughput": hip_ms / complete_ms,
        },
    )
    if weight is not None:
        report["medoid_weight"] = weight
    return report


def _weighted_prior_summary(
    reports: list[TimingReport],
) -> dict[str, float] | None:
    prior_reports = [report for report in reports if "medoid_weight" in report]
    if not prior_reports:
        return None
    weight_sum = sum(float(report["medoid_weight"]) for report in prior_reports)
    if abs(weight_sum - 1.0) > 1.0e-12:
        raise ValueError("timed prior medoid weights do not sum to one")

    def weighted_median_ms(name: str) -> float:
        return sum(
            float(report["medoid_weight"])
            * cast(TimingSummary, report.get(name))["median_ms"]
            for report in prior_reports
        )

    hip_ms = weighted_median_ms("hip_complete")
    candidate_ms = weighted_median_ms("candidate_complete")
    return {
        "medoid_weight_sum": weight_sum,
        "hip_weighted_median_ms": hip_ms,
        "candidate_weighted_median_ms": candidate_ms,
        "candidate_to_hip_weighted_throughput": hip_ms / candidate_ms,
        "minimum_medoid_throughput": min(
            float(report["candidate_to_hip_throughput"]) for report in prior_reports
        ),
    }


def _require_correctness(report: dict[str, object]) -> None:
    metric_names = (
        "candidate_vs_hip",
        "deterministic_rerun",
        "grad_output_mutation_vs_hip",
        "route_mutation_vs_hip",
        "active_weight_mutation_vs_hip",
    )
    inactive_metrics = report.get("inactive_weight_mutation_vs_baseline")
    if (
        isinstance(inactive_metrics, dict)
        and inactive_metrics.get("applicable") is not False
    ):
        metric_names += ("inactive_weight_mutation_vs_baseline",)
    failures = []
    for name in metric_names:
        metrics = report[name]
        if not isinstance(metrics, dict):
            failures.append(name)
            continue
        if metrics.get("different_bf16_elements") != 0 or not metrics.get("finite"):
            failures.append(name)
    zero_controls = (
        "tail_changed_elements",
        "invalid_expert_first_route_changed_elements",
        "invalid_offset_first_route_changed_elements",
        "invalid_final_offset_last_route_changed_elements",
    )
    failures.extend(name for name in zero_controls if report.get(name) != 0)
    changed_controls = (
        "grad_output_mutation_changed_elements",
        "route_mutation_changed_elements",
        "active_weight_mutation_changed_elements",
        "invalid_expert_later_route_written_elements",
        "invalid_offset_later_route_written_elements",
        "invalid_final_offset_prior_routes_written_elements",
    )
    for name in changed_controls:
        value = report.get(name, 0)
        if not isinstance(value, int | float) or value <= 0:
            failures.append(name)
    independent = report.get("candidate_vs_independent_bf16")
    if not isinstance(independent, dict):
        failures.append("candidate_vs_independent_bf16")
    else:
        normalized_rmse = independent.get("normalized_rmse")
        if (
            independent.get("finite") is not True
            or not isinstance(normalized_rmse, int | float)
            or normalized_rmse > 0.01
        ):
            failures.append("candidate_vs_independent_bf16")
    if failures:
        raise RuntimeError(f"grouped backward correctness controls failed: {failures}")


def main() -> None:
    args = _parser().parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    key = SolutionKey.from_json_file(args.solution_key)
    if key.problem_type.operation_type != "GroupedMMQBackward":
        raise ValueError("solution key is not grouped MMQ backward")
    tensor_name = args.tensor or DEFAULT_TENSORS.get(key.problem_type.quant_data_type)
    if tensor_name is None:
        raise ValueError("--tensor is required for this quant type")
    size = key.problem_size
    prior_family = args.prior_family
    if prior_family == "auto":
        prior_family = (
            "deepseek" if key.problem_type.quant_data_type == "Q2_K" else "qwen"
        )
    rows_per_batch = 12_288 if prior_family == "deepseek" else 16_384
    if size.m % rows_per_batch:
        raise ValueError(
            f"aggregate rows do not map to the {prior_family} physical batch"
        )
    batch = size.m // rows_per_batch
    if batch not in (1, 4, 16):
        raise ValueError("grouped benchmark supports physical batch 1, 4, or 16")
    available = route_distributions(size.m, batch)
    prior_distributions: tuple[RouteDistribution, ...] = ()
    prior_weights: dict[str, float] = {}
    if args.routing_prior is not None:
        prior_distributions, prior_weights = _prior_distributions(
            args.routing_prior,
            args.prior_bank,
            batch,
            prior_family,
        )
    control_distributions = (
        ()
        if args.prior_only
        else tuple(available[name] for name in _distribution_names(args.distributions))
    )
    distributions = control_distributions + prior_distributions
    if not distributions:
        raise ValueError("no timing distributions were selected")
    packed_weight, quant_type, physical_shape = _load_packed(
        args.model, tensor_name, key
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    grad_output = torch.randn(
        (size.m, size.k),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )

    correctness = None
    timing_reports: list[TimingReport] = []
    with contextlib.ExitStack() as stack:
        module = stack.enter_context(GroupedBackwardModule(key, args.code_object))
        if not args.skip_correctness:
            correctness_rows = (
                size.m if args.correctness_rows <= 0 else args.correctness_rows
            )
            correctness = _correctness(
                module,
                grad_output,
                packed_weight,
                quant_type,
                available["boundary"],
                correctness_rows,
                size.n,
            )
            _require_correctness(correctness)
        if not args.skip_timing:
            for distribution in distributions:
                timing = _timings(
                    module,
                    grad_output,
                    packed_weight,
                    quant_type,
                    distribution,
                    size.n,
                    args.warmup,
                    args.repeats,
                    prior_weights.get(distribution.name),
                    args.reverse_order,
                )
                timing_reports.append(timing)
                print(
                    f"{distribution.name:<8} "
                    f"HIP={timing['hip_complete']['median_ms']:.3f} ms "
                    f"GGT={timing['candidate_complete']['median_ms']:.3f} ms "
                    f"ratio={timing['candidate_to_hip_throughput']:.3f}x",
                    flush=True,
                )

    report = {
        "solution_key": key.to_mapping(),
        "solution_hash": key.hash,
        "kernel_name": key.kernel_name,
        "code_object": str(args.code_object),
        "model": str(args.model),
        "tensor": tensor_name,
        "physical_batch": batch,
        "prior_family": prior_family,
        "physical_weight_shape": physical_shape,
        "grad_output_shape": [size.m, size.k],
        "grad_input_shape": [size.m, size.n],
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "rotating_order": True,
            "base_order": "reverse" if args.reverse_order else "forward",
            "hip_includes_output_and_row_task_allocation": True,
            "candidate_complete_includes_output_allocation": True,
            "candidate_kernel_uses_preallocated_output": True,
        },
        "correctness": correctness,
        "timings": timing_reports,
        "prior_summary": _weighted_prior_summary(timing_reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
