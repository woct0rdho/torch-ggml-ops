#!/usr/bin/env python3
"""Benchmark production grouped MMQ forward against BF16 AITER GMM.

The packed path measures the complete public operator, including Q8_1 activation
quantization and grouped multiplication. Gate/up uses grouped_mmq_pair so both
projections share one Q8_1 workspace. The BF16 reference dequantizes the same
GGUF experts once, then runs AITER GMM with the production heuristic from
~/test_no_unsloth/fast_moe_lora.py.
"""

import argparse
import gc
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gguf
import numpy as np
import torch
from aiter.ops.triton.gmm import gmm
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops
from mmq_benchmark_common import (
    DEFAULT_MODEL,
    cuda_event_times_ms,
    incremental_peak_bytes,
    load_packed_tensor,
    make_bf16_input,
    parse_int_list,
    synchronize,
)


DEFAULT_OUTPUT = Path("/tmp/torch_ggml_ops_grouped_mmq_fwd_benchmark.json")
DEFAULT_AITER_HEURISTIC_DIR = Path.home() / "test_no_unsloth"


@dataclass(frozen=True)
class GroupedForwardCase:
    name: str
    kind: str
    tensor_names: tuple[str, ...]
    expected_out_features: int
    expected_in_features: int
    model_layer_count: int
    priority: str
    description: str

    @property
    def projections(self) -> int:
        return len(self.tensor_names)


CASES = (
    GroupedForwardCase(
        "gate_up_q3_k",
        "pair",
        ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"),
        512,
        2048,
        20,
        "primary",
        "paired routed gate/up for layers 0-9 and 30-39",
    ),
    GroupedForwardCase(
        "gate_up_iq2_s",
        "pair",
        ("blk.10.ffn_gate_exps.weight", "blk.10.ffn_up_exps.weight"),
        512,
        2048,
        20,
        "primary",
        "paired routed gate/up for layers 10-29",
    ),
    GroupedForwardCase(
        "down_iq2_s",
        "single",
        ("blk.10.ffn_down_exps.weight",),
        2048,
        512,
        20,
        "primary",
        "routed down projection for layers 10-29",
    ),
    GroupedForwardCase(
        "down_q4_k",
        "single",
        ("blk.2.ffn_down_exps.weight",),
        2048,
        512,
        18,
        "primary",
        "routed down projection for layers 2-9 and 30-39",
    ),
    GroupedForwardCase(
        "down_q5_k",
        "single",
        ("blk.0.ffn_down_exps.weight",),
        2048,
        512,
        2,
        "secondary",
        "routed down projection for layers 0-1",
    ),
)


@dataclass(frozen=True)
class RouteDistribution:
    name: str
    expert_indices_cpu: tuple[int, ...]
    group_sizes_cpu: tuple[int, ...]

    @property
    def rows(self) -> int:
        return sum(self.group_sizes_cpu)


def parse_name_list(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated nonempty list")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batches", type=parse_int_list, default=(1, 4, 16))
    parser.add_argument(
        "--distributions",
        type=parse_name_list,
        default=("uniform", "skewed", "sparse", "boundary"),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--correctness-rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--aiter-heuristic-dir", type=Path, default=DEFAULT_AITER_HEURISTIC_DIR)
    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="comma-separated case names; empty selects every production case",
    )
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"GGUF model not found: {args.model}")
    heuristic_file = args.aiter_heuristic_dir / "fast_moe_lora.py"
    if not heuristic_file.is_file():
        parser.error(f"AITER heuristic not found: {heuristic_file}")
    if args.sequence_length <= 0 or args.top_k <= 0:
        parser.error("--sequence-length and --top-k must be positive")
    if args.warmup < 0 or args.repeats <= 0 or args.correctness_rows <= 0:
        parser.error("warmup must be nonnegative; repeats/correctness rows must be positive")
    known_distributions = {"uniform", "skewed", "sparse", "boundary"}
    unknown = sorted(set(args.distributions) - known_distributions)
    if unknown:
        parser.error(f"unknown distributions {unknown}; expected {sorted(known_distributions)}")
    return args


def select_cases(case_names: str, primary_only: bool) -> tuple[GroupedForwardCase, ...]:
    by_name = {case.name: case for case in CASES}
    if case_names:
        names = tuple(name.strip() for name in case_names.split(",") if name.strip())
        unknown = sorted(set(names) - set(by_name))
        if unknown:
            raise ValueError(f"unknown cases {unknown}; available cases are {sorted(by_name)}")
        selected = tuple(by_name[name] for name in names)
    else:
        selected = CASES
    if primary_only:
        selected = tuple(case for case in selected if case.priority == "primary")
    if not selected:
        raise ValueError("no grouped benchmark cases selected")
    return selected


def load_gmm_config(heuristic_dir: Path) -> Callable[[int, int], dict[str, int]]:
    sys.path.insert(0, str(heuristic_dir))
    try:
        from fast_moe_lora import _gmm_config
    finally:
        sys.path.pop(0)
    return _gmm_config


def adjust_positive_sizes(values: list[int], total: int) -> tuple[int, ...]:
    if not values or total < len(values):
        raise ValueError("cannot construct positive grouped sizes")
    values = [max(1, int(value)) for value in values]
    difference = total - sum(values)
    index = 0
    while difference != 0:
        slot = index % len(values)
        if difference > 0:
            values[slot] += 1
            difference -= 1
        elif values[slot] > 1:
            values[slot] -= 1
            difference += 1
        index += 1
    return tuple(values)


def centered_sizes(total: int, groups: int, amplitude: int) -> tuple[int, ...]:
    center = total / groups
    values = [
        round(center + (((index * 37) % 129) - 64) * amplitude / 64)
        for index in range(groups)
    ]
    return adjust_positive_sizes(values, total)


def route_distributions(rows: int, batch: int) -> dict[str, RouteDistribution]:
    all_experts = tuple(range(256))
    uniform = adjust_positive_sizes([rows // 256] * 256, rows)
    amplitude = {1: 22, 4: 64, 16: 256}.get(batch, max(1, rows // 1024))
    skewed = centered_sizes(rows, 256, amplitude)

    sparse_groups = {1: 192, 4: 224, 16: 240}.get(batch, 224)
    sparse_experts = tuple(
        int(value) for value in np.linspace(0, 255, sparse_groups, dtype=np.int64)
    )
    sparse_amplitude = max(1, round((rows / sparse_groups) * 0.25))
    sparse_sizes = centered_sizes(rows, sparse_groups, sparse_amplitude)

    boundary_prefix = (1, 15, 16, 17, 63, 64, 65, 127, 128, 129)
    tail_groups = 256 - len(boundary_prefix)
    tail_total = rows - sum(boundary_prefix)
    if tail_total < tail_groups:
        raise ValueError(f"rows={rows} is too small for boundary distribution")
    boundary_tail = centered_sizes(
        tail_total,
        tail_groups,
        max(1, round((tail_total / tail_groups) * 0.10)),
    )
    boundary_sizes = boundary_prefix + boundary_tail

    return {
        "uniform": RouteDistribution("uniform", all_experts, uniform),
        "skewed": RouteDistribution("skewed", all_experts, skewed),
        "sparse": RouteDistribution("sparse", sparse_experts, sparse_sizes),
        "boundary": RouteDistribution("boundary", all_experts, boundary_sizes),
    }


def distribution_summary(distribution: RouteDistribution) -> dict:
    sizes = distribution.group_sizes_cpu
    return {
        "active_experts": len(sizes),
        "min_rows": min(sizes),
        "max_rows": max(sizes),
        "mean_rows": statistics.mean(sizes),
        "median_rows": statistics.median(sizes),
        "stdev_rows": statistics.pstdev(sizes),
        "non_multiple_16_groups": sum(size % 16 != 0 for size in sizes),
        "non_multiple_128_groups": sum(size % 128 != 0 for size in sizes),
    }


def device_metadata(
    distribution: RouteDistribution,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expert_indices = torch.tensor(
        distribution.expert_indices_cpu, device="cuda", dtype=torch.int64
    )
    group_sizes = torch.tensor(
        distribution.group_sizes_cpu, device="cuda", dtype=torch.int32
    )
    expert_offsets = group_sizes.cumsum(0).to(torch.int32).contiguous()
    return expert_indices, expert_offsets, group_sizes


def truncate_distribution(
    distribution: RouteDistribution, max_rows: int
) -> RouteDistribution:
    experts = []
    sizes = []
    remaining = min(max_rows, distribution.rows)
    for expert, size in zip(
        distribution.expert_indices_cpu, distribution.group_sizes_cpu, strict=True
    ):
        if remaining <= 0:
            break
        take = min(size, remaining)
        experts.append(expert)
        sizes.append(take)
        remaining -= take
    return RouteDistribution(
        f"{distribution.name}_correctness", tuple(experts), tuple(sizes)
    )


def timing_summary(
    times_ms: list[float], rows: int, n: int, k: int, projections: int
) -> dict:
    median_ms = statistics.median(times_ms)
    logical_flops = 2 * projections * rows * n * k
    return {
        "samples_ms": times_ms,
        "median_ms": median_ms,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "logical_tflops": logical_flops / (median_ms * 1.0e9),
    }


def benchmark_function(
    function: Callable[[], object],
    rows: int,
    n: int,
    k: int,
    projections: int,
    warmup: int,
    repeats: int,
) -> dict:
    times = cuda_event_times_ms(function, warmup, repeats)
    allocated, reserved = incremental_peak_bytes(function)
    result = timing_summary(times, rows, n, k, projections)
    result.update(
        {
            "incremental_peak_allocated_bytes": allocated,
            "incremental_peak_reserved_bytes": reserved,
        }
    )
    return result


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = actual.float() - expected.float()
    reference_rms = expected.float().square().mean().sqrt()
    error_rms = difference.square().mean().sqrt()
    return {
        "reference_rms": float(reference_rms),
        "error_rms": float(error_rms),
        "normalized_rmse": float(error_rms / reference_rms),
        "max_absolute_error": float(difference.abs().max()),
        "different_bf16_elements": int(torch.count_nonzero(actual != expected)),
        "elements": actual.numel(),
    }


def dense_grouped_mmq_reference(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    distribution: RouteDistribution,
    quant_type: int,
    out_features: int,
) -> torch.Tensor:
    outputs = []
    row_begin = 0
    for expert, size in zip(
        distribution.expert_indices_cpu, distribution.group_sizes_cpu, strict=True
    ):
        row_end = row_begin + size
        input_group = input[row_begin:row_end].clone()
        expert_weight = packed_weight[expert].clone()
        outputs.append(
            torch_ggml_ops.mmq(input_group, expert_weight, quant_type, out_features)
        )
        row_begin = row_end
    return torch.cat(outputs, dim=0)


def correctness_metrics(
    case: GroupedForwardCase,
    input: torch.Tensor,
    packed_weights: tuple[torch.Tensor, ...],
    logical_weights: tuple[torch.Tensor, ...],
    distribution: RouteDistribution,
    quant_type: int,
    gmm_config: dict[str, int],
) -> dict:
    checked_distribution = truncate_distribution(distribution, input.shape[0])
    expert_indices, expert_offsets, group_sizes = device_metadata(checked_distribution)
    selected_logical = tuple(
        weight.index_select(0, expert_indices).transpose(1, 2)
        for weight in logical_weights
    )

    with torch.inference_mode():
        if case.kind == "pair":
            actual = torch_ggml_ops.grouped_mmq_pair(
                input,
                packed_weights[0],
                packed_weights[1],
                expert_indices,
                expert_offsets,
                quant_type,
                case.expected_out_features,
            )
            dense_reference = tuple(
                dense_grouped_mmq_reference(
                    input,
                    weight,
                    checked_distribution,
                    quant_type,
                    case.expected_out_features,
                )
                for weight in packed_weights
            )
            bf16_reference = tuple(
                gmm(
                    input,
                    weight,
                    group_sizes,
                    preferred_element_type=input.dtype,
                    config=gmm_config,
                )
                for weight in selected_logical
            )
            return {
                "rows": input.shape[0],
                "same_q8_dense": [
                    error_metrics(actual[index], dense_reference[index])
                    for index in range(2)
                ],
                "bf16_aiter": [
                    error_metrics(actual[index], bf16_reference[index])
                    for index in range(2)
                ],
            }

        actual_single = torch_ggml_ops.grouped_mmq(
            input,
            packed_weights[0],
            expert_indices,
            expert_offsets,
            quant_type,
            case.expected_out_features,
        )
        dense_single = dense_grouped_mmq_reference(
            input,
            packed_weights[0],
            checked_distribution,
            quant_type,
            case.expected_out_features,
        )
        bf16_single = gmm(
            input,
            selected_logical[0],
            group_sizes,
            preferred_element_type=input.dtype,
            config=gmm_config,
        )
        return {
            "rows": input.shape[0],
            "same_q8_dense": error_metrics(actual_single, dense_single),
            "bf16_aiter": error_metrics(actual_single, bf16_single),
        }


def print_result(result: dict) -> None:
    mmq = result["grouped_mmq"]
    aiter = result["aiter_bf16"]
    if result["kind"] == "pair":
        nrmse = max(
            metric["normalized_rmse"]
            for metric in result["correctness"]["bf16_aiter"]
        )
    else:
        nrmse = result["correctness"]["bf16_aiter"]["normalized_rmse"]
    print(
        f"{result['case']:<18} B={result['batch']:>2} "
        f"{result['distribution']:<8} R={result['rows']:>6} "
        f"G={result['group_summary']['active_experts']:>3} "
        f"{result['quant_type']:<5} "
        f"MMQ={mmq['median_ms']:>8.3f} ms {mmq['logical_tflops']:>6.2f} TF "
        f"AITER={aiter['median_ms']:>8.3f} ms {aiter['logical_tflops']:>6.2f} TF "
        f"ratio={result['mmq_to_aiter_tflops_ratio']:>5.2f}x "
        f"NRMSE={nrmse:.3e}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a HIP/CUDA device is required")

    cases = select_cases(args.cases, args.primary_only)
    gmm_config_for = load_gmm_config(args.aiter_heuristic_dir)
    reader = gguf.GGUFReader(args.model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    quant_names = {int(value): value.name for value in gguf.GGMLQuantizationType}
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())

    missing = [
        name for case in cases for name in case.tensor_names if name not in tensors
    ]
    if missing:
        raise RuntimeError(f"checkpoint is missing benchmark tensors: {missing}")

    report = {
        "model": str(args.model),
        "device": {
            "name": properties.name,
            "gcn_arch_name": getattr(properties, "gcnArchName", None),
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
        },
        "configuration": {
            "sequence_length": args.sequence_length,
            "top_k": args.top_k,
            "batches": list(args.batches),
            "distributions": list(args.distributions),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "correctness_rows": args.correctness_rows,
            "aiter_heuristic": str(args.aiter_heuristic_dir / "fast_moe_lora.py"),
            "reference": "BF16 AITER gmm with _gmm_config",
            "aiter_work_stealing": False,
        },
        "results": [],
    }

    print(
        f"device={properties.name} arch={getattr(properties, 'gcnArchName', None)} "
        f"torch={torch.__version__} hip={torch.version.hip}",
        flush=True,
    )
    print(f"model={args.model}", flush=True)

    with torch.inference_mode():
        for case_index, case in enumerate(cases):
            case_tensors = tuple(tensors[name] for name in case.tensor_names)
            quant_types = {int(tensor.tensor_type) for tensor in case_tensors}
            if len(quant_types) != 1:
                raise RuntimeError(f"{case.name} paired tensors have different qtypes")
            quant_type = quant_types.pop()
            quant_name = quant_names.get(quant_type, str(quant_type))

            packed_weights = tuple(load_packed_tensor(tensor) for tensor in case_tensors)
            for tensor, packed in zip(case_tensors, packed_weights, strict=True):
                logical_shape = tuple(int(value) for value in reversed(tensor.shape[:-1]))
                if logical_shape != (
                    case.expected_out_features,
                    case.expected_in_features,
                ):
                    raise RuntimeError(
                        f"{tensor.name} has logical per-expert shape {logical_shape}, "
                        f"expected {(case.expected_out_features, case.expected_in_features)}"
                    )
                if packed.shape[0] != 256:
                    raise RuntimeError(f"{tensor.name} has {packed.shape[0]} experts, expected 256")

            logical_weights = tuple(
                dequantize_gguf_tensor(
                    packed,
                    tensor.tensor_type,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                .reshape(256, case.expected_out_features, case.expected_in_features)
                .contiguous()
                for tensor, packed in zip(case_tensors, packed_weights, strict=True)
            )
            gmm_config = gmm_config_for(
                case.expected_in_features, case.expected_out_features
            )

            for batch_index, batch in enumerate(args.batches):
                rows = batch * args.sequence_length * args.top_k
                available_distributions = route_distributions(rows, batch)
                for distribution_index, distribution_name in enumerate(args.distributions):
                    distribution = available_distributions[distribution_name]
                    expert_indices, expert_offsets, group_sizes = device_metadata(distribution)
                    selected_logical = tuple(
                        weight.index_select(0, expert_indices).transpose(1, 2)
                        for weight in logical_weights
                    )
                    input_seed = (
                        args.seed
                        + case_index * 10000
                        + batch_index * 100
                        + distribution_index
                    )
                    input = make_bf16_input(rows, case.expected_in_features, input_seed)

                    if case.kind == "pair":
                        def mmq_function():
                            return torch_ggml_ops.grouped_mmq_pair(
                                input,
                                packed_weights[0],
                                packed_weights[1],
                                expert_indices,
                                expert_offsets,
                                quant_type,
                                case.expected_out_features,
                            )

                        def aiter_function():
                            return tuple(
                                gmm(
                                    input,
                                    weight,
                                    group_sizes,
                                    preferred_element_type=input.dtype,
                                    config=gmm_config,
                                )
                                for weight in selected_logical
                            )
                    else:
                        def mmq_function():
                            return torch_ggml_ops.grouped_mmq(
                                input,
                                packed_weights[0],
                                expert_indices,
                                expert_offsets,
                                quant_type,
                                case.expected_out_features,
                            )

                        def aiter_function():
                            return gmm(
                                input,
                                selected_logical[0],
                                group_sizes,
                                preferred_element_type=input.dtype,
                                config=gmm_config,
                            )

                    mmq_result = benchmark_function(
                        mmq_function,
                        rows,
                        case.expected_out_features,
                        case.expected_in_features,
                        case.projections,
                        args.warmup,
                        args.repeats,
                    )
                    aiter_result = benchmark_function(
                        aiter_function,
                        rows,
                        case.expected_out_features,
                        case.expected_in_features,
                        case.projections,
                        args.warmup,
                        args.repeats,
                    )

                    checked_distribution = truncate_distribution(
                        distribution, args.correctness_rows
                    )
                    correctness_input = input[: checked_distribution.rows].clone()
                    correctness = correctness_metrics(
                        case,
                        correctness_input,
                        packed_weights,
                        logical_weights,
                        checked_distribution,
                        quant_type,
                        gmm_config,
                    )

                    workspace_bytes = (
                        rows
                        * (case.expected_in_features // (4 * 32))
                        * 144
                    )
                    output_bytes = (
                        case.projections
                        * rows
                        * case.expected_out_features
                        * torch.bfloat16.itemsize
                    )
                    model_calls = case.model_layer_count
                    result = {
                        "case": case.name,
                        "kind": case.kind,
                        "description": case.description,
                        "priority": case.priority,
                        "batch": batch,
                        "rows": rows,
                        "n": case.expected_out_features,
                        "k": case.expected_in_features,
                        "projections": case.projections,
                        "quant_type": quant_name,
                        "quant_type_id": quant_type,
                        "distribution": distribution.name,
                        "group_summary": distribution_summary(distribution),
                        "expert_indices": list(distribution.expert_indices_cpu),
                        "group_sizes": list(distribution.group_sizes_cpu),
                        "aiter_config": dict(gmm_config),
                        "q8_workspace_bytes": workspace_bytes,
                        "expected_output_bytes": output_bytes,
                        "pair_shares_one_q8_workspace": case.kind == "pair",
                        "model_calls_per_forward": model_calls,
                        "checkpointed_calls_per_optimizer_step": 2 * model_calls,
                        "grouped_mmq": mmq_result,
                        "aiter_bf16": aiter_result,
                        "mmq_to_aiter_tflops_ratio": (
                            mmq_result["logical_tflops"]
                            / aiter_result["logical_tflops"]
                        ),
                        "estimated_mmq_optimizer_step_ms": (
                            2 * model_calls * mmq_result["median_ms"]
                        ),
                        "estimated_aiter_optimizer_step_ms": (
                            2 * model_calls * aiter_result["median_ms"]
                        ),
                        "correctness": correctness,
                        "metadata_device_resident": True,
                        "host_group_descriptor_build_in_timed_path": False,
                        "current_stream_operator_contract_tested": True,
                    }
                    report["results"].append(result)
                    print_result(result)

                    del (
                        input,
                        correctness_input,
                        selected_logical,
                        expert_indices,
                        expert_offsets,
                        group_sizes,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    synchronize()

            del logical_weights, packed_weights
            gc.collect()
            torch.cuda.empty_cache()
            synchronize()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
