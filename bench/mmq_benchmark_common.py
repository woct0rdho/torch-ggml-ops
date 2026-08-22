import argparse
import gc
import json
import math
import os
import statistics
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict, TypeVar, cast

import gguf
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = Path(
    os.environ.get(
        "GGUF_MMQ_BENCH_MODEL",
        os.path.expanduser("~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"),
    )
)
DEFAULT_LM_HEAD_CHUNKS = {
    "qwen": (64, 128, 256),
    "deepseek": (32, 64, 128, 256, 512),
}


@dataclass(frozen=True)
class DenseMMQCase:
    name: str
    tensor_name: str
    out_features: int
    in_features: int
    model_calls: int
    description: str
    quant_type: str
    priority: str = "primary"
    lm_head: bool = False


class _SelectableCase(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def priority(self) -> str: ...


_CaseT = TypeVar("_CaseT", bound=_SelectableCase)


class SampleSummary(TypedDict):
    samples_ms: list[float]
    median_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float


class TimingSummary(SampleSummary):
    logical_tflops: float


class BenchmarkTiming(TimingSummary):
    incremental_peak_allocated_bytes: int
    incremental_peak_reserved_bytes: int


class PerformanceComparison(TypedDict):
    packed_to_reference_tflops_ratio: float
    estimated_packed_model_ms: float
    estimated_reference_model_ms: float


class ErrorMetrics(TypedDict):
    reference_rms: float
    error_rms: float
    normalized_rmse: float | None
    max_absolute_error: float
    different_bf16_elements: int
    elements: int


# One real checkpoint tensor represents each (N, K, quant_type) combination
# dispatched by dense MMQ in the model. model_calls records how often
# that exact geometry and quantization appears among the 160 ordinary weights.
QWEN_DENSE_CASES = (
    DenseMMQCase(
        "attn_q_q3_k",
        "blk.3.attn_q.weight",
        8192,
        2048,
        9,
        "full-attention query plus query-gate projection",
        "Q3_K",
    ),
    DenseMMQCase(
        "attn_q_q4_k",
        "blk.39.attn_q.weight",
        8192,
        2048,
        1,
        "final-layer full-attention query plus query-gate projection",
        "Q4_K",
        priority="secondary",
    ),
    DenseMMQCase(
        "narrow_q4_k",
        "blk.5.ffn_gate_shexp.weight",
        512,
        2048,
        70,
        "dominant k/v/shared-gate/shared-up geometry",
        "Q4_K",
    ),
    DenseMMQCase(
        "narrow_q5_k",
        "blk.0.ffn_gate_shexp.weight",
        512,
        2048,
        21,
        "q5 k/v/shared-gate/shared-up geometry",
        "Q5_K",
        priority="secondary",
    ),
    DenseMMQCase(
        "narrow_q3_k",
        "blk.3.attn_k.weight",
        512,
        2048,
        9,
        "q3 full-attention key geometry",
        "Q3_K",
        priority="secondary",
    ),
    DenseMMQCase(
        "attn_output_q4_k",
        "blk.3.attn_output.weight",
        2048,
        4096,
        10,
        "full-attention output projection",
        "Q4_K",
    ),
    DenseMMQCase(
        "shared_down_q4_k",
        "blk.5.ffn_down_shexp.weight",
        2048,
        512,
        30,
        "dominant shared-expert down projection",
        "Q4_K",
    ),
    DenseMMQCase(
        "shared_down_q5_k",
        "blk.0.ffn_down_shexp.weight",
        2048,
        512,
        10,
        "q5 shared-expert down projection",
        "Q5_K",
        priority="secondary",
    ),
    DenseMMQCase(
        "lm_head_q6_k",
        "output.weight",
        248320,
        2048,
        1,
        "chunked language-model head",
        "Q6_K",
        lm_head=True,
    ),
)


DEEPSEEK_DENSE_CASES = (
    DenseMMQCase(
        "ds4_attn_q_a_q8_0",
        "blk.0.attn_q_a.weight",
        1024,
        4096,
        43,
        "attention query A projection",
        "Q8_0",
    ),
    DenseMMQCase(
        "ds4_attn_q_b_q8_0",
        "blk.0.attn_q_b.weight",
        32768,
        1024,
        43,
        "attention query B projection",
        "Q8_0",
    ),
    DenseMMQCase(
        "ds4_attn_kv_q8_0",
        "blk.0.attn_kv.weight",
        512,
        4096,
        43,
        "attention key/value projection",
        "Q8_0",
    ),
    DenseMMQCase(
        "ds4_attn_output_b_q8_0",
        "blk.0.attn_output_b.weight",
        4096,
        8192,
        43,
        "attention output B projection",
        "Q8_0",
    ),
    DenseMMQCase(
        "ds4_shared_gate_up_q8_0",
        "blk.0.ffn_gate_shexp.weight",
        2048,
        4096,
        86,
        "shared-expert gate/up geometry",
        "Q8_0",
    ),
    DenseMMQCase(
        "ds4_shared_down_q8_0",
        "blk.0.ffn_down_shexp.weight",
        4096,
        2048,
        43,
        "shared-expert down projection",
        "Q8_0",
    ),
    DenseMMQCase(
        "ds4_lm_head_q8_0",
        "output.weight",
        129280,
        4096,
        1,
        "chunked language-model head",
        "Q8_0",
        lm_head=True,
    ),
)

DENSE_CASES_BY_MODEL_FAMILY = {
    "qwen": QWEN_DENSE_CASES,
    "deepseek": DEEPSEEK_DENSE_CASES,
}


def parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    return result


def make_benchmark_parser(
    description: str,
    *,
    default_output: Path | None,
    seed: int,
    repeats: int,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-family",
        choices=("qwen", "deepseek"),
        required=True,
    )
    if default_output is None:
        parser.add_argument("--output", type=Path, required=True)
    else:
        parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batches", type=parse_int_list, default=(1, 4, 16))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=repeats)
    parser.add_argument("--seed", type=int, default=seed)
    return parser


def validate_benchmark_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if not args.model.is_file():
        parser.error(f"GGUF model not found: {args.model}")
    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")


def parse_dense_benchmark_args(
    description: str,
    default_output: Path,
    seed: int,
    *,
    transient_bf16_control: bool = False,
) -> argparse.Namespace:
    parser = make_benchmark_parser(
        description,
        default_output=default_output,
        seed=seed,
        repeats=7,
    )
    parser.add_argument(
        "--lm-head-chunks",
        type=parse_int_list,
        default=None,
        help="comma-separated chunks; defaults to the selected model family",
    )
    parser.add_argument("--correctness-rows", type=int, default=8)
    parser.add_argument(
        "--cases",
        default="",
        help="comma-separated case names; empty selects every production case",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="benchmark only cases marked primary",
    )
    if transient_bf16_control:
        parser.add_argument(
            "--transient-bf16-control",
            action="store_true",
            help=(
                "time an optimistic transient representation floor: allocate and "
                "copy a predecoded BF16 weight, then run torch.mm"
            ),
        )
    args = parser.parse_args()
    validate_benchmark_args(parser, args)
    if args.correctness_rows <= 0:
        parser.error("--correctness-rows must be positive")
    return args


def clear_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def cuda_event_times_ms(
    function: Callable[[], object], warmup: int, repeats: int
) -> list[float]:
    for _ in range(warmup):
        output = function()
        del output
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
        del output
    return times


def incremental_peak_bytes(
    function: Callable[[], object],
    *,
    clear_cache: bool = False,
) -> tuple[int, int]:
    if clear_cache:
        clear_cuda_cache()
    else:
        torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    output = function()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del output
    return (
        peak_allocated - baseline_allocated,
        max(0, peak_reserved - baseline_reserved),
    )


def summarize_samples(values: list[float]) -> SampleSummary:
    return {
        "samples_ms": values,
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": statistics.fmean(values),
    }


def summarize_timing(
    times_ms: list[float],
    rows: int,
    out_features: int,
    in_features: int,
    projections: int = 1,
) -> TimingSummary:
    summary = summarize_samples(times_ms)
    logical_flops = 2 * projections * rows * out_features * in_features
    return {
        **summary,
        "logical_tflops": logical_flops / (summary["median_ms"] * 1.0e9),
    }


def benchmark_callable(
    function: Callable[[], object],
    rows: int,
    out_features: int,
    in_features: int,
    warmup: int,
    repeats: int,
    *,
    projections: int = 1,
) -> BenchmarkTiming:
    times = cuda_event_times_ms(function, warmup, repeats)
    allocated, reserved = incremental_peak_bytes(function)
    return {
        **summarize_timing(
            times,
            rows,
            out_features,
            in_features,
            projections,
        ),
        "incremental_peak_allocated_bytes": allocated,
        "incremental_peak_reserved_bytes": reserved,
    }


def performance_comparison(
    packed: TimingSummary,
    reference: TimingSummary,
    model_calls: int,
) -> PerformanceComparison:
    return {
        "packed_to_reference_tflops_ratio": (
            packed["logical_tflops"] / reference["logical_tflops"]
        ),
        "estimated_packed_model_ms": model_calls * packed["median_ms"],
        "estimated_reference_model_ms": model_calls * reference["median_ms"],
    }


def relative_error(error: float, reference: float) -> float | None:
    if reference != 0.0:
        return error / reference
    return 0.0 if error == 0.0 else None


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> ErrorMetrics:
    difference = actual.float() - expected.float()
    reference_rms = float(expected.float().square().mean().sqrt())
    error_rms = float(difference.square().mean().sqrt())
    return {
        "reference_rms": reference_rms,
        "error_rms": error_rms,
        "normalized_rmse": relative_error(error_rms, reference_rms),
        "max_absolute_error": float(difference.abs().max()),
        "different_bf16_elements": int(torch.count_nonzero(actual != expected)),
        "elements": actual.numel(),
    }


def cuda_device_info() -> dict[str, object]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "name": properties.name,
        "gcn_arch_name": getattr(properties, "gcnArchName", None),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
    }


def print_benchmark_header(
    device: Mapping[str, object],
    model: Path,
    model_family: str,
    case_names: tuple[str, ...] = (),
) -> None:
    print(
        f"device={device['name']} arch={device['gcn_arch_name']} "
        f"torch={device['torch_version']} hip={device['hip_version']}",
        flush=True,
    )
    cases = f" cases={','.join(case_names)}" if case_names else ""
    print(f"model={model} family={model_family}{cases}", flush=True)


def print_dense_result(result: Mapping[str, object]) -> None:
    packed = cast(BenchmarkTiming, result["packed"])
    reference = cast(BenchmarkTiming, result["bf16_reference"])
    correctness = cast(ErrorMetrics, result["correctness"])
    normalized_rmse = correctness["normalized_rmse"]
    assert normalized_rmse is not None
    print(
        f"{result['case']:<23} "
        f"B={result['batch']:>2} calls={result['model_calls']:>3} "
        f"M={result['rows']:>6} N={result['out_features']:>6} "
        f"K={result['in_features']:>4} {result['quant_type']:<5} "
        f"PACKED={packed['median_ms']:>8.3f} ms "
        f"{packed['logical_tflops']:>6.2f} TF "
        f"BF16={reference['median_ms']:>8.3f} ms "
        f"{reference['logical_tflops']:>6.2f} TF "
        f"ratio={result['packed_to_reference_tflops_ratio']:>5.2f}x "
        f"NRMSE={normalized_rmse:.3e}",
        flush=True,
    )


def load_gguf_tensors(
    model: Path,
    required_names: tuple[str, ...],
) -> tuple[gguf.GGUFReader, dict[str, gguf.ReaderTensor]]:
    reader = gguf.GGUFReader(model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    missing = sorted(set(required_names) - set(tensors))
    if missing:
        raise RuntimeError(f"checkpoint is missing benchmark tensors: {missing}")
    return reader, tensors


def load_packed_tensor(
    tensor: gguf.ReaderTensor,
    out_features: int | None = None,
) -> torch.Tensor:
    data = tensor.data if out_features is None else tensor.data[:out_features]
    host = np.array(data, dtype=np.uint8, copy=True, order="C")
    packed = torch.from_numpy(host).to("cuda")
    del host
    return packed


def write_json_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def make_bf16_input(rows: int, features: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        rows,
        features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )


def resolve_lm_head_chunks(
    requested: tuple[int, ...] | None,
    model_family: str,
) -> tuple[int, ...]:
    return DEFAULT_LM_HEAD_CHUNKS[model_family] if requested is None else requested


def validate_weight_case(
    tensor: gguf.ReaderTensor,
    case: DenseMMQCase,
) -> tuple[int, str, tuple[int, ...]]:
    logical_shape = tuple(int(value) for value in reversed(tensor.shape))
    expected_shape = (case.out_features, case.in_features)
    if logical_shape != expected_shape:
        raise RuntimeError(
            f"{case.tensor_name} has logical shape {logical_shape}, expected "
            f"{expected_shape}"
        )
    quant_type = int(tensor.tensor_type)
    quant_name = tensor.tensor_type.name
    if quant_name != case.quant_type:
        raise RuntimeError(
            f"{case.tensor_name} has quant type {quant_name}, expected "
            f"{case.quant_type}"
        )
    physical_shape = tuple(int(value) for value in tensor.data.shape)
    return quant_type, quant_name, physical_shape


def dense_result_metadata(
    case: DenseMMQCase,
    row_spec: Mapping[str, int],
    sequence_length: int,
    physical_weight_shape: tuple[int, ...],
    quant_name: str,
    quant_type: int,
) -> dict[str, object]:
    calls_per_weight = row_spec["calls"]
    return {
        "case": case.name,
        "description": case.description,
        "priority": case.priority,
        "tensor_name": case.tensor_name,
        "model_weight_count": case.model_calls,
        "batch": row_spec["batch"],
        "sequence_length": sequence_length,
        "rows": row_spec["m"],
        "out_features": case.out_features,
        "in_features": case.in_features,
        "logical_weight_shape": [case.out_features, case.in_features],
        "physical_weight_shape": list(physical_weight_shape),
        "quant_type": quant_name,
        "quant_type_id": quant_type,
        "model_rows": row_spec["model_rows"],
        "calls_per_weight": calls_per_weight,
        "model_calls": calls_per_weight * case.model_calls,
    }


def make_row_specs(
    case: DenseMMQCase,
    batches: tuple[int, ...],
    sequence_length: int,
    lm_head_chunks: tuple[int, ...],
) -> tuple[list[dict[str, int]], tuple[int, ...]]:
    if case.lm_head:
        specs = [
            {
                "batch": batch,
                "m": chunk,
                "model_rows": batch * sequence_length,
                "calls": math.ceil(batch * sequence_length / chunk),
            }
            for chunk in lm_head_chunks
            for batch in batches
        ]
        return specs, lm_head_chunks
    specs = [
        {
            "batch": batch,
            "m": batch * sequence_length,
            "model_rows": batch * sequence_length,
            "calls": 1,
        }
        for batch in batches
    ]
    return specs, tuple(spec["m"] for spec in specs)


def select_family_cases(
    case_names: str,
    primary_only: bool,
    model_family: str,
    cases_by_model_family: Mapping[str, tuple[_CaseT, ...]],
    *,
    description: str = "benchmark",
) -> tuple[_CaseT, ...]:
    if model_family not in cases_by_model_family:
        raise ValueError(
            f"unknown model family {model_family!r}; available families are "
            f"{sorted(cases_by_model_family)}"
        )
    available_cases = cases_by_model_family[model_family]

    by_name = {case.name: case for case in available_cases}
    if case_names:
        names = tuple(name.strip() for name in case_names.split(",") if name.strip())
        unknown = sorted(set(names) - set(by_name))
        if unknown:
            raise ValueError(
                f"unknown {description} cases {unknown}; available cases are "
                f"{sorted(by_name)}"
            )
        selected = tuple(by_name[name] for name in names)
    else:
        selected = available_cases
    if primary_only:
        selected = tuple(case for case in selected if case.priority == "primary")
    if not selected:
        raise ValueError(f"no {description} cases selected")
    return selected


def select_cases(
    case_names: str,
    primary_only: bool,
    model_family: str,
) -> tuple[DenseMMQCase, ...]:
    return select_family_cases(
        case_names,
        primary_only,
        model_family,
        DENSE_CASES_BY_MODEL_FAMILY,
        description="dense benchmark",
    )
