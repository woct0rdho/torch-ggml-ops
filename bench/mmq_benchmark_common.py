"""Shared utilities for dense MMQ forward and backward benchmarks."""

import argparse
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gguf
import numpy as np
import torch

DEFAULT_MODEL = Path(
    os.environ.get(
        "GGUF_MMQ_BENCH_MODEL",
        os.path.expanduser("~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"),
    )
)


@dataclass(frozen=True)
class WeightCase:
    name: str
    tensor_name: str
    expected_out_features: int
    expected_in_features: int
    model_tensor_count: int
    priority: str
    description: str
    lm_head: bool = False


# One real checkpoint tensor represents each (N, K, quant_type) combination
# dispatched by dense MMQ in the model. model_tensor_count records how often
# that exact geometry and quantization appears among the 160 ordinary weights.
CASES = (
    WeightCase(
        "attn_q_q3_k",
        "blk.3.attn_q.weight",
        8192,
        2048,
        9,
        "primary",
        "full-attention query plus query-gate projection",
    ),
    WeightCase(
        "attn_q_q4_k",
        "blk.39.attn_q.weight",
        8192,
        2048,
        1,
        "secondary",
        "final-layer full-attention query plus query-gate projection",
    ),
    WeightCase(
        "narrow_q4_k",
        "blk.5.ffn_gate_shexp.weight",
        512,
        2048,
        70,
        "primary",
        "dominant k/v/shared-gate/shared-up geometry",
    ),
    WeightCase(
        "narrow_q5_k",
        "blk.0.ffn_gate_shexp.weight",
        512,
        2048,
        21,
        "secondary",
        "q5 k/v/shared-gate/shared-up geometry",
    ),
    WeightCase(
        "narrow_q3_k",
        "blk.3.attn_k.weight",
        512,
        2048,
        9,
        "secondary",
        "q3 full-attention key geometry",
    ),
    WeightCase(
        "attn_output_q4_k",
        "blk.3.attn_output.weight",
        2048,
        4096,
        10,
        "primary",
        "full-attention output projection",
    ),
    WeightCase(
        "shared_down_q4_k",
        "blk.5.ffn_down_shexp.weight",
        2048,
        512,
        30,
        "primary",
        "dominant shared-expert down projection",
    ),
    WeightCase(
        "shared_down_q5_k",
        "blk.0.ffn_down_shexp.weight",
        2048,
        512,
        10,
        "secondary",
        "q5 shared-expert down projection",
    ),
    WeightCase(
        "lm_head_q6_k",
        "output.weight",
        248320,
        2048,
        1,
        "primary",
        "chunked language-model head",
        lm_head=True,
    ),
)


def parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    return result


def synchronize() -> None:
    torch.cuda.synchronize()


def cuda_event_times_ms(
    function: Callable[[], object], warmup: int, repeats: int
) -> list[float]:
    for _ in range(warmup):
        output = function()
        del output
    synchronize()

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


def incremental_peak_bytes(function: Callable[[], object]) -> tuple[int, int]:
    synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    output = function()
    synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del output
    return (
        peak_allocated - baseline_allocated,
        max(0, peak_reserved - baseline_reserved),
    )


def summarize_timing(times_ms: list[float], m: int, n: int, k: int) -> dict:
    median_ms = statistics.median(times_ms)
    logical_flops = 2 * m * n * k
    return {
        "samples_ms": times_ms,
        "median_ms": median_ms,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "logical_tflops": logical_flops / (median_ms * 1.0e9),
    }


def make_bf16_input(rows: int, features: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        rows,
        features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )


def load_packed_tensor(tensor: gguf.ReaderTensor) -> torch.Tensor:
    host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    packed = torch.from_numpy(host).to("cuda")
    del host
    return packed


def select_cases(case_names: str, primary_only: bool) -> tuple[WeightCase, ...]:
    by_name = {case.name: case for case in CASES}
    if case_names:
        names = tuple(name.strip() for name in case_names.split(",") if name.strip())
        unknown = sorted(set(names) - set(by_name))
        if unknown:
            raise ValueError(
                f"unknown cases {unknown}; available cases are {sorted(by_name)}"
            )
        cases = tuple(by_name[name] for name in names)
    else:
        cases = CASES
    if primary_only:
        cases = tuple(case for case in cases if case.priority == "primary")
    if not cases:
        raise ValueError("no benchmark cases selected")
    return cases
