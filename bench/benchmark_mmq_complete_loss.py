#!/usr/bin/env python3
"""Benchmark the complete packed LM-head loss loop across chunk schedules."""

import importlib
import statistics
import sys
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import torch
from mmq_benchmark_common import (
    DenseMMQCase,
    SampleSummary,
    clear_cuda_cache,
    cuda_device_info,
    cuda_event_times_ms,
    incremental_peak_bytes,
    load_gguf_tensors,
    make_benchmark_parser,
    make_bf16_input,
    parse_int_list,
    relative_error,
    resolve_lm_head_chunks,
    select_cases,
    summarize_samples,
    validate_benchmark_args,
    validate_weight_case,
    write_json_report,
)

from tests.mmq_test_support import load_packed_tensor


class ChunkCorrectness(TypedDict):
    loss: float
    loss_relative_to_first_chunk: float | None
    grad_exact_to_first_chunk: bool
    grad_cosine_to_first_chunk: float
    grad_relative_l2_to_first_chunk: float | None


class ChunkPhase(SampleSummary):
    phase: int
    chunk_size: int


def parse_args() -> Namespace:
    parser = make_benchmark_parser(
        __doc__,
        default_output=None,
        seed=20260705,
        repeats=25,
    )
    parser.add_argument("--chunks", type=parse_int_list)
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.add_argument(
        "--loss-module-root",
        type=Path,
        required=True,
        help="directory containing the production gguf_liger_loss module",
    )
    args = parser.parse_args()
    validate_benchmark_args(parser, args)
    return args


def load_loss_function(root: Path) -> Callable[..., tuple[torch.Tensor, ...]]:
    if not (root / "gguf_liger_loss.py").is_file():
        raise FileNotFoundError(f"missing {root / 'gguf_liger_loss.py'}")
    sys.path.insert(0, str(root))
    module = importlib.import_module("gguf_liger_loss")
    return module._packed_q8_linear_cross_entropy_forward


def measure_chunk_correctness(
    run: Callable[[int], tuple[torch.Tensor, ...]],
    chunks: tuple[int, ...],
) -> dict[str, ChunkCorrectness]:
    correctness: dict[str, ChunkCorrectness] = {}
    reference_loss = None
    reference_grad = None
    for chunk_size in chunks:
        loss, _, _, grad_input = run(chunk_size)
        torch.cuda.synchronize()
        loss_value = float(loss)
        if reference_loss is None:
            reference_loss = loss_value
            reference_grad = grad_input.clone()
        assert reference_loss is not None and reference_grad is not None
        grad_float = grad_input.float()
        reference_float = reference_grad.float()
        loss_difference = abs(loss_value - reference_loss)
        grad_difference_norm = float(
            torch.linalg.vector_norm(grad_float - reference_float)
        )
        grad_reference_norm = float(torch.linalg.vector_norm(reference_float))
        correctness[str(chunk_size)] = {
            "loss": loss_value,
            "loss_relative_to_first_chunk": relative_error(
                loss_difference, abs(reference_loss)
            ),
            "grad_exact_to_first_chunk": bool(torch.equal(grad_input, reference_grad)),
            "grad_cosine_to_first_chunk": float(
                torch.nn.functional.cosine_similarity(
                    grad_float.flatten(), reference_float.flatten(), dim=0
                )
            ),
            "grad_relative_l2_to_first_chunk": relative_error(
                grad_difference_norm, grad_reference_norm
            ),
        }
        del loss, grad_input, grad_float, reference_float
    del reference_grad
    return correctness


def measure_chunk_peaks(
    run: Callable[[int], tuple[torch.Tensor, ...]],
    chunks: tuple[int, ...],
) -> dict[str, dict[str, int]]:
    peaks = {}
    for chunk_size in chunks:
        allocated, reserved = incremental_peak_bytes(
            lambda chunk_size=chunk_size: run(chunk_size),
            clear_cache=True,
        )
        peaks[str(chunk_size)] = {
            "incremental_peak_allocated_bytes": allocated,
            "incremental_peak_reserved_bytes": reserved,
        }
    return peaks


def measure_chunk_phases(
    run: Callable[[int], tuple[torch.Tensor, ...]],
    phase_order: tuple[int, ...],
    repeats: int,
    batch: int,
) -> list[ChunkPhase]:
    phases: list[ChunkPhase] = []
    for phase_index, chunk_size in enumerate(phase_order):
        summary = summarize_samples(
            cuda_event_times_ms(
                lambda chunk_size=chunk_size: run(chunk_size),
                warmup=0,
                repeats=repeats,
            )
        )
        phase: ChunkPhase = {
            "phase": phase_index,
            "chunk_size": chunk_size,
            **summary,
        }
        phases.append(phase)
        print(
            f"B={batch} phase={phase_index} M={chunk_size} "
            f"median={phase['median_ms']:.3f} ms",
            flush=True,
        )
    return phases


def benchmark_batch(
    args: Namespace,
    case: DenseMMQCase,
    packed_weight: torch.Tensor,
    quant_type: int,
    chunks: tuple[int, ...],
    phase_order: tuple[int, ...],
    loss_function: Callable[..., tuple[torch.Tensor, ...]],
    batch: int,
) -> dict[str, object]:
    rows = batch * args.sequence_length
    input_tensor = make_bf16_input(rows, case.in_features, args.seed)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    target = torch.randint(
        0,
        case.out_features,
        (rows,),
        dtype=torch.long,
        device="cuda",
        generator=generator,
    )
    target[::257] = args.ignore_index

    def run(
        chunk_size: int,
        input_tensor=input_tensor,
        packed_weight=packed_weight,
        target=target,
    ) -> tuple[torch.Tensor, ...]:
        return loss_function(
            input_tensor,
            packed_weight,
            target,
            quant_type,
            case.out_features,
            chunk_size,
            args.ignore_index,
            0.0,
            0.0,
            "mean",
            None,
            False,
            False,
        )

    for chunk_size in chunks:
        for _ in range(args.warmup):
            output = run(chunk_size)
            del output
        torch.cuda.synchronize()

    correctness = measure_chunk_correctness(run, chunks)
    peaks = measure_chunk_peaks(run, chunks)
    phases = measure_chunk_phases(run, phase_order, args.repeats, batch)
    by_chunk = {}
    for chunk_size in chunks:
        chunk_phases = [phase for phase in phases if phase["chunk_size"] == chunk_size]
        phase_medians = [phase["median_ms"] for phase in chunk_phases]
        by_chunk[str(chunk_size)] = {
            "bracket_median_ms": statistics.fmean(phase_medians),
            "phase_medians_ms": phase_medians,
            **peaks[str(chunk_size)],
            **correctness[str(chunk_size)],
        }

    del run, input_tensor, target
    clear_cuda_cache()
    return {
        "batch": batch,
        "rows": rows,
        "by_chunk": by_chunk,
        "phases": phases,
    }


def main() -> None:
    args = parse_args()
    device = cuda_device_info()
    loss_function = load_loss_function(args.loss_module_root)
    chunks = resolve_lm_head_chunks(args.chunks, args.model_family)
    phase_order = chunks + tuple(reversed(chunks))
    lm_cases = tuple(
        case for case in select_cases("", False, args.model_family) if case.lm_head
    )
    if len(lm_cases) != 1:
        raise RuntimeError(
            f"expected one {args.model_family} LM-head case, found {len(lm_cases)}"
        )
    case = lm_cases[0]
    reader, tensors = load_gguf_tensors(args.model, (case.tensor_name,))
    tensor = tensors[case.tensor_name]
    quant_type, quant_name, physical_shape = validate_weight_case(tensor, case)
    packed_weight = load_packed_tensor(tensor, case.out_features)
    report = {
        "model": str(args.model),
        "model_family": args.model_family,
        "operation": "complete_loss",
        "device": device,
        "configuration": {
            "sequence_length": args.sequence_length,
            "batches": list(args.batches),
            "in_features": case.in_features,
            "out_features": case.out_features,
            "physical_weight_shape": list(physical_shape),
            "quant_type": quant_name,
            "quant_type_id": quant_type,
            "chunks": list(chunks),
            "phase_order": list(phase_order),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "scope": (
                "MMQ forward + in-place Liger cross entropy + packed MMQ grad-input"
            ),
        },
        "results": [],
    }

    for batch in args.batches:
        report["results"].append(
            benchmark_batch(
                args,
                case,
                packed_weight,
                quant_type,
                chunks,
                phase_order,
                loss_function,
                batch,
            )
        )
        write_json_report(args.output, report)

    del reader
    write_json_report(args.output, report)
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
