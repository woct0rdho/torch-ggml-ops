#!/usr/bin/env python3

import argparse
import contextlib
import json
import statistics
import sys
from pathlib import Path

import gguf
import numpy as np
import torch

import torch_ggml_ops  # noqa: F401 Register the installed HIP kernels.

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ggtensile.inspection import inspect_artifact  # noqa: E402
from tools.ggtensile.kernel_writer_assembly_mmq_bwd import (  # noqa: E402
    DiagnosticMode,
    KernelWriterAssembly,
)
from tools.ggtensile.model import SolutionKey  # noqa: E402
from tools.ggtensile.runtime import DenseBackwardModule  # noqa: E402
from tools.ggtensile.toolchain import Toolchain  # noqa: E402

DEFAULT_MODEL = Path.home() / "models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"
DEFAULT_TENSOR = "blk.0.ffn_gate_shexp.weight"
BF16_WMMA_ROOFLINE_TFLOPS = 59.4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and time exact GGTensile lower-bound diagnostics"
    )
    parser.add_argument("--solution-key", type=Path, required=True)
    parser.add_argument("--complete-code-object", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tensor", default=DEFAULT_TENSOR)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser


def _event_time(function) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _timing_summary(samples_ms: list[float]) -> dict[str, object]:
    return {
        "SamplesMs": samples_ms,
        "MedianMs": statistics.median(samples_ms),
        "MeanMs": statistics.fmean(samples_ms),
        "MinMs": min(samples_ms),
        "MaxMs": max(samples_ms),
    }


def _build_diagnostic(
    key: SolutionKey,
    toolchain: Toolchain,
    output_dir: Path,
    mode: DiagnosticMode,
) -> tuple[Path, str, dict[str, object]]:
    stem = mode.value
    assembly = output_dir / f"{stem}.s"
    object_path = output_dir / f"{stem}.o"
    code_object = output_dir / f"{stem}.hsaco"
    source_hash = KernelWriterAssembly(
        key,
        toolchain,
        diagnostic_mode=mode,
    ).write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    normal_wmmas = (
        key.solution.matrix_instruction[5]
        * key.solution.matrix_instruction[6]
        * key.solution.depth_u
        // 16
    )
    expected_wmmas = normal_wmmas if mode == DiagnosticMode.WMMA_FLOOR else 0
    inspection = inspect_artifact(
        key,
        code_object,
        toolchain,
        expected_wmma_count=expected_wmmas,
        expected_barrier_count=1,
    )
    return code_object, source_hash, inspection.to_mapping()


def main() -> None:
    args = _parser().parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    key = SolutionKey.from_json_file(args.solution_key)
    toolchain = Toolchain.discover()
    complete_inspection = inspect_artifact(key, args.complete_code_object, toolchain)
    artifacts = {}
    for mode in DiagnosticMode:
        code_object, source_hash, inspection = _build_diagnostic(
            key,
            toolchain,
            args.output_dir,
            mode,
        )
        artifacts[mode] = code_object
        artifacts[f"{mode.value}_source_hash"] = source_hash
        artifacts[f"{mode.value}_inspection"] = inspection

    reader = gguf.GGUFReader(args.model)
    tensor = next((item for item in reader.tensors if item.name == args.tensor), None)
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {args.tensor}")
    expected_quant = key.problem_type.quant_data_type
    if tensor.tensor_type.name != expected_quant:
        raise ValueError(
            f"expected {expected_quant} tensor, found {tensor.tensor_type.name}"
        )
    size = key.problem_size
    logical_shape = tuple(int(value) for value in reversed(tensor.shape))
    if logical_shape != (size.k, size.n):
        raise ValueError(
            f"tensor shape {logical_shape} does not match K,N={(size.k, size.n)}"
        )

    packed_host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    packed_weight = torch.from_numpy(packed_host).cuda()
    del packed_host
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    grad_output = torch.randn(
        size.m,
        size.k,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    outputs = {
        name: torch.empty((size.m, size.n), dtype=torch.bfloat16, device="cuda")
        for name in ("complete", "wmma_floor", "decode_floor")
    }

    with contextlib.ExitStack() as stack:
        modules = {
            "complete": stack.enter_context(
                DenseBackwardModule(key, args.complete_code_object)
            ),
            "wmma_floor": stack.enter_context(
                DenseBackwardModule(key, artifacts[DiagnosticMode.WMMA_FLOOR])
            ),
            "decode_floor": stack.enter_context(
                DenseBackwardModule(key, artifacts[DiagnosticMode.DECODE_FLOOR])
            ),
        }

        def launch(name: str) -> None:
            modules[name].launch(
                grad_output,
                packed_weight,
                outputs[name],
                stream=torch.cuda.current_stream().cuda_stream,
            )

        names = list(modules)
        for _ in range(args.warmup):
            for name in names:
                launch(name)
        torch.cuda.synchronize()

        samples = {name: [] for name in names}
        for repeat in range(args.repeats):
            offset = repeat % len(names)
            for name in names[offset:] + names[:offset]:
                samples[name].append(_event_time(lambda name=name: launch(name)))

    timings = {name: _timing_summary(values) for name, values in samples.items()}
    logical_flops = 2 * size.m * size.n * size.k
    for name in ("complete", "wmma_floor"):
        median_ms = timings[name]["MedianMs"]
        assert isinstance(median_ms, float)
        tflops = logical_flops / (median_ms * 1.0e9)
        timings[name]["EquivalentTflops"] = tflops
        timings[name]["WmmaRooflineFraction"] = tflops / BF16_WMMA_ROOFLINE_TFLOPS

    complete_ms = timings["complete"]["MedianMs"]
    wmma_ms = timings["wmma_floor"]["MedianMs"]
    decode_ms = timings["decode_floor"]["MedianMs"]
    assert isinstance(complete_ms, float)
    assert isinstance(wmma_ms, float)
    assert isinstance(decode_ms, float)
    report = {
        "SolutionKey": key.to_mapping(),
        "CompleteCodeObject": str(args.complete_code_object),
        "CompleteInspection": complete_inspection.to_mapping(),
        "Diagnostics": {
            mode.value: {
                "CodeObject": str(artifacts[mode]),
                "AssemblySHA256": artifacts[f"{mode.value}_source_hash"],
                "Inspection": artifacts[f"{mode.value}_inspection"],
            }
            for mode in DiagnosticMode
        },
        "Model": str(args.model),
        "Tensor": args.tensor,
        "Timings": timings,
        "WmmaFloorToCompleteLatency": wmma_ms / complete_ms,
        "DecodeFloorToCompleteLatency": decode_ms / complete_ms,
        "FloorSumToCompleteLatency": (wmma_ms + decode_ms) / complete_ms,
    }
    report_path = args.output_dir / "lower_bounds.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
