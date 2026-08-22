#!/usr/bin/env python3

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import gguf
import numpy as np
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ggtensile.benchmark_report import (
    ErrorMetrics,
    error_metrics,
    rotating_timings,
)
from tools.ggtensile.benchmark_routes import (
    fitted_prior_distribution_for_rows,
    make_route_tensors,
)
from tools.ggtensile.grouped_mmq_fwd_pair_model import (
    GroupedForwardPairSolutionKey,
    GroupedPairRouteOwnership,
)
from tools.ggtensile.grouped_mmq_fwd_pair_runtime import (
    GroupedForwardPairModule,
    GroupedForwardPairRowTaskModule,
    GroupedForwardPairRowTaskWorkspace,
    InstalledGroupedForwardPairIQ2XXSSerialControl,
    InstalledGroupedForwardPairQ3RowTaskControl,
    InstalledGroupedForwardPairQ3SerialControl,
    InstalledGroupedForwardPairRowTaskControl,
    InstalledGroupedForwardPairSerialControl,
    InstalledGroupedForwardRowTaskSetup,
)
from tools.ggtensile.grouped_mmq_fwd_pair_spec import DerivedGroupedForwardPairState
from tools.ggtensile.runtime import FixedQ81F32D4QuantizerModule
from tools.ggtensile.workload_prior import EXPERT_PRIOR_NAMES, expert_prior_metadata

MAX_CONTROL_NORMALIZED_RMSE = 5e-4
MAX_CONTROL_ABSOLUTE_ERROR = 0.015625
MAX_REFERENCE_NORMALIZED_RMSE = 0.04


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one exact paired grouped GGTensile forward artifact"
    )
    parser.add_argument("--solution-key", type=Path, required=True)
    parser.add_argument("--code-object", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--first-tensor", required=True)
    parser.add_argument("--second-tensor", required=True)
    parser.add_argument(
        "--expert-prior",
        choices=EXPERT_PRIOR_NAMES,
        required=True,
        help="the sole fitted law used to materialize the routed rows",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--skip-reference", action="store_true")
    return parser


def _load_key(path: Path) -> GroupedForwardPairSolutionKey:
    return GroupedForwardPairSolutionKey.from_mapping(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _load_weight(
    reader: gguf.GGUFReader,
    tensor_name: str,
    key: GroupedForwardPairSolutionKey,
    state: DerivedGroupedForwardPairState,
) -> tuple[gguf.ReaderTensor, torch.Tensor]:
    tensor = next((item for item in reader.tensors if item.name == tensor_name), None)
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {tensor_name}")
    if tensor.tensor_type.name != key.problem.quant_data_type:
        raise ValueError(f"paired tensor {tensor_name} has the wrong quant type")
    logical_shape = tuple(int(value) for value in reversed(tensor.shape[:-1]))
    expected_logical = (
        key.problem.output_features,
        key.problem.input_features,
    )
    if logical_shape != expected_logical:
        raise ValueError(
            f"paired logical shape {logical_shape} does not match {expected_logical}"
        )
    host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
    packed = torch.from_numpy(host).view(state.expected_packed_weight_shape).cuda()
    return tensor, packed


def _control_types(key: GroupedForwardPairSolutionKey) -> tuple[type[Any], Any]:
    quant_type = key.problem.quant_data_type
    row_tasks = key.solution.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks
    if quant_type == "IQ2_S":
        control = (
            InstalledGroupedForwardPairRowTaskControl
            if row_tasks
            else InstalledGroupedForwardPairSerialControl
        )
    elif quant_type == "Q3_K":
        control = (
            InstalledGroupedForwardPairQ3RowTaskControl
            if row_tasks
            else InstalledGroupedForwardPairQ3SerialControl
        )
    elif quant_type == "IQ2_XXS" and not row_tasks:
        control = InstalledGroupedForwardPairIQ2XXSSerialControl
    else:
        raise ValueError("no installed HIP control matches the paired specification")
    candidate = (
        GroupedForwardPairRowTaskModule if row_tasks else GroupedForwardPairModule
    )
    return candidate, control


def _reference(
    input_tensor: torch.Tensor,
    logical_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    group_sizes: tuple[int, ...],
) -> torch.Tensor:
    selected = logical_weight.index_select(0, expert_indices)
    outputs = []
    row_begin = 0
    for index, rows in enumerate(group_sizes):
        row_end = row_begin + rows
        outputs.append(
            torch.mm(input_tensor[row_begin:row_end], selected[index].transpose(0, 1))
        )
        row_begin = row_end
    return torch.cat(outputs)


def _require_correctness(correctness: dict[str, object], reference: bool) -> None:
    for name in ("candidate_vs_hip_kernel", "candidate_vs_public_complete"):
        for metrics in cast(list[ErrorMetrics], correctness[name]):
            if (
                not metrics["finite"]
                or metrics["normalized_rmse"] is None
                or metrics["normalized_rmse"] > MAX_CONTROL_NORMALIZED_RMSE
                or metrics["max_absolute_error"] is None
                or metrics["max_absolute_error"] > MAX_CONTROL_ABSOLUTE_ERROR
            ):
                raise RuntimeError(f"paired correctness failed: {name}")
    if reference:
        for metrics in cast(
            list[ErrorMetrics], correctness["candidate_vs_bf16_reference"]
        ):
            if (
                metrics["normalized_rmse"] is None
                or metrics["normalized_rmse"] > MAX_REFERENCE_NORMALIZED_RMSE
            ):
                raise RuntimeError("paired BF16 reference tolerance failed")
    producer = cast(dict[str, int], correctness["producer_repeat"])
    if producer["different_bytes"]:
        raise RuntimeError("paired Q8_1 producer is not deterministic")


def main() -> None:
    args = _parser().parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    key = _load_key(args.solution_key)
    state = DerivedGroupedForwardPairState.from_solution_key(key)
    reader = gguf.GGUFReader(args.model)
    first_tensor, first_weight = _load_weight(reader, args.first_tensor, key, state)
    second_tensor, second_weight = _load_weight(reader, args.second_tensor, key, state)
    rows = key.problem.aggregate_rows
    distribution = fitted_prior_distribution_for_rows(args.expert_prior, rows)
    assert distribution.profile is not None
    expert_indices, expert_offsets, _ = make_route_tensors(distribution)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    input_tensor = torch.randn(
        (rows, key.problem.input_features),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    first_candidate = torch.empty(
        state.expected_output_shape, dtype=torch.bfloat16, device="cuda"
    )
    second_candidate = torch.empty_like(first_candidate)
    first_hip = torch.empty_like(first_candidate)
    second_hip = torch.empty_like(first_candidate)
    stream = torch.cuda.current_stream().cuda_stream
    candidate_type, control_type = _control_types(key)
    row_tasks = key.solution.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks

    with contextlib.ExitStack() as stack:
        quantizer = stack.enter_context(FixedQ81F32D4QuantizerModule())
        workspace = quantizer.allocate(input_tensor)
        candidate = stack.enter_context(candidate_type(key, args.code_object))
        if key.problem.quant_data_type == "IQ2_XXS":
            first_control = stack.enter_context(control_type(key.solution.macro_tile0))
            second_control = stack.enter_context(control_type(key.solution.macro_tile0))
        else:
            first_control = stack.enter_context(control_type())
            second_control = stack.enter_context(control_type())
        tasks = None
        setup = None
        if row_tasks:
            assert state.kernel_spec.row_task_rows is not None
            tasks = GroupedForwardPairRowTaskWorkspace.allocate(
                input_tensor,
                aggregate_rows=rows,
                route_entries=expert_indices.numel(),
                row_tile=state.kernel_spec.row_task_rows,
            )
            setup = stack.enter_context(InstalledGroupedForwardRowTaskSetup())

        def quantize() -> None:
            quantizer.launch(input_tensor, workspace, stream=stream)

        def setup_tasks() -> None:
            if setup is not None and tasks is not None:
                setup.launch(
                    expert_indices,
                    expert_offsets,
                    tasks,
                    aggregate_rows=rows,
                    stream=stream,
                )

        if row_tasks:
            assert tasks is not None

            def candidate_kernel() -> None:
                candidate.launch(
                    first_weight,
                    second_weight,
                    workspace,
                    first_candidate,
                    second_candidate,
                    tasks,
                    stream=stream,
                )

            def hip_kernel() -> None:
                for control, weight, output in (
                    (first_control, first_weight, first_hip),
                    (second_control, second_weight, second_hip),
                ):
                    control.launch(
                        weight,
                        workspace,
                        output,
                        tasks,
                        aggregate_rows=rows,
                        stream=stream,
                    )

        else:

            def candidate_kernel() -> None:
                candidate.launch(
                    first_weight,
                    second_weight,
                    workspace,
                    first_candidate,
                    second_candidate,
                    expert_indices,
                    expert_offsets,
                    stream=stream,
                )

            def hip_kernel() -> None:
                for control, weight, output in (
                    (first_control, first_weight, first_hip),
                    (second_control, second_weight, second_hip),
                ):
                    control.launch(
                        weight,
                        workspace,
                        output,
                        expert_indices,
                        expert_offsets,
                        aggregate_rows=rows,
                        stream=stream,
                    )

        def candidate_complete() -> None:
            quantize()
            setup_tasks()
            candidate_kernel()

        def hip_complete() -> None:
            quantize()
            setup_tasks()
            hip_kernel()

        quantize()
        workspace_control = workspace.clone()
        quantize()
        setup_tasks()
        task_control = tasks.storage.clone() if tasks is not None else None
        setup_tasks()
        hip_kernel()
        candidate_kernel()
        public_outputs = torch_ggml_ops.grouped_mmq_pair(
            input_tensor,
            first_weight,
            second_weight,
            expert_indices,
            expert_offsets,
            int(first_tensor.tensor_type),
            key.problem.output_features,
        )
        torch.cuda.synchronize()
        correctness: dict[str, object] = {
            "producer_repeat": {
                "different_bytes": int(
                    torch.count_nonzero(workspace != workspace_control)
                ),
                "bytes": workspace.numel(),
            },
            "row_task_setup_repeat": (
                None
                if tasks is None or task_control is None
                else {
                    "different_elements": int(
                        torch.count_nonzero(tasks.storage != task_control)
                    ),
                    "elements": tasks.storage.numel(),
                }
            ),
            "candidate_vs_hip_kernel": [
                error_metrics(first_candidate, first_hip),
                error_metrics(second_candidate, second_hip),
            ],
            "candidate_vs_public_complete": [
                error_metrics(first_candidate, public_outputs[0]),
                error_metrics(second_candidate, public_outputs[1]),
            ],
        }
        if not args.skip_reference:
            references = []
            for tensor, weight in (
                (first_tensor, first_weight),
                (second_tensor, second_weight),
            ):
                logical = dequantize_gguf_tensor(
                    weight,
                    tensor.tensor_type,
                    dtype=torch.bfloat16,
                    device="cuda",
                ).reshape(
                    key.problem.physical_experts,
                    key.problem.output_features,
                    key.problem.input_features,
                )
                references.append(
                    _reference(
                        input_tensor,
                        logical,
                        expert_indices,
                        distribution.group_sizes_cpu,
                    )
                )
            correctness["candidate_vs_bf16_reference"] = [
                error_metrics(first_candidate, references[0]),
                error_metrics(second_candidate, references[1]),
            ]

        logical_flops = (
            4 * rows * key.problem.output_features * key.problem.input_features
        )
        timing = rotating_timings(
            {
                "hip_complete": hip_complete,
                "candidate_complete": candidate_complete,
                "hip_kernel": hip_kernel,
                "candidate_kernel": candidate_kernel,
            },
            warmup=args.warmup,
            repeats=args.repeats,
            logical_flops=logical_flops,
        )

    report = {
        "solution_key": key.to_mapping(),
        "solution_hash": key.hash,
        "kernel_name": key.kernel_name,
        "code_object": str(args.code_object),
        "model": str(args.model),
        "tensors": [args.first_tensor, args.second_tensor],
        "expert_prior": expert_prior_metadata(args.expert_prior),
        "expert_prior_profile": distribution.profile.to_mapping(),
        "route": {
            "expert_indices": list(distribution.expert_indices_cpu),
            "group_sizes": list(distribution.group_sizes_cpu),
        },
        "input_shape": list(input_tensor.shape),
        "packed_weight_shape": list(first_weight.shape),
        "output_shapes": [list(first_candidate.shape), list(second_candidate.shape)],
        "logical_flops": logical_flops,
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "rotating_order": True,
            "row_task_setup_in_complete_timing": row_tasks,
            "complete_includes_quantization": True,
            "complete_includes_output_allocation": False,
            "kernel_uses_preallocated_output": True,
        },
        "correctness_thresholds": {
            "max_control_normalized_rmse": MAX_CONTROL_NORMALIZED_RMSE,
            "max_control_absolute_error": MAX_CONTROL_ABSOLUTE_ERROR,
            "max_reference_normalized_rmse": MAX_REFERENCE_NORMALIZED_RMSE,
        },
        "correctness": correctness,
        "timing": timing,
        "candidate_complete_to_hip_latency": (
            timing["candidate_complete"]["median_ms"]
            / timing["hip_complete"]["median_ms"]
        ),
        "candidate_complete_to_hip_throughput": (
            timing["candidate_complete"]["median_tflops"]
            / timing["hip_complete"]["median_tflops"]
        ),
        "candidate_kernel_to_hip_latency": (
            timing["candidate_kernel"]["median_ms"] / timing["hip_kernel"]["median_ms"]
        ),
        "candidate_kernel_to_hip_throughput": (
            timing["candidate_kernel"]["median_tflops"]
            / timing["hip_kernel"]["median_tflops"]
        ),
    }
    _require_correctness(correctness, not args.skip_reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
