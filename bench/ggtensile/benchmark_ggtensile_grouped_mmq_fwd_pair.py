#!/usr/bin/env python3

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch

import torch_ggml_ops

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BENCH_ROOT = REPO_ROOT / "bench"
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from benchmark_report import error_metrics, rotating_timings
from benchmark_routes import (
    fitted_prior_distribution_for_rows,
    make_route_tensors,
)
from benchmark_support import require_case
from workload_prior import EXPERT_PRIOR_NAMES, expert_prior_metadata

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
from tools.mmq_correctness import (
    dequantize_active_routed_weight,
    routed_forward_reference,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one exact paired grouped GGTensile forward artifact"
    )
    parser.add_argument("--solution-key", type=Path, required=True)
    parser.add_argument("--code-object", type=Path, required=True)
    parser.add_argument("--hip-code-object", type=Path)
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
    parser.add_argument("--skip-timing", action="store_true")
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


def main() -> None:
    args = _parser().parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    key = _load_key(args.solution_key)
    state = DerivedGroupedForwardPairState.from_solution_key(key)
    case = require_case(
        "GroupedForwardPair",
        key,
        args.code_object,
        (args.first_tensor, args.second_tensor),
    )
    reader = gguf.GGUFReader(args.model)
    first_tensor, first_weight = _load_weight(reader, args.first_tensor, key, state)
    second_tensor, second_weight = _load_weight(reader, args.second_tensor, key, state)
    rows = key.problem.aggregate_rows
    distribution = fitted_prior_distribution_for_rows(args.expert_prior, rows)
    assert distribution.profile is not None
    expert_indices, expert_offsets, group_sizes = make_route_tensors(distribution)
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
    hip_reason = "no legacy HIP control artifact was supplied"

    with contextlib.ExitStack() as stack:
        quantizer = stack.enter_context(FixedQ81F32D4QuantizerModule())
        workspace = quantizer.allocate(input_tensor)
        candidate = stack.enter_context(candidate_type(key, args.code_object))
        controls = None
        if args.hip_code_object is not None:
            if key.problem.quant_data_type == "IQ2_XXS":
                controls = (
                    stack.enter_context(
                        control_type(key.solution.macro_tile0, args.hip_code_object)
                    ),
                    stack.enter_context(
                        control_type(key.solution.macro_tile0, args.hip_code_object)
                    ),
                )
            else:
                controls = (
                    stack.enter_context(control_type(args.hip_code_object)),
                    stack.enter_context(control_type(args.hip_code_object)),
                )
            hip_reason = None
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
                if controls is None:
                    return
                for control, weight, output in (
                    (controls[0], first_weight, first_hip),
                    (controls[1], second_weight, second_hip),
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
                if controls is None:
                    return
                for control, weight, output in (
                    (controls[0], first_weight, first_hip),
                    (controls[1], second_weight, second_hip),
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

        def public_api() -> tuple[torch.Tensor, torch.Tensor]:
            return torch_ggml_ops.grouped_mmq_pair(
                input_tensor,
                first_weight,
                second_weight,
                expert_indices,
                expert_offsets,
                int(first_tensor.tensor_type),
                key.problem.output_features,
            )

        quantize()
        setup_tasks()
        hip_kernel()
        candidate_kernel()
        public_outputs = public_api()
        torch.cuda.synchronize()
        correctness: dict[str, object] = {
            "candidate_vs_public_api": [
                error_metrics(first_candidate, public_outputs[0]),
                error_metrics(second_candidate, public_outputs[1]),
            ],
        }
        if controls is not None:
            correctness["candidate_vs_hip"] = [
                error_metrics(first_candidate, first_hip),
                error_metrics(second_candidate, second_hip),
            ]
        if not args.skip_reference:
            references = []
            for tensor, weight in (
                (first_tensor, first_weight),
                (second_tensor, second_weight),
            ):
                logical = dequantize_active_routed_weight(
                    weight,
                    tensor.tensor_type,
                    expert_indices,
                    key.problem.output_features,
                    key.problem.input_features,
                )
                references.append(
                    routed_forward_reference(
                        input_tensor, logical, expert_indices, group_sizes
                    )
                )
            correctness["candidate_vs_bf16_baseline"] = [
                error_metrics(first_candidate, references[0]),
                error_metrics(second_candidate, references[1]),
            ]

        del public_outputs
        if not args.skip_reference:
            del references, logical
        logical_flops = (
            4 * rows * key.problem.output_features * key.problem.input_features
        )
        timing = {}
        if not args.skip_timing:
            functions = {
                "public_api": public_api,
                "ggtensile_complete": candidate_complete,
                "ggtensile_kernel": candidate_kernel,
            }
            if controls is not None:
                functions["hip_complete"] = hip_complete
                functions["hip_kernel"] = hip_kernel
            timing = rotating_timings(
                functions,
                warmup=args.warmup,
                repeats=args.repeats,
                logical_flops=logical_flops,
            )

    report = {
        **case.to_mapping(),
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
            "row_task_setup_in_complete_timing": row_tasks,
            "correctness_before_timing": True,
        },
        "implementations": {
            "public_api": {
                "available": True,
                "label": "torch_ggml_ops.grouped_mmq_pair",
            },
            "ggtensile": {"available": True, "artifact": str(args.code_object)},
            "hip": {"available": controls is not None, "reason": hip_reason},
            "bf16_baseline": {
                "available": not args.skip_reference,
                "label": "routed torch.mm on dequantized BF16 banks",
            },
        },
        "correctness": correctness,
        "timing": timing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
