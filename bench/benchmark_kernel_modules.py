"""Prepare allocation-free GGTensile/HIP launch pairs for benchmarks."""

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tools.ggtensile.dense_mmq_bwd_runtime import InstalledDenseBackwardModule
from tools.ggtensile.family_registry import instance_name
from tools.ggtensile.fixed_grouped_mmq_bwd_model import FixedBackwardProblem
from tools.ggtensile.fixed_grouped_mmq_bwd_spec import FixedBackwardKernelSpec
from tools.ggtensile.fixed_grouped_mmq_fwd_model import FixedForwardProblem
from tools.ggtensile.fixed_grouped_mmq_fwd_spec import FixedForwardKernelSpec
from tools.ggtensile.grouped_mmq_bwd_pair_model import GroupedBackwardPairProblem
from tools.ggtensile.grouped_mmq_bwd_pair_runtime import (
    GroupedBackwardPairModule,
    InstalledGroupedBackwardPairIQ2SControl,
    InstalledGroupedBackwardPairIQ2XXSControl,
    InstalledGroupedBackwardPairQ3KControl,
)
from tools.ggtensile.grouped_mmq_bwd_pair_spec import GroupedBackwardPairKernelSpec
from tools.ggtensile.grouped_mmq_bwd_runtime import InstalledGroupedBackwardControl
from tools.ggtensile.grouped_mmq_bwd_spec import GroupedBackwardKernelSpec
from tools.ggtensile.grouped_mmq_fwd_model import GroupedForwardProblem
from tools.ggtensile.grouped_mmq_fwd_pair_model import (
    GroupedForwardPairProblem,
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
from tools.ggtensile.grouped_mmq_fwd_pair_spec import (
    DerivedGroupedForwardPairState,
    GroupedForwardPairKernelSpec,
)
from tools.ggtensile.grouped_mmq_fwd_spec import GroupedForwardKernelSpec
from tools.ggtensile.mmq_bwd_spec import BackwardKernelSpec
from tools.ggtensile.mmq_fwd_spec import ForwardKernelSpec
from tools.ggtensile.model import ProblemSize
from tools.ggtensile.runtime import (
    BackwardModule,
    FixedGroupedQ8BackwardModule,
    FixedGroupedQ8ForwardModule,
    FixedHipForwardModule,
    FixedQ81F16D2S6QuantizerModule,
    FixedQ81F16D4S4QuantizerModule,
    FixedQ81F32D4QuantizerModule,
    ForwardModule,
    GroupedBackwardModule,
    GroupedForwardModule,
    InstalledFixedGroupedQ8BackwardModule,
    InstalledFixedGroupedQ8ForwardModule,
    InstalledGroupedForwardIQ2SJ64J32Module,
    InstalledGroupedForwardIQ2SJ64Module,
    InstalledGroupedForwardModule,
    InstalledGroupedForwardQ2J32J16Module,
    InstalledGroupedForwardQ2J32Module,
    InstalledGroupedForwardQ5J32Module,
    InstalledGroupedForwardQ5Module,
)
from tools.mmq_correctness import PreparedCase
from tools.mmq_deployment_cases import DeploymentCase, public_artifact_path


@dataclass
class KernelComparison:
    launch_ggtensile: Callable[[], object]
    launch_hip: Callable[[], object]
    ggtensile_output: Any
    hip_output: Any
    metadata: dict[str, object]


def _quantizer(case: DeploymentCase):
    if case.quant_type == "Q2_K":
        return FixedQ81F16D2S6QuantizerModule
    if case.quant_type in {"Q4_K", "Q5_K"} and case.operation in {
        "OrdinaryForward",
        "GroupedForward",
    }:
        return FixedQ81F16D4S4QuantizerModule
    return FixedQ81F32D4QuantizerModule


def _grouped_forward_control(case: DeploymentCase, route_entries: int, hip_root: Path):
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, GroupedForwardProblem)
    assert isinstance(spec, GroupedForwardKernelSpec)
    if case.quant_type == "Q4_K":
        return InstalledGroupedForwardModule(problem, spec, hip_root)
    if case.quant_type == "Q5_K":
        cls = (
            InstalledGroupedForwardQ5J32Module
            if case.rows < 128 * route_entries
            else InstalledGroupedForwardQ5Module
        )
        return cls(problem, spec, hip_root)
    if case.quant_type == "Q2_K":
        cls = (
            InstalledGroupedForwardQ2J32J16Module
            if case.rows == 49_152 or case.rows < 64 * route_entries
            else InstalledGroupedForwardQ2J32Module
        )
        return cls(problem, spec, hip_root)
    if case.quant_type == "IQ2_S":
        cls = (
            InstalledGroupedForwardIQ2SJ64J32Module
            if case.rows == 65_536 or case.rows < 128 * route_entries
            else InstalledGroupedForwardIQ2SJ64Module
        )
        return cls(problem, spec, hip_root)
    raise ValueError(f"no grouped-forward HIP control for {case.quant_type}")


def _pair_forward_control(
    problem: GroupedForwardPairProblem,
    spec: GroupedForwardPairKernelSpec,
    row_tasks: bool,
    hip_root: Path,
):
    if problem.quant_data_type == "IQ2_S":
        cls = (
            InstalledGroupedForwardPairRowTaskControl
            if row_tasks
            else InstalledGroupedForwardPairSerialControl
        )
        return cls(hip_root)
    if problem.quant_data_type == "Q3_K":
        cls = (
            InstalledGroupedForwardPairQ3RowTaskControl
            if row_tasks
            else InstalledGroupedForwardPairQ3SerialControl
        )
        return cls(hip_root)
    if problem.quant_data_type == "IQ2_XXS" and not row_tasks:
        return InstalledGroupedForwardPairIQ2XXSSerialControl(
            spec.geometry.macro_tile[0], hip_root
        )
    raise ValueError("no paired-forward HIP control for the selected specification")


def _pair_backward_control(
    problem: GroupedBackwardPairProblem,
    spec: GroupedBackwardPairKernelSpec,
    hip_root: Path,
):
    controls = {
        "Q3_K": InstalledGroupedBackwardPairQ3KControl,
        "IQ2_S": InstalledGroupedBackwardPairIQ2SControl,
        "IQ2_XXS": InstalledGroupedBackwardPairIQ2XXSControl,
    }
    control = controls.get(problem.quant_data_type)
    if control is None:
        raise ValueError(
            f"no paired-backward HIP control for {problem.quant_data_type}"
        )
    return control(spec.compute.geometry.macro_tile0, hip_root)


def _metadata(
    case: DeploymentCase, artifact: Path, hip_modules: tuple[Any, ...]
) -> dict[str, object]:
    return {
        "ggtensile": {"artifact": str(artifact), "symbol": case.symbol},
        "hip": {
            "artifacts": [str(module.code_object) for module in hip_modules],
            "launches_per_comparison_call": len(hip_modules),
        },
    }


def _ordinary_forward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    artifact: Path,
    hip_root: Path,
) -> KernelComparison:
    if prepared.input is None:
        raise ValueError("ordinary forward input is missing")
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, ProblemSize)
    assert isinstance(spec, ForwardKernelSpec)
    stream = torch.cuda.current_stream().cuda_stream
    quantizer = stack.enter_context(_quantizer(case)())
    workspace = quantizer.allocate(prepared.input)
    ggtensile = stack.enter_context(
        ForwardModule(
            problem,
            case.quant_type,
            spec,
            artifact,
            instance_name(case.instance),
        )
    )
    hip = stack.enter_context(
        FixedHipForwardModule(problem, case.quant_type, spec, hip_root)
    )
    ggtensile_output = torch.empty(
        case.rows, case.out_features, device="cuda", dtype=torch.bfloat16
    )
    hip_output = torch.empty_like(ggtensile_output)
    quantizer.launch(prepared.input, workspace, stream=stream)

    def launch_ggtensile() -> torch.Tensor:
        ggtensile.launch(
            prepared.packed_weights[0], workspace, ggtensile_output, stream=stream
        )
        return ggtensile_output

    def launch_hip() -> torch.Tensor:
        hip.launch(prepared.packed_weights[0], workspace, hip_output, stream=stream)
        return hip_output

    return KernelComparison(
        launch_ggtensile,
        launch_hip,
        ggtensile_output,
        hip_output,
        _metadata(case, artifact, (hip,)),
    )


def _ordinary_backward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    artifact: Path,
    hip_root: Path,
) -> KernelComparison:
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, ProblemSize)
    assert isinstance(spec, BackwardKernelSpec)
    grad_output = prepared.grad_outputs[0]
    stream = torch.cuda.current_stream().cuda_stream
    ggtensile = stack.enter_context(
        BackwardModule(
            problem,
            case.quant_type,
            spec,
            artifact,
            instance_name(case.instance),
        )
    )
    hip = stack.enter_context(
        InstalledDenseBackwardModule(problem, case.quant_type, spec, hip_root)
    )
    ggtensile_output = torch.empty(
        case.rows, case.in_features, device="cuda", dtype=torch.bfloat16
    )
    hip_output = torch.empty_like(ggtensile_output)

    def launch_ggtensile() -> torch.Tensor:
        ggtensile.launch(
            grad_output,
            prepared.packed_weights[0],
            ggtensile_output,
            stream=stream,
        )
        return ggtensile_output

    def launch_hip() -> torch.Tensor:
        hip.launch(grad_output, prepared.packed_weights[0], hip_output, stream=stream)
        return hip_output

    return KernelComparison(
        launch_ggtensile,
        launch_hip,
        ggtensile_output,
        hip_output,
        _metadata(case, artifact, (hip,)),
    )


def _grouped_forward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    artifact: Path,
    hip_root: Path,
) -> KernelComparison:
    input_tensor = prepared.input
    route = prepared.route
    if input_tensor is None or route is None:
        raise ValueError("grouped forward input or routes are missing")
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, GroupedForwardProblem)
    assert isinstance(spec, GroupedForwardKernelSpec)
    stream = torch.cuda.current_stream().cuda_stream
    quantizer = stack.enter_context(_quantizer(case)())
    workspace = quantizer.allocate(input_tensor)
    ggtensile = stack.enter_context(
        GroupedForwardModule(problem, spec, artifact, instance_name(case.instance))
    )
    hip = stack.enter_context(
        _grouped_forward_control(case, route.expert_indices.numel(), hip_root)
    )
    ggtensile_output = torch.empty(
        case.rows, case.out_features, device="cuda", dtype=torch.bfloat16
    )
    hip_output = torch.empty_like(ggtensile_output)
    quantizer.launch(input_tensor, workspace, stream=stream)

    def launch(module, output) -> torch.Tensor:
        module.launch(
            prepared.packed_weights[0],
            workspace,
            output,
            route.expert_indices,
            route.expert_offsets,
            stream=stream,
        )
        return output

    return KernelComparison(
        lambda: launch(ggtensile, ggtensile_output),
        lambda: launch(hip, hip_output),
        ggtensile_output,
        hip_output,
        _metadata(case, artifact, (hip,)),
    )


def _grouped_backward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    artifact: Path,
    hip_root: Path,
) -> KernelComparison:
    route = prepared.route
    if route is None:
        raise ValueError("grouped backward routes are missing")
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, ProblemSize)
    assert isinstance(spec, GroupedBackwardKernelSpec)
    stream = torch.cuda.current_stream().cuda_stream
    ggtensile = stack.enter_context(
        GroupedBackwardModule(
            problem,
            case.quant_type,
            spec,
            artifact,
            instance_name(case.instance),
        )
    )
    hip = stack.enter_context(
        InstalledGroupedBackwardControl(
            case.quant_type,
            case.rows,
            route.expert_indices.numel(),
            hip_root,
        )
    )
    ggtensile_output = torch.empty(
        case.rows, case.in_features, device="cuda", dtype=torch.bfloat16
    )
    hip_output = torch.empty_like(ggtensile_output)

    def launch(module, output) -> torch.Tensor:
        module.launch(
            prepared.grad_outputs[0],
            prepared.packed_weights[0],
            output,
            route.expert_indices,
            route.expert_offsets,
            stream=stream,
        )
        return output

    return KernelComparison(
        lambda: launch(ggtensile, ggtensile_output),
        lambda: launch(hip, hip_output),
        ggtensile_output,
        hip_output,
        _metadata(case, artifact, (hip,)),
    )


def _grouped_pair_forward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    artifact: Path,
    hip_root: Path,
) -> KernelComparison:
    input_tensor = prepared.input
    route = prepared.route
    if input_tensor is None or route is None:
        raise ValueError("grouped pair forward input or routes are missing")
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, GroupedForwardPairProblem)
    assert isinstance(spec, GroupedForwardPairKernelSpec)
    row_tasks = spec.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks
    stream = torch.cuda.current_stream().cuda_stream
    quantizer = stack.enter_context(_quantizer(case)())
    workspace = quantizer.allocate(input_tensor)
    candidate_type = (
        GroupedForwardPairRowTaskModule if row_tasks else GroupedForwardPairModule
    )
    ggtensile: Any = stack.enter_context(
        candidate_type(problem, spec, artifact, instance_name(case.instance))
    )
    first_hip: Any = stack.enter_context(
        _pair_forward_control(problem, spec, row_tasks, hip_root)
    )
    second_hip: Any = stack.enter_context(
        _pair_forward_control(problem, spec, row_tasks, hip_root)
    )
    first_ggtensile = torch.empty(
        case.rows, case.out_features, device="cuda", dtype=torch.bfloat16
    )
    second_ggtensile = torch.empty_like(first_ggtensile)
    first_hip_output = torch.empty_like(first_ggtensile)
    second_hip_output = torch.empty_like(first_ggtensile)
    tasks: Any = None
    if row_tasks:
        state = DerivedGroupedForwardPairState.from_problem_spec(problem, spec)
        if state.kernel_spec.row_task_rows is None:
            raise ValueError("row-task specification is missing its task row count")
        tasks = GroupedForwardPairRowTaskWorkspace.allocate(
            input_tensor,
            aggregate_rows=case.rows,
            route_entries=route.expert_indices.numel(),
            row_tile=state.kernel_spec.row_task_rows,
        )
        setup = stack.enter_context(InstalledGroupedForwardRowTaskSetup(hip_root))
    quantizer.launch(input_tensor, workspace, stream=stream)
    if row_tasks:
        setup.launch(
            route.expert_indices,
            route.expert_offsets,
            tasks,
            aggregate_rows=case.rows,
            stream=stream,
        )

    def launch_ggtensile() -> tuple[torch.Tensor, torch.Tensor]:
        if row_tasks:
            ggtensile.launch(
                prepared.packed_weights[0],
                prepared.packed_weights[1],
                workspace,
                first_ggtensile,
                second_ggtensile,
                tasks,
                stream=stream,
            )
        else:
            ggtensile.launch(
                prepared.packed_weights[0],
                prepared.packed_weights[1],
                workspace,
                first_ggtensile,
                second_ggtensile,
                route.expert_indices,
                route.expert_offsets,
                stream=stream,
            )
        return first_ggtensile, second_ggtensile

    def launch_hip() -> tuple[torch.Tensor, torch.Tensor]:
        for module, weight, output in (
            (first_hip, prepared.packed_weights[0], first_hip_output),
            (second_hip, prepared.packed_weights[1], second_hip_output),
        ):
            if row_tasks:
                module.launch(
                    weight,
                    workspace,
                    output,
                    tasks,
                    aggregate_rows=case.rows,
                    stream=stream,
                )
            else:
                module.launch(
                    weight,
                    workspace,
                    output,
                    route.expert_indices,
                    route.expert_offsets,
                    aggregate_rows=case.rows,
                    stream=stream,
                )
        return first_hip_output, second_hip_output

    metadata = _metadata(case, artifact, (first_hip, second_hip))
    metadata["row_task_setup_in_timing"] = False
    return KernelComparison(
        launch_ggtensile,
        launch_hip,
        (first_ggtensile, second_ggtensile),
        (first_hip_output, second_hip_output),
        metadata,
    )


def _grouped_pair_backward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    artifact: Path,
    hip_root: Path,
) -> KernelComparison:
    route = prepared.route
    if route is None:
        raise ValueError("grouped pair backward routes are missing")
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, GroupedBackwardPairProblem)
    assert isinstance(spec, GroupedBackwardPairKernelSpec)
    stream = torch.cuda.current_stream().cuda_stream
    ggtensile = stack.enter_context(
        GroupedBackwardPairModule(problem, spec, artifact, instance_name(case.instance))
    )
    hip = stack.enter_context(_pair_backward_control(problem, spec, hip_root))
    ggtensile_output = torch.empty(
        case.rows, case.in_features, device="cuda", dtype=torch.bfloat16
    )
    hip_output = torch.empty_like(ggtensile_output)

    def launch(module, output) -> torch.Tensor:
        module.launch(
            prepared.grad_outputs[0],
            prepared.grad_outputs[1],
            prepared.packed_weights[0],
            prepared.packed_weights[1],
            output,
            route.expert_indices,
            route.expert_offsets,
            stream=stream,
        )
        return output

    return KernelComparison(
        lambda: launch(ggtensile, ggtensile_output),
        lambda: launch(hip, hip_output),
        ggtensile_output,
        hip_output,
        _metadata(case, artifact, (hip,)),
    )


def _fixed_forward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    artifact: Path,
    hip_root: Path,
) -> KernelComparison:
    if prepared.input is None:
        raise ValueError("fixed forward input is missing")
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, FixedForwardProblem)
    assert isinstance(spec, FixedForwardKernelSpec)
    stream = torch.cuda.current_stream().cuda_stream
    flat_input = prepared.input.view(case.rows * 8, case.in_features)
    quantizer = stack.enter_context(_quantizer(case)())
    workspace = quantizer.allocate(flat_input)
    ggtensile = stack.enter_context(
        FixedGroupedQ8ForwardModule(
            problem, spec, artifact, instance_name(case.instance)
        )
    )
    hip = stack.enter_context(
        InstalledFixedGroupedQ8ForwardModule(problem, spec, hip_root)
    )
    ggtensile_output = torch.empty(
        case.rows, 8, case.out_features, device="cuda", dtype=torch.bfloat16
    )
    hip_output = torch.empty_like(ggtensile_output)
    quantizer.launch(flat_input, workspace, stream=stream)

    def launch(module, output) -> torch.Tensor:
        module.launch(prepared.packed_weights[0], workspace, output, stream=stream)
        return output

    return KernelComparison(
        lambda: launch(ggtensile, ggtensile_output),
        lambda: launch(hip, hip_output),
        ggtensile_output,
        hip_output,
        _metadata(case, artifact, (hip,)),
    )


def _fixed_backward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    artifact: Path,
    hip_root: Path,
) -> KernelComparison:
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, FixedBackwardProblem)
    assert isinstance(spec, FixedBackwardKernelSpec)
    stream = torch.cuda.current_stream().cuda_stream
    ggtensile = stack.enter_context(
        FixedGroupedQ8BackwardModule(
            problem, spec, artifact, instance_name(case.instance)
        )
    )
    hip = stack.enter_context(
        InstalledFixedGroupedQ8BackwardModule(problem, spec, hip_root)
    )
    ggtensile_output = torch.empty(
        case.rows, 8, case.in_features, device="cuda", dtype=torch.bfloat16
    )
    hip_output = torch.empty_like(ggtensile_output)

    def launch(module, output) -> torch.Tensor:
        module.launch(
            prepared.grad_outputs[0],
            prepared.packed_weights[0],
            output,
            stream=stream,
        )
        return output

    return KernelComparison(
        lambda: launch(ggtensile, ggtensile_output),
        lambda: launch(hip, hip_output),
        ggtensile_output,
        hip_output,
        _metadata(case, artifact, (hip,)),
    )


_PREPARERS = {
    "OrdinaryForward": _ordinary_forward,
    "OrdinaryBackward": _ordinary_backward,
    "GroupedForward": _grouped_forward,
    "GroupedBackward": _grouped_backward,
    "GroupedForwardPair": _grouped_pair_forward,
    "GroupedBackwardPair": _grouped_pair_backward,
    "FixedGroupedForward": _fixed_forward,
    "FixedGroupedBackward": _fixed_backward,
}


@contextlib.contextmanager
def prepare_kernel_comparison(
    case: DeploymentCase,
    prepared: PreparedCase,
    *,
    ggtensile_root: Path | None,
    hip_root: Path,
) -> Iterator[KernelComparison]:
    artifact = public_artifact_path(case, ggtensile_root)
    if not artifact.is_file():
        raise FileNotFoundError(f"deployed GGTensile artifact not found: {artifact}")
    with contextlib.ExitStack() as stack:
        comparison = _PREPARERS[case.operation](
            stack,
            case,
            prepared,
            artifact,
            hip_root,
        )
        torch.cuda.synchronize()
        yield comparison
