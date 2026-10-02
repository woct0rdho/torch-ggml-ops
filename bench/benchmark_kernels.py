"""Prepare direct HIP and GGTensile kernel implementations."""

import contextlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from bench.benchmark_common import Implementation
from bench.benchmark_data import RouteSelection
from tools.ggtensile.family_registry import instance_name
from tools.ggtensile.fixed_grouped_mmq_bwd_model import FixedBackwardProblem
from tools.ggtensile.fixed_grouped_mmq_bwd_spec import FixedBackwardKernelSpec
from tools.ggtensile.fixed_grouped_mmq_fwd_model import FixedForwardProblem
from tools.ggtensile.fixed_grouped_mmq_fwd_spec import FixedForwardKernelSpec
from tools.ggtensile.grouped_mmq_bwd_pair_model import GroupedBackwardPairProblem
from tools.ggtensile.grouped_mmq_bwd_pair_spec import GroupedBackwardPairKernelSpec
from tools.ggtensile.grouped_mmq_bwd_spec import GroupedBackwardKernelSpec
from tools.ggtensile.grouped_mmq_fwd_model import GroupedForwardProblem
from tools.ggtensile.grouped_mmq_fwd_pair_model import (
    GroupedForwardPairProblem,
    GroupedPairRouteOwnership,
)
from tools.ggtensile.grouped_mmq_fwd_pair_spec import (
    DerivedGroupedForwardPairState,
    GroupedForwardPairKernelSpec,
)
from tools.ggtensile.grouped_mmq_fwd_spec import GroupedForwardKernelSpec
from tools.ggtensile.mmq_bwd_spec import BackwardKernelSpec
from tools.ggtensile.mmq_fwd_spec import ForwardKernelSpec
from tools.ggtensile.model import ProblemSize
from tools.mmq_correctness import PreparedCase
from tools.mmq_deployment_cases import DeploymentCase, public_artifact_path
from tools.mmq_hip_dense import InstalledDenseBackwardModule
from tools.mmq_hip_deployment import (
    select_grouped_forward_control,
    select_routed_control,
)
from tools.mmq_hip_launchers import (
    build_backward_control,
    build_backward_pair_control,
    build_forward_pair_control,
    is_row_task_body,
    quantizer_module,
)
from tools.mmq_hip_row_task import (
    GroupedForwardPairRowTaskWorkspace,
    InstalledGroupedForwardRowTaskSetup,
    InstalledGroupedRowTaskSetup,
    RowTaskWorkspace,
)
from tools.mmq_pair_runtime import (
    GroupedBackwardPairModule,
    GroupedForwardPairModule,
    GroupedForwardPairRowTaskModule,
)
from tools.mmq_runtime import (
    BackwardModule,
    FixedGroupedQ8BackwardModule,
    FixedGroupedQ8ForwardModule,
    FixedHipForwardModule,
    ForwardModule,
    GroupedBackwardModule,
    GroupedForwardModule,
    InstalledFixedGroupedQ8BackwardModule,
    InstalledFixedGroupedQ8ForwardModule,
    InstalledGroupedForwardModule,
)


def _grouped_forward_control(case: DeploymentCase, route_entries: int, hip_root: Path):
    """Resolve the deployed HIP grouped-forward control for one route bank."""

    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, GroupedForwardProblem)
    assert isinstance(spec, GroupedForwardKernelSpec)

    control = select_grouped_forward_control(
        case.quant_type,
        case.out_features,
        case.in_features,
        case.rows,
        route_entries,
    )
    return InstalledGroupedForwardModule(problem, spec, control, hip_root)


def _pair_forward_controls(
    case: DeploymentCase, selection: RouteSelection, hip_root: Path
):
    """Build both projection controls for one deployed paired-forward body."""

    entries = max(route.expert_indices.numel() for route in selection.routes)
    choice = select_routed_control(
        "GroupedForwardPair",
        case.quant_type,
        case.out_features,
        case.in_features,
        case.rows,
        entries,
    )
    return choice, [build_forward_pair_control(choice, hip_root) for _ in range(2)]


def _pair_backward_control(
    case: DeploymentCase, selection: RouteSelection, hip_root: Path
):
    """Build the deployed paired-backward control for one route bank."""

    entries = max(route.expert_indices.numel() for route in selection.routes)
    choice = select_routed_control(
        "GroupedBackwardPair",
        case.quant_type,
        case.out_features,
        case.in_features,
        case.rows,
        entries,
    )
    return build_backward_pair_control(choice, hip_root)


def _metadata(
    name: str,
    case: DeploymentCase,
    *,
    artifact: Path | None = None,
    modules: Iterable[Any] = (),
) -> dict[str, object]:
    if name == "ggtensile":
        assert artifact is not None
        return {"artifact": str(artifact), "symbol": case.symbol}
    return {"artifacts": [str(module.code_object) for module in modules]}


def _forward_resources(
    stack: contextlib.ExitStack, case: DeploymentCase, prepared: PreparedCase
) -> tuple[torch.Tensor, int]:
    if prepared.input is None:
        raise ValueError("forward input is missing")
    input_tensor = prepared.input
    source = (
        input_tensor.view(case.rows * 8, case.in_features)
        if case.operation.startswith("Fixed")
        else input_tensor
    )
    quantizer = stack.enter_context(quantizer_module(case.operation, case.quant_type)())
    workspace = quantizer.allocate(source)
    stream = torch.cuda.current_stream().cuda_stream
    quantizer.launch(source, workspace, stream=stream)
    return workspace, stream


def _prepare_ordinary_forward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    names: tuple[str, ...],
    artifact: Path | None,
    hip_root: Path | None,
) -> dict[str, Implementation]:
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, ProblemSize)
    assert isinstance(spec, ForwardKernelSpec)
    workspace, stream = _forward_resources(stack, case, prepared)
    implementations = {}
    for name in names:
        if name == "ggtensile":
            if artifact is None:
                raise ValueError("GGTensile artifact is missing")
            module: Any = stack.enter_context(
                ForwardModule(
                    problem,
                    case.quant_type,
                    spec,
                    artifact,
                    instance_name(case.instance),
                )
            )
        else:
            if hip_root is None:
                raise ValueError("HIP root is missing")
            module = stack.enter_context(
                FixedHipForwardModule(problem, case.quant_type, spec, hip_root)
            )
        output = torch.empty(
            case.rows, case.out_features, device="cuda", dtype=torch.bfloat16
        )

        def launch(module=module, output=output) -> torch.Tensor:
            module.launch(prepared.packed_weights[0], workspace, output, stream=stream)
            return output

        implementations[name] = Implementation(
            name, launch, _metadata(name, case, artifact=artifact, modules=(module,))
        )
    return implementations


def _prepare_ordinary_backward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    names: tuple[str, ...],
    artifact: Path | None,
    hip_root: Path | None,
) -> dict[str, Implementation]:
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, ProblemSize)
    assert isinstance(spec, BackwardKernelSpec)
    grad_output = prepared.grad_outputs[0]
    stream = torch.cuda.current_stream().cuda_stream
    implementations = {}
    for name in names:
        if name == "ggtensile":
            if artifact is None:
                raise ValueError("GGTensile artifact is missing")
            module: Any = stack.enter_context(
                BackwardModule(
                    problem,
                    case.quant_type,
                    spec,
                    artifact,
                    instance_name(case.instance),
                )
            )
        else:
            if hip_root is None:
                raise ValueError("HIP root is missing")
            module = stack.enter_context(
                InstalledDenseBackwardModule(problem, case.quant_type, spec, hip_root)
            )
        output = torch.empty(
            case.rows, case.in_features, device="cuda", dtype=torch.bfloat16
        )

        def launch(module=module, output=output) -> torch.Tensor:
            module.launch(
                grad_output, prepared.packed_weights[0], output, stream=stream
            )
            return output

        implementations[name] = Implementation(
            name, launch, _metadata(name, case, artifact=artifact, modules=(module,))
        )
    return implementations


def _prepare_grouped_forward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    selection: RouteSelection,
    names: tuple[str, ...],
    artifact: Path | None,
    hip_root: Path | None,
) -> dict[str, Implementation]:
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, GroupedForwardProblem)
    assert isinstance(spec, GroupedForwardKernelSpec)
    assert prepared.input is not None
    workspace, stream = _forward_resources(stack, case, prepared)
    implementations = {}
    ggtensile = None
    if "ggtensile" in names:
        if artifact is None:
            raise ValueError("GGTensile artifact is missing")
        ggtensile = stack.enter_context(
            GroupedForwardModule(problem, spec, artifact, instance_name(case.instance))
        )

    hip_by_entries = {}
    if "hip" in names:
        if hip_root is None:
            raise ValueError("HIP root is missing")
        for route in selection.routes:
            entries = route.expert_indices.numel()
            if entries not in hip_by_entries:
                hip_by_entries[entries] = stack.enter_context(
                    _grouped_forward_control(case, entries, hip_root)
                )

    for name in names:
        output = torch.empty(
            case.rows, case.out_features, device="cuda", dtype=torch.bfloat16
        )
        if name == "ggtensile":
            assert ggtensile is not None
            module: Any = ggtensile

            def launch(module=module, output=output) -> torch.Tensor:
                route = selection.current
                module.launch(
                    prepared.packed_weights[0],
                    workspace,
                    output,
                    route.expert_indices,
                    route.expert_offsets,
                    stream=stream,
                )
                return output

            module_values = (ggtensile,)
        else:

            def launch(output=output) -> torch.Tensor:
                route = selection.current
                hip_by_entries[route.expert_indices.numel()].launch(
                    prepared.packed_weights[0],
                    workspace,
                    output,
                    route.expert_indices,
                    route.expert_offsets,
                    stream=stream,
                )
                return output

            module_values = tuple(hip_by_entries.values())
        implementations[name] = Implementation(
            name,
            launch,
            _metadata(
                name,
                case,
                artifact=artifact,
                modules=module_values,
            ),
        )
    return implementations


def _prepare_grouped_backward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    selection: RouteSelection,
    names: tuple[str, ...],
    artifact: Path | None,
    hip_root: Path | None,
) -> dict[str, Implementation]:
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, ProblemSize)
    assert isinstance(spec, GroupedBackwardKernelSpec)
    stream = torch.cuda.current_stream().cuda_stream
    implementations = {}
    ggtensile = None
    if "ggtensile" in names:
        if artifact is None:
            raise ValueError("GGTensile artifact is missing")
        ggtensile = stack.enter_context(
            GroupedBackwardModule(
                problem, case.quant_type, spec, artifact, instance_name(case.instance)
            )
        )
    hip_by_entries = {}
    hip_modules: list[Any] = []
    hip_control = None
    row_task_controls = {}
    if "hip" in names:
        if hip_root is None:
            raise ValueError("HIP root is missing")
        entries = max(route.expert_indices.numel() for route in selection.routes)
        choice = select_routed_control(
            "GroupedBackward",
            case.quant_type,
            case.out_features,
            case.in_features,
            case.rows,
            entries,
        )
        if is_row_task_body(choice.symbol):
            setup = stack.enter_context(InstalledGroupedRowTaskSetup())
            hip_modules.append(setup)
            for index, route in enumerate(selection.routes):
                entries = route.expert_indices.numel()
                workspace = RowTaskWorkspace.allocate(
                    prepared.grad_outputs[0],
                    aggregate_rows=case.rows,
                    route_entries=entries,
                )
                setup.launch(
                    route.expert_indices,
                    route.expert_offsets,
                    workspace,
                    aggregate_rows=case.rows,
                    stream=stream,
                )
                row_task_controls[index] = (
                    stack.enter_context(build_backward_control(choice, hip_root)),
                    workspace,
                )
        else:
            key = (case.quant_type, case.rows, entries)
            if key not in hip_by_entries:
                hip_by_entries[key] = stack.enter_context(
                    build_backward_control(choice, hip_root)
                )
                hip_modules.append(hip_by_entries[key])
            hip_control = hip_by_entries[key]
    for name in names:
        output = torch.empty(
            case.rows, case.in_features, device="cuda", dtype=torch.bfloat16
        )
        if name == "ggtensile":
            assert ggtensile is not None
            module: Any = ggtensile

            def launch(module=module, output=output) -> torch.Tensor:
                route = selection.current
                module.launch(
                    prepared.grad_outputs[0],
                    prepared.packed_weights[0],
                    output,
                    route.expert_indices,
                    route.expert_offsets,
                    stream=stream,
                )
                return output

            module_values = (ggtensile,)
        else:

            def launch(output=output) -> torch.Tensor:
                route = selection.current
                if row_task_controls:
                    control, workspace = row_task_controls[selection.index]
                    control.launch(
                        prepared.grad_outputs[0],
                        prepared.packed_weights[0],
                        output,
                        workspace,
                        stream=stream,
                    )
                    return output
                assert hip_control is not None
                hip_control.launch(
                    prepared.grad_outputs[0],
                    prepared.packed_weights[0],
                    output,
                    route.expert_indices,
                    route.expert_offsets,
                    stream=stream,
                )
                return output

            module_values = tuple(hip_modules)
        implementations[name] = Implementation(
            name,
            launch,
            _metadata(name, case, artifact=artifact, modules=module_values),
        )
    return implementations


def _prepare_grouped_pair_forward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    selection: RouteSelection,
    names: tuple[str, ...],
    artifact: Path | None,
    hip_root: Path | None,
) -> dict[str, Implementation]:
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, GroupedForwardPairProblem)
    assert isinstance(spec, GroupedForwardPairKernelSpec)
    assert prepared.input is not None
    workspace, stream = _forward_resources(stack, case, prepared)
    row_tasks = spec.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks
    hip_choice = None
    hip_row_tasks = False
    if "hip" in names:
        if hip_root is None:
            raise ValueError("HIP root is missing")
        hip_choice, hip_controls = _pair_forward_controls(case, selection, hip_root)
        hip_row_tasks = is_row_task_body(hip_choice.symbol)
    tasks_bank = []
    if row_tasks or hip_row_tasks:
        state = DerivedGroupedForwardPairState.from_problem_spec(problem, spec)
        if state.kernel_spec.row_task_rows is None:
            raise ValueError("row-task specification is missing its task row count")
        if hip_root is None and "hip" in names:
            raise ValueError("HIP root is missing")
        setup = stack.enter_context(InstalledGroupedForwardRowTaskSetup(hip_root))
        for route in selection.routes:
            tasks = GroupedForwardPairRowTaskWorkspace.allocate(
                prepared.input,
                aggregate_rows=case.rows,
                route_entries=route.expert_indices.numel(),
                row_tile=state.kernel_spec.row_task_rows,
            )
            setup.launch(
                route.expert_indices,
                route.expert_offsets,
                tasks,
                aggregate_rows=case.rows,
                stream=stream,
            )
            tasks_bank.append(tasks)

    ggtensile: Any = None
    if "ggtensile" in names:
        if artifact is None:
            raise ValueError("GGTensile artifact is missing")
        module_type = (
            GroupedForwardPairRowTaskModule if row_tasks else GroupedForwardPairModule
        )
        ggtensile = stack.enter_context(
            module_type(problem, spec, artifact, instance_name(case.instance))
        )

    hip_modules = []
    if "hip" in names:
        hip_modules.extend(stack.enter_context(control) for control in hip_controls)

    implementations = {}
    for name in names:
        outputs: tuple[torch.Tensor, torch.Tensor] = (
            torch.empty(
                case.rows, case.out_features, device="cuda", dtype=torch.bfloat16
            ),
            torch.empty(
                case.rows, case.out_features, device="cuda", dtype=torch.bfloat16
            ),
        )
        if name == "ggtensile":
            assert ggtensile is not None
            module: Any = ggtensile

            def launch(
                module=module, outputs=outputs
            ) -> tuple[torch.Tensor, torch.Tensor]:
                route = selection.current
                if row_tasks:
                    module.launch(
                        prepared.packed_weights[0],
                        prepared.packed_weights[1],
                        workspace,
                        outputs[0],
                        outputs[1],
                        tasks_bank[selection.index],
                        stream=stream,
                    )
                else:
                    module.launch(
                        prepared.packed_weights[0],
                        prepared.packed_weights[1],
                        workspace,
                        outputs[0],
                        outputs[1],
                        route.expert_indices,
                        route.expert_offsets,
                        stream=stream,
                    )
                return outputs

            module_values = (ggtensile,)
        else:

            def launch(outputs=outputs) -> tuple[torch.Tensor, torch.Tensor]:
                route = selection.current
                for module, weight, output in zip(
                    hip_modules, prepared.packed_weights, outputs, strict=True
                ):
                    if hip_row_tasks:
                        module.launch(
                            weight,
                            workspace,
                            output,
                            tasks_bank[selection.index],
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
                return outputs

            module_values = tuple(hip_modules)
        implementations[name] = Implementation(
            name,
            launch,
            _metadata(name, case, artifact=artifact, modules=module_values),
        )
    return implementations


def _prepare_fixed_forward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    names: tuple[str, ...],
    artifact: Path | None,
    hip_root: Path | None,
) -> dict[str, Implementation]:
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, FixedForwardProblem)
    assert isinstance(spec, FixedForwardKernelSpec)
    workspace, stream = _forward_resources(stack, case, prepared)
    implementations = {}
    for name in names:
        if name == "ggtensile":
            if artifact is None:
                raise ValueError("GGTensile artifact is missing")
            module = stack.enter_context(
                FixedGroupedQ8ForwardModule(
                    problem, spec, artifact, instance_name(case.instance)
                )
            )
        else:
            if hip_root is None:
                raise ValueError("HIP root is missing")
            module = stack.enter_context(
                InstalledFixedGroupedQ8ForwardModule(problem, spec, hip_root)
            )
        output = torch.empty(
            case.rows, 8, case.out_features, device="cuda", dtype=torch.bfloat16
        )

        def launch(module=module, output=output) -> torch.Tensor:
            module.launch(prepared.packed_weights[0], workspace, output, stream=stream)
            return output

        implementations[name] = Implementation(
            name, launch, _metadata(name, case, artifact=artifact, modules=(module,))
        )
    return implementations


def _prepare_fixed_backward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    names: tuple[str, ...],
    artifact: Path | None,
    hip_root: Path | None,
) -> dict[str, Implementation]:
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, FixedBackwardProblem)
    assert isinstance(spec, FixedBackwardKernelSpec)
    grad_output = prepared.grad_outputs[0]
    stream = torch.cuda.current_stream().cuda_stream
    implementations = {}
    for name in names:
        if name == "ggtensile":
            if artifact is None:
                raise ValueError("GGTensile artifact is missing")
            module = stack.enter_context(
                FixedGroupedQ8BackwardModule(
                    problem, spec, artifact, instance_name(case.instance)
                )
            )
        else:
            if hip_root is None:
                raise ValueError("HIP root is missing")
            module = stack.enter_context(
                InstalledFixedGroupedQ8BackwardModule(problem, spec, hip_root)
            )
        output = torch.empty(
            case.rows, 8, case.in_features, device="cuda", dtype=torch.bfloat16
        )

        def launch(module=module, output=output) -> torch.Tensor:
            module.launch(
                grad_output, prepared.packed_weights[0], output, stream=stream
            )
            return output

        implementations[name] = Implementation(
            name, launch, _metadata(name, case, artifact=artifact, modules=(module,))
        )
    return implementations


def prepare_direct_implementations(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    selection: RouteSelection,
    names: tuple[str, ...],
    *,
    ggtensile_root: Path | None,
    hip_root: Path | None,
) -> dict[str, Implementation]:
    direct_names = tuple(name for name in names if name != "baseline")
    if not direct_names:
        raise ValueError("at least one direct implementation is required")
    artifact = (
        public_artifact_path(case, ggtensile_root)
        if "ggtensile" in direct_names
        else None
    )
    if artifact is not None and not artifact.is_file():
        raise FileNotFoundError(f"deployed GGTensile artifact not found: {artifact}")
    if case.operation == "OrdinaryForward":
        return _prepare_ordinary_forward(
            stack, case, prepared, direct_names, artifact, hip_root
        )
    if case.operation == "OrdinaryBackward":
        return _prepare_ordinary_backward(
            stack, case, prepared, direct_names, artifact, hip_root
        )
    if case.operation == "GroupedForward":
        return _prepare_grouped_forward(
            stack, case, prepared, selection, direct_names, artifact, hip_root
        )
    if case.operation == "GroupedBackward":
        return _prepare_grouped_backward(
            stack, case, prepared, selection, direct_names, artifact, hip_root
        )
    if case.operation == "GroupedForwardPair":
        return _prepare_grouped_pair_forward(
            stack, case, prepared, selection, direct_names, artifact, hip_root
        )
    if case.operation == "GroupedBackwardPair":
        return _prepare_grouped_pair_backward(
            stack, case, prepared, selection, direct_names, artifact, hip_root
        )
    if case.operation == "FixedGroupedForward":
        return _prepare_fixed_forward(
            stack, case, prepared, direct_names, artifact, hip_root
        )
    if case.operation == "FixedGroupedBackward":
        return _prepare_fixed_backward(
            stack, case, prepared, direct_names, artifact, hip_root
        )
    raise ValueError(f"unsupported operation {case.operation}")


def _prepare_grouped_pair_backward(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    selection: RouteSelection,
    names: tuple[str, ...],
    artifact: Path | None,
    hip_root: Path | None,
) -> dict[str, Implementation]:
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    assert isinstance(problem, GroupedBackwardPairProblem)
    assert isinstance(spec, GroupedBackwardPairKernelSpec)
    stream = torch.cuda.current_stream().cuda_stream
    ggtensile: Any = None
    if "ggtensile" in names:
        if artifact is None:
            raise ValueError("GGTensile artifact is missing")
        ggtensile = stack.enter_context(
            GroupedBackwardPairModule(
                problem, spec, artifact, instance_name(case.instance)
            )
        )
    hip = None
    if "hip" in names:
        if hip_root is None:
            raise ValueError("HIP root is missing")
        hip = stack.enter_context(_pair_backward_control(case, selection, hip_root))
    implementations = {}
    for name in names:
        output = torch.empty(
            case.rows, case.in_features, device="cuda", dtype=torch.bfloat16
        )
        module: Any = ggtensile if name == "ggtensile" else hip
        if module is None:
            raise ValueError(f"no direct module prepared for {name}")

        def launch(module=module, output=output) -> torch.Tensor:
            route = selection.current
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

        implementations[name] = Implementation(
            name,
            launch,
            _metadata(name, case, artifact=artifact, modules=(module,)),
        )
    return implementations
