"""Prepare direct HIP and GGTensile kernel implementations."""

import contextlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from bench.benchmark_common import Implementation
from bench.benchmark_data import RouteSelection
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
)
from tools.mmq_correctness import PreparedCase
from tools.mmq_deployment_cases import DeploymentCase, public_artifact_path


def _quantizer(case: DeploymentCase):
    if case.quant_type == "Q2_K":
        return FixedQ81F16D2S6QuantizerModule
    if case.quant_type in {"Q4_K", "Q5_K"} and case.operation in {
        "OrdinaryForward",
        "GroupedForward",
    }:
        return FixedQ81F16D4S4QuantizerModule
    return FixedQ81F32D4QuantizerModule


def _grouped_forward_control_type(case: DeploymentCase, route_entries: int):
    problem = case.instance.problem
    assert isinstance(problem, GroupedForwardProblem)
    if case.quant_type == "Q4_K":
        from tools.ggtensile.runtime import InstalledGroupedForwardModule

        return InstalledGroupedForwardModule
    if case.quant_type == "Q5_K":
        from tools.ggtensile.runtime import (
            InstalledGroupedForwardQ5J32Module,
            InstalledGroupedForwardQ5Module,
        )

        return (
            InstalledGroupedForwardQ5J32Module
            if case.rows < 128 * route_entries
            else InstalledGroupedForwardQ5Module
        )
    if case.quant_type == "Q2_K":
        from tools.ggtensile.runtime import (
            InstalledGroupedForwardQ2J32J16Module,
            InstalledGroupedForwardQ2J32Module,
        )

        return (
            InstalledGroupedForwardQ2J32J16Module
            if case.rows == 49_152 or case.rows < 64 * route_entries
            else InstalledGroupedForwardQ2J32Module
        )
    if case.quant_type == "IQ2_S":
        from tools.ggtensile.runtime import (
            InstalledGroupedForwardIQ2SJ64J32Module,
            InstalledGroupedForwardIQ2SJ64Module,
        )

        return (
            InstalledGroupedForwardIQ2SJ64J32Module
            if case.rows == 65_536 or case.rows < 128 * route_entries
            else InstalledGroupedForwardIQ2SJ64Module
        )
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
    quantizer = stack.enter_context(_quantizer(case)())
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
            control_type = _grouped_forward_control_type(case, entries)
            if control_type not in hip_by_entries:
                hip_by_entries[entries] = stack.enter_context(
                    control_type(problem, spec, hip_root)
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
    hip_by_route_entries = {}
    if "hip" in names:
        if hip_root is None:
            raise ValueError("HIP root is missing")
        for route in selection.routes:
            entries = route.expert_indices.numel()
            control_spec = InstalledGroupedBackwardControl.select_spec(
                case.quant_type, case.rows, entries
            )
            key = (
                control_spec.symbol,
                control_spec.out_features,
                control_spec.in_features,
                control_spec.packed_row_bytes,
            )
            if key not in hip_by_entries:
                hip_by_entries[key] = stack.enter_context(
                    InstalledGroupedBackwardControl(
                        case.quant_type, case.rows, entries, hip_root
                    )
                )
            hip_by_route_entries[entries] = hip_by_entries[key]
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
                hip_by_route_entries[route.expert_indices.numel()].launch(
                    prepared.grad_outputs[0],
                    prepared.packed_weights[0],
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
    tasks_bank = []
    if row_tasks:
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
        if hip_root is None:
            raise ValueError("HIP root is missing")
        for _ in range(2):
            hip_modules.append(
                stack.enter_context(
                    _pair_forward_control(problem, spec, row_tasks, hip_root)
                )
            )

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
                    if row_tasks:
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
        hip = stack.enter_context(_pair_backward_control(problem, spec, hip_root))
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
