"""Launch all MMQ implementations for one exact deployment case."""

import contextlib
from pathlib import Path
from typing import Any

import torch

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
from tools.mmq_correctness import CorrectnessPrerequisite, PreparedCase
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


def _public(case: DeploymentCase, prepared: PreparedCase) -> Any:
    import torch_ggml_ops

    quant = int(prepared.tensor_types[0])
    if case.operation == "OrdinaryForward":
        assert prepared.input is not None
        return torch_ggml_ops.mmq(
            prepared.input, prepared.packed_weights[0], quant, case.out_features
        )
    if case.operation == "OrdinaryBackward":
        input_tensor = torch.zeros(
            case.rows,
            case.in_features,
            device=prepared.grad_outputs[0].device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        output = torch_ggml_ops.mmq(
            input_tensor, prepared.packed_weights[0], quant, case.out_features
        )
        return torch.autograd.grad(output, input_tensor, prepared.grad_outputs[0])[0]
    if case.operation == "GroupedForward":
        assert prepared.input is not None and prepared.route is not None
        return torch_ggml_ops.grouped_mmq(
            prepared.input,
            prepared.packed_weights[0],
            prepared.route.expert_indices,
            prepared.route.expert_offsets,
            quant,
            case.out_features,
        )
    if case.operation == "GroupedForwardPair":
        assert prepared.input is not None and prepared.route is not None
        return torch_ggml_ops.grouped_mmq_pair(
            prepared.input,
            prepared.packed_weights[0],
            prepared.packed_weights[1],
            prepared.route.expert_indices,
            prepared.route.expert_offsets,
            quant,
            case.out_features,
        )
    if case.operation == "GroupedBackward":
        assert prepared.route is not None
        input_tensor = torch.zeros(
            case.rows,
            case.in_features,
            device=prepared.grad_outputs[0].device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        output = torch_ggml_ops.grouped_mmq(
            input_tensor,
            prepared.packed_weights[0],
            prepared.route.expert_indices,
            prepared.route.expert_offsets,
            quant,
            case.out_features,
        )
        return torch.autograd.grad(output, input_tensor, prepared.grad_outputs[0])[0]
    if case.operation == "GroupedBackwardPair":
        assert prepared.route is not None
        input_tensor = torch.zeros(
            case.rows,
            case.in_features,
            device=prepared.grad_outputs[0].device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        first, second = torch_ggml_ops.grouped_mmq_pair(
            input_tensor,
            prepared.packed_weights[0],
            prepared.packed_weights[1],
            prepared.route.expert_indices,
            prepared.route.expert_offsets,
            quant,
            case.out_features,
        )
        return torch.autograd.grad(
            (first, second),
            input_tensor,
            (prepared.grad_outputs[0], prepared.grad_outputs[1]),
        )[0]
    if case.operation == "FixedGroupedForward":
        assert prepared.input is not None
        return torch_ggml_ops.fixed_grouped_mmq(
            prepared.input, prepared.packed_weights[0]
        )
    input_tensor = torch.zeros(
        case.rows,
        8,
        case.in_features,
        device=prepared.grad_outputs[0].device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output = torch_ggml_ops.fixed_grouped_mmq(input_tensor, prepared.packed_weights[0])
    return torch.autograd.grad(output, input_tensor, prepared.grad_outputs[0])[0]


def _route_entry_count(prepared: PreparedCase) -> int:
    if prepared.route is None:
        raise ValueError("route metadata is required")
    return len(prepared.route.expert_indices_cpu)


def _grouped_forward_control(
    case: DeploymentCase,
    route_entries: int,
    root: Path | None,
):
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
    return InstalledGroupedForwardModule(problem, spec, control, root)


def _pair_forward_control(
    case: DeploymentCase,
    route_entries: int,
    root: Path | None,
):
    """Build one projected control for the deployed paired-forward body."""

    choice = select_routed_control(
        "GroupedForwardPair",
        case.quant_type,
        case.out_features,
        case.in_features,
        case.rows,
        route_entries,
    )
    return build_forward_pair_control(choice, root), is_row_task_body(choice.symbol)


def _pair_backward_control(
    case: DeploymentCase,
    route_entries: int,
    root: Path | None,
):
    """Build the control for the deployed paired-backward body."""

    choice = select_routed_control(
        "GroupedBackwardPair",
        case.quant_type,
        case.out_features,
        case.in_features,
        case.rows,
        route_entries,
    )
    return build_backward_pair_control(choice, root)


def _direct_forward(case: DeploymentCase, prepared: PreparedCase) -> Any:
    artifact = public_artifact_path(case)
    if not artifact.is_file():
        raise CorrectnessPrerequisite(
            f"public GGTensile artifact is unavailable: {artifact}"
        )
    stream = torch.cuda.current_stream().cuda_stream
    with contextlib.ExitStack() as stack:
        quantizer = stack.enter_context(
            quantizer_module(case.operation, case.quant_type)()
        )
        if case.operation == "FixedGroupedForward":
            assert prepared.input is not None
            quantizer_input = prepared.input.reshape(
                case.rows * 8, case.in_features
            ).contiguous()
        else:
            assert prepared.input is not None
            quantizer_input = prepared.input
        workspace = quantizer.allocate(quantizer_input)
        quantizer.launch(quantizer_input, workspace, stream=stream)
        if case.operation == "OrdinaryForward":
            problem = case.instance.problem
            spec = case.instance.kernel_spec
            assert isinstance(problem, ProblemSize)
            assert isinstance(spec, ForwardKernelSpec)
            output = torch.empty(
                case.rows,
                case.out_features,
                device=quantizer_input.device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                ForwardModule(
                    problem,
                    case.quant_type,
                    spec,
                    artifact,
                    instance_name(case.instance),
                )
            )
            module.launch(prepared.packed_weights[0], workspace, output, stream=stream)
        elif case.operation == "GroupedForward":
            assert prepared.route is not None
            problem = case.instance.problem
            spec = case.instance.kernel_spec
            assert isinstance(problem, GroupedForwardProblem)
            assert isinstance(spec, GroupedForwardKernelSpec)
            output = torch.empty(
                case.rows,
                case.out_features,
                device=quantizer_input.device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                GroupedForwardModule(
                    problem, spec, artifact, instance_name(case.instance)
                )
            )
            module.launch(
                prepared.packed_weights[0],
                workspace,
                output,
                prepared.route.expert_indices,
                prepared.route.expert_offsets,
                stream=stream,
            )
        elif case.operation == "GroupedForwardPair":
            assert prepared.route is not None
            problem = case.instance.problem
            spec = case.instance.kernel_spec
            assert isinstance(problem, GroupedForwardPairProblem)
            assert isinstance(spec, GroupedForwardPairKernelSpec)
            first = torch.empty(
                case.rows,
                case.out_features,
                device=quantizer_input.device,
                dtype=torch.bfloat16,
            )
            second = torch.empty_like(first)
            row_tasks = spec.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks
            if row_tasks:
                state = DerivedGroupedForwardPairState.from_problem_spec(problem, spec)
                row_task_rows = state.kernel_spec.row_task_rows
                assert row_task_rows is not None
                tasks = GroupedForwardPairRowTaskWorkspace.allocate(
                    quantizer_input,
                    aggregate_rows=case.rows,
                    route_entries=_route_entry_count(prepared),
                    row_tile=row_task_rows,
                )
                setup = stack.enter_context(InstalledGroupedForwardRowTaskSetup())
                setup.launch(
                    prepared.route.expert_indices,
                    prepared.route.expert_offsets,
                    tasks,
                    aggregate_rows=case.rows,
                    stream=stream,
                )
                module = stack.enter_context(
                    GroupedForwardPairRowTaskModule(
                        problem, spec, artifact, instance_name(case.instance)
                    )
                )
                module.launch(
                    prepared.packed_weights[0],
                    prepared.packed_weights[1],
                    workspace,
                    first,
                    second,
                    tasks,
                    stream=stream,
                )
            else:
                module = stack.enter_context(
                    GroupedForwardPairModule(
                        problem, spec, artifact, instance_name(case.instance)
                    )
                )
                module.launch(
                    prepared.packed_weights[0],
                    prepared.packed_weights[1],
                    workspace,
                    first,
                    second,
                    prepared.route.expert_indices,
                    prepared.route.expert_offsets,
                    stream=stream,
                )
            output = (first, second)
        else:
            assert prepared.input is not None
            problem = case.instance.problem
            spec = case.instance.kernel_spec
            assert isinstance(problem, FixedForwardProblem)
            assert isinstance(spec, FixedForwardKernelSpec)
            output = torch.empty(
                case.rows,
                8,
                case.out_features,
                device=quantizer_input.device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                FixedGroupedQ8ForwardModule(
                    problem, spec, artifact, instance_name(case.instance)
                )
            )
            module.launch(prepared.packed_weights[0], workspace, output, stream=stream)
        torch.cuda.synchronize()
        if isinstance(output, tuple):
            return tuple(value.clone() for value in output)
        return output.clone()


def _direct_backward(case: DeploymentCase, prepared: PreparedCase) -> torch.Tensor:
    artifact = public_artifact_path(case)
    if not artifact.is_file():
        raise CorrectnessPrerequisite(
            f"public GGTensile artifact is unavailable: {artifact}"
        )
    stream = torch.cuda.current_stream().cuda_stream
    with contextlib.ExitStack() as stack:
        if case.operation == "OrdinaryBackward":
            problem = case.instance.problem
            spec = case.instance.kernel_spec
            assert isinstance(problem, ProblemSize)
            assert isinstance(spec, BackwardKernelSpec)
            output = torch.empty(
                case.rows,
                case.in_features,
                device=prepared.grad_outputs[0].device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                BackwardModule(
                    problem,
                    case.quant_type,
                    spec,
                    artifact,
                    instance_name(case.instance),
                )
            )
            module.launch(
                prepared.grad_outputs[0],
                prepared.packed_weights[0],
                output,
                stream=stream,
            )
        elif case.operation == "GroupedBackward":
            assert prepared.route is not None
            problem = case.instance.problem
            spec = case.instance.kernel_spec
            assert isinstance(problem, ProblemSize)
            assert isinstance(spec, GroupedBackwardKernelSpec)
            output = torch.empty(
                case.rows,
                case.in_features,
                device=prepared.grad_outputs[0].device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                GroupedBackwardModule(
                    problem,
                    case.quant_type,
                    spec,
                    artifact,
                    instance_name(case.instance),
                )
            )
            module.launch(
                prepared.grad_outputs[0],
                prepared.packed_weights[0],
                output,
                prepared.route.expert_indices,
                prepared.route.expert_offsets,
                stream=stream,
            )
        elif case.operation == "GroupedBackwardPair":
            assert prepared.route is not None
            problem = case.instance.problem
            spec = case.instance.kernel_spec
            assert isinstance(problem, GroupedBackwardPairProblem)
            assert isinstance(spec, GroupedBackwardPairKernelSpec)
            output = torch.empty(
                case.rows,
                case.in_features,
                device=prepared.grad_outputs[0].device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                GroupedBackwardPairModule(
                    problem, spec, artifact, instance_name(case.instance)
                )
            )
            module.launch(
                prepared.grad_outputs[0],
                prepared.grad_outputs[1],
                prepared.packed_weights[0],
                prepared.packed_weights[1],
                output,
                prepared.route.expert_indices,
                prepared.route.expert_offsets,
                stream=stream,
            )
        else:
            problem = case.instance.problem
            spec = case.instance.kernel_spec
            assert isinstance(problem, FixedBackwardProblem)
            assert isinstance(spec, FixedBackwardKernelSpec)
            output = torch.empty(
                case.rows,
                8,
                case.in_features,
                device=prepared.grad_outputs[0].device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                FixedGroupedQ8BackwardModule(
                    problem, spec, artifact, instance_name(case.instance)
                )
            )
            module.launch(
                prepared.grad_outputs[0],
                prepared.packed_weights[0],
                output,
                stream=stream,
            )
        torch.cuda.synchronize()
        return output.clone()


def _hip_forward(
    case: DeploymentCase, prepared: PreparedCase, root: Path | None
) -> Any:
    stream = torch.cuda.current_stream().cuda_stream
    with contextlib.ExitStack() as stack:
        quantizer = stack.enter_context(
            quantizer_module(case.operation, case.quant_type)()
        )
        assert prepared.input is not None
        quantizer_input = (
            prepared.input.reshape(case.rows * 8, case.in_features).contiguous()
            if case.operation == "FixedGroupedForward"
            else prepared.input
        )
        workspace = quantizer.allocate(quantizer_input)
        quantizer.launch(quantizer_input, workspace, stream=stream)
        if case.operation == "OrdinaryForward":
            problem = case.instance.problem
            spec = case.instance.kernel_spec
            assert isinstance(problem, ProblemSize)
            assert isinstance(spec, ForwardKernelSpec)
            output = torch.empty(
                case.rows,
                case.out_features,
                device=quantizer_input.device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                FixedHipForwardModule(problem, case.quant_type, spec, root)
            )
            module.launch(prepared.packed_weights[0], workspace, output, stream=stream)
        elif case.operation == "GroupedForward":
            assert prepared.route is not None
            output = torch.empty(
                case.rows,
                case.out_features,
                device=quantizer_input.device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                _grouped_forward_control(case, _route_entry_count(prepared), root)
            )
            module.launch(
                prepared.packed_weights[0],
                workspace,
                output,
                prepared.route.expert_indices,
                prepared.route.expert_offsets,
                stream=stream,
            )
        elif case.operation == "GroupedForwardPair":
            assert prepared.route is not None
            problem = case.instance.problem
            spec = case.instance.kernel_spec
            assert isinstance(problem, GroupedForwardPairProblem)
            assert isinstance(spec, GroupedForwardPairKernelSpec)
            first = torch.empty(
                case.rows,
                case.out_features,
                device=quantizer_input.device,
                dtype=torch.bfloat16,
            )
            second = torch.empty_like(first)
            entries = _route_entry_count(prepared)
            control0, hip_row_tasks = _pair_forward_control(case, entries, root)
            control1, _ = _pair_forward_control(case, entries, root)
            module0 = stack.enter_context(control0)
            module1 = stack.enter_context(control1)
            if hip_row_tasks:
                state = DerivedGroupedForwardPairState.from_problem_spec(problem, spec)
                row_task_rows = state.kernel_spec.row_task_rows
                assert row_task_rows is not None
                tasks = GroupedForwardPairRowTaskWorkspace.allocate(
                    quantizer_input,
                    aggregate_rows=case.rows,
                    route_entries=_route_entry_count(prepared),
                    row_tile=row_task_rows,
                )
                setup = stack.enter_context(InstalledGroupedForwardRowTaskSetup())
                setup.launch(
                    prepared.route.expert_indices,
                    prepared.route.expert_offsets,
                    tasks,
                    aggregate_rows=case.rows,
                    stream=stream,
                )
                for module, weight, output in (
                    (module0, prepared.packed_weights[0], first),
                    (module1, prepared.packed_weights[1], second),
                ):
                    module.launch(
                        weight,
                        workspace,
                        output,
                        tasks,
                        aggregate_rows=case.rows,
                        stream=stream,
                    )
            else:
                for module, weight, output in (
                    (module0, prepared.packed_weights[0], first),
                    (module1, prepared.packed_weights[1], second),
                ):
                    module.launch(
                        weight,
                        workspace,
                        output,
                        prepared.route.expert_indices,
                        prepared.route.expert_offsets,
                        aggregate_rows=case.rows,
                        stream=stream,
                    )
            output = (first, second)
        else:
            problem = case.instance.problem
            spec = case.instance.kernel_spec
            assert isinstance(problem, FixedForwardProblem)
            assert isinstance(spec, FixedForwardKernelSpec)
            output = torch.empty(
                case.rows,
                8,
                case.out_features,
                device=quantizer_input.device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                InstalledFixedGroupedQ8ForwardModule(problem, spec, root)
            )
            module.launch(prepared.packed_weights[0], workspace, output, stream=stream)
        torch.cuda.synchronize()
        return (
            tuple(value.clone() for value in output)
            if isinstance(output, tuple)
            else output.clone()
        )


def _hip_backward(
    case: DeploymentCase, prepared: PreparedCase, root: Path | None
) -> torch.Tensor:
    stream = torch.cuda.current_stream().cuda_stream
    with contextlib.ExitStack() as stack:
        if case.operation == "OrdinaryBackward":
            problem = case.instance.problem
            spec = case.instance.kernel_spec
            assert isinstance(problem, ProblemSize)
            assert isinstance(spec, BackwardKernelSpec)
            output = torch.empty(
                case.rows,
                case.in_features,
                device=prepared.grad_outputs[0].device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                InstalledDenseBackwardModule(problem, case.quant_type, spec, root)
            )
            module.launch(
                prepared.grad_outputs[0],
                prepared.packed_weights[0],
                output,
                stream=stream,
            )
        elif case.operation == "GroupedBackward":
            assert prepared.route is not None
            output = torch.empty(
                case.rows,
                case.in_features,
                device=prepared.grad_outputs[0].device,
                dtype=torch.bfloat16,
            )
            choice = select_routed_control(
                "GroupedBackward",
                case.quant_type,
                case.out_features,
                case.in_features,
                case.rows,
                _route_entry_count(prepared),
            )
            module = stack.enter_context(build_backward_control(choice, root))
            if is_row_task_body(choice.symbol):
                tasks = RowTaskWorkspace.allocate(
                    prepared.grad_outputs[0],
                    aggregate_rows=case.rows,
                    route_entries=_route_entry_count(prepared),
                    row_tile=module.ROW_TASK_ROWS,
                )
                setup = stack.enter_context(InstalledGroupedRowTaskSetup())
                setup.launch(
                    prepared.route.expert_indices,
                    prepared.route.expert_offsets,
                    tasks,
                    aggregate_rows=case.rows,
                    stream=stream,
                )
                module.launch(
                    prepared.grad_outputs[0],
                    prepared.packed_weights[0],
                    output,
                    tasks,
                    stream=stream,
                )
                return output
            module.launch(
                prepared.grad_outputs[0],
                prepared.packed_weights[0],
                output,
                prepared.route.expert_indices,
                prepared.route.expert_offsets,
                stream=stream,
            )
        elif case.operation == "GroupedBackwardPair":
            assert prepared.route is not None
            output = torch.empty(
                case.rows,
                case.in_features,
                device=prepared.grad_outputs[0].device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                _pair_backward_control(case, _route_entry_count(prepared), root)
            )
            module.launch(
                prepared.grad_outputs[0],
                prepared.grad_outputs[1],
                prepared.packed_weights[0],
                prepared.packed_weights[1],
                output,
                prepared.route.expert_indices,
                prepared.route.expert_offsets,
                stream=stream,
            )
        else:
            problem = case.instance.problem
            spec = case.instance.kernel_spec
            assert isinstance(problem, FixedBackwardProblem)
            assert isinstance(spec, FixedBackwardKernelSpec)
            output = torch.empty(
                case.rows,
                8,
                case.in_features,
                device=prepared.grad_outputs[0].device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                InstalledFixedGroupedQ8BackwardModule(problem, spec, root)
            )
            module.launch(
                prepared.grad_outputs[0],
                prepared.packed_weights[0],
                output,
                stream=stream,
            )
        torch.cuda.synchronize()
        return output.clone()


def run_implementation(
    case: DeploymentCase,
    prepared: PreparedCase,
    implementation: str,
    *,
    hip_root: Path | None = None,
) -> Any:
    """Run one implementation and return detached BF16 output(s).

    `public` is the registered PyTorch operator, `ggtensile` is the exact
    selected HSACO, and `hip` is the historical standalone control. The
    latter two are deliberately launched through their native ABIs rather than
    through another project implementation.
    """
    if implementation not in {"public", "ggtensile", "hip"}:
        raise ValueError(f"unknown MMQ implementation: {implementation}")
    if implementation == "public":
        result = _public(case, prepared)
    elif case.operation.endswith("Forward") or case.operation == "GroupedForwardPair":
        result = (
            _direct_forward(case, prepared)
            if implementation == "ggtensile"
            else _hip_forward(case, prepared, hip_root)
        )
    else:
        result = (
            _direct_backward(case, prepared)
            if implementation == "ggtensile"
            else _hip_backward(case, prepared, hip_root)
        )
    if isinstance(result, tuple):
        return tuple(value.detach().contiguous() for value in result)
    return result.detach().contiguous()
