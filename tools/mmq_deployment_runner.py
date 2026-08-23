"""Launch all MMQ implementations for one exact deployment case."""

import contextlib
from pathlib import Path
from typing import Any, cast

import torch

from tools.ggtensile.fixed_grouped_mmq_bwd_model import FixedBackwardSolutionKey
from tools.ggtensile.fixed_grouped_mmq_fwd_model import FixedForwardSolutionKey
from tools.ggtensile.grouped_mmq_bwd_pair_model import GroupedBackwardPairSolutionKey
from tools.ggtensile.grouped_mmq_bwd_pair_runtime import (
    GroupedBackwardPairModule,
    InstalledGroupedBackwardPairIQ2SControl,
    InstalledGroupedBackwardPairIQ2XXSControl,
    InstalledGroupedBackwardPairQ3KControl,
)
from tools.ggtensile.grouped_mmq_bwd_runtime import InstalledGroupedBackwardControl
from tools.ggtensile.grouped_mmq_fwd_model import GroupedForwardSolutionKey
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
from tools.ggtensile.model import SolutionKey
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
    HIPRuntimeError,
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
from tools.mmq_correctness import CorrectnessPrerequisite, PreparedCase
from tools.mmq_deployment_cases import DeploymentCase, public_artifact_path


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


def _quantizer(case: DeploymentCase):
    if case.quant_type == "Q2_K":
        return FixedQ81F16D2S6QuantizerModule
    if case.quant_type in {"Q4_K", "Q5_K"} and case.operation in {
        "OrdinaryForward",
        "GroupedForward",
    }:
        return FixedQ81F16D4S4QuantizerModule
    return FixedQ81F32D4QuantizerModule


def _route_entry_count(prepared: PreparedCase) -> int:
    if prepared.route is None:
        raise ValueError("route metadata is required")
    return len(prepared.route.expert_indices_cpu)


def _grouped_forward_control(
    case: DeploymentCase,
    route_entries: int,
    root: Path | None,
):
    key = cast(GroupedForwardSolutionKey, case.key)
    if case.quant_type == "Q4_K":
        return InstalledGroupedForwardModule(key, root)
    if case.quant_type == "Q5_K":
        if case.rows < 128 * route_entries:
            return InstalledGroupedForwardQ5J32Module(key, root)
        return InstalledGroupedForwardQ5Module(key, root)
    if case.quant_type == "Q2_K":
        if case.rows == 49152 or case.rows < 64 * route_entries:
            return InstalledGroupedForwardQ2J32J16Module(key, root)
        return InstalledGroupedForwardQ2J32Module(key, root)
    if case.quant_type == "IQ2_S":
        if case.rows == 65536 or case.rows < 128 * route_entries:
            return InstalledGroupedForwardIQ2SJ64J32Module(key, root)
        return InstalledGroupedForwardIQ2SJ64Module(key, root)
    raise HIPRuntimeError(f"no grouped-forward HIP control for {case.quant_type}")


def _pair_forward_control(case: DeploymentCase, row_tasks: bool, root: Path | None):
    if case.quant_type == "IQ2_S":
        cls = (
            InstalledGroupedForwardPairRowTaskControl
            if row_tasks
            else InstalledGroupedForwardPairSerialControl
        )
        return cls(root)
    if case.quant_type == "Q3_K":
        cls = (
            InstalledGroupedForwardPairQ3RowTaskControl
            if row_tasks
            else InstalledGroupedForwardPairQ3SerialControl
        )
        return cls(root)
    if case.quant_type == "IQ2_XXS" and not row_tasks:
        return InstalledGroupedForwardPairIQ2XXSSerialControl(64, root)
    raise HIPRuntimeError("no paired-forward HIP control matches the route ownership")


def _pair_backward_control(case: DeploymentCase, root: Path | None):
    key = cast(GroupedBackwardPairSolutionKey, case.key)
    macro_tile = key.solution.compute.macro_tile0
    cls = {
        "Q3_K": InstalledGroupedBackwardPairQ3KControl,
        "IQ2_S": InstalledGroupedBackwardPairIQ2SControl,
        "IQ2_XXS": InstalledGroupedBackwardPairIQ2XXSControl,
    }.get(case.quant_type)
    if cls is None:
        raise HIPRuntimeError(f"no paired-backward HIP control for {case.quant_type}")
    return cls(macro_tile, root)


def _direct_forward(case: DeploymentCase, prepared: PreparedCase) -> Any:
    artifact = public_artifact_path(case)
    if not artifact.is_file():
        raise CorrectnessPrerequisite(
            f"public GGTensile artifact is unavailable: {artifact}"
        )
    stream = torch.cuda.current_stream().cuda_stream
    with contextlib.ExitStack() as stack:
        quantizer = stack.enter_context(_quantizer(case)())
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
            output = torch.empty(
                case.rows,
                case.out_features,
                device=quantizer_input.device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                ForwardModule(cast(SolutionKey, case.key), artifact)
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
                GroupedForwardModule(
                    cast(GroupedForwardSolutionKey, case.key), artifact
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
            first = torch.empty(
                case.rows,
                case.out_features,
                device=quantizer_input.device,
                dtype=torch.bfloat16,
            )
            second = torch.empty_like(first)
            key = cast(GroupedForwardPairSolutionKey, case.key)
            row_tasks = (
                key.solution.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks
            )
            if row_tasks:
                state = __import__(
                    "tools.ggtensile.grouped_mmq_fwd_pair_spec",
                    fromlist=["DerivedGroupedForwardPairState"],
                ).DerivedGroupedForwardPairState.from_solution_key(key)
                tasks = GroupedForwardPairRowTaskWorkspace.allocate(
                    quantizer_input,
                    aggregate_rows=case.rows,
                    route_entries=_route_entry_count(prepared),
                    row_tile=state.kernel_spec.row_task_rows,
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
                    GroupedForwardPairRowTaskModule(key, artifact)
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
                module = stack.enter_context(GroupedForwardPairModule(key, artifact))
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
            output = torch.empty(
                case.rows,
                8,
                case.out_features,
                device=quantizer_input.device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                FixedGroupedQ8ForwardModule(
                    cast(FixedForwardSolutionKey, case.key), artifact
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
            output = torch.empty(
                case.rows,
                case.in_features,
                device=prepared.grad_outputs[0].device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                BackwardModule(cast(SolutionKey, case.key), artifact)
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
            module = stack.enter_context(
                GroupedBackwardModule(cast(SolutionKey, case.key), artifact)
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
            output = torch.empty(
                case.rows,
                case.in_features,
                device=prepared.grad_outputs[0].device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                GroupedBackwardPairModule(
                    cast(GroupedBackwardPairSolutionKey, case.key), artifact
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
            output = torch.empty(
                case.rows,
                8,
                case.in_features,
                device=prepared.grad_outputs[0].device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                FixedGroupedQ8BackwardModule(
                    cast(FixedBackwardSolutionKey, case.key), artifact
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
        quantizer = stack.enter_context(_quantizer(case)())
        assert prepared.input is not None
        quantizer_input = (
            prepared.input.reshape(case.rows * 8, case.in_features).contiguous()
            if case.operation == "FixedGroupedForward"
            else prepared.input
        )
        workspace = quantizer.allocate(quantizer_input)
        quantizer.launch(quantizer_input, workspace, stream=stream)
        if case.operation == "OrdinaryForward":
            output = torch.empty(
                case.rows,
                case.out_features,
                device=quantizer_input.device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                FixedHipForwardModule(cast(SolutionKey, case.key), root)
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
            first = torch.empty(
                case.rows,
                case.out_features,
                device=quantizer_input.device,
                dtype=torch.bfloat16,
            )
            second = torch.empty_like(first)
            key = cast(GroupedForwardPairSolutionKey, case.key)
            row_tasks = (
                key.solution.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks
            )
            module0 = stack.enter_context(_pair_forward_control(case, row_tasks, root))
            module1 = stack.enter_context(_pair_forward_control(case, row_tasks, root))
            if row_tasks:
                state = __import__(
                    "tools.ggtensile.grouped_mmq_fwd_pair_spec",
                    fromlist=["DerivedGroupedForwardPairState"],
                ).DerivedGroupedForwardPairState.from_solution_key(key)
                tasks = GroupedForwardPairRowTaskWorkspace.allocate(
                    quantizer_input,
                    aggregate_rows=case.rows,
                    route_entries=_route_entry_count(prepared),
                    row_tile=state.kernel_spec.row_task_rows,
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
            output = torch.empty(
                case.rows,
                8,
                case.out_features,
                device=quantizer_input.device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                InstalledFixedGroupedQ8ForwardModule(
                    cast(FixedForwardSolutionKey, case.key), root
                )
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
            output = torch.empty(
                case.rows,
                case.in_features,
                device=prepared.grad_outputs[0].device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                __import__(
                    "tools.ggtensile.dense_mmq_bwd_runtime",
                    fromlist=["InstalledDenseBackwardModule"],
                ).InstalledDenseBackwardModule(cast(SolutionKey, case.key), root)
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
            module = stack.enter_context(
                InstalledGroupedBackwardControl(
                    case.quant_type, case.rows, _route_entry_count(prepared), root
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
            output = torch.empty(
                case.rows,
                case.in_features,
                device=prepared.grad_outputs[0].device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(_pair_backward_control(case, root))
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
            output = torch.empty(
                case.rows,
                8,
                case.in_features,
                device=prepared.grad_outputs[0].device,
                dtype=torch.bfloat16,
            )
            module = stack.enter_context(
                InstalledFixedGroupedQ8BackwardModule(
                    cast(FixedBackwardSolutionKey, case.key), root
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


def run_implementation(
    case: DeploymentCase,
    prepared: PreparedCase,
    implementation: str,
    *,
    hip_root: Path | None = None,
) -> Any:
    """Run one implementation and return detached BF16 output(s).

    ``public`` is the registered PyTorch operator, ``ggtensile`` is the exact
    selected HSACO, and ``hip`` is the historical standalone control.  The
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


def implementation_available(
    case: DeploymentCase, implementation: str, *, hip_root: Path | None = None
) -> tuple[bool, str | None]:
    if implementation == "ggtensile":
        artifact = public_artifact_path(case)
        return (
            artifact.is_file(),
            None if artifact.is_file() else f"missing GGTensile artifact: {artifact}",
        )
    if implementation == "hip":
        if hip_root is None:
            return False, "historical HIP-control root is unavailable"
        return True, None
    return True, None
