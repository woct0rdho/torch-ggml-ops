"""Launch all MMQ implementations for one exact deployment case."""

import contextlib
from pathlib import Path
from typing import Any

import torch

from tools.ggtensile.family_registry import instance_name
from tools.mmq_correctness import CorrectnessPrerequisite, PreparedCase
from tools.mmq_deployment_cases import DeploymentCase, public_artifact_path
from tools.mmq_launch import (
    FORWARD_OPERATIONS,
    activation_workspace,
    build_launcher,
    output_tensors,
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


def _deployed(
    case: DeploymentCase,
    prepared: PreparedCase,
    implementation: str,
    root: Path | None,
) -> Any:
    """Run one deployed HSACO implementation through the shared launch plan."""

    artifact = None
    if implementation == "ggtensile":
        artifact = public_artifact_path(case)
        if not artifact.is_file():
            raise CorrectnessPrerequisite(
                f"public GGTensile artifact is unavailable: {artifact}"
            )
    with contextlib.ExitStack() as stack:
        workspace = (
            activation_workspace(case, prepared, stack)
            if case.operation in FORWARD_OPERATIONS
            else None
        )
        route = prepared.route
        launcher = build_launcher(
            case,
            prepared,
            implementation,
            stack=stack,
            root=root,
            artifact=artifact,
            instance=instance_name(case.instance),
            route_entries=(
                int(route.expert_indices.numel()) if route is not None else None
            ),
            workspace=workspace,
        )
        outputs = output_tensors(case, launcher.device)
        launch = launcher.prepare_route(route if launcher.needs_route else None, stack)
        launch.call(outputs)
        torch.cuda.synchronize()
    if len(outputs) == 1:
        return outputs[0].clone()
    return tuple(value.clone() for value in outputs)


def run_implementation(
    case: DeploymentCase,
    prepared: PreparedCase,
    implementation: str,
    *,
    hip_root: Path | None = None,
) -> Any:
    """Run one implementation and return detached BF16 output(s).

    `public` is the registered PyTorch operator, `ggtensile` is the exact
    selected HSACO, and `hip` is the standalone control. The
    latter two are deliberately launched through their native ABIs rather than
    through another project implementation.
    """
    if implementation not in {"public", "ggtensile", "hip"}:
        raise ValueError(f"unknown MMQ implementation: {implementation}")
    result = (
        _public(case, prepared)
        if implementation == "public"
        else _deployed(case, prepared, implementation, hip_root)
    )
    if isinstance(result, tuple):
        return tuple(value.detach().contiguous() for value in result)
    return result.detach().contiguous()
