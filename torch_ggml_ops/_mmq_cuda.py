import torch

from .runtime_contract import (
    paired_row_task_capacity,
    paired_row_task_rows,
    quant_workspace_elements,
)


def mmq_inplace(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_type: int,
    out_features: int,
    output: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    torch.ops.torch_ggml_ops._mmq_launch.default(
        input, packed_weight, quant_type, out_features, output, workspace
    )


def mmq_cuda(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_type: int,
    out_features: int,
) -> torch.Tensor:
    output = torch.empty(
        (*input.shape[:-1], out_features), dtype=input.dtype, device=input.device
    )
    workspace = torch.empty(
        quant_workspace_elements(input.numel()),
        dtype=torch.uint8,
        device=input.device,
    )
    mmq_inplace(input, packed_weight, quant_type, out_features, output, workspace)
    return output


def mmq_grad_input_inplace(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_type: int,
    in_features: int,
    grad_input: torch.Tensor,
) -> None:
    torch.ops.torch_ggml_ops._mmq_grad_input_launch.default(
        grad_output, packed_weight, quant_type, in_features, grad_input
    )


def mmq_grad_input(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_type: int,
    in_features: int,
) -> torch.Tensor:
    grad_input = torch.empty(
        (*grad_output.shape[:-1], in_features),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    mmq_grad_input_inplace(
        grad_output, packed_weight, quant_type, in_features, grad_input
    )
    return grad_input


def fixed_grouped_mmq_cuda(
    input: torch.Tensor, packed_weight: torch.Tensor
) -> torch.Tensor:
    output = torch.empty(
        (*input.shape[:-1], packed_weight.shape[1]),
        dtype=input.dtype,
        device=input.device,
    )
    workspace = torch.empty(
        quant_workspace_elements(input.numel()),
        dtype=torch.uint8,
        device=input.device,
    )
    torch.ops.torch_ggml_ops._fixed_grouped_mmq_launch.default(
        input, packed_weight, output, workspace
    )
    return output


def fixed_grouped_mmq_grad_input(
    grad_output: torch.Tensor, packed_weight: torch.Tensor
) -> torch.Tensor:
    grad_input = torch.empty(
        (*grad_output.shape[:-1], packed_weight.shape[2] // 34 * 32),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    torch.ops.torch_ggml_ops._fixed_grouped_mmq_grad_input_launch.default(
        grad_output, packed_weight, grad_input
    )
    return grad_input


def grouped_mmq_cuda(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    quant_type: int,
    out_features: int,
) -> torch.Tensor:
    output = torch.empty(
        (input.shape[0], out_features), dtype=input.dtype, device=input.device
    )
    workspace = torch.empty(
        quant_workspace_elements(input.numel()),
        dtype=torch.uint8,
        device=input.device,
    )
    torch.ops.torch_ggml_ops._grouped_mmq_launch.default(
        input,
        packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        out_features,
        output,
        workspace,
    )
    return output


def grouped_mmq_grad_input(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    quant_type: int,
    in_features: int,
) -> torch.Tensor:
    grad_input = torch.empty(
        (grad_output.shape[0], in_features),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    torch.ops.torch_ggml_ops._grouped_mmq_grad_input_launch.default(
        grad_output,
        packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        in_features,
        grad_input,
    )
    return grad_input


def grouped_mmq_pair_cuda(
    input: torch.Tensor,
    first_packed_weight: torch.Tensor,
    second_packed_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    quant_type: int,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_output = torch.empty(
        (input.shape[0], out_features), dtype=input.dtype, device=input.device
    )
    second_output = torch.empty_like(first_output)
    workspace = torch.empty(
        quant_workspace_elements(input.numel()),
        dtype=torch.uint8,
        device=input.device,
    )
    row_task_rows = paired_row_task_rows(quant_type)
    task_capacity = (
        paired_row_task_capacity(input.shape[0], expert_indices.numel(), row_task_rows)
        if row_task_rows
        else 0
    )
    task_count = torch.empty(
        1 if row_task_rows else 0, dtype=torch.int32, device=input.device
    )
    task_experts = torch.empty(task_capacity, dtype=torch.int32, device=input.device)
    task_row_starts = torch.empty(task_capacity, dtype=torch.int32, device=input.device)
    task_row_ends = torch.empty(task_capacity, dtype=torch.int32, device=input.device)
    torch.ops.torch_ggml_ops._grouped_mmq_pair_launch.default(
        input,
        first_packed_weight,
        second_packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        out_features,
        first_output,
        second_output,
        workspace,
        task_count,
        task_experts,
        task_row_starts,
        task_row_ends,
    )
    return first_output, second_output


def grouped_mmq_pair_grad_input(
    first_grad_output: torch.Tensor,
    second_grad_output: torch.Tensor,
    first_packed_weight: torch.Tensor,
    second_packed_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    quant_type: int,
    in_features: int,
) -> torch.Tensor:
    grad_input = torch.empty(
        (first_grad_output.shape[0], in_features),
        dtype=first_grad_output.dtype,
        device=first_grad_output.device,
    )
    torch.ops.torch_ggml_ops._grouped_mmq_pair_grad_input_launch.default(
        first_grad_output,
        second_grad_output,
        first_packed_weight,
        second_packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        in_features,
        grad_input,
    )
    return grad_input
