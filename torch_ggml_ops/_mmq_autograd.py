import torch

from ._mmq_cuda import (
    fixed_grouped_mmq_cuda,
    fixed_grouped_mmq_grad_input,
    grouped_mmq_cuda,
    grouped_mmq_grad_input,
    grouped_mmq_pair_cuda,
    grouped_mmq_pair_grad_input,
    mmq_cuda,
    mmq_grad_input,
)


class MMQFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        packed_weight: torch.Tensor,
        quant_type: int,
        out_features: int,
    ) -> torch.Tensor:
        ctx.quant_type = quant_type
        ctx.in_features = input.shape[-1]
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(packed_weight)
        return mmq_cuda(input, packed_weight, quant_type, out_features)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        grad_input = None
        if ctx.needs_input_grad[0]:
            (packed_weight,) = ctx.saved_tensors
            grad_input = mmq_grad_input(
                grad_output,
                packed_weight,
                ctx.quant_type,
                ctx.in_features,
            )
        return grad_input, None, None, None


class FixedGroupedMMQFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(packed_weight)
        return fixed_grouped_mmq_cuda(input, packed_weight)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        grad_input = None
        if ctx.needs_input_grad[0]:
            (packed_weight,) = ctx.saved_tensors
            grad_input = fixed_grouped_mmq_grad_input(grad_output, packed_weight)
        return grad_input, None


class GroupedMMQFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        packed_weight: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        quant_type: int,
        out_features: int,
    ) -> torch.Tensor:
        ctx.quant_type = quant_type
        ctx.in_features = input.shape[1]
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(packed_weight, expert_indices, expert_offsets)
        return grouped_mmq_cuda(
            input,
            packed_weight,
            expert_indices,
            expert_offsets,
            quant_type,
            out_features,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        grad_input = None
        if ctx.needs_input_grad[0]:
            packed_weight, expert_indices, expert_offsets = ctx.saved_tensors
            grad_input = grouped_mmq_grad_input(
                grad_output,
                packed_weight,
                expert_indices,
                expert_offsets,
                ctx.quant_type,
                ctx.in_features,
            )
        return grad_input, None, None, None, None, None


class GroupedMMQPairFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        first_packed_weight: torch.Tensor,
        second_packed_weight: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        quant_type: int,
        out_features: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.quant_type = quant_type
        ctx.in_features = input.shape[1]
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(
                first_packed_weight,
                second_packed_weight,
                expert_indices,
                expert_offsets,
            )
        return grouped_mmq_pair_cuda(
            input,
            first_packed_weight,
            second_packed_weight,
            expert_indices,
            expert_offsets,
            quant_type,
            out_features,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        first_grad_output, second_grad_output = grad_outputs
        grad_input = None
        if ctx.needs_input_grad[0]:
            if first_grad_output is None and second_grad_output is None:
                return grad_input, None, None, None, None, None, None
            if first_grad_output is None:
                assert second_grad_output is not None
                first_grad_output = torch.zeros_like(second_grad_output)
            if second_grad_output is None:
                second_grad_output = torch.zeros_like(first_grad_output)
            (
                first_packed_weight,
                second_packed_weight,
                expert_indices,
                expert_offsets,
            ) = ctx.saved_tensors
            grad_input = grouped_mmq_pair_grad_input(
                first_grad_output,
                second_grad_output,
                first_packed_weight,
                second_packed_weight,
                expert_indices,
                expert_offsets,
                ctx.quant_type,
                ctx.in_features,
            )
        return grad_input, None, None, None, None, None, None
