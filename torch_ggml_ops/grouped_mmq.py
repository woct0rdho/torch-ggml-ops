import torch


@torch.library.register_fake("torch_ggml_ops::grouped_mmq")
def _grouped_mmq_fake(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    quant_type: int,
    out_features: int,
) -> torch.Tensor:
    return input.new_empty((input.shape[0], out_features))


@torch.library.register_fake("torch_ggml_ops::grouped_mmq_grad_input")
def _grouped_mmq_grad_input_fake(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    quant_type: int,
    in_features: int,
) -> torch.Tensor:
    return grad_output.new_empty((grad_output.shape[0], in_features))


@torch.library.register_fake("torch_ggml_ops::grouped_mmq_pair_grad_input")
def _grouped_mmq_pair_grad_input_fake(
    first_grad_output: torch.Tensor,
    second_grad_output: torch.Tensor,
    first_packed_weight: torch.Tensor,
    second_packed_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    quant_type: int,
    in_features: int,
) -> torch.Tensor:
    return first_grad_output.new_empty((first_grad_output.shape[0], in_features))


@torch.library.register_fake("torch_ggml_ops::grouped_mmq_pair")
def _grouped_mmq_pair_fake(
    input: torch.Tensor,
    first_packed_weight: torch.Tensor,
    second_packed_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    quant_type: int,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_shape = (input.shape[0], out_features)
    return input.new_empty(output_shape), input.new_empty(output_shape)


def _grouped_packed_input_gradient(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    quant_type: int,
    in_features: int,
) -> torch.Tensor:
    return torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
        grad_output.contiguous(),
        packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        in_features,
    )


def _setup_grouped_mmq_context(ctx, inputs, output) -> None:
    (
        input,
        packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        _out_features,
    ) = inputs
    ctx.quant_type = quant_type
    ctx.in_features = input.shape[1]
    if ctx.needs_input_grad[0]:
        ctx.save_for_backward(packed_weight, expert_indices, expert_offsets)


def _grouped_mmq_backward(ctx, grad_output: torch.Tensor):
    grad_input = None
    if ctx.needs_input_grad[0]:
        packed_weight, expert_indices, expert_offsets = ctx.saved_tensors
        grad_input = _grouped_packed_input_gradient(
            grad_output,
            packed_weight,
            expert_indices,
            expert_offsets,
            ctx.quant_type,
            ctx.in_features,
        )
    return grad_input, None, None, None, None, None


torch.library.register_autograd(
    "torch_ggml_ops::grouped_mmq",
    _grouped_mmq_backward,
    setup_context=_setup_grouped_mmq_context,
)


def _grouped_mmq_grad_input_backward(ctx, grad_grad_input: torch.Tensor):
    raise RuntimeError(
        "torch_ggml_ops::grouped_mmq_grad_input does not support higher-order gradients"
    )


torch.library.register_autograd(
    "torch_ggml_ops::grouped_mmq_grad_input",
    _grouped_mmq_grad_input_backward,
)


def _grouped_mmq_pair_grad_input_backward(ctx, grad_grad_input: torch.Tensor):
    raise RuntimeError(
        "torch_ggml_ops::grouped_mmq_pair_grad_input does not support higher-order gradients"
    )


torch.library.register_autograd(
    "torch_ggml_ops::grouped_mmq_pair_grad_input",
    _grouped_mmq_pair_grad_input_backward,
)


def _setup_grouped_mmq_pair_context(ctx, inputs, output) -> None:
    (
        input,
        first_packed_weight,
        second_packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        _out_features,
    ) = inputs
    ctx.quant_type = quant_type
    ctx.in_features = input.shape[1]
    if ctx.needs_input_grad[0]:
        ctx.save_for_backward(
            first_packed_weight,
            second_packed_weight,
            expert_indices,
            expert_offsets,
        )


def _grouped_mmq_pair_backward(
    ctx,
    first_grad_output: torch.Tensor | None,
    second_grad_output: torch.Tensor | None,
):
    grad_input = None
    if ctx.needs_input_grad[0]:
        (
            first_packed_weight,
            second_packed_weight,
            expert_indices,
            expert_offsets,
        ) = ctx.saved_tensors
        if first_grad_output is not None and second_grad_output is not None:
            grad_input = torch.ops.torch_ggml_ops.grouped_mmq_pair_grad_input.default(
                first_grad_output.contiguous(),
                second_grad_output.contiguous(),
                first_packed_weight,
                second_packed_weight,
                expert_indices,
                expert_offsets,
                ctx.quant_type,
                ctx.in_features,
            )
        elif first_grad_output is not None:
            grad_input = _grouped_packed_input_gradient(
                first_grad_output,
                first_packed_weight,
                expert_indices,
                expert_offsets,
                ctx.quant_type,
                ctx.in_features,
            )
        elif second_grad_output is not None:
            grad_input = _grouped_packed_input_gradient(
                second_grad_output,
                second_packed_weight,
                expert_indices,
                expert_offsets,
                ctx.quant_type,
                ctx.in_features,
            )
    return grad_input, None, None, None, None, None, None


torch.library.register_autograd(
    "torch_ggml_ops::grouped_mmq_pair",
    _grouped_mmq_pair_backward,
    setup_context=_setup_grouped_mmq_pair_context,
)


def grouped_mmq(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    quant_type: int,
    out_features: int,
) -> torch.Tensor:
    """Run grouped packed GGUF MMQ over expert-sorted BF16 rows."""

    return torch.ops.torch_ggml_ops.grouped_mmq.default(
        input,
        packed_weight,
        expert_indices,
        expert_offsets,
        int(quant_type),
        int(out_features),
    )


def grouped_mmq_pair(
    input: torch.Tensor,
    first_packed_weight: torch.Tensor,
    second_packed_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    quant_type: int,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run two grouped projections while sharing one Q8_1 activation workspace."""

    return torch.ops.torch_ggml_ops.grouped_mmq_pair.default(
        input,
        first_packed_weight,
        second_packed_weight,
        expert_indices,
        expert_offsets,
        int(quant_type),
        int(out_features),
    )


__all__ = ["grouped_mmq", "grouped_mmq_pair"]
