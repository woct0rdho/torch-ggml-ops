import torch


@torch.library.register_fake("torch_ggml_ops::mmq")
def _mmq_fake(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_type: int,
    out_features: int,
) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], out_features))


@torch.library.register_fake("torch_ggml_ops::mmq_grad_input")
def _mmq_grad_input_fake(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_type: int,
    in_features: int,
) -> torch.Tensor:
    return grad_output.new_empty((*grad_output.shape[:-1], in_features))


def _setup_mmq_context(ctx, inputs, output) -> None:
    input, packed_weight, quant_type, _out_features = inputs
    ctx.quant_type = quant_type
    ctx.in_features = input.shape[-1]
    if ctx.needs_input_grad[0]:
        ctx.save_for_backward(packed_weight)


def _mmq_backward(ctx, grad_output: torch.Tensor):
    grad_input = None
    if ctx.needs_input_grad[0]:
        (packed_weight,) = ctx.saved_tensors
        # AOTAutograd may supply a transposed synthetic cotangent. Production
        # linear cotangents are contiguous; normalize only this autograd-owned
        # temporary while the public native operator keeps its fail-fast ABI.
        grad_input = torch.ops.torch_ggml_ops.mmq_grad_input.default(
            grad_output.contiguous(),
            packed_weight,
            ctx.quant_type,
            ctx.in_features,
        )
    return grad_input, None, None, None


torch.library.register_autograd(
    "torch_ggml_ops::mmq",
    _mmq_backward,
    setup_context=_setup_mmq_context,
)


def _mmq_grad_input_backward(ctx, grad_grad_input: torch.Tensor):
    raise RuntimeError(
        "torch_ggml_ops::mmq_grad_input does not support higher-order gradients"
    )


torch.library.register_autograd(
    "torch_ggml_ops::mmq_grad_input",
    _mmq_grad_input_backward,
)


def mmq(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_type: int,
    out_features: int,
) -> torch.Tensor:
    """Run packed GGUF WnA8 MMQ. Currently supports BF16 input and output."""

    return torch.ops.torch_ggml_ops.mmq.default(
        input,
        packed_weight,
        int(quant_type),
        int(out_features),
    )


__all__ = ["mmq"]
