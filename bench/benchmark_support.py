"""Shared benchmark launch and inventory guards."""

import sys
from pathlib import Path

import torch

BENCH_ROOT = Path(__file__).resolve().parent
ROOT = BENCH_ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_inventory import BenchmarkCase, validate_direct_artifact

import torch_ggml_ops


def require_case(
    operation: str,
    key,
    code_object: Path,
    tensor_names: tuple[str, ...],
) -> BenchmarkCase:
    case = validate_direct_artifact(operation, key, code_object)
    expected = case.tensor_source.names
    if tensor_names != expected:
        raise ValueError(
            f"tensor candidates for {case.identity} must be {expected!r}; "
            f"received {tensor_names!r}"
        )
    return case


def public_dense_backward(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_type: int,
    in_features: int,
) -> torch.Tensor:
    input_tensor = torch.zeros(
        (*grad_output.shape[:-1], in_features),
        dtype=grad_output.dtype,
        device=grad_output.device,
        requires_grad=True,
    )
    output = torch_ggml_ops.mmq(
        input_tensor, packed_weight, quant_type, grad_output.shape[-1]
    )
    return torch.autograd.grad(output, input_tensor, grad_output)[0]


def public_grouped_backward(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    quant_type: int,
    in_features: int,
) -> torch.Tensor:
    input_tensor = torch.zeros(
        grad_output.shape[0],
        in_features,
        dtype=grad_output.dtype,
        device=grad_output.device,
        requires_grad=True,
    )
    output = torch_ggml_ops.grouped_mmq(
        input_tensor,
        packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        grad_output.shape[-1],
    )
    return torch.autograd.grad(output, input_tensor, grad_output)[0]


def public_grouped_pair_backward(
    first_grad_output: torch.Tensor,
    second_grad_output: torch.Tensor,
    first_packed_weight: torch.Tensor,
    second_packed_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    quant_type: int,
    in_features: int,
) -> torch.Tensor:
    input_tensor = torch.zeros(
        first_grad_output.shape[0],
        in_features,
        dtype=first_grad_output.dtype,
        device=first_grad_output.device,
        requires_grad=True,
    )
    first_output, second_output = torch_ggml_ops.grouped_mmq_pair(
        input_tensor,
        first_packed_weight,
        second_packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        first_grad_output.shape[-1],
    )
    return torch.autograd.grad(
        (first_output, second_output),
        input_tensor,
        (first_grad_output, second_grad_output),
    )[0]


def public_fixed_backward(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    in_features: int,
) -> torch.Tensor:
    tokens, groups, _ = grad_output.shape
    input_tensor = torch.zeros(
        tokens,
        groups,
        in_features,
        dtype=grad_output.dtype,
        device=grad_output.device,
        requires_grad=True,
    )
    output = torch_ggml_ops.fixed_grouped_mmq(input_tensor, packed_weight)
    return torch.autograd.grad(output, input_tensor, grad_output)[0]
