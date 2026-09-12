import gguf
import torch
from transformers.integrations.moe import _grouped_linear

from tests.mmq_test_support import find_tensor, load_packed_experts
from tools.gguf_dequant_compat import dequantize_gguf_tensor


def small_route() -> tuple[torch.Tensor, torch.Tensor]:
    experts = torch.tensor([0, 2, 5, 7], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([1, 4, 8, 10], device="cuda", dtype=torch.int32)
    return experts, offsets


def load_expert_weight(
    reader: gguf.GGUFReader,
    tensor_name: str,
    *,
    num_experts: int = 8,
    out_features: int = 37,
    row_offset: int = 0,
) -> tuple[torch.Tensor, gguf.GGMLQuantizationType, int]:
    tensor = find_tensor(reader, tensor_name)
    packed = load_packed_experts(
        tensor,
        num_experts=num_experts,
        out_features=out_features,
        row_offset=row_offset,
    )
    return packed, tensor.tensor_type, int(tensor.shape[0])


def dequantize_experts(
    packed: torch.Tensor,
    experts: torch.Tensor,
    quant_type: gguf.GGMLQuantizationType,
    out_features: int,
    in_features: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return dequantize_gguf_tensor(
        packed.index_select(0, experts),
        quant_type,
        dtype=dtype,
        device=packed.device,
    ).reshape(experts.numel(), out_features, in_features)


def grouped_forward_reference(
    input: torch.Tensor,
    packed: torch.Tensor,
    experts: torch.Tensor,
    offsets: torch.Tensor,
    quant_type: gguf.GGMLQuantizationType,
    out_features: int,
) -> torch.Tensor:
    logical = dequantize_experts(
        packed,
        experts,
        quant_type,
        out_features,
        input.shape[1],
        dtype=input.dtype,
    )
    return _grouped_linear(input, logical, offsets)


def fused_pair_grad_input_reference(
    first_grad_output: torch.Tensor,
    second_grad_output: torch.Tensor,
    first_logical_weight: torch.Tensor,
    second_logical_weight: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    expected = torch.empty(
        first_grad_output.shape[0],
        first_logical_weight.shape[-1],
        device=first_grad_output.device,
        dtype=first_grad_output.dtype,
    )
    row_begin = 0
    for group, row_end in enumerate(offsets.cpu().tolist()):
        combined = (
            first_grad_output[row_begin:row_end].float()
            @ first_logical_weight[group].float()
        )
        combined.addmm_(
            second_grad_output[row_begin:row_end].float(),
            second_logical_weight[group].float(),
        )
        expected[row_begin:row_end] = combined.to(first_grad_output.dtype)
        row_begin = row_end
    return expected


def grouped_grad_input_reference(
    grad_output: torch.Tensor,
    packed: torch.Tensor,
    experts: torch.Tensor,
    offsets: torch.Tensor,
    quant_type: gguf.GGMLQuantizationType,
    in_features: int,
) -> torch.Tensor:
    logical = dequantize_experts(
        packed,
        experts,
        quant_type,
        grad_output.shape[1],
        in_features,
        dtype=grad_output.dtype,
    )
    expected = torch.empty(
        grad_output.shape[0],
        in_features,
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    row_begin = 0
    for group, row_end in enumerate(offsets.cpu().tolist()):
        expected[row_begin:row_end] = grad_output[row_begin:row_end] @ logical[group]
        row_begin = row_end
    return expected
