import gguf
import numpy as np
import torch


def find_tensor(reader: gguf.GGUFReader, name: str) -> gguf.ReaderTensor:
    tensor = next(
        (tensor for tensor in reader.tensors if tensor.name == name),
        None,
    )
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {name}")
    return tensor


def _to_cuda_uint8(data: np.ndarray) -> torch.Tensor:
    host = np.array(data, dtype=np.uint8, copy=True, order="C")
    packed = torch.from_numpy(host).to("cuda")
    del host
    return packed


def load_packed_tensor(
    tensor: gguf.ReaderTensor,
    out_features: int | None = None,
) -> torch.Tensor:
    data = tensor.data if out_features is None else tensor.data[:out_features]
    return _to_cuda_uint8(data)


def load_packed_rows(
    tensor: gguf.ReaderTensor,
    out_features: int,
    *,
    row_offset: int = 0,
    expert: int = 0,
) -> torch.Tensor:
    rows = tensor.data[expert] if tensor.data.ndim == 3 else tensor.data
    return _to_cuda_uint8(rows[row_offset : row_offset + out_features])


def load_packed_experts(
    tensor: gguf.ReaderTensor,
    *,
    num_experts: int = 8,
    out_features: int = 37,
    row_offset: int = 0,
) -> torch.Tensor:
    row_slice = slice(row_offset, row_offset + out_features)
    if tensor.data.ndim == 3:
        data = tensor.data[:num_experts, row_slice]
    else:
        data = np.repeat(tensor.data[row_slice][None, ...], num_experts, axis=0)
    return _to_cuda_uint8(data)


def load_packed_fixed_groups(
    tensor: gguf.ReaderTensor,
    *,
    groups: int,
    group_out_features: int,
    out_features: int,
) -> torch.Tensor:
    grouped = tensor.data.reshape(groups, group_out_features, tensor.data.shape[-1])
    return _to_cuda_uint8(grouped[:, :out_features])


def random_bf16(
    *shape: int,
    seed: int,
    requires_grad: bool = False,
) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        *shape,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=requires_grad,
    )


def assert_normalized_rmse(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    maximum: float = 0.04,
) -> None:
    error_rms = (actual.float() - expected.float()).square().mean().sqrt()
    reference_rms = expected.float().square().mean().sqrt()
    assert torch.isfinite(actual).all()
    assert (error_rms / reference_rms).item() < maximum


def assert_fused_pair_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    assert_normalized_rmse(actual, expected, maximum=5e-5)
    assert (actual.float() - expected.float()).abs().max().item() <= 2**-12
