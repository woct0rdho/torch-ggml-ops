"""Device-aware GGUF payload dequantization for the current transformers layout.

The reference dequantizer moved from `transformers.integrations.gguf_dequant`
to `transformers.integrations.gguf.dequant` and no longer takes a device.
This module keeps the single call shape used by the correctness references and
the benchmark baselines.
"""

from collections.abc import Sequence

import gguf
import torch
from transformers.integrations.gguf.dequant import dequantize


def dequantize_gguf_tensor(
    data: torch.Tensor,
    quant_type: gguf.GGMLQuantizationType | int,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Dequantize a GGUF payload to flat `dtype` values on `device`.

    Args:
        data: Packed payload bytes, addressed as `ggml` blocks along its last axis.
        quant_type: The payload's `gguf.GGMLQuantizationType` value.
        dtype: Floating-point output dtype, defaulting to `torch.float32`.
        device: Device on which to run the dequantization.
    """
    if dtype is None:
        dtype = torch.float32
    payload = data.contiguous()
    if device is not None:
        payload = payload.to(torch.device(device), non_blocking=True)
    return dequantize(payload, int(quant_type), dtype=dtype)


_DEQUANT_CHUNK_VALUES = 1 << 22


def dequantize_logical(
    packed: torch.Tensor,
    quant_type: gguf.GGMLQuantizationType | int,
    logical_shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
    chunk_values: int = _DEQUANT_CHUNK_VALUES,
) -> torch.Tensor:
    """Dequantize a packed payload into one preallocated logical tensor.

    The shared dequantizer allocates fp32 temporaries proportional to its whole
    input, peaking at several times the final weight for large payloads.
    Chunking rows of the leading axes bounds every temporary while producing
    identical values, because a chunk boundary never splits a GGML block along
    the last axis.
    """

    output = torch.empty(
        tuple(logical_shape),
        dtype=dtype,
        device=torch.device(device) if device is not None else packed.device,
    )
    rows = packed.reshape(-1, packed.shape[-1])
    flat = output.reshape(-1)
    if rows.shape[0] == 0:
        if flat.numel():
            raise ValueError(
                f"logical shape {tuple(logical_shape)} does not match packed "
                f"payload shape {tuple(packed.shape)}"
            )
        return output
    values_per_row = flat.numel() // rows.shape[0]
    if values_per_row * rows.shape[0] != flat.numel():
        raise ValueError(
            f"logical shape {tuple(logical_shape)} does not match packed "
            f"payload shape {tuple(packed.shape)}"
        )
    chunk_rows = max(1, max(1, chunk_values) // max(1, values_per_row))
    for start in range(0, rows.shape[0], chunk_rows):
        stop = min(start + chunk_rows, rows.shape[0])
        piece = dequantize_gguf_tensor(
            rows[start:stop],
            quant_type,
            dtype=dtype,
            device=output.device,
        )
        expected_values = (stop - start) * values_per_row
        if piece.numel() != expected_values:
            raise ValueError(
                f"logical shape {tuple(logical_shape)} does not match packed "
                f"payload shape {tuple(packed.shape)}"
            )
        flat[start * values_per_row : stop * values_per_row] = piece.reshape(-1)
    return output
