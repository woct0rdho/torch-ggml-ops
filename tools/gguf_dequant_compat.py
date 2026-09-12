"""Device-aware GGUF payload dequantization for the current transformers layout.

The reference dequantizer moved from ``transformers.integrations.gguf_dequant``
to ``transformers.integrations.gguf.dequant`` and no longer takes a device.
This module keeps the single call shape used by the correctness references and
the benchmark baselines.
"""

import gguf
import torch
from transformers.integrations.gguf.dequant import dequantize


def dequantize_gguf_tensor(
    data: torch.Tensor,
    quant_type: gguf.GGMLQuantizationType | int,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Dequantize a GGUF payload to flat ``dtype`` values on ``device``.

    Args:
        data: Packed payload bytes, addressed as ``ggml`` blocks along its last
            axis.
        quant_type: The payload's ``gguf.GGMLQuantizationType`` value.
        dtype: Floating-point output dtype, defaulting to ``torch.float32``.
        device: Device on which to run the dequantization.
    """
    if dtype is None:
        dtype = torch.float32
    payload = data.contiguous()
    if device is not None:
        payload = payload.to(torch.device(device), non_blocking=True)
    return dequantize(payload, int(quant_type), dtype=dtype)
