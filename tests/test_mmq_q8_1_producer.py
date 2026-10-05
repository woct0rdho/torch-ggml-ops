"""The Q8_1 activation producer dispatches between its two HIP bodies by size."""

import ctypes

import pytest
import torch

from tools.mmq_runtime import (
    Q8_1_QUANTIZER_ABI,
    FixedQ81F16D4S4QuantizerModule,
    FixedQ81F32D4QuantizerModule,
    _find_installed_kernel,
    _HIPModule,
)

_PRODUCERS = (FixedQ81F32D4QuantizerModule, FixedQ81F16D4S4QuantizerModule)


def _launch_row_producer(
    module: FixedQ81F32D4QuantizerModule | FixedQ81F16D4S4QuantizerModule,
    source: torch.Tensor,
    output: torch.Tensor,
) -> None:
    row = _HIPModule(_find_installed_kernel(module.SYMBOL), None, module.SYMBOL)
    try:
        arguments = Q8_1_QUANTIZER_ABI.pack(
            {
                "input": source.data_ptr(),
                "output": output.data_ptr(),
                "rows": source.shape[0],
                "rows_padded": source.shape[0],
                "k": source.shape[1],
            }
        )
        row._check(
            row._lib.hipModuleLaunchKernel(
                row._function,
                source.shape[0],
                1,
                1,
                512,
                1,
                1,
                0,
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )
    finally:
        row.close()


@pytest.mark.parametrize("producer", _PRODUCERS)
def test_q8_1_producer_dispatch_matches_the_row_producer(
    producer: type[FixedQ81F32D4QuantizerModule | FixedQ81F16D4S4QuantizerModule],
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("the Q8_1 producers need a HIP device")
    torch.manual_seed(0)
    module = producer()
    try:
        source = torch.randn(512, 1024, dtype=torch.bfloat16, device="cuda") * 0.25
        assert module._uses_grouped_producer(*source.shape)
        workspace = module.allocate(source)
        reference = torch.empty_like(workspace)
        module.launch(source, workspace, stream=torch.cuda.current_stream().cuda_stream)
        _launch_row_producer(module, source, reference)
        torch.cuda.synchronize()
        # The grouped body keeps the row body's arithmetic, so the workspace
        # must be bitwise identical and the dispatch costs nothing in accuracy.
        assert torch.equal(reference, workspace)
    finally:
        module.close()


@pytest.mark.parametrize("producer", _PRODUCERS)
def test_q8_1_producer_dispatch_keeps_the_row_body_for_large_rows(
    producer: type[FixedQ81F32D4QuantizerModule | FixedQ81F16D4S4QuantizerModule],
) -> None:
    module = producer()
    try:
        assert not module._uses_grouped_producer(32768, 2048)
        assert not module._uses_grouped_producer(8192, 2048)
        assert module._uses_grouped_producer(2048, 2048)
        assert module._uses_grouped_producer(2048, 4096)
        # The threshold is on the activation bytes the producer reads.
        assert module.GROUPED_MAX_INPUT_BYTES == 20 * 1024 * 1024
    finally:
        module.close()
