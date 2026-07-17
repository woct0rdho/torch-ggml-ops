# torch-ggml-ops

This package provides PyTorch bindings for GGML operators with quantized weights. It currently provides dense MMQ and grouped expert MMQ.

## Operators

```text
torch_ggml_ops::mmq(
    Tensor input,
    Tensor packed_weight,
    int quant_type,
    int out_features,
) -> Tensor

torch_ggml_ops::mmq_grad_input(
    Tensor grad_output,
    Tensor packed_weight,
    int quant_type,
    int in_features,
) -> Tensor

torch_ggml_ops::grouped_mmq_grad_input(
    Tensor grad_output,
    Tensor packed_weight,
    Tensor expert_indices,
    Tensor expert_offsets,
    int quant_type,
    int in_features,
) -> Tensor

torch_ggml_ops::grouped_mmq_pair_grad_input(
    Tensor first_grad_output,
    Tensor second_grad_output,
    Tensor first_packed_weight,
    Tensor second_packed_weight,
    Tensor expert_indices,
    Tensor expert_offsets,
    int quant_type,
    int in_features,
) -> Tensor

torch_ggml_ops::grouped_mmq(
    Tensor input,
    Tensor packed_weight,
    Tensor expert_indices,
    Tensor expert_offsets,
    int quant_type,
    int out_features,
) -> Tensor

torch_ggml_ops::grouped_mmq_pair(
    Tensor input,
    Tensor first_packed_weight,
    Tensor second_packed_weight,
    Tensor expert_indices,
    Tensor expert_offsets,
    int quant_type,
    int out_features,
) -> (Tensor, Tensor)
```

`grouped_mmq` consumes expert-sorted rows and applies one packed matrix per group. `grouped_mmq_pair` runs two equal-geometry projections, such as MoE gate and up, while sharing one dynamic `Q8_1` activation workspace.

## Contract

All native operators require:

- contiguous HIP `torch.bfloat16` activation or cotangent input with zero storage offset.
- contiguous HIP `torch.uint8` packed GGUF payloads with zero storage offset.
- `IQ2_S`, `Q3_K`, `Q4_K`, `Q5_K`, or `Q6_K` weights.
- logical weight input width (`in_features`) divisible by 256.
- direct BF16 output on PyTorch's current stream.
- no hidden input, output, or packed-weight copy.
- currently only supports gfx1151.

Grouped MMQ additionally requires:

- two-dimensional input `[rows, in_features]` sorted into contiguous expert groups.
- physical packed weight shape `[num_experts, out_features, packed_row_bytes]`.
- contiguous, strictly increasing `torch.int64` active expert indices.
- contiguous, positive, strictly increasing `torch.int32` cumulative expert offsets.
- one positive group per active expert and a final offset equal to the input row count.

The forward operators allocate their compact `Q8_1` activation workspace through PyTorch. BF16 values are converted to FP32 only in registers for `Q8_1` scaling and rounding, then the packed-weight multiplication uses RDNA3 int8 WMMA with int32 accumulation. Grouped blocks read active expert payloads in place. Inactive experts are not copied or dequantized.

The registered autograd formulas define the frozen logical-weight Jacobian rather than differentiating through Q8 rounding. Dense `mmq` backward calls `mmq_grad_input`, while grouped backward calls `grouped_mmq_grad_input`. Both read BF16 `grad_output`, decode authoritative GGUF blocks directly to BF16 WMMA fragments, accumulate in FP32, and write BF16 `grad_input` without materializing logical weights. `grouped_mmq_pair_grad_input` accumulates both gate and up logical Jacobians in one FP32 accumulator and performs one BF16 output write. The narrow gfx11 WMMA wrapper is adapted from pinned MIT-licensed AMD Composable Kernel sources; provenance is recorded under `csrc/ck/`.

Grouped backward launch geometry uses only tensor shape metadata. Expert IDs and cumulative offsets remain device-resident and are read by the kernel on PyTorch's current stream; no host grouped-GEMM descriptor construction, device-to-host metadata copy, maximum-group synchronization, or CPU expert-index calculation is performed. Packed weights never receive gradients, and higher-order differentiation through the native input-gradient operators is rejected explicitly. The Qwen3.5 integration retains AITER GMM for the rank-small LoRA branches and AITER PTGMM for LoRA factor gradients.

The native grouped operators are projection primitives. Routing sort, gate activation, LoRA, routing-weight multiplication, inverse permutation, and top-k reduction remain framework responsibilities.

## Installation

The C++ extension uses Python 3.10 ABI3 and the LibTorch 2.10 stable ABI.

Extract source code from the llama.cpp repo:

```bash
python3 tools/generate_vendor.py --llama-cpp=/path/to/llama.cpp/
```

Build against ROCm and PyTorch in the current environment, and install in place:

```bash
pip install --no-build-isolation --no-deps -e .
```

Run unit tests:

```bash
GGUF_MMQ_TEST_MODEL=/path/to/model.gguf pytest tests/
```

## TODO

- Shape-aware grouped geometry and scheduling for very large per-expert route counts.
- Windows support.
