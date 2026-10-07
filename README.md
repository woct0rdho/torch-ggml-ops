# torch-ggml-ops

This package provides PyTorch bindings of GGML operators with quantized weights. Currently it provides dense MMQ, grouped MMQ, grouped MMQ pair (for fused gate-up), and fixed grouped (batched) MMQ. All ops support gradients on input activations. See https://github.com/woct0rdho/transformers5-qwen3.5-recipe for example usage.

All ops support bf16 input activations.

The kernels are tested on Strix Halo (gfx1151), and they should also work on RDNA3 GPUs. More work is needed to support other GPUs.

The kernel parameters are tuned for weight shapes of Qwen3.5-35B-A3B, DeepSeek-V4-Flash, and Qwen3.8-Flash-Next, with typical context lengths. An autotune system is possible but not yet implemented.

The forward kernels use int8 WMMA, following llama.cpp . The GGUF format is only optimized for forward where each matmul tile requires only one scale, but when doing backward each matmul tile crosses multiple quantized rows and requires multiple scales. So we do not use int8 WMMA, but dequantize each tile into bf16 and run bf16 WMMA. This is still faster and saves most of the VRAM compared to dequantizing the whole weights into bf16.

The backward kernels are implemented with CK Tile, and it should be straightforward to port them to CuTe on Nvidia GPUs.

## Installation

The C++ extension uses Python 3.10 ABI3 and libtorch 2.10 stable ABI. It requires PyTorch >= 2.10 . Build the package with ROCm and PyTorch in the current environment, and install in place:

```bash
pip install --no-build-isolation --no-deps -e .
```

Currently my forked [transformers with GGUF quantizer](https://github.com/woct0rdho/transformers/tree/gguf) is required to run the tests. Install it, and download the example GGUF models:
- https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/blob/main/Qwen3.6-35B-A3B-APEX-I-Mini.gguf
- https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
- https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/main/Q2_0

Then run the tests:

```bash
pytest tests/
```

## Expert distribution prior

When optimizing grouped MMQ, you need to know the model and the dataset. By definition, a grouped MMQ kernel's running time depends on the input expert partition. We fit a prior distribution of expert partitions from the model and the dataset, and use the median time over the distribution as the optimization target, see `docs/expert_distribution_prior.md`.

## GGTensile

GGTensile is an ongoing work to apply Tensile-like asm-level optimization on MMQ kernels. We already see it's faster than HIP in many cases.
