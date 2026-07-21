# torch-ggml-ops

This package provides PyTorch bindings of GGML operators with quantized weights. Currently it provides dense MMQ, grouped MMQ, and grouped MMQ pair for fused gate-up. All ops support gradients on inputs. See https://github.com/woct0rdho/transformers5-qwen3.5-recipe for example usage.

Currently all ops support bf16 input and output activations, and quant types `IQ2_S, Q3_K, Q4_K, Q5_K, Q6_K`, which are enough to run the APEX Mini quantized model.

The kernels are tested on Strix Halo (gfx1151), and they should also work on RDNA3 GPUs. More work is needed to support other GPUs.

The kernel parameters are tuned for typical input and weight shapes of Qwen3.5-35B-A3B. An autotune system is possible but not yet implemented.

The forward kernels are modified from llama.cpp . The GGUF format is only optimized for forward where each matmul tile needs only one scale, but when doing backward each matmul tile crosses multiple quantized blocks and requires multiple scales. So we do not use int8 MMA, but dequantize each tile into bf16 and run bf16 MMA. This is still faster and saves most of the VRAM compared to dequantizing the whole weights into bf16.

The backward kernels are implemented with CK Tile, and it should be straightforward to port them to CuTe on Nvidia GPUs.

## Installation

The C++ extension uses Python 3.10 ABI3 and libtorch 2.10 stable ABI. It requires PyTorch >= 2.10 .

Extract source code from the llama.cpp repo:

```bash
python3 tools/generate_vendor.py --llama-cpp=/path/to/llama.cpp/
```

Build the package with ROCm and PyTorch in the current environment, and install in place:

```bash
pip install --no-build-isolation --no-deps -e .
```

Currently my forked [transformers with GGUF quantizer](https://github.com/woct0rdho/transformers/tree/gguf) is required to run the tests. Install it, and download the [example GGUF model](https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/blob/main/Qwen3.6-35B-A3B-APEX-I-Mini.gguf), then run the tests:

```bash
GGUF_MMQ_TEST_MODEL=/path/to/model.gguf pytest tests/
```

## TODO

- fp16
- other quant types
- other GPUs
- autotune
