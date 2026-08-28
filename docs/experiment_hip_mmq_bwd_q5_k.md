# HIP MMQ Backward Q5_K Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ input-gradient kernels for Q5_K weights:

```text
grad_input[M,N] = grad_output[M,K] @ dequant_q5_k(weight[N,K])
```

BF16 cotangents and gradients use FP32 WMMA accumulation. Q5_K low/high payloads, signed scales, and block factors are reconstructed from packed GGUF weights and staged in the WMMA-facing LDS layout.

| Family | Forward weight `(N,K)` | Backward shape `(M,N,K)` | M values |
| --- | ---: | ---: | ---: |
| Narrow K/V/shared gate/up | `(512,2048)` | `(M,2048,512)` | `2048,8192,32768` |
| Shared-expert down | `(2048,512)` | `(M,512,2048)` | `2048,8192,32768` |

`HIP TFLOPS` is `2*M*N*K/time`. `HIP/torch.mm` is the throughput ratio against BF16 `torch.mm`; values above `1.00x` favor the packed kernel.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm |
| --- | ---: | ---: | ---: |
| Narrow K/V/gate/up | `(2048,2048,512)` | 20.649 | 0.907x |
| Narrow K/V/gate/up | `(8192,2048,512)` | 22.082 | 0.969x |
| Narrow K/V/gate/up | `(32768,2048,512)` | 22.762 | 0.985x |
| Shared down | `(2048,512,2048)` | 17.971 | 1.414x |
| Shared down | `(8192,512,2048)` | 12.342 | 0.814x |
| Shared down | `(32768,512,2048)` | 12.258 | 0.756x |

These are the current complete packed-gradient path values and their direct BF16 `torch.mm` comparison.

## Kernel implementation

The retained body uses four wave32 waves, exact 128x128/K32 ownership where applicable, cooperative width-16 low/high payload decode, decoded-weight LDS staging, FP32 correction, and BF16 stores. Exact K512 and K2048 bodies keep Q5-specific packed-row and signed-scale state separate from Q4_K.

Q5_K uses the forward-layout packed weight directly. Cotangents are not quantized, and no dense shadow or transposed weight is built in the kernel.

## Optimization log

### Tiled redesign and shape specialization

The first four-wave Qwen body replaced scalar decode and serial 16x16 ownership with cooperative extraction, LDS staging, multiple WMMA accumulators, and exact shape/row geometry. Exact Q5_K wrappers then removed runtime bounds and address state while retaining bounds-safe fallback code.

The exact Q5_K matrix improved its generic controls by `11.66-32.60%`. The smallest narrow row count is the main extraction discriminator; larger rows favor the packed path.

### Extraction and LDS controls

Q5_K reused the Q4_K framework for width-16 low/high decode and bounded packed prefetch. Prefetch improved the initial port by `12-24%`. A universal padded layout regressed Q5_K and was rejected, and aligned fragment loads regressed Q5_K by about `2-4%` in cross-format controls.

Narrow Q5 retained scalar extraction only at 2,048 rows, where it improved `17.41%`; it regressed `2.34%` and `6.52%` at 8,192 and 32,768 rows. Shared-down Q5 retained its four-BF16 XOR layout after an isolated swizzle8 improvement did not survive the complete matrix.

The accepted body keeps packed/decode temporaries dead before long WMMA phases. Cross-iteration packed prefetch, K64, double buffering, decoded-weight LDS caching, and broad swizzle changes either extended register lifetimes or reduced residency.

### Closed directions

The current source closes global J64, I128, alternate workgroup sizes, split-K, GSU, Stream-K, persistent workgroups, generic local-load rules, and compiler-managed prefetch arrays. Q5-specific results must not be inferred from Q4_K or Q3_K because packed high-bit reconstruction changes the register and LDS cost model.

## Correctness and resources

The retained Q5_K bodies use `247 VGPR / 17 SGPR / 8 KiB LDS` for narrow scalar extraction and `253 VGPR / 16 SGPR / 8 KiB LDS` for shared down. They have zero private storage, zero spills, no dynamic stack, no scratch, and no calls. Validation covers low/high payload planes, signed scale fields, block and tile boundaries, input-gradient and packed-weight mutation, independent GGUF dequantization, finite values, autograd, and exact output comparison.

## Evidence

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_narrow_q5_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_narrow_selected_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_narrow_scalar_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_narrow_selected_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_shared_swizzle4_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_shared_swizzle8_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_shared_swizzle4_after_25.json
```

The shared-down Q5_K body reaches 253 VGPRs but remains spill-free. A new experiment must reduce decode state or change the packed representation before reopening the closed geometry neighborhood.
