# HIP MMQ Backward Q6_K Experiment

## Scope

This record covers the Qwen language-model-head Q6_K packed input-gradient kernel on gfx1151:

```text
grad_input[M,2048] = grad_output[M,248320] @ dequant_q6_k(weight[248320,2048])
```

The packed Q6_K weight has logical shape `(248320,2048)`. Cotangents and gradients are BF16, accumulation is FP32 WMMA, and Q6_K blocks are decoded directly from packed GGUF storage.

## Final kernel result

`HIP TFLOPS` is `2*M*N*K/time`. `HIP/torch.mm` is the throughput ratio against BF16 `torch.mm`; values above `1.00x` favor the packed HIP kernel.

| `(M,N,K)` | HIP TFLOPS | HIP/torch.mm |
| ---: | ---: | ---: |
| `(64,2048,248320)` | 12.152 | 1.84x |
| `(128,2048,248320)` | 14.332 | 1.59x |
| `(256,2048,248320)` | 22.206 | 1.62x |

The values use the current Q6_K packed/BF16 kernel matrix. M256 is the primary large chunk, with M64 and M128 as smaller exact geometries.

## Kernel implementation

Q6_K uses its own 210-byte block decoder:

```text
ql[128]      low four bits
qh[64]       high two bits
scales[16]   signed int8 scale per 16 values
d             FP16 block multiplier
```

For value `i`, the kernel preserves:

```text
q = low4(i) | (high2(i) << 4)
value = fp16(d) * float(int8(scale[i/16])) * float(q - 32)
```

The retained exact bodies use M64/N32/K64 for M64, M128/N64/K32 for M128, and two M128-style workgroups for M256. The decoded-weight LDS layout, packed extraction, and register lifetime are Q6-specific.

## Optimization log

### Initial tiled redesign

The first redesign added four wave32 waves, LDS-staged decoded weights, multiple WMMA accumulators, cooperative packed decode, and exact row geometry. The representative LM-head Q6_K case moved from `158.209 ms` to `26.130 ms` for M256.

The selected small-row geometry screen was:

| M | Retained geometry and layout | Packed/BF16 timing |
| ---: | --- | ---: |
| 64 | M64/N32/K64, 16-BF16 XOR | `5.357/9.863 ms`, `1.84x` |
| 128 | M128/N64/K32, eight-BF16 XOR | `9.084/14.480 ms`, `1.59x` |
| 256 | Two M128/N64/K32 workgroups, packed extraction, eight-BF16 XOR | `11.726/19.010 ms`, `1.62x` |

M64 benefited from a 16-BF16 XOR layout that changed the decoded row bank phase without increasing LDS. M128 benefited from exact loader ownership. M256 was faster as two smaller workgroups because parallelism and lower accumulator pressure outweighed repeated packed decode.

### Invalid and rejected shapes

Two apparently fast M256 N=5/N=7 measurements were invalid: the selected N tile did not divide the 2,048-column result. The corrected N=7 body measured `29.628 ms`. K16/K64, M128 N3, M64 N3/N4, and alternate four-/16-BF16 layouts were rejected.

Packed extraction remained selected for M256; scalar extraction regressed `2.14%`. Exact bounds and K state accounted for only `2.09-2.56%` of the final M64/M128/M256 result, so the remaining M256 cost is packed Q6 arithmetic/representation rather than generic bounds handling.

### Arithmetic boundary

An effective-scale loader could stage `float(block_d * scale)` once per decoded row/K iteration. That changes the current integer-product-first rounding order, so it is outside bitwise current-output parity. A future approximate-order experiment must pass independent GGUF correctness, zero private storage/spills/stack, and stable complete-kernel timing across all three chunks.

Global J64, I128, broad K64, activation double buffering, decoded-weight caching, speculative prefetch, split-K, persistent workgroups, and broad swizzle sweeps are closed for the present arithmetic contract.

## Correctness and resources

Retained Q6_K bodies use `87/138/137 VGPR` for M64/M128/M256, `15/16/15 SGPR`, and 4 KiB LDS. They have zero private storage, zero spills, no dynamic stack, and no scratch or calls. Validation covers low/high payloads, signed scale extremes, FP16 `d`, all scale groups, block boundaries, one-hot decode, input-gradient and packed-weight mutation, independent GGUF references, finite output, and exact valid-tile divisibility.

## Evidence

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_packed_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_scalar_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_packed_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_narrow_q5_25.json
```

The Q6-specific shape and arithmetic controls are:

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_packed_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_scalar_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_packed_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_post_ds4_p3_control_9.json
```

The invalid N5/N7 result remains a permanent warning: exact tile divisibility is part of the Q6_K kernel contract.
