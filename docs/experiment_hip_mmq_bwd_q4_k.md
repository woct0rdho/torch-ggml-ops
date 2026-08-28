# HIP MMQ Backward Q4_K Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ input-gradient kernels for Q4_K weights:

```text
grad_input[M,N] = grad_output[M,K] @ dequant_q4_k(weight[N,K])
```

The cotangent and gradient are BF16. Q4_K weights are decoded directly from packed GGUF storage, staged for WMMA, accumulated in FP32, and rounded to BF16 on store.

| Family | Forward weight `(N,K)` | Backward shape `(M,N,K)` | M values |
| --- | ---: | ---: | ---: |
| Query/query gate | `(8192,2048)` | `(M,2048,8192)` | `2048,8192,32768` |
| Narrow K/V/shared gate/up | `(512,2048)` | `(M,2048,512)` | `2048,8192,32768` |
| Attention output | `(2048,4096)` | `(M,4096,2048)` | `2048,8192,32768` |
| Shared-expert down | `(2048,512)` | `(M,512,2048)` | `2048,8192,32768` |

`HIP TFLOPS` is `2*M*N*K/time`. `HIP/torch.mm` is the throughput ratio against BF16 `torch.mm`; values above `1.00x` favor HIP.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm |
| --- | ---: | ---: | ---: |
| Query/query gate | `(2048,2048,8192)` | 20.355 | 1.245x |
| Query/query gate | `(8192,2048,8192)` | 21.873 | 1.268x |
| Query/query gate | `(32768,2048,8192)` | 22.618 | 1.264x |
| Narrow K/V/gate/up | `(2048,2048,512)` | 18.674 | 0.824x |
| Narrow K/V/gate/up | `(8192,2048,512)` | 22.168 | 0.958x |
| Narrow K/V/gate/up | `(32768,2048,512)` | 22.998 | 0.988x |
| Attention output | `(2048,4096,2048)` | 25.661 | 1.110x |
| Attention output | `(8192,4096,2048)` | 23.247 | 1.009x |
| Attention output | `(32768,4096,2048)` | 23.096 | 0.990x |
| Shared down | `(2048,512,2048)` | 16.456 | 1.295x |
| Shared down | `(8192,512,2048)` | 11.178 | 0.737x |
| Shared down | `(32768,512,2048)` | 13.050 | 0.802x |

The current source matrix is the complete packed-gradient path. The ratios compare that HIP path with BF16 `torch.mm`.

## Kernel implementation

The retained Q4_K body is a four-wave 128x128/K32 tiled decoder with packed nibble and scale/minimum reconstruction, decoded-weight LDS staging, FP32 WMMA, and BF16 RNE stores. Exact N coverage includes 512, 2048, and 4096 output features with K-specific packed-row accounting.

The backward cotangent is consumed directly in BF16. The kernel computes `dY @ W` from forward-layout packed weights without materializing a transposed or dense weight. Bounds-safe tails and generic fallbacks remain outside the exact tiled bodies.

## Optimization log

### Qwen tiled geometry

The first redesign replaced scalar 16x16 ownership with four wave32 waves, multiple accumulators, cooperative packed decode, and LDS-staged weights. The common retained geometry is 128x128/K32. Early global J64 and larger ownership alternatives lost reuse, parallelism, or residency.

Q4_K shape-specific controls retain padded vector local loads for query/narrow and a 16-BF16 XOR layout for shared down. Shared-down K512 is a distinct cost model and does not inherit the query layout.

### Padding and local-load experiments

Pre-layout Q4_K controls reported a repeated `79.2%` LDS-bank-conflict metric. Eight BF16 values of row padding changed the K32 stride from 64 to 80 bytes and improved query, narrow, and attention-output representative points, while shared down regressed:

| Shape | Unpadded | Padded | Decision |
| --- | ---: | ---: | --- |
| Query | 54.895 ms | 46.295 ms | retain typed padding |
| Narrow | 3.273 ms | 2.907 ms | retain typed padding |
| Attention output | 26.875 ms | 23.177 ms | retain typed padding |
| Shared down | 5.261 ms | 5.389 ms | reject padding for this body |

Explicit aligned fragment loads helped narrow Q3_K and attention-output Q4_K but regressed query Q4_K, Q5_K, and shared-down controls. Lower instruction count did not predict lower event time or LDS stalls.

### Pipeline and exact-shape work

The true two-buffer decoded-weight pipeline reduced aggregate wave cycles and wait/barrier stalls in the selected controls. It required a live-range repair after K64 reused addresses still needed by later WMMA pairs. The corrected handoff is exact at K32, K64, K96, and K512.

Exact-shape simplifications removed dead dimension loads, shortened address state, strength-reduced power-of-two strides, removed a fixed final `s_nop 7`, and normalized packed Q4 nibbles once per dword. These changes are resource-neutral or reducing and preserve the 40-byte ABI and output order.

## Correctness and resources

Retained Q4_K bodies use `226 VGPR / 17 SGPR / 8 KiB LDS` for query/narrow, `222 VGPR / 20 SGPR / 10 KiB LDS` for attention output, and `222 VGPR / 16 SGPR / 8 KiB LDS` for shared down. They have zero private storage, zero spills, no scratch or calls, and no dynamic stack. Validation covers Q4_K nibbles, scale/minimum fields, block boundaries, K tails, input-gradient and packed-weight mutation, independent BF16 references, producer handoff, autograd, finite output, and deterministic rebuilds.

## Evidence

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_narrow_q5_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_post_db8_control_9.json
~/tmp/torch-ggml-ops/mmq_bwd_final_full_v3.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb0_folded_25.json
```

Additional retained Q4 controls include the generic/exact DeepSeek build bracket and the Qwen baseline matrix:

```text
~/tmp/torch-ggml-ops/mmq_bwd_baseline_primary_sequential.json
~/tmp/torch-ggml-ops/mmq_bwd_final_full_v3.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_exact_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_generic_after_25.json
```

Global J64, I128, broad K64, split-K, GSU, Stream-K, persistent workgroups, compiler-managed prefetch arrays, decoded-weight LDS caching, and universal swizzle policies are closed for the current Q4_K representation.
