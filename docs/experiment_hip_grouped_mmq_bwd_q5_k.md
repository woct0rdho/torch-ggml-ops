# HIP Grouped MMQ Backward Q5_K Experiment

## Scope

This record covers the routed Q5_K down input-gradient kernel for Qwen:

```text
dX[R,512] = dY[R,2048] @ W[2048,512]
```

The packed Q5_K weight is decoded in the kernel from forward layout. BF16 cotangents and gradients use FP32 WMMA accumulation. Aggregate rows are `R=16384,65536,262144`.

## Final kernel result

`HIP TFLOPS` is `2*R*N*K/time`. `HIP/AITER GMM` is the packed HIP throughput ratio against BF16 AITER GMM. Values above `1.00x` favor HIP.

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM |
| ---: | ---: | ---: | ---: |
| 1 | `(16384,512,2048)` | 9.38 | 0.849x |
| 4 | `(65536,512,2048)` | 14.54 | 0.789x |
| 16 | `(262144,512,2048)` | 17.37 | 0.761x |

The remaining loss is Q5 high-bit reconstruction and packed metadata cost relative to predecoded BF16 weights.

## Kernel implementation

The retained body uses four wave32 waves, M64/N64 for small groups, M128/N128 M-major row tasks for large groups, width-16 low/high decode, Q5 scale/minimum reconstruction, and BF16 stores. Q5 row-task decode uses swizzle8 while the smaller serial body uses swizzle4.

Wholly inactive 16-row M minitiles skip cotangent loads, WMMA, and stores while decode and barriers remain uniform. The packed Q5 decoder and LDS state are distinct from Q4_K.

## Optimization log

### Tiled body and task ownership

The original generic grouped kernel used eight waves, N16/K16 ownership, scalar decode, and serial row chunks. The Q5 body was rebuilt on the four-wave Q4 framework and specialized for low/high payload reconstruction. Device-built M-major row tasks improved the representative B16 uniform point from `37.421 ms` to `34.074 ms`.

N-major order nearly doubled B16 latency. Fixed 1,024-program traversal spilled Q5 state; runtime full/tail branching and split task lists regressed nonuniform routes. Row-task setup is approximately `0.004 ms` and is not the residual cost.

### Extraction and swizzle controls

Width-16 low/high decode and bounded packed prefetch improved the initial port by `12-24%`. A universal swizzle8 policy regressed M64/B1 by `5.6-22.5%`; sequential row-task controls improved B4/B16 by `4.2-6.4%` and `5.2-6.6%`, so swizzle8 remains row-task-specific.

Inactive-M suppression was retained for Q5 row tasks. The row-task body fell from 256 to 233 VGPRs before final suppression and remains near the resource warning boundary. M256/N64, width-8 decode, broad J changes, two-LDS caches, split-K, Stream-K, and persistent traversal were rejected.

### Shared backward arithmetic controls

The grouped backward campaign tested reduced-precision accumulation as a separate kernel mechanism. Direct BF16-C reached `0.86089` NRMSE at 513 rows and was rejected for accuracy. Full-N FP32 K32/K64 slabs reached 256 VGPRs with 647/2,069 spills and 1,568/5,248 private bytes and were rejected before timing. Pair-serial slabs reached `0.01343/0.00959` NRMSE and were 35.0%/82.9% slower. Row-normalized FP16-C reached `0.00723-0.00727` NRMSE; even without its scale scan it was 33.3% slower. Q5_K therefore retains exact FP32 WMMA accumulation.

## Correctness and resources

Retained Q5_K bodies use 115 VGPR/22 SGPR/4096 B LDS for M64/N64 and 234/26/8192 for row-task M128/N128. They have zero private bytes, zero spills, no scratch, calls, or dynamic stack. Validation covers low/high payloads, scale/minimum fields, inactive experts, sparse/repeated IDs, malformed offsets, non-aligned tails, input/weight mutation, independent BF16 reference error, finite output, and deterministic reruns.

## Evidence

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step5_q5_prefetch.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_q5_sparse_s2_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_q5_swizzle8_rowtask_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_q5_swizzle4_rowtask_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_rowtask_tail_predicate_control_true_25.json
```

The remaining Q5_K loss is packed high-bit decode and accumulator/resource pressure. Further generic geometry work is closed.
