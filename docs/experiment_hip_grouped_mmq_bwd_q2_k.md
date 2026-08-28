# HIP Grouped MMQ Backward Q2_K Experiment

## Scope

This record covers the routed Q2_K down input-gradient kernel for DeepSeek:

```text
dX[R,2048] = dY[R,4096] @ W[4096,2048]
```

Aggregate routed rows are `R=12288,49152,196608`. The kernel consumes forward-layout packed Q2_K weights, reconstructs scale/minimum state, accumulates with FP32 WMMA, and stores BF16 gradients.

## Final kernel result

`HIP TFLOPS` is `2*R*N*K/time`. `HIP/AITER GMM` compares the packed HIP kernel with BF16 AITER GMM. Values above `1.00x` favor HIP.

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM |
| ---: | ---: | ---: | ---: |
| 1 | `(12288,2048,4096)` | 11.43 | 1.162x |
| 4 | `(49152,2048,4096)` | 17.19 | 0.849x |
| 16 | `(196608,2048,4096)` | 19.20 | 0.712x |

B4/B16 remain slower than the predecoded BF16 baseline because packed scale/minimum reconstruction and limited N64 reuse dominate.

## Kernel implementation

The retained Q2_K body uses four wave32 waves, M64/N64 U1 and M128/N64 U1/U2 geometries, width-16 decode, shared scale/minimum and packed-shift work, inactive-M consumer suppression, and serial route ownership. A bounded J16 tail is used only in measured small-route/exact-row bodies.

## Optimization log

### Exact body and reduction unroll

The generic grouped body used narrow N16/K16 ownership and scalar decode. Exact M64/M128 N64/K32 ownership increased reuse and removed much of the serial overhead. Width-16 decode shares each scale/minimum group and packed shift across sixteen values.

U2 improved all B4 routes but regressed B1 and one B16 boundary route, so it was kept as a bounded geometry. Sequential B4 controls confirmed U2 over U1 by `1.83-2.83%`. U4 lost to U2 by `1.2-4.3%`.

### Inactive-M and N geometry

Inactive-M suppression skips cotangent loads, WMMA, and stores for wholly inactive 16-row minitiles while decode and barriers remain unconditional. It improved B1 by `8.69-15.15%`, kept B4 changes within `0.67%`, and improved the complete matrix geometrically by `4.73%`.

A valid N128/U1 body compiled at 255 VGPRs but lost every B16 route by `1.18-4.18%`, or `3.06%` geometrically. Row tasks were not added: N64 already launches 32 N workgroups per expert and thousands of workgroups overall, so extra descriptors would not remove rounded tail arithmetic.

### Bottleneck attribution

The selected B16 counter review measured:

| Metric | HIP/AITER result |
| --- | ---: |
| Total instructions | `10.88x` |
| VALU instructions | `24.27x` |
| VALU issue cycles | `22.39x` |
| Instruction-fetch waits | `10.36x` |
| Mean occupancy per active CU | `6.25 / 10.67` waves |
| Video-memory fetch | `0.585x` AITER bytes |
| ALU stalled by LDS | `0.088% / 33.78%` |

The packed kernel fetches fewer bytes but executes a much larger decode/instruction stream at lower residency. This is not a total-memory or LDS-bank problem.

### Shared backward arithmetic controls

The grouped backward campaign tested reduced-precision accumulation as a separate kernel mechanism. Direct BF16-C reached `0.86089` NRMSE at 513 rows and was rejected for accuracy. Full-N FP32 K32/K64 slabs reached 256 VGPRs with 647/2,069 spills and 1,568/5,248 private bytes and were rejected before timing. Pair-serial slabs reached `0.01343/0.00959` NRMSE and were 35.0%/82.9% slower. Row-normalized FP16-C reached `0.00723-0.00727` NRMSE; even without its scale scan it was 33.3% slower. Q2_K therefore retains exact FP32 WMMA accumulation.

## Correctness and resources

Retained Q2_K bodies use 102 VGPR/30 SGPR/4096 B LDS for M64/N64/U1, 146/31/4096 for M128/N64/U1, and 160/31/4096 for U2. All have zero private bytes, zero spills, no scratch, calls, or dynamic stack. Validation covers 2-bit values, scale/minimum fields, inactive M tiles, malformed routes, non-aligned tails, input/weight mutation, independent BF16 references, finite output, and deterministic reruns.

## Evidence

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_deepseek_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_u2_dispatch_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_u1_dispatch_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_n128_candidate_valid_b16_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_n64_bracket_b16_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_tail_predicate_candidate_25.json
```

The Q2_K campaign started from this generic DeepSeek control and exact tiled harness:

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_baseline_b1_b4.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_pre_ds4_control.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_routed_harness_check.json
```

The remaining Q2_K loss is packed decode and instruction pressure. Broad N, U, row-task, and persistent traversal sweeps are closed.
