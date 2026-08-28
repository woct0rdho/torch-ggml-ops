# HIP Grouped MMQ Backward Q4_K Experiment

## Scope

This record covers the routed Q4_K down input-gradient kernel for Qwen:

```text
dX[R,512] = dY[R,2048] @ W[2048,512]
```

Inputs and gradients are BF16. The packed Q4_K weights are decoded directly into LDS and accumulated with FP32 WMMA. Aggregate rows are `R=16384,65536,262144`.

## Final kernel result

`HIP TFLOPS` is `2*R*N*K/time`. `HIP/AITER GMM` compares the packed HIP kernel with the exact BF16 AITER GMM baseline. Values above `1.00x` favor HIP.

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM |
| ---: | ---: | ---: | ---: |
| 1 | `(16384,512,2048)` | 9.70 | 0.869x |
| 4 | `(65536,512,2048)` | 13.56 | 0.808x |
| 16 | `(262144,512,2048)` | 17.19 | 0.758x |

The large-route deficit remains after exact geometry, row tasks, and inactive-M controls. AITER starts from predecoded BF16 weights.

## Kernel implementation

The retained Q4_K body uses four wave32 waves, M64/N64 for small groups, M128/N64 for intermediate groups, and M128/N128 M-major row tasks for large groups. It uses width-16 packed Q4 decode, a sixteen-BF16 XOR LDS layout, FP32 accumulation, and BF16 stores.

Q4_K row tasks suppress wholly inactive 16-row M minitiles while leaving decode and barriers uniform. The row-task descriptor setup is device-resident and atomics-free.

## Optimization log

### Initial tiled kernel

The generic grouped body used eight waves, N16/K16 ownership, scalar Q4 decode, and serial 128-row chunks. The first tiled Q4_K body established M128/N128/K32 ownership, width-16 decode, and sixteen-BF16 XOR LDS. It improved representative B4/B16 controls by 5-8x over the generic baseline.

M64/N64 bodies were added for small groups. Universal M128 ownership was rejected because uniform 64-row groups became half-empty. M-major row tasks were retained for large routes; N-major ordering nearly doubled B16 latency.

### Row-task and tail controls

The row-task body improved the representative Q4_K B16 uniform point from `40.270 ms` to `35.821 ms`. Fixed 1,024-program traversal, runtime full/tail branching, and split full/tail task lists were rejected for timing or resources. The task setup is about `0.004 ms`, so it is not the residual bottleneck.

Inactive-M suppression was applied around cotangent loads, WMMA, and stores while packed decode and barriers remained unconditional. It reduced Q4_K B4/B16 route costs by `5.27-11.60%` in the final controls and was retained. Q4 and Q5 share this consumer mechanism but retain separate decode/resource identities.

### Decode and layout controls

Width-16 decode, Q4-specific scale/minimum reconstruction, and the sixteen-BF16 XOR layout were retained. K64, broad swizzles, two LDS buffers, decoded-weight caching, direct-to-VGPR, split-K, GSU, Stream-K, and persistent traversal were rejected. The remaining loss is repeated packed scale/minimum decode and limited reuse, not a saturated LDS interface.

### Shared backward arithmetic controls

The grouped backward campaign tested reduced-precision accumulation as a separate kernel mechanism. Direct BF16-C reached `0.86089` NRMSE at 513 rows and was rejected for accuracy. Full-N FP32 K32/K64 slabs reached 256 VGPRs with 647/2,069 spills and 1,568/5,248 private bytes and were rejected before timing. Pair-serial slabs reached `0.01343/0.00959` NRMSE and were 35.0%/82.9% slower. Row-normalized FP16-C reached `0.00723-0.00727` NRMSE; even without its scale scan it was 33.3% slower. Q4_K therefore retains exact FP32 WMMA accumulation.

## Correctness and resources

Retained Q4_K bodies use 96 VGPR/22 SGPR/4096 B LDS for M64/N64, 173/30/4096 for M128/N64, and 216/30/8192 for row-task M128/N128. They have zero private bytes, zero spills, no scratch, calls, or dynamic stack. Validation covers Q4 nibbles, scale/minimum fields, inactive experts, malformed routes, row tails, input/weight mutation, independent BF16 references, finite outputs, and deterministic reruns.

## Evidence

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step1_q4_matrix.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step3_s1.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step4_row_tasks_mmajor.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step4_row_tasks_nmajor.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_split_tasks.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_rowtask_tail_predicate_control_false_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_rowtask_tail_predicate_control_true_25.json
```

The Q4 sparse and small-group controls that established the M64/M128 boundary are also retained:

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step3_s2.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_q4_sparse_s2_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_q4_sparse_s1_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step3_s1.json
```

The Q4_K residual is representation-level packed decode cost. No new generic row-task or LDS sweep is justified.
