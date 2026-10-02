# HIP Grouped MMQ Backward Q5_K Experiment

## Scope

This record covers the routed Q5_K down input-gradient kernel for Qwen.

Aggregate rows are `R=16384,65536,262144`.

## Final kernel result

`HIP/AITER GMM` is the packed HIP throughput ratio against BF16 AITER GMM.

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | --- | --- | --- | --- |
| 1 | `(16384,512,2048)` | 10.22 | 0.934x | `grouped_bwd_row_task_q5_k_n2048_k512_mt128_nt64_s2` |
| 4 | `(65536,512,2048)` | 16.36 | 0.994x | `grouped_bwd_row_task_q5_k_n2048_k512_mt128_nt64_s2` |
| 16 | `(262144,512,2048)` | 20.73 | 1.001x | `grouped_bwd_row_task_q5_k_n2048_k512_mt128_nt64_s2` |

The remaining loss is Q5 high-bit reconstruction and packed metadata cost relative to predecoded BF16 weights, and it is now visible only at B1.

## Kernel implementation

The retained Q5_K bodies use an M128/N64 tile, a 32-wide contraction stage, two LDS buffers and one plain barrier per stage: four waves own 32 rows each, every thread decodes one 16-value low/high segment per stage, and the accumulator budget is half the previous M128/N128 body. Activation rows are clamped so no load is predicated.

## Optimization log

### Staged row-task redesign

PC sampling of the first-generation row-task body attributed most stalls to ALU dependencies, barrier waits and memory waits, with only about four resident waves per SIMD. The staged redesign keeps the same decode primitives and changes the skeleton: an M128/N64 tile with a 32-wide contraction stage, two LDS buffers, one plain barrier per stage, and clamped activation rows. The smaller column tile halves the accumulator budget and roughly doubles the resident wave count; the extra column blocks only add L2-resident activation traffic. Two buffers measure better than three for this decoder, so the deployed body keeps two.

The deployed body is `147` VGPR / `24` SGPR / `4` KB LDS per stage. Its bench result is `10.22/16.36/20.73` TFLOPS at B1/B4/B16 against `8.97/13.95/16.80` for the previous selection.

### Tiled body and task ownership

The original generic grouped kernel used eight waves, N16/K16 ownership, scalar decode, and serial row chunks. The Q5 body was rebuilt on the four-wave Q4 framework and specialized for low/high payload reconstruction. Device-built M-major row tasks improved the representative B16 uniform point from `37.421 ms` to `34.074 ms`.

N-major order nearly doubled B16 latency. Fixed 1,024-program traversal spilled Q5 state; runtime full/tail branching and split task lists regressed nonuniform routes. Row-task setup is approximately `0.004 ms` and is not the residual cost.

### Extraction and swizzle controls

Width-16 low/high decode and bounded packed prefetch improved the initial port by `12-24%`. A universal swizzle8 policy regressed M64/B1 by `5.6-22.5%`; sequential row-task controls improved B4/B16 by `4.2-6.4%` and `5.2-6.6%`, so swizzle8 remains row-task-specific.

Inactive-M suppression was retained for Q5 row tasks. The row-task body fell from 256 to 233 VGPRs before final suppression and remains near the resource warning boundary. M256/N64, width-8 decode, broad J changes, two-LDS caches, split-K, Stream-K, and persistent traversal were rejected.

### Shared backward arithmetic controls

The grouped backward campaign tested reduced-precision accumulation as a separate kernel mechanism. Direct BF16-C reached `0.86089` NRMSE at 513 rows and was rejected for accuracy. Full-N FP32 K32/K64 slabs reached 256 VGPRs with 647/2,069 spills and 1,568/5,248 private bytes and were rejected before timing. Pair-serial slabs reached `0.01343/0.00959` NRMSE and were 35.0%/82.9% slower. Row-normalized FP16-C reached `0.00723-0.00727` NRMSE; even without its scale scan it was 33.3% slower.

## Resources

Retained Q5_K bodies use 115 VGPR/22 SGPR/4096 B LDS for M64/N64, 234/26/8192 for the retired row-task M128/N128, and 147/24/4096 per stage for the deployed row-task M128/N64.

## Evidence

```text
~/tmp/torch-ggml-ops/retune/official/GroupedBackward_qwen.json
~/tmp/torch-ggml-ops/retune/run_final.py
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step5_q5_prefetch.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_q5_sparse_s2_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_q5_swizzle8_rowtask_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_q5_swizzle4_rowtask_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_rowtask_tail_predicate_control_true_25.json
```

The remaining Q5_K loss is packed high-bit decode and accumulator/resource pressure. Further generic geometry work is closed.
