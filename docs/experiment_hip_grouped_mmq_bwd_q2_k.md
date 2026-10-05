# HIP Grouped MMQ Backward Q2_K Experiment

## Scope

This record covers the routed Q2_K down input-gradient kernel for DeepSeek.

Aggregate routed rows are `R=12288,49152,196608`.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | --- | --- | --- | --- |
| 1 | `(12288,2048,4096)` | 12.87 | 1.388x | `grouped_bwd_row_task_q2_k_n4096_k2048_mt128_nt64_s3_g4_abar` |
| 4 | `(49152,2048,4096)` | 21.48 | 1.347x | `grouped_bwd_row_task_q2_k_n4096_k2048_mt128_nt64_s3_g4_abar` |
| 16 | `(196608,2048,4096)` | 26.01 | 1.440x | `grouped_bwd_row_task_q2_k_n4096_k2048_mt256_nt64_s3_g4_k64` |

All three shapes are now ahead of predecoded BF16 AITER, including the smallest groups once inactive waves skip their consumer work.

## Kernel implementation

The retained Q2_K single-projection body uses an M128/N64 tile, a 32-wide contraction stage, three LDS buffers and one plain barrier per stage, with the packed two-bit segment loaded as one vector per thread and stage. Waves whose first row is past the task end skip their activation loads, matrix work and stores while still taking part in the shared decode, which is what lets one body serve every route size.

## Optimization log

### Swizzle fold

The decoded tile is written down its rows: a thread decodes sixteen consecutive output columns of one packed row, so its sixteen stores advance by the contraction stride while every lane starts at the same column. The plain chunk XOR (`chunk = (column / 4) ^ (row & 7)`) therefore gives the writer only the row bits the fragment reader also uses, and the counter run on the deployed Q2_0 pair body of the same shape reports `LDSBankConflict 58.75 %`. Folding the row bits one sixteen-row step above the low ones (`row ^ (row >> 4)`) adds the writer's row step without moving the reader's. The change is layout-only and the body is bit-identical to the deployed one (`max|base-candidate| 0.0` against the BF16 product for both).

Paired against the previous control in one process, both orders, rotating banks, and with both arms pinned by symbol:

| Route | against the previous control |
| --- | ---: |
| B1 | +5.9 % / +6.3 % |
| B4 | +7.1 % / +7.8 % |
| B16 | +7.3 % / +8.1 % |

The fold is not universal: it wins on the chunk-4 pair layouts and on the deep-contraction Q2_K single, and it loses on the Q4_K and Q5_K singles (`+0.1 % / +2.5 % / +2.9 %` and `+6.7 % / +6.2 % / +4.1 %` over the same protocol), whose decoders are expensive enough that the tile's bank pattern is not what limits them. Those two candidates are not deployed and their controls are removed.

### Staged row-task redesign

The DeepSeek body kept its M128/N64 geometry but adopted the staged skeleton: a 32-wide contraction stage, three LDS buffers, one plain barrier per stage, a vectorised two-bit packed segment per thread, clamped activation rows, and inactive-wave suppression. Suppression is worth `29%` at B1 and `4%` at B16 here, because the learned prior leaves most experts with fewer rows than one tile. It also retires the separate M64 body that used to serve the smallest groups.

The deployed body is `149` VGPR / `24` SGPR / `4` KB LDS per stage. Its bench result is `11.97/19.67/22.57` TFLOPS at B1/B4/B16 against `9.34/15.95/16.96` for the previous selection, and it now leads predecoded BF16 AITER by `34%`, `37%` and `44%`.

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

### A-fragment hoist

PC sampling of the deployed body attributed `52-55 %` of wave stalls to the stage barrier, with `s_barrier` the single most sampled instruction, while VALU ran at a few percent of peak issue. The gradient fragments are private to the wave, so their global loads do not have to follow the barrier: issuing the first projection's fragment load before `s_barrier` lets the barrier wait cover the L2 latency the multiply would otherwise stall behind. Nothing else moves, and the hoisted arm is bit-identical to its parent (`max|base-candidate| 0.0`).

Paired in one process, both orders, rotating route banks, both arms pinned by symbol:

| Route | hoisted against its parent |
| --- | ---: |
| B1 | +2.5 % / +2.4 % |
| B4 | +0.7 % / +0.6 % |
| B16 | -0.3 % / +0.4 % |

### Shared backward arithmetic controls

The grouped backward campaign tested reduced-precision accumulation as a separate kernel mechanism. Direct BF16-C reached `0.86089` NRMSE at 513 rows and was rejected for accuracy. Full-N FP32 K32/K64 slabs reached 256 VGPRs with 647/2,069 spills and 1,568/5,248 private bytes and were rejected before timing. Pair-serial slabs reached `0.01343/0.00959` NRMSE and were 35.0%/82.9% slower. Row-normalized FP16-C reached `0.00723-0.00727` NRMSE. Even without its scale scan it was 33.3% slower.

### Wide task tile

The task descriptor height sets how often the packed weights are decoded: a task decodes its own copy of the weight rows it needs, so the decode per output row is one weight matrix per descriptor, and a 256-row descriptor halves it. The wide body keeps the per-wave tile, the four-column tile count and therefore the accumulator budget of the deployed shape - eight waves of `M_TILES = 2` give the 256 rows - so the only costs are twice the shared memory per stage at the same stage count and more waves arriving at each barrier. Measured at the largest route, where a 256-row descriptor is well filled, paired in one process with both arms pinned by symbol:

| Variant | against the four-wave body at B16 |
| --- | ---: |
| K64 twin | `+3.5 % / +3.4 %` (official `24.75` to `26.01` TFLOPS). The K32 twin reaches `+1.5 %` |

At the smaller routes the same body loses to the four-wave one (`-19 % / -6 %` for Q5_K's K32 twin at B1/B4, `-13 % / -6 %` for IQ2_S, `-30 % / -25 %` for Q4_K, `-7 %` for the IQ2_XXS pair at B16), because a 58-row expert cannot fill a 256-row descriptor and the wider workgroup pays more drift per barrier than it saves. The deployment therefore takes a `rows_at_least` rule per family instead of replacing the body outright, and Q4_K and the IQ2_XXS pair keep the four-wave body everywhere.

## Resources

Retained Q2_K bodies use 102 VGPR/30 SGPR/4096 B LDS for M64/N64/U1, 146/31/4096 for M128/N64/U1, 160/31/4096 for U2, and 149/24/4096 per stage for the pre-hoist row-task M128/N64.

The hoisted body allocates 200 VGPR for the same 12288 B of shared memory, which trades resident waves for hidden gradient-load latency.

The wide B16 body allocates 240 VGPR / 24576 B LDS, twice the shared memory of the four-wave body at the same three stages.

## Evidence

```text
~/tmp/torch-ggml-ops/retune/official/GroupedBackward_deepseek.json
~/tmp/torch-ggml-ops/retune/official/GroupedBackward_deepseek_q2k.json
~/tmp/torch-ggml-ops/retune/run_final.py
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
