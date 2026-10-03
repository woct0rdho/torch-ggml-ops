# HIP Grouped MMQ Backward Pair Q3_K Experiment

## Scope

This record covers the fused routed Q3_K gate/up input-gradient kernel for Qwen on gfx1151.

Aggregate routed rows are `R=16384,65536,262144`.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | --- | --- | --- | --- |
| 1 | `2 x (16384,512,2048)` | 15.68 | 1.895x | `grouped_bwd_pair_task_q3_k_n512_k2048_mt128_nt64_s2_skip` |
| 4 | `2 x (65536,512,2048)` | 25.08 | 2.018x | `grouped_bwd_pair_task_q3_k_n512_k2048_mt128_nt64_s2_skip` |
| 16 | `2 x (262144,512,2048)` | 29.48 | 1.882x | `grouped_bwd_pair_task_q3_k_n512_k2048_mt128_nt64_s2_skip` |

Every shape uses the same device row-task body, so one workgroup owns one 128-row tile of one expert instead of walking a whole expert's rows. The pair body beats the AITER GMM baseline at all three shapes, by `88%` to `102%`.

## Kernel implementation

The retained pair body consumes a device row-task bank with an N64 x M128 tile, a 32-wide contraction stage, two projection weight tiles per stage, two LDS stages and one plain barrier per stage, with vectorised payload and mask loads feeding the Q3_K decode. Four waves own 32 rows each, activation rows are clamped so no load is predicated, and waves whose first row is past the task end skip their activation loads, matrix work and stores while still taking part in the shared decode.

The retained pair bodies use an N64 x M128 tile, a 32-wide contraction stage, two projection weight tiles per stage, two LDS stages and one plain barrier per stage, with vectorised payload and mask loads feeding the Q3_K decode. Four waves own 32 rows each, activation rows are clamped so no load is predicated, and waves whose first row is past the task end skip their activation loads, matrix work and stores while still taking part in the shared decode.

Device-resident route indices and offsets identify active experts. Invalid routes, inactive experts, and partial row tiles remain inert without host descriptor construction.

## Optimization log

### Staged pair redesign

The deployed pair body used the generic per-value Q3_K decode, two `__syncthreads()` fences per contraction stage, predicated activation loads, and 206 VGPR / 26 SGPR at M128/N64.

The staged redesign keeps the exact `(N,K)=(512,2048)` geometry, the padded LDS rows and the single fused FP32 accumulation, and changes the skeleton: two projection tiles per stage, two LDS stages, one plain barrier per stage, clamped activation rows, one vectorised payload/mask load per thread and projection, and inactive-wave suppression. The vectorised decode alone is worth `22%` at B4 over the same body with the generic decoder, so the payload and mask reads are a first-order cost for this format.

A sweep over M64/M128/M256 x two/three stages x suppression measured M128/N64 with two stages and suppression best at all three shapes. The staged body is `233` VGPR / `26` SGPR / `16` KB LDS and is bitwise identical to the deployed M128/N64 body. Its bench result is `14.40/22.67/24.79` TFLOPS at B1/B4/B16 against `12.37/19.12/20.98` for the previous selection. The wider M256 body loses `14%` at B16 for this decoder, so every shape keeps the M128/N64 two-stage body.

### Generic-to-tiled redesign

The original grouped body used eight waves, N16/K16 ownership, scalar decode, and serial 128-row chunks. The first tiled Q3_K pair introduced four-wave N64/K32 ownership, cooperative payload/scale extraction, padded LDS rows, and one pair accumulation. Representative B4/B16 points improved by 8-14x over the generic baseline and beat AITER by about 2.1-2.9x.

Universal M128 ownership was rejected because uniform 64-row groups became half-empty bounded tiles. M64 and M128 pair bodies were kept as separate exact geometry controls.

### Row tasks and N geometry

Large Q3_K pair controls use M-major row-task ownership where measured. N64 reduced pair accumulator pressure and improved representative points by `1-7%` over larger N ownership. N-major ordering nearly doubled B16 latency and was rejected.

The learned-route B1 ownership retune compared serial and row-task bodies. The fitted prior gain was `1.1074x`, but captured-route gain was only `1.0179x`; B1 therefore retains serial ownership while the M128 body serves the larger routed rows in the current deployment. The typed campaign found no alternate J geometry that passed the full route controls.

### Decode and layout controls

Width-16 decode shares packed payload and scale work. Width-8 duplicated metadata and loader work and was rejected. Q3_K pair uses padded LDS rows; pair and down layouts remain separate because their reuse and accumulator lifetimes differ. Cross-iteration packed prefetch, universal swizzle, decoded-weight caching, split-K, persistent workgroups, and compiler-managed local arrays were rejected.

### Cross-family accumulation controls

The reduced-precision controls were:

| Candidate | Resources | Result | Decision |
| --- | --- | --- | --- |
| Direct paired BF16-C | 122 VGPR, 31 SGPR, 4096 B LDS | `0.86089` NRMSE at 513 rows | reject for accuracy |
| Full-N FP32 slabs into BF16 | 256 VGPR, 647/2069 spills | not timed | reject by resource gate |
| Pair-serial K32/K64 slabs | 215/212 VGPR | `0.01343/0.00959` NRMSE; 35.0%/82.9% slower | reject |
| Row-normalized paired FP16-C | 156 VGPR, 5120 B LDS | `0.00723-0.00727` NRMSE; no-scan floor 33.3% slower | reject |

### Shared backward arithmetic controls

The grouped backward campaign tested reduced-precision accumulation as a separate kernel mechanism. Direct BF16-C reached `0.86089` NRMSE at 513 rows and was rejected for accuracy. Full-N FP32 K32/K64 slabs reached 256 VGPRs with 647/2,069 spills and 1,568/5,248 private bytes and were rejected before timing. Pair-serial slabs reached `0.01343/0.00959` NRMSE and were 35.0%/82.9% slower. Row-normalized FP16-C reached `0.00723-0.00727` NRMSE; even without its scale scan it was 33.3% slower. The fused Q3_K pair therefore retains one FP32 accumulation and one BF16 rounding per output.

### Balanced row-task ownership

Learned route banks carry a large `max/mean` expert row spread, so the serial pair bodies kept column tiles of small experts idle while tall experts still walked rows. The deployed body now consumes a device row-task bank built once per route by `grouped_row_task_setup`: one task per 128-row tile of each active expert, so a small expert keeps one task and a tall expert splits into many, and each workgroup runs exactly one tile. Measured against the serial M128/N64 body on identical prepared inputs, kernel-only gains were `1.066x/1.074x/1.126x` at B1/B4/B16, and the official protocol moved `14.40/22.67/24.79` to `15.68/25.08/29.48` TFLOPS. The task body carries inactive-wave suppression because the last task of every expert is partial. Wide-M task bodies were rejected (`m4n4` tasks reached only `0.46-0.54x` of the serial body since a 128-row task cannot fill an M256 body), as were a third LDS stage and the projection-split decode. Reduction and projection splits were not pursued: the fused output is `rows x 512`, so a global f32 partial round trip costs more than the whole multiply.

### Occupancy and stage sweep

A launch-bound and stage-count sweep over the retired staged serial pair bodies (`__launch_bounds__` second argument `2/3/4`, two, three and four LDS stages, and the inactive-wave suppression flag for the bodies that do not deploy it) changed no shape by more than measurement noise, and three or four stages lost `5-15%` on the small-route shapes through the larger LDS footprint. The two-stage, two-wave-per-SIMD geometry is retained. Evidence: `~/tmp/torch-ggml-ops/retune_pairs/run_pair_v2.py`.

## Resources

The deployed row-task pair body uses 233 VGPR / 26 SGPR / 20480 B LDS and runs one 128-row task per workgroup.

The deployed staged pair body uses 233 VGPR / 26 SGPR / 16384 B LDS; the retired M128/N64 body used 206 / 26 / 10240.

## Evidence

```text
~/tmp/torch-ggml-ops/retune_pairs/official/pair_qwen.json
~/tmp/torch-ggml-ops/retune_pairs/official/pair_qwen_final.json
~/tmp/torch-ggml-ops/retune_pairs/sweep.py
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step2_q3_pair_matrix.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_q3_pair_n64.json
~/tmp/torch-ggml-ops/grouped-bwd-production-final-correctness.json
~/tmp/torch-ggml-ops/grouped-bwd-production-final-resources.json
```

The shared arithmetic rejection artifacts are preserved with this pair record because the tests were run as one grouped-backward generator campaign:

```text
~/tmp/torch-ggml-ops/grouped_bwd_bf16_c_control_accuracy.json
~/tmp/torch-ggml-ops/grouped_bwd_bf16_c_candidate_accuracy.json
~/tmp/torch-ggml-ops/grouped_bwd_bf16_c_k32_9.json
~/tmp/torch-ggml-ops/grouped_bwd_bf16_c_k64_9.json
~/tmp/torch-ggml-ops/grouped_bwd_fp16_c_scaled_9.json
~/tmp/torch-ggml-ops/grouped_bwd_fp16_c_no_scale_floor_9.json
```

Q3_K pair geometry, ownership, padded LDS, and exact FP32 accumulation are closed for the current packed representation. Further work needs a lower-state Q3 decoder or explicit prepared-weight reuse.
