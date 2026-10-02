# HIP Grouped MMQ Backward Pair IQ2_S Experiment

## Scope

This record covers the fused routed IQ2_S gate/up input-gradient kernel for Qwen.

Aggregate rows are `R=16384,65536,262144`.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | --- | --- | --- | --- |
| 1 | `2 x (16384,512,2048)` | 14.51 | 1.802x | `grouped_bwd_pair_task_iq2_s_n512_k2048_mt128_nt64_s2_skip` |
| 4 | `2 x (65536,512,2048)` | 23.30 | 1.864x | `grouped_bwd_pair_task_iq2_s_n512_k2048_mt128_nt64_s2_skip` |
| 16 | `2 x (262144,512,2048)` | 26.65 | 1.705x | `grouped_bwd_pair_task_iq2_s_n512_k2048_mt128_nt64_s2_skip` |

Every shape uses the same device row-task body, so one workgroup owns one 128-row tile of one expert instead of walking a whole expert's rows. The fused pair beats two predecoded BF16 AITER GMM calls on all three shapes, by `70%` to `86%`.

The fused pair beats two predecoded BF16 AITER GMM calls on all three shapes, by `59%` to `73%`. The B16 row keeps the wider M256 body because this decoder turns the extra row reuse into throughput at that route size, where the median active expert owns more than one M256 tile.

## Kernel implementation

The deployed pair body uses a device row-task bank with an N64 x M128 tile, a 32-wide contraction stage, two projection weight tiles per stage, two LDS stages, one plain barrier per stage and inactive-wave suppression: four waves own 32 rows each, every thread decodes one 16-value grid segment per projection and stage, and activation rows are clamped so no load is predicated. Pair and single-down IQ2_S keep separate LDS swizzles; the decode and epilogue state must not be generalized from the single-down path.

The retained pair bodies use an N64 x M128 tile, a 32-wide contraction stage, two projection weight tiles per stage, two LDS stages and one plain barrier per stage: four waves own 32 rows each, every thread decodes one 16-value grid segment per projection and stage, and activation rows are clamped so no load is predicated. Pair and single-down IQ2_S keep separate LDS swizzles; the decode and epilogue state must not be generalized from the single-down path.

## Optimization log

### Staged pair redesign

The deployed pair body decoded both projections, waited on two `__syncthreads()` fences per contraction stage, predicated every activation load, and carried 219 VGPR / 54 SGPR at M128/N64, which held it to roughly six resident waves per SIMD.

The staged redesign keeps the N64/M128 tile and the pair accumulation order and changes the skeleton: two LDS stages of both projection tiles with a single plain barrier per stage, clamped activation rows, and one vectorised packed row load per thread, projection and stage. A sweep over M64/M128/M256 x two/three stages x inactive-wave suppression measured M128/N64 with two stages as the best or near-best point at every shape except B16, where an M256 three-stage body is about `2.6%` ahead; the deployed body keeps the simpler M128/N64 two-stage shape. Suppression costs `2-3%` at the small batches, where nearly every wave owns rows, so only the Q3_K and IQ2_XXS B16 rules enable it.

The staged body is `208` VGPR / `28` SGPR / `16` KB LDS against `219`/`54`/`8` KB for the deployed M128/N64 body, and it is bitwise identical to it on the same inputs. Its bench result is `13.39/21.62/24.74` TFLOPS at B1/B4/B16 against `11.23/18.69/20.59` for the previous selection. The B16 row uses an M256 three-stage body with suppression, worth `12%` there; the same wider body loses `8-14%` on the Q3_K and IQ2_XXS decoders, so only this family's large-route rule adopts it.

### Initial pair body

The tiled pair body switched to four-wave N64/K32 ownership, cooperative lookup/sign/scale decode, padded/format-specific LDS staging, and fused accumulation. It won all measured Qwen pair points against AITER.

### Bounded extraction and layout controls

Width-16 decode shares grid, sign, and scale work across natural packed groups. Width-8 duplicated scale work and loader groups and was rejected. Reducing N to N64 relieved pair register pressure and improved representative large-route points. A universal M256/N64 body added state and regressed B4/B16.

Swizzle experiments showed that pair and down consumers prefer different layouts: the pair swizzle improved pair timings by roughly `15-25%`, while the same direction regressed down by `20-35%`. This is a kernel ownership result, not a global IQ2_S policy.

### Ownership retune

The learned B1 screen exposed tall physical experts and compared serial with row-task pair ownership. The row-task body was retained for the pair B1 screen, while the final typed campaign found no alternate J or M geometry with a stable gain across the required profiles. N-major task ordering and broad persistent traversal were rejected.

The remaining theoretical ceiling is IQ2_S decode state and pair accumulator pressure.

### Shared backward arithmetic controls

The grouped backward campaign tested reduced-precision accumulation as a separate kernel mechanism. Direct BF16-C reached `0.86089` NRMSE at 513 rows and was rejected for accuracy. Full-N FP32 K32/K64 slabs reached 256 VGPRs with 647/2,069 spills and 1,568/5,248 private bytes and were rejected before timing. Pair-serial slabs reached `0.01343/0.00959` NRMSE and were 35.0%/82.9% slower. Row-normalized FP16-C reached `0.00723-0.00727` NRMSE; even without its scale scan it was 33.3% slower. The fused IQ2_S pair therefore retains one FP32 accumulation and one BF16 rounding per output.

### ### Balanced row-task ownership

Route banks drawn from the fitted prior carry a large `max/mean` expert row spread (`13-27x` on the measured vectors), so the serial bodies left column tiles of small experts idle behind the tallest expert. Isolating that effect on the deployed serial body: uniform routes of the same aggregate rows ran `2.80/2.77/2.60 ms` at B1 against `5.56-3.68 ms` for the learned bank, i.e. the skew alone cost up to `45%` at B1 and `5-10%` at B4.

The deployed body therefore consumes a device row-task bank built once per route by `grouped_row_task_setup`: one task per 128-row tile of each active expert, so small experts keep a single task and tall experts split into many, and every workgroup runs exactly one tile. Measured against the serial bodies on the same prepared inputs (same packed banks, same bf16 gradients, same learned routes), kernel-only gains were `1.027x/1.036x/1.028x` at B1/B4/B16 for the harness geometry, and the official protocol moved `13.39/21.62/24.74` to `14.51/23.30/26.65` TFLOPS. Inactive-wave suppression is part of the task body: the last task of each expert is partial, so waves past the task end would otherwise decode and multiply empty rows.

Alternatives measured and rejected for this decoder: the wide-M task bodies (`m4n4` tasks were `0.62-0.72x` of the deployed serial body because a 128-row task cannot fill an M256 body), a third LDS stage, and the projection-split decode used by the N32 sweeps. Splitting the reduction or the projections into separate blocks was not pursued for this operator: the fused output is `rows x 2048` bf16, so a global f32 partial round trip costs more than the entire multiply at these shapes.

## Resources

The deployed row-task pair body uses 208 VGPR / 28 SGPR / 16384 B LDS and runs one 128-row task per workgroup; the retired M128/N64 serial body used the same resources over a serial row walk, and the M256 three-stage body used 256 VGPR with 44 spills and 24576 B LDS.

The deployed staged pair body uses 208 VGPR / 28 SGPR / 16384 B LDS; the retired M128/N64 body used 219 / 54 / 8192.

## Evidence

```text
~/tmp/torch-ggml-ops/retune_pairs/official/pair_qwen.json
~/tmp/torch-ggml-ops/retune_pairs/official/pair_qwen_final.json
~/tmp/torch-ggml-ops/retune_pairs/sweep.py
~/tmp/torch-ggml-ops/retune_pairs/compare.py
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_pair_n64.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_width8.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_swizzle4.json
~/tmp/torch-ggml-ops/grouped-bwd-production-final-correctness.json
```

The pair-specific source artifacts include the exact N64 comparison and the rejected width/swizzle controls:

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_pair_n64.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_width8.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_swizzle0.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_swizzle4.json
```

The exact FP32 pair contract and IQ2_S pair layout are complete for the current packed kernel.
