# HIP Grouped MMQ Backward Q4_K Experiment

## Scope

This record covers the routed Q4_K down input-gradient kernel for Qwen.

Aggregate rows are `R=16384,65536,262144`.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | --- | --- | --- | --- |
| 1 | `(16384,512,2048)` | 9.63 | 0.916x | `grouped_bwd_row_task_q4_k_n2048_k512_mt128_nt64_s3` |
| 4 | `(65536,512,2048)` | 16.60 | 0.999x | `grouped_bwd_row_task_q4_k_n2048_k512_mt128_nt64_s3` |
| 16 | `(262144,512,2048)` | 20.94 | 1.005x | `grouped_bwd_row_task_q4_k_n2048_k512_mt128_nt64_s3` |

The deficit at the larger shapes is closed: the routed body is level with predecoded BF16 AITER at B4 and ahead at B16. AITER starts from predecoded BF16 weights.

## Kernel implementation

The retained Q4_K bodies use an M128/N64 tile, a 32-wide contraction stage, three LDS buffers and one plain barrier per stage: four waves own 32 rows each, every thread decodes one 16-value packed segment per stage, and the accumulator budget is half the previous M128/N128 body. Activation rows are clamped so no load is predicated.

## Optimization log

### Staged row-task redesign

The first-generation row-task body spent most of its time waiting: PC sampling attributed `36%` of stalls to ALU dependencies, `19%` to barriers, and `18%` to memory waits, and the body was limited to about four resident waves per SIMD by its 128 SGPR/216 VGPR footprint. The staged redesign attacks every component at once.

The measured component costs, from an ablation of the deployed body at B16 (decode removed `-20%`, column-fragment LDS reads removed `-38%`, activation reads removed `-27%`, barriers removed `-25%`, matrix work removed `-8%`), drove three changes: a single plain `s_barrier` per stage instead of two `__syncthreads()` fences, double or triple LDS buffering so the decode of stage `i+1` overlaps the matrix work of stage `i`, and clamped activation rows so the boundary no longer costs a predicated load per element.

Halving the column tile from N128 to N64 halves the accumulator budget. The aspect-ratio sweep at B16 measured `20.61` TF for N64/M128 against `15.13` for N128/M128, `12.84` for N64/M64 and `17.34` for N64/M256, and the stage sweep measured `20.96/21.40` TF for two/three buffers. Removing the L1-invalidating fence is worth about `2%`, linear layouts about `5%`, and an explicit column-fragment prefetch nothing.

The deployed body is `150` VGPR / `24` SGPR / `4` KB LDS per stage against `216`/`128`/`8` KB, which raises the resident wave count per SIMD from about four to nine. Its bench result is `9.63/16.60/20.94` TFLOPS at B1/B4/B16 against `7.65/13.02/16.23` for the previous selection.

### Initial tiled kernel

The generic grouped body used eight waves, N16/K16 ownership, scalar Q4 decode, and serial 128-row chunks. The first tiled Q4_K body established M128/N128/K32 ownership, width-16 decode, and sixteen-BF16 XOR LDS. It improved representative B4/B16 controls by 5-8x over the generic baseline.

M64/N64 bodies were added for small groups. Universal M128 ownership was rejected because uniform 64-row groups became half-empty. M-major row tasks were retained for large routes. N-major ordering nearly doubled B16 latency.

### Row-task and tail controls

The row-task body improved the representative Q4_K B16 uniform point from `40.270 ms` to `35.821 ms`. Fixed 1,024-program traversal, runtime full/tail branching, and split full/tail task lists were rejected for timing or resources. The task setup is about `0.004 ms`, so it is not the residual bottleneck.

Inactive-M suppression was applied around cotangent loads, WMMA, and stores while packed decode and barriers remained unconditional. It reduced Q4_K B4/B16 route costs by `5.27-11.60%` in the final controls and was retained. Q4 and Q5 share this consumer mechanism but retain separate decode/resource identities.

### Decode and layout controls

Width-16 decode, Q4-specific scale/minimum reconstruction, and the sixteen-BF16 XOR layout were retained. K64, broad swizzles, two LDS buffers, decoded-weight caching, direct-to-VGPR, split-K, GSU, Stream-K, and persistent traversal were rejected. The remaining loss is repeated packed scale/minimum decode and limited reuse, not a saturated LDS interface.

### Shared backward arithmetic controls

The grouped backward campaign tested reduced-precision accumulation as a separate kernel mechanism. Direct BF16-C reached `0.86089` NRMSE at 513 rows and was rejected for accuracy. Full-N FP32 K32/K64 slabs reached 256 VGPRs with 647/2,069 spills and 1,568/5,248 private bytes and were rejected before timing. Pair-serial slabs reached `0.01343/0.00959` NRMSE and were 35.0%/82.9% slower. Row-normalized FP16-C reached `0.00723-0.00727` NRMSE. Even without its scale scan it was 33.3% slower.

## Resources

The deployed Q4_K row-task body uses 150 VGPR / 24 SGPR / 4096 B LDS per stage. The retired M128/N128 row-task body used 216 / 128 / 8192.

## Evidence

```text
~/tmp/torch-ggml-ops/retune/official/GroupedBackward_qwen.json
~/tmp/torch-ggml-ops/retune/run_final.py
~/tmp/torch-ggml-ops/retune/run_ablation.py
~/tmp/torch-ggml-ops/retune/run_flex.py
~/tmp/torch-ggml-ops/retune/run_v6.py
~/tmp/torch-ggml-ops/retune/prof/pcs
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
