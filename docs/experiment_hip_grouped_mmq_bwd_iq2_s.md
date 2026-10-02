# HIP Grouped MMQ Backward IQ2_S Experiment

## Scope

This record covers the routed single-projection IQ2_S down input-gradient kernel for Qwen.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | --- | --- | --- | --- |
| 1 | `(16384,512,2048)` | 11.21 | 1.052x | `grouped_bwd_row_task_iq2_s_n2048_k512_mt128_nt64_s2` |
| 4 | `(65536,512,2048)` | 17.49 | 1.054x | `grouped_bwd_row_task_iq2_s_n2048_k512_mt128_nt64_s2` |
| 16 | `(262144,512,2048)` | 21.05 | 1.014x | `grouped_bwd_row_task_iq2_s_n2048_k512_mt128_nt64_s2` |

The single-down IQ2_S body is ahead of predecoded BF16 AITER at all three shapes; the residual is grid lookup and sign reconstruction in the packed decode.

## Kernel implementation

The retained IQ2_S single-projection bodies use an M128/N64 tile, a 32-wide contraction stage, two LDS buffers and one plain barrier per stage: four waves own 32 rows each, every thread decodes one 16-value grid segment per stage, and the accumulator budget is half the previous M128/N128 body. Activation rows are clamped so no load is predicated, and waves whose first row is past the task end skip their activation loads, matrix work and stores.

## Optimization log

### Staged row-task redesign

PC sampling of the first-generation row-task body attributed most stalls to ALU dependencies, barrier waits and memory waits, with only about four resident waves per SIMD. The staged redesign keeps the same grid decode and changes the skeleton: an M128/N64 tile with a 32-wide contraction stage, two LDS buffers, one plain barrier per stage, and clamped activation rows. The smaller column tile halves the accumulator budget and roughly doubles the resident wave count; the extra column blocks only add L2-resident activation traffic. Two buffers measure better than three for this decoder, so the deployed body keeps two.

Waves whose first row is already past the task end skip their activation loads, matrix work and stores while still taking part in the shared decode; for this decoder that is worth `19%` at B1, `9%` at B4 and `7%` at B16, so the deployed body keeps the suppression. The same switch costs `3-5%` on the Q4_K and Q5_K decoders, which therefore keep it off.

The deployed body is `155` VGPR / `26` SGPR / `4` KB LDS per stage. Its bench result is `11.21/17.49/21.05` TFLOPS at B1/B4/B16 against `8.54/13.02/16.28` for the previous selection.

### Generic-to-tiled redesign

The original eight-wave N16/K16 body was spill-free but ownership-limited. Representative task timing improved the original serial B16 point from `38.992 ms` to `32.806 ms`.

N-major ordering nearly doubled B16 latency. A fixed 1,024-program traversal was slower and spilled Q5 controls. Runtime full/tail branches created private segments and spills; split full/tail lists regressed nonuniform routes due to the second launch.

### Decode, swizzle, and ownership controls

Width-8 IQ2_S decode duplicated scale work and loader groups. M256/N64 doubled N workgroups and regressed B4/B16. The pair/down swizzle comparison showed opposite timing preferences, so the down layout remains independent. Inactive-M suppression was tested for the IQ2_S row-task body but produced mixed route movement and two regressions; it was rejected.

A learned B1 ownership screen compared the row-task body with the serial body for the exact B1 single-down geometry. Other route sizes retain their measured ownership bodies; no broad new J geometry passed the coefficient-only campaign.

### Bottleneck attribution

The residual is repeated grid lookup, sign reconstruction, shared scale extraction, `d` application, and packed LDS staging for each output tile. A decoded BF16 AITER baseline has already paid that representation cost outside timing. The kernel-level profiler and resource controls found no broad LDS-bank or spill issue that would justify another generic scheduling sweep.

### Shared backward arithmetic controls

The grouped backward campaign tested reduced-precision accumulation as a separate kernel mechanism. Direct BF16-C reached `0.86089` NRMSE at 513 rows and was rejected for accuracy. Full-N FP32 K32/K64 slabs reached 256 VGPRs with 647/2,069 spills and 1,568/5,248 private bytes and were rejected before timing. Pair-serial slabs reached `0.01343/0.00959` NRMSE and were 35.0%/82.9% slower. Row-normalized FP16-C reached `0.00723-0.00727` NRMSE; even without its scale scan it was 33.3% slower.

## Resources

Retained IQ2_S down bodies use 90 VGPR/22 SGPR/4096 B LDS for M64/N64, 161/30/4096 for M128/N64, 238/24/8192 for the retired row-task M128/N128, and 155/26/4096 per stage for the deployed row-task M128/N64.

## Evidence

```text
~/tmp/torch-ggml-ops/retune/official/GroupedBackward_qwen.json
~/tmp/torch-ggml-ops/retune/official/GroupedBackward_qwen_iq2s.json
~/tmp/torch-ggml-ops/retune/run_final.py
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step4_row_tasks_mmajor.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step4_row_tasks_nmajor.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step6_iq2.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step6_iq2_n64_reuse.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_width8.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_swizzle0.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_swizzle4.json
```

The remaining IQ2_S down loss is packed representation and decode cost. Another broad row-task, swizzle, or width sweep is closed.
