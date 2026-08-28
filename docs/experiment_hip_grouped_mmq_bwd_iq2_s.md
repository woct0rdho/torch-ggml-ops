# HIP Grouped MMQ Backward IQ2_S Experiment

## Scope

This record covers the routed single-projection IQ2_S down input-gradient kernel for Qwen:

```text
dX[R,512] = dY[R,2048] @ W[2048,512]
```

The packed IQ2_S weights are decoded directly in the kernel. Cotangents and gradients are BF16, WMMA accumulation is FP32, and aggregate rows are `R=16384,65536,262144`.

## Final kernel result

`HIP TFLOPS` is `2*R*N*K/time`. `HIP/AITER GMM` compares the packed gradient kernel with the exact BF16 AITER GMM baseline. Values above `1.00x` favor HIP.

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM |
| ---: | ---: | ---: | ---: |
| 1 | `(16384,512,2048)` | 9.12 | 0.824x |
| 4 | `(65536,512,2048)` | 13.95 | 0.738x |
| 16 | `(262144,512,2048)` | 16.55 | 0.742x |

The single-down IQ2_S body remains slower than predecoded BF16 AITER on all three final shapes.

## Kernel implementation

The retained body uses four wave32 waves, M64/N64 for small groups, M128/N64 for intermediate ownership, and M128/N128 device row-task geometry for large groups. It uses cooperative width-16 codebook/sign/scale decode and a sixteen-BF16 XOR LDS layout.

The row-task path suppresses inactive consumer M minitiles only where measured. The paired IQ2_S kernel uses a different four-BF16 layout and separate accumulation semantics.

## Optimization log

### Generic-to-tiled redesign

The original eight-wave N16/K16 body was spill-free but ownership-limited. The tiled IQ2_S decoder introduced exact `(N,K)=(2048,512)` geometry, four-wave WMMA ownership, cooperative codebook reconstruction, and device-built M-major 128-row tasks. Representative task timing improved the original serial B16 point from `38.992 ms` to `32.806 ms`.

N-major ordering nearly doubled B16 latency. A fixed 1,024-program traversal was slower and spilled Q5 controls. Runtime full/tail branches created private segments and spills; split full/tail lists regressed nonuniform routes due to the second launch.

### Decode, swizzle, and ownership controls

Width-8 IQ2_S decode duplicated scale work and loader groups. M256/N64 doubled N workgroups and regressed B4/B16. The pair/down swizzle comparison showed opposite timing preferences, so the down layout remains independent. Inactive-M suppression was tested for the IQ2_S row-task body but produced mixed route movement and two regressions; it was rejected.

A learned B1 ownership screen promoted the existing row-task body only for the exact B1 single-down geometry. Other route sizes retain their measured ownership bodies; no broad new J geometry passed the coefficient-only campaign.

### Bottleneck attribution

The residual is repeated grid lookup, sign reconstruction, shared scale extraction, `d` application, and packed LDS staging for each output tile. A decoded BF16 AITER baseline has already paid that representation cost outside timing. The kernel-level profiler and resource controls found no broad LDS-bank or spill issue that would justify another generic scheduling sweep.

### Shared backward arithmetic controls

The grouped backward campaign tested reduced-precision accumulation as a separate kernel mechanism. Direct BF16-C reached `0.86089` NRMSE at 513 rows and was rejected for accuracy. Full-N FP32 K32/K64 slabs reached 256 VGPRs with 647/2,069 spills and 1,568/5,248 private bytes and were rejected before timing. Pair-serial slabs reached `0.01343/0.00959` NRMSE and were 35.0%/82.9% slower. Row-normalized FP16-C reached `0.00723-0.00727` NRMSE; even without its scale scan it was 33.3% slower. All grouped bodies therefore retain FP32 WMMA accumulation.

## Correctness and resources

Retained IQ2_S down bodies use 90 VGPR/22 SGPR/4096 B LDS for M64/N64, 161/30/4096 for M128/N64, and 238/24/8192 for row-task M128/N128. All have zero private bytes, zero spills, no scratch, calls, or dynamic stack. Validation covers codebook/sign/scale decode, inactive experts, malformed offsets, non-aligned rows, input/weight mutation, independent BF16 references, finite outputs, and deterministic reruns.

## Evidence

```text
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
