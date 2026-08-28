# HIP Grouped MMQ Backward Pair Q3_K Experiment

## Scope

This record covers the fused routed Q3_K gate/up input-gradient kernel for Qwen on gfx1151:

```text
dX0[R,2048] = dY0[R,512] @ W0[512,2048]
dX1[R,2048] = dY1[R,512] @ W1[512,2048]
```

The two projections use one fused kernel, one FP32 accumulator contract, and two BF16 gradient outputs. Packed Q3_K weights remain in forward layout. Aggregate routed rows are `R=16384,65536,262144`.

## Final kernel result

Pair throughput counts both input-gradient matrices: `4*R*N*K/time`. `HIP/AITER GMM` is the packed HIP throughput ratio against two BF16 AITER GMM calls. Values above `1.00x` favor HIP.

| Batch | Logical shape | HIP TFLOPS | HIP/AITER GMM |
| ---: | --- | ---: | ---: |
| 1 | `2 x (16384,512,2048)` | 19.09 | 2.392x |
| 4 | `2 x (65536,512,2048)` | 26.09 | 1.922x |
| 16 | `2 x (262144,512,2048)` | 24.54 | 1.501x |

The rows are the final uniform-route packed-kernel matrix.

## Kernel implementation

The retained pair body uses four wave32 waves, exact `(N,K)=(512,2048)` geometry, M64/M128 ownership variants, separate Q3_K decoded-weight LDS tiles, cooperative width-16 decode, and one fused FP32 accumulation path. It rounds each output to BF16 after accumulation and preserves projection isolation.

Device-resident route indices and offsets identify active experts. Invalid routes, inactive experts, and partial row tiles remain inert without host descriptor construction. Cotangents are consumed directly in BF16 and are not quantized.

## Optimization log

### Generic-to-tiled redesign

The original grouped body used eight waves, N16/K16 ownership, scalar decode, and serial 128-row chunks. The first tiled Q3_K pair introduced four-wave N64/K32 ownership, cooperative payload/scale extraction, padded LDS rows, and one pair accumulation. Representative B4/B16 points improved by 8-14x over the generic baseline and beat AITER by about 2.1-2.9x.

Universal M128 ownership was rejected because uniform 64-row groups became half-empty bounded tiles. M64 and M128 pair bodies were kept as separate exact geometry controls.

### Row tasks and N geometry

Large Q3_K pair controls use M-major row-task ownership where measured. N64 reduced pair accumulator pressure and improved representative points by `1-7%` over larger N ownership. N-major ordering nearly doubled B16 latency and was rejected.

The learned-route B1 ownership retune compared serial and row-task bodies. The fitted prior gain was `1.1074x`, but captured-route gain was only `1.0179x`; B1 therefore retains serial ownership while the existing row-task J64 body remains the measured large-route choice. The typed campaign found no alternate J geometry that passed the full route controls.

### Decode and layout controls

Width-16 decode shares packed payload and scale work. Width-8 duplicated metadata and loader work and was rejected. Q3_K pair uses padded LDS rows; pair and down layouts remain separate because their reuse and accumulator lifetimes differ. Cross-iteration packed prefetch, universal swizzle, decoded-weight caching, split-K, persistent workgroups, and compiler-managed local arrays were rejected.

### Cross-family accumulation controls

The grouped backward campaign kept exact FP32 accumulation as the kernel contract. The reduced-precision controls were:

| Candidate | Resources | Result | Decision |
| --- | --- | --- | --- |
| Direct paired BF16-C | 122 VGPR, 31 SGPR, 4096 B LDS | `0.86089` NRMSE at 513 rows | reject for accuracy |
| Full-N FP32 slabs into BF16 | 256 VGPR, 647/2069 spills | not timed | reject by resource gate |
| Pair-serial K32/K64 slabs | 215/212 VGPR | `0.01343/0.00959` NRMSE; 35.0%/82.9% slower | reject |
| Row-normalized paired FP16-C | 156 VGPR, 5120 B LDS | `0.00723-0.00727` NRMSE; no-scan floor 33.3% slower | reject |

The Q3 pair and every other grouped backward family therefore retain FP32 WMMA accumulation, one fused pair rounding contract where applicable, and no approximate accumulator shortcut.

### Shared backward arithmetic controls

The grouped backward campaign tested reduced-precision accumulation as a separate kernel mechanism. Direct BF16-C reached `0.86089` NRMSE at 513 rows and was rejected for accuracy. Full-N FP32 K32/K64 slabs reached 256 VGPRs with 647/2,069 spills and 1,568/5,248 private bytes and were rejected before timing. Pair-serial slabs reached `0.01343/0.00959` NRMSE and were 35.0%/82.9% slower. Row-normalized FP16-C reached `0.00723-0.00727` NRMSE; even without its scale scan it was 33.3% slower. The fused Q3_K pair therefore retains one FP32 accumulation and one BF16 rounding per output.

## Correctness and resources

The retained Q3 pair bodies use 183 VGPR/26 SGPR/10240 B LDS for M64/N64 and 206 VGPR/26 SGPR/10240 B LDS for M128/N64. They have zero private bytes, zero spills, no scratch, calls, or dynamic stack. Validation covers paired-output isolation, active/inactive weight mutation, input-gradient mutation, malformed routes, non-aligned tails, independent BF16 reference error, finite output, and deterministic reruns.

## Evidence

```text
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
