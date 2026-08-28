# HIP Grouped MMQ Backward Pair IQ2_XXS Experiment

## Scope

This record covers the fused routed IQ2_XXS gate/up input-gradient kernel for DeepSeek:

```text
dX0[R,4096] = dY0[R,2048] @ W0[2048,4096]
dX1[R,4096] = dY1[R,2048] @ W1[2048,4096]
```

Aggregate routed rows are `R=12288,49152,196608`. The two packed IQ2_XXS banks are decoded directly, accumulated in one FP32 pair dataflow, and written to BF16 gradients.

## Final kernel result

Pair throughput counts both matrices as `4*R*N*K/time`. `HIP/AITER GMM` compares the packed pair with two BF16 AITER GMM calls. Values above `1.00x` favor HIP.

| Batch | Logical shape | HIP TFLOPS | HIP/AITER GMM |
| ---: | --- | ---: | ---: |
| 1 | `2 x (12288,2048,4096)` | 13.99 | 1.513x |
| 4 | `2 x (49152,2048,4096)` | 17.59 | 1.172x |
| 16 | `2 x (196608,2048,4096)` | 18.15 | 0.698x |

The B16 loss remains a packed two-weight decode and pair-accumulator cost relative to predecoded BF16 AITER.

## Kernel implementation

The retained pair uses four wave32 waves, M64/N64 at small rows, a qualified M128/N64 geometry at larger routed rows, cooperative width-16 IQ2_XXS lookup/sign/scale decode, two weight LDS tiles, swizzle4, and inactive-M consumer suppression. It preserves one FP32 accumulation and one BF16 rounding for each fused pair output.

## Optimization log

### Initial exact pair body

The generic grouped backward path used narrow N16/K16 ownership and serial rows. The exact IQ2_XXS pair body moved to M64/N64/K32 ownership, cooperative width-16 decode, two weight LDS images, and pair accumulation. Swizzle4 improved all 12 historical points by roughly `20-28%` over unswizzled staging; swizzle16 lost `24-39%` relative to swizzle4.

An early M128/N64 source form created an 8-byte private segment and was rejected before timing. A historical M192/N64 body used 32 private bytes, seven VGPR spills, and scratch instructions. Those failures do not close the later lower-state exact M128 identity, but they do close the spilling bodies.

### Unroll and inactive-M controls

Width32 decode had mixed route movement and no legal shape-only separator, so width16 was retained. Inactive-M suppression improved every measured point by `0.58-8.44%` and reduced the complete matrix geometrically by `4.08%`. N-major ownership and broad row-task traversal were rejected.

The coefficient-only campaign promoted M128/N64 only at exact aggregate rows `R=49152` and `R=196608`. The repaired lower-state body confirmed approximately `1.1678x` at B4 and `1.1644x` at B16 against the retained parent, with every learned, hash, and mandatory control above `1.05x` in the final qualification.

## Correctness and resources

### Shared backward arithmetic controls

The grouped backward campaign tested reduced-precision accumulation as a separate kernel mechanism. Direct BF16-C reached `0.86089` NRMSE at 513 rows and was rejected for accuracy. Full-N FP32 K32/K64 slabs reached 256 VGPRs with 647/2,069 spills and 1,568/5,248 private bytes and were rejected before timing. Pair-serial slabs reached `0.01343/0.00959` NRMSE and were 35.0%/82.9% slower. Row-normalized FP16-C reached `0.00723-0.00727` NRMSE; even without its scale scan it was 33.3% slower. IQ2_XXS therefore retains exact FP32 WMMA accumulation.

## Correctness and resources

The retained M64/N64 and M128/N64 bodies use 209/54/8192 B LDS and 239/54/8192 B LDS respectively, with zero private bytes, zero spills, no scratch, calls, or dynamic stack. Validation covers IQ2_XXS codebook/sign selectors, paired-output isolation, active and inactive weight mutations, malformed routes, non-aligned rows, independent BF16 reference error, finite output, and deterministic reruns.

## Evidence

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_deepseek_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_iq2xxs_swizzle4_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_iq2xxs_swizzle16_focus.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_iq2xxs_width32_focus.json
~/tmp/torch-ggml-ops/grouped-bwd-finalists/deepseek_iq2xxs_m128_confirmation_25.json
~/tmp/torch-ggml-ops/grouped-bwd-production-final-correctness.json
```

The current HIP pair kernel is closed for broad J, width, and swizzle sweeps. The next material direction is lower-state IQ2_XXS decode or prepared weight reuse.
