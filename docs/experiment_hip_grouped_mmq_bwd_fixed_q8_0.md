# HIP Fixed-Grouped MMQ Backward Q8_0 Experiment

## Scope

This record covers the fixed eight-group DeepSeek Q8_0 input-gradient kernel:

```text
dX[t,8,4096] = dY[t,8,1024] @ W[8,1024,4096]
```

The eight groups are fixed, not routed. The packed Q8_0 weights are decoded directly, BF16 cotangents and gradients use FP32 WMMA accumulation, and the kernel preserves the fixed group-major weight and public tensor layout.

## Final kernel result

Fixed-group throughput counts all eight matrices: `2*8*M*N*K/time`. `HIP/torch.bmm` compares the packed fixed-group kernel with BF16 `torch.bmm`. Values above `1.00x` favor HIP.

| Batch | Logical shape | HIP TFLOPS | HIP/torch.bmm |
| ---: | --- | ---: | ---: |
| 1 | `8 x (2048,1024,4096)` | 22.74 | 1.228x |
| 4 | `8 x (8192,1024,4096)` | 22.34 | 1.155x |
| 16 | `8 x (32768,1024,4096)` | 23.82 | 1.202x |

The fixed packed kernel beats the BF16 BMM baseline at all three token counts in the current matrix.

## Kernel implementation

The retained body uses exact per-group `(N,K)=(1024,4096)` geometry, four wave32 waves, M256/N64 ownership, width-16 Q8_0 decode, one unswizzled N64/K32 weight tile per fixed group, and BF16 stores. The M192/N64 geometry is a separate exact body for the 32,768-token case.

Q8_0 packed payloads and scales are reconstructed cooperatively before WMMA. No routed task descriptors, inactive-expert suppression, or fabricated route metadata are involved.

## Optimization log

### Initial fixed-group redesign

The generic baseline used narrow eight-wave N16/K16 ownership and serial row work. The first exact four-wave M256/N64/K32 body improved representative B1/B4/B16 times by `8.09x`, `8.33x`, and `8.75x` over that generic fixed kernel.

M64 and M128 alternatives were slower at B1/B4. Width32 decode, swizzle4, and M512 were rejected by timing or resource limits. The selected fixed body retains unswizzled staging because its layout and group ownership are distinct from routed Q8_0.

### Exact-token geometry

The coefficient-only campaign screened M256 and the exact M192 alternative at token rows `2048`, `8192`, and `32768`, along with decoder width, swizzle, and ownership controls. M192/N64 was retained for the 32,768-token geometry after independent correctness, resource, and timing confirmation; M256 remains the other exact fixed-group body.

The fixed path has no route imbalance or inactive expert work to remove. Its remaining comparison advantage comes from compact packed traffic and geometry, while any direct comparison with BMM must account for BMM starting from decoded BF16 weights.

### Reduced-precision controls

Direct BF16-C was rejected for `0.86089` NRMSE at 513 rows. Full-N FP32 slabs crossed the resource gate with 647/2,069 spills and 1,568/5,248 private bytes. Pair-serial K32/K64 slabs had `0.01343/0.00959` NRMSE and were 35.0%/82.9% slower. Row-normalized FP16-C remained `0.00723-0.00727` NRMSE and was 33.3% slower even without its scale scan.

Exact FP32 accumulation therefore remains part of the fixed Q8_0 kernel contract. A prepared lossless payload/scale layout is the only credible next representation mechanism.

### Shared backward arithmetic controls

The grouped backward campaign tested reduced-precision accumulation as a separate kernel mechanism. Direct BF16-C reached `0.86089` NRMSE at 513 rows and was rejected for accuracy. Full-N FP32 K32/K64 slabs reached 256 VGPRs with 647/2,069 spills and 1,568/5,248 private bytes and were rejected before timing. Pair-serial slabs reached `0.01343/0.00959` NRMSE and were 35.0%/82.9% slower. Row-normalized FP16-C reached `0.00723-0.00727` NRMSE; even without its scale scan it was 33.3% slower. Fixed Q8_0 therefore retains exact FP32 WMMA accumulation.

## Correctness and resources

Retained M256/N64 uses 209 VGPR/23 SGPR/4096 B LDS; M192/N64 uses 173/22/4096. Both have zero private bytes, zero spills, no scratch, calls, or dynamic stack. Validation covers all eight groups, token/group boundaries, independent GGUF decode, fixed-layout conversion, input and packed-weight mutation, group isolation, finite output, and deterministic reruns.

## Evidence

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_deepseek_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m256_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m128_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_generic_control_25.json
~/tmp/torch-ggml-ops/grouped-bwd-finalists/deepseek_fixed_m192_b16_confirmation_25.json
~/tmp/torch-ggml-ops/grouped-bwd-production-final-correctness.json
```

The fixed-group geometry screen also retained the generic and per-M controls that led to the M256 choice:

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_baseline_b1_b4.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m64_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m128_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m256_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m256_w32_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m256_s4_full.json
```

The fixed Q8_0 kernel is complete for the current packed representation and exact FP32 arithmetic contract.
