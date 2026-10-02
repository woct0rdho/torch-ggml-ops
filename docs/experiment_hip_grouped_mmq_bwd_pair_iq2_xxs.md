# HIP Grouped MMQ Backward Pair IQ2_XXS Experiment

## Scope

This record covers the fused routed IQ2_XXS gate/up input-gradient kernel for DeepSeek.

Aggregate routed rows are `R=12288,49152,196608`.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | --- | --- | --- | --- |
| 1 | `2 x (12288,2048,4096)` | 10.59 | 1.261x | `grouped_bwd_pair_iq2_xxs_n2048_k4096_mt64_nt64` * |
| 4 | `2 x (49152,2048,4096)` | 15.71 | 1.154x | `grouped_bwd_tuned_pair_iq2_xxs_n2048_k4096_mt128_nt64` |
| 16 | `2 x (196608,2048,4096)` | 17.24 | 1.156x | `grouped_bwd_tuned_pair_iq2_xxs_n2048_k4096_mt128_nt64` |

Rows marked `*` look prior-sensitive: the current learned-route result is more than 10% below the same body measured on a single uniform partition, so the deployed body may need retuning for the current route distribution.

The B16 residual is a packed two-weight decode and pair-accumulator cost; the current result still beats predecoded BF16 AITER.

## Kernel implementation

The retained pair uses four wave32 waves, M64/N64 at small rows, a qualified M128/N64 geometry at larger routed rows, cooperative width-16 IQ2_XXS lookup/sign/scale decode, two weight LDS tiles, swizzle4, and inactive-M consumer suppression.

## Optimization log

### Initial exact pair body

The generic grouped backward path used narrow N16/K16 ownership and serial rows. The exact IQ2_XXS pair body moved to M64/N64/K32 ownership, cooperative width-16 decode, two weight LDS images, and pair accumulation. Swizzle4 improved all 12 historical points by roughly `20-28%` over unswizzled staging; swizzle16 lost `24-39%` relative to swizzle4.

An early M128/N64 source form created an 8-byte private segment and was rejected before timing. A historical M192/N64 body used 32 private bytes, seven VGPR spills, and scratch instructions. Those failures do not close the later lower-state exact M128 identity, but they do close the spilling bodies.

### Unroll and inactive-M controls

Width32 decode had mixed route movement and no legal shape-only separator, so width16 was retained. Inactive-M suppression improved every measured point by `0.58-8.44%` and reduced the complete matrix geometrically by `4.08%`. N-major ownership and broad row-task traversal were rejected.

The coefficient-only campaign promoted M128/N64 only at exact aggregate rows `R=49152` and `R=196608`. The repaired lower-state body confirmed approximately `1.1678x` at B4 and `1.1644x` at B16 against the retained parent, with every learned, hash, and mandatory control above `1.05x` in the final qualification.

## Resources

The retained M64/N64 and M128/N64 bodies use 209/54/8192 B LDS and 239/54/8192 B LDS respectively.

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
