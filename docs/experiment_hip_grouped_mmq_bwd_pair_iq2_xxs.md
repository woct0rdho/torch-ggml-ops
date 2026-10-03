# HIP Grouped MMQ Backward Pair IQ2_XXS Experiment

## Scope

This record covers the fused routed IQ2_XXS gate/up input-gradient kernel for DeepSeek.

Aggregate routed rows are `R=12288,49152,196608`.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | --- | --- | --- | --- |
| 1 | `2 x (12288,2048,4096)` | 12.01 | 1.442x | `grouped_bwd_pair_task_iq2_xxs_n2048_k4096_mt128_nt64_s2_skip` |
| 4 | `2 x (49152,2048,4096)` | 21.34 | 1.536x | `grouped_bwd_pair_task_iq2_xxs_n2048_k4096_mt128_nt64_s2_skip` |
| 16 | `2 x (196608,2048,4096)` | 25.56 | 1.705x | `grouped_bwd_pair_task_iq2_xxs_n2048_k4096_mt128_nt64_s2_skip` |

Every shape uses the same device row-task body, so one workgroup owns one 128-row tile of one expert instead of walking a whole expert's rows, and partial tasks carry suppression. All three shapes beat two predecoded BF16 AITER GMM calls.

## Kernel implementation

The retained pair body consumes a device row-task bank with an N64 x M128 tile, a 32-wide contraction stage, two projection weight tiles per stage, two LDS stages and one plain barrier per stage. Four waves own 32 rows each, every thread decodes one 16-value lookup/sign/scale segment per projection and stage from one adjacent-code-word read, and activation rows are clamped so no load is predicated.

The retained pair body uses an N64 x M128 tile, a 32-wide contraction stage, two projection weight tiles per stage, two LDS stages and one plain barrier per stage. Four waves own 32 rows each, every thread decodes one 16-value lookup/sign/scale segment per projection and stage from one adjacent-code-word read, and activation rows are clamped so no load is predicated.

## Optimization log

### Staged pair redesign

PC sampling of the deployed M128/N64 pair at B16 attributed `29%` of stalls to barrier waits, `26%` to ALU dependencies and `7%` to memory waits, with the barrier instruction itself the single most sampled PC. The body carried 239 VGPR / 54 SGPR, decoded the two packed rows with the generic per-value path, and predicated every activation load.

The staged redesign keeps the qualified M128/N64 geometry and the pair accumulation order and changes the skeleton: two projection tiles per stage, two LDS stages, one plain barrier per stage, one adjacent-code-word packed load per thread, projection and stage, and clamped activation rows. Reading the two 32-bit code words as one 8-byte load is timing-neutral here, so the decode keeps whichever form the compiler schedules best; two LDS stages with M128/N64 is the best point of the M64/M128/M256 x two/three stage sweep at B1 and B4, and inactive-wave suppression takes B16 (`2-3%`). Suppression costs `5%` at B1, so the catalog enables it only for the large-route rule.

The staged bodies are `208`-`247` VGPR / `28` SGPR / `16` KB LDS and are bitwise identical to the deployed body. Their bench result is `11.09/20.14/20.52` TFLOPS at B1/B4/B16 against `10.59/15.71/17.24` for the previous selection. The wider M256 body loses `8%` at B16 for this decoder, so only the M128/N64 two-stage bodies are deployed.

### Initial exact pair body

The generic grouped backward path used narrow N16/K16 ownership and serial rows. The exact IQ2_XXS pair body moved to M64/N64/K32 ownership, cooperative width-16 decode, two weight LDS images, and pair accumulation. Swizzle4 improved all 12 historical points by roughly `20-28%` over unswizzled staging; swizzle16 lost `24-39%` relative to swizzle4.

An early M128/N64 source form created an 8-byte private segment and was rejected before timing. A historical M192/N64 body used 32 private bytes, seven VGPR spills, and scratch instructions. Those failures do not close the later lower-state exact M128 identity, but they do close the spilling bodies.

### Unroll and inactive-M controls

Width32 decode had mixed route movement and no legal shape-only separator, so width16 was retained. Inactive-M suppression improved every measured point by `0.58-8.44%` and reduced the complete matrix geometrically by `4.08%`. N-major ownership and broad row-task traversal were rejected.

The coefficient-only campaign promoted M128/N64 only at exact aggregate rows `R=49152` and `R=196608`. The repaired lower-state body confirmed approximately `1.1678x` at B4 and `1.1644x` at B16 against the retained parent, with every learned, hash, and mandatory control above `1.05x` in the final qualification.

### Balanced row-task ownership

Learned route banks carry a large `max/mean` expert row spread, so the serial pair bodies kept column tiles of small experts idle while tall experts still walked rows. The deployed body now consumes a device row-task bank built once per route by `grouped_row_task_setup`: one task per 128-row tile of each active expert, so a small expert keeps one task and a tall expert splits into many, and each workgroup runs exactly one tile. Measured against the serial M128/N64 bodies on identical prepared inputs, kernel-only gains were `1.056x/1.016x/1.230x` at B1/B4/B16, and the official protocol moved `20.52` to `25.56` TFLOPS at B16. The task body carries inactive-wave suppression because the last task of every expert is partial. Wide-M task bodies were rejected (`m4n4` tasks reached only `0.61-0.67x` of the serial body), as were a third LDS stage and the projection-split decode. Reduction and projection splits were not pursued for this operator: the fused output is `rows x 4096` at a `2048`-wide reduction, so a global f32 partial round trip costs more than the multiply it would parallelize.

### Occupancy and stage sweep

A launch-bound and stage-count sweep over the retired staged serial pair bodies (`__launch_bounds__` second argument `2/3/4`, two, three and four LDS stages, and the inactive-wave suppression flag for the bodies that do not deploy it) changed no shape by more than measurement noise, and three or four stages lost `5-15%` on the small-route shapes through the larger LDS footprint. The two-stage, two-wave-per-SIMD geometry is retained. Evidence: `~/tmp/torch-ggml-ops/retune_pairs/run_pair_v2.py`.

## Resources

The deployed row-task pair body uses 208 VGPR / 28 SGPR / 16384 B LDS and runs one 128-row task per workgroup.

The deployed staged pair bodies use 208 (B1/B4) and 247 (B16) VGPR / 28 SGPR / 16384 B LDS; the retired M128/N64 body used 239 / 54 / 8192.

## Evidence

```text
~/tmp/torch-ggml-ops/retune_pairs/official/pair_deepseek.json
~/tmp/torch-ggml-ops/retune_pairs/official/pair_deepseek_final.json
~/tmp/torch-ggml-ops/retune_pairs/sweep.py
~/tmp/torch-ggml-ops/retune_pairs/prof/pcs_results.db
~/tmp/torch-ggml-ops/grouped_mmq_bwd_deepseek_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_iq2xxs_swizzle4_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_iq2xxs_swizzle16_focus.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_iq2xxs_width32_focus.json
~/tmp/torch-ggml-ops/grouped-bwd-finalists/deepseek_iq2xxs_m128_confirmation_25.json
~/tmp/torch-ggml-ops/grouped-bwd-production-final-correctness.json
```

The current HIP pair kernel is closed for broad J, width, and swizzle sweeps. The next material direction is lower-state IQ2_XXS decode or prepared weight reuse.
