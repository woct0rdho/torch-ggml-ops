# HIP Grouped MMQ Forward Pair IQ2_XXS Experiment

## Scope

This record covers the fused routed IQ2_XXS gate/up forward kernel for DeepSeek.

There are 256 routed experts and aggregate rows `R=12288,49152,196608`.

The benchmark samples the `blk.3` expert tensors; `blk.0`-`blk.2` are hash-routed in this model, so the learned prior that the benchmark infers by default matches the sampled layers.

## Final kernel result

| Batch | Logical shape | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | ---: | ---: | ---: | --- |
| 1 | `2 x (12288,2048,4096)` | 12.11 | 2.635x | `grouped_fwd_serial_iq2_xxs_n2048_k4096_j64` |
| 4 | `2 x (49152,2048,4096)` | 18.29 | 1.600x | `grouped_fwd_serial_iq2_xxs_n2048_k4096_j64` |
| 16 | `2 x (196608,2048,4096)` | 21.98 | 1.796x | `grouped_fwd_serial_iq2_xxs_n2048_k4096_j80` |

The serial body is level at B1 and B16 (`1%`/`2%` against the uniform-route control) and trails by `15%` at B4, where the residual is the repeated two-weight IQ2_XXS decode and pair accumulator pressure. Flagged prior-sensitive at B4: the J64/J80 shape split may need a learned-route retune.

The table kernel is the deployed HIP body for these shapes and rebuilds byte-identically from the current sources; the retained choice was measured on other route distributions, so a learned-route candidate sweep is the follow-up for the flagged batches.

## Kernel implementation

The retained body uses exact N2048/K4096 pair geometry, four wave32 waves, cooperative width-16 IQ2_XXS decode, separate weight LDS tiles, swizzle4 staging, and device-side inactive-M consumer suppression. J64/J80 ownership is measured as a kernel geometry, not a host route label.

## Optimization log

### Initial exact decoder

The generic grouped body used narrow N16/K16 ownership and serial row work. Exact IQ2_XXS pair geometry and cooperative width-16 decode removed repeated metadata work and made the B1/B4 packed path substantially faster than the baseline. Early width32 controls had mixed route movement without a legal shape-only separator and were rejected.

The initial exact J64 body moved the B1 uniform pair from `77.549 ms` to `33.581 ms`. This is the historical tiled-kernel milestone behind the retained four-wave body, not an additional final workload result.

### J80 safety repair

The J80 candidate originally faulted for a non-divisible activation load: it attempted to load `2880` activation integers, leaving 64 excess lanes in the final 128-thread iteration. J16 had the same structural defect. The repaired kernel guards only the final non-divisible iteration while leaving complete cooperative iterations unchanged. Divisible J32/J64 code objects remain unchanged.

The repaired J80 result completed the previously faulting B16 profile and was approximately `1.096x` faster than J64 in the focused B16 comparison. Wider and smaller ownership alternatives were rejected by timing or resource gates.

### Coefficient-only campaign

The final typed search retained serial J64 at B1/B4 and the repaired J80 geometry for the large B16 workload. J16 and J64 at B16 were slower. The production body uses 208 VGPRs and 54 SGPRs for the J80 point.

The corresponding J64 artifact uses `229 VGPR / 77 SGPR / 8192 B LDS`; the repaired J80 artifact uses `208 VGPR / 54 SGPR / 8192 B LDS`. Both are resource-clean. The failed historical M192 body had 32 private bytes, seven VGPR spills, and scratch instructions, so its rejection does not transfer to the later lower-state J80 result.

The remaining bottleneck is two independent IQ2_XXS lookup/sign/scale decoders feeding one pair accumulation. A prepared weight layout or decode reuse is a representation change, not another broad J sweep.

## Evidence

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/grouped_fwd_current/pass13_pair_deepseek_default_small.json
~/tmp/torch-ggml-ops/grouped_fwd_current/pass13_pair_deepseek_default_b16.json
```

The original campaign evidence is:

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_deepseek_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped-fwd-all-screen/deepseek_iq2xxs_pair_identity_guard_b16_search_5.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/deepseek_iq2xxs_production_b16_replay.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
```

No further width or J sweep is justified without a new decode-state premise.

The final route campaign and repair qualification are additionally recorded in:

```text
~/tmp/torch-ggml-ops/grouped-fwd-prior-v1.json
~/tmp/torch-ggml-ops/grouped-fwd-candidate-resources.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
~/tmp/torch-ggml-ops/grouped-fwd-production-final-rebuild.json
~/tmp/torch-ggml-ops/grouped-fwd-production-final-resources.json
```
