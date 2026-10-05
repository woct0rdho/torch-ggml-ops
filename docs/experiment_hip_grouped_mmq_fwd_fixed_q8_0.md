# HIP Fixed-Grouped MMQ Forward Q8_0 Experiment

## Scope

This record covers the fixed eight-group DeepSeek Q8_0 forward kernel.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/torch.bmm | HIP time (ms) | Kernel |
| ---: | --- | --- | --- | --- | --- |
| 1 | `8 x (2048,1024,4096)` | 13.63 | 0.768x | 10.084 | `grouped_fwd_fixed_q8_0_g8_k4096_j64_full` |
| 4 | `8 x (8192,1024,4096)` | 13.61 | 0.753x | 40.386 | `grouped_fwd_fixed_q8_0_g8_k4096_j64_full` |
| 16 | `8 x (32768,1024,4096)` | 13.68 | 0.757x | 160.692 | `grouped_fwd_fixed_q8_0_g8_k4096_j64_full` |

The packed kernel is slower than BMM because the baseline starts from already decoded BF16 weights.

That step is not free: it costs `0.964/3.820/15.267 ms` at B1/B4/B16, `8.7%` of the kernel-plus-quantizer time, and adding it back gives `12.41/12.46/12.50` TFLOPS. The body is unchanged from the earlier recording, so that reconstruction reproduces the earlier `12.57/12.57/12.58` reading within `1.3%`: the higher TFLOPS above is the protocol change, not a kernel change.

## Kernel implementation

The retained fixed-group body uses exact `(N,K)=(1024,4096)` per group, four wave32 waves, J64 ownership, width-16 Q8_0 decode, one unswizzled N64/K32 weight tile per group, and fixed group-Z ownership. It performs the group-major to token-major output conversion in the kernel path. The `_j64_bounded` twin is built for out-of-contract shapes and measured `0.9-1.2%` slower at the three batch sizes in the table, so it is not deployed for them.

The kernel does not rely on fabricated route metadata.

## Optimization log

### Initial geometry

The generic fixed body used narrow eight-wave ownership, N16/K16 work, and serial row handling. The first exact four-wave M256/N64/K32 body improved the generic fixed kernel by `8.09x`, `8.33x`, and `8.75x` at B1/B4/B16.

The retained fixed body uses `213 VGPR / 48 SGPR / 4096 B LDS`. M64 and M128 were slower at the smaller fixed batches; width32, swizzle4, and M512 were rejected by timing or the resource warning boundary.

M64 and M128 were screened but M256 remained the better fixed-group geometry at B1/B4. Width32 decode, swizzle4, M512, broad scale staging, rolled dot loops, and group-major Q8_1 workspace alternatives were invalid, resource-heavy, or slower.

### Coefficient-only geometry

The full typed campaign retained fixed Q8_0 J64 full/bounded bodies and tested J16, J80, wider ownership, bounded tails, decoder width, and LDS layout. No alternate geometry passed the timing and resource gates. The fixed path retains token rows rather than routed rows and has no inactive-expert work to suppress.

The residual is representation mismatch: Q8_0 packed payloads and scales must be reconstructed before WMMA, while BMM consumes a predecoded BF16 tensor. A lossless prepared payload/scale layout would be a separate representation experiment.

### Hoisted epilogue metadata

The epilogue metadata of this quant was hoisted out of the column loop in the same way the Q6_K body deploys it. Over the full ordinary-shape A/B with the official protocol the change is neutral here (per-shape ratios inside `0.99x` to `1.01x`, no consistent direction), so the shipped body keeps the vendored target. The result is independent of tolerance because the body only moves loads: the arithmetic and the accumulation order are unchanged. Evidence: `~/tmp/torch-ggml-ops/retune_fwd/official/`.

## Evidence

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/hip_vs_baseline/pass11_fixedfwd_deepseek.json
~/tmp/torch-ggml-ops/hip_vs_baseline/quantizer_probe.json   (excluded input quantization cost)
~/tmp/torch-ggml-ops/hip_selection/                         (per-case candidate campaign)
```

The original campaign evidence is:

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_deepseek_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped-fwd-prior-v1.json
~/tmp/torch-ggml-ops/grouped-fwd-candidate-resources.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
```

The fixed Q8_0 kernel is locally complete for the current packed representation. Further work must reduce representation or decode cost rather than reopen generic J choices.

The fixed-group campaign provenance is:

```text
~/tmp/torch-ggml-ops/grouped-fwd-prior-v1.json
~/tmp/torch-ggml-ops/grouped-fwd-candidate-resources.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
~/tmp/torch-ggml-ops/grouped-fwd-production-final-rebuild.json
~/tmp/torch-ggml-ops/grouped-fwd-production-final-resources.json
```
