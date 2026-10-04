# HIP MMQ Forward Q5_K Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ forward for Q5_K weights.

Q5_K carries ordinary projections of the Qwen3.6-35B-A3B (APEX-I-Mini) and Qwen3.8-Flash-Next (GSQ-RCO-Q2_0) checkpoints, so alongside the families measured first the record covers the QSA attention key/value and output, shared-expert gate/up, GatedDeltaNet `in_proj_qkv`, and the language model head, at the training token counts of a sequence length 2048 batch (B1/B4/B16). The head is the untied `output.weight` of `(248320,2560)` and is called in chunks of 64, 128 and 256 rows.

GatedDeltaNet `out_proj` is deferred because wiring it needs the activation permutation.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| ---: | --- | --- | --- | --- |
| Narrow K/V/gate/up | `(2048,512,2048)` | 24.954 | 1.82x | `dense_fwd_q5_k_k2048_j128_full` |
| Narrow K/V/gate/up | `(8192,512,2048)` | 27.296 | 1.48x | `dense_fwd_q5_k_k2048_j128_full` |
| Narrow K/V/gate/up | `(32768,512,2048)` | 27.986 | 1.46x | `dense_fwd_q5_k_k2048_j128_full` |
| Shared down | `(2048,2048,512)` | 24.546 | 8.23x | `dense_fwd_q5_k_k512_j128_full` |
| Shared down | `(8192,2048,512)` | 26.827 | 8.22x | `dense_fwd_q5_k_k512_j128_full` |
| Shared down | `(32768,2048,512)` | 27.691 | 8.36x | `dense_fwd_q5_k_k512_j128_full` |
| QSA key/value | `(2048,512,2560)` | 25.311 | 1.83x | `dense_fwd_q5_k_k2560_j128_full` |
| QSA key/value | `(8192,512,2560)` | 27.431 | 1.45x | `dense_fwd_q5_k_k2560_j128_full` |
| QSA key/value | `(32768,512,2560)` | 28.176 | 1.39x | `dense_fwd_q5_k_k2560_j128_full` |
| QSA output | `(2048,2560,6144)` | 28.030 | 1.19x | `dense_fwd_q5_k_k6144_j128_full` |
| QSA output | `(8192,2560,6144)` | 28.432 | 1.41x | `dense_fwd_q5_k_k6144_j128_full` |
| QSA output | `(32768,2560,6144)` | 27.158 | 1.38x | `dense_fwd_q5_k_k6144_j128_full` |
| Shared-expert gate/up | `(2048,640,2560)` | 24.513 | 1.72x | `dense_fwd_q5_k_k2560_j128_full` |
| Shared-expert gate/up | `(8192,640,2560)` | 27.916 | 1.88x | `dense_fwd_q5_k_k2560_j128_full` |
| Shared-expert gate/up | `(32768,640,2560)` | 28.232 | 1.88x | `dense_fwd_q5_k_k2560_j128_full` |
| GatedDeltaNet QKV APEX-I-Mini | `(2048,8192,2048)` | 28.112 | 1.28x | `dense_fwd_q5_k_k2048_j128_full` |
| GatedDeltaNet QKV APEX-I-Mini | `(8192,8192,2048)` | 28.042 | 1.24x | `dense_fwd_q5_k_k2048_j128_full` |
| GatedDeltaNet QKV APEX-I-Mini | `(32768,8192,2048)` | 28.232 | 1.25x | `dense_fwd_q5_k_k2048_j128_full` |
| Language model head | `(64,248320,2560)` | 25.840 | 3.40x | `dense_fwd_q5_k_k2560_j64_full` |
| Language model head | `(128,248320,2560)` | 27.878 | 2.66x | `dense_fwd_q5_k_k2560_j128_full` |
| Language model head | `(256,248320,2560)` | 27.763 | 1.32x | `dense_fwd_q5_k_k2560_j128_full` |

## Kernel implementation

The retained body uses the four-wave `I=64`, `J=128`, K256 geometry with Q5-specific low/high payload reconstruction, signed scales, and FP32 correction. Exact K512, K2048, K2560, and K6144 wrappers fold packed-row bytes and full-tile bounds while preserving a generic bounds-safe fallback. The 64-row head chunk uses a J64 exact body so a chunk smaller than one J128 tile does not idle half of it.

Q5-specific payload state is kept bounded to the active reduction phase. Broad cross-iteration packed prefetch was rejected because longer live ranges outweighed the load savings.

## Optimization log

### Initial producer and tiled body

The shared Q8_1 producer was changed from one 64-thread workgroup per padded row/block to one 512-thread workgroup per real row. The narrow Q4_K producer diagnostic fell from `5,105.979 us` to `886.928 us` at M32768. The four-wave tiled MMQ body then became the retained Q5_K foundation, replacing scalar 16x16 ownership with cooperative decode, LDS staging, and multiple WMMA accumulators.

### Q5 extraction and LDS choices

Q5_K padding did not transfer from Q4_K: the broad padded layout regressed Q5_K and was rejected. Explicit aligned fragment loads also regressed Q5_K by about `2-4%` in the shared cross-format controls. The final Q5 path therefore keeps its own packed extraction and four-BF16 shared-down layout rather than inheriting a global padding or vector-load rule.

The exact Qwen wrappers improved their generic controls by `11.66-32.60%`. Narrow Q5 extraction is selected by row regime in the measured kernel records: scalar extraction improved the smallest row count by `17.41%`, but regressed the larger row counts by `2.34%` and `6.52%`. Shared-down Q5 retained the four-BF16 XOR layout after an isolated swizzle8 gain failed to survive the complete matrix.

The cross-format extraction controls also established that Q5 padding regresses, while generic aligned fragment loads lose about `2-4%`. Shared-down alternatives with 4x4 or 1x16 ownership reached roughly `7.3-9.2 ms`. K64 reached `6.565 ms`, and disabling packed-byte prefetch reduced the artifact to 234 VGPRs but reached `6.186 ms`. These controls closed the local tile and register-pressure neighborhood rather than only one source variant.

The standalone small-route J32 body improved nonuniform routes by approximately `14-16%` but regressed the uniform route. That is why the small-row mechanism is retained as a bounded Q5 identity and not treated as a universal replacement.

### Closed mechanisms

Global J64, I128, alternate workgroup sizes, K64, activation-half double buffering, decoded-weight LDS caching, broad packed prefetch, split-K, persistent workgroups, generic swizzle rules, and the wide `I=128` dense tile are closed for the current Q5 representation. A future experiment must first demonstrate lower decode state or a lossless prepared representation.

### Hoisted epilogue metadata

The epilogue metadata of this quant was hoisted out of the column loop in the same way the Q6_K body deploys it. Over the full ordinary-shape A/B with the official protocol the change is neutral here (per-shape ratios inside `0.99x` to `1.01x`, no consistent direction), so the shipped body keeps the vendored target. The result is independent of tolerance because the body only moves loads: the arithmetic and the accumulation order are unchanged. Evidence: `~/tmp/torch-ggml-ops/retune_fwd/official/`.

### K2560, K6144, and the chunked head

The Qwen4-Exp shapes added two contraction lengths, `K=2560` and `K=6144`, and the untied head. Exact specialization over the generic control is worth `2.4-13.6%` on the new projection shapes (`1.082-1.136x` on the narrow `(512,2560)`, `1.024-1.046x` on the QSA output, `1.051-1.066x` on shared-expert gate/up, `1.081-1.089x` on the APEX-I-Mini `(8192,2048)`), in the same range as the K512 and K2048 wrappers.

The wide `I=128` tile was rendered for this type as well. It does not pay: `1.210x` slower on `(512,2560)` at `M=2048` and `1.052-1.057x` slower at `M=32768` on the two widest shapes, so it is not deployed.

The head is a `(248320,2560)` projection called in chunks, and the chunk size is a real decision. At `M=64` the J128 generic control runs `1.918x` slower than the exact J64 body (`6.131 ms` against `3.196 ms`), because a 64-row chunk fills half of a J128 tile, so the 64-row chunk gets its own J64 body and the 128- and 256-row chunks use the J128 one. `M=512` was measured as well: `27.860` against `27.763 TFLOPS` at `M=256`, `+0.35%`, i.e. no gain, while the chunk's logits footprint doubles from `127 MB` to `254 MB`, so the head stays at chunks of 64, 128 and 256 rows. Evidence: `~/tmp/torch-ggml-ops/qwen4_fwd/sweep_q5k_v1.txt`.

## Resources

Retained Q5_K J128 bodies use `226 VGPR / 27 SGPR / 38,400 B LDS`. The J64 head body uses `146 VGPR / 27 SGPR / 28,928 B LDS`.

## Evidence

```text
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_after_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_bracket_25.txt
~/tmp/torch-ggml-ops/mmq_fwd_final_components.txt
```

The Qwen4-Exp, GatedDeltaNet and head points come from outside the public case list, because those shapes have HIP controls but no GGTensile problem key yet, so they cannot be selected as deployment cases. Their runs use the deployed Q8_1 producer and the same prepared inputs, the same `torch.mm` BF16 baseline on the same activation tensor, and the same paired timing as the official protocol:

```text
~/tmp/torch-ggml-ops/qwen4_fwd/sweep_q5k_v1.txt
```

The exact Q5 qualification also uses the common sequential baseline and full exact-wrapper bracket:

```text
~/tmp/torch-ggml-ops/mmq_fwd_baseline_primary_sequential.json
~/tmp/torch-ggml-ops/mmq_fwd_final_full.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_plan_control_9.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_after_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_bracket_25.txt
```

The current result is a shape-specific packed decoder, not evidence that Q5_K should use the Q4_K schedule or that a single padding/swizzle policy is portable across quant types.
