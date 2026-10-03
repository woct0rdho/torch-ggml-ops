# HIP MMQ Forward Q5_K Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ forward for Q5_K weights.

| Family | Logical weight `(N,K)` | M values | Tensors |
| --- | ---: | ---: | ---: |
| Narrow K/V/shared gate/up | `(512,2048)` | `2048,8192,32768` | 21 |
| Shared-expert down | `(2048,512)` | `2048,8192,32768` | 10 |

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| ---: | --- | --- | --- | --- |
| Narrow K/V/gate/up | `(2048,512,2048)` | 24.954 | 1.82x | `dense_fwd_q5_k_k2048_j128_full` |
| Narrow K/V/gate/up | `(8192,512,2048)` | 27.296 | 1.48x | `dense_fwd_q5_k_k2048_j128_full` |
| Narrow K/V/gate/up | `(32768,512,2048)` | 27.986 | 1.46x | `dense_fwd_q5_k_k2048_j128_full` |
| Shared down | `(2048,2048,512)` | 24.546 | 8.23x | `dense_fwd_q5_k_k512_j128_full` |
| Shared down | `(8192,2048,512)` | 26.827 | 8.22x | `dense_fwd_q5_k_k512_j128_full` |
| Shared down | `(32768,2048,512)` | 27.691 | 8.36x | `dense_fwd_q5_k_k512_j128_full` |

## Kernel implementation

The retained body uses the four-wave `I=64`, `J=128`, K256 geometry with Q5-specific low/high payload reconstruction, signed scales, and FP32 correction. Exact K512 and K2048 wrappers fold packed-row bytes and full-tile bounds while preserving a generic bounds-safe fallback.

Q5-specific payload state is kept bounded to the active reduction phase; broad cross-iteration packed prefetch was rejected because longer live ranges outweighed the load savings.

## Optimization log

### Initial producer and tiled body

The shared Q8_1 producer was changed from one 64-thread workgroup per padded row/block to one 512-thread workgroup per real row. The narrow Q4_K producer diagnostic fell from `5,105.979 us` to `886.928 us` at M32768. The four-wave tiled MMQ body then became the retained Q5_K foundation, replacing scalar 16x16 ownership with cooperative decode, LDS staging, and multiple WMMA accumulators.

### Q5 extraction and LDS choices

Q5_K padding did not transfer from Q4_K: the broad padded layout regressed Q5_K and was rejected. Explicit aligned fragment loads also regressed Q5_K by about `2-4%` in the shared cross-format controls. The final Q5 path therefore keeps its own packed extraction and four-BF16 shared-down layout rather than inheriting a global padding or vector-load rule.

The exact Qwen wrappers improved their generic controls by `11.66-32.60%`. Narrow Q5 extraction is selected by row regime in the measured kernel records: scalar extraction improved the smallest row count by `17.41%`, but regressed the larger row counts by `2.34%` and `6.52%`. Shared-down Q5 retained the four-BF16 XOR layout after an isolated swizzle8 gain failed to survive the complete matrix.

The cross-format extraction controls also established that Q5 padding regresses, while generic aligned fragment loads lose about `2-4%`. Shared-down alternatives with 4x4 or 1x16 ownership reached roughly `7.3-9.2 ms`; K64 reached `6.565 ms`; and disabling packed-byte prefetch reduced the artifact to 234 VGPRs but reached `6.186 ms`. These controls closed the local tile and register-pressure neighborhood rather than only one source variant.

The standalone small-route J32 body improved nonuniform routes by approximately `14-16%` but regressed the uniform route. That is why the small-row mechanism is retained as a bounded Q5 identity and not treated as a universal replacement.

### Closed mechanisms

Global J64, I128, alternate workgroup sizes, K64, activation-half double buffering, decoded-weight LDS caching, broad packed prefetch, split-K, persistent workgroups, and generic swizzle rules are closed for the current Q5 representation. A future experiment must first demonstrate lower decode state or a lossless prepared representation.

### Hoisted epilogue metadata

The epilogue metadata of this quant was hoisted out of the column loop in the same way the Q6_K body deploys it. Over the full ordinary-shape A/B with the official protocol the change is neutral here (per-shape ratios inside `0.99x` to `1.01x`, no consistent direction), so the shipped body keeps the vendored target. The result is independent of tolerance because the body only moves loads: the arithmetic and the accumulation order are unchanged. Evidence: `~/tmp/torch-ggml-ops/retune_fwd/official/`.

## Resources

Retained Q5_K J128 bodies use `244 VGPR / 28 SGPR / 38,400 B LDS`.

## Evidence

```text
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_after_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_bracket_25.txt
~/tmp/torch-ggml-ops/mmq_fwd_final_components.txt
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
