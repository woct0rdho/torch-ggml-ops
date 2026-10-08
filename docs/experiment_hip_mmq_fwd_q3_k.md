# HIP MMQ Forward Q3_K Experiment

## Scope

This record covers the gfx1151 HIP packed-MMQ forward kernels for Q3_K weights.

Q3_K carries the ordinary projections of the Qwen3.6-35B-A3B (APEX-I-Mini) and Qwen3.8-Flash-Next (GSQ-RCO-Q2_0) checkpoints, so the record also covers the QSA attention, shared-expert and GatedDeltaNet shapes those checkpoints add at hidden size 2560, at the training token counts of a sequence length 2048 batch (B1/B4/B16).

Every GatedDeltaNet projection is in scope. `in_proj_qkv` and `in_proj_z` need no permutation because the loader applies their tiled -> grouped value-head reorder to packed rows as whole blocks (`gated_delta_net_layout.md`), and `out_proj` needs one only if the model converts that reorder back: with the file's value-head order kept end to end (`gdn_tiled_value_heads.py` in the training project) the projection consumes the packed columns in their own order. Its key is `(N,K) = (2560,6144)` in Qwen4, carried by 3 layers, and `(2048,4096)` in Qwen3.6-35B-A3B APEX-I-Mini, carried by 25 layers, so those are the two contraction lengths this record adds. `token_embd.weight` is an embedding gather rather than a multiply.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| ---: | --- | --- | --- | --- |
| Query/query gate | `(2048,8192,2048)` | 24.013 | 1.09x | `dense_fwd_q3_k_k2048_j128_full` |
| Query/query gate | `(8192,8192,2048)` | 23.701 | 1.06x | `dense_fwd_q3_k_k2048_j128_full` |
| Query/query gate | `(32768,8192,2048)` | 23.740 | 1.05x | `dense_fwd_q3_k_k2048_j128_full` |
| Narrow key | `(2048,512,2048)` | 21.219 | 1.55x | `dense_fwd_q3_k_k2048_j128_full` |
| Narrow key | `(8192,512,2048)` | 24.257 | 1.35x | `dense_fwd_q3_k_k2048_j128_full` |
| Narrow key | `(32768,512,2048)` | 24.087 | 1.25x | `dense_fwd_q3_k_k2048_j128_full` |
| QSA query | `(2048,12288,2560)` | 24.312 | 1.02x | `dense_fwd_q3_k_k2560_j128_full` |
| QSA query | `(8192,12288,2560)` | 23.283 | 1.01x | `dense_fwd_q3_k_k2560_j128_full` |
| QSA query | `(32768,12288,2560)` | 22.915 | 0.98x | `dense_fwd_q3_k_k2560_j128_full` |
| QSA key/value | `(2048,512,2560)` | 21.673 | 1.56x | `dense_fwd_q3_k_k2560_j128_full` |
| QSA key/value | `(8192,512,2560)` | 24.504 | 1.33x | `dense_fwd_q3_k_k2560_j128_full` |
| QSA key/value | `(32768,512,2560)` | 23.995 | 1.19x | `dense_fwd_q3_k_k2560_j128_full` |
| Shared-expert gate/up | `(2048,640,2560)` | 22.096 | 1.56x | `dense_fwd_q3_k_k2560_j128_full` |
| Shared-expert gate/up | `(8192,640,2560)` | 24.712 | 1.68x | `dense_fwd_q3_k_k2560_j128_full` |
| Shared-expert gate/up | `(32768,640,2560)` | 24.046 | 1.60x | `dense_fwd_q3_k_k2560_j128_full` |
| GatedDeltaNet QKV | `(2048,10240,2560)` | 24.139 | 1.02x | `dense_fwd_q3_k_k2560_j128_full` |
| GatedDeltaNet QKV | `(8192,10240,2560)` | 23.002 | 1.03x | `dense_fwd_q3_k_k2560_j128_full` |
| GatedDeltaNet QKV | `(32768,10240,2560)` | 22.811 | 0.99x | `dense_fwd_q3_k_k2560_j128_full` |
| GatedDeltaNet Z | `(2048,6144,2560)` | 24.096 | 1.04x | `dense_fwd_q3_k_k2560_j128_full` |
| GatedDeltaNet Z | `(8192,6144,2560)` | 23.076 | 1.04x | `dense_fwd_q3_k_k2560_j128_full` |
| GatedDeltaNet Z | `(32768,6144,2560)` | 22.706 | 0.99x | `dense_fwd_q3_k_k2560_j128_full` |
| GatedDeltaNet Z APEX-I-Mini | `(2048,4096,2048)` | 24.093 | 1.18x | `dense_fwd_q3_k_k2048_j128_full` |
| GatedDeltaNet Z APEX-I-Mini | `(8192,4096,2048)` | 23.765 | 1.16x | `dense_fwd_q3_k_k2048_j128_full` |
| GatedDeltaNet Z APEX-I-Mini | `(32768,4096,2048)` | 23.893 | 1.16x | `dense_fwd_q3_k_k2048_j128_full` |
| GatedDeltaNet `out_proj` | `(2048,2560,6144)` | 24.221 | 1.056x | `dense_fwd_q3_k_k6144_j128_full` |
| GatedDeltaNet `out_proj` | `(8192,2560,6144)` | 24.063 | 1.199x | `dense_fwd_q3_k_k6144_j128_full` |
| GatedDeltaNet `out_proj` | `(32768,2560,6144)` | 23.937 | 1.201x | `dense_fwd_q3_k_k6144_j128_full` |
| GatedDeltaNet `out_proj` APEX-I-Mini | `(2048,2048,4096)` | 24.339 | 1.280x | `dense_fwd_q3_k_k4096_j128_full` |
| GatedDeltaNet `out_proj` APEX-I-Mini | `(8192,2048,4096)` | 23.723 | 1.193x | `dense_fwd_q3_k_k4096_j128_full` |
| GatedDeltaNet `out_proj` APEX-I-Mini | `(32768,2048,4096)` | 23.243 | 1.156x | `dense_fwd_q3_k_k4096_j128_full` |

Every listed point is at or above the BF16 baseline on the multiply-only surface. The narrow families are well above it and the wide ones sit at the baseline.

## Kernel implementation

Exact Q3_K K2048, K2560, K4096 and K6144 bodies fold the contraction length, matrix dimensions and packed offsets into the generated kernel while retaining bounds-safe fallback code for other shapes. The two longer lengths are the ones `out_proj` needs, and they are the same body with a different folded stage count (16 and 24 `K=256` blocks per packed row), so they inherit the K2048/K2560 geometry without a new premise: `I=64`, `J=128`, four waves, `40,448 B` of LDS and the same register count. Packed payload and scale state are held only for the active decode phase so it does not extend through the WMMA loop.

## Optimization log

### Activation and initial geometry

The initial Q8_1 launch used one 64-thread workgroup per `(padded row, K/256 block)`. Replacing it with one 512-thread workgroup per real row removed repeated row work and reduced the narrow Q4_K producer from `5,105.979 us` to `886.928 us` at M32768. The same producer change exposed the Q3_K multiplication body as the remaining narrow cost.

The first four-wave tiled body replaced scalar 16x16 decode ownership with cooperative packed decode, LDS-staged weights, multiple WMMA accumulators, and exact row/type geometry. The final general geometry remained 128x128/K32. Global J64, I128, smaller workgroups, and broad grouped-M rules did not improve the production mix.

The narrow source timings are `0.223/0.919/3.869 ms` at M2048/M8192/M32768. Their dense-equivalent rates are the three narrow rows in the final table. The initial Qwen query improvement came from the same four-wave foundation and the 512-thread Q8_1 producer, while the exact wrapper is measured against the source-built control.

### Packed extraction and LDS layout

Wide Q3_K packed extraction, two-row prefetch, and the eight-BF16 XOR LDS layout were retained for the query body. Removing the extraction/prefetch/layout combination regressed the measured controls by `1.06-11.78%` depending on the point. Narrow Q3_K uses explicit packed state but does not prefetch because its 110-byte block layout made that path slower.

The direct wide-Q3 extraction control reduced query latency from `48.694 ms` to `46.980 ms`. Keeping the next reduction iteration's packed fragments live across WMMA instead reached `48.606 ms`, so the retained prefetch is bounded to the active decode phase.

Aligned local fragment loads improved narrow Q3_K but were not a universal rule: the same change regressed query Q4_K, Q5_K, and shared-down bodies. The final choice is therefore type- and shape-specific rather than a global vector-load policy.

### Exact specialization

The Q3_K K2048 exact wrapper was retained with a full J128 body. Across its measured matrix, exact specialization improved the generic control by `0.90-5.26%`. The detached build confirmed that the Q3_K artifact was unchanged while the separate DeepSeek Q8_0 phase was optimized.

### Hoisted epilogue metadata

The epilogue metadata of this quant was hoisted out of the column loop in the same way the Q6_K body deploys it. Over the full ordinary-shape A/B with the official protocol the change is neutral here (per-shape ratios inside `0.99x` to `1.01x`, no consistent direction), so the shipped body keeps the vendored target. The result is independent of tolerance because the body only moves loads: the arithmetic and the accumulation order are unchanged. Evidence: `~/tmp/torch-ggml-ops/retune_fwd/official/`.

### Out-projection contraction lengths (K4096 and K6144)

The two `out_proj` keys needed contraction lengths the type did not have yet, and both are exact in the sense of the other K wrappers: the stage count is folded and the row is a whole number of 256-value blocks. Over the six new points the exact body is `1.010-1.028x` of the generic control, the same small margin the K2560 work found at these result widths, and its output is bitwise identical to it. The three points at `(2560,6144)` run `23.9-24.2 TFLOPS` and the three at `(2048,4096)` run `23.2-24.3`.

The other four levers this record already closed were re-measured on these shapes rather than inherited, because the result width is narrower here (`N=2560` or `2048`) while the contraction is the longest of any deployed Q3_K key, which is the one combination the earlier campaigns did not cover:

| Arm | `(2048,2560,6144)` | `(8192,2560,6144)` | `(32768,2560,6144)` | `(2048,2048,4096)` | `(8192,2048,4096)` | `(32768,2048,4096)` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wide `I=128` tile | `1.017x` | `1.021x` | `1.014x` | `1.090x` | `1.013x` | `1.015x` |
| `J=64` tile | `1.083x` | `1.086x` | `1.089x` | `1.066x` | `1.068x` | `1.081x` |
| fragment-order activation | `1.590x` | `1.594x` | `1.620x` | `1.633x` | `1.551x` | `1.575x` |
| hoisted epilogue | `1.003x` | `0.999x` | `0.996x` | `1.002x` | `1.000x` | `1.001x` |

All four lose or are neutral, so the deployed body is the plain exact one. The wide tile is the one worth a sentence: at this result width it makes the activation re-read per workgroup smaller in absolute terms than at `N=12288`, yet it still loses by the same `1.4-2.1%` it did at K2560, which is the occupancy cost of its `61,952 B` request (one resident workgroup against three). The fragment-order body loses much more heavily here than it does on Q2_0 or IQ4_XS: the Q3_K decode is the heaviest of the family, but the fragment body replaces a coalesced activation copy with sixteen scattered 32-byte group reads per instruction, and at these contractions the body has no decode slack left to absorb them. Evidence: `~/tmp/torch-ggml-ops/outproj/`.

### K2560 exact specialization and the wide tile

The Qwen4-Exp and GatedDeltaNet shapes needed a second contraction length, `K=2560` (ten `K=256` blocks per packed row). Exact specialization of that length over the generic control is worth `1.8-5.8%` on every new shape, the same direction and size as the K2048 result, and its output is bitwise identical to the generic body. The epilogue hoist is again neutral at this length (`0.996-1.002x` over the same eighteen points), so the shipped body keeps the vendored target.

The new geometries change the weight-to-activation byte ratio: at `K=2560` and `N=8192` or more, a workgroup reads `368 KB` of activation for `70 KB` of weight, and the activation tile is re-read once per 64-row column tile. A wide tile (`I=128`, 256 threads, eight waves, `61,952 B` of LDS, half the column tiles and therefore half the activation traffic) was built as `dense_fwd_q3_k_k{2048,2560}_j128_wide` and measured against the retained body on all eight shape families. It does not pay: `1.009-1.042x` slower than the retained body over the twenty-four points, worst on the narrow families, and it is not deployed.

The two limits that bound this body are therefore both already reached. Instruction-side, `rocprofv3` PC sampling of the retained K2560 body at `(8192,12288,2560)` reports `60%` VALU against `7.5%` WMMA, a ratio of `8.8` VALU instructions per WMMA, with the epilogue at one fused multiply-add per output element per sixteen-element MMA step, which is the minimum this decomposition allows. Memory-side, the large-M points run at `238-261 GB/s`, i.e. at the DRAM rate, and halving their traffic does not move the time, so neither side can be bought with the other.

An isolated `v_wmma_i32_16x16x16_iu8` probe with this body's wave count, accumulator count and workgroup size was built to see what the instruction reaches on its own (`~/tmp/torch-ggml-ops/qwen4_fwd/wmma_i8_probe.cu`). Its figure is deliberately not used here as a ceiling for a whole kernel: nothing in the profiles above shows this body limited by the matrix pipe, so a probe number would bound the wrong resource, and the record keeps only the two limits it did measure. Evidence: `~/tmp/torch-ggml-ops/qwen4_fwd/`.

## Resources

All exact J128 bodies of this type (K2048, K2560, K4096 and K6144) use `188 VGPR / 27 SGPR / 40,448 B LDS`, so the longer contractions cost nothing in resources and keep three resident workgroups per WGP. The rejected wide tile uses 256 threads and `61,952 B LDS` at the same register count.

## Evidence

The current source-of-record measurements are:

```text
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_after_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_bracket_25.txt
~/tmp/torch-ggml-ops/mmq_fwd_final_components.txt
```

The Qwen4-Exp and GatedDeltaNet points come from outside the public case list, because those shapes have HIP controls but no GGTensile problem key yet, so they cannot be selected as deployment cases. Their runs use the deployed Q8_1 producer and the same prepared inputs, the same `torch.mm` BF16 baseline on the same activation tensor, and the same paired timing as the official protocol:

```text
~/tmp/torch-ggml-ops/qwen4_fwd/sweep_v1.txt
~/tmp/torch-ggml-ops/qwen4_fwd/verify.py
~/tmp/torch-ggml-ops/qwen4_fwd/probe_v1.txt
```

The `out_proj` round is staged from `~/tmp/torch-ggml-ops/outproj/`: the screen (`fwd_v1.txt`, `fwd_v1.json`), the higher-sample confirmations (`confirm_fwd.txt`, `confirm_frag.txt`, `confirm_frag_m2048.txt`, `confirm_frag_large.txt`), the bitwise check against the generic body (`verify_v1.txt`), and the deployed-key audit through the public route (`deployed_v1.txt`).

The standalone artifact controls also include the retained generic sequential baseline and the detached Qwen control used to check that DeepSeek exact-body work did not alter Q3_K bytes:

```text
~/tmp/torch-ggml-ops/mmq_fwd_baseline_primary_sequential.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_plan_control_9.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p1_control_9.json
~/tmp/torch-ggml-ops/mmq_fwd_pre_bundle.json
~/tmp/torch-ggml-ops/mmq_fwd_post_bundle.json
~/tmp/torch-ggml-ops/mmq_fwd_embedded_pre_control_25.json
~/tmp/torch-ggml-ops/mmq_fwd_bundle_control_25.json
~/tmp/torch-ggml-ops/mmq_fwd_embedded_post_control_25.json
```

Broad J64, I128, workgroup-size, split-K, decoded-weight LDS caching, speculative cross-iteration prefetch, and generic swizzle sweeps are closed for this packed representation. The wide `I=128` dense tile measured above is closed for the Q3_K shapes as well. A future change needs a new decode, representation, or activation-reuse premise and must be measured against both matrix families.
