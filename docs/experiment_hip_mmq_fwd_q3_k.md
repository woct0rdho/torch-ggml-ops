# HIP MMQ Forward Q3_K Experiment

## Scope

This record covers the gfx1151 HIP packed-MMQ forward kernels for Q3_K weights.

Q3_K carries the ordinary projections of the Qwen3.6-35B-A3B (APEX-I-Mini) and Qwen3.8-Flash-Next (GSQ-RCO-Q2_0) checkpoints, so the record also covers the QSA attention, shared-expert and GatedDeltaNet shapes those checkpoints add at hidden size 2560, at the training token counts of a sequence length 2048 batch (B1/B4/B16).

The GatedDeltaNet `in_proj_qkv` and `in_proj_z` projections are in scope: the loader applies their tiled -> grouped value-head reorder to packed rows as whole blocks, so their packed weights are in the model's ordinary layout and need no permutation, copy or transpose (`gated_delta_net_layout.md`). GatedDeltaNet `out_proj` is deferred because wiring it needs the activation permutation, and `token_embd.weight` is an embedding gather rather than a multiply.

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

Every listed point is at or above the BF16 baseline on the multiply-only surface. The narrow families are well above it and the wide ones sit at the baseline.

## Kernel implementation

Exact Q3_K K2048 and K2560 bodies fold the contraction length, matrix dimensions and packed offsets into the generated kernel while retaining bounds-safe fallback code for other shapes. Packed payload and scale state are held only for the active decode phase so it does not extend through the WMMA loop.

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

### K2560 exact specialization and the wide tile

The Qwen4-Exp and GatedDeltaNet shapes needed a second contraction length, `K=2560` (ten `K=256` blocks per packed row). Exact specialization of that length over the generic control is worth `1.8-5.8%` on every new shape, the same direction and size as the K2048 result, and its output is bitwise identical to the generic body. The epilogue hoist is again neutral at this length (`0.996-1.002x` over the same eighteen points), so the shipped body keeps the vendored target.

The new geometries change the weight-to-activation byte ratio: at `K=2560` and `N=8192` or more, a workgroup reads `368 KB` of activation for `70 KB` of weight, and the activation tile is re-read once per 64-row column tile. A wide tile (`I=128`, 256 threads, eight waves, `61,952 B` of LDS, half the column tiles and therefore half the activation traffic) was built as `dense_fwd_q3_k_k{2048,2560}_j128_wide` and measured against the retained body on all eight shape families. It does not pay: `1.009-1.042x` slower than the retained body over the twenty-four points, worst on the narrow families, and it is not deployed.

The two limits that bound this body are therefore both already reached. Instruction-side, `rocprofv3` PC sampling of the retained K2560 body at `(8192,12288,2560)` reports `60%` VALU against `7.5%` WMMA, a ratio of `8.8` VALU instructions per WMMA, with the epilogue at one fused multiply-add per output element per sixteen-element MMA step, which is the minimum this decomposition allows. Memory-side, the large-M points run at `238-261 GB/s`, i.e. at the DRAM rate, and halving their traffic does not move the time, so neither side can be bought with the other.

The ceilings above are what a TFLOP/s figure should be read against. A pure `v_wmma_i32_16x16x16_iu8` probe with the same wave count, the same eight independent accumulators per wave and the same 128-thread workgroup reaches `48.7 TFLOPS` flat from two to eight workgroups per CU (`~/tmp/torch-ggml-ops/qwen4_fwd/wmma_i8_probe.cu`), so the `60 TFLOPS` nominal int8/bf16 roof is not reachable through this instruction. The retained body sits at `45-51%` of that practical ceiling, and the rejected wide tile at `44-50%`. Evidence: `~/tmp/torch-ggml-ops/qwen4_fwd/`.

## Resources

Both exact J128 bodies (K2048 and K2560) use `188 VGPR / 27 SGPR / 40,448 B LDS`. The rejected wide tile uses 256 threads and `61,952 B LDS` at the same register count.

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
