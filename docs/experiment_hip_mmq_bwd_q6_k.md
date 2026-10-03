# HIP MMQ Backward Q6_K Experiment

## Scope

This record covers the Qwen language-model-head Q6_K packed input-gradient kernel on gfx1151.

## Final kernel result

| Family | `(M,K,N)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | ---: |
| Language model head | `(64,2048,248320)` | 13.970 | 2.119x | `dense_bwd_q6_k_exact_lm_head_splitk_m64_s2` |
| Language model head | `(128,2048,248320)` | 23.170 | 2.572x | `dense_bwd_q6_k_exact_lm_head_splitk_m128_s40` |
| Language model head | `(256,2048,248320)` | 21.620 | 1.577x | `dense_bwd_q6_k_m256_nt64_ki32_full` |

The values use the current Q6_K packed/BF16 kernel matrix and come from one official run. M256 is the primary large chunk, with M64 and M128 as smaller exact geometries. The `Kernel` column names the deployed body for each chunk. The M64 and M128 chunks deploy the split-contraction body described below; M256 keeps its single-pass body because the slice body measures within noise there. The `_full_*` bodies are the unbounded exact variants (`_bounded` builds exist for shapes outside the exact-tile contract).

### Split-contraction deployment

The Q6_K language-model-head chunks launch 32 or 64 workgroups against a contraction of 248,320, so the machine is starved exactly as it is for the Q8_0 chunks, and the same mechanism applies: each workgroup takes one contiguous slice of the contraction, writes an FP32 partial tile, and a deterministic reduce kernel sums the slices in ascending order with a single BF16 rounding. M64 runs two slices, M128 runs forty, and both kernels are inside the timed region.

Parity at one slice was the prerequisite, and it needed two changes beyond the Q8_0 form. First, the deployed bodies use the unguarded full-tile path (their activations are exactly `rows x in_features` and their weights exactly `out_features x in_features`), and the same body with the bounded path costs more than half the throughput: the bounded slice body reached only `0.49x` of the deployed single-pass body at one slice, while the full-tile form with the deployed exact dimensions reaches `1.04x`. The exact dimensions alone account for that `1.04x`, since the single-pass bodies resolve both bounds at runtime. Second, the paired local-prefetch schedule of the deployed bodies is part of the slice body as well (`PREFETCH_LOCAL`), which is worth a few percent here but was required for the Q8_0 slice bodies to match their single-pass twins.

The Q6_K decode stage is 64 wide on M64, so a slice width has to be a multiple of the stage width for every slice, including the truncated last one. The host rounds the width up to `max(32, k_iteration)` and the contraction bound is a multiple of the same step, so both the width and the bound are multiples of the stage width and the last slice is too; no stage is ever truncated, and the full-tile path needs no per-element guard. The Q8_0 slice bodies have the same invariant with a 16- or 32-wide stage. Where a slice body is built but the stage alignment cannot be guaranteed, it is the bounded body that carries the guard, and there the shared-tile bound is the *slice* end rather than the contraction: a tile column beyond the slice would otherwise keep the previous stage's decoded weights and multiply them with real activations, which measured as an NRMSE of `4.5e-2` against the `1.9e-3` of the correct form.

Measured against the deployed single-pass bodies with identical inputs, and with the reduce inside the timed region, the slice bodies gain `1.25x` at M64 with two slices and `1.71x` at M128 with forty; at M256 the same body is worth `1.07x` in the A/B harness and `+2%` under the official protocol, which is inside the run-to-run spread, so M256 keeps its single-pass body and no partial workspace is allocated for it. The A/B slice sweep, all against the same prepared gradient and packed weights: M64 `1.041x` at one slice, `1.251x` at two, `1.169x` at twenty, `1.247x` at forty, with a reproducible dip to `0.72x` at four slices; M128 `1.050x` at one, `1.566x` at four, `1.577x` at sixteen, `1.715x` at forty, `1.749x` at eighty; M256 `1.041x` at one, `0.947x` at sixteen, `1.071x` at forty, `1.077x` at eighty, and `0.239x` at 1,940 slices where the partial workspace dominates.

The slice bodies change the accumulation order, so their output is not bitwise equal to the single-pass body; the difference against that body is an NRMSE of `1.9e-3` at 40 or 80 slices and `1.5e-3` at two, and the external oracle check passes on the deployed routes.

### Exact-dimension twins

The deployed bodies on this record resolve both the contraction and the result width at runtime. Twins that copy the deployed geometry exactly and only substitute the two compile-time bounds were built and measured against the deployed bodies in one interleaved A/B run, with identical prepared inputs and bitwise identical output: `0.990x` at `(256,2048,248320)`. Exact dimensions are therefore not deployed on these keys; where the mechanism looked positive on the Q6_K M64 chunk, the measurement also carried the split-contraction body.

## Kernel implementation

The exact bodies are `dense_bwd_q6_k_m64_nt32_ki64_full`, `dense_bwd_q6_k_m128_nt64_ki32_full`, and `dense_bwd_q6_k_m256_nt64_ki32_full`: M64/N32/K64 for M64, M128/N64/K32 for M128, and two M128-style workgroups for M256. The decoded-weight LDS layout, packed extraction, and register lifetime are Q6-specific. The `_bounded` twins and the `nt128_ki16_g2`/`nt256_ki16_g2` generic bodies are built but are not competitive on the three deployment chunks.

## Exact-shape closure detail

The exact-shape twins were measured against the deployed single-pass bodies and lose (`0.990x` here, `0.858x` and `0.886x` on the two Q4_K keys), so exact dimensions are not a general win. The one case that looked positive, the M64 language-model-head chunk, was measured with the slice body, so its gain is not separable from the slice and the exact-dimension bound cannot be credited on its own.

## Optimization log

### Initial tiled redesign

The first redesign added four wave32 waves, LDS-staged decoded weights, multiple WMMA accumulators, cooperative packed decode, and exact row geometry. The representative LM-head Q6_K case moved from `158.209 ms` to `26.130 ms` for M256.

The selected small-row geometry screen was:

| M | Retained geometry and layout | Packed/BF16 timing |
| ---: | --- | ---: |
| 64 | M64/N32/K64, 16-BF16 XOR | `5.357/9.863 ms`, `1.84x` |
| 128 | M128/N64/K32, eight-BF16 XOR | `9.084/14.480 ms`, `1.59x` |
| 256 | Two M128/N64/K32 workgroups, packed extraction, eight-BF16 XOR | `11.726/19.010 ms`, `1.62x` |

M64 benefited from a 16-BF16 XOR layout that changed the decoded row bank phase without increasing LDS. M128 benefited from exact loader ownership. M256 was faster as two smaller workgroups because parallelism and lower accumulator pressure outweighed repeated packed decode.

### Invalid and rejected shapes

Two apparently fast M256 N=5/N=7 measurements were invalid: the selected N tile did not divide the 2,048-column result. The corrected N=7 body measured `29.628 ms`. K16/K64, M128 N3, M64 N3/N4, and alternate four-/16-BF16 layouts were rejected.

Packed extraction remained selected for M256; scalar extraction regressed `2.14%`. Exact bounds and K state accounted for only `2.09-2.56%` of the final M64/M128/M256 result, so the remaining M256 cost is packed Q6 arithmetic/representation rather than generic bounds handling.

### Arithmetic boundary

An effective-scale loader could stage `float(block_d * scale)` once per decoded row/K iteration. A future approximate-order experiment must show stable complete-kernel timing across all three chunks.

Global J64, I128, broad K64, activation double buffering, decoded-weight caching, speculative prefetch, split-K, persistent workgroups, and broad swizzle sweeps are closed for the present arithmetic contract.

### Split-contraction measurement

The split-K closure above was a contract deferral, not a measurement, so the language-model-head keys were retested with a dedicated split-contraction body: the deployed tile, decode and matrix work are unchanged, each workgroup takes one contiguous slice of the contraction, writes an FP32 partial tile, and a second kernel sums the slices in ascending order and rounds once to BF16. Both sides consume the same prepared gradient and the same packed weights.

Two forms were measured. The runtime-dimension form loses on all three chunks (`0.56-0.59x` at one slice, `0.74-0.78x` at the best slice count), and unlike the Q8_0 keys the deployed Q6_K bodies also carry runtime dimensions, so the remaining gap belongs to the experimental body's missing vectorised fragment loads and local prefetch rather than to shape specialization alone. The mechanism is therefore neither confirmed nor rejected here: the next step is a Q6_K split body that carries the deployed body's loader and exact dimensions and matches it at one slice before any slice count is judged. Evidence and the experimental body live under `~/tmp/torch-ggml-ops/retune_dense_bwd/`.

## Resources

Retained Q6_K bodies use `87/138/137 VGPR` for M64/M128/M256, `15/16/15 SGPR`, and 4 KiB LDS.

## Evidence

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/hip_vs_baseline/pass11_ordbwd_qwen.json
~/tmp/torch-ggml-ops/hip_selection/           (per-key candidate campaign)
tools/configs/hip_deployment.json             (deployed body per chunk)
```

The original campaign evidence is:

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_packed_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_scalar_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_packed_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_narrow_q5_25.json
```

The Q6-specific shape and arithmetic controls are:

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_packed_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_scalar_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_packed_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_post_ds4_p3_control_9.json
```
