# HIP MMQ Backward Q6_K Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ input-gradient kernels for Q6_K weights.

Backward shapes are written `(M, in_features, out_features)`, matching the weight's `(N,K) = (out_features, in_features)`. The type carries the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing): the QSA attention key/value and output projections and the shared-expert gate/up projection, at the training token counts of a sequence length 2048 batch (B1/B4/B16), and the chunked language model head of the Qwen3.6-35B-A3B (APEX-I-Mini) checkpoint at 64/128/256 rows. The 48 layers mix recipes - the same projection family is a different quant type in different layers - so every type that appears needs a body, or those layers fall back to a dequantizing multiply.

## GatedDeltaNet target shapes

`in_proj_qkv` and `in_proj_z` are in scope. `out_proj` is deferred. Both carry only the tiled -> grouped value-head reorder, which the loader applies to their packed *rows* as whole blocks, so the packed weight as loaded is already in the model's ordinary layout: a kernel takes the packed tensor and the layer input as they are, with no permutation, copy or transpose (`gated_delta_net_layout.md`).

Neither checkpoint carries this type in `in_proj_qkv` or `in_proj_z`, so this section adds no key. Still deferred: `out_proj`, which this type carries in the Qwen4-Exp checkpoint as `ssm_out` `(2560,6144)` twice.

## Final kernel result

| Family | `(M,K,N)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | ---: |
| Language model head | (64,2048,248320) | 14.340 | 2.174x | `dense_bwd_q6_k_pipesplit_m64_s2` |
| Language model head | (128,2048,248320) | 23.876 | 2.653x | `dense_bwd_q6_k_pipesplit_m128_s40` |
| Language model head | (256,2048,248320) | 22.443 | 1.634x | `dense_bwd_q6_k_pipesplit_m256_s40` |
| QSA key/value | (2048,2560,512) | 29.732 | 1.410x | `dense_bwd_q6_k_pipe_nt4_ki64_mw4_sw16` |
| QSA key/value | (8192,2560,512) | 30.633 | 1.170x | `dense_bwd_q6_k_pipe_nt4_ki64_mw4_sw16` |
| QSA key/value | (32768,2560,512) | 32.626 | 1.233x | `dense_bwd_q6_k_pipe_nt4_ki64_mw4_sw16` |
| QSA output | (2048,6144,2560) | 34.519 | 1.423x | `dense_bwd_q6_k_pipe_nt4_ki64_mw4_sw16` |
| QSA output | (8192,6144,2560) | 31.636 | 1.236x | `dense_bwd_q6_k_pipe_nt4_ki64_mw4_sw16` |
| QSA output | (32768,6144,2560) | 29.497 | 1.127x | `dense_bwd_q6_k_pipe_nt4_ki64_mw4_sw16` |
| Shared-expert gate/up | `(2048,2560,640)` | 33.369 | 1.553x | `dense_bwd_q6_k_mt128_nt128_ki32_full_k2048_nt4_ki64_mw4` |
| Shared-expert gate/up | `(8192,2560,640)` | 35.648 | 1.328x | `dense_bwd_q6_k_mt128_nt128_ki32_full_k2048_nt4_ki64_mw4` |
| Shared-expert gate/up | `(32768,2560,640)` | 36.641 | 1.326x | `dense_bwd_q6_k_mt128_nt128_ki32_full_k2048_nt4_ki64_mw4` |

The values use the current Q6_K packed/BF16 kernel matrix and come from one official run. M256 is the primary large chunk, with M64 and M128 as smaller exact geometries. The `Kernel` column names the deployed body for each chunk. All three chunks deploy a split-contraction body. M64 and M128 use the slice bodies measured below, and M256 moved to the pipelined slice body when it measured `1.07x` ahead of the single-pass body that chunk used to keep. The `_full_*` bodies are the unbounded exact variants (`_bounded` builds exist for shapes outside the exact-tile contract).

### Split-contraction deployment

The Q6_K language-model-head chunks launch 32 or 64 workgroups against a contraction of 248,320, so the machine is starved exactly as it is for the Q8_0 chunks, and the same mechanism applies: each workgroup takes one contiguous slice of the contraction, writes an FP32 partial tile, and a deterministic reduce kernel sums the slices in ascending order with a single BF16 rounding. M64 runs two slices, M128 runs forty, and both kernels are inside the timed region.

Parity at one slice was the prerequisite, and it needed two changes beyond the Q8_0 form. First, the deployed bodies use the unguarded full-tile path, and the same body with the bounded path costs more than half the throughput: the bounded slice body reached only `0.49x` of the deployed single-pass body at one slice, while the full-tile form with the deployed exact dimensions reaches `1.04x`. The exact dimensions alone account for that `1.04x`, since the single-pass bodies resolve both bounds at runtime. Second, the paired local-prefetch schedule of the deployed bodies is part of the slice body as well, which is worth a few percent here but was required for the Q8_0 slice bodies to match their single-pass twins.

A slice whose last stage is shorter than the stage width has to zero-fill the tile columns past the slice: a column beyond the slice would otherwise keep the previous stage's decoded weights and multiply them with real activations. Where a slice body is built but the stage alignment cannot be guaranteed, it is the bounded body that carries the guard.

Measured against the deployed single-pass bodies, the slice bodies gain `1.25x` at M64 with two slices and `1.71x` at M128 with forty. At M256 the same body was worth `1.07x` in the A/B harness and `+2%` under the official protocol, which at the time read as inside the run-to-run spread, so M256 kept its single-pass body. The pipelined slice body later measured `1.07x` ahead of that body at the same chunk and took the key. The A/B slice sweep, all against the same prepared gradient and packed weights: M64 `1.041x` at one slice, `1.251x` at two, `1.169x` at twenty, `1.247x` at forty, with a reproducible dip to `0.72x` at four slices. M128 `1.050x` at one, `1.566x` at four, `1.577x` at sixteen, `1.715x` at forty, `1.749x` at eighty. M256 `1.041x` at one, `0.947x` at sixteen, `1.071x` at forty, `1.077x` at eighty, and `0.239x` at 1,940 slices where the partial workspace dominates.

### Exact-dimension twins

The deployed bodies on this record resolve both the contraction and the result width at runtime. Twins that copy the deployed geometry exactly and only substitute the two compile-time bounds were built and measured against the deployed bodies in one interleaved A/B run: `0.990x` at `(256,2048,248320)`. Exact dimensions are therefore not deployed on these keys. Where the mechanism looked positive on the Q6_K M64 chunk, the measurement also carried the split-contraction body.

## Kernel implementation

The three head chunks use `dense_bwd_q6_k_m64_nt32_ki64_full`, `dense_bwd_q6_k_m128_nt64_ki32_full`, and `dense_bwd_q6_k_m256_nt64_ki32_full`: M64/N32/K64 for M64, M128/N64/K32 for M128, and two M128-style workgroups for M256. The Qwen4-Exp projection keys use two `mt128_nt128_ki32_full` bodies: `_k2048_nt4` (64 output columns per workgroup, 128 rows) and `_k2048_nt4_ki64_mw4` (64 output columns, a 64-value contract stage, four row tiles per wave). The decoded-weight LDS layout, packed extraction, and register lifetime are Q6-specific. The `_bounded` twins, the `nt128_ki16_g2`/`nt256_ki16_g2` generics, and the head lineage on the projection shapes are built but are not competitive.

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

Packed extraction remained selected for M256. Scalar extraction regressed `2.14%`. Exact bounds and K state accounted for only `2.09-2.56%` of the final M64/M128/M256 result, so the remaining M256 cost is packed Q6 arithmetic/representation rather than generic bounds handling.

### Arithmetic boundary

An effective-scale loader could stage `float(block_d * scale)` once per decoded row/K iteration. A future approximate-order experiment must show stable complete-kernel timing across all three chunks.

Global J64, I128, broad K64, activation double buffering, decoded-weight caching, speculative prefetch, split-K, persistent workgroups, and broad swizzle sweeps are closed for the present arithmetic contract.

### Projection-body screen

The head lineage does not transfer to the Qwen4-Exp projection shapes. On them its best member reaches only `6-17 TFLOPS` against `28-30` for the BF16 baseline, while a `mt128_nt128_ki32_full` body built from the Q4_K and Q5_K knob set reaches `25-27`. On the shared-expert gate/up at `M=32768` the ratio between the two is `4.5-5.0x`. The projection bodies below are that family.

Its first screen, on all three shapes, compared the deployed knob set (`_k2048`: eight output tiles, 32-wide contract stage, two row tiles per wave, swizzle eight) against a 16-tile and a four-tile variant, a swizzle-zero variant with `8`-word padding, a zero decoder-width and a non-prefetching variant, and 64-wide contract stages. Four output tiles (`64` columns per workgroup) is the large win, `+12-16%` over eight tiles. A 64-wide contract stage and four row tiles per wave add `+10-21%` on the two shapes whose result is 2560 wide, and nothing beyond noise on the 6144-wide one, where four output tiles with the deployed row geometry stay ahead. A zero decoder width and disabling packed prefetch measure within noise, and padding or 16 output tiles lose `+7%` and `5x` respectively.

### Pipelined tile

The Q2_0 record showed that a two-tile pipeline, one barrier per contraction stage, pays on the backward skeleton. The pilot here reuses that body (`csrc/ck/mmq_backward_pipelined.cuh`, generalized to the six staged weight types) at the deployed geometry, and includes the same body with the pipeline switched off so the two effects and the width-16 group decode can be read apart. Against the deployed `_k2048_nt4_ki64_mw4` body: the pipelined body is `1.02x` on the QSA output rows, `1.06-1.52x` on the QSA key/value rows (the narrowest result and the starved grid), and `0.96-1.04x` on the shared-expert gate/up rows, where the contraction is only ten stages deep and the pipeline's fill and drain dominate. Isolating the pipeline at fixed geometry and decode gives `1.003-1.18x` on all six measured points. The width-16 group decode on its own is not better than the deployed per-value path (`0.93-1.08x`), so the deployed bodies keep the pipelined tile with the group decode paired with the 64-value stage and four row tiles per wave. The key/value and output projection keys now deploy `_pipe_nt4_ki64_mw4_sw16`. The shared-expert keys keep their single-tile body.

### Split-contraction measurement

The split-K closure above was a contract deferral, not a measurement, so the language-model-head keys were retested with a dedicated split-contraction body: the deployed tile, decode and matrix work are unchanged, each workgroup takes one contiguous slice of the contraction, writes an FP32 partial tile, and a second kernel sums the slices in ascending order and rounds once to BF16. Both sides consume the same prepared gradient and the same packed weights.

Two forms were measured. The runtime-dimension form loses on all three chunks (`0.56-0.59x` at one slice, `0.74-0.78x` at the best slice count), and unlike the Q8_0 keys the deployed Q6_K bodies also carry runtime dimensions, so the remaining gap belongs to the experimental body's missing vectorised fragment loads and local prefetch rather than to shape specialization alone. The mechanism is therefore neither confirmed nor rejected here: the next step is a Q6_K split body that carries the deployed body's loader and exact dimensions and matches it at one slice before any slice count is judged. Evidence and the experimental body live under `~/tmp/torch-ggml-ops/retune_dense_bwd/`.

### Pipelined split-contraction head

The head chunks keep their split-contraction contract but take the pipelined stage order (`dense_mmq_pipelined_splitk_body`). Against the deployed slice bodies it is `1.06x` ahead at `M=64` and `1.03x` at `M=128`. At `M=256` it is `1.07x` ahead of the single-pass body that chunk used to keep, so the chunk moves to the pipelined slice body as well and all three keys deploy one. The prefetched decode that the projection rows use is neutral here (`0.94-1.01x`), so the projection keys keep the group decode while the head keys take the prefetched one because it is what the split body is built with.

## Resources

Retained head bodies use `87/138/137 VGPR` for M64/M128/M256, `15/16/15 SGPR`, and 4 KiB LDS. The projection bodies use `141 VGPR / 16 SGPR / 4 KiB` for `_k2048_nt4` and `213 VGPR / 17 SGPR / 8 KiB` for `_k2048_nt4_ki64_mw4`, both spill-free.

## Next

The pipelined tile is deployed on the key/value and output projections here and on every Q2_0 key. The remaining rollout is one screen per type on the same pattern: Q5_K and Q4_K first (their tiles are 8-10 KiB, so the pipeline costs a workgroup per WGP rather than the occupancy a Q8_0-shaped 4 KiB tile keeps), then Q8_0 (whose decode is as cheap as Q2_0's, so its barrier share should resemble this record's), then Q3_K. Each type needs its own per-key screen afterwards, because this pilot reproduced the Q2_0 finding that the pipeline changes which tile geometry wins: here the pipelined body only pays at the 64-value stage with four row tiles per wave, and the width-16 group decode that comes with it is slower than the deployed per-value path at the smaller geometry.

## Evidence

The Qwen4-Exp projection rows come from `~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q6k_final.txt` (with the screens in `bwd_q6k_v2.txt` and `bwd_q6k_v4.txt`), which uses the deployed preparation, the same `torch.mm` BF16 baseline and the same paired timing as the official protocol, under the same harness as the Q4_K and Q5_K records.

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
