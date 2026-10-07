# HIP MMQ Backward Q4_K Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ input-gradient kernels for Q4_K weights.

Backward shapes are written `(M,N,K)`, matching the weight's `(N,K) = (out_features, in_features)`. Beyond the Qwen query, narrow, attention-output and shared-down families measured first, the type carries the QSA attention key/value and output, shared-expert gate/up and GatedDeltaNet `in_proj_qkv` and `in_proj_z` projections of the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (hidden size 2560) and the GatedDeltaNet `in_proj_z` of the Qwen3.6-35B-A3B (APEX-I-Mini) checkpoint, at the training token counts of a sequence length 2048 batch (B1/B4/B16). The APEX-I-Mini `in_proj_qkv` `(8192,2048)` shares the already tabulated query shape.

GatedDeltaNet `out_proj` is deferred because wiring it needs the activation permutation.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| Query/query gate | `(2048,8192,2048)` | 26.705 | 1.570x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| Query/query gate | `(8192,8192,2048)` | 27.017 | 1.543x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| Query/query gate | `(32768,8192,2048)` | 24.071 | 1.368x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| Narrow K/V/gate/up | `(2048,512,2048)` | 27.536 | 1.192x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_pad8` |
| Narrow K/V/gate/up | `(8192,512,2048)` | 29.654 | 1.256x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| Narrow K/V/gate/up | `(32768,512,2048)` | 30.301 | 1.255x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| Attention output | (2048,2048,4096) | 34.240 | 1.493x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| Attention output | (8192,2048,4096) | 29.368 | 1.225x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| Attention output | (32768,2048,4096) | 29.537 | 1.224x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| Shared down | `(2048,2048,512)` | 20.553 | 1.600x | `dense_bwd_q4_k_mt128_nt128_ki32_full_sw16` |
| Shared down | `(8192,2048,512)` | 16.355 | 1.073x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_pad8` |
| Shared down | `(32768,2048,512)` | 19.019 | 1.141x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| QSA key/value | `(2048,512,2560)` | 28.576 | 1.414x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_pad8` |
| QSA key/value | `(8192,512,2560)` | 30.310 | 1.184x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_pad8` |
| QSA key/value | `(32768,512,2560)` | 31.021 | 1.211x | `dense_bwd_q4_k_pipea_nt64_ki64_mw4_pad8_prefetch` |
| QSA output | `(2048,2560,6144)` | 35.022 | 1.479x | `dense_bwd_q4_k_pipea_nt64_ki64_mw4_sw16_prefetch` |
| QSA output | `(8192,2560,6144)` | 33.108 | 1.351x | `dense_bwd_q4_k_pipea_nt64_ki64_mw4_sw16_prefetch` |
| QSA output | `(32768,2560,6144)` | 30.527 | 1.214x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_sw16` |
| Shared-expert gate/up | `(2048,640,2560)` | 33.915 | 1.601x | `dense_bwd_q4_k_pipea_nt64_ki64_mw4_pad8_prefetch` |
| Shared-expert gate/up | `(8192,640,2560)` | 34.740 | 1.302x | `dense_bwd_q4_k_pipea_nt64_ki64_mw4_pad8_prefetch` |
| Shared-expert gate/up | `(32768,640,2560)` | 35.223 | 1.305x | `dense_bwd_q4_k_pipea_nt64_ki64_mw4_pad8_prefetch` |
| GatedDeltaNet Z | `(2048,6144,2560)` | 28.106 | 1.268x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet Z | `(8192,6144,2560)` | 28.096 | 1.171x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet Z | `(32768,6144,2560)` | 25.648 | 1.041x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet QKV | `(2048,10240,2560)` | 27.916 | 1.253x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet QKV | `(8192,10240,2560)` | 26.831 | 1.101x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet QKV | `(32768,10240,2560)` | 24.392 | 0.986x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| GatedDeltaNet Z APEX-I-Mini | `(2048,4096,2048)` | 31.362 | 1.871x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_pad8` |
| GatedDeltaNet Z APEX-I-Mini | `(8192,4096,2048)` | 27.583 | 1.587x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet Z APEX-I-Mini | `(32768,4096,2048)` | 27.004 | 1.505x | `dense_bwd_q4_k_pipea_nt64_ki64_mw2_sw16_prefetch` |

### Qwen4-Exp and GatedDeltaNet shapes

The new shapes span contraction lengths from `2048` to `10240` and result widths from `512` to `2560`, all whole multiples of the `ki32` step and the 128-wide tiles, so the three tuned variants serve them without a new body. All three were timed on every point.

The `_k512` variant loses everywhere on the new points (`1.13-1.32x` behind the winner), which leaves the choice between `_k4096` and `_k2048`: the QSA key/value, QSA output, shared-expert gate/up and APEX-I-Mini `in_proj_z` rows take `_k4096` at every M (`1.02-1.06x` ahead), while the two long-contraction Qwen4 GatedDeltaNet rows, `(M,6144,2560)` and `(M,10240,2560)`, take `_k4096` at `M=2048` and `_k2048` above it, where it is `1.9-3.0%` faster. The deployed body therefore follows the measurement per key rather than the suffix.

Six of the new rows are below the BF16 `torch.mm` baseline (the long contractions, at `0.92-0.96x`) and twelve are above it (`0.95-1.44x`), the same spread the rows measured first show, so the new shapes change the mixture rather than the conclusion: the remaining cost is the packed decode the BF16 baseline does not pay.

### Exact-dimension twins

The deployed bodies on this record resolve both the contraction and the result width at runtime. Twins that copy the deployed geometry exactly and only substitute the two compile-time bounds were built and measured against the deployed bodies in one interleaved A/B run: `0.858x` at `(8192,8192,2048)` and `0.886x` at `(32768,2048,512)`. Exact dimensions are therefore not deployed on these keys. Where the mechanism looked positive on the Q6_K M64 chunk, the measurement also carried the split-contraction body.

## Kernel implementation

Q4_K's payload is a 144-byte block: two 32-byte high planes, a 64-byte low plane and 16 bytes of scale metadata for `QK_K = 256` weights. The deployed bodies are the reusable pipelined tile (`csrc/ck/mmq_backward_pipelined.cuh`) on twenty-four of the thirty keys and the first redesign's padded `mt128_nt128_ki32_full_sw16` tile on the six whose shapes the pipelined screen did not cover.

The pipelined tile runs a 128-thread workgroup, a 64-value contraction stage, `n_tiles = 4` result columns of 16 per wave, 128-row result blocks (`mw2`) or 256-row ones (`mw4`), two shared tiles with one barrier per stage, register double-buffered activation fragments, and either the width-16 cooperative decode or the prefetched `uint4` form.

The catalog carries the six bodies these keys deploy and nothing else: the bounded tails and generic fallbacks the first screens measured are recorded here, not built.

## Optimization log

### Swizzle-only twins of the padded tile

The `_k4096` body carried `lds_padding = 8` instead of the LDS swizzle its `_k2048` and `_k512` siblings use, and the Q2_0 campaign showed the padding is the LDS cost while the swizzle is what actually breaks the decoded tile's bank pattern. Swizzle-only twins (`_k4096_sw8`, `_k4096_sw16`, both at 8 KiB rather than 10 KiB) were screened against it. Padding survives on the narrow contraction shapes (`(512,2560)` and `(640,2560)`, `0.97-1.01x` either way) and loses everywhere else: `1.064/1.058/1.075x` on `(6144,2560)`, `1.066/1.067/1.070x` on `(10240,2560)` and `1.040/1.066/1.074x` on `(2048,4096)` across `M=2048/8192/32768`.

The swizzle twin also beats the `_k2048` body that those two wide shapes deploy above `M=2048` (`1.044-1.053x` at `M=8192` and `32768`), so all nine wide-contraction keys now deploy `_k4096_sw16` and the Qwen4 and APEX GatedDeltaNet keys gain `4-7%` over the previous bodies. The A/B harness for this screen is `~/tmp/torch-ggml-ops/qwen4_fwd/bwd_final_decisions.txt` and the deployed-body confirmation is `bwd_deploy_confirm.txt`.

### Qwen tiled geometry

The first redesign replaced scalar 16x16 ownership with four wave32 waves, multiple accumulators, cooperative packed decode, and LDS-staged weights. The common retained geometry is 128x128/K32. Early global J64 and larger ownership alternatives lost reuse, parallelism, or residency.

Q4_K shape-specific controls retain padded vector local loads for query/narrow and a 16-BF16 XOR layout for shared down. Shared-down K512 is a distinct cost model and does not inherit the query layout.

### Padding and local-load experiments

Pre-layout Q4_K controls reported a repeated `79.2%` LDS-bank-conflict metric. Eight BF16 values of row padding changed the K32 stride from 64 to 80 bytes and improved query, narrow, and attention-output representative points, while shared down regressed:

| Shape | Unpadded | Padded | Decision |
| --- | ---: | ---: | --- |
| Query | 54.895 ms | 46.295 ms | retain typed padding |
| Narrow | 3.273 ms | 2.907 ms | retain typed padding |
| Attention output | 26.875 ms | 23.177 ms | retain typed padding |
| Shared down | 5.261 ms | 5.389 ms | reject padding for this body |

Explicit aligned fragment loads helped narrow Q3_K and attention-output Q4_K but regressed query Q4_K, Q5_K, and shared-down controls. Lower instruction count did not predict lower event time or LDS stalls.

### Pipeline and exact-shape work

The true two-buffer decoded-weight pipeline reduced aggregate wave cycles and wait/barrier stalls in the selected controls. It required a live-range repair after K64 reused addresses still needed by later WMMA pairs. The corrected handoff is exact at K32, K64, K96, and K512.

Exact-shape simplifications removed dead dimension loads, shortened address state, strength-reduced power-of-two strides, removed a fixed final `s_nop 7`, and normalized packed Q4 nibbles once per dword. These changes are resource-neutral or reducing and preserve the 40-byte ABI and output order.

### Pipelined tile on the narrow-result rows

The Q5_K and Q6_K records show the pipelined stage order paying where the result is narrow. The same body was built here at that geometry (`_pipea_nt64_ki64_mw4_sw16`) and screened against the deployed swizzle tile on three families at `M=2048` and `32768`: it wins on the QSA key/value rows (`1.12x` and `1.15x`), ties on the GatedDeltaNet `(2560,6144)` rows (`1.07x` at `M=2048`, `0.99x` at `32768`) and loses on the APEX-I-Mini `(8192,2048)` rows (`0.96x` and `0.92x`), so only the two measured QSA key/value rows took it at the time.

The screen covered one row-tile count on five shapes, and the losses it recorded belong to the 256-row block: the `M=2048` APEX loss is a grid that cannot fill from 256-row blocks, and the 128-row twin of the same body is what every record tuned after this one deploys. With both decode forms that twin takes twenty-three of these keys, `4-29%` ahead of the swizzle and padded tiles: `26-29%` on the shared-expert gate/up rows, where the 256-row prefetch twin is the fastest of the three, `13-26%` on the QSA output rows, `8-21%` on the QSA key/value rows, and `10-21%` on the query and GatedDeltaNet rows.

The QSA key/value row at `M=32768` and the shared-down and attention-output shapes outside the screen keep their first-generation bodies. The deeper three-tile rotation screened on Q2_0, Q4_0, IQ4_XS, Q5_K and Q8_0 loses here as well (`1.03-1.21x`), so two shared tiles stay.

The pipelined twins of the eight-column tile, at `sw8` and `sw16`, lose `1.03-1.39x` on all six points, and the eight-column tile pays a decode penalty the deployed `prefetch_packed` path does not.

### Prefetched decode on the pipelined tile

The pipelined QSA key/value rows used the width-16 group decode, which is what caps them against the deployed `prefetch_packed` bodies. Reading the payload planes as `uint4` before decoding closes that gap: the prefetched twin is `1.02x` and `1.00x` ahead of the group-decoding pipelined tile on those two rows, which now deploy it.

## Resources

The deployed pipelined bodies use `208 VGPR / 22 SGPR` at 16 KiB (`mw2`) and `216 VGPR / 22 SGPR` at 16 KiB (`mw4`), spill-free. The first-generation bodies kept for the shapes outside the pipelined screen use `222 VGPR / 20 SGPR / 10 KiB LDS` (`mt128_nt128_ki32_full_sw16`) and `220 VGPR / 16 SGPR / 8 KiB LDS` (`_k512`).

### Padding on the pipelined tile

The pipelined tile's layout was chosen when the tile was first built - sixteen-value swizzle chunks, no padding - and the chunk was later swept on this body without revisiting padding. The layout is now screened over every deployed dense-backward key: the padding twin of each deployed body (eight values of row padding, no swizzle) is timed against it at the key's own row counts, four repeats per block, and every screen winner is then re-timed at eight repeats over two blocks in both measurement orders, so a key moves only when the padding body wins in both directions.

Padding is not a general replacement for the swizzle. It wins where the resident grid is thin and the result is narrow - the `M=2048` rows and the 512- and 640-wide results - and loses on the wide results at the largest row counts, where the decoded tile's shared-memory traffic is high enough that the swizzle's bank pattern still pays.

The split is per key rather than per type: on this type eleven rows move, the largest group of any type: `14.4%` on the GatedDeltaNet `in_proj_z` eye at `M=2048`, `10.7%` on the narrow key at that row count, `8.0%` on the query row, and `4-6%` on the QSA key/value rows at `M=8192`, the narrow key at `M=8192`, the shared-expert gate/up rows and the remaining `M=2048` rows. The wide attention-output rows and the APEX GatedDeltaNet rows at `M=32768` keep their swizzled bodies.

A second layout round then screened three further combinations on every officially measured body - sixteen-value padding, four-value padding, and eight-value padding with an eight-value swizzle chunk - at the deployed row counts. Nothing survives the confirmation: each candidate measures within `1-3%` of the deployed tile in one measurement order and loses in the other, so all three are rejected and none is built. The layout neighborhood is closed at the combinations the deployment uses.

### Reassignment of the sub-parity rows

The tables' slowest rows were then revisited as a group. For every key whose ratio to BF16 sat at or below parity, each catalog body of the same type whose exact dimensions and tile geometry can express the shape was timed against the deployed body, four repeats per block, and every winner was re-timed at eight repeats over two blocks in both measurement orders. The pattern the screen found is systematic rather than specific: these families still ran first-generation tiles that the pipelined rollout never revisited for those particular shapes, so the fix is reassignment rather than a new mechanism.

On this type the shared-down `(M,512,2048)` rows were the worst in the matrix (`0.745x` and `0.789x` against BF16 at `M=8192` and `32768`), because their 2048-wide contraction ran the first-generation `_k4096` and `_k512` tiles. The pipelined four-column tile takes `M=8192` and `M=32768` (`32.8%` and `34.3%` ahead in both measurement orders, `16.4` and `19.0` TFLOPS against BF16 baselines of `15.2` and `16.6`), and `M=2048` keeps a single-tile body but takes the swizzled `_k4096_sw16` twin (`9.4%` ahead, `20.6` TFLOPS). The GatedDeltaNet `(10240,2560)` row at `M=32768` also moves from the group-decoding pipelined tile to the prefetched one (`1.8%`).

### The narrow-result shape is a shape cost, not a body cost

The shared-down family came out of the reassignment with the best body the catalog holds, but still `1.6x` behind its mirror (`(M,512,2048)` against `(M,2048,512)`), so the tile space around it was probed directly instead of assumed closed. The same padding-prefetch body was measured on both orientations at `M=32768` and gives `18.4` against `30.1` TFLOPS: with a 512-wide result the contraction is 2048 deep, so a block runs 32 stages for the same tile that the mirror clears in 8, and the grid is only 8 result blocks deep, so the resident set spans many more activation slabs.

Three further tiles were built and rejected against the deployed four-column, 64-value, two-row-tile body in both measurement orders: a 128-column tile at the same stage depth (`+17%` on the mirror and `+37%` on the narrow shape at `M=32768`, `+45%` at `M=8192`), the same tile with a 128-value stage (`+35%` to `+90%`), and 32-column tiles with eight and four row tiles (`+10%` to `+46%` on the narrow shape, up to `+316%` at the lower row counts).

Padding, swizzle, prefetching and the group decode were already screened on the family, and the single-row-tile and wider-stage directions are the two the earlier Q5_K and Q6_K screens had rejected as well. Every axis of the expressible tile space is therefore measured on this shape: width `2/4/8`, height `2/4` row tiles, depth `32/64/128`, both layouts, both decodes.

The deployed body is the optimum of that space, and the residual gap to the mirror is the price of a long contraction over a narrow result - the barrier wait at the pipeline sync is `29%` of issue slots on the narrow orientation against `13.6%` on the mirror, with the same instruction and byte mix per stage. Removing it would need a barrier-free decode that gives up the cooperative decode's fourfold sharing, i.e. a different data flow rather than another tile.

The same screen also closed the two knobs that had never been swept on this type's keys. The attention-output `(M,2048,4096)` family was the last one still on a first-generation tile - its record's layout screen had compared padding and swizzle *within* that family, never against the pipelined tile - and the pipelined four-column padding body takes all three row counts: `26.6%`, `15.0%` and `14.3%` ahead in both measurement orders, i.e. `34.2`, `29.4` and `29.5` TFLOPS at ratios `1.224-1.493x` against `18.3`, `23.6` and `24.5` before.

The traversal knob (`group_m = 0`) was built as twins of both the deployed and the pipelined bodies for the starved narrow-result `M=2048` keys. It loses there by `1.14x` and by `1.8-4.4x` at the larger row counts, so the M-grouped order stays everywhere except the 640-wide family where the earlier traversal screen deployed it.

## Evidence

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q4k_v1.txt
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q4k_v2.txt
~/tmp/torch-ggml-ops/hip_vs_baseline/pass11_ordbwd_qwen.json
~/tmp/torch-ggml-ops/hip_selection/           (per-key candidate campaign)
```

The original campaign evidence is:

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_narrow_q5_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_post_db8_control_9.json
~/tmp/torch-ggml-ops/mmq_bwd_final_full_v3.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb0_folded_25.json
```

Additional retained Q4 controls include the generic/exact DeepSeek build bracket and the Qwen baseline matrix:

```text
~/tmp/torch-ggml-ops/mmq_bwd_baseline_primary_sequential.json
~/tmp/torch-ggml-ops/mmq_bwd_final_full_v3.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_exact_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_generic_after_25.json
```

Global J64, I128, broad K64, split-K, GSU, Stream-K, persistent workgroups, compiler-managed prefetch arrays, decoded-weight LDS caching, and universal swizzle policies are closed for the current Q4_K representation. The pipelined tile below supersedes part of that list for the keys it took: the activation prefetch, the swizzle-only shared tile and the stage overlap are deployed there, while global J64, I128, broad K64, split-K, GSU, Stream-K and the persistent-workgroup forms stay closed.
