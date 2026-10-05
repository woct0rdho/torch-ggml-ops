# HIP MMQ Backward Q5_K Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ input-gradient kernels for Q5_K weights.

Backward shapes are written `(M, in_features, out_features)`, matching the weight's `(N,K) = (out_features, in_features)`. Beyond the Qwen narrow and shared-down families measured first, the type carries the QSA attention key/value and output, shared-expert gate/up and the chunked language model head of the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (hidden size 2560), and the GatedDeltaNet `in_proj_qkv` of the Qwen3.6-35B-A3B (APEX-I-Mini) checkpoint, at the training token counts of a sequence length 2048 batch (B1/B4/B16). the head is called in chunks of 64, 128 and 256 rows.

GatedDeltaNet `out_proj` is deferred because wiring it needs the activation permutation.

## Final kernel result

| Family | `(M,K,N)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| Narrow K/V/gate/up | `(2048,2048,512)` | 27.246 | 1.191x | `dense_bwd_q5_k_pipea_nt4_ki64_mw2_pad8_prefetch` |
| Narrow K/V/gate/up | `(8192,2048,512)` | 29.332 | 1.249x | `dense_bwd_q5_k_pipea_nt4_ki64_mw2_pad8_prefetch` |
| Narrow K/V/gate/up | `(32768,2048,512)` | 30.003 | 1.248x | `dense_bwd_q5_k_pipea_nt4_ki64_mw2_pad8_prefetch` |
| Shared down | (2048,512,2048) | 17.315 | 1.348x | `dense_bwd_q5_k_mt128_nt128_ki32_full_k512` |
| Shared down | (8192,512,2048) | 15.187 | 0.997x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_pad8` |
| Shared down | (32768,512,2048) | 18.308 | 1.106x | `dense_bwd_q5_k_pipea_nt4_ki64_mw2_pad8_prefetch` |
| QSA key/value | (2048,2560,512) | 29.084 | 1.436x | `dense_bwd_q5_k_pipea_nt4_ki64_mw2_pad8_prefetch` |
| QSA key/value | (8192,2560,512) | 28.228 | 1.120x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_pad8` |
| QSA key/value | (32768,2560,512) | 30.497 | 1.214x | `dense_bwd_q5_k_pipea_nt4_ki64_mw2_pad8_prefetch` |
| QSA output | (2048,6144,2560) | 33.643 | 1.370x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw4_sw16` |
| QSA output | (8192,6144,2560) | 30.265 | 1.171x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_sw16` |
| QSA output | (32768,6144,2560) | 30.010 | 1.145x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_sw16` |
| Shared-expert gate/up | (2048,2560,640) | 30.356 | 1.418x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_sw16` |
| Shared-expert gate/up | (8192,2560,640) | 30.025 | 1.107x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_sw16` |
| Shared-expert gate/up | (32768,2560,640) | 33.861 | 1.193x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw4_sw16` |
| GatedDeltaNet QKV APEX-I-Mini | (2048,2048,8192) | 25.419 | 1.500x | `dense_bwd_q5_k_pipea_nt4_ki64_mw2_pad8_prefetch` |
| GatedDeltaNet QKV APEX-I-Mini | (8192,2048,8192) | 25.570 | 1.430x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_sw16` |
| GatedDeltaNet QKV APEX-I-Mini | (32768,2048,8192) | 24.091 | 1.330x | `dense_bwd_q5_k_pipea_nt4_ki64_mw2_pad8_prefetch` |
| Language model head | (64,2560,248320) | 17.798 | 1.222x | `dense_bwd_q5_k_pipesplit_m64_s4` |
| Language model head | (128,2560,248320) | 21.841 | 1.724x | `dense_bwd_q5_k_pipesplit_m128_s8` |
| Language model head | (256,2560,248320) | 25.548 | 1.348x | `dense_bwd_q5_k_pipesplit_m256_s32` |

The shared-down rows show the largest per-sample spread in the matrix (`1.20-1.53` max/min here). The narrow rows stay at `1.05-1.07`.

## Kernel implementation

The retained body uses four wave32 waves, exact 128x128/K32 ownership where applicable, decoded-weight LDS staging. The catalog carries the two tuned `mt128_nt128_ki32_full` variants (`_k512`, `_k2048`) for the shared-down and APEX shapes, the pipelined `nt4_ki64` tile at 128 and 256 rows with both decode forms, and three split-contraction head bodies. The screened-out candidates are recorded here rather than built. The language model head uses split-contraction bodies on the pattern of the Q6_K and Q8_0 records, with the partial reduction the deployment already carries. Q5-specific packed-row and signed-scale state stays separate from Q4_K.

## Optimization log

### Tiled redesign and shape specialization

The first four-wave Qwen body replaced scalar decode and serial 16x16 ownership with cooperative extraction, LDS staging, multiple WMMA accumulators, and exact shape/row geometry. Exact Q5_K wrappers then removed runtime bounds and address state while retaining bounds-safe fallback code.

The exact Q5_K matrix improved its generic controls by `11.66-32.60%`. The smallest narrow row count is the main extraction discriminator. Larger rows favor the packed path.

### Split-contraction head and the projection probe

The head contracts `248320` values with a result of `(M,2560)`, so its non-split grid is `1 x 40` workgroups, far below the machine's wave slots, and the contraction splits scale it directly: `m64_s4` beats `m64_s2` by `1.37x`, `m128_s8` beats `m128_s16` by `1.036x`, and `m256_s32` beats `m256_s16` by `1.036x`. Those three bodies deploy, and their partials are `(slices, M, 2560)` f32, which stays at `1.3-2.6 MiB`.

The same mechanism was then measured on the longest-contraction projection, the APEX-I-Mini `in_proj_qkv` `(M,2048,8192)`, where the grid is already `16 x 16 x 1` at `M=2048` and `256 x 16 x 1` at `M=32768`. Split-contraction loses there by `1.52-1.60x` at `M=2048` and `2.77-3.15x` at `M=32768`, because the partial workspace becomes `(slices, M, 2048)` f32, up to `1 GiB`, and the parallelism the split buys is parallelism the grid already had. Re-testing it with the pipelined slice body changes nothing: two and four slices are `1.73x` and `1.88x` behind at `M=2048` and `3.5-4.9x` behind at the two larger row counts, so the projection closure holds with both bodies. Split-contraction is therefore deployed for the chunked head only.

The projection keys were screened against `_k2048`, `_k512` and the scalar-extraction control: `_k2048` wins every point except the APEX-I-Mini `in_proj_qkv` at `M=2048`, where `_k512` is `3.6%` ahead, and the scalar control is `2-6%` behind the winner on every point it was timed.

### Extraction and LDS controls

Q5_K reused the Q4_K framework for width-16 low/high decode and bounded packed prefetch. Prefetch improved the initial port by `12-24%`. A universal padded layout regressed Q5_K and was rejected, and aligned fragment loads regressed Q5_K by about `2-4%` in cross-format controls.

Narrow Q5 retained scalar extraction only at 2,048 rows, where it improved `17.41%`. It regressed `2.34%` and `6.52%` at 8,192 and 32,768 rows. Shared-down Q5 retained its four-BF16 XOR layout after an isolated swizzle8 improvement did not survive the complete matrix.

The accepted body keeps packed/decode temporaries dead before long WMMA phases. Cross-iteration packed prefetch, K64, double buffering, decoded-weight LDS caching, and broad swizzle changes either extended register lifetimes or reduced residency.

### Closed directions

The current source closes global J64, I128, alternate workgroup sizes, split-K, GSU, Stream-K, persistent workgroups, generic local-load rules, and compiler-managed prefetch arrays. Q5-specific results must not be inferred from Q4_K or Q3_K because packed high-bit reconstruction changes the register and LDS cost model.

### Pipelined tile

The Q2_0 and Q6_K records established a two-tile backward stage order, one barrier per contraction stage, and this record is where it pays most. The pipelined twin of the deployed `_k2048` geometry, at the 64-value stage with two and four row tiles per wave, beats the deployed bodies on every measured point: `1.31x/1.08x/1.09x` on the QSA value rows, `1.42x/1.11x/1.19x` on the shared-expert gate/up rows, `1.37x/1.17x/1.15x` on the QSA output rows, and `1.03x/1.13x/1.16x` on the APEX-I-Mini `(512,2048)` rows, against `M=2048/8192/32768`.

On the APEX-I-Mini `(2048,8192)` rows the pipelined variant beats the eight-column tile by `1.15x` at `M=2048` and `1.02x` at `M=32768`. The `M=2048` row keeps the `_k512` wrapper it deploys and the `M=8192` row was not measured against the deployed body, so only the `M=32768` row takes the pipelined tile. The narrow-result rows prefer the two-row-tile variant and the wide rows the four-row-tile one at `M=2048`.

The `(2048,512)` shared-down rows kept their single-tile body after the first screen, which read the 256-row pipelined block as starving the `M=2048` grid and left a `1.10x` `M=32768` win unconfirmed. Both readings belong to that block: the 128-row prefetch twin is `14.5%`, `23.3%` and `21.4%` ahead of the single-tile body across `M=2048/8192/32768`, and the 256-row twin is `12-15%` ahead as well, so all three rows now deploy the 128-row prefetch tile. The swizzle-only twin of the deployed 8 KiB tile loses (`1.03-1.21x` behind) on this type, so the pipeline inherited the swizzle the deployed bodies already use. The padding screen at the end of this log then moves nine of these rows off it.

### Pipelined split-contraction head

The head keys were the one place the earlier records measured the tile pipeline on a single-tile body: the split-contraction body's stage loop was still the two-barrier order. The same pipelined body now carries the split window (`dense_mmq_pipelined_splitk_body`, sharing the projection tile's implementation), and against the deployed split bodies it is `1.40x` ahead at `M=64`, `1.04x` at `M=128` and `1.14x` at `M=256`, all three keys taking it. Its prefetched decode is what the projection rows took as well: `1.07x` and `1.06x` over the group-decoding pipelined tile on the QSA value rows, so those two rows moved to the prefetched twin too.

## Resources

The split head bodies use `74/110/198 VGPR` at `M=64/128/256` with `4 KiB` LDS.

The retained Q5_K bodies use `247 VGPR / 17 SGPR / 8 KiB LDS` for narrow scalar extraction and `253 VGPR / 16 SGPR / 8 KiB LDS` for shared down.

### Padding on the pipelined tile

The pipelined tile's layout was chosen when the tile was first built - sixteen-value swizzle chunks, no padding - and the chunk was later swept on this body without revisiting padding. The layout is now screened over every deployed dense-backward key: the padding twin of each deployed body (eight values of row padding, no swizzle) is timed against it at the key's own row counts, four repeats per block, and every screen winner is then re-timed at eight repeats over two blocks in both measurement orders, so a key moves only when the padding body wins in both directions.

Padding is not a general replacement for the swizzle. It wins where the resident grid is thin and the result is narrow - the `M=2048` rows and the 512- and 640-wide results - and loses on the wide results at the largest row counts, where the decoded tile's shared-memory traffic is high enough that the swizzle's bank pattern still pays. The split is per key rather than per type: on this type nine rows move, led by the narrow `(2048,512)` rows at `M=2048` (`10.7%`), the QSA key/value rows (`1.7-3.1%`) and the APEX GatedDeltaNet rows at the two larger counts (`2.5-6.0%`). This supersedes the earlier reading that the pipeline keeps the swizzle the deployed bodies use - it keeps it only on the wide rows, which this type does not have.

The shared-down rows also had to be re-based, and then reassigned. Their earlier table entries came from a harness run whose declared geometry did not match the tensor it loaded (the APEX shared-down tensor read as the narrow shape), which reported `23-28 TFLOPS` where the official run measures the same first-generation body at `13.0/11.4/14.6` for `M=2048/8192/32768`. Measured on the official basis the family is the type's worst, and the body-swap screen described below then moved `M=2048` and `M=32768` to other bodies: `17.3` and `18.3` TFLOPS against BF16 baselines of `12.8` and `16.6`, with `M=8192` level at `15.2` TFLOPS (`0.997x`). The replacement is not a regression: it is the same key measured against the same BF16 baseline as every other official row, which is why the ratio rises while the first-generation TFLOPS fall.

A second layout round then screened three further combinations on every officially measured body - sixteen-value padding, four-value padding, and eight-value padding with an eight-value swizzle chunk - at the deployed row counts. Nothing survives the confirmation: each candidate measures within `1-3%` of the deployed tile in one measurement order and loses in the other, so all three are rejected and none is built. The layout neighborhood is closed at the combinations the deployment uses.

### Reassignment of the sub-parity rows

The tables' slowest rows were then revisited as a group. For every key whose ratio to BF16 sat at or below parity, each catalog body of the same type whose exact dimensions and tile geometry can express the shape was timed against the deployed body, four repeats per block, and every winner was re-timed at eight repeats over two blocks in both measurement orders. This type's two sub-parity families are the shared-down rows above and the APEX GatedDeltaNet rows: the GatedDeltaNet `(2048,8192)` rows gain the most, moving from the single-tile `_k512`/`_k2048` bodies to the clean prefetched pipelined tile (`21.0%`, `14.2%` and `3.2%` ahead at `M=2048/8192/32768`, so `25.4`, `25.6` and `24.1` TFLOPS at ratios `1.33-1.50x`), and the shared-down rows take the single-tile `_k512` body at `M=2048` (`16.7%`) and the prefetched padding tile at `M=32768` (`10.0%`). The pattern the screen found is systematic rather than specific: these families still ran first-generation tiles that the pipelined rollout never revisited for those particular shapes, so the fix is reassignment, not a new mechanism.

The traversal knob was also swept on this type's starved `M=2048` keys as twins of both the deployed and the pipelined bodies. `group_m = 0` does not win on them (`1.14x` behind at `M=2048` and `1.8-4.4x` behind at the larger row counts), so no key here moves to it.

## Evidence

The Qwen4-Exp and GatedDeltaNet rows come from the backward harness under `~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q5k_v1.txt`.

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/hip_vs_baseline/pass11_ordbwd_qwen.json
~/tmp/torch-ggml-ops/hip_selection/           (per-key candidate campaign)
```

The original campaign evidence is:

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_narrow_q5_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_narrow_selected_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_narrow_scalar_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_narrow_selected_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_shared_swizzle4_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_shared_swizzle8_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_shared_swizzle4_after_25.json
```

The shared-down Q5_K body reaches 253 VGPRs but remains spill-free. A new experiment must reduce decode state or change the packed representation before reopening the closed geometry neighborhood.
