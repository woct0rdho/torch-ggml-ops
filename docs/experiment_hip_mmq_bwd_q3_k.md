# HIP MMQ Backward Q3_K Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ input-gradient kernels for Q3_K weights.

Backward shapes are written `(M,N,K)`, matching the weight's `(N,K) = (out_features, in_features)`. Beyond the Qwen query and narrow projections measured first, the type carries the QSA attention query, key/value and shared-expert gate/up projections of the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (hidden size 2560) and the GatedDeltaNet `in_proj_qkv` and `in_proj_z` projections of both checkpoints in use, at the training token counts of a sequence length 2048 batch (B1/B4/B16).

GatedDeltaNet `out_proj` is deferred because wiring it needs the activation permutation. `token_embd.weight` `(248320,2560)` is an embedding gather rather than a multiply and stays on the GGUF embedding module.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| Query/query gate | `(2048,8192,2048)` | 26.938 | 1.583x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| Query/query gate | `(8192,8192,2048)` | 27.307 | 1.530x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| Query/query gate | `(32768,8192,2048)` | 24.114 | 1.356x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| Narrow key | `(2048,512,2048)` | 27.333 | 1.193x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| Narrow key | `(8192,512,2048)` | 29.327 | 1.223x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| Narrow key | `(32768,512,2048)` | 29.704 | 1.224x | `dense_bwd_q3_k_pipea_nt64_ki64_mw4_sw16_prefetch` |
| QSA query | `(2048,12288,2560)` | 27.795 | 1.205x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| QSA query | `(8192,12288,2560)` | 26.728 | 1.048x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| QSA query | `(32768,12288,2560)` | 24.123 | 0.934x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| QSA key/value | `(2048,512,2560)` | 29.261 | 1.399x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| QSA key/value | `(8192,512,2560)` | 30.108 | 1.181x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| QSA key/value | `(32768,512,2560)` | 32.202 | 1.221x | `dense_bwd_q3_k_pipea_nt64_ki64_mw4_sw16_prefetch` |
| Shared-expert gate/up | `(2048,640,2560)` | 32.766 | 1.537x | `dense_bwd_q3_k_pipea_nt64_ki64_mw4_pad8_prefetch` |
| Shared-expert gate/up | `(8192,640,2560)` | 34.165 | 1.254x | `dense_bwd_q3_k_pipea_nt64_ki64_mw4_sw16_prefetch` |
| Shared-expert gate/up | `(32768,640,2560)` | 35.876 | 1.282x | `dense_bwd_q3_k_pipea_nt64_ki64_mw4_sw16_prefetch` |
| GatedDeltaNet QKV | `(2048,10240,2560)` | 28.327 | 1.222x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet QKV | `(8192,10240,2560)` | 26.894 | 1.057x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet QKV | `(32768,10240,2560)` | 23.832 | 0.959x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet Z | `(2048,6144,2560)` | 28.563 | 1.254x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet Z | `(8192,6144,2560)` | 28.234 | 1.116x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet Z | `(32768,6144,2560)` | 25.635 | 1.035x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet Z APEX-I-Mini | `(2048,4096,2048)` | 30.971 | 1.847x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| GatedDeltaNet Z APEX-I-Mini | `(8192,4096,2048)` | 27.546 | 1.522x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet Z APEX-I-Mini | `(32768,4096,2048)` | 27.504 | 1.501x | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_sw16_prefetch` |

## Kernel implementation

Two bodies carry the twenty-four keys. The first is the pipelined tile of the Q2_0, Q4_0 and Q5_0 records (`csrc/ck/mmq_backward_pipelined.cuh`), which this type was the first to build: a 128-thread workgroup, four rows of 16 and four result columns of 16 per wave, `n_tiles * 16` result columns, a 64-value contraction stage, two shared tiles with one barrier per stage, and register double buffering of the activation fragments. Its 128-row result block is `M_TILES_PER_WAVE = 2`, and the prefetched payload form deploys on twenty-two of the keys. The second is the padded single-tile `mt128_nt128_ki32_full_narrow` body of the first redesign, which keeps the two keys whose 12288- or 10240-wide contraction at `M=32768` prefers it.

Q3_K's payload is a 110-byte block: a 32-byte high plane, a 64-byte low plane and 12 bytes of scale metadata for `QK_K = 256` weights. A value is one low nibble, two high bits and one six-bit scale, times the block's fp16 `d`. The pipelined tile's decode takes one sixteen-value group per pass and resolves it through the width-16 cooperative form, or through the prefetched form that reads the planes as whole `uint4` words before decoding.

## Optimization log

### First tiled redesign

The original body decoded one packed value at a time into a 16x16 tile and sustained only about `0.05-0.13x` BF16 throughput. The retained redesign added four wave32 waves, LDS-staged decoded weights, multiple WMMA accumulator tiles, cooperative pair/quad/width-16 decode, and measured row/type-specific ownership.

Representative first redesign results included query Q3_K M32768 moving from `1,108.314 ms` to `162.884 ms`. The same four-wave structure became the Q3_K foundation for narrow and query shapes.

### Extraction, prefetch, and LDS layout

Wide Q3_K packed extraction, two-row prefetch, and the eight-BF16 XOR LDS layout were retained for the query body. Removing those controls regressed by `1.06-11.78%` depending on the measured point. Narrow Q3_K does not prefetch because its 110-byte block layout made the path slower.

K-loop unrolling, activation-half double buffering, generic padding, decoded-weight LDS caching, and broad local-load rules were rejected. The accepted Q3 path keeps packed state bounded to the active decode phase. Keeping the next iteration's packed fragments live across WMMA extended register lifetimes without a timing benefit.

### Qwen4-Exp and GatedDeltaNet shapes

The new shapes span contraction lengths from `512` to `12288` and output widths from `2048` to `2560`. The padded body covered them first at `22.3-27.5 TFLOPS`, inside the band of the six rows measured first (`22.6-25.3`). The pipelined tile below then took twenty of the twenty-four keys.

The rate is not uniform in the contraction length: the short-contraction key, value and shared-expert shapes run at `24.3-27.5 TFLOPS` while the long-contraction query and QKV shapes run at `22.3-24.2`. The BF16 `torch.mm` baseline does not pay the packed decode, so the ratio against it crosses one inside this set: the narrow Qwen4 shapes are above it (`1.14-1.15x` at `M=2048`) and the longest contractions are below it (`0.93x`). That is the representation cost this record already names, and bringing it down needs a cheaper decode or a denser weight layout rather than another tiling sweep.

### Packaged-kernel controls

The source-built HSACO conversion produced a `+0.56%` initial geometric movement and a `+1.12%` embedded/bundle bracket movement, while embedded controls themselves drifted by `+1.04%`. Q3 query showed `2.9-7.0%` placement-sensitive movement without a device semantic change. Warm standalone modules, normalized ISA, and sequential controls are required before treating a timing change as a kernel result.

### Swizzle-only twin of the padded tile

The retained `_full_narrow` body carries `lds_padding = 8` and no LDS swizzle. Swizzle-only twins (8 KiB rather than 10 KiB) were screened on two families at `M=2048` and `32768`: padding wins, `1.08-1.29x` at `sw8` and `1.18-1.35x` at `sw16`, so this body keeps its padded tile. Evidence: `~/tmp/torch-ggml-ops/qwen4_fwd/bwd_swz_q3q4.txt`.

### Pipelined tile

The pipelined stage order was screened here on the same pattern as Q5_K and Q6_K: on the narrow-result shared-expert gate/up rows the 64-value-stage, four-row-tile body is `1.37x` and `1.20x` ahead of the deployed body at `M=2048` and `32768`, and on the QSA key/value rows it is `1.13x` ahead at `M=32768` while losing `0.77x` at `M=2048`, where its 256-row blocks leave too few workgroups. Those three rows took the pipelined body, and the rest of the family kept the padded tile on the strength of that `M=2048` loss and of the eight-column pipelined twins, which lose `1.06-1.13x` everywhere they were measured.

### The 128-row pipelined tile

That screen built the pipelined twin at `M_TILES_PER_WAVE = 4` only, so its `M=2048` loss is a property of the 256-row block rather than of the stage order: every type tuned after this record deploys the same body at 128 rows. The 128-row twin is `1.03-1.24x` ahead of the padded body on twenty of these keys - `21-22%` on the QSA query, GatedDeltaNet QKV and GatedDeltaNet Z rows, `17%` on the APEX-I-Mini rows at the two larger row counts, and `3-16%` on the narrow-result rows - while the 256-row twin keeps the shared-expert gate/up rows, where it is `22%` and `20%` ahead of the padded body. Two keys keep the padded body: the QSA query and the GatedDeltaNet QKV at `M=32768`, where it is `1.3%` and `1.0%` ahead.

### Prefetched decode and the shared-tile rotation

For the 128-row tile the catalog carries the prefetched decode, which reads the payload planes as `uint4` before decoding: it wins on sixteen of the twenty changed keys (up to `5%` at `M=2048`, less at the larger row counts) and ties elsewhere, and the group-decoding twin it ties with is not built. The deeper three-tile rotation screened on Q2_0, Q4_0, IQ4_XS, Q5_K and Q8_0 was screened here as well and loses `1.03-1.21x`, so the body keeps two shared tiles.

## Resources

The deployed pipelined bodies use `208 VGPR / 22 SGPR` at 16 KiB (`mw2`) and `216 VGPR / 22 SGPR` at 16 KiB (`mw4`), spill-free. The padded single-tile body keeps the last two keys at `216 VGPR / 17 SGPR / 10 KiB LDS`.

### Padding on the pipelined tile

The pipelined tile's layout was chosen when the tile was first built - sixteen-value swizzle chunks, no padding - and the chunk was later swept on this body without revisiting padding. The layout is now screened over every deployed dense-backward key: the padding twin of each deployed body (eight values of row padding, no swizzle) is timed against it at the key's own row counts, four repeats per block, and every screen winner is then re-timed at eight repeats over two blocks in both measurement orders, so a key moves only when the padding body wins in both directions.

Padding is not a general replacement for the swizzle. It wins where the resident grid is thin and the result is narrow - the `M=2048` rows and the 512- and 640-wide results - and loses on the wide results at the largest row counts, where the decoded tile's shared-memory traffic is high enough that the swizzle's bank pattern still pays. The split is per key rather than per type: on this type the four `M=2048` rows and two `M=8192` rows move: `15.3%` on the GatedDeltaNet `in_proj_z` eye, `10.6%` on the narrow key and `8.6%` on the query row at that row count, `6.9%` on the shared-expert gate/up, `5.9%` on the narrow key at `M=8192` and `3.8%` on the QSA key/value row there. The wide query row at `M=32768`, the GatedDeltaNet eye at `M=32768` and the QSA key/value row at `M=2048` keep the swizzle.

A second layout round then screened three further combinations on every officially measured body - sixteen-value padding, four-value padding, and eight-value padding with an eight-value swizzle chunk - at the deployed row counts. Nothing survives the confirmation: each candidate measures within `1-3%` of the deployed tile in one measurement order and loses in the other, so all three are rejected and none is built. The layout neighborhood is closed at the combinations the deployment uses.

### Reassignment of the sub-parity rows

The tables' slowest rows were then revisited as a group. For every key whose ratio to BF16 sat at or below parity, each catalog body of the same type whose exact dimensions and tile geometry can express the shape was timed against the deployed body, four repeats per block, and every winner was re-timed at eight repeats over two blocks in both measurement orders. The pattern the screen found is systematic rather than specific: these families still ran first-generation tiles that the pipelined rollout never revisited for those particular shapes, so the fix is reassignment rather than a new mechanism. On this type the two wide rows at `M=32768` moved off the `_full_narrow` tile onto the pipelined bodies the same shapes already use at the smaller row counts: `2.8%` on the 10240-wide GatedDeltaNet row (swizzled, prefetched, `23.8` TFLOPS) and `2.2%` on the 12288-wide query row (padded, prefetched, `24.1` TFLOPS). Both remain below BF16 (`0.959x` and `0.934x`), which is this type's structural position on a 12288-wide result: the activation operand is re-read once per 64-column result block, so the wide rows are the most bandwidth-bound shapes in the matrix.

## Evidence

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q3k_v1.txt
~/tmp/torch-ggml-ops/hip_vs_baseline/pass11_ordbwd_qwen.json
~/tmp/torch-ggml-ops/hip_selection/           (per-key candidate campaign)
```

The original campaign evidence is:

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_narrow_q5_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb0_folded_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_selected_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_scalar_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_no_prefetch_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_no_swizzle_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_selected_after_25.json
```

The source log's shared-body and packaging controls are:

```text
~/tmp/torch-ggml-ops/mmq_bwd_baseline_primary_sequential.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb0_folded_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_post_ds4_p1_control_9.json
~/tmp/torch-ggml-ops/mmq_bwd_pre_bundle.json
~/tmp/torch-ggml-ops/mmq_bwd_post_bundle.json
~/tmp/torch-ggml-ops/mmq_bwd_embedded_pre_control_25.json
~/tmp/torch-ggml-ops/mmq_bwd_bundle_control_25.json
~/tmp/torch-ggml-ops/mmq_bwd_embedded_post_control_25.json
```

The current Q3_K kernel record closes global J64, I128, alternate workgroup sizes, split-K, persistent workgroups, speculative prefetch, and broad swizzle changes for the existing packed representation. The pipelined tile below supersedes the prefetch and swizzle items on that list: the activation prefetch, the swizzle-only shared tile and the stage overlap are deployed, while split-K, the global J64/I128 ownership and the workgroup-size sweep stay closed.
