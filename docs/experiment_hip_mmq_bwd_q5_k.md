# HIP MMQ Backward Q5_K Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ input-gradient kernels for Q5_K weights.

Backward shapes are written `(M, in_features, out_features)`, matching the weight's `(N,K) = (out_features, in_features)`. Beyond the Qwen narrow and shared-down families measured first, the type carries the QSA attention key/value and output, shared-expert gate/up and the chunked language model head of the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (hidden size 2560), and the GatedDeltaNet `in_proj_qkv` of the Qwen3.6-35B-A3B (APEX-I-Mini) checkpoint, at the training token counts of a sequence length 2048 batch (B1/B4/B16). the head is called in chunks of 64, 128 and 256 rows.

GatedDeltaNet `out_proj` is deferred because wiring it needs the activation permutation.

## Final kernel result

| Family | `(M,K,N)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| Narrow K/V/gate/up | `(2048,2048,512)` | 20.862 | 0.898x | `dense_bwd_q5_k_mt128_nt128_ki32_full_k512` |
| Narrow K/V/gate/up | `(8192,2048,512)` | 22.190 | 0.903x | `dense_bwd_q5_k_mt128_nt128_ki32_full_k2048` |
| Narrow K/V/gate/up | `(32768,2048,512)` | 23.240 | 0.946x | `dense_bwd_q5_k_mt128_nt128_ki32_full_k2048` |
| Shared down | (2048,512,2048) | 23.290 | 1.028x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_sw16` |
| Shared down | (8192,512,2048) | 26.970 | 1.129x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_sw16` |
| Shared down | (32768,512,2048) | 28.047 | 1.156x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_sw16` |
| QSA key/value | (2048,2560,512) | 29.207 | 1.393x | `dense_bwd_q5_k_pipea_nt4_ki64_mw2_sw16_prefetch` |
| QSA key/value | (8192,2560,512) | 28.242 | 1.081x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_sw16` |
| QSA key/value | (32768,2560,512) | 30.542 | 1.148x | `dense_bwd_q5_k_pipea_nt4_ki64_mw2_sw16_prefetch` |
| QSA output | (2048,6144,2560) | 33.643 | 1.370x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw4_sw16` |
| QSA output | (8192,6144,2560) | 30.265 | 1.171x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_sw16` |
| QSA output | (32768,6144,2560) | 30.010 | 1.145x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_sw16` |
| Shared-expert gate/up | (2048,2560,640) | 30.356 | 1.418x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_sw16` |
| Shared-expert gate/up | (8192,2560,640) | 30.025 | 1.107x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_sw16` |
| Shared-expert gate/up | (32768,2560,640) | 33.861 | 1.193x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw4_sw16` |
| GatedDeltaNet QKV APEX-I-Mini | (2048,2048,8192) | 21.114 | 1.242x | `dense_bwd_q5_k_mt128_nt128_ki32_full_k512` |
| GatedDeltaNet QKV APEX-I-Mini | (8192,2048,8192) | 22.915 | 1.271x | `dense_bwd_q5_k_mt128_nt128_ki32_full_k2048` |
| GatedDeltaNet QKV APEX-I-Mini | (32768,2048,8192) | 23.869 | 1.295x | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_sw16` |
| Language model head | (64,2560,248320) | 17.798 | 1.222x | `dense_bwd_q5_k_pipesplit_m64_s4` |
| Language model head | (128,2560,248320) | 21.841 | 1.724x | `dense_bwd_q5_k_pipesplit_m128_s8` |
| Language model head | (256,2560,248320) | 25.548 | 1.348x | `dense_bwd_q5_k_pipesplit_m256_s32` |

The `Kernel` column names the deployed body for each exact key. It is the fastest built body whose output is bitwise equal to the reference body in the per-key candidate campaign. The shared-down rows show the largest per-sample spread in the matrix (`1.20-1.53` max/min here). The narrow rows stay at `1.05-1.07`.

## Kernel implementation

The retained body uses four wave32 waves, exact 128x128/K32 ownership where applicable, decoded-weight LDS staging. The build carries two tuned `mt128_nt128_ki32_full` variants (`_k512`, `_k2048`), the `full_k2048_scalar_extraction` control, and five legacy generic bodies. The tuned variants and the scalar control were timed on every deployment key, and the deployed body per key is the fastest of those. The language model head uses split-contraction bodies on the pattern of the Q6_K and Q8_0 records, with the partial reduction the deployment already carries. Q5-specific packed-row and signed-scale state stays separate from Q4_K.

## Optimization log

### Tiled redesign and shape specialization

The first four-wave Qwen body replaced scalar decode and serial 16x16 ownership with cooperative extraction, LDS staging, multiple WMMA accumulators, and exact shape/row geometry. Exact Q5_K wrappers then removed runtime bounds and address state while retaining bounds-safe fallback code.

The exact Q5_K matrix improved its generic controls by `11.66-32.60%`. The smallest narrow row count is the main extraction discriminator. Larger rows favor the packed path.

### Split-contraction head and the projection probe

The head contracts `248320` values with a result of `(M,2560)`, so its non-split grid is `1 x 40` workgroups, far below the machine's wave slots, and the contraction splits scale it directly: `m64_s4` beats `m64_s2` by `1.37x`, `m128_s8` beats `m128_s16` by `1.036x`, and `m256_s32` beats `m256_s16` by `1.036x`. Those three bodies deploy, and their partials are `(slices, M, 2560)` f32, which stays at `1.3-2.6 MiB`.

The same mechanism was then measured on the longest-contraction projection, the APEX-I-Mini `in_proj_qkv` `(M,2048,8192)`, where the grid is already `16 x 16 x 1` at `M=2048` and `256 x 16 x 1` at `M=32768`. Split-contraction loses there by `1.52-1.60x` at `M=2048` and `2.77-3.15x` at `M=32768`, because the partial workspace becomes `(slices, M, 2048)` f32, up to `1 GiB`, and the parallelism the split buys is parallelism the grid already had. Split-contraction is therefore deployed for the chunked head only.

The projection keys were screened against `_k2048`, `_k512` and the scalar-extraction control: `_k2048` wins every point except the APEX-I-Mini `in_proj_qkv` at `M=2048`, where `_k512` is `3.6%` ahead, and the scalar control is `2-6%` behind the winner on every point it was timed.

### Extraction and LDS controls

Q5_K reused the Q4_K framework for width-16 low/high decode and bounded packed prefetch. Prefetch improved the initial port by `12-24%`. A universal padded layout regressed Q5_K and was rejected, and aligned fragment loads regressed Q5_K by about `2-4%` in cross-format controls.

Narrow Q5 retained scalar extraction only at 2,048 rows, where it improved `17.41%`. It regressed `2.34%` and `6.52%` at 8,192 and 32,768 rows. Shared-down Q5 retained its four-BF16 XOR layout after an isolated swizzle8 improvement did not survive the complete matrix.

The accepted body keeps packed/decode temporaries dead before long WMMA phases. Cross-iteration packed prefetch, K64, double buffering, decoded-weight LDS caching, and broad swizzle changes either extended register lifetimes or reduced residency.

### Closed directions

The current source closes global J64, I128, alternate workgroup sizes, split-K, GSU, Stream-K, persistent workgroups, generic local-load rules, and compiler-managed prefetch arrays. Q5-specific results must not be inferred from Q4_K or Q3_K because packed high-bit reconstruction changes the register and LDS cost model.

### Pipelined tile

The Q2_0 and Q6_K records established a two-tile backward stage order, one barrier per contraction stage, and this record is where it pays most. The pipelined twin of the deployed `_k2048` geometry, at the 64-value stage with two and four row tiles per wave, beats the deployed bodies on every measured point: `1.31x/1.08x/1.09x` on the QSA value rows, `1.42x/1.11x/1.19x` on the shared-expert gate/up rows, `1.37x/1.17x/1.15x` on the QSA output rows, and `1.03x/1.13x/1.16x` on the APEX-I-Mini `(512,2048)` rows, against `M=2048/8192/32768` at eight repeats.

On the APEX-I-Mini `(2048,8192)` rows the pipelined variant beats the eight-column tile by `1.15x` at `M=2048` and `1.02x` at `M=32768`. The `M=2048` row keeps the `_k512` wrapper it deploys and the `M=8192` row was not measured against the deployed body, so only the `M=32768` row takes the pipelined tile. The narrow-result rows prefer the two-row-tile variant and the wide rows the four-row-tile one at `M=2048`.

The shared-expert down `(2048,512)` rows keep their single-tile body (`0.73x` at `M=2048`, where the pipelined variant's 256-row blocks starve the grid. `1.10x` at `M=32768` was left undeployed for the want of a confirmation run). The swizzle-only twin of the deployed 8 KiB tile loses (`1.03-1.21x` behind) on this type, so the pipeline is deployed with the swizzle the deployed bodies already use.

### Pipelined split-contraction head

The head keys were the one place the earlier records measured the tile pipeline on a single-tile body: the split-contraction body's stage loop was still the two-barrier order. The same pipelined body now carries the split window (`dense_mmq_pipelined_splitk_body`, sharing the projection tile's implementation), and against the deployed split bodies at eight repeats it is `1.40x` ahead at `M=64`, `1.04x` at `M=128` and `1.14x` at `M=256`, all three keys taking it. Its prefetched decode is what the projection rows took as well: `1.07x` and `1.06x` over the group-decoding pipelined tile on the QSA value rows, so those two rows moved to the prefetched twin too.

## Resources

The split head bodies use `74/110/198 VGPR` at `M=64/128/256` with `4 KiB` LDS.

The retained Q5_K bodies use `247 VGPR / 17 SGPR / 8 KiB LDS` for narrow scalar extraction and `253 VGPR / 16 SGPR / 8 KiB LDS` for shared down.

## Evidence

The Qwen4-Exp and GatedDeltaNet rows come from the backward harness under `~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q5k_v1.txt`, which uses the deployed preparation, the same `torch.mm` BF16 baseline and the same paired timing as the official protocol. The split-contraction rows include the partial reduction in the timed region.

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/hip_vs_baseline/pass11_ordbwd_qwen.json
~/tmp/torch-ggml-ops/hip_selection/           (per-key candidate campaign)
tools/configs/hip_deployment.json             (deployed body per key)
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
