# GGTensile Grouped MMQ Backward IQ2_S Experiment

## Scope

This record covers the routed gfx1151 non-paired IQ2_S input gradient for the Qwen down projection, one GEMM per routed expert.

## Final Results

`TFLOPS = 2*R*N*K / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Each row is the median over the benchmark's sampled expert partitions, and both implementations are timed on the same partitions.

| Batch | Logical shape `(R,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `(16384,2048,512)` | 12.152 | 1.0055x | `grouped_mmq_bwd_iq2_s_r16384_n2048_k512_456cd11e362196dc` | `grouped_bwd_row_task_iq2_s_n2048_k512_mt128_nt64_s2_abar` |
| 4 | `(65536,2048,512)` | 20.829 | 1.1434x | `grouped_mmq_bwd_iq2_s_r65536_n2048_k512_e700b5ce6f216c1e` | `grouped_bwd_row_task_iq2_s_n2048_k512_mt128_nt64_s2_abar` |
| 16 | `(262144,2048,512)` | 23.629 | 1.0699x | `grouped_mmq_bwd_iq2_s_r262144_n2048_k512_688fdac4a24cf1b7` | `grouped_bwd_row_task_iq2_s_n2048_k512_mt256_nt64_s2` |

GGTensile is ahead on all 3 rows, with speedups from `1.0055x` to `1.1434x` (mean `1.0729x`).

These medians replace the earlier recorded values, which were paired against HIP bodies that have since been retuned.

## Accepted Experiments

### Dedicated IQ2_S decoder and identity

IQ2_S was given a strict grouped backward identity rather than being treated as Q2_K with different labels. The decoder emits the authoritative local codebook and keeps its codebook address and metadata state separate from the existing Q2_K, Q4_K, and Q5_K paths.

The initial pilot exposed three decoder-local address defects: destructive reuse of the second high-index nibble, a duplicated packed-row base for nonzero K rows, and a half-tile static coordinate in the N128 path. After correction, one-hot probes across representative K rows reproduced all 512 independently dequantized BF16 weights exactly.

### M128/N64 and M128/N128 ownership

The B1 parent uses M128/N64 because it avoids repeated codebook decoding without paying the M128/N128 accumulator envelope on sparse learned routes. B4 and B16 use M128/N128 with split64 ownership because larger aggregate rows amortize the additional accumulator and LDS state.

M128/N64 is retained for B1. M128/N128 split64 is retained for B4 and B16.

### Signed-dword codebook preapplication

The decoder applies four sign nibbles to loaded codebook dwords with the proven `v_perm_b32` selector construction, leaving one signed-byte extraction per element. This removes 48 static VALU issues from the mixed M128/M64 body without adding registers. A same-process bracket improved every learned medoid and reduced weighted latency by `3.01%` against the scalar-sign implementation. Signed-dword preapplication is part of every final identity.

### Workgroup-local codebook staging

The global decoder required two random 64-bit codebook loads per lane in every reduction iteration. The accepted implementation stages the full 8 KiB codebook into a disjoint LDS region once per workgroup and uses `ds_load_b64` for subsequent lookups. The decoded-B buffer and toggle remain unchanged. The added setup barrier and LDS footprint are explicit in the identity.

The staged implementation improved weighted latency by `9.57%`, `5.96%`, and `9.17%` at B1, B4, and B16 relative to global codebook lookup. Local codebook staging is retained unconditionally.

### Final LDS, schedule, split, and tail choices

Post-staging comparisons retained SIA5/PGR2 with pad8 single-buffer LDS. The B1 kernel retains `Mixed128_64` because it improves the learned medoids at the small aggregate size. B4 and B16 retain full M128 tails because mixed tails regress their dominant medoids.

Split64 is retained for B4 and B16. It improves B4 weighted latency by `0.61%` over split16 and improves every B16 learned medoid over split32, with a smaller `0.07%` weighted gain. These choices are part of the final kernel identities rather than runtime-only policies.

### Decode extraction cleanup

The lane-half selector is hoisted out of the per-row preparation loop and equivalent shift/mask pairs use direct bitfield extraction. This removes six static VALU issues at N64 and eight at N128 without changing the resource class. Same-process brackets improved weighted latency by `0.58%`, `0.51%`, and `0.58%` at B1, B4, and B16. The cleanup is included in the final emitted streams.

### Inactive-M suppression

The final kernels guard wholly inactive waves and inactive 16-row M minitiles around the WMMA consumer while keeping codebook staging, packed decode, required LDS traffic, barriers, and masked global access uniform. This preserves correctness for active siblings and avoids consuming work for empty route tiles.

Disjoint five-warmup, 25-repeat confirmations improved retained-parent weighted latency by `2.25%` at B1, `2.60%` at B4, and `1.15%` at B16. Every learned medoid improved, with minimum parent-to-candidate ratios of `1.0182x`, `1.0239x`, and `1.0074x`. The suppression is retained in the final kernels.

## Rejected Experiments

### M64/N64 and B1 M128/N128 geometry

M64/N64 reduced accumulator state but repeated the IQ2_S decoder over more M tiles. B1 M128/N128 increased the accumulator and register envelope without offsetting the route-size cost. Both alternatives were spill-free but lost to B1 M128/N64 across the learned-route objective.

### Initial double-LDS pipeline variant

The first M128/N128 SIA4/swizzle8 double-LDS artifact failed exactness because IQ2_S metadata remained live in registers later reused as decoded-B LDS addresses. Moving the post-lookup values into dead qh/scale slots repaired the register lifetime, and the repaired mechanism passed correctness. It was not retained in the final layout because pad8 single LDS was faster after local codebook staging.

### Plain, padded, swizzled, and alternate SIA layouts

Plain LDS, pad16, swizzle4/8/16, and SIA2 controls were slower or statistically flat against the selected pad8/SIA5 layout. B4 and B16 mixed tails also lost on their dominant medoids. Padded and swizzle16 double-LDS requests were rejected by the typed pipeline boundary because the two-buffer address toggle is implemented only for XOR-8. They were not benchmarked.

### Serial and smaller split factors

Serial ownership was poor on the larger learned route distributions. B4 split4 and split8, and B16 split8 and split16, were slower than the retained split choices. Split64 is therefore restricted to the large-key M128/N128 identities. It is not generalized to B1.

### Global codebook lookup

The global-codebook path was resource-clean, but it lost the workgroup-local codebook implementation by `9.57%`, `5.96%`, and `9.17%` on the weighted B1, B4, and B16 objectives. It is closed as a final implementation choice.

### FMA/output-modifier arithmetic replacement

Replacing the proven `(scale + 0.5) * 0.25` add/multiply sequence with an FMA/output-modifier form assembled and reduced one issue per row, but the form is rejected. The proven arithmetic sequence was restored.

### Processor or metadata-only variants

Metadata-only changes that did not alter the generated executable were not retained as kernel identities. No performance result is assigned to a byte-identical artifact.

## Closure

The retained result is the local-codebook width-16 IQ2_S decoder with signed-dword preapplication, SIA5/PGR2 pad8 LDS staging, M128/N64 with a mixed 128/64 tail at B1 and M128/N128 split64 with a full M128 tail at B4/B16, and inactive-M suppression.
