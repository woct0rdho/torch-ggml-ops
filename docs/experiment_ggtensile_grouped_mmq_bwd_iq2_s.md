# GGTensile Grouped MMQ Backward IQ2_S Experiment

## Scope And Contract

This record covers the isolated gfx1151 grouped non-paired IQ2_S backward kernel for the Qwen routed down projection. For each routed expert, the operation is:

```text
dY_g[M_g,2048] @ W_g[2048,512] -> dX_g[M_g,512]
```

The measured aggregate-row shapes are `R=16384`, `65536`, and `262144`. The packed expert bank is `[256,2048,164]`: each 256-value block occupies 82 bytes, each packed row contains two blocks, and each expert occupies 335,872 bytes. Inputs and outputs are contiguous BF16, accumulation is FP32 WMMA V1, and stores use BF16 round-to-nearest-even.

The grouped research ABI is:

```text
grad_output, packed_weight, grad_input,
expert_indices, expert_offsets, num_experts, rows, bytes_per_expert
```

IQ2_S uses a dedicated width-16 decoder. Each aligned group reconstructs two ten-bit codebook indices, loads two 64-bit entries from an assembly-local 1024-entry codebook, applies sign bits and the packed scale, rounds the decoded weights to BF16 in LDS, and uses the established FP32-WMMA/BF16 store path. The codebook is local read-only data and is not an ABI argument.

Timing uses the five Qwen learned medoids and complete-call latency including output allocation. Logical throughput is `2 * R * 512 * 2048 / (latency_ms * 1e9)`. The speedup ratio is HIP time divided by GGTensile time.

## Final Benchmark Results

The table shows the fastest qualified identity found for each measured aggregate-row shape. Candidate values use the latest disjoint inactive-M confirmation; HIP values are the established disjoint HIP controls used to normalize the final comparison. The table reports complete-call TFLOPS and does not include separate A/B timing columns.

| Matrix shape `(R,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(16384,512,2048)` | `ggsol_67d58c3f849e005f` | `12.613` | `1.5746x` |
| `(65536,512,2048)` | `ggsol_d82241ad1072e2f0` | `17.994` | `1.3060x` |
| `(262144,512,2048)` | `ggsol_8684bd239b31ec32` | `23.122` | `1.4412x` |

All three final identities were bitwise exact and faster than HIP under the final qualification and confirmation protocol.

## Final Kernel Profiles

| Kernel hash | Geometry and ownership | LDS and schedule | VGPR / SGPR | WMMAs | Barriers |
| --- | --- | --- | ---: | ---: | ---: |
| `ggsol_67d58c3f849e005f` | M128/N64, serial routes, `Mixed128_64` tail | SIA5/PGR2, pad8 single LDS, local codebook | `137 / 38` | 24 | 5 |
| `ggsol_d82241ad1072e2f0` | M128/N128, split64, full M128 tail | SIA5/PGR2, pad8 single LDS, local codebook | `210 / 38` | 32 | 3 |
| `ggsol_8684bd239b31ec32` | M128/N128, split64, full M128 tail | SIA5/PGR2, pad8 single LDS, local codebook | `210 / 38` | 32 | 3 |

The B1 profile uses 13,312 LDS bytes and 745 static VALU issues. B4 and B16 use 18,432 LDS bytes and 763 static VALU issues. All final artifacts are gfx1151 code-object-v5 wave32 kernels with zero private bytes, spills, scratch, calls, and dynamic stack.

## Accepted Kernel Experiments

### Dedicated IQ2_S decoder and identity

IQ2_S was given a strict grouped backward identity rather than being treated as Q2_K with different labels. The decoder emits the authoritative local codebook and keeps its codebook address and metadata state separate from the existing Q2_K, Q4_K, and Q5_K paths.

The initial pilot exposed three decoder-local address defects: destructive reuse of the second high-index nibble, a duplicated packed-row base for nonzero K rows, and a half-tile static coordinate in the N128 path. After correction, one-hot probes across representative K rows reproduced all 512 independently dequantized BF16 weights exactly. The final decoder passes the full packed-HIP and independent-reference checks.

### M128/N64 and M128/N128 ownership

The B1 parent uses M128/N64 because it avoids repeated codebook decoding without paying the M128/N128 accumulator envelope on sparse learned routes. B4 and B16 use M128/N128 with split64 ownership because larger aggregate rows amortize the additional accumulator and LDS state.

The final geometry identities pass exactness, route-tail coverage, mutation checks, and deterministic reruns. M128/N64 is retained for B1; M128/N128 split64 is retained for B4 and B16.

### Signed-dword codebook preapplication

The decoder applies four sign nibbles to loaded codebook dwords with the proven `v_perm_b32` selector construction, leaving one signed-byte extraction per element. This removes 48 static VALU issues from the mixed M128/M64 body without adding registers. A same-process bracket improved every learned medoid and reduced weighted latency by `3.01%` against the scalar-sign implementation. Signed-dword preapplication is part of every final identity.

### Workgroup-local codebook staging

The global decoder required two random 64-bit codebook loads per lane in every reduction iteration. The accepted implementation stages the full 8 KiB codebook into a disjoint LDS region once per workgroup and uses `ds_load_b64` for subsequent lookups. The decoded-B buffer and toggle remain unchanged; the added setup barrier and LDS footprint are explicit in the identity.

The staged implementation improved weighted latency by `9.57%`, `5.96%`, and `9.17%` at B1, B4, and B16 relative to global codebook lookup. It passed the complete packed-HIP matrix and all mutation controls. Local codebook staging is retained unconditionally.

### Final LDS, schedule, split, and tail choices

Post-staging comparisons retained SIA5/PGR2 with pad8 single-buffer LDS. The B1 kernel retains `Mixed128_64` because it improves the learned medoids at the small aggregate size. B4 and B16 retain full M128 tails because mixed tails regress their dominant medoids.

Split64 is retained for B4 and B16. It improves B4 weighted latency by `0.61%` over split16 and improves every B16 learned medoid over split32, with a smaller `0.07%` weighted gain. These choices are part of the final kernel identities rather than runtime-only policies.

### Decode extraction cleanup

The lane-half selector is hoisted out of the per-row preparation loop and equivalent shift/mask pairs use direct bitfield extraction. This removes six static VALU issues at N64 and eight at N128 without changing the resource class. Same-process brackets improved weighted latency by `0.58%`, `0.51%`, and `0.58%` at B1, B4, and B16. The cleanup is included in the final emitted streams.

### Inactive-M suppression

The final kernels guard wholly inactive waves and inactive 16-row M minitiles around the WMMA consumer while keeping codebook staging, packed decode, required LDS traffic, barriers, and masked global access uniform. This preserves correctness for active siblings and avoids consuming work for empty route tiles.

Disjoint five-warmup, 25-repeat confirmations improved retained-parent weighted latency by `2.25%` at B1, `2.60%` at B4, and `1.15%` at B16. Every learned medoid improved, with minimum parent-to-candidate ratios of `1.0182x`, `1.0239x`, and `1.0074x`. The suppression is retained in the final kernels.

## Rejected Kernel Experiments

### M64/N64 and B1 M128/N128 geometry

M64/N64 reduced accumulator state but repeated the IQ2_S decoder over more M tiles. B1 M128/N128 increased the accumulator and register envelope without offsetting the route-size cost. Both alternatives remained exact and spill-free but lost to B1 M128/N64 across the learned-route objective.

### Initial double-LDS pipeline variant

The first M128/N128 SIA4/swizzle8 double-LDS artifact failed exactness because IQ2_S metadata remained live in registers later reused as decoded-B LDS addresses. Moving the post-lookup values into dead qh/scale slots repaired the register lifetime, and the repaired mechanism passed correctness. It was not retained in the final layout because pad8 single LDS was faster after local codebook staging.

### Plain, padded, swizzled, and alternate SIA layouts

Plain LDS, pad16, swizzle4/8/16, and SIA2 controls were exact but slower or statistically flat against the selected pad8/SIA5 layout. B4 and B16 mixed tails also lost on their dominant medoids. Padded and swizzle16 double-LDS requests were rejected by the typed pipeline boundary because the two-buffer address toggle is implemented only for XOR-8; they were not benchmarked.

### Serial and smaller split factors

Serial ownership was poor on the larger learned route distributions. B4 split4 and split8, and B16 split8 and split16, were exact but slower than the retained split choices. Split64 is therefore restricted to the large-key M128/N128 identities; it is not generalized to B1.

### Global codebook lookup

The global-codebook path was exact and resource-clean, but it lost the workgroup-local codebook implementation by `9.57%`, `5.96%`, and `9.17%` on the weighted B1, B4, and B16 objectives. It is closed as a final implementation choice.

### FMA/output-modifier arithmetic replacement

Replacing the proven `(scale + 0.5) * 0.25` add/multiply sequence with an FMA/output-modifier form assembled and reduced one issue per row, but failed candidate, mutation, and independent-oracle comparisons. The proven arithmetic sequence was restored; no correctness waiver was accepted.

### Processor or metadata-only variants

Metadata-only changes that did not alter the generated executable were not retained as kernel identities. No performance result is assigned to a byte-identical artifact.

## Qualification Summary

The final B1, B4, and B16 kernels were independently regenerated and inspected. They passed exact packed-HIP comparison, independent BF16 reference checks, finite-output and full-row coverage checks, deterministic reruns, gradient and route mutations, active and inactive weight mutations, malformed-route sentinels, and non-aligned route tails.

The independent BF16 reference maximum absolute error is `0.015625`, with NRMSE below `8.1e-5` for all three aggregate sizes. Final resources are zero private bytes and spills, with the profile listed above. The selected result is the IQ2_S width-16 local-codebook decoder combined with signed-dword preapplication, SIA5/PGR2 pad8 LDS staging, per-size M/N ownership, split64 for B4/B16, and inactive-M suppression.
