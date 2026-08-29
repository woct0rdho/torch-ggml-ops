# GGTensile Grouped MMQ Backward Q5_K Experiment

## Scope And Contract

This record covers the isolated gfx1151 grouped non-paired Q5_K backward kernel for the Qwen expert down projection. For each routed expert, the operation is:

```text
dY_g[M_g,2048] @ W_g[2048,512] -> dX_g[M_g,512]
```

The measured aggregate-row shapes are `R=16384`, `65536`, and `262144`. The packed Q5_K bank is `[256,2048,352]`; each 256-value block occupies 176 bytes and each packed weight row occupies 352 bytes. Inputs and outputs are contiguous BF16, accumulation is FP32 WMMA, and output conversion is BF16 round-to-nearest-even.

The grouped kernel argument contract contains `grad_output`, `packed_weight`, `grad_input`, `expert_indices`, `expert_offsets`, `num_experts`, `rows`, and `bytes_per_expert`. One serial workgroup owns a column tile and walks row tiles for a route. Invalid expert IDs, empty or reversed ranges, negative starts, and out-of-range ends are inert. Final row tiles mask reads and stores while packed decode, LDS barriers, and WMMA remain uniform.

The kernel uses direct packed Q5_K decode, including low- and high-bit payloads and scale/minimum reconstruction, followed by decoded-weight LDS, FP32 WMMA, and BF16 stores. Timing uses complete-call allocation in both HIP and GGTensile paths with fitted Qwen learned-route medoids. Logical throughput is `2 * R * 512 * 2048 / (latency_ms * 1e9)`, and the speedup ratio is HIP time divided by GGTensile time.

## Final Benchmark Results

The table shows the fastest qualified kernel found for each aggregate-row shape across the full experiment record.

| Matrix shape `(R,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(16384,512,2048)` | `ggsol_98b9ce678d4cb09f` | `12.187` | `1.4027x` |
| `(65536,512,2048)` | `ggsol_a45127a1814ae728` | `18.054` | `1.2208x` |
| `(262144,512,2048)` | `ggsol_c518830e3a59d061` | `22.185` | `1.2822x` |

The final rows use the fastest established HIP-normalized baseline. Later inactive-M comparisons are retained below as parent-versus-candidate evidence because they did not replace the three-shape bracket.

## Final Kernel Profiles

| Kernel hash | Geometry and ownership | Decode and memory policy | VGPR / SGPR | LDS bytes | WMMAs / barriers |
| --- | --- | --- | ---: | ---: | ---: |
| `ggsol_98b9ce678d4cb09f` | M128/N64, serial, `Mixed128_64` tail | SIA5/PGR2/PLR1, padded single LDS, packed extraction, hoisted nibble shift, scalar metadata | `140 / 35` | `5,120` | `24 / 4` |
| `ggsol_a45127a1814ae728` | M128/N128, `SplitRoutes16` | SIA5/PGR2/PLR1, padded single LDS, packed extraction, hoisted nibble shift, scalar metadata | `216 / 35` | `10,240` | `32 / 2` |
| `ggsol_c518830e3a59d061` | M128/N128, `SplitRoutes16` | SIA5/PGR2/PLR1, padded single LDS, packed extraction, hoisted nibble shift, scalar metadata | `216 / 35` | `10,240` | `32 / 2` |

All selected artifacts target gfx1151 wave32 code object v5 with zero private storage and zero VGPR/SGPR spills. The B1 profile uses an M64 body only for the restricted short-row tail; B4 and B16 retain the M128/N128 split16 body.

## Accepted Kernel Experiments

### Q5_K packed decode and grouped route contract

The strict Q5_K identity carries the 176-byte block geometry and 352-byte packed row stride. The selected decoder uses packed extraction, hoisted nibble shifts, scalar metadata loads, current-tile packed-weight prefetch, and one padded LDS buffer. It preserves exact Q5_K arithmetic through scale/minimum reconstruction, decoded-weight LDS, WMMA accumulation, and BF16 stores.

The selected implementation passes full and partial route matrices, non-aligned tails, sparse and repeated expert IDs, deterministic reruns, gradient and route mutations, active and inactive weight mutations, malformed offsets, invalid experts, and untouched sentinels. Independent BF16-reference NRMSE remains below `0.01`.

### Shape-specific geometry and route ownership

M128/N64 serial ownership is retained at B1 because the fitted short-route population benefits from the smaller column tile. Its `Mixed128_64` tail uses M64 only where that reduces the fitted objective. The disjoint bracket measured mixed tails at `2.8353 ms` versus `3.1084 ms` for pure M128/N64, a `9.63%` improvement.

M128/N128 with SplitRoutes16 is retained at B4 and B16. On the confirmation medoids, SplitRoutes16 beat SplitRoutes8 at B4 with `7.8985 ms` versus `8.0076 ms` and at B16 with `24.9272 ms` versus `25.4524 ms`. Split16 passed the same exactness and resource gates as the parent geometries.

### Q5 pipeline and padded LDS

SIA5 with packed extraction and hoisted nibble shifts is retained on all three shapes. The B1 control screen rejected inline shifts (`2.9642 ms`), scalar extraction (`2.9626 ms`), swizzle4 (`3.0641 ms`), swizzle8 (`3.0044 ms`), and plain LDS (`3.3103 ms`) against the `2.9121 ms` padded SIA5 parent.

Double-LDS SIA4 was exact after its address and lifetime constraints were repaired, but its extra LDS traffic and synchronization did not provide a stable advantage. The selected layouts therefore use one padded LDS buffer with eight-byte row padding.

### Inactive-M consumer suppression

The accepted suppression guards wholly inactive waves and inactive 16-row M minitiles around the WMMA consumer while keeping packed low/high decode, required LDS traffic, barriers, and masked global access uniform. It adds no solution field and preserves the resource envelope.

B1 was retained after disjoint confirmation improved weighted latency from `2.8357` to `2.7996 ms` (`1.0129x` parent throughput). The 75-repeat top-up confirmed `2.8741` to `2.8215 ms` (`1.0186x`) with a robust 95% candidate-minus-parent interval of `[-2.743,-0.923]%`.

B4 was retained after disjoint confirmation improved weighted latency from `7.8109` to `7.6883 ms` (`1.0159x`). The 75-repeat top-up confirmed `7.9329` to `7.7282 ms` (`1.0265x`) with interval `[-3.835,-1.345]%`. All B1 and B4 confirmation medoids remained bitwise exact.

## Rejected Kernel Experiments

### Initial and oversized geometries

The M128/N128 SIA2 pilot and M64 primary were exact where qualified but lost the fitted objective. The M256/N64 control was also exact, but reached 238 VGPRs and regressed to `8.6587 ms` at B4 and `31.3354 ms` at B16. The final geometry set is therefore limited to the B1 mixed M128/N64 body and the B4/B16 M128/N128 split16 body.

### N64 metadata and lane-sharing variants

The N64 vector-metadata artifact assembled at 140 VGPRs and removed three static VMEM instructions, but failed packed-HIP, gradient/route/weight mutation, and independent-reference comparisons. N64 packed lane sharing failed the same correctness gates. Vector metadata was exact on an N128 double-LDS control, so the rejection is specific to the N64 sharing combinations rather than the vector load mechanism in isolation.

### Decode, LDS, and split alternatives

Inline nibble shifts, scalar extraction, plain LDS, swizzle4, and swizzle8 lost the selected padded SIA5 control. Split2 and Split4 did not beat the final larger-key owner; Split8 was superseded by Split16 on the confirmation medoids. Split32 was exact but regressed to `7.9267 ms` at B4 and `24.8375 ms` at B16, while helping only low-weight skewed profiles.

The B4 search-bank order between SIA5 and double-LDS SIA4 was inconsistent: a separate screen favored double-LDS at `7.8789` versus `8.0056 ms`, while a 15-repeat same-process bracket favored SIA5 at `7.8958` versus `7.9175 ms`, only a `0.27%` gap. Disjoint confirmation and the split16 bracket closed this layout choice in favor of the selected single-LDS path.

### Inactive-M suppression at B16

The B16 search-bank suppression result improved weighted latency from `24.7676` to `24.4248 ms` (`1.0140x`), but the disjoint 25-repeat bracket reversed it to `24.9318` to `25.0716 ms` (`0.9944x`). A 75-repeat top-up then measured a near tie in the other direction, `25.0824` to `25.0314 ms` (`1.0020x`), with an interval crossing zero.

The 200-repeat top-up resolved the inconsistency against suppression: `25.0288` to `25.0906 ms` (`0.9975x`), with a robust 95% candidate slowdown interval of `[0.015,0.479]%`. Four of five medoids were slower despite bitwise equality, so B16 retains the parent consumer path. The B1/B4 retention and B16 rejection are the final suppression decisions.

### Device J128 ownership

The typed J128 device-row-task kernel passed independent reproducibility, full B1 correctness, malformed-route controls, tail sentinels, and the independent BF16 reference. It assembled at `220 VGPR / 44 SGPR / 8,192 B LDS`, with 32 WMMAs, two barriers, and no private storage or spills.

It was rejected against the selected serial GGTensile parent, not against HIP. The search-bank comparison regressed complete-call weighted latency from `2.5524` to `2.8141 ms` (`0.9070x` parent/task throughput); the disjoint confirmation regressed `2.7770` to `2.8545 ms` (`0.9728x`). The dominant medoid regressed to `0.9609x`, while only low-weight long-tail medoids improved. Fixed J128 was removed, and J64, mixed task sizes, and ownership-conditioned task/decode variants were not opened.

### Other arithmetic and ownership changes

No approximate accumulation, cross-workgroup reduction, atomics, persistent traversal, prepared-weight shadow, alternate arithmetic order, or Q4_K-specific dependency batching was retained. These mechanisms either changed the Q5_K arithmetic or ownership contract or failed to establish a measured advantage under the fitted objective.

## Qualification Summary

The final B1, B4, and B16 kernels pass exact packed-HIP comparison, independent BF16-reference checks, deterministic reruns, full-row coverage, non-aligned tails, gradient and route mutations, active and inactive weight mutations, malformed-route sentinels, and invalid-route controls.

| Key | Rows | HIP BF16 differences | Independent NRMSE | Deterministic differences / tail writes |
| --- | ---: | ---: | ---: | ---: |
| B1 | `16384` | `0` | `6.47e-5` | `0 / 0` |
| B4 | `65536` | `0` | `8.78e-5` | `0 / 0` |
| B16 | `262144` | `0` | `8.06e-5` | `0 / 0` |

Two independent generation, build, and inspection roots reproduce the selected source and HSACO byte-for-byte. The retained result is the direct Q5_K decoder with packed extraction, hoisted nibble shifts, scalar metadata, padded single-LDS storage, shape-specific serial or SplitRoutes16 ownership, the B1-only mixed tail, and inactive-M suppression only for B1 and B4.
