# GGTensile Grouped MMQ Backward Q5_K Experiment

## Scope

This record covers the routed gfx1151 Q5_K input gradient for the Qwen down projection, one GEMM per routed expert.

## Final Results

`TFLOPS = 2*R*N*K / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Each row is the median over the benchmark's sampled expert partitions, and both implementations are timed on the same partitions.

| Batch | Logical shape `(R,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `(16384,2048,512)` | 11.084 | 1.0899x | `grouped_mmq_bwd_q5_k_r16384_n2048_k512_0b56d860a4fc8ae6` | `grouped_bwd_row_task_q5_k_n2048_k512_mt128_nt64_s2_abar` |
| 4 | `(65536,2048,512)` | 20.469 | 1.2217x | `grouped_mmq_bwd_q5_k_r65536_n2048_k512_59d4269ed118bb8c` | `grouped_bwd_row_task_q5_k_n2048_k512_mt128_nt64_s2_abar` |
| 16 | `(262144,2048,512)` | 22.011 | 1.0127x | `grouped_mmq_bwd_q5_k_r262144_n2048_k512_3c3bd36fe4f6d0e8` | `grouped_bwd_row_task_q5_k_n2048_k512_mt256_nt64_s2` |

GGTensile is ahead on all 3 rows, with speedups from `1.0127x` to `1.2217x` (mean `1.1081x`).

These medians replace the earlier recorded values, which were paired against HIP bodies that have since been retuned.

## Accepted Experiments

### Q5_K packed decode and grouped route contract

The strict Q5_K identity carries the 176-byte block geometry and 352-byte packed row stride. The selected decoder uses packed extraction, hoisted nibble shifts, scalar metadata loads, current-tile packed-weight prefetch, and one padded LDS buffer. It reconstructs scale and minimum values through decoded-weight LDS, WMMA accumulation, and BF16 stores.

### Shape-specific geometry and route ownership

M128/N64 serial ownership is retained at B1 because the fitted short-route population benefits from the smaller column tile. Its `Mixed128_64` tail uses M64 only where that reduces the fitted objective. The disjoint bracket measured mixed tails at `2.8353 ms` versus `3.1084 ms` for pure M128/N64, a `9.63%` improvement.

M128/N128 with SplitRoutes16 is retained at B4 and B16. On the confirmation medoids, SplitRoutes16 beat SplitRoutes8 at B4 with `7.8985 ms` versus `8.0076 ms` and at B16 with `24.9272 ms` versus `25.4524 ms`.

### Q5 pipeline and padded LDS

SIA5 with packed extraction and hoisted nibble shifts is retained on all three shapes. The B1 control screen rejected inline shifts (`2.9642 ms`), scalar extraction (`2.9626 ms`), swizzle4 (`3.0641 ms`), swizzle8 (`3.0044 ms`), and plain LDS (`3.3103 ms`) against the `2.9121 ms` padded SIA5 parent.

Double-LDS SIA4 was repaired, but its extra LDS traffic and synchronization did not provide a stable advantage. The selected layouts therefore use one padded LDS buffer with eight-byte row padding.

### Inactive-M consumer suppression

The accepted suppression guards wholly inactive waves and inactive 16-row M minitiles around the WMMA consumer while keeping packed low/high decode, required LDS traffic, barriers, and masked global access uniform. It adds no solution field and preserves the resource envelope.

B1 was retained after disjoint confirmation improved weighted latency from `2.8357` to `2.7996 ms` (`1.0129x` parent throughput). The 75-repeat top-up confirmed `2.8741` to `2.8215 ms` (`1.0186x`) with a robust 95% candidate-minus-parent interval of `[-2.743,-0.923]%`.

B4 was retained after disjoint confirmation improved weighted latency from `7.8109` to `7.6883 ms` (`1.0159x`). The 75-repeat top-up confirmed `7.9329` to `7.7282 ms` (`1.0265x`) with interval `[-3.835,-1.345]%`.

## Rejected Experiments

### Initial and oversized geometries

The M128/N128 SIA2 pilot and M64 primary lost the fitted objective. The M256/N64 control reached 238 VGPRs and regressed to `8.6587 ms` at B4 and `31.3354 ms` at B16. The final geometry set is therefore limited to the B1 mixed M128/N64 body and the B4/B16 M128/N128 split16 body.

### N64 metadata and lane-sharing variants

The N64 vector-metadata artifact assembled at 140 VGPRs and removed three static VMEM instructions, but the N64 sharing combinations are rejected: the same mechanism worked on an N128 double-LDS control, so the rejection is specific to those combinations rather than to the vector load mechanism in isolation.

### Decode, LDS, and split alternatives

Inline nibble shifts, scalar extraction, plain LDS, swizzle4, and swizzle8 lost the selected padded SIA5 control. Split2 and Split4 did not beat the final larger-key owner. Split8 was superseded by Split16 on the confirmation medoids. Split32 regressed to `7.9267 ms` at B4 and `24.8375 ms` at B16, while helping only low-weight skewed profiles.

The B4 search-bank order between SIA5 and double-LDS SIA4 was inconsistent: a separate screen favored double-LDS at `7.8789` versus `8.0056 ms`, while a 15-repeat same-process bracket favored SIA5 at `7.8958` versus `7.9175 ms`, only a `0.27%` gap. Disjoint confirmation and the split16 bracket closed this layout choice in favor of the selected single-LDS path.

### Inactive-M suppression at B16

The B16 search-bank suppression result improved weighted latency from `24.7676` to `24.4248 ms` (`1.0140x`), but the disjoint 25-repeat bracket reversed it to `24.9318` to `25.0716 ms` (`0.9944x`). A 75-repeat top-up then measured a near tie in the other direction, `25.0824` to `25.0314 ms` (`1.0020x`), with an interval crossing zero.

The 200-repeat top-up resolved the inconsistency against suppression: `25.0288` to `25.0906 ms` (`0.9975x`), with a robust 95% candidate slowdown interval of `[0.015,0.479]%`. Four of five medoids were slower, so B16 retains the parent consumer path. The B1/B4 retention and B16 rejection are the final suppression decisions.

### Device J128 ownership

The J128 device-row-task artifact assembled at `220 VGPR / 44 SGPR / 8,192 B LDS`, with 32 WMMAs, two barriers, and no private storage or spills.

It was rejected against the selected serial GGTensile parent, not against HIP. The search-bank comparison regressed complete-call weighted latency from `2.5524` to `2.8141 ms` (`0.9070x` parent/task throughput). The disjoint confirmation regressed `2.7770` to `2.8545 ms` (`0.9728x`). The dominant medoid regressed to `0.9609x`, while only low-weight long-tail medoids improved. Fixed J128 was removed, and J64, mixed task sizes, and ownership-conditioned task/decode variants were not opened.

### Other arithmetic and ownership changes

No approximate accumulation, cross-workgroup reduction, atomics, persistent traversal, prepared-weight shadow, alternate arithmetic order, or Q4_K-specific dependency batching was retained. These mechanisms either changed the Q5_K arithmetic or ownership contract or failed to establish a measured advantage under the fitted objective.

## Closure

The retained result is the direct Q5_K decoder with packed extraction, hoisted nibble shifts, scalar metadata, padded single-LDS storage, shape-specific serial or SplitRoutes16 ownership, the B1-only mixed tail, and inactive-M suppression only for B1 and B4.
