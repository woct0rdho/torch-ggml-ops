# GGTensile Grouped MMQ Forward Q2_K Experiment

## Scope

This record covers the routed gfx1151 Q2_K down projection for the DeepSeek expert bank, one GEMM per routed expert:

```text
X_g[M_g,K] @ W_g[K,N] -> Y_g[M_g,N]
```

## Final Results

`TFLOPS = 2*R*N*K / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Each row is the median over the benchmark's sampled expert partitions, and both implementations are timed on the same partitions.

| Batch | Logical shape `(R,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `(12288,4096,2048)` | 12.996 | 1.0881x | `grouped_mmq_fwd_q2_k_r12288_n4096_k2048_f32cc7d2292b1282` | `grouped_fwd_serial_q2_k_n4096_k2048_j32_j16` |
| 4 | `(49152,4096,2048)` | 14.407 | 1.1046x | `grouped_mmq_fwd_q2_k_r49152_n4096_k2048_686731e4c99c1ffd` | `grouped_fwd_serial_q2_k_n4096_k2048_j32_j16` |
| 16 | `(196608,4096,2048)` | 15.607 | 1.1761x | `grouped_mmq_fwd_q2_k_r196608_n4096_k2048_37ce03e143353b9d` | `grouped_fwd_serial_q2_k_n4096_k2048_j32` |

GGTensile is ahead on all 3 rows, with speedups from `1.0881x` to `1.1761x` (mean `1.1229x`).

These medians replace the earlier recorded values, which were paired against HIP bodies that have since been retuned.

## Accepted Experiments

### Dedicated Q2_K decoder and exact arithmetic association

Q2_K received a dedicated grouped forward identity rather than a renamed Q4_K or Q5_K path. The decoder maps each lane to one aligned 16-value group, shares the two-bit payload shift and scale/minimum pair across those values, reconstructs the nibble scale and minimum, and writes BF16 decoded weights to the existing WMMA leaf. Groups 0-5 use stored activation sums. Groups 6-7 issue all-ones integer WMMAs and reconstruct the missing sums without changing the producer contract.

The first reopened probe reproduced the installed rolled HIP arithmetic association: stored groups compute `Cd*d`, apply the missing-sum term inside the temporary for groups 6-7, multiply by `dB`, and apply the `dmin*sB` correction in the installed order. The 32-row association body is bitwise exact on the complete route and mutation matrix and remains the exactness control for structural work, though it is not competitive by itself.

### Partial LDS retirement and static group lowering

A dedicated exact schedule stages activation scale/sum reads first, then WMMA payloads, then four `d/dmin` reads, releasing the payload dependencies with `lgkmcnt(4)` while the correction reads remain pending and issuing `lgkmcnt(0)` immediately before the first dependent correction. It improved the exact structural parent to `0.7270x`, `0.7337x`, and `0.7922x` installed/candidate at B1/B4/B16 versus `0.7138x`, `0.7288x`, and `0.7866x` for the full-wait control.

Static group lowering then removed the scalar group loop and selected direct group bodies, and the distributed installed-style producer queued eight payload, scale-byte, and `d/dmin` loads per lane across all four waves, retired them with descending `vmcnt` waits, and wrote the compact 320-byte LDS rows. Pairing decoded payload stores with `ds_write2_b32` and the metadata stores with `ds_write2st64_b32` reduced the producer to 2,161 instructions while preserving the resource class. These mechanisms form the exact structural parent.

### Eight-wide correction dependency schedule

The dominant scheduling defect was that each output element completed its dependent `i32->f32`, scale product, activation-scale accumulation, and minimum-correction chain before the next independent element began. The exact same per-element arithmetic was regrouped into eight-wide phases. Stored-sum groups reuse the post-WMMA activation registers for eight independent products, and missing-sum groups use the otherwise idle `v98:v105` window so the persistent all-ones results remain intact.

The corrected body stays at 135 VGPRs, 40 SGPRs, 25,600 LDS bytes, 60 static WMMAs, and four barriers. Nine-repeat fitted-prior timing measured `15.7222`/`55.0022`/`215.3602 ms` versus installed `19.3646`/`65.9245`/`263.0708 ms`, or `1.2317x`/`1.1986x`/`1.2215x` at B1/B4/B16. Reversed-order 25-repeat confirmation measured `1.2283x`/`1.1972x`/`1.2219x`, and the search, captured, synthetic, sequential, and complete-call controls preserved the gain with minimum profile ratios above `1.17x` and bitwise-exact output throughout.

### Larger ownership after correction phasing

A composed 64-row body uses the distributed producer, paired stores, partial LDS retirement, and four-tile phased correction at 159 VGPRs, 40 SGPRs, 30,208 LDS bytes, 80 static WMMAs, and four barriers. It is rejected for B1/B4 but beats the 32/16 body at B16: 25-repeat confirmation measured `206.6285 ms` versus `265.1847 ms` for installed (`1.2834x`), with every learned/hash medoid between `1.2821x` and `1.2855x`. Search, captured, and synthetic B16 controls measured `1.2847x`, `1.2829x`, and `1.2915x`, and complete-call timing with fixed quantization measured `1.2764x`. J64 is retained for B16. Its four-tile epilogue was confirmed against one- and two-tile alternatives, and its dependency width two and priority two were confirmed against widths one and four and priority zero. No alternate epilogue is retained.

## Rejected Experiments

### Dynamic parents and unconditional static lowering

The dynamic 32-row and 64-row parents measured `0.7476x`/`0.7647x`/`0.8227x` and `0.6014x`/`0.7840x`/`0.8949x` installed/candidate at B1/B4/B16. Static group lowering alone improved to `0.7998x`/`0.8229x`/`0.8766x` but was not bitwise exact until the association probe was corrected. Both parents are rejected because the installed body was faster on every production shape.

### Alternative LDS strides and pipelining

The installed 400-byte decoded-row stride and a 336-byte variant with separate metadata banking both remained exact but regressed. The two-group arithmetic pipeline, the two-group partial-LDS dependency graph, and the compact next-group consumer prefetch were exact but lost to the single-group partial-wait parent. A second live group does not amortize its larger live state, and additional group lifetime is rejected with both full and descending partial waits.

### Schedule and epilogue alternatives

Standalone producer `s_clause 2`, activation `s_clause 7`, explicit `buffer_gl0_inv` on the barriers, epilogue priority zero, BF16 dependency widths one and four, and a two-tile-ahead epilogue were all exact and resource-identical but neutral or regressive. The retained epilogue point is one tile ahead at dependency width two and priority two.

### Pure J32, J128, and address reductions

Pure J32 removed the mixed-tail branch but was slower than mixed J32/J16 on every shape, even at B16. J128 at 239 VGPRs, 40 SGPRs, and 39,424 LDS bytes measured `239.1084 ms` versus `263.6021 ms` for installed (`1.1024x`), about 16% slower than J64, and is removed. VOPD accumulator initialization was neutral to regressive and is rejected as an independent mechanism. The shared packed-kernarg and paired route-bound reductions did not clear the Q4_K transfer gate, and the Q2 R3 guarded-stride/output-address reduction regressed B16 complete-call time by `1.529%`. Existing identities remain unchanged.

## Open Items

The R3 guarded-stride and output-address reduction was rejected under an older complete-call protocol, so a prequantized multiply-only re-screen is allowed. The common R1/R2 route-prologue gate, the J128 resource point, and the zero-bank probes have larger or better-explained negative evidence and are not reopened.

## Closure

The retained result is the width-16 Q2_K decoder with compact 320-byte LDS rows, the distributed producer with partial LDS retirement, the eight-wide correction schedule, J32/J16 ownership at B1/B4 and J64 ownership at B16.
