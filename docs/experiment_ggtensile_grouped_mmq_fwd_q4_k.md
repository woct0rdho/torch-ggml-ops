# GGTensile Grouped MMQ Forward Q4_K Experiment

## Scope

This record covers the routed gfx1151 Q4_K down projection, one GEMM per routed expert:

```text
X_g[M_g,K] @ W_g[K,N] -> Y_g[M_g,N]
```

## Final Results

`TFLOPS = 2*R*N*K / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Each row is the median over the benchmark's sampled expert partitions, and both implementations are timed on the same partitions.

| Batch | Logical shape `(R,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `(16384,2048,512)` | 18.072 | 0.9887x | `grouped_mmq_fwd_q4_k_r16384_n2048_k512_c95d0acee91c7c55` | `grouped_fwd_serial_q4_k_n2048_k512_j32` |
| 4 | `(65536,2048,512)` | 23.078 | 0.9943x | `grouped_mmq_fwd_q4_k_r65536_n2048_k512_61ba675510b0ed6f` | `grouped_fwd_serial_q4_k_n2048_k512_j64` |
| 16 | `(262144,2048,512)` | 24.385 | 1.0001x | `grouped_mmq_fwd_q4_k_r262144_n2048_k512_c6a7f10afcf493b2` | `grouped_fwd_serial_q4_k_n2048_k512_j64` |

HIP is ahead on 2 of the 3 rows, with speedups from `0.9887x` to `1.0001x` (mean `0.9944x`), and within 0.5% of parity on the remaining 1.

These medians replace the earlier recorded values, which were paired against HIP bodies that have since been retuned.

## Accepted Experiments

### Direct-global routing and arithmetic control

The first grouped Q4_K artifact emits one-wave serial GEMM ownership with a 16-row macro tile, masked activation reads, and masked BF16 stores for arbitrary `M_g`. It assembles at 88 VGPRs, 32 SGPRs, zero LDS, and 16 static WMMAs, and matched the installed grouped HIP path exactly on a real Qwen bank with route lengths of 1, 15, 16, and 3 rows. This body remains the routing and arithmetic control. The production tiled schedule is not bitwise identical to the one-wave control. The measured production difference is `3.05e-5` maximum absolute error and `6.04e-9` RMS, far below the independent reference envelope.

### Decoded-weight LDS bodies and the mixed 64/32 tail

The first decoded-weight LDS body reused the typed dense Q4_K decode and scaled WMMA emitters under serial routed ownership. It staged a 128-by-64 tile at 239 VGPRs, 40 SGPRs, 38,400 LDS bytes, 32 static WMMAs, and four barriers but reached only `0.734x`, `0.917x`, and `0.960x` installed/candidate at B1/B4/B16. Parameterizing the layout and register plan by row fragments produced a 64-by-64 tile at 159 VGPRs, 40 SGPRs, 29,184 LDS bytes, 16 static WMMAs, and four barriers. Independent scale/minimum extraction and metadata reads between the low and high WMMAs reduced the 64-row low-half wait ladder to `7,5,3,1` and improved the weighted results to `1.060x`, `0.986x`, and `0.971x`.

The retained mixed 64/32 tail body decodes weights once per K block, then branches only activation staging, local reads, WMMA correction, and masked stores when the final routed tile has at most 32 rows. Its `a1d4-p2` schedule uses independent groups of four BF16 roundings at priority two and restores priority before serial traversal continues. The confirmation measured `2.142 ms` versus `2.503 ms` for HIP at B1 (`1.168x` weighted) and `6.073 ms` versus `6.202 ms` at B4 (`1.021x`), with the independent search bank reproducing `1.205x` and `1.025x`. The complete-call audit with the fixed quantizer measured `1.2167x`, `1.0311x`, and `1.0006x` installed/candidate. The dominant medoids contain many 32-row-or-smaller remainders, which explains the transfer. Some low-weight large-group medoids remain slower, with confirmation minima of `0.931x` at B1 and `0.953x` at B4.

### B16 128-row control

The scheduled 128-row `a1d2-p2` body is the better large-group control at B16. Its reversed-order 25-repeat confirmation measured `21.756 ms` versus `21.823 ms` for HIP (`1.003x`), with every medoid at least `0.997x`. The movement is below the normal promotion margin, so this identity is retained as a resource-clean correctness and schedule control rather than a durable large-key win.

## Rejected Experiments

### Four-wave direct-global geometry

Grouping four WaveN owners into a 128-thread workgroup while retaining direct-global arithmetic improved the smallest key only marginally and regressed the large keys: uniform `R=262144` moved from `99.42` to `108.09 ms` and the boundary route to `105.46 ms`, leaving the candidate 2.1 to 5.3 times slower than HIP. Four-wave workgroup formation without decoded-weight or activation reuse is not an actionable optimization.

### Route-persistent full-K decoded weights

Decoding both K512 blocks once before the row loop into two immutable LDS images, with disjoint activation storage, produced a 57,856-byte LDS artifact at 239 VGPRs, 40 SGPRs, 64 static WMMAs, and eight barriers. An operand-order error in the image-pointer restore was repaired and the body became bitwise equal to the selected 128-row parent for route lengths `1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 256` and for uniform, skewed, sparse, and boundary distributions at all three shapes. It nevertheless failed the performance discriminator before confirmation: parent/full-weight body ratios were `0.805x` to `0.849x` at B1, `0.860x` to `0.882x` at B4, and `0.891x` to `0.921x` at B16, including an approximately 12% regression on the B16 uniform route with the strongest decode-amortization premise. The larger immutable LDS footprint and doubled static body outweigh the removed re-decodes, so the mechanism is rejected and its speculative lowering removed.

### Routed-prologue and address reductions

The packed-kernarg, paired cumulative-offset, and exact address reductions rebuilt deterministically and matched the parent bitwise on all boundary, repeated-ID, sparse/skewed, first-route, unaligned-index, invalid-expert, and invalid-offset controls. Their Q4_K B1 candidate-time movements were `+0.030%`, `-0.086%`, and `-0.083%`, far below the advancement gate, so no confirmation or cross-format transfer was run. Direct-to-LDS is unavailable: the configured gfx1151 assembler rejects the buffer and LLVM spellings and reports `global_load_lds_dword` unsupported.

### Zero-bank initialization

The later paired-source audit does not transfer a zero-bank hoisting candidate here. The decoded-LDS Q4_K body already initializes its zero bank once per row tile, so it has no repeated per-projection lifetime matching the paired finding.

## Open Items

The R1-R3 route and address reductions may be retimed on the current multiply-only surface as a diagnostic requalification. The route-persistent full-K rejection is not reopened: its direct body evidence already showed a large LDS and resource regression.

## Closure

The retained result is the grouped Q4_K direct decoder with padded single-LDS storage: the mixed 64/32 `a1d4-p2` body at B1/B4 and the scheduled 128-row `a1d2-p2` body at B16.
