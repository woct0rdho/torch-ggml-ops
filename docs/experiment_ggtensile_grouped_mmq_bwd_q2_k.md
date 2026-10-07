# GGTensile Grouped MMQ Backward Q2_K Experiment

## Scope

This record covers the routed gfx1151 Q2_K input gradient for the DeepSeek down projection, one GEMM per routed expert.

## Final Results

`TFLOPS = 2*R*N*K / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Each row is the median over the benchmark's sampled expert partitions, and both implementations are timed on the same partitions.

| Batch | Logical shape `(R,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `(12288,4096,2048)` | 15.211 | 1.1999x | `grouped_mmq_bwd_q2_k_r12288_n4096_k2048_81eb73416d91e2ec` | `grouped_bwd_row_task_q2_k_n4096_k2048_mt128_nt64_s3_g4_abar` |
| 4 | `(49152,4096,2048)` | 21.243 | 1.0213x | `grouped_mmq_bwd_q2_k_r49152_n4096_k2048_f0166c97652b9f00` | `grouped_bwd_row_task_q2_k_n4096_k2048_mt128_nt64_s3_g4_abar` |
| 16 | `(196608,4096,2048)` | 26.355 | 1.0376x | `grouped_mmq_bwd_q2_k_r196608_n4096_k2048_f91249735df1e8e0` | `grouped_bwd_row_task_q2_k_n4096_k2048_mt256_nt64_s3_g4_ki64` |

GGTensile is ahead on all 3 rows, with speedups from `1.0213x` to `1.1999x` (mean `1.0862x`).

These medians replace the earlier recorded values, which were paired against HIP bodies that have since been retuned.

## Accepted Experiments

### Dedicated Q2_K decoder

Q2_K received a strict grouped backward identity and a dedicated packed reader rather than being treated as Q4_K or Q5_K. The decoder maps each lane to one aligned 16-value group, performs the shared two-bit extraction, reconstructs scale/minimum values, and writes BF16 decoded weights to the existing WMMA leaf.

The M128/N128 SIA2 pilot's initial resource profile was 190 VGPRs, 35 SGPRs, 8,192 LDS bytes, 32 WMMAs, and two barriers.

### Shape-specific geometry and tail ownership

The final geometry follows the measured route-size tradeoff: M64/N64 for B1, M128/N64 for B4, and M128/N128 for B16. Eight-byte LDS padding is retained because unpadded rows materially regress both short and medium shapes. The B4 `Mixed128_64` tail selects M64 only through 64 rows. B16 uses SplitRoutes32 to avoid the dominant two-tile route loop.

Geometry, padding, SIA choice, and route split are encoded in the selected identities rather than inferred at timing time.

### DependencyBatch4 decode scheduling

The Q2-specific `DependencyBatch4` schedule reuses four dead `valu_b` register pairs without changing the resource envelope. It improves B1 by `2.14%` and B4 by `6.38%` against their serial parents. The same schedule regresses B16 by `1.46%` and loses every fitted medoid, so B1 and B4 retain DependencyBatch4 and B16 retains serial decode.

### Inactive-M consumer suppression

The accepted suppression guards wholly inactive waves and inactive 16-row M minitiles around the WMMA consumer while keeping packed decode, LDS traffic, barriers, and masked global access uniform. It uses dead post-prologue scalar state and adds no solution field or resource cost.

Disjoint confirmation improved retained-parent weighted latency by `13.91%` at B1, `1.60%` at B4, and `4.09%` at B16. Nine of ten B4 medoids improved and every B16 medoid improved, so suppression remains part of the accepted kernel design.

## Rejected Experiments

### M128/N128 SIA2 and losing geometries

The initial M128/N128 SIA2 pilot was not competitive at B1. M64/N64 was rejected for larger keys because repeated Q2 scale/minimum reconstruction outweighed its lower accumulator cost. M128/N128 was rejected for B1 because its register envelope did not pay back on sparse small routes. These alternatives remain valid correctness controls but are not selected identities.

### Double LDS and alternate layouts

The first M128/N64 double-LDS artifact failed exactness, determinism, mutations, and independent-reference checks because Q2 metadata aliased a register later used for decoded-B LDS addressing. Reserving the address register repaired the lifetime, but double LDS was later slower than the padded single-LDS path and was not retained.

Plain LDS, SIA2, SIA5/PGR1, and swizzle variants were slower or statistically flat. The final path retains padded single LDS. No alternate layout is selected.

### M-tail threshold policy

A threshold-only tail artifact was invalid because the inherited M64 branch emitted one tile by construction. A typed Q2-specific route loop repaired correctness, but the candidate regressed to `39.9595 ms` versus `37.2586 ms` for the existing `<=64` policy. Rows 65-127 favor one masked M128 tile over two M64 tiles, so the broader threshold was rejected.

### Split64 and smaller split factors

Split32 improved the B16 parent. Split64 lost a balanced bracket by `0.30%`, including the dominant learned profile and every hash medoid. Split64 was removed. SplitRoutes32 is the maximum retained split.

### DependencyBatch4 at B16

The B16 DependencyBatch4 candidate regressed `1.46%` against serial decode and lost every medoid. A temporary dependency-width-two endpoint also lost B16 by `0.51%`. Neither schedule is retained for B16.

### Arithmetic replacement

An FMA/output-modifier replacement for the proven scale/minimum arithmetic assembled and reduced one issue per row, but the form is rejected. The proven add/multiply sequence was restored.

### Unrepresented or broad searches

Row-task ownership, U4, broad split sweeps, and N128 reopening after the selected parent were not accepted as generic follow-ups. Existing controls either failed the resource/timing premise or did not remove repeated Q2 scale/minimum reconstruction. No new identity was created for them.

## Open Items

Focused B4 requalification: B4 is the only selected shape that may need retuning. Start from the existing B4 M128/N64 `SecondaryTile`/serial-body alternatives and their DependencyBatch4 decode, with the corrected HIP control and the fitted learned/hash objective, before changing geometry. Do not broaden the split sweep until that comparison is resolved.

Tail-boundary check: recheck the existing B4 `Mixed128_64` and threshold candidates around the 64/128-row boundary on the current protocol. Reopen only those exact threshold identities. A new row-task or broad ownership search needs a separate mechanism that removes Q2 scale/minimum reconstruction work. B1 and B16 do not justify retuning, and the B16 DependencyBatch4 rejection, SplitRoutes32 selection, and broad geometry closures remain closed.

## Closure

The retained result is the width-16 Q2_K decoder with padded LDS, shape-specific M/N ownership, DependencyBatch4 at B1/B4, serial decode at B16, SplitRoutes32 for B16, and inactive-M suppression.
