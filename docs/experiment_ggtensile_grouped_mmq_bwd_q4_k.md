# GGTensile Grouped MMQ Backward Q4_K Experiment

## Scope

This record covers the routed gfx1151 Q4_K input gradient for the Qwen down projection, one GEMM per routed expert.

## Final Results

`TFLOPS = 2*R*N*K / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Each row is the median over the benchmark's sampled expert partitions, and both implementations are timed on the same partitions.

| Batch | Logical shape `(R,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `(16384,2048,512)` | 12.343 | 1.2674x | `grouped_mmq_bwd_q4_k_r16384_n2048_k512_58259cc5470024c6` | `grouped_bwd_row_task_q4_k_n2048_k512_mt128_nt64_s3` |
| 4 | `(65536,2048,512)` | 19.708 | 1.1847x | `grouped_mmq_bwd_q4_k_r65536_n2048_k512_0615d2b1df5db67d` | `grouped_bwd_row_task_q4_k_n2048_k512_mt128_nt64_s3` |
| 16 | `(262144,2048,512)` | 22.014 | 1.0489x | `grouped_mmq_bwd_q4_k_r262144_n2048_k512_d056ae54e7d11a1d` | `grouped_bwd_row_task_q4_k_n2048_k512_mt128_nt64_s3` |

GGTensile is ahead on all 3 rows, with speedups from `1.0489x` to `1.2674x` (mean `1.1670x`).

These medians replace the earlier recorded values, which were paired against HIP bodies that have since been retuned.

## Accepted Experiments

### Grouped Q4_K identity and arithmetic reuse

The grouped implementation uses a dedicated problem and route contract while reusing the validated ordinary Q4_K packed reader, scale/minimum reconstruction, decoded-weight LDS layout, WMMA leaf, wait schedule, and BF16 store conversion. Expert rebasing and interleaved route strides are handled by the grouped lowering.

This arithmetic boundary remains accepted. The initial performance geometry was superseded by the shape-specific selections below.

### Shape-specific geometry and route ownership

M128/N64 with serial ownership is retained at B1 because short and sparse routes favor the lower accumulator footprint. M128/N128 with SplitRoutes4 and SplitRoutes8 is retained at B4 and B16 because larger routes amortize the wider body. The B1 `Mixed128_64` tail uses an M64 body only where it improves the fitted route objective.

Geometry and split ownership are encoded in the selected identities.

### DependencyBatch4 decode schedule

The Q4_K `DependencyBatch4` schedule reuses dead register pairs without increasing the final resource class. It is retained on all three selected keys after fitted-medoid timing checks. Unbatched decode and dependency-width alternatives did not provide a stable advantage.

### Inactive-M consumer suppression

The accepted suppression guards wholly inactive waves and inactive 16-row M minitiles around the WMMA consumer while leaving packed decode, required LDS reads, barriers, and masked global access uniform. It adds no solution field and preserves the resource envelope.

Disjoint confirmation improved retained-parent weighted latency by `2.70%` at B1 and `2.71%` at B4. A longer B16 confirmation reversed the initial short-screen result and improved weighted latency by `1.00%`. Suppression is retained across B1, B4, and B16.

## Rejected Experiments

### M64 primary geometry

M64 primary bodies reduced accumulator state but repeated decode and route traversal work. They lost the selected M128 parents on the fitted objective. M64 remains useful only as the restricted B1 mixed-tail body.

### Unbatched and alternate decode schedules

Unbatched Q4_K decode and dependency-width alternatives were slower or inconsistent against DependencyBatch4. No alternate decode schedule is retained.

### Double-LDS SIA4 alternatives

Double-LDS SIA4 variants were repaired, but the added LDS traffic and synchronization did not beat the padded single-LDS parents. They are rejected as final layouts.

### Nonselected split factors

B4 split2, split8, split16, and larger alternatives did not beat SplitRoutes4. B16 split2, split4, split16, and SplitRoutes32 did not beat SplitRoutes8 after balanced timing. The final split factors remain shape-specific.

### Mixed tails outside B1

A mixed M128/M64 tail was useful at B1 but regressed B4 and B16. B4's direct parent bracket was `0.9854x` candidate/parent throughput, while B16 had no useful short-route population in its dominant medoid. Mixed tails are rejected for those keys.

### Initial inactive-M B16 result

The first nine-repeat B16 suppression screen measured a small regression, so B16 was temporarily left on its parent. The subsequent full confirmation improved every medoid and produced a robust candidate-minus-parent interval of `[-1.512,-0.462]%`. The short-screen rejection is superseded by the longer confirmation, and B16 suppression is accepted.

### Device row tasks

The typed J128 device-row-task kernel assembled at 208 VGPRs, 44 SGPRs, 10,240 LDS bytes, 32 WMMAs, and two barriers.

It was rejected by disjoint fitted timing: complete-call weighted latency regressed from `2.4790` to `2.5800 ms`, or `0.9608x` parent/task throughput. Kernel-only weighted throughput was `0.9724x`, and the dominant medoid regressed to `0.9477x`. J64, mixed task sizes, and ownership-conditioned task/decode variants were not opened.

### Arithmetic and broad ownership changes

No approximate accumulator, cross-workgroup reduction, atomics, persistent traversal, prepared-weight shadow, or alternate arithmetic order was retained. These mechanisms either changed the arithmetic/ownership contract or lacked a measured fitted-prior premise after the selected direct kernels were qualified.

## Closure

The retained result is the grouped Q4_K direct decoder with SIA5/PGR2/PLR1 padded LDS, DependencyBatch4 decode, shape-specific serial or split ownership, the B1-only mixed tail, and inactive-M suppression.
