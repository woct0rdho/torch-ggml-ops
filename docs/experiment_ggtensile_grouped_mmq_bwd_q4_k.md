# GGTensile Grouped MMQ Backward Q4_K Experiment

## Scope And Contract

This record covers the isolated gfx1151 grouped non-paired Q4_K backward kernel for the Qwen expert down projection. For each routed expert, the operation is:

```text
dY_g[M_g,2048] @ W_g[2048,512] -> dX_g[M_g,512]
```

The measured aggregate-row shapes are `R=16384`, `65536`, and `262144`. The packed Q4_K bank is `[256,2048,288]`; `dY` and `dX` are contiguous BF16 with logical shapes `[R,2048]` and `[R,512]`. Routes can have arbitrary runtime heights, including short, non-aligned, sparse, repeated, and boundary ranges.

The grouped research ABI contains `grad_output`, `packed_weight`, `grad_input`, `expert_indices`, `expert_offsets`, `num_experts`, `rows`, and `bytes_per_expert`. One serial workgroup owns a column tile and walks the row tiles for one route. Invalid expert IDs, empty or reversed ranges, negative starts, and out-of-range ends are inert. Final row tiles mask reads and stores while decode, LDS barriers, and WMMA remain uniform.

The kernel uses direct packed Q4_K decode, FP32 WMMA accumulation, and BF16 round-to-nearest-even stores. The grouped lowering owns route loads, expert rebasing, row traversal, and tail predicates; ordinary Q4_K arithmetic, LDS, local reads, WMMA emission, waits, and stores are reused only where their contracts match.

Timing uses fitted Qwen learned-route medoids with complete-call allocation included. Logical throughput is `2 * R * 512 * 2048 / (latency_ms * 1e9)`. The speedup ratio is HIP time divided by GGTensile time.

## Final Benchmark Results

The table shows the fastest qualified HIP-normalized kernel found for each measured aggregate-row shape. It contains only the matrix shape, kernel hash, GGTensile speed, and speedup requested for this record.

| Matrix shape `(R,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(16384,512,2048)` | `ggsol_356ef05b9c33365d` | `13.698` | `1.8458x` |
| `(65536,512,2048)` | `ggsol_9950e632db681be3` | `18.655` | `1.3146x` |
| `(262144,512,2048)` | `ggsol_14b7c7089cdc2480` | `23.089` | `1.3565x` |

The final table uses the fastest established HIP-normalized values. The later inactive-M confirmations are recorded below as parent-to-candidate evidence because they did not constitute a replacement three-shape HIP bracket.

## Final Kernel Profiles

| Kernel hash | Geometry and ownership | Decode and LDS policy | VGPR / SGPR | LDS bytes | WMMAs / barriers |
| --- | --- | --- | ---: | ---: | ---: |
| `ggsol_356ef05b9c33365d` | M128/N64, serial, `Mixed128_64` tail | SIA5/PGR2/PLR1, padded single LDS, DependencyBatch4 | `136 / 35` | `5,120` | `24 / 4` |
| `ggsol_9950e632db681be3` | M128/N128, SplitRoutes4 | SIA5/PGR2/PLR1, padded single LDS, DependencyBatch4 | `208 / 35` | `10,240` | `32 / 4` |
| `ggsol_14b7c7089cdc2480` | M128/N128, SplitRoutes8 | SIA5/PGR2/PLR1, padded single LDS, DependencyBatch4 | `208 / 35` | `10,240` | `32 / 4` |

All final artifacts are gfx1151 code-object-v5 wave32 kernels with zero private storage, spills, scratch, calls, and dynamic stack. Eight-byte LDS row padding and the selected route ownership are part of the kernel identities.

## Accepted Kernel Experiments

### Grouped Q4_K identity and arithmetic reuse

The grouped implementation uses a dedicated problem and route contract while reusing the validated ordinary Q4_K packed reader, scale/minimum reconstruction, decoded-weight LDS layout, WMMA leaf, wait schedule, and BF16 store conversion. Expert rebasing and interleaved route strides are handled by the grouped lowering.

The initial exact artifacts passed full and partial-route correctness, deterministic rebuilds, mutation checks, malformed-route sentinels, and the independent BF16 reference. This arithmetic boundary remains accepted; the initial performance geometry was superseded by the shape-specific selections below.

### Shape-specific geometry and route ownership

M128/N64 with serial ownership is retained at B1 because short and sparse routes favor the lower accumulator footprint. M128/N128 with SplitRoutes4 and SplitRoutes8 is retained at B4 and B16 because larger routes amortize the wider body. The B1 `Mixed128_64` tail uses an M64 body only where it improves the fitted route objective.

All three selected geometries pass exact packed-HIP comparison, non-aligned tails, route mutations, active and inactive weight mutations, deterministic reruns, and full-row coverage. Geometry and split ownership are encoded in the selected identities.

### DependencyBatch4 decode schedule

The Q4_K `DependencyBatch4` schedule reuses dead register pairs without increasing the final resource class. It is retained on all three selected keys after exactness and fitted-medoid timing checks. Unbatched decode and dependency-width alternatives did not provide a stable advantage.

### Inactive-M consumer suppression

The accepted suppression guards wholly inactive waves and inactive 16-row M minitiles around the WMMA consumer while leaving packed decode, required LDS reads, barriers, and masked global access uniform. It adds no solution field and preserves the resource envelope.

Disjoint confirmation improved retained-parent weighted latency by `2.70%` at B1 and `2.71%` at B4. A longer B16 confirmation reversed the initial short-screen result and improved weighted latency by `1.00%`. Every final key remained bitwise exact, and suppression is retained across B1, B4, and B16.

## Rejected Kernel Experiments

### M64 primary geometry

M64 primary bodies reduced accumulator state but repeated decode and route traversal work. They were exact but lost the selected M128 parents on the fitted objective. M64 remains useful only as the restricted B1 mixed-tail body.

### Unbatched and alternate decode schedules

Unbatched Q4_K decode and dependency-width alternatives were exact but slower or inconsistent against DependencyBatch4. No alternate decode schedule is retained.

### Double-LDS SIA4 alternatives

Double-LDS SIA4 variants were exact after their address and lifetime contracts were repaired, but the added LDS traffic and synchronization did not beat the padded single-LDS parents. They are rejected as final layouts.

### Nonselected split factors

B4 split2, split8, split16, and larger alternatives did not beat SplitRoutes4. B16 split2, split4, split16, and SplitRoutes32 did not beat SplitRoutes8 after balanced timing. The final split factors remain shape-specific.

### Mixed tails outside B1

A mixed M128/M64 tail was useful at B1 but regressed B4 and B16. B4's direct parent bracket was `0.9854x` candidate/parent throughput, while B16 had no useful short-route population in its dominant medoid. Mixed tails are rejected for those keys.

### Initial inactive-M B16 result

The first nine-repeat B16 suppression screen measured a small regression despite exact output, so B16 was temporarily left on its parent. The subsequent full confirmation improved every medoid and produced a robust candidate-minus-parent interval of `[-1.512,-0.462]%`. The short-screen rejection is superseded by the longer confirmation, and B16 suppression is accepted.

### Device row tasks

The typed J128 device-row-task kernel passed full B1 correctness, deterministic rebuild, malformed-route, sentinel, and independent-reference checks. It assembled at 208 VGPRs, 44 SGPRs, 10,240 LDS bytes, 32 WMMAs, and two barriers.

It was rejected by disjoint fitted timing: complete-call weighted latency regressed from `2.4790` to `2.5800 ms`, or `0.9608x` parent/task throughput. Kernel-only weighted throughput was `0.9724x`, and the dominant medoid regressed to `0.9477x`. J64, mixed task sizes, and ownership-conditioned task/decode variants were not opened.

### Arithmetic and broad ownership changes

No approximate accumulator, cross-workgroup reduction, atomics, persistent traversal, prepared-weight shadow, or alternate arithmetic order was retained. These mechanisms either changed the arithmetic/ownership contract or lacked a measured fitted-prior premise after the selected direct kernels were qualified.

## Qualification Summary

The final B1, B4, and B16 kernels pass exact packed-HIP comparison, independent BF16-reference checks, finite-output and full-row coverage, deterministic reruns, gradient and route mutations, active and inactive weight mutations, malformed-route sentinels, and non-aligned route tails. The independent BF16 NRMSE remains below `0.01` for every selected shape.

Two independent generation/build/inspection roots produce byte-identical artifacts. The retained result is the grouped Q4_K direct decoder with SIA5/PGR2/PLR1 padded LDS, DependencyBatch4 decode, shape-specific serial or split ownership, the B1-only mixed tail, and confirmed inactive-M suppression.
