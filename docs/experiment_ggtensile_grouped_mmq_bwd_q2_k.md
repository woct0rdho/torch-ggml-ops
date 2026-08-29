# GGTensile Grouped MMQ Backward Q2_K Experiment

## Scope And Contract

This record covers the isolated gfx1151 grouped Q2_K backward kernel for the DeepSeek routed down projection. For each expert, the operation is:

```text
dY_g[M_g,4096] @ W_g[4096,2048] -> dX_g[M_g,2048]
```

The measured aggregate-row shapes are `R=12288`, `49152`, and `196608`. The Q2_K expert bank is `[256,4096,672]`: each 256-value block occupies 84 bytes, each packed row contains eight blocks, and each expert occupies 2,752,512 bytes. Inputs and outputs are contiguous BF16, accumulation is FP32 WMMA V1, and stores use BF16 round-to-nearest-even.

The grouped research ABI is:

```text
grad_output, packed_weight, grad_input,
expert_indices, expert_offsets, num_experts, rows, bytes_per_expert
```

Q2_K uses a dedicated width-16 decoder. Each lane owns one aligned 16-value group, shares its two-bit payload shift and scale/minimum pair across those values, writes decoded weights to LDS as BF16, and uses the established FP32-WMMA/BF16 store path. Q2-specific packed reads, metadata lifetimes, LDS layout, schedule, and route splitting are separate from Q4_K and Q5_K arithmetic.

Timing uses five fitted medoids from each DeepSeek learned/hash component, with reporting weights `40/43` and `3/43`. Complete-call timing includes output allocation. Logical throughput is `2 * R * 2048 * 4096 / (latency_ms * 1e9)`, and the speedup ratio is HIP time divided by GGTensile time.

## Final Benchmark Results

The table shows the fastest qualified HIP-normalized kernel found for each aggregate-row shape. It contains only the requested matrix shape, kernel hash, GGTensile speed, and HIP speedup.

| Matrix shape `(R,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(12288,2048,4096)` | `ggsol_f90ec6b01eaaa1c0` | `15.749` | `1.6930x` |
| `(49152,2048,4096)` | `ggsol_49e0749ea2ff3aa7` | `23.604` | `1.4648x` |
| `(196608,2048,4096)` | `ggsol_8356a591b014fc2c` | `27.289` | `1.5847x` |

These identities remained bitwise exact on the final full-row checks. The later inactive-M confirmations improved the selected GGTensile parents, but were parent-to-candidate measurements rather than a replacement three-shape HIP bracket.

## Final Kernel Profiles

| Kernel hash | Geometry and ownership | Decode schedule | VGPR / SGPR | LDS bytes |
| --- | --- | --- | ---: | ---: |
| `ggsol_f90ec6b01eaaa1c0` | M64/N64, masked M tail | DependencyBatch4 | `87 / 35` | `5,120` |
| `ggsol_49e0749ea2ff3aa7` | M128/N64, `Mixed128_64` tail | DependencyBatch4 | `135 / 35` | `5,120` |
| `ggsol_8356a591b014fc2c` | M128/N128, SplitRoutes32 | Serial | `206 / 35` | `10,240` |

All final artifacts are gfx1151 code-object-v5 wave32 kernels with zero private bytes, spills, scratch, calls, and dynamic stack. The final LDS layouts use eight-byte row padding. The selected B1, B4, and B16 identities have 8, 16, and 32 static WMMAs respectively, with two barriers and no undeclared register use.

## Accepted Kernel Experiments

### Dedicated Q2_K decoder

Q2_K received a strict grouped backward identity and a dedicated packed reader rather than being treated as Q4_K or Q5_K. The decoder maps each lane to one aligned 16-value group, performs the shared two-bit extraction, reconstructs scale/minimum values, and writes BF16 decoded weights to the existing WMMA leaf.

The M128/N128 SIA2 pilot passed the 625-row boundary matrix bit-for-bit against packed HIP, including deterministic reruns, gradient and route mutations, active and inactive weight mutations, malformed routes, and sentinels. The independent-reference NRMSE was `3.78e-5`. The initial resource profile was 190 VGPRs, 35 SGPRs, 8,192 LDS bytes, 32 WMMAs, and two barriers.

### Shape-specific geometry and tail ownership

The final geometry follows the measured route-size tradeoff: M64/N64 for B1, M128/N64 for B4, and M128/N128 for B16. Eight-byte LDS padding is retained because unpadded rows materially regress both short and medium shapes. The B4 `Mixed128_64` tail selects M64 only through 64 rows; B16 uses SplitRoutes32 to avoid the dominant two-tile route loop.

All selected geometries and tail paths pass exact packed-HIP comparison, route-tail checks, mutations, and deterministic reruns. Geometry, padding, SIA choice, and route split are encoded in the selected identities rather than inferred at timing time.

### DependencyBatch4 decode scheduling

The Q2-specific `DependencyBatch4` schedule reuses four dead `valu_b` register pairs without changing the resource envelope. It improves B1 by `2.14%` and B4 by `6.38%` against their serial parents while preserving exactness. The same schedule regresses B16 by `1.46%` and loses every fitted medoid, so B1 and B4 retain DependencyBatch4 and B16 retains serial decode.

### Inactive-M consumer suppression

The accepted suppression guards wholly inactive waves and inactive 16-row M minitiles around the WMMA consumer while keeping packed decode, LDS traffic, barriers, and masked global access uniform. It uses dead post-prologue scalar state and adds no solution field or resource cost.

Disjoint confirmation improved retained-parent weighted latency by `13.91%` at B1, `1.60%` at B4, and `4.09%` at B16. Every relevant confirmation medoid remained bitwise exact; nine of ten B4 medoids improved, and every B16 medoid improved. Full-row correctness also passed at all three aggregate sizes, so suppression remains part of the accepted kernel design.

## Rejected Kernel Experiments

### M128/N128 SIA2 and losing geometries

The initial M128/N128 SIA2 pilot was exact but was not competitive at B1. M64/N64 was rejected for larger keys because repeated Q2 scale/minimum reconstruction outweighed its lower accumulator cost. M128/N128 was rejected for B1 because its register envelope did not pay back on sparse small routes. These alternatives remain valid correctness controls but are not selected identities.

### Double LDS and alternate layouts

The first M128/N64 double-LDS artifact failed exactness, determinism, mutations, and independent-reference checks because Q2 metadata aliased a register later used for decoded-B LDS addressing. Reserving the address register repaired the lifetime and produced exact artifacts, but double LDS was later slower than the padded single-LDS path and was not retained.

Plain LDS, SIA2, SIA5/PGR1, and swizzle variants were exact but slower or statistically flat. The final path retains padded single LDS; no alternate layout is selected.

### M-tail threshold policy

A threshold-only tail artifact failed correctness because the inherited M64 branch emitted one tile by construction. A typed Q2-specific route loop repaired correctness, but the candidate regressed to `39.9595 ms` versus `37.2586 ms` for the existing `<=64` policy. Rows 65-127 favor one masked M128 tile over two M64 tiles, so the broader threshold was rejected.

### Split64 and smaller split factors

Split32 improved the B16 parent and passed all exactness checks. Split64 was exact but lost a balanced bracket by `0.30%`, including the dominant learned profile and every hash medoid. Split64 was removed; SplitRoutes32 is the maximum retained split.

### DependencyBatch4 at B16

The B16 DependencyBatch4 candidate was exact but regressed `1.46%` against serial decode and lost every medoid. A temporary dependency-width-two endpoint also lost B16 by `0.51%`. Neither schedule is retained for B16.

### Arithmetic replacement

An FMA/output-modifier replacement for the proven scale/minimum arithmetic assembled and reduced one issue per row, but failed candidate, mutation, and independent-reference comparisons. The proven add/multiply sequence was restored.

### Unrepresented or broad searches

Row-task ownership, U4, broad split sweeps, and N128 reopening after the selected parent were not accepted as generic follow-ups. Existing controls either failed the resource/timing premise or did not remove repeated Q2 scale/minimum reconstruction. No new identity was created for them.

## Remaining Work

### Focused B4 requalification

A later direct-kernel triage run measured Q2_K B4 at `1.3666x` HIP, below the documented `1.4648x` table value. The separate learned and hash profiles were `1.3604x` and `1.4585x`, indicating route-law sensitivity rather than a correctness issue. B4 is the only selected shape requiring focused retuning or requalification.

Start with the existing B4 M128/N64 `SecondaryTile`/serial-body alternatives and their DependencyBatch4 decode. Use the current complete-call protocol with output allocation, the corrected HIP control, the fitted learned/hash objective, and a longer confirmation bank before changing geometry. Do not broaden the split sweep until this comparison is resolved.

### Tail-boundary check

Recheck the existing B4 `Mixed128_64` and threshold candidates around the 64/128-row boundary under the current timing protocol. Reopen only these exact threshold identities; a new row-task or broad ownership search requires a separate mechanism that removes Q2 scale/minimum reconstruction work.

B1 and B16 do not currently justify retuning: the triage results were `1.6630x` versus `1.6930x` for B1 and `1.5984x` versus `1.5847x` for B16. The B16 DependencyBatch4 rejection, SplitRoutes32 selection, and broad geometry closures remain closed.

## Qualification Summary

The final kernels pass exact packed-HIP comparison at 12,288, 49,152, and 196,608 rows; independent BF16-reference checks; finite-output and full-row coverage checks; deterministic reruns; gradient, route, active-weight, and inactive-weight mutations; malformed-route sentinels; and non-aligned route tails.

Independent BF16-reference NRMSE is `8.55e-6`, `5.60e-5`, and `9.81e-5` for B1, B4, and B16. Two independent generation/build/inspection roots produce byte-identical artifacts. The retained result is the Q2_K width-16 decoder with padded LDS, shape-specific M/N ownership, DependencyBatch4 at B1/B4, serial decode at B16, SplitRoutes32 for B16, and inactive-M suppression.
