# GGTensile Fixed-Group MMQ Backward Q8_0 Experiment

## Scope And Contract

This record covers the fixed-group Q8_0 backward kernel for gfx1151. There are eight independent non-routed groups. For each group, the operation is:

```text
dY [M, K] @ W [K, N] -> dX [M, N]
```

The measured production shapes use `M` equal to 2,048, 8,192, or 32,768, `N=4096`, and `K=1024`. The tensor layouts are:

```text
grad_output   [tokens, 8, 1024] BF16
grad_input    [tokens, 8, 4096] BF16
packed_weight [8, 1024, 4352] uint8
```

Rows are token-major with the group axis interleaved. Each Q8_0 weight row contains 128 blocks of 34 bytes, for a packed row stride of 4,352 bytes and a per-group size of 4,456,448 bytes.

The fixed six-argument ABI is:

```text
grad_output, packed_weight, grad_input,
tokens (u32), out_features (u32), bytes_per_group (u64)
```

The selected kernels use gfx1151 code-object version 5, wave32, 128 threads, and two-dimensional M/N tiling with fixed group-Z ownership. Correctness requires exact BF16 output equality with the installed HIP fixed-group kernel and the independent dequantized reference.

Timing measures the fixed grouped multiply body with the same packed weights and input tensors. Setup, allocation, dequantization, and reference work are outside kernel timing. Logical throughput counts all eight groups as `2*tokens*8*4096*1024/(median_ms*1e9)`.

## Final Benchmark Results

The table contains the fastest qualified kernels found across the complete experiment. The speedup is HIP time divided by GGTensile time; values above `1.0x` favor GGTensile. These are the latest direct-kernel measurements using 20 warmups, 25 samples, and launch batching.

| Matrix shape `(M,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(2048,4096,1024)` | `ggsol_e3f23cb5c4ad7aae` | `29.459` | `1.3148x` |
| `(8192,4096,1024)` | `ggsol_8fb858b844ca8109` | `30.059` | `1.3462x` |
| `(32768,4096,1024)` | `ggsol_d0bc5b94d26fe2b3` | `30.072` | `1.2616x` |

The three hashes are the same selected E6-derived identities at the three token counts. All remained bitwise exact and faster than HIP in the latest direct-kernel run.

## Final Kernel Profile

| Kernel hash | Geometry and pipeline | Workgroup | VGPR / SGPR | LDS bytes | WMMAs | Barriers |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `ggsol_e3f23cb5c4ad7aae` | N-major M128/N128, DepthU64, PGR2/PLR1/SIA4 | `32x4x1` | `216 / 17` | `18,432` | 64 | 2 |
| `ggsol_8fb858b844ca8109` | N-major M128/N128, DepthU64, PGR2/PLR1/SIA4 | `32x4x1` | `216 / 17` | `18,432` | 64 | 2 |
| `ggsol_d0bc5b94d26fe2b3` | N-major M128/N128, DepthU64, PGR2/PLR1/SIA4 | `32x4x1` | `216 / 17` | `18,432` | 64 | 2 |

The final identity uses the pad-8 single-buffer LDS layout, ordinary packed Q8 extraction, raised-priority element-serial stores, and fixed group-Z address ownership. The artifacts have a 40-byte kernarg segment, no private segment, no spills, no scratch, and no undeclared register use.

## Accepted Kernel Experiments

### Fixed ownership with ordinary Q8 arithmetic

The fixed lowering keeps the mature ordinary Q8_0 decode, WMMA, correction, and BF16 store arithmetic while supplying fixed group-Z bases, interleaved token-major row strides, and the six-argument address contract. This separated fixed ownership from routed grouped addressing without creating a second arithmetic implementation.

The independent M256/N64/K32 artifacts were exact at all three shapes and passed finite-output, full-row coverage, repeated-output, poisoned-output, reference, and eight-gradient/eight-weight mutation checks. Their profile was 231 VGPRs, 17 SGPRs, 5,120 LDS bytes, 32 WMMAs, and two barriers. This arithmetic anchor was accepted as the correctness base, but its M-major traversal was rejected as the performance selection because the B16 candidate/HIP time ratio reached `6.275x`.

### N-major workgroup traversal

N-major traversal places N tiles on the fastest grid coordinate, completing all N tiles for one activation tile before moving to the next M tile. It preserves the arithmetic and group-Z ownership while increasing reuse of the same gradient-output rows.

The N-major artifacts retained the anchor resource profile and passed the full fixed-group correctness and mutation checks. Relative to HIP, they improved latency by `16.6%`, `16.1%`, and `14.5%` at M2K, M8K, and M32K. N-major ownership is part of the final identity.

### M128/N128 square geometry

The M128/N128 geometry halves the number of N workgroups per M tile while retaining 128 accumulator elements per wave. It was exact at all required shapes and passed the complete fixed-group mutation and reference checks. The artifact profile was 202 VGPRs, 17 SGPRs, 10,240 LDS bytes, 32 WMMAs, and two barriers.

M128/N128 reduced latency relative to N-major M256/N64 by `3.1%`, `3.9%`, and `4.3%` at M2K, M8K, and M32K. It was retained as the parent for reduction-loop experiments and then superseded by DepthU64.

### DepthU64 reduction pipeline

DepthU64 with PGR2, PLR1, and SIA4 halves reduction-loop iterations, synchronization, and scalar loop work. It raises the decoder-row count and LDS allocation, but the M128/N128 ownership provides enough reuse for the larger staging footprint to pay back.

The selected artifacts use 216 VGPRs, 17 SGPRs, 18,432 LDS bytes, 64 WMMAs, and two barriers. They were exact at all three shapes and beat M128/N128 by `7.1%`, `2.8%`, and `2.9%` in the initial balanced screen. The later direct-kernel run retained the same identities and measured the final speed table above.

## Rejected Kernel Experiments

### M64/N256 representability

M64/N256 would use sixteen N repeats per wave and was expected to reduce the number of N workgroups again. The shared backward writer and validator support only 2, 4, and 8 N repeats, so this geometry was rejected before artifact generation. No timing result is inferred from the representability failure.

### Packed-VOPD extraction on M128/N128 DepthU32

Packed-VOPD Q8 extraction reduced static VALU issues from 608 to 594 but increased VGPRs from 202 to 204. The artifact was bitwise exact, yet its B16 latency was `76.204 ms` versus `75.699 ms` for the M128/N128 parent, a `0.67%` regression. The candidate was rejected.

### M64/N128 occupancy tradeoff

Reducing M repeats from two to one lowered the declaration to 122 VGPRs while retaining N128. The artifact was exact, but its B16 latency was `107.295 ms`, `41.7%` slower than the M128/N128 parent and `15.5%` slower than HIP. The duplicated decode and workgroup cost dominated the occupancy benefit.

### Paired-row clause store

The paired-row store epilogue added eight static clauses without changing the other resources. It was exact, but B16 latency increased by `1.65%` against the M128/N128 parent. The store path was not the limiting stage for this geometry, so the clause variant was rejected.

### Packed-VOPD extraction on DepthU64

The DepthU64 variant was exact but increased the resource declaration from 216 to 218 VGPRs. Its B16 latency was `74.014 ms` versus `73.510 ms` for the DepthU64 parent, a `0.69%` regression. The extra packing did not offset its physical cost.

### Next packed-weight prefetch

Current-and-next packed-weight prefetch was rejected by the selected DepthU64 decoder contract: `depthu64.prefetchpacked.quant` is not implemented. No artifact was emitted. The ordinary decoder's unsupported combination is not a timing result.

### M64/N256 DepthU64 repair

A dedicated M64/N256 DepthU64 attempt first exposed a register-planner collision: decoder row pointers occupied `v200:v207`, overlapping the LDS and quantization state registers. The planner repair moved state to `v208` and `v209`; the repaired artifact inspected cleanly at 230 VGPRs, 17 SGPRs, 36,864 LDS bytes, 64 WMMAs, and two barriers.

The repaired B1 kernel wrote every output element with finite values, but its normalized RMSE against the HIP control was approximately `1.4149`. The fragment ownership or N-tile addressing for sixteen N repeats was therefore incorrect. The repaired identity was rejected before timing.

### BF16 store and clause variants

Exact BF16 conversion-chain interleaving produced mixed body and complete-call movement with no stable advantage. Store-clause widths 16 and 8 produced occasional sub-percent shape-local signals but did not reproduce a stable gain in independent confirmation. Removing the final store clause was a material regression. No alternate epilogue identity is retained.

### Processor-mode metadata

Adding the HIP processor-mode metadata produced byte-identical objects and linked code objects under the configured gfx1151 assembler and linker. It created no distinct executable kernel and was rejected as an optimization identity.

## Qualification Summary

The final M128/N128/DepthU64 artifacts were independently regenerated and inspected. They passed exact HIP comparison at M2K, M8K, and M32K, finite-output and full-row coverage checks, independent dequantized-reference checks, poisoned-output replacement, repeated-output determinism, and mutations of every gradient-output and packed-weight group.

The final artifacts are deterministic gfx1151 code-object-v5 wave32 kernels with the profile above. The selected identity combines N-major ownership, M128/N128 tiling, DepthU64 PGR2/PLR1/SIA4 staging, pad-8 single-buffer LDS, packed Q8 extraction, and fixed group-Z addressing. No rejected or unqualified kernel identity is included in the final benchmark table.
