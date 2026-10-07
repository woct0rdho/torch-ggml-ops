# GGTensile Grouped MMQ Backward Pair IQ2_XXS Experiment

## Scope

This record covers the routed gfx1151 paired IQ2_XXS input gradients for the DeepSeek gate and up projections, accumulated in one workgroup dataflow.

## Final Results

`TFLOPS = 4*R*N*K / (median_ms * 1e9)` (both projections), and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Each row is the median over the benchmark's sampled expert partitions, and both implementations are timed on the same partitions.

| Batch | Logical shape `(R,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `(12288,2048,4096)` | 12.385 | 1.0706x | `grouped_mmq_bwd_pair_iq2_xxs_r12288_n2048_k4096_016b151a7552373f` | `grouped_bwd_pair_task_iq2_xxs_n2048_k4096_mt128_nt64_s2_skip_g4_abar` |
| 4 | `(49152,2048,4096)` | 21.477 | 0.9739x | `grouped_mmq_bwd_pair_iq2_xxs_r49152_n2048_k4096_0416c0e8a240d489` | `grouped_bwd_pair_task_iq2_xxs_n2048_k4096_mt128_nt64_s2_skip_g4_abar` |
| 16 | `(196608,2048,4096)` | 23.180 | 0.8628x | `grouped_mmq_bwd_pair_iq2_xxs_r196608_n2048_k4096_b7bf8fc3b8cf66b8` | `grouped_bwd_pair_task_iq2_xxs_n2048_k4096_mt128_nt64_s2_skip_g4_abar` |

HIP is ahead on 2 of the 3 rows, with speedups from `0.8628x` to `1.0706x` (mean `0.9691x`).

These medians replace the earlier recorded values, which were paired against HIP bodies that have since been retuned.

## Accepted Experiments

### Dedicated decoder and fused anchors

The dedicated reader maps each of 128 threads to one aligned 16-value group in the 64-column by K32 decoded tile, loads one 64-bit index/sign word and the shared FP16 block scale, reconstructs the two authoritative grid entries, expands each seven-bit sign group with odd parity, forms `d * (2*scale+1) / 8` in FP32, and stores BF16-RNE values through the XOR-4 decoded-weight layout. A lane-mapping review caught and fixed the half-selection: the lower or upper 16-value half of a K32 group is selected by lane bit zero. The single-LDS anchors inspect at `87 VGPR / 41 SGPR / 6,144 B` for M64 and `127 / 41 / 6,144 B` for M128, each with five barriers.

### M64 swizzle8 activation-prefetch and SIA5 K pipeline at B1

The M64 staged sequence split the two projection bodies at a full-tile boundary, issued their packed reads concurrently, and prefetched the next activation tile with PGR2/SIA4. Changing only the decoded-weight XOR chunk from swizzle4 to swizzle8 replaced eight 64-bit LDS reads per N16 fragment pair with four 128-bit reads, reducing the physical plan from 105 to 101 VGPRs, static LDS instructions from `201` to `137`, and VALU issues from `1,020` to `944`. Swizzle16 regressed sharply. swizzle8 was confirmed over swizzle4 (`37.2481` versus `38.8967 ms`) and against installed HIP (`1.0089x` anchor speedup, `1.0443x` over the swizzle4 anchor).

The final M64 identity is the SIA5 K pipeline with current and next packed-weight prefetch, paired codebook overlap, direct second-projection pointers, and a geometry-derived second-A wait frontier. It inspects at `101 VGPR / 41 SGPR / 10,240 B` LDS with 32 WMMAs and seven barriers and contains `1,438` VALU issues, `209` LDS operations, and `105` VMEM operations. Its disjoint B1 confirmation improves complete-call latency over the installed control by `16.612%` independently and `16.562%` paired, with both 95% intervals excluding zero.

### M128 direct-pointers and swizzle8 baseline

The current M128 activation-prefetch body was rebuilt from the shared IQ2_XXS decode at `153 VGPR / 41 SGPR / 10,240 B` LDS. Binding the second gradient and packed-bank pointers directly, eliminating projection pointer swaps while preserving PGR2/SIA4, swizzle4, concurrent reads, and codebook overlap, improved the parent by `1.0039x` at B4 and `1.0083x` at B16.

Changing only the decoded-weight XOR chunk to swizzle8 then improved inspection from 153 to 149 VGPRs and from `201` to `137` LDS operations, and the common screen measured `1.0279x` at B4 and `1.0166x` at B16 over the swizzle4 direct-pointer parent. A shared lowering defect in the M128 wait frontier was also repaired: the second-projection schedule now derives `2 * m_tiles` (`vmcnt(4)` for M128) instead of reusing the M64 literal.

### M192 ownership at B16

The M192/N64 identity uses matrix instruction `(16,16,16,1,1,3,4,4,1)`, direct second-projection pointers, activation prefetch, swizzle8, and overlap of the second packed read with the first activation prefetch. M192-sensitive power-of-two shifts in coordinate generation were replaced with exact scale helpers, and the ordinary and paired backward paths for power-of-two geometries remain source-byte-identical. Independent builds for rows 35, 257, 577, 49,152, and 196,608 are byte-identical and inspect at `188 VGPR / 41 SGPR / 10,240 B` LDS with 96 WMMAs and five barriers. The B16 nine-repeat screen measured `261.8385 ms` versus `288.2859 ms` for retained M128 (`1.1010x`), and the disjoint 25-repeat confirmation measured `267.7232 ms` versus `291.6244 ms` for M128 and `381.2457 ms` for HIP: `1.0893x` faster than M128 and `1.4240x` faster than HIP, with robust intervals excluding parity. Only the exact B16 key is retained.

## Rejected Experiments

### M64 geometry and the initial fitted baseline

The initial fitted geometry screen rejected plain M64 at all keys and M128 at B1/B16, leaving only M128 at B4 as a finalist. M64 was only revived as the B1 winner after swizzle8 and the SIA5 K pipeline removed its decode overhead.

### M128 pipeline and swizzle variants

Porting the M64 SIA5 K pipeline to M128 at `149 VGPR / 41 SGPR / 10,240 B` LDS regressed to `0.9178x` of the parent at B4 and `0.7512x` at B16, so the identity was removed. Keeping SIA4 and enabling only next-weight prefetch (after fixing the geometry-derived wait frontier) likewise regressed by `3.15%` at B4 and `13.12%` at B16 and was removed. Swizzle16 was rejected by the assembler for repeated `src0` bank conflicts in paired `v_dual_mul_f32`. swizzle0 assembled and measured `18.11%` slower at B4 and `11.06%` slower at B16 than swizzle8, with both robust intervals excluding parity.

### Lower-state M128 projection reads

Two lower-state M128 identities were admitted at the source-only prediction of `139 VGPR / 41 SGPR / 10,240 B` LDS, a real occupancy step from 9 to 10 waves per SIMD. Both assembled exactly at that envelope and passed the R35 and R257 matrices. The fully serial read schedule regressed the retained B16 parent by `1.46%` and was rejected. The second-read/A-overlap schedule improved B16 by `0.13%` and advanced, but the B4 screen regressed by `3.69%` with 9 of 10 medoids slower, closing the whole X1 premise and removing its admission.

### M256 ownership

M256/N64 assembled after a geometry-specific VOPD bank repair at `237 VGPR / 41 SGPR / 10,240 B` LDS with 128 WMMAs and five barriers, and passed the R35 and R769 route matrices. Its B16 screen measured `283.4892 ms` versus `260.9493 ms` for retained M192 (`0.9206x`) and only `1.0245x` versus M128, so it was rejected and its constructor, admission, and scratch repair were removed.

### Other closed mechanisms

Store interleaving, width-16 decode, deferred codebook issue, midpoint codebook waits, SIA4 K pipelining, and VMEM-frontier overrides were measured and rejected. The frontier overrides regressed direct SIA5 by about `0.054%` and pipeline SIA5 by about `0.026%` weighted, so the geometry-derived frontier remains the control. Codebook placement, further LDS swizzles, dual buffering, traversal, wider N, and DepthU changes remain closed unless an exact profile establishes a new bottleneck and an occupancy-safe mechanism.

## Open Items

B16 should be retuned or requalified on the current confirmation bank while keeping the exact M192 scope, starting with route-sensitive schedule, wait, and ownership choices before opening another geometry family. The lower-state M128 and M256 rejections are not reopened by that deficit.

## Closure

The retained result is the M64 SIA5 K pipeline at B1, the M128 direct-pointer/swizzle8 body at B4, and the M192 body at B16.
