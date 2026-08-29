# GGTensile Grouped MMQ Backward Pair IQ2_XXS Experiment

## Scope And Contract

This record covers the isolated gfx1151 paired IQ2_XXS backward kernel for the DeepSeek routed gate and up projections. For each routed expert the kernel accumulates both gate/up input-gradient matmuls in one workgroup dataflow:

```text
dG_g[M_g,2048] @ W_gate_g[2048,4096]
+ dU_g[M_g,2048] @ W_up_g[2048,4096]
-> dX_g[M_g,4096]
```

The measured aggregate-row shapes are `R=12288`, `49152`, and `196608`. Each packed IQ2_XXS bank is `[256,2048,1056]`; each expert occupies 2,162,688 bytes. The gate and up gradient outputs and the one shared gradient input are contiguous BF16. The arithmetic uses signed codebook values, the odd per-K32 group scale, FP32 scale arithmetic, one shared FP32 accumulator set for both projections, and one final BF16 RNE store. The authoritative 256-entry codebook is extracted at generation time from the llama.cpp grid.

The paired research ABI carries both gradient outputs, both packed weights, the gradient input, the device-resident route description, and the shape values. Logical paired throughput is `4 * R * 2048 * 4096 / (latency_ms * 1e9)`, and the speedup ratio is HIP time divided by GGTensile time.

## Final Benchmark Results

The table shows the fastest qualified kernel found for each measured aggregate-row shape.

| Matrix shape `(R,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(12288,4096,2048)` | `ggbpair_648838e6b6e5e97b` | `12.984` | `1.1984x` |
| `(49152,4096,2048)` | `ggbpair_dd16a88d7a365ea0` | `23.513` | `1.3953x` |
| `(196608,4096,2048)` | `ggbpair_13f676c087b0a7f7` | `24.641` | `1.4240x` |

The B1 row uses the retained M64 SIA5 K pipeline; B4 uses the retained M128 direct-pointer/swizzle8 body; B16 uses the retained M192 body. All timed retained outputs were bitwise exact.

## Final Kernel Profiles

| Kernel hash | Geometry and ownership | Decode and schedule | VGPR / SGPR | LDS bytes | WMMAs / barriers |
| --- | --- | --- | ---: | ---: | ---: |
| `ggbpair_648838e6b6e5e97b` | M64/N64 serial routes | dual-LDS full/tail split, SIA5 K pipeline, swizzle8, codebook overlap | `101 / 41` | `10,240` | `32 / 7` |
| `ggbpair_dd16a88d7a365ea0` | M128/N64 serial routes | dual-LDS full/tail split, direct pointers, activation prefetch, swizzle8 | `149 / 41` | `10,240` | `64 / 5` |
| `ggbpair_13f676c087b0a7f7` | M192/N64 serial routes | dual-LDS full/tail split, direct pointers, second-read/A overlap, swizzle8 | `188 / 41` | `10,240` | `96 / 5` |

All retained artifacts are gfx1151 code-object-v5 wave32 kernels with zero private storage, zero spills, and no scratch, calls, or dynamic stack.

## Accepted Kernel Experiments

### Dedicated decoder and fused anchors

The dedicated reader maps each of 128 threads to one aligned 16-value group in the 64-column by K32 decoded tile, loads one 64-bit index/sign word and the shared FP16 block scale, reconstructs the two authoritative grid entries, expands each seven-bit sign group with odd parity, forms `d * (2*scale+1) / 8` in FP32, and stores BF16-RNE values through the XOR-4 decoded-weight layout. A lane-mapping review caught and fixed the half-selection: the lower or upper 16-value half of a K32 group is selected by lane bit zero. The single-LDS anchors inspect at `87 VGPR / 41 SGPR / 6,144 B` for M64 and `127 / 41 / 6,144 B` for M128, each with five barriers. Both are bit-exact to the installed controls on the R35 route, mutation, sentinel, and independent-oracle matrices, and the M128 production body is exact over 50,331,648 B1, 201,326,592 B4, and 805,306,368 B16 BF16 elements.

### M64 swizzle8 activation-prefetch and SIA5 K pipeline at B1

The M64 staged sequence split the two projection bodies at a full-tile boundary, issued their packed reads concurrently, and prefetched the next activation tile with PGR2/SIA4. Changing only the decoded-weight XOR chunk from swizzle4 to swizzle8 replaced eight 64-bit LDS reads per N16 fragment pair with four 128-bit reads, reducing the physical plan from 105 to 101 VGPRs, static LDS instructions from `201` to `137`, and VALU issues from `1,020` to `944`. Swizzle16 regressed sharply; swizzle8 was confirmed over swizzle4 (`37.2481` versus `38.8967 ms`) and against installed HIP (`1.0089x` anchor speedup, `1.0443x` over the swizzle4 anchor).

The final M64 identity is the SIA5 K pipeline with current and next packed-weight prefetch, paired codebook overlap, direct second-projection pointers, and a geometry-derived second-A wait frontier. It inspects at `101 VGPR / 41 SGPR / 10,240 B` LDS with 32 WMMAs and seven barriers and contains `1,438` VALU issues, `209` LDS operations, and `105` VMEM operations. Its disjoint B1 confirmation improves complete-call latency over the installed control by `16.612%` independently and `16.562%` paired, with both 95% intervals excluding zero.

### M128 direct-pointers and swizzle8 baseline

The current M128 activation-prefetch body was rebuilt from the shared IQ2_XXS decode at `153 VGPR / 41 SGPR / 10,240 B` LDS. Binding the second gradient and packed-bank pointers directly, eliminating projection pointer swaps while preserving PGR2/SIA4, swizzle4, concurrent reads, and codebook overlap, improved the parent by `1.0039x` at B4 and `1.0083x` at B16. Changing only the decoded-weight XOR chunk to swizzle8 then improved inspection from 153 to 149 VGPRs and from `201` to `137` LDS operations, and the common screen measured `1.0279x` at B4 and `1.0166x` at B16 over the swizzle4 direct-pointer parent. A shared lowering defect in the M128 wait frontier was also repaired: the second-projection schedule now derives `2 * m_tiles` (`vmcnt(4)` for M128) instead of reusing the M64 literal.

### M192 ownership at B16

The M192/N64 identity uses matrix instruction `(16,16,16,1,1,3,4,4,1)`, direct second-projection pointers, activation prefetch, swizzle8, and overlap of the second packed read with the first activation prefetch. M192-sensitive power-of-two shifts in coordinate generation were replaced with exact scale helpers, and the ordinary and paired backward paths for power-of-two geometries remain source-byte-identical. Independent builds for rows 35, 257, 577, 49,152, and 196,608 are byte-identical and inspect at `188 VGPR / 41 SGPR / 10,240 B` LDS with 96 WMMAs and five barriers. The B16 nine-repeat screen measured `261.8385 ms` versus `288.2859 ms` for retained M128 (`1.1010x`), and the disjoint 25-repeat confirmation measured `267.7232 ms` versus `291.6244 ms` for M128 and `381.2457 ms` for HIP: `1.0893x` faster than M128 and `1.4240x` faster than HIP, with robust intervals excluding parity. Full R196,608 boundary-route correctness is exact over all `805,306,368` BF16 elements. Only the exact B16 key is retained.

## Rejected Kernel Experiments

### M64 geometry and the initial fitted baseline

The initial fitted geometry screen rejected plain M64 at all keys and M128 at B1/B16, leaving only M128 at B4 as a finalist. M64 was only revived as the B1 winner after swizzle8 and the SIA5 K pipeline removed its decode overhead.

### M128 pipeline and swizzle variants

Porting the M64 SIA5 K pipeline to M128 at `149 VGPR / 41 SGPR / 10,240 B` LDS was exact on every timed medoid but regressed to `0.9178x` of the parent at B4 and `0.7512x` at B16, so the identity was removed. Keeping SIA4 and enabling only next-weight prefetch (after fixing the geometry-derived wait frontier) likewise regressed by `3.15%` at B4 and `13.12%` at B16 and was removed. Swizzle16 was rejected by the assembler for repeated `src0` bank conflicts in paired `v_dual_mul_f32`; swizzle0 assembled and was exact but measured `18.11%` slower at B4 and `11.06%` slower at B16 than swizzle8, with both robust intervals excluding parity.

### Lower-state M128 projection reads

Two lower-state M128 identities were admitted at the source-only prediction of `139 VGPR / 41 SGPR / 10,240 B` LDS, a real occupancy step from 9 to 10 waves per SIMD. Both assembled exactly at that envelope and passed the R35 and R257 matrices. The fully serial read schedule regressed the retained B16 parent by `1.46%` and was rejected. The second-read/A-overlap schedule improved B16 by `0.13%` and advanced, but the B4 screen regressed by `3.69%` with 9 of 10 medoids slower, closing the whole X1 premise and removing its admission.

### M256 ownership

M256/N64 assembled after a geometry-specific VOPD bank repair at `237 VGPR / 41 SGPR / 10,240 B` LDS with 128 WMMAs and five barriers, and passed the R35 and R769 route matrices. Its B16 screen measured `283.4892 ms` versus `260.9493 ms` for retained M192 (`0.9206x`) and only `1.0245x` versus M128, so it was rejected and its constructor, admission, and scratch repair were removed.

### Other closed mechanisms

Store interleaving, width-16 decode, deferred codebook issue, midpoint codebook waits, SIA4 K pipelining, and VMEM-frontier overrides were measured and rejected; the frontier overrides regressed direct SIA5 by about `0.054%` and pipeline SIA5 by about `0.026%` weighted, so the geometry-derived frontier remains the control. Codebook placement, further LDS swizzles, dual buffering, traversal, wider N, and DepthU changes remain closed unless an exact profile establishes a new bottleneck and an occupancy-safe mechanism.

## Remaining Work

The current direct-kernel benchmark separates the DeepSeek route laws at B16. The M128 B16 baseline measured `1.2681x` GGTensile/HIP for the learned law and `1.2524x` in the 50-repeat top-up, versus its earlier documented `1.3203x`; the hash law is closer at `1.3090x`. The accepted M192 B16 body is faster at `1.4240x`. B1 and B4 remain healthy at `1.2136x` and `1.4122x`. B16 should be retuned or requalified on the current confirmation bank while retaining the exact M192 scope, starting with route-sensitive schedule, wait, and ownership choices before opening another geometry family. The lower-state M128 and M256 rejections are not reopened by this deficit.

## Qualification Summary

The retained kernels pass exact packed-HIP comparison, independent FP32-accumulating references, finite-output and full-row coverage checks, deterministic reruns, first and non-first routes, sparse and repeated experts, both-gradient and both-active-bank mutations, inactive-expert inertness, malformed-route sentinels, and mixed full/tail ownership. Two independent build roots for rows 35, 257, and 12,288 reproduce byte-identical source, object, and HSACO. The selected result is the M64 SIA5 K pipeline at B1, the M128 direct-pointer/swizzle8 body at B4, and the M192 body at B16, all with zero private storage and zero spills.
