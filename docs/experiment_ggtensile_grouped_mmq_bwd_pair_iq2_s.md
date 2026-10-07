# GGTensile Grouped MMQ Backward Pair IQ2_S Experiment

## Scope

This record covers the routed gfx1151 paired IQ2_S input gradients for the Qwen gate and up projections, accumulated in one workgroup dataflow.

## Final Results

`TFLOPS = 4*R*N*K / (median_ms * 1e9)` (both projections), and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Each row is the median over the benchmark's sampled expert partitions, and both implementations are timed on the same partitions.

| Batch | Logical shape `(R,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `(16384,512,2048)` | 15.350 | 1.0380x | `grouped_mmq_bwd_pair_iq2_s_r16384_n512_k2048_922b13af53272cd3` | `grouped_bwd_pair_task_iq2_s_n512_k2048_mt128_nt64_s2_skip_g4_abar` |
| 4 | `(65536,512,2048)` | 21.577 | 0.9120x | `grouped_mmq_bwd_pair_iq2_s_r65536_n512_k2048_8de6682658390612` | `grouped_bwd_pair_task_iq2_s_n512_k2048_mt128_nt64_s2_skip_g4_abar` |
| 16 | `(262144,512,2048)` | 22.993 | 0.8309x | `grouped_mmq_bwd_pair_iq2_s_r262144_n512_k2048_470be6ff9e23a91c` | `grouped_bwd_pair_task_iq2_s_n512_k2048_mt256_nt64_s2_skip_g4` |

HIP is ahead on 2 of the 3 rows, with speedups from `0.8309x` to `1.0380x` (mean `0.9270x`).

These medians replace the earlier recorded values, which were paired against HIP bodies that have since been retuned.

## Accepted Experiments

### Fused anchors and dual-LDS combined decode

The first fused anchor interleaves the pair at each K32 step: decode bank zero into LDS, accumulate gradient zero, retire the LDS consumers, overwrite the image from bank one, and accumulate gradient one into the same registers. It established the fused semantics and resource envelope at `81 VGPR / 41 SGPR / 12,288 B` for M64 and `121 / 41 / 12,288 B` for M128, with one emitted store phase after both reductions. The M64 geometry was rejected by fitted timing because repeated route-tile decode cannot repay the lower accumulator pressure.

`DualLdsInterleavedDepthU` then assigned decoded bank zero to LDS bytes `0..4095`, bank one to `4096..8191`, and the codebook to `8192..16383`, so both banks decode before one consumer barrier, followed by both projection WMMA phases and one overwrite barrier. This reduced the dynamic K32 loop from four barriers to two and improved the sequential parent by `2.40%`, `3.20%`, and `4.88%` at B1/B4/B16, becoming the optimization parent.

### Full-tile split, concurrent reads, and prefetch

`DualLdsFullTileSplitInterleavedDepthU` emits an unbounded body for complete M128 tiles and a separately bounded tail body, removing route masks and inactive-M guards from the dominant path while preserving malformed-route behavior in the tail. It improved staged dual LDS by `0.39%`, `1.52%`, and `2.28%`.

Issuing both banks' metadata reads before decoding either bank then raised allocation to 133 VGPRs but improved latency by `1.83%`, `1.66%`, and `1.22%`. Activation prefetch depth two with schedule algorithm four reused dead A registers for projection one and improved concurrent reads by `4.96%`, `2.34%`, and `0.71%`, making the candidate faster than HIP at B1. Binding projection one's gradient and packed bank directly to registers removed the scalar pointer swaps and improved a further `1.67%`, `1.29%`, and `0.91%`.

### K32 packed-read pipeline and final identity

`DualLdsFullTileSplitKPipelineInterleavedDepthU` overlaps the next K32 pair's ten packed metadata reads with the current second-projection WMMA. The first version exposed a tail synchronization defect in which inactive consumer waves skipped the private K-counter advance and blocked at the next barrier. The retained lowering advances K and issues cooperative packed reads outside the consumer-guarded WMMA, and a regression test fixes the required ordering. The K pipeline inspects at `149 VGPR`, `41 SGPR`, `16,384 B` LDS, 64 WMMAs, and seven barriers.

The final identity changes the iteration schedule to SIA5, so the first WMMA begins as soon as its A and first LDS fragment are ready and the second M tile is admitted as its A reads retire. It retains direct global codebook reads, packed grid-Y route split 8, deferred A scheduling, batch4 IQ2_S decode, batched software BF16 RNE dependencies, and eight-store full-tile epilogue clauses, while tail paths keep masking and individual stores.

Independent regeneration of rows 35, 257, and all production keys produces byte-identical assembly and HSACO. The codebook sign transform uses `0x01010100 - payload`, which is bit-identical to `~payload + 0x01010101` for all authoritative nonzero grid bytes. The identity was checked over all 1,024 entries.

## Rejected Experiments

### Global-codebook staging scope

Reading the four lane-owned codebook entries directly from global constant storage removed codebook LDS staging and reduced LDS to 8,192 bytes, but the result was key-dependent: it improved staged dual LDS by `2.48%` at B1, tied at B4, and regressed by `1.34%` at B16. It was rejected as a universal parent and only the direct global-codebook read used by the final schedule is retained.

### N128/global geometry

`DualLdsGlobalCodebookN128InterleavedDepthU` used a 128-wide column tile with eight N WMMA tiles per projection. It passed mixed full/tail qualification at `194 VGPR`, `41 SGPR`, `16,384 B` LDS, `64` WMMAs, and `2` barriers, but the fitted B16 screen measured `57.1737 ms` versus `46.4331 ms` for HIP and `50.2026 ms` for the retained K pipeline, reaching only `0.8121x` HIP and `0.8781x` the parent. The larger accumulator envelope and N128 occupancy dominate the packed-read savings. The candidate is rejected by timing.

### Late experiments

Native `v_cvt_pk_bf16_f32` staging was rejected because the gfx1151 assembler reports the instruction unsupported. Four-way selector/sign batching reduced static issue count further but was neutral against the RNE-batch4 parent. A serial one-subtract form was retained because it removed 32 static issues without moving resources and improved the balanced B16 search result.

### Planned reopenings not pursued

Site-separated approximate BF16 policies, generated dependency-derived synchronization, and M192 ownership were identified as possible follow-ups but were not implemented or timed. They require a separate exactness or model-integration contract and are not part of the retained result.

## Closure

The retained result is the fused dual-LDS pair kernel with a full/tail K pipeline, SIA5, batch4 decode, packed-8 routes, activation prefetch, and split 8 on all three shapes.
