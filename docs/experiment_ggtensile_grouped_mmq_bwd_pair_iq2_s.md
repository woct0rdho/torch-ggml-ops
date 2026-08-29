# GGTensile Grouped MMQ Backward Pair IQ2_S Experiment

## Scope And Contract

This record covers the isolated gfx1151 paired IQ2_S backward kernel for the Qwen routed gate and up projections. For each routed expert the kernel accumulates both gate/up input-gradient matmuls in one workgroup dataflow:

```text
dG_g[M_g,512] @ W_gate_g[512,2048]
+ dU_g[M_g,512] @ W_up_g[512,2048]
-> dX_g[M_g,2048]
```

The measured aggregate-row shapes are `R=16384`, `65536`, and `262144`. Each packed IQ2_S bank is `[256,512,656]`; each expert occupies 335,872 bytes. The gate and up gradient outputs and the one shared gradient input are contiguous BF16, and both projections are accumulated into one shared FP32 WMMA accumulator set before one BF16 RNE conversion and store. The IQ2_S codebook is read directly from global constant storage rather than staged in LDS.

The paired research ABI carries both gradient outputs, both packed weights, the gradient input, the device-resident route description, and the shape values. Logical paired throughput is `4 * R * 512 * 2048 / (latency_ms * 1e9)`, and the speedup ratio is HIP time divided by GGTensile time.

## Final Benchmark Results

The table shows the fastest qualified kernel found for each measured aggregate-row shape.

| Matrix shape `(R,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(16384,2048,512)` | `ggbpair_6fc121781141a30e` | `14.548` | `1.2387x` |
| `(65536,2048,512)` | `ggbpair_64ba6a8b68b4c2c6` | `22.126` | `1.1036x` |
| `(262144,2048,512)` | `ggbpair_b63deec8280c235f` | `23.290` | `1.0130x` |

The candidate is independently and paired-confirmed faster than HIP at all three keys. Independent 95% confidence intervals for candidate-minus-HIP latency are `-21.34..-17.17%`, `-10.05..-8.70%`, and `-2.13..-0.39%` at B1/B4/B16.

## Final Kernel Profiles

| Kernel hash | Geometry and ownership | Decode and schedule | VGPR / SGPR | LDS bytes | WMMAs / barriers |
| --- | --- | --- | ---: | ---: | ---: |
| `ggbpair_6fc121781141a30e` | M128/N64, split 8 | two-LDS full/tail K pipeline, SIA5, batch4 decode, packed-8 routes | `149 / 41` | `8,192` | `64 / 6` |
| `ggbpair_64ba6a8b68b4c2c6` | M128/N64, split 8 | two-LDS full/tail K pipeline, SIA5, batch4 decode, packed-8 routes | `149 / 41` | `8,192` | `64 / 6` |
| `ggbpair_b63deec8280c235f` | M128/N64, split 8 | two-LDS full/tail K pipeline, SIA5, batch4 decode, packed-8 routes | `149 / 41` | `8,192` | `64 / 6` |

All production artifacts are gfx1151 code-object-v5 wave32 kernels with 128 threads, zero private storage, zero spills, and no scratch, calls, or dynamic stack. The final stream contains `1,598` static VALU issues, `232` VMEM operations, `192` LDS operations, and `60` waits.

## Accepted Kernel Experiments

### Fused anchors and dual-LDS combined decode

The first fused anchor interleaves the pair at each K32 step: decode bank zero into LDS, accumulate gradient zero, retire the LDS consumers, overwrite the image from bank one, and accumulate gradient one into the same registers. It established the fused semantics and resource envelope at `81 VGPR / 41 SGPR / 12,288 B` for M64 and `121 / 41 / 12,288 B` for M128, with one emitted store phase after both reductions. Both geometries matched all 71,680 installed-HIP BF16 outputs on the first 35-row three-route check and passed the complete route, rerun, gradient, active-bank, inactive-bank, and malformed-route matrix. The M64 geometry was rejected by fitted timing because repeated route-tile decode cannot repay the lower accumulator pressure.

`DualLdsInterleavedDepthU` then assigned decoded bank zero to LDS bytes `0..4095`, bank one to `4096..8191`, and the codebook to `8192..16383`, so both banks decode before one consumer barrier, followed by both projection WMMA phases and one overwrite barrier. This reduced the dynamic K32 loop from four barriers to two and improved the sequential parent by `2.40%`, `3.20%`, and `4.88%` at B1/B4/B16, becoming the optimization parent.

### Full-tile split, concurrent reads, and prefetch

`DualLdsFullTileSplitInterleavedDepthU` emits an unbounded body for complete M128 tiles and a separately bounded tail body, removing route masks and inactive-M guards from the dominant path while preserving malformed-route behavior in the tail. It improved staged dual LDS by `0.39%`, `1.52%`, and `2.28%`.

Issuing both banks' metadata reads before decoding either bank then raised allocation to 133 VGPRs but improved latency by `1.83%`, `1.66%`, and `1.22%`. Activation prefetch depth two with schedule algorithm four reused dead A registers for projection one and improved concurrent reads by `4.96%`, `2.34%`, and `0.71%`, making the candidate faster than HIP at B1. Binding projection one's gradient and packed bank directly to registers removed the scalar pointer swaps and improved a further `1.67%`, `1.29%`, and `0.91%`.

### K32 packed-read pipeline and final identity

`DualLdsFullTileSplitKPipelineInterleavedDepthU` overlaps the next K32 pair's ten packed metadata reads with the current second-projection WMMA. The first version exposed a tail synchronization defect in which inactive consumer waves skipped the private K-counter advance and blocked at the next barrier; the retained lowering advances K and issues cooperative packed reads outside the consumer-guarded WMMA, and a regression test fixes the required ordering. The K pipeline passed the full adversarial, mixed full/tail, and mutation matrix at `149 VGPR`, `41 SGPR`, `16,384 B` LDS, 64 WMMAs, and seven barriers.

The final identity changes the iteration schedule to SIA5, so the first WMMA begins as soon as its A and first LDS fragment are ready and the second M tile is admitted as its A reads retire. It retains direct global codebook reads, packed grid-Y route split 8, deferred A scheduling, batch4 IQ2_S decode, batched software BF16 RNE dependencies, and eight-store full-tile epilogue clauses, while tail paths keep masking and individual stores. Production exactness compares zero differing BF16 elements over 33,554,432 B1, 134,217,728 B4, and 536,870,912 B16 elements, with zero deterministic-rerun differences. Independent regeneration of rows 35, 257, and all production keys produces byte-identical assembly and HSACO. The codebook sign transform uses `0x01010100 - payload`, which is bit-identical to `~payload + 0x01010101` for all authoritative nonzero grid bytes; the identity was checked over all 1,024 entries.

## Rejected Kernel Experiments

### Global-codebook staging scope

Reading the four lane-owned codebook entries directly from global constant storage removed codebook LDS staging and reduced LDS to 8,192 bytes, but the result was key-dependent: it improved staged dual LDS by `2.48%` at B1, tied at B4, and regressed by `1.34%` at B16. It was rejected as a universal parent and only the direct global-codebook read used by the final schedule is retained.

### N128/global geometry

`DualLdsGlobalCodebookN128InterleavedDepthU` used a 128-wide column tile with eight N WMMA tiles per projection. It passed mixed full/tail qualification at `194 VGPR`, `41 SGPR`, `16,384 B` LDS, `64` WMMAs, and `2` barriers, but the fitted B16 screen measured `57.1737 ms` versus `46.4331 ms` for HIP and `50.2026 ms` for the retained K pipeline, reaching only `0.8121x` HIP and `0.8781x` the parent. The larger accumulator envelope and N128 occupancy dominate the packed-read savings; the candidate is rejected by timing.

### Late experiments

Native `v_cvt_pk_bf16_f32` staging was rejected because the gfx1151 assembler reports the instruction unsupported. Four-way selector/sign batching reduced static issue count further but was neutral against the RNE-batch4 parent. A serial one-subtract form was retained because it removed 32 static issues without moving resources and improved the balanced B16 search result.

### Planned reopenings not pursued

Site-separated approximate BF16 policies, generated dependency-derived synchronization, and M192 ownership were identified as possible follow-ups but were not implemented or timed. They require a separate exactness or model-integration contract and are not part of the retained result.

## Qualification Summary

The final kernels pass exact packed-HIP comparison, independent FP32-accumulating references, finite-output checks, deterministic reruns, first and non-first routes, sparse and repeated experts, both-gradient and both-active-bank mutations, inactive-expert inertness, and malformed-route sentinels. The row-35 independent oracle result is 23 differing BF16 values out of 71,680 with NRMSE `6.60e-5`. Production correctness compares zero differing elements at all three aggregate sizes, and two independent build roots reproduce the selected artifacts byte-for-byte.
