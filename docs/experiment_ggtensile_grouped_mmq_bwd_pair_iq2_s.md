# GGTensile Grouped MMQ Backward Pair IQ2_S Experiment

## Purpose

Implement and optimize isolated gfx1151 grouped paired IQ2_S backward kernels for the Qwen routed gate/up projections. Public dispatch, generated bundles, extension registration, packaging, and HIP fallback remain unchanged pending a separate integration review.

The promotion objective is the Qwen learned routing-prior weighted sum of per-medoid median complete-call latency. Fitted routes rank candidates. Captured and uniform, skewed, sparse-ID, repeated-ID, and boundary routes remain correctness and diagnostic controls rather than substitutes for the fitted objective.

## Exact Contract

For routed GEMM `g`:

```text
dG_g[M_g,512] x W_gate_g[512,2048]
+ dU_g[M_g,512] x W_up_g[512,2048]
-> dX_g[M_g,2048]
```

The exact aggregate-row keys are `R={16384,65536,262144}`. Each physical IQ2_S expert bank is `[256,512,656]`: each 256-value block occupies 82 bytes, each packed row contains eight blocks, and each expert occupies 335,872 bytes. Both gradient outputs and the shared gradient input are contiguous BF16. The required arithmetic accumulates both projections into one FP32 WMMA accumulator set before one BF16 RNE conversion and store.

The isolated research ABI is the production-compatible 72-byte specialized pair ABI: `first_grad_output`, `second_grad_output`, `first_packed_weight`, `second_packed_weight`, `grad_input`, `expert_indices`, `expert_offsets`, `num_experts`, `rows`, and `bytes_per_expert`. Route count and any typed split ownership remain launch geometry. A launch that assigns independent projection workgroups, writes an intermediate destination, uses atomics, or reloads a stored first projection is not a fused pair candidate.

The authoritative packed controls are `~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf` gate/up IQ2_S expert tensors with logical shape `[256,512,2048]`. The independent oracle dequantizes only selected routed experts to BF16 and sums the two routed matmuls in FP32 before BF16 conversion.

## Production Control

Installed HIP selects `GroupedBwdPairIQ2SN512K2048M64N64` below `128 * num_groups` rows and `GroupedBwdPairIQ2SN512K2048M128N64` otherwise. Both launch 128 threads as four wave32 waves, use N64 and K32, decode aligned width-16 IQ2_S groups from both banks into two LDS tiles, accumulate the first and second projections into the same FP32 registers, and perform one final BF16 store. The grid is `(2048 / 64, num_groups, 1)`.

The installed specialized ABI validates exact row count, physical bank geometry, common route geometry, expert count, and the 335,872-byte expert stride. Invalid experts, non-increasing offsets, negative starts, and offsets beyond `rows` make that route inert.

## Qualification

Correctness covers exact rows, non-aligned route tails, first and non-first routes, sparse and repeated expert IDs, deterministic reruns, independent mutation of both gradient outputs and both active weight banks, inactive-expert mutations, malformed route controls, and untouched sentinels. Candidate versus packed HIP must be BF16 bit-exact; the independently dequantized BF16 reference must remain finite with NRMSE below `0.01`.

Inspection requires gfx1151, wave32, code object v5, exact pair metadata, bounded VGPR/SGPR indices, zero private bytes and spills, no scratch, calls, or dynamic stack, and derived static WMMA/barrier counts. Independent generate/build/inspect roots must produce byte-identical source and HSACO.

Timing uses warmed rotating GPU events and includes output allocation in both installed HIP and candidate complete-call paths. Search uses five Qwen learned medoids per physical batch. Retained changes receive disjoint confirmation with 5 warmups, 25 repeats, reversed rotating order, and median-log robust confidence analysis. Pair TFLOPS is `4 * R * 512 * 2048 / (latency_ms * 1e9)`.

## Planned Search

- Keep the final M128/N64 SIA5/global-codebook/packed-route-8 identity as the production candidate. Preserve B1 and B4 as regression gates and require paired confirmation for the narrow B16 margin.
- Compare only actionable in-contract changes against the final body: exact FP32 accumulation order, one final BF16 RNE store, route semantics, zero spills, and reproducible source identity are hard gates.
- Rank on fitted weighted complete-call latency with five learned medoids per production key. Use 5 warmups, 25 repeats, balanced six-permutation launch order, exact BF16 controls, and independent plus paired robust confidence analysis.
- Retain only mechanisms with a direct correctness, resource, or timing record. Leave installed dispatch, packaging, generated bundles, and HIP fallback for a separate integration review.

## Recursive Final Review

After each optimization round, reread the contract, retained source and machine code, installed HIP control, fitted reports, correctness reports, resource inspection, and rejected experiments from first principles. Do not treat an earlier rejection as permanent when a retained mechanism changes its premise.

Classify every remaining idea as retained and measured; rejected by correctness, resources, timing, or reproducibility; contract-incompatible or deferred with an explicit prerequisite; or actionable with an exact target and qualification gate. Implement every actionable finding and repeat the review from the new premise. Completion requires a fresh recursive pass with no actionable in-contract mechanism and every selected production key correct, deterministic, resource-clean, independently reproducible, and faster than its exact HIP control.

## Completion Record

This section is updated after every coherent implementation, correctness, timing, or failed experiment.

### Campaign opened and production path audited

The pair target was confirmed independently from the non-paired down projection. The API receives two `[R,512]` BF16 gradient outputs and two IQ2_S banks with physical shape `[256,512,656]`, then returns one `[R,2048]` BF16 gradient input. The two banks each retain the familiar 335,872-byte expert stride, but their row count and packed row width are the transpose-side geometry of the down bank; the non-paired `[256,2048,164]` contract does not apply.

The production M64 and M128 bodies are true fused controls. For every K32 step they decode both packed banks into separate LDS regions, accumulate projection zero and projection one serially into the same FP32 WMMA registers, and store only after the complete 512-wide reduction. The specialized launch ABI and grid were confirmed from the generated wrapper and bundle selector. This establishes the minimum acceptable GGTensile semantic boundary: reusable decode and WMMA writers are allowed, but two independent kernels, atomic accumulation, or a BF16 intermediate are not.

Installed fitted-prior baseline timing is the next checkpoint. No GGTensile code change or performance claim is recorded yet.

### Installed fitted-prior search baseline

The research-only direct launcher for the installed specialized artifacts matches the public pair bit-for-bit on a 35-row three-route check across 71,680 BF16 outputs. This validates the 72-byte argument layout, `(32, groups, 1)` grid, 128-thread workgroup, zero dynamic LDS, and direct M64/M128 symbol selection before using the launcher for kernel-only timing.

Three warmups and nine rotating GPU-event repeats over all five fitted search medoids give:

| Key | Public complete ms / TFLOPS | Direct kernel ms / TFLOPS | Public minus kernel |
| --- | ---: | ---: | ---: |
| B1 | `6.0837 / 11.30` | `6.0913 / 11.28` | `-0.0076 ms` |
| B4 | `13.6510 / 20.14` | `13.6629 / 20.12` | `-0.0119 ms` |
| B16 | `47.4477 / 23.17` | `47.4505 / 23.17` | `-0.0028 ms` |

The warmed output allocation and public wrapper are below event noise, so complete-call and kernel-only ranking are effectively identical for this control. The 94.14%-weight B1 medoid dispatches M64. Two other M64 medoids carry 1.95% weight in total; the three sparse B1 medoids carrying 3.91% cross the production threshold to M128. Every B4 and B16 medoid dispatches M128. All direct outputs are bit-identical to the public result.

The durable report is `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-s-installed-search9.json`. These search medians establish the first GGTensile promotion targets; confirmation-bank timing remains reserved for competitive candidates.

### First fused GGTensile build checkpoint

A separate paired backward problem, solution, contract, kernel-spec, ABI, validation, physical-plan, lowering, writer, runtime, and inspection boundary now represents the fused operation. It does not overload the existing non-paired grouped identity. Canonical M64/N64 and M128/N64 keys round-trip for bounded and all production row counts. Both use four wave32 waves, K32, width-16 IQ2_S decode, serial route ownership, one decoded-weight LDS tile, one shared FP32 accumulator set, and one BF16 epilogue.

The first arithmetic anchor interleaves the pair at each K32 step: decode bank zero into LDS, accumulate gradient zero, retire LDS consumers, overwrite that LDS image from bank one, accumulate gradient one into the same registers, then advance K. Scalar pointer-pair swaps reuse the ordinary decoder and WMMA writer without copying an assembly body. The codebook is staged once per workgroup. This sequential single-LDS design deliberately pays two synchronization pairs per K32 and is a semantic anchor; a two-LDS combined-decode body is the first optimization premise after timing.

Both bounded artifacts assemble and link for gfx1151. Strict inspection reports:

| Geometry | VGPR / SGPR / LDS | Static WMMA / barriers | Private bytes / spills |
| --- | ---: | ---: | ---: |
| M64/N64 | `81 / 41 / 12,288 B` | `16 / 5` | `0 / 0` |
| M128/N64 | `121 / 41 / 12,288 B` | `32 / 5` | `0 / 0` |

Both have bounded register indices, exact 72-byte metadata, wave32, code object v5, no scratch, calls, or dynamic stack, and one emitted store phase after both projection reductions. On the first 35-row three-route check, each geometry matches all 71,680 public-HIP BF16 outputs exactly. The build record is `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-s-pilot/build.json`. Full bounded mutation and independent-oracle qualification remains outstanding; no candidate timing is admitted yet.

### Bounded fused correctness qualification

M64/N64 and M128/N64 pass the complete 35-row route-boundary matrix on real gate/up IQ2_S banks. Candidate output is bit-identical to public packed HIP for the baseline, deterministic rerun, independent first- and second-gradient mutations, route mutation, and independent active first- and second-bank mutations. Mutating an inactive expert in either bank is inert. Both gradient mutations and both active-bank mutations change output elements, proving that each fused projection remains live.

Invalid first experts, negative first cumulative offsets, and final offsets beyond `rows` preserve sentinel rows owned by the malformed routes while later or prior valid routes continue to write. The independent oracle dequantizes the selected gate/up experts to BF16, performs both matmuls in FP32, sums before BF16 conversion, and differs in only 23 of 71,680 values. Maximum absolute error is `0.00390625` and NRMSE is `6.60e-5` for both geometries.

The durable report is `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-s-pilot/correctness.json`. The first fused anchors clear correctness and resource gates and may now be built for production rows and timed. The M64 and M128 anchors are both retained until fitted timing establishes the geometry parent.

### First fitted geometry screen

Both geometries build and inspect without resource movement at all three production keys and remain bit-identical to public HIP on every fitted medoid. Three warmups and nine rotating event repeats include output allocation in each complete-call path:

| Key | Public complete ms | M64 complete ms / speedup | M128 complete ms / speedup | Parent |
| --- | ---: | ---: | ---: | --- |
| B1 | `5.6191` | `7.2442 / 0.7757x` | `5.6646 / 0.9920x` | M128 |
| B4 | `13.4632` | `24.1794 / 0.5568x` | `13.9185 / 0.9673x` | M128 |
| B16 | `47.6074` | `96.6552 / 0.4925x` | `52.7478 / 0.9025x` | M128 |

M64 is rejected for every key. Its low accumulator pressure cannot repay doubling route-tile decode and synchronization work, and its deficit grows to roughly twofold at B16. M128 becomes the sole arithmetic parent. It is effectively tied with HIP at B1, trails by 3.3% at B4, and trails by 9.7% at B16; no confirmation is warranted before optimization.

The durable report is `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-s-production/geometry-search9.json`. The growing long-route deficit supports the first planned mechanism rather than a local geometry sweep: decode both banks into disjoint LDS images within each K32 step, synchronize once before both WMMA phases, and synchronize once before overwrite. This removes one producer/consumer barrier pair per dynamic K32 while preserving first-then-second FP32 accumulation order. M64 does not transfer to that experiment.

### Dual-LDS combined decode

`DualLdsInterleavedDepthU` assigns decoded bank zero to LDS bytes `0..4095`, bank one to `4096..8191`, and the IQ2_S codebook to `8192..16383`. Both banks are decoded before one consumer barrier, followed by the two projection WMMA phases and one overwrite barrier. This reduces the dynamic K32 loop from four barriers to two without changing accumulation order.

The production artifacts use `121 VGPR`, `41 SGPR`, and `16,384 B` LDS with zero private memory or spills. The static body has 32 WMMAs and three barriers, including codebook staging. Against the sequential M128 parent, weighted complete-call latency improves by `2.40%`, `3.20%`, and `4.88%` at B1/B4/B16. It still trails contemporaneous public HIP by `3.1%`, `3.2%`, and `11.3%`, so dual LDS is retained as an optimization parent rather than a promotion candidate.

Build, correctness, and timing records are under `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-s-dual-lds/`.

### Global-codebook screen

`DualLdsGlobalCodebookInterleavedDepthU` removes codebook staging and reads the four lane-owned codebook entries directly from global constant storage. LDS falls to `8,192 B`, static barriers fall to two, and the artifact remains at `121 VGPR` with zero spills.

The result is key-dependent: it improves staged dual LDS by `2.48%` at B1, is effectively tied at B4 (`0.9996x`), and regresses by `1.34%` at B16. The mechanism is rejected as a universal parent because the long-route objective is the unresolved case. Records are under `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-s-global-codebook/`.

### Full-tile specialization

`DualLdsFullTileSplitInterleavedDepthU` emits an unbounded body for complete M128 tiles and a separately bounded tail body. This removes route-row masks and inactive-M guards from the dominant full-tile path while preserving malformed-route behavior in the tail path. Rows 35 and 257 exercise tail-only and mixed full/tail ownership.

Resources remain `121 VGPR`, `41 SGPR`, and `16,384 B` LDS with zero spills. The source contains 64 WMMAs and five barriers because both bodies are present; each executed path performs the original 32 WMMAs. Candidate output is bit-exact to HIP for adversarial tails, exact full tiles, mixed partitions, deterministic reruns, and independent mutations. Weighted latency improves over staged dual LDS by `0.39%`, `1.52%`, and `2.28%` at B1/B4/B16. Records are under `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-s-full-tile-split/`.

### Concurrent packed-bank reads

`DualLdsFullTileSplitConcurrentReadsInterleavedDepthU` issues both banks' IQ2_S metadata reads before decoding either bank. Separate payload registers raise allocation to `133 VGPR`; LDS remains `16,384 B`, with no private memory or spills. The candidate passes the same row-35 and mixed row-257 controls.

Over full-tile split, weighted complete-call latency improves by `1.83%`, `1.66%`, and `1.22%` at B1/B4/B16. The mechanism is retained. Build and timing records are under `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-s-concurrent-reads/`.

### Activation prefetch and direct pointers

`DualLdsFullTileSplitConcurrentReadsPrefetchAInterleavedDepthU` uses global-read prefetch depth two and schedule algorithm four. While projection zero consumes its current A half, the dead A registers are refilled from projection one. The candidate uses `149 VGPR`, `41 SGPR`, and `16,384 B` LDS, with zero spills; static VALU issues fall from 1,228 to 1,148. It improves concurrent reads by `4.96%`, `2.34%`, and `0.71%`, becoming the first candidate faster than public HIP at B1.

`DualLdsFullTileSplitDirectPointersPrefetchAInterleavedDepthU` then binds projection one's gradient and packed bank directly to `s12:15`. It removes scalar pointer swaps and reuses projection zero's IQ2_S block/group addresses. Resources stay at `149 VGPR`; static VALU issues fall to 1,122 with 26 VOPD instructions and zero spills. It improves the A-prefetch parent by `1.67%`, `1.29%`, and `0.91%`. Its fitted report records public-relative speedups of `1.062x`, `0.985x`, and `0.929x`, so it becomes the retained non-pipelined parent.

Both variants are bit-exact on the adversarial and mixed full/tail suites. Records are under `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-s-prefetch-a/` and `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-s-direct-pointers/`.

### K32 packed-read pipeline

`DualLdsFullTileSplitKPipelineInterleavedDepthU` overlaps the next K32 pair's ten packed metadata reads with the current second-projection WMMA. After each current LDS image is fully consumed, the next pair is decoded into the same disjoint LDS banks while eight next first-projection A reads remain pending. It preserves first-then-second FP32 accumulation and one final BF16 RNE store.

The first generated schedule exposed a tail synchronization defect during qualification: inactive M-consumer waves skipped a callback that advanced their private scalar K counter, then re-entered the loop after active waves had exited and blocked at the next barrier. The retained lowering advances K and issues cooperative packed reads outside the consumer-guarded WMMA. Active waves retain A prefetch callbacks; inactive decoder waves use explicit `vmcnt(5)` and `vmcnt(0)` waits while active waves use `vmcnt(13)` and `vmcnt(8)`. A regression test fixes the required ordering of K advance before the inactive-wave branch.

All production artifacts inspect at `149 VGPR`, `41 SGPR`, `16,384 B` LDS, 64 static WMMAs, seven static barriers, zero private bytes, and zero VGPR/SGPR spills. Row 35 passes the packed-HIP baseline, deterministic rerun, both gradient mutations, both active-bank mutations, inactive-bank mutations, route mutation, malformed-route sentinels, and the independent FP32 oracle. Row 257 passes two-full-plus-tail, exact-full-then-tail, tail-full-tail, deterministic, and full-path mutation profiles bit-for-bit.

A focused three-warmup, nine-repeat rotating comparison against the direct-pointer parent gives:

| Key | Public complete ms | Direct parent ms | K pipeline ms | Pipeline / direct | Pipeline / public |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | `5.6062` | `5.4731` | `5.2944` | `1.0338x` | `1.0589x` |
| B4 | `13.5230` | `13.5906` | `13.2933` | `1.0224x` | `1.0173x` |
| B16 | `47.2263` | `52.7560` | `51.0386` | `1.0336x` | `0.9253x` |

The K pipeline is retained as the fitted-search parent. It wins every key against direct pointers and now beats public HIP at B1 and B4, but B16 remains approximately `7.5%` slower and is the next optimization target. Production integration and fallback dispatch remain out of scope. Build, correctness, full-path, and timing records are under `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-s-k-pipeline/`.

### HIP-shaped N128/global-codebook geometry rejection

The installed HIP body uses a 128-wide N tile, so a distinct `M128/N128` identity was screened before deeper scheduling work. `DualLdsGlobalCodebookN128InterleavedDepthU` uses the existing direct global codebook mechanism, eight N WMMA tiles per projection, and no codebook LDS staging. Row 35 and row 257 mixed full/tail qualification remain bit-exact, and strict inspection reports zero private bytes/spills with `194 VGPR`, `41 SGPR`, `16,384 B` LDS, `64` static WMMAs, `2` barriers, `172` VMEM operations, and `128` LDS operations.

The static stream resembles HIP, but the fitted B16 search rejects it decisively. With three warmups and nine rotating complete-call repeats over the five learned medoids, weighted latency is `46.4331 ms` for installed HIP, `50.2026 ms` for the retained K pipeline, and `57.1737 ms` for N128/global. The candidate reaches only `0.8121x` HIP and `0.8781x` the retained parent. The larger accumulator/register envelope and N128 occupancy behavior dominate the apparent packed-read savings. The candidate is rejected by timing, not correctness or resources. Build, correctness, and timing records are under `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-s-n128-global/`.

### Interleaved WMMA waits

`DualLdsFullTileSplitKPipelineInterleaveWmmaWaitsDepthU` changes the retained K-pipeline iteration schedule from SIA4 to SIA5. It begins the first WMMA as soon as its A and first LDS fragment are ready, drains the second LDS fragment separately, and then admits the second M tile as its A reads retire. Packed-read, direct-pointer, projection ordering, K advance, and LDS barrier semantics are unchanged.

The candidate stays at `149 VGPR`, `41 SGPR`, and `16,384 B` LDS with zero private memory or spills. Static work is otherwise identical to the retained K pipeline, while explicit waits increase from `46` to `70`. Row 35 passes the full adversarial and malformed-route matrix; row 257 passes mixed full/tail ownership and deterministic mutation checks bit-for-bit.

Three-warmup, nine-repeat fitted searches give:

| Key | Public ms | K-pipeline parent ms | SIA5 ms | SIA5 / parent | SIA5 / public |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | `5.7069` | `5.6238` | `5.3401` | `1.0531x` | `1.0687x` |
| B4 | `13.4140` | `13.4536` | `13.2529` | `1.0151x` | `1.0122x` |
| B16 | `47.1559` | `52.0769` | `50.6613` | `1.0279x` | `0.9308x` |

SIA5 becomes the fitted-search parent because it improves every key and preserves the B1/B4 HIP wins. Promotion still requires the disjoint 5-warmup/25-repeat confirmation, robust confidence analysis, and independent rebuild. B16 remains about `6.9%` behind HIP. Records are under `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-s-k-pipeline-sia5/`.

### Final SIA5 global-codebook packed-route identity

The retained constructor is `GroupedBackwardPairSolution.iq2_s_m128_n64_sia5_global_codebook_packed_split_routes_8()`. It combines the two-LDS full/tail K pipeline, SIA5 interleaved WMMA waits, direct global codebook reads, packed grid-Y route split factor 8, deferred A scheduling, four contiguous global codebook loads per projection pair, batch4 IQ2_S decode, batched software BF16 RNE dependencies, and eight-store full-tile epilogue clauses. Tail paths retain masking and the original individual-store form. Projection order and one shared FP32 accumulator set are unchanged.

Every production artifact is gfx1151 code object v5 with wave32 and 128 threads. Rows 35, 257, 16384, 65536, and 262144 inspect at `149 VGPR`, `41 SGPR`, `8192 B` LDS, zero private bytes, zero VGPR/SGPR spills, 64 static WMMAs, six barriers, 232 VMEM operations, 192 LDS operations, and 60 waits. The final direct-codebook decoder has 1598 static VALU issues, 1688 VALU operations, and 90 VOPD instructions. The codebook sign transform uses `0x01010100 - payload`: all authoritative IQ2_S grid bytes are nonzero, so this is bit-identical to `~payload + 0x01010101` without the separate NOT instruction. The identity was checked over all 1024 codebook entries.

Balanced five-warmup confirmation over the five fitted medoids gives 25 repeats for B1/B4 and 50 repeats for B16:

| Key | HIP complete ms | SIA5/global candidate ms | Candidate over HIP | Independent 95% CI | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | `5.8515` | `4.7238` | `-19.28%` | `-21.34..-17.17%` | `-19.92..-17.44%` |
| B4 | `13.7111` | `12.4234` | `-9.38%` | `-10.05..-8.70%` | `-9.90..-8.66%` |
| B16 | `47.8251` | `47.2100` | `-1.27%` | `-2.13..-0.39%` | `-1.54..-0.76%` |

The candidate is independently and paired-confirmed faster than HIP at all three production keys. Against the RNE-batch4 parent, independent intervals are entirely faster at all keys: `-19.44..-14.49%`, `-10.03..-8.33%`, and `-3.03..-0.65%` for B1/B4/B16.

The final production correctness report compares exact candidate output with installed HIP over 33,554,432 B1 elements, 134,217,728 B4 elements, and 536,870,912 B16 elements. Every comparison has zero differing BF16 elements, zero absolute error, finite output, and zero differences on the deterministic rerun. All five fitted medoids per key also have zero candidate/HIP differences. The row-35 adversarial report retains the independent FP32 oracle result of 23 differing BF16 values out of 71,680 with NRMSE `6.60e-5`; malformed-route sentinels, active/inactive bank mutation, route mutation, gradient mutation, mixed full/tail ownership, and deterministic controls pass.

Independent regeneration of rows 35, 257, 16384, 65536, and 262144 produces byte-identical assembly, solution JSON, and HSACO. The retained artifact root is `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-s-k-pipeline-sia5-global-rne-batch4-sign-sub-packed8/`; reproducibility is recorded in `reproducibility.json`, production exactness in `production-correctness.json`, and full-path controls in `full-path-correctness.json`.

### Final rejected late experiments

Native `v_cvt_pk_bf16_f32` staging was rejected because the gfx1151 assembler reports the instruction unsupported. The source-only artifact was not treated as a valid code object. Four-way selector/sign batching reduced static issue count further but was neutral against the RNE-batch4 parent, so it is not retained. The serial one-subtract form was retained because it reduced 32 static issues without moving resources and improved the balanced B16 search result.

The final identity has no actionable in-contract mechanism left in this isolated scope. Installed dispatch and packaging remain intentionally unchanged.
