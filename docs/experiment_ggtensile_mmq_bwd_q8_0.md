# GGTensile Dense MMQ Backward Q8_0 Plan

## Purpose

Extend the strict gfx1151 GGTensile dense MMQ backward campaign to the production Q8_0 weight shapes used by the DeepSeek dense workload. The campaign must improve the complete fused packed-weight backward kernel, not a predecoded or prepared-weight surrogate.

Q8_0 is a separate quantization campaign from Q3_K, Q4_K, and Q5_K. It may reuse quant-neutral WMMA, A-address, LDS, synchronization, store, inspection, and campaign infrastructure only when the emitted code and identity remain quant-aware. Q8_0 has its own byte/block decoder and must receive its own tuning knobs, correctness fixtures, inventory, catalog, and resource evidence.

The campaign is complete only after every exact production key that is retained for GGTensile is faster than the HIP control, all exact keys have independent correctness and mutation coverage, and a recursive optimization-exhaustion review finds no new valid in-contract mechanism. The final review must explain the remaining bottleneck with lower-bound, resource, timing, or counter evidence.

## Contract

Target only:

- gfx1151, wave32, WMMA V1, and the existing 40-byte dense-backward kernarg ABI.
- BF16 grad-output and grad-input, FP32 WMMA accumulation, and packed GGUF Q8_0 weights.
- In-kernel Q8_0 decode from the authoritative packed tensor.
- Exact `ProblemType` plus exact `ProblemSize` identity for each generated kernel.
- Complete fused-kernel timing, including global packed reads, Q8_0 decode, LDS staging, WMMA, synchronization, and BF16 stores.

Do not introduce prepared weights, BF16 shadows, external decode workspaces, split-K, persistent workgroups, grouped MMQ, hidden caches, or model-owned paired backward APIs. Unsupported shapes must fail closed to HIP or the existing generic path; an exact campaign artifact must not silently repair a mismatched shape.

Dense backward coordinates are:

```text
M = rows
N = in_features
K = out_features

grad_input[M,N] = grad_output[M,K] @ dequant(weight[K,N])
```

Q8_0 uses 32-value blocks with 34 packed bytes. For an exact key, the logical weight shape is `[K,N]` and the packed shape is `[K, (N/32)*34]`.

## Exact Production Scope

The ordinary DeepSeek workload has six `(N,K)` families crossed with `M={2048,8192,32768}`. This produces 18 ordinary exact keys:

| Family | `(N,K)` | Packed weight shape `[K,bytes]` | Calls |
| --- | ---: | ---: | ---: |
| Attention Q-A | `(4096,1024)` | `[1024,4352]` | 43 |
| Attention Q-B | `(1024,32768)` | `[32768,1088]` | 43 |
| Attention KV | `(4096,512)` | `[512,4352]` | 43 |
| Attention output B | `(8192,4096)` | `[4096,8704]` | 43 |
| Shared gate/up | `(4096,2048)` | `[2048,4352]` | 86 combined |
| Shared down | `(2048,4096)` | `[4096,2176]` | 43 |

The Q8_0 language-model head adds five exact chunk keys:

```text
(M,N,K) =
(32,4096,129280)
(64,4096,129280)
(128,4096,129280)
(256,4096,129280)
(512,4096,129280)
```

The complete campaign scope is therefore 23 exact keys. M512 is the primary complete-loss chunk; M256 is the lower-memory alternative; M32/M64/M128 remain required for chunk fallback, capacity behavior, and correctness coverage.

The ordinary shapes and call counts are sourced from `tests/deepseek_dense_cases.py` and `bench/mmq_benchmark_common.py`. Existing dispatch evidence and historical HIP measurements are recorded in `docs/mmq_bwd_optimization.md`.

## Baseline And Priorities

Use the existing HIP Q8_0 kernels as both correctness and performance controls. Establish fresh same-process controls before selecting any GGTensile solution. Do not compare timing from prior source-built bundles directly with the new catalog.

Prioritize large margins in this order:

1. Q-B `(N,K)=(1024,32768)` and attention output B `(8192,4096)`, which have the largest long-batch kernel costs.
2. Shared gate/up `(4096,2048)`, which has twice the projection call count.
3. Shared down `(2048,4096)` and Q-A `(4096,1024)`.
4. KV `(4096,512)`.
5. LM-head M512/M256, then M128/M64/M32 for chunk fallback and capacity coverage.

The first screening matrix should include all six ordinary families at M2048, M8192, and M32768. Use exact rows as the primary timing axis and retain call-weighted totals for model priority. Complete-loss timing must remain a separate selector from isolated LM-head timing.

## GGTensile Architecture

### Quant-neutral shared body

Reuse the existing writer layers only where the Q8_0 tile contract matches:

- exact workgroup flattening and launch mapping;
- A global addressing, row-tile traversal, and optional row-state lifetime management;
- WMMA accumulator allocation, issue order, and FP32-to-BF16 output conversion;
- LDS barriers, wait dependencies, local-read ownership, and final stores;
- diagnostic floor generation and static resource inspection;
- strict source hashing, immutable generation/build/inspect/correctness/screen/confirmation phases;
- resource rejection for private storage, spills, scratch, calls, and dynamic stack.

Q8_0 must not be forced through K-family decode helpers. The writer should have a quant specification or backend that owns block bytes, payload width, scale placement, vector load shape, decode arithmetic, decoder rows, packed-row addressing, and LDS-facing value layout.

### Q8_0 backend boundary

The Q8_0 backend must define:

- 32-value block and 34-byte packed-row layout;
- scalar `d` loading and byte payload loading;
- exact signed int8 reconstruction and scale application;
- metadata and payload register lifetimes;
- packed load width and coalescing strategy;
- Q8_0-specific LDS layout and row padding;
- exact reduced-K behavior for block-aligned and partial campaign fixtures;
- an independent reference decoder for correctness.

Q8_0-specific solution identity fields should be added only for real alternate emitters. Candidate controls include packed payload load width, scale load strategy, row padding, packed payload/scale ordering, decoder-row ownership, and Q8-specific schedule or traversal. A field is invalid unless it changes emitted ISA or ownership and has correctness coverage.

Existing Q3_K/Q4_K/Q5_K controls must remain strict and inert for Q8_0. Q8_0 artifacts must have distinct `ProblemType`, solution identity, symbols, packed-row accounting, and catalogs.

### Possible reuse boundary

Reuse the proven `128x128x32` WMMA body only as the initial Q8 control. The Q8 decoder may make a different `MacroTile1`, LDS stride, or `DepthU` necessary. A common writer helper is acceptable when it takes a quant backend contract and emits identical quant-neutral code around a quant-specific decode block. Copying Q4/Q5 byte logic or adding Q8 behavior through unchecked conditionals is not acceptable.

## Campaign Phases

### Phase 1: Inventory and control

- Add a versionless Q8_0 inventory for the 23 exact keys, representative tensors, call counts, packed shapes, and historical HIP controls.
- Add a Q8_0 selected-solution catalog with strict identity validation.
- Extend campaign loading and physical packed-row accounting from fixed K-family tables to an explicit Q8_0 specification.
- Generate a fresh Q8_0 control before optimization.
- Add reduced-K and one-hot packed fixtures that exercise `d`, every payload byte, signed extremes, block boundaries, and row boundaries.

### Phase 2: Backend and correctness

- Add strict Q8_0 `ProblemType` construction and validation.
- Implement packed Q8_0 decode with no dense shadow or external workspace.
- Validate exact HIP equality for all ordinary keys and LM-head chunks.
- Require independent dequantized BF16 reference checks wherever accumulation-order differences are understood.
- Rewrite the complete grad-output and packed-weight tensors and repeat candidate/HIP comparisons.
- Reject any solution with private storage, spills, scratch, calls, dynamic stack, invalid ABI, or source/resource mismatch.

### Phase 3: Large-margin search

Start from the current Q8 HIP assembly and a generated GGTensile control. Measure complete solutions, not isolated instruction fragments.

Search in this order:

1. Q8 decoder load coalescing and scale/payload ordering.
2. Q8 LDS row padding and unswizzled versus existing swizzled layouts.
3. One versus two decoded-B buffers, only after lower bounds identify overlap potential.
4. `128x64`, `256x64`, and other geometries only when accumulator/register/resource estimates justify them.
5. Exact M traversal and WGM1/2/4/8 for high-cost long-row families.
6. DepthU, packed payload sharing, PGR/PLR, SIA, store priority, and Q8-specific address lifetime changes.
7. LM-head chunk geometry and active-wave ownership after ordinary families are competitive.

Each candidate must pass correctness and inspection before timing. Use serial warmed rotating controls. Screening uses nine repeats; final confirmation uses 25 repeats. Timing is authoritative. Static instruction reductions are explanatory unless they produce a stable measured gain.

### Phase 4: Selection and confirmation

Retain an exact candidate only when:

- it is bit-exact against HIP for the exact key;
- independent-reference behavior is understood;
- grad-output and packed-weight mutation checks pass;
- independent source generation is byte-identical;
- resource and ABI gates pass;
- every resource-bearing mechanism beats its exact assembly control by more than 2% in a stable 25-repeat bracket;
- no exact key using the same retained solution suffers a stable material regression.

Every selected exact Q8 key must beat HIP. HIP fallback is acceptable only for unmatched keys or while a campaign key remains unselected; it is not an acceptable final result for an exact selected key.

### Phase 5: Lower bounds and bottlenecks

For each major family, measure complete, WMMA/A/LDS-floor, and decode/LDS-floor artifacts with the same ABI and launch contract. Explain the remaining gap using:

- WMMA throughput and accumulator occupancy;
- Q8 payload and scale traffic;
- decode VALU issue pressure;
- LDS bank conflicts and synchronization;
- active waves, VGPR allocation, and launch geometry;
- model call count and complete-loss amortization.

## Recursive Optimization-Exhaustion Review

The campaign cannot stop after a single successful Q8 implementation. Before completion, reread this plan, the Q8 HIP and GGTensile logs, Q3/Q4/Q5 experiment records, CK and TensileLite notes, normalized disassembly, profiler/counter evidence, lower bounds, inventories, rejected candidates, correctness reports, and gfx1151 ISA/LLVM definitions.

Classify every remaining idea as:

- retained and measured;
- rejected by correctness, resource, timing, or reproducibility evidence;
- contract-incompatible or explicitly deferred with a prerequisite; or
- actionable and requiring another implementation and measurement cycle.

A plan or implementation change creates a new premise and invalidates the previous stopping condition. The review must be the final step of the campaign and cannot pass in the same iteration that discovers an actionable mechanism.

The campaign is exhausted only when every valid large-margin mechanism has been implemented or rejected, smaller plausible mechanisms have been tested after the large margins close, all 23 exact keys have final evidence, and the remaining bottleneck is explained quantitatively. Public runtime dispatch remains deferred until broader dense multi-quant coverage, artifact packaging, dispatch engineering, and complete Qwen/DeepSeek workload validation are complete.

## Completion Record

This section is updated after every coherent implementation milestone. Code milestones are committed; documentation-only updates remain uncommitted unless explicitly requested.

- [x] Q8_0 exact inventory and strict quant identity. The versionless inventory contains 18 ordinary and five LM-head keys, uses 34-byte/32-value physical-row accounting, and keeps `Q8KExtraction` inert for other quant types.
- [x] Initial Q8_0 backend and shared WMMA-body boundary. The backend has quant-specific packed reads, FP16 scale conversion, signed-int8 extraction, LDS writes, resource accounting, and inspection while reusing the quant-neutral WMMA body across ordinary, compact, and small-M geometries.
- [x] Fresh HIP/GGTensile controls and independent packed decoder fixtures. K32/K64 one-hot fixtures are exact across complete rows, and all 23 production keys pass HIP, independent-reference, grad-output mutation, and packed-weight mutation checks.
- [x] Ordinary 18-key correctness and mutation coverage. Every selected key passed exact HIP comparison, independent reference, full grad-output mutation, and packed-weight mutation.
- [x] LM-head five-key correctness and chunk fallback coverage. M32 `32x64`, M64 `64x64`, M128 `128x64`, and M256/M512 `256x64` controls pass independent reference/mutation checks and final confirmation.
- [ ] Large-margin decoder, LDS, ownership, and geometry search. Ordinary screening has established `LdsPadB=8` as the dominant reusable mechanism, retained `256x64` for attention output/Q-A/KV, found a Q-B-only DepthU64 branch, and rejected broad WGM2, pad16/24, XOR4/8/16 one-buffer layouts, `64x128`, `256x128`, two decoded-B buffers, broad scalar loads, PLR2, and broad next-packed prefetch. Smaller per-key closure and LM-head geometry remain open.
- [x] Per-key selection with every retained key faster than HIP. The ordinary and LM-head catalogs are selected and confirmed; every one of 23 exact keys beats HIP.
- [x] Lower-bound and bottleneck explanation. Representative complete/WMMA-A-LDS/decode-LDS floors were measured for Q-A, Q-B, output-B, shared gate/up, shared-down, and LM M512.
- [x] Independent reproducibility and final 25-repeat confirmation. Current ordinary roots are byte-identical across 18 keys; the LM roots are byte-identical across five keys; ordinary and LM confirmation phases use 25 serial repeats.
- [ ] Recursive optimization-exhaustion review with no actionable mechanism remaining. The review is intentionally deferred until the final layout neighborhood and any newly actionable lower-bound idea are tested.
- [ ] Public runtime dispatch, deferred.

### Ordinary screening record

The first fresh 18-key `128x128x32` control measured a call-weighted candidate/HIP ratio of `1.0329126789987204`; it was not competitive at several M2048/M8192 keys. Unswizzled eight-BF16 row padding then reduced representative long-row candidate latency by roughly 15-25%. PGR2/SIA5 plus pad8 produced candidate/HIP ratios from about `0.66` to `0.95` across the ordinary matrix and made every screened exact key faster than HIP.

Geometry is quant- and shape-specific. Padded `256x64` reduced attention-output M2048/M8192/M32768 from `5.041/19.522/75.521 ms` to `4.234/16.776/65.925 ms`; it also improved Q-A and KV, but regressed Q-B. Padded `128x64` is preferred over `256x64` for long-row shared gate/up and shared-down. Broad WGM2 regressed compact bodies.

Q8 scalar `global_load_b32` extraction was neutral for most families but improved Q-B M8192 by about 5% versus packed `global_load_b128`. A corrected Q8 DepthU64/XOR8 control measured `22.285 ms` and `0.702x` HIP at Q-B M8192, versus `24.113 ms` for the padded DepthU32 packed control, but regressed output-B. The initial DepthU64 failure exposed overlapping Q8 block-base, payload-pointer, and persistent-A state; a reusable temporary payload pointer fixed the ownership without increasing resources. The corrected body uses 220 VGPR, 16 SGPR, and 16 KiB LDS with no spills or private storage.

The two decoded-B buffer path was extended to the Q8 XOR8 store layout and passed exact HIP, independent-reference, grad-output mutation, and packed-weight mutation checks. It was timing-neutral on the discriminator keys and is rejected. Pad16/24, XOR4/8/16 one-buffer layouts, PLR2, SIA3, `64x128`, and `256x128` also lost. Next-tile packed prefetch remains only a possible exact-key small mechanism because its broad effects were neutral or unfavorable.

The selected ordinary catalog uses compact padded `256x64` for Q-A, KV, attention output, and the M2048 shared projections; padded `128x64` for M8192/M32768 shared gate/up and shared-down; DepthU64/XOR8 for Q-B M8192; padded scalar next-prefetch for Q-B M2048; and padded packed `128x128` for Q-B M32768. Final serial 25-repeat candidate/HIP ratios range from `0.5169531577037376` to `0.8682106182049584`. The call-weighted ordinary ratio is `0.6892668262402388`. A second independent root generated byte-identical assembly and matching resource tuples for all 18 selected artifacts.

### LM-head screening record

The first representable LM-head controls all beat HIP: padded `64x128` measured `5.873 ms` at M64 (`0.836x` HIP), padded `128x64` measured `6.032 ms` at M128 (`0.789x`), and padded `256x64` measured `9.519/18.784 ms` at M256/M512 (`0.928/0.858x`). Wider `128x128` and `256x128` bodies were slower.

M32 required a true two-wave exact geometry rather than a partial 64-row tile. Q8-specific 64-thread `32x64` and `32x128` bodies were added with generalized decoder-row spacing and strict rejection for other quant types. Both pass independent reference and producer mutation checks. `32x64` measured `4.787 ms` versus `8.259 ms` HIP (`0.5796040876830744x`); `32x128` measured `0.908x` HIP and is rejected. The initial M32 control uses 90 VGPR, 16 SGPR, and 5 KiB LDS with no private storage or spills.

### Lower bounds and residual bottleneck

| Representative key | Complete | WMMA/A/LDS floor | Decode/LDS floor | Floor sum / complete |
| --- | ---: | ---: | ---: | ---: |
| Q-A M32768 | 8.208 ms | 6.260 ms | 1.746 ms | 0.975x |
| Q-B M8192 | 22.329 ms | 12.504 ms | 8.717 ms | 0.950x |
| Output-B M32768 | 65.968 ms | 49.257 ms | 15.346 ms | 0.979x |
| Shared gate/up M32768 | 18.261 ms | 12.330 ms | 5.684 ms | 0.986x |
| Shared-down M32768 | 18.728 ms | 12.373 ms | 5.743 ms | 0.967x |
| LM M512 | 17.401 ms | 13.185 ms | 5.967 ms | 1.101x |

The ordinary floors sum to within 2-5% of complete timing, so the residual is not an unhidden large scheduling gap: decode VALU/VMEM and WMMA/LDS synchronization are the two dominant components, with their overlap already close to the measured complete path. Q-B has the largest decode fraction and is the only ordinary key where DepthU64 remains beneficial. LM M512 has a floor sum above complete because its two isolated floors double-count work that overlaps in the complete next-prefetch body; its residual bottleneck is packed Q8 payload/scale traffic plus WMMA occupancy under the two-M-tile launch, not a missing correctness mechanism.

Adding the HIP-analogous `64x64` ownership reduced M64 from `5.873` to `4.487 ms` (`0.6308024774361413x` HIP). Q8-specific unswizzled pad8 DepthU64 was then generalized to compact geometries and passed all five LM correctness screens. It improves M32/M64/M128 by about 8.0%/4.7%/2.8%, measuring `4.405/4.275/5.863 ms`, but regresses M256/M512. The tentative selection therefore uses DepthU64 through M128 and DepthU32 at M256/M512. Packed extraction wins every LM key; scalar extraction is rejected. M512 alone retains next-tile packed prefetch after a `18.784` to `18.057 ms` screen improvement; SIA4 without prefetch, PGR1, store-priority removal, and WGM2 lose.
