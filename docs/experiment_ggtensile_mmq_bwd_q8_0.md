# GGTensile MMQ Backward Q8_0 Plan

## Purpose

Extend the strict gfx1151 GGTensile MMQ backward campaign to the production Q8_0 weight shapes used by the DeepSeek dense workload. The campaign must improve the complete fused packed-weight backward kernel, not a predecoded or prepared-weight surrogate.

Q8_0 is a separate quantization campaign from Q3_K, Q4_K, and Q5_K. It may reuse quant-neutral WMMA, A-address, LDS, synchronization, store, inspection, and campaign infrastructure only when the emitted code and identity remain quant-aware. Q8_0 has its own byte/block decoder and must receive its own tuning knobs, correctness fixtures, inventory, catalog, and resource evidence.

The campaign is complete only after every exact production key that is retained for GGTensile is faster than the HIP control, all exact keys have independent correctness and mutation coverage, and a recursive optimization-exhaustion review finds no new valid in-contract mechanism. The final review must explain the remaining bottleneck with lower-bound, resource, timing, or counter evidence.

## Contract

Target only:
- gfx1151, wave32, WMMA V1, and the existing 40-byte MMQ backward kernarg ABI.
- BF16 grad-output and grad-input, FP32 WMMA accumulation, and packed GGUF Q8_0 weights.
- In-kernel Q8_0 decode from the authoritative packed tensor.
- Exact `ProblemType` plus exact `ProblemSize` identity for each generated kernel.
- Complete fused-kernel timing, including global packed reads, Q8_0 decode, LDS staging, WMMA, synchronization, and BF16 stores.

Do not introduce prepared weights, BF16 shadows, external decode workspaces, split-K, persistent workgroups, grouped MMQ, hidden caches, or model-owned paired backward APIs. Unsupported shapes must fail closed to HIP or the existing generic path; an exact campaign artifact must not silently repair a mismatched shape.

MMQ backward coordinates are:

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
- Q-B `(N,K)=(1024,32768)` and attention output B `(8192,4096)`, which have the largest long-batch kernel costs.
- Shared gate/up `(4096,2048)`, which has twice the projection call count.
- Shared down `(2048,4096)` and Q-A `(4096,1024)`.
- KV `(4096,512)`.
- LM-head M512/M256, then M128/M64/M32 for chunk fallback and capacity coverage.

The first screening matrix should include all six ordinary families at M2048, M8192, and M32768. Use exact rows as the primary timing axis and retain call-weighted totals for model priority. Complete-loss timing must remain a separate selector from isolated LM-head timing.

## GGTensile Architecture

### Quant-neutral shared body

Reuse the existing writer layers only where the Q8_0 tile contract matches:
- exact workgroup flattening and launch mapping.
- A global addressing, row-tile traversal, and optional row-state lifetime management.
- WMMA accumulator allocation, issue order, and FP32-to-BF16 output conversion.
- LDS barriers, wait dependencies, local-read ownership, and final stores.
- diagnostic floor generation and static resource inspection.
- strict source hashing, immutable generation/build/inspect/correctness/screen/confirmation phases.
- resource rejection for private storage, spills, scratch, calls, and dynamic stack.

Q8_0 must not be forced through K-family decode helpers. The writer should have a quant specification or backend that owns block bytes, payload width, scale placement, vector load shape, decode arithmetic, decoder rows, packed-row addressing, and LDS-facing value layout.

### Q8_0 backend boundary

The Q8_0 backend must define:
- 32-value block and 34-byte packed-row layout.
- scalar `d` loading and byte payload loading.
- exact signed int8 reconstruction and scale application.
- metadata and payload register lifetimes.
- packed load width and coalescing strategy.
- Q8_0-specific LDS layout and row padding.
- exact reduced-K behavior for block-aligned and partial campaign fixtures.
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
- Q8 decoder load coalescing and scale/payload ordering.
- Q8 LDS row padding and unswizzled versus existing swizzled layouts.
- One versus two decoded-B buffers, only after lower bounds identify overlap potential.
- `128x64`, `256x64`, and other geometries only when accumulator/register/resource estimates justify them.
- Exact M traversal and WGM1/2/4/8 for high-cost long-row families.
- DepthU, packed payload sharing, PGR/PLR, SIA, store priority, and Q8-specific address lifetime changes.
- LM-head chunk geometry and active-wave ownership after ordinary families are competitive.

Each candidate must pass correctness and inspection before timing. Use serial warmed rotating controls. Screening uses nine repeats; final confirmation uses 25 repeats. Timing is authoritative. Static instruction reductions are explanatory unless they produce a stable measured gain.

### Phase 4: Selection and confirmation

Retain an exact candidate only when:
- it is bit-exact against HIP for the exact key.
- independent-reference behavior is understood.
- grad-output and packed-weight mutation checks pass.
- independent source generation is byte-identical.
- resource and ABI gates pass.
- every resource-bearing mechanism beats its exact assembly control by more than 2% in a stable 25-repeat bracket.
- no exact key using the same retained solution suffers a stable material regression.

Every selected exact Q8 key must beat HIP. HIP fallback is acceptable only for unmatched keys or while a campaign key remains unselected; it is not an acceptable final result for an exact selected key.

### Phase 5: Lower bounds and bottlenecks

For each major family, measure complete, WMMA/A/LDS-floor, and decode/LDS-floor artifacts with the same ABI and launch contract. Explain the remaining gap using:
- WMMA throughput and accumulator occupancy.
- Q8 payload and scale traffic.
- decode VALU issue pressure.
- LDS bank conflicts and synchronization.
- active waves, VGPR allocation, and launch geometry.
- model call count and complete-loss amortization.

## Recursive Optimization-Exhaustion Review

The campaign cannot stop after a single successful Q8 implementation. Before completion, reread this plan, the Q8 HIP and GGTensile logs, Q3/Q4/Q5 experiment records, CK and TensileLite notes, normalized disassembly, profiler/counter evidence, lower bounds, inventories, rejected candidates, correctness reports, and gfx1151 ISA/LLVM definitions.

Classify every remaining idea as:
- retained and measured.
- rejected by correctness, resource, timing, or reproducibility evidence.
- contract-incompatible or explicitly deferred with a prerequisite. or
- actionable and requiring another implementation and measurement cycle.

A plan or implementation change creates a new premise and invalidates the previous stopping condition. The review must be the final step of the campaign and cannot pass in the same iteration that discovers an actionable mechanism.

The campaign is exhausted only when every valid large-margin mechanism has been implemented or rejected, smaller plausible mechanisms have been tested after the large margins close, all 23 exact keys have final evidence, and the remaining bottleneck is explained quantitatively. Public runtime dispatch remains deferred until broader MMQ multi-quant coverage, artifact packaging, dispatch engineering, and complete Qwen/DeepSeek workload validation are complete.

## Completion Record

This section is updated after every coherent implementation milestone. Code milestones are committed; documentation-only updates remain uncommitted unless explicitly requested.
- Q8_0 exact inventory and strict quant identity. The versionless inventory contains 18 ordinary and five LM-head keys, uses 34-byte/32-value physical-row accounting, and keeps `Q8KExtraction` inert for other quant types.
- Initial Q8_0 backend and shared WMMA-body boundary. The backend has quant-specific packed reads, FP16 scale conversion, signed-int8 extraction, LDS writes, resource accounting, and inspection while reusing the quant-neutral WMMA body across ordinary, compact, and small-M geometries.
- Fresh HIP/GGTensile controls and independent packed decoder fixtures. K32/K64 one-hot fixtures are exact across complete rows, and all 23 production keys pass HIP, independent-reference, grad-output mutation, and packed-weight mutation checks.
- Ordinary 18-key correctness and mutation coverage. Every selected key passed exact HIP comparison, independent reference, full grad-output mutation, and packed-weight mutation.
- LM-head five-key correctness and chunk fallback coverage. M32 `32x64`, M64 `64x64`, M128 `128x64`, and M256/M512 `256x64` controls pass independent reference/mutation checks and final confirmation; M32/M64/M128 use the retained packed-VOPD decoder.
- Large-margin decoder, LDS, ownership, and geometry search. Ordinary screening established `LdsPadB=8` as the dominant reusable mechanism, retained `256x64` for attention output/Q-A/KV, found a Q-B-only DepthU64 branch, and rejected broad WGM2, pad16/24, XOR4/8/16 one-buffer layouts, `64x128`, `256x128`, two decoded-B buffers, broad scalar loads, PLR2, broad next-packed prefetch, and non-qualifying LM pad/XOR neighborhoods. Packed VOPD was then tested as the remaining decode arithmetic mechanism and retained only for LM M32/M64/M128.
- Per-key selection with every retained key faster than HIP. The ordinary and LM-head catalogs are selected and confirmed; every one of 23 exact keys beats HIP.
- Lower-bound and bottleneck explanation. Representative complete/WMMA-A-LDS/decode-LDS floors were measured for Q-A, Q-B, output-B, shared gate/up, shared-down, and LM M512.
- Independent reproducibility and final 25-repeat confirmation. Two independent complete Q8 catalog roots produced byte-identical assembly and matching resources for all 23 keys; ordinary and LM confirmation phases use serial 25-repeat controls, including the final five-key LM catalog.
- Recursive optimization-exhaustion review with no actionable mechanism remaining. The final layout neighborhood and the actionable packed-VOPD decode mechanism were tested under the exact-key gates; remaining valid mechanisms are either outside contract or fail timing/resource thresholds.
- Public runtime dispatch, deferred.

### Final result

The current `mmq_bwd_q8_0_catalog.json` is loaded by `tools/mmq_deployment_spec.py:kernels()` as `OrdinaryBackward`; `public_deployment_cases()` exposes these exact rows within the 50-case ordinary-backward public set wired by commit `1924d4b`. The `ggsol_...` value is the current public catalog hash. Logical throughput is `2*M*N*K/(median_ms*1e9)`, and speedup is `HIP time / GGTensile time`, so values above `1.0x` favor the deployed GGTensile entry.

| Family | `(M,N,K)` | Public catalog hash | HIP ms | GGTensile ms | HIP TFLOPS | GGTensile TFLOPS | HIP time / GGTensile time |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Attention Q-A | `(2048,4096,1024)` | `ggsol_79ab814fefd7e86a` | `0.9077` | `0.4693` | `18.926` | `36.605` | `1.9341x` |
| Attention Q-A | `(8192,4096,1024)` | `ggsol_1605fb2e42cdae3b` | `2.7321` | `2.1500` | `25.153` | `31.963` | `1.2707x` |
| Attention Q-A | `(32768,4096,1024)` | `ggsol_89a0cc4919ba7c07` | `13.5272` | `8.4746` | `20.320` | `32.435` | `1.5962x` |
| Attention Q-B | `(2048,1024,32768)` | `ggsol_e3718bcd339bd337` | `7.0054` | `6.1495` | `19.619` | `22.350` | `1.1392x` |
| Attention Q-B | `(8192,1024,32768)` | `ggsol_0b6ccdec8d6ac04d` | `31.6286` | `22.9929` | `17.382` | `23.910` | `1.3756x` |
| Attention Q-B | `(32768,1024,32768)` | `ggsol_b13812666df0bb0b` | `113.0838` | `81.0581` | `19.446` | `27.129` | `1.3951x` |
| Attention K/V | `(2048,4096,512)` | `ggsol_8f3a1d10bca36750` | `0.4501` | `0.2415` | `19.083` | `35.566` | `1.8638x` |
| Attention K/V | `(8192,4096,512)` | `ggsol_39af6b70988355da` | `1.3919` | `1.0525` | `24.685` | `32.646` | `1.3225x` |
| Attention K/V | `(32768,4096,512)` | `ggsol_652fc754d956a001` | `5.2502` | `4.1654` | `26.178` | `32.996` | `1.2604x` |
| Attention output-B | `(2048,8192,4096)` | `ggsol_fefcdf7f91800920` | `6.1033` | `4.2766` | `22.519` | `32.138` | `1.4271x` |
| Attention output-B | `(8192,8192,4096)` | `ggsol_6e86102f7f92d667` | `26.6558` | `16.5955` | `20.624` | `33.127` | `1.6062x` |
| Attention output-B | `(32768,8192,4096)` | `ggsol_fd380e0978854da7` | `103.1480` | `66.3253` | `21.319` | `33.155` | `1.5552x` |
| Shared gate/up | `(2048,4096,2048)` | `ggsol_32f1aae7d3d5eb0e` | `1.6443` | `0.9512` | `20.896` | `36.122` | `1.7286x` |
| Shared gate/up | `(8192,4096,2048)` | `ggsol_1d2a8412b22e9423` | `5.7490` | `4.7499` | `23.907` | `28.935` | `1.2103x` |
| Shared gate/up | `(32768,4096,2048)` | `ggsol_fd08c244467fe1fb` | `27.1389` | `18.5362` | `20.257` | `29.659` | `1.4641x` |
| Shared down | `(2048,2048,4096)` | `ggsol_919fc04046025eec` | `1.6639` | `1.0669` | `20.651` | `32.204` | `1.5595x` |
| Shared down | `(8192,2048,4096)` | `ggsol_8fd5945b4e44a8f7` | `7.3495` | `4.7587` | `18.700` | `28.882` | `1.5444x` |
| Shared down | `(32768,2048,4096)` | `ggsol_b1d9d1c8312555c9` | `27.4390` | `19.1847` | `20.036` | `28.656` | `1.4303x` |
| LM head | `(32,4096,129280)` | `ggsol_5fc69a7e5053332d` | `8.2403` | `4.2374` | `4.113` | `7.998` | `1.9447x` |
| LM head | `(64,4096,129280)` | `ggsol_f47e1d4c0b3ea1d1` | `7.1113` | `4.1412` | `9.531` | `16.367` | `1.7172x` |
| LM head | `(128,4096,129280)` | `ggsol_18248d7c66f61eef` | `7.6594` | `5.7257` | `17.698` | `23.676` | `1.3377x` |
| LM head | `(256,4096,129280)` | `ggsol_dd5d0a2953b79821` | `10.2869` | `9.5085` | `26.356` | `28.514` | `1.0819x` |
| LM head | `(512,4096,129280)` | `ggsol_ccc7671dafbce6e8` | `22.3029` | `17.9027` | `24.313` | `30.288` | `1.2458x` |

Ordinary rows average the elapsed medians from `ggtensile-q8-ordinary-final-a` and `ggtensile-q8-ordinary-final-c`. LM M32/M64/M128 use the current packed-VOPD confirmation; LM M256/M512 average the matching packed entries from `ggtensile-q8-all-vopd-final-a` and `ggtensile-q8-lm-final-a`. The call-weighted ordinary-catalog speedup is `1.4522x` (candidate/HIP latency ratio `0.6886x`), the five-key LM-head speedup is `1.3393x` (ratio `0.7467x`), and the all-23-key speedup is `1.4518x`. The research-only `ggtensile-final-vs-hip-q8-m256-{a,b}.json` identity is not used for these public rows.

### Ordinary screening record

The first fresh 18-key `128x128x32` control measured a call-weighted candidate/HIP ratio of `1.0329126789987204`; it was not competitive at several M2048/M8192 keys. Unswizzled eight-BF16 row padding then reduced representative long-row candidate latency by roughly 15-25%. PGR2/SIA5 plus pad8 produced candidate/HIP ratios from about `0.66` to `0.95` across the ordinary matrix and made every screened exact key faster than HIP.

Geometry is quant- and shape-specific. Padded `256x64` reduced attention-output M2048/M8192/M32768 from `5.041/19.522/75.521 ms` to `4.234/16.776/65.925 ms`; it also improved Q-A and KV, but regressed Q-B. Padded `128x64` is preferred over `256x64` for long-row shared gate/up and shared-down. Broad WGM2 regressed compact bodies.

Q8 scalar `global_load_b32` extraction was neutral for most families but improved Q-B M8192 by about 5% versus packed `global_load_b128`. A corrected Q8 DepthU64/XOR8 control measured `22.285 ms` and `0.702x` HIP at Q-B M8192, versus `24.113 ms` for the padded DepthU32 packed control, but regressed output-B. The initial DepthU64 failure exposed overlapping Q8 block-base, payload-pointer, and persistent-A state; a reusable temporary payload pointer fixed the ownership without increasing resources. The corrected body uses 220 VGPR, 16 SGPR, and 16 KiB LDS with no spills or private storage.

The two decoded-B buffer path was extended to the Q8 XOR8 store layout and passed exact HIP, independent-reference, grad-output mutation, and packed-weight mutation checks. It was timing-neutral on the discriminator keys and is rejected. Pad16/24, XOR4/8/16 one-buffer layouts, PLR2, SIA3, `64x128`, and `256x128` also lost. Next-tile packed prefetch remains only a possible exact-key small mechanism because its broad effects were neutral or unfavorable.

The selected ordinary catalog uses compact padded `256x64` for Q-A, KV, attention output, and the M2048 shared projections; padded `128x64` for M8192/M32768 shared gate/up and shared-down; DepthU64/XOR8 for Q-B M8192; padded scalar next-prefetch for Q-B M2048; and padded packed `128x128` for Q-B M32768. The historical per-key throughput and multiplicative HIP speedups are consolidated in the selected-catalog table; the call-weighted candidate/HIP latency ratio is `0.6879975522199318`. The complete final catalog was independently rebuilt with byte-identical assembly and matching resource tuples for all 23 selected keys.

### LM-head screening record

The first representable LM-head controls all beat HIP: padded `64x128` measured `5.873 ms` at M64 (`0.836x` HIP), padded `128x64` measured `6.032 ms` at M128 (`0.789x`), and padded `256x64` measured `9.519/18.784 ms` at M256/M512 (`0.928/0.858x`). Wider `128x128` and `256x128` bodies were slower.

M32 required a true two-wave exact geometry rather than a partial 64-row tile. Q8-specific 64-thread `32x64` and `32x128` bodies were added with generalized decoder-row spacing and strict rejection for other quant types. Both pass independent reference and producer mutation checks. The retained packed body with DepthU64/pad8 measured `4.237 ms` versus `8.240 ms` HIP (`0.5142x`) in final confirmation. Packed VOPD decode reduced M32/M64/M128 screening latency by `3.55%`/`2.90%`/`3.57%`; the retained bodies use 106/92/140 VGPR respectively, 16 SGPR, 9 KiB LDS, and no private storage or spills. `32x128`, alternate pad16/24, and XOR8 remain rejected.

### Lower bounds and residual bottleneck

| Representative key | Complete | WMMA/A/LDS floor | Decode/LDS floor | Floor sum / complete |
| --- | ---: | ---: | ---: | ---: |
| Q-A M32768 | 8.208 ms | 6.260 ms | 1.746 ms | 0.975x |
| Q-B M8192 | 22.329 ms | 12.504 ms | 8.717 ms | 0.950x |
| Output-B M32768 | 65.968 ms | 49.257 ms | 15.346 ms | 0.979x |
| Shared gate/up M32768 | 18.261 ms | 12.330 ms | 5.684 ms | 0.986x |
| Shared-down M32768 | 18.728 ms | 12.373 ms | 5.743 ms | 0.967x |
| LM M512 | 17.401 ms | 13.185 ms | 5.967 ms | 1.101x |

The ordinary floors sum to within 2-5% of complete timing, so the residual is not an unhidden large scheduling gap: decode VALU/VMEM and WMMA/LDS synchronization are the two dominant components, with their overlap already close to the measured complete path. Q-B has the largest decode fraction and is the only ordinary key where DepthU64 remains beneficial. LM M512 has a floor sum above complete because its two isolated floors double-count work that overlaps in the complete next-prefetch body; its residual bottleneck is packed Q8 payload/scale traffic plus WMMA occupancy under the two-M-tile launch, not a missing correctness mechanism. Packed VOPD removes a measurable portion of the decoder multiply issue cost only on the small LM ownership bodies; it does not expose a second broad bottleneck that clears the resource-bearing threshold.

Adding the HIP-analogous `64x64` ownership reduced M64 from `5.873` to `4.487 ms` (`0.6308024774361413x` HIP). Q8-specific unswizzled pad8 DepthU64 was then generalized to compact geometries and passed all five LM correctness screens. Packed VOPD decode was tested against the retained packed decoder: M32/M64/M128 screened at `4.248/4.151/5.654 ms`, versus `4.405/4.275/5.863 ms`, and final confirmation measured `4.237/4.141/5.726 ms` with HIP ratios `0.5142/0.5823/0.7475`. M256 improved only `1.44%` with extra VGPRs and was rejected by the resource-bearing threshold; M512 regressed `2.62%`. Alternate pad16/24 and XOR8 lost on every LM geometry. The final LM choices therefore use packed VOPD at M32/M64/M128, DepthU32 packed at M256, and DepthU32 packed with next-tile prefetch at M512. Scalar extraction, SIA4 without prefetch, PGR1, store-priority removal, and WGM2 lose.

The promoted final LM confirmation is fully serial and uses the selected packed-VOPD/packed mix. Its historical per-key throughput and multiplicative HIP speedups are consolidated in the selected-catalog table; the five-key aggregate candidate/HIP latency ratio is `0.7512901204186552`, and all five keys remain faster than HIP.

## M256 packed-VOPD stability-policy reopening

The historical M256 packed-VOPD result was closed solely by the former fixed two-percent resource-bearing threshold. It was therefore rebuilt against the current canonical M256 parent as the already serialized `decode.extraction=packed_vopd` identity. Two independent source/object/HSACO builds were byte-deterministic. Parent and candidate both inspect in the same 231-VGPR, 16-SGPR, 5,120-byte LDS allocation class with the 40-byte ABI, wave32, code-object v5, 32 WMMAs, two barriers, zero private bytes, and zero spills. The candidate uses 20 VOPD instructions rather than 12 and reduces VALU issue count from 548 to 541; VMEM, LDS, waits, and clauses are unchanged.

The current candidate and parent both matched installed HIP bitwise over all 1,048,576 BF16 outputs. The candidate also passed the independent reference with the same `0.0004991045` normalized RMSE as HIP, finite-output checks, full gradient-output mutation, and packed-weight mutation. Each mutation remained bitwise equal to HIP and changed the candidate output. The exact research identity is `ggsol_45234dc0448fafb4`; the instruction-identical current parent is `ggsol_dd5d0a2953b79821`.

Two seven-warmup, alternating 25-repeat brackets used independent deterministic gradients and reversed launch order. Candidate/parent medians were `9.23366/9.38050 ms` and `9.10986/9.26879 ms`, gains of `1.59%` and `1.72%`. The robust log-time 95% candidate-over-parent intervals were `-2.79%..-0.33%` and `-2.09%..-1.33%`, so both exclude parity in favor of packed VOPD. The adjacent nine-repeat HIP screen measured candidate/parent `9.28981/9.44039 ms`, while both remained faster than HIP.

Packed VOPD is retained for the exact M256 research identity under focused deterministic build/resource coverage. This does not transfer to M512, whose historical packed-VOPD artifact regressed, and it does not alter the selected deployment catalog, generated bundle, public dispatch, package, registration, or HIP fallback. Every future composition or exact-key selection requires a new identity and full requalification.

### Final retained throughput versus HIP

The current typed M256 packed-VOPD identity was benchmarked against the installed HIP kernel with seven warmups and 25 alternating repeats in each of two independent brackets. Logical throughput uses `2 * M * N * K / (median_ms * 1e9)`. The speedup is `HIP median / packed-VOPD median`, so values above `1.0x` favor the retained kernel. The summary averages the two bracket medians.

| Bracket | Candidate ms | HIP ms | Candidate TFLOPS | HIP TFLOPS | Speedup vs HIP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| A | 9.4599 | 10.3371 | 28.660 | 26.228 | 1.0927x |
| B | 9.3879 | 10.2873 | 28.880 | 26.355 | 1.0958x |
| Mean of medians | 9.4239 | 10.3122 | 28.769 | 26.291 | 1.0943x |

All candidate and HIP outputs matched exactly before and after gradient and packed-weight mutation. The fresh reports are `ggtensile-final-vs-hip-q8-m256-{a,b}.json`. The earlier typed-parent ratios remain acceptance evidence for packed VOPD; the final ratios above are the retained-kernel-versus-HIP results. No deployment-catalog or public-dispatch claim follows from this research-only identity.
