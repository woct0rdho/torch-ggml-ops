# GGTensile Dense MMQ Backward Q6_K Plan

## Purpose

Build and exhaust a strict gfx1151 GGTensile assembly campaign for the three production Q6_K dense-MMQ-backward shapes used by the Qwen language-model head. The result must decode the authoritative packed GGUF tensor inside the fused kernel, beat the existing HIP kernel on every selected exact key, and continue until no new valid in-contract optimization mechanism remains.

Q6_K is a separate quant backend from Q3_K, Q4_K, Q5_K, and Q8_0. It may reuse quant-neutral A addressing, decoded-B LDS staging, WMMA issue, synchronization, stores, diagnostics, inspection, and immutable campaign phases. It must own its 210-byte block addressing, split low/high payload extraction, signed 8-bit scale handling, FP16 block multiplier, register lifetimes, load strategy, and tuning identity.

The final review rule is recursive: any newly actionable mechanism invalidates the current stopping condition. The review must be repeated after that mechanism is implemented and measured. The campaign stops only when all valid large-margin and then smaller-margin mechanisms are retained or rejected, all exact keys have final evidence, and the remaining bottleneck is explained quantitatively.

## Contract

Target only:
- gfx1151, wave32, WMMA V1, and the existing 40-byte dense-backward kernarg ABI.
- BF16 grad output and grad input, FP32 accumulation, and packed GGUF Q6_K weights.
- Fused in-kernel decode with no prepared weights, dense shadows, external decode workspace, hidden cache, split-K, persistent workgroups, or grouped MMQ.
- One exact `ProblemType` and one exact `ProblemSize` per artifact.
- Strict rejection for unsupported solutions and fallback to HIP outside exact selected keys.
- Serial warmed rotating-control timing; correctness/build/reproducibility work may run independently.
- Bit-exact HIP equality, independent-reference behavior, grad-output mutation, and packed-weight mutation for retained kernels.
- Zero private storage, spills, scratch instructions, calls, and dynamic stack.

Dense backward coordinates are:

```text
M = rows
N = in_features
K = out_features

grad_input[M,N] = grad_output[M,K] @ dequant(weight[K,N])
```

Q6_K has 256 logical values per 210-byte block. The production weight has logical shape `[248320,2048]` and packed shape `[248320,1680]` because `(2048/256)*210 = 1680` bytes per row.

## Exact Production Scope

The campaign has one family and three exact keys:

| Role | `(M,N,K)` | Packed weight shape | Calls |
| --- | ---: | ---: | ---: |
| Smallest production chunk | `(64,2048,248320)` | `[248320,1680]` | 1 |
| Lower-memory fallback | `(128,2048,248320)` | `[248320,1680]` | 1 |
| Primary complete-loss chunk | `(256,2048,248320)` | `[248320,1680]` | 1 |

M256 is the first optimization priority because it is the public complete-loss schedule. M128 and M64 are required exact fallback keys. M32, M512, and ordinary M2048/M8192/M32768 shapes are outside the production inventory unless the scheduler or checkpoint changes.

The existing HIP controls provide strong target evidence:

| M | Retained HIP ownership | Historical packed/BF16 |
| ---: | --- | ---: |
| 64 | M64/N32/K64, 16-BF16 XOR | `5.357/9.863 ms` |
| 128 | M128/N64/K32, 8-BF16 XOR | `9.084/14.480 ms` |
| 256 | Two M128/N64/K32 workgroups, packed extraction, 8-BF16 XOR | `11.726/19.010 ms` |

HIP proves that exact ownership, split-six-bit decode, and packed extraction can outperform the BF16 control. GGTensile must at least match HIP and then continue optimizing.

## Multi-Quant GGTensile Design

### Quant-neutral body

The shared writer owns behavior that is independent of packed format:
- exact launch flattening and workgroup mapping.
- A global coordinates, loads, prefetch, and row-state lifetime.
- decoded-B LDS buffer allocation and local-read coordinates.
- barriers, wait dependencies, optional pipeline schedules, and store priority.
- WMMA accumulator allocation and issue order.
- FP32-to-BF16 output conversion and exact stores.
- lower-bound diagnostics, source hashing, build, inspection, resource gates, runtime ABI, and campaign phases.

Shared helpers are valid only when they preserve identical quant-neutral semantics. Quant selection must not silently alter unsupported controls.

### Quant backend contract

Each quant type owns:
- values and bytes per packed block.
- packed-row and macro-tile block addressing.
- payload and metadata register allocation.
- global load widths, lane ownership, and optional lane sharing.
- decode preparation and per-chunk decode emission.
- LDS-facing decoded-value order.
- quant-specific temporary lifetime and resource accounting.
- independent decoder fixtures and mutation coverage.
- tuning fields that are strict, explicit, and inert for other quant types.

Q6_K adds only controls backed by real alternate emitters. `Q6KExtraction` now has `packed`, `packed_vopd`, and `scalar` variants. The retained `packed_vopd` path preserves the packed six-bit reconstruction and exact FP operand order while pairing adjacent subtracts and multiplies with legal gfx11 VOPD instructions. Further candidates such as scale broadcast or metadata vector loading are added only with emitted-ISA and correctness tests.

### Q6_K backend

A 256-value Q6_K block contains:

```text
ql[128]      low four bits
qh[64]       high two bits
scales[16]   signed int8 scale per 16 values
d             FP16 block multiplier
```

For value `i`:

```text
q = low4(i) | (high2(i) << 4)
value = fp16(d) * float(int8(scale[i/16])) * float(q - 32)
```

The backend must preserve this arithmetic and BF16 rounding order. Packed extraction should coalesce low/high planes and amortize `d*scale` across each 16-value group. Scalar extraction is a measured alternative, not a fallback repair.

## Phases

### Phase 1: Strict identity, inventory, and control

- Add Q6_K `ProblemType`, exact-size validation, 210-byte physical-row accounting, runtime checks, campaign specification, and a versionless three-key inventory/catalog.
- Add `Q6KExtraction` to strict solution identity and require it to be inert for other quant types.
- Add Q6-aware inspection/resource accounting without changing existing quant artifacts.
- Generate and inspect an initial shared-WMMA control.
- Confirm existing Q3/Q4/Q5/Q8 catalogs remain valid.

### Phase 2: Correct decoder backend

- Implement Q6_K block addresses, low/high payload loads, signed scales, FP16 `d`, split-six-bit reconstruction, fused scale application, BF16 rounding, and LDS stores.
- Add reduced-K and one-hot tests covering low nibbles, high pairs, signed scale extremes, `d`, all 16 scale groups, both 128-value chunks, block boundaries, and packed-row boundaries.
- Validate all three production keys against HIP, an independent dequantized reference, grad-output mutation, and packed-weight mutation before timing.

### Phase 3: Large-margin search

Search M256 first, then transfer only measured mechanisms to M128/M64:
- Match HIP M128/N64/K32 ownership for M256 and M128; use two exact M tiles for M256.
- Add exact M64/N32/K64 ownership if the initial representable geometry leaves a material margin.
- Compare packed versus scalar extraction and low/high/scale load grouping.
- Compare HIP-analogous XOR8/XOR16 layouts with unswizzled padding and unpadded controls.
- Measure decoder-row ownership, payload lane sharing, and metadata broadcast.
- Test one versus two decoded-B buffers only if lower bounds expose actionable overlap.
- Test DepthU, PGR/PLR, SIA, bounded next-tile packed prefetch, and address-state lifetime.
- Test WGM only where more than one exact M tile exists and launch ordering can matter.

Candidates pass correctness and resource inspection before timing. Nine-repeat screens narrow candidates; retained resource-bearing mechanisms require a stable gain above 2% in 25-repeat confirmation. Timing is authoritative.

### Selected geometry and current evidence

The retained exact catalog uses different Q6-owned tuning for each production M while sharing the same quant-neutral A/LDS/WMMA/store implementation:

| M | Retained body | Resources | 25-repeat candidate/HIP |
| ---: | --- | ---: | ---: |
| 64 | `64x32x64`, unswizzled pad8, packed VOPD | 76 VGPR, 4608 B LDS | `5.1708 / 5.4057 ms = 0.9565x` |
| 128 | `128x32x64`, unswizzled pad8, packed VOPD | 108 VGPR, 4608 B LDS | `6.6851 / 9.3648 ms = 0.7139x` |
| 256 | `256x64x32`, next-packed-tile prefetch, pad8, packed VOPD | 240 VGPR, 5120 B LDS | `10.1445 / 12.1927 ms = 0.8320x` |

The equal-call weighted candidate/HIP ratio is `0.8159412939177061`. M64 packed VOPD beats the packed pad8 control by 3.2% in a 25-repeat control comparison. M128 compact `128x32x64` beats the prior `128x64x32` next-prefetch/VOPD body by 2.77% while reducing VGPRs from 140 to 108 and LDS from 5120 B to 4608 B. M256 wide ownership beats the narrow next-prefetch/VOPD body by 18.0% in the same protocol.

Closed large-margin neighborhoods include:
- M64 `64x32x64` XOR8/XOR16, scalar extraction, packed VOPD, and pad8/pad16/pad24. Pad8 and pad24 tie; pad8 wins on smaller LDS.
- M128/M256 scalar extraction, SIA/PGR/PLR/WGM variants, XOR8/XOR16, next-packed-tile prefetch, and packed VOPD.
- M128 `128x64x32` next-prefetch and `128x32x64` compact ownership. Compact pad8 is retained after direct 25-repeat control timing; it is both faster and smaller.
- M256 `64x64`, `128x64`, `256x32`, and `256x64` ownership. Wide `256x64` is retained only after next-prefetch, pad8, and VOPD establish a stable large gain. The `256x32x64` occupancy candidate was correct but 32-43% slower than HIP.
- VMEM clauses and `buffer_gl0_inv`; both were neutral or slower in 25-repeat or same-process controls.
- DepthU64 two-decoder-row variants, which failed exact correctness for wider N ownership.
- A proposed two-decoded-B `128x64` pipeline, rejected because exactly every fourth output column was invalid under reduced and production K tests. The existing strict gate remains `128x128x32` only.
- DepthU64 next-packed-tile prefetch, rejected after an illegal-address correctness failure; the strict validator continues to forbid it.

### Phase 4: Lower bounds and final selection

For each selected geometry, measure complete, WMMA/A/LDS-floor, and Q6-decode/LDS-floor artifacts under the same ABI and launch contract. Explain:
- packed low/high/scale/d VMEM traffic.
- decode VALU issue pressure and VOPD opportunities.
- WMMA occupancy and accumulator allocation.
- LDS conflicts, synchronization, and decoded-row layout.
- active workgroups for M64/M128/M256.
- complete-loss relevance of the M256 key.

Every selected key must beat HIP, pass independent rebuild reproducibility, and have a 25-repeat serial confirmation.

## Recursive Optimization-Exhaustion Review

Before declaring completion, reread this plan, Q6 HIP source and normalized ISA, all Q6 GGTensile artifacts and timing reports, Q3/Q4/Q5/Q8 experiment records, lower bounds, resource reports, rejected candidates, `rdna35-isa-markdown`, AMD LLVM definitions/tests, and relevant CK/TensileLite notes.

Classify every remaining idea as:
- retained and measured.
- rejected by correctness, resources, timing, or reproducibility.
- contract-incompatible or deferred with an explicit prerequisite.
- actionable and requiring another implementation/measurement cycle.

If the review identifies an actionable idea, implement and measure it, update this plan, and repeat the review from the new premise. The final review cannot pass in the same iteration that first discovers an actionable mechanism.

The campaign is exhausted only when no valid in-contract optimization idea remains, all three keys beat HIP with final evidence, and the residual bottleneck is quantitatively explained. Public runtime dispatch and artifact packaging remain deferred until the multi-quant integration project explicitly takes ownership.

## Completion Record

Update this section after every coherent implementation milestone before committing code. Documentation-only evidence updates do not require their own commit.
- Strict Q6_K identity, tuning controls, inventory, packed-row accounting, and campaign validation. `Q6KExtraction` has packed/scalar emitters and is rejected as non-inert for other quant types; the inventory contains exactly M64/M128/M256 with physical shape `[248320,1680]`.
- Correct Q6_K assembly backend with independent decoder fixtures and resource inspection. The backend owns 210-byte blocks, low/high payload planes, signed scales, FP16 `d`, exact BF16 rounding, and quant-specific resource accounting while reusing the shared WMMA body.
- Fresh three-key HIP/GGTensile controls with reference and producer-mutation coverage. All keys are bit-exact to HIP under baseline, grad-output mutation, and packed-weight mutation; independent-reference error matches HIP. Initial candidate/HIP screening ratios are M64 `1.5409`, M128 `0.9926`, and M256 `1.1216`.
- Large-margin M256 ownership, decoder, and LDS search. Retain wide `256x64x32` ownership with next-packed-tile prefetch, pad8, and packed VOPD; it confirms at `0.8220x` HIP. Narrow ownership, clauses, GL0 invalidation, lane sharing, DepthU64, and two-buffer variants were rejected by timing or correctness.
- M128 and M64 exact geometry closure. M128 retains `128x32x64` pad8/VOPD at `0.7139x` HIP. M64 retains `64x32x64` pad8/VOPD at `0.9565x` HIP; packed VOPD clears the 2% resource-bearing threshold against packed pad8.
- Per-key selected catalog with every exact key faster than HIP. The equal-call weighted 25-repeat ratio is `0.8159412939177061`.
- Complete/WMMA/decode lower bounds and residual bottleneck explanation. M64 complete/WMMA/decode are `5.230/2.659/3.283 ms`; M128 `6.631/3.776/3.291 ms`; M256 `9.721/7.166/5.573 ms`. The floor sums are `1.136x`, `1.066x`, and `1.311x` complete respectively, showing overlap in the pipelined M256 body and balanced compact M128 halves. M256 remains WMMA/accumulator limited; M64 remains decode-heavy; M128 has no isolated dominant half.
- Independent assembly reproducibility. Three independent builds are byte-identical with identical resources: M64 `76 VGPR/4608 B`, M128 `108 VGPR/4608 B`, M256 `240 VGPR/5120 B`, with no private storage or spills. Final serial 25-repeat confirmation is complete for all three keys with independent reference and producer-mutation coverage.
- Recursive optimization-exhaustion review with no actionable mechanism remaining. The final review rechecked geometry/ownership, packed versus scalar and VOPD decode, padding/XOR layouts, clauses, GL0 invalidation, lane sharing, DepthU, prefetch, SIA/PGR/PLR/WGM, two-buffer correctness, lower-bound overlap, and occupancy-oriented M256 `256x32` ownership. Remaining alternatives were rejected by contract, correctness, resources, or serial timing.
- Public dispatch and artifact packaging, deferred.
