# GGTensile MMQ Backward Q6_K Plan

## Purpose

Build and exhaust a strict gfx1151 GGTensile assembly campaign for the three production Q6_K MMQ backward shapes used by the Qwen language-model head. The result must decode the authoritative packed GGUF tensor inside the fused kernel, beat the existing HIP kernel on every selected exact key, and continue until no new valid in-contract optimization mechanism remains.

Q6_K is a separate quant backend from Q3_K, Q4_K, Q5_K, and Q8_0. It may reuse quant-neutral A addressing, decoded-B LDS staging, WMMA issue, synchronization, stores, diagnostics, inspection, and immutable campaign phases. It must own its 210-byte block addressing, split low/high payload extraction, signed 8-bit scale handling, FP16 block multiplier, register lifetimes, load strategy, and tuning identity.

The final review rule is recursive: any newly actionable mechanism invalidates the current stopping condition. The review must be repeated after that mechanism is implemented and measured. The campaign stops only when all valid large-margin and then smaller-margin mechanisms are retained or rejected, all exact keys have final evidence, and the remaining bottleneck is explained quantitatively.

## Contract

Target only:
- gfx1151, wave32, WMMA V1, and the existing 40-byte MMQ backward kernarg ABI.
- BF16 grad output and grad input, FP32 accumulation, and packed GGUF Q6_K weights.
- Fused in-kernel decode with no prepared weights, dense shadows, external decode workspace, hidden cache, split-K, persistent workgroups, or grouped MMQ.
- One exact `ProblemType` and one exact `ProblemSize` per artifact.
- Strict rejection for unsupported solutions and fallback to HIP outside exact selected keys.
- Serial warmed rotating-control timing; correctness/build/reproducibility work may run independently.
- Bit-exact HIP equality, independent-reference behavior, grad-output mutation, and packed-weight mutation for retained kernels.
- Zero private storage, spills, scratch instructions, calls, and dynamic stack.

MMQ backward coordinates are:

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

### Final result and selected geometry

The current `mmq_bwd_q6_k_catalog.json` is loaded by `tools/mmq_deployment_spec.py:kernels()` as `OrdinaryBackward`. The public bundle wiring in commit `1924d4b` exposes these three exact cases through `public_deployment_cases()`; HIP remains fallback outside these keys. The `ggsol_...` value is the current public catalog hash. The final result reports logical arithmetic throughput; speedup is `HIP time / GGTensile time`.

| `(M,N,K)` | Public catalog hash | Resources | HIP ms | GGTensile ms | HIP TFLOPS | GGTensile TFLOPS | HIP time / GGTensile time |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(64,2048,248320)` | `ggsol_47dc5792dc6a3094` | 76 VGPR, 4608 B LDS | `5.4057` | `5.1708` | `12.042` | `12.589` | `1.0454x` |
| `(128,2048,248320)` | `ggsol_322cef48a5fa0b12` | 108 VGPR, 4608 B LDS | `9.3648` | `6.6851` | `13.902` | `19.475` | `1.4008x` |
| `(256,2048,248320)` | `ggsol_83942eadf7602aa1` | 240 VGPR, 5120 B LDS | `12.1927` | `10.1445` | `21.356` | `25.667` | `1.2019x` |

The all-key `selected-final-h` confirmation supplies the elapsed medians above. The single-key `selected-final-e` and `selected-final-b` roots are assembly-control checks, not independent all-key confirmations, so they are not blended into this table. The equal-call weighted speedup is `1.2256x`, corresponding to a candidate/HIP latency ratio of `0.8159x`. M64 packed VOPD beats the packed pad8 control by 3.2% in a 25-repeat control comparison. M128 compact `128x32x64` beats the `128x64x32` next-prefetch/VOPD body by 2.77% while reducing VGPRs from 140 to 108 and LDS from 5120 B to 4608 B. M256 wide ownership beats the narrow next-prefetch/VOPD body by 18.0% in the same protocol. Forward `q6-typed-wavefront` artifacts are not part of this backward catalog result.

Closed large-margin neighborhoods include:
- M64 `64x32x64` XOR8/XOR16, scalar extraction, packed VOPD, and pad8/pad16/pad24. Pad8 and pad24 tie; pad8 wins on smaller LDS.
- M128/M256 scalar extraction, SIA/PGR/PLR/WGM variants, XOR8/XOR16, next-packed-tile prefetch, and packed VOPD.
- M128 `128x64x32` next-prefetch and `128x32x64` compact ownership. Compact pad8 is retained after direct 25-repeat control timing; it is both faster and smaller.
- M256 `64x64`, `128x64`, `256x32`, and `256x64` ownership. Wide `256x64` is retained only after next-prefetch, pad8, and VOPD establish a stable large gain. The `256x32x64` occupancy candidate was correct but 32-43% slower than HIP.
- VMEM clauses and `buffer_gl0_inv`; both were neutral or slower in 25-repeat or same-process controls.
- DepthU64 two-decoder-row variants historically failed exact correctness for wider N ownership. The swizzled decoded-LDS placement now derives its transform from the logical K offset, and the formerly failing M64 and M256 N64-ownership production repros pass HIP, independent reference, and both producer mutations. They require renewed serial timing before selection.
- The proposed two-decoded-B `128x64` pipeline historically produced invalid columns. The geometry-derived final handoff and Q6 temporary/address-state fixes now pass the M128 and M256 production repros. The validator accepts the corrected Q4_K/Q6_K N64 pipeline, but the selected catalog remains unchanged pending timing.
- DepthU64 next-packed-tile prefetch historically faulted after current A pointers were overwritten too early. Packed reads are now delayed until the second current-A half is consumed, and the exact M64 production repro passes all correctness and mutation checks. It is valid again but unselected pending timing.
- Q6 lane sharing historically replicated both payload planes even though only the 2-bit high plane is shared by lane pairs. The specialized emitter now loads unique low planes per lane, owner-loads only high planes, and passes the M256 production repro. It also requires renewed timing before any catalog change.

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

### Renewed executable work

The Q6 repairs reopen four candidate mechanisms for timing: high-plane-only lane sharing, compact DepthU64 ownership at M64, wide DepthU64 ownership at M256, and DepthU64 next-packed prefetch at M64. The corrected M128 and M256 decoded-B pipeline artifacts are also retimed as a separate ownership bracket. Each screen is serial against the selected exact-key control and HIP; only stable resource-bearing gains above 2% advance to 25-repeat confirmation and possible catalog selection.

Metadata owner-load plus wave-local DPP broadcast is retained as a Q4-targeted first emitter experiment and cannot be assumed to help Q6's signed scale path. Combined padded-stride plus logical-K XOR placement requires supported LDS-counter evidence and a new occupancy-safe emitter. A genuinely new exact-N software pipeline is a separate structural campaign, not another DepthU or WGM repetition. Exact-shape literal/fixed-trip work is low priority, and model-owned integer-plus-scale preparation, prepared weights, and external decode storage remain outside this direct-packed contract.

Fresh nine-repeat execution closes all repaired Q6 candidates against the current selected exact-key assemblies. Every candidate remained bit-exact to HIP, matched HIP's independent-reference error, and passed both producer mutations.

| Candidate | M | Candidate ms | Selected control ms | Candidate/control | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| High-plane-only lane sharing | 256 | `12.8034` | `9.8869` | `1.29498x` | Reject |
| Wider DepthU64 ownership | 64 | `7.0339` | `5.1141` | `1.37538x` | Reject |
| Wider DepthU64 ownership | 256 | `12.1769` | `9.9054` | `1.22932x` | Reject |
| DepthU64 next-packed prefetch | 64 | `5.3565` | `5.1167` | `1.04688x` | Reject |
| Two-buffer N64 pipeline | 128 | `9.6587` | `6.6179` | `1.45949x` | Reject |
| Two-buffer N64 pipeline | 256 | `14.5063` | `9.8659` | `1.47034x` | Reject |

No candidate clears the greater-than-2% resource-bearing gate, none advances to confirmation, and the selected catalog remains unchanged.

### Reopened exact compact pipeline

The rejected N64 pipelines do not cover a two-buffer pipeline that preserves the selected M64 `64x32x64` ownership. The selected exact `(64,2048,248320)` kernel uses four waves, 76 VGPRs, one 4,608-byte unswizzled pad8 decoded-B buffer, and 3,880 exact DepthU64 reduction trips. Its complete/WMMA/decode floors of `5.230/2.659/3.283 ms` leave a specific overlap premise even though packed-next-only prefetch was slower.

Add one strict Q6-only emitter with the same macro tile, workgroup, DepthU64, pad8 row layout, packed VOPD extraction, A ownership, and store path, but two independent 4,608-byte decoded-B buffers for 9,216 bytes total LDS. All four waves remain symmetric compute/decoder waves. Prime B0, decode the next packed Q6 tile into B1 in bounded chunks while WMMA consumes B0, perform one full-barrier handoff per steady trip, swap the buffer base, and peel the final trip. Do not add dedicated decoder waves, widen MacroTile1, or reuse the rejected N64 handoff unchanged.

The first implementation must pass reduced K64/K128/K192 fixtures that isolate prime, one steady swap, repeated swaps, and the final peel before the production key runs. Production correctness requires bit-exact HIP output, the independent reference, grad-output and packed-weight mutations, exact launch rejection, the 40-byte ABI, and byte-identical rebuilds. Inspection must report exactly 9,216-byte LDS, zero private storage and spills, and the expected prime/steady/final WMMA and barrier paths. Prefer the selected 96-VGPR allocation class and use 120 VGPRs as the initial screen ceiling; any proposal above that ceiling requires an explicit occupancy calculation and a gain above the ordinary resource-bearing gate before it is run.

Measure exact-trip-normalized dynamic decode, LDS, wait, and barrier work against the selected one-buffer body. A serial nine-repeat screen advances only for a stable greater-than-2% gain against the selected M64 control while remaining faster than HIP. Promotion then requires two independent rotating 25-repeat confirmations, all correctness and mutation gates, and an independent byte-identical rebuild. If the complete candidate does not improve despite moving decoded next-tile work under current WMMA, close the premise without widening it to M128 or M256.

This newly actionable compact-N pipeline invalidated the prior Q6 backward stopping condition while it was pending. The implementation retained the exact `64x32x64` ownership and pad8 row mapping, allocated one independent unswizzled write-address VGPR, and used parity-controlled add/subtract swaps because each 4,608-byte buffer is not power-of-two aligned. The reduced fixtures exposed and closed three structural errors before production timing: the pipeline read register had to hold a pure buffer base rather than a decoder coordinate, nonzero-K unswizzled reads had to retain ordinary linear addressing rather than the XOR-8 helper, and `v_sub_nc_u32` operands had to subtract 4,608 from the address. K64 prime-only, K128 one-swap, and K192 repeated-swap/final-peel fixtures then matched HIP and the independent BF16 reference with zero differing elements.

The exact production artifact passed bit-exact HIP and independent-reference comparison, grad-output mutation, and packed-weight mutation. It used 77 VGPRs, 16 SGPRs, 9,216-byte LDS, two static barriers, and zero private storage or spills, preserving the parent's 96-VGPR allocation class and remaining below the 120-VGPR screen ceiling. The serial rotating nine-repeat discriminator was:

| Body | Median ms | TFLOPS | Relative to selected parent | Decision |
| --- | ---: | ---: | ---: | --- |
| HIP | `5.351755` | `12.1634` | `1.028838x` | Control |
| Selected one-buffer parent | `5.201748` | `12.5142` | `1.000000x` | Retain |
| Compact two-buffer pipeline | `5.248087` | `12.4037` | `1.008908x` | Reject |

The candidate remained `1.01975x` faster than HIP but was 0.89% slower than the selected parent, so it did not clear the stable greater-than-2% advancement gate and no 25-repeat confirmation was justified. Exact-trip accounting explains the result. Both bodies execute 3,880 decoded tiles, 31,040 WMMAs, 62,080 decoded-B stores, 62,080 LDS loads, and 27,160 explicit waits. The pipeline reduces full barriers from 7,760 to 3,880, but all four waves remain symmetric producers and consumers: each wave's four decode chunks are still on that wave's issue/dependency path between current-tile WMMAs. The non-power-of-two buffers additionally require per-trip parity and base-update instructions, while 9,216-byte LDS and 77 logical VGPRs cross no occupancy threshold relative to the 4,608-byte/76-VGPR parent. Barrier reduction therefore does not hide the decode work and does not offset buffer bookkeeping or the larger loop body.

The compact-N pipeline is a measured timing rejection. Its temporary code is removed after preserving artifacts under `~/tmp/torch-ggml-ops/ggtensile-q6-compact-m64-pipeline/`; the selected catalog remains authoritative. A complete recursive review still follows as a separate iteration and cannot reuse this measurement iteration as the final stopping pass.

Before declaring completion, reread this plan, Q6 HIP source and normalized ISA, all Q6 GGTensile artifacts and timing reports, Q3/Q4/Q5/Q8 experiment records, lower bounds, resource reports, rejected candidates, `rdna35-isa-markdown`, AMD LLVM definitions/tests, and relevant CK/TensileLite notes.

Classify every remaining idea as:
- retained and measured.
- rejected by correctness, resources, timing, or reproducibility.
- contract-incompatible or deferred with an explicit prerequisite.
- actionable and requiring another implementation/measurement cycle.

If the review identifies an actionable idea, implement and measure it, update this plan, and repeat the review from the new premise. The final review cannot pass in the same iteration that first discovers an actionable mechanism.

The campaign is exhausted only when no valid in-contract optimization idea remains, all three keys beat HIP with final evidence, and the residual bottleneck is quantitatively explained. Public runtime dispatch and artifact packaging remain deferred until the multi-quant integration project explicitly takes ownership.

### Subsequent final recursive review

The final review was repeated only after the compact pipeline result was recorded, its unselected implementation was removed, all 379 catalog-valid sources again matched the frozen source snapshot byte-for-byte, the 179-kernel bundle check passed, and the backward writer/coverage suite passed 99 tests. It classifies the remaining ideas as follows:
- Retained and measured: exact M64/M128/M256 ownership, pad8 LDS, packed VOPD decode, M256 next-packed prefetch, selected A/WMMA/store schedules, and exact immediate trip/row/block arithmetic.
- Timing rejections: scalar and ordinary packed extraction, alternate geometries and DepthU, XOR8/XOR16 and pad16/pad24 layouts, WGM/SIA/priority variants, high-plane lane sharing, repaired wider pipelines, M64 next-packed prefetch, and the exact compact M64 two-buffer pipeline.
- Unsupported or without a gain premise: power-of-two spacing for the compact pipeline removes only parity/base bookkeeping while increasing LDS; it leaves the measured 3,880 decodes, 31,040 WMMAs, 124,160 LDS operations, and 27,160 waits unchanged and cannot turn the rejected `1.008908x` candidate into a greater-than-2% parent win. Q6 signed scales are lane-unique. Pair-owner loading of the shared block `d` would retain the same issued VMEM instruction with fewer active duplicate lanes, whose addresses already coalesce, while adding EXEC and DPP work analogous to the materially rejected high-plane-sharing path. A metadata vector load spans unused scale bytes before the separately located FP16 `d` and adds extraction rather than removing an issued dependency.
- Deferred with explicit prerequisites: combined padding/XOR requires supported residual LDS-conflict counters and an occupancy-safe gain premise; a changed integer-plus-scale representation, prepared weights, shared scratchpads, producer fusion, persistent/grouped execution, split K, and external decode workspaces require model/API ownership outside this direct-packed 40-byte contract. Public dispatch and packaging remain integration work.

No actionable in-contract mechanism remains. M64 is quantitatively decode-limited (`5.230/2.659/3.283 ms` complete/WMMA/decode); M128 remains balanced (`6.631/3.776/3.291 ms`); M256 remains WMMA/accumulator-limited (`9.721/7.166/5.573 ms`). The compact pipeline demonstrates that barrier count is not the missing M64 overlap mechanism because all four symmetric waves still serialize their own decode chunks with their WMMA dependency chain. This separate stopping pass restores Q6 backward optimization exhaustion with all three selected exact keys faster than HIP and the selected catalog unchanged.

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
- Repaired-path retiming and renewed exhaustion. High-plane lane sharing, wider DepthU64 M64/M256 ownership, DepthU64 next-packed prefetch, and M128/M256 N64 pipelines pass exact correctness and mutation gates but regress selected controls by `4.7%` to `47.0%`; all are timing rejections and the selected catalog remains unchanged.
- Rejected compact M64 pipeline. The ownership-preserving `64x32x64` two-buffer emitter passed K64/K128/K192 and production correctness, used `77 VGPR/9216 B LDS` with no spills, and remained `1.01975x` faster than HIP, but regressed the selected parent by 0.89% (`5.248087` versus `5.201748 ms`). Exact dynamic accounting shows unchanged decode, WMMA, LDS, and wait work; halving barriers cannot hide symmetric same-wave decode dependencies. No candidate was retained, and the selected catalog remains unchanged.
- Subsequent recursive final review with no actionable mechanism remaining. From the restored byte-identical selected writer, the review closes power-of-two pipeline spacing, Q6 metadata sharing/vectorization, remaining layout/schedule variants, and model-owned representations by measured evidence or explicit prerequisites. M64 remains decode-limited, M128 balanced, and M256 WMMA/accumulator-limited.
- Public dispatch and artifact packaging, deferred.

## Post-Audit Relaxed-BF16 Reopening

Status: planned and unmeasured. The compact-pipeline rejection remains valid under exact RNE. The changed premise is conversion cost inside the Q6 decoder and output epilogue.

Add independent decoded-weight and output policies for `RNEPreserveNaN`, one-instruction `BiasRound`, and zero-instruction `Truncate`. Begin with decoded-weight staging on M64, whose `5.230/2.659/3.283 ms` complete/WMMA/decode floors make it the discriminator. Test output conversion separately. Advance to M128 only after a resource-neutral M64 candidate shows a repeatable body gain; do not infer an M256 benefit from M64 because M256 is WMMA/accumulator-limited.

This is the strongest backward conversion target in the cross-record priority review. The decode floor is 62.8% of complete M64 latency, and each staged value currently pays the tie-bit extraction plus correction add. A resource-neutral `Truncate` candidate therefore has a mid- to high-single-digit body-gain prior; `BiasRound` has a smaller instruction delta but may be the better numerical compromise. These are explicitly unmeasured planning ranges, not retention thresholds. Output-only conversion ranks below decoded staging, and M128/M256 remain transfer gates rather than part of the first screen.

The numerical harness, not the kernel, checks finite output for finite inputs and records differing elements, normalized RMSE, maximum and high-percentile error against the exact parent and independent Q6 reference. No NaN/Inf branch, clamp, or repair sequence belongs in the generated kernel. Approximate policies remain isolated from the exact catalog and require model-training integration, including loss and gradient stability, before they can be accepted.
