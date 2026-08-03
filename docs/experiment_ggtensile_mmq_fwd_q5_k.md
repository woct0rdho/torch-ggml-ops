# GGTensile Dense MMQ Forward Q5_K Plan

## Purpose

Build and optimize strict gfx1151 wave32 GGTensile assembly kernels for dense Q5_K forward over every exact production shape used by the current Qwen workload. The first objective is to close the largest kernel-level gaps against the installed HIP kernels. After every exact key is at least as fast as HIP, optimize the complete six-key model mix as far as repeatable evidence permits.

The campaign is autonomous and iterative. After each coherent implementation, correctness, measurement, or review change, update this file with the result and the next premise. Commit changes that are worth retaining; documentation-only checkpoints do not require commits.

## Contract

Target only:
- gfx1151, wave32, WMMA V1, BF16 input and output.
- Authoritative packed GGUF Q5_K weights consumed directly by the kernel.
- The installed HIP Q8_1 `F16_D4S4` activation producer and exact 144-byte activation workspace blocks.
- Exact-key artifacts with one `ProblemType`, `ProblemSize`, solution identity, assembly source, and code object.
- Zero private bytes, spills, scratch instructions, calls, and dynamic stack.
- Serial warmed rotating timing. Builds and independent correctness work may run in parallel, but timed GPU work must remain serial.
- No prepared weight representation, dense shadow, external decode workspace, split-K, Stream-K, persistent or grouped workgroups, online tuning, producer fusion, or public dispatch change during research.

Forward coordinates are:

```text
M = flattened activation rows
N = out_features
K = in_features

output[M,N] = input[M,K] @ dequant_q5_k(weight[N,K]).T
```

Q5_K has 256 logical values per 176-byte packed block. It shares Q4_K's FP16 `d`/`dmin`, eight six-bit scale/minimum fields, and 4-bit low payload, and adds a 32-byte high-bit plane. The activation workspace stores four Q8_1 subblocks per 128 values with `F16_D4S4` metadata.

## Exact Production Scope

The ordinary production rows are `M={2048,8192,32768}`. Q5_K appears in two shape families:

| Family | `(N,K)` | Representative tensor | Calls |
| --- | ---: | --- | ---: |
| Narrow K/V/shared gate/up | `(512,2048)` | `blk.0.ffn_gate_shexp.weight` | 21 |
| Shared-expert down | `(2048,512)` | `blk.0.ffn_down_shexp.weight` | 10 |

The six exact keys are the Cartesian product of those families and the three M values. There is no production Q5_K attention-output or attention-query family.

Priority is evidence-driven:
- Large candidate/HIP kernel margins.
- Large absolute kernel time.
- High model call count, especially the 21-call narrow family.
- Shapes with architectural or neighboring-format evidence for a materially faster mechanism.
- Smaller repeatable margins only after larger opportunities are exhausted.

A weighted total guides effort but never authorizes retaining a slower exact key.

## Initial Technical Premise

The completed Q4_K forward writer establishes a strong common body: direct packed input, decoded signed-int8 weight staging, Q8_1 activation staging, WMMA accumulation, metadata correction, and contiguous BF16 stores. Q5_K may reuse quant-neutral ownership, LDS, WMMA, synchronization, addressing, and epilogue mechanisms only after its decoder and metadata arithmetic are represented explicitly and tested independently.

The first Q5_K control should retain the Q4_K `128x64`, 128-thread decoded-staged geometry while adding the Q5 high-bit payload decode. The high-bit plane must be consumed during decode and must not remain live through the WMMA body. Initial search order:
- Port the strict direct and retained decoded-staged controls to Q5_K.
- Measure all six keys against exact installed HIP controls.
- Attack the largest observed margins with resource-neutral decode scheduling, high-bit extraction, metadata extraction, and low/high WMMA issue placement.
- Reuse Q4_K metadata-after-low and independent-extraction schedules only after exact Q5_K measurements.
- Search bounded Q5-specific payload/vector-load, nibble/high-bit extraction, VOPD, clause, wait, and epilogue neighborhoods.
- Change geometry or buffering only when profiling and lower bounds establish a new premise.

The existing exact HIP Q5_K kernels report approximately 244 VGPRs, 28 SGPRs, 38,400-byte LDS, and no private storage. A generated candidate may use a different envelope, but any resource increase needs a stable gain above 2% in confirmation.

## Correctness and Resource Gates

Before timing a candidate:
- Validate the exact 40-byte forward ABI, symbol, gfx1151 target, wave32 geometry, LDS, WMMA count, private segment, spills, scratch, calls, and dynamic stack.
- Require finite output and input, packed-weight, and Q8_1-workspace mutation sensitivity.
- When arithmetic order is unchanged, require bit-exact candidate/HIP output.
- Otherwise require candidate/HIP normalized RMSE `<=5e-4`, maximum absolute error `<=0.015625`, and independent-reference normalized RMSE `<=0.04`.
- Cover zero, one-hot, positive and negative extrema, low/high payload bits, scale/minimum fields, block boundaries, tile boundaries, reduced K trips, and output row/column boundaries.
- Require byte-identical rebuilds for any retained identity.

Every executable branch and emitted line in the assembly writer must be covered by unit tests. Coverage must include Q4_K regression paths as well as every new Q5_K decoder, schedule, geometry, and epilogue path.

## Measurement and Promotion

- Nine-repeat screens are pruning evidence only.
- Promotion requires two independent warmed rotating 25-repeat confirmations against both HIP and the retained parent.
- Measure complete fixed-quantizer calls and prequantized multiply bodies separately.
- The complete candidate call must be no slower than HIP for every exact key in both confirmations.
- Resource-bearing mechanisms require a stable gain above 2%.
- Resource-neutral unconditional instruction reductions or schedules may be retained when neutral or consistently favorable, but must not regress any exact selected key materially.
- Preserve immutable JSON timing, correctness, inspection, and source artifacts under `~/tmp/torch-ggml-ops/`.

## Optimization Phases

### Phase 0: Infrastructure and fresh controls

Add strict Q5_K forward problem modeling, exact inventory and catalog, writer/runtime/benchmark support, independent dequantized correctness, mutation gates, inspection, and line-complete unit coverage. Generate and measure a direct control and a Q4-shaped retained decoded-staged control for all six keys.

### Phase 1: Large-margin decoder and schedule work

Prioritize the largest fresh candidate/HIP margins. Search:
- Q5 high-bit extraction ownership and address formation.
- Packed low/high payload load width and clauses.
- Six-bit scale/minimum extraction and conversion placement.
- High-bit merge operations, shift hoists, masks, and legal gfx1151 VOPD pairs.
- Metadata reads between low/high WMMA batches.
- Independent scale/minimum extraction.
- Dependency-safe LDS wait ladders and bounded instruction scheduling.

### Phase 2: Family-specific main-loop and epilogue work

- Narrow: optimize the worst absolute or relative key first, then retime all M values because it carries 21 model calls.
- Shared-down: emphasize M2048 when store/stage cost dominates and M32768 when absolute body time creates a larger payoff.
- Search exact-key epilogue tiles-ahead, dependency width, priority, store clauses, and output-row increments only after the main body is fixed.

### Phase 3: Changed-premise geometry and buffering

Only after lower bounds and counters justify it, test reduced decoded-weight LDS, compact narrow ownership, bounded next-block payload overlap, or alternate DepthU. Do not repeat Q4_K geometries or double-buffer arrangements that lost without explaining why Q5_K's extra high-bit work changes the premise.

### Phase 4: Complete-call selection

Confirm every exact selected key twice against HIP and its retained parent. Then optimize the weighted complete-call mix without permitting an exact-key regression. Preserve per-key solution identities rather than silently generalizing a family schedule.

## Recursive Optimization-Exhaustion Review

Before declaring Q5_K complete, reread this plan; the Q5_K HIP source and normalized ISA; all Q5 forward artifacts, timings, lower bounds, profiles, counters, selected and rejected solutions; the completed Q4_K forward record; dense and grouped Q5_K forward/backward histories; relevant CK, TensileLite, EvoTensile, and hipBLASLt evidence; `~/rdna35-isa-markdown/`; and the AMD LLVM gfx11 instruction, scheduling, hazard, wait, and VOPD definitions and tests.

Classify every remaining idea as:
- retained and measured;
- rejected by correctness, resources, timing, or reproducibility;
- contract-incompatible or deferred with an explicit prerequisite; or
- actionable with a target key and measurement gate.

Every actionable idea must be implemented and measured, after which the entire review repeats from the new premise. Completion is allowed only when a fresh recursive review finds no actionable in-contract mechanism, every selected exact key beats HIP in two independent complete-call confirmations, the residual bottleneck is quantified, all writer lines are covered by unit tests, and retained artifacts rebuild byte-identically.

This is the same mandatory final-review rule used by the completed Q4_K campaign. Reaching HIP parity does not waive it.

## Campaign Record

### Plan opened

- Defined the six exact production keys and model call weights.
- Fixed the Q5_K packed-weight and Q8_1 `F16_D4S4` activation contracts.
- Selected the Q4-shaped decoded-staged body as the first architectural control, with Q5-specific high-bit decode kept explicit.
- Required complete-call HIP parity per exact key, two independent 25-repeat confirmations, byte-identical rebuilds, and recursive optimization exhaustion.
- Next: establish fresh HIP timings, add strict Q5_K forward infrastructure, and generate the first direct and decoded-staged controls.

### Fresh installed-HIP baseline

A serial warmed 25-repeat production benchmark was captured at `~/tmp/torch-ggml-ops/ggtensile-fwd-q5-k/hip-baseline-25.json`:

| Family | M=2048 | M=8192 | M=32768 |
| --- | ---: | ---: | ---: |
| Narrow `(512,2048)` | `0.198 ms` | `0.830 ms` | `3.235 ms` |
| Shared-down `(2048,512)` | `0.176 ms` | `0.737 ms` | `2.949 ms` |

The installed packed kernels are already `1.12-1.59x` faster than BF16 for narrow and `7.02-8.17x` faster for shared-down. The largest absolute kernel bodies are the two M32768 keys; narrow carries the larger 21-call model weight. Candidate/HIP margins remain unknown until the first generated controls exist, so infrastructure and exact-key correctness remain the immediate priority.

### Strict Q5_K forward infrastructure and first controls

- Added quant-aware forward `ProblemType`, inventory, catalog, validation, runtime, benchmark, and writer support for Q5_K without weakening Q4_K identities.
- Added a direct packed Q5_K high-bit decoder to the retained `128x64`, 128-thread decoded-staged body. The decoder loads the 32-byte high plane, merges bit 4 into the low nibbles, and releases high-bit state before WMMA.
- Added unit coverage for every new Q5_K writer branch and strict artifact inspection. All three initial schedules compile with 239 VGPRs, 16 SGPRs, 38,400-byte LDS, 32 static WMMAs, four barriers, eight store clauses, and zero private storage or spills.
- Generated all 18 six-key-by-three-schedule artifacts under `~/tmp/torch-ggml-ops/ggtensile-fwd-q5-k/initial-controls/`.
- The first strict M32768 narrow comparison was bit-exact. The serialized retained parent was `1.0712x` HIP for the multiply, metadata-after-low was `1.0468x`, and independent-extraction-plus-metadata-after-low was `1.0412x`; the last is the initial common control.

Nine-repeat common-control screens over all six keys were bit-exact to HIP and had independent-reference NRMSE `0.01371-0.01384`:

| Family | M=2048 complete/multiply | M=8192 complete/multiply | M=32768 complete/multiply |
| --- | ---: | ---: | ---: |
| Narrow | `1.0165/1.0203x` | `0.9960/0.9912x` | `1.0144/1.0412x` |
| Shared-down | `0.9969/1.0026x` | `1.0210/1.0279x` | `1.0099/1.0217x` |

The largest actionable kernel margin and absolute weighted opportunity is narrow M32768. Shared-down M8192 and M32768 follow. Next: reduce Q5 high-plane decode traffic/instructions and compare the normalized Q5 HIP decoder schedule before broader epilogue work.

### Q5_K payload-address contraction

The high-plane and low-nibble addresses now use two `v_mad_u32_u24` instructions plus immediate VMEM offsets instead of separate shifts and adds. This removes four static VALU instructions from each decoded block without changing registers, LDS, arithmetic, or payload traffic. The odd high-bit merge also masks `0x02020202` and shifts by three directly, removing one shift per packed dword. In a direct 25-repeat rotation on narrow M32768, the combined form was `0.99413x` the original common control and `1.01972x` HIP for the multiply; the address-only form was neutral at `0.99913x` the parent. Both unconditional resource-neutral reductions are retained.

A lane-shared high-plane load loaded `qh` only on two of eight row lanes and broadcast 16 dwords with `ds_bpermute_b32`. After correcting the lane mask it remained bit-exact, but a direct 25-repeat rotation was `1.00705x` the retained reduced-instruction parent and `1.02476x` HIP. A lower-overhead `dpp8:[0,1,0,1,0,1,0,1]` broadcast was also bit-exact but measured `1.00756x` and `1.00714x` its parent on narrow M2048 and M32768. The reduced global transactions do not repay either cross-lane distribution sequence, so high-plane lane sharing is rejected.

### High-bit schedule and lower-bound work

The decoder now shifts all four loaded `qh` dwords in place before starting low-payload nibble extraction, then batches the four low/high nibble preparations before the merge chains. This removes the single shared shift temporary from the critical dependency pattern without changing instruction count, registers, or arithmetic. On narrow M32768, batching measured `0.99067x` the prior parent in a direct 25-repeat rotation. Moving the four independent `qh` shifts to the front of each decoded row added another repeatable `0.99748x` reduction. The combined schedule is retained for every key.

A high-bit-free diagnostic on the original reduced-instruction parent was `0.96845x` the parent with merge arithmetic removed and `0.96702x` with both high-plane loads and merge arithmetic removed. The small difference between those floors shows that high-plane traffic is not the residual problem; byte-wise bit placement and merge issue are. Repeating the floor from the final selected narrow M32768 body found the merge-free artifact at `0.98190x` the final parent. The remaining approximately `1.8%` body floor is bounded by 16 byte-wise dwords, and gfx1151 has no shift/mask/move form that reduces both Q5 bit planes in fewer exact operations.

Measured alternatives were closed:
- Full-row and two-row preparation batching were neutral at `1.00005x` and `1.00019x` the parent.
- Even-first, odd-first, plane-grouped, shift-plus-`v_and_or_b32`, and in-place mask schedules either collapsed in a second rotation or moved different M values in opposite directions.
- Block/decode `s_setprio` controls were neutral or slower.
- Serialized and metadata-after-low controls at narrow M2048 measured `1.04899x` and `1.01516x` HIP, versus `1.00708x` for independent extraction before epilogue selection.
- Dynamic eight-group rolling, two-way unrolling, and full unrolling measured neutral to `1.00515x` slower. The larger static bodies increase instruction-fetch pressure without removing enough dynamic control.

### Exact epilogues and accumulator initialization

The complete 64-point epilogue grid was generated for priority narrow keys, and the Q4_K exact epilogues were transferred to shared-down. The retained exact schedules are:

| Exact key | Epilogue | Accumulator initialization |
| --- | --- | --- |
| Narrow M2048 | `a1d8-p0` | scalar copy |
| Narrow M8192 | `a8d1-p0` | scalar copy |
| Narrow M32768 | `a8d4-p3` | VOPD pairs |
| Shared-down M2048 | `a1d2-p2` | scalar copy |
| Shared-down M8192 | `a1d4-p2` | VOPD pairs |
| Shared-down M32768 | `a1d2-p2` | scalar copy |

Narrow M2048 is a complete-call selection sensitive to exact code-object identity. After default scalar initialization was normalized out of the JSON identity to preserve Q4_K default solution identities, the epilogue grid was rebuilt and retimed; `a1d8-p0` cleared two strict complete rotations at `0.99863x` and `0.99759x` HIP even though its prequantized body remained approximately tied. Narrow M32768 and shared-down use the measured Q4-shaped epilogue families, while the full narrow M32768 grid selected `a8d4-p3` over the common schedule.

`AccumulatorInitialization=VopdPair` represents a real emitted path rather than a post-generation patch. It replaces 72 scalar accumulator clears/copies with 36 legal `v_dual_mov_b32` instructions, using `v0` and `v1` as distinct source banks after the first zero pair. It is retained only on exact keys where complete-call evidence survives transfer. Narrow M32768 passed strict complete confirmation at `0.99555x` and `0.99547x` HIP. Shared-down M8192 improved its scalar parent by `0.99929x/0.99805x` in multiply rotations and `0.99929x/0.99805x` in direct complete rotations, then passed standalone strict calls at `0.98590x/0.99040x` HIP. Narrow M2048 and shared-down M2048/M32768 transfer controls were neutral or inconsistent in complete timing and remain scalar-copy identities.

### Final six-key confirmation

The native selected artifacts are bit-exact to the installed HIP multiply, finite, mutation-sensitive for input, packed weight, and Q8_1 workspace, and retain independent-reference NRMSE near `0.0137-0.0138`. Every artifact uses 239 VGPRs, 16 SGPRs, 38,400-byte LDS, 32 static WMMAs, four barriers, eight output clauses, and zero private bytes, spills, scratch, calls, or dynamic stack.

Two independent warmed rotating 25-repeat complete-call confirmations produced:

| Family/key | Confirmation A complete/multiply | Confirmation B complete/multiply |
| --- | ---: | ---: |
| Narrow M2048 | `0.998630/1.001844x` | `0.997586/1.003578x` |
| Narrow M8192 | `0.984979/0.969062x` | `0.983814/0.968110x` |
| Narrow M32768 | `0.995147/1.010124x` | `0.993254/1.009587x` |
| Shared-down M2048 | `0.976222/0.971238x` | `0.976573/0.970889x` |
| Shared-down M8192 | `0.988404/0.989870x` | `0.989655/0.989962x` |
| Shared-down M32768 | `0.981714/0.984611x` | `0.981746/0.987550x` |

All six exact complete calls beat HIP in both confirmations. Using 21 narrow calls and 10 shared-down calls, weighted complete latency was `131.006 ms` versus `132.296 ms` HIP in confirmation A (`0.99024x`) and `131.012 ms` versus `132.453 ms` HIP in confirmation B (`0.98912x`). Multiply-only ratios remain slightly above HIP for narrow M2048/M32768, but the campaign contract is complete-call parity and both exact calls clear that gate in both rotations.

Independent rebuilds were byte-identical to the retained timed artifacts. The six selected solution keys, generated assembly, code objects, inspection reports, and both confirmation reports are consolidated under `~/tmp/torch-ggml-ops/ggtensile-fwd-q5-k/retained-authoritative/`.

### Recursive optimization-exhaustion review

The review was repeated after the high-bit schedule gain and again after VOPD accumulator initialization transferred to shared-down M8192. The final classification is:
- Retained and measured: packed direct Q5 decode; address contraction; direct odd-plane mask/shift; in-place, `qh`-first batched decode; independent metadata extraction after low WMMA; six exact epilogues; exact-key VOPD accumulator initialization; and the common 128x64 four-wave LDS/WMMA body.
- Rejected by timing: LDS and DPP8 high-plane sharing; alternate high-bit merge forms and orders; scalar/serialized metadata paths; priority changes; full/two-row preparation; loop rolling and unrolling; broad startup VOPD compositions; unchanged 64x64, 128x128, 256x64, and compact 128x32 forward geometries; payload-only prefetch; larger activation or decoded-weight buffers; and nonselected epilogues.
- Rejected by ISA or premise: gfx1151 AND/AND VOPD pairing, exact two-plane byte insertion in fewer bitwise operations, direct-to-LDS, software instruction prefetch, split barriers, and useful shift/shift VOPD forms.
- Contract-incompatible: prepared or dense weights, external decode workspaces, split-K, Stream-K, persistent/grouped workgroups, producer fusion, online tuning, and public dispatch changes.

Backward Q5_K scalar extraction and padded 256x64 evidence does not reopen the forward body: it changes transposed ownership and LDS shape, while unchanged forward 256x64/compact ownership already lost and the final high-plane traffic floor is negligible. Grouped J32/J64 and row-task results are likewise small-row or routing mechanisms rather than dense-forward exact-key premises.

No remaining in-contract mechanism has an unmeasured first-order path. The residual final M32768 high-bit merge floor is approximately `1.8%`, but every legal exact lowering found either preserves the same operation count, introduces cross-lane distribution, or regresses another exact key. Reopening the campaign requires a materially new ISA operation, decoded layout, or occupancy-preserving geometry premise. Q5_K dense forward optimization is therefore complete; public dispatch remains a separate deferred integration phase.
