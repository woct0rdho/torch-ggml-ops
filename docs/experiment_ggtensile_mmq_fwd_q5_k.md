# GGTensile MMQ Forward Q5_K Plan

## Purpose

Build and optimize strict gfx1151 wave32 GGTensile assembly kernels for Q5_K forward over every exact production shape used by the current Qwen workload. The authoritative objective is the prequantized packed multiply body using the fixed HIP-produced Q8_1 `F16_D4S4` workspace. Every exact key must be at least as fast as the installed HIP multiply within measurement error, after which optimization continues only while repeatable in-contract upside remains.

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
- A replacement schedule must first demonstrate repeatable improvement over its retained parent in direct candidate comparisons.
- Final promotion requires two independent warmed rotating 25-repeat confirmations containing exactly the HIP multiply and selected GGTensile multiply.
- Prefer both GGTensile/HIP latency medians `<=1.0x`. A small positive median counts as parity only when that confirmation's paired bootstrap confidence interval includes zero.
- The fixed HIP Q8_1 quantizer is identical for both paths and excluded from ranking and promotion. Complete-call timing is diagnostic only.
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

### Phase 4: Multiply-only selection

Confirm every exact selected key twice with only the HIP and GGTensile prequantized multiplies in the rotating set. Apply the paired measurement-parity rule independently to every confirmation and preserve per-key solution identities rather than silently generalizing a family schedule.

## Recursive Optimization-Exhaustion Review

Before declaring Q5_K complete, reread this plan; the Q5_K HIP source and normalized ISA; all Q5 forward artifacts, timings, lower bounds, profiles, counters, selected and rejected solutions; the completed Q4_K forward record; dense and grouped Q5_K forward/backward histories; relevant CK, TensileLite, EvoTensile, and hipBLASLt evidence; `~/rdna35-isa-markdown/`; and the AMD LLVM gfx11 instruction, scheduling, hazard, wait, and VOPD definitions and tests.

Classify every remaining idea as:
- retained and measured;
- rejected by correctness, resources, timing, or reproducibility;
- contract-incompatible or deferred with an explicit prerequisite; or
- actionable with a target key and measurement gate.

Every actionable idea must be implemented and measured, after which the entire review repeats from the new premise. Completion is allowed only when a fresh recursive review finds no actionable in-contract mechanism, every selected exact key beats or reaches measurement parity with the HIP multiply in two independent dedicated confirmations, the residual bottleneck is quantified, all writer lines are covered by unit tests, frozen Q4_K assembly remains byte-identical, and retained artifacts rebuild byte-identically.

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

The original complete-call campaign generated the 64-point power-of-two epilogue grid for priority narrow keys and transferred measured Q4_K epilogues to shared-down. The multiply-only continuation then expanded the Q5_K exact space to every tiles-ahead and dependency width in `[1,8]`. The final retained schedules are:

| Exact key | Epilogue | Accumulator initialization |
| --- | --- | --- |
| Narrow M2048 | `a1d8-p0` | scalar copy |
| Narrow M8192 | `a8d1-p0` | scalar copy |
| Narrow M32768 | `a7d3-p3` | VOPD pairs |
| Shared-down M2048 | `a1d2-p2` | scalar copy |
| Shared-down M8192 | `a1d4-p2` | VOPD pairs |
| Shared-down M32768 | `a1d2-p2` | scalar copy |

Narrow M2048 remains `a1d8-p0`: it cleared the former complete-call gate, met the final multiply-only parity rule, and no expanded scalar schedule confirmed a repeatable improvement. Narrow M32768 formerly used `a8d4-p3`, but the multiply-only campaign promoted resource-neutral `a7d3-p3` after two parent comparisons and two dedicated HIP comparisons. The three shared-down identities and narrow M8192 remain unchanged.

`AccumulatorInitialization=VopdPair` represents a real emitted path rather than a post-generation patch. It replaces 72 scalar accumulator clears/copies with 36 legal `v_dual_mov_b32` instructions, using `v0` and `v1` as distinct source banks after the first zero pair. It is retained only on exact keys with repeatable evidence: narrow M32768 and shared-down M8192. Narrow M2048 and shared-down M2048/M32768 transfer controls were neutral or inconsistent and remain scalar-copy identities.

### Former six-key confirmation

The native selected artifacts are bit-exact to the installed HIP multiply, finite, mutation-sensitive for input, packed weight, and Q8_1 workspace, and retain independent-reference NRMSE near `0.0137-0.0138`. Every artifact uses 239 VGPRs, 16 SGPRs, 38,400-byte LDS, 32 static WMMAs, four barriers, eight output clauses, and zero private bytes, spills, scratch, calls, or dynamic stack.

The last full six-key cohort predates the multiply-only continuation. The table therefore reports only its prequantized multiply bodies, not quantization plus multiply. Logical throughput is `2*M*N*K/(median_ms*1e9)`, and speedup is `HIP median time / GGTensile median time`. A/B are the two independent rotating 25-repeat confirmations.

| Family | `(M,N,K)` | HIP TFLOPS A/B | GGTensile TFLOPS A/B | Speedup vs HIP A/B |
| --- | ---: | ---: | ---: | ---: |
| Narrow | `(2048,512,2048)` | `24.749/24.432` | `24.704/24.345` | `0.9982x/0.9964x` |
| Narrow | `(8192,512,2048)` | `26.018/25.915` | `26.848/26.769` | `1.0319x/1.0329x` |
| Narrow | `(32768,512,2048)` | `27.208/27.256` | `26.935/26.997` | `0.9900x/0.9905x` |
| Shared down | `(2048,2048,512)` | `24.127/24.516` | `24.842/25.251` | `1.0296x/1.0300x` |
| Shared down | `(8192,2048,512)` | `25.817/25.811` | `26.081/26.073` | `1.0102x/1.0101x` |
| Shared down | `(32768,2048,512)` | `26.106/26.147` | `26.514/26.477` | `1.0156x/1.0126x` |

All six exact complete calls beat HIP in both confirmations. Using 21 narrow calls and 10 shared-down calls, weighted complete latency was `131.006 ms` versus `132.296 ms` HIP in confirmation A (`0.99024x`) and `131.012 ms` versus `132.453 ms` HIP in confirmation B (`0.98912x`). Those complete-call totals are historical diagnostics. Under the old mixed protocol, narrow M2048 and M32768 were the only rows with multiply speedup below `1.0x`, which triggered the continuation below.

Independent rebuilds were byte-identical to the retained timed artifacts. The six selected solution keys, generated assembly, code objects, inspection reports, and both confirmation reports are consolidated under `~/tmp/torch-ggml-ops/ggtensile-fwd-q5-k/retained-authoritative/`.

### Recursive optimization-exhaustion review

The review was repeated after the high-bit schedule gain and again after VOPD accumulator initialization transferred to shared-down M8192. The final classification is:
- Retained and measured: packed direct Q5 decode; address contraction; direct odd-plane mask/shift; in-place, `qh`-first batched decode; independent metadata extraction after low WMMA; six exact epilogues; exact-key VOPD accumulator initialization; and the common 128x64 four-wave LDS/WMMA body.
- Rejected by timing: LDS and DPP8 high-plane sharing; alternate high-bit merge forms and orders; scalar/serialized metadata paths; priority changes; full/two-row preparation; loop rolling and unrolling; broad startup VOPD compositions; unchanged 64x64, 128x128, 256x64, and compact 128x32 forward geometries; payload-only prefetch; larger activation or decoded-weight buffers; and nonselected epilogues.
- Rejected by ISA or premise: gfx1151 AND/AND VOPD pairing, exact two-plane byte insertion in fewer bitwise operations, direct-to-LDS, software instruction prefetch, split barriers, and useful shift/shift VOPD forms.
- Contract-incompatible: prepared or dense weights, external decode workspaces, split-K, Stream-K, persistent/grouped workgroups, producer fusion, online tuning, and public dispatch changes.

Backward Q5_K scalar extraction and padded 256x64 evidence does not reopen the forward body: it changes transposed ownership and LDS shape, while unchanged forward 256x64/compact ownership already lost and the final high-plane traffic floor is negligible. Grouped J32/J64 and row-task results are likewise small-row or routing mechanisms rather than MMQ forward exact-key premises.

No remaining in-contract mechanism had an unmeasured first-order path under the complete-call objective. The residual final M32768 high-bit merge floor is approximately `1.8%`, but every legal exact lowering found either preserves the same operation count, introduces cross-lane distribution, or regresses another exact key.

## Multiply-only continuation

The promotion objective is now the prequantized packed multiply only. The fixed HIP Q8_1 `F16_D4S4` producer is identical for HIP and GGTensile and is no longer part of selection, ranking, or completion. Complete-call timing remains diagnostic but cannot retain a slower multiply body.

The former six-key table leaves narrow M2048 and M32768 as the only old-protocol rows below `1.0x` multiply speedup. Narrow M8192 and all three shared-down keys remain selected under the multiply-only objective; the two narrow rows are retimed with the dedicated protocol below.

Nine-repeat screens still prune only. Multiply promotion requires two independent warmed rotating 25-repeat confirmations. A key is at parity only when both confirmation medians are no slower than HIP, or when a small positive median is not statistically distinguishable from HIP under a paired bootstrap confidence interval computed from that confirmation's rotating samples. The campaign should target `<=1.0x` in both medians rather than relying on the statistical exception. Correctness, mutation, resource, exact-key identity, byte-identical rebuild, and recursive-review requirements are unchanged.

The review restarted from narrow M32768 and then M2048. Multiply-specific epilogues, startup schedules, decode/WMMA issue order, and lower-bound mechanisms were reconsidered even when neutral or unfavorable for the complete call. Previously rejected mechanisms reopened only when their recorded multiply result or a changed scheduling premise could close one of the two exact gaps. Both keys then cleared the multiply gate, all six keys completed the dedicated confirmation protocol, and the recursive review was repeated before closure.

Q4_K assembly is a frozen regression contract throughout this continuation. Before changing any shared forward model, validator, or writer path, capture the current generated assembly for every retained Q4_K production identity. After each retained implementation change, regenerate and require byte-identical Q4_K assembly. Any Q4_K assembly difference must be isolated, explained, revalidated across all 12 exact keys, and explicitly approved rather than accepted as incidental fallout from Q5_K work.

### Dedicated multiply-only protocol

The earlier `1.01x` narrow M32768 result came from a four-way timing rotation containing HIP complete, GGTensile complete, HIP multiply, and GGTensile multiply. That protocol is valid for complete-call selection but injects two quantizer launches between multiply samples and is not authoritative for the new objective. The dedicated protocol quantizes once, warms only the two prequantized multiply kernels, and alternates HIP/GGTensile launch order for 25 repeats.

Early dedicated retiming established that the apparent narrow-key deficits were primarily a four-operation timing-context artifact. Those runs were used to decide whether the continuation remained plausible; the final authoritative six-key measurements are reported together below rather than duplicated here.

### Multiply-specific epilogue search

The Q5_K epilogue schema was expanded from powers of two to every exact tiles-ahead and dependency width in `[1,8]`, preserving priorities `[0,3]` and both accumulator-initialization emitters. Validation remains strict: normalization must equal the implemented independent-extraction control after those four fields are removed.

All 256 VOPD epilogue combinations were built and screened on narrow M32768. `a7d3-p3-vopd` improved the old `a8d4-p3-vopd` parent by `0.998237x` and `0.998231x` in two independent 25-repeat rotating candidate comparisons. Its initial dedicated paired intervals were `[-14.54, 4.78] us` and `[-14.34, -2.39] us`; the final all-key protocol below confirmed parity again. This exact schedule is the selected multiply-only M32768 identity; it is resource-neutral and leaves the 239-VGPR, 16-SGPR, 38,400-byte-LDS envelope unchanged.

For narrow M2048, the original 64-point power-of-two grid, a focused 60-point non-power neighborhood, and all 196 previously unmeasured scalar combinations were screened. `a2d5-p3` led the exhaustive screen at approximately `0.9948x` the retained parent, but direct 25-repeat parent ratios were `0.99817x` and `1.00168x`; the apparent gain did not confirm. The retained `a1d8-p0` therefore remains selected, and no scalar epilogue schedule is left unmeasured.

### Final multiply-only result

This is the authoritative prequantized multiply result for all six production keys. Each A/B entry is an independent warmed 25-repeat rotation containing exactly the HIP multiply and selected GGTensile multiply. Logical throughput is `2*M*N*K/(median_ms*1e9)`, and speedup is `HIP median time / GGTensile median time`.

| Family | `(M,N,K)` | Final identity | HIP TFLOPS A/B | GGTensile TFLOPS A/B | Speedup vs HIP A/B | Paired bootstrap 95% CI, us A/B |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Narrow | `(2048,512,2048)` | `a1d8-p0` | `24.184/24.462` | `24.110/24.438` | `0.9970x/0.9990x` | `[-0.120,0.430]/[-0.650,0.080]` |
| Narrow | `(8192,512,2048)` | `a8d1-p0` | `27.235/27.211` | `28.216/28.047` | `1.0360x/1.0307x` | `[-22.980,-19.220]/[-20.809,-18.059]` |
| Narrow | `(32768,512,2048)` | `a7d3-p3`, VOPD | `28.021/27.962` | `27.990/27.968` | `0.9989x/1.0002x` | `[-5.681,7.100]/[-7.710,3.740]` |
| Shared down | `(2048,2048,512)` | `a1d2-p2` | `23.871/24.587` | `24.494/25.261` | `1.0261x/1.0274x` | `[-5.330,-3.970]/[-5.350,-4.110]` |
| Shared down | `(8192,2048,512)` | `a1d4-p2`, VOPD | `26.846/27.004` | `27.247/27.383` | `1.0149x/1.0141x` | `[-13.980,-8.161]/[-11.880,-7.570]` |
| Shared down | `(32768,2048,512)` | `a1d2-p2` | `27.294/26.894` | `27.762/27.362` | `1.0172x/1.0174x` | `[-49.850,-35.029]/[-49.828,-34.180]` |

Narrow M2048 and M32768 satisfy the stated measurement-error exception: their small latency deficits are not statistically distinguishable from HIP because both paired intervals include zero in both confirmations. Narrow M8192 and all shared-down keys are faster than HIP in both medians with intervals entirely below zero. Every candidate output was bit-exact to the HIP multiply; strict correctness, finite-output, independent-reference, and mutation checks remain satisfied.

The final selected artifacts, correctness reports, inspection reports, and dedicated confirmation reports are consolidated under `~/tmp/torch-ggml-ops/ggtensile-fwd-q5-k/retained-multiply-authoritative/`. Two independent rebuilds reproduced all six solution keys, generated assemblies, and code objects byte-for-byte.

### HIP/GGTensile issue evidence

A multiply-only profiler pass on narrow M32768 found these median per-workgroup counters:

| Counter | HIP | GGTensile | Interpretation |
| --- | ---: | ---: | --- |
| All SQ instructions | `103,776` | `88,056` | GGTensile executes about 15% fewer instructions |
| Branch instructions | `288` | `352` | two rolled four-group loops add 64 branches |
| Instruction-fetch waits | `2,467` | `2,817` | GGTensile has about 14% more fetch waiting |
| SQ busy cycles | `64,021` | `63,714` | effectively tied under profiling |

The profiler perturbs absolute timing, so these counters are diagnostic rather than promotion evidence. They exclude raw instruction count, packed traffic, occupancy, and total SQ busy work as explanations for a material residual deficit. The remaining variance is most consistent with instruction placement, branch/fetch behavior, and clock/cache state under the old mixed timing protocol.

The installed HIP assembly aggressively pairs startup address operations with accumulator moves through VOPD and statically schedules long decode/WMMA/metadata regions. GGTensile instead uses a smaller rolled body, fewer total instructions, and VOPD-paired accumulator initialization. Earlier full and two-way unrolling regressed, so HIP's larger static layout is evidence for scheduling review, not evidence that copying its unrolling policy will win. Q5 high-bit insertion remains the largest quantified local body floor at approximately `1.8%`; HIP pre-shifts high planes and uses `v_and_or_b32`, while the retained GGTensile path uses direct masks plus `v_lshl_or_b32`. Measured equivalent four-instruction substitutions and cross-lane sharing did not improve the final body.

The apparent remaining slowdown was therefore primarily a measurement-scope artifact: once complete calls are removed from the rotation, both formerly open narrow keys meet multiply parity. The final recursive review found no new in-contract premise with plausible unmeasured upside. All expanded scalar and VOPD epilogues were covered where they could matter; startup pairing, decode schedules, merge forms, lane sharing, loop unrolling, priority controls, alternate geometries, payload prefetch, and larger buffering had already been measured or rejected by ISA and resource premises. The M32768 merge-free lower bound remains only approximately `1.8%`, while tested exact replacements did not realize it without offsetting work.

The multiply-only continuation is complete. `a7d3-p3-vopd` is represented in the exact catalog, all six keys pass the two-confirmation gate, independent rebuilds are byte-identical, and regenerated assembly for all 12 frozen Q4_K identities was byte-identical to the pre-continuation baseline when the campaign closed. The later direction-naming refactor intentionally changed only GGTensile operation identities, kernel symbols, and descriptive comments: after normalizing those names, every instruction and directive is unchanged across all 12 Q4_K and all six selected Q5_K artifacts, and every artifact reassembles and passes strict inspection with its retained resource envelope. No further Q5_K forward performance experiment is justified without a changed compiler, ISA, hardware, or contract premise.
