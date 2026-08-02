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

1. Large candidate/HIP kernel margins.
2. Large absolute kernel time.
3. High model call count, especially the 21-call narrow family.
4. Shapes with architectural or neighboring-format evidence for a materially faster mechanism.
5. Smaller repeatable margins only after larger opportunities are exhausted.

A weighted total guides effort but never authorizes retaining a slower exact key.

## Initial Technical Premise

The completed Q4_K forward writer establishes a strong common body: direct packed input, decoded signed-int8 weight staging, Q8_1 activation staging, WMMA accumulation, metadata correction, and contiguous BF16 stores. Q5_K may reuse quant-neutral ownership, LDS, WMMA, synchronization, addressing, and epilogue mechanisms only after its decoder and metadata arithmetic are represented explicitly and tested independently.

The first Q5_K control should retain the Q4_K `128x64`, 128-thread decoded-staged geometry while adding the Q5 high-bit payload decode. The high-bit plane must be consumed during decode and must not remain live through the WMMA body. Initial search order:

1. Port the strict direct and retained decoded-staged controls to Q5_K.
2. Measure all six keys against exact installed HIP controls.
3. Attack the largest observed margins with resource-neutral decode scheduling, high-bit extraction, metadata extraction, and low/high WMMA issue placement.
4. Reuse Q4_K metadata-after-low and independent-extraction schedules only after exact Q5_K measurements.
5. Search bounded Q5-specific payload/vector-load, nibble/high-bit extraction, VOPD, clause, wait, and epilogue neighborhoods.
6. Change geometry or buffering only when profiling and lower bounds establish a new premise.

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
- Preserve immutable JSON timing, correctness, inspection, hashes, and source artifacts under `~/tmp/torch-ggml-ops/`.

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
