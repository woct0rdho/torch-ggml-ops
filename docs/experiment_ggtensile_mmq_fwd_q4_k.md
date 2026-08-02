# GGTensile Dense MMQ Forward Q4_K Plan

## Purpose

Continue the strict gfx1151 wave32 GGTensile assembly optimization campaign for dense Q4_K forward. The immediate objective is kernel performance only: close the remaining exact-key gaps against the existing HIP kernels, then optimize the complete production mix. Public dispatch, generated bundle tables, source distribution, and API integration are outside the current phase and must not be changed by optimization work.

The campaign must stop only after a recursive review finds no new valid in-contract optimization mechanism. The final review rule remains mandatory even after all currently open keys reach HIP parity.

The existing HIP DS4 Q8_1 activation quantizer is fixed producer infrastructure. It is launched before both the HIP and GGTensile multiply controls and is not a forward tuning knob.

## Contract

Target only:

- gfx1151, wave32, WMMA V1, and BF16 input/output activations.
- Authoritative packed GGUF Q4_K weights with direct in-kernel packed decode.
- The existing HIP Q8_1 DS4 workspace and its exact 144-byte block layout.
- One exact forward `ProblemType` and one exact `ProblemSize` per research artifact.
- Direct packed consumption with no prepared weights, dense shadows, external decode workspace, split-K, persistent workgroups, grouped MMQ, online tuning, or private storage.
- Zero private bytes, spills, scratch instructions, calls, and dynamic stack.
- Serial warmed rotating-control timing. Builds and independent correctness jobs may run in parallel; timed GPU work never does.
- No changes to public dispatch, `csrc/generated/`, bundle packaging, or the installed HIP implementation while optimizing candidates.

Forward coordinates are:

```text
M = flattened activation rows
N = out_features
K = in_features

output[M,N] = input[M,K] @ dequant_q4_k(weight[N,K]).T
```

Q4_K has 256 logical values per 144-byte packed block. The fixed DS4 producer emits Q8_1 blocks with 128 signed int8 values and four `(d,sum)` FP16 pairs. The producer may allocate a padded row stride for the selected tile, but only real rows are launched. Every candidate must use the exact producer stride and must never consume a padded row as a valid result.

## Exact Production Scope

Each ordinary projection runs at physical batches 1, 4, and 16, yielding `M={2048,8192,32768}`. The 12 exact keys are:

| Family | `(N,K)` | Representative tensor | Calls | Current research status |
| --- | ---: | --- | ---: | --- |
| Narrow K/V/shared gate/up | `(512,2048)` | `blk.5.ffn_gate_shexp.weight` | 70 | 3 open |
| Shared-expert down | `(2048,512)` | `blk.5.ffn_down_shexp.weight` | 30 | M2048 open; M8192/M32768 measured selected schedules |
| Attention output | `(2048,4096)` | `blk.3.attn_output.weight` | 10 | 3 open |
| Attention query | `(8192,2048)` | `blk.39.attn_q.weight` | 1 | 3 open |

The two shared-down M8192 and M32768 extraction schedules are measured research selections, not permission to add or modify public API wiring in this phase. The remaining ten keys stay on their existing fallback path while optimization continues.

## Current Baseline and Priority

The retained complete-call 25-repeat snapshots provide the following candidate/HIP ratios. They are suitable for prioritization, but fresh paired rotations are required before promotion because the historical captures span different clock and thermal regimes.

| Family `(N,K)` | M=2048 | M=8192 | M=32768 |
|---|---:|---:|---:|
| Narrow `(512,2048)` | `1.0217x` | `1.0410x` | `1.0038x` |
| Shared-down `(2048,512)` | `1.0282x` | `0.9915x` selected | `0.9837x` selected |
| Attention-output `(2048,4096)` | `1.0108x` | `1.0057x` | `0.9989x` |
| Query `(8192,2048)` | `1.0038x` | `1.0075x` | `0.9973x` |

The first optimization target is narrow M8192. Its current complete gap is approximately `0.034 ms` per call and its call count is 70, giving the largest observed weighted deficit. Narrow M32768 is next by absolute weighted cost, followed by attention-output M8192, narrow M2048, shared-down M2048, and attention-output M2048. Attention-output M32768 and query M32768 should receive fresh confirmation before new search because their retained snapshots are already at or below HIP.

The weighted objective is diagnostic and prioritizes work. It never authorizes retaining a slower exact key.

## Fixed Activation Producer

The installed HIP `quantize_bf16_q8_1_ds4` HSACO remains the producer control. For each 32-value group it computes the absolute maximum, `d=amax/127`, signed-int8 nearest quantization, and the input sum needed by Q4_K zero-point correction.

The benchmark must keep three explicit measurements:

- Fixed HIP quantizer plus HIP packed multiply: public HIP control for comparison only.
- Fixed HIP quantizer plus GGTensile packed multiply: complete candidate.
- Prequantized DS4 workspace with HIP and GGTensile multiply bodies: diagnosis only.

The produced DS4 workspace must be byte-identical for HIP and GGTensile. No alternate rounding, clamp, reciprocal, reduction, metadata layout, workspace lifetime, or fused producer may be selected through this campaign.

## Established Evidence

The retained decoded-staged body uses a `128x64` tile with 128 threads, 239 VGPRs, 16 SGPRs, 38,400 bytes of LDS, 32 static WMMA instructions, four barriers, and eight output-store clauses. It has already passed strict correctness, input mutation, packed-weight mutation, DS4-workspace mutation, independent-reference checks, and byte-identical rebuild checks for all 12 keys.

Current profiling shows fewer VALU instructions and flat loads than HIP, but higher instruction-fetch waiting and aggregate wait-any cycles. Retained and HIP K512 controls have comparable LDS conflict and L2-hit rates. The residual target is issue/dependency balance and synchronization, not raw packed traffic.

The lower bounds show that store-only work is approximately `24-27%` of retained latency for K512 and `2-7%` for K2048/K4096. Decode/stage/store work is approximately `44-57%`, while memoryless math/control/store work is approximately `86-88%`. These floors are diagnostic and are not additive.

Already rejected or closed mechanisms include the unchanged `64x64`, `128x128`, and `256x64` geometries; flattened workgroup clustering; broad activation prefetch; loop NOP padding; instruction-prefetch descriptor tuning; cross-lane B32/B64/B128 output packing; activation-load and packed-weight clauses; broad metadata byte permutes; broad priority settings; and gfx1151 split barriers. The compiler-shaped assembly oracle remains diagnostic evidence, not a packaged solution.

## Optimization Phases

### Phase 0: Fresh paired controls

Before new emitter work, measure all 12 exact keys in fresh warmed rotating processes. Compare HIP, the retained writer, and the two existing research schedules where applicable. Record complete-call and prequantized medians, paired differences, clock regime, and outlier handling.

Run attention-output M32768 and query M32768 through fresh confirmation first if they remain below HIP. Do not infer selection from one favorable median or from a weighted aggregate.

### Phase 1: Static schedule and barrier reduction

Search the following resource-neutral or bounded-resource schedules first on narrow M8192 and shared-down M2048:

- Merge the two emitted four-group loops into one dynamic eight-group loop while preserving group order and accumulation order. The current body has 32 static WMMA instructions for two four-group loop bodies; one rolled body should reduce static code and instruction-fetch pressure.
- Elide only the final barrier after the last K block, after proving that no later workgroup or wave consumes the overwritten LDS state.
- Retune LDS-read and WMMA issue distance using explicit one-tile, two-tile, and four-tile local-read ladders. Vary `lgkmcnt` thresholds only with a generated dependency-safe emitter.
- Interleave independent metadata conversion, LDS reads, and WMMA issue without changing the integer accumulation or FP32 correction order.
- Keep the existing eight contiguous output-store clauses while testing whether conversion dependencies can be exposed earlier without increasing live VGPRs.

The primary counters for this phase are `SQ_WAIT_IFETCH`, aggregate wait-any cycles, barrier cycles, VALU issue, LDS issue, and total elapsed time.

### Phase 2: Operand-buffer pipelines

Test only buffer arrangements that fit the gfx1151 LDS and register contract:

- A second 18,432-byte DS4 activation plane raises the current allocation from 38,400 to 56,832 bytes. Compare eager two-plane staging with a schedule that stages the second plane while the first four groups compute.
- A second decoded-weight buffer raises the current allocation to approximately 57,856 bytes. For K2048 and K4096, test bounded next-Q4-block packed reads and decode into the alternate buffer while the current block is consumed.
- Start with one next-block vector or one next-plane chunk. Expand only when inspection proves the candidate remains below 256 VGPRs with no spills or private bytes.
- Treat PGR1-style overlap as the first candidate. Do not attempt PGR2 until a preceding schedule creates enough register lifetime headroom.

The expected winning shape is a generated software pipeline with fewer barriers and hidden next-block staging, not simply more preloaded data. K512 M2048 is still important because it has a large store/stage fraction, but K2048/K4096 receive priority for next-block overlap.

### Phase 3: Geometry and ownership

Do not repeat rejected geometries without a changed implementation premise. Add a narrow-specific compact bracket:

- `128x32` ownership with 64 or 128 threads, using reduced decoded-weight LDS and a strict wave mapping.
- `64x32` ownership only if the `128x32` body demonstrates a resource or residency benefit.
- `DepthU=64` only as a bundled local-read and group-loop schedule, not as an isolated parameter change.
- WGM values `{1,2,4}` only after a new geometry wins; workgroup mapping and macro-tile shape must be measured together.

For attention-output and query, retain `128x64` unless a structural pipeline produces a clear resource-neutral improvement. Do not trade the established four-wave ownership for a larger tile merely to reduce instruction count.

### Phase 4: Family-specific tuning

- Narrow: optimize M8192 first, transfer only measured mechanisms to M32768 and M2048, then confirm all three exact keys.
- Shared-down: focus M2048 on loop/barrier reduction and conversion/store overlap. Reuse the selected M8192/M32768 extraction identities only as parent controls; do not generalize independent extraction without fresh exact-key evidence.
- Attention-output: focus on main-loop issue balance and bounded next-block prefetch because the store floor is small.
- Query: confirm M32768, then address M8192 and M2048 only with mechanisms that improve the complete call, not just the multiply body.

After a family-specific candidate clears confirmation, retime the other two M values before treating the mechanism as transferable.

## TensileLite Knob Mapping

TensileLite concepts are research guidance and require real alternate emitters in the custom writer. They are not valid merely because a field exists in a YAML file.

- `MacroTile`, `MIWaveTile`, and `WorkGroup`: map to the new narrow compact ownership bracket and any changed wave-to-output mapping.
- `DepthU`: test 32 versus 64 only with matching group-loop, LDS, and register schedules.
- `PrefetchGlobalRead`: implement bounded next-block Q4_K or DS4 overlap; start at one tile.
- `PrefetchLocalRead` and `ScheduleIterAlg`: implement explicit LDS-read/WMMA dependency schedules and measure wait thresholds.
- `1LDSBuffer`: model activation-only or weight-only ping-pong because a full double copy is too large for the current LDS allocation.
- `StorePriorityOpt` and store scheduling: use only for exact epilogue candidates after the main-loop schedule is fixed.
- `WorkGroupMapping` and `StaggerU`: defer until geometry changes expose a cache or launch-order problem; current cache data does not justify broad mapping work.
- `DirectToLds`: unavailable on gfx1151; LLVM gfx11 assembly rejects the required `buffer_load_* ... lds` form.
- `DirectToVgpr`: existing direct activation controls were slower and require a new lower-bound reason before reopening.
- `SwInstructionPrefetch` and split barrier signal/wait: gfx12-or-later mechanisms and not available to this campaign.
- `StoreRemap`, cross-lane output packing, split-K, Stream-K, and persistent workgroups: outside the current direct-packed contract or already rejected by correctness/timing.

## Correctness and Resource Gates

Before timing any candidate:

- Inspect symbol, exact forward ABI, gfx1151 metadata, LDS size, WMMA count, private segment, spills, scratch, calls, dynamic stack, and workgroup geometry.
- Compare the multiply output bit-for-bit with HIP when the integer WMMA and output order are unchanged.
- For deliberate reorderings, require candidate/HIP normalized RMSE `<=5e-4`, maximum absolute error `<=0.015625`, and independent-reference normalized RMSE `<=0.04`.
- Run baseline, input mutation, packed-weight mutation, DS4-workspace mutation, finite-output checks, and independent GGUF-dequantized BF16 reference checks.
- Preserve reduced-K, one-hot, zero, positive/negative maximum, scale/sum, packed payload, metadata, block-boundary, tile-boundary, and output row/column boundary fixtures.
- Require no private bytes, spills, scratch instructions, calls, or dynamic stack. A candidate that changes the producer or consumes a prepared representation is rejected.

Nine-repeat screens are exploratory only. A resource-bearing mechanism requires a stable gain above 2% in rotating 25-repeat confirmation. Unconditional instruction reductions and resource-neutral scheduling improvements may be retained when neutral or consistently favorable.

Final selection requires at least two independent rotating 25-repeat confirmations against HIP and the retained parent, with the complete fixed-quantizer call below HIP and no material regression in either rotation. A weighted average never authorizes retaining a slower exact key.

## Research and Public-API Boundary

This phase produces research artifacts, immutable timing records, strict solution identities, and optimization evidence only. Do not add kernels to public dispatch, regenerate public kernel tables, modify public kernel IDs, alter bundle packaging, or change source-distribution inputs as part of this campaign.

The two measured shared-down schedules may remain in the research catalog as exact-key parents. Any future public integration is a separate release phase after the full 12-key optimization and final review are complete.

## Recursive Optimization-Exhaustion Review

Before declaring completion, reread this plan, the Q4_K HIP source and normalized ISA, all forward artifacts and timing reports, the dense-forward HIP record, completed GGTensile backward records, grouped histories, lower bounds, rejected candidates, `~/rdna35-isa-markdown/`, AMD LLVM definitions/tests, and relevant CK/TensileLite material.

Classify every remaining idea as retained and measured; rejected by correctness, resources, timing, or reproducibility; contract-incompatible or deferred with a prerequisite; or actionable with a target key and measurement gate. An actionable idea must be implemented and measured, then the full review repeated from the new premise. Completion is allowed only after a fresh review finds no actionable in-contract mechanism, every selected key beats HIP, and the residual bottleneck is quantified.

This rule applies to kernel optimization only. Public API integration is not a completion criterion for the current phase and must remain deferred.

## Completion Record

- [x] Forward identity, fixed-quantizer control runtime, inspection, and exact 12-key inventory.
- [x] Correct Q4_K signed-int8 WMMA control for K512/K2048/K4096.
- [x] Retained decoded-staged writer with strict resources, mutation coverage, independent reference checks, and byte-identical rebuilds.
- [x] Complete and prequantized lower bounds with residual bottleneck explanation.
- [x] Two exact shared-down research schedules confirmed below HIP.
- [ ] Fresh paired controls and confirmation for all twelve exact keys.
- [ ] Per-key research catalog with every exact key at or below HIP.
- [ ] Weighted complete-call optimization after all exact keys clear HIP.
- [ ] Recursive optimization-exhaustion review with no actionable mechanism remaining.
- [ ] Public dispatch and bundle integration. Deferred to a separate later phase.

## Current Research Record

The current retained identity combines selective weight and metadata LDS-base hoists, short-lived `v_mad_u32_u24` activation addressing, direct bounded six-bit FP16 conversion, eight contiguous clause-backed output-store runs, and incremental output-row addressing. It is generated from `DenseForwardSolution` and is reproducible from independent build roots.

Independent metadata extraction plus exact epilogue scheduling confirmed two research parents:

- Shared-down M8192: independent extraction, `TilesAhead=1`, dependency width 4, priority 2. Fresh complete-call ratio approximately `0.99154x` HIP.
- Shared-down M32768: independent extraction, `TilesAhead=1`, dependency width 2, priority 2. Fresh complete-call ratio approximately `0.98368x` HIP.

Metadata width-8 exposure, upper-field prepacking, B64/B128 metadata stores, early extraction, and scoped priority variants remain measured evidence but did not displace the two selected parents across their tested exact keys.

The retained writer remains non-selected for the other ten keys until fresh complete-call confirmation establishes a stable HIP win. All transformed assembly, lower-bound sources, post-processed candidates, profiler output, and one-off timing screens remain diagnostic unless represented by a generated strict solution identity and revalidated through the gates above.
