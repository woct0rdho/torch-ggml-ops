# GGTensile MMQ Forward Q4_K Plan

## Purpose

Continue the strict gfx1151 wave32 GGTensile assembly optimization campaign for Q4_K forward. The immediate objective is kernel performance only: close the remaining exact-key gaps against the existing HIP kernels, then optimize the complete production mix. Public dispatch, generated bundle tables, source distribution, and API integration are outside the current phase and must not be changed by optimization work.

The campaign must stop only after a recursive review finds no new valid in-contract optimization mechanism. The final review rule remains mandatory even after all currently open keys reach HIP parity.

The existing HIP Q8_1 activation quantizer is fixed producer infrastructure. It is launched before both the HIP and GGTensile multiply controls and is not a forward tuning knob.

## Contract

Target only:
- gfx1151, wave32, WMMA V1, and BF16 input/output activations.
- Authoritative packed GGUF Q4_K weights with direct in-kernel packed decode.
- The existing HIP Q8_1 workspace with F16_D4S4 metadata and its exact 144-byte block layout.
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

Q4_K has 256 logical values per 144-byte packed block. The fixed Q8_1 F16_D4S4 producer emits each 144-byte workspace block as four standard 32-value Q8_1 subblocks: 128 signed int8 values and four `(d,sum)` FP16 pairs in total. The producer may allocate a padded row stride for the selected tile, but only real rows are launched. Every candidate must use the exact producer stride and must never consume a padded row as a valid result.

## Exact Production Scope

Each ordinary projection runs at physical batches 1, 4, and 16, yielding `M={2048,8192,32768}`. The 12 exact keys are:

| Family | `(N,K)` | Representative tensor | Calls | Current research status |
| --- | ---: | --- | ---: | --- |
| Narrow K/V/shared gate/up | `(512,2048)` | `blk.5.ffn_gate_shexp.weight` | 70 | Common composition on M2048/M8192; exact `a4d4-p2` on M32768 |
| Shared-expert down | `(2048,512)` | `blk.5.ffn_down_shexp.weight` | 30 | Exact epilogues on all M values: `a1d2/a1d4/a1d2-p2` |
| Attention output | `(2048,4096)` | `blk.3.attn_output.weight` | 10 | Common composition on all M values; exact variants were neutral |
| Attention query | `(8192,2048)` | `blk.39.attn_q.weight` | 1 | Exact epilogues on all M values: `a1d2/a4d4/a2d2-p2` |

All 12 keys have generated strict independent-extraction-plus-metadata-after-low research identities. Seven keys additionally use exact resource-neutral epilogues: all shared-down and query sizes plus narrow M32768. These are research selections, not permission to add or modify public API wiring in this phase; installed dispatch remains unchanged.

## Current Baseline and Priority

The retained complete-call 25-repeat snapshots provide the following candidate/HIP ratios. They are suitable for prioritization, but fresh paired rotations are required before promotion because the historical captures span different clock and thermal regimes.

| Family `(N,K)` | M=2048 | M=8192 | M=32768 |
|---|---:|---:|---:|
| Narrow `(512,2048)` | `1.0217x` | `1.0410x` | `1.0038x` |
| Shared-down `(2048,512)` | `1.0282x` | `0.9915x` selected | `0.9837x` selected |
| Attention-output `(2048,4096)` | `1.0108x` | `1.0057x` | `0.9989x` |
| Query `(8192,2048)` | `1.0038x` | `1.0075x` | `0.9973x` |

Fresh native independent-extraction-plus-metadata-after-low strict complete-call rotations supersede that priority snapshot:

| Family `(N,K)` | M=2048 A/B | M=8192 A/B | M=32768 A/B |
|---|---:|---:|---:|
| Narrow `(512,2048)` | `0.9972/0.9873x` | `0.9874/0.9892x` | `0.9807/0.9786x` |
| Shared-down `(2048,512)` | `0.9884/0.9900x` | `0.9860/0.9864x` common | `0.9834/0.9813x` common |
| Attention-output `(2048,4096)` | `0.9664/0.9687x` | `0.9748/0.9716x` | `0.9793/0.9782x` |
| Query `(8192,2048)` | `0.9729/0.9706x` | `0.9744/0.9706x` | `0.9752/0.9755x` |

Every common exact key beats HIP in two strict complete-call rotations. Seven exact epilogues improve the common identity without changing the `239 VGPR / 16 SGPR / 38,400 LDS` envelope. Native strict complete screens remain below HIP for shared-down M2048 (`0.96895x`), narrow M32768 (`0.97318x`), query M2048 (`0.96411x`), and query M8192 (`0.97737x`); the previously selected shared-down M8192/M32768 and query M32768 schedules remain at `0.96883x`, `0.96535x`, and `0.96816x` HIP in their strict confirmations.

Large-margin work now prioritizes changed-premise decode/WMMA scheduling and family-specific epilogue composition. Attention-output and query have the largest absolute single-call bodies; narrow retains the largest call weighting. The weighted objective remains diagnostic and never authorizes retaining a slower exact key.

## Fixed Activation Producer

The installed HIP `quantize_bf16_q8_1_f16_d4s4` HSACO remains the producer control. For each 32-value group it computes the absolute maximum, `d=amax/127`, signed-int8 nearest quantization, and the input sum needed by Q4_K zero-point correction.

The benchmark must keep three explicit measurements:
- Fixed HIP quantizer plus HIP packed multiply: public HIP control for comparison only.
- Fixed HIP quantizer plus GGTensile packed multiply: complete candidate.
- Prequantized Q8_1 workspace with HIP and GGTensile multiply bodies: diagnosis only.

The produced Q8_1 workspace must be byte-identical for HIP and GGTensile. No alternate rounding, clamp, reciprocal, reduction, metadata layout, workspace lifetime, or fused producer may be selected through this campaign.

## Established Evidence

The retained decoded-staged body uses a `128x64` tile with 128 threads, 239 VGPRs, 16 SGPRs, 38,400 bytes of LDS, 32 static WMMA instructions, four barriers, and eight output-store clauses. It has already passed strict correctness, input mutation, packed-weight mutation, Q8_1-workspace mutation, independent-reference checks, and byte-identical rebuild checks for all 12 keys.

The native `MetadataAfterLowWmma` schedule defers eight independent metadata LDS reads until between the low- and high-half WMMA batches and changes the dependency-safe low wait ladder from `23,21,...,9` to `15,13,...,1`. It preserves all instruction/resource counts and arithmetic order. Across all 12 keys it improved the retained parent by approximately `2.1-5.0%`.

`IndependentExtractionMetadataAfterLowWmma` additionally exposes all Q4_K scale/min fields before conversion. The default `a8d1p0` form improves every metadata-after-low parent by another `0.57-1.60%`, rebuilds byte-identically, and passes two strict complete-call rotations on all 12 keys. Composing the same local-read schedule with the two exact shared-down epilogues adds `2.7-3.4%` over either mechanism alone without resource growth.

Current profiling shows fewer VALU instructions and flat loads than HIP, but higher instruction-fetch waiting and aggregate wait-any cycles. Retained and HIP K512 controls have comparable LDS conflict and L2-hit rates. The residual target is issue/dependency balance and synchronization, not raw packed traffic.

The lower bounds show that store-only work is approximately `24-27%` of retained latency for K512 and `2-7%` for K2048/K4096. Decode/stage/store work is approximately `44-57%`, while memoryless math/control/store work is approximately `86-88%`. These floors are diagnostic and are not additive.

Already rejected or closed mechanisms include the unchanged `64x64`, `128x128`, and `256x64` geometries; flattened workgroup clustering; broad activation prefetch; loop NOP padding; instruction-prefetch descriptor tuning; cross-lane B32/B64/B128 output packing; activation-load and packed-weight clauses; broad metadata byte permutes; broad priority settings; and gfx1151 split barriers. Phase-1 also closed final-block barrier elision as neutral, a single static eight-group loop as only `0.3%` favorable after confirmation, eager dual-activation staging as `10-13%` slower, and a 248-VGPR overlapped activation-plane prototype as incorrect and already `19%` slower. The compiler-shaped assembly oracle remains diagnostic evidence, not a packaged solution.

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

Phase-1 status: metadata-read/WMMA interleaving is retained and generated natively. It is complete for all families and composes with the selected exact epilogues. Dependency-safe after-2, after-4, and after-6 placements were all exact but `0.65-2.97%` slower than issuing metadata after all eight low WMMAs. Moving the independent extraction block into the first packed-payload decode gap was resource-neutral and exact but neutral on K512 and `0.32-0.37%` slower on narrow/attention K2048/K4096. A direct payload-mask VOPD reduction is unavailable on gfx1151 because `V_AND_B32` is Y-only and there is no useful simultaneous X-side operation; the assembler correctly rejects AND/AND pairing. Rolled-loop-only and final-barrier-only variants are also closed; any reopened local-read or loop reduction must bundle a new dependency or pipeline premise.

### Phase 2: Operand-buffer pipelines

Test only buffer arrangements that fit the gfx1151 LDS and register contract:
- A second 18,432-byte Q8_1 activation plane raises the current allocation from 38,400 to 56,832 bytes. Compare eager two-plane staging with a schedule that stages the second plane while the first four groups compute.
- A second decoded-weight buffer raises the current allocation to approximately 57,856 bytes. For K2048 and K4096, test bounded next-Q4-block packed reads and decode into the alternate buffer while the current block is consumed.
- Start with one next-block vector or one next-plane chunk. Expand only when inspection proves the candidate remains below 256 VGPRs with no spills or private bytes.
- Treat PGR1-style overlap as the first candidate. Do not attempt PGR2 until a preceding schedule creates enough register lifetime headroom.

The expected winning shape is a generated software pipeline with fewer barriers and hidden next-block staging, not simply more preloaded data. K512 M2048 is still important because it has a large store/stage fraction, but K2048/K4096 receive priority for next-block overlap.

Phase-2 status: all tested activation double-buffer forms are closed. Eager staging loses overlap, the bounded 248-VGPR prototype loses performance before correctness repair would matter, and a sequential split-plane form that preserved staging order while removing one overwrite barrier was exact but `12-14%` slower because 56,832-byte LDS residency dominates. A resource-neutral group-7 prefetch reused dead `v112:v139` registers for the next packed payload, but was `0.13-0.28%` slower on narrow M8192, attention-output M8192, and query M32768; packed payload VMEM is already hidden well enough that duplicated address/branch issue loses. Continue only with decoded-weight overlap that also removes current decode work, or with a changed register/address premise; do not retry unchanged Q8_1 or payload-only prefetch.

### Phase 3: Geometry and ownership

Do not repeat rejected geometries without a changed implementation premise. Add a narrow-specific compact bracket:
- `128x32` ownership with 64 or 128 threads, using reduced decoded-weight LDS and a strict wave mapping.
- `64x32` ownership only if the `128x32` body demonstrates a resource or residency benefit.
- `DepthU=64` only as a bundled local-read and group-loop schedule, not as an isolated parameter change.
- WGM values `{1,2,4}` only after a new geometry wins; workgroup mapping and macro-tile shape must be measured together.

For attention-output and query, retain `128x64` unless a structural pipeline produces a clear resource-neutral improvement. Do not trade the established four-wave ownership for a larger tile merely to reduce instruction count.

Phase-3 status: the changed-premise narrow `128x32` bracket was implemented with 64 threads, 28,672-byte LDS, 32 decoded weight rows, and two activation staging passes. It was bit-exact and resource-clean but `21.6%` slower than `128x64` on priority narrow M8192; duplicated activation staging dominates, so unchanged `128x32` and conditional `64x32` are closed.

### Phase 4: Family-specific tuning

- Narrow: optimize M8192 first, transfer only measured mechanisms to M32768 and M2048, then confirm all three exact keys.
- Shared-down: focus M2048 on loop/barrier reduction and conversion/store overlap. Reuse the selected M8192/M32768 extraction identities only as parent controls; do not generalize independent extraction without fresh exact-key evidence.
- Attention-output: focus on main-loop issue balance and bounded next-block prefetch because the store floor is small.
- Query: confirm M32768, then address M8192 and M2048 only with mechanisms that improve the complete call, not just the multiply body.

After a family-specific candidate clears confirmation, retime the other two M values before treating the mechanism as transferable.

Phase-4 status: the exact epilogue sweep is complete. Attention-output candidates are neutral across M; the best M2048 result fell from `0.37%` to `0.13%` in independent confirmations. Narrow M8192 `a4d1-p0` is exactly neutral in a direct complete rotation, and narrow M2048 candidates remain below the promotion margin. Narrow M32768 `a4d4-p2` confirms `0.41-0.78%` multiply gain and a smaller but repeatable complete-call gain, so it is retained. Shared-down selects M2048 `a1d2-p2`, M8192 `a1d4-p2`, and M32768 `a1d2-p2`. Query selects M2048 `a1d2-p2`, M8192 `a4d4-p2`, and M32768 `a2d2-p2`; each cleared two 25-repeat multiply rotations, complete-call timing, strict mutation checks, native source continuity, and byte-identical rebuilds.

## TensileLite Knob Mapping

TensileLite concepts are research guidance and require real alternate emitters in the custom writer. They are not valid merely because a field exists in a YAML file.
- `MacroTile`, `MIWaveTile`, and `WorkGroup`: map to the new narrow compact ownership bracket and any changed wave-to-output mapping.
- `DepthU`: test 32 versus 64 only with matching group-loop, LDS, and register schedules.
- `PrefetchGlobalRead`: implement bounded next-block Q4_K or Q8_1 overlap; start at one tile.
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
- Run baseline, input mutation, packed-weight mutation, Q8_1-workspace mutation, finite-output checks, and independent GGUF-dequantized BF16 reference checks.
- Preserve reduced-K, one-hot, zero, positive/negative maximum, scale/sum, packed payload, metadata, block-boundary, tile-boundary, and output row/column boundary fixtures.
- Require no private bytes, spills, scratch instructions, calls, or dynamic stack. A candidate that changes the producer or consumes a prepared representation is rejected.

Nine-repeat screens are exploratory only. A resource-bearing mechanism requires a stable gain above 2% in rotating 25-repeat confirmation. Unconditional instruction reductions and resource-neutral scheduling improvements may be retained when neutral or consistently favorable.

Final selection requires at least two independent rotating 25-repeat confirmations against HIP and the retained parent, with the complete fixed-quantizer call below HIP and no material regression in either rotation. A weighted average never authorizes retaining a slower exact key.

## Research and Public-API Boundary

This phase produces research artifacts, immutable timing records, strict solution identities, and optimization evidence only. Do not add kernels to public dispatch, regenerate public kernel tables, modify public kernel IDs, alter bundle packaging, or change source-distribution inputs as part of this campaign.

The measured common identity, extraction-only controls, and seven exact epilogue schedules may remain in the research catalog. Any future public integration is a separate release phase after this completed 12-key optimization and review.

## Recursive Optimization-Exhaustion Review

Before declaring completion, reread this plan, the Q4_K HIP source and normalized ISA, all forward artifacts and timing reports, the MMQ forward HIP record, completed GGTensile backward records, grouped histories, lower bounds, rejected candidates, `~/rdna35-isa-markdown/`, AMD LLVM definitions/tests, and relevant CK/TensileLite material.

Classify every remaining idea as retained and measured; rejected by correctness, resources, timing, or reproducibility; contract-incompatible or deferred with a prerequisite; or actionable with a target key and measurement gate. An actionable idea must be implemented and measured, then the full review repeated from the new premise. Completion is allowed only after a fresh review finds no actionable in-contract mechanism, every selected key beats HIP, and the residual bottleneck is quantified.

This rule applies to kernel optimization only. Public API integration is not a completion criterion for the current phase and must remain deferred.

## Reopened Compact-LDS Campaign

The prior review closed schedules around the fully decoded 304-byte LDS weight row, but it did not test a hybrid row that keeps the packed Q4 payload raw while preserving cooperative exact metadata preparation. This is a changed representation, ownership, and residency premise rather than another schedule or compact output geometry.

For the retained `128x64`, 128-thread ownership, add one strict hybrid row layout:
- 128 bytes of raw Q4 payload.
- 32 bytes containing the same eight exact packed FP16 scale/minimum pairs produced by the retained decoded path.
- 160 bytes per weight row and 10,240 bytes for 64 rows.
- the unchanged 18,432-byte Q8_1 `F16_D4S4` activation plane at LDS offset 0, followed immediately by weight row `r` at `18,432 + 160*r`, for 28,672 bytes total LDS and no leading gap.

The workgroup must still load each packed weight block cooperatively. Metadata is converted once by its producer lanes and read from LDS by the owning consumers. Raw nibbles are read from LDS and expanded immediately before the existing integer WMMA sequence. No consumer may duplicate another wave's payload decode, and scale/minimum conversion, integer accumulation, FP32 correction order, BF16 rounding, activation staging, four barriers, and exact epilogue identity remain unchanged.

The target crosses the gfx1151 LDS residency boundary while preserving productive ownership: 38,400 bytes permits three four-wave workgroups in 128 KiB WGP LDS, while 28,672 bytes permits four. The current 239 VGPR declaration rounds to the 240-register wave32 allocation class and already permits six waves per SIMD, so the candidate must keep `NumVgpr <= 239`; a minor VGPR reduction without the LDS transition is not this experiment. WGP mode is the primary control. CU mode is compared only after a correct compact body exists and only when inspection proves the same resident-workgroup class.

The old `HipStagedBatch8` artifact at 26,624-byte LDS does not close this premise. It kept metadata in per-consumer global/register paths and emitted the older unrolled body with 128 static WMMAs and 3,000 static VALU issues; its approximately `0.252 ms` multiply versus `0.174 ms` HIP demonstrates that occupancy alone is insufficient. The new candidate must compose compact payload staging with cooperative decoded metadata, the retained rolled group loops, metadata-after-low placement, and the selected epilogue.

Falsify the premise first on exact narrow `(2048,512,2048)`:
- generate a serialized correctness control and at most two dependency-safe placements that decode raw nibbles after the LDS read while independent activation or metadata reads are outstanding.
- require bit-exact HIP output, the independent reference, all three mutation checks, exact 40-byte ABI, 28,672-byte LDS, 32 static WMMAs, four barriers, zero private storage or spills, and byte-identical rebuilds.
- compare exact-trip-normalized dynamic ISA rather than unrolled static counts; global packed bytes must remain unchanged, while payload LDS writes and reads must each fall by 8 KiB per staged 64-row Q4 tile.
- use a nine-repeat multiply screen only for pruning. Stop this Q4 premise if both dependency-safe placements remain more than 5% slower than the retained body or inspection shows duplicated decode or failure to realize the four-workgroup class.
- advance only a candidate with a stable greater-than-2% gain to two independent rotating 25-repeat confirmations against both the retained parent and HIP, then retest every exact key affected by its strict identity.

### Compact-LDS result and remaining bottleneck

The compact representation was implemented with serialized consumer decode, decode after activation LDS reads, and a corrected pair-reuse schedule. The first two controls reread each 32-byte packed pair for its low- and high-nibble groups and therefore did not satisfy the planned LDS-read reduction; pair reuse retained the raw pair across both groups and is the traffic-correct falsification. All three forms are bit-exact to HIP and the independent reference, finite, sensitive to input/packed-weight/workspace mutations, and resource-clean.

| Exact narrow `(2048,512,2048)` | Median ms | Candidate/parent | Candidate/HIP |
| --- | ---: | ---: | ---: |
| HIP | `0.179463` | - | `1.00000x` |
| Retained decoded parent | `0.175113` | `1.00000x` | `0.97576x` |
| Serialized compact control | `0.206893` | `1.18148x` | `1.15284x` |
| Activation-read overlap | `0.203853` | `1.16412x` | `1.13591x` |
| Traffic-correct pair reuse | `0.204072` | `1.16537x` | `1.13713x` |

The pair-reuse body has 239 VGPRs, 16 SGPRs, 28,672-byte LDS, 32 static WMMAs, four barriers, and zero private storage or spills, so it realizes the intended three-to-four-workgroup transition. It also reduces raw-payload LDS writes and exact-trip reads by 8 KiB per staged 64-row tile. The remaining bottleneck is consumer decode issue and dependency depth: exact-trip normalization replaces 48 producer-side nibble operations per wave/block with 128 operations in the rolled consumer loop, a net 80 vector operations on the WMMA critical path. The fourth resident workgroup cannot hide that work, and both dependency-safe placements exceed the 5% stop gate by more than three times.

The compact-LDS premise is rejected and its unselected implementation is removed. The existing selected catalog remains authoritative. A fresh recursive review after this rejection finds no remaining actionable in-contract Q4_K mechanism; another attempt requires a new instruction, representation, or ownership premise that removes consumer decode rather than rescheduling it. Public integration remains deferred.

## Completion Record

- Forward identity, fixed-quantizer control runtime, inspection, and exact 12-key inventory.
- Correct Q4_K signed-int8 WMMA control for K512/K2048/K4096.
- Retained decoded-staged writer with strict resources, mutation coverage, independent reference checks, and byte-identical rebuilds.
- Complete and prequantized lower bounds with residual bottleneck explanation.
- Seven exact resource-neutral epilogue schedules confirmed below HIP.
- Fresh paired controls and confirmation for all twelve exact keys.
- Per-key research identities with every exact key below HIP in two strict complete-call rotations.
- Weighted complete-call optimization after all exact keys clear HIP.
- Compact-LDS representation measured and rejected; fresh recursive optimization-exhaustion review with no actionable mechanism remaining.
- Public dispatch and bundle integration. Deferred to a separate later phase.

## Current Research Record

The current common research identity combines selective weight and metadata LDS-base hoists, short-lived `v_mad_u32_u24` activation addressing, independent scale/min extraction, metadata LDS reads between low/high WMMA batches, eight contiguous clause-backed output-store runs, and incremental output-row addressing. `ForwardSolution` represents it as `MetadataSchedule=IndependentExtractionMetadataAfterLowWmma` with the default `a8d1p0` epilogue; independent build roots produce byte-identical code objects.

### Final multiply result

The table uses only the prequantized HIP and GGTensile multiply bodies. Both consume the same workspace from the same fixed `torch_ggml_ops_mmq_gfx1151_v1_quantize_bf16_q8_1_f16_d4s4` kernel; activation quantization is excluded from every throughput and speedup below. Logical throughput is `2*M*N*K/(median_ms*1e9)`. Speedup is `HIP median time / GGTensile median time`, so values above `1.0x` favor GGTensile. Each value combines the two independent rotating 25-repeat confirmations by averaging their median times; the largest per-key A/B speed difference was `0.65` percentage points.

| Family | `(M,N,K)` | Selected schedule | HIP TFLOPS | GGTensile TFLOPS | Speedup vs HIP |
| --- | ---: | --- | ---: | ---: | ---: |
| Narrow | `(2048,512,2048)` | common `a8d1-p0` | `23.899` | `24.225` | `1.0136x` |
| Narrow | `(8192,512,2048)` | common `a8d1-p0` | `27.946` | `28.621` | `1.0241x` |
| Narrow | `(32768,512,2048)` | `a4d4-p2` | `28.184` | `28.974` | `1.0280x` |
| Shared down | `(2048,2048,512)` | `a1d2-p2` | `24.514` | `25.144` | `1.0257x` |
| Shared down | `(8192,2048,512)` | `a1d4-p2` | `26.997` | `27.879` | `1.0327x` |
| Shared down | `(32768,2048,512)` | `a1d2-p2` | `27.120` | `28.104` | `1.0363x` |
| Attention output | `(2048,2048,4096)` | common `a8d1-p0` | `28.135` | `29.050` | `1.0325x` |
| Attention output | `(8192,2048,4096)` | common `a8d1-p0` | `28.328` | `29.121` | `1.0280x` |
| Attention output | `(32768,2048,4096)` | common `a8d1-p0` | `28.507` | `29.224` | `1.0252x` |
| Query | `(2048,8192,2048)` | `a1d2-p2` | `28.087` | `29.104` | `1.0362x` |
| Query | `(8192,8192,2048)` | `a4d4-p2` | `28.252` | `29.287` | `1.0366x` |
| Query | `(32768,8192,2048)` | `a2d2-p2` | `28.263` | `29.185` | `1.0326x` |

#### Same-Producer Complete-Call Diagnostic

The separate complete-call audit used one loaded F16_D4S4 producer instance and one workspace for each HIP/GGTensile pair. Multiply and complete phases were timed separately with sustained batches, alternating order, and two reversed 25-repeat passes. The table reports `HIP complete median / GGTensile complete median`; no complete-call value contributes to the final multiply table above.

| Family | M2048 complete A/B | M8192 complete A/B | M32768 complete A/B |
| --- | ---: | ---: | ---: |
| Narrow | `1.0091x/1.0116x` | `1.0160x/1.0165x` | `1.0225x/1.0228x` |
| Shared down | `1.0296x/0.9685x` | `1.0178x/1.0174x` | `1.0343x/1.0341x` |
| Attention output | `1.0226x/1.0146x` | `1.0287x/1.0273x` | `1.0268x/1.0264x` |
| Query | `1.0210x/1.0217x` | `1.0396x/1.0410x` | `1.0388x/1.0376x` |

No Q4 family has the broad reduction seen in low-N Q8. Shared-down M8192 showed the largest repeatable initial-pass dilution, with within-audit reductions of `0.99` and `0.57` speedup percentage points. Shared-down M2048 is timing-context sensitive: the initial complete ratios split at `1.0296x/0.9685x`, and two longer 51-repeat confirmations measured `0.9855x/0.9745x`; it must be treated as complete-call parity with unresolved direction, not as a precise end-to-end speedup. Narrow M2048 longer confirmations remained faster at `1.0171x/1.0219x`. Full evidence is in `~/tmp/torch-ggml-ops/fwd-complete-audit-q4-{a,b}.json` and `fwd-complete-audit-q4-short-{c,d}.json`.

Independent metadata extraction and the local-read schedule compose with the seven exact schedules shown above: all shared-down and query sizes plus narrow M32768.

All common and exact-epilogue compositions are bit-exact to HIP and retain 239 VGPRs, 16 SGPRs, 38,400-byte LDS, 32 static WMMAs, four barriers, eight store clauses, and zero private storage or spills. Strict independent-reference NRMSE remains approximately `0.0133-0.0138`; independent rebuilds are byte-identical; input, packed-weight, and workspace mutations all change outputs.

The prior recursive review found no remaining actionable in-contract mechanism under the fully decoded LDS premise. Resource-neutral local-read, extraction, loop, barrier, payload-prefetch, metadata-overlap, priority, and epilogue neighborhoods are measured. Larger activation or decoded-weight ping-pong allocations inherit the measured 56-KiB LDS residency loss and cannot plausibly recover it without a new reduced-LDS geometry; the implemented `128x32` premise was already `21.6%` slower. Gfx1151 lacks direct-to-LDS, instruction-prefetch, split-barrier, and useful payload-mask VOPD forms. Remaining sub-`0.5%` epilogue screens either collapsed in confirmation or were diluted below repeatability in the complete call.

The compact raw-payload plus decoded-metadata row above is the materially new layout and ownership premise required by that review. It preserves `128x64` output ownership rather than repeating `128x32`, and its pending status prevents a new exhaustion claim until it is measured and the recursive review is repeated.

Transformed assembly, lower-bound sources, rejected buffering prototypes, profiler output, and one-off timing screens remain diagnostic unless represented by a generated strict solution identity and revalidated through the gates above. Public dispatch and bundle integration remain untouched.

## Cross-Campaign Reopening Review

The prior compact-LDS rejection remains valid for the fully decoded Q4_K body: reducing the row representation without removing consumer nibble expansion moved work onto the WMMA critical path and exceeded the stop gate. A new current-parent register and lifetime review nevertheless found a separate resource-neutral address premise that was not covered by the historical closure.

### Activation-base lifetime experiment

The historical `v88` assumption does not describe the current writer. In the current Q4 output body, `v88` is an active `v_bfe_u32` extraction destination and cannot be reused. A first current-parent transform instead kept the final metadata base in `v232` across the decode stage. That artifact produced NaNs because Q4 decode clobbers `v232` before the group loop; it is rejected as a lifetime error, despite passing static resource inspection.

The corrected diagnostic transform reads the invariant wave predicate into `s12`, preserves the per-block metadata-base recomputation, and reuses `v236` for the persistent activation LDS base. It removes the rolled activation-base MADs without changing the Q8_1 `F16_D4S4` producer, arithmetic order, ABI, synchronization, or output mapping. The four representative artifacts were exact to HIP and the retained parent, finite, mutation-sensitive, and within the independent-reference envelope. Each remains at `239 VGPR / 16 SGPR / 38,400 LDS`, with 32 WMMAs, four barriers, zero private storage, and zero spills; static VALU issue count falls by one.

| Exact representative | Dynamic activation MADs saved per launch | Paired candidate/parent median A/B |
| --- | ---: | ---: |
| Attention output `(8192,2048,4096)` | `255` | `0.99617x / 0.99574x` |
| Shared down `(8192,2048,512)` | `31` | `0.99903x / 0.99893x` |

The result is promising as a derived lowering change, not as a new solution knob. The measured movement is sub-percent and the artifacts are assembly diagnostics, so no catalog or public selection changes follow from this screen. The next implementation premise is a typed `DecodedWeightLdsRegisterPlan` lifetime alias, followed by writer-line coverage, deterministic rebuilds, and independent exact-key qualification across every affected Q4 identity.

### Conditional loop-form follow-up

The earlier single eight-group loop was only approximately `0.3%` favorable after confirmation and was closed under an older parent. A current-parent composition with the typed activation-base lifetime is a bounded reopening because both changes remove address or loop-control issue without changing resources. If pursued, `GroupLoopForm={TwoByFour,OneByEight}` must be a real linked typed lowering; a post-emission text rewrite is not sufficient. Q5 loop results do not transfer automatically.

The follow-up was implemented as a linked typed `OneByEight` lowering and screened against each exact selected M8192 parent. The candidate dynamically executes the same eight groups, preserves the midpoint activation restage and synchronization, and is bit-exact to both the parent and installed HIP. It remains at `239 VGPR / 16 SGPR / 38,400 LDS`, with zero private storage and spills. Reusing one body reduces the static counts from 32 to 16 WMMAs, four to three barriers, 858 to 617 VALU issues, 141 to 105 VMEM instructions, and 104 to 60 LDS instructions; those reductions describe code footprint, not dynamic work.

| Exact representative | Parent ms | One-by-eight ms | Candidate/parent |
| --- | ---: | ---: | ---: |
| Attention output `(8192,2048,4096)` | `4.780266` | `4.779639` | `0.99987x` |
| Shared down `(8192,2048,512)` | `0.849491` | `0.845782` | `0.99563x` |

These serialized nine-repeat screens reproduce the historical noise-scale result rather than establish a changed-premise gain. Attention output is effectively flat, and the sub-percent shared-down movement is below the confirmation threshold and inside the observed run variance. Broader exact-key qualification is not warranted; the unselected loop-form field and lowering are removed, and `TwoByFour` remains the only production form. Reopening this direction again requires a premise that reduces dynamic work or creates measurable overlap, not another static-body composition.

The recursive final-review rule remains global. It covers forward and backward records, all supported quant types, and every exact shape, and it may reopen a locally complete Q4 record when another direction or format supplies a valid changed premise. No contract, producer, catalog, public bundle, or dispatch change is authorized by this review entry.
