# GGTensile MMQ Forward Q6_K Plan

## Purpose

Build and exhaust a strict gfx1151 wave32 GGTensile assembly campaign for Q6_K forward over the three exact production language-model-head chunks. The first objective is to make every exact complete call at least as fast as the installed HIP kernel. After exact-key parity, optimize the complete 2,048-row forward mix as far as repeatable evidence permits without regressing any production key.

Q6_K is a separate forward backend from Q4_K and Q5_K. It may reuse proven launch, WMMA, synchronization, output, inspection, correctness, and measurement mechanisms only when their ownership and physical layouts remain valid. It must own its 210-byte packed block, split low/high payload decode, signed scale fields, block multiplier, `Q8_1` `F32_D4` activation metadata, small-M geometry, and exact tuning identities.

Public dispatch, generated bundle tables, source packaging, and producer fusion remain outside the research phase.

## Contract

Target only:
- gfx1151, wave32, WMMA V1, BF16 input/output, and FP32 correction/accumulation.
- Authoritative packed GGUF Q6_K weights with direct in-kernel decode.
- The installed HIP Q8_1 `F32_D4` activation producer and exact 144-byte workspace block.
- Exact 40-byte MMQ forward kernarg ABI.
- One exact `ProblemType`, `ProblemSize`, and complete `ForwardSolution` per artifact.
- Exact-key launch geometry with strict rejection for every mismatch.
- Zero private bytes, spills, scratch instructions, calls, and dynamic stack.
- Serial warmed rotating-control timing. Builds and independent correctness work may run separately; timed GPU work never overlaps.
- No prepared weights, dense shadows, external decode workspaces, split-K, Stream-K, persistent or grouped workgroups, online tuning, public dispatch changes, or producer fusion.

Forward coordinates are:

```text
M = token chunk rows
N = vocabulary rows
K = hidden width

output[M,N] = input[M,K] @ dequant_q6_k(weight[N,K]).T
```

A Q6_K block contains 256 logical values in 210 bytes:

```text
ql[128]      low four bits
qh[64]       high two bits
scales[16]   signed int8 scale per 16 values
d             FP16 block multiplier
```

For logical value `i`:

```text
q = low4(i) | (high2(i) << 4)
value = fp16(d) * float(int8(scale[i/16])) * float(q - 32)
```

The installed forward arithmetic performs integer WMMA, multiplies the integer accumulator by the signed Q6 scale, converts through the existing expression order, and then applies the Q6 block factor and Q8_1 activation scale. The strict campaign preserves that arithmetic and BF16 rounding order. Effective-scale or other reordered floating-point experiments require an explicit changed-correctness phase and are not eligible for exact HIP selection by default.

## Exact Production Scope

The packed Q6_K language-model head has logical weight shape `[248320,2048]` and packed shape `[248320,1680]`. A 2,048-row loss traversal uses:

| Exact key | Calls per 2,048 rows | Installed HIP ownership |
| --- | ---: | --- |
| `(64,248320,2048)` | 32 | exact full J64 |
| `(128,248320,2048)` | 16 | exact full J128 |
| `(256,248320,2048)` | 8 | two exact full J128 row tiles |

M256 is the first absolute-time priority because it is the selected complete-loss chunk. M128 and M64 remain required exact fallback keys and may need distinct geometry or schedules. Weighted totals use the call counts above for prioritization only; every exact complete call must independently clear HIP in both promotion confirmations.

## Correctness and Reproducibility Gates

Before timing, every candidate must pass:
- finite output.
- candidate/HIP normalized RMSE `<= 5e-4`.
- candidate/HIP maximum absolute error `<= 0.015625`.
- independent dequantized-reference normalized RMSE `<= 0.04`.
- input mutation sensitivity.
- packed-weight mutation sensitivity.
- Q8_1 workspace mutation sensitivity.
- exact packed byte count and `F32_D4` workspace shape.
- static inspection proving the 40-byte ABI, gfx1151 wave32 metadata, expected WMMA count, and zero private storage, spills, scratch, calls, or dynamic stack.
- byte-identical independent rebuilds of every selected artifact.

Every executable assembly-writer branch and every emitted line must have unit coverage. Q4_K and Q5_K regression paths remain mandatory while Q6_K branches are added.

## Measurement and Promotion

- Nine-repeat screens prune candidates only.
- Promotion requires two independent warmed rotating 25-repeat confirmations against both HIP and the retained parent.
- Measure fixed-quantizer plus multiply complete calls and prequantized multiply bodies separately.
- Every exact complete call must be no slower than HIP in both confirmations.
- The final result table must report each exact shape's prequantized HIP and GGTensile multiply TFLOPS plus multiplicative speedup `HIP median time / GGTensile median time`; quantization plus multiply is diagnostic only.
- Resource-bearing mechanisms require a stable gain above 2%.
- Resource-neutral instruction reductions or schedules may be retained when neutral or consistently favorable, but may not materially regress another selected key.
- Preserve timing, correctness, inspection, generated source, and code-object artifacts under `~/tmp/torch-ggml-ops/ggtensile-fwd-q6-k/`.

## Initial Evidence

A fresh warmed 25-repeat public HIP baseline for `output.weight` produced:

| M | Packed complete call | BF16 control | Packed/BF16 |
| ---: | ---: | ---: | ---: |
| 64 | `4.050 ms` | `8.708 ms` | `0.465x` |
| 128 | `7.626 ms` | `12.606 ms` | `0.605x` |
| 256 | `15.203 ms` | `13.145 ms` | `1.157x` |

Independent-reference NRMSE was `0.006315`, `0.006214`, and `0.005784` for M64, M128, and M256. M64/M128 strongly beat BF16, while M256 remains slower than isolated persistent BF16 GEMM but is still the production complete-loss chunk because end-to-end selection also includes in-place cross-entropy and packed backward.

The project-owned HIP exact bodies use 158 VGPRs and 28,928-byte LDS for J64, and 210 VGPRs and 38,400-byte LDS for J128, with no private storage. Historical normalized ISA shows the J128 Q6 body carries materially more instructions than Q8 and repeatedly multiplies WMMA integer accumulators by signed per-16-value scales. This makes decode ownership, signed-scale application, static issue balance, and small-M geometry the first-order areas.

The completed Q6_K backward campaign provides decoder facts but not a transferable forward solution. Its packed VOPD extraction, scale ownership, and lane-sharing results use transposed output-row ownership and different LDS geometry. Each mechanism must be re-derived under forward J64/J128 ownership before use.

## Optimization Phases

### Phase 0: Strict Q6_K forward infrastructure

- Add Q6_K MMQ forward `ProblemType`, exact three-key inventory, versionless catalog, strict validation, runtime packed-size checks, and launch support.
- Add a sibling fixed HIP Q8_1 `F32_D4` producer launcher rather than overloading the incompatible `F16_D4S4` launcher.
- Add exact installed HIP Q6_K J64/J128 multiply launchers.
- Extend correctness and independent-reference handling to the Q6_K block and `F32_D4` workspace.
- Add line-complete tests for model identity, strict rejection, launch geometry, producer layout, packed size, writer branches, and Q4_K/Q5_K regressions.
- Capture fresh prequantized HIP body and complete-call baselines for all three keys.

### Phase 1: Correct decoded-staged J128 control

Start with M128 and M256 using the proven `128x64`, 128-thread, four-wave ownership only where Q6 physical LDS rows can be represented without repair.
- Load `ql`, `qh`, signed scales, and `d` directly from 210-byte blocks.
- Reconstruct exact six-bit signed values in the LDS order consumed by WMMA.
- Preserve integer-accumulator-times-int8-scale arithmetic before block and activation factors.
- Use Q8_1 `F32_D4` metadata directly; do not emit Q4/Q5 sum correction paths.
- Release decoder temporaries before accumulation where possible.
- Compare a serialized correctness control with a Q6-specific decoded-staged schedule derived from the installed HIP source and ISA.

No timing interpretation is valid until the decoder is bit-exact to HIP, independently correct, mutation-sensitive, and resource-clean.

### Phase 2: Exact J64 small-M control

M64 cannot be represented by the J128 launch without padded-row work or an unsupported partial tile. Add an exact `64x64` output ownership matching the installed J64 premise:
- 128 threads, four wave32 waves unless inspection proves a lower-thread ownership is required.
- 28,928-byte or smaller LDS target.
- exact 64-row launch with no padded output row accepted as valid.
- Q6-specific decoded-weight row ownership and Q8_1 `F32_D4` activation staging.
- strict comparison against both installed J64 HIP and a diagnostic padded-J128 control if one is safe.

Do not generalize J64 to M128/M256 unless it wins those exact complete calls in confirmation.

### Phase 3: Large-margin decoder and scale work

Prioritize M256 absolute time, then transfer only measured mechanisms:
- packed low/high address contraction and immediate offsets.
- vector low/high extraction schedules and in-place high-plane shifts.
- signed-scale load width, lane ownership, and broadcast.
- legal gfx1151 VOPD pairs for adjacent subtracts or scale multiplies.
- decode preparation batching by row or 16-value scale group.
- early temporary release and accumulator initialization.
- WMMA/local-read issue distance and exact epilogue scheduling.
- bounded instruction priority only when tied to a measured dependency phase.

Lane sharing must reduce enough global work to repay EXEC or cross-lane distribution. Backward evidence alone is not sufficient.

### Phase 4: Changed-premise geometry and overlap

The Q4_K/Q5_K review establishes one first-order representation and residency opportunity that must be tested after the correctness-first decoded control. Add a hybrid Q6 LDS row containing:
- 128 raw `ql` bytes.
- 64 raw `qh` bytes.
- a 32-byte exact metadata area for the 16 signed scale bytes and FP16 `d`, without forming an effective floating-point scale or changing expression order.

The 224-byte row uses 14,336 bytes for 64 weight rows. Consumer waves reconstruct only their owned six-bit payload after LDS reads, then preserve the exact sequence of integer WMMA accumulation, signed-int8 scale multiplication, floating conversion, Q6 block factor `d`, and Q8_1 activation scale. Cooperative global loading, signed scale ownership, exact BF16 rounding, and mutation behavior remain mandatory.

Test these exact ownerships. In each layout the activation plane starts at LDS offset 0, weight row `r` starts at `activation_bytes + 224*r`, and there is no leading gap:
- M64 `64x64`, 128 threads and four waves, with 9,216 activation bytes and 23,552 bytes total LDS. This permits five workgroups in 128 KiB WGP LDS, versus four at the installed 28,928-byte J64 allocation.
- M128 `128x64`, 128 threads and four waves, with 18,432 activation bytes and exactly 32,768 bytes total LDS. This permits four workgroups instead of three at 38,400 bytes.
- M256 `256x64`, 256 threads and eight waves, with 36,864 activation bytes and 51,200 bytes total LDS. Two workgroups provide four waves per SIMD while staging and decoding each 64-row weight tile once instead of twice through two J128 row tiles.

M128 is the representation control. M256 is a coupled exact-shape experiment, not a transfer of the rejected Q4_K/Q5_K `256x64` geometry: Q6 has heavier low/high reconstruction and the production M256 call otherwise duplicates J128 weight staging. M64 remains an exact four-wave J64 body; do not reduce it to two waves merely to lower thread count.

For every hybrid candidate:
- preserve packed global bytes, the exact 40-byte ABI, expected WMMA and barrier structure, zero private storage and spills, and byte-identical rebuilds.
- report exact-trip-normalized dynamic ISA. Relative to a 256-byte decoded payload row, Q6 raw payload must remove 4 KiB of LDS writes and 4 KiB of LDS reads per staged 64-row Q6 tile without duplicating decode across consumers.
- keep register allocation within the addressable no-spill envelope and record the 24-VGPR allocation class together with LDS-derived resident workgroups and waves per SIMD.
- compare a serialized correctness form and at most two dependency-safe consumer-decode placements before broader schedule work.
- prune with serial nine-repeat multiply screens. Stop an exact geometry if both placements remain more than 5% slower than its decoded parent or fail to realize the stated resident-workgroup class; do not transfer that result automatically to another M.
- require a stable greater-than-2% parent gain before two independent 25-repeat confirmations against installed HIP and the retained parent.

WGP mode is the primary launch control. Compare CU mode only after a compact candidate is correct and only where two workgroups fit in each 64 KiB CU half without lowering residency. Mode changes are not an independent campaign on the 38,400-byte body.

After the hybrid representation is decided, the remaining conditional experiments are one bounded next-block packed prefetch that removes current decode issue rather than merely moving VMEM; a second decoded-weight plane only if it remains below the register/LDS gate; alternate DepthU only as a bundled decode/local-read schedule; and WGM only where M256 retains multiple exact row tiles after geometry selection.

Do not repeat Q4/Q5 activation double buffering, payload-only prefetch, broad geometry sweeps, or rolled-loop forms without a Q6-specific changed premise.

### Phase 5: Exact selection and weighted complete-call closure

- Select one explicit identity per exact M.
- Confirm each exact complete call twice against installed HIP and its retained parent.
- Rebuild selected artifacts independently and compare byte-for-byte.
- Publish the final per-shape result using prequantized multiply throughput and multiplicative speedup rather than complete-call throughput.
- Compute the 32/16/8-call weighted 2,048-row forward total for prioritization, while retaining the exact-key HIP gate.
- Keep public dispatch deferred.

## Recursive Optimization-Exhaustion Review

Before declaring Q6_K complete, reread this plan; the installed Q6_K forward source and normalized gfx1151 ISA; every Q6 forward artifact, timing, lower bound, profile, counter, selected and rejected candidate; the completed Q4_K/Q5_K forward records; the Q6_K backward record; the local HIP optimization history; relevant CK, TensileLite, EvoTensile, hipBLASLt, RDNA3.5 ISA, and AMD LLVM gfx11 scheduling/VOPD evidence.

Classify every remaining idea as:
- retained and measured.
- rejected by correctness, resources, timing, or reproducibility.
- contract-incompatible or deferred with an explicit prerequisite.
- actionable with an exact target and measurement gate.

Every actionable idea must be implemented and measured, then the review repeats from the changed premise. The final review cannot pass in the same iteration that first discovers, implements, or measures an actionable mechanism. Completion requires no remaining actionable in-contract mechanism, two strict complete-call HIP wins for all three exact keys, quantified residual bottlenecks, line-complete writer tests, and byte-identical retained rebuilds.

## Campaign Record

### Plan opened and fresh baseline

- Defined the exact M64/M128/M256 language-model-head inventory and 32/16/8 call weighting.
- Fixed direct packed Q6_K decode, Q8_1 `F32_D4`, exact 210-byte block, 40-byte ABI, and zero-private-resource contracts.
- Recorded fresh public HIP complete-call and BF16 baselines for all three keys.
- Identified M256 as the first absolute-time target and exact J64 support as a required independent M64 path.
- Added the pending raw-payload plus exact-metadata LDS representation, exact J64/J128 residency targets, and the coupled M256 `256x64` weight-reuse experiment.
- Next: implement strict Q6_K forward modeling, sibling `F32_D4` producer launch, installed J64/J128 HIP controls, exact catalogs, and correctness-first decoded-staged controls.

### Decoded-staged controls and M256 shared ownership

The strict forward infrastructure is now implemented for the exact `(64,248320,2048)`, `(128,248320,2048)`, and `(256,248320,2048)` keys. All controls use the fixed HIP-produced Q8_1 `F32_D4` workspace, direct 210-byte packed Q6_K weights, integer WMMA accumulation, signed int8 scale multiplication, Q6 block factor application, and the exact 40-byte ABI. The M256 catalog identity is `decoded_staged_j256_control`; it uses a flat `(256,1,1)` workgroup and shares one decoded 64-row weight tile across eight waves. Waves `0..3` and `4..7` use the same four weight columns, while the upper wave bit selects the second 128-row output half. The wave column is derived from the preserved physical wave register at each use, so the lane register remains available for every K block.

The M256 body initially faulted because upper waves used full wave IDs as N-column ownership. A padded-canary isolation launch showed complete finite writes after changing ownership to `wave & 3` plus the 128-row output-half offset. The unpadded production launch then passed exact HIP comparison over all `63,569,920` BF16 outputs with zero differing elements, input mutation, packed-weight mutation, workspace mutation, finite output, and independent-reference normalized RMSE `0.0060818`. M64 and M128 likewise passed exact HIP comparison over `15,892,480` and `31,784,960` outputs, with independent-reference normalized RMSE `0.0061594` and `0.0061055`.

Current inspected resources are:

| M | Workgroup | VGPR | SGPR | LDS | Static WMMA | Barriers | Static VOPD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 128 | 124 | 16 | 28,672 B | 4 | 4 | 32 |
| 128 | 128 | 208 | 16 | 37,888 B | 8 | 4 | 64 |
| 256 | 256 | 208 | 16 | 56,320 B | 8 | 4 | 64 |

All three have zero private bytes and zero VGPR/SGPR spills. Independent generation, assembly, linking, and inspection reproduced each Q6 source and HSACO byte-for-byte. The restored Q4_K/Q5_K and backward catalog-valid source set remains byte-identical across all `379` sources.

Two independent warmed 25-repeat multiply-only confirmations are now recorded. The medians and multiplicative speedups `HIP median / GGTensile median` were:

| M | Confirmation | HIP ms | GGTensile ms | Speedup | HIP TFLOPS | GGTensile TFLOPS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | A | 4.074592 | 4.858241 | 0.838697 | 15.975979 | 13.399005 |
| 64 | B | 4.102910 | 4.879833 | 0.840789 | 15.865714 | 13.339718 |
| 128 | A | 7.526049 | 8.078001 | 0.931672 | 17.298744 | 16.116759 |
| 128 | B | 7.683649 | 8.262901 | 0.929897 | 16.943928 | 15.756112 |
| 256 | A | 14.906138 | 15.584709 | 0.956459 | 17.468132 | 16.707555 |
| 256 | B | 15.091413 | 15.695733 | 0.961498 | 17.253679 | 16.589374 |

Both confirmations reject performance promotion for all three exact keys. The 32/16/8-call weighted multiply totals were `370.052832 ms` HIP versus `409.389400 ms` GGTensile in confirmation A, and `374.962808 ms` versus `413.926936 ms` in confirmation B, or weighted speedups `0.903914x` and `0.905867x`. The M256 shared-decode ownership saves about `0.56 ms` against two current J128 calls in the isolated comparison, but remains about `4%` behind the installed HIP J128 control.

The measured bottleneck is the packed low/high Q6 decode, signed-scale application, and FP32 epilogue issue path; shared M256 staging removes duplicate weight staging but does not remove the per-output-row scale and arithmetic work. A 32-value-group rewrite was not retained because preserving exact per-group scale lifetimes and accumulation order would require a new register/epilogue regime rather than a bounded schedule edit.

The controls are therefore correctness-qualified and cataloged, but not performance-promoted. `CurrentStatus` remains open for all three exact keys because none reaches HIP parity. The final recursive exhaustion review remains pending until the remaining representation premise is classified and any newly actionable optimization is resolved.

### LDS issue controls and compact raw-payload representation

An instruction-level control paired the Q8_1 FP32 activation-scale reads with `ds_read2st64_b32`, corrected the reordered-read waits, and then also paired each decoded low/high weight store with `ds_write2_b32`. All M64/M128/M256 outputs remained bit-exact to the decoded controls. The combined form reduced static LDS instructions from `73/99/83` to `55/79/71`, but its rotating nine-repeat medians were only `1.016x`, `1.003x`, and `0.998x` of the decoded-parent throughput. This is too small and shape-inconsistent to address the open HIP margins, so the candidate-only schema and lowering were removed. Artifacts remain under `paired-scale-reads/` and `paired-lds-issue/`.

The separate compact regime staged one exact 224-byte row per weight row: 128 raw `ql` bytes, 64 raw `qh` bytes, 16 signed scale bytes, FP32-converted `d`, and padding. Consumer waves reconstructed their owned 16-value signed payload immediately before WMMA while preserving scale ownership and arithmetic order. Both admitted controls were bit-exact to the decoded parent over every BF16 output and had zero private bytes or spills:

| M | VGPR | LDS | Static VMEM/LDS | Parent ms | Compact ms | Parent/compact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 124 | 23,552 B | 76 / 48 | 4.628572 | 5.447662 | 0.849645x |
| 128 | 208 | 32,768 B | 144 / 74 | 7.987497 | 8.688813 | 0.919286x |

M64 crossed from four to five LDS-limited workgroups per WGP but regressed `17.7%`. M128 crossed from three to four workgroups but regressed `8.8%`. Repeating packed reconstruction in every consumer group dominates the cooperative staging and residency gains. M256 was not admitted: compact LDS remains at two workgroups per WGP, its consumer decode cost matches M128 per wave, and the M128 representation control failed the campaign's greater-than-2% transfer gate. The compact premise is therefore rejected, and its experimental model, validation, inspection, and writer surfaces were removed after preserving artifacts under `compact-raw/`.

The later recursive exhaustion review remains pending. A full installed-HIP ownership and schedule transfer is newly actionable because a fixed-LDS assembly control is bit-exact and removes the large M64/M128 gap; this must be resolved before any final stopping review.

An M256 expanded-scale control then replaced eight per-group signed-byte LDS reads with two 128-bit reads from a transposed plane of sign-extended i32 scales. Its 260-byte decoded payload/d rows plus 4,096-byte scale plane used `57,600 B` LDS, retained `208 VGPR`, `16 SGPR`, two-workgroup WGP residency, zero private storage, and zero spills. It was bit-exact over all `63,569,920` outputs. Exact-trip accounting removed 81 LDS instructions per K block after including the additional staging stores, but the nine-repeat median improved only from `15.589506` to `15.505320 ms` (`1.005429x` parent speedup) and remained behind HIP at `14.865567 ms` (`0.958739x`). This does not clear the greater-than-2% resource-bearing gate, so the candidate was not transferred to M64/M128 and its schema/writer surface was removed after preserving `expanded-scale/` artifacts.

### Selected HIP-scheduled exact controls

The large residual margins were closed by transferring the project-owned HIP J64 and J128 instruction bodies into two immutable GGTensile exact-schedule emitters. GGTensile owns the exact solution identity, generated symbol, 40-byte ABI metadata, fixed LDS allocation, build, inspection, and dispatch geometry. The emitters contain no HIP symbol or dynamic-LDS metadata. They use structured `(32,4,1)` work-item IDs; M64 owns one J64 row tile, M128 owns one J128 row tile, and the exact M256 key launches two J128 row tiles with grid Y=2. The previous flat-eight-wave M256 body remains the measured parent but is no longer selected.

All three selected artifacts are bit-exact to HIP and their decoded parents, finite, input/packed-weight/workspace mutation-sensitive, and independently qualified. Independent-reference normalized RMSE remains `0.0061594`, `0.0061055`, and `0.0060818`. Resources are:

| M | Workgroup | Grid Y | VGPR | SGPR | LDS | Static WMMA | Barriers |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | `(32,4,1)` | 1 | 158 | 27 | 28,928 B | 8 | 4 |
| 128 | `(32,4,1)` | 1 | 210 | 27 | 38,400 B | 16 | 4 |
| 256 | `(32,4,1)` | 2 | 210 | 27 | 38,400 B | 16 | 4 |

Every artifact has zero private bytes, dynamic stack, and VGPR/SGPR spills. Independent generation, assembly, linking, and inspection reproduced all selected sources and HSACOs byte-for-byte. The frozen 379-source Q4_K/Q5_K forward and backward inventory also remains byte-identical.

Two independent warmed rotating 25-repeat multiply-only confirmations produced:

| M | Confirmation | HIP ms | Selected ms | HIP/selected | HIP TFLOPS | Selected TFLOPS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | A | 4.063981 | 4.040481 | 1.005816x | 16.017692 | 16.110853 |
| 64 | B | 4.072780 | 4.042761 | 1.007425x | 15.983087 | 16.101768 |
| 128 | A | 7.492860 | 7.445651 | 1.006340x | 17.375368 | 17.485536 |
| 128 | B | 7.493968 | 7.466090 | 1.003734x | 17.372798 | 17.437667 |
| 256 | A | 15.016318 | 14.966780 | 1.003310x | 17.339962 | 17.397356 |
| 256 | B | 15.030686 | 14.983447 | 1.003153x | 17.323387 | 17.378003 |

Using the final per-shape confirmation components, the weighted 32/16/8-call totals were `370.063698 ms` HIP versus `368.160049 ms` selected (`1.005171x`) and `370.477943 ms` versus `368.693367 ms` (`1.004840x`). Relative to the previous decoded controls, the weighted speedups were `1.098865x` and `1.095803x`. All inventory entries are now `selected`; M64 and M128 select their respective schedule identities, while M256 selects the J128 identity without duplicating its solution object.

A later M64-only review removed all 47 compiler `s_delay_alu` hints. The candidate remained bit-exact and improved over the fixed-LDS transferred parent by `1.007340x` and `1.004319x` in two rotating 25-repeat confirmations; it also remained faster than HIP by `1.005816x` and `1.007425x`. The J128 delay-free control was rejected because it regressed. The final J64 emitter records this intentional divergence from the compiler body.

### Structured emitter maintenance

Production generation now composes the Q6_K `rocisa.code.Module` components directly in `kernel_writer_assembly_mmq_fwd.py`, alongside the Q4_K and Q5_K forward lowering. A flat Tensile-style schedule state exposes `MacroTile`, `WorkGroup`, `MatrixInstruction`, `MIWaveGroup`, `MIWaveTile`, `DepthU`, `GlobalReadVectorWidthA/B`, `LocalReadVectorWidth`, `PrefetchGlobalRead`, `PrefetchLocalRead`, `1LDSBuffer`, `ScheduleGlobalRead`, `ScheduleLocalWrite`, `ScheduleIterAlg`, `ClusterLocalRead`, `StoreVectorWidth`, `EpilogueDependencyWidth`, `EpiloguePipelineScope`, and `StoreRemapVectorWidth`, plus the measured dependency-delay and global-read cache policies. Geometry consistency is derived and checked rather than duplicated: the matrix instruction, wave group, and wave tile must produce the selected macro tile; the workgroup must contain the selected wave group; and `DepthU` must match the matrix-instruction K depth and dot phases. The iteration fields are validated as one coupled schedule point rather than independent booleans. A bounded offline candidate enumerator forks only named delay, cache, BF16 dependency-width, and BF16 pipeline-scope choices. Candidate search is not part of generation: one complete schedule state lowers directly and deterministically to one stream.

All Q6 body components lower through a semantic schedule emitter built on the same shared `Assembly` primitive as Q4_K/Q5_K. It constructs work-item mapping, argument loads, 64-bit pointer additions, global-read clauses, global-to-LDS write batches, packed six-bit reconstruction, barriers, local stage commits, generated dot phases, NaN-preserving BF16 conversion, and vector-width-controlled output-store clauses. J64 uses a `1x4` wave tile, no dependency-delay hints, a two-value BF16 dependency window, and `StoreBatch` pipeline scope. J128 uses a `2x4` wave tile, explicit selected dependency-delay hints, the same two-value dependency window, and `FullTile` scope so rounding and NaN preparation can cross store-clause boundaries. The accumulator order, output-index register, scratch-ring base, and store grouping are declarative physical relationships. There are no numbered epilogue batches, absolute issue slots, raw assembly blocks, external instruction data, generic rescheduler, source-order fallback, or build-time HIP/LLVM pass. Compiler output remains only an offline migration, analysis, and benchmarking oracle.

The BF16 rewrite was separately generated, assembled, linked, inspected, correctness-screened, and timed. It preserved zero differing parent elements and the selected 158/210 VGPR, 27 SGPR, 28,928/38,400-byte LDS outcomes with no private storage or spills. Two 25-repeat confirmations retained a repeatable J64 improvement: candidate/parent median latency was `0.996657x` and `0.996203x`. J128 and the M256 two-tile launch were timing parity: candidate/parent medians were `1.002164x`/`0.998585x` and `0.998545x`/`0.997713x`; paired bootstrap intervals included zero for both shapes and both HIP comparisons. This resource-neutral structural promotion therefore keeps the J64 gain while replacing the J128 copied batch schedule without claiming a new material M128/M256 speedup.

The Q6 forward artifact test rebuilds and compares the selected generated source, executable text, and code objects after assembly and linking.

A pre-BF16-maintenance independent rotating confirmation used five warmups and 25 timed repeats per exact key against the same fixed HIP-produced Q8_1 `F32_D4` workspace. The then-selected medians were `3.956011 ms` versus HIP `4.030690 ms` at M64 (`1.018877x` HIP/selected), `7.446487 ms` versus `7.487755 ms` at M128 (`1.005542x`), and `14.844875 ms` versus `14.916552 ms` at M256 (`1.004828x`). Candidate outputs matched the decoded parents at zero differing elements for all three keys. The later BF16 confirmation above qualifies the current structural stream.

### Deterministic schedule-policy reconstruction

Offline compiler-oracle experiments reopened the physical scheduling premise without changing production generation. Saved J64/J128 bitcode was compiled under bounded AMDGPU machine-scheduler policies and repackaged only as temporary benchmark candidates. All screened candidates were output-identical to the retained parent. Exploratory nine-repeat medians found approximately `0.939x` candidate/HIP latency for J64 under the max-ILP policy and `0.955x` for J128 under a top-down policy, versus near-parity retained bodies. Disabling clustering, post-register-allocation scheduling, or the normal allocator was weaker; low-register schedules alone were substantially slower. This is evidence of scheduling headroom, not a production selection.

LLVM remains an offline oracle only. Its max-ILP strategy uses pressure heuristics, clustering, physical-register bias, and an original-node-order fallback, while post scheduling changes VOPD formation. None of those opaque mechanisms may enter GGTensile generation. The actionable work is to express the useful behavior as named stage traversal, load clustering, decode ordering, local-write deferral, BF16 scope, latency spacing, fixed-register pressure, and explicit VOPD policies. Offline search may choose a complete discrete point; generation must directly lower that point without a local optimizer or hidden tie breaker.

### Recursive exhaustion review

The earlier review after the M64 delay-free confirmation closed the fixed transferred-schedule premise. Its retained and rejected classifications remain useful evidence, but the offline scheduler-oracle result creates a new actionable premise and invalidates a final Q6 exhaustion claim until deterministic reconstruction is complete. Current classifications are:
- Retained and measured: fixed-LDS J64/J128 ownership, structured `(32,4,1)` launches, two J128 row tiles for M256, M64 delay-hint removal, and the semantic BF16 pipelines.
- Rejected by timing: paired activation-scale reads and decoded stores, compact raw payload, expanded transposed scales, global-L0 invalidation removal, J128 delay removal, alternate decoded geometries, dedicated decoder waves, no-cluster schedules, disabled post scheduling, and low-register-only policies.
- Correctness or contract-incompatible: NaN-path simplification for arbitrary packed data, reordered effective scales, prepared weights, shared decode workspaces, producer fusion, persistent/grouped traversal, public dispatch changes, and build-time invocation of LLVM scheduling or allocation passes.
- Actionable: deterministic semantic reconstruction of the promising J64 and J128 schedule behavior, followed by exact correctness, M64/M128/M256 resource, and warmed timing qualification.
- Deferred with explicit prerequisites: Q3_K/Q8_0 forward campaigns before any global forward-exhaustion claim.

The selected exact keys remain production-safe while this work proceeds. Q6 performance and structural exhaustion are not final until the new schedule premise is either promoted through all gates or rejected with measured deterministic candidates.
