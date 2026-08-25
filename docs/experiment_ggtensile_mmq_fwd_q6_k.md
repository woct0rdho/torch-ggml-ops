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
d            FP16 block multiplier
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

The controls are therefore correctness-qualified and cataloged, but not performance-promoted. All three exact keys remained open at this checkpoint because none reached HIP parity. The final recursive exhaustion review remained pending until the representation premise was classified and newly actionable optimization was resolved.

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

### Former HIP-scheduled exact controls

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

Production generation now composes the Q6_K `rocisa.code.Module` components directly in `kernel_writer_assembly_mmq_fwd.py`, alongside the Q4_K and Q5_K forward lowering. A flat Tensile-style schedule state exposes `MacroTile`, `WorkGroup`, `MatrixInstruction`, `MIWaveGroup`, `MIWaveTile`, `DepthU`, `GlobalReadVectorWidthA/B`, `LocalReadVectorWidth`, `PrefetchGlobalRead`, `PrefetchLocalRead`, `1LDSBuffer`, `ScheduleGlobalRead`, `ScheduleLocalWrite`, `ScheduleIterAlg`, `ClusterLocalRead`, `StoreVectorWidth`, `EpilogueDependencyWidth`, `EpiloguePipelineScope`, and `StoreRemapVectorWidth`, plus the measured dependency-delay and global-read cache policies. Geometry consistency is derived and checked rather than duplicated: the matrix instruction, wave group, and wave tile must produce the selected macro tile; the workgroup must contain the selected wave group; and `DepthU` must match the matrix-instruction K depth and dot phases. The iteration fields are validated as one coupled schedule point rather than independent booleans. A bounded offline candidate enumerator forks only complete named semantic, delay, cache, BF16 dependency-width, and BF16 pipeline-scope choices. Candidate search is not part of generation: one complete schedule state lowers directly and deterministically to one stream.

All Q6 body components lower through a semantic schedule emitter built on the same shared `Assembly` primitive as Q4_K/Q5_K. It constructs work-item mapping, argument loads, 64-bit pointer additions, global-read clauses, global-to-LDS write batches, packed six-bit reconstruction, barriers, local stage commits, generated dot phases, NaN-preserving BF16 conversion, and vector-width-controlled output-store clauses. J64 uses a `1x4` wave tile, no dependency-delay hints, a two-value BF16 dependency window, and `StoreBatch` pipeline scope. J128 uses a `2x4` wave tile, explicit selected dependency-delay hints, the same two-value dependency window, and `FullTile` scope so rounding and NaN preparation can cross store-clause boundaries. The accumulator order, output-index register, scratch-ring base, and store grouping are declarative physical relationships. There are no numbered epilogue batches, absolute issue slots, raw assembly blocks, external instruction data, generic rescheduler, source-order fallback, or build-time HIP/LLVM pass. Compiler output remains only an offline migration, analysis, and benchmarking oracle.

The BF16 rewrite was separately generated, assembled, linked, inspected, correctness-screened, and timed. It preserved zero differing parent elements and the selected 158/210 VGPR, 27 SGPR, 28,928/38,400-byte LDS outcomes with no private storage or spills. Two 25-repeat confirmations retained a repeatable J64 improvement: candidate/parent median latency was `0.996657x` and `0.996203x`. J128 and the M256 two-tile launch were timing parity: candidate/parent medians were `1.002164x`/`0.998585x` and `0.998545x`/`0.997713x`; paired bootstrap intervals included zero for both shapes and both HIP comparisons. This resource-neutral structural promotion therefore keeps the J64 gain while replacing the J128 copied batch schedule without claiming a new material M128/M256 speedup.

The Q6 forward artifact test rebuilds and compares the selected generated source, executable text, and code objects after assembly and linking.

A pre-BF16-maintenance independent rotating confirmation used five warmups and 25 timed repeats per exact key against the same fixed HIP-produced Q8_1 `F32_D4` workspace. The then-selected medians were `3.956011 ms` versus HIP `4.030690 ms` at M64 (`1.018877x` HIP/selected), `7.446487 ms` versus `7.487755 ms` at M128 (`1.005542x`), and `14.844875 ms` versus `14.916552 ms` at M256 (`1.004828x`). Candidate outputs matched the decoded parents at zero differing elements for all three keys. The later BF16 confirmation above qualifies the current structural stream.

### Deterministic schedule-policy reconstruction

Offline compiler-oracle experiments reopened the physical scheduling premise without changing production generation. Saved J64/J128 bitcode was compiled under bounded AMDGPU machine-scheduler policies and repackaged only as temporary benchmark candidates. All screened candidates were output-identical to the retained parent. Exploratory nine-repeat medians found approximately `0.939x` candidate/HIP latency for J64 under the max-ILP policy and `0.955x` for J128 under a top-down policy, versus near-parity retained bodies. Disabling clustering, post-register-allocation scheduling, or the normal allocator was weaker; low-register schedules alone were substantially slower. This is evidence of scheduling headroom, not a production selection.

LLVM remains an offline oracle only. Its max-ILP strategy uses pressure heuristics, clustering, physical-register bias, and an original-node-order fallback, while post scheduling changes VOPD formation. None of those opaque mechanisms may enter GGTensile generation. The actionable work is to express the useful behavior as named stage traversal, load clustering, decode ordering, local-write deferral, BF16 scope, latency spacing, fixed-register pressure, and explicit VOPD policies. Offline search may choose a complete discrete point; generation must directly lower that point without a local optimizer or hidden tie breaker.

The reconstruction now exposes two complete six-field policies. The existing default remains `OutputRoleGroupMajor`, `StageDependencyOrder`, `SerializedDependencyDistance`, `ExplicitRoleLifetime`, `ProducerFirstUse`, and `DependencyCompatibleDualIssue`. The selected alternative changes the first three fields to `OutputRoleWavefront`, `RowBatchedDecodeOrder`, and `WavefrontDependencyDistance`. Validation, solution serialization, candidate hashing, and bounded neighborhood enumeration accept only the two complete tuples; partial or unknown combinations remain rejected.

The wavefront lowering is hazard-aware and decode-local. It first shifts every destructive packed-QH source, walks the physical high-role handoff in atom order, completes each low merge before that atom's QH register is reused, and then applies the independent signed-normalization frontiers. It uses the existing `Q6DecodeRegisterPlan`, output registers, LDS write order, and stage lifetimes. It adds no scheduler, allocator, register repair, raw instruction table, or compiler pass.

Two independent warmed rotating confirmations produced:

| M | Parent A ms | Wavefront A ms | A/parent | Parent B ms | Wavefront B ms | B/parent |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 3.981262 | 3.930113 | 0.987153x | 4.040238 | 3.959504 | 0.980018x |
| 128 | 7.400774 | 7.349287 | 0.993043x | 7.457979 | 7.359931 | 0.986853x |
| 256 | 14.831588 | 14.701471 | 0.991227x | 15.076605 | 14.915268 | 0.989299x |

Every full exact-shape candidate matched both the selected parent and installed HIP control bit-for-bit, including negated-input, packed-weight-mutation, and Q8_1-workspace-mutation runs over `15,892,480`, `31,784,960`, and `63,569,920` BF16 outputs. All outputs were finite, producer repetition changed zero bytes, and every mutation changed output. Reduced exact correctness checks against independently dequantized BF16 GEMM measured normalized RMSE `0.006142` at M64 and `0.006089` at M128, identical to the public control.

M64 remains at 158 VGPR, 27 SGPR, and 28,928-byte LDS; M128/M256 remain at 210 VGPR, 27 SGPR, and 38,400-byte LDS. All have zero private bytes, spills, scratch instructions, calls, and dynamic stack. Repeated source generation, assembly, and linking reproduced source, object, and HSACO hashes exactly. The strict deployment catalog therefore selects the M64 wavefront specification and shares the M128 wavefront specification between M128 and M256. The default schedule remains implemented and deterministically tested, while public dispatch, generated bundle tables, and installed HIP code remain unchanged.

### Producer-readiness overlap

The oracle-parity continuation first corrected the static comparison to isolate the kernel symbol and stop treating linked device-library bodies as part of the LLVM candidate. The typed and LLVM M64 kernels both contain 107 VMEM operations, 75 LDS operations, eight WMMAs, four barriers, and 13 clauses; M128 contains 175 VMEM operations, 105 LDS operations, 16 WMMAs, four barriers, and 18-19 clauses. The typed body is smaller overall. The useful difference is therefore issue placement and readiness distance, not missing memory or matrix work.

Three temporary deterministic variants targeted the coarse initial `vmcnt(0)`. `wavefront-factor` began decode once the packed payload and Q6 factor were ready, then waited before signed-scale and Q8_1 LDS stores. `wavefront-stream` added monotonic producer-index waits for the existing QH/QL decode frontier and deferred the independent factor conversion. `group-stream` applied the same readiness waits to the default atom-major decode. All variants preserved the physical register map and produced zero differing BF16 elements from the selected parent.

Serial rotating nine-repeat screens rejected the mechanism:

| M | Parent ms | Group stream / parent | Wavefront factor / parent | Wavefront stream / parent |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 3.971232 | 1.002771x | 1.000326x | 1.000344x |
| 128 | 7.418838 | 1.003395x | 0.999692x | 0.998277x |

The largest observed gain was only 0.17% at M128 and M64 did not improve. Fine-grained producer waits alone do not close the approximately 4-5% M64 and 1-2% M128 gaps to the best LLVM machine-scheduler oracle, so no readiness policy or production lowering was added. Artifacts are retained under `q6-readiness-overlap/`.

The oracle trace also showed decoded LDS stores beginning before all later signed-normalization instructions had issued. Three temporary typed variants interleaved the existing B32 signed-add/XOR pairs with their corresponding decoded LDS store, in batches of one, two, or four output roles. This retained the existing `Q6DecodeRegisterPlan`, physical output registers, LDS offsets, and barrier ownership; all candidates matched the wavefront parent at zero differing BF16 elements.

The M64 nine-repeat medians were `1.013435x`, `1.015569x`, and `1.016646x` parent latency for batches one, two, and four. M128 medians were `1.013926x`, `1.009071x`, and `1.008277x`. The store issue does not repay the additional LDS/VALU interleaving in the typed B32 representation, so the candidate lowering was not retained. Artifacts are retained under `q6-decode-store-interleave/`.

### Dot-frontier issue order

The next typed oracle-parity experiment targeted the hot inner dot loop rather than the packed decode. The LLVM trace skews product-scale issue order, carries one late product multiply into an early accumulator update on J64, and places the J128 factor loads before the eight signed scale reads. Temporary lowering-only variants expressed those behaviors as deterministic dot-frontier policies while retaining the selected wavefront physical register map, LDS addresses, WMMA sequence, accumulator ownership, and exact output contract:
- M64 `scale-frontier` reordered the pointer-update/product-scale frontier; `scale-carry` additionally folded the final independent product multiply into a legal existing VOPD accumulator pair and emitted the displaced accumulator update as a scalar FMAC.
- M128 `factor-first` moved factor LDS reads before scale reads; `factor-first-scale` combined that load order with the M128 product-scale frontier; and `factor-first-scale-ready` also removed the now-redundant accumulator LDS wait chain after the scale frontier had reached `lgkmcnt(0)`.

All six candidates assembled and linked. The first six-way screen was accidentally run concurrently for M64 and M128 on the same GPU, so its timing ratios are discarded; its only retained result is the zero-difference correctness check. A serial 12-warmup/9-repeat screen found no material gain:

| M | Parent ms | Best candidate | Best / parent | Candidate family |
| ---: | ---: | ---: | ---: | --- |
| 64 | 3.970453 | 3.955621 | 0.996264x | `scale-carry` |
| 128 | 7.359631 | 7.348888 | 0.998540x | `scale-frontier` |

The serial 20-warmup/25-repeat confirmation reduced both margins further: M64 `scale-carry` was `0.997820x` of parent latency and M128 `scale-frontier` was `0.997778x`. Every candidate produced zero differing BF16 elements from the selected parent. The scale/frontier, carried-multiply, factor-read ordering, and reduced-wait mechanisms are therefore rejected as noise-scale under the fixed physical map. Generated artifacts and the temporary builder remain under `~/tmp/torch-ggml-ops/q6-dot-frontier/` and `~/tmp/torch-ggml-ops/build_q6_dot_frontier.py`.

### Kernel-only issue counters

A kernel-only rocprofv3 pass then loaded the selected typed HSACO and the LLVM max-ILP HSACO under the same validated Q6 launch state, alternated them in one process, and filtered by their distinct kernel symbols. This avoids attributing linked device-library bodies or unrelated producer work to the comparison. Counter collection was run in three serial passes; profiler dispatch times are not used as timing evidence.

| M | Kernel | `SQ_INSTS_VALU` | `SQ_INST_CYCLES_VALU` | `SQ_INSTS_DUAL_VALU_WAVE32` | `SQ_INSTS_LDS` | `SQ_CYCLES` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | typed wavefront | 256,405,920 | 449,335,040 | 76,234,240 | 35,385,600 | 207,837,840 |
| 64 | LLVM max-ILP | 301,336,320 | 494,265,440 | 71,035,040 | 35,385,600 | 210,162,280 |
| 128 | typed wavefront | 469,619,680 | 853,320,640 | 137,972,800 | 49,539,840 | 397,480,380 |
| 128 | LLVM max-ILP | 507,519,520 | 891,220,480 | 139,804,160 | 49,539,840 | 406,050,160 |

The typed and LLVM kernels therefore issue identical LDS volume, identical SALU/SMEM volume, and identical measured LDS-bank-conflict counts. LLVM instead executes more scalar VALU operations and more VALU issue cycles at both shapes; M64 also forms fewer dual-VALU instructions. M128 has a smaller dual-VALU difference, but still spends more VALU cycles. The same-process pass does not identify one universally dominant wait counter - M64 has lower LLVM `SQ_WAIT_INST_LDS`, while M128 has higher LLVM LDS-wait activity - so readiness alone is not a sufficient production policy. The remaining gap is best classified as issue mix and fine-grained address/VALU placement rather than missing LDS work or bank conflicts. Counter artifacts and the profiling helper remain under `~/tmp/torch-ggml-ops/q6-counter-pair-both-j64/`, `~/tmp/torch-ggml-ops/q6-counter-pair-both-j128/`, and `~/tmp/torch-ggml-ops/profile_q6_counter_pair.py`.

### Setup pairing policy

The counter result motivated one bounded VOPD test. The selected typed bodies contain 110 M64 and 182 M128 VOPD instructions, including many setup pairs that combine an independent zero move with address add, shift, or mask work. LLVM retains the dot-loop pairs but forms fewer setup pairs. Two temporary typed policies therefore unpaired only the named `lane_and_address_setup` component: `address` split zero-move/address-operation pairs, while `all` also split eligible setup move/move pairs. Decode, dot, epilogue, physical registers, and every memory operation remained unchanged.

The candidates stayed at 158/210 VGPR, 27 SGPR, 28,928/38,400-byte LDS, and zero private storage or spills. They reduced static VOPD counts to 78/153 for `address` and 78/140 for `all` at M64/M128. Every output matched the selected parent exactly. The nine-repeat screen initially showed small `address` ratios of `0.996724x` at M64 and `0.997046x` at M128, but the serial 20-warmup/25-repeat confirmation reversed both to `1.001720x` and `1.001719x`. The broad form confirmed at `1.000863x` and `0.998932x`. Setup unpairing is therefore rejected as noise-scale rather than promoted from the counter correlation. Artifacts and the builder remain under `~/tmp/torch-ggml-ops/q6-setup-pairing/` and `~/tmp/torch-ggml-ops/build_q6_setup_pairing.py`.

### Global factor-read frontier

The final address-order candidate reproduced the one narrow global-read behavior available without register reassignment: both typed layouts already form the Q6 block-factor address near the loop header, so `factor-first` issued that typed `Q6GlobalRead` immediately rather than leaving it behind the packed-weight batches. `factor-first-overlap` combined the same read frontier with monotonic producer-index waits and converted the factor before the existing row/role decode wavefront. The candidate did not copy LLVM instructions, allocate registers, or reorder arithmetic.

Both forms assembled, linked, retained the selected resources, and produced zero differing BF16 elements. The serial nine-repeat screen gave M64 candidate/parent ratios of `1.005036x` and `0.999378x`, and M128 ratios of `0.996255x` and `0.995713x`, for factor-first and factor-first-overlap respectively. The 20-warmup/25-repeat confirmations rejected that apparent M128 signal: M64 ratios were `1.002161x` and `0.998398x`; M128 ratios were `1.000498x` and `0.998984x`. Moving the factor producer and exposing its first-use frontier is therefore another sub-0.2% noise-scale result. Artifacts and the builder remain under `~/tmp/torch-ggml-ops/q6-factor-frontier/` and `~/tmp/torch-ggml-ops/build_q6_factor_frontier.py`.

### Final recursive exhaustion review

This is a fresh review iteration after the setup-pairing and factor-frontier mechanisms were measured and rejected. The earlier fixed-schedule review, typed reconstruction, oracle analysis, and every newly actionable semantic mechanism have now been resolved:
- Retained and measured: fixed-LDS J64/J128 ownership, structured `(32,4,1)` launches, two J128 row tiles for M256, M64 delay-hint removal, semantic BF16 pipelines, and the selected row/role decode wavefront.
- Rejected by timing: paired activation-scale reads and decoded stores, compact raw payload, expanded transposed scales, coarse producer-readiness overlap, decode/store interleaving, dot-frontier skewing, factor-first LDS and global reads, setup VOPD unpairing, global-L0 invalidation removal, J128 delay removal, alternate decoded geometries, dedicated decoder waves, no-cluster schedules, disabled post scheduling, and low-register-only policies.
- Correctness or contract-incompatible: NaN-path simplification for arbitrary packed data, reordered effective scales, prepared weights, shared decode workspaces, producer fusion, persistent/grouped traversal, public dispatch changes, and build-time invocation of LLVM scheduling or allocation passes.
- Deferred with explicit prerequisites: public dispatch and bundle integration remain a separate deployment task; Q3_K/Q8_0 remain separate forward campaigns before any global forward-exhaustion claim.

The residual LLVM behavior is no longer an untested semantic policy. Its max-ILP bodies use 56 SGPR at M64 and 71 SGPR at M128, versus 27 for both selected typed bodies, and split many 64-bit address additions into low-half VALU frontiers with SGPR carry lifetimes. LLVM also changes physical allocation and schedules ready nodes across setup, global reads, decode, local writes, dot preparation, and the epilogue. Basic allocator substitutions erase most of the LLVM max-ILP gain, while disabling machine scheduling is catastrophic. Reproducing that remainder would require a new physical register map, register repair/allocation, an instruction-DAG scheduler, or a copied raw schedule. All four violate the fixed-map typed-generation contract; no narrower wait, pairing, load, store, decode, or dot policy remains unmeasured.

The Q6 stopping condition therefore passes without claiming universal oracle parity. The typed implementation exceeds the exploratory LLVM candidates at M256, while the measured approximately 4-5% M64 and 1-2% M128 oracle gaps are recorded as compiler-level allocation and cross-stage issue scheduling outside the accepted policy surface. All three selected exact keys retain two strict complete-call HIP wins, exact mutation qualification, zero private resources, deterministic source/object/HSACO rebuilds, and the line-complete 408-test repository qualification. The selected Q6 exact keys are closed under the fixed packed-weight and Q8_1 producer contract; a global forward-exhaustion claim remains deferred until the separate Q3_K and Q8_0 campaigns are closed.

## Cross-Campaign Schedule Result

The deterministic scheduler-oracle result changed the implementation premise, not the packed-data contract. The semantic policy fields now describe two distinct direct lowerings, and the selected wavefront policy passed exact correctness, mutation, resource, deterministic-build, and repeated timing gates for M64, M128, and M256.

### Final multiply result

The final selected identity is the typed wavefront lowering. The table reports prequantized multiply bodies only. HIP and GGTensile consume the same workspace from the same fixed `quantize_bf16_q8_1_f32_d4` kernel; activation production is excluded from every throughput, speedup, and weighted result below. Logical throughput is `2*M*N*K/(median_ms*1e9)`, and speedup is `HIP median time / GGTensile median time`, so values above `1.0x` favor GGTensile. Each value combines the two independent warmed rotating 25-repeat confirmations by averaging their median times; the largest per-key A/B speed difference was `0.74` percentage points.

| `(M,N,K)` | Public catalog hash | HIP TFLOPS | GGTensile TFLOPS | GGTensile/HIP speedup |
| ---: | --- | ---: | ---: | ---: |
| `(64,248320,2048)` | `ggsol_e4a90622ebc83190` | `16.197` | `16.502` | `1.0188x` |
| `(128,248320,2048)` | `ggsol_67c5d2386851ff85` | `17.550` | `17.702` | `1.0087x` |
| `(256,248320,2048)` | `ggsol_45df63b663fc39b5` | `17.419` | `17.584` | `1.0095x` |

The effective 32/16/8-call weighted speedup is `1.0125x`. The prior HIP-scheduled rows below are retained as historical selection evidence; they are not the final wavefront result.

#### Same-Producer Complete-Call Diagnostic

Two separate 25-repeat audit passes timed quantization plus multiply with one shared F32_D4 producer instance and workspace per HIP/GGTensile pair. The complete-call speedups A/B were `1.0174x/1.0165x` for M64, `1.0086x/1.0073x` for M128, and `1.0065x/1.0044x` for M256. Within each audit pass, producer inclusion changed the speedup by at most `0.21` percentage points and did not consistently reduce it. Q6 therefore has no notable complete-call dilution. The full timing distributions are in `~/tmp/torch-ggml-ops/fwd-complete-audit-q6-{a,b}.json`.

M256 did not inherit the J128 result automatically. It remained a separate default-policy control until two full-shape 25-repeat comparisons and exact mutation qualification independently supported the same wavefront policy. This preserves the campaign rule that a shared geometry does not imply a shared selection without shape-specific evidence.

The earlier shared decoded M256 ownership was a conditional follow-up. It saved staging work against two J128 calls but was approximately `4%` behind HIP under the older semantic schedule. The reopened campaign below reconstructed the missing typed MT256 physical plan against the selected deterministic schedule and rejected it independently under the current protocol.

No effective-scale arithmetic reorder, prepared representation, shared decode workspace, producer fusion, or model-owned contract change is part of this reopening. The recursive final-review rule is global: it spans forward and backward directions, all quant types, and all exact shapes. A scheduler or lifetime finding from another format may transfer only when its semantic dependencies and physical resource envelope are re-derived for Q6; a Q6 result may likewise become evidence for another format without silently becoming a selection.

## Reopened Physical-Plan Campaign

The fresh post-refactor review invalidates the preceding Q6 stopping condition. It found two actionable in-contract physical-plan experiments that are absent from the current MT64/MT128 bounded domain. Neither finding changes the packed Q6_K representation, HIP-produced Q8_1 workspace, arithmetic order, 40-byte ABI, gfx1151 wave32 target, or zero-spill requirement.

The first experiment is exact shared MT256 ownership for `(256,248320,2048)`. Reconstruct a dedicated typed physical plan for the previously qualified 256-thread/eight-wave ownership, in which the upper wave bit selects the second 128-row output half and both halves share one decoded 64-row weight stage. Compose that ownership with the currently selected row/role wavefront policy, fixed-LDS schedule, semantic BF16 pipeline, and dependency-compatible pairing. The old shared body is correctness evidence only: its physical instruction stream and older schedule are not production inputs. The new plan must derive workgroup ownership, wave-column masking, LDS, register roles, synchronization, output indexing, and resources from canonical typed state. It must not alias an M256 problem to two MT128 workgroups or import a raw assembly template.

The MT256 candidate first passes reduced-K structural fixtures, independent-reference comparison, full exact HIP comparison, input mutation, packed-weight mutation, Q8_1-workspace mutation, finite-output checks, deterministic source/object/HSACO rebuilds, code-object-v5 metadata, the exact ABI, and zero private bytes, spills, scratch instructions, calls, or dynamic stack. A serial warmed nine-repeat screen compares it with the selected two-MT128 wavefront parent and HIP. A resource-bearing candidate advances only after a stable greater-than-2% parent gain, then receives two independent reversed-order 25-repeat multiply-only confirmations. Promotion remains exact-key-only.

### Shared MT256 Result

The typed shared-MT256 candidate is `ggsol_e58cee5eac9afbbc` at exact key `(256,248320,2048)`. Its `(32,8,1)` workgroup is two explicit four-wave M groups. The upper group normalizes packed local Y to the qualified J128 subplan, adds 128 only at activation-global and output-row ownership boundaries, and uses a second activation LDS plane after the complete retained `38,400 B` lower-group image. Only the lower group decodes and publishes the shared 64-row weight stage. This preserves the selected lower 128-row stream while deriving the eight-wave ownership without importing the historical body.

The artifact passed exact HIP and public comparison with zero differing elements across all `63,569,920` BF16 outputs, finite-output checks, deterministic producer replay, input/packed-weight/workspace mutation sensitivity, and independent-reference NRMSE `0.006081817`. It is code-object v5 with the exact 40-byte ABI, 256-thread workgroup, `210 VGPR`, `27 SGPR`, `57,344 B` LDS, 16 static WMMAs, four barriers, zero private bytes, zero spills, and no scratch, calls, or dynamic stack. Two independent builds matched exactly.

The warmed serial nine-repeat multiply-only screen rejected it:

| Artifact | Median ms | Candidate-relative result |
| --- | ---: | ---: |
| selected two-MT128 wavefront parent | `14.8034` | candidate is `1.0332x` slower |
| HIP | `14.8658` | candidate is `1.0289x` slower |
| shared MT256 candidate | `15.2952` | rejected |

Candidate and parent still differed on zero BF16 elements in the timing harness. The candidate missed the required greater-than-2% parent gain by a wide margin, so the two 25-repeat confirmations were not run. The selected two-MT128 exact-key state and public bundle remain unchanged. Evidence is under `~/tmp/torch-ggml-ops/q6-shared-mt256-wavefront/`, including `qualification-1.json`, `screen-9.json`, and both deterministic rebuilds.

The second experiment is a named deterministic Q6 physical register-role alternative, beginning with `(64,248320,2048)`. The prior compiler-oracle result proves resource-clean headroom but does not authorize LLVM scheduling in production. Implement at most one complete static plan that owns the widened scalar address/carry frontier, register-role order, and corresponding block-level stage traversal directly. The plan must serialize through strict identity, lower without an allocator, repair pass, arbitrary instruction permutation, absolute issue table, source rewrite, or copied LLVM stream, and reject every unsupported partial combination. If the useful behavior cannot be expressed at that semantic granularity, record an architectural rejection rather than weakening the writer contract.

The M64 physical-plan candidate uses the same correctness, mutation, resource, ABI, and deterministic-artifact gates as MT256. It advances from a serial nine-repeat screen only with a stable greater-than-2% gain over the selected M64 wavefront parent and then requires two reversed-order 25-repeat parent/HIP confirmations. M128 is tested only if the same complete mechanism qualifies at M64 and remains resource-clean when formula-derived for two output rows; no shape-selected register map is allowed.

### Wide Scalar-Carry Result

The one complete named alternative is `WideScalarCarryFrontier`, serialized as `Q6PhysicalPlan` in exact identity `ggsol_079a7580f46d8812`. It retains the selected M64 VGPR semantic roles, arithmetic, loads, LDS operations, WMMA order, waits, barriers, VOPD policy, and BF16 epilogue. Each bounded batch of at most eight independent 64-bit addresses instead owns fixed carry roles `s25..s32` and lowers all low halves before the corresponding high halves. The plan is valid only for wavefront MT64; unknown values, partial schedule combinations, MT128, and MT256 reject before lowering. It uses no allocator, repair pass, issue table, source rewrite, compiler pass, or imported instruction stream.

The exact `(64,248320,2048)` artifact passed HIP/public comparison with zero differences over `15,892,480` BF16 outputs, finite-output checks, deterministic producer replay, all three mutation-sensitivity gates, and independent-reference NRMSE `0.006159443`. It is code-object v5 with the exact 40-byte ABI, `158 VGPR`, `33 SGPR`, `28,928 B` LDS, eight static WMMAs, four barriers, zero private bytes, zero spills, and no scratch, calls, or dynamic stack. Its source/object/HSACO rebuilt identically.

The warmed serial nine-repeat multiply-only screen measured parent `3.96808 ms`, candidate `3.97024 ms`, and HIP `4.07555 ms`. Candidate/parent latency was `1.00054x`, or 0.05% slower; candidate/HIP latency was `0.97416x`. The plan therefore fails the required greater-than-2% parent gain. No 25-repeat confirmation was run, and the contract forbids deriving M128 from a mechanism that did not qualify at M64. The selected M64 wavefront exact key and public bundle remain unchanged. Evidence is under `~/tmp/torch-ggml-ops/q6-wide-scalar-carry-m64/`.

The recursive final-review rule remains mandatory. A finding cannot be discovered, implemented or rejected, and declared exhausted in the same review iteration. After each experiment is retained or rejected, restore or promote the authoritative exact-key state, update this record with artifacts and measurements, and perform a new read-only review spanning every forward and backward experiment record, the current writer and physical plans, bounded search domains, selected catalogs, HIP controls, and target ISA. Stop only when that fresh final pass finds no new actionable in-contract mechanism with an exact target, plausible gain path, and qualification gate.

## Fresh Post-Experiment Global Review

The required fresh review was performed as a separate read-only iteration after both reopened Q6 results were recorded and the selected state was restored. It reread all ten forward/backward experiment records, all ten strict catalogs, the forward and backward candidate-domain implementations, the current Q6 spec/physical/lowering/validation surfaces, the selected and rejected Q6 artifacts, the bundle boundary, and the gfx1151 contract exclusions.

The apparent openings classify as follows:
- Retained and measured: the 106 exact catalog mappings, selected Q6 MT64/MT128 wavefront plans, two-MT128 ownership for problem M256, Q3 full-weight ownership, Q4/Q5 typed activation-base lifetime, Q8 compact DepthU32 and Q-A `HipTile`, and the selected backward geometry/decoder/pipeline compositions.
- Evidence-rejected: shared Q6 MT256, `WideScalarCarryFrontier`, Q4 `OneByEight`, Q5 compact consumer decode, Q8 small/compact controls that missed their gates, and the repaired backward Q4/Q5/Q6 paths. None supplies a changed performance premise after its exact correctness, resource, and timing result.
- Contract-incompatible or deferred: prepared/model-owned weights, producer fusion, persistent or grouped traversal, split-K/Stream-K, external decode workspaces, ABI changes, public dispatch, and alternate recurrent permutations. The remaining compiler-oracle Q6 differences require whole-stream scheduling/register allocation or copied instruction order, which are outside the accepted typed physical-plan contract.
- Stale chronology or tooling-only: Q4's formerly pending activation-base alias is implemented; Q8's historical active reopening is closed by the compact-depth32 and hot Q-A evidence; backward renewed-work sections terminate in later stopping passes; and bounded candidate domains intentionally enumerate linked implemented neighbors without claiming exhaustion or reproducing every selected composition.
- Newly actionable in-contract mechanism: none. No reviewed item has a new exact target, plausible greater-than-2% gain path, and qualification gate that is not already retained, measured, rejected, or contract-excluded.

The strict catalogs therefore remain at 56 forward and 50 backward exact keys, the rejected Q6 identities remain outside deployment selection, and the 179-kernel public bundle remains unchanged. This separate pass restores the global optimization-exhaustion conclusion under the fixed gfx1151 wave32, packed-GGUF, HIP-Q8_1, BF16, 40-byte ABI, code-object-v5, and zero-spill contract.

## Candidate-Domain Evidence Boundary

The automated Q6 domain is a bounded deterministic enumerator of complete implemented policies, not evidence that the kernel is exhausted. The M64 neighborhood now includes the linked `WideScalarCarryFrontier` identity alongside the canonical instruction and epilogue policies. Shared MT256 is formula-capable and constructible through the same typed schedule path, but remains an exact-key experiment rather than a generic domain seed. That improved coverage records implemented mechanisms but does not retroactively establish closure or replace the measurements above. Exact artifacts, rejection evidence, and the recursive manual review remain authoritative. A missing domain value is tooling-only unless it exposes an implemented or implementable in-contract mechanism with an exact target and qualification gate.

## Post-Audit Oracle and Arithmetic Reopening

Status: O1 measured and rejected; O2 remains planned and unmeasured. This section deliberately broadens the research policy surface beyond `WideScalarCarryFrontier`; it does not authorize an allocator, generic rescheduler, copied machine stream, or compiler invocation in production generation.

### O1: Dependency-DAG compiler oracle

Recreate paired offline LLVM controls for the selected M64 and M128 bodies, then compare their machine streams with the typed parents through a semantic dependency DAG. Record stage-to-stage issue distances, live role counts, VMEM/LDS readiness, legal VOPD pairs, and `s_delay_alu` dependency tokens, including only target-valid `VALU_DEP_1` through `VALU_DEP_4` relationships. Compiler timing is oracle evidence, not a selectable artifact.

Encode at most one complete named physical plan per shape. The plan may change role lifetimes, stage traversal, metadata interleaving, delay tokens, and constrained VOPD formation, but it must lower directly from serialized semantic state with fixed resources and reject partial combinations. It may not contain absolute issue slots or an imported instruction list. Start at M64, where the prior oracle gap was approximately 4-5%, then M128. M256 is tested only if the same semantic mechanism is formula-derived and independently exact.

All O1 candidates retain arithmetic order and must be bit-exact, deterministic, zero-spill, and faster than the selected parent in two paired confirmations. O1 requires no model integration. A reusable dependency/liveness representation should be shared with the Q4 wait experiment, while physical plans and timing remain format-specific.

### O1 result: cooperative address/read frontiers

The saved M64 LLVM max-ILP body confirms that ready address halves and VMEM producers are issued across narrower, nonuniform frontiers than the selected typed body's three eight-address/eight-read groups. A bounded typed screen therefore split each matching eight-address/read group into either two- or four-read readiness groups while preserving the selected wavefront decode, fixed VGPR map, arithmetic, LDS ownership, waits, barriers, WMMA order, and epilogue. The scalar-carry forms additionally reused the admitted `WideScalarCarryFrontier` roles so each readiness group issued its low address halves before its high halves. These were direct emitter experiments; no LLVM stream, instruction slot table, allocator, source rewrite, or production identity was added.

All four artifacts assembled and linked as code-object v5, retained `158 VGPR`, `28,928 B` LDS, eight WMMAs, four barriers, zero private bytes, zero spills, and no scratch or calls. Plain forms retained `27 SGPR`; scalar-carry forms retained the already admitted `33 SGPR` envelope. Every candidate produced zero differing BF16 elements from the selected wavefront parent over all `15,892,480` outputs in its serial timing run.

The nine-repeat screens measured candidate/parent latency ratios of `1.005664x` and `0.997671x` for plain two- and four-read groups, and `1.005404x` and `0.999652x` for scalar-carry two- and four-read groups. The only apparent signal was therefore 0.23%, far below the prior oracle gap. A single-process 20-warmup/25-repeat confirmation compared both four-read forms against the same parent and HIP. Parent was `3.975948 ms`, plain interleave was `3.977750 ms` (`1.000453x` parent latency), scalar-carry interleave was `3.983655 ms` (`1.001938x`), and HIP was `4.059363 ms`. Both candidates reversed to slower than parent.

No M128/M256 transfer, deterministic rebuild, or production policy was justified. O1 is rejected as timing-neutral under the fixed typed register map. Its residual LLVM behavior still requires cross-stage physical allocation and whole-DAG issue selection rather than a load/address readiness policy. Artifacts, inspection reports, screen JSON, confirmation samples, and the temporary builder are under `~/tmp/torch-ggml-ops/q6-load-frontier/` and `~/tmp/torch-ggml-ops/build_q6_load_frontier.py`.

### O2: Changed Q6 arithmetic policies

Run two independent numerical campaigns. The first compares final-output `RNEPreserveNaN`, one-instruction `BiasRound`, and zero-instruction `Truncate`. The second compares the installed scale order with a named effective-scale reassociation that computes a reusable Q6 block/scale factor before applying it to converted accumulators. Do not compose conversion and reassociation until each has an isolated timing and numerical result.

Cross-record prioritization makes the reassociation the highest-upside still-unmeasured arithmetic experiment. The selected typed dot phase currently carries `8 * tile_count` scaled products: 32 at M64 and 64 at M128, in each of two dot phases. A reusable block/scale factor can therefore alter a repeated hot-loop frontier, whereas output conversion runs once per final accumulator. Those counts bound the affected work; they are not a claim that every product multiply can be removed. The planning prior is a high-single-digit body gain, with low-double-digit upside only if reuse actually reduces per-product scale issue without raising allocation or lengthening another dependency chain.

Keep output conversion as the lower-cost independent screen because replacing the Q6 epilogue's exact NaN-preserving sequence with `BiasRound` or `Truncate` removes more instructions than replacing the common two-instruction rounding pair. Report body and complete-call movement separately. Neither planning prior is a timing result, and numerical/model qualification remains authoritative.

Finite-input numerical tests reject any non-finite output and report differing BF16 elements, normalized RMSE, maximum and high-percentile error against the exact parent and dequantized reference. No kernel-side NaN/Inf test, clamp, or repair sequence is added. O2 identities remain outside exact catalogs and require model integration for inference quality, loss, and gradient stability before retention.
