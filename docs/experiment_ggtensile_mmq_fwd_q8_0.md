# GGTensile MMQ Forward Q8_0 Experiment

## Purpose

Implement and optimize dense MMQ forward Q8_0 assembly kernels for the DeepSeek workload on gfx1151. The campaign must optimize the complete packed-weight multiply, including global reads, Q8_0 decode, Q8_1 activation-workspace consumption, LDS movement, integer WMMA, correction/scaling, synchronization, and BF16 stores.

The existing HIP dense Q8_0 forward kernels are correctness and performance controls. They establish that the operation is feasible, but their templated implementation and generated assembly are evidence only. GGTensile must express the same operation through the repository's typed forward contracts, semantic stages, deterministic register allocation, and explicit mechanism policies rather than copying a large HIP assembly body.

This document is the active Q8_0 forward campaign record. Update it after every coherent implementation, correctness, resource, measurement, rejection, or review change. Code changes worth retaining receive commits; documentation-only checkpoints need not be committed.

## Contract

Target only:
- gfx1151, code-object version 5, wave32, WMMA V1, and the existing 40-byte MMQ forward ABI.
- BF16 input, upstream Q8_1 `F32_D4` activation workspace, FP32 integer-dot correction/accumulation, and BF16 output with the established rounding contract.
- authoritative packed GGUF Q8_0 weights consumed directly by the generated kernel.
- exact `ProblemType`, `ProblemSize`, and complete `ForwardSolution` identity per artifact.
- deterministic generation, assembly, linking, inspection, independent correctness, and warmed rotating measurement.
- zero private storage, spills, scratch instructions, calls, and dynamic stack.

Do not introduce prepared weights, dense shadows, external decode workspaces, split-K, persistent or grouped workgroups, producer fusion, hidden caches, online tuning, or public dispatch changes during this experiment. Unsupported shapes must reject to the existing HIP/runtime path rather than silently generalizing an artifact.

Q8_0 has 32 logical values in 34 packed bytes:

```text
d       FP16 block scale
qs[32]  signed int8 values
value[i] = fp16(d) * int8(qs[i])
```

The forward dot uses the Q8_0 signed int8 payload and the Q8_1 F32_D4 signed int8 activation payload. The semantic correction is the product of the Q8_0 block scale, the Q8_1 activation scale, and the integer WMMA result. There is no Q4/Q5 minimum/sum correction.

Activation quantization is upstream. The forward writer addresses and consumes the fixed 144-byte Q8_1 F32_D4 workspace; it does not implement the producer.

## Exact Production Scope

There are 23 unique exact shape keys. Ordinary DeepSeek has six physical `(N,K)` families, crossed with `M={2048,8192,32768}`. The shared gate and shared up tensors have the same shape and can share one shape-specific artifact while remaining separate tensor cases in correctness and call-count accounting.

| Family | `(N,K)` | Representative tensor(s) | Calls per tensor |
| --- | ---: | --- | ---: |
| Attention Q-A | `(1024,4096)` | `blk.0.attn_q_a.weight` | 43 |
| Attention Q-B | `(32768,1024)` | `blk.0.attn_q_b.weight` | 43 |
| Attention KV | `(512,4096)` | `blk.0.attn_kv.weight` | 43 |
| Attention output B | `(4096,8192)` | `blk.0.attn_output_b.weight` | 43 |
| Shared gate/up | `(2048,4096)` | `blk.0.ffn_gate_shexp.weight`, `blk.0.ffn_up_shexp.weight` | 43 each |
| Shared down | `(4096,2048)` | `blk.0.ffn_down_shexp.weight` | 43 |

The language-model head adds five chunk keys:

```text
(M,N,K) =
(32,129280,4096)
(64,129280,4096)
(128,129280,4096)
(256,129280,4096)
(512,129280,4096)
```

M512 is the primary complete-loss chunk; M256 is the lower-memory alternative; M32/M64/M128 remain required fallback and capacity keys. Ordinary and language-model-head ownership are separate campaign families even when a formula-compatible lowering can be shared.

## Priority and Controls

Fresh same-process HIP controls must be collected before selecting a GGTensile candidate. Historical timings are prioritization evidence only.

Priority order:
- Q-B `(N,K)=(32768,1024)` at M8192/M32768 and attention-output B `(4096,8192)` at long M, because they have large absolute kernel cost and large decode/reduction depth.
- Shared gate/up `(2048,4096)`, weighted as two tensor calls per layer.
- Shared down and Q-A, including their long-M keys.
- Attention KV, after the larger ordinary margins are closed.
- LM-head M512/M256, then M128/M64/M32 for chunk fallback and capacity behavior.

Complete-call timing and prequantized multiply timing are recorded separately. The fixed HIP Q8_1 producer is byte-identical for HIP and GGTensile multiply controls and is excluded from isolated multiply promotion. Complete-call and call-weighted totals guide workload decisions but never retain a slower exact key.

## Current State

The canonical forward model has typed Q8_0 direct-global and LDS-free register-tiled controls. The register-tiled `128x32` source is the retained research parent, but no GGTensile key is performance-promoted. The exact inventory covers all 23 keys and explicitly selects HIP fallback for production; the parser-valid control catalog remains available for continued offline work without changing public dispatch.

Existing HIP controls are available through the dense forward bundle sources and dispatch in `csrc/mmq_bundle.cpp` and `csrc/mmq_core.cuh`. Existing Q8_0 GGTensile artifacts under `~/tmp/torch-ggml-ops/` are backward artifacts and are not forward baselines.

Baseline preservation checkpoint: all currently valid catalog sources were regenerated before Q8 edits into `~/tmp/torch-ggml-ops/q8-fwd-baseline-sources-20260805/`. The manifest contains 447 unique valid sources: 155 Q4_K/Q5_K/Q6_K forward sources and 292 Q3_K/Q4_K/Q5_K/Q6_K/Q8_0 backward sources. Every retained Q8 change must reproduce this set byte-for-byte unless an existing-stream change is explicitly separated and qualified.

The canonical 23-key inventory now records fresh same-process HIP multiply medians for every exact key. All 23 entries have `CurrentStatus: selected` and `SelectedSolution: hip_fallback`: no GGTensile candidate reaches HIP parity, so the public HIP path remains the explicit decision for every key. The timing fields are evidence for that decision, not placeholders.

The initial isolated backend lowers one wave to a `16x16` output tile. It directly loads four 32-value Q8_0 blocks and one 128-value Q8_1 F32_D4 block per reduction-loop iteration, performs eight signed integer WMMAs, and applies the FP32 correction in the HIP expression order `integer_result * weight_scale * activation_scale` before BF16 RNE stores. `Q8DirectGroupRole` carries the payload and scale offsets, while `Q8DirectRegisterPlan` allocates explicit lifetime-bound roles deterministically. This branch does not use LDS and does not alter the Q4_K/Q5_K/Q6_K body methods.

Initial control evidence is under `~/tmp/torch-ggml-ops/q8-fwd-direct-control/m2048-n1024-k4096/`:
- strict assembly inspection passes with code-object v5, gfx1151, wave32, the 40-byte ABI, 88 VGPRs, 16 SGPRs, zero LDS, eight static WMMAs, zero private storage, and zero spills.
- after preserving the HIP source expression order `integer_result * weight_scale * activation_scale`, the Q-A `M=2048,N=1024,K=4096` control is bit-exact with both the same-workspace HIP multiply and public complete path across 2,097,152 BF16 elements. Candidate and public independent-reference normalized RMSE are both `0.00604494`.
- the LM-head `M=32,N=129280,K=4096` control is also bit-exact with HIP/public across 4,136,960 BF16 elements; candidate and public independent-reference normalized RMSE are both `0.00606697`.
- input, packed-weight, and Q8_1-workspace mutations change candidate output for both controls while remaining bit-exact with HIP.
- one-repeat diagnostics measured Q-A isolated multiply at `1.9162 ms` versus `0.5880 ms` HIP and LM-head M32 at `7.4937 ms` versus `2.7742 ms` HIP. These are not warmed promotion evidence.

The focused contract/reference/writer suite passes with `159 passed`. The full repository suite passes with `358 passed` and 14 existing warnings; Ruff, formatting, ty, compileall, diff checks, pre-commit, and the current 179-kernel public-bundle check also pass. Regeneration of the frozen eight pre-Q8 catalog families produced 447 sources with zero missing, added, or byte-changed files; the report is `~/tmp/torch-ggml-ops/q8-fwd-final-sources-20260805/comparison.json`.

The first optimization adds an LDS-free four-wave register tile. Each wave owns four `16x16` fragments and retains weight and activation operands through four cross-products. The formula-derived resource count is `67 + 25*wave_tile_n + 10*wave_tile_m` VGPRs, 16 SGPRs, and zero LDS. Strictly inspected `1x4`, `2x2`, and `4x1` ownership geometries use 177, 137, and 132 VGPRs respectively, with zero private storage/spills and 32 static WMMAs each.

Warmed Q-A `M=2048,N=1024,K=4096` geometry screening under `~/tmp/torch-ggml-ops/q8-fwd-register-tile-geometry/` measured isolated multiply medians of `1.3217 ms`, `0.8800 ms`, and `0.8955 ms` for `1x4`, `2x2`, and `4x1`. Their static `(VMEM, VALU)` counts are `(204,1038)`, `(136,875)`, and `(120,804)`. The balanced `2x2` `128x32` geometry is the narrow winner and is retained in the Q8 solution catalog. It is bit-exact with HIP/public and passes all mutation and independent-reference gates, but its `1.4300x` HIP ratio is not a production promotion.

Cooperative raw-weight LDS staging was implemented and then removed after failing the promotion rule. The `128x32` form used 121 VGPRs, 4,608-byte LDS, 59 VMEM, 83 LDS instructions, two barriers, and 32 WMMAs. It remained bit-exact, but measured `0.8972 ms`, or `1.4357x` HIP, against register-parent runs at `1.4300x` and `1.4374x` HIP. Wider `128x64` and `64x64` variants measured `0.9390 ms` (`1.5512x` HIP) and `0.9830 ms` (`1.5952x` HIP). The added LDS traffic and synchronization produced no stable gain. Artifacts remain under `~/tmp/torch-ggml-ops/q8-fwd-weight-lds-register-tile*/`.

Dependency-independent VOPD pairing of register zeros and scale multiplies reduced static VALU issues from 875 to 731 with 144 VOPD instructions and unchanged 137-VGPR resources. It remained bit-exact but regressed to `0.9554 ms` (`1.4949x` HIP) versus the same scalar artifact at `0.9300 ms` (`1.4374x` HIP), so the paired lowering was removed. Pairing the FP32 FMACs was also rejected structurally because a pair reuses one activation scale and cannot satisfy gfx1151's distinct-bank requirement for both sources. Evidence is under `~/tmp/torch-ggml-ops/q8-fwd-register-tile-vopd/`.

The retained register-tiled source then hoisted the four fixed Q8_0 block offsets into `Q8DirectGroupRole` and advanced the per-lane packed addresses by 136 bytes per activation block. The Q-A artifact remained bit-exact with 137 VGPRs, 16 SGPRs, zero LDS, and 136 VMEM instructions; its warmed multiply diagnostic improved to approximately `0.8714 ms` (`1.3674x` HIP). A subsequent linear store traversal computes the first output address once and advances by fixed row/column deltas. It removes 37 static VALU instructions and stayed correct in the short recovery check at `0.8592 ms`; a rotating comparison reported `0.81055 ms` for a scale-overlap experiment against `0.80945 ms` for the retained parent, so only the linear traversal is retained.

Three follow-up scheduling tests were rejected. The first store `s_clause` was invalid because address updates occurred between stores and caused the first launch to fail to complete; its artifact is quarantined. Moving weight-scale loads to the front of each clause and partially waiting for scale conversion was bit-exact but neutral in a rotating 25-repeat comparison (`1.00136x` parent). Advancing the packed-weight and activation scalar base pointers instead of 20 per-lane VGPR addresses was also bit-exact, but the loop-carried scalar dependency regressed to `1.02210x` parent. Both schedules were removed.

The store-clause mechanism was then repaired by materializing eight output addresses in the dead WMMA product registers before each fragment's contiguous store sequence. The corrected form is bit-exact, remains at 137 VGPRs/16 SGPRs with zero LDS, and has eight static clauses. Two independent rotating 25-repeat comparisons measured candidate/parent median ratios of `0.96645x` and `0.96411x`. A follow-up reused all 32 dead product registers to issue one 32-store clause; it remained exact and reduced static clauses from eight to five, with two rotating comparisons against the four-clause-store parent at `1.00111x` and `1.00034x`. The one-clause epilogue is retained as a resource-neutral simplification. The active writer therefore uses the simple 26-read clause, full wait, expression-order correction, and address-materialized 32-store clause.

An offline no-extra-register payload pipeline then moved each next group's weight and activation payload reads immediately after the current operands' last WMMA use, while loading the next scales after the current correction. It kept the exact `137/16/0` resources, 136 VMEM instructions, five waits, and five clauses, and remained bit-exact. A rotating 25-repeat comparison measured `0.83267 ms` versus `0.78333 ms` for the retained parent (`1.06299x`), so the schedule was rejected. Splitting the 26-read clause and injecting VMEM into the WMMA/correction stream costs more than the hidden payload latency saves.

An offline pressure experiment packed pairs of FP16 weight scales into eight VGPRs and retained one packed workitem coordinate instead of separate lane and wave registers. The linked artifact declared 128 VGPRs with no reference above `v127`, remained bit-exact, and used on-demand low/high scale conversion to preserve scale reuse across the two activation fragments. The extra extraction arithmetic dominated: rotating medians were `0.86205 ms` versus `0.78846 ms` parent (`1.09334x`). The 128-VGPR form was rejected; pressure reduction is not useful when it duplicates scale conversion in every product fragment.

A second offline 128-VGPR experiment preserved the original FP32 scale reuse by keeping eight scale-address VGPRs and alternating those addresses between the two N fragments across the four groups. It also used one packed workitem coordinate, declared 128 VGPRs/16 SGPRs, referenced at most `v127`, and had zero LDS/private storage/spills. The form remained bit-exact, but the added address dependencies and split read clauses were neutral-to-slower: rotating medians were `0.78901 ms` versus `0.78665 ms` parent (`1.00301x`). The mechanism was rejected because it adds substantial semantic complexity without a target gain.

Store-only `s_setprio` values 1, 2, and 3 were screened offline after the reduction loop. All three were bit-exact and resource-neutral, but rotating 21-repeat ratios were `1.00581x`, `1.00190x`, and `1.00550x` parent. The default priority remains retained.

Moving each next fragment's eight zero-initializations immediately after the previous fragment's second WMMA was also bit-exact and resource-neutral. The intended result-latency coverage did not materialize: rotating medians were `0.78377 ms` versus `0.78089 ms` parent (`1.00369x`). The original local WMMA/correction order remains retained.

The 20 packed-weight, scale-address, and activation-address advances from the last group were distributed five at a time after each second WMMA to remove the serial loop tail. The schedule remained exact and resource-neutral, but VALU issue contention outweighed latency coverage: rotating medians were `0.78884 ms` versus `0.78322 ms` parent (`1.00717x`). Address advancement remains after the correction body.

Deferring the eight n1 FP16-to-FP32 scale conversions until immediately after the first fragment's second WMMA was exact and resource-neutral, but rotating medians were `0.79378 ms` versus `0.79214 ms` parent (`1.00207x`). Scale conversion placement is not retained.

The retained Q-A checkpoint is `~/tmp/torch-ggml-ops/q8-fwd-register-tile-store-clause-all/m2048-n1024-k4096/`. Strict inspection reports code-object v5, gfx1151, wave32, the 40-byte ABI, 137 VGPRs, 16 SGPRs, zero LDS/private storage/spills, 32 WMMAs, 136 VMEM instructions, 784 VALU issues, five waits, and five clauses. It is bit-exact with HIP/public for all 2,097,152 BF16 outputs, passes every mutation gate, and matches the independent-reference envelope at normalized RMSE `0.00604494`. Source, object, and HSACO hashes are respectively `3bd8a96d1e44b6819eb2f31469aeacea7c75529ed337d70528705bd50cc10069`, `3e96c302b1b4dbf4778b96cf86e66b8e5379828e9c937f5168c61b0a3a749691`, and `afd6aa768415373a934f800bce3d8f71f9880691a3ed89b074ef09a5846ad3d4`; two independent rebuilds reproduced all three hashes.

The current quality checkpoint passes the focused Q8/forward suite with `159 passed`, the full repository suite with `358 passed` and 14 existing warnings, Ruff checking and formatting, `ty`, compileall, pre-commit, and the 179-kernel bundle-current check. Regenerating the frozen pre-Q8 gate reproduced all 447 sources byte-for-byte. The report is `~/tmp/torch-ggml-ops/q8-fwd-final-sources-20260805/comparison.json`.

## Exact-Key Qualification And Fallback

The complete exact-key campaign used the retained store-clause register parent and fresh HIP controls. Every listed candidate passed strict inspection, full HIP / public correctness, independent-reference checks where recorded, and input, packed-weight, and Q8_1-workspace mutation gates. The values below are warmed multiply candidate/HIP median ratios; values above `1.0` are slower and therefore fall back to HIP.

| Family | M2048 | M8192 | M32768 |
| --- | ---: | ---: | ---: |
| Attention Q-A (`2x2`) | `1.3891x` | `1.2756x` | `1.6559x` |
| Attention Q-B (`4x1`) | `1.2003x` | `1.2364x` | `1.2599x` |
| Attention KV (`2x2`) | `1.3630x` | `1.3489x` | `1.2828x` |
| Attention output B (`4x1`) | `1.4133x` | `2.2685x` | `2.0106x` |
| Shared gate/up (`4x1`) | `1.2641x` | `1.4580x` | `1.6928x` |
| Shared down (`2x2`) | `1.2823x` | `1.3073x` | `1.3814x` |

LM-head multiply ratios were M32 direct-global `2.7551x`, M64 direct-global `5.2410x`, M128 register `2x2` `1.2424x`, M256 register `2x2` `1.4101x`, and M512 register `2x2` `1.4714x`. These controls were exact across the required output domains; the register geometry does not remove the head's HIP gap.

A bounded larger direct-register probe (`M256xN32`, eight fragments per wave) used 221 VGPRs, zero LDS, 64 WMMAs, 192 VMEM instructions, and 1,440 VALU issues. It was exact but measured `0.8500 ms` against `0.6081 ms` HIP on Q-A (`1.3977x` HIP and about `1.06x` the retained parent), so it was removed. The probe confirmed that adding accumulators without cooperative operand reuse is not a viable path.

The final recursive review implemented the remaining concrete mechanism as an isolated wave-N-major `128x64` tile with cooperative staging of all four Q8_1 activation groups. It passed full Q-A BF16 equality and all mutation gates, but used 240 VGPRs and 18,432 bytes of LDS, with 64 WMMAs, 116 VMEM instructions, 108 LDS instructions, nine waits, and two barriers. Its warmed multiply median was `0.87385 ms` versus `0.59900 ms` HIP (`1.4589x`), so the complete source was removed after measurement. The experiment is retained only as the final rejection record; the machine-readable static lower-bound comparison is `~/tmp/torch-ggml-ops/q8-fwd-residual-lower-bounds/q_a_m2048_n1024_k4096.json`.

The residual gap is quantitative: the retained Q-A source has 128 scalar FMACs, 128 scalar scale multiplies, and 64 FP16 scale conversions. The HIP J128 control has 62 scalar FMACs plus one dual-FMAC issue, 64 dual-multiply issues for 128 logical scale multiplies, and four FP16 scale conversions. Direct VOPD scale/zero pairing, DPP scale sharing, clause changes, address overlap, and payload/scale schedule variants were exact but neutral or slower. The completed cross-wave scale/accumulator ownership and activation staging probe also failed the timing gate, so no remaining schedule-only mechanism justifies promoting the current register path.

## Design And Implementation Plan

### Phase 1: contract, inventory, and semantic reference

- Add Q8_0 to the forward format contract only after defining its block bytes, activation layout, signedness, payload planes, scale semantics, and correction formula.
- Add a versionless Q8_0 forward inventory with the 23 unique keys, tensor cases, call counts, controls, and selection status.
- Add strict Q8_0 forward solution/catalog support. Q4/Q5/Q6-only fields must reject or remain fixed; Q8-specific fields must affect lowering or reject.
- Add an independent packed Q8_0 decoder/reference for unit and GPU correctness. Cover every payload byte, positive/negative extrema, zero scale, block boundaries, reduction boundaries, and output boundaries.
- Establish exact HIP/public agreement, finite outputs, input mutation, packed-weight mutation, and Q8_1 workspace mutation before timing any candidate.

### Phase 2: semantic Q8_0 backend

Implement the smallest complete Q8 backend through the current forward architecture:
- typed Q8 payload-plane and scale operands.
- formula-derived packed-row and block addressing.
- explicit Q8 signed-byte decode and Q8_1 activation roles.
- a Q8-specific LDS representation or direct operand path, chosen from the ownership contract rather than copied HIP register order.
- integer WMMA product roles and a typed scale-accumulation correction.
- deterministic register lifetimes and resource derivation.
- semantic setup, global-read, decode/local-write, barrier/local-read, dot, refill, and epilogue stages.

Start from one ordinary large-margin shape and one LM-head shape, then expand only after exact correctness and strict inspection pass. Preserve Q4/Q5/Q6 generated sources byte-for-byte while the new Q8 path is isolated.

### Phase 3: exact geometry and catalog coverage

Expand the backend to all ordinary families and five LM-head keys. Geometry choices are exact and formula-derived; edge tiles reject. Ordinary shape transfer does not imply head transfer.

Every accepted solution must round-trip through the normal `ForwardSolution`/`SolutionKey` boundary, generate deterministically, and have an explicit exact-key catalog decision. HIP fallback remains the decision for a key without a qualified GGTensile winner.

### Phase 4: large-margin optimization

Search complete candidates in this order:
- packed Q8 payload/scale load coalescing and decode clustering.
- Q8-specific LDS row padding, swizzle, and local-read ownership.
- direct-versus-decoded operand representation and one-versus-two stage buffering, only with lower-bound and resource justification.
- `128x64`, `128x128`, `256x64`, and exact small-M geometries when accumulator pressure, LDS, and occupancy estimates support them.
- DepthU, global/local prefetch, Q8 payload sharing, VOPD, clauses, dependency delays, store traversal, and exact-N/wave-group mapping.
- LM-head chunk geometry and active-wave ownership after ordinary margins are understood.

No search point is timed before it passes correctness and strict inspection. Generation never invokes HIP/LLVM scheduling or allocation and never stores a physical instruction schedule as data.

### Phase 5: lower bounds and diagnosis

For each major family, build exact diagnostics with the same ABI, launch ownership, and declared resources:
- WMMA/activation/LDS floor.
- packed-read/decode/LDS floor.
- complete fused body.

Use these floors, static instruction classes, supported counters, register/LDS occupancy, and warmed timings to explain residual margins. Counters are diagnostic and unsupported collection requests are recorded as unavailable, not treated as missing candidate evidence.

### Phase 6: selection, confirmation, and packaging boundary

Promote a candidate only after:
- exact HIP/public output agreement or a documented independent numerical envelope.
- independent reference, finite output, and all mutation gates.
- strict gfx1151/code-object-v5/40-byte-ABI inspection.
- zero private storage, spills, scratch, calls, and dynamic stack.
- deterministic source/object/HSACO rebuilds.
- two independent warmed 25-repeat confirmations against HIP and the retained same-policy parent.
- no material regression for any exact key sharing the candidate's mechanism.

Public runtime dispatch and bundle packaging remain outside this research phase until the full dense Q8 workload has final catalog decisions and end-to-end validation.

## Validation Gates

### Identity-preserving structural changes

Compare generated source, executable text, symbols, ABI metadata, waits, VOPD pairings, LDS offsets, barriers, resource counts, and code objects exactly for unaffected Q4_K/Q5_K/Q6_K pairs. Run focused and full repository tests.

### Deliberate Q8 streams

Require finite output, exact HIP agreement when arithmetic order is unchanged, independent packed/dequantized reference checks, input/weight/workspace mutation sensitivity, deterministic rebuilds, strict artifact inspection, representative and blind divisible-shape coverage, retained-parent comparison, and warmed confirmations.

Resource-bearing mechanisms require a stable gain above 2%. Resource-neutral reductions may be retained only when neutral or consistently favorable across shared exact keys. Static line or instruction reduction alone is not promotion evidence.

## Recursive Final Review

Before declaring Q8_0 forward complete, reread this document, the generic GGTensile design/progress document, all Q8 implementation and experiment records, Q3/Q4/Q5/Q6 forward records, relevant backward and grouped records, current and HIP sources, normalized disassembly, lower bounds, profiles, counters, inventories, rejected candidates, and gfx1151 ISA/LLVM definitions.

Classify every remaining mechanism as retained and measured, rejected by correctness/resources/timing/reproducibility, contract-incompatible or deferred with an explicit prerequisite, or actionable. Every actionable finding must be implemented and measured, followed by a complete fresh review from the changed premise. The review cannot pass in the same iteration that first discovers an actionable mechanism.

Completion requires all 23 exact keys to have final selected-or-HIP-fallback decisions, all retained GGTensile keys to meet the promotion rule, all writer lines to be covered by unit tests, all shared Q4/Q5/Q6 paths to remain qualified, and the residual bottleneck to be explained quantitatively.

## Completion Record

Update this section after each coherent change. Keep detailed timing and artifact paths in `~/tmp/torch-ggml-ops/`.

- Contract, canonical inventory dimensions, and independent packed Q8_0 forward reference.
- Initial semantic Q8_0 backend and strict ordinary control.
- Ordinary 18-key correctness, mutation, and exact-key fallback coverage.
- LM-head five-key correctness, mutation, and chunk fallback coverage.
- Large-margin geometry/decode/LDS/ownership search and rejected-candidate records.
- Lower-bound and residual-bottleneck analysis.
- Per-key HIP-fallback selection; no GGTensile candidate passed promotion.
- Complete writer line coverage and regression identity gates.
- Recursive final review with no actionable mechanism remaining.
