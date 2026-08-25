# GGTensile MMQ Forward Q8_0 Experiment

## Purpose

Implement and optimize dense MMQ forward Q8_0 assembly kernels for the DeepSeek workload on gfx1151. The campaign must optimize the complete packed-weight multiply, including global reads, Q8_0 decode, Q8_1 activation-workspace consumption, LDS movement, integer WMMA, correction/scaling, synchronization, and BF16 stores.

The existing HIP dense Q8_0 forward kernels are correctness and performance controls. They establish that the operation is feasible, but their templated implementation and generated assembly are evidence only. GGTensile must express the same operation through the repository's typed forward contracts, semantic stages, deterministic register allocation, and explicit mechanism policies rather than copying a large HIP assembly body.

This document is the Q8_0 forward campaign record. Update it after every coherent implementation, correctness, resource, measurement, rejection, or review change. Keep detailed build and timing artifacts under the recorded temporary paths.

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

M512 is the primary complete-loss chunk; M256 is the lower-memory alternative; M32/M64/M128 remain required capacity keys. Ordinary and language-model-head ownership are separate campaign families even when a formula-compatible lowering can be shared.

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

The canonical catalog has typed Q8_0 direct-global, LDS-free register-tiled, HIP-shaped LDS, exact small-M LDS, and compact depth32 row controls. The composed `CompactDepth32WeightRows` source is selected for 20 exact catalog keys; the ordinary Q-A M2048 key remains on `HipTile`, and LM-head M32/M64 remain on their exact small-M controls. Public runtime dispatch and the generated 179-kernel bundle remain unchanged.

Existing HIP controls are available through the dense forward bundle sources and dispatch in `csrc/mmq_bundle.cpp` and `csrc/mmq_core.cuh`. Existing Q8_0 GGTensile artifacts under `~/tmp/torch-ggml-ops/` are backward artifacts and are not forward baselines.

Baseline preservation checkpoint: all currently valid catalog sources were regenerated before Q8 edits into `~/tmp/torch-ggml-ops/q8-fwd-baseline-sources-20260805/`. The manifest contains 447 unique valid sources: 155 Q4_K/Q5_K/Q6_K forward sources and 292 Q3_K/Q4_K/Q5_K/Q6_K/Q8_0 backward sources. Every retained Q8 change must reproduce this set byte-for-byte unless an existing-stream change is explicitly separated and qualified.

This experiment record preserves the fresh same-process HIP multiply medians and selection chronology for all 23 exact keys. The deployment catalog contains four deduplicated winner specifications and 23 exact mappings: 20 use the composed compact depth32 row mechanism, one uses the ordinary HIP-shaped LDS mechanism, and two use exact LM-head small-M controls. The catalog is qualified for offline GGTensile generation, while public HIP dispatch remains the explicit runtime path until a separate bundle integration review.

The initial isolated backend lowers one wave to a `16x16` output tile. It directly loads four 32-value Q8_0 blocks and one 128-value Q8_1 F32_D4 block per reduction-loop iteration, performs eight signed integer WMMAs, and applies the FP32 correction in the HIP expression order `integer_result * weight_scale * activation_scale` before BF16 RNE stores. `SignedInt8MmaGroupRole` carries the payload and scale offsets, while `SignedInt8DirectRegisterPlan` allocates explicit lifetime-bound roles deterministically. This branch does not use LDS and does not alter the Q4_K/Q5_K/Q6_K body methods.

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

The retained register-tiled source then hoisted the four fixed Q8_0 block offsets into `SignedInt8MmaGroupRole` and advanced the per-lane packed addresses by 136 bytes per activation block. The Q-A artifact remained bit-exact with 137 VGPRs, 16 SGPRs, zero LDS, and 136 VMEM instructions; its warmed multiply diagnostic improved to approximately `0.8714 ms` (`1.3674x` HIP). A subsequent linear store traversal computes the first output address once and advances by fixed row/column deltas. It removes 37 static VALU instructions and stayed correct in the short recovery check at `0.8592 ms`; a rotating comparison reported `0.81055 ms` for a scale-overlap experiment against `0.80945 ms` for the retained parent, so only the linear traversal is retained.

Three follow-up scheduling tests were rejected. The first store `s_clause` was invalid because address updates occurred between stores and caused the first launch to fail to complete; its artifact is quarantined. Moving weight-scale loads to the front of each clause and partially waiting for scale conversion was bit-exact but neutral in a rotating 25-repeat comparison (`1.00136x` parent). Advancing the packed-weight and activation scalar base pointers instead of 20 per-lane VGPR addresses was also bit-exact, but the loop-carried scalar dependency regressed to `1.02210x` parent. Both schedules were removed.

The store-clause mechanism was then repaired by materializing eight output addresses in the dead WMMA product registers before each fragment's contiguous store sequence. The corrected form is bit-exact, remains at 137 VGPRs/16 SGPRs with zero LDS, and has eight static clauses. Two independent rotating 25-repeat comparisons measured candidate/parent median ratios of `0.96645x` and `0.96411x`. A follow-up reused all 32 dead product registers to issue one 32-store clause; it remained exact and reduced static clauses from eight to five, with two rotating comparisons against the four-clause-store parent at `1.00111x` and `1.00034x`. The one-clause epilogue is retained as a resource-neutral simplification. The active writer therefore uses the simple 26-read clause, full wait, expression-order correction, and address-materialized 32-store clause.

An offline no-extra-register payload pipeline then moved each next group's weight and activation payload reads immediately after the current operands' last WMMA use, while loading the next scales after the current correction. It kept the exact `137/16/0` resources, 136 VMEM instructions, five waits, and five clauses, and remained bit-exact. A rotating 25-repeat comparison measured `0.83267 ms` versus `0.78333 ms` for the retained parent (`1.06299x`), so the schedule was rejected. Splitting the 26-read clause and injecting VMEM into the WMMA/correction stream costs more than the hidden payload latency saves.

An offline pressure experiment packed pairs of FP16 weight scales into eight VGPRs and retained one packed workitem coordinate instead of separate lane and wave registers. The linked artifact declared 128 VGPRs with no reference above `v127`, remained bit-exact, and used on-demand low/high scale conversion to preserve scale reuse across the two activation fragments. The extra extraction arithmetic dominated: rotating medians were `0.86205 ms` versus `0.78846 ms` parent (`1.09334x`). The 128-VGPR form was rejected; pressure reduction is not useful when it duplicates scale conversion in every product fragment.

A second offline 128-VGPR experiment preserved the original FP32 scale reuse by keeping eight scale-address VGPRs and alternating those addresses between the two N fragments across the four groups. It also used one packed workitem coordinate, declared 128 VGPRs/16 SGPRs, referenced at most `v127`, and had zero LDS/private storage/spills. The form remained bit-exact, but the added address dependencies and split read clauses were neutral-to-slower: rotating medians were `0.78901 ms` versus `0.78665 ms` parent (`1.00301x`). The mechanism was rejected because it adds substantial semantic complexity without a target gain.

Store-only `s_setprio` values 1, 2, and 3 were screened offline after the reduction loop. All three were bit-exact and resource-neutral, but rotating 21-repeat ratios were `1.00581x`, `1.00190x`, and `1.00550x` parent. The default priority remains retained.

Moving each next fragment's eight zero-initializations immediately after the previous fragment's second WMMA was also bit-exact and resource-neutral. The intended result-latency coverage did not materialize: rotating medians were `0.78377 ms` versus `0.78089 ms` parent (`1.00369x`). The original local WMMA/correction order remains retained.

The 20 packed-weight, scale-address, and activation-address advances from the last group were distributed five at a time after each second WMMA to remove the serial loop tail. The schedule remained exact and resource-neutral, but VALU issue contention outweighed latency coverage: rotating medians were `0.78884 ms` versus `0.78322 ms` parent (`1.00717x`). Address advancement remains after the correction body.

Deferring the eight n1 FP16-to-FP32 scale conversions until immediately after the first fragment's second WMMA was exact and resource-neutral, but rotating medians were `0.79378 ms` versus `0.79214 ms` parent (`1.00207x`). Scale conversion placement is not retained.

The retained Q-A checkpoint is `~/tmp/torch-ggml-ops/q8-fwd-register-tile-store-clause-all/m2048-n1024-k4096/`. Strict inspection reports code-object v5, gfx1151, wave32, the 40-byte ABI, 137 VGPRs, 16 SGPRs, zero LDS/private storage/spills, 32 WMMAs, 136 VMEM instructions, 784 VALU issues, five waits, and five clauses. It is bit-exact with HIP/public for all 2,097,152 BF16 outputs, passes every mutation gate, and matches the independent-reference envelope at normalized RMSE `0.00604494`. Two independent rebuilds reproduced matching generated source and normalized inspection results.

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

## Reopened Performance Plan (Initial control complete; parity work reopened)

The fallback result reopened the Q8_0 research campaign because the HIP kernel demonstrates a different complete dataflow that the direct/register controls do not express: a 128-thread `I=64, J=128` tile, cooperative Q8_0 payload and FP16-scale decode into LDS, activation reuse across eight output-column fragments, 32 WMMAs per reduction stage, and dual-issue scale/epilogue work. That dataflow has now been implemented through typed ownership and LDS roles without importing HIP's physical instruction order.

The work is staged as follows:
- Record the HIP tile mapping, LDS byte domains, local-read coordinates, accumulator ownership, and static lower bounds from the source and normalized disassembly.
- Add one isolated `Q8HipTiledLds` solution family with formula-derived `I=64, J=128` ownership, LDS layout, resource admission, and deterministic register lifetimes. Keep the existing direct and register controls intact.
- Validate the ordinary Q-A candidate at full output size against HIP/public and the independent reference, including all input, packed-weight, and Q8_1-workspace mutations, strict ABI/resource inspection, and deterministic rebuilds.
- Measure warmed complete and multiply-only Q-A timings against the same HIP kernel. Use the result to tune LDS ownership, local-read reuse, load grouping, scale conversion placement, dual issue, and the 64x128 epilogue as linked mechanism bundles.
- Confirm every gain or parity result on representative ordinary families and LM-head chunks before changing any exact-key fallback. A candidate that is only statically smaller or only faster on a partial body is not a promotion.
- Repeat the recursive review after the HIP-shaped mechanism and all actionable ownership findings have been implemented and measured.

The immediate performance gate is Q-A `M=2048,N=1024,K=4096`: the candidate must beat the warmed HIP multiply median, remain exact, and retain zero spills, private storage, scratch, calls, and dynamic stack. The secondary gate is parity or better on the highest-cost Q-B and attention-output-B keys. Production fallback decisions remain unchanged until those gates pass.

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

Selection requires repeated warmed multiply evidence that the exact GGTensile source is faster than HIP; there is no fixed minimum speedup. The earlier 2% resource-bearing threshold remains useful as a confidence diagnostic but is not a hard selection gate. LDS/VGPR pressure and static instruction counts may identify a bottleneck, but they are not optimization targets and never override multiply timing. A source that is smaller but slower than its retained parent is rejected.

## Active Parity Reopening

The Q8_0 campaign is reopened because the HIP control proves that the current GGTensile gap is not a feasibility boundary. The active objective is to find a complete, correct, resource-clean GGTensile dataflow at or below the HIP median on the exact production keys, beginning with the largest remaining operand-reuse and decode gaps. Existing HIP fallback selections, the 23-key inventory, public dispatch, public bundle, and frozen source snapshot remain unchanged until promotion gates pass.

New work must start from a changed ownership, representation, or dataflow premise. The already rejected schedule-only variants remain closed unless a prerequisite changes. Every candidate still requires exact HIP/public agreement, independent-reference and mutation checks, strict zero-spill inspection, deterministic rebuilds, and warmed rotating timing. A candidate that beats HIP on only a partial body or one unrepresentative shape is not promotable.

The recursive final review in this document remains the final campaign step. After the active mechanisms are exhausted, reread the complete Q8/Q3/GGTensile/HIP evidence, classify every remaining mechanism as retained, rejected, incompatible/deferred, or actionable, implement and measure every actionable item, and repeat the review after any new premise. The campaign is not complete until that fresh final review finds no actionable in-contract mechanism and explains the residual bottleneck.

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
- The initial per-key HIP-fallback selection was reopened for the HIP-shaped performance campaign.
- Complete writer line coverage and regression identity gates remain mandatory for the isolated family.
- The activation-read-address-hoisted `Q8HipTiledLds` source was selected for 20 exact catalog keys after two independent 25-repeat confirmations per key; the later exact small-M follow-up selected LM-head M32/M64, leaving only KV M2048 on HIP fallback before the compact exact-key review.
- Public runtime dispatch, generated bundle packaging, and the frozen 447-source boundary remain unchanged pending a separate public integration review.

### Initial Reopened HIP-Shaped Control Result

The isolated `Q8HipTiledLds` lowering uses a wave32 workgroup `(32,4,1)`, a `128x64` macro tile, cooperative Q8_1 activation and packed Q8_0 weight staging in LDS, wave-N ownership, 64 integer WMMAs, typed scale copies for legal VOPD correction, and one 64-store BF16 RNE epilogue. It is a research control only: it is not present in the production catalog, exact-key inventory selections, public dispatch, or the 179-kernel bundle.

The full Q-A gate `(M,N,K)=(2048,1024,4096)` is bit-exact with HIP multiply and the public complete path across all 2,097,152 outputs. Input, packed-weight, and workspace mutations remain sensitive, and the independent packed/dequantized reference remains within normalized RMSE `0.00604494` with maximum absolute error `0.0625`. Strict inspection reports code-object v5, gfx1151, wave32, the 40-byte ABI, `240` VGPRs, `16` SGPRs, `38,400` bytes of LDS, zero private bytes, zero VGPR/SGPR spills, `64` WMMAs, and `2` barriers.

The then-final warmed rotating 25-sample comparison measured a `0.675784 ms` HIP complete median versus `0.685826 ms` GGTensile complete (`1.01486x`), and a `0.617289 ms` HIP multiply median versus `0.644354 ms` GGTensile multiply (`1.04384x`). At that stage the candidate appeared to miss both parity gates, so no Q8 key changed selection. The Q-A measurement-error reconciliation below supersedes this single-enqueue timing conclusion while preserving it as historical evidence.

The initial phase gates are complete: focused coverage and the full repository suite pass (`160` and `359` tests respectively, with the existing `14` warnings), Ruff/formatting/`ty`/compileall/pre-commit pass, the public bundle remains current at `179` kernels, frozen pre-Q8 regeneration reports `447/447` byte-identical sources, and two independent CLI rebuilds reproduced matching generated source and normalized inspection results. Detailed artifacts are under `/tmp/q8-hip-test-artifact/` and `/tmp/q8-hip-deterministic-{a,b}/`.

## DepthU=64 And Persistent-Zero Follow-Up

The HIP-shaped control was reopened with two linked research changes. First, `DepthU=64` stages eight Q8_0 groups and two Q8_1 activation planes per reduction iteration, advances the packed-weight and activation pointers by the widened depth, and halves the loop count for divisible shapes. The intended three-barrier schedule was not legal for the current ownership: the second activation-plane staging still requires a separate barrier, so the emitted control has four barriers. The `DepthU=64` source is an ordinary `ForwardSolution` identity and remains isolated from production catalogs.

Second, the depth32 and depth64 controls retain one typed eight-VGPR zero WMMA operand across the reduction loop instead of reinitializing each product fragment. The two activation-scale-copy registers are reused by parity of the M fragment, and setup-only staging payload/address roles are packed into dead low-register ranges. The resulting plan allocates 235 registers and the artifact remains in the declared `240 VGPR / 16 SGPR / 38,400-byte LDS` class. A probe that restored the staging payload to its former high-register range failed correctness because the asynchronous load still overlapped the scale-address value; that placement is rejected.

The depth32 persistent-zero control inspects at 64 WMMAs, 2 barriers, 82 VMEM operations, 154 LDS operations, 40 waits, and 3 clauses, with 900 VALU issues, 1,208 VALU operations, and 308 VOPD instructions. The depth64 control inspects at 128 WMMAs, 4 barriers, 100 VMEM operations, 308 LDS operations, 79 waits, and 5 clauses, with 1,534 VALU issues, 2,114 VALU operations, and 580 VOPD instructions. Both have zero private bytes and zero VGPR/SGPR spills. The depth64 source remains exact across the full Q-A output and all producer-repeat, input, packed-weight, workspace, finiteness, and public-path checks; its independent-reference normalized RMSE is `0.00606098` with maximum absolute error `0.0625`.

The warmed results are shape-dependent:

| Control and shape | HIP multiply median | GGTensile multiply median | Multiply ratio | Complete ratio |
| --- | ---: | ---: | ---: | ---: |
| Persistent zero, depth32, M2048 Q-A, 25 repeats | `0.61000 ms` | `0.63626 ms` | `1.04305x` | `0.98478x` |
| Persistent zero, depth64, M2048 Q-A, 25 repeats | `0.59913 ms` | `0.62177 ms` | `1.03779x` | `0.98763x` |
| Persistent zero, depth32, M8192 Q-A, confirmation A | `2.47521 ms` | `2.43440 ms` | `0.98351x` | `0.96571x` |
| Persistent zero, depth32, M8192 Q-A, confirmation B | `2.49921 ms` | `2.44065 ms` | `0.97657x` | `0.96818x` |
| Persistent zero, depth64, M8192 Q-A | `2.46226 ms` | `2.42491 ms` | `0.98483x` | `0.96798x` |
| Persistent zero, depth32, M32768 Q-A, nine repeats | `9.61868 ms` | `9.63052 ms` | `1.00123x` | `0.99237x` |

The M8192 depth32 result is a real shape-specific lead: the committed parent measured `1.01760x` multiply and `0.99899x` complete in a separate 25-repeat confirmation, while the persistent-zero candidate measured below HIP in both candidate confirmations. The same control is materially slower at M512 (`1.18511x` multiply) and M2048, and it is neutral at M32768. The depth64 widening does not improve on depth32 and doubles the static reduction body. The candidate therefore remained research-only at that stage: no exact-key selection, inventory, public dispatch, generated bundle, or production fallback changes were justified. The subsequent activation-hoist review reopened the catalog decision.

The post-follow-up frozen-source comparison still reports `ExpectedCount=447`, `GeneratedCount=447`, and `ChangedCount=0`; the new isolated writer controls do not alter the existing Q4/Q5/Q6/Q8 frozen streams. The focused suite passes 182 tests and the full repository suite passes 372 tests with the existing 14 warnings. Ruff, formatting, `ty`, compileall, pre-commit, diff checks, and the 179-kernel bundle-current check pass, and independent depth32/depth64 rebuilds reproduce identical generated source and normalized inspection results. Qualification artifacts are under `/tmp/q8-persistent-zero-depth32/`, `/tmp/q8-persistent-zero-depth64/`, `/tmp/q8-pz-m8192-d32/`, `/tmp/q8-pz-m32768-d32/`, and `/tmp/q8-pz-m512-d32/`.

## Activation LDS Read-Address Hoist

The depth32 HIP-shaped control was reopened for the two highest-cost ordinary families. Each reduction group previously rebuilt the same eight lane-local activation LDS row addresses with eight masks, seven adds, and eight multiplies. The retained change allocates a typed `activation_read_address` role, aliases it with the dead setup-only `activation_row` register, computes `144 * (lane & 15)` once, and expresses every M-fragment row through the DS immediate offset. The register plan still allocates 235 registers and the linked artifact remains in the `240 VGPR / 16 SGPR / 38,400-byte LDS` class with zero private storage and spills.

The first hoisted source was faster but invalid. It retained the old `lgkmcnt` thresholds, which guaranteed the even M fragment but not the odd activation scale copied in the same VOPD instruction. The removed address chain had accidentally supplied enough latency. The failure was localized to M-fragment rows 16 through 31, with 49,453 differing values in the 2,097,152-element Q-A diagnostic. The corrected schedule waits for both scales before each even/odd copy, using paired thresholds `(18,18,12,12,6,6,0,0)` per group. The corrected artifact is bit-exact with HIP and public output across every screened shape and passes producer-repeat, input, packed-weight, workspace, and finiteness checks.

Static depth32 inspection falls from 900 to 810 VALU issues and from 1,208 to 1,118 VALU operations. The source retains 64 WMMAs, 2 barriers, 82 VMEM operations, 154 LDS operations, 40 waits, 3 clauses, and the same resource class. The initial warmed five-repeat screens are:

| Family and shape | HIP multiply median | GGTensile multiply median | Multiply ratio | Complete ratio |
| --- | ---: | ---: | ---: | ---: |
| Attention output B, M2048 | `5.36976 ms` | `4.46433 ms` | `0.83138x` | `0.83221x` |
| Attention output B, M8192 | `20.06020 ms` | `17.54886 ms` | `0.87481x` | `0.87606x` |
| Attention output B, M32768 | `78.64648 ms` | `70.04539 ms` | `0.89064x` | `0.89457x` |
| Attention Q-B, M2048 | `5.46528 ms` | `4.85398 ms` | `0.88815x` | `0.88489x` |
| Attention Q-B, M8192 | `20.72696 ms` | `18.65777 ms` | `0.90017x` | `0.90030x` |
| Attention Q-B, M32768 | `82.40228 ms` | `75.12784 ms` | `0.91172x` | `0.91067x` |

The M8192 pre-hoist persistent-zero parents measured `19.38755 ms` for attention output B and `19.81874 ms` for Q-B, so the hoist improves the GGTensile multiply medians by approximately 9.5% and 5.9% in the separate warmed screens. The M32768 controls now save approximately 8.6 ms and 7.3 ms per multiply relative to HIP. These are the first broad, workload-significant HIP wins in the Q8 campaign.

Two independent warmed 25-repeat rotations against both HIP and the exact pre-hoist artifact confirm the M8192 result. Attention output B candidate/parent multiply ratios are `0.92118x` and `0.92032x`, with candidate/HIP ratios `0.87558x` and `0.87455x`; complete ratios are `0.92677x` and `0.92483x` parent and `0.87911x` and `0.87784x` HIP. Q-B candidate/parent multiply ratios are `0.93551x` and `0.93798x`, with candidate/HIP ratios `0.90534x` and `0.90599x`; complete ratios are `0.93695x` and `0.93664x` parent and `0.90596x` and `0.90323x` HIP.

The implementation is retained in the isolated writer and covered by focused register-alias, immediate-address, and dependency-wait assertions. Two independent builds reproduce byte-identical source and identical normalized inspection. The GGTensile suite passes 287 tests and the full repository suite passes 372 tests with the existing 14 warnings. Ruff, formatting, `ty`, compileall, pre-commit, and diff checks pass; the frozen comparison remains `447/447` byte-identical and the public bundle remains current at 179 kernels. Exact-key inventory, selected catalogs, public dispatch, and the generated bundle remain unchanged until the remaining shared-family gate is complete. Artifacts are under `/tmp/q8-activation-read-hoist-correct-{output,qb}-m{2048,8192,32768}-d32/`; the exact Q-A recovery diagnostic is `/tmp/q8-activation-read-hoist-correct-qa-m2048-d32/`.

## Post-Hoist Loop-Cost Review

The retained activation-address hoist was used as the exact parent for a focused M8192 review of the remaining LDS and loop-address costs. Every executable candidate preserved exact HIP/public agreement unless explicitly identified below, and every rejected source was removed from the writer.

Paired activation-scale reads reduce the depth32 body from 154 to 138 LDS instructions, but neither legal issue order improves the parent. Issuing each `ds_read2st64_b32` before its two payload rows measures `1.00261x` parent multiply and `1.00264x` complete on attention output B, and `1.00414x` multiply and `1.00038x` complete on Q-B. Issuing the pair after both payload rows initially failed exactness under the old waits; the dependency-correct thresholds `(15,15,10,10,5,5,0,0)` recover exact output but measure `1.00006x` parent multiply and `1.00502x` complete on output B, and `1.00480x` multiply and `1.00073x` complete on Q-B. The paired representation is therefore rejected despite its lower static LDS count.

The current weight scales are 608 bytes apart in LDS. Normal `ds_read2_b32` has an 8-bit offset field and rejects `offset1:608`; the stride-64 form advances in 256-byte units, so it cannot represent 608 either. Paired weight-scale reads require a different scale-plane layout and are not a legal source-only substitution. Hoisting the existing scalar weight-scale read address across K loops is exact but only measures `0.99582x` parent multiply on output B and regresses Q-B to `1.00152x`; its complete ratios are `0.99613x` and `1.00167x`. That added lifetime does not meet the 2% resource-bearing threshold.

The remaining schedule-only probes are also closed. Moving each scalar activation-scale read ahead of its payloads measures `1.00534x` parent multiply and `1.00592x` complete on output B, and `1.00686x` multiply and `1.00411x` complete on Q-B. Batching all eight WMMA pairs before conversion and scale correction is exact but raises the output-B multiply median to `18.44087 ms`, versus approximately `17.5 ms` for the parent, showing that the retained per-fragment correction hides useful latency. Replacing the bank-separated scale copy with the direct same-scale VGPR assembles but fails exactness; the duplicated value in a different source bank is semantically required by the legal VOPD schedule.

Finally, a typed loop-carried staging-address probe moved the invariant LDS payload and scale addresses plus the global weight-stage pointer into `v235:v237`. It removes eleven address instructions per stage invocation, inspects at 808 VALU issues and 1,116 VALU operations, and retains the declared `240 VGPR / 16 SGPR / 38,400-byte LDS` class with zero spills. The K8192 output-B rotation improves to `0.99657x` parent multiply and `0.99402x` complete, but the K1024 Q-B control regresses to `1.00145x` and `1.00189x`. The gain is below threshold, costs three additional live logical VGPRs, and is shape-specific, so the complete staging-address state is rejected. Supporting artifacts are under `/tmp/q8-{paired-activation,paired-activation-tail,weight-address-hoist,activation-scale-first,wmma-batch,scale-stage-address-hoist,stage-address-hoist}-*/`.

## Historical Exact-Key Promotion Review

The activation-read-address hoist was qualified across all 23 inventory keys. The depth32 `Q8HipTiledLds` artifact remained in one inspected resource class: `240` VGPRs, `16` SGPRs, `38,400` LDS bytes, code-object v5, gfx1151, wave32, the 40-byte ABI, zero private bytes, and zero register spills. Across 21 tile-valid keys, the five-repeat audit found zero differing BF16 elements against both HIP multiply and the public complete path, deterministic Q8_1 producer output, and nonzero input, packed-weight, and workspace mutation responses. The independent-reference checks for shared gate/up M2048 and LM-head M128 had normalized RMSE `0.00606097` and `0.00602456`; the existing Q-A reference check remains `0.00604494`.

Each of the 20 promoted keys received two serialized seven-warmup, 25-repeat rotations against the pre-hoist same-policy parent and HIP. The candidate/HIP multiply ratios below are the two confirmation values; both values must clear the 2% resource-bearing threshold.

| Family | M2048 | M8192 | M32768 |
| --- | ---: | ---: | ---: |
| Attention Q-A | `0.9151x` / `0.9277x` | `0.8761x` / `0.8808x` | `0.9029x` / `0.8985x` |
| Attention Q-B | `0.8871x` / `0.8865x` | `0.9077x` / `0.9051x` | `0.9049x` / `0.9092x` |
| Attention KV | HIP fallback | `0.8953x` / `0.8944x` | `0.8827x` / `0.8803x` |
| Attention output B | `0.8426x` / `0.8425x` | `0.8736x` / `0.8742x` | `0.8862x` / `0.8912x` |
| Shared gate/up | `0.9172x` / `0.9087x` | `0.8903x` / `0.8879x` | `0.9086x` / `0.9069x` |
| Shared down | `0.9046x` / `0.9042x` | `0.9013x` / `0.9012x` | `0.9191x` / `0.9199x` |

At the initial common `128x64` promotion checkpoint, the KV M2048 candidate was measured at `0.9878x` and `0.9709x` HIP multiply in its two confirmations, after a five-repeat `1.0042x` screen, so it remained HIP fallback rather than being retained on noise. LM-head M128 and M256 were confirmed at `0.8574x`/`0.8538x` and `0.8663x`/`0.8671x`; M512 was confirmed at `0.8768x`/`0.8811x`. The subsequent exact small-M follow-up qualified LM-head M32/M64, as recorded below.

At this initial promotion checkpoint, the catalog contained one deterministic `hip_tiled_lds_selected` mapping, exact `small_m32_tiled_lds_selected` and `small_m64_tiled_lds_selected` mappings, and one explicit HIP fallback. The generated public bundle remained at 179 kernels, public dispatch remained unchanged, and the frozen Q8 baseline comparison remained `ExpectedCount=447`, `GeneratedCount=447`, `ChangedCount=0`. The exact KV fallback was reopened as a separately reviewed remaining-key campaign; this research catalog does not silently change runtime dispatch.

## Remaining-Key Follow-Up

The two LM-head fallbacks were reopened with a distinct wave-N small-M ownership. The exact controls use workgroup `(32,4,1)`, `N` macro tile 64, `DepthU=32`, and per-wave M tiles of `(2,1)` for M32 and `(4,1)` for M64. Each activation row is cooperatively staged across four ways for M32 or two ways for M64; the LDS layout places `M*144` activation bytes before 64 padded 304-byte weight rows. Formula-derived declarations are `96 VGPR / 16 SGPR / 24,064 LDS` for M32 and `144 / 16 / 28,672` for M64, while deterministic logical plans use 91 and 139 registers.

Both exact controls passed strict code-object-v5/gfx1151/40-byte-ABI inspection with zero private storage, spills, scratch, calls, and dynamic stack. M32 used 16 WMMAs, two barriers, 25 VMEM, 73 LDS, 13 waits, and three clauses; M64 used 32 WMMAs, two barriers, 44 VMEM, 100 LDS, 22 waits, and three clauses. HIP/public outputs were identical across all outputs and all mutation, finiteness, repeatability, and independent-reference gates passed. Independent-reference normalized RMSE was `0.006066967333` for M32 and `0.006039657922` for M64.

Two serialized seven-warmup, 25-repeat confirmations per key cleared the resource-bearing 2% gate. Throughput is equivalent dense throughput, calculated as `2*M*N*K / median_time`; speedup is `HIP median / GGTensile median`, so values above `1.0x` are faster than HIP.

| Key | Confirmation | Multiply TFLOPS | Multiply speedup vs HIP | Complete TFLOPS | Complete speedup vs HIP |
| --- | --- | ---: | ---: | ---: | ---: |
| LM-head M32 | A | `13.0520` | `1.0967x` | `13.1203` | `1.1031x` |
| LM-head M32 | B | `13.1516` | `1.0990x` | `13.0247` | `1.0971x` |
| LM-head M64 | A | `25.4823` | `1.1170x` | `25.3375` | `1.1126x` |
| LM-head M64 | B | `25.3601` | `1.1057x` | `25.3080` | `1.1070x` |

Independent M32/M64 roots reproduced source, object, code object, and normalized inspection byte-identically.

The initial remaining-key review left Attention KV M2048 on HIP fallback. Its common `Q8HipTiledLds` candidate did not clear the promotion gate, and the terminal-LDS rotation was rejected as a sub-percent parent movement that regressed shared Q-B M8192. The compact exact-key follow-up below reopened this one inventory entry; all other catalog selections and the public bundle remained unchanged.

The first KV latency experiment elided terminal LDS reuse work. The depth32 loop entered the reduction body through one prologue branch, and a taken continuation branch passed through the LDS reuse barrier and both packed-weight/activation pointer advances before the next iteration. The final iteration fell directly into the epilogue because no later stage could overwrite LDS. The artifact remained bit-exact across all 1,048,576 KV M2048 outputs and every mutation gate, and inspection retained the `240 VGPR / 16 SGPR / 38,400-byte LDS` class with unchanged static compute and memory counts.

The mechanism is rejected from the common writer. Two 25-repeat KV rotations measured candidate/HIP multiply ratios of `0.97623x` and `0.97277x`, but candidate/parent ratios were only `0.99043x` and `0.99932x`. More importantly, the shared Q-B M8192 representative regressed to `1.00907x` parent multiply and `1.00792x` parent complete. The source and focused assertions were restored to the selected activation-hoist parent; the result does not justify an exact-key specialization based on sub-percent parent movement. Artifacts remain under `/tmp/q8-final-barrier-{kv-m2048,qb-m8192,output-m8192}/`. Catalog selections and public dispatch remain unchanged.

The exact small-M ownership was then screened on KV M2048. Reusing the M32 control remained exact at `96 VGPR / 16 SGPR / 24,064-byte LDS`, but one nine-repeat rotation measured `0.99273x` HIP multiply and `0.95427x` complete. M64 remained exact at `144 / 16 / 28,672`; two 25-repeat rotations measured `0.98235x` and `0.99029x` HIP multiply, with complete ratios `0.92354x` and `0.93126x`. M64 therefore becomes the remaining-key optimization baseline: it matches or beats HIP, materially reduces complete latency and resources, but does not yet clear the existing 2% resource-bearing promotion gate in both confirmations. Artifacts are under `/tmp/q8-kv-small-{m32,m64}-probe/`.

A transposed weight-scale LDS plane tested whether the M64 baseline was limited by its eight 608-byte-stride scalar scale reads per group. The exact layout appended 1,024 bytes, grouped each lane's eight scales contiguously, and raised LDS to 29,696 bytes. Two `ds_read_b128` loads per group reduced static LDS instructions from 100 to 76 but measured `1.01985x` HIP multiply and `0.94868x` complete. Four paired `ds_read2_b32` loads per group retained broadcast-friendly 32-bit accesses and reduced static LDS instructions to 84; it measured `1.00135x` HIP multiply and `0.92771x` complete, versus `0.99855x` and `0.94104x` for a contemporaneous retained-source rotation. Neither form improves multiply latency enough to justify the larger LDS allocation and extra transposition address work, so the scale plane is rejected and the original padded-row scale layout is restored. Artifacts are under `/tmp/q8-kv-small-m64-scale-{plane-v2,paired}/`.

The final-barrier rotation was repeated on the M64 ownership rather than the common M128 writer. It retained exact output and the `144 VGPR / 16 SGPR / 28,672-byte LDS` class, with the LDS reuse barrier and pointer advances executed only when another reduction block followed. A nine-repeat screen measured `1.00273x` HIP multiply. Two direct 25-repeat parent/candidate/HIP rotations then measured candidate/parent at `1.00474x` and `0.98854x`, with candidate/HIP at `0.99988x` and `0.96269x`. The opposite parent movements reproduce the earlier terminal-barrier instability and do not establish a durable source improvement, so the retained loop topology is restored. Artifacts are under `/tmp/q8-kv-small-m64-terminal-barrier/`.

The strongest compact ownership experiment specialized the depth-32 weight LDS rows instead of reserving the depth-64 304-byte layout. Four 32-byte payload groups occupy bytes 0 through 127, four FP32 scales occupy bytes 128 through 143, and the row stride becomes 144 bytes. M64 LDS falls from 28,672 to 18,432 bytes with unchanged `144 VGPR / 16 SGPR`, 32 WMMAs, 44 VMEM, 100 LDS instructions, 22 waits, and two barriers. The lower allocation admits three resident workgroups instead of two. The first exact nine-repeat screen measured `0.94452x` HIP multiply and `0.87933x` complete. Two warmed 25-repeat parent/candidate/HIP rotations measured candidate/HIP multiply at `0.94598x` and `0.93719x`, equivalent to `1.0571x` and `1.0670x` speedups; candidate/parent was `0.93861x` and `0.98026x` despite visible clock drift.

The adjacent compact M32 ownership used `96 VGPR / 16 SGPR / 13,824-byte LDS` and remained exact, but its nine-repeat screen measured `0.96272x` HIP multiply and `0.89205x` complete. Its `0.32526 ms` multiply median trails compact M64's corresponding `0.32049 ms` while launching twice as many workgroups, so M32 is rejected for KV. The exact M64 compact layout advanced to formal qualification. Artifacts are under `/tmp/q8-kv-small-m{32,64}-compact-lds144/`.

The compact-row padding review kept the three-workgroup residency class. A 176-byte aligned row restored the original bank-step residue at 20,480 LDS bytes and remained exact; its screen measured `0.94782x` HIP multiply and `0.87987x` complete. A direct 25-repeat rotation against 144 bytes measured `1.00031x`, so the extra 2,048 LDS bytes have no performance value. A 148-byte row used a coprime bank step at 18,688 LDS bytes but made 128-bit row accesses unaligned; despite exact output, it regressed to `1.30941x` HIP multiply and `1.22231x` complete. The minimum aligned 144-byte row advanced as the candidate baseline. Artifacts are under `/tmp/q8-kv-small-m64-compact-lds{148,176}/`.

An eight-wave `M64xN128` compact tile was the final ownership review. It used `144 VGPR / 16 SGPR / 27,648-byte LDS`, 32 WMMAs, 41 VMEM, 97 LDS instructions, 21 waits, and two barriers. Doubling N ownership halves activation staging per output and permits two workgroups, or 16 resident waves. After deriving both weight and store workgroup strides from N128, the artifact was exact, but its nine-repeat screen measured `1.06580x` HIP multiply and `1.00950x` complete. The larger workgroup loses more scheduling flexibility than its activation reuse saves, so the four-wave `M64xN64` compact tile remained the ownership baseline. Artifacts are under `/tmp/q8-kv-compact-m64n128-wg8-v2/`.

The scalar-scale KV candidate used the compact `M64xN64` ownership with 64 activation rows of 144 bytes and 64 weight rows of 144 bytes. Each weight row held four 32-byte payload groups followed by four FP32 scales at byte offset 128; the row stride was 144 bytes. The formula-derived artifact used `18,432` LDS bytes, `144 VGPR`, `16 SGPR`, 32 WMMAs, 44 VMEM instructions, 100 LDS instructions, 22 waits, three clauses, and two barriers. It had zero private storage, spills, scratch instructions, calls, and dynamic stack. The exact output, independent-reference, producer, mutation, finiteness, repeatability, and strict inspection gates passed.

A 160-byte aligned row was screened as a bank-padding alternative. It raised LDS to `19,456` bytes without changing the instruction mix; the first screen and two rotations were approximately `0.965x`, `0.9866x`, and `0.9768x` HIP multiply. The weight-first variant also produced contradictory warmed rotations, including one at `1.0407x` HIP. The extra padding has no qualified value, so the 160-byte form is rejected and the 144-byte layout remained the candidate baseline. The 148-byte and 176-byte row results remain rejected as recorded above; the 192-byte aligned form was likewise screened and regressed without a resource-bearing gain.

The temporary compact HIP-shaped `M128xN64` branch was also closed. It linked at `240 VGPR / 16 SGPR / 27,648-byte LDS`, with 64 WMMAs, 82 VMEM instructions, 154 LDS instructions, and 40 waits. Its nine-repeat screen measured `0.96845x` HIP multiply, but two serialized rotations measured `1.0487x` and `1.0286x`; the common M128 ownership does not provide a stable gain for this key. The separate eight-fragment small-M allocator requires 243 logical registers and is outside the 240-VGPR resource limit. A depth64 small-M prototype fit the declared resource class but failed correctness with normalized RMSE approximately `0.0285`, so both mechanisms are permanently rejected.

A weight-first schedule issues the six packed-weight VMEM loads before the six activation VMEM loads. It then commits the weight rows with `vmcnt(6)` before committing activation rows with the dependency-derived `(3,0)` waits for M64; no unrelated wait is removed. This changes only independent load order and legal LDS-write ordering, preserves the arithmetic order and all static compute/memory counts, and is shared with the exact LM-head small-M controls. The scalar-scale 144-byte KV candidate's warmed rotating 25-repeat multiply ratios were `0.9416x`, `0.9592x`, `0.9822x`, `0.9443x`, and `0.9422x` HIP, with zero differing BF16 elements in every rotation. A larger serialized 50-repeat rotation on a byte-identical rebuild measured `0.98563x` HIP multiply. All six measurements are faster than HIP, so the source qualifies under the final evidence-based selection rule even though the 50-repeat result is within 2%.

The shared LM-head controls were requalified against the weight-first writer. M32 remains at `96 VGPR / 16 SGPR / 24,064 LDS` and M64 at `144 VGPR / 16 SGPR / 28,672 LDS`, both with zero spills and exact HIP/public output equality. Fresh nine-repeat screens measured multiply ratios of `0.9087x` and `0.8977x`, respectively, with complete ratios `0.9030x` and `0.8966x`. Rebuilt assembly, object, code object, solution key, and normalized inspection match the timed weight-first artifacts byte-for-byte.

The compact 144-byte row makes a direct weight-scale pairing legal that was impossible at the standard 608-byte scale stride. A second LDS base at `weight_scale_address + 1152` allows four `ds_read2_b32` instructions to replace eight scalar scale reads per group with legal dword offsets no larger than 251. This reduces static LDS instructions from 100 to 84 without changing `144 VGPR / 16 SGPR / 18,432 LDS`, WMMA, VMEM, wait, clause, or barrier counts. Two 25-repeat confirmations measured `0.99445x` and `0.99468x` of the scalar parent and `0.92530x` and `0.92903x` HIP. Hoisting the second LDS base out of the four-group body reduced static VALU issues to 443 and measured `0.98332x` and `0.99896x` of the unhoisted paired parent, with candidate/HIP ratios `0.95294x` and `0.95244x`. The direct paired-read source and invariant address hoist are retained.

The changed compact premise triggered a second recursive review. Pairing activation scales with `ds_read2st64_b32` reduced static LDS instructions again, from 84 to 76, but measured `1.00577x` and `1.01952x` of the retained weight-paired parent; it is rejected because multiply latency regressed. Applying the retained direct weight pairing to compact M32 reduced that artifact to 57 LDS instructions, but it measured `1.02032x` of compact M64 and `1.04230x` HIP, so M64 ownership remains final. Padded rows cannot encode the same two-base paired reads without extra address updates, while M128, M64xN128, depth64, transposed scale planes, terminal-barrier changes, and direct/register-tiled ownership were already rejected by timing, resources, or correctness.

The final qualification artifact has 84 LDS instructions, 443 VALU issues, 599 VALU operations, 32 WMMAs, 44 VMEM instructions, 22 waits, three clauses, and two barriers. Strict inspection reports code-object v5, gfx1151, wave32, the 40-byte ABI, `144 VGPR`, `16 SGPR`, `18,432` LDS bytes, zero private storage, and zero spills. It is bit-exact with HIP multiply and public complete output across 1,048,576 BF16 elements, remains exact under input, packed-weight, and workspace mutations, and has independent-reference normalized RMSE `0.00604597`. A fresh nine-repeat qualification measured `0.95025x` HIP multiply and `0.92284x` complete. Two clean final roots reproduce assembly, object, code object, solution key, and inspection byte-for-byte.

The pre-composition recursive review found no actionable in-contract multiply mechanism. Its retained source combined the best measured M64 ownership, minimum aligned compact row, three-workgroup LDS residency, weight-first VMEM schedule, direct paired weight-scale reads, and invariant paired-read address. This was the valid pre-reopening checkpoint; the bounded M128/ordinary composition review below supersedes its exact-key catalog conclusions.

The pre-composition gates passed with the then-current focused/full suite, frozen-source, and public-bundle checks. Those historical counts remain recorded here; the current final verification is recorded in the compact-depth32 requalification section below.

The pre-composition Q8 research catalog had 23 selected GGTensile entries and no inventory fallback. That catalog state is superseded by the exact-key decisions in the compact-depth32 requalification section: 20 compact-row mappings, one ordinary `HipTile` control, and two LM-head small-M mappings. Public dispatch and the generated bundle remain unchanged, and unsupported shapes continue to reject to HIP.

## Cross-Campaign Reopening Review

The final compact-KV result remains valid for its exact selected key. A later cross-format review identified one bounded changed premise that was not measured before the compact M128 rejection: the rejected M128 compact row was tested before the later weight-first VMEM order, direct paired weight-scale reads, and invariant paired-scale-base hoist were established on compact M64.

### Compact depth32 row composition

The candidate to revisit is a typed `CompactDepth32WeightRows` dataflow combining:
- a 144-byte depth32 weight row rather than the padded 304-byte row;
- weight-first global-read and LDS-write ordering;
- legal paired weight-scale reads using a second LDS base;
- invariant hoisting of that second base;
- the existing wave-N ownership, integer WMMA sequence, correction order, and BF16 epilogue.

The old M128 compact candidate used `240 VGPR / 27,648 LDS` and lost in two serialized rotations, so this is not a presumptive generalization. First test one ordinary high-cost output-B key and one short-K Q-B key, then the exact LM-head M32/M64 shapes. Require exact-key correctness, mutations, finiteness, strict resources, deterministic rebuilds, and warmed parent/HIP comparisons before any catalog decision. The existing KV result does not authorize neighboring shapes.

The current `KvCompactTile` name and exact `(2048,512,4096)` validation combine mechanism capability with selection qualification. If the composed layout survives, split those concepts into a formula-derived depth32 compact layout and an explicit exact-key inventory, while retaining independent qualification for every shape. Do not remove the exact-key boundary merely to make the implementation appear generic.

At this checkpoint the Q8 candidate-domain tooling was incomplete. The current writer now exposes a bounded Q8 domain containing the direct-global, register-tiled, HIP-tiled DepthU32/DepthU64, exact small-M, and compact-DepthU32 complete policies. It intentionally has no free knob groups and does not reopen the already rejected DepthU64, terminal-barrier, transposed-scale, or broad geometry searches without a new premise. The earlier statement that no Q8 domain exists is stale chronology, not an open kernel mechanism.

The recursive final-review rule remains global rather than limited to Q8 forward. It rereads related Q3/Q4/Q5/Q6 and backward evidence, and findings may transfer across directions, quant types, and shapes only after the receiving physical layout, lane ownership, synchronization, arithmetic, resource, and exact-key gates are satisfied. The bounded compact-depth32 reopening below satisfies those gates for its retained exact keys and updates the research catalog only; public dispatch and bundle integration remain separate work.

## Compact Depth32 Row Composition Requalification

The bounded reopening tested the changed premise identified above: the earlier common M128 compact-row source was measured before the later compact-M64 dataflow had established weight-first staging, legal paired weight-scale reads, and the invariant second scale base. The new typed identity is `ForwardSolution.q8_0_compact_depth32_tiled_lds()` with `LdsAddressHoist="CompactDepth32WeightRows"`. It retains the existing wave-N ownership, WG32x4, N64, DepthU32, signed WMMA correction order, and BF16 epilogue while changing only the typed LDS row layout and linked staging/read order.

For M128, the formula-derived layout is 144-byte activation rows followed by 64 144-byte weight rows: 18,432 activation bytes, a 9,216-byte weight plane, weight scales at byte offset 128, 288-byte scale-element stride, and a 1,152-byte paired-scale base delta. Total LDS falls from 38,400 to 27,648 bytes. The same identity derives 13,824 bytes for M32 and 18,432 bytes for M64; those small-M variants were tested separately and rejected on timing. The M128 artifact inspects at 240 VGPRs, 16 SGPRs, zero private bytes, zero spills, 64 WMMAs, two barriers, 82 VMEM instructions, 138 LDS instructions, 40 waits, and three clauses. The parent has the same VGPR/SGPR class but 38,400 LDS bytes and 154 LDS instructions. All retained artifacts use the exact 40-byte ABI, code-object v5, gfx1151, and wave32.

The two ordinary gate representatives were bit-exact against both the selected parent and installed HIP across every BF16 output. The Q8_1 producer repeated deterministically; all outputs were finite; input, packed-weight, and workspace mutations changed the candidate output while the mutated candidate remained bit-exact with HIP. The output-B mutation record is `~/tmp/torch-ggml-ops/q8-compact-depth32-all/attention_output_b-m8192-n4096-k8192/mutation.json`; the Q-B record is `~/tmp/torch-ggml-ops/q8-compact-depth32-all/attention_q_b-m8192-n32768-k1024/mutation.json`.

The ordinary nine-repeat screen was followed by two serialized seven-warmup, 25-repeat rotations for every favorable exact key. Launch order alternated within one process between HIP, parent, and candidate; the second rotation reversed the order. Every comparison reported zero differing BF16 elements for candidate/parent, candidate/HIP, and parent/HIP. The retained candidate/parent median ratios are summarized below; the two values are the independent rotations.

| Family | M2048 | M8192 | M32768 |
| --- | ---: | ---: | ---: |
| Attention Q-A | `HipTile` parent retained | `0.9784x` / `0.9747x` | `0.9598x` / `0.9806x` |
| Attention Q-B | `0.9624x` / `0.9481x` | `0.9514x` / `0.9589x` | `0.9577x` / `0.9560x` |
| Attention KV | `0.7504x` / `0.6673x` | `0.9863x` / `0.9890x` | `0.9688x` / `0.9659x` |
| Attention output B | `0.9497x` / `0.9756x` | `0.9727x` / `0.9625x` | `0.9644x` / `0.9663x` |
| Shared gate/up | `0.9803x` / `0.9923x` | `0.9705x` / `0.9778x` | `0.9744x` / `0.9643x` |
| Shared down | `0.9758x` / `0.9749x` | `0.9724x` / `0.9708x` | `0.9631x` / `0.9655x` |

The corresponding candidate/HIP ratios were below one in both rotations for every retained key. The largest absolute gains are on Q-B and output-B, while KV M2048 is a distinct ownership replacement: its compact M128 candidate beat the selected compact-M64 parent in both direct rotations and beat HIP in both, despite the earlier M128 source rejection. This is admissible evidence for this exact key only; it does not generalize neighboring keys without their own entries and comparisons.

LM-head M128, M256, and M512 also passed two reversed 25-repeat rotations, with candidate/parent ratios of `0.9651x` / `0.9570x`, `0.9449x` / `0.9688x`, and `0.9626x` / `0.9549x`. The exact compact M32 and M64 probes remained rejected at `1.0171x` and `1.0140x` parent in their nine-repeat screens, so the existing small-M catalog mappings remain unchanged. The Q-A M2048 compact candidate likewise regressed to `1.0375x` of its existing `HipTile` parent, so the ordinary mapping remained selected. The later measurement-error review below qualifies that retained parent directly against HIP.

A deterministic rebuild of the representative M128 output-B and Q-B artifacts reproduced identical generated source, object, and HSACO hashes; every 21-case static build passed inspection with zero private bytes and zero spills. The production catalog now contains four deduplicated solutions and 23 exact mappings: 20 `CompactDepth32WeightRows` mappings, one `HipTile` mapping for Q-A M2048, and the two exact small-M LM mappings. The catalog round-trips through `load_catalog()` with no unreferenced solution indices. This is the selected production research state; public dispatch, generated bundle packaging, and installed HIP remain unchanged.

The compact-depth32 reopening is closed. Its changed premise was implemented, independently qualified across every retained exact key, and fully reviewed. Remaining Q8 findings are either the explicitly rejected small-M/short-M controls, exact-key timing controls, or mechanisms already closed in the preceding Q8 record. The bounded Q8 domain now enumerates the complete implemented policies, but it remains tooling rather than optimization-exhaustion evidence. No new in-contract mechanism remains actionable in this composition.

## Q-A Measurement-Error Reconciliation

The last apparent multiply deficit was the retained `Q8HipTiledLds` key `(M,N,K)=(2048,1024,4096)`. A fresh catalog build reproduced the previously timed source and HSACO byte-for-byte. The existing strict result remains exact to HIP and public output, deterministic, independently referenced, mutation-sensitive, code-object-v5, gfx1151, wave32, 40-byte ABI, `240 VGPR / 16 SGPR / 38,400 LDS`, and zero private bytes or spills.

The timing decision follows EvoTensile's noisy-measurement model: median log time, robust log scale `max(stdev, 1.4826*MAD, IQR/1.349)`, median standard error `1.253314*sigma/sqrt(n)`, `90%` confidence, and the default `2%` practical-equivalence zone. The older single-enqueue runs measured arithmetic candidate/HIP median ratios of `1.01381x` and `1.01236x`, but their confidence intervals were too broad to establish a slowdown. Pooling all 50 samples produced an independent-arm interval of `0.98160x-1.04551x` and a paired interval of `0.99498x-1.05396x`. That evidence is unresolved, not a valid fallback result.

Two additional single-enqueue processes used 200 warmup launches and 80 samples per arm. Their arithmetic candidate/HIP median ratios moved to `0.99728x` and `0.99882x`. The pooled independent-arm interval `0.98116x-1.01577x` resolves as equivalent within `2%`; the paired interval `0.99602x-1.02675x` remains unresolved because individual-launch log noise is still large. Thus even the single-enqueue protocol provides no evidence that GGTensile is slower.

Final confirmation used 20 warmup batches, 80 samples per arm, and 10 enqueues per timed sample, with alternating backend order and a fresh reversed-order process. The fixed Q8_1 producer ran once before timing, so these are prequantized multiply bodies only.

| Confirmation | HIP median | GGTensile median | HIP/GGTensile speedup | Independent candidate/HIP 90% CI | Paired candidate/HIP 90% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | `0.603805 ms` | `0.545028 ms` | `1.1078x` | `0.89881x-0.90651x` | `0.90078x-0.90578x` |
| B, reversed | `0.612754 ms` | `0.556184 ms` | `1.1017x` | `0.90383x-0.91154x` | `0.90546x-0.91013x` |

Both confirmations are confidently faster than HIP beyond the `2%` zone in both analyses and in both launch-order strata. The measurement controls resolve as equivalent: candidate-vs-candidate paired CI `0.99932x-1.00431x`, and HIP-vs-HIP `0.99815x-1.00243x`. The pooled 160-sample candidate/HIP interval is `0.90381x-0.90729x`, well outside those control floors. The retained Q-A `HipTile` mapping therefore passes the multiply gate; there is no real residual slowdown and no optimization reopening is justified. Full samples and calculations are under `~/tmp/torch-ggml-ops/q8-qa-hot-{pair-a,pair-b,candidate-control,hip-control}.json`, `q8-qa-single-hot-{a,b}.json`, and `q8-qa-measurement-error.json`.

## Final Multiply Result

The final matrix follows the authoritative public catalog's `(M,N,K)` orientation and reports prequantized multiply-only bodies. HIP and GGTensile consume the same workspace from the same fixed `quantize_bf16_q8_1_f32_d4` producer; activation production is excluded from every throughput and speedup below. Logical throughput is `2*M*N*K/(median_ms*1e9)`. `GGTensile/HIP` speedup is `GGTensile TFLOPS / HIP TFLOPS`, equivalently `HIP median time / GGTensile median time`, so values above `1.0x` favor GGTensile. Each row averages the two final public-catalog audit medians; the largest A/B median spread is below one percent.

| Family | `(M,N,K)` | Public catalog hash | HIP TFLOPS | GGTensile TFLOPS | GGTensile/HIP speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| Attention Q-A | `(2048,1024,4096)` | `ggsol_620516f662da29f3` | `31.036` | `34.245` | `1.1034x` |
| Attention Q-A | `(8192,1024,4096)` | `ggsol_f78e834f7bb82a2e` | `30.822` | `36.074` | `1.1704x` |
| Attention Q-A | `(32768,1024,4096)` | `ggsol_1389b8104bc29061` | `32.121` | `36.974` | `1.1511x` |
| Attention Q-B | `(2048,32768,1024)` | `ggsol_505d33dd36cd5790` | `28.606` | `34.092` | `1.1918x` |
| Attention Q-B | `(8192,32768,1024)` | `ggsol_323c022d43fb2435` | `28.880` | `34.641` | `1.1995x` |
| Attention Q-B | `(32768,32768,1024)` | `ggsol_f15bc6c956ce7c52` | `29.031` | `35.059` | `1.2077x` |
| Attention K/V | `(2048,512,4096)` | `ggsol_b40f08e30ff5bc83` | `28.615` | `30.285` | `1.0584x` |
| Attention K/V | `(8192,512,4096)` | `ggsol_de612ce7a470579d` | `30.623` | `34.944` | `1.1411x` |
| Attention K/V | `(32768,512,4096)` | `ggsol_0898037fe8783040` | `31.613` | `36.766` | `1.1630x` |
| Attention output-B | `(2048,4096,8192)` | `ggsol_eac11aed4c3b4229` | `29.298` | `36.512` | `1.2463x` |
| Attention output-B | `(8192,4096,8192)` | `ggsol_6525bb03932d9c84` | `30.379` | `37.513` | `1.2348x` |
| Attention output-B | `(32768,4096,8192)` | `ggsol_da2d79e7129cd5d7` | `30.733` | `37.938` | `1.2344x` |
| Shared gate/up | `(2048,2048,4096)` | `ggsol_abdcb1bf8cb8e28d` | `31.067` | `35.066` | `1.1287x` |
| Shared gate/up | `(8192,2048,4096)` | `ggsol_b4500bc7bd68b338` | `31.950` | `36.748` | `1.1502x` |
| Shared gate/up | `(32768,2048,4096)` | `ggsol_64f2005dfe00adb8` | `32.479` | `37.628` | `1.1585x` |
| Shared down | `(2048,4096,2048)` | `ggsol_e9250a27a783dd49` | `30.508` | `34.968` | `1.1462x` |
| Shared down | `(8192,4096,2048)` | `ggsol_253f1a95ce85cd9e` | `31.817` | `36.326` | `1.1417x` |
| Shared down | `(32768,4096,2048)` | `ggsol_f3d23f4df3973903` | `32.232` | `37.049` | `1.1495x` |
| LM head | `(32,129280,4096)` | `ggsol_b9b1167f6cbe898a` | `12.360` | `12.922` | `1.0455x` |
| LM head | `(64,129280,4096)` | `ggsol_ba13b32d0f3ff9ec` | `24.066` | `24.540` | `1.0197x` |
| LM head | `(128,129280,4096)` | `ggsol_796295ce34f4992d` | `29.373` | `36.335` | `1.2370x` |
| LM head | `(256,129280,4096)` | `ggsol_0da11f24c6cb2f0c` | `29.647` | `36.587` | `1.2341x` |
| LM head | `(512,129280,4096)` | `ggsol_ee38be695060a665` | `29.956` | `37.203` | `1.2419x` |

The Q-A M2048 compact candidate remains rejected at `1.0375x` of the retained `HipTile` source. The measurement-error reconciliation qualifies that existing source directly against HIP, so no kernel or catalog identity changes. Complete-call matrices remain in the chronology above and are not mixed into this multiply-only final table. The audit medians are in `~/tmp/torch-ggml-ops/fwd-complete-audit-q8-{a,b}.json`.

### Same-Producer Complete-Call Diagnostic

A separate catalog-wide audit timed quantization plus multiply with one loaded F32_D4 producer instance and one workspace shared by each HIP/GGTensile pair. Multiply and complete phases were separated; short kernels used sustained batches; backend order alternated; and the second 25-repeat pass reversed mode and backend order. The table reports `HIP complete median / GGTensile complete median`. These diagnostic runs use a different sustained timing context from the immutable promotion evidence, so their complete ratios and within-run reductions must not be combined numerically with the final multiply table above.

| Family | M2048 complete A/B | M8192 complete A/B | M32768 complete A/B |
| --- | ---: | ---: | ---: |
| Attention Q-A | `1.0897x/1.0852x` | `1.1537x/1.1540x` | `1.1331x/1.1337x` |
| Attention Q-B | `1.1880x/1.1872x` | `1.1978x/1.1998x` | `1.2072x/1.2079x` |
| Attention K/V | `1.0520x/1.0465x` | `1.1101x/1.1115x` | `1.1178x/1.1184x` |
| Attention output-B | `1.2491x/1.2496x` | `1.2270x/1.2270x` | `1.2235x/1.2215x` |
| Shared gate/up | `1.1137x/1.1146x` | `1.1492x/1.1494x` | `1.1465x/1.1468x` |
| Shared down | `1.1356x/1.1337x` | `1.1474x/1.1460x` | `1.1480x/1.1450x` |

| LM-head M | M32 | M64 | M128 | M256 | M512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Complete speedup A/B | `1.0464x/1.0459x` | `1.0185x/1.0208x` | `1.2374x/1.2365x` | `1.2367x/1.2350x` | `1.2411x/1.2404x` |

The notable reduction is concentrated in low-N ordinary families, where producer work is large relative to multiply work. Attention K/V M8192 lost `2.90` and `3.16` speedup percentage points within the two audit passes, while M32768 lost `4.47` and `4.51` points. Q-A lost `1.54/1.64`, `1.72/1.59`, and `2.07/1.47` points for M2048/M8192/M32768. Smaller but repeatable reductions appeared for K/V M2048 (`0.96/0.86` points), shared gate/up M2048 (`1.42/1.49`) and M32768 (`1.22/1.16`), and shared down M2048 (`1.11/1.20`). Q-B and LM-head ratios barely moved, and output-B retained approximately `1.22-1.25x` complete speedup. Full medians, means, standard deviations, MADs, outliers, order splits, and samples are in `~/tmp/torch-ggml-ops/fwd-complete-audit-q8-{a,b}.json`.

## Candidate-Domain Evidence Boundary

The automated Q8 domain is a bounded deterministic enumerator of complete implemented policies, not evidence that the kernel is exhausted. It can reproduce admissible ownership families but does not encode the historical exact-key timing matrix or prove that every future linked composition has been rejected. Q8 closure remains grounded in the exact artifacts, mutation and reference evidence, measurements, and recursive manual review above. A domain omission is actionable only when it supplies a concrete in-contract mechanism, exact target, plausible gain path, and qualification gate.
