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
1. Q-B `(N,K)=(32768,1024)` at M8192/M32768 and attention-output B `(4096,8192)` at long M, because they have large absolute kernel cost and large decode/reduction depth.
2. Shared gate/up `(2048,4096)`, weighted as two tensor calls per layer.
3. Shared down and Q-A, including their long-M keys.
4. Attention KV, after the larger ordinary margins are closed.
5. LM-head M512/M256, then M128/M64/M32 for chunk fallback and capacity behavior.

Complete-call timing and prequantized multiply timing are recorded separately. The fixed HIP Q8_1 producer is byte-identical for HIP and GGTensile multiply controls and is excluded from isolated multiply promotion. Complete-call and call-weighted totals guide workload decisions but never retain a slower exact key.

## Current State

The canonical forward model now has a fixed typed Q8_0 direct-global control. It is a correctness baseline only; optimized staged Q8 ownership and promotion decisions remain open. The exact inventory and parser-valid control catalog cover all 23 keys but do not claim any selected production winner.

Existing HIP controls are available through the dense forward bundle sources and dispatch in `csrc/mmq_bundle.cpp` and `csrc/mmq_core.cuh`. Existing Q8_0 GGTensile artifacts under `~/tmp/torch-ggml-ops/` are backward artifacts and are not forward baselines.

Baseline preservation checkpoint: all currently valid catalog sources were regenerated before Q8 edits into `~/tmp/torch-ggml-ops/q8-fwd-baseline-sources-20260805/`. The manifest contains 447 unique valid sources: 155 Q4_K/Q5_K/Q6_K forward sources and 292 Q3_K/Q4_K/Q5_K/Q6_K/Q8_0 backward sources. Every retained Q8 change must reproduce this set byte-for-byte unless an existing-stream change is explicitly separated and qualified.

The new canonical 23-key inventory uses `1.0` millisecond timing sentinels until fresh same-process HIP controls are collected. These values are parser-valid placeholders, not performance evidence and not selection inputs.

The initial isolated backend lowers one wave to a `16x16` output tile. It directly loads four 32-value Q8_0 blocks and one 128-value Q8_1 F32_D4 block per reduction-loop iteration, performs eight signed integer WMMAs, and applies the FP32 correction in the HIP expression order `integer_result * weight_scale * activation_scale` before BF16 RNE stores. `Q8DirectGroupRole` carries the payload and scale offsets, while `Q8DirectRegisterPlan` allocates explicit lifetime-bound roles deterministically. This branch does not use LDS and does not alter the Q4_K/Q5_K/Q6_K body methods.

Initial control evidence is under `~/tmp/torch-ggml-ops/q8-fwd-direct-control/m2048-n1024-k4096/`:
- strict assembly inspection passes with code-object v5, gfx1151, wave32, the 40-byte ABI, 88 VGPRs, 16 SGPRs, zero LDS, eight static WMMAs, zero private storage, and zero spills.
- after preserving the HIP source expression order `integer_result * weight_scale * activation_scale`, the Q-A `M=2048,N=1024,K=4096` control is bit-exact with both the same-workspace HIP multiply and public complete path across 2,097,152 BF16 elements. Candidate and public independent-reference normalized RMSE are both `0.00604494`.
- the LM-head `M=32,N=129280,K=4096` control is also bit-exact with HIP/public across 4,136,960 BF16 elements; candidate and public independent-reference normalized RMSE are both `0.00606697`.
- input, packed-weight, and Q8_1-workspace mutations change candidate output for both controls while remaining bit-exact with HIP.
- one-repeat diagnostics measured Q-A isolated multiply at `1.9162 ms` versus `0.5880 ms` HIP and LM-head M32 at `7.4937 ms` versus `2.7742 ms` HIP. These are not warmed promotion evidence.

The focused contract/reference/writer suite passes with `158 passed`. The full repository suite passes with `355 passed` and 14 existing warnings; Ruff, formatting, ty, compileall, diff checks, pre-commit, and the current 179-kernel public-bundle check also pass. Regeneration of the frozen eight pre-Q8 catalog families produced 447 sources with zero missing, added, or byte-changed files; the report is `~/tmp/torch-ggml-ops/q8-fwd-contract-current-sources-20260805/comparison.json`.

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
1. packed Q8 payload/scale load coalescing and decode clustering.
2. Q8-specific LDS row padding, swizzle, and local-read ownership.
3. direct-versus-decoded operand representation and one-versus-two stage buffering, only with lower-bound and resource justification.
4. `128x64`, `128x128`, `256x64`, and exact small-M geometries when accumulator pressure, LDS, and occupancy estimates support them.
5. DepthU, global/local prefetch, Q8 payload sharing, VOPD, clauses, dependency delays, store traversal, and exact-N/wave-group mapping.
6. LM-head chunk geometry and active-wave ownership after ordinary margins are understood.

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

- [x] Contract, canonical inventory dimensions, and independent packed Q8_0 forward reference.
- [x] Initial semantic Q8_0 backend and strict ordinary control.
- [ ] Ordinary 18-key correctness, mutation, and catalog coverage.
- [ ] LM-head five-key correctness, mutation, and chunk coverage.
- [ ] Large-margin geometry/decode/LDS/ownership search.
- [ ] Lower-bound and residual-bottleneck analysis.
- [ ] Per-key selection and two-confirmation promotion.
- [ ] Complete writer line coverage and regression identity gates.
- [ ] Recursive final review with no actionable mechanism remaining.
