# GGTensile Dense MMQ Backward Q3_K Experiment Plan

## Purpose

This experiment extends GGTensile dense MMQ backward to Q3_K on gfx1151, wave32, and WMMA V1. It covers the six exact production keys in the current Qwen workload and keeps HIP as the correctness, performance, and runtime fallback.

The generic lifecycle, strict identity rules, phase separation, validation policy, retention gates, and public-integration roadmap remain in [ggtensile_plan.md](ggtensile_plan.md). The completed Q4_K and Q5_K campaigns are architectural references only; no Q4_K or Q5_K tuning result transfers to Q3_K without measurement.

## Exact Scope

Dense backward uses `M=rows`, `N=in_features`, and `K=out_features`:

```text
grad_input[M,N] = grad_output[M,K] @ dequant(weight[K,N])
```

Each generated artifact supports exactly one `ProblemType`, `ProblemSize`, and decoder contract. Q3_K uses 256-value blocks and 110 bytes per block. For the production `N=2048` rows, the packed weight shape is `[K,880]`. No dense shadow, prepared weight, external decode workspace, split-K reduction, persistent workgroup, or grouped MMQ path is in scope.

## Production Inventory

The current Qwen dense workload has two Q3_K geometries:

| Family | `(N,K)` | Packed weight `[K,N]` | Representative tensor | Calls |
| --- | ---: | ---: | --- | ---: |
| Narrow/full-attention key | `(2048,512)` | `[512,2048]`, physical `[512,880]` | `blk.3.attn_k.weight` | 9 |
| Query | `(2048,8192)` | `[8192,2048]`, physical `[8192,880]` | `blk.3.attn_q.weight` | 9 |

The six exact tuning keys are:

```text
(2048, 2048,  512)   (8192, 2048,  512)   (32768, 2048,  512)
(2048, 2048, 8192)   (8192, 2048, 8192)   (32768, 2048, 8192)
```

There is no current production Q3_K shared-down `(N=512,K=2048)` family and no production Q3_K attention-output `(N=4096,K=2048)` family. Do not add those shapes without workload evidence.

The existing HIP dispatch distinguishes two dense full-tile Q3 bodies:
- `N=2048,K=512`: `DenseBwdQ3KFullNarrow`.
- `N=2048,K=8192`: `DenseBwdQ3KFullWide`.

Those are initial workload facts, not public GGTensile force controls.

## Q3_K Decoder Contract

Q3_K has a 110-byte block with a 32-byte high-mask plane, a 64-byte low 2-bit payload plane, and 12 bytes of scale metadata followed by FP16 `d`. The decoder must combine the low 2-bit payload and high mask into signed 3-bit values, apply the Q3_K scale metadata, convert to BF16, and store the same LDS-facing BF16 tile consumed by the shared WMMA body.

The Q3 backend must own block-byte sizing, high-mask and low-payload loads, scale metadata extraction, signed 3-bit reconstruction, and LDS stores. The common body may own launch flattening, A addressing, prefetch, reduction-trip specialization, WMMA issue order, LDS reads, accumulation, BF16 stores, barriers, and termination only where Q3's LDS tile contract is identical.

Correctness-only reduced reduction depths must include `K=32`, `64`, `96`, and `512`, plus production `K=8192`. Tests must exercise the high-mask plane, all low 2-bit combinations, signed reconstruction boundaries, all eight scale groups, metadata boundaries, the final block boundary, and packed-weight mutation.

## Multi-Quant Architecture

- `ProblemType` identifies Q3_K, Q4_K, and Q5_K through strict quant data fields. Quant types must have distinct hashes and symbols.
- A quant specification owns block bytes, payload planes, metadata layout, decoder width, signed reconstruction, and quant-specific register demand.
- A shared body owns the fused A/LDS/WMMA/accumulation/store contract only after Q3's LDS tile layout is proven identical.
- Quant-specific controls belong in strict identity fields. Q3 candidates may include high-mask load ownership, signed 3-bit reconstruction mode, scale metadata layout, packed payload sharing, and Q3-specific LDS swizzle or prefetch. A control is exposed only with a real alternate emitter and validation coverage.
- Inspection must derive resource accounting from the quant backend, while retaining zero private storage, spills, scratch, calls, dynamic stack, ABI, ISA, and exact-WMMA gates.

Do not expose inert Q3 fields on Q4_K or Q5_K solutions. Do not assume the Q4/Q5 176/144-byte payload ordering or decoder register layout applies to Q3's 110-byte block.

## Campaign Phases

### Inventory and Controls

Create a versionless six-key Q3 inventory with representative tensors, call counts, refreshed HIP medians, and an initial generated control. Keep narrow and query as separate families even though they share `N=2048`, because `K=512` and `K=8192` change decode-to-WMMA balance and call-weighted importance.

### Q3 Backend and Strict Validation

Implement Q3 through the quant backend boundary. Add unit coverage for strict problem identity, Q3 block bytes, payload offsets, high-mask/low-payload emission, signed 3-bit reconstruction, scale metadata, physical weight sizing, strict unsupported-quant rejection, and resource-clean independent builds.

Before production timing, pass reduced-trip correctness at `K=32/64/96/512` and a one-hot packed matrix covering every Q3 payload bit, signed 3-bit boundary, scale group, metadata byte, and final block row.

### Baseline Correctness

Require bit-exact HIP comparison for all six production keys, independent Q3 dequantized BF16 reference comparison, complete grad-output mutation, and packed-weight mutation. No timing result is accepted before these checks pass.

### Large-Margin-First Optimization

Use serial immutable generate, build, inspect, correctness, screening, and confirmation phases with warmed rotating same-process controls. Prioritize:
- The query family first if its long `K=8192` decode path is materially slower or has the largest call-weighted deficit.
- The narrow family first if its 9 calls and short `K=512` path expose a large decoder overhead or a clear HIP/GGTensile gap.
- Q3 high-mask/low-payload load ownership and signed reconstruction, especially mechanisms that remove instructions without increasing VGPRs.
- Two-buffer versus one-buffer decoded-B pipelines only if Q3's 110-byte layout changes overlap or occupancy.
- Exact Q3 metadata load width, scale extraction, high-mask normalization, and packed payload sharing only through complete emitters.
- WGM, SIA, PLR, LDS swizzle, and exact-trip controls only where per-key timing or lower-bound evidence justifies them.
- Exact power-of-two address lowerings and ISA fused operations before resource-bearing ownership changes.

Do not scan Q4/Q5 controls that do not change Q3 assembly. Timing is authoritative; static VALU/VMEM counts, resources, code size, locality, and counters explain the result.

### Selection and Confirmation

Retain only exact-shape-correct, reproducible, resource-clean candidates:
- zero private bytes, spills, scratch instructions, calls, and dynamic stack.
- independent generated assembly byte-identical.
- no stable regression above 1% on another exact key sharing the emitted path.
- resource-bearing mechanisms require a stable gain above 2%.
- unconditional instruction/resource reductions may remain when neutral-to-favorable.

Select per exact key, then evaluate the six-key weighted result using call counts. HIP remains fallback for every unmatched or slower key.

### Lower Bounds and Bottlenecks

For representative long narrow and query keys, generate complete, WMMA/A/LDS-floor, and decode/LDS-floor artifacts when timing diagnosis is ambiguous. Explain whether remaining time is dominated by WMMA/A/LDS, Q3 metadata/payload decode, LDS synchronization, launch/occupancy, or overlap limits.

The campaign is incomplete until narrow and query bottlenecks are separately explained, or evidence proves one common bottleneck dominates both.

## Recursive Optimization-Exhaustion Review

This is the permanent final campaign step. Re-read this plan; the Q4_K and Q5_K experiment logs; dense and grouped MMQ optimization histories; current Q3/Q4/Q5 HIP, CK, GGTensile, and normalized-disassembly sources; inventories, selected and rejected solutions, manifests, lower bounds, profiles, counters, timing brackets, and mutation results; TensileLite, EvoTensile, CK, hipBLASLt, and relevant mechanism studies; and the gfx1151 ISA reference plus LLVM AMDGPU instruction, VOPD, hazard, delay, and scheduling definitions and tests.

Classify every idea as duplicate or closed, contract-incompatible, unsupported, deferred with an explicit prerequisite, or actionable with an exact target key, mechanism, expected gain path, and measurement gate. If the review finds an actionable in-contract idea, implement and measure it before the review can pass, update this plan, and repeat the complete review. The Q3_K campaign is exhausted only when a fresh final pass finds no new valid actionable idea, all prior actionable ideas are retained, rejected, or explicitly deferred, and remaining bottlenecks are explained with evidence.

A plan edit or implementation change creates a new premise and invalidates the prior stopping condition. The review must remain the final step and cannot pass in the same iteration that discovers actionable work.

## Campaign Evidence

The initial two-buffer `128x128x32` Q3 path was correct but left a large opportunity in the one-buffer ownership experiment. The retained emitter uses one LDS buffer, packed extraction, the fused Q3 signed-scale multiply, SIA5 scheduling, and store priority. The final guarded VOPD artifacts are resource-clean at 222 VGPRs, 16 SGPRs, 8 KiB LDS, zero private bytes and spills, 32 static WMMAs, 146 VMEM instructions, and 64 LDS instructions. After merging the original Q3 high mask with the low payload using `v_lshl_or_b32`, the retained full-pair path uses 693 static VALU issues and 55 VOPD pairs; the two HIP-fallback narrow keys use the partial-pair path at 707 static VALU issues and 38 VOPD pairs. Relative to the initial two-buffer control, the retained path removes one LDS buffer and reduces static VALU issues from 1102 to 693. The four extra logical temporaries remain within the same 224-VGPR allocation granule as the prior 218-register declaration.

The bounded optimization results were:

| Mechanism | Result | Decision |
| --- | ---: | --- |
| One-buffer decoded-B ownership | `0.9869x` versus the two-buffer control | Retain as the production candidate |
| Fused Q3 signed-scale multiply | `0.9984x` versus the prior one-buffer emitter | Retain |
| SIA5 plus store priority | `0.9973x` versus the fused default | Retain in the combined candidate |
| Low/high decode-shift hoist | `0.9953x` versus the fused default | Retain as an unconditional instruction reduction |
| Packed Q3 scale VOPD pairs | 16 fewer standalone scale/decode issues; 25-repeat weighted ratio `0.8661x` versus HIP | Retain as the partial-pair baseline |
| Full Q3 subtract/multiply VOPD pairing | 15 fewer static VALU issues after one dual row-scale copy; guarded weighted ratio `0.8667x` versus HIP | Retain for narrow M8192 and all query keys; use partial pairing for narrow M2048/M32768 |
| Q3 original-mask payload merge | 30 fewer static VALU issues on full-pair keys and 31 on partial-pair keys; 25-repeat weighted ratio `0.8615x` versus HIP | Retain for all six exact keys; reduced K32/K64/K96/K512 checks remain correct |
| HIP-like post-decode A placement | `1.1806x` versus the VOPD control | Reject; lost B-decode/A-load overlap |
| VOPD-era WGM2/WGM4/WGM8 | Weighted ratios `1.0077x`/`1.0265x`/`1.0496x` versus the VOPD control | Reject |
| VOPD-era PGR1 and SIA4 | Weighted ratios `1.0289x`/`1.0064x` versus the VOPD control | Reject |
| 128x64 macro tile | `0.9752x` versus HIP, weighted | Reject |
| M64 macro tile | `1.6449x` versus HIP, narrow | Reject |
| DepthU64 | `0.9163x` versus HIP, weighted | Reject |
| PGR1 | `1.0466x` versus HIP, narrow | Reject |
| Scalar extraction | `0.9559x` versus the packed path | Reject |
| LDS swizzle chunks 4 and 16 | `0.9065x` and `1.0737x` versus HIP | Reject |
| Q3 vector metadata load | Failed reduced one-hot coverage for the second block half | Remove; no invalid alternate remains |

The reduced-trip fixtures use `K=32`, `64`, `96`, and `512` with sliced packed rows. Candidate/HIP and candidate/independent BF16 reference mismatches were zero at `K=32`, `64`, and `96`; the `K=512` candidate also matched HIP exactly. The production six-key correctness phase passed HIP comparison, independent reference checks, grad-output mutation, and packed-weight mutation. The small independent-reference difference at the full production depth is identical between HIP and GGTensile and is attributable to BF16 matmul accumulation order, not packed decode.

Serial lower-bound measurements on the two-buffer control at `M=32768` were:

| Family | Complete | WMMA/A/LDS floor | Decode/LDS floor | Floor sum |
| --- | ---: | ---: | ---: | ---: |
| Narrow, `K=512` | `2.618 ms` | `1.621 ms` | `1.166 ms` | `106.4%` |
| Query, `K=8192` | `46.780 ms` | `26.666 ms` | `21.019 ms` | `101.9%` |

Both families are close to overlapped floors. The narrow path is sensitive to launch and short-trip scheduling overhead; the long query path is primarily an overlap-limited WMMA/decode/LDS pipeline. No remaining resource-bearing mechanism met the required stable gain threshold.

The final 25-repeat confirmation selected the combined candidate only for exact keys that beat HIP:

| Family | M | Candidate/HIP | Action |
| --- | ---: | ---: | --- |
| Narrow | 2048 | `1.0820x` | HIP fallback; partial VOPD plus payload merge |
| Narrow | 8192 | `0.9546x` | Retain GGTensile; full VOPD plus payload merge |
| Narrow | 32768 | `1.0032x` | HIP fallback; partial VOPD plus payload merge |
| Query | 2048 | `0.8373x` | Retain GGTensile; full VOPD plus payload merge |
| Query | 8192 | `0.8600x` | Retain GGTensile; full VOPD plus payload merge |
| Query | 32768 | `0.8534x` | Retain GGTensile; full VOPD plus payload merge |

Using HIP for the two slower narrow keys, the raw six-key call-weighted candidate/HIP latency ratio is `0.8615`; the retained per-key fallback policy remains unchanged. The merged guarded source is byte-reproducible under independent generation/build roots, with matching resource tuples and strict inspection. Public runtime dispatch remains deferred; the selected catalog is campaign evidence and does not change the current public dispatch boundary.

## Recursive Optimization-Exhaustion Review

The final review rechecked the Q3 plan, Q4/Q5 evidence, dense and grouped MMQ paths, current assembly emitters, inventories, rejection results, lower bounds, mutation checks, and the gfx1151 ISA/toolchain constraints. The actionable in-contract ideas were the one-buffer pipeline, fused signed-scale multiply, SIA5/store priority, decode-shift hoist, partial packed Q3 scale VOPD pairing, full subtract/multiply Q3 VOPD pairing, and original-mask payload merging with `v_lshl_or_b32`; each was implemented, correctness-checked, timed, and either retained or rejected per exact key. Full pairing is guarded by exact production shape because the two narrow HIP-fallback keys regressed in the 25-repeat branch comparison, while payload merging is retained across all six keys after a neutral-to-favorable confirmation. The follow-up HIP-like A placement, WGM, PGR1, and SIA4 scans are closed by fresh post-VOPD timing. Macro-tile changes, deeper reduction trips, scalar extraction, swizzle changes, and vector metadata loading are closed by measurement or correctness. Persistent workgroups, split-K, prepared weights, external decode storage, grouped MMQ, and public dispatch remain contract-incompatible or explicitly deferred. A fresh final pass found no new valid actionable mechanism, so this review closes the Q3 campaign.

## Completion Record

Documentation-only plan updates remain uncommitted unless explicitly requested.
- Six exact Q3_K production shapes identified from the current Qwen inventory.
- Strict Q3_K ProblemType, solution identity, and multi-quant backend support.
- Q3_K reduced-trip, one-hot, HIP, independent-reference, and producer-mutation correctness.
- Large-margin-first bounded optimization and exact-key selection.
- Lower-bound diagnosis for narrow and query families.
- Final six-key confirmation and reproducibility rebuild.
- Recursive final optimization-exhaustion review with no actionable mechanism remaining.
- Public runtime dispatch. Deferred until broader dense quant coverage and complete workload validation.
