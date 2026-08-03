# GGTensile Dense MMQ Backward Q3_K Experiment Plan

## Purpose

This experiment extends GGTensile dense MMQ backward to Q3_K on gfx1151, wave32, and WMMA V1. It covers the six exact production keys in the current Qwen workload. HIP remains the correctness and performance oracle and the runtime fallback for unmatched keys; every selected exact production key now beats HIP.

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

Select per exact key, then evaluate the six-key weighted result using call counts. HIP remains fallback for every unmatched key; a GGTensile solution is selected only after it beats HIP on its exact key.

### Lower Bounds and Bottlenecks

For representative long narrow and query keys, generate complete, WMMA/A/LDS-floor, and decode/LDS-floor artifacts when timing diagnosis is ambiguous. Explain whether remaining time is dominated by WMMA/A/LDS, Q3 metadata/payload decode, LDS synchronization, launch/occupancy, or overlap limits.

The campaign is incomplete until narrow and query bottlenecks are separately explained, or evidence proves one common bottleneck dominates both.

## Recursive Optimization-Exhaustion Review

This is the permanent final campaign step. Re-read this plan; the Q4_K and Q5_K experiment logs; dense and grouped MMQ optimization histories; current Q3/Q4/Q5 HIP, CK, GGTensile, and normalized-disassembly sources; inventories, selected and rejected solutions, manifests, lower bounds, profiles, counters, timing brackets, and mutation results; TensileLite, EvoTensile, CK, hipBLASLt, and relevant mechanism studies; and the gfx1151 ISA reference plus LLVM AMDGPU instruction, VOPD, hazard, delay, and scheduling definitions and tests.

Classify every idea as duplicate or closed, contract-incompatible, unsupported, deferred with an explicit prerequisite, or actionable with an exact target key, mechanism, expected gain path, and measurement gate. If the review finds an actionable in-contract idea, implement and measure it before the review can pass, update this plan, and repeat the complete review. The Q3_K campaign is exhausted only when a fresh final pass finds no new valid actionable idea, all prior actionable ideas are retained, rejected, or explicitly deferred, and remaining bottlenecks are explained with evidence.

A plan edit or implementation change creates a new premise and invalidates the prior stopping condition. The review must remain the final step and cannot pass in the same iteration that discovers actionable work.

## Campaign Evidence

The initial `128x128x32` one-buffer Q3 path was correct and fast, but the short-K campaign exposed a missing layout mechanism. HIP's narrow kernel uses an unswizzled decoded-B LDS tile with eight rows of padding. Adding real `LdsPadB=8` support, then reducing the narrow geometry to `256x64x32`, lowered decoder rows, LDS traffic, and accumulator pressure while preserving packed in-kernel decode. The four-M-tile emitter also required separate A-row, LDS, quant-shift, and Q3 low-shift state; aliasing the cached low shift with the third A pointer caused the first `256x64` correctness failure and was fixed before timing.

The retained Q3 emitters use one LDS buffer, packed extraction, fused signed-scale multiplication, SIA5/store priority, decode-shift hoisting, Q3 VOPD pairing where the exact geometry is favorable, and `v_lshl_or_b32` to merge the original high mask with the low payload. All selected artifacts use `LdsPadB=8`, unswizzled B LDS, zero private storage and spills, and pass the hard ABI/resource gates.

| Mechanism | Result | Decision |
| --- | ---: | --- |
| Padded `256x64x32`, WGM2 | Q3 narrow M2048 screen `0.8783x` versus HIP; 25-repeat final `0.8831x` | Select for narrow M2048 |
| Padded `256x64x32`, WGM1 | Q3 narrow M8192/M32768 final `0.7874x`/`0.8073x` versus HIP | Select for narrow M8192 and M32768 |
| Padded `128x64x32`, WGM1 | Query M2048/M8192 screen gains of about 5%/9% versus prior selected paths | Select for query M2048 and M8192 |
| Padded `128x128x32`, WGM1 | Query M32768 confirmation `0.9733x` versus the unpadded assembly control | Select for query M32768 |
| Padded `256x64` for Q4/Q5 shared-down | Q4/Q5 weighted candidate/control `1.4262x`/`1.4577x` | Reject; shared-down remains two-buffer bound |
| SIA4/store-priority alternatives | Q3, Q4, and Q5 screens were neutral or regressing; Q5's apparent M2048 win became `1.0178x` at 25 repeats | Reject |
| WGM4/WGM8, PGR1, scalar extraction, alternate swizzles, metadata vector loads | Earlier correctness or timing gates failed | Reject |

### Final result

The authoritative 25-repeat confirmation reports logical arithmetic throughput. Speedup is `HIP median time / GGTensile median time`, so values above `1.0x` favor GGTensile.

| Family | `(M,N,K)` | Geometry/schedule | HIP TFLOPS | GGTensile TFLOPS | Speedup vs HIP |
| --- | ---: | --- | ---: | ---: | ---: |
| Narrow | `(2048,2048,512)` | `256x64`, WGM2, padded, SIA5 | `23.671` | `26.804` | `1.1323x` |
| Narrow | `(8192,2048,512)` | `256x64`, WGM1, padded, SIA5 | `22.890` | `29.072` | `1.2700x` |
| Narrow | `(32768,2048,512)` | `256x64`, WGM1, padded, SIA5 | `24.277` | `30.070` | `1.2386x` |
| Query | `(2048,2048,8192)` | `128x64`, WGM1, padded, SIA5 | `19.649` | `24.692` | `1.2566x` |
| Query | `(8192,2048,8192)` | `128x64`, WGM1, padded, SIA5 | `21.406` | `26.715` | `1.2480x` |
| Query | `(32768,2048,8192)` | `128x128`, WGM1, padded, SIA5 | `22.520` | `27.162` | `1.2062x` |

The call-weighted speedup is `1.2180x`, corresponding to the recorded `0.8210x` candidate/HIP latency ratio. The final selected resource classes are 243 VGPR/5 KiB LDS for narrow `256x64`, 143 VGPR/5 KiB LDS for query `128x64`, and 218 VGPR/10 KiB LDS for query M32768 `128x128`; each has 16 SGPR, zero private bytes, zero spills, and the expected static WMMA/VMEM/LDS structure. The two independent final roots produce byte-identical assembly and matching resource tuples for all six keys.

The final correctness phase passed HIP comparison, independent references, complete `grad_output` mutation, and packed-weight mutation on all six exact keys. Reduced-K checks against the final padded `256x64` geometry matched HIP and the independent reference at K32/K64/K96; at K512 candidate and HIP matched each other, while both shared the known 511-element BF16 accumulation-order difference from the independent reference.

Serial lower-bound measurements on the earlier two-buffer control at M32768 remain useful for bottleneck diagnosis:

| Family | Complete | WMMA/A/LDS floor | Decode/LDS floor | Floor sum |
| --- | ---: | ---: | ---: | ---: |
| Narrow, K512 | `2.618 ms` | `1.621 ms` | `1.166 ms` | `106.4%` |
| Query, K8192 | `46.780 ms` | `26.666 ms` | `21.019 ms` | `101.9%` |

The short-K gap was therefore primarily layout, decoder-row, and launch/overlap sensitivity rather than code size. The long query gap remains an overlapped WMMA/decode/LDS pipeline. The final pass also reviewed the HIP and GGTensile logs, CK/TensileLite mechanism notes, normalized disassembly, rocprofiler results, and gfx1151 LLVM/ISA constraints. No remaining in-contract mechanism has a measured path to a stable gain above the resource-bearing threshold. Persistent workgroups, split-K, prepared weights, external decode storage, grouped MMQ, and public runtime dispatch remain outside this campaign or explicitly deferred.

## Recursive Optimization-Exhaustion Review

The padded LDS and compact-geometry rewrite invalidated the earlier HIP-fallback conclusion for narrow M2048 and M32768, so those exact keys were regenerated, correctness-checked, screened, confirmed, and independently reproduced rather than retaining the historical fallback. The follow-up checks covered padded `128x128`, padded `128x64`, padded `256x64`, WGM1/2/4/8, SIA4/SIA5, store priority, address-state allocation for four M tiles, and Q3 reduced-K behavior. Large-margin layout and geometry opportunities are exhausted. Remaining timing is bounded by fused WMMA/decode/LDS overlap and short-K launch/occupancy behavior; smaller instruction-count changes did not clear the stable timing gate. A fresh final pass finds no new valid actionable mechanism, so this review closes the Q3 campaign.

## Completion Record

Documentation-only plan updates remain uncommitted unless explicitly requested.
- Six exact Q3_K production shapes identified from the current Qwen inventory.
- Strict Q3_K ProblemType, solution identity, and multi-quant backend support.
- Q3_K reduced-trip, one-hot, HIP, independent-reference, and producer-mutation correctness.
- Padded LDS support, compact `256x64` short-K geometry, four-M-tile address-state separation, and exact-key selection.
- Final six-key confirmation and independent reproducibility rebuild.
- Recursive final optimization-exhaustion review with no actionable mechanism remaining.
- Public runtime dispatch deferred until broader dense quant coverage and complete workload validation.
