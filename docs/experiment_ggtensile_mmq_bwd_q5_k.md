# GGTensile Dense MMQ Backward Q5_K Experiment Plan

## Purpose

This experiment extends GGTensile to dense MMQ backward Q5_K on gfx1151, wave32, and WMMA V1. It covers the six exact production keys used by the current Qwen dense workload and keeps HIP as the correctness, performance, and runtime fallback.

The generic lifecycle, strict identity rules, phase separation, validation policy, retention gates, and public-integration roadmap remain in [ggtensile_plan.md](ggtensile_plan.md). The completed Q4_K campaign remains the architectural reference, but no Q4_K result is assumed to transfer to Q5_K without measurement.

## Exact Scope

Dense backward uses `M=rows`, `N=in_features`, and `K=out_features`:

```text
grad_input[M,N] = grad_output[M,K] @ dequant(weight[K,N])
```

Each generated artifact supports exactly one `ProblemType`, `ProblemSize`, and quant decoder contract. The Q5_K packed row has logical weight shape `[K,N]`, uses 256-value blocks, and occupies `(N/256)*176` bytes per row. No dense shadow, prepared weight, external decode workspace, split-K reduction, persistent workgroup, or grouped MMQ path is in scope.

The production matrix is the Cartesian product of `M={2048,8192,32768}` and these two Q5_K families:

| Family | `(N,K)` | Packed weight `[K,N]` | Model weight count | Historical HIP ms at M2048/M8192/M32768 |
| --- | ---: | ---: | ---: | ---: |
| Narrow | `(2048,512)` | `[512,2048]` | 21 | `0.208/0.778/3.019` |
| Shared down | `(512,2048)` | `[2048,512]` | 10 | `0.239/1.392/5.606` |

The six exact keys are:

```text
(2048, 2048,  512)   (8192, 2048,  512)   (32768, 2048,  512)
(2048,  512, 2048)   (8192,  512, 2048)   (32768,  512, 2048)
```

There is no production Q5_K attention-output `(N=4096,K=2048)` or query `(N=2048,K=8192)` family. Those shapes are compatibility experiments only and are excluded from the initial inventory.

The first inventory must preserve the six exact keys, representative Qwen tensors, model call counts, refreshed same-process HIP controls, and a selected-solution reference. A Q5_K campaign must not reuse the Q4_K inventory schema or silently reinterpret a Q4_K packed-row size.

## Q5_K-Specific Existing Evidence

Q5_K shares Q4_K's 256-value block, 8 groups of 32 values, `d`/`dmin`, six-bit scale/minimum metadata, 4-bit low payload, Q8_1 activation layout, and BF16 WMMA arithmetic. Its additional payload is one high bit per value in a 32-byte `qh` plane, making the block 176 bytes instead of 144.

The existing HIP dispatch already exposes a shape-specific Q5 behavior that the campaign must represent explicitly:
- narrow `M=2048` uses scalar extraction.
- narrow `M=8192` and `M=32768` use packed extraction.
- shared-down uses the Q5_K packed body and its four-BF16 XOR LDS layout.

This is a derived starting policy, not a public force control. The implementation now records `Q5KExtraction` strictly, with complete `packed` and `scalar` emitters, and records `Q5KNibbleShiftHoist` as a real Q5-only decode emitter choice. The scalar emitter is correct but is retained only as a measured rejection candidate until it demonstrates a production gain.

## Multi-Quant GGTensile Design

### Shared kernel layers

The writer should be split into a quant-neutral fused body and a small quant-family backend:
- `ProblemType` identifies the quant format through strict data-type fields, with factories for each supported dense quant family rather than a Q4-only constructor.
- A quant specification resolves block values, packed row bytes, payload planes, metadata layout, decoder width, and quant-specific register demand.
- The shared body owns work-item flattening, exact launch mapping, A addressing and prefetch, reduction-trip specialization, WMMA issue order, LDS read ownership, accumulation, BF16 rounding, stores, barriers, waits, and termination.
- The quant backend owns packed-row global reads, packed payload replication, metadata normalization, dequantization, and LDS stores. It emits a complete named path, not a collection of unchecked string substitutions.
- Inspection and validation derive expected payload and synchronization structure from the quant specification, while preserving common hard gates for ISA, ABI, resources, private storage, spills, calls, and scratch.

Q4_K and Q5_K should share the WMMA/LDS body only when their LDS-facing tile contract is identical. Q5's extra high-bit plane must remain live only through decode and must not be carried into the WMMA body. Register allocation must reserve payload state from the quant backend before allocating transient address and decode temporaries.

### Quant-specific knobs

Common knobs remain common only when they produce distinct validated emitters for both relevant quant families. Quant-specific knobs belong in a strict quant-parameter section of the solution identity, with a schema that rejects parameters for unrelated formats rather than accepting inert fields. Q5-specific controls now include extraction mode, nibble-shift hoisting, and vector metadata loading; each has a complete emitter and a measured retention or rejection result.

Do not add a generic `DecoderMode`, `PackedWeightBytes`, or high-plane flag to the shared solution surface merely because Q5 needs it. Quant metadata and block geometry are identity facts derived from `ProblemType`; extraction mode, high-plane load width, metadata sharing, and payload prefetch are tuning choices only after they change emitted ISA or ownership and pass correctness.

### Compatibility and identity

The solution key must include the quantized problem type, so Q4_K and Q5_K artifacts cannot collide even when their common solution fields match. Kernel symbols must include the quant family. Assembly hashes cover generated source only. Build and code-object identities remain inspection outputs, not mutable solution identity fields.

## Campaign Phases

### Inventory and controls

Create a versionless Q5_K inventory with the six keys above. Establish the existing HIP implementation and a generated Q5 control as independent controls. Record the exact packed tensor, producer mutation requirements, refreshed HIP medians, and model call counts.

The initial generated control reuses the selected Q4-compatible four-wave `128x128x32`, WGM1, XOR-8, SIA4/PGR2/PLR1, two decoded-B buffer body. Q5-specific payload loads and decode are independent code paths. The generated control is resource-clean at 220 VGPRs, 16 SGPRs, and 16 KiB LDS for all six keys.

### Multi-quant emitter and strict validation

Implement Q5_K through the quant backend boundary and retain Q4_K output byte identity where its source inputs and solution are unchanged. Add unit coverage for:
- strict Q4_K and Q5_K `ProblemType` construction and round trips.
- distinct Q4_K/Q5_K solution hashes and kernel names.
- Q5 block-byte and payload-offset calculations for `N=512` and `N=2048`.
- Q5 high-plane and low-plane decode emission, including all eight scale groups and both nibble halves.
- rejection of unsupported quant families, dimensions, and quant-specific parameters.
- independent builds with zero private storage, spills, scratch, calls, dynamic stack, and register-bound violations.

Before production execution, run reduced Q5 reduction depths such as `K=32`, `64`, `96`, and `512`, plus an independent one-hot packed matrix that exercises low payload, high payload, scale/minimum fields, group boundaries, and the final block boundary.

### Baseline correctness

Require bit-exact comparison against HIP for all six production keys and against an independent Q5_K dequantized BF16 reference where the accumulation order matches. Rewrite complete `grad_output` and packed-weight tensors and repeat the comparisons. Execute the reduced-trip and one-hot tests before any timing.

### Bounded optimization

Use serial immutable generate, build, inspect, correctness, screening, and confirmation phases. Warm rotating same-process HIP, current generated control, and candidate measurements; never overlap GPU phases.

Start with the common Q4-compatible control, then scan only complete solutions under changed Q5 premises:
- narrow M2048 scalar versus packed extraction, then confirm the winning Q5-specific emitter across M8192 and M32768.
- shared-down packed extraction, four-BF16 XOR LDS layout, and exact M-specific traversal.
- one versus two decoded-B buffers if Q5 high-plane state changes occupancy or overlap.
- WGM1/2/4/8 only when the Q5 tile grid or locality evidence changes the Q4 decision.
- SIA/PGR/PLR interactions only after payload-load latency and decode/LDS timing are isolated.
- high-plane load width, low/high payload coalescing, and packed-byte normalization only with a complete emitter and bit-exact producer-mutation checks.
- exact-trip and address lowerings before resource-bearing ownership changes.

Do not scan Q4-only controls that do not alter Q5 assembly. Do not infer a win from lower static VALU or VMEM counts; timing is authoritative, while counters, resources, code size, and locality explain it.

### Selection and confirmation

Retain a candidate only when it is exact-shape correct, source-reproducible, resource-clean, and measured under the hard gates:
- no correctness failures.
- independent generated assembly is byte-identical.
- zero private bytes, spills, scratch instructions, calls, and dynamic stack.
- no stable regression above 1% on another exact key sharing the emitted path.
- resource-bearing mechanisms require a stable gain above 2%.
- unconditional instruction or resource reductions may remain when neutral-to-favorable on representative short- and long-reduction controls.

Select per exact key, then evaluate the complete six-key weighted result using call counts. A family average cannot authorize a slower exact key. HIP remains fallback for every unmatched or slower key.

### Initial bounded-scan evidence

The first six-key screen used 10 warmups, 9 rotating same-process repeats, serial execution, and the generated Q5 control as the assembly control. Ratios below are candidate latency divided by the two-buffer control latency; values above `1.0` are regressions.

| Candidate | Weighted ratio | Result |
| --- | ---: | --- |
| One decoded-B buffer | `1.06448` | Rejected; all six exact keys regressed or were neutral, with shared-down M8192 at `1.29020`. |
| WGM2, WGM4, WGM8 one-buffer variants | `1.08477`, `1.14106`, `1.23394` | Rejected; each lost materially on the short and long shared-down keys. |
| One-buffer SIA5 plus store priority | `1.06780` | Rejected; the interaction did not transfer the Q4 attention-family gain to Q5. |
| One-buffer packed lane sharing 2 | `1.09449` | Rejected; sharing overhead outweighed any payload-load benefit. |
| Two-buffer scalar extraction at narrow M2048 | `1.01395` | Correct but rejected under the 2% resource-bearing gate. |
| Two-buffer SIA5 plus store priority | `1.00057` | Correct but neutral; retain SIA4 as the control. |
| Two-buffer SIA3/PGR1 | Historical runtime illegal memory access | The emitter defect is fixed: the steady loop now loads current A before restoring the next-tile reduction coordinate, and the exact production key passes HIP, independent reference, and both producer mutations. It remains unselected pending renewed serial timing. |
| Two-buffer DepthU64 | Historical mismatch against HIP | The four-row Q5 metadata address, k32/k48 decoded-LDS reads, current-A pointer restoration, and final A-pointer handoff are fixed. The exact production key now passes HIP, independent reference, and both producer mutations; it remains unselected pending renewed serial timing. |
| Vector metadata load plus dynamic scale-byte extraction | `1.00876` | Correct, with 12 fewer VMEM instructions but 16 more VALU issues; rejected, including `1.05063` at shared-down M8192. |

The Q5 low-payload nibble-half shift is invariant within a decode phase, so a two-instruction hoist was implemented and passed all six-key producer-mutation checks. Its first matrix screen reached `0.98519` weighted candidate/control latency, with five exact keys improving. Isolated shared-down `M8192` rechecked at `1.06616`, so the hoist is selected for the other five exact keys and the original per-chunk path is selected for that key. A follow-on `v_lshl_or_b32` fusion replaced the high-plane shift and OR, reduced the selected N128 static VALU issue count to 920 (948 for the no-hoist M8192 key), and produced the prior pipeline catalog's 25-repeat weighted candidate/HIP latency ratio of `0.76592`.

### Lower bounds and bottleneck explanation

For each materially different Q5 family, generate complete fused, WMMA/A/LDS-floor, and decode/LDS-floor artifacts when timing diagnosis is ambiguous. Preserve launch geometry, ABI, and hard resource checks. Explain the remaining gap using the floor relationship, payload traffic, decode issue rate, LDS synchronization, occupancy, and supported counters.

Initial long-key lower bounds now identify separate behavior. Narrow `M32768,N2048,K512` measured complete `2.4545 ms`, WMMA floor `1.5872 ms`, decode floor `1.1003 ms`, and floor sum `109.5%` of complete, indicating overlap between WMMA and decode/LDS work. Shared-down `M32768,N512,K2048` measured complete `3.5218 ms`, WMMA floor `2.4191 ms`, decode floor `1.0844 ms`, and floor sum `99.5%`, making the WMMA/A/LDS path dominant. The final selected path removes one high-plane fusion issue without changing those resource limits; no remaining payload or synchronization mechanism produced a qualifying gain.

### Current padded-geometry confirmation

A changed-premise review transferred unswizzled `LdsPadB=8`, `256x64`, PGR2, and SIA5 from the reopened Q3 short-K campaign. The combination improves the three Q5 narrow keys by 15-18% versus the prior selected assembly and is retained with WGM2 at M2048 and WGM1 at M8192/M32768. Shared-down retains the prior `128x128` two-buffer path: padded `256x64`, `128x64`, and `128x128` one-buffer candidates remain 7-46% slower than selected assembly controls.

The authoritative mixed-catalog 25-repeat candidate/HIP ratios are narrow `0.7127/0.7088/0.6942` and shared-down `0.8201/0.7442/0.6240` in M2048/M8192/M32768 order. The call-weighted ratio is `0.67843x`, improving the prior approximately `0.763x` catalog. Every exact key beats HIP. Narrow candidates use 238 VGPRs, 16 SGPRs, 5 KiB LDS, 32 static WMMAs, 150 VMEM instructions, and 32 LDS instructions. Shared-down retains 220 VGPRs, 16 SGPRs, and 16 KiB LDS. All selected artifacts have zero private bytes and spills, pass both producer mutations and independent references, and are byte-identical across independent final roots.

The final small schedule scan compared SIA4 and SIA5 on the new geometry. Q5 WGM1 SIA4 was below the 2% retention gate. WGM2/SIA4 appeared 6.9% faster at M2048 in a nine-repeat screen but became `1.0178x` versus SIA5 in the 25-repeat rotating bracket, so it is rejected. WGM4/WGM8, store-priority changes, pad 16/24, and packed lane sharing remain measured rejections. The lower bounds continue to explain shared-down as WMMA/A/LDS dominated and narrow as an overlapped WMMA/decode/LDS path; no remaining in-contract Q5 mechanism has a qualifying measured gain path.

### Prior pipeline confirmation

The prior catalog used the Q5 packed extraction path, `v_lshl_or_b32` high-plane fusion on all six keys, the nibble-shift hoist on five keys, and the original nibble-shift path for shared-down `M8192`. Its authoritative protocol used 10 warmups, 25 rotating same-process repeats, serial phases, and HIP as the performance control.

| Family | `(M,N,K)` | HIP ms | GGTensile ms | HIP TFLOPS | GGTensile TFLOPS | Relative gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Narrow | `(2048,2048,512)` | `0.214` | `0.180` | `20.112` | `23.889` | `18.8%` |
| Narrow | `(8192,2048,512)` | `0.810` | `0.680` | `21.211` | `25.271` | `19.1%` |
| Narrow | `(32768,2048,512)` | `3.032` | `2.590` | `22.664` | `26.532` | `17.1%` |
| Shared down | `(2048,512,2048)` | `0.244` | `0.204` | `17.581` | `21.014` | `19.5%` |
| Shared down | `(8192,512,2048)` | `1.483` | `1.156` | `11.583` | `14.861` | `28.3%` |
| Shared down | `(32768,512,2048)` | `5.572` | `3.461` | `12.332` | `19.853` | `61.0%` |

Using call counts, weighted HIP latency is `158.166 ms` and weighted GGTensile latency is `120.663 ms`, a `23.7%` reduction. Narrow latency falls `14.9%` and shared-down latency falls `33.9%`. The largest absolute throughput gain is shared-down `M32768` at `+7.520 TFLOPS`; the largest relative gain is the same key at `61.0%`. The throughput formula is `2*M*N*K/(median_ms*1e9)` and represents complete fused-kernel arithmetic throughput, not WMMA-only throughput.

## Optimization-Exhaustion Review

### Renewed executable work

The emitter repairs reopened two exact Q5 candidates: SIA3/PGR1 with the corrected two-buffer handoff and the corrected double-buffer DepthU64 path. They must receive a fresh serial nine-repeat screen followed by 25-repeat confirmation against the current selected assembly and HIP controls. The repaired candidates remain candidate-only until every exact key they affect passes independent-reference, grad-output mutation, packed-weight mutation, resource, and reproducibility gates.

The Q5 review also records the metadata owner-load plus wave-local DPP broadcast as a Q4-targeted emitter experiment, not a transferable Q5 knob. A Q5 implementation requires its own metadata ownership proof and exact-format timing. Combined padded-stride plus logical-K XOR placement is conditional on supported LDS counters and an occupancy-preserving alternate emitter. An exact-N software-pipeline family is a separate structural experiment, while exact-shape literal/fixed-trip specialization is low priority. Model-owned integer-plus-scale preparation and all prepared representations remain outside scope.

Fresh execution closes both repaired candidates against the current selected M2048 narrow assembly. SIA3/PGR1 measured `0.214825 ms` versus `0.153590 ms` for the selected control (`1.39869x`), and double-buffer DepthU64 measured `0.217225 ms` versus `0.152452 ms` (`1.42487x`). Both remained bit-exact to HIP, matched HIP's independent-reference error, and passed grad-output and packed-weight mutations. Neither advances to confirmation, and the selected catalog is unchanged.

This is the permanent final step, and it must be repeated recursively. Re-read this plan and the Q4_K experiment log; all dense and grouped MMQ optimization histories; current Q5 and Q4 HIP, CK, GGTensile, and normalized-disassembly sources; inventories, selected and rejected solutions, manifests, lower bounds, profiles, counters, timing brackets, and mutation results; TensileLite, EvoTensile, CK, hipBLASLt, and relevant external mechanism studies; and the gfx1151 ISA reference plus LLVM AMDGPU instruction, VOPD, hazard, delay, and scheduling definitions and tests.

The renewed recursive pass classifies the remaining mechanisms as follows: padded one-buffer `256x64` ownership is retained for narrow after the changed Q3 premise; padded `128x64`/`128x128` and all padded shared-down variants are measured rejections; alternate WGM and SIA schedules, scalar extraction, lane sharing, vector metadata, repaired SIA3/PGR1, and repaired double-buffer DepthU64 are measured rejections; persistent workgroups, split-K, prepared weights, and cache-policy ownership expand the contract without a first-order lower-bound opportunity; native packed BF16 arithmetic and additional VOPD pairings are unavailable or non-bit-exact on gfx1151; and b128 payload loads, `v_lshl_or_b32`, exact-trip specialization, and the selected exact-key WMMA/LDS ownership are retained. The renewed timing closes the repaired paths and restores the optimization-exhaustion conclusion with the selected catalog unchanged.

A plan edit or implementation change that creates a new premise invalidates the prior stopping condition. The review must remain the last campaign step and cannot pass in the same iteration that discovers actionable work.

## Completion Record

This section is updated after each coherent implementation or experiment milestone. Documentation-only updates remain uncommitted unless explicitly requested.
- Q5_K multi-quant model and strict ProblemType support. Q4_K and Q5_K now have distinct strict problem identities, hashes, and symbols; Q5 validation accepts only the initial production N values 512 and 2048.
- Q5_K quant backend with shared WMMA body. Q5_K loads the 176-byte block's `qh` and `qs` planes, reuses the K-family scale/minimum path, and shares the Q4-compatible A, LDS, WMMA, accumulation, store, and exact-trip body.
- Six-key inventory and immutable campaign runner. The existing runner now derives family and packed-row rules from the inventory quant type; `q5_k_dense_inventory.json` and `q5_k_selected_solutions.json` define the six exact keys and initial two-buffer control.
- Complete correctness and producer-mutation validation. All six keys pass HIP, independent Q5 dequantized BF16 reference, grad-output mutation, and packed-weight mutation checks. The initial two-buffer control is resource-clean at 220 VGPRs, 16 SGPRs, and 16 KiB LDS.
- Initial bounded optimization screen. One-buffer ownership, WGM2/4/8, store-priority, lane-sharing, scalar extraction, SIA5, SIA3, and DepthU64 alternatives were generated, inspected, and measured or rejected by correctness/runtime gates; no common candidate cleared the retention threshold.
- Q5 decode lowering and exact-key selection. The nibble-shift hoist is selected for five keys, while the isolated shared-down M8192 regression uses the original emitter; `v_lshl_or_b32` is selected for all six keys. Both paths are strict, reproducible, resource-clean, and mutation-correct.
- Q5 metadata-load review. The vector metadata path is correct but rejected after its VMEM reduction lost to dynamic scale-byte extraction overhead.
- Initial lower-bound diagnosis. Narrow long-key execution has overlapping WMMA/decode components; shared-down long-key execution is WMMA/A/LDS-bound.
- Per-key optimization and confirmation. The final six-key catalog passes complete correctness and 25-repeat confirmation, with every exact key faster than HIP.
- Changed-premise padded-geometry review. Padded `256x64` is selected for all narrow keys; padded shared-down ownership and alternate SIA/WGM schedules are rejected by final rotating brackets.
- Repaired-path retiming. Corrected SIA3/PGR1 and double-buffer DepthU64 pass exact correctness and mutation gates but regress the current selected M2048 assembly by `39.9%` and `42.5%`; both are closed without confirmation.
- Recursive final optimization-exhaustion review with no actionable mechanism remaining. The review was repeated after the `v_lshl_or_b32` discovery and after the rejected vector metadata emitter.
- Public runtime dispatch. Deferred until broader quant coverage and complete workload validation.
