# GGTensile MMQ Forward Q3_K Experiment

## Purpose

This record covers the isolated GGTensile MMQ forward Q3_K control and its later 12-key dense expansion on gfx1151. The control validates the Q3_K packed decoder, Q8_1 `F32_D4` workspace contract, wave32 WMMA mapping, and row-major BF16 output ownership. The expansion records independently qualified exact research candidates for all 12 dense keys; public dispatch remains on HIP during this campaign.

HIP remains the correctness and timing control and the exact fallback at the public boundary.

## Exact Scope

The operation is:

```text
output[M,N] = input[M,K] @ dequant(weight[N,K]).T
```

The first exact control is:

```text
(M,N,K) = (2048, 4096, 2048)
tensor = blk.4.attn_gate.weight
logical weight = (4096, 2048)
physical GGUF weight = (4096, 880)
```

Q3_K uses 110-byte blocks containing `hmask[32]`, `qs[64]`, `scales[12]`, and FP16 `d`. The activation producer is the fixed HIP Q8_1 `F32_D4` workspace with shape `(16, 2048, 144)`. No prepared weights, dense shadow, external decode storage, split-K reduction, persistent workgroup, grouped path, inventory coverage, or public integration is in scope.

The control uses workgroup `(32,4,1)`, a `128x64` macro tile, depth `16`, four wave32 waves, and 128 integer WMMA operations. It cooperatively stages activation rows and directly decodes signed Q3 payload rows into LDS before scale-corrected WMMA accumulation and BF16 RNE stores.

## Implementation

The Q3 backend adds typed semantic support for:
- Q3 block planes and physical byte sizing.
- Signed 3-bit reconstruction from the low 2-bit payload and high mask.
- Packed signed six-bit scale fields across all 16 scale groups.
- FP16 block factor `d` and FP32 scale correction.
- Q8_1 `F32_D4` activation addressing and scale use.
- Formula-derived Q3 LDS layout and register resources.
- An independent Q3 dequantization and matmul reference.

The isolated writer uses explicit Q3 register roles and deterministic lifetime-aware first-fit allocation. The qualified shared-VMEM and loop-carried weight-prefetch control uses formula-derived resources of 144 VGPRs, 16 SGPRs, and 28,672 bytes of LDS, with zero private storage and zero spills. It retains the persistent WMMA zero fragment, activation LDS-base hoist, activation-scale deduplication, full activation-stage batching, shared activation/weight VMEM issue, and typed next-block weight prefetch.

The initial device mismatch exposed two separate scale/control issues. First, encoded Q3 scale value `33` was lowered through signed six-bit extraction as `-31`; Q3 scale values are unsigned six-bit codes offset by 32, so the writer now subtracts 32 and converts the result as signed FP32. Second, the direct HIP test launcher initially allocated 38,400 bytes of LDS for Q3. The HIP Q3 shared layout requires 40,448 bytes from its 36-dword activation tile and 84-dword packed Q3 row stride. Correcting that control allocation removed the HIP-only tail-column corruption.

## Correctness and Resources

The final 25-repeat qualification run passed:
- Candidate versus HIP multiply: 0 differing BF16 elements out of 8,388,608; normalized RMSE `0.0`; maximum absolute error `0.0`.
- Candidate versus public complete path: 0 differing BF16 elements; normalized RMSE `0.0`.
- Candidate versus independent reference: normalized RMSE `0.006061011`, maximum absolute error `0.0390625`.
- Producer repeat: 0 differing bytes out of 4,718,592.
- Input mutation: 8,388,608 changed output elements.
- Packed-weight mutation: 2,007 changed output elements.
- Workspace mutation: 548 changed output elements.
- All outputs were finite.

Strict code-object inspection accepted code-object version 5, target gfx1151, wave32, the 40-byte kernarg ABI, 144 VGPRs, 16 SGPRs, 28,672 bytes of LDS, zero private bytes, and zero VGPR/SGPR spills. Static counts were 128 WMMAs, 4 barriers, 2,757 VALU issues, 91 VMEM operations, 370 LDS operations, 21 waits, and 8 clauses. The assembly contains 1,028 VOPD instructions and 3,785 VALU operations in total.

The accepted artifact is:

```text
kernel = torch_ggml_ops_ggtensile_gfx1151_v1_mmq_fwd_q3_k_m2048_n4096_k2048_31c9031538e012be
solution = ggsol_31c9031538e012be
```

Two independent CLI generate, assemble, build, and inspect roots reproduced matching generated source and normalized inspection results.

## Timing and Decision

The authoritative warmed rotating run used five warmup launches and 25 samples. Speedup is HIP median divided by GGTensile median; values below `1.0x` would favor GGTensile.

| Path | HIP median | GGTensile median | GGTensile/HIP time |
| --- | ---: | ---: | ---: |
| Multiply only | `1.553431 ms` | `1.457108 ms` | `0.9380x` |
| Complete call | `1.590976 ms` | `1.473285 ms` | `0.9260x` |

The final control is correctness-qualified, resource-clean, and faster than HIP on this exact shape in both measured paths. It remains isolated measured research evidence: no Q3 inventory, selected solution, dispatch rule, or public bundle entry is authorized until a separate broader-shape promotion phase is complete.

## Verification

The focused GGTensile suite passes with 284 tests and the full repository suite passes with 369 tests and the existing 14 warnings. Ruff, formatting, `ty`, compileall, pre-commit, and `git diff --check` pass. Regeneration of the frozen pre-Q8 catalog sources produces 447 expected and generated sources with zero missing, added, or changed files. The public gfx1151 bundle remains current at 179 kernels.

This exact-shape Q3 control has completed its isolated parity campaign. Expansion still requires a new exact-shape premise and fresh correctness, resource, deterministic-build, and warmed timing evidence; no inventory, selected catalog, dispatch, or public bundle change is authorized by this result.

## Active Optimization Plan

Start with the exact `(2048,4096,2048)` control and use the HIP source/disassembly as an operand-ownership oracle. Prioritize complete mechanisms that can remove the current decode/LDS/WMMA overlap deficit:
- Reconstruct HIP's Q3 decoded-row layout, packed payload reuse, scale ownership, and activation-read reuse as typed semantic alternatives, without importing its physical instruction stream.
- Measure alternate Q3 LDS representations and wave ownership only when the layout, resource formula, synchronization, and output mapping are complete.
- Test decode/scale overlap, payload load grouping, local-read grouping, and epilogue ownership as linked dataflow variants; previously rejected schedule-only changes remain closed.
- Retain the exact VOPD scale-correction mechanism as the current arithmetic baseline: it reduces the scalar correction body to 512 dual multiplies and 512 dual FMACs while preserving bit identity, but does not by itself establish promotion.
- Retain batched activation payload and scale LDS reads as the current local-read baseline: one group wait replaces eight serialized group waits, reducing static waits from 151 to 39 at 136 VGPRs. Its first warmed screen reached 1.825502 ms multiply and 1.842309 ms complete versus HIP at 1.527567 ms and 1.563694 ms, so it remains research-only.
- Test a persistent eight-VGPR zero accumulator source for WMMA. The current body clears all eight temporary accumulator registers before every WMMA even though the instruction accepts a distinct accumulator source; a typed zero-fragment role initialized once can remove 1,016 static zeroing issues without changing integer results, scale order, ownership, synchronization, or ABI. Reject it immediately on any bit mismatch, spill/private-storage result, resource-limit violation, or warmed timing regression.
- The persistent zero-fragment candidate passed exact HIP/public identity, mutation, independent-reference, finiteness, and resource checks at 144 VGPRs, 16 SGPRs, and 28,672 bytes of LDS. It reached 1.687218 ms multiply and 1.695372 ms complete versus HIP at 1.543170 ms and 1.572854 ms, improving but not reaching parity.
- Test a two-fragment compute batch that keeps two WMMA results live and pairs FMACs across their two distinct activation-scale registers. This removes the per-fragment scale-copy moves while retaining `integer_result * weight_scale * activation_scale` order; the formula-derived resource target is 152 VGPRs, 16 SGPRs, and 28,672 bytes of LDS. Reject it on any bank, correctness, resource, deterministic-build, or warmed timing failure.
- Reject the two-fragment compute batch. It remained bit-exact and spill-free at 152 VGPRs, but reached 1.685468 ms multiply and 1.709811 ms complete versus HIP at 1.533288 ms and 1.573708 ms. The higher register footprint did not provide a stable gain over the 144-VGPR persistent-zero baseline.
- Hoist the activation LDS lane base into one typed register before the block loop. The base `(lane & 15) * 144` is invariant across every Q3 group, half, and block, but the current lowering recomputes it before every one of the eight activation-row reads in every group. The hoist removes 254 redundant static address issues while preserving the existing LDS offsets, waits, arithmetic, ABI, and 144-VGPR resource class.
- The activation-address-hoist candidate passed exact HIP/public identity, mutation, independent-reference, finiteness, and zero-spill checks at 144 VGPRs, 16 SGPRs, and 28,672 bytes of LDS. A 25-repeat rotating run reached 1.467826 ms multiply and 1.489548 ms complete versus HIP at 1.523732 ms and 1.559132 ms, or 0.963310x and 0.955370x. It is the retained Q3 performance baseline, but no production promotion is authorized yet.
- Test paired weight-scale LDS reads. The eight scale values for each Q3 group are at four pairs of offsets within the 8-bit `ds_load_2addr_b32` range when four address roles are based at rows 0, 4, 8, and 12. This should halve the weight-scale local-read instruction count without changing values; its formula-derived target is 152 VGPRs, 16 SGPRs, and 28,672 bytes of LDS. Reject it on any assembler bank/address error, correctness mismatch, resource failure, deterministic rebuild mismatch, or warmed regression.
- Test activation-scale read deduplication independently of the paired weight-scale mechanism. Q3 local groups `0/1`, `2/3`, `4/5`, and `6/7` use the same Q8_1 activation-scale LDS offsets, so odd groups can reuse the even-group register values after the existing group wait. This removes 64 static LDS reads while preserving the 144-VGPR class, scale arithmetic order, and exact output mapping. Reject it on any mapping, mutation, correctness, resource, deterministic-build, or warmed-timing failure.
- The activation-scale deduplication passed exact correctness, mutation, independent-reference, and resource gates at 144 VGPRs, with 370 static LDS operations. Its 25-repeat run reached 1.492152 ms multiply and 1.502795 ms complete versus HIP at 1.543599 ms and 1.584539 ms, or 0.966671x and 0.948412x. It is retained as an ingredient of the final prefetch control.
- Re-test the two-fragment cross-M FMAC batch after address hoisting and activation-scale deduplication. The earlier 152-VGPR result was measured before the 254-address-issue hoist; this isolated combination can remove the per-fragment scale-copy moves while retaining the now-reduced local-read body. Its target remains 152 VGPRs, 16 SGPRs, and 28,672 bytes of LDS, with no change to per-output arithmetic order.
- Reject the post-hoist cross-M FMAC batch. It remained exact, mutation-sensitive, and spill-free at 152 VGPRs and 370 static LDS operations, but its 25-repeat run reached 1.497351 ms multiply and 1.512065 ms complete versus HIP at 1.538512 ms and 1.583485 ms, or 0.973246x and 0.954897x. The retained 144-VGPR activation-scale-dedup control has the better multiply path.
- Test full activation-stage VMEM batching. The 144-byte Q8_1 row currently uses nine serialized `global_load_b128`/`vmcnt(0)`/LDS-store sequences per half. The activation-stage lifetime is disjoint from the 69-register compute peak, so a typed 36-VGPR stage can issue all nine loads before one wait without changing the formula-derived 144-VGPR kernel class, LDS layout, barriers, or ABI. Reject it on any register-allocation, correctness, resource, or warmed-timing failure.
- Retain full activation-stage VMEM batching. Explicit allocation order reuses the 36-register stage with lifetime-disjoint compute storage, so inspection remains at 144 VGPRs rather than the fragmented first-fit maximum of 152. The accepted intermediate artifact has 144 VGPRs, 16 SGPRs, 28,672 bytes of LDS, zero spills/private storage, 128 WMMAs, 370 LDS operations, and 23 waits. Its 25-repeat rotating run reached 1.456794 ms multiply and 1.478746 ms complete versus HIP at 1.540078 ms and 1.569895 ms, or 0.945922x and 0.941939x; the later prefetch control supersedes it.
- Test low-pressure paired weight-scale reads on top of the retained stage-batched control. Keep two persistent row-pair bases and derive the other two bases into the existing temporary pair before each group; four `ds_load_2addr_b32` instructions then replace eight scalar scale reads. The formula remains in the rounded 144-VGPR class. This candidate is actionable because it combines both independent LDS reductions without the rejected 152-VGPR four-address footprint; reject it on any correctness, bank/address, resource, or warmed-timing failure.
- Reject low-pressure paired weight-scale reads. The candidate assembled at 144 VGPRs with 306 LDS operations, remained bit-exact and mutation-sensitive, and passed the independent reference, but its 25-repeat run reached 1.490659 ms multiply and 1.509192 ms complete versus HIP at 1.543457 ms and 1.585781 ms, or 0.965792x and 0.951703x. It regressed the retained stage-batched control's 0.945922x multiply and 0.941939x complete medians; the static LDS reduction does not justify the extra address arithmetic.
- Test shared activation/weight VMEM issue. Issue the nine activation row loads and three raw Q3 loads before one `s_waitcnt vmcnt(0)`, store the now-ready activation rows before weight decode overwrites any lifetime-disjoint staging registers, and then continue with the existing decode and barriers. Model the raw weight operands, weight stage address, and scale shift as overlapping the activation-stage lifetime; reject on any allocator/resource, correctness, or warmed-timing failure.
- Retain shared activation/weight VMEM issue. The packed 144-VGPR role order preserves typed raw operands, stage addresses, scale shifts, activation staging, and persistent WMMA zero operands without aliasing. The corrected candidate is exact and resource-clean at 21 waits; its nine-repeat screen reached approximately 0.9377x multiply and 0.9331x complete versus HIP. The unsafe temporary-alias implementation is rejected after memory faults and is not part of the retained source.
- Test loop-carried Q3 weight prefetch. Issue the first half's raw weight loads before the block loop, then after each completed half-1 barrier increment the weight pointer, issue the next half-0 raw loads, and let the next block's activation loads share the eventual VMEM wait. Keep the raw roles in their existing stage lifetimes and preserve the final-block bounds check; reject on any synchronization, address, correctness, resource, or warmed-timing failure.
- Retain loop-carried Q3 weight prefetch. The exact 144-VGPR candidate inspected with 91 VMEM operations, 370 LDS operations, 21 waits, and zero spills/private storage. Its final-root five-warmup, 25-repeat rotating run reached 1.457108 ms multiply and 1.473285 ms complete versus HIP at 1.553431 ms and 1.590976 ms, or 0.9380x and 0.9260x. It is the final isolated Q3 research control.
- Test Q3 half-metadata reuse on the prefetch control. The 16-byte metadata read is block-invariant across the two Q3 halves; retain its four registers through half-0 decode and skip the identical half-1 global read while still waiting for the half-1 payload reads. The first attempt was rejected because the four metadata registers overlapped the stage-3 `c` fragment registers and half-0 WMMA overwrote them before half 1, producing grossly incorrect output despite passing static resource inspection. A correctly typed lifetime extension forced a 152-VGPR placement and produced exact output with 90 VMEM operations, but its 25-repeat run reached 1.474360 ms multiply and 1.496622 ms complete versus HIP at 1.549685 ms and 1.590317 ms, or 0.9514x and 0.9411x; the selected 144-VGPR prefetch control remains faster.
- Use one-hot and cross-coordinate tests before timing every changed mapping, then run the full exact correctness, mutation, reference, inspection, and deterministic-build gates.
- Final Q3 qualification is complete for the retained prefetch control. Independent roots reproduce source and normalized inspection content; rebuilt objects and code objects pass the deterministic inspection checks; the final 25-repeat run is exact, mutation-sensitive, finite, and independently referenced; the focused/full suites, frozen 447-source comparison, and 179-kernel bundle check pass. Q3 remains isolated regardless of the result.

At the isolated-control stage, the exact-shape promotion gate was met for both multiply and complete medians, but the candidate remained research-only because Q3 catalog selection, broader shape validation, and public integration were separate decisions.

## Recursive Final Review

The final recursive review reread this record, the generic GGTensile plan, the Q4_K/Q5_K/Q6_K/Q8_0 forward records, the relevant Q3 HIP sources and disassembly, the generated source and inspection reports, rejected artifacts, resource formulas, and warmed timing evidence. The classification is:
- Retained and measured: typed Q3 signed reconstruction; persistent eight-VGPR WMMA zero source; activation LDS-base hoist; activation-scale deduplication; full activation VMEM batching; shared activation/weight VMEM issue; legal VOPD scale correction; deterministic 144-VGPR role order; and loop-carried half-0 weight prefetch with a final-block bounds check.
- Rejected by correctness or resource lifetime: unsafe temporary aliasing across the VMEM wait; the first metadata-reuse attempt, where metadata overlapped `c` and was overwritten; and any untyped reuse of raw weight operands, stage addresses, scale shifts, activation staging, or persistent zero operands.
- Rejected by warmed timing: paired weight-scale LDS reads, the two-fragment cross-M FMAC batch, low-pressure paired scale reads, the 152-VGPR prefetch placement, and the correctly typed 152-VGPR metadata-reuse variant. The earlier VOPD, batched activation, and persistent-zero intermediate variants remain recorded with their parent comparisons.
- Contract-incompatible or deferred: prepared or dense weights, external decode workspaces, split-K, persistent/grouped traversal, producer fusion, hidden caches, public dispatch/bundle changes, copied HIP/LLVM schedules, and broader Q3 shape/catalog expansion without exact-key evidence. Additional LDS buffering would require a new ownership and residency premise rather than a schedule edit.
- No actionable in-contract mechanism remains under the fixed exact shape, ABI, LDS layout, wave ownership, and zero-spill resource contract. The remaining measured envelope is the 144-VGPR body at 128 WMMAs, 370 LDS operations, 91 VMEM operations, 21 waits, and 1,028 VOPD instructions; every identified traffic, lifetime, scale-correction, and register-class opportunity either is retained or has an exact rejection. The final control beats HIP on both required paths, so there is no residual performance deficit requiring an unmeasured mechanism.

The isolated-shape campaign completed without changing the public boundary. Production remained on HIP pending a separate promotion phase with additional exact-key validation and an explicit catalog and dispatch decision.

## Dense Expansion Campaign

The dense campaign is reopened against the local `Qwen3.6-35B-A3B-APEX-I-Mini.gguf` model. Direct GGUF enumeration found 159 Q3_K tensors in seven logical shape classes. The dense MMQ scope contains the four 2-D matrix families whose dimensions satisfy the existing 64-column dense tile contract, each at flattened activation rows `M={2048,8192,32768}`:

| Family | Logical weight `(N,K)` | Representative tensor | Tensor count | Exact keys |
| --- | ---: | --- | ---: | ---: |
| attention K | `(512,2048)` | `blk.3.attn_k.weight` | 9 | 3 |
| attention Q | `(8192,2048)` | `blk.3.attn_q.weight` | 9 | 3 |
| attention gate | `(4096,2048)` | `blk.4.attn_gate.weight` | 25 | 3 |
| SSM output | `(2048,4096)` | `blk.4.ssm_out.weight` | 25 | 3 |

The 40 three-dimensional `(256,512,2048)` expert tensors belong to grouped MMQ and are excluded from this dense campaign. The 50 `(32,2048)` SSM alpha/beta tensors do not satisfy the dense 64-column tile and installed HIP MMQ control contract. `token_embd.weight` is an embedding lookup rather than a dense MMQ call. The model output tensor is Q6_K and remains covered by its separate campaign.

The initial priority follows estimated weighted matrix work before measurement: the 25-call attention-gate and SSM-output families first, then the wider nine-call attention-Q family, then attention K. Within each family the `M=32768` and `M=8192` keys receive the first optimization attention, while every exact key must independently beat HIP multiply before selection. The existing `(2048,4096,2048)` attention-gate control is a qualified seed, not evidence for another key. Baseline work will regenerate the current seed for all 12 keys, measure serial warmed HIP and GGTensile medians, and replace this estimate with measured weighted latency.

### Current-Control Expansion Screen

The first serial screen regenerated the final 144-VGPR prefetch control for every key. All nine `K=2048` runs passed exact candidate/HIP and candidate/public agreement, finiteness, producer repeatability, and input, packed-weight, and workspace mutation gates. Independent-reference work was intentionally deferred from this prioritization screen. Every artifact inspected at 144 VGPRs, 16 SGPRs, 28,672 LDS bytes, zero private bytes, zero spills, 128 static WMMAs, 2,757 VALU issues, 91 VMEM operations, 370 LDS operations, 21 waits, and eight clauses.

| Family | `M` | HIP multiply | Control multiply | Control/HIP |
| --- | ---: | ---: | ---: | ---: |
| attention K | 2048 | `1.307646 ms` | `2.523370 ms` | `1.92970x` |
| attention K | 8192 | `1.775234 ms` | `3.394456 ms` | `1.91212x` |
| attention K | 32768 | `3.799897 ms` | `5.382511 ms` | `1.41649x` |
| attention Q | 2048 | `3.787316 ms` | `5.427558 ms` | `1.43309x` |
| attention Q | 8192 | `12.774316 ms` | `13.717089 ms` | `1.07380x` |
| attention Q | 32768 | `48.742268 ms` | `48.143223 ms` | `0.98771x` |
| attention gate | 2048 | `2.438180 ms` | `4.216428 ms` | `1.72933x` |
| attention gate | 8192 | `6.799119 ms` | `8.019713 ms` | `1.17952x` |
| attention gate | 32768 | `24.866899 ms` | `25.330601 ms` | `1.01865x` |

These ratios are prioritization evidence only. The regenerated attention-gate `M=2048` source and code object are byte-identical to the final isolated qualification artifact, but a second 25-repeat run measured `1.72433x` instead of the prior retained `0.9380x`. A clock monitor showed the gfx1151 shader clock ramping from approximately 0.66 GHz to only 2.13 GHz during a short process despite a 2.9 GHz ceiling. The control and HIP paths respond differently to this cold operating range, so future screens use materially longer in-process warmup while remaining below five minutes per experiment.

The first `K=4096` run generated, built, and inspected cleanly with the same resource envelope, but benchmarking initially stopped before launch because `FixedHipForwardModule` rejected Q3_K when `K != 2048`. The tooling oracle now selects the existing exact installed `k2048_j128_full` symbol for `K=2048` and the already bundled generic `j128` symbol for `K=4096`; focused tests lock both choices and preserve the 40,448-byte Q3 HIP LDS allocation. This is tooling-only and does not change public dispatch or the 179-kernel bundle.

The resulting SSM-output `(2048,2048,4096)` run passed exact candidate/HIP and candidate/public agreement for baseline and mutations, zero producer differences, finiteness, and nonzero input, weight, and workspace mutation sensitivity. After 500 rotating warmup rounds, 25 measured repeats gave HIP multiply `2.536122 ms` and GGTensile multiply `4.378514 ms`, or `1.72646x`; complete medians were `2.547640 ms` and `4.356055 ms`, or `1.70984x`. Longer warmup did not recover the historical isolated-control operating regime. This is a correctness-retained but timing-rejected control for K=4096. No new exact key is selected; the widest `M=32768` attention-Q result advances to repeated warm qualification, while SSM output, narrow N, and short M require different ownership or geometry.

### MT64 Tiled-LDS Experiment

The first geometry alternative halves the activation ownership from 128 to 64 rows while retaining four N-owning waves and the 64-column output tile. Canonical Q3 ownership is now explicitly wave-N: `(MIWaveGroupM,MIWaveGroupN)=(1,4)` with four or eight M fragments for MT64 or MT128. The previous `(4,1)` projection was inconsistent with the emitted mapping but happened to produce the same MT128 static count; strict MT64 inspection exposed and corrected it. The new physical plan derives 104 VGPRs, 16 SGPRs, and 19,456 LDS bytes, compared with 144 VGPRs and 28,672 bytes for MT128. It emits 64 WMMAs, 59 VMEM operations, 274 LDS operations, 21 waits, and four clauses with zero private storage and zero spills. Only lanes 0 through 15 of each wave stage the 64 unique activation rows; full execution is restored before weight decode and every barrier. The default MT128 source remains byte-identical.

The `(2048,512,2048)` attention-K screen passed exact HIP/public agreement, repeatability, finiteness, and all mutation gates. After 500 rotating warmup rounds, HIP and MT64 multiply medians were `1.047420 ms` and `1.847428 ms`, or `1.76379x`; complete medians were `1.058482 ms` and `1.853931 ms`, or `1.75150x`. MT64 materially reduces the control's candidate latency but remains decisively slower than HIP.

The remaining narrow keys also passed the same correctness and artifact gates but rejected MT64 on timing: `M=8192` measured HIP `1.589474 ms` versus candidate `3.273259 ms` (`2.05933x` multiply, `1.58137x` complete), and `M=32768` measured HIP `3.776948 ms` versus candidate `5.404684 ms` (`1.43097x` multiply, `1.24120x` complete). The lower LDS/VGPR footprint does not compensate for predicated activation staging and reduced per-workgroup reuse. MT64 is rejected for the full narrow family; MT128 remains the only retained Q3 geometry while a different ownership/dataflow mechanism is investigated.

### Rolled Scale-Group Experiment

The HIP disassembly uses one 40,448-byte workgroup and 196 VGPRs but rolls its scale-group compute to 16 static WMMAs. An explicit `Q3RolledTiledLds` alternative tested whether the 3,520-line unrolled instruction footprint was the residual cause while retaining MT128, the compact half-tile LDS layout, arithmetic order, four barriers, and 144-VGPR class. Typed incremental payload, weight-scale, activation-payload, and activation-scale addresses reduced static WMMAs from 128 to 16, VALU issues from 2,757 to 869, LDS instructions from 370 to 84, and waits from 21 to seven. The artifact remained at 144 VGPRs, 16 SGPRs, and 28,672 LDS bytes with no private storage or spills.

The rolled `(2048,4096,2048)` candidate passed exact HIP/public agreement, repeatability, finiteness, and mutation gates, but 500-warmup, 25-repeat medians were HIP `2.570757 ms` and candidate `4.496834 ms`, or `1.74923x` multiply; complete was `1.74477x`. It was also slightly slower than the unrolled control in the same operating regime. Static instruction footprint is not the governing deficit; loop-carried addresses and scalar group branches reduce scheduling freedom. The rolled mechanism is rejected and no key selects it.

### Linear Activation-Staging Experiment

An explicit `Q3LinearActivationTiledLds` alternative tested the HIP control's linear cooperative activation load pattern. Each thread staged nine 16-byte chunks at a 2,048-byte workgroup stride, making adjacent lanes access adjacent chunks while producing the same row-major LDS plane. A temporary VMEM base advanced by 4,096 bytes kept global offsets within the gfx1151 signed 13-bit range. The default source remained byte-identical and the alternative artifact retained 144 VGPRs, 16 SGPRs, 28,672 LDS bytes, four barriers, and zero private storage or spills.

The linear candidate passed exact HIP/public agreement, repeatability, finiteness, and mutation gates. For `(2048,4096,2048)`, 500-warmup, 25-repeat medians were HIP `2.566249 ms` and candidate `4.474045 ms`, or `1.74342x` multiply; complete was `1.73891x`. This was indistinguishable from the strided control deficit. Activation transaction shape is not the governing limitation, so the mechanism is rejected and no key selects it.

### Batched-WMMA Experiment

HIP issues the eight independent M-fragment WMMAs for one Q3 scale group before converting and correcting any result, while the retained writer immediately corrected each fragment. `Q3BatchedWmmaTiledLds` widened transient C ownership from eight to 64 VGPRs and preserved each output's group accumulation order. Strict inspection reported 200 VGPRs, 16 SGPRs, 28,672 LDS bytes, four barriers, 128 static WMMAs, and no private storage or spills; the 200-VGPR result is close to the HIP control's 196.

The candidate passed exact HIP/public agreement, repeatability, finiteness, and all mutation gates. For `(2048,4096,2048)`, 500-warmup, 25-repeat medians in the observed clock regime were HIP `3.893572 ms` and candidate `5.784447 ms`, or `1.48564x` multiply; complete was `1.40757x`. Batching materially narrows the ratio from roughly `1.74x`, confirming that WMMA issue order matters, but does not independently beat HIP. The standalone mechanism is rejected; the 64-result ownership is retained only as an experimental basis for dependency-derived partial LDS waits.

`Q3BatchedWmmaPartialWaitTiledLds` then launched each WMMA when its activation payload became ready. Even groups used dependency-derived waits `15,13,...,1` followed by `0` for scales; odd groups used `7,6,...,0`. The 200-VGPR artifact remained spill-free and passed every correctness gate. Medians were HIP `4.006524 ms` and candidate `5.856987 ms`, or `1.46186x` multiply; complete was `1.46726x`. Partial waits provide only a modest additional gain and remain slower than HIP. The mechanism is rejected independently and carried forward only into the rolled, batched schedule experiment.

The combined `Q3RolledBatchedWmmaPartialWaitTiledLds` artifact reduced static WMMAs to 16, VALU issues to 869, LDS instructions to 84, and waits to 23 while retaining 200 VGPRs and zero spills. It passed all correctness gates, but medians were HIP `3.977258 ms` and candidate `5.869282 ms`, or `1.47571x` multiply; complete was `1.47079x`. Rolling again failed to improve the batched mechanism, confirming that static footprint is not the residual limit. The combined mechanism is rejected and no key selects it.

### Paired-Scale Experiment

`Q3PairedScaleBatchedWmmaPartialWaitTiledLds` replaced eight scalar weight-scale reads with four `ds_read2_b32` operations and paired activation-scale reads with `ds_read2st64_b32`. The exact candidate inspected at 208 VGPRs, 16 SGPRs, 28,672 LDS bytes, 274 LDS instructions, 141 waits, and zero spills or private storage. It remained bit-exact and mutation-sensitive, but measured `5.880064 ms` versus HIP `3.916767 ms`, or `1.50125x` multiply; complete was `1.48759x`. Reducing static LDS instructions did not improve latency. The mechanism is rejected and its implementation was removed after preserving this evidence.

### Timing Environment Diagnosis

The retained source and code object were byte-identical across the historical and current runs. The old and regenerated HSACOs were byte-identical. Nevertheless, the short `(2048,4096,2048)` control changed from `0.93799x` HIP multiply historically to approximately `1.72x` under the later `auto` DPM regime. Sysfs reported a 600 MHz idle state and a 2.9 GHz ceiling; direct performance-level control required unavailable elevated privileges.

A sustained 64-launch qualifier raised observed SCLK from approximately 1.83 GHz to 2.53 GHz, but the source-identical control still measured `2.447658 ms` versus HIP `1.612404 ms`, or `1.51802x`. Clock ramp alone therefore does not explain the historical reversal. Short-shape promotion evidence from the later environment is rejected. Long-M kernels keep the device active long enough to produce tight repeated distributions, so final selection uses 100 rotating warmups and 101 measured repeats per exact key.

### Stable 12-Key Baseline

The uniform protocol completed every dense key. All candidates passed candidate/HIP and candidate/public agreement, finiteness, producer repeatability, mutation sensitivity, and strict artifact inspection. Only attention-Q at `M=32768` beat HIP multiply.

| Family | `M` | HIP multiply | GGTensile multiply | GGTensile/HIP |
| --- | ---: | ---: | ---: | ---: |
| attention K | 2048 | `1.470738 ms` | `3.462637 ms` | `2.35435x` |
| attention K | 8192 | `1.959970 ms` | `3.895425 ms` | `1.98749x` |
| attention K | 32768 | `4.029305 ms` | `5.705802 ms` | `1.41608x` |
| attention Q | 2048 | `4.006775 ms` | `5.977945 ms` | `1.49196x` |
| attention Q | 8192 | `13.558021 ms` | `14.631325 ms` | `1.07916x` |
| attention Q | 32768 | `53.398323 ms` | `52.925510 ms` | `0.99115x` |
| attention gate | 2048 | `2.575271 ms` | `4.486917 ms` | `1.74231x` |
| attention gate | 8192 | `7.235206 ms` | `8.483326 ms` | `1.17251x` |
| attention gate | 32768 | `28.400726 ms` | `30.313105 ms` | `1.06734x` |
| SSM output | 2048 | `3.551789 ms` | `4.866368 ms` | `1.37012x` |
| SSM output | 8192 | `7.141935 ms` | `8.683723 ms` | `1.21588x` |
| SSM output | 32768 | `28.440058 ms` | `30.227186 ms` | `1.06284x` |

Selecting the GGTensile control for all 12 keys would regress call-count-weighted multiply latency by `11.9901%`. Exact dispatch to only `(32768,8192,2048)` changes the weighted total from `2639.432809 ms` to `2635.177496 ms`, or `0.998388x`, saving `4.255314 ms` in the measured workload. The other 11 exact entries use `hip_fallback`.

### Historical Exact Promotion (Reverted)

Attention-Q `(32768,8192,2048)` passed two 25-repeat runs at `0.99221x` and `0.99568x` HIP multiply, a 101-repeat confirmation at `0.99115x`, and a rebuilt-public 101-repeat confirmation at `0.99185x`. The final public-environment medians were `52.881584 ms` versus HIP `53.315998 ms` for multiply and `53.250736 ms` versus `53.505711 ms` for complete, or `0.99185x` and `0.99523x`. Candidate and public outputs were bit-exact to HIP. Candidate and public independent-reference normalized RMSE was `0.006065104`, with finite output and maximum absolute error `0.046875`.

The temporary selected source was generated from the former split inventory and named-solution files. Those files were retired with the reverted integration; the current `mmq_fwd_q3_k_catalog.json` records only the later typed full-weight winners. The historical bundled HSACO was byte-identical to the independently qualified artifact. Inspection reported code-object v5, gfx1151, wave32, the 40-byte ABI, 144 VGPRs, 16 SGPRs, 28,672 static LDS bytes, zero private storage, and zero spills.

A temporary public integration experiment selected this assembly only for exact `(quant_type, M, N, K) = (Q3_K,32768,8192,2048)`. It launched with zero dynamic shared memory because LDS was statically declared by the assembly code object, and a traced public call remained bit-exact through all mutation gates. That temporary 180-kernel package was then removed when the campaign policy deferred all Q3 public wiring; the current source-built bundle contains 179 HIP-wrapper kernels.

All rejected experimental mechanism implementations were removed after recording their correctness, resource, and timing evidence. Current production retains the existing HIP selector; the typed full-weight mechanism remains research-catalog state only.

## Optimization Continuation Plan

The campaign is reopened because the HIP kernels prove an in-contract parity implementation exists and 11 exact keys remain on fallback. Priority follows call-count-weighted deficit and timing stability rather than raw ratio alone:

| Priority | Exact key | Calls | Stable baseline deficit |
| ---: | --- | ---: | ---: |
| 1 | attention gate `(32768,4096,2048)` | 25 | `+1.912379 ms/call`, `+47.809458 ms` weighted |
| 2 | SSM output `(32768,2048,4096)` | 25 | `+1.787128 ms/call`, `+44.678211 ms` weighted |
| 3 | SSM output `(8192,2048,4096)` | 25 | `+1.541789 ms/call`, `+38.544714 ms` weighted |
| 4 | attention gate `(8192,4096,2048)` | 25 | `+1.248120 ms/call`, `+31.202996 ms` weighted |
| 5 | attention Q `(8192,8192,2048)` | 9 | `+1.073304 ms/call`, `+9.659738 ms` weighted |

Short `M=2048` and narrow attention-K keys remain important but follow the stable large-M keys. Their larger ratios are confounded by the unlocked DPM regime and lower useful work per launch.

The first phase requalifies the previously correctness-clean WMMA-batched and dependency-derived local-read schedules on `M=32768`. Their short-M timings reject those exact short keys but are not evidence about long-M throughput. If batching cannot close the approximately 6% large-M deficit, the next mechanism is a typed HIP-shaped Q3 LDS and compute dataflow derived from source-level ownership: 64 output columns by 128 activation rows, rolled 16-WMMA groups, HIP-equivalent payload/scale reuse, and explicit semantic scheduling without copying a physical instruction stream.

Preliminary screens may use 25 rotating repeats after enough warmup to sustain the device, but promotion requires at least two independent long runs and a final 100-warmup, 101-repeat rotating confirmation. Every coherent retained or failed experiment updates this record immediately. Only exact keys with stable multiply medians below HIP may enter the inventory and public selector; all others remain HIP fallbacks.

The final recursive-review rule remains mandatory. Completion requires a fresh review of this plan, the retained implementation, HIP source and disassembly, generated artifacts, rejected evidence, target ISA, and related format mechanisms. Actionable in-contract findings must be implemented and qualified before review repeats; completion is valid only when a fresh pass finds none.

### Large-M Batched-WMMA Requalification

The first continuation experiment requalified the previously exact `Q3BatchedWmmaTiledLds` stream on attention gate `(32768,4096,2048)`. A diagnostic shape adaptation was permitted because source comparison proved that changing `M` modifies only the exported exact-key symbol and two activation-plane stride constants; applying the same transformation to the retained `M=2048` parent reproduced the canonical `M=32768` parent source byte-for-byte. No production source or solution identity was changed.

The canonical 144-VGPR parent measured `30.168192 ms` versus HIP `27.670347 ms`, or `1.09027x` multiply, under 100 warmups and 25 rotating repeats. The 200-VGPR batched candidate remained bit-exact to HIP and public output, finite, repeatable, and sensitive to input, packed-weight, and workspace mutations, but measured `30.534044 ms` versus HIP `28.088524 ms`, or `1.08706x`; complete was `1.08695x`. Its absolute candidate latency did not improve the parent and it remained approximately 8.7% slower than HIP. Batched WMMA ownership is rejected for this large-M key as well as the prior short key.

Dependency-derived local-read waits materially changed the large-M result. A direct 100-warmup, 101-repeat comparison of HIP, the 144-VGPR parent, plain 200-VGPR batching, and 200-VGPR batching with partial waits measured `28.027727 ms`, `29.309143 ms`, `29.283928 ms`, and `28.794504 ms`. Plain batching was neutral, while partial waits improved the parent by `1.756%` and reduced the candidate/HIP ratio to `1.02736x`. The mechanism remains unselected because it did not beat HIP, but it is retained as a large-M research ingredient.

The next 101-repeat comparison tested the partial-wait parent, its rolled form, and paired weight/activation-scale reads. HIP, canonical parent, partial-wait, rolled, and paired medians were `28.050556 ms`, `29.338560 ms`, `29.087126 ms`, `29.314873 ms`, and `28.720013 ms`. All outputs were bit-exact to HIP. Rolling remained ineffective and is rejected. Contrary to its short-M result, paired scale reads improved the canonical parent by `2.108%` and reached `1.02387x` HIP. The paired mechanism becomes the large-M research parent, but no exact selection changes until repeated runs cross HIP.

The paired source used only through `v201` but declared 208 VGPRs. A pressure experiment replaced the persistent lane-varying activation address with an SGPR activation-plane base plus the existing row offset, reused a dead temporary for paired scale-read addresses, and reduced the artifact to 200 VGPRs with unchanged LDS, ABI, arithmetic, and output. It remained exact, mutation-sensitive, and spill-free, but measured `30.124828 ms` versus HIP `27.807213 ms`, or `1.08335x`, under 100 warmups and 25 repeats. Address recomputation outweighed any allocation benefit; the 200-VGPR form is rejected and the 208-VGPR paired stream remains the research parent.

An activation-scale overlap experiment replaced each even group's full `lgkmcnt(0)` drain with dependency waits `3,2,1,0` before correcting fragment pairs `(0,1)`, `(2,3)`, `(4,5)`, and `(6,7)`. The 208-VGPR artifact remained exact and spill-free, but measured `30.411064 ms` versus HIP `28.180796 ms`, or `1.07914x`. The preceding eight WMMAs already covered the four paired scale reads; four explicit waits added scalar issue cost without exposing useful overlap. The schedule is rejected.

### HIP-Shaped Full-Weight Continuation

Recovered HIP disassembly showed a dependency-specific LDS order: activation row 0, weight payload, activation row 1, scale reads, then activation rows 2 through 7. Transplanting that order into the paired compact layout remained exact and improved an interleaved 101-repeat parent from `28.965729 ms` to `28.782827 ms`, while HIP measured `27.545406 ms`. The mechanism is retained, but its `1.04492x` HIP ratio is not selectable and demonstrates that HIP's order alone is insufficient without its LDS ownership.

The next prototype decoded the full 256-value Q3 block once and reused it across both Q8_1 activation halves. A compact 320-byte weight-row stride remained neutral at `29.497225 ms`, while HIP's 336-byte padded stride reached `28.810064 ms` versus HIP `27.834356 ms`, or `1.03505x`; the paired parent was `29.476768 ms`. Both were bit-exact and spill-free. The 16-byte row pad is therefore required for the full-tile LDS bank pattern. Composing the 336-byte tile with HIP-style read ordering improved a separate 101-repeat run to `28.933870 ms` versus HIP `27.962547 ms`, or `1.03474x`, and was retained as the new diagnostic parent.

Full-tile ownership exposed two redundant half-1 global reads: the high-bit mask and block metadata are shared across both Q3 halves. Retaining those registers and prefetching only the second low payload in free pre-WMMA registers preserved 208 VGPRs and 39,936 LDS bytes. The exact artifact improved its composed parent from `29.019480 ms` to `28.755852 ms` in an interleaved run, reaching `1.03273x` HIP. Replacing the three-instruction high-bit merge with a shifted source plus `v_and_or_b32` removed 32 VALU instructions per K block; a later interleaved run improved `29.138067 ms` to `28.584364 ms`, or `1.03812x` HIP under that DPM state. Both mechanisms are retained for typed implementation.

A dependency-derived VMEM overlap then used `vmcnt(9)` to begin both weight decodes after the four older weight reads while nine activation reads remained outstanding, followed by `vmcnt(0)` immediately before activation LDS stores. The first diagnostic incorrectly reused `v64:v70` for decode while pending activation values occupied the same registers and correctly failed finiteness and agreement gates. Remapping decode temporaries to free pre-WMMA destinations `v126:v132` restored bit-exactness without changing resources. The corrected overlap measured `28.785507 ms` versus its `28.856873 ms` parent and HIP `27.517069 ms`; the small `0.247%` gain is retained but remains unselected.

Duplicating the activation LDS plane raised static LDS from 39,936 to 58,368 bytes and removed the two middle synchronization points. Although exact and spill-free, it regressed from `29.095230 ms` to `30.498276 ms` in an interleaved run, or `1.10443x` HIP, showing that the larger LDS allocation changes residency or power behavior. The dual-activation mechanism is rejected. HIP-style phased correction, with all 64 conversions followed by all scale multiplies and then all accumulator updates, also regressed from `28.590860 ms` to `29.100750 ms`; fragment-local correction remains preferred.

Finally, adjacent fragments were paired for accumulator VOPD instructions so X and Y used distinct activation-scale registers. A lane-`xor 1` mapping preserved opposite destination parity and distinct source banks, eliminated all eight activation-scale copy moves per group, and removed 128 instructions per K block. The 208-VGPR, 39,936-byte artifact remained bit-exact, finite, mutation-sensitive, and spill-free. Its short validation reached `1.00149x` HIP; the authoritative interleaved screen improved the overlap parent from `28.982132 ms` to `28.626179 ms` while HIP measured `27.601080 ms`, or `1.03714x`. The mechanism is the current diagnostic parent, but DPM-sensitive runs have ranged down to approximately `1.028x`; no exact dispatch changes until independent confirmations beat HIP.

### Large-M Schedule Grid and Shape Checks

A focused fragment-pair and lane-bank grid screened `cross4/xor1`, `gap3/xor3`, and related mappings on attention-gate `(32768,4096,2048)`. `cross4/xor1` briefly reached `27.948685 ms` versus HIP `27.642548 ms`, but its 101-repeat confirmations were unstable. `gap3/xor3` ranged from `1.03273x` to `1.04648x`; no bank mapping produced a repeatable win. Four-group static unrolling reduced the generated source from `153,113` to `91,890` bytes and briefly reached `0.99863x`, but its authoritative 101-repeat result was `28.998917 ms` versus HIP `27.620768 ms` (`1.04990x`). Combining unroll-four with `gap3/xor3` did not improve that sustained result.

The same three schedules were independently adapted to SSM-output `(32768,2048,4096)`. All candidates were exact and resource-clean; HIP measured `27.740662 ms`, while adjacent FMAC pairing, `gap3/xor3`, and unroll-four measured `29.024971 ms` (`1.04630x`), `28.901819 ms` (`1.04186x`), and `28.762163 ms` (`1.03682x`). The `M=8192` SSM key was also exact, but measured HIP `7.091802 ms` versus `8.272422 ms`, `8.269580 ms`, and `8.282276 ms` for the same candidates, or approximately `1.166x` in every case. These exact keys remain on HIP.

A prior 200-VGPR scalar-address rejection was revisited because it had not been interleaved against its optimized parent. Replacing the persistent activation-plane pointer with the scalar plane base plus a row offset was exact and reduced the optimized 208-VGPR parent from `29.142824 ms` to `28.872631 ms` in a 101-repeat gate comparison; the scalar form measured `1.04467x` HIP. Composing it with `gap3/xor3` reached `28.864442 ms` versus HIP `27.627840 ms` (`1.04476x`). The mechanism is retained as a diagnostic ingredient, while the earlier isolated paired-parent result (`30.124828 ms` versus `27.807213 ms`) remains a rejection of that older composition rather than of the optimized form.

The same parent was then checked with unroll-two and a lower-register unroll-four lifetime remap. Unroll-two was exact, 208 VGPRs, and 60,794 source bytes; its 101-repeat gate result was `28.904478 ms` versus HIP `27.564207 ms` (`1.04862x`). Unroll-four with the remapped loop-address lifetimes was exact at 208 VGPRs and then 200 VGPRs after scalar addressing. The 200-VGPR form reached `28.737890 ms` versus HIP `27.576628 ms` (`1.04211x`); adding `gap3/xor3` reached `28.766096 ms` versus HIP `27.598431 ms` (`1.04231x`). The smaller body and lower register declaration do not yet cross HIP.

### Barrier Audit and Cache Invalidation

Auditing the VMEM-overlap source found an accidental fifth barrier between full-weight decode and activation LDS stores. The final `s_waitcnt lgkmcnt(0)` before the existing barrier already orders both disjoint LDS write sets; removing the earlier `s_waitcnt lgkmcnt(0)` plus `s_barrier` restored the intended four-barrier ownership sequence. The corrected 200-VGPR, 39,936-byte artifact remained code-object-v5, wave32, 40-byte ABI, exact, finite, and sensitive to input, packed-weight, and workspace mutations. A short validation reached `0.99510x` HIP.

Sustained results were governor-sensitive but consistently improved over the five-barrier parent. In one 100-warmup/101-repeat rotation, five barriers, four barriers, and four barriers with cache invalidation measured `29.195617 ms`, `28.806353 ms`, and `28.522600 ms` versus HIP `27.646192 ms`, or `1.05604x`, `1.04196x`, and `1.03170x`. A reversed candidate-order rotation measured HIP `27.566860 ms`, four-barrier cache invalidation `28.875341 ms` (`1.04747x`), and four-barrier without invalidation `28.185945 ms` (`1.02246x`). `buffer_gl0_inv` is rejected for this Q3 dataflow despite its benefit in a separate Q4 experiment; the four-barrier correction is retained for further qualification.

Exact-shape adaptations of the corrected four-barrier source did not generalize. Attention-gate `(8192,4096,2048)` remained exact but measured `8.262635 ms` versus HIP `7.053321 ms` (`1.17145x`), and attention-Q `(8192,8192,2048)` measured `14.027324 ms` versus HIP `13.312505 ms` (`1.05370x`). Neighboring keys therefore remain explicit HIP fallbacks.

Two standalone 100-warmup/101-repeat confirmations of the corrected four-barrier gate artifact measured candidate/HIP ratios `1.00062x` (`25.344852 ms` versus `25.329088 ms`) and `1.00236x` (`25.492165 ms` versus `25.432096 ms`). Paired-sample medians were `1.00346x` and `1.00463x`, and both launch orders favored HIP. A bounded corrected-schedule composition grid confirmed that scalar activation addressing and `gap3/xor3` were independently beneficial, but the best combination still measured `28.129391 ms` versus HIP `27.581703 ms` (`1.01986x`) in a five-way rotation. The four-barrier artifact therefore remains unselected.

A final address audit replaced each `v_lshrrev_b32` plus `v_and_b32` lane-half extraction with one `v_bfe_u32`. Seven static substitutions remove 40 dynamically executed VALU instructions across initial prefetch, both half decodes, weight-row setup, loop-carried prefetch, and epilogue setup while preserving 200 VGPRs, 16 SGPRs, 39,936 LDS bytes, and zero spills. The exact, mutation-sensitive artifact improved its parent from `28.883024 ms` to `28.336884 ms` in an interleaved run, or `1.02648x` HIP. Standalone confirmations split: one reached `25.607712 ms` versus HIP `25.695215 ms` (`0.99659x`), while the independent repeat measured `25.641495 ms` versus HIP `25.550003 ms` (`1.00358x`). The simpler half-BFE form is retained as the diagnostic parent but fails the repeated-win promotion gate.

Hoisting the four invariant weight-scale LDS row bases out of the K loop removed 15 redundant address-setup executions without increasing registers. Two independent 100-warmup/101-repeat diagnostic confirmations both favored the resulting composition: `25.551126 ms` versus HIP `25.684200 ms` (`0.99482x`) and `25.507643 ms` versus HIP `25.685352 ms` (`0.99308x`). Both launch-order strata and paired-sample medians favored the candidate (`0.99576x` and `0.99460x`). Strict correctness was bit-exact to HIP and public output, finite, producer-repeatable, mutation-sensitive, and within the independent-reference bound at normalized RMSE `0.006066316`.

### Typed Full-Weight Lowering

The qualified composition is now represented structurally rather than as a text-adapted artifact. A distinct `Q3FullWeightTiledLds` mechanism owns the fixed Q3 data contract, scalar activation-plane addressing, 128 activation LDS rows, 64 decoded-weight rows, the 336-byte padded full-weight row, shared metadata/high-mask prefetch, VMEM/decode overlap, four barriers, dependency-derived local-read waits, `gap3/xor3` fragment correction, single-instruction lane-half extraction, and K-loop-invariant weight-scale row bases. Its physical plan derives `18,432 + 64 * 336 = 39,936` LDS bytes and explicit register roles ending at 200 VGPRs and 16 SGPRs. The typed lowering is validated against each exact dense inventory key; no neighboring shape inherits applicability.

Two independent typed builds for every one of the 12 keys generated byte-identical sources, code objects, and inspection reports. Every artifact is code-object v5 for gfx1151, wave32, with the 40-byte ABI, 200 VGPRs, 16 SGPRs, 39,936 static LDS bytes, 128 static WMMAs, four barriers, zero private storage, and zero VGPR/SGPR spills.

The rebuilt typed artifacts passed the full correctness gates for all 12 keys: zero differing BF16 elements versus HIP multiply and public complete output, finite output, deterministic producer workspace, nonzero input/packed-weight/workspace mutation sensitivity, and independent-reference normalized RMSE from `0.0060610` to `0.0060685`.

#### TFLOPS-Equivalent Confirmation

The authoritative comparisons used 100 warmup batches and 101 measured repeats with rotating HIP/GGTensile launch order. Each timing is a per-launch median; batch sizes were increased for short shapes so the GPU remained in a sustained operating regime. Effective dense throughput is reported as:

```text
TFLOPS-equivalent = (2 * M * N * K) / (median_ms * 1e9)
speedup = HIP median_ms / GGTensile median_ms
```

The TFLOPS values are an effective dense-operation rate for comparison. Q3_K uses integer WMMA accumulation and FP32 scale correction, so they are not a claim that the kernel executes native FP32 fused multiply-add instructions.

Run A:

| Family | M | N | K | HIP ms | HIP TFLOPS-eq | GGTensile ms | GGTensile TFLOPS-eq | Speedup vs HIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attention K | 2048 | 512 | 2048 | 0.212513 | 20.210 | 0.196734 | 21.831 | 1.08021x |
| attention K | 8192 | 512 | 2048 | 0.761084 | 22.573 | 0.708009 | 24.265 | 1.07496x |
| attention K | 32768 | 512 | 2048 | 3.002010 | 22.891 | 2.781559 | 24.705 | 1.07925x |
| attention Q | 2048 | 8192 | 2048 | 2.988109 | 22.998 | 2.792357 | 24.610 | 1.07010x |
| attention Q | 8192 | 8192 | 2048 | 11.860503 | 23.176 | 11.067318 | 24.837 | 1.07167x |
| attention Q | 32768 | 8192 | 2048 | 47.580341 | 23.109 | 44.279305 | 24.831 | 1.07455x |
| attention gate | 2048 | 4096 | 2048 | 1.518511 | 22.627 | 1.403740 | 24.477 | 1.08176x |
| attention gate | 8192 | 4096 | 2048 | 5.954988 | 23.080 | 5.561356 | 24.713 | 1.07078x |
| attention gate | 32768 | 4096 | 2048 | 23.747124 | 23.150 | 22.147079 | 24.823 | 1.07225x |
| SSM output | 2048 | 2048 | 4096 | 1.529714 | 22.462 | 1.405908 | 24.440 | 1.08806x |
| SSM output | 8192 | 2048 | 4096 | 6.001223 | 22.902 | 5.565976 | 24.693 | 1.07820x |
| SSM output | 32768 | 2048 | 4096 | 23.487139 | 23.407 | 21.830744 | 25.183 | 1.07587x |

Run B used the independent deterministic rebuild and the reversed launch-order rotation:

| Family | M | N | K | HIP ms | HIP TFLOPS-eq | GGTensile ms | GGTensile TFLOPS-eq | Speedup vs HIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attention K | 2048 | 512 | 2048 | 0.201069 | 21.361 | 0.186414 | 23.040 | 1.07861x |
| attention K | 8192 | 512 | 2048 | 0.759417 | 22.622 | 0.707294 | 24.290 | 1.07369x |
| attention K | 32768 | 512 | 2048 | 2.984819 | 23.023 | 2.783813 | 24.685 | 1.07221x |
| attention Q | 2048 | 8192 | 2048 | 3.004088 | 22.875 | 2.801410 | 24.530 | 1.07235x |
| attention Q | 8192 | 8192 | 2048 | 11.687331 | 23.519 | 10.922523 | 25.166 | 1.07002x |
| attention Q | 32768 | 8192 | 2048 | 47.707127 | 23.047 | 44.394245 | 24.767 | 1.07462x |
| attention gate | 2048 | 4096 | 2048 | 1.499275 | 22.918 | 1.383588 | 24.834 | 1.08361x |
| attention gate | 8192 | 4096 | 2048 | 5.992367 | 22.936 | 5.584771 | 24.610 | 1.07298x |
| attention gate | 32768 | 4096 | 2048 | 23.847000 | 23.053 | 22.233656 | 24.726 | 1.07256x |
| SSM output | 2048 | 2048 | 4096 | 1.511326 | 22.735 | 1.387041 | 24.772 | 1.08960x |
| SSM output | 8192 | 2048 | 4096 | 5.913688 | 23.241 | 5.469110 | 25.130 | 1.08129x |
| SSM output | 32768 | 2048 | 4096 | 23.907846 | 22.995 | 22.172792 | 24.794 | 1.07825x |

Across the two runs, all 12 speedup ratios exceed `1.0x`; the per-run call-count-weighted GGTensile/HIP time ratios are `0.93067` and `0.92978`, equivalent to weighted speedups of approximately `1.0745x` and `1.0755x`. This is research-catalog evidence only. The Q3 inventory records the exact qualified candidates, while `csrc/mmq_bundle.cpp`, `csrc/generated/mmq_bundle_table.cuh`, and public runtime dispatch remain unchanged at the 179-kernel HIP bundle.

### Final Full-Weight Review

The recursive review then tested the two remaining low-cost in-contract ideas found in the generated body. Pairing the 64 startup sum clears into 32 `v_dual_mov_b32` issues preserved 200 VGPRs, 16 SGPRs, 39,936 bytes of LDS, four static barriers, exact HIP/public output, deterministic producer bytes, and mutation sensitivity. Final-block barrier and activation-plane-advance elision was also exact with the same resource envelope. On the large attention-gate key, paired sums, tail elision, and the composition measured `1.00022x`, `1.00180x`, and `1.00066x` of the typed parent in one 100-warmup/101-repeat rotation. On narrow attention-K with sustained 64-launch timing batches, parent-first and candidate-first rotations measured respectively: paired sums `0.99853x` and `1.00073x`, tail elision `0.99877x` and `1.00007x`, and the composition `0.99871x` and `0.99914x` of the parent. None has a repeatable promotion margin or transfers to the large key; all three are rejected.

The final review classified the remaining alternatives as already rejected by correctness, resource lifetime, or repeated timing: compact 320-byte rows, dual activation planes, cache invalidation, phased correction, paired scale reads, rolled or partial-WMMA schedules, unstable fragment/bank mappings, smaller activation geometry, and untyped register reuse. Prepared weights, dense shadows, external decode storage, split-K, persistent/grouped traversal, producer fusion, hidden caches, and public dispatch changes remain outside the contract. No actionable in-contract optimization remains for the 12 exact research keys under the fixed ABI, arithmetic order, wave ownership, zero-spill envelope, and public-unwired policy.

## Cross-Campaign Review Boundary

The Q3 full-weight campaign remains closed for its 12 exact keys. The later cross-format review found no new Q3-specific reopening: invariant activation-base addressing and lifetime-derived weight-scale bases are already represented in `Q3FullWeightTiledLds`, and the full-weight ownership, 336-byte padded row, and four-barrier schedule cannot transfer to another quant type without matching packed layout, lane ownership, synchronization, register lifetimes, correction order, and resource residency.

The recursive final-review rule is global rather than a per-format, per-direction, or per-shape waiver. A fresh review must reread the related forward and backward records, HIP and GGTensile evidence, target ISA, generated artifacts, and exact inventories. A finding discovered in another problem type, direction, quant type, or shape may reopen this record when it supplies a changed premise; Q3 evidence may transfer outward only after the receiving ownership and contract are independently implemented and qualified. This boundary does not authorize prepared weights, external workspaces, producer fusion, arithmetic-contract changes, public dispatch, or other contract changes.
