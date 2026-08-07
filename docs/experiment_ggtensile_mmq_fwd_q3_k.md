# GGTensile MMQ Forward Q3_K Experiment

## Purpose

This record covers the isolated GGTensile MMQ forward Q3_K control and its later 12-key dense expansion on gfx1151. The control validates the Q3_K packed decoder, Q8_1 `F32_D4` workspace contract, wave32 WMMA mapping, and row-major BF16 output ownership. The expansion promotes one independently qualified exact key and leaves every other key on HIP.

HIP remains the correctness and timing control and the exact fallback for every unselected shape.

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

The first geometry alternative halves the activation ownership from 128 to 64 rows while retaining four N-owning waves and the 64-column output tile. Canonical Q3 ownership is now explicitly wave-N: `(MIWaveGroupM,MIWaveGroupN)=(1,4)` with four or eight M fragments for MT64 or MT128. The previous `(4,1)` projection was inconsistent with the emitted mapping but happened to produce the same MT128 static count; strict MT64 inspection exposed and corrected it. The new physical plan derives 104 VGPRs, 16 SGPRs, and 19,456 LDS bytes, compared with 144 VGPRs and 28,672 bytes for MT128. It emits 64 WMMAs, 59 VMEM operations, 274 LDS operations, 21 waits, and four clauses with zero private storage and zero spills. Only lanes 0 through 15 of each wave stage the 64 unique activation rows; full execution is restored before weight decode and every barrier. The default MT128 source remains byte-identical at SHA-256 `64a2dd8886662c810b9b99276cf65722657ac2d36080f37e2163ae106cf9a4c6`.

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

The retained source and code object were byte-identical across the historical and current runs. The authoritative assembly SHA-256 remained `64a2dd8886662c810b9b99276cf65722657ac2d36080f37e2163ae106cf9a4c6`, and the old and regenerated HSACOs were byte-identical. Nevertheless, the short `(2048,4096,2048)` control changed from `0.93799x` HIP multiply historically to approximately `1.72x` under the later `auto` DPM regime. Sysfs reported a 600 MHz idle state and a 2.9 GHz ceiling; direct performance-level control required unavailable elevated privileges.

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

### Exact Promotion

Attention-Q `(32768,8192,2048)` passed two 25-repeat runs at `0.99221x` and `0.99568x` HIP multiply, a 101-repeat confirmation at `0.99115x`, and a rebuilt-public 101-repeat confirmation at `0.99185x`. The final public-environment medians were `52.881584 ms` versus HIP `53.315998 ms` for multiply and `53.250736 ms` versus `53.505711 ms` for complete, or `0.99185x` and `0.99523x`. Candidate and public outputs were bit-exact to HIP. Candidate and public independent-reference normalized RMSE was `0.006065104`, with finite output and maximum absolute error `0.046875`.

The selected source is generated from `mmq_fwd_q3_k_inventory.json` and `mmq_fwd_q3_k_solutions.json`. The bundle builder enforces exact source SHA-256 `c9500b3364a5d309886548a4858987ba272928bd21b20a430707e9ccd2fe570c` for this shape. The bundled HSACO is byte-identical to the independently qualified artifact at SHA-256 `3c1122dd270b81ba6df40bddbe8016f25509457f4251d26ac34303575bb59604`. Inspection reports code-object v5, gfx1151, wave32, the 40-byte ABI, 144 VGPRs, 16 SGPRs, 28,672 static LDS bytes, zero private storage, and zero spills.

Public dispatch now selects this assembly only for exact `(quant_type, M, N, K) = (Q3_K,32768,8192,2048)`. It launches with zero dynamic shared memory because LDS is statically declared by the assembly code object. A traced public call opened the selected package HSACO and remained bit-exact through all mutation gates. Every other Q3 shape preserves the existing HIP selector and dynamic-LDS launch. The source-built bundle now contains 180 kernels.

All rejected experimental mechanism implementations were removed after recording their correctness, resource, and timing evidence. Production retains only the qualified `Q3HipTiledLds` lowering and the one exact selection.
