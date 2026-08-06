# GGTensile MMQ Forward Q3_K Experiment

## Purpose

This record covers the first isolated GGTensile MMQ forward Q3_K control on gfx1151. The control validates the Q3_K packed decoder, Q8_1 `F32_D4` workspace contract, wave32 WMMA mapping, and row-major BF16 output ownership on one exact ordinary shape. It is research evidence only. It does not add a Q3_K inventory, selected catalog, runtime dispatch entry, or public bundle kernel.

HIP remains the correctness and timing control. The existing public HIP path remains authoritative for production selection.

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

The exact-shape promotion gate is met for both multiply and complete medians, but the candidate remains research-only because Q3 catalog selection, broader shape validation, and public integration are separate decisions.

## Recursive Final Review

The final recursive review reread this record, the generic GGTensile plan, the Q4_K/Q5_K/Q6_K/Q8_0 forward records, the relevant Q3 HIP sources and disassembly, the generated source and inspection reports, rejected artifacts, resource formulas, and warmed timing evidence. The classification is:
- Retained and measured: typed Q3 signed reconstruction; persistent eight-VGPR WMMA zero source; activation LDS-base hoist; activation-scale deduplication; full activation VMEM batching; shared activation/weight VMEM issue; legal VOPD scale correction; deterministic 144-VGPR role order; and loop-carried half-0 weight prefetch with a final-block bounds check.
- Rejected by correctness or resource lifetime: unsafe temporary aliasing across the VMEM wait; the first metadata-reuse attempt, where metadata overlapped `c` and was overwritten; and any untyped reuse of raw weight operands, stage addresses, scale shifts, activation staging, or persistent zero operands.
- Rejected by warmed timing: paired weight-scale LDS reads, the two-fragment cross-M FMAC batch, low-pressure paired scale reads, the 152-VGPR prefetch placement, and the correctly typed 152-VGPR metadata-reuse variant. The earlier VOPD, batched activation, and persistent-zero intermediate variants remain recorded with their parent comparisons.
- Contract-incompatible or deferred: prepared or dense weights, external decode workspaces, split-K, persistent/grouped traversal, producer fusion, hidden caches, public dispatch/bundle changes, copied HIP/LLVM schedules, and broader Q3 shape/catalog expansion without exact-key evidence. Additional LDS buffering would require a new ownership and residency premise rather than a schedule edit.
- No actionable in-contract mechanism remains under the fixed exact shape, ABI, LDS layout, wave ownership, and zero-spill resource contract. The remaining measured envelope is the 144-VGPR body at 128 WMMAs, 370 LDS operations, 91 VMEM operations, 21 waits, and 1,028 VOPD instructions; every identified traffic, lifetime, scale-correction, and register-class opportunity either is retained or has an exact rejection. The final control beats HIP on both required paths, so there is no residual performance deficit requiring an unmeasured mechanism.

The campaign is complete for this isolated Q3 exact shape. Production remains on HIP until a separate promotion phase validates additional exact keys and makes an explicit catalog/dispatch decision; no public boundary changed in this campaign.
