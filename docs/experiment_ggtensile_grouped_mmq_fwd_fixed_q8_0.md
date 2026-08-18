# GGTensile Fixed-Group MMQ Forward Q8_0 Experiment

## Purpose

Implement and optimize a GGTensile assembly kernel for the DeepSeek fixed-group Q8_0 output-A projection on gfx1151. The operation has eight independent fixed weight groups, a shared Q8_1 F32_D4 activation workspace, and BF16 output. This is a separate ABI and ownership problem from routed grouped MMQ and from dense MMQ: one workgroup owns an output-feature tile, a token tile, and one fixed group.

The campaign record is updated after each coherent implementation, correctness, resource, timing, rejection, or review result. Durable scripts, reports, and artifacts live below `~/tmp/torch-ggml-ops/`.

## Contract

Target only:
- gfx1151, code-object version 5, wave32, WMMA V1, and the six-argument fixed grouped ABI.
- Input BF16 with shape `[tokens, 8, 4096]`; upstream Q8_1 F32_D4 workspace with one quantized row per `(token, group)`.
- Eight packed Q8_0 weight groups, each with logical shape `[out_features, 4096]` and packed row stride `4352` bytes.
- BF16 output `[tokens, 8, out_features]` in the existing fixed output layout.
- Required token counts `2048`, `8192`, and `32768`, and the production output-A shape `out_features=1024, K=4096`.
- Exact arithmetic and BF16 rounding compatible with the authoritative HIP fixed-group kernel and the existing public operator.
- Deterministic source and artifact generation, strict ABI/resource inspection, independent packed-weight correctness, mutation checks, and warmed complete and prequantized timing.

The fixed kernel ABI is:

```text
packed_weight, activations, output, tokens, out_features, bytes_per_group
```

Its launch is `(ceil(out_features/64), ceil(tokens/64), 8)` with workgroup `(32,4,1)`. The upstream quantizer and complete-call allocation remain outside the assembly writer but inside complete-call timing.

Q8_0 stores one FP16 scale and 32 signed int8 values in 34 bytes. The dot is signed integer WMMA followed by the established FP32 correction:

```text
fp32_output += int32_dot * fp32(q8_0_scale) * fp32(q8_1_activation_scale)
```

No routing metadata, expert-index lookup, CPU descriptor, hidden copy, split-K, prepared weight, dense shadow, or public dispatch rewrite is part of the first candidate. Unsupported shapes must use the existing HIP path.

## Controls And Baseline

The authoritative performance control is `fixed_grouped_q8_0_mmq_bf16_body` in `csrc/mmq_core.cuh`, launched by `launch_fixed_grouped_forward` in `csrc/mmq_bundle.cpp`. It uses the same Q8_1 producer and packed GGUF weights as the GGTensile path. The public `fixed_grouped_mmq` operator is the complete-call control.

Before retaining a candidate, collect same-process HIP and public timings. Keep multiply-only and complete-call measurements separate. Complete-call timing includes activation quantization, workspace allocation, kernel launch setup, and the fixed grouped multiply. Never use `.item()`, host route/workspace metadata, implicit synchronization, or a CPU descriptor in a timed path.

Existing ordinary Q8_0, Q3_K, Q4_K, Q5_K, and Q6_K writer streams are a regression boundary. Changes to shared mechanisms require source and behavioral qualification; unrelated generated identities are not to be pinned in this document.

## Design Plan

- Add a typed fixed-group Q8_0 problem/solution contract with explicit `(tokens, N, K, groups)`, six-argument ABI, grid, packed group stride, and output addressing.
- Reuse signed-int8 arithmetic roles only where their data contract matches; keep fixed-group grid and addressing in a dedicated lowering.
- Add deterministic physical ownership and resource derivation for the fixed candidate. Retained gfx1151 artifacts require wave32, zero private storage, no spills, no scratch, no calls, and no dynamic stack.
- Build the smallest complete candidate, qualify exact output equality against HIP/public and an independent GGUF dequantized reference, then screen ownership and LDS variants.
- Promote only after two independent warmed confirmations on each required token count and no regression for a shape sharing the mechanism. Public dispatch, generated bundle registration, packaging, and fallback changes are deferred until the research artifact has final evidence.

The likely high-value dataflow is HIP-shaped cooperative LDS reuse: activation rows are shared across output columns, while each fixed group selects a disjoint packed-weight base through `z`. A candidate that merely launches one independent projection per workgroup or reloads activations for each output tile is not considered fused fixed-group work.

## Active Optimization Plan

P0 is the retained parent, not the performance endpoint. The installed fixed-group HIP multiply remains the external floor, while each new candidate must also beat P0 on the exact same prequantized workspace. Search starts with changed ownership or representation premises that have evidence for large gains and moves to smaller scheduling margins only after those are exhausted.

- First evaluate the ordinary Q8 campaign's compact `DepthU=32` small-M layout at the fixed `64x64` ownership. It reduces fixed LDS from 28,672 to 18,432 bytes and replaces scalar weight-scale row reads with the validated paired-base mechanism; this is the clearest occupancy and LDS-traffic opportunity.
- If compact staging wins, treat it as the new parent and optimize its fixed-group-specific activation, weight-scale, and output-row address work. Prefer removing repeated work or shortening dependency chains over reducing static instruction count without a timing premise.
- Inspect linked resource occupancy, static VMEM/LDS/VALU/wait structure, and normalized disassembly before opening wider changes. Evaluate staging order, scale-read ownership, legal VOPD, wait thresholds, loop-tail address advancement, and epilogue traversal only when the fixed ownership changes the premise of an ordinary rejected experiment.
- Use three warmups and nine order-rotated repeats for screens. A candidate must be exact and resource-clean before timing, must beat the retained parent at a required token count to remain active, and must not materially regress another required token count. Retention requires two independent seven-warmup, 25-repeat confirmations at `2048`, `8192`, and `32768` tokens for both prequantized and complete-call surfaces.
- After large mechanisms are exhausted, investigate smaller margins only with a concrete dependency, occupancy, or instruction-throughput hypothesis. Remove failed implementation experiments after recording their source identity, correctness/resources, timings, and rejection reason.
- Public dispatch, generated bundle registration, packaging, fallback changes, prepared weights, hidden caches, split-K, persistent workgroups, and producer fusion remain outside this optimization campaign.

The recursive final review remains mandatory. After the active search appears exhausted, reread the complete fixed and ordinary Q8 records, current and HIP sources, normalized disassembly, retained and rejected candidates, benchmark reports, and relevant gfx1151 ISA/compiler evidence. Classify every remaining mechanism as retained, rejected, incompatible/deferred with an explicit prerequisite, or actionable. Implement and measure every actionable finding, then repeat the complete review from the changed premise; the final review cannot pass in the same iteration that first discovers an actionable mechanism. Any correctness, ABI, resource, executable, benchmark, test, or documentation finding restarts qualification.

## Validation Gates

Every candidate must pass, in this order:
- strict fixed ABI and exact-shape validation;
- deterministic source generation and successful gfx1151 assembly/link;
- code-object v5, wave32, zero-spill resource inspection;
- bitwise equality with HIP and the public operator on all required token sizes;
- finite output, independent Q8_0 reference, input/weight/workspace mutation, and repeated-producer checks;
- complete-call and prequantized warmed timing against HIP and the retained parent;
- focused tests, full tests, lint/type/format/compile checks, and diff checks.

Any correctness, ABI, compiler, runtime, executable, or documentation finding that affects the contract restarts qualification. Failed candidates remain recorded with their artifact paths and are removed from active lowering unless a later premise explicitly reopens them.

## Completion Record

- Initial fixed-group contract boundary and ABI recorded.
- Reopened P0 for autonomous performance optimization. The search prioritizes compact depth-32 ownership and other evidence-backed large mechanisms before schedule-level margins, retains exact same-workspace parent/HIP comparisons, and preserves the recursive final-review restart rule.
- Added the typed fixed-group model, validation boundary, derived state, physical-plan wrapper, six-argument assembly writer, direct prequantized runtime, and artifact inspection in the fixed-group modules under `tools/ggtensile/`.
- Candidate P0 uses the existing signed-int8 Q8 small-M LDS mechanism with a dedicated fixed lowering. It emits code-object v5 metadata for a 40-byte six-argument kernarg segment, wave32, `(32,4,1)` workgroups, 144 VGPRs, 16 SGPRs, 28,672 fixed LDS bytes, 32 static signed-int8 WMMAs, two barriers, zero private storage, and zero register spills.
- P0 first assembled and linked successfully, but the first device attempt exposed three address-contract bugs. The output store initially used `s8` (the activation pointer low word) instead of `s13` (`out_features`); the lowering then loaded none of the scalar values at `s12:s15`; and the first flattened store used dense-row spacing rather than eight-group spacing. These failures are retained as review evidence, and the invalid artifact was not retained as a candidate.
- After loading the scalar kernargs and applying `((tile_token + dense_row) * 8 + group)` output addressing, P0 at `tokens=2048`, `N=1024`, and `K=4096` is bitwise identical to the public `fixed_grouped_mmq` control. The independent GGUF Q8_0 dequantized reference measured normalized RMSE `0.00605183` and maximum BF16 absolute error `0.0625`. Required-token qualification and timing remain open.
- Added focused model, rejection, source-dataflow, deterministic-build, and resource-inspection tests. The fixed-group test module passes all six tests; the artifact inspector independently confirms the 40-byte ABI, 32 WMMAs, two barriers, fixed LDS size, and prohibited-resource zeros.
- P0 passed bitwise public-control qualification at all required token counts: `2048` (`exact token count 2048`), `8192` (`8192`), and `32768` (`32768`), with `N=1024` and `K=4096`. Candidate and public outputs had equal shape, equal BF16 values, and zero maximum difference at each size.
- The `2048` qualification also passed input, packed-weight, and activation workspace mutation checks, and two independent Q8_1 F32_D4 producer launches produced identical workspaces. The authoritative packed source was `blk.0.attn_output_a.weight`; no synthetic quantized bytes were used.
- The first order-rotated timing screen used three warmups and nine repeats per size. P0 prequantized multiply medians were `4.046`, `17.562`, and `71.585 ms` at `2048`, `8192`, and `32768` tokens; the installed HIP control was `10.069`, `40.002`, and `159.427 ms`, giving candidate/control ratios `0.4019`, `0.4390`, and `0.4490`. Manual complete-call medians, including quantization and workspace allocation, were `4.982`, `21.428`, and `87.034 ms`; the public complete-call control was `11.045`, `43.676`, and `174.564 ms`, giving ratios `0.4511`, `0.4906`, and `0.4986`.
- Screen reports are durable under `~/tmp/torch-ggml-ops/fixed-grouped-q8-candidate/`. Confirmation repeats and the final retain/reject decision remain open.
- The first 25-repeat confirmation used seven warmups and seed `9012`. P0 prequantized medians were `4.145`, `17.694`, and `71.542 ms` versus HIP `10.188`, `40.073`, and `159.694 ms`, with ratios `0.4069`, `0.4415`, and `0.4480`. Complete-call medians were `5.004`, `21.397`, and `86.326 ms` versus public `11.054`, `43.737`, and `174.552 ms`, with ratios `0.4527`, `0.4892`, and `0.4946`.
- A second independent correctness run at seed `9013` remained bitwise exact at all three token counts and repeated the no-mutation and producer-repeat checks. Its second 25-repeat confirmation produced prequantized ratios `0.4045`, `0.4387`, and `0.4501`, and complete-call ratios `0.4514`, `0.4897`, and `0.4870` for `2048/8192/32768` tokens.
- P0 is retained as the qualified research candidate for this campaign. Public dispatch, generated bundle registration, packaging, fallback selection, and prepared-weight integration remain deliberately deferred.
- Repository verification passed: `546` tests, Ruff check and format, `ty check`, compileall, diff checks, and all pre-commit hooks.
- Compact P1 tested the first active changed premise offline: the fixed `64x64`, DepthU32 dataflow used `CompactDepth32WeightRows` with 144-byte weight rows and paired weight-scale reads. The linked artifact retained 144 VGPRs, 16 SGPRs, 32 WMMAs, two barriers, 40-byte ABI, zero private storage and spills, reduced fixed LDS from 28,672 to 18,432 bytes, and reduced static LDS operations from 100 to 84. The `2048` artifact was bitwise exact to public/HIP and passed input, packed-weight, workspace, and repeated-producer checks.
- Compact P1's three-warmup, nine-repeat screen beat P0 at every required token count. Prequantized medians were `3.984/17.107/69.846 ms` versus P0 `4.392/18.091/72.179 ms`, giving P1/P0 ratios `0.9072/0.9456/0.9677` at `2048/8192/32768`; P1/HIP ratios were `0.3919/0.4274/0.4382`. Complete-call P1/P0 ratios were `0.8985/0.9750/0.9746`, and P1/public ratios were `0.4360/0.4821/0.4868`. P1 is favorable and advances to a typed fixed identity and full qualification; reports are under `~/tmp/torch-ggml-ops/fixed-grouped-q8-compact-p1/`.
- Added `LdsAddressHoist` to the strict fixed solution identity and a typed `q8_0_compact_depth32_tiled_lds()` factory while preserving P0's `SmallMTile` factory. Serialization, validation, ordinary-mechanism projection, resource derivation, inspection, and focused tests now distinguish both complete policies. Typed P1 artifacts at all three token counts reproduce the offline 144 VGPR, 16 SGPR, 18,432 LDS, 32-WMMA, two-barrier profile with zero prohibited resources.
- Typed P1 is bitwise exact to public/HIP at `2048`, `8192`, and `32768`; all three sizes pass input, packed-weight, workspace, and repeated-producer checks. The independent authoritative GGUF reference at `2048` remains normalized RMSE `0.00605183` with maximum BF16 absolute error `0.0625`. Typed artifacts and inspection records are under `~/tmp/torch-ggml-ops/fixed-grouped-q8-compact-p1-typed/`.
- Fixed-invariant P2 hoisted the lane/wave-derived weight-scale LDS addresses out of the 32-iteration reduction loop, retained the paired second-row address in otherwise unused rounded VGPR `v139`, and moved the activation-plane stride to dead-after-setup scalar `s15`. The candidate removed six repeated VALU instructions and one repeated vector multiply per reduction iteration without changing the 144 VGPR, 16 SGPR, 18,432 LDS, 32-WMMA, two-barrier occupancy/resource class. It was bitwise exact at `2048` and passed all mutation and producer checks.
- P2's three-warmup, nine-repeat screen beat P1 at every size. Prequantized medians were `3.872/16.892/68.947 ms` versus P1 `4.304/17.659/70.309 ms`, giving P2/P1 ratios `0.8996/0.9565/0.9806`; P2/HIP ratios were `0.3797/0.4226/0.4333`. Complete-call P2/P1 ratios were `0.9078/0.9893/0.9878`, and P2/public ratios were `0.4272/0.4796/0.4824`. The short-size parent samples remain noisy, but the direction is favorable at all sizes and the mechanism advances to typed implementation and full qualification. Reports are under `~/tmp/torch-ggml-ops/fixed-grouped-q8-invariant-p2/`.
- Added `FixedAddressHoist` to the strict fixed solution identity and a typed `q8_0_compact_depth32_tiled_lds_hoisted()` factory. The fixed physical authority owns rounded free `v139` and dead-after-setup `s15`; the default path still emits P0/P1 unchanged, and the shared signed-int8 group helper accepts an optional second paired-scale address while preserving ordinary generated streams.
- Typed P2 artifacts reproduce the screened instruction streams exactly after kernel-symbol normalization. They pass inspection and bitwise public/HIP correctness at all three token counts, all mutation and repeated-producer checks, and the independent `2048` GGUF reference with normalized RMSE `0.00605183` and maximum BF16 absolute error `0.0625`. Typed artifacts are under `~/tmp/torch-ggml-ops/fixed-grouped-q8-invariant-p2-typed/`.
- Weight-address P3 used three remaining rounded VGPR slots to retain the lane-parity packed-weight byte offset and the payload/scale LDS write addresses. This replaces eleven per-iteration address instructions with one vector add; the linked candidate remains at 144 VGPRs, 16 SGPRs, 18,432 LDS bytes, 32 WMMAs, two barriers, and zero prohibited resources. It was bitwise exact at `2048` and passed mutation and repeated-producer checks.
- Balanced pairwise three-warmup, nine-repeat screening retained P3. Prequantized P3/P2 ratios were `0.9825/0.9859/0.9870` at `2048/8192/32768`, with P3/HIP ratios `0.3721/0.4156/0.4261`. Complete-call P3/P2 ratios were `0.9867/0.9847/0.9888`, and P3/public ratios were `0.4311/0.4713/0.4748`. The earlier three-way screen had order imbalance and is superseded for parent comparisons by these pairwise results. Reports are under `~/tmp/torch-ggml-ops/fixed-grouped-q8-weight-address-p3/`.
- Added retained `ReductionLoopAndWeightStage` identity and `q8_0_compact_depth32_tiled_lds_weight_hoisted()` factory. The fixed physical plan owns `v140:v142`, and shared signed-int8 weight staging is split into address generation, global loads, and ready-address LDS writes so ordinary callers preserve the same generated instruction order while fixed P3 supplies precomputed addresses.
- Typed P3 reproduces the screened instruction stream exactly after kernel-symbol normalization. All three artifacts pass strict inspection, bitwise public/HIP correctness, mutation checks, and repeated Q8_1 producer checks; the independent `2048` GGUF reference remains normalized RMSE `0.00605183` with maximum BF16 absolute error `0.0625`. Typed artifacts are under `~/tmp/torch-ggml-ops/fixed-grouped-q8-weight-address-p3-typed/`.
- Overlap P4 removed the second weight-write `vmcnt(0)` because the preceding `vmcnt(6)` already covers all six older packed-weight VMEM operations, leaving the activation-specific `vmcnt(3)/vmcnt(0)` waits intact. The candidate was bitwise exact and reduced static waits from 23 to 22, but balanced pairwise screening was neutral: prequantized P4/P3 ratios were `1.0008/0.9983/1.0002`, and complete-call ratios were `0.9954/1.0011/1.0004` at `2048/8192/32768`. P4 is rejected; P3 remains the parent. Reports are under `~/tmp/torch-ggml-ops/fixed-grouped-q8-overlap-p4/`.
- Persistent-weight P5 used the last rounded VGPR slot for a loop-carried global weight-stage pointer, removing the per-iteration base-plus-lane-offset add. It remained exact and resource-clean, but prequantized P5/P3 ratios were `1.0124` at `2048` and `0.9968` at `8192`; complete-call ratios were `0.9982` and `0.9955`. The required short size regressed while the longer-size movement stayed below one percent, so P5 is rejected and the `32768` screen was intentionally skipped. Reports are under `~/tmp/torch-ggml-ops/fixed-grouped-q8-persistent-weight-p5/`.
- Unroll2 P6 duplicated the exact DepthU32 reduction body and performed one loop-counter update, compare, and branch per two stages while preserving arithmetic order, barriers, pointer movement, and resource metadata. The `2048` artifact was exact, but prequantized P6/P3 was `1.0028` and complete-call P6/P3 was `1.0016`. Every workgroup executes the same K4096 loop at all token counts, so the larger code body was rejected without redundant longer-token screens. Reports are under `~/tmp/torch-ggml-ops/fixed-grouped-q8-unroll2-p6/`.
- Recursive review pass 1 found that P3 lowering defended paired physical addresses with Python `assert`; optimized Python may remove that contract check. The emitter now raises an explicit `ValueError`, focused inspection locks maximum register indices `v142/s15`, and the 238 preserved ordinary forward/backward/grouped writer records remain byte-identical. Generated P3 assembly is unchanged, but the contract finding restarts artifact and device qualification.
- Restarted P3 qualification rebuilt and reinspected all three typed artifacts with unchanged source/resources, then repeated bitwise public/HIP equality, mutation checks, producer determinism, and the independent `2048` reference. Recursive review pass 2 found no remaining contract or generated-stream defect: all hoisted values are initialized before the loop and read-only thereafter, 238 preserved writer records remain byte-identical, and 225 focused ordinary/fixed tests pass. Remaining mechanisms are measured rejections, incompatible with fixed `(32,4,1)` ownership, or deferred integration work.
- The first seven-warmup, 25-repeat confirmation at seed `9012` measured P3 prequantized medians `3.834/16.718/68.407 ms` versus HIP `10.133/40.072/159.510 ms`, giving P3/HIP ratios `0.3784/0.4172/0.4289` at `2048/8192/32768`. Complete-call medians were `4.720/20.593/81.495 ms` versus public `11.064/43.747/174.696 ms`, giving ratios `0.4266/0.4707/0.4665`. Relative to P0's matching confirmation, P3 ratios were `0.9248/0.9449/0.9562` prequantized and `0.9433/0.9624/0.9440` complete.
- The independent seed `9013` confirmation measured P3/HIP prequantized ratios `0.3792/0.4161/0.4266` and P3/public complete-call ratios `0.4290/0.4702/0.4661`. Relative to P0's matching confirmation, P3 ratios were `0.9337/0.9485/0.9475` prequantized and `0.9506/0.9621/0.9589` complete. Confirmation reports are under `~/tmp/torch-ggml-ops/fixed-grouped-q8-weight-address-p3-typed/`.

### Final retained throughput

Effective TFLOPS is nominal dense-equivalent throughput, calculated as `2 * tokens * 8 * N * K / seconds`: two FLOPs per FMA across all eight fixed groups. A/B values are the independent seed `9012/9013` confirmations. The multiply surface compares directly with the installed HIP body; the complete surface compares with the public installed HIP operator and includes activation quantization, workspace allocation, launch setup, and multiplication.

| Tokens | Surface | GGTensile ms A/B | HIP ms A/B | GGTensile effective TFLOPS A/B | HIP effective TFLOPS A/B | HIP/GGTensile speedup A/B |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 2,048 | Prequantized multiply | 3.8338 / 3.8507 | 10.1325 / 10.1555 | 35.85 / 35.69 | 13.56 / 13.53 | 2.6430x / 2.6373x |
| 2,048 | Complete call | 4.7202 / 4.7635 | 11.0643 / 11.1029 | 29.12 / 28.85 | 12.42 / 12.38 | 2.3440x / 2.3308x |
| 8,192 | Prequantized multiply | 16.7181 / 16.6541 | 40.0725 / 40.0250 | 32.88 / 33.01 | 13.72 / 13.74 | 2.3970x / 2.4033x |
| 8,192 | Complete call | 20.5932 / 20.5480 | 43.7469 / 43.7042 | 26.70 / 26.75 | 12.57 / 12.58 | 2.1243x / 2.1269x |
| 32,768 | Prequantized multiply | 68.4072 / 68.0473 | 159.5104 / 159.5055 | 32.15 / 32.32 | 13.79 / 13.79 | 2.3318x / 2.3440x |
| 32,768 | Complete call | 81.4954 / 81.5626 | 174.6957 / 174.9922 | 26.98 / 26.96 | 12.59 / 12.57 | 2.1436x / 2.1455x |

- P3 is retained as the qualified fixed-group Q8_0 research winner. Final verification passed `552` tests with the existing 14 warnings, all 238 protected writer streams byte-identical, Ruff check and format, `ty check`, compileall, diff checks, and all pre-commit hooks. Public dispatch, generated bundle registration, packaging, fallback selection, and prepared-weight integration remain deferred to a separate change.

## Reopened source-level and artifact probes

The P3 retention and the earlier recursive review remain valid for the fixed Q8_0 body, comparator, and geometry that were actually measured. A later source review did not find a new fixed-family zero-bank opportunity: important addresses are already hoisted and the fixed path already uses its dual zero moves. The following bounded probes remain separate, unqualified work items rather than retained results.

### F1 processor-mode metadata A/B

The installed HIP control contains `.amdhsa_workgroup_processor_mode 1`, while the inspected GGTensile fixed artifacts do not. F1 adds or removes only this metadata in independently built artifacts, then checks code-object version, ABI, resources, disassembly, correctness, and deterministic rebuilds before any device timing. No performance benefit is presumed, and the result cannot be transferred to paired or ordinary families without their own artifact and device checks.

### F2 exact epilogue and clause scheduling screens

The fixed lowerer forms flattened `[tokens * groups, N]` addresses, increments each fragment address, applies exact BF16 RNE, and emits the final store clause. F2 may test an interleaving of independent BF16 conversion chains or a clause-boundary change only when the physical plan proves dead scratch lifetimes, unchanged store order, and legal assembler output. Static instruction counts are separate from timing, and any candidate that changes address ownership or the flattened layout is a new representation experiment rather than a local schedule probe.

### F3 delay-ALU and bank-valid VOPD screen

The current grouped lowerer does not directly emit `s_delay_alu`; compiler-produced delay instructions may be inspected in the linked gfx1151 artifact when a concrete dependency schedule changes. VOPD is eligible only for the exact GFX11 X/Y instruction lists and register-bank masks. It is not an automatic replacement for scalar or vector moves, and `v_bfe_u32` is not assumed to be a legal generic VOPD operand. Existing wait and barrier behavior remains the control because the fixed LDS and store hazards have not changed.

F1-F3 are pending only if an exact target-level mechanism can be materialized. Each requires a distinct typed identity and serialized policy, independent assembly and linking, ABI and metadata inspection, resource and disassembly inspection, exactness and mutation checks, deterministic rebuilds, and warmed prequantized and complete-call comparison against the retained P3 body and the installed controls. Public dispatch, generated bundles, packaging, fallback selection, and prepared representations remain outside the experiment.

## Recursive Final Review Continuation

The prior recursive final-review rule is still mandatory. Its retained and rejected classifications are scoped to the old P3 premise and must not be rewritten by these pending probes. Implement and qualify every actionable finding, then repeat the complete source, artifact, resource, correctness, determinism, and timing review from the changed premise. Completion is valid only when a fresh recursive pass finds no actionable in-contract mechanism and the selected fixed key remains exact, deterministic, resource-clean, and faster than its exact controls.

### F1 processor-mode result

F1 rebuilt current canonical P3 and inserted `.amdhsa_workgroup_processor_mode 1` at 2,048, 8,192, and 32,768 tokens. This avoided the invalid comparison against archived pre-refactor symbols. Both passes were deterministic. For keys `ggsol_65ea830f809a878c`, `ggsol_03c6517b032619d5`, and `ggsol_e80f2d214da77367`, the transformed object and linked HSACO were byte-identical to the parent. ABI, metadata visible to inspection, 144 VGPRs, 16 SGPRs, 18,432 LDS bytes, 32 WMMAs, two barriers, private storage, spills, and every instruction count were unchanged.

F1 is closed as executable-inert under this exact gfx1151 assembler/linker artifact. Timing cannot distinguish byte-identical executables. No serialized policy, typed identity, source retention, or transfer to another family follows.

### F2 exact epilogue and clause results

F2 independently built exact BF16 width-two interleaving and store-clause widths 32, 16, 8, and none at every required token count. Two builds of each variant were deterministic. Every artifact retained 144 VGPRs, 16 SGPRs, 18,432 LDS bytes, 32 WMMAs, two barriers, 451 VALU issues, zero private bytes, and zero spills. The parent has three total clauses, including one 32-store epilogue clause; width 16 has four total clauses, width 8 has six, and no-clause has two. `fixed-grouped-q8-f2-semantic.json` records bitwise parent/public equality and finite output for every variant and token count, repeated-producer equality, and active packed-weight and activation mutations at 2,048 tokens.

The BF16 width-two schedule was mixed in the nine-repeat screen: parent/candidate body ratios were `0.98619x`, `0.99992x`, and `1.01261x` at 2,048/8,192/32,768 tokens, while complete ratios were `1.00200x`, `1.00230x`, and `0.99956x`. Complete/body direction disagrees at both endpoint shapes, so the resource-neutral instruction motion is closed as timing-neutral.

Removing the store clause was a genuine regression, not a threshold closure. Parent/candidate body ratios were `0.94610x`, `0.95758x`, and `0.96067x`, and complete ratios were `0.95964x`, `0.95567x`, and `0.96245x`. The 32-store clause therefore carries material executable scheduling value on this artifact.

Width 16 and width 8 received two independent seven-warmup, 25-repeat top-ups. Width 16's first bracket had all aggregate medians slightly favorable, but five of six robust intervals crossed parity; the second bracket reversed body direction at 2,048 and 8,192 tokens and complete direction at 8,192 tokens. It is closed as timing-neutral. Width 8 repeatedly favored the 32,768-token complete call by about 0.25%-0.28%, with both complete intervals below parity, but its body intervals crossed parity and the 8,192-token complete direction reversed between brackets.

Because the 32,768-token width-8 result was the only surviving shape-specific signal, it was represented locally as distinct canonical `BFloat16RNEClause8`, restricted to the exact retained P3 32,768-token key. Typed builds reproduced the probe instruction-for-instruction after symbol normalization, were deterministic and resource-clean, and passed exact full-size public equality. The fresh typed 25-repeat A/B did not reproduce the gain: candidate/parent was `69.6846/69.5596 ms` in the body and `85.7680/85.7610 ms` complete, movements of +0.18% and +0.008%. Robust 95% intervals were `[-1.82%,+2.22%]` for body and `[-1.37%,+1.41%]` complete. The typed finalist is timing-neutral, so its source changes were removed and no identity is retained.

F2 is complete: exact BF16 chain interleaving, 32/16/8/no-clause boundaries, and the only plausible typed shape-specific finalist have all reached a measured disposition. P3's original 32-store clause remains canonical.

### F3 delay-ALU and VOPD result

The durable target audit is `fixed-grouped-q8-f3-target-scheduling-audit.json`, generated from current P3 key `ggsol_65ea830f809a878c`. The linked hand-authored artifact contains zero `s_delay_alu` instructions. No compiler scheduling pass lies between this assembly source and the assembler/linker, and no producer/consumer motion survived F2 that would create a concrete delay requirement; inserting a delay without such a change can only add a stall. The delay screen is therefore inapplicable on the current premise rather than an unmeasured performance claim.

The same audit applied the exact GFX11 VOPD X/Y lists. P3 already contains 156 accepted VOPD pairs. Every remaining standalone integer add, shift, and AND is Y-only on GFX11, while 33 `v_bfe_u32`, 32 `v_add3_u32`, 130 conversions, ten `v_mul_lo_u32`, four right shifts, and 32 WMMAs are not eligible X instructions. The artifact has zero standalone X-eligible operations, so no additional legal VOPD pair exists even before register-bank masks are considered. In particular, `v_bfe_u32` is not treated as a generic VOPD operand.

F3 is closed by exact target capability and artifact inspection. No delay or VOPD source policy is retained. F1-F3 now have complete dispositions, and the fixed P3 source remains unchanged.

### Post-F3 recursive final review

A separate read-only pass over the restored P3 model, physical plan, lowering, generated artifacts, P0-P6 history, F1-F3 evidence, ordinary Q8 controls, and gfx1151 target definitions found no new exact fixed-group mechanism. Compact ownership, weight-address lifetime, overlap, persistence, loop unrolling, exact BF16 scheduling, clause boundaries, processor mode, delay insertion, and legal VOPD formation are retained or closed by their recorded correctness, resource, executable, and timing evidence. The original 32-store clause is the only tested clause form with material value.

No wait or barrier change has an artifact-specific hazard proof, and ordinary-Q8 scheduling does not transfer without a fixed-group identity. Prepared weights, changed activation production, persistent or split-K launch ownership, public dispatch, bundles, packaging, and fallback remain contract or integration changes. The fresh pass therefore finds no actionable in-contract fixed Q8_0 mechanism.
