# GGTensile Grouped MMQ Backward Q4_K Experiment

## Purpose

Implement and optimize the first non-paired routed GGTensile backward kernel on gfx1151. The initial format is Q4_K and the workload is the Qwen expert down projection. This is an isolated research campaign: public dispatch, generated bundle tables, extension registration, packaging, and HIP fallback remain unchanged until a separate integration review.

Correctness precedes timing. Every retained artifact must consume authoritative packed GGUF weights directly, preserve FP32 WMMA accumulation and BF16 RNE output, and pass route, mutation, resource, and reproducibility controls.

## Exact Problem

For routed GEMM `g`, backward computes

```text
dY_g[M_g,2048] x W_g[2048,512] -> dX_g[M_g,512]
```

`expert_indices[g]` selects one physical expert from a 256-expert packed bank. `expert_offsets[g]` is the cumulative aggregate-row end, and `M_g` is runtime metadata rather than a generation constant. The exact generation keys are:

| Physical batch | Aggregate rows `R` | Exact `(R,N,K)` |
| ---: | ---: | ---: |
| 1 | 16,384 | `(16384,512,2048)` |
| 4 | 65,536 | `(65536,512,2048)` |
| 16 | 262,144 | `(262144,512,2048)` |

The physical Q4_K bank is `[256,2048,288]`, `dY` is contiguous BF16 `[R,2048]`, and `dX` is contiguous BF16 `[R,512]`. Route metadata contains at most 256 entries. Uniform, fitted-prior, skewed, sparse-ID, repeated-ID, and boundary routes are required controls. Runtime group heights need not be tile multiples and include values around 1, 16, 32, 64, 128, and 129.

## Kernel Contract

The target is gfx1151, wave32, WMMA V1, code-object version 5, direct packed Q4_K decode, FP32 accumulation, and BF16 RNE stores. No dense shadow, predecoded expert bank, hidden cache, host route inspection, atomics, or output workspace is part of the baseline.

The serial routed ABI follows the installed specialized single-backward body: five pointers named `grad_output`, `packed_weight`, `grad_input`, `expert_indices`, and `expert_offsets`; two i32 values named `num_experts` and `rows`; and one i64 `bytes_per_expert`. Route count is launch geometry.

One serial workgroup owns one `(gemmIndex, MacroTile1)` column tile and walks that route's row tiles. Invalid expert IDs, empty or reversed ranges, negative starts, and ends beyond exact `R` are inert. Final row tiles mask `dY` reads and `dX` stores while all lanes still participate in decode, LDS barriers, and WMMA. Any row-task mechanism must expose its task ABI, setup launch, capacity, and complete timing separately.

Retained artifacts require zero private storage, zero spills, no scratch instructions, no calls, and no dynamic stack. Every companion kernel and workspace introduced later is included in those gates.

## Writer Reuse Decision

The ordinary `BackwardKernelWriterAssembly` facade cannot represent routed ownership: its ABI, rectangular M grid, pointer bases, and full-tile contract are ordinary-only. A grouped problem identity, grouped solution identity, route state, facade, inspection path, and launcher are therefore required.

The ordinary backward physical and arithmetic layers are reusable. Q4_K packed loads, scale/minimum reconstruction, decoded-weight LDS layout, local reads, WMMA emission, wait scheduling, and BF16 RNE conversion have the same semantics after the grouped prologue rebases the packed expert and aggregate row pointers. The grouped lowerer subclasses that arithmetic layer and owns route loads, expert rebasing, row traversal, and tail predicates. Shared code uses no `fwd`/`forward` suffix; forward-only modules retain one.

## Phases

### B0: Strict identity and correctness control

Add Q4_K grouped-backward problem, solution, canonical mapping, validation, physical route state, ABI, lowering, writer, inspection, and direct launcher. Build independently twice, require byte-identical source and HSACO, and qualify full and partial routes against the installed packed HIP kernel and an independently dequantized reference. Exercise active/inactive expert and input mutations before timing.

### B1: Exact-key baseline and ownership

Measure the three exact aggregate-row keys on fitted-prior medoids and mandatory controls with warmed alternating-order GPU events. Compare serial M/N geometry and existing HIP serial/row-task ownership before changing decode arithmetic. Setup and allocation time are included for row-task candidates.

### B2: Geometry, pipeline, and tails

Search linked MatrixInstruction, MacroTile0, MacroTile1, DepthU, active-wave, LDS layout/buffering, packed prefetch, A prefetch, decode dependency width, and tail policy fields implemented by the shared backward lowering. Every candidate is exact, typed, resource-inspected, and independently reproducible.

### B3: New mechanisms and selection

Open a new decoder, overlap, mapping, store, persistent, or split-reduction mechanism only from a measured residual bottleneck. Cross-workgroup reduction requires explicit FP32 partial storage, deterministic reduction semantics, and complete-call timing. Public selection remains outside this experiment.

## Ranking Objective

Promotion ranking is the fitted Qwen routing-prior weighted sum of per-medoid median complete-call latency across its five search or confirmation medoids. Complete-call timing includes output allocation and launch. The uniform, skewed, sparse-ID, and boundary distributions are correctness and diagnostic controls; low-weight medoid regressions do not block a candidate when the fitted weighted objective improves. Results below use warmed GPU events and disjoint reversed-order confirmation.

## Retained Parent History

The pre-refactor campaign retained these direct bodies before the inactive-M refinement:

| Key | Retained parent body | Resource envelope |
| --- | --- | --- |
| B1, `R=16384` | `M128/N64`, SIA5, DependencyBatch4, `SerialRoutes`, `Mixed128_64` | 136 VGPR, 35 SGPR, 5 KiB LDS |
| B4, `R=65536` | `M128/N128`, SIA5, DependencyBatch4, `SplitRoutes4`, `Masked` | 208 VGPR, 35 SGPR, 10 KiB LDS |
| B16, `R=262144` | `M128/N128`, SIA5, DependencyBatch4, `SplitRoutes8`, `Masked` | 208 VGPR, 35 SGPR, 10 KiB LDS |

The parent-only B1 mixed refinement measured `2.8250 -> 2.5762 ms` on its disjoint confirmation bank. The later inactive-M confirmations supersede that parent timing for the accepted performance summary below. Every retained key passed the full-row packed-HIP, deterministic, mutation, sentinel, and independent-oracle gates; independent source and HSACO rebuilds were byte-identical.

## Mixed-Tail Decision

`Mixed128_64` composes a primary M128 body with a separately lowered M64 body. Its typed contract is restricted to the qualified two-M-tile, 128-thread, single-LDS SIA5/PGR2/PLR1 primary pipeline with no next-tile packed prefetch. Its labels are namespaced, its lane-state handoff is explicit, and its inspection expectation is 48 WMMAs/four barriers for M128/N128. The handoff is required because the two physical plans allocate different serial VGPRs.

The mechanism is retained only for B1. A direct same-process B4 parent bracket (5 warmups, 25 repeats, reverse-rotating order) measured weighted parent/mixed throughput `0.9854x` (`7.6477 / 7.7609 ms`), so B4 mixed is rejected. B16 mixed was also rejected after a fitted screen: `1.3620x` versus the parent `1.3616x`; its 95.51%-weight medoid has no routes at or below 64 rows, making the extra body pure overhead for the dominant profile. The partial correctness harness initially used `active_rows + 1` for its malformed final offset; this was corrected to exact `R + 1` before final qualification.

## Qualification Gates

Correctness covers full tiles, non-aligned tails, first and non-first routes, sparse and repeated physical IDs, invalid experts and offsets, deterministic reruns, aggregate boundaries, input/gradient mutation, active packed-weight mutation, and inactive-expert mutation. Changed accumulation order uses an explicit BF16 numerical envelope rather than a bitwise claim.

Inspection checks symbol identity, ABI metadata, gfx1151, wave32, workgroup, VGPR, SGPR, LDS, private bytes, spills, scratch, calls, stack, barriers, waits, and static WMMA count. Timing uses shared inputs and route metadata, warmed alternating order, a disjoint confirmation corpus, and reversed-order longer runs for retained changes. Small stable gains may accumulate; there is no hard two-percent promotion gate. No selected key may regress a required control beyond the noise-supported retention bound.

## Final Accepted Performance

The accepted identities add wave-uniform inactive-M consumer suppression to the selected Q4_K bodies at all three keys.

| Key (`R`) | Accepted body | HIP / final latency (ms) | HIP / final TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: | ---: |
| B1 (`R=16384`) | M128/N64, SIA5, DependencyBatch4, `SerialRoutes`, `Mixed128_64`, inactive-M | `4.6301 / 2.5084` | `7.4209 / 13.6978` | `1.8458x` |
| B4 (`R=65536`) | M128/N128, SIA5, DependencyBatch4, `SplitRoutes4`, `Masked`, inactive-M | `9.6852 / 7.3674` | `14.1906 / 18.6550` | `1.3146x` |
| B16 (`R=262144`) | M128/N128, SIA5, DependencyBatch4, `SplitRoutes8`, `Masked`, inactive-M | `32.2976 / 23.8101` | `17.0216 / 23.0892` | `1.3565x` |

The inactive-M reports measure retained parent versus final candidate, not a fresh three-way HIP bracket. To avoid mixing timing sessions, the HIP and retained-parent values use the earlier disjoint HIP confirmation, and the final latency/TFLOPS and speedup are normalized with the independently confirmed candidate/parent ratios. The latest raw brackets are `2.5845 -> 2.5165 ms` at B1, `7.5515 -> 7.3519 ms` at B4, and `24.0434 -> 23.8061 ms` at B16; reports are `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q4-b1-confirm25.json`, `...-b4-confirm25.json`, and `...-b16-confirm25.json`.

## Experiment Log

### Campaign completed

The implementation reuses ordinary Q4_K decode, LDS, WMMA, and store arithmetic beneath a grouped problem identity, strict 56-byte ABI, route scalar plan, lowering facade, inspection path, benchmark, and direct research launcher. The selected ownership is serial at B1, four-way split at B4, and eight-way split at B16. Split ownership is encoded in grid Z and preserves the ABI.

Retained mechanisms are M128/N64 for B1, M128/N128 for B4/B16, SIA5/PGR2/PLR1, dependency-batched Q4_K decode, fitted-prior split-route ownership, and the B1-only mixed M128/M64 tail. Rejected timing paths include the initial pilot, M64 primaries, unbatched decode, double-LDS SIA4 alternatives, nonselected split factors, and B4/B16 mixed tails. Row-task construction, persistent ownership, atomics, and cross-workgroup reduction were deferred because the selected direct bodies leave no measured fitted-prior prerequisite that justifies a new ABI, setup launch, or workspace.

The final repository gates were `706 passed` for `pytest -q tests` and a clean `pre-commit run --all-files`. No production dispatch, generated bundle table, registration, packaging, or HIP fallback file changed.

## Recursive Final Review

Before declaring this campaign complete, reread this record, `docs/ggtensile_plan.md`, `docs/grouped_mmq_bwd_optimization.md`, the ordinary Q4_K backward experiment, grouped forward histories, current HIP and generated sources, normalized ISA, profiler and timing reports, rejected candidates, gfx1151 ISA/LLVM material, and relevant TensileLite/CK implementations.

Classify every remaining idea as retained and measured; rejected by correctness, resources, timing, or reproducibility; contract-incompatible or deferred with an explicit prerequisite; or actionable with an exact target and qualification gate. Implement every actionable finding and repeat the review from the new premise. Completion requires a fresh recursive pass with no actionable in-contract mechanism and every selected exact key correct, deterministic, resource-clean, and faster than its exact HIP control.

This rule is global across related GGTensile directions, formats, and shapes. Evidence may transfer, but ownership, lifetimes, synchronization, arithmetic order, resources, correctness, and exact-key timing must be re-derived here.

## Reopened post-refactor optimization program

The recursive writer/ISA review found two changed premises that were not represented by the completed serial/split campaign. They are ranked only by the fitted Qwen expert-medoid objective; uniform, skewed, sparse-ID, and boundary routes remain correctness and diagnostic controls rather than promotion vetoes:
- Add wave-uniform inactive-M consumer suppression while keeping packed decode, required LDS reads, and barriers uniform. Existing GGTensile tails mask invalid global accesses but still issue WMMAs for wholly inactive 16-row minitiles. The HIP analog improved every Q4_K B4/B16 route by `5.27-11.60%`, or `9.16%` geometrically. The first control must preserve the 56-byte serial ABI and exact BF16 output and compare the selected B1/B4/B16 identities against source-preserved parents on fitted medoids.
- Add a separate numeric `DeviceRowTasks` ownership and task ABI, using the existing device-only prefix-sum setup and M-major descriptors. The focused HIP B1 screen measured fitted-prior Q4_K row-task throughput at `1.2194x` relative to serial and captured-profile throughput at `1.1351x`. Its old rejection came only from synthetic timing controls, so it does not close this fitted-objective experiment. Setup, allocation, and bounded inactive task slots must be included in complete-call timing.
- If fixed J128 tasks advance, compare one J64 endpoint or a single-launch mixed J128/J64 tail policy. Do not reopen N-major ordering, a second full/tail launch, fixed persistent traversal, or another global `SplitRoutes` sweep: those mechanisms already have direct negative controls.
- Reconfirm the selected `DependencyBatch4` schedule under any retained ownership or suppression mechanism. It is already implemented and selected on all three Q4_K keys; it is an interaction control, not missing work.
- Reopen scale/minimum decode sharing only if the ownership/tail changes expose a resource-clean lower-state design. Standalone metadata broadcast, dedicated decoder waves, approximate accumulators, Stream-K, XCC remapping, and generic prepared shadows remain closed by direct evidence or contract mismatch.

The implementation order is inactive-M suppression, device J128 ownership, task-size/mixed ownership, then decode interaction. Every coherent success or failure is recorded here immediately. A retained mechanism requires exact packed-HIP results, deterministic independent rebuilds, zero private bytes and spills, fitted confirmation with complete-call setup/allocation, and no malformed-route or sentinel failure.

### Inactive-M suppression build and correctness checkpoint

The first identity-neutral implementation guards each wholly inactive wave and each inactive 16-row M minitile around the WMMA consumer, while leaving packed decode, LDS traffic, barriers, and masked global access uniform. It reuses dead post-prologue scalar state and adds no solution field. Every grouped Q4_K manifest control assembles, covering M128/M256, serial ownership, split2 through split32, masked, mixed, and secondary tails. All retain their prior VGPR/SGPR/LDS envelope, zero private bytes and spills, identity hash, and static WMMA/barrier counts. The complete build record is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-manifest-build/build.json`.

The strict 625-row boundary matrix for the M256 plus secondary-M128 `ggsol_93f56dc0fb4e25c8` control passes packed HIP, independent BF16, deterministic rerun, gradient/route/active/inactive-weight mutations, malformed expert/offset controls, and tail sentinels. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q4-secondary-correctness.json`. This is a build/correctness checkpoint only; selected-key fitted parent/candidate timing remains the promotion gate.

The selected B1 `ggsol_356ef05b9c33365d` source was built independently from the pre-change `c41913f` worktree and the suppression worktree. Both inspect at `136 VGPR / 35 SGPR / 5120 B LDS`, zero private bytes and spills, 24 static WMMAs, and four barriers. A same-process fitted search-bank bracket with allocation in both paths, three warmups, nine repeats, and rotating order improves weighted latency `2.3495 -> 2.3169 ms`, or `1.0141x` parent throughput. Outputs are bitwise equal on all five medoids. The dominant medoid and four of five medoids improve; one low-weight medoid measures `0.9577x`, which is diagnostic rather than a fitted-objective veto. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q4-b1-search9.json`. This narrow result advances to disjoint confirmation.

Disjoint confirmation with five warmups, 25 repeats, reversed base order, and the same complete-call allocation contract confirms `2.5845 -> 2.5165 ms`, or `1.0270x`. Every confirmation medoid is bitwise exact; four improve and the fifth is a `0.9981x` tie. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q4-b1-confirm25.json`. Inactive-M suppression is retained for Q4_K B1 and advances to B4/B16 qualification.

The selected B4 split4 `ggsol_9950e632db681be3` and B16 split8 `ggsol_14b7c7089cdc2480` artifacts reproduce their parent envelopes at `208 VGPR / 35 SGPR / 10240 B LDS`, with zero private bytes or spills. Both pass the strict 625-row packed-HIP, independent BF16, deterministic, mutation, malformed-route, and sentinel matrix. Reports are `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q4-b4-correctness.json` and `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q4-b16-correctness.json`. Fitted parent/candidate timing remains outstanding.

The B4 fitted search-bank bracket improves `7.5691 -> 7.3541 ms`, or `1.0292x`, with every medoid bitwise exact and faster. The minimum ratio is `1.0279x`. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q4-b4-search9.json`; B4 advances to disjoint confirmation.

The B16 fitted search-bank bracket rejects suppression for that exact key: weighted latency regresses `23.7528 -> 23.8766 ms`, or `0.9948x` parent throughput, and every medoid is slower despite bitwise equality. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q4-b16-search9.json`. The retained implementation must leave the selected Q4_K B16 source on its parent path unless a later ownership interaction supplies a separately measured changed premise.

The disjoint B4 confirmation bracket measures `7.5515 -> 7.3519 ms`, or `1.0271x`, with every medoid bitwise exact and faster. The minimum ratio is `1.0217x`. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q4-b4-confirm25.json`. Suppression is retained for selected Q4_K B1/B4 and rejected for selected Q4_K B16.

The EvoTensile median-log robust-scale analysis marks B1 and B4 confidently faster at 95%, with candidate-minus-parent intervals `[-3.868,-1.381]%` and `[-4.442,-0.821]%`. The aggregate analysis is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-confirm25-confidence.json`. B16 receives a full confirmation run before its source path is finalized because its rejection currently has only nine-repeat search evidence.

The full B16 confirmation reverses the short search result: `24.0434 -> 23.8061 ms`, or `1.0100x`, with every medoid bitwise exact and faster. Its separate median-log robust-scale analysis gives a 95% candidate-minus-parent interval of `[-1.512,-0.462]%`, so this is resolved rather than a point-estimate tie. Reports are `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q4-b16-confirm25.json` and `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q4-b16-confirm25-confidence.json`. The nine-repeat B16 rejection is superseded as noise; suppression is retained across selected Q4_K B1/B4/B16.

Final selected full-row qualification passes at `16384`, `65536`, and `262144` rows with packed-HIP bit equality, deterministic reruns, all active/inactive mutation controls, malformed-route sentinels, and independent BF16 NRMSE below `0.01`. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q4-full-correctness.json`.

### Device J128 ownership build checkpoint

The first typed `DeviceRowTasks` identity uses a separate 64-byte backward task ABI, numeric `row_task_rows=128`, the existing device-only prefix-sum setup contract, bounded capacity `ceil(R/128) + route_entries`, and an M-major grid with N tiles in X and task slots in Y. The Q4_K B1 candidate `ggsol_dc03b37d72af1924` composes the qualified SIA5/PGR2/PLR1 padded M128/N128 body and `DependencyBatch4` decode with one descriptor-owned M128 tile per workgroup. It assembles at `208 VGPR / 44 SGPR / 10240 B LDS`, 32 static WMMAs, and two barriers, with zero private bytes, spills, scratch, or calls. The initial build record is `~/tmp/torch-ggml-ops/ggtensile-device-row-tasks-build-a/build.json`. Correctness and independent reproducibility remain outstanding; this checkpoint is not a timing result.

An independent second build is byte-identical. The second record is `~/tmp/torch-ggml-ops/ggtensile-device-row-tasks-build-b/build.json`. Reproducibility and static resource gates pass; runtime correctness remains the next gate.

The full `R=16,384` boundary matrix passes for `ggsol_dc03b37d72af1924`. Candidate output is BF16 bit-identical to the installed packed HIP control across 8,388,608 elements, deterministic reruns differ in zero elements, inactive task slots leave zero tail writes, malformed first descriptors are inert while later valid descriptors remain active, and the independent BF16 oracle NRMSE is `6.78e-5`. The report is `~/tmp/torch-ggml-ops/ggtensile-device-row-tasks-q4-full-correctness.json`. The task ABI and J128 ownership pass correctness; fitted timing is the remaining promotion gate.

An adjacent nine-repeat public-HIP screen measures fitted weighted latency `4.2565 -> 2.4797 ms`, or `1.7166x` HIP/task throughput, with a minimum medoid ratio of `1.5599x`. This validates the stale HIP-relative premise but is not the retention comparison because the selected serial GGTensile parent is materially faster than HIP. The report is `~/tmp/torch-ggml-ops/ggtensile-device-row-tasks-q4-search9-vs-hip.json`.

The nine-repeat fitted search-bank bracket rejects fixed J128 against the selected serial GGTensile parent. Complete-call weighted latency regresses `2.2934 -> 2.5345 ms`, or `0.9049x` parent/task throughput; kernel-only weighted throughput is `0.9271x`, so setup and allocation are not the primary loss. The 94.14%-weight medoid regresses to `0.8802x`, while the four low-weight long-tail medoids improve from `1.2166x` through `1.6098x`. This confirms that tasks redistribute the rare tall-expert profiles but lose the fitted objective to the selected M128/N64 mixed-tail serial body. All medoids are bitwise exact. The report is `~/tmp/torch-ggml-ops/ggtensile-device-row-tasks-q4-vs-parent-search9.json`. A disjoint reversed-order bracket will confirm rejection; J64 and mixed task sizes remain closed because fixed J128 did not advance.

The disjoint five-warmup, 25-repeat reversed-order confirmation agrees: complete-call weighted latency regresses `2.4790 -> 2.5800 ms`, or `0.9608x`; kernel-only weighted throughput is `0.9724x`. The 97.27%-weight confirmation medoid regresses to `0.9477x`, while all four low-weight medoids again improve. The EvoTensile median-log robust-scale 95% candidate-minus-parent slowdown intervals are `[2.503,5.419]%` complete and `[1.174,4.220]%` kernel-only. Reports are `~/tmp/torch-ggml-ops/ggtensile-device-row-tasks-q4-vs-parent-confirm25.json` and `~/tmp/torch-ggml-ops/ggtensile-device-row-tasks-confirm25-confidence.json`. Fixed J128 is rejected and its implementation is removed. By the declared ordering, J64, mixed J128/J64 tasks, and ownership-conditioned decode requalification are not opened. The selected serial B1 identity remains authoritative.
