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

## Retained Candidates

| Key | Retained body | Confirmation latency HIP / GGTensile | HIP / GGTensile TFLOPS | Speedup vs HIP | Resource envelope |
| --- | --- | ---: | ---: | ---: | --- |
| B1, `R=16384` | `M128/N64`, SIA5, DependencyBatch4, `SerialRoutes`, `Mixed128_64` | `4.6301 / 2.5762 ms` | `7.4210 / 13.3375` | `1.7973x` | 136 VGPR, 35 SGPR, 5 KiB LDS |
| B4, `R=65536` | `M128/N128`, SIA5, DependencyBatch4, `SplitRoutes4`, `Masked` | `9.6852 / 7.5674 ms` | `14.1907 / 18.1619` | `1.2798x` | 208 VGPR, 35 SGPR, 10 KiB LDS |
| B16, `R=262144` | `M128/N128`, SIA5, DependencyBatch4, `SplitRoutes8`, `Masked` | `32.2976 / 24.0474 ms` | `17.0216 / 22.8614` | `1.3431x` | 208 VGPR, 35 SGPR, 10 KiB LDS |

TFLOPS uses `2 * R * 512 * 2048 / (latency_ms * 1e9)` and is reported from the same weighted confirmation summaries. All retained bodies have zero private bytes, zero VGPR/SGPR spills, and no scratch. The B4/B16 bodies have 32 static WMMA instructions and two barriers; B1 mixed has 24 static WMMAs and four barriers because the mutually exclusive M64/N64 tail is emitted alongside the M128 primary body. The B1 mixed search comparison was `2.5869 -> 2.3491 ms` versus its masked parent. On the disjoint confirmation bank it improved `2.8250 -> 2.5762 ms`; the 97.27%-weight confirmation medoid improved `2.8202 -> 2.5665 ms`.

Every retained key passed full-row qualification at its exact `R`: candidate output versus packed HIP was BF16 bit-exact, deterministic reruns were bit-exact, and the independent BF16 dequantized oracle NRMSEs were B1 `6.78e-5`, B4 `8.86e-5`, and B16 `8.05e-5`. Route, gradient, active/inactive weight, invalid expert, malformed offset, sentinel, and complete output-coverage controls passed. Independent source and HSACO rebuilds were byte-identical.

## Mixed-Tail Decision

`Mixed128_64` composes a primary M128 body with a separately lowered M64 body. Its typed contract is restricted to the qualified two-M-tile, 128-thread, single-LDS SIA5/PGR2/PLR1 primary pipeline with no next-tile packed prefetch. Its labels are namespaced, its lane-state handoff is explicit, and its inspection expectation is 48 WMMAs/four barriers for M128/N128. The handoff is required because the two physical plans allocate different serial VGPRs.

The mechanism is retained only for B1. A direct same-process B4 parent bracket (5 warmups, 25 repeats, reverse-rotating order) measured weighted parent/mixed throughput `0.9854x` (`7.6477 / 7.7609 ms`), so B4 mixed is rejected. B16 mixed was also rejected after a fitted screen: `1.3620x` versus the parent `1.3616x`; its 95.51%-weight medoid has no routes at or below 64 rows, making the extra body pure overhead for the dominant profile. The partial correctness harness initially used `active_rows + 1` for its malformed final offset; this was corrected to exact `R + 1` before final qualification.

## Qualification Gates

Correctness covers full tiles, non-aligned tails, first and non-first routes, sparse and repeated physical IDs, invalid experts and offsets, deterministic reruns, aggregate boundaries, input/gradient mutation, active packed-weight mutation, and inactive-expert mutation. Changed accumulation order uses an explicit BF16 numerical envelope rather than a bitwise claim.

Inspection checks symbol identity, ABI metadata, gfx1151, wave32, workgroup, VGPR, SGPR, LDS, private bytes, spills, scratch, calls, stack, barriers, waits, and static WMMA count. Timing uses shared inputs and route metadata, warmed alternating order, a disjoint confirmation corpus, and reversed-order longer runs for retained changes. Small stable gains may accumulate; there is no hard two-percent promotion gate. No selected key may regress a required control beyond the noise-supported retention bound.

## Experiment Log

### Campaign completed

The implementation reuses ordinary Q4_K decode, LDS, WMMA, and store arithmetic beneath a grouped problem identity, strict 56-byte ABI, route scalar plan, lowering facade, inspection path, benchmark, and direct research launcher. The selected ownership is serial at B1, four-way split at B4, and eight-way split at B16. Split ownership is encoded in grid Z and preserves the ABI.

Retained mechanisms are M128/N64 for B1, M128/N128 for B4/B16, SIA5/PGR2/PLR1, dependency-batched Q4_K decode, fitted-prior split-route ownership, and the B1-only mixed M128/M64 tail. Rejected timing paths include the initial pilot, M64 primaries, unbatched decode, double-LDS SIA4 alternatives, nonselected split factors, and B4/B16 mixed tails. Row-task construction, persistent ownership, atomics, and cross-workgroup reduction were deferred because the selected direct bodies leave no measured fitted-prior prerequisite that justifies a new ABI, setup launch, or workspace.

The final repository gates were `706 passed` for `pytest -q tests` and a clean `pre-commit run --all-files`. No production dispatch, generated bundle table, registration, packaging, or HIP fallback file changed.

## Recursive Final Review

Before declaring this campaign complete, reread this record, `docs/ggtensile_plan.md`, `docs/grouped_mmq_bwd_optimization.md`, the ordinary Q4_K backward experiment, grouped forward histories, current HIP and generated sources, normalized ISA, profiler and timing reports, rejected candidates, gfx1151 ISA/LLVM material, and relevant TensileLite/CK implementations.

Classify every remaining idea as retained and measured; rejected by correctness, resources, timing, or reproducibility; contract-incompatible or deferred with an explicit prerequisite; or actionable with an exact target and qualification gate. Implement every actionable finding and repeat the review from the new premise. Completion requires a fresh recursive pass with no actionable in-contract mechanism and every selected exact key correct, deterministic, resource-clean, and faster than its exact HIP control.

This rule is global across related GGTensile directions, formats, and shapes. Evidence may transfer, but ownership, lifetimes, synchronization, arithmetic order, resources, correctness, and exact-key timing must be re-derived here.
