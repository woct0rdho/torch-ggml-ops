# GGTensile Grouped MMQ Backward Pair IQ2_XXS Experiment

## Purpose

Implement and optimize isolated gfx1151 grouped paired IQ2_XXS backward kernels for the DeepSeek routed gate/up projections. Public dispatch, generated bundles, extension registration, packaging, and HIP fallback remain unchanged pending a separate integration review.

The production objective is lower complete-call latency than the exact installed HIP control at every required aggregate-row key. Search profiles may rank intermediate candidates, but promotion requires exact production-row correctness, zero spills/private storage, deterministic artifacts, and formally confirmed timing at B1, B4, and B16.

## Exact Contract

For routed GEMM `g`:

```text
dG_g[M_g,2048] x W_gate_g[2048,4096]
+ dU_g[M_g,2048] x W_up_g[2048,4096]
-> dX_g[M_g,4096]
```

The exact aggregate-row keys are `R={12288,49152,196608}`. Each physical IQ2_XXS expert bank is `[256,2048,1056]`: each 256-value block occupies 66 bytes, each packed row contains sixteen blocks, and each expert occupies 2,162,688 bytes. Both gradient outputs and the shared gradient input are contiguous BF16. The arithmetic contract uses signed IQ2_XXS codebook values, the odd per-K32 group scale, FP32 scale arithmetic, one shared FP32 accumulator set for both projections, and one final BF16 RNE store.

The isolated research ABI is the production-compatible 72-byte specialized pair ABI: `first_grad_output`, `second_grad_output`, `first_packed_weight`, `second_packed_weight`, `grad_input`, `expert_indices`, `expert_offsets`, `num_experts`, `rows`, and `bytes_per_expert`. Route count and any typed split ownership remain launch geometry. Independent projection workgroups, an intermediate destination, atomic accumulation, or reloading a stored first projection are outside the fused-pair contract.

The authoritative codebook is extracted at generation time from `csrc/vendor/llama_cpp/iq2_xxs_grid.cuh`. The independent oracle dequantizes only selected routed experts and sums the two routed matmuls in FP32 before BF16 conversion.

## Production Control

Installed HIP dispatches `GroupedBwdPairIQ2XXSN2048K4096M64N64` at B1. Exact aggregate rows 49,152 and 196,608 dispatch `GroupedBwdTunedPairIQ2XXSN2048K4096M128N64`. Both use 128 threads as four wave32 waves, N64, K32, cooperative width-16 decode, swizzle4, separate LDS tiles for the two weight banks, inactive-M consumer suppression, and one final store. The grid is `(4096 / 64, num_groups, 1)`.

The specialized ABI validates exact row count, physical bank geometry, common route geometry, expert count, and the 2,162,688-byte expert stride. Invalid experts, non-increasing offsets, negative starts, and offsets beyond `rows` make that route inert.

## Qualification

Correctness covers exact rows, non-aligned route tails, first and non-first routes, sparse and repeated expert IDs, deterministic reruns, independent mutation of both gradient outputs and both active weight banks, inactive-expert mutations, malformed route controls, and untouched sentinels. Candidate versus installed packed HIP must be BF16 bit-exact. An independently dequantized bounded reference must remain finite and meet the established normalized-error gate.

Inspection requires gfx1151, wave32, code object v5, exact pair metadata, bounded register indices, zero private bytes and spills, no scratch, calls, or dynamic stack, and derived static WMMA/barrier counts. Independent generate/build/inspect roots must produce byte-identical source and HSACO.

Timing uses warmed rotating GPU events and includes equivalent output allocation in candidate and installed complete-call paths. Retained finalists receive disjoint confirmation with at least 5 warmups and 25 repeats, reversed rotating order, independent and paired robust confidence intervals, and production-row validation. Pair TFLOPS is `4 * R * 2048 * 4096 / (latency_ms * 1e9)`.

## Final Accepted Performance

The final accepted dispatch is mixed by key: the confirmed M64 SIA5 K pipeline at B1 and the confirmed M128 direct-pointer/swizzle8 body at B4/B16.

| Key (`R`) | Accepted body | HIP / final latency (ms) | HIP / final TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: | ---: |
| B1 (`R=12288`) | M64/N64, SIA5 K pipeline, dual LDS, swizzle8, codebook overlap | `38.0585 / 31.7568` | `10.8338 / 12.9836` | `1.1984x` |
| B4 (`R=49152`) | M128/N64, direct second-projection pointers, activation prefetch, swizzle8 | `97.7674 / 69.9801` | `16.8693 / 23.5677` | `1.3971x` |
| B16 (`R=196608`) | M128/N64, direct second-projection pointers, activation prefetch, swizzle8 | `378.3588 / 285.1829` | `17.4360 / 23.1328` | `1.3267x` |

HIP and GGTensile were measured in the same paired confirmation runs, with equivalent output allocation and pre-timing exactness checks. B1 uses `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-m64-k-pipeline-sia5-codebook-overlap/timing-b1-confirmation25.json`; B4/B16 use `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-m128-direct-swizzle8/timing-confirm25.json`.

## Planned Search

- Reuse the ordinary backward tile, WMMA, route, ABI, store, runtime, and inspection machinery.
- Add typed IQ2_XXS width-16 packed reads and signed codebook reconstruction. Keep its 66-byte format and odd K32 scale distinct from IQ2_S.
- Establish a correct M64/N64 `InterleavedDepthU` fused anchor. It decodes and consumes each bank serially through one LDS tile and intentionally prioritizes semantic clarity over synchronization count.
- Compare M64 and M128 early. Then test only mechanisms justified by measured deficits: dual LDS, concurrent bank reads, activation and packed-weight pipelining, route ownership, decode scheduling, swizzle4, and exact geometry.
- Retain only mechanisms with explicit correctness, resource, timing, and identity records. Remove failed-only source identities before final review.

## Continued Optimization Plan

- Preserve the retained M64 SIA5 pipeline as the B1 control while prioritizing the production M128 path, where historical evidence shows larger B4/B16 gains and therefore the highest remaining payoff.
- Rebuild and remeasure the exact M128 activation-prefetch schedule from the current writer at R49,152 and R196,608. Require BF16 exactness, bounded resources, and stable search timing before treating the historical gain as current evidence.
- Compare installed HIP and GGTensile machine code and inspect `~/amd-llvm-project/` where compiler scheduling or instruction selection needs explanation. Translate mechanisms, not compiler accidents: activation/weight latency hiding, codebook/decode issue width, LDS access shape, wait frontiers, WMMA ordering, and loop-control overhead.
- Port the retained M64 IQ2_XXS decode batching, paired codebook overlap, direct pointers, and SIA5 K pipeline to M128 one mechanism at a time. Reject any step that spills, changes BF16 output, or loses on the fitted-prior timing gate.
- Once large M128 margins are exhausted, revisit smaller B1 gains and cross-geometry mechanisms. Confirm every finalist with disjoint 25-repeat timing and independent/paired confidence intervals.
- After every coherent retained or failed experiment, append its exact identity, artifact/resources, correctness, timing, and disposition to this document. Commit each source/test improvement worth retaining; documentation-only records need no separate commit.
- Keep the recursive final-review rule below unchanged and repeat it after each retained mechanism and at campaign completion.

## Recursive Final Review

After each optimization round, reread the contract, retained source and machine code, installed HIP control, timing reports, correctness reports, resource inspection, and rejected experiments from first principles. Classify every remaining idea as retained and measured; rejected by correctness, resources, timing, or reproducibility; contract-incompatible or deferred with a prerequisite; or actionable with an exact target and gate. Implement every actionable finding and repeat the review.

Completion requires exact bounded and production correctness, zero spills/private storage, deterministic source and HSACO, complete production-row timing with confidence, focused and broad tests, documentation, cleanup of failed-only identities, and a final recursive pass with no actionable in-contract mechanism.

## Completion Record

### Campaign opened and implementation boundary audited

The target is the DeepSeek gate/up backward pair, not an ordinary backward kernel and not either projection in isolation. The physical bank shape is `[256,2048,1056]`, giving a 1,056-byte packed row and a 2,162,688-byte expert stride. B1 uses the installed M64 body; B4 and B16 use the separately qualified M128 body.

The existing paired GGTensile framework already owns the fused 72-byte ABI, route validation and rebasing, bounded M tails, a shared FP32 accumulator set, first-then-second projection accumulation, and one BF16 epilogue. Its simplest `InterleavedDepthU` schedule is quant-agnostic once packed reads and decode are supplied. Later dual-LDS and pipelined schedules currently contain IQ2_S-specific metadata helpers and are not admitted for the first IQ2_XXS identity.

The authoritative forward IQ2_XXS lowering and HIP `decode_backward_tile_group<GGML_TYPE_IQ2_XXS,16>` establish the packed semantics. The first retained source change will add a dedicated width-16 IQ2_XXS reader/decoder while reusing the generic backward tile and paired writer. No performance claim is recorded until the bounded artifact is bit-exact, resource-clean, and independently inspected.

### First fused IQ2_XXS anchor

The backward format contract now admits IQ2_XXS without conflating its 66-byte block with IQ2_S. The dedicated reader maps each of 128 threads to one aligned 16-value group in the 64-column by K32 decoded tile, loads one 64-bit index/sign word plus the shared FP16 block scale, reconstructs the two authoritative grid entries in staged LDS, expands each seven-bit sign group with odd parity, forms `d * (2*scale+1) / 8` in FP32, and stores BF16 RNE values through the reusable XOR-4 decoded-weight layout. The pair body reuses the routed ABI, bounded M ownership, BF16 WMMA, shared FP32 accumulators, projection pointer swaps, and one final store.

Canonical M64/N64 and M128/N64 single-LDS keys round-trip at bounded and production rows. Strict gfx1151 assembly inspection gives:

| Geometry | VGPR / SGPR / LDS | Static WMMA / barriers | Private bytes / spills |
| --- | ---: | ---: | ---: |
| M64/N64 | `87 / 41 / 6,144 B` | `16 / 5` | `0 / 0` |
| M128/N64 | `127 / 41 / 6,144 B` | `32 / 5` | `0 / 0` |

Both artifacts have bounded register indices, exact 72-byte metadata, wave32, code object v5, no scratch, calls, or dynamic stack. The M64 R35 candidate matches the installed specialized M64 control in all 143,360 BF16 outputs on real DeepSeek gate/up banks. An expanded R257 matrix also has zero differing elements for one-route full/tail ownership, a short first route followed by full tiles, three nonaligned routes, repeated experts, and mixed 64/65-row boundaries. Every result is finite and no destination sentinel remains.

The first lane mapping review caught and corrected an implementation error before this qualification: the lower/upper 16-value half of a K32 group is selected by lane bit zero. The retained source and test fix that mapping explicitly. The single-LDS body is now the correctness anchor; its reproducibility and timing qualification are recorded below.

### Coverage closure for the retained decoder

The writer coverage test now dispatches a normal IQ2_XXS emitter through the generic decode-prepare and decode-chunk helpers, exercises the unsupported current-address and cross-row decode guards, and separately verifies that a physical plan without the staged codebook is rejected. The direct targeted test passes. A run containing only the writer-coverage and pair modules reports the expected shared-session coverage failure because other backward-writer paths are not imported/executed in that process; it is not a product failure. The complete `tests/ggtensile` session passes with `698 passed`.

### M128 artifact qualification

The production M128/N64 IQ2_XXS pair was independently generated, assembled, linked, and inspected for the B4 row key. It passed the strict gfx1151/code-object-v5 and paired-ABI gates: 72-byte kernarg metadata, wave32, `127` VGPR, `41` SGPR, `6,144 B` LDS, `32` static WMMA instructions, and `5` barriers. The artifact has zero private bytes, zero VGPR/SGPR spills, bounded register indices, no scratch instructions, calls, dynamic stack, or delayed-ALU/clause metadata. Static inspection is complete for both production geometries.

### Bounded fused correctness qualification

Fresh backward-pair artifacts, distinct from the similarly named forward-pair experiment roots, were generated from the committed writer for R35 M64 and M128 and R257 M128. Both R35 geometries are BF16 bit-exact to the installed specialized M64 control for one-route, boundary, sparse-ID, repeated-expert, first/non-first, and unowned-final-row profiles. Deterministic reruns, independent mutations of both gradient outputs, route IDs, and active rows in both packed banks are exact to HIP. Mutations of an inactive expert in either bank are inert after restoration.

Invalid first and middle experts, a negative route start, and a final offset beyond `rows` preserve every sentinel in the malformed route span while valid prior or later routes continue to write. The R257 M128 body is exact for two-full-plus-tail, exact-full-then-tail, tail-full-tail, and repeated-boundary ownership, including deterministic reruns and independent full-route gradient mutations.

The independent oracle dequantizes only the four routed gate/up experts to BF16, performs each routed pair in FP32, sums before BF16 conversion, and differs in 274 of 143,360 values. Maximum absolute error is `0.03125`, error RMS is `2.773e-4`, and NRMSE is `1.656e-4` for both geometries. The durable build and bounded reports are `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-anchor-current/build.json` and `bounded-correctness.json`.

### Exact production-row correctness

The production controls use M64 at B1 and M128 at B4/B16. Boundary-route comparisons cover 50,331,648 B1 values, 201,326,592 B4 values, and 805,306,368 B16 values. Every candidate/control comparison and deterministic rerun has zero differing BF16 elements, zero absolute error, and finite output. Candidate, installed, and rerun destinations start from three different sentinels, so exact equality also proves complete valid-row coverage rather than shared preservation of an unwritten value.

The durable report is `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-anchor-current/production-correctness.json`. Correctness gates are complete for the single-LDS anchors; the later reproducibility and warmed complete-call timing records are documented in the subsequent qualification sections.

### Fitted-prior geometry baseline

The first ranking pass uses all five learned and five hash medoids in the DeepSeek search bank at each production key. Combined medoid weights use the fitted reporting mixture (`40/43` learned and `3/43` hash). Every profile passed an exact candidate/control comparison before timing. The protocol uses three warmups, nine rotating repeats, equivalent output allocation in each complete-call path, and a preallocated kernel-only diagnostic. The search rejected M64 at all keys and M128 at B1/B16; only M128 at B4 advanced as a finalist. It was a search-bank result, not a promotion claim, and was later superseded by the disjoint confirmations in the accepted summary. Public dispatch remains unchanged. The durable report, including all samples and per-medoid route metadata, is `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-anchor-current/geometry-search9.json`.

### M64 activation-prefetch LDS layout selection

The staged M64 sequence retained separate 4 KiB weight tiles, split the two projection bodies at a full-tile boundary, issued their packed reads concurrently, and prefetched the next activation tile with PGR2/SIA4. On the B1 search bank, the swizzle4 activation-prefetch body reached `37.7968 ms` weighted complete-call latency, compared with `38.6737 ms` for the single-LDS anchor, `39.6159 ms` for dual LDS, `39.4362 ms` for the full-tile split, and `39.2062 ms` after concurrent packed reads. This made activation prefetch the schedule anchor but left it `3.18%` behind installed HIP.

The installed body uses paired stride-64 LDS reads, while the swizzle4 GGTensile body issued eight 64-bit LDS reads for each pair of N16 fragments. Changing only the decoded-weight XOR chunk tested the same LDS issue-width premise through the writer's existing vector path. Swizzle8 and swizzle16 both replaced those reads with four 128-bit reads and reduced the physical plan from `105` to `101` VGPR. Static LDS instructions fell from `201` to `137`. Swizzle8 reduced static VALU issues from `1,020` to `944`; swizzle16 reached `940` but regressed sharply and was rejected by timing.

The nine-repeat B1 search-bank screen favored swizzle8 over the swizzle4 activation-prefetch anchor; swizzle16 regressed and was rejected. A disjoint confirmation then selected the swizzle8 path.

A disjoint confirmation used the five learned and five hash medoids from the confirmation bank, five warmups, 25 repeats, and every rotation of reversed function order. Swizzle8 measured `37.2481 ms`, versus `38.8967 ms` for swizzle4 and `37.5809 ms` for installed HIP: a `1.0443x` anchor speedup and `1.0089x` installed speedup. Every pre-timing result was BF16 bit-exact.

Fresh swizzle8 R35/R257 artifacts passed the route-tail, sparse/repeated expert, malformed-route, sentinel ownership, deterministic rerun, independent gradient/weight/route mutation, inactive-expert, and independently dequantized FP32-oracle gates. The production B1 boundary route matched all `50,331,648` installed BF16 elements exactly on the initial run and deterministic rerun. Strict inspection reports `101` VGPR, `41` SGPR, `10,240 B` LDS, 32 WMMAs, five barriers, zero private bytes, and zero spills. The retained typed identity is `iq2_xxs_m64_n64_dual_lds_full_tile_split_concurrent_reads_prefetch_a`; it selects swizzle8 without changing the canonical M128 identities or public dispatch.

Durable records are under `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-m64-swizzle/`: `build.json`, `timing-b1-search9.json`, `timing-b1-confirmation25.json`, `bounded-correctness-swizzle8.json`, `production-correctness-swizzle8.json`, and `selected-identity.json`.

### Final M64 SIA5 K-pipeline

The retained final constructor is `iq2_xxs_m64_n64_sia5_dual_lds_full_tile_split_k_pipeline()`. Its exact identity is M64/N64/K32 with matrix instruction `(16,16,16,1,1,1,4,4,1)`, decoder width 16, swizzle8, PGR2 activation prefetch, current and next packed-weight prefetch, direct second-projection pointers, serial route ownership, and SIA5 (`ScheduleIterAlg=5`). The physical plan is `101 VGPR / 41 SGPR / 10,240 B LDS`, with two disjoint 4 KiB decoded-weight tiles at offsets `0` and `4,096` and one shared IQ2_XXS codebook at offset `8,192`. It has zero private bytes and zero spills.

The final lowering overlaps the paired codebook reads and decode preparation, retains the geometry-derived second-A wait frontier (`vmcnt(2)` for M64), and uses explicit full/tail K-pipeline labels with seven barriers and 32 WMMA instructions. Focused permanent-identity tests cover round-trip admission, exact schedule fields, resource/LDS placement, direct-pointer ownership, absence of pointer swaps, SIA5 and K-pipeline markers, codebook-overlap ordering, M64/M128 frontier behavior, exact altered-identity rejection, and two-root source/object/HSACO reproducibility. The focused regression run passes `4` tests.

The R257 and R12,288 hashes and inspections are recorded in the same `build.json`. Strict source inspection reports `1,438` VALU issues, `209` LDS operations, `105` VMEM operations, `32` WMMAs, and `7` barriers, with bounded registers and no scratch, calls, stack, private storage, or spills.

The disjoint B1 confirmation used five warmups, 25 repeats, reversed rotating order, and independent and paired robust confidence intervals. Relative to installed HIP, the final pipeline improves complete-call latency by `16.612%` in the independent comparison and `16.562%` in the paired comparison; both 95% confidence intervals exclude zero. The confidence report is `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-m64-k-pipeline-sia5-codebook-overlap/timing-b1-confirmation25-confidence.json`.

### Rejected and deferred mechanisms

The following complete mechanisms were measured and rejected: store interleaving, width-16 decode, deferred codebook issue, midpoint codebook wait, SIA4 K pipelining, and VMEM-frontier overrides. The VMEM frontier regressed direct SIA5 by approximately `0.054%` weighted and pipeline SIA5 by approximately `0.026%`; the retained frontier remains the matrix-geometry-derived LDS/VMEM schedule. Public dispatch, generated bundle integration, packaging, and HIP fallback remain deferred.

### Recursive final review

The final review reread the exact paired contract, retained lowering and source, installed HIP control, static inspection, bounded and production correctness, deterministic artifact records, timing confidence report, and rejected mechanisms. Independent durable rebuilds under `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-m64-k-pipeline-sia5-independent-a/` and `...-independent-b/` match for R35, R257, and R12,288 source, object, and HSACO hashes and inspections. Remaining items are classified as retained and measured (the final M64 SIA5 K pipeline at B1 and M128 direct-pointer/swizzle8 at B4/B16), rejected by timing or resource/identity gates (the listed alternatives), or deferred pending public integration review. No actionable in-contract mechanism remains in the current typed lowering surface. The complete `tests/ggtensile` suite passes (`698 passed`); Ruff, compileall, `git diff --check`, and the complete diff review are also complete.

### Continued campaign: current M128 activation-prefetch qualification

The historical staged M128 search was rebuilt from the current writer into `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-m128-current/` for the production B4 and B16 rows. The current source is intentionally different from the historical source because the retained shared IQ2_XXS decode now batches four codebook-derived values and overlaps the paired codebook preparation. Current R49,152 and R196,608 `prefetch-a` artifacts are nevertheless identical in resource envelope: `153` VGPR, `41` SGPR, `10,240 B` LDS, `64` WMMAs, `5` barriers, zero private bytes, and zero VGPR/SGPR spills. The current inspection also reports `169` VMEM operations, `201` LDS operations, and `92` VOPD issues.

Every current candidate/control comparison passed before timing for all ten learned/hash search medoids at both production geometries. The protocol was three warmups, nine rotating repeats, equivalent output allocation in complete-call timing, and a preallocated kernel-only diagnostic. The current M128 prefetch-a search was faster than HIP at both large keys, but remained a finalist until the later disjoint swizzle8 confirmation.

This search result was a strong M128 finalist and is superseded by the disjoint confirmation recorded in the `Final Accepted Performance` summary above. B1 remains assigned to the retained M64 SIA5 pipeline, whose independent and paired confirmations are both faster than installed HIP. The durable search report is `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-m128-current/timing-search9-current.json`.

### Continued campaign: M128 direct second-projection pointers

The activation-prefetch body was changed only to use the already-derived second gradient and packed-bank pointers directly, eliminating projection pointer swaps while preserving M128/N64/K32, PGR2/SIA4, swizzle4, concurrent reads, and the paired codebook overlap. Fresh R49,152 and R196,608 artifacts remain `153` VGPR, `41` SGPR, `10,240 B` LDS, zero private bytes, and zero spills. Every search-medoid output was BF16 bit-exact to installed HIP.

In a common three-warmup, nine-repeat rotating screen, direct pointers measured `72.5805 ms` at B4 and `290.9745 ms` at B16, versus `72.8630 ms` and `293.3981 ms` for the current activation-prefetch parent. That is a `1.0039x` parent speedup at B4 and `1.0083x` at B16. Relative to installed HIP (`98.8737 ms` and `379.7460 ms`), direct pointers measured `1.3623x` and `1.3051x`. The mechanism is retained as the next M128 baseline; confirmation is deferred until higher-confidence LDS and K-pipeline variants are screened. Artifacts are under `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-m128-direct-current/`; the timing report is `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-m128-current/timing-direct-search9.json`.

### Continued campaign: M128 swizzle8 promotion

Changing only M128's decoded-weight XOR chunk from swizzle4 to swizzle8 widened paired LDS reads while retaining activation prefetch and direct second pointers. Strict inspection improved from `153` to `149` VGPR, from `201` to `137` LDS operations, and from `1,240` to `1,164` VALU issues. The artifact remains `41` SGPR, `10,240 B` LDS, `64` WMMAs, `5` barriers, zero private bytes, and zero spills. All search-medoid results are BF16 bit-exact to installed HIP.

The common nine-repeat screen measured `69.8231 ms` at B4 and `285.7568 ms` at B16, versus `71.7719 ms` and `290.5035 ms` for swizzle4 direct pointers. Swizzle8 therefore improves its parent by `1.0279x` and `1.0166x`; relative to installed HIP (`98.9624 ms` and `376.2326 ms`) it reaches `1.4173x` and `1.3166x`. The exact typed identity is `iq2_xxs_m128_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a()`. Artifacts and timing are under `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-m128-direct-swizzle8/`.

### Continued campaign: M128 SIA5 K-pipeline rejection

The M64 SIA5/K-pipeline identity was ported exactly to M128 on top of swizzle8 and direct pointers: M128/N64/K32, `ScheduleIterAlg=5`, current and next packed-weight prefetch, paired codebook overlap, and geometry-derived activation waits. The generated artifacts were exact to HIP on every timed medoid and remained resource-clean at `149` VGPR, `41` SGPR, `10,240 B` LDS, `64` WMMAs, `7` barriers, zero private bytes, and zero spills. Static counts were `193` VMEM, `209` LDS, `1,728` VALU issues, and `108` VOPD issues.

The nine-repeat fitted-prior screen rejected the port. Relative to the swizzle8/direct-pointer parent, weighted complete-call latency changed from `68.9036 ms` to `75.0698 ms` at B4 (`0.9178x`) and from `276.0649 ms` to `367.5013 ms` at B16 (`0.7512x`). It still measured `1.3131x` and `1.0198x` against installed HIP, but that is materially worse than the retained parent and does not justify the added control-flow and VMEM work. The M128 SIA5 identity is therefore failed by timing and was removed from the typed model and capability admission. Artifacts and timing are under `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-m128-sia5-pipeline/`.

### Continued campaign: M128 SIA4 next-weight rejection

The narrower K-pipeline screen retained SIA4 and changed only `prefetch_packed_weight_next=True`. Its first build exposed a shared lowering defect: the second-projection WMMA schedule encoded `vmcnt(14)` from the historical M64 issue frontier even though M128 has only eight activation loads in flight. It therefore consumed incomplete activation fragments and produced `194,101,544` mismatches at B4. Replacing that literal with the existing geometry-derived `2 * m_tiles` frontier emits `vmcnt(4)` for M128 and preserves the geometry-correct M64 frontier. The rebuilt candidate is BF16 bit-exact on every timed medoid and remains resource-clean at `149` VGPR, `41` SGPR, `10,240 B` LDS, `64` WMMAs, `7` barriers, zero private bytes, and zero spills.

Correctness did not rescue performance. In the corrected nine-repeat fitted-prior screen, weighted complete-call latency changed from `68.9298 ms` for the swizzle8/direct-pointer parent to `71.1020 ms` at B4, a `3.15%` regression, and from `285.3708 ms` to `322.8013 ms` at B16, a `13.12%` regression. The candidate still beats installed HIP by `1.3858x` and `1.1662x`, but it is materially worse than its parent. The SIA4 next-weight identity was rejected by timing and removed from the typed model and capability admission. The geometry-derived wait fix and permanent M128 assertion are retained as a shared correctness repair. Artifacts and timing are under `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-m128-sia4-next-weight/`.

### Continued campaign: M128 LDS swizzle variants

Two remaining typed LDS-layout values were screened on the retained M128 direct-pointer/swizzle8 identity. Swizzle16 passed structural validation and predicted the same `149` VGPR, `41` SGPR, and `10,240 B` LDS envelope, but the gfx1151 assembler rejected the generated paired `v_dual_mul_f32` instructions with repeated `src0 operands must use different VGPR banks` errors at the first production build. No code object, correctness result, or timing claim exists for swizzle16, and it was not added to source admission.

Swizzle0 assembled for both production rows and was exact against installed HIP, including deterministic reruns, over `201,326,592` B4 and `805,306,368` B16 BF16 elements. Inspection reports `145` VGPR, `41` SGPR, `10,240 B` LDS, `64` WMMAs, `5` barriers, `1,120` static VALU issues, zero private bytes, and zero spills. The 25-repeat reverse-rotating confirmation used the confirmation profile bank, five warmups, equivalent output allocation, and pre-timing exactness checks. Weighted complete-call medians were:

| Key | Installed HIP | Swizzle8 parent | Swizzle0 | Swizzle0 vs swizzle8 |
| --- | ---: | ---: | ---: | ---: |
| B4 / R49,152 | `97.9710 ms` | `70.3040 ms` | `83.0214 ms` | `18.11%` slower |
| B16 / R196,608 | `374.7943 ms` | `282.4092 ms` | `313.5470 ms` | `11.06%` slower |

Per-key independent and paired robust 95% confidence intervals both exclude zero for the swizzle0 regression: B4 independent `+17.82%..+18.39%`, paired `+17.84%..+18.40%`; B16 independent `+10.22%..+11.91%`, paired `+10.25%..+11.93%`. Swizzle0 remains faster than installed HIP, but it is decisively slower than the retained swizzle8 parent and was rejected. The swizzle16 assembler failure and swizzle0 records are under `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-iq2-xxs-m128-swizzle-variants/`, with corrected per-key confidence in `timing-confirm25-confidence-by-batch.json`. The accepted M128 swizzle8 confirmation values are the B4/B16 rows in the top-level table; no actionable in-contract M128 identity remains after this screen.
