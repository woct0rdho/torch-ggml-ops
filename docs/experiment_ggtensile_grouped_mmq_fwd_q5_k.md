# GGTensile Grouped MMQ Forward Q5_K Experiment

## Purpose

Build and optimize isolated gfx1151 wave32 GGTensile grouped Q5_K forward kernels for the Qwen routed-down production shapes `(16384,2048,512)`, `(65536,2048,512)`, and `(262144,2048,512)`. The fitted Qwen route prior is the primary speed metric. Correctness, resources, deterministic generation, and complete-call timing remain mandatory gates.

This experiment is research-only. It does not change public dispatch, generated bundle tables, extension registration, packaging, or the HIP fallback. The grouped Q4_K selected sources are a frozen regression contract while shared decoded-LDS mechanisms are reused.

## Contract

The packed expert bank is `[256,2048,352]` bytes per logical row. Q5_K has 256 values per 176-byte block, so `K=512` stores two blocks per output row. Each block contains FP16 `d` and `dmin`, twelve packed scale/minimum bytes, a 32-byte high-bit plane, and a 128-byte low-nibble plane. The generated kernel consumes this packed representation directly.

The activation workspace is the installed Q8_1 `F16_D4S4` representation with shape `[4,R,144]`. Output is contiguous BF16 `[R,2048]`. Route metadata remains device-resident: int64 physical expert IDs and cumulative int32 row offsets, both with at most 256 entries. The final valid offset is `R`.

The research ABI is the existing grouped 64-byte multiply contract: packed weights, Q8_1 activations, output, expert IDs, expert offsets, physical expert count, output rows per expert, aggregate activation rows, blocks per packed row, and bytes per expert. No host route readback or per-route descriptor construction is permitted.

Every retained artifact must target gfx1151 code-object version 5 and wave32, have zero private storage, spills, scratch instructions, calls, and dynamic stack, and rebuild byte-identically. Any companion setup, workspace, atomics, split-K, or reduction mechanism must expose and time its full contract. Prepared weights, dense shadows, producer fusion, public selector changes, and online tuning are outside this campaign.

## Installed Controls

The installed bundle provides exact `N=2048`, `K=512` Q5_K serial J64 and J32 controls:
- `grouped_fwd_serial_q5_k_n2048_k512_j64`
- `grouped_fwd_serial_q5_k_n2048_k512_j32`

The public policy selects J32 when `R < 128 * G` and J64 otherwise. The research launcher must reproduce that policy without reading route contents on the host. Both controls launch a `32 x G` grid with a 128-thread workgroup; J64 uses 28,928 dynamic LDS bytes and J32 uses 24,192.

## Initial Premise

Q5_K shares Q4_K scale/minimum arithmetic, Q8_1 activation staging, signed integer WMMA, correction, and BF16 stores. Its distinct cost is high-bit reconstruction. The shared typed decoded-LDS emitter already models this distinction: it loads `qh`, shifts lane-selected bit masks, merges bit 4 into both low-nibble halves, and releases the high-bit payload before the WMMA loop.

The first grouped control uses serial routed ownership with a 128-row by 64-column decoded tile. It deliberately starts from the resource-qualified shared body rather than copying installed HIP assembly. The first schedule controls are serialized metadata, independent metadata extraction after low WMMA, and the resource-neutral `a1d2-p2` epilogue used by the dense Q5 shared-down family.

If the 128-row control is not competitive, the first changed premise is the Q4-qualified true 64-row layout. Q5 high-bit reconstruction increases decode pressure without extending the multiply body, so reduced row ownership may improve residency while preserving decoded-weight reuse. A mixed 32-row tail branch is eligible only after its Q5 path is independently correct and the fitted prior shows enough tail incidence to repay the extra static body.

## Qualification

Before timing:
- Validate strict problem and solution mappings and reject cross-format identities.
- Build all three aggregate-row keys twice and compare source and code objects byte-for-byte.
- Inspect ABI, symbol, target, wave size, workgroup, VGPRs, SGPRs, LDS, barriers, waits, WMMAs, private bytes, spills, scratch, calls, and dynamic stack.
- Compare full, partial, boundary, sparse-ID, repeated-ID, and invalid routes against installed HIP.
- Require input, active packed-weight, and Q8_1 workspace mutation sensitivity; inactive expert mutations must remain inert.
- Compare against an independently dequantized BF16 grouped reference. Unchanged arithmetic order should remain bitwise equal to installed HIP.
- Regenerate every retained grouped Q4_K identity and require byte-identical source.

The fitted Qwen prior uses deterministic 512-draw confirmation and search banks, five weighted medoids, three warmups, and nine alternating-order event repeats for screens. A competitive candidate receives reversed-order 25-repeat confirmation. Search-bank, captured, synthetic, and sequential controls follow only after confirmation-prior competitiveness. Complete-call timing uses the fixed HIP quantizer and one allocated shared workspace per HIP/GGTensile pair.

## Mechanism Order

- Establish the typed 128-row Q5 decoded-LDS correctness and installed J64/J32 timing controls.
- Measure serialized, metadata-scheduled, and `a1d2-p2` 128-row bodies on the confirmation prior.
- If the resource envelope loses materially, port the true 64-row register/LDS layout and remeasure before local schedule tuning.
- Add a 32-row tail body only when the 64-row parent is competitive and route-tail incidence supplies material expected margin.
- Investigate Q5-specific high-bit load width, merge schedule, and dependency overlap only when a lower bound or ISA comparison identifies recoverable work.
- Consider flattened mapping, persistence, or split-K only with a measured ownership or K-loop bottleneck and an explicit complete mechanism contract.

## Experiment Log

### Campaign opened

The grouped Q4_K result, completed dense Q5_K forward record, grouped HIP optimization history, Q5_K packed semantics, installed bundle selection, and shared decoded-LDS emitter were audited. The exact grouped bank is `[256,2048,352]`, the Q8_1 workspace is `[4,R,144]`, and all three production aggregate-row keys retain the same compact routed ABI as Q4_K.

A strict Q5 problem identity, typed decoded solution identities, and research-only installed J64/J32 launchers were added without changing public dispatch. The generated source contains explicit Q5 high-bit and low-nibble address formation and merge instructions. The first 128-row artifact assembles at 239 VGPRs, 40 SGPRs, 38,400 bytes of LDS, 32 static WMMAs, four barriers, and zero private storage or spills.

### Correctness and installed controls

The installed J32 research launcher initially used an incorrect `64 x G` grid by treating `J32` as output-column ownership. The bundle source shows that `J` is routed-row ownership while output columns remain fixed at 64, so both J32 and J64 use `grid.x=2048/64=32`. Correcting the J32 grid made the first R35 route bitwise equal to installed HIP; no generated device code ran during the incorrect comparison.

The R35 route contains 1-, 15-, 16-, and 3-row groups. The generated Q5 output is bitwise equal to installed J32, finite, and has maximum absolute error `0.009765625` and RMS error `0.00229125` against the independently dequantized BF16 grouped reference. Uniform, skewed, sparse-ID, boundary, and repeated-ID routes at all three production row counts are also bitwise exact. Active packed-weight, input, and Q8_1 workspace mutations change output; an inactive-expert mutation is inert; an invalid expert route leaves the destination untouched.

All 30 grouped Q4_K production sources, spanning ten solution identities and three row counts, remain byte-identical to the frozen Q4 baseline. Q5 source and code objects rebuild byte-identically.

### 128-row control rejected as a broad selector

Serialized, metadata-scheduled, and `a1d2-p2` 128-row controls all use 239 VGPRs and 38,400 bytes of LDS. Weighted confirmation-prior installed-over-candidate ratios for the serialized body were `0.650x`, `0.954x`, and `0.945x` at B1/B4/B16. Independent metadata extraction after low WMMA improved those to `0.679x`, `0.987x`, and `0.969x`; adding `a1d2-p2` reached `0.683x`, `0.995x`, and `0.981x`. Scheduling recovers part of the deficit, but B1 remains dominated by the installed J32 ownership and B16 remains below the promotion margin.

### True 64-row and mixed 32-row tails

The Q4-qualified true 64-row layout transfers to Q5 without a new register plan. It uses 159 VGPRs, 40 SGPRs, 29,184 bytes of LDS, 16 static WMMAs, and four barriers. The scheduled `a1d4-p2` form reached `0.910x`, `1.010x`, and `0.946x` at B1/B4/B16. Adding the mutually exclusive 32-row tail body raises the static WMMA count to 24 while retaining the same resources. Its first screen reached `1.000x` at B1 and `1.036x` at B4; B16 remained approximately `0.946x`, confirming that large groups still need 128-row reuse.

A true 32-row decoded layout was also assembled and qualified before rejection. Its dedicated two-fragment plan reduced resources to 135 VGPRs and 24,576 bytes of LDS, with eight static WMMAs and bitwise-exact output. B1 nevertheless fell to `0.9872x` weighted, with four medoids at `0.896-0.916x`. Lower residency cannot repay decoding each Q5 high-bit payload twice as often. The 32-row parent mechanism was reverted; only the shorter residual body inside the 64-row parent is retained.

Dense Q5's resource-neutral VOPD accumulator initialization was transferred as a final schedule control. It remained exact and changed scalar-parent weighted timing by `1.0023x`, `0.9920x`, and `1.0019x` at B1/B4/B16. The sub-percent movements are inconsistent and the material B4 key regresses, so the grouped VOPD path was reverted.

The selected B1 64/32 `a1d4-p2` body brackets weighted parity in two reversed-order 25-repeat confirmation passes at `1.0020x` and `0.9987x`. The independent search bank is `1.0007x` and the captured corpus is `1.0154x`. This is not universal: low-weight confirmation and captured medoids regress by as much as about 4-6%, and the uniform synthetic route is `0.955x`. The identity is retained as a fitted-prior research control, not as a public or universally dominant selector.

### Three-way 128/64/32 ownership

A Q5-specific three-way final-tile policy emits mutually exclusive 128-, 64-, and 32-row activation, MMA, correction, and store bodies. Full routed tiles preserve 128-row decoded reuse; residuals at most 64 or 32 rows take shorter paths. Both scalar predicates are converged before common barriers. The artifact remains at 239 VGPRs, 40 SGPRs, and 38,400 bytes of LDS with 56 static WMMAs and four barriers. A 388-row route containing 17-, 48-, 128-, and 195-row groups exercises all three branches and matches installed HIP bitwise while remaining within the independent-reference envelope.

This mechanism is the material B4 result. The first nine-repeat confirmation screen reached `1.0636x`; two independent 25-repeat confirmations reached `1.0597x` and `1.0770x`, with every medoid at least `1.0274x` and `1.0360x`. The independent search bank is `1.0531x`, the captured corpus is `1.0557x` with every medoid faster, and candidate-first/HIP-first sequential 25-repeat controls are `1.0701x` and `1.0655x`. Synthetic skewed, sparse, and boundary routes win; the uniform route improves from the 64/32 body's `0.912x` to `0.978x`, so the former cliff is mostly but not completely removed.

At B16, the same `a1d4-p2` identity reaches `1.0015x` and `1.0039x` in the two 25-repeat confirmations, with every confirmation medoid above parity. Search is `0.9966x`, captured is `0.9998x`, and the four synthetic controls aggregate to `1.0026x`. B16 is therefore retained as a resource-clean parity control, not claimed as a material speed selector.

### Complete-call audit

A final reversed-order 25-repeat complete-call audit used the fixed HIP Q8_1 F16_D4S4 producer, one allocated shared workspace per pair, sustained short-key batches, and fresh quantization before each multiply. Weighted installed-over-candidate ratios are `1.0539x` at B1, `1.0813x` at B4, and `1.0070x` at B16. Every output remains bitwise exact. Producer inclusion preserves the B4 gain and creates no complete-call parity failure.

### Final prequantized multiply-only performance

The final table reports prequantized multiply-only throughput. HIP and GGTensile consume the same activation workspace produced by the shared HIP quantizer, so activation quantization and other complete-call work are excluded. Nominal dense-equivalent throughput is `2 * aggregate rows * N * K / time`, doubled for paired two-projection kernels. GGTensile/HIP speedup is HIP body time divided by GGTensile body time.

| Aggregate rows | Public catalog hash | HIP TFLOPS | GGTensile TFLOPS | GGTensile/HIP speedup |
| ---: | :--- | ---: | ---: | ---: |
| 16,384 | `ggsol_a35d42962fcb240d` | 15.302 | 15.283 | 0.9987x |
| 65,536 | `ggsol_38e2fe81d0ce5aea` | 21.068 | 22.691 | 1.0770x |
| 262,144 | `ggsol_a9d33673184bf5ae` | 25.677 | 25.776 | 1.0039x |

The selected entries passed the retained route-correctness, resource, and deterministic-build checks.

### Reopened routed prologue and address micro-experiments

Q5_K shares the grouped route emitter and decoded BF16 output mapping with Q4_K, so the reopened R1 packed-kernarg, R2 paired cumulative-offset, and R3 guarded-stride/output-address mechanisms apply without changing Q5 high-bit reconstruction or arithmetic order. The Q4_K B1 five-medoid screen is the transfer gate because it has the shorter decode body and therefore bounds the plausible relative benefit of common route setup. A mechanism advances to Q5_K production-shape qualification only if that shared screen clears two percent while preserving the exact route contract, deterministic build, resources, and installed-control agreement.

The shared build and correctness matrix included Q5_K R35 and all three production keys. R1-R3 rebuilt byte-identically in two independent passes, retained the selected parent's VGPR, SGPR, LDS, WMMA, barrier, wait, private-storage, and spill resources, and matched the parent bitwise on first, odd/even non-first, repeated-ID, boundary, skewed, mutation, invalid-expert, and invalid-offset routes.

The Q4_K B1 transfer screen measured R1 at +0.030% candidate time relative to its parent, R2 at -0.086%, and R3 at -0.083%. None cleared two percent. Because Q5_K performs the same route setup around a longer high-bit decode body, the prespecified upper-bound gate rejects production timing transfer. R1-R3 are closed as neutral shared micro-optimizations, not retained Q5_K mechanisms. No Q5 solution identity, selected artifact, or public integration changed.

### Post-review disposition: zero-bank initialization

The later paired-source audit does not transfer a zero-bank hoisting candidate to Q5_K. The decoded-LDS Q5_K body already initializes its zero bank once per row tile, and grouped VOPD accumulator initialization was separately measured and rejected. The Q5 route/address transfer closure and representation-level bottleneck classification therefore remain unchanged; no new Q5 local accumulator experiment is pending.

## Recursive Final Review

A fresh pass reread the grouped Q4 record, dense Q5 forward and backward records, grouped forward and backward histories, installed HIP ownership and normalized ISA, final generated artifacts and reports, the TensileLite and Composable Kernel grouped mechanisms, and the gfx1151 instruction constraints. The resulting classification is:
- Retained and measured: shared typed Q5 high-bit reconstruction; independent metadata extraction after low WMMA; the 64-row parent with a 32-row residual at B1; the three-way 128/64/32 final-tile body at B4/B16; `a1d4-p2`; converged masked stores; and serial route ownership with 32 independent output-column workgroups per route.
- Rejected or superseded by timing/resources: serialized and scheduled 128-row parents, the unqualified 64-row parent at B1/B16, two-way 128/64 ownership, true 32-row ownership, alternate `a1d2-p2` schedules, grouped VOPD accumulator initialization, the earlier body without the final residual split, and the shared R1-R3 route/address micro-optimizations. Dense Q5 evidence also closes high-plane lane sharing, alternate merge forms/orders, payload prefetch, loop rolling/unrolling, larger buffers, and compact consumer-decode LDS layouts; the grouped body already consumes that qualified shared decoder.
- Deferred behind an explicit changed prerequisite: flattened or persistent routing requires a profile showing launch imbalance after the existing `32 x G` output-column parallelism; SplitK/StreamK requires enough K-loop underutilization to repay FP32 partial-output storage and reduction at fixed `K=512`; a prepared or persistent decoded representation requires model-owned lifetime, invalidation, memory, and forward/backward amortization; approximate accumulation requires a new bitwise-acceptable arithmetic contract; and further instruction scheduling requires a compiler, ISA, or hardware change that alters the measured high-bit operation floor.
- Contract-incompatible in this campaign: host route readback, hidden caches, dense weight shadows, producer fusion, online tuning, public selector changes, generated-bundle changes, extension registration, and packaging changes.

The remaining B1 profile sensitivity is decode-reuse versus J32 ownership, not missing occupancy: true 32-row ownership reduced resources but lost materially. B16 movement remains at parity scale across independent banks. B4 is the only broad material result, and its gain survives both sequential launch orders and complete-call timing. No remaining in-contract mechanism has an unmeasured first-order path with material expected margin.

The final selected keys are bitwise exact over the five-route production matrix, mutation-correct, independently referenced, resource-clean, and deterministic. All 30 frozen grouped Q4 sources remain byte-identical to the frozen baseline. The recursive review therefore closes with no public integration authorized.

## Post-Audit B16 Ownership Reopening

Status: planned and unmeasured. Existing high-bit merge, compact consumer-decode, true-32-row, route-prologue, and epilogue conclusions remain closed.

Add a Q5-specific route-persistent full-K identity. Decode both K512 packed blocks once per route/output-column workgroup into two immutable LDS weight images, keep activation storage disjoint, and reuse the images across the existing serial row loop and three-way 128/64/32 row dispatch. This targets repeated Q5 high-bit reconstruction rather than moving it to consumers. The complete LDS/resource formula and high-bit ownership must be derived independently; a Q4 implementation or timing result cannot select or reject Q5 automatically.

The shared decoded layout now fixes the first resource discriminator. A 128-row activation stage occupies `512 + 128 * 144 = 18,944` bytes including the prefix, and each decoded image occupies `64 * 304 = 19,456` bytes, so the two-image total is 57,856 bytes. That is exactly the LDS size of the measured Q4 G4 body, whose strongest uniform B16 case still regressed about 12%. Q5's additional high-bit reconstruction leaves a format-specific decode-saving premise, but it must quantitatively explain how that work can repay the analogous large-LDS deficit before emission. This lowers the experiment's priority without treating Q4 timing as a Q5 result.

Screen B16 first against the selected three-way `a1d4-p2` parent, including fitted route tile counts, body and complete-call timing, and exact decode work removed. Advance only after a stable greater-than-two-percent complete-call gain, then consider B4. Direct packed inputs, route and tail behavior, exact arithmetic, mutations, reproducible artifacts, zero private storage, and spill freedom remain hard gates. This exact workgroup-local representation requires no model integration.

If Q4 qualifies generated wait/liveness analysis, re-derive it for the Q5 high-bit images and require identity mode to reproduce the Q5 parent before changing a wait or barrier. Separately, an approximate output-only experiment may compare `RNEPreserveNaN`, `BiasRound`, and `Truncate`; finite-output checks live in numerical tests, not in kernel branches, and model integration is required before retaining that numerical policy. Do not compose synchronization, persistent images, and output relaxation before isolated results exist.

Screen the resource-neutral output policy before implementing route persistence. Generated synchronization remains exact shared infrastructure with a low-single-digit performance prior; neither priority statement is a timing result.

## Current One-Law Retuning and Reopening Flags

This annotation records the current direct-kernel benchmark only; it does not change the selected Q5_K identities, catalog, dispatch, or integration status. The primary reports are under `~/tmp/torch-ggml-ops/fresh-grouped-fwd-one-law-multiply-20260823/`, with a 75-repeat top-up under `~/tmp/torch-ggml-ops/fresh-grouped-fwd-one-law-multiply-topup-20260823/`.

### Retune flag

- Q5_K B1, `ggsol_a35d42962fcb240d` (`R=16,384`) may need further tuning or retuning. The current `qwen-learned` realization measured `0.9529x` GGTensile/HIP, with a 75-repeat log-time interval of `[0.9457x, 0.9602x]`, versus the documented `0.9987x`. This is a stable current short-row regression, not a sample-count ambiguity.
- Q5_K B4 and B16 do not receive a retune flag from this pass: their current speedups are `1.0635x` and `0.9990x`, respectively. Their differences from the documented rows are smaller and do not identify a separate shape-wide failure.

### Reopen flag

- Reopen the Q5-specific B1 R1-R3 route/address transfer experiment. Its historical Q5 closure was inherited from the Q4 B1 transfer gate rather than from Q5 direct-kernel timing under the current one-law, prequantized multiply-only protocol. Re-run R1-R3 against the current Q5 B1 parent before treating the shared route/address transformations as closed for Q5. This is a requalification request, not evidence that any R1-R3 variant should be retained.
- The Q5 route-persistent full-K image remains a planned, format-specific experiment. The current table does not reopen it or transfer the Q4 G4 rejection; it needs its own LDS/resource and timing discriminator if pursued.
