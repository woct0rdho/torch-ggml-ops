# GGTensile Assembly-Writer Refactor Plan

## Purpose

This document records the completed first MMQ forward assembly-writer refactor and the follow-up work reopened by a fresh recursive review. The durable contracts and design principles remain authoritative in `ggtensile_plan.md`; format-specific arithmetic, timing, and rejected mechanisms remain in the experiment records. Phases 0-7 below are historical evidence, not a waiver for findings discovered by a later review.

The current research catalogs now provide broader evidence than the original work order. Q3_K selects the typed full-weight mechanism on all 12 forward research keys. Q8_0 has 23 exact research decisions: 20 compact depth-32 rows, one ordinary `HipTile` mapping, and two exact small-M mappings. Q6_K selects the typed row/role wavefront on all three keys. These results preserve the need for distinct packed-three-bit, packed scale/minimum, structured-Q6, and signed-int8 mechanisms, while exposing narrower duplicated authorities and compatibility gates that the first refactor did not remove.

This follow-up explicitly opens the separate backward structural project deferred by the first plan. It does not authorize a backward algorithm change: the current five quant decoders, decoded-B pipeline, BF16 WMMA body, store order, and selected source identities remain controls until a deliberate stream change passes the full gates.

This is a structural project. Its target is clearer ownership, single derivation of physical facts, typed component boundaries, formula-based capability, and a smaller blast radius for later format work. File count and line count are not success metrics, but boilerplate that only renames one access or forwards one call must not survive without a concrete invariant or semantic boundary.

## Fixed Boundaries

The refactor must preserve these project boundaries:
- `ForwardKernelWriterAssembly` remains the public forward assembly-writer entry point with the existing constructor, `source()`, and `write()` behavior.
- `BackwardKernelWriterAssembly` remains the public backward assembly-writer entry point with the existing constructor, diagnostics, `source()`, and `write()` behavior.
- `ProblemType`, `ProblemSize`, `ForwardSolution`, `SolutionKey`, `ForwardProblemContract`, and `ForwardKernelSpec` remain the generation and identity boundary.
- Capability validation remains formula-based. Exact inventories and selected catalogs remain the only selection authority.
- A lowering mechanism may dispatch on validated mechanism identity such as `operand_source`; it may not inspect tensor names, exact-key catalogs, measured winners, or benchmark artifacts.
- gfx1151, code-object v5, wave32, WMMA V1, the 40-byte MMQ ABI, Q8_1 workspace contracts, arithmetic order, FP32 accumulation where selected, BF16 RNE output, and zero private storage remain unchanged.
- Public runtime dispatch and the current 179-kernel bundle remain unchanged. Public integration is a separate review.
- The 447 preserved Q8_0 forward sources remain byte-identical unless an instruction-stream change is deliberately separated and fully qualified.
- Backward physical planning and lowerer extraction are in scope. Forward and backward algorithms remain separate, and changes to direction-neutral assembly primitives must preserve all affected sources and tests.

## Baseline Responsibility Map

Before the first refactor, the approximately 6,500-line `kernel_writer_assembly_mmq_fwd.py` combined four different responsibilities: the public source envelope, lowering-family dispatch, mechanism-specific physical plans, and final instruction emission. It also contained older direct and decoded paths beside the structured Q6, Q8, and Q3 implementations. The size itself was not the defect; the defect was that mechanism ownership and shared ownership were not explicit at the module boundary. The table is the historical Phase 0 map; the reopened map adds the current Q3 full-weight and Q8 compact mechanisms.

The baseline families provided the following evidence:

| Mechanism | Baseline domain | Evidence that must be preserved | Physical behavior that remains mechanism-owned |
| --- | --- | --- | --- |
| `Global` | direct Q4_K control | retained generated controls and tests | one-wave global payload reads, direct nibble decode, scale/minimum correction, scalar output traversal |
| `DecodedWeightLdsBatch8` | shared Q4_K/Q5_K | selected and catalog-valid Q4_K/Q5_K sources | decoded-weight LDS rows, F16_D4S4 activation staging, metadata scheduling, clamp behavior, pipelined epilogue |
| `Q6StructuredDecoded` | selected Q6_K | structured semantic plans, selected sources, resource and timing qualification | single-row/dual-row ownership, near/far reads, signed Q6 decode, refill traversal, explicit dependency delays |
| `Q3HipTiledLds` | one isolated Q3_K control | exact HIP/public agreement, independent reference, mutation gates, deterministic rebuilds, faster multiply | 110-byte multi-plane decode, half-block staging, signed six-bit group scales, loop-carried weight prefetch |
| `Q3FullWeightTiledLds` | 12 selected Q3_K research keys | exact output, independent reference, mutation, deterministic rebuilds, zero spills, repeated HIP confirmation | full 256-value decode, 336-byte padded rows, invariant scale bases, four-barrier traversal |
| `Q8DirectGlobal` and `Q8RegisterTiled` | Q8_0 research controls | accepted candidate generation and writer coverage | direct/register ownership with no shared tiled-LDS pipeline |
| `Q8HipTiledLds` | one selected Q-A M2048 key plus retained controls | exact output, mutation, deterministic rebuild, zero spills, noisy-measurement qualification | 128x64 wave-N ownership, 304-byte ordinary rows, depth-32/depth-64 traversal |
| `CompactDepth32WeightRows` | 20 selected Q8_0 research keys across ordinary and LM-head families | exact output, mutation, deterministic rebuild, zero spills, repeated parent/HIP confirmation | 144-byte rows, weight-first stage order, paired scale reads, activation rows 32/64/128 |
| `Q8SmallMTiledLds` | LM-head M32/M64 controls | two exact selected controls and frozen-source evidence | exact small-M ownership, 32/64 activation rows, 304-byte weight rows |

This map rules out one universal physical pipeline. Q3_K, Q6_K, and Q8_0 all consume F32_D4 activation workspaces, but they do not share packed-weight layout, decode ownership, reduction depth, LDS row shape, or correction arithmetic. Q4_K/Q5_K share scale/minimum semantics and a decoded-LDS mechanism. The Q8 tiled variants share substantial staging, integer-dot, and epilogue behavior while retaining distinct layouts and wait schedules. Those are the initial convergence boundaries.

## Target Architecture

### Public facade

`kernel_writer_assembly_mmq_fwd.py` becomes a thin facade that owns:
- validation of one exact `SolutionKey`.
- construction of the immutable forward context and derived physical plan.
- ROCISA initialization and the code-object-v5 signature.
- the fixed MMQ kernarg description and kernel description.
- closed dispatch to exactly one validated lowering mechanism.
- source writing and error translation at the public boundary.

The facade must not own quant-format register constants, LDS offsets, decode loops, WMMA loops, waits, barriers, or epilogues. It must not provide a generic fallback body for an unknown mechanism.

### Semantic and physical state

`mmq_fwd_spec.py` continues to own fixed format semantics and complete candidate choices. Physical facts currently split between `mmq_fwd_spec.py` and the writer must be reorganized beneath one pure, non-ROCISA planning boundary.

The target physical state is a closed tagged union of concrete mechanism plans. Each plan derives, once:
- launch and wave ownership from `ForwardKernelSpec`.
- packed-weight, activation, and output strides from the exact problem and contract.
- the concrete LDS layout and all plane offsets.
- typed register roles, widths, alignments, lifetimes, assignments, and reusable pools.
- accumulator and loop counts.
- producer and first-use dependency facts used to derive waits.
- formula-derived VGPR, SGPR, LDS, private-segment, and spill usage.

The plan contains semantic coordinates and derived ownership, not assembly text, opcodes, issue slots, rank lists, or a stored instruction schedule. Resource admission and instruction emission must consume the same physical plan. Separate arithmetic that predicts a register count already established by a register plan must be deleted. In particular, the Q3_K and Q8_0 resource formulas must no longer duplicate register-plan widths or hard-coded layout totals.

`DerivedForwardState` may move to the pure physical-planning boundary if needed to maintain an acyclic import graph. Stable public records remain in their current modules; temporary re-exports or compatibility writers must not remain after the migration phase.

### Typed assembly operands

Direction-neutral operand types that are already proven by Q6 should move into `kernel_writer_assembly.py` only when they are genuinely ISA-level concepts. The expected shared set is typed VGPR, SGPR, VGPR range, half-register, and immediate operands plus deterministic register-role projection. Q6-specific ownership, address recurrence, and dependency-delay records remain Q6-owned unless another mechanism uses the same invariant.

Q3_K and Q8_0 helper boundaries must stop passing unrelated raw integer register numbers. Their lowerers should receive concrete register-role records or typed operands and layouts. Final string formatting remains at the `Assembly.inst()` boundary; this refactor does not introduce a general instruction IR.

### Mechanism lowerers

Use one immutable forward-lowering context, concrete format-family lowerers, and short closed dispatch. A common lowerer protocol is unnecessary while no production consumer requires one. Intended ownership is:

| Module | Ownership |
| --- | --- |
| `kernel_writer_assembly_mmq_fwd.py` | public facade, source envelope, signature, closed dispatch |
| `mmq_fwd_physical.py` | pure physical-plan union, shared role primitives, resource authority, no ROCISA or instruction emission |
| `mmq_fwd_lowering.py` | immutable lowering context and forward writer error; no unused protocol or forwarding registry |
| `mmq_fwd_lowering_packed_3bit.py` | `Q3HipTiledLds` body and Q3-specific stage emitters |
| `mmq_fwd_lowering_packed_direct.py` | direct packed scale/minimum body |
| `mmq_fwd_lowering_decoded_lds.py` | decoded-weight LDS, F16_D4S4 activation staging, and scaled integer MMA |
| `mmq_fwd_lowering_metadata.py` | packed scale/minimum metadata reconstruction shared by compatible mechanisms |
| `mmq_fwd_lowering_mma.py` | one source-identical signed-int8 WMMA constructor shared by compatible mechanisms |
| `mmq_fwd_lowering_q6.py` | structured Q6 semantic emitters, stateful schedule operations, and final Q6 body |
| `mmq_fwd_lowering_signed_i8.py` | signed-int8 direct, register-tiled, wave-N tiled, small-M, and compact-KV bodies |

The exact filenames may change to keep imports acyclic, but these ownership boundaries may not collapse back into the facade. A format module is a lowerer, not a sibling public writer: it does not initialize ROCISA, emit a signature, own file output, revalidate a `SolutionKey`, or forward to another writer.

Each concrete lowerer owns its orchestration. There is no required universal `Setup -> GlobalRead -> Decode -> LDS -> Dot -> Epilogue` runner. Shared semantic components are called explicitly by mechanisms whose inputs, outputs, and dependency contracts match.

### Shared semantic components

A component may be shared only when all of the following are true:
- at least two mechanisms have the same semantic input and output roles.
- producer, first-use, barrier, and arithmetic-order requirements are the same.
- the abstraction owns a formula or invariant and deletes duplicated derivation.
- selected generated streams remain identical, or a separate deliberate-stream qualification succeeds.
- the component does not need exact-shape, winner, or schedule tables.

The first components to retain or establish are:
- the existing kernel signature, pointer-load, trailer, BF16 RNE, assembly module, and deterministic register allocation primitives.
- Q4_K/Q5_K packed scale/minimum reconstruction and correction semantics.
- Q8 tiled weight staging, scale staging, integer group accumulation, and wave-N store primitives where the ordinary, small-M, and compact-KV implementations already use the same emitter behavior.
- typed integer-WMMA operand and accumulator roles without imposing one ownership traversal.
- Q8_1 F32_D4 and F16_D4S4 addressing formulas without imposing one global-read or LDS schedule.
- output-coordinate and BF16 store helpers only where actual fragment ownership and emitted traversal are equal.

Q3 signed group-scale correction, Q6 signed-scale/block-factor correction, Q8 scale multiplication, and Q4_K/Q5_K scale/minimum correction remain distinct typed correction leaves. Similar `v_*` instruction spelling is not sufficient evidence to merge them.

## Work Order

## Execution Status

- Phase 0: freeze scope and baselines. Baseline: 516 unique sources (224 forward, 292 backward), including the isolated Q3_K control and 68 current Q8_0 catalog-valid sources; the preserved 447-source gate regenerated with zero changes, and 189 focused writer/spec tests passed.
- Phase 1: establish the pure physical-plan authority. Added the pure `mmq_fwd_physical` layer, concrete mechanism plans, plan-owned Q3/Q8 register limits, an explicit ordinary-Q8 512-byte LDS allocation pad, and one resource authority consumed by `DerivedForwardState`; 190 focused tests, all 516 baseline sources, and all 447 preserved sources pass unchanged.
- Phase 2: introduce the lowering boundary with the existing Q3_K control. Added the immutable format-neutral lowering context/protocol and moved the Q3 body plus six typed stage emitters into `mmq_fwd_lowering_packed_3bit`; no Q3 shape, catalog, dispatch, or bundle scope changed. The facade fell by 462 lines, 190 focused tests pass, and the 516-source and preserved 447-source matrices are byte-identical.
- Phase 3: extract and consolidate Q8_0. Q8 direct, register-tiled, wave-N tiled, small-M, and compact-KV bodies now use one physical-plan-dispatched lowerer; ordinary and small activation staging share parameterized helpers, and internal names no longer encode HIP provenance. The facade is 3,955 lines (down from 6,547), 190 focused tests pass, and all 516 baseline plus 447 preserved sources remain byte-identical.
- Phase 4: extract and simplify structured Q6_K. The typed schedule emitter and all Q6 body stages now live in `mmq_fwd_lowering_q6`; ownership, decode, typed operands, register roles, and LDS state live in the pure physical planner, whose layout now owns the `(158|210,27,38400)` resource facts used by both admission and emission. Duplicate/unreachable decode checks were removed. The facade is 1,249 lines, 191 focused tests and `ty check` pass, and all 516 baseline plus 447 preserved sources remain byte-identical.
- Phase 5: extract packed scale/minimum mechanisms and retire facade-local bodies. Direct-global lowering now lives in `mmq_fwd_lowering_packed_direct`, decoded-weight LDS lowering lives in `mmq_fwd_lowering_decoded_lds`, and shared metadata reconstruction lives in `mmq_fwd_lowering_metadata`. Their physical plans own register assignments, activation metadata, decoded LDS layout, resources, and disjoint metadata/output-address lifetimes. The facade is 104 lines; 192 focused tests, all 516 baseline sources, and all 447 preserved sources pass unchanged.
- Phase 6: complete evidence-based convergence, gate review, and tuning-knob review. Internal modules and plans are now named for packed-three-bit, packed-scale/minimum, decoded-LDS, structured-Q6, or signed-int8 contracts. `F16D4S4ActivationMetadata` owns shared activation group formulas and `mmq_fwd_lowering_mma` owns the one genuinely shared signed-int8 WMMA spelling. `QuantFormat` is the single fixed-format trait authority, and `ForwardMechanismContract` now owns lowering compatibility and reduction granularity. Forward validation dispatches by lowering mechanism rather than quant type. The gate audit removed the spurious universal `K % 256` check: signed-int8 consumes one 128-value D4 activation block while packed 3/4/5/6-bit mechanisms retain 256-value iterations. The final Phase 6 checkpoint passes 196 focused tests, 306 GGTensile tests, all 516 source identities, and all 447 preserved sources with zero changes.
- Phase 7: complete final qualification and recursive completion review. The final review found one actionable boundary defect: shared signed-int8 tiled-LDS helpers still accepted bundles of unrelated raw register integers. Immutable `SignedInt8TiledLdsRegisters` and `SignedInt8TiledLdsScaleLayout` projections now carry the common semantic roles and formula-derived scale layout. Requalification passes 203 focused forward tests, 306 GGTensile tests, and 391 repository tests with the 14 existing warnings. Three independent 516-source snapshots are byte-identical to the baseline and each other, all 447 preserved sources report zero changes, and the 179-kernel bundle is current and reproducible.

### Phase 0: Freeze scope and baselines

- Record that Q3_K shape discovery and catalog expansion were paused for original Phases 0-7.
- Enumerate every accepted forward `operand_source`, its quant-format domain, its physical-plan constructor, and the tests that execute its writer lines.
- Generate a baseline matrix for every selected forward exact key, every retained candidate needed for writer coverage, the isolated Q3_K control, and every preserved Q8_0 source.
- Preserve exact source, object, code-object, normalized executable text, symbol, ABI, metadata, resource, wait, barrier, and instruction-count evidence for the selected mechanisms.
- Update source-path and line-coverage support so future lowerer modules cannot escape the same writer coverage requirements.

No implementation move begins until this matrix can detect a missing family, an unexercised module, and a source change at the exact-key level.

### Phase 1: Establish the pure physical-plan authority

- Move LDS layouts, mechanism register roles, deterministic allocations, and resource derivation below one no-ROCISA planning boundary.
- Add one concrete physical-plan type per mechanism rather than one optional-field record covering all mechanisms.
- Make capability validation and the writer consume the same plan-derived `ForwardResourceUsage`.
- Add structural tests that compare declared VGPR high water, allocated VGPR granularity, SGPR count, and LDS total with inspected metadata.
- Delete duplicated resource arithmetic and magic layout totals only after exact source generation remains unchanged.

This phase is substantive even before files shrink: it removes two authorities for the same physical facts.

### Phase 2: Introduce the lowering boundary with the Q3_K control

- Add the immutable lowering context and closed mechanism lookup.
- Move `Packed3BitTiledLdsRegisterPlan`, its layout use, body orchestration, activation stage, weight prefetch/decode, half accumulation, and store into the Q3 lowerer.
- Replace raw register-number helper parameters with Q3 stage-role records or typed operands.
- Keep the exact half ordering, loop-carried prefetch, VMEM/LDS waits, barriers, WMMA count, arithmetic order, and store traversal unchanged.
- Validate only the existing qualified Q3_K exact control. Do not add inventory keys, candidates, selected entries, dispatch, or bundle sources.

The Q3 move is the bounded pilot for the lowerer interface because it exercises packed multi-plane decode, LDS staging, signed correction, and an epilogue without expanding Q3 coverage.

### Phase 3: Extract and consolidate Q8_0

- Move all Q8 register plans, role records, and body emitters into the Q8 lowerer.
- Add an explicit formula-derived physical layout for the ordinary `Q8HipTiledLds` path so its 128 activation rows, 64 weight rows, 304-byte weight stride, and scale offset are not body-local constants.
- Retain distinct physical plans for direct global, register tiled, ordinary HIP tiled, small-M 32/64, and compact KV layouts.
- Consolidate only the Q8 emitters already shared by matching roles: weight stage and writes, activation stage variants, integer group accumulation, scale reads, and wave-N stores.
- Preserve dependency-derived waits, weight-first compact-KV staging, paired compact weight-scale reads, the invariant second-base hoist, terminal barriers, and exact BF16 store order.
- Regenerate and compare all 23 selected exact keys and all 447 preserved Q8 sources.

The Q8 phase must demonstrate that one quant semantics can select multiple physical plans without conditionals leaking into the facade or exact-shape selection leaking into capability validation.

### Phase 4: Extract and simplify structured Q6_K

- Move the Q6 typed operands, ownership and decode plans, physical register map, semantic plan, schedule emitter, and body emission into the Q6 lowerer.
- Consolidate overlapping `Q6OwnershipRegisterPlan`, `Q6DecodeRegisterPlan`, `Q6PhysicalLayout`, schedule, and resource derivations into one immutable Q6 lowering state where they express the same fact.
- Preserve pinned assignments that are measured ownership facts and deterministic first-fit allocation for formula-derived roles.
- Keep single-row and dual-row setup, near/far reads, activation reads, and refill traversal as bounded Q6 policies unless one complete formula replaces them source-identically.
- Do not generalize the Q6 residual traversal from superficial similarity to Q3_K or Q8_0. Their evidence currently supports distinct ownership.
- Compare every selected and frozen Q6 source and its inspected artifact exactly.

### Phase 5: Extract Q4_K/Q5_K and retire facade-local bodies

- Move the direct Q4_K body and the decoded-weight-LDS Q4_K/Q5_K body into the Q4/Q5 lowerer.
- Replace facade class constants and raw cross-helper register integers with concrete direct and decoded physical plans.
- Keep Q4_K/Q5_K packed scale/minimum reconstruction shared through `QuantForwardSemantics`; retain the Q5 high-bit reconstruction leaf.
- Preserve metadata scheduling, F16_D4S4 staging, clamp behavior, accumulation order, epilogue policies, VOPD pairings, waits, and stores.
- Remove `_body_*` and `_emit_*` format methods from the facade only after every accepted mechanism resolves through the closed lookup with no fallback.

### Phase 6: Evidence-based cross-format convergence

- Rebuild the responsibility map from the extracted physical plans and lowerers.
- Search for duplicated formulas in activation addressing, WMMA operand projection, accumulator ownership, wait derivation, and output coordinates.
- Extract a shared component only through the shared-component criteria above.
- Reject a proposed component when it merely moves instruction text, introduces optional-field branching, or obscures materially different ownership.
- Remove transitional adapters, duplicate constants, dead helpers, stale imports, and source-path exceptions in the same phase that makes them unnecessary.

This review may conclude that the shared boundary is deliberately small. A thin facade plus explicit mechanism lowerers is preferable to a generic pipeline that stores or reconstructs selected schedules.

#### Gate-audit result

The completed audit classifies the remaining narrow conditions as follows:
- `QuantForwardSemantics.for_quant_type` and the packed Q3/Q6 semantic accessors remain format-specific because their payload planes, bit reconstruction, and correction formulas are intrinsic data contracts.
- Lowerer reads of `contract.quant_type` remain only where existing generated labels or comments contain the serialized format name. Removing those reads would change frozen assembly sources without changing mechanism behavior.
- The packed-three-bit 128x64 tile, structured-Q6 one/two-row ownership, signed-int8 wave-N depth domain, exact LM-head small-M layouts, and compact-KV layout remain mechanism geometry constraints backed by concrete register and LDS plans.
- Repeated quant-type sets for activation layout, clamp, decoder, scale arithmetic, and arithmetic contract were removed. `QuantFormat` owns those facts once.
- Quant-type dispatch for lowering compatibility and forward validation was removed. `ForwardMechanismContract.lowering`, semantic decoder capabilities, and formula-derived reduction granularity now own admission.
- Exact inventories and selected catalogs remain untouched and continue to own runtime selection. The newly expressed signed-int8 128-value capability does not add an exact selected key or public dispatch entry.

#### Tuning-knob review

At the Phase 6 checkpoint, no new knob was silently active and existing specs retained their exact serialized values. The following initial bounded candidates were deferred; the reopened tuning-field review below refines and extends this historical list. Unsupported values or mechanisms must reject during candidate construction.

| Candidate | Complete codegen choices | Disposition |
| --- | --- | --- |
| decoded metadata timing | existing `Serialized`, `MetadataAfterLowWmma`, `IndependentExtraction`, and `IndependentExtractionMetadataAfterLowWmma` domains | Already explicit in `DecodeSpec`; high-bit decode rejects the unsupported independent-before-low form. |
| epilogue batching | existing tiles-ahead `1..8`, dependency width `1..8`, priority `0..3`, and scalar-copy/VOPD initialization domains | Already explicit for decoded LDS; structured Q6 retains its separate scope and dependency-width contract. |
| structured-Q6 delay/cache policy | existing `None|Explicit`, `Default|InvalidateL0`, and `StoreBatch|FullTile` values | Already explicit and mechanism-owned. |
| signed-int8 tiled stage order | `WeightThenActivation`, `ActivationThenWeight`, or `Interleaved` | Deferred. It requires a new signed-int8 spec field with every serialized candidate carrying one value and complete timing qualification. |
| mechanism prefetch distance | integer `0|1` for a named packed-three-bit, decoded-LDS, or signed-int8 prefetch mechanism | Deferred and mechanism-specific; it may not become a universal inferred default. |
| scale-read policy | `Scalar`, `Paired`, or `PairedHoistedSecondBase` | Deferred. Paired/hoisted forms require layout-proven legal offsets; compact KV is the current proven owner and other layouts must reject. |
| LDS address policy | existing mechanism-specific recompute/hoist values, with any future `HoistPairBase` value explicit | Existing `LdsSpec` remains authoritative; no inferred repair or shape-based override is allowed. |
| clause policy | `None`, `LoadBatch`, `StoreBatch`, or `LoadAndStoreBatch`, with clause length derived from the selected batch | Deferred. A candidate field must control every emitted clause and preserve dependency order. |
| register-role order | named per-plan orders such as `Current` plus individually defined alternatives | Deferred. Arbitrary permutations and allocator-side retuning are rejected; each alternative must be a complete deterministic order. |
| VOPD pairing | `Disabled` or `DependencyCompatible` at a mechanism-owned site | Partly explicit today through Q6 policy and decoded initialization. A broader field is deferred until role and dependency equivalence is proven. |

Raw wait-count biases, arbitrary instruction permutations, optional terminal barrier removal, exact-shape winner maps, and post-emission rewriting are not tuning knobs. Waits remain producer/first-use facts, and the retained terminal barriers remain part of the qualified mechanisms unless a separate deliberate stream change passes correctness and multiply-performance qualification.

### Phase 7: Final qualification and recursive review

- Run all structural, identity, artifact, correctness, and repository gates.
- Confirm that every accepted solution field still affects canonical lowering or rejects.
- Confirm that every supported mechanism has complete writer-line coverage in its new module.
- Perform a fresh recursive review of the implementation, generated streams, responsibility map, and remaining duplication.
- Classify each finding as duplicate or closed, contract-incompatible, deferred with an explicit prerequisite, or actionable.
- Implement and requalify every actionable in-contract finding, then repeat the review until none remain.
- Update `ggtensile_plan.md` with the final durable writer architecture and leave this document with only any genuinely unfinished follow-up.

#### Final review result

- Actionable and resolved: the common signed-int8 tiled-LDS stage, group, and store helpers passed raw register-number bundles. They now consume immutable physical-plan projections while mechanism-specific setup retains its concrete register plan. Generated text remains byte-identical.
- Closed as already owned once: `QuantFormat` owns fixed packed-format traits, `ForwardMechanismContract` owns lowering admission and reduction granularity, concrete physical plans own registers/LDS/resources, and exact inventories own selection. No second accepted-field, resource, or lowering authority remains.
- Contract-incompatible sharing: packed-three-bit correction, packed scale/minimum correction, structured-Q6 traversal, signed-int8 scaling, ordinary wave-N ownership, exact small-M ownership, and compact-KV paired reads retain separate mechanism control because their data contracts, lifetimes, waits, or arithmetic differ.
- Compatibility-sensitive and closed: remaining quant-format reads in lowerers spell serialized labels/comments. Serialized operand-source values retain their historical Q3/Q8 names while internal plans and modules use mechanism names.
- Deferred with explicit prerequisites: stage order, prefetch distance, scale-read policy, clause policy, register-role order, broader VOPD pairing, Q3_K shape expansion, public dispatch, and bundle expansion require complete spec fields or a separate qualified campaign as described above.
- Qualification: all accepted mechanism families participate in complete writer-line coverage; accepted solution fields project into canonical state or reject; no obsolete Q3/Q4-Q5/Q8 transitional lowerer modules or imports remain. Focused, GGTensile, repository, source-identity, preserved-source, artifact, bundle-currency, and reproducible-build gates pass.

A repeated review after the typed signed-int8 projection found no remaining actionable mechanism within the original Phase 7 review scope. The later review below supersedes that completion statement.

## Reopened Recursive Review

The fresh review covered `ggtensile_plan.md`, this plan, the shared writer, both public direction writers, every forward lowerer, forward specification and physical planning, validation, inspection, search, and writer-coverage support. It found actionable structural work. None of the findings authorizes a selected instruction-stream change by itself.

### Findings and disposition

| Finding | Evidence | Disposition |
| --- | --- | --- |
| Forward mechanism identity has several authorities | Operand-source sets and branches recur in the facade signature, facade dispatch, mechanism contracts, spec projection, physical-plan construction, validation, inspection, and search. | Actionable. Extend the existing mechanism contract with the small set of shared capability facts and dispatch by its typed lowering kind. Do not create a callable registry or winner table. |
| Capability validation contains exact selection logic | Q3 full-weight validation embeds the 12 inventory sizes; Q8 small-M and compact-KV validation embeds exact LM-head/KV sizes. | Actionable. Retain positive/divisible geometry and physical-plan checks, and move exact applicability entirely to deployment catalogs. Add blind formula-compatible correctness tests before claiming broader capability. |
| Two Q8 layout classes are source-equivalent | At M64, `KvCompactTile` and `CompactDepth32WeightRows` derive identical dimensions and produced byte-identical normalized 645-line bodies in a direct generation probe. | Actionable. Use one compact depth-32 physical layout. Retire the unselected compatibility value through an explicit canonical-identity migration; do not retain two accepted names with identical lowering. |
| Q8 layout accidentally selects scheduling policy | `weight_scale_pair_base_delta is not None` currently selects paired scale reads, weight-first stage order, and wait placement. | Actionable. Split row representation, stage order, and scale-read policy in typed internal state. Existing serialized values map to the current composite so unchanged specs emit unchanged text. |
| Typed signed-int8 projections copy rather than constrain | `SignedInt8TiledLdsRegisters.from_plan()` copies 23 assignments, and `SignedInt8TiledLdsScaleLayout.from_layout()` copies three layout facts. | Actionable. Replace the constructed projections with structural protocols for the exact common roles/facts. Helpers consume the original concrete plans and layouts. |
| Q3 full-weight register access is mostly aliases | One generic lookup and 25 uppercase properties occupy about 100 lines while only renaming `RegisterAssignment.first_register`; the store uses the `activation_stage` alias where the plan already declares `output_address` at the same lifetime-reused register. | Actionable. Use the concrete typed register plan and short local aliases inside substantive emitters. Emit through the declared output role. Preserve every register number and line of generated text. |
| Q6 has a ceremonial fixed stage graph and a local micro-IR tail | `Q6SemanticStage`, `Q6SemanticPlan`, stage dispatch, and validation encode one fixed nine-call sequence. Several one-use BF16 methods wrap one instruction each. | Actionable. Call the nine named semantic emitters directly from the Q6 body and inline the one-use BF16 instruction wrappers with one orienting comment. Retain producer tracking, first-use waits, dependency delays, dual issue, LDS-role resolution, and store-width checks. |
| Forward has dead or one-hop public-looking state | `ForwardBodyLowering` is unused; `ForwardFormatTraits` only copies `QuantFormat`; `DerivedForwardState.lds_bytes`, three family resource wrappers, Q6 physical-plan aliases, and direct/decoded register aliases add no authority. | Actionable. Delete them or access their sole owner directly. Keep `write()`, `Assembly` formatting methods, and formula-bearing properties because they are public API or enforce clear semantics. |
| Backward remains a monolithic second authority | The 2,510-line writer owns validation assumptions, resource allocation, LDS layout, five readers/decoders, common pipeline scheduling, and final emission. Inspection independently reimplements its register-count arithmetic. | Actionable. Add typed backward contract/spec/derived/physical layers, then extract substantial decoder leaves and the common BF16-WMMA pipeline while retaining one public facade. Inspection consumes the same resource plan. |
| Backward contains hidden winner and production gates | `_q3_k_full_vopd_decode()` chooses an instruction form from exact `(M,N,K)` and geometry; validation embeds per-quant production `N` lists and quant allowlists for small geometry, SIA3, DepthU64, N64 pipelining, and next-packed prefetch. | Actionable. Materialize Q3 pairing as a complete policy. Replace production lists with formula checks. Replace quant allowlists with decoder/layout/lifetime capability predicates, then qualify each newly admitted cross-quant case independently. |
| Backward VOPD formation parses emitted text | `_Assembly.inst()` regex-matches three VALU spellings and rewrites them against pending zero moves. | Actionable. Call a typed, bounded `emit_pairable_with_pending_zero()` operation at the known eligible sites. Ordinary instruction emission must not inspect or rewrite text. |
| Search coverage does not match implemented mechanisms | Repository-owned candidate domains cover Q4/Q5/Q6 only; Q3/Q8 campaigns required temporary scripts. | Actionable after policy normalization. Add linked Q3/Q8 neighborhoods only for implemented alternatives; do not expose one broad Cartesian domain. |

### Boilerplate rule

Delete a helper when it only forwards one call, returns one nested attribute, copies a typed record field-for-field, or wraps one instruction at one call site. A helper remains justified when it does at least one of the following:
- enforces a reusable invariant or rejects an invalid semantic state.
- owns a nontrivial formula used by validation, planning, and emission.
- tracks producer/consumer state, register lifetime, or synchronization.
- is a stable public operation such as `source()`, `write()`, or assembly formatting.
- removes repeated substantive logic across at least two contract-equivalent mechanisms.

Apply that rule concretely:
- remove the unused `ForwardBodyLowering` protocol rather than inventing a consumer for it.
- remove `_quant_type`, `_direct_registers`, `_decoded_registers`, Q3 uppercase register properties, Q6 fixed stage records, one-use Q6 BF16 wrappers, dead resource wrappers, and one-hop physical-plan aliases.
- replace signed-int8 register/layout copy projections with read-only structural protocols; do not replace them with another constructed adapter.
- retain `Assembly.line/comment/label/inst`, deterministic register allocation, packed scale/minimum reconstruction, the shared signed-int8 WMMA spelling, Q6 producer/wait tracking, and Q3 decode closures. Their definitions are smaller than the duplicated logic or carry a real invariant.
- do not add base lowerer classes, one-method sibling writers, decorator registries, generic visitors, or an instruction IR merely to reduce visible repetition.

### Single mechanism authority

`ForwardMechanismContract` becomes the only source for facts shared across layers. In addition to its existing data contract and lowering kind, it may carry only:
- workitem-ID metadata use.
- wave-M versus wave-N ownership orientation.
- the internal physical-plan kind where it is not identical to the lowering kind.
- legacy serialized compatibility facts that are required to reproduce existing source.

The facade, `ForwardKernelSpec`, physical planning, validation, inspection, and search consume that descriptor. Concrete geometry, registers, LDS offsets, waits, instruction counts, and selected shapes do not belong in it. Lowerer construction remains a short explicit closed `match`; a registry of classes or callables would cost at least as much boilerplate and obscure imports.

Inspection derives expected WMMA/barrier counts by matching the concrete physical-plan type and formulas. It must not recreate operand-source sets. Backward inspection similarly consumes `BackwardPhysicalPlan.resources` instead of replaying `RegisterPool` arithmetic.

### Capability generalization

| Current narrow gate | Generalized rule | Required proof before catalog use |
| --- | --- | --- |
| Q3 full-weight exact 12-shape set | Any positive `(M,N,K)` divisible by its fixed macro tile and 256-value reduction contract | blind divisible shapes, exact/reference/mutation checks, resources, deterministic artifacts |
| Q8 small-M exact LM-head and compact-KV sizes | Any positive shape divisible by the selected MT32/64/128 x N64 geometry and 128-value reduction contract | each ownership/layout combination tested outside current inventory; no automatic selection |
| `KvCompactTile` exact M64 path | Canonical compact depth-32 layout with M64 ownership | normalized source identity for migrated controls and full exact-key requalification if selected identity changes |
| Backward per-quant production `N` lists | positive tile-divisible `N`, packed-block divisibility, and decoder-row formula | blind shape generation/build/inspection plus independent backward correctness and mutation |
| Backward small geometry and pipeline quant allowlists | decoder supports prepare/chunk emission; register roles do not overlap; LDS layout and selected schedule satisfy formulas | cross-quant source/build tests first, then exact gradient/reference and timing gates per candidate |
| Backward Q4/Q5 duplicate reader and scale/minimum preparation | one packed scale/minimum reader/decoder with an optional high-bit payload leaf | all Q4/Q5 selected and coverage sources byte-identical |

Do not lift gates that still encode a real physical contract. The packed-three-bit half-tile and full-weight tiles remain fixed 128x64 mechanisms; structured Q6 remains one/two output rows until a new physical plan exists; signed-int8 small-M register ownership remains MT32/64; Q8 lane sharing remains unsupported until its global-read and replication path is implemented and qualified.

### Tuning fields

Only add a serialized field after every listed value has a complete lowering and validation path. Existing specs first map to explicit internal policies that reproduce the current source.

| Policy | Initial values | Scope and linkage |
| --- | --- | --- |
| `SignedInt8LdsRows` | `Full304`, `Compact144` | Q8 tiled LDS only; determines row representation and legal depth, not stage order or scale reads. |
| `SignedInt8StageOrder` | existing `WeightThenActivation` and `Interleaved`; add `ActivationThenWeight` only with a complete lowering | Q8 tiled LDS; wait counts derive from producer order and first use rather than becoming knobs. |
| `SignedInt8ScaleRead` | existing `Scalar` and `PairedHoistedSecondBase`; add `Paired` only with a complete lowering | Legal only when layout offsets prove the pair/base delta. Decoupled from stage order. |
| `Q6Traversal` | `GroupMajor`, `RowRoleWavefront` | Replaces the six-field composite in canonical typed state. Clustering and latency derive from traversal; pressure, wait, and pairing stay fixed until alternate implementations exist. |
| `BackwardDecodeExtraction` | format-supported subset of `Scalar`, `Packed`, `PackedVopd` | One active field in `BackwardKernelSpec`; legacy quant-specific serialized fields project to it and inactive values reject. |
| `BackwardQ3Pairing` | `PartialVopd`, `FullVopd` | Replaces the exact-shape branch. Catalog controls must carry the value explicitly before the legacy branch is removed. |
| `BackwardMetadataLoad` | `ScalarFields`, `Vector128` | Packed scale/minimum decoder only; Q5 is the existing implementation proof, Q4 requires its own qualification. |
| `BackwardNibbleShift` | `PerChunk`, `Hoisted` | High-bit packed decoder only; do not expose it for formats without the same shift lifetime. |
| `BackwardPackedReadShare` | `Independent`, `LanePair` | Requires payload-plane ownership support from the selected decoder. Q8 remains rejected until implemented. |
| `BackwardPipeline` | named complete schedules such as `SingleBuffer`, `DoubleBufferSia3`, `DoubleBufferPgr2` | Links buffer count, PGR/PLR, schedule algorithm, LDS layout, and prefetch distance. Invalid partial combinations are not candidate states. |
| `ClausePolicy` | `None`, `LoadBatch`, `StoreBatch`, `LoadAndStoreBatch` | Mechanism-local, with length derived from the batch and dependency legality. |
| `RegisterRoleOrder` | `Current` plus individually implemented named alternatives | Physical-plan local. No arbitrary permutation, allocator repair, or shape-selected order. |

Raw wait-count offsets, absolute register numbers, exact-shape switches, arbitrary instruction permutations, optional barrier deletion, and post-emission rewriting are not tuning fields. Cache invalidation and dependency delay remain explicit only where a lowering already implements both values.

### Reopened work order

### Reopened execution status

- Phase 8: complete. Two detached-root snapshots regenerated all 106 authoritative exact-key artifacts (56 forward, 50 backward). Source, object, code object, normalized disassembly, inspection metadata, waits, barriers, clauses, VOPD counts, resources, symbols, and ABI records are identical. The historical 516-source archive remains the frozen writer-control baseline. Structural tests now inventory the known exact-shape, projection, resource-replay, and text-rewrite debt and automatically discover future backward physical/lowering modules. The complete GGTensile suite passes with 332 tests.
- Phase 9: complete. Removed the unused forward lowering protocol, copied format-traits record, dead state/resource projections, one-hop direct/decoded accessors, Q3 uppercase register bank, Q6 fixed semantic-stage records/dispatcher, one-use Q6 BF16 wrappers, Q6 physical aliases, signed-int8 field-copy adapters, and the backward quant-type accessor. Shared signed-int8 helpers now accept structural register/layout protocols, the Q3 store uses its declared output role, and Q6 retains only stateful schedule operations. Ruff and all 332 GGTensile tests pass. A fresh 106-artifact snapshot is byte-identical to Phase 8 in source, object, code object, normalized disassembly, and inspection metadata.
- Phase 10: complete. `ForwardMechanismContract` now owns lowering, physical-plan kind, wave ownership, workitem-ID use, and legacy matrix suffix; facade, spec, physical planning, inspection, and search consume those facts instead of parallel operand-source sets. Q3 full-weight and Q8 small/compact capability validation is formula-based while catalogs remain unchanged. The duplicate `SignedInt8KvTiledLdsLayout` and serialized `KvCompactTile` value are retired; the legacy factory maps `ggsol_13768a4d22953b59` to canonical compact-M64 `ggsol_11f54a8999dc20e6`, whose symbol-normalized source and disassembly hashes are identical (`6294ac0e...` and `185313d0...`). Typed Q8 stage-order and scale-read policies are independent of row layout. All 106 selected artifacts remain byte-identical to Phase 8, all 335 GGTensile tests pass, and blind GPU controls at Q3 `(128,64,256)` and Q8 `(64,64,128)` pass independent-reference, deterministic-repeat, mutation, metadata, and zero-spill gates with normalized RMSE `0.00571` and `0.00548`.
- Phase 11: complete. The 2,510-line backward monolith is now a 90-line public facade over `DerivedBackwardState`, active structured kernel spec, pure `BackwardPhysicalPlan`, common BF16-WMMA/pipeline lowerer, and a substantial quant reader/decoder component. The physical plan is the single register, LDS, address-state, decoder-row, load-count, and signature-resource authority and matches every legacy derivation for all 50 selected backward keys. Q4/Q5 share packed scale/minimum preparation while Q5 high-bit reconstruction remains distinct. `_Assembly.inst()` no longer parses instruction text: typed pending-zero pairing is confined to known static-coordinate sites. Automatic structural and complete-line coverage includes every `mmq_bwd_*.py` module. All 336 GGTensile tests pass, and the complete 106-artifact source/object/code-object/disassembly/metadata set remains byte-identical to Phase 8.
- Phase 12: complete. Backward validation now derives N/block, tile ownership, decoder rows, workgroup mapping, formula WMMA geometry, and physical capacity without production-size lists or enumerated geometry tables. `BackwardMechanismContract` owns decoder/layout/schedule capability, including the independently discovered rejection of padded DepthU64 Q4/Q5 while compact DepthU32 is valid across all formats. `Q3KPairing` is a strict serialized `Inactive`/`Partial`/`Full` policy; all exact-size pairing branches are gone, the shared legacy Q3 solution was split where policy differed, and all 50 backward old/new key-symbol mappings have symbol-normalized source/disassembly plus ABI/metadata/resource identity. Forward search now exposes Q3/Q8 complete-policy domains, and bounded backward decoder/pipeline/LDS neighborhoods contain only capability-valid implemented candidates. Six blind GPU controls spanning both Q3 policies and Q4/Q5/Q6/Q8 are bit-exact to HIP and independent dequantized BF16 matmul, deterministic, mutation-sensitive, code-object-v5, and spill/private-storage free. All 344 GGTensile tests pass; the final 106 selected artifacts are identical to the reviewed migration set.
- Phase 13: complete. The final recursive pass found and closed the remaining in-contract structural defects: read-only signed-int8 protocols now satisfy frozen concrete plans without copy adapters; backward deep-pipeline geometry is formula-derived rather than represented by a ten-entry table; dead mechanism fields, one-hop lowering aliases, a tautological LDS-resource check, and the duplicate backward `Assembly.inst()` override are gone; LDS buffer toggling consumes `BackwardPhysicalPlan`; and Q6 scalar/dual multiply emission no longer rewrites an opcode string. Structural tests now guard numeric geometry tables, planner/lowerer import purity, canonical backward field projection, inherited instruction formatting, and opcode-text replacement. The final gates pass with 350 GGTensile tests and 435 repository tests plus the 14 existing Python 3.14 TorchScript deprecation warnings. Ruff, formatting, `ty`, compileall, pre-commit, bundle currency, and `git diff --check` pass. Two final independent roots regenerated all 106 authoritative artifacts identically to each other and to the Phase 12 set across source, object, code object, normalized disassembly, symbols, ABI, metadata, resources, waits, barriers, clauses, and VOPD counts; the normalized-disassembly digest remains `5145bdbe0a52ce3220c0b93829764dbbb3e6788ede48fb7f64b373ae063f5dec`. The Q3 pairing migration recheck records 50 backward identity changes, 56 unchanged forward identities, and normalized source/disassembly/inspection identity. The 179-kernel public bundle is current. Historical 516/447 archives remain evidence under the Phase 8 definition rather than current deployment inventories. Because every final structural edit is source and executable identical, the qualified Phase 10/12 blind reference, determinism, mutation, metadata, and zero-spill GPU evidence remains applicable without a new timing campaign. A repeated global forward/backward/quant/shape review found no remaining actionable in-contract structural mechanism.

#### Phase 8: Refresh the identity baseline and structural guards

- Regenerate every selected forward/backward key, every writer-coverage key, all frozen controls, and the current public bundle from two clean roots.
- Record source, normalized text, object, code object, symbol, metadata, resources, waits, barriers, clauses, and VOPD identity. Historical 516/447 counts remain evidence, not the refreshed inventory definition.
- Extend writer-line coverage discovery to future `mmq_bwd_physical.py` and `mmq_bwd_lowering*.py` files automatically.
- Add focused structural checks for inventory logic in capability validation, exact-shape branches in writers, duplicated resource arithmetic in inspection, text-parsing emission, unused protocols, and field-for-field projection adapters.

#### Phase 9: Delete forward boilerplate source-identically

- Remove the dead and one-hop items listed above.
- Replace Q3 full-weight property aliases with concrete typed plan access and the declared output role.
- Replace signed-int8 copy projections with structural protocols.
- Remove the fixed Q6 semantic-stage records/dispatcher and call the named stage emitters directly in the same order.
- Inline only one-use instruction wrappers; retain stateful Q6 emitter operations.
- Require byte-identical source and executable artifacts for the complete refreshed forward matrix.

#### Phase 10: Normalize forward mechanism and capability ownership

- Extend `ForwardMechanismContract` with the bounded shared facts and delete repeated operand-source sets.
- Remove exact Q3/Q8 inventory checks from capability validation while leaving catalogs unchanged.
- Collapse the duplicate Q8 compact layout implementation and split internal layout, stage-order, and scale-read policies.
- Add formula-compatible negative/positive validation tests and blind correctness cases. Broader acceptance does not add a catalog entry.
- Treat retirement of `KvCompactTile` as a separate canonical-identity migration; require normalized body/executable identity and regenerate any retained manifests.

#### Phase 11: Establish the backward physical and lowering boundary

- Add `BackwardProblemContract`, active `BackwardKernelSpec`, `DerivedBackwardState`, and a closed `BackwardPhysicalPlan` with typed register/LDS/resource ownership.
- Keep one thin public writer facade and one common BF16-WMMA/pipeline lowerer. Extract substantial Q3, packed scale/minimum Q4/Q5, Q6, and Q8 reader/decoder components; do not create five one-method adapter classes.
- Share Q4/Q5 packed metadata addressing and scale/minimum preparation, with Q5 high-bit reconstruction remaining a distinct leaf.
- Replace `_Assembly` regex rewriting with a typed bounded pending-zero pairing operation at known call sites.
- Make inspection consume the same backward resource plan. Preserve all selected and writer-coverage streams byte-for-byte.

#### Phase 12: Remove hidden gates and expose implemented neighborhoods

- Replace backward production-size lists with formula capability and replace quant allowlists with typed decoder/layout/schedule predicates.
- Materialize Q3 partial/full VOPD pairing explicitly. Because adding the field changes canonical key identity and symbols, migrate catalogs/manifests in a separately reviewed identity change while requiring normalized executable identity for the legacy choice.
- Add Q3/Q8 linked forward neighborhoods and backward policy neighborhoods only for values with implemented lowerings.
- Run blind cross-quant correctness before timing; timing and exact catalogs remain the only selection authority.

#### Phase 13: Final qualification and recursive review

- Run focused writer/spec/validation/search tests, complete GGTensile tests, the complete repository suite, Ruff, formatting, `ty`, compileall, pre-commit, bundle currency, and `git diff --check`.
- Rebuild and compare every affected source, executable, object, code object, symbol, ABI, metadata field, resource, wait, barrier, clause, and VOPD count from independent roots.
- Run independent references, finite checks, input/weight/workspace mutation, gradients for backward, and deterministic rebuilds on every deliberate identity or stream exception.
- Repeat a global forward/backward/quant/shape review. Classify every finding as closed, contract-incompatible, deferred with an explicit prerequisite, or actionable; implement and requalify every actionable finding before repeating the review.
- Update `ggtensile_plan.md` only after a fresh pass finds no actionable in-contract structural mechanism.

## Validation Gates

### Identity-preserving migration

Every refactor phase is identity-preserving by default. For every affected selected, frozen, and writer-coverage key:
- generated assembly source must compare byte-for-byte.
- executable text, object, code object, symbol, ABI, metadata, launch ownership, waits, VOPD pairings, LDS offsets, barriers, and resources must remain identical.
- the generated artifact must retain zero private storage, spills, scratch instructions, calls, and dynamic stack.
- independent clean roots must rebuild byte-identically.
- focused forward and backward tests, complete GGTensile tests, and the complete repository suite must pass.

Executable identity discharges a new GPU timing run for a purely structural move. A phase that cannot retain source and executable identity must stop and be split from the structural migration.

An explicit canonical-field migration is not an identity-preserving phase because the solution hash and symbol change even when the body does not. Such a phase must record old/new key and symbol mappings, compare source after symbol normalization, compare executable text and metadata exactly, rebuild catalogs/manifests deliberately, and run the normal correctness gates. It may not conceal an instruction change.

### Deliberate instruction-stream change

An intentional stream change is a separate optimization change, even when discovered during refactoring. It requires:
- exact retained-parent agreement when arithmetic order is unchanged, plus finite outputs.
- the independent reference and direction-appropriate input, gradient, packed-weight, and activation-workspace mutation gates.
- strict inspection of VGPR, SGPR, LDS, private segment, spills, waits, barriers, WMMA, VMEM, LDS instructions, VOPD, calls, scratch, and code-object properties.
- deterministic source, object, and code-object rebuilds.
- every affected selected exact key plus blind formula-compatible shapes for any shared lowering change.
- serial warmed comparisons against the retained same-policy parent and HIP.
- repeated evidence that every affected selected exact key remains faster than HIP multiply, with no stable multiply regression against its retained parent.

Multiply timing decides retention. Lower VGPR use, lower LDS use, fewer instructions, or a smaller writer never overrides slower multiply timing. There is no fixed percentage promotion threshold.

### Structural tests

Add or update tests that prove:
- every validated `operand_source` maps to exactly one lowerer and every unknown source rejects before emission.
- lowerers cannot import inventories, selected catalogs, benchmark results, or runtime winner maps.
- capability validation contains no production-shape list, tensor family, model name, or exact selected size.
- pure physical planning does not import ROCISA, invoke the toolchain, or emit instructions.
- physical register roles and LDS layouts are the sole resource derivation authority.
- inspection consumes plan-derived resources and does not replay register allocation.
- accepted solution fields are projected into the physical plan or explicitly rejected as inactive.
- all lowerer source files participate in writer line coverage and forbidden-pattern checks.
- no raw assembly templates, opcode tables, issue-slot maps, rank lists, source-order fallbacks, compatibility writers, exact-shape instruction switches, or post-emission rewriting are introduced.
- shared component protocols do not construct field-for-field copies of concrete plans, and unused or one-call forwarding abstractions are rejected during review.
- shared `kernel_writer_assembly.py` changes preserve both forward and backward source generation.

Repository gates remain Ruff, formatting, `ty check`, `compileall`, pre-commit, bundle currency, frozen-source verification, and `git diff --check` in addition to the test suites.

## Non-Goals

This plan does not include:
- selecting new deployment keys merely because formula capability becomes broader.
- changing Q3_K or Q8_0 public runtime dispatch or adding either to the public bundle.
- retuning any selected forward or backward kernel as part of a structural file move.
- changing the backward mathematical algorithm, arithmetic order, ownership, or store order during lowerer extraction.
- forcing all formats through one physical tile, LDS layout, stage graph, register plan, or epilogue.
- a generic scheduler, ready list, learned ranker, opaque cost model, repair path, or hidden allocation policy.
- prepared weights, dense shadows, external decode workspaces, split-K, persistent/grouped workgroups, producer fusion, hidden caches, or online tuning.
- a cosmetic split that leaves duplicate derivation, facade-local physical knowledge, or forwarding writers in place.

## Completion Rule

The refactor is complete only when:
- `ForwardKernelWriterAssembly` is a public facade with no format-specific body or register-layout implementation.
- `BackwardKernelWriterAssembly` is a public facade with no quant decoder, register allocator, pipeline body, or textual VOPD reconstruction.
- every accepted forward and backward mechanism has one concrete physical plan and one substantial explicit lowerer.
- shared mechanism facts have one typed authority; facades, validation, planning, inspection, and search contain no parallel operand-source or quant allowlists for the same fact.
- contract/specification state, physical planning, mechanism emission, exact selection, and public runtime dispatch have separate owners.
- register assignment, LDS layout, and resource usage are derived once and agree with inspected artifacts.
- typed operands and role records cross component boundaries; raw physical register integers remain only inside the final local emission boundary where unavoidable.
- capability validation is formula-based and exact deployment applicability remains catalog-owned.
- no accepted compatibility value produces the same canonical lowering as another accepted value.
- no writer parses emitted text or chooses an instruction policy from an exact problem size.
- all affected selected and frozen streams are byte-identical, or every deliberate exception has passed the complete correctness, artifact, determinism, and multiply-performance gates.
- all 56 selected forward and all 50 selected backward research keys retain their qualified behavior; broader capability does not imply new selection.
- public dispatch and the 179-kernel bundle remain unchanged.
- a fresh recursive review finds no actionable in-contract structural mechanism.

Phases 0-7 met the original forward-only completion rule. Phases 8-13 now meet the reopened bidirectional completion rule, and the final recursive review found no actionable in-contract structural mechanism. Public integration remains a separate deferred campaign, not unfinished work in this refactor.
