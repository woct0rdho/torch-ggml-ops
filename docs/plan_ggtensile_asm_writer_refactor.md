# GGTensile Forward Assembly-Writer Refactor Plan

## Purpose

This document records the completed MMQ forward assembly-writer refactor. The durable contracts and design principles remain authoritative in `ggtensile_plan.md`; format-specific arithmetic, timing, and rejected mechanisms remain in the experiment records.

The previous work order waited for complete Q3_K and Q8_0 campaigns before beginning cross-format convergence. That prerequisite is retired. Q8_0 is now complete for its 23 required exact keys, with 20 ordinary HIP-shaped LDS selections, exact small-M M32/M64 selections, and the compact KV M2048 selection. The isolated Q3_K `(M,N,K)=(2048,4096,2048)` control supplies a correctness-qualified, resource-clean, faster-than-HIP example of a materially different packed decode and correction path. Together with the mature structured Q6_K path and the retained Q4_K/Q5_K mechanisms, this is enough evidence to define the writer boundaries now.

Q3_K inventory expansion, new Q3_K shape implementation, canonical Q3_K catalog work, and Q3_K public integration are paused. They are not entry criteria for this refactor and are not deliverables of it. The refactor must preserve the existing Q3_K control so that a later Q3_K campaign starts from the same qualified mechanism.

This is a structural project. Its target is clearer ownership, single derivation of physical facts, typed component boundaries, and a smaller blast radius for later format work. File count and line count are not success metrics.

## Fixed Boundaries

The refactor must preserve these project boundaries:
- `ForwardKernelWriterAssembly` remains the public forward assembly-writer entry point with the existing constructor, `source()`, and `write()` behavior.
- `ProblemType`, `ProblemSize`, `ForwardSolution`, `SolutionKey`, `ForwardProblemContract`, and `ForwardKernelSpec` remain the generation and identity boundary.
- Capability validation remains formula-based. Exact inventories and selected catalogs remain the only selection authority.
- A lowering mechanism may dispatch on validated mechanism identity such as `operand_source`; it may not inspect tensor names, exact-key catalogs, measured winners, or benchmark artifacts.
- gfx1151, code-object v5, wave32, WMMA V1, the 40-byte MMQ ABI, Q8_1 workspace contracts, arithmetic order, FP32 accumulation where selected, BF16 RNE output, and zero private storage remain unchanged.
- Public runtime dispatch and the current 179-kernel bundle remain unchanged. Public integration is a separate review.
- The 447 preserved Q8_0 forward sources remain byte-identical unless an instruction-stream change is deliberately separated and fully qualified.
- Backward lowering is outside this project. Changes to direction-neutral assembly primitives must still preserve all affected backward sources and tests.

## Baseline Responsibility Map

Before the refactor, the approximately 6,500-line `kernel_writer_assembly_mmq_fwd.py` combined four different responsibilities: the public source envelope, lowering-family dispatch, mechanism-specific physical plans, and final instruction emission. It also contained older direct and decoded paths beside the structured Q6, Q8, and Q3 implementations. The size itself was not the defect; the defect was that mechanism ownership and shared ownership were not explicit at the module boundary.

The baseline families provided the following evidence:

| Mechanism | Baseline domain | Evidence that must be preserved | Physical behavior that remains mechanism-owned |
| --- | --- | --- | --- |
| `Global` | direct Q4_K control | retained generated controls and tests | one-wave global payload reads, direct nibble decode, scale/minimum correction, scalar output traversal |
| `DecodedWeightLdsBatch8` | shared Q4_K/Q5_K | selected and catalog-valid Q4_K/Q5_K sources | decoded-weight LDS rows, F16_D4S4 activation staging, metadata scheduling, clamp behavior, pipelined epilogue |
| `Q6StructuredDecoded` | selected Q6_K | structured semantic plans, selected sources, resource and timing qualification | single-row/dual-row ownership, near/far reads, signed Q6 decode, refill traversal, explicit dependency delays |
| `Q3HipTiledLds` | one isolated Q3_K control | exact HIP/public agreement, independent reference, mutation gates, deterministic rebuilds, faster multiply | 110-byte multi-plane decode, half-block staging, signed six-bit group scales, loop-carried weight prefetch |
| `Q8DirectGlobal` and `Q8RegisterTiled` | Q8_0 research controls | accepted candidate generation and writer coverage | direct/register ownership with no shared tiled-LDS pipeline |
| `Q8HipTiledLds` | 20 selected Q8_0 keys | all exact selections and frozen-source evidence | 128x64 wave-N ownership, 304-byte weight rows, depth-32/depth-64 traversal, hoisted activation LDS addresses |
| `Q8SmallMTiledLds` | LM-head M32/M64 and compact KV M2048 | three exact selected controls and frozen-source evidence | exact small-M ownership, 32/64 activation rows, 304-byte or compact 144-byte weight rows, paired KV scale reads |

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

Use one closed forward-lowering protocol and concrete format-family modules. Intended ownership is:

| Module | Ownership |
| --- | --- |
| `kernel_writer_assembly_mmq_fwd.py` | public facade, source envelope, signature, closed dispatch |
| `mmq_fwd_physical.py` | pure physical-plan union, shared role primitives, resource authority, no ROCISA or instruction emission |
| `mmq_fwd_lowering.py` | immutable lowering context, minimal lowering protocol, closed mechanism lookup |
| `mmq_fwd_lowering_packed_3bit.py` | `Q3HipTiledLds` body and Q3-specific stage emitters |
| `mmq_fwd_lowering_packed_direct.py` | direct packed scale/minimum body |
| `mmq_fwd_lowering_decoded_lds.py` | decoded-weight LDS, F16_D4S4 activation staging, and scaled integer MMA |
| `mmq_fwd_lowering_metadata.py` | packed scale/minimum metadata reconstruction shared by compatible mechanisms |
| `mmq_fwd_lowering_mma.py` | one source-identical signed-int8 WMMA constructor shared by compatible mechanisms |
| `mmq_fwd_lowering_q6.py` | structured Q6 state adapters, schedule emitter, and final Q6 body |
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

- Record that Q3_K shape discovery and catalog expansion are paused for the duration of this plan.
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

No new knob is silently active after this structural refactor. Existing specs retain their exact serialized values. Future optimization work may introduce only the following bounded candidates as explicit complete fields; unsupported values or mechanisms must reject during candidate construction.

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

A repeated review after the typed signed-int8 projection found no remaining actionable in-contract structural mechanism.

## Validation Gates

### Identity-preserving migration

Every refactor phase is identity-preserving by default. For every affected selected, frozen, and writer-coverage key:
- generated assembly source must compare byte-for-byte.
- executable text, object, code object, symbol, ABI, metadata, launch ownership, waits, VOPD pairings, LDS offsets, barriers, and resources must remain identical.
- the generated artifact must retain zero private storage, spills, scratch instructions, calls, and dynamic stack.
- independent clean roots must rebuild byte-identically.
- focused forward tests, complete GGTensile tests, and the complete repository suite must pass.

Executable identity discharges a new GPU timing run for a purely structural move. A phase that cannot retain source and executable identity must stop and be split from the structural migration.

### Deliberate instruction-stream change

An intentional stream change is a separate optimization change, even when discovered during refactoring. It requires:
- exact retained-parent agreement when arithmetic order is unchanged, plus finite outputs.
- the independent reference and input, packed-weight, and activation-workspace mutation gates.
- strict inspection of VGPR, SGPR, LDS, private segment, spills, waits, barriers, WMMA, VMEM, LDS instructions, VOPD, calls, scratch, and code-object properties.
- deterministic source, object, and code-object rebuilds.
- representative affected selected shapes; the existing Q3_K control is the complete Q3_K scope for this project.
- serial warmed comparisons against the retained same-policy parent and HIP.
- repeated evidence that every affected selected exact key remains faster than HIP multiply, with no stable multiply regression against its retained parent.

Multiply timing decides retention. Lower VGPR use, lower LDS use, fewer instructions, or a smaller writer never overrides slower multiply timing. There is no fixed percentage promotion threshold.

### Structural tests

Add or update tests that prove:
- every validated `operand_source` maps to exactly one lowerer and every unknown source rejects before emission.
- lowerers cannot import inventories, selected catalogs, benchmark results, or runtime winner maps.
- pure physical planning does not import ROCISA, invoke the toolchain, or emit instructions.
- physical register roles and LDS layouts are the sole resource derivation authority.
- accepted solution fields are projected into the physical plan or explicitly rejected as inactive.
- all lowerer source files participate in writer line coverage and forbidden-pattern checks.
- no raw assembly templates, opcode tables, issue-slot maps, rank lists, source-order fallbacks, compatibility writers, or post-emission rewriting are introduced.
- shared `kernel_writer_assembly.py` changes preserve both forward and backward source generation.

Repository gates remain Ruff, formatting, `ty check`, `compileall`, pre-commit, bundle currency, frozen-source verification, and `git diff --check` in addition to the test suites.

## Non-Goals

This plan does not include:
- discovering, implementing, timing, or selecting additional Q3_K shapes.
- changing Q3_K or Q8_0 public runtime dispatch or adding either to the public bundle.
- retuning selected Q4_K, Q5_K, Q6_K, or Q8_0 kernels as part of a file move.
- rewriting the backward assembly algorithm.
- forcing all formats through one physical tile, LDS layout, stage graph, register plan, or epilogue.
- a generic scheduler, ready list, learned ranker, opaque cost model, repair path, or hidden allocation policy.
- prepared weights, dense shadows, external decode workspaces, split-K, persistent/grouped workgroups, producer fusion, hidden caches, or online tuning.
- a cosmetic split that leaves duplicate derivation, facade-local physical knowledge, or forwarding writers in place.

## Completion Rule

The refactor is complete only when:
- `ForwardKernelWriterAssembly` is a public facade with no format-specific body or register-layout implementation.
- every accepted forward mechanism has one concrete physical plan and one explicit lowerer.
- contract/specification state, physical planning, mechanism emission, exact selection, and public runtime dispatch have separate owners.
- register assignment, LDS layout, and resource usage are derived once and agree with inspected artifacts.
- typed operands and role records cross component boundaries; raw physical register integers remain only inside the final local emission boundary where unavoidable.
- all affected selected and frozen streams are byte-identical, or every deliberate exception has passed the complete correctness, artifact, determinism, and multiply-performance gates.
- all 23 selected Q8_0 keys and the existing isolated Q3_K control retain their qualified behavior without adding Q3_K coverage.
- public dispatch and the 179-kernel bundle remain unchanged.
- a fresh recursive review finds no actionable in-contract structural mechanism.

All completion conditions are met at the Phase 7 checkpoint above. Q3_K inventory expansion and public integration remain separate deferred campaigns, not unfinished work in this refactor.
