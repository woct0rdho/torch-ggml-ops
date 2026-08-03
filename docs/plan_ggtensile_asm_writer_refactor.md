# Remaining GGTensile Assembly-Writer Work

## Purpose

This document contains only unfinished MMQ forward-writer work. Completed architecture, principles, implementation status, and verification evidence are maintained in `ggtensile_plan.md`. Format-specific timing and optimization history remain in the experiment records.

The Q4_K/Q5_K/Q6_K writer refactor is closed for its implemented domain. The current qualified lowering must remain unchanged unless a complete replacement passes the gates below. Global forward-format completion still requires Q3_K and Q8_0, after which the additional format evidence may justify a deeper convergence refactor.

## Work Order

The remaining work has three stages:
- Complete the Q3_K MMQ forward campaign and canonical catalogs.
- Complete both ordinary and language-model-head Q8_0 MMQ forward campaigns and canonical catalogs.
- Reevaluate and, if justified, implement the cross-format forward convergence path.

The Q3_K and Q8_0 campaigns may be developed independently, but the convergence review must not begin until both are complete.

## Q3_K Forward Campaign

Build Q3_K through the current contract/specification/derived-state architecture rather than adding a standalone physical writer.

Required deliverables:
- a canonical Q3_K forward problem contract and `QuantForwardSemantics` covering packed planes, signedness, scale reconstruction, activation layout, correction arithmetic, BF16 output, and ABI.
- formula-derived geometry, ownership, packed strides, activation strides, loop counts, LDS, registers, and resources.
- one or more complete lowering mechanisms expressed through existing semantic stages and typed operands.
- an exact workload inventory and strict selected-solution catalog; capability validation must remain formula-based and contain no winner map.
- complete candidate and exact-pair round trips through the normal `ForwardSolution` and `SolutionKey` boundary.
- deterministic generation, build, inspection, correctness, independent-reference, mutation, screen, and confirmation artifacts.
- regression coverage proving that accepted fields affect canonical lowering or reject and that existing Q4_K/Q5_K/Q6_K selected sources remain unchanged unless a deliberate shared change is separately qualified.
- a fresh recursive optimization-exhaustion review after the final Q3_K implementation or measurement premise.

Q3_K completion requires every required exact key to have an explicit selected or fallback decision. A successful ordinary shape does not authorize another family or language-model-head shape without exact evidence.

## Q8_0 Forward Campaigns

Treat ordinary Q8_0 and language-model-head Q8_0 as separate exact-shape and ownership campaigns. Do not infer either from the legacy HIP bundle, MMQ backward Q8_0, or fixed-group Q8_0.

Required deliverables:
- canonical ordinary and language-model-head inventories with representative tensors, call counts, exact shapes, controls, and selection status.
- a Q8_0 forward contract and quant semantics that explicitly define packed-weight layout, activation-workspace layout, integer-dot or alternate arithmetic, scaling, destination rounding, and ABI.
- complete geometry, ownership, global-read, LDS, decode, dot, correction, epilogue, instruction, and resource policies for each admitted mechanism.
- formula-derived capability validation and exact catalogs kept as separate authorities.
- deterministic candidate manifests and exact-pair evidence through the normal forward model.
- independent-reference and adversarial packed-data coverage, including all scale fields, block boundaries, ownership boundaries, reduced K trips, and output boundaries.
- input, packed-weight, and activation-workspace mutation sensitivity.
- strict artifact inspection and byte-identical source/code-object rebuilds.
- warmed rotating screens and two confirmations for every promoted exact key.
- a recursive optimization-exhaustion review for ordinary Q8_0 and another for the language-model-head domain when their mechanisms or ownership differ.

Ordinary Q8_0 completion does not count as language-model-head completion, and neither counts as grouped fixed-group Q8_0 coverage.

## Post-Format Forward Convergence Review

### Entry criteria

Do not begin this work until:
- Q3_K forward has a completed campaign, a canonical catalog with an explicit selected-or-HIP-fallback decision for every required key, qualification evidence, and a final recursive review.
- ordinary Q8_0 forward has the same completed catalog decisions, qualification evidence, and final recursive review.
- language-model-head Q8_0 forward has the same completed evidence.
- the selected Q4_K/Q5_K/Q6_K artifacts still pass their frozen regression and resource gates.

Before selecting an abstraction, produce one responsibility map across Q3_K, Q4_K, Q5_K, Q6_K, and Q8_0 covering:
- packed-weight payload decode and metadata reconstruction.
- Q8_1 activation-workspace addressing and staging.
- integer or BF16 MMA operand ownership.
- post-MMA scale, minimum, sum, signed-scale, and block-factor correction.
- LDS stage, refill, barrier, and first-use wait behavior.
- accumulator ownership, BF16 conversion, and output traversal.

The map must identify actual shared semantics and actual format-specific semantics. Similar instruction text alone is not evidence of a shared component.

### Highest-value path to evaluate

Reevaluate this sequence from the complete five-format evidence:
- Consolidate the overlapping Q6 ownership, decode, physical-register, LDS, lifetime, payload, accumulator, and output derivations into one immutable lowering state. The state must remain formula-derived and contain no instruction stream or issue-order table.
- Define a complete semantic ownership descriptor for the residual single-row/dual-row Q6 setup, near/far reads, activation reads, and refill traversal. Derive owner lane and row, payload atom, address recurrence, destination role, refill slot, and first-use stage from geometry so one emitter can lower both modes.
- Separate forward lowering into typed semantic domains equivalent to `WeightTile`, `Q81ActivationTile`, `IntegerDot`, `CorrectionTerms`, and `Epilogue`.
- Let Q4_K/Q5_K share scale/minimum correction semantics while Q3_K, Q6_K, and Q8_0 provide their own typed correction terms. Do not force distinct arithmetic or memory ownership into one physical pipeline.
- Lower those domains through deterministic mechanism policies while preserving distinct geometry and memory strategies where campaign evidence shows they are material.

The Q3_K/Q8_0 evidence may revise, narrow, or reject this proposed decomposition. No abstraction is accepted solely because it shortens the current Q6 implementation. If the completed format map shows no defensible common ownership descriptor, record that conclusion and preserve the bounded Q6 policies.

### What does not count

The following do not satisfy the convergence work:
- splitting the current writer into more files without deleting duplicated derivation.
- moving Q6 into a sibling or forwarding writer.
- encoding physical register or instruction order in tuples, tables, JSON, templates, ranks, or issue slots.
- introducing a generic scheduler, learned ranker, opaque cost model, repair path, or source-order fallback.
- invoking HIP, LLVM, TensileLite, or another compiler's scheduler or allocator during generation.
- forcing all formats through one representation when selected geometry, arithmetic, resources, or timing show distinct mechanisms are required.

## Validation Gates

### Identity-preserving changes

For a structural change intended to preserve behavior:
- compare generated source exactly for every affected selected and frozen catalog-valid pair.
- compare executable text, code-object identity, symbol, ABI, metadata, launch ownership, waits, VOPD pairings, LDS offsets, barriers, and resources.
- preserve zero private storage, spills, scratch, calls, and dynamic stack.
- run all focused writer tests and the complete repository suite.

### Deliberate stream changes

For an intentional stream change:
- require finite output and exact retained-parent agreement when arithmetic order is unchanged.
- require an independent reference and input, packed-weight, and workspace mutation sensitivity.
- inspect exact VGPR, SGPR, LDS-byte, private-segment, spill, wait, barrier, WMMA, VOPD, VMEM, LDS-instruction, and code-object properties.
- rebuild source and code objects deterministically.
- qualify representative selected shapes and at least one blind formula-compatible shape for shared lowering.
- compare with the retained same-policy parent, not only an unrelated HIP endpoint.
- require serial warmed screens and two independent confirmations before promotion.

Repository gates remain the complete GGTensile and repository tests, Ruff, formatting, `ty check`, `compileall`, pre-commit, bundle currency, and `git diff --check`.

## Completion Rule

This plan is complete only when:
- required Q3_K forward exact keys have final catalog decisions.
- ordinary and language-model-head Q8_0 exact keys have final catalog decisions.
- the five-format responsibility map is complete.
- the convergence path is either implemented through all gates or explicitly rejected from measured and structural evidence.
- a fresh recursive review after the last actionable change finds no additional in-contract structural mechanism.
- `ggtensile_plan.md` is updated with the resulting durable design and project status, leaving this file with no completed work log.
