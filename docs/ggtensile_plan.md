# GGTensile Design and Progress

## Purpose

GGTensile is the repository-local assembly kernel generator for packed GGUF matrix multiplication. It accepts one exact `ProblemType`, one exact `ProblemSize`, and one complete direction-specific solution, then either emits reproducible assembly or returns a structured rejection reason.

This document is authoritative for the generic design, implementation principles, supported scope, and current project progress. Format-specific problem definitions, measurements, rejected mechanisms, artifact identities, and campaign chronology belong in the experiment records. The remaining forward-writer work is tracked separately in `plan_ggtensile_asm_writer_refactor.md`.

GGTensile uses a deliberately small ROCISA surface inspired by TensileLite: structured modules and metadata, explicit register pools, and direct assembler/linker invocation. It does not import TensileLite's solution, problem-type, search, scheduling, allocation, or library-generation machinery. Existing HIP kernels remain the correctness control, performance control, and runtime fallback.

## Project Scope

The current backend targets gfx1151, code-object version 5, wave32, and WMMA V1. Support expands explicitly by operation, direction, quant format, exact shape family, activation layout, destination type, and ISA.

The project boundary is:
- exact positive problem sizes supplied at generation time.
- explicit complete solution identity with no silent repair or inferred tuning values.
- authoritative packed GGUF weights, decoded inside the generated kernel unless a separate public representation contract is designed.
- shape dimensions exactly divisible by the selected ownership, vector widths, and reduction depth; edge tiles reject.
- exact-key runtime applicability and HIP fallback for every mismatch.
- formula-derived resource admission under the target VGPR, SGPR, LDS, and code-object constraints.
- zero private storage, spills, scratch instructions, calls, and dynamic stack.
- no split reduction, atomics, persistent traversal, prepared-weight cache, hidden dense shadow, or external decode workspace unless the mechanism receives an explicit model, ABI, lifetime, validation, and dispatch contract.

Unsupported problems and solutions are normal validation results, not generator failures. Broader ISAs, edge handling, new operations, grouped ownership, prepared representations, and multi-kernel reductions are separate expansion projects rather than implicit capabilities.

## Required Format Inventory

The workload, not the set of types understood by generic GGUF code, defines required production coverage.

| Operation | Workload | Required formats |
| --- | --- | --- |
| MMQ | Qwen ordinary projections and language-model head | `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` |
| MMQ | DeepSeek ordinary projections and language-model head | `Q8_0` |
| Grouped MMQ | Qwen experts | `Q3_K`, `Q4_K`, `Q5_K`, `IQ2_S` |
| Grouped MMQ | DeepSeek experts | `Q2_K`, `IQ2_XXS`, fixed-group `Q8_0` |

The MMQ union is `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0`. The grouped union is `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `IQ2_XXS`, `IQ2_S`, and fixed-group `Q8_0`.

`Q6_K` has no current grouped production workload. `Q2_K`, `IQ2_XXS`, and `IQ2_S` have no current required ordinary MMQ campaign. Ordinary `IQ2_S` tests and production inventory are intentionally excluded. Fixed-group `Q8_0` is a distinct grouped contract and is not implied by ordinary `Q8_0` coverage.

## Design Principles

### Exact contracts, no repair

One complete problem and solution must predict one physical assembly stream. Unknown fields, missing fields, implicit numeric coercions, incompatible linked values, and unsupported policies reject before lowering. The generator never rewrites a request into a nearby valid solution.

Every accepted serialized field must affect canonical identity and lowering, or be explicitly fixed by the problem contract. Mutation tests enforce that accepted fields are projected or rejected rather than silently ignored.

### Deterministic, self-contained generation

Production generation does not search, benchmark, invoke HIP or LLVM code generation, consult measured winners, run an allocator or scheduler, repair register pressure, or fall back to source or insertion order. Search, profiling, compiler experiments, and GPU timing are offline evidence only.

The same complete input must produce byte-identical source. Assembly, linking, and inspection then verify deterministic object and code-object translation.

### Capability, candidate, and selection are separate

Capability validation describes the formula-supported domain. Candidate manifests describe complete parameter points. Exact-key inventories and selected-solution catalogs describe measured production choices.

Winner maps never belong in capability validation. Candidate identity is parameter-only; exact-pair identity additionally includes the problem type and exact shape; artifact identity additionally includes generated source.

### Generate semantics, not stored schedules

The writer describes logical work, ownership, dependencies, and traversal. It must not become a repository of copied physical instruction streams. Shortening by moving a schedule into another module, template, JSON file, tuple, opcode table, issue-slot map, rank list, or compatibility writer is not simplification.

Line count is not the objective. A valid simplification deletes duplicated derivation or schedule-shaped implementation by replacing it with a typed invariant, formula, or genuinely shared semantic mechanism.

### Preserve direction-specific algorithms

Forward and backward share only mechanisms whose semantics and emitted text are genuinely direction-neutral. They retain separate solution contracts and lowering algorithms:
- forward consumes a previously produced Q8_1 activation workspace, expands integer weights, uses integer WMMA, and applies joint weight/activation correction.
- backward reads BF16 activations, dequantizes weights to BF16, and converges on a shared BF16-WMMA pipeline.
- packed-weight addressing, LDS ownership, output ownership, launch mapping, and epilogue traversal remain direction-specific where required.

GGTensile forward does not quantize activations. The Q8_1 producer is an upstream kernel with its own public workspace contract.

### Derive redundant state once

Geometry, ownership, grids, packed strides, activation strides, loop counts, accumulator counts, register roles, lifetimes, LDS offsets, waits, and resource usage derive from one authoritative contract/specification boundary. Exact shape names and selected resources are not derivation inputs.

Formula-derived capability may accept divisible shapes beyond the selected inventory. Catalog coverage and GPU evidence remain exact-key facts.

### Typed boundaries carry invariants

Components consume typed logical operands, register roles, memory roles, and dependency records rather than opaque `vN`, `sN`, half-register, address-expression, or instruction strings. Small helpers are retained only when they enforce a typed invariant, own a nontrivial formula, or form a reused abstraction boundary.

Contractual invalid states use explicit rejection reasons. Natural programming errors propagate. Python `try` blocks are reserved for releasing acquired resources or restoring process-global state before re-raising.

### Completion requires recursive review

Every campaign and structural refactor ends with a fresh recursive review of the plan, implementation, generated artifacts, selected and rejected evidence, target ISA, and related kernel work. Findings are classified as duplicate or closed, contract-incompatible, unsupported, deferred with an explicit prerequisite, or actionable.

An actionable finding must be implemented and qualified before the review is repeated. Completion is valid only when a fresh pass finds no actionable in-contract mechanism.

## Model and Identity

### Public records

`tools.ggtensile` exposes immutable, JSON-serializable records:
- `ProblemType`: operation, quant and activation types, destination and compute types, and transpose/layout semantics.
- `ProblemSize`: exact GEMM coordinates and operation-specific dimensions.
- `ForwardSolution` and `BackwardSolution`: complete direction-specific kernel choices.
- `SolutionKey`: the exact problem-type/problem-size/solution tuple with canonical JSON and a stable content hash.
- `RejectReason`: stable rule ID, diagnostic, involved parameters, and source.
- `KernelArtifact`: symbol, source/object/code-object paths, source identity, ABI, launch geometry, and inspected resources.

Named defaults are explicit and remain part of canonical identity. Parsing is strict and generation never mutates a solution.

### Forward contract layers

The forward implementation separates fixed semantics, complete choices, and derived facts:

```text
ForwardProblemContract
  quant format and packed block
  activation layout and block
  arithmetic and signedness
  destination and BF16 rounding
  ISA, wavefront, ABI

ForwardKernelSpec
  geometry and ownership
  global memory and LDS representation
  decode and iteration policy
  dot and epilogue policy
  instruction policy
  resource limits

DerivedForwardState
  macro tile and workgroup ownership
  exact grid and K-block count
  packed and activation strides
  accumulator and loop state
  resource usage

QuantForwardSemantics
  packed payload planes
  scale/minimum or signed-scale fields
  high-bit reconstruction
  post-WMMA correction semantics
```

`ForwardResourceUsage` is the shared authority for writer metadata, capability admission, and artifact inspection. Resource limits are candidate constraints; VGPR, SGPR, LDS, private-segment, and spill outcomes are derived facts.

For gfx1151, admission and inspection account for the 64 KiB workgroup LDS ceiling and wave32's 24-VGPR allocation granularity. A logical high-water register index is not substituted for the allocated resource count reported in kernel metadata.

### Canonical candidates

Complete candidates round-trip through normal serialized solution inputs. Canonical hashes exclude problem shape so a candidate may be tested on another formula-compatible shape; exact-pair manifests preserve the shape-specific evidence. Family-inactive legacy fields reject before hashing.

## Lowering Architecture

### Shared assembly infrastructure

`kernel_writer_assembly.py` owns the direction-neutral assembly module, ROCISA setup, source writing, pointer and address primitives, BF16 RNE emission, metadata/trailer emission, and deterministic register-pool mechanism. Direction writers own their algorithms and do not forward through compatibility writers.

### Forward lowering

The current forward writer has three implementation families:
- direct-global Q4_K for the small direct tile.
- shared decoded-weight-LDS Q4_K/Q5_K for the staged `128x64` family.
- structured Q6_K with single-row and dual-row wave ownership; M256 reuses two exact dual-row tiles.

All forward families consume Q8_1 bytes and metadata produced upstream. Weight decode, activation-workspace staging, integer dot, correction, and output conversion are distinct semantic responsibilities even when a selected physical schedule overlaps their instructions.

Q4_K/Q5_K use `F16_D4S4` activation metadata and scale/minimum correction. Q6_K uses `F32_D4`, signed six-bit values, signed int8 scales, and block-factor correction. These arithmetic differences remain typed quant semantics rather than conditionals scattered through orchestration.

### Backward lowering

Backward retains one geometry-derived allocation, reduction pipeline, WMMA lowering, and store path. Quant-specific global-read and decoder leaves produce the common BF16 weight representation. This convergence point is why five backward formats can share a smaller common body without forcing the forward integer-WMMA algorithm into the same design.

### Semantic stages and scheduling

Lowering first constructs named stages:

```text
Setup
GlobalRead
Decode
LocalWrite
Barrier
LocalRead
Dot
ScaleAccumulate
LoopCommit
Epilogue
```

Operations carry semantic coordinates such as row, payload plane, decode atom, K phase, output tile, LDS pair, and store batch. Dependencies are checked before emission.

The second level applies explicit deterministic policies for traversal, clustering, lookahead, local-write placement, local reads, dot grouping, epilogue scope, VOPD pairing, delays, clauses, and cache behavior. Policies are complete mechanism choices, not generic heuristic scheduling. Waits derive from typed producers and first-use boundaries; redundant weaker waits are suppressed monotonically.

### Register allocation and physical roles

Register roles declare width, alignment, lifetime, reuse class, ownership, and deterministic role order. `DeterministicRegisterPlan` uses lifetime-aware first fit with no optimization, repair, or insertion-order fallback. `DeterministicRegisterPool` supports explicit checkout/checkin reuse at known last-use boundaries.

Irregular selected assignments may remain pinned when they are measured ownership facts. Regular accumulators, outputs, pointers, decode values, LDS pairs, and scratch values derive from formulas or pool allocation. Lowering receives assignments; it does not discover pressure while emitting.

### Q6 semantic boundary

Q6 uses typed payload/address roles, 16 semantic decode atoms, explicit low/high extraction, signed-byte normalization, formula-derived LDS roles, producer-first-use VMEM waits, typed dependency delays, deterministic source-to-output register reuse, and shared dot/refill/epilogue orchestration.

The residual single-row/dual-row setup, near/far read, activation-read, and refill leaves preserve selected physical traversal where no complete semantic ownership formula has yet replaced that order. They are bounded policies, not duplicate full kernels. Replacing them is deferred until Q3_K and Q8_0 forward evidence clarifies the correct cross-format abstraction.

### Prohibited forward production mechanisms

The refactored forward lowering may not use:
- raw assembly templates or large assembly-string collections.
- sibling forwarding writers or compatibility aliases.
- external instruction data, opcode tables, rank tables, or absolute issue maps.
- learned rankers, opaque cost models, repair logic, or hidden ready-list priorities.
- source-order or insertion-order fallbacks.
- build-time HIP/LLVM scheduling or allocation.
- post-emission textual filtering or VOPD reconstruction.
- accepted fields that affect neither lowering nor validation.

Backward currently retains its direction-specific deferred-zero/VOPD assembly emitter. That selected backward mechanism is not a precedent for reintroducing textual scheduling into forward lowering; changing it belongs to a separate backward structural project with its own identity and performance gates.

## Tuning and Search

Only values with implemented distinct lowerings may enter a candidate domain. Fixed arithmetic, ABI, ISA, activation layout, and format facts are contracts rather than genes.

The main linked tuning groups are:
- geometry and ownership.
- global-memory widths, assignment, clustering, prefetch, and cache policy.
- LDS representation, buffering, layout, swizzle, and plane placement.
- decode traversal, lane sharing, grouping, lookahead, and write deferral.
- iteration, local reads, dot grouping, and activation/weight overlap.
- epilogue traversal, dependency width, scope, priority, and stores.
- explicit VOPD, delay, clause, and cache policies.
- resource limits and desired occupancy.

Repository-owned search tooling lives outside the writer and operates on complete candidates:

```text
candidate_domains(quant_type, shape)
candidate_neighbors(seed, knob_group)
explain_invalid(candidate, shape)
canonical_candidate(candidate)
```

Search explores linked neighborhoods rather than a broad Cartesian product. Generation never benchmarks or selects. LLVM, HIP, TensileLite, EvoTensile, CK, and hipBLASLt may provide mechanism vocabulary or offline evidence, but cannot supply production instruction order, allocation, validity, or winner selection.

Exact-shape constants, fixed trip counts, peeled tails, affine-address reductions, register-lifetime shortening, and legal VOPD formation are derived lowering work rather than public knobs unless complete alternate mechanisms are implemented. Their value is judged by the same correctness, resource, and timing gates as larger policies.

### Q8_0 performance campaign (initial control complete; parity work reopened)

The Q8_0 forward campaign was reopened as an isolated performance experiment. Its HIP fallback decisions remained the production control throughout, and the completed GGTensile comparison below did not meet the promotion gates.

The first mechanism is a typed HIP-shaped `I=64, J=128` workgroup: 128 wave32 threads, cooperative Q8_0 payload/scale decode into LDS, reusable Q8_1 activation rows, 32 integer WMMAs per reduction stage, and a denser output ownership/scale epilogue. This is a new dataflow boundary, not a schedule-only variation of the retained `128x32` register tile. The lowering must derive its LDS layout, ownership, local-read coordinates, accumulator roles, and resource count from the semantic tile contract; it must not copy the HIP instruction stream or import a physical schedule.

The campaign proceeds in measurable gates:
- document the exact HIP tile mapping and static work/resource floors from the source and disassembly.
- implement one isolated ordinary Q8_0 control through typed Q8 roles and semantic LDS stages, then assemble, inspect, and validate it before timing.
- match the HIP operand reuse and epilogue ownership while preserving the established integer-result, weight-scale, activation-scale arithmetic order and BF16 results.
- screen the ordinary Q-A control with warmed rotating HIP comparisons, then confirm any gain on Q-B, attention-output B, shared gate/up, shared down, KV, and LM-head chunks before changing exact-key decisions.
- require parity or better on every promoted exact key, strict zero-spill/resource/ABI checks, mutation and independent-reference correctness, and deterministic rebuilds.

Only a measured complete dataflow may replace a fallback. The initial HIP-shaped `Q8HipTiledLds` lowering was exact and resource-qualified but missed its Q-A gate; the activation-read-address-hoisted form subsequently cleared two 25-repeat confirmations for 20 exact keys. The Q8 catalog now selects that form for 20 keys and retains HIP fallback for KV M2048 and LM-head M32/M64. This catalog promotion does not change public dispatch or the generated bundle; the initial Q8 campaign remains historical evidence and the public-boundary review is separate.

### Initial Q3_K forward exact-shape control (isolated)

The isolated forward Q3_K control is qualified only for the ordinary exact shape `(M,N,K)=(2048,4096,2048)`, representative tensor `blk.4.attn_gate.weight`. It uses the 110-byte Q3_K block format, Q8_1 `F32_D4` activations, a wave32 `(32,4,1)` workgroup, a `128x64` macro tile, decoded Q3 rows in LDS, integer WMMA accumulation, FP32 scale correction, and BF16 RNE stores. The final shared-VMEM plus loop-carried weight-prefetch lowering is resource-clean at 144 VGPRs, 16 SGPRs, and 28,672 bytes of LDS, with zero private storage and spills.

The control is bit-exact with the direct HIP control and public path across 8,388,608 outputs, passes input/weight/workspace mutation and independent-reference checks, and reproduces source, object, HSACO, and normalized inspection content across two independent builds. The final warmed 25-sample medians are `0.9380x` HIP for multiply and `0.9260x` for complete execution. It remains isolated research evidence because Q3 inventory, selected catalog, runtime dispatch, and public bundle promotion are still deferred pending broader shape validation and a separate integration decision. Evidence is in [experiment_ggtensile_mmq_fwd_q3_k.md](experiment_ggtensile_mmq_fwd_q3_k.md).

### Active Q8_0/Q3_K parity reopening

The forward parity campaigns are reopened under an explicit premise: the existing HIP kernels demonstrate a feasible complete dataflow, so a GGTensile implementation must continue until the remaining in-contract mechanisms are either measured or classified with evidence. The Q8 catalog phase is now complete for its 20 promoted and three fallback keys. Public runtime dispatch, public bundles, and the frozen-source boundary remain separate integration decisions.

The active order is:
- Establish an exact HIP-shaped mapping and resource/performance floor for each format, then identify the largest non-excluded ownership or dataflow difference rather than repeating already rejected schedule-only variants.
- Optimize Q8_0 across the 23 exact keys, starting with the remaining decode, operand-reuse, LDS, and epilogue gaps; retain HIP fallback until every promoted key clears its exact timing gate.
- Optimize the isolated Q3_K ordinary control first, then expand only when a complete mechanism is correct, resource-clean, deterministic, and faster than HIP on the first exact shape.
- Use warmed rotating measurements and independent controls. Static instruction reductions, partial-body timings, and compiler scheduling observations are diagnostic only.

The recursive final review remains mandatory and is the final step of the reopened campaign. No campaign may be declared complete immediately after discovering an actionable mechanism. Every actionable finding must be implemented and measured, then followed by a fresh review that classifies all remaining ideas and explains the residual bottleneck.

### Diagnostic lower bounds and profiling

When a bottleneck is ambiguous, exact diagnostic kernels may isolate matrix/activation/LDS work from packed decode/LDS work while preserving launch geometry and declared resources. They pass the same symbol, ABI, resource, and forbidden-storage inspection as candidates. Hardware counters and normalized disassembly explain first-order limits but never select winners; unsupported or over-capacity counter requests remain recorded evidence rather than silently reduced measurements.

## Artifact Lifecycle

Generation, build, inspection, correctness, screening, and confirmation are separate immutable phases:
- `generate` validates one exact key, emits source or structured rejection, and records the source hash.
- `build` verifies an approved generation manifest and invokes the assembler and linker.
- `inspect` verifies symbol, ABI, ISA, metadata, resources, and forbidden storage or instructions.
- `correctness` runs only inspected artifacts against the operation control and independent references.
- `screen` ranks qualified candidates under warmed rotating controls.
- `confirmation` retimes finalists under a fresh, longer rotating protocol.

Each phase refuses to overwrite an existing artifact. Manifests are strict. Source is the primary generated content identity; object and code-object identities are deterministic translations verified by rebuild and inspection.

## Verification and Promotion

### Structural migrations

When behavior is intended to remain unchanged, compare generated source, executable text, metadata, symbols, resources, and code objects exactly. Preserve ABI, launch ownership, waits, VOPD pairings, LDS offsets, barriers, and resource envelopes.

### Deliberate stream changes

A deliberate instruction-stream change requires:
- finite output and exact control agreement when arithmetic order is unchanged.
- an independent dequantized or independently formulated reference.
- input, packed-weight, and activation-workspace mutation sensitivity.
- deterministic source and code-object rebuilds.
- strict ABI, ISA, VGPR, SGPR, LDS, spill, and private-storage inspection.
- representative and blind formula-compatible shapes where shared lowering changes.
- retained-parent comparisons and warmed GPU confirmation.

Static inspection never substitutes for execution correctness.

### Measurement and retention

GPU timing is serial, warmed, and rotating, with the same tensors and launch conditions. Reports preserve raw samples, medians, dispersion, paired comparisons, protocol identity, and normalized assembly around finalists.

The standing gates are:
- zero correctness failures.
- byte-identical independent generation and rebuild.
- zero private storage, spills, scratch, calls, or dynamic stack.
- no stable regression above 1% on another exact key sharing the emitted path.
- a stable gain above 2% for a resource-bearing mechanism.
- neutral-to-favorable representative timing for an unconditional resource-neutral reduction.

Timing selects winners. Static issue counts, counters, code size, locality, and resources explain results but do not promote candidates by themselves. Weighted workload totals guide effort and reporting; they never authorize a slower exact key.

## Current Progress

### Coverage status

| Area | Current status |
| --- | --- |
| Shared generator, toolchain, inspection, runtime, and campaign infrastructure | Implemented for the current gfx1151 exact-key workflow |
| MMQ backward | Required `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0` campaigns complete; 50 exact inventory keys are selected |
| MMQ forward Q4_K | Lowering and 12-key research coverage implemented; the canonical inventory currently records 2 selected and 10 open keys |
| MMQ forward Q5_K | Six exact inventory keys selected and recursively exhausted under the current contract |
| MMQ forward Q6_K | Three exact language-model-head keys selected; structured semantic lowering and the current writer refactor are complete for the implemented domain |
| MMQ forward Q3_K | One exact ordinary control is correctness-qualified and faster than HIP with the retained shared-VMEM/prefetch dataflow; inventory and selected catalog remain deferred |
| MMQ forward Q8_0 | 20 exact catalog keys select the activation-read-hoisted HIP-shaped LDS control; KV M2048 and LM-head M32/M64 retain HIP fallback; public integration is deferred |
| Grouped GGTensile | Deferred until grouped ownership and routing receive an explicit generator contract |
| Public GGTensile runtime selection | Deferred; existing HIP bundle dispatch remains authoritative |

Global MMQ forward format exhaustion is not complete until Q3_K and both ordinary and language-model-head Q8_0 are qualified. Ordinary Q8_0 coverage will not imply fixed-group Q8_0 coverage.

### Implemented forward architecture

The completed Q4_K/Q5_K/Q6_K refactor established:
- strict `ForwardProblemContract`, complete `ForwardKernelSpec`, `DerivedForwardState`, quant semantics, and formula-derived resource usage.
- formula-based divisible-shape capability separated from exact inventories and winners.
- canonical candidate and exact-pair manifests with linked external search neighborhoods.
- a shared decoded-LDS Q4_K/Q5_K pipeline and semantic packed scale/minimum and Q5 high-bit reconstruction.
- one structured Q6 orchestration with typed setup/read/decode/LDS/dot/refill/epilogue boundaries.
- 16 semantic Q6 decode atoms with deterministic lifetime-aware register reuse.
- typed address, payload, LDS-pair, accumulator, product, output, wait, and dependency-delay roles.
- deterministic allocation and formula-derived VGPR, SGPR, and LDS admission.
- removal of flat Q6 bodies, raw templates, forwarding writers, post-emission scheduling, inactive fields, and physical decode leaves.
- structural tests for forbidden schedule representations, ignored fields, raw operands, legacy reads, and pass-through helpers.

The retained residual Q6 setup/read/refill traversal is intentionally unchanged until the post-Q3/Q8 convergence review can define or reject a complete semantic ownership replacement. The active work and its entry criteria are in `plan_ggtensile_asm_writer_refactor.md`.

### Latest qualified verification snapshot

The latest completed forward-refactor checkpoint records:
- 284 focused GGTensile tests and 369 repository tests passing, with only the existing Python 3.14 PyTorch deprecation warnings.
- Ruff, formatting, `ty check`, `compileall`, pre-commit, bundle currency, and `git diff --check` passing.
- the gfx1151 MMQ bundle current at 179 kernels.
- Q6 MT64 resources of 158 VGPRs, 27 SGPRs, and 28,928 bytes of LDS.
- Q6 MT128/M256 resources of 210 VGPRs, 27 SGPRs, and 38,400 bytes of LDS.
- zero private storage and zero VGPR/SGPR spills for the selected Q6 artifacts.
- exact HIP/public agreement, finite output, independent-reference qualification, mutation sensitivity, deterministic rebuilds, blind divisible-shape coverage, and warmed confirmation for the selected Q6 paths.

Exact source identities and normalized executable/code-object checks remain in artifact tests and experiment evidence rather than this generic design document.

## Experiment Records

Experiment records own exact problem scopes, timing tables, rejected mechanisms, debugging history, resource details, evidence paths, and campaign closure.

Completed MMQ backward records:
- `experiment_ggtensile_mmq_bwd_q3_k.md`
- `experiment_ggtensile_mmq_bwd_q4_k.md`
- `experiment_ggtensile_mmq_bwd_q5_k.md`
- `experiment_ggtensile_mmq_bwd_q6_k.md`
- `experiment_ggtensile_mmq_bwd_q8_0.md`

MMQ forward records:
- `experiment_ggtensile_mmq_fwd_q4_k.md`
- `experiment_ggtensile_mmq_fwd_q5_k.md`
- `experiment_ggtensile_mmq_fwd_q6_k.md`
- `experiment_ggtensile_mmq_fwd_q3_k.md`

Those documents retain campaign chronology and may describe historical premises that were later superseded. This document is authoritative for the current generic architecture and coverage status; selected catalogs and artifact tests are authoritative for current exact identities.

## Integration and Expansion

Each generated artifact owns one exact problem symbol. Runtime selection may use an artifact only when every problem type, exact size, ABI, architecture, solution, and source-identity assertion matches. Every mismatch falls back to the existing implementation.

Public GGTensile integration remains deferred until a useful production set is selected, packaging and identity are stable, exact dispatch engineering is complete, and end-to-end workloads pass correctness and weighted performance validation. Experimental force controls are not public policy.

Grouped GGTensile remains a separate design project. Paired routed, single routed, row-task, and fixed-group ownership require explicit routing, task, memory, synchronization, and fallback contracts. Ordinary coverage of an overlapping format does not count as grouped coverage.

Prepared weights, compact alternate public layouts, BF16 shadows, paired projections, persistent workgroups, split reduction, GSU, Stream-K, and multi-kernel fixup require model-visible ownership, lifetime, invalidation, memory accounting, ABI, and fallback design. They are not hidden extensions of the current exact single-kernel backend.

Bounded scan scripts may construct, cache, validate, and rank complete exact candidates outside the direction writers. A larger automated search system remains optional; deterministic `SolutionKey` identity, explainable rejection, immutable artifact phases, and reproducible evidence remain mandatory.
