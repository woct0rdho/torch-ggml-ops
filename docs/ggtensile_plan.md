# GGTensile Design and Progress

## Purpose

GGTensile is the repository-local assembly kernel generator for packed GGUF matrix multiplication. It accepts one exact `ProblemType`, one exact `ProblemSize`, and one complete direction-specific solution, then either emits reproducible assembly or returns a structured rejection reason.

This document is authoritative for the generic design, implementation principles, supported scope, and current project progress. Format-specific problem definitions, measurements, rejected mechanisms, artifact identities, and campaign chronology belong in the experiment records. The completed forward-writer migration and its phase evidence are recorded in `plan_ggtensile_asm_writer_refactor.md`.

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

Every campaign and structural refactor ends with a fresh recursive review of the plan, implementation, generated artifacts, selected and rejected evidence, target ISA, and related kernel work. The review is global across forward and backward directions, problem types, quant types, and exact shapes; a local completion statement is scoped to the contract and inventory it actually qualified, not a waiver for related work. Findings are classified as duplicate or closed, contract-incompatible, unsupported, deferred with an explicit prerequisite, or actionable.

An actionable finding must be implemented and qualified before the review is repeated. Completion is valid only when a fresh global pass finds no actionable in-contract mechanism. A result from one direction, quant type, or shape may transfer as evidence to another, but never as an automatic selection: the receiving semantics, ownership, lifetimes, synchronization, arithmetic order, resources, and exact-key gates must be re-derived and tested.

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

### Deployment catalogs

`tools/ggtensile/configs/` contains one `mmq_<direction>_<quant>_catalog.json` deployment file per supported direction and quant type. Each file has only three root fields: the canonical `ProblemType`, a deduplicated list of complete `Solutions`, and `ExactLogic` entries that map an exact `ProblemSize` to a solution index. Every listed solution must be referenced by at least one exact key. An absent key has no GGTensile deployment decision and falls back outside this catalog.

Deployment catalogs do not contain model names, tensor labels, family names, call counts, benchmark medians, experiment status, historical controls, candidate names, or rejected alternatives. Experiment chronology and current research status belong in the corresponding Markdown record. Immutable benchmark reports may contain their own workload and timing context, but they are evidence rather than deployment logic.

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

ForwardMechanismContract
  lowering/data-contract compatibility
  activation and weight block domains
  formula-derived reduction granularity

ForwardPhysicalPlan
  concrete mechanism layout and register roles
  deterministic assignments and lifetimes
  sole VGPR, SGPR, LDS, private, and spill usage
```

`ForwardResourceUsage` is the shared authority for writer metadata, capability admission, and artifact inspection. Resource limits are candidate constraints; VGPR, SGPR, LDS, private-segment, and spill outcomes are derived facts.

For gfx1151, admission and inspection account for the 64 KiB workgroup LDS ceiling and wave32's 24-VGPR allocation granularity. A logical high-water register index is not substituted for the allocated resource count reported in kernel metadata.

### Canonical candidates

Complete candidates round-trip through normal serialized solution inputs. Canonical hashes exclude problem shape so a candidate may be tested on another formula-compatible shape; exact-pair manifests preserve the shape-specific evidence. Family-inactive legacy fields reject before hashing.

## Lowering Architecture

### Shared assembly infrastructure

`kernel_writer_assembly.py` owns the direction-neutral assembly module, ROCISA setup, source writing, pointer and address primitives, BF16 RNE emission, metadata/trailer emission, and deterministic register-pool mechanism. Direction writers own their algorithms and do not forward through compatibility writers.

### Forward lowering

`ForwardKernelWriterAssembly` is a small public facade. It validates one key, constructs `DerivedForwardState`, emits the code-object envelope and ABI, and performs closed dispatch. It owns no decode, LDS, WMMA, wait, or epilogue body.

Pure planning in `mmq_fwd_physical.py` returns one concrete physical-plan type. Mechanism lowerers are organized by mechanism or data contract:
- packed-three-bit half-tile LDS lowering for the serialized Q3 control.
- direct packed scale/minimum lowering.
- decoded-weight LDS plus `F16_D4S4` activation lowering shared by compatible low-nibble and high-bit formats.
- structured signed-six-bit lowering with single-row or dual-row ownership.
- signed-int8 direct, register-tiled, wave-N LDS, exact small-M, and compact-KV physical mechanisms.

`F16D4S4ActivationMetadata` owns the shared activation group formula. Packed scale/minimum reconstruction is shared only by compatible correction paths, and one source-identical signed-int8 WMMA constructor is shared without merging ownership or post-WMMA arithmetic. Serialized operand-source names remain stable compatibility values even where internal classes use mechanism names.

All forward families consume Q8_1 bytes and metadata produced upstream. Weight decode, activation staging, integer dot, correction, waits, and output conversion remain separate semantic responsibilities when ownership or dependency order differs.

### Backward lowering

Backward retains one geometry-derived allocation, reduction pipeline, WMMA lowering, and store path. Quant-specific global-read and decoder leaves produce the common BF16 weight representation. This convergence point is why five backward formats can share a smaller common body without forcing the forward integer-WMMA algorithm into the same design.

### Semantic stages and scheduling

There is no universal forward stage runner. Each mechanism owns orchestration and calls a shared component only when roles, dependencies, and arithmetic order are equal. Structured Q6 constructs named semantic stages such as:

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

Other mechanisms retain direct typed emitters where a stage graph would only repackage their instruction order. Operations carry semantic coordinates such as row, payload plane, decode atom, K phase, output tile, LDS pair, and store batch. Dependencies are checked before emission.

The second level applies explicit deterministic policies for traversal, clustering, lookahead, local-write placement, local reads, dot grouping, epilogue scope, VOPD pairing, delays, clauses, and cache behavior. Policies are complete mechanism choices, not generic heuristic scheduling. Waits derive from typed producers and first-use boundaries; redundant weaker waits are suppressed monotonically.

### Register allocation and physical roles

Register roles declare width, alignment, lifetime, reuse class, ownership, and deterministic role order. `DeterministicRegisterPlan` uses lifetime-aware first fit with no optimization, repair, or insertion-order fallback. `DeterministicRegisterPool` supports explicit checkout/checkin reuse at known last-use boundaries.

Irregular selected assignments may remain pinned when they are measured ownership facts. Regular accumulators, outputs, pointers, decode values, LDS pairs, and scratch values derive from formulas or pool allocation. Lowering receives assignments; it does not discover pressure while emitting.

### Q6 semantic boundary

Q6 uses typed payload/address roles, 16 semantic decode atoms, explicit low/high extraction, signed-byte normalization, formula-derived LDS roles, producer-first-use VMEM waits, typed dependency delays, deterministic source-to-output register reuse, and shared dot/refill/epilogue orchestration.

The residual single-row/dual-row setup, near/far read, activation-read, and refill leaves preserve selected physical traversal. The completed Q3/Q8 convergence review found no equivalent ownership contract, so these remain bounded Q6 policies rather than duplicate full kernels.

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

### Cross-campaign reopening policy

The forward records currently carry two bounded changed premises for later work: a typed invariant activation-LDS-base lifetime for Q4_K/Q5_K, and a composed Q8_0 depth32 compact-row dataflow using the later weight-first and paired-scale findings. The Q6_K scheduler-oracle behavior has been reconstructed as a typed row/role wavefront, qualified on all three exact keys, and promoted in the research catalog. These experiments do not change a contract, public dispatch path, or bundle.

The same review rule applies when a finding crosses a direction or format boundary. Shared vocabulary such as affine address hoisting, dependency-derived waits, load clustering, or explicit register lifetimes may be reused only after the receiving lowering proves the corresponding physical roles and resource envelope. A failed experiment must remain recorded with its failure reason, even when a later changed premise makes a bounded re-test reasonable.

### Q8_0 performance campaign (complete)

The Q8_0 forward campaign ran as an isolated performance experiment. HIP fallback remained the control until each exact GGTensile source cleared its qualification gates; the final research catalog selects GGTensile for all 23 required keys while public integration remains unchanged.

The first mechanism is a typed HIP-shaped `I=64, J=128` workgroup: 128 wave32 threads, cooperative Q8_0 payload/scale decode into LDS, reusable Q8_1 activation rows, 32 integer WMMAs per reduction stage, and a denser output ownership/scale epilogue. This is a new dataflow boundary, not a schedule-only variation of the retained `128x32` register tile. The lowering must derive its LDS layout, ownership, local-read coordinates, accumulator roles, and resource count from the semantic tile contract; it must not copy the HIP instruction stream or import a physical schedule.

The campaign used these measurable gates:
- document the exact HIP tile mapping and static work/resource floors from the source and disassembly.
- implement one isolated ordinary Q8_0 control through typed Q8 roles and semantic LDS stages, then assemble, inspect, and validate it before timing.
- match the HIP operand reuse and epilogue ownership while preserving the established integer-result, weight-scale, activation-scale arithmetic order and BF16 results.
- screen the ordinary Q-A control with warmed rotating HIP comparisons, then confirm any gain on Q-B, attention-output B, shared gate/up, shared down, KV, and LM-head chunks before changing exact-key decisions.
- require parity or better on every promoted exact key, strict zero-spill/resource/ABI checks, mutation and independent-reference correctness, and deterministic rebuilds.

Only a measured complete dataflow may replace a fallback. The initial HIP-shaped `Q8HipTiledLds` lowering was exact and resource-qualified but missed its Q-A gate; the activation-read-address-hoisted form subsequently cleared repeated confirmations for 20 exact keys. Exact `Q8SmallMTiledLds` controls then cleared the same qualification gates for LM-head M32/M64, and the compact M64 KV mechanism cleared its repeated faster-than-HIP gate. The Q8 catalog now selects GGTensile for all 23 required exact keys with no HIP research fallback. This catalog result does not change public dispatch or the generated bundle; the public-boundary review remains separate.

### Current Q3_K forward campaign

The dense Q3_K inventory contains 12 exact keys across attention K, attention Q, attention gate, and SSM output at `M={2048,8192,32768}`. The typed `Q3FullWeightTiledLds` lowering is independently qualified on every exact key. It uses the 110-byte Q3_K block format, Q8_1 `F32_D4` activations, wave32 workgroup `(32,4,1)`, a `128x64` macro tile, a 336-byte padded decoded-weight LDS row, integer WMMA accumulation, FP32 scale correction, and BF16 RNE stores. The research catalog selects the typed candidate for all 12 exact keys; public dispatch remains on HIP and the generated bundle remains at 179 kernels.

Two independent 100-warmup/101-repeat rotating confirmations favored GGTensile on every exact key. Per-key speedups were between `1.0700x` and `1.0896x`, and call-count-weighted speedups were `1.0745x` and `1.0755x`. All 12 keys passed exact HIP/public agreement, independent-reference, finiteness, mutation, deterministic-producer, deterministic-rebuild, and zero-spill/resource gates. The experiment record reports the corresponding effective TFLOPS-equivalent rates.

The recursive Q3 review is complete. MT64, rolled, linear-staging, batched-WMMA, partial-wait, paired-scale, compact-row, extra-barrier, cache-invalidation, dual-plane, phased-correction, fragment/bank, paired-sum, and final-tail alternatives were either correctness/resource-invalid or failed repeated timing. Prepared weights, dense shadows, external decode storage, split-K, persistent/grouped traversal, producer fusion, hidden caches, and public dispatch changes remain outside the contract. No actionable in-contract optimization remains for the 12 exact research keys.

### Current Q8_0 forward status

Q8 production coverage is complete for all 23 required exact keys: 20 ordinary wave-N LDS selections, two exact LM-head small-M selections, and one compact-KV selection. Every selected key has repeated warmed evidence below HIP multiply and is integrated through exact public bundle dispatch.

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
- repeated warmed evidence that every selected exact GGTensile multiply is faster than its exact HIP control.
- retained-parent comparisons for every deliberate stream change, with slower candidates rejected regardless of static resource or instruction reductions.
- representative checks on every exact key sharing an unconditional changed stream.

There is no fixed percentage threshold. Timing selects winners. Static issue counts, counters, code size, locality, and resources explain results but do not promote candidates by themselves. Weighted workload totals guide effort and reporting; they never authorize a slower exact key.

## Current Progress

### Coverage status

| Area | Current status |
| --- | --- |
| Shared generator, toolchain, inspection, runtime, and campaign infrastructure | Implemented for the current gfx1151 exact-key workflow |
| MMQ backward | Required `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0` campaigns complete; 50 exact inventory keys are selected |
| MMQ forward Q4_K | Twelve exact research identities qualified; the typed activation-base lifetime is accepted, with the conditional loop-form review and public wiring deferred |
| MMQ forward Q5_K | Six exact inventory keys selected; the compact/high-bit body and shared Q4/Q5 activation-base lifetime are qualified, with public wiring deferred |
| MMQ forward Q6_K | Three exact language-model-head keys select the typed row/role wavefront; the default policy remains implemented, and public wiring is deferred |
| MMQ forward Q3_K | Dense 12-key inventory complete; all 12 exact keys select the typed full-weight research candidate, while public wiring remains deferred to the 179-kernel HIP bundle |
| MMQ forward Q8_0 | All 23 required exact keys select research GGTensile controls; public wiring remains deferred, and one bounded compact-depth32 cross-key review is open |
| Grouped GGTensile | Deferred until grouped ownership and routing receive an explicit generator contract |
| Public GGTensile runtime selection | Deferred; existing HIP bundle dispatch remains authoritative |

Global MMQ forward format exhaustion is not complete while any format record or cross-campaign review has an actionable in-contract premise. Ordinary Q8_0 coverage does not imply fixed-group grouped-Q8_0 coverage, and completion in one direction or quant type does not close a related direction or quant type automatically.

### Implemented forward architecture

The completed forward-writer refactor established:
- a 104-line `ForwardKernelWriterAssembly` facade with validation, source envelope, output writing, and closed mechanism dispatch.
- strict `ForwardProblemContract`, complete `ForwardKernelSpec`, `ForwardMechanismContract`, `DerivedForwardState`, quant semantics, and formula-derived resource usage.
- a closed union of concrete physical plans as the sole register, LDS, and resource authority.
- formula-based divisible-shape capability separated from exact inventories and winners.
- canonical candidate and exact-pair manifests with linked external search neighborhoods.
- packed-three-bit, packed scale/minimum direct, decoded-weight LDS, structured-Q6, and signed-int8 lowerer modules.
- shared `F16D4S4ActivationMetadata`, packed scale/minimum reconstruction, and a source-identical signed-int8 WMMA component at proven semantic boundaries.
- immutable signed-int8 tiled-LDS register and scale-layout projections at the shared stage, group, and store helper boundary.
- a shared decoded-LDS low-nibble/high-bit pipeline with a distinct high-bit reconstruction leaf.
- one structured Q6 orchestration with typed setup/read/decode/LDS/dot/refill/epilogue boundaries.
- 16 semantic Q6 decode atoms with deterministic lifetime-aware register reuse.
- typed address, payload, LDS-pair, accumulator, product, output, wait, and dependency-delay roles.
- deterministic allocation and formula-derived VGPR, SGPR, and LDS admission.
- removal of facade-local bodies, raw templates, forwarding writers, post-emission scheduling, inactive fields, and duplicate resource/decode authorities.
- structural tests for forbidden dependencies, ignored fields, raw operands, legacy reads, pass-through helpers, and complete lowerer/physical-plan line coverage.

The retained residual Q6 setup/read/refill traversal is intentionally mechanism-owned; the completed Q3/Q8 convergence review found no contract-equivalent replacement.

### Latest qualified verification snapshot

The latest completed forward-refactor checkpoint records:
- 203 focused forward tests, 306 GGTensile tests, and 391 repository tests passing, with only the 14 existing Python 3.14 PyTorch deprecation warnings.
- all 516 baseline writer sources and all 447 preserved sources byte-identical, plus three independent 516-source regenerations identical to the baseline and each other.
- Ruff, formatting, `ty check`, `compileall`, pre-commit, bundle currency, `git diff --check`, and complete writer-line coverage passing.
- the gfx1151 MMQ bundle current and reproducible at 179 kernels.
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
- `experiment_ggtensile_mmq_fwd_q8_0.md`
- `experiment_ggtensile_mmq_fwd_q3_k.md`
- `plan_ggtensile_asm_writer_refactor.md` (completed structural phase record)

Those documents retain campaign chronology and may describe historical premises that were later superseded. This document is authoritative for the current generic architecture and coverage status; selected catalogs and artifact tests are authoritative for current exact identities.

## Integration and Expansion

Each generated artifact owns one exact problem symbol. Runtime selection may use an artifact only when every problem type, exact size, ABI, architecture, solution, and source-identity assertion matches. Every mismatch falls back to the existing implementation.

Public GGTensile integration remains deferred until a useful production set is selected, packaging and identity are stable, exact dispatch engineering is complete, and end-to-end workloads pass correctness and weighted performance validation. Experimental force controls are not public policy.

Grouped GGTensile remains a separate design project. Paired routed, single routed, row-task, and fixed-group ownership require explicit routing, task, memory, synchronization, and fallback contracts. Ordinary coverage of an overlapping format does not count as grouped coverage.

Prepared weights, compact alternate public layouts, BF16 shadows, paired projections, persistent workgroups, split reduction, GSU, Stream-K, and multi-kernel fixup require model-visible ownership, lifetime, invalidation, memory accounting, ABI, and fallback design. They are not hidden extensions of the current exact single-kernel backend.

Bounded scan scripts may construct, cache, validate, and rank complete exact candidates outside the direction writers. A larger automated search system remains optional; deterministic `SolutionKey` identity, explainable rejection, immutable artifact phases, and reproducible evidence remain mandatory.
