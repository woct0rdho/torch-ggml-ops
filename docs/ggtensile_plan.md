# GGTensile generation and tuning design

## Purpose

GGTensile is the repository-local assembly generator for packed GGUF matrix multiplication. It turns one exact typed problem and one complete solution into deterministic gfx1151 assembly. It does not select a runtime kernel, inspect PyTorch tensors, allocate workspaces, package code objects, or implement public API dispatch. Those responsibilities are documented in `kernel_bundle.md`.

This document is authoritative for generation, exact identity, validation, physical planning, lowering, offline search, and promotion evidence. Format-specific measurements and rejected mechanisms remain in the `experiment_ggtensile_*.md` records.

## Scope

The current generator targets:
- gfx1151, wave32, WMMA V1, and code-object version 5.
- ordinary MMQ forward and input-gradient kernels.
- routed grouped forward and input-gradient kernels.
- fused paired grouped forward and input-gradient kernels.
- fixed eight-group Q8_0 forward and input-gradient kernels.
- authoritative packed GGUF weights decoded by the generated multiply kernel.
- BF16 activation and destination contracts.
- exact positive dimensions and explicit complete mechanism choices.

Current quant-format coverage is:

| Family | Formats |
| --- | --- |
| Ordinary forward/backward | `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0` |
| Grouped single forward/backward | `Q2_K`, `Q4_K`, `Q5_K`, `IQ2_S` |
| Grouped paired forward/backward | `Q3_K`, `IQ2_S`, `IQ2_XXS` |
| Fixed grouped forward/backward | `Q8_0` |

This is generator coverage, not a promise that every formula-capable shape is deployed. The exact public deployment matrix is in `kernel_bundle.md`.

GGTensile does not implicitly provide edge repair, nearest-shape selection, atomics, split reduction with fixup, Stream-K, persistent traversal, prepared weights, dense shadows, hidden caches, external decode workspaces, online tuning, or runtime compilation. A new mechanism requires an explicit problem model, ABI, physical plan, lowering, validation, inspection, and qualification campaign.

## Exact model and identity

Generation accepts immutable typed records:
- problem contracts define format, arithmetic, ABI, ISA, and fixed representation facts.
- exact problems define positive operation-specific dimensions.
- kernel specifications define every active mechanism choice.
- derived states compute geometry, strides, loop counts, ownership, and resources.
- exact solution keys combine contract, problem, and specification and own stable hashes and symbols.

Every canonical exact key contains `KernelFamily`, `Target`, `ProblemType`, `Problem`, and `KernelSpec`. `ProblemType` carries operation, data types, arithmetic, and transpose identity; `Problem` carries exact mathematical and family axes. Parameter-only candidates omit the exact problem.

The public record roles are explicit:
- `ProblemType` identifies operation, quantization, data types, arithmetic, and transposition facts used by ordinary construction and validation.
- `ProblemSize` carries exact positive operation coordinates.
- `ForwardKernelSpec`, `BackwardKernelSpec`, and the family-specific kernel-spec records are complete immutable mechanism choices; they are the canonical generation authority and are not serialized as separate deployment identities.
- `KernelInstance` is the exact contract/problem/kernel-spec selection; it is passive, while the family registry owns canonical mapping, stable hashing, naming, and family dispatch.
- Validation functions assert contract, capability, geometry, and resource invariants directly and return no rejection data.
- `KernelArtifact` joins one exact key to source/object/code-object paths, symbol identity, ABI, launch facts, and inspected resources.

### Schema and identity rules

Canonical serializers do not emit flat construction records or a descriptive document-kind tag. Exact-key, candidate, catalog, archive, and deployment loaders are separate entry points, so a loader never probes document shapes or catches a constructor failure to try another interpretation. A generic loader dispatches on the closed `KernelFamily` value only where that field belongs to the exact-key schema.

Every catalog root is exactly `KernelFamily`, `Target`, `ProblemType`, `KernelSpecs`, and `ExactLogic`. `KernelSpecs` is nonempty and contains distinct parameter-only specifications. Each `ExactLogic` entry is exactly a positive canonical `Problem` and an in-range `KernelSpecIndex`; problem sizes are unique, and every listed specification is referenced by at least one exact entry. The family, target, problem type, and specification must round-trip to the same typed lowering.

Deployment metadata is derived from catalog entries rather than stored in a second identity or inventory schema. Each selected entry contributes one exact operation/problem route; its identity, symbol, ABI, workgroup, grid, LDS size, ownership, and row-task bounds are derived from that key. Timing reports, model names, rejected candidates, and experiment chronology remain outside deployment data.

Parsing is strict:
- unknown and missing fields reject.
- JSON booleans and integers are not coerced.
- finite policy domains use exact enums.
- aliases, case folding, compatibility spellings, and inferred defaults are not accepted.
- inactive policy fields reject instead of being ignored.
- accepted fields must affect canonical identity and either validation or lowering.

Candidate identity is parameter-only so one candidate can be tested on another formula-compatible problem. Exact-key identity includes the exact problem. Generated source and code-object digests are artifact evidence, not candidate parameters. Exact-key and candidate hashes are computed directly from their canonical mappings without a document-kind or compatibility projection.

Serialized schema changes are one-time migrations: replace the old schema and update all checked-in catalogs, manifests, tests, and artifact references together. There is no compatibility translation or dual parser; old documents are rejected at the current loader boundary. A migration must preserve or deliberately requalify source, executable, ABI, resource, correctness, and timing identity rather than silently accepting two shapes.

`schema.py` owns the shared strict mapping, scalar, tuple, boolean, enum, and canonical-value primitives. Family loaders own their complete outer schemas and perform enum conversion once at the canonical contract/specification boundary. Named in-memory constructors are test and experiment conveniences only; they are not parser defaults, capability whitelists, or admission authorities.

Finite serialized domains use one exact spelling and strict JSON types. Boolean fields remain JSON booleans; a group of booleans that jointly selects a mechanism is represented internally by one typed policy, and helpers that need an independent boolean take it explicitly rather than by positional convention. No conversion may infer a nearby policy, merge distinct values into one lowering, or accept an inactive field.

Capability, candidates, and selection are separate:
- capability predicates describe implemented formula-supported combinations.
- candidate domains enumerate complete implemented policy points for offline search.
- canonical catalogs retain selected exact winners; deployment metadata is derived from them.
- benchmark reports and archive manifests are evidence and never runtime selection inputs.

The generator never repairs a rejected candidate or substitutes another selected solution.

## Generation pipeline

One generation follows this closed sequence:
- Parse or construct one exact typed key.
- Validate contract, linked policy values, divisibility, and target capability.
- Derive one canonical operation state.
- Derive one concrete physical plan with register roles, lifetimes, LDS layout, and resource usage.
- Dispatch to one substantial mechanism lowerer.
- Render one common `KernelEmissionPlan` with the exact ABI, target metadata, symbol, resources, body, and ordered trailing sections.
- Emit deterministic assembly text.

Expected contract failures raise `AssertionError` before emission. Programming, allocator, and toolchain failures retain their original exception rather than being converted into candidate filtering results.

`kernel_abi.py` is the sole authority for argument order, widths, signedness, aligned offsets, kernarg size, metadata projection, and research-runtime packing. Writers and lowerers do not maintain parallel ABI tables.

### Contract layers

Forward separates:

```text
ForwardProblemContract -> ForwardKernelSpec -> DerivedForwardState
                       -> QuantForwardSemantics
                       -> ForwardMechanismContract
                       -> ForwardPhysicalPlan
```

Forward consumes an explicit Q8_1 activation workspace produced outside the multiply kernel. Its lowerings decode integer weights, stage the selected activation layout, issue integer WMMA, apply format-specific correction, and convert to BF16.

Backward separately uses:

```text
BackwardProblemContract -> BackwardKernelSpec -> DerivedBackwardState
                        -> BackwardMechanismContract
                        -> BackwardPhysicalPlan
```

Backward reads BF16 output gradients, decodes packed weights to the common BF16 representation, and uses a shared BF16-WMMA pipeline. Forward and backward share only direction-neutral assembly and typed invariants; they do not share a universal tile pipeline.

Grouped families compose route ownership around the relevant arithmetic contract. Serial routes, static split routes, device row tasks, paired projection interleave, and fixed group-Z ownership are distinct typed policies. Runtime route values are not generator or tuning inputs.

### Responsibility boundaries

The current module ownership is:

| Module or family | Sole responsibility |
| --- | --- |
| `kernel_abi.py` | Argument order, names, value kinds, alignment, kernarg sizes, metadata projection, and runtime packing for every assembly ABI |
| `kernel_writer_assembly.py` | ROCISA setup, direction-neutral assembly primitives, deterministic register helpers, `KernelEmissionPlan`, and ordered source rendering |
| `kernel_writer_assembly_mmq_fwd.py` / `kernel_writer_assembly_mmq_bwd.py` | Ordinary forward/backward validation, closed lowerer dispatch, and public source envelopes |
| `mmq_fwd_spec.py` / `mmq_bwd_spec.py` | Direction-specific contracts, complete policies, canonical serialization, derived logical state, and capability predicates |
| `mmq_fwd_physical.py` / `mmq_bwd_physical.py` | One concrete physical-plan authority per direction for register roles, decoder/address state, LDS layout, lifetimes, and resources |
| `mmq_fwd_lowering*.py` | Ordinary forward mechanism orchestration, quant decode, staging, WMMA, correction, synchronization, and stores |
| `mmq_bwd_lowering_quant.py` / `mmq_bwd_lowering.py` / `mmq_bwd_emission.py` | Backward quant-specific readers, common BF16-WMMA pipeline, stores, and bounded typed VOPD formation |
| `grouped_mmq_fwd_model.py` / `grouped_mmq_fwd_spec.py` / `grouped_mmq_fwd_validation.py` | Routed forward identity, route bounds, typed policies, geometry, derived state, and validation rules |
| `grouped_mmq_fwd_physical.py` / `grouped_mmq_fwd_route.py` | Grouped LDS/register plans and the complete routed ABI prologue and expert rebasing |
| `grouped_mmq_fwd_lowering*.py` / `grouped_mmq_fwd_inspection.py` | Grouped mechanism emission, row dispatch, decode/correction, output, and plan-derived artifact checks |
| `grouped_mmq_fwd_pair_*` and its writer facade | Paired forward contracts, shared route/task ownership, physical plans, K128 mechanics, runtime packing, lowering, and inspection |
| `grouped_mmq_bwd_spec.py` / `grouped_mmq_bwd_physical.py` / `grouped_mmq_bwd_lowering.py` | Routed backward contracts, ownership/tail policies, physical plans, and grouped compute emission |
| `grouped_mmq_bwd_pair_*` and its writer facade | Paired backward contracts, pair physical plans, projection interleave, runtime controls, lowering, and inspection |
| `fixed_grouped_mmq_fwd_*` and `fixed_grouped_mmq_bwd_*` | Fixed eight-group Q8_0 contracts, fixed launch ownership, composed physical plans, lowering, runtime packing, and validation |
| `inspection.py` and family inspection modules | Artifact checks derived from typed keys and physical plans; no independent resource allocation or winner selection |
| `mmq_*_search.py` / `campaign.py` / `cli.py` | Complete-candidate construction, bounded neighborhoods, validity predicates, phase manifests, and offline evidence handling |
| `deployment.py` | Strict catalog-derived deployment validation |
| `bench/benchmark_common.py` / `bench/benchmark_routes.py` | Shared timing protocol, report summaries, and deterministic benchmark route construction |

Pure specification and physical-planning modules do not import ROCISA, invoke the toolchain, benchmark, or read selected inventories. Lowerers do not import catalogs, deployment state, model names, tensor names, or timing reports.

Every writer facade accepts one exact key, constructs one derived state and one physical plan, initializes the common code-object emission plan, and dispatches to one explicit lowerer. It owns neither decoder loops, register allocation, LDS offsets, waits, WMMA bodies, epilogues, nor resource formulas. Lowering results are immutable bodies plus ordered mechanism-owned trailing sections; `KernelEmissionPlan` renders them without knowing quant-specific details.

The routed and shared lowering boundaries are equally explicit:
- `GroupedRouteEmitter` owns the complete routed ABI prologue, launch guards, expert-ID and cumulative-offset loads, invalid-route inertness, and full-width expert-bank rebasing.
- `GroupedActivationStagingPlan`, `GroupedRowTileDispatchPolicy`, and `GroupedRowTileDispatchEmitter` own bounds-masked staging and the row-body topology derived from macro and tail geometry; that topology derives store-clause widths and masked tails.
- `GroupedPairRouteEmitter` and `GroupedPairRowTaskEmitter` own paired route/task setup. Paired lowerers consume their public operations and do not call sibling private emitters.
- `DecodedWeightLdsStageEmitter` is the common boundary only for compatible decoded-weight staging, scaled WMMA, and BF16 stores; ordinary and grouped orchestration retain separate contexts.
- Grouped backward derives primary and optional secondary tile state together from one contract/specification. Fixed grouped forward and backward derive fixed address and group-Z ownership directly from their own typed contracts rather than projecting through an ordinary key.

## Physical planning and lowering

Register roles declare width, alignment, lifetime, reuse class, ownership, and deterministic order. Lifetime-aware first fit and explicit checkout/checkin are deterministic allocation mechanisms, not optimizing schedulers. Lowering consumes assignments; it does not discover pressure, repair allocation, or fall back to source or insertion order while emitting.

Resource usage derives once and is shared by validation, writer metadata, and inspection. Accepted production candidates must fit the gfx1151 VGPR, SGPR, and 64 KiB LDS limits and must use zero private storage, spills, scratch instructions, calls, and dynamic stack.

A component is shared only when typed operands, ownership, producer/consumer boundaries, barriers, lifetimes, and arithmetic order are equivalent. Similar instruction spelling alone is not sufficient. In particular, Q3 signed-scale correction, Q4/Q5 scale/minimum correction, Q6 signed-scale correction, and Q8 scale multiplication remain distinct semantics.

Semantic phase names such as setup, global read, decode, local write, barrier, local read, dot, accumulation, loop commit, and epilogue describe ownership and dependency boundaries; they are not a stored universal stage graph or copied instruction schedule. Waits derive from typed producers and first-use boundaries, and deterministic policies may control traversal, clustering, lookahead, local-write placement, dot grouping, epilogue scope, VOPD pairing, delays, clauses, and cache behavior only when those policies select a complete implemented lowering.

The implementation deliberately avoids raw assembly templates, instruction tables, absolute issue maps, post-emission rewriting, sibling private-method borrowing, compatibility writers, and generic stage runners that merely store a selected schedule.

## Offline tuning and search

Generation is pure and never benchmarks or selects. Search operates on complete typed candidates outside writers and lowerers. Candidate domains may vary only policies with distinct implemented lowerings, such as:
- workgroup and macro-tile geometry.
- wave and route ownership.
- global-read width, clustering, prefetch, and cache policy.
- activation and decoded-weight LDS representation, buffering, padding, and swizzle.
- decode lane sharing, traversal, grouping, and overlap.
- WMMA iteration and local-read scheduling.
- epilogue traversal, dependency scope, priority, and store policy.
- explicit delay, clause, and legal VOPD policy.

Arithmetic, GGUF block layout, ABI field widths, ISA, route-entry bounds, and expert-ID/offset representation are contracts rather than tuning genes.

Search should explore bounded linked neighborhoods rather than an unreviewed Cartesian product. Every candidate value needs strict serialization, capability validation, a physical formula, a complete lowering, inspection expectations, and mutation coverage. A candidate domain is tooling coverage, not proof that optimization is exhausted.

Repository-owned search helpers construct complete typed candidates outside generation. Forward search exposes `candidate_domains`, `candidate_neighbors`, `is_valid_candidate`, and `canonical_candidate`; backward search exposes the corresponding domain, neighbor, validity, and canonical-mapping helpers. These higher-level search predicates catch `AssertionError` only to filter candidates through normal capability and physical validation. They preserve parameter-only identity and emit exact-pair evidence where a candidate is tested on more than one shape. Search helpers never benchmark, choose a deployment winner, or repair an invalid candidate.

The bounded domain is evidence about implemented coverage, not a proof of exhaustive optimization. A missing value becomes actionable work only when it identifies an in-contract mechanism with an exact target, plausible gain path, complete lowering, and qualification gate. Results can transfer between directions, formats, or shapes as evidence, but the receiving contract, ownership, lifetimes, synchronization, arithmetic order, resources, and exact-key gates must be re-derived.

Canonical candidates round-trip through contract/specification inputs and exclude exact problem shape from parameter identity. Only active mechanism policies serialize; derived geometry and inactive sentinels do not. Linked policies such as route split factor, row-task dimensions, secondary tails, decode dependency width, LDS buffering, paired interleave, and store scheduling are validated together, and numeric choices serialize as numeric fields rather than enum names that merely encode numbers. A new field is admitted only when it changes validation or lowering and has mutation, inspection, and behavioral coverage.

LLVM, HIP, TensileLite, Composable Kernel, and hipBLASLt may supply mechanism vocabulary or offline controls, but they do not supply production instruction order, allocation, validity, or winner selection. When a bottleneck is ambiguous, an exact diagnostic kernel may isolate matrix, activation, LDS, or packed-decode work while preserving the declared ABI, launch geometry, and resource checks. Counters and normalized disassembly explain limits; they never promote a slower exact key.

### Workload evidence

Ordinary kernels are timed on their exact shapes. Grouped kernels use the one fitted expert prior or prior law declared for their workload family, evaluated at the exact physical size. The normative fitting, profile-generation, and benchmark-input rules are in `ggtensile_workload_prior.md`.

A performance benchmark uses one deterministic route profile generated from that law for each exact problem key. Captured routes and language corpora are used only by the offline fitting job; benchmark tooling does not read them afterward. Medoid banks, weighted profile mixtures, alternate fitted laws, and synthetic distributions are not optimization inputs.

Alternate route shapes may still be exercised in non-timed correctness, malformed-route, tail, mutation, and ABI tests. They do not rank candidates, supply benchmark weights, or establish performance direction. Paired gate/up projections reuse the one canonical route profile while retaining separate packed-weight mutation checks and output references. Complete-call timing includes any explicit activation workspace and row-task setup required by the candidate contract.

### Qualification phases

Research artifacts advance through separate immutable phases:
- Generate - emit one exact source and record its canonical key and source digest.
- Build - assemble and link with pinned gfx1151/code-object-v5 options.
- Inspect - verify symbol, ABI, ISA, wave size, resources, and forbidden storage/instructions.
- Correctness - compare packed execution with an independent dequantized or separately formulated reference; test determinism, tails, sentinels, and input/weight/route mutations.
- Screen - rank qualified candidates with warmed rotating order and retained-parent controls.
- Confirm - retime finalists with fresh samples, longer runs, and reversed base order.

Each phase consumes a strict accepted manifest and refuses to overwrite an existing manifest, source, object, code object, or inspection result. Source is the primary generated identity; object and code-object identities are deterministic translations that must be checked by independent rebuild. Artifact records pin the target, assembler/linker, compiler identity, relevant flags, source and include digests, symbol, ABI, launch metadata, and inspected resources.

Benchmark reports retain raw timing samples, median and dispersion, paired candidate/parent comparisons, exact protocol identity, warmup and repeat counts, control order, and the route/profile inputs or their complete digest. Complete-call timing and preallocated-kernel timing are separate fields. Candidate and retained-parent measurements use the same frozen tensors, launch contract, and allocation rules; a report cannot be reconstructed from a summary median alone.

When a canonical field is added, removed, or renamed, record every old/new key and symbol mapping, compare source after symbol normalization, and compare executable text, metadata, ABI, and resources exactly. Update catalogs and manifests deliberately and rerun correctness, mutation, determinism, and forbidden-storage checks. An identity migration cannot conceal an instruction-stream change; any changed stream follows the full deliberate-change gates.

Structural moves should preserve source and executable identity. Canonical identity migrations may rename hashes and symbols only with normalized-source, executable, ABI, and resource equivalence. Any deliberate instruction-stream change must pass full correctness, independent rebuild, inspection, and timing gates.

Timing selects winners. Static counts, disassembly, counters, occupancy, and resources explain a result but never promote a slower exact key. There is no fixed percentage threshold; a selected kernel must show repeatable improvement under its declared exact workload and controls.

HIP kernels or other implementations may be used as offline correctness/performance controls. They are not generator dependencies and cannot act as a deployment fallback for a GGTensile key.

## Determinism and verification

The same exact key and toolchain must produce byte-identical source, object, and code object. Determinism requires stable canonical serialization, register-role order, labels, trailing-section order, toolchain flags, and source paths.

Tests enforce:
- strict canonical round trips and mutation of every accepted field.
- formula capability independent of selected exact keys.
- one physical-plan/resource authority per mechanism.
- no catalog, benchmark, deployment, or runtime imports from planning/lowering.
- no ROCISA/toolchain imports from pure specification modules.
- no duplicate ABI layouts or inspection-side resource allocation.
- complete writer-line coverage for every concrete mechanism.
- deterministic source and independent assembly/link results.
- gfx1151, wave32, code-object-v5 metadata and zero spills/private storage.

Repository gates include focused generator tests, the complete `tests/ggtensile` suite, the full project suite, pre-commit, and `git diff --check`. Deployment-bundle checks are separate and are described in `kernel_bundle.md`.

## Current generator organization

The implementation has dedicated model/specification, physical-plan, lowering, writer, inspection, and research-runtime modules for:
- ordinary forward and backward.
- grouped single forward and backward.
- grouped paired forward and backward.
- fixed grouped forward and backward.
- grouped forward device-row-task setup contracts.

All selected winners live in canonical catalogs under `tools/ggtensile/configs/mmq_*_catalog.json`. Research-only catalogs are kept outside the public catalog directory. Catalogs retain exact identity only; rejected candidates, timing data, model labels, and experiment chronology stay outside deployment configuration.

The experiment records under `docs/experiment_ggtensile_*.md` own per-format scopes, candidate history, benchmark protocols, resource observations, rejected mechanisms, and campaign closure. Historical statements in those records are evidence for their point in time; this document and the checked-in typed catalogs are authoritative for current generator architecture and selected identities.

## Adding or retuning a kernel

- Define or extend the typed problem contract and complete mechanism policy.
- Add strict parsing, canonical projection, and only simple formula, ABI, mechanism, or resource validation needed by the lowerer.
- Derive logical geometry and one concrete physical plan.
- Implement one substantial lowerer and exact ABI envelope.
- Add structural, serialization, mutation, source, and inspection tests.
- Search only complete implemented candidates outside generation.
- Qualify correctness, determinism, resources, and timing on every affected exact key.
- Retain only the measured winner in the selected catalog.
- Rebuild the deployment bundle and validate public execution as described in `kernel_bundle.md`.

A fresh cross-format review should classify remaining ideas as duplicate/closed, contract-incompatible, deferred with a prerequisite, or actionable. Actionable in-contract findings require implementation and qualification; results transfer between directions or formats only as evidence, never as automatic selections.
