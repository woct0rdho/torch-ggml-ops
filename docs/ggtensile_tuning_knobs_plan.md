# GGTensile tuning knobs implementation plan

## Purpose

This plan defines how to finish the useful TensileLite-style tuning surface in GGTensile while preserving the current generated kernels. It covers ordinary MMQ forward/backward, grouped MMQ, and paired grouped MMQ on gfx1151 wave32.

The plan is deliberately typed. A field is a tuning knob only when it is serialized, included in candidate identity, validated for the selected problem and quant format, reflected in resource inspection, and consumed by a distinct lowering or physical plan. Inert fields are not accepted.

"Consistent" means that equivalent behavior has one semantic name and one enum domain where possible. It does not mean that every operation or quant format must accept the same values. Quant-specific decoding, LDS ownership, route ownership, and arithmetic remain separate contracts.

## Alignment with the GGTensile design

This plan extends `ggtensile_plan.md`; it does not replace its ownership boundaries.

- `ProblemType` and the operation/quantization contract own arithmetic, representation, ABI, ISA, and fixed dimensions. Those are not tuning genes.
- A complete `ForwardKernelSpec`, `BackwardKernelSpec`, or family-specific grouped spec owns every active policy. A partial bag of optional fields is not a candidate.
- Derived logical state and one mechanism-specific physical plan own geometry, register roles, lifetimes, LDS offsets, synchronization requirements, and resource formulas.
- Lowerers consume typed policies and emit one complete mechanism. They do not search, repair invalid values, import catalogs or timing results, or fall back to source-order scheduling.
- Search helpers construct complete parameter-only candidates outside generation. They enumerate bounded linked neighborhoods, validate capability, and record exact-pair evidence; they never benchmark or select deployment winners.
- Catalogs contain selected canonical specifications and exact logic only. Benchmark reports, rejected candidates, deployment assignments, and experiment chronology remain evidence outside the generator.
- Shared components are permitted only when operands, ownership, producer/consumer boundaries, barriers, lifetimes, arithmetic order, and edge behavior are equivalent. Matching instruction spelling is insufficient.
- ABI, route bounds, expert ownership, packed GGUF layout, and output semantics remain contracts. A policy that changes them is a new mechanism and needs its own family contract, not another common knob.

A new policy must therefore flow through:

```text
complete typed spec -> derived state -> physical plan -> substantial lowerer
                    -> inspection and validation -> exact candidate identity
```

Pure specification and physical-plan modules must remain free of ROCISA, toolchain, benchmark, catalog, and deployment imports. Generation remains deterministic and side-effect-free apart from writing its requested artifact; tuning remains an offline activity.

## Evidence and scope

The strongest hypotheses come from FP16/TensileLite and EvoTensile. They guide candidate design but do not establish quantized-MMQ performance. Every quantized forward, backward, grouped, and paired result needs its own exact correctness and timing evidence.

Useful measured hypotheses are:
- `LdsPadB 16 -> 4` reached `+8.23%` on `m1024_n128_b1_k256`.
- `LdsPadA=4, LdsPadB=16` reached `+5.88%` on one target.
- `PrefetchGlobalRead 1 -> 2`, coupled with one `DepthU 32 -> 16` child, reached `+14.61%` in the 100-shape campaign.
- `DepthU=32`, `PrefetchGlobalRead=2`, and `ClusterLocalRead=0` was the strongest repeated staging interaction family.
- `NumElementsPerBatchStore 10 -> 20` helped selected `N=512-1024` shapes by `1-9%`, but lost on square and some `M=512` shapes.
- Vector-width changes produced shape-local gains and conditional validation failures.
- StreamK was highly shape-dependent and required a separate workspace and fixup contract. It is not part of this ordinary quantized-MMQ knob plan.

These values define starting experiments, not universal defaults.

## Compatibility rule

Existing specs and catalogs are frozen inputs. Generation from an unchanged existing spec must produce the same assembly source, symbol, resources, and artifact identity. This is a hard regression requirement.

New implementation fields must follow one of these rules:
- A legacy spec continues through the old lowering path and has no new serialized policy fields.
- A new candidate explicitly contains the new policy block and receives a new candidate hash and kernel identity.
- A schema migration is performed as one explicit operation with updated catalogs, manifests, tests, and artifact references. There is no silent compatibility projection or inferred nearby default.

A refactor may change internal types or introduce enums without changing emitted assembly. Any source change for an unchanged logical spec must fail the identity regression unless the plan records the reason and the affected catalog is deliberately requalified.

## Common tuning model

Use one semantic vocabulary across applicable operation families, but expose a field only where the selected lowering can consume it:

| Semantic group | Preferred fields | Type guidance |
| --- | --- | --- |
| Geometry | `WorkGroup`, `MatrixInstruction`, `DepthU` | Tuples and numeric `DepthU`; these are structural values, not enums. |
| Ownership | `MIWaveGroup`, `MIWaveTile`, `WorkGroupMapping`, `StaggerU` | Tuples and bounded numeric mapping/stride values. |
| Activation reads | `GlobalReadVectorWidthA`, `PrefetchGlobalRead`, `WaveSeparateGlobalReadA` | Width is a logical vector quantity; stage and producer ownership are enums. |
| Weight reads | `GlobalReadVectorWidthB`, `PrefetchGlobalRead`, `WaveSeparateGlobalReadB` | Generic B width applies only to dense-compatible data; packed payloads use the separate payload field. |
| Quantized reads | `PayloadGlobalReadVectorWidth`, `MetadataLoadVectorWidth`, `PackedLoadGrouping`, `MetadataPrefetch` | Physical transaction widths are numeric; grouping and prefetch are enums. |
| Staging | `ActivationStaging`, `WeightStaging`, `LdsBuffering` | High-level mechanism selectors are enums and replace/decompose lower-level source fields. |
| LDS | `LdsLayout`, `LdsPadA/B`, swizzle and block interval | Layout and swizzle are enums; pad and interval are numeric. |
| LDS writes | `PayloadLdsWriteVectorWidth`, `MetadataLdsWriteVectorWidth` | Physical bytes per issued LDS write, with format-specific alignment. |
| Local reads | `LocalReadVectorWidth`, `PrefetchLocalRead`, `ClusterLocalRead` | Width is numeric; stage and clustering modes are enums. |
| Iteration schedule | `ScheduleGlobalRead`, `ScheduleLocalWrite`, `ScheduleIterAlg` | Use named enums, never unexplained numeric algorithm IDs. |
| Decode ownership | `DecodeProducerCount`, decode traversal, dependency width | Producer count and dependency width are numeric; traversal is an enum. |
| Stores | `StorePriorityOpt`, `NumElementsPerBatchStore`, `StoreVectorWidth`, `StoreClauseWidths`, `StoreTraversal`, `StoreSyncOpt` | Logical widths/counts are numeric; priority, traversal, and sync are enums; clause widths are an ordered physical-width tuple. |
| Decode semantics | extraction, lane sharing, metadata placement, correction policy | Quant-family enums and bounded numeric dependencies. |

Suggested enum names are `SingleStage`, `DoubleStage`, `Single`, `Double`, `Default`, `InvalidateL0`, `Disabled`, `Enabled`, `Shared`, `WaveSeparated`, `Normal`, `Raised`, `Canonical`, `TileMajor`, `RowMajor`, `StoreBatch`, `FullTile`, and named schedules such as `DependencyOrdered` and `ExplicitPipeline`. Serialized names should follow established TensileLite names where useful, while Python APIs should expose enum members rather than raw integers or booleans when the value selects a mechanism. A renamed or newly canonicalized field requires one explicit schema migration; do not maintain a dual parser or silently translate old and new documents.

Numeric values remain numeric when they represent a real physical or logical quantity. The units must be part of the field contract: tile dimensions are elements, LDS padding and transaction widths are bytes, vector widths are either logical elements or physical bytes as explicitly stated below, and dependency/producer/batch counts are counts.

### Data-movement decomposition

The implementation must distinguish these planes instead of applying one unconstrained vector-width knob to every load and store:
- `GlobalReadVectorWidthA/B` is the logical element width for dense-compatible activation/operand reads. For BF16, values `1`, `2`, `4`, and `8` are the initial domain. It must not silently describe packed sub-byte payload bytes.
- `PayloadGlobalReadVectorWidth` is the physical byte width of each raw quantized payload transaction. Start with `B32`, `B64`, and `B128`, serialized as `4`, `8`, and `16` bytes or represented internally by a `PayloadReadWidth` enum. It controls emitted `global_load_b32/b64/b128`, destination allocation, payload address increments, and masking.
- `MetadataLoadVectorWidth` is the physical byte count fetched by one metadata global transaction, including bytes that require extraction after the load. Start with `2`, `4`, `8`, and `16` bytes, but admit only format-specific aligned widths. The field must not mean metadata elements or packed words.
- `PayloadLdsWriteVectorWidth` and `MetadataLdsWriteVectorWidth` are physical bytes per issued LDS write, initially `4`, `8`, and `16`. They must derive LDS offsets, bank-layout requirements, producer ownership, and wait counts.
- `StoreClauseWidths` is an ordered tuple of actual physical output store widths, such as `(B32, B64, B128)`. It is not a compound mechanism enum and is distinct from logical `StoreVectorWidth`.

The high-level mechanism selectors are intentionally not additive fields:

```text
ActivationStaging -> direct, single-LDS, double-LDS, or register-tiled activation ownership
WeightStaging     -> direct, decoded-LDS, full-weight-LDS, or register-tiled weight ownership
```

`ActivationStaging` must replace or decompose the existing `ActivationAddressing` choices. `WeightStaging` must replace or decompose `OperandSource`; it must not be added beside `OperandSource` as an independent selector. The exact enum members are family-specific where the physical contracts differ. Route ownership, paired projection ownership, fixed grouped ownership, and external workspaces remain outside these generic selectors.

### Producer ownership, metadata, and traversal

`WaveSeparateGlobalReadA` and `WaveSeparateGlobalReadB` select producer ownership, not a vector width. Their initial enum domain is `Shared` and `WaveSeparated`. `Shared` preserves the current producer assignment. `WaveSeparated` is legal only for a multi-wave workgroup with a physical plan that assigns each wave a disjoint read responsibility, address state, LDS destination range, and wait dependency. The two fields may differ, but combinations must be validated with the A/B LDS layout and cannot be enabled by a generic boolean in a one-wave plan.

`DecodeProducerCount` is the number of logical wave32 producer groups writing a decode destination, with an initial domain of `1`, `2`, and `4` where geometry permits. It changes payload/metadata destination ownership, LDS write grouping, barriers, wait thresholds, register roles, and edge masks. It is not equivalent to workgroup size or `WaveSeparated` global reads.

`MetadataPrefetch` has the enum domain `Disabled` and `Enabled`. `Enabled` is admissible only when metadata reads have a separate producer/consumer frontier from payload reads and the physical plan carries the required address and temporary state. It must not merely move the existing metadata load earlier in source order.

`StoreTraversal` describes the order in which logical output fragments become global stores. Start with `Canonical`, `TileMajor`, and `RowMajor`; add `ColumnMajor` only if a family has a concrete fragment transform. `StoreClauseWidths` describes the physical width of each emitted clause and must be derived consistently with traversal, masking, and address advancement. It cannot be hidden in a traversal or mechanism enum.

`PackedLoadGrouping` is the canonical name for quantized payload lane sharing/grouping. Its initial semantic values are `PerLane` and `LanePair`; an implementation may add `LaneQuad` only with a format-specific ownership proof. Existing `PackedWeightLaneShare=1/2` maps to these meanings during an explicit schema migration. New specs must not carry both fields.

## Existing knobs to make effective

### Geometry and ownership

Keep geometry common in name and validation, but derive its legal domain from the mechanism contract. `DepthU`, workgroup size, matrix instruction, macro tile, wave ownership, and register/LDS resources must be validated together.

A geometry candidate is valid only if the physical plan can derive all register roles, LDS offsets, waits, WMMA phases, edge behavior, and resource usage. Do not let a changed `DepthU` select an old plan with stale loop counts or stale pointer handoff.

`WorkGroupMapping` and `StaggerU` are launch-order controls. Keep them out of body-only lowering contracts and apply them only where grid shape and cache behavior make them meaningful. Grouped route/task mappings remain separate from ordinary mapping.

### Prefetch and pipeline controls

Backward already has active `PrefetchGlobalRead`, `PrefetchLocalRead`, activation prefetch, packed-weight prefetch, next-packed-weight prefetch, and decoded-LDS buffering. Preserve the current serialized values and emitted paths for existing specs. Internally, normalize the meanings to typed `SingleStage`/`DoubleStage` and `Single`/`Double` policies only when the canonical mapping and assembly remain unchanged. A future schema migration may replace awkward booleans or integers, but it must update every catalog and reject the old schema afterward.

Forward should gain a complete staged-loop policy rather than isolated booleans. It must own current and next global reads, LDS writes, local reads, decode first-use points, wait thresholds, and register lifetimes. A candidate with `PrefetchGlobalRead=2` must therefore lower a different complete pipeline.

The same semantic policy may be shared between operation types only when the producer/consumer graph and ABI ownership are equivalent. Forward activation staging and backward decoded-weight staging are not automatically equivalent. A shared policy type may be reused at the model boundary while each direction supplies its own physical-plan implementation and capability matrix.

### LDS layouts

Replace ad hoc layout strings with a typed `LdsLayout` policy where the selected family supports it. Candidate plans should derive row strides, address transforms, buffer size, and resource usage from the policy.

Initial forward probe values:
- `LdsPadA`: `0`, `4`, `8`, `16` where A is staged and the layout supports A padding.
- `LdsPadB`: `0`, `4`, `8`, `16` where B is staged and the layout supports B padding.
- `LdsSwizzleChunk`: `0`, `4`, `8`, `16` only for families with a proven swizzle address transform.
- `LdsBlockSizePerPad`: `64`, `128`, `256` only when padding repeats by a block interval.

Do not form a full Cartesian product. Start with the retained layout, one-axis changes, and the measured pair `A4/B16`. Reject values that change the packed representation, exceed 64 KiB, invalidate vector alignment, or require an unimplemented address transform.

Backward's existing `LdsPadB` and swizzle policies should use the same semantic layout vocabulary, while retaining direction-specific formulas. Its current `LdsPadB=8` and swizzle-8 winners remain unchanged until new candidates are qualified.

### Epilogue and stores

The ordinary decoded-LDS forward epilogue already has active `TilesAhead`, `DependencyWidth`, `Priority`, and `Scope`. Make their domains explicit and share the conceptual policy with grouped forward where the store ownership is equivalent. The lowerer must receive one typed epilogue policy and derive conversion order, address advancement, clauses, waits, and masking from it.

Initial values:
- `TilesAhead`: `1`, `2`, `4`, `8` for an eight-tile epilogue.
- `DependencyWidth`: `1`, `2`, `4`, `8`, constrained by store vector width and tile width.
- `Priority`: `Normal` and `Raised`.
- `Scope`: `StoreBatch` and `FullTile` where the lowering supports both.
- `NumElementsPerBatchStore`: `4`, `8`, `10`, `16`, `20`, `24`, `32` for emitters that can form those store batches.
- `StoreVectorWidth`: `1`, `2`, `4` logical output elements only when the output lane mapping and ABI store alignment support it.
- `StoreClauseWidths`: ordered physical clauses from `B32`, `B64`, and `B128`, subject to the output type and alignment.
- `StoreTraversal`: `Canonical`, `TileMajor`, and `RowMajor` initially.
- `StoreSyncOpt`: `Default`, `Reduced`, and `Grouped` only when each selects a different valid synchronization path.

`StorePriorityOpt=False` from the FP16 HHS result should be represented semantically as `Normal`, not as a generic false flag. Do not assume that `Normal` wins: the HHS result changed with `ScheduleIterAlg`, and quantized correction/store paths have different dependencies.

`NumElementsPerBatchStore`, `StoreVectorWidth`, `StoreClauseWidths`, and `StoreTraversal` are one linked store policy. The physical plan must account for output fragments, conversion temporaries, edge masking, clause alignment, and address advancement. A field that only changes a comment or candidate name is invalid. `StoreClauseWidths` must report the actual emitted byte widths rather than encode a compound mechanism name.

## New forward knob families

### Forward staging family

Add a typed `ForwardPipelinePolicy` for decoded-LDS and tiled quantized families. Its first bounded domain should include:
- `PrefetchGlobalRead`: `SingleStage`, `DoubleStage`.
- `PrefetchLocalRead`: `SingleStage`, `DoubleStage` only after an explicit local-read pipeline exists.
- `LdsBuffering`: `Single`, `Double` only for a complete ping-pong emitter.
- `ClusterLocalRead`: `Disabled`, `Enabled` where local-read clustering is implemented.
- `ScheduleGlobalRead`: `Default`, `Interleaved`, `WeightThenActivation` where supported.
- `ScheduleLocalWrite`: `Default`, `Grouped`, `Split` where supported.
- `ScheduleIterAlg`: `DependencyOrdered`, `ExplicitPipeline`; do not expose `2` or `3` without a named semantic definition.

The first useful combinations are:
- Retained geometry plus `DoubleStage` global reads.
- Retained geometry plus `DoubleStage` global reads and `DepthU=32 -> 16` where register/LDS limits permit it.
- The preceding combination with `ClusterLocalRead=Disabled`.
- One-buffer versus two-buffer LDS only when the alternate emitter has explicit barriers and resource accounting.

Probe each complete combination on exact parent-competitive shapes. Do not transfer a winning staging child to another quant type without validating its decode and LDS contracts.

### Forward LDS and vector family

Add `GlobalReadVectorWidthA/B`, `VectorWidthA/B`, and `LocalReadVectorWidth` only to lowerers that have a format-aware load ownership implementation.

Starting values are `1`, `2`, `4`, and `8` for dense logical global reads, `1`, `2`, and `4` for vector widths, and `8`, `16`, and `32` for local reads. Packed payload reads use `PayloadGlobalReadVectorWidth` and must not inherit this domain. The capability matrix must reject values that cross packed sub-byte fields, misalign `buffer_load` widths, or alter WMMA lane mapping without a corresponding transform.

The first vector probes should be linked combinations, not independent changes:
- `GlobalReadVectorWidthB 8 -> 4` with the corresponding `VectorWidthB` change.
- `VectorWidthB 2 -> 4` only in the family where the load-to-lane transform is proven.
- `VectorWidthA=2` only after exact lane mapping and validation fixtures exist.

Validation failures remain candidate evidence, not global format rules.

## Backward consistency work

Backward's existing fields should be renamed internally to the common semantic vocabulary without changing their legacy serialization or assembly:
- `DoubleBufferLds` maps to `LdsBuffering`.
- `Store.Priority` maps to `StorePriority` with `Normal` and `Raised` enum members.
- `PrefetchGlobalRead` and `PrefetchLocalRead` map to `SingleStage` and `DoubleStage` policies internally.
- `LdsLayout.Kind`, `LdsPadB`, and `LdsSwizzleChunkB` map to the common layout policy.
- `ActivationAddressing` and `OperandSource` remain only in unchanged legacy specs; new canonical specs use the decomposed `ActivationStaging` and `WeightStaging` fields. A migration must update all catalogs and reject the old schema rather than parse both forms.
- `PackedWeightLaneShare` is decomposed into canonical `PackedLoadGrouping`; existing `1/2` meanings become `PerLane/LanePair` only in that explicit migration.
- `MetadataLoadVectorWidth`, `PayloadLdsWriteVectorWidth`, and `MetadataLdsWriteVectorWidth` describe physical byte transactions, not metadata elements or generic packed words, and are admitted only for formats with aligned lowering support.

The backward search domain remains quant-aware. Q3/Q4/Q5/Q6/Q8 decode extraction and lane sharing are not one common domain merely because they use the same field name. The physical planner must continue to derive decoder register demand, A address lifetime, LDS size, and wait thresholds from the selected quant backend.

Useful future backward combinations are limited to already supported complete mechanisms:
- `LdsPadB=8` with `PrefetchGlobalRead=2`, `PrefetchLocalRead=1`.
- `DoubleBufferLds=True` only with the required activation prefetch, global-read depth, local-read mode, geometry, and barrier path.
- Q6 `PrefetchNextPackedWeight=True` only at qualified exact geometries.
- `InterleaveWmmaWaits` coupled with the activation-prefetch wait policy.
- `DecodeDependencyWidth=1` or `4` only for formats whose decoder register and issue contracts support it.

Do not reopen measured rejected combinations without a changed physical premise.

## Grouped and paired policy boundaries

Grouped forward should share the common names for geometry, LDS, staging, data-plane widths, and epilogue only where its route and row-task ownership remain unchanged. `PayloadPrefetch`, Q2 producer pairing, distributed producers, row-tile dispatch, and grouped route policies remain grouped-specific enums. Payload/metadata read widths and decode producer counts must still be re-derived for the grouped physical plan.

Paired grouped kernels require separate projection-interleave, dual-LDS, route-task, and accumulation policies. A common `PrefetchGlobalRead`, `WaveSeparateGlobalReadA/B`, or vector-width name is acceptable only if it selects an emitter with the paired ABI and both projection ownerships accounted for.

Q3 pair, IQ2 pair, fixed Q8 group ownership, device row tasks, and route split thresholds must not be represented as ordinary MMQ knobs. Their values are part of specialized problem contracts.

## StreamK and deferred mechanisms

Do not implement generic `StreamK`, persistent workgroups, prepared weights, external decode workspaces, split-K reduction, or producer fusion as tuning fields in this plan. Each changes ABI, grid ownership, workspace, or output semantics and needs a separate mechanism contract.

StreamK may be revisited later as an enum-backed execution mode with explicit partial-result storage, final-owner epilogue rules, auxiliary-output ownership, workspace sizing, dispatch selection, and format-specific correctness. It must not be a boolean added to existing MMQ specs.

## Implementation priorities

### Priority 0: compatibility and architecture foundations

- Freeze existing source hashes, spec mappings, resources, exact correctness outputs, and artifact identities. Add regression tests that unchanged specs regenerate byte-identical assembly.
- Define the typed semantic schema, canonical serialization, candidate hashing, and per-operation/per-quant capability matrix. Keep the old schema valid only until one explicit migration is complete; do not add a second parser or silent compatibility projection.
- Implement the `complete typed spec -> derived state -> physical plan -> lowering -> inspection/validation -> candidate identity` boundary from `ggtensile_plan.md`. Keep generation independent of search, catalogs, benchmark reports, and deployment policy.
- Add lowering-effect tests that prove accepted fields reach derived state and change the physical plan or emitted assembly. Reject inert fields and fields that alter ABI, route ownership, or workspace without a mechanism contract.
- Define applicability rows for ordinary forward/backward, grouped forward, and paired grouped kernels. Make specialized route, task, projection, and fixed ownership contracts explicit.

### Priority 1: existing mechanisms and backward-compatible behavior

- Make existing forward and backward knobs demonstrably effective without changing unchanged specs. Preserve current serialized values and emitted assembly for existing catalogs.
- Normalize existing prefetch, LDS, store, `ActivationAddressing`, `OperandSource`, and quant decode values into typed internal meanings only when the mapping is exact. `ActivationStaging` replaces or decomposes `ActivationAddressing`; `WeightStaging` replaces or decomposes `OperandSource` rather than coexisting with it.
- Preserve measured starting points while testing complete linked policies: `PrefetchGlobalRead=1/2`, `DepthU=16/32`, `ClusterLocalRead=Disabled`, backward `LdsPadB=8`, and the existing direction-specific prefetch and swizzle combinations.
- Add effect, resource, correctness, and exact-parent timing checks to `mmq_fwd_search.py` and `mmq_bwd_search.py`. Search helpers enumerate complete valid candidates; they do not construct invalid Cartesian products or make deployment decisions.
- Retain rejected evidence and artifact provenance. Do not reopen a measured loser without a changed physical premise.

### Priority 2: high-value staging, LDS, and decode mechanisms

- Implement family-specific `LdsLayout` variants with derived strides, transforms, buffer sizes, and resource formulas. Probe pads `0`, `4`, `8`, and `16` bytes, starting with one-axis changes and the measured `LdsPadA=4, LdsPadB=16` pair. Retain backward's measured `LdsPadB=8` until a replacement is qualified.
- Implement a complete forward staged loop, beginning with double global-read staging and the bounded `PrefetchGlobalRead=2` plus `DepthU=16` interaction. The plan must cover current/next reads, LDS writes, local reads, decode first use, waits, and register lifetimes.
- Implement `ActivationStaging` and `WeightStaging` as decomposed policies with family-specific capability rows. Do not expose a selector unless the lowerer has a complete direct, single-stage, double-stage, decoded-LDS, full-LDS, or register-tiled implementation for that value.
- Add quantized data-plane widths where a physical plan exists: `PayloadGlobalReadVectorWidth` and `MetadataLoadVectorWidth`, followed by `PayloadLdsWriteVectorWidth` and `MetadataLdsWriteVectorWidth`. Initial payload/global/LDS byte widths are `4`, `8`, and `16`; metadata load/LDS byte widths are `2`, `4`, `8`, and `16`, constrained by format alignment and extraction.
- Add `DecodeProducerCount` with initial values `1`, `2`, and `4` only where producer ownership, destination grouping, barriers, wait thresholds, registers, and edge masks are all derived.
- Qualify ordinary families before transferring a staging or decode policy to grouped or paired families.

### Priority 3: linked store and vector mechanisms

- Implement one linked store policy containing `NumElementsPerBatchStore`, `StoreVectorWidth`, `StoreClauseWidths`, `StoreTraversal`, `StoreSyncOpt`, `TilesAhead`, and dependency width. Start store batches at `4`, `8`, `10`, `16`, `20`, `24`, and `32`; logical store widths at `1`, `2`, and `4`; clause widths at `B32`, `B64`, and `B128`.
- Make every store policy derive output fragment order, conversion temporaries, clause alignment, edge masking, address advancement, and synchronization. New store policies must change assembly and candidate identity; they cannot be labels around one emitter.
- Add dense-compatible `GlobalReadVectorWidthA/B` and `LocalReadVectorWidth` only after exact lane mappings are proven. Start with dense logical widths `1`, `2`, `4`, and `8`, and local-read widths `8`, `16`, and `32` where the format and emitter support them.
- Expand payload and metadata global/LDS widths through linked candidates rather than independent knobs. Include `PackedLoadGrouping=PerLane/LanePair`, and consider `LaneQuad` only with a format-specific proof. `PackedWeightLaneShare` is decomposed and removed by migration, not retained as a second field.
- Align common policy names across grouped and paired lowering only where route, row-task, projection, dual-LDS, and accumulation ownership remains equivalent.

### Priority 4: producer, traversal, and synchronization experiments

- Add `WaveSeparateGlobalReadA/B` with `Shared` and `WaveSeparated` only for multi-wave plans that derive disjoint read ownership, address state, LDS ranges, and waits for each wave.
- Add `MetadataPrefetch=Disabled/Enabled` only when metadata has a separate producer/consumer frontier and the physical plan carries its address and temporary state.
- Expand `StoreTraversal` from `Canonical` to `TileMajor` and `RowMajor` only with concrete fragment transforms and complete edge/mask tests. Add broader traversal or synchronization modes only after the linked store policy is stable.
- Qualify `StoreSyncOpt=Default/Reduced/Grouped` and any wider dependency schedule against resource and correctness gates. Do not make a synchronization enum a source-order hint.

### Deferred mechanisms

Keep `StreamK`, persistent workgroups, prepared weights, external decode workspaces, split-K reduction, and producer fusion out of this knob plan. Each changes ABI, grid ownership, workspace, or output semantics and needs a separate mechanism contract with its own spec, physical plan, dispatch, correctness, and deployment evidence.

## Qualification gates

Every new candidate must pass, in order:
- strict schema round-trip and candidate identity checks;
- exact geometry, quant-format, ABI, and capability validation;
- deterministic independent generation and build;
- resource inspection with zero private bytes, spills, scratch, calls, and dynamic stack;
- packed-HIP correctness and independent dequantized-reference checks;
- input, packed-weight, gradient, route, and edge mutation checks appropriate to the family;
- serial warmed screening, followed by longer confirmation for retained changes;
- exact parent and HIP timing comparisons on the target shapes.

Resource-increasing candidates require a stable material gain before promotion. Screening leaders are not production evidence. FP16/TensileLite timing, quantized forward timing, quantized backward timing, and grouped timing must remain separate reports and must not be pooled.

## Assembly change policy

No existing kernel changes assembly unless its canonical spec changes. Internal enum conversion, shared validation, and physical-plan refactoring should be guarded by source/hash regression tests.

New policy values and linked combinations are expected to change assembly. They must receive new candidate identities, new artifacts, and new qualification records. A changed existing default is a catalog change, not an implementation detail, and requires an explicit remeasurement and deployment review.

The implementation is complete when every accepted field has a demonstrated lowering effect, every shared field has an applicability matrix, every quant-specific exception is explicit, unchanged specs remain byte-identical, and future tuning can select only complete, inspectable, correctness-qualified candidates.
