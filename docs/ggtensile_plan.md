# GGTensile Implementation Plan

## Purpose

GGTensile is a repository-local assembly kernel generator for packed GGUF matrix multiplication. It takes one `ProblemType`, one exact `ProblemSize`, and one complete `Solution`, then either emits reproducible assembly or returns structured rejection reasons. Its scope covers multiple dense and grouped operation types, matrix-shape families, and GGUF quant formats, with exact measured solutions rather than one universal kernel.

Packed GGUF tensors remain authoritative. Generated kernels reconstruct their operands without hidden dense shadows, implicit prepared-weight caches, or unreported external decode workspaces. Existing HIP kernels remain the correctness oracle, performance control, and runtime fallback.

The implementation intentionally uses a small ROCISA surface inspired by TensileLite: structured code modules and metadata, explicit register pools, and direct assembler/linker invocation. It does not import TensileLite's solution, problem-type, or library-generation machinery. "Full Tensile" is reserved for rocBLAS and source identifiers.

## Project Boundary

The current backend targets gfx1151, wave32, and WMMA V1. Support expands deliberately by operation, quant format, exact shape family, data type, and ISA. Unsupported problem types, sizes, and solutions are normal validation results, not generator failures.

The core contract is:
- exact positive problem sizes supplied at generation time.
- explicit, complete solution identity with no silent repair.
- packed-weight decode inside the generated kernel unless a separate public ownership contract says otherwise.
- shape dimensions compatible with the selected ownership, vector widths, and reduction depth.
- exact-key runtime applicability and HIP fallback for every mismatch.
- no private storage, spills, scratch instructions, calls, or dynamic stack.
- no split reduction, atomics, persistent traversal, edge handling, or external workspace unless those mechanisms receive explicit model, ABI, validation, and dispatch support.

Broader architectures, edge handling, operation types, data types, quant formats, prepared representations, and multi-kernel reductions are expansion projects rather than implicit capabilities of the initial backend.

## Required Quant Formats

The required format inventory comes from the workload and compatibility contracts recorded in the HIP optimization logs, not from every quant type accepted by a generic operator or represented in the kernel bundle.

Dense MMQ uses:
- Qwen: `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K`.
- DeepSeek: `Q8_0`.
- Dense union: `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0`.

The ordinary Qwen projections and language-model head use `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K`; DeepSeek ordinary projections and its language-model head use `Q8_0`. Qwen `IQ2_S` tensors are expert weights and belong only to grouped production coverage. Dense IQ2_S tests and correctness-oracle use are intentionally excluded. The existing HIP dense IQ2_S path remains as reference code, not as a production inventory or GGTensile campaign requirement.

Grouped MMQ uses:
- Qwen: `Q3_K`, `Q4_K`, `Q5_K`, and `IQ2_S`.
- DeepSeek: `Q2_K`, `IQ2_XXS`, and fixed-group `Q8_0`.
- Grouped union: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `IQ2_XXS`, `IQ2_S`, and `Q8_0`.

`Q6_K` has no current grouped production workload. `Q2_K`, `IQ2_XXS`, and `IQ2_S` have no current required dense campaign; their relevant production ownership is grouped MMQ. Fixed-group `Q8_0` is a distinct grouped operation and must not be inferred from dense `Q8_0` coverage.

These inventories describe formats that need workload coverage. They do not imply that one generated decoder, geometry, or artifact is valid across dense, paired routed, single routed, row-task, and fixed-group ownership.

## Public Model

`tools.ggtensile` exposes immutable, JSON-serializable records using TensileLite terminology where the concepts match:
- `ProblemType`: operation, quantized and activation data types, destination and compute data types, and transpose/layout semantics.
- `ProblemSize`: exact GEMM coordinates and operation-specific dimensions.
- `Solution`: kernel language, ISA, wave and workgroup geometry, matrix instruction, macro tile, reduction depth, global/local read widths, LDS and prefetch choices, decoder strategy, scheduling choices, and resource limits.
- `SolutionKey`: the exact problem-type/problem-size/solution tuple with canonical JSON and a stable content hash.
- `RejectReason`: stable rule ID, message, involved parameters, and source.
- `KernelArtifact`: kernel name, assembly/object/code-object paths, source identity, launch geometry, ABI, and measured static resources.

Solution parsing is strict: unknown parameters, implicit numeric coercions, and missing parameters fail before validation. Named defaults are explicit and remain part of `SolutionKey`. GGTensile never mutates or repairs a requested solution.

## Artifact Lifecycle

Generation, build, inspection, correctness, screening, and confirmation are separate phases with immutable outputs:
- `generate` validates an exact `SolutionKey`, emits assembly or structured rejection, and records the generated source hash.
- `build` accepts only an approved generation manifest, verifies source identity, and invokes the assembler and linker.
- `inspect` accepts only an approved build manifest and validates symbol, ABI, ISA, metadata, resources, and forbidden instructions or storage.
- `correctness` executes only inspected artifacts against the operation oracle and independent references required by the experiment.
- `screen` ranks correct candidates under warmed rotating controls.
- `confirmation` retimes finalists under a fresh, longer rotating protocol.

Every phase refuses to overwrite an existing artifact. Manifests are strict and versionless. The generated source assembly hash is the sole artifact content identity; object and code-object output is deterministic toolchain translation validated by inspection rather than a second mutable identity.

This separation supports reproducible manual experiments, bounded per-shape scans, and compile caching without coupling measurement to generation or requiring a large-grid search system.

## Kernel Design Contracts

Each generated kernel supports one exact `ProblemType` and `ProblemSize`. Exact dispatch permits constants, fixed loop counts, peeled tails, fixed launch geometry, shape-specific ownership, and immediate addressing without preserving cross-shape validity.

The solution surface follows emitted behavior:
- a field is tunable only when at least two supported values produce distinct, validated ISA or ownership.
- linked validation covers divisibility, workgroup and wave ownership, load coverage and alignment, LDS size and layout, matrix-instruction structure, and resource limits.
- unsupported combinations are rejected instead of rewritten into a nearby valid solution.
- internal experiments remain derived choices until they have complete alternate emitters and stable identity.
- generic GEMM controls are not exposed when they do not describe fused packed decode.

Architecture-specific behavior is checked against `~/rdna35-isa-markdown/`, `~/amd-llvm-project/`, and LLVM AMDGPU definitions and tests. Packed work-item coordinates must be decoded before their incoming VGPRs are reused. Matrix-operand lane mapping, sub-dword load/store semantics, conversion rounding, wait dependencies, VOPD legality, and hazard sequences require ISA evidence and execution validation.

## Toolchain And Inspection

The toolchain resolver accepts explicit paths and otherwise discovers the ROCm SDK shipped with the active Python environment. Builds use deterministic commands and the target code-object format. Temporary build files live outside the output directory, and accepted artifacts are installed only after assembly and linking succeed.

The artifact inspector requires:
- exactly one expected global kernel symbol.
- target metadata matching the declared kernarg ABI.
- the requested wavefront and maximum workgroup sizes.
- expected LDS, matrix-instruction, wait, and barrier structure.
- no private segment, dynamic stack, scratch instructions, calls, or spills.
- no register indices beyond metadata declarations.
- no unsupported cache, hazard, or control-flow instructions.

Inspection also records explanatory static metrics: allocated VGPRs and SGPRs, LDS bytes, VALU issues and operations, VOPD pairs, VMEM and LDS instructions, waits, barriers, clauses, dependency delays, cache invalidations, and code size.

## Correctness Gates

Unit tests cover canonical identity, JSON round trips, strict parsing, rejection rules, deterministic generation, tool discovery, and inspection parsers. GPU validation is an explicit phase so ordinary unit tests do not require ROCm hardware.

Every experiment defines its own independent reference, adversarial packed-data cases, exact production keys, and mutation checks. The reusable validation policy is:
- compare with the existing operation implementation.
- compare with an independently decoded or independently formulated reference where practical.
- cover metadata fields, packed bit planes, block and tile boundaries, and all specialized control-flow paths.
- rerun after complete input and packed-weight rewrites when cache policy or producer handoff is relevant.
- require bit-exact output when operand conversion and reduction order match.
- otherwise establish and document a fixed numerical envelope before timing.
- execute reduced problems before production launches whenever control flow, induction, ownership, or pipeline lifetimes change.

Static inspection never substitutes for execution correctness.

## Measurement And Retention

Compilation, inspection, correctness, profiling, screening, and confirmation never overlap on the GPU. Timing uses warmed rotating same-process controls with the same tensors and launch conditions. Reports preserve raw samples, median, dispersion, candidate/control ratio, protocol identity, and normalized assembly around finalists.

The hard gates are:
- zero correctness failures.
- byte-identical generated source on independent rebuilds.
- zero private storage, spills, scratch, calls, or dynamic stack.
- no stable regression above 1% on another exact key that shares the emitted path.
- a stable gain above 2% for a new resource-bearing mechanism.
- neutral-to-favorable representative timing for an unconditional instruction or resource reduction.

Lower static instruction count alone is insufficient. Timing selects winners; counters, static issues, locality, code size, and resources explain them.

Per-key finalists are selected by fresh local brackets. Campaign success is evaluated by the complete target workload: weight exact-key medians by production call counts, report family totals and aggregate latency, and retain the existing implementation as fallback for every slower or unmatched key. A family average never authorizes dispatch of a slower exact key.

## Tuning Method

### Inventory and controls

Begin with a machine-readable exact-key inventory containing representative tensors, call counts, historical controls, fresh control requirements, correctness obligations, and selected-solution references. Establish the current generated kernel and existing implementation as independent controls before changing ownership or scheduling.

Bounded scans construct only explicit complete solutions. They do not build broad Cartesian products, infer missing values, repair invalid combinations, benchmark while compiling, time correctness failures, or treat missing observations as rejections.

### Control taxonomy

For every field or proposed mechanism, classify it as one of:
- implemented search axis with distinct validated emitters.
- fixed identity required by the current backend.
- internal derived lowering with no public alternate.
- proposed mechanism awaiting a complete emitter.
- unsupported on the target ISA.
- contract-expanding and deferred.
- measured and closed for the current experiment.

TensileLite and other kernel generators supply mechanism vocabulary and linked-constraint evidence, not parameter lists to copy. Geometry, matrix-wave ownership, reduction depth, global and local prefetch, LDS buffer count and layout, vectorized reads, traversal, store scheduling, and instruction scheduling enter a search only when they change this generator's emitted behavior.

### Exact-shape lowering

Apply exact-shape compiler work before adding resource-bearing mechanisms:
- remove dead kernarg state while preserving the public ABI.
- fold dimensions, strides, tile counts, launch divisors, and packed block offsets into immediates.
- specialize fixed trip counts, peel prime and final iterations, and remove unreachable tails.
- strength-reduce affine addressing when it does not extend live ranges or raise allocation.
- choose immediate-offset, scalar-base, and explicit-address forms from the exact geometry.
- recompute register lifetimes after each ownership or pipeline change.
- place long-lived accumulators and fragments before transient decode/address state.
- form VOPD pairs only when opcode slots, dependencies, and register-bank classes are legal.
- remove cache or hazard instructions only after the memory and dependency contract is proven by mutation tests and target-ISA evidence.

Exact lowerings are derived behavior, not user-visible tuning knobs.

### Geometry, ownership, and locality

Search high-level mechanisms before low-level instruction cadence because geometry changes register pressure, wave residency, data duplication, LDS capacity, launch parallelism, and useful overlap. Focused neighborhoods include:
- macro-tile and wave-tile ownership.
- active compute and decoder waves.
- one versus multiple decoded-weight buffers.
- decoder width and packed-payload assignment.
- LDS transposition, padding, and XOR layouts.
- workgroup traversal and mapping.
- cooperative versus repeated input or weight staging.
- epilogue ownership and store traversal.

Traversal is an exact-shape axis, not a universal constant. Treat L0/L2 behavior, occupancy, LDS conflicts, and stall counters as explanatory evidence and keep timing authoritative.

### Scheduling and ISA

Low-level scheduling follows selected ownership and layout. A schedule names placement of packed reads, metadata reads, decode chunks, activation reads, LDS reads/writes, matrix instructions, waits, barriers, and stores. Numeric schedule identifiers are aliases for complete inspectable schedules.

Vary wait thresholds only at actual first-use boundaries. Compare compiler output and expert assembly for instruction selection, clauses, VOPD, dependency distance, and termination sequences, but do not copy generic bounds, pointer state, or hazard padding without a matching contract. Add `s_delay_alu`, dependency waits, clauses, priorities, or explicit resource deallocation only from target-ISA requirements or measured evidence.

### Lower bounds and profiling

When diagnosis is ambiguous, generate exact diagnostic kernels that isolate major work classes while preserving launch geometry and declared resources. Typical floors separate matrix/activation/LDS work from packed decode/LDS work. Each diagnostic has its own expected instruction and synchronization structure and passes the same symbol, ABI, resource, and forbidden-storage checks.

Collect only target-supported counter groups. Preserve raw runs when derived combinations exceed hardware collection capability. Profiles identify first-order opportunities; they do not select winners.

## Optimization-Exhaustion Review

Every campaign ends with a recursive optimization-exhaustion review. Re-read:
- the project plan and active experiment record.
- related dense and grouped optimization histories.
- current generated and existing implementation sources and normalized disassembly.
- manifests, selected and rejected candidates, lower bounds, profiles, counters, and timing brackets.
- TensileLite, EvoTensile, CK, hipBLASLt, and relevant external mechanism studies.
- target ISA documentation and LLVM AMDGPU instruction, VOPD, hazard, delay, and scheduling definitions and tests.

Classify every inferred idea as duplicate or closed, contract-incompatible, unsupported, deferred with an explicit prerequisite, or actionable with a target key, mechanism, expected gain path, and measurement gate. If the review finds an actionable idea, insert and execute that experiment before the review and then repeat the complete review. A campaign is exhausted only when a fresh final pass finds no actionable in-contract mechanism.

This review remains the permanent final step of every experiment. A later implementation or plan change that creates a new premise invalidates the stopping condition until the review is repeated.

## Experiment Records

Experiment-specific problem definitions, ABI details, production inventories, selected solutions, resources, timing, correctness, rejected mechanisms, debugging history, evidence paths, and completion state belong in experiment logs.

Completed dense-backward campaign records cover [Q3_K](experiment_ggtensile_mmq_bwd_q3_k.md), [Q4_K](experiment_ggtensile_mmq_bwd_q4_k.md), [Q5_K](experiment_ggtensile_mmq_bwd_q5_k.md), [Q6_K](experiment_ggtensile_mmq_bwd_q6_k.md), and [Q8_0](experiment_ggtensile_mmq_bwd_q8_0.md). Each document is authoritative only for its own exact keys and must not be generalized to another operation, quant format, shape family, or architecture without measurement. These five campaigns cover the required dense-backward format inventory.

## Integration And Expansion

Each artifact uses a separate exact-problem symbol and does not replace an existing range symbol. Production dispatch may select an assembly artifact only when every `ProblemType`, `ProblemSize`, ABI, architecture, and source-identity assertion matches; otherwise it uses the existing implementation. Cross-shape correctness is deliberately outside an exact artifact's contract.

Public runtime integration remains deferred until a useful production set is covered, per-key selection is complete, artifact packaging and identity are stable, and complete end-to-end workloads pass correctness and weighted performance validation. Experimental force controls are not public dispatch policy.

Dense-backward GGTensile format coverage is complete for the required `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0` inventory. IQ2_S decoder development belongs to grouped GGTensile, where codebook access, sign and scale reconstruction, paired and single ownership, routing, row tasks, resource accounting, and correctness fixtures can be validated under the production contract.

The dense-backward format prerequisite for considering dense forward is satisfied, though dense forward still requires its own explicit project and workload validation. Grouped GGTensile remains deferred until dense assembly demonstrates durable advantages and the grouped ownership contract is designed explicitly. Its eventual format scope is the seven-format grouped union above, including grouped-only `Q2_K`, `IQ2_XXS`, and `IQ2_S` plus fixed-group `Q8_0`; dense coverage of an overlapping format does not count as grouped coverage.

Prepared weights, compact alternate layouts, BF16 shadows, paired projections, persistent workgroups, split reduction, GSU, and Stream-K require explicit model-visible ownership, lifetime, invalidation, memory accounting, fixup, ABI, and fallback design. They are separate projects, not hidden extensions of a single-kernel tuning campaign.

Bounded scan scripts remain outside `KernelWriterAssembly`. They may construct, cache, validate, and rank complete solutions for explicit exact keys, while GGTensile preserves deterministic `SolutionKey` identity, explainable rejection, immutable artifact phases, and reproducible evidence. A larger automated search system remains optional.
