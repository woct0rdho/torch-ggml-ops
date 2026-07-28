# GGTensile Implementation Plan

## Purpose

GGTensile is a repository-local assembly kernel generator for packed GGUF matrix multiplication. It takes one `ProblemType`, one exact `ProblemSize`, and one complete `Solution`, then either emits reproducible gfx1151 assembly or returns structured rejection reasons. Its long-term scope covers multiple dense and grouped operation types, matrix-shape families, and GGUF quant formats, with exact measured solutions rather than one universal kernel. The first implemented kernel is dense MMQ backward for Q4_K. It computes

```text
grad_input[rows, in_features] = grad_output[rows, out_features] @ dequant(weight[out_features, in_features])
```

without materializing a dense or transposed weight. The existing HIP kernel is the correctness oracle, performance control, and runtime fallback.

The implementation intentionally uses a small ROCISA surface inspired by TensileLite: structured code modules and metadata, explicit register pools, and direct assembler/linker invocation. It does not import TensileLite's solution, problem-type, or library-generation machinery. "Full Tensile" is reserved for rocBLAS and source identifiers. Dense backward remains the proving ground; dense forward and grouped kernels are admitted only after the generator demonstrates durable gains across their required shape and quant families.

## Initial Boundary

The first supported target is deliberately narrow:

- architecture `gfx1151`, wave32, WMMA V1;
- dense backward only;
- Q4_K packed weights;
- BF16 activation, output, and dequant staging with FP32 accumulation;
- exact positive shapes supplied at generation time;
- `in_features == 2048` for the pilot;
- shape dimensions divisible by the selected macro tiles and reduction depth;
- one workgroup computes one complete output tile, with no split reduction, atomics, edge masks, persistent traversal, or external decode workspace;
- the existing 40-byte dense-backward kernarg ABI.

Unsupported problem types, problem sizes, and solutions are normal results, not generator failures. Broader shapes, edge handling, quantization types, forward kernels, and grouped kernels are expansion work gated by measured pilot results.

## Public Model

`tools.ggtensile` exposes immutable, JSON-serializable records using TensileLite terminology where the concepts match:

- `ProblemType`: operation, quantized and activation data types, destination and compute data types, and transpose/layout semantics;
- `ProblemSize`: exact GEMM coordinates `M=rows`, `N=in_features`, and `K=out_features`;
- `Solution`: `KernelLanguage`, `ISA`, `WavefrontSize`, `WorkGroup`, `MatrixInstruction`, `MacroTile0/1`, `DepthU`, global/local read widths, LDS and prefetch choices, GGUF decoder strategy, scheduling choices, and resource limits;
- `SolutionKey`: the exact problem-type/problem-size/solution tuple with canonical JSON and a stable content hash;
- `RejectReason`: stable rule ID, message, involved parameters, and source;
- `KernelArtifact`: kernel name, assembly/object/code-object paths, hashes, launch geometry, ABI, and measured static resources.

Solution parsing is strict: unknown parameters, implicit numeric coercions, and missing parameters fail before validation. Pilot defaults are available explicitly through `Solution.pilot()`, but are included in `SolutionKey`. GGTensile never mutates or repairs a requested solution.

The command-line interface accepts strict JSON solution files, writes a manifest for both accepted and rejected solutions, and has separate `generate`, `build`, and `inspect` commands. `generate` runs `KernelWriterAssembly`; `build` accepts only an accepted generate manifest and verifies the source assembly hash before invoking the assembler and linker; `inspect` accepts only an accepted build manifest and validates the resulting code object. Every phase refuses to overwrite an existing artifact. This phase separation supports reproducible manual experiments, bounded per-shape scan scripts, and compile caching without requiring a large-grid search system.

## Pilot Kernel

The pilot uses a `128 x 128 x 32` work tile and 128 threads (four wave32 waves). Each wave owns 32 rows as two M16 WMMA tiles. During every N32 reduction iteration, the workgroup:

1. Cooperatively loads Q4_K headers and packed nibbles for the corresponding `32 x 128` weight tile.
2. Unpacks Q4_K six-bit scale/min fields, decodes values with the GGUF formula, converts them to BF16, and stages the transposed WMMA-facing weight tile in LDS.
3. Synchronizes, loads BF16 activation fragments directly from global memory, reads weight fragments from LDS, and issues 32 static `v_wmma_f32_16x16x16_bf16` instructions for the `DepthU=32` iteration.
4. Reuses LDS after a second barrier and advances to the next reduction tile.
5. Stores one exact `128 x 128` output tile with no edge predicates.

The initial `Solution` model exposes only choices represented by implemented `KernelWriterAssembly` paths. A parameter is not tunable merely because TensileLite has a similarly named parameter. Linked constraints cover tile divisibility, wave ownership, load coverage/alignment, `LdsNumBytes`, WMMA issue structure, and estimated accumulator/register pressure.

The gfx1151-specific contracts are checked against `~/rdna35-isa-markdown/` and `~/amd-llvm-project/`: gfx11 uses a packed work-item ID in `v0`, with X in bits 0-9 and enabled Y in bits 10-19, so `WorkGroup=[32,4,1]` is explicitly flattened before `v0` becomes accumulator storage; `v_fma_mix_f32` uses `op_sel_hi` to select FP16 interpretation and `op_sel` to select the low or high half; BF16 WMMA V1 requires lanes 16-31 to replicate lanes 0-15 for A and B and maps eight FP32 accumulator VGPRs to two physical output rows per lane half; and `ds_store_b16_d16_hi` plus `global_store_d16_hi_b16` write bits 31-16 after the explicit BF16 round-to-nearest-even integer transform.

The manual tuning process follows the case study in `~/ComfyUI-FeatherOps/doc/tensile_fp16_nt_hhs.md`: begin from a measured control, change complete solutions in focused neighborhoods, preserve correctness as a hard gate, use generated-client-style timing only for screening, retime finalists with a hot loop, compare normalized assembly and resources around winners, and derive exact-shape fast paths from selection predicates instead of exposing unsafe force knobs. Its high-impact `ScheduleIterAlg`, `StorePriorityOpt`, and `NumElementsPerBatchStore` findings justify corresponding GGTensile parameters only after distinct schedules are implemented. `WorkGroupMapping`, `PrefetchGlobalRead`, `PrefetchLocalRead`, `1LDSBuffer`, vector widths, and LDS layout likewise enter the search space only when they alter this fused decode kernel's emitted ISA. Epilogue, GSU, activation, bias, scale-vector, and general-layout controls from that FP16 NT HHS case study do not apply to the pilot.

## Toolchain And Reproducibility

The toolchain resolver accepts explicit paths and otherwise discovers the ROCm SDK shipped with the active Python environment. Builds use code-object v5 and deterministic commands. Temporary build files live outside the output directory and accepted artifacts are installed only after assembly and linking succeed.

The artifact inspector requires:

- exactly one expected global kernel symbol;
- gfx1151 code-object-v5 metadata matching the 40-byte kernarg ABI;
- wavefront size 32, 128-thread maximum workgroup size, and expected LDS;
- no private segment, dynamic stack, scratch instructions, calls, or spills;
- no register indices beyond metadata declarations;
- the expected static WMMA and barrier structure.

The generated source assembly hash is the sole artifact content identity. Object and code-object translation is treated as deterministic toolchain output rather than a separately controlled identity; inspection validates its symbol, ABI, ISA, and resources.

## Correctness And Performance Gates

Unit tests cover canonical identity, JSON round trips, strict schema parsing, every rejection rule, deterministic source generation, tool discovery, and artifact inspection parsers. GPU validation is a separate explicit command so ordinary unit tests do not require ROCm hardware.

Pilot correctness compares the generated kernel with both the existing HIP dense-backward kernel and an independently dequantized BF16 reference. It includes one-hot Q4_K blocks spanning every scale group and nibble plane, random packed weights, tile-boundary rows, and production row counts. Accepted results are bit-exact where the accumulation order matches; otherwise the error bound must be justified and fixed in the validation protocol.

Timing is performed only after correctness passes, using the same tensors and launch geometry for warmed alternating control/candidate samples. Report raw samples, median, dispersion, and candidate/control ratio. Expansion requires:

- zero correctness failures;
- zero private storage, spills, scratch, calls, or dynamic stack;
- byte-identical source assembly rebuilds;
- no stable regression above 1%; and
- at least a stable 2% Q4_K `in_features=2048` gain, or equivalent latency with a useful VGPR reduction.

## Dense Backward Q4_K Experiment

The first measured pair covers `ProblemSize(M=32768, N=2048, K=8192)` and the companion production reduction `K=512`. It established exact Q4_K decoding, WGM1 traversal, the XOR-8 LDS layout, SIA4/PGR2 scheduling, scalar-base global addressing, and the current 212-VGPR resource-clean implementation.

The selected four-wave, two-buffer decoded-B pipeline measures 38.012 ms and 28.93 TFLOP/s at K8192 versus HIP at 47.779 ms. The K512 companion measures 2.5063 ms and 27.42 TFLOP/s versus HIP at 3.0246 ms while preserving the independently verified numerical envelope. Detailed measurements, retained and rejected mechanisms, lower bounds, debugging lessons, and the 12-shape campaign are recorded in [experiment_ggtensile_mmq_bwd_q4_k.md](experiment_ggtensile_mmq_bwd_q4_k.md).

This experiment is evidence for the generator and search process, not a universal solution. Geometry, decoder ownership, LDS layout, traversal, scheduling, and epilogue policy remain shape- and quant-specific in the multi-shape roadmap below.

## Multi-Shape GGTensile Tuning Plan

### Performance position

The current selected Q4_K kernel sustains 28.93 TFLOP/s at K8192, or about 48.7% of the gfx1151 BF16 WMMA roof of approximately 59.4 TFLOP/s. It reaches about 64% of the approximately 45 TFLOP/s delivered by a well-tuned hipBLASLt BF16 GEMM on this machine. The hipBLASLt result is an upper reference rather than an immediate fused-kernel target because GGTensile repeatedly loads packed data, reconstructs BF16 weights, stages LDS, and synchronizes before WMMA.

The exact-shape experiment has already measured ordinary geometry changes, DepthU 64, one- and two-LDS-buffer schedules, dedicated decoder waves, decoder batching, local prefetch, traversal, accumulator-clear VOPD, clauses, cache invalidation, and dependency placement. The two-buffer pipeline is retained; dedicated decoder waves and sub-percent scheduling neighborhoods are closed. Revisit this shape only for a materially different mechanism with a plausible multi-percent gain, while prioritizing the remaining production Q4_K shape families.

ROCm rocm-libraries PR 9385 provides additional gfx1151 evidence. Its relevant transferable mechanisms are correct sub-dword WMMA local reads, capping local-read buffers by actual loop iterations, placing long-lived values before transient address registers, and using interaction-based shape tuning rather than one universal configuration. GGTensile should port those principles where they match the fused decoder rather than copying general GEMM code paths.

### Production shape inventory and priority

The dense Q4_K campaign is the exact Cartesian product of `M={2048,8192,32768}` and `(N,K)={(2048,512),(512,2048),(4096,2048),(2048,8192)}` in GGTensile coordinates. Every generated kernel is correct for one exact `ProblemType` and `ProblemSize`; dispatch requires an exact key match and falls back to HIP otherwise. Grouped MMQ is outside this campaign.

The established packaged-HIP timings below are planning seeds from the dense-backward optimization record. The campaign inventory must refresh them in the same process and build used for GGTensile decisions. The current experiment's fresh M32768 controls, 3.0246 ms for narrow and 47.779 ms for query, illustrate why historical absolute values cannot be mixed with a new bracket.

| Family `(N,K)` | Calls | HIP ms at M2048/M8192/M32768 | Weighted Q4_K share | Current GGTensile status | First optimization question |
| --- | ---: | ---: | ---: | --- | --- |
| Narrow `(2048,512)` | 70 | `0.230/0.775/2.988` | about 32-40% | M32768 selected at 2.5063 ms; M2048/M8192 open | Can exact K512 trip specialization, lower fixed overhead, or lower-resource ownership beat the retained pipeline? |
| Shared down `(512,2048)` | 30 | `0.261/1.537/5.266` | about 19-27% | all three open | Does N512 prefer different M ownership, XOR-16-style LDS layout, or traversal that reuses packed B across M? |
| Attention output `(4096,2048)` | 10 | `1.339/5.912/23.803` | about 33-36% | all three open | Does wide N favor different traversal, N ownership, and epilogue cadence while preserving A locality? |
| Query `(2048,8192)` | 1 | `3.376/12.567/48.613` | about 7-8% | M32768 selected at 38.012 ms; M2048/M8192 open | Does the same decode/WMMA pipeline win at lower M, or does occupancy and traversal change the balance? |

Narrow is first by call count, but attention output is first by aggregate latency at M8192 and M32768. Shared down has fewer weighted milliseconds but stronger evidence of HIP underperformance at those row counts. Candidate scheduling should therefore use fresh `call_count * HIP_median_ms` contribution and measured gap, not call count alone. Query remains last unless a mechanism discovered on another long-K shape transfers directly.

The first coverage change is complete and is not a tuning parameter. Validation accepts exactly `N={512,2048,4096}`, and the writer derives packed Q4_K row stride as `(N/256)*144` bytes instead of assuming the N2048 value. Duplicate exact `(2048,512,2048)` and `(2048,4096,2048)` artifacts build and inspect cleanly. Representative shared-down and attention-output tensors are bit-exact to HIP before and after grad-output and packed-weight mutations. One-repeat execution checks already beat HIP, but fresh rotating brackets remain authoritative.

### Control taxonomy

GGTensile must distinguish implemented search axes, fixed identity fields, fused-decoder mechanisms that still need alternate emitters, and compiler-style lowering. A field does not become tunable merely because it exists in `Solution` or TensileLite.

The writer currently emits genuinely different paths for these coupled axes:

- geometry through the implemented `MatrixInstruction`, `MacroTile`, `WorkGroup`, and `DepthU` combinations;
- `WorkGroupMapping`, constrained by the exact M tile count;
- the implemented `ScheduleIterAlg`, `PrefetchGlobalRead`, and `PrefetchLocalRead` combinations;
- one decoded-B LDS buffer versus the complete two-buffer pipeline, represented by `1LDSBuffer=1` and `0` respectively;
- `LdsSwizzleChunkB` values `0`, `4`, `8`, and `16`;
- `StorePriorityOpt`, packed-next-only prefetch, and packed-weight lane sharing.

These paths are valid bounded-scan inputs, but a value rejected on the measured M32768 pair is not automatically rejected for a different exact key. It enters a new scan only when the shape changes the stated cost model. For example, a K512 key may change prologue and epilogue weight, while N512 changes decode duplication and launch locality. The scan records the changed premise instead of silently reopening a closed experiment.

The following `Solution` fields remain fixed identity until an alternate path changes emitted ISA: `DecoderWidth=16`, both global-read vector widths, `LocalReadVectorWidth=16`, `NumElementsPerBatchStore=8`, `StoreVectorWidth=1`, `TransposeLDS=0`, both LDS pad fields, and `PrefetchPackedWeight=true`. In particular, the current store loop ignores `NumElementsPerBatchStore` as a scheduling choice; accepting another number would be a fake knob.

Potential future mechanisms are not public controls yet:

- active compute-wave count and compute/decoder ownership;
- decoder width, decode chunk size, metadata-sharing method, and packed-payload assignment;
- independent A, packed-payload, and metadata prefetch leads;
- explicit wait budgets and decode operations per WMMA group;
- alternative packed and metadata load widths;
- row-padded LDS formulas beyond the implemented XOR layouts;
- real store batching, store traversal, vector stores, or LDS store remap;
- fixed-trip reduction unroll policy.

Each becomes a strict solution field only after at least two complete, validated emitters exist. Until then it is an internal named experiment or an unconditional derived choice.

TensileLite supplies mechanism vocabulary and linked-constraint evidence, not a parameter list to copy. `MIWaveTile`, `MIWaveGroup`, `DepthU`, PGR, PLR, LDS buffer count, vectorized reads, workgroup mapping, store batching, and scheduling are relevant where GGTensile implements the same behavior. General-GEMM controls such as GSU, Stream-K, DirectToLds, DirectToVgpr, sparse metadata paths, source swap, activation, bias, and workspace epilogues do not describe this fused packed decoder. `StaggerU` is admitted only if a profile identifies K-partition or cache-set contention that ordinary traversal does not explain.

### Exact-shape lowering

Apply exact-shape compiler work before adding resource-bearing mechanisms. These passes are derived from `ProblemSize` and the selected complete solution; they are not search knobs.

- Remove dead kernarg state. Completed: the writer now loads only the three live pointers, preserves the 40-byte ABI, and reduces declared SGPRs from 20 to 16. Serial 25-repeat brackets improved K512 by 0.42% and K8192 by 0.25% with no other static-resource change.
- Fold exact dimensions, strides, tile counts, launch divisors, and Q4_K block offsets into immediates. Retain dynamic workgroup and lane coordinates, which are not shape constants.
- Generate fixed reduction trip counts. Completed branch-only specialization preserves the prime/steady/final pipeline, removes the pre-stage exit test and unconditional back branch, and uses one post-stage compare. K32 emits no steady body; K64 and larger exact keys retain the steady path. Serial 25-repeat brackets improved K512 by 0.72% and K8192 by 0.30% with unchanged resources and static issue counts. Duff-style body unroll is closed: factors 2/4/8 and complete K512 factor 15 or K8192 factor 16 grew code objects by 19-289%; K512 gains were at most 0.97%, and K8192 was neutral to 0.63% slower. No unroll control remains in the writer.
- Peel the exact prime and final iterations, remove impossible tails and bounds, and delete branch, counter, and pointer state only when no dynamic consumer remains.
- Strength-reduce affine A and packed-weight addressing only when the induction form does not extend a live range or raise VGPR allocation. The rejected persistent-A experiment remains the control for resource-growing pointer state.
- Derive immediate-offset versus explicit-address forms per exact geometry and keep scalar-base addressing wherever offsets fit.
- Recompute register lifetimes after each geometry or pipeline change. Keep long-lived accumulators and fragments before transient decode/address state, cap fragment buffers by actual modulo reuse, and compact only after auditing every consumer.
- Form legal VOPD pairs where instruction selection and register modulo classes permit them. Keep the 20 retained accumulator-clear pairs. The steady decoder is dominated by `v_bfe`, conversion, `v_fma_f32`, and BF16 rounding operations that are not useful gfx11 VOPD-Y candidates, so dynamic VOPD requires a concrete alternate lowering rather than an assumed compiler pass gain.
- Omit `buffer_gl0_inv` only under the established immutable-input and producer-handoff proof. Preserve the instruction if a future memory contract invalidates that proof.

An unconditional change is retained when it is bit-exact, reproducible, resource-clean, and neutral-to-favorable on representative K512 and K8192 brackets. Lower static instruction count alone is not sufficient: prior HIP work showed that removing conversion or LDS work can also remove latency-hiding cadence.

### High-level shape experiments

Start every open key from the selected `128x128x32`, four-wave, XOR-8, SIA4/PGR2/PLR1 two-buffer pipeline when compatible. Also preserve HIP and the best one-buffer assembly as controls. Search in focused neighborhoods, promoting only winners to the next neighborhood.

For narrow K512 keys:

- run exact lowering and fixed-trip unroll before new resource ownership;
- compare one versus two decoded-B buffers and the implemented four-wave M64/M128 ownership where exact divisibility holds;
- scan valid traversal divisors for each M and recheck XOR-8 against unpadded/XOR-16 only because short K changes fixed-cost weighting;
- profile prologue, final-store, VMEM-wait, and barrier shares; implement real store batching or active-wave alternatives only if those shares are first-order.

For shared-down N512/K2048 keys:

- compare M64/M128 ownership and one/two-buffer overlap while keeping N tile coverage exact;
- prioritize `WorkGroupMapping` and packed-B locality because only four N128 tiles exist and prior dense traversal changes produced large L2 gains;
- compare XOR-8, XOR-16, and unpadded LDS layouts with static LDS issue counts and dynamic stalls, using the HIP XOR-16 winner only as a seed;
- add wider-N ownership only if the implemented neighborhood shows repeated A traffic or launch overhead as the dominant floor.

For attention-output N4096/K2048 keys:

- scan traversal first because the wide N grid can trade A reuse against packed-B reuse;
- compare the retained four-wave geometries and buffer count, then profile A traffic, L2 hit rate, LDS waits, and store share;
- consider wider-N ownership and a real batched epilogue only after those profiles show a multi-percent ceiling. Larger accumulator sets must remain below the resource warning boundary and cannot be justified by fewer workgroups alone.

For query K8192 keys:

- preserve the true two-buffer pipeline and scan only M-dependent traversal and compatible geometry on M2048/M8192;
- use lower-bound kernels when the complete/WMMA/decode relationship materially differs from M32768;
- do not reopen clauses, dependency batching, early A0 placement, packed-next-only prefetch, lane sharing, PLR2, SIA5, DepthU64, or dedicated decoder waves without a changed resource or stall premise;
- revisit M32768 only for a mechanism plausibly exceeding its remaining approximately 3.6% latency gap to 30 TFLOP/s.

### Traversal and locality

Traversal is a first-order exact-shape axis. Prior dense-backward DB7/DB8 controls changed only grouped-M launch mapping yet cut large-M latency by about 39-42% on the first families and raised L2 hit rate by 30.6-47.4 percentage points in the complete sweep. Earlier Q4_K pilot WGM1 likewise changed the kernel from roughly 119 ms to roughly 52 ms. Geometry-specific counterexamples also prove that no global mapping rule is valid.

For each exact key, scan only `WorkGroupMapping` values that divide `M/MacroTile0`. Compare normalized assembly to ensure the body is otherwise unchanged, then retain mappings by event timing and locality evidence. Treat L2 hit rate, occupancy, LDS stalls, and cache traffic as explanatory counters; timing remains authoritative. Do not infer a winner from launch-order intuition or conflict percentages alone.

### Scheduling and ISA experiments

Low-level scheduling follows high-level geometry, traversal, and pipeline selection because those choices change live ranges and the useful latency-hiding mix.

- Express a schedule as named placement of packed reads, metadata reads, decode chunks, A reads, LDS reads/writes, WMMAs, waits, and the swap barrier. Numeric `ScheduleIterAlg` values remain aliases for complete inspectable schedules.
- Vary wait thresholds only at actual first-use boundaries. Validate VMEM and LDS dependencies statically and with reduced execution before production launch.
- Compare hipcc output with GGTensile as evidence for instruction selection, clauses, VOPD, and dependency distances. Do not copy generic bounds, 64-bit pointer chains, or compiler state into an exact kernel.
- Keep `s_clause` closed by default: A-load clauses gained only 0.035% on K8192 and 0.29% on K512. Reopen clauses only for a different load group with profile evidence of arbitration loss.
- Treat `s_delay_alu` and `s_waitcnt_depctr` as hazard/scheduling instructions, not decorative optimizations. Add them only from a verified gfx1151 dependency requirement or a measured expert-scheduling experiment.
- Evaluate static issue classes, VOPD operations versus issues, VMEM, LDS, waits, barriers, code size, and resources together. A shorter assembly that slows the rotating bracket is rejected.

### Lower bounds and profiling

For each materially different family, generate the complete fused kernel and, when diagnosis is ambiguous, exact `wmma_floor` and `decode_floor` kernels. The floors preserve launch geometry and declared resources so their differences isolate matrix/A/LDS work from packed decode/LDS work rather than occupancy.

Collect only gfx1151-supported counter groups. Useful evidence includes aggregate wave cycles, wait-count and barrier stalls, VMEM/LDS issue counts, L0/L2 behavior, LDS conflicts, occupancy, and resource allocation. Preserve raw runs when derived-counter combinations exceed hardware collection capacity. Counter values guide the next experiment but never replace same-process event timing.

### Bounded campaign runner

The campaign script operates outside `KernelWriterAssembly` and consumes an explicit inventory plus an explicit candidate list. It does not construct a broad Cartesian product or repair unsupported solutions.

For every exact `(key, candidate)` pair it must:

1. Generate into an immutable output directory, preserve accepted or structured-rejected manifests, and hash only generated source assembly.
2. Build only accepted generation manifests and refuse source-hash mismatches or overwrites.
3. Inspect symbol, ABI, gfx1151/wave32 metadata, register bounds, LDS, WMMAs, waits, barriers, and all hard resource exclusions.
4. Run reduced K32/K64/K96-style tests when a new control-flow, induction, ownership, or pipeline mechanism is introduced.
5. Run exact production-shape comparison against HIP, independent GGUF/BF16 reference checks, and producer-handoff mutation checks where cache policy is affected.
6. Screen accepted candidates with warmed rotating nine-repeat HIP/current/candidate controls in one process.
7. Confirm finalists with fresh 25-repeat rotating brackets and preserve raw samples, median, dispersion, ratio, normalized assembly, and rejection reasoning.

The runner may cache compilation by exact solution and assembly identity, but validation and timing evidence retain their own protocol identity. It never benchmarks while compiling, never times a correctness failure, and never converts absence into rejection. Non-finite timings, malformed result rows, or incomplete rotations fail the observation.

### Measurement and retention gates

Every retained candidate must be exact-shape correct, source-reproducible, and free of private storage, spills, scratch, calls, and dynamic stack. New resource-bearing mechanisms require a stable gain above 2% on their target key and no repeatable regression above 1% on another key that shares the same emitted path. Resource-neutral compiler simplifications may remain when neutral or favorable on both a short-K and long-K control.

Per-key finalists are selected by fresh local brackets. Campaign success is selected by complete-model impact: run all 12 exact keys under the appropriate M workload, weight by checkpoint call counts, and report raw family totals as well as the aggregate. Keep HIP as fallback for any key where assembly does not win. A family improvement does not justify dispatching a slower exact key.

### Deferred directions

Do not prioritize GSU, Stream-K, persistent workgroups, generic DirectToLds, or broad DirectToVgpr controls. They require output fixup/workspace, address a regular-grid deficit not yet observed, or cannot perform cooperative GGUF reconstruction. A is already loaded directly into WMMA-facing VGPRs, while packed B must be decoded before LDS consumption.

Prepared weights, compact alternate layouts, and BF16 shadows remain outside GGTensile kernel tuning because they change model-visible ownership, lifetime, invalidation, and memory contracts. Public integration remains deferred until GGTensile covers the production matrix and quant families well.

Broad PLR2, packed-next-only, K64, two-LDS-buffer without decode overlap, and half-area Q4_K tile sweeps remain closed for the measured pair. Reopen one only when an exact shape or a preceding retained change demonstrably changes its resource or stall regime.

### Campaign sequence

1. Create the machine-readable 12-key inventory with representative tensors, call counts, fresh HIP medians, historical controls, and exact validation requirements. Exact N512/N4096 validation and packed-row addressing are complete; inventory automation remains.
2. Add the bounded runner with immutable generate, build, inspect, correctness, screen, and confirmation evidence.
3. Apply and bracket exact-shape lowering on the selected M32768 K512 and K8192 controls. Dead kernarg dimensions and branch-only fixed-trip reduction are retained; body unroll is rejected, so the current exact-shape loop-lowering stage is complete.
4. Run the retained pipeline and one-buffer control on every compatible key, establish lower bounds only where diagnosis is unclear, and rank open keys by fresh weighted latency and plausible gap.
5. Optimize the two open narrow keys, then interleave attention-output and shared-down work by weighted contribution. Optimize the two open query keys last. Within each key, test traversal before expanding geometry/dataflow and test low-level scheduling only on high-level finalists.
6. Reconfirm every per-key finalist, fall back to HIP where assembly does not win, and run complete weighted Qwen backward correctness and latency across all 12 keys. Update the experiment log after every retained or rejected mechanism.
7. **Final step, always: optimization-exhaustion review.** Re-read every resource used by the campaign: this plan and experiment log; all dense/grouped MMQ optimization records; current HIP and GGTensile sources; normalized HIP and assembly disassemblies; manifests, rejected candidates, lower bounds, profiles, counters, and timing brackets; TensileLite solution, scheduling, LDS, register, and store mechanisms; EvoTensile search and measurement records; FeatherOps/hipBLASLt/CK studies; the gfx1151 ISA reference; and LLVM AMDGPU instruction, VOPD, hazard, delay, and scheduling sources/tests. Classify every newly inferred idea as duplicate/closed, contract-incompatible, unsupported, deferred with an explicit prerequisite, or actionable with a target key, mechanism, expected multi-percent path, and measurement gate. If any actionable idea remains, insert its implementation and experiment immediately before this final step, execute it, record the result, and perform the complete final review again. This review must remain the last step after every plan edit and cannot pass in the same iteration that discovers actionable work. The dense Q4_K optimization campaign is exhausted only when a complete review produces no new valid actionable kernel idea and every prior actionable idea has been retained, rejected, or explicitly deferred outside the current contract.

Quant-family expansion starts only after step 7 reaches that stopping condition. Public runtime dispatch remains a later project.

## Integration And Expansion

Each GGTensile artifact uses a separate exact-problem symbol and does not replace an existing HIP range symbol. Production dispatch may select an assembly artifact only when every exact `ProblemType` and `ProblemSize` assertion and source-assembly identity check matches; otherwise it uses HIP. Cross-shape correctness is deliberately not part of a generated kernel's contract.

After Q4_K passes the full gate, dense backward expands in this order: Q8_0, Q6_K, Q3_K, Q5_K, and IQ2_S. Dense forward is considered only after all six dense-backward types pass complete Qwen and DeepSeek correctness and weighted benchmarks. Grouped MMQ remains deferred until dense assembly demonstrates a useful measured advantage.

Bounded scan scripts operate outside `KernelWriterAssembly`. They may construct, cache, validate, and rank complete solutions for explicit exact keys, but GGTensile continues to provide deterministic `SolutionKey` identity, explainable rejection, isolated generation/build/inspection phases, and immutable evidence manifests. A larger automated search system remains optional and is not needed to complete the 12-shape Q4_K campaign.
