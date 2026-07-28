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

The command-line interface accepts strict JSON solution files, writes a manifest for both accepted and rejected solutions, and has separate `generate`, `build`, and `inspect` commands. `generate` runs `KernelWriterAssembly`; `build` accepts only an accepted generate manifest and verifies the source assembly hash before invoking the assembler and linker; `inspect` accepts only an accepted build manifest and validates the resulting code object. Every phase refuses to overwrite an existing artifact. This phase separation keeps interfaces suitable for a future EvoTensile-style search scheduler and compile cache.

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

## Q4_K Pilot Case Study

The first measured case study covers `ProblemSize(M=32768, N=2048, K=8192)` and the companion production reduction `K=512`. It established exact Q4_K decoding, WGM1 traversal, the XOR-8 LDS layout, SIA4/PGR2 scheduling, scalar-base global addressing, and the current 212-VGPR resource-clean implementation.

The K8192 path improved from the initial 122.90 ms assembly baseline to approximately 40 ms and now outperforms the HIP control. The K512 companion measures approximately 2.653 ms versus HIP at 3.030 ms while preserving the independently verified numerical envelope. Detailed measurements, retained/rejected candidates, resources, and remaining exact-shape experiments are recorded in [ggtensile_q4_k_m32768_n2048_case_study.md](ggtensile_q4_k_m32768_n2048_case_study.md).

This case study is evidence for the generator and search process, not a universal solution. Geometry, decoder ownership, LDS layout, traversal, scheduling, and epilogue policy remain shape- and quant-specific in the multi-shape roadmap below.

## Multi-Shape GGTensile Tuning Plan

### Performance position

The current selected Q4_K kernel sustains approximately `27.5-27.8 TFLOP/s`, or `46-47%` of the gfx1151 BF16 WMMA roof of approximately `59.4 TFLOP/s`. It reaches about `61-62%` of the approximately `45 TFLOP/s` delivered by a well-tuned hipBLASLt BF16 GEMM on this machine. The hipBLASLt result is an upper reference rather than an immediate fused-kernel target because GGTensile repeatedly loads packed data, reconstructs BF16 weights, stages LDS, and synchronizes before WMMA.

The existing HIP optimization logs have already closed broad ordinary-geometry, K64, two-LDS-buffer, decoder-width, local-prefetch, and traversal sweeps. GGTensile should revisit a closed direction only when direct assembly control creates a materially different mechanism. The strongest untested distinction is gfx1151 dual-issue and dependency scheduling: hipcc emits extensive VOPD, `s_clause`, and `s_delay_alu` scheduling for the HIP Q4_K kernel, while the current handwritten GGTensile path emits no VOPD.

ROCm rocm-libraries PR 9385 provides additional gfx1151 evidence. Its relevant transferable mechanisms are correct sub-dword WMMA local reads, capping local-read buffers by actual loop iterations, placing long-lived values before transient address registers, and using interaction-based shape tuning rather than one universal configuration. GGTensile should port those principles where they match the fused decoder rather than copying general GEMM code paths.

### Proposed tuning controls

Every new control must change emitted ISA or resource ownership, participate in strict solution identity and validation, and remain rejected at unsupported values. Controls should use TensileLite terminology where the mechanism matches and explicit GGUF terminology where packed decode differs.

- Main-loop scheduling: add structured controls for global reads per WMMA group, local reads per WMMA group, decode operations per WMMA group, WMMA group size, and explicit VMEM/LDS wait budgets. Numeric `ScheduleIterAlg` values should map to named, inspectable issue schedules rather than accumulating opaque special cases.
- Pipeline ownership: separate A prefetch lead, packed-Q prefetch lead, and metadata prefetch lead. Add `LdsBufferCount` and `DecodeAhead` only with complete implementations. Initial useful ranges are A lead `{1,2}`, decode lead `{0,1}`, and LDS buffers `{1,2}`.
- Decoder ownership: add active compute waves, decoder waves, decoder width, and metadata-sharing method. Initial wave counts are `{1,2,4}` and decoder widths are `{8,16,32}` only where a quant-specific implementation provides complete decode coverage.
- Geometry: retain `MacroTile`, `MIWaveTile`, `MIWaveGroup`, and `DepthU` as coupled controls. Candidate M/N dimensions come from `{32,64,128,256}` after exact divisibility filtering; initial DepthU values are `{16,32,64}`.
- Traversal: distinguish grouped-M traversal, M/N launch order, and exact `WorkGroupMapping`. K staggering is lower priority and should be admitted only after profiling shows cache-set or partition contention rather than ordinary packed-weight reuse.
- Global reads: separate A vector width, packed-payload vector width, and metadata vector width. Add scalar-global versus buffer-SRD addressing and cache policy only when both forms are implemented and inspected.
- LDS: retain swizzle chunk as one layout dimension and add row padding, block-size-per-pad, and local-read grouping where they produce distinct address formulas. Layout remains quant-, geometry-, and shape-specific.
- Epilogue: implement `NumElementsPerBatchStore` as a real store-scheduling control, along with store traversal and rounding/store interleave. Initial batch sizes are `{4,8,10,16,32}`. `StorePriorityOpt` must be retuned jointly with the selected main-loop and store schedule.
- ISA lowering: add internal policies for VOPD pairing, VMEM clause formation, dependency-aware ordering, and register allocation modulo constraints. These should begin as unconditional lowering passes or internal experiments, not public tuning knobs.

### Unconditional assembly work

The first experiments should improve generated ISA without adding resources or changing mathematical order.

- Implement a gfx1151 VOPD pairing pass. Start with accumulator zeroing, independent address arithmetic, scale/min unpacking, and prologue/epilogue operations. Pairing must enforce wave32, opcode-slot, source-bank, literal-sharing, destination, and old-value read rules from the gfx11 VOPD specification.
- Make register allocation VOPD-aware. Preserve the current value-first order of accumulators, A fragments, B fragments, and transient state while selecting compatible register modulo classes for profitable pairs.
- Compare hipcc and GGTensile blocks as scheduling evidence. Transfer legal VOPD pairs, memory clauses, and issue distances, but do not copy compiler-generated pointer state or its larger allocation.
- Add measured `s_clause` candidates around the four A loads and packed-Q/metadata load groups. Clause length is retained only when timing and counters improve; lower instruction count alone is insufficient.
- Audit `buffer_gl0_inv`. It invalidates vector L0 rather than LDS. An omission candidate is legal only after proving that the corresponding barrier protects LDS-only producer/consumer traffic, all global inputs are immutable, and exact execution remains correct across all target shapes.
- Centralize fragment-buffer allocation and cap each buffer by actual loop iterations and modulo reuse, following the mechanism validated by PR 9385.
- Extend direct lane/offset formulas and scalar-base addressing to every quant decoder before adding broader address-mode knobs.

### True decoded-B pipeline

The largest high-level opportunity is a true two-buffer decoded-B pipeline. The rejected packed-next experiment moved only VMEM reads, and DepthU64 reduced loop/barrier frequency without overlapping next-tile decode with current WMMA. Neither tested the complete mechanism.

The intended pipeline primes B0 in LDS0, consumes LDS0 while loading and decoding B1 into LDS1, issues A at a measured lead, waits only when a specific fragment becomes live, and performs one buffer-swap barrier per DepthU tile instead of separate decode-ready and LDS-reuse barriers. Current packed-Q, d/min, and scale registers are dead after decode and can normally be reused for the next tile, so the primary fixed cost should be an additional 8 KiB LDS rather than a second complete packed-register set.

Decode should be interleaved in bounded chunks between current-tile WMMA pairs. Candidate schedules vary the number of decoded values between WMMA groups and the first-use wait thresholds. Retention requires a stable gain above 2% because the mechanism adds LDS and scheduling complexity.

### Shape and quant strategy

GGTensile should tune shape families rather than seek one universal kernel. The 1,135-shape EvoTensile campaign retained many solution families, and the local dense/grouped logs show the same shape dependence.

The first multi-shape Q4_K campaign covers `(N,K)=(2048,8192)`, `(2048,512)`, `(4096,2048)`, and `(512,2048)` in GGTensile GEMM coordinates, at production M values `2048`, `8192`, and `32768` where each shape exists. Long-K kernels emphasize steady-loop scheduling, decode overlap, and traversal. K512 kernels emphasize prologue, epilogue, active waves, and store batching.

After Q4_K, expand in the established order: Q8_0, Q6_K, Q3_K, Q5_K, then IQ2_S. Each quant family owns its packed-read width, metadata reuse, extraction method, decoder ownership, and LDS layout. Existing HIP winners are seeds rather than conclusions: Q8_0 G2/padding/traversal, Q6 small-M geometries, and Q3/Q5 extraction choices define the first neighborhoods.

Small-M language-model-head shapes require a separate active-wave and geometry campaign. GSU, Stream-K, and persistent grids remain deferred unless a small-M/large-K profile demonstrates inadequate workgroup parallelism and the project explicitly accepts the required fixup or workspace contract.

### Lower bounds and profiling

Before a broad search, build three exact-shape lower-bound kernels: WMMA consuming an already staged decoded-B tile, packed decode plus LDS without WMMA, and the complete fused kernel. These isolate matrix-pipeline, decode, LDS, and synchronization floors and prevent tuning the wrong subsystem.

Collect static and dynamic evidence for VOPD eligibility, VALU issue count, WMMA issue density, VMEM wait cycles, LDS wait cycles, barriers, L0/L2 behavior, LDS conflicts, occupancy, and resource allocation. Use only counter groups supported together by gfx1151; preserve narrower raw-counter runs when derived combinations exceed hardware collection capacity.

The initial target is a reproducible move above `30 TFLOP/s` on K8192 without sacrificing K512. Approximately `35 TFLOP/s` is an ambitious second-stage target for VOPD plus a true decoded-B pipeline. The `45 TFLOP/s` dense GEMM result remains a comparison ceiling, not an acceptance requirement for fused packed decode.

### Deferred directions

Do not prioritize GSU, Stream-K, persistent workgroups, generic DirectToLds, or broad DirectToVgpr controls. They require output fixup/workspace, address a regular-grid deficit not yet observed, or cannot perform cooperative GGUF reconstruction. A is already loaded directly into WMMA-facing VGPRs, while packed B must be decoded before LDS consumption.

Prepared weights, compact alternate layouts, and BF16 shadows remain outside GGTensile kernel tuning because they change model-visible ownership, lifetime, invalidation, and memory contracts. Public integration remains deferred until GGTensile covers the production matrix/quant families well.

Broad PLR2, packed-next-only, K64, two-LDS-buffer without decode overlap, and half-area Q4_K tile sweeps remain closed. Reopen one only if VOPD, register compaction, or the complete pipeline changes its resource or stall regime.

### Campaign sequence

1. Establish per-shape lower bounds and a same-process profile for the current selected artifacts.
2. Implement and bracket unconditional VOPD, clause, dependency-ordering, and cache-invalidation candidates on K8192 and K512.
3. Add structured schedule controls and search small coupled grids around the selected SIA4/PGR2 path.
4. Implement the true two-LDS decoded-B pipeline and retain it only if the measured gain pays for the added LDS and complexity.
5. Complete the four Q4_K production shape families, including M-dependent active-wave and traversal policies.
6. Expand quant coverage in the Q8_0, Q6_K, Q3_K, Q5_K, and IQ2_S order.
7. Run an EvoTensile-style measured campaign with exact candidate-shape evidence, candidate-family promotion, workload weighting, and preserved rejection records.
8. Finalize only after fresh rotating brackets and complete Qwen/DeepSeek workload validation; public runtime dispatch remains a later project.

Screens use warmed rotating nine-repeat controls, followed by 25-repeat brackets for finalists. New resource-bearing mechanisms require accepted correctness, reproducible source assembly, no private storage/spills/scratch/calls/dynamic stack, and a stable gain above 2%. Unconditional instruction or resource reductions may be retained when neutral or favorable across both long-K and short-K controls.

## Integration And Expansion

The pilot uses a separate exact-problem symbol and does not replace the existing HIP `in_features=2048` range symbol. Production dispatch may select an assembly artifact only when every exact `ProblemType` and `ProblemSize` assertion and source-assembly identity check matches; otherwise it uses HIP.

After Q4_K passes the full gate, dense backward expands in this order: Q8_0, Q6_K, Q3_K, Q5_K, and IQ2_S. Dense forward is considered only after all six dense-backward types pass complete Qwen and DeepSeek correctness and weighted benchmarks. Grouped MMQ remains deferred until dense assembly demonstrates a useful measured advantage.

Future automated search should operate outside `KernelWriterAssembly`. It may construct, repair, mutate, cache, validate, and rank complete solutions, but GGTensile continues to provide deterministic `SolutionKey` identity, explainable rejection, isolated generation/build/inspection phases, and immutable evidence manifests.
