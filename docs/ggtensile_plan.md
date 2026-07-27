# GGTensile Implementation Plan

## Purpose

GGTensile is a repository-local assembly kernel generator for packed GGUF matrix multiplication. It takes one `ProblemType`, one exact `ProblemSize`, and one complete `Solution`, then either emits a reproducible gfx1151 code object or returns structured rejection reasons. The first kernel is dense MMQ backward for Q4_K. It computes

```text
grad_input[rows, in_features] = grad_output[rows, out_features] @ dequant(weight[out_features, in_features])
```

without materializing a dense or transposed weight. The existing HIP kernel is the correctness oracle, performance control, and runtime fallback.

The implementation intentionally uses a small ROCISA surface inspired by TensileLite: structured code modules and metadata, explicit register pools, and direct assembler/linker invocation. It does not import TensileLite's solution, problem-type, or library-generation machinery. "Full Tensile" is reserved for rocBLAS and source identifiers.

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

The command-line interface accepts JSON files or an inline pilot problem, writes a manifest for both accepted and rejected solutions, and has separate `generate`, `build`, and `inspect` commands. `generate` runs `KernelWriterAssembly`; `build` invokes the assembler and linker; `inspect` validates the resulting code object. This phase separation keeps interfaces suitable for a future EvoTensile-style search scheduler and compile cache.

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

The toolchain resolver accepts explicit paths and otherwise discovers the ROCm SDK shipped with the active Python environment. It records compiler, assembler, linker, ROCISA, target, and generator identities in the manifest. Builds use code-object v5 and deterministic commands. Temporary files live outside the source tree, and output installation is atomic.

The artifact inspector requires:

- exactly one expected global kernel symbol;
- gfx1151 code-object-v5 metadata matching the 40-byte kernarg ABI;
- wavefront size 32, 128-thread maximum workgroup size, and expected LDS;
- no private segment, dynamic stack, scratch instructions, calls, or spills;
- no register indices beyond metadata declarations;
- the expected static WMMA and barrier structure.

Normalized assembly and code-object identities stay in the generated inspection manifest. Building the same candidate twice must produce byte-identical artifacts before it is eligible for integration.

## Correctness And Performance Gates

Unit tests cover canonical identity, JSON round trips, strict schema parsing, every rejection rule, deterministic source generation, tool discovery, and artifact inspection parsers. GPU validation is a separate explicit command so ordinary unit tests do not require ROCm hardware.

Pilot correctness compares the generated kernel with both the existing HIP dense-backward kernel and an independently dequantized BF16 reference. It includes one-hot Q4_K blocks spanning every scale group and nibble plane, random packed weights, tile-boundary rows, and production row counts. Accepted results are bit-exact where the accumulation order matches; otherwise the error bound must be justified and fixed in the validation protocol.

Timing is performed only after correctness passes, using the same tensors and launch geometry for warmed alternating control/candidate samples. Report raw samples, median, dispersion, and candidate/control ratio. Expansion requires:

- zero correctness failures;
- zero private storage, spills, scratch, calls, or dynamic stack;
- byte-identical rebuilds;
- no stable regression above 1%; and
- at least a stable 2% Q4_K `in_features=2048` gain, or equivalent latency with a useful VGPR reduction.

## Exact M32768 N2048 K8192 Optimization Campaign

Status legend: `pending`, `active`, `completed`, `rejected`, and `retained` describe measured campaign state; every completed step records its artifact, correctness result, resources, and timing before the next step begins.

The exact target is `ProblemSize(M=32768, N=2048, K=8192)`, corresponding to BF16 `grad_output[32768,8192]`, packed Q4_K logical weight `[8192,2048]`, and BF16 `grad_input[32768,2048]`. It performs 1,099,511,627,776 logical FLOPs. The machine's approximately 59.4 TFLOP/s BF16 WMMA roof gives an 18.51 ms compute floor. The existing HIP kernel's historical approximately 48 ms latency is the first control, not the final target.

Campaign procedure:

1. **Baseline and harness (`completed`)**
   - Inspection reports 196 VGPRs, 19 SGPRs, 8192 LDS bytes, 32 static WMMAs, zero disallowed resources, and the required gfx1151 code-object-v5 ABI.
   - All 67,108,864 BF16 outputs match HIP bit-for-bit on real `blk.39.attn_q.weight` and deterministic random cotangent.
   - Candidate and HIP have the same 409,880 differences versus independently dequantized BF16 matmul, maximum absolute error 0.0625, and normalized RMSE 0.0001924804.
   - Nine warmed alternating samples measured HIP at 47.61 ms and 23.09 TFLOP/s versus assembly at 122.90 ms and 8.95 TFLOP/s.
   - The reusable driver is `tools/benchmark_ggtensile.py`; the full report is `/tmp/ggtensile-m32768-n2048-k8192-baseline/benchmark.json`.
2. **Obvious assembly corrections (`active`)**
   - Remove waits that have no current dependency.
   - Hoist exact-shape affine address state and repeated K-loop arithmetic.
   - Treat unconditional improvements as writer changes rather than tuning knobs.
3. **LDS layout (`completed`)**
   - `LdsSwizzleChunkB={0,8}` now emits distinct decoded-store and WMMA-read addressing.
   - Adding 16 logical K positions moves N-row residues 0/1 forward 32 bytes but residues 2/3 backward 32 bytes under XOR-8.
   - The XOR-8 solution is bit-exact and uses 200 VGPRs, 19 SGPRs, and 8192 LDS bytes with no disallowed resources.
   - A 25-repeat rotating bracket measured XOR-8 at 121.383 ms versus unpadded at 121.299 ms under SIA2, a 0.07% regression.
   - Under SIA3, XOR-8 measured 120.035 ms versus unpadded at 118.962 ms, a 0.90% regression.
   - XOR-8 is rejected for this geometry because it adds four VGPRs without a stable latency gain; the correct mechanism remains available to future solutions.
4. **Main-loop schedule (`active`)**
   - `ScheduleIterAlg=2` preserves the original full-wait schedule.
   - `ScheduleIterAlg=3` waits for the oldest A and B loads, issues independent WMMAs, and delays full waits until their operands are consumed.
   - The SIA3 solution is bit-exact and remains at 196 VGPRs, 19 SGPRs, and 8192 LDS bytes.
   - A 25-repeat rotating bracket measured SIA3 at 117.981 ms versus SIA2 at 120.101 ms, a 1.80% throughput gain.
   - SIA3 is the current assembly control, but it remains below the 2% finalist gate.
   - Next compare cross-pair local-read prefetch, bounded next-DepthU packed/A prefetch, and only schedules that emit distinct ISA.
5. **Tile and ownership geometry (`pending`)**
   - Admit focused complete solutions around `MacroTile 128x128, DepthU 32`.
   - Implement `128x64`, `64x128`, `256x64`, and DepthU 16/64 only with consistent wave ownership, decode coverage, LDS, and register allocation.
   - Measure accumulator pressure, workgroup residency, repeated decode, and A/B reuse together.
6. **Global traversal (`pending`)**
   - Implement exact `WorkGroupMapping`/`GroupM` traversal without changing output ownership.
   - Compare the current order with focused static factors 1/2/4/8.
   - Treat packed-weight and cotangent L2 reuse as geometry-specific measured effects.
7. **A and packed-weight traffic (`pending`)**
   - Compare hipcc and GGTensile load widths, lane duplication, address induction, cache flags, and waits.
   - Evaluate legal half-wave replication, wider aligned Q4_K loads, scalar uniform metadata, and bounded decode reuse.
   - Expose a knob only if multiple correct mechanisms remain competitive.
8. **Epilogue and low-level scheduling (`pending`)**
   - Tune store order, `NumElementsPerBatchStore`, `StorePriorityOpt`, waits, and instruction priority only after the main loop is competitive.
   - Treat TensileLite's SIA3/no-store-priority result as mechanism evidence rather than a value to copy.
9. **Advanced exact-shape mechanisms (`pending`)**
   - Consider persistent traversal, decode-sharing across M tiles, or split reduction only after profiling identifies the remaining limit.
   - Split-K/Stream-K requires an explicit FP32 fixup contract before it becomes a parameter.
   - Never introduce a dense shadow weight or external decode workspace.
10. **Retention and integration (`pending`)**
   - Bracket finalists with warmed rotating 25-repeat controls.
   - Require accepted correctness, byte-identical rebuilds, no disallowed resources, and a stable gain above 2%.
   - Retain the fastest exact solution, add static dispatch with HIP fallback, and rerun the full suite and target benchmark.

Tuning parameters are added conservatively. Existing schema names remain rejected at non-pilot values until a distinct correct writer path exists. `LdsSwizzleChunkB`, active-wave ownership, and persistent or decode-sharing policies require explicit semantics and linked validation. Requested solutions are never silently repaired.

## Integration And Expansion

The pilot uses a separate exact-problem symbol and does not replace the existing HIP `in_features=2048` range symbol. Production dispatch may select an assembly artifact only when every exact `ProblemType` and `ProblemSize` assertion and artifact-identity check matches; otherwise it uses HIP.

After Q4_K passes the full gate, dense backward expands in this order: Q8_0, Q6_K, Q3_K, Q5_K, and IQ2_S. Dense forward is considered only after all six dense-backward types pass complete Qwen and DeepSeek correctness and weighted benchmarks. Grouped MMQ remains deferred until dense assembly demonstrates a useful measured advantage.

Future automated search should operate outside `KernelWriterAssembly`. It may construct, repair, mutate, cache, validate, and rank complete solutions, but GGTensile continues to provide deterministic `SolutionKey` identity, explainable rejection, isolated generation/build/inspection phases, and immutable evidence manifests.
