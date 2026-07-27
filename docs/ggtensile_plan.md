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

Normalized assembly and HSACO SHA-256 values are recorded. Building the same candidate twice must produce byte-identical artifacts before it is eligible for integration.

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

1. **Baseline and harness (`completed`)**: solution `ggsol_13512e1fccbf43e7` generated assembly SHA-256 `153c890c31b5fa031016b261f7f6a7393b4b39d6af5008884c927f2f5d659715`, code object SHA-256 `5a33bd66b727315ff423af43bd5712a637be53fae20bc443d8abfe64b16de921`, and normalized ISA SHA-256 `73e58cc21d555654c67637fa24e2e75b31e6068bc0487e9024a459b939c20fdd`. Inspection reports 196 VGPRs, 19 SGPRs, 8192 LDS bytes, 32 static WMMAs, zero private storage/spills/scratch/calls/dynamic stack, and the required gfx1151 code-object-v5 ABI. On real `blk.39.attn_q.weight` and deterministic random BF16 cotangent, all 67,108,864 BF16 outputs match HIP bit-for-bit; candidate and HIP have the same 409,880 differences versus independently dequantized BF16 matmul, maximum absolute error 0.0625, and normalized RMSE 0.0001924804. Nine warmed alternating samples measured HIP at median 47.61 ms, 23.09 TFLOP/s, and 38.9% of roof versus assembly at median 122.90 ms, 8.95 TFLOP/s, and 15.1% of roof. The 2.58x latency deficit makes LDS conflicts and scheduling the immediate blockers. The reusable driver is `tools/benchmark_ggtensile.py`; the full baseline report is `/tmp/ggtensile-m32768-n2048-k8192-baseline/benchmark.json`.
2. **Obvious assembly corrections (`active`)**: remove full-wait serialization that has no dependency, hoist exact-shape affine address state, avoid repeated waits and arithmetic inside the K loop, and use the retained Q4_K query LDS bank-conflict solution. Changes without a meaningful alternative remain implementation improvements rather than tuning knobs.
3. **LDS layout and local-read schedule (`pending`)**: implement honest `LdsPadB`, `LdsBlockSizePerPadB`, and a GGUF-specific `LdsSwizzleChunkB` path; compare unpadded, padding-8, XOR-8, and only valid combinations. Add `PrefetchLocalRead` schedules that double-buffer B fragments and overlap DS reads with independent WMMAs. Reject layouts that do not preserve exact WMMA lane mapping.
4. **Main-loop schedule (`pending`)**: implement distinct `ScheduleIterAlg` paths for decode/global-read, LDS write, A read, B local read, and WMMA ordering. Sweep `ScheduleIterAlg`, `PrefetchGlobalRead`, `PrefetchPackedWeight`, and `1LDSBuffer` only where emitted ISA differs. The first neighborhood is current serial scheduling, local-read pipelining, and a bounded next-DepthU packed/A prefetch; prior HIP evidence warns that cross-iteration packed prefetch can lose through VGPR lifetime.
5. **Tile and ownership geometry (`pending`)**: admit complete valid solutions around `MacroTile 128x128, DepthU 32`, then focused `128x64`, `64x128`, `256x64`, and DepthU 16/64 variants where register allocation, decode ownership, and LDS coverage are implemented. Model wave ownership explicitly through `MatrixInstruction` and `WorkGroup`; do not infer performance from nominal tile area. Accumulator pressure, active waves, workgroup residency, repeated decode, and A/B reuse are measured together.
6. **Global traversal (`pending`)**: implement exact-shape `WorkGroupMapping`/`GroupM` mappings that change launch-to-tile traversal while preserving one workgroup per output tile. Compare current two-dimensional order with focused M clustering and static mapping factors 1/2/4/8. The large M dimension makes packed-weight and cotangent L2 reuse a first-order concern, but prior HIP results show mappings are geometry-specific.
7. **A and packed-weight traffic (`pending`)**: inspect hipcc's gfx1151 lowering for Q4_K and compare global load widths, lane duplication, address induction, cache flags, and wait placement. Evaluate half-wave load plus legal lane replication for WMMA A/B operands, wider aligned Q4_K header/quant loads, scalarized uniform metadata, and bounded decode reuse. These become knobs only if multiple correct emitted mechanisms remain competitive.
8. **Epilogue and low-level scheduling (`pending`)**: tune store order, `NumElementsPerBatchStore`, `StorePriorityOpt`, wait placement, instruction priority, and independent VALU placement only after the main loop is competitive. TensileLite's SIA3/no-store-priority result is mechanism evidence, not a value to copy blindly; this target's 256 dynamic reduction iterations make epilogue tuning secondary.
9. **Advanced exact-shape mechanisms (`pending`)**: consider persistent tile traversal, decode-sharing across multiple M tiles, or split reduction only if profiles show a remaining launch/locality/repeated-decode limit. Split-K/Stream-K requires an explicit FP32 fixup contract and is not admitted as a parameter beforehand. No dense shadow weight or external decode workspace is allowed.
10. **Retention and integration (`pending`)**: bracket finalists with warmed alternating 25-repeat HIP/assembly/HIP runs, require bit-exact HIP agreement or a justified fixed bound, byte-identical rebuilds, zero private storage/spills/scratch/calls/dynamic stack, and a stable gain above 2%. Retain the fastest exact solution, add static dispatch with HIP fallback, and rerun the full project suite and target benchmark.

Tuning parameters are added conservatively. `MacroTile`, `DepthU`, `MatrixInstruction`, `WorkGroup`, `GlobalReadVectorWidthA/B`, `LocalReadVectorWidth`, `PrefetchGlobalRead`, `PrefetchLocalRead`, `1LDSBuffer`, `ScheduleIterAlg`, `StorePriorityOpt`, `NumElementsPerBatchStore`, `StoreVectorWidth`, `WorkGroupMapping`, `TransposeLDS`, `LdsPadB`, `LdsBlockSizePerPadB`, `DecoderWidth`, and `PrefetchPackedWeight` already exist in the schema but remain rejected at non-pilot values until a distinct correct writer path exists. `LdsSwizzleChunkB`, active-wave ownership, and any persistent/decode-sharing policy are added only with explicit semantics and linked validation. Requested solutions are never silently repaired.

## Integration And Expansion

The pilot uses a separate exact-problem symbol and does not replace the existing HIP `in_features=2048` range symbol. Production dispatch may select an assembly artifact only when every exact `ProblemType` and `ProblemSize` assertion and artifact-identity check matches; otherwise it uses HIP.

After Q4_K passes the full gate, dense backward expands in this order: Q8_0, Q6_K, Q3_K, Q5_K, and IQ2_S. Dense forward is considered only after all six dense-backward types pass complete Qwen and DeepSeek correctness and weighted benchmarks. Grouped MMQ remains deferred until dense assembly demonstrates a useful measured advantage.

Future automated search should operate outside `KernelWriterAssembly`. It may construct, repair, mutate, cache, validate, and rank complete solutions, but GGTensile continues to provide deterministic `SolutionKey` identity, explainable rejection, isolated generation/build/inspection phases, and immutable evidence manifests.
