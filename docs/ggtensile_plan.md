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
   - A persistent A-pointer hoist removed repeated exact-shape address work but raised VGPRs from 200 to 204.
   - The hoist measured 45.285 ms versus 45.179 ms for the selected control, a 0.23% regression, and was reverted.
   - SIA4's prefetched first-half A row coordinates and pointers survive fused B decode in existing address VGPRs. Reusing them removes duplicate coordinate reconstruction and derives second-half pointers with two 32-byte increments without increasing live state.
   - A 25-repeat bracket measured pointer reuse at 42.646 ms versus the original SIA4 path at 43.081 ms, a 1.01% latency reduction with unchanged 200 VGPR and 20 SGPR allocation.
   - Retain pointer reuse as an unconditional writer improvement rather than a tuning knob. Independent builds remain byte-reproducible.
   - Q4_K nibble decode originally extracted a byte and then a nibble with two `v_bfe` instructions per value. Four per-lane bit offsets in dead address VGPRs permit one direct nibble extract, removing 28 VALU instructions per DepthU iteration without changing loads, LDS, registers, or numerical order.
   - A 25-repeat bracket measured direct nibble extraction at 40.644 ms and 27.05 TFLOP/s versus 41.210 ms and 26.68 TFLOP/s for the two-step control, a 1.37% latency reduction. Output remains bit-exact and independent builds are byte-reproducible.
   - Retain direct nibble extraction as an unconditional writer improvement. Continue examining shorter-lived affine and decode state that does not increase persistent register pressure.
   - Reordering transient Q4_K scale pointers keeps A row coordinates live in existing address VGPRs across the reduction loop. This hoist removes seven VALU instructions per DepthU iteration without adding registers or changing exact output.
   - A 25-repeat bracket measured the hoisted path at 40.135 ms and 27.40 TFLOP/s versus 40.396 ms and 27.22 TFLOP/s, a 0.65% favorable instruction reduction. Retain it under the neutral-or-better simplification rule.
   - Storing invariant A byte offsets instead of row indices in those same VGPRs removes two more multiplies per DepthU iteration. A 25-repeat bracket measured 40.116 ms and 27.41 TFLOP/s versus 40.247 ms and 27.32 TFLOP/s, a 0.33% resource-neutral gain; retain it.
   - For the retained 128-wide N tile, packed Q-byte offsets map directly from lane bits 0 and 2 plus a scalar tile offset, while scale-byte selection maps from lane bits 1-2. Encoding those mappings directly removes three more VALU instructions per iteration.
   - A 25-repeat bracket measured direct packed-offset mapping at 39.964 ms and 27.51 TFLOP/s versus 40.141 ms and 27.39 TFLOP/s, a 0.44% resource-neutral gain with bit-exact output; retain it.
   - All exact-shape A and packed-weight offsets fit in 32 bits. Using gfx11 scalar-base `global_load` addressing removes vector high-half pointer construction and carry chains while preserving the 64-bit kernarg bases in SGPRs.
   - A 25-repeat bracket measured scalar-base loads at 39.579 ms and 27.78 TFLOP/s versus 39.990 ms and 27.49 TFLOP/s, a 1.03% resource-neutral gain with bit-exact output; retain the address mode.
3. **LDS layout (`completed`)**
   - `LdsSwizzleChunkB={0,8}` emits distinct decoded-store and WMMA-read addressing.
   - Adding 16 logical K positions moves N-row residues 0/1 forward 32 bytes but residues 2/3 backward 32 bytes under XOR-8.
   - XOR-8 is bit-exact and uses 200 VGPRs, 20 SGPRs, and 8192 LDS bytes with no disallowed resources.
   - Under the old all-M traversal, XOR-8 was neutral under SIA2 and 0.90% slower under SIA3.
   - The layout was retested after WGM1 corrected cotangent locality because layout and cache traversal interact.
   - A 25-repeat WGM1/SIA3 bracket measured XOR-8 at 45.136 ms, unpadded at 52.604 ms, and HIP at 47.733 ms.
   - XOR-8 is retained with WGM1 for this exact shape: it is 14.2% faster than unpadded and 5.44% faster than HIP.
   - Profiling reports 68.75% LDS bank conflicts for both selected XOR-8 and HIP, versus 79.17% for unpadded.
   - XOR-16 is bit-exact and reduces allocation to 198 VGPRs, but its conflict ratio returns to 79.17%.
   - A 25-repeat bracket measured XOR-16 at 52.633 ms versus XOR-8 at 45.006 ms, a 16.95% regression, so XOR-16 is rejected for this shape.
   - XOR-4 uses eight bank phases, 204 VGPRs, and four `ds_load_b64` operations per fragment instead of two `ds_load_b128` operations.
   - XOR-4 still reports 68.75% bank conflicts and measures 45.178 ms versus XOR-8 at 45.053 ms, so its doubled LDS issue count has no compensating benefit.
   - XOR-4 is rejected and XOR-8 remains selected.
   - At the LDS-layout milestone, the selected assembly reached approximately 24.36 TFLOP/s and 41.0% of the WMMA roof; later schedule results are recorded below.
4. **Main-loop schedule (`active`)**
   - `ScheduleIterAlg=2` preserves the original full-wait schedule.
   - `ScheduleIterAlg=3` waits for the oldest A and B loads, issues independent WMMAs, and delays full waits until their operands are consumed.
   - The SIA3 solution is bit-exact and remains at 196 VGPRs, 19 SGPRs, and 8192 LDS bytes before the selected XOR-8 layout's additional registers.
   - Before locality correction, a 25-repeat bracket measured SIA3 at 117.981 ms versus SIA2 at 120.101 ms.
   - With WGM1 and XOR-8 selected, SIA3 measured 45.133 ms versus SIA2 at 45.315 ms.
   - `PrefetchLocalRead=2` uses 16 additional VGPRs to ping-pong decoded-B fragments, issuing the next N pair's LDS reads while WMMAs consume the current pair. Half-pair waits preserve SIA3's oldest-ready issue order.
   - The PLR2 kernel is bit-exact and resource-clean at 216 VGPRs, 20 SGPRs, and 8192 LDS bytes. A nine-repeat screen measured 46.521 ms versus 46.120 ms for PLR1, a 0.87% regression.
   - Reject PLR2 for this shape because its additional register pressure does not produce useful latency hiding. The PLR1 source and code object remain byte-identical after the writer refactor.
   - `ScheduleIterAlg=4` issues the first DepthU half's A loads after the older packed-weight loads, waits only for Q4_K data, and overlaps A completion with fused B decode. It requires no additional registers or LDS.
   - SIA4 is bit-exact and passes inspection at 200 VGPRs, 20 SGPRs, 8192 LDS bytes, 32 static WMMAs, and zero disallowed resources. Independent builds produced byte-identical assembly, object, and code object.
   - A 25-repeat rotating bracket measured SIA4 at 42.666 ms and 25.77 TFLOP/s versus SIA3 at 45.137 ms and 24.36 TFLOP/s, a 5.47% latency reduction and 5.79% throughput gain.
   - The same bracket measured HIP at 47.841 ms, so SIA4 is 10.82% lower latency and 12.13% higher throughput. SIA4 reaches 43.38% of the BF16 WMMA roof.
   - The unconditional pointer-reuse improvement is recorded under obvious assembly corrections.
   - `PrefetchGlobalRead=2` adds a second 16-VGPR A fragment set. Both DepthU halves issue behind older packed-weight reads, first-half A overlaps fused decode, and second-half A remains pending through first-half WMMAs.
   - PGR2 is bit-exact and resource-clean at 216 VGPRs, 20 SGPRs, and 8192 LDS bytes. Independent builds produce byte-identical assembly, object, and code object.
   - A 25-repeat bracket measured PGR2 at 40.906 ms and 26.88 TFLOP/s versus PGR1 at 42.723 ms and 25.74 TFLOP/s, a 4.25% latency reduction and 4.44% throughput gain.
   - The same bracket measured HIP at 47.886 ms. PGR2 is 14.58% lower latency and 17.06% higher throughput, reaching 45.25% of the BF16 WMMA roof.
   - `ScheduleIterAlg=5` combines PGR2 with SIA3's oldest-ready A and half-pair B waits. It is bit-exact and resource-identical to SIA4/PGR2.
   - A nine-repeat screen measured SIA5 at 40.753 ms versus SIA4/PGR2 at 41.134 ms, a 0.93% latency reduction that does not clear the retention gate.
   - Reject SIA5 for this shape and retain SIA4 with PGR2 as the selected assembly control.
5. **Tile and ownership geometry (`active`)**
   - Admit focused complete solutions around `MacroTile 128x128, DepthU 32`.
   - The writer now supports a complete `256x64x32` solution with four M16 tiles and four N16 tiles per wave, one decoder-owned packed row per lane, 4 KiB LDS, and geometry-derived register allocation, decode coverage, WMMA ownership, and stores.
   - The `256x64x32` kernel is bit-exact to HIP across all 67,108,864 outputs and passes inspection at 208 VGPRs, 20 SGPRs, 4096 LDS bytes, 32 static WMMAs, and zero disallowed resources.
   - This geometry halves decoded-B work but doubles cotangent A traffic across the expanded N-tile traversal. A 25-repeat bracket measured 66.597 ms versus 45.300 ms for selected `128x128x32` and 48.172 ms for HIP, a 47.0% regression to the assembly control.
   - Reject `256x64x32` for this exact shape and retain `128x128x32`.
   - A complete `256x128x32` solution uses eight waves while keeping two M16 and eight N16 tiles per wave. The first four waves cooperatively decode B, so total packed-weight decode is halved without increasing A traffic.
   - The `256x128x32` kernel is bit-exact and passes inspection at 200 VGPRs, 20 SGPRs, 8192 LDS bytes, 32 static WMMAs, and zero disallowed resources.
   - A nine-repeat screen measured `256x128x32` at 50.469 ms versus 46.021 ms for `128x128x32` and 48.381 ms for HIP, a 9.66% regression to the assembly control. The eight-wave workgroup's scheduling and residency cost outweighs reduced B decode, so this geometry is rejected.
   - The generalized writer rebuilt the selected geometry at 45.446 ms versus 45.456 ms for its prior artifact, confirming neutral performance on the retained path.
   - A complete `64x128x32` solution halves each wave's accumulator footprint and uses 136 VGPRs, 20 SGPRs, and 8192 LDS bytes. It is bit-exact but measured 69.308 ms versus 41.129 ms for `128x128x32`, a 68.5% regression. The doubled workgroup and packed-weight decode count dominates the added residency; reject the smaller M tile.
   - The complementary `128x64x32` solution uses 144 VGPRs, 20 SGPRs, and 4096 LDS bytes. It is bit-exact but measured 46.317 ms versus 41.245 ms for `128x128x32`, a 12.3% regression. Duplicating A traffic over twice as many N workgroups also outweighs the residency gain; reject both half-area orientations.
   - A complete `128x128x64` solution derives the 128-byte LDS row stride from DepthU, decodes four packed rows per lane, emits 64 static WMMAs, and rolls two prefetched A halves after the first 32 reduction positions.
   - DU64 is bit-exact and resource-clean at 236 VGPRs, 20 SGPRs, and 16384 LDS bytes. Its first launch exposed and corrected a reduction induction step that still advanced by 32; exact execution validation remains mandatory beyond assembly and inspection.
   - A 25-repeat bracket measured DU64 at 40.394 ms and 27.22 TFLOP/s versus DU32 at 40.772 ms and 26.97 TFLOP/s, only a 0.93% latency reduction.
   - Reject DU64 because the sub-gate gain does not justify 20 additional VGPRs and doubled LDS. Retain the four-wave `128x128x32` geometry; the generalized DU32 source and code object remain byte-identical.
6. **Global traversal (`completed`)**
   - The writer enables the workgroup-Z system SGPR and maps `m_block = blockIdx.z * WorkGroupMapping + blockIdx.x`.
   - The runtime launches X as `WorkGroupMapping`, Y as the 16 N tiles, and Z as the remaining M groups.
   - WGM1 reduced SIA3 from 119.36 ms for the all-M control to 52.40 ms, a 2.28x speedup, by scheduling all N workgroups for one M tile together.
   - Nine-repeat screens measured WGM2 at 54.97 ms, WGM4 at 56.86 ms, and WGM8 at 57.46 ms; WGM1 won every comparison.
   - A 25-repeat WGM1/WGM2/HIP bracket measured 52.588/54.307/47.811 ms.
   - WGM1 sustains 20.91 TFLOP/s and 35.2% of the WMMA roof, and remains 10.0% slower than HIP.
   - WGM1 is retained for this exact shape; traversal remains an explicit complete-solution parameter rather than a global rule.
7. **A and packed-weight traffic (`active`)**
   - Compare hipcc and GGTensile load widths, lane duplication, address induction, cache flags, and waits.
   - Raw selected-kernel counters place wait-count stalls at approximately 17.9-19.0% of aggregate wave cycles and barrier stalls at approximately 8.0-8.1%. VALU and LDS instruction-cycle counters are much smaller fractions; these are workload-wide ratios, not mutually exclusive cycle attribution.
   - `PrefetchPackedWeight` retains its existing current-tile meaning. The separate `PrefetchPackedWeightNext` mechanism issues the next packed Q4_K tile before current WMMAs, lets VMEM run while current LDS fragments are consumed, then decodes into the same LDS after the read-side barrier.
   - The next-tile pipeline is bit-exact and remains at 216 VGPRs, 20 SGPRs, and 8192 LDS bytes. It adds no dynamic barrier relative to the two-barrier-per-DepthU steady state.
   - A nine-repeat screen measured next-tile prefetch at 42.345 ms versus 42.410 ms for the selected control, only 0.15% lower latency. Reject it for this shape because packed-weight VMEM overlap is not a material remaining limit.
   - `PackedWeightLaneShare=2` loads each duplicated 16-byte packed-q span only on low-nibble lanes and replicates it to the corresponding high-nibble lanes. It halves packed-q bytes while leaving lane-specific scale/min traffic unchanged.
   - The LDS-crossbar implementation uses eight `ds_bpermute_b32` operations per DepthU iteration and measured 42.638 ms versus 42.152 ms, a 1.15% regression.
   - The VALU-crossbar implementation replaces the LDS operations and wait with DPP shifts plus conditional selection. It measured 43.129 ms versus 42.335 ms, a 1.88% regression.
   - Both lane-sharing implementations are bit-exact and resource-identical to the 216-VGPR control, but their cross-lane work costs more than the saved packed-q traffic. Reject lane sharing for this shape.
   - Continue with wider aligned Q4_K loads, scalar uniform metadata, and bounded decode reuse. Expose a knob only if multiple correct mechanisms remain competitive.
8. **Epilogue and low-level scheduling (`active`)**
   - `StorePriorityOpt=false` removes the two epilogue `s_setprio` instructions.
   - A 25-repeat bracket measured no priority at 45.327 ms versus priority at 45.413 ms.
   - The 0.19% difference is not independently significant; no priority is the provisional control because it is simpler and showed no regression.
   - Store order, `NumElementsPerBatchStore`, and other epilogue changes remain secondary to the 256-iteration main loop.
9. **Advanced exact-shape mechanisms (`pending`)**
   - Consider persistent traversal, decode-sharing across M tiles, or split reduction only after profiling identifies the remaining limit.
   - Split-K/Stream-K requires an explicit FP32 fixup contract before it becomes a parameter.
   - Never introduce a dense shadow weight or external decode workspace.
10. **Retention and integration (`pending`)**
   - Bracket finalists with warmed rotating 25-repeat controls.
   - Require accepted correctness, byte-identical rebuilds, no disallowed resources, and a stable gain above 2%.
   - Retain the fastest exact solution, add static dispatch with HIP fallback, and rerun the full suite and target benchmark.

Tuning parameters are added conservatively. Existing schema names remain rejected at non-pilot values until a distinct correct writer path exists. `LdsSwizzleChunkB`, active-wave ownership, and persistent or decode-sharing policies require explicit semantics and linked validation. Requested solutions are never silently repaired.

## Production M32768 N2048 K512 Result

The selected WGM1, XOR-8, SIA4, PGR2, no-store-priority solution was also built for `ProblemSize(M=32768, N=2048, K=512)` and tested on real `blk.5.ffn_gate_shexp.weight`. This is the dominant Q4_K narrow geometry with 70 model calls.

- All 67,108,864 candidate outputs match HIP bit-for-bit.
- Candidate and HIP have the same 16,684 differences versus independently dequantized BF16 matmul, maximum absolute error 0.00390625, and normalized RMSE 0.0000345316.
- The initial PGR1 bracket measured SIA4 at 2.755 ms and 24.94 TFLOP/s, SIA3 at 2.848 ms and 24.13 TFLOP/s, and HIP at 3.045 ms and 22.57 TFLOP/s.
- A subsequent 25-repeat bracket measured PGR2 at 2.702 ms and 25.43 TFLOP/s, PGR1 at 2.753 ms and 24.96 TFLOP/s, and HIP at 3.025 ms and 22.71 TFLOP/s.
- PGR2 reduces latency by 1.86% versus PGR1 and 10.69% versus HIP. At 70 calls, the direct candidate-to-HIP delta is approximately 22.6 ms per complete model workload.
- The selected exact artifact uses 216 VGPRs, 20 SGPRs, 8192 LDS bytes, and zero disallowed resources.

This result clears the per-shape performance gate and demonstrates that the K8192 schedule is not overfit to a long reduction. Production integration still waits for guarded runtime dispatch, immutable manifests, and complete workload validation.

## Integration And Expansion

The pilot uses a separate exact-problem symbol and does not replace the existing HIP `in_features=2048` range symbol. Production dispatch may select an assembly artifact only when every exact `ProblemType` and `ProblemSize` assertion and artifact-identity check matches; otherwise it uses HIP.

After Q4_K passes the full gate, dense backward expands in this order: Q8_0, Q6_K, Q3_K, Q5_K, and IQ2_S. Dense forward is considered only after all six dense-backward types pass complete Qwen and DeepSeek correctness and weighted benchmarks. Grouped MMQ remains deferred until dense assembly demonstrates a useful measured advantage.

Future automated search should operate outside `KernelWriterAssembly`. It may construct, repair, mutate, cache, validate, and rank complete solutions, but GGTensile continues to provide deterministic `SolutionKey` identity, explainable rejection, isolated generation/build/inspection phases, and immutable evidence manifests.
