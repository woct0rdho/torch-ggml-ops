# Grouped MMQ forward optimization

## Scope

This document covers routed grouped MMQ forward on gfx1151.

The production operators are:
- `grouped_mmq_pair` for gate and up.
- `grouped_mmq` for down.

Dense MMQ forward and backward are documented separately in `docs/mmq_fwd_optimization.md` and `docs/mmq_bwd_optimization.md`.

The grouped optimization pass is complete for the current packed representation. The original spill-heavy baseline and all experiment logs remain below for provenance, but the source of record is the spill-free G11 dispatch and `/tmp/grouped_mmq_fwd_final_full_v2.json`.

## Current status

Done:
- specialized production gate/up and down shapes at compile time with `I=64`, `J=64`, and 128 threads.
- removed all production private segments and register spills.
- split exact full-row and bounded tail bodies.
- retained a two-block fixed-K down schedule and pointer-increment gate/up traversal.
- added atomics-free device row-task descriptors for large gate/up groups and reused them across paired projections.
- retained serial row ownership for down after descriptors and a complete decoded-weight LDS cache regressed.
- added contiguous bounded down-tail activation loads.
- validated all 60 production benchmark points exactly against dense MMQ.

Final outcome:
- packed MMQ wins 54 of 60 individual points against BF16 AITER.
- every gate/up point and every Q4_K/Q5_K down point wins.
- checkpoint-weighted packed grouped projections are 1.53-2.82x faster than AITER across the measured batch/distribution matrix.
- the only remaining individual losses are nonuniform IQ2_S down at batch 1 and batch 4.

Remaining work is representation-level: compact lossless IQ2_S decode caching, cross-call decoded-weight reuse, or a transient project-owned decoded dense stage. The local tile, scheduler, K-loop, bounds, cache, and synchronization neighborhoods are closed.

## Production contract

The public activation and output dtype is BF16.

Activations are quantized internally to Q8_1. Packed GGUF weights remain the authoritative representation.

`expert_indices` and `expert_offsets` remain device-resident. The optimized path must not inspect group sizes through `.item()`, a device-to-host copy, a CPU descriptor, or an implicit synchronization.

The metadata ABI is:
- `expert_indices`: contiguous CUDA `torch.int64`, shape `[G]`.
- `expert_offsets`: contiguous CUDA `torch.int32`, shape `[G]`.
- `expert_offsets[-1] = R`.
- `G <= 256`.

The sequence length is 2,048 and top-k is 8. The routed row count is therefore:

| Physical batch | Routed rows |
|---:|---:|
| 1 | 16,384 |
| 4 | 65,536 |
| 16 | 262,144 |

Batch 1 commonly has about 150-256 active experts per layer. The observed mean was about 198.5 active experts.

One representative batch-1 layer had 192 active experts and group sizes around 60-106 rows. Batch 4 and batch 16 usually activate all 256 experts, although the sizes remain skewed.

## Production checkpoint matrix

The benchmark uses real tensors from `Qwen3.6-35B-A3B-APEX-I-Mini.gguf`.

| Case | Logical expert shape | GGUF type | Layers | Operator |
|---|---:|---|---:|---|
| Gate/up outer | `512 x 2048` | Q3_K | 20 | `grouped_mmq_pair` |
| Gate/up middle | `512 x 2048` | IQ2_S | 20 | `grouped_mmq_pair` |
| Down outer edge | `2048 x 512` | Q5_K | 2 | `grouped_mmq` |
| Down outer main | `2048 x 512` | Q4_K | 18 | `grouped_mmq` |
| Down middle | `2048 x 512` | IQ2_S | 20 | `grouped_mmq` |

Gate and up use one shared Q8_1 activation workspace. The two packed projections still execute as two grouped multiplication launches.

## Benchmark infrastructure

`bench/benchmark_grouped_mmq_fwd.py` follows the dense benchmark conventions.

It records:
- complete public packed-operator latency.
- logical throughput.
- incremental peak allocation and reservation growth.
- AITER configuration.
- routing metadata and distribution statistics.
- checkpoint-weighted forward and optimizer-step estimates.
- grouped-versus-dense-MMQ exactness.
- grouped-versus-dequantized-BF16 error.

The packed timing includes Q8_1 quantization and grouped multiplication.

The BF16 performance reference is AITER Triton `gmm`, using the project-owned `torch_ggml_ops.aiter_gmm_heuristics.gmm_config`. It is not `torch.matmul`.

AITER receives independently dequantized BF16 versions of the same logical GGUF experts. Dequantization and active-weight selection are setup costs and are not inside the timed GMM call.

The AITER reference uses the production transposed weight metadata expected by the grouped kernel. `work_stealing` remains disabled, matching the production call.

The full baseline command was:

```bash
python bench/benchmark_grouped_mmq_fwd.py \
  --warmup 2 \
  --repeats 5 \
  --correctness-rows 128 \
  --output /tmp/grouped_mmq_fwd_baseline_full.json
```

The historical pre-optimization artifact is:

```text
/tmp/grouped_mmq_fwd_baseline_full.json
```

The final retained command uses the same arguments and writes:

```text
/tmp/grouped_mmq_fwd_final_full_v2.json
```

GPU benchmarks and profiler runs were sequential.

## Routing distributions

The benchmark covers four deterministic distributions.

`uniform` activates all 256 experts with equal group sizes.

`skewed` activates all 256 experts with non-multiple group sizes centered around the production mean.

`sparse` uses sparse active-expert IDs. It uses 192, 224, and 240 active experts for batches 1, 4, and 16.

`boundary` includes sizes 1, 15, 16, 17, 63, 64, 65, 127, 128, and 129 before filling the remaining groups. It exercises edge handling without making it the only performance distribution.

Representative summaries are:

| Batch | Distribution | Active experts | Minimum | Maximum | Mean |
|---:|---|---:|---:|---:|---:|
| 1 | uniform | 256 | 64 | 64 | 64.0 |
| 1 | skewed | 256 | 42 | 86 | 64.0 |
| 1 | sparse | 192 | 64 | 106 | 85.3 |
| 1 | boundary | 256 | 1 | 129 | 64.0 |
| 4 | uniform | 256 | 256 | 256 | 256.0 |
| 4 | skewed | 256 | 192 | 320 | 256.0 |
| 4 | sparse | 224 | 221 | 367 | 292.6 |
| 16 | uniform | 256 | 1,024 | 1,024 | 1,024.0 |
| 16 | skewed | 256 | 768 | 1,280 | 1,024.0 |
| 16 | sparse | 240 | 821 | 1,367 | 1,092.3 |

## Historical pre-G1 baseline results

The speedup column is `AITER time / packed MMQ time`. Values above 1.0 mean the packed path is faster.

### Uniform routing

| Case | Batch | Packed ms | AITER ms | Packed logical TFLOP/s | Speedup |
|---|---:|---:|---:|---:|---:|
| Gate/up Q3_K | 1 | 11.185 | 13.245 | 6.14 | 1.18x |
| Gate/up Q3_K | 4 | 32.572 | 33.679 | 8.44 | 1.03x |
| Gate/up Q3_K | 16 | 130.408 | 83.293 | 8.43 | 0.64x |
| Gate/up IQ2_S | 1 | 11.453 | 13.212 | 6.00 | 1.15x |
| Gate/up IQ2_S | 4 | 33.408 | 35.066 | 8.23 | 1.05x |
| Gate/up IQ2_S | 16 | 133.528 | 87.740 | 8.23 | 0.66x |
| Down IQ2_S | 1 | 6.326 | 3.578 | 5.43 | 0.57x |
| Down IQ2_S | 4 | 18.226 | 7.707 | 7.54 | 0.42x |
| Down IQ2_S | 16 | 69.996 | 44.702 | 7.85 | 0.64x |
| Down Q4_K | 1 | 6.874 | 3.524 | 5.00 | 0.51x |
| Down Q4_K | 4 | 21.977 | 7.730 | 6.25 | 0.35x |
| Down Q4_K | 16 | 83.663 | 44.266 | 6.57 | 0.53x |
| Down Q5_K | 1 | 6.953 | 3.575 | 4.94 | 0.51x |
| Down Q5_K | 4 | 21.990 | 7.777 | 6.25 | 0.35x |
| Down Q5_K | 16 | 84.674 | 43.794 | 6.49 | 0.52x |

### Sparse observed-style routing

| Case | Batch | Packed ms | AITER ms | Packed logical TFLOP/s | Speedup |
|---|---:|---:|---:|---:|---:|
| Gate/up Q3_K | 1 | 9.678 | 14.412 | 7.10 | 1.49x |
| Gate/up Q3_K | 4 | 35.091 | 36.875 | 7.83 | 1.05x |
| Gate/up Q3_K | 16 | 133.880 | 100.466 | 8.21 | 0.75x |
| Gate/up IQ2_S | 1 | 9.953 | 14.225 | 6.90 | 1.43x |
| Gate/up IQ2_S | 4 | 35.966 | 36.829 | 7.64 | 1.02x |
| Gate/up IQ2_S | 16 | 136.851 | 96.521 | 8.03 | 0.71x |
| Down IQ2_S | 1 | 5.544 | 2.839 | 6.20 | 0.51x |
| Down IQ2_S | 4 | 19.089 | 8.909 | 7.20 | 0.47x |
| Down IQ2_S | 16 | 71.064 | 44.474 | 7.74 | 0.63x |
| Down Q4_K | 1 | 6.342 | 2.821 | 5.42 | 0.44x |
| Down Q4_K | 4 | 22.196 | 8.817 | 6.19 | 0.40x |
| Down Q4_K | 16 | 84.262 | 44.430 | 6.52 | 0.53x |
| Down Q5_K | 1 | 6.394 | 2.842 | 5.37 | 0.44x |
| Down Q5_K | 4 | 22.310 | 8.805 | 6.16 | 0.39x |
| Down Q5_K | 16 | 84.894 | 43.764 | 6.48 | 0.52x |

At the historical baseline, gate/up was strong only for batch 1, approximately tied at batch 4, and behind at batch 16. Down was the dominant deficit for every batch. G1-G11 resolve those deficits except for six nonuniform IQ2_S down points documented in the final evaluation.

### Routing sensitivity

Across all four distributions, the average per-case speedups were:

| Case | Batch 1 | Batch 4 | Batch 16 |
|---|---:|---:|---:|
| Gate/up Q3_K | 1.35x | 1.04x | 0.71x |
| Gate/up IQ2_S | 1.31x | 1.03x | 0.69x |
| Down IQ2_S | 0.54x | 0.45x | 0.62x |
| Down Q4_K | 0.49x | 0.39x | 0.52x |
| Down Q5_K | 0.49x | 0.39x | 0.51x |

Packed gate/up benefits from sparse batch-1 routing because inactive experts do not launch grouped workgroups. AITER's fixed persistent grid still scans the active group list.

The historical pre-G1 kernel was mildly sensitive to skew at batch 4. Each `(expert, output tile)` workgroup serially processed every 128-row chunk for its expert, so large groups created longer-lived workgroups. G4 reduced the row tile to 64, and G8 exposed large gate/up row tiles as independent device tasks.

### Checkpoint-weighted grouped base-projection estimate

The estimate applies the checkpoint layer counts and two executions per optimizer step under activation checkpointing. It covers grouped base projections only, not routing, activation functions, LoRA, or other model work.

| Batch | Distribution | Packed seconds | AITER seconds | Packed speedup |
|---:|---|---:|---:|---:|
| 1 | uniform | 1.434 | 1.343 | 0.94x |
| 1 | skewed | 1.430 | 1.500 | 1.05x |
| 1 | sparse | 1.261 | 1.372 | 1.09x |
| 1 | boundary | 1.439 | 1.501 | 1.04x |
| 4 | uniform | 4.247 | 3.367 | 0.79x |
| 4 | skewed | 4.546 | 3.754 | 0.83x |
| 4 | sparse | 4.494 | 3.657 | 0.81x |
| 4 | boundary | 4.626 | 3.643 | 0.79x |
| 16 | uniform | 16.708 | 10.398 | 0.62x |
| 16 | skewed | 17.022 | 11.405 | 0.67x |
| 16 | sparse | 17.045 | 11.433 | 0.67x |
| 16 | boundary | 17.015 | 10.861 | 0.64x |

This table is the historical checkpoint-weighted baseline that motivated the implementation pass. The final checkpoint-weighted results are reported in `Final retained evaluation` and are faster than AITER for every batch and distribution.

AITER is the production BF16 reference. It is not a performance ceiling for a packed kernel with much smaller authoritative weights.

## Correctness

Every one of the 60 measured production points matched the concatenation of per-group dense MMQ outputs exactly in BF16.

This checks the grouped scheduler against the same Q8_1 activation semantics and the same packed GGUF decode path.

Against independently dequantized BF16 weights and BF16 AITER GMM, normalized RMSE ranges were:

| Type | Minimum NRMSE | Maximum NRMSE |
|---|---:|---:|
| Q3_K | 0.00599 | 0.00613 |
| IQ2_S | 0.00600 | 0.00611 |
| Q4_K | 0.01120 | 0.01381 |
| Q5_K | 0.01300 | 0.01650 |

These errors include the intended internal activation quantization.

## Incremental memory

The pair path uses one Q8_1 workspace for both outputs.

| Operator shape | Batch | Packed peak | AITER peak | Q8_1 workspace | Output bytes |
|---|---:|---:|---:|---:|---:|
| Pair `R x 2048 -> 2 x R x 512` | 1 | 68 MiB | 32 MiB | 36 MiB | 32 MiB |
| Pair `R x 2048 -> 2 x R x 512` | 4 | 272 MiB | 128 MiB | 144 MiB | 128 MiB |
| Pair `R x 2048 -> 2 x R x 512` | 16 | 1,088 MiB | 512 MiB | 576 MiB | 512 MiB |
| Down `R x 512 -> R x 2048` | 1 | 73 MiB | 64 MiB | 9 MiB | 64 MiB |
| Down `R x 512 -> R x 2048` | 4 | 292 MiB | 256 MiB | 36 MiB | 256 MiB |
| Down `R x 512 -> R x 2048` | 16 | 1,168 MiB | 1,024 MiB | 144 MiB | 1,024 MiB |

The measurements exclude resident packed or BF16 reference weights.

The pair workspace saving is material: two independent packed calls would need two quantization launches and two workspace lifetimes.

## Final packed kernel structure

The retained production arithmetic geometry is:

```text
I = 64 output columns
J = 64 routed rows
threads = 128, four wave32 waves
K iteration = 256 packed values
```

Both production families are compile-time specialized:
- gate/up: `NRowsWeight=512`, `BlocksPerWeightRow=8`.
- down: `NRowsWeight=2048`, `BlocksPerWeightRow=2`.

Common behavior:
- exact output tiles have no output-row fallback.
- full row tiles use contiguous Q8_1 loads and unmasked BF16 stores.
- only the final partial row tile uses bounded zero fill and masked stores.
- packed weights and Q8_1 activations are staged in LDS.
- public inputs and outputs remain BF16 and packed GGUF weights remain authoritative.

Gate/up scheduling is shape- and row-count-specific:
- batch-1-sized groups retain one `(expert, output tile)` workgroup with a serial row loop, preserving sparse launch behavior.
- when the host-visible average reaches at least two 64-row tiles, one GPU setup workgroup builds atomics-free `(expert, row_start, row_end)` task descriptors.
- `grouped_mmq_pair` builds descriptors once and reuses them for gate and up.

Down retains serial expert row ownership because it already launches 32 output tiles per active expert. Its two fixed K blocks are emitted as two explicit calls, and partial activation tiles use one contiguous integer-span predicate.

Dynamic LDS is 30,976 bytes for Q3_K/IQ2_S and 28,928 bytes for Q4_K/Q5_K. A general `J=128` fallback remains only for tests and non-production shapes.

## Historical pre-G1 packed-kernel profiling

The following traces describe the original spill-heavy baseline and are retained to explain why G1 was prioritized. Kernel-trace artifacts are:

```text
/tmp/rocprof_grouped_gate_b1
/tmp/rocprof_grouped_gate_b16
/tmp/rocprof_grouped_down_q4_b4
```

### Gate/up Q3_K, batch 1, uniform

The profiled pair decomposed into:
- Q8_1 quantization: 0.829 ms.
- first grouped projection: 5.408 ms.
- second grouped projection: 5.359 ms.

The grouped projections account for about 93% of the packed operator time.

### Gate/up Q3_K, batch 16, uniform

The profiled pair decomposed into:
- Q8_1 quantization: 7.894 ms.
- two grouped projections: 122.445 ms total.

The grouped projections account for about 94% of the packed operator time.

### Down Q4_K, batch 4, uniform

The profiled single projection decomposed into:
- Q8_1 quantization: 0.916 ms.
- grouped projection: 21.964 ms.

The multiplication kernel is the first-order bottleneck. More quantizer work is not justified before fixing it.

## Historical pre-G1 packed code-object resources

The pre-G1 extension code object was extracted from `.hip_fatbin` and inspected with `clang-offload-bundler`, `llvm-readobj`, `llvm-nm`, and `llvm-objdump`. These resources are historical. Final retained resources are reported later.

| Grouped kernel | VGPRs | SGPRs | Private bytes/thread | VGPR spills |
|---|---:|---:|---:|---:|
| Q3_K | 256 | 74 | 124 | 30 |
| Q4_K | 256 | 76 | 512 | 127 |
| Q5_K | 256 | 74 | 520 | 129 |
| Q6_K | 256 | 74 | 272 | 67 |
| IQ2_S | 255 | 78 | 132 | 32 |

The corresponding dense J=128 kernels have zero private segment and zero spills:

| Dense kernel | VGPRs | SGPRs | Private bytes/thread | VGPR spills |
|---|---:|---:|---:|---:|
| Q3_K | 216 | 29 | 0 | 0 |
| Q4_K | 254 | 30 | 0 | 0 |
| Q5_K | 230 | 29 | 0 | 0 |
| IQ2_S | 208 | 34 | 0 | 0 |

The packed arithmetic body is not inherently forced to spill. The grouped expert metadata and dynamic row-chunk loop push the inherited dense body over the register limit.

Static disassembly reinforces this conclusion. The grouped Q4_K function has 121 scratch-load and 76 scratch-store instruction sites, compared with 7 and 6 in the profiled AITER down kernel.

Q4_K and Q5_K spill roughly half a kilobyte per thread. This is the clearest explanation for the severe down-projection gap.

## AITER source, lowering, and historical pre-G1 comparison profile

The inspected AITER source is:

```text
~/venv_torch/lib/python3.14/site-packages/aiter/ops/triton/gmm.py
~/venv_torch/lib/python3.14/site-packages/aiter/ops/triton/_triton_kernels/gmm.py
```

AITER uses a 256-program persistent grid.

Each program starts from its program ID and advances through logical GMM tiles by `GRID_DIM`. It walks the device-resident group-size array and never requires a host-side group descriptor.

The tile mapping uses XCD remapping. Edge tiles wrap input row and output-column load offsets with modulo arithmetic, then mask only the final store.

The K loop uses direct BF16 loads and `tl.dot`. Generated gfx1151 ISA contains `v_wmma_f32_16x16x16_bf16` instructions.

The production heuristic selects:

| Shape | M tile | N tile | K tile | Threads | Persistent programs |
|---|---:|---:|---:|---:|---:|
| Gate/up `K=2048, N=512` | 64 | 128 | 64 | 256 | 256 |
| Down `K=512, N=2048` | 128 | 128 | 64 | 256 | 256 |

The generated gate/up kernel uses 176 VGPRs, 58 SGPRs, no private segment, and no fixed LDS.

The generated down kernel uses 255 VGPRs, 57 SGPRs, and 48 private bytes per thread. It is near the register limit but its spill footprint is much smaller than packed Q4_K/Q5_K.

The gate/up AITER profile launched 256 workgroups of 256 threads. One profiled projection took 7.849 ms under kernel tracing.

The down AITER profile used the same persistent grid and took 8.156 ms under kernel tracing for the batch-4 uniform case.

Profiler time is perturbed and is used only for decomposition and resource comparison. CUDA-event medians remain the latency source of record.

Counter runs on down Q4_K batch 4 reported:

| Kernel | Occupancy | L2 hit rate |
|---|---:|---:|
| Packed grouped Q4_K | 18.44% | 69.73% |
| BF16 AITER GMM | 23.20% | 71.68% |

Packed `ALUStalledByLDS` was only 0.0154%. LDS bank stalls are therefore not the primary problem.

The similar L2 hit rates argued against treating cache hit rate as the first optimization target in the pre-G1 kernel. Register spills, tile shape, and the amount of scheduled work were stronger baseline explanations.

`MemUnitBusy` was unavailable through dispatch-windowed counter collection on this gfx1151 profiler configuration and is not treated as a result.

## TensileLite, CK, and dense-MMQ mechanism study

This study predates G1-G11 and is retained for mechanism and design provenance. The outcomes are stated explicitly so that the original hypotheses are not mistaken for remaining tasks.

### TensileLite evidence

Principal sources:

```text
~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Tests/common/groupedgemm/grouped_gemm.yaml
~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Tests/common/groupedgemm/gfx11/grouped_gemm_gfx11.yaml
~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Tests/common/groupedgemm/gfx11/grouped_gemm_userargs_gfx11.yaml
~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Solution.py
~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/KernelWriterAssembly.py
~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Components/SIA.py
~/rocm-libraries/projects/hipblaslt/tensilelite/client/src/ClientProblemFactory.cpp
~/rocm-libraries/projects/hipblaslt/tensilelite/client/src/SolutionIterator.cpp
~/rocm-libraries/projects/hipblaslt/tensilelite/src/ContractionSolution.cpp
```

TensileLite can tune a complete grouped BF16 workload, but every GEMM in one group uses one static solution. That model matches separate fixed-shape gate/up and down families, but it cannot directly express Q8_1 activation handling, packed GGUF decode, scale application, device routing, and direct BF16 output.

The grouped client combines exact GEMMs into one `ContractionProblemGroupedGemm`, checks a candidate against every member, and times the complete enqueue. Its displayed grouped GFLOP/s is not a valid aggregate throughput because `BenchmarkTimer.cpp` uses only `problem->gemms[0].flopCount()` as the numerator. Historical grouped client results therefore remain capability evidence, not a performance bound.

TensileLite's runtime helper normally constructs grouped user-argument records on the host and copies them to the device. That setup is incompatible with the production routing ABI. The useful transferable idea was compact cumulative device metadata, which G8 implemented with one GPU prefix setup and direct row-task indexing.

| Mechanism | Durable lesson | Final outcome |
|---|---|---|
| Static macro-tile and matrix-instruction selection | Measure compile-time `I`, `J`, and thread-count families | `I=64`, `J=64`, 128 threads retained. G3/G9 rejected wider tiles |
| Fixed `DepthU` and assertions | Specialize eight-block gate/up and two-block down traversal | Retained in G1, G6, and G7 |
| SGPR/immediate global-read offsets | Prefer affine pointer increments and scalar fixed offsets on exact tiles | Gate/up pointer increments retained in G7 |
| Algorithm-3 issue scheduling | Split read/decode/commit lifetimes before adding pacing | Deeper staged prefetch closed because final kernels are already near the VGPR limit |
| Wave-separated reads | Keep weight ownership wave-local where geometry divides | Useful layout principle. No separate retained launch variant |
| `DirectToVgpr` | At most one immediately consumed decoded-weight fragment could be plausible | Wholesale and both-operands paths rejected. Not pursued after resource closure |
| One/two LDS buffers | Extra staging needs a measured residency argument | Complete down decoded cache lost 27-40%. Second-stage work remains closed |
| Store remapping | C-shuffle can trade LDS and synchronization for coalesced output | Not retained. Direct BF16 writeback is not the final bottleneck |
| GSU, split-K, Stream-K | Duplicates packed decode or requires reduction | Rejected for fixed K=512/2048. Grouped Stream-K is unsupported |

`UseSgprForGRO` remains a useful code-generation lesson: one per-lane VGPR base plus scalar offsets can reduce address state for affine exact tiles. Shift-pointer edge handling is incompatible with this form, which reinforces separate full and tail bodies. Explicit `readfirstlane` is justified only when final ISA proves that a wave-uniform value remained in VGPRs.

Algorithm-3 scheduling also established an ordering rule that remains useful for any future representation:
- read a bounded packed fragment.
- decode with temporary metadata and bit-extraction state.
- commit decoded BF16 values to LDS or form one immediately consumed register operand.
- end decode-temporary lifetimes.
- issue LDS reads and WMMA while only bounded next-fragment state is live.

Adding barriers to an unchanged monolithic decoder does not reproduce this dataflow. G5 confirmed that removing one existing write barrier is neutral, while the final VGPR allocations make a deeper live prefetch window unattractive.

TensileLite direct-to-VGPR assumes ordinary layout conversion. GGUF formats require scale reconstruction, metadata interpretation, arbitrary bit extraction, and BF16 formation. Direct global-to-LDS likewise cannot perform GGUF decode. Activations must remain in LDS because all four waves reuse them.

### Composable Kernel evidence

Relevant sources:

```text
~/rocm-libraries/projects/composablekernel/example/15_grouped_gemm/grouped_gemm_wmma_fixed_nk_fp16.cpp
~/rocm-libraries/projects/composablekernel/example/15_grouped_gemm/grouped_gemm_wmma_splitk_bf16.cpp
~/rocm-libraries/projects/composablekernel/example/ck_tile/17_grouped_gemm/grouped_gemm.cpp
~/rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_fixed_nk_common.hpp
~/rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_wmma_fixed_nk.hpp
~/rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_wmma_cshuffle_tile_loop_v3.hpp
~/rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v1.hpp
~/rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp
~/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp
```

The useful CK lessons were structural rather than directly reusable kernels:
- fixed-NK specialization removes dynamic N/K scheduler state.
- load, decode, LDS commit, LDS read, and WMMA phases need bounded lifetimes.
- tile ordering should preserve locality without adding excessive persistent control state.
- C-shuffle and extra pipeline stages are worthwhile only with a resource argument.
- direct global-to-LDS is not a gfx1151 CK mechanism.

The fixed-NK hypothesis produced the largest retained change. Production shapes are exact and narrow:

| Projection | Output rows per expert | Input features | GGUF blocks per weight row |
|---|---:|---:|---:|
| Gate/up | 512 | 2,048 | 8 |
| Down | 2,048 | 512 | 2 |

G1 specialized these dimensions at compile time and removed output-tile edge predicates. G6 retained an explicit two-block down schedule, while G7 retained a rolled eight-block gate/up loop with pointer increments.

The complete decoded-down-weight cache was also a direct CK-inspired hypothesis. Its estimated dynamic LDS was 48,384 bytes for Q4_K/Q5_K and 52,480 bytes for IQ2_S at `J=64`. G2 measured 27-40% regressions because the additional 19-22 KiB LDS cost outweighed cross-row decode reuse. This design must not be retried without a substantially more compact cached representation.

The device scheduling study led to G8. With `G <= 256`, one 256-thread workgroup computes task counts, performs an atomics-free prefix sum, and writes device-resident `(expert, row_start, row_end)` arrays with capacity:

```text
ceil(R / 64) + G
```

Output tile is the fastest-changing launch dimension. Large gate/up groups gain approximately 1-2%, and paired projections reuse the setup. Down descriptors lost 2-14% because 32 output tiles per expert already expose enough parallelism. Persistent grid-stride traversal was not pursued because the nonpersistent gain was small and extra control state would threaten sparse behavior and register headroom.

G4 retained separate full and tail bodies in one public launch. Full rows have no activation bounds handling and use unmasked BF16 stores. G11 further reduced down-tail address work to one contiguous integer-span predicate. A second full-tile launch, clamped tail arithmetic, and C-shuffle do not have enough remaining headroom to justify more local work.

### Dense-MMQ lessons carried into grouped MMQ

- Inspect the final code object after every structural change. Source-level register intuition was insufficient.
- Use compile-time typed variants and measured dispatch because decode and resource behavior differ by quant type.
- Optimize complete public-operator latency. Final multiplication still accounts for approximately 87-91% of retained kernel time.
- Larger cooperative tiles help only when reuse exceeds their LDS, accumulator, and workgroup-residency costs. G3 and G9 show that this condition does not hold here.
- Type-specific packed extraction can win, but the final grouped limit is IQ2_S representation cost rather than a universal decoder schedule.
- gfx1151 int8 WMMA has no decisive raw throughput advantage over BF16 WMMA. Packed MMQ must win through authoritative-weight compression, reuse, scheduling, and lower traffic.

## Historical baseline diagnosis and final resolution

### Register spilling was the first blocker

The original grouped Q4_K/Q5_K kernels reached 256 VGPRs and spilled 127-129 VGPRs per thread. G1's compile-time `J=64` and fixed production shapes removed every production private segment and spill. Final kernels remain spill-free.

### Larger output or row tiles did not provide the next gain

The original packed tile was narrower than AITER, which made `I=128` and `J=128` reasonable hypotheses. G3 and G9 tested those neighborhoods and both regressed despite reduced or eliminated spilling. The final `I=64, J=64` geometry is retained.

### Scheduling must remain shape-specific

The original serial expert loop limited gate/up balance at larger batches. G8's atomics-free row-task descriptors provide a small but repeatable 1-2% gain for large gate/up groups and are reused by paired projections.

Down already exposes 32 output tiles per expert. Device row descriptors regressed it by 2-14%, so down retains serial row ownership. A complete decoded-weight LDS cache also regressed by 27-40% because its extra 19-22 KiB LDS cost outweighed reuse.

### Quantization and pair fusion were not the main limit

Gate/up already shares one Q8_1 workspace. Final profiling still assigns approximately 87-91% of operator kernel time to packed multiplication. A fused two-weight arithmetic kernel would require a second live accumulator set and is not justified by the remaining headroom.

### Final bottleneck

The remaining six losses are nonuniform IQ2_S down points at batch 1 and batch 4. They are caused by IQ2_S decode and zero-padded partial `J=64` work repeated across 32 output tiles, not by spills, descriptor setup, or generic metadata traversal. Further worthwhile work requires a changed representation or cross-call decode reuse.

## Optimization experiment log

### G1: compile-time `J=64` and fixed production shapes

Status: retained.

The first implementation combined the two highest-priority spill-removal changes:
- compile-time `J=64` for both production shape families.
- fixed gate/up `NRowsWeight=512, BlocksPerWeightRow=8` and down `NRowsWeight=2048, BlocksPerWeightRow=2` variants.
- no output-row fallback in the fixed variants because every production output tile is complete.
- a fixed-trip K loop while retaining the general `J=128` fallback for tests and non-production shapes.

The focused timing artifact is:

```text
/tmp/grouped_step1_j64_fixed.json
```

| Point | Baseline ms | G1 ms | Speedup |
|---|---:|---:|---:|
| Gate/up Q3_K batch 1 uniform | 11.185 | 6.170 | 1.81x |
| Gate/up Q3_K batch 1 sparse | 9.678 | 7.166 | 1.35x |
| Gate/up Q3_K batch 4 boundary | 35.929 | 24.609 | 1.46x |
| Gate/up Q3_K batch 16 uniform | 130.408 | 94.200 | 1.38x |
| Gate/up IQ2_S batch 1 sparse | 9.953 | 7.566 | 1.32x |
| Down Q4_K batch 1 uniform | 6.874 | 2.842 | 2.42x |
| Down Q4_K batch 4 uniform | 21.977 | 10.599 | 2.07x |
| Down Q4_K batch 4 boundary | 22.934 | 11.418 | 2.01x |
| Down Q4_K batch 16 uniform | 83.663 | 42.110 | 1.99x |
| Down Q5_K batch 4 uniform | 21.990 | 10.688 | 2.06x |
| Down IQ2_S batch 16 uniform | 69.996 | 45.963 | 1.52x |

The fixed production kernels are spill-free:

| Production kernel | VGPRs | SGPRs | Private bytes/thread | Dynamic LDS |
|---|---:|---:|---:|---:|
| Gate/up Q3_K | 164 | 46 | 0 | 30,976 bytes |
| Gate/up IQ2_S | 190 | 47 | 0 | 30,976 bytes |
| Down IQ2_S | 190 | 49 | 0 | 30,976 bytes |
| Down Q4_K | 168 | 49 | 0 | 28,928 bytes |
| Down Q5_K | 181 | 48 | 0 | 28,928 bytes |

This removes the original 124-520 byte private segments and 30-129 VGPR spills without introducing dynamic stack use. The retained kernels also reduce dynamic LDS by 9,472 bytes relative to the original `J=128` grouped path.

Production-shape correctness was checked against dense MMQ:
- gate/up Q3_K batch 1 sparse: both outputs had zero differing BF16 elements.
- down Q4_K batch 4 uniform: zero differing BF16 elements.

The complete public gate/up point measured 7.155 ms versus 14.155 ms AITER, or 1.98x faster. The complete public down Q4_K batch-4 point measured 10.617 ms versus 7.805 ms AITER, or 0.74x. Spill removal therefore explains and resolves most of the original deficit, but batch-4 down and batch-16 gate/up remain the primary performance gaps.

### G1 decision

Keep the combined specialization. Separating fixed shape from `J=64` is not necessary for acceptance because the combined variant is spill-free and materially faster at every focused production point. Future variants must compare against G1 rather than the original baseline.

### G2: complete decoded-weight LDS cache for down

Status: rejected and reverted.

G2 decoded both fixed `K=512` weight blocks into immutable LDS before the serial row loop and reused them across all `J=64` row chunks. The cache was dispatched only when the host-visible average was at least two row tiles, preserving the G1 batch-1 path.

The artifact is:

```text
/tmp/grouped_step2_down_cache.json
```

| Point | G1 ms | G2 ms | Relative |
|---|---:|---:|---:|
| Down Q4_K batch 4 uniform | 10.599 | 15.427 | 0.69x |
| Down Q4_K batch 4 boundary | 11.418 | 15.705 | 0.73x |
| Down Q4_K batch 16 uniform | 42.110 | 70.334 | 0.60x |
| Down Q5_K batch 4 uniform | 10.688 | 15.503 | 0.69x |
| Down IQ2_S batch 16 uniform | 45.963 | 72.384 | 0.64x |

The additional immutable weight tile increased dynamic LDS from 28,928 to 48,384 bytes for Q4_K/Q5_K and from 30,976 to 52,480 bytes for IQ2_S. The saved packed loads and decode did not compensate for the resulting resource and residency loss. This also shows that the original down deficit is no longer caused primarily by repeated decode once G1 has removed scratch traffic.

Do not retry the complete cache with the same 64-row output tile. Any future cross-row weight reuse must either use a more compact decoded representation or change the output/workgroup geometry enough to recover residency.

### G3: fixed-shape `J=128`

Status: rejected and reverted.

G3 kept fixed production N/K specialization but restored `J=128` to determine whether halving the serial row-loop count could beat G1 after dynamic shape state was removed.

The artifact is:

```text
/tmp/grouped_step3_j128_fixed.json
```

Every focused point regressed by 25-50% relative to G1. Representative results were:

| Point | G1 `J=64` ms | G3 `J=128` ms | Relative |
|---|---:|---:|---:|
| Gate/up Q3_K batch 1 uniform | 6.170 | 11.167 | 0.55x |
| Gate/up Q3_K batch 16 uniform | 94.200 | 129.555 | 0.73x |
| Down Q4_K batch 4 uniform | 10.599 | 16.764 | 0.63x |
| Down Q4_K batch 16 uniform | 42.110 | 65.776 | 0.64x |
| Down Q5_K batch 4 uniform | 10.688 | 17.727 | 0.60x |
| Down IQ2_S batch 16 uniform | 45.963 | 66.939 | 0.69x |

Fixed specialization reduced but did not eliminate `J=128` pressure for the main down types: Q4_K used 256 VGPRs and 184 private bytes/thread, while Q5_K used 256 VGPRs and 232 private bytes/thread. Q3_K retained 12 private bytes/thread. IQ2_S was spill-free at 241 VGPRs but still lost heavily, showing that the larger accumulator and LDS footprint is itself unfavorable even without scratch traffic.

Keep `J=64` for all production grouped-forward types. The fixed `I=64, J=128` neighborhood is closed.

### G4: separate exact full-row and bounded tail bodies

Status: retained.

G4 split the serial expert loop into compile-time full-row and tail helpers. A full `J=64` row tile now:
- loads the Q8_1 activation region as one contiguous integer span with no row division, remainder, or predicate.
- writes a complete BF16 row tile with no row predicate.
- retains the bounded zero-fill and masked-store path only for the final partial tile.

The focused artifact is:

```text
/tmp/grouped_step4_full_rows.json
```

| Point | G1 ms | G4 ms | Speedup |
|---|---:|---:|---:|
| Gate/up Q3_K batch 1 uniform | 6.170 | 3.735 | 1.65x |
| Gate/up Q3_K batch 1 sparse | 7.166 | 5.427 | 1.32x |
| Gate/up Q3_K batch 4 boundary | 24.609 | 16.755 | 1.47x |
| Gate/up Q3_K batch 16 uniform | 94.200 | 57.956 | 1.63x |
| Gate/up IQ2_S batch 1 sparse | 7.566 | 6.065 | 1.25x |
| Down Q4_K batch 1 uniform | 2.842 | 1.599 | 1.78x |
| Down Q4_K batch 4 uniform | 10.599 | 6.063 | 1.75x |
| Down Q4_K batch 4 boundary | 11.418 | 6.889 | 1.66x |
| Down Q4_K batch 16 uniform | 42.110 | 24.292 | 1.73x |
| Down Q5_K batch 4 uniform | 10.688 | 6.067 | 1.76x |
| Down IQ2_S batch 16 uniform | 45.963 | 32.846 | 1.40x |

The production kernels remain spill-free. Q3_K uses 168 VGPRs, Q4_K uses 168, Q5_K uses 175 for down, and IQ2_S uses 225. The extra compile-time full/tail code raises some register counts but does not create a private segment.

Production correctness remained exact against dense MMQ for both paths:
- gate/up Q3_K batch 1 sparse: zero differing BF16 elements for both outputs.
- down Q4_K batch 4 boundary: zero differing BF16 elements.

The complete gate/up Q3_K batch-1 sparse point measured 5.368 ms versus 14.131 ms AITER, or 2.63x. The complete down Q4_K batch-4 boundary point measured 6.962 ms versus 9.286 ms AITER, or 1.33x. G4 moves every focused production class ahead of its baseline AITER reference.

The magnitude of this gain shows that integer row decomposition and per-load tail control, not only register spilling, were first-order costs in the inherited grouped loop.

### G5: remove the post-write row barrier

Status: rejected and reverted.

The barrier after BF16 writeback is not required for shared-memory correctness because the preceding post-dot barrier already ends the decoded-weight and activation LDS lifetime. Removing it, however, produced only noise-level changes from 0.975x to 1.004x across the focused matrix. The artifact is:

```text
/tmp/grouped_step5_no_write_barrier.json
```

Keep the barrier in the retained source. It is not a measurable bottleneck, and retaining the simpler row-iteration synchronization structure is preferable to a neutral change.

### G6: compile-time two-block down unroll

Status: retained for the fixed `K=512` down shape.

G6 factored one packed K-block operation into a force-inlined helper and emits two explicit calls for down's exact `BlocksPerWeightRow=2`. Gate/up and the general fallback retain the original fixed-trip loop, avoiding full unrolling of the eight-block gate/up path.

The retained artifact is:

```text
/tmp/grouped_step6b_down_unroll.json
```

| Point | G4 ms | G6 ms | Speedup |
|---|---:|---:|---:|
| Down Q4_K batch 1 uniform | 1.599 | 1.565 | 1.02x |
| Down Q4_K batch 4 uniform | 6.063 | 5.877 | 1.03x |
| Down Q4_K batch 4 boundary | 6.889 | 6.882 | 1.00x |
| Down Q4_K batch 16 uniform | 24.292 | 23.057 | 1.05x |
| Down Q5_K batch 4 uniform | 6.067 | 5.849 | 1.04x |
| Down IQ2_S batch 16 uniform | 32.846 | 27.102 | 1.21x |

The gate/up body remained neutral after restoring its original loop inline: Q3_K batch 16 changed from 57.956 to 58.107 ms and batch-1 sparse from 5.427 to 5.383 ms.

All down variants remain spill-free. The explicit two-block schedule raises VGPR allocation to 207 for Q3_K, 223 for Q4_K, 230 for Q5_K, 217 for Q6_K, and 218 for IQ2_S, with 46 SGPRs and zero private segment. The register increase is acceptable because the fixed-shape kernels remain below the spill threshold and the complete operator improves.

Production correctness remained exact against dense MMQ for down Q4_K batch-4 boundary and down IQ2_S batch-16 uniform, with zero differing BF16 elements. Complete timings were 6.931 versus 9.278 ms AITER for Q4_K and 28.042 versus 45.382 ms AITER for IQ2_S.

The large IQ2_S gain and smaller Q4_K/Q5_K gains show that fixed-trip branch/address cleanup remains useful after G4, but its value is type-dependent and bounded by register growth.

### G7: pointer-increment gate/up K traversal

Status: retained.

G7 keeps the eight-block gate/up loop rolled but replaces per-iteration weight-block and activation-plane multiplication with incremented packed-weight indices and Q8_1 pointers. The full-row path now loads directly from the current and next activation plane.

The main artifact is:

```text
/tmp/grouped_step7_gate_pointers.json
```

Relative to G4, gate/up Q3_K improved by 0.4-1.7% across the focused matrix:

| Point | G4 ms | G7 ms | Speedup |
|---|---:|---:|---:|
| Gate/up Q3_K batch 1 uniform | 3.735 | 3.713 | 1.01x |
| Gate/up Q3_K batch 1 sparse | 5.427 | 5.358 | 1.01x |
| Gate/up Q3_K batch 4 boundary | 16.755 | 16.480 | 1.02x |
| Gate/up Q3_K batch 16 uniform | 57.956 | 57.708 | 1.00x |

The batch-1 IQ2_S sparse point was neutral, but separate large-group A/B measurements showed the pointer path improving IQ2_S from 17.306 to 16.930 ms at batch 4 and from 67.928 to 66.909 ms at batch 16. Those artifacts are `/tmp/grouped_step6_iq2_large.json` and `/tmp/grouped_step7_iq2_large.json`.

Production gate/up kernels remain spill-free. The pointer state raises VGPR allocation to 177 for Q3_K/Q4_K, 188 for Q5_K, 190 for Q6_K, and 240 for IQ2_S while reducing Q3_K SGPR allocation from 49 to 44. The complete gains are small but consistent at the important Q3_K and large IQ2_S points, so the affine pointer form is retained.

### G8: atomics-free row-task descriptors for large gate/up groups

Status: retained for gate/up when the host-visible average is at least two `J=64` row tiles. Rejected for down.

G8 adds a one-workgroup GPU setup pass. Each of at most 256 threads computes one group's task count, participates in a shared-memory prefix sum, and writes compact `(expert, row_start, row_end)` records without atomics. The capacity remains bounded by:

```text
ceil(R / 64) + G
```

The task count and three descriptor arrays remain device-resident. The compute grid indexes descriptors directly with output tile as the fastest-changing launch dimension. `grouped_mmq_pair` builds descriptors once and reuses them for gate and up. Batch-1 keeps G7's sparse serial dispatch because the descriptor threshold is not met.

The focused artifacts are:

```text
/tmp/grouped_step8_gate_descriptors.json
/tmp/grouped_step8_gate_descriptors_15.json
/tmp/grouped_step7_gate_serial_15.json
/tmp/grouped_step8_iq2_descriptors.json
```

The fair 15-repeat sequential A/B comparison, including setup, descriptor allocation, excess bounded workgroups, quantization, and both gate/up projections, measured:

| Point | G7 serial ms | G8 descriptors ms | Speedup |
|---|---:|---:|---:|
| Gate/up Q3_K batch 4 boundary | 16.576 | 16.348 | 1.01x |
| Gate/up Q3_K batch 16 uniform | 57.660 | 56.329 | 1.02x |
| Gate/up IQ2_S batch 16 uniform | 66.966 | 66.314 | 1.01x |

The descriptor arithmetic kernel remains spill-free and uses fewer VGPRs than the serial G7 body: 173 for Q3_K and 213 for IQ2_S, with 54/50 SGPRs. The setup kernel uses 13 VGPRs, 23 SGPRs, 1 KiB LDS, and no private segment. Production gate/up Q3_K batch-4 boundary remained exactly equal to dense MMQ for both outputs.

A down-only descriptor dispatch was also tested and reverted. The artifact is:

```text
/tmp/grouped_step8b_down_descriptors.json
```

Down already launches 32 output tiles per active expert and saturates the GPU without row descriptors. Extra task parallelism regressed batch-4 Q4_K by 2.8%, boundary Q4_K by 4.4%, batch-16 Q4_K by 1.8%, batch-4 Q5_K by 1.9%, and batch-16 IQ2_S by 13.9%. Keep G6's serial row ownership for down.

### G8 correctness coverage

`tests/test_grouped_mmq.py::test_grouped_pair_production_row_tasks_match_dense` exercises the production `512 x 2048` paired descriptor path with both full and tail row tasks and requires bit-exact BF16 equality against concatenated dense MMQ calls.

### G9: `I=128, J=64`, 256 threads

Status: rejected and reverted.

G9 doubled the output tile and workgroup size globally for the focused production experiment. This kept 32 FP32 accumulators per thread and all selected production kernels remained spill-free, but dynamic LDS rose to approximately 50-55 KiB and workgroup-level scheduling became less favorable.

The artifact is:

```text
/tmp/grouped_step9_i128.json
```

Every focused point regressed relative to the retained `I=64` dispatch. Representative comparisons were:

| Point | Retained ms | G9 ms | Relative |
|---|---:|---:|---:|
| Gate/up Q3_K batch 1 uniform | 3.52-3.71 | 3.918 | 0.90-0.95x |
| Gate/up Q3_K batch 4 boundary | 16.348 | 17.068 | 0.96x |
| Gate/up Q3_K batch 16 uniform | 56.329 | 61.084 | 0.92x |
| Gate/up IQ2_S batch 16 uniform | 66.314 | 69.075 | 0.96x |
| Down Q4_K batch 4 uniform | 5.877 | 6.744 | 0.87x |
| Down Q4_K batch 16 uniform | 23.057 | 26.287 | 0.88x |
| Down Q5_K batch 4 uniform | 5.849 | 6.853 | 0.85x |
| Down IQ2_S batch 16 uniform | 27.102 | 29.160 | 0.93x |

The wider task kernel actually reduced Q3_K/IQ2_S VGPR allocation to 150/201, confirming that register pressure was not the cause. The regression comes from the larger LDS footprint, eight-wave workgroups, reduced workgroup residency/flexibility, and less favorable balance between packed-weight work and activation reuse. The `I=128, J=64` neighborhood is closed.

### G10: 256 threads with `I=64, J=64`

Status: invalid and reverted. Timings are non-results.

G10 changed only the workgroup from four to eight waves while retaining the 64-row output tile. The inherited MMQ wave mapping assigns output fragments by `threadIdx.y`, so the additional four waves mapped beyond the logical 64-row output tile and overlapped neighboring output work. The focused test appeared implausibly fast for that reason.

`tests/test_grouped_mmq.py::test_grouped_pair_production_row_tasks_match_dense` rejected the variant with 15,624 mismatched elements out of 262,144 and NaNs in the output. The artifact `/tmp/grouped_step10_256_threads.json` must not be used as performance evidence.

A correct eight-wave kernel would require a different decomposition such as split-K or duplicated output ownership and reduction. Those mechanisms add packed decode work or reduction overhead and are already outside the accepted design space. Keep 128 threads.

### G11: contiguous bounded activation-tail loads for down

Status: retained for the fixed two-block down body.

G11 observes that each valid partial Q8_1 row tile is still one contiguous integer span. The down helper now computes `(j_max + 1) * q8_block_ints` once and uses a single `l < valid_activation_ints` predicate, eliminating per-load row division, remainder, source-row reconstruction, and nested row bounds. Gate/up retains its G7/G8 tail code because applying the same source change there produced mixed noise-level results.

The retained 15-repeat artifact is:

```text
/tmp/grouped_step11b_down_contiguous_tails_15.json
```

| Point | Previous ms | G11 ms | Speedup |
|---|---:|---:|---:|
| Down Q4_K batch 1 sparse | 2.304 | 2.270 | 1.02x |
| Down Q4_K batch 4 boundary | 6.704 | 6.592 | 1.02x |
| Down Q4_K batch 16 uniform | 23.153 | 22.742 | 1.02x |
| Down Q5_K batch 4 boundary | 6.897 | 6.608 | 1.04x |
| Down IQ2_S batch 1 skewed | 3.836 | 3.659 | 1.05x |
| Down IQ2_S batch 1 sparse | 3.889 | 3.730 | 1.04x |
| Down IQ2_S batch 1 boundary | 3.988 | 3.764 | 1.06x |
| Down IQ2_S batch 4 skewed | 9.816 | 9.580 | 1.02x |
| Down IQ2_S batch 4 sparse | 9.525 | 9.336 | 1.02x |
| Down IQ2_S batch 4 boundary | 10.068 | 9.683 | 1.04x |
| Down IQ2_S batch 16 uniform | 26.832 | 26.478 | 1.01x |

The source cleanup increases down VGPR allocation to 229 for Q3_K, 244 for Q4_K, 248 for Q5_K, 239 for Q6_K, and 232 for IQ2_S, but every production kernel remains spill-free with zero private segment. The complete operator improves despite the higher allocation.

Production down IQ2_S batch-1 sparse and Q5_K batch-4 boundary remained bit-exact against dense MMQ. IQ2_S sparse improved to 3.745 ms in the correctness run but still trails 2.846 ms AITER, identifying the remaining final deficit.

## Final retained evaluation

The complete retained 60-point artifact after G11 is:

```text
/tmp/grouped_mmq_fwd_final_full_v2.json
```

All 60 points match concatenated dense MMQ exactly with zero differing BF16 elements. Normalized RMSE against independently dequantized BF16 AITER remains within the original expected ranges:
- Q3_K: 0.00599-0.00613.
- IQ2_S: 0.00600-0.00611.
- Q4_K: 0.01120-0.01381.
- Q5_K: 0.01300-0.01650.

The packed operator wins 54 of 60 individual points against AITER. All gate/up Q3_K and IQ2_S points win, all down Q4_K and Q5_K points win, and all batch-16 down IQ2_S points win. The only remaining losses are nonuniform down IQ2_S at batch 1 and batch 4.

Representative final complete-operator results are:

| Point | Packed ms | AITER ms | Packed/AITER speedup |
|---|---:|---:|---:|
| Gate/up Q3_K batch 1 sparse | 5.307 | 14.215 | 2.68x |
| Gate/up Q3_K batch 4 boundary | 16.197 | 37.887 | 2.34x |
| Gate/up Q3_K batch 16 uniform | 56.601 | 89.336 | 1.58x |
| Gate/up IQ2_S batch 16 uniform | 66.798 | 89.102 | 1.33x |
| Down Q4_K batch 4 uniform | 5.644 | 7.878 | 1.40x |
| Down Q4_K batch 16 uniform | 22.816 | 45.213 | 1.98x |
| Down Q5_K batch 4 uniform | 5.761 | 7.820 | 1.36x |
| Down IQ2_S batch 16 uniform | 26.299 | 44.365 | 1.69x |
| Down IQ2_S batch 1 sparse | 3.736 | 2.851 | 0.76x |
| Down IQ2_S batch 4 boundary | 9.570 | 9.271 | 0.97x |

### Final checkpoint-weighted estimates

The following estimates multiply each complete public-operator latency by the checkpoint's projection-call count and by two for checkpointed forward recomputation during an optimizer step:

| Batch | Distribution | Original packed ms | Final packed ms | Final AITER ms | Final versus original | Final versus AITER |
|---:|---|---:|---:|---:|---:|---:|
| 1 | uniform | 1,433.8 | 473.5 | 1,336.4 | 3.03x | 2.82x |
| 1 | skewed | 1,429.5 | 739.1 | 1,510.4 | 1.93x | 2.04x |
| 1 | sparse | 1,260.9 | 700.0 | 1,370.0 | 1.80x | 1.96x |
| 1 | boundary | 1,439.5 | 733.1 | 1,510.8 | 1.96x | 2.06x |
| 4 | uniform | 4,247.4 | 1,750.0 | 3,384.0 | 2.43x | 1.93x |
| 4 | skewed | 4,546.0 | 2,051.5 | 3,792.7 | 2.22x | 1.85x |
| 4 | sparse | 4,494.2 | 2,012.8 | 3,649.8 | 2.23x | 1.81x |
| 4 | boundary | 4,625.6 | 2,053.5 | 3,761.4 | 2.25x | 1.83x |
| 16 | uniform | 16,707.8 | 6,899.6 | 10,725.2 | 2.42x | 1.55x |
| 16 | skewed | 17,021.7 | 7,067.5 | 11,370.6 | 2.41x | 1.61x |
| 16 | sparse | 17,044.8 | 7,051.0 | 11,084.8 | 2.42x | 1.57x |
| 16 | boundary | 17,015.1 | 7,086.6 | 10,818.2 | 2.40x | 1.53x |

### Final code-object resources

All retained production arithmetic kernels have zero private segment, zero VGPR spills, zero SGPR spills, and no dynamic stack:
- large-group gate/up row-task Q3_K/IQ2_S: 173/213 VGPRs and 54/50 SGPRs.
- batch-1 serial gate/up Q3_K/IQ2_S: 177/240 VGPRs and 44/56 SGPRs.
- down Q4_K/Q5_K/IQ2_S: 244/248/232 VGPRs and 46 SGPRs.
- descriptor setup: 13 VGPRs, 23 SGPRs, and 1 KiB LDS.

Dynamic LDS remains 28,928 bytes for Q4_K/Q5_K and 30,976 bytes for Q3_K/IQ2_S at `I=64, J=64`.

## Final bottleneck

Final sequential kernel traces are under:

```text
/tmp/rocprof_grouped_final_gate_b16_csv
/tmp/rocprof_grouped_final_v2_down_q4_b4
/tmp/rocprof_grouped_final_v2_down_iq2_b4_skewed
```

Ignoring benchmark input-generation kernels, the packed multiplication remains dominant:
- gate/up Q3_K batch 16 uniform: 50.875 ms across two row-task projections, 7.703 ms quantization, and 0.009 ms descriptor setup. Multiplication is approximately 87% of operator kernel time.
- down Q4_K batch 4 uniform: 5.555 ms multiplication and 0.522 ms quantization. Multiplication is approximately 91%.
- down IQ2_S batch 4 skewed: 9.576 ms multiplication and 0.945 ms quantization. Multiplication is approximately 91%.

The remaining deficit is not register spilling, launch setup, LDS banking, or generic metadata traversal. It is the packed IQ2_S arithmetic representation on irregular down groups:
- IQ2_S packed metadata interpretation, grid lookup, scale formation, and BF16 decoded-weight construction remain inside every output-tile workgroup.
- Down launches 32 output tiles per active expert, so the same expert tail and packed decode structure is repeated many times.
- Nonuniform groups require one partial `J=64` tile per expert. G11 removed row division and source reconstruction, but the WMMA tile still computes zero-padded rows and the packed weights still must be decoded for that partial tile.
- BF16 AITER has no packed decode cost and therefore retains a narrow advantage on the six short/nonuniform IQ2_S down points.

The tested same-representation alternatives do not offer more headroom:
- a complete decoded-weight LDS cache loses 27-40%.
- down row descriptors lose 2-14%.
- `J=128` and `I=128` lose materially.
- the post-write barrier is neutral.
- eight-wave `I=64` ownership is invalid without split-K-style reduction.
- split-K, larger LDS, a second stage, and persistent control state conflict with measured regressions and resource constraints.

No further local tile, scheduler, bounds, or synchronization change is worthwhile for the current packed representation. A higher ceiling requires a representation-level change, most plausibly a compact lossless IQ2_S decoded cache that is substantially smaller than BF16, cross-call decoded-weight reuse, or a transient project-owned decoded dense stage. Those are separate architectural projects and should be judged against the now-strong complete-operator baseline rather than added to this kernel as more control state.

## Completed plan and remaining work

| Phase | Outcome |
|---|---|
| Compile-time row and fixed-shape specialization | Completed in G1. Removed all production spills |
| Down complete decoded-weight LDS cache | Rejected in G2. 27-40% slower |
| `J=128` and wider output tiles | Rejected in G3/G9 |
| Exact full-row and bounded tail paths | Retained in G4 and refined for down in G11 |
| Fixed two-block down and gate/up pointer traversal | Retained in G6/G7 |
| Device row-task descriptors | Retained for large gate/up in G8. Rejected for down |
| Barrier removal | Neutral and reverted in G5 |
| Eight-wave `I=64` workgroup | Invalid output ownership in G10 |
| End-to-end validation | Complete: 60/60 exact against dense MMQ, 39 project tests, 9 integration tests |

### What remains

No additional local tile, scheduler, bounds, prefetch-toggle, LDS-cache, or synchronization sweep is planned for the current representation.

The only material remaining per-point deficit is nonuniform IQ2_S down at batch 1 and batch 4. Higher-ceiling follow-up projects are:
- a compact lossless IQ2_S decoded cache substantially smaller than BF16.
- cross-call decoded-weight reuse.
- a transient project-owned decoded dense stage.

Any future implementation must preserve device-resident routing metadata, batch-1 sparse launch behavior, exact grouped-versus-dense BF16 output, and complete-operator timing. It must compare against `/tmp/grouped_mmq_fwd_final_full_v2.json`, not the historical baseline.

If grouped source changes resume, validation remains:

```bash
pytest -q tests/
python -m compileall -q bench
git diff --check
```

Then rerun the complete 60-point matrix and the local project test suite.

## Acceptance criteria

Status: satisfied for the retained production dispatch.

Every retained kernel must preserve exact grouped-versus-dense-MMQ BF16 output.

Production grouped arithmetic kernels require zero private segment and zero spills. The original Q4_K/Q5_K 512-520 bytes per thread was not acceptable. Every retained production kernel satisfies the zero-private-segment requirement.

Batch-1 sparse gate/up must retain a clear advantage over AITER.

Batch-4 down and batch-16 gate/up are the primary performance gates. A local improvement that regresses the model-weighted optimizer-step estimate should be rejected.

AITER remains the reference, not the ceiling. The final target is the best end-to-end packed grouped latency that preserves the production contract.

## Non-priorities and rejected directions

- Do not replace AITER GMM with `torch.matmul` as the performance reference.
- Do not link packed kernels against AITER or hipBLASLt.
- Do not construct full logical BF16 expert matrices in the packed production path.
- Do not introduce CPU expert descriptors or metadata synchronization.
- Do not prioritize another quantizer rewrite: production spills are already removed, and final multiplication still accounts for approximately 87-91% of retained kernel time.
- Do not use runtime environment variables or online autotuning for dispatch.
- Do not assume raw int8 WMMA throughput will overcome decode, LDS, scratch, or scheduling overhead.
- Do not treat PC sampling on these short kernels as quantitative latency data.
- Do not use shipped TensileLite grouped YAML choices or their displayed grouped GFLOP/s as a performance bound. Use them only as generator and interface evidence.
- Do not copy TensileLite's host-built grouped user-argument setup or per-workgroup GEMM search into the production routing path. Build compact metadata on the device and index it directly.
- Do not retry a wholesale direct-to-VGPR conversion, both-operands direct-to-VGPR, a two-LDS pipeline, GSU, split-K, or grouped Stream-K for the current packed representation.

## Final status

G1 completed the compile-time `J=64` and fixed-production-shape work and removed all production grouped-forward spills.

The complete fixed-`K=512` decoded-weight LDS cache was rejected in G2 because its LDS residency cost overwhelmed decode reuse.

G3 rejected fixed-shape `J=128`. `J=64` remains the production row tile. G4 then removed full-row address decomposition and moved every focused production class ahead of AITER.

G5 showed that removing the post-write barrier is neutral. G6 retained a compile-time two-block down schedule, and G7 retained pointer-increment traversal for gate/up.

G8 retained device row-task descriptors for large gate/up groups and rejected them for down. Batch-1 sparse routing keeps the serial G7 path.

G9 rejected the final high-value wider-output geometry despite zero spills. G10 confirmed that 256 threads cannot be applied to `I=64` without redesigning output ownership or introducing split-K-style reduction. G11 retained contiguous bounded activation-tail loads for down.

The current tile, scheduler, addressing, K-loop, synchronization, cache, and tail-control neighborhoods are exhausted. Further worthwhile work must change IQ2_S packed decode or representation rather than add geometry or control state.
