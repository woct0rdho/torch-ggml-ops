# Grouped MMQ forward optimization

## Scope

This document covers routed grouped MMQ forward on gfx1151.

The production operators are:

- `grouped_mmq_pair` for gate and up;
- `grouped_mmq` for down.

Dense MMQ forward and backward are documented separately in `docs/mmq_fwd_optimization.md` and `docs/mmq_bwd_optimization.md`.

The grouped baseline is now measured. The next implementation pass should first remove the grouped kernel's register spills, then revisit grouped tile geometry and scheduling.

## Production contract

The public activation and output dtype is BF16.

Activations are quantized internally to Q8_1. Packed GGUF weights remain the authoritative representation.

`expert_indices` and `expert_offsets` remain device-resident. The optimized path must not inspect group sizes through `.item()`, a device-to-host copy, a CPU descriptor, or an implicit synchronization.

The metadata ABI is:

- `expert_indices`: contiguous CUDA `torch.int64`, shape `[G]`;
- `expert_offsets`: contiguous CUDA `torch.int32`, shape `[G]`;
- `expert_offsets[-1] = R`;
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

- complete public packed-operator latency;
- logical throughput;
- incremental peak allocation and reservation growth;
- AITER configuration;
- routing metadata and distribution statistics;
- checkpoint-weighted forward and optimizer-step estimates;
- grouped-versus-dense-MMQ exactness;
- grouped-versus-dequantized-BF16 error.

The packed timing includes Q8_1 quantization and grouped multiplication.

The BF16 performance reference is AITER Triton `gmm`, using `_gmm_config` from `~/test_no_unsloth/fast_moe_lora.py`. It is not `torch.matmul`.

AITER receives independently dequantized BF16 versions of the same logical GGUF experts. Dequantization and active-weight selection are setup costs and are not inside the timed GMM call.

The AITER reference uses the same production transposed weight metadata as `fast_moe_lora.py`. `work_stealing` remains disabled, matching the production call.

The full baseline command was:

```bash
python bench/benchmark_grouped_mmq_fwd.py \
  --warmup 2 \
  --repeats 5 \
  --correctness-rows 128 \
  --output /tmp/grouped_mmq_fwd_baseline_full.json
```

The full artifact is:

```text
/tmp/grouped_mmq_fwd_baseline_full.json
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

## Baseline results

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

The gate/up pair is already strong for batch 1. It remains approximately tied at batch 4 and loses at batch 16.

The down projection is the dominant deficit for every batch. Q4_K and Q5_K are especially weak at batch 4.

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

The current packed kernel is mildly sensitive to skew at batch 4. Each `(expert, output tile)` workgroup serially processes every 128-row chunk for its expert, so large groups create longer-lived workgroups.

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

Batch-1 observed-style routing is already favorable overall. Batch 4 and batch 16 require substantial kernel improvement.

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

## Current packed kernel structure

The current grouped kernel uses:

- a `64 x 128` output-by-routed-row tile;
- four wave32 waves and 128 threads;
- one workgroup for each `(64-output tile, active expert)` pair;
- a serial loop over 128-row chunks within that expert;
- LDS staging for packed decoded weights and Q8_1 activations;
- int8 WMMA with FP32 accumulation;
- masked activation staging and BF16 stores for partial groups.

The dynamic LDS allocation is 40,448 bytes for Q3_K/IQ2_S and 38,400 bytes for Q4_K/Q5_K.

For gate/up, the launch has 8 output tiles per active expert. For down, it has 32 output tiles per active expert.

The paired operator quantizes once, then launches this grouped kernel twice. It does not maintain two accumulator sets in one kernel.

## Packed-kernel profiling

Kernel-trace artifacts are:

```text
/tmp/rocprof_grouped_gate_b1
/tmp/rocprof_grouped_gate_b16
/tmp/rocprof_grouped_down_q4_b4
```

### Gate/up Q3_K, batch 1, uniform

The profiled pair decomposed into:

- Q8_1 quantization: 0.829 ms;
- first grouped projection: 5.408 ms;
- second grouped projection: 5.359 ms.

The grouped projections account for about 93% of the packed operator time.

### Gate/up Q3_K, batch 16, uniform

The profiled pair decomposed into:

- Q8_1 quantization: 7.894 ms;
- two grouped projections: 122.445 ms total.

The grouped projections account for about 94% of the packed operator time.

### Down Q4_K, batch 4, uniform

The profiled single projection decomposed into:

- Q8_1 quantization: 0.916 ms;
- grouped projection: 21.964 ms.

The multiplication kernel is the first-order bottleneck. More quantizer work is not justified before fixing it.

## Packed code-object resources

The final extension code object was extracted from `.hip_fatbin` and inspected with `clang-offload-bundler`, `llvm-readobj`, `llvm-nm`, and `llvm-objdump`.

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

## AITER source, lowering, and profile

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

The similar L2 hit rates also argue against treating cache hit rate as the first optimization target. Register spills, tile shape, and the amount of scheduled work are stronger explanations.

`MemUnitBusy` was unavailable through dispatch-windowed counter collection on this gfx1151 profiler configuration and is not treated as a result.

## TensileLite grouped-GEMM generator assessment

The TensileLite investigation focused on generator capability rather than the quality or measured results of its shipped grouped-GEMM configurations. The grouped YAML files are evidence that the path is wired up, not evidence that their tile sizes or parameter choices are well tuned for gfx1151.

The principal sources were:

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

### TensileLite supports real grouped-workload tuning, but one shared solution

When grouped GEMM is enabled, the client combines the configured exact GEMMs into one `ContractionProblemGroupedGemm`. A candidate solution is checked against every member and the group is launched as one kernel call. The generator can therefore sweep ordinary solution parameters against the latency of a complete grouped workload rather than tuning each listed size independently.

The important limitation is that every GEMM in the group uses one shared static solution. Macro tiles, matrix-instruction geometry, `DepthU`, vector widths, LDS layout, and pipeline policy do not vary by expert inside one launch. This matches the fixed-shape production families if gate/up and down receive separate kernels:

| Family | Shared fixed shape | Dynamic dimension |
|---|---|---|
| Gate/up | output 512, K 2,048 | routed rows per expert |
| Down | output 2,048, K 512 | routed rows per expert |

TensileLite could explore a much broader standard BF16 grouped-GEMM search space than the shipped test YAMLs demonstrate. It still cannot directly generate this packed operator. Q8_1 activation handling, Q3_K/Q4_K/Q5_K/IQ2_S block decode, expert routing, scale application, and direct BF16 output are embedded in the project operation and are not expressible as an ordinary Tensile contraction problem without substantial new problem-type and code-writer work.

The grouped client also reports absolute throughput incorrectly for a multi-GEMM group: the complete grouped enqueue is timed, but `BenchmarkTimer.cpp` uses only `problem->gemms[0].flopCount()` for the displayed GFLOP/s. Candidate ordering for one fixed workload remains equivalent to elapsed-time ordering because the numerator is constant, but the reported grouped GFLOP/s is not a valid aggregate throughput. Historical grouped client results must not be used as a generator-quality bound.

### Grouped dispatch and device user arguments reinforce compact device descriptors

TensileLite computes cumulative workgroup ranges for the grouped problems and passes a workgroup table referenced through `wiTablePtr`. After a workgroup resolves its GEMM, it loads that GEMM's pointers, dimensions, strides, alpha/beta values, and epilogue arguments from a compact record. Device user arguments are supported by the generated kernel interface.

The inspected assembly writer contains a linear grouped-GEMM search path, while conversion code elsewhere contains a binary-search path. Repeating either search in every packed arithmetic workgroup is weaker than directly indexing a project-owned task descriptor, especially with up to 256 experts and an arithmetic body already spilling heavily.

TensileLite's runtime helper normally builds grouped user-argument records on the host and copies them to the device. That setup procedure is incompatible with the device-resident routing ABI, even though the generated device-argument interface itself is useful evidence. The project should borrow the compact record and cumulative-range ideas but construct its prefix metadata or row-tile descriptors on the GPU.

Kernel-argument preloading is not a substitute for this setup. Dispatch-time common arguments may be preloaded, but expert-specific pointers and row bounds are not known until the workgroup resolves a device record.

### Code-generation mechanisms and project applicability

| TensileLite mechanism | Generator capability | Project-owned interpretation |
|---|---|---|
| Matrix instruction and macro-tile selection | Static WMMA shape, wave tile, wave group, and thread-count variants | Sweep `I`, `J`, and 128/256-thread bodies only after spill removal |
| `DepthU` and fixed assertions | Compile-time K-loop depth and exact-divisibility paths | Specialize eight-block gate/up and two-block down loops |
| `UseSgprForGRO` | One VGPR global-read base plus multiple SGPR offsets for affine buffer loads | First-order candidate for reducing grouped address VGPR state |
| `UseInstOffsetForGRO` | Uses small buffer instruction offsets, especially with direct-to-LDS addressing | Express fixed packed-field displacements as compile-time offsets and verify ISA folding |
| `ScheduleIterAlg=3` | Distributes global reads and local writes across matrix instructions and models LDS-read latency | Stage packed read, decode, LDS commit, LDS read, and WMMA before adding explicit pacing |
| `GlobalReadPerMfma` and `LocalWritePerMfma` | Controls the density of VMEM and LDS-write issue relative to matrix instructions | Bounded source-level experiments such as one next-fragment load every fixed number of WMMAs |
| Wave-separated global reads | Assigns distinct global-read regions to waves when divisibility constraints hold | Let each wave own a disjoint packed-weight row slice while retaining shared activation LDS |
| `DirectToVgpr` | Skips local reads for one matrix operand under strict datatype, layout, and scheduler constraints | At most one immediately consumed wave-owned decoded-weight fragment |
| One/two LDS buffers and prefetch depth | Supports low-resource and deeper software pipelines | Retain one activation stage and one decoded-weight stage; do not copy a two-LDS pipeline |
| Store remapping | Uses LDS to turn scattered accumulator ownership into coalesced stores | Lower-priority C-shuffle using dead activation LDS after the final K step |
| Workgroup mapping and source swapping | Changes spatial tile order and operand ownership | Encode locality in descriptor order and measure wider project-owned output tiles |
| GSU and Stream-K families | Split-K mechanisms exist for ordinary GEMM; grouped Stream-K is explicitly unsupported | Do not use split-K; defer any project persistence until the row-tile body is spill-free |

### SGPR-based global-read addressing is the strongest direct code-generation lesson

`UseSgprForGRO` replaces a set of per-load VGPR offsets with one per-lane VGPR base and scalar offsets. The assembly writer allocates `ScalarGlobalReadOffsetA/B` only for the additional load locations and feeds them to buffer-load `soffset` operands. The automatic policy avoids this mode when the projected scalar-offset count becomes too large, and the implementation requires affine input layouts and precise buffer bounds.

Shift-pointer edge handling cannot generally use this representation because it needs to modify per-lane pointers. TensileLite disables SGPR offsets for such paths unless alignment and partial-load assertions guarantee that shifting is unnecessary. This maps cleanly to separate project full and tail bodies:

- exact production output tiles and fixed K blocks use one per-lane packed-load base plus wave-uniform scalar or compile-time offsets;
- the partial row tile retains bounded per-lane handling;
- expert pointer, row start, row count, output tile, K-block index, and fixed strides should be scalarized after descriptor resolution;
- explicit `readfirstlane` should be added only where disassembly proves that an actually uniform value remained in VGPRs.

The shipped gfx1151 standard-GEMM logic frequently resolves `_UseSgprForGRO=1`, confirming that this code-generation path is active on the target architecture. This is capability evidence, not evidence that the surrounding shipped solutions are optimal.

### Algorithm-3 scheduling is useful only after loader lifetimes are split

TensileLite's algorithm-3 scheduler computes how widely global reads and local writes can be spread across matrix instructions, attempts to avoid VMEM FIFO pressure, accounts for local-read instruction width and latency, and places waits and barriers relative to the compute sequence. It also has special wait accounting for direct-to-VGPR prefetch.

The project cannot reproduce this by adding barriers to the current monolithic decoder. The corresponding controlled sequence is:

- read a bounded packed byte fragment;
- decode that fragment with temporary metadata and bit-extraction state;
- commit decoded BF16 values to LDS or form one direct-register WMMA operand;
- let decode temporaries die;
- issue LDS reads and WMMA from the current fragment while only the next bounded read is live.

Only after this sequence is spill-free should source ordering or `__builtin_amdgcn_sched_group_barrier` be used to tune the number of VMEM and LDS-write operations between WMMAs. `GlobalReadPerMfma` and `LocalWritePerMfma` are design guidance for measured instruction-density ablations, not runtime configuration to reproduce.

### Wave-separated reads and direct-to-VGPR are alternatives, not one combined plan

TensileLite permits wave-separated reads only when the operand geometry divides cleanly across the workgroup's waves. For a TLU operand, `DepthU` must be divisible by the number of waves; for a non-TLU operand, the relevant macro-tile dimension must be divisible by the number of waves. The current four-wave packed geometry and fixed production dimensions provide natural candidates for disjoint 16-row weight ownership.

`DirectToVgpr` is much more constrained. It requires matrix instructions, buffer loads, `ScheduleIterAlg >= 3`, `PrefetchGlobalRead >= 1`, `InnerUnroll == 1`, compatible vector widths and matrix-instruction block replication, and it rejects wave-separated global reads. Enabling both A and B direct-to-VGPR is normally rejected because its wait handling performs poorly. Datatype conversion and sub-dword packing can also multiply the required register buffers across loop iterations.

More importantly, TensileLite direct-to-VGPR assumes that a global representation can become a matrix operand through ordinary layout and limited packing conversion. Packed GGUF decode requires scales, metadata, arbitrary bit extraction, and BF16 formation. A wholesale direct-to-VGPR conversion is therefore not applicable. The only justified first experiment is one wave-owned packed-weight fragment decoded into the exact register operand and consumed immediately. Activations remain in LDS because all four waves reuse them.

### TensileLite mechanisms that should not be copied directly

A direct global-to-LDS transfer cannot perform GGUF bit decode and scale application, even where the generator and architecture can emit the underlying load form. Packed bytes must pass through a decode lifetime before becoming the BF16 LDS representation.

Two live direct-to-VGPR operands, two LDS K stages, and deep prefetch are especially poor starting points while Q4_K/Q5_K already spill 127-129 VGPRs per thread.

GSU or split-K duplicates packed decode and requires a reduction or atomics for K dimensions that are already fixed at 512 or 2,048. TensileLite's grouped path does not support Stream-K, and that absence is consistent with deferring project-owned persistence.

Stagger-U and broad workgroup remapping are lower priority than direct descriptor ordering. Gate/up has only eight packed K blocks and down has two, while group-major and row-tile-major descriptors can express the important activation and weight locality without adding K-loop state.

## Composable Kernel and prior grouped-GEMM lessons

The relevant CK mechanisms were traced through:

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

The earlier BF16 grouped-GEMM investigation in `~/transformers-qwen3-moe-fused/doc/CK_REFERENCE.md` and `HIP_OPTIMIZATION_PLAN.md` is also relevant. It established that merely increasing LDS, adding a second LDS stage, inserting scheduler barriers into an unchanged loop, or under-launching a persistent grid did not reproduce CK performance. The useful CK lessons are structural: fixed-shape specialization, explicit load/compute ownership, bounded live ranges, local tile ordering, and instruction pacing after the dataflow is correct.

### Fixed-NK specialization matches the production problem

CK's fixed-NK grouped path keeps the shared N and K dimensions out of per-group scheduling state and uses offset block-to-tile maps for the dynamic M dimension.

The packed production scope is even narrower:

| Projection | Output rows per expert | Input features | GGUF blocks per weight row |
|---|---:|---:|---:|
| Gate/up | 512 | 2,048 | 8 |
| Down | 2,048 | 512 | 2 |

Project-owned kernels should therefore specialize `NRowsWeight` and `BlocksPerWeightRow` at compile time in addition to `I`, `J`, thread count, and quant type. Both output sizes are exact multiples of 64 and 128, so the production variants can remove output-tile edge predicates. The fixed K loop can be fully unrolled for down and statically scheduled for gate/up.

This is more than a small integer-arithmetic cleanup. A static two-block down kernel can use a different lifetime and LDS strategy than an eight-block gate/up kernel.

### The down projection can predecode its complete K tile before accumulators become live

For down, each weight row has exactly two packed GGUF blocks. Both decoded 256-value blocks for one 64-row output tile fit in LDS at the same time as one activation row tile.

Estimated complete dynamic LDS, including the existing shared prefix and activation tile, is:

| Type family | `J=64` | `J=128` |
|---|---:|---:|
| Q4_K/Q5_K layout | 48,384 bytes | 57,856 bytes |
| Q3_K/IQ2_S layout | 52,480 bytes | 61,952 bytes |

The production down types are IQ2_S, Q4_K, and Q5_K. All of these configurations remain below the 64 KiB workgroup limit. The current 38-40 KiB kernels already consume enough LDS that this is unlikely to reduce active workgroups further, but final occupancy must still be measured.

A down-specific kernel can load and decode both weight blocks into two immutable LDS regions before declaring or clearing the FP32 accumulator array. Decode temporaries then die before the accumulator lifetime begins. The existing `(expert, output tile)` workgroup can reuse those decoded weights across every row chunk in that expert.

This directly attacks both dominant down problems:

- Q4_K/Q5_K decode no longer overlaps the accumulator and dynamic row-loop state that currently pushes the kernel into 127-129 VGPR spills;
- packed weight loads and decode are amortized across two batch-4 uniform row chunks and up to eight batch-16 `J=128` row chunks.

This is not a CK-style double-LDS pipeline. The extra LDS holds immutable decoded weights for the fixed `K=512` problem; it does not create a second activation stage or ping-pong buffer. It should be tested before replacing the down scheduler with one-workgroup-per-row-tile descriptors, because the latter would discard this cross-row decoded-weight reuse.

### A CK-like pipeline requires split load, decode, and commit phases

The current grouped MMQ loop calls a monolithic packed-weight loader, fills the complete activation LDS tile, synchronizes, executes the dot-product helper, and synchronizes again. The compiler has little freedom to overlap packed VMEM, decode VALU, LDS writes, LDS reads, and WMMA.

CK's low-resource WMMA v1 pipeline uses one global prefetch stage and one LDS buffer. Its compute-optimized variants add register prefetch and explicit issue pacing without requiring a second LDS buffer. The transferable experiment is therefore:

- split each project-owned packed loader into `read packed bytes`, `decode`, and `commit decoded LDS` phases;
- prefetch only the next packed-weight fragment into a bounded register buffer while the current fragment is consumed;
- use one decoded-weight LDS stage and one activation LDS stage;
- add `__builtin_amdgcn_sched_group_barrier` pacing only after the staged structure is spill-free and generated instruction counts are known.

The two fixed-K families should not share one pipeline policy. Gate/up has eight packed blocks per row and is the candidate for a CK-v3-like staged hot loop. Down has only two blocks and should first use a low-resource fully unrolled schedule or the complete decoded-weight cache above; a deep prologue is unlikely to amortize.

The CK `BSkipLDS` mechanism also suggests a narrower project-owned ablation. Every current wave owns a disjoint 16-row slice of the 64-row weight tile, while all four waves reuse the activation tile. A direct-register decoded-weight fragment is therefore a plausible LDS-bypass path. Bypassing activation LDS would duplicate activation traffic across four waves and is not the corresponding opportunity. This experiment requires a new wave-owned decoder and should hold only one WMMA fragment at a time; directly reusing the monolithic loader would simply move the spill problem.

CK's direct global-to-LDS transfer path is not a gfx1151 mechanism. CK enables it for gfx90a, gfx942, gfx950, and gfx125-class targets, not gfx11. It should not be treated as an available optimization here.

### Device scheduling should use compact prefix metadata and locality-preserving tile order

CK's tile-loop kernels and CK Tile persistent grouped GEMM keep scheduling on device, advance logical tile IDs by the physical grid size, and preserve per-group block ranges. They also use adaptive tile maps so nearby workgroups cover a small spatial tile neighborhood instead of traversing one matrix dimension in a purely linear order.

For this project, `G <= 256` makes an atomics-free row-tile setup practical. One workgroup can compute a prefix sum of `ceil(group_size / J)` and write compact 64-bit descriptors containing `(expert, row_start, row_count)`. Each group's descriptor range is known from the prefix; no per-expert atomic reservation is required. The existing upper bound remains:

```text
ceil(R / J) + G
```

The descriptor order should be group-major and row-tile-major, with output tile as the fastest-changing launch dimension. That causes the 8 gate/up or 32 down output tiles for one activation row tile to run near each other, preserving activation L2 locality while exposing row tiles independently for load balancing.

A prefix-only alternative can binary-search the device tile offsets in the compute kernel, similar to CK's non-persistent group lookup. It avoids materializing descriptors but reintroduces dynamic scheduler state into the arithmetic kernel. Because register spilling is the current blocker, the compact descriptor load is the preferred first implementation.

Persistent grid-stride traversal is a later ablation, not the initial descriptor design. The prior BF16 HIP kernel regressed when persistence increased control-state VGPRs or under-launched the grid. If tested here, the one-row-tile body must already be spill-free, the physical grid must be large enough to saturate gfx1151, and the grid size should be a multiple of both 8 and 32 so gate/up and down output-tile clusters remain aligned. Candidate physical grids are 160, 256, and 320 workgroups, subject to measured occupancy.

CK repeatedly uses `readfirstlane` for tile coordinates and K-loop counts. Generated ISA should be checked to verify that expert ID, row bounds, fixed-shape strides, tile coordinates, and base pointers are held as wave-uniform scalar state rather than duplicated VGPR state. Explicit `__builtin_amdgcn_readfirstlane` should be introduced only where the compiler failed to scalarize an actually uniform value.

### Full tiles and the epilogue deserve separate paths inside one launch

Status: complete in G4 for `J=64`. Full row tiles use contiguous Q8_1 loads and unmasked BF16 row stores; only the final partial tile retains bounds handling.

Most batch-4 and batch-16 uniform row tiles are full. A full-row-tile body should have no activation bounds checks. The one tail tile per active expert can use a bounded partial path.

The AITER edge strategy also suggests a branchless tail-load ablation: clamp or wrap invalid local rows to any valid row from the same expert and mask only the final store. Values computed for padded rows are unobservable. This trades some tail traffic for simpler activation staging and must be judged carefully on batch-1 sparse routing, where padding can be a large fraction of work.

The prior BF16 HIP experiments showed that a second full-tile launch loses to launch overhead. Full and partial bodies should therefore remain in one public compute launch unless a later measurement establishes a different precondition.

CK's gfx1151 path uses C-shuffle rather than its gfx12 direct-store path. The current MMQ writeback follows the WMMA accumulator layout and should be inspected for scattered BF16 stores. A project-owned C-shuffle can reuse the activation LDS region after the final K step: a complete `64 x 128` BF16 output tile is 16 KiB and fits inside the 18 KiB `J=128` activation region. This is lower priority than spill removal, and the earlier BF16 C-shuffle experiments show that ownership and barriers must be explicit to avoid overwrite hazards.

### CK mechanisms that are not current priorities

Split-K is not attractive for fixed `K=512/2048`: it duplicates packed decode and requires atomics or a reduction while the production K dimensions are already modest.

A second LDS K stage should not be retried without a new resource argument. It regressed the earlier BF16 grouped kernel and is unnecessary for the decoded-weight-cache design.

Scheduler barriers should not be sprinkled into the current monolithic loop. The earlier BF16 ablation was inconclusive, and the barriers become meaningful only after load, decode, LDS commit, LDS read, and WMMA phases are structurally separable.

## Lessons from dense MMQ forward and backward

The dense work provides several directly applicable rules.

First, inspect the final code object after every structural change. Source-level register intuition was repeatedly insufficient, and the grouped baseline is already a concrete example.

Second, use compile-time typed variants and measured dispatch. Q3_K, Q4_K, Q5_K, and IQ2_S have different decode and resource behavior.

Third, optimize full operator latency. The grouped quantizer is only about 4-8% of the profiled operator, so multiplication changes dominate the current opportunity.

Fourth, larger cooperative tiles can win when they amortize metadata, packed weight decode, and activation staging without spilling. The dense backward pass benefited from multiple WMMA tiles per wave and shape-specific layouts.

Fifth, LDS layout changes matter only after the dominant resource problem is under control. The dense backward pass gained from vector LDS loads, padding, and XOR swizzles, but the grouped Q4_K kernel currently has 127 VGPR spills.

Sixth, packed extraction and multi-value decode should be type-specific. Dense Q3_K, Q5_K, and Q6_K did not share one universal winner.

Finally, gfx1151 int8 WMMA has no decisive raw throughput advantage over BF16 WMMA. Packed grouped MMQ must win through smaller authoritative weights, decode quality, traffic reduction, reuse, and lower scheduling overhead.

## Bottleneck interpretation

### Register spilling is the first blocker

Q4_K and Q5_K reach 256 VGPRs and spill 127-129 VGPRs per thread.

The current dynamic grouped loop adds enough live state to turn a spill-free dense arithmetic body into a scratch-heavy grouped kernel.

Removing this spill footprint has higher priority than cache tuning, extra prefetching, or quantizer changes.

### The current output tile is narrower than AITER

Packed MMQ produces 64 output columns per workgroup. AITER uses 128.

For down, this means 32 packed output tiles per expert instead of 16 AITER output tiles. Packed decode and Q8 staging must compensate for that extra scheduling and tile overhead, but currently do not.

A future project-owned `I=128` tile is therefore a meaningful higher-ceiling direction. It must be paired with 256 threads or another accumulator layout that does not increase per-thread register pressure.

### Serial expert-chunk loops limit balancing, but down can exploit their reuse

One packed workgroup owns one expert and output tile, then serially processes all 128-row chunks for that expert.

This couples workgroup lifetime to group size and is especially problematic for skewed gate/up groups. A row-tile scheduler can expose more parallel work and compile a dense-like single-tile body.

The ownership is not uniformly bad, however. For fixed `K=512` down projections, it allows one workgroup to predecode both complete weight blocks before the row loop and reuse them across every row tile. Scheduler policy should therefore be shape-specific: descriptor-based row tiles are the stronger default direction for gate/up, while down should first test the decoded-weight-cache family and retain serial row ownership if its reuse outweighs load-balance costs.

### Quantization is not the dominant deficit

The quantizer contributes about 0.8-0.9 ms at the smaller profiled points and 7.9 ms for the batch-16 gate/up pair.

It is already shared by gate/up. Removing all quantization cost would not close the down or batch-16 gate/up gaps.

### Pair fusion is not yet justified

A fused two-weight compute kernel could load each Q8 activation tile once, but it would also need two accumulator sets or sequential projection phases.

Two live accumulator sets would worsen the current register problem. Pair compute fusion should be reconsidered only after a spill-free single-projection kernel exists.

## Optimization experiment log

### G1: compile-time `J=64` and fixed production shapes

Status: retained.

The first implementation combined the two highest-priority spill-removal changes:

- compile-time `J=64` for both production shape families;
- fixed gate/up `NRowsWeight=512, BlocksPerWeightRow=8` and down `NRowsWeight=2048, BlocksPerWeightRow=2` variants;
- no output-row fallback in the fixed variants because every production output tile is complete;
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

- gate/up Q3_K batch 1 sparse: both outputs had zero differing BF16 elements;
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

- loads the Q8_1 activation region as one contiguous integer span with no row division, remainder, or predicate;
- writes a complete BF16 row tile with no row predicate;
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

- gate/up Q3_K batch 1 sparse: zero differing BF16 elements for both outputs;
- down Q4_K batch 4 boundary: zero differing BF16 elements.

The complete gate/up Q3_K batch-1 sparse point measured 5.368 ms versus 14.131 ms AITER, or 2.63x. The complete down Q4_K batch-4 boundary point measured 6.962 ms versus 9.286 ms AITER, or 1.33x. G4 moves every focused production class ahead of its baseline AITER reference.

The magnitude of this gain shows that integer row decomposition and per-load tail control, not only register spilling, were first-order costs in the inherited grouped loop.

## Optimization plan

### Phase 1: compile-time row, fixed-shape, and scalar-address specialization

Status: complete in G1 and bounded by G3. The retained implementation uses `J=64`, fixed production output/K shapes, and full output tiles. Fixed-shape `J=128` regressed every focused point and restored private segments for Q3_K/Q4_K/Q5_K. Further scalar-address changes remain valid only as focused ISA-guided ablations against the spill-free G1 body.

Generalize the grouped kernel over compile-time `J`, as dense forward already is.

Measure `J=64` and `J=128` for every production type and shape. `J=64` halves the accumulator footprint and activation LDS footprint.

At the same time, introduce shape-typed launch variants for:

- gate/up: `NRowsWeight=512`, `BlocksPerWeightRow=8`;
- down: `NRowsWeight=2048`, `BlocksPerWeightRow=2`.

The fixed production output sizes permit full output-tile loaders and stores with no `i_max` path. Keep a general fallback only for tests and non-production shapes.

Within the exact full-tile variants, test a TensileLite-style affine buffer-load addressing structure: one per-lane VGPR base for each packed input plus wave-uniform scalar or compile-time offsets for fixed packed blocks, metadata fields, and fragment positions. Keep shift-pointer or per-lane edge handling out of this path and inspect generated ISA for actual `soffset`, immediate-offset, and scalar pointer use rather than relying on source-level uniformity.

Resolve expert ID, row bounds, output tile, fixed strides, and K-loop bounds once. Add explicit `readfirstlane` only where the final ISA shows failed scalarization. Reject any variant that merely exchanges VGPR pressure for excessive SGPR allocation or dynamic stack use.

Batch-1 gate/up is the strongest initial `J=64` candidate because its uniform groups are exactly 64 rows and its observed-style groups are mostly below 128 rows.

Structure the row loop as full tiles followed by at most one partial tile. Test a clamped or wrapped activation tail load against the existing zero-fill path, but retain the batch-1 sparse winner.

Inspect the code object after each build. Reject variants that retain large private segments even if one short timing appears favorable.

Also test whether extracting one row-tile arithmetic body from the dynamic group loop reduces live state. A no-inline device helper is only a controlled ablation; it should not be retained if it creates dynamic stack use or device call overhead.

Required focused points are:

- gate/up Q3_K batch 1 sparse and uniform;
- gate/up IQ2_S batch 1 sparse;
- gate/up Q3_K batch 4 boundary;
- gate/up Q3_K batch 16 uniform;
- down Q4_K batch 4 uniform and boundary;
- down Q5_K batch 4 uniform;
- down IQ2_S batch 16 uniform.

### Phase 2: fixed-K decoded-weight cache for down

Status: rejected in G2. The complete two-block BF16 LDS cache reduced performance by 27-40% relative to G1 because the additional 19-22 KiB LDS cost outweighed decode reuse.

The phase is closed for the current `I=64` geometry. Do not retry the same cache at `J=128`, whose estimated 57,856-61,952 byte allocation would further reduce resource headroom. Revisit cross-row weight reuse only if a later output geometry provides a substantially more compact cached representation or recovers workgroup residency.

### Phase 3: atomics-free device-resident row-tile descriptors

For gate/up, or for down if the decoded-weight cache does not resolve the deficit, move row-chunk scheduling out of the multiplication kernel.

A one-workgroup GPU setup pass should compute a prefix sum of `ceil(group_size / J)` for `G <= 256` and write compact `(expert, row_start, row_count)` descriptors into prefix-assigned slots. Avoid one atomic reservation per expert.

The descriptor storage upper bound is known without a device read:

```text
ceil(R / J) + G
```

A compute grid can launch to that upper bound and have excess workgroups return after reading a device-resident task count.

Order descriptors by group and row tile, and make output tile the fastest-changing launch dimension. This preserves activation locality across the 8 gate/up or 32 down output tiles for one row tile.

The multiplication kernel then processes exactly one row tile. Its structure can closely reuse the already benchmarked dense-MMQ tile body and should recover the dense kernel's spill-free lowering. Direct descriptor indexing is also preferable to reproducing TensileLite's per-workgroup linear or binary grouped-GEMM search inside the arithmetic kernel.

`grouped_mmq_pair` should build the prefix and descriptors once and reuse them for both packed projections.

The setup launch, descriptor storage, and empty upper-bound workgroups must be included in public-operator timing.

Only after the non-persistent descriptor body is spill-free, test persistent grid-stride traversal with 160, 256, and 320 workgroups. Reject configurations that increase control-state spills, under-launch the GPU, or disrupt output-tile locality.

### Phase 4: larger project-owned output tiles

After obtaining a spill-free arithmetic body, measure larger output tiles.

The primary neighborhood is:

| Output tile `I` | Row tile `J` | Threads | Purpose |
|---:|---:|---:|---|
| 64 | 64 | 128 | minimum accumulator and LDS pressure |
| 64 | 128 | 128 | current arithmetic reuse without the grouped loop |
| 128 | 64 | 256 | wider output tile with 32 FP32 accumulators/thread |
| 128 | 128 | 256 | AITER-like logical tile with 64 FP32 accumulators/thread |

`I=128, J=128, 256 threads` keeps the current 64 accumulators per thread while halving output-tile count and Q8 activation reloads. Its estimated dynamic LDS is near the 64 KiB workgroup limit, so exact code-object and launch-resource checks are mandatory.

For the 256-thread variants, the per-thread share of a prefetched packed-weight tile is similar to or smaller than the 128-thread `I=64` path. This is the first sensible point to test a bounded CK-like register-prefetch stage.

Keep this logic in project-owned files. Do not modify `csrc/vendor/llama_cpp/*`.

Use measured dispatch based on quant type, logical shape, total rows, and group count. `R / G` is available from tensor dimensions and does not require reading device metadata.

### Phase 5: staged packed decode, LDS bypass, and type-specific tuning

Once the tile and scheduler are stable, split project-owned packed loaders into read, decode, and LDS-commit phases.

For gate/up `BlocksPerWeightRow=8`, test a single-LDS staged pipeline that prefetches only the next packed-weight fragment in registers and interleaves packed VMEM, decode, LDS writes, LDS reads, and WMMA. Use TensileLite algorithm-3 scheduling as design guidance: measure a small neighborhood of global-read and LDS-commit issue densities relative to WMMA rather than adding an unconstrained prefetch window. Do not add a second activation or decoded-weight LDS stage.

For down `BlocksPerWeightRow=2`, prefer the fully unrolled decoded-weight-cache schedule over a deep pipeline unless measurement shows otherwise.

Test wave-separated packed-weight reads and a wave-owned direct-register decoded-weight fragment as separate alternatives. The first gives each wave a disjoint weight slice when the fixed tile divides naturally; the second holds one decoded WMMA fragment and consumes it immediately. Do not combine them initially, because TensileLite's own direct-to-VGPR path rejects wave-separated global reads. Keep activations in LDS because all four waves reuse them.

After the dataflow is spill-free, apply the successful dense techniques:

- cooperative multi-value packed decode;
- vector global loads where alignment permits;
- vector LDS fragment loads;
- type-specific packed extraction;
- padding or XOR swizzles selected by measurement;
- explicit `sched_group_barrier` pacing derived from generated instruction counts;
- exact full-tile specializations with bounded partial-tile paths.

Q4_K/Q5_K down is the first resource gate. Q3_K/IQ2_S gate/up should preserve the existing batch-1 advantage while improving large batches.

Do not assume one decode or scheduling layout will win across all four production types.

### Phase 6: epilogue and pair-specific reuse

After spill removal, inspect the generated BF16 store pattern. If direct writeback is scattered, test a C-shuffle that converts to BF16 and reuses the dead activation LDS region. Do not allocate a separate epilogue LDS buffer.

Only after the single-projection kernel is spill-free, test pair-specific changes.

Possible controlled experiments are:

- sharing row descriptors;
- sharing quantization, which already exists;
- interleaving the two projection launches for cache behavior;
- a fused activation-load path if register accounting proves it viable.

Do not retain a C-shuffle with ownership hazards or a fused pair kernel that increases scratch traffic or loses the batch-1 sparse advantage.

### Phase 7: end-to-end validation

Re-run the full 60-point matrix after every retained dispatch change.

Report model-weighted grouped base-projection estimates for every routing distribution, not only one favorable case.

Then validate the real `fast_moe_lora.py` training path with current-stream correctness, activation checkpointing, and production routing metadata.

Run:

```bash
pytest -q tests/
python -m compileall -q bench

git diff --check
```

## Acceptance criteria

Every retained kernel must preserve exact grouped-versus-dense-MMQ BF16 output.

Q4_K and Q5_K grouped kernels should have no private segment or a small, measured, justified remainder. The current 512-520 bytes per thread is not acceptable as a final configuration.

Batch-1 sparse gate/up must retain a clear advantage over AITER.

Batch-4 down and batch-16 gate/up are the primary performance gates. A local improvement that regresses the model-weighted optimizer-step estimate should be rejected.

AITER remains the reference, not the ceiling. The final target is the best end-to-end packed grouped latency that preserves the production contract.

## Non-priorities and rejected directions

Do not replace AITER GMM with `torch.matmul` as the performance reference.

Do not link packed kernels against AITER or hipBLASLt.

Do not construct full logical BF16 expert matrices in the packed production path.

Do not introduce CPU expert descriptors or metadata synchronization.

Do not prioritize another quantizer rewrite before multiplication spills are removed.

Do not use runtime environment variables or online autotuning for dispatch.

Do not assume raw int8 WMMA throughput will overcome decode, LDS, scratch, or scheduling overhead.

Do not treat PC sampling on these short kernels as quantitative latency data.

Do not use shipped TensileLite grouped YAML choices or their displayed grouped GFLOP/s as a performance bound. Use them only as generator and interface evidence.

Do not copy TensileLite's host-built grouped user-argument setup or per-workgroup GEMM search into the production routing path. Build compact metadata on the device and index it directly.

Do not attempt a wholesale direct-to-VGPR conversion, both-operands direct-to-VGPR, a two-LDS pipeline, GSU, split-K, or grouped Stream-K for the initial packed implementation.

## Immediate next step

G1 completed the compile-time `J=64` and fixed-production-shape work and removed all production grouped-forward spills.

The complete fixed-`K=512` decoded-weight LDS cache was rejected in G2 because its LDS residency cost overwhelmed decode reuse.

G3 rejected fixed-shape `J=128`; `J=64` remains the production row tile. G4 then removed full-row address decomposition and moved every focused production class ahead of AITER.

Continue with low-risk synchronization and fixed-K loop cleanup on the spill-free G4 body. Remove only barriers that are unnecessary by shared-memory lifetime analysis, then test a two-block down unroll and pointer-increment gate/up loop. If those neighborhoods plateau, evaluate whether descriptor setup can beat the already saturated serial grid rather than assuming it will help.
