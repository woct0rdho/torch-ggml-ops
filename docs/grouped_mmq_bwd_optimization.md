# Grouped MMQ backward optimization

## Status at a glance

The production optimization pass for grouped MMQ backward on gfx1151 is complete for the current packed GGUF representations.

Sources of record:

```text
Historical baseline: /tmp/grouped_mmq_bwd_baseline_full.json
Final retained result: /tmp/grouped_mmq_bwd_final_full.json
Final code object: /tmp/grouped_bwd_final_readobj.txt
Final disassembly: /tmp/grouped_bwd_final_disasm.txt
```

Final outcome:
- packed wins 31/60 individual case/batch/routing points against BF16 AITER GMM.
- both fused gate/up families win all 24 points.
- the checkpoint-weighted grouped-MMQ backward estimate wins all 12 batch/routing combinations.
- final weighted speedup over AITER is 1.25-1.85x at batch 1, 1.54-1.63x at batch 4, and 2.07-2.16x at batch 16.
- final weighted speedup over the historical packed baseline is 3.50-4.61x at batch 1, 7.34-9.30x at batch 4, and 12.33-13.15x at batch 16.
- all retained production kernels have zero private segment, zero VGPR spills, zero SGPR spills, and no dynamic stack.

### Done

- Production benchmark and correctness infrastructure.
- Real-checkpoint 60-point baseline and final matrices.
- BF16 AITER GMM production reference using the application heuristic.
- Q4_K, Q5_K, and IQ2_S tiled single-down kernels.
- Fused Q3_K and IQ2_S pair kernels with one FP32 accumulator set and one BF16 rounding.
- Small S1, selective S2, large serial, and M-major device-row-task scheduling.
- Project-owned cooperative IQ2_S width-16 decode.
- Type- and operator-specific LDS layouts.
- Final trace, counter, code-object, correctness, allocation, dense-control, and integration validation.
- Documentation of retained and rejected experiments.

### Remaining work

There is no pending local tile, scheduler, decoder-width, swizzle, prefetch, or integration task for the current execution model.

Further performance work is representation-level and should begin only with a design that changes reuse across calls or changes the format consumed by the arithmetic kernel. Candidate directions are:
- a compact lossless decoded cache substantially smaller than BF16.
- cross-call decoded-weight reuse.
- reusable device task/decode metadata.
- a transient active-expert decode amortized over several projections or calls.
- a persistent lossless integer-plus-scale representation consumed directly by grouped WMMA.

Any future path must include decode time, workspace allocation, routing behavior, and complete public-operator latency in its acceptance benchmark. It must preserve sparse active-expert behavior and must not become a permanent full BF16 shadow copy.

## Scope and production contract

This document covers routed grouped MMQ input-gradient operators:
- `grouped_mmq_grad_input` for one frozen packed expert projection.
- `grouped_mmq_pair_grad_input` for the fused gate/up input gradient.

For a routed expert group, forward computes:

```text
Y[M, N] = X[M, K] @ W[N, K].T
```

The backward operator covered here computes:

```text
dX[M, K] = dY[M, N] @ W[N, K]
```

The public cotangent and result are BF16. The authoritative weights remain packed GGUF tensors. Unlike grouped forward, grouped backward consumes BF16 cotangents directly and does not quantize them to Q8_1.

The retained implementation preserves:
- direct BF16 `grad_input` output.
- the current CUDA/HIP stream.
- no transient logical dense weight matrix in the ordinary path.
- no arithmetic workgroups for inactive experts.
- fused pair accumulation of gate and up into one FP32 accumulator set followed by one BF16 rounding.
- output-only fused-pair allocation rather than two public outputs plus `torch.add`.

The metadata ABI is unchanged:
- `expert_indices`: contiguous CUDA `torch.int64`, shape `[G]`.
- `expert_offsets`: contiguous CUDA `torch.int32`, shape `[G]`.
- `expert_offsets[-1] = R`.
- `G <= 256`.

Production code must not add `.item()`, CPU descriptors, device-to-host metadata copies, or metadata synchronization.

Related documentation:
- dense backward: `docs/mmq_bwd_optimization.md`.
- grouped forward: `docs/grouped_mmq_fwd_optimization.md`.

Current implementation files:

```text
csrc/ck/grouped_mmq_backward.cuh
csrc/ck/grouped_mmq_backward_tiled.cuh
csrc/ck/gguf_decode.cuh
csrc/mmq_hip.cu
```

Do not edit `csrc/vendor/llama_cpp/*` for project-specific decode logic.

## Production workload and reference

The benchmark uses real tensors from `Qwen3.6-35B-A3B-APEX-I-Mini.gguf`.

| Case | Logical expert weight | Backward GEMM | GGUF type | Layers | Operator |
|---|---:|---:|---|---:|---|
| Gate/up outer | `512 x 2048` | `M x 512` by `512 x 2048` | Q3_K | 20 | fused pair |
| Gate/up middle | `512 x 2048` | `M x 512` by `512 x 2048` | IQ2_S | 20 | fused pair |
| Down middle | `2048 x 512` | `M x 2048` by `2048 x 512` | IQ2_S | 20 | single |
| Down outer main | `2048 x 512` | `M x 2048` by `2048 x 512` | Q4_K | 18 | single |
| Down outer edge | `2048 x 512` | `M x 2048` by `2048 x 512` | Q5_K | 2 | single |

The production sequence length is 2,048 and top-k is 8.

| Physical batch | Routed rows | Uniform rows per expert |
|---:|---:|---:|
| 1 | 16,384 | 64 |
| 4 | 65,536 | 256 |
| 16 | 262,144 | 1,024 |

### Routing distributions

The benchmark covers four deterministic routing distributions:
- `uniform`: all 256 experts have equal row counts.
- `skewed`: all 256 experts are active with deterministic nonuniform sizes.
- `sparse`: 192, 224, and 240 active experts at batches 1, 4, and 16.
- `boundary`: includes group sizes 1, 15, 16, 17, 63, 64, 65, 127, 128, and 129.

Representative route ranges:

| Batch | Distribution | Active experts | Minimum rows | Maximum rows | Mean rows |
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

For a 128-row tile, the total number of expert-local row chunks is:

| Batch | Uniform | Skewed | Sparse | Boundary |
|---:|---:|---:|---:|---:|
| 1 | 256 | 256 | 192 | 257 |
| 4 | 512 | 639 | 613 | 660 |
| 16 | 2,048 | 2,174 | 2,166 | 2,167 |

These counts explain why row-task ordering and nonuniform launch overhead were measured explicitly rather than inferred from uniform routing.

### BF16 AITER reference

The production comparison is AITER Triton `gmm`, not `torch.matmul`.

The benchmark uses the project-owned heuristic:

```text
torch_ggml_ops.aiter_gmm_heuristics.gmm_config
```

The timed AITER path starts with independently dequantized BF16 experts. Dequantization and active-expert `index_select` are setup costs and are not included in the GMM timing.

For fused gate/up backward, the AITER reference includes two GMM calls and `torch.add` of the two BF16 results. This is the correct production comparison, but it is not numerically identical to the packed fused pair because AITER rounds each projection before the add.

The production heuristic selects:

| Backward shape | M tile | N tile | K tile | Threads | Persistent programs |
|---|---:|---:|---:|---:|---:|
| Gate/up: `K=512, N=2048` | 128 | 128 | 64 | 256 | 256 |
| Down: `K=2048, N=512` | 64 | 128 | 64 | 256 | 256 |

AITER is the production reference, not a performance ceiling.

### Benchmark infrastructure and rules

Primary files:

```text
bench/benchmark_grouped_mmq_bwd.py
bench/grouped_mmq_benchmark_common.py
bench/benchmark_grouped_mmq_fwd.py
```

The benchmark records complete public-operator latency, logical throughput, allocation growth, AITER configuration, routing summaries, checkpoint-weighted estimates, grouped-versus-dense correctness, and grouped-versus-AITER correctness.

All GPU benchmarks and profiler runs must be sequential. Concurrent GPU measurements are invalid because of contention. Inputs must be real and nonzero. Zero WMMA operands can misrepresent performance.

Final matrix command:

```bash
python bench/benchmark_grouped_mmq_bwd.py \
  --warmup 3 \
  --repeats 9 \
  --correctness-rows 256 \
  --output /tmp/grouped_mmq_bwd_final_full.json
```

## Final retained implementation

### Production dispatch

Dispatch uses host-visible `rows`, `num_groups`, tensor shape, and quant type. It does not read device offsets on the host and does not use online autotuning or environment-variable selection.

| Operator/type | Dispatch |
|---|---|
| Q3_K pair | S1 `64x64x32` when `rows / num_groups < 128`. Large `128x64x32` otherwise |
| IQ2_S pair | S1 `64x64x32` when `rows / num_groups < 128`. Large `128x64x32` otherwise |
| Q4_K down | S1 below average 80 rows. S2 `128x64x32` at 80-127. M-major row tasks `128x128x32` at 128+ |
| Q5_K down | S1 below average 128 rows. M-major row tasks `128x128x32` at 128+ |
| IQ2_S down | S1 below average 80 rows. S2 `128x64x32` at 80-127. M-major row tasks `128x128x32` at 128+ |
| Unsupported shapes/types | Original generic grouped kernel |

The average is a dispatch hint only. Every selected kernel still handles skewed groups, tails, inactive experts, and the boundary distribution correctly.

### Arithmetic and layout choices

Retained common mechanisms:
- four wave32 waves and 128 threads.
- K=32.
- cooperative width-16 packed decode.
- exact fixed production shapes.
- bounded decode-temporary lifetimes.
- no compiler-managed local arrays as prefetch state.
- separate exact small/full handling and bounded tails where measured.
- BF16 output stores after FP32 WMMA accumulation.

Retained type/operator layouts:

| Family | LDS layout or decode choice |
|---|---|
| Q3_K pair | padded rows. No long-lived packed-byte prefetch for the 110-byte block |
| Q4_K down | sixteen-BF16 XOR swizzle |
| Q5_K down | four-BF16 XOR swizzle with bounded low/high packed prefetch |
| IQ2_S down | sixteen-BF16 XOR swizzle |
| IQ2_S pair | four-BF16 XOR swizzle |

IQ2_S uses a project-owned width-16 decoder that loads two grid entries, two sign bytes, one shared scale nibble, and one `d` factor for sixteen aligned values. Vendored tables and structs remain unchanged.

### M-major device row tasks

Large Q4_K, Q5_K, and IQ2_S down use device-built 128-row tasks. The existing atomics-free 256-thread prefix-sum setup emits device-resident task metadata and launches four adjacent N workgroups per row task.

M-major ordering is required: all four N workgroups for one row task stay adjacent. N-major ordering nearly doubled latency. Pair and batch-1 paths remain serial, preserving sparse behavior and the fused pair's output-only allocation.

## Final results

### Individual points

Packed wins 31/60 individual case/batch/distribution points.

| Family | Wins | Packed/AITER throughput ratio range | Median ratio |
|---|---:|---:|---:|
| Gate/up Q3_K fused pair | 12/12 | 1.60-3.03 | 2.25 |
| Gate/up IQ2_S fused pair | 12/12 | 1.46-2.94 | 2.23 |
| Down Q4_K single | 1/12 | 0.79-1.27 | 0.86 |
| Down Q5_K single | 3/12 | 0.83-1.29 | 0.92 |
| Down IQ2_S single | 3/12 | 0.83-1.32 | 0.93 |

Representative final event-timed points:

| Family | Batch/distribution | Packed ms | AITER ms | Packed/AITER throughput |
|---|---|---:|---:|---:|
| Gate/up Q3_K | B1 uniform | 3.640 | 11.014 | 3.03x |
| Gate/up Q3_K | B4 uniform | 10.605 | 24.418 | 2.30x |
| Gate/up Q3_K | B16 uniform | 45.079 | 127.244 | 2.82x |
| Gate/up IQ2_S | B1 uniform | 4.034 | 11.026 | 2.73x |
| Gate/up IQ2_S | B4 uniform | 10.708 | 24.885 | 2.32x |
| Gate/up IQ2_S | B16 uniform | 45.666 | 129.742 | 2.84x |
| Down Q4_K | B1 uniform | 3.885 | 3.379 | 0.87x |
| Down Q4_K | B4 uniform | 10.459 | 8.999 | 0.86x |
| Down Q4_K | B16 uniform | 36.195 | 45.869 | 1.27x |
| Down Q5_K | B1 uniform | 3.850 | 3.386 | 0.88x |
| Down Q5_K | B4 uniform | 10.454 | 9.134 | 0.87x |
| Down Q5_K | B16 uniform | 34.316 | 44.300 | 1.29x |
| Down IQ2_S | B1 sparse | 3.595 | 3.186 | 0.89x |
| Down IQ2_S | B4 uniform | 9.643 | 8.941 | 0.93x |
| Down IQ2_S | B16 uniform | 33.549 | 44.339 | 1.32x |

The pair families are the dominant checkpoint-weighted win. The remaining individual losses are single down projections, especially small and nonuniform routes.

### Checkpoint-weighted estimate

The estimate multiplies each case by its checkpoint layer/call count and sums the five grouped-MMQ backward families. It is not the complete model backward including unrelated operators.

| Batch | Distribution | Final packed ms | AITER ms | Packed speedup over AITER | Historical packed / final |
|---:|---|---:|---:|---:|---:|
| 1 | uniform | 311.7 | 576.3 | 1.85x | 4.61x |
| 1 | skewed | 411.2 | 579.5 | 1.41x | 3.55x |
| 1 | sparse | 381.4 | 476.7 | 1.25x | 3.67x |
| 1 | boundary | 397.4 | 581.1 | 1.46x | 3.50x |
| 4 | uniform | 828.3 | 1,345.1 | 1.62x | 9.30x |
| 4 | skewed | 983.3 | 1,585.7 | 1.61x | 7.34x |
| 4 | sparse | 969.9 | 1,496.8 | 1.54x | 7.87x |
| 4 | boundary | 987.3 | 1,613.5 | 1.63x | 7.46x |
| 16 | uniform | 3,206.0 | 6,940.8 | 2.16x | 13.15x |
| 16 | skewed | 3,339.8 | 6,961.4 | 2.08x | 12.56x |
| 16 | sparse | 3,371.2 | 7,036.8 | 2.09x | 12.57x |
| 16 | boundary | 3,358.9 | 6,936.8 | 2.07x | 12.33x |

The production-weighted operator wins every measured batch/routing combination even though single down loses most isolated points.

### Correctness and allocation

All final single-projection correctness samples are bitwise equal to both per-expert dense packed backward and BF16 AITER: zero differing BF16 elements, zero maximum absolute error, and zero normalized RMSE.

The fused pair intentionally differs from two separately rounded BF16 projections followed by BF16 addition. It accumulates gate and up in FP32 and rounds once. Against the dense-BF16-sum and AITER references:
- normalized RMSE is 0.002844-0.002878.
- maximum absolute error is 0.0078125.

The fused pair retains output-only incremental allocation:

| Batch | Packed pair allocation | AITER pair allocation |
|---:|---:|---:|
| 1 | 64 MiB | 192 MiB |
| 4 | 256 MiB | 768 MiB |
| 16 | 1,024 MiB | 3,072 MiB |

Large single-down row-task paths add only small device metadata above output: 9,216-9,728 bytes at batch 4 and 27,648-28,160 bytes at batch 16.

### Final code-object resources

| Retained production kernel | VGPR | SGPR | LDS bytes |
|---|---:|---:|---:|
| Q3_K pair S1 `64x64x32` | 183 | 26 | 10,240 |
| Q3_K pair large `128x64x32` | 206 | 26 | 10,240 |
| IQ2_S pair S1 `64x64x32` | 194 | 54 | 8,192 |
| IQ2_S pair large `128x64x32` | 219 | 54 | 8,192 |
| Q4_K down S1 `64x64x32` | 96 | 22 | 4,096 |
| Q4_K down S2 `128x64x32` | 173 | 30 | 4,096 |
| Q4_K down row task `128x128x32` | 244 | 24 | 8,192 |
| Q5_K down S1 `64x64x32` | 115 | 22 | 4,096 |
| Q5_K down row task `128x128x32` | 256 | 24 | 8,192 |
| IQ2_S down S1 `64x64x32` | 90 | 22 | 4,096 |
| IQ2_S down S2 `128x64x32` | 161 | 30 | 4,096 |
| IQ2_S down row task `128x128x32` | 233 | 24 | 8,192 |

Every listed kernel has zero private segment, zero VGPR spills, zero SGPR spills, and `uses_dynamic_stack: false`. Q5_K row tasks are at the 256-VGPR ceiling, so no additional long-lived prefetch state can be added safely.

### Final representative profiling

Artifacts:

```text
/tmp/rocprof_grouped_bwd_final_gate_q3_b16
/tmp/rocprof_grouped_bwd_final_gate_q3_b16_occ
/tmp/rocprof_grouped_bwd_final_gate_q3_b16_l2
/tmp/rocprof_grouped_bwd_final_gate_q3_b16_lds
/tmp/rocprof_grouped_bwd_final_down_q4_b4
/tmp/rocprof_grouped_bwd_final_down_q4_b4_occ
/tmp/rocprof_grouped_bwd_final_down_q4_b4_l2
/tmp/rocprof_grouped_bwd_final_down_q4_b4_lds
/tmp/rocprof_grouped_bwd_final_down_iq2_b4
```

| Point | Arithmetic kernel | Mean traced latency | OccupancyPercent | L2CacheHit | ALUStalledByLDS |
|---|---|---:|---:|---:|---:|
| Q3_K pair, B16 uniform | N=64 large pair | 47.592 ms | 43.45 | 58.24 | 0.16 |
| Q4_K down, B4 skewed | M-major row task | 11.271 ms | 30.85 | 82.45 | 0.77 |
| IQ2_S down, B4 sparse | M-major row task | 10.675 ms | - | - | - |

Profiler timing is perturbed and is used for decomposition. Event medians in the JSON artifact remain the latency source of record.

The Q4_K and IQ2_S row-task builder averaged approximately 0.004 ms. Setup is negligible relative to the 10-11 ms arithmetic bodies. Low LDS stalls, zero scratch/private memory, and the high Q4_K L2 hit rate rule out task metadata, LDS banking, and spills as the primary remaining gap.

### Dense shared-down control

Shared primitives were checked against dense packed backward after grouped changes:

```text
/tmp/mmq_bwd_grouped_final_dense_control.json
```

Representative controls:
- Q4_K shared down: 5.148 ms packed versus 4.223 ms BF16.
- Q5_K shared down: 5.547 ms packed versus 4.210 ms BF16.

These dense gaps support the conclusion that the remaining grouped down loss is representation/decode-related rather than unique to grouped task construction.

## Why local optimization stops here

The remaining single-down losses are representation-level for the retained execution model:
- AITER consumes BF16 weights directly, while packed down extracts quant fields, reconstructs scales, and forms BF16 WMMA operands from authoritative GGUF bytes on every call.
- The fused pair amortizes cotangent handling, decode structure, and one output write across two projections. Single down has no corresponding second projection to amortize against.
- Large down already uses the measured best `128x128x32` arithmetic geometry and M-major row-task order.
- Small routing already uses measured S1/S2 dispatch.
- IQ2_S already uses width-16 cooperative decode matched to its grid/sign/scale-sharing boundary.
- Q4_K, Q5_K, and IQ2_S row tasks use 244, 256, and 233 VGPRs. More local reuse would extend decode lifetimes and reduce residency or spill, especially for Q5_K.
- The task builder is approximately 0.004 ms, LDS stalls are low, and all retained kernels are spill-free.
- Dense shared-down Q4_K/Q5_K also remain behind BF16.

The arithmetic geometry, full/tail strategy, row ordering, direct tasks, persistence, decoder width, LDS layout, and bounded prefetch neighborhoods have all been measured. Do not restart those sweeps without new profiler evidence.

## Historical baseline and diagnosis

This section is historical. It records why the tile architecture was replaced and should not be read as a description of the current kernel.

Baseline source of record:

```text
/tmp/grouped_mmq_bwd_baseline_full.json
```

The baseline won 0/60 points. Its checkpoint-weighted packed latency was 2.39-6.07x slower than AITER.

Mean baseline packed/AITER logical-throughput ratios:

| Family | Mean ratio | Best | Worst |
|---|---:|---:|---:|
| Gate/up Q3_K | 0.316x | 0.511x | 0.193x |
| Gate/up IQ2_S | 0.281x | 0.458x | 0.166x |
| Down Q4_K | 0.185x | 0.313x | 0.114x |
| Down Q5_K | 0.181x | 0.308x | 0.109x |
| Down IQ2_S | 0.168x | 0.308x | 0.066x |

Representative event-timed baseline points:

| Family | Point | Baseline packed ms | AITER ms | Packed/AITER latency |
|---|---|---:|---:|---:|
| Gate/up Q3_K | B16 uniform | 636.629 | 129.137 | 4.93x |
| Down Q4_K | B4 uniform | 65.633 | 8.893 | 7.38x |
| Down IQ2_S | B4 sparse | 84.170 | 8.672 | 9.71x |

### Baseline architecture

The old generic kernel used:
- 256 threads and eight wave32 waves.
- 128 routed rows per serial row chunk.
- only 16 input-gradient columns per workgroup.
- K=16.
- one FP32 accumulator per wave.
- scalar packed-weight decode.
- 512 bytes of LDS for single and 1,024 bytes for pair.
- two barriers per K iteration.
- expert-owned row chunks processed serially inside each workgroup.

The grid did not grow with batch. Each workgroup repeated 1, 2, or 8 uniform row chunks at batches 1, 4, and 16. Gate/up launched 128 narrow column workgroups per expert and down launched 32.

Baseline code-object resources were low and spill-free: 46-52 VGPRs for singles and 58-89 VGPRs for pairs. This ruled out spill removal as the main opportunity.

Baseline profiling artifacts:

```text
/tmp/rocprof_grouped_bwd_baseline_gate_b16_packed
/tmp/rocprof_grouped_bwd_baseline_gate_b16_aiter
/tmp/rocprof_grouped_bwd_baseline_down_q4_b4_packed
/tmp/rocprof_grouped_bwd_baseline_down_q4_b4_aiter
/tmp/rocprof_grouped_bwd_baseline_down_iq2_b4_sparse_packed
/tmp/rocprof_grouped_bwd_baseline_down_iq2_b4_sparse_aiter
/tmp/rocprof_grouped_bwd_baseline_gate_b16_occ
/tmp/rocprof_grouped_bwd_baseline_gate_b16_l2
/tmp/rocprof_grouped_bwd_baseline_gate_b16_lds
/tmp/rocprof_grouped_bwd_baseline_down_q4_b4_occ
/tmp/rocprof_grouped_bwd_baseline_down_q4_b4_l2
/tmp/rocprof_grouped_bwd_baseline_down_q4_b4_lds
```

| Baseline point | Occupancy | L2 hit rate | ALUStalledByLDS |
|---|---:|---:|---:|
| Q3_K pair, B16 uniform | 3.59% | 79.03% | 0.0010% |
| Q4_K down, B4 uniform | 34.92% | 92.39% | 0.0057% |

Very low LDS stalls and high Q4_K L2 hit rate showed that LDS banking and cache-hit tuning were not first-order problems.

### One-expert diagnostic

Artifact:

```text
/tmp/grouped_mmq_bwd_one_expert.txt
```

At 1,024 rows for one active expert:

| Family | Baseline grouped ms | Dense packed sequence ms | Dense/baseline speedup |
|---|---:|---:|---:|
| Gate/up Q3_K pair | 1.068 | 0.210 for two dense kernels plus add | 5.08x |
| Down Q4_K | 0.656 | 0.158 | 4.15x |
| Down IQ2_S | 0.813 | 0.417 | 1.95x |

At 64 rows, baseline grouped and dense single-projection paths were approximately equal for Q4_K and IQ2_S. This established two durable conclusions:
- grouped metadata and public-operator overhead were not the dominant large-group limitation.
- small and large groups needed separate measured families rather than one global geometry.

### Baseline bottlenecks that motivated the redesign

- Narrow N=16 ownership repeated routing state, packed metadata, scales, barriers, and loop control across 32-128 workgroups per expert.
- The same BF16 cotangent fragment was reloaded across those narrow N workgroups instead of being reused across four or eight N accumulators.
- Scalar decode repeated packed-byte, scale, minimum, high-bit, grid, and sign work that natural width-16 decoders can share.
- Dynamic packed addressing recomputed block and value coordinates inside the inner loop.
- K=16 doubled synchronization rounds relative to K=32.
- Eight waves each owned only one accumulator, leaving little instruction-level parallelism.
- Serial row traversal made each workgroup increasingly long as rows per expert grew.

The retained redesign addressed these architectural causes together rather than applying isolated instruction-count changes to the old narrow body.

## Optimization log

All timings in this section are historical step measurements. They explain retained decisions and rejected alternatives. The final artifact remains the current source of record. GB labels preserve the original experiment-plan identifiers, while the section order follows implementation chronology. Scheduling labeled GB4 was finalized after the Q5_K and IQ2_S arithmetic families existed.

### GB0: benchmark and baseline lock

- full 60-point benchmark.
- real GGUF weights and nonzero BF16 cotangents.
- production AITER heuristic comparison.
- exact single correctness and fused-pair numerical characterization.
- peak-allocation accounting.
- code-object, trace, counter, and one-expert diagnostics.

The baseline artifact must remain unchanged for historical comparison.

### GB1: Q4_K large tile retained

The first retained tiled kernel used `M=128, N=128, K=32`, 128 threads, width-16 decode, paired local-fragment loads, a sixteen-BF16 XOR swizzle, and separate compile-time full/tail bodies.

Artifacts:

```text
/tmp/grouped_mmq_bwd_step1_q4_l1.json
/tmp/grouped_mmq_bwd_step1_q4_matrix.json
/tmp/grouped_bwd_step1_readobj.txt
/tmp/grouped_bwd_step1_disasm.txt
```

| Point | Baseline ms | GB1 ms | Speedup | AITER ms |
|---|---:|---:|---:|---:|
| Q4_K B4 uniform | 65.633 | 10.846 | 6.05x | 9.033 |
| Q4_K B4 boundary | 63.250 | 11.636 | 5.44x | 10.254 |
| Q4_K B16 uniform | 302.946 | 40.270 | 7.52x | 44.022 |
| Q4_K B16 skewed | 293.581 | 40.906 | 7.18x | 35.540 |
| Q4_K B16 sparse | 293.758 | 41.669 | 7.05x | 32.887 |
| Q4_K B16 boundary | 290.333 | 40.225 | 7.22x | 34.768 |

Historical resource result: 244 VGPRs, 22 SGPRs, 8,192-byte LDS, no private segment or spills.

Reason retained: broad cotangent reuse, cooperative decode, K=32, and exact row handling produced the expected 5-8x architectural gain.

### GB2: Q3_K fused pair retained

The initial pair staged two padded `N=128, K=32` Q3_K B tiles, kept one FP32 accumulator set, and executed projection fragment scopes sequentially. It preserved one output allocation and one final BF16 rounding.

Artifacts:

```text
/tmp/grouped_mmq_bwd_step2_q3_pair.json
/tmp/grouped_mmq_bwd_step2_q3_pair_matrix.json
/tmp/grouped_bwd_step2_readobj.txt
/tmp/grouped_bwd_step2_disasm.txt
```

| Point | Baseline ms | GB2 ms | Speedup | AITER ms |
|---|---:|---:|---:|---:|
| Q3_K pair B4 uniform | 128.320 | 11.276 | 11.38x | 24.759 |
| Q3_K pair B4 boundary | 109.649 | 14.362 | 7.63x | 30.619 |
| Q3_K pair B4 sparse | 110.148 | 13.761 | 8.00x | 28.661 |
| Q3_K pair B16 uniform | 636.629 | 45.016 | 14.14x | 129.758 |
| Q3_K pair B16 boundary | 581.575 | 48.455 | 12.00x | 138.254 |
| Q3_K pair B16 sparse | 572.634 | 48.714 | 11.76x | 143.105 |

Historical initial resources: 248 VGPRs, 27 SGPRs, 20,480-byte LDS, no private segment or spills.

Reason retained: fusing two packed projections into one accumulator made the packed path 2.1-2.9x faster than AITER while preserving its allocation advantage.

### GB3: S1 and selective S2 retained

S1 uses `M=64, N=64, K=32`, four waves, one width-16 decode group per loader thread, exact 64-row full bodies, and one bounded tail.

Artifacts:

```text
/tmp/grouped_mmq_bwd_step3_s1.json
/tmp/grouped_bwd_step3_readobj.txt
/tmp/grouped_bwd_step3_disasm.txt
/tmp/grouped_mmq_bwd_step3_s2.json
/tmp/grouped_mmq_bwd_q4_sparse_s2_25.json
/tmp/grouped_mmq_bwd_q4_sparse_s1_25.json
```

| Point | Baseline ms | S1 ms | Speedup | AITER ms |
|---|---:|---:|---:|---:|
| Q3_K pair B1 uniform | 21.539 | 3.614 | 5.96x | 11.010 |
| Q3_K pair B1 skewed | 22.898 | 5.653 | 4.05x | 10.837 |
| Q3_K pair B1 sparse | 24.557 | 5.328 | 4.61x | 8.709 |
| Q3_K pair B1 boundary | 21.370 | 5.303 | 4.03x | 10.891 |
| Q4_K B1 uniform | 11.518 | 3.590 | 3.21x | 3.417 |
| Q4_K B1 skewed | 12.689 | 4.190 | 3.03x | 3.644 |
| Q4_K B1 sparse | 13.554 | 4.503 | 3.01x | 3.246 |
| Q4_K B1 boundary | 11.776 | 4.129 | 2.85x | 3.716 |

Universal S2 was rejected because 64-row uniform groups became half-empty bounded tiles: Q3_K uniform regressed from 3.614 to 6.936 ms and Q4_K uniform from 3.590 to 4.185 ms.

Q4_K S2 was retained only when `rows / num_groups >= 80` below the large-family threshold. A 25-repeat sparse control measured 4.037 ms S2 versus 5.201 ms S1. The retained S2 used 173 VGPRs, 30 SGPRs, 4,096-byte LDS, and no private segment or spills.

Reason retained: exact 64-row geometry is best for uniform batch 1, while selective S2 repays its larger M tile only for the measured larger sparse groups.

### GB5: Q5_K port retained

Q5_K reused the Q4_K framework with width-16 decode, a four-BF16 XOR swizzle, scalar low/high extraction, and bounded packed low/high prefetch. The prefetch improved the initial Q5_K port by 12-24%.

Artifacts:

```text
/tmp/grouped_mmq_bwd_step5_q5.json
/tmp/grouped_mmq_bwd_step5_q5_prefetch.json
/tmp/grouped_mmq_bwd_step5_q5_small_select.json
/tmp/grouped_bwd_q5_readobj.txt
/tmp/grouped_bwd_q5_disasm.txt
```

| Point | Baseline ms | GB5 ms | Speedup | AITER ms |
|---|---:|---:|---:|---:|
| Q5_K B1 uniform | 11.728 | 3.694 | 3.17x | 3.377 |
| Q5_K B1 sparse | 13.713 | 3.913 | 3.50x | 3.221 |
| Q5_K B1 boundary | 11.898 | 3.544 | 3.36x | 3.626 |
| Q5_K B4 uniform | 68.550 | 9.862 | 6.95x | 8.952 |
| Q5_K B4 sparse | 65.608 | 10.797 | 6.08x | 8.625 |
| Q5_K B16 uniform | 315.237 | 37.421 | 8.42x | 43.023 |
| Q5_K B16 sparse | 304.028 | 38.851 | 7.83x | 32.650 |

Historical resources: L1 used 256 VGPRs, 22 SGPRs, and 8,192-byte LDS. S1 used 115 VGPRs, 22 SGPRs, and 4,096-byte LDS. Both were spill-free.

Q5_K S2 was rejected: a 25-repeat sparse control measured 4.065 ms S2 versus 3.913 ms S1. Artifacts: `/tmp/grouped_mmq_bwd_q5_sparse_s2_25.json` and `/tmp/grouped_mmq_bwd_q5_sparse_s1_25.json`.

### GB6: cooperative IQ2_S decode and selective S2 retained

The project-owned decoder reconstructs sixteen aligned values from two grid entries, two sign bytes, one shared scale nibble, and one `d` factor. It does not modify vendored tables or structs.

Artifacts:

```text
/tmp/grouped_mmq_bwd_step6_iq2.json
/tmp/grouped_bwd_iq2_readobj.txt
/tmp/grouped_bwd_iq2_disasm.txt
/tmp/grouped_mmq_bwd_step6_iq2_n64_reuse.json
/tmp/grouped_bwd_iq2_reuse_readobj.txt
```

| Point | Baseline ms | Initial width-16 ms | Speedup | AITER ms |
|---|---:|---:|---:|---:|
| IQ2_S pair B1 uniform | 26.129 | 5.291 | 4.94x | 11.029 |
| IQ2_S pair B1 sparse | 21.435 | 7.894 | 2.72x | 8.702 |
| IQ2_S pair B4 uniform | 135.136 | 13.399 | 10.09x | 25.141 |
| IQ2_S pair B4 sparse | 123.222 | 16.103 | 7.65x | 29.036 |
| IQ2_S pair B16 uniform | 771.007 | 53.998 | 14.28x | 130.139 |
| IQ2_S pair B16 sparse | 768.413 | 56.595 | 13.58x | 145.218 |
| IQ2_S down B1 uniform | 12.698 | 4.281 | 2.97x | 3.401 |
| IQ2_S down B1 sparse | 10.502 | 5.537 | 1.90x | 3.198 |
| IQ2_S down B4 uniform | 55.716 | 10.307 | 5.41x | 8.676 |
| IQ2_S down B4 sparse | 84.170 | 11.101 | 7.58x | 8.765 |
| IQ2_S down B16 uniform | 395.499 | 38.992 | 10.14x | 43.613 |
| IQ2_S down B16 sparse | 483.385 | 40.570 | 11.91x | 35.166 |

The initial L1 kernels were already near the VGPR ceiling: IQ2_S down used 255 VGPRs and IQ2_S pair used 250.

An IQ2_S-only `M=256, N=64, K=32` large-down control was rejected despite being spill-free. B4 uniform regressed from 10.307 to 11.697 ms and B16 uniform from 38.992 to 44.059 ms. Halving N doubled N workgroups, and extra M state did not repay repeated scheduling and cotangent traffic.

Selective S2 `M=128, N=64, K=32` was retained for IQ2_S down at average 80-127 rows. It improved B1 sparse from 5.537 to 3.618 ms. The kernel used 161 VGPRs, 30 SGPRs, 4,096-byte LDS, and no private segment or spills.

### GB4 scheduling: M-major row tasks retained

Scheduling was finalized after Q4_K, Q5_K, and IQ2_S arithmetic families existed. Large down uses device-built 128-row tasks when average group size is at least 128 rows. Pair and batch-1 paths remain serial.

Artifacts:

```text
/tmp/grouped_mmq_bwd_step4_row_tasks_mmajor.json
/tmp/grouped_mmq_bwd_step4_row_tasks_bounded.json
/tmp/grouped_mmq_bwd_step4_row_tasks_bounded_tail.json
/tmp/tasks_final_readobj.txt
/tmp/tasks_final_disasm.txt
```

| Point | Serial ms | M-major row-task ms | Speedup |
|---|---:|---:|---:|
| Q4_K B4 uniform | 10.846 | 10.461 | 1.04x |
| Q4_K B16 uniform | 40.270 | 35.821 | 1.12x |
| Q4_K B16 sparse | 41.669 | 37.143 | 1.12x |
| Q5_K B4 sparse | 10.797 | 10.438 | 1.03x |
| Q5_K B16 uniform | 37.421 | 34.074 | 1.10x |
| Q5_K B16 boundary | 37.849 | 34.725 | 1.09x |
| IQ2_S B4 skewed | 11.493 | 10.096 | 1.14x |
| IQ2_S B16 uniform | 38.992 | 32.806 | 1.19x |
| IQ2_S B16 boundary | 39.785 | 33.569 | 1.19x |

The B4 task workspace was only 9,728 bytes above the 64 MiB output.

Rejected scheduler controls:
- N-major order nearly doubled B16 latency: Q4_K 35.821 to 65.116 ms, Q5_K 34.074 to 62.936 ms, and IQ2_S 32.806 to 62.841 ms. Artifact: `/tmp/grouped_mmq_bwd_step4_row_tasks_nmajor.json`.
- Fixed 1,024-program persistent traversal regressed Q4_K to 38.717 ms and IQ2_S to 36.870 ms. Q5_K also introduced 10 VGPR spills and a 44-byte private segment. Artifact: `/tmp/grouped_mmq_bwd_step4_persistent1024.json`.
- A runtime full/tail branch produced private segments and 2-4 VGPR spills in Q4_K/Q5_K and was rejected.

Reason retained: direct M-major tasks expose sufficient parallel work while preserving packed-weight and cotangent locality. Persistent control state and N-major ordering reduced locality without repaying their control cost.

### GB7: pair geometry and IQ2_S layout retained

IQ2_S down retains a sixteen-BF16 XOR swizzle. IQ2_S pair uses a four-BF16 XOR swizzle. Four-BF16 swizzle regressed down by 20-35% but improved pair by 15-25%.

The retained IQ2_S large pair is `M=128, N=64, K=32`. Reducing N relieved the spill pressure introduced by the pair-specific layout while retaining exact decoder coverage.

| IQ2_S pair point | Previous ms | Retained ms | Speedup |
|---|---:|---:|---:|
| B1 uniform | 5.291 | 4.044 | 1.31x |
| B1 sparse | 7.894 | 5.925 | 1.33x |
| B4 uniform | 13.399 | 10.618 | 1.26x |
| B4 sparse | 16.103 | 13.586 | 1.19x |
| B16 uniform | 53.998 | 44.101 | 1.22x |
| B16 skewed | 57.274 | 46.807 | 1.22x |

Artifacts:

```text
/tmp/grouped_mmq_bwd_step7_iq2_swizzle0.json
/tmp/grouped_mmq_bwd_step7_iq2_swizzle4.json
/tmp/grouped_mmq_bwd_step7_iq2_split_swizzle.json
/tmp/grouped_mmq_bwd_step7_iq2_pair_n64.json
/tmp/iq2_pair_n64_readobj.txt
```

The same N=64 large geometry was retained for Q3_K. It reduced the large kernel from 248 to 206 VGPRs.

| Q3_K pair point | N=128 ms | N=64 ms | Speedup |
|---|---:|---:|---:|
| B4 uniform | 11.276 | 10.560 | 1.07x |
| B4 boundary | 14.362 | 13.892 | 1.03x |
| B16 uniform | 45.016 | 44.630 | 1.01x |
| B16 skewed | 49.202 | 47.229 | 1.04x |
| B16 sparse | 48.714 | 47.461 | 1.03x |

Artifacts: `/tmp/grouped_mmq_bwd_step7_q3_pair_n64.json`, `/tmp/grouped_mmq_bwd_step7_q3_pair_n64_b1_control.json`, and `/tmp/q3_pair_n64_readobj.txt`.

Width-8 IQ2_S decode was rejected and reverted. It duplicated scale work and doubled loader groups. Pair B16 uniform regressed from 44.101 to 50.274 ms and down B16 uniform from 32.959 to 42.949 ms. Artifact: `/tmp/grouped_mmq_bwd_step7_iq2_width8.json`.

### GB7: split full/tail task lists rejected

Separate full-task and tail-task lists improved uniform B16 modestly: Q4_K 35.821 to 34.067 ms and Q5_K 34.074 to 32.007 ms. They regressed every nonuniform B4 point by 6-14% because the second arithmetic launch and tail-list scheduling outweighed removal of row bounds.

Artifact: `/tmp/grouped_mmq_bwd_step7_split_tasks.json`.

The retained single bounded row-task body is the stopping point. Reconsider split lists only if task metadata is already reusable across several backward calls or a device-side launch mechanism removes the extra public launch cost.

## Rejected and closed neighborhoods

The following results should prevent repeated local sweeps without new evidence.

| Candidate | Result | Reason | Artifact/evidence |
|---|---|---|---|
| Universal S2 | Rejected | Half-empty 64-row uniform groups. Regressed Q3_K and Q4_K | `/tmp/grouped_mmq_bwd_step3_s2.json` |
| Q5_K selective S2 | Rejected | 4.065 ms versus 3.913 ms S1 on sparse B1 | `/tmp/grouped_mmq_bwd_q5_sparse_s2_25.json` |
| IQ2_S `M=256, N=64` | Rejected | More M state and twice the N workgroups regressed all B4/B16 points | `/tmp/grouped_mmq_bwd_step6_iq2_n64_reuse.json` |
| N-major row tasks | Rejected | Nearly doubled B16 latency. Harmed row-task locality | `/tmp/grouped_mmq_bwd_step4_row_tasks_nmajor.json` |
| Fixed 1,024-program persistence | Rejected | Slower Q4_K/IQ2_S and spilled Q5_K | `/tmp/grouped_mmq_bwd_step4_persistent1024.json` |
| Runtime full/tail branch | Rejected | Produced private segments and 2-4 VGPR spills | final row-task resource controls |
| Split full/tail task lists | Rejected | Extra launch regressed every nonuniform B4 point by 6-14% | `/tmp/grouped_mmq_bwd_step7_split_tasks.json` |
| IQ2_S width 8 | Rejected | Duplicated scale work and doubled loader groups | `/tmp/grouped_mmq_bwd_step7_iq2_width8.json` |
| IQ2_S no-swizzle | Rejected | Neutral or slower | `/tmp/grouped_mmq_bwd_step7_iq2_swizzle0.json` |
| Four-BF16 IQ2_S down swizzle | Rejected for down | Regressed down 20-35%. Retained only for pair | `/tmp/grouped_mmq_bwd_step7_iq2_swizzle4.json` |

Do not retry wholesale direct-to-VGPR, both-operands direct-to-VGPR, two-LDS pipelines, GSU, split-K, grouped Stream-K, custom LDS barriers, ordinary K=64, or architecture-gated direct-global-to-LDS without new profiler evidence. Related dense/grouped passes already found those mechanisms neutral, slower, or resource-invalid.

Do not use compiler-managed local arrays as prefetch state. Do not keep next-iteration packed fragments live across current WMMA. Cross-iteration packed prefetch previously lost to longer VGPR live ranges.

Do not replace the fused pair with two public outputs and `torch.add`. That changes rounding semantics and triples peak pair output allocation.

Do not retain split full/tail task lists without host-visible routing-distribution evidence or reusable task metadata.

## Durable design rules

These rules remain relevant to future representation-level work:
- Judge complete public-operator and checkpoint-weighted latency, not isolated instruction counts.
- Use real nonzero cotangents and authoritative packed weights.
- Benchmark and profile sequentially.
- Preserve device-resident metadata and sparse active-expert behavior.
- Use exact loader coverage: width-16 decode maps cleanly onto `N=64/128, K=32` tiles.
- Keep decode temporaries bounded and dead before long WMMA phases.
- Use a resource escape ladder: specialize and shorten state first, reduce N from 128 to 64 second, never retain spills.
- Keep pair and down layouts separate when their measured swizzles differ.
- Keep all four down N workgroups for a row task adjacent in M-major order.
- Require zero private segment, zero spills, and no dynamic stack for retained production kernels.
- If dense/grouped primitives are shared, retain a no-change dense benchmark and code-object control.
- Match AITER and hipBLASLt families by configuration/kernel fields, not unstable runtime indices.
- Do not add persistent control state until a new arithmetic or representation design demonstrates a complete-latency win.

## Future representation-level acceptance gates

A future decoded-cache or representation change must answer all of the following:
- What exact bytes are stored per active expert and per projection?
- Is the representation lossless relative to the authoritative GGUF weights?
- Is decode paid once per call, once per layer, or reused across calls?
- Does the workspace preserve inactive-expert sparsity?
- Does the path preserve BF16 public inputs/outputs and fused-pair single rounding?
- Does it preserve output-only pair allocation or clearly justify any additional workspace?
- Is complete latency, including decode and workspace setup, faster across all four routing distributions?
- Does it avoid a permanent approximately 512 MiB BF16 decode per projection or approximately 1 GiB gate/up pair shadow copy?
- Does it improve shared-down dense controls as well as grouped down, confirming that representation reuse addresses the common gap?

Local tile or scheduler work should resume only if these experiments expose a new arithmetic bottleneck or profiler evidence contradicts the current stopping conclusion.

## Validation record

Final retained source and documentation passed:
- `python -m compileall -q bench torch_ggml_ops tests`.
- `python -m pytest -q`: 58 passed, 14 warnings.
- `ty check`.
- `ruff check --extend-select=I`.
- `git diff --check`.

The correctness matrix covers dense and grouped single forward/backward for Q3_K, Q4_K, Q5_K, Q6_K, and IQ2_S. Grouped pair forward and fused pair backward are also parameterized across all five accepted types. Pair forward is bitwise equal to two single projections. The small generic fused-backward test uses deterministic inputs and limits of NRMSE below `5e-5` and maximum absolute error at most `2^-12`. The observed worst case is approximately `1.31e-5` NRMSE and `6.10e-5` maximum absolute error. The separate full-production benchmark retains its established 0.002844-0.002878 NRMSE range because the production specialized kernel's single FP32 accumulation differs from two separately rounded BF16 projections plus add.

Validation commands for future retained changes:

```bash
python -m compileall -q bench torch_ggml_ops tests
python -m pytest -q
ty check
ruff check --extend-select=I
git diff --check
```

Then run the full 60-point matrix, inspect the final gfx1151 code object, and collect representative profiler traces sequentially.

Do not edit `~/transformers-qwen3-moe-fused`. It is legacy/reference-only.

## Artifact index

Final and control artifacts:

```text
/tmp/grouped_mmq_bwd_final_full.json
/tmp/grouped_bwd_final_readobj.txt
/tmp/grouped_bwd_final_disasm.txt
/tmp/mmq_bwd_grouped_final_dense_control.json
```

Historical baseline:

```text
/tmp/grouped_mmq_bwd_baseline_full.json
/tmp/grouped_mmq_bwd_one_expert.txt
```

Retained-step artifacts:

```text
/tmp/grouped_mmq_bwd_step1_q4_l1.json
/tmp/grouped_mmq_bwd_step1_q4_matrix.json
/tmp/grouped_mmq_bwd_step2_q3_pair.json
/tmp/grouped_mmq_bwd_step2_q3_pair_matrix.json
/tmp/grouped_mmq_bwd_step3_s1.json
/tmp/grouped_mmq_bwd_step5_q5_prefetch.json
/tmp/grouped_mmq_bwd_step6_iq2.json
/tmp/grouped_mmq_bwd_step4_row_tasks_bounded.json
/tmp/grouped_mmq_bwd_step4_row_tasks_bounded_tail.json
/tmp/grouped_mmq_bwd_step7_iq2_pair_n64.json
/tmp/grouped_mmq_bwd_step7_q3_pair_n64.json
```

Rejected-control artifacts:

```text
/tmp/grouped_mmq_bwd_step3_s2.json
/tmp/grouped_mmq_bwd_step6_iq2_n64_reuse.json
/tmp/grouped_mmq_bwd_step4_row_tasks_nmajor.json
/tmp/grouped_mmq_bwd_step4_persistent1024.json
/tmp/grouped_mmq_bwd_step7_split_tasks.json
/tmp/grouped_mmq_bwd_step7_iq2_width8.json
/tmp/grouped_mmq_bwd_step7_iq2_swizzle0.json
/tmp/grouped_mmq_bwd_step7_iq2_swizzle4.json
```
