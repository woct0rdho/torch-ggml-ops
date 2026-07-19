# Grouped MMQ backward optimization plan

## Scope

This document covers routed grouped MMQ input-gradient operators on gfx1151:

- `grouped_mmq_grad_input` for one frozen packed expert projection;
- `grouped_mmq_pair_grad_input` for the fused gate/up input gradient.

For a routed expert group, forward computes:

```text
Y[M, N] = X[M, K] @ W[N, K].T
```

The backward operator covered here computes:

```text
dX[M, K] = dY[M, N] @ W[N, K]
```

The public cotangent and result are BF16. The authoritative weights remain packed GGUF tensors. Unlike grouped forward, backward does not quantize its BF16 input to Q8_1.

Dense MMQ backward is documented in `docs/mmq_bwd_optimization.md`. Grouped forward is documented separately in `docs/grouped_mmq_fwd_optimization.md`.

## Current status

Benchmark and profiling infrastructure is complete. The grouped-backward kernel has not yet received a production optimization pass.

The baseline source of record is:

```text
/tmp/grouped_mmq_bwd_baseline_full.json
```

The current packed implementation loses all 60 measured production points against BF16 AITER GMM. Depending on batch and routing distribution, the checkpoint-weighted packed estimate is approximately 2.39-6.07x slower than AITER.

This is not a spill-removal problem:

- production kernels have no private segment, VGPR spills, SGPR spills, scratch traffic, or dynamic stack;
- single kernels use only 46-52 VGPRs and 512 bytes of LDS;
- paired kernels use 58-89 VGPRs and 1,024 bytes of LDS;
- measured LDS stalls are negligible.

The first optimization priority is therefore a tile-architecture replacement: transplant the retained dense-backward cooperative decode and WMMA organization into grouped, fixed-production-shape kernels while preserving device-resident routing and the fused pair output.

The plan below has been revised using the complete dense-forward, dense-backward, and grouped-forward experiment histories. It now emphasizes separate small- and large-group families, exact loader utilization, retained type-specific LDS/decode choices, early full/tail specialization, resource escape paths, scheduler tile ordering, and a project-owned packed-group IQ2_S decoder.

## Production contract

The optimized implementation must preserve:

- BF16 public cotangents;
- authoritative packed GGUF expert weights;
- direct BF16 `grad_input` output;
- no transient logical dense base-weight matrix in the ordinary path;
- the current stream;
- sparse routing, including no arithmetic workgroups for inactive experts;
- the fused pair contract: gate and up accumulate into one FP32 accumulator and round once to BF16.

The metadata ABI remains:

- `expert_indices`: contiguous CUDA `torch.int64`, shape `[G]`;
- `expert_offsets`: contiguous CUDA `torch.int32`, shape `[G]`;
- `expert_offsets[-1] = R`;
- `G <= 256`.

Production code must not add `.item()`, CPU descriptors, device-to-host metadata copies, or metadata synchronization.

## Production checkpoint matrix

The benchmark uses real tensors from `Qwen3.6-35B-A3B-APEX-I-Mini.gguf`.

| Case | Logical expert weight | Backward GEMM | GGUF type | Layers | Operator |
|---|---:|---:|---|---:|---|
| Gate/up outer | `512 x 2048` | `M x 512` by `512 x 2048` | Q3_K | 20 | fused pair |
| Gate/up middle | `512 x 2048` | `M x 512` by `512 x 2048` | IQ2_S | 20 | fused pair |
| Down middle | `2048 x 512` | `M x 2048` by `2048 x 512` | IQ2_S | 20 | single |
| Down outer main | `2048 x 512` | `M x 2048` by `2048 x 512` | Q4_K | 18 | single |
| Down outer edge | `2048 x 512` | `M x 2048` by `2048 x 512` | Q5_K | 2 | single |

The sequence length is 2,048 and top-k is 8.

| Physical batch | Routed rows | Uniform rows per expert |
|---:|---:|---:|
| 1 | 16,384 | 64 |
| 4 | 65,536 | 256 |
| 16 | 262,144 | 1,024 |

## Benchmark infrastructure

`bench/benchmark_grouped_mmq_bwd.py` follows the existing dense and grouped-forward benchmark conventions.

Shared grouped benchmark definitions and helpers are in:

```text
bench/grouped_mmq_benchmark_common.py
```

`bench/benchmark_grouped_mmq_fwd.py` now imports the same production cases, route distributions, metadata construction, timing, memory, and correctness helpers. This keeps forward and backward workloads aligned.

The backward benchmark records:

- complete public packed-operator latency;
- logical throughput;
- incremental peak allocation and reservation growth;
- AITER configuration selected by the production heuristic;
- routing metadata and distribution statistics;
- per-case checkpoint-weighted latency estimates;
- packed grouped versus per-expert dense packed correctness;
- packed grouped versus BF16 AITER correctness.

The full baseline command was:

```bash
python bench/benchmark_grouped_mmq_bwd.py \
  --warmup 3 \
  --repeats 9 \
  --correctness-rows 256 \
  --output /tmp/grouped_mmq_bwd_baseline_full.json
```

GPU benchmarks and profiler runs were sequential. No two GPU measurements were run concurrently.

## Routing distributions

The benchmark covers four deterministic routing distributions.

- `uniform`: all 256 experts have equal row counts.
- `skewed`: all 256 experts are active with deterministic nonuniform sizes.
- `sparse`: 192, 224, and 240 active experts for batches 1, 4, and 16.
- `boundary`: explicitly includes sizes 1, 15, 16, 17, 63, 64, 65, 127, 128, and 129.

Representative ranges are:

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

These counts are relevant when comparing serial expert-owned row traversal with device-built row-task scheduling.

## BF16 AITER reference

The production comparison is AITER Triton `gmm`, not `torch.matmul`.

The benchmark loads `_gmm_config` from:

```text
~/test_no_unsloth/fast_moe_lora.py
```

The timed AITER path starts with independently dequantized BF16 experts. Dequantization and active-expert `index_select` are setup costs and are not included in the GMM timing.

For paired gate/up backward, the AITER reference includes:

- one GMM for gate;
- one GMM for up;
- `torch.add` of the two BF16 results.

The production heuristic selects:

| Backward shape | M tile | N tile | K tile | Threads | Persistent programs |
|---|---:|---:|---:|---:|---:|
| Gate/up: `K=512, N=2048` | 128 | 128 | 64 | 256 | 256 |
| Down: `K=2048, N=512` | 64 | 128 | 64 | 256 | 256 |

AITER is the production reference, not a performance ceiling.

## Full baseline results

The packed path wins 0 of 60 points.

The mean packed/AITER logical-throughput ratio across all batches and distributions is:

| Case | Mean packed/AITER throughput | Best point | Worst point |
|---|---:|---:|---:|
| Gate/up Q3_K | 0.316x | 0.511x | 0.193x |
| Gate/up IQ2_S | 0.281x | 0.458x | 0.166x |
| Down Q4_K | 0.185x | 0.313x | 0.114x |
| Down Q5_K | 0.181x | 0.308x | 0.109x |
| Down IQ2_S | 0.168x | 0.308x | 0.066x |

The following table averages the four route distributions at each batch. `Packed/AITER latency` above 1.0 means packed is slower.

| Case | Batch | Packed ms | AITER ms | Packed/AITER latency |
|---|---:|---:|---:|---:|
| Gate/up Q3_K | 1 | 22.591 | 10.348 | 2.18x |
| Gate/up Q3_K | 4 | 113.928 | 28.612 | 3.98x |
| Gate/up Q3_K | 16 | 591.288 | 137.659 | 4.30x |
| Gate/up IQ2_S | 1 | 24.026 | 10.353 | 2.32x |
| Gate/up IQ2_S | 4 | 123.674 | 28.660 | 4.32x |
| Gate/up IQ2_S | 16 | 761.765 | 137.030 | 5.56x |
| Down Q4_K | 1 | 12.384 | 3.518 | 3.52x |
| Down Q4_K | 4 | 63.533 | 9.143 | 6.95x |
| Down Q4_K | 16 | 295.155 | 36.931 | 7.99x |
| Down Q5_K | 1 | 12.516 | 3.491 | 3.59x |
| Down Q5_K | 4 | 65.784 | 9.310 | 7.07x |
| Down Q5_K | 16 | 306.321 | 36.628 | 8.36x |
| Down IQ2_S | 1 | 12.092 | 3.481 | 3.47x |
| Down IQ2_S | 4 | 72.611 | 9.310 | 7.80x |
| Down IQ2_S | 16 | 449.084 | 36.915 | 12.17x |

Representative event-timed points are:

| Case | Batch/distribution | Packed ms | AITER ms | Packed/AITER latency |
|---|---|---:|---:|---:|
| Gate/up Q3_K | B16 uniform | 636.629 | 129.137 | 4.93x |
| Down Q4_K | B4 uniform | 65.633 | 8.893 | 7.38x |
| Down IQ2_S | B4 sparse | 84.170 | 8.672 | 9.71x |

The gap grows with rows per expert, which is consistent with the current kernel serially repeating a scalar-decode, barrier-heavy K loop for each 128-row chunk.

## Checkpoint-weighted estimate

The estimate multiplies each case by its checkpoint layer count and sums all five cases. It covers the frozen packed expert input-gradient operators in one model backward, not the complete optimizer step.

| Batch | Distribution | Packed ms | AITER ms | Packed/AITER latency |
|---:|---|---:|---:|---:|
| 1 | uniform | 1,438.087 | 575.924 | 2.50x |
| 1 | skewed | 1,458.063 | 579.261 | 2.52x |
| 1 | sparse | 1,401.272 | 478.864 | 2.93x |
| 1 | boundary | 1,391.062 | 581.815 | 2.39x |
| 4 | uniform | 7,701.936 | 1,346.649 | 5.72x |
| 4 | skewed | 7,215.316 | 1,599.806 | 4.51x |
| 4 | sparse | 7,631.986 | 1,500.726 | 5.09x |
| 4 | boundary | 7,368.434 | 1,612.098 | 4.57x |
| 16 | uniform | 42,146.187 | 6,947.471 | 6.07x |
| 16 | skewed | 41,935.492 | 7,009.803 | 5.98x |
| 16 | sparse | 42,384.350 | 7,033.386 | 6.03x |
| 16 | boundary | 41,406.702 | 6,889.803 | 6.01x |

Across batches, the packed weighted contribution is approximately:

- gate/up Q3_K: 28-32%;
- gate/up IQ2_S: 33-36%;
- down IQ2_S: 17-21%;
- down Q4_K: 13-16%;
- down Q5_K: about 1.5-1.8%.

Common tile work affects every case. After that common work, IQ2_S and fused gate/up are the highest-priority specializations.

## Correctness and memory

Every single-projection point is bitwise identical to both:

- concatenated per-expert dense packed MMQ backward;
- BF16 AITER GMM using the dequantized version of the same weights.

For all single cases, the correctness prefix has zero differing BF16 elements, zero maximum absolute error, and zero normalized RMSE.

The paired packed kernel intentionally differs from two separately rounded BF16 projections followed by BF16 addition. It accumulates both projections in FP32 and rounds once. Across the measured correctness prefixes:

- normalized RMSE versus the dense-BF16-sum and AITER references is `0.002844-0.002878`;
- maximum absolute error is `0.0078125`;
- both references produce the same error metrics because each single packed projection is bitwise equal to AITER.

The fused packed pair has a material memory advantage. Its incremental peak allocation is one output tensor. The AITER pair allocates two GMM outputs plus the sum:

| Batch | Packed pair allocation | AITER pair allocation |
|---:|---:|---:|
| 1 | 64 MiB | 192 MiB |
| 4 | 256 MiB | 768 MiB |
| 16 | 1,024 MiB | 3,072 MiB |

This output-only allocation and single-rounding FP32 accumulation are production properties to preserve, not incidental baseline behavior.

## Current packed kernel architecture

The implementation is in `csrc/ck/grouped_mmq_backward.cuh`.

The current geometry is:

- 256 threads, or eight wave32 waves;
- 128 routed rows per row chunk, with 16 rows owned by each wave;
- 16 input-gradient columns per workgroup;
- K iteration of 16 output-feature values;
- one FP32 WMMA accumulator per wave for the single kernel;
- one shared FP32 accumulator per wave for the fused pair result;
- one scalar packed-weight decode per participating thread;
- one 16 x 16 BF16 decoded-weight tile in LDS for single;
- two 16 x 16 decoded-weight tiles in LDS for pair;
- two workgroup barriers per K iteration;
- expert-owned row chunks processed serially inside each workgroup.

The launch grid is:

```text
grid.x = in_features / 16
grid.y = active expert groups
```

Therefore an all-expert production layer launches:

| Projection | Input-gradient columns | Workgroups per expert | Total workgroups |
|---|---:|---:|---:|
| Gate/up | 2,048 | 128 | 32,768 |
| Down | 512 | 32 | 8,192 |

The workgroup count does not grow with batch. Instead, each workgroup serially loops over 1, 2, or 8 uniform 128-row chunks at batches 1, 4, and 16.

The pair kernel decodes two packed weight tiles and issues two WMMA operations into the same accumulator for every K step. This avoids materializing two BF16 outputs, but the current scalar decode and narrow N tile leave substantial arithmetic and synchronization overhead.

## Code-object resources and ISA

The extracted gfx1151 code object reports:

| Kernel | GGUF type | VGPRs | SGPRs | Fixed LDS | Private segment | Spills |
|---|---|---:|---:|---:|---:|---:|
| Single | Q3_K | 52 | 24 | 512 B | 0 | 0 |
| Single | Q4_K | 46 | 24 | 512 B | 0 | 0 |
| Single | Q5_K | 48 | 24 | 512 B | 0 | 0 |
| Single | IQ2_S | 49 | 24 | 512 B | 0 | 0 |
| Pair | Q3_K | 58 | 42 | 1,024 B | 0 | 0 |
| Pair | Q4_K | 80 | 42 | 1,024 B | 0 | 0 |
| Pair | Q5_K | 89 | 42 | 1,024 B | 0 | 0 |
| Pair | IQ2_S | 70 | 41 | 1,024 B | 0 | 0 |

All listed kernels have zero VGPR spills, zero SGPR spills, no scratch instructions, and no dynamic stack.

Static final-ISA inspection found:

| Kernel | GGUF type | Instructions | WMMA instructions in loop body | Barriers in loop body |
|---|---|---:|---:|---:|
| Single | Q3_K | 594 | 1 | 2 |
| Single | Q4_K | 598 | 1 | 2 |
| Single | Q5_K | 609 | 1 | 2 |
| Single | IQ2_S | 576 | 1 | 2 |
| Pair | Q3_K | 897 | 2 | 2 |
| Pair | IQ2_S | 868 | 2 | 2 |

The low register and LDS footprint leaves room for materially broader N tiles, multiple M accumulators per wave, and cooperative multi-value decode. Reducing register use is not an initial objective.

## Profiling results

Representative kernel-trace artifacts are:

```text
/tmp/rocprof_grouped_bwd_baseline_gate_b16_packed
/tmp/rocprof_grouped_bwd_baseline_gate_b16_aiter
/tmp/rocprof_grouped_bwd_baseline_down_q4_b4_packed
/tmp/rocprof_grouped_bwd_baseline_down_q4_b4_aiter
/tmp/rocprof_grouped_bwd_baseline_down_iq2_b4_sparse_packed
/tmp/rocprof_grouped_bwd_baseline_down_iq2_b4_sparse_aiter
```

Named arithmetic-kernel durations under kernel tracing were:

| Case | Packed arithmetic | AITER arithmetic |
|---|---:|---:|
| Gate/up Q3_K B16 uniform | 640.341 ms, one fused kernel | 122.758 ms, two GMM kernels, plus 15.460 ms add |
| Down Q4_K B4 uniform | 66.723 ms | 9.145 ms |
| Down IQ2_S B4 sparse | 74.055 ms | 9.023 ms |

Profiler timing is perturbed and is used for decomposition. CUDA-event medians in the JSON artifact remain the latency source of record.

Counter runs were collected one counter at a time. The artifacts are:

```text
/tmp/rocprof_grouped_bwd_baseline_gate_b16_occ
/tmp/rocprof_grouped_bwd_baseline_gate_b16_l2
/tmp/rocprof_grouped_bwd_baseline_gate_b16_lds
/tmp/rocprof_grouped_bwd_baseline_down_q4_b4_occ
/tmp/rocprof_grouped_bwd_baseline_down_q4_b4_l2
/tmp/rocprof_grouped_bwd_baseline_down_q4_b4_lds
```

| Case | Occupancy | L2 hit rate | `ALUStalledByLDS` |
|---|---:|---:|---:|
| Gate/up Q3_K pair, B16 uniform | 3.59% | 79.03% | 0.0010% |
| Down Q4_K, B4 uniform | 34.92% | 92.39% | 0.0057% |

The very low LDS-stall values rule out LDS banking as the first-order baseline problem. The high Q4_K L2 hit rate also argues against beginning with cache-hit tuning. Occupancy counters for these long kernels should be treated qualitatively, but the gate pair's very low measured occupancy reinforces the need to replace the current long eight-wave scalar-decode loop rather than add more control state to it.

## One-expert geometry diagnostic

A controlled one-active-expert benchmark compared the grouped kernel with the retained dense packed backward sequence using the same real expert weights.

At 1,024 rows per expert:

| Case | Current grouped ms | Dense packed sequence ms | Dense/current speedup |
|---|---:|---:|---:|
| Gate/up Q3_K pair | 1.068 | 0.210 for two dense kernels plus add | 5.08x |
| Down Q4_K | 0.656 | 0.158 | 4.15x |
| Down IQ2_S | 0.813 | 0.417 | 1.95x |

At 64 rows, the current and dense single-projection paths are approximately equal for Q4_K and IQ2_S. This supports measured dispatch rather than replacing the sparse small-group path unconditionally.

The 1,024-row result is the key diagnostic: grouped metadata and public operator overhead are not the dominant limitation. The retained dense tile core is multiple times faster on the same packed representation and same expert data.

Diagnostic artifact:

```text
/tmp/grouped_mmq_bwd_one_expert.txt
```

## Bottleneck diagnosis

The baseline gap is primarily architectural.

### Narrow N ownership

A workgroup computes only 16 input-gradient columns. Packed block metadata, scale formation, address arithmetic, barriers, and loop control are repeated across 128 gate/up column workgroups or 32 down column workgroups per expert.

The retained dense Q3_K/Q4_K/Q5_K backward path computes 128 columns per workgroup.

### Repeated cotangent loads across narrow N workgroups

Every current `N=16` workgroup reloads the same BF16 cotangent fragment for its own input-column tile. Gate/up repeats those cotangent loads across 128 workgroups per expert, while down repeats them across 32.

A broad `N=128` workgroup loads each A fragment once and reuses it across eight N accumulators. At fixed M coverage, that reduces duplicated cotangent traffic by approximately 8x. Combined with moving from K=16 to K=32, it also reduces workgroup-level barrier rounds across the complete N dimension by approximately 16x.

This A-fragment reuse is a separate gain from packed-weight decode. The number of decoded B values per logical output tile does not disappear, but the broad tile amortizes cotangent loads, routing metadata, loop control, and synchronization over much more WMMA work.

### Scalar decode

The current grouped kernel calls a scalar decoder for each decoded BF16 weight. The retained dense kernel uses cooperative multi-value decoders that share packed bytes, scales, minima, and high-bit extraction across contiguous values.

IQ2_S has an especially strong sharing opportunity. Eight adjacent values use one grid lookup and one sign byte, while sixteen adjacent values share one scale nibble. The scalar baseline repeats those operations per BF16 value.

### Dynamic packed addressing

The scalar body recomputes packed-row offsets, `input_column / 256`, and `input_column % 256` inside the decode loop. Exact `N=64` or `N=128` tiles begin at known sub-block boundaries, so a fixed-shape kernel can compute one packed block index and one value base per tile, then advance packed-row pointers affinely through K.

### Small K iteration

The current K iteration is 16. Gate/up executes 32 K iterations per row chunk and down executes 128. Every iteration includes two workgroup barriers.

The retained dense production path uses K=32 for Q3_K/Q4_K/Q5_K, halves the number of synchronization rounds, and overlaps more decoded-weight reuse with WMMA work.

### Eight-wave workgroups with one accumulator per wave

The current 256-thread workgroup uses eight waves but gives each wave only one 16 x 16 accumulator. Resource use is low, yet there is little instruction-level parallelism around decode, LDS reads, and WMMA.

The retained dense production path uses four waves, two M tiles per wave, and eight N tiles, producing a 128 x 128 output tile.

### Serial row-chunk traversal

At batch 16, each current workgroup repeats the complete decode and K traversal for eight 128-row chunks. Serial ownership preserves sparse behavior but can produce very long workgroups and limits scheduling flexibility.

This should be revisited only after the tile core is efficient. A scheduler cannot repair scalar decode and a 16-column tile.

### IQ2_S representation cost

IQ2_S remains the hardest type. Its grid lookup, metadata interpretation, scale formation, and BF16 construction are paid repeatedly for every row tile. The one-expert diagnostic shows that the existing dense IQ2_S core is itself only about 1.95x faster than grouped at 1,024 rows, while production grouped IQ2_S down can be more than 12x slower than AITER.

This means local grouped tile work is necessary but may not be sufficient to close the final IQ2_S gap.

## Lessons transferred from previous MMQ optimization passes

The dense-forward, dense-backward, and grouped-forward projects provide stronger guidance than a generic GEMM analogy. The grouped-backward plan should reuse their retained mechanisms and avoid their closed neighborhoods.

### Reuse the dense-backward core, not only its headline tile

The dense-backward redesign produced 4-7x gains by changing several coupled mechanisms together:

- four wave32 waves rather than eight low-work waves;
- multiple M and N accumulators per wave;
- cooperative pair, quad, and sixteen-value decode;
- K=32 for ordinary Q3_K/Q4_K/Q5_K;
- paired local-fragment prefetch;
- exact full-tile bodies;
- shape-specific packed prefetch and LDS layouts.

The grouped kernel should not test `N=128` while retaining scalar decode, K=16, generic addressing, and bounded row handling, then conclude that the dense geometry does not transfer. The meaningful ablation is a complete dense-style tile core with grouped routing around it.

To avoid dense and grouped decoder drift, first extract or share project-owned primitives such as:

- `backward_shared_b_tile`;
- sixteen-value Q3_K/Q4_K/Q5_K decode helpers;
- fragment coordinate and vector-load helpers;
- selected swizzle/padding address functions.

A no-performance-change dense control must be benchmarked if this requires moving code from `csrc/ck/mmq_backward.cuh` into a shared project-owned header. Source refactoring is not allowed to silently change the retained dense dispatch or ISA.

### Use exact loader coverage as a design rule

Previous Q6_K tuning showed that the best tiles often assign an exact, useful decode group to every loader thread.

For a sixteen-value decoder:

| B tile | BF16 values | Decode groups | Groups per 128-thread workgroup |
|---|---:|---:|---:|
| `N=64, K=32` | 2,048 | 128 | 1 per thread |
| `N=128, K=32` | 4,096 | 256 | 2 per thread |
| Pair `N=64, K=32` | 4,096 | 256 | 2 per thread |
| Pair `N=128, K=32` | 8,192 | 512 | 4 per thread |

These mappings avoid idle loader threads and irregular loops. Non-divisor N tiles such as 96 columns should not be tested as nominal full-tile candidates: prior dense Q6_K N=3/N=7 experiments showed that irregular final tiles and universal bounds handling can erase an apparently favorable accumulator count.

### Carry over type- and shape-specific dense choices

The production grouped-backward shapes correspond to already measured dense-backward families.

| Grouped-backward case | Matching dense family | Initial retained mechanisms to transfer |
|---|---|---|
| Gate/up Q3_K, `K=512, N=2048` | Narrow Q3_K backward | sixteen-value decode, K=32, paired local-fragment prefetch, eight-BF16 row padding, no packed-byte prefetch for the misaligned 110-byte Q3_K block |
| Down Q4_K, `K=2048, N=512` | Shared-down Q4_K backward | sixteen-value decode, K=32, two-row packed-byte prefetch, 16-BF16-chunk XOR swizzle |
| Down Q5_K, `K=2048, N=512` | Shared-down Q5_K backward | sixteen-value decode, K=32, packed-byte prefetch, four-BF16-chunk XOR swizzle, scalar quant extraction rather than the narrow-only packed extraction |

These are starting points, not universal truths. Final grouped metadata and row traversal can change resource pressure, so every transferred choice still needs a grouped A/B control.

### Preserve bounded decode lifetimes

TensileLite, CK, FeatherOps, and the retained dense kernel all point to the same ordering:

- load a bounded packed fragment into fixed scalar or vector VGPR state;
- reconstruct scales, minima, signs, and quant values;
- commit one decoded B tile to LDS;
- end decode-temporary lifetimes;
- load A and B fragments;
- issue WMMA while accumulator state is live.

Do not keep the next K iteration's packed fragments live across the current WMMA phase. Dense Q3_K cross-iteration packed prefetch regressed because the longer VGPR live range cost more than the overlap saved. Compiler-managed local arrays are also prohibited as prefetch state because earlier projects showed that they can lower unexpectedly into LDS or scratch.

For the pair kernel, decode or stage both B operands but keep the first and second A-fragment scopes separate. The pair needs one accumulator set, not two. Avoid keeping both projection A fragments and both B fragment sets live simultaneously.

### Full and tail specialization should arrive early

Grouped-forward G4 produced 1.25-1.78x gains by removing row decomposition and predicates from exact row tiles. Grouped-backward cotangents are simpler contiguous BF16 data, so the corresponding specialization should be at least as direct:

- full M tiles use unmasked BF16 A loads and output stores;
- tail tiles compute `valid_rows` once and use contiguous row spans;
- no per-element row division, remainder, or source-row reconstruction appears in the full body.

This also creates a new batch-1 opportunity. An `M=64` small-group tile makes uniform 64-row expert groups exact full tiles instead of running half of an eight-wave `M=128` block inactive.

### Use separate small-group and large-group geometry

Dense backward deliberately keeps narrow N tiles for rows `<=128`, while the broad 128x128 tile is selected only for larger exact rows. Dense forward likewise retained a special small-row Q6_K tile and rejected a global tile reduction.

Grouped backward should therefore use measured row-count dispatch rather than replacing every route group with one geometry.

Initial small-group candidates are:

| Candidate | Threads | M | N | K | Accumulators per wave | Purpose |
|---|---:|---:|---:|---:|---:|---|
| Current control | 256 | 128 | 16 | 16 | 1 | preserve the known sparse baseline |
| S1 | 128 | 64 | 64 | 32 | 4 | exact batch-1 uniform rows and exact one-group-per-thread decode |
| S2 | 128 | 128 | 64 | 32 | 8 | cover most batch-1 groups in one row tile with moderate N reuse |
| S3 | 128 | 64 | 128 | 32 | 8 | maximize A reuse while retaining a 64-row full path |

The large-group candidate remains `M=128, N=128, K=32` with 16 accumulators per wave.

The host can select a family from `rows`, `num_groups`, and tensor shapes without reading device metadata. As in grouped forward, the host-visible average is only a dispatch hint; every selected kernel must still handle skewed and boundary groups correctly.

### Keep a resource escape ladder

The retained dense shared-down Q4_K/Q5_K kernels are already close to the VGPR limit. Adding grouped metadata and a serial row loop could make a literal 128x128 transplant spill even though the current narrow kernel is low-resource.

The response to a spill must be ordered:

- specialize all production dimensions and remove dynamic bounds/address state;
- shorten metadata and decoder lifetimes;
- move row ownership into compact device tasks if that reduces live serial-loop state and also improves complete latency;
- reduce N from 128 to the exact divisor 64 while keeping K=32 and multi-value decode;
- only then consider a different M tile.

Do not jump directly to an irregular N tile or restore eight waves. Grouped-forward G9/G10 and dense Q6_K irregular-N experiments already closed those shortcuts.

### Adapt the grouped-forward scheduling result rather than copying it

Grouped forward retained row-task descriptors only for large gate/up groups and rejected them for down because forward down already had 32 output tiles per expert.

Backward with `N=128` has a different balance:

- gate/up has 16 N tiles per expert;
- down has only 4 N tiles per expert.

Therefore the forward conclusion is not directly reusable. Backward down may benefit more from row tasks than forward down, especially at batch 16, while the fused pair cannot amortize descriptor setup across two arithmetic launches because it is already one kernel.

Start with serial `(expert, N tile)` ownership because it keeps repeated packed tiles close in time and requires no setup. Then compare:

- serial row chunks within one workgroup;
- nonpersistent device row tasks;
- a fixed-program grid-stride traversal over device tasks.

For descriptor ordering, test both M-major and N-major traversal. The old dense `GROUP_M` experiment showed that tile ordering can change latency by 35-58%, while the final 128-row geometry preferred a different order. An N-major order can keep the same expert/N packed tile hot across consecutive row chunks; an M-major order can improve cotangent locality and balance. This is a measured geometry choice, not a universal scheduler rule.

### Use fixed affine pointers and exact K trip counts

Grouped-forward G6/G7 showed that fixed-trip cleanup remains useful after the main tile is efficient, particularly for IQ2_S.

For backward:

- gate/up has 16 K=32 iterations;
- down has 64 K=32 iterations;
- packed-row bytes, expert bytes, input block index, and local value base are compile-time or tile-invariant;
- packed-row and cotangent pointers should advance rather than be remultiplied.

Keep both loops rolled initially. Test modest two- or four-iteration unrolling only after pointer form is retained. Full unrolling of 64 down iterations is not justified, and deep cross-iteration prefetch is already a measured non-priority.

### Build an IQ2_S decoder around its natural packed groups

The project-owned scalar IQ2_S decoder currently performs one grid lookup per output value. The packed format naturally exposes much more reuse:

- eight adjacent values share one 64-bit `iq2s_grid` entry and one sign byte;
- sixteen adjacent values share one scale nibble and `d` scale;
- two grid indices and two sign bytes therefore produce one aligned 16-value decoder group.

The vendored forward loader already demonstrates the useful mechanism without requiring vendor edits: it loads the two 32-bit halves of a grid entry, expands sign bits with packed byte operations such as `__vcmpne4`, and applies signs with packed subtraction. A project-owned backward decoder should adapt that mechanism in `csrc/ck/*`, then convert the resulting signed magnitude bytes to BF16 under one shared scale.

Evaluate an eight-value decoder first as a low-live-state control and a sixteen-value decoder as the expected winner. Inspect final ISA to verify that grid loads, sign expansion, and scale formation are not duplicated per BF16 output.

This IQ2_S decoder is a higher-priority idea than another generic tile sweep because IQ2_S contributes approximately half of weighted baseline latency and its scalar representation is visibly redundant.

### Treat M=256, N=64 only as an IQ2_S decode-reuse experiment

Dense shared-down Q4_K/Q5_K already rejected the 4x4-per-wave equivalent: doubling M reuse while halving N increased cotangent traffic and live A fragments enough to regress materially.

IQ2_S has a different decode cost. After a natural-group IQ2_S decoder and the standard 128x128/128x64 candidates are measured, one controlled `M=256, N=64, K=32` experiment is justified for large groups because it keeps 16 accumulators per wave while doubling decoded-B reuse across M. It must not be generalized to Q4_K/Q5_K, and it should be reverted immediately if the extra A traffic dominates as it did in dense shared down.

### Do not reopen closed local mechanisms

Previous projects already provide negative evidence against:

- custom LDS-only barriers;
- removing a neutral post-write barrier as a primary optimization;
- K=64 for ordinary Q3_K/Q4_K/Q5_K;
- a second LDS pipeline without a fresh exposed-wait argument;
- complete decoded-weight LDS caching;
- wholesale direct-to-VGPR or direct-global-to-LDS paths;
- split-K, GSU, or Stream-K that duplicates packed decode or requires reduction;
- larger eight-wave workgroups;
- broad non-shape-specific tile policies.

The grouped-backward pass should spend its measurement budget on cooperative decode, A reuse, exact full/tail paths, resource-safe dispatch, and IQ2_S packed-group reconstruction.

## Revised ranked hypotheses

| Rank | Hypothesis | Evidence from previous kernels | Expected scope | Main risk |
|---:|---|---|---|---|
| 1 | Four-wave multi-M/multi-N tile with width-16 decode and K=32 | Dense backward improved 4-7x; one-expert grouped diagnostic is 2-5x behind the dense core | Q3_K/Q4_K/Q5_K and scaffolding for IQ2_S | grouped metadata pushes the broad tile into spills |
| 2 | Reuse each BF16 cotangent fragment across 4-8 N accumulators | Current N=16 reloads A across 32-128 column workgroups; retained dense tile uses broad N | every type, especially gate/up pair | broader accumulators reduce residency |
| 3 | Exact full 64-row/128-row bodies plus one bounded tail body | Grouped-forward G4 gained 1.25-1.78x | every route distribution | code growth raises VGPR allocation |
| 4 | IQ2_S width-8/width-16 packed-group decoder using packed sign operations | Forward IQ2_S loader already shares grids/signs; scalar backward repeats grid work per value | roughly 50-58% of weighted baseline latency | table lookup/conversion state raises VGPRs or changes rounding |
| 5 | Separate small-group and large-group dispatch | Dense backward and dense forward both rejected one global geometry; grouped forward retained batch-sensitive scheduling | B1 versus B4/B16 | too many variants for small aggregate gain |
| 6 | Shape-specific serial/task/persistent ordering | Dense `GROUP_M` changed latency by up to 1.58x; grouped-forward descriptors were shape-specific | large and irregular groups | setup/control state outweighs balance |
| 7 | Fixed affine pointers and modest K-loop unrolling | Grouped-forward G6/G7 gave 1-21% type-dependent gains | especially IQ2_S and fixed down | longer live ranges and code size |
| 8 | IQ2_S-only `M=256, N=64` decode reuse | possible representation-specific exception to the rejected dense shared-down 4x4 tile | large IQ2_S down groups only | repeated A traffic dominates again |

Ranks 1-4 are structural and should receive the first implementation effort. Ranks 6-8 are conditional follow-ups and must not delay the arithmetic-core replacement.

## Target tile architecture

The initial candidates should follow the retained dense-backward mechanism while preserving a separate small-group family.

| Family | M tile | N tile | K iteration | Threads | M tiles per wave | N tiles | Initial scope |
|---|---:|---:|---:|---:|---:|---:|---|
| Current control | 128 | 16 | 16 | 256 | 1 | 1 | sparse/small baseline |
| Small S1 | 64 | 64 | 32 | 128 | 1 | 4 | exact 64-row groups |
| Small S2 | 128 | 64 | 32 | 128 | 2 | 4 | one-tile coverage for most batch-1 groups |
| Small S3 | 64 | 128 | 32 | 128 | 1 | 8 | high A reuse with 64-row full tiles |
| Large L1 | 128 | 128 | 32 | 128 | 2 | 8 | batch 4 and batch 16 |
| IQ2_S-only reuse test | 256 | 64 | 32 | 128 | 4 | 4 | large groups after the natural-group decoder |

For one decoded BF16 B tile:

- `N=64, K=32` requires 4 KiB of LDS;
- `N=128, K=32` requires 8 KiB;
- a simultaneous pair requires twice that before type-specific padding.

Gate/up Q3_K's initial eight-BF16 row padding raises the pair footprint above the nominal 16 KiB, but it remains comfortably below the 64 KiB workgroup limit. LDS capacity alone is not acceptance evidence; final occupancy, VGPR allocation, bank behavior, and complete latency decide the dispatch.

Every candidate remains compile-time specialized for:

- gate/up: `OutFeatures=512`, `InFeatures=2048`, `BlocksPerWeightRow=8`, 16 K=32 iterations;
- down: `OutFeatures=2048`, `InFeatures=512`, `BlocksPerWeightRow=2`, 64 K=32 iterations.

The production N dimensions are divisible by both 64 and 128, so neither family needs an input-column tail.

## Optimization phases

### GB0: lock the baseline

Completed:

- full 60-point benchmark;
- real GGUF weights and nonzero BF16 cotangents;
- AITER production heuristic comparison;
- exact single-projection correctness;
- fused-pair numerical characterization;
- kernel traces, code-object resources, ISA inspection, and one-counter-at-a-time profiling;
- one-expert geometry diagnostic.

Do not alter the baseline artifact. Every retained step should write a new `/tmp/grouped_mmq_bwd_stepN_*.json` artifact.

### GB1: share the proven backward tile core and establish Q4_K

Start with down Q4_K because it is a single projection, covers 18 layers, has a mature dense decoder, and avoids pair-specific resource questions.

First make the proven dense primitives reusable from grouped code. If helpers are moved into a shared project-owned header, rebuild and benchmark the unchanged dense Q4_K controls before changing grouped dispatch. Retain the refactor only when dense latency, correctness, and code-object resources remain within measurement noise.

Then implement the fixed-shape grouped Q4_K families:

- L1: `M=128, N=128, K=32`, 128 threads, two M and eight N accumulators per wave;
- S1/S2 controls using `N=64, K=32` for small expert groups;
- cooperative width-16 Q4_K decode with exact loader coverage;
- paired local-fragment prefetch;
- two-row packed-byte prefetch limited to the current decode phase;
- the retained shared-down 16-BF16-chunk XOR swizzle;
- compile-time `OutFeatures=2048`, `InFeatures=512`, two packed blocks per weight row, and exact N bounds;
- serial `(expert, N tile)` row traversal for the first arithmetic ablation.

The L1 kernel should load each cotangent fragment once and reuse it across eight N accumulators. The N=64 controls reuse each fragment across four accumulators while providing a lower-VGPR escape path.

Focused gates:

- down Q4_K B1 uniform, sparse, and boundary for small-family dispatch;
- down Q4_K B4 uniform and boundary;
- down Q4_K B16 uniform and skewed;
- one-expert rows 64, 256, and 1,024;
- exact BF16 equality with the existing grouped single result;
- zero private segment and zero spills.

The first large-group checkpoint is to approach the dense one-expert Q4_K result at 1,024 rows. The first small-group checkpoint is to beat or match the current B1 packed operator, not merely the dense one-expert microbenchmark.

If literal L1 spills after grouped metadata is added, apply the resource escape ladder before rejecting cooperative decode: shorten lifetimes and fixed-shape state first, then compare S2 `M=128, N=64, K=32`.

#### GB1 result: retained Q4_K L1

Status: retained.

The first implementation added a fixed down-Q4_K `M=128, N=128, K=32`, 128-thread serial expert kernel in `csrc/ck/grouped_mmq_backward_tiled.cuh`. It reuses the dense width-16 preloaded Q4_K decoder, paired local-fragment loads, and 16-BF16 XOR swizzle. Exact full 128-row chunks and one bounded tail execute in separate compile-time bodies.

Artifacts:

```text
/tmp/grouped_mmq_bwd_step1_q4_l1.json
/tmp/grouped_mmq_bwd_step1_q4_matrix.json
/tmp/grouped_bwd_step1_readobj.txt
/tmp/grouped_bwd_step1_disasm.txt
```

| Point | Baseline ms | GB1 ms | Speedup | AITER ms |
|---|---:|---:|---:|---:|
| Down Q4_K B4 uniform | 65.633 | 10.846 | 6.05x | 9.033 |
| Down Q4_K B4 boundary | 63.250 | 11.636 | 5.44x | 10.254 |
| Down Q4_K B16 uniform | 302.946 | 40.270 | 7.52x | 44.022 |
| Down Q4_K B16 skewed | 293.581 | 40.906 | 7.18x | 35.540 |
| Down Q4_K B16 sparse | 293.758 | 41.669 | 7.05x | 32.887 |
| Down Q4_K B16 boundary | 290.333 | 40.225 | 7.22x | 34.768 |

All checked outputs remained bitwise equal to dense packed MMQ and BF16 AITER. Batch 1 still uses the original small-group path and remained within its prior range.

Final code-object resources for the new kernel are:

```text
244 VGPRs
22 SGPRs
8,192-byte LDS
0-byte private segment
0 VGPR spills
0 SGPR spills
no dynamic stack
```

The result validates the main hypothesis: broad A-fragment reuse, K=32, cooperative decode, and exact full-row handling are the first-order grouped-backward gains. GB1 is already near AITER and exceeds it on the measured B16 uniform control, but nonuniform B16 and B4 still leave room for scheduler, geometry, and address refinements.

### GB2: add Q3_K and preserve fused gate/up

Port the retained narrow-Q3_K width-16 decoder and its measured layout to gate/up:

- K=32;
- paired local-fragment prefetch;
- eight-BF16 row padding;
- no explicit packed-byte prefetch for the misaligned 110-byte Q3_K block.

Evaluate L1 and the small S1/S2/S3 candidates rather than forcing one geometry across batch 1 and batch 16.

Keep one FP32 accumulator set for the final fused gradient. Compare two pair-staging mechanisms:

- decode both packed weight tiles into two padded LDS regions, synchronize once, and issue both projection WMMAs;
- reuse one padded LDS region sequentially, accepting extra barriers but reducing LDS and live state.

The simultaneous form is the preferred first candidate because it retains two barriers per K=32 iteration. However, final pair VGPR and LDS resources decide the result; the grouped-forward decoded-cache regression shows that extra LDS is not free even when capacity permits it.

Within the WMMA phase, keep first-projection and second-projection A/B fragment scopes separate. Do not hold both A fragments live together. Preserve projection and K accumulation order so the optimized pair can be checked bitwise against the current fused kernel.

Focused gates:

- gate/up Q3_K B1 uniform and sparse for small-family selection;
- gate/up Q3_K B4 boundary;
- gate/up Q3_K B16 uniform;
- one-expert 1,024-row pair diagnostic;
- output-only peak allocation;
- zero spills.

Do not replace the fused pair with two public outputs and `torch.add` as the production design. That would triple pair output allocation and change the single-rounding contract.

#### GB2 result: retained Q3_K L1 fused pair

Status: retained.

The retained kernel stages two padded `N=128, K=32` Q3_K B tiles in LDS, keeps one 128x128 FP32 accumulator set, and executes the two projection A/B fragment scopes sequentially. It uses the dense narrow-Q3_K width-16 decoder, eight-BF16 padding, K=32, and no explicit packed-byte prefetch.

Artifacts:

```text
/tmp/grouped_mmq_bwd_step2_q3_pair.json
/tmp/grouped_mmq_bwd_step2_q3_pair_matrix.json
/tmp/grouped_bwd_step2_readobj.txt
/tmp/grouped_bwd_step2_disasm.txt
```

| Point | Baseline ms | GB2 ms | Speedup | AITER ms |
|---|---:|---:|---:|---:|
| Gate/up Q3_K B4 uniform | 128.320 | 11.276 | 11.38x | 24.759 |
| Gate/up Q3_K B4 boundary | 109.649 | 14.362 | 7.63x | 30.619 |
| Gate/up Q3_K B4 sparse | 110.148 | 13.761 | 8.00x | 28.661 |
| Gate/up Q3_K B16 uniform | 636.629 | 45.016 | 14.14x | 129.758 |
| Gate/up Q3_K B16 boundary | 581.575 | 48.455 | 12.00x | 138.254 |
| Gate/up Q3_K B16 sparse | 572.634 | 48.714 | 11.76x | 143.105 |

The fused output retains the established `0.00285-0.00287` NRMSE against two separately rounded BF16 projections plus add. Batch 1 remains on the original small-group kernel.

Final resources are:

```text
248 VGPRs
27 SGPRs
20,480-byte LDS
0-byte private segment
0 VGPR spills
0 SGPR spills
no dynamic stack
```

The pair is 2.1-2.9x faster than AITER across the measured B4/B16 controls while preserving one output allocation. Its 248 VGPR allocation leaves little room for added live state, so later scheduling and pointer work must shorten or preserve lifetimes rather than add deep prefetch.

### GB3: split full-row and tail paths before scheduler work

Create separate compile-time bodies for:

- exact full 64-row tasks in S1/S3;
- exact full 128-row tasks in S2/L1;
- bounded tail tasks.

The full bodies should use:

- unmasked BF16 cotangent fragment loads;
- unmasked BF16 result stores;
- no per-element row division or remainder;
- one precomputed row stride and affine K pointers;
- compile-time output-feature and input-feature bounds.

The tail body computes `valid_rows` once, treats each cotangent/output region as contiguous BF16 spans, and zero-fills only invalid WMMA rows. Boundary routing remains a required benchmark, but tail control must not tax uniform batch-4 and batch-16 work.

Measure this phase before row descriptors. Grouped-forward G4 showed that full/tail specialization can be larger than the later scheduler gain, and an M=64 body may turn batch-1 uniform routing into an entirely full-tile workload.

#### GB3 result: retained S1 `M=64, N=64, K=32`

Status: retained as the batch-1 family for Q3_K pair and Q4_K single.

S1 assigns one sixteen-value decoder group to every loader thread, uses four waves, four N accumulators per wave, and separates exact 64-row bodies from bounded tails. Uniform batch-1 groups execute one exact full tile. Skewed and sparse groups execute one full tile plus one bounded tail when they exceed 64 rows.

Artifact:

```text
/tmp/grouped_mmq_bwd_step3_s1.json
/tmp/grouped_bwd_step3_readobj.txt
/tmp/grouped_bwd_step3_disasm.txt
```

| Point | Baseline ms | S1 ms | Speedup | AITER ms |
|---|---:|---:|---:|---:|
| Gate/up Q3_K B1 uniform | 21.539 | 3.614 | 5.96x | 11.010 |
| Gate/up Q3_K B1 skewed | 22.898 | 5.653 | 4.05x | 10.837 |
| Gate/up Q3_K B1 sparse | 24.557 | 5.328 | 4.61x | 8.709 |
| Gate/up Q3_K B1 boundary | 21.370 | 5.303 | 4.03x | 10.891 |
| Down Q4_K B1 uniform | 11.518 | 3.590 | 3.21x | 3.417 |
| Down Q4_K B1 skewed | 12.689 | 4.190 | 3.03x | 3.644 |
| Down Q4_K B1 sparse | 13.554 | 4.503 | 3.01x | 3.246 |
| Down Q4_K B1 boundary | 11.776 | 4.129 | 2.85x | 3.716 |

Gate/up Q3_K now beats AITER by 1.6-3.0x at batch 1. Q4_K reaches near parity on uniform/boundary and remains 13-39% behind on skewed/sparse, leaving a focused small-group geometry or row-ownership opportunity.

Resources remain well below the spill boundary:

| Kernel | VGPRs | SGPRs | LDS | Private/spills |
|---|---:|---:|---:|---:|
| Q4_K S1 single | 117 | 24 | 4,096 B | 0 |
| Q3_K S1 pair | 183 | 26 | 10,240 B | 0 |

A controlled S2 `M=128, N=64, K=32` follow-up was rejected as a universal small family. It regressed Q3_K uniform from 3.614 to 6.936 ms and Q4_K uniform from 3.590 to 4.185 ms because every 64-row group became a half-empty bounded tile. It also did not improve Q3_K sparse routing.

S2 was retained only for Q4_K when `rows / num_groups >= 80`. This device-metadata-independent host dispatch identifies the measured batch-1 sparse case without reading offsets. A 25-repeat control measured Q4_K sparse at 4.037 ms with S2 versus 5.201 ms with S1. The retained S2 code object uses 173 VGPRs, 30 SGPRs, and 4,096 bytes of LDS with no private segment or spills. The broader experiment artifact is `/tmp/grouped_mmq_bwd_step3_s2.json`; the controlled artifacts are `/tmp/grouped_mmq_bwd_q4_sparse_s2_25.json` and `/tmp/grouped_mmq_bwd_q4_sparse_s1_25.json`.

### GB4: measure serial ownership, tile ordering, and device row tasks

Do not assume descriptors are required.

With L1 `N=128`, a serial all-expert launch has:

- 16 workgroups per gate/up expert, 4,096 total;
- 4 workgroups per down expert, 1,024 total.

With `N=64`, those counts double. All are sufficient to cover 40 CUs even for the measured sparse expert counts.

Serial `(expert, N tile)` ownership has two useful properties: no setup and immediate reuse of the same packed tile across row chunks inside one workgroup. Test it first.

If long or nonuniform groups benefit from independent row tasks, reuse the project-owned forward mechanism:

- one 256-thread device setup kernel;
- atomics-free shared-memory prefix sum;
- device-resident `(expert, row_start, row_end)` records;
- capacity `ceil(R / M_TILE) + G`;
- no host-built descriptors or synchronization.

Unlike grouped forward, the fused backward pair is already one arithmetic launch, so descriptor setup is not amortized across gate and up. Require a stricter complete-latency win.

For a descriptor path, compare two flattened tile orders:

- M-major: N tile fastest within each row task;
- N-major: row tasks for one expert/N tile adjacent, favoring packed-weight locality.

Also test a fixed-program grid-stride kernel over the device task count with 256, 512, or 1,024 workgroups only after the nonpersistent task path is understood. This is the closest project-owned analogue to AITER's persistent grid, but it must preserve sparse behavior and remain spill-free.

Measure serial, descriptor, and persistent candidates separately for gate/up and down. Backward down has only four L1 N tiles per expert, so the grouped-forward down-descriptor rejection is evidence to consider, not a conclusion to copy. Keep direct serial ownership for batch-1 sparse routing unless a measured device path wins without launching inactive-expert arithmetic.

#### GB4 result: retained M-major device row tasks for large down

Status: retained for Q4_K, Q5_K, and IQ2_S down when the average group has at least 128 rows. Gate/up pair and all batch-1 paths remain serial and preserve output-only pair allocation.

The retained path allocates a small device workspace, builds 128-row tasks with the existing atomics-free 256-thread prefix-sum kernel, then launches four N workgroups per task. The grid is M-major: all four N tiles for one row task are adjacent. No CPU descriptors, `.item()`, device-to-host metadata copies, or synchronization were added.

Artifacts:

```text
/tmp/grouped_mmq_bwd_step4_row_tasks_mmajor.json
/tmp/grouped_mmq_bwd_step4_row_tasks_bounded.json
/tmp/grouped_mmq_bwd_step4_row_tasks_bounded_tail.json
/tmp/tasks_final_readobj.txt
/tmp/tasks_final_disasm.txt
```

Representative serial-to-task changes are:

| Point | Serial ms | Row-task ms | Speedup |
|---|---:|---:|---:|
| Down Q4_K B4 uniform | 10.846 | 10.461 | 1.04x |
| Down Q4_K B16 uniform | 40.270 | 35.821 | 1.12x |
| Down Q4_K B16 sparse | 41.669 | 37.143 | 1.12x |
| Down Q5_K B4 sparse | 10.797 | 10.438 | 1.03x |
| Down Q5_K B16 uniform | 37.421 | 34.074 | 1.10x |
| Down Q5_K B16 boundary | 37.849 | 34.725 | 1.09x |
| Down IQ2_S B4 skewed | 11.493 | 10.096 | 1.14x |
| Down IQ2_S B16 uniform | 38.992 | 32.806 | 1.19x |
| Down IQ2_S B16 boundary | 39.785 | 33.569 | 1.19x |

The task workspace adds only the task metadata above the single public output. For B4 down the measured incremental allocation was 67,118,592 bytes versus 67,108,864 output bytes, a 9,728-byte descriptor overhead.

N-major task order was rejected and reverted with git. Making row tasks adjacent for one N tile nearly doubled complete latency: Q4_K B16 uniform moved from 35.821 to 65.116 ms, Q5_K from 34.074 to 62.936 ms, and IQ2_S from 32.806 to 62.841 ms. The result shows that keeping all four N tiles of one row task adjacent is essential for packed-weight and cotangent locality. Artifact: `/tmp/grouped_mmq_bwd_step4_row_tasks_nmajor.json`.

A fixed 1,024-program grid-stride traversal was also rejected and reverted with git. Q4_K B16 uniform regressed from 35.821 to 38.717 ms and IQ2_S from 32.806 to 36.870 ms; the Q5_K persistent body additionally introduced 10 VGPR spills and a 44-byte private segment. At this arithmetic intensity, direct nonpersistent tasks already provide enough workgroups and better preserve one-task locality. Artifact: `/tmp/grouped_mmq_bwd_step4_persistent1024.json`.

A runtime full/tail branch was rejected despite slightly better latency because it produced private segments and 2-4 VGPR spills in Q4_K/Q5_K. The retained task kernels use one bounded body for all tasks and are spill-free:

| Kernel | VGPRs | SGPRs | LDS | Private/spills |
|---|---:|---:|---:|---:|
| Q4_K row task | 244 | 24 | 8,192 B | 0 |
| Q5_K row task | 256 | 24 | 8,192 B | 0 |
| IQ2_S row task | 233 | 24 | 8,192 B | 0 |

### GB5: port Q5_K with Q4_K

Q5_K is only two layers. It should reuse the Q4_K tile framework and differ only in decode and layout details.

Port the retained shared-down Q5_K choices:

- sixteen-value decode;
- packed-byte prefetch within the decode phase;
- four-BF16-chunk XOR swizzle;
- scalar low/high quant extraction, because the packed extraction retained for narrow Q5_K regressed shared down.

Reject any Q5_K specialization that regresses Q4_K, introduces spills, or adds broad complexity for a low-weight secondary case.

#### GB5 result: retained Q5_K port

Status: retained.

Q5_K now uses the same L1/S1 full-and-tail framework as Q4_K, with width-16 decode, four-BF16 XOR swizzle, and scalar low/high extraction. The L1 loader prefetches two packed rows per thread into bounded local state. Adding that packed prefetch improved the initial Q5_K port by 12-24% across the B4/B16 matrix.

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
| Down Q5_K B1 uniform | 11.728 | 3.694 | 3.17x | 3.377 |
| Down Q5_K B1 sparse | 13.713 | 3.913 | 3.50x | 3.221 |
| Down Q5_K B1 boundary | 11.898 | 3.544 | 3.36x | 3.626 |
| Down Q5_K B4 uniform | 68.550 | 9.862 | 6.95x | 8.952 |
| Down Q5_K B4 sparse | 65.608 | 10.797 | 6.08x | 8.625 |
| Down Q5_K B16 uniform | 315.237 | 37.421 | 8.42x | 43.023 |
| Down Q5_K B16 sparse | 304.028 | 38.851 | 7.83x | 32.650 |

Q5_K now reaches AITER parity at B1 boundary and beats AITER by 15% at B16 uniform. Nonuniform B4/B16 remains 13-25% behind AITER.

The L1 kernel uses 256 VGPRs, 22 SGPRs, and 8,192 bytes of LDS. The S1 kernel uses 115 VGPRs, 22 SGPRs, and 4,096 bytes of LDS. Both have zero private segment and zero spills. L1 is at the architectural per-thread VGPR ceiling, so no additional long-lived prefetch state is allowed.

A Q5_K S2 sparse-routing experiment was rejected and removed: a 25-repeat control measured 4.065 ms for S2 versus 3.913 ms for S1. Unlike Q4_K, the extra Q5_K accumulator state did not repay its half-empty-row reduction. Artifacts are `/tmp/grouped_mmq_bwd_q5_sparse_s2_25.json` and `/tmp/grouped_mmq_bwd_q5_sparse_s1_25.json`.

### GB6: specialize IQ2_S around eight- and sixteen-value packed groups

IQ2_S is the highest representation-specific priority because gate/up IQ2_S plus down IQ2_S account for approximately 50-58% of weighted packed latency.

Do not begin with another scalar-decoder geometry sweep. First add project-owned natural-group decoders:

- width 8: one grid index, one 64-bit grid load, one sign byte, and one scale selected for the eight values;
- width 16: two grid entries and sign bytes under one shared scale nibble and `d` factor.

Adapt the packed sign mechanism demonstrated by the vendored forward loader without editing vendor files: use packed byte comparisons/XOR/subtraction to form signed magnitude bytes, then convert those values to BF16 under the shared scale. Inspect final ISA to ensure grid loads, sign expansion, and scale formation are not repeated per output value.

Evaluate geometry in this order:

- S1/S2 `N=64, K=32` with width 8 and width 16;
- L1 `N=128, K=32` with width 16 if final resources remain spill-free;
- K=16 controls only if K=32 creates exposed decode latency or resource loss;
- the IQ2_S-only `M=256, N=64, K=32` decode-reuse test for large down groups.

Test gate/up pair and down separately. Pair staging may select a smaller N than single down because it owns two decoded B tiles and more decode state.

The decoder must be bitwise consistent with the existing scalar decode. Do not modify `csrc/vendor/llama_cpp/*`; divergent logic belongs in project-owned headers.

If the best spill-free local kernel remains substantially behind AITER after natural-group decode and efficient A reuse, stop the local geometry sweep and move IQ2_S to the representation-level phase.

#### GB6 initial result: retained cooperative width-16 IQ2_S decode

Status: retained; IQ2_S-only M geometry controls remain open.

The project-owned decoder loads two grid entries, two sign bytes, one shared scale nibble, and one `d` factor for each aligned group of sixteen values. It forms all sixteen BF16 values cooperatively without changing the vendored IQ2_S tables or structs. Both gate/up pair and down single now use S1/L1 K=32 kernels with exact full-row and bounded-tail paths.

Artifact:

```text
/tmp/grouped_mmq_bwd_step6_iq2.json
/tmp/grouped_bwd_iq2_readobj.txt
/tmp/grouped_bwd_iq2_disasm.txt
```

| Point | Baseline ms | GB6 ms | Speedup | AITER ms |
|---|---:|---:|---:|---:|
| Gate/up IQ2_S B1 uniform | 26.129 | 5.291 | 4.94x | 11.029 |
| Gate/up IQ2_S B1 sparse | 21.435 | 7.894 | 2.72x | 8.702 |
| Gate/up IQ2_S B4 uniform | 135.136 | 13.399 | 10.09x | 25.141 |
| Gate/up IQ2_S B4 sparse | 123.222 | 16.103 | 7.65x | 29.036 |
| Gate/up IQ2_S B16 uniform | 771.007 | 53.998 | 14.28x | 130.139 |
| Gate/up IQ2_S B16 sparse | 768.413 | 56.595 | 13.58x | 145.218 |
| Down IQ2_S B1 uniform | 12.698 | 4.281 | 2.97x | 3.401 |
| Down IQ2_S B1 sparse | 10.502 | 5.537 | 1.90x | 3.198 |
| Down IQ2_S B4 uniform | 55.716 | 10.307 | 5.41x | 8.676 |
| Down IQ2_S B4 sparse | 84.170 | 11.101 | 7.58x | 8.765 |
| Down IQ2_S B16 uniform | 395.499 | 38.992 | 10.14x | 43.613 |
| Down IQ2_S B16 sparse | 483.385 | 40.570 | 11.91x | 35.166 |

All gate/up IQ2_S points now beat AITER, by 1.1-2.6x. Down IQ2_S beats AITER at B16 uniform but remains 12-73% behind across the other measured routing points.

Resources are spill-free:

| Kernel | VGPRs | SGPRs | LDS | Private/spills |
|---|---:|---:|---:|---:|
| IQ2_S down L1 | 255 | 22 | 8,192 B | 0 |
| IQ2_S down S1 | 90 | 22 | 4,096 B | 0 |
| IQ2_S pair L1 | 250 | 56 | 16,384 B | 0 |
| IQ2_S pair S1 | 159 | 54 | 8,192 B | 0 |

The L1 kernels are already at or near the VGPR ceiling. Further IQ2_S work must use the planned `M=128/256, N=64` controls to exchange N accumulators for row reuse; it must not add long-lived prefetch state.

#### GB6 M-reuse controls

The IQ2_S-only `M=256, N=64, K=32` large-down control was rejected and reverted with git. It regressed every B4/B16 routing point: B4 uniform moved from 10.307 to 11.697 ms and B16 uniform moved from 38.992 to 44.059 ms. Halving N doubled the N workgroups, and the extra four-M-tile cotangent live state did not repay that repeated scheduling and decode work. The rejected kernel was spill-free at 225 VGPRs, 24 SGPRs, and 4,096 bytes of LDS, so the loss is geometry rather than spilling.

S2 `M=128, N=64, K=32` was retained only for IQ2_S down when `rows / num_groups >= 80` but the large-family threshold is not reached. This selects batch-1 sparse routing and reduces its latency from 5.537 to 3.618 ms, a 1.53x improvement and only 11% behind AITER. The retained S2 kernel uses 161 VGPRs, 30 SGPRs, and 4,096 bytes of LDS with no private segment or spills.

Artifact:

```text
/tmp/grouped_mmq_bwd_step6_iq2_n64_reuse.json
/tmp/grouped_bwd_iq2_reuse_readobj.txt
```

### GB7: K-loop, LDS, and addressing refinements

After the arithmetic families are retained, run controlled ablations for:

- simultaneous versus sequential pair LDS staging;
- vector versus mixed scalar/vector LDS fragment loads;
- grouped controls for the transferred dense swizzles and padding;
- affine packed-row/cotangent pointer increments versus recomputed offsets;
- fixed 16-iteration gate and 64-iteration down loops with modest unroll factors;
- selective current-phase packed-byte prefetch for Q4_K/Q5_K;
- explicit wave-uniform scalar offsets only when ISA shows avoidable VGPR address state.

Do not retry cross-iteration packed prefetch, K=64 ordinary reduction depth, custom barriers, or a second LDS buffer without new profiler evidence. Previous dense and grouped passes already measured those mechanisms as neutral or negative.

Judge every ablation by complete public-operator latency. Instruction-count reductions without latency wins are not sufficient.

#### GB7 split full/tail task result: rejected

A device setup that emitted separate full-task and tail-task lists was implemented for Q4_K/Q5_K and reverted with git. Separate compile-time full and bounded kernels improved uniform B16 modestly, from 35.821 to 34.067 ms for Q4_K and from 34.074 to 32.007 ms for Q5_K. It regressed every nonuniform B4 point by 6-14% because the second arithmetic launch and tail-list scheduling outweighed removal of row bounds. Since routing distribution cannot be selected from host-visible `rows` and `num_groups`, retaining this path would regress the production dynamic-routing matrix. Artifact: `/tmp/grouped_mmq_bwd_step7_split_tasks.json`.

The retained single bounded row-task body is therefore the stopping point for full/tail task scheduling. A future split is only justified if task metadata is already reusable across multiple backward calls or a device-side launch mechanism removes the extra public-operator launch cost.

#### GB7 IQ2_S LDS-layout result: retained pair-specific swizzle and N=64 large tile

The original shared IQ2_S swizzle was split by operator. Down retains a sixteen-BF16 XOR swizzle; pair uses a four-BF16 XOR swizzle. Using four for down regressed B4/B16 by 20-35%, while using it for the pair reduced latency by 15-25%.

The pair-specific four-BF16 swizzle made the original `N=128` L1 body spill, so the resource escape ladder was applied. The retained large pair is `M=128, N=64, K=32`: it uses 219 VGPRs, 54 SGPRs, and 8,192 bytes of LDS with no private segment or spills. The small S1 pair uses 194 VGPRs, 54 SGPRs, and 8,192 bytes of LDS with no private segment or spills.

| Point | Previous ms | Retained ms | Speedup |
|---|---:|---:|---:|
| Gate/up IQ2_S B1 uniform | 5.291 | 4.044 | 1.31x |
| Gate/up IQ2_S B1 sparse | 7.894 | 5.925 | 1.33x |
| Gate/up IQ2_S B4 uniform | 13.399 | 10.618 | 1.26x |
| Gate/up IQ2_S B4 sparse | 16.103 | 13.586 | 1.19x |
| Gate/up IQ2_S B16 uniform | 53.998 | 44.101 | 1.22x |
| Gate/up IQ2_S B16 skewed | 57.274 | 46.807 | 1.22x |

Artifacts:

```text
/tmp/grouped_mmq_bwd_step7_iq2_swizzle0.json
/tmp/grouped_mmq_bwd_step7_iq2_swizzle4.json
/tmp/grouped_mmq_bwd_step7_iq2_split_swizzle.json
/tmp/grouped_mmq_bwd_step7_iq2_pair_n64.json
/tmp/iq2_pair_n64_readobj.txt
```

#### GB7 Q3_K pair N=64 result: retained

The same `M=128, N=64, K=32` large-pair geometry was tested for Q3_K and retained. It reduces the large kernel from 248 to 206 VGPRs, keeps 26 SGPRs and 10,240 bytes of LDS, and remains free of private segments and spills.

| Point | N=128 ms | N=64 ms | Speedup |
|---|---:|---:|---:|
| Gate/up Q3_K B4 uniform | 11.276 | 10.560 | 1.07x |
| Gate/up Q3_K B4 boundary | 14.362 | 13.892 | 1.03x |
| Gate/up Q3_K B16 uniform | 45.016 | 44.630 | 1.01x |
| Gate/up Q3_K B16 skewed | 49.202 | 47.229 | 1.04x |
| Gate/up Q3_K B16 sparse | 48.714 | 47.461 | 1.03x |

The batch-1 S1 path was unchanged within noise. Artifacts: `/tmp/grouped_mmq_bwd_step7_q3_pair_n64.json`, `/tmp/grouped_mmq_bwd_step7_q3_pair_n64_b1_control.json`, and `/tmp/q3_pair_n64_readobj.txt`.

#### GB7 IQ2_S width-8 control: rejected

The natural width-8 decoder was measured after the width-16 path was retained and then reverted with git. It duplicated scale work and doubled loader groups. Gate/up B16 uniform regressed from 44.101 to 50.274 ms, while down B16 uniform regressed from 32.959 to 42.949 ms. Width 16 is the final local decoder because it matches the scale-nibble sharing boundary and provides exact loader coverage with less repeated state. Artifact: `/tmp/grouped_mmq_bwd_step7_iq2_width8.json`.

### GB8: representation-level ceiling

This phase begins only after the grouped tile reaches the retained dense-core neighborhood and final kernels are spill-free.

Potential architectural alternatives are:

- a compact lossless decoded IQ2_S cache substantially smaller than BF16;
- cross-call decoded-weight reuse;
- a transient project-owned packed-to-dense decode stage for sufficiently large routed batches;
- a persistent lossless int8-plus-scale representation consumed directly by a grouped WMMA kernel.

A complete BF16 expert decode is expensive: one logical projection is about 512 MiB across 256 experts, and the gate/up pair is about 1 GiB. Any transient dense path must include decode time, workspace allocation, active-expert behavior, and all projections in the benchmark. It must not silently turn into a permanent second model copy.

## Experiment order and priorities

The recommended retained-step order is:

- shared dense/grouped backward tile primitives with a no-change dense control;
- Q4_K large- and small-group cooperative tiles;
- Q3_K fused pair with measured small/large dispatch;
- exact 64-row/128-row full bodies and bounded tails;
- serial tile ordering versus device tasks and fixed-program traversal;
- Q5_K reuse of the Q4_K framework as a low-cost extension;
- IQ2_S eight-/sixteen-value packed-group decode and geometry;
- local pointer/LDS/prefetch refinements;
- representation-level IQ2_S work if needed.

Primary performance gates are:

- batch-4 down, especially Q4_K and IQ2_S;
- batch-16 gate/up, especially IQ2_S and Q3_K;
- checkpoint-weighted latency across all five cases.

Batch-1 sparse behavior is a correctness and launch-efficiency gate even when it is not the largest arithmetic workload.

## Rejected starting directions

Do not begin with:

- register-count reduction in the current narrow baseline: it is already spill-free and low-resource;
- LDS bank-conflict tuning of the current 512-byte tile: measured stalls are negligible, although the future broad tile must remeasure its own layout;
- irregular N tiles that do not divide 512 and 2,048;
- K=64 ordinary tiles copied from AITER;
- eight-wave workgroups or larger accumulator tiles copied from grouped forward;
- cross-iteration packed prefetch or compiler-managed prefetch arrays;
- custom LDS barriers or synchronization-only changes;
- host-built grouped descriptors;
- per-expert Python or CPU launch loops;
- split-K or atomic accumulation into BF16 output;
- two independent pair outputs plus an add as the production replacement;
- complete BF16 decoded-weight caching in LDS;
- a full-model dense BF16 shadow copy;
- persistent scheduling before the arithmetic tile is fixed;
- broad online autotuning or runtime environment-variable dispatch.

Use compile-time typed templates and measured dispatch.

## Correctness gates for retained changes

Every retained step must verify:

- single kernels are bitwise equal to dense packed MMQ backward;
- optimized single kernels remain bitwise equal to the current grouped single operator;
- optimized pair kernels are bitwise equal to the current fused pair whenever accumulation order is unchanged;
- pair NRMSE against the BF16 AITER sum stays within the established range if a justified accumulation-order change is tested;
- full and tail paths agree;
- inactive experts launch no arithmetic workgroups;
- boundary sizes 1, 15, 16, 17, 63, 64, 65, 127, 128, and 129 remain valid;
- metadata stays device-resident and synchronization-free;
- the output remains BF16 and contiguous.

Add focused production-shape tests to `tests/test_grouped_mmq.py` as new scheduling or full/tail paths are retained.

## Performance and resource gates

For every candidate:

- benchmark sequentially;
- use real nonzero cotangents and real packed GGUF weights;
- record complete public-operator latency and peak allocation;
- compare against the preceding retained step and AITER;
- inspect the final gfx1151 code object;
- require zero VGPR spills, zero SGPR spills, no private segment, and no dynamic stack for retained production kernels;
- verify the intended sixteen-value decode, LDS load width, barrier count, and pointer form in final ISA;
- profile only representative retained candidates.

When differences are below approximately 3%, use at least 15 sequential repeats and compare adjacent A/B artifacts collected under the same conditions.

The one-expert 1,024-row diagnostic should be rerun after each major large-tile change. It tests whether grouped arithmetic has reached the dense packed core. It is not sufficient for small-group selection: B1 uniform, sparse, and boundary complete-operator points decide S1/S2/S3 dispatch because many active experts provide much more parallelism than the one-expert diagnostic.

If common dense/grouped primitives are refactored, retain an unchanged dense benchmark control in the same artifact series so a grouped gain cannot hide a dense regression.

## Completion criteria

The grouped-backward local optimization phase is complete when:

- the retained production kernels use fixed production shapes, cooperative multi-value decode, and measured small-/large-group dispatch;
- full 64-row/128-row and bounded-tail bodies are resolved;
- single and fused pair correctness gates pass;
- all production kernels are spill-free with no private segment;
- serial ordering, device row tasks, and any fixed-program traversal are resolved separately for gate/up and down;
- IQ2_S uses a measured natural-group decoder or is explicitly stopped at a representation-level limit;
- the full 60-point matrix and checkpoint-weighted estimate are updated;
- representative kernel traces and final code-object resources are documented;
- the retained grouped large-row path is close to the corresponding dense packed core;
- remaining AITER losses, if any, are attributable to packed representation/decode rather than local tile geometry, cotangent reload amplification, barriers, bounds, or metadata handling.

The final artifact should use a stable path such as:

```text
/tmp/grouped_mmq_bwd_final_full.json
```

## Validation checklist

For a retained code or dispatch change, run:

```bash
python -m compileall -q bench
pytest -q
cd ~/test_no_unsloth && pytest -q
python -m compileall -q ~/test_no_unsloth
cd ~/torch-ggml-ops && git diff --check
```

Then run the full 60-point grouped-backward matrix, inspect final code-object resources, and collect representative profiler traces sequentially.

Do not edit `~/transformers-qwen3-moe-fused`; it is legacy/reference-only.
