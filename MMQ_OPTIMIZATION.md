# Dense MMQ optimization status and plan

## Scope

This work covers dense `torch_ggml_ops::mmq` forward and dense `mmq_grad_input` backward on gfx1151.

Included:

- BF16 activations and cotangents;
- internal Q8_1 activation quantization for forward;
- packed GGUF Q3_K, Q4_K, Q5_K, Q6_K, and IQ2_S weights;
- BF16 outputs and input gradients;
- the 160 ordinary model projections;
- the packed Q6_K language-model head;
- production batch sizes 1, 4, and 16 at sequence length 2048.

Excluded from optimization in this pass:

- `grouped_mmq` and routed-expert scheduling;
- GatedDeltaNet projections with physical layout permutations;
- LoRA GEMMs and residual accumulation;
- changes to the public operator schema;
- changes to `csrc/vendor/llama_cpp/*`.

The grouped path still uses the rewritten shared activation quantizer because the old launch shape is incompatible with the new quantizer, but its multiplication kernel and scheduling were not optimized.

## Hardware and measurement rules

Measurements were taken on:

```text
GPU: Radeon 8060S Graphics
architecture: gfx1151, wave32, 40 CUs
PyTorch: 2.12.0+rocm7.15.0a20260701
HIP: 7.14.60850
```

The performance reference is `torch.mm` with the same BF16 activation or cotangent and the same logical GGUF weight dequantized to BF16. On this machine, PyTorch normally delegates these GEMMs to hipBLASLt.

The first forward and backward baseline runs were mistakenly executed at the same time and contended for the GPU. Those timings were discarded. Every number in this document comes from sequential runs with no concurrent GPU benchmark or profiler.

On gfx1151, int8 WMMA is approximately as fast as BF16 WMMA rather than twice as fast. Packed MMQ therefore wins through lower weight traffic, lower LDS/VGPR pressure, or better geometry, not through a nominal 2x int8 arithmetic rate.

## Benchmark harness

The repository now includes:

- `bench/benchmark_mmq_fwd.py`;
- `bench/benchmark_mmq_bwd.py`;
- `bench/mmq_benchmark_common.py`.

They read representative tensors directly from the production GGUF checkpoint, benchmark all dense production geometries, compare against BF16 `torch.mm`, and report correctness and allocation data.

Default runs:

```bash
source ~/venv_torch/bin/activate
python bench/benchmark_mmq_fwd.py
python bench/benchmark_mmq_bwd.py
```

Focused examples:

```bash
python bench/benchmark_mmq_fwd.py \
  --cases narrow_q4_k --batches 1,4,16 --warmup 3 --repeats 9

python bench/benchmark_mmq_bwd.py \
  --cases lm_head_q6_k --lm-head-chunks 64,128,256 \
  --warmup 3 --repeats 9
```

Reference artifacts from this pass:

```text
Sequential forward baseline: /tmp/mmq_fwd_baseline_primary_sequential.json
Final full forward:           /tmp/mmq_fwd_final_full.json
Sequential backward baseline:/tmp/mmq_bwd_baseline_primary_sequential.json
Final full backward:          /tmp/mmq_bwd_final_full_v3.json
```

The `/tmp` paths are measurement provenance, not repository inputs.

## Production shape matrix

For ordinary projections, `M = batch * 2048`:

| Batch | M |
| ---: | ---: |
| 1 | 2,048 |
| 4 | 8,192 |
| 16 | 32,768 |

Representative dense shapes:

| Family | `(N, K)` | Weight types | Model tensors |
| --- | ---: | --- | ---: |
| Query plus query gate | `(8192, 2048)` | Q3_K, Q4_K | 10 |
| Key/value/shared gate/up | `(512, 2048)` | Q3_K, Q4_K, Q5_K | 100 |
| Attention output | `(2048, 4096)` | Q4_K | 10 |
| Shared-expert down | `(2048, 512)` | Q4_K, Q5_K | 40 |

The LM head uses:

```text
N = 248320
K = 2048
weight = Q6_K
production forward chunk M = 64
```

Forward and backward comparison chunks are `M = 64, 128, 256`.

## Accepted forward changes

### One activation-quantization workgroup per real row

The original Q8_1 launch was:

```text
grid = [rows_padded, K / 256]
block = 64 threads
```

At `K=2048`, that created eight workgroups per activation row and also quantized padded rows. At `M=32768`, it launched 262,144 small workgroups. This dominated narrow-N forward calls.

The accepted kernel is:

```text
grid = [rows, 1]
block = 512 threads
```

Each workgroup owns one real row. Every thread processes four BF16 values per loop iteration, and the thread block loops only when `K > 2048`. The Q8_1 D4 and DS4 layouts, 32-value reductions, rounding, and workspace representation are unchanged. Padded workspace rows are not written because their corresponding MMQ outputs are bounds-masked.

The same launch is required by dense and grouped callers of the shared quantizer. This is not a grouped-MMQ multiplication optimization.

### Compile-time forward row tile and Q6_K `J=64`

Dense forward was generalized from a fixed `J=128` to a compile-time `J` template in project-owned code. The ordinary path remains:

```text
I = 64
J = 128
threads = 128, four wave32 waves
K iteration = 256
```

Q6_K calls whose padded row count is 64 use:

```text
I = 64
J = 64
threads = 128
K iteration = 256
```

This removes the production LM head's 64 padded rows and halves its accumulator and activation-tile footprint. Q6_K calls above 64 rows retain `J=128`.

No file under `csrc/vendor/llama_cpp/` was modified. The template adaptation is in `csrc/mmq_core.cuh` and dispatch is in `csrc/mmq_hip.cu`.

## Forward results

The table below compares the valid sequential primary baseline with the final full benchmark. Ratio means packed-MMQ throughput divided by BF16 throughput.

| Case | M | Baseline ms | Final ms | Speedup | Baseline ratio | Final ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Query Q3_K | 2,048 | 3.213 | 3.054 | 1.05x | 0.97x | 1.05x |
| Query Q3_K | 8,192 | 13.324 | 12.403 | 1.07x | 0.94x | 1.03x |
| Query Q3_K | 32,768 | 52.589 | 48.828 | 1.08x | 0.94x | 1.03x |
| Narrow Q4_K | 2,048 | 0.235 | 0.214 | 1.10x | 1.35x | 1.48x |
| Narrow Q4_K | 8,192 | 1.990 | 0.899 | 2.21x | 0.47x | 1.04x |
| Narrow Q4_K | 32,768 | 7.636 | 3.590 | 2.13x | 0.47x | 0.99x |
| Attention output Q4_K | 2,048 | 1.811 | 1.410 | 1.28x | 0.97x | 1.26x |
| Attention output Q4_K | 8,192 | 7.638 | 5.717 | 1.34x | 0.91x | 1.23x |
| Attention output Q4_K | 32,768 | 30.624 | 22.986 | 1.33x | 0.89x | 1.20x |
| Shared down Q4_K | 2,048 | 0.256 | 0.257 | 0.99x | 5.63x | 5.59x |
| Shared down Q4_K | 8,192 | 1.053 | 0.985 | 1.07x | 5.00x | 5.35x |
| Shared down Q4_K | 32,768 | 4.150 | 3.939 | 1.05x | 5.00x | 5.26x |
| LM head Q6_K | 64 | 7.413 | 4.202 | 1.76x | 1.17x | 2.05x |
| LM head Q6_K | 128 | 8.001 | 7.988 | 1.00x | 1.56x | 1.56x |
| LM head Q6_K | 256 | 16.097 | 16.127 | 1.00x | 0.79x | 0.81x |

Secondary final forward ratios:

| Case | M=2,048 | M=8,192 | M=32,768 |
| --- | ---: | ---: | ---: |
| Query Q4_K | 1.13x | 1.12x | 1.10x |
| Narrow Q5_K | 1.21x | 1.00x | 0.95x |
| Narrow Q3_K | 1.38x | 0.97x | 0.92x |
| Shared down Q5_K | 5.28x | 5.15x | 5.13x |

The dominant 70-tensor narrow Q4_K path now reaches BF16 parity at large M. The production 64-row LM head exceeds the 15 TFLOPS target at 15.49 TFLOPS.

## Forward profiling

### Narrow Q4_K, `M=32768, N=512, K=2048`

`rocprofv3` kernel tracing gave:

| Kernel | Baseline average | Final average | Final resources |
| --- | ---: | ---: | --- |
| Q8_1 quantizer | 5,105.979 us | 886.928 us | 32 allocated VGPR, no LDS, no private segment |
| Dense MMQ | 2,565.848 us | 2,645.314 us | 256 allocated VGPR, 38,400-byte LDS, no private segment |

The quantizer is 5.76x faster. Combined traced kernel time fell from about 7.67 ms to 3.53 ms, matching the benchmark's approximately 2.1x end-to-end improvement. The multiplication kernel did not become faster; the improvement came from removing quantization scheduling overhead.

The code-object metadata reports 254 architectural VGPRs for the Q4_K `J=128` kernel; rocprofv3 reports the rounded allocation as 256.

### Q6_K LM head, `M=64`

The `J=64` multiplication kernel averages about 4,190 us and uses:

```text
176 allocated VGPRs
28,928-byte LDS
0-byte private segment
```

The Q8_1 quantizer is only about 3.4 us at this row count. The 64-row forward case is therefore almost entirely multiplication time, and removing padded-row MMQ work is what produced the 1.76x speedup.

## Accepted backward redesign

The original backward kernel decoded one packed BF16 value at a time into a small 16x16 tile and had insufficient reuse and poor geometry. It sustained only about 0.05-0.13x of BF16 on most production shapes.

The accepted dense kernel in `csrc/ck/mmq_backward.cuh` now:

- uses four wave32 waves per 128-thread workgroup;
- computes one 16-row WMMA tile per wave, or 64 output rows per workgroup;
- is templated as `<quant_type, N_TILES, K_ITERATION, GROUP_M>`;
- retains multiple 16-column input-gradient tiles in WMMA accumulators;
- stages decoded packed-weight tiles in LDS;
- uses pair decoders for Q3_K, Q4_K, and Q5_K where profitable;
- uses a Q3_K four-value decoder for the wide-query `N_TILES=4` case;
- uses a shape-specific Q6_K decoder: four adjacent values for the one-N-tile 64-row kernel and sixteen adjacent values for kernels with at least two N tiles;
- selects geometry from row count, quantization type, and feature width.

Current Q6_K dispatch:

| Rows | N tiles | K iteration |
| ---: | ---: | ---: |
| `<=64` | 1 | 64 |
| `<=128` | 2 | 64 |
| `<=256` | 4 | 32 |
| `<=2048` | 8 | 16 |
| larger | 16 | 16 |

Current non-Q6 dispatch:

- rows `<=128`: one N tile;
- rows `<=256`: four N tiles;
- rows `<=2048`: type/shape-specific four, eight, or sixteen N tiles;
- rows `<=8192`: normally twelve N tiles, with sixteen for Q4_K/Q5_K at `in_features == 2048`;
- larger rows: sixteen N tiles.

For Q3_K at 2,048 rows, wide query shapes use the quad decoder with four N tiles. The narrow 512-output shape uses eight N tiles and pair decoding; this reduced its 2,048-row latency from about 1.19 ms to about 0.58 ms.

### Grouped M traversal for ordinary backward

The FeatherOps-inspired workgroup traversal experiment produced a large further improvement without changing kernel arithmetic. The old two-dimensional grid ran all M blocks for one N block before advancing N. For rows above 256, the accepted grid groups two M blocks at a time:

```text
grid = [2, n_blocks, ceil(m_blocks / 2)]
m_block = blockIdx.z * 2 + blockIdx.x
n_block = blockIdx.y
```

This keeps one `grad_output` tile hot while more input-gradient N blocks consume it, at the cost of a smaller packed-weight reuse window. `GROUP_M=1,2,4,8,16` and the old all-M ordering were benchmarked sequentially. Two was the best overall production compromise; one was nearly tied on narrow and Q3_K shapes, while four was slightly better only on attention output. Rows up to 256 retain the original launch order so the LM-head dispatch is unchanged.

Focused M=32,768 results before and after the accepted mapping:

| Case | Previous ms | `GROUP_M=2` ms | Improvement |
| --- | ---: | ---: | ---: |
| Query Q3_K | 162.884 | 112.026 | 1.45x |
| Narrow Q4_K | 11.130 | 7.051 | 1.58x |
| Attention output Q4_K | 80.633 | 55.768 | 1.45x |
| Shared down Q4_K | 14.095 | 10.458 | 1.35x |

The same mapping also improved the primary M=2,048 and M=8,192 cases. The next bounded ordinary-backward experiment remains the four-wave M=2 geometry family with one LDS buffer, followed by sixteen-value decoders and `K_ITERATION=64` only if geometry wins independently.

## Backward results

The following compares the valid sequential primary baseline with the final full benchmark.

| Case | M | Baseline ms | Final ms | Speedup | Baseline ratio | Final ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Query Q3_K | 2,048 | 34.240 | 10.548 | 3.25x | 0.12x | 0.38x |
| Query Q3_K | 8,192 | 272.857 | 43.353 | 6.29x | 0.06x | 0.36x |
| Query Q3_K | 32,768 | 1,108.314 | 162.884 | 6.80x | 0.06x | 0.37x |
| Narrow Q4_K | 2,048 | 3.057 | 0.461 | 6.64x | 0.06x | 0.42x |
| Narrow Q4_K | 8,192 | 12.590 | 2.291 | 5.49x | 0.06x | 0.30x |
| Narrow Q4_K | 32,768 | 48.332 | 11.130 | 4.34x | 0.06x | 0.25x |
| Attention output Q4_K | 2,048 | 15.067 | 3.534 | 4.26x | 0.09x | 0.40x |
| Attention output Q4_K | 8,192 | 62.211 | 17.782 | 3.50x | 0.09x | 0.32x |
| Attention output Q4_K | 32,768 | 487.945 | 80.633 | 6.05x | 0.05x | 0.28x |
| Shared down Q4_K | 2,048 | 2.722 | 0.579 | 4.70x | 0.13x | 0.58x |
| Shared down Q4_K | 8,192 | 10.771 | 3.208 | 3.36x | 0.11x | 0.35x |
| Shared down Q4_K | 32,768 | 66.903 | 14.095 | 4.75x | 0.06x | 0.29x |
| LM head Q6_K | 64 | 25.914 | 12.708 | 2.04x | 0.38x | 0.78x |
| LM head Q6_K | 128 | 54.352 | 18.501 | 2.94x | 0.27x | 0.78x |
| LM head Q6_K | 256 | 158.209 | 26.130 | 6.05x | 0.12x | 0.73x |

Secondary final backward ratios:

| Case | M=2,048 | M=8,192 | M=32,768 |
| --- | ---: | ---: | ---: |
| Query Q4_K | 0.40x | 0.33x | 0.34x |
| Narrow Q5_K | 0.34x | 0.29x | 0.26x |
| Narrow Q3_K | 0.33x | 0.31x | 0.28x |
| Shared down Q5_K | 0.58x | 0.36x | 0.29x |

Backward improved by 3.25-6.80x on most ordinary primary shapes. Sharing Q6_K decode across four or sixteen adjacent values improved the previously retiled kernel by another 1.9-2.8x and gives a 6.05x total speedup on the 256-row case. Backward remains materially slower than BF16.

## Backward profiling and final bottleneck

For narrow Q4_K at `M=32768`, the selected `dense_mmq_grad_input_kernel<Q4_K,16,16>` uses:

```text
160 allocated VGPRs
8,192-byte LDS
0-byte private segment
about 11.2 ms traced kernel time
```

All 35 emitted dense forward, quantization, and backward specializations have a zero-byte private segment in the final code object. Some pre-existing grouped forward specializations still have private segments; grouped MMQ is outside this pass.

Stochastic PC sampling of the same narrow Q4_K kernel reported the following reasons among samples where no instruction issued:

| Reason | Share |
| --- | ---: |
| ALU dependency | 40.3% |
| waitcnt / memory dependency | 27.2% |
| barrier wait | 20.2% |
| execution-pipe arbiter stall | 10.7% |
| no instruction available | 1.6% |

Frequently stalled program counters included `s_barrier`, BF16 `v_wmma`, and the shifts/masks used by packed Q4_K decode.

For the final Q6_K `M=256` kernel, tracing reports about 26.1 ms and:

```text
96 allocated VGPRs (91 architectural VGPRs in code-object metadata)
4,096-byte LDS
0-byte private segment
```

Final Q6_K PC sampling reported:

| Reason | Share |
| --- | ---: |
| ALU dependency | 36.0% |
| waitcnt / memory dependency | 23.1% |
| barrier wait | 22.2% |
| execution-pipe arbiter stall | 18.5% |
| no instruction available | 0.2% |

Before multi-value decoding, shifts and signed bit-field extracts dominated the hottest stalled program counters. In the final profile, `s_barrier`, BF16 `v_wmma`, and BF16 fragment moves are hottest. Multi-value decoding therefore removed the largest Q6_K scalar-decode bottleneck and shifted the limit toward synchronization, WMMA dependencies, and LDS-to-fragment movement.

These profiles show that backward is not limited by one missing launch tweak. Its remaining cost is the combination of:

1. repeated packed-weight bit extraction and scale/min reconstruction;
2. LDS staging and a synchronization pair for every reduction chunk;
3. memory wait before WMMA consumes the staged tile;
4. high accumulator/register usage for wide N tiling;
5. reuse constraints caused by the transposed operation `dY @ W`.

Unlike forward, packed scales vary along the backward reduction dimension, so an int8 WMMA rewrite cannot simply factor one original GGUF scale outside each 16-value WMMA without changing the math. A full BF16 dequantization cache would avoid repeated decode, but it was not accepted in this pass because it consumes memory that packed weights are intended to save. It remains a valid future tradeoff if latency is more important than resident memory.

## TensileLite and shipped hipBLASLt assessment

TensileLite is useful here as a record of tile geometries and instruction schedules that were measured on gfx1151, not as a generator that can directly emit this operation. Its dense GEMM kernels assume typed A and B operands with regular affine addresses. Packed MMQ backward instead reconstructs GGUF values, scales, and affine minima inside the reduction loop, so reusing Tensile's solution database, universal arguments, Python generator, or assembly kernel infrastructure would require turning MMQ into a new custom operation in that generator. That is a larger project than writing the relevant geometry and schedule directly in project-owned HIP.

Runtime hipBLASLt logging for the BF16 reference calls identified two dominant gfx1151 solution families. Runtime solution indices are library-global and do not equal the `SolutionIndex` values in an individual logic YAML; families must be matched by kernel name and parameter fields.

| Production backward reference | Logged column-major GEMM | Runtime solution | Selected gfx1151 family |
| --- | --- | ---: | --- |
| Query, M=32768 | `M=2048, N=32768, K=8192` | 817 | `MT96x96x32`, `MIWaveTile=3x3`, `MIWaveGroup=2x2` |
| Narrow, M=32768 | `M=2048, N=32768, K=512` | 891 | `MT128x32x32`, `MIWaveTile=2x2`, `MIWaveGroup=4x1` |
| Attention output, M=32768 | `M=4096, N=32768, K=2048` | 891 | `MT128x32x32`, `MIWaveTile=2x2`, `MIWaveGroup=4x1` |
| Shared down, M=32768 | `M=512, N=32768, K=2048` | 817 | `MT96x96x32`, `MIWaveTile=3x3`, `MIWaveGroup=2x2` |
| LM head, M=256 | `M=2048, N=256, K=248320` | 817 | `MT96x96x32`, `MIWaveTile=3x3`, `MIWaveGroup=2x2` |

The 96x96 family uses four wave32 waves, `DepthU=32`, `PrefetchGlobalRead=2`, `PrefetchLocalRead=1`, `ClusterLocalRead=1`, `ScheduleIterAlg=3`, transposed LDS, 16-element local-read width, two LDS buffers, and about 30,336 bytes of LDS. The 128x32 family also uses four wave32 waves, `DepthU=32`, `PrefetchGlobalRead=2`, `ScheduleIterAlg=3`, and 16-element local reads, but it selects one LDS buffer, `PrefetchLocalRead=0`, `ClusterLocalRead=0`, 8-element LDS padding, and about 10,496 bytes of LDS. Both use `SourceSwap=true`. Both leave `DirectToLdsA/B` and `DirectToVgprA/B` disabled, using the conventional global-to-VGPR-to-LDS path instead.

The BF16 references reached 18.166 logical TFLOP/s for query, 24.540 for narrow, 24.280 for attention output, and 16.724 for shared down. The selected families are therefore more relevant than generic Tensile tuning templates: they show which gfx1151 choices actually won on the production dimensions.

### Directly transferable patterns

1. **Use a balanced two-dimensional WMMA tile per wave.** The current ordinary backward kernel gives each wave one 16-row tile and as many as sixteen 16-column tiles. This maximizes reuse of one `grad_output` fragment but consumes 128 accumulator VGPRs per wave before decoder temporaries, producing the final 160-VGPR narrow Q4_K kernel. The selected dense kernels instead use 2x2 or 3x3 WMMA tiles per wave. The first packed prototype should use 2x2: two M fragments, two N fragments, and four accumulators. With a 4x1 wave group this forms a 128-row by 32-column workgroup tile, doubles reuse of each decoded weight tile from 64 to 128 rows, and reduces accumulator storage from 128 to 32 VGPRs per wave. The tradeoff is more N workgroups and therefore more repeated `grad_output` traffic, so it must be measured rather than assumed to win.

2. **Keep four wave32 waves and test wave grouping, not larger workgroups.** Tensile's selected solutions use the same 128-thread workgroup size as the final packed kernel. This agrees with the rejected larger-wave-count experiments. The useful variable is whether the four waves are grouped 4x1, as in the 128x32 family, or 2x2, as in the 96x96 family. A 4x1 prototype is simpler because the existing kernel already shares B across four M waves. A 2x2 layout becomes attractive only if A is also staged or otherwise shared across the waves that cover different N subtiles.

3. **Prefetch local fragments before attempting a full dual-LDS pipeline.** The final narrow Q4_K code object already emits wide `ds_load_b128` instructions for each BF16 B fragment, but its inner sequence is effectively LDS load, `s_waitcnt`, WMMA, then the next LDS load. Tensile's `PrefetchLocalRead` and `ClusterLocalRead` mechanisms move multiple local reads ahead of independent matrix instructions. A bounded HIP experiment can load two B fragments into ping-pong VGPR fragments before issuing their two independent WMMAs. This costs one extra eight-VGPR BF16 fragment but may reduce the 27.2% waitcnt and 10.7% execution-pipe stall shares without adding another LDS buffer.

4. **Treat K depth and global-read prefetch as geometry-dependent.** The selected production BF16 families use a reduction depth of 32, while the FeatherOps `MT128x96` family uses 64. Isolated `K_ITERATION` changes regressed the current 1xN geometry, but that does not rule out 32 or 64 after accumulator pressure and loader work assignment change. For packed operands, two-stage global prefetch maps most naturally to vector-loading the next decoder group's packed bytes and metadata into VGPRs while the current LDS tile is consumed. Wider 16-value decoders make that possible; blindly double-buffering scalar decode calls does not.

5. **Benchmark one and two LDS buffers separately.** The shipped logic selects both schedules on real shapes. The 128x32 family wins narrow and attention-output references with one approximately 10 KiB LDS buffer, while the 96x96 family wins query, shared-down, and LM-head references with two buffers and approximately 30 KiB LDS. This proves that dual buffering is not a universal gfx1151 rule. Start every new packed geometry with one buffer, then add an explicit two-buffer schedule only after the geometry and decoder width are accepted.

6. **Retain the existing LDS transpose and wide local reads.** `shared_b[local_input_column][k]` already transposes the packed row-major source into a K-contiguous WMMA fragment layout, and the compiler emits 128-bit LDS loads. Tensile's `TransposeLDS` and `LocalReadVectorWidth=16` therefore validate the current layout rather than suggesting a rewrite. Additional LDS padding was already slower in the present geometry and should be reconsidered only if a new tile produces a measured bank conflict.

7. **Retain the existing source/output orientation.** Tensile describes `SourceSwap` as swapping matrix-instruction inputs to improve the output-store pattern. The packed kernel already accounts for gfx11's transposed physical C fragment: lanes 0-15 write adjacent input-gradient columns in one row and lanes 16-31 write the next row. That is the favorable half-wave-coalesced store orientation, so a separate source-swap rewrite is not a first-order opportunity.

8. **Use Tensile's scheduler as a model, not a dependency.** `ScheduleIterAlg=3`, custom main-loop schedules, and the newer subtile scheduler explicitly interleave global reads, LDS writes, local reads, waits, and independent matrix instructions. The transferable result is the ordering discipline. HIP C++ can express small two-fragment or two-stage schedules; a larger schedule may require low-level HIP or inline assembly for stable instruction placement. Importing the generator would not solve packed decode or resource control.

### Patterns that do not transfer directly

- `DirectToLds` cannot bypass the GGUF reconstruction step, and `DirectToVgpr` cannot efficiently distribute one cooperatively decoded packed tile to all consumer waves. The selected dense reference solutions disable both features, and the attempted shuffle replacement for LDS was slower.
- Tensile's universal arguments, solution-selection database, Stream-K, GlobalSplitU, and dense epilogue machinery do not address the dominant repeated decode cost. Workgroup traversal itself does transfer and should be tested independently for cache locality, but the generator's mapping infrastructure is not reusable.
- gfx1151 does not support Tensile's WMMA arb-stall register optimization. `HasWmmaArbStallBit` is gated to ISA 12.5, so `disableWmmaArbStall()` emits nothing on ISA 11.5.1.
- A generated dense BF16 kernel can only be used after materializing a dense or regularly quantized representation. That is the separate transient BF16 or persistent Q8 cache strategy, not a direct fused packed-MMQ code-generation path.

The main conclusion is that Tensile strengthens the case for changing ordinary backward tile geometry and instruction ordering, but it does not remove the fundamental ceiling of the fused design: the same packed weights are still decoded once per M workgroup. A two-dimensional per-wave tile can plausibly reduce decoder repetition, VGPR pressure, LDS waits, and WMMA dependency chains together, but decode-once or persistent-cache paths retain the higher asymptotic ceiling.

## Additional implications from the FeatherOps gfx1151 logs

The optimization logs under `~/ComfyUI-FeatherOps/doc/` add several constraints and opportunities that are not obvious from the dense hipBLASLt logic alone.

- PC samples identify the instruction waiting to issue, not necessarily the unit consuming the most execution time. FeatherOps repeatedly found large `ds_load_b128` and conversion sample counts that were queue or instruction-fetch stalls. Before another large MMQ rewrite, use controlled ablations in a temporary benchmark build to estimate exposed packed-decode, global-refill, A-fragment-load, and B-fragment-load costs separately. Use real nonzero data so zero-valued WMMA shortcuts do not invalidate the result.
- `ConvertAfterDS` is successful for regularly packed fp8 because a compact byte operand can be loaded to LDS and cheaply expanded into each consumer fragment. Moving raw GGUF blocks to LDS and decoding after local reads is not directly analogous: the current backward kernel reconstructs each BF16 weight once cooperatively and shares it across four M waves. Consumer-side GGUF decode would repeat the expensive bit extraction and scale/min work in every wave. Retain producer-side cooperative decode unless the source representation has first been simplified.
- FeatherOps found that larger reduction chunks can win by reducing the frequency of long global-wait and barrier boundaries, even without a true asynchronous copy. This makes `K_ITERATION=64` worth revisiting only with a new geometry and a 16-value decoder that keep all 128 loader threads active; it is not a reason to repeat the rejected Q6_K chunk changes in the old geometry.
- Explicit scalar register prefetch state is safer than arrays or reference-based buffers. Compiler-managed arrays that were intended to live in VGPRs sometimes became large LDS allocations. Any packed-byte prefetch prototype should use a small fixed set of scalar `uint4` or dword values and verify code-object LDS/VGPR metadata before benchmarking.
- Workgroup traversal can be as important as the local tile. FeatherOps measured large WorkGroupMapping effects because traversal determines which operand remains hot in cache. The backward grid currently makes M blocks the fastest-varying dimension, maximizing decoded-weight reuse but repeatedly streaming the much larger `grad_output` tensor for every N block. A grouped 3D launch can trade some packed-weight reuse for `grad_output` tile reuse without adding arithmetic.
- On-the-fly conversion is useful only when the shorter representation and conversion schedule save more traffic than they add issue pressure. This favors a backward-specific lossless int8-plus-scale cache over immediately moving to an approximate int8-WMMA design: gfx1151 has no raw int8 WMMA throughput advantage, while eliminating GGUF bit extraction can still be valuable.

## Remaining large-margin opportunities

Forward has no large margin on the current production path: ordinary forward is at or above BF16 for the dominant shapes, and the production 64-row Q6_K LM head is 2.05x the BF16 throughput. The large remaining margins are in backward.

At batch 16, the benchmark's estimated serial time across all ordinary dense projections is 4,110.7 ms for packed backward versus 1,275.3 ms for BF16, leaving a 2,835.4 ms gap and a theoretical 3.22x latency margin. The 64-row LM-head schedule adds 6,506.6 ms packed versus 5,049.4 ms BF16. These estimates are useful for prioritization but are not end-to-end training times and do not model overlap with other work.

| Priority family at batch 16 | Packed estimated ms | BF16 estimated ms | Remaining gap |
| --- | ---: | ---: | ---: |
| Q6_K LM head with 64-row chunks | 6,506.6 | 5,049.4 | 1,457.2 ms |
| Query Q3_K | 1,466.0 | 544.7 | 921.2 ms |
| Narrow Q4_K | 779.1 | 196.0 | 583.1 ms |
| Attention output Q4_K | 806.3 | 226.4 | 579.9 ms |
| Shared down Q4_K | 422.8 | 123.3 | 299.6 ms |
| Narrow Q5_K | 229.2 | 59.1 | 170.2 ms |

### 1. Aggregate LM-head backward chunks

This is the largest low-risk scheduling opportunity. The final Q6_K kernel becomes substantially more efficient as M increases:

| Backward chunk | Kernel time | Calls for batch 16 | Estimated serial time |
| ---: | ---: | ---: | ---: |
| 64 | 12.708 ms | 512 | 6,506.6 ms |
| 128 | 18.501 ms | 256 | 4,736.3 ms |
| 256 | 26.130 ms | 128 | 3,344.6 ms |

Combining four 64-row cotangent chunks into one 256-row `mmq_grad_input` call would nearly halve LM-head backward time without changing the kernel or numerical result. The implementation can collect consecutive `[64, 248320]` BF16 cotangents in a ring buffer, launch one `[256, 248320]` backward call, and split the resulting `[256, 2048]` input gradient back into four logical chunks. The cost is memory: a 256-row BF16 cotangent is about 121 MiB, versus about 30 MiB for 64 rows. A 128-row mode is a lower-memory intermediate point. This should be optimized at the loss/LM-head scheduling layer rather than inside the MMQ kernel.

### 2. Retune backward traversal, then combine larger M reuse with wider decoders

This is the best remaining low-workspace experiment that preserves the fused packed-kernel design. Q6_K improved dramatically when one thread shared block and scale work across four or sixteen adjacent values, while the FeatherOps and hipBLASLt results show that tile geometry, K depth, and workgroup traversal must be tuned together.

First test workgroup traversal without changing arithmetic. The current launch is logically `(m_block, n_block)` with all M blocks adjacent for one N block. Add a grouped 3D mapping such as `grid=(GROUP_M, n_blocks, ceil(m_blocks/GROUP_M))`, with `m_block=blockIdx.z*GROUP_M+blockIdx.x` and `n_block=blockIdx.y`. Sweep `GROUP_M=1,4,8,16` plus the current all-M ordering on narrow Q4_K and query Q3_K at M=32,768. Small groups keep each `grad_output` row tile hot while several N blocks consume it; large groups keep one packed-weight tile hot across more M blocks. This is a prologue-only change with no extra workspace. Measure `L2CacheHit`, latency, and code-object resources rather than assuming either operand should always win.

Then add a template M-tile count and test a small family of four-wave, 4x1 geometries with two M tiles per wave: 2x2, 2x4, and 2x6 WMMA tiles per wave. They form logical workgroup tiles of 128x32, 128x64, and 128x96. The 2x2 form minimizes accumulator VGPRs; 2x6 matches the measured `MT128x96` FeatherOps/hipBLASLt family and preserves more A reuse. Retain `K_ITERATION=16`, one LDS buffer, and the current pair decoder for the first geometry isolation. Acceptance requires zero private segment, no occupancy cliff, and a measured improvement over the current 11.130 ms narrow Q4_K result.

Add format-specific 16-value Q3_K/Q4_K/Q5_K loaders to the winning geometry. Q3_K shares one scale across an aligned 16-value group. Q4_K and Q5_K share `d`, `dmin`, scale, and minimum metadata across aligned values within their 32-value group. Load adjacent quant bytes with dword or `uint4` accesses, generate the sixteen BF16 values in a short packed loop, and write them directly to LDS.

The most interesting new reduction-depth candidate is `K_ITERATION=64`, not 32, after 16-value decoding exists. For a 128x32 block, `(32/16)*64 = 128` decoder groups, so exactly one aligned sixteen-value group is assigned to each loader thread. The corresponding decoded B LDS tile is only 4 KiB; 128x64 and 128x96 forms use 8 KiB and 12 KiB. A 64-deep chunk reduces barrier/global-wait frequency by four relative to the current ordinary path. Test it only after grouped traversal controls the extra `grad_output` traffic caused by narrower N tiles.

After traversal, geometry, decoder width, and K depth are independently accepted, test two-fragment local-read prefetch and finally a second LDS buffer. The next shape targets are query Q3_K and attention-output Q4_K at M=32,768. Even a successful fused prototype cannot eliminate weight decode repetition across all M workgroups, so compare it against decode-once and persistent-cache paths rather than treating it as the only design.

### 3. Decode each ordinary weight once per call, then use hipBLASLt

The current kernel decodes the same packed weight tile again for every 64-row M workgroup. At M=32,768 that means up to 512 repetitions. A two-stage path can instead dequantize the complete ordinary weight once into a temporary BF16 workspace and then call a tuned BF16 GEMM for `dY @ W`.

The temporary BF16 workspace is modest for ordinary projections: about 32 MiB for query, 2 MiB for narrow, 16 MiB for attention output, and 2 MiB for shared down. The path should be dispatched only when M is large enough to amortize dequantization, likely starting at M=8,192 or M=32,768. The likely ceiling is close to the BF16 time plus one packed-decode pass, which could be substantially faster than the current 3-4x BF16 latency on large ordinary shapes.

This is the simplest high-ceiling design, but it is no longer an all-in-one packed MMQ kernel. Implementation options are a project-owned packed-to-BF16 kernel followed by a stable PyTorch GEMM operator, or direct hipBLASLt/rocBLAS integration in the extension. Benchmark allocation overhead and reuse the temporary workspace through the framework allocator.

### 4. Build a lossless backward int8-plus-scale cache before an approximate Q8 cache

Frozen weights permit a middle representation between packed GGUF and BF16 that was not included in the previous plan. Expand each original quant value losslessly to one signed or unsigned int8 and precompute the original floating scale metadata, but do not requantize the weight. Store the cache in backward traversal order, for example `[input_group][output_row][16 quant values]`, so one loader thread receives a contiguous sixteen-value record.

For Q3_K and Q6_K, store one FP32 `scaled_d` per 16 values. For Q4_K and Q5_K, store one FP32 quant scale and one FP32 minimum per 32 values. IQ2_S can similarly expand the grid/sign result to int8 plus one FP32 scale per 16 values. The cache then costs approximately one byte per value plus 0.25 byte of metadata, or 1.25 bytes/value. It is 37.5% smaller than BF16 while removing packed bit extraction, scale/min unpacking, indirect IQ grids, and half-to-float metadata conversion from every backward call.

This representation can reproduce the current decoder result exactly: quant integers are unchanged, and the cached FP32 scale/min values are the same products currently reconstructed before the final BF16 conversion. The 160 ordinary weights would occupy roughly 475 MiB instead of about 760 MiB as BF16. The LM head can remain on the current Q6_K path unless its extra cache memory is justified.

The first kernel for this cache should still cooperatively expand int8-plus-scale records into BF16 LDS once per workgroup, preserving cross-wave sharing. Consumer-side conversion after LDS would repeat scale application in every M wave. Compare this exact cache against transient BF16 decode-plus-GEMM and the improved fused GGUF kernel.

Only if more memory reduction is required should an approximate backward-specific Q8 transpose cache be tested. That design requantizes weights and cotangents for int8 WMMA and introduces additional error. Since int8 WMMA is not faster than BF16 WMMA on gfx1151, it should follow the lossless cache rather than precede it.

### 5. Add scheduling stages only after geometry and decoder widening

PC sampling still shows 20-27% memory waits and about 20% barrier waits. The first scheduling experiment should be Tensile-style local-read prefetch: load two B fragments before issuing two independent WMMAs and inspect the emitted `ds_load_b128`/`s_waitcnt` sequence. Next, prefetch packed bytes and metadata for the following decoder group into VGPRs. Only then test a second LDS buffer that overlaps decoded writes with WMMA consumption. A full dual-buffer or producer/consumer-wave kernel requires precise software pipelining or named barriers; merely increasing `K_ITERATION` already regressed. These are measured candidates, not assumptions, because the shipped dense logic selects both one- and two-buffer schedules on different production shapes.

### 6. Reuse forward Q8 activation workspaces across projections

Single-call forward multiplication is lower priority, but the separate activation quantizer creates a cross-call opportunity. Every dense `mmq` call currently allocates a Q8 workspace and quantizes its BF16 input even when several projections consume the same activation tensor. The final narrow profile still spends about 0.887 ms in quantization versus about 2.645 ms in multiplication.

Add either an explicit prepared-activation internal operator or a dense pair/multi-projection operator analogous to the existing grouped pair path. Projections that use the same Q8 metadata layout can share one workspace directly: Q4_K/Q5_K use the scale-plus-sum layout, while Q3_K/Q6_K/IQ2_S use the scale-only layout. If a layer needs both classes, at most two quantizations are required. For three same-layout projections, this can remove two quantizer launches and two workspace allocations without changing MMQ arithmetic or accuracy.

Prefer an explicit prepared workspace or compiler-visible pair operator over an implicit pointer cache. Any cache must account for tensor storage identity, tensor version, device, stream ordering, shape, row padding, and Q8 metadata layout. Benchmark this at the model scheduling level because the existing single-projection harness cannot show the reuse benefit.

The only clear single-call forward deficit remains Q6_K at M=256, where MMQ is 16.127 ms versus 13.053 ms BF16, or about 1.23x slower. It is not the current production chunk. A major forward rewrite would need to reduce the Q6_K `J=128` kernel's 225 architectural VGPRs without repeating the failed global `J=64` or `I=128` experiments. Narrow Q3_K and Q5_K at M=32,768 are only 5-9% behind BF16.

### Recommended order

1. Benchmark and, if memory permits, implement 128- or 256-row LM-head backward aggregation.
2. Add temporary controlled ablations for ordinary backward decode, global refill, and A/B fragment loads so subsequent work targets exposed cost rather than PC-sample counts alone.
3. Sweep grouped backward traversal (`GROUP_M=1,4,8,16,current`) on the existing narrow Q4_K and query Q3_K kernels.
4. Sweep four-wave 2x2, 2x4, and 2x6 per-wave geometries with one LDS buffer and the current pair decoder.
5. Integrate 16-value Q3_K/Q4_K/Q5_K loaders into the winning geometry, then test `K_ITERATION=64` and two-fragment local-read prefetch.
6. Prototype the transient BF16 decode-plus-GEMM path for M=32,768 and compare total latency and workspace cost against the improved fused path.
7. Prototype the lossless backward int8-plus-scale cache for dominant ordinary weights; test approximate Q8 only if its additional memory reduction is needed.
8. Add explicit forward Q8-workspace reuse for same-input dense projections and benchmark it at the model scheduling level.
9. Attempt a second LDS buffer or named-barrier pipeline only after the traversal, geometry, decoder, and representation experiments.

## Experiments rejected

The following candidates were measured and reverted when they were slower or failed to improve the target shapes:

- global forward `J=64`;
- forward `I=128`;
- alternative smaller forward thread/tile combinations;
- larger backward wave counts;
- additional row padding;
- shuffle-based replacements for LDS fragment loads;
- oversized backward N tiles that reduced workgroup count too far;
- Q6_K `N_TILES=2` at 64 rows;
- Q6_K `N_TILES=3` at 128 rows;
- Q6_K `K_ITERATION=128` at 64 and 128 rows;
- Q6_K `K_ITERATION=32` at 64 rows;
- Q6_K `K_ITERATION=64` at 256 rows;
- eight-value Q6_K decode at 64 rows, where it underfilled the 128-thread loader.

The accepted shape heuristics are measured dispatches, not an autotuning system.

## Correctness and compatibility

Validation completed after the current source changes:

```text
pytest -q tests/
38 passed
```

This includes dense MMQ, dense backward, grouped MMQ compatibility, FakeTensor, autograd, opcheck, and compile-composition coverage.

Forward normalized RMSE remains in the existing Q8_1 envelope:

- approximately 0.6% for Q3_K and Q6_K;
- approximately 1.1-2.0% for Q4_K/Q5_K.

Backward computes against the authoritative packed payload without cotangent quantization. Production benchmark NRMSE is generally below `5.2e-4` and often much lower. A direct check of four complete Q6_K rows showed that the sixteen-value project-owned decoder produced BF16 values exactly equal to the reference dequantizer for all 8,192 checked values.

The final dense code object uses BF16 WMMA for backward and int8 WMMA through the inherited forward templates. Dense specializations have no private-memory spill segment.

## Profiler and tool issues

- PC sampling perturbs short kernels heavily. The narrow Q4_K backward call was several times slower under sampling, so stall percentages are qualitative. The longer Q6_K sample had much lower relative perturbation.
- After one PC-sampling session, the first subsequent full benchmark process encountered a transient `hipErrorLaunchFailure`. A fresh small test and a rerun succeeded without source changes.
- The `roc-obj-ls` entry point in the active environment fails with an import error from `rocm_sdk_core._cli`. Code-object metadata was inspected by dumping `.hip_fatbin`, extracting the embedded gfx1151 ELF, and using LLVM `llvm-readobj`/`llvm-objdump` instead.
