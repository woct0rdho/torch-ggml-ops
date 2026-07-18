# Dense MMQ backward optimization status and plan

## Scope

This document covers dense `torch_ggml_ops::mmq_grad_input` backward on gfx1151.

Included:

- BF16 cotangents;
- packed GGUF Q3_K, Q4_K, Q5_K, Q6_K, and IQ2_S weights;
- BF16 input gradients;
- the 160 ordinary model projections;
- the packed Q6_K language-model head;
- production batch sizes 1, 4, and 16 at sequence length 2048.

Excluded:

- `grouped_mmq` backward and routed-expert scheduling;
- GatedDeltaNet physical-layout permutations;
- LoRA GEMMs and residual accumulation;
- public operator-schema changes;
- changes to `csrc/vendor/llama_cpp/*`;
- direct linkage of the fused packed kernel against hipBLASLt.

## Hardware and measurement rules

Measurements were taken on:

```text
GPU: Radeon 8060S Graphics
architecture: gfx1151, wave32, 40 CUs
LDS limit: 64 KiB per workgroup, 128 KiB per WGP
VGPR capacity: 1536 per SIMD
PyTorch: 2.12.0+rocm7.15.0a20260701
HIP: 7.14.60850
```

The reference is `torch.mm` using the same BF16 cotangent and the logical GGUF weight dequantized to BF16. PyTorch normally dispatches these calls to hipBLASLt.

The first forward and backward baselines were mistakenly run concurrently and were discarded. Accepted timings come from sequential runs with no concurrent GPU benchmark or profiler.

On gfx1151, int8 WMMA is approximately as fast as BF16 WMMA. An int8 design must win through representation size, decode removal, LDS/VGPR pressure, or scheduling rather than nominal arithmetic throughput.

Use real nonzero benchmark data. Zero-valued WMMA operands can produce misleading results.

## Benchmark harness and artifacts

The backward benchmark is:

```bash
source ~/venv_torch/bin/activate
python bench/benchmark_mmq_bwd.py
```

Focused LM-head example:

```bash
python bench/benchmark_mmq_bwd.py \
  --cases lm_head_q6_k --lm-head-chunks 64,128,256 \
  --warmup 3 --repeats 9
```

Important artifacts:

```text
Sequential baseline:             /tmp/mmq_bwd_baseline_primary_sequential.json
First redesigned full result:    /tmp/mmq_bwd_final_full_v3.json
Grouped-M traversal result:      /tmp/mmq_bwd_ordinary_group_2.json
128x128 ordinary geometry:       /tmp/mmq_bwd_ordinary_final_geometry.json
Q4_K LDS-padding experiment:     /tmp/mmq_bwd_q4_lds_pad8.json
Q3_K/Q5_K padding experiment:    /tmp/mmq_bwd_q3_q5_lds_pad8.json
Final Q6_K small-row geometry:   /tmp/mmq_bwd_lm_final_geometry.json
```

The `/tmp` paths record measurement provenance and are not repository inputs.

## Production shapes

For ordinary projections, `M = batch * 2048`:

| Batch | M |
| ---: | ---: |
| 1 | 2,048 |
| 4 | 8,192 |
| 16 | 32,768 |

Representative packed weights:

| Family | Forward `(N, K)` | Weight types | Model tensors |
| --- | ---: | --- | ---: |
| Query plus query gate | `(8192, 2048)` | Q3_K, Q4_K | 10 |
| Key/value/shared gate/up | `(512, 2048)` | Q3_K, Q4_K, Q5_K | 100 |
| Attention output | `(2048, 4096)` | Q4_K | 10 |
| Shared-expert down | `(2048, 512)` | Q4_K, Q5_K | 40 |

Backward computes the transposed operation `dY @ W` without materializing a transposed dense weight.

The LM head uses:

```text
N = 248320
K = 2048
weight = Q6_K
production backward chunk M = 64
```

Comparison and aggregation candidates use `M = 64, 128, 256`.

## Current source status

The dense kernel is in `csrc/ck/mmq_backward.cuh`.

The current source contains uncommitted work after the committed 128x128 ordinary retile:

- final Q6_K small-row geometries for M=64, 128, and 256;
- configurable decoded-weight LDS row padding;
- accepted padding for full-tile Q3_K;
- accepted padding for Q4_K when `in_features >= 2048`;
- unpadded Q5_K and small shared-down Q4_K dispatches.

The source was rebuilt after reverting the slower Q5_K padded path.

That rebuild exposed a correctness bug in the new Q6_K small-row dispatch. The production-only full-tile flag was selected for arbitrary bounds-safe test shapes, causing the exact Jacobian test with 10 rows and 37 output features to fail.

The dispatch now guards every full-tile specialization by exact row and feature divisibility. The focused exact-Jacobian suite passes for all five packed types.

This also revealed that the earlier M=256 N=7 timing was invalid: 112 input columns per workgroup do not divide 2048, so the full-tile kernel accessed the final N tile out of bounds. The corrected bounds-safe N=7 kernel measures 29.628 ms rather than 18.823 ms. M=256 geometry must be reselected before final validation.

## Current kernel design

The templated kernel configuration is:

```text
<type,
 N_TILES,
 K_ITERATION,
 GROUP_M,
 M_TILES_PER_WAVE,
 DECODER_WIDTH,
 PREFETCH_LOCAL,
 FULL_TILES,
 PREFETCH_PACKED,
 LDS_PADDING>
```

The kernel:

- uses four wave32 waves in a 128-thread workgroup;
- retains multiple 16x16 WMMA accumulators per wave;
- cooperatively decodes packed weights into BF16 LDS;
- loads BF16 A fragments directly from the cotangent;
- shares each decoded B tile across four waves;
- uses shape- and type-specific decoder width and geometry;
- avoids cotangent quantization;
- writes BF16 input gradients directly;
- has zero private segment in every accepted specialization inspected so far.

## Current dispatch

### Full-tile Q3_K, Q4_K, and Q5_K ordinary path

For rows above 256 with exact production divisibility:

```text
four waves
2 M tiles per wave
8 N tiles per wave
logical workgroup tile: 128x128
K iteration: 32
GROUP_M: 1
16-value decoder
paired LDS-fragment prefetch
optional packed-byte prefetch
```

Format-specific choices:

| Type/shape | Packed-byte prefetch | LDS padding |
| --- | --- | ---: |
| Wide Q3_K query | yes, with packed quant-byte extraction | XOR swizzle, no padding |
| Narrow Q3_K | no | 8 BF16 values per row |
| Q4_K query/narrow | yes | XOR swizzle, no padding |
| Attention-output Q4_K | yes | 8 BF16 values per row |
| Shared-down Q4_K | yes | 16-BF16-chunk XOR swizzle, no padding |
| Narrow Q5_K | yes | 8-BF16-chunk XOR swizzle plus packed quant-byte extraction |
| Shared-down Q5_K | yes | 4-BF16-chunk XOR swizzle, no padding |

Non-divisible and smaller-row shapes retain bounds-safe measured kernels.

### Q6_K small-row path

| Rows | M tiles/wave | N tiles/wave | K iteration | Logical tile |
| ---: | ---: | ---: | ---: | --- |
| `<=64` | 1 | 2 | 64 | 64x32 |
| `<=128` | 2 | 4 | 32 | 128x64 |
| `<=256` | 2 | 4 | 32 | 128x64 |
| `<=2048` | 1 | 8 | 16 | 64x128 |
| larger | 1 | 16 | 16 | 64x256 |

The small-row kernels use sixteen-value Q6_K decoding and paired local-fragment prefetch. M=64, M=128, and M=256 use exact full-tile paths on production dimensions. Non-production shapes use the same geometries with bounds checks.

M=64 uses a 16-BF16-chunk XOR LDS swizzle, while M=128 uses an eight-BF16-chunk swizzle. Row-dependent chunk permutation distributes 128-bit fragment loads across bank phases without increasing LDS footprint.

M=256 now reuses the same 128x64 swizzled workgroup tile as M=128. Two M workgroups cover the 256 rows, producing 64 workgroups instead of the previous 32-workgroup 128x128 launch.

Its Q6_K decoder additionally combines four low/high byte pairs into one packed six-bit quant word before BF16 conversion. M=64 and M=128 retain scalar extraction because the packed schedule was neutral or slightly slower there.

### Other bounds-safe ordinary paths

- rows `<=128`: one N tile;
- rows `<=256`: four N tiles;
- rows `<=2048`: type- and shape-specific four, eight, or sixteen N tiles;
- rows `<=8192`: usually twelve N tiles, with sixteen for Q4_K/Q5_K at `in_features == 2048`;
- larger rows that are not eligible for the full-tile specialization: sixteen N tiles.

## Experiment log

### Initial backward redesign

The original kernel decoded one packed BF16 value at a time into a small 16x16 tile. It had little M/N reuse and sustained only about 0.05-0.13x BF16 throughput on most production shapes.

The first accepted redesign introduced:

- four wave32 waves;
- LDS-staged decoded weights;
- multiple WMMA accumulator tiles;
- pair, quad, and sixteen-value decoders;
- measured row/type dispatch;
- Q6_K scale and bit extraction shared across adjacent values.

Representative improvement from the sequential baseline to `/tmp/mmq_bwd_final_full_v3.json`:

| Case | M | Baseline ms | First redesign ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Query Q3_K | 32,768 | 1,108.314 | 162.884 | 6.80x |
| Narrow Q4_K | 32,768 | 48.332 | 11.130 | 4.34x |
| Attention output Q4_K | 32,768 | 487.945 | 80.633 | 6.05x |
| Shared down Q4_K | 32,768 | 66.903 | 14.095 | 4.75x |
| LM head Q6_K | 256 | 158.209 | 26.130 | 6.05x |

The custom sixteen-value Q6_K decoder exactly matched the BF16 reference dequantizer for all 8,192 checked values.

### Grouped M traversal

The original grid ran all M blocks for one N block before advancing N. A grouped three-dimensional mapping was tested:

```text
grid = [GROUP_M, n_blocks, ceil(m_blocks / GROUP_M)]
m_block = blockIdx.z * GROUP_M + blockIdx.x
n_block = blockIdx.y
```

`GROUP_M=1,2,4,8,16` and the old all-M ordering were benchmarked sequentially.

For the earlier 64-row workgroup geometry, `GROUP_M=2` was the best compromise:

| Case, M=32,768 | Before | `GROUP_M=2` | Improvement |
| --- | ---: | ---: | ---: |
| Query Q3_K | 162.884 ms | 112.026 ms | 1.45x |
| Narrow Q4_K | 11.130 ms | 7.051 ms | 1.58x |
| Attention output Q4_K | 80.633 ms | 55.768 ms | 1.45x |
| Shared down Q4_K | 14.095 ms | 10.458 ms | 1.35x |

Traversal is geometry-dependent. The later 128-row workgroup tile prefers `GROUP_M=1`.

### Ordinary 128x128 geometry and decoder sweep

The accepted full-tile ordinary geometry uses two M tiles and eight N tiles per wave with `K_ITERATION=32`.

Narrow Q4_K M=32,768 sequence:

| Experiment | Median ms | Outcome |
| --- | ---: | --- |
| Earlier grouped 1x16 geometry | 7.053 | reference |
| 2x2, pair decoder, K=16 | 11.424 | rejected |
| 2x4, pair decoder, K=16 | 5.845 | improved |
| 2x6, pair decoder, K=16 | 5.514 | improved |
| 2x6, `GROUP_M=1` | 5.364 | improved |
| 2x6, sixteen-value decoder, K=16 | 4.274 | improved |
| 2x6, sixteen-value decoder, K=64 | 5.120 | rejected |
| 2x6, sixteen-value decoder, K=32 | 3.855 | improved |
| 2x4 / 2x5 / 2x7 / 2x8 at K=32 | 4.692 / 4.169 / 3.732 / 3.683 | 2x8 selected |
| Eight-value decoder | 4.745 | rejected |
| Paired LDS-fragment prefetch | 3.601 | improved |
| Exact full-tile specialization | 3.399 | improved |
| Two-row packed-byte prefetch | 3.293 | improved |
| Final clean pre-padding dispatch | 3.273 | accepted milestone |

Additional rejected experiments:

- 3x6 and 4x4 M/N tiles;
- a 1x16 K=32 schedule;
- `GROUP_M=2` and `GROUP_M=4` on the final 128-row geometry;
- eight-value Q3_K/Q4_K/Q5_K decoding;
- full vector loading of the Q4_K metadata header;
- replacing `__syncthreads()` with an LDS-only inline-assembly barrier;
- runtime environment-driven production dispatch.

The inline-assembly barrier removed `buffer_gl0_inv` but was neutral or slower. It was removed.

### Packed-byte prefetch

Q3_K, Q4_K, and Q5_K gained explicit fixed `uint4` packed-byte prefetch helpers.

Q4_K and Q5_K preload packed data for two output rows before decoding the first. Wide Q3_K uses the analogous path.

Wide Q3_K now combines four low/high byte pairs into one unsigned three-bit quant word before scalar BF16 conversion. This reduced query latency from 48.694 to 46.980 ms without changing correctness.

Narrow Q3_K does not use packed prefetch. Its misaligned 110-byte block layout made the prefetch slightly slower than the regular sixteen-value decoder.

Fixed scalar/vector prefetch state was retained. Compiler-managed arrays were avoided because FeatherOps experiments showed that intended VGPR arrays can become unexpected LDS allocations.

### LDS padding and bank-layout experiment

Counter profiling of the pre-padding Q4_K kernels reported a repeated 79.2% LDS-bank-conflict stall percentage.

The decoded B layout is logically:

```text
shared_b[input_column][k]
```

Without padding, K=32 gives a 64-byte LDS row stride. On 32 four-byte banks, row starts alternate between only two bank phases.

Eight BF16 values of row padding change the stride to 80 bytes. This preserves 16-byte alignment while distributing row starts over more banks.

Q4_K M=32,768 results:

| Shape | Unpadded | Padded | Speedup |
| --- | ---: | ---: | ---: |
| Query | 54.895 ms | 46.295 ms | 1.19x |
| Narrow | 3.273 ms | 2.907 ms | 1.13x |
| Attention output | 26.875 ms | 23.177 ms | 1.16x |
| Shared down | 5.261 ms | 5.389 ms | regression |

Padding is selected for Q4_K only when `in_features >= 2048`.

Q3_K also improved:

| Shape | Unpadded | Padded | Speedup |
| --- | ---: | ---: | ---: |
| Query | 53.879 ms | 49.803 ms | 1.08x |
| Narrow | approximately 3.316 ms | 3.138 ms | 1.06x |

Q5_K did not benefit from padding. Narrow Q5_K regressed from 3.378 to 3.622 ms, so padded Q5_K remains rejected.

A later no-padding XOR swizzle sweep showed that the result is shape-specific. Query Q3_K/Q4_K measured 49.040/47.454 ms, narrow Q4_K/Q5_K measured 2.969/3.114 ms, and those cases improved relative to the immediately preceding selected dispatch.

The eight-BF16-chunk swizzle regressed narrow Q3_K to 3.093 ms, attention-output Q4_K to 22.812 ms, and both shared-down formats to 5.401/5.651 ms. The selected dispatch uses that granularity only for wide Q3_K query, Q4_K query/narrow, and narrow Q5_K; padded vector loads remain selected for narrow Q3_K and attention-output Q4_K.

A coarser 16-BF16-chunk swizzle regressed the high-frequency Q3_K query from 48.694 to 57.001 ms. A finer four-BF16 swizzle with four 64-bit fragment loads also regressed to 50.873 ms. Its selected eight-BF16 granularity is retained.

The 16-BF16 swizzle also regressed Q4_K query/narrow from 47.083/2.915 ms to 53.638/3.420 ms. A four-BF16 swizzle was closer at 48.051/2.989 ms but still slower. Their selected eight-BF16 granularity is retained.

A follow-up 16-BF16-chunk swizzle improved shared-down Q4_K from 5.316 to 5.166 ms in the initial run and 5.252 ms in the final selected dispatch. It regressed attention-output Q4_K to 26.526 ms, while a four-BF16 swizzle also trailed its padded layout at 22.952 ms.

Shared-down Q5_K and the original Q6_K M=256 geometry were neutral with the coarser swizzle. The 16-BF16 variant is accepted only for shared-down Q4_K and Q6_K M=64.

Before packed Q3_K/Q5_K quant extraction, the selected ordinary run measured 48.694/47.083 ms for Q3_K/Q4_K query, 2.691/2.915/3.065 ms for narrow Q3_K/Q4_K/Q5_K, and 22.402 ms for attention-output Q4_K. The later packed extraction changes reduced Q3_K query to 46.980 ms and narrow Q5_K to 2.868 ms.

### LDS vectorization inspection

The unpadded ordinary Q4_K kernel emits 32 `ds_load_b128` instructions for B fragments.

The padded Q4_K kernel emits:

```text
16 ds_load_b128
64 ds_load_u16_d16
64 ds_load_u16_d16_hi
```

Padding improves overall speed despite scalarizing half of the fragment loads. The element-wise padded-storage indexing prevents the compiler from recognizing every 32-byte BF16 fragment as two aligned 128-bit loads.

The decoded-weight stores remain scalar 16-bit stores. One decoder thread produces 16 adjacent input columns for one K position, but the selected LDS representation stores K contiguously within each input column.

Those stores are an intentional transpose. Vectorizing them requires a register/lane transpose or a different producer mapping.

Because each value is stored once and then loaded by four waves, restoring fully vectorized fragment loads was the next measured experiment.

An explicit aligned helper restored the padded and unpadded kernels to two `ds_load_b128` operations per 16-BF16 fragment. The result was strongly shape-dependent rather than universally positive.

At M=32,768, narrow Q3_K improved 3.138 to 2.739 ms, attention-output Q4_K improved 23.177 to 22.802 ms, and query Q3_K improved slightly from 49.803 to 49.392 ms. Query Q4_K regressed 46.295 to 48.976 ms, while Q5_K and shared-down shapes also regressed by about 2-4%.

The likely cause is that wider transactions reduce instruction count but change the bank-conflict and wait schedule. The Q6_K M=64/128/256 results were neutral at 10.945/13.449/20.795 ms versus 10.938/13.402/20.834 ms.

A typed selection was added for Q3_K and attention-output Q4_K, but the first scalar fallback still used direct physical indexing. The compiler recognized that fallback as contiguous and continued to emit fully vectorized loads, so Q4_K query remained regressed at 48.805 ms.

The logical-indexed fallback restored the prior mixed padded ISA: Q4_K controls again emit 16 `ds_load_b128` operations plus 128 scalar 16-bit loads, while selected Q3_K and attention-output Q4_K emit 32 `ds_load_b128` operations.

The final selective run measured 49.324 ms for query Q3_K, 2.691 ms for narrow Q3_K, and 22.437 ms for attention-output Q4_K. Q4_K query/narrow measured 48.716/3.071 ms in the same run, so their earlier 46.295/2.907 ms padding result needs a repeated control before attributing the difference to the vector-load implementation.

### Q6_K small-row geometry sweep

The initial M=256 M=2/N=6/K=32 candidate improved 26.082 to 22.958 ms.

The M=256 neighborhood sweep was:

| Configuration | Time ms | Outcome |
| --- | ---: | --- |
| M=2/N=5/K=32 | 19.204 | invalid full-tile measurement; N width does not divide 2048 |
| M=2/N=7/K=32 | 18.861 | invalid full-tile measurement; corrected bounds-safe time is 29.628 ms |
| M=2/N=8/K=32 | 20.834 | selected valid exact-tile geometry |
| M=2/N=7/K=16 | 24.588 | rejected |
| M=2/N=7/K=64 | 25.631 | rejected |
| M=2/N=7/K=32, `GROUP_M=1` | 18.827 | neutral |

The original conclusion that N=7 won was invalid because the last N workgroup ran outside the logical 2048-column input gradient. N=8 is the selected valid geometry at 20.834 ms. K=16 still underfills the loader, while K=64 lengthens the decode phase and uses about 14 KiB LDS.

For M=128:

| Configuration | Time ms | Outcome |
| --- | ---: | --- |
| Previous M=1/N=2/K=64 | 18.618 | reference |
| M=2/N=3/K=32 | 19.448 | rejected |
| M=2/N=3/K=64 | 20.760 | rejected |
| M=2/N=4/K=32 | 13.443 | selected |

The selected M=128 geometry creates exactly 128 sixteen-value decoder groups per K chunk. Every loader thread receives one group.

For M=64:

| Configuration | Time ms | Outcome |
| --- | ---: | --- |
| Previous N=1/K=64 quad decoder | 12.599 | reference |
| N=2/K=64 sixteen-value decoder | 10.892 | selected |
| N=4/K=32 | 11.416 | rejected |

The final clean Q6_K benchmark is:

| M | Selected geometry | Packed ms | BF16 ms | Throughput ratio |
| ---: | --- | ---: | ---: | ---: |
| 64 | M=1/N=2/K=64 with 16-BF16-chunk XOR LDS swizzle | 5.356 | 9.850 | 1.84x |
| 128 | M=2/N=4/K=32 with XOR LDS swizzle | 9.172 | 14.444 | 1.57x |
| 256 | two M blocks of M=2/N=4/K=32, packed quant extraction, XOR swizzle | 11.758 | 19.023 | 1.62x |

The XOR LDS swizzle changed M=64 from the largest Q6_K deficit into the fastest production-relative kernel at 1.84x BF16 throughput. The final 16-BF16-chunk layout measures about 5.36 ms, with unchanged 4 KiB LDS capacity and exact benchmark correctness.

M=128 improved from 13.402 to 9.187 ms and reached 1.57x BF16. Applying the swizzle to the eight-N-tile M=256 geometry regressed, but reusing the four-N-tile swizzled geometry with two M workgroups reduced 20.894 to 12.156 ms and also reached 1.57x BF16.

The smaller N tile doubles workgroup count and reduces accumulator pressure. Its repeated packed decode across two M blocks costs less than the parallelism and LDS-bank gains it enables.

Packed Q6_K quant-byte extraction measured 5.361/9.370/11.754 ms at M=64/128/256 when enabled globally. Shape-specific selection restored the scalar M=64/M=128 times and retained the M=256 gain, with a final 11.758 ms result.

A post-swizzle M=64 N=3/K=64 candidate regressed from 5.829 to 12.690 ms. Its 48-column tile does not divide 2048, so bounds checks on every workgroup and the irregular final tile overwhelm its extra N reuse.

The exact N=4/K=64 candidate also regressed to 10.258 ms because its 32 workgroups do not cover the 40 CUs and accumulator pressure rises. A finer four-BF16-chunk swizzle regressed the selected N=2 geometry to 7.880 ms because four 64-bit fragment loads cost more than the additional bank distribution saves.

A coarser 16-BF16-chunk swizzle then improved the exact N=2 geometry from 5.829 to 5.365 ms. It remains selected.

## Latest results

### Ordinary M=32,768

The latest accepted measurements combine the 128x128 retile, packed-byte prefetch, full-tile specialization, and selected LDS padding.

| Case | Packed ms | BF16 ms | Throughput ratio | Source |
| --- | ---: | ---: | ---: | --- |
| Query Q3_K | 46.980 | 59.411 | 1.26x | packed quant-byte extraction and XOR-swizzled vector loads |
| Query Q4_K | 47.083 | 59.468 | 1.26x | XOR-swizzled vector local loads |
| Narrow Q3_K | 2.691 | 2.802 | 1.04x | padded vector local loads |
| Narrow Q4_K | 2.915 | 2.894 | 0.99x | XOR-swizzled vector local loads |
| Narrow Q5_K | 2.868 | 2.959 | 1.03x | packed quant-byte extraction and 8-BF16-chunk XOR swizzle |
| Attention output Q4_K | 22.402 | 22.832 | 1.02x | padded vector local loads |
| Shared down Q4_K | 5.252 | 4.261 | 0.81x | 16-BF16-chunk XOR-swizzled vector loads |
| Shared down Q5_K | 5.618 | 4.138 | 0.74x | scalar quant extraction and 4-BF16-chunk XOR swizzle |

The ordinary batch-16 serial estimate is now approximately 1.19-1.20 seconds across the 160 projections. The corresponding BF16 estimate is approximately 1.28 seconds across the recent runs.

This is an aggregate scheduling estimate, not an end-to-end training measurement. It does not model overlap with other model work.

### Pre-padding full shape matrix

`/tmp/mmq_bwd_ordinary_final_geometry.json` remains the latest complete ordinary 2,048/8,192/32,768 matrix before LDS padding:

| Case | M=2,048 | M=8,192 | M=32,768 | BF16 ratio at M=32,768 |
| --- | ---: | ---: | ---: | ---: |
| Query Q3_K | 3.622 ms | 13.792 ms | 53.879 ms | 1.11x |
| Query Q4_K | 4.836 ms | 14.574 ms | 54.895 ms | 1.08x |
| Narrow Q4_K | 0.300 ms | 0.907 ms | 3.273 ms | 0.86x |
| Narrow Q5_K | 0.253 ms | 0.914 ms | 3.378 ms | 0.83x |

A new full matrix is required after rebuilding the current padded source.

### LM-head aggregation implications

At batch 16:

| Chunk M | Calls | Packed serial estimate | BF16 serial estimate |
| ---: | ---: | ---: | ---: |
| 64 | 512 | 2,742 ms | 5,043 ms |
| 128 | 256 | 2,348 ms | 3,697 ms |
| 256 | 128 | 1,505 ms | 2,435 ms |

M=256 now has the lowest batch-16 serial estimate at about 1.50 seconds, but it requires about 121 MiB of cotangent storage. M=128 remains a lower-memory 61 MiB aggregation point, while the production M=64 schedule needs no additional aggregation buffer.

Approximate BF16 cotangent storage is:

| Chunk M | Storage |
| ---: | ---: |
| 64 | 30 MiB |
| 128 | 61 MiB |
| 256 | 121 MiB |

Aggregation belongs at the loss/LM-head scheduling layer rather than inside the packed kernel.

## Profiling and resource findings

### Ordinary Q4_K before padding

Narrow Q4_K at M=32,768 reported:

```text
192 VGPRs
128 SGPRs
8 KiB LDS
0 scratch
89.4% L2 hit rate
79.2% LDS-bank-conflict stall percentage
about 7,413 VALU instructions per work-item
```

Attention-output Q4_K reported the same resource counts and 79.2% LDS-bank-conflict percentage. Its L2 hit rate was 85.4% with about 26,517 VALU instructions per work-item.

Shared-down Q4_K also reported the same LDS conflict percentage and VALU count. Its L2 hit rate fell to 67.3%, and it has only four N workgroups per M tile.

The repeated conflict value indicates a structural fragment-access pattern. Shared down additionally suffers weaker packed-weight locality and limited N-direction parallelism.

### Q6_K small rows

M=64 reported:

```text
88 VGPRs
128 SGPRs
4 KiB LDS
0 scratch
69.8% L2 hit rate
85.3% LDS-bank-conflict stall percentage
about 946,927 VALU instructions per work-item
```

M=128 reported:

```text
112 VGPRs
128 SGPRs
4 KiB LDS
0 scratch
66.6% L2 hit rate
75.0% LDS-bank-conflict stall percentage
about 1,979,376 VALU instructions per work-item
```

M=128 wins despite higher VGPR and VALU counts because it doubles M reuse and exactly matches 128 decoder groups to the loader.

The equivalent multi-counter M=256 profile triggered a profiler-induced HSA memory fault and queue-sync timeouts. The clean benchmark completed correctly immediately beforehand, so no M=256 counters are accepted from that run.

### Earlier PC-sampling evidence

PC sampling of the earlier narrow Q4_K kernel showed non-issued samples dominated by:

| Reason | Share |
| --- | ---: |
| ALU dependency | 40.3% |
| waitcnt / memory dependency | 27.2% |
| barrier wait | 20.2% |
| execution-pipe arbitration | 10.7% |

Earlier Q6_K M=256 sampling showed:

| Reason | Share |
| --- | ---: |
| ALU dependency | 36.0% |
| waitcnt / memory dependency | 23.1% |
| barrier wait | 22.2% |
| execution-pipe arbitration | 18.5% |

Multi-value decoding removed the hottest scalar bit-extraction bottleneck. The remaining limit shifted toward barriers, LDS-to-fragment movement, and WMMA dependency chains.

PC sampling heavily perturbs short kernels. Treat its percentages qualitatively and prefer controlled ablations for performance decisions.

## Current bottleneck assessment

### Ordinary Q3_K and Q4_K

Wide query is now 1.26x BF16 after packed Q3_K extraction. Narrow Q4_K is at parity, and attention output is within measurement noise.

The remaining Q3_K/Q4_K issue is no longer packed-global bandwidth alone. High L2 hit rates and the successful padding ablation show that LDS bank layout and fragment-load instruction form matter materially.

### Q5_K

Packing four Q5_K low/high byte pairs into one 32-bit quant word reduced repeated per-byte high-bit extraction. Narrow Q5_K improved from about 3.04 to 2.845-2.868 ms and now reaches 1.03-1.05x BF16 throughput.

The same extraction schedule regressed shared down to 5.759 ms with its four-BF16 swizzle. Pairing packed extraction with the eight-BF16 swizzle still measured 5.695 ms, above the selected scalar-extraction path.

Packed extraction is therefore selected only for the narrow shape. Shared-down Q5_K still needs lower decode issue pressure or cross-call representation reuse.

### Shared down

Shared-down Q4_K/Q5_K remain around 0.76-0.79x BF16.

They combine:

- only four N workgroups per M tile;
- lower L2 hit rate;
- structural LDS fragment cost;
- insufficient arithmetic amortization for the added padded footprint.

A shared-down-specific 4x4 per-wave geometry kept 16 accumulator tiles while doubling M reuse of decoded weights. It regressed Q4_K/Q5_K from about 5.25/5.57 ms to 7.338/7.388 ms because doubled cotangent traffic and four live A fragments outweighed reduced decode repetition.

The complementary 1x16 geometry halved cotangent reloads but doubled packed decode across M blocks and lost the N=8 packed-prefetch specialization. It regressed Q4_K/Q5_K further to 9.208/7.322 ms.

Increasing shared-down reduction depth from 32 to 64 measured 5.215 ms for Q4_K, below the meaningful margin over the selected K=32 result, and regressed Q5_K to 6.565 ms. Halving barrier frequency does not offset the longer decode phase and loss of the K=32 packed-prefetch specialization.

A finer four-BF16-chunk XOR swizzle with four 64-bit loads per fragment regressed Q4_K to 5.486 ms. For Q5_K, the initial shared-down result was 5.353 ms, while an immediate repeat measured 5.560 versus 5.642 ms for the eight-BF16 control. Narrow Q5_K measured 3.025 versus 3.043 ms in the same comparison.

The shared-down Q5_K gain is modest but repeatable. The narrow difference is below the meaningful selection margin.

Q4_K retains the 16-BF16-chunk/two-128-bit-load swizzle for shared down. Shared-down Q5_K selects the finer swizzle, while narrow Q5_K retains two 128-bit loads.

Local geometry and K-depth tuning have plateaued for shared down. A larger gain likely requires decoded-weight reuse across calls or a changed representation.

### Q6_K small rows

M=64 was limited by its K=64 decoded-weight row stride of 128 bytes, which mapped every row start to the same bank phase. A 16-BF16-chunk XOR swizzle reduced 10.938 ms to about 5.38 ms without increasing LDS footprint and while preserving two 128-bit loads per fragment.

This confirms that the earlier 85.3% conflict percentage represented a first-order bottleneck. The same swizzle is accepted for M=128 at about 9.17 ms. It regressed the original eight-N-tile M=256 geometry, but pairing it with the four-N-tile geometry improved M=256 to 12.156 ms. Packed quant extraction then reduced M=256 further to 11.758 ms.

Bank distribution must be selected with workgroup count, accumulator pressure, and decode repetition rather than by K depth alone.

## TensileLite and hipBLASLt lessons that remain relevant

TensileLite is a source of measured architectural patterns, not a direct packed-MMQ generator. Dense kernels assume typed affine operands, while MMQ reconstructs GGUF values and metadata inside the reduction loop.

Runtime logging identified two selected gfx1151 BF16 families:

| Reference shapes | Family | Key geometry |
| --- | --- | --- |
| Query, shared down, LM head | `MT96x96x32` | `MIWaveTile=3x3`, `MIWaveGroup=2x2`, two LDS buffers |
| Narrow, attention output | `MT128x32x32` | `MIWaveTile=2x2`, `MIWaveGroup=4x1`, one LDS buffer, 8-element LDS padding |

Both use:

- four wave32 waves;
- `DepthU=32`;
- scheduled global and local prefetch;
- conventional global-to-VGPR-to-LDS staging;
- wide local reads;
- `SourceSwap=true`;
- no DirectToLds or DirectToVgpr.

Transferred and already validated ideas:

- four-wave workgroups;
- multi-M and multi-N WMMA tiles per wave;
- `K_ITERATION=32` for the accepted ordinary tile;
- paired local-fragment prefetch;
- one LDS buffer as the default;
- eight-element LDS padding where measured;
- favorable transposed output orientation;
- fine-grained geometry-specific dispatch.

Still relevant:

- explicit two-stage packed-byte prefetch into fixed VGPR state;
- fully vectorized local reads after padding;
- a second LDS buffer only after the one-buffer load path is efficient;
- low-level instruction scheduling if HIP cannot retain the desired load/wait/WMMA order.

Not directly transferable:

- DirectToLds cannot bypass GGUF reconstruction;
- DirectToVgpr does not distribute one cooperatively decoded tile efficiently to four waves;
- Stream-K and GlobalSplitU do not remove repeated decode;
- gfx1151 lacks the gfx1250 WMMA arb-stall control;
- dense solution databases cannot describe packed decode inside the reduction loop.

## FeatherOps lessons that remain relevant

- Use controlled decode/global/LDS/fragment-load ablations. PC samples alone have repeatedly failed to predict speed.
- Keep producer-side cooperative GGUF decode. Consumer-side decode would repeat scale and bit extraction in every M wave.
- Keep packed prefetch state in explicit scalar or `uint4` VGPR values.
- Treat traversal, geometry, K depth, and decoder width as one measured configuration.
- Use lossless int8-plus-scale representation before approximate int8 WMMA if a persistent cache becomes acceptable.
- Verify code-object VGPR, LDS, and private-segment metadata after every layout change.

## Next plan

### Finalize the selected M=64 swizzle

Rebuild the selected 16-BF16-chunk M=64 source, repeat its benchmark, and run the complete test suite, benchmark compilation, and diff checks before committing.

Inspect the final code object if the larger swizzle changes VGPR, LDS, or private-segment metadata.

### Aggregate LM-head cotangent chunks

At model scheduling level, evaluate M=128 and M=256 aggregation under the 61 MiB and 121 MiB cotangent-buffer budgets.

M=256 now has the lowest serial estimate at about 1.50 seconds for batch 16 and runs at 1.62x BF16 throughput. M=128 remains the lower-memory aggregation option.

### Address shared-down through representation or cross-call reuse

Do not repeat ordinary geometry or K-depth sweeps that have plateaued.

The next high-ceiling options are:

- transient packed-to-BF16 decode followed by a project-owned dense stage;
- a persistent lossless int8-plus-scale cache;
- a decoded-weight cache shared across repeated backward calls.

Project fused kernels should remain independent of hipBLASLt. hipBLASLt remains an architectural reference unless the linkage policy changes.

### Reduce shared-down Q5_K representation cost

Narrow Q5_K now beats BF16 after packed quant-byte extraction. Shared-down Q5_K still trails BF16 because that schedule interacts poorly with its low-reuse shape.

A useful next experiment must reduce decode cost without increasing its register or LDS-fragment burden. Repeating Q4_K layout changes is unlikely to help.

### Consider a lossless int8-plus-scale cache before approximate Q8

Expand each original quant integer losslessly to int8 and cache the original FP32 scale metadata in backward traversal order.

Approximate cost is 1.25 bytes per value. The 160 ordinary weights would require roughly 475 MiB, versus approximately 760 MiB in BF16.

This removes packed bit extraction and metadata unpacking while preserving the current decoded BF16 result. Approximate int8 WMMA remains later because gfx1151 has no raw int8 WMMA throughput advantage.

### Consider a second LDS buffer only with profile evidence

Explicit vector loads and swizzles are already selected. A second buffer is justified only if fresh profiling shows exposed barrier or global-wait latency after the current layout changes.

Measure one and two buffers separately because shipped gfx1151 dense kernels select both schedules on different shapes.

## Ideas investigated and retained as deferred options

### Decode once per ordinary call

The fused kernel decodes the same packed weight tile once per M workgroup. At M=32,768, this can repeat decode hundreds of times.

A two-stage path can materialize the ordinary weight once into temporary BF16 and use a tuned dense GEMM. Approximate temporary sizes are:

| Shape | BF16 workspace |
| --- | ---: |
| Query | 32 MiB |
| Narrow | 2 MiB |
| Attention output | 16 MiB |
| Shared down | 2 MiB |

This has a high ceiling but is no longer an all-in-one packed kernel. It remains deferred under the current no-direct-hipBLASLt-linkage preference.

### True LDS swizzles

Padding is the simplest bank-layout change and has already produced large Q3_K/Q4_K gains.

A true XOR swizzle may distribute rows without increasing LDS footprint. It complicates address generation and can prevent compiler vectorization, so it should be attempted only with explicit vector loads and a controlled unpadded/padded/swizzled comparison.

### Wider LDS stores

The current scalar stores are a transpose from decoder-friendly registers into WMMA-friendly LDS.

Possible approaches include:

- cross-lane register transpose;
- decoder remapping to fixed input columns across multiple K values;
- a second LDS transpose stage.

These options may lose metadata sharing or add shuffle/barrier overhead. They are lower priority than repeated fragment-load optimization.

## Rejected experiments

The following were measured and reverted or superseded:

- larger backward wave counts;
- shuffle-based replacement of LDS sharing;
- oversized N tiles that reduced workgroup count too far;
- 2x2, 3x6, 4x4, and 1x16 ordinary geometries;
- eight-value ordinary decoders;
- ordinary `K_ITERATION=64` after the sixteen-value decoder;
- `GROUP_M=2` or `GROUP_M=4` on the final 128-row geometry;
- vector loading the complete Q4_K metadata header;
- custom LDS-only inline-assembly barriers;
- environment-driven production configuration;
- Q5_K eight-BF16 LDS row padding;
- Q4_K padding on small shared-down input width;
- Q6_K M=256 N=5 and the invalid N=7 full-tile dispatch;
- Q6_K M=256 K=16 and K=64;
- Q6_K M=128 N=3 at K=32 and K=64;
- Q6_K M=64 N=4/K=32.

The dispatch is a measured shape heuristic, not an autotuning system.

## Correctness and compatibility

The complete suite passes on the current Q6_K exact-tile guard, valid M=256 N=8 geometry, and selected LDS-padding source:

```text
pytest -q tests/
38 passed
```

Focused Q6_K and padding experiments preserved the benchmark correctness envelope. Final LM-head NRMSE was approximately:

| M | NRMSE |
| ---: | ---: |
| 64 | `5.100e-04` |
| 128 | `5.179e-04` |
| 256 | `5.100e-04` in the valid N=8 benchmark |

Backward uses the authoritative packed payload and does not quantize cotangents.

The extension was rebuilt after the Q5_K revert. `python -m compileall -q bench` and `git diff --check` also pass. A complete repeated ordinary benchmark remains required after the final LDS-load work.

## Profiler and tool issues

- Never run two GPU benchmarks or profilers in parallel.
- PC sampling perturbs short kernels heavily and should be interpreted qualitatively.
- One multi-counter M=256 Q6_K run caused an HSA memory fault and queue-sync timeouts. Use one counter at a time for that specialization.
- One earlier post-profiler process encountered a transient `hipErrorLaunchFailure`; a fresh focused test and full rerun succeeded without source changes.
- `roc-obj-ls` fails because of a `rocm_sdk_core._cli` import error.
- Inspect code objects through `.hip_fatbin`, `clang-offload-bundler`, `llvm-readobj`, `llvm-nm`, and `llvm-objdump`.
