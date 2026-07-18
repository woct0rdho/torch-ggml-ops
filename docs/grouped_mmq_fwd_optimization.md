# Grouped MMQ forward optimization targets

## Scope

This document defines the production forward shapes and dtypes to optimize for the Qwen3.6-35B-A3B GGUF LoRA model at:

- sequence length: 2048;
- top-k routing: 8;
- experts: 256;
- physical batch sizes: 1, 4, and 16;
- activation dtype at the public operator boundary: BF16;
- internal activation quantization: Q8_1;
- output dtype: BF16;
- packed-weight storage dtype: `torch.uint8`.

Gradient accumulation does not change these shapes. Only the physical per-device batch size changes the number of routed rows.

## Routed-row counts

The expert-sorted row count is:

```text
R = batch_size * sequence_length * top_k
  = batch_size * 2048 * 8
```

| Physical batch | Sorted rows `R` | Mean rows/expert if all 256 experts are active |
| ---: | ---: | ---: |
| 1 | 16,384 | 64 |
| 4 | 65,536 | 256 |
| 16 | 262,144 | 1,024 |

The actual per-expert row counts are dynamic. At batch 1, the full-model audit observed approximately 150-256 active experts per layer, with a mean near 198.5. A representative layer-10 workload had 192 active experts and group sizes around 60-106.

At batch 4 and 16, all 256 experts are expected to be active in most layers, but optimization benchmarks should still include skewed distributions rather than only uniform offsets.

## Operator metadata

Both grouped forward operators receive:

```text
expert_indices: [G] torch.int64, contiguous, CUDA
expert_offsets: [G] torch.int32, contiguous, CUDA
```

`G` is the active-expert count and is at most 256. `expert_offsets` contains cumulative end offsets; its last value is `R`. The first group begins at row zero.

Routing metadata must remain device-resident. Optimization must not add `.item()`, CPU descriptors, device-to-host copies, or synchronization to inspect group sizes.

## Production operator use

The current checkpoint uses:

1. `grouped_mmq_pair` for routed gate and up projections;
2. `grouped_mmq` for the routed down projection.

Gate and up have matching logical shape, physical geometry, and quantization type in every layer, so the paired operator is always used in production. Single gate/up `grouped_mmq` is only a compatibility fallback and is not a primary optimization target for this checkpoint.

## Gate/up paired forward

### Public tensors

```text
input:               [R, 2048] torch.bfloat16
gate_packed_weight:  [256, 512, packed_row_bytes] torch.uint8
up_packed_weight:    [256, 512, packed_row_bytes] torch.uint8
expert_indices:      [G] torch.int64
expert_offsets:      [G] torch.int32
gate_output:         [R, 512] torch.bfloat16
up_output:           [R, 512] torch.bfloat16
```

Conceptual logical GEMMs for each expert group are:

```text
A[group_rows, 2048] @ W_gate[512, 2048].T -> gate[group_rows, 512]
A[group_rows, 2048] @ W_up[512, 2048].T   -> up[group_rows, 512]
```

The pair shares one Q8_1 activation workspace.

### Activation and output shapes by batch

| Batch | Input `(shape, dtype)` | Gate output `(shape, dtype)` | Up output `(shape, dtype)` |
| ---: | --- | --- | --- |
| 1 | `([16384, 2048], bfloat16)` | `([16384, 512], bfloat16)` | `([16384, 512], bfloat16)` |
| 4 | `([65536, 2048], bfloat16)` | `([65536, 512], bfloat16)` | `([65536, 512], bfloat16)` |
| 16 | `([262144, 2048], bfloat16)` | `([262144, 512], bfloat16)` | `([262144, 512], bfloat16)` |

### Weight shapes and quantization

| Layers | Quant type | Quant ID | Logical shape per weight | Physical shape per weight | Storage dtype |
| --- | --- | ---: | ---: | ---: | --- |
| `0-9`, `30-39` | Q3_K | 11 | `[256, 512, 2048]` | `[256, 512, 880]` | `torch.uint8` |
| `10-29` | IQ2_S | 22 | `[256, 512, 2048]` | `[256, 512, 656]` | `torch.uint8` |

Each paired call receives two independent weights of the same shape and quantization type.

Resident payload sizes:

| Quant type | One weight | Gate/up pair |
| --- | ---: | ---: |
| Q3_K | 110 MiB | 220 MiB |
| IQ2_S | 82 MiB | 164 MiB |

### Internal Q8_1 workspace

Q8_1 uses 36 bytes per 32 activation values. For `K=2048`:

```text
Q8_1 row bytes = 2048 / 32 * 36 = 2304
```

| Batch | Internal workspace `(shape, dtype)` | Size |
| ---: | --- | ---: |
| 1 | `([16384, 2304], uint8)` | 36 MiB |
| 4 | `([65536, 2304], uint8)` | 144 MiB |
| 16 | `([262144, 2304], uint8)` | 576 MiB |

The workspace must be shared between gate and up. Creating one Q8_1 workspace per projection would double this allocation and is not acceptable.

### BF16 tensor sizes

| Batch | Input | Each output | Both outputs |
| ---: | ---: | ---: | ---: |
| 1 | 64 MiB | 16 MiB | 32 MiB |
| 4 | 256 MiB | 64 MiB | 128 MiB |
| 16 | 1 GiB | 256 MiB | 512 MiB |

## Down forward

### Public tensors

```text
input:          [R, 512] torch.bfloat16
packed_weight:  [256, 2048, packed_row_bytes] torch.uint8
expert_indices: [G] torch.int64
expert_offsets: [G] torch.int32
output:         [R, 2048] torch.bfloat16
```

The conceptual logical GEMM for each expert group is:

```text
A[group_rows, 512] @ W_down[2048, 512].T -> output[group_rows, 2048]
```

### Activation and output shapes by batch

| Batch | Input `(shape, dtype)` | Output `(shape, dtype)` |
| ---: | --- | --- |
| 1 | `([16384, 512], bfloat16)` | `([16384, 2048], bfloat16)` |
| 4 | `([65536, 512], bfloat16)` | `([65536, 2048], bfloat16)` |
| 16 | `([262144, 512], bfloat16)` | `([262144, 2048], bfloat16)` |

### Weight shapes and quantization

| Layers | Quant type | Quant ID | Logical shape | Physical shape | Storage dtype |
| --- | --- | ---: | ---: | ---: | --- |
| `0-1` | Q5_K | 13 | `[256, 2048, 512]` | `[256, 2048, 352]` | `torch.uint8` |
| `2-9`, `30-39` | Q4_K | 12 | `[256, 2048, 512]` | `[256, 2048, 288]` | `torch.uint8` |
| `10-29` | IQ2_S | 22 | `[256, 2048, 512]` | `[256, 2048, 164]` | `torch.uint8` |

Resident payload sizes:

| Quant type | Weight size |
| --- | ---: |
| Q5_K | 176 MiB |
| Q4_K | 144 MiB |
| IQ2_S | 82 MiB |

### Internal Q8_1 workspace

For `K=512`:

```text
Q8_1 row bytes = 512 / 32 * 36 = 576
```

| Batch | Internal workspace `(shape, dtype)` | Size |
| ---: | --- | ---: |
| 1 | `([16384, 576], uint8)` | 9 MiB |
| 4 | `([65536, 576], uint8)` | 36 MiB |
| 16 | `([262144, 576], uint8)` | 144 MiB |

### BF16 tensor sizes

| Batch | Input | Output |
| ---: | ---: | ---: |
| 1 | 16 MiB | 64 MiB |
| 4 | 64 MiB | 256 MiB |
| 16 | 256 MiB | 1 GiB |

## Group-size targets

The kernel must handle dynamic group sizes efficiently. Benchmark at least the following distributions.

### Batch 1

```text
R = 16384
primary group sizes: 64-128
uniform baseline: 256 groups of 64 rows
production-like baseline: about 192-200 active groups, centered around 80-86 rows
boundary sizes: 1, 15, 16, 17, 63, 64, 65, 127, 128, 129
```

### Batch 4

```text
R = 65536
primary group sizes: 192-320
uniform baseline: 256 groups of 256 rows
important tile counts: 128, 256, 384 rows
```

### Batch 16

```text
R = 262144
primary group sizes: 768-1280
uniform baseline: 256 groups of 1024 rows
important tile counts: 768, 896, 1024, 1152, 1280 rows
```

For each batch, include:

- exactly uniform groups;
- realistic skew around the mean;
- sparse active-expert IDs;
- non-multiple-of-16 and non-multiple-of-128 tails;
- a few large and small groups while preserving the total row count.

## Required benchmark matrix

### `grouped_mmq_pair`

Primary shape:

```text
input:   [R, 2048] bfloat16
weights: 2 x packed [256, 512, row_bytes] uint8
outputs: 2 x [R, 512] bfloat16
```

Required cases:

| R | Batch | Q3_K | IQ2_S |
| ---: | ---: | :---: | :---: |
| 16,384 | 1 | required | required |
| 65,536 | 4 | required | required |
| 262,144 | 16 | required | required |

Q3_K and IQ2_S each occur in 20 layers, so they have equal production priority.

### `grouped_mmq` down

Primary shape:

```text
input:  [R, 512] bfloat16
weight: packed [256, 2048, row_bytes] uint8
output: [R, 2048] bfloat16
```

Required cases:

| R | Batch | Q5_K | Q4_K | IQ2_S |
| ---: | ---: | :---: | :---: | :---: |
| 16,384 | 1 | required | required | required |
| 65,536 | 4 | required | required | required |
| 262,144 | 16 | required | required | required |

Priority by layer count:

1. IQ2_S: 20 layers;
2. Q4_K: 18 layers;
3. Q5_K: 2 layers.

## Effective training call frequency

Per model forward:

- 40 paired gate/up calls;
- 40 down calls.

Non-reentrant checkpointing recomputes the layer forwards during backward. One optimizer step therefore executes the grouped forward path approximately twice:

- 80 paired gate/up forward calls;
- 80 down forward calls.

Forward optimization priority is consequently:

1. `grouped_mmq_pair` Q3_K and IQ2_S;
2. down `grouped_mmq` IQ2_S and Q4_K;
3. down `grouped_mmq` Q5_K;
4. single gate/up `grouped_mmq` only as a compatibility case.

## Measurement requirements

For every benchmark case, report:

- median and distribution of kernel time after warmup;
- total forward time including Q8_1 quantization;
- effective logical TFLOPS for one projection and for the paired projection;
- peak live allocation and allocator-reserved growth;
- Q8_1 workspace size and confirmation that gate/up shares one workspace;
- active-expert count and group-size distribution;
- correctness against the same Q8_1 activation and independently decoded logical weights;
- maximum absolute error and normalized RMS error;
- current-stream behavior;
- absence of CPU/GPU synchronization;
- ISA resource changes, including VGPR, SGPR, LDS, and private spills.

Compare both isolated kernel time and complete routed projection time. Do not accept an isolated speedup that increases whole-layer allocation enough to threaten the demonstrated under-16-GiB training target.

## Constraints for optimization

Optimization must preserve:

- authoritative GGUF payload values;
- independent gate/up/down metadata;
- Q8_1 activation quantization in forward;
- BF16 outputs;
- original BF16 inputs for LoRA-A outside the native operator;
- device-resident `expert_indices` and `expert_offsets`;
- current Torch stream execution;
- no hidden copies in the public native ABI;
- no host-built grouped descriptors;
- no packed-weight gradients;
- no persistent transposed or requantized weight representation;
- no cross-expert reads at group tails;
- exact package support for FakeTensor, opcheck, autograd composition, and explicit higher-order-gradient behavior.
