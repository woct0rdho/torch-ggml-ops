# Dense MMQ forward optimization status and plan

## Scope

This document covers dense `torch_ggml_ops::mmq` forward on gfx1151.

Included:

- BF16 activations;
- internal Q8_1 activation quantization;
- packed GGUF Q3_K, Q4_K, Q5_K, Q6_K, and IQ2_S weights;
- BF16 outputs;
- the 160 ordinary model projections;
- the packed Q6_K language-model head;
- production batch sizes 1, 4, and 16 at sequence length 2048.

Excluded:

- `grouped_mmq` multiplication and routed-expert scheduling;
- GatedDeltaNet physical-layout permutations;
- LoRA GEMMs and residual accumulation;
- public operator-schema changes;
- changes to `csrc/vendor/llama_cpp/*`.

The grouped path shares the rewritten activation quantizer but now has its own independently optimized multiplication and scheduling implementation, documented in `docs/grouped_mmq_fwd_optimization.md`.

## Current status

Dense forward optimization is complete for the current fused packed representation and all retained source is committed.

Done:

- replaced the excessive small-workgroup Q8_1 launch with one 512-thread workgroup per real activation row;
- added compile-time row-tile selection and retained `J=64` only for the 64-row Q6_K fallback;
- kept the measured ordinary `I=64, J=128`, 128-thread geometry;
- validated zero private segment for the retained quantizer and dense forward specializations;
- selected 256 rows as the production LM-head loss chunk at the scheduling layer.

The source-of-record benchmark remains `/tmp/mmq_fwd_final_full.json`. The production LM-head decision must be evaluated with the complete loss loop: its M=256 forward call is slower than BF16 in isolation, but the 2,048-row packed loss is faster because M=256 sharply reduces call count and uses the optimized backward kernel.

Remaining work is architectural rather than another broad fused-kernel sweep:

- reuse Q8_1 activation workspaces across same-input projections;
- change the Q6_K M=256 representation or accumulator organization if a new dense-forward project is authorized;
- consider cross-call decoded-weight reuse or a transient project-owned decoded dense stage.

## Hardware and measurement rules

Measurements were taken on:

```text
GPU: Radeon 8060S Graphics
architecture: gfx1151, wave32, 40 CUs
PyTorch: 2.12.0+rocm7.15.0a20260701
HIP: 7.14.60850
```

The performance reference is `torch.mm` using the same BF16 activation and the logical GGUF weight dequantized to BF16. PyTorch normally dispatches these references to hipBLASLt.

The first forward and backward baselines were mistakenly run concurrently and contended for the GPU. They were discarded. All accepted numbers come from sequential runs with no concurrent GPU benchmark or profiler.

On gfx1151, int8 WMMA is approximately as fast as BF16 WMMA. Packed forward wins through lower weight traffic, compact staging, or better geometry rather than a nominal 2x arithmetic advantage.

## Benchmark harness and artifacts

The forward benchmark is:

```bash
source ~/venv_torch/bin/activate
python bench/benchmark_mmq_fwd.py
```

A focused example is:

```bash
python bench/benchmark_mmq_fwd.py \
  --cases narrow_q4_k --batches 1,4,16 --warmup 3 --repeats 9
```

Primary artifacts:

```text
Sequential baseline: /tmp/mmq_fwd_baseline_primary_sequential.json
Final full forward:  /tmp/mmq_fwd_final_full.json
```

The `/tmp` paths record measurement provenance and are not repository inputs.

## Production shapes

For ordinary projections, `M = batch * 2048`:

| Batch | M |
| ---: | ---: |
| 1 | 2,048 |
| 4 | 8,192 |
| 16 | 32,768 |

Representative shapes:

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
production chunk M = 256
```

Comparison chunks are `M = 64, 128, 256`.

## Current implementation

The dense forward path is implemented in project-owned `csrc/mmq_core.cuh` and dispatched from `csrc/mmq_hip.cu`.

The ordinary multiplication geometry remains:

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

Q6_K calls above 64 rows retain `J=128`.

## Accepted changes

### One activation-quantization workgroup per real row

The original Q8_1 launch used:

```text
grid = [rows_padded, K / 256]
block = 64 threads
```

At `K=2048`, it created eight workgroups per activation row and also quantized padded rows. At `M=32768`, it launched 262,144 small workgroups and dominated narrow-N calls.

The accepted launch is:

```text
grid = [rows, 1]
block = 512 threads
```

Each workgroup owns one real row. Every thread processes four BF16 values per loop iteration, and the block loops only when `K > 2048`.

The Q8_1 D4 and DS4 layouts, 32-value reductions, rounding, and workspace representation are unchanged. Padded workspace rows are not written because their corresponding MMQ outputs are bounds-masked.

### Compile-time `J` and the Q6_K `J=64` specialization

Dense forward was generalized from fixed `J=128` to compile-time `J` in project-owned code.

The 64-row Q6_K fallback specialization removes 64 padded rows and halves the accumulator and activation-tile footprint. The production 256-row chunk uses `J=128` because a global `J=64` policy regressed ordinary and larger-row workloads.

## Final retained results

The following compares the valid sequential baseline with `/tmp/mmq_fwd_final_full.json`. Ratio means packed-MMQ throughput divided by BF16 throughput.

| Case | M | Baseline ms | Final ms | Speedup | Final ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Query Q3_K | 2,048 | 3.213 | 3.054 | 1.05x | 1.05x |
| Query Q3_K | 8,192 | 13.324 | 12.403 | 1.07x | 1.03x |
| Query Q3_K | 32,768 | 52.589 | 48.828 | 1.08x | 1.03x |
| Narrow Q4_K | 2,048 | 0.235 | 0.214 | 1.10x | 1.48x |
| Narrow Q4_K | 8,192 | 1.990 | 0.899 | 2.21x | 1.04x |
| Narrow Q4_K | 32,768 | 7.636 | 3.590 | 2.13x | 0.99x |
| Attention output Q4_K | 2,048 | 1.811 | 1.410 | 1.28x | 1.26x |
| Attention output Q4_K | 8,192 | 7.638 | 5.717 | 1.34x | 1.23x |
| Attention output Q4_K | 32,768 | 30.624 | 22.986 | 1.33x | 1.20x |
| Shared down Q4_K | 2,048 | 0.256 | 0.257 | 0.99x | 5.59x |
| Shared down Q4_K | 8,192 | 1.053 | 0.985 | 1.07x | 5.35x |
| Shared down Q4_K | 32,768 | 4.150 | 3.939 | 1.05x | 5.26x |
| LM head Q6_K | 64 | 7.413 | 4.202 | 1.76x | 2.05x |
| LM head Q6_K | 128 | 8.001 | 7.988 | 1.00x | 1.56x |
| LM head Q6_K | 256 | 16.097 | 16.127 | 1.00x | 0.81x |

Secondary final ratios:

| Case | M=2,048 | M=8,192 | M=32,768 |
| --- | ---: | ---: | ---: |
| Query Q4_K | 1.13x | 1.12x | 1.10x |
| Narrow Q5_K | 1.21x | 1.00x | 0.95x |
| Narrow Q3_K | 1.38x | 0.97x | 0.92x |
| Shared down Q5_K | 5.28x | 5.15x | 5.13x |

The dominant 70-tensor narrow Q4_K path reaches BF16 parity at large M. The 64-row LM-head fallback reaches about 15.49 logical TFLOP/s and 2.05x BF16 throughput.

## Profiling and generated ISA

### Narrow Q4_K at `M=32768, N=512, K=2048`

`rocprofv3` tracing reported:

| Kernel | Baseline average | Final average | Final resources |
| --- | ---: | ---: | --- |
| Q8_1 quantizer | 5,105.979 us | 886.928 us | 32 allocated VGPRs, no LDS, no private segment |
| Dense MMQ | 2,565.848 us | 2,645.314 us | 256 allocated VGPRs, 38,400-byte LDS, no private segment |

The quantizer is 5.76x faster. Combined traced time fell from about 7.67 ms to 3.53 ms.

The multiplication kernel itself did not improve in this experiment. The end-to-end gain came from eliminating quantization scheduling overhead.

Code-object metadata reports 254 architectural VGPRs for Q4_K `J=128`; rocprofv3 rounds the allocation to 256.

### Q6_K LM head at `M=64`

The `J=64` multiplication kernel averages about 4,190 us and uses:

```text
176 allocated VGPRs
28,928-byte LDS
0-byte private segment
```

The Q8_1 quantizer is about 3.4 us at this row count. The call is almost entirely multiplication time, and removing padded-row multiplication produced the 1.76x speedup.

### LDS vectorization and layout

Generated gfx1151 ISA shows that the main forward LDS path is already substantially vectorized.

Q4_K and Q5_K `J=128` use `ds_load_b128` for their principal LDS reads. Their staged operands use dual-address 32-bit stores such as `ds_store_2addr_b32` and `ds_store_2addr_stride64_b32`.

Q3_K and Q6_K use paired 32-bit or 64-bit LDS operations where their packed layouts permit them. Some scale, metadata, and packed fields remain 32-bit because they are not naturally contiguous per thread.

Forward retains the inherited `GGML_PAD(...)` separation and type-specific SRAM strides used by the selectively vendored MMQ templates. This pass did not perform a controlled forward bank-swizzle sweep, so the layout should not be described as proven optimal.

There is no current evidence that forward LDS layout is a large production bottleneck. The dominant shapes already match or beat BF16, and the observed narrow improvement came from the quantizer rather than multiplication.

## Experiment log

### Accepted

| Experiment | Result |
| --- | --- |
| One 512-thread Q8_1 block per real row | Removed padded-row work and excessive small workgroups; first-order narrow gain |
| Compile-time forward `J` | Enabled bounded row-tile specialization without vendor changes |
| Q6_K `J=64` for padded rows `<=64` | 7.413 to 4.202 ms on the low-memory LM-head fallback |
| Ordinary `I=64`, `J=128`, 128 threads | Best measured general production configuration |

### Rejected or not generalized

| Experiment | Reason |
| --- | --- |
| Global `J=64` | Regressed ordinary and larger-row workloads |
| `I=128` | Regressed the measured shape mix |
| Smaller forward thread/tile combinations | Did not improve the production aggregate |
| A major WMMA representation rewrite | No large remaining production margin and high implementation risk |
| DirectToLds/DirectToVgpr-style rewrite | Packed reconstruction remains in the path; selected hipBLASLt references also disable these modes |
| gfx1250 WMMA arb-stall programming | The capability is not available on gfx1151 |

## Relevant architecture lessons

TensileLite and shipped hipBLASLt remain useful as records of measured gfx1151 geometry and scheduling. They are not directly reusable generators for packed GGUF MMQ.

Relevant forward lessons:

- four wave32 waves remain a sound workgroup size;
- conventional global-to-VGPR-to-LDS staging is competitive on gfx1151;
- one and two LDS buffers both win in different dense shapes, so dual buffering is not a universal rule;
- wide local reads and aligned LDS layouts are desirable, but must be judged by end-to-end time;
- source-swap and transposed LDS concepts are useful orientation references rather than drop-in code;
- generated dense-GEMM solution databases do not model Q8 activation workspaces or GGUF decode.

FeatherOps reinforces several measurement rules:

- use controlled ablations rather than PC samples alone;
- inspect generated ISA and resource metadata after layout or vectorization changes;
- use real nonzero operands because zero WMMA inputs can mislead;
- keep explicit prefetch state in fixed scalar/vector VGPR values rather than compiler-managed arrays;
- optimize whole-call time, including quantization and workspace allocation.

## Completed work and stopping point

The retained dense-forward implementation has bounded the useful local neighborhoods for quantizer launch shape, `I`, `J`, thread count, and the current fused packed WMMA organization.

The remaining single-call deficits are:

- Q6_K at M=256: 16.127 ms versus about 13.05 ms BF16, or approximately 0.81x throughput;
- narrow Q3_K and Q5_K at M=32,768: about 5-9% behind BF16;
- repeated Q8_1 quantization and workspace allocation when several projections consume the same BF16 activation.

These deficits do not change the production LM-head selection. M=256 is the production chunk because the complete 2,048-row packed-loss loop measured 229.958 ms versus 312.690 ms for M=64. The faster backward schedule and lower call count outweigh the forward single-call deficit. M=128 and M=64 remain lower-memory fallbacks.

No further global `J=64`, `I=128`, or broad tile sweep should be repeated. Those neighborhoods already regressed the production mix.

## Remaining work

### Reuse Q8_1 activations across same-input projections

This is the highest-confidence dense-forward opportunity. Every dense `mmq` call currently allocates and fills a Q8_1 workspace even when several projections consume the same BF16 tensor.

A future implementation should use either:

- an explicit prepared-activation internal operator; or
- a dense pair/multi-projection operator analogous to `grouped_mmq_pair`.

Q4_K/Q5_K use the scale-plus-sum Q8 layout, while Q3_K/Q6_K/IQ2_S use the scale-only layout. A layer requiring both classes needs at most two quantizations. Avoid an implicit pointer cache unless it tracks storage identity, tensor version, device, stream ordering, shape, row padding, and metadata layout.

### Revisit Q6_K M=256 only with a new representation argument

A future Q6_K project must materially reduce the `J=128` kernel's approximately 225 architectural VGPRs or avoid reconstructing the same packed data in the same way. Acceptance requires a complete M=256 gain without regressing M=64, M=128, or ordinary projections.

Promising higher-ceiling directions are:

- cross-call decoded-weight reuse;
- a compact lossless int8-plus-scale cache;
- transient packed-to-BF16 decode followed by a project-owned dense stage.

On gfx1151, approximate int8 WMMA is not justified by raw arithmetic throughput alone.

### Lower-priority ISA-guided work

Wider activation staging or a changed LDS producer mapping is justified only if fresh profiling identifies exposed instruction or wait overhead. The compiler already combines many stores, and the accepted production gains came from quantizer scheduling rather than a multiplication rewrite.

hipBLASLt remains a geometry and scheduling reference. The packed project kernels must remain independent of direct hipBLASLt linkage under the current policy.

## Correctness and compatibility

The current complete project suite passes:

```text
pytest -q tests/
39 passed
```

The `~/test_no_unsloth` integration suite also passes 9 tests, including the production 256-row packed-loss schedule.

Forward normalized RMSE remains within the existing Q8_1 envelope:

- approximately 0.6% for Q3_K and Q6_K;
- approximately 1.1-2.0% for Q4_K and Q5_K.

Dense forward and quantizer specializations have zero-byte private segments. No file under `csrc/vendor/llama_cpp/*` was modified.

## Tool notes

- PC sampling heavily perturbs short kernels, so use it qualitatively.
- `roc-obj-ls` is broken in the active environment because of a `rocm_sdk_core._cli` import error.
- Code-object inspection uses `.hip_fatbin`, `clang-offload-bundler`, `llvm-readobj`, `llvm-nm`, and `llvm-objdump`.
