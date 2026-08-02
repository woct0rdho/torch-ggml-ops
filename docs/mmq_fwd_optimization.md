# Dense MMQ forward optimization

## Current status

Dense `torch_ggml_ops::mmq` forward is at its practical local stopping point on gfx1151 for the existing packed-weight, single-call operator contract.
The retained implementation includes:
- one 512-thread Q8_1 quantization workgroup per real activation row.
- six exact DeepSeek Q8_0 wrappers covering K=1024/2048/4096/8192 and the J64 language-model-head bodies.
- eight exact Qwen wrappers covering Q3_K, Q4_K, Q5_K, and Q6_K production K/J combinations.
- static quant/shape dispatch with generic fallbacks for unsupported or bounded shapes.
- zero private storage, zero spills, and no dynamic stack for every retained resource-gated specialization.

Against their generic controls, exact specialization improved the complete DeepSeek matrix by `21.62%` geometrically and the complete Qwen matrix by `10.44%`. Every point improved in its sequential generic/candidate/generic bracket.

No broad local tile, workgroup, unroll, prefetch, buffering, or LDS-layout sweep remains open. The remaining opportunities require model-owned activation reuse, a prepared Q6 representation, or a narrowly scoped arithmetic experiment that changes Q6 floating-point operation order.

Final source-of-record artifacts:
- DeepSeek: `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_stride76_25.json`.
- Qwen: `~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_25.json`.
- Qwen complete packed loss: `~/tmp/torch-ggml-ops/mmq_fwd_qwen_complete_loss_final_25.json`.

## Latest results

All tables report complete public `mmq` latency, including Q8_1 activation quantization and workspace ownership. Ratio is BF16 `torch.mm` latency divided by packed-MMQ latency, so values above `1.00x` favor MMQ. Ordinary rows use `M = batch * 2048`.

### Qwen ordinary projections

Each latency cell is `MMQ ms / BF16 ratio` from the final 25-repeat matrix.

| Family | `(N, K)` | Type | Tensors | B1, M=2,048 | B4, M=8,192 | B16, M=32,768 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Attention query/query gate | `(8192, 2048)` | Q3_K | 9 | `2.991 / 1.10x` | `12.370 / 1.06x` | `50.727 / 1.02x` |
| Attention query/query gate | `(8192, 2048)` | Q4_K | 1 | `2.591 / 1.26x` | `11.046 / 1.20x` | `43.788 / 1.18x` |
| Narrow K/V/shared gate/up | `(512, 2048)` | Q4_K | 70 | `0.196 / 1.61x` | `0.823 / 1.16x` | `3.433 / 1.07x` |
| Narrow K/V/shared gate/up | `(512, 2048)` | Q5_K | 21 | `0.196 / 1.62x` | `0.840 / 1.13x` | `3.413 / 1.07x` |
| Narrow attention key | `(512, 2048)` | Q3_K | 9 | `0.223 / 1.42x` | `0.919 / 1.03x` | `3.869 / 0.95x` |
| Attention output | `(2048, 4096)` | Q4_K | 10 | `1.347 / 1.38x` | `5.682 / 1.30x` | `23.492 / 1.24x` |
| Shared-expert down | `(2048, 512)` | Q4_K | 30 | `0.176 / 8.20x` | `0.711 / 7.42x` | `2.980 / 6.99x` |
| Shared-expert down | `(2048, 512)` | Q5_K | 10 | `0.183 / 7.89x` | `0.711 / 7.43x` | `2.973 / 7.01x` |

The ordinary 24-point geometric ratio is `1.89x` BF16. Narrow Q3_K B16 is the only ordinary point below parity. Its exact packed multiplication is `2.725 ms` versus `3.691 ms` BF16, but Q8_1 activation quantization adds `0.879 ms`, or about `24.4%` of the public call. This is an activation-preparation deficit, not an open Q3_K multiplication-body deficit.

### Qwen language-model head

The packed Q6_K head has `(N,K)=(248320,2048)`. Forward-only timings are:

| Chunk M | Calls for 2,048 rows | MMQ ms/call | BF16 ms/call | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 32 | 4.192 | 8.909 | 2.13x |
| 128 | 16 | 7.983 | 12.892 | 1.61x |
| 256 | 8 | 16.568 | 13.462 | 0.81x |

M256 is slower than persistent BF16 GEMM in isolation, but it remains the production packed-loss schedule because scheduling must include forward MMQ, in-place cross-entropy, and packed MMQ grad-input.

The final complete 2,048-row bracket used phase order `64/128/256/256/128/64` with 25 repeats per phase:

| Chunk M | Complete loss ms | Incremental peak allocation | Relative to M256 |
| ---: | ---: | ---: | ---: |
| 64 | 328.447 | 69.03 MiB | 1.33x slower |
| 128 | 293.991 | 130.04 MiB | 1.19x slower |
| 256 | 246.323 | 253.57 MiB | selected |

M256 is `16.22%` faster than M128 and `25.00%` faster than M64. Loss and packed hidden gradients are identical across chunk sizes.

### DeepSeek ordinary projections

DeepSeek has 301 ordinary dense Q8_0 calls across six geometry families. Each latency cell is `MMQ ms / BF16 ratio`.

| Family | `(N, K)` | Tensors | B1, M=2,048 | B4, M=8,192 | B16, M=32,768 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention Q-A | `(1024, 4096)` | 43 | `0.630 / 1.57x` | `2.890 / 1.40x` | `11.635 / 1.40x` |
| Attention Q-B | `(32768, 1024)` | 43 | `5.466 / 1.31x` | `21.945 / 1.30x` | `87.020 / 1.32x` |
| Attention KV | `(512, 4096)` | 43 | `0.342 / 2.01x` | `1.650 / 1.16x` | `6.761 / 1.13x` |
| Attention output B | `(4096, 8192)` | 43 | `5.467 / 1.29x` | `22.580 / 1.21x` | `87.816 / 1.25x` |
| Shared gate/up | `(2048, 4096)` | 86 | `1.293 / 1.46x` | `5.432 / 1.41x` | `22.908 / 1.30x` |
| Shared down | `(4096, 2048)` | 43 | `1.261 / 1.41x` | `5.247 / 1.39x` | `22.406 / 1.30x` |

Every ordinary point beats BF16. Ratios span `1.13-2.01x`, with a geometric ratio of `1.36x` across the 18 unique points.

### DeepSeek language-model head

The packed Q8_0 head has `(N,K)=(129280,4096)`:

| Chunk M | MMQ ms/call | BF16 ms/call | Ratio |
| ---: | ---: | ---: | ---: |
| 32 | 2.849 | 5.353 | 1.88x |
| 64 | 2.947 | 9.044 | 3.07x |
| 128 | 5.234 | 13.510 | 2.58x |
| 256 | 10.547 | 23.715 | 2.25x |
| 512 | 21.243 | 27.873 | 1.31x |

Every isolated forward chunk beats BF16. These timings do not select a production chunk: DeepSeek still needs dense Q8_0 backward and a complete packed-loss loop.

### Exact-specialization gains

| Model | Final matrix | Geometric latency gain vs generic | Checkpoint-weighted gain | Final BF16 comparison |
| --- | ---: | ---: | ---: | --- |
| DeepSeek | 33 report points | 21.62% | 14.64-25.36% | Ordinary `1.13-2.01x`. LM head `1.31-3.07x` |
| Qwen | 33 report points | 10.44% | 3.86-5.10% | Ordinary `0.95-8.20x`. LM head `0.81-2.13x` |

## Production dispatch

Dense forward device bodies are in project-owned `csrc/mmq_core.cuh`. `tools/build_mmq_bundle.py` generates concrete gfx1151 wrappers, `csrc/mmq_bundle.cpp` performs static selection and launch, and `csrc/mmq_hip.cu` owns validation and tensor/workspace allocation.

The common full-tile geometry is:

```text
I = 64
J = 128
threads = 128, four wave32 waves
K iteration = 256
```

Retained exceptions and exact bodies:
- Q8_0 K4096 M32 uses bounded J64. M64 uses full J64.
- Q8_0 K1024/2048/4096/8192 full rows use exact-K J128.
- Q3_K K2048 uses exact full J128.
- Q4_K K512/2048/4096 uses exact full J128.
- Q5_K K512/2048 uses exact full J128.
- Q6_K K2048 M64 uses exact full J64. rows divisible by 128 use exact full J128.
- generic bounds-safe J128 bodies remain for every other supported shape. Q6_K rows at or below 64 retain the generic J64 fallback.

Exact dispatch requires N divisible by I64 and M divisible by the selected J, except for the intentionally bounded Q8_0 M32 body. Dispatch depends only on quant type and matrix geometry. There is no online autotuning or pointer-identity cache.

### Resources

| Body | VGPR | SGPR | Dynamic LDS | Private/spills/stack |
| --- | ---: | ---: | ---: | --- |
| Generic Q8_0 J128 control | 248 | 29 | 38,400 B | zero |
| Exact Q8_0 J128 | 216 | 28 | 38,400 B | zero |
| Exact/bounded Q8_0 J64 | 132 | 28 | 28,928 B | zero |
| Exact Q3_K J128 | 196 | 27 | 40,448 B | zero |
| Exact Q4_K J128 | 239 | 28-29 | 38,400 B | zero |
| Exact Q5_K J128 | 244 | 28 | 38,400 B | zero |
| Exact Q6_K J128 | 210 | 27 | 38,400 B | zero |
| Exact Q6_K J64 | 158 | 27 | 28,928 B | zero |
| Q8_1 row quantizers | 23-28 (32 allocated) | 17 | zero | zero |

The Q3_K/Q4_K/Q5_K/Q6_K generic controls used 216/254/230/225 VGPRs at J128, and generic Q6_K J64 used 171. Exact specialization reduced allocation except for Q5_K, which rose to 244 VGPRs but still improved every measured point.

Compatibility entries are not production-tuned without a concrete workload. In particular, generic dense Q2_K J128 uses 112 private bytes and 27 VGPR spills and is deliberately not resource-gated. The DeepSeek Q2_K production workload is routed and uses grouped MMQ instead.

## Remaining work

### Local packed-kernel status

DeepSeek Q8_0 local optimization is exhausted under the current representation and arithmetic contract. Exact full bodies beat BF16 everywhere, and all profiler-supported geometry, loop, barrier, and LDS controls lost.

Qwen Q3_K/Q4_K/Q5_K multiplication bodies are also at a practical stopping point. Every exact body improved its generic control. The only ordinary below-parity public call is narrow Q3_K B16, where the multiplication already beats BF16 and activation preparation is the deficit.

Do not reopen global J64, I128, workgroup-size, split-K, persistent-workgroup, speculative-prefetch, decoded-weight-LDS-cache, or broad swizzle sweeps without a new profile-supported mechanism.

### Optional Q6 effective-scale experiment

One narrowly bounded local Q6_K experiment remains defensible if a small floating-point operation-order change is acceptable.

The current Q6 dot computes an integer WMMA accumulator times an int8 per-16-value scale, converts the product to float, and then applies the block and activation scales. The exact J128 Q6 body has 2,093 normalized instructions versus 1,453 for Q8 and includes 128 `v_mul_lo_u32` instructions. The packed loader already performs vector low/high reconstruction, and generated ISA already hoists packed scales and block factors, so another packed-extraction or metadata-prefetch toggle is not justified.

A project-owned Q6 loader could instead unpack and stage `float(block_d * scale)` once per weight row/K iteration in LDS. The dot would convert the WMMA accumulator first and consume the effective scale, potentially removing the repeated integer scale multiply at a modest LDS cost.

This changes rounding from the current integer-product-first order. It is not allowed under bitwise current-output parity. If tested, retain it only after:
- zero private storage, spills, and dynamic stack.
- at least a stable 2% complete-call M256 gain in a 25-repeat bracket.
- no repeatable M64/M128 regression above 1%.
- independent GGUF-reference correctness and complete packed-loss acceptance.

This is one semantic experiment, not authorization for another Q6 geometry or layout sweep.

### Same-input activation reuse

This is the highest-confidence whole-call opportunity, but it is a model-integration change.

The DeepSeek call graph confirms two same-input families:
- attention Q-A and KV consume the same attention `cur`.
- shared gate and up are parallel branches over the same `build_ffn` input.

All use the Q8_1 F32_D4 metadata layout. Profiling in `~/tmp/torch-ggml-ops/mmq_fwd_final_components.txt` attributes about `7.5%` of shared-gate B1 and `29.7%` of KV B16 call time to quantization. Qwen narrow Q3_K B16 spends about `24.4%` in quantization.

A prepared-activation or pair/multi-projection API must:
- quantize once per required Q8_1 metadata layout.
- own workspace lifetime explicitly for the current stream.
- preserve ordinary `mmq` as a fallback.
- preserve output and autograd behavior.
- avoid a hidden pointer/version/stream cache.

Q4_K/Q5_K use the scale-plus-sum workspace layout. Production dense Q3_K/Q6_K use scale-only; the retained reference-only dense IQ2_S path uses the same layout. A model layer may need one workspace per production layout, but never duplicate preparation for the same input/layout.

This repository does not own the model scheduler or prepared-activation lifetime, so ordinary single-call `mmq` remains allocation-owning.

### Prepared Q6 representation

Qwen Q6_K M256 remains representation/arithmetic-bound. Q8_1 quantization is below `0.1%` of that call, exact bounds/K state added only `2.09-2.56%`, and forward remains `16.568 ms` versus `13.462 ms` BF16 in the final matrix.

A transient BF16 stage is rejected. The optimistic control allocated and copied an already-decoded BF16 weight and then ran `torch.mm`, deliberately excluding decode computation:

| Path | Median |
| --- | ---: |
| Direct packed Q6_K | 15.464 ms |
| Persistent BF16 GEMM reference | 13.555 ms |
| Optimistic transient BF16 floor | 22.302 ms |

The transient floor loses by `44.21%`, requires a 1,017,118,720-byte workspace, and reaches about 1.145 GB incremental peak allocation. Artifact: `~/tmp/torch-ggml-ops/mmq_fwd_qwen_q6_transient_floor_25.json`.

The packed LM-head weight is 417,177,600 bytes. A lossless prepared integer-plus-scale layout needs at least 508,559,360 int8 value bytes plus per-16-value and block scales, approximately 519-546 MiB depending on alignment. It must be model-owned and specify preparation cost, lifetime, invalidation, device placement, memory budget, and forward/backward reuse. A hidden persistent BF16 shadow is not acceptable.

### DeepSeek loss scheduling

DeepSeek isolated LM-head forward is complete for M32/M64/M128/M256/M512, but forward timing cannot select a production loss chunk. Dense Q8_0 grad-input and the complete packed-loss loop must exist first. Final selection must include complete-loop latency and peak allocation at physical batches 1, 4, and 16.

## Optimization log

### Q0: activation quantizer and initial Qwen geometry

The original Q8_1 launch used one 64-thread workgroup per `(padded row, K/256 block)`:

```text
grid = [rows_padded, K / 256]
block = 64 threads
```

At K=2048 and M=32768 it launched 262,144 small workgroups and quantized padded rows. The retained launch is:

```text
grid = [real rows, 1]
block = 512 threads
```

Each thread processes four BF16 values per loop iteration. The block loops only when K exceeds 2048. F32_D4/F16_D4S4/F16_D2S6 metadata, 32-value reductions, rounding, and workspace semantics are unchanged.

On narrow Q4_K M32768, traced quantizer time fell from `5,105.979 us` to `886.928 us`, while multiplication remained roughly `2,565.848 us` versus `2,645.314 us`. Combined traced time fell from about 7.67 to 3.53 ms. This established activation scheduling, not multiplication, as the first-order narrow bottleneck.

The pre-exact Qwen source `~/tmp/torch-ggml-ops/mmq_fwd_final_full.json` recorded the retained effects of the quantizer and initial geometry:
- query Q3_K improved about 5-8% across batches.
- narrow Q4_K improved about 2.1-2.2x at B4/B16.
- attention-output Q4_K improved about 1.28-1.34x.
- Q6_K M64 improved from 7.413 to 4.202 ms after J64 removed padded-row arithmetic.
- Q6_K M128/M256 did not benefit from a global J64 policy.

The retained general geometry remained I64/J128 with 128 threads. I128, global J64, and smaller workgroup/tile combinations regressed the production mix.

### B0: standalone bundle conversion

Dense forward and its activation quantizers moved from the extension fatbinary into independent gfx1151 HSACOs. F32_D4/F16_D4S4/F16_D2S6 selection, J128 ordinary geometry, J64 small-row Q6 dispatch, and arithmetic semantics remained unchanged.

The initial nine-repeat before/after comparison measured `-0.23%` geometric movement. A sequential embedded/bundle/embedded 25-repeat control measured the bundle at `+0.87%` geometrically, `+0.76%` by median point, and `+1.05%` by estimated model latency against the embedded midpoint. The two embedded controls drifted by `2.18%`.

Approximately 3.2% movements in attention-query Q3_K B4 and attention-output Q4_K B4, and a `-1.5%` narrow Q5_K B1 movement, occurred without a HIP-semantic change. They are code-object placement variance, not optimization evidence.

Artifacts:
- `~/tmp/torch-ggml-ops/mmq_fwd_pre_bundle.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_post_bundle.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_embedded_pre_control_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_bundle_control_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_embedded_post_control_25.json`.

### D0: DeepSeek baseline and decomposition

The first accepted DeepSeek matrix established:
- ordinary Q8_0 sustained about 18.4-23.4 logical TFLOP/s.
- generic LM-head M32/M64/M128 took 5.395/5.630/5.859 ms, exposing J128 padded-row arithmetic.
- quantization was only `0.3%` of LM-head M32 and `0.4%` of Q-B B1, but `26.3%` of KV B16 and `3.2%` of output-B B1.
- Qwen narrow Q3_K B16 spent `23.8%` in quantization, while Q6_K M256 spent below `0.1%`.

Artifacts:
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_plan_baseline_9.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_qwen_plan_control_9.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_p0_components.txt`.

The first forward/backward baselines had accidentally run concurrently and were discarded. All retained measurements are sequential.

### D1: exact DeepSeek Q8_0 bodies

The generic Q8_0 J128 body used runtime M/N/K state, 248 VGPRs, 29 SGPRs, and 38,400 bytes of dynamic LDS. Production dimensions permit exact K and full I/J specialization.

The first I64/J32/K4096 full body was rejected before installation because it used 48 private bytes and spilled 11 VGPRs. The retained policy is:
- M32: bounded I64/J64.
- M64: full I64/J64.
- all other production rows: exact-K full I64/J128.

Exact J128 reduced allocation to 216 VGPRs. J64 uses 132. All retained bodies have zero private storage, spills, and dynamic stack.

Against the generic bracket midpoint:
- all 18 ordinary points improved by `9.18-21.75%`.
- LM M32/M64 improved by `47.40%`/`49.47%`.
- LM M128/M256/M512 improved by `12.02-12.96%`.
- the 33-point geometric latency improved by `21.62%`.
- checkpoint-weighted latency improved by `14.64-25.36%`, depending on provisional LM chunk.

Artifacts:
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p1_generic_before_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p1_exact_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p1_generic_after_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p1_exact_bracket_25.txt`.
- `~/tmp/torch-ggml-ops/mmq_fwd_qwen_p1_control_9.json`.

A detached pre-change build produced byte-identical Qwen quantizer/Q3_K/Q4_K/Q5_K/Q6_K HSACOs and a byte-identical generic Q8_0 fallback, confirming no Qwen device-code change during this phase.

### D2: rejected DeepSeek local controls

Ordinary J64 was rejected for every K class:
- K4096 Q-A/KV/shared gate/up lost `4.03-8.85%`. LM M128 lost `14.11%`.
- K1024 Q-B lost `6.03-8.78%`.
- K8192 output-B lost `3.75-7.12%`.
- K2048 shared down lost `4.94-5.58%`.

J128 packed-weight reuse outweighs J64's lower VGPR/LDS footprint on every full tile. J64 remains only where M32/M64 would otherwise be padded.

Artifacts:
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_k4096_j64_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_k4096_j128_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_k4096_j64_control_25.txt`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_other_j64_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_other_j128_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_other_j64_control_25.txt`.

K-loop unrolling was closed from normalized ISA. Every exact K body has the same 1,453-instruction loop body. Each 256-value iteration has only three scalar loop-control instructions around hundreds of load, WMMA, conversion, and FMA instructions. Duplicating the body would increase instruction-cache pressure without removing meaningful control work.

Activation-half double buffering loaded both Q8_1 halves before compute and reduced four barriers per K iteration to two. It remained spill-free at 214 VGPRs/56,832 LDS bytes for J128 and 130 VGPRs/38,144 bytes for J64, but regressed every point:
- ordinary shapes lost `6.83-25.80%`.
- LM M32/M64 lost `4.22%`/`2.03%`.
- full LM tiles lost `12.44-13.16%`.

The extra LDS did not create occupancy, and loading both halves up front disrupted the accepted load/compute cadence.

Artifacts:
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_prefetch_y_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_prefetch_baseline_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_prefetch_y_control_25.txt`.

The accepted stride-76 Q8 body measured `7.73%` LDS bank conflict, about `7.5%` ALU stalled by LDS, 128-cycle derived LDS latency, about 11.6 active waves/CU, and about 66% L2 hit rate. Stride 77 reduced the conflict metric to `5.12%` but doubled ALU stalled by LDS to about `14.3%`. It regressed every point by `3.83-12.86%`, or `8.58%` geometrically. The accepted `+4` Q8 row padding remains.

Artifacts:
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_stride77_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_stride76_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_stride77_control_25.txt`.

### Q1: exact Qwen bodies

Eight exact wrappers were retained:
- Q3_K K2048 J128.
- Q4_K K512/K2048/K4096 J128.
- Q5_K K512/K2048 J128.
- Q6_K K2048 J64/J128.

Every point improved against its generic midpoint:
- Q3_K: `0.90-5.26%`.
- Q4_K: `3.60-31.39%`.
- Q5_K: `11.66-32.60%`.
- Q6_K: `2.09-2.56%` at M64/M128/M256.
- complete 33-point geometric latency: `10.44%`.
- checkpoint-weighted latency: `3.86-5.10%`.

Artifacts:
- `~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_before_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_after_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_bracket_25.txt`.

Exact specialization closed runtime bounds/K state as explanations for Q6 M256. Its residual gap is packed Q6 arithmetic/representation, not geometry.

### R0: representation controls

Transient BF16 materialization was rejected as recorded under Remaining work. The control deliberately excluded decode computation and still lost by `44.21%` with about 1.145 GB incremental peak allocation.

Persistent BF16 shadows are also rejected: they duplicate approximately 1.017 GB for the Q6 head alone and do not preserve packed storage as the authoritative model representation.

A compact lossless integer-plus-scale representation remains a separate model-owned project. Dense evidence does not authorize changing routed or fixed-group representations, which retain their own grouped paths and acceptance matrices.

### L0: complete-loss and release acceptance

The final Qwen complete-loss bracket retained M256 over M64/M128 with exact loss and hidden-gradient agreement. DeepSeek complete-loss selection remains blocked on Q8_0 grad-input.

Final validation passed:
- 83 project tests.
- Ruff and compileall.
- in-place extension build.
- forced 132-kernel bundle build and freshness check.
- all requested resource gates.
- independent two-build reproducibility.
- `git diff --check`.

## Closed directions and durable reasoning

| Direction | Status and reason |
| --- | --- |
| Global J64 | Regressed all ordinary DeepSeek shapes and larger Qwen rows. Retain only for padded M32/M64 classes |
| I128 | Wider output reuse did not repay accumulator/LDS/residency costs in dense and grouped controls |
| 64/256-thread alternatives | Smaller configurations did not improve the production aggregate. Grouped eight-wave ownership is a different contract |
| K unroll 2/4 | Loop control is negligible beside the 1,453-instruction reduction body |
| Activation-half double buffering | Spill-free but lost 2-26%. Extra LDS and altered cadence outweighed fewer barriers |
| Generic LDS padding/swizzles | Stride-77 Q8 reduced the conflict metric but increased actual LDS stalls and lost every point |
| Decoded-weight LDS cache | Grouped controls lost occupancy from the larger LDS footprint. Dense full tiles lack a new reuse argument |
| Speculative packed prefetch | Longer packed/decode live ranges repeatedly lost or crossed resource cliffs in dense/grouped work |
| Split-K, GSU, Stream-K, persistent workgroups | No reduction/parallelism deficit in the regular dense production grid. Grouped controls were neutral or slower |
| Inactive-M suppression and row tasks | Every dense production tile is active and directly exposed by the regular M/N launch grid |
| DirectToLds/DirectToVgpr rewrite | Direct paths do not perform GGUF reconstruction. Selected BF16 references also do not imply a packed-MMQ implementation |
| Major WMMA representation rewrite | Requires changed arithmetic/representation ownership and is not a local schedule optimization |
| gfx1250 arb-stall programming | Capability is unavailable on gfx1151 |
| Raw code-object or symbol movement | Embedded controls drifted more than the bundle delta. Only normalized ISA or HIP-semantic movement is evidence |

Transfer lessons from the other kernel logs:

| Source | Transferable result | Dense-forward boundary |
| --- | --- | --- |
| Dense backward | Exact shapes and quant-specific extraction can matter. LDS layouts are body-specific | Backward transposes ownership and uses different producers/consumers, so its swizzle constants do not transfer |
| Grouped forward | Compile-time N/K, exact full/tail bodies, and bounded decoder unrolling can remove major overhead | Routed/fixed scheduling, serial row reuse, mixed tails, and group-major workspaces do not describe dense full tiles |
| Grouped backward | Resource cliffs and explicit representation ownership are first-class constraints | Inactive-M consumers, row tasks, pair accumulation, and route policies are grouped-only mechanisms |
| Kernel bundle | Warm modules, normalized ISA, concrete wrappers, resource checks, and reproducibility are mandatory | Offsets, symbol ordering, and compiler instruction ordering alone are not performance evidence |

Architecture lessons retained for future work:
- four wave32 waves remain a sound dense-forward workgroup.
- conventional global-to-VGPR-to-LDS staging is competitive on gfx1151.
- Q4_K/Q5_K principal LDS reads are already `ds_load_b128`. Q3_K/Q6_K use paired 32/64-bit forms where packed layouts permit.
- one versus two LDS buffers is shape-specific. Buffer count is not an optimization by itself.
- int8 WMMA is approximately as fast as BF16 WMMA on gfx1151, not nominally 2x faster.
- packed forward wins through compact traffic/staging and useful geometry, not an assumed arithmetic-rate advantage.
- real nonzero tensors, event-timed complete calls, normalized ISA, and resource metadata are required for decisions.

## Scope and contracts

Included:
- BF16 activations and outputs.
- internal Q8_1 activation quantization.
- packed GGUF Q3_K/Q4_K/Q5_K/Q6_K/Q8_0 production compatibility.
- 160 ordinary Qwen projections and the packed Q6_K head.
- 301 ordinary DeepSeek Q8_0 projections and the packed Q8_0 head.
- sequence length 2048 at physical batches 1, 4, and 16.
- direct packed-weight execution without logical weight materialization.

Excluded:
- production dense IQ2_S expert weights; their model execution and correctness coverage belong to grouped MMQ. The existing HIP dense IQ2_S path remains reference code.
- grouped/routed/fixed-group multiplication and scheduling.
- GatedDeltaNet layout permutations.
- LoRA GEMMs and residual accumulation.
- model-scheduler ownership and public operator-schema changes.
- modifications to selectively vendored llama.cpp source.

DeepSeek routed IQ2_XXS gate/up, routed Q2_K down, and fixed eight-group output-A are grouped workloads. They must not be flattened into dense semantics. See `docs/grouped_mmq_fwd_optimization.md`.

The dense-forward optimization and bundle conversion remained in project-owned code and did not modify the selectively vendored llama.cpp sources.

Correctness and compatibility requirements:
- independent GGUF-reference coverage for every affected quant/shape family.
- exact current-output comparison when accumulation order is unchanged.
- normalized RMSE remains approximately 0.6% for Q3_K/Q6_K and 1.1-2.0% for Q4_K/Q5_K.
- generic bounds-safe fallbacks remain available.
- no online autotuning or hidden pointer cache.

## Measurement and validation

Hardware and toolchain:

```text
GPU: Radeon 8060S Graphics
architecture: gfx1151, wave32, 40 CUs
PyTorch: 2.12.0+rocm7.15.0a20260701
HIP: 7.14.60850
```

The BF16 reference is `torch.mm(BF16 activation, dequantized BF16 weight.T)`, normally dispatched to hipBLASLt. It is an arithmetic reference, not a storage-equivalent implementation.

Measurement rules:
- run GPU benchmarks sequentially with no concurrent profiler or benchmark.
- warm module loading before timing.
- use real nonzero tensors, three warmups, and the complete production matrix.
- bracket any fresh movement above 1% with sequential 25-repeat controls.
- report unweighted per-point and checkpoint-weighted movement.
- include quantization, allocation, and workspace ownership in public-call timing.
- inspect normalized disassembly and VGPR/SGPR/LDS/private/spill/stack metadata.
- retain resource-gated kernels only with zero private storage, spills, and dynamic stack.

Benchmark examples:

```bash
source ~/venv_torch/bin/activate

PYTHONPATH=. python bench/benchmark_mmq_fwd.py \
  --model ~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf \
  --model-family qwen --batches 1,4,16 \
  --lm-head-chunks 64,128,256 --warmup 3 --repeats 9

PYTHONPATH=. python bench/benchmark_mmq_fwd.py \
  --model ~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf \
  --model-family deepseek --batches 1,4,16 \
  --lm-head-chunks 32,64,128,256,512 --warmup 3 --repeats 9
```

`--transient-bf16-control` enables the optimistic allocate/copy-predecoded-BF16/GEMM representation floor. It deliberately excludes decode compute and must not be interpreted as an implementable transient path by itself.

Final project validation:

```text
pytest -q tests/
83 passed, 14 warnings
```

The downstream complete-loss harness independently validated the final Qwen M64/M128/M256 schedule. No DeepSeek complete-loss result exists yet.

## Artifact index

Latest acceptance:
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_stride76_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_qwen_complete_loss_final_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_final_components.txt`.
- `~/tmp/torch-ggml-ops/mmq_fwd_qwen_q6_transient_floor_25.json`.

Historical retained baselines:
- `~/tmp/torch-ggml-ops/mmq_fwd_baseline_primary_sequential.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_final_full.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_plan_baseline_9.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_qwen_plan_control_9.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_p0_components.txt`.

DeepSeek exact-body controls:
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p1_generic_before_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p1_exact_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p1_generic_after_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p1_exact_bracket_25.txt`.
- `~/tmp/torch-ggml-ops/mmq_fwd_qwen_p1_control_9.json`.

DeepSeek rejected local controls:
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_k4096_j64_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_k4096_j128_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_k4096_j64_control_25.txt`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_other_j64_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_other_j128_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_other_j64_control_25.txt`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_prefetch_y_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_prefetch_baseline_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_prefetch_y_control_25.txt`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_stride77_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_stride77_control_25.txt`.

Qwen exact-body controls:
- `~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_before_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_after_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_bracket_25.txt`.

Bundle controls:
- `~/tmp/torch-ggml-ops/mmq_fwd_pre_bundle.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_post_bundle.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_embedded_pre_control_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_bundle_control_25.json`.
- `~/tmp/torch-ggml-ops/mmq_fwd_embedded_post_control_25.json`.

The `~/tmp/torch-ggml-ops` artifacts are measurement provenance, not repository inputs.

## Tool notes

- PC sampling heavily perturbs short kernels. Use it qualitatively.
- `roc-obj-ls` is broken in the active environment because of a `rocm_sdk_core._cli` import error.
- Code-object inspection uses `.hip_fatbin`, `clang-offload-bundler`, `llvm-readobj`, `llvm-nm`, and `llvm-objdump`.
