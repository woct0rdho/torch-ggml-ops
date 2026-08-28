# HIP MMQ Forward Q3_K Experiment

## Scope

This record covers the gfx1151 HIP packed-MMQ forward kernels for Q3_K weights. The kernel computes:

```text
output[M,N] = input[M,K] @ dequant_q3_k(weight[N,K]).T
```

The activation is BF16 and the output is BF16. Activations use the fixed Q8_1 F32_D4 workspace layout. The Q3_K weights remain packed GGUF data and are decoded cooperatively inside the kernel. The exact workload families are:

| Family | Logical weight `(N,K)` | M values | Tensors |
| --- | ---: | ---: | ---: |
| Attention query/query gate | `(8192,2048)` | `2048,8192,32768` | 9 |
| Narrow attention key | `(512,2048)` | `2048,8192,32768` | 9 |

The measurements below use the current packed-path matrix. `HIP TFLOPS` is dense-equivalent throughput, `2*M*N*K/time`; `HIP/torch.mm` is the throughput ratio against the BF16 `torch.mm` reference. Values above `1.00x` favor the packed HIP kernel.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm |
| --- | ---: | ---: | ---: |
| Query/query gate | `(2048,8192,2048)` | 22.975 | 1.10x |
| Query/query gate | `(8192,8192,2048)` | 22.221 | 1.06x |
| Query/query gate | `(32768,8192,2048)` | 21.675 | 1.02x |
| Narrow key | `(2048,512,2048)` | 19.260 | 1.42x |
| Narrow key | `(8192,512,2048)` | 18.694 | 1.03x |
| Narrow key | `(32768,512,2048)` | 17.762 | 0.95x |

The narrow Q3_K B16 point is below the BF16 baseline in the complete packed path. Its packed multiply alone was `2.725 ms` versus `3.691 ms` for BF16; the remaining loss was the Q8_1 activation producer, which contributed about `24.4%` of that call.

## Kernel implementation

The retained body uses `I=64`, `J=128`, 128 threads, four wave32 waves, and a `K=256` reduction step. Exact Q3_K K2048 bodies fold the matrix dimensions and packed offsets into the generated kernel while retaining bounds-safe fallback code for other shapes. Packed payload and scale state are held only for the active decode phase so it does not extend through the WMMA loop.

The fixed Q8_1 producer uses one 512-thread workgroup per real activation row. It preserves the F32_D4 metadata, signed-int8 rounding, reduction, and workspace semantics used by the packed multiply. The producer and multiply are separate kernels.

## Optimization log

### Activation and initial geometry

The initial Q8_1 launch used one 64-thread workgroup per `(padded row, K/256 block)`. Replacing it with one 512-thread workgroup per real row removed repeated row work and reduced the narrow Q4_K producer from `5,105.979 us` to `886.928 us` at M32768. The same producer change exposed the Q3_K multiplication body as the remaining narrow cost.

The first four-wave tiled body replaced scalar 16x16 decode ownership with cooperative packed decode, LDS-staged weights, multiple WMMA accumulators, and exact row/type geometry. The final general geometry remained 128x128/K32; global J64, I128, smaller workgroups, and broad grouped-M rules did not improve the production mix.

The narrow source timings are `0.223/0.919/3.869 ms` at M2048/M8192/M32768; their dense-equivalent rates are the three narrow rows in the final table. The initial Qwen query improvement came from the same four-wave foundation and the 512-thread Q8_1 producer, while the exact wrapper is measured against the source-built control.

### Packed extraction and LDS layout

Wide Q3_K packed extraction, two-row prefetch, and the eight-BF16 XOR LDS layout were retained for the query body. Removing the extraction/prefetch/layout combination regressed the measured controls by `1.06-11.78%` depending on the point. Narrow Q3_K uses explicit packed state but does not prefetch because its 110-byte block layout made that path slower.

The direct wide-Q3 extraction control reduced query latency from `48.694 ms` to `46.980 ms`. Keeping the next reduction iteration's packed fragments live across WMMA instead reached `48.606 ms`, so the retained prefetch is bounded to the active decode phase.

Aligned local fragment loads improved narrow Q3_K but were not a universal rule: the same change regressed query Q4_K, Q5_K, and shared-down bodies. The final choice is therefore type- and shape-specific rather than a global vector-load policy.

### Exact specialization

The Q3_K K2048 exact wrapper was retained with a full J128 body. Across its measured matrix, exact specialization improved the generic control by `0.90-5.26%`. The detached build confirmed that the Q3_K artifact was unchanged while the separate DeepSeek Q8_0 phase was optimized.

## Correctness and resources

The exact Q3_K J128 body uses `196 VGPR / 27 SGPR / 40,448 B LDS`; retained artifacts have zero private storage, zero spills, and no dynamic stack. Correctness coverage includes independent GGUF decode, one-hot and block-boundary payloads, signed scale fields, Q8_1 workspace mutation, input mutation, packed-weight mutation, finite output, and exact output comparison whenever accumulation order is unchanged. Generic bounds-safe bodies remain available outside exact shape coverage.

## Evidence

The current source-of-record measurements are:

```text
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_after_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_bracket_25.txt
~/tmp/torch-ggml-ops/mmq_fwd_final_components.txt
```

The standalone artifact controls also include the retained generic sequential baseline and the detached Qwen control used to check that DeepSeek exact-body work did not alter Q3_K bytes:

```text
~/tmp/torch-ggml-ops/mmq_fwd_baseline_primary_sequential.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_plan_control_9.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p1_control_9.json
~/tmp/torch-ggml-ops/mmq_fwd_pre_bundle.json
~/tmp/torch-ggml-ops/mmq_fwd_post_bundle.json
~/tmp/torch-ggml-ops/mmq_fwd_embedded_pre_control_25.json
~/tmp/torch-ggml-ops/mmq_fwd_bundle_control_25.json
~/tmp/torch-ggml-ops/mmq_fwd_embedded_post_control_25.json
```

Broad J64, I128, workgroup-size, split-K, decoded-weight LDS caching, speculative cross-iteration prefetch, and generic swizzle sweeps are closed for this packed representation. A future change needs a new decode, representation, or activation-reuse premise and must be measured against both matrix families.
