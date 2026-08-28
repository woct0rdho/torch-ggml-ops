# HIP MMQ Backward Q3_K Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ input-gradient kernels for Q3_K weights:

```text
grad_input[M,N] = grad_output[M,K] @ dequant_q3_k(weight[N,K])
```

The cotangent and gradient are BF16. The packed Q3_K weight remains in forward layout and is decoded cooperatively with FP32 WMMA accumulation. The measured families are the Qwen query and narrow projections.

| Family | Forward weight `(N,K)` | Backward shape `(M,N,K)` | M values |
| --- | ---: | ---: | ---: |
| Query/query gate | `(8192,2048)` | `(M,2048,8192)` | `2048,8192,32768` |
| Narrow attention key | `(512,2048)` | `(M,2048,512)` | `2048,8192,32768` |

`HIP TFLOPS` is `2*M*N*K/time`. `HIP/torch.mm` is the packed-throughput ratio against BF16 `torch.mm`; values above `1.00x` favor the HIP kernel.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm |
| --- | ---: | ---: | ---: |
| Query/query gate | `(2048,2048,8192)` | 20.141 | 1.230x |
| Query/query gate | `(8192,2048,8192)` | 21.745 | 1.278x |
| Query/query gate | `(32768,2048,8192)` | 22.212 | 1.241x |
| Narrow key | `(2048,2048,512)` | 24.403 | 1.079x |
| Narrow key | `(8192,2048,512)` | 23.828 | 1.059x |
| Narrow key | `(32768,2048,512)` | 24.257 | 1.059x |

The values use the current packed/BF16 matrix and report the complete packed-gradient path represented there.

## Kernel implementation

The retained body uses four wave32 waves, a 128x128/K32 ordinary tile, cooperative Q3_K extraction, decoded-weight LDS staging, FP32 WMMA accumulation, and BF16 stores. The Q3_K decoder reconstructs packed payload and signed scale state during each K block while keeping the forward-layout packed weight authoritative.

Exact shape and row-count specialization removes runtime bounds and address state from the common production shapes. Generic bounds-safe bodies remain available for unmatched shapes. Cotangents are not quantized and no dense transposed weight is materialized.

## Optimization log

### First tiled redesign

The original body decoded one packed value at a time into a 16x16 tile and sustained only about `0.05-0.13x` BF16 throughput. The retained redesign added four wave32 waves, LDS-staged decoded weights, multiple WMMA accumulator tiles, cooperative pair/quad/width-16 decode, and measured row/type-specific ownership.

Representative first redesign results included query Q3_K M32768 moving from `1,108.314 ms` to `162.884 ms`. The same four-wave structure became the Q3_K foundation for narrow and query shapes.

### Extraction, prefetch, and LDS layout

Wide Q3_K packed extraction, two-row prefetch, and the eight-BF16 XOR LDS layout were retained for the query body. Removing those controls regressed by `1.06-11.78%` depending on the measured point. Narrow Q3_K does not prefetch because its 110-byte block layout made the path slower.

K-loop unrolling, activation-half double buffering, generic padding, decoded-weight LDS caching, and broad local-load rules were rejected. The accepted Q3 path keeps packed state bounded to the active decode phase; keeping the next iteration's packed fragments live across WMMA extended register lifetimes without a timing benefit.

### Packaged-kernel controls

The source-built HSACO conversion produced a `+0.56%` initial geometric movement and a `+1.12%` embedded/bundle bracket movement, while embedded controls themselves drifted by `+1.04%`. Q3 query showed `2.9-7.0%` placement-sensitive movement without a device semantic change. Warm standalone modules, normalized ISA, and sequential controls are required before treating a timing change as a kernel result.

## Correctness and resources

The retained Q3_K query body uses `237 VGPR / 27 SGPR / 8 KiB LDS`; it has zero private storage, zero VGPR/SGPR spills, and no dynamic stack. Validation includes independent GGUF dequantization, one-hot and block-boundary decode, input-gradient comparison, autograd, input/weight/cotangent mutation, exact-tile guards, and row-boundary coverage.

## Evidence

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_narrow_q5_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb0_folded_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_selected_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_scalar_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_no_prefetch_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_no_swizzle_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_selected_after_25.json
```

The source log's shared-body and packaging controls are:

```text
~/tmp/torch-ggml-ops/mmq_bwd_baseline_primary_sequential.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb0_folded_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_post_ds4_p1_control_9.json
~/tmp/torch-ggml-ops/mmq_bwd_pre_bundle.json
~/tmp/torch-ggml-ops/mmq_bwd_post_bundle.json
~/tmp/torch-ggml-ops/mmq_bwd_embedded_pre_control_25.json
~/tmp/torch-ggml-ops/mmq_bwd_bundle_control_25.json
~/tmp/torch-ggml-ops/mmq_bwd_embedded_post_control_25.json
```

The current Q3_K kernel record closes global J64, I128, alternate workgroup sizes, split-K, persistent workgroups, speculative prefetch, and broad swizzle changes for the existing packed representation.
