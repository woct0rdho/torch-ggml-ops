# HIP MMQ Forward Q5_K Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ forward for Q5_K weights:

```text
output[M,N] = input[M,K] @ dequant_q5_k(weight[N,K]).T
```

The input and output are BF16. The packed Q5_K weight is decoded cooperatively from GGUF storage. The Q8_1 F16_D4S4 workspace provides the signed-int8 values and scale/sum metadata consumed by the Q5_K correction path.

| Family | Logical weight `(N,K)` | M values | Tensors |
| --- | ---: | ---: | ---: |
| Narrow K/V/shared gate/up | `(512,2048)` | `2048,8192,32768` | 21 |
| Shared-expert down | `(2048,512)` | `2048,8192,32768` | 10 |

`HIP TFLOPS` is the dense-equivalent `2*M*N*K/time`. `HIP/torch.mm` is the throughput ratio against BF16 `torch.mm`; values above `1.00x` favor the packed HIP path.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm |
| --- | ---: | ---: | ---: |
| Narrow K/V/gate/up | `(2048,512,2048)` | 21.913 | 1.62x |
| Narrow K/V/gate/up | `(8192,512,2048)` | 20.452 | 1.13x |
| Narrow K/V/gate/up | `(32768,512,2048)` | 20.135 | 1.07x |
| Shared down | `(2048,2048,512)` | 23.470 | 7.89x |
| Shared down | `(8192,2048,512)` | 24.163 | 7.43x |
| Shared down | `(32768,2048,512)` | 23.115 | 7.01x |

The table uses the current complete packed-path matrix and its BF16 `torch.mm` ratios.

## Kernel implementation

The retained body uses the four-wave `I=64`, `J=128`, K256 geometry with Q5-specific low/high payload reconstruction, signed scales, and FP32 correction. Exact K512 and K2048 wrappers fold packed-row bytes and full-tile bounds while preserving a generic bounds-safe fallback.

Q5_K uses the Q4_K-style scale-plus-sum activation workspace. Q5-specific payload state is kept bounded to the active reduction phase; broad cross-iteration packed prefetch was rejected because longer live ranges outweighed the load savings.

## Optimization log

### Initial producer and tiled body

The shared Q8_1 producer was changed from one 64-thread workgroup per padded row/block to one 512-thread workgroup per real row. The narrow Q4_K producer diagnostic fell from `5,105.979 us` to `886.928 us` at M32768. The four-wave tiled MMQ body then became the retained Q5_K foundation, replacing scalar 16x16 ownership with cooperative decode, LDS staging, and multiple WMMA accumulators.

### Q5 extraction and LDS choices

Q5_K padding did not transfer from Q4_K: the broad padded layout regressed Q5_K and was rejected. Explicit aligned fragment loads also regressed Q5_K by about `2-4%` in the shared cross-format controls. The final Q5 path therefore keeps its own packed extraction and four-BF16 shared-down layout rather than inheriting a global padding or vector-load rule.

The exact Qwen wrappers improved their generic controls by `11.66-32.60%`. Narrow Q5 extraction is selected by row regime in the measured kernel records: scalar extraction improved the smallest row count by `17.41%`, but regressed the larger row counts by `2.34%` and `6.52%`. Shared-down Q5 retained the four-BF16 XOR layout after an isolated swizzle8 gain failed to survive the complete matrix.

The cross-format extraction controls also established that Q5 padding regresses, while generic aligned fragment loads lose about `2-4%`. Shared-down alternatives with 4x4 or 1x16 ownership reached roughly `7.3-9.2 ms`; K64 reached `6.565 ms`; and disabling packed-byte prefetch reduced the artifact to 234 VGPRs but reached `6.186 ms`. These controls closed the local tile and register-pressure neighborhood rather than only one source variant.

The standalone small-route J32 body improved nonuniform routes by approximately `14-16%` but regressed the uniform route. That is why the small-row mechanism is retained as a bounded Q5 identity and not treated as a universal replacement.

### Closed mechanisms

Global J64, I128, alternate workgroup sizes, K64, activation-half double buffering, decoded-weight LDS caching, broad packed prefetch, split-K, persistent workgroups, and generic swizzle rules are closed for the current Q5 representation. A future experiment must first demonstrate lower decode state or a lossless prepared representation.

## Correctness and resources

Retained Q5_K J128 bodies use `244 VGPR / 28 SGPR / 38,400 B LDS`; they have zero private storage, zero spills, and no dynamic stack. Correctness covers low and high payload planes, signed scale fields, block boundaries, Q8_1 scale/sum metadata, workspace/input/weight mutations, independent GGUF reference output, finite values, and exact full-tile guards. The shared-down body is near the practical allocation warning point at 253 VGPRs but remains spill-free.

## Evidence

```text
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_after_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_bracket_25.txt
~/tmp/torch-ggml-ops/mmq_fwd_final_components.txt
```

The exact Q5 qualification also uses the common sequential baseline and full exact-wrapper bracket:

```text
~/tmp/torch-ggml-ops/mmq_fwd_baseline_primary_sequential.json
~/tmp/torch-ggml-ops/mmq_fwd_final_full.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_plan_control_9.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_after_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_bracket_25.txt
```

The current result is a shape-specific packed decoder, not evidence that Q5_K should use the Q4_K schedule or that a single padding/swizzle policy is portable across quant types.
