# HIP MMQ Forward Q6_K Experiment

## Scope

This record covers the Qwen language-model-head Q6_K forward kernel on gfx1151:

```text
output[M,248320] = input[M,2048] @ dequant_q6_k(weight[248320,2048]).T
```

The packed Q6_K weight has logical shape `(N,K)=(248320,2048)`. Inputs and outputs are BF16, activations use the Q8_1 F32_D4 workspace, and Q6_K values are decoded inside the packed kernel.

## Final kernel result

`HIP TFLOPS` is `2*M*N*K/time`. `HIP/torch.mm` is the throughput ratio against BF16 `torch.mm`; values above `1.00x` favor the packed HIP kernel.

| `(M,N,K)` | HIP TFLOPS | HIP/torch.mm |
| ---: | ---: | ---: |
| `(64,248320,2048)` | 15.529 | 2.13x |
| `(128,248320,2048)` | 16.309 | 1.61x |
| `(256,248320,2048)` | 15.716 | 0.81x |

The isolated M256 packed kernel is slower than BF16 `torch.mm`, even though the smaller chunks win.

## Kernel implementation

Q6_K retains the four-wave MMQ foundation but uses Q6-specific low/high payload reconstruction, signed 8-bit scales, FP16 block factors, and effective-scale application. The exact production bodies use J64 for small rows and J128 for larger rows. M256 uses two M128-style workgroups rather than one wider accumulator tile.

The common exact geometry is `I=64`, with 128 threads and K256 reduction steps. Q6-specific state is kept separate from Q3_K, Q4_K, and Q5_K identities. It is not valid to infer Q6 performance from another quant decoder's extraction or LDS layout.

## Optimization log

### Small-row geometry

The forward-specific source matrix is the final table above. A historical Q6 forward control improved M64 from `7.413 ms` to `4.202 ms` after J64 removed padded-row arithmetic; M128 and M256 did not benefit from a global J64 rule. The `5.357/9.863`, `9.084/14.480`, and `11.726/19.010 ms` rows belong to the separate backward Q6 experiment and are intentionally not copied into this forward record.

The M64 128-byte decoded-weight stride mapped row starts to the same LDS bank phase. The 16-BF16 XOR layout roughly halved latency without increasing LDS. M128 benefited from exact loader ownership. M256 was faster as two smaller workgroups because added workgroup parallelism and lower accumulator pressure outweighed repeated packed decode.

Two apparently fast M256 N=5/N=7 measurements were invalid because the N tile did not divide the 2,048-column result. The corrected N=7 body measured `29.628 ms`. K16/K64, M128 N3, M64 N3/N4, and alternate four-/16-BF16 layouts were rejected.

### Exact specialization and arithmetic boundary

Exact Q6 specialization reduced bounds and K state while retaining the packed arithmetic order. Q6 M256 is representation/arithmetic-bound: Q8_1 preparation is below `0.1%` of the call, exact bounds/K state adds only `2.09-2.56%`, and the packed result remains `16.568 ms` versus `13.462 ms` BF16.

A possible effective-scale loader would stage `float(block_d * scale)` once per row/K iteration. It changes the integer-product-first rounding order and is therefore outside bitwise current-output parity. Any future test requires independent GGUF correctness, zero private storage/spills/stack, a stable complete-call gain, and no M64/M128 regression.

### Closed mechanisms

Global J64, I128, workgroup-size, K-unroll, activation double buffering, decoded-weight LDS caching, speculative prefetch, split-K, persistent workgroups, and broad swizzle sweeps are closed. A transient BF16 stage lost by `44.21%` even while excluding decode computation and required about 1.145 GB incremental peak allocation.

## Correctness and resources

Retained Q6_K J128 bodies use `210 VGPR / 27 SGPR / 38,400 B LDS`; the small-row J64 body uses `158 VGPR / 27 SGPR / 28,928 B LDS`. Both have zero private storage, zero spills, and no dynamic stack. Validation covers low/high payload planes, all 16 signed scale groups, FP16 block factors, block boundaries, one-hot decode, Q8_1 workspace mutation, input and packed-weight mutations, independent GGUF references, finite output, and exact valid-tile divisibility.

## Evidence

```text
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_25.json
~/tmp/torch-ggml-ops/mmq_fwd_final_components.txt
```

The source-of-record set also includes the generic Qwen matrix and the exact wrapper bracket used to separate Q6 movement from placement variance:

```text
~/tmp/torch-ggml-ops/mmq_fwd_baseline_primary_sequential.json
~/tmp/torch-ggml-ops/mmq_fwd_final_full.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_plan_control_9.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_after_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_bracket_25.txt
```

A prepared integer-plus-scale representation remains the only credible path beyond the current Q6 packed decoder. It would need explicit storage, preparation, invalidation, and memory accounting; those are outside this kernel record.
