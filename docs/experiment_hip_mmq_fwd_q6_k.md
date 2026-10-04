# HIP MMQ Forward Q6_K Experiment

## Scope

This record covers the Qwen language-model-head Q6_K forward kernel on gfx1151, and the Qwen4-Exp ordinary projections that share the type.

Q6_K also carries the QSA attention key/value and output and shared-expert gate/up shapes of the `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` checkpoint at hidden size 2560, at the training token counts of a sequence length 2048 batch (B1/B4/B16).

The type appears in no GatedDeltaNet `in_proj_qkv` or `in_proj_z` tensor, and GatedDeltaNet `out_proj` stays deferred because wiring it needs the activation permutation.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| ---: | --- | --- | --- | --- |
| Language model head | `(64,248320,2048)` | 20.84 | 2.77x | `dense_fwd_q6_k_k2048_j64_full_hoisted` |
| Language model head | `(128,248320,2048)` | 22.81 | 2.19x | `dense_fwd_q6_k_k2048_j128_full_hoisted` |
| Language model head | `(256,248320,2048)` | 22.78 | 1.11x | `dense_fwd_q6_k_k2048_j128_full_hoisted` |
| QSA key/value | `(2048,512,2560)` | 21.355 | 1.54x | `dense_fwd_q6_k_k2560_j128_full_hoisted` |
| QSA key/value | `(8192,512,2560)` | 24.214 | 1.31x | `dense_fwd_q6_k_k2560_j128_full_hoisted` |
| QSA key/value | `(32768,512,2560)` | 23.608 | 1.17x | `dense_fwd_q6_k_k2560_j128_full_hoisted` |
| QSA output | `(2048,2560,6144)` | 23.922 | 1.04x | `dense_fwd_q6_k_k6144_j128_full_hoisted` |
| QSA output | `(8192,2560,6144)` | 23.688 | 1.18x | `dense_fwd_q6_k_k6144_j128_full_hoisted` |
| QSA output | `(32768,2560,6144)` | 22.375 | 1.15x | `dense_fwd_q6_k_k6144_j128_full_hoisted` |
| Shared-expert gate/up | `(2048,640,2560)` | 21.907 | 1.53x | `dense_fwd_q6_k_k2560_j128_full_hoisted` |
| Shared-expert gate/up | `(8192,640,2560)` | 24.239 | 1.68x | `dense_fwd_q6_k_k2560_j128_full_hoisted` |
| Shared-expert gate/up | `(32768,640,2560)` | 23.479 | 1.56x | `dense_fwd_q6_k_k2560_j128_full_hoisted` |

The Qwen head shapes beat BF16 `torch.mm` by `2.77x` for the widest and `1.11x` for the narrowest, previously the one losing case. The Qwen4 projection shapes stay above the baseline as well, from `1.04x` up.

## Kernel implementation

Q6_K retains the four-wave MMQ foundation. The exact production bodies use J64 for small rows and J128 for larger rows. M256 uses two M128-style workgroups rather than one wider accumulator tile. The deployed bodies use the hoisted epilogue described below: the per-row scale pair is read and converted once per contraction step instead of once per accumulator element and column tile.

The common exact geometry is `I=64`, with 128 threads and K256 reduction steps, and exact wrappers for K2048, K2560, and K6144. Q6-specific state is kept separate from Q3_K, Q4_K, and Q5_K identities. It is not valid to infer Q6 performance from another quant decoder's extraction or LDS layout.

## Optimization log

### Small-row geometry

The forward-specific source matrix is the final table above. A historical Q6 forward control improved M64 from `7.413 ms` to `4.202 ms` after J64 removed padded-row arithmetic. M128 and M256 did not benefit from a global J64 rule. The `5.357/9.863`, `9.084/14.480`, and `11.726/19.010 ms` rows belong to the separate backward Q6 experiment and are intentionally not copied into this forward record.

The M64 128-byte decoded-weight stride mapped row starts to the same LDS bank phase. The 16-BF16 XOR layout roughly halved latency without increasing LDS. M128 benefited from exact loader ownership. M256 was faster as two smaller workgroups because added workgroup parallelism and lower accumulator pressure outweighed repeated packed decode.

Two apparently fast M256 N=5/N=7 measurements were invalid because the N tile did not divide the 2,048-column result. The corrected N=7 body measured `29.628 ms`. K16/K64, M128 N3, M64 N3/N4, and alternate four-/16-BF16 layouts were rejected.

### Exact specialization and arithmetic boundary

Exact Q6 specialization reduced bounds and K state while retaining the packed arithmetic order. Q6 M256 is representation/arithmetic-bound: Q8_1 preparation is below `0.1%` of the call, exact bounds/K state adds only `2.09-2.56%`, and the packed result remains `16.568 ms` versus `13.462 ms` BF16.

A possible effective-scale loader would stage `float(block_d * scale)` once per row/K iteration. Any future test requires a stable complete-call gain and no M64/M128 regression.

### Hoisted epilogue metadata

The vendored Q6_K vec-dot target reloads `x_df` and the int8 scale array inside the column loop even though both depend only on the row and the contraction step. The hoisted body computes the row product `x_df[i * sram_stride] * sc[k01 / 4]` once per row and contraction step, outside the column loop, and keeps the same association and accumulation order, so the arithmetic is unchanged and only the placement of the loads moves.

A dense-forward A/B over all 50 ordinary shapes with the official protocol reports `1.270x`, `1.300x` and `1.299x` for the three language-model-head shapes and neutral or sub-percent effects for Q3_K, Q4_K, Q5_K and Q8_0, whose targets are either dominated by other costs or already hoisted by the compiler. Only the Q6_K keys deploy the hoisted body. Evidence: `~/tmp/torch-ggml-ops/retune_fwd/official/hoisted_OrdinaryForward_qwen.json`, `~/tmp/torch-ggml-ops/retune_fwd/ab.py`.

### K2560, K6144, and the hoisted epilogue on the new shapes

The Qwen4-Exp shapes added two contraction lengths, `K=2560` and `K=6144`, to the head's `K=2048`. The hoisted epilogue is not a head-only effect: on the nine new projection points it beats the vendored target by `1.247-1.354x`, and it beats the generic control by `1.336-1.404x`, so the new keys deploy the hoisted K2560 and K6144 bodies. On the head it reproduces the recorded movement (`1.286-1.317x` over the vendored target, `2.404x` at `M=64` where the generic control is also stuck on a J128 tile), so the head rows are unchanged. Hoisted and vendored outputs are bitwise identical on all new shapes and both head chunks, which is the expected result of moving loads without touching the arithmetic.

### Closed mechanisms

Global J64, I128, workgroup-size, K-unroll, activation double buffering, decoded-weight LDS caching, speculative prefetch, split-K, persistent workgroups, and broad swizzle sweeps are closed. A transient BF16 stage lost by `44.21%` even while excluding decode computation and required about 1.145 GB incremental peak allocation.

## Resources

Retained Q6_K J128 bodies use `195 VGPR / 27 SGPR / 38,400 B LDS`. The small-row J64 body uses `127 VGPR / 27 SGPR / 28,928 B LDS`.

## Evidence

```text
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_25.json
~/tmp/torch-ggml-ops/mmq_fwd_final_components.txt
```

The Qwen4-Exp points come from outside the public case list, because those shapes have HIP controls but no GGTensile problem key yet, so they cannot be selected as deployment cases. Their runs use the deployed Q8_1 producer and the same prepared inputs, the same `torch.mm` BF16 baseline on the same activation tensor, and the same paired timing as the official protocol:

```text
~/tmp/torch-ggml-ops/qwen4_fwd/sweep_q6k_v1.txt
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

A prepared integer-plus-scale representation remains the only credible path beyond the current Q6 packed decoder. It would need explicit storage, preparation, invalidation, and memory accounting. Those are outside this kernel record.
