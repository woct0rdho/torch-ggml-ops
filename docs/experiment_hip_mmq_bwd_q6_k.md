# HIP MMQ Backward Q6_K Experiment

## Scope

This record covers the Qwen language-model-head Q6_K packed input-gradient kernel on gfx1151.

## Final kernel result

| `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| ---: | ---: | ---: | --- |
| `(64,2048,248320)` | 12.111 | 1.841x | `dense_bwd_q6_k_m64_nt32_ki64_full` |
| `(128,2048,248320)` | 13.757 | 1.533x | `dense_bwd_q6_k_m128_nt64_ki32_full` |
| `(256,2048,248320)` | 21.449 | 1.564x | `dense_bwd_q6_k_m256_nt64_ki32_full` |

The values use the current Q6_K packed/BF16 kernel matrix. M256 is the primary large chunk, with M64 and M128 as smaller exact geometries. The `Kernel` column names the deployed body for each chunk; it is the fastest built body whose output is bitwise equal to the reference body in the per-key candidate campaign, and the `_full_*` bodies are the unbounded exact variants (`_bounded` builds exist for shapes outside the exact-tile contract).

## Kernel implementation

The exact bodies are `dense_bwd_q6_k_m64_nt32_ki64_full`, `dense_bwd_q6_k_m128_nt64_ki32_full`, and `dense_bwd_q6_k_m256_nt64_ki32_full`: M64/N32/K64 for M64, M128/N64/K32 for M128, and two M128-style workgroups for M256. The decoded-weight LDS layout, packed extraction, and register lifetime are Q6-specific. The `_bounded` twins and the `nt128_ki16_g2`/`nt256_ki16_g2` generic bodies are built but are not competitive on the three deployment chunks.

## Optimization log

### Initial tiled redesign

The first redesign added four wave32 waves, LDS-staged decoded weights, multiple WMMA accumulators, cooperative packed decode, and exact row geometry. The representative LM-head Q6_K case moved from `158.209 ms` to `26.130 ms` for M256.

The selected small-row geometry screen was:

| M | Retained geometry and layout | Packed/BF16 timing |
| ---: | --- | ---: |
| 64 | M64/N32/K64, 16-BF16 XOR | `5.357/9.863 ms`, `1.84x` |
| 128 | M128/N64/K32, eight-BF16 XOR | `9.084/14.480 ms`, `1.59x` |
| 256 | Two M128/N64/K32 workgroups, packed extraction, eight-BF16 XOR | `11.726/19.010 ms`, `1.62x` |

M64 benefited from a 16-BF16 XOR layout that changed the decoded row bank phase without increasing LDS. M128 benefited from exact loader ownership. M256 was faster as two smaller workgroups because parallelism and lower accumulator pressure outweighed repeated packed decode.

### Invalid and rejected shapes

Two apparently fast M256 N=5/N=7 measurements were invalid: the selected N tile did not divide the 2,048-column result. The corrected N=7 body measured `29.628 ms`. K16/K64, M128 N3, M64 N3/N4, and alternate four-/16-BF16 layouts were rejected.

Packed extraction remained selected for M256; scalar extraction regressed `2.14%`. Exact bounds and K state accounted for only `2.09-2.56%` of the final M64/M128/M256 result, so the remaining M256 cost is packed Q6 arithmetic/representation rather than generic bounds handling.

### Arithmetic boundary

An effective-scale loader could stage `float(block_d * scale)` once per decoded row/K iteration. A future approximate-order experiment must show stable complete-kernel timing across all three chunks.

Global J64, I128, broad K64, activation double buffering, decoded-weight caching, speculative prefetch, split-K, persistent workgroups, and broad swizzle sweeps are closed for the present arithmetic contract.

## Resources

Retained Q6_K bodies use `87/138/137 VGPR` for M64/M128/M256, `15/16/15 SGPR`, and 4 KiB LDS.

## Evidence

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/hip_vs_baseline/pass11_ordbwd_qwen.json
~/tmp/torch-ggml-ops/hip_selection/           (per-key candidate campaign)
tools/configs/hip_deployment.json             (deployed body per chunk)
```

The original campaign evidence is:

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_packed_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_scalar_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_packed_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_narrow_q5_25.json
```

The Q6-specific shape and arithmetic controls are:

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_packed_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_scalar_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_packed_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_post_ds4_p3_control_9.json
```
