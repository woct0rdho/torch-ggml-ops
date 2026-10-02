# HIP MMQ Backward Q5_K Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ input-gradient kernels for Q5_K weights.

| Family | Forward weight `(N,K)` | Backward shape `(M,N,K)` | M values |
| --- | ---: | ---: | ---: |
| Narrow K/V/shared gate/up | `(512,2048)` | `(M,2048,512)` | `2048,8192,32768` |
| Shared-expert down | `(2048,512)` | `(M,512,2048)` | `2048,8192,32768` |

Backward shapes are written `(M, in_features, out_features)`, matching the weight's `(N,K) = (out_features, in_features)`.

## Final kernel result

| Family | `(M,K,N)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| Narrow K/V/gate/up | `(2048,2048,512)` | 20.034 | 0.895x | `dense_bwd_q5_k_mt128_nt128_ki32_full_k512` |
| Narrow K/V/gate/up | `(8192,2048,512)` | 21.898 | 0.918x | `dense_bwd_q5_k_mt128_nt128_ki32_full_k2048` |
| Narrow K/V/gate/up | `(32768,2048,512)` | 22.789 | 0.936x | `dense_bwd_q5_k_mt128_nt128_ki32_full_k2048` |
| Shared down | `(2048,512,2048)` | 17.612 | 1.372x | `dense_bwd_q5_k_mt128_nt128_ki32_full_k2048` |
| Shared down | `(8192,512,2048)` | 11.141 | 0.731x | `dense_bwd_q5_k_mt128_nt128_ki32_full_k512` |
| Shared down | `(32768,512,2048)` | 12.194 | 0.732x | `dense_bwd_q5_k_mt128_nt128_ki32_full_k2048` |

The `Kernel` column names the deployed body for each exact key; it is the fastest built body whose output is bitwise equal to the reference body in the per-key candidate campaign. The shared-down rows show the largest per-sample spread in the matrix (`1.20-1.53` max/min here); the narrow rows stay at `1.05-1.07`.

## Kernel implementation

The retained body uses four wave32 waves, exact 128x128/K32 ownership where applicable, decoded-weight LDS staging. The build carries two tuned `mt128_nt128_ki32_full` variants (`_k512`, `_k2048`), the `full_k2048_scalar_extraction` control, and five legacy generic bodies; the tuned variants and the scalar control were timed on every deployment key, and the deployed body per key is the fastest of those. Q5-specific packed-row and signed-scale state stays separate from Q4_K.

## Optimization log

### Tiled redesign and shape specialization

The first four-wave Qwen body replaced scalar decode and serial 16x16 ownership with cooperative extraction, LDS staging, multiple WMMA accumulators, and exact shape/row geometry. Exact Q5_K wrappers then removed runtime bounds and address state while retaining bounds-safe fallback code.

The exact Q5_K matrix improved its generic controls by `11.66-32.60%`. The smallest narrow row count is the main extraction discriminator; larger rows favor the packed path.

### Extraction and LDS controls

Q5_K reused the Q4_K framework for width-16 low/high decode and bounded packed prefetch. Prefetch improved the initial port by `12-24%`. A universal padded layout regressed Q5_K and was rejected, and aligned fragment loads regressed Q5_K by about `2-4%` in cross-format controls.

Narrow Q5 retained scalar extraction only at 2,048 rows, where it improved `17.41%`; it regressed `2.34%` and `6.52%` at 8,192 and 32,768 rows. Shared-down Q5 retained its four-BF16 XOR layout after an isolated swizzle8 improvement did not survive the complete matrix.

The accepted body keeps packed/decode temporaries dead before long WMMA phases. Cross-iteration packed prefetch, K64, double buffering, decoded-weight LDS caching, and broad swizzle changes either extended register lifetimes or reduced residency.

### Closed directions

The current source closes global J64, I128, alternate workgroup sizes, split-K, GSU, Stream-K, persistent workgroups, generic local-load rules, and compiler-managed prefetch arrays. Q5-specific results must not be inferred from Q4_K or Q3_K because packed high-bit reconstruction changes the register and LDS cost model.

## Resources

The retained Q5_K bodies use `247 VGPR / 17 SGPR / 8 KiB LDS` for narrow scalar extraction and `253 VGPR / 16 SGPR / 8 KiB LDS` for shared down.

## Evidence

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/hip_vs_baseline/pass11_ordbwd_qwen.json
~/tmp/torch-ggml-ops/hip_selection/           (per-key candidate campaign)
tools/configs/hip_deployment.json             (deployed body per key)
```

The original campaign evidence is:

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_narrow_q5_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_narrow_selected_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_narrow_scalar_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_narrow_selected_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_shared_swizzle4_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_shared_swizzle8_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_shared_swizzle4_after_25.json
```

The shared-down Q5_K body reaches 253 VGPRs but remains spill-free. A new experiment must reduce decode state or change the packed representation before reopening the closed geometry neighborhood.
