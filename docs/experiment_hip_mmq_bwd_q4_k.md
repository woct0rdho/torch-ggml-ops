# HIP MMQ Backward Q4_K Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ input-gradient kernels for Q4_K weights.

Backward shapes are written `(M, in_features, out_features)`, matching the weight's `(N,K) = (out_features, in_features)`. Beyond the Qwen query, narrow, attention-output and shared-down families measured first, the type carries the QSA attention key/value and output, shared-expert gate/up and GatedDeltaNet `in_proj_qkv` and `in_proj_z` projections of the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (hidden size 2560) and the GatedDeltaNet `in_proj_z` of the Qwen3.6-35B-A3B (APEX-I-Mini) checkpoint, at the training token counts of a sequence length 2048 batch (B1/B4/B16). The APEX-I-Mini `in_proj_qkv` `(8192,2048)` shares the already tabulated query shape.

GatedDeltaNet `out_proj` is deferred because wiring it needs the activation permutation.

## Final kernel result

| Family | `(M,K,N)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| Query/query gate | `(2048,2048,8192)` | 20.834 | 1.225x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| Query/query gate | `(8192,2048,8192)` | 22.605 | 1.254x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k2048` |
| Query/query gate | `(32768,2048,8192)` | 23.240 | 1.261x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k2048` |
| Narrow K/V/gate/up | `(2048,2048,512)` | 20.580 | 0.884x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| Narrow K/V/gate/up | `(8192,2048,512)` | 22.688 | 0.928x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| Narrow K/V/gate/up | `(32768,2048,512)` | 24.021 | 0.978x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| Attention output | `(2048,4096,2048)` | 26.648 | 1.133x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| Attention output | `(8192,4096,2048)` | 23.975 | 1.004x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| Attention output | `(32768,4096,2048)` | 24.381 | 1.009x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| Shared down | `(2048,512,2048)` | 18.269 | 1.422x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| Shared down | `(8192,512,2048)` | 11.368 | 0.745x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| Shared down | `(32768,512,2048)` | 13.118 | 0.789x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k512` |
| QSA key/value | `(2048,2560,512)` | 23.237 | 1.111x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| QSA key/value | `(8192,2560,512)` | 24.377 | 0.914x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| QSA key/value | `(32768,2560,512)` | 25.726 | 0.963x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| QSA output | `(2048,6144,2560)` | 26.524 | 1.075x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| QSA output | `(8192,6144,2560)` | 27.329 | 1.063x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| QSA output | `(32768,6144,2560)` | 26.211 | 1.046x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| Shared-expert gate/up | `(2048,2560,640)` | 24.031 | 1.111x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| Shared-expert gate/up | `(8192,2560,640)` | 26.262 | 0.945x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| Shared-expert gate/up | `(32768,2560,640)` | 26.927 | 0.951x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| GatedDeltaNet Z | `(2048,2560,6144)` | 22.269 | 0.955x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| GatedDeltaNet Z | `(8192,2560,6144)` | 23.467 | 0.943x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k2048` |
| GatedDeltaNet Z | `(32768,2560,6144)` | 23.586 | 0.949x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k2048` |
| GatedDeltaNet QKV | `(2048,2560,10240)` | 21.899 | 0.923x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| GatedDeltaNet QKV | `(8192,2560,10240)` | 23.905 | 0.926x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k2048` |
| GatedDeltaNet QKV | `(32768,2560,10240)` | 23.391 | 0.939x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k2048` |
| GatedDeltaNet Z APEX-I-Mini | `(2048,2048,4096)` | 24.702 | 1.439x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| GatedDeltaNet Z APEX-I-Mini | `(8192,2048,4096)` | 22.459 | 1.230x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |
| GatedDeltaNet Z APEX-I-Mini | `(32768,2048,4096)` | 22.381 | 1.256x | `dense_bwd_q4_k_mt128_nt128_ki32_full_k4096` |

The current source matrix is the complete packed-gradient path. The `Kernel` column names the deployed body for each exact key. It is the fastest built body whose output is bitwise equal to the reference body in the per-key candidate campaign.

### Qwen4-Exp and GatedDeltaNet shapes

The new shapes span contraction lengths from `2048` to `10240` and result widths from `512` to `2560`, all whole multiples of the `ki32` step and the 128-wide tiles, so the three tuned variants serve them without a new body. All three were timed on every point and their outputs are bitwise identical, as on the keys measured first.

The `_k512` variant loses everywhere on the new points (`1.13-1.32x` behind the winner), which leaves the choice between `_k4096` and `_k2048`: the QSA key/value, QSA output, shared-expert gate/up and APEX-I-Mini `in_proj_z` rows take `_k4096` at every M (`1.02-1.06x` ahead), while the two long-contraction Qwen4 GatedDeltaNet rows, `(M,2560,6144)` and `(M,2560,10240)`, take `_k4096` at `M=2048` and `_k2048` above it, where it is `1.9-3.0%` faster. The deployed body therefore follows the measurement per key rather than the suffix.

Six of the new rows are below the BF16 `torch.mm` baseline (the long contractions, at `0.92-0.96x`) and twelve are above it (`0.95-1.44x`), the same spread the rows measured first show, so the new shapes change the mixture rather than the conclusion: the remaining cost is the packed decode the BF16 baseline does not pay.

### Exact-dimension twins

The deployed bodies on this record resolve both the contraction and the result width at runtime. Twins that copy the deployed geometry exactly and only substitute the two compile-time bounds were built and measured against the deployed bodies in one interleaved A/B run, with identical prepared inputs and bitwise identical output: `0.858x` at `(8192,2048,8192)` and `0.886x` at `(32768,512,2048)`. Exact dimensions are therefore not deployed on these keys. Where the mechanism looked positive on the Q6_K M64 chunk, the measurement also carried the split-contraction body.

## Kernel implementation

The retained Q4_K bodies are a four-wave 128x128/K32 tiled decoder with decoded-weight LDS staging. The build carries three tuned `mt128_nt128_ki32_full` variants (`_k512`, `_k2048`, `_k4096`) plus five legacy generic bodies. All three tuned variants were timed on every deployment key. The deployed body per key is the fastest one, which does not follow the variant suffix: `_k4096` covers the query, narrow, and attention-output rows, `_k2048` the two longer query rows, and `_k512` the B16 shared-down row.

Bounds-safe tails and generic fallbacks remain outside the exact tiled bodies and are not competitive on these keys.

## Optimization log

### Qwen tiled geometry

The first redesign replaced scalar 16x16 ownership with four wave32 waves, multiple accumulators, cooperative packed decode, and LDS-staged weights. The common retained geometry is 128x128/K32. Early global J64 and larger ownership alternatives lost reuse, parallelism, or residency.

Q4_K shape-specific controls retain padded vector local loads for query/narrow and a 16-BF16 XOR layout for shared down. Shared-down K512 is a distinct cost model and does not inherit the query layout.

### Padding and local-load experiments

Pre-layout Q4_K controls reported a repeated `79.2%` LDS-bank-conflict metric. Eight BF16 values of row padding changed the K32 stride from 64 to 80 bytes and improved query, narrow, and attention-output representative points, while shared down regressed:

| Shape | Unpadded | Padded | Decision |
| --- | ---: | ---: | --- |
| Query | 54.895 ms | 46.295 ms | retain typed padding |
| Narrow | 3.273 ms | 2.907 ms | retain typed padding |
| Attention output | 26.875 ms | 23.177 ms | retain typed padding |
| Shared down | 5.261 ms | 5.389 ms | reject padding for this body |

Explicit aligned fragment loads helped narrow Q3_K and attention-output Q4_K but regressed query Q4_K, Q5_K, and shared-down controls. Lower instruction count did not predict lower event time or LDS stalls.

### Pipeline and exact-shape work

The true two-buffer decoded-weight pipeline reduced aggregate wave cycles and wait/barrier stalls in the selected controls. It required a live-range repair after K64 reused addresses still needed by later WMMA pairs. The corrected handoff is exact at K32, K64, K96, and K512.

Exact-shape simplifications removed dead dimension loads, shortened address state, strength-reduced power-of-two strides, removed a fixed final `s_nop 7`, and normalized packed Q4 nibbles once per dword. These changes are resource-neutral or reducing and preserve the 40-byte ABI and output order.

## Resources

The three tuned variants use `222 VGPR / 20 SGPR / 10 KiB LDS` (`_k4096`), `233 VGPR / 16 SGPR / 8 KiB LDS` (`_k2048`) and `220 VGPR / 16 SGPR / 8 KiB LDS` (`_k512`), all static and spill-free.

## Evidence

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q4k_v1.txt
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q4k_v2.txt
~/tmp/torch-ggml-ops/hip_vs_baseline/pass11_ordbwd_qwen.json
~/tmp/torch-ggml-ops/hip_selection/           (per-key candidate campaign)
tools/configs/hip_deployment.json             (deployed body per key)
```

The original campaign evidence is:

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_narrow_q5_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_post_db8_control_9.json
~/tmp/torch-ggml-ops/mmq_bwd_final_full_v3.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb0_folded_25.json
```

Additional retained Q4 controls include the generic/exact DeepSeek build bracket and the Qwen baseline matrix:

```text
~/tmp/torch-ggml-ops/mmq_bwd_baseline_primary_sequential.json
~/tmp/torch-ggml-ops/mmq_bwd_final_full_v3.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_exact_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_generic_after_25.json
```

Global J64, I128, broad K64, split-K, GSU, Stream-K, persistent workgroups, compiler-managed prefetch arrays, decoded-weight LDS caching, and universal swizzle policies are closed for the current Q4_K representation.
