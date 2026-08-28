# HIP Grouped MMQ Forward Q2_K Experiment

## Scope

This record covers the routed Q2_K down kernel for DeepSeek:

```text
Y[R,4096] = X[R,2048] @ W[4096,2048].T
```

The routed row counts are `R=12288,49152,196608` for physical batches 1, 4, and 16. Packed Q2_K weights are decoded cooperatively into the WMMA-facing LDS layout.

## Final kernel result

`HIP TFLOPS` is `2*R*N*K/time`. `HIP/AITER GMM` is the packed HIP throughput ratio against BF16 AITER GMM. Values above `1.00x` favor HIP.

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM |
| ---: | ---: | ---: | ---: |
| 1 | `(12288,4096,2048)` | 11.65 | 1.464x |
| 4 | `(49152,4096,2048)` | 12.75 | 0.811x |
| 16 | `(196608,4096,2048)` | 12.62 | 0.540x |

The B4/B16 gap is dominated by repeated Q2_K scale/minimum reconstruction and limited N64 reuse relative to predecoded BF16 AITER weights.

## Kernel implementation

The retained body uses four wave32 waves, exact N4096/K2048 geometry, width-16 Q2_K decode, shared scale/minimum and packed-shift work, serial J32 ownership, and a bounded J16 tail body where measured. Inactive-M consumer suppression skips only inactive cotangent/WMMA/store work while keeping decode and barriers uniform.

Q2_K has its own packed scale/minimum reconstruction and does not inherit Q4_K/Q5_K layouts. The kernel handles routed tails, inactive experts, and device-resident route metadata without host offset reads.

## Optimization log

### Exact geometry and decode

The generic grouped baseline used narrow N16/K16 ownership. Exact Q2_K M64/M128 and J32 geometry increased N reuse and removed much of the serial overhead. Width-16 decode shares each scale/min group and packed shift across natural value groups.

Reduction unroll U2 improved all B4 routes but regressed B1 and one B16 boundary point, so it remained a bounded kernel variant. U4 lost to U2 by `1.2-4.3%`. A valid N128/U1 control compiled at 255 VGPRs but lost every B16 route by `1.18-4.18%`; N128 was rejected.

The retained Q2_K bodies use `208 VGPR / 38 SGPR / 4096 B LDS` for the regular J32 body and `213 VGPR / 43 SGPR / 4096 B LDS` for the J32/J16 mixed body. Both remain zero-private and spill-free. The exact B4 mixed control reached `1.0227x` in search and `1.0247x` in disjoint confirmation, with minimum learned/hash gains `1.0169x/1.0173x` and minimum controls `0.9998x/0.9996x`.

### B4 mixed tail

The existing J32/J16 body was measured at exact B4 aggregate rows `R=49152`. The longer confirmation improved the selected body by `1.0247x`; minimum learned/hash prior gains were `1.0169x/1.0173x`, and minimum mandatory controls were `0.9998x/0.9996x`. Outputs remained bitwise identical.

The coefficient-only campaign found no benefit from row tasks: Q2_K already exposes 32 N workgroups per expert and thousands of total workgroups, while row tasks would not remove rounded tail arithmetic.

### Bottleneck attribution

The selected B16 counter review found:

| Metric | HIP/AITER result |
| --- | ---: |
| Total instructions | `10.88x` |
| VALU instructions | `24.27x` |
| VALU issue cycles | `22.39x` |
| Instruction-fetch waits | `10.36x` |
| Mean occupancy per active CU | `6.25 / 10.67` waves |
| Video-memory fetch | `0.585x` AITER bytes |
| ALU stalled by LDS | `0.088% / 33.78%` |

HIP fetches less data but executes a much larger decode/instruction stream with lower residency. The limit is not total DRAM traffic or an LDS-bank bottleneck.

## Correctness and resources

Retained Q2_K bodies have zero private storage, zero spills, no dynamic stack, scratch, or calls. Validation covers 2-bit values, scale/minimum groups, inactive M tiles, malformed routes, non-aligned tails, input/weight mutation, independent BF16 references, finite output, and deterministic rebuilds.

## Evidence

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_deepseek_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/deepseek_q2_down_mixed_b4_search_25.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/deepseek_q2_down_mixed_b4_confirmation_25.json
~/tmp/torch-ggml-ops/profile-grouped-fwd-20260812/deepseek_q2k_down_b16_packed_v2/trace_results.db
~/tmp/torch-ggml-ops/profile-grouped-fwd-20260812/deepseek_q2k_down_b16_aiter/trace_results.db
```

The remaining Q2_K loss is packed decode and instruction pressure. Another broad N/tail sweep is closed.

The counter-qualified profile behind this attribution is:

```text
~/tmp/torch-ggml-ops/profile-grouped-fwd-20260812/deepseek_q2_b16_profile_summary.md
~/tmp/torch-ggml-ops/profile-grouped-fwd-20260812/deepseek_q2k_down_b16_packed_v2/trace_results.db
~/tmp/torch-ggml-ops/profile-grouped-fwd-20260812/deepseek_q2k_down_b16_aiter/trace_results.db
```
