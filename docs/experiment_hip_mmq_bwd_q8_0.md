# HIP MMQ Backward Q8_0 Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ input-gradient kernels for DeepSeek Q8_0 weights.

The final ordinary matrix contains six families at `M=2048,8192,32768`, and the separate LM-head chunk matrix uses `M=32,64,128,256,512`. Shapes are written `(M, in_features, out_features)`, matching the weight's `(N,K) = (out_features, in_features)`.

## Final ordinary-kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| Q-A | `(2048,4096,1024)` | 26.090 | 1.082x | `dense_bwd_q8_0_exact_n1024k4096_g2_group_m2_padding8` |
| Q-A | `(8192,4096,1024)` | 26.358 | 1.059x | `dense_bwd_q8_0_exact_n1024k4096_g2_group_m2_padding8` |
| Q-A | `(32768,4096,1024)` | 27.368 | 1.104x | `dense_bwd_q8_0_exact_n1024k4096_g2_group_m2_padding8` |
| Q-B | `(2048,1024,32768)` | 19.765 | 0.874x | `dense_bwd_q8_0_exact_n32768k1024_g2_group_m1_padding8` |
| Q-B | `(8192,1024,32768)` | 19.530 | 0.870x | `dense_bwd_q8_0_exact_n32768k1024_g2_group_m1_padding8` |
| Q-B | `(32768,1024,32768)` | 22.473 | 0.987x | `dense_bwd_q8_0_exact_n32768k1024_g2_group_m1_padding8` |
| KV | `(2048,4096,512)` | 24.655 | 1.110x | `dense_bwd_q8_0_exact_n512k4096_g2_group_m2_padding8` |
| KV | `(8192,4096,512)` | 25.659 | 1.036x | `dense_bwd_q8_0_exact_n512k4096_g2_group_m2_padding8` |
| KV | `(32768,4096,512)` | 27.608 | 1.131x | `dense_bwd_q8_0_exact_n512k4096_g2_group_m2_padding8` |
| Output B | `(2048,8192,4096)` | 26.284 | 1.067x | `dense_bwd_q8_0_exact_n4096k8192_g2_padding8` |
| Output B | `(8192,8192,4096)` | 21.445 | 0.834x | `dense_bwd_q8_0_exact_n4096k8192_g2_group_m2` |
| Output B | `(32768,8192,4096)` | 20.806 | 0.837x | `dense_bwd_q8_0_exact_n4096k8192_g2_group_m2` |
| Shared gate/up | `(2048,4096,2048)` | 26.452 | 1.112x | `dense_bwd_q8_0_exact_n2048k4096_g2_group_m2_padding8` |
| Shared gate/up | `(8192,4096,2048)` | 24.662 | 1.029x | `dense_bwd_q8_0_exact_n2048k4096_g2_group_m2_padding8` |
| Shared gate/up | `(32768,4096,2048)` | 24.688 | 1.024x | `dense_bwd_q8_0_exact_n2048k4096_g2_group_m2_padding8` |
| Shared down | `(2048,2048,4096)` | 21.604 | 1.262x | `dense_bwd_q8_0_exact_n4096k2048_g2_padding8` |
| Shared down | `(8192,2048,4096)` | 19.533 | 1.076x | `dense_bwd_q8_0_exact_n4096k2048_g2_group_m2` |
| Shared down | `(32768,2048,4096)` | 20.255 | 1.110x | `dense_bwd_q8_0_exact_n4096k2048_g2_group_m2` |

The `Kernel` column names the deployed body for each exact key; every one of them is the fastest built body whose output is bitwise equal to the reference body in the per-key candidate campaign.

| `(M,N,K)` | HIP time (ms) | HIP TFLOPS | HIP/torch.mm | Kernel |
| ---: | ---: | ---: | ---: | --- |
| `(32,4096,129280)` | 8.321 | 4.073 | 0.556x | `dense_bwd_q8_0_exact_lm_head_bounded` |
| `(64,4096,129280)` | 7.068 | 9.590 | 0.667x | `dense_bwd_q8_0_exact_lm_head_full` |
| `(128,4096,129280)` | 7.393 | 18.335 | 1.143x | `dense_bwd_q8_0_exact_lm_head_g1` |
| `(256,4096,129280)` | 10.020 | 27.059 | 1.750x | `dense_bwd_q8_0_exact_lm_head_g3` |
| `(512,4096,129280)` | 21.338 | 25.412 | 1.536x | `dense_bwd_q8_0_exact_lm_head_g3` |

## Kernel implementation

The initial generic body used 64x64/reduction-16 ownership, 92 VGPRs, 17 SGPRs, and 2 KiB LDS. The ordinary bodies use exact Q8_0 shapes with a four-wave 128x128/K32 geometry, width-16 decode, and row-dependent LDS padding. M1/M2 traversal is measured per shape and row count.

The build carries four ordinary variant generations: the plain `exact_n{N}k{K}` wrappers, the `_g1`/`_g2`/`_g3` geometry screen winners, and the `_g2_padding8`/`_g2_group_m{1,2}`/`_g2_group_m2_padding8` traversal variants. The deployed ordinary bodies are all G2-family (`n_tiles=8`, `k_iteration=32`, `decoder_width=16`, `active_waves=4`) with `lds_padding=8` and/or `group_m` set per key; `K` in the variant name is the reduction length, and no single variant wins every shape. The plain `exact_n{N}k{K}` wrappers are the pre-G generation that the `_g*` screen superseded and are not part of the deployment campaign.

The LM head uses active-two-wave M32, G0 M64, G1 M128, and G3 M256/M512 bodies. M512 launches two exact M256-style tiles. The campaign timed the built chunk bodies per `M` and deploys the fastest valid one: `_bounded` at M32, `_full` at M64, `_g1` at M128, and `_g3` at M256/M512. `_g2` lost at M128/M256/M512, `_full` lost at M128 and above, `_bounded` lost at M64 and above, `_m32_active2` tied with the deployed `_bounded` body inside `0.2%` at M32, and `_g3` faults at M128, which `_g1` covers.

The ordinary G2 body is `192 VGPR / 14 SGPR / 8 KiB LDS`; the isolated LM bodies use 91-194 VGPR and 2-4 KiB LDS.

## Optimization log

### Baseline and exact shape specialization

The generic baseline was resource-light but ownership-limited:

| Batch | Historical packed/BF16 weighted ms | Throughput ratio |
| ---: | ---: | ---: |
| 1 | `3266.8/767.4` | `0.235x` |
| 4 | `16920.7/3075.7` | `0.182x` |
| 16 | `72146.2/12140.1` | `0.168x` |

The first exact Q8_0 wrappers removed runtime shape, bounds, and address state. Exact specialization improved all 18 ordinary points by `11.32-246.14%`, with a `60.58%` geometric gain. The full ordinary wrappers remained resource-clean.

The generic body used `92 VGPR / 17 SGPR / 2 KiB LDS`; the geometry screen compared G0 `64x64/K16` at 92 VGPR, G1 `128x64/K32` at 118, G2 `128x128/K32` at 192, and G3 `256x64/K32` at 194. G2 won all six ordinary shape families. The first exact-shape bracket improved the full ordinary matrix by `77.49%` geometrically and by `86.99%/96.51%/99.95%` at B1/B4/B16 weighted latency.

### Ordinary geometry and padding

G2 128x128/K32 beat G0 64x64/K16, G1 128x64/K32, and G3 256x64/K32 across the six ordinary families. Unpadded G2 reported a repeated `79.17%` LDS-conflict metric. Padding8 lowered Q-A B1 conflict from `79.17%` to `58.33%`, derived LDS latency from about 585 to 245 cycles, and ALU stall from LDS from `24.19%` to `15.26%`.

Padding was retained only where exact row cost justified it. Q-B and output-B required separate traversal and padding decisions; global J64 lost full ordinary tiles by `3.75-8.85%` depending on K and family.

### LM chunk geometries

The isolated backward medians for Q8_0 LM chunks were `8.297/7.459/7.569/10.386/23.206 ms` at M32/M64/M128/M256/M512. M32 keeps all waves for decode and barriers but limits cotangent loads, WMMA, and stores to two waves. M128 improved `19.48%`; M256/M512 improved `51.85%/60.64%` over the preceding controls.

### Traversal and closed controls

Changing only grouped-M traversal reduced Q-B B4/B16 by `38.77%/40.16%` and output-B B4/B16 by `42.03%/42.21%`. The later complete traversal correction reduced weighted packed latency by `4.51%/18.85%/26.41%` at B1/B4/B16. Normalized disassembly was `98.16-98.83%` opcode-identical, so locality and mapping, not a new decoder, supplied the improvement.

The DB8 screen retained M2 for Q-A, KV, shared gate/up, and shared down at the longer row counts, and M1 only for Q-B B1. Representative all-M-to-M2 times were Q-A B16 `32.386 -> 13.184 ms`, KV B16 `23.194 -> 5.253 ms`, shared gate B16 `54.401 -> 27.955 ms`, and shared down B16 `49.452 -> 27.961 ms`; Q-B B1 used the `7.241/6.923/7.509 ms` M2/M1/M2 bracket. L2 hit rate rose by `30.6-47.4` percentage points, while occupancy changed only modestly. `MemUnitBusy` was unavailable because rocprofv3 rejected its non-windowable `TA_TA_BUSY` dependency.

Activation-half double buffering, K-loop unrolling, width32 decode, stride77 padding, decoded-weight LDS caching, split-K, GSU, Stream-K, persistent workgroups, and direct-to-LDS/direct-to-VGPR rewrites are closed for the current packed representation. The remaining ordinary deficit is repeated packed decode versus a BF16 baseline that starts from decoded weights.

## Evidence

Current measurement evidence for both tables:

```text
~/tmp/torch-ggml-ops/hip_vs_baseline/pass11_ordbwd_deepseek.json
~/tmp/torch-ggml-ops/hip_selection/           (per-key candidate campaign)
tools/configs/hip_deployment.json             (deployed body per key)
```

The original campaign evidence is:

```text
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db8_final_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p0_baseline_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_exact_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_generic_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p2_corrected_selected_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p4_control_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p4_selected_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_final_25.json
```

The source log's retained DB7/DB8 controls include the shape-specific traversal brackets:

```text
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db6_final_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_qb_control_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_qb_m2_candidate_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_qb_control_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_output_control_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_output_m2_candidate_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_output_control_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_m2_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_m1_candidate_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_m2_after_25.json
```

The current source closes the broad dense Q8_0 geometry and traversal neighborhood. A lossless prepared payload/scale representation is the next kernel premise worth testing.
