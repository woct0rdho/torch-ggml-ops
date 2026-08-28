# HIP MMQ Backward Q8_0 Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ input-gradient kernels for DeepSeek Q8_0 weights:

```text
grad_input[M,N] = grad_output[M,K] @ dequant_q8_0(weight[N,K])
```

BF16 cotangents and gradients use FP32 WMMA accumulation. The Q8_0 weight remains packed in forward layout; the kernel decodes signed int8 payloads and scales into the reduction LDS image.

The final ordinary matrix contains six families at `M=2048,8192,32768`. The separate LM-head chunk experiment uses `M=32,64,128,256,512`, but the current dense-backward source does not record a matching BF16 `torch.mm` timing for those five isolated chunk shapes. Those chunk timings are retained in the optimization log rather than presented with an invented baseline ratio.

## Final ordinary-kernel result

`HIP TFLOPS` is `2*M*N*K/time`. `HIP/torch.mm` is the packed-throughput ratio against BF16 `torch.mm`; values above `1.00x` favor HIP.

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm |
| --- | ---: | ---: | ---: |
| Q-A | `(2048,4096,1024)` | 20.774 | 0.887x |
| Q-A | `(8192,4096,1024)` | 26.461 | 1.168x |
| Q-A | `(32768,4096,1024)` | 20.703 | 0.908x |
| Q-B | `(2048,1024,32768)` | 19.390 | 0.889x |
| Q-B | `(8192,1024,32768)` | 17.150 | 0.787x |
| Q-B | `(32768,1024,32768)` | 19.083 | 0.851x |
| KV | `(2048,4096,512)` | 24.826 | 1.146x |
| KV | `(8192,4096,512)` | 26.010 | 1.133x |
| KV | `(32768,4096,512)` | 26.324 | 1.108x |
| Output B | `(2048,8192,4096)` | 25.358 | 1.066x |
| Output B | `(8192,8192,4096)` | 20.729 | 0.845x |
| Output B | `(32768,8192,4096)` | 20.544 | 0.833x |
| Shared gate/up | `(2048,4096,2048)` | 22.710 | 0.983x |
| Shared gate/up | `(8192,4096,2048)` | 24.308 | 1.034x |
| Shared gate/up | `(32768,4096,2048)` | 19.900 | 0.866x |
| Shared down | `(2048,2048,4096)` | 22.225 | 1.318x |
| Shared down | `(8192,2048,4096)` | 19.567 | 1.106x |
| Shared down | `(32768,2048,4096)` | 19.941 | 1.129x |

The ratios are direct HIP-versus-`torch.mm` kernel evidence.

The current source log also retains the isolated LM-head backward shapes. It records HIP timing and therefore HIP TFLOPS, but no matching BF16 `torch.mm` timing; the unavailable ratio is shown explicitly rather than borrowed from another measurement.

| `(M,N,K)` | HIP time (ms) | HIP TFLOPS | HIP/torch.mm |
| ---: | ---: | ---: | ---: |
| `(32,4096,129280)` | 8.297 | 4.085 | unavailable in source log |
| `(64,4096,129280)` | 7.459 | 9.087 | unavailable in source log |
| `(128,4096,129280)` | 7.569 | 17.910 | unavailable in source log |
| `(256,4096,129280)` | 10.386 | 26.104 | unavailable in source log |
| `(512,4096,129280)` | 23.206 | 23.366 | unavailable in source log |

## Kernel implementation

The initial generic body used 64x64/reduction-16 ownership, 92 VGPRs, 17 SGPRs, and 2 KiB LDS. The retained ordinary body uses exact Q8_0 shapes with a four-wave 128x128/K32 geometry, width-16 decode, and row-dependent LDS padding. M1/M2 traversal is measured per shape and row count.

The LM head uses active-two-wave M32, G0 M64, G1 M128, and G3 M256/M512 bodies. M512 launches two exact M256-style tiles. All retained bodies decode packed Q8_0 values directly and store BF16 gradients.

The ordinary G2 body is `192 VGPR / 14 SGPR / 8 KiB LDS`; the isolated LM bodies use 91-194 VGPR and 2-4 KiB LDS. The selected artifacts remain zero-private, zero-spill, and stack-free.

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

## Correctness and resources

Retained Q8_0 bodies have zero private storage, zero spills, no dynamic stack, no scratch, and no calls. Validation covers signed 32-value blocks, scales, exact K/M boundaries, LM small chunks, input-gradient and packed-weight mutation, independent GGUF references, finite output, autograd, and a 65-row launch boundary.

## Evidence

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
