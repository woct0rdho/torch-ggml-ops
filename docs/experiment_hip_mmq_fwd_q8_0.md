# HIP MMQ Forward Q8_0 Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ forward for DeepSeek Q8_0 weights:

```text
output[M,N] = input[M,K] @ dequant_q8_0(weight[N,K]).T
```

Inputs and outputs are BF16. Activations use the Q8_1 F32_D4 workspace. Q8_0 payloads and FP16 scales are decoded cooperatively into the WMMA-facing LDS layout. The ordinary workload contains six geometry families; the language-model head is also retained as a separate Q8_0 chunk geometry.

## Final kernel result

`HIP TFLOPS` is `2*M*N*K/time`. `HIP/torch.mm` is the throughput ratio against BF16 `torch.mm`; values above `1.00x` favor HIP. The rows use the current packed-path medians.

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm |
| --- | ---: | ---: | ---: |
| Q-A | `(2048,1024,4096)` | 27.270 | 1.57x |
| Q-A | `(8192,1024,4096)` | 23.778 | 1.40x |
| Q-A | `(32768,1024,4096)` | 23.625 | 1.40x |
| Q-B | `(2048,32768,1024)` | 25.144 | 1.31x |
| Q-B | `(8192,32768,1024)` | 25.052 | 1.30x |
| Q-B | `(32768,32768,1024)` | 25.270 | 1.32x |
| KV | `(2048,512,4096)` | 25.117 | 2.01x |
| KV | `(8192,512,4096)` | 20.824 | 1.16x |
| KV | `(32768,512,4096)` | 20.328 | 1.13x |
| Output B | `(2048,4096,8192)` | 25.140 | 1.29x |
| Output B | `(8192,4096,8192)` | 24.347 | 1.21x |
| Output B | `(32768,4096,8192)` | 25.041 | 1.25x |
| Shared gate/up | `(2048,2048,4096)` | 26.574 | 1.46x |
| Shared gate/up | `(8192,2048,4096)` | 25.302 | 1.41x |
| Shared gate/up | `(32768,2048,4096)` | 23.998 | 1.30x |
| Shared down | `(2048,4096,2048)` | 27.248 | 1.41x |
| Shared down | `(8192,4096,2048)` | 26.194 | 1.39x |
| Shared down | `(32768,4096,2048)` | 24.536 | 1.30x |
| LM head | `(32,129280,4096)` | 11.895 | 1.88x |
| LM head | `(64,129280,4096)` | 23.000 | 3.07x |
| LM head | `(128,129280,4096)` | 25.900 | 2.58x |
| LM head | `(256,129280,4096)` | 25.706 | 2.25x |
| LM head | `(512,129280,4096)` | 25.526 | 1.31x |

The ordinary rows and isolated LM-head chunks use the current packed-path source tables. The effective rates include the matrix arithmetic represented by each shape; the activation producer remains a separate kernel.

## Kernel implementation

The generic Q8_0 body used runtime M/N/K state, 248 VGPRs, 29 SGPRs, and 38,400 bytes of dynamic LDS. Exact specialization folds K, full-tile bounds, address state, and LM chunk geometry. The retained ordinary body uses a four-wave 128x128/K32 layout with width-16 Q8_0 decode and row-dependent LDS padding. The LM head uses active-two-wave M32, compact M64/M128 bodies, and two M256-style workgroups for M512.

The decoder loads packed int8 payloads and scales, reconstructs signed values, stages them for WMMA, and preserves BF16 output conversion. Width32 decode, broad scalarization, and generic alternate row layouts were evaluated as kernel mechanisms rather than as host policy.

## Optimization log

### Baseline and exact shapes

The first DeepSeek baseline found ordinary Q8_0 at about `18.4-23.4` logical TFLOP/s. The generic LM-head M32/M64/M128 bodies took `5.395/5.630/5.859 ms`, exposing padded-row work. Quantization was only `0.3%` of LM M32 and `0.4%` of Q-B B1, but reached `26.3%` of KV B16 and `3.2%` of output-B B1.

The first exact Q8_0 bodies covered six ordinary geometries plus full and bounded LM rows. Exact specialization improved all 18 ordinary points by `9.18-21.75%`; LM M32/M64 improved by `47.40%/49.47%`, and M128/M256/M512 improved by `12.02-12.96%`. All retained bodies are zero-private and zero-spill.

The generic ordinary body was `248 VGPR / 29 SGPR / 38,400 B LDS`; exact J128 reduced it to `216 VGPR / 28 SGPR`, and exact/bounded J64 used `132 VGPR / 28 SGPR`. A first I64/J32/K4096 body was rejected before timing because it created 48 private bytes and 11 VGPR spills. The retained exact policy therefore uses J64 only where small rows would otherwise be padded.

### Geometry, traversal, and LDS

J64 lost on full ordinary tiles: K4096 families lost `4.03-8.85%`, Q-B lost `6.03-8.78%`, K8192 output-B lost `3.75-7.12%`, and K2048 shared down lost `4.94-5.58%`. J64 remains useful only where small M would otherwise carry padded work.

The retained ordinary geometry was G2 128x128/K32. G0 64x64/K16, G1 128x64/K32, and G3 256x64/K32 were slower across the production families. Q8_0 row padding reduced representative LDS conflicts and derived latency, but stride 77 lowered the conflict metric while increasing actual LDS stalls and regressing every point by `3.83-12.86%`.

The accepted stride-76 body measured `7.73%` LDS bank conflict, about `7.5%` ALU stalled by LDS, 128-cycle derived LDS latency, about 11.6 active waves per CU, and about 66% L2 hit rate. Stride 77 reduced conflict to `5.12%` but raised LDS-stalled ALU to about `14.3%`; its lower conflict score was not a performance win.

LM geometry retained active-two-wave M32, G0 M64, G1 M128, and G3 M256/M512. M32 limits cotangent loads, WMMA, and stores to two waves while retaining cooperative decode and barriers. Packed extraction and compact ownership improve the small chunks; M256/M512 remain constrained by packed traffic and accumulator occupancy.

### Traversal and closed controls

Changing only grouped-M traversal reduced Q-B B4/B16 by `38.77%/40.16%` and output-B B4/B16 by `42.03%/42.21%`. Further complete traversal correction reduced weighted ordinary packed latency by `4.51%/18.85%/26.41%` at B1/B4/B16. The normalized disassembly remained `98.16-98.83%` opcode-identical, showing that locality and launch mapping, not a new decoder, supplied the gain.

Activation-half double buffering was spill-free but lost `2-26%` because extra LDS and altered cadence outweighed fewer barriers. K-loop unrolling, stride-77 padding, broad width32 decode, decoded-weight LDS caching, split-K, GSU, Stream-K, persistent workgroups, and direct-to-LDS/direct-to-VGPR rewrites are closed for the current representation.

The remaining Q8_0 limit is representation cost: the packed kernel reconstructs int8 weights and scales while BF16 `torch.mm` starts from already decoded weights. A lossless payload/scale preparation layout is the next meaningful mechanism; a hidden BF16 shadow is not.

## Correctness and resources

Exact ordinary J128 bodies use `216 VGPR / 28 SGPR / 38,400 B LDS`; exact/bounded J64 bodies use `132 VGPR / 28 SGPR / 28,928 B LDS`. All retained Q8_0 bodies have zero private storage, zero spills, and no dynamic stack. Correctness covers 32-value payload blocks, signed values, scale conversion, exact K and M boundaries, LM small chunks, Q8_1 workspace mutation, input and packed-weight mutation, independent GGUF references, and finite output. One multi-counter Q6 run's HSA fault was an instrumentation issue; Q8 qualification was rerun sequentially.

## Evidence

```text
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_stride76_25.json
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p1_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p1_exact_25.json
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p1_generic_after_25.json
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_k4096_j64_25.json
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_other_j64_25.json
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_stride77_25.json
~/tmp/torch-ggml-ops/mmq_fwd_final_components.txt
```

Additional source-of-record artifacts for the initial DeepSeek geometry and standalone-bundle controls are:

```text
~/tmp/torch-ggml-ops/mmq_fwd_ds4_plan_baseline_9.json
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p1_exact_bracket_25.txt
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_k4096_j128_25.json
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_k4096_j64_control_25.txt
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_other_j128_25.json
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_other_j64_control_25.txt
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_prefetch_y_25.json
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_prefetch_baseline_25.json
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_prefetch_y_control_25.txt
~/tmp/torch-ggml-ops/mmq_fwd_ds4_p2_stride77_control_25.txt
~/tmp/torch-ggml-ops/mmq_fwd_qwen_plan_control_9.json
```

The kernel result is complete for the current packed Q8_0 arithmetic and traversal contract. Further gains require a new lossless representation or explicit activation reuse.
