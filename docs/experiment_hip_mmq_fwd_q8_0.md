# HIP MMQ Forward Q8_0 Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ forward for DeepSeek Q8_0 weights.

The ordinary workload contains six geometry families; the language-model head is also retained as a separate Q8_0 chunk geometry.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| ---: | --- | --- | --- | --- |
| Q-A | `(2048,1024,4096)` | 31.043 | 1.81x | `dense_fwd_q8_0_k4096_j128_full` |
| Q-A | `(8192,1024,4096)` | 29.660 | 1.67x | `dense_fwd_q8_0_k4096_j128_full` |
| Q-A | `(32768,1024,4096)` | 29.838 | 1.65x | `dense_fwd_q8_0_k4096_j128_full` |
| Q-B | `(2048,32768,1024)` | 27.036 | 1.34x | `dense_fwd_q8_0_k1024_j128_full` |
| Q-B | `(8192,32768,1024)` | 27.111 | 1.33x | `dense_fwd_q8_0_k1024_j128_full` |
| Q-B | `(32768,32768,1024)` | 26.621 | 1.31x | `dense_fwd_q8_0_k1024_j128_full` |
| KV | `(2048,512,4096)` | 28.632 | 2.04x | `dense_fwd_q8_0_k4096_j128_full` |
| KV | `(8192,512,4096)` | 30.346 | 1.64x | `dense_fwd_q8_0_k4096_j128_full` |
| KV | `(32768,512,4096)` | 29.473 | 1.51x | `dense_fwd_q8_0_k4096_j128_full` |
| Output B | `(2048,4096,8192)` | 27.641 | 1.33x | `dense_fwd_q8_0_k8192_j128_full` |
| Output B | `(8192,4096,8192)` | 28.379 | 1.32x | `dense_fwd_q8_0_k8192_j128_full` |
| Output B | `(32768,4096,8192)` | 27.759 | 1.30x | `dense_fwd_q8_0_k8192_j128_full` |
| Shared gate/up | `(2048,2048,4096)` | 30.313 | 1.61x | `dense_fwd_q8_0_k4096_j128_full` |
| Shared gate/up | `(8192,2048,4096)` | 29.710 | 1.56x | `dense_fwd_q8_0_k4096_j128_full` |
| Shared gate/up | `(32768,2048,4096)` | 28.683 | 1.46x | `dense_fwd_q8_0_k4096_j128_full` |
| Shared down | `(2048,4096,2048)` | 29.741 | 1.47x | `dense_fwd_q8_0_k2048_j128_full` |
| Shared down | `(8192,4096,2048)` | 29.241 | 1.47x | `dense_fwd_q8_0_k2048_j128_full` |
| Shared down | `(32768,4096,2048)` | 28.401 | 1.39x | `dense_fwd_q8_0_k2048_j128_full` |
| LM head | `(32,129280,4096)` | 12.178 | 1.92x | `dense_fwd_q8_0_k4096_j64_bounded` |
| LM head | `(64,129280,4096)` | 24.053 | 3.07x | `dense_fwd_q8_0_k4096_j64_full` |
| LM head | `(128,129280,4096)` | 28.418 | 2.68x | `dense_fwd_q8_0_k4096_j128_full` |
| LM head | `(256,129280,4096)` | 28.248 | 2.35x | `dense_fwd_q8_0_k4096_j128_full` |
| LM head | `(512,129280,4096)` | 27.757 | 1.34x | `dense_fwd_q8_0_k4096_j128_full` |

The ordinary rows run the exact `dense_fwd_q8_0_k<K>_j128_full` artifacts; the LM head runs the bounded J64 body at M32, the full J64 body at M64, and the J128 bodies at M128 and above. The measured values sit above the earlier complete-call records because the producer is no longer inside the window, not because the multiply body changed.

## Kernel implementation

The generic Q8_0 body used runtime M/N/K state, 248 VGPRs, 29 SGPRs, and 38,400 bytes of dynamic LDS. Exact specialization folds K, full-tile bounds, address state, and LM chunk geometry. The retained ordinary body uses a four-wave 128x128/K32 layout with width-16 Q8_0 decode and row-dependent LDS padding. The LM head uses active-two-wave M32, compact M64/M128 bodies, and two M256-style workgroups for M512.

Width32 decode, broad scalarization, and generic alternate row layouts were evaluated as kernel mechanisms rather than as host policy.

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

## Resources

Exact ordinary J128 bodies use `216 VGPR / 28 SGPR / 38,400 B LDS`; exact/bounded J64 bodies use `132 VGPR / 28 SGPR / 28,928 B LDS`.

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
