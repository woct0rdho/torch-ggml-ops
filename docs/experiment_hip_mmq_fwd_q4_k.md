# HIP MMQ Forward Q4_K Experiment

## Scope

This record covers the gfx1151 HIP packed-MMQ forward kernels for Q4_K weights:

```text
output[M,N] = input[M,K] @ dequant_q4_k(weight[N,K]).T
```

Inputs and outputs are BF16. The packed Q4_K weights are decoded in the kernel. The fixed Q8_1 F16_D4S4 activation workspace supplies the signed-int8 values and scale/sum metadata required by Q4_K correction.

| Family | Logical weight `(N,K)` | M values | Tensors |
| --- | ---: | ---: | ---: |
| Attention query/query gate | `(8192,2048)` | `2048,8192,32768` | 1 |
| Narrow K/V/shared gate/up | `(512,2048)` | `2048,8192,32768` | 70 |
| Attention output | `(2048,4096)` | `2048,8192,32768` | 10 |
| Shared-expert down | `(2048,512)` | `2048,8192,32768` | 30 |

`HIP TFLOPS` is `2*M*N*K/time` for the packed path. `HIP/torch.mm` is the throughput ratio against BF16 `torch.mm`; values above `1.00x` favor HIP.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm |
| --- | ---: | ---: | ---: |
| Query/query gate | `(2048,8192,2048)` | 26.522 | 1.26x |
| Query/query gate | `(8192,8192,2048)` | 24.885 | 1.20x |
| Query/query gate | `(32768,8192,2048)` | 25.110 | 1.18x |
| Narrow K/V/gate/up | `(2048,512,2048)` | 21.913 | 1.61x |
| Narrow K/V/gate/up | `(8192,512,2048)` | 20.875 | 1.16x |
| Narrow K/V/gate/up | `(32768,512,2048)` | 20.017 | 1.07x |
| Attention output | `(2048,2048,4096)` | 25.508 | 1.38x |
| Attention output | `(8192,2048,4096)` | 24.188 | 1.30x |
| Attention output | `(32768,2048,4096)` | 23.402 | 1.24x |
| Shared down | `(2048,2048,512)` | 24.403 | 8.20x |
| Shared down | `(8192,2048,512)` | 24.163 | 7.42x |
| Shared down | `(32768,2048,512)` | 23.060 | 6.99x |

The table uses the current complete packed-path timings, including the required Q8_1 producer work recorded in the source matrix.

## Kernel implementation

The retained Q4_K body uses four wave32 waves, `I=64`, `J=128`, and K256 packed reductions. Exact K512, K2048, and K4096 wrappers fold packed-row bytes, block counts, and full-tile bounds. The Q4_K decoder reconstructs packed nibbles, scale/minimum metadata, and BF16-compatible LDS rows before integer WMMA and FP32 correction.

The Q8_1 F16_D4S4 producer emits four 32-value subblocks per 256-value Q4_K block. Its workspace layout and rounding are fixed for this experiment. Q4_K is the scale-plus-sum consumer; it does not use the scale-only Q3_K/Q6_K workspace contract.

## Optimization log

### Producer and geometry

The first activation-quantizer campaign changed the launch from one 64-thread workgroup per padded row/block to one 512-thread workgroup per real row. At narrow M32768, quantizer time fell from `5,105.979 us` to `886.928 us`, while multiplication remained about `2,565.848 us`. This separated activation scheduling from the Q4_K multiplication body.

The first tiled Q4_K body established cooperative decode, LDS-staged weights, four-wave WMMA ownership, and exact full-tile arithmetic. The common geometry remained 128x128/K32. I128, global J64, smaller workgroups, and broad 2xM ownership did not improve the measured production mix.

The representative Qwen Q4_K redesign moved narrow M32768 from `48.332 ms` to `11.130 ms` and attention output M32768 from `487.945 ms` to `80.633 ms`. Compile-time shape specialization and full/tail separation later moved representative Q4 points from `6.874` to `2.842 ms`, then from `2.842` to `1.599 ms`; these are historical kernel milestones, not current final-matrix timings.

### Packed extraction, padding, and local reads

Q4_K pre-layout controls reported a repeated `79.2%` LDS-bank-conflict metric. Eight BF16 values of row padding changed the K32 LDS stride from 64 to 80 bytes and improved three of four M32768 controls:

| Shape | Unpadded | Padded | Decision |
| --- | ---: | ---: | --- |
| Query | 54.895 ms | 46.295 ms | retain typed padding |
| Narrow | 3.273 ms | 2.907 ms | retain typed padding |
| Attention output | 26.875 ms | 23.177 ms | retain typed padding |
| Shared down | 5.261 ms | 5.389 ms | reject padding for this body |

Q4_K retains explicit packed-byte state and bounded current-phase prefetch. Keeping the next reduction iteration's packed fragments live across WMMA increased register lifetime and was rejected. Complete metadata vector loads, generic aligned fragment loads, broad swizzles, and custom barriers were not reliable improvements; actual event timing, not instruction count or a conflict percentage alone, selected the layout.

The four-shape padding bracket was measured at M32768: query `54.895 -> 46.295 ms`, narrow `3.273 -> 2.907 ms`, attention output `26.875 -> 23.177 ms`, and shared down `5.261 -> 5.389 ms`. Padding is therefore retained for the first three Q4 bodies and rejected for shared down. K64 was neutral for Q4 shared down; disabling packed-byte prefetch lowered allocation to 234 VGPRs but regressed that body to `6.186 ms`.

### Exact Qwen bodies

The exact Q4_K wrappers cover K512, K2048, and K4096. Exact specialization improved its generic controls by `3.60-31.39%` across the measured matrix. Q4_K query/narrow and attention-output bodies use shape-specific local-load and LDS choices; shared-down has a distinct 16-BF16 layout because its K512 cost model differs.

The Q4_K path remains representation- and decode-bound at the lower margin. A transient BF16 materialization floor was slower even when real decode work was excluded, so another global tile or buffer sweep is not justified without a changed representation premise.

## Correctness and resources

Retained Q4_K J128 bodies use `239 VGPR / 28-29 SGPR / 38,400 B LDS`; they have zero private storage, zero spills, and no dynamic stack. Validation covers Q4_K nibble fields, scale/minimum reconstruction, all four Q8_1 metadata pairs, block and tile boundaries, workspace mutation, packed-weight mutation, input mutation, independent GGUF dequantization, and finite outputs. Generic bounds-safe wrappers remain the fallback for unsupported shapes.

## Evidence

```text
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_after_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_bracket_25.txt
~/tmp/torch-ggml-ops/mmq_fwd_final_components.txt
```

The shared dense-forward controls used to qualify the exact Q4 body are also retained here:

```text
~/tmp/torch-ggml-ops/mmq_fwd_baseline_primary_sequential.json
~/tmp/torch-ggml-ops/mmq_fwd_final_full.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_plan_control_9.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_after_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_bracket_25.txt
```

Closed Q4_K directions include global J64/I128 rules, activation-half double buffering, decoded-weight LDS caching, speculative prefetch, split-K, persistent workgroups, direct-to-LDS forms unavailable on gfx1151, and global padding/swizzle policies. Any reopening must name a new decode or residency mechanism.
