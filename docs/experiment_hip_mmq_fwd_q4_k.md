# HIP MMQ Forward Q4_K Experiment

## Scope

This record covers the gfx1151 HIP packed-MMQ forward kernels for Q4_K weights.

| Family | Logical weight `(N,K)` | M values | Tensors |
| --- | ---: | ---: | ---: |
| Attention query/query gate | `(8192,2048)` | `2048,8192,32768` | 1 |
| Narrow K/V/shared gate/up | `(512,2048)` | `2048,8192,32768` | 70 |
| Attention output | `(2048,4096)` | `2048,8192,32768` | 10 |
| Shared-expert down | `(2048,512)` | `2048,8192,32768` | 30 |

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| ---: | --- | --- | --- | --- |
| Query/query gate | `(2048,8192,2048)` | 28.260 | 1.28x | `dense_fwd_q4_k_k2048_j128_full` |
| Query/query gate | `(8192,8192,2048)` | 28.095 | 1.25x | `dense_fwd_q4_k_k2048_j128_full` |
| Query/query gate | `(32768,8192,2048)` | 27.541 | 1.22x | `dense_fwd_q4_k_k2048_j128_full` |
| Narrow K/V/gate/up | `(2048,512,2048)` | 25.162 | 1.84x | `dense_fwd_q4_k_k2048_j128_full` |
| Narrow K/V/gate/up | `(8192,512,2048)` | 28.133 | 1.52x | `dense_fwd_q4_k_k2048_j128_full` |
| Narrow K/V/gate/up | `(32768,512,2048)` | 28.319 | 1.47x | `dense_fwd_q4_k_k2048_j128_full` |
| Attention output | `(2048,2048,4096)` | 28.667 | 1.46x | `dense_fwd_q4_k_k4096_j128_full` |
| Attention output | `(8192,2048,4096)` | 28.482 | 1.43x | `dense_fwd_q4_k_k4096_j128_full` |
| Attention output | `(32768,2048,4096)` | 28.506 | 1.42x | `dense_fwd_q4_k_k4096_j128_full` |
| Shared down | `(2048,2048,512)` | 25.024 | 8.38x | `dense_fwd_q4_k_k512_j128_full` |
| Shared down | `(8192,2048,512)` | 26.865 | 8.23x | `dense_fwd_q4_k_k512_j128_full` |
| Shared down | `(32768,2048,512)` | 27.817 | 8.39x | `dense_fwd_q4_k_k512_j128_full` |

## Kernel implementation

The retained Q4_K body uses four wave32 waves, `I=64`, `J=128`, and K256 packed reductions. Exact K512, K2048, and K4096 wrappers fold packed-row bytes, block counts, and full-tile bounds.

The Q8_1 F16_D4S4 producer emits four 32-value subblocks per 256-value Q4_K block.

### Minimum-term ablation

The dense forward body evaluates two epilogue terms per accumulator element and k block: `sum += dmA.x*dsB.x*C` for the quantized product, and `sum += dmA.y*dsB.y` for the block minimum, where `dmA.y` is the Q4_K block minimum and `dsB.y` the activation block sum. The second term does not depend on the matrix result, so the plan's affine-fold item proposes moving it into an extra contraction slot. Its size was measured first.

On `dense_fwd_q4_k_k2048_j128_full` (`(32768,8192,2048)`) two ablations were built from the same rendered source with the same flags and timed against the unmodified build under the official case protocol, in one harness (`~/tmp/torch-ggml-ops/r3_epilogue/`); both are wrong by construction and only their timing is meaningful.

| variant | time | TFLOPS | against unmodified |
| --- | ---: | ---: | ---: |
| unmodified | 38.343 ms | 28.68 | 1.000x |
| minimum term dropped | 35.693 ms | 30.80 | 1.074x |
| the whole epilogue replaced by an integer sink | 38.425 ms | 28.61 | 0.998x |

The minimum term is worth `7.4%` on its own, which makes it the largest clean ceiling any dense body in this campaign has produced, and it is a clean ceiling because dropping a term removes work rather than replacing it. Replacing the whole epilogue with an integer chain, by contrast, buys nothing here.

The measured ceiling does not translate directly into a mechanism. Folding the term into the contraction needs both of its factors quantized to int8, and then the term's scale is the product of the two quantization scales, which is per block in `i` and per block in `j`, so the epilogue would still apply a per-element, per-block factor and the fold would remove the term's arithmetic and add a scale application. Removing that factor requires coarsening the two quantization scales to a common per-stage value, which is the same structural requirement that closes the coarse-effective-scale item: the epilogue evaluates products of a row factor and a column factor, and only a group over which both are constant can be deferred. The `7.4%` is therefore an upper bound for a change that also carries the re-quantization of the minimum and block-sum values, and any implementation has to measure the equivalence error against the `5e-3` limit before it can claim any of it.

### Minimum-term precision measurement

The minimum term was worth `7.4%`, so the fold's other gate is precision. It was measured on the deployed case's real weights and activations (`(32768,8192,2048)`, the qwen attention projection) with `~/tmp/torch-ggml-ops/r3_epilogue/r4_precision.py`, which reads the packed Q4_K blocks and the Q8_1 activation metadata exactly as the kernel's tiles carry them (`d = amax/127`, the block sum `s = d*sum(q)`), forms the exact term `T = minimum_w @ s_act`, and re-forms it with both factors quantized to int8 under a scale constant over a group of blocks. The main term of the same outputs is `w_scaled @ x`, so the output-level error of the fold is measured directly rather than estimated.

| quantization group | term relative error | output NRMSE | gate |
| --- | ---: | ---: | --- |
| 32 values (one block, no coarsening) | 0.00000 | 0.00000 | passes, but saves nothing |
| 256 values (one k stage) | 0.0051 | 0.0114 | fails `5e-3` |
| 2048 values (whole row) | 0.0074 | 0.0166 | fails `5e-3` |

The flat case at 32 values is exact because one scale per block means one value per scale, which is exactly the granularity the deployed epilogue already applies; every group large enough to remove a per-element factor from the epilogue fails the gate.

The reason is visible in the same run: the minimum term's RMS is `2.245` times the output RMS. The term is not a small correction, it is the larger component of a cancelling pair, so a relative error on it is amplified by about `2.2` at the output: `0.5%` on the term becomes `1.1%` on the output, more than twice the `5e-3` limit. That amplification also explains why the item looked so attractive in the ablation: dropping the term removes a large, real contribution to every output element.

The item is therefore closed, and no tolerance changes. The `7.4%` ceiling stands as a measurement of what the term costs, not as a reachable gain: the only exact quantization granularity is the one the epilogue already uses, and coarsening it costs about twice the accuracy the equivalence gate allows.

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

### Hoisted epilogue metadata

The epilogue metadata of this quant was hoisted out of the column loop in the same way the Q6_K body deploys it. Over the full ordinary-shape A/B with the official protocol the change is neutral here (per-shape ratios inside `0.99x` to `1.01x`, no consistent direction), so the shipped body keeps the vendored target. The result is independent of tolerance because the body only moves loads: the arithmetic and the accumulation order are unchanged. Evidence: `~/tmp/torch-ggml-ops/retune_fwd/official/`.

## Resources

Retained Q4_K J128 bodies use `239 VGPR / 28-29 SGPR / 38,400 B LDS`.

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
