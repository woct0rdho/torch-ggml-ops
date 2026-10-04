# HIP MMQ Forward Q4_K Experiment

## Scope

This record covers the gfx1151 HIP packed-MMQ forward kernels for Q4_K weights.

Q4_K carries ordinary projections of the Qwen3.6-35B-A3B (APEX-I-Mini) and Qwen3.8-Flash-Next (GSQ-RCO-Q2_0) checkpoints, so alongside the families measured first the record covers the QSA attention key/value and output, shared-expert gate/up, and GatedDeltaNet `in_proj_qkv`/`in_proj_z` shapes those checkpoints add at hidden size 2560, at the training token counts of a sequence length 2048 batch (B1/B4/B16).

GatedDeltaNet `out_proj` is deferred because wiring it needs the activation permutation, and the size-32 `ssm_alpha` projections are below the kernel's 64-row tile.

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
| QSA key/value | `(2048,512,2560)` | 25.555 | 1.84x | `dense_fwd_q4_k_k2560_j128_full` |
| QSA key/value | `(8192,512,2560)` | 28.365 | 1.50x | `dense_fwd_q4_k_k2560_j128_full` |
| QSA key/value | `(32768,512,2560)` | 28.595 | 1.40x | `dense_fwd_q4_k_k2560_j128_full` |
| QSA output | `(2048,2560,6144)` | 28.252 | 1.20x | `dense_fwd_q4_k_k6144_j128_full` |
| QSA output | `(8192,2560,6144)` | 28.927 | 1.44x | `dense_fwd_q4_k_k6144_j128_full` |
| QSA output | `(32768,2560,6144)` | 27.566 | 1.40x | `dense_fwd_q4_k_k6144_j128_full` |
| Shared-expert gate/up | `(2048,640,2560)` | 24.637 | 1.73x | `dense_fwd_q4_k_k2560_j128_full` |
| Shared-expert gate/up | `(8192,640,2560)` | 27.983 | 1.89x | `dense_fwd_q4_k_k2560_j128_full` |
| Shared-expert gate/up | `(32768,640,2560)` | 28.395 | 1.89x | `dense_fwd_q4_k_k2560_j128_full` |
| GatedDeltaNet QKV | `(2048,10240,2560)` | 28.315 | 1.20x | `dense_fwd_q4_k_k2560_j128_full` |
| GatedDeltaNet QKV | `(8192,10240,2560)` | 28.559 | 1.20x | `dense_fwd_q4_k_k2560_j128_full` |
| GatedDeltaNet QKV | `(32768,10240,2560)` | 27.369 | 1.19x | `dense_fwd_q4_k_k2560_j128_full` |
| GatedDeltaNet Z | `(2048,6144,2560)` | 28.263 | 1.21x | `dense_fwd_q4_k_k2560_j128_full` |
| GatedDeltaNet Z | `(8192,6144,2560)` | 28.315 | 1.19x | `dense_fwd_q4_k_k2560_j128_full` |
| GatedDeltaNet Z | `(32768,6144,2560)` | 27.312 | 1.18x | `dense_fwd_q4_k_k2560_j128_full` |
| GatedDeltaNet Z APEX-I-Mini | `(2048,4096,2048)` | 28.010 | 1.35x | `dense_fwd_q4_k_k2048_j128_full` |
| GatedDeltaNet Z APEX-I-Mini | `(8192,4096,2048)` | 28.190 | 1.38x | `dense_fwd_q4_k_k2048_j128_full` |
| GatedDeltaNet Z APEX-I-Mini | `(32768,4096,2048)` | 27.265 | 1.38x | `dense_fwd_q4_k_k2048_j128_full` |

## Kernel implementation

The retained Q4_K body uses four wave32 waves, `I=64`, `J=128`, and K256 packed reductions. Exact K512, K2048, K2560, K4096, and K6144 wrappers fold packed-row bytes, block counts, and full-tile bounds.

The Q8_1 F16_D4S4 producer emits four 32-value subblocks per 256-value Q4_K block.

### Minimum-term ablation

The dense forward body evaluates two epilogue terms per accumulator element and k block: `sum += dmA.x*dsB.x*C` for the quantized product, and `sum += dmA.y*dsB.y` for the block minimum, where `dmA.y` is the Q4_K block minimum and `dsB.y` the activation block sum. The second term does not depend on the matrix result, so the plan's affine-fold item proposes moving it into an extra contraction slot. Its size was measured first.

On `dense_fwd_q4_k_k2048_j128_full` (`(32768,8192,2048)`) two ablations were built from the same rendered source with the same flags and timed against the unmodified build under the official case protocol, in one harness (`~/tmp/torch-ggml-ops/r3_epilogue/`). Both are wrong by construction and only their timing is meaningful.

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

The flat case at 32 values is exact because one scale per block means one value per scale, which is exactly the granularity the deployed epilogue already applies. Every group large enough to remove a per-element factor from the epilogue fails the gate.

The reason is visible in the same run: the minimum term's RMS is `2.245` times the output RMS. The term is not a small correction, it is the larger component of a cancelling pair, so a relative error on it is amplified by about `2.2` at the output: `0.5%` on the term becomes `1.1%` on the output, more than twice the `5e-3` limit. That amplification also explains why the item looked so attractive in the ablation: dropping the term removes a large, real contribution to every output element.

The item is therefore closed, and no tolerance changes. The `7.4%` ceiling stands as a measurement of what the term costs, not as a reachable gain: the only exact quantization granularity is the one the epilogue already uses, and coarsening it costs about twice the accuracy the equivalence gate allows.

## Optimization log

### Producer and geometry

The first activation-quantizer campaign changed the launch from one 64-thread workgroup per padded row/block to one 512-thread workgroup per real row. At narrow M32768, quantizer time fell from `5,105.979 us` to `886.928 us`, while multiplication remained about `2,565.848 us`. This separated activation scheduling from the Q4_K multiplication body.

The first tiled Q4_K body established cooperative decode, LDS-staged weights, four-wave WMMA ownership, and exact full-tile arithmetic. The common geometry remained 128x128/K32. I128, global J64, smaller workgroups, and broad 2xM ownership did not improve the measured production mix.

The representative Qwen Q4_K redesign moved narrow M32768 from `48.332 ms` to `11.130 ms` and attention output M32768 from `487.945 ms` to `80.633 ms`. Compile-time shape specialization and full/tail separation later moved representative Q4 points from `6.874` to `2.842 ms`, then from `2.842` to `1.599 ms`. These are historical kernel milestones, not current final-matrix timings.

### Packed extraction, padding, and local reads

Q4_K pre-layout controls reported a repeated `79.2%` LDS-bank-conflict metric. Eight BF16 values of row padding changed the K32 LDS stride from 64 to 80 bytes and improved three of four M32768 controls:

| Shape | Unpadded | Padded | Decision |
| --- | ---: | ---: | --- |
| Query | 54.895 ms | 46.295 ms | retain typed padding |
| Narrow | 3.273 ms | 2.907 ms | retain typed padding |
| Attention output | 26.875 ms | 23.177 ms | retain typed padding |
| Shared down | 5.261 ms | 5.389 ms | reject padding for this body |

Q4_K retains explicit packed-byte state and bounded current-phase prefetch. Keeping the next reduction iteration's packed fragments live across WMMA increased register lifetime and was rejected. Complete metadata vector loads, generic aligned fragment loads, broad swizzles, and custom barriers were not reliable improvements. Actual event timing, not instruction count or a conflict percentage alone, selected the layout.

The four-shape padding bracket was measured at M32768: query `54.895 -> 46.295 ms`, narrow `3.273 -> 2.907 ms`, attention output `26.875 -> 23.177 ms`, and shared down `5.261 -> 5.389 ms`. Padding is therefore retained for the first three Q4 bodies and rejected for shared down. K64 was neutral for Q4 shared down. Disabling packed-byte prefetch lowered allocation to 234 VGPRs but regressed that body to `6.186 ms`.

### Exact Qwen bodies

The exact Q4_K wrappers cover K512, K2048, and K4096. Exact specialization improved its generic controls by `3.60-31.39%` across the measured matrix. Q4_K query/narrow and attention-output bodies use shape-specific local-load and LDS choices. Shared-down has a distinct 16-BF16 layout because its K512 cost model differs.

The Q4_K path remains representation- and decode-bound at the lower margin. A transient BF16 materialization floor was slower even when real decode work was excluded, so another global tile or buffer sweep is not justified without a changed representation premise.

### Hoisted epilogue metadata

The epilogue metadata of this quant was hoisted out of the column loop in the same way the Q6_K body deploys it. Over the full ordinary-shape A/B with the official protocol the change is neutral here (per-shape ratios inside `0.99x` to `1.01x`, no consistent direction), so the shipped body keeps the vendored target. The result is independent of tolerance because the body only moves loads: the arithmetic and the accumulation order are unchanged. Evidence: `~/tmp/torch-ggml-ops/retune_fwd/official/`.

### K2560 and K6144 exact specialization and the wide tile

The Qwen4-Exp and GatedDeltaNet shapes added two contraction lengths, `K=2560` and `K=6144` (ten and twenty-four `K=256` blocks per packed row). Exact specialization over the generic control is worth `3.6-11.9%` on the new shapes, in the same range as the K2048, K512 and K4096 wrappers: `1.086-1.119x` on the narrow `(512,2560)`, `1.046-1.083x` on the two Qwen4 GatedDeltaNet families, `1.038-1.046x` on the QSA output, and `1.079-1.095x` on the APEX-I-Mini `(4096,2048)`. The APEX-I-Mini `in_proj_qkv` `(8192,2048)` reuses the already-tabulated query shape and reproduces its rows.

The wide `I=128` dense tile that was built for the Q3_K shapes was rendered for this type as well and measured on the same points. It does not pay here either: `1.214x` slower on `(512,2560)` at `M=2048` and `1.066-1.068x` slower at `M=32768` on the three shapes with the largest activation re-read (`(2560,6144)`, `(10240,2560)`, `(6144,2560)`), so it is not deployed.

Against the ceiling the Q3_K record measures on the same instruction (`48.7 TFLOPS` for `v_wmma_i32_16x16x16_iu8` with this accumulator count, from `~/tmp/torch-ggml-ops/qwen4_fwd/wmma_i8_probe.cu`), the retained Q4_K bodies run at `27-29 TFLOPS`, i.e. `55-60%`, against `45-51%` for the Q3_K bodies. Evidence: `~/tmp/torch-ggml-ops/qwen4_fwd/sweep_q4k_v1.txt`.

## Resources

Retained Q4_K J128 bodies use `223 VGPR / 28-29 SGPR / 38,400 B LDS`.

## Evidence

```text
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_generic_after_25.json
~/tmp/torch-ggml-ops/mmq_fwd_qwen_p4_exact_bracket_25.txt
~/tmp/torch-ggml-ops/mmq_fwd_final_components.txt
```

The Qwen4-Exp and GatedDeltaNet points come from outside the public case list, because those shapes have HIP controls but no GGTensile problem key yet, so they cannot be selected as deployment cases. Their runs use the deployed Q8_1 producer and the same prepared inputs, the same `torch.mm` BF16 baseline on the same activation tensor, and the same paired timing as the official protocol:

```text
~/tmp/torch-ggml-ops/qwen4_fwd/sweep_q4k_v1.txt
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

Closed Q4_K directions include global J64/I128 rules, activation-half double buffering, decoded-weight LDS caching, speculative prefetch, split-K, persistent workgroups, direct-to-LDS forms unavailable on gfx1151, global padding/swizzle policies, and the wide `I=128` dense tile measured above. Any reopening must name a new decode or residency mechanism.
