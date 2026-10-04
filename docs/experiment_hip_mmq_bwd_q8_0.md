# HIP MMQ Backward Q8_0 Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ input-gradient kernels for DeepSeek Q8_0 weights.

The final ordinary matrix contains seven families at `M=2048,8192,32768`, and the separate LM-head chunk matrix uses `M=32,64,128,256,512`. Shapes are written `(M, in_features, out_features)`, matching the weight's `(N,K) = (out_features, in_features)`. One of the seven is the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` shared-expert down projection, whose weight is `(2560,640)`. Its M values count tokens of a sequence length 2048 batch (B1/B4/B16), and the training path calls `M=2048`. The 48 layers mix recipes - the same projection family is a different quant type in different layers - so every type that appears needs a body, or those layers fall back to a dequantizing multiply.

## Final ordinary-kernel result

| Family | `(M,K,N)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| Q-A | (2048,4096,1024) | 28.984 | 1.221x | `dense_bwd_q8_0_exact_n1024k4096_g2_group_m2_padding8_pipe` |
| Q-A | `(8192,4096,1024)` | 26.482 | 1.064x | `dense_bwd_q8_0_exact_n1024k4096_g2_group_m2_padding8` |
| Q-A | (32768,4096,1024) | 27.415 | 1.106x | `dense_bwd_q8_0_exact_n1024k4096_g2_group_m2_padding8_pipe` |
| Q-B | `(2048,1024,32768)` | 19.033 | 0.859x | `dense_bwd_q8_0_exact_n32768k1024_g2_group_m1_padding8` |
| Q-B | `(8192,1024,32768)` | 18.476 | 0.823x | `dense_bwd_q8_0_exact_n32768k1024_g2_group_m1_padding8` |
| Q-B | `(32768,1024,32768)` | 22.170 | 1.023x | `dense_bwd_q8_0_exact_n32768k1024_g2_group_m1_padding8` |
| KV | `(2048,4096,512)` | 24.861 | 1.115x | `dense_bwd_q8_0_exact_n512k4096_g2_group_m2_padding8` |
| KV | `(8192,4096,512)` | 25.566 | 1.031x | `dense_bwd_q8_0_exact_n512k4096_g2_group_m2_padding8` |
| KV | `(32768,4096,512)` | 27.747 | 1.132x | `dense_bwd_q8_0_exact_n512k4096_g2_group_m2_padding8` |
| Output B | `(2048,8192,4096)` | 26.268 | 1.061x | `dense_bwd_q8_0_exact_n4096k8192_g2_padding8` |
| Output B | `(8192,8192,4096)` | 21.574 | 0.832x | `dense_bwd_q8_0_exact_n4096k8192_g2_group_m2` |
| Output B | `(32768,8192,4096)` | 20.844 | 0.835x | `dense_bwd_q8_0_exact_n4096k8192_g2_group_m2` |
| Shared gate/up | `(2048,4096,2048)` | 26.571 | 1.112x | `dense_bwd_q8_0_exact_n2048k4096_g2_group_m2_padding8` |
| Shared gate/up | `(8192,4096,2048)` | 24.880 | 1.033x | `dense_bwd_q8_0_exact_n2048k4096_g2_group_m2_padding8` |
| Shared gate/up | `(32768,4096,2048)` | 25.107 | 1.043x | `dense_bwd_q8_0_exact_n2048k4096_g2_group_m2_padding8` |
| Shared down | (2048,2048,4096) | 22.034 | 1.287x | `dense_bwd_q8_0_exact_n4096k2048_g2_group_m2_sw8` |
| Shared down | (8192,2048,4096) | 19.572 | 1.072x | `dense_bwd_q8_0_exact_n4096k2048_g2_group_m2_sw8` |
| Shared down | (32768,2048,4096) | 22.807 | 1.232x | `dense_bwd_q8_0_exact_n4096k2048_g2_group_m2_sw8` |
| Shared down Qwen4-Exp | `(2048,640,2560)` | 26.950 | 1.658x | `dense_bwd_q8_0_exact_n640k2560_g2_group_m2_padding8` |
| Shared down Qwen4-Exp | `(8192,640,2560)` | 25.446 | 1.513x | `dense_bwd_q8_0_exact_n640k2560_g2_group_m2_padding8` |
| Shared down Qwen4-Exp | `(32768,640,2560)` | 27.959 | 1.553x | `dense_bwd_q8_0_exact_n640k2560_g2_group_m2_padding8` |

The `Kernel` column names the deployed body for each exact key. The ordinary bodies are bitwise equal to the reference body of their per-key candidate campaign. The four split-contraction LM-head bodies keep the deployed decode and matrix work but sum FP32 partial tiles in ascending slice order and round once at the end, so they belong to the precision-changing class of the accuracy policy rather than the bitwise class.

| `(M,N,K)` | HIP time (ms) | HIP TFLOPS | HIP/torch.mm | Kernel |
| ---: | ---: | ---: | ---: | --- |
| `(32,4096,129280)` | 5.921 | 5.72 | 0.789x | `dense_bwd_q8_0_exact_lm_head_splitk_m32_s2` |
| `(64,4096,129280)` | 3.750 | 18.07 | 1.254x | `dense_bwd_q8_0_exact_lm_head_splitk_m64_s4` |
| `(128,4096,129280)` | 5.701 | 23.78 | 1.894x | `dense_bwd_q8_0_exact_lm_head_splitk_m128_s16` |
| `(256,4096,129280)` | 8.226 | 32.96 | 2.184x | `dense_bwd_q8_0_exact_lm_head_splitk_m256_s16` |
| `(512,4096,129280)` | 20.448 | 26.52 | 1.598x | `dense_bwd_q8_0_exact_lm_head_splitk_m512_s32` |

All five chunks run the split-contraction body described below, with the slice count fitted per chunk. Every row in the table comes from one official run against the same BF16 `torch.mm` baseline.

### Split-contraction deployment

The LM-head chunks launch a small number of workgroups against a contraction of 129,280, so the machine is starved at the narrow chunk and the ordinary tile cannot grow without giving up resident workgroups. The deployed mechanism splits the contraction: the same tile, decode and matrix work run over one contiguous slice per workgroup (`grid.z` is the slice index), each slice writes an FP32 partial tile, and a second kernel sums the slices in ascending order and rounds once to BF16. The slice width is a per-chunk constant rounded up to the 32-wide contraction step, so no slice can overlap its neighbour, and the partial workspace is sized `[slices][rows][in_features]` in FP32.

Both kernels run inside the timed region, and the slice bodies carry the same exact dimensions as the single-pass bodies, which is what makes them competitive: with runtime contraction and result widths the same tile reached only `0.43-0.59x` of the single-pass body at one slice. With exact dimensions one slice matches or beats the single-pass body (`0.977-1.010x`) and the slices then pay for themselves on the starved chunks: `0.789x` at M32 with two slices, `1.258x` at M64 with four, `1.894x` at M128 and `2.184x` at M256 with sixteen, and `1.598x` at M512 with thirty-two, against the BF16 `torch.mm` baseline, all measured under the official protocol in one run. The Q8_0 numbers above are that run.

## Kernel implementation

The initial generic body used 64x64/reduction-16 ownership, 92 VGPRs, 17 SGPRs, and 2 KiB LDS. The ordinary bodies use exact Q8_0 shapes with a four-wave 128x128/K32 geometry, width-16 decode, and row-dependent LDS padding. M1/M2 traversal is measured per shape and row count.

The build carries four ordinary variant generations: the plain `exact_n{N}k{K}` wrappers, the `_g1`/`_g2`/`_g3` geometry screen winners, and the `_g2_padding8`/`_g2_group_m{1,2}`/`_g2_group_m2_padding8` traversal variants. The deployed ordinary bodies are all G2-family (`n_tiles=8`, `k_iteration=32`, `decoder_width=16`, `active_waves=4`) with `lds_padding=8` and/or `group_m` set per key. `K` in the variant name is the reduction length, and no single variant wins every shape. The plain `exact_n{N}k{K}` wrappers are the pre-G generation that the `_g*` screen superseded and are not part of the deployment campaign.

The LM head uses active-two-wave M32, G0 M64, G1 M128, and G3 M256/M512 bodies. M512 launches two exact M256-style tiles. The single-pass campaign timed the built chunk bodies per `M` and selected `_bounded` at M32, `_full` at M64, `_g1` at M128, and `_g3` at M256/M512. The split-contraction campaign then replaced every chunk's single-pass body with an exact slice body whose slice count is fitted per chunk. `_g2` lost at M128/M256/M512, `_full` lost at M128 and above, `_bounded` lost at M64 and above, `_m32_active2` tied with the deployed `_bounded` body inside `0.2%` at M32, and `_g3` faults at M128, which `_g1` covers.

The ordinary G2 body is `192 VGPR / 14 SGPR / 8 KiB LDS`, and `padding8` adds 2 KiB of LDS to it. The isolated LM bodies use 91-194 VGPR and 2-4 KiB LDS.

### Pipelined tile

The pipelined stage order was screened on five of the seven ordinary families at `M=2048` and `32768`. It wins on the query A rows (`1.17x` at `M=2048`, level at `32768`), which are the deepest grid-starved points, and loses or collapses elsewhere: the swizzle-flavoured twins are `1.03-1.21x` behind on the query B, output B and shared down rows, the padded twin is `1.06x` behind on the shared down rows, and two configurations degenerate at `M=32768` (`10.1-10.8` against `19.6-22.8 TFLOPS`), the same pathology the non-pipelined padded body shows on that family. Only the two measured query A rows deploy the pipelined tile.

## Closed dtype and distribution avenues

The compute dtype is never the lever on this hardware. A pure WMMA probe measured `v_wmma_f32_16x16x16_bf16` at `54.93` TFLOPS, `v_wmma_i32_16x16x16_iu8` at `54.25` and both f16 forms at `54.5-54.9`, so int8, bf16 and f16 all run at the same rate, and the f16-accumulate form occupies the same eight registers as the f32 form, so it buys no occupancy either. There is no dtype swap that improves throughput by itself.

For the same reason an int8 backward path for the q8_0-class quants is not queued: it would add activation quantization and an epilogue to a body that is already decode and barrier limited, so it needs a profile that shows the bf16 decode dominating before it is worth building.

Stream-K-style work distribution was screened over all one hundred deployed dense keys. Eleven of them have a starved grid, and every one of those has a contraction of at least `2,048` rows, i.e. at least `64` slices, so split-contraction already reaches thousands of blocks and the slice sweep saturates before the contraction bound. Stream-K's extra freedom is granularity below the k stage and smoothing of the tail wave, which can only pay where the contraction is short enough to bound the slice count, so the mechanism stays screened out until a key appears with a short contraction and a starved grid.

## Optimization log

### Baseline and exact shape specialization

The generic baseline was resource-light but ownership-limited:

| Batch | Historical packed/BF16 weighted ms | Throughput ratio |
| ---: | ---: | ---: |
| 1 | `3266.8/767.4` | `0.235x` |
| 4 | `16920.7/3075.7` | `0.182x` |
| 16 | `72146.2/12140.1` | `0.168x` |

The first exact Q8_0 wrappers removed runtime shape, bounds, and address state. Exact specialization improved all 18 ordinary points by `11.32-246.14%`, with a `60.58%` geometric gain. The full ordinary wrappers remained resource-clean.

The generic body used `92 VGPR / 17 SGPR / 2 KiB LDS`. The geometry screen compared G0 `64x64/K16` at 92 VGPR, G1 `128x64/K32` at 118, G2 `128x128/K32` at 192, and G3 `256x64/K32` at 194. G2 won all six ordinary shape families. The first exact-shape bracket improved the full ordinary matrix by `77.49%` geometrically and by `86.99%/96.51%/99.95%` at B1/B4/B16 weighted latency.

### Ordinary geometry and padding

G2 128x128/K32 beat G0 64x64/K16, G1 128x64/K32, and G3 256x64/K32 across the six ordinary families. Unpadded G2 reported a repeated `79.17%` LDS-conflict metric. Padding8 lowered Q-A B1 conflict from `79.17%` to `58.33%`, derived LDS latency from about 585 to 245 cycles, and ALU stall from LDS from `24.19%` to `15.26%`.

Padding was retained only where exact row cost justified it. Q-B and output-B required separate traversal and padding decisions. Global J64 lost full ordinary tiles by `3.75-8.85%` depending on K and family.

### LM chunk geometries

The isolated backward medians for Q8_0 LM chunks were `8.297/7.459/7.569/10.386/23.206 ms` at M32/M64/M128/M256/M512. M32 keeps all waves for decode and barriers but limits cotangent loads, WMMA, and stores to two waves. M128 improved `19.48%`. M256/M512 improved `51.85%/60.64%` over the preceding controls.

### Traversal and closed controls

Changing only grouped-M traversal reduced Q-B B4/B16 by `38.77%/40.16%` and output-B B4/B16 by `42.03%/42.21%`. The later complete traversal correction reduced weighted packed latency by `4.51%/18.85%/26.41%` at B1/B4/B16. Normalized disassembly was `98.16-98.83%` opcode-identical, so locality and mapping, not a new decoder, supplied the improvement.

The DB8 screen retained M2 for Q-A, KV, shared gate/up, and shared down at the longer row counts, and M1 only for Q-B B1. Representative all-M-to-M2 times were Q-A B16 `32.386 -> 13.184 ms`, KV B16 `23.194 -> 5.253 ms`, shared gate B16 `54.401 -> 27.955 ms`, and shared down B16 `49.452 -> 27.961 ms`. Q-B B1 used the `7.241/6.923/7.509 ms` M2/M1/M2 bracket. L2 hit rate rose by `30.6-47.4` percentage points, while occupancy changed only modestly. `MemUnitBusy` was unavailable because rocprofv3 rejected its non-windowable `TA_TA_BUSY` dependency.

Activation-half double buffering, K-loop unrolling, width32 decode, stride77 padding, decoded-weight LDS caching, split-K, GSU, Stream-K, persistent workgroups, and direct-to-LDS/direct-to-VGPR rewrites are closed for the current packed representation.

### Qwen4-Exp shared-expert down

The seventh ordinary family is the Qwen4-Exp shared-expert down projection, `(M,640,2560)`: the result is only `640` wide, so a G2 workgroup covers a fifth of it and the whole grid is `64 x 5` workgroups at `M=8192`. The family's eight candidate bodies were timed on all three row counts at `4` repeats. The `_g2_group_m2_padding8` body wins every point, by `1.18-1.33x` over the unpadded `_g2` bodies, `1.25-3.61x` over `_g1`, `1.49-5.60x` over `_g3`, and `2.10-3.49x` over the pre-G plain wrapper. The padding carries the win at `M=2048` and the grouped-M traversal carries it at the larger row counts. The body is the family's standing choice for a narrow result and deploys on all three keys.

### Swizzle against padding

Seven swizzle-only twins of the padded G2 bodies were built (`_g2_*_sw8`/`_sw16`, 8 KiB rather than 10 KiB) and screened on the DeepSeek families at `M=2048` and `32768`. Padding wins on six of the seven: the swizzle twins are `1.05-1.18x` slower on the query A, query B, KV, output B and shared gate/up shapes, and `1.15-1.36x` slower at `sw16`. The exception is the shared down projection, whose result is `(M,2048)`: there `sw8` is `1.20x` ahead at `M=2048`, level at `M=8192` and `1.11x` ahead at `M=32768`, and the padded body collapses at the larger row counts (`10.8` against `19.6-22.8 TFLOPS`), which is why those three keys were never deployed with padding above `M=2048`. The three shared-down keys now deploy `exact_n4096k2048_g2_group_m2_sw8`. Everything else keeps padding. Evidence: `~/tmp/torch-ggml-ops/qwen4_fwd/bwd_swz_q80.txt`, `bwd_final_decisions.txt` and `bwd_deploy_confirm.txt`.

### Pipelined split-contraction head

The head chunks were retried with the pipelined slice body. The first attempt returned results 2.9x too large, which was two bugs in the new decode: the Q8_0 payload word was read as unsigned bytes, so every negative quant shifted by `+256`, and a slice whose last stage is shorter than the stage width needed the shipped body's zero-fill guard (the head's `129,280`-value contraction with sixteen or thirty-two slices does not divide by the 32-value step). With both fixed the body is correct on all four chunks, and it is *not* faster: `0.81x`/`0.87x` behind the deployed slice bodies at `M=64`/`128` and level at `M=256`/`512` (`1.01x`). The deployed bodies therefore stay, and the finding is that this type's slices are already deep in stage terms (about 250 stages each), so a stage pipeline has nothing left to overlap.

### Split-contraction measurement

The split-K closure above was a contract deferral, not a measurement. Retested with a dedicated split-contraction body: the deployed tile, decode and matrix work are unchanged, each workgroup takes one contiguous slice of the contraction, writes an FP32 partial tile, and a second kernel sums the slices in ascending order and rounds once to BF16. Both sides consume the same prepared gradient and the same packed weights, and the partial write plus the reduction are inside the timed region.

The first form used runtime contraction and result widths, so its loop bounds and the gradient row stride were runtime values. It lost everywhere (`0.43-0.59x` at one slice). Folding the exact dimensions into the body the way the deployed twins do restores parity at one slice (`0.977-1.010x`), and the split then pays where the ordinary grid is starved: `1.916x` at the M64 chunk (four slices), `1.275x` at M128, `1.202x` at M256 and `1.058x` at M512, with the best slice count between sixteen and thirty-two. The mechanism therefore needs per-shape, per-slice-count exact instantiations. The slice count is a per-key choice. Evidence and the experimental body live under `~/tmp/torch-ggml-ops/retune_dense_bwd/`.

## Evidence

The Qwen4-Exp rows come from `~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q80_final.txt` (screen in `bwd_q80_v1.txt`), which uses the deployed preparation, the same `torch.mm` BF16 baseline and the same paired timing as the official protocol, under the same harness as the Q4_K, Q5_K and Q6_K records.

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
