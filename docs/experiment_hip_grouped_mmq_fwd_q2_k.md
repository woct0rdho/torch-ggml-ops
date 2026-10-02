# HIP Grouped MMQ Forward Q2_K Experiment

## Scope

This record covers the routed Q2_K down kernel for DeepSeek.

The routed row counts are `R=12288,49152,196608` for physical batches 1, 4, and 16.

The benchmark samples the `blk.3` expert tensors; `blk.0`-`blk.2` are hash-routed in this model, so the learned prior that the benchmark infers by default matches the sampled layers.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | ---: | ---: | ---: | --- |
| 1 | `(12288,4096,2048)` | 11.12 | 1.767x | `grouped_fwd_serial_q2_k_n4096_k2048_j32_j16` |
| 4 | `(49152,4096,2048)` | 12.41 | 0.905x | `grouped_fwd_serial_q2_k_n4096_k2048_j32_j16` |
| 16 | `(196608,4096,2048)` | 12.54 | 0.793x | `grouped_fwd_serial_q2_k_n4096_k2048_j32` |

The B1/B4 bodies are the mixed J32/J16 and J32 variants; they trail their uniform-route control by `9%`/`4%`, and B16 is level (`0%`). The B4/B16 gap remains repeated Q2_K scale/minimum reconstruction and limited N64 reuse against predecoded BF16 weights. Flagged prior-sensitive at B1: the J32/J16 routing threshold may need a learned-route retune.

The table kernel is the deployed HIP body for these shapes and the selection rule that picks it is part of the deployed-control rule table; every candidate the retune measured was verified bitwise against the body in the table.

## Kernel implementation

The retained body uses four wave32 waves, exact N4096/K2048 geometry, width-16 Q2_K decode, shared scale/minimum and packed-shift work, serial J32 ownership, and a bounded J16 tail body where measured. Inactive-M consumer suppression skips only inactive cotangent/WMMA/store work while keeping decode and barriers uniform.

The kernel handles routed tails, inactive experts, and device-resident route metadata without host offset reads.

## Optimization log

### Exact geometry and decode

The generic grouped baseline used narrow N16/K16 ownership. Exact Q2_K M64/M128 and J32 geometry increased N reuse and removed much of the serial overhead. Width-16 decode shares each scale/min group and packed shift across natural value groups.

Reduction unroll U2 improved all B4 routes but regressed B1 and one B16 boundary point, so it remained a bounded kernel variant. U4 lost to U2 by `1.2-4.3%`. A valid N128/U1 control compiled at 255 VGPRs but lost every B16 route by `1.18-4.18%`; N128 was rejected.

The retained Q2_K bodies use `208 VGPR / 38 SGPR / 4096 B LDS` for the regular J32 body and `213 VGPR / 43 SGPR / 4096 B LDS` for the J32/J16 mixed body. The exact B4 mixed control reached `1.0227x` in search and `1.0247x` in disjoint confirmation, with minimum learned/hash gains `1.0169x/1.0173x` and minimum controls `0.9998x/0.9996x`.

### B4 mixed tail

The existing J32/J16 body was measured at exact B4 aggregate rows `R=49152`. The longer confirmation improved the selected body by `1.0247x`; minimum learned/hash prior gains were `1.0169x/1.0173x`, and minimum mandatory controls were `0.9998x/0.9996x`. Outputs remained bitwise identical.

The coefficient-only campaign found no benefit from row tasks: Q2_K already exposes 32 N workgroups per expert and thousands of total workgroups, while row tasks would not remove rounded tail arithmetic.

### Compact single-stage weight tile

The weight tile now holds one 128-value stage per row - 32 quant ints plus that stage's eight 16-value scale/minimum pairs - instead of a whole 256-value block, so the two stages of a k block share one footprint and the per-row stride drops from `100` to `40` ints. The dynamic LDS of the J32 body falls from `30,336` to `14,976` bytes, which lifts the resident workgroups per WGP from four to eight. Everything else (decode width, tail bodies, activation staging) is unchanged, and the new body is bitwise identical to the old one.

Under the benchmark protocol the three batches gain `4.0-4.5%` (`11.12`/`12.41`/`12.54` TFLOPS against `10.65`/`11.85`/`12.04`), and the within-tree A/B is `5-8%`. A narrower row stride alone (`80` ints, the minimum the plain layout allows) is neutral, so the gain is not simply occupancy: this family is instruction-bound, which is also why the activation prefetch and row-task ownership were rejected here.

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

### Learned-route retune

The flagged prior-sensitive batches were re-swept under the learned route law: J bodies of the same geometry (J16, J32, J64, J80, J128), adaptive J cascades (64/32/16 by remaining rows), 8-wave workgroups with the J tile split across two warp groups, and a compact single-stage weight tile that halves the LDS footprint. All candidates were checked bitwise against the deployed body.

What the measurements show for this kernel family:
- The limiting resource is the number of independent workgroups resident per WGP, not the wave count. Reserving more dynamic LDS on one fixed body costs `15%` at four to three resident workgroups and `40%` at four to two, and a J32 body with a fifth workgroup gains about `11%`. Doubling the waves per workgroup (8-wave, J-split) changes nothing because the waves inside a workgroup stay phase-locked by the barriers.
- The kernel is not barrier- or traffic-bound: removing the barriers saves `2%`, and removing the activation or weight global loads (keeping the LDS stores) saves `22%`/`4%`.
- The scale/minimum epilogue is latency filling, not waste. Replacing its FMAs with a bare accumulate makes the body `1.8x` slower because nothing covers the WMMA and LDS latency any more.
- Masked rows of a large J tile cost much less than the per-tile weight decode: adaptive cascades, J16 and J128/J80 bodies all lose to the plain J32 (small routes) and J64 (large routes) bodies.
- The compact single-stage weight tile is bitwise-identical and fits half the LDS, but the extra per-stage loader calls cancel the occupancy gain at the same J.

Still open after this round: the same compact single-stage tile for the IQ2_S bodies (their 84-int stride would fall to 40 and fit six resident workgroups instead of four), a permute-based nibble expansion for this decode-bound decoder, and a swizzled activation tile with matching dot addressing for the `14%` LDS bank-conflict share.

### Activation prefetch

The activation stream was the next target: staging the second activation plane in registers before the first dot and storing it afterwards costs `1-2%` on this family, so the Q2_K controls keep the `prefetch_activation` knob off. Their decode-bound instruction stream leaves no slack for the staging registers to fill, unlike the three Qwen families where the same change is worth `1.6-3.2%` at B1/B4. Row-task ownership of the `n4096k2048` shapes was not pursued further because the single-projection row-task bodies already lose on the Qwen shapes that share the tiling.

Measured instructions are `34%` of issue slots and WMMA is about `8%` of sampled stalls (VALU `52%`, barriers `14%`, LDS `10%`), so the remaining limit is the tile load, decode and LDS-store stream rather than the matrix unit.

The deployed mixed J32/J16 (small routes) and J32 bodies remain the fastest; the J64 body is `2x` slower and J16 loses to the mixed tail.

## Evidence

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/grouped_fwd_current/pass13_fwd_deepseek_default.json
~/tmp/torch-ggml-ops/grouped_fwd_current/pass13b_fwd_deepseek_default.json
```

The original campaign evidence is:

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
