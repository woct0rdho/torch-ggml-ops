# HIP Grouped MMQ Forward IQ2_S Experiment

## Scope

This record covers the routed single-projection IQ2_S down kernel for Qwen on gfx1151.

The exact aggregate rows are `R=16384,65536,262144`.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | ---: | ---: | ---: | --- |
| 1 | `(16384,2048,512)` | 13.28 | 1.229x | `grouped_fwd_serial_iq2_s_n2048_k512_j64_j32` |
| 4 | `(65536,2048,512)` | 18.15 | 0.953x | `grouped_fwd_serial_iq2_s_n2048_k512_j64_j32` |
| 16 | `(262144,2048,512)` | 19.80 | 0.871x | `grouped_fwd_serial_iq2_s_n2048_k512_j64` |

Under the learned routes the B1/B4 batches trail their uniform-route control by `24%`/`14%` while B16 is level (`4%`). The B4/B16 loss is repeated IQ2_S lookup, sign, scale and packed-weight staging against predecoded BF16 weights. Flagged prior-sensitive at B1/B4; the J sweep confirmed the deployed mixed J64/J32 body is still the fastest of the J space.

The table kernel is the deployed HIP body for these shapes and the selection rule that picks it is part of the deployed-control rule table; every candidate the retune measured was verified bitwise against the body in the table.

## Kernel implementation

The retained single-down body uses exact `(N,K)=(2048,512)` geometry, J64 main ownership with a bounded J32 tail variant, cooperative width-16 IQ2_S decode, and a single-projection LDS layout. The kernel masks inactive or partial rows without moving route data to the host.

The paired IQ2_S kernel uses a different LDS swizzle and is not a valid replacement for this single-down body.

## Optimization log

### Mixed-tail retune

The existing J64/J32 mixed-tail body was tested at exact aggregate rows `R=65536`. It changed only the kernel geometry used for the final tail and preserved packed decode, output semantics, and route ABI. The prior weighted speedup was `1.0321x`; reversed-order 25-repeat confirmation measured `1.0228x`, with minimum prior and synthetic controls of `1.0114x` and `1.0021x`. Outputs were bitwise identical.

A standalone pure J32 typed probe passed resource checks but reached only `0.8970x` weighted prior speedup and `0.8466x` on the worst synthetic control, so pure J32 was rejected as a general replacement.

### B1 ownership and coefficient search

The broad coefficient-only campaign retained the existing serial J64/J32 ownership for IQ2_S down and found no new geometry that passed all controls. Width-8 decode duplicated metadata work. M256/N64 doubled workgroups and regressed larger routes. Inactive-M suppression was unstable for IQ2_S and was rejected.

### Bottleneck attribution

A selected B16 trace separated the packed body from activation preparation:

| Component | Mean kernel time |
| --- | ---: |
| HIP IQ2_S body | `25.703 ms` |
| HIP BF16-to-Q8_1 quantizer | `1.849 ms` |
| AITER GMM | `24.094 ms` |

The packed body was `1.067x` slower than AITER, while the traced HIP total was `1.144x` slower. Quantization accounts for `6.71%` of HIP traced kernel time and about `53.5%` of the small traced excess; the packed decoder accounts for the rest.

The selected-region source-of-record traces are:

```text
~/tmp/torch-ggml-ops/profile-grouped-fwd-20260812/qwen_iq2s_down_b16_packed/trace_results.db
~/tmp/torch-ggml-ops/profile-grouped-fwd-20260812/qwen_iq2s_down_b16_aiter/trace_results.db
```

### Learned-route retune

The flagged prior-sensitive batches were re-swept under the learned route law: J bodies of the same geometry (J16, J32, J64, J80, J128), adaptive J cascades (64/32/16 by remaining rows), 8-wave workgroups with the J tile split across two warp groups, and a compact single-stage weight tile that halves the LDS footprint. All candidates were checked bitwise against the deployed body.

What the measurements show for this kernel family:
- The limiting resource is the number of independent workgroups resident per WGP, not the wave count. Reserving more dynamic LDS on one fixed body costs `15%` at four to three resident workgroups and `40%` at four to two, and a J32 body with a fifth workgroup gains about `11%`. Doubling the waves per workgroup (8-wave, J-split) changes nothing because the waves inside a workgroup stay phase-locked by the barriers.
- The kernel is not barrier- or traffic-bound: removing the barriers saves `2%`, and removing the activation or weight global loads (keeping the LDS stores) saves `22%`/`4%`.
- The scale/minimum epilogue is latency filling, not waste. Replacing its FMAs with a bare accumulate makes the body `1.8x` slower because nothing covers the WMMA and LDS latency any more.
- Masked rows of a large J tile cost much less than the per-tile weight decode: adaptive cascades, J16 and J128/J80 bodies all lose to the plain J32 (small routes) and J64 (large routes) bodies.
- The compact single-stage weight tile is bitwise-identical and fits half the LDS, but the extra per-stage loader calls cancel the occupancy gain at the same J. Re-measured with the corrected staging it is a loss at every batch (`-6.0%` at B1, `-1.1%` at B4, `-6.6%` at B16), because halving the threads per row doubles the grid-decode work per thread in this loader.

Still open after this round, in measured-payoff order: a swizzled activation tile and matching dot addressing to remove the `14%` LDS bank-conflict share (every 16-byte-aligned row stride this layout allows still conflicts, so this needs a layout change rather than padding), a permute-based nibble expansion for the decode-bound Q2_K bodies, and the I=32 tile that trades activation reuse for a smaller LDS footprint and a higher resident-workgroup count. The activation tile copy is closed: its 128-bit vectorised form is neutral, and the register prefetch above already covers the load latency.

### Activation prefetch

The ablation that removed the activation global reads (keeping the LDS stores) was worth about a fifth of the runtime, so the next retune overlapped that latency instead of shrinking the copy: the second activation plane of each k block is now loaded into registers before the first dot and only stored to LDS after it. The switch is the `prefetch_activation` build knob, so it is a property of the control rather than of the kernel family.

Within one build tree the staged body is `1.9%` faster at the largest batch and `2-3.5%` faster at the smaller ones; under the benchmark protocol the three Qwen families gain `1.6-3.2%` at B1/B4 and are neutral at B16, where the groups are large enough that the serial row-tile loop already covers the load latency. The DeepSeek Q2_K bodies keep the knob off: their decode-bound instruction stream has no slack to fill, and the staging registers cost `1-2%`.

Two neighbouring ideas were measured and rejected on the same route banks: a 128-bit vectorised activation copy is neutral (so it is the load latency, not the copy instruction count, that matters), and row-task ownership of the `n2048k512` shapes loses at B16 (`+2.2%` for Q4_K, `-2.4%` for Q5_K, `-8.9%` for IQ2_S) because one workgroup per J tile pays the per-workgroup setup that the serial row loop amortises.

The sign-mask table that the paired study introduced is shared with this body and was measured paired in one process against the previous build: `4.7%` at B1 with the J64/J32 mixed body and level at B16, bitwise identical. The table replaces the per-byte `__vcmpne4` chain with one eight-byte load.

Stochastic PC sampling of the deployed J64 body at B16 shows where the remaining time sits: `22.0%` of samples are half/int conversion and `11.6%` are scale math in the dot epilogue, against `2.9%` WMMA (`22.7%` other VALU, `15.3%` barrier, `8.4%` scalar, `7.5%` LDS). Stall reasons are dominated by issue arbitration (`ARBITER_NOT_WIN` `42.0%`, `ALU_DEPENDENCY` `22.5%`, `ARBITER_WIN_EX_STALL` `16.4%`) with a `14.2%` barrier share, which is the largest of the grouped forward bodies: at `30,976` bytes of dynamic LDS the J64 body only fits four workgroups per WGP, so barrier waits stay exposed. The compact single-stage tile was the obvious way to raise that occupancy and it was measured again on the current tree: it is bitwise identical but a loss at every batch, because the halved threads-per-row doubles the grid decode per thread.

Measured instructions are `34%` of issue slots and WMMA is about `8%` of sampled stalls (VALU `52%`, barriers `14%`, LDS `10%`), so the remaining limit is the tile load, decode and LDS-store stream rather than the matrix unit.

The deployed mixed J64/J32 and J64 bodies remain the fastest of the J space for this decoder; J16 is `4-5x` slower and J80 loses at both larger row counts.

## Evidence

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/grouped_fwd_current/pass13_fwd_qwen_default.json
```

The original campaign evidence is:

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/fwd-dispatch-screen/qwen_iq2s_j32_b4_9.json
~/tmp/torch-ggml-ops/fwd-dispatch-screen/qwen_iq2s_j32_b4_25.json
~/tmp/torch-ggml-ops/fwd-dispatch-screen/qwen_iq2s_pure_j32_b4_5.json
~/tmp/torch-ggml-ops/profile-grouped-fwd-20260812/qwen_iq2s_down_b16_packed/trace_results.db
```

The remaining improvement requires packed IQ2_S representation or explicit decode reuse. Another generic J or tail sweep is closed.
