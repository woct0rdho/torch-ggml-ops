# HIP Grouped MMQ Forward Q4_K Experiment

## Scope

This record covers the routed Q4_K down kernel for Qwen on gfx1151.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | ---: | ---: | ---: | --- |
| 1 | `(16384,2048,512)` | 18.30 | 1.595x | `grouped_fwd_serial_q4_k_n2048_k512_j32` |
| 4 | `(65536,2048,512)` | 23.30 | 1.230x | `grouped_fwd_serial_q4_k_n2048_k512_j64` |
| 16 | `(262144,2048,512)` | 24.42 | 1.077x | `grouped_fwd_serial_q4_k_n2048_k512_j64` |

The small-route batch now uses the pure J32 body: it removes masked rows and fits a fifth workgroup per WGP, which is worth `5-10%` at B1. B4/B16 keep the J64 mixed-tail body, which is level with the uniform-route control at B16 (`4%`). The B4/B16 residual remains packed scale/minimum reconstruction against predecoded BF16 weights. Flagged prior-sensitive at B1/B4.

The table kernel is the deployed HIP body for these shapes and the selection rule that picks it is part of the deployed-control rule table; every candidate the retune measured was verified bitwise against the body in the table.

## Kernel implementation

The retained bodies use exact Q4_K N2048/K512 geometry and four wave32 waves: a pure J32 body for the small-route batch and J64 ownership with a bounded J32 tail for the larger batches. Both use width-16 packed decode, Q4 scale/minimum reconstruction, and BF16 output stores. It keeps inactive route handling in the device kernel and does not build host descriptors from offsets.

## Optimization log

### Mixed J32 tails

A bounded J64/J32 tail body was promoted at aggregate rows `R=16384` and `R=65536`. Weighted search/confirmation gains were `1.0591x/1.0828x` at B1 and `1.0243x/1.0279x` at B4. B16 retained the pure J64 tail after its confirmed movement was only `1.0124x` in the final campaign.

The pure J32 B4 typed probe was resource-clean but reached only `0.8970x` weighted prior speedup and `0.8466x` on the worst synthetic control, so it was rejected. The mixed tail is a bounded kernel mechanism, not a universal J32 replacement.

### Geometry campaign

The coefficient-only campaign screened exact J values, row-task alternatives, linked tail fields, decoder width, LDS layout, and bounded prefetch. Q4_K retained J64 with the measured J32 tails; broad row-task geometry and inactive-M policies were not promoted for this forward body. The retained Q4 path uses width-16 decode and a Q4-specific LDS arrangement.

The measured Q4 residual is repeated packed scale/minimum reconstruction rather than cache misses or high LDS stalls. Wider ownership, broad swizzles, two-LDS decoded-weight caches, split-K, Stream-K, persistent groups, and direct-to-LDS forms are closed.

The early Q4 redesign moved a representative B1 point from `6.874 ms` to `2.842 ms`; exact full-row and bounded-tail specialization later moved it to `1.599 ms`. The retained Q4 row-task body is `242 VGPR / 46 SGPR`. These measurements explain the accepted tiled/tail mechanism without replacing the final matrix above.

### Learned-route retune

The flagged prior-sensitive batches were re-swept under the learned route law: J bodies of the same geometry (J16, J32, J64, J80, J128), adaptive J cascades (64/32/16 by remaining rows), 8-wave workgroups with the J tile split across two warp groups, and a compact single-stage weight tile that halves the LDS footprint. All candidates were checked bitwise against the deployed body.

What the measurements show for this kernel family:
- The limiting resource is the number of independent workgroups resident per WGP, not the wave count. Reserving more dynamic LDS on one fixed body costs `15%` at four to three resident workgroups and `40%` at four to two, and a J32 body with a fifth workgroup gains about `11%`. Doubling the waves per workgroup (8-wave, J-split) changes nothing because the waves inside a workgroup stay phase-locked by the barriers.
- The kernel is not barrier- or traffic-bound: removing the barriers saves `2%`, and removing the activation or weight global loads (keeping the LDS stores) saves `22%`/`4%`.
- The scale/minimum epilogue is latency filling, not waste. Replacing its FMAs with a bare accumulate makes the body `1.8x` slower because nothing covers the WMMA and LDS latency any more.
- Masked rows of a large J tile cost much less than the per-tile weight decode: adaptive cascades, J16 and J128/J80 bodies all lose to the plain J32 (small routes) and J64 (large routes) bodies.
- The compact single-stage weight tile is bitwise-identical and fits half the LDS, but the extra per-stage loader calls cancel the occupancy gain at the same J. Re-measured on the current tree (activation prefetch enabled, staged-barrier redundancy removed) it is still inside the run-to-run spread: `+0.8%`/`+1.1%`/`-2.8%` for Q4_K and `+0.8%`/`-0.5%`/`-2.9%` for Q5_K at B1/B4/B16, against a `0.7%` spread between two builds of the identical deployed body. The LDS saving is real (`28,928` to `18,688` bytes, four to seven resident workgroups) but this family does not convert occupancy into time.

Still open after this round, in measured-payoff order: a swizzled activation tile and matching dot addressing to remove the `14%` LDS bank-conflict share (every 16-byte-aligned row stride this layout allows still conflicts, so this needs a layout change rather than padding), a permute-based nibble expansion for the decode-bound Q2_K bodies, and the I=32 tile that trades activation reuse for a smaller LDS footprint and a higher resident-workgroup count. The activation tile copy is closed: its 128-bit vectorised form is neutral, and the register prefetch above already covers the load latency.

### Activation prefetch

The ablation that removed the activation global reads (keeping the LDS stores) was worth about a fifth of the runtime, so the next retune overlapped that latency instead of shrinking the copy: the second activation plane of each k block is now loaded into registers before the first dot and only stored to LDS after it. The switch is the `prefetch_activation` build knob, so it is a property of the control rather than of the kernel family.

Within one build tree the staged body is `1.9%` faster at the largest batch and `2-3.5%` faster at the smaller ones; under the benchmark protocol the three Qwen families gain `1.6-3.2%` at B1/B4 and are neutral at B16, where the groups are large enough that the serial row-tile loop already covers the load latency. The DeepSeek Q2_K bodies keep the knob off: their decode-bound instruction stream has no slack to fill, and the staging registers cost `1-2%`.

Two neighbouring ideas were measured and rejected on the same route banks: a 128-bit vectorised activation copy is neutral (so it is the load latency, not the copy instruction count, that matters), and row-task ownership of the `n2048k512` shapes loses at B16 (`+2.2%` for Q4_K, `-2.4%` for Q5_K, `-8.9%` for IQ2_S) because one workgroup per J tile pays the per-workgroup setup that the serial row loop amortises.

Measured instructions are `34%` of issue slots and WMMA is about `8%` of sampled stalls (VALU `52%`, barriers `14%`, LDS `10%`), so the remaining limit is the tile load, decode and LDS-store stream rather than the matrix unit.

The small-route batch now runs a pure J32 body (`grouped_fwd_serial_q4_k_n2048_k512_j32`, 24,192 B LDS) while the J64 mixed-tail body stays for the larger batches. J80 loses at every row count.

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
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/qwen_q4_down_bounded_mixed_search_25.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/qwen_q4_down_bounded_mixed_confirmation_25.json
```

The next meaningful Q4_K experiment must reduce packed scale/minimum decode or change the weight representation.

The final campaign also retained the shared grouped-forward controls and exact-key resource evidence:

```text
~/tmp/torch-ggml-ops/grouped-fwd-prior-v1.json
~/tmp/torch-ggml-ops/grouped-fwd-candidate-resources.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/qwen_q4_down_bounded_mixed_search_25.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/qwen_q4_down_bounded_mixed_confirmation_25.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
~/tmp/torch-ggml-ops/grouped-fwd-production-final-resources.json
```
