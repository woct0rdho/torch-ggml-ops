# HIP Grouped MMQ Forward Q5_K Experiment

## Scope

This record covers the routed Q5_K down kernel for Qwen on gfx1151.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | ---: | ---: | ---: | --- |
| 1 | `(16384,2048,512)` | 17.27 | 1.591x | `grouped_fwd_serial_q5_k_n2048_k512_j32` |
| 4 | `(65536,2048,512)` | 21.61 | 1.136x | `grouped_fwd_serial_q5_k_n2048_k512_j64` |
| 16 | `(262144,2048,512)` | 24.71 | 1.070x | `grouped_fwd_serial_q5_k_n2048_k512_j64` |

The B1 body is the J32 small-route variant and the B4/B16 bodies are J64; they trail their uniform-route control by `19%`/`16%` and are level at B16 (`4%`). The larger-route deficit remains Q5 high-bit reconstruction and packed metadata cost. The J sweep confirmed the deployed selection for all three batches; flagged prior-sensitive at B1/B4.

The table kernel is the deployed HIP body for these shapes and the selection rule that picks it is part of the deployed-control rule table; every candidate the retune measured was verified bitwise against the body in the table.

## Kernel implementation

The retained Q5_K body uses exact N2048/K512 geometry, four wave32 waves, serial J64 ownership with a measured small-route J32 body, width-16 low/high decode, Q5 scale/minimum reconstruction, and BF16 output stores. It handles inactive and partial routed groups in device code.

Q5_K row-task and serial bodies use different LDS/resource points. A standalone J32 body uses 189 VGPRs and 32 SGPRs.

## Optimization log

### Early grouped redesign

The generic grouped baseline used eight waves, narrow N16/K16 ownership, scalar decode, and serial row chunks. The Q5 path was rebuilt on the four-wave tiled decoder used for Q4_K, then specialized for Q5 low/high payload reconstruction. Exact shape specialization removed generic bounds and address state.

### J and route-shape controls

The coefficient-only full campaign screened J16, J32, J64, J80, J128, row-task ownership, and linked tail policies. Q5 retained current J64/J32 ownership; missing geometries either failed resources or timing. A universal row-task swizzle8 policy improved large row-task controls but regressed small B1 routes by `5.6-22.5%`, so it was not applied globally.

Q5 high-bit reconstruction raises register pressure. The remaining loss is decoder work rather than an untested generic launch ordering. Width-8 decode, M256/N64, broad inactive-M suppression, N-major tasks, split task lists, two-LDS caches, split-K, Stream-K, and persistent workgroups are closed.

The standalone small-route J32 body improved nonuniform routes by approximately `14-16%` but regressed the uniform route. Its retained resource point is `189 VGPR / 32 SGPR`; the row-task body reached 242-256 VGPR before final suppression and remained spill-free. Q5 padding and universal swizzle8 were rejected, so this record keeps the small-route geometry and row-task mechanism separate.

### Learned-route retune

The flagged prior-sensitive batches were re-swept under the learned route law: J bodies of the same geometry (J16, J32, J64, J80, J128), adaptive J cascades (64/32/16 by remaining rows), 8-wave workgroups with the J tile split across two warp groups, and a compact single-stage weight tile that halves the LDS footprint. All candidates were checked bitwise against the deployed body.

What the measurements show for this kernel family:
- The limiting resource is the number of independent workgroups resident per WGP, not the wave count. Reserving more dynamic LDS on one fixed body costs `15%` at four to three resident workgroups and `40%` at four to two, and a J32 body with a fifth workgroup gains about `11%`. Doubling the waves per workgroup (8-wave, J-split) changes nothing because the waves inside a workgroup stay phase-locked by the barriers.
- The kernel is not barrier- or traffic-bound: removing the barriers saves `2%`, and removing the activation or weight global loads (keeping the LDS stores) saves `22%`/`4%`.
- The scale/minimum epilogue is latency filling, not waste. Replacing its FMAs with a bare accumulate makes the body `1.8x` slower because nothing covers the WMMA and LDS latency any more.
- Masked rows of a large J tile cost much less than the per-tile weight decode: adaptive cascades, J16 and J128/J80 bodies all lose to the plain J32 (small routes) and J64 (large routes) bodies.
- The compact single-stage weight tile is bitwise-identical and fits half the LDS, but the extra per-stage loader calls cancel the occupancy gain at the same J.

Still open after this round, in measured-payoff order: a 128-bit activation tile load/store path (the activation stream is worth about a fifth of the time and sits on the critical path), an I=32 tile that trades activation reuse for a much smaller LDS footprint and a higher resident-workgroup count, and combining the compact weight tile with a four-wave workgroup so the extra resident workgroups are actually used.

Measured instructions are `34%` of issue slots and WMMA is about `8%` of sampled stalls (VALU `52%`, barriers `14%`, LDS `10%`), so the remaining limit is the tile load, decode and LDS-store stream rather than the matrix unit.

The deployed J32 (small routes) and J64 bodies remain the fastest of the J space; J16, J80 and J128 lose at every row count.

## Evidence

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/grouped_fwd_current/pass13_fwd_qwen_default.json
```

The original campaign evidence is:

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped-fwd-prior-v1.json
~/tmp/torch-ggml-ops/grouped-fwd-candidate-resources.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
```

Further Q5_K gains require a lower-state high-bit decoder or a prepared lossless representation.

The final Q5 record is anchored by these campaign artifacts:

```text
~/tmp/torch-ggml-ops/grouped-fwd-prior-v1.json
~/tmp/torch-ggml-ops/grouped-fwd-candidate-resources.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
~/tmp/torch-ggml-ops/grouped-fwd-production-final-resources.json
```
