# HIP Grouped MMQ Forward Q5_K Experiment

## Scope

This record covers the routed Q5_K down kernel for Qwen on gfx1151.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | ---: | ---: | ---: | --- |
| 1 | `(16384,2048,512)` | 17.85 | 1.651x | `grouped_fwd_serial_q5_k_n2048_k512_j32` |
| 4 | `(65536,2048,512)` | 21.91 | 1.159x | `grouped_fwd_serial_q5_k_n2048_k512_j64` |
| 16 | `(262144,2048,512)` | 24.27 | 1.071x | `grouped_fwd_serial_q5_k_n2048_k512_j64` |

The B1 body is the J32 small-route variant and the B4/B16 bodies are J64. They trail their uniform-route control by `19%`/`16%` and are level at B16 (`4%`). The larger-route deficit remains Q5 high-bit reconstruction and packed metadata cost. The J sweep confirmed the deployed selection for all three batches. Flagged prior-sensitive at B1/B4.

## Decode ablation

The Q5_K staging loader reconstructs each staged value from a low nibble word and a high-bit word (`ql0 | qh0`, `ql1 | qh1`) and rebuilds the per-block scale and minimum pair (`dm * make_half2(sc, m)`) before writing them to shared memory. One ablation replaced both with the raw loaded words while keeping the global loads, the shared stores, the barriers and the matrix work, and was timed against the unmodified build on the largest deployed route, in the harness at `~/tmp/torch-ggml-ops/r3_epilogue/`:

| variant | time | TFLOPS | against unmodified |
| --- | ---: | ---: | ---: |
| unmodified | 21.529 ms | 25.54 | 1.000x |
| loader decode replaced by the raw words | 22.090 ms | 24.89 | 0.975x |

Removing the entire Q5_K decode arithmetic buys nothing, and the same ablation on the grouped Q2_K forward measures `0.992x`. Together with the load-side ablations (Q2_K packed weights `0.995x`, IQ2_XXS packed weights worth `1.241x` through latency alone rather than instruction count) this closes the per-quant decode item for the bodies measured: the staged decode is not on the critical path, and where the packed loads do matter it is their latency, which the tile budget leaves no room to hide.

## Optimization log

### Early grouped redesign

The generic grouped baseline used eight waves, narrow N16/K16 ownership, scalar decode, and serial row chunks. The Q5 path was rebuilt on the four-wave tiled decoder used for Q4_K, then specialized for Q5 low/high payload reconstruction. Exact shape specialization removed generic bounds and address state.

### J and route-shape controls

The coefficient-only full campaign screened J16, J32, J64, J80, J128, row-task ownership, and linked tail policies. Q5 retained current J64/J32 ownership. Missing geometries either failed resources or timing. A universal row-task swizzle8 policy improved large row-task controls but regressed small B1 routes by `5.6-22.5%`, so it was not applied globally.

Q5 high-bit reconstruction raises register pressure. The remaining loss is decoder work rather than an untested generic launch ordering. Width-8 decode, M256/N64, broad inactive-M suppression, N-major tasks, split task lists, two-LDS caches, split-K, Stream-K, and persistent workgroups are closed.

The standalone small-route J32 body improved nonuniform routes by approximately `14-16%` but regressed the uniform route. Its retained resource point is `189 VGPR / 32 SGPR`. The row-task body reached 242-256 VGPR before final suppression and remained spill-free. Q5 padding and universal swizzle8 were rejected, so this record keeps the small-route geometry and row-task mechanism separate.

### Learned-route retune

The flagged prior-sensitive batches were re-swept under the learned route law: J bodies of the same geometry (J16, J32, J64, J80, J128), adaptive J cascades (64/32/16 by remaining rows), 8-wave workgroups with the J tile split across two warp groups, and a compact single-stage weight tile that halves the LDS footprint.

What the measurements show for this kernel family:
- The limiting resource is the number of independent workgroups resident per WGP, not the wave count. Reserving more dynamic LDS on one fixed body costs `15%` at four to three resident workgroups and `40%` at four to two, and a J32 body with a fifth workgroup gains about `11%`. Doubling the waves per workgroup (8-wave, J-split) changes nothing because the waves inside a workgroup stay phase-locked by the barriers.
- The kernel is not barrier- or traffic-bound: removing the barriers saves `2%`, and removing the activation or weight global loads (keeping the LDS stores) saves `22%`/`4%`.
- The scale/minimum epilogue is latency filling, not waste. Replacing its FMAs with a bare accumulate makes the body `1.8x` slower because nothing covers the WMMA and LDS latency any more.
- Masked rows of a large J tile cost much less than the per-tile weight decode: adaptive cascades, J16 and J128/J80 bodies all lose to the plain J32 (small routes) and J64 (large routes) bodies.
- The compact single-stage weight tile fits half the LDS, but the extra per-stage loader calls cancel the occupancy gain at the same J. Re-measured on the current tree (activation prefetch enabled, staged-barrier redundancy removed) it is still inside the run-to-run spread: `+0.8%`/`+1.1%`/`-2.8%` for Q4_K and `+0.8%`/`-0.5%`/`-2.9%` for Q5_K at B1/B4/B16, against a `0.7%` spread between two builds of the identical deployed body. The LDS saving is real (`28,928` to `18,688` bytes, four to seven resident workgroups) but this family does not convert occupancy into time.

Still open after this round, in measured-payoff order: a swizzled activation tile and matching dot addressing to remove the `14%` LDS bank-conflict share (every 16-byte-aligned row stride this layout allows still conflicts, so this needs a layout change rather than padding), a permute-based nibble expansion for the decode-bound Q2_K bodies, and the I=32 tile that trades activation reuse for a smaller LDS footprint and a higher resident-workgroup count. The activation tile copy is closed: its 128-bit vectorised form is neutral, and the register prefetch above already covers the load latency.

### Activation prefetch

The ablation that removed the activation global reads (keeping the LDS stores) was worth about a fifth of the runtime, so the next retune overlapped that latency instead of shrinking the copy: the second activation plane of each k block is now loaded into registers before the first dot and only stored to LDS after it.
Within one build tree the staged body is `1.9%` faster at the largest batch and `2-3.5%` faster at the smaller ones. Under the benchmark protocol the three Qwen families gain `1.6-3.2%` at B1/B4 and are neutral at B16, where the groups are large enough that the serial row-tile loop already covers the load latency. The DeepSeek Q2_K bodies keep the knob off: their decode-bound instruction stream has no slack to fill, and the staging registers cost `1-2%`.

Two neighbouring ideas were measured and rejected on the same route banks: a 128-bit vectorised activation copy is neutral (so it is the load latency, not the copy instruction count, that matters), and row-task ownership of the `n2048k512` shapes loses at B16 (`+2.2%` for Q4_K, `-2.4%` for Q5_K, `-8.9%` for IQ2_S) because one workgroup per J tile pays the per-workgroup setup that the serial row loop amortises.

Measured instructions are `34%` of issue slots and WMMA is about `8%` of sampled stalls (VALU `52%`, barriers `14%`, LDS `10%`), so the remaining limit is the tile load, decode and LDS-store stream rather than the matrix unit.

The deployed J32 (small routes) and J64 bodies remain the fastest of the J space. J16, J80 and J128 lose at every row count.

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
