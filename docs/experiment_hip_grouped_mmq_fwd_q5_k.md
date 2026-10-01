# HIP Grouped MMQ Forward Q5_K Experiment

## Scope

This record covers the routed Q5_K down kernel for Qwen on gfx1151.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | ---: | ---: | ---: | --- |
| 1 | `(16384,2048,512)` | 17.22 | 1.577x | `grouped_fwd_serial_q5_k_n2048_k512_j32` |
| 4 | `(65536,2048,512)` | 21.89 | 1.150x | `grouped_fwd_serial_q5_k_n2048_k512_j64` |
| 16 | `(262144,2048,512)` | 24.81 | 1.072x | `grouped_fwd_serial_q5_k_n2048_k512_j64` |

The B1 body is the J32 small-route variant and trails its uniform-route control by `19%`; the J64 B4 batch trails by `16%`, and B16 is level (`4%`). The larger-route deficit remains Q5 high-bit reconstruction and packed metadata cost. Flagged prior-sensitive at B1/B4: the J32/J64 thresholds are route-entry based and may need a learned-route retune.

The table kernel is the deployed HIP body for these shapes and rebuilds byte-identically from the current sources; the retained choice was measured on other route distributions, so a learned-route candidate sweep is the follow-up for the flagged batches.

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
