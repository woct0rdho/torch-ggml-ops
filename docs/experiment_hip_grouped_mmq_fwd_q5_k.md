# HIP Grouped MMQ Forward Q5_K Experiment

## Scope

This record covers the routed Q5_K down kernel for Qwen on gfx1151:

```text
Y[R,2048] = X[R,512] @ W[2048,512].T
```

The packed Q5_K weight is decoded directly in the grouped kernel. BF16 activations and outputs use the fixed Q8_1 producer layout, while route metadata stays on the device.

## Final kernel result

`HIP TFLOPS` is `2*R*N*K/time`. `HIP/AITER GMM` compares with the exact BF16 AITER GMM baseline. Values above `1.00x` favor HIP.

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM |
| ---: | ---: | ---: | ---: |
| 1 | `(16384,2048,512)` | 19.56 | 1.651x |
| 4 | `(65536,2048,512)` | 23.65 | 1.026x |
| 16 | `(262144,2048,512)` | 24.33 | 0.968x |

The larger-route deficit is attributed to Q5 high-bit reconstruction and packed metadata cost relative to predecoded BF16 weights.

## Kernel implementation

The retained Q5_K body uses exact N2048/K512 geometry, four wave32 waves, serial J64 ownership with a measured small-route J32 body, width-16 low/high decode, Q5 scale/minimum reconstruction, and BF16 output stores. It handles inactive and partial routed groups in device code.

Q5_K row-task and serial bodies use different LDS/resource points. The final retained resources include a standalone J32 body at 189 VGPRs and 32 SGPRs with no private storage or spills.

## Optimization log

### Early grouped redesign

The generic grouped baseline used eight waves, narrow N16/K16 ownership, scalar decode, and serial row chunks. The Q5 path was rebuilt on the four-wave tiled decoder used for Q4_K, then specialized for Q5 low/high payload reconstruction. Exact shape specialization removed generic bounds and address state.

### J and route-shape controls

The coefficient-only full campaign screened J16, J32, J64, J80, J128, row-task ownership, and linked tail policies. Q5 retained current J64/J32 ownership; missing geometries either failed resources or timing. A universal row-task swizzle8 policy improved large row-task controls but regressed small B1 routes by `5.6-22.5%`, so it was not applied globally.

Q5 high-bit reconstruction raises register pressure. The remaining loss is decoder work rather than an untested generic launch ordering. Width-8 decode, M256/N64, broad inactive-M suppression, N-major tasks, split task lists, two-LDS caches, split-K, Stream-K, and persistent workgroups are closed.

The standalone small-route J32 body improved nonuniform routes by approximately `14-16%` but regressed the uniform route. Its retained resource point is `189 VGPR / 32 SGPR`; the row-task body reached 242-256 VGPR before final suppression and remained spill-free. Q5 padding and universal swizzle8 were rejected, so this record keeps the small-route geometry and row-task mechanism separate.

## Correctness and resources

Validation covers Q5 low/high payload fields, scale/minimum state, inactive experts, repeated and sparse IDs, malformed offsets, non-aligned tails, input/weight mutations, independent BF16 references, finite outputs, and deterministic reruns. Retained kernels are wave32, zero-private, zero-spill, scratch-free, call-free, and stack-free.

## Evidence

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped-fwd-prior-v1.json
~/tmp/torch-ggml-ops/grouped-fwd-candidate-resources.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
```

Further Q5_K gains require a lower-state high-bit decoder or a prepared lossless representation.

The final Q5 record is anchored by the shared campaign baseline, resource screen, and production finalist correctness artifact:

```text
~/tmp/torch-ggml-ops/grouped-fwd-prior-v1.json
~/tmp/torch-ggml-ops/grouped-fwd-candidate-resources.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
~/tmp/torch-ggml-ops/grouped-fwd-production-final-resources.json
```
