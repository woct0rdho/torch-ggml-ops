# HIP Grouped MMQ Forward Pair Q3_K Experiment

## Scope

This record covers the fused routed Q3_K gate/up forward kernel on gfx1151. For each route group it computes two packed projections with logical weight shape `(N,K)=(512,2048)`:

```text
Y0[R,512] = X[R,2048] @ W0[512,2048].T
Y1[R,512] = X[R,2048] @ W1[512,2048].T
```

The activation is BF16 and the two outputs are BF16. Packed Q3_K weights remain authoritative; the kernel consumes device-resident route indices and offsets and handles inactive experts and partial row tiles.

## Final kernel result

Pair throughput counts both matrices: `4*R*N*K/time`. `HIP/AITER GMM` is the packed HIP throughput ratio against two BF16 AITER GMM calls. Values above `1.00x` favor HIP.

| Batch | Logical shape | HIP TFLOPS | HIP/AITER GMM |
| ---: | --- | ---: | ---: |
| 1 | `2 x (16384,512,2048)` | 18.04 | 2.212x |
| 4 | `2 x (65536,512,2048)` | 19.14 | 1.501x |
| 16 | `2 x (262144,512,2048)` | 19.24 | 1.042x |

These are the current uniform-route complete packed-path results.

## Kernel implementation

The retained pair body uses 128 threads, four wave32 waves, exact Q3_K N/K geometry, cooperative width-16 payload decode, separate decoded-weight LDS tiles, and one activation workspace shared by both projections. Each projection keeps its own accumulation and output state; the pair does not replace packed decode with a dense weight.

Q3_K scale and high-mask reconstruction are performed cooperatively. Activation data is staged once for the pair, while packed weight tiles are decoded for each projection. Partial and nonuniform routed groups use masked row accesses without host-side route inspection.

## Optimization log

### Initial grouped redesign

The historical grouped baseline used eight waves, narrow N16/K16 ownership, scalar decode, and serial row work. The first four-wave tiled pair introduced exact `(N,K)=(512,2048)` ownership, cooperative decode, separate weight LDS images, pair accumulation, and one final BF16 store path per output. Representative B4/B16 points improved by roughly 8-14x over the generic baseline and beat AITER by about 2.1-2.9x in the early controls.

The small-row M64 body and the larger M128 body were both measured. Universal M128 ownership was rejected because uniform 64-row groups would become half-empty bounded tiles; serial and row-task ownership were therefore evaluated as separate kernel variants.

### Ownership retune

A learned-route B1 screen compared the existing serial body with the existing 64-row device-task body. Q3_K row tasks improved the fitted prior by `1.1074x` but reached only `1.0179x` on captured routes, so B1 retains serial ownership. The existing row-task J64 body remains the measured large-route choice at B4/B16. The screen changed ownership only and added no route readback or new decode mechanism.

The coefficient-only campaign retained exact Q3_K pair geometry and rejected alternate J choices. Width-16 decode, padded Q3 LDS rows, bounded decode state, and separate pair/down layouts remain the measured kernel foundation.

### Closed mechanisms

Width-8 decode duplicated metadata work and lost to width16. Larger N ownership increased pair accumulator pressure. Broad M256/N64, universal inactive-M suppression, generic swizzles, two-LDS decoded-weight caches, split-K, persistent workgroups, and compiler-managed prefetch arrays did not provide a valid timing/resource improvement under this packed contract.

## Correctness and resources

The pair kernel preserves independent projection outputs, route isolation, inactive-expert inertness, malformed-route sentinels, non-aligned row tails, input mutation, and independent packed-weight mutation for each projection. The retained artifacts are wave32, zero-private, zero-spill, scratch-free, call-free, and stack-free.

The early grouped redesign moved representative Q3 pair B4/B16 points by roughly `8-14x` over the generic body. Compile-time J64 and exact N/K specialization removed the original row-decomposition cost; a later exact full-row/bounded-tail split moved a representative Q3 point from `6.170 ms` to `3.735 ms`. These historical points explain the retained ownership and are not substitutes for the final table above.

Pair correctness must not be judged by two separately rounded single-projection outputs: the packed pair's accumulation and output ownership are tested as one fused kernel contract.

## Evidence

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped-fwd-prior-v1.json
~/tmp/torch-ggml-ops/hip-b1-rowtask-screen-9.json
~/tmp/torch-ggml-ops/hip-b1-rowtask-confirm-25.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
```

The Q3_K pair is complete for the current packed decoder and ownership mechanisms. Future work must target Q3 payload/scale representation or a measured reuse mechanism, not another unqualified ownership sweep.

The retained source-of-record set also includes the campaign baseline and resource manifest used before the final route replay:

```text
~/tmp/torch-ggml-ops/grouped-fwd-prior-v1.json
~/tmp/torch-ggml-ops/grouped-fwd-candidate-resources.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
~/tmp/torch-ggml-ops/grouped-fwd-production-final-rebuild.json
~/tmp/torch-ggml-ops/grouped-fwd-production-final-resources.json
```
