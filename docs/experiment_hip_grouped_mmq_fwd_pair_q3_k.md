# HIP Grouped MMQ Forward Pair Q3_K Experiment

## Scope

This record covers the fused routed Q3_K gate/up forward kernel on gfx1151.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | --- | --- | --- | --- |
| 1 | `2 x (16384,512,2048)` | 14.01 | 2.080x | `grouped_fwd_row_task_q3_k_n512_k2048_j64` |
| 4 | `2 x (65536,512,2048)` | 18.92 | 1.657x | `grouped_fwd_row_task_q3_k_n512_k2048_j64` |
| 16 | `2 x (262144,512,2048)` | 19.51 | 1.004x | `grouped_fwd_row_task_q3_k_n512_k2048_j64` |

The deployed row-task body trails its uniform-route control by `36%` at B1 and `13%` at B4, and is level at B16 (`1%`). The earlier B1 screen preferred serial ownership on other route distributions, so the B1/B4 rows need a learned-route serial-versus-row-task re-screen. Flagged prior-sensitive at B1/B4.

The Q3_K pair code object was not changed by the IQ2_S/IQ2_XXS sign-table work; these numbers come from the same run as the sibling pair records, and the `1-2%` movement against earlier runs of the identical artifact is benchmark drift between sessions. The table kernel is the deployed HIP body for these shapes and rebuilds byte-identically from the current sources; the retained choice was measured on other route distributions, so a learned-route candidate sweep is the follow-up for the flagged batches.

## Kernel implementation

The retained pair body uses 128 threads, four wave32 waves, exact Q3_K N/K geometry, cooperative width-16 payload decode, separate decoded-weight LDS tiles, and one activation workspace shared by both projections. The pair does not replace packed decode with a dense weight.

Activation data is staged once for the pair, while packed weight tiles are decoded for each projection. Partial and nonuniform routed groups use masked row accesses without host-side route inspection.

## Address arithmetic sweep

A disassembly sweep of every installed control scores integer multiplies and 64-bit address add pairs, and this body is one of only two deployed controls that carries forty or more integer multiplies: `40` of its `3335` instructions, about one percent. It is not an addressing dependency chain of the kind the Q6_K forward fix removed (`1.27-1.30x` for 128 multiplies per unrolled loop body), so no hoisting is warranted here.

## Optimization log

### Initial grouped redesign

The historical grouped baseline used eight waves, narrow N16/K16 ownership, scalar decode, and serial row work. The first four-wave tiled pair introduced exact `(N,K)=(512,2048)` ownership, cooperative decode, separate weight LDS images, pair accumulation, and one final BF16 store path per output. Representative B4/B16 points improved by roughly 8-14x over the generic baseline and beat AITER by about 2.1-2.9x in the early controls.

The small-row M64 body and the larger M128 body were both measured. Universal M128 ownership was rejected because uniform 64-row groups would become half-empty bounded tiles; serial and row-task ownership were therefore evaluated as separate kernel variants.

### Ownership retune

A learned-route B1 screen compared the existing serial body with the existing 64-row device-task body. Q3_K row tasks improved the fitted prior by `1.1074x` but reached only `1.0179x` on captured routes, so B1 retains serial ownership. The existing row-task J64 body remains the measured large-route choice at B4/B16. The screen changed ownership only and added no route readback or new decode mechanism.

The coefficient-only campaign retained exact Q3_K pair geometry and rejected alternate J choices. Width-16 decode, padded Q3 LDS rows, bounded decode state, and separate pair/down layouts remain the measured kernel foundation.

### Closed mechanisms

Width-8 decode duplicated metadata work and lost to width16. Larger N ownership increased pair accumulator pressure. Broad M256/N64, universal inactive-M suppression, generic swizzles, two-LDS decoded-weight caches, split-K, persistent workgroups, and compiler-managed prefetch arrays did not provide a valid timing/resource improvement under this packed contract.

Re-checked against the learned route with the existing bodies: the device row-task body wins at every batch (`0.966x`, `0.848x`, `0.915x` of the serial body's time), so the deployed ownership is confirmed. Its payload decode is the largest decode item (`13.2%` of the kernel when ablated) but has no cheap reformulation: the bitfield extraction is already two shifts, two masks and one saturating subtract per value group. Still open: a fused two-projection body that stages the activation tile once, and the grid-lookup latency shared with the IQ2_S family.

## Evidence

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/grouped_fwd_current/pass12_pair_qwen_learned.json
```

The original campaign evidence is:

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
