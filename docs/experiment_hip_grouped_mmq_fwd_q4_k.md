# HIP Grouped MMQ Forward Q4_K Experiment

## Scope

This record covers the routed Q4_K down kernel for Qwen on gfx1151.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | ---: | ---: | ---: | --- |
| 1 | `(16384,2048,512)` | 16.94 | 1.463x | `grouped_fwd_serial_q4_k_n2048_k512_j64` |
| 4 | `(65536,2048,512)` | 23.00 | 1.204x | `grouped_fwd_serial_q4_k_n2048_k512_j64` |
| 16 | `(262144,2048,512)` | 24.65 | 1.060x | `grouped_fwd_serial_q4_k_n2048_k512_j64` |

The retained J64 body trails its uniform-route control by `29%`/`13%` at B1/B4 and is level at B16 (`4%`). The B16 comparison remains a representation gap against predecoded BF16 weights. Flagged prior-sensitive at B1/B4: the retention decision was made on other route distributions and may need a learned-route retune.

The table kernel is the deployed HIP body for these shapes and rebuilds byte-identically from the current sources; the retained choice was measured on other route distributions, so a learned-route candidate sweep is the follow-up for the flagged batches.

## Kernel implementation

The retained body uses exact Q4_K N2048/K512 geometry, four wave32 waves, J64 ownership with a bounded J32 tail where measured, width-16 packed decode, Q4 scale/minimum reconstruction, and BF16 output stores. It keeps inactive route handling in the device kernel and does not build host descriptors from offsets.

## Optimization log

### Mixed J32 tails

A bounded J64/J32 tail body was promoted at aggregate rows `R=16384` and `R=65536`. Weighted search/confirmation gains were `1.0591x/1.0828x` at B1 and `1.0243x/1.0279x` at B4. B16 retained the pure J64 tail after its confirmed movement was only `1.0124x` in the final campaign.

The pure J32 B4 typed probe was resource-clean but reached only `0.8970x` weighted prior speedup and `0.8466x` on the worst synthetic control, so it was rejected. The mixed tail is a bounded kernel mechanism, not a universal J32 replacement.

### Geometry campaign

The coefficient-only campaign screened exact J values, row-task alternatives, linked tail fields, decoder width, LDS layout, and bounded prefetch. Q4_K retained J64 with the measured J32 tails; broad row-task geometry and inactive-M policies were not promoted for this forward body. The retained Q4 path uses width-16 decode and a Q4-specific LDS arrangement.

The measured Q4 residual is repeated packed scale/minimum reconstruction rather than cache misses or high LDS stalls. Wider ownership, broad swizzles, two-LDS decoded-weight caches, split-K, Stream-K, persistent groups, and direct-to-LDS forms are closed.

The early Q4 redesign moved a representative B1 point from `6.874 ms` to `2.842 ms`; exact full-row and bounded-tail specialization later moved it to `1.599 ms`. The retained Q4 row-task body is `242 VGPR / 46 SGPR`. These measurements explain the accepted tiled/tail mechanism without replacing the final matrix above.

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
