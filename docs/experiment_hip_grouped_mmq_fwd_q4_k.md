# HIP Grouped MMQ Forward Q4_K Experiment

## Scope

This record covers the routed Q4_K down kernel for Qwen on gfx1151:

```text
Y[R,2048] = X[R,512] @ W[2048,512].T
```

The packed Q4_K weight is decoded cooperatively from GGUF storage. Inputs and outputs are BF16, and routed rows may be inactive, skewed, or not aligned to the tile size.

## Final kernel result

`HIP TFLOPS` is `2*R*N*K/time`. `HIP/AITER GMM` is the packed HIP throughput ratio against the exact BF16 AITER GMM baseline. Values above `1.00x` favor HIP.

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM |
| ---: | ---: | ---: | ---: |
| 1 | `(16384,2048,512)` | 21.60 | 1.805x |
| 4 | `(65536,2048,512)` | 24.04 | 1.054x |
| 16 | `(262144,2048,512)` | 24.16 | 0.955x |

The final matrix retains the current Q4_K packed decoder. The B16 comparison remains a representation gap against predecoded BF16 AITER weights.

## Kernel implementation

The retained body uses exact Q4_K N2048/K512 geometry, four wave32 waves, J64 ownership with a bounded J32 tail where measured, width-16 packed decode, Q4 scale/minimum reconstruction, and BF16 output stores. It keeps inactive route handling in the device kernel and does not build host descriptors from offsets.

The Q4_K decoder uses its own scale/minimum and LDS layout. It is separate from Q5_K high-bit reconstruction and from IQ2_S codebook decode.

## Optimization log

### Mixed J32 tails

A bounded J64/J32 tail body was promoted at aggregate rows `R=16384` and `R=65536`. The candidate passed exact packed-reference, independent-reference, inactive-route, non-aligned-tail, mutation, and resource controls. Weighted search/confirmation gains were `1.0591x/1.0828x` at B1 and `1.0243x/1.0279x` at B4. B16 retained the pure J64 tail after its confirmed movement was only `1.0124x` in the final campaign.

The pure J32 B4 typed probe was resource-clean but reached only `0.8970x` weighted prior speedup and `0.8466x` on the worst synthetic control, so it was rejected. The mixed tail is a bounded kernel mechanism, not a universal J32 replacement.

### Geometry campaign

The coefficient-only campaign screened exact J values, row-task alternatives, linked tail fields, decoder width, LDS layout, and bounded prefetch. Q4_K retained J64 with the measured J32 tails; broad row-task geometry and inactive-M policies were not promoted for this forward body. The retained Q4 path uses width-16 decode and a Q4-specific LDS arrangement.

The measured Q4 residual is repeated packed scale/minimum reconstruction rather than cache misses or high LDS stalls. Wider ownership, broad swizzles, two-LDS decoded-weight caches, split-K, Stream-K, persistent groups, and direct-to-LDS forms are closed.

The early Q4 redesign moved a representative B1 point from `6.874 ms` to `2.842 ms`; exact full-row and bounded-tail specialization later moved it to `1.599 ms`. The retained Q4 row-task body is `242 VGPR / 46 SGPR` with zero private bytes, spills, scratch, calls, and dynamic stack. These measurements explain the accepted tiled/tail mechanism without replacing the final matrix above.

## Correctness and resources

Retained Q4_K kernels are wave32, zero-private, zero-spill, scratch-free, call-free, and stack-free. Validation covers nibble and scale/min fields, inactive experts, malformed and reversed offsets, non-tile-aligned row tails, input/weight mutations, independent BF16 references, finite outputs, and paired route isolation where applicable.

## Evidence

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
