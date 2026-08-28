# HIP Grouped MMQ Forward IQ2_S Experiment

## Scope

This record covers the routed single-projection IQ2_S down kernel for Qwen on gfx1151:

```text
Y[R,2048] = X[R,512] @ W[2048,512].T
```

Inputs and outputs are BF16. The packed IQ2_S weight is decoded in the kernel, while routed row metadata remains device-resident. The exact aggregate rows are `R=16384,65536,262144`.

## Final kernel result

`HIP TFLOPS` uses `2*R*N*K/time`. `HIP/AITER GMM` compares the packed kernel with the exact BF16 AITER GMM baseline. Values above `1.00x` favor HIP.

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM |
| ---: | ---: | ---: | ---: |
| 1 | `(16384,2048,512)` | 15.32 | 1.279x |
| 4 | `(65536,2048,512)` | 19.80 | 0.862x |
| 16 | `(262144,2048,512)` | 20.44 | 0.801x |

The B4/B16 loss is attributed to repeated IQ2_S lookup, sign, scale, and packed-weight staging relative to predecoded BF16 weights.

## Kernel implementation

The retained single-down body uses exact `(N,K)=(2048,512)` geometry, J64 main ownership with a bounded J32 tail variant, cooperative width-16 IQ2_S decode, and a single-projection LDS layout. The kernel masks inactive or partial rows without moving route data to the host.

The paired IQ2_S kernel uses a different LDS swizzle and is not a valid replacement for this single-down body. The decoder reconstructs two grid entries, sign bytes, a shared scale nibble, and the `d` factor for aligned value groups.

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

## Correctness and resources

Validation covers codebook/sign/scale decoding, partial and inactive experts, malformed offsets, non-tile-aligned rows, input and packed-weight mutations, finite output, and independent BF16 references. Retained bodies are wave32, zero-private, zero-spill, scratch-free, call-free, and stack-free.

## Evidence

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/fwd-dispatch-screen/qwen_iq2s_j32_b4_9.json
~/tmp/torch-ggml-ops/fwd-dispatch-screen/qwen_iq2s_j32_b4_25.json
~/tmp/torch-ggml-ops/fwd-dispatch-screen/qwen_iq2s_pure_j32_b4_5.json
~/tmp/torch-ggml-ops/profile-grouped-fwd-20260812/qwen_iq2s_down_b16_packed/trace_results.db
```

The remaining improvement requires packed IQ2_S representation or explicit decode reuse. Another generic J or tail sweep is closed.
