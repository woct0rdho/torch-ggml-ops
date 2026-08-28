# HIP Fixed-Grouped MMQ Forward Q8_0 Experiment

## Scope

This record covers the fixed eight-group DeepSeek Q8_0 forward kernel:

```text
Y[t,8,1024] = X[t,8,4096] @ W[8,1024,4096].T
```

The eight groups are fixed rather than routed. Q8_0 weights are decoded directly from packed GGUF storage, inputs and outputs are BF16, and the kernel preserves the group-major logical weights and token-major output layout.

## Final kernel result

Fixed-group throughput counts all eight matrices: `2*8*M*N*K/time`. `HIP/torch.bmm` compares the packed fixed-group kernel with BF16 `torch.bmm` over the eight groups. Values above `1.00x` favor HIP.

| Batch | Logical shape | HIP TFLOPS | HIP/torch.bmm |
| ---: | --- | ---: | ---: |
| 1 | `8 x (2048,1024,4096)` | 12.57 | 0.750x |
| 4 | `8 x (8192,1024,4096)` | 12.57 | 0.727x |
| 16 | `8 x (32768,1024,4096)` | 12.58 | 0.716x |

The packed kernel is slower than BMM because the baseline starts from already decoded BF16 weights.

## Kernel implementation

The retained fixed-group body uses exact `(N,K)=(1024,4096)` per group, four wave32 waves, J64 ownership, width-16 Q8_0 decode, one unswizzled N64/K32 weight tile per group, and fixed group-Z ownership. It performs the group-major to token-major output conversion in the kernel path.

The Q8_0 decoder loads packed int8 payloads and FP16 scales, reconstructs WMMA operands, and writes BF16 output. It does not rely on fabricated route metadata.

## Optimization log

### Initial geometry

The generic fixed body used narrow eight-wave ownership, N16/K16 work, and serial row handling. The first exact four-wave M256/N64/K32 body improved the generic fixed kernel by `8.09x`, `8.33x`, and `8.75x` at B1/B4/B16.

The retained fixed body uses `213 VGPR / 48 SGPR / 4096 B LDS` and remains zero-private and spill-free. M64 and M128 were slower at the smaller fixed batches; width32, swizzle4, and M512 were rejected by timing or the resource warning boundary.

M64 and M128 were screened but M256 remained the better fixed-group geometry at B1/B4. Width32 decode, swizzle4, M512, broad scale staging, rolled dot loops, and group-major Q8_1 workspace alternatives were invalid, resource-heavy, or slower.

### Coefficient-only geometry

The full typed campaign retained fixed Q8_0 J64 full/bounded bodies and tested J16, J80, wider ownership, bounded tails, decoder width, and LDS layout. No alternate geometry passed all timing, correctness, and resource gates. The fixed path retains token rows rather than routed rows and has no inactive-expert work to suppress.

The residual is representation mismatch: Q8_0 packed payloads and scales must be reconstructed before WMMA, while BMM consumes a predecoded BF16 tensor. A lossless prepared payload/scale layout would be a separate representation experiment.

## Correctness and resources

Retained fixed Q8_0 kernels are wave32, zero-private, zero-spill, scratch-free, call-free, and stack-free. Validation covers all eight groups, token and group boundaries, independent GGUF decode, output-layout conversion, input and packed-weight mutations, group isolation, finite output, and deterministic reruns.

## Evidence

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_deepseek_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped-fwd-prior-v1.json
~/tmp/torch-ggml-ops/grouped-fwd-candidate-resources.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
```

The fixed Q8_0 kernel is locally complete for the current packed representation. Further work must reduce representation or decode cost rather than reopen generic J choices.

The fixed-group campaign provenance is:

```text
~/tmp/torch-ggml-ops/grouped-fwd-prior-v1.json
~/tmp/torch-ggml-ops/grouped-fwd-candidate-resources.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
~/tmp/torch-ggml-ops/grouped-fwd-production-final-rebuild.json
~/tmp/torch-ggml-ops/grouped-fwd-production-final-resources.json
```
