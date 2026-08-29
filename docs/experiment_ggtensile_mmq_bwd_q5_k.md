# GGTensile MMQ Backward Q5_K Kernel Experiments

## Scope

These experiments cover the ordinary Q5_K MMQ backward kernel on gfx1151, wave32, and WMMA V1. The kernel consumes BF16 grad-output and packed Q5_K weights, accumulates in FP32, and stores BF16 grad-input. Q5_K uses 256-value blocks and 176 packed bytes per block. The direct packed-weight kernel and its exact matrix shape are the unit of identity; prepared weights, external decode storage, grouped ownership, split-K, and persistent workgroups are not Q5 kernel variants in this record.

MMQ backward uses `M=rows`, `N=in_features`, and `K=out_features`:

```text
grad_input[M,N] = grad_output[M,K] @ dequant(weight[K,N])
```

The exact matrices are:

```text
(2048, 2048,  512)   (8192, 2048,  512)   (32768, 2048,  512)
(2048,  512, 2048)   (8192,  512, 2048)   (32768,  512, 2048)
```

The first three are the narrow family. The last three are the shared-down family. Q5_K has its own high-bit payload and decoder contract; Q4_K results are not transferred without Q5-specific validation.

## Final Kernels

The table reports the fastest final-qualified kernel found for each exact matrix across the completed experiments. The speedup is `HIP time / GGTensile time`; values above `1.0x` favor GGTensile. The final confirmations were consistent, so no A/B timing columns are included.

| Matrix shape `(M,N,K)` | Kernel hash | Speed (TFLOPS) | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(2048,2048,512)` | `ggsol_0581c5eaea62c710` | 28.037 | 1.4032x |
| `(8192,2048,512)` | `ggsol_2df9140465728c7b` | 29.387 | 1.4108x |
| `(32768,2048,512)` | `ggsol_3ac1ebb6845e1cbd` | 31.913 | 1.4406x |
| `(2048,512,2048)` | `ggsol_989c15a1f268e54c` | 20.679 | 1.2193x |
| `(8192,512,2048)` | `ggsol_4d8fcfa4c4f8ec9d` | 15.779 | 1.3437x |
| `(32768,512,2048)` | `ggsol_d953a19ba8487f78` | 19.794 | 1.6026x |

The six-key call-weighted speedup is `1.4740x`. Every exact matrix has a Q5-specific kernel that is faster than HIP under the final comparison protocol.

## Final Validation And Resources

All six final kernels passed:
- bit-exact comparison with HIP;
- independent Q5 dequantized BF16 reference comparison;
- complete grad-output mutation;
- packed-weight mutation;
- reduced-trip checks at K32, K64, K96, and K512 where the geometry permits; and
- independent deterministic source and code-object rebuilds.

The final artifacts have zero private bytes, spills, scratch instructions, calls, and dynamic stack. They use 16 SGPRs and the following resource classes:

| Kernel cohort | Geometry and layout | VGPRs | LDS |
| --- | --- | ---: | ---: |
| Narrow M2048 | padded `256x64x32`, WGM2, one buffer | 238 | 5 KiB |
| Narrow M8192/M32768 | padded `256x64x32`, WGM1, one buffer | 238 | 5 KiB |
| Shared-down M2048/M32768 | `128x128x32`, WGM1, XOR8, two buffers | 220 | 16 KiB |
| Shared-down M8192 | `128x128x32`, WGM1, XOR8, two buffers, inline nibble shift | 220 | 16 KiB |

The Q5 backend owns the 32-byte high-bit `qh` plane, 128-byte low 4-bit payload, scale/minimum metadata, FP16 `d` and `dmin`, signed reconstruction, BF16 conversion, and decoded-B LDS stores. The shared body owns activation addressing, LDS reads, WMMA, accumulation, and BF16 output stores only where the decoded tile contract matches.

## Profile Results

The representative lower-bound measurements separate complete execution into a WMMA/A/LDS floor and a decode/LDS floor. These are diagnostic serial bounds, not alternative kernels.

| Cohort and shape | Complete | WMMA/A/LDS floor | Decode/LDS floor | Floor sum |
| --- | ---: | ---: | ---: | ---: |
| Narrow `(32768,2048,512)` | 2.4545 ms | 1.5872 ms | 1.1003 ms | 109.5% |
| Shared-down `(32768,512,2048)` | 3.5218 ms | 2.4191 ms | 1.0844 ms | 99.5% |

The narrow body is an overlapped WMMA/decode/LDS pipeline. Shared-down is dominated by WMMA, activation traffic, and LDS work. The lower bounds and final resource measurements do not expose a first-order uncovered component for another exact schedule or resource-growing ownership variant.

## Accepted Experiments

### Q5 backend and packed decode

The Q5 backend was kept separate from Q4 while reusing the compatible WMMA and accumulation body. It emits the 176-byte block layout, low and high payload reads, six-bit scale/minimum extraction, signed high-bit reconstruction, BF16 conversion, and decoded-B LDS stores. Strict Q5 identities, reduced-trip fixtures, one-hot payload checks, scale-group checks, metadata-boundary checks, packed-weight mutation, and independent references passed before production timing.

Packed extraction is retained for all six matrices. The Q5-specific emitter choices that survived qualification are:
- low-payload nibble-shift hoisting for five matrices;
- the original per-chunk nibble shift for shared-down `(8192,512,2048)`, where the isolated recheck was `1.06616x` candidate/control latency;
- `v_lshl_or_b32` high-bit fusion for all six matrices; and
- per-lane packed payload loading with the Q5-specific high-bit path.

The nibble-shift hoist first measured `0.98519` weighted candidate/control latency and improved five exact matrices. The follow-up high-bit fusion reduced static VALU issues to 920 for the selected N128 path and 948 for the no-hoist shared-down M8192 path without changing the hard resource class.

### Narrow padded ownership

The initial four-wave `128x128x32` two-buffer control was replaced for the narrow family after the Q3 padded-LDS result supplied a changed layout premise. Unswizzled `LdsPadB=8` with `256x64x32` reduced decoder rows, LDS traffic, and accumulator pressure. The narrow M2048 kernel uses WGM2; M8192 and M32768 use WGM1. The changed geometry and layout improved the three narrow Q5 keys by approximately 15-18% against their earlier selected assemblies and remained exact and resource-clean.

### Shared-down two-buffer ownership

Shared-down retains the Q5-compatible `128x128x32`, WGM1, XOR8, two-decoded-B-buffer arrangement. The shared-down shapes have a different ownership balance from the narrow shapes, and the padded one-buffer alternatives were materially slower. M8192 keeps the original nibble-shift placement after its isolated hoist regression; M2048 and M32768 use the hoisted form. This shape-specific decode choice is part of the complete kernel identity.

## Rejected Experiments

The following kernel alternatives were built, corrected where necessary, and rejected by correctness, resources, or timing:

| Experiment | Evidence and disposition |
| --- | --- |
| One decoded-B buffer | Weighted candidate/control ratio `1.06448`; all six exact keys regressed or were neutral, with shared-down M8192 at `1.29020`. Rejected. |
| One-buffer WGM2, WGM4, and WGM8 | Weighted ratios `1.08477`, `1.14106`, and `1.23394`. Larger traversal ownership duplicated work or lost residency. Rejected. |
| One-buffer SIA5 plus store priority | Weighted ratio `1.06780`; the Q4 schedule interaction did not transfer to Q5. Rejected. |
| Packed lane sharing 2 | Weighted ratio `1.09449`; cross-lane overhead outweighed payload-load savings. Rejected. |
| Scalar extraction | The initial two-buffer M2048 screen was `1.01395x` of the control. A later current-parent reopening, candidate `ggsol_5c64e62dbe925d6b`, was exact and resource-identical but moved `+0.29%` and `+1.40%` in two parent brackets. Rejected as timing-neutral to mildly regressive. |
| SIA5 plus store priority on the two-buffer control | Ratio `1.00057`; correct but neutral. SIA4 remained the control. |
| Repaired SIA3/PGR1 | The historical illegal-memory-access path was fixed and passed HIP, independent-reference, grad-output, and packed-weight checks. It then measured `0.214825 ms` versus `0.153590 ms` for the selected M2048 control, or `1.39869x`. Rejected. |
| Repaired double-buffer DepthU64 | The metadata addresses, decoded-LDS reads, A-pointer restoration, and final handoff were fixed and passed all correctness gates. It measured `0.217225 ms` versus `0.152452 ms`, or `1.42487x`. Rejected. |
| Vector metadata load with dynamic scale-byte extraction | Correct with 12 fewer VMEM instructions and 16 more VALU issues, but ratio `1.00876`; shared-down M8192 reached `1.05063`. Rejected. |
| Padded shared-down ownership changes | Padded `256x64`, `128x64`, and `128x128` one-buffer candidates were 7-46% slower than the selected two-buffer controls. Rejected. |
| Alternate padding and LDS layouts | Pad16/24, alternate swizzles, and related XOR layouts did not produce a qualifying exact-shape gain. Rejected. |
| Broad geometry and store schedules | WGM alternatives, store-priority changes, alternate SIA schedules, and larger or smaller tiles were neutral, regressing, or resource-inferior. Rejected. |

The repaired paths remained exact after mutation testing, but their large losses against the current parent made final-length confirmation unnecessary. The selected narrow and shared-down ownerships therefore remain separate rather than being merged into one universal Q5 body.

## Deferred Kernel Experiments

Relaxed BF16 conversion is not part of the exact kernels above. A future kernel-only experiment may test site-separated `RNEPreserveNaN`, `BiasRound`, and `Truncate` policies, starting with narrow `(32768,2048,512)` parent `ggsol_3ac1ebb6845e1cbd` and considering shared-down `(32768,512,2048)` parent `ggsol_d953a19ba8487f78` only after a resource-neutral first result removes at least 1% of body latency. The packed high-bit reconstruction, geometry, LDS, prefetch, store policy, and arithmetic order must remain fixed. `BiasRound` and `Truncate` require finite-input error distributions and model-training validation; they cannot enter the exact kernel set from isolated timing.

A Q5 metadata-prefetch or payload-width experiment is also deferred. The current backward schema and lowerer have no distinct Q5 producer/consumer or physical transaction plan for those controls, so a new field would be inert until a complete emitter, physical plan, resource accounting, and validation contract exists.
