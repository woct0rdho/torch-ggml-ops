# GGTensile MMQ Backward Q5_K Kernel Experiments

## Scope

This record covers the ordinary dense Q5_K backward kernels on gfx1151:

```text
grad_input[M,N] = grad_output[M,K] @ dequant_q5_k(weight[K,N])
```

## Final Results

`TFLOPS = 2*M*K*N / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Medians are the repository benchmark's, re-measured in the current clock state. They replace the earlier recorded values, which came from a different clock state with the same artifacts.

| Family | `(M,K,N)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| --- | --- | ---: | ---: | --- | --- |
| Narrow K/V/gate/up | `(2048,2048,512)` | 28.557 | 1.0475x | `mmq_bwd_q5_k_m2048_n2048_k512_4b4c3b8c85dc07e8` | `dense_bwd_q5_k_pipea_nt4_ki64_mw2_pad8_prefetch` |
| Narrow K/V/gate/up | `(8192,2048,512)` | 29.232 | 1.0041x | `mmq_bwd_q5_k_m8192_n2048_k512_60125cd8f8d220c6` | `dense_bwd_q5_k_pipea_nt4_ki64_mw2_pad8_prefetch` |
| Narrow K/V/gate/up | `(32768,2048,512)` | 32.447 | 1.0912x | `mmq_bwd_q5_k_m32768_n2048_k512_455ad6f1b34fb108` | `dense_bwd_q5_k_pipea_nt4_ki64_mw2_pad8_prefetch` |
| Shared down | `(2048,512,2048)` | 21.178 | 1.2146x | `mmq_bwd_q5_k_m2048_n512_k2048_44b19e571fc6fb31` | `dense_bwd_q5_k_mt128_nt128_ki32_full_k512` |
| Shared down | `(8192,512,2048)` | 15.862 | 0.9517x | `mmq_bwd_q5_k_m8192_n512_k2048_d631cd9051ca2d0e` | `dense_bwd_q5_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw2_pad8` |
| Shared down | `(32768,512,2048)` | 19.741 | 1.0687x | `mmq_bwd_q5_k_m32768_n512_k2048_cf28142433ab61f5` | `dense_bwd_q5_k_pipea_nt4_ki64_mw2_pad8_prefetch` |

GGTensile is ahead on 5 of the 6 entries, with speedups from `0.952x` to `1.215x` (mean `1.063x`).

The earlier recordings of this family were paired against HIP backward bodies that have since been retiled and repipelined, so these speedups are lower than the recorded ones even though GGTensile's own throughput is unchanged.

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
- low-payload nibble-shift hoisting for five matrices.
- the original per-chunk nibble shift for shared-down `(8192,512,2048)`, where the isolated recheck was `1.06616x` candidate/control latency.
- `v_lshl_or_b32` high-bit fusion for all six matrices.
- per-lane packed payload loading with the Q5-specific high-bit path.

The nibble-shift hoist first measured `0.98519` weighted candidate/control latency and improved five exact matrices. The follow-up high-bit fusion reduced static VALU issues to 920 for the selected N128 path and 948 for the no-hoist shared-down M8192 path without changing the hard resource class.

### Narrow padded ownership

The initial four-wave `128x128x32` two-buffer control was replaced for the narrow family after the Q3 padded-LDS result supplied a changed layout premise. Unswizzled `LdsPadB=8` with `256x64x32` reduced decoder rows, LDS traffic, and accumulator pressure. The narrow M2048 kernel uses WGM2. M8192 and M32768 use WGM1. The changed geometry and layout improved the three narrow Q5 keys by approximately 15-18% against their earlier selected assemblies and remained exact and resource-clean.

### Shared-down two-buffer ownership

Shared-down retains the Q5-compatible `128x128x32`, WGM1, XOR8, two-decoded-B-buffer arrangement. The shared-down shapes have a different ownership balance from the narrow shapes, and the padded one-buffer alternatives were materially slower. M8192 keeps the original nibble-shift placement after its isolated hoist regression. M2048 and M32768 use the hoisted form. This shape-specific decode choice is part of the complete kernel identity.

## Rejected Experiments

The following kernel alternatives were built, corrected where necessary, and rejected by correctness, resources, or timing:

| Experiment | Evidence and disposition |
| --- | --- |
| One decoded-B buffer | Weighted candidate/control ratio `1.06448`. All six exact keys regressed or were neutral, with shared-down M8192 at `1.29020`. Rejected. |
| One-buffer WGM2, WGM4, and WGM8 | Weighted ratios `1.08477`, `1.14106`, and `1.23394`. Larger traversal ownership duplicated work or lost residency. Rejected. |
| One-buffer SIA5 plus store priority | Weighted ratio `1.06780`. The Q4 schedule interaction did not transfer to Q5. Rejected. |
| Packed lane sharing 2 | Weighted ratio `1.09449`. Cross-lane overhead outweighed payload-load savings. Rejected. |
| Scalar extraction | The initial two-buffer M2048 screen was `1.01395x` of the control. A later current-parent reopening, candidate `ggsol_5c64e62dbe925d6b`, was exact and resource-identical but moved `+0.29%` and `+1.40%` in two parent brackets. Rejected as timing-neutral to mildly regressive. |
| SIA5 plus store priority on the two-buffer control | Ratio `1.00057`. Correct but neutral. SIA4 remained the control. |
| Repaired SIA3/PGR1 | The historical illegal-memory-access path was fixed and passed HIP, independent-reference, grad-output, and packed-weight checks. It then measured `0.214825 ms` versus `0.153590 ms` for the selected M2048 control, or `1.39869x`. Rejected. |
| Repaired double-buffer DepthU64 | The metadata addresses, decoded-LDS reads, A-pointer restoration, and final handoff were fixed and passed all correctness gates. It measured `0.217225 ms` versus `0.152452 ms`, or `1.42487x`. Rejected. |
| Vector metadata load with dynamic scale-byte extraction | Correct with 12 fewer VMEM instructions and 16 more VALU issues, but ratio `1.00876`. Shared-down M8192 reached `1.05063`. Rejected. |
| Padded shared-down ownership changes | Padded `256x64`, `128x64`, and `128x128` one-buffer candidates were 7-46% slower than the selected two-buffer controls. Rejected. |
| Alternate padding and LDS layouts | Pad16/24, alternate swizzles, and related XOR layouts did not produce a qualifying exact-shape gain. Rejected. |
| Broad geometry and store schedules | WGM alternatives, store-priority changes, alternate SIA schedules, and larger or smaller tiles were neutral, regressing, or resource-inferior. Rejected. |

The repaired paths remained exact after mutation testing, but their large losses against the current parent made final-length confirmation unnecessary. The selected narrow and shared-down ownerships therefore remain separate rather than being merged into one universal Q5 body.

## Deferred Kernel Experiments

Relaxed BF16 conversion is not part of the exact kernels above. A future kernel-only experiment may test site-separated `RNEPreserveNaN`, `BiasRound`, and `Truncate` policies, starting with narrow `(32768,2048,512)` parent `ggsol_3ac1ebb6845e1cbd` and considering shared-down `(32768,512,2048)` parent `ggsol_d953a19ba8487f78` only after a resource-neutral first result removes at least 1% of body latency. The packed high-bit reconstruction, geometry, LDS, prefetch, store policy, and arithmetic order must remain fixed. `BiasRound` and `Truncate` require finite-input error distributions and model-training validation. They cannot enter the exact kernel set from isolated timing.

A Q5 metadata-prefetch or payload-width experiment is also deferred. The current backward schema and lowerer have no distinct Q5 producer/consumer or physical transaction plan for those controls, so a new field would be inert until a complete emitter, physical plan, resource accounting, and validation contract exists.
