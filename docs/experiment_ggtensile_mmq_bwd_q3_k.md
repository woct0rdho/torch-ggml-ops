# GGTensile MMQ Backward Q3_K Kernel Experiments

## Scope

This record covers the ordinary dense Q3_K backward kernels on gfx1151.

## Final Results

`TFLOPS = 2*M*N*K / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Medians are the repository benchmark's, re-measured in the current clock state. They replace the earlier recorded values, which came from a different clock state with the same artifacts.

| Family | `(M,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| --- | --- | ---: | ---: | --- | --- |
| Narrow key | `(2048,512,2048)` | 27.455 | 0.9843x | `mmq_bwd_q3_k_m2048_n512_k2048_416bdbcce317b9e4` | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| Narrow key | `(8192,512,2048)` | 29.084 | 1.0055x | `mmq_bwd_q3_k_m8192_n512_k2048_23acc09a8f37cfc2` | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| Narrow key | `(32768,512,2048)` | 31.727 | 1.0602x | `mmq_bwd_q3_k_m32768_n512_k2048_ac3c04eb79feab9d` | `dense_bwd_q3_k_pipea_nt64_ki64_mw4_sw16_prefetch` |
| Query/query gate | `(2048,8192,2048)` | 25.581 | 0.9556x | `mmq_bwd_q3_k_m2048_n8192_k2048_cc0b11d9efb64bba` | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_pad8_prefetch` |
| Query/query gate | `(8192,8192,2048)` | 27.504 | 0.9985x | `mmq_bwd_q3_k_m8192_n8192_k2048_6dde5ee7b45b02d1` | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_sw16_prefetch` |
| Query/query gate | `(32768,8192,2048)` | 26.384 | 1.0856x | `mmq_bwd_q3_k_m32768_n8192_k2048_01e43b4b6eed6030` | `dense_bwd_q3_k_pipea_nt64_ki64_mw2_sw16_prefetch` |

GGTensile is ahead on 3 of the 6 entries, with speedups from `0.956x` to `1.086x` (mean `1.015x`).

The earlier recordings of this family were paired against HIP backward bodies that have since been retiled and repipelined, so these speedups are lower than the recorded ones even though GGTensile's own throughput is unchanged.

## Profile Results

The representative final diagnostics separate complete execution into a WMMA/A/LDS floor and a decode/LDS floor. The floors are serial lower bounds used to explain overlap. They are not alternative kernels.

| Cohort and shape | Complete | WMMA/A/LDS floor | Decode/LDS floor | Floor sum |
| --- | ---: | ---: | ---: | ---: |
| Narrow `(32768,512,2048)` | 2.618 ms | 1.621 ms | 1.166 ms | 106.4% |
| Query `(32768,8192,2048)` | 46.780 ms | 26.666 ms | 21.019 ms | 101.9% |

The short-K path is sensitive primarily to decoded-B layout, decoder rows, and launch/overlap behavior. The long query path is an already overlapped WMMA/decode/LDS pipeline. The profile results do not show a first-order uncovered component that would justify reopening a schedule-only or resource-growing variant.

## Accepted Experiments

### Padded LDS and compact ownership

The original `128x128x32` one-buffer body was correct but left the short-K path behind the HIP layout. The accepted layout is unswizzled decoded-B LDS with `LdsPadB=8`. Reducing the narrow body to `256x64x32` lowered decoder rows, LDS traffic, and accumulator pressure without changing the packed representation.

The accepted exact ownership is shape-specific:
- `256x64x32`, WGM2 for narrow M2048.
- `256x64x32`, WGM1 for narrow M8192 and M32768.
- `128x64x32`, WGM1 for query M2048 and M8192.
- `128x128x32`, WGM1 for query M32768.

The four-M-tile emitter required separate A-row, LDS, quant-shift, and Q3 low-shift state. That separation corrected an early `256x64` aliasing failure before production timing.

### Decode and instruction selection

Packed extraction is retained for every exact matrix. The retained decoder uses fused signed-scale reconstruction, decode-shift hoisting, Q3-specific legal VOPD pairing where the geometry permits it, and `v_lshl_or_b32` to merge the high mask with the low payload. Decode pairing is `Full` for five final keys. The narrow M8192 key uses the qualified `Partial` pairing.

All final kernels use `DepthU=32`, `PrefetchGlobalRead=2`, `PrefetchLocalRead=1`, activation prefetch, one decoded-B LDS buffer, packed per-lane payload loading, and the selected interleaved WMMA-wait schedule. Store priority remains raised. These controls are retained only as part of complete, exact shape-specific kernels.

## Rejected Experiments

The following alternatives were built or screened and did not replace the final kernels:

| Experiment | Evidence and disposition |
| --- | --- |
| Unpadded decoded-B LDS | The later `LdsPadB=8` body improved the short-K layouts and superseded the unpadded control. The unpadded `128x128` body remained slower on the relevant exact keys. |
| Alternate LDS swizzles and padding layouts | Correctness or timing gates failed. The final rows use unswizzled pad8. |
| Two-buffer versus one-buffer decoded-B pipeline | The earlier two-buffer control did not expose a stable gain over the accepted padded one-buffer ownership. The lower-bound profiles show that the long path already overlaps decode and WMMA/LDS work. |
| WGM4 and WGM8 | Larger traversal ownership did not produce a stable gain and was removed. WGM2 remains only for narrow M2048. |
| PGR1 and alternate SIA schedules | Reduced prefetch or alternate scheduling was neutral or regressing against the selected PGR2/SIA5 body. |
| Scalar Q3 extraction | It increased decoder work without a qualifying timing benefit and was rejected. |
| Metadata vector loads | Earlier correctness and timing gates did not show a gain over the format-aware scalar metadata path. |
| Store-priority alternatives | The no-priority and related alternatives were neutral or regressing. Raised priority remains in the final kernels. |
| Broad geometry and ownership changes | Larger tiles, smaller ownership, and extra-wave arrangements either duplicated decode/A work, increased residency pressure, or lost to the selected shape-specific body. |

The final follow-up reviewed the accepted padded `128x128`, padded `128x64`, and padded `256x64` bodies, WGM1/2/4/8, SIA4/SIA5, store priority, four-M-tile address state, reduced-K behavior, decoder extraction, LDS layouts, and the two-buffer control. No remaining direct Q3_K kernel experiment has a measured path to a stable gain above the resource-bearing threshold. Historical arithmetic, ABI, unsupported-resource, and non-direct execution mechanisms remain outside this kernel log.
