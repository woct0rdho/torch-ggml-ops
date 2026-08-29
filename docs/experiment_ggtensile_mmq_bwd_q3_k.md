# GGTensile MMQ Backward Q3_K Kernel Experiments

## Scope

These experiments cover the ordinary Q3_K MMQ backward kernel on gfx1151, wave32, and WMMA V1. The kernel consumes BF16 grad-output and packed Q3_K weights, accumulates in FP32, and stores BF16 grad-input. Q3_K uses 256-value blocks and 110 packed bytes per block. The direct packed-weight kernel and its exact matrix shape are the unit of identity; prepared weights, external decode storage, grouped ownership, split-K, and persistent workgroups are not Q3 kernel variants in this record.

The exact matrices are:

```text
(2048, 2048,  512)   (8192, 2048,  512)   (32768, 2048,  512)
(2048, 2048, 8192)   (8192, 2048, 8192)   (32768, 2048, 8192)
```

The K512 matrices are the narrow family. The K8192 matrices are the query family. Both use `N=2048`; the different reduction lengths require separate ownership and timing decisions.

## Final Kernels

The table reports the fastest retained kernel found for each exact matrix across the completed experiments. The speedup is `HIP time / GGTensile time`; values above `1.0x` favor GGTensile. The final confirmations were consistent, so no A/B timing columns are included.

| Matrix shape `(M,N,K)` | Kernel hash | Speed (TFLOPS) | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(2048,2048,512)` | `ggsol_47d5421dfdac5c2d` | 26.877 | 1.1383x |
| `(8192,2048,512)` | `ggsol_74ab47fc6de41c02` | 29.156 | 1.2727x |
| `(32768,2048,512)` | `ggsol_66ffd967978c0ef1` | 30.049 | 1.2393x |
| `(2048,2048,8192)` | `ggsol_3f90cc6f487aad42` | 24.840 | 1.2667x |
| `(8192,2048,8192)` | `ggsol_ccb5384f18e397cd` | 26.703 | 1.2472x |
| `(32768,2048,8192)` | `ggsol_0e05cc1a23ff2d3e` | 27.162 | 1.2062x |

The six-key call-weighted speedup is `1.2184x`. The selected kernels use the same direct packed Q3_K contract and remain faster than HIP on every exact matrix.

## Final Validation And Resources

All six final kernels passed:
- bit-exact comparison with HIP;
- independent Q3 dequantized BF16 reference comparison;
- complete grad-output mutation;
- packed-weight mutation;
- one-hot packed-bit, signed-boundary, scale-group, and metadata-edge fixtures;
- reduced-trip checks at K32, K64, K96, and K512 where the geometry permits; and
- independent deterministic source and code-object rebuilds.

The final artifacts have zero private bytes, spills, scratch instructions, calls, and dynamic stack. They use 16 SGPRs and the following resource classes:

| Kernel cohort | Geometry | VGPRs | LDS |
| --- | --- | ---: | ---: |
| Narrow M2048 | padded `256x64x32`, WGM2 | 243 | 5 KiB |
| Narrow M8192/M32768 | padded `256x64x32`, WGM1 | 243 | 5 KiB |
| Query M2048/M8192 | padded `128x64x32`, WGM1 | 143 | 5 KiB |
| Query M32768 | padded `128x128x32`, WGM1 | 218 | 10 KiB |

The Q3 decoder owns the 32-byte high-mask plane, 64-byte low 2-bit payload plane, scale metadata, FP16 `d`, signed 3-bit reconstruction, BF16 conversion, and decoded-B LDS stores. The shared body owns activation addressing, LDS reads, WMMA, accumulation, and BF16 output stores only where the decoded tile contract matches.

## Profile Results

The representative final diagnostics separate complete execution into a WMMA/A/LDS floor and a decode/LDS floor. The floors are serial lower bounds used to explain overlap; they are not alternative kernels.

| Cohort and shape | Complete | WMMA/A/LDS floor | Decode/LDS floor | Floor sum |
| --- | ---: | ---: | ---: | ---: |
| Narrow `(32768,2048,512)` | 2.618 ms | 1.621 ms | 1.166 ms | 106.4% |
| Query `(32768,2048,8192)` | 46.780 ms | 26.666 ms | 21.019 ms | 101.9% |

The short-K path is sensitive primarily to decoded-B layout, decoder rows, and launch/overlap behavior. The long query path is an already overlapped WMMA/decode/LDS pipeline. The profile results do not show a first-order uncovered component that would justify reopening a schedule-only or resource-growing variant.

## Accepted Experiments

### Padded LDS and compact ownership

The original `128x128x32` one-buffer body was correct but left the short-K path behind the HIP layout. The accepted layout is unswizzled decoded-B LDS with `LdsPadB=8`. Reducing the narrow body to `256x64x32` lowered decoder rows, LDS traffic, and accumulator pressure without changing the packed representation.

The accepted exact ownership is shape-specific:
- `256x64x32`, WGM2 for narrow M2048;
- `256x64x32`, WGM1 for narrow M8192 and M32768;
- `128x64x32`, WGM1 for query M2048 and M8192; and
- `128x128x32`, WGM1 for query M32768.

The four-M-tile emitter required separate A-row, LDS, quant-shift, and Q3 low-shift state. That separation corrected an early `256x64` aliasing failure before production timing.

### Decode and instruction selection

Packed extraction is retained for every exact matrix. The retained decoder uses fused signed-scale reconstruction, decode-shift hoisting, Q3-specific legal VOPD pairing where the geometry permits it, and `v_lshl_or_b32` to merge the high mask with the low payload. Decode pairing is `Full` for five final keys; the narrow M8192 key uses the qualified `Partial` pairing.

All final kernels use `DepthU=32`, `PrefetchGlobalRead=2`, `PrefetchLocalRead=1`, activation prefetch, one decoded-B LDS buffer, packed per-lane payload loading, and the selected interleaved WMMA-wait schedule. Store priority remains raised. These controls are retained only as part of complete, exact shape-specific kernels.

## Rejected Experiments

The following alternatives were built or screened and did not replace the final kernels:

| Experiment | Evidence and disposition |
| --- | --- |
| Unpadded decoded-B LDS | The later `LdsPadB=8` body improved the short-K layouts and superseded the unpadded control. The unpadded `128x128` body remained slower on the relevant exact keys. |
| Alternate LDS swizzles and padding layouts | Correctness or timing gates failed; the final rows use unswizzled pad8. |
| Two-buffer versus one-buffer decoded-B pipeline | The earlier two-buffer control did not expose a stable gain over the accepted padded one-buffer ownership. The lower-bound profiles show that the long path already overlaps decode and WMMA/LDS work. |
| WGM4 and WGM8 | Larger traversal ownership did not produce a stable gain and was removed. WGM2 remains only for narrow M2048. |
| PGR1 and alternate SIA schedules | Reduced prefetch or alternate scheduling was neutral or regressing against the selected PGR2/SIA5 body. |
| Scalar Q3 extraction | It increased decoder work without a qualifying timing benefit and was rejected. |
| Metadata vector loads | Earlier correctness and timing gates did not show a gain over the format-aware scalar metadata path. |
| Store-priority alternatives | The no-priority and related alternatives were neutral or regressing; raised priority remains in the final kernels. |
| Broad geometry and ownership changes | Larger tiles, smaller ownership, and extra-wave arrangements either duplicated decode/A work, increased residency pressure, or lost to the selected shape-specific body. |

The final follow-up reviewed the accepted padded `128x128`, padded `128x64`, and padded `256x64` bodies; WGM1/2/4/8; SIA4/SIA5; store priority; four-M-tile address state; reduced-K behavior; decoder extraction; LDS layouts; and the two-buffer control. No remaining direct Q3_K kernel experiment has a measured path to a stable gain above the resource-bearing threshold. Historical arithmetic, ABI, unsupported-resource, and non-direct execution mechanisms remain outside this kernel log.
