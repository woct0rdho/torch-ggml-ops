# GGTensile Grouped MMQ Forward Pair IQ2_XXS Experiment

## Scope

This record covers the routed gfx1151 paired IQ2_XXS gate and up projections for the DeepSeek expert bank, computed in one workgroup dataflow:

```text
X_g[M_g,K] @ W_gate_g[K,N] -> Y_gate_g[M_g,N]
X_g[M_g,K] @ W_up_g[K,N]   -> Y_up_g[M_g,N]
```

## Final Results

`TFLOPS = 4*R*N*K / (median_ms * 1e9)` (both projections), and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Each row is the median over the benchmark's sampled expert partitions, and both implementations are timed on the same partitions.

| Batch | Logical shape `(R,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `(12288,2048,4096)` | 12.915 | 1.0540x | `grouped_mmq_fwd_pair_iq2_xxs_r12288_n2048_k4096_3d464d989430091a` | `grouped_fwd_serial_iq2_xxs_n2048_k4096_j64` |
| 4 | `(49152,2048,4096)` | 20.070 | 1.0583x | `grouped_mmq_fwd_pair_iq2_xxs_r49152_n2048_k4096_1906eee569a25cf5` | `grouped_fwd_serial_iq2_xxs_n2048_k4096_j64` |
| 16 | `(196608,2048,4096)` | 22.870 | 1.0109x | `grouped_mmq_fwd_pair_iq2_xxs_r196608_n2048_k4096_8468144fc20a1e49` | `grouped_fwd_serial_iq2_xxs_n2048_k4096_j80` |

GGTensile is ahead on all 3 rows, with speedups from `1.0109x` to `1.0583x` (mean `1.0411x`).

These medians replace the earlier recorded values, which were paired against HIP bodies that have since been retuned.

## Accepted Experiments

### Dedicated IQ2_XXS decoder and K32 arithmetic

The decoder keeps IQ2_XXS distinct from IQ2_S and reconstructs the two authoritative grid entries per group, expands each seven-bit sign group with odd parity, forms `d * (2*scale+1) / 8` in FP32, and writes BF16-RNE decoded weights to LDS. Three corrections were required. A scale staging write to the wrong LDS address produced non-finite output. The inherited IQ2_S BF16 store shift of 10 implies a 1,024-byte row stride and caused intermittent zero tiles at N=2048. The required shift is 12. Finally, IQ2_XXS scales cover K32, so both K16 integer WMMA fragments are accumulated before one FP32 correction per K32 scale. The corrected R35 body is bitwise identical to the installed J64, J80, serial, and fused paired controls.

### X1 paired sign-selector fusion

The decoder used the older `0x204081`/`0x01010101`/`v_lshl_or_b32` selector sequence. The fused form materializes `0x03020100` once in a plan-owned SGPR and replaces each three-instruction construction with the algebraically equivalent `0x810204` multiply plus `v_and_or_b32`, removing 64 VALU operations per body (`2,216` to `2,152` VALU issues). It was initially rejected under a fixed two-percent advancement gate, then reopened under the stability policy: a direct same-session 25-repeat A/B measured complete-call gains of `1.93%`/`1.67%`/`1.73%` and prequantized body gains of `1.97%`/`1.62%`/`1.91%`, with every medoid faster and conservative intervals excluding parity on fourteen of fifteen complete-call and fourteen of fifteen body medoids. The typed source reproduced the qualified probe instruction-for-instruction and is retained as `iq2_xxs_k128_interleaved_fused_selector()`.

### X5 J80 geometry at B16

The J80 plan increases the activation tile from 64 to 80 rows, the activation image from 9,216 to 11,520 bytes, total fixed LDS to 21,760 bytes, activation payload from 16 to 20 VGPRs, and M fragments from four to five, reaching 180 VGPRs, 44 SGPRs, 160 static WMMAs, and eight barriers. All four row counts were byte-deterministic and passed the exact route, malformed-route, reference, rerun, and both-projection mutation gates.

The final 25-repeat typed A/B rejected J80 at B1 by `+8.97%` to `+10.93%` weighted complete latency and at B4 as profile-dependent. At B16, every complete-call and body interval was below parity: J80 measured `294.6356 ms` versus `301.8988 ms` for X1 J64 (`1.02465x` complete, `1.02700x` body) and `1.05366x` against the installed serial J80 control. Only the exact `R=196608` B16 key is retained.

## Rejected Experiments

### X2 paired zero-accumulator lifetime

Initializing the dedicated `v124:v131` bank once before the K4096 block loop removed 24 statically emitted VALU issues while retaining 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, and zero spills. The fitted complete-call medians moved `+1.09%`, `+0.99%`, and `+0.28%` at B1/B4/B16, and one B16 profile fell below parity. X2 is rejected as neutral-to-regressive.

### X3 pre-scaled d

Moving the exact `* 0.125` factor into the four static FP32 `d` conversions removed eight local multiplies and added four invariant multiplies. The artifact changed from `2,216` to `2,212` VALU issues and stayed exact, but fitted complete-call movement was `+0.24%`, `-0.42%`, and `-0.52%` at B1/B4/B16. No point cleared the advancement gate, so X3 is rejected as timing-neutral.

### X4 epilogue probes

The shared-column setup probe removed six artifacts VALU issues. The broader materialized-address form removed 25. The exact width-two BF16 RNE interleave preserved instruction and resource counts. All three passed the exact route and mutation matrix, but their complete-call and body movements were mixed or contradictory, so none is retained.

### X6 processor-mode metadata and X1+X3 composition

Applying the processor-mode spelling produced byte-identical objects and HSACOs at every row count, so it is assembler-accepted but executable-inert and no timing or typed identity follows. Rebuilding X3 on top of the retained X1 parent removed four VALU issues and remained exact, but a 25-repeat top-up measured complete ratios within `1.00166x` and body movement approximately flat with all intervals crossing parity. The composition is closed as timing-neutral.

## Open Items

B1 separates the route laws in the direct-kernel benchmark, so its tuning should be re-evaluated separately under each law. X2-X4 are not reopened by that split.

## Closure

The retained result is the K128-interleaved fused-selector J64 body for B1/B4 and the J80 body for B16.
