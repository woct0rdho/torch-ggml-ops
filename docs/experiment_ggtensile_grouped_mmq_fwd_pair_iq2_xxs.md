# GGTensile Grouped MMQ Forward Pair IQ2_XXS Experiment

## Scope And Contract

This record covers the isolated gfx1151 paired IQ2_XXS forward kernel for the DeepSeek routed gate and up projections. For each routed expert the kernel computes both projections in one workgroup dataflow:

```text
X_g[M_g,4096] @ W_gate_g[4096,2048] -> Y_gate_g[M_g,2048]
X_g[M_g,4096] @ W_up_g[4096,2048]   -> Y_up_g[M_g,2048]
```

The measured aggregate-row shapes are `R=12288`, `49152`, and `196608`, plus the `R=35` boundary key. Each packed bank is `[256,2048,1056]`, with sixteen 256-value blocks of 66 bytes per row. One shared Q8_1 `F32_D4` activation workspace with shape `[32,R,144]` and two independent BF16 destinations are used. IQ2_XXS carries one FP16 `d` followed by 64 bytes of four-byte grid indices and parity-expanded sign words, with one scale per K32; the 256-entry codebook is extracted at generation time from the llama.cpp grid.

The paired research ABI carries two packed weights, one activation workspace, two destinations, the device-resident route description, and the shape values. Logical paired throughput is `4 * R * 2048 * 4096 / (latency_ms * 1e9)`, and the speedup ratio is HIP time divided by GGTensile time.

## Final Benchmark Results

The table shows the fastest qualified kernel found for each measured aggregate-row shape.

| Matrix shape `(R,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(12288,2048,4096)` | `ggpair_3d464d989430091a` | `13.741` | `1.1594x` |
| `(49152,2048,4096)` | `ggpair_1906eee569a25cf5` | `20.329` | `1.1303x` |
| `(196608,2048,4096)` | `ggpair_8468144fc20a1e49` | `23.329` | `1.0578x` |

The B1 and B4 rows use the retained J64 fused-selector body; B16 uses the retained J80 body. All timed retained outputs were bitwise exact.

## Final Kernel Profiles

| Kernel hash | Geometry and ownership | Decode and schedule | VGPR / SGPR | LDS bytes | WMMAs / barriers |
| --- | --- | --- | ---: | ---: | ---: |
| `ggpair_3d464d989430091a` | J64/N64, serial routes, fused selector | K128 interleaved half-weight LDS, SIA4-era schedule | `148 / 44` | `19,456` | `128 / 8` |
| `ggpair_1906eee569a25cf5` | J64/N64, serial routes, fused selector | K128 interleaved half-weight LDS | `148 / 44` | `19,456` | `128 / 8` |
| `ggpair_8468144fc20a1e49` | J80/N64, serial routes, fused selector | K128 interleaved half-weight LDS | `180 / 44` | `21,760` | `160 / 8` |

All retained artifacts are gfx1151 code-object-v5 wave32 kernels with zero private storage, spills, scratch, calls, and dynamic stack.

## Accepted Kernel Experiments

### Dedicated IQ2_XXS decoder and K32 arithmetic

The decoder keeps IQ2_XXS distinct from IQ2_S and reconstructs the two authoritative grid entries per group, expands each seven-bit sign group with odd parity, forms `d * (2*scale+1) / 8` in FP32, and writes BF16-RNE decoded weights to LDS. Three corrections were required. A scale staging write to the wrong LDS address produced non-finite output. The inherited IQ2_S BF16 store shift of 10 implies a 1,024-byte row stride and caused intermittent zero tiles at N=2048; the required shift is 12. Finally, IQ2_XXS scales cover K32, so both K16 integer WMMA fragments are accumulated before one FP32 correction per K32 scale. The corrected R35 body is bitwise identical to the installed J64, J80, serial, and fused paired controls. Independent bounded references measured maximum absolute error `0.015625` at R35 and `0.03125` at production sizes, with RMSE between `0.000116` and `0.000120`.

### X1 paired sign-selector fusion

The decoder used the older `0x204081`/`0x01010101`/`v_lshl_or_b32` selector sequence. The fused form materializes `0x03020100` once in a plan-owned SGPR and replaces each three-instruction construction with the algebraically equivalent `0x810204` multiply plus `v_and_or_b32`, removing 64 VALU operations per body (`2,216` to `2,152` VALU issues). It was initially rejected under a fixed two-percent advancement gate, then reopened under the stability policy: a direct same-session 25-repeat A/B measured complete-call gains of `1.93%`/`1.67%`/`1.73%` and prequantized body gains of `1.97%`/`1.62%`/`1.91%`, with every medoid faster and conservative intervals excluding parity on fourteen of fifteen complete-call and fourteen of fifteen body medoids. The typed source reproduced the qualified probe instruction-for-instruction and is retained as `iq2_xxs_k128_interleaved_fused_selector()`.

### X5 J80 geometry at B16

The J80 plan increases the activation tile from 64 to 80 rows, the activation image from 9,216 to 11,520 bytes, total fixed LDS to 21,760 bytes, activation payload from 16 to 20 VGPRs, and M fragments from four to five, reaching 180 VGPRs, 44 SGPRs, 160 static WMMAs, and eight barriers. All four row counts were byte-deterministic and passed the exact route, malformed-route, reference, rerun, and both-projection mutation gates.

The final 25-repeat typed A/B rejected J80 at B1 by `+8.97%` to `+10.93%` weighted complete latency and at B4 as profile-dependent. At B16, every complete-call and body interval was below parity: J80 measured `294.6356 ms` versus `301.8988 ms` for X1 J64 (`1.02465x` complete, `1.02700x` body) and `1.05366x` against the installed serial J80 control. Only the exact `R=196608` B16 key is retained.

## Rejected Kernel Experiments

### X2 paired zero-accumulator lifetime

Initializing the dedicated `v124:v131` bank once before the K4096 block loop removed 24 statically emitted VALU issues while retaining 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, and zero spills. The fitted complete-call medians moved `+1.09%`, `+0.99%`, and `+0.28%` at B1/B4/B16, and one B16 profile fell below parity. X2 is rejected as neutral-to-regressive.

### X3 pre-scaled d

Moving the exact `* 0.125` factor into the four static FP32 `d` conversions removed eight local multiplies and added four invariant multiplies. The artifact changed from `2,216` to `2,212` VALU issues and stayed exact, but fitted complete-call movement was `+0.24%`, `-0.42%`, and `-0.52%` at B1/B4/B16. No point cleared the advancement gate, so X3 is rejected as timing-neutral.

### X4 epilogue probes

The shared-column setup probe removed six artifacts VALU issues; the broader materialized-address form removed 25; the exact width-two BF16 RNE interleave preserved instruction and resource counts. All three passed the exact route and mutation matrix, but their complete-call and body movements were mixed or contradictory, so none is retained.

### X6 processor-mode metadata and X1+X3 composition

Applying the processor-mode spelling produced byte-identical objects and HSACOs at every row count, so it is assembler-accepted but executable-inert and no timing or typed identity follows. Rebuilding X3 on top of the retained X1 parent removed four VALU issues and remained exact, but a 25-repeat top-up measured complete ratios within `1.00166x` and body movement approximately flat with all intervals crossing parity; the composition is closed as timing-neutral.

## Remaining Work

The current direct-kernel benchmark separates DeepSeek route laws at B1. The learned realization measured `1.1296x` GGTensile/HIP (25-repeat interval `[1.1247x, 1.1345x]`) while the hash realization measured `1.2799x` (75-repeat interval `[1.2775x, 1.2822x]`), versus the documented `1.1594x`. B4 and B16 are close to their documented rows at approximately `1.1692x` and `1.0610x`. The B1 spread is route-law sensitivity rather than a single-kernel regression; B1 should be re-evaluated separately under each law before its tuning is considered settled. X2-X4 are not reopened by this split.

## Qualification Summary

The retained kernels passed exact packed-HIP comparison, independent bounded BF16-reference checks, finite-output checks, deterministic reruns, valid and malformed route matrices, both-projection and activation mutations, inactive-expert inertness, and two independent byte-identical builds. The independent reference error envelope is `0.03125` maximum absolute error with RMSE near `0.000118` at production sizes. The selected result is the K128-interleaved fused-selector J64 body for B1/B4 and the J80 body for B16, all with zero private storage, spills, scratch, calls, and dynamic stack.
