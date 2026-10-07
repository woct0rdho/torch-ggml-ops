# GGTensile MMQ Backward Q6_K Results and Experiment Log

This record covers the ordinary dense Q6_K LM-head backward kernels on gfx1151.

## Final Results

`TFLOPS = 2*M*N*K / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Medians are the repository benchmark's, re-measured in the current clock state. They replace the earlier recorded values, which came from a different clock state with the same artifacts.

| Family | `(M,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| --- | --- | ---: | ---: | --- | --- |
| Language model head | `(64,248320,2048)` | 12.787 | 0.9834x | `mmq_bwd_q6_k_m64_n248320_k2048_afa65b05cf14eb2f` | `dense_bwd_q6_k_pipesplit_m64_s2` |
| Language model head | `(128,248320,2048)` | 19.626 | 0.8839x | `mmq_bwd_q6_k_m128_n248320_k2048_788ae3f1307542df` | `dense_bwd_q6_k_pipesplit_m128_s40` |
| Language model head | `(256,248320,2048)` | 26.239 | 1.1946x | `mmq_bwd_q6_k_m256_n248320_k2048_0b9c1ef20d16bee6` | `dense_bwd_q6_k_pipesplit_m256_s40` |

GGTensile is ahead on 1 of the 3 entries, with speedups from `0.884x` to `1.195x` (mean `1.021x`).

The earlier recordings of this family were paired against HIP backward bodies that have since been retiled and repipelined, so these speedups are lower than the recorded ones even though GGTensile's own throughput is unchanged.

## Accepted Kernel Designs

- Q6 packed VOPD decode. The retained decoder coalesces low/high payload reads, applies the signed scale and FP16 multiplier in the required operand order, and uses legal gfx11 VOPD pairs for adjacent subtract/multiply work. Scalar and ordinary packed extraction did not provide a better body.
- M64 `64x32x64`, pad8, one decoded-B LDS buffer. The 76-VGPR body is the fastest M64 kernel found. Packed VOPD improved the packed pad8 control by about 3.2% in the confirming control run, while pad8 tied the larger pad24 alternative with less LDS.
- M128 `128x32x64`, pad8, one decoded-B LDS buffer. Compact ownership beat the `128x64x32` next-prefetch body by about 2.77% while reducing resources from 140 to 108 VGPRs and from 5120 to 4608 bytes of LDS.
- M256 `256x64x32`, pad8, packed VOPD, next packed-tile prefetch. Wide ownership beat the narrow next-prefetch body by about 18.0% and is the fastest M256 body found. The selected body retains 240 VGPRs and 5120 bytes of LDS without spills.
- Exact Q6 arithmetic and row mapping. The selected bodies preserve the 210-byte block layout, signed scales, FP16 `d`, packed six-bit reconstruction, BF16 rounding, and decoded LDS order across block and packed-row boundaries.

## Experiment Log

### Decoder and extraction

- The initial Q6 decoder was accepted after independent reduced fixtures covering low and high payload bits, signed scale extremes, FP16 `d`, all 16 scale groups, both 128-value chunks, block boundaries, and packed-row boundaries. Production M64, M128, and M256 outputs then matched HIP and the independent reference.
- Packed VOPD was accepted after exact correctness and resource checks. It keeps the packed six-bit reconstruction but pairs adjacent subtract and multiply operations. Scalar extraction and the ordinary packed emitter were rejected by timing. Neither removed enough issue pressure to beat packed VOPD.
- High-plane-only lane sharing was implemented correctly and passed all mutation checks, but the M256 screen measured `12.8034 ms` against a `9.8869 ms` selected control (`1.29498x` candidate/control). The extra EXEC and DPP work outweighed the reduced active high-plane loads.
- Pair-owner loading of the shared FP16 `d` and vector metadata loading were closed without a gain premise. Their VMEM addresses already coalesce, while ownership or extraction adds control and dependency work.

### Ownership, LDS layout, and scheduling

- M64 `64x32x64`, M128 `128x32x64`, and M256 `256x64x32` were accepted as the best ownership choices after exact shape searches. Narrower or wider alternatives were correct in some cases but slower, including the M256 `256x32x64` occupancy candidate, which was 32% to 43% slower than HIP.
- Pad8 was retained. Pad16, pad24 where it did not reduce throughput, unpadded LDS, and XOR8/XOR16 placement were neutral, slower, or used more LDS without a measured gain. Pad24 tied pad8 for the compact M64 body, so the smaller pad8 allocation won.
- WGM, SIA, store-priority, VMEM-clause, and `buffer_gl0_inv` variants were neutral or slower in repeated controls. They did not change the selected instruction dependency or resource bottleneck.
- M256 next packed-tile prefetch was accepted. The same prefetch at M64 was rejected after a `5.3565 ms` candidate versus a `5.1167 ms` control (`1.04688x` candidate/control). M128 compact ownership was preferred over transferring this wider prefetch body.
- Wider DepthU64 ownership was rejected: M64 measured `7.0339 ms` versus `5.1141 ms` (`1.37538x`), and M256 measured `12.1769 ms` versus `9.9054 ms` (`1.22932x`).

### Repaired pipeline experiments

- The first wider-N two-buffer pipelines produced invalid columns. Geometry-derived handoff, Q6 temporary allocation, and address-state repairs restored exact HIP/reference agreement for the M128 and M256 production repros, but the corrected screens still measured `9.6587 ms` versus `6.6179 ms` for M128 (`1.45949x`) and `14.5063 ms` versus `9.8659 ms` for M256 (`1.47034x`). They were rejected by timing.
- DepthU64 next-prefetch initially faulted because packed reads overwrote current A pointers too early. Delaying those reads until the second current-A half was consumed restored exact M64 correctness and both mutations. Its `1.04688x` candidate/control screen rejected the mechanism.
- The compact M64 two-buffer candidate preserved `64x32x64` ownership and used two 4608-byte decoded-B buffers. Reduced K64, K128, and K192 fixtures exposed three structural errors: the read base was mixed with decoder coordinates, unswizzled nonzero-K reads used the wrong XOR helper, and subtracting the 4608-byte buffer size used the wrong operand order. Each was repaired before production timing.

The corrected compact candidate, `ggsol_48942af62f3decd6`, passed production correctness, grad-output and packed-weight mutations, independent rebuilds, and resource inspection with 77 VGPRs, 16 SGPRs, 9216 bytes of LDS, two barriers, no private bytes, and no spills. Its balanced 25-sample run measured `5.2116 ms` against `5.0513 ms` for the retained M64 body. The candidate/parent ratio was `1.0317x`, with a robust 95% interval of `1.0110x..1.0529x`. It was still about 1.58x faster than HIP. The stable parent regression rejected the extra buffer.

The profile explains the result: both bodies execute the same 3880 decoded tiles, 31040 WMMAs, 62080 decoded-B stores, 62080 LDS loads, and 27160 explicit waits. The two-buffer body halves full barriers, but all four waves remain symmetric producers and consumers, so each wave still carries its decode chunks on the WMMA dependency path. Parity and base-update instructions for the non-power-of-two buffers add work without creating independent decode ownership.

### Correctness-only outcomes

- Q6 temporary and address-state allocation was kept separate from the decoder payload registers. This prevented the pipeline LDS-read helper from clobbering the live A global-load pointer.
- Unswizzled decoded rows use linear `2*k_tile` LDS byte offsets. Swizzled rows retain their existing transform. The distinction is covered by writer-level source tests and reduced runtime fixtures.
- The corrected pipeline source is retained as an emitter for reproducibility, but it is not an accepted kernel design because its final timing is slower than the retained M64 body.

## Deferred Kernel Experiments

- Approximate decoded-weight or output BF16 conversion (`BiasRound` or `Truncate`) remains unmeasured. Exact RNE conversion is the retained numerical behavior. No approximate kernel is accepted without independent error and timing evidence.
- A combined padded-stride/XOR layout remains deferred because it needs supported LDS-conflict evidence and an occupancy-safe gain. Existing layout screens did not establish that premise.
- Prepared decoded weights, producer fusion, persistent execution, split-K, and external decode storage were not treated as Q6 packed in-kernel optimizations. They require a different ownership and measurement contract.

A final recursive review after the corrected compact pipeline found no additional actionable in-kernel mechanism. M64 remains decode-limited, M128 remains balanced, and M256 remains WMMA/accumulator-limited. The retained kernels and their measured profiles are therefore the final Q6_K backward results in this record.
