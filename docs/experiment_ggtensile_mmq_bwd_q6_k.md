# GGTensile MMQ Backward Q6_K Results and Experiment Log

This record covers the gfx1151 ordinary dense Q6_K backward kernels for `output.weight` in `Qwen3.6-35B-A3B-APEX-I-Mini.gguf`. Matrix dimensions are `(M,N,K) = (rows, in_features, out_features)`:

```text
grad_input[M,N] = grad_output[M,K] @ dequant(Q6_K weight[K,N])
```

Q6_K uses 256 logical values in each 210-byte block: two 128-value low/high payload planes, 16 signed int8 scales, and one FP16 block multiplier. The generated kernels decode this representation in place, accumulate in FP32, and round stores to BF16.

## Final Benchmark Results

The table shows the fastest retained kernel found for each production matrix shape across the complete experiment record. These are the latest 25-sample direct-kernel medians after 20 warmups, using balanced controls and launch batching for the shortest kernel. Speed is logical arithmetic throughput, and speedup is `HIP time / GGTensile time`.

| Matrix shape | Kernel hash | Speed (TFLOPS) | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(64,2048,248320)` | `ggsol_47dc5792dc6a3094` | 12.677 | 1.0628x |
| `(128,2048,248320)` | `ggsol_322cef48a5fa0b12` | 19.845 | 1.4049x |
| `(256,2048,248320)` | `ggsol_83942eadf7602aa1` | 26.641 | 1.2165x |

The latest robust 95% speedup intervals were `1.0521x..1.0737x` for M64, `1.3888x..1.4213x` for M128, and `1.2058x..1.2274x` for M256. All three retained kernels are faster than HIP under the same direct benchmark.

## Final Profile Results

The final code objects are resource-clean: no private storage, no VGPR or SGPR spills, and a 40-byte kernarg segment. Static instruction profiles are:

| Matrix shape | VGPR | SGPR | LDS (B) | WMMA | VALU issues | VOPD | VMEM | LDS ops | Waits | Barriers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `(64,2048,248320)` | 76 | 16 | 4608 | 8 | 214 | 25 | 28 | 32 | 8 | 2 |
| `(128,2048,248320)` | 108 | 16 | 4608 | 16 | 275 | 26 | 52 | 32 | 8 | 2 |
| `(256,2048,248320)` | 240 | 16 | 5120 | 32 | 703 | 44 | 168 | 48 | 11 | 3 |

Lower-bound diagnostics use the same kernel bodies and ABI but remove selected work. Their median latencies and floor sums are:

| Matrix shape | Complete (ms) | WMMA/A/LDS floor (ms) | Q6 decode/LDS floor (ms) | Floor sum / complete |
| --- | ---: | ---: | ---: | ---: |
| `(64,2048,248320)` | 5.230 | 2.659 | 3.283 | 1.136x |
| `(128,2048,248320)` | 6.631 | 3.776 | 3.291 | 1.066x |
| `(256,2048,248320)` | 9.721 | 7.166 | 5.573 | 1.311x |

M64 is decode-heavy: the decode floor is 62.8% of complete latency. M128 has no isolated dominant floor and remains balanced. M256 is constrained primarily by WMMA and accumulator work; its floor sum shows substantial overlap in the complete body.

All three retained kernels are bit-exact to HIP and to an independent BF16 reference under baseline inputs, repeated execution, grad-output mutation, and packed-weight mutation. Independent builds are byte-identical and retain the resources listed above.

## Accepted Kernel Designs

- Q6 packed VOPD decode. The retained decoder coalesces low/high payload reads, applies the signed scale and FP16 multiplier in the required operand order, and uses legal gfx11 VOPD pairs for adjacent subtract/multiply work. Scalar and ordinary packed extraction did not provide a better body.
- M64 `64x32x64`, pad8, one decoded-B LDS buffer. The 76-VGPR body is the fastest M64 kernel found. Packed VOPD improved the packed pad8 control by about 3.2% in the confirming control run, while pad8 tied the larger pad24 alternative with less LDS.
- M128 `128x32x64`, pad8, one decoded-B LDS buffer. Compact ownership beat the `128x64x32` next-prefetch body by about 2.77% while reducing resources from 140 to 108 VGPRs and from 5120 to 4608 bytes of LDS.
- M256 `256x64x32`, pad8, packed VOPD, next packed-tile prefetch. Wide ownership beat the narrow next-prefetch body by about 18.0% and is the fastest M256 body found. The selected body retains 240 VGPRs and 5120 bytes of LDS without spills.
- Exact Q6 arithmetic and row mapping. The selected bodies preserve the 210-byte block layout, signed scales, FP16 `d`, packed six-bit reconstruction, BF16 rounding, and decoded LDS order across block and packed-row boundaries.

## Experiment Log

### Decoder and extraction

- The initial Q6 decoder was accepted after independent reduced fixtures covering low and high payload bits, signed scale extremes, FP16 `d`, all 16 scale groups, both 128-value chunks, block boundaries, and packed-row boundaries. Production M64, M128, and M256 outputs then matched HIP and the independent reference.
- Packed VOPD was accepted after exact correctness and resource checks. It keeps the packed six-bit reconstruction but pairs adjacent subtract and multiply operations. Scalar extraction and the ordinary packed emitter were rejected by timing; neither removed enough issue pressure to beat packed VOPD.
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

The corrected compact candidate, `ggsol_48942af62f3decd6`, passed production correctness, grad-output and packed-weight mutations, independent rebuilds, and resource inspection with 77 VGPRs, 16 SGPRs, 9216 bytes of LDS, two barriers, no private bytes, and no spills. Its balanced 25-sample run measured `5.2116 ms` against `5.0513 ms` for the retained M64 body. The candidate/parent ratio was `1.0317x`, with a robust 95% interval of `1.0110x..1.0529x`; it was still about 1.58x faster than HIP. The stable parent regression rejected the extra buffer.

The profile explains the result: both bodies execute the same 3880 decoded tiles, 31040 WMMAs, 62080 decoded-B stores, 62080 LDS loads, and 27160 explicit waits. The two-buffer body halves full barriers, but all four waves remain symmetric producers and consumers, so each wave still carries its decode chunks on the WMMA dependency path. Parity and base-update instructions for the non-power-of-two buffers add work without creating independent decode ownership.

### Correctness-only outcomes

- Q6 temporary and address-state allocation was kept separate from the decoder payload registers. This prevented the pipeline LDS-read helper from clobbering the live A global-load pointer.
- Unswizzled decoded rows use linear `2*k_tile` LDS byte offsets. Swizzled rows retain their existing transform. The distinction is covered by writer-level source tests and reduced runtime fixtures.
- The corrected pipeline source is retained as a research-capable emitter for reproducibility, but it is not an accepted kernel design because its final timing is slower than the retained M64 body.

## Deferred Kernel Experiments

- Approximate decoded-weight or output BF16 conversion (`BiasRound` or `Truncate`) remains unmeasured. Exact RNE conversion is the retained numerical behavior; no approximate kernel is accepted without independent error and timing evidence.
- A combined padded-stride/XOR layout remains deferred because it needs supported LDS-conflict evidence and an occupancy-safe gain. Existing layout screens did not establish that premise.
- Prepared decoded weights, producer fusion, persistent execution, split-K, and external decode storage were not treated as Q6 packed in-kernel optimizations. They require a different ownership and measurement contract.

A final recursive review after the corrected compact pipeline found no additional actionable in-kernel mechanism. M64 remains decode-limited, M128 remains balanced, and M256 remains WMMA/accumulator-limited. The retained kernels and their measured profiles are therefore the final Q6_K backward results in this record.
