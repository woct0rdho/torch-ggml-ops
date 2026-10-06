# GGTensile MMQ Backward Q8_0

This record covers the ordinary dense Q8_0 backward kernels on gfx1151:

```text
grad_input[M,N] = grad_output[M,K] @ dequant_q8_0(weight[K,N])
```

## Final Results

`TFLOPS = 2*M*K*N / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Medians are the repository benchmark's, re-measured in the current clock state. They replace the earlier recorded values, which came from a different clock state with the same artifacts.

| Family | `(M,K,N)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| --- | --- | ---: | ---: | --- | --- |
| Attention Q-A | `(2048,4096,1024)` | 33.519 | 1.0007x | `mmq_bwd_q8_0_m2048_n4096_k1024_5d99198f5a5b1531` | `dense_bwd_q8_0_pipea_nt4_ki64_mw2_pad8` |
| Attention Q-A | `(8192,4096,1024)` | 32.983 | 1.0475x | `mmq_bwd_q8_0_m8192_n4096_k1024_2b0fbcdba62f88ea` | `dense_bwd_q8_0_pipea_nt4_ki64_mw4_sw16_prefetch` |
| Attention Q-A | `(32768,4096,1024)` | 33.272 | 1.0158x | `mmq_bwd_q8_0_m32768_n4096_k1024_1f803f9e9f2e2943` | `dense_bwd_q8_0_pipea_nt4_ki64_mw4_sw16_prefetch` |
| Attention Q-B | `(2048,1024,32768)` | 22.886 | 1.1566x | `mmq_bwd_q8_0_m2048_n1024_k32768_411e043d82a04f16` | `dense_bwd_q8_0_exact_n32768k1024_g2_group_m1_padding8` |
| Attention Q-B | `(8192,1024,32768)` | 24.870 | 1.2498x | `mmq_bwd_q8_0_m8192_n1024_k32768_d99de81b4c72bbf7` | `dense_bwd_q8_0_pipea_nt4_ki64_mw2_pad8` |
| Attention Q-B | `(32768,1024,32768)` | 26.542 | 1.2049x | `mmq_bwd_q8_0_m32768_n1024_k32768_b0d446542d5be3dd` | `dense_bwd_q8_0_exact_n32768k1024_g2_group_m1_padding8` |
| Attention K/V | `(2048,4096,512)` | 33.349 | 1.0504x | `mmq_bwd_q8_0_m2048_n4096_k512_3a5738c0aa9c1e29` | `dense_bwd_q8_0_pipea_nt4_ki64_mw2_pad8` |
| Attention K/V | `(8192,4096,512)` | 33.225 | 1.0699x | `mmq_bwd_q8_0_m8192_n4096_k512_f93b7a54f677aa30` | `dense_bwd_q8_0_pipea_nt4_ki64_mw4_sw16_prefetch` |
| Attention K/V | `(32768,4096,512)` | 34.079 | 1.0220x | `mmq_bwd_q8_0_m32768_n4096_k512_681a5edf8b824c29` | `dense_bwd_q8_0_pipea_nt4_ki64_mw4_sw16_prefetch` |
| Attention output-B | `(2048,8192,4096)` | 33.134 | 1.0481x | `mmq_bwd_q8_0_m2048_n8192_k4096_55ae41807fbd0868` | `dense_bwd_q8_0_pipea_nt4_ki64_mw4_sw16_prefetch` |
| Attention output-B | `(8192,8192,4096)` | 33.609 | 1.0670x | `mmq_bwd_q8_0_m8192_n8192_k4096_fc26b587879c3242` | `dense_bwd_q8_0_pipea_nt4_ki64_mw4_sw16_prefetch` |
| Attention output-B | `(32768,8192,4096)` | 32.284 | 1.1570x | `mmq_bwd_q8_0_m32768_n8192_k4096_5f37e19fffe72524` | `dense_bwd_q8_0_pipea_nt4_ki64_mw2_sw16` |
| Shared gate/up | `(2048,4096,2048)` | 35.167 | 1.0677x | `mmq_bwd_q8_0_m2048_n4096_k2048_f668b8070bf0b672` | `dense_bwd_q8_0_pipea_nt4_ki64_mw4_sw16_prefetch` |
| Shared gate/up | `(8192,4096,2048)` | 29.610 | 1.0204x | `mmq_bwd_q8_0_m8192_n4096_k2048_497b16a21524fd35` | `dense_bwd_q8_0_pipea_nt4_ki64_mw2_sw16` |
| Shared gate/up | `(32768,4096,2048)` | 28.739 | 0.9916x | `mmq_bwd_q8_0_m32768_n4096_k2048_c34883a1d6cdfa4d` | `dense_bwd_q8_0_pipea_nt4_ki64_mw2_sw16` |
| Shared down | `(2048,2048,4096)` | 31.298 | 1.0428x | `mmq_bwd_q8_0_m2048_n2048_k4096_ba37265667a0332a` | `dense_bwd_q8_0_pipea_nt4_ki64_mw2_pad8` |
| Shared down | `(8192,2048,4096)` | 28.886 | 1.0077x | `mmq_bwd_q8_0_m8192_n2048_k4096_8fdc9c95311213b6` | `dense_bwd_q8_0_pipea_nt4_ki64_mw2_sw16_prefetch` |
| Shared down | `(32768,2048,4096)` | 28.311 | 1.0226x | `mmq_bwd_q8_0_m32768_n2048_k4096_3ed17a6b54e2e995` | `dense_bwd_q8_0_pipea_nt4_ki64_mw2_sw16` |
| LM head | `(32,4096,129280)` | 7.986 | 0.9231x | `mmq_bwd_q8_0_m32_n4096_k129280_29a74eb9e61887ed` | `dense_bwd_q8_0_exact_lm_head_splitk_m32_s4_full_aw2` |
| LM head | `(64,4096,129280)` | 16.327 | 0.9181x | `mmq_bwd_q8_0_m64_n4096_k129280_e406d4a997a19be1` | `dense_bwd_q8_0_exact_lm_head_splitk_m64_s4` |
| LM head | `(128,4096,129280)` | 23.707 | 1.0400x | `mmq_bwd_q8_0_m128_n4096_k129280_cc9c423da8db42f9` | `dense_bwd_q8_0_exact_lm_head_splitk_m128_s16` |
| LM head | `(256,4096,129280)` | 29.285 | 0.9037x | `mmq_bwd_q8_0_m256_n4096_k129280_94d472bf4644b5c2` | `dense_bwd_q8_0_exact_lm_head_splitk_m256_s16` |
| LM head | `(512,4096,129280)` | 30.205 | 1.1346x | `mmq_bwd_q8_0_m512_n4096_k129280_8ef0a3439f56bcda` | `dense_bwd_q8_0_exact_lm_head_splitk_m512_s32` |

GGTensile is ahead on 19 of the 23 entries, with speedups from `0.904x` to `1.250x` (mean `1.050x`). The five language-model-head chunks are the rows where the deployed HIP bodies use split contraction.

The earlier recordings of this family were paired against HIP backward bodies that have since been retiled and repipelined, so these speedups are lower than the recorded ones even though GGTensile's own throughput is unchanged.

## Accepted Experiments

Fused Q8_0 backend. The dedicated 32-value/34-byte decoder, signed-int8 reconstruction, FP16 scale conversion, packed-row addressing, and LDS-facing layout were accepted after one-hot K32/K64 fixtures and reduced-K boundary coverage. The implementation consumes the packed tensor directly and remains separate from K-family decoders.

Padded LDS and ordinary pipeline. Unswizzled `LdsPadB=8` was accepted as the ordinary layout. The initial unpadded `128x128x32` body was slower on representative keys, while pad8 reduced representative long-row cost by roughly 15-25%. PGR2/PLR1, single LDS buffering, activation prefetch, interleaved waits, per-lane packed loads, packed-weight prefetch, and raised store priority form the accepted ordinary pipeline.

Per-key geometry and schedule. `256x64` was accepted for Q-A, K/V, attention output-B, and the M2048 shared projections. `128x64` was accepted for long shared projections. Q-B uses exact-key variants: scalar `2x8` with next-packed prefetch at M2048, corrected `DepthU=64` with pad/swizzle 8 at M8192, and packed `2x8` at M32768. A temporary payload pointer repaired the original Q8 `DepthU=64` address-lifetime collision without increasing resources.

Compact LM ownership and packed VOPD. Exact `32x64`, `64x64`, and `128x64` ownership bodies were accepted. Packed VOPD decode reduced their screening latency by about 3.55%, 2.90%, and 3.57%. The final bodies use 106, 92, and 140 VGPR respectively with no spills or private storage.

M256 packed-VOPD promotion. The exact `256x64`, `DepthU=32`, pad8 body was promoted to the canonical M256 entry after two independent 25-repeat brackets improved on its typed parent by 1.59% and 1.72%, with robust intervals excluding parity. The promoted identity is `ggsol_94d472bf4644b5c2. Its brackets against HIP average `28.769 TFLOPS` and `1.0943x`. It remains scoped to M256 and is not transferred to M512 or other shapes.

K/V M2048 control calibration. The current `ggsol_3a5738c0aa9c1e29` body was remeasured with 20 warmups and 25 alternating paired samples. Single launch measured `0.2424 ms` versus `0.7230 ms` HIP (`2.983x`). 16 launches/sample measured `0.2440 ms` versus `0.6955 ms` HIP (`2.851x`). Independent and paired log-time intervals excluded parity, so no top-up or body change was accepted. Reports are `~/tmp/torch-ggml-ops/q8-kv-m2048-current-single-20x25.json` and `~/tmp/torch-ggml-ops/q8-kv-m2048-current-batched16-20x25.json`.

## Rejected Experiments

Initial unpadded body. The first `128x128x32` control was slower on the weighted ordinary workload, with a candidate/HIP latency ratio of about `1.033`. It was rejected in favor of padded LDS.

Broad geometry and mapping changes. WGM2, `64x128`, `128x128` where it did not win, and `256x128` were rejected by timing or occupancy. Exact compact ownership was retained only where the shape-specific evidence supported it.

Padding, XOR, and decoded-B alternatives. Pad16/24 and XOR4/8/16 one-buffer layouts were rejected. The two-decoded-B pipeline passed correctness but was timing-neutral on its discriminator keys, so it was rejected. It did not reveal a reusable Q8 overlap mechanism.

Broad decoder and pipeline variants. Broad scalar extraction, PLR2, SIA3, SIA4 without the accepted prefetch combination, PGR1, normal store priority, broad next-packed prefetch, and unrelated dependency-width variants were rejected by neutral or unfavorable timing. The Q8 scalar and next-prefetch choices remain exact-key exceptions rather than transferable defaults.

Depth and transfer variants. Corrected `DepthU=64` improved Q-B M8192 but regressed attention output-B and was not generalized. The corrected body was retained only for that Q-B key. Other Q8 geometry and schedule transfers failed their exact-key timing or resource evidence.

LM alternatives. `32x128`, wider compact tiles, pad16/24, XOR8, WGM2, and non-VOPD decoder variants were rejected. M512 packed-VOPD regressed by about 2.62%. The earlier fixed-threshold rejection of M256 VOPD was superseded by the qualified exact-key promotion above.

## Bottleneck and Final Review

The accepted kernels have already closed the large layout, ownership, geometry, schedule, and decoder alternatives that fit the contract. The lower bounds show no large hidden scheduling gap. The remaining opportunity is the overlap of Q8 payload/scale VMEM, decode VALU, LDS synchronization, and WMMA occupancy. The broad mechanisms that could alter that overlap either lost timing, consumed unacceptable resources, or failed to transfer across exact shapes.

The final recursive review finds no actionable in-contract mechanism remaining for the 23-key Q8_0 kernel set. The K/V reopening was control calibration and leaves its kernel unchanged. The qualified M256 packed-VOPD identity is now part of the retained exact-key mix and remains scoped to M256. Further work would require a new kernel identity, a concrete code-generation premise, exact correctness and resource evidence, and fresh noisy timing rather than another broad search.
