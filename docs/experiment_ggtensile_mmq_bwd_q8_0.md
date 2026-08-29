# GGTensile MMQ Backward Q8_0

This record covers the complete fused packed-weight Q8_0 backward kernels for gfx1151, wave32, WMMA V1, BF16 inputs and outputs, FP32 accumulation, and the existing 40-byte backward kernarg ABI. Q8_0 decode remains in the kernel: no dequantized weight shadow, external decode workspace, split-K path, or persistent workgroup is part of the contract.

The backward operation is:

```text
grad_input[M,N] = grad_output[M,K] @ dequant(weight[K,N])
```

Q8_0 stores 32 logical values in each 34-byte block. The logical weight shape is `[K,N]` and the packed shape is `[K,(N/32)*34]`.

## Exact Scope

The 18 ordinary keys use `(M,N,K)` with `M` in `{2048,8192,32768}` and these six `(N,K)` families:

| Family | `(N,K)` |
| --- | --- |
| Attention Q-A | `(4096,1024)` |
| Attention Q-B | `(1024,32768)` |
| Attention K/V | `(4096,512)` |
| Attention output-B | `(8192,4096)` |
| Shared gate/up | `(4096,2048)` |
| Shared down | `(2048,4096)` |

The five language-model-head chunks are `(M,N,K)` equal to `(32,4096,129280)`, `(64,4096,129280)`, `(128,4096,129280)`, `(256,4096,129280)`, and `(512,4096,129280)`. The complete scope is 23 exact matrix keys.

## Fastest Kernels Found

The table shows the fastest exact-key kernel recorded across the experiments. Throughput is `2*M*N*K/(kernel_ms*1e9)`. Speedup is the matched HIP-control time divided by kernel time; values above `1.0x` favor GGTensile. Hashes are typed kernel identities. The M256 row uses the promoted packed-VOPD configuration and is now the fastest exact-key result for that shape.

| Family | Matrix shape `(M,N,K)` | Kernel hash | TFLOPS | Speedup vs HIP |
| --- | --- | --- | ---: | ---: |
| Attention Q-A | `(2048,4096,1024)` | `ggsol_5d99198f5a5b1531` | 36.605 | 1.9341x |
| Attention Q-A | `(8192,4096,1024)` | `ggsol_2b0fbcdba62f88ea` | 31.963 | 1.2707x |
| Attention Q-A | `(32768,4096,1024)` | `ggsol_1f803f9e9f2e2943` | 32.435 | 1.5962x |
| Attention Q-B | `(2048,1024,32768)` | `ggsol_411e043d82a04f16` | 22.350 | 1.1392x |
| Attention Q-B | `(8192,1024,32768)` | `ggsol_d99de81b4c72bbf7` | 23.910 | 1.3756x |
| Attention Q-B | `(32768,1024,32768)` | `ggsol_b0d446542d5be3dd` | 27.129 | 1.3951x |
| Attention K/V | `(2048,4096,512)` | `ggsol_3a5738c0aa9c1e29` | 35.566 | 1.8638x |
| Attention K/V | `(8192,4096,512)` | `ggsol_f93b7a54f677aa30` | 32.646 | 1.3225x |
| Attention K/V | `(32768,4096,512)` | `ggsol_681a5edf8b824c29` | 32.996 | 1.2604x |
| Attention output-B | `(2048,8192,4096)` | `ggsol_55ae41807fbd0868` | 32.138 | 1.4271x |
| Attention output-B | `(8192,8192,4096)` | `ggsol_fc26b587879c3242` | 33.127 | 1.6062x |
| Attention output-B | `(32768,8192,4096)` | `ggsol_5f37e19fffe72524` | 33.155 | 1.5552x |
| Shared gate/up | `(2048,4096,2048)` | `ggsol_f668b8070bf0b672` | 36.122 | 1.7286x |
| Shared gate/up | `(8192,4096,2048)` | `ggsol_497b16a21524fd35` | 28.935 | 1.2103x |
| Shared gate/up | `(32768,4096,2048)` | `ggsol_c34883a1d6cdfa4d` | 29.659 | 1.4641x |
| Shared down | `(2048,2048,4096)` | `ggsol_ba37265667a0332a` | 32.204 | 1.5595x |
| Shared down | `(8192,2048,4096)` | `ggsol_8fdc9c95311213b6` | 28.882 | 1.5444x |
| Shared down | `(32768,2048,4096)` | `ggsol_3ed17a6b54e2e995` | 28.656 | 1.4303x |
| LM head | `(32,4096,129280)` | `ggsol_29a74eb9e61887ed` | 7.998 | 1.9447x |
| LM head | `(64,4096,129280)` | `ggsol_e406d4a997a19be1` | 16.367 | 1.7172x |
| LM head | `(128,4096,129280)` | `ggsol_cc9c423da8db42f9` | 23.676 | 1.3377x |
| LM head | `(256,4096,129280)` | `ggsol_94d472bf4644b5c2` | 28.769 | 1.0943x |
| LM head | `(512,4096,129280)` | `ggsol_8ef0a3439f56bcda` | 30.288 | 1.2458x |

The K/V row retains the fastest earlier candidate timing because the later reopening was a control calibration. The current equivalent typed identity is `ggsol_3a5738c0aa9c1e29`; the older log used `ggsol_8f3a1d10bca36750` before the catalog schema cleanup.

## Final Profile Results

The selected ordinary body is a padded `256x64` tile with `DepthU=32`, WGM1, PGR2/PLR1, one LDS buffer, activation prefetch, interleaved WMMA waits, packed per-lane Q8 extraction, packed-weight prefetch, and raised store priority. The exact-key exceptions are Q-B M2048 (`2x8`, scalar extraction, next-packed prefetch), Q-B M8192 (`2x8`, `DepthU=64`, pad/swizzle 8), Q-B M32768 (`2x8`), long shared projections (`2x4`), and LM M32/M64/M128/M256 packed-VOPD bodies.

| Profile | VGPR | SGPR | LDS bytes | Instruction profile |
| --- | ---: | ---: | ---: | --- |
| Ordinary `256x64`, `DepthU=32` | 231 | 16 | 5120 | 808 instructions, 544 VALU issues, 146 VMEM, 16 VOPD, 32 WMMA, 2 barriers, 18 waits |
| Q-B M8192 `2x8`, `DepthU=64` | 220 | 16 | 16384 | 1208 instructions, 808 VALU issues, 152 VMEM, 22 VOPD, 64 WMMA |
| LM M32/M64/M128 packed-VOPD | 106/92/140 | 16 | 9216 | 41/25/26 VOPD; zero private bytes and zero spills |
| LM M256 packed-VOPD | 231 | 16 | 5120 | 805 instructions, 541 VALU issues, 146 VMEM, 20 VOPD, 32 WMMA |
| LM M512 packed body | 231 | 16 | 5120 | 946 instructions, 650 VALU issues, 164 VMEM, 12 VOPD, 32 WMMA |

All final artifacts use zero private storage and zero VGPR or SGPR spills. The ordinary short-K artifact also has the existing 40-byte ABI and code-object v5.

The lower-bound profiles explain the remaining cost without retaining full candidate/control timing matrices:

| Representative key | Complete | WMMA/A/LDS floor | Decode/LDS floor | Floor sum / complete |
| --- | ---: | ---: | ---: | ---: |
| Q-A M32768 | 8.208 ms | 6.260 ms | 1.746 ms | 0.975x |
| Q-B M8192 | 22.329 ms | 12.504 ms | 8.717 ms | 0.950x |
| Output-B M32768 | 65.968 ms | 49.257 ms | 15.346 ms | 0.979x |
| Shared gate/up M32768 | 18.261 ms | 12.330 ms | 5.684 ms | 0.986x |
| Shared down M32768 | 18.728 ms | 12.373 ms | 5.743 ms | 0.967x |
| LM M512 | 17.401 ms | 13.185 ms | 5.967 ms | 1.101x |

Ordinary floor sums are within 2-5% of complete timing. Decode VALU/VMEM and WMMA/LDS synchronization are the dominant residual costs, with their overlap already close to the measured path. Q-B has the largest decode fraction; LM M512 is dominated by packed payload/scale traffic and WMMA occupancy under its two-tile launch.

Every one of the 23 kernels passed exact HIP comparison, independent-reference checks, finite-output checks, gradient-output mutation, and packed-weight mutation. Independent source/object/HSACO builds were byte-identical with matching resources. The current K/V direct comparison also matched HIP over 8,388,608 BF16 outputs before timing.

## Accepted Experiments

Fused Q8_0 backend. The dedicated 32-value/34-byte decoder, signed-int8 reconstruction, FP16 scale conversion, packed-row addressing, and LDS-facing layout were accepted after one-hot K32/K64 fixtures and reduced-K boundary coverage. The implementation consumes the packed tensor directly and remains separate from K-family decoders.

Padded LDS and ordinary pipeline. Unswizzled `LdsPadB=8` was accepted as the ordinary layout. The initial unpadded `128x128x32` body was slower on representative keys, while pad8 reduced representative long-row cost by roughly 15-25%. PGR2/PLR1, single LDS buffering, activation prefetch, interleaved waits, per-lane packed loads, packed-weight prefetch, and raised store priority form the accepted ordinary pipeline.

Per-key geometry and schedule. `256x64` was accepted for Q-A, K/V, attention output-B, and the M2048 shared projections. `128x64` was accepted for long shared projections. Q-B uses exact-key variants: scalar `2x8` with next-packed prefetch at M2048, corrected `DepthU=64` with pad/swizzle 8 at M8192, and packed `2x8` at M32768. A temporary payload pointer repaired the original Q8 `DepthU=64` address-lifetime collision without increasing resources.

Compact LM ownership and packed VOPD. Exact `32x64`, `64x64`, and `128x64` ownership bodies were accepted. Packed VOPD decode reduced their screening latency by about 3.55%, 2.90%, and 3.57%; the final bodies use 106, 92, and 140 VGPR respectively with no spills or private storage.

M256 packed-VOPD promotion. The exact `256x64`, `DepthU=32`, pad8 body was promoted to the canonical M256 entry after two independent 25-repeat brackets improved on its typed parent by 1.59% and 1.72%, with robust intervals excluding parity. The promoted identity is `ggsol_94d472bf4644b5c2`; its brackets against HIP average `28.769 TFLOPS` and `1.0943x`. It remains scoped to M256 and is not transferred to M512 or other shapes.

K/V M2048 control calibration. The current `ggsol_3a5738c0aa9c1e29` body was remeasured with 20 warmups and 25 alternating paired samples. Single launch measured `0.2424 ms` versus `0.7230 ms` HIP (`2.983x`); 16 launches/sample measured `0.2440 ms` versus `0.6955 ms` HIP (`2.851x`). Independent and paired log-time intervals excluded parity, so no top-up or body change was accepted. Reports are `~/tmp/torch-ggml-ops/q8-kv-m2048-current-single-20x25.json` and `~/tmp/torch-ggml-ops/q8-kv-m2048-current-batched16-20x25.json`.

## Rejected Experiments

Initial unpadded body. The first `128x128x32` control was slower on the weighted ordinary workload, with a candidate/HIP latency ratio of about `1.033`. It was rejected in favor of padded LDS.

Broad geometry and mapping changes. WGM2, `64x128`, `128x128` where it did not win, and `256x128` were rejected by timing or occupancy. Exact compact ownership was retained only where the shape-specific evidence supported it.

Padding, XOR, and decoded-B alternatives. Pad16/24 and XOR4/8/16 one-buffer layouts were rejected. The two-decoded-B pipeline passed correctness but was timing-neutral on its discriminator keys, so it was rejected. It did not reveal a reusable Q8 overlap mechanism.

Broad decoder and pipeline variants. Broad scalar extraction, PLR2, SIA3, SIA4 without the accepted prefetch combination, PGR1, normal store priority, broad next-packed prefetch, and unrelated dependency-width variants were rejected by neutral or unfavorable timing. The Q8 scalar and next-prefetch choices remain exact-key exceptions rather than transferable defaults.

Depth and transfer variants. Corrected `DepthU=64` improved Q-B M8192 but regressed attention output-B and was not generalized. The corrected body was retained only for that Q-B key. Other Q8 geometry and schedule transfers failed their exact-key timing or resource evidence.

LM alternatives. `32x128`, wider compact tiles, pad16/24, XOR8, WGM2, and non-VOPD decoder variants were rejected. M512 packed-VOPD regressed by about 2.62%; the earlier fixed-threshold rejection of M256 VOPD was superseded by the qualified exact-key promotion above.

## Bottleneck and Final Review

The accepted kernels have already closed the large layout, ownership, geometry, schedule, and decoder alternatives that fit the contract. The lower bounds show no large hidden scheduling gap. The remaining opportunity is the overlap of Q8 payload/scale VMEM, decode VALU, LDS synchronization, and WMMA occupancy; the broad mechanisms that could alter that overlap either lost timing, consumed unacceptable resources, or failed to transfer across exact shapes.

The final recursive review finds no actionable in-contract mechanism remaining for the 23-key Q8_0 kernel set. The K/V reopening was control calibration and leaves its kernel unchanged. The qualified M256 packed-VOPD identity is now part of the retained exact-key mix and remains scoped to M256. Further work would require a new kernel identity, a concrete code-generation premise, exact correctness and resource evidence, and fresh noisy timing rather than another broad search.
