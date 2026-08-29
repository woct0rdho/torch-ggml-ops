# GGTensile MMQ Backward Q4_K Experiment Log

## Scope

These experiments cover ordinary dense Q4_K MMQ backward kernels on gfx1151, wave32, and WMMA V1. Kernels consume BF16 grad-output and packed Q4_K weights, accumulate in FP32, and store BF16 grad-input. Each kernel is an exact-shape artifact; grouped MMQ, paired projections, prepared weights, external decode storage, split-K, and persistent workgroups are outside this log.

The campaign covers the 12 exact matrices formed by:

```text
M = 2048, 8192, 32768
(N,K) = (2048,512), (512,2048), (4096,2048), (2048,8192)
```

The final kernel set uses shape-specific ownership. Narrow keys use padded `256x64` bodies, attention-output keys use padded `256x64` at M2048 and padded `128x64` at larger M, shared-down keys use the true `128x128` two-buffer pipeline, and query keys use that pipeline at M2048/M32768 plus padded `128x64` at M8192.

## Final Kernels

The table reports the fastest measured kernel found for each exact matrix across the completed experiments. Speed is logical fused-kernel throughput, calculated as `2*M*N*K/(median_ms*1e9)`. The speedup is `HIP time / GGTensile time`; values above `1.0x` favor GGTensile. Timing brackets were consistent, so A/B columns are omitted.

| Matrix shape `(M,N,K)` | Kernel hash | Speed (TFLOPS) | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(2048,2048,512)` | `ggsol_f8ae4100f87d6dac` | 29.719 | 1.6217x |
| `(8192,2048,512)` | `ggsol_ea25b77ecc82d61d` | 30.364 | 1.4452x |
| `(32768,2048,512)` | `ggsol_a1288e5bd8984359` | 34.583 | 1.5595x |
| `(2048,512,2048)` | `ggsol_f2ae729b0b3d54ba` | 21.290 | 1.2948x |
| `(8192,512,2048)` | `ggsol_283a88d55d119a89` | 14.740 | 1.2600x |
| `(32768,512,2048)` | `ggsol_4fd4e2ab584ba308` | 20.339 | 1.5206x |
| `(2048,4096,2048)` | `ggsol_ce0506ff376281f4` | 34.586 | 1.4009x |
| `(8192,4096,2048)` | `ggsol_196b5482b29aaf64` | 28.960 | 1.2579x |
| `(32768,4096,2048)` | `ggsol_554a1ee0cbb2a3fe` | 29.060 | 1.2236x |
| `(2048,2048,8192)` | `ggsol_aa1f0fec231c7a19` | 28.368 | 1.4702x |
| `(8192,2048,8192)` | `ggsol_e1bb6e35c73c2a15` | 27.961 | 1.2975x |
| `(32768,2048,8192)` | `ggsol_7cb50a7c50a4b278` | 29.733 | 1.2990x |

The M32768 narrow and query rows use the accepted typed `DependencyBatch4` identity and are now promoted in the exact-shape kernel set. The other rows are the fastest shape-specific kernels from the padded exact-key campaign. `DependencyBatch4` changes dependency presentation around the same exact Q4 arithmetic and does not change resources, ABI, or packed representation.

## Final Validation And Resources

All final kernels passed the applicable hard gates:
- bit-exact comparison with HIP;
- independent dequantized BF16 reference comparison;
- complete grad-output mutation;
- packed-weight mutation;
- reduced K32, K64, K96, and K512 checks where the selected geometry permits;
- Q4 one-hot scale and nibble coverage;
- deterministic independent source and code-object rebuilds; and
- zero private bytes, spills, scratch instructions, calls, and dynamic stack.

The final resource classes are:

| Kernel class | Shape coverage | VGPR | SGPR | LDS | Static WMMA | VMEM | LDS instructions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Padded `256x64` | narrow all M; attention-output M2048 | 234 | 16 | 5 KiB | 32 | 149 | 32 |
| Padded `128x64` | attention-output M8192/M32768; query M8192 | 136 | 16 | 5 KiB | 16 | 77 | 32 |
| True two-buffer `128x128` | shared-down all M; query M2048/M32768 | 212 | 16 | 16 KiB | 64 | 164 | 128 |
| `DependencyBatch4` narrow | `(32768,2048,512)` | 234 | 16 | 5 KiB | 32 | unchanged | unchanged |
| `DependencyBatch4` query | `(32768,2048,8192)` | 212 | 16 | 16 KiB | 64 | unchanged | unchanged |

Independent final roots were byte-identical for the corresponding source and code-object builds. The true two-buffer class has 23 waits and two barriers in its representative long-K artifact; the typed `DependencyBatch4` candidates preserve the parent resource class and synchronization structure.

## Profile Results

The representative lower-bound profiles separate complete execution into a WMMA/activation/LDS floor and a decode/LDS floor. These are serial lower bounds used to explain overlap, not alternative kernels.

| Cohort and shape | Complete | WMMA/A/LDS floor | Decode/LDS floor | Floor sum |
| --- | ---: | ---: | ---: | ---: |
| Narrow `(32768,2048,512)` | 2.3131 ms | 1.6197 ms | 1.0915 ms | 117.21% |
| Query `(32768,2048,8192)` | 37.868 ms | 24.149 ms | 16.842 ms | 108.25% |
| Attention-output `(32768,4096,2048)` | 18.686 ms | 12.008 ms | 8.532 ms | 109.9% |

The two-buffer profile reduced median aggregate wave cycles from the one-buffer control by approximately 4.9%. Wait-count stalls fell from 23.74% to 21.38%, and barrier stalls fell from 8.58% to 6.11%, with unchanged LDS instruction count. The floor sums being close to complete latency show that decode and WMMA/LDS work are already substantially overlapped.

## Accepted Experiments

### Shape-specific geometry and LDS ownership

- Unswizzled decoded-B LDS with `LdsPadB=8` replaced the unpadded layout. Padded `256x64` improved all three narrow keys by 13-17% over their prior selected bodies and improved attention-output M2048 by about 14%.
- Padded `128x64` improved attention-output M8192/M32768 by about 3% and query M8192 by about 4%. Shared-down and the other query keys retained the two-buffer `128x128` ownership after direct comparisons.
- WGM2 is retained only for narrow M2048; WGM1 is retained for the remaining shape-specific bodies.
- The four-M-tile `256x64` emitter received separate A-row, LDS, quant-shift, and Q4-specific address state. This corrected an early aliasing failure before production timing.

### Pipelining, traversal, and scheduling

- Corrected traversal to WGM1 reduced an early SIA3 path from about 119.36 ms to 52.40 ms. The later exact-key screens retained WGM2 only where the M2048 narrow geometry justified it.
- XOR-8 was selected before the padded campaign; the changed-premise padded winner is unswizzled `LdsPadB=8`. SIA4 and PGR2 were retained where their concrete A/B placement reduced latency; SIA5 was retained on the selected padded attention bodies.
- The true two-buffer decoded-B pipeline primes B0, decodes B1 while WMMA consumes B0, reloads next A after its last use, swaps once per steady tile, and peels the final tile to avoid out-of-range prefetch. Its historical K8192 control was 3.69% faster than one-buffer, and K512 was 5.97% faster.

### Lowering and instruction simplifications

The following changes remained bit-exact and resource-clean, with neutral-to-positive timing evidence:
- reuse prefetched A pointers, about 1.01% faster in the focused bracket;
- direct Q4 nibble extraction, removing 28 VALU instructions per reduction iteration and improving one control by 1.37%;
- invariant A-coordinate and packed-Q offset hoisting, with measured gains of about 0.65% and 0.44%;
- gfx11 scalar-base addressing for exact-shape A, packed-weight, and output accesses;
- compact address state, reducing allocation from 216 to 212 VGPRs;
- VOPD pairing for 20 accumulator clears and independent integer VALU;
- removal of `buffer_gl0_inv` after producer-handoff correctness checks;
- removal of unused runtime dimension loads while preserving the 40-byte kernarg ABI, reducing allocation from 20 to 16 SGPRs;
- exact-K trip specialization, including reduced K32/K64/K96/K512 checks;
- power-of-two row-stride shifts, increasing VOPD pairs from 20 to 22 without changing resources;
- removal of the fixed final `s_nop 7`; and
- one-time packed-dword normalization with `v_and_b32 0x0f0f0f0f` and `v_cvt_f32_ubyte0..3`, removing 40 N128 static VALU issues and improving the complete matrix by 1.50% over the no-final-NOP control.

Exact N coverage was also validated for N512, N2048, and N4096. The Q4 packed-row byte formula `(N/256)*144` produced resource-clean, bit-exact artifacts for the shared-down and attention-output shapes.

### DependencyBatch4 research identity

The typed `DependencyBatch4` identity groups the four independent Q4 nibble conversions for each packed dword. It keeps exact FP32 scale/minimum FMAs and BF16 RNE corrections, uses dead local-read storage for temporaries, and changes no VGPR, SGPR, LDS, ABI, wait, barrier, spill, or packed-representation resource.

The width-four candidate improved the current K512 parent by 6.10-6.25% and the current K8192 parent by 2.55-2.65% in independent alternating 20-warmup/25-repeat brackets. It was exact to HIP before and after gradient and packed-weight mutations, matched the independent-reference error, and rebuilt deterministically. The reopened width-two identity was superseded by width four; the earlier source-order dependency experiments were rejected on their original parents. The typed width-four identity is promoted for these two exact shapes, which is why its two fastest rows appear in the final table.

## Rejected Experiments

The rejected entries below retain only measurements that explain the decision. They are not alternate final kernels.

### LDS layouts and geometry

- XOR-16 was bit-exact and reduced allocation to 198 VGPRs, but measured 52.633 ms versus 45.006 ms for XOR-8, a 16.95% regression.
- XOR-4 retained the 68.75% conflict ratio but doubled LDS fragment loads and measured 45.178 ms versus 45.053 ms. The extra LDS issue count had no compensating gain.
- Unpadded `256x64x32` was bit-exact at 208 VGPRs and 4 KiB LDS, but measured 66.597 ms versus 45.300 ms, a 47.0% regression. The later padded body is a changed premise and is retained on narrow and attention-output M2048.
- `256x128x32` regressed 9.66% at 200 VGPRs and 8 KiB LDS; `64x128x32` regressed 68.5% at 136 VGPRs and 8 KiB LDS. Larger ownership duplicated work or lost residency.
- The early unpadded `128x64x32` body regressed 12.3% and is distinct from the later padded `128x64` winner. `128x128x64` gained only 0.93% while adding 20 VGPRs and doubling LDS.
- Dedicated decoder waves passed reduced correctness checks and used 212 VGPRs, 21 SGPRs, and 16 KiB LDS, but production K512 measured 2.7300 ms versus 2.4951 ms, a 9.41% regression.
- True two-buffer `64x128` and `128x64` alternatives passed reduced and exact-shape checks but regressed weighted timing by 71.95% and 4.39%. The selected two-buffer path is the separate `128x128` ownership regime.

### Scheduling, traversal, and prefetch

- PLR2 added 16 VGPRs and regressed 0.87%.
- The original SIA5 screen was only 0.93% faster than SIA4 and was below the retention threshold. Reopening SIA5 on the current K8192 parent produced candidate/parent movements of `+1.19%` and `+0.94%`; both brackets favored SIA4 or parity, so SIA5 was closed for that geometry.
- Packed-next-only prefetch was exact and resource-identical but gained only 0.15%.
- The original source-order two-value dependency batch improved by about 0.78%; the four-value version regressed K8192 by 0.62% and K512 by 1.57%. Those results were tied to retired parents and do not reject the later typed `DependencyBatch4` identity.
- Moving next-A0 loads to their first-half death point produced a mixed 0.87% regression and 0.61% gain. A-load `s_clause 3` pairs gained only 0.035% on K8192 and 0.29% on K512.
- Complete WGM2/WGM4/WGM8 matrices regressed weighted latency by 1.45%/5.22%/16.27% against WGM1. The later exact narrow M2048 WGM2 exception was retained only after a changed padded geometry screen.
- Store priority was neutral or worse on 11 exact keys. Query M2048's apparent 2.01% screen signal confirmed at only 1.45%, so it was rejected.
- Duff-style exact-trip unroll factors were bit-exact and resource-clean but gained at most 0.97%; larger source and code objects did not justify the change.

### Packed sharing, addressing, and metadata

- `PackedWeightLaneShare=2` halved packed-Q bytes but regressed the LDS crossbar by 1.15% and the DPP/VALU crossbar by 1.88%.
- Persistent A-pointer hoisting raised allocation from 200 to 204 VGPRs and regressed 0.23%. A persistent packed-row offset gained 1.37% on K512 but regressed K8192 by 0.09% while raising allocation from 212 to 213 VGPRs.
- Replacing the packed-row multiply with `v_lshl_add_u32` plus a shift gained only 0.61% on K512 and 0.45% on K8192.
- Coalescing each N128 metadata header into one 16-byte load gained only 0.07% on K512 and regressed K8192 by 0.28% after its wait budget was corrected.
- A strict owner-load/DPP8 metadata prototype reduced static VMEM from 164 to 152 but increased VALU issues from 865 to 895. Its candidate/control ratios were `1.00468x` at M8192 and `1.00926x` at M32768, so the emitter was removed after exact mutation checks.

### Epilogue and structural follow-ups

- An explicit `s_nop 0; s_sendmsg sendmsg(MSG_DEALLOC_VGPRS); s_endpgm` tail preserved correctness but regressed K512 by 1.71%, improved K8192 by only 0.45%, and regressed the pair by 1.32%. Direct `s_endpgm` remains selected.
- A combined padding-plus-logical-K XOR layout had no supporting profile premise. The selected M32768 query control showed 68.75% LDS bank conflict but only 0.13068% ALU-stalled-by-LDS, median LDS latency 241.664, and mean active-CU occupancy 27.8267. Conflict percentage alone did not identify a bottleneck.
- An exact-N half-buffer software pipeline was not admitted. The retained two-buffer path already beats the ordinary one-buffer K8192 control by 3.69%, and the lower-bound profile shows no occupancy or LDS-stall margin to repay extra handoff barriers.
- Literal/frontend specialization found the packed 144-byte block, 1152-byte row stride, 8192 reduction extent, peeled 8160 bound, LDS offsets, and launch divisors already encoded as immediates. No additional kernel emitter was justified.

## Deferred Kernel Experiments

### A1: Site-separated BF16 conversion policy

A1 is planned but unmeasured. The decoded-weight staging conversion and the output conversion are separate kernel sites and must be tested independently. `RNEPreserveNaN` remains the exact control; `BiasRound` and `Truncate` are approximate policies requiring finite-input error reporting and model-training validation. No approximate result can replace an exact kernel without a separate numerical contract.

### A2: Dependency-derived waits and LDS liveness

A2 is planned but unmeasured. It requires a typed producer/consumer event model for packed VMEM reads, decoded LDS writes, local reads, WMMAs, and LDS overwrite points. Identity mode must reproduce the selected source byte-for-byte. A candidate may replace a `vmcnt` or `lgkmcnt` wait only when the event model proves the threshold; barrier removal additionally requires a liveness proof that all cross-wave readers finish before LDS overwrite or exit.

The first targets are the current K512 and K8192 M32768 parent kernels, with geometry, LDS layout, prefetch, decode arithmetic, store policy, and packed representation fixed. A2 must remain exact to HIP and the independent reference, pass gradient and packed-weight mutations, preserve clean resources and deterministic builds, and show a stable greater-than-2% improvement before confirmation. `DependencyBatch4` is a decode dependency-batching identity, not the A2 wait/liveness model, and the two experiments must not be conflated.

## Kernel Correctness Notes

- gfx1151 packed work-item X/Y must be flattened before accumulator storage is assigned to `v0`.
- Assembly and metadata inspection did not catch the initial wrong reduction advance and hardcoded LDS row stride; execution checks exposed both faults.
- The true pipeline initially reused VGPRs holding later WMMA LDS addresses. Reduced K32/K64/K96 checks exposed the live-range corruption before production timing.
- Decoder-ID normalization depends on `v_sub_nc_u32` operand order; the reversed form caused a 32-bit address wrap in the dedicated-wave experiment.
- Hoisted address state must be scoped to the schedule that consumes it. Combining SIA4/5 precomputed offsets with SIA2/3 row reconstruction multiplied the row stride and corrupted a `256x64` output.

The current Q4_K kernel experiments have no further measured in-contract mechanism with a stable qualifying gain. Any new result must introduce a concrete changed premise and repeat the exact correctness, mutation, resource, reproducibility, and timing gates.
