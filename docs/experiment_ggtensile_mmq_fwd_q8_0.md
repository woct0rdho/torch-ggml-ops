# GGTensile MMQ Forward Q8_0 Experiment

## Final Results

This record covers dense Q8_0 forward kernels for gfx1151:

```text
output[M,N] = input[M,K] @ dequant_q8_0(weight[N,K]).T
```

`TFLOPS = 2*M*N*K / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Medians are the repository benchmark's, re-measured in the current clock state. They replace the earlier recorded values, which came from a different clock state with the same artifacts.

| Family | `(M,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| --- | --- | ---: | ---: | --- | --- |
| Attention Q-A | `(2048,1024,4096)` | 32.327 | 1.1142x | `mmq_fwd_q8_0_m2048_n1024_k4096_ac9f6ba25864f335` | `dense_fwd_q8_0_k4096_j128_full` |
| Attention Q-A | `(8192,1024,4096)` | 33.075 | 1.1399x | `mmq_fwd_q8_0_m8192_n1024_k4096_c3b72b29ea9cdb96` | `dense_fwd_q8_0_k4096_j128_full` |
| Attention Q-A | `(32768,1024,4096)` | 33.025 | 1.1056x | `mmq_fwd_q8_0_m32768_n1024_k4096_7204a17cfb45ad4a` | `dense_fwd_q8_0_k4096_j128_full` |
| Attention Q-B | `(2048,32768,1024)` | 30.859 | 1.1385x | `mmq_fwd_q8_0_m2048_n32768_k1024_f7c942492cb08af8` | `dense_fwd_q8_0_k1024_j128_full` |
| Attention Q-B | `(8192,32768,1024)` | 30.843 | 1.1335x | `mmq_fwd_q8_0_m8192_n32768_k1024_399bb3eba1cde513` | `dense_fwd_q8_0_k1024_j128_full` |
| Attention Q-B | `(32768,32768,1024)` | 29.443 | 1.1338x | `mmq_fwd_q8_0_m32768_n32768_k1024_54fd9c6683513603` | `dense_fwd_q8_0_k1024_j128_full` |
| Attention K/V | `(2048,512,4096)` | 29.205 | 1.0481x | `mmq_fwd_q8_0_m2048_n512_k4096_749eaab54e3b1379` | `dense_fwd_q8_0_k4096_j128_full` |
| Attention K/V | `(8192,512,4096)` | 32.737 | 1.1382x | `mmq_fwd_q8_0_m8192_n512_k4096_e82d066d8f538884` | `dense_fwd_q8_0_k4096_j128_full` |
| Attention K/V | `(32768,512,4096)` | 32.788 | 1.1211x | `mmq_fwd_q8_0_m32768_n512_k4096_793eb166548cd497` | `dense_fwd_q8_0_k4096_j128_full` |
| Attention output B | `(2048,4096,8192)` | 32.420 | 1.1763x | `mmq_fwd_q8_0_m2048_n4096_k8192_842d264ebf9533e3` | `dense_fwd_q8_0_k8192_j128_full` |
| Attention output B | `(8192,4096,8192)` | 32.663 | 1.1502x | `mmq_fwd_q8_0_m8192_n4096_k8192_43ecfa00ae6cb264` | `dense_fwd_q8_0_k8192_j128_full` |
| Attention output B | `(32768,4096,8192)` | 31.314 | 1.1465x | `mmq_fwd_q8_0_m32768_n4096_k8192_8ae04c3a600a30c6` | `dense_fwd_q8_0_k8192_j128_full` |
| Shared gate/up | `(2048,2048,4096)` | 32.540 | 1.1252x | `mmq_fwd_q8_0_m2048_n2048_k4096_4a9dab3b39257cb6` | `dense_fwd_q8_0_k4096_j128_full` |
| Shared gate/up | `(8192,2048,4096)` | 32.647 | 1.1197x | `mmq_fwd_q8_0_m8192_n2048_k4096_d964370550753fe6` | `dense_fwd_q8_0_k4096_j128_full` |
| Shared gate/up | `(32768,2048,4096)` | 32.533 | 1.1141x | `mmq_fwd_q8_0_m32768_n2048_k4096_20568015e2e2c57a` | `dense_fwd_q8_0_k4096_j128_full` |
| Shared down | `(2048,4096,2048)` | 32.429 | 1.1432x | `mmq_fwd_q8_0_m2048_n4096_k2048_17fcf11967cf2c59` | `dense_fwd_q8_0_k2048_j128_full` |
| Shared down | `(8192,4096,2048)` | 32.368 | 1.1105x | `mmq_fwd_q8_0_m8192_n4096_k2048_38dc14e5687d64fc` | `dense_fwd_q8_0_k2048_j128_full` |
| Shared down | `(32768,4096,2048)` | 32.160 | 1.1121x | `mmq_fwd_q8_0_m32768_n4096_k2048_ae2a7e90efb21145` | `dense_fwd_q8_0_k2048_j128_full` |
| LM head | `(32,129280,4096)` | 13.329 | 1.1108x | `mmq_fwd_q8_0_m32_n129280_k4096_2ecfddcfa3d28f81` | `dense_fwd_q8_0_k4096_j64_bounded` |
| LM head | `(64,129280,4096)` | 25.747 | 1.0979x | `mmq_fwd_q8_0_m64_n129280_k4096_18ea51a79693a48d` | `dense_fwd_q8_0_k4096_j64_full` |
| LM head | `(128,129280,4096)` | 32.266 | 1.1600x | `mmq_fwd_q8_0_m128_n129280_k4096_b1e120938911d11d` | `dense_fwd_q8_0_k4096_j128_full` |
| LM head | `(256,129280,4096)` | 32.412 | 1.1585x | `mmq_fwd_q8_0_m256_n129280_k4096_242054725717ca9a` | `dense_fwd_q8_0_k4096_j128_full` |
| LM head | `(512,129280,4096)` | 32.216 | 1.1575x | `mmq_fwd_q8_0_m512_n129280_k4096_be542889b65a26ca` | `dense_fwd_q8_0_k4096_j128_full` |

GGTensile is ahead on all 23 entries, by `1.048x` to `1.176x` (mean `1.129x`). The selected identities are 20 `CompactDepth32WeightRows` kernels, one `HipTile` kernel for Q-A M2048, and two `SmallMTile` kernels for LM-head M32 and M64.

These kernels are the clock-sensitive ones in this record set. With the benchmark's real weight and activation data the APU sustains about `2.53 GHz` sclk, against about `2.80 GHz` for low-switching data, so the same artifacts read between 5% lower and 21% higher in the earlier recordings depending on the key. The HIP multiply body is insensitive to this and reproduces its own record, which is why the recorded lead over HIP shrinks here.

## Experiment Log: Accepted

- The signed-int8 Q8_0 lowering is accepted as the arithmetic base.
- The balanced `2x2` register-tiled ownership, linear store traversal, materialized store addresses, and simplified store clause were accepted as intermediate exact controls. They established the final signed-int8 register ownership and arithmetic schedule, but the compact LDS composition later replaced them on the exact keys where it was faster.
- The activation-read-address hoist was accepted after its wait schedule was corrected. The first version failed on odd M-fragment activation scales. The corrected version became the final `HipTile` Q-A M2048 kernel.
- The compact depth32 composition was accepted for 20 exact keys. It combines the 144-byte depth32 rows, weight-first staging, legal paired weight-scale reads, and invariant paired-scale addressing. Every retained exact key was rebuilt and requalified after the composition changed the parent.
- The M32 and M64 small-M ownerships were accepted for LM-head M32 and M64 after independent warmed confirmations. The M64 compact ownership was also a useful exact KV control, but it was superseded by the final exact-key composition and is not a final identity.

## Experiment Log: Rejected

### Ownership, Geometry, And Staging

- The direct-global kernel and the first LDS-free register-tiled geometries were correct but slower than HIP. The `1x4`, `2x2`, and `4x1` register geometries did not provide a stable exact-key improvement. The balanced `2x2` form was retained only as an intermediate parent.
- Cooperative raw-weight LDS staging, wider `128x64` and `64x64` ownership, clustered ownership, generic producer ownership, and double-buffered or reordered staging added LDS traffic, barriers, or resource pressure without a stable multiply gain. These forms are not part of the final domain.
- `DepthU=64` reduced loop count but required an additional barrier for the current ownership and doubled the reduction body. Persistent-zero depth32/depth64 variants were exact, but their improvements were shape-specific and did not establish a stable parent improvement. Both were rejected as final identities.
- Terminal-LDS-barrier elision, loop-carried staging addresses, and other address-hoist variants were either shape-specific, unstable across parent rotations, or neutral within measurement noise. The strongest address-hoist probe improved one long-K shape while regressing Q-B, so it was rejected.

### Scale, Scheduling, And Register Experiments

- Paired activation-scale reads, scalar-scale reordering, WMMA batching, scale-conversion movement, next-zero overlap, store-priority changes, and payload prefetch schedules were exact when their dependencies were correct but neutral or slower than the retained parent. Old wait thresholds and a direct same-scale VOPD copy also produced correctness failures and were removed.
- Transposed weight-scale planes reduced static LDS instructions but required extra LDS space and address work without a multiply gain. The 608-byte standard scale spacing cannot be represented by the compact paired-read encoding without a changed scale plane.
- Two 128-VGPR pressure experiments reduced declared registers but duplicated scale extraction or added address dependencies. Both were exact and spill-free, yet slower than the parent. Register pressure reduction is not retained as an objective.
- The exact KV M32 ownership was slower than compact M64. KV terminal-barrier and padded-row variants produced contradictory or sub-percent parent movement, so neither became an exact-key specialization. Extra 160-, 176-, 192-byte row padding and an eight-wave N128 tile likewise provided no stable multiply improvement.

### Compact Depth32 Width And Padding Reopening

- Linked payload width `8` changed both global payload reads and LDS writes and added four VMEM operations and two LDS writes per stage. It remained exact, but candidate/parent ratios were `1.0142x`, `1.0102x`, `1.0142x`, and `1.0075x` on the large Q-B/output-B keys and `1.0251x`, `1.0128x`, and `1.0138x` on LM M128/M256/M512. It is rejected.
- Linked payload width `4` added twelve VMEM operations and six LDS writes per stage. It remained exact, but candidate/parent ratios were `1.0388x`, `1.0327x`, `1.0302x`, and `1.0285x` on the large Q-B/output-B keys and `1.0372x`, `1.0437x`, and `1.0364x` on LM M128/M256/M512. It is rejected.
- Activation pad-A initially exposed a lowering defect: the padded 148-byte LDS stride was incorrectly used for the fixed global Q8_1 workspace. After separating the global 144-byte stride from the LDS stride, all candidates were exact. The corrected pad-A candidates were nevertheless much slower, at `2.2643x`, `2.2530x`, `2.4204x`, and `2.3403x` of the parent on the large keys and `2.4966x`, `2.4147x`, and `2.3946x` on LM M128/M256/M512. Pad-A is rejected.
- Positive pad-B is contract-incompatible for the compact paired-scale layout. It produces paired LDS offsets `256` and `257`, beyond the gfx11 8-bit offset encoding, and is rejected before assembly. The typed validation and writer regression test preserve this rejection.

## Final Disposition

The accepted Q8_0 forward kernels are the four identities in the speed table, selected per exact matrix shape. All width and padding candidates are rejected. No actionable in-contract ownership, staging, scale, scheduling, or transaction-width mechanism remains from the reviewed search space.

The implementation and regression coverage are in the typed Q8 search, specification validation, signed-int8 lowering, and writer tests. Detailed temporary build, inspection, timing, gate, and deterministic-rebuild records are under `~/tmp/torch-ggml-ops/q8-payload-reopening/`.
