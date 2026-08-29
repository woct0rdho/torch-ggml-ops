# GGTensile MMQ Forward Q8_0 Experiment

## Scope And Method

This record covers dense Q8_0 matrix multiplication kernels for gfx1151. The kernel consumes authoritative packed Q8_0 weights and an upstream Q8_1 F32_D4 activation workspace, performs signed integer WMMA with FP32 scale correction, and stores BF16 output. The contract is exact shape and exact kernel identity: wave32, WMMA V1, code object v5, and the existing 40-byte forward ABI.

The fixed Q8_1 producer runs before every multiply measurement and is outside the timed interval. HIP and GGTensile multiply kernels consume the same packed weights and the same prepared workspace. Throughput is `2*M*N*K/(median_ms*1e9)`. The speedup ratio is `HIP median / GGTensile median`; values above `1.0x` favor GGTensile. The final table uses the final repeatable audit for the fastest retained kernel at each exact shape. The two audit passes showed no notable inconsistency, with the largest per-key difference below one percent.

Every retained and screened executable candidate was required to pass exact HIP output comparison, independent packed/dequantized reference checks, finiteness, input/weight/workspace mutation checks, repeatability, strict code-object inspection, zero private storage, zero register spills, and deterministic rebuilds. No fixed speed threshold is used: a candidate is retained only when the timing evidence shows it is faster than the exact parent and HIP control.

## Fastest Kernels Found

The final exact-key set contains 23 matrix shapes. The selected kernel identities are 20 `CompactDepth32WeightRows` kernels, one `HipTile` kernel for Q-A M2048, and two `SmallMTile` kernels for LM-head M32 and M64.

| Family | Matrix shape `(M,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | --- | ---: | ---: |
| Attention Q-A | `(2048,1024,4096)` | `ggsol_620516f662da29f3` | `34.245` | `1.1034x` |
| Attention Q-A | `(8192,1024,4096)` | `ggsol_f78e834f7bb82a2e` | `36.074` | `1.1704x` |
| Attention Q-A | `(32768,1024,4096)` | `ggsol_1389b8104bc29061` | `36.974` | `1.1511x` |
| Attention Q-B | `(2048,32768,1024)` | `ggsol_505d33dd36cd5790` | `34.092` | `1.1918x` |
| Attention Q-B | `(8192,32768,1024)` | `ggsol_323c022d43fb2435` | `34.641` | `1.1995x` |
| Attention Q-B | `(32768,32768,1024)` | `ggsol_f15bc6c956ce7c52` | `35.059` | `1.2077x` |
| Attention K/V | `(2048,512,4096)` | `ggsol_b40f08e30ff5bc83` | `30.285` | `1.0584x` |
| Attention K/V | `(8192,512,4096)` | `ggsol_de612ce7a470579d` | `34.944` | `1.1411x` |
| Attention K/V | `(32768,512,4096)` | `ggsol_0898037fe8783040` | `36.766` | `1.1630x` |
| Attention output B | `(2048,4096,8192)` | `ggsol_eac11aed4c3b4229` | `36.512` | `1.2463x` |
| Attention output B | `(8192,4096,8192)` | `ggsol_6525bb03932d9c84` | `37.513` | `1.2348x` |
| Attention output B | `(32768,4096,8192)` | `ggsol_da2d79e7129cd5d7` | `37.938` | `1.2344x` |
| Shared gate/up | `(2048,2048,4096)` | `ggsol_abdcb1bf8cb8e28d` | `35.066` | `1.1287x` |
| Shared gate/up | `(8192,2048,4096)` | `ggsol_b4500bc7bd68b338` | `36.748` | `1.1502x` |
| Shared gate/up | `(32768,2048,4096)` | `ggsol_64f2005dfe00adb8` | `37.628` | `1.1585x` |
| Shared down | `(2048,4096,2048)` | `ggsol_e9250a27a783dd49` | `34.968` | `1.1462x` |
| Shared down | `(8192,4096,2048)` | `ggsol_253f1a95ce85cd9e` | `36.326` | `1.1417x` |
| Shared down | `(32768,4096,2048)` | `ggsol_f3d23f4df3973903` | `37.049` | `1.1495x` |
| LM head | `(32,129280,4096)` | `ggsol_b9b1167f6cbe898a` | `12.922` | `1.0455x` |
| LM head | `(64,129280,4096)` | `ggsol_ba13b32d0f3ff9ec` | `24.540` | `1.0197x` |
| LM head | `(128,129280,4096)` | `ggsol_796295ce34f4992d` | `36.335` | `1.2370x` |
| LM head | `(256,129280,4096)` | `ggsol_0da11f24c6cb2f0c` | `36.587` | `1.2341x` |
| LM head | `(512,129280,4096)` | `ggsol_ee38be695060a665` | `37.203` | `1.2419x` |

## Final Kernel Profiles

All final kernels use code object v5, gfx1151, wave32, the 40-byte ABI, zero private storage, and zero VGPR/SGPR spills. Static instruction counts are from strict inspection of the selected artifacts.

| Kernel identity | Exact-key coverage | Workgroup / macro tile / DepthU | VGPR / SGPR / LDS bytes | WMMAs / VMEM / LDS ops | Waits / clauses / barriers |
| --- | --- | --- | --- | --- | --- |
| `CompactDepth32WeightRows` | 20 keys | `32x4x1 / 128x64 / 32` | `240 / 16 / 27,648` | `64 / 82 / 138` | `40 / 3 / 2` |
| `HipTile` | Q-A `(2048,1024,4096)` | `32x4x1 / 128x64 / 32` | `240 / 16 / 38,400` | `64 / 82 / 154` | `40 / 3 / 2` |
| `SmallMTile` M32 | LM head `(32,129280,4096)` | `32x4x1 / 32x64 / 32` | `96 / 16 / 24,064` | `16 / 25 / 73` | `13 / 3 / 2` |
| `SmallMTile` M64 | LM head `(64,129280,4096)` | `32x4x1 / 64x64 / 32` | `144 / 16 / 28,672` | `32 / 44 / 100` | `22 / 3 / 2` |

`CompactDepth32WeightRows` stores 128 activation rows with a 144-byte row stride and 64 weight rows with the same stride. Its activation plane is 18,432 bytes, its weight plane is 9,216 bytes, and weight scales begin at byte 128. Weight-first staging, direct paired weight-scale reads, and an invariant second scale base make the compact layout legal and reduce LDS pressure without changing the WMMA correction order. The Q8_0 independent-reference normalized RMSE values recorded for representative compact keys are approximately `0.00602` to `0.00606`.

`HipTile` retains the standard LDS row layout and the activation-read-address hoist. The hoist computes the lane-local activation row address once and uses dependency-correct waits for the paired activation-scale copies. Its final Q-A M2048 control passed two warmed confirmations at approximately `1.10x` HIP multiply speed; an earlier apparent deficit was unresolved measurement noise rather than a kernel regression.

The two `SmallMTile` kernels use the same wave-N ownership with M-specific activation staging. M32 uses 16 WMMAs and M64 uses 32. Both are exact under the Q8 arithmetic contract and remain within their declared resource classes. Their independent-reference normalized RMSE values are `0.006066967333` and `0.006039657922`.

## Experiment Log: Accepted

- The signed-int8 Q8_0 lowering is accepted as the arithmetic base. It consumes packed Q8_0 payloads directly, consumes the fixed Q8_1 F32_D4 workspace, applies `integer_result * weight_scale * activation_scale` in the established order, and stores BF16 with the required rounding behavior. Direct and LDS-backed forms remained exact under the full mutation and reference checks.
- The balanced `2x2` register-tiled ownership, linear store traversal, materialized store addresses, and simplified store clause were accepted as intermediate exact controls. They established the final signed-int8 register ownership and arithmetic schedule, but the compact LDS composition later replaced them on the exact keys where it was faster.
- The activation-read-address hoist was accepted after its wait schedule was corrected. The first version failed on odd M-fragment activation scales; the corrected version remained exact and became the final `HipTile` Q-A M2048 kernel.
- The compact depth32 composition was accepted for 20 exact keys. It combines the 144-byte depth32 rows, weight-first staging, legal paired weight-scale reads, and invariant paired-scale addressing. Every retained exact key was rebuilt and requalified after the composition changed the parent.
- The M32 and M64 small-M ownerships were accepted for LM-head M32 and M64 after independent warmed confirmations, exact output checks, mutation checks, and deterministic rebuilds. The M64 compact ownership was also a useful exact KV control, but it was superseded by the final exact-key composition and is not a final identity.
- The final qualification pass covered representative Q-B M8192, attention-output B M8192, and LM-head M128 kernels. Nine width/layout candidates were finite, repeatable, exact against the independent reference, exact against the parent and HIP, sensitive to input/weight/workspace mutations, and exact through the normal kernel path. Deterministic rebuilding matched source, object, code-object, and inspection results for all 28 admissible artifacts.

## Experiment Log: Rejected

### Ownership, Geometry, And Staging

- The direct-global kernel and the first LDS-free register-tiled geometries were correct but slower than HIP. The `1x4`, `2x2`, and `4x1` register geometries did not provide a stable exact-key improvement; the balanced `2x2` form was retained only as an intermediate parent.
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

The accepted Q8_0 forward kernels are the four profiles in the final profile table, selected per exact matrix shape. The final speed table is multiply-only and reports the fastest retained identity found during the complete kernel experiments. All width and padding candidates are rejected; no actionable in-contract ownership, staging, scale, scheduling, or transaction-width mechanism remains from the reviewed search space.

The implementation and regression coverage are in the typed Q8 search, specification validation, signed-int8 lowering, and writer tests. Detailed temporary build, inspection, timing, gate, and deterministic-rebuild records are under `~/tmp/torch-ggml-ops/q8-payload-reopening/`.
