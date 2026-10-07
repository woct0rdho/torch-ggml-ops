# GGTensile MMQ Forward Q6_K Kernel Report

## Final Results

This record covers the dense Q6_K forward kernel keys used by the workload.

The accepted kernel is the deterministic typed wavefront lowering.

`TFLOPS = 2*M*N*K / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Medians are the repository benchmark's, re-measured in the current clock state. They replace the earlier recorded values, which came from a different clock state with the same artifacts.

| Family | `(M,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| --- | --- | ---: | ---: | --- | --- |
| LM head | `(64,248320,2048)` | 16.492 | 0.8085x | `mmq_fwd_q6_k_m64_n248320_k2048_ef5c904640b7f2fe` | `dense_fwd_q6_k_k2048_j64_full_hoisted` |
| LM head | `(128,248320,2048)` | 17.470 | 0.7676x | `mmq_fwd_q6_k_m128_n248320_k2048_7f187e39bc287c24` | `dense_fwd_q6_k_k2048_j128_full_hoisted` |
| LM head | `(256,248320,2048)` | 17.464 | 0.7678x | `mmq_fwd_q6_k_m256_n248320_k2048_91905def3e2625cc` | `dense_fwd_q6_k_k2048_j128_full_hoisted` |

GGTensile's own throughput is unchanged from the earlier recording. The HIP column is the deployed `_hoisted` body, which is `1.26-1.31x` faster than the HIP multiply body this record was originally paired against, so HIP now leads on all three shapes.

## Accepted Experiment Logs

### Canonical Q6 wavefront schedule

The retained schedule directly decodes the 210-byte Q6_K block, preserves integer WMMA accumulation and signed int8 scale application, then applies the Q6 block factor, Q8_1 activation scale, and exact BF16 `RNEPreserveNaN` output sequence.

M64 owns one J64 row tile. M128 owns one J128 row tile. M256 uses two exact J128 row tiles with the same physical profile. The selected semantic policy is output-role wavefront traversal, row-batched decode order, explicit dependency distance, producer-first waits, explicit role lifetimes, and dependency-compatible dual issue. M64 has no delay hints. M128 and M256 retain the selected dependency-delay schedule.

The M256 ownership defect was fixed by deriving the weight-column owner as `wave & 3` and applying the 128-row offset only to the upper output half. A padded-canary check and the unpadded full-shape run then completed without missing or out-of-bounds writes.

### Exact arithmetic and output pipeline

The selected arithmetic order is retained. The BF16 output pipeline is also retained: the resource-neutral rewrite preserved the selected VGPR, SGPR, LDS, and zero-spill profile. M64 retained its measured delay-hint removal. M128 and M256 retain the full-tile BF16 dependency scope needed by the exact output sequence.

## Rejected Experiment Logs

### Raw-payload and shared-ownership representations

The compact raw-payload representation staged raw `ql`, raw `qh`, signed metadata, and the block factor in a 224-byte row. It was exact and resource-clean, but reduced LDS enough to add resident workgroups while forcing reconstruction in each consumer group. M64 changed from `4.628572 ms` to `5.447662 ms` (`0.849645x` parent throughput), and M128 changed from `7.987497 ms` to `8.688813 ms` (`0.919286x`). It was rejected. M256 did not satisfy the transfer gate.

The expanded-scale M256 representation was exact and used `57,600 B` LDS with zero spills. It reduced LDS instructions but moved only from `15.589506 ms` to `15.505320 ms` (`1.005429x` parent speedup) and remained behind HIP at `0.958739x`. The resource-bearing greater-than-2% gate was not met.

The regenerated shared-MT256 ownership was exact, deterministic, and resource-clean at `210 VGPR`, `27 SGPR`, and `57,344 B` LDS. Its nine-repeat median was `15.2952 ms`, versus `14.8034 ms` for the selected two-row-tile parent and `14.8658 ms` for HIP. It was `1.0332x` slower than its parent and was rejected without 25-repeat promotion.

### LDS, readiness, and dot-frontier variants

Paired activation-scale reads and paired decoded stores remained exact, but the combined screen was shape-inconsistent at `1.016x`, `1.003x`, and `0.998x` of the decoded parent for M64, M128, and M256. The candidate was rejected as noise-scale.

Fine-grained producer waits, decode/store interleaving, factor-read frontiers, and setup VOPD unpairing all remained exact and resource-clean. Their best screens were at most sub-percent effects. Confirmations reversed the apparent gains. In particular, setup unpairing moved from screen ratios below one to confirmation ratios of `1.001720x` and `1.001719x`, while the broad form measured `1.000863x` and `0.998932x`.

Dot-frontier variants also remained exact. The 25-repeat confirmations reduced the best M64 and M128 parent ratios to `0.997820x` and `0.997778x`, below the material-gain gate and not stable enough to retain as a separate policy. Alternate factor ordering, J128 delay removal, compact geometries, dedicated decoder waves, and low-register-only controls likewise produced no repeatable gain that survived the fixed resource and exactness gates.

### Wide scalar-carry frontier

`WideScalarCarryFrontier`, kernel hash `ggsol_19d7aac957efe95f`, assigned fixed SGPR carry roles to batches of 64-bit addresses. It matched HIP and the selected parent exactly, used `158 VGPR`, `33 SGPR`, and `28,928 B` LDS, and had zero private bytes and spills.

The nine-repeat candidate/parent ratio was `0.998895x`, too close to noise for promotion. The two 25-repeat confirmations measured candidate/parent ratios of approximately `1.000061x` and `0.999531x`, then `0.999744x` and `1.000887x`. The candidate was rejected as parity/noise rather than transferred to another shape. Evidence is under `q6-wide-scalar-carry-current-v1/`.

### Offline issue-schedule oracle

Offline compiler schedules demonstrated headroom but did not identify a bounded typed policy. The strongest exploratory controls reached approximately `0.939x` candidate/HIP latency for J64 and `0.955x` for J128, while their useful behavior depended on whole-stream physical allocation, cross-stage issue selection, and register lifetimes.

Kernel-only counters support that conclusion: typed and oracle bodies had identical LDS volume and bank-conflict counts, while the oracle used more VALU operations and VALU issue cycles. Reproducing the remaining difference would require a new physical register map and whole-DAG scheduler or a copied instruction stream. Those mechanisms are outside this kernel's fixed typed lowering contract and were not retained.

### BF16 rounding alternatives

`BiasRound` was faster but not exact. Differing elements were 135, 236, and 515 for M64, M128, and M256. Maximum absolute error was `0.015625`, with normalized RMSE from `1.543e-5` to `1.586e-5`.

`Truncate` changed approximately half of the outputs. Its normalized RMSE was approximately `0.00405`, maximum absolute error was `0.03125`, and differing elements were 7,934,498, 15,869,873, and 31,741,799. Both alternatives were rejected by the exact output contract.

## Deferred and Contract-Excluded Ideas

Effective-scale reassociation was not implemented or timed. It changes the current `Int32ScaleF32` plus signed-Q8 correction arithmetic contract. A future exact experiment requires a new serialized arithmetic contract, a formula-derived register and lifetime plan, and independent numerical and quality gates before any timing result can be considered.

Prepared weights, external decode workspaces, producer fusion, split-K or Stream-K traversal, persistent or grouped traversal, ABI changes, and arbitrary recurrent permutations are outside the fixed kernel contract. NaN-path simplification for arbitrary packed inputs is also excluded. No source change is justified for these ideas.

The retained evidence leaves no additional exact in-contract experiment with a demonstrated material-gain path. The selected hashes and profiles above are therefore the final kernel results for this report.
