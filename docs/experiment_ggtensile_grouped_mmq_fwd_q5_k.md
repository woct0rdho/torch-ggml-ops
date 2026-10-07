# GGTensile Grouped MMQ Forward Q5_K Experiment

## Scope

This record covers the routed gfx1151 Q5_K down projection, one GEMM per routed expert.

## Final Results

`TFLOPS = 2*R*N*K / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Each row is the median over the benchmark's sampled expert partitions, and both implementations are timed on the same partitions.

| Batch | Logical shape `(R,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `(16384,2048,512)` | 17.372 | 0.9726x | `grouped_mmq_fwd_q5_k_r16384_n2048_k512_03954bfadeddff04` | `grouped_fwd_serial_q5_k_n2048_k512_j32` |
| 4 | `(65536,2048,512)` | 23.059 | 1.0365x | `grouped_mmq_fwd_q5_k_r65536_n2048_k512_ff24d8b5206a10b6` | `grouped_fwd_serial_q5_k_n2048_k512_j64` |
| 16 | `(262144,2048,512)` | 24.429 | 0.9956x | `grouped_mmq_fwd_q5_k_r262144_n2048_k512_05410d219cb70dc0` | `grouped_fwd_serial_q5_k_n2048_k512_j64` |

GGTensile is ahead on 1 of the 3 rows, with speedups from `0.9726x` to `1.0365x` (mean `1.0016x`), and within 0.5% of parity on the remaining 1.

These medians replace the earlier recorded values, which were paired against HIP bodies that have since been retuned.

## Accepted Experiments

### Shared Q5_K high-bit reconstruction

Q5_K shares the Q4_K scale/minimum arithmetic, Q8_1 activation staging, signed integer WMMA, correction, and BF16 stores. Its distinct cost is high-bit reconstruction. The typed decoded-LDS emitter loads `qh`, shifts lane-selected bit masks, merges bit 4 into both low-nibble halves, and releases the high-bit payload before the WMMA loop. This decoder is used by every retained identity.

### 64-row parent with 32-row residual at B1

The 128-row bodies were rejected as broad selectors: serialized, metadata-scheduled, and `a1d2-p2` forms all used 239 VGPRs and 38,400 LDS bytes and reached only `0.650x`/`0.679x`/`0.683x` at B1, `0.954x`/`0.987x`/`0.995x` at B4, and `0.945x`/`0.969x`/`0.981x` at B16 installed/candidate.

The Q4-qualified 64-row layout transfers without a new register plan at 159 VGPRs, 40 SGPRs, 29,184 LDS bytes, 16 static WMMAs, and four barriers. Its scheduled `a1d4-p2` form reached `0.910x`, `1.010x`, and `0.946x` at B1/B4/B16. Adding a mutually exclusive 32-row tail body, which raises the static WMMA count to 24 while retaining the same resources, reached `1.000x` at B1 and `1.036x` at B4. The B1 confirmation brackets weighted parity at `1.0020x` and `0.9987x`. Low-weight medoids regress by up to about 4-6% and the uniform synthetic route is `0.955x`, so the identity is retained as a fitted-prior control.

### Three-way 128/64/32 final-tile ownership at B4/B16

A Q5-specific final-tile policy emits mutually exclusive 128-, 64-, and 32-row activation, MMA, correction, and store bodies at 239 VGPRs, 40 SGPRs, 38,400 LDS bytes, 56 static WMMAs, and four barriers. A 388-row route containing 17-, 48-, 128-, and 195-row groups exercises all three branches and matches installed HIP bitwise.

This is the material B4 result. The first nine-repeat screen reached `1.0636x`, and two independent 25-repeat confirmations reached `1.0597x` and `1.0770x`, with every medoid at least `1.0274x` and `1.0360x`. The full `a1d4-p2` identity also reaches `1.0015x` and `1.0039x` at B16 with every confirmation medoid above parity. A final complete-call audit with the fixed quantizer measured `1.0539x`, `1.0813x`, and `1.0070x` installed/candidate and preserved the B4 gain.

## Rejected Experiments

### True 32-row ownership

A dedicated two-fragment 32-row layout reduced resources to 135 VGPRs and 24,576 LDS bytes with eight static WMMAs and bitwise-exact output, but B1 fell to `0.9872x` weighted with four medoids at `0.896x` to `0.916x`. Lower residency cannot repay decoding each Q5 high-bit payload twice as often, so the mechanism was reverted and only the shorter residual body inside the 64-row parent is retained.

### Grouped VOPD accumulator initialization

The dense Q5 resource-neutral VOPD accumulator initialization was exact and changed scalar-parent weighted timing by `1.0023x`, `0.9920x`, and `1.0019x` at B1/B4/B16. The movements are sub-percent and inconsistent, with the material B4 key regressing, so the grouped VOPD path was reverted.

### Routed-prologue and address reductions

The shared packed-kernarg, paired cumulative-offset, and guarded high-stride/address reductions rebuilt deterministically, preserved the selected parent's resources, and matched the parent bitwise on all route controls. The Q4_K B1 transfer screen measured `+0.030%`, `-0.086%`, and `-0.083%` candidate time, none clearing the gate. Because Q5_K performs the same route setup around a longer high-bit decode body, the prespecified upper-bound gate rejects production timing transfer and the mechanisms are closed as neutral.

### Contract-incompatible and deferred mechanisms

Host route readback, hidden caches, dense weight shadows, producer fusion, and online tuning remain contract-incompatible. Flattened or persistent routing requires a profile showing launch imbalance after the existing output-column parallelism, SplitK requires enough K-loop underutilization to repay partial-output storage at fixed `K=512`, and a route-persistent full-K decoded image remains unmeasured. The analogous Q4 body regressed about 12% on its strongest uniform case and the Q5 form requires the same 57,856-byte LDS class.

## Open Items

The Q5-specific R1-R3 route and address transfer may be re-run against the current B1 parent. The route-persistent full-K image is not reopened.

## Closure

The retained result is the shared Q5 high-bit decoder with padded single-LDS storage: the 64-row parent plus 32-row residual at B1 and the three-way 128/64/32 final-tile body at B4/B16, all with serial route ownership.
