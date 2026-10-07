# GGTensile Grouped MMQ Backward Pair Q3_K Experiment

## Scope

This record covers the routed gfx1151 paired Q3_K input gradients for the Qwen gate and up projections, accumulated in one workgroup dataflow.

## Final Results

`TFLOPS = 4*R*N*K / (median_ms * 1e9)` (both projections), and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Each row is the median over the benchmark's sampled expert partitions, and both implementations are timed on the same partitions.

| Batch | Logical shape `(R,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `(16384,512,2048)` | 13.043 | 0.8588x | `grouped_mmq_bwd_pair_q3_k_r16384_n512_k2048_0cf06f8bc3cbdcc8` | `grouped_bwd_pair_task_q3_k_n512_k2048_mt128_nt64_s2_skip_abar` |
| 4 | `(65536,512,2048)` | 20.924 | 0.8439x | `grouped_mmq_bwd_pair_q3_k_r65536_n512_k2048_adbd49ecb4ebc02b` | `grouped_bwd_pair_task_q3_k_n512_k2048_mt128_nt64_s2_skip_abar` |
| 16 | `(262144,512,2048)` | 22.126 | 0.7295x | `grouped_mmq_bwd_pair_q3_k_r262144_n512_k2048_2d9b8bd8e73ac248` | `grouped_bwd_pair_task_q3_k_n512_k2048_mt256_nt64_s2_skip` |

HIP is ahead on all 3 rows. GGTensile is at `0.7295x` to `0.8588x` (mean `0.8107x`).

These medians replace the earlier recorded values, which were paired against HIP bodies that have since been retuned.

## Accepted Experiments

### Fused single-LDS anchors and geometry baseline

The paired problem and contract admit only the exact Qwen Q3_K geometry. Canonical M64/N64 and M128/N64 identities use pad8, packed extraction, explicit full Q3 VOPD pairing, K32, serial routes, and the interleaved projection schedule. M64 inspects at `87 VGPR / 41 SGPR / 5,120 B` LDS with 16 WMMAs and four barriers. M128 at `127 / 41 / 5,120 B` with 32 WMMAs and four barriers.

The fitted baseline found M64 effectively tied with HIP at B1 (`5.3885` versus `5.3661 ms`, `0.9958x`) but rejected at B4/B16, where it reached only `0.8290x` and `0.7586x`. M128 won B4 at `1.0176x` but trailed B16 at `0.9247x`, so the B16 deficit became the first optimization target. The first control probe also used the wrong dynamic LDS allocation for the installed J64 control (29,952 instead of 30,976 bytes) and differed in 356 BF16 values. Correcting only the control restored bitwise agreement.

### Dual padded LDS and full-tile split

The first paired-specific schedule assigned the two decoded Q3_K projections to disjoint 5 KiB padded LDS images, decoded both banks before one consumer barrier, and synchronized once before overwrite. M128 stayed at `127 VGPR / 41 SGPR`, LDS rose to `10,240 B`, barriers fell from four to two, and B16 improved from `52.8582` to `51.8399 ms` (`1.0196x`) while remaining behind contemporaneous HIP.

Splitting each route into an unbounded M128 full-tile body plus a bounded tail body kept the same resources and made the full-tile split the new B16 parent, improving dual LDS from `52.3253` to `50.0691 ms` (`1.0451x`) and reaching `0.9706x` contemporaneous HIP. Keeping the second gradient and packed-weight pointers live removed 24 scalar pointer moves per K32 iteration without changing resources and improved the parent to `49.8709 ms` (`1.0029x`).

### Activation prefetch, serial-prefetch selection, and B16 overlap

PGR2/SIA4 activation prefetch was composed with direct pointers while retaining serial packed-bank reads. Resources rose only for the second activation-fragment bank, to `143 VGPR / 41 SGPR / 10,240 B` LDS, while static VALU issues fell from 1,230 to 1,150. The candidate won all three production search banks, including B1 where the installed HIP control selects M64, with weighted complete-call medians of `5.1071`/`12.0159`/`45.5463 ms` against HIP `5.4059`/`13.8792`/`48.4686 ms` (`1.0585x`/`1.1551x`/`1.0642x`).

Reversed-order 25-repeat confirmation measured `5.2683`/`11.9553`/`45.3026 ms` against HIP `5.4349`/`13.6357`/`48.0428 ms`, or `1.0316x`, `1.1405x`, and `1.0605x`, with robust independent and paired intervals classifying every key as confidently faster. A later resource-neutral successor issued the second packed-bank VMEM and first-projection A VMEM together, waited for both with one exact `vmcnt(0)`, and then decoded the second bank.

It retained the same resources and static counts. Disjoint confirmation selected it only at B16: overlap/serial ratios were `0.9886x`, `0.9964x`, and `1.0064x` at B1/B4/B16, so the retained M128 schedule is serial prefetch at B1/B4 and overlap at B16. Against HIP the matching confirmed speedups are `1.0316x`, `1.1405x`, and `1.0683x`.

### M64 geometry and mixed selection

The M64 activation-prefetch artifact inspects at `95 VGPR / 41 SGPR / 10,240 B` LDS with 32 WMMAs and four barriers. At B1 it measured `5.21436 ms` versus `5.37263 ms` for the serial-prefetch M128 candidate and `5.45459 ms` for HIP (`1.0461x`). Applying the installed threshold gives a mixed M64/M128 choice at `5.17862 ms`, `1.0533x` against HIP, so the research selector uses M64 serial prefetch below the threshold, M128 serial prefetch at B1/B4 at or above it, and M128 overlap for B16.

## Rejected Experiments

### Concurrent packed-bank reads

Issuing both packed projection reads before either decode kept correctness and spill freedom but duplicated the eight Q3 payload/high-mask registers plus `d` and scale metadata, raising usage from 127 to 139 VGPRs. The common bracket regressed from `50.0691` to `50.7209 ms` (`0.9871x`), so the identity and allocation were removed.

### SIA5 WMMA waits

The ordinary Q3_K PGR2/SIA5 wait schedule was ported without changing packed-read order but raised static waits from 26 to 50. Independent artifacts remained resource-identical, but the B16 screen regressed from `45.8688` to `46.4371 ms` (`0.9878x` of SIA4), so the identity was removed. This is a timing-only closure on an older protocol and may be rechecked on the current parent under the current complete-call contract. It is not reopened unconditionally.

### Paired zero-accumulator lifetime

Moving the dedicated `v124:v131` initialization before the K2048 block loop removed 24 statically emitted moves and changed the artifact from `3,000` to `2,976` VALU issues. Resources were unchanged, but weighted complete-call movement was `+0.01%`, `-0.01%`, and `-0.08%` at B1/B4/B16 and body movement was `-0.28%`, `-0.07%`, and `+0.16%`. The mechanism is closed as timing-neutral.

### Q3-3 epilogue probes and Q3-4 metadata

Shared-column setup removed six VALU issues, the broader materialized-address form removed 25, and the exact width-two BF16 RNE interleave preserved instruction and resource counts. Complete-call and body movements were mixed or contradictory, so no epilogue form is retained. Applying the processor-mode metadata spelling produced byte-identical objects and HSACOs at every row count, so it is executable-inert and no timing result exists.

## Open Items

B1 and B16 may need focused retuning. The lower current margins may reflect a one-profile versus five-medoid corpus. The rejected concurrent packed-bank-read identity remains closed because it raised the register envelope and lost its bracket. The B4 serial-prefetch selection remains closed unless a new candidate changes that resource or ownership premise.

## Closure

The retained result is the fused single-LDS pair kernel with PGR2/SIA4 activation prefetch and direct pointers: M64 serial routes at B1, M128 serial routes with dual padded LDS at B4, and the M128 second-bank-read/A-overlap body at B16.
