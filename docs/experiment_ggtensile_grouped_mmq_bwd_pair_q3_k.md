# GGTensile Grouped MMQ Backward Pair Q3_K Experiment

## Scope And Contract

This record covers the isolated gfx1151 paired Q3_K backward kernel for the Qwen routed gate and up projections. For each routed expert the kernel performs both input-gradient matmuls in one workgroup dataflow:

```text
dG_g[M_g,512] @ W_gate_g[512,2048] + dU_g[M_g,512] @ W_up_g[512,2048] -> dX_g[M_g,2048]
```

The measured aggregate-row shapes are `R=16384`, `65536`, and `262144`. Each packed Q3_K bank is `[256,512,880]`: each 256-value block occupies 110 bytes, each packed row contains eight blocks, and each expert occupies 450,560 bytes. Both gradient outputs and the shared gradient input are contiguous BF16, and both projections are accumulated into one shared FP32 WMMA accumulator set before one BF16 RNE conversion and store.

The paired research ABI is the 72-byte specialized pair contract carrying both gradient outputs, both packed weights, the gradient input, the device-resident route description, and the shape values. Logical paired throughput is `4 * R * 512 * 2048 / (latency_ms * 1e9)`, and the speedup ratio is HIP time divided by GGTensile time.

## Final Benchmark Results

The table shows the fastest qualified kernel found for each measured aggregate-row shape.

| Matrix shape `(R,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(16384,2048,512)` | `ggbpair_72eedb6708bbbe0c` | `13.179` | `1.0461x` |
| `(65536,2048,512)` | `ggbpair_d6b74a04787d53ff` | `22.774` | `1.1409x` |
| `(262144,2048,512)` | `ggbpair_ac794f796fe331a9` | `24.281` | `1.0683x` |

The B1 row uses the M64 serial-prefetch body; B4 uses the M128 serial-prefetch body; B16 uses the M128 second-read/A-overlap body. All route, malformed-route, mutation, and finite-output checks were exact.

## Final Kernel Profiles

| Kernel hash | Geometry and ownership | Decode and schedule | VGPR / SGPR | LDS bytes | WMMAs / barriers |
| --- | --- | --- | ---: | ---: | ---: |
| `ggbpair_72eedb6708bbbe0c` | M64 serial routes | dual padded LDS, PGR2/SIA4 activation prefetch | `95 / 41` | `10,240` | `32 / 4` |
| `ggbpair_d6b74a04787d53ff` | M128 serial routes | dual padded LDS, PGR2/SIA4 activation prefetch, direct pointers | `143 / 41` | `10,240` | `64 / 4` |
| `ggbpair_ac794f796fe331a9` | M128 serial routes | dual padded LDS, second-bank read overlapped with first activation prefetch | `143 / 41` | `10,240` | `64 / 4` |

All selected artifacts are gfx1151 code-object-v5 wave32 kernels with exact 72-byte metadata and zero private storage, spills, scratch, calls, and dynamic stack.

## Accepted Kernel Experiments

### Fused single-LDS anchors and geometry baseline

The paired problem and contract admit only the exact Qwen Q3_K geometry. Canonical M64/N64 and M128/N64 identities use pad8, packed extraction, explicit full Q3 VOPD pairing, K32, serial routes, and the interleaved projection schedule. M64 inspects at `87 VGPR / 41 SGPR / 5,120 B` LDS with 16 WMMAs and four barriers; M128 at `127 / 41 / 5,120 B` with 32 WMMAs and four barriers. Both were bit-exact to installed HIP and the installed M64 control on the 35-row matrix, with the independent FP32 pair reference differing in only 3 of 71,680 BF16 values.

The fitted baseline found M64 effectively tied with HIP at B1 (`5.3885` versus `5.3661 ms`, `0.9958x`) but rejected at B4/B16, where it reached only `0.8290x` and `0.7586x`. M128 won B4 at `1.0176x` but trailed B16 at `0.9247x`, so the B16 deficit became the first optimization target. The first control probe also used the wrong dynamic LDS allocation for the installed J64 control (29,952 instead of 30,976 bytes) and differed in 356 BF16 values; correcting only the control restored bitwise agreement.

### Dual padded LDS and full-tile split

The first paired-specific schedule assigned the two decoded Q3_K projections to disjoint 5 KiB padded LDS images, decoded both banks before one consumer barrier, and synchronized once before overwrite. M128 stayed at `127 VGPR / 41 SGPR`, LDS rose to `10,240 B`, barriers fell from four to two, and B16 improved from `52.8582` to `51.8399 ms` (`1.0196x`) while remaining behind contemporaneous HIP.

Splitting each route into an unbounded M128 full-tile body plus a bounded tail body kept the same resources and made the full-tile split the new B16 parent, improving dual LDS from `52.3253` to `50.0691 ms` (`1.0451x`) and reaching `0.9706x` contemporaneous HIP. Keeping the second gradient and packed-weight pointers live removed 24 scalar pointer moves per K32 iteration without changing resources and improved the parent to `49.8709 ms` (`1.0029x`).

### Activation prefetch, serial-prefetch selection, and B16 overlap

PGR2/SIA4 activation prefetch was composed with direct pointers while retaining serial packed-bank reads. Resources rose only for the second activation-fragment bank, to `143 VGPR / 41 SGPR / 10,240 B` LDS, while static VALU issues fell from 1,230 to 1,150. The candidate won all three production search banks, including B1 where the installed HIP control selects M64, with weighted complete-call medians of `5.1071`/`12.0159`/`45.5463 ms` against HIP `5.4059`/`13.8792`/`48.4686 ms` (`1.0585x`/`1.1551x`/`1.0642x`).

Reversed-order 25-repeat confirmation measured `5.2683`/`11.9553`/`45.3026 ms` against HIP `5.4349`/`13.6357`/`48.0428 ms`, or `1.0316x`, `1.1405x`, and `1.0605x`, with robust independent and paired intervals classifying every key as confidently faster. A later resource-neutral successor issued the second packed-bank VMEM and first-projection A VMEM together, waited for both with one exact `vmcnt(0)`, and then decoded the second bank. It retained the same resources and static counts. Disjoint confirmation selected it only at B16: overlap/serial ratios were `0.9886x`, `0.9964x`, and `1.0064x` at B1/B4/B16, so the retained M128 schedule is serial prefetch at B1/B4 and overlap at B16. Against HIP the matching confirmed speedups are `1.0316x`, `1.1405x`, and `1.0683x`.

### M64 geometry and mixed selection

The M64 activation-prefetch artifact inspects at `95 VGPR / 41 SGPR / 10,240 B` LDS with 32 WMMAs and four barriers and passes the two-full-plus-tail, exact-full-then-tail, tail-full-tail, and tail-only route layouts bit-exactly. At B1 it measured `5.21436 ms` versus `5.37263 ms` for the serial-prefetch M128 candidate and `5.45459 ms` for HIP (`1.0461x`). Applying the installed threshold gives a mixed M64/M128 choice at `5.17862 ms`, `1.0533x` against HIP, so the research selector uses M64 serial prefetch below the threshold, M128 serial prefetch at B1/B4 at or above it, and M128 overlap for B16.

## Rejected Kernel Experiments

### Concurrent packed-bank reads

Issuing both packed projection reads before either decode kept correctness and spill freedom but duplicated the eight Q3 payload/high-mask registers plus `d` and scale metadata, raising usage from 127 to 139 VGPRs. The common bracket regressed from `50.0691` to `50.7209 ms` (`0.9871x`), so the identity and allocation were removed.

### SIA5 WMMA waits

The ordinary Q3_K PGR2/SIA5 wait schedule was ported without changing packed-read order but raised static waits from 26 to 50. Independent artifacts remained resource-identical and exact, but the B16 screen regressed from `45.8688` to `46.4371 ms` (`0.9878x` of SIA4), so the identity was removed. This is a timing-only closure on an older protocol and may be rechecked on the current parent under the current complete-call contract; it is not reopened unconditionally.

### Paired zero-accumulator lifetime

Moving the dedicated `v124:v131` initialization before the K2048 block loop removed 24 statically emitted moves and changed the artifact from `3,000` to `2,976` VALU issues. Outputs were exact and resources unchanged, but weighted complete-call movement was `+0.01%`, `-0.01%`, and `-0.08%` at B1/B4/B16 and body movement was `-0.28%`, `-0.07%`, and `+0.16%`. The mechanism is closed as timing-neutral.

### Q3-3 epilogue probes and Q3-4 metadata

Shared-column setup removed six VALU issues, the broader materialized-address form removed 25, and the exact width-two BF16 RNE interleave preserved instruction and resource counts. All passed the exact route and mutation matrix, but complete-call and body movements were mixed or contradictory, so no epilogue form is retained. Applying the processor-mode metadata spelling produced byte-identical objects and HSACOs at every row count, so it is executable-inert and no timing result exists.

## Remaining Work

The current direct-kernel benchmark measured B1 near parity at `1.0026x` (50-repeat top-up `1.0131x`) versus the documented `1.0461x`, and B16 at `1.0250x` versus `1.0683x`. B4 is healthy at `1.1355x` versus `1.1409x`. Both B1 and B16 may need focused retuning; the lower current margins may reflect a one-profile versus five-medoid corpus. The rejected concurrent packed-bank-read identity remains closed because it raised the register envelope and lost its bracket. The B4 serial-prefetch selection remains closed unless a new candidate changes that resource or ownership premise.

## Qualification Summary

The retained kernels pass exact packed-HIP comparison, independent FP32-accumulating references, finite-output checks, deterministic reruns, first and non-first routes, sparse and repeated experts, both-gradient and both-active-bank mutations, inactive-expert inertness, malformed-route sentinels, and mixed full/tail ownership. The independent pair reference retains 3 differing BF16 values out of 71,680 with NRMSE `6.89e-8`. Two independent generation, build, and inspection roots reproduce the selected artifacts byte-for-byte with the profile above.
