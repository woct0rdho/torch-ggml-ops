# GGTensile Grouped MMQ Forward Q5_K Experiment

## Scope And Contract

This record covers the isolated gfx1151 grouped Q5_K forward kernel for the Qwen routed down projection. For each routed expert, the operation is:

```text
X_g[M_g,512] @ W_g[512,2048] -> Y_g[M_g,2048]
```

The measured aggregate-row shapes are `R=16384`, `65536`, and `262144`. The packed Q5_K expert bank is `[256,2048,352]`; each 256-value block occupies 176 bytes, so `K=512` stores two blocks per output row. A block contains FP16 `d` and `dmin`, twelve packed scale/minimum bytes, a 32-byte high-bit plane, and a 128-byte low-nibble plane. The activation workspace is the fixed HIP Q8_1 `F16_D4S4` layout with shape `[4,R,144]`. Output is contiguous BF16 `[R,2048]`.

The grouped research ABI is the existing 64-byte routed multiply contract with device-resident int64 expert IDs and cumulative int32 offsets. Logical throughput is `2 * R * 2048 * 512 / (latency_ms * 1e9)`, and the speedup ratio is HIP time divided by GGTensile time. The final timing uses the fixed HIP quantizer and prequantized multiply-only throughput.

## Final Benchmark Results

The table shows the fastest qualified kernel found for each measured aggregate-row shape.

| Matrix shape `(R,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(16384,2048,512)` | `ggsol_a35d42962fcb240d` | `15.283` | `0.9987x` |
| `(65536,2048,512)` | `ggsol_38e2fe81d0ce5aea` | `22.691` | `1.0770x` |
| `(262144,2048,512)` | `ggsol_a9d33673184bf5ae` | `25.776` | `1.0039x` |

The selected entries passed the retained route-correctness, resource, and deterministic-build checks.

## Final Kernel Profiles

| Kernel hash | Geometry and ownership | Decode and epilogue | VGPR / SGPR | LDS bytes | WMMAs / barriers |
| --- | --- | --- | ---: | ---: | ---: |
| `ggsol_a35d42962fcb240d` | M128/N64 serial routes, 64-row parent plus 32-row residual | `a1d4-p2` epilogue, qh high-bit merge | `159 / 40` | `29,184` | `24 / 4` |
| `ggsol_38e2fe81d0ce5aea` | M128/N128 serial routes, three-way 128/64/32 final-tile body | `a1d4-p2` epilogue, qh high-bit merge | `239 / 40` | `38,400` | `56 / 4` |
| `ggsol_a9d33673184bf5ae` | M128/N128 serial routes, three-way 128/64/32 final-tile body | `a1d4-p2` epilogue, qh high-bit merge | `239 / 40` | `38,400` | `56 / 4` |

The 56 static WMMAs represent mutually exclusive 128-, 64-, and 32-row bodies. All selected artifacts target gfx1151 code-object version 5 and wave32 with zero private storage, spills, scratch, calls, and dynamic stack.

## Accepted Kernel Experiments

### Shared Q5_K high-bit reconstruction

Q5_K shares the Q4_K scale/minimum arithmetic, Q8_1 activation staging, signed integer WMMA, correction, and BF16 stores; its distinct cost is high-bit reconstruction. The typed decoded-LDS emitter loads `qh`, shifts lane-selected bit masks, merges bit 4 into both low-nibble halves, and releases the high-bit payload before the WMMA loop. This decoder is used by every retained identity and was qualified bitwise against the installed J64/J32 controls on the R35 route, uniform, skewed, sparse-ID, boundary, and repeated-ID routes at all three production row counts.

### 64-row parent with 32-row residual at B1

The 128-row bodies were rejected as broad selectors: serialized, metadata-scheduled, and `a1d2-p2` forms all used 239 VGPRs and 38,400 LDS bytes and reached only `0.650x`/`0.679x`/`0.683x` at B1, `0.954x`/`0.987x`/`0.995x` at B4, and `0.945x`/`0.969x`/`0.981x` at B16 installed/candidate.

The Q4-qualified 64-row layout transfers without a new register plan at 159 VGPRs, 40 SGPRs, 29,184 LDS bytes, 16 static WMMAs, and four barriers. Its scheduled `a1d4-p2` form reached `0.910x`, `1.010x`, and `0.946x` at B1/B4/B16. Adding a mutually exclusive 32-row tail body, which raises the static WMMA count to 24 while retaining the same resources, reached `1.000x` at B1 and `1.036x` at B4. The B1 confirmation brackets weighted parity at `1.0020x` and `0.9987x`; low-weight medoids regress by up to about 4-6% and the uniform synthetic route is `0.955x`, so the identity is retained as a fitted-prior control.

### Three-way 128/64/32 final-tile ownership at B4/B16

A Q5-specific final-tile policy emits mutually exclusive 128-, 64-, and 32-row activation, MMA, correction, and store bodies at 239 VGPRs, 40 SGPRs, 38,400 LDS bytes, 56 static WMMAs, and four barriers. A 388-row route containing 17-, 48-, 128-, and 195-row groups exercises all three branches and matches installed HIP bitwise.

This is the material B4 result. The first nine-repeat screen reached `1.0636x`, and two independent 25-repeat confirmations reached `1.0597x` and `1.0770x`, with every medoid at least `1.0274x` and `1.0360x`. The full `a1d4-p2` identity also reaches `1.0015x` and `1.0039x` at B16 with every confirmation medoid above parity. A final complete-call audit with the fixed quantizer measured `1.0539x`, `1.0813x`, and `1.0070x` installed/candidate and preserved the B4 gain.

## Rejected Kernel Experiments

### True 32-row ownership

A dedicated two-fragment 32-row layout reduced resources to 135 VGPRs and 24,576 LDS bytes with eight static WMMAs and bitwise-exact output, but B1 fell to `0.9872x` weighted with four medoids at `0.896x` to `0.916x`. Lower residency cannot repay decoding each Q5 high-bit payload twice as often, so the mechanism was reverted and only the shorter residual body inside the 64-row parent is retained.

### Grouped VOPD accumulator initialization

The dense Q5 resource-neutral VOPD accumulator initialization was exact and changed scalar-parent weighted timing by `1.0023x`, `0.9920x`, and `1.0019x` at B1/B4/B16. The movements are sub-percent and inconsistent, with the material B4 key regressing, so the grouped VOPD path was reverted.

### Routed-prologue and address reductions

The shared packed-kernarg, paired cumulative-offset, and guarded high-stride/address reductions rebuilt deterministically, preserved the selected parent's resources, and matched the parent bitwise on all route controls. The Q4_K B1 transfer screen measured `+0.030%`, `-0.086%`, and `-0.083%` candidate time, none clearing the gate. Because Q5_K performs the same route setup around a longer high-bit decode body, the prespecified upper-bound gate rejects production timing transfer and the mechanisms are closed as neutral.

### Contract-incompatible and deferred mechanisms

Host route readback, hidden caches, dense weight shadows, producer fusion, and online tuning remain contract-incompatible. Flattened or persistent routing requires a profile showing launch imbalance after the existing output-column parallelism, SplitK requires enough K-loop underutilization to repay partial-output storage at fixed `K=512`, and a route-persistent full-K decoded image remains unmeasured; the analogous Q4 body regressed about 12% on its strongest uniform case and the Q5 form requires the same 57,856-byte LDS class.

## Remaining Work

The current direct-kernel benchmark measured B1 at `0.9529x` GGTensile/HIP (75-repeat interval `[0.9457x, 0.9602x]`) versus the documented `0.9987x`, a stable short-row regression rather than sample ambiguity. B4 and B16 remain at `1.0635x` and `0.9990x`, close to their documented rows. The Q5-specific R1-R3 route/address transfer may be re-run against the current B1 parent on the current multiply-only protocol; the route-persistent full-K image is not reopened by this table.

## Qualification Summary

The final kernels pass exact packed-HIP comparison, independent BF16-reference checks, finite-output and full-row coverage, deterministic reruns, uniform, skewed, sparse-ID, repeated-ID, and boundary routes, active input and weight mutations, inactive-expert inertness, malformed-route sentinels, and non-aligned tails. Two independent generation, build, and inspection roots produce byte-identical artifacts. The retained result is the shared Q5 high-bit decoder with padded single-LDS storage, the 64-row parent plus 32-row residual at B1, and the three-way 128/64/32 final-tile body at B4/B16, all using serial route ownership.
