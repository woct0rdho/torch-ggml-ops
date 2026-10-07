# GGTensile MMQ Forward Q4_K Kernel Record

## Scope

This record covers direct packed GGUF Q4_K dense forward kernels for gfx1151.

## Final Results

`TFLOPS = 2*M*N*K / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Medians are the repository benchmark's, re-measured in the current clock state. They replace the earlier recorded values, which came from a different clock state with the same artifacts.

| Family | `(M,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| --- | --- | ---: | ---: | --- | --- |
| Narrow K/V/gate/up | `(2048,512,2048)` | 26.833 | 1.0454x | `mmq_fwd_q4_k_m2048_n512_k2048_753ef81fa5bc326a` | `dense_fwd_q4_k_k2048_j128_full` |
| Narrow K/V/gate/up | `(8192,512,2048)` | 29.011 | 1.0423x | `mmq_fwd_q4_k_m8192_n512_k2048_edf26f8be572c69d` | `dense_fwd_q4_k_k2048_j128_full` |
| Narrow K/V/gate/up | `(32768,512,2048)` | 29.023 | 1.0356x | `mmq_fwd_q4_k_m32768_n512_k2048_917314a3cf93ca5a` | `dense_fwd_q4_k_k2048_j128_full` |
| Shared down | `(2048,2048,512)` | 27.758 | 1.0614x | `mmq_fwd_q4_k_m2048_n2048_k512_4b5e4be66d47912c` | `dense_fwd_q4_k_k512_j128_full` |
| Shared down | `(8192,2048,512)` | 28.000 | 1.0542x | `mmq_fwd_q4_k_m8192_n2048_k512_7b2f9b5e15d65249` | `dense_fwd_q4_k_k512_j128_full` |
| Shared down | `(32768,2048,512)` | 28.176 | 1.0522x | `mmq_fwd_q4_k_m32768_n2048_k512_ea97ae0eb9b4b154` | `dense_fwd_q4_k_k512_j128_full` |
| Attention output | `(2048,2048,4096)` | 29.198 | 1.0437x | `mmq_fwd_q4_k_m2048_n2048_k4096_665eb7995f7cde3c` | `dense_fwd_q4_k_k4096_j128_full` |
| Attention output | `(8192,2048,4096)` | 29.157 | 1.0303x | `mmq_fwd_q4_k_m8192_n2048_k4096_041b661cf32a27b8` | `dense_fwd_q4_k_k4096_j128_full` |
| Attention output | `(32768,2048,4096)` | 29.399 | 1.0303x | `mmq_fwd_q4_k_m32768_n2048_k4096_382a91e3812e7fb3` | `dense_fwd_q4_k_k4096_j128_full` |
| Query/query gate | `(2048,8192,2048)` | 28.951 | 1.0408x | `mmq_fwd_q4_k_m2048_n8192_k2048_5f6c7838510c76b4` | `dense_fwd_q4_k_k2048_j128_full` |
| Query/query gate | `(8192,8192,2048)` | 29.262 | 1.0400x | `mmq_fwd_q4_k_m8192_n8192_k2048_0659f7a435c29a87` | `dense_fwd_q4_k_k2048_j128_full` |
| Query/query gate | `(32768,8192,2048)` | 28.119 | 1.0412x | `mmq_fwd_q4_k_m32768_n8192_k2048_d3c137a521e71c57` | `dense_fwd_q4_k_k2048_j128_full` |

GGTensile is ahead on all twelve entries, by `1.030x` to `1.061x` (mean `1.043x`).

## Final Kernel Notes

The metadata-after-low schedule reduced the low-half dependency ladder from `23,21,...,9` to `15,13,...,1` and improved its prior by approximately `2.1-5.0%`. Independent metadata extraction added approximately `0.57-1.60%`. Their composition is used by all final kernels. Exact resource-neutral epilogues were retained where they cleared paired timing: narrow M32768 uses `a4d4-p2`. Shared-down uses `a1d2-p2`, `a1d4-p2`, and `a1d2-p2` for M2048, M8192, and M32768. Query uses `a1d2-p2`, `a4d4-p2`, and `a2d2-p2`. Neutral attention-output and other narrow keys retain the common epilogue.

Profiling shows fewer VALU instructions and flat loads than HIP, but higher instruction-fetch waiting and aggregate wait-any cycles. K512 retained and HIP controls have comparable LDS conflict and L2-hit rates. The remaining bottleneck is issue/dependency balance and synchronization rather than raw packed traffic. Measured lower bounds put store-only work at approximately `24-27%` of retained K512 latency and `2-7%` for K2048/K4096. Decode/stage/store work is approximately `44-57%`. Memoryless math/control/store work is approximately `86-88%`. These diagnostic floors are not additive.

## Accepted Experiment Log

### Retained decoded-LDS representation

The decoded Q4_K/Q5_K LDS body was accepted across all 12 exact shapes. The representation preserves direct packed reads and moves the fixed Q8_1 activation plane plus decoded weight rows into 38,400-byte LDS without prepared weights or external decode storage.

### Metadata extraction and placement

Independent extraction of the eight Q4_K scale/minimum fields and metadata reads between the low and high WMMA batches were accepted. The same placement and extraction policies are used on narrow, shared-down, attention-output, and query shapes.

### Exact epilogue selection

The dependency-width, tiles-ahead, and store-priority neighborhood was searched with exact output requirements. The selected per-key epilogues above were retained because they cleared two rotating 25-repeat multiply confirmations and did not change the resource envelope. Candidates that were neutral or inconsistent were not generalized.

### Register-lifetime lowering

A failed diagnostic reused `v232` for the persistent metadata base and produced NaNs after Q4 decode clobbered that register. The corrected lowering preserves the per-block metadata-base recomputation, reads the invariant wave predicate into `s12`, and reuses `v236` for the persistent activation LDS base. It removes one rolled activation-base MAD per iteration while preserving output and resources. The typed lifetime alias is accepted as a lowering correction. Its measured movement is sub-percent and it is not an independent tunable schedule.

### Dependency frontier

`DecodedLdsDependencyFrontier` now derives the VMEM and LDS wait thresholds from payload planes, vector widths, deferred metadata, and the call-time row-tile count. Canonical ordinary and grouped Q4/Q5 source hashes remain unchanged. The event trace proves that both loop barriers remain required: one protects the activation plane before the second group pass, and the other protects activation and decoded-weight rows before the next packed block overwrites them.

### Short-M retuning and transaction widths

Fresh current-parent controls measured `0.353943 ms` GGTensile versus `0.288072 ms` HIP for narrow M2048 and `0.386532 ms` versus `0.292434 ms` for shared-down M2048. This replaced older mixed-regime evidence. Payload global-read, metadata-load, payload LDS-write, and metadata LDS-write widths were then tested at widths 8 and 4 on both keys. Nine-repeat screens and two reversed 25-repeat confirmations found only noise-scale movements. The largest confirmed parent speedups were `1.0025x` and `1.0023x`, with reverse rotations at `1.0012x` and `0.9997x`. No width was retained.

## Rejected Experiment Log

### Compact and alternate geometries

Unchanged `64x64`, `128x128`, and `256x64` geometries were rejected. A changed-premise narrow `128x32` body used 64 threads and 28,672 bytes of LDS, passed exact correctness and resource checks, but was `21.6%` slower than `128x64` because duplicated activation staging dominated. Conditional `64x32` was therefore not pursued.

### Loop, barrier, and local-read schedules

The linked one-by-eight loop reduced static code but reproduced the noise-scale result: approximately `0.3%` favorable after confirmation and no stable complete-call gain. Final-block barrier elision was neutral. Dependency-safe metadata placements after two, four, and six low WMMAs were exact but `0.65-2.97%` slower than placement after all eight low WMMAs. Moving independent extraction into the first packed-payload decode gap was exact and resource-neutral but neutral on K512 and `0.32-0.37%` slower on narrow and attention K2048/K4096.

### Buffering and payload prefetch

Eager dual activation staging was `10-13%` slower. A sequential split-plane form that removed one overwrite barrier was exact but `12-14%` slower because 56,832-byte LDS residency reduced useful occupancy. A 248-VGPR overlapped activation-plane prototype was incorrect and already `19%` slower. Resource-neutral group-7 packed-payload prefetch reused dead registers but was `0.13-0.28%` slower. Packed payload VMEM was already sufficiently hidden. Broad activation prefetch, packed-weight clauses, loop NOP padding, and instruction-prefetch descriptor changes were also rejected.

### Compact raw-payload LDS

A hybrid row with 128 raw Q4_K payload bytes and 32 decoded metadata bytes reduced LDS to 28,672 bytes and realized the intended higher residency class. It was exact and resource-clean, but all legal placements moved consumer nibble expansion onto the WMMA critical path:

| Variant | Candidate/parent |
| --- | ---: |
| Serialized compact control | `1.18148x` |
| Activation-read overlap | `1.16412x` |
| Traffic-correct pair reuse | `1.16537x` |

The traffic-correct pair-reuse body reduced payload LDS traffic as planned but remained more than 16% slower. The premise is rejected unless a future representation removes consumer decode work rather than rescheduling it.

### Padded decoded LDS

Formula-derived `(LdsPadA,LdsPadB)` candidates first exposed an address-model error: padding advances the 16-row activation tile stride, not each individual Q8_1 row. After correcting the model, decoded-Q4/Q5 padded rows were rejected before lowering because fixed-width B128 LDS transactions have no padding-safe cooperative address transform in the current writer. A focused test covers this capability rejection. Canonical layout generation is unchanged.

### Output conversion approximation

The `BiasRound` and `Truncate` candidates kept the 239 VGPR, 16 SGPR, and 38,400-byte LDS envelope. Both were finite and sensitive to input, packed-weight, and workspace mutation. `BiasRound` differed from the exact parent by normalized RMSE `1.76e-5`, maximum absolute error `0.00390625`, and p99/p999 absolute error `0/0`. `Truncate` differed by normalized RMSE `4.05e-3`, maximum absolute error `0.03125`, and p99/p999 error `0.00390625/0.00390625`. Reversed 25-repeat candidate/parent speedups were `0.9993x/1.0035x` and `1.0026x/1.0044x`, respectively. Neither was stable or exact, so both were rejected.

### Compiler and ISA scheduling probes

Offline gfx1151 AMDGPU compilation of an LDS-staged signed-int8 WMMA oracle emitted four WMMA instructions, one barrier, four waits, three delay instructions, 42 VGPRs, and 128 bytes of LDS. A forced-volatile load variant expanded to 66 waits, six delay instructions, 12 SGPRs, and 1,092 bytes of code. The oracle confirms legal WMMA dependency delays but cannot represent the complete Q4 event graph, raw GGUF decode, Q8_1 correction, or four-wave ownership. No compiler physical instruction stream was translated into the Q4 writer.

### Other closed mechanisms

Flattened workgroup clustering, cross-lane output packing, broad metadata byte permutes, broad priority settings, gfx1151 split barriers, direct-to-LDS, direct-to-VGPR alternatives, VOPD payload-mask reduction, and unchanged Q8_1 or decoded-weight ping-pong were rejected by ISA legality, correctness, resource residency, or timing evidence. The fixed producer and decoded representation remain the controlling premises.

## Kernel Closure

The final kernels are faster than HIP in the two independent rotating multiply confirmations for every listed shape. The remaining gap is dominated by instruction-fetch waiting, wait-any/dependency balance, synchronization, and decode/stage/store work. All tested kernel-level mechanisms under the current direct-packed Q4_K, fixed-producer, WMMA V1, wave32, and zero-private-resource constraints are either retained above or rejected with evidence. A future reopening requires a materially new instruction, representation, ownership, or arithmetic premise.
