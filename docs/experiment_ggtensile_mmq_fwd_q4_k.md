# GGTensile MMQ Forward Q4_K Kernel Record

## Scope

This record covers direct packed GGUF Q4_K dense forward kernels for gfx1151, wave32, WMMA V1, BF16 input/output, and the fixed Q8_1 F16_D4S4 activation workspace. The kernel computes:

```text
output[M,N] = input[M,K] @ dequant_q4_k(weight[N,K]).T
```

The packed Q4_K block is 144 bytes and contains 256 logical values. The activation producer supplies four Q8_1 32-value subblocks per 144-byte workspace block. The producer, workspace layout, arithmetic order, ABI, and packed input representation are fixed throughout the record.

Every selected kernel is direct-packed, has zero private bytes, spills, scratch instructions, calls, and dynamic stack, and uses the retained four-wave `128x64` ownership unless a rejected experiment says otherwise. Correctness gates include bitwise HIP equality for exact candidates, independent GGUF dequantized reference checks, finiteness, input/packed-weight/workspace mutation, strict code-object inspection, and deterministic rebuilds.

## Final Results

The table reports prequantized multiply-only timing. HIP and GGTensile consume the same Q8_1 workspace from the same fixed producer. Speedup is `HIP median time / GGTensile median time`; values above `1.0x` favor GGTensile. The TFLOPS value is the GGTensile logical throughput.

| Family | Matrix shape `(M,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | ---: | --- | ---: | ---: |
| Narrow | `(2048,512,2048)` | `ggsol_3ff59047beab3eaa` | `24.225` | `1.0136x` |
| Narrow | `(8192,512,2048)` | `ggsol_197fb1d1dcba435c` | `28.621` | `1.0241x` |
| Narrow | `(32768,512,2048)` | `ggsol_0631196a836a3532` | `28.974` | `1.0280x` |
| Shared down | `(2048,2048,512)` | `ggsol_3d508299e01b6d3f` | `25.144` | `1.0257x` |
| Shared down | `(8192,2048,512)` | `ggsol_ebf8da198b49dad6` | `27.879` | `1.0327x` |
| Shared down | `(32768,2048,512)` | `ggsol_a32bcac489fa119f` | `28.104` | `1.0363x` |
| Attention output | `(2048,2048,4096)` | `ggsol_4f3ca224b7440378` | `29.050` | `1.0325x` |
| Attention output | `(8192,2048,4096)` | `ggsol_01a4e41ee6ae6723` | `29.121` | `1.0280x` |
| Attention output | `(32768,2048,4096)` | `ggsol_31ea775de2173388` | `29.224` | `1.0252x` |
| Query | `(2048,8192,2048)` | `ggsol_04db9854941608d0` | `29.104` | `1.0362x` |
| Query | `(8192,8192,2048)` | `ggsol_cb4214664a51f7e7` | `29.287` | `1.0366x` |
| Query | `(32768,8192,2048)` | `ggsol_a9b75ae588052cf8` | `29.185` | `1.0326x` |

The final multiply values combine two independent rotating 25-repeat confirmations. The largest per-key A/B median difference was `0.65` percentage points. A separate complete-call audit timed the fixed producer and multiply together; it is diagnostic and does not replace the table. Shared-down M2048 was materially inconsistent in that audit, measuring `1.0296x/0.9685x` and later `0.9855x/0.9745x`, so it is treated as complete-call parity with unresolved direction.

## Final Kernel And Profile

The retained body uses 128 threads, a `128x64` matrix tile, 239 VGPRs, 16 SGPRs, 38,400 bytes of LDS, 32 static WMMA instructions, four barriers, and eight output-store clauses. The selected kernels remain bitwise equal to HIP and use the same packed Q4_K and Q8_1 workspace bytes.

The common accepted body combines:
- Cooperative Q4_K payload and metadata staging into decoded LDS rows.
- Independent scale/minimum field extraction followed by exact FP16 conversion.
- Metadata LDS reads between the low- and high-half WMMA batches.
- Four-wave signed-int8 WMMA accumulation with FP32 scale/minimum correction.
- Eight contiguous BF16 output-store clauses with incremental output-row addressing.
- The exact two-instruction `RNEPreserveNaN` FP32-to-BF16 conversion.

The metadata-after-low schedule reduced the low-half dependency ladder from `23,21,...,9` to `15,13,...,1` and improved its prior by approximately `2.1-5.0%`. Independent metadata extraction added approximately `0.57-1.60%`. Their composition is used by all final kernels. Exact resource-neutral epilogues were retained where they cleared paired timing: narrow M32768 uses `a4d4-p2`; shared-down uses `a1d2-p2`, `a1d4-p2`, and `a1d2-p2` for M2048, M8192, and M32768; query uses `a1d2-p2`, `a4d4-p2`, and `a2d2-p2`. Neutral attention-output and other narrow keys retain the common epilogue.

Profiling shows fewer VALU instructions and flat loads than HIP, but higher instruction-fetch waiting and aggregate wait-any cycles. K512 retained and HIP controls have comparable LDS conflict and L2-hit rates. The remaining bottleneck is issue/dependency balance and synchronization rather than raw packed traffic. Measured lower bounds put store-only work at approximately `24-27%` of retained K512 latency and `2-7%` for K2048/K4096. Decode/stage/store work is approximately `44-57%`; memoryless math/control/store work is approximately `86-88%`. These diagnostic floors are not additive.

## Accepted Experiment Log

### Retained decoded-LDS representation

The decoded Q4_K/Q5_K LDS body was accepted after strict HIP equality, independent-reference checks, mutation checks, resource inspection, and deterministic rebuilds across all 12 exact shapes. The representation preserves direct packed reads and moves the fixed Q8_1 activation plane plus decoded weight rows into 38,400-byte LDS without prepared weights or external decode storage.

### Metadata extraction and placement

Independent extraction of the eight Q4_K scale/minimum fields and metadata reads between the low and high WMMA batches were accepted. The generated writer preserves integer accumulation and FP32 correction order, and the source and code object rebuild byte-identically. The same placement and extraction policies were independently qualified on narrow, shared-down, attention-output, and query shapes.

### Exact epilogue selection

The dependency-width, tiles-ahead, and store-priority neighborhood was searched with exact output requirements. The selected per-key epilogues above were retained because they cleared two rotating 25-repeat multiply confirmations and did not change the resource envelope. Candidates that were neutral or inconsistent were not generalized.

### Register-lifetime lowering

A failed diagnostic reused `v232` for the persistent metadata base and produced NaNs after Q4 decode clobbered that register. The corrected lowering preserves the per-block metadata-base recomputation, reads the invariant wave predicate into `s12`, and reuses `v236` for the persistent activation LDS base. It removes one rolled activation-base MAD per iteration while preserving output and resources. The typed lifetime alias is accepted as a lowering correction; its measured movement is sub-percent and it is not an independent tunable schedule.

### Dependency frontier

`DecodedLdsDependencyFrontier` now derives the VMEM and LDS wait thresholds from payload planes, vector widths, deferred metadata, and the call-time row-tile count. Canonical ordinary and grouped Q4/Q5 source hashes remain unchanged. The event trace proves that both loop barriers remain required: one protects the activation plane before the second group pass, and the other protects activation and decoded-weight rows before the next packed block overwrites them.

### Short-M retuning and transaction widths

Fresh current-parent controls measured `0.353943 ms` GGTensile versus `0.288072 ms` HIP for narrow M2048 and `0.386532 ms` versus `0.292434 ms` for shared-down M2048. This replaced older mixed-regime evidence. Payload global-read, metadata-load, payload LDS-write, and metadata LDS-write widths were then tested at widths 8 and 4 on both keys. Nine-repeat screens and two reversed 25-repeat confirmations found only noise-scale movements; the largest confirmed parent speedups were `1.0025x` and `1.0023x`, with reverse rotations at `1.0012x` and `0.9997x`. No width was retained.

## Rejected Experiment Log

### Compact and alternate geometries

Unchanged `64x64`, `128x128`, and `256x64` geometries were rejected. A changed-premise narrow `128x32` body used 64 threads and 28,672 bytes of LDS, passed exact correctness and resource checks, but was `21.6%` slower than `128x64` because duplicated activation staging dominated. Conditional `64x32` was therefore not pursued.

### Loop, barrier, and local-read schedules

The linked one-by-eight loop reduced static code but reproduced the noise-scale result: approximately `0.3%` favorable after confirmation and no stable complete-call gain. Final-block barrier elision was neutral. Dependency-safe metadata placements after two, four, and six low WMMAs were exact but `0.65-2.97%` slower than placement after all eight low WMMAs. Moving independent extraction into the first packed-payload decode gap was exact and resource-neutral but neutral on K512 and `0.32-0.37%` slower on narrow and attention K2048/K4096.

### Buffering and payload prefetch

Eager dual activation staging was `10-13%` slower. A sequential split-plane form that removed one overwrite barrier was exact but `12-14%` slower because 56,832-byte LDS residency reduced useful occupancy. A 248-VGPR overlapped activation-plane prototype was incorrect and already `19%` slower. Resource-neutral group-7 packed-payload prefetch reused dead registers but was `0.13-0.28%` slower; packed payload VMEM was already sufficiently hidden. Broad activation prefetch, packed-weight clauses, loop NOP padding, and instruction-prefetch descriptor changes were also rejected.

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

Research-only `BiasRound` and `Truncate` candidates kept the 239 VGPR, 16 SGPR, and 38,400-byte LDS envelope. Both were finite and sensitive to input, packed-weight, and workspace mutation. `BiasRound` differed from the exact parent by normalized RMSE `1.76e-5`, maximum absolute error `0.00390625`, and p99/p999 absolute error `0/0`. `Truncate` differed by normalized RMSE `4.05e-3`, maximum absolute error `0.03125`, and p99/p999 error `0.00390625/0.00390625`. Reversed 25-repeat candidate/parent speedups were `0.9993x/1.0035x` and `1.0026x/1.0044x`, respectively. Neither was stable or exact, so both were rejected.

### Compiler and ISA scheduling probes

Offline gfx1151 AMDGPU compilation of an LDS-staged signed-int8 WMMA oracle emitted four WMMA instructions, one barrier, four waits, three delay instructions, 42 VGPRs, and 128 bytes of LDS. A forced-volatile load variant expanded to 66 waits, six delay instructions, 12 SGPRs, and 1,092 bytes of code. The oracle confirms legal WMMA dependency delays but cannot represent the complete Q4 event graph, raw GGUF decode, Q8_1 correction, or four-wave ownership. No compiler physical instruction stream was translated into the Q4 writer.

### Other closed mechanisms

Flattened workgroup clustering, cross-lane output packing, broad metadata byte permutes, broad priority settings, gfx1151 split barriers, direct-to-LDS, direct-to-VGPR alternatives, VOPD payload-mask reduction, and unchanged Q8_1 or decoded-weight ping-pong were rejected by ISA legality, correctness, resource residency, or timing evidence. The fixed producer and decoded representation remain the controlling premises.

## Kernel Closure

The final exact kernels are resource-clean, independently rebuilt, mutation-sensitive, finite, and faster than HIP in the two independent rotating multiply confirmations for every listed shape. The remaining gap is dominated by instruction-fetch waiting, wait-any/dependency balance, synchronization, and decode/stage/store work. All tested kernel-level mechanisms under the current direct-packed Q4_K, fixed-producer, WMMA V1, wave32, and zero-private-resource constraints are either retained above or rejected with evidence. A future reopening requires a materially new instruction, representation, ownership, or arithmetic premise.
