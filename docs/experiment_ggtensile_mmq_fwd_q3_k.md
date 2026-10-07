# GGTensile MMQ Forward Q3_K

## Scope

This record covers the GGTensile MMQ forward kernels for packed Q3_K weights.

## Final Results

`TFLOPS = 2*M*N*K / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Medians are the repository benchmark's, re-measured in the current clock state. They replace the earlier recorded values, which came from a different clock state with the same artifacts.

| Family | `(M,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| --- | --- | ---: | ---: | --- | --- |
| Narrow key | `(2048,512,2048)` | 25.153 | 1.1663x | `mmq_fwd_q3_k_m2048_n512_k2048_a92c014d131faecb` | `dense_fwd_q3_k_k2048_j128_full` |
| Narrow key | `(8192,512,2048)` | 27.215 | 1.1508x | `mmq_fwd_q3_k_m8192_n512_k2048_4afaa0c59f10f76d` | `dense_fwd_q3_k_k2048_j128_full` |
| Narrow key | `(32768,512,2048)` | 27.369 | 1.1354x | `mmq_fwd_q3_k_m32768_n512_k2048_569ed02d69052a73` | `dense_fwd_q3_k_k2048_j128_full` |
| Query/query gate | `(2048,8192,2048)` | 27.394 | 1.1378x | `mmq_fwd_q3_k_m2048_n8192_k2048_a117996848638902` | `dense_fwd_q3_k_k2048_j128_full` |
| Query/query gate | `(8192,8192,2048)` | 27.411 | 1.1341x | `mmq_fwd_q3_k_m8192_n8192_k2048_18350ff526b6b121` | `dense_fwd_q3_k_k2048_j128_full` |
| Query/query gate | `(32768,8192,2048)` | 25.997 | 1.1406x | `mmq_fwd_q3_k_m32768_n8192_k2048_cff119fa7866705c` | `dense_fwd_q3_k_k2048_j128_full` |

GGTensile is ahead on all six entries, by `1.134x` to `1.166x` (mean `1.144x`).

## Accepted Kernel Design

- Typed Q3 reconstruction. The lowerer owns the 110-byte Q3_K block layout, signed 3-bit reconstruction, unsigned six-bit scale decoding with the offset-32 correction, FP16 block factor `d`, and FP32 scale correction. The low payload and high mask are combined before WMMA without changing the packed representation.
- Full-weight LDS ownership. The accepted layout uses 128 Q8_1 activation rows, 64 decoded weight rows, and a 336-byte weight-row stride. The LDS formula is `18,432 + 64 * 336 = 39,936` bytes.
- Typed register and lifetime plan. The final placement keeps raw packed operands, stage addresses, scale shifts, activation staging, persistent WMMA zero operands, and accumulator fragments disjoint. The 200-VGPR profile is spill-free and deterministic.
- Decode-ready frontier. A bottom-up AMDGPU scheduler oracle showed a small improvement when independent payload and scale reconstruction chains were widened before LDS publication. The useful ownership fact was encoded as a typed lowering policy rather than copied as an instruction stream. Top-down scheduling and disabled clustering regressed and were not retained.
- Memory and correction schedule. The accepted body retains the activation LDS-base hoist, activation-scale deduplication, full activation VMEM batching, shared activation/weight VMEM issue, legal VOPD scale correction, loop-carried half-0 weight prefetch, and final-block bounds handling. These mechanisms preserve the four-barrier ownership sequence and the Q3 arithmetic order.

The earlier isolated `M=2048,N=4096,K=2048` control established the kernel foundation at 144 VGPRs and 28,672 bytes of LDS. The later full-weight design replaced that compact half-tile ownership with the 200-VGPR, 39,936-byte profile above. The isolated accepted control measured `1.457108 ms` versus HIP at `1.553431 ms`. The final full-weight table is the authoritative result for the six matrix entries above.

## Rejected Kernel Experiments

- Incorrect register aliasing. Reusing temporary decode or address registers across outstanding VMEM operations caused memory faults or incorrect output. The unsafe aliases are rejected. Explicit typed lifetimes are required.
- First metadata reuse. Keeping Q3 metadata across both halves overlapped live accumulator fragments and produced grossly incorrect output. The corrected 152-VGPR form was exact but slower, so neither form is accepted.
- Paired scale reads. Four paired weight-scale reads reduced static LDS instructions but added address arithmetic and lost timing. The short-key result was `1.50125x` HIP. The low-pressure variant reached `1.490659 ms` versus HIP at `1.543457 ms` and regressed the retained control. This mechanism is rejected and was not reopened.
- Cross-fragment and batched WMMA schedules. Two-fragment FMAC pairing, 64-result ownership, dependency-derived partial waits, rolled scale groups, and combined rolled/batched forms remained exact in their qualified variants but were slower than the retained schedule. The 200-VGPR batched form measured `1.48564x` HIP on the initial control. The later large-key composition remained above HIP as well.
- Geometry and LDS alternatives. MT64, linear activation staging, compact 320-byte weight rows, dual activation planes, alternate swizzles, and smaller or rolled ownership either failed correctness or lost timing. The 336-byte full-weight row and canonical activation plane remain the accepted layout.
- Schedule-only alternatives. HIP-shaped read order, phased correction, cache invalidation, barrier variants, unroll changes, fragment/bank mappings, and scalar-address variants did not produce a repeatable improvement. Static instruction reductions without a corresponding dependency or lifetime benefit are rejected.
- Compiler-oracle alternatives. Top-down scheduler policies and disabled clustering regressed the exact bitcode. The oracle remains diagnostic evidence. Only the typed decode-ready frontier is retained in the kernel.
- Width and pipeline variants. Payload global widths `8` and `4`, metadata widths `8` and `4`, payload LDS write widths `8` and `4`, and single-stage global prefetch were generated, rebuilt, inspected, and tested on `(32768,8192,2048)` and `(32768,4096,2048)`. All 16 candidates passed correctness and mutation gates, but every non-parent variant was rejected for promotion. Payload LDS write width 4 was clearly slower by `+7.421%` and `+5.956%`. The closest finalists were also slower in independent confirmations:
  - `(32768,8192,2048)`, metadata width 4: parent `40.807545/40.847385 ms`. Finalist `40.831799/40.924603 ms`, or `+0.059%/+0.189%`.
  - `(32768,4096,2048)`, payload global width 8: parent `20.720903/20.733936 ms`. Finalist `20.744219/20.783213 ms`, or `+0.113%/+0.238%`.
- Padded full-tile rows. `(4,0)` padding faulted with `HSA_STATUS_ERROR_MEMORY_FAULT`. Separating external 144-byte addressing and fixing the activation-plane stride removed the initial fault but still produced incorrect or non-finite output for aligned `(16,0)`, `(0,16)`, and `(16,16)` probes. The current ownership map has no padding-aware swizzle, so every nonzero Q3 full-tile row pad is rejected before lowering by the typed layout contract.
- Out-of-contract mechanisms. Prepared or dense weights, external decode storage, split-K, persistent or grouped traversal, producer fusion, hidden caches, and other ABI or grid-ownership changes are deferred as separate mechanisms rather than accepted tuning values.

## Final Review

The final kernel review covered the Q3 GGTensile specification, physical plan, lowering, generated artifacts, HIP Q3 implementation and disassembly, Q3 backward ownership, Q4/Q5/Q6/Q8 forward mechanisms, related backward mechanisms, the Q8_1 workspace ABI, and the target ISA/compiler evidence.

- Retained: typed signed Q3 reconstruction, full-weight 336-byte LDS ownership, decode-ready frontier, persistent WMMA zero source, activation-base hoist, activation-scale deduplication, VMEM batching and overlap, legal VOPD correction, loop-carried weight prefetch, deterministic 200-VGPR resource plan.
- Measured and rejected: all width/pipeline variants, padded-row variants, paired-scale forms, alternate geometries, rolled/batched schedules, barrier/cache variants, fragment mappings, and compiler policies listed above.
- Deferred: a new padding-aware ownership/swizzle map, broader exact-shape qualification, and any mechanism that changes the packed layout, ABI, grid ownership, or workspace contract. Cross-format or backward results do not transfer without a separately derived physical plan and correctness proof.
- Actionable: none remains inside the fixed Q3 forward contract. The remaining optimization envelope has an exact accepted implementation or an explicit rejection. The experiment log is closed.
