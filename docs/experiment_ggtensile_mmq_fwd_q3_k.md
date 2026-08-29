# GGTensile MMQ Forward Q3_K

## Scope

This record covers the gfx1151 wave32 GGTensile MMQ forward kernel for packed Q3_K weights:

```text
output[M,N] = input[M,K] @ dequant_q3_k(weight[N,K]).T
```

The kernel consumes BF16 input represented by the fixed Q8_1 `F32_D4` workspace, decodes the authoritative packed GGUF Q3_K weights in the multiply kernel, accumulates with integer WMMA and FP32 scale correction, and stores BF16 output. Q3_K blocks contain 110 bytes: 32 high-mask bytes, 64 low-payload bytes, and 12 metadata bytes.

The final measured entries use `K=2048`, `Q3FullWeightTiledLds`, `DecodeProducerCount=2`, `Q3FullTile336`, the decode-ready frontier, four barriers, and the canonical arithmetic order. Activation quantization is outside the timing interval. HIP and GGTensile use the same prequantized workspace and the same packed weights.

## Final Benchmark/Profile Results

The reported speed is the effective dense-equivalent rate:

```text
TFLOPS = (2 * M * N * K) / (median_multiply_ms * 1e9)
HIP/GGTensile = HIP median multiply time / GGTensile median multiply time
```

A ratio above `1.0x` favors GGTensile. The final table contains the six exact matrix entries qualified by the final typed full-weight kernel. The hash is the exact kernel identity used for the corresponding artifact.

| Matrix shape `(M,N,K)` | Kernel hash | GGTensile TFLOPS | HIP/GGTensile |
| --- | --- | ---: | ---: |
| `(2048,512,2048)` | `ggsol_7d2eaa3a36b44b3a` | `22.704` | `1.1350x` |
| `(8192,512,2048)` | `ggsol_b24a95504e619789` | `24.918` | `1.1392x` |
| `(32768,512,2048)` | `ggsol_2cf291d183897bc1` | `25.460` | `1.1366x` |
| `(2048,8192,2048)` | `ggsol_a6f505f60a259b51` | `25.057` | `1.1326x` |
| `(8192,8192,2048)` | `ggsol_06b967e9e7315b01` | `25.416` | `1.1377x` |
| `(32768,8192,2048)` | `ggsol_d71bb5adc3c332b2` | `25.517` | `1.1425x` |

The aggregate HIP/GGTensile ratio across these six matrix entries is `1.1408x`. The common final code-object profile is gfx1151, code-object-v5, wave32, a 40-byte kernarg segment, 200 VGPRs, 16 SGPRs, 39,936 bytes of LDS, zero private bytes, zero spills, 128 WMMAs, four barriers, 90 VMEM operations, 274 LDS operations, 134 waits, and eight clauses.

The final typed artifacts passed exact HIP output comparison, finite-output checks, deterministic producer checks, input/packed-weight/workspace mutation checks, and an independent BF16 reference. Independent-reference normalized RMSE was between `0.0060610` and `0.0060684` across the six entries. Two independent generation/build roots reproduced the source, object, code object, and normalized inspection results.

## Accepted Kernel Design

- Typed Q3 reconstruction. The lowerer owns the 110-byte Q3_K block layout, signed 3-bit reconstruction, unsigned six-bit scale decoding with the offset-32 correction, FP16 block factor `d`, and FP32 scale correction. The low payload and high mask are combined before WMMA without changing the packed representation.
- Full-weight LDS ownership. The accepted layout uses 128 Q8_1 activation rows, 64 decoded weight rows, and a 336-byte weight-row stride. The LDS formula is `18,432 + 64 * 336 = 39,936` bytes. The fixed Q8_1 external workspace remains 144 bytes per activation row.
- Typed register and lifetime plan. The final placement keeps raw packed operands, stage addresses, scale shifts, activation staging, persistent WMMA zero operands, and accumulator fragments disjoint. The 200-VGPR profile is spill-free and deterministic.
- Decode-ready frontier. A bottom-up AMDGPU scheduler oracle showed a small improvement when independent payload and scale reconstruction chains were widened before LDS publication. The useful ownership fact was encoded as a typed lowering policy rather than copied as an instruction stream. Top-down scheduling and disabled clustering regressed and were not retained.
- Memory and correction schedule. The accepted body retains the activation LDS-base hoist, activation-scale deduplication, full activation VMEM batching, shared activation/weight VMEM issue, legal VOPD scale correction, loop-carried half-0 weight prefetch, and final-block bounds handling. These mechanisms preserve the four-barrier ownership sequence and the Q3 arithmetic order.
- Fixed contracts. The ABI, wave ownership, WMMA geometry, Q8_1 workspace layout, packed weights, output type, and correction order are fixed kernel contracts. A candidate that changes one of these requires a separate mechanism and qualification record.

The earlier isolated `M=2048,N=4096,K=2048` control established the kernel foundation at 144 VGPRs and 28,672 bytes of LDS. The later full-weight design replaced that compact half-tile ownership with the 200-VGPR, 39,936-byte profile above. The isolated accepted control measured `1.457108 ms` versus HIP at `1.553431 ms`; the final full-weight table is the authoritative result for the six matrix entries above.

## Rejected Kernel Experiments

- Incorrect register aliasing. Reusing temporary decode or address registers across outstanding VMEM operations caused memory faults or incorrect output. The unsafe aliases are rejected; explicit typed lifetimes are required.
- First metadata reuse. Keeping Q3 metadata across both halves overlapped live accumulator fragments and produced grossly incorrect output. The corrected 152-VGPR form was exact but slower, so neither form is accepted.
- Paired scale reads. Four paired weight-scale reads reduced static LDS instructions but added address arithmetic and lost timing. The short-key result was `1.50125x` HIP; the low-pressure variant reached `1.490659 ms` versus HIP at `1.543457 ms` and regressed the retained control. This mechanism is rejected and was not reopened.
- Cross-fragment and batched WMMA schedules. Two-fragment FMAC pairing, 64-result ownership, dependency-derived partial waits, rolled scale groups, and combined rolled/batched forms remained exact in their qualified variants but were slower than the retained schedule. The 200-VGPR batched form measured `1.48564x` HIP on the initial control; the later large-key composition remained above HIP as well.
- Geometry and LDS alternatives. MT64, linear activation staging, compact 320-byte weight rows, dual activation planes, alternate swizzles, and smaller or rolled ownership either failed correctness or lost timing. The 336-byte full-weight row and canonical activation plane remain the accepted layout.
- Schedule-only alternatives. HIP-shaped read order, phased correction, cache invalidation, barrier variants, unroll changes, fragment/bank mappings, and scalar-address variants did not produce a repeatable improvement. Static instruction reductions without a corresponding dependency or lifetime benefit are rejected.
- Compiler-oracle alternatives. Top-down scheduler policies and disabled clustering regressed the exact bitcode. The oracle remains diagnostic evidence; only the typed decode-ready frontier is retained in the kernel.
- Width and pipeline variants. Payload global widths `8` and `4`, metadata widths `8` and `4`, payload LDS write widths `8` and `4`, and single-stage global prefetch were generated, rebuilt, inspected, and tested on `(32768,8192,2048)` and `(32768,4096,2048)`. All 16 candidates passed correctness and mutation gates, but every non-parent variant was rejected for promotion. Payload LDS write width 4 was clearly slower by `+7.421%` and `+5.956%`. The closest finalists were also slower in independent confirmations:
  - `(32768,8192,2048)`, metadata width 4: parent `40.807545/40.847385 ms`; finalist `40.831799/40.924603 ms`, or `+0.059%/+0.189%`.
  - `(32768,4096,2048)`, payload global width 8: parent `20.720903/20.733936 ms`; finalist `20.744219/20.783213 ms`, or `+0.113%/+0.238%`.
- Padded full-tile rows. `(4,0)` padding faulted with `HSA_STATUS_ERROR_MEMORY_FAULT`. Separating external 144-byte addressing and fixing the activation-plane stride removed the initial fault but still produced incorrect or non-finite output for aligned `(16,0)`, `(0,16)`, and `(16,16)` probes. The current ownership map has no padding-aware swizzle, so every nonzero Q3 full-tile row pad is rejected before lowering by the typed layout contract.
- Out-of-contract mechanisms. Prepared or dense weights, external decode storage, split-K, persistent or grouped traversal, producer fusion, hidden caches, and other ABI or grid-ownership changes are deferred as separate mechanisms rather than accepted tuning values.

## Final Review

The final kernel review covered the Q3 GGTensile specification, physical plan, lowering, generated artifacts, HIP Q3 implementation and disassembly, Q3 backward ownership, Q4/Q5/Q6/Q8 forward mechanisms, related backward mechanisms, the Q8_1 workspace ABI, and the target ISA/compiler evidence.

- Retained: typed signed Q3 reconstruction; full-weight 336-byte LDS ownership; decode-ready frontier; persistent WMMA zero source; activation-base hoist; activation-scale deduplication; VMEM batching and overlap; legal VOPD correction; loop-carried weight prefetch; deterministic 200-VGPR resource plan.
- Measured and rejected: all width/pipeline variants, padded-row variants, paired-scale forms, alternate geometries, rolled/batched schedules, barrier/cache variants, fragment mappings, and compiler policies listed above.
- Deferred: a new padding-aware ownership/swizzle map, broader exact-shape qualification, and any mechanism that changes the packed layout, ABI, grid ownership, or workspace contract. Cross-format or backward results do not transfer without a separately derived physical plan and correctness proof.
- Actionable: none remains inside the fixed Q3 forward contract. The remaining optimization envelope has an exact accepted implementation or an explicit rejection. The experiment log is closed.
