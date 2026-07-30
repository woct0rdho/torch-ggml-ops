# GGTensile Dense MMQ Backward Q4_K Experiment Log

## Scope

This document is the current result summary, next plan, and historical evidence log for the gfx1151 dense MMQ backward Q4_K experiment. The first measured pair is `ProblemSize(M=32768, N=2048, K=8192)` and its production companion `ProblemSize(M=32768, N=2048, K=512)`. All campaign shapes use packed Q4_K weights, BF16 inputs and outputs, FP32 WMMA accumulation, wave32, and the existing 40-byte dense-backward kernarg ABI.

The generated kernels compute

```text
grad_input[rows, in_features] = grad_output[rows, out_features] @ dequant(weight[out_features, in_features])
```

without materializing a dense or transposed weight.

The overall generator architecture, multi-quant roadmap, and deferred integration policy remain in [ggtensile_plan.md](ggtensile_plan.md). Results here are exact-shape evidence and must not be generalized without measurement.

## Campaign Contract And Coverage

GGTensile generates a kernel for one exact `ProblemType` and `ProblemSize`. Dense backward uses `M=rows`, `N=in_features`, and `K=out_features`. A generated kernel is required to be correct only for that matrix shape, and runtime dispatch may use it only on an exact key match. This is weaker than the repository's hipBLASLt tuning contract, where a kernel selected while tuning one matrix shape may still run on another shape in its supported assertion domain. GGTensile can therefore use shape constants, exact loop counts, fully peeled tails, fixed launch geometry, and shape-specific ownership or traversal without preserving cross-shape validity. The HIP kernel remains the correctness and compatibility fallback for every unmatched shape.

The dense Q4_K campaign contains only 12 production shapes: the Cartesian product of `M={2048,8192,32768}` and these backward `(N,K)` families:
- `(2048,512)`: narrow K/V/shared gate/up, 70 model calls.
- `(512,2048)`: shared-expert down, 30 model calls.
- `(4096,2048)`: attention output, 10 model calls.
- `(2048,8192)`: attention query/query gate, one model call.

All 12 production shapes now have exact selected solutions. Ten use the retained `128x128` true two-buffer pipeline; attention-output M2048 uses one-buffer `128x64`, and attention-output M8192 uses one-buffer `128x64` with SIA5/store priority. The catalog-driven runner prepares, checks, and confirms the complete matrix. Grouped MMQ is explicitly outside this campaign.

The packaged-HIP values below are historical planning controls. Selected values come from the final byte-normalized 25-repeat matrix and must not be mixed with another process or build when making a new decision.

| Family `(N,K)` | Calls | Historical HIP ms at M2048/M8192/M32768 | Final selected ms at M2048/M8192/M32768 | Decision |
| --- | ---: | ---: | ---: | --- |
| Narrow `(2048,512)` | 70 | `0.230/0.775/2.988` | `0.165/0.622/2.504` | `128x128` two-buffer pipeline at all M |
| Shared down `(512,2048)` | 30 | `0.261/1.537/5.266` | `0.197/1.166/3.425` | `128x128` two-buffer pipeline at all M |
| Attention output `(4096,2048)` | 10 | `1.339/5.912/23.803` | `1.177/4.764/18.740` | `128x64` one-buffer at M2048/M8192; M8192 adds SIA5/store priority; M32768 uses `128x128` two-buffer |
| Query `(2048,8192)` | 1 | `3.376/12.567/48.613` | `2.415/9.739/38.116` | `128x128` two-buffer pipeline at all M |

Weighted optimization priority used fresh `call_count * HIP_median_ms`, not call count alone. Narrow led by calls, attention output led aggregate latency at larger M, shared down showed the strongest HIP underperformance, and query remained last unless a long-K mechanism transferred directly.

## Kernel And Campaign Design

The initial pilot deliberately supported only gfx1151, wave32, WMMA V1, dense backward Q4_K, BF16 inputs and outputs, FP32 accumulation, exact tile-divisible shapes, `in_features == 2048`, one complete output tile per workgroup, and the existing 40-byte ABI. Production coverage later generalized exact N to `{512,2048,4096}` while retaining exact-key dispatch. Split reduction, atomics, edge masks, persistent traversal, and external decode workspaces remained outside the campaign.

The pilot used a `128x128x32` work tile and 128 threads in four wave32 waves. Each wave owned 32 output rows as two M16 WMMA tiles. Every reduction iteration:
- cooperatively loaded Q4_K headers and packed nibbles for a `32x128` weight tile.
- unpacked six-bit scale/min fields, decoded the GGUF formula, converted to BF16, and staged transposed WMMA-facing weights in LDS.
- synchronized, loaded BF16 activation fragments from global memory, read weight fragments from LDS, and issued 32 static BF16 WMMAs.
- synchronized before LDS reuse and advanced to the next reduction tile.
- stored one exact output tile without edge predicates.

The gfx1151 contracts were checked against `~/rdna35-isa-markdown/` and `~/amd-llvm-project/`. Packed work-item X/Y was flattened before `v0` became accumulator storage. `v_fma_mix_f32` operand selection, BF16 WMMA V1 lane replication and accumulator mapping, sub-dword LDS/global stores, explicit BF16 round-to-nearest-even, wait dependencies, and VOPD slot/bank rules were validated against ISA and LLVM definitions and by execution.

The manual search followed the measurement discipline from `~/ComfyUI-FeatherOps/doc/tensile_fp16_nt_hhs.md`: start from a measured control, change complete solutions in focused neighborhoods, keep correctness as a hard gate, use shorter timing only for screening, retime finalists with fresh hot-loop brackets, and compare normalized assembly and resources around winners. TensileLite supplied mechanism vocabulary; a parameter entered this experiment only after a distinct emitter changed ISA or ownership.

ROCm rocm-libraries PR 9385 supplied additional gfx1151 mechanism evidence: correct sub-dword WMMA local reads, capping local-read buffers by actual loop reuse, placing long-lived values before transient address registers, and tuning interactions rather than one universal configuration. The campaign applied those principles where they matched fused decode instead of copying general-GEMM paths.

## Latest Result

### Selected solution

The default selected kernel is the four-wave true decoded-B pipeline:
- `MacroTile=128x128`, `DepthU=32`, and `WorkGroup=[32,4,1]`.
- WGM1 traversal and XOR-8 decoded-B LDS layout.
- SIA4, PGR2, and PLR1.
- `1LDSBuffer=0`, giving two 8 KiB decoded-B buffers.
- no store priority, no packed-next-only prefetch, and no packed-weight lane sharing.
- byte-normalized packed-dword decode, scalar-base global loads and stores, compact address VGPRs, power-of-two row-stride shifts, accumulator-clear VOPD, no fixed final WMMA padding, and no `buffer_gl0_inv`.

The default artifact uses 212 VGPRs, 16 SGPRs, and 16 KiB LDS. It has no private storage, spills, scratch instructions, calls, or dynamic stack. Static inspection sees 64 WMMAs, 128 LDS instructions, 164 VMEM instructions, 861 non-WMMA VALU issues representing 883 operations, 22 VOPD pairs, 23 waits, and two barriers. WMMAs and LDS instructions appear in both the steady and peeled-final static paths, so those counts are not per-iteration dynamic counts. The two attention M2048/M8192 variants use 140 VGPRs, 16 SGPRs, 4 KiB LDS, 354 non-WMMA VALU issues representing 376 operations, 22 VOPD pairs, 77 VMEM instructions, 32 LDS instructions, and 16 WMMAs.

### Performance

The final byte-normalized catalog passes a fresh 25-repeat rotating matrix at 671.12 ms call-weighted candidate latency versus 860.65 ms for HIP, a ratio of 0.77978 and a 22.02% reduction. In M2048/M8192/M32768 order, selected medians are narrow `0.165/0.622/2.504` ms, shared down `0.197/1.166/3.425` ms, attention output `1.177/4.764/18.740` ms, and query `2.415/9.739/38.116` ms. Family reductions are 19.63% narrow, 30.57% shared down, 18.48% attention output, and 21.88% query. Every exact key beats HIP.

Throughput below is derived from each final confirmation median as `2*M*N*K / (median_ms * 1e9)`. It is the arithmetic throughput of the complete fused kernel, not a separate WMMA-only rate. GGTensile is faster than HIP on all 12 exact keys, with gains from 21.6% to 51.4% in TFLOPS.

| Family | M | N | K | HIP TFLOPS | GGTensile TFLOPS | GGTensile / HIP | Difference TFLOPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Narrow | 2048 | 2048 | 512 | 19.414 | 25.952 | 1.337x | +6.537 |
| Narrow | 8192 | 2048 | 512 | 21.014 | 27.610 | 1.314x | +6.596 |
| Narrow | 32768 | 2048 | 512 | 22.482 | 27.446 | 1.221x | +4.964 |
| Shared down | 2048 | 512 | 2048 | 16.434 | 21.814 | 1.327x | +5.380 |
| Shared down | 8192 | 512 | 2048 | 11.849 | 14.734 | 1.244x | +2.885 |
| Shared down | 32768 | 512 | 2048 | 13.255 | 20.065 | 1.514x | +6.810 |
| Attention output | 2048 | 4096 | 2048 | 24.004 | 29.191 | 1.216x | +5.187 |
| Attention output | 8192 | 4096 | 2048 | 22.883 | 28.852 | 1.261x | +5.969 |
| Attention output | 32768 | 4096 | 2048 | 24.072 | 29.336 | 1.219x | +5.264 |
| Query | 2048 | 2048 | 8192 | 19.131 | 28.456 | 1.487x | +9.326 |
| Query | 8192 | 2048 | 8192 | 21.732 | 28.226 | 1.299x | +6.494 |
| Query | 32768 | 2048 | 8192 | 22.857 | 28.846 | 1.262x | +5.990 |

The largest relative improvement is shared-down M32768 at 51.4%; the smallest is attention-output M2048 at 21.6%. The largest absolute throughput improvement is query M2048 at 9.326 TFLOPS. The final weighted latency result above remains the model-level comparison because it includes production call counts.

Against the no-final-NOP assembly catalog, the final matrix improves 1.50% call-weighted. The focused M32768 decoder confirmation measured K8192 at 38.0774 ms versus 38.5496 ms, a 1.22% reduction, and K512 at 2.5053 ms versus 2.5525 ms, a 1.85% reduction. The focused pair improves 1.74% call-weighted. A separate shared-down M8192 confirmation measured 1.1628 ms versus 1.1791 ms, a 1.39% gain, so the one regressing complete-matrix sample did not reproduce.

The earlier one-buffer comparison remains the pipeline mechanism control: K8192 measured 38.012 ms versus 39.467 ms, a 3.69% reduction, and K512 measured 2.5063 ms versus 2.6654 ms, a 5.97% reduction. The current complete-matrix K8192 result sustains 28.85 TFLOP/s, about 48.6% of the approximately 59.4 TFLOP/s gfx1151 BF16 WMMA roof and about 64% of the approximately 45 TFLOP/s delivered by a tuned hipBLASLt BF16 GEMM on this machine. hipBLASLt is an upper reference rather than a fused-kernel target because GGTensile repeatedly reads packed data, reconstructs BF16 weights, stages LDS, and synchronizes. Reaching 30 TFLOP/s requires approximately 36.65 ms, another 3.8% from the final selected result.

### Correctness

Every exact key in the final byte-normalized catalog matches HIP bit-for-bit before and after complete `grad_output` rewrites and packed-weight mutation. This covers all 12 production shapes and guards the retained L0-invalidation omission. Reduced K32/K64/K96/K512 execution and a synthetic one-hot matrix additionally cover all eight Q4_K groups, both nibble halves, scale packing, and boundary K rows.

Against independently dequantized BF16 matmul on the original M32768 N2048 controls:
- K8192 candidate and HIP each differ in 409,880 BF16 elements, with maximum absolute error 0.0625 and normalized RMSE 0.0001924804.
- K512 candidate and HIP each differ in 16,684 BF16 elements, with maximum absolute error 0.00390625 and normalized RMSE 0.0000345316.

Independent source generation is byte-reproducible, and both artifacts pass the hard resource and ABI gates.

## Work Completed

### Measurement and inspection infrastructure

- `tools/benchmark_ggtensile.py` provides warmed rotating HIP, candidate, and assembly-control timing; full-output comparison; independent BF16-reference comparison; and producer-handoff checks.
- Artifact inspection records ABI and hard resources plus static VALU issues and operations, VOPD pairs, VMEM, LDS, waits, barriers, clauses, dependency delays, and L0 invalidations.
- `tools/benchmark_ggtensile_lower_bounds.py` generates, builds, inspects, and rotates exact `wmma_floor` and `decode_floor` diagnostic artifacts while leaving production `Solution` and dispatch contracts unchanged.
- `tools/ggtensile/q4_k_dense_inventory.json` records the exact 12 keys, representative tensors, call counts, historical HIP controls, validation contract, and selected-solution references without a schema version. `tools/ggtensile/q4_k_selected_solutions.json` contains the three complete selected solution mappings. `tools/run_ggtensile_q4_k_campaign.py` runs immutable prepare, correctness, nine-repeat screen, and 25-repeat confirmation phases serially and reports call-weighted totals.

### Campaign search space and control taxonomy

The writer emitted genuinely different Q4_K paths for:
- implemented matrix instruction, macro-tile, workgroup, and DepthU geometries.
- exact-M `WorkGroupMapping` values.
- implemented SIA, PGR, and PLR schedules.
- one decoded-B LDS buffer versus the true two-buffer pipeline.
- LDS swizzle chunks `0`, `4`, `8`, and `16`.
- store priority, packed-next-only prefetch, and packed-weight lane sharing.

A rejected value was reconsidered on another exact key only when shape changed its cost model. Short K changed prologue and epilogue weight; narrow N changed decode duplication, LDS ownership, and launch locality. Every reopened scan recorded that changed premise.

The following fields remained fixed identity because alternate values did not have complete distinct emitters: `DecoderWidth=16`, both global-read vector widths, `LocalReadVectorWidth=16`, `NumElementsPerBatchStore=8`, `StoreVectorWidth=1`, `TransposeLDS=0`, both LDS pad fields, and `PrefetchPackedWeight=true`. In particular, `NumElementsPerBatchStore` did not control the assembly store loop and was not treated as a tuning knob.

Potential mechanisms stayed internal rather than becoming public controls: active compute/decoder ownership, other decoder widths and chunk sizes, metadata-sharing and packed-payload assignments, independent activation/payload/metadata prefetch leads, explicit wait budgets, alternate metadata and payload load widths, new padded LDS formulas, real store batching or remap, and fixed-trip unroll policy. Each required two complete validated emitters before admission to `Solution`.

Exact-shape lowering preceded resource-bearing experiments. The checklist was to remove dead kernarg state; fold dimensions, strides, tile counts, launch divisors, and packed block offsets; specialize fixed trips and peel impossible tails; strength-reduce affine addresses without extending live ranges; choose immediate, scalar-base, or explicit addressing from exact geometry; recompute lifetimes after ownership changes; and form VOPD only under gfx1151 opcode and bank legality. Retained and rejected outcomes are recorded below.

### Retained kernel mechanisms

- Corrected global traversal from the initial all-M ordering to WGM1. This reduced an early SIA3 path from about 119.36 ms to 52.40 ms. WGM2, WGM4, and WGM8 were slower.
- Selected XOR-8 LDS addressing after traversal was corrected. A 25-repeat bracket measured XOR-8 at 45.136 ms versus 52.604 ms unpadded. XOR-8 and HIP reported 68.75% LDS conflicts versus 79.17% unpadded; LDS issue count, not the conflict percentage alone, explained the rejected alternatives.
- Selected SIA4, which overlaps first-half A completion with fused B decode. It measured 42.666 ms versus 45.137 ms for SIA3, a 5.47% latency reduction.
- Selected PGR2, which keeps both A halves in separate fragment sets. It measured 40.906 ms versus 42.723 ms for PGR1, a 4.25% reduction.
- Kept PLR1 and `StorePriorityOpt=false`; broader local prefetch and store priority did not help.
- Implemented and retained the true two-buffer decoded-B pipeline. It primes B0, decodes B1 into the opposite LDS buffer in eight chunks while WMMA consumes B0, reloads next A after current use, performs one steady swap barrier, and peels the final tile to avoid out-of-range prefetch.

### Retained unconditional writer simplifications

These changes are bit-exact, resource-neutral or resource-reducing, and neutral-to-favorable under rotating controls:
- Reuse prefetched A pointers: 42.646 ms versus 43.081 ms, a 1.01% reduction.
- Direct nibble extraction: removes 28 VALU instructions per DepthU iteration and measured 40.644 ms versus 41.210 ms, a 1.37% reduction.
- Hoist invariant A coordinates through reordered Q4_K scale temporaries: removes seven VALU instructions per iteration and measured a 0.65% gain.
- Store invariant A byte offsets instead of row indices: removes two multiplies per iteration and measured a 0.33% gain.
- Use direct packed-Q and scale offset formulas for the selected 128-wide N tile: removes three VALU instructions per iteration and measured a 0.44% gain.
- Use gfx11 scalar-base addressing for exact-shape A and packed-weight loads: removes vector high-half and carry chains and measured a 1.03% gain.
- Compact the address block from 12 to 8 VGPRs: reduces allocation from 216 to 212 VGPRs and was neutral-to-favorable.
- Use scalar-base output stores: removes vector pointer carry chains and was neutral-to-favorable on K8192 and K512.
- Pair 20 accumulator clears with independent pre-loop integer VALU using gfx1151 VOPD: reduces static non-WMMA VALU issues from 675 to 655 in the one-buffer body. A 25-repeat bracket was slightly favorable on both shapes.
- Remove `buffer_gl0_inv`: K8192 measured 39.757 ms versus 39.886 ms; K512 was neutral within 0.15%. Producer-handoff checks establish correctness after input updates.
- Remove unused runtime dimension kernarg loads while preserving the 40-byte ABI: allocation falls from 20 to 16 SGPRs. Serial 25-repeat brackets measured K8192 at 38.2548 ms versus 38.3512 ms, a 0.25% reduction, and K512 at 2.5224 ms versus 2.5331 ms, a 0.42% reduction. Both production tensors and producer-handoff mutations remain bit-exact to HIP.
- Specialize the decoded-B pipeline loop for the exact K trip count: remove the pre-stage exit test and unconditional back branch, then test the next-trip condition after the steady stage. K32 emits no steady body and passes the reduced trip tests at K32/K64/K96/K512. Serial 25-repeat brackets measured K8192 at 38.1932 ms versus 38.3088 ms, a 0.30% reduction, and K512 at 2.5054 ms versus 2.5236 ms, a 0.72% reduction. Static resources and issue counts are unchanged.
- Generalize exact production N coverage by deriving Q4_K packed-row bytes as `(N/256)*144` and accepting only N512/N2048/N4096. Duplicate `(2048,512,2048)` and `(2048,4096,2048)` artifacts are reproducible and resource-clean. Shared-down and attention-output tensors are bit-exact to HIP, including both producer mutations. One-repeat execution checks measured about 0.201 versus 0.293 ms and 1.166 versus 1.310 ms respectively; these are coverage checks, not retention brackets.
- Strength-reduce exact power-of-two A and output row strides to shifts while retaining multiply fallback for reduced K96. Two shifts pair with accumulator clears, increasing VOPD pairs from 20 to 22 and reducing static VALU issues from 903 to 901 with unchanged resources. Serial 25-repeat brackets are neutral on K512 and improve K8192 by 0.30%; all producer checks remain bit-exact.
- Remove the fixed final `s_nop 7`. Output-address setup already separates the final WMMA from the first accumulator consumer, and gfx1151 resolves the dependency without fixed SALU padding. All 12 exact keys and both producer mutations remain bit-exact. Focused K512/K8192 brackets improve 0.50%/0.36% with unchanged resources and no remaining `s_nop`.
- Normalize each packed Q4_K dword once with the lane-selected nibble shift and `v_and_b32 0x0f0f0f0f`, then convert its four byte fields with `v_cvt_f32_ubyte0..3`. This removes the four packed-nibble offset builders and 40 N128 static VALU issues without changing allocation, LDS, VMEM, barriers, or WMMAs. Focused K512/K8192 confirmation improves 1.85%/1.22%, and the complete matrix improves 1.50% against the no-final-NOP assembly control. The common lowering remains selected after a separate shared-down M8192 bracket improved 1.39%.

### True-pipeline validation and profiling

The first K64 pipeline launch exposed a live-range error: decode preparation reused `v205:v208`, which still held LDS read addresses for later WMMA pairs. The first N pair was correct and the remaining pairs were misaddressed. The corrected writer uses temporary addresses for the first pair, then moves steady read addresses into Q4_K metadata VGPRs after that metadata dies. Reduced real-weight tests are bit-exact at K32, K64, K96, and K512, covering prime/final execution, one overlap, repeated swaps, and next-A handoff.

A fair raw-counter comparison of current artifacts showed that the two-buffer pipeline reduces median aggregate wave cycles by approximately 4.9%, wait-count stalls from 23.74% to 21.38%, and barrier stalls from 8.58% to 6.11%. LDS instruction count is unchanged. Additional address and peeled-path VALU is outweighed by overlap.

### Exact lower bounds

The diagnostics preserve the selected ABI, 128-thread workgroup, 212-VGPR declaration, 16 SGPRs, and 16 KiB LDS occupancy. Each generated source is hashed, and each code object passes the normal symbol, metadata, register, private-storage, spill, scratch, stack, and call checks with diagnostic-specific WMMA and barrier expectations.
- K8192: complete pipeline 37.868 ms, WMMA/A/LDS floor 24.149 ms or 45.53 equivalent TFLOP/s, and decode/LDS floor 16.842 ms. The floor sum is 1.0825 times complete latency; complete is 1.568 times the dominant WMMA floor.
- K512: complete pipeline 2.3131 ms, WMMA floor 1.6197 ms or 42.43 equivalent TFLOP/s, and decode floor 1.0915 ms. The floor sum is 1.1721 times complete latency; complete is 1.428 times the dominant floor.

The complete kernel already beats the sum of isolated floors through overlap, but the remaining gap to the WMMA/A/LDS floor confirms that the hardware ceiling is not yet exhausted.

Attention-output M32768 measures 18.686 ms complete, 12.008 ms for the WMMA/A/LDS floor at 45.78 equivalent TFLOP/s, and 8.532 ms for the decode/LDS floor. The floor sum is only 1.099 times complete. This balanced overlap does not support cooperative A staging as a large-margin direction, especially because it would add LDS traffic and require the rejected eight-wave/one-buffer regime.

### Family-level decisions

- Narrow K512 retains the `128x128` two-buffer pipeline at all three M values. One buffer, half tiles, WGM alternatives, body unroll, and wider high-M ownership failed their gates; exact-trip and instruction lowerings carried the useful short-K gains.
- Shared-down N512/K2048 retains the same pipeline. `256x64` regressed weighted family latency by 101%, and `256x128` by 47%; reduced packed-B duplication did not repay one-buffer scheduling and eight-wave residency.
- Attention-output N4096/K2048 is the only family with selected half-tile ownership. M2048 and M8192 use one-buffer `128x64`; M8192 additionally selects SIA5 plus store priority. M32768 retains two-buffer `128x128`. XOR-8 and WGM1 remain selected for the N64 geometry.
- Query K8192 retains the true two-buffer pipeline and WGM1 at all M values. Lower bounds show useful overlap already, so only a materially different mechanism with a plausible multi-percent path can reopen it.

Traversal was treated as exact-shape behavior. The complete Q4_K WGM2/WGM4/WGM8 matrices regressed weighted latency by 1.45%/5.22%/16.27% against WGM1. The two sub-gate M2048 alternate wins decayed at larger mappings and did not justify per-key exceptions.

Scheduling followed selected ownership and LDS layout. The campaign named placement of packed and metadata reads, decode chunks, activation reads, LDS operations, WMMAs, waits, barriers, and stores; numeric SIA values were only aliases for those concrete streams. Clauses, dependency delays, priorities, and explicit VGPR deallocation were admitted only after ISA inspection and direct timing.

## Revisit And Completion Policy

### Revisit policy for the measured pair

There is no currently open sub-percent tuning neighborhood. Clauses, dependency batching, early A0 movement, packed-next-only prefetch, lane sharing, local-read double buffering, store priority, and ordinary geometry changes have measured neutral or negative results. Reopening one requires a materially different resource or scheduling regime.

A future K8192 attempt should begin only with a mechanism plausibly worth more than the approximately 3.8% needed to cross 30 TFLOP/s. Candidate ideas must explain how they improve on the measured two-buffer overlap without repeating the rejected eight-wave ownership cost. Packed-dword normalization cannot form a gfx1151 VOPD pair: `v_and_b32` is Y-only, `v_lshrrev_b32` is absent from gfx11 VOPD, and the mask consumes the shift result.

Any new resource-bearing mechanism must remain bit-exact, reproducible, resource-clean, and more than 2% faster on both K8192 and K512. Given the explicit request to prioritize larger margins, a practical threshold for revisiting this case is closer to the remaining 3.8% K8192 gap.

### Twelve-shape dense Q4_K campaign

The dense Q4_K 12-key optimization campaign and its repeated optimization-exhaustion review are complete. The selected-solution catalog, exact validation, final correctness, and byte-normalized weighted 25-repeat matrix are the current controls. Further Q4_K work requires a newly actionable large-margin direction with a changed premise; ordinary local tuning is closed.

Quant-family expansion and public runtime dispatch remain governed by [ggtensile_plan.md](ggtensile_plan.md). Public integration remains deferred until dispatch engineering, complete end-to-end Qwen/DeepSeek validation, and broader dense multi-quant coverage are complete; exact-kernel campaign completion does not by itself authorize runtime exposure.

Prepared weights, BF16 shadows, external decode workspaces, GSU, Stream-K, and persistent workgroups remain outside this experiment contract unless the broader plan explicitly accepts their ownership, workspace, or fixup requirements.

### Campaign sequence and completion

- Recorded the versionless 12-key inventory with representative tensors, call counts, historical controls, validation requirements, and selected-solution references.
- Added the serial immutable runner for generation, build, inspection, correctness, screening, and confirmation without overlapping GPU phases.
- Applied exact-shape lowering. Dead kernarg dimensions, fixed-trip branches, row-stride shifts, final-NOP removal, and byte-normalized decode were retained; body unroll and resource-growing induction were rejected.
- Established retained-pipeline, one-buffer, traversal, geometry, schedule, and lower-bound controls for every materially different family.
- Optimized all exact keys by weighted priority and recorded three selected solution mappings covering ten `128x128` pipeline keys and two attention `128x64` keys.
- Reconfirmed every finalist. The final byte-normalized catalog passed all 12 exact keys, both producer mutations, independent rebuilds, strict inspection, and the weighted 25-repeat matrix.
- Repeated the complete optimization-exhaustion review from the final retained state. It found no new actionable in-contract mechanism.

The campaign stopping condition is met. A future change that creates a credible new premise must execute that experiment and repeat step 7 before exhaustion can be claimed again.

### Final exhaustion classification

The permanent review was rerun from the retained byte-normalized, no-final-NOP state and found no new actionable in-contract mechanism. Directly measured geometry, traversal, one/two-buffer ownership, schedules, waits, metadata width, packed sharing, addressing, epilogue priority, and fixed-trip unroll remain closed. Paired BF16 LDS stores require cross-lane packing and EXEC work; a `128x256` tile requires cross-wave A sharing already closed by lower-bound evidence; gfx1151 has neither useful temporal cache hints nor a bit-exact packed FP32-to-BF16 RNE instruction. Persistent workgroups, split K, and prepared representations expand the launch, fixup, workspace, or model-ownership contract. Any future actionable idea must be executed and followed by another complete review before this stopping condition can be claimed again.

## Rejected And Closed Experiments

### LDS layouts

- XOR-16 was bit-exact and reduced allocation to 198 VGPRs, but restored the 79.17% conflict ratio and measured 52.633 ms versus 45.006 ms for XOR-8, a 16.95% regression.
- XOR-4 retained the 68.75% conflict ratio but required four `ds_load_b64` operations per fragment instead of two `ds_load_b128` operations. It measured 45.178 ms versus 45.053 ms, so doubled LDS issue count had no compensating gain.

### Geometry and ownership

- `256x64x32`: bit-exact at 208 VGPRs and 4 KiB LDS, but 66.597 ms versus 45.300 ms, a 47.0% regression. Halving decoded-B work did not repay doubled A traffic.
- `256x128x32`: bit-exact at 200 VGPRs and 8 KiB LDS, but 50.469 ms versus 46.021 ms, a 9.66% regression. Eight-wave scheduling and residency outweighed reduced decode.
- `64x128x32`: bit-exact at 136 VGPRs and 8 KiB LDS, but 69.308 ms versus 41.129 ms, a 68.5% regression. More workgroups duplicated packed decode.
- `128x64x32`: bit-exact at 144 VGPRs and 4 KiB LDS, but 46.317 ms versus 41.245 ms, a 12.3% regression. More N workgroups duplicated A traffic.
- `128x128x64`: bit-exact at 236 VGPRs and 16 KiB LDS. It measured 40.394 ms versus 40.772 ms, only a 0.93% gain, which did not justify 20 additional VGPRs and doubled LDS.
- Dedicated decoder waves: an exact eight-wave workgroup used four compute waves and four decoder waves. After correcting reversed `v_sub_nc_u32` decoder-ID normalization, reduced K32/K64/K96/K512 tests were bit-exact and the artifact was clean at 212 VGPRs, 21 SGPRs, and 16 KiB LDS. Production K512 measured 2.7300 ms versus 2.4951 ms, a 9.41% regression. The path was removed without a K8192 run.

### Scheduling and prefetch

- PLR2 added 16 VGPRs and measured 46.521 ms versus 46.120 ms, a 0.87% regression.
- SIA5 was resource-identical to SIA4/PGR2 and screened 0.93% faster, below the retention threshold.
- Packed-next-only prefetch was bit-exact and resource-identical but measured only a 0.15% gain. It moved VMEM without overlapping complete next-tile decode.
- Two-value dependency-batched decode was bit-exact and resource-identical and improved both 25-repeat controls by about 0.78%; four-value batches regressed K8192 by 0.62% and K512 by 1.57%. Both were rejected under the larger-margin requirement.
- Moving next-A0 loads to their first-half death point measured 39.618 ms versus 39.278 ms on K8192 and 2.5146 ms versus 2.5299 ms on K512. The mixed 0.87% regression and 0.61% gain was rejected.
- A-load `s_clause 3` pairs measured gains of only 0.035% on K8192 and 0.29% on K512. Added SALU and constrained arbitration were not justified.
- The complete one-buffer matrix is slower than the retained two-buffer pipeline on all 12 keys. Nine-repeat three-way screens regress individual keys by 3.49-24.65% and weighted latency by 8.97%, despite reducing LDS from 16 KiB to 8 KiB. Lost fused decode/WMMA overlap dominates at every production M/N/K.
- Complete WGM2/WGM4/WGM8 matrices regress weighted latency by 1.45%/5.22%/16.27% against WGM1. WGM1 wins 10 keys; the only alternate wins are sub-gate M2048 results of 1.61% for narrow WGM2 and 0.54% for attention-output WGM2, and both decay at larger mappings. WGM1 remains the traversal seed.
- Four-wave half-tile geometry scans reject `64x128` and retain `128x128` for ten keys. One-buffer `128x64` is selected for attention-output M2048 and M8192: fresh 25-repeat brackets measure 1.1909 versus 1.2700 ms, a 6.23% gain, and 4.9407 versus 5.0825 ms, a 2.79% gain. The selected artifacts use 140 VGPRs, 16 SGPRs, and 4 KiB LDS. Attention-output M32768 regressed in the screen and retains two-buffer `128x128`.
- A current focused `256x64` scan first exposed and then validated a stale SIA2/3 A-coordinate bug. After correction, all six narrow/attention keys are bit-exact, but narrow regresses by 4.69-49.84%, attention M8192/M32768 regress by over 24%, and attention M2048 gains only 0.94% versus retained `128x128`, which is slower than selected `128x64`. `256x64` remains rejected.
- Shared-down high-M ownership is decisively rejected. `256x64` regresses weighted family latency by 101%; eight-wave `256x128` regresses by 47%, with only M2048 near neutral at a 2.57% regression. Halving repeated B decode does not repay one-buffer scheduling and residency losses.
- Eight-wave `256x128` also regresses narrow by 16-30% and attention output by 9-39%. A larger workgroup without a new cross-wave sharing mechanism is closed across the production families.
- True two-buffer `64x128` and `128x64` pipelines were implemented and passed reduced K32/K64/K96 plus all 12 exact production checks. Their complete screens regressed weighted latency by 71.95% and 4.39%; two-buffer `128x64` also lost by roughly 4-6% to the selected one-buffer attention artifacts. The alternate pipeline emitters were removed and the strict solution surface remains narrow.
- Selected N64 attention geometry retains XOR-8 on every M. XOR-4 regresses weighted attention latency by 3.57%, while unpadded and XOR-16 regress by about 20%.
- Selected N64 attention geometry also retains WGM1. WGM2/WGM4/WGM8 regress weighted M2048/M8192 latency by 3.30%/10.81%/37.47%; M2048's best alternate gain is only 0.15%.
- Attention-output M8192 further selects SIA5 plus store priority. It confirms 4.8913 versus 5.0526 ms, a 3.19% gain over SIA4/no-priority, then confirms another 2.24% directly against store-priority-only at about 4.8884 ms. M2048 retains SIA4/no-priority because its individual changes were below 0.9%.
- Store priority on the complete two-buffer matrix is neutral or worse on 11 keys. Query M2048's only 2.01% screen signal confirms at 2.4509 versus 2.4869 ms, just 1.45%, and is rejected.
- Duff-style exact-trip body unroll factors 2/4/8 and complete K512 factor 15 or K8192 factor 16 were bit-exact and resource-clean, but grew source from about 54 KiB to 69-278 KiB and code objects from 13.8 KiB to 16.4-53.5 KiB. Nine-repeat K512 screens gained at most 0.97%; K8192 screens ranged from neutral to 0.63% slower. None reached the 2% resource-bearing gate, so no factor advanced or remains in the writer.

### Packed-weight sharing and addressing

- `PackedWeightLaneShare=2` halved duplicated packed-Q bytes. The LDS crossbar regressed 1.15%; the DPP/VALU crossbar regressed 1.88%. Cross-lane work cost more than the saved traffic.
- Persistent A-pointer hoisting raised allocation from 200 to 204 VGPRs and regressed 0.23%; the resource-neutral coordinate/byte-offset hoists were retained instead.
- A persistent lane-specific packed-row byte offset reduced static VALU operations by three and issues by four, with one more VOPD pair, but raised allocation from 212 to 213 VGPRs. Nine-repeat screens gained only 1.37% on K512 and regressed K8192 by 0.09%, so it failed the resource-bearing gate and was removed.
- Replacing the packed-row multiply by `v_lshl_add_u32` plus a shift used one extra dynamic issue per tile. It remained bit-exact but gained only 0.61% on K512 and 0.45% on K8192, so the single multiply remains selected.
- Coalescing each N128 Q4_K metadata header into one 16-byte load reduced static VMEM by 12 issues across prime/steady bodies but added 10 VALU extracts. It passed one-hot scale/nibble formulas and reduced K32/K64/K96/K512 execution after correcting a changed VMEM wait budget, then gained only 0.07% on K512 and regressed K8192 by 0.28%. The emitter was removed.

### Epilogue and traversal

- Store priority was neutral: no priority measured 45.327 ms versus 45.413 ms. The simpler no-priority path remains selected.
- WGM2, WGM4, and WGM8 screened at 54.97, 56.86, and 57.46 ms after traversal correction; WGM1 won every comparison.
- Adding LLVM's explicit `s_nop 0; s_sendmsg sendmsg(MSG_DEALLOC_VGPRS); s_endpgm` tail preserved correctness and allocation but regressed K512 by 1.71%, improved K8192 by only 0.45%, and regressed the call-weighted pair by 1.32%. Direct `s_endpgm` remains selected.

## Debugging Lessons Worth Preserving

- gfx1151 packed work-item X/Y must be flattened before `v0` becomes accumulator storage.
- Assembly and metadata inspection cannot replace execution tests. DU64 initially advanced the reduction counter by 32 instead of `DepthU`, then used hardcoded 64-byte LDS row strides; both faults assembled and passed structural checks before execution exposed them.
- Compact address-register changes require auditing every consumer. An early compact artifact still decoded through old `address+10/+11` indices and failed correctness until updated to `address+6/+7`.
- The true pipeline showed that temporary lifetime is part of correctness: interleaved decode clobbered later-pair LDS addresses even though one-buffer execution was valid.
- The dedicated-wave fault showed that `v_sub_nc_u32` operand order can turn a local-ID correction into a 32-bit global-address wrap. Reduced K32/K64 tests should precede production launches for every new control-flow or induction regime.
- Static counts for peeled or branched kernels include mutually exclusive paths and must not be interpreted as dynamic work without control-flow analysis.
- Hoisted address state must be scoped to schedules that consume it. SIA4/5 use precomputed A byte offsets, while SIA2/3 rebuild row coordinates inside `_emit_wmma`; applying both transformations multiplied the row stride twice and corrupted about half of a `256x64` output despite clean assembly and metadata.
- Broad derived-counter groups may exceed gfx1151 collection capabilities; preserve supported raw groups and compare artifacts under the same collection protocol.

## Evidence Locations

- Initial K8192 baseline: `/tmp/ggtensile-m32768-n2048-k8192-baseline/benchmark.json`.
- Retained K8192 pipeline: `/tmp/ggtensile-m32768-n2048-k8192-true-pipeline/`.
- Retained K512 pipeline: `/tmp/ggtensile-m32768-n2048-k512-true-pipeline/`.
- Current one-buffer and pipeline profiles: `/tmp/ggtensile-profile-current-one-buffer/` and `/tmp/ggtensile-profile-true-pipeline/`.
- K8192 and K512 lower bounds: `/tmp/ggtensile-m32768-n2048-k8192-lower-bounds/` and `/tmp/ggtensile-m32768-n2048-k512-lower-bounds/`.
- Attention-output M32768 lower bounds: `/tmp/ggtensile-m32768-n4096-k2048-lower-bounds-a/`.
- Dead-kernarg lowering and serial 25-repeat brackets: `/tmp/ggtensile-m32768-n2048-k8192-dead-kernargs-a/` and `/tmp/ggtensile-m32768-n2048-k512-dead-kernargs-a/`.
- Exact-trip branch lowering and serial 25-repeat brackets: `/tmp/ggtensile-m32768-n2048-k8192-postcheck-loop-a/` and `/tmp/ggtensile-m32768-n2048-k512-postcheck-loop-a/`.
- Power-of-two row-stride lowering and serial brackets: `/tmp/ggtensile-m32768-n2048-k{512,8192}-row-shifts-a/`.
- Rejected persistent packed-row artifacts: `/tmp/ggtensile-m32768-n2048-k{512,8192}-persistent-packed-row-a/`.
- Rejected packed shift-add artifacts: `/tmp/ggtensile-m32768-n2048-k{512,8192}-packed-shift-add-a/`.
- Rejected wide-metadata artifacts: `/tmp/ggtensile-m32768-n2048-k{512,8192}-wide-metadata-a/`.
- Retained no-final-NOP catalog and complete correctness: `/tmp/ggtensile-q4-k-no-final-nop-a/`.
- Retained byte-normalized focused controls: `/tmp/ggtensile-q4-k-byte-nibbles-a/`.
- Final byte-normalized 12-key catalog, correctness, and confirmation: `/tmp/ggtensile-q4-k-byte-nibbles-final-a/`.
- Shared-down M8192 isolated recheck: `/tmp/ggtensile-q4-k-byte-nibbles-shared-down-m8192-recheck-a/`.
- Rejected explicit VGPR-deallocation tail: `/tmp/ggtensile-q4-k-dealloc-vgprs-a/`.
- Rejected body-unroll artifacts use `/tmp/ggtensile-m32768-n2048-k{512,8192}-unroll{factor}-a/`, with factor-specific generate, inspect, correctness, and nine-repeat timing evidence.
- Exact production-N coverage artifacts: `/tmp/ggtensile-m2048-n512-k2048-production-n-a/` and `/tmp/ggtensile-m2048-n4096-k2048-production-n-a/`.
- Complete retained-pipeline prepare, correctness, and nine-repeat screen: `/tmp/ggtensile-q4-k-retained-matrix-a/`.
- Catalog-driven final prepare, correctness, and 25-repeat matrix: `/tmp/ggtensile-q4-k-selected-catalog-final-a/`.
- Rejected complete one-buffer correctness and three-way screen: `/tmp/ggtensile-q4-k-one-buffer-matrix-a/`.
- Rejected complete traversal matrices: `/tmp/ggtensile-q4-k-wgm{2,4,8}-matrix-a/`.
- Geometry matrices and attention-output confirmations: `/tmp/ggtensile-q4-k-geometry-{64x128,128x64}-matrix-a/`.
- Corrected and rejected focused `256x64` matrix: `/tmp/ggtensile-q4-k-geometry-256x64-focused-fixed-a/`; exact `256x128` and SIA3/PLR2 checks are under `/tmp/ggtensile-q4-k-{geometry-256x128,sia3-plr2}-fixed-check-a/`.
- Rejected shared-down high-M ownership matrices: `/tmp/ggtensile-q4-k-geometry-{256x64,256x128}-shared-down-a/`.
- Rejected narrow/attention eight-wave ownership: `/tmp/ggtensile-q4-k-geometry-256x128-narrow-attention-a/`.
- Rejected two-buffer half-tile matrices: `/tmp/ggtensile-q4-k-pipeline-{64x128,128x64}-matrix-a/`; reduced trip checks use `/tmp/ggtensile-pipeline-{64x128,128x64}-debug-k{32,64,96}/`.
- Rejected N64 LDS layouts: `/tmp/ggtensile-q4-k-geometry-128x64-xor{0,4,16}-attention-a/`.
- Rejected N64 traversal: `/tmp/ggtensile-q4-k-geometry-128x64-wgm{2,4,8}-attention-a/`.
- N64 schedule screens and M8192 confirmations: `/tmp/ggtensile-q4-k-geometry-128x64-{sia5,store-priority,sia5-store-priority}-attention-a/`; the direct finalist bracket is `/tmp/ggtensile-q4-k-geometry-128x64-sia5-store-priority-final-a/`.
- Rejected complete two-buffer store-priority matrix and query M2048 confirmation: `/tmp/ggtensile-q4-k-retained-store-priority-matrix-a/`.

The original K8192 baseline was 122.90 ms versus HIP at 47.61 ms. It remains useful as the start of the trajectory, but it is not a current performance control.
