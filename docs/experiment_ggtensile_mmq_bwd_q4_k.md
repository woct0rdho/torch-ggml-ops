# GGTensile Dense MMQ Backward Q4_K Experiment Log

## Scope

This document is the current result summary, next plan, and historical evidence log for the gfx1151 dense MMQ backward Q4_K experiment. The first measured pair is `ProblemSize(M=32768, N=2048, K=8192)` and its production companion `ProblemSize(M=32768, N=2048, K=512)`. All campaign shapes use packed Q4_K weights, BF16 inputs and outputs, FP32 WMMA accumulation, wave32, and the existing 40-byte dense-backward kernarg ABI.

The overall generator architecture, multi-quant roadmap, and deferred integration policy remain in [ggtensile_plan.md](ggtensile_plan.md). Results here are exact-shape evidence and must not be generalized without measurement.

## Campaign Contract And Coverage

GGTensile generates a kernel for one exact `ProblemType` and `ProblemSize`. A generated kernel is required to be correct only for that matrix shape, and runtime dispatch may use it only on an exact key match. This is weaker than the repository's hipBLASLt tuning contract, where a kernel selected while tuning one matrix shape may still run on another shape in its supported assertion domain. GGTensile can therefore use shape constants, exact loop counts, fully peeled tails, fixed launch geometry, and shape-specific ownership or traversal without preserving cross-shape validity. The HIP kernel remains the correctness and compatibility fallback for every unmatched shape.

The dense Q4_K campaign contains only 12 production shapes: the Cartesian product of `M={2048,8192,32768}` and these backward `(N,K)` families:

- `(2048,512)`: narrow K/V/shared gate/up, 70 model calls;
- `(512,2048)`: shared-expert down, 30 model calls;
- `(4096,2048)`: attention output, 10 model calls;
- `(2048,8192)`: attention query/query gate, one model call.

The two `M=32768, N=2048` shapes are measured and selected below. Ten production shapes remain. The next campaign phase is to optimize all 12 manually, supported by small scripts that generate, build, inspect, check, and bracket focused candidate sets. A large-grid EvoTensile campaign is unnecessary for this inventory. Grouped MMQ is explicitly outside this campaign.

## Latest Result

### Selected solution

The selected kernel is the four-wave true decoded-B pipeline:

- `MacroTile=128x128`, `DepthU=32`, and `WorkGroup=[32,4,1]`;
- WGM1 traversal and XOR-8 decoded-B LDS layout;
- SIA4, PGR2, and PLR1;
- `1LDSBuffer=0`, giving two 8 KiB decoded-B buffers;
- no store priority, no packed-next-only prefetch, and no packed-weight lane sharing;
- direct nibble and packed-offset formulas, scalar-base global loads and stores, compact address VGPRs, accumulator-clear VOPD, and no `buffer_gl0_inv`.

The artifact uses 212 VGPRs, 16 SGPRs, and 16 KiB LDS. It has no private storage, spills, scratch instructions, calls, or dynamic stack. Static inspection sees 64 WMMAs, 128 LDS instructions, 164 VMEM instructions, 903 non-WMMA VALU issues representing 923 operations, 20 VOPD pairs, 23 waits, and two barriers. WMMAs and LDS instructions appear in both the steady and peeled-final static paths, so those counts are not per-iteration dynamic counts.

### Performance

The retained 25-repeat rotating brackets measured:

- K8192: 38.012 ms and 28.93 TFLOP/s. The one-buffer assembly control measured 39.467 ms, so the pipeline reduces latency by 3.69%. HIP measured 47.779 ms.
- K512: 2.5063 ms and 27.42 TFLOP/s. The one-buffer control measured 2.6654 ms, so the pipeline reduces latency by 5.97%. HIP measured 3.0246 ms.

The K8192 result is about 48.7% of the approximately 59.4 TFLOP/s gfx1151 BF16 WMMA roof. It remains below the 30 TFLOP/s experiment target, which requires approximately 36.65 ms and therefore another 3.6% reduction from the selected result.

### Correctness

All 67,108,864 outputs match HIP bit-for-bit on both production tensors. The same equality holds after rewriting the complete `grad_output` tensor and after mutating a packed-weight byte between launches, which guards the retained L0-invalidation omission.

Against independently dequantized BF16 matmul:

- K8192 candidate and HIP each differ in 409,880 BF16 elements, with maximum absolute error 0.0625 and normalized RMSE 0.0001924804.
- K512 candidate and HIP each differ in 16,684 BF16 elements, with maximum absolute error 0.00390625 and normalized RMSE 0.0000345316.

Independent source generation is byte-reproducible, and both artifacts pass the hard resource and ABI gates.

## Work Completed

### Measurement and inspection infrastructure

- `tools/benchmark_ggtensile.py` provides warmed rotating HIP, candidate, and assembly-control timing; full-output comparison; independent BF16-reference comparison; and producer-handoff checks.
- Artifact inspection records ABI and hard resources plus static VALU issues and operations, VOPD pairs, VMEM, LDS, waits, barriers, clauses, dependency delays, and L0 invalidations.
- `tools/benchmark_ggtensile_lower_bounds.py` generates, builds, inspects, and rotates exact `wmma_floor` and `decode_floor` diagnostic artifacts while leaving production `Solution` and dispatch contracts unchanged.

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

### True-pipeline validation and profiling

The first K64 pipeline launch exposed a live-range error: decode preparation reused `v205:v208`, which still held LDS read addresses for later WMMA pairs. The first N pair was correct and the remaining pairs were misaddressed. The corrected writer uses temporary addresses for the first pair, then moves steady read addresses into Q4_K metadata VGPRs after that metadata dies. Reduced real-weight tests are bit-exact at K32, K64, K96, and K512, covering prime/final execution, one overlap, repeated swaps, and next-A handoff.

A fair raw-counter comparison of current artifacts showed that the two-buffer pipeline reduces median aggregate wave cycles by approximately 4.9%, wait-count stalls from 23.74% to 21.38%, and barrier stalls from 8.58% to 6.11%. LDS instruction count is unchanged. Additional address and peeled-path VALU is outweighed by overlap.

### Exact lower bounds

The diagnostics preserve the selected ABI, 128-thread workgroup, 212-VGPR declaration, 16 SGPRs, and 16 KiB LDS occupancy. Each generated source is hashed, and each code object passes the normal symbol, metadata, register, private-storage, spill, scratch, stack, and call checks with diagnostic-specific WMMA and barrier expectations.

- K8192: complete pipeline 37.868 ms, WMMA/A/LDS floor 24.149 ms or 45.53 equivalent TFLOP/s, and decode/LDS floor 16.842 ms. The floor sum is 1.0825 times complete latency; complete is 1.568 times the dominant WMMA floor.
- K512: complete pipeline 2.3131 ms, WMMA floor 1.6197 ms or 42.43 equivalent TFLOP/s, and decode floor 1.0915 ms. The floor sum is 1.1721 times complete latency; complete is 1.428 times the dominant floor.

The complete kernel already beats the sum of isolated floors through overlap, but the remaining gap to the WMMA/A/LDS floor confirms that the hardware ceiling is not yet exhausted.

## What Remains To Do

### Revisit policy for the measured pair

There is no currently open sub-percent tuning neighborhood. Clauses, dependency batching, early A0 movement, packed-next-only prefetch, lane sharing, local-read double buffering, store priority, and ordinary geometry changes have measured neutral or negative results. Reopening one requires a materially different resource or scheduling regime.

A future K8192 attempt should begin only with a mechanism plausibly worth more than the approximately 3.6% needed to cross 30 TFLOP/s. Candidate ideas must explain how they improve on the measured two-buffer overlap without repeating the rejected eight-wave ownership cost. Dynamic-loop VOPD would require different instruction selection and register ownership because the current decode chain is dominated by operations that are not directly pairable in VOPD Y slots.

Any new resource-bearing mechanism must remain bit-exact, reproducible, resource-clean, and more than 2% faster on both K8192 and K512. Given the explicit request to prioritize larger margins, a practical threshold for revisiting this case is closer to the remaining 3.6% K8192 gap.

### Twelve-shape dense Q4_K campaign

The next work is to establish controls and optimize the ten unmeasured production shapes. Start by running the selected two-buffer design unchanged across all compatible shapes to expose where M, N, and K alter occupancy, traversal, decode amortization, and epilogue cost. Then tune each exact shape manually or with bounded per-shape scans over implemented solution mechanisms. Reuse a solution only when it wins independently; no generated kernel needs to remain valid for another shape.

Prioritize by complete-model frequency: remaining `(N,K)=(2048,512)` narrow shapes first, then `(512,2048)` shared down, `(4096,2048)` attention output, and finally the remaining `(2048,8192)` query shapes. For each exact shape, retain one HIP control, one current assembly control, candidate measurements, correctness evidence, resource inspection, and rejection reasoning.

Quant-family expansion and public runtime dispatch remain governed by [ggtensile_plan.md](ggtensile_plan.md). The selected assembly artifacts are controls and evidence; public integration remains deferred until all 12 dense Q4_K shapes are covered and complete-workload validation passes.

Prepared weights, BF16 shadows, external decode workspaces, GSU, Stream-K, and persistent workgroups remain outside this experiment contract unless the broader plan explicitly accepts their ownership, workspace, or fixup requirements.

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
- Duff-style exact-trip body unroll factors 2/4/8 and complete K512 factor 15 or K8192 factor 16 were bit-exact and resource-clean, but grew source from about 54 KiB to 69-278 KiB and code objects from 13.8 KiB to 16.4-53.5 KiB. Nine-repeat K512 screens gained at most 0.97%; K8192 screens ranged from neutral to 0.63% slower. None reached the 2% resource-bearing gate, so no factor advanced or remains in the writer.

### Packed-weight sharing and addressing

- `PackedWeightLaneShare=2` halved duplicated packed-Q bytes. The LDS crossbar regressed 1.15%; the DPP/VALU crossbar regressed 1.88%. Cross-lane work cost more than the saved traffic.
- Persistent A-pointer hoisting raised allocation from 200 to 204 VGPRs and regressed 0.23%; the resource-neutral coordinate/byte-offset hoists were retained instead.

### Epilogue and traversal

- Store priority was neutral: no priority measured 45.327 ms versus 45.413 ms. The simpler no-priority path remains selected.
- WGM2, WGM4, and WGM8 screened at 54.97, 56.86, and 57.46 ms after traversal correction; WGM1 won every comparison.

## Debugging Lessons Worth Preserving

- gfx1151 packed work-item X/Y must be flattened before `v0` becomes accumulator storage.
- Assembly and metadata inspection cannot replace execution tests. DU64 initially advanced the reduction counter by 32 instead of `DepthU`, then used hardcoded 64-byte LDS row strides; both faults assembled and passed structural checks before execution exposed them.
- Compact address-register changes require auditing every consumer. An early compact artifact still decoded through old `address+10/+11` indices and failed correctness until updated to `address+6/+7`.
- The true pipeline showed that temporary lifetime is part of correctness: interleaved decode clobbered later-pair LDS addresses even though one-buffer execution was valid.
- The dedicated-wave fault showed that `v_sub_nc_u32` operand order can turn a local-ID correction into a 32-bit global-address wrap. Reduced K32/K64 tests should precede production launches for every new control-flow or induction regime.
- Static counts for peeled or branched kernels include mutually exclusive paths and must not be interpreted as dynamic work without control-flow analysis.
- Broad derived-counter groups may exceed gfx1151 collection capabilities; preserve supported raw groups and compare artifacts under the same collection protocol.

## Evidence Locations

- Initial K8192 baseline: `/tmp/ggtensile-m32768-n2048-k8192-baseline/benchmark.json`.
- Retained K8192 pipeline: `/tmp/ggtensile-m32768-n2048-k8192-true-pipeline/`.
- Retained K512 pipeline: `/tmp/ggtensile-m32768-n2048-k512-true-pipeline/`.
- Current one-buffer and pipeline profiles: `/tmp/ggtensile-profile-current-one-buffer/` and `/tmp/ggtensile-profile-true-pipeline/`.
- K8192 and K512 lower bounds: `/tmp/ggtensile-m32768-n2048-k8192-lower-bounds/` and `/tmp/ggtensile-m32768-n2048-k512-lower-bounds/`.
- Dead-kernarg lowering and serial 25-repeat brackets: `/tmp/ggtensile-m32768-n2048-k8192-dead-kernargs-a/` and `/tmp/ggtensile-m32768-n2048-k512-dead-kernargs-a/`.
- Exact-trip branch lowering and serial 25-repeat brackets: `/tmp/ggtensile-m32768-n2048-k8192-postcheck-loop-a/` and `/tmp/ggtensile-m32768-n2048-k512-postcheck-loop-a/`.
- Rejected body-unroll artifacts use `/tmp/ggtensile-m32768-n2048-k{512,8192}-unroll{factor}-a/`, with factor-specific generate, inspect, correctness, and nine-repeat timing evidence.
- Exact production-N coverage artifacts: `/tmp/ggtensile-m2048-n512-k2048-production-n-a/` and `/tmp/ggtensile-m2048-n4096-k2048-production-n-a/`.

The original K8192 baseline was 122.90 ms versus HIP at 47.61 ms. It remains useful as the start of the trajectory, but it is not a current performance control.
