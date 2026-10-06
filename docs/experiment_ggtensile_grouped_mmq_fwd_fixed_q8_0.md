# GGTensile Fixed-Group MMQ Forward Q8_0 Experiment

## Scope

This record covers the fixed-group Q8_0 forward kernel for gfx1151: eight independent packed weight groups over one Q8_1 activation workspace, with BF16 output.

## Final Results

`TFLOPS = 2*M*N*K / (median_ms * 1e9)` across all eight groups, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Medians are the repository benchmark's, re-measured in the current clock state. They replace the earlier recorded values, which came from a different clock state with the same artifacts.

| Family | `(M,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| --- | --- | ---: | ---: | --- | --- |
| Fixed grouped | `(2048,1024,4096)` | 31.298 | 2.2880x | `fixed_grouped_mmq_fwd_q8_0_t2048_n1024_k4096_80d2a010f2ca3571` | `grouped_fwd_fixed_q8_0_g8_k4096_j64_full` |
| Fixed grouped | `(8192,1024,4096)` | 30.189 | 2.2153x | `fixed_grouped_mmq_fwd_q8_0_t8192_n1024_k4096_da8fb61dcac7e099` | `grouped_fwd_fixed_q8_0_g8_k4096_j64_full` |
| Fixed grouped | `(32768,1024,4096)` | 31.120 | 2.2713x | `fixed_grouped_mmq_fwd_q8_0_t32768_n1024_k4096_716bf2d5468c5672` | `grouped_fwd_fixed_q8_0_g8_k4096_j64_full` |

GGTensile is ahead on all three token counts, by `2.215x` to `2.288x`. The HIP side reproduces its own record. The GGTensile side is the noisier arm and moves a few percent between runs, and it is also clock-sensitive: the same artifacts read 3.6-14% higher in the earlier recordings, which were taken in a faster clock state.

## Accepted Kernel Experiments

### Correct Small-M baseline

The initial `Q8SmallMTiledLds` kernel established the fixed ABI, group-Z ownership, flattened output addressing, signed-int8 WMMA correction, and exact BF16 stores. Its resource profile was 144 VGPRs, 16 SGPRs, 28,672 LDS bytes, 32 WMMAs, and two barriers.

An early emission was rejected before timing because it used the wrong scalar pointer for output, omitted scalar kernarg loads, and used dense-row rather than eight-group output spacing. Correcting those address-contract defects produced the accepted baseline.

### Compact depth-32 LDS ownership

`CompactDepth32WeightRows` changed the fixed `64x64`, `DepthU=32` representation to 144-byte LDS rows and paired weight-scale reads. It reduced LDS from 28,672 to 18,432 bytes without changing the 144-VGPR, 16-SGPR, 32-WMMA, two-barrier resource class. It was faster than the small-M baseline at every required token count on the prequantized and complete-call surfaces.

This is the retained data-layout mechanism. The compact row layout is also the basis for all final kernels.

### Reduction-loop address hoisting

`ReductionLoop` hoisted lane- and wave-derived weight-scale LDS addresses out of the reduction loop, retained the paired second-row address in a free rounded VGPR, and moved the activation-plane stride to a dead-after-setup SGPR. It removed repeated address work without changing resources or arithmetic. The candidate was exact and faster than the compact parent in balanced screening at all three token counts.

### Weight-stage address hoisting

`ReductionLoopAndWeightStage` retained the lane-parity packed-weight offset and the payload and scale LDS write addresses across the reduction loop. The weight staging emitter was split into address generation, global loads, and ready-address LDS writes so the fixed lowerer could consume the hoisted values without changing ordinary signed-int8 streams.

The candidate remained at 144 VGPRs, 16 SGPRs, 18,432 LDS bytes, 32 WMMAs, and two barriers. Two independent seven-warmup, 25-repeat confirmations selected it as the fastest fixed-group kernel at all three matrix shapes.

### Store-clause scheduling

The final kernel retains the 32-store BF16 clause. Removing the clause materially regressed the kernel at every tested token count. Exact BF16 conversion-chain interleaving was tested separately and was not retained, but the original 32-store clause remains an accepted part of the final emitted schedule.

## Rejected Kernel Experiments

### Weight-write wait removal

`Overlap` removed the second packed-weight `vmcnt(0)` because the preceding `vmcnt(6)` appeared to cover the required operations. The artifact remained exact and resource-identical, but candidate/parent body ratios were `1.0008x`, `0.9983x`, and `1.0002x`. Complete-call ratios were `0.9954x`, `1.0011x`, and `1.0004x`. The movement was not coherent, so the candidate was rejected and the original waits were restored.

### Persistent weight-stage pointer

`PersistentWeight` carried the global weight-stage pointer through the loop and removed one repeated base-plus-lane add. It was exact and resource-clean, but candidate/parent prequantized ratios were `1.0124x` at 2,048 tokens and `0.9968x` at 8,192 tokens. Complete-call ratios were `0.9982x` and `0.9955x`. The required short shape regressed and the longer-shape movement was sub-percent, so the candidate was rejected.

### Two-iteration loop unrolling

`Unroll2` duplicated the exact `DepthU=32` body and reduced loop-counter updates and branches. The 2,048-token candidate/parent ratios were `1.0028x` for the body and `1.0016x` complete. The larger code body supplied no measurable benefit and the candidate was rejected without extending the test to the longer shapes.

### BF16 conversion-chain interleaving

The exact width-two conversion schedule preserved rounding, address ownership, and resources but moved in different directions between body and complete-call measurements. It had no stable advantage over the final schedule and was rejected.

Store-clause widths 16 and 8 produced occasional sub-percent shape-local signals, but neither reproduced a stable benefit in independent confirmation. The no-clause form was a material regression. No alternate clause identity is retained.

### Processor-mode metadata

Adding `.amdhsa_workgroup_processor_mode 1` produced byte-identical objects and linked code objects under the configured gfx1151 assembler and linker. It therefore created no distinct executable kernel and was rejected as an optimization identity.

### Delay-ALU and additional VOPD pairing

The hand-authored final artifact contains no `s_delay_alu` instructions, and no changed producer/consumer schedule created a concrete delay requirement. The target audit also found no remaining standalone X-eligible operation that could form an additional legal VOPD pair. The remaining shifts, adds, conversions, `v_bfe_u32`, `v_add3_u32`, multiplies, and WMMA operations are not eligible for that pairing. No delay or VOPD candidate was emitted.

## Qualification Summary

The final kernels were independently regenerated and linked deterministically.

The retained artifacts are gfx1151 code-object-v5 wave32 kernels of the `ReductionLoopAndWeightStage` lowering. All protected ordinary writer streams remained byte-identical, and the final fixed-group source and artifacts contain no unqualified kernel identity. The final result is the `ReductionLoopAndWeightStage` compact depth-32 kernel at each measured matrix shape.
