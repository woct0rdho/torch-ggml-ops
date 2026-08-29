# GGTensile Fixed-Group MMQ Forward Q8_0 Experiment

## Scope And Contract

This record covers the fixed-group Q8_0 forward kernel for gfx1151. The operation has eight independent packed weight groups, one Q8_1 F32_D4 activation workspace, and BF16 output. One workgroup owns an output-feature tile, a token tile, and one fixed group.

The measured contract is:
- gfx1151, code-object version 5, wave32, WMMA V1;
- BF16 input with logical shape `[tokens, 8, 4096]`;
- Q8_1 F32_D4 activation workspace with one quantized row per token and group;
- eight packed Q8_0 weight groups with row stride 4,352 bytes;
- BF16 output with logical shape `[tokens, 8, N]`;
- `tokens` equal to 2,048, 8,192, or 32,768; and
- the evaluated output-feature and reduction dimensions `N=1024` and `K=4096`.

The six-argument kernel ABI is:

```text
packed_weight, activations, output, tokens, out_features, bytes_per_group
```

The launch uses `(ceil(N/64), ceil(tokens/64), 8)` workgroups of `(32,4,1)` workitems. Q8_0 stores one FP16 scale and 32 signed int8 values in 34 bytes. Each output uses the established signed integer WMMA correction:

```text
fp32_output += int32_dot * fp32(q8_0_scale) * fp32(q8_1_activation_scale)
```

The Q8_1 producer is fixed infrastructure and is outside multiply-only timing. The speed table therefore compares the GGTensile multiply body with the HIP fixed-group multiply body on the same packed weights and prequantized workspace.

## Final Benchmark Results

These are the fastest qualified kernels found across the complete fixed-group experiment. Speed is prequantized multiply-only TFLOPS across all eight groups. The speedup is HIP median time divided by GGTensile median time; values above `1.0x` favor GGTensile.

| Matrix shape `(M,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(2048,1024,4096)` | `ggsol_65ea830f809a878c` | `35.771` | `2.6401x` |
| `(8192,1024,4096)` | `ggsol_03c6517b032619d5` | `32.947` | `2.4001x` |
| `(32768,1024,4096)` | `ggsol_e80f2d214da77367` | `32.231` | `2.3379x` |

The two independent confirmation runs preserved the same result direction at all three token counts. No alternative identity was faster than the final `ReductionLoopAndWeightStage` kernels.

## Final Kernel Profiles

| Kernel hash | Lowering identity | Workgroup | Macro tile | DepthU | VGPR / SGPR | LDS bytes | WMMAs | Barriers |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ggsol_65ea830f809a878c` | `ReductionLoopAndWeightStage` | `32x4x1` | `64x64` | 32 | `144 / 16` | `18,432` | 32 | 2 |
| `ggsol_03c6517b032619d5` | `ReductionLoopAndWeightStage` | `32x4x1` | `64x64` | 32 | `144 / 16` | `18,432` | 32 | 2 |
| `ggsol_e80f2d214da77367` | `ReductionLoopAndWeightStage` | `32x4x1` | `64x64` | 32 | `144 / 16` | `18,432` | 32 | 2 |

All final artifacts use the 40-byte ABI, have zero private storage and zero VGPR/SGPR spills, and contain no scratch instructions, calls, or dynamic stack. The final layout uses compact 144-byte LDS rows, paired weight-scale reads, a hoisted activation-plane stride, and fixed-group address ownership.

## Accepted Kernel Experiments

### Correct Small-M baseline

The initial `Q8SmallMTiledLds` kernel established the fixed ABI, group-Z ownership, flattened output addressing, signed-int8 WMMA correction, and exact BF16 stores. Its resource profile was 144 VGPRs, 16 SGPRs, 28,672 LDS bytes, 32 WMMAs, and two barriers.

An early emission was rejected before timing because it used the wrong scalar pointer for output, omitted scalar kernarg loads, and used dense-row rather than eight-group output spacing. Correcting those address-contract defects produced the accepted correctness baseline. The corrected kernel was finite, bitwise equal to the HIP control at all required token counts, mutation-sensitive, and consistent with the independent Q8_0 reference.

### Compact depth-32 LDS ownership

`CompactDepth32WeightRows` changed the fixed `64x64`, `DepthU=32` representation to 144-byte LDS rows and paired weight-scale reads. It reduced LDS from 28,672 to 18,432 bytes without changing the 144-VGPR, 16-SGPR, 32-WMMA, two-barrier resource class. It remained bitwise exact and faster than the small-M baseline at every required token count on the prequantized and complete-call surfaces.

This is the retained data-layout mechanism. The compact row layout is also the basis for all final kernels.

### Reduction-loop address hoisting

`ReductionLoop` hoisted lane- and wave-derived weight-scale LDS addresses out of the reduction loop, retained the paired second-row address in a free rounded VGPR, and moved the activation-plane stride to a dead-after-setup SGPR. It removed repeated address work without changing resources or arithmetic. The candidate was exact and faster than the compact parent in balanced screening at all three token counts.

### Weight-stage address hoisting

`ReductionLoopAndWeightStage` retained the lane-parity packed-weight offset and the payload and scale LDS write addresses across the reduction loop. The weight staging emitter was split into address generation, global loads, and ready-address LDS writes so the fixed lowerer could consume the hoisted values without changing ordinary signed-int8 streams.

The candidate remained at 144 VGPRs, 16 SGPRs, 18,432 LDS bytes, 32 WMMAs, and two barriers. It was bitwise exact at every required token count and passed the active input, packed-weight, activation-workspace, and repeated-producer checks. Two independent seven-warmup, 25-repeat confirmations selected it as the fastest fixed-group kernel at all three matrix shapes.

### Store-clause scheduling

The final kernel retains the 32-store BF16 clause. Removing the clause materially regressed the kernel at every tested token count. Exact BF16 conversion-chain interleaving was tested separately and was not retained, but the original 32-store clause remains an accepted part of the final emitted schedule.

## Rejected Kernel Experiments

### Weight-write wait removal

`Overlap` removed the second packed-weight `vmcnt(0)` because the preceding `vmcnt(6)` appeared to cover the required operations. The artifact remained exact and resource-identical, but candidate/parent body ratios were `1.0008x`, `0.9983x`, and `1.0002x`; complete-call ratios were `0.9954x`, `1.0011x`, and `1.0004x`. The movement was not coherent, so the candidate was rejected and the original waits were restored.

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

The hand-authored final artifact contains no `s_delay_alu` instructions, and no changed producer/consumer schedule created a concrete delay requirement. The target audit also found no remaining standalone X-eligible operation that could form an additional legal VOPD pair; the remaining shifts, adds, conversions, `v_bfe_u32`, `v_add3_u32`, multiplies, and WMMA operations are not eligible for that pairing. No delay or VOPD candidate was emitted.

## Qualification Summary

The final kernels were independently regenerated and linked deterministically. They passed exact HIP comparison at 2,048, 8,192, and 32,768 tokens, finite-output checks, independent dequantized-reference checks, input and active-weight mutation checks, activation-workspace mutation checks, inactive-group checks, and repeated Q8_1 producer checks.

The retained artifacts are gfx1151 code-object-v5 wave32 kernels with the final profile listed above. All protected ordinary writer streams remained byte-identical, and the final fixed-group source and artifacts contain no unqualified kernel identity. The final result is the `ReductionLoopAndWeightStage` compact depth-32 kernel at each measured matrix shape.
