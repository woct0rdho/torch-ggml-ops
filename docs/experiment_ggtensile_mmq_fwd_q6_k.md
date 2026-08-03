# GGTensile Dense MMQ Forward Q6_K Plan

## Purpose

Build and exhaust a strict gfx1151 wave32 GGTensile assembly campaign for dense Q6_K forward over the three exact production language-model-head chunks. The first objective is to make every exact complete call at least as fast as the installed HIP kernel. After exact-key parity, optimize the complete 2,048-row forward mix as far as repeatable evidence permits without regressing any production key.

Q6_K is a separate forward backend from Q4_K and Q5_K. It may reuse proven launch, WMMA, synchronization, output, inspection, correctness, and measurement mechanisms only when their ownership and physical layouts remain valid. It must own its 210-byte packed block, split low/high payload decode, signed scale fields, block multiplier, `Q8_1` `F32_D4` activation metadata, small-M geometry, and exact tuning identities.

Public dispatch, generated bundle tables, source packaging, and producer fusion remain outside the research phase.

## Contract

Target only:
- gfx1151, wave32, WMMA V1, BF16 input/output, and FP32 correction/accumulation.
- Authoritative packed GGUF Q6_K weights with direct in-kernel decode.
- The installed HIP Q8_1 `F32_D4` activation producer and exact 144-byte workspace block.
- Exact 40-byte dense-forward kernarg ABI.
- One exact `ProblemType`, `ProblemSize`, and complete `Solution` per artifact.
- Exact-key launch geometry with strict rejection for every mismatch.
- Zero private bytes, spills, scratch instructions, calls, and dynamic stack.
- Serial warmed rotating-control timing. Builds and independent correctness work may run separately; timed GPU work never overlaps.
- No prepared weights, dense shadows, external decode workspaces, split-K, Stream-K, persistent or grouped workgroups, online tuning, public dispatch changes, or producer fusion.

Forward coordinates are:

```text
M = token chunk rows
N = vocabulary rows
K = hidden width

output[M,N] = input[M,K] @ dequant_q6_k(weight[N,K]).T
```

A Q6_K block contains 256 logical values in 210 bytes:

```text
ql[128]      low four bits
qh[64]       high two bits
scales[16]   signed int8 scale per 16 values
d             FP16 block multiplier
```

For logical value `i`:

```text
q = low4(i) | (high2(i) << 4)
value = fp16(d) * float(int8(scale[i/16])) * float(q - 32)
```

The installed forward arithmetic performs integer WMMA, multiplies the integer accumulator by the signed Q6 scale, converts through the existing expression order, and then applies the Q6 block factor and Q8_1 activation scale. The strict campaign preserves that arithmetic and BF16 rounding order. Effective-scale or other reordered floating-point experiments require an explicit changed-correctness phase and are not eligible for exact HIP selection by default.

## Exact Production Scope

The packed Q6_K language-model head has logical weight shape `[248320,2048]` and packed shape `[248320,1680]`. A 2,048-row loss traversal uses:

| Exact key | Calls per 2,048 rows | Installed HIP ownership |
| --- | ---: | --- |
| `(64,248320,2048)` | 32 | exact full J64 |
| `(128,248320,2048)` | 16 | exact full J128 |
| `(256,248320,2048)` | 8 | two exact full J128 row tiles |

M256 is the first absolute-time priority because it is the selected complete-loss chunk. M128 and M64 remain required exact fallback keys and may need distinct geometry or schedules. Weighted totals use the call counts above for prioritization only; every exact complete call must independently clear HIP in both promotion confirmations.

## Correctness and Reproducibility Gates

Before timing, every candidate must pass:
- finite output.
- candidate/HIP normalized RMSE `<= 5e-4`.
- candidate/HIP maximum absolute error `<= 0.015625`.
- independent dequantized-reference normalized RMSE `<= 0.04`.
- input mutation sensitivity.
- packed-weight mutation sensitivity.
- Q8_1 workspace mutation sensitivity.
- exact packed byte count and `F32_D4` workspace shape.
- static inspection proving the 40-byte ABI, gfx1151 wave32 metadata, expected WMMA count, and zero private storage, spills, scratch, calls, or dynamic stack.
- byte-identical independent rebuilds of every selected artifact.

Every executable assembly-writer branch and every emitted line must have unit coverage. Q4_K and Q5_K regression paths remain mandatory while Q6_K branches are added.

## Measurement and Promotion

- Nine-repeat screens prune candidates only.
- Promotion requires two independent warmed rotating 25-repeat confirmations against both HIP and the retained parent.
- Measure fixed-quantizer plus multiply complete calls and prequantized multiply bodies separately.
- Every exact complete call must be no slower than HIP in both confirmations.
- The final result table must report each exact shape's prequantized HIP and GGTensile multiply TFLOPS plus multiplicative speedup `HIP median time / GGTensile median time`; quantization plus multiply is diagnostic only.
- Resource-bearing mechanisms require a stable gain above 2%.
- Resource-neutral instruction reductions or schedules may be retained when neutral or consistently favorable, but may not materially regress another selected key.
- Preserve timing, correctness, inspection, generated source, and code-object artifacts under `~/tmp/torch-ggml-ops/ggtensile-fwd-q6-k/`.

## Initial Evidence

A fresh warmed 25-repeat public HIP baseline for `output.weight` produced:

| M | Packed complete call | BF16 control | Packed/BF16 |
| ---: | ---: | ---: | ---: |
| 64 | `4.050 ms` | `8.708 ms` | `0.465x` |
| 128 | `7.626 ms` | `12.606 ms` | `0.605x` |
| 256 | `15.203 ms` | `13.145 ms` | `1.157x` |

Independent-reference NRMSE was `0.006315`, `0.006214`, and `0.005784` for M64, M128, and M256. M64/M128 strongly beat BF16, while M256 remains slower than isolated persistent BF16 GEMM but is still the production complete-loss chunk because end-to-end selection also includes in-place cross-entropy and packed backward.

The project-owned HIP exact bodies use 158 VGPRs and 28,928-byte LDS for J64, and 210 VGPRs and 38,400-byte LDS for J128, with no private storage. Historical normalized ISA shows the J128 Q6 body carries materially more instructions than Q8 and repeatedly multiplies WMMA integer accumulators by signed per-16-value scales. This makes decode ownership, signed-scale application, static issue balance, and small-M geometry the first-order areas.

The completed Q6_K backward campaign provides decoder facts but not a transferable forward solution. Its packed VOPD extraction, scale ownership, and lane-sharing results use transposed output-row ownership and different LDS geometry. Each mechanism must be re-derived under forward J64/J128 ownership before use.

## Optimization Phases

### Phase 0: Strict Q6_K forward infrastructure

- Add Q6_K dense-forward `ProblemType`, exact three-key inventory, versionless catalog, strict validation, runtime packed-size checks, and launch support.
- Add a sibling fixed HIP Q8_1 `F32_D4` producer launcher rather than overloading the incompatible `F16_D4S4` launcher.
- Add exact installed HIP Q6_K J64/J128 multiply launchers.
- Extend correctness and independent-reference handling to the Q6_K block and `F32_D4` workspace.
- Add line-complete tests for model identity, strict rejection, launch geometry, producer layout, packed size, writer branches, and Q4_K/Q5_K regressions.
- Capture fresh prequantized HIP body and complete-call baselines for all three keys.

### Phase 1: Correct decoded-staged J128 control

Start with M128 and M256 using the proven `128x64`, 128-thread, four-wave ownership only where Q6 physical LDS rows can be represented without repair.
- Load `ql`, `qh`, signed scales, and `d` directly from 210-byte blocks.
- Reconstruct exact six-bit signed values in the LDS order consumed by WMMA.
- Preserve integer-accumulator-times-int8-scale arithmetic before block and activation factors.
- Use Q8_1 `F32_D4` metadata directly; do not emit Q4/Q5 sum correction paths.
- Release decoder temporaries before accumulation where possible.
- Compare a serialized correctness control with a Q6-specific decoded-staged schedule derived from the installed HIP source and ISA.

No timing interpretation is valid until the decoder is bit-exact to HIP, independently correct, mutation-sensitive, and resource-clean.

### Phase 2: Exact J64 small-M control

M64 cannot be represented by the J128 launch without padded-row work or an unsupported partial tile. Add an exact `64x64` output ownership matching the installed J64 premise:
- 128 threads, four wave32 waves unless inspection proves a lower-thread ownership is required.
- 28,928-byte or smaller LDS target.
- exact 64-row launch with no padded output row accepted as valid.
- Q6-specific decoded-weight row ownership and Q8_1 `F32_D4` activation staging.
- strict comparison against both installed J64 HIP and a diagnostic padded-J128 control if one is safe.

Do not generalize J64 to M128/M256 unless it wins those exact complete calls in confirmation.

### Phase 3: Large-margin decoder and scale work

Prioritize M256 absolute time, then transfer only measured mechanisms:
- packed low/high address contraction and immediate offsets.
- vector low/high extraction schedules and in-place high-plane shifts.
- signed-scale load width, lane ownership, and broadcast.
- legal gfx1151 VOPD pairs for adjacent subtracts or scale multiplies.
- decode preparation batching by row or 16-value scale group.
- early temporary release and accumulator initialization.
- WMMA/local-read issue distance and exact epilogue scheduling.
- bounded instruction priority only when tied to a measured dependency phase.

Lane sharing must reduce enough global work to repay EXEC or cross-lane distribution. Backward evidence alone is not sufficient.

### Phase 4: Changed-premise geometry and overlap

Only after lower bounds or counters show a first-order opportunity, test:
- reduced decoded-weight LDS for J64/J128.
- one bounded next-block packed prefetch that removes current decode issue rather than merely moving VMEM.
- a second decoded-weight plane only if it remains below the register/LDS resource gate.
- alternate DepthU only as a bundled decode/local-read schedule.
- WGM only where M256 has multiple exact row tiles and launch order can matter.

Do not repeat Q4/Q5 activation double buffering, payload-only prefetch, broad geometry sweeps, or rolled-loop forms without a Q6-specific changed premise.

### Phase 5: Exact selection and weighted complete-call closure

- Select one explicit identity per exact M.
- Confirm each exact complete call twice against installed HIP and its retained parent.
- Rebuild selected artifacts independently and compare byte-for-byte.
- Publish the final per-shape result using prequantized multiply throughput and multiplicative speedup rather than complete-call throughput.
- Compute the 32/16/8-call weighted 2,048-row forward total for prioritization, while retaining the exact-key HIP gate.
- Keep public dispatch deferred.

## Recursive Optimization-Exhaustion Review

Before declaring Q6_K complete, reread this plan; the installed Q6_K forward source and normalized gfx1151 ISA; every Q6 forward artifact, timing, lower bound, profile, counter, selected and rejected candidate; the completed Q4_K/Q5_K forward records; the Q6_K backward record; the local HIP optimization history; relevant CK, TensileLite, EvoTensile, hipBLASLt, RDNA3.5 ISA, and AMD LLVM gfx11 scheduling/VOPD evidence.

Classify every remaining idea as:
- retained and measured.
- rejected by correctness, resources, timing, or reproducibility.
- contract-incompatible or deferred with an explicit prerequisite.
- actionable with an exact target and measurement gate.

Every actionable idea must be implemented and measured, then the review repeats from the changed premise. Completion requires no remaining actionable in-contract mechanism, two strict complete-call HIP wins for all three exact keys, quantified residual bottlenecks, line-complete writer tests, and byte-identical retained rebuilds.

## Campaign Record

### Plan opened and fresh baseline

- Defined the exact M64/M128/M256 language-model-head inventory and 32/16/8 call weighting.
- Fixed direct packed Q6_K decode, Q8_1 `F32_D4`, exact 210-byte block, 40-byte ABI, and zero-private-resource contracts.
- Recorded fresh public HIP complete-call and BF16 baselines for all three keys.
- Identified M256 as the first absolute-time target and exact J64 support as a required independent M64 path.
- Next: implement strict Q6_K forward modeling, sibling `F32_D4` producer launch, installed J64/J128 HIP controls, exact catalogs, and correctness-first decoded-staged controls.
