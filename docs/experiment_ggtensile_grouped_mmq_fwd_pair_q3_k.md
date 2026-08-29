# GGTensile Grouped MMQ Forward Pair Q3_K Experiment

## Scope And Contract

This record covers the isolated gfx1151 paired Q3_K forward kernel for the Qwen routed gate and up projections. For each routed expert the kernel computes both projections in one workgroup dataflow:

```text
X_g[M_g,2048] @ W_gate_g[2048,512] -> Y_gate_g[M_g,512]
X_g[M_g,2048] @ W_up_g[2048,512]   -> Y_up_g[M_g,512]
```

The measured aggregate-row shapes are `R=16384`, `65536`, and `262144`. Each packed Q3_K bank has shape `[256,512,880]`: every 256-value block occupies 110 bytes, each packed row contains eight blocks, and each expert occupies 450,560 bytes. One fixed Q8_1 `F32_D4` activation workspace with shape `[16,R,144]` and two independent BF16 destinations are used. Q3_K reconstruction owns the 110-byte block layout, signed 3-bit payload, unsigned six-bit scales, and FP16 block factor.

The paired research ABI carries two packed weights, one activation workspace, two destinations, the device-resident route description, and the shape values. Logical paired throughput is `4 * R * 512 * 2048 / (latency_ms * 1e9)`, and the speedup ratio is HIP time divided by GGTensile time.

## Final Benchmark Results

The table shows the fastest qualified kernel found for each measured aggregate-row shape.

| Matrix shape `(R,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(16384,512,2048)` | `ggpair_0aed3be2b92c0f07` | `15.525` | `1.2120x` |
| `(65536,512,2048)` | `ggpair_5a0b32e2a6c0ec93` | `20.987` | `1.1120x` |
| `(262144,512,2048)` | `ggpair_6d9aa5f305a16663` | `22.874` | `1.0833x` |

The retained identity is `q3_k_k128_interleaved_row_tasks_variable_bfe()`. All route, malformed-route, mutation, and finite-output checks were exact.

## Final Kernel Profiles

| Kernel hash | Geometry and ownership | Decode and schedule | VGPR / SGPR | LDS bytes | WMMAs / barriers |
| --- | --- | --- | ---: | ---: | ---: |
| `ggpair_0aed3be2b92c0f07` | 64-row task, 64-column, two accumulator banks | selected-half K128, variable-offset BFE scales | `148 / 44` | `19,456` | `128 / 8` |
| `ggpair_5a0b32e2a6c0ec93` | 64-row task, 64-column, two accumulator banks | selected-half K128, variable-offset BFE scales | `148 / 44` | `19,456` | `128 / 8` |
| `ggpair_6d9aa5f305a16663` | 64-row task, 64-column, two accumulator banks | selected-half K128, variable-offset BFE scales | `148 / 44` | `19,456` | `128 / 8` |

The final artifacts use 2,968 VALU issue instructions, 128 static WMMAs, and eight barriers. They are gfx1151 code-object-v5 wave32 kernels with zero private storage, spills, scratch, calls, and dynamic stack. The row-task ABI is 96 bytes and the serial-route ABI is 80 bytes.

## Accepted Kernel Experiments

### Selected-half K128 serial and device row-task anchors

Q1 uses one 128-thread wave32 workgroup to own 64 routed rows and 64 output columns in both projections, with two independent 32-VGPR FP32 sum banks and one 9,216-byte activation image. For each K128 half it decodes the selected Q3_K half from the first packed bank into a reusable 10,240-byte LDS weight image, computes the first projection, overwrites only the weight image from the second packed bank, and computes the second projection while activation remains resident. Two producer lanes own each output row and reconstruct the low two-bit payload with the matching high-mask bytes and signed six-bit group scales.

Q2 preserves the arithmetic, register plan, and LDS plan but consumes device-built 64-row tasks through the existing row-task ABI. Both anchors derive 148 VGPRs, 44 SGPRs, and 19,456 bytes of fixed LDS with 128 static integer WMMAs and eight barriers. The first control probe used 29,952 dynamic LDS bytes instead of the installed J64 allocation of 30,976 bytes and differed in 356 BF16 values per projection; correcting only the control launch allocation restored bitwise agreement.

The R35 matrix matched the installed controls and the fused paired control bit-for-bit for both destinations on first, odd, even, last, repeated, sparse, skewed, boundary, and expert-255 routes. The production-row runner then confirmed repeated and boundary routes at all three aggregate sizes, and synthetic uniform, skewed, sparse, and boundary controls preserved both correctness and the retained performance direction.

### P2 performance qualification

The B16 nine-repeat screen measured `54.5254 ms` complete versus `58.0474 ms` for the installed control and `58.0001 ms` for the adjacent row-task parent; all five medoids were exact, with weakest ratios `1.0598x` and `1.0597x`. Reversed-order 25-repeat confirmation produced GGTensile/HIP complete-call speedups of `1.1966x`, `1.0898x`, and `1.0630x` at B1/B4/B16, with weakest per-medoid ratios `1.0838x`, `1.0641x`, and `1.0584x`. The synthetic route families also kept the retained direction, with weighted HIP ratios `1.1667x`, `1.0800x`, and `1.0622x`.

### Q3-2 variable-offset BFE scale extraction

Each Q3_K scale field previously used `v_lshrrev_b32` followed by `v_and_b32`. Q3-2 replaces each pair with one variable-offset `v_bfe_u32`, which the configured gfx1151 assembler accepts for the exact operand form. The artifact changed from `3,000` to `2,968` VALU issue instructions while retaining 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, and zero spills.

Initial same-session A/B complete-call gains were `1.82%`/`1.03%`/`0.70%` at B1/B4/B16, below the fixed two-percent gate then in force. Under the stability policy, a 25-repeat confirmation measured `0.95%`/`0.81%`/`0.88%` complete-call gains and `0.91%`/`0.82%`/`1.05%` body gains with every one of the fifteen medoids faster; conservative 95% log-time intervals excluded parity for twelve complete-call and thirteen body medoids, and no remaining interval favored the parent. Q3-2 is retained as the decode schedule for the device-row-task identity. The typed lowering reproduced the qualified probe instruction-for-instruction and two typed builds were byte-deterministic.

## Rejected Kernel Experiments

### Dense Q3_K mechanical pairing and direct-to-LDS

Dense Q3_K mechanical pairing was rejected before implementation: the existing 200-VGPR, 39,936-byte-LDS dense point cannot accommodate a second 64-VGPR accumulator bank. Direct-to-LDS remains closed because the configured gfx1151 assembler rejects both the instruction form and the target capability.

### Q3-1 paired zero-accumulator lifetime

Moving the dedicated `v124:v131` initialization before the K2048 block loop removed 24 statically emitted moves and changed the artifact from `3,000` to `2,976` VALU issues. Outputs were exact and resources were unchanged, but weighted complete-call movement was `+0.01%`, `-0.01%`, and `-0.08%` at B1/B4/B16 and body movement was `-0.28%`, `-0.07%`, and `+0.16%`. Q3-1 is rejected as timing-neutral and is not composed with Q3-2.

### Q3-3 epilogue probes

Three independently materialized epilogue forms were tested against the retained Q3-2 body. Shared column setup removed six VALU issues; the broader materialized-address form removed 25; the exact width-two BF16 RNE form preserved instruction and resource counts while interleaving adjacent chains through proven-dead scratch. All passed the exact route and mutation matrix, but complete-call and body movements were mixed or contradictory, so none is retained.

### Q3-4 processor-mode metadata

Applying the processor-mode spelling to the retained Q3-2 source at all row counts produced byte-identical objects and linked HSACOs, so the metadata is executable-inert under this toolchain and no timing or typed identity follows. The existing `s_clause 7`, waits, and barriers remain controls; no wait or barrier is removed without an LDS-overwrite hazard proof.

## Remaining Work

The current direct-kernel benchmark measured B1 at `1.1546x` GGTensile/HIP (75-repeat interval `[1.1444x, 1.1649x]`) versus the documented `1.2120x`, so B1 may need focused retuning. B4 and B16 remain close to their documented rows at `1.1203x` and `1.0839x`. A B1 requalification of the Q3-2 transfer decision and a body-only recheck of the Q3-3 closure are the only conditionally reopened items; Q3-1 and Q3-4 are not reopened.

## Qualification Summary

The retained kernels passed exact packed-HIP comparison, independent dequantized references, finite-output and full-row coverage checks, deterministic reruns, first and non-first routes, sparse and repeated experts, both-gradient and both-active-bank mutations, inactive-expert inertness, malformed-route sentinels, and non-aligned route tails. Independent reference normalized RMSE measured `0.00586` through `0.00624` with maximum absolute error `0.009765625`. Two independent generation, build, and inspection roots reproduce the selected artifacts byte-for-byte with the profile above.
