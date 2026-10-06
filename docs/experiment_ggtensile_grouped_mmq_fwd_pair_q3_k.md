# GGTensile Grouped MMQ Forward Pair Q3_K Experiment

## Scope

This record covers the routed gfx1151 paired Q3_K gate and up projections for the Qwen expert bank, computed in one workgroup dataflow:

```text
X_g[M_g,K] @ W_gate_g[K,N] -> Y_gate_g[M_g,N]
X_g[M_g,K] @ W_up_g[K,N]   -> Y_up_g[M_g,N]
```

## Final Results

`TFLOPS = 4*R*N*K / (median_ms * 1e9)` (both projections), and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Each row is the median over the benchmark's sampled expert partitions, and both implementations are timed on the same partitions.

| Batch | Logical shape `(R,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `(16384,512,2048)` | 15.814 | 1.1281x | `grouped_mmq_fwd_pair_q3_k_r16384_n512_k2048_f6a68bb6041cf068` | `grouped_fwd_row_task_q3_k_n512_k2048_j64` |
| 4 | `(65536,512,2048)` | 21.533 | 1.1073x | `grouped_mmq_fwd_pair_q3_k_r65536_n512_k2048_bbb4021253bf5093` | `grouped_fwd_row_task_q3_k_n512_k2048_j64` |
| 16 | `(262144,512,2048)` | 22.350 | 1.0765x | `grouped_mmq_fwd_pair_q3_k_r262144_n512_k2048_7c4a8883db4338d8` | `grouped_fwd_row_task_q3_k_n512_k2048_j64` |

GGTensile is ahead on all 3 rows, with speedups from `1.0765x` to `1.1281x` (mean `1.1039x`).

These medians replace the earlier recorded values, which were paired against HIP bodies that have since been retuned.

## Accepted Experiments

### Selected-half K128 serial and device row-task anchors

Q1 uses one 128-thread wave32 workgroup to own 64 routed rows and 64 output columns in both projections, with two independent 32-VGPR FP32 sum banks and one 9,216-byte activation image. For each K128 half it decodes the selected Q3_K half from the first packed bank into a reusable 10,240-byte LDS weight image, computes the first projection, overwrites only the weight image from the second packed bank, and computes the second projection while activation remains resident. Two producer lanes own each output row and reconstruct the low two-bit payload with the matching high-mask bytes and signed six-bit group scales.

Q2 preserves the arithmetic, register plan, and LDS plan but consumes device-built 64-row tasks through the existing row-task ABI. Both anchors derive 148 VGPRs, 44 SGPRs, and 19,456 bytes of fixed LDS with 128 static integer WMMAs and eight barriers. The first control probe used 29,952 dynamic LDS bytes instead of the installed J64 allocation of 30,976 bytes and differed in 356 BF16 values per projection. Correcting only the control launch allocation restored bitwise agreement.

The R35 matrix matched the installed controls and the fused paired control bit-for-bit for both destinations on first, odd, even, last, repeated, sparse, skewed, boundary, and expert-255 routes. The production-row runner then confirmed repeated and boundary routes at all three aggregate sizes, and synthetic uniform, skewed, sparse, and boundary controls preserved both correctness and the retained performance direction.

### P2 performance qualification

The B16 nine-repeat screen measured `54.5254 ms` complete versus `58.0474 ms` for the installed control and `58.0001 ms` for the adjacent row-task parent. All five medoids were exact, with weakest ratios `1.0598x` and `1.0597x`. Reversed-order 25-repeat confirmation produced GGTensile/HIP complete-call speedups of `1.1966x`, `1.0898x`, and `1.0630x` at B1/B4/B16, with weakest per-medoid ratios `1.0838x`, `1.0641x`, and `1.0584x`. The synthetic route families also kept the retained direction, with weighted HIP ratios `1.1667x`, `1.0800x`, and `1.0622x`.

### Q3-2 variable-offset BFE scale extraction

Each Q3_K scale field previously used `v_lshrrev_b32` followed by `v_and_b32`. Q3-2 replaces each pair with one variable-offset `v_bfe_u32`, which the configured gfx1151 assembler accepts for the exact operand form. The artifact changed from `3,000` to `2,968` VALU issue instructions while retaining 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, and zero spills.

Initial same-session A/B complete-call gains were `1.82%`/`1.03%`/`0.70%` at B1/B4/B16, below the fixed two-percent gate then in force. Under the stability policy, a 25-repeat confirmation measured `0.95%`/`0.81%`/`0.88%` complete-call gains and `0.91%`/`0.82%`/`1.05%` body gains with every one of the fifteen medoids faster. Conservative 95% log-time intervals excluded parity for twelve complete-call and thirteen body medoids, and no remaining interval favored the parent. Q3-2 is retained as the decode schedule for the device-row-task identity. The typed lowering reproduced the qualified probe instruction-for-instruction and two typed builds were byte-deterministic.

## Rejected Experiments

### Dense Q3_K mechanical pairing and direct-to-LDS

Dense Q3_K mechanical pairing was rejected before implementation: the existing 200-VGPR, 39,936-byte-LDS dense point cannot accommodate a second 64-VGPR accumulator bank. Direct-to-LDS remains closed because the configured gfx1151 assembler rejects both the instruction form and the target capability.

### Q3-1 paired zero-accumulator lifetime

Moving the dedicated `v124:v131` initialization before the K2048 block loop removed 24 statically emitted moves and changed the artifact from `3,000` to `2,976` VALU issues. Outputs were exact and resources were unchanged, but weighted complete-call movement was `+0.01%`, `-0.01%`, and `-0.08%` at B1/B4/B16 and body movement was `-0.28%`, `-0.07%`, and `+0.16%`. Q3-1 is rejected as timing-neutral and is not composed with Q3-2.

### Q3-3 epilogue probes

Three independently materialized epilogue forms were tested against the retained Q3-2 body. Shared column setup removed six VALU issues. The broader materialized-address form removed 25. The exact width-two BF16 RNE form preserved instruction and resource counts while interleaving adjacent chains through proven-dead scratch. All passed the exact route and mutation matrix, but complete-call and body movements were mixed or contradictory, so none is retained.

### Q3-4 processor-mode metadata

Applying the processor-mode spelling to the retained Q3-2 source at all row counts produced byte-identical objects and linked HSACOs, so the metadata is executable-inert under this toolchain and no timing or typed identity follows. The existing `s_clause 7`, waits, and barriers remain controls. No wait or barrier is removed without an LDS-overwrite hazard proof.

## Open Items

A B1 requalification of the Q3-2 transfer decision and a body-only recheck of the Q3-3 closure are the only conditionally reopened items. Q3-1 and Q3-4 are not reopened.

## Closure

The retained identity is `q3_k_k128_interleaved_row_tasks_variable_bfe()`: 64-row device tasks with 64 output columns, two accumulator banks, selected-half K128 decode, and variable-offset BFE scales.
