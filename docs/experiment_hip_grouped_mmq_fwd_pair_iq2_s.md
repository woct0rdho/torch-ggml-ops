# HIP Grouped MMQ Forward Pair IQ2_S Experiment

## Scope

This record covers the fused routed IQ2_S gate/up forward kernel for Qwen on gfx1151.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | --- | --- | --- | --- |
| 1 | `2 x (16384,512,2048)` | 12.46 | 1.936x | `grouped_fwd_row_task_iq2_s_n512_k2048_j64` |
| 4 | `2 x (65536,512,2048)` | 17.59 | 1.546x | `grouped_fwd_row_task_iq2_s_n512_k2048_j64` |
| 16 | `2 x (262144,512,2048)` | 18.40 | 0.975x | `grouped_fwd_row_task_iq2_s_n512_k2048_j64` |

The row-task body trails its uniform-route control by `38%` at B1 and `13%` at B4, and is level at B16. The B16 deficit is a packed IQ2_S decode/representation cost against predecoded BF16 weights. Flagged prior-sensitive at B1/B4: the row-task ownership choice may need a learned-route retune against the serial body.

The retained choice was measured on other route distributions, so a learned-route candidate sweep is the follow-up for the flagged batches.

## Optimization log

### B1 ownership retune

The learned-route corpus exposed tall individual experts even when the aggregate average was below the original task threshold. The existing row-task pair body was tested as an ownership-only change. Prior, captured, and synthetic gains were `1.1122x`, `1.0213x`, and `1.0024x`. Reversed-order 25-repeat confirmation measured `1.1177x`, `1.0224x`, and `1.0038x`. The weakest captured and synthetic profiles were `0.9964x` and `0.9973x`, so the exact B1 row-task candidate passed its kernel-level retention controls.

### Decoder width and pair layout

Width-8 duplicated scale work and loader groups and was rejected. IQ2_S pair retained its four-BF16 XOR layout. The single-down path prefers a different layout, and the pair/down swizzle experiment showed opposite timing directions.

The coefficient-only full campaign found no alternate J geometry that passed the exact-key timing and control gates. The M128 body is already near the register warning range, so wider ownership requires a lower-state decoder rather than another tile sweep.

### Sign-mask table

The loader rebuilt both per-value sign masks for every sign byte with a shift/or chain followed by two `__vcmpne4` emulations. Both masks are a pure function of that byte, so they now come from a `256`-entry table of `int2` masks: one eight-byte load replaces roughly ten vector ALU operations per value group.

The change was measured paired in one process against the previous build: `2.1%` at B1, `3.4%` at B4 and `4.6%` at B16 on the pair body, and `4.7%` at B1 on the single-projection body that shares the loader. Under the benchmark protocol the paired batches gain `1.9%` at B4 and B16 (the B1 difference is inside the run-to-run spread), which moves B16 from `0.960x` to `0.975x` against AITER GMM. The same table now serves every IQ2_S forward control.

### Ownership and prefetch checks

Both existing ownership variants were measured against the learned route: device row tasks win at every batch (`0.898x`, `0.812x`, `0.910x` of the serial body's time at B1/B4/B16), so the deployed choice is confirmed rather than prior-sensitive. Staging the second activation plane in registers (`MMQ_PREFETCH_ACT`) is neutral here (`1.000x` to `1.001x`), unlike the single-projection bodies: with K2048 the decode dominates and the pair body has no load latency to fill.

### Remaining bottleneck

The paired path shares activation preparation but each projection still performs its own IQ2_S lookup/sign decode, LDS staging, WMMA, and epilogue. A prepared activation can remove repeated producer work, but only a prepared weight representation or decode reuse can remove the dominant repeated packed work.

Pair and single-down swizzle controls moved in opposite directions: the pair-preferred layout improved paired points by roughly `15-25%`, while applying that layout to single down regressed by roughly `20-35%`. This is why the two bodies keep distinct LDS arrangements. The pair's row-task retune remained an ownership-only experiment. Its weakest captured and synthetic profiles were `0.9964x` and `0.9973x`, above the retention floor.

Still open for this pair: the grid table lookup is now the largest single decode item (`10.3%` of the kernel when ablated, against `15.1%` for the sign chain that the table removed), and the two projections still stage the activation tile separately, so a fused body that keeps one activation tile in LDS across both projections is the remaining structural idea.

## Evidence

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/grouped_fwd_current/pass12_pair_qwen_learned.json
```

The original campaign evidence is:

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/hip-b1-rowtask-screen-9.json
~/tmp/torch-ggml-ops/hip-b1-rowtask-confirm-25.json
~/tmp/torch-ggml-ops/hip-b1-rowtask-retuned/qwen_learned_forward.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
```

IQ2_S pair kernel work is closed at the current packed representation. The next useful mechanism is lower-state IQ2_S decode or explicit model-owned weight reuse.
