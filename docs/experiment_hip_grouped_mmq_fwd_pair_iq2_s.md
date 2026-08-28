# HIP Grouped MMQ Forward Pair IQ2_S Experiment

## Scope

This record covers the fused routed IQ2_S gate/up forward kernel for Qwen on gfx1151. Both projections have `(N,K)=(512,2048)` and share routed activation rows:

```text
Y0[R,512] = X[R,2048] @ W0[512,2048].T
Y1[R,512] = X[R,2048] @ W1[512,2048].T
```

The kernel decodes packed IQ2_S codebook, sign, scale, and `d` state cooperatively. Inputs and outputs are BF16, and the packed banks remain authoritative.

## Final kernel result

Pair throughput counts both projections as `4*R*N*K/time`. `HIP/AITER GMM` compares the packed pair with two BF16 AITER GMM calls. Values above `1.00x` favor HIP.

| Batch | Logical shape | HIP TFLOPS | HIP/AITER GMM |
| ---: | --- | ---: | ---: |
| 1 | `2 x (16384,512,2048)` | 15.66 | 1.937x |
| 4 | `2 x (65536,512,2048)` | 16.20 | 1.283x |
| 16 | `2 x (262144,512,2048)` | 16.65 | 0.913x |

The B16 deficit is a packed IQ2_S decode/representation cost relative to predecoded BF16 AITER weights.

## Kernel implementation

The pair body uses four wave32 waves, exact N512/K2048 geometry, cooperative width-16 IQ2_S decode, separate projection weight staging, and one shared activation workspace. Grid lookup, sign reconstruction, shared scale extraction, and `d` application are performed in the packed kernel before WMMA.

Pair and single-down IQ2_S use distinct LDS layouts because their consumers and decode reuse differ. The fused pair preserves two independent outputs and the pair accumulation/rounding contract.

## Optimization log

### B1 ownership retune

The learned-route corpus exposed tall individual experts even when the aggregate average was below the original task threshold. The existing row-task pair body was tested as an ownership-only change. Prior, captured, and synthetic gains were `1.1122x`, `1.0213x`, and `1.0024x`; reversed-order 25-repeat confirmation measured `1.1177x`, `1.0224x`, and `1.0038x`. The weakest captured and synthetic profiles were `0.9964x` and `0.9973x`, so the exact B1 row-task candidate passed its kernel-level retention controls.

### Decoder width and pair layout

Width-16 decode shares codebook/grid/sign/scale work across natural metadata boundaries. Width-8 duplicated scale work and loader groups and was rejected. IQ2_S pair retained its four-BF16 XOR layout; the single-down path prefers a different layout, and the pair/down swizzle experiment showed opposite timing directions.

The coefficient-only full campaign found no alternate J geometry that passed the exact-key timing and control gates. The M128 body is already near the register warning range, so wider ownership requires a lower-state decoder rather than another tile sweep.

### Remaining bottleneck

The paired path shares activation preparation but each projection still performs its own IQ2_S lookup/sign decode, LDS staging, WMMA, and epilogue. A prepared activation can remove repeated producer work, but only a prepared weight representation or decode reuse can remove the dominant repeated packed work.

Pair and single-down swizzle controls moved in opposite directions: the pair-preferred layout improved paired points by roughly `15-25%`, while applying that layout to single down regressed by roughly `20-35%`. This is why the two bodies keep distinct LDS arrangements. The pair's row-task retune remained an ownership-only experiment; its weakest captured and synthetic profiles were `0.9964x` and `0.9973x`, above the retention floor.

## Correctness and resources

Validation covers codebook entries, sign and scale fields, packed-bank isolation, paired-output independence, input mutation, inactive experts, malformed offsets, non-aligned tails, finite BF16 outputs, and independent dequantized references. Retained artifacts are wave32, zero-private, zero-spill, scratch-free, call-free, and stack-free.

## Evidence

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/hip-b1-rowtask-screen-9.json
~/tmp/torch-ggml-ops/hip-b1-rowtask-confirm-25.json
~/tmp/torch-ggml-ops/hip-b1-rowtask-retuned/qwen_learned_forward.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
```

IQ2_S pair kernel work is closed at the current packed representation. The next useful mechanism is lower-state IQ2_S decode or explicit model-owned weight reuse.
