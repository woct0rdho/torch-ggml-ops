# HIP Grouped MMQ Backward Pair IQ2_S Experiment

## Scope

This record covers the fused routed IQ2_S gate/up input-gradient kernel for Qwen.

Aggregate rows are `R=16384,65536,262144`.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | --- | --- | --- | --- |
| 1 | `2 x (16384,512,2048)` | 11.23 | 1.402x | `grouped_bwd_pair_iq2_s_n512_k2048_mt64_nt64` * |
| 4 | `2 x (65536,512,2048)` | 18.69 | 1.492x | `grouped_bwd_pair_iq2_s_n512_k2048_mt128_nt64` * |
| 16 | `2 x (262144,512,2048)` | 20.59 | 1.319x | `grouped_bwd_pair_iq2_s_n512_k2048_mt128_nt64` * |

Rows marked `*` look prior-sensitive: the current learned-route result is more than 10% below the same body measured on a single uniform partition, so the deployed body may need retuning for the current route distribution.

All three final shapes beat the AITER GMM baseline.

## Kernel implementation

Pair and single-down IQ2_S use separate LDS swizzles. The pair path uses a four-BF16 XOR layout; its decode and epilogue state must not be generalized from the single-down path.

## Optimization log

### Initial pair body

The tiled pair body switched to four-wave N64/K32 ownership, cooperative lookup/sign/scale decode, padded/format-specific LDS staging, and fused accumulation. It won all measured Qwen pair points against AITER.

### Bounded extraction and layout controls

Width-16 decode shares grid, sign, and scale work across natural packed groups. Width-8 duplicated scale work and loader groups and was rejected. Reducing N to N64 relieved pair register pressure and improved representative large-route points. A universal M256/N64 body added state and regressed B4/B16.

Swizzle experiments showed that pair and down consumers prefer different layouts: the pair swizzle improved pair timings by roughly `15-25%`, while the same direction regressed down by `20-35%`. This is a kernel ownership result, not a global IQ2_S policy.

### Ownership retune

The learned B1 screen exposed tall physical experts and compared serial with row-task pair ownership. The row-task body was retained for the pair B1 screen, while the final typed campaign found no alternate J or M geometry with a stable gain across the required profiles. N-major task ordering and broad persistent traversal were rejected.

The remaining theoretical ceiling is IQ2_S decode state and pair accumulator pressure.

### Shared backward arithmetic controls

The grouped backward campaign tested reduced-precision accumulation as a separate kernel mechanism. Direct BF16-C reached `0.86089` NRMSE at 513 rows and was rejected for accuracy. Full-N FP32 K32/K64 slabs reached 256 VGPRs with 647/2,069 spills and 1,568/5,248 private bytes and were rejected before timing. Pair-serial slabs reached `0.01343/0.00959` NRMSE and were 35.0%/82.9% slower. Row-normalized FP16-C reached `0.00723-0.00727` NRMSE; even without its scale scan it was 33.3% slower. The fused IQ2_S pair therefore retains one FP32 accumulation and one BF16 rounding per output.

## Resources

Retained pair bodies use 194 VGPR/54 SGPR/8192 B LDS at M64/N64 and 219 VGPR/54 SGPR/8192 B LDS at M128/N64.

## Evidence

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_pair_n64.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_width8.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_swizzle4.json
~/tmp/torch-ggml-ops/grouped-bwd-production-final-correctness.json
```

The pair-specific source artifacts include the exact N64 comparison and the rejected width/swizzle controls:

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_pair_n64.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_width8.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_swizzle0.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_swizzle4.json
```

The exact FP32 pair contract and IQ2_S pair layout are complete for the current packed kernel.
