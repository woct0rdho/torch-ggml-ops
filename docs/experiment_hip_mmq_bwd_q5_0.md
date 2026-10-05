# HIP MMQ Backward Q5_0 Experiment

## Scope

This record covers the gfx1151 HIP input-gradient backward kernels for Q5_0 weights. The type appears only in the shared-expert down projection of 7 of the 48 layers of the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, hidden size 2560), 7.5 MiB of packed weights, so the record has one dense family: a `(2560,640)` weight, backward `(M,640,2560)` over the B1/B4/B16 token counts of a sequence length 2048 batch. The 48 layers mix recipes, and a projection family whose type has no body falls back to a dequantizing multiply.

The type is Q4_0's nibble-plane payload plus one bit plane. `QK5_0 = 32` weights share a 22-byte block: an fp16 scale, a 32-bit little-endian `qh` word whose bit `j` is bit 4 of weight `j`, and the 16 payload bytes holding weight `j` in the low nibble and weight `j + 16` in the high nibble. The level is `q - 16` once the fifth bit is merged as bit 4, so the extra plane makes the value signed without a codebook, at the cost of one word load, one variable shift and one or per value over Q4_0. The 32-wide block puts one scale per weight row under a 32-wide contraction step.

## Final kernel result

| Family | `(M,K,N)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| Shared-expert down | `(2048,640,2560)` | 23.662 | 1.447x | `dense_bwd_q5_0_pipea_nt4_ki64_mw2_sw16_g0` |
| Shared-expert down | `(8192,640,2560)` | 28.818 | 1.737x | `dense_bwd_q5_0_pipea_nt4_ki64_mw2_pad8` |
| Shared-expert down | `(32768,640,2560)` | 29.657 | 1.664x | `dense_bwd_q5_0_pipea_nt4_ki64_mw2_sw16` |

## Kernel implementation

The body is the reusable pipelined one (`csrc/ck/mmq_backward_pipelined.cuh`), the same body the Q2_0, Q4_0, Q3_K, Q4_K, Q5_K and Q6_K records use: a 128-thread workgroup, `n_tiles * 16` result columns, `K_ITERATION` contraction values per stage, two shared tiles with one barrier per stage, and register double buffering of the activation fragments.

The decode reads the nibble plane and the fifth bit per value: `(qs[value & 15] >> (4 * ((value >> 4) & 1))) & 0x0f` into bit 4 from `(qh_word >> value) & 1`, minus sixteen, times the block's fp16 `d`. The width-16 group form is what the deployed tile uses. A prefetched form reads the block's sixteen payload bytes as one `uint4` and takes the fifth bits from one word.

The forward records' 16-entry fifth-bit spread table does not transfer here. It pays there because the forward tile expands a whole 32-value block into byte lanes at once, while this tile decodes sixteen values per group and the fifth-bit merge is one shift and one or. The Q4_0 record, which has no fifth bit at all, measures the same band as this one, so the merge is not a visible cost.

## Resources

The deployed body uses `160 VGPR`, `19 SGPR` and `16 KiB` LDS (two 64x64 bf16 tiles), spill-free.

## Optimization log

### Candidate screen

The Q4_0 record's six candidates were rendered for this type and screened on all three row counts. The result is the same shape as Q4_0's, with the same reasons: the two-row-tile pipelined tile wins every row count (`1.09-1.44x` over the four-row-tile variants and `1.12-1.27x` over the 32-value stage), padding loses to the swizzle-only 8 KiB tile, and the three variants that differ in the activation prefetch and the payload read are within the run-to-run spread of each other (`0.95-1.05x`). The deployed body is the swizzle-only pipelined tile with the activation prefetch, chosen for its smaller LDS footprint at equal speed. The prefetched payload read is neutral on this type, as it was on Q4_0.

### Where this type sits

`23.7/27.9/29.7 TFLOPS` is above the BF16 baseline on every row. The type's shape is the Q4_0 family's, and the two compare directly: Q4_0 measures `24.0/28.1/31.1` on the same keys. The fifth-bit plane therefore costs about four percent at the largest row count and nothing at the smallest, which is the expected size of one extra word load and shift per value against a payload that is already the cheapest in the bundle. The `M=2048` key deploys the M-fastest traversal that the Q4_0 record describes (`1.23x` ahead of the deployed order under repeated launches). The single-row-tile, 32-column-tile and slice twins were screened with it and lose or tie there. Both types sit behind the K-quant families at the equivalent width - Q5_K measures `30.4/30.0/33.9` and Q6_K `36.6/37.7/38.0` on `(M,2560,640)` - and the reason is the result width rather than the decode: at `M=2048` this family's grid is `16 x 10` workgroups, which is why the larger-tile variants lose and why the `M=2048` row is the weakest of the three.

### Padding on the pipelined tile

The pipelined tile's layout was chosen when the tile was first built - sixteen-value swizzle chunks, no padding - and the chunk was later swept on this body without revisiting padding. The layout is now screened over every deployed dense-backward key: the padding twin of each deployed body (eight values of row padding, no swizzle) is timed against it at the key's own row counts, four repeats per block, and every screen winner is then re-timed at eight repeats over two blocks in both measurement orders, so a key moves only when the padding body wins in both directions.

Padding is not a general replacement for the swizzle. It wins where the resident grid is thin and the result is narrow - the `M=2048` rows and the 512- and 640-wide results - and loses on the wide results at the largest row counts, where the decoded tile's shared-memory traffic is high enough that the swizzle's bank pattern still pays. The split is per key rather than per type: on this type the `(M,640,2560)` shared-expert down row at `M=8192` moves, `3.7%` ahead.

A second layout round then screened three further combinations on every officially measured body - sixteen-value padding, four-value padding, and eight-value padding with an eight-value swizzle chunk - at the deployed row counts. Nothing survives the confirmation: each candidate measures within `1-3%` of the deployed tile in one measurement order and loses in the other, so all three are rejected and none is built. The layout neighborhood is closed at the combinations the deployment uses.

## Evidence

```text
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q50_v1.txt              (candidate screen)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q50_v2.txt              (confirmation)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_reopen_gridshape.txt    (grouped-M, row-tile and slice screens)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_reopen_confirm.txt      (repeated-launch paired confirmation)
```
