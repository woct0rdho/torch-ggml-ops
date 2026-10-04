# HIP MMQ Forward IQ4_NL Experiment

## Scope

This record covers the gfx1151 HIP forward kernels for IQ4_NL weights, opened for the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing).

The type appears in one family only, the shared-expert down projection `(2560,640)` of 7 of the 48 layers, 6.2 MiB of packed weights, which is the smallest dense footprint of the group, at the training token counts of a sequence length 2048 batch (B1/B4/B16). `per_layer_token_embd.weight` is also IQ4_NL but is an embedding gather rather than a multiply, so it stays on the GGUF embedding module.

### Packed format

`GGML_TYPE_IQ4_NL` (id 20) is `QK4_NL = 32` weights in an 18-byte block: one fp16 `d`, then `uint8_t qs[16]` with Q4_0's nibble planes, but each nibble indexes a sixteen-entry level table instead of being offset arithmetic. The table is the codebook `-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113`, identical for every block, and the levels are signed int8 with a magnitude up to 127 rather than the four-bit range, so the scale does all the range work.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| ---: | --- | --- | --- | --- |
| Shared-expert down | `(2048,2560,640)` | 27.955 | 1.47x | `dense_fwd_iq4_nl_k640_j128_full` |
| Shared-expert down | `(8192,2560,640)` | 28.624 | 1.20x | `dense_fwd_iq4_nl_k640_j128_full` |
| Shared-expert down | `(32768,2560,640)` | 28.640 | 1.16x | `dense_fwd_iq4_nl_k640_j128_full` |

All three points are above the BF16 `torch.mm` baseline on the multiply-only surface.

## Kernel implementation

The body is the four-wave `I=64`, `J=128` skeleton of the other dense types. A stage holds eight 18-byte blocks, whose levels fill the same LDS slots the Q8_0 loader produces: the low nibbles of four payload bytes in the lower four ints of the 32-value group and the high nibbles in the upper four, with one scale per 32-value group, so the vendored Q8_0 vector dot consumes the tile unchanged.

The codebook is resolved with byte permutes rather than a memory gather. For each group of eight nibbles, the low three bits of every index select a byte of one half of the table with one permute per plane and half, and the fourth bit of every index selects between the halves with a second pair of permutes, so the whole lookup is six permutes and about ten index operations. The table itself is the sixteen-byte constant the quantizer defines, so no per-block state is loaded.

The contraction is two and a half stages, so the body loads a tail window that ends at the row end and consumes only its upper half, the same tail stage the Q8_0, Q2_0, Q4_0 and Q5_0 records describe. The generic stage-indexed control cannot express that length at all: it silently computes a 512-value product, so a K=640 key must always select the tail body.

## Optimization log

### Exact specialization and the tail stage

The tail body is worth `1.13-1.18x` against the generic control, the sum of the exact contraction length, the full-tile bounds and the tail stage the generic control does not have. Because the generic control would be wrong rather than slow on this key, the deployment keys exist only for the tail body.

### Codebook lookup cost

The lookup is the only decode question this type raises, since the table is not offset arithmetic. The measured answer is that it is free relative to the arithmetic form: the retained body reaches `27.96/28.62/28.64 TFLOPS`, which is the same band as the Q4_0 body at the same geometry (`27.91/28.67/28.72`), so six permutes per eight values cost what two shifts, two masks and two per-byte subtractions cost. That is why this type carries no level-table mechanism: the permute form already is the cheap form, and unlike the fifth-bit merge of Q5_0 it does not leave a shift-and-mask chain to remove.

### Closed mechanisms

Global J64, the wide `I=128` tile, activation-half double buffering of the weight tile, decoded-weight LDS caching, split-K, persistent workgroups, and the hoisted epilogue are closed for this family of shapes on the sibling types: the wide tile lost by `1.05-1.21x` on Q3_K, Q4_K and Q5_K, and the hoisted epilogue was neutral for Q3_K, Q4_K, Q5_K and Q8_0. The remaining decode-side cost is the int32-to-fp32 conversion and the scale multiply of the epilogue, which every type in the family pays.

## Resources

The tail body uses `213 VGPR / 33 SGPR / 38,400 B LDS` and is spill-free.

## Evidence

```text
~/tmp/torch-ggml-ops/qwen4_fwd/sweep_iq4nl_v1.txt
```

The Qwen4-Exp shape has a HIP control but no GGTensile problem key yet, so it cannot be selected as a deployment case. Its runs use the deployed Q8_1 producer and the same prepared inputs, the same `torch.mm` BF16 baseline on the same activation tensor, and the same paired timing as the official protocol.
