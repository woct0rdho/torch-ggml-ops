# HIP MMQ Forward IQ4_XS Experiment

## Scope

This record covers the gfx1151 HIP forward kernels for IQ4_XS weights, opened for the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing).

The dense keys are the QSA attention query `(12288,2560)`, QSA key/value `(512,2560)`, shared-expert gate/up `(640,2560)`, GatedDeltaNet `in_proj_qkv` `(10240,2560)` and `in_proj_z` `(6144,2560)`, all at the training token counts of a sequence length 2048 batch (B1/B4/B16). That is 77.5 MiB of packed weights, including the query projection of 4 of the 12 attention layers. GatedDeltaNet `out_proj` is deferred because wiring it needs the activation permutation. This type carries `(2560,6144)` in 17 layers.

### Packed format

`GGML_TYPE_IQ4_XS` (id 23) is `QK_K = 256` weights in a 136-byte block: one fp16 `d`, `uint16_t scales_h`, `uint8_t scales_l[4]`, then `uint8_t qs[128]`. The 256 weights form eight 32-value sub-blocks. Sub-block `i` owns 16 payload bytes, whose low nibbles are its values `0..15` and high nibbles its values `16..31`, each indexing the sixteen-entry codebook IQ4_NL uses. Its 6-bit signed scale has its low nibble in `scales_l[i/2]` and its top two bits at `scales_h` bits `2i`, and the value is `d * (scale6 - 32) * level`. One stage therefore consumes a whole block, with eight scale entries per row.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| ---: | --- | --- | --- | --- |
| QSA query | `(2048,12288,2560)` | 29.384 | 1.25x | `dense_fwd_iq4_xs_k2560_j128_full` |
| QSA query | `(8192,12288,2560)` | 29.475 | 1.23x | `dense_fwd_iq4_xs_k2560_j128_full` |
| QSA query | `(32768,12288,2560)` | 28.119 | 1.21x | `dense_fwd_iq4_xs_k2560_j128_full` |
| QSA key/value | `(2048,512,2560)` | 26.856 | 1.93x | `dense_fwd_iq4_xs_k2560_j128_full` |
| QSA key/value | `(8192,512,2560)` | 30.178 | 1.61x | `dense_fwd_iq4_xs_k2560_j128_full` |
| QSA key/value | `(32768,512,2560)` | 29.642 | 1.47x | `dense_fwd_iq4_xs_k2560_j128_full` |
| Shared-expert gate/up | `(2048,640,2560)` | 26.296 | 1.84x | `dense_fwd_iq4_xs_k2560_j128_full` |
| Shared-expert gate/up | `(8192,640,2560)` | 30.093 | 2.03x | `dense_fwd_iq4_xs_k2560_j128_full` |
| Shared-expert gate/up | `(32768,640,2560)` | 29.831 | 1.98x | `dense_fwd_iq4_xs_k2560_j128_full` |
| GatedDeltaNet QKV | `(2048,10240,2560)` | 29.163 | 1.25x | `dense_fwd_iq4_xs_k2560_j128_full` |
| GatedDeltaNet QKV | `(8192,10240,2560)` | 29.286 | 1.24x | `dense_fwd_iq4_xs_k2560_j128_full` |
| GatedDeltaNet QKV | `(32768,10240,2560)` | 28.030 | 1.22x | `dense_fwd_iq4_xs_k2560_j128_full` |
| GatedDeltaNet Z | `(2048,6144,2560)` | 30.125 | 1.30x | `dense_fwd_iq4_xs_k2560_j128_full` |
| GatedDeltaNet Z | `(8192,6144,2560)` | 29.704 | 1.24x | `dense_fwd_iq4_xs_k2560_j128_full` |
| GatedDeltaNet Z | `(32768,6144,2560)` | 28.072 | 1.22x | `dense_fwd_iq4_xs_k2560_j128_full` |

Every point is above the BF16 `torch.mm` baseline on the multiply-only surface, from `1.21x` to `2.03x`.

## Kernel implementation

The body is the four-wave `I=64`, `J=128` skeleton of the other dense types, with the 256-wide stage of the K-quant bodies. One stage is exactly one block: the 128 payload bytes are read as 32 groups of four bytes, each decoded through the IQ4 byte-permute codebook lookup of the IQ4_NL record, so the eight 32-value groups land in the Q8_0 LDS slots in value order and the eight sub-block scales land in the eight f32 scale entries. The vendored Q8_0 vector dot consumes the tile unchanged.

Unlike the IQ4_NL, Q4_0 and Q5_0 bodies there is no tail stage: `K=2560` is ten whole blocks, so the contraction is exact in both directions and the generic control differs only by folding.

## Optimization log

### Exact specialization

Exact specialization is worth `1.04-1.09x` over the generic control across the fifteen points, the full-tile bounds and the folded block count.

### Decode cost against the sibling types

The decode here carries two metadata planes and a codebook lookup per value, so it costs more than the nibble types, and the cost shows at the narrow end: the body runs `26.3 TFLOPS` on `(2048,640,2560)` and `26.9` on `(2048,512,2560)`, against `29.4-30.2` on the wide shapes, where the same decode is a smaller share of a longer stage.

Nothing in the decode is left to fold: the codebook is resolved by byte permutes at the cost of the offset arithmetic the Q4_0 body uses, the sub-block scales are six-bit fields whose extraction is two shifts and two masks per 32-value group, and the payload planes are already one nibble per byte. The body is therefore at the same place as its siblings: the epilogue, one int32-to-fp32 conversion and one scale multiply per output element per MMA step, is the remaining decode-side cost, and every type in the family pays it.

### Closed mechanisms

Global J64, the wide `I=128` tile, activation-half double buffering of the weight tile, decoded-weight LDS caching, split-K, persistent workgroups, and the hoisted epilogue are closed for this family of shapes on the sibling types: the wide tile lost by `1.05-1.21x` on Q3_K, Q4_K and Q5_K, and the hoisted epilogue was neutral for Q3_K, Q4_K, Q5_K and Q8_0.

## Resources

The exact body uses `222 VGPR / 30 SGPR / 38,400 B LDS` and is spill-free.

## Evidence

```text
~/tmp/torch-ggml-ops/qwen4_fwd/sweep_iq4xs_v1.txt
```

The Qwen4-Exp shapes have HIP controls but no GGTensile problem key yet, so they cannot be selected as deployment cases. Their runs use the deployed Q8_1 producer and the same prepared inputs, the same `torch.mm` BF16 baseline on the same activation tensor, and the same paired timing as the official protocol.
