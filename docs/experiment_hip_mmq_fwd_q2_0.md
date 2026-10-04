# HIP MMQ Forward Q2_0 Experiment

## Scope

This record covers the gfx1151 HIP forward kernels for Q2_0 weights, opened for the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing).

The dense keys are the QSA attention query `(12288,2560)`, shared-expert gate/up `(640,2560)`, shared-expert down `(2560,640)`, the PLE key projection `(10240,2560)`, and the GatedDeltaNet `in_proj_z` `(6144,2560)`, all at the training token counts of a sequence length 2048 batch (B1/B4/B16). GatedDeltaNet `in_proj_qkv` shares the PLE key shape. Those dense tensors are 45.9 MiB over the layers that use Q2_0 outside the experts. GatedDeltaNet `out_proj` is deferred because wiring it needs the activation permutation, and the type carries no such tensor in either checkpoint.

The type also carries all 144 routed-expert tensors, 32,400 MiB, which are the grouped records.

### Packed format

`GGML_TYPE_Q2_0` (id 42) is `QK2_0 = 64` weights in an 18-byte block: one fp16 `d`, then `uint8_t qs[16]` holding four *consecutive* 2-bit codes per byte with the lowest bits first, where code `q` is the level `q - 1` (so -1, 0, 1, 2, times `d`). Three properties matter for a kernel: the block is 64 wide, not the 32 of the legacy `_0` types or the 256 of the `_K` types. The four codes of one byte are consecutive weights rather than a nibble-plane pair, and the payload has no codebook, no second scale plane and no sign table, so a value costs a spread, a per-byte level offset and one fp32 multiply with the shared scale.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| ---: | --- | --- | --- | --- |
| QSA query | `(2048,12288,2560)` | 29.556 | 1.24x | `dense_fwd_q2_0_k2560_j128_full_table` |
| QSA query | `(8192,12288,2560)` | 29.813 | 1.24x | `dense_fwd_q2_0_k2560_j128_full_table` |
| QSA query | `(32768,12288,2560)` | 28.487 | 1.22x | `dense_fwd_q2_0_k2560_j128_full_table` |
| Shared-expert gate/up | `(2048,640,2560)` | 25.415 | 1.79x | `dense_fwd_q2_0_k2560_j128_full_table` |
| Shared-expert gate/up | `(8192,640,2560)` | 29.667 | 2.00x | `dense_fwd_q2_0_k2560_j128_full_table` |
| Shared-expert gate/up | `(32768,640,2560)` | 29.587 | 1.97x | `dense_fwd_q2_0_k2560_j128_full_table` |
| Shared-expert down | `(2048,2560,640)` | 26.954 | 1.42x | `dense_fwd_q2_0_k640_j128_full` |
| Shared-expert down | `(8192,2560,640)` | 27.447 | 1.14x | `dense_fwd_q2_0_k640_j128_full` |
| Shared-expert down | `(32768,2560,640)` | 27.600 | 1.12x | `dense_fwd_q2_0_k640_j128_full` |
| PLE key / GDN QKV | `(2048,10240,2560)` | 29.302 | 1.24x | `dense_fwd_q2_0_k2560_j128_full_table` |
| PLE key / GDN QKV | `(8192,10240,2560)` | 28.559 | 1.26x | `dense_fwd_q2_0_k2560_j128_full_table` |
| PLE key / GDN QKV | `(32768,10240,2560)` | 28.394 | 1.23x | `dense_fwd_q2_0_k2560_j128_full_table` |
| GatedDeltaNet Z | `(2048,6144,2560)` | 29.510 | 1.26x | `dense_fwd_q2_0_k2560_j128_full_table` |
| GatedDeltaNet Z | `(8192,6144,2560)` | 29.732 | 1.24x | `dense_fwd_q2_0_k2560_j128_full_table` |
| GatedDeltaNet Z | `(32768,6144,2560)` | 28.520 | 1.23x | `dense_fwd_q2_0_k2560_j128_full_table` |

Every point is above the BF16 `torch.mm` baseline on the multiply-only surface, from `1.12x` to `2.00x`.

## Kernel implementation

The body is the four-wave `I=64`, `J=128`, K256 skeleton of the other dense types. A stage holds four 18-byte blocks, whose 256 expanded int8 values fill the same LDS slots the Q8_0 loader produces: eight ints per 32-value group, so the vendored Q8_0 vector dot consumes the tile unchanged. Each block scale is repeated over the two 32-value scale entries it covers, which is what makes the 64-wide block fit a 32-value step.

The decode spreads the four codes of a payload byte to one byte each and subtracts the level offset of one per byte, which is four operations plus one subtract for four values, and the deployed K2560 bodies read the expansion from a 256-entry level table held in LDS behind the weight tile instead, leaving one lookup per four values. The K=640 contraction is two and a half stages, so its body loads a tail window that ends at the row end and consumes only its upper half, the same tail stage the Q8_0 record describes. The window load cannot read the level table, so the tail body keeps the arithmetic decode.

## Optimization log

### Exact specialization and the tail stage

Exact specialization folds the contraction length and the full-tile bounds into the kernel. It is worth `3.8-7.5%` over the generic control on the K2560 shapes (`1.038-1.067x`), and `1.12-1.16x` on the K=640 shape whose generic control cannot express a two-and-a-half-stage contraction at all: it silently computes a 512-value product, so a K=640 key must always select the tail body.

### Level table

The 2-bit decode is a real share of the instruction stream for this type: `rocprofv3` PC sampling of the retained K2560 body at `(8192,12288,2560)` reports `18.0%` WMMA, the highest of the dense bodies (`7.5%` for Q3_K at the same geometry), with `7.3%` in the spread and shift instructions and the rest in the epilogue and barriers. Reading the expansion from a 256-entry LDS table instead of recomputing it removes most of that share and is worth `1.2-2.4%` on all six measured points (`0.975-1.002x`), which is why the K2560 keys deploy the table body. The table costs `1,024 B` of LDS behind the weight tile, which does not change the number of resident workgroups.

The same sampling puts the body at `58-61%` of the `48.7 TFLOPS` practical WMMA ceiling the Q3_K record measures for `v_wmma_i32_16x16x16_iu8` with this accumulator count, against `45-51%` for Q3_K and `55-60%` for Q4_K, so the expectation that the simple decode pays off holds: this is the most MMA-bound dense body of the family.

### Closed mechanisms

Global J64, the wide `I=128` tile, activation-half double buffering of the weight tile, decoded-weight LDS caching, split-K, persistent workgroups, and the hoisted epilogue are closed for this family of shapes on the sibling types: the wide tile lost by `1.05-1.21x` on Q3_K, Q4_K and Q5_K, the hoisted epilogue was neutral for Q3_K, Q4_K, Q5_K and Q8_0, and the remaining gap to the ceiling is the int32-to-fp32 conversion and the scale multiply of the epilogue, which every type in the family pays.

## Resources

The exact K2560 bodies use `212 VGPR / 29 SGPR / 39,424 B LDS` with the level table and `38,400 B` without it. The K=640 tail body uses `212 VGPR / 29 SGPR / 38,400 B LDS`. All are spill-free.

## Evidence

```text
~/tmp/torch-ggml-ops/qwen4_fwd/sweep_q20_v1.txt
~/tmp/torch-ggml-ops/qwen4_fwd/sweep_q20_v2.txt
```

The Qwen4-Exp shapes have HIP controls but no GGTensile problem key yet, so they cannot be selected as deployment cases. Their runs use the deployed Q8_1 producer and the same prepared inputs, the same `torch.mm` BF16 baseline on the same activation tensor, and the same paired timing as the official protocol.
