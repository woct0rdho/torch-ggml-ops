# HIP MMQ Forward Q5_0 Experiment

## Scope

This record covers the gfx1151 HIP forward kernels for Q5_0 weights, opened for the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing).

The type appears in one family only, the shared-expert down projection `(2560,640)` of 7 of the 48 layers, 7.5 MiB of packed weights, at the training token counts of a sequence length 2048 batch (B1/B4/B16).

### Packed format

`GGML_TYPE_Q5_0` (id 6) is `QK5_0 = 32` weights in a 22-byte block: one fp16 `d`, `uint8_t qh[4]` holding the fifth bit of each weight as one bit per weight in a little-endian 32-bit word, then `uint8_t qs[16]` with Q4_0's nibble planes, and the level is `nibble | 16*bit - 16`. The extra plane makes the value signed without a codebook, at the cost of one more payload word per block.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| ---: | --- | --- | --- | --- |
| Shared-expert down | `(2048,2560,640)` | 26.909 | 1.42x | `dense_fwd_q5_0_k640_j128_full_table` |
| Shared-expert down | `(8192,2560,640)` | 27.155 | 1.14x | `dense_fwd_q5_0_k640_j128_full_table` |
| Shared-expert down | `(32768,2560,640)` | 27.350 | 1.11x | `dense_fwd_q5_0_k640_j128_full_table` |

All three points are above the BF16 `torch.mm` baseline on the multiply-only surface.

## Kernel implementation

The body is the four-wave `I=64`, `J=128` skeleton of the other dense types and reuses their two mechanisms for this contraction. A stage holds eight 22-byte blocks, whose expanded int8 values fill the same LDS slots the Q8_0 loader produces: the low nibbles of four payload bytes in the lower four ints of the 32-value group and the high nibbles in the upper four, with the fifth bits of the group merged as bit 4 of each byte and the level offset of sixteen subtracted once per byte. The 32-wide block is one 32-value step, so the tile holds one scale per group, and the vendored Q8_0 vector dot consumes the tile unchanged.

The deployed body reads the fifth-bit spread from a 16-entry table in LDS instead of building it with four shifts, four masks and three ors per group. The nibble plane needs no expansion because it already is one nibble per byte.

The contraction is two and a half stages, so the body loads a tail window that ends at the row end and consumes only its upper half, the same tail stage the Q8_0, Q2_0 and Q4_0 records describe, and the window load reads the table as well. The generic stage-indexed control cannot express that length at all: it silently computes a 512-value product, so a K=640 key must always select the tail body.

## Optimization log

### Exact specialization and the tail stage

The tail body is worth `1.11-1.19x` against the generic control, the sum of the exact contraction length, the full-tile bounds and the tail stage the generic control does not have. Because the generic control would be wrong rather than slow on this key, the deployment keys exist only for the tail body.

### Fifth-bit table

`rocprofv3` PC sampling of the plain body at `(8192,2560,640)` reports `15.7%` WMMA with `10.6%` in shifts and logic, against `17.2%` WMMA and `6.5%` shifts for the Q4_0 body at the same geometry: the fifth-bit merge is the largest decode share of any type in the bundle, which is why this type is the one where the level-table mechanism pays most.

The first table attempt was wrong and slower, and the measurement is worth recording: it expanded *both* planes through 256-entry nibble tables, but the nibble plane is already one nibble per byte, so the table replicated each payload byte's nibble over all four output bytes and lost `2.3-3.4%` on top of returning wrong values. The corrected single-table body, one 16-entry fifth-bit spread per group, is bit-exact against the plain body and worth `3.0-4.2%` on all three points, so the deployed keys select it. The table costs `64 B` of LDS behind the weight tile and does not change the number of resident workgroups.

### Closed mechanisms

Global J64, the wide `I=128` tile, activation-half double buffering of the weight tile, decoded-weight LDS caching, split-K, persistent workgroups, and the hoisted epilogue are closed for this family of shapes on the sibling types: the wide tile lost by `1.05-1.21x` on Q3_K, Q4_K and Q5_K, and the hoisted epilogue was neutral for Q3_K, Q4_K, Q5_K and Q8_0. The remaining decode-side cost is the int32-to-fp32 conversion and the scale multiply of the epilogue, which every type in the family pays.

## Resources

The deployed table body uses `214 VGPR / 30 SGPR / 38,464 B LDS` and is spill-free. The plain tail body is the same at `38,400 B`.

## Evidence

```text
~/tmp/torch-ggml-ops/qwen4_fwd/sweep_q50_v1.txt
```

The Qwen4-Exp shape has a HIP control but no GGTensile problem key yet, so it cannot be selected as a deployment case. Its runs use the deployed Q8_1 producer and the same prepared inputs, the same `torch.mm` BF16 baseline on the same activation tensor, and the same paired timing as the official protocol.
