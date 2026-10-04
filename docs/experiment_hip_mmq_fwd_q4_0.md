# HIP MMQ Forward Q4_0 Experiment

## Scope

This record covers the gfx1151 HIP forward kernels for Q4_0 weights, opened for the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing).

The type appears in one family only, the shared-expert down projection `(2560,640)` of 16 of the 48 layers, 14.1 MiB of packed weights, at the training token counts of a sequence length 2048 batch (B1/B4/B16).

### Packed format

`GGML_TYPE_Q4_0` (id 2) is `QK4_0 = 32` weights in an 18-byte block: one fp16 `d`, then `uint8_t qs[16]` whose low nibble holds weight `j` and high nibble weight `j + 16` for byte `j`, with the level `q - 8`. This is the cheapest payload in the bundle after Q2_0: one nibble-plane pair, one offset, one scale, no codebook and no second metadata plane.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| ---: | --- | --- | --- | --- |
| Shared-expert down | `(2048,2560,640)` | 27.912 | 1.47x | `dense_fwd_q4_0_k640_j128_full` |
| Shared-expert down | `(8192,2560,640)` | 28.673 | 1.19x | `dense_fwd_q4_0_k640_j128_full` |
| Shared-expert down | `(32768,2560,640)` | 28.724 | 1.17x | `dense_fwd_q4_0_k640_j128_full` |

All three points are above the BF16 `torch.mm` baseline on the multiply-only surface.

## Kernel implementation

The body is the four-wave `I=64`, `J=128` skeleton of the other dense types and reuses their two mechanisms for this contraction. A stage holds eight 18-byte blocks, whose expanded int8 values fill the same LDS slots the Q8_0 loader produces: the low nibbles of four payload bytes go to the lower four ints of the 32-value group and the high nibbles to the upper four, each with the level offset of eight subtracted per byte. The 32-wide block is exactly one 32-value step, so the tile holds one scale per group with no replication, and the vendored Q8_0 vector dot consumes the tile unchanged.

The contraction is two and a half stages, so the body loads a tail window that ends at the row end and consumes only its upper half, the same tail stage the Q8_0 and Q2_0 records describe. The generic stage-indexed control cannot express that length at all: it silently computes a 512-value product, so a K=640 key must always select the tail body.

## Optimization log

### Exact specialization and the tail stage

The tail body is worth `1.13-1.19x` against the generic control, which is the sum of the exact contraction length, the full-tile bounds and the tail stage that the generic control does not have. Because the generic control would be wrong rather than slow on this key, the deployment keys exist only for the tail body.

### Instruction mix

`rocprofv3` PC sampling of the tail body at `(8192,2560,640)` reports `17.2%` WMMA, `19.7%` conversions, `16.0%` arithmetic and `6.5%` shifts and logic, i.e. the same profile as the Q2_0 body and the most MMA-bound shape in the family, with the WMMA share level with the Q8_0 bodies and above Q3_K and Q4_K at their own geometries.

The nibble decode is only `6.5%` of the stream, two shifts, two masks and two per-byte level subtractions per eight values, so the level-table mechanism that pays for Q2_0 was not applied here: the Q2_0 table buys `1.2-2.4%` by removing a `7.3%` share, and a table read costs one LDS lookup per four values against two arithmetic operations, which is not a clear trade at this share.

### Closed mechanisms

Global J64, the wide `I=128` tile, activation-half double buffering of the weight tile, decoded-weight LDS caching, split-K, persistent workgroups, and the hoisted epilogue are closed for this family of shapes on the sibling types: the wide tile lost by `1.05-1.21x` on Q3_K, Q4_K and Q5_K, and the hoisted epilogue was neutral for Q3_K, Q4_K, Q5_K and Q8_0. The remaining decode-side cost is the int32-to-fp32 conversion and the scale multiply of the epilogue, which every type in the family pays.

## Resources

The tail body uses `213 VGPR / 30 SGPR / 38,400 B LDS` and is spill-free.

## Evidence

```text
~/tmp/torch-ggml-ops/qwen4_fwd/sweep_q40_v1.txt
```

The Qwen4-Exp shape has a HIP control but no GGTensile problem key yet, so it cannot be selected as a deployment case. Its runs use the deployed Q8_1 producer and the same prepared inputs, the same `torch.mm` BF16 baseline on the same activation tensor, and the same paired timing as the official protocol.
