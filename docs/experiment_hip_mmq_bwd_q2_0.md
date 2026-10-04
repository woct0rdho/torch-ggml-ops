# HIP MMQ Backward Q2_0 Experiment

## Scope

This record covers the gfx1151 HIP input-gradient backward kernels for Q2_0 weights, opened for the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing). The dense keys are 45.9 MiB over the layers that use Q2_0 outside the experts. The type also carries all 144 routed-expert tensors, 32,400 MiB, which are the grouped records.

### Packed format

`GGML_TYPE_Q2_0` (id 42) is `QK2_0 = 64` weights in an 18-byte block: `ggml_half d`, then `uint8_t qs[16]` holding four *consecutive* 2-bit codes per byte with the lowest bits first, where code `q` is the level `q - 1` (so -1, 0, 1, 2, times `d`). The block is declared in `ggml/src/ggml-common.h` and the CPU reference is `dequantize_row_q2_0` in `ggml/src/ggml-quants.c`. Three properties matter for a kernel: the block is 64 wide, not the 32 of the legacy `_0` types or the 256 of the `_K` and IQ4_XS types. The four codes of one byte are consecutive weights rather than a nibble-plane pair, and the payload has no codebook, no second scale plane and no sign table, so a value costs a shift, a mask, an integer subtract and one fp16 multiply.

### Qwen4-Exp dense families

| Family | Forward weight `(N,K)` | Backward shape `(M,in_features,out_features)` | M values | Tensors |
| --- | ---: | ---: | ---: | ---: |
| QSA attention query | `(12288,2560)` | `(M,2560,12288)` | `2048,8192,32768` | 2 |
| Shared-expert gate/up | `(640,2560)` | `(M,2560,640)` | `2048,8192,32768` | 34 |
| Shared-expert down | `(2560,640)` | `(M,640,2560)` | `2048,8192,32768` | 16 |
| PLE key projection | `(10240,2560)` | `(M,2560,10240)` | `2048,8192,32768` | 2 |

The M values count tokens of a batch at sequence length 2048 (B1/B4/B16). The training path calls `M=2048` for these projections. The 48 layers mix recipes - the same projection family is a different quant type in different layers - so every type that appears needs a body, or those layers fall back to a dequantizing multiply.

## GatedDeltaNet target shapes

`in_proj_qkv` and `in_proj_z` are in scope. `out_proj` is deferred. Both carry only the tiled -> grouped value-head reorder, which the loader applies to their packed *rows* as whole blocks, so the packed weight as loaded is already in the model's ordinary layout: a kernel takes the packed tensor and the layer input as they are, with no permutation, copy or transpose (`gated_delta_net_layout.md`).

`in_proj_qkv` `(10240,2560)` and `in_proj_z` `(6144,2560)` share the shape of the PLE key projection and the QSA key/value family in the table above, so the same two deployment keys serve them, one tensor each. `Packed row` is 720 B for a `(N,2560)` weight. The third projection, `out_proj`, is deferred, and this type carries no `out_proj` tensor in either checkpoint, so no GatedDeltaNet tensor of it stays out of scope.

## Final kernel result

| Family | `(M,K,N)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| QSA query | `(2048,2560,12288)` | 26.672 | 1.135x | `dense_bwd_q2_0_mt128_nt128_ki32_full_nt4mw2_sw0_pad8` |
| QSA query | `(8192,2560,12288)` | 26.386 | 1.022x | `dense_bwd_q2_0_mt128_nt128_ki32_full_nt4mw2_sw0_pad8` |
| QSA query | `(32768,2560,12288)` | 25.210 | 0.960x | `dense_bwd_q2_0_mt128_nt128_ki32_full_nt4mw2_sw0_pad8` |
| Shared-expert gate/up | `(2048,2560,640)` | 33.122 | 1.544x | `dense_bwd_q2_0_mt128_nt128_ki32_full_nt4mw2_pad8` |
| Shared-expert gate/up | `(8192,2560,640)` | 34.313 | 1.272x | `dense_bwd_q2_0_mt128_nt128_ki32_full_nt4mw2_sw0_pad8` |
| Shared-expert gate/up | `(32768,2560,640)` | 33.916 | 1.215x | `dense_bwd_q2_0_mt128_nt128_ki32_full_nt4mw2_sw0_pad8` |
| Shared-expert down | `(2048,640,2560)` | 21.149 | 1.302x | `dense_bwd_q2_0_mt128_nt128_ki32_full_nt8_ki32_pad8` |
| Shared-expert down | `(8192,640,2560)` | 27.203 | 1.629x | `dense_bwd_q2_0_mt128_nt128_ki32_full_nt4mw2_pad8` |
| Shared-expert down | `(32768,640,2560)` | 30.682 | 1.704x | `dense_bwd_q2_0_mt128_nt128_ki32_full_nt4mw2_sw0_pad8` |
| PLE key / GDN `in_proj_qkv` | `(2048,2560,10240)` | 26.800 | 1.143x | `dense_bwd_q2_0_mt128_nt128_ki32_full_nt4mw2_sw0_pad8` |
| PLE key / GDN `in_proj_qkv` | `(8192,2560,10240)` | 26.551 | 1.026x | `dense_bwd_q2_0_mt128_nt128_ki32_full_nt4mw2_sw0_pad8` |
| PLE key / GDN `in_proj_qkv` | `(32768,2560,10240)` | 24.530 | 0.970x | `dense_bwd_q2_0_mt128_nt128_ki32_full_nt4mw2_sw0_pad8` |
| GDN `in_proj_z` | `(2048,2560,6144)` | 27.679 | 1.202x | `dense_bwd_q2_0_mt128_nt128_ki32_full_nt4mw2_pad8` |
| GDN `in_proj_z` | `(8192,2560,6144)` | 28.198 | 1.087x | `dense_bwd_q2_0_mt128_nt128_ki32_full_nt4mw2_sw0_pad8` |
| GDN `in_proj_z` | `(32768,2560,6144)` | 26.643 | 1.064x | `dense_bwd_q2_0_mt128_nt128_ki32_full_nt4mw2_sw0_pad8` |

Every row comes from one run at eight repeats and two blocks against a BF16 `torch.mm` baseline on the same prepared gradient and packed weights. The deployed body is the fastest of the built Q2_0 bodies on that row. The M=8192 and M=32768 rows of the four wide families use the swizzle-free variant of the same tile, which is one to four percent ahead of its swizzled twin there.

## Kernel implementation

The Q2_0 decode shares the `mt128_nt128_ki32_full` skeleton of the Q4_K, Q5_K and Q6_K records: a 128-thread workgroup, four waves, `n_tiles * 16` result columns, `k_iteration` contraction values per stage, and LDS-staged decoded weights. `WEIGHT_BLOCK_VALUES` resolves to `QK2_0 = 64` for this type, so the block index and the in-block value index are computed against 64 rather than 256.

Three decode forms are built. The generic form indexes the payload byte per value, `(qs[value >> 2] >> (2 * (value & 3))) & 0x03`, and subtracts one from the code before scaling by the block's fp16 `d`. It serves the cooperative width-16 group loader and the fallback value, pair and quad paths. The packed form, used when the tile is eight result columns wide with a 32-value stage and a width-16 decoder, loads the four payload bytes covering its 16 values as one unaligned `uint32` and extracts the sixteen codes from it, halving the loader's addressing work. The level table the forward records use is not applicable here: the backward tile carries no codebook, only the per-block scale, so a table would still need the scale multiply.

The deployed bodies are `_nt4mw2_pad8` (64 result columns, a 64-value stage, two row tiles per wave, 8-word LDS padding), `_nt4mw2_sw0_pad8` (the same tile with the LDS swizzle turned off) and `_nt8_ki32_pad8` (the packed eight-column form with padding). `lds_padding` is load-bearing: the decoded tile is written one column per thread at a stride of `k_iteration` values, so without padding every one of a thread's sixteen stores lands in the same LDS bank. Padding of eight values fixes that. Padding of four or sixteen is worse than none.

The launch path also had to stop assuming a 256-value block: `blocks_per_weight_row` was `in_features // 256` in the ordinary and split-contraction launchers and in the measurement harness, which is wrong for every type whose block is not `QK_K`. Both launchers now take it from the type's format record, so a body without exact dimensions gets the right packed-row stride.

## Resources

The deployed bodies use `120-127 VGPR`, `17 SGPR` and `9 KiB` LDS (`_nt4mw2_*`), and `232 VGPR`, `17 SGPR` and `10 KiB` LDS (`_nt8_ki32_pad8`), all spill-free.

## Optimization log

### Packed decode and the padded tile

The first eight bodies (the plain `_k2048` packed tile, a four-column `_nt4`, its ungrouped traversal, two 64-wide-stage variants, an eight-column four-row-tile variant, and two whose decode width or swizzle differed) were screened at four repeats on all five shapes. The four-column tile with a 64-value stage and two row tiles per wave won nearly every point, `1.5-2.0x` over the packed eight-column tile on the wide shapes and `1.09-1.26x` over its own 32-value-stage twin. A four-row-tile variant of it lost `1.18-1.26x`, a one-row-tile variant lost `1.26-1.52x`, a two-column tile lost `1.8x`, and decoder widths of eight or 32 were `1.04-1.23x` behind. The M=2048 row of the shared-expert down projection, whose result is only 640 wide, is the one point that prefers the packed eight-column tile.

Padding the decoded-weight tile by eight values then added `1.01-1.07x`, and turning the LDS swizzle off on top of the padding added another `1.02-1.04x` at M=8192 and above on all four wide families, which is the split the table above deploys. A 128-value contraction stage, a two-wave workgroup and a four-row-tile padded variant all lost, and `m_tiles_per_wave = 3` faults with a memory error, so that geometry is built but must not be selected.

### Why Q2_0 does not outrun the K-quants

Q2_0's decode is cheaper than any K-quant's, but the backward kernel does not spend its time decoding. Stochastic PC sampling of the deployed tile at `(2048,2560,12288)` and `(2048,2560,640)` puts `s_barrier` at `27%` of samples with another `27%` attributed to `BARRIER_WAIT`, against `11%` at the WMMA instruction. The instruction mix is `16%` LDS loads, `11%` matrix ops, `10%` integer and float arithmetic and `10%` global loads, and the stall reasons are `31%` ALU dependency, `27%` barrier wait and `35%` arbiter. The decode's own instructions are the shift, logic and convert entries, `11%` together.

That is the whole explanation of why this record lands at `21-34 TFLOPS` instead of the `48.7` the int8/bf16 WMMA probe measures as this machine's practical ceiling: each contraction stage decodes into LDS, barriers, then multiplies, so the matrix units idle while the tile is filled. A double-buffered decoded-weight tile, which is the standard fix and the one lever the profile points at, is a change to the shared body rather than a Q2_0 knob and is not part of this record.

For the same reason the family ranks with the K-quants rather than ahead of them on comparable shapes: the Q6_K record measures `33.4/35.6/36.6 TFLOPS` on `(M,2560,640)` and `33.7/31.3/27.1` on `(M,6144,2560)` where this record measures `33.1/34.3/33.9` and `27.7/28.2/26.6`.

## Evidence

```text
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q20_doc.txt   (final table)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q20_v{1..7}.txt  (candidate screens)
~/tmp/torch-ggml-ops/q20_bwd/profile_one.py      (single-tile launcher)
~/tmp/torch-ggml-ops/q20_bwd/prof_counters/      (PMC run: L2 88.9%, LDS conflicts 58.3%)
~/tmp/torch-ggml-ops/q20_bwd/prof_pc/            (stochastic PC sampling)
```

The screens and the final table use the deployed preparation, the same BF16 `torch.mm` baseline and the same paired timing as the protocol of the other backward records.
