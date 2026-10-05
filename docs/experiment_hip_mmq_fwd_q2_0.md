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
| QSA query | `(2048,12288,2560)` | 30.382 | 1.30x | `dense_fwd_q2_0_k2560_j128_full_table_frag` |
| QSA query | `(8192,12288,2560)` | 30.650 | 1.28x | `dense_fwd_q2_0_k2560_j128_full_table_frag` |
| QSA query | `(32768,12288,2560)` | 30.559 | 1.27x | `dense_fwd_q2_0_k2560_j128_full_table_frag` |
| Shared-expert gate/up | `(2048,640,2560)` | 26.103 | 1.85x | `dense_fwd_q2_0_k2560_j128_full_table_frag` |
| Shared-expert gate/up | `(8192,640,2560)` | 32.249 | 2.25x | `dense_fwd_q2_0_k2560_j128_full_table_frag` |
| Shared-expert gate/up | `(32768,640,2560)` | 30.771 | 2.04x | `dense_fwd_q2_0_k2560_j128_full_table_frag` |
| Shared-expert down | `(2048,2560,640)` | 26.873 | 1.42x | `dense_fwd_q2_0_k640_j128_full` |
| Shared-expert down | `(8192,2560,640)` | 27.571 | 1.14x | `dense_fwd_q2_0_k640_j128_full` |
| Shared-expert down | `(32768,2560,640)` | 27.763 | 1.13x | `dense_fwd_q2_0_k640_j128_full` |
| PLE key / GDN QKV | `(2048,10240,2560)` | 30.333 | 1.30x | `dense_fwd_q2_0_k2560_j128_full_table_frag` |
| PLE key / GDN QKV | `(8192,10240,2560)` | 30.513 | 1.29x | `dense_fwd_q2_0_k2560_j128_full_table_frag` |
| PLE key / GDN QKV | `(32768,10240,2560)` | 30.388 | 1.28x | `dense_fwd_q2_0_k2560_j128_full_table_frag` |
| GatedDeltaNet Z | `(2048,6144,2560)` | 30.864 | 1.36x | `dense_fwd_q2_0_k2560_j128_full_table_frag` |
| GatedDeltaNet Z | `(8192,6144,2560)` | 30.240 | 1.28x | `dense_fwd_q2_0_k2560_j128_full_table_frag` |
| GatedDeltaNet Z | `(32768,6144,2560)` | 30.496 | 1.28x | `dense_fwd_q2_0_k2560_j128_full_table_frag` |

Every point is above the BF16 `torch.mm` baseline on the multiply-only surface, from `1.13x` to `2.25x`.

## Kernel implementation

The body is the four-wave `I=64`, `J=128`, K256 skeleton of the other dense types. A stage holds four 18-byte blocks, whose 256 expanded int8 values fill the same LDS slots the Q8_0 loader produces: eight ints per 32-value group, so the vendored Q8_0 vector dot consumes the tile unchanged. Each block scale is repeated over the two 32-value scale entries it covers, which is what makes the 64-wide block fit a 32-value step.

The decode spreads the four codes of a payload byte to one byte each and subtracts the level offset of one per byte, which is four operations plus one subtract for four values, and the deployed K2560 bodies read the expansion from a 256-entry level table held in LDS behind the weight tile instead, leaving one lookup per four values. The K=640 contraction is two and a half stages, so its body loads a tail window that ends at the row end and consumes only its upper half, the same tail stage the Q8_0 record describes. The window load cannot read the level table, so the tail body keeps the arithmetic decode.

## Optimization log

### Exact specialization and the tail stage

Exact specialization folds the contraction length and the full-tile bounds into the kernel. It is worth `3.8-7.5%` over the generic control on the K2560 shapes (`1.038-1.067x`), and `1.12-1.16x` on the K=640 shape whose generic control cannot express a two-and-a-half-stage contraction at all: it silently computes a 512-value product, so a K=640 key must always select the tail body.

### Level table

The 2-bit decode is a real share of the instruction stream for this type: `rocprofv3` PC sampling of the retained K2560 body at `(8192,12288,2560)` reports `18.0%` WMMA, the highest of the dense bodies (`7.5%` for Q3_K at the same geometry), with `7.3%` in the spread and shift instructions and the rest in the epilogue and barriers. Reading the expansion from a 256-entry LDS table instead of recomputing it removes most of that share and is worth `1.2-2.4%` on all six measured points (`0.975-1.002x`), which is why the K2560 keys deploy the table body. The table costs `1,024 B` of LDS behind the weight tile, which does not change the number of resident workgroups.

That WMMA share is the highest sampled on any dense body of the family (`7.5%` for Q3_K at the same geometry, `17.2%` for the Q4_0 tail body), so the expectation that the simple decode pays off holds: this is the most MMA-bound dense body of the family.

### Fragment-order activation reads

The activation used to take the long way round: the producer writes a Q8_1 block per token, the body copies every stage's blocks into an LDS tile, and the vector dot reads its B fragments back out of that tile with `load_ldmatrix`. The fragment the int8 WMMA wants is, per lane, the thirty-two contiguous bytes of one 32-value group of one token - two sixteen-byte chunks at `xs0 + (lane % 16)*stride` and `+16 B` - and a Q8_1 block already stores its four 32-byte groups back to back behind a 16-byte metadata header, so those bytes are contiguous in the workspace too. The LDS tile in between was a copy that bought nothing.

The fragment body reads the quants straight from global memory with the same addressing arithmetic and keeps only the scales in LDS: one sixteen-byte metadata header per token, copied in a single load per token per stage ahead of the barrier that publishes it. The request falls from `39,424 B` to `23,040 B`, which takes the body from three to five workgroups per WGP, and the stage loses its activation copy and one of its two barrier-separated halves.

Two ablations set the design before anything was written. Staging the activation once per block instead of per stage - the conservative bound, since it keeps every barrier - is worth `+5.2 %` to `+6.2 %` on four deployed rows, so the LDS round trip was worth attacking. Then, on the first fragment body, replacing the metadata loads with a constant is worth another `+7.6 %` to `+8.2 %`: the per-group scales are read inside the innermost loop and feed the accumulator scaling directly, so as cold global loads they are entirely exposed. That is why they, and only they, stay in LDS.

The deployed body reproduces the staged body byte for byte (verified on every screened key) and measures:

| rows x k | before, three workgroups | after, five workgroups | delta |
| --- | ---: | ---: | ---: |
| `32768 x 2560` (`N=12288`) | 29.90 TF | 30.58 TF | `+2.29 %` |
| `8192 x 2560` (`N=12288`) | 29.59 TF | 30.39 TF | `+2.69 %` |
| `2048 x 2560` (`N=12288`) | 29.22 TF | 30.58 TF | `+4.64 %` |
| `32768 x 2560` (`N=640`) | 28.76 TF | 30.44 TF | `+5.84 %` |

The producer is untouched: the workspace layout the fragment order needs is the layout it already writes, so the "rearrange the activation at quantization time" step that motivated the experiment turned out to be unnecessary for the multiply. The tail body (`K=640`, which needs a half stage loader) does not yet have a fragment form, so those keys keep the staged body.

The mechanism does not transfer to the 8-bit types as it stands. The same dot serves Q4_0, Q5_0, Q8_0, IQ4_NL and IQ4_XS, and Q8_0 was screened the same way:

| rows x k | staged, three workgroups | fragment, five workgroups | delta |
| --- | ---: | ---: | ---: |
| `32768 x 4096` (`N=1024`) | 29.68 TF | 27.47 TF | `-7.46 %` |
| `2048 x 4096` (`N=1024`) | 29.03 TF | 28.60 TF | `-1.49 %` |
| `32768 x 2048` (`N=4096`) | 29.58 TF | 29.57 TF | `-0.04 %` |

Output is byte-identical there too, so this is a memory-pattern effect. The staged body copies the activation with perfectly coalesced loads. The fragment body reads, per instruction, sixteen 32-byte groups that sit 144 bytes apart, which is sixteen cache lines where the tile copy touched one. For Q2_0 the decode-heavy body has enough slack to absorb that and the removed stage work wins. For an 8-bit type the decode is nearly free, the body is closer to its memory path, and the extra line touches cost more than the tile they replace.

The two halves of the idea are separable, and that is why this record keeps only the Q2_0 half: the multiply path stays on the workspace layout the producer already writes, and the producer was not rearranged at all. Moving the other types onto it in future would mean writing the quants as `[32-value group][token][32 B]`, which puts a warp's sixteen groups in one contiguous 512-byte span - strictly better than the copy it replaces, and what would also make the metadata strip's copy coalesced. That is a producer-and-workspace change with its own contract, worth considering for other quant types in future rather than for this deployment.

### Closed mechanisms

Global J64, the wide `I=128` tile, activation-half double buffering of the weight tile, decoded-weight LDS caching, split-K, persistent workgroups, and the hoisted epilogue are closed for this family of shapes on the sibling types: the wide tile lost by `1.05-1.21x` on Q3_K, Q4_K and Q5_K, the hoisted epilogue was neutral for Q3_K, Q4_K, Q5_K and Q8_0, and the remaining decode-side cost is the int32-to-fp32 conversion and the scale multiply of the epilogue, which every type in the family pays.

## Resources

The exact K2560 bodies use `212 VGPR / 29 SGPR / 39,424 B LDS` with the level table and `38,400 B` without it. The K=640 tail body uses `212 VGPR / 29 SGPR / 38,400 B LDS`. All are spill-free.

## Evidence

```text
~/tmp/torch-ggml-ops/qwen4_fwd/sweep_q20_v1.txt
~/tmp/torch-ggml-ops/qwen4_fwd/sweep_q20_v2.txt
```

The Qwen4-Exp shapes have HIP controls but no GGTensile problem key yet, so they cannot be selected as deployment cases. Their runs use the deployed Q8_1 producer and the same prepared inputs, the same `torch.mm` BF16 baseline on the same activation tensor, and the same paired timing as the official protocol.
