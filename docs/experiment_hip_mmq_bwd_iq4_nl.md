# HIP MMQ Backward IQ4_NL Experiment

## Scope

This record covers the gfx1151 HIP input-gradient backward kernels for IQ4_NL weights. The type appears only in the shared-expert down projection of 7 of the 48 layers of the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, hidden size 2560), 6.2 MiB of packed weights, the smallest dense footprint of the group, so the record has one dense family: a `(2560,640)` weight, backward `(M,640,2560)` over the B1/B4/B16 token counts of a sequence length 2048 batch. The 48 layers mix recipes, and a projection family whose type has no body falls back to a dequantizing multiply. The checkpoint's `per_layer_token_embd.weight` is also this type, but it is an embedding gather rather than a multiply and stays on the GGUF embedding module.

`QK4_NL = 32` weights share an 18-byte block: an fp16 scale and Q4_0's nibble planes, where byte `j` holds weight `j` in the low nibble and weight `j + 16` in the high nibble. The nibbles index a shared 16-entry signed level table (`kvalues_iq4nl`: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113) instead of being offset arithmetic, so the payload is the same size as Q4_0's and the decode adds one small-table lookup per value.

## Final kernel result

| Family | `(M,K,N)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| Shared-expert down | `(2048,640,2560)` | 17.788 | 1.092x | `dense_bwd_iq4_nl_pipea_nt4_ki64_mw2_sw16_prefetch` |
| Shared-expert down | `(8192,640,2560)` | 28.288 | 1.707x | `dense_bwd_iq4_nl_pipea_nt4_ki64_mw2_sw16_prefetch` |
| Shared-expert down | `(32768,640,2560)` | 30.411 | 1.689x | `dense_bwd_iq4_nl_pipea_nt4_ki64_mw2_sw16_prefetch` |

Shapes are written `(M, in_features, out_features)`, matching the weight's `(N,K) = (out_features, in_features)` in the table's `(M,K,N)` column. Every row comes from one run at eight repeats and two blocks against a BF16 `torch.mm` baseline on the same prepared gradient and packed weights.

## Kernel implementation

The body is the reusable pipelined one (`csrc/ck/mmq_backward_pipelined.cuh`), the body the Q2_0, Q4_0, Q5_0, Q3_K, Q4_K, Q5_K and Q6_K records use: 128 threads, `n_tiles * 16` result columns, `KITERATION` contraction values per stage, two shared tiles with one barrier per stage, and register double buffering of the activation fragments.

The codebook lookup is the same permute form the forward records deployed: three byte permutes select the low-half entry and three more select the high-half entry, and the fourth index bit picks between the halves, so eight indices resolve without a memory gather. The register form was moved out of the forward header into the shared decoder header so both directions use one definition; the forward call sites are unchanged apart from qualifying the name.

The deployed tile takes the prefetched payload form: one `uint4` covers the block's whole 16-byte payload, and each 32-bit word of it feeds both a low-nibble and a high-nibble lookup at once, so sixteen values cost four permute pairs. It measured `1.06x` ahead of the group form at `M=8192` and level elsewhere, which is why it deploys.

## Resources

The deployed body uses `143 VGPR`, `23 SGPR` and `16 KiB` LDS (two 64x64 bf16 tiles), spill-free.

## Optimization log

### Candidate screen

The Q4_0 and Q5_0 records' six candidates were rendered for this type and screened at four repeats on all three row counts, then the three that survived were confirmed at eight repeats. The verdicts repeat for the same reasons: the two-row-tile pipelined tile wins every row count (`1.12-1.54x` over the four-row-tile variants and `1.18-1.33x` over the 32-value stage), and padding loses to the swizzle-only 8 KiB tile. Among the three the prefetched payload read is worth `1.06x` at `M=8192` and level at the other two row counts, so it deploys. The differences at the outer row counts are inside the spread of a shape whose grid is only `16 x 10` workgroups at `M=2048`.

No level-table mechanism is needed. The forward records' table trick targets Q2_0's byte-spread arithmetic and Q5_0's fifth-bit merge, and here the lookup is already the cheap form: the Q4_0 record, whose payload needs two shifts and two masks instead, measures the same band (`18.3/28.1/31.1`) as this one, so the codebook is not a visible cost.

### Where this type sits

`17.8/28.3/30.4 TFLOPS` is above the BF16 baseline on every row and sits with its offset-arithmetic siblings: Q4_0 measures `18.3/28.1/31.1` and Q5_0 `18.5/27.9/29.7` on the same keys, so the three cheapest payloads of the bundle are within a few percent of each other and the decode is not what separates them. The K-quant families at the equivalent width are ahead (Q5_K `30.4/30.0/33.9`, Q6_K `36.6/37.7/38.0` on `(M,2560,640)`), and the limiter is the 640-wide result rather than the payload: at `M=2048` this family's grid is `16 x 10` workgroups, which is why the larger-tile variants lose and why the smallest row count is the weakest.

## Evidence

```text
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_iq4nl_v1.txt   (candidate screen)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_iq4nl_v2.txt   (eight-repeat confirmation)
```
