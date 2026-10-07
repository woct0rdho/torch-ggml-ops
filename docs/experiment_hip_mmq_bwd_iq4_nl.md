# HIP MMQ Backward IQ4_NL Experiment

## Scope

This record covers the gfx1151 HIP input-gradient backward kernels for IQ4_NL weights. The type appears only in the shared-expert down projection of 7 of the 48 layers of the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, hidden size 2560), 6.2 MiB of packed weights, the smallest dense footprint of the group, so the record has one dense family: a `(2560,640)` weight, backward `(M,2560,640)` over the B1/B4/B16 token counts of a sequence length 2048 batch. The 48 layers mix recipes, and a projection family whose type has no body falls back to a dequantizing multiply. The checkpoint's `per_layer_token_embd.weight` is also this type, but it is an embedding gather rather than a multiply and stays on the GGUF embedding module.

`QK4_NL = 32` weights share an 18-byte block: an fp16 scale and Q4_0's nibble planes, where byte `j` holds weight `j` in the low nibble and weight `j + 16` in the high nibble. The nibbles index a shared 16-entry signed level table (`kvalues_iq4nl`: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113) instead of being offset arithmetic, so the payload is the same size as Q4_0's and the decode adds one small-table lookup per value.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| Shared-expert down | `(2048,2560,640)` | 23.846 | 1.459x | `dense_bwd_iq4_nl_pipea_nt64_ki64_mw2_sw16_prefetch_g0` |
| Shared-expert down | `(8192,2560,640)` | 29.444 | 1.786x | `dense_bwd_iq4_nl_pipea_nt64_ki64_mw2_pad8_prefetch` |
| Shared-expert down | `(32768,2560,640)` | 30.546 | 1.778x | `dense_bwd_iq4_nl_pipea_nt64_ki64_mw2_pad8_prefetch` |

## Kernel implementation

The body is the reusable pipelined one (`csrc/ck/mmq_backward_pipelined.cuh`), the body the Q2_0, Q4_0, Q5_0, Q3_K, Q4_K, Q5_K and Q6_K records use: 128 threads, `n_tiles * 16` result columns, `KITERATION` contraction values per stage, two shared tiles with one barrier per stage, and register double buffering of the activation fragments.

The codebook lookup is the same permute form the forward records deployed: three byte permutes select the low-half entry and three more select the high-half entry, and the fourth index bit picks between the halves, so eight indices resolve without a memory gather. The register form was moved out of the forward header into the shared decoder header so both directions use one definition. The forward call sites are unchanged apart from qualifying the name.

The deployed tile takes the prefetched payload form: one `uint4` covers the block's whole 16-byte payload, and each 32-bit word of it feeds both a low-nibble and a high-nibble lookup at once, so sixteen values cost four permute pairs. It measured `1.06x` ahead of the group form at `M=8192` and level elsewhere, which is why it deploys.

## Resources

The deployed body uses `143 VGPR`, `23 SGPR` and `16 KiB` LDS (two 64x64 bf16 tiles), spill-free.

## Optimization log

### Candidate screen

The Q4_0 and Q5_0 records' six candidates were rendered for this type and screened on all three row counts. The verdicts repeat for the same reasons: the two-row-tile pipelined tile wins every row count (`1.12-1.54x` over the four-row-tile variants and `1.18-1.33x` over the 32-value stage), and padding loses to the swizzle-only 8 KiB tile. Among the three the prefetched payload read is worth `1.06x` at `M=8192` and level at the other two row counts, so it deploys. The differences at the outer row counts are inside the spread of a shape whose grid is only `16 x 10` workgroups at `M=2048`.

No level-table mechanism is needed. The forward records' table trick targets Q2_0's byte-spread arithmetic and Q5_0's fifth-bit merge, and here the lookup is already the cheap form: the Q4_0 record, whose payload needs two shifts and two masks instead, measures the same band (`18.3/28.1/31.1`) as this one, so the codebook is not a visible cost.

### Where this type sits

`23.8/28.3/30.4 TFLOPS` is above the BF16 baseline on every row and sits with its offset-arithmetic siblings: Q4_0 measures `24.0/28.1/31.1` and Q5_0 `23.7/27.9/29.7` on the same keys, so the three cheapest payloads of the bundle are within a few percent of each other and the decode is not what separates them. The `M=2048` key deploys the M-fastest traversal that the Q4_0 record describes (`1.31x` ahead of the deployed order under repeated launches). tThe single-row-tile and slice twins were screened with it and tie or lose there. The K-quant families at the equivalent width are ahead (Q5_K `30.4/30.0/33.9`, Q6_K `36.6/37.7/38.0` on `(M,2560,640)`), and the limiter is the 640-wide result rather than the payload: at `M=2048` this family's grid is `16 x 10` workgroups, which is why the larger-tile variants lose and why the smallest row count is the weakest.

### Padding on the pipelined tile

The pipelined tile's layout was chosen when the tile was first built - sixteen-value swizzle chunks, no padding - and the chunk was later swept on this body without revisiting padding. The layout is now screened over every deployed dense-backward key: the padding twin of each deployed body (eight values of row padding, no swizzle) is timed against it at the key's own row counts, four repeats per block, and every screen winner is then re-timed at eight repeats over two blocks in both measurement orders, so a key moves only when the padding body wins in both directions.

Padding is not a general replacement for the swizzle. It wins where the resident grid is thin and the result is narrow - the `M=2048` rows and the 512- and 640-wide results - and loses on the wide results at the largest row counts, where the decoded tile's shared-memory traffic is high enough that the swizzle's bank pattern still pays. The split is per key rather than per type: on this type only the `(M,2560,640)` shared-expert down row moves, `8.0%` at `M=8192` and `1.9%` at `M=32768`.

A second layout round then screened three further combinations on every officially measured body - sixteen-value padding, four-value padding, and eight-value padding with an eight-value swizzle chunk - at the deployed row counts. Nothing survives the confirmation: each candidate measures within `1-3%` of the deployed tile in one measurement order and loses in the other, so all three are rejected and none is built. The layout neighborhood is closed at the combinations the deployment uses.

## Evidence

```text
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_iq4nl_v1.txt            (candidate screen)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_iq4nl_v2.txt            (confirmation)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_reopen_gridshape.txt    (grouped-M, row-tile and slice screens)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_reopen_confirm.txt      (repeated-launch paired confirmation)
```
