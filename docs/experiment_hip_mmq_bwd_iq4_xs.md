# HIP MMQ Backward IQ4_XS Experiment

## Scope

This record covers the gfx1151 HIP input-gradient backward kernels for IQ4_XS weights on the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing). The type carries the QSA query projection of 4 of the 12 attention layers, a few key, value and shared-expert tensors, and both GatedDeltaNet projections that are in scope, 77.5 MiB of packed weights. The 48 layers mix recipes, and a projection family whose type has no body falls back to a dequantizing multiply.

The type is the only one of the dense backward family with a 256-wide block *and* sub-block scales. `QK_K = 256` weights share a 136-byte block: an fp16 base scale, a 16-bit high plane and a 4-byte low plane carrying eight 6-bit signed sub-block scales, then a 128-byte payload. Each 32-value sub-block has its own scale, and within it the sixteen payload bytes carry both nibble planes: byte `j` holds value `j` in its low nibble and value `j + 16` in its high nibble. Every nibble indexes the same 16-entry level table IQ4_NL uses, and a value is `d * (scale6 - 32) * level`.

## Final kernel result

| Family | `(M,K,N)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| QSA query | `(2048,2560,12288)` | 28.708 | 1.232x | `dense_bwd_iq4_xs_pipea_nt4_ki64_mw2_sw16_prefetch` |
| QSA query | `(8192,2560,12288)` | 27.220 | 1.077x | `dense_bwd_iq4_xs_pipea_nt4_ki64_mw2_sw16_prefetch` |
| QSA query | `(32768,2560,12288)` | 24.757 | 0.958x | `dense_bwd_iq4_xs_pipea_nt4_ki64_mw2_sw16_prefetch` |
| QSA key/value | `(2048,2560,512)` | 29.179 | 1.412x | `dense_bwd_iq4_xs_pipea_nt4_ki64_mw2_sw16_prefetch` |
| QSA key/value | `(8192,2560,512)` | 29.633 | 1.153x | `dense_bwd_iq4_xs_pipea_nt4_ki64_mw2_sw16_prefetch` |
| QSA key/value | `(32768,2560,512)` | 32.212 | 1.220x | `dense_bwd_iq4_xs_pipea_nt4_ki64_mw4_sw16_prefetch` |
| Shared-expert gate/up | `(2048,2560,640)` | 30.483 | 1.427x | `dense_bwd_iq4_xs_pipea_nt4_ki64_mw2_sw16_prefetch` |
| Shared-expert gate/up | `(8192,2560,640)` | 34.263 | 1.258x | `dense_bwd_iq4_xs_pipea_nt4_ki64_mw4_sw16_prefetch` |
| Shared-expert gate/up | `(32768,2560,640)` | 36.135 | 1.295x | `dense_bwd_iq4_xs_pipea_nt4_ki64_mw4_sw16_prefetch` |
| GatedDeltaNet `in_proj_qkv` | `(2048,2560,10240)` | 29.094 | 1.258x | `dense_bwd_iq4_xs_pipea_nt4_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet `in_proj_qkv` | `(8192,2560,10240)` | 27.652 | 1.094x | `dense_bwd_iq4_xs_pipea_nt4_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet `in_proj_qkv` | `(32768,2560,10240)` | 24.574 | 0.991x | `dense_bwd_iq4_xs_pipea_nt4_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet `in_proj_z` | `(2048,2560,6144)` | 29.241 | 1.290x | `dense_bwd_iq4_xs_pipea_nt4_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet `in_proj_z` | `(8192,2560,6144)` | 28.756 | 1.138x | `dense_bwd_iq4_xs_pipea_nt4_ki64_mw2_sw16_prefetch` |
| GatedDeltaNet `in_proj_z` | `(32768,2560,6144)` | 27.406 | 1.066x | `dense_bwd_iq4_xs_pipea_nt4_ki64_mw2_sw16_prefetch` |


## Kernel implementation

The body is the reusable pipelined one (`csrc/ck/mmq_backward_pipelined.cuh`), as on every other dense backward record: 128 threads, `n_tiles * 16` result columns, `KITERATION` contraction values per stage, two shared tiles with one barrier per stage, and register double buffering of the activation fragments. This is the first type to carry the K-quant tile class *and* the IQ4 codebook: the 256-wide block gives the K-quant geometry, while the payload and its lookup are IQ4_NL's.

The decode reads the sub-block scale from the two metadata planes, takes the payload byte of its plane, and resolves the nibble through the same permute lookup the forward records deploy. The deployed tile uses the prefetched payload form, where one `uint4` is the whole 16-byte payload of a sub-block, so each 32-bit word feeds one permute pair and the plane selects the half of the pair: sixteen values cost four lookups.

Two details of the layout are worth recording because the first implementation got them wrong: within a sub-block the sixteen payload bytes are *not* split into a low-byte and a high-byte range. The ggml reference reads byte `j` for value `j` and byte `j+16` for value `j+16`, so the two planes share the same sixteen bytes and differ only in which nibble is read. Reading the plane as a byte range rather than a nibble selector is the mistake to avoid here.

## Resources

The deployed bodies use `147 VGPR` at `22 SGPR` (two-row-tile) and `212 VGPR` at `22 SGPR` (four-row-tile), both at `16 KiB` LDS (two 64x64 bf16 tiles), spill-free.

## Optimization log

### Candidate screen

Six candidates were screened on all five families: the pipelined tile with and without the activation prefetch, its prefetched-decode twin, a four-row-tile variant, a padded two-row-tile variant, and a 32-value-stage variant. The narrow-result families behave like the Q4_0, Q5_0 and IQ4_NL records - the two-row-tile tile wins at the small row counts - but this type is the first of the group whose result is wide enough (`12288`, `10240`, `6144`) for the four-row-tile variant to be the wrong shape: there it loses `1.07-1.33x` on every row count, because the wider tile halves the workgroup count on a grid that is already the binding constraint at `M=2048` and buys nothing back at the larger row counts. The 32-value stage loses `1.06-1.16x` everywhere, and padding loses to the swizzle-only 8 KiB tile (`1.01-1.07x`).

The two exceptions are the narrow-result families, where the decode is amortized over fewer columns and the larger tile pays: the shared-expert gate/up rows take the four-row-tile body at `M=8192` and `32768` (`1.10x` and `1.17x` over the two-row-tile body), and the QSA key/value rows take it at `M=32768` (`1.05x`). Those are the only four-row-tile keys in the deployment, and the table above names them.

### Where this type sits

With the codebook and the sub-block scales this is the most decode-heavy payload of the dense family, and it shows at the largest row count: the two widest families measure `0.958x` and `0.991x` against the BF16 baseline at `M=32768`, the only dense backward rows below it since the Q2_0 record. Everywhere else it is ahead, `1.07-1.43x`. The type is nonetheless the strongest of the four cheap-payload records on the narrow shapes - `36.1 TFLOPS` on `(M,2560,640)` against Q4_0's `31.1`, Q5_0's `29.7` and IQ4_NL's `30.4` at the same width - which is the opposite of what its decode cost suggests and points at the block width rather than the payload: a 256-wide block amortizes its metadata over eight times the values that a 32-wide block does.

## Evidence

```text
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_iq4xs_v3.txt    (candidate screen)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_iq4xs_doc.txt   (confirmation)
```
