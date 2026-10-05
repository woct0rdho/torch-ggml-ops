# HIP Grouped MMQ Backward Q2_0 Experiment

## Scope

This record covers the routed Q2_0 down input-gradient kernel for Qwen4-Exp on gfx1151.

The model is `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing). Every routed-expert tensor of the checkpoint is Q2_0, the gradient contracts the down projection's output back to the expert intermediate size, and the training path calls `R=20480`.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/bf16 GMM | Kernel |
| ---: | --- | ---: | ---: | --- |
| 1 | `(20480,640,2560)` | 13.29 | 2.37x | `grouped_bwd_row_task_q2_0_n2560_k640_mt128_nt64_s3_sw8` |
| 4 | `(81920,640,2560)` | 23.07 | 2.47x | `grouped_bwd_row_task_q2_0_n2560_k640_mt128_nt64_s3_sw8` |
| 16 | `(327680,640,2560)` | 29.82 | 2.33x | `grouped_bwd_row_task_q2_0_n2560_k640_mt128_nt64_s3_sw8` |

Banks come from the `qwen3.8-learned` law, one profile per size at the declared seed (`352 / 458 / 497` active experts, largest group `1261 / 4735 / 17896` rows). The baseline is the same routed product computed in BF16 over predecoded expert weights, expert by expert, with `torch.mm`: it is what an unquantized multiply-only grouped kernel reaches on this part, in the role the AITER GMM number plays in the plan. The quantized kernel is `2.33x` to `2.47x` ahead of it, and it is above every routed sibling at every shape (`Q4_K` `9.63 / 16.60 / 20.94`, `Q5_K` `10.22 / 16.36 / 20.73`, `IQ2_S` `11.21 / 17.49 / 21.05`, `Q2_K` `11.97 / 19.67 / 22.57` TFLOPS), which is what the smallest decode in the bundle should buy.

## Why the body looks like this

The body is the staged row-task one the routed backward family already deploys: one M128 task per expert row tile from the device task bank, a 32-wide contraction stage, three LDS buffers, one plain barrier per stage, and waves whose first row is past the task end skipping their consumer work. Only the decode is new, and it is the smallest one in the bundle: `QK2_0 = 64` gives ten 18-byte blocks per weight row, one fp16 scale per block and four consecutive two-bit codes per payload byte, so a sixteen-value segment is one unaligned 32-bit word and a value costs a shift, a mask, a subtract and one multiply. None of the sibling decode machinery - grid lookups, sign tables, high-bit reconstruction, sub-block scales - exists here.

## Optimization log

### Swizzle chunk

The staged tile's swizzle decides which LDS bank each decoded value lands in. A thread writes sixteen *consecutive* weight columns for one of the eight packed rows it owns, so a chunk of sixteen pairs the four writes of a row with itself and the reads collide. Halving the chunk splits the row into four and spreads both. Paired, order-balanced, two passes on the same banks:

| Variant | B1 | B4 | B16 |
| --- | ---: | ---: | ---: |
| chunk 8 against chunk 16 | `+9.0 % / +9.0 %` | `+16.4 % / +16.4 %` | `+20.4 % / +20.4 %` |
| chunk 4 against chunk 8 | `-2.8 % / -2.7 %` | `-3.6 % / -3.7 %` | `-4.2 % / -4.1 %` |

The chunk-8 body is the shipped one and its outputs are byte-identical to the chunk-4 body, which is the cheapest confirmation that the change is a bank-layout change and not an arithmetic one.

### Closed and deferred mechanisms

- Stage count: two LDS stages are level with three (paired `+0.04 % / +0.00 % / +0.05 %`), so the family keeps the three-stage skeleton it inherited.
- Inactive-wave suppression: skipping the consumer work of waves whose first row is past the task end is worth `+44 % / +17 % / +2 %` at B1/B4/B16 against the same body without it, which is the opposite of what the Q4_K and Q5_K records measured for their decodes. It stays on for this type.
- The task bank cap: the routed family's builder scanned 256 route entries per pass, which silently produced an empty bank for this model's 352 to 497 active experts. The scan now runs once per chunk of 256 entries and carries the finished count as its base, which is what makes the 512-expert contract work at all.
- Wider N ownership, the first-generation non-staged tile, and the sibling decode items are closed by the sibling records and do not apply here.
- Deployment wiring. The body is catalogued as `grouped_bwd_row_task_q2_0_n2560_k640_mt128_nt64_s3_sw8` and builds into the HIP control bundle, but it carries no routed rule yet: the routed table is asserted to mirror the public route set exactly, and the public backward routes stop at Q2_K, Q4_K, Q5_K and IQ2_S, so the family cannot be registered until a Q2_0 route exists.

## Evidence

The campaign lives in `~/tmp/torch-ggml-ops/grouped_bwd_q2_0/`: `model.py` slices one expert out of the checkpoint and unpacks Q2_0, `routes.py` samples the fitted law into a route bank, and `bench_bwd.py` launches the deployed control through the production task bank and launcher, checks it against the BF16 grouped product and times it against that baseline.
