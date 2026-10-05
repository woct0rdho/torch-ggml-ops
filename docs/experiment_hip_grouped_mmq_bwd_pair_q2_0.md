# HIP Grouped MMQ Backward Pair Q2_0 Experiment

## Scope

This record covers the routed Q2_0 gate/up input-gradient pair for Qwen4-Exp on gfx1151, the pair half of the Q2_0 backward family.

The model is `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing), whose every routed-expert tensor is Q2_0 and whose gate and up projections are separate per-expert tensors. The pair shares its routed rows with the single-projection records, so one fitted law drives all of them.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/bf16 GMM | Kernel |
| ---: | --- | ---: | ---: | --- |
| 1 | `2 x (20480,640,2560)` | 13.34 | 2.08x | `grouped_bwd_pair_task_q2_0_n640_k2560_mt128_nt64_s2_skip` |
| 4 | `2 x (81920,640,2560)` | 22.78 | 1.92x | `grouped_bwd_pair_task_q2_0_n640_k2560_mt128_nt64_s2_skip` |
| 16 | `2 x (327680,640,2560)` | 29.30 | 1.80x | `grouped_bwd_pair_task_q2_0_n640_k2560_mt128_nt64_s2_skip` |

Banks come from the `qwen3.8-learned` law, one profile per size at the declared seed (`352 / 458 / 497` active experts, largest group `1261 / 4735 / 17896` rows). The baseline is the same routed product computed in BF16 over predecoded expert weights, two grouped products per expert plus the add, with `torch.mm`: it is what an unquantized multiply-only grouped kernel reaches on this part, in the role the AITER GMM number plays in the plan. The quantized kernel is `1.80x` to `2.08x` ahead of it, and it is level with the Q3_K pair at B16 (`29.30` against `29.48` TFLOPS) while carrying the cheaper decode.

## Why the body looks like this

The body is the paired staged task body the family already deploys: one M128 task per expert row tile, an N64 x M128 tile per workgroup, two weight tiles per contraction stage (one per projection, summed into the same accumulators), two LDS stages, one plain barrier per stage, and inactive-wave suppression. Only the decode is new: `QK2_0 = 64` gives forty 18-byte blocks per weight row, one fp16 scale per block and four consecutive two-bit codes per payload byte, so a sixteen-value segment is one unaligned 32-bit word.

## Optimization log

### Swizzle chunk

The tile's swizzle decides which LDS bank each decoded value lands in. A thread writes sixteen *consecutive* weight columns for one packed row, so those sixteen writes sit a fixed stride apart and collide unless the swizzle spreads them. Paired and order-balanced on the same banks, two passes each:

| Variant against chunk 4 | B1 | B16 |
| --- | ---: | ---: |
| chunk 8 | `-0.3 % / -0.3 %` | `-1.2 % / -1.2 %` |
| chunk 0 (no swizzle) | `-7.7 % / -7.9 %` | `-21.0 % / -21.2 %` |
| chunk 4 with eight columns of padding | `-2.0 %` | `-1.6 %` |

Chunk 4 is the shipped one. Two further chunks were built and reject themselves: chunks 2 and 1 violate the fragment reader's assumption that a swizzle chunk holds a whole vector, and they produce scrambled output rather than a slower kernel, which is worth remembering because the tile's only guard is that the chunk count stays a power of two.

### Closed and deferred mechanisms

- Two LDS stages are what the paired family deploys and what this shape wants. The single-projection sibling's three-stage body was not carried over.
- The packed row width is the type's own geometry: a Q2_0 weight row of 2560 values is forty 64-value blocks, not the ten 256-value blocks the forward ABI's `blocks_per_weight_row` counts. The first build used the forward count and produced an oracle error of `1.45` relative. The bytes per expert follow the same arithmetic (`640 x 40 x 18`).
- Wider N ownership, the non-staged pair bodies, and the sibling decode items are closed by the sibling pair records and do not apply here.
- Deployment wiring. The body is catalogued as `grouped_bwd_pair_task_q2_0_n640_k2560_mt128_nt64_s2_skip` and builds into the HIP control bundle, but it carries no routed rule yet: the routed table is asserted to mirror the public route set exactly, and the public pair-backward routes stop at Q3_K, IQ2_S and IQ2_XXS.

## Evidence

The campaign lives in `~/tmp/torch-ggml-ops/grouped_bwd_pair_q2_0/`: `model.py` slices one expert out of the checkpoint for either projection and unpacks Q2_0, `routes.py` samples the fitted law into a route bank, and `bench_pair_bwd.py` launches the deployed control through the production task bank and launcher, checks it against the BF16 grouped product of both projections and times it against that baseline.
