# HIP Grouped MMQ Forward Pair Q2_0 Experiment

## Scope

This record covers the routed Q2_0 gate/up forward kernel for Qwen4-Exp on gfx1151, the pair half of the Q2_0 routed family.

The model is `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing). Every routed-expert tensor of the checkpoint is Q2_0, the gate and up projections are separate per-expert tensors, and the pair shares its routed rows with the down projection, so one fitted law drives all three.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/bf16 GMM | Kernel |
| ---: | --- | ---: | ---: | --- |
| 1 | `2 x (20480,640,2560)` | 18.05 | 3.68x | `grouped_fwd_serial_q2_0_n640_k2560_j64_j32_j16_frag` |
| 4 | `2 x (81920,640,2560)` | 22.74 | 2.68x | `grouped_fwd_serial_q2_0_n640_k2560_j64_j32_j16_frag` |
| 16 | `2 x (327680,640,2560)` | 24.60 | 2.16x | `grouped_fwd_serial_q2_0_n640_k2560_j128_frag` |

Banks come from the `qwen3.8-learned` law, one profile per size at the declared seed (`352 / 458 / 497` active experts, largest group `1261 / 4735 / 17896` rows). The baseline is the same routed product computed in BF16 over predecoded expert weights, one grouped product per projection with `torch.mm`: it is what an unquantized multiply-only grouped kernel reaches on this part, in the role the AITER GMM number plays in the plan.

The quantized kernel is `2.16x` to `3.68x` ahead of it. One pair launch carries the same work as one routed down launch, and the two records land at the same rate at B16. Both sit below the int8/bf16 WMMA roofline rather than against it.

## Why the body looks like this

The pair operation runs the routed single-projection body once per projection over the same route bank and the same activation workspace, so the body is the one the routed down record deploys and the two launches differ only in which packed bank and which output they are given. A fused body that takes both packed banks in one launch exists in the GGTensile family, not here.

## Optimization log

### Row tile

Ten 256-value stages amortize a row tile well, so the padding the prior's small groups cost is the whole of the tile-height trade: at B1 a J64 tile pays `1.82x` of its rows as padding, at B16 `1.045x`. The mixed J32 tail gives a group whose remainder is at most thirty-two rows a J32 tile, which is what removes that padding at the small route, and the wide tile needs the fragment reads to pay at all: the staged J128 tile loses fourteen percent at B16 while the fragment one gains four, because the staged form's `38,400 B` request leaves three workgroups where the fragment form's `22,016 B` leaves five. Against the J64 staged body, both orders, eight repeats per block:

| Candidate | B1 | B4 | B16 |
| --- | ---: | ---: | ---: |
| J64 fragment | `+1.8 % / +1.9 %` | `+2.5 % / -0.9 %` | `-0.0 % / +0.5 %` |
| J64 fragment + J32 tails | `+21.6 % / +21.8 %` | `+10.0 % / +10.2 %` | `+0.0 % / -0.0 %` |
| J64 fragment + J32 + J16 tails | `+23.6 % / +24.0 %` | `+9.4 % / +9.6 %` | `+0.3 % / +2.8 %` |
| J80 fragment | `-5.8 % / -7.7 %` | `+1.6 % / +1.8 %` | `+3.3 % / +3.3 %` |
| J128 staged | `-81.6 % / -80.2 %` | `-38.8 % / -38.8 %` | `-14.4 % / -14.7 %` |
| J128 fragment | `-35.0 % / -34.5 %` | `-9.1 % / -8.9 %` | `+3.8 % / +3.8 %` |
| J32 fragment | `+12.6 % / +13.0 %` | `-6.5 % / -6.3 %` | `-19.7 % / -19.2 %` |

The B1 gain is padding arithmetic and nothing else: the same body without the J32 tail is worth two percent there, and the J16 stage below it another two (`+2.0 % / +2.3 %` paired against the J32 body), at a measured cost of half a percent on the B4 route where almost no group is that small. The wide tile needs both of its conditions - the fragment reads and the large route - and the staged wide tile is a fourteen percent loss at B16, which is the same split the dense Q2_0 record measured. A global J32 tile wins at B1 and loses everywhere else, so the tile height follows the route rather than the batch, and the two deployed bodies split at the largest route.

### Closed and deferred mechanisms

- The fused single-launch pair body (two packed banks, two outputs) is the GGTensile family's shape. The HIP pair operation is two launches of one single-projection body, which is what the sibling HIP pair records deploy as well.
- The wide tile (`MMQ_I 128` with a 256-thread workgroup) is closed for this shape: it halves the activation passes but also halves the independent workgroups per WGP, and it measures `-13 % / -7 % / +0.9 %` at B1/B4/B16, with the tail variants a further nine points behind at B16.
- The shipped body is the J16 variant: it wins on the training route and gives up `0.5 %` at B4, which the family accepts for a two-and-a-half percent B1 gain.
- J80 is dominated rather than rejected: it is `3.3 %` ahead of the J64 staged body at B16 but six to eight percent behind at B1, and the J64+J32 / J128 split beats it at both ends.
- Deployment wiring. Both bodies are catalogued and build into the HIP control bundle, but they carry no routed rule yet: the routed table is asserted to mirror the public route set exactly, and the public pair routes stop at IQ2_S, IQ2_XXS and Q3_K, so the family cannot be registered until a Q2_0 pair route exists.

## Evidence

The campaign lives in `~/tmp/torch-ggml-ops/grouped_fwd_pair_q2_0/`: `model.py` slices one expert out of the checkpoint for either projection and unpacks Q2_0, `routes.py` samples the fitted law into a route bank, `launcher.py` binds a rendered control to the grouped forward ABI, `sweep_pair.py` builds and times the tile and mechanism candidates, and `bench_pair.py` measures the installed controls through the deployment geometry against the BF16 baseline.
