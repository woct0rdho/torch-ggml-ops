# HIP Grouped MMQ Forward Q2_0 Experiment

## Scope

This record covers the routed Q2_0 down forward kernel for Qwen4-Exp on gfx1151, the first grouped body for this quant type.

The model is `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing). Every routed-expert tensor of the checkpoint is Q2_0.

## Final kernel result

| Batch | Logical shape | Aggregate rows | Experts | HIP TFLOPS | HIP/bf16 GMM | Kernel |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `(20480,2560,640)` | 20480 | 352 | 12.002 | 2.08x | `grouped_fwd_serial_q2_0_n2560_k640_j64` |
| 4 | `(81920,2560,640)` | 81920 | 458 | 19.153 | 1.83x | `grouped_fwd_serial_q2_0_n2560_k640_j64` |
| 16 | `(327680,2560,640)` | 327680 | 497 | 23.703 | 1.60x | `grouped_fwd_serial_q2_0_n2560_k640_j64` |

Banks come from the `qwen3.8-learned` law, one profile per size at the declared seed (`352 / 458 / 497` active experts, largest group `1261 / 4735 / 17896` rows). The baseline is the same routed product computed in BF16 over predecoded expert weights, expert by expert, with `torch.mm`: it is what an unquantized multiply-only grouped kernel reaches on this part, in the role the AITER GMM number plays in the plan. The quantized kernel is `1.60x` to `2.08x` ahead of it.

## Why the body looks like this

`QK2_0 = 64` makes `K = 640` two 256-value stages plus a 128-value tail, which the K-quant grouped bodies could not express, so the grouped serial body gained the dense body's contraction tail and its decode now runs through the same 18-byte-block loader the dense Q2_0 records use.

`J = 64` is the deployed tile, the same choice the routed IQ2_S and Q4_K records made. The sweep below shows the balance.

## Optimization log

### Tile height

The prior's padding inflation is the whole cost structure of this family: at B1 the groups are so small that most of a tile is empty rows. Pairing each tile height against `J = 128` on the same bank, both orders, six repeats per block:

| Batch | J=128 | J=64 | J=32 | trade |
| ---: | ---: | ---: | ---: | --- |
| 1 | 6.59 TF | 11.93 TF | 11.23 TF | `1.81x` |
| 4 | 11.30 TF | 18.85 TF | 14.10 TF | `1.67x` |
| 16 | 15.20 TF | 23.64 TF | 15.00 TF | `1.56x` |

The padding ratio alone would predict `3.04x / 1.50x / 1.09x` at J=128 against the padded row counts, so the measured gains are smaller than the saved arithmetic: each row tile re-reads the expert's packed rows, so shrinking the tile raises the weight traffic per output row. `J = 32` overshoots that trade at every batch. `J = 64` is the optimum of the three and the deployed height.

### Closed and deferred mechanisms

- Mixed J32 tails (`mixed_j32_tails`, which routes groups under 32 rows to a J=32 tile) is closed for this type: the Q2_0 instantiation faults with an illegal memory access on a single 20-row group, and its apparent `+8%` was measured before that fault was noticed. The mechanism stays enabled where it is qualified (IQ2_S, Q4_K J64 K512).
- Activation prefetch (`MMQ_PREFETCH_ACT`, the grouped register staging of the second activation plane) is neutral here: `+0.5%`, `+0.6%`, `+4.8%` at B1/B4/B16 against J=64, all within the `+/- 2.6%` control band except the largest, which is inside the two repeated readings' spread of each other.
- The level table (`table_decode`) is a dense mechanism in this bundle. The grouped renderer does not carry it, and the dense Q2_0 record already measured its worth at `1.2-2.4%`. Threading it through the grouped body is the cheap next decode-side experiment.
- The weight pipeline (`MMQ_WEIGHT_PIPELINE`) is an IQ2_XXS payload mechanism and does not apply to an 18-byte block.
- Row-task ownership is the lever the padding analysis points at, and it is the one the sibling records name for small-route families: a static grid issues `sum(ceil(M_g/J))` tiles, a row-task bank re-buckets the same rows into full tiles so the empty rows disappear. It is not implemented for this family. A second reason to expect it to pay here is imbalance: the static grid gives one block per route entry, so at B16 a block whose group holds `17896` rows runs `280` row tiles while thousands of blocks with a few hundred rows finish immediately, and the tail of the wave is one long group.
- Deployment wiring. The body is catalogued as `grouped_fwd_serial_q2_0_n2560_k640_j64` and builds into the HIP control bundle, but it carries no routed rule yet: the routed table is asserted to mirror the public route set exactly, and no Q2_0 grouped-forward route exists in the public bundle, so the family cannot be registered until that route does. Every number above comes from the built control, launched through the grouped forward ABI with the route bank the law produced.

## Evidence

The campaign lives in `~/tmp/torch-ggml-ops/grouped_fwd_q2_0/`: `model.py` slices one expert out of the checkpoint and unpacks Q2_0, `routes.py` samples the fitted law into a route bank, `check_expert.py` checks one expert at any group size, `bench_bank.py` measures a full bank with the BF16 baseline, and `sweep_j.py` / `sweep_v.py` run the tile swarms.
