# HIP Grouped MMQ Forward Q2_0 Experiment

## Scope

This record covers the routed Q2_0 down forward kernel for Qwen4-Exp on gfx1151, the first grouped body for this quant type.

The model is `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing). Every routed-expert tensor of the checkpoint is Q2_0.

## Final kernel result

| Batch | Logical shape | Aggregate rows | Experts | HIP TFLOPS | HIP/bf16 GMM | Kernel |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `(20480,2560,640)` | 20480 | 352 | 17.170 | 2.970x | `grouped_fwd_serial_q2_0_n2560_k640_j64_j32_j16_frag` |
| 4 | `(81920,2560,640)` | 81920 | 458 | 22.100 | 2.115x | `grouped_fwd_serial_q2_0_n2560_k640_j64_j32_j16_frag` |
| 16 | `(327680,2560,640)` | 327680 | 497 | 24.720 | 1.674x | `grouped_fwd_serial_q2_0_n2560_k640_j64_j32_j16_frag` |

Banks come from the `qwen3.8-learned` law, one profile per size at the declared seed (`352 / 458 / 497` active experts, largest group `1261 / 4735 / 17896` rows). The baseline is the same routed product computed in BF16 over predecoded expert weights, expert by expert, with `torch.mm`: it is what an unquantized multiply-only grouped kernel reaches on this part, in the role the AITER GMM number plays in the plan. The quantized kernel is `1.67x` to `2.97x` ahead of it.

## Why the body looks like this

`QK2_0 = 64` makes `K = 640` two 256-value stages plus a 128-value tail, which the K-quant grouped bodies could not express, so the grouped serial body gained the dense body's contraction tail and its decode now runs through the same 18-byte-block loader the dense Q2_0 records use.

The activation no longer passes through LDS: the body keeps one sixteen-byte metadata header per token and reads the payload fragments of the int8 WMMA straight from the workspace the producer already writes, which is what frees the fourth and fifth workgroups. A group whose remainder is at most thirty-two rows runs a J32 tile instead of a J64 one, and one whose remainder is at most sixteen rows runs a J16 tile.

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

### Mixed J32 tails

The serial body gives every group a J64 row tile, so an eleven-row group pays for sixty-four rows. The mixed tail routes a group whose remainder is at most thirty-two rows to a J32 tile, which against the padded row counts removes a quarter of the row-tile work at B1 and a tenth at B4.

The mechanism is the one the routed IQ2_S and Q4_K records deploy, and for this type it was broken rather than slow: its J32 instantiation was the one call site that still dropped the contraction tail, so for `K = 640` it loaded three 256-value stages out of a 640-value row and faulted on the last expert. Threading `k_tail_values` through it fixes that, and the qualified shape is now this family's grouped K640 geometry.

Paired and order-balanced on the same banks, two passes each:

| Basis | B1 | B4 | B16 |
| --- | ---: | ---: | ---: |
| staged | `+9.7 % / +10.5 %` | `+5.4 % / +5.1 %` | `+1.1 % / +1.1 %` |
| fragment | `+3.9 % / +4.1 %` | `+2.7 % / +0.8 %` | `-0.75 % / -0.73 %` |

The B16 reading is a small real cost: a J32 tile doubles the packed-weight traffic of every row it covers, and the second instantiation widens the kernel. It is inside the control band while a quarter of the small-route work is not, so this family ships one body, and the B16 row of the table is `0.7 %` below what the fragment body alone measures there.

### J16 tail

The J32 tail left the smallest groups paying for thirty-two rows, and at B1 the prior's median group is eleven rows. A J16 stage below it removes another ten percent of the padded work at B1 (`1.36x` to `1.29x` of the real rows, against `1.82x` for a plain J64 tile) and one percent at B16. Paired, order-balanced on the same banks:

| Basis | B1 | B4 | B16 |
| --- | ---: | ---: | ---: |
| J16 stage below the J32 tail | `+14.5 % / +14.3 %` | `+4.9 % / +4.7 %` | `+0.9 % / +1.0 %` |

The J16 tile doubles the packed-weight traffic of the rows it covers, so it only pays where the remainder distribution is dominated by very small groups. The pair record measures the same change at `+2 %` on its B1 route and a fraction of a percent elsewhere.

### Fragment-order activation reads

The activation used to take the long way round: the grouped producer writes one `block_q8_1_mmq` per token per 128-value plane - four scale floats in front of the 128 payload bytes - the body copied the plane into an LDS tile, and the vector dot read its B fragments back out of that tile with `load_ldmatrix`. The fragment the int8 WMMA wants is, per lane, two sixteen-byte chunks of one token's payload at `xs0 + (lane % 16) * 144 B` and `+16 B`, which is where the producer had already put them: the tile bought nothing but a copy.

The fragment body keeps only each token's metadata header in LDS, refreshed ahead of the barrier that publishes it, and reads the payload straight from the workspace with the same addressing arithmetic. The request falls from `28,928 B` to `20,736 B`, which takes the body from four to six workgroups per WGP. The stage loses its activation copy and its `load_ldmatrix` reads: at B16 the LDS instruction samples fall from `4.8 / 5.6 %` to `2.3 / 1.7 %`, the global loads rise from `1.8 %` to `3.8 %`, and the barrier samples fall from `20.5 %` to `13.4 %`, with `s_barrier` still the single most sampled instruction in the profile.

Any valid expert partition leaves a last row tile that is not full, and a fragment read addresses a whole `J`-row tile, so such a tile reads up to `J - 1` rows past its group's end. Every plane but the last keeps that read inside the workspace and the last one does not, so the workspace allocation carries one padded row tile behind a view whose shape is unchanged. The fast path needs no per-load bound, which matters because the load sits in the innermost loop.

Paired and order-balanced against the staged body on the same banks:

| Batch | harness builds, pass 1 / pass 2 | installed controls, pass 1 / pass 2 |
| ---: | ---: | ---: |
| 1 | `+15.4 % / +13.7 %` | `+10.0 % / +10.7 %` |
| 4 | `+11.4 % / +9.7 %` | `+9.7 % / +10.9 %` |
| 16 | `+4.7 % / +3.9 %` | `+4.1 % / +4.9 %` |

The two halves are separable, and a request sweep on the fragment body splits them. At four workgroups (`28,928 B`) instead of six the fragment body measures `13.06 / 23.09 TF` at B1/B16, against the staged body's `11.75 / 23.61`: the removed copy is worth about ten percent at the small route and nothing at the large one, where sixteen scattered 32-byte groups per load cost what the tile copy did, and the two extra workgroups carry the rest. The request rather than the body's own footprint is what sets the workgroups per WGP: the staged body at `38,400 B` and at `43,690 B` both ask for three, and the second measures like the two-workgroup arm, so the allocation granularity eats its two bytes of slack.

The stage code is shared with every other routed type, so the mechanism is available to them, but the dense record's Q8_0 screen is the caveat: where the decode is nearly free the scattered fragment loads cost more than the tile they replace.

### Closed and deferred mechanisms

- Activation prefetch (`MMQ_PREFETCH_ACT`, the grouped register staging of the second activation plane) is inert for this family: the staging lives in the two-stage grouped block, and a `K = 640` control takes the row-tile path with its contraction tail instead, so the two builds of the earlier record differed only in a define that no reachable code reads. The knob stays in the bundle for the routed types that do take that block.
- The level table (`table_decode`) is threaded through the grouped body and measured: paired against the shipped body it is `-0.2 % / -0.8 % / -0.8 %` at B1/B4/B16. The table's `1,024 B` leave the request at `21,760 B` and six workgroups, so the lookup replaces a spread and a subtract for an LDS read in the inner loop and nothing else, which is a small loss where the dense record measured a small win.
- The weight pipeline (`MMQ_WEIGHT_PIPELINE`) is an IQ2_XXS payload mechanism and does not apply to an 18-byte block.
- Row-task ownership was the lever the padding analysis first pointed at, and it does not remove the padding: a row tile cannot span two experts because its packed weights differ, so a task bank changes who issues the tiles, not how many empty rows the tile height costs. The empty rows need a tile height that follows each group's remainder, which is what the J32 tail now does. What stays open is balance: the static grid gives one block per route entry, so at B16 a block whose group holds `17,896` rows walks `280` row tiles while thousands of single-tile blocks finish immediately.
- Deployment wiring. The staged body is catalogued as `grouped_fwd_serial_q2_0_n2560_k640_j64`, the fragment body as `grouped_fwd_serial_q2_0_n2560_k640_j64_frag`, and the shipped body as `grouped_fwd_serial_q2_0_n2560_k640_j64_j32_frag`. All three build into the HIP control bundle, but none carries a routed rule yet: the routed table is asserted to mirror the public route set exactly, and no Q2_0 grouped-forward route exists in the public bundle, so the family cannot be registered until that route does. Every number above comes from a built control, launched through the grouped forward ABI with the route bank the law produced.

## Evidence

The campaign lives in `~/tmp/torch-ggml-ops/grouped_fwd_q2_0/`: `model.py` slices one expert out of the checkpoint and unpacks Q2_0, `routes.py` samples the fitted law into a route bank, `check_expert.py` checks one expert at any group size, `bench_bank.py` measures a full bank with the BF16 baseline, `sweep_j.py` / `sweep_v.py` run the tile swarms, `frag_probe.py` and `debug_frag.py` build and separate the two activation paths, `tail_probe.py` builds the four tile/path combinations, `deploy_measure.py` measures the installed controls through the deployment geometry, `occ_probe.py` sweeps the LDS request, and `profile_bank.py` / `read_prof.py` / `cat_pc.py` collect and categorize the counters.
