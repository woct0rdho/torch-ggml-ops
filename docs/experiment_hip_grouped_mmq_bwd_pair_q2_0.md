# HIP Grouped MMQ Backward Pair Q2_0 Experiment

## Scope

This record covers the routed Q2_0 gate/up input-gradient pair for Qwen4-Exp on gfx1151, the pair half of the Q2_0 backward family.

The model is `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing), whose every routed-expert tensor is Q2_0 and whose gate and up projections are separate per-expert tensors. The pair shares its routed rows with the single-projection records, so one fitted law drives all of them.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/bf16 GMM | Kernel |
| ---: | --- | ---: | ---: | --- |
| 1 | `2 x (20480,640,2560)` | 13.49 | 2.11x | `grouped_bwd_pair_task_q2_0_n640_k2560_mt128_nt64_s2_skip_g4_mb3_abar` |
| 4 | `2 x (81920,640,2560)` | 23.25 | 1.96x | `grouped_bwd_pair_task_q2_0_n640_k2560_mt128_nt64_s2_skip_g4_mb3_abar` |
| 16 | `2 x (327680,640,2560)` | 32.25 | 1.98x | `grouped_bwd_pair_task_q2_0_n640_k2560_mt256_nt64_s2_skip_k64_g4_mb3_abar` |

The B16 row uses the wide task tile. B1 and B4 keep the four-wave body, whose measured rates are `13.49` and `23.25` TFLOPS. Banks come from the `qwen3.8-learned` law, one profile per size at the declared seed (`352 / 458 / 497` active experts, largest group `1261 / 4735 / 17896` rows). The baseline is the same routed product computed in BF16 over predecoded expert weights, two grouped products per expert plus the add, with `torch.mm`: it is what an unquantized multiply-only grouped kernel reaches on this part, in the role the AITER GMM number plays in the plan. The quantized kernel is `1.85x` to `2.11x` ahead of it, and it leads the Q3_K pair at B16 (`30.21` against `29.47` TFLOPS) while carrying the cheaper decode.

## Kernel implementation

The body is the paired staged row-task body the family deploys, with two layout decisions of its own. The decoded tile is swizzled with a folded chunk XOR (`chunk = (column / 4) ^ ((row ^ (row >> 4)) & 7)`) so that the decoded row writes spread across banks as well as the fragment reads, and the control reserves registers for three workgroups per compute unit, which keeps residency off the six-wave register ceiling. It consumes a device row-task bank, uses an N64 x M128 tile, a 32-wide contraction stage, two projection weight tiles per stage, two LDS stages and one plain barrier per stage, and clamps activation rows so no load is predicated.

The deployed hoisted body allocates 256 VGPR / 16384 B LDS.

## Optimization log

### The zero-grain defect this screen exposed

The knob was first written as `row ^ (row >> SWIZZLE_GRAIN)`, which is zero when the grain is zero: every deployed body silently lost its swizzle and became plain row-major, and screens run in that window measured candidates against a control that was itself regressed. The profiler sees it on the deployed body, same symbol and same batch:

| paired-backward Q2_0 body, B16 | duration | VGPR | LDS bank conflicts | VALU instructions per wave |
| --- | ---: | ---: | ---: | ---: |
| recorded build | 71.29 ms | 240 | 58.8 % | 7935 |
| with the zero-grain defect | 86.88 ms | 216 | 75.0 % | 6641 |
| after the `grain == 0` repair | 72.89 ms | 248 | 58.8 % | 7807 |

The repair restores the recorded build within a couple of percent, so the defect, not the machine or the protocol, was the whole difference. Two harness mistakes grew out of it: a candidate body with a different column tile needs the launcher's grid width to follow that tile, and the launcher now derives it from the control (`COLUMN_TILES`) and rejects a width that does not divide the output, because a body owning half the columns while the grid still steps by the full tile writes only half the output and looks fast while being wrong, and a screen whose base arm is resolved from the deployment table silently compares a kernel with itself once that table has been repointed, so both arms are now pinned by symbol. The rule that follows: a table row moves only when its symbol moves, and a re-measurement of an unchanged kernel belongs in the log.

### Small-route fill probe

Whether the small-route deficit is empty tile rows was measured before building anything. The paired body was run on the same 352 experts and the same packed banks at uniform row counts, with the learned B1 bank as reference:

| bank | rows | rows/expert | tasks | ms | TFLOPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned B1 | 20480 | 58 | 448 | 9.953 | 13.49 |
| uniform | 10560 | 30 | 352 | 7.722 | 8.96 |
| uniform | 20416 | 58 | 352 | 7.911 | 16.91 |
| uniform | 40832 | 116 | 352 | 9.054 | 29.56 |
| uniform | 81664 | 232 | 704 | 18.224 | 29.37 |

Two conclusions. Pairing two small experts into one workgroup is not the fix it looked like: each expert still needs its own decode, so the pairing amortizes nothing and the candidate was dropped before it was built. What does show up is the decode paid per task: the learned bank needs 448 tasks for the rows a uniform bank covers with 352, and it loses `(448/352 - 1) = 27 %` - almost exactly its measured deficit against the uniform bank at the same average fill (`13.49` against `16.91`). The lever is rows per decode, i.e. rows per task, which is the `M256 / N32 / K64` shape: four row tiles per wave and a 256-row task bank decode the same column block for twice the rows.

### Launch bounds

The body owns 248 registers with four LDS stages of two projection tiles, which caps residency at six waves per SIMD while the shared footprint would allow eight. Reserving the registers for three workgroups per compute unit (the second `__launch_bounds__` argument) is free at the source level and worth, paired against the unfolded control in one process:

| Route | three-block launch bounds |
| --- | ---: |
| B1 | -3.39 % |
| B4 | -4.93 % |
| B16 | -6.42 % |

All three measured routes are bit-identical to the control (`max|base-candidate| 0.0`), and four workgroups per unit is level with three (`-6.23 %` at B16), so three is deployed. The knob is not general: on the IQ2_S pair it is exactly neutral (`+0.15 % / +0.02 % / +0.14 %` over the same protocol) and on the Q2_0 single it changes nothing (`+0.03 % / +0.02 %`), because those bodies already sit inside the three-block register budget. Those candidates are not deployed and their controls are removed.

### Swizzle fold

The decoded tile is written down its rows: a thread decodes sixteen consecutive output columns of one packed row, so its sixteen stores advance by the contraction stride while every lane starts at the same column. The plain chunk XOR (`chunk = (column / 4) ^ (row & 7)`) therefore gives the writer only the row bits the fragment reader also uses, and the counter run on the deployed Q2_0 pair body of the same shape reports `LDSBankConflict 58.75 %`. Folding the row bits one sixteen-row step above the low ones (`row ^ (row >> 4)`) adds the writer's row step without moving the reader's. The change is layout-only and the body is bit-identical to the deployed one (`max|base-candidate| 0.0` against the BF16 product for both).

Paired against the previous control in one process, both orders, rotating banks, and with both arms pinned by symbol:

| Route | against the previous control |
| --- | ---: |
| B1 | +3.4 % / +3.6 % |
| B4 | +4.8 % / +5.2 % |
| B16 | +6.0 % / +6.4 % |

The fold is not universal: it wins on the chunk-4 pair layouts and on the deep-contraction Q2_K single, and it loses on the Q4_K and Q5_K singles (`+0.1 % / +2.5 % / +2.9 %` and `+6.7 % / +6.2 % / +4.1 %` over the same protocol), whose decoders are expensive enough that the tile's bank pattern is not what limits them. Those two candidates are not deployed and their controls are removed.

The Q3_K pair keeps its padded tile: a swizzled twin of that body measured `+1.9 %` at every route against its (unaffected) parent, and the grain twin of that twin raised an illegal memory access, so the padded layout is what that decoder deploys and the fault is still unexplained.

## Device ceiling

A register-only bf16 WMMA probe on this part (`~/tmp/torch-ggml-ops/grouped_bwd_pair_q2_0/wmma_peak.cu`: one `v_wmma_f32_16x16x16_bf16` after another on register operands, no shared memory, no decode, no global traffic in the loop) sustains `55.3 TFLOP/s`, against the `59.4 TFLOP/s` the part's clock, lane count and matrix instruction allow. That is the ceiling the rates in the table are fractions of, and any recorded rate above the probe would mean the kernel is not doing the work its FLOPs count.

### Swizzle chunk

The tile's swizzle decides which LDS bank each decoded value lands in. A thread writes sixteen *consecutive* weight columns for one packed row, so those sixteen writes sit a fixed stride apart and collide unless the swizzle spreads them. Paired and order-balanced on the same banks, two passes each:

| Variant against chunk 4 | B1 | B16 |
| --- | ---: | ---: |
| chunk 8 | `-0.3 % / -0.3 %` | `-1.2 % / -1.2 %` |
| chunk 0 (no swizzle) | `-7.7 % / -7.9 %` | `-21.0 % / -21.2 %` |
| chunk 4 with eight columns of padding | `-2.0 %` | `-1.6 %` |

Chunk 4 is the shipped one. Two further chunks were built and reject themselves: chunks 2 and 1 violate the fragment reader's assumption that a swizzle chunk holds a whole vector, and they produce scrambled output rather than a slower kernel, which is worth remembering because the tile's only guard is that the chunk count stays a power of two.

### A-fragment hoist

PC sampling of the deployed body attributed `52-55 %` of wave stalls to the stage barrier, with `s_barrier` the single most sampled instruction, while VALU ran at a few percent of peak issue. The gradient fragments are private to the wave, so their global loads do not have to follow the barrier: issuing the first projection's fragment load before `s_barrier` lets the barrier wait cover the L2 latency the multiply would otherwise stall behind. Nothing else moves, and the hoisted arm is bit-identical to its parent (`max|base-candidate| 0.0`).

Paired in one process, both orders, rotating route banks, both arms pinned by symbol:

| Route | hoisted against its parent |
| --- | ---: |
| B1 | +1.5 % / +1.5 % |
| B4 | +0.6 % / +0.6 % |
| B16 | 0.0 % / 0.0 % |

The same screen closed the neighbouring latency hypotheses on this family. Rotating the stage geometry - K64/N2 keeps the shared footprint and the decode calls per thread and halves the barrier count, K64/N4 doubles the footprint, K32/N2 halves the columns - lost on both bodies: `-1.8/-2.5/-2.5 %`, `-12.1/-15.2/-18.8 %` and `-22.6/-20.6/-18.5 %` at B1/B4/B16 for the single, and `-4.6/-4.9/-5.2 %`, `-9.5/-12.4/-15.9 %` and `-15.8/-18.5/-22.6 %` for the pair.

The barrier *count* is therefore not the lever: the wait per barrier scales with the work between barriers, and the variants that buy fewer barriers with registers or shared memory lose more than they save. Hoisting the second projection as well needs a second fragment set and measured neutral (`-0.1/+0.3/-0.3 %`), so only the first projection is hoisted. The `k64n2` column-tile mapping left in the launcher by an earlier session had no recorded measurement. This screen replaces it with one.

The neighbouring latency variants are closed with numbers. Hoisting the second projection as well needs a third fragment set and measured neutral. Hoisting only that projection's first contraction tile splits its multiply and costs `5 %` to `15 %` on all four paired families, so the first projection remains the only hoisted load. A 512-row descriptor was rejected on arithmetic rather than measured: sixteen waves per workgroup at this register count leave one resident workgroup per compute unit, which halves the resident waves the barrier needs to hide behind.

### Closed and deferred mechanisms

- Two LDS stages are what the paired family deploys and what this shape wants. The single-projection sibling's three-stage body was not carried over.
- The packed row width is the type's own geometry: a Q2_0 weight row of 2560 values is forty 64-value blocks, not the ten 256-value blocks the forward ABI's `blocks_per_weight_row` counts. The first build used the forward count and produced an oracle error of `1.45` relative. The bytes per expert follow the same arithmetic (`640 x 40 x 18`).
- Wider N ownership, the non-staged pair bodies, and the sibling decode items are closed by the sibling pair records and do not apply here.
- Deployment wiring. The body is catalogued as `grouped_bwd_pair_task_q2_0_n640_k2560_mt128_nt64_s2_skip_g4_mb3_abar` and builds into the HIP control bundle, but it carries no routed rule yet: the routed table is asserted to mirror the public route set exactly, and the public pair-backward routes stop at Q3_K, IQ2_S and IQ2_XXS.

### Wide task tile

The task descriptor height sets how often the packed weights are decoded: a task decodes its own copy of the weight rows it needs, so the decode per output row is one weight matrix per descriptor, and a 256-row descriptor halves it. The wide body keeps the per-wave tile, the four-column tile count and therefore the accumulator budget of the deployed shape - eight waves of `M_TILES = 2` give the 256 rows - so the only costs are twice the shared memory per stage at the same stage count and more waves arriving at each barrier. Measured at the largest route, where a 256-row descriptor is well filled, paired in one process with both arms pinned by symbol:

| Variant | against the four-wave body at B16 |
| --- | ---: |
| K64 twin | `+6.5 % / +6.4 %` (harness `30.29` to `32.25` TFLOPS). The K32 twin reaches `+2.0 %` |

At the smaller routes the same body loses to the four-wave one (`-19 % / -6 %` for Q5_K's K32 twin at B1/B4, `-13 % / -6 %` for IQ2_S, `-30 % / -25 %` for Q4_K, `-7 %` for the IQ2_XXS pair at B16), because a 58-row expert cannot fill a 256-row descriptor and the wider workgroup pays more drift per barrier than it saves. The deployment therefore takes a `rows_at_least` rule per family instead of replacing the body outright, and Q4_K and the IQ2_XXS pair keep the four-wave body everywhere.

## Evidence

The campaign lives in `~/tmp/torch-ggml-ops/grouped_bwd_pair_q2_0/`: `model.py` slices one expert out of the checkpoint for either projection and unpacks Q2_0, `routes.py` samples the fitted law into a route bank, and `bench_pair_bwd.py` launches the deployed control through the production task bank and launcher, checks it against the BF16 grouped product of both projections and times it against that baseline.
