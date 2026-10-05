# HIP Grouped MMQ Forward Pair IQ2_XXS Experiment

## Scope

This record covers the fused routed IQ2_XXS gate/up forward kernel for DeepSeek.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/AITER GMM | Kernel |
| ---: | --- | --- | --- | --- |
| 1 | `2 x (12288,2048,4096)` | 11.29 | 2.444x | `grouped_fwd_serial_iq2_xxs_n2048_k4096_j64` |
| 4 | `2 x (49152,2048,4096)` | 17.88 | 1.569x | `grouped_fwd_serial_iq2_xxs_n2048_k4096_j64` |
| 16 | `2 x (196608,2048,4096)` | 22.18 | 1.797x | `grouped_fwd_serial_iq2_xxs_n2048_k4096_j80` |

The serial body is level at B1 and B16 (`1%`/`2%` against the uniform-route control) and trails by `15%` at B4, where the residual is the repeated two-weight IQ2_XXS decode and pair accumulator pressure. Flagged prior-sensitive at B4: the J64/J80 shape split may need a learned-route retune.

The retained choice was measured on other route distributions, so a learned-route candidate sweep is the follow-up for the flagged batches.

## Issue-mix measurement

The forward pair kernel for `(196608, 2048, 4096)` with two projections, `grouped_fwd_serial_iq2_xxs_n2048_k4096_j80`, was profiled with `rocprofv3` counter collection on the deployed route (`~/tmp/torch-ggml-ops/mmb_probe/pmc_gfpair_iq2/`). The kernel issues `251.4G` instructions, of which `VALU` is `86.4%`, `LDS` `9.9%`, `SALU` `3.7%` and `VMEM` effectively `0%`. Against the `12.9G` sixteen-by-sixteen matrix steps of the same problem that is `19.5` non-matrix instructions per matrix step, where the analytic budget for a body of this shape is `10-15` (activation loads `4-6`, weight loads about `1`, decode `2-4`, addressing `2-4`, epilogue).

The kernel is therefore ALU-bound with roughly a third to a half of its issue slots above the budget, which is where the decode and addressing recipes of the plan's R2 item have to be applied. The packed-weight loads are not the constraint.

### Epilogue ablation

PC sampling on the same route (`~/tmp/torch-ggml-ops/mmb_probe/pcs_gfpair_iq2/`) attributes `23.6%` of samples to issued `VALU` instructions, and among those the single most frequent instruction is the accumulator's int-to-float conversion `v_cvt_f32_i32_e32` at `22.0%`, ahead of the matrix instruction at `10.7%` and the float epilogue arithmetic at `13.1%`. That looks like a large prize, so it was measured directly instead of assumed.

Two ablations of the shared epilogue were built from the same rendered source with the same compiler flags and timed against the unmodified build on the same prepared case. Both are wrong by construction, so only their timing is meaningful:

| variant | time | against unmodified |
| --- | ---: | ---: |
| unmodified | 273.7 ms | 1.000x |
| one scale product per matrix step, conversion kept | 256.4 ms | 1.067x |
| accumulator consumed by an integer sink, no per-element float work | 257.0 ms | 1.065x |

The first row's variant is not a usable mechanism, for the reason the dense record spells out: the lane-major C fragment gives each thread eight different result rows, so one scale per thread tile coarsens the scale across rows where the quantizer assigns different values, and no coarsening along the contraction can make one scale serve them. The sink variant is a ceiling and it is sound: it consumes every accumulator value, so it replaces the whole per-element float epilogue with one integer add per element and bounds that epilogue at `6.5%` of the runtime, which is far below the `35%` share the conversions and float multiplies take in the issued-instruction mix.

The stall evidence agrees: issued samples are only `29%` of all samples (`70.8%` are samples with no instruction available), and the stalled reasons are `ARBITER_WIN_EX_STALL` `30.7%`, `ARBITER_NOT_WIN` `22.4%`, `ALU_DEPENDENCY` `20.9%`, `BARRIER_WAIT` `13.8%` and `WAITCNT` `9.4%`. The kernel is latency and synchronization bound with idle issue slots, so instruction-count reductions in the shadow of those stalls do not translate into time. The non-matrix instruction budget in the plan's yardstick therefore does not predict this kernel's time, and the levers are occupancy, barrier frequency and dependency chains rather than the decode or the epilogue.

### Latency experiments

The epilogue ablation said the kernel is latency bound, so the staging was attacked directly. All variants below are built from the same rendered source with the same flags and timed against the unmodified build in one harness (`~/tmp/torch-ggml-ops/r3_epilogue/`). The ablations are wrong by construction, so only their timing is meaningful.

| variant | time | against unmodified |
| --- | ---: | ---: |
| unmodified | 273.7 ms | 1.000x |
| packed-weight loads removed entirely (decode, shared stores, barriers and matrix work kept) | 220.5 ms | 1.241x |
| cache-warming prefetch of the next k block, discarded | 295.3 ms | 0.928x |
| general loop routed through the shared one-block helper | 274.4 ms | 0.997x |
| the same with the activation-plane register prefetch enabled | 275.7 ms | 0.992x |
| one-stage weight-payload pipeline (loads issued one block early, decoded after the block's dots) | 273.0 ms | 1.024x |

The packed-weight load latency is the largest identified cost: removing the loads altogether is worth `1.241x`. Everything that tries to overlap it without restructuring is neutral or negative. A cache-warming prefetch doubles the traffic and inserts a scheduling barrier, so it loses `7%`. The activation-plane prefetch facility that already exists in the shared one-block helper is neutral when the general loop is routed through it (`0.997x`) and negative when enabled (`0.992x`), which matches the activations being shared across the 32 output tiles and therefore mostly L2-resident.

A one-stage weight pipeline, which moves the packed payload into registers one k block ahead and decodes it after the current block's dots, is worth `1.024x`: real but small, because one stage of lookahead only covers the hundred-odd cycles of one block's matrix work against a memory latency an order of magnitude larger.

The kernel's structural constraint is that a workgroup needs 31.5 KiB of shared memory for the decoded weight tile and 208 registers per thread, which allows exactly two workgroups per compute unit (eight waves, a quarter of the machine's wave capacity). With that occupancy the load latency of every stage cannot be hidden by other work, and hiding it inside one workgroup needs a pipeline several blocks deep, which costs registers the body does not have to spare. The structural alternatives are the same ones the plan's R3 note now lists: a smaller output tile to buy a third workgroup per unit, or a deeper register pipeline, and both trade occupancy or registers for latency rather than instruction count.

## Weight-side redesign: wider activation tiles

The ablation that removed the packed-weight loads entirely was worth `1.241x`, which made a wider activation tile the campaign's largest remaining ceiling: a workgroup re-reads its weight tile once per J-row activation chunk, so the total packed-weight traffic of an expert is proportional to `1 / J`, and a wider tile trades shared memory and registers for it. The design that was built does that in the general way rather than as a one-off: the row tile and the workgroup width became overridable per control, and the grouped-forward launch geometry follows the control instead of a module constant.

Two candidates were built and measured against the deployed J80 route on the same case (`(196608, 2048, 4096)`, paired grouped forward, six even repeats, both routings through the catalog so that the quantified activations, the packed weights and the route banks are identical on both sides).

| candidate | row tile | threads | accumulators | registers | spills | LDS | time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deployed | 80 | 128 | 40 | 208 | 0 | 31,552 B | 271.3 ms, 273.8 ms |
| narrow, widest that keeps two workgroups per unit | 88 | 128 | 44 | no spill | 0 | 32,608 B | 284.2 ms |
| wide, widest spill-free wide tile | 128 | 256 | 64 | 247 | 0 | 57,856 B | 274.8 ms, 276.3 ms |

The register sweep that chose the candidates is in `~/tmp/torch-ggml-ops/r8_wide/sweep_j.py`: it renders the real control for each J, compiles it with the build flags and reads the code-object metadata, and it puts the spill cliff between J128 (`247` registers, no spills) and J136 (`256` registers, seven spills, a `32`-byte private segment). J160 needs `63,104` bytes of LDS and spills sixty registers, which is why the plan's first check was register pressure rather than time.

Both candidates lose, and the loss is informative. The narrow one removes about ten percent of the weight traffic while keeping the deployed occupancy and loses four percent, and the wide one removes about thirty-eight percent of the traffic and loses one percent: the traffic is not what this body pays. The `1.241x` of the load ablation came from removing the loads' latency and their arbiter pressure, which a wider tile does not remove, while a wider tile changes the work decomposition for the worse: the narrow candidate pads its `88 x 36` activation tile up to `128` and the wide candidate halves the number of workgroups per expert and drops to one workgroup per compute unit, so a single workgroup's barrier-separated stages can no longer overlap with another's. On a body whose profile is arbiter and barrier stalls, that is exactly the wrong trade.

The redesign is therefore closed, with the machinery for it left in the tree as the starting point for any future tiling change and the two rejected candidates kept as the measured evidence.

## Weight-side latency: register pipeline

The ablation that removed the packed-weight loads outright was worth `1.241x` on this body, and it also carried a variant labelled as a one-stage register pipeline that measured `1.024x`. That variant had no prologue: it skipped staging the first k block of every row tile, so it did fifteen sixteenths of the staging work and could not produce a correct tile for the first matrix step.

The honest implementation was built into the tree instead, with the staging split into a payload load (three registers per staged row: two packed words and the block scale) and a payload store (the decode, the grid lookups, the sign application and the shared stores, all unchanged), a prologue that stages block zero, and a rotation of `MMQ_WEIGHT_PIPELINE` register stages whose indices are constants inside an unrolled inner loop, so the payload array stays in registers.

The measurements go through the catalog on the same case as the deployed route, so the quantified activations, the packed weights and the route banks are identical on both sides.

| variant | stages | registers | spills | time |
| --- | ---: | ---: | ---: | ---: |
| deployed | - | 208 | 0 | 271.3 ms, 273.8 ms |
| pipeline | 1 | 234 | 0 | 280.8 ms, 281.5 ms |
| pipeline | 2 | 253 | 0 | 278.4 ms |
| pipeline | 4 | 256 | 81 (192-byte private segment) | not timed |

Every depth that fits the register file is slower than the deployed body, and the ceiling is out of reach for a structural reason: each stage buys about one percent (`281.5` at one stage, `278.4` at two) while the pipeline's own cost is about three percent, and the payload registers cap the depth at two because a thread already holds forty accumulators and the body needs two hundred and eight registers before any staging. Covering a memory latency an order of magnitude larger than one iteration would need several stages, which neither the register file nor the second workgroup per compute unit can pay for. Four stages spills eighty-one registers outright.

The pipeline is therefore closed, with the mechanism left in the tree and the three candidates kept as the measured evidence.

## Remaining decode and traffic findings

The staging loader spends three loads per thread per staged row (two adjacent payload words and the block scale), and the two payload words cannot be merged into one wider load because the packed layout leaves them two-byte aligned. Together with the loader being about six to eight instructions per code word, this closes the decode side: the measured `1.241x` for removing the packed-weight loads is latency, not instruction count, and the register budget closes that avenue as well (see the tile and pipeline sections above).

A disassembly sweep of every installed control scores integer multiplies and 64-bit address add pairs. This body is one of only two deployed controls with forty or more integer multiplies, and its `42` of `4043` instructions are about one percent. The PC sampling agrees, putting integer multiplies at `1.1%` of issued samples behind the accumulator conversion at `25.6%`, the matrix step at `12.4%` and the float arithmetic at `15.2%`. They are not an addressing dependency chain of the kind the Q6_K forward fix removed.

The one idea left on the weight side is a resident weight window: the packed weights of a workgroup are small enough that keeping a window of them in cache would remove the load latency the ablation priced at `1.241x`, but it needs a cheaper form than widening the tile, because widening the tile is measured to be the wrong trade on this body (the tile removes traffic and loses time because it halves the workgroups per expert).

## Optimization log

### Sign-mask table

IQ2_XXS expanded each sign byte with the parity trick plus two `__vcmpne4` emulations per value group, which is the most expensive sign path of the grouped forward family. The byte passed to `unpack_ksigns` is the only input of both masks, so a `256`-entry `int2` table now supplies them.

Paired in-process A/B: `1.1%` at B1, `0.3%` at B4 (inside noise) and `2.6%` at B16 on the J80 body. Under the benchmark protocol all three batches improve (`1.3%`, `1.6%`, `2.9%`), with B16 at `1.797x` against AITER GMM. The DeepSeek J64 body and the dense IQ2_XXS controls share the table.

The J64/J80 split was re-checked against the learned route with the existing bodies: J80 is `8.2%` slower at B1, level at B4 and `4.0%` faster at B16, which matches the deployed split.

### Initial exact decoder

The generic grouped body used narrow N16/K16 ownership and serial row work. Exact IQ2_XXS pair geometry and cooperative width-16 decode removed repeated metadata work and made the B1/B4 packed path substantially faster than the baseline. Early width32 controls had mixed route movement without a legal shape-only separator and were rejected.

The initial exact J64 body moved the B1 uniform pair from `77.549 ms` to `33.581 ms`. This is the historical tiled-kernel milestone behind the retained four-wave body, not an additional final workload result.

### J80 safety repair

The J80 candidate originally faulted for a non-divisible activation load, leaving 64 excess lanes in the final 128-thread iteration. J16 had the same structural defect. The repaired J80 was approximately `1.096x` faster than J64 in the focused B16 comparison. Wider and smaller ownership alternatives were rejected by timing or resource gates.

### Coefficient-only campaign

The final typed search retained serial J64 at B1/B4 and the repaired J80 geometry for the large B16 workload. J16 and J64 at B16 were slower. The production body uses 208 VGPRs and 54 SGPRs for the J80 point.

The corresponding J64 artifact uses `229 VGPR / 77 SGPR / 8192 B LDS`. The repaired J80 artifact uses `208 VGPR / 54 SGPR / 8192 B LDS`. Both are resource-clean. The failed historical M192 body had 32 private bytes, seven VGPR spills, and scratch instructions, so its rejection does not transfer to the later lower-state J80 result.

The remaining bottleneck is two independent IQ2_XXS lookup/sign/scale decoders feeding one pair accumulation. A prepared weight layout or decode reuse is a representation change, not another broad J sweep.

Still open for this pair: the grid lookup and the IQ2_XXS grid selector remain the decode cost after the sign table, and a fused two-projection body that stages the activation tile once is the remaining structural idea.

## Evidence

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/grouped_fwd_current/pass13_pair_deepseek_default_small.json
~/tmp/torch-ggml-ops/grouped_fwd_current/pass13_pair_deepseek_default_b16.json
```

The original campaign evidence is:

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_deepseek_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped-fwd-all-screen/deepseek_iq2xxs_pair_identity_guard_b16_search_5.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/deepseek_iq2xxs_production_b16_replay.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
```

No further width or J sweep is justified without a new decode-state premise.

The final route campaign and repair qualification are additionally recorded in:

```text
~/tmp/torch-ggml-ops/grouped-fwd-prior-v1.json
~/tmp/torch-ggml-ops/grouped-fwd-candidate-resources.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
~/tmp/torch-ggml-ops/grouped-fwd-production-final-rebuild.json
~/tmp/torch-ggml-ops/grouped-fwd-production-final-resources.json
```
