# HIP Grouped MMQ Backward Q2_0 Experiment

## Scope

This record covers the routed Q2_0 down input-gradient kernel for Qwen4-Exp on gfx1151.

The model is `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing). Every routed-expert tensor of the checkpoint is Q2_0, the gradient contracts the down projection's output back to the expert intermediate size, and the training path calls `R=20480`.

## Final kernel result

| Batch | Logical shape `(R,N,K)` | HIP TFLOPS | HIP/bf16 GMM | Kernel |
| ---: | --- | ---: | ---: | --- |
| 1 | `(20480,2560,640)` | 14.00 | 2.50x | `grouped_bwd_row_task_q2_0_n2560_k640_mt128_nt64_s3_sw8_g4_abar` |
| 4 | `(81920,2560,640)` | 25.63 | 2.74x | `grouped_bwd_row_task_q2_0_n2560_k640_mt256_nt64_s3_ki64_sw8_g4_abar` |
| 16 | `(327680,2560,640)` | 35.21 | 2.75x | `grouped_bwd_row_task_q2_0_n2560_k640_mt256_nt64_s3_ki64_sw8_g4_abar` |

Banks come from the `qwen3.8-learned` law, one profile per size at the declared seed (`352 / 458 / 497` active experts, largest group `1261 / 4735 / 17896` rows). The baseline is the same routed product computed in BF16 over predecoded expert weights, expert by expert, with `torch.mm`: it is what an unquantized multiply-only grouped kernel reaches on this part, in the role the AITER GMM number plays in the plan. The quantized kernel is `2.50x` to `2.75x` ahead of it, and it is above every routed sibling at every shape (`Q4_K` `9.63 / 16.60 / 20.94`, `Q5_K` `10.31 / 16.96 / 21.86`, `IQ2_S` `12.34 / 18.19 / 22.24`, `Q2_K` `12.87 / 21.48 / 26.01` TFLOPS), which is what the smallest decode in the bundle should buy.

## Kernel implementation

The body is the staged row-task one the routed backward family deploys, with the folded chunk XOR of the pair record applied to the decoded tile (`chunk = (column / 4) ^ ((row ^ (row >> 4)) & 7)`), which spreads the decoded row writes across banks without moving the fragment reader's own spread. It uses an M128 task per expert row tile, a 32-wide contraction stage, three LDS buffers and one plain barrier per stage, and waves whose first row is past the task end skip their consumer work.

The deployed hoisted body allocates 200 VGPR / 12288 B LDS.

## Optimization log

### Swizzle fold

The decoded tile is written down its rows: a thread decodes sixteen consecutive output columns of one packed row, so its sixteen stores advance by the contraction stride while every lane starts at the same column. The plain chunk XOR (`chunk = (column / 4) ^ (row & 7)`) therefore gives the writer only the row bits the fragment reader also uses, and the counter run on the deployed Q2_0 pair body of the same shape reports `LDSBankConflict 58.75 %`. Folding the row bits one sixteen-row step above the low ones (`row ^ (row >> 4)`) adds the writer's row step without moving the reader's. The change is layout-only and the body is bit-identical to the deployed one (`max|base-candidate| 0.0` against the BF16 product for both).

Paired against the previous control in one process, both orders, rotating banks, and with both arms pinned by symbol:

| Route | against the previous control |
| --- | ---: |
| B1 | +3.0 % / +3.1 % |
| B4 | +4.5 % / +4.6 % |
| B16 | +5.6 % / +5.7 % |

The fold is not universal: it wins on the chunk-4 pair layouts and on the deep-contraction Q2_K single, and it loses on the Q4_K and Q5_K singles (`+0.1 % / +2.5 % / +2.9 %` and `+6.7 % / +6.2 % / +4.1 %` over the same protocol), whose decoders are expensive enough that the tile's bank pattern is not what limits them. Those two candidates are not deployed and their controls are removed.

The fold's knob was added with a zero grain that silently dropped the swizzle term from every deployed body. The defect, its profiler evidence and the harness rules that follow are in the pair record of this family.

## Device ceiling

A register-only bf16 WMMA probe on this part (`~/tmp/torch-ggml-ops/grouped_bwd_pair_q2_0/wmma_peak.cu`: one `v_wmma_f32_16x16x16_bf16` after another on register operands, no shared memory, no decode, no global traffic in the loop) sustains `55.3 TFLOP/s`, against the `59.4 TFLOP/s` the part's clock, lane count and matrix instruction allow. That is the ceiling the rates in the table are fractions of, and any recorded rate above the probe would mean the kernel is not doing the work its FLOPs count.

### Swizzle chunk

The staged tile's swizzle decides which LDS bank each decoded value lands in. A thread writes sixteen *consecutive* weight columns for one of the eight packed rows it owns, so a chunk of sixteen pairs the four writes of a row with itself and the reads collide. Halving the chunk splits the row into four and spreads both. Paired, order-balanced, two passes on the same banks:

| Variant | B1 | B4 | B16 |
| --- | ---: | ---: | ---: |
| chunk 8 against chunk 16 | `+9.0 % / +9.0 %` | `+16.4 % / +16.4 %` | `+20.4 % / +20.4 %` |
| chunk 4 against chunk 8 | `-2.8 % / -2.7 %` | `-3.6 % / -3.7 %` | `-4.2 % / -4.1 %` |

The chunk-8 body is the shipped one and its outputs are byte-identical to the chunk-4 body, which is the cheapest confirmation that the change is a bank-layout change and not an arithmetic one.

### A-fragment hoist

PC sampling of the deployed body attributed `52-55 %` of wave stalls to the stage barrier, with `s_barrier` the single most sampled instruction, while VALU ran at a few percent of peak issue. The gradient fragments are private to the wave, so their global loads do not have to follow the barrier: issuing the first projection's fragment load before `s_barrier` lets the barrier wait cover the L2 latency the multiply would otherwise stall behind. Nothing else moves, and the hoisted arm is bit-identical to its parent (`max|base-candidate| 0.0`).

Paired in one process, both orders, rotating route banks, both arms pinned by symbol:

| Route | hoisted against its parent |
| --- | ---: |
| B1 | +4.3 % / +4.3 % |
| B4 | +2.2 % / +2.1 % |
| B16 | +1.3 % / +1.4 % |

The same screen closed the neighbouring latency hypotheses on this family. Rotating the stage geometry - K64/N2 keeps the shared footprint and the decode calls per thread and halves the barrier count, K64/N4 doubles the footprint, K32/N2 halves the columns - lost on both bodies: `-1.8/-2.5/-2.5 %`, `-12.1/-15.2/-18.8 %` and `-22.6/-20.6/-18.5 %` at B1/B4/B16 for the single, and `-4.6/-4.9/-5.2 %`, `-9.5/-12.4/-15.9 %` and `-15.8/-18.5/-22.6 %` for the pair.

The barrier *count* is therefore not the lever: the wait per barrier scales with the work between barriers, and the variants that buy fewer barriers with registers or shared memory lose more than they save. Hoisting the second projection as well needs a second fragment set and measured neutral (`-0.1/+0.3/-0.3 %`), so only the first projection is hoisted. The `k64n2` column-tile mapping left in the launcher by an earlier session had no recorded measurement. This screen replaces it with one.

### Wide task tile

A task descriptor decodes its own copy of the weight rows it covers, so the decode per output row is one weight matrix per descriptor and a 256-row descriptor halves it. The wide body keeps the per-wave tile, four column tiles and the accumulator budget of the deployed shape - eight waves of `M_TILES = 2` give the 256 rows - and pairs that with a 64-value contraction stage so the stage count over the 2560-wide reduction halves as well. Against the four-wave body, paired in one process with both arms pinned by symbol:

| Route | `mt256_nt64_s3_ki64` against `mt128_nt64_s3_sw8_g4` |
| --- | ---: |
| B1 | `-0.4 % / -0.4 %` |
| B4 | `+7.1 % / +7.2 %` |
| B16 | `+11.5 % / +12.4 %` |

B1 is a wash because the learned bank's 58-row experts cannot fill a 256-row descriptor, and the K32 twin of the same body lands between the two (`+3 % / +5.8 %` at B4/B16, `-3 %` at B1). The deployment takes the wide body from B4 up. The same screen shows the pair is not as lucky: its `mt256` twins measure `-12 %` at B1 and `-2 %` at B4 against `+6.5 %` at B16, so its table keeps the four-wave body below B16.

### Closed and deferred mechanisms

- Stage count: two LDS stages are level with three (paired `+0.04 % / +0.00 % / +0.05 %`), so the family keeps the three-stage skeleton it inherited.
- Inactive-wave suppression: skipping the consumer work of waves whose first row is past the task end is worth `+44 % / +17 % / +2 %` at B1/B4/B16 against the same body without it, which is the opposite of what the Q4_K and Q5_K records measured for their decodes. It stays on for this type.
- The task bank cap: the routed family's builder scanned 256 route entries per pass, which silently produced an empty bank for this model's 352 to 497 active experts. The scan now runs once per chunk of 256 entries and carries the finished count as its base, which is what makes the 512-expert contract work at all.
- Wider N ownership, the first-generation non-staged tile, and the sibling decode items are closed by the sibling records and do not apply here.
- Deployment wiring. The body is catalogued as `grouped_bwd_row_task_q2_0_n2560_k640_mt128_nt64_s3_sw8_g4_abar` and builds into the HIP control bundle, but it carries no routed rule yet: the routed table is asserted to mirror the public route set exactly, and the public backward routes stop at Q2_K, Q4_K, Q5_K and IQ2_S, so the family cannot be registered until a Q2_0 route exists.

## Evidence

The campaign lives in `~/tmp/torch-ggml-ops/grouped_bwd_q2_0/`: `model.py` slices one expert out of the checkpoint and unpacks Q2_0, `routes.py` samples the fitted law into a route bank, and `bench_bwd.py` launches the deployed control through the production task bank and launcher, checks it against the BF16 grouped product and times it against that baseline.
