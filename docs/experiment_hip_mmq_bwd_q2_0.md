# HIP MMQ Backward Q2_0 Experiment

## Scope

This record covers the gfx1151 HIP input-gradient backward kernels for Q2_0 weights, opened for the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, 48 layers, hidden size 2560, 512 experts with top-10 routing). The dense keys are 45.9 MiB over the layers that use Q2_0 outside the experts. The type also carries all 144 routed-expert tensors, 32,400 MiB, which are the grouped records. GatedDeltaNet `in_proj_qkv` and `in_proj_z` share their shapes with the PLE key projection and the QSA key/value family, so the deployment keys of those families serve them as well. The third projection, `out_proj`, is deferred and this type carries no `out_proj` tensor in either checkpoint. M counts tokens of a sequence length 2048 batch (B1/B4/B16), and the training path calls `M=2048` for these projections.

Two properties of the type drive the kernels. Four 2-bit codes share one payload byte and are *consecutive* weights with the lowest bits first and level `q - 1`, so a value costs a shift, a mask, an integer subtract and one fp16 multiply, and there is no codebook, second scale plane or sign table. The block is `QK2_0 = 64` weights wide, so a 32-value contraction step crosses two payload bytes of a weight row rather than two block scales, and the 256-wide exact-K wrappers of the `_K` types do not transfer.

## Final kernel result

| Family | `(M,N,K)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| QSA query | `(2048,12288,2560)` | 30.580 | 1.312x | `dense_bwd_q2_0_mt128_full_pipea_nt64_ki64` |
| QSA query | `(8192,12288,2560)` | 29.729 | 1.159x | `dense_bwd_q2_0_mt128_full_pipea_nt64_ki64` |
| QSA query | `(32768,12288,2560)` | 27.212 | 1.040x | `dense_bwd_q2_0_mt128_full_pipea_nt64_ki64` |
| Shared-expert gate/up | `(2048,640,2560)` | 34.679 | 1.616x | `dense_bwd_q2_0_mt256_full_pipea_nt64_ki64_mw4_pad8` |
| Shared-expert gate/up | `(8192,640,2560)` | 37.710 | 1.390x | `dense_bwd_q2_0_mt256_full_pipea_nt64_ki64_mw4_pad8` |
| Shared-expert gate/up | `(32768,640,2560)` | 37.954 | 1.351x | `dense_bwd_q2_0_mt256_full_pipea_nt64_ki64_mw4_pad8` |
| Shared-expert down | `(2048,2560,640)` | 22.928 | 1.410x | `dense_bwd_q2_0_mt128_full_pipea_nt64_ki64_sw0_pad8` |
| Shared-expert down | `(8192,2560,640)` | 30.437 | 1.829x | `dense_bwd_q2_0_mt128_full_pipea_nt64_ki64_pad8` |
| Shared-expert down | `(32768,2560,640)` | 31.923 | 1.774x | `dense_bwd_q2_0_mt128_full_pipea_nt64_ki64_pad8` |
| PLE key / GDN `in_proj_qkv` | `(2048,10240,2560)` | 30.265 | 1.292x | `dense_bwd_q2_0_mt128_full_pipea_nt64_ki64` |
| PLE key / GDN `in_proj_qkv` | `(8192,10240,2560)` | 30.131 | 1.169x | `dense_bwd_q2_0_mt128_full_pipea_nt64_ki64` |
| PLE key / GDN `in_proj_qkv` | `(32768,10240,2560)` | 27.028 | 1.080x | `dense_bwd_q2_0_mt128_full_pipea_nt64_ki64` |
| GDN `in_proj_z` | `(2048,6144,2560)` | 31.522 | 1.375x | `dense_bwd_q2_0_mt128_full_pipea_nt64_ki64_pad8` |
| GDN `in_proj_z` | `(8192,6144,2560)` | 31.008 | 1.208x | `dense_bwd_q2_0_mt128_full_pipea_nt64_ki64` |
| GDN `in_proj_z` | `(32768,6144,2560)` | 29.756 | 1.196x | `dense_bwd_q2_0_mt128_full_pipea_nt64_ki64` |

Backward shapes are written `(M,N,K)`, matching the weight's `(N,K) = (out_features, in_features)` in the table's `(M,N,K)` column. `_pad8` names the eight-word LDS padding. The unpadded variants hold 16 KiB rather than 18 KiB and admit a fourth resident workgroup per WGP.

## Kernel implementation

### Shared-tile body

The first generation shares the `mt128_nt128_ki32_full` skeleton of the Q4_K, Q5_K and Q6_K records: a 128-thread workgroup, four waves, `n_tiles * 16` result columns, `k_iteration` contraction values per stage, and LDS-staged decoded weights. `WEIGHT_BLOCK_VALUES` resolves to `QK2_0` for this type.

The decode has two forms. The generic form indexes the payload byte per value, `(qs[value >> 2] >> (2 * (value & 3))) & 0x03`, subtracts one from the code and scales by the block's fp16 `d`. The packed form, taken when the tile is eight result columns wide with a 32-value stage and a width-16 decoder, loads the four payload bytes covering its 16 values as one unaligned `uint32` and extracts the sixteen codes from it. The forward records' level table does not apply: the backward tile has no codebook, only the per-block scale, so a table would still need the scale multiply.

### Pipelined body

The second generation is `csrc/ck/mmq_backward_q2_0.cuh`, a Q2_0-only body that keeps the same tile geometry, decode and matrix work but holds two shared tiles and issues the decode of stage `s + 1` before the matrix work of stage `s`:

```text
decode(stage 0 -> tile 0); s_barrier;
for stage in 0 .. stages - 1:
    decode(stage + 1 -> tile[(stage + 1) & 1]);   // no barrier needed: tile is not read yet
    multiply(tile[stage & 1]);
    s_barrier;                                    // retires the decode, frees the other tile
```

That is one barrier per stage instead of two, and the decode's LDS stores and global loads are in flight under the current stage's matrix work instead of in front of it. The tile is also where the LDS budget goes, so the body is built in a padded and an unpadded flavour: `_pad8` keeps the decoded tile in sixteen-value-aligned banks, while the unpadded `_nt64_ki64` variant spends 16 KiB instead of 18 KiB and admits a fourth resident workgroup per WGP, which is what the four wide families deploy. Turning the LDS swizzle off is not an option: the decoded tile is written one column per thread at a stride of `k_iteration` values, and without the swizzle the same body measures `14.8 TFLOPS`.

The body additionally double-buffers the activation fragments in registers: the sixteen bf16 cotangent values of k tile `k + 1` are issued before k tile `k`'s fragment loads and matrix work, so their global latency retires under that work rather than in front of it. The rotation costs `M_TILES_PER_WAVE` fragments of registers.

## Resources

| Body | LDS | VGPR | SGPR |
| --- | ---: | ---: | ---: |
| `_pipea_nt64_ki64` | 16 KiB | 130 | 18 |
| `_pipea_nt64_ki64_sw0_pad8` | 18 KiB | 147 | 18 |
| `_pipea_nt64_ki64_mw4_pad8` | 18 KiB | 216 | 18 |

All are spill-free. The single-tile bodies use `120-232 VGPR` at `9-10 KiB` LDS.

## Optimization log

### Decode, tile shape and padding

The first-generation screen covered the plain packed tile, a four-column tile, its ungrouped traversal, two 64-wide-stage variants, an eight-column four-row-tile variant, decoder widths of eight and 32, and padding of four, eight and sixteen values. The four-column tile with a 64-value stage and two row tiles per wave won nearly every point, `1.5-2.0x` over the packed eight-column tile on the wide shapes and `1.09-1.26x` over its own 32-value-stage twin. A four-row-tile variant lost `1.18-1.26x`, a one-row-tile variant lost `1.26-1.52x`, a two-column tile lost `1.8x`, and decoder widths of eight or 32 were `1.04-1.23x` behind. Padding of eight values then added `1.01-1.07x`, and padding of four or sixteen was worse than no padding. The one point preferring the packed eight-column tile is the shared-expert down projection at `M=2048`, whose result is only 640 wide. `m_tiles_per_wave = 3` faults with a memory error, so that geometry is not built.

The launch path also had to stop assuming a 256-value block: `blocks_per_weight_row` was `in_features // 256` in both dense launchers and in the measurement harness, which is wrong for every type whose block is not `QK_K`. They now take it from the type's format record.

### Why the pipeline, and why not warp specialization

PC sampling of the single-tile body put `s_barrier` at `27%` of samples with another `27%` at `BARRIER_WAIT`, against `11%` at the WMMA instruction: each stage decoded into LDS, barrier, multiplied, barrier, so the matrix units idled while the tile filled. The pipelined body moves barrier waits to `26%` and WMMA samples to `17%`, and gains `1.11-1.13x` on every family. The sampled LDS loads drop from `16%` to `12%`.

Warp specialization - dedicated producer waves decoding while consumer waves multiply - is the other standard answer, and it is not expressible on this machine. gfx1151 (RDNA3.5) has only the workgroup-wide `S_BARRIER`. The named producer/consumer barriers (`S_BARRIER_SIGNAL` / `S_BARRIER_WAIT`) arrive with the gfx12 ISA, and the only partial-sync alternative, the global wave sync (GWS) resource, is a graphics-oriented cross-CU mechanism with a driver round trip. On a four-wave workgroup every handoff would therefore be workgroup-wide, which is what the pipelined body already does, and the remaining overlap has to come from the hardware wave scheduler interleaving resident waves.

That is also why the occupancy of this body matters more than its instruction count. The kernel issues roughly `1130 FLOP` per issued instruction, so at 29 TFLOPS it uses about `11%` of the machine's issue slots: it is nowhere near issue-bound, and the `32%` arbiter and `25%` ALU-dependency samples in the profile are the signature of too few resident waves rather than of too much work. The 18 KiB padded tile holds three workgroups per WGP, three waves per SIMD. The unpadded 16 KiB tile holds four, which is the whole reason it wins at `M=2048`.

The second `__launch_bounds__` argument is the switch for that trade - it asks the compiler to leave a minimum number of waves per execution unit resident, which caps the VGPR budget at `512 / min_waves` per lane - and it was swept on six bodies at `min_waves` of one, three and four against the deployed two. It cannot buy residency the way an LDS change can: no body changed its VGPR, SGPR or spill count at any setting. Three of the six bodies compiled to byte-identical code at every setting (the pipelined four-row-tile and split-head bodies), and the other three compiled to a different schedule without a different register budget, the largest change being this type's four-row-tile pipelined body at `2349` to `1966` instructions. Every setting ties or loses: within `1.5%` on the wide and head bodies, and `1.01-1.11x` behind on this type, where the rescheduled body is the slowest. The deployed hint therefore stays at two.

### Grid-shape screens on the pipelined tile

The pipelined body leaves four grid-shape knobs open - the grouped-M order, the row tile per wave, the column tile, and slicing the contraction - and all four were screened on this type's three families. The M-fastest order is `2.1-3.4x` behind the deployed order at `M=8192` and above and `1.05x` behind at `M=2048`, so the deployed order stands. A two-block M group stays within `1.1x` everywhere. One row tile per wave doubles the grid but also doubles the decode-to-matrix ratio, so it lands `1.47-1.58x` behind at the two larger row counts. A 32-column tile is `1.8-2.9x` behind at every row count, including the starved `M=2048` point, where the extra column blocks cost more activation traffic than the added parallelism buys. Slicing the contraction twice or four times is `2.4-5.1x` behind at the larger row counts with the partial write and reduction inside the timed region, and `0.91-1.01x` at `M=2048`, where the small result makes the partial workspace cheap. None of the four displaces the deployed geometry.

### Where the pipeline puts Q2_0 relative to the K-quants

With the pipeline the type now leads the K-quants on comparable shapes, which the cheaper decode alone never bought: the Q6_K record measures `33.4/35.6/36.6 TFLOPS` on `(M,2560,640)` and `33.7/31.3/27.1` on `(M,6144,2560)` where this record measures `36.6/37.7/38.0` and `31.3/31.0/29.8`.

### Padding on the pipelined tile

The pipelined tile's layout was chosen when the tile was first built - sixteen-value swizzle chunks, no padding - and the chunk was later swept on this body without revisiting padding. The layout is now screened over every deployed dense-backward key: the padding twin of each deployed body (eight values of row padding, no swizzle) is timed against it at the key's own row counts, four repeats per block, and every screen winner is then re-timed at eight repeats over two blocks in both measurement orders, so a key moves only when the padding body wins in both directions.

Padding is not a general replacement for the swizzle. It wins where the resident grid is thin and the result is narrow - the `M=2048` rows and the 512- and 640-wide results - and loses on the wide results at the largest row counts, where the decoded tile's shared-memory traffic is high enough that the swizzle's bank pattern still pays. The split is per key rather than per type: on this type the three rows that carry the padded `_pad8` tile were deploying it *with* an eight-value swizzle chunk on top, and dropping that chunk is worth `30.8%`, `31.1%` and `29.7%` on the GatedDeltaNet `in_proj_z` eye, the shared-expert down row at `M=8192` and the same row at `M=32768`, with the shared-expert gate/up row at `M=2048` `3.4%` behind its old body. The padded bodies are now named for what they are: `_pad8` carries padding only and `_pad8_sw8` carries both.

A second layout round then screened three further combinations on every officially measured body - sixteen-value padding, four-value padding, and eight-value padding with an eight-value swizzle chunk - at the deployed row counts. Nothing survives the confirmation: each candidate measures within `1-3%` of the deployed tile in one measurement order and loses in the other, so all three are rejected and none is built. The layout neighborhood is closed at the combinations the deployment uses.

## Evidence

```text
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q20_final.txt      (single-tile table)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q20_pipe_doc.txt   (pipelined confirmation)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q20_final2.txt     (padding/occupancy split)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q20_v{1..7}.txt    (candidate screens)
~/tmp/torch-ggml-ops/q20_bwd/profile_one.py           (single-tile launcher)
~/tmp/torch-ggml-ops/q20_bwd/prof_counters/           (PMC run, single tile: L2 88.9%, LDS conflicts 58.3%)
~/tmp/torch-ggml-ops/q20_bwd/prof_pc/                 (stochastic PC sampling, single tile)
~/tmp/torch-ggml-ops/q20_bwd/prof_pipe_pc/            (stochastic PC sampling, pipelined)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_reopen_gridshape.txt (grid-shape screens, this type and the Q4_0 shape family)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_reopen_traversal.txt (grouped-M screen, this type and four others)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_reopen_minwaves.txt  (launch-bounds minimum-wave sweep, six bodies)
```
