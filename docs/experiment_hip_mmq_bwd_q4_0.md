# HIP MMQ Backward Q4_0 Experiment

## Scope

This record covers the gfx1151 HIP input-gradient backward kernels for Q4_0 weights. The type appears only in the shared-expert down projection of 16 of the 48 layers of the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (`qwen4_exp_text`, hidden size 2560), 14.1 MiB of packed weights, so the record has one dense family: a `(2560,640)` weight, backward `(M,640,2560)` over the B1/B4/B16 token counts of a sequence length 2048 batch. The 48 layers mix recipes, and a projection family whose type has no body falls back to a dequantizing multiply.

Three properties of the type drive the kernel. `QK4_0 = 32` weights share an 18-byte block, so a 32-value contraction step sits exactly on one block per weight row and the tile needs one scale per row and no cross-block metadata. The payload is a nibble-plane pair: byte `j` holds weight `j` in its low nibble and weight `j + 16` in its high nibble, at level `q - 8`. There is no codebook, no second plane and no sub-block scale, so a value costs one shift, one mask, one subtract and one fp16 multiply, which is the cheapest payload in the bundle after Q2_0.

## Final kernel result

| Family | `(M,K,N)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| Shared-expert down | `(2048,640,2560)` | 24.002 | 1.467x | `dense_bwd_q4_0_pipea_nt4_ki64_mw2_sw16_g0` |
| Shared-expert down | `(8192,640,2560)` | 29.419 | 1.776x | `dense_bwd_q4_0_pipea_nt4_ki64_mw2_pad8` |
| Shared-expert down | `(32768,640,2560)` | 31.118 | 1.347x | `dense_bwd_q4_0_pipea_nt4_ki64_mw2_sw16` |

## Kernel implementation

The body is the reusable pipelined one (`csrc/ck/mmq_backward_pipelined.cuh`), which the Q2_0, Q3_K, Q4_K, Q5_K and Q6_K records already use: a 128-thread workgroup, `n_tiles * 16` result columns, `K_ITERATION` contraction values per stage, two shared tiles with one barrier per stage, and register double buffering of the activation fragments. This is the first type it carries that was not previously served by the shared single-tile body, so the whole family comes from it.

The Q4_0 decode reads the nibble plane per value: `(qs[value & 15] >> (4 * ((value >> 4) & 1))) & 0x0f`, minus eight, times the block's fp16 `d`. The width-16 group form is what the deployed tile uses. A prefetched form reads the block's 16 payload bytes as one `uint4` and selects the plane by the in-block half, and measures the same as the group form here.

## Resources

The deployed body uses `144 VGPR`, `19 SGPR` and `16 KiB` LDS (two 64x64 bf16 tiles), spill-free.

## Optimization log

### Candidate screen

Six candidates were built and screened on all three row counts: the pipelined tile with and without the activation prefetch, its prefetched-decode twin, a four-row-tile variant with the prefetched decode, a padded four-row-tile variant, and a 32-value-stage variant. The two-row-tile tile wins every row count: `1.09-1.48x` over the four-row-tile variants and `1.11-1.41x` over the 32-value stage. The 32-value stage is the notable loss, because this family's contraction is only 2560 values and its result only 640 wide, so a 64-value stage already amortizes the barrier over the whole slice and the narrower stage pays barrier frequency instead. Padding loses to the swizzle-only 8 KiB tile (`1.09-1.21x` at the two larger row counts), and padding four or more values costs the fourth resident workgroup on a tile this small.

The three surviving variants - group decode, prefetched decode, and the activation prefetch - are within the run-to-run spread of each other on every row count (`0.95-1.06x`, no consistent direction), so the choice is structural rather than measured: the deployed body is the swizzle-only pipelined tile with the activation prefetch, which is the smallest LDS footprint of the three. The type's decode is cheap enough that neither the prefetched payload read nor the activation prefetch buys anything measurable here, unlike Q3_K, Q4_K and Q5_K where the prefetched decode is worth `1.00-1.07x`.

### Traversal order at the starved row count

The `M=2048` point of this family is the narrowest grid in the dense set: `16 x 10` workgroups, four per CU. The deployed body resolves one M block across every result block before it moves on, so the resident set shares a 655 KiB activation slab and streams the weight strips. The `_g0` twin runs every M block of one result block first, so the resident set shares the 92 KiB weight strip and streams the activations.

Under repeated launches, which is the state a training step runs in, `_g0` is `1.35x` ahead of the deployed order - `0.280` against `0.378 ms`, `24.0` against `17.8 TFLOPS`, `95%` interval `-26.4%..-25.0%` over forty samples per body - and the `M=2048` key deploys it. Single-launch samples of the same pair tie (`0.278`/`0.279 ms`), because an isolated launch is latency-bound before either order's cache behavior matters, so a screen that times one launch per sample cannot see the difference, and the earlier single-shot screens of this key (which recorded `18.3 TFLOPS`) did not. The larger row counts keep the deployed order: there the M-fastest order is `2.1x` and `3.4x` behind, because 64 and 256 M blocks no longer fit the resident set.

The other grid-shape knobs are closed at every row count. One row tile per wave doubles the grid but also doubles the decode-to-matrix ratio, so it lands at `0.78-0.96x` on the `M=2048` row and `1.3-1.6x` behind elsewhere. A 32-column tile is `1.5-2.0x` behind at every row count. Slicing the contraction forces the same M-fastest order and lands in the same band at `M=2048` while carrying a partial workspace and a reduction kernel, and it is `2.2-4.9x` behind at the larger row counts, so the plain traversal body is what deploys.

### Where this type sits

`24.0/28.1/31.1 TFLOPS` is above the BF16 baseline on every row and behind the sibling types at the same shape family: Q5_K measures `30.4/30.0/33.9` and Q6_K `36.6/37.7/38.0` on `(M,2560,640)`, and Q8_0 `22.0/19.6/22.8` on the DeepSeek shared down family. The reason is the result width rather than the decode: this shape is only 640 wide, so the grid at `M=2048` is `16 x 10` workgroups and the machine is starved - which is also why the four-row-tile variants that would enlarge the tile lose - and the `M=2048` row is the weakest of the three even with the traversal that suits it.

### Padding on the pipelined tile

The pipelined tile's layout was chosen when the tile was first built - sixteen-value swizzle chunks, no padding - and the chunk was later swept on this body without revisiting padding. The layout is now screened over every deployed dense-backward key: the padding twin of each deployed body (eight values of row padding, no swizzle) is timed against it at the key's own row counts, four repeats per block, and every screen winner is then re-timed at eight repeats over two blocks in both measurement orders, so a key moves only when the padding body wins in both directions.

Padding is not a general replacement for the swizzle. It wins where the resident grid is thin and the result is narrow - the `M=2048` rows and the 512- and 640-wide results - and loses on the wide results at the largest row counts, where the decoded tile's shared-memory traffic is high enough that the swizzle's bank pattern still pays. The split is per key rather than per type: on this type only the `(M,640,2560)` shared-expert down row at `M=8192` moves, `9.8%` ahead of the swizzle body.

A second layout round then screened three further combinations on every officially measured body - sixteen-value padding, four-value padding, and eight-value padding with an eight-value swizzle chunk - at the deployed row counts. Nothing survives the confirmation: each candidate measures within `1-3%` of the deployed tile in one measurement order and loses in the other, so all three are rejected and none is built. The layout neighborhood is closed at the combinations the deployment uses.

## Evidence

```text
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q40_v2.txt              (candidate screen)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q40_v3.txt              (confirmation)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_reopen_traversal.txt    (grouped-M screen)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_reopen_gridshape.txt    (row-tile, column-tile, slice and 32-column-tile screens)
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_reopen_confirm.txt      (repeated-launch paired confirmation)
```
