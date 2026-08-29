# GGTensile Grouped MMQ Forward Q4_K Experiment

## Scope And Contract

This record covers the isolated gfx1151 grouped Q4_K forward kernel for the Qwen routed down projection. For each routed expert, the operation is:

```text
X_g[M_g,512] @ W_g[512,2048] -> Y_g[M_g,2048]
```

The measured aggregate-row shapes are `R=16384`, `65536`, and `262144`. The authoritative packed Q4_K expert bank has shape `[256,2048,288]`; each 256-value block occupies 144 bytes. Runtime route metadata supplies up to 256 physical expert IDs and cumulative row ends, so `M_g` is neither a generation constant nor necessarily a tile multiple. The activation operand is the fixed HIP Q8_1 `F16_D4S4` workspace with physical shape `[4,R,144]`. Output is contiguous BF16 `[R,2048]`.

The grouped research ABI is the 64-byte multiply contract:

```text
weights, activations, dst, expert_indices, expert_offsets,
num_experts, nrows_weight, nrows_activation,
blocks_per_weight_row, bytes_per_expert
```

Logical throughput is `2 * R * 2048 * 512 / (latency_ms * 1e9)`, and the speedup ratio is HIP time divided by GGTensile time. The final timing uses the fixed HIP quantizer and prequantized multiply-only throughput.

## Final Benchmark Results

The table shows the fastest qualified kernel found for each measured aggregate-row shape.

| Matrix shape `(R,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(16384,2048,512)` | `ggsol_9c98efab3bdeda2b` | `16.042` | `1.1684x` |
| `(65536,2048,512)` | `ggsol_110dda8ec3bcfc8d` | `22.633` | `1.0213x` |
| `(262144,2048,512)` | `ggsol_2aee91a9195cc10a` | `25.270` | `1.0031x` |

The selected entries passed the retained route-correctness, resource, and deterministic-build checks.

## Final Kernel Profiles

| Kernel hash | Geometry and ownership | Decode and LDS policy | VGPR / SGPR | LDS bytes | WMMAs / barriers |
| --- | --- | --- | ---: | ---: | ---: |
| `ggsol_9c98efab3bdeda2b` | M128/N64 serial, mixed 64/32 tail | `a1d4-p2` epilogue, padded single LDS | `159 / 40` | `29,184` | `24 / 4` |
| `ggsol_110dda8ec3bcfc8d` | M128/N64 serial, mixed 64/32 tail | `a1d4-p2` epilogue, padded single LDS | `159 / 40` | `29,184` | `24 / 4` |
| `ggsol_2aee91a9195cc10a` | M128/N64 serial, 128-row body | `a1d2-p2` epilogue, padded single LDS | `239 / 40` | `38,400` | `32 / 4` |

The 24 static WMMAs in the mixed-tail identities represent mutually exclusive 64-row and 32-row bodies. All selected artifacts target gfx1151 code-object version 5 and wave32 with zero private storage, spills, scratch, calls, and dynamic stack.

## Accepted Kernel Experiments

### Direct-global routing and arithmetic control

The first grouped Q4_K artifact emits one-wave serial GEMM ownership with a 16-row macro tile, masked activation reads, and masked BF16 stores for arbitrary `M_g`. It assembles at 88 VGPRs, 32 SGPRs, zero LDS, and 16 static WMMAs, and matched the installed grouped HIP path exactly on a real Qwen bank with route lengths of 1, 15, 16, and 3 rows. Against an independently dequantized reference it stayed within the expected quantized arithmetic envelope. This body remains the routing, ABI, and arithmetic control. The production tiled schedule is not bitwise identical to the one-wave control; the measured production difference is `3.05e-5` maximum absolute error and `6.04e-9` RMS, far below the independent reference envelope.

### Decoded-weight LDS bodies and the mixed 64/32 tail

The first decoded-weight LDS body reused the typed dense Q4_K decode and scaled WMMA emitters under serial routed ownership. It staged a 128-by-64 tile at 239 VGPRs, 40 SGPRs, 38,400 LDS bytes, 32 static WMMAs, and four barriers but reached only `0.734x`, `0.917x`, and `0.960x` installed/candidate at B1/B4/B16. Parameterizing the layout and register plan by row fragments produced a 64-by-64 tile at 159 VGPRs, 40 SGPRs, 29,184 LDS bytes, 16 static WMMAs, and four barriers. Independent scale/minimum extraction and metadata reads between the low and high WMMAs reduced the 64-row low-half wait ladder to `7,5,3,1` and improved the weighted results to `1.060x`, `0.986x`, and `0.971x`.

The retained mixed 64/32 tail body decodes weights once per K block, then branches only activation staging, local reads, WMMA correction, and masked stores when the final routed tile has at most 32 rows. Its `a1d4-p2` schedule uses independent groups of four BF16 roundings at priority two and restores priority before serial traversal continues. The confirmation measured `2.142 ms` versus `2.503 ms` for HIP at B1 (`1.168x` weighted) and `6.073 ms` versus `6.202 ms` at B4 (`1.021x`), with the independent search bank reproducing `1.205x` and `1.025x`. The complete-call audit with the fixed quantizer measured `1.2167x`, `1.0311x`, and `1.0006x` installed/candidate. The dominant medoids contain many 32-row-or-smaller remainders, which explains the transfer; some low-weight large-group medoids remain slower, with confirmation minima of `0.931x` at B1 and `0.953x` at B4.

### B16 128-row control

The scheduled 128-row `a1d2-p2` body is the better large-group control at B16. Its reversed-order 25-repeat confirmation measured `21.756 ms` versus `21.823 ms` for HIP (`1.003x`), with every medoid at least `0.997x`. The movement is below the normal promotion margin, so this identity is retained as a resource-clean correctness and schedule control rather than a durable large-key win.

## Rejected Kernel Experiments

### Four-wave direct-global geometry

Grouping four WaveN owners into a 128-thread workgroup while retaining direct-global arithmetic improved the smallest key only marginally and regressed the large keys: uniform `R=262144` moved from `99.42` to `108.09 ms` and the boundary route to `105.46 ms`, leaving the candidate 2.1 to 5.3 times slower than HIP. Four-wave workgroup formation without decoded-weight or activation reuse is not an actionable optimization.

### Route-persistent full-K decoded weights

Decoding both K512 blocks once before the row loop into two immutable LDS images, with disjoint activation storage, produced a 57,856-byte LDS artifact at 239 VGPRs, 40 SGPRs, 64 static WMMAs, and eight barriers. An operand-order error in the image-pointer restore was repaired and the body became bitwise equal to the selected 128-row parent for route lengths `1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 256` and for uniform, skewed, sparse, and boundary distributions at all three shapes. It nevertheless failed the performance discriminator before confirmation: parent/full-weight body ratios were `0.805x` to `0.849x` at B1, `0.860x` to `0.882x` at B4, and `0.891x` to `0.921x` at B16, including an approximately 12% regression on the B16 uniform route with the strongest decode-amortization premise. The larger immutable LDS footprint and doubled static body outweigh the removed re-decodes, so the mechanism is rejected and its speculative lowering removed.

### Routed-prologue and address reductions

The packed-kernarg, paired cumulative-offset, and exact address reductions rebuilt deterministically and matched the parent bitwise on all boundary, repeated-ID, sparse/skewed, first-route, unaligned-index, invalid-expert, and invalid-offset controls. Their Q4_K B1 candidate-time movements were `+0.030%`, `-0.086%`, and `-0.083%`, far below the advancement gate, so no confirmation or cross-format transfer was run. Direct-to-LDS is unavailable: the configured gfx1151 assembler rejects the buffer and LLVM spellings and reports `global_load_lds_dword` unsupported.

### Zero-bank initialization

The later paired-source audit does not transfer a zero-bank hoisting candidate here. The decoded-LDS Q4_K body already initializes its zero bank once per row tile, so it has no repeated per-projection lifetime matching the paired finding.

## Remaining Work

The current direct-kernel benchmark measured B1 at `1.0688x` GGTensile/HIP (75-repeat interval `[1.0618x, 1.0758x]`) versus the documented `1.1684x`; the route has 233 active experts and a maximum group of 1,297, so this is a stable short-row result rather than timing noise. B4 and B16 remain close to their documented rows at `1.0327x` and `0.9962x`. The R1-R3 route/address transformations may be retimed on the current multiply-only surface as a diagnostic requalification. The route-persistent full-K rejection is not reopened by this discrepancy, because its direct body evidence already shows a large LDS/resource regression.

## Qualification Summary

The final kernels pass exact packed-HIP comparison, independent BF16-reference checks, finite-output and full-row coverage, deterministic reruns, uniform, skewed, sparse-ID, repeated-ID, and boundary routes, gradient and active-weight mutations, inactive-expert inertness, malformed-route sentinels, and non-aligned tails. Two independent generation, build, and inspection roots produce byte-identical artifacts. The retained result is the grouped Q4_K direct decoder with padded single-LDS storage, the mixed 64/32 `a1d4-p2` body for B1/B4, and the scheduled 128-row `a1d2-p2` control for B16.
