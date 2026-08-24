# Dense MMQ backward optimization

## Current status

Dense `torch_ggml_ops::mmq_grad_input` is at its repository-local stopping point on gfx1151 for the existing packed-weight, independent-call API.

The retained implementation provides:
- Direct BF16-cotangent by packed-GGUF multiplication with FP32 WMMA accumulation and BF16 dX.
- Static quant, shape, row-count, traversal, and LDS-layout dispatch with bounds-safe fallbacks.
- Qwen production bodies for Q3_K, Q4_K, Q5_K, and Q6_K.
- DeepSeek exact Q8_0 bodies for six ordinary geometries and the language-model head.
- 83 dense MMQ backward specializations, unchanged inside the current 181-kernel source-built gfx1151 bundle; the two additions are separately qualified grouped-backward kernels.
- Zero private storage, zero VGPR/SGPR spills, and no dynamic stack for every retained MMQ backward artifact.

Repository-local Qwen work closed after QB1. DeepSeek work closed after DB8. No additional geometry, traversal, decoder, prefetch, LDS, active-wave, or arithmetic sweep is authorized under the current API. Further material gains require explicit model-owned prepared weights, paired grad-input, BF16-shadow, or shared-scratch ownership.

Source-of-record artifacts:

```text
Qwen ordinary matrix:       ~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_narrow_q5_25.json
Qwen post-DB8 control:      ~/tmp/torch-ggml-ops/mmq_bwd_qwen_post_db8_control_9.json
Qwen complete loss:         ~/tmp/torch-ggml-ops/mmq_bwd_qwen_db6_complete_loss_control_25.json
DeepSeek ordinary matrix:   ~/tmp/torch-ggml-ops/mmq_bwd_ds4_db8_final_25.json
DeepSeek complete loss B1:  ~/tmp/torch-ggml-ops/mmq_bwd_ds4_db6_complete_loss_b1_25.json
DeepSeek complete loss B4:  ~/tmp/torch-ggml-ops/mmq_bwd_ds4_db6_complete_loss_b4_25.json
DeepSeek M512 B16 capacity: ~/tmp/torch-ggml-ops/mmq_bwd_ds4_db6_complete_loss_b16_m512_capacity_3.json
```

The `~/tmp/torch-ggml-ops` paths are measurement provenance, not repository inputs.

## Latest results

Ratios are packed throughput divided by the BF16 `torch.mm` reference throughput. Values above `1.0x` favor packed MMQ. Ordinary rows use `M = batch * 2048`.

### Checkpoint-weighted ordinary backward

| Model | B1 packed/BF16 ms | B1 ratio | B4 packed/BF16 ms | B4 ratio | B16 packed/BF16 ms | B16 ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen | `79.733/89.300` | `1.120x` | `322.527/340.988` | `1.057x` | `1244.217/1315.164` | `1.057x` |
| DeepSeek | `784.797/783.534` | `0.998x` | `3475.582/3079.686` | `0.886x` | `13914.331/12217.398` | `0.878x` |

DB8 reduced DeepSeek packed ordinary latency by `4.51%/18.85%/26.41%` at B1/B4/B16 relative to DB7. The Qwen post-DB8 control measured `1.124x/1.059x/1.079x`, consistent with its 25-repeat source of record.

### Qwen ordinary projections

Each timing and ratio cell is B1/B4/B16.

| Family | Packed ms | Packed/BF16 ratio | Retained mechanism |
| --- | ---: | ---: | --- |
| Query Q3_K | `3.412/12.641/49.501` | `1.230x/1.278x/1.241x` | Packed extraction, two-row prefetch, XOR LDS layout |
| Query Q4_K | `3.376/12.567/48.613` | `1.245x/1.268x/1.264x` | Full 128x128/K32 body |
| Narrow Q3_K | `0.176/0.721/2.833` | `1.079x/1.059x/1.059x` | Padded vector local loads |
| Narrow Q4_K | `0.230/0.775/2.988` | `0.824x/0.958x/0.988x` | Full body. B1 remains below BF16 |
| Narrow Q5_K | `0.208/0.778/3.019` | `0.907x/0.969x/0.985x` | Scalar extraction at 2,048 rows. Packed above |
| Attention output Q4_K | `1.339/5.912/23.803` | `1.110x/1.009x/0.990x` | Padded vector local loads |
| Shared down Q4_K | `0.261/1.537/5.266` | `1.295x/0.737x/0.802x` | 16-BF16 XOR layout |
| Shared down Q5_K | `0.239/1.392/5.606` | `1.414x/0.814x/0.756x` | Scalar extraction, four-BF16 XOR layout |

The remaining ordinary Qwen deficits are narrow B1 Q4_K/Q5_K and shared-down B4/B16. Their local geometry, K-depth, extraction, prefetch, and swizzle neighborhoods are closed.

### DeepSeek ordinary projections

| Family | Packed ms B1/B4/B16 | Packed/BF16 ratio | Retained traversal and layout |
| --- | ---: | ---: | --- |
| Attention Q-A | `0.827/2.597/13.277` | `0.887x/1.168x/0.908x` | all-M padded/M2 padded/M2 unpadded |
| Attention Q-B | `7.088/32.056/115.233` | `0.889x/0.787x/0.851x` | M1 padded/M2 unpadded/M2 unpadded |
| Attention KV | `0.346/1.321/5.221` | `1.146x/1.133x/1.108x` | all-M padded/M2 padded/M2 padded |
| Attention output B | `5.420/26.521/107.038` | `1.066x/0.845x/0.833x` | all-M padded/M2 unpadded/M2 unpadded |
| Shared gate/up | `1.513/5.654/27.626` | `0.983x/1.034x/0.866x` | all-M padded/M2 padded/M2 unpadded |
| Shared down | `1.546/7.024/27.569` | `1.318x/1.106x/1.129x` | all-M padded/M2 unpadded/M2 unpadded |

Traversal is exhausted. Remaining Q-A, Q-B, output-B, and shared gate/up deficits are representation or repeated-decode problems, not untested launch-order cases.

### Complete packed-loss schedules

Production chunk selection includes packed forward, in-place cross-entropy, and packed grad-input. Isolated LM-head kernel time is not the selector.

| Model | Selected chunk | Complete-loop result | Accepted peak or capacity |
| --- | ---: | --- | --- |
| Qwen | M256 | `251.852 ms` at B1 | 253.57 MiB incremental peak allocation |
| DeepSeek | M512 | `173.583/705.975 ms` at B1/B4 | 274.76/322.79 MiB incremental peak allocation |
| DeepSeek B16 | M512 | `2775.945 ms` capacity timing | 514.91 MiB allocation, 532 MiB reservation |

Qwen M64/M128/M256 complete-loop times are `337.030/301.309/251.852 ms`. DeepSeek M512 improves over M256 by `1.28%` at B1 and `1.95%` at B4. Tested schedules produce bit-exact loss and BF16 hidden gradients.

## Production dispatch

### Common contract

`csrc/ck/mmq_backward.cuh` owns the reusable four-wave device body. `tools/build_mmq_bundle.py` generates concrete gfx1151 wrappers, and `csrc/mmq_bundle.cpp` performs static selection from quant type, exact shape, and public row count.

Backward computes `dY @ W` directly from the forward-layout packed weight. Cotangents are not quantized. The operator does not materialize a dense or transposed logical weight. Unsupported shapes use generic bounds-safe wrappers.

Main template state is:

```text
<type,
 N_TILES,
 K_ITERATION,
 GROUP_M,
 M_TILES_PER_WAVE,
 DECODER_WIDTH,
 PREFETCH_LOCAL,
 FULL_TILES,
 PREFETCH_PACKED,
 LDS_PADDING>
```

`all-M` denotes the original ungrouped traversal. M1 and M2 denote `GROUP_M=1` and `GROUP_M=2` launch mappings.

### Qwen dispatch

Ordinary full rows use a measured 128x128/K32 four-wave body with type- and shape-specific extraction, prefetch, and LDS layouts. The important row-specific exception is narrow Q5_K: scalar extraction is selected only at 2,048 rows, while 8,192 and 32,768 rows retain packed extraction.

Q6_K language-model-head rows use:

| Chunk M | Retained body |
| ---: | --- |
| 64 | M64/N32/K64 ownership with a 16-BF16 XOR layout |
| 128 | M128/N64/K32 ownership with an eight-BF16 XOR layout |
| 256 | Two M128-style workgroups, packed quant extraction, eight-BF16 XOR layout |

The public complete-loss schedule selects M256. M128 remains the first lower-memory fallback.

### DeepSeek dispatch

Ordinary rows use the exact G2 128x128/K32 body with width-16 Q8_0 decode.

| Shape `(N,K)` | 2,048 rows | 8,192 rows | 32,768 rows |
| --- | --- | --- | --- |
| Q-A `(1024,4096)` | all-M padding8 | M2 padding8 | M2 unpadded |
| Q-B `(32768,1024)` | M1 padding8 | M2 unpadded | M2 unpadded |
| KV `(512,4096)` | all-M padding8 | M2 padding8 | M2 padding8 |
| Output-B `(4096,8192)` | all-M padding8 | M2 unpadded | M2 unpadded |
| Shared gate/up `(2048,4096)` | all-M padding8 | M2 padding8 | M2 unpadded |
| Shared down `(4096,2048)` | all-M padding8 | M2 unpadded | M2 unpadded |

DeepSeek LM-head rows use active-two-wave M32, G0 M64, G1 M128, and G3 M256/M512. M512 launches two exact M256 workgroups.

### Resources

Representative retained allocations:

| Body | VGPR | SGPR | LDS |
| --- | ---: | ---: | ---: |
| Qwen Q3_K query | 237 | 27 | 8 KiB |
| Qwen Q4_K query/narrow | 226 | 17 | 8 KiB |
| Qwen Q4_K attention output | 222 | 20 | 10 KiB |
| Qwen Q4_K shared down | 222 | 16 | 8 KiB |
| Qwen Q5_K narrow scalar B1 | 247 | 17 | 8 KiB |
| Qwen Q5_K shared down | 253 | 16 | 8 KiB |
| Qwen Q6_K M64/M128/M256 | 87/138/137 | 15/16/15 | 4 KiB |
| DeepSeek G2 unpadded | 192 | 14 | 8 KiB |
| DeepSeek G2 padding8 | 192 | 14 | 10 KiB |

Every retained MMQ backward artifact has zero private bytes, zero VGPR/SGPR spills, and no dynamic stack. The 253-VGPR Qwen shared-down Q5_K body is the practical allocation warning point.

## Remaining work

No repository-local MMQ backward experiment remains pending. The following are separate model-owned projects.

| Priority | Project | First target | Required mechanism |
| ---: | --- | --- | --- |
| 1 | Prepared Q8_0 payload/scale layout | DeepSeek Q-A and shared gate/up | Lossless 34-byte-per-32-value layout with better cross-lane transaction packing |
| 2 | Paired grad-input API (DB5) | DeepSeek shared gate/up | Accumulate two dX contributions in FP32 and write BF16 once |
| 3 | Prepared Q4_K/Q5_K shared-down layout (QB2) | Qwen dense shared down | Tile-major integer-plus-scale representation shared by forward and backward |
| 4 | Decode-once or BF16-shadow floor | Large repeated projections | Explicit memory/lifetime accounting and complete-backward amortization |

### Prepared Q8_0

The leading DeepSeek experiment is a size-neutral `[all qs][all d]` or equivalent tile-major layout with contiguous int8 payloads and FP16 scales. It must keep the packed GGUF tensor as the independent source of truth and define preparation, lifetime, invalidation, device placement, memory, cold cost, and forward/backward sharing.

Start with one Q-A and one shared gate/up tensor while keeping G2 geometry and DB8 traversal unchanged. Measure event time, transactions, L2 hit rate, occupancy, resources, preparation cost, amortized complete backward, one-hot decode equality, and packed-forward impact. Hidden pointer, tensor-version, or stream-local caches are prohibited.

### Paired grad-input

DeepSeek shared gate/up are matching `(N,K)=(2048,4096)` projections over the same input and represent 43 pairs. Q-A `(1024,4096)` and KV `(512,4096)` also share an input but have unequal reductions and are lower priority.

A pair API must be explicit, model-owned, current-stream correct, autograd-integrated, and memory-accounted. Two independent kernels plus `torch.add` remain the control. Retention requires complete model-level backward timing and a defined one-rounding reference. DwarfStar's paired forward path is ownership evidence, not the required summed backward contract.

### Qwen shared-down representation

Before building a persistent representation, measure an optimistic transient floor: copy an already-decoded BF16 weight, run BF16 GEMM while excluding real decode compute, include allocation/copy in the timing, and report it separately from persistent BF16 GEMM. Close the direction if even this floor cannot beat packed MMQ usefully.

The preferred production direction is compact lossless integer-plus-scale storage. Approximate ordinary storage is about 475 MiB compact versus 760 MiB BF16. Acceptance requires gains for both dense Q4_K and Q5_K shared down, no packed-forward regression, explicit invalidation, and preservation of inactive-expert sparsity plus all four route distributions if grouped kernels share the representation.

### Decode-once or BF16 shadow

Approximate BF16 workspaces are 32 MiB for query, 2 MiB for narrow, 16 MiB for attention output, and 2 MiB for shared down. Any two-stage path must report preparation/copy, cold call, steady state, peak memory, and amortized complete-backward time. It is not an all-in-one packed kernel and must not be implemented as a hidden transient cache.

## Optimization log

Historical timings below explain retained choices and closed neighborhoods. They should not be compared directly with the latest source-of-record matrices across different builds or code-object layouts.

### Phase map

| Phase | Result |
| --- | --- |
| Q0-Q3 | Four-wave tiled Qwen body, 128x128/K32 ordinary geometry, bounded packed prefetch, and shape-specific LDS layouts |
| B0 | Converted dense backward to deterministic source-built gfx1151 HSACOs |
| QB0 | Re-established a warmed packaged Qwen control after active-wave predicate folding |
| QB1 | Retained scalar narrow-Q5_K extraction only at B1. Rejected other bounded retunes |
| QB2 | Deferred model-owned Qwen shared-down representation |
| DB0 | Established the generic DeepSeek ownership/decode bottleneck |
| DB1 | Retained exact Q8_0 wrappers for all production geometries |
| DB2 | Retained G2 128x128/K32 for all six ordinary shapes |
| DB3 | Retained row-dependent padding. Rejected other local lowering controls |
| DB4 | Retained M32/M64/M128/M256/M512-specific LM bodies |
| DB5 | Deferred model-owned paired grad-input API |
| DB6 | Selected DeepSeek M512 and retained Qwen M256 complete-loss schedules |
| DB7 | Retained M2 for long-row Q-B and output-B |
| DB8 | Retained M2 for the remaining B4/B16 families and M1 for Q-B B1. Local search exhausted |

### Q0-Q1: Qwen tiled redesign and ordinary geometry

The original body decoded one packed BF16 value at a time into a 16x16 tile and sustained only about `0.05-0.13x` BF16 throughput. The first retained redesign added four wave32 waves, LDS-staged decoded weights, multiple WMMA accumulator tiles, cooperative pair/quad/width-16 decode, and measured row/type dispatch.

Representative initial gains were:

| Case | Historical baseline | First tiled body | Speedup |
| --- | ---: | ---: | ---: |
| Query Q3_K, M32768 | 1,108.314 ms | 162.884 ms | 6.80x |
| Narrow Q4_K, M32768 | 48.332 ms | 11.130 ms | 4.34x |
| Attention output Q4_K, M32768 | 487.945 ms | 80.633 ms | 6.05x |
| LM head Q6_K, M256 | 158.209 ms | 26.130 ms | 6.05x |

An early 64-row geometry preferred `GROUP_M=2`, improving representative M32768 cases by `1.35-1.58x`. That rule did not transfer to the final 128-row tile. The selected ordinary body became 128x128/K32 with width-16 decode and all-M/M1 traversal according to the exact wrapper.

The narrow Q4_K geometry sequence established the durable choices:
- 2x2 ownership was slower than the earlier 1x16 control.
- Increasing N reuse through 2x4 and 2x6 improved latency.
- Width-16 decode beat pair and width-8 decode.
- K32 beat K16 and K64.
- 2x8 was the best valid N ownership.
- Exact full-tile specialization, paired LDS-fragment prefetch, and bounded two-row packed-byte prefetch each added gains.
- 3x6, 4x4, 1x16/K32, broad `GROUP_M=2/4`, complete Q4_K metadata vector loads, custom barriers, and environment-driven dispatch were rejected.

The clean pre-layout narrow Q4_K milestone was about `3.273 ms`, down from `7.053 ms` for the earlier grouped geometry.

### Q2: packed extraction, prefetch, and LDS layout

Q3_K/Q4_K/Q5_K retain explicit scalar or `uint4` packed-byte state. Wide Q3_K packed extraction reduced query latency from 48.694 to 46.980 ms. Keeping the next reduction iteration's packed fragments live across WMMA regressed to 48.606 ms, so prefetch remains bounded to the current decode phase. Narrow Q3_K does not prefetch because its 110-byte block layout made the path slower.

Pre-layout Q4_K profiles reported a repeated `79.2%` LDS-bank-conflict metric. Eight BF16 values of row padding change the K32 LDS stride from 64 to 80 bytes:

| Q4_K M32768 shape | Unpadded | Padded | Decision |
| --- | ---: | ---: | --- |
| Query | 54.895 ms | 46.295 ms | retain padding/layout specialization |
| Narrow | 3.273 ms | 2.907 ms | retain |
| Attention output | 26.875 ms | 23.177 ms | retain |
| Shared down | 5.261 ms | 5.389 ms | reject padding |

Q3_K query/narrow also improved with padding. Q5_K padding regressed and was rejected. Later XOR controls showed that layout is body-specific: eight-BF16 granularity is selected for wide Q3_K and Q4_K query/narrow, 16-BF16 for shared-down Q4_K, and four-BF16 for shared-down Q5_K.

Explicit aligned fragment loads improved narrow Q3_K and attention-output Q4_K but regressed query Q4_K, Q5_K, and shared-down bodies by about 2-4%. Production therefore uses typed local-load selection rather than a global vector-load rule. Lower instruction count did not reliably predict lower LDS stalls or event time.

Shared-down controls closed the local neighborhood:
- 4x4 and 1x16 ownership regressed Q4_K/Q5_K to roughly 7.3-9.2 ms.
- K64 was neutral for Q4_K and regressed Q5_K to 6.565 ms.
- Disabling local prefetch did not lower the 253-VGPR allocation and was slower.
- Disabling packed-byte prefetch lowered allocation to 234 VGPRs but regressed to 6.186 ms.

The remaining shared-down limit is repeated decode and insufficient shape-local reuse, not an untested tile or spill-removal opportunity.

### Q3: Q6_K small-row geometry

The final Q6_K bodies are:

| M | Retained geometry and layout | Historical packed/BF16 |
| ---: | --- | ---: |
| 64 | M64/N32/K64, 16-BF16 XOR | `5.357/9.863 ms`, `1.84x` |
| 128 | M128/N64/K32, eight-BF16 XOR | `9.084/14.480 ms`, `1.59x` |
| 256 | Two M128/N64/K32 workgroups, packed extraction, eight-BF16 XOR | `11.726/19.010 ms`, `1.62x` |

The M64 128-byte decoded-weight row stride mapped row starts to the same bank phase. The 16-BF16 XOR layout roughly halved latency without increasing LDS. M128 benefited from exact loader ownership. M256 was faster as two smaller workgroups than as the original wider N tile because extra workgroup parallelism and lower accumulator pressure outweighed repeated packed decode.

Two apparently fast M256 N=5/N=7 measurements were invalid because the N tile did not divide the logical 2,048-column result. The corrected N=7 body measured 29.628 ms. K16/K64, M128 N3, M64 N3/N4, and four-/16-BF16 alternatives on the selected 128x64 geometry were rejected. Packed extraction remains selected only for M256.

### B0: source-built HSACO conversion

Dense backward moved from embedded extension entry points to independently compiled concrete gfx1151 wrappers. The initial nine-repeat comparison moved `+0.56%` geometrically. A sequential embedded/bundle/embedded 25-repeat bracket measured the bundle at `+1.12%` geometrically and `+0.93%` by estimated model latency, while the embedded controls themselves drifted `+1.04%`.

Q3_K query showed the largest repeatable placement-sensitive movement, about `2.9-7.0%`, even though no dispatch or device-body algorithm changed. This established the rule used by QB0/QB1: warm standalone modules, compare normalized ISA, and do not infer a semantic regression from raw code-object placement. Bundle construction and packaging details are maintained in `docs/kernel_bundle.md`.

### QB0-QB1: packaged Qwen retune

QB0 found that a runtime `wave < ACTIVE_WAVES` predicate had changed normalized Qwen ISA. Folding it away when `ACTIVE_WAVES == 4` restored the intended control.

QB1 tested only four bounded questions: wide-Q3 packed extraction/prefetch/swizzle, narrow-Q5 extraction, Q6 M256 extraction, and shared-down-Q5 swizzle. Results:
- Wide Q3_K kept packed extraction, two-row prefetch, and eight-BF16 XOR. Removing them regressed by `1.06-11.78%` depending on the control.
- Narrow Q5_K retained scalar extraction only at 2,048 rows, improving `17.41%`. It regressed `2.34%/6.52%` at 8,192/32,768 rows.
- Q6_K M256 kept packed extraction. Scalar extraction regressed `2.14%`.
- Shared-down Q5_K kept swizzle4 because the isolated swizzle8 gain did not survive the complete matrix.

The retained QB1 matrix improved checkpoint-weighted Qwen latency by `0.94%/0.16%/0.63%` relative to QB0 and closed repository-local Qwen tuning.

### DB0: DeepSeek baseline and diagnosis

The generic Q8_0 body used 64x64/reduction-16 ownership, 92 VGPRs, 17 SGPRs, and 2 KiB LDS with no private storage or spills. It was resource-light but ownership-limited:

| Batch | Historical packed/BF16 weighted ms | Throughput ratio |
| ---: | ---: | ---: |
| 1 | `3266.8/767.4` | `0.235x` |
| 4 | `16920.7/3075.7` | `0.182x` |
| 16 | `72146.2/12140.1` | `0.168x` |

Per-family ratios were `0.15-0.30x`. The small tile repeated decode, exposed too little M/N reuse, and collapsed as rows grew. Spills, launcher overhead, and allocation were not the first-order explanation.

### DB1: exact Q8_0 shapes

Eight wrappers covered six full ordinary geometries plus full and bounded LM-head rows. Exact specialization removed runtime shape/bounds/address state while preserving the generic fallback.

The generic/exact/generic bracket improved all 18 ordinary points by `11.32-246.14%`, with a `60.58%` geometric gain. Checkpoint-weighted ordinary latency improved `71.16%/44.23%/41.46%` at B1/B4/B16. LM-head gains ranged from `2.76%` at M32 to more than `130%` at M256/M512.

Full wrappers used 91 VGPRs, 14 SGPRs, and 2 KiB LDS. The bounded LM wrapper used 18 SGPRs. All remained resource-clean.

### DB2: ordinary geometry

DB2 compared G0 64x64/K16, G1 128x64/K32, G2 128x128/K32, and G3 256x64/K32. G2 won all six ordinary families.

| Body | VGPR | LDS | Result |
| --- | ---: | ---: | --- |
| G0 64x64/K16 | 92 | 2 KiB | control |
| G1 128x64/K32 | 118 | 4 KiB | slower on every production family |
| G2 128x128/K32 | 192 | 8 KiB | retained |
| G3 256x64/K32 | 194 | 4 KiB | slower on every shape/batch |

The G0/G2/G0 bracket improved all 18 points by `28.44-130.43%`, with a `77.49%` geometric latency gain. Checkpoint-weighted latency improved `86.99%/96.51%/99.95%` at B1/B4/B16.

Two audit warnings remain important:
- An early screen transposed G1/G3 axes into invalid 64x128/64x256 interpretations. Those results were discarded and rebuilt correctly.
- The generated candidates actually used all-M traversal despite the plan's intended grouped start. Because every geometry shared the same ordering, the geometry result remains valid. DB7/DB8 later isolated traversal.

### DB3: Q8_0 lowering and LDS padding

Generated ISA already carried affine reduction state and emitted `global_load_b128` for each width-16 Q8 payload. An explicit source vector-loader or address-carry rewrite would not add a new mechanism. Paired LDS-fragment prefetch was neutral/slower (`-1.26%/-0.74%/+0.11%` on Q-A B1/B4/B16) and was rejected.

The unpadded G2 profile reported `79.17%` LDS bank conflict on Q-A, Q-B, and output-B. On Q-A B1, padding8 changed the K32 row stride from 64 to 80 bytes and reduced:
- bank conflict from `79.17%` to `58.33%`.
- derived LDS latency from about 585 to 245 cycles.
- ALU stall from LDS from `24.19%` to `15.26%`.

The complete bracket retained padding for Q-A and shared gate/up through 8,192 rows, Q-B/output-B/shared-down at 2,048 rows, and KV at every production row count. Retained gains were `6.17-31.69%`. Shared-down B4 rejected padding after a `2.63%` regression. Width32 remained closed because the selected ISA already vector-loads the payload and repeated scale loads were not the measured limit.

### DB4-DB6: LM geometry and complete-loss selection

DB4 retained active-two-wave M32, G0 M64, G1 M128, and G3 M256/M512. M32 keeps all waves for decode/barriers but limits cotangent loads, WMMA, and stores to two waves. M128 improved `19.48%`. M256/M512 improved `51.85%/60.64%`. Selected bodies used 91-194 VGPRs and 2-4 KiB LDS with no private storage or spills.

Historical per-call backward latency was `8.297/7.459/7.569/10.386/23.206 ms` at M32/M64/M128/M256/M512. Isolated backward favored M256, but DB6 selected from the complete loop:

| DeepSeek chunk | B1 complete ms | B1 peak MiB | B4 complete ms | B4 peak MiB |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 721.928 | 32.32 | 2929.059 | 80.35 |
| 64 | 332.749 | 48.79 | 1350.252 | 96.82 |
| 128 | 219.147 | 81.57 | 895.870 | 129.60 |
| 256 | 175.808 | 147.45 | 719.764 | 195.48 |
| 512 | 173.583 | 274.76 | 705.975 | 322.79 |

M512 was retained. The full five-chunk B16 sweep was intentionally stopped because of cost. A selected-M512 capacity run measured `2775.945 ms`, 514.91 MiB allocation, and 532 MiB reservation. DB5 paired grad-input remained deferred because it requires a model-owned API and autograd integration.

### DB7: Q-B and output-B traversal

DB7 changed only grouped-M launch traversal on the G2 body. M2 was retained for Q-B and output-B above 2,048 rows:
- Q-B B4/B16 fell from `51.419/194.356 ms` to `31.482/116.296 ms`, reductions of `38.77%/40.16%`.
- Output-B B4/B16 fell from `45.779/187.065 ms` to `26.537/108.106 ms`, reductions of `42.03%/42.21%`.

M4 was slower. M1 and M2 were within `0.75%`. M2 was retained from its stronger direct all-M brackets. Q-B L2 hit rate rose `9.6% -> 13.8%` and occupancy `12.0% -> 21.1%`. Output-B changed `17.8% -> 54.5%` L2 and `12.5% -> 23.2%` occupancy.

DB7 improved weighted throughput to `0.972x/0.737x/0.658x`, but four families and Q-B B1 still had untested traversal, motivating DB8.

### DB8: complete grouped-M traversal

DB8 preserved geometry, padding policy, arithmetic, reduction order, active waves, decoder width, prefetch, buffering, and K depth. It changed only `GROUP_M` and the launch-grid mapping.

All nine M2 targets exceeded the 1% screen threshold:

| Family | Batch | all-M before | M2 | all-M after | M2/control midpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q-A | 4 | 4.468 | 2.554 | 4.649 | `0.560x` |
| Q-A | 16 | 32.386 | 13.184 | 32.368 | `0.407x` |
| Q-B | 1 | 8.179 | 7.475 | 7.673 | `0.943x` |
| KV | 4 | 3.278 | 1.338 | 3.305 | `0.406x` |
| KV | 16 | 23.194 | 5.253 | 23.211 | `0.226x` |
| Shared gate/up | 4 | 9.361 | 5.656 | 9.497 | `0.600x` |
| Shared gate/up | 16 | 54.401 | 27.955 | 54.583 | `0.513x` |
| Shared down | 4 | 12.718 | 7.070 | 12.272 | `0.566x` |
| Shared down | 16 | 49.452 | 27.961 | 49.479 | `0.565x` |

M1 was used only as a tie-breaker. Seven non-Q-B points were within 1% of M2. Shared gate/up B4 was `1.36%` slower. Q-B B1 was the exception: the M2/M1/M2 bracket was `7.241/6.923/7.509 ms`, so M1 won by `6.1%` against the control midpoint.

Normalized disassembly was `98.16-98.83%` opcode-identical to the all-M controls. Every variant retained 10 `global_load_b128`, 32 `ds_load_b128`, and 32 BF16 WMMA instructions. Only two to five mapping-prologue instructions changed.

Matched profiles identified locality as the first-order mechanism:

| Family | L2 hit all-M/winner | Occupancy all-M/winner | ALU stalled by LDS | LDS latency cycles |
| --- | ---: | ---: | ---: | ---: |
| Q-A B16 | `33.3%/80.7%` | `44.9%/49.6%` | `0.49%/2.12%` | `434/658` |
| Q-B B1 | `27.3%/57.9%` | `38.8%/38.4%` | `13.27%/19.52%` | `173/184` |
| KV B16 | `37.4%/76.7%` | `43.5%/49.5%` | `0.39%/1.35%` | `301/248` |
| Shared gate B16 | `26.6%/73.1%` | `42.7%/49.6%` | `0.39%/1.98%` | `388/635` |
| Shared down B16 | `18.6%/57.2%` | `46.2%/49.3%` | `0.67%/3.67%` | `356/615` |

L2 hit rate rose by 30.6-47.4 percentage points. LDS stalls and latency often increased, but the shorter packed-weight/cotangent reuse window dominated. `MemUnitBusy` could not be collected because rocprofv3 rejects its non-windowable `TA_TA_BUSY` dependency under dispatch-windowed gfx1151 collection.

DB8 reduced weighted packed latency by `4.51%/18.85%/26.41%` from DB7 and closed repository-local DeepSeek MMQ backward work. Its seven wrappers were appended as bundle IDs 172-178, preserving all established IDs.

## Closed directions and durable reasoning

| Direction | Decision and reason |
| --- | --- |
| Larger wave counts | Rejected or invalid ownership. Four wave32 waves remain the dense foundation |
| 2x2, 3x6, 4x4, 1x16 ordinary geometry | Lost reuse, parallelism, or cotangent traffic tradeoffs |
| Broad `GROUP_M=2/4` | Traversal is geometry and shape dependent. Retain only exact measured DeepSeek branches |
| M4 traversal | Slower than M2 in DB7/DB8 screens |
| Width-8 ordinary decode | Duplicated metadata work and lost to width16 |
| Width-32 Q8_0 decode | Grouped controls lost 2.4-4.3%. Local ISA already uses wide payload loads |
| Ordinary K64 | Longer decode/live ranges outweighed fewer barriers. Retain only Q6 M64's measured body |
| Cross-iteration packed prefetch | Longer VGPR lifetime regressed Q3_K |
| Compiler-managed prefetch arrays | Risk unintended LDS/private storage. Explicit scalar/vector state is required |
| Complete metadata vector loads | Did not improve Q4_K and changed scheduling unfavorably |
| Global LDS padding or swizzle | Layout is quant-, shape-, and geometry-specific. Conflict percentage alone is not a selector |
| Two LDS buffers or decoded-weight LDS cache | Extra LDS/residency cost lost in dense and grouped controls |
| Custom LDS-only barriers | Neutral or slower. Simpler synchronization retained |
| Shared-down 4x4/1x16/K64 | Direct controls regressed or were neutral. Representation is the remaining lever |
| Q6 invalid N5/N7 tiles | Out-of-bounds full-tile assumptions produced invalid timing. Exact divisibility is mandatory |
| Split-K, GSU, Stream-K, persistent workgroups | No K-split/fixup mechanism or regular-grid deficit justifies them |
| DirectToLds/DirectToVgpr | These paths do not perform cooperative GGUF reconstruction and sharing |
| Environment or online autotuning | Production dispatch remains static and explainable |
| Raw code-object offsets/order | Byte-identical controls drift. Only semantic/normalized-ISA changes are evidence |
| Hidden prepared-weight/activation caches | Lifetime, invalidation, stream, and memory ownership must be model-visible |

### Portability evidence

| Source | Durable conclusion |
| --- | --- |
| llama.cpp PR #21344 | gfx1151 benefits from workload-specific 128-thread ownership. Do not import RDNA4 256-thread/eight-wave settings |
| llama.cpp PR #21527 | Lossless Q8_0 payload/scale separation is the strongest future representation experiment |
| llama.cpp PR #22051 and PR #21168 | No missing local vector-load mechanism. Selected ISA already emits wide global/LDS loads |
| llama.cpp PR #22298 | Stream-K is inapplicable because this backward has no K split or fixup |
| Composable Kernel PR #2663 | Prepared quant layouts are credible, but the published architectures and quant families differ |
| hipBLASLt PR #539 | XCC remapping does not apply to single-XCC gfx1151. Locality is already covered by DB7/DB8 |
| DwarfStar | Explicit model-owned Q8 preparation and paired projections are useful ownership evidence. Its static cache and forward pair are not this backward contract |

Architecture transfer rules:
- Treat RDNA3/RDNA4 results as mechanism evidence only. Do not assume gfx12 scheduler/cache behavior applies to gfx1151.
- Keep producer-side cooperative GGUF decode. Consumer-side decode repeats metadata work in every M wave.
- Treat traversal, geometry, K depth, decoder width, and LDS layout as one measured configuration.
- Prefer controlled ablations and event timing over PC-sampling percentages or instruction count alone.
- Use a lossless integer-plus-scale representation before considering approximate int8 arithmetic.

## Scope and contracts

Included:
- BF16 cotangents and BF16 input gradients.
- Packed GGUF Q3_K, Q4_K, Q5_K, Q6_K, and Q8_0 production compatibility.
- Qwen's 160 ordinary projections and Q6_K head.
- DeepSeek's 301 ordinary Q8_0 calls and Q8_0 head.
- Sequence length 2,048 at physical batches 1, 4, and 16.
- Direct packed-weight execution with static host-visible dispatch.

Excluded:
- Production dense IQ2_S expert weights; they are covered by grouped/routed backward, documented in `docs/grouped_mmq_bwd_optimization.md`.
- Grouped/routed expert backward, documented in `docs/grouped_mmq_bwd_optimization.md`.
- GatedDeltaNet layout permutations.
- LoRA GEMMs and residual accumulation.
- Model-scheduler ownership and public operator-schema changes.
- Changes under `csrc/vendor/llama_cpp`.
- Direct fused-kernel linkage against hipBLASLt.

Hardware and toolchain:

```text
GPU: Radeon 8060S Graphics
architecture: gfx1151, RDNA3.5, wave32, 40 CUs
LDS limit: 64 KiB per workgroup, 128 KiB per WGP
VGPR capacity: 1536 per SIMD
PyTorch: 2.12.0+rocm7.15.0a20260701
HIP: 7.14.60850
```

The reference is `torch.mm` with the same BF16 cotangent and the authoritative production GGUF weight independently dequantized to BF16. The existing HIP dense IQ2_S implementation remains available as decoder and dispatch reference code, but IQ2_S is intentionally absent from the dense production test inventory. Correctness requirements include independent GGUF references, exact one-hot row decode where applicable, direct grad-input comparison, autograd, exact-tile guards, and row-boundary coverage.

Qwen ordinary workload:

| Family/type | Forward weight `(N,K)` | Checkpoint tensors |
| --- | ---: | ---: |
| Query Q3_K / Q4_K | `(8192,2048)` | 9 / 1 |
| Narrow Q3_K / Q4_K / Q5_K | `(512,2048)` | 9 / 70 / 21 |
| Attention output Q4_K | `(2048,4096)` | 10 |
| Shared down Q4_K / Q5_K | `(2048,512)` | 30 / 10 |
| LM head Q6_K | `(248320,2048)` | 1 |

DeepSeek ordinary workload:

| Family | Forward weight `(N,K)` | Checkpoint tensors |
| --- | ---: | ---: |
| Q-A | `(1024,4096)` | 43 |
| Q-B | `(32768,1024)` | 43 |
| KV | `(512,4096)` | 43 |
| Output-B | `(4096,8192)` | 43 |
| Shared gate/up | `(2048,4096)` | 86 |
| Shared down | `(4096,2048)` | 43 |
| LM head | `(129280,4096)` | 1 |

Dispatch may use quant type, exact matrix shape, token count, and public rows. It must not use pointer identity, tensor version, hidden stream state, or online autotuning.

## Measurement and validation

Measurement rules:
- Run GPU benchmarks and profilers sequentially.
- Run correctness validation before benchmarks.
- Use real nonzero tensors and warm packaged modules before timing.
- Use three warmups and nine repeats for complete matrices.
- The public benchmark prepares the forward graph before timing and measures only `torch.autograd.grad` against the direct `torch.mm` gradient baseline.
- The direct-kernel benchmark uses preallocated outputs and invokes GGTensile and HIP without PyTorch autograd.
- Bracket any fresh median movement above 1% with sequential 25-repeat control/candidate/control runs.
- Inspect normalized disassembly and VGPR/SGPR/LDS/private/spill/stack metadata.
- Require zero private storage, zero VGPR/SGPR spills, and no dynamic stack for retained production bodies.

The public and direct-kernel surfaces are separate scripts:

```bash
PYTHONPATH=. python bench/benchmark_mmq_bwd_api.py \
  --model /path/to/model.gguf --model-family qwen \
  --warmup 3 --repeats 9 \
  --output ~/tmp/torch-ggml-ops/mmq_bwd_api.json

PYTHONPATH=. python bench/benchmark_mmq_bwd_kernels.py \
  --model /path/to/model.gguf --model-family qwen \
  --warmup 3 --repeats 9 \
  --hip-root build/mmq_hip_controls/gfx1151 \
  --output ~/tmp/torch-ggml-ops/mmq_bwd_kernels.json
```

Complete packed-loss measurements remain downstream application benchmarks rather than a third core MMQ benchmark surface.

Build and validation:

```bash
python tools/build_mmq_bundle.py --force --jobs "$(nproc)"
PYTHONPATH=. pytest -q
ruff check .
python -m compileall -q bench tools torch_ggml_ops tests
python tools/build_mmq_bundle.py --check
git diff --check
```

Final status:
- `100 passed, 14 warnings` from the complete project suite.
- Bundle freshness reports 179 current kernels.
- All 83 MMQ backward artifacts pass resource gates.
- Independent ccache-bypassed reproducibility passes for all 179 artifacts.
- The installed extension was rebuilt against the final selector.
- Generated HSACOs remain ignored by Git and excluded from source distributions. Local wheels may contain verified artifacts.

The DeepSeek tests cover independent one-hot decode, random direct grad-input, autograd for all dense tensor families, and a 65-row launch-grid boundary. Final Qwen LM-head NRMSE remained approximately `4.9e-4` at M64, `4.7e-4` at M128, and `5.2e-4` at M256.

## Artifact index

### Latest acceptance

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_narrow_q5_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_post_db8_control_9.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_db6_complete_loss_control_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db8_final_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db6_complete_loss_b1_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db6_complete_loss_b4_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db6_complete_loss_b16_m512_capacity_3.json
```

### Qwen historical and QB1 controls

```text
~/tmp/torch-ggml-ops/mmq_bwd_baseline_primary_sequential.json
~/tmp/torch-ggml-ops/mmq_bwd_final_full_v3.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb0_folded_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_selected_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_scalar_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_no_prefetch_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_no_swizzle_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_selected_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_narrow_selected_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_narrow_scalar_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_narrow_selected_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_packed_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_scalar_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q6_m256_packed_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_shared_swizzle4_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_shared_swizzle8_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q5_shared_swizzle4_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_qb1_control_9.json
```

### DeepSeek DB0-DB4 controls

```text
~/tmp/torch-ggml-ops/mmq_bwd_ds4_q8_correctness.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p0_baseline_9.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_pre_ds4_control_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_exact_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_generic_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_exact_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p1_generic_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_post_ds4_p1_control_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p2_corrected_g1_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p2_corrected_g3_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p2_g0_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p2_corrected_selected_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p2_corrected_g0_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p3_qa_prefetch_control_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p3_qa_prefetch_candidate_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p3_qa_prefetch_control_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p3_padding_selected_control_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p3_padding_selected_candidate_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p3_padding_selected_control_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p3_final_9.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_post_ds4_p3_control_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p4_lm_g1_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p4_lm_g2_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p4_lm_g3_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p4_control_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p4_selected_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_p4_control_after_25.json
```

### DeepSeek DB6-DB8 controls

```text
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db6_complete_loss_b1_screen_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db6_final_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_qb_m2_screen_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_qb_m4_screen_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_output_m2_screen_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_output_m4_screen_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_qb_control_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_qb_m2_candidate_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_qb_control_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_output_control_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_output_m2_candidate_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_output_control_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_m2_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_m1_candidate_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_m2_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_group_m_final_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_post_group_m_control_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db8_group_m_control_screen_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db8_group_m_candidate_screen_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db8_group_m_control_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db8_group_m_m2_candidate_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db8_group_m_control_after_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db8_group_m_m1_screen_9.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db8_qb_m2_control_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db8_qb_m1_candidate_25.json
~/tmp/torch-ggml-ops/mmq_bwd_ds4_db8_qb_m2_control_after_25.json
```

### Profiler and packaging evidence

```text
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-db2-qa
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-db2-qb
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-db2-output
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-db3-qa-padding8
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-group-m-qb-control
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-group-m-qb-m2
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-group-m-output-control
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-group-m-output-m2
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-db8-qa-control
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-db8-qa-m2
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-db8-qb-control
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-db8-qb-m1
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-db8-kv-control
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-db8-kv-m2
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-db8-shared-gate-control
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-db8-shared-gate-m2
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-db8-shared-down-control
~/tmp/torch-ggml-ops/rocprof-mmq-bwd-db8-shared-down-m2
~/tmp/torch-ggml-ops/mmq_bwd_pre_bundle.json
~/tmp/torch-ggml-ops/mmq_bwd_post_bundle.json
~/tmp/torch-ggml-ops/mmq_bwd_embedded_pre_control_25.json
~/tmp/torch-ggml-ops/mmq_bwd_bundle_control_25.json
~/tmp/torch-ggml-ops/mmq_bwd_embedded_post_control_25.json
```

## Tool notes

- PC sampling heavily perturbs short kernels. Use it qualitatively.
- One multi-counter Q6_K M256 run caused an HSA memory fault and queue-sync timeouts. Collect one counter at a time for that body.
- One post-profiler process reported a transient `hipErrorLaunchFailure`. Focused and complete reruns passed without source changes.
- `roc-obj-ls` is broken in the active environment because of a `rocm_sdk_core._cli` import error.
- Inspect code objects through `.hip_fatbin`, `clang-offload-bundler`, `llvm-readobj`, `llvm-nm`, and `llvm-objdump`.
- The complete bundle and packaging contract is documented in `docs/kernel_bundle.md`.
