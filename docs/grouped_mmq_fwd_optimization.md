# Grouped MMQ forward optimization

## Status at a glance

This document is the source of truth for grouped MMQ forward kernel behavior, dispatch, benchmark evidence, and retuning decisions on gfx1151. It covers:
- `grouped_mmq_pair` for routed gate/up projections.
- `grouped_mmq` for routed down projections.
- `fixed_grouped_mmq` for the eight-group DeepSeek output-A projection.

Dense MMQ forward and backward are documented separately in `docs/mmq_fwd_optimization.md` and `docs/mmq_bwd_optimization.md`. The generic HSACO bundle and loader contract are documented in `docs/kernel_bundle.md`.

The local kernel pass is complete for the current Qwen and DeepSeek packed representations. The retained dispatch is exact and resource-clean for all enforced production entries. Backward reduced-precision controls closed the approximate-accumulator prerequisite before forward implementation. Prepared-weight work remains outside this repository-local pass.

Current source-of-record matrices use the target-specific tuned AITER configurations:

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_aiter_tuned_9.json
~/tmp/torch-ggml-ops/grouped_mmq_fwd_ds4_aiter_tuned_9.json
```

They use warmup 3, 9 sequential repeats, correctness rows 256, and the consolidated MMQ bundle.

| Model | Matrix | Packed-reference checks | Packed wins | Remaining packed losses |
|---|---:|---:|---:|---|
| Qwen | 60 points | 84/84 exact | 45/60 vs BF16 AITER | IQ2_S gate/up at B16 and IQ2_S down except B1 uniform |
| DeepSeek-V4-Flash | 27 points | 39/39 exact | 12/24 routed points vs AITER | Gate/up at B16, down at B4/B16, and fixed Q8_0 output A |

Qwen still wins every Q3_K gate/up and Q4_K/Q5_K down point. The tuned reference exposes B16 IQ2_S gate/up and broader IQ2_S down deficits. DeepSeek wins all routed B1 points and all gate/up B4 points, but AITER wins all routed B16 points and all down B4 points. Fixed Q8_0 remains slower than BF16 BMM.

## Retained production dispatch

Dispatch is static and explainable. It depends on quant type, operator kind, exact production geometry, and a coarse total-row threshold. It does not inspect device offsets on the host, use route names, use environment variables, or perform online autotuning.

Here `rows` is the total routed row count and `num_groups` is the number of route groups. Thresholds are launch hints. The device kernels still handle inactive experts, skew, and partial row tiles correctly.

| Workload | Exact geometry | Retained branch | Threshold |
|---|---|---|---|
| Qwen gate/up | `(N,K)=(512,2048)` | serial J64 for small groups. Device row-task J64 for large groups | row tasks when `rows >= 128 * num_groups` |
| Qwen down Q5_K | `(N,K)=(2048,512)` | standalone serial J32 for small groups. Serial J64 otherwise | J32 when `rows < 128 * num_groups` |
| Qwen down IQ2_S | `(N,K)=(2048,512)` | serial J64 with a J32 final-tail body for small groups. Serial J64 otherwise | mixed tail when `rows < 128 * num_groups` |
| Qwen down other production types | `(N,K)=(2048,512)` | serial J64 | no extra branch |
| DeepSeek output A | eight groups, `(N,K)=(1024,4096)` | fixed-group J64 | always |
| DeepSeek gate/up | `(N,K)=(2048,4096)` IQ2_XXS | serial J64 for small/medium groups. Serial J80 for large groups | J80 when `rows >= 512 * num_groups` |
| DeepSeek down | `(N,K)=(4096,2048)` Q2_K | serial J32 with factor-4 scale/min unrolling. Optional J16 tail | J16 tail when `rows < 64 * num_groups` |
| Other supported shapes | runtime geometry | bounds-safe generic J128 fallback | none |

The common retained tile is `I=64`, with 128 threads and four wave32 waves. Qwen and DeepSeek exact branches specialize their production N/K and packed K-block counts at compile time. The fixed Q8_0 path uses a full-N J64 body when the output shape is complete and a bounded variant only for non-multiple output widths.

Gate/up row tasks are built once on the device and reused for the two projections. Each task stores an expert ID and a bounded row interval. Batch-1-sized Qwen groups retain serial ownership to preserve sparse launch behavior. Down retains serial expert row ownership because its many output tiles already expose enough parallelism. The down descriptor experiment regressed.

## Production contract

The public activation and output dtype is BF16. Activations are quantized internally to Q8_1 for the packed arithmetic path. Packed GGUF weights remain authoritative. The production path does not construct logical BF16 expert matrices.

The route metadata ABI is:
- `expert_indices`: contiguous CUDA `torch.int64`, shape `[G]`.
- `expert_offsets`: contiguous CUDA `torch.int32`, shape `[G]`.
- `expert_offsets[-1] = R`.
- `G <= 256`.

The timed path must not call `.item()`, copy offsets to the host, build CPU descriptors, or introduce an implicit synchronization. Complete public-operator timing includes Q8_1 quantization and workspace allocation. Launcher, loader, module-load, and host dispatch overhead are outside optimization scope.

### Qwen workload

The Qwen checkpoint is `Qwen3.6-35B-A3B-APEX-I-Mini.gguf`, with sequence length 2,048 and top-k 8.

| Case | Logical expert shape `(N,K)` | GGUF type | Layers | Operator |
|---|---:|---|---:|---|
| Gate/up outer | `(512,2048)` | Q3_K | 20 | paired routed |
| Gate/up middle | `(512,2048)` | IQ2_S | 20 | paired routed |
| Down outer edge | `(2048,512)` | Q5_K | 2 | routed single |
| Down outer main | `(2048,512)` | Q4_K | 18 | routed single |
| Down middle | `(2048,512)` | IQ2_S | 20 | routed single |

| Physical batch | Routed rows | Uniform rows per expert |
|---:|---:|---:|
| 1 | 16,384 | 64 |
| 4 | 65,536 | 256 |
| 16 | 262,144 | 1,024 |

Gate and up share one Q8_1 activation workspace but execute as two grouped arithmetic launches. Batch 1 commonly has fewer than 256 active experts. Batches 4 and 16 generally activate all experts with skewed group sizes.

### DeepSeek-V4-Flash workload

The DeepSeek checkpoint is `DeepSeek-V4-Flash-IQ2XXS.gguf`, with sequence length 2,048 and top-k 6.

| Case | Groups/experts | Logical shape `(N,K)` | Physical packed shape | GGUF type | Operator |
|---|---:|---:|---:|---|---|
| Output A | 8 fixed groups | `(1024,4096)` | raw `(8192,4352)`, view `(8,1024,4352)` | Q8_0 | fixed grouped |
| Gate/up | 256 experts | `(2048,4096)` | `(256,2048,1056)` per projection | IQ2_XXS | paired routed |
| Down | 256 experts | `(4096,2048)` | `(256,4096,672)` | Q2_K | routed single |

| Physical batch | Token rows M | Routed rows R | Uniform rows per expert |
|---:|---:|---:|---:|
| 1 | 2,048 | 12,288 | 48 |
| 4 | 8,192 | 49,152 | 192 |
| 16 | 32,768 | 196,608 | 768 |

Output A consumes logical input `[..., 8, 4096]` and preserves eight independent logical weights. It uses token rows M, not routed rows R, and includes the group-major to public token-major output conversion in the BF16 reference. The routed pair shares one Q8_1 workspace. DeepSeek backward decoders are now implemented and correctness-tested. Their production dispatch and optimization remain in `docs/grouped_mmq_bwd_optimization.md`.

## Measurement and acceptance

`bench/benchmark_grouped_mmq_fwd.py` records complete packed latency, logical throughput, incremental allocation, packed shapes, quant type, route statistics, reference latency, dense-MMQ exactness, independent-reference error, and checkpoint-weighted estimates.

Routed references are BF16 AITER Triton GMM with the benchmark-owned `bench/aiter_gmm_heuristics.py` exact configuration table. Dispatch keys include total routed rows, K, N, and RHS layout. Unsupported shapes fail closed. Forward uses the transposed logical-weight view. Active weights are independently dequantized during setup, not in the timed reference. The fixed output-A reference is BF16 `torch.bmm` over eight groups and includes public-layout conversion. AITER and BMM are references, not a claim about the maximum possible packed throughput.

The four deterministic routing distributions are:
- `uniform`: all 256 experts active with equal group sizes.
- `skewed`: all 256 experts active with deterministic nonuniform sizes.
- `sparse`: 192, 224, or 240 active experts at batches 1, 4, and 16.
- `boundary`: includes group sizes around 1, 16, 64, and 128 to exercise tails.

DeepSeek uses the same distribution families with top-six mean sizes 48, 192, and 768. The fixed output-A case has no route distribution.

A retained change must satisfy:
- exact grouped-versus-dense packed BF16 output for every production point.
- the established independent-reference error envelope.
- zero private segment, zero VGPR/SGPR spills, and no dynamic stack for every enforced production arithmetic entry.
- no repeatable Qwen regression in the complete 60-point matrix.
- no regression in the checkpoint-weighted Qwen estimate.
- complete-operator timing, including quantization and workspace allocation.
- device-resident routing metadata and current-stream behavior.
- dense controls as well as grouped controls when shared quantization or MMQ code changes.

A movement above 1% in a fresh median triggers a sequential 25-repeat A/B control. Concurrent benchmark or profiler runs are not accepted.

The current matrices can be reproduced with:

```bash
PYTHONPATH=. python bench/benchmark_grouped_mmq_fwd.py \\
  --model ~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf \\
  --model-family qwen --warmup 3 --repeats 9 --correctness-rows 256 \\
  --output ~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_aiter_tuned_9.json

PYTHONPATH=. python bench/benchmark_grouped_mmq_fwd.py \\
  --model ~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf \\
  --model-family deepseek --warmup 3 --repeats 9 --correctness-rows 256 \\
  --output ~/tmp/torch-ggml-ops/grouped_mmq_fwd_ds4_aiter_tuned_9.json
```

## Latest evaluation

### Qwen

The latest artifact is `~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_aiter_tuned_9.json`. It contains 60 points from five cases, three physical batches, and four distributions. The table below shows uniform routing for compact comparison. The win count covers all four distributions.

| Case | Batch 1 packed/reference ms | Batch 4 packed/reference ms | Batch 16 packed/reference ms | Wins |
|---|---:|---:|---:|---:|
| Gate/up Q3_K | `3.825/8.471` | `14.759/21.595` | `58.152/60.225` | 12/12 |
| Gate/up IQ2_S | `4.497/8.471` | `17.427/21.596` | `68.014/60.144` | 8/12 |
| Down IQ2_S | `2.316/2.854` | `7.270/6.138` | `27.484/24.242` | 1/12 |
| Down Q4_K | `1.611/2.851` | `5.920/6.217` | `23.674/24.526` | 12/12 |
| Down Q5_K | `1.793/2.862` | `6.012/6.201` | `23.486/24.363` | 12/12 |

The historical acceptance artifact matched dense packed MMQ exactly at all 84 Qwen pair/single checks. Current IQ2_S correctness coverage instead uses grouped single/pair consistency and an independently dequantized BF16 reference; dense IQ2_S is retained only as reference code. Independent BF16-reference NRMSE is `0.00597-0.01653` across the matrix, consistent with the quantization-specific envelopes. Q3_K gate/up and Q4_K/Q5_K down win all 36 points. IQ2_S gate/up wins B1/B4 and loses all four B16 points. IQ2_S down wins only B1 uniform.

Across the four distributions, checkpoint-weighted packed/reference speedup ranges are:

| Physical batch | Packed/reference speedup range |
|---:|---:|
| 1 | `1.540-1.846x` |
| 4 | `1.125-1.224x` |
| 16 | `0.954-0.973x` |

The estimate covers checkpoint projection-call counts and two checkpointed executions per optimizer step. It excludes routing, activation functions, LoRA, and unrelated model work.

### DeepSeek-V4-Flash

The latest artifact is `~/tmp/torch-ggml-ops/grouped_mmq_fwd_ds4_aiter_tuned_9.json`. It contains 27 points: three fixed output-A points, 12 routed gate/up points, and 12 routed down points. The table shows uniform routing for the routed cases.

| Case | Batch 1 packed/reference ms | Batch 4 packed/reference ms | Batch 16 packed/reference ms | Wins |
|---|---:|---:|---:|---:|
| Fixed output A Q8_0 | `10.957/8.413` | `44.351/33.156` | `175.233/128.155` | 0/3 vs BMM |
| Gate/up IQ2_XXS | `33.498/54.272` | `81.320/126.065` | `321.699/314.718` | 8/12 vs AITER |
| Down Q2_K | `18.151/25.873` | `68.101/51.498` | `278.992/141.815` | 4/12 vs AITER |

All 39 fixed and paired projection checks are exact against dense packed MMQ. Independent BF16-reference NRMSE is `0.00602-0.01140`. Including fixed output A, checkpoint-weighted packed/reference ratios are `1.403-1.632x` at batch 1, `0.964-1.087x` at batch 4, and `0.708-0.759x` at batch 16. Fixed Q8_0 remains `0.73-0.77x` the BMM reference.

The earlier isolated accepted DeepSeek artifact and same-build Qwen control remain useful for code-object provenance:

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_ds4_final_isolated_full.json
~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_post_ds4_isolated_final.json
```

They are superseded as current source-of-record timing by the tuned-reference artifacts above.

### Resource status

The complete bundle contains 179 source-built HSACOs. All enforced production resource gates pass with zero private bytes, zero VGPR/SGPR spills, and no dynamic stack. The broad all-artifact scan still reports known fallback spills. Those fallback artifacts are not part of the production zero-spill contract.

Representative retained arithmetic allocations from the retuning pass are:

| Branch | VGPR | SGPR | Private bytes | Spills | Dynamic stack |
|---|---:|---:|---:|---:|---|
| Fixed Q8_0 J64 | 213 | 48 | 0 | 0 | no |
| DeepSeek IQ2_XXS J64 | 229 | 77 | 0 | 0 | no |
| DeepSeek IQ2_XXS J80 | 253 | 51 | 0 | 0 | no |
| DeepSeek Q2_K J32 | 208 | 38 | 0 | 0 | no |
| DeepSeek Q2_K J32/J16 | 213 | 43 | 0 | 0 | no |
| Qwen Q5_K standalone J32 | 189 | 32 | 0 | 0 | no |

The Qwen row-task and Qwen down J64 entries are also enforced and spill-free. Exact per-entry resource metadata is generated with the bundle and is not a dispatch input.

## Remaining bottleneck and work boundary

No additional local schedule or accumulator change is justified for the current packed representations. Backward established that direct native BF16-C is numerically invalid for long reductions, K32/K64 FP32 slabs increase resources and latency, and normalized FP16-C remains 33.3% slower even when scale calculation is removed. Forward approximate accumulation, wider J tiles based on reclaimed registers, and static persistent scheduling are therefore closed at their prerequisite.

Reopen that sequence only after a new arithmetic mechanism demonstrates lower register use, acceptable real-weight and dynamic-range error, zero private storage and spills, and better public backward latency in an existing N64 body. Forward must then measure its existing geometry first, preserve integer MMQ and Q8_1 workspace semantics, and test only DeepSeek Q2_K J64, DeepSeek IQ2_XXS J128, or Qwen IQ2_S J128 where the latest B16 results justify it. Static grids from `{20,40,80,160,256}` come last.

The tuned reference exposes representation deficits in Qwen IQ2_S down, Qwen IQ2_S gate/up at B16, DeepSeek Q2_K at B4/B16, and DeepSeek IQ2_XXS gate/up at B16. Partial routed tiles repeat packed metadata interpretation, scale formation, packed loads, and BF16 decoded-weight construction. The retained kernels have already removed unnecessary row decomposition and bounded-tail work, while AITER begins from predecoded BF16 weights.

DeepSeek fixed Q8_0 is also representation-limited. The complete public operator remains slower than BMM because the comparison begins from already dequantized BF16 weights. Quantization is not the dominant cost: accepted traces put multiplication at roughly 87-99% of retained operator kernel time depending on family and batch.

The only worthwhile follow-ups are representation-level designs:
- a compact lossless decoded-weight cache substantially smaller than BF16.
- cross-call decoded-weight reuse.
- a transient project-owned decoded dense stage amortized across projections or calls.
- a persistent lossless integer-plus-scale representation consumed directly by grouped WMMA.

Any such project must account for weight lifetime, memory footprint, cache invalidation, routing sparsity, allocation, and complete public-operator latency. It must not become an unconditional full BF16 shadow copy. The current final matrices are the baseline for that work.

Closed directions for the current representation:
- global J64, J128, or I128 rules.
- eight-wave ownership of an I64 output tile without a valid reduction design.
- complete decoded-weight LDS caching at the current 64-row tile.
- down row-task descriptors or fixed-size persistent traversal.
- split full/tail launches or replicated tail arithmetic.
- compiler-managed local arrays as packed prefetch state.
- wholesale direct-to-VGPR or both-operands direct-to-VGPR.
- two-LDS pipelines, GSU, split-K, and grouped Stream-K.
- broad swizzle/padding sweeps without measured LDS-bank evidence.
- another shared quantizer rewrite while multiplication remains dominant.

## Historical optimization log

The entries below are ordered by experiment rather than by the current implementation structure. A rejected entry records the reason it must not be silently retried. Timing values are historical controls and should not be compared directly with the latest tuned-reference artifacts across different builds or instruction-cache states.

### Baseline diagnosis

The original grouped path used runtime shape state and a J128 serial row loop. Its historical artifact was `~/tmp/torch-ggml-ops/grouped_mmq_fwd_baseline_full.json`. The original grouped Q4_K/Q5_K code reached 256 VGPRs with 512-520 private bytes per thread and 127-129 reported VGPR spills. The corresponding dense bodies were spill-free. Historical profiling showed:
- gate/up Q3_K batch 1: 0.829 ms Q8_1 quantization and 10.767 ms for two grouped projections.
- gate/up Q3_K batch 16: 7.894 ms quantization and 122.445 ms for two grouped projections.
- down Q4_K batch 4: 0.916 ms quantization and 21.964 ms grouped arithmetic.

The grouped multiplication was the first-order bottleneck. A pre-G1 down Q4_K counter run reported 18.44% occupancy and 69.73% L2 hit rate versus 23.20% and 71.68% for BF16 AITER. LDS bank stalls were only 0.0154%. This directed the first pass toward compile-time geometry and spill removal rather than cache or LDS-bank tuning.

### G1: compile-time J64 and fixed production shapes

Status: retained.

G1 combined:
- compile-time J64.
- fixed gate/up `(NRowsWeight, BlocksPerWeightRow)=(512,8)`.
- fixed down `(2048,2)`.
- no output-row fallback in exact production bodies.
- fixed production K traversal while retaining J128 fallback kernels for other shapes.

The focused artifact was `~/tmp/torch-ggml-ops/grouped_step1_j64_fixed.json`. Representative improvements over the historical baseline were:

| Point | Baseline ms | G1 ms | Speedup |
|---|---:|---:|---:|
| Gate/up Q3_K batch 1 uniform | 11.185 | 6.170 | 1.81x |
| Gate/up Q3_K batch 16 uniform | 130.408 | 94.200 | 1.38x |
| Down Q4_K batch 1 uniform | 6.874 | 2.842 | 2.42x |
| Down Q4_K batch 4 uniform | 21.977 | 10.599 | 2.07x |
| Down Q4_K batch 16 uniform | 83.663 | 42.110 | 1.99x |
| Down IQ2_S batch 16 uniform | 69.996 | 45.963 | 1.52x |

All focused production bodies became spill-free. This established compile-time fixed-N/K specialization and J64 as the production foundation.

### G2: complete decoded-weight LDS cache for down

Status: rejected and reverted.

G2 decoded both fixed down K blocks into immutable LDS and reused them across serial row tiles. It increased dynamic LDS from 28,928 to 48,384 bytes for Q4_K/Q5_K and from 30,976 to 52,480 bytes for IQ2_S.

| Point | G1 ms | G2 ms | Relative |
|---|---:|---:|---:|
| Down Q4_K batch 4 uniform | 10.599 | 15.427 | 0.69x |
| Down Q4_K batch 16 uniform | 42.110 | 70.334 | 0.60x |
| Down Q5_K batch 4 uniform | 10.688 | 15.503 | 0.69x |
| Down IQ2_S batch 16 uniform | 45.963 | 72.384 | 0.64x |

The extra LDS and residency loss outweighed decode reuse. A complete cache at the same output tile is closed unless a future representation is materially smaller.

### G3: fixed-shape J128

Status: rejected and reverted.

G3 removed runtime N/K state but restored J128. It regressed every focused point by roughly 25-50% relative to G1. Q4_K and Q5_K still reached the register limit with private storage. IQ2_S was spill-free but still lost heavily. J128 is not a production tile for these grouped representations.

### G4: separate exact full-row and bounded-tail bodies

Status: retained.

G4 split full row tiles from final partial tiles. Full tiles use one contiguous Q8_1 activation span and unmasked BF16 stores. Only the final partial tile uses bounded zero fill and masked stores.

| Point | G1 ms | G4 ms | Speedup |
|---|---:|---:|---:|
| Gate/up Q3_K batch 1 uniform | 6.170 | 3.735 | 1.65x |
| Gate/up Q3_K batch 16 uniform | 94.200 | 57.956 | 1.63x |
| Down Q4_K batch 1 uniform | 2.842 | 1.599 | 1.78x |
| Down Q4_K batch 4 uniform | 10.599 | 6.063 | 1.75x |
| Down Q5_K batch 4 uniform | 10.688 | 6.067 | 1.76x |
| Down IQ2_S batch 16 uniform | 45.963 | 32.846 | 1.40x |

The gain established row decomposition and per-load tail control as first-order costs independent of spills.

### G5: remove the post-write row barrier

Status: rejected and reverted.

The preceding barrier already ends the decoded-weight and activation LDS lifetime. Removing the post-write barrier produced only noise-level movement, from 0.975x to 1.004x, in `~/tmp/torch-ggml-ops/grouped_step5_no_write_barrier.json`. The barrier remains for the simpler synchronization structure.

### G6: compile-time two-block down unroll

Status: retained for fixed K=512 down.

G6 emits two explicit calls for down's exact two-block K traversal while keeping gate/up's eight-block loop rolled.

| Point | G4 ms | G6 ms | Speedup |
|---|---:|---:|---:|
| Down Q4_K batch 1 uniform | 1.599 | 1.565 | 1.02x |
| Down Q4_K batch 4 uniform | 6.063 | 5.877 | 1.03x |
| Down Q4_K batch 16 uniform | 24.292 | 23.057 | 1.05x |
| Down Q5_K batch 4 uniform | 6.067 | 5.849 | 1.04x |
| Down IQ2_S batch 16 uniform | 32.846 | 27.102 | 1.21x |

The fixed two-block schedule was retained because it was spill-free and consistently improved complete operator timing.

### G7: pointer-increment gate/up traversal

Status: retained.

G7 replaces repeated block and activation-plane multiplication with affine pointer increments in the rolled eight-block gate/up loop. Q3_K improved by roughly 0.4-1.7% across the focused matrix, and large IQ2_S groups also improved. The important result is the lowering principle: exact full-row bodies should expose affine addresses and bounded lifetimes to the compiler.

### G8: device row-task descriptors for large gate/up groups

Status: retained for gate/up. Rejected for down.

One device setup workgroup computes an atomics-free prefix sum and writes `(expert, row_start, row_end)` descriptors with capacity `ceil(R / 64) + G`. The descriptor arrays remain device-resident. The paired gate/up operator builds them once and reuses them for both projections.

Fair sequential A/B results including setup and complete operator work were:

| Point | Serial ms | Descriptors ms | Speedup |
|---|---:|---:|---:|
| Gate/up Q3_K batch 4 boundary | 16.576 | 16.348 | 1.01x |
| Gate/up Q3_K batch 16 uniform | 57.660 | 56.329 | 1.02x |
| Gate/up IQ2_S batch 16 uniform | 66.966 | 66.314 | 1.01x |

A down descriptor variant regressed Q4_K by 1.8-4.4% and down IQ2_S by 13.9% at the measured large-row points. Down keeps serial row ownership.

### G9: I128, J64, 256 threads

Status: rejected and reverted.

G9 doubled the output tile and workgroup size. It remained spill-free in some cases but increased LDS and reduced workgroup residency/flexibility. Representative relative results were 0.90-0.95x for gate/up Q3_K batch 1, 0.92x for gate/up Q3_K batch 16, 0.87x for down Q4_K batch 4, and 0.85x for down Q5_K batch 4. Wider output reuse does not repay its resource cost on gfx1151.

### G10: 256 threads with I64, J64

Status: invalid and reverted.

G10 changed only the workgroup from four to eight waves. The inherited wave mapping assigned output fragments beyond the logical 64-row output tile, producing overlapping output ownership. Correctness found 15,624 mismatches out of 262,144 elements and NaNs. The implausibly fast timing is not performance evidence. A correct eight-wave design would require a new reduction or output-ownership contract and is outside the current representation pass.

### G11: contiguous bounded activation-tail loads

Status: retained for fixed two-block down.

G11 recognizes that a valid partial Q8_1 row tile is one contiguous integer span. The down tail computes `(j_max + 1) * q8_block_ints` once and uses one `l < valid_activation_ints` predicate, eliminating row division, remainder, source-row reconstruction, and nested row bounds.

The 15-repeat artifact `~/tmp/torch-ggml-ops/grouped_step11b_down_contiguous_tails_15.json` showed 1-6% improvements across selected Q4_K, Q5_K, and IQ2_S tails, including 5-6% on Qwen batch-1 nonuniform IQ2_S down. The gate/up version was neutral and was not retained.

### G12: Qwen IQ2_S dot-loop unroll

Status: rejected.

A Qwen down IQ2_S unroll-1 body remained spill-free and looked slightly better in a first comparison, but isolated 25-repeat controls showed it was 0.41% slower geometrically. The apparent gain came from translation-unit/code-object placement, not loop rolling. Unroll 2 and 4 crossed the spill cliff. The helper was removed.

### G13: same-launch mixed-size tails

Status: retained only in narrow bounded forms.

A literal IQ2_S J64/J16/J32/J48 same-launch body reached 256 VGPRs, 1,644 private bytes, and 455 spills. A narrower Qwen J64/J32 tail body remained spill-free and improved batch-1 nonuniform routes by 2.8-9.0%, but regressed batch-4 and batch-16 uniform controls. It is retained only for `rows < 128 * num_groups`.

The DeepSeek Q2_K J32/J16 body remained spill-free and improved batch 1 by 5.1-15.7%. Large-row controls showed a repeatable regression. It is retained only for `rows < 64 * num_groups`.

### G14: pre-bundle last-version checkpoint

Status: historical packaging checkpoint.

G14 preserved Qwen code-object order and linked the DeepSeek specializations around it. It produced reproducible enough measurements for handoff but made translation-unit and link order part of the observed performance. The associated artifacts were:

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_ds4_last_version_baseline.json
~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_last_version_baseline.json
```

A 25-repeat control showed that unchanged Qwen kernels moved by more than 1% under layout changes. This is why later performance claims use isolated HSACO entries and bracketed controls rather than a monolithic code-object layout.

### G15: standalone gfx1151 kernel bundle

Status: retained and generalized.

Grouped-forward arithmetic was moved into independently compiled HSACOs with deterministic symbols and per-entry resource gates. The first repeated build exposed staging-path-derived compilation-unit IDs. Deterministic per-symbol `-cuid` values made two-directory builds byte-identical. The standalone loader and packaging details now live in `docs/kernel_bundle.md`.

The standalone bundle exposed a repeatable Qwen Q5_K batch-1 nonuniform opportunity. J32 improved skewed, sparse, and boundary routes by approximately 14-16% while regressing uniform by 2.6-3.1%, so dispatch selected J32 only below `rows = 128 * num_groups`. This source-level branch survived the generalized bundle conversion.

The old standalone matrices were:

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_ds4_kernel_bundle_final.json
~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_kernel_bundle_final.json
```

They established the current Qwen/DeepSeek shape-specific direction but are superseded for aggregate timing by the latest tuned-reference artifacts at the top of this document.

## DeepSeek phase log

The DeepSeek work was executed as D0-D4 after the Qwen G-series pass. These entries preserve the shape-specific evidence that led to the final dispatch. They are separate from the later post-bundle Qwen retuning decisions.

### D0: baseline and diagnosis

Status: complete.

The same-session baseline artifacts were `~/tmp/torch-ggml-ops/grouped_mmq_fwd_ds4_baseline_full.json` and `~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_pre_ds4_control.json`, from source checkpoint `aa3ebd4`. The generic routed kernels used J128 and crossed the resource cliff:

| Family | Batch-1 packed/reference ms | Batch-4 packed/reference ms | Batch-16 packed/reference ms | Baseline resource signal |
|---|---:|---:|---:|---|
| Fixed Q8_0 | `10.990/8.162` | `44.336/32.874` | `175.484/129.540` | J64, 213 VGPR, spill-free |
| IQ2_XXS pair | `66.104-77.549/56.055-73.039` | `135.713-148.871/134.315-138.103` | `335.363-400.420/654.188-735.394` | J128, 256 VGPR, 360 private bytes, 117 spills |
| Q2_K down | `62.426-76.312/31.472-40.433` | `155.004-162.014/81.587-82.465` | `421.356-482.628/434.581-477.845` | J128, 256 VGPR, 576 private bytes, 230 spills |

Quantization was only about 9% of fixed Q8_0 batch-1 time and less than 1% of the routed arithmetic cases. D0 therefore prioritized exact-shape arithmetic specialization and deferred shared quantizer changes.

### D1: fixed-group Q8_0 output A

Status: complete. J64 retained.

J128 raised the fixed kernel to 256 VGPRs, 88 private bytes, and 21 spills, and regressed complete latency from `10.990/44.336/175.484 ms` to `15.153/60.563/242.127 ms`. J32 also failed the resource gate at 256 VGPRs, 344 private bytes, and 85 spills. The current J64 tile is the only accepted local tile family.

A direct DwarfStar-style DP4A schedule was rejected because it starts from a different activation and accumulation contract and cannot preserve bitwise equality with dense packed MMQ. A rolled Q8_0 dot loop was neutral to slightly slower (`11.066/44.595/177.793 ms`).

Two representation-local variants were then tested against same-build controls. Corrected Q8_0 scale staging restored exact output and the `0.0061` independent-reference NRMSE envelope at 165 VGPRs, but measured `10.987/44.571/177.887 ms` versus `11.026/44.444/176.981 ms`. Group-major Q8_1 workspace repacking was exact and measured `11.001/44.464/177.121 ms` versus the same control. Both were rejected as neutral or slower. Reopening Q8_0 now requires a persistent decoded-weight cache or a justified transient-dense stage.

### D2: routed IQ2_XXS gate/up

Status: complete. J64/J80 retained.

Exact `(N,K)=(2048,4096)` J64 reduced the arithmetic body from 256 VGPRs with 360 private bytes and 117 spills to 229 VGPRs with zero private bytes and spills. Batch-1 uniform complete pair latency improved from `77.549 ms` to `33.581 ms`, and the complete 12-point route matrix improved by a 1.52x geometric mean. Batch-16 uniform exposed the cost of the smaller row tile, so a large-row alternative was measured.

J96 reached 256 VGPRs, 76 private bytes, and 18 spills and was rejected. J32 reached 256 VGPRs, 156 private bytes, and 48 spills and was rejected. J80 stayed spill-free at 253 VGPRs and 51 SGPRs. It improved every batch-16 distribution over J64, but was 1.2-2.2% slower at batch 4. The retained threshold is J80 when `rows >= 512 * num_groups`. Batch 1/4 use J64.

### D3: routed Q2_K down

Status: complete. Exact J32 with factor-4 unrolling and a J16 tail retained.

Exact J64 increased the kernel to 2,784 private bytes and 1,188 spills and regressed batch-1 uniform by 37.1%. Runtime-shape J64 still used 3,068 private bytes and 1,132 spills and regressed 9.9%. J32 reduced the generic body to 500 private bytes and 135 spills and improved batch-1 uniform from `76.312 ms` to `37.029 ms`. Exact `(N,K)=(4096,2048)` specialization then reduced this to 404 private bytes and 109 spills.

Splitting full and tail launches, and replicating invalid activation lanes, lowered some tail-body resource counts but lost complete-operator time because of the extra launch or clamp/address work. Both were reverted. A generator-owned `unroll 1` on the eight Q2_K scale/min phases produced a 14 KiB body at 122 VGPRs, 30 SGPRs, zero private bytes, zero spills, and no dynamic stack. Explicit unroll 2 remained spill-free at 158 VGPRs and was 1.28% better geometrically than unroll 1, but unroll 4 was faster on all 12 points while remaining spill-free at 208 VGPRs and 38 SGPRs. Factor 4 was retained.

Staging packed scale/min metadata across both J16 minitiles remained spill-free but regressed every point by 1.5-4.0%. The retained decoder reloads metadata and uses the J16 tail only when `rows < 64 * num_groups`.

### D4: integration and acceptance

Status: complete.

The final DeepSeek dispatch is fixed Q8_0 J64, IQ2_XXS J64/J80, and Q2_K J32 with factor-4 unrolling and J16 small-row tails. The final isolated DeepSeek matrix had 39/39 exact packed-reference checks and won all 24 routed comparisons. Code-object isolation restored the Qwen control after the monolithic translation unit caused layout-sensitive Q5_K movement. The generalized bundle now preserves these kernel bodies and thresholds. Packaging details are owned by `docs/kernel_bundle.md`.

## Post-bundle retuning decisions

These experiments were run after the standalone bundle conversion and are the final local retuning pass.

### Approximate-accumulator prerequisite

Status: closed before forward implementation.

Backward tested the prerequisite mechanisms in a DeepSeek Q2_K M128/N64 body. Direct paired BF16-C reduced VGPR use but produced `0.86089` NRMSE. Resource-clean pair-serial K32/K64 BF16-state slabs regressed latency by 35.0%/82.9% while using more VGPRs. Row-normalized paired FP16-C remained scale-stable but used more VGPRs and was 33.3% slower even after removing the row-scale scan. No low-register arithmetic body survived, so copying these mechanisms into forward K128 sum arrays or using them to reopen J128 and persistent scheduling was not justified.

### Q5_K mixed J64/J32 body

Status: rejected.

The retained standalone Q5_K J32 control improved Qwen batch-1 skewed, sparse, and boundary routes by 14-16% but lost uniform by 2.6-3.1%. A spill-free one-launch J64/J32 body improved uniform by 2.4% but regressed skewed, sparse, and boundary by 6.3%, 3.6%, and 3.6%. A second bounded variant had the same tradeoff with 8.8%, 8.2%, and 6.1% regressions. Packed-reference checks remained exact, but neither body preserved the nonuniform benefit. Standalone J32/J64 dispatch is retained.

### Fixed Q8_0 scale staging

Status: rejected.

The first candidate staged too few Q8_0 scales and raised independent-reference NRMSE to about 0.2015, so it was invalid. The corrected candidate staged one scale per accumulator fragment, remained exact and spill-free at 165 VGPRs, but measured `10.987/44.571/177.887 ms` versus a same-build `11.026/44.444/176.981 ms` control. Reduced scale rereads are not a bottleneck.

### Fixed Q8_0 group-major Q8_1 workspace

Status: rejected.

A fixed-only lossless workspace permutation remained exact and kept NRMSE at `0.006053-0.006062`. It measured `11.001/44.464/177.121 ms` versus `11.026/44.444/176.981 ms`. The largest movement was only 0.23%, with no durable batch-4 or batch-16 benefit. The extra quantizer/layout branch is not justified.

### DeepSeek Q2_K scale/min metadata staging

Status: rejected.

The factor-4 Q2_K decoder normally rereads packed scale/min metadata for both J16 accumulator minitiles. Keeping that metadata live across both minitiles remained spill-free, but regressed every point by 1.5-4.0% across batches 1, 4, and 16. Longer metadata lifetime and scheduling cost outweighed the eliminated LDS rereads.

### IQ2_S decoded-weight LDS cache

Status: rejected.

A separate IQ2_S J64 artifact decoded both weight tiles once per output workgroup and reused them across serial row tiles. It was exact and spill-free at 203 VGPRs and 46 SGPRs, but doubled the decoded-weight LDS allocation. Batches 4 and 16 regressed by 4.0-12.5% and 12.9-13.8%, respectively. Shared-memory occupancy dominates under the current representation.

## Mechanism evidence and durable lessons

### TensileLite and Composable Kernel

TensileLite and Composable Kernel were used as generator and interface evidence, not as a performance ceiling. Their durable lessons are:
- fixed-NK specialization removes dynamic scheduler state.
- compile-time tile and matrix-instruction selection must be measured per type.
- read, decode, LDS commit, LDS read, and WMMA lifetimes must be bounded.
- affine SGPR/immediate offsets and pointer increments help exact full tiles.
- extra LDS stages, C-shuffle, GSU, split-K, and persistent control require a resource and reduction argument.
- host-built grouped descriptors are incompatible with the device-resident route ABI.
- direct global-to-LDS does not perform GGUF decode, and activations remain in LDS because all waves reuse them.

The relevant study sources include:

```text
~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Tests/common/groupedgemm/grouped_gemm.yaml
~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Tests/common/groupedgemm/gfx11/grouped_gemm_gfx11.yaml
~/rocm-libraries/projects/composablekernel/example/15_grouped_gemm/grouped_gemm_wmma_fixed_nk_fp16.cpp
~/rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_fixed_nk.hpp
```

The useful grouped scheduling result was G8's compact device task list, not a copied host-side grouped GEMM setup. The useful decode result was bounded temporary state, not a wholesale direct-to-VGPR conversion.

### Compiler and ISA interpretation

Every structural change is judged from the final code object. Source-level local-array intuition was not reliable. Raw ELF offsets, code-object placement, and padding are not optimization signals. Normalized disassembly and resource metadata are.

The bundle transformation from `static __global__` bodies to `static __device__ __forceinline__` bodies under exported wrappers preserves compile-time template behavior. Controlled variants changing C++ standard, HIP/Torch defines, `-fPIC`, `-fno-gpu-rdc`, direct-HSACO mode, and related build flags did not explain representative ISA movement. The accepted explanation for the old monolithic movement was code-object layout and resource interaction, which the independent HSACO bundle removes from the arithmetic kernels.

The strict assembly/performance overlap study remains useful as prioritization evidence:
- dense forward: 0/2 strict over-2% overlaps.
- dense backward: 7/7.
- grouped forward: 3/3.
- grouped backward: 4/4.

Assembly movement is therefore useful when it identifies resource or work-decomposition changes, but not as a substitute for complete public-operator timing.

### Final structural conclusions

- Register spilling was the first blocker. Compile-time shape specialization removed the original 124-520 byte private segments.
- Full-row address decomposition was a separate first-order cost. G4 and G11 removed unnecessary bounds work.
- J64 is the retained geometry. J128 and I128 lose from accumulator/LDS/residency costs even when spills are absent.
- Gate/up benefits from device row tasks only at large row buckets. Down already has enough output-tile parallelism.
- Two explicit down K blocks and affine gate/up traversal are type- and shape-specific wins.
- Quantization and launcher setup are not the remaining limit. Packed multiplication accounts for roughly 87-99% of retained operator kernel time.
- Packed decode representation, especially IQ2_S on partial down tiles, is the remaining ceiling.

## Validation and reproducibility

The current source-only workflow keeps HSACOs ignored by Git and materializes them during local extension or wheel builds. Generic artifact, loader, and packaging behavior belongs in `docs/kernel_bundle.md`. This document records only the grouped-forward consequences and dispatch decisions.

The final validation set includes:

```bash
pytest -q tests/
ruff check .
python -m compileall -q tools
python tools/build_mmq_bundle.py --check
git diff --check
```

The complete final matrices must be preserved as regression controls for future representation-level work. A future change must rerun both Qwen and DeepSeek matrices, packed-reference exactness, independent-reference error checks, resource gates, and the relevant 25-repeat sequential controls.

## Final status

The grouped MMQ forward pass is complete for the current packed Qwen and DeepSeek representations.

The retained implementation uses compile-time exact shapes, J64/J32/J16/J80 only where measured, separate full and bounded tail bodies, device row tasks only for large gate/up groups, two-block down unrolling, pointer-increment gate/up traversal, and deterministic standalone HSACO artifacts. Against the exact target-specific AITER table, Qwen wins 45/60 comparisons and DeepSeek wins 12/24 routed comparisons, with exact packed-reference outputs throughout.

The remaining losses are representation-level comparisons against already decoded BF16 weights. Reduced-precision accumulation is also closed by the completed backward controls. No further local tile, accumulator, or scheduling sweep is planned until a compact reusable decoded representation, a transient dense stage, or a new profile-supported arithmetic mechanism can be evaluated end to end.
