# Grouped MMQ backward optimization

## Current status

Grouped MMQ backward is production-tuned for the current Qwen and DeepSeek GGUF representations on gfx1151. The exact FP32-accumulator pass, prior-aware AITER comparator retune, and coefficient-only full HIP retune are complete. The approximate-accumulator controls are also complete; none met the combined numerical, resource, and latency requirements. Bottleneck attribution is complete from representative profiler controls plus family-specific geometry, mechanism, resource, and source evidence; not every exact production key has a separate hardware-counter run.

The final benchmark authorities are:

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_deepseek_prior_retuned_final_9.json
```

Latest outcome:
- Qwen packed kernels win `33/60` case, batch, and mandatory-control points against the promoted BF16 AITER GMM table. Both fused gate/up families win all 24 points.
- DeepSeek routed IQ2_XXS wins `8/12` and routed Q2_K wins `5/12`; fixed Q8_0 wins `3/3` against BF16 BMM.
- The coefficient-only campaign promoted DeepSeek IQ2_XXS pair M128/N64 only at aggregate rows `49,152` and `196,608`, plus fixed Q8_0 M192/N64 only at `32,768` tokens. Qwen and DeepSeek Q2_K retain their prior bodies.
- Reversed-order 25-repeat confirmation measured IQ2_XXS speedups of `1.1678x` at B4 and `1.1644x` at B16, with a `1.0562x` worst profile/control. Fixed Q8_0 B16 confirmed at `1.03227x`.
- Independent production qualification reached maximum sampled normalized RMSE `0.00027772` and minimum cosine `0.99999994`; all gradient, packed-weight, inactive-expert, projection, and fixed-group mutation gates passed.
- Both new kernels are wave32, call-free, scratch-free, zero-private, zero-spill, and stack-free. The complete gfx1151 bundle contains 181 kernels, including 34 grouped-backward entries.
- Direct BF16-C, BF16 slab, and normalized FP16-C controls are complete and rejected. Production continues to dispatch exact FP32 accumulation.
- Prepared-weight and changed-API work remain deferred model-owned projects with explicit lifetime, invalidation, memory, cold-start, and sharing requirements.

The exact comparator table was subsequently retuned against learned-route priors, captured route medoids, and the mandatory synthetic controls. The retuned Qwen backward replay is:

```text
~/tmp/torch-ggml-ops/aiter-prior-replay-retuned/qwen_learned_backward.json
~/tmp/torch-ggml-ops/aiter-prior-replay-retuned/comparator_only_analysis.json
```

## Prior-aware AITER comparator retune

The 24-target AITER screen used the bounded current/neighbor/cross-batch/archive candidate domain, five weighted learned medoids per Qwen/DeepSeek family, five hash medoids per DeepSeek family, and the `40/43` learned plus `3/43` hash objective. Finalists were then checked over captured Qwen and DeepSeek learned states, exact DeepSeek token/`tid2eid` hash projections, and the mandatory `uniform`, `skewed`, `sparse`, and `boundary` controls. Promotion required correctness, a stable greater-than-2% prior gain, no greater-than-1% route/profile regression, and reversed-order 25-repeat controls on all three corpora.

Three exact backward keys were promoted. Tuples are ordered `(BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, GROUP_SIZE, GRID_DIM, num_warps, num_stages)`:

| Key and workload | Previous tuple | Promoted tuple |
|---|---|---|
| Qwen B1 gate/up pair `(16384,512,2048,False)` | `(64,32,64,8,256,8,1)` | `(64,32,64,8,256,2,1)` |
| Qwen B16 gate/up pair `(262144,512,2048,False)` | `(64,32,256,8,40,8,2)` | `(64,32,256,8,40,8,1)` |
| Qwen B16 down `(262144,2048,512,False)` | `(64,32,256,8,160,8,3)` | `(64,32,128,1,80,8,2)` |

The GPU qualification used boundary-heavy nonuniform groups. All three promoted outputs were bitwise identical to the replaced configs, passed independent per-group BF16 reference checks, and changed under a one-element input mutation. The evidence is retained in:

```text
~/tmp/torch-ggml-ops/aiter-prior-config-correctness.json
~/tmp/torch-ggml-ops/aiter_captured_control_profiles.json
~/tmp/torch-ggml-ops/aiter_synthetic_control_profiles.json
~/tmp/torch-ggml-ops/aiter-compromise-screen/summary.json
~/tmp/torch-ggml-ops/aiter-compromise-confirm-prior/summary.json
~/tmp/torch-ggml-ops/aiter-compromise-confirm-captured/summary.json
~/tmp/torch-ggml-ops/aiter-compromise-confirm-synthetic/summary.json
```

Holding archived HIP medians fixed and replacing only changed AITER medians moved the model-call-weighted Qwen backward ratios from `1.143/1.302/1.153x` to `1.029/1.302/1.066x` at B1/B4/B16. DeepSeek backward has no promoted key and remains unchanged. The initial pooled winners that lost uniform or boundary controls were rejected; no table entry was changed from the prior screen alone.

The installed AITER `get_config` accepts `(M,K,N,G)` but returns an architecture-level default and currently does not use shape lookup. Because the installed package lacks a `gfx1151-GMM.json` and benchmark calls pass explicit configs, `tools/aiter_gmm_heuristics.py:gmm_config` remains the exact benchmark-owned authority. Unsupported keys fail closed and exact layout separation is tested in `tests/test_aiter_gmm_heuristics.py`.

## Learned-B1 HIP ownership retune

The learned-route corpus reopened the existing B1 serial/row-task choice for Qwen single-down backward. Five prior medoids had maximum expert heights of `800-2048`; five captured medoids reached `801-1960`, even when aggregate average rows stayed below the 128-row task gate. An isolated build compared the current predicate with the existing M-major 128-row task body across those ten profiles and the four mandatory synthetic controls. It did not add a new kernel or read device offsets on the host.

In the nine-repeat screen, IQ2_S prior/captured/synthetic gains were `1.1600/1.1369/1.1075x`. Reversed-order 25-repeat confirmation measured `1.1680/1.1298/1.1017x`; the minimum profile was `0.9903x`, and every output was bitwise identical. Q4_K and Q5_K retained current dispatch: despite prior gains above `1.21x`, their synthetic aggregates fell to `0.9637x` and `0.9572x`, with worst profiles of `0.8481x` and `0.8927x`.

Production now selects the existing row-task body for the exact IQ2_S single-down backward key `(rows,N,K)=(16384,2048,512)`. Other quant types and row counts retain the average-row predicate. At that campaign stage, the 179-kernel bundle and all kernel resources were unchanged. The post-change replay measured a `1.2600x` key gain. Holding promoted AITER medians and unaffected archived HIP medians fixed moved the model-call-weighted Qwen B1 backward ratio from `1.0291x` to `1.0822x`; B4/B16 remain `1.3023/1.0664x`.

```text
~/tmp/torch-ggml-ops/hip-b1-rowtask-screen-9.json
~/tmp/torch-ggml-ops/hip-b1-rowtask-confirm-25.json
~/tmp/torch-ggml-ops/hip-b1-rowtask-retuned/qwen_learned_backward.json
~/tmp/torch-ggml-ops/hip-b1-rowtask-retuned/hip_only_analysis.json
```

## Coefficient-only full HIP retuning campaign

Status: complete. This is the backward half of the coefficient-only campaign defined in `docs/grouped_mmq_fwd_optimization.md`. The final production bundle contains exactly two qualified backward additions and 181 kernels total.

The fitted prior covers the seven routed production families and the fixed Q8_0 family below at physical batches 1, 4, and 16. Generic bundle wrappers without Qwen or DeepSeek prior mass remain correctness/resource fallbacks rather than weighted tuning targets. Existing ABI-compatible bodies are still eligible during the dispatch stage.

| Production family | Current arithmetic choices | First retuning surface | Reopen only after that surface closes |
|---|---|---|---|
| Qwen Q3_K fused pair `(N,K)=(512,2048)` | M64/N64 and M128/N64 | M ownership threshold and existing-body selection | decoder width, LDS padding, decode/WMMA schedule |
| Qwen IQ2_S fused pair `(512,2048)` | M64/N64 and M128/N64 | M ownership threshold and existing-body selection | IQ2_S decode and accumulation schedule |
| Qwen Q4_K down `(2048,512)` | M64/N64, M128/N64, row-task M128/N128 | serial/row-task ownership, M geometry, task rows | decode preload and inactive-M mechanism |
| Qwen Q5_K down `(2048,512)` | M64/N64, row-task M128/N128 | serial/row-task ownership and task rows | N64/N128 decode path and swizzle |
| Qwen IQ2_S down `(2048,512)` | M64/N64, M128/N64, row-task M128/N128 | serial/row-task ownership, M geometry, task rows | IQ2_S decode, swizzle, inactive-M mechanism |
| DeepSeek IQ2_XXS fused pair `(2048,4096)` | M64/N64 at B1; qualified M128/N64 at routed rows 49,152 and 196,608 | exact-row ownership and M geometry | decode width, LDS swizzle, pair-load schedule |
| DeepSeek Q2_K down `(4096,2048)` | M64/N64 U1, M128/N64 U1/U2 | M ownership and reduction-unroll selection | Q2_K decode/load schedule |
| DeepSeek fixed Q8_0 output A `(1024,4096)` | M256/N64 except M192/N64 at 32,768 tokens | exact-token ownership and M geometry | Q8_0 decode/load and store schedule |

The bounded search screened every existing Qwen and DeepSeek body before compiling 24 typed geometry candidates. Twenty-two candidates were resource-clean; the two promoted finalists were the only new geometries that passed the strict timing and control gates. DeepSeek IQ2_XXS pair M128/N64 confirmed at `1.1678x` and `1.1644x` for B4/B16, with every learned, hash, and mandatory control above `1.05x`. Fixed Q8_0 M192/N64 confirmed at `1.0322713651x` at B16; B1 regressed and B4 remained below the promotion floor.

Production adds `GroupedBwdTunedPairIQ2XXSN2048K4096M128N64` and `GroupedBwdTunedFixedQ80G8K4096M192N64`. IQ2_XXS dispatch uses the tuned pair only for exact aggregate rows `49152` and `196608`; fixed Q8_0 dispatch uses M192 only for exact `tokens == 32768`. All other existing dispatch thresholds and kernel identities remain unchanged.

Campaign and production evidence:

```text
~/tmp/torch-ggml-ops/grouped-bwd-candidate-resources.json
~/tmp/torch-ggml-ops/grouped-bwd-finalists/deepseek_iq2xxs_m128_confirmation_25.json
~/tmp/torch-ggml-ops/grouped-bwd-finalists/deepseek_fixed_m192_b16_confirmation_25.json
~/tmp/torch-ggml-ops/grouped-bwd-finalists/finalist-correctness.json
~/tmp/torch-ggml-ops/grouped-bwd-production-final-correctness.json
~/tmp/torch-ggml-ops/grouped-bwd-production-final-resources.json
~/tmp/torch-ggml-ops/grouped-bwd-production-final-rebuild.json
```

As in forward, the work is deliberately split into dispatch, geometry, and mechanism levels. Dispatch among current HSACOs changes no device instructions. Typed geometry candidates may vary N tiles, M tiles per wave, row-task height, decoder width, reduction unroll, LDS padding/swizzle, and the implemented inactive-M predicate only in combinations supported by the body. Geometry fields that determine launch dimensions, shared-tile types, or task coverage are linked and validated together. Decode reordering, prefetch, vectorization, WMMA issue order, accumulator organization, and store scheduling are mechanism changes, not `constexpr` retuning.

The 128-thread/four-wave specialized workgroup, WMMA 16x16x16 instruction, FP32 accumulation, fused pair's single accumulator/result/rounding contract, packed GGUF layout, and device-resident route ABI are frozen. The 256-thread generic fallback is an existing control, not a free thread-count coordinate. Approximate BF16/FP16 accumulator variants remain closed by the numerical campaigns already recorded below.

### Prior and tool contract

Backward uses the same self-contained `tools/tune_grouped_mmq_prior.py` as forward. The script embeds the exact Qwen learned, DeepSeek learned, and capture-free DeepSeek hash state from [Fitted model-level routing prior](ggtensile_plan.md#fitted-model-level-routing-prior). It depends only on `T = physical_batch * 2048`, deterministically creates disjoint 512-draw search and confirmation banks and five weighted physical-ID medoids per bank, router component, and batch, and locally generates `uniform`, `skewed`, `sparse`, and `boundary` controls. The exact seeds, normalized-L1 k-medoids rule, deterministic tie breaking, and serialized profile fields are shared with the forward contract. It reads packed weights from the requested GGUF models but reads no route capture, checkpoint report, adapter state, or file under `~/tmp` or `~/test_no_unsloth`. This campaign built and tuned HIP kernels only; no stage generated or qualified GGTensile kernels.

Qwen row sums are `{16384,65536,262144}` and DeepSeek routed row sums are `{12288,49152,196608}`. DeepSeek learned and hash profiles are measured and gated separately; `40/43` plus `3/43` weighting is ranking/reporting metadata only. The coefficient-only hash profile is explicitly a surrogate for exact token/`tid2eid` projection. Fixed output A has no route profile and is measured at token rows `{2048,8192,32768}`. Captured routes can be optional external replay evidence, but search and promotion remain reproducible without them.

The script owns profile generation, candidate manifests, timing, correctness, and confirmation. A candidate record includes exact operator/quant/shape/batch, source and transitive-include hashes, typed geometry and mechanism state, launch dimensions, compiler identity and arguments, HSACO/disassembly/metadata hashes, VGPR/SGPR/LDS/private resources, profile seeds and group vectors, timing samples, and rejection reason. There is no hidden source mutation or environment-controlled production selector.

### HIP compiler boundary

Backward headers currently encode several geometries as shared constants and named bodies. The campaign must not edit one of those constants in place and rebuild all wrappers. That would confound the intended parameter with unrelated `hipcc` register-allocation and scheduling movement. Missing fields are first moved into typed `GroupedBackwardConfig` state and emitted as explicit template arguments. Each candidate wrapper then compiles alone into a content-addressed HSACO while the frozen production baseline remains byte-identical.

Finalists compile twice in clean directories and must have identical generated source, HSACO, disassembly, metadata, and resource reports. A formatting-only source variant is a separate compiler candidate. Every artifact with private bytes, VGPR/SGPR spills, scratch instructions, calls, or dynamic stack is rejected before timing. A pinned compiler is part of the selected identity; a toolchain change requires requalification. Candidate loading remains tool-owned and does not alter the public operator or production dispatch path.

### Executed backward campaign stages

| Stage | Work | Exit condition |
|---|---|---|
| B0 | Generate coefficient-only medoids/controls and freeze complete-call plus arithmetic baselines for all eight families and three batches | Profiles are byte-reproducible and current artifacts reproduce |
| B1 | Screen every ABI-compatible existing body at each exact `(operator,quant,R,N,K)` key, including serial/task and M64/M128/U1/U2 choices | No existing-body winner remains above the promotion floor |
| B2 | Run bounded typed coordinate search and small linked crosses over M/N tiles, task rows, decoder width, reduction unroll, LDS layout, and supported inactive-M policy | Finalists are correct, resource-clean, and deterministic |
| B3 | Open one named HIP decode/load/WMMA/store mechanism at a time only for material residual losses | Mechanism gain is isolated from geometry and source-form noise |
| B4 | Run reversed-order confirmation, independent references, mutation checks, all controls, and clean rebuild identity; then update exact dispatch/catalogs | Every promotion gate passes for the exact key |

Promotion requires a stable greater-than-2% aggregate gain on the disjoint confirmation medoids, minimum individual search/confirmation profile and router-component speedup of `0.99x`, at least `0.99x` on each mandatory synthetic control, and reversed-order 25-repeat confirmation. Complete public-operator timing includes output allocation and row-task setup. Arithmetic-only timing is diagnostic. No aggregate model score may hide a regression in one production family.

Correctness uses the independent per-group FP32-accumulating reference, old/new output comparison where applicable, input/gradient and packed-weight mutation sensitivity, zero-length/inactive experts, non-multiple M tails, and route changes. Fused pair candidates must preserve one FP32 accumulation of both projections and one BF16 rounding; comparing two separately rounded single calls is not an admissible reference for exact pair semantics.

Promotion must not introduce host inspection of `expert_offsets`, model/router/checkpoint dispatch dimensions, hidden synchronization, atomics where the retained body is atomics-free, prepared weights, or a changed public ABI. The final campaign report includes all rejected candidates, reruns the complete forward/backward and dense controls for shared code, verifies `tools/build_mmq_bundle.py --check`, records artifact hashes, and updates this document with the resulting bundle count.

## Final benchmark results

The authoritative final matrices use warmup 3, nine sequential repeats, correctness rows 256, and all four mandatory routing distributions. The table below displays the uniform-route result per logical matrix shape; skewed, sparse, and boundary rows remain in the JSON authorities. Logical shapes use `(M,N,K)`. Paired gate/up executes two matrices with one fused FP32 accumulation, while fixed output A executes eight independent group matrices. Ratios above `1.000x` favor packed HIP.

| Model/operator | GGUF type | B | Full logical shape `(M,N,K)` | Packed TFLOPS | Reference | HIP/reference |
|---|---|---:|---:|---:|---|---:|
| Qwen paired gate/up | Q3_K | 1 | `2 x (16384,512,2048)` | 19.09 | AITER GMM | `2.392x` |
| Qwen paired gate/up | Q3_K | 4 | `2 x (65536,512,2048)` | 26.09 | AITER GMM | `1.922x` |
| Qwen paired gate/up | Q3_K | 16 | `2 x (262144,512,2048)` | 24.54 | AITER GMM | `1.501x` |
| Qwen paired gate/up | IQ2_S | 1 | `2 x (16384,512,2048)` | 17.26 | AITER GMM | `2.158x` |
| Qwen paired gate/up | IQ2_S | 4 | `2 x (65536,512,2048)` | 26.02 | AITER GMM | `1.899x` |
| Qwen paired gate/up | IQ2_S | 16 | `2 x (262144,512,2048)` | 24.17 | AITER GMM | `1.479x` |
| Qwen down | IQ2_S | 1 | `(16384,2048,512)` | 9.12 | AITER GMM | `0.824x` |
| Qwen down | IQ2_S | 4 | `(65536,2048,512)` | 13.95 | AITER GMM | `0.738x` |
| Qwen down | IQ2_S | 16 | `(262144,2048,512)` | 16.55 | AITER GMM | `0.742x` |
| Qwen down | Q4_K | 1 | `(16384,2048,512)` | 9.70 | AITER GMM | `0.869x` |
| Qwen down | Q4_K | 4 | `(65536,2048,512)` | 13.56 | AITER GMM | `0.808x` |
| Qwen down | Q4_K | 16 | `(262144,2048,512)` | 17.19 | AITER GMM | `0.758x` |
| Qwen down | Q5_K | 1 | `(16384,2048,512)` | 9.38 | AITER GMM | `0.849x` |
| Qwen down | Q5_K | 4 | `(65536,2048,512)` | 14.54 | AITER GMM | `0.789x` |
| Qwen down | Q5_K | 16 | `(262144,2048,512)` | 17.37 | AITER GMM | `0.761x` |
| DeepSeek fixed output A | Q8_0 | 1 | `8 x (2048,1024,4096)` | 22.74 | BF16 BMM | `1.228x` |
| DeepSeek fixed output A | Q8_0 | 4 | `8 x (8192,1024,4096)` | 22.34 | BF16 BMM | `1.155x` |
| DeepSeek fixed output A | Q8_0 | 16 | `8 x (32768,1024,4096)` | 23.82 | BF16 BMM | `1.202x` |
| DeepSeek paired gate/up | IQ2_XXS | 1 | `2 x (12288,2048,4096)` | 13.99 | AITER GMM | `1.513x` |
| DeepSeek paired gate/up | IQ2_XXS | 4 | `2 x (49152,2048,4096)` | 17.59 | AITER GMM | `1.172x` |
| DeepSeek paired gate/up | IQ2_XXS | 16 | `2 x (196608,2048,4096)` | 18.15 | AITER GMM | `0.698x` |
| DeepSeek down | Q2_K | 1 | `(12288,4096,2048)` | 11.43 | AITER GMM | `1.162x` |
| DeepSeek down | Q2_K | 4 | `(49152,4096,2048)` | 17.19 | AITER GMM | `0.849x` |
| DeepSeek down | Q2_K | 16 | `(196608,4096,2048)` | 19.20 | AITER GMM | `0.712x` |

Qwen wins `33/60` comparisons. DeepSeek routed kernels win `13/24` comparisons and fixed Q8_0 wins `3/3`. The final model-call-weighted ratios span `1.287-1.569x`, `1.358-1.775x`, and `1.106-1.180x` for Qwen B1/B4/B16; DeepSeek spans `1.288-1.396x`, `1.076-1.124x`, and `0.759-0.782x`. Standard-matrix BF16-reference NRMSE is at most `0.002878` for Qwen and `0.002869` for routed DeepSeek; fixed Q8_0 is approximately `5.2e-5` against BF16 BMM. The independent sampled-FP32 finalist qualification is stricter for the promoted rows and is recorded separately below.

Final benchmark JSON authorities:

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_deepseek_prior_retuned_final_9.json
```

## Historical pre-retune results

### Qwen

The latest Qwen matrix uses `~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf` and includes physical batches 1, 4, and 16 with uniform, skewed, sparse, and boundary routing.

| Family | Wins | Reference/packed latency range | Median | Packed latency by batch |
| --- | ---: | ---: | ---: | --- |
| Fused Q3_K gate/up | 12/12 | `1.448-2.335x` | `1.846x` | B1 `3.672-5.740 ms`, B4 `10.717-14.224 ms`, B16 `46.307-50.024 ms` |
| Fused IQ2_S gate/up | 12/12 | `1.458-2.113x` | `1.774x` | B1 `4.070-6.187 ms`, B4 `10.838-14.636 ms`, B16 `46.420-49.232 ms` |
| IQ2_S down | 3/12 | `0.711-1.393x` | `0.775x` | B1 `3.534-4.523 ms`, B4 `9.799-10.440 ms`, B16 `33.986-35.046 ms` |
| Q4_K down | 3/12 | `0.753-1.468x` | `0.835x` | B1 `3.541-4.260 ms`, B4 `9.659-9.850 ms`, B16 `32.623-33.300 ms` |
| Q5_K down | 3/12 | `0.740-1.493x` | `0.852x` | B1 `3.682-3.965 ms`, B4 `9.493-9.921 ms`, B16 `32.596-32.951 ms` |

Q4_K and Q5_K include the final inactive-M row-task suppression. Relative to sequential 25-repeat false controls:
- Q4_K improves every B4/B16 point by `5.27-11.60%`, with a `9.16%` geometric gain.
- Q5_K improves every B4/B16 point by `1.05-6.34%`, with a `3.68%` geometric gain.
- IQ2_S suppression was rejected. It improved geometrically by only `1.06%` and regressed B4 uniform and B16 boundary.

Checkpoint-weighted Qwen estimate, covering the five grouped-backward families and their checkpoint call counts:

| Batch | Route | Packed ms | AITER ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 1 | uniform | 312.7 | 466.9 | 1.493x |
| 1 | skewed | 413.3 | 566.6 | 1.371x |
| 1 | sparse | 383.0 | 595.8 | 1.556x |
| 1 | boundary | 399.9 | 569.6 | 1.424x |
| 4 | uniform | 823.9 | 1,090.3 | 1.323x |
| 4 | skewed | 976.0 | 1,604.3 | 1.644x |
| 4 | sparse | 960.5 | 1,556.8 | 1.621x |
| 4 | boundary | 978.8 | 1,687.3 | 1.724x |
| 16 | uniform | 3,186.7 | 3,738.0 | 1.173x |
| 16 | skewed | 3,330.1 | 3,929.0 | 1.180x |
| 16 | sparse | 3,346.3 | 3,899.5 | 1.165x |
| 16 | boundary | 3,340.1 | 3,985.4 | 1.193x |

The isolated single-down deficits do not overturn the checkpoint-weighted result. The fused pair families dominate the aggregate win because each combines two projections, keeps one FP32 accumulator set, rounds once, and allocates one output.

### DeepSeek

The latest DeepSeek matrix uses `~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf`. Routed cases use top-six routing with uniform, skewed, sparse, and boundary distributions. Fixed output-A uses its dedicated `[tokens, 8, features]` layout and has no fabricated routing metadata.

| Family | Wins | Reference/packed latency range | Median | Packed latency by batch |
| --- | ---: | ---: | ---: | --- |
| Fixed Q8_0 output-A | 3/3 | `1.164-1.222x` | `1.177x` | B1 `6.171 ms`, B4 `24.936 ms`, B16 `96.303 ms` |
| Fused IQ2_XXS gate/up | 7/12 | `0.599-1.483x` | `1.038x` | B1 `30.406-34.878 ms`, B4 `101.652-112.845 ms`, B16 `430.977-446.138 ms` |
| Q2_K down | 5/12 | `0.692-1.128x` | `0.826x` | B1 `18.525-21.184 ms`, B4 `47.993-49.665 ms`, B16 `174.583-187.766 ms` |

Checkpoint-weighted DeepSeek estimate includes 43 calls each of fixed output-A, fused gate/up, and routed down:

| Batch | Route | Packed ms | BF16 reference ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 1 | uniform | 2,369.4 | 3,137.4 | 1.324x |
| 1 | skewed | 2,676.0 | 3,513.7 | 1.313x |
| 1 | sparse | 2,566.2 | 3,347.9 | 1.305x |
| 1 | boundary | 2,515.7 | 3,138.0 | 1.247x |
| 4 | uniform | 7,574.2 | 7,683.5 | 1.014x |
| 4 | skewed | 8,008.7 | 7,865.5 | 0.982x |
| 4 | sparse | 7,863.4 | 7,698.5 | 0.979x |
| 4 | boundary | 8,060.2 | 8,164.9 | 1.013x |
| 16 | uniform | 30,180.1 | 21,312.0 | 0.706x |
| 16 | skewed | 30,866.2 | 22,170.3 | 0.718x |
| 16 | sparse | 31,272.8 | 22,010.5 | 0.704x |
| 16 | boundary | 31,308.9 | 22,066.3 | 0.705x |

The tuned predecoded reference is now faster for most DeepSeek B4 routed points and all B16 routed points. This is evidence for a model-owned prepared representation, not an operator-internal hidden cache: the reference still excludes decode, storage, invalidation, and preparation costs.

## Scope and contracts

This document covers three input-gradient operators:
- `grouped_mmq_grad_input` for one routed frozen packed projection.
- `grouped_mmq_pair_grad_input` for fused routed gate/up input gradients.
- `fixed_grouped_mmq_grad_input` for DeepSeek's fixed eight-group output-A layout.

For one routed expert group:

```text
forward:  Y[M, N]  = X[M, K]  @ W[N, K].T
backward: dX[M, K] = dY[M, N] @ W[N, K]
```

The public cotangent and result are BF16. WMMA accumulation is FP32. The authoritative weights remain packed GGUF tensors.

Production invariants:
- Use the current CUDA/HIP stream.
- Do not materialize a logical dense weight matrix in the ordinary path.
- Do not launch arithmetic workgroups for inactive experts.
- Keep route metadata device-resident.
- Do not add `.item()`, host descriptors, device-to-host route copies, or hidden synchronization.
- Preserve the fused pair's one FP32 accumulator set, one BF16 rounding, and output-only allocation.
- Preserve fixed output-A's `[tokens, 8, features]` layout. It is not a routed operator.

Routed metadata ABI:

```text
expert_indices: contiguous CUDA int64, shape [G]
expert_offsets: contiguous CUDA int32, shape [G]
expert_offsets[-1] = total routed rows R
G <= 256
```

Main implementation files:

```text
csrc/ck/grouped_mmq_backward.cuh
csrc/ck/grouped_mmq_backward_tiled.cuh
csrc/ck/gguf_decode.cuh
csrc/mmq_bundle.cpp
csrc/generated/mmq_bundle_table.cuh
tools/build_mmq_bundle.py
tools/mmq_bundle_wrapper_source.py
```

Project-specific decode logic stays outside `csrc/vendor/llama_cpp/*`.

## Workloads and references

### Qwen workload

| Case | Weight `(N, K)` | GGUF type | Calls/layers | Operator |
| --- | ---: | --- | ---: | --- |
| Gate/up outer | `(512, 2048)` | Q3_K | 20 pairs | fused pair |
| Gate/up middle | `(512, 2048)` | IQ2_S | 20 pairs | fused pair |
| Down middle | `(2048, 512)` | IQ2_S | 20 | single |
| Down outer main | `(2048, 512)` | Q4_K | 18 | single |
| Down outer edge | `(2048, 512)` | Q5_K | 2 | single |

Qwen sequence length is 2,048 and top-k is 8.

| Physical batch | Routed rows | Uniform rows/expert |
| ---: | ---: | ---: |
| 1 | 16,384 | 64 |
| 4 | 65,536 | 256 |
| 16 | 262,144 | 1,024 |

### DeepSeek workload

| Case | Weight `(N, K)` | GGUF type | Calls | Operator |
| --- | ---: | --- | ---: | --- |
| Fixed output-A, 8 groups | `(1024, 4096)` per group | Q8_0 | 43 | fixed-group single |
| Routed gate/up, 256 experts | `(2048, 4096)` | IQ2_XXS | 43 pairs | fused pair |
| Routed down, 256 experts | `(4096, 2048)` | Q2_K | 43 | single |

DeepSeek sequence length is 2,048 and top-k is 6.

| Physical batch | Token rows | Routed rows | Uniform rows/expert |
| ---: | ---: | ---: | ---: |
| 1 | 2,048 | 12,288 | 48 |
| 4 | 8,192 | 49,152 | 192 |
| 16 | 32,768 | 196,608 | 768 |

### Routing distributions

Routed benchmarks cover four deterministic distributions:
- `uniform`: all experts have equal group sizes.
- `skewed`: all experts are active with deterministic nonuniform sizes.
- `sparse`: 192, 224, and 240 active experts at batches 1, 4, and 16.
- `boundary`: includes sizes 1, 15, 16, 17, 63, 64, 65, 127, 128, and 129.

These distributions are mandatory. Uniform-only movement is not sufficient to retain a dispatch rule because it does not expose inactive experts, rounded tails, or route imbalance.

For Qwen 128-row tasks, expert-local task counts grow from 192-257 at B1 to 512-660 at B4 and 2,048-2,174 at B16, depending on route distribution. This is why task ordering and nonuniform launch cost were measured directly rather than inferred from uniform routing.

### BF16 references

Routed performance uses AITER Triton `gmm` configured by the benchmark-owned `tools/aiter_gmm_heuristics.py` exact table. Dispatch keys include total routed rows, K, N, and RHS layout. Unsupported shapes fail closed. Backward uses contiguous row-major logical weights. Fixed Q8_0 uses BF16 BMM in the public fixed-group layout.

The timed references start with independently dequantized BF16 weights. Dequantization and active-expert selection are setup costs and are not included. This makes AITER an ideal predecoded arithmetic reference, not a complete packed-weight alternative.

The table contains target-specific B1/B4/B16 configurations for both Qwen and DeepSeek rather than a shape-only heuristic. AITER is a production reference, not a performance ceiling.

For fused pairs, AITER runs two GMM calls and adds two BF16 outputs. It is the production performance reference but not a bitwise numerical oracle because the packed pair accumulates both projections in FP32 and rounds once.

Correctness references:
- Qwen Q3_K, Q4_K, and Q5_K may use the independent dense packed path.
- Qwen IQ2_S uses independently dequantized BF16 weights plus grouped single/pair consistency; dense IQ2_S is not a current correctness oracle.
- DeepSeek Q2_K and IQ2_XXS use independently dequantized BF16 weights.
- Fixed Q8_0 uses independent GGUF decode and BF16 BMM in `[tokens, 8, K]` layout.

### Measurement rules

- Run GPU benchmarks sequentially with real nonzero tensors.
- Warm module loading before timing.
- Use three warmups and nine repeats for matrix coverage.
- Any fresh median movement above 1% requires a sequential 25-repeat A/B control.
- Include route-task setup, allocation, decode, arithmetic, and layout handling in public latency.
- Retain a local candidate only when a sequential control shows at least one material target gain, normally 2% or more, without a repeatable family regression above 1%.
- Use `PYTHONPATH=.` so benchmarks load the in-tree extension.
- Build with all CPU cores:

```bash
python tools/build_mmq_bundle.py --force --jobs "$(nproc)"
```

Current public-API commands:

```bash
PYTHONPATH=. python bench/benchmark_grouped_mmq_bwd_api.py \
  --model ~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf \
  --model-family qwen --expert-prior qwen-learned \
  --warmup 3 --repeats 9 \
  --output ~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_api.json

PYTHONPATH=. python bench/benchmark_grouped_mmq_pair_bwd_api.py \
  --model ~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf \
  --model-family deepseek --expert-prior deepseek-learned \
  --warmup 3 --repeats 9 \
  --output ~/tmp/torch-ggml-ops/grouped_mmq_pair_bwd_deepseek_api.json

PYTHONPATH=. python bench/benchmark_fixed_grouped_mmq_bwd_api.py \
  --model ~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf \
  --model-family deepseek --warmup 3 --repeats 9 \
  --output ~/tmp/torch-ggml-ops/fixed_grouped_mmq_bwd_deepseek_api.json
```

Use the corresponding `_kernels.py` entry point plus `--hip-root build/mmq_hip_controls/gfx1151` for direct GGTensile/HIP timing.

## Production implementation

### Dispatch

Dispatch uses only host-visible shape, quant type, total rows, and group count. It does not inspect device offsets or use online autotuning.

| Model/family | Production dispatch |
| --- | --- |
| Qwen Q3_K pair | M64/N64/K32 below `128 * num_groups`, M128/N64/K32 otherwise |
| Qwen IQ2_S pair | M64/N64/K32 below `128 * num_groups`, M128/N64/K32 otherwise |
| Qwen Q4_K down | M64/N64 below 80 rows/group, M128/N64 at 80-127, M128/N128 row tasks at 128+ |
| Qwen Q5_K down | M64/N64 below 128 rows/group, M128/N128 row tasks at 128+ |
| Qwen IQ2_S down | M64/N64 below 80 rows/group, M128/N64 at 80-127, M128/N128 row tasks at 128+ or exact aggregate rows 16,384 |
| DeepSeek IQ2_XXS pair | M64/N64/K32 at B1; M128/N64/K32 only at exact aggregate rows 49,152 and 196,608; width-16 decode and swizzle4 |
| DeepSeek Q2_K down | M64/U1 below 128 rows/group, M128/U2 at 128-511, M128/U1 at 512+ |
| DeepSeek fixed Q8_0 | M256/N64/K32 except M192/N64/K32 at exact token count 32,768; width-16 decode and no swizzle |
| Unsupported type or shape | Generic grouped compatibility kernel |

Every selected kernel remains bounded-correct for nonuniform groups. Average rows per group is only a host-visible dispatch hint.

### Retained arithmetic mechanisms

Common choices:
- Four wave32 waves and 128 threads.
- K32 reduction tiles.
- Cooperative width-16 packed decode matched to natural quant metadata sharing.
- FP32 WMMA accumulation and BF16 stores.
- Exact production shapes and static template arguments.
- Short decode-temporary lifetimes.
- Separate pair and down LDS layouts where measured behavior differs.
- Zero private storage and zero spills as hard retention gates.

Qwen-specific choices:
- Q3_K pair uses padded LDS rows.
- Q4_K down uses a sixteen-BF16 XOR layout.
- Q5_K uses swizzle4 for M64 and swizzle8 for row tasks.
- IQ2_S down uses a sixteen-BF16 XOR layout.
- IQ2_S pair uses a four-BF16 XOR layout.
- Large down kernels use device-built M-major 128-row tasks. All four adjacent N workgroups for a row task remain contiguous.
- Q4_K and Q5_K row tasks suppress wholly inactive 16-row M minitiles while keeping decode and barriers unconditional.
- IQ2_S row-task suppression is disabled because its route-level result was unstable.

DeepSeek-specific choices:
- IQ2_XXS pair uses two separate weight LDS tiles, width-16 decode, swizzle4, inactive-M consumer suppression, and M64/M128 ownership selected only by exact aggregate rows.
- Q2_K uses width-16 decode that shares each scale/min group and packed shift. Inactive-M consumer suppression is enabled in all three production wrappers.
- Fixed Q8_0 preserves token-major public layout and stages one unswizzled N64/K32 weight tile per fixed group.

### Device row tasks

Qwen large single-down paths use an atomics-free 256-thread prefix-sum setup to build device-resident 128-row tasks. Setup averages approximately `0.004 ms`, so caching route tasks is not a meaningful optimization target.

M-major ordering is required. N-major ordering nearly doubled B16 latency. Pair backward and ordinary small-row paths stay serial; exact IQ2_S aggregate rows 16,384 use the separately qualified row-task exception. DeepSeek Q2_K stays serial because it already exposes 32 N workgroups per expert and thousands of workgroups overall. Adding row tasks would not remove rounded tail arithmetic.

### Code-object resources

Current specialized resources:

| Kernel | VGPR | SGPR | LDS bytes |
| --- | ---: | ---: | ---: |
| Qwen Q3_K pair M64/N64 | 183 | 26 | 10,240 |
| Qwen Q3_K pair M128/N64 | 206 | 26 | 10,240 |
| Qwen IQ2_S pair M64/N64 | 194 | 54 | 8,192 |
| Qwen IQ2_S pair M128/N64 | 219 | 54 | 8,192 |
| Qwen Q4_K down M64/N64 | 96 | 22 | 4,096 |
| Qwen Q4_K down M128/N64 | 173 | 30 | 4,096 |
| Qwen Q4_K row task M128/N128 | 216 | 30 | 8,192 |
| Qwen Q5_K down M64/N64 | 115 | 22 | 4,096 |
| Qwen Q5_K row task M128/N128 | 234 | 26 | 8,192 |
| Qwen IQ2_S down M64/N64 | 90 | 22 | 4,096 |
| Qwen IQ2_S down M128/N64 | 161 | 30 | 4,096 |
| Qwen IQ2_S row task M128/N128 | 238 | 24 | 8,192 |
| DeepSeek IQ2_XXS pair M64/N64 | 209 | 54 | 8,192 |
| DeepSeek IQ2_XXS pair M128/N64 | 239 | 54 | 8,192 |
| DeepSeek Q2_K M64/N64/U1 | 102 | 30 | 4,096 |
| DeepSeek Q2_K M128/N64/U1 | 146 | 31 | 4,096 |
| DeepSeek Q2_K M128/N64/U2 | 160 | 31 | 4,096 |
| DeepSeek fixed Q8_0 M256/N64 | 209 | 23 | 4,096 |
| DeepSeek fixed Q8_0 M192/N64 | 173 | 22 | 4,096 |

Every listed kernel has zero private bytes, zero VGPR/SGPR spills, no scratch or call instructions, wavefront size 32, and `uses_dynamic_stack: false`.

### Correctness and allocation

Qwen and DeepSeek single-projection samples are exact against their packed or independently dequantized BF16 references. Q2_K has zero NRMSE. Differing-element counts with zero absolute error are signed-zero differences.

Fused Qwen and DeepSeek pairs remain in the expected one-rounding envelope, approximately `0.00286` NRMSE against two separately rounded BF16 projections plus addition. Fixed Q8_0 is approximately `5e-5` NRMSE against BF16 BMM.

The production-finalist qualification additionally sampled an independent FP32-accumulating reference at routed rows 49,152 and 196,608 and fixed tokens 32,768. Maximum normalized RMSE was `0.0002777186`, minimum cosine was `0.99999994`, both pair projections and packed weights were mutation-sensitive, inactive experts were bitwise inert, and fixed gradient/weight mutations changed only their own group.

Qwen fused pair incremental allocation remains output-only:

| Batch | Packed pair | AITER pair |
| ---: | ---: | ---: |
| 1 | 64 MiB | 192 MiB |
| 4 | 256 MiB | 768 MiB |
| 16 | 1,024 MiB | 3,072 MiB |

Qwen row-task metadata adds only about 9-10 KiB at batch 4 and 27-28 KiB at batch 16.

## Bottleneck attribution

Grouped backward has no activation-quantization launch. It consumes BF16 cotangents directly, cooperatively decodes forward-layout GGUF weights into BF16 LDS tiles, performs FP32 WMMA accumulation, and writes BF16 `dX`. Relative to AITER or BMM, the extra work is therefore packed-weight decode and layout conversion, grouped work decomposition, and any reuse lost to the packed representation.

The retained profiler controls establish two family-wide negatives:

| Profile point | Occupancy | L2 hit | `ALUStalledByLDS` |
|---|---:|---:|---:|
| Q3_K pair B16 uniform | `43.45%` | `58.24%` | `0.16%` |
| Q4_K down B4 skewed | `30.85%` | `82.45%` | `0.77%` |

Device task construction is approximately `0.004 ms`, selected kernels have zero private storage and spills, and the representative LDS-stall percentages are below one percent. Route-task setup, scratch traffic, spilling, and a saturated LDS interface are therefore not first-order explanations for the remaining losses. These profiler values are retained causal controls, not current latency authorities and not measurements for every format.

Family-level attribution is:

| Family | Current bottleneck conclusion |
|---|---|
| Qwen Q3_K pair | The fused M64/M128 N64/K32 bodies beat AITER at every final point. Padded LDS rows, pair accumulation, and ownership are closed. Decode and accumulator resources remain the theoretical ceiling, but there is no material current deficit to reopen. |
| Qwen IQ2_S pair | The fused bodies also win every final point. Width-16 decode shares grid/sign/scale work; width-8 duplicated scale work and loader groups and regressed. The M128 body already uses `219` VGPRs, so wider ownership requires a new lower-state decoder rather than another tile sweep. |
| Qwen Q4_K/Q5_K down | The large-route deficits remain after M-major row tasks, width-16 decode, format-specific swizzles, bounded prefetch, and inactive-M controls. For Q4_K, the measured high L2 hit and low LDS stalls rule out cache misses and LDS banking as primary causes; negligible task setup and zero spills apply to both formats. Repeated packed scale/min decode is the residual Q4_K cost. Q5_K adds high-bit reconstruction, reaches `234` VGPRs in the row-task body, and is attributed from source and campaign controls rather than the Q4_K counter values. |
| Qwen IQ2_S down | Grid lookup, sign reconstruction, shared scale extraction, and `d` application are repeated for every staged weight tile. Pair and down require different swizzles; width-8, M256/N64, and inactive-M suppression all regressed. The remaining loss is format decode and representation cost, not an untested generic scheduling rule. |
| DeepSeek IQ2_XXS pair | M128/N64 materially improves B4/B16, but B16 remains below AITER. Each K32 step decodes two independent packed weights into separate LDS tiles before one FP32 pair accumulation. The production HIP body uses `239` VGPRs; its historical M192 candidate failed private/spill gates. A later isolated GGTensile M128 body uses `149` VGPRs, so that old resource closure does not apply to the new lower-state schedule and wider-M experiments. The remaining limit is two-weight IQ2_XXS decode plus pair accumulator pressure. |
| DeepSeek Q2_K down | Width-16 decode already shares each scale/min group and packed shift, inactive-M suppression is enabled, and M64/M128 plus U1/U2 ownership are selected by route size. N128 and row-task variants lost. Since serial dispatch already exposes 32 N workgroups per expert, more task descriptors do not remove the repeated Q2_K decode. The remaining B4/B16 gap is packed scale/min reconstruction and limited N64 reuse. |
| DeepSeek fixed Q8_0 | The retained M256/M192 bodies beat BF16 BMM at all final points. Wider decode, swizzle4, and M512 lost. No current bottleneck justifies reopening this family. |

The evidence boundary is deliberate. The Q3_K/Q4_K profiler controls support the conclusions about task setup, spills, L2 behavior, and the LDS interface. IQ2_S, Q5_K, IQ2_XXS, Q2_K, and fixed Q8_0 are attributed from complete-call controls, resource failures, accepted/rejected geometry and mechanism experiments, and the exact decoder dataflow. The table does not assign uncollected counter percentages to those families.

FP32 accumulation is part of the remaining resource floor, not an optional optimization knob. Direct BF16-C was numerically invalid, BF16-state slabs spilled or regressed, and normalized FP16-C remained `33.3%` slower even without its scale scan. Any wider-N or persistent design must first reduce decode/accumulator state while retaining FP32 accuracy.

## Remaining work

### Local kernel work

One evidence-backed isolated GGTensile queue remains pending for the DeepSeek IQ2_XXS pair. It first compares two M128 lower-state projection-read schedules predicted to reduce the retained research artifact from 149 to 139 VGPRs and cross the gfx1151 168-to-144 allocation boundary. It then tests M192/N64, after geometry-derived indexing and wait frontiers are proven, and conditionally M256/N64. Exact identities, resource estimates, occupancy gates, profiling rules, and timing order are recorded in `docs/experiment_ggtensile_grouped_mmq_bwd_pair_iq2_xxs.md`.

The approximate-accumulator sequence remains complete: direct paired BF16-C, K32/K64 FP32 slabs into BF16 state, and normalized FP16-C all failed accuracy, resource, or public-latency requirements. These failures do not relax the exact FP32-accumulation contract for the pending queue.

Wider-N approximate bodies and exact-grid static persistent scheduling remain conditional, not active work. Reopen them only after a new arithmetic mechanism first demonstrates lower register use, acceptable real-weight and dynamic-range error, zero private storage and spills, and better complete-operator latency in an existing N64 body. If that prerequisite is met, evaluate N128/N256 exact shapes before static grids from `{20,40,80,160,256}`. Keep batch-1 sparse dispatch unchanged until measured evidence supports a boundary.

The following neighborhoods are closed by direct controls:
- Runtime full/tail branching and split full/tail task lists.
- N-major tasks, fixed persistent traversal, and broad row-task geometry sweeps.
- K64, two-LDS buffering, GSU, split-K, grouped Stream-K, and direct-to-VGPR variants.
- Broad swizzle, decoder-width, prefetch, and extraction sweeps.
- DeepSeek Q2_K N128 and row-task variants.
- DeepSeek IQ2_XXS width32 and larger swizzles. The historical production/HIP M192 artifact remains rejected under its spilling parent; only the lower-state isolated GGTensile M192/N64 identity is reopened, first at B16. M128 remains retained only at the two qualified aggregate row counts.
- Fixed Q8_0 wider decode, swizzle4, and M512.
- Qwen IQ2_S inactive-M row-task suppression.

Outside the explicit IQ2_XXS queue, a local kernel experiment should be reopened only if new profiler evidence contradicts the current diagnosis. Raw instruction ordering, code-object offsets, and timing movement from byte-identical binaries are not sufficient evidence.

### Deferred model-owned representation and API work

Model-owned compact representations and API changes remain deferred and are not part of the active kernel-local queue. If a separate project takes ownership, the strongest representation target is Qwen Q4_K and IQ2_S single-down weights. This must not become an operator-internal cache.

A viable project must define:
- A prepare API and the exact lossless integer-plus-scale bytes stored per projection.
- Ownership and lifetime across forward and backward.
- Mutation/version invalidation.
- Model-load and peak-memory policy.
- Cold-build and steady-state latency.
- Forward/backward sharing that repays representation construction.
- Behavior for inactive experts and route changes.

The packed GGUF tensor remains authoritative. The likely direction is a WMMA-friendly tile-major integer-plus-scale representation, not a BF16 shadow.

Already rejected representation controls:
- A transient BF16 materialization floor was `1.15-1.34x` slower for IQ2_S and `1.08-1.60x` slower for Q4_K before real packed reads and decode were added.
- One transient projection needs 512 MiB of BF16 workspace and 528-768 MiB incremental peak memory.
- Persistent BF16 shadows require 19 GiB for the remaining Q4_K/IQ2_S down tensors and 60 GiB for all benchmarked Qwen expert projections.
- Persistent expansion is 6.24x for IQ2_S and 3.56x for Q4_K over GGUF storage.
- For the Q4_K/IQ2_S down families, tuned predecoded AITER saves `15.6-33.2 ms` model-wide at B1 and `311.3-338.7 ms` at B16. B4 depends on routing: AITER saves `90.3 ms` for uniform routing, while packed wins the three nonuniform routes by `121.3-163.5 ms`.

Acceptance gates for a compact representation:
- Preserve lossless GGUF semantics and BF16 public inputs/outputs.
- Include preparation, workspace, synchronization, and invalidation costs.
- Preserve sparse inactive-expert behavior.
- Improve grouped and dense shared-down controls, proving that common decode cost was removed.
- Improve all four route distributions or provide a legal shape-based dispatch boundary.
- Preserve pair one-rounding and output allocation when pair families use the representation.
- State cold-call and steady-state memory/latency explicitly.

DeepSeek remains packed in the current API, but the tuned predecoded reference now wins most B4 routed points and all B16 routed points. Any prepared alternative still requires explicit model-owned lifetime, invalidation, memory accounting, cold-start timing, and forward/backward sharing.

## Optimization logs

All timings below are historical step measurements. They explain production choices and should not replace the latest acceptance artifacts.

### DeepSeek P0-P1: harness, baseline, and diagnosis

DeepSeek support added exact Q8_0, Q2_K, and IQ2_XXS decoding, independent BF16 references, routed single/pair APIs, and a dedicated fixed-group operator. Fixed output-A was not represented with fabricated routing metadata.

Generic B1/B4 baseline:

| Family | B1 packed | B4 packed | Reference ratio |
| --- | ---: | ---: | ---: |
| Fixed Q8_0 | 51.038 ms | 212.396 ms | `0.140-0.151x` |
| IQ2_XXS pair | 172.158-189.815 ms | 1,046.712-1,075.508 ms | `0.155-0.411x` |
| Q2_K down | 72.464-94.062 ms | 475.348-519.821 ms | `0.142-0.372x` |

Generic resources were spill-free: Q2_K used 46 VGPRs/24 SGPRs/512-byte LDS, IQ2_XXS pair used 65/40/3,072, and fixed Q8_0 used 35/18/512. The defect was narrow eight-wave N16/K16 ownership and serial row work, not spills. This justified spending resources on four-wave N64/K32 reuse.

Artifacts:

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_baseline_b1_b4.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_pre_ds4_control.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_fixed_harness_check.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_routed_harness_check.json
```

### DeepSeek P2: fixed Q8_0

P2 replaced the generic body with an exact four-wave N64/K32 kernel.

| M tile | B1 | B4 | B16 |
| ---: | ---: | ---: | ---: |
| 64 | 10.693 ms | 43.965 ms | 173.436 ms |
| 128 | 6.400 ms | 26.059 ms | 101.364 ms |
| 256 | 6.237 ms | 25.427 ms | 97.204 ms |

A sequential 25-repeat control confirmed M256 over M128 by `3.91-4.42%`. Width32 regressed `2.4-4.3%` and raised VGPR use. Swizzle4 regressed `3.8-5.1%`. M512 was not compiled because doubling the M256 accumulator set would cross the practical 256-VGPR warning boundary without adding weight reuse.

The retained M256/N64/K32 width16 unswizzled body provides `8.09x`, `8.33x`, and `8.75x` speedups over the generic fixed kernel at B1/B4/B16.

Artifacts:

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m64_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m128_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m256_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m256_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m128_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m256_w32_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m256_s4_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_generic_control_25.json
```

### DeepSeek P3: IQ2_XXS pair

P3 introduced exact `(N,K)=(2048,4096)` M64/N64/K32 pair ownership and cooperative width16 decode. The decoder shares each packed word, two grid lookups, parity-adjusted signs, and scale across sixteen values.

An early decoder took the address of a local packed word and created an 8-byte private segment. Shift/mask extraction restored a register-only M64 body. The contemporaneous M128 source form also retained 8 private bytes and was rejected before timing. The later typed coefficient-only campaign generated a distinct M128 wrapper that compiled with zero private storage and is now retained at the two exact qualified row counts.

Layout controls:
- Width32 had mixed movement: two B1 wins but sparse/boundary and B4 regressions with no legal host-visible separator. Width16 was retained.
- Swizzle4 improved all 12 points by about 20-28% over unswizzled.
- Swizzle16 regressed 24-39% relative to swizzle4.
- Inactive-M suppression improved every point: B1 `5.45-8.44%`, B4 `0.89-4.44%`, B16 `0.58-1.60%`, and `4.08%` geometrically.

Artifacts:

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_tiled_focus.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_iq2xxs_width32_focus.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_iq2xxs_swizzle4_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_iq2xxs_swizzle16_focus.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_iq2xxs_tail_predicate_baseline_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_iq2xxs_tail_predicate_candidate_25.json
```

### DeepSeek P4: Q2_K down

P4 introduced M64 and M128 N64/K32 bodies. Width16 decode shares one Q2_K scale/min group and packed shift across sixteen values.

Reduction unroll controls:
- U2 improved all B4 routes but regressed B1 and one B16 boundary point, so it is dispatched only for 128-511 rows/group.
- Sequential 25-repeat B4 controls confirmed U2 over U1 by `1.83-2.83%` across all four routes.
- U4 lost to U2 by `1.2-4.3%` despite remaining spill-free at 174 VGPRs.

Inactive-M suppression keeps decode and barriers unconditional and skips only cotangent loads, WMMA, and stores for wholly inactive 16-row minitiles. It improved B1 by `8.69-15.15%`, kept every B4 movement within `0.67%`, and improved the complete 12-point matrix geometrically by `4.73%`.

The bounded N128/U1 control compiled at 255 VGPRs and 8 KiB LDS with no spills, but a valid sequential B16 bracket lost every route by `1.18-4.18%`, or `3.06%` geometrically. The temporary wrapper and N-width generalization were removed.

Q2_K row tasks were not added. N64 already launches 32 N workgroups per expert and thousands overall, while row tasks would not eliminate rounded tail arithmetic.

Artifacts:

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_u2_dispatch_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_u1_dispatch_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_tail_predicate_baseline_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_tail_predicate_candidate_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_n128_candidate_valid_b16_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_n64_bracket_b16_25.json
```

### Qwen GB0: baseline and diagnosis

The historical generic baseline won 0/60 points and was `2.39-6.07x` slower than AITER by checkpoint-weighted latency.

| Family | Mean packed/reference throughput | Best | Worst |
| --- | ---: | ---: | ---: |
| Q3_K pair | 0.316x | 0.511x | 0.193x |
| IQ2_S pair | 0.281x | 0.458x | 0.166x |
| Q4_K down | 0.185x | 0.313x | 0.114x |
| Q5_K down | 0.181x | 0.308x | 0.109x |
| IQ2_S down | 0.168x | 0.308x | 0.066x |

The old kernel used eight waves, N16/K16 ownership, one accumulator per wave, scalar decode, and serial 128-row chunks. It was spill-free at 46-52 VGPRs for singles and 58-89 VGPRs for pairs. Profiling showed very low LDS stalls and high Q4_K L2 hit rate, ruling out spills and LDS banking as first-order causes.

A historical one-expert 1,024-row diagnostic showed dense packed speedups of `5.08x` for Q3_K pair, `4.15x` for Q4_K, and `1.95x` for IQ2_S. At 64 rows, grouped and dense Q4_K/IQ2_S were approximately equal. This established that small and large groups needed different bodies and that metadata overhead was not the large-group bottleneck. Dense IQ2_S remains reference code for interpreting this historical result, but current correctness coverage uses the grouped contract and independent dequantization.

Artifacts:

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_baseline_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_one_expert.txt
~/tmp/torch-ggml-ops/rocprof_grouped_bwd_baseline_gate_b16_packed
~/tmp/torch-ggml-ops/rocprof_grouped_bwd_baseline_down_q4_b4_packed
~/tmp/torch-ggml-ops/rocprof_grouped_bwd_baseline_down_iq2_b4_sparse_packed
```

### Qwen GB1-GB3: tiled Q4_K, fused Q3_K, and small bodies

GB1 retained Q4_K M128/N128/K32 with width16 decode and a sixteen-BF16 XOR layout. It improved representative B4/B16 points by 5-8x over baseline.

GB2 retained a fused Q3_K pair with separate padded weight tiles, one accumulator set, and one final BF16 rounding. Representative B4/B16 points improved by 8-14x over baseline and beat AITER by about 2.1-2.9x.

GB3 introduced exact M64/N64/K32 S1 bodies. Universal M128 S2 was rejected because 64-row uniform groups became half-empty bounded tiles: Q3_K uniform regressed from 3.614 to 6.936 ms and Q4_K from 3.590 to 4.185 ms. Q4_K S2 was retained only at 80-127 rows/group. A 25-repeat sparse control measured 4.037 ms S2 versus 5.201 ms S1.

Artifacts:

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step1_q4_matrix.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step2_q3_pair_matrix.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step3_s1.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step3_s2.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_q4_sparse_s2_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_q4_sparse_s1_25.json
```

### Qwen GB4-GB6: row tasks, Q5_K, and IQ2_S

Large Q4_K, Q5_K, and IQ2_S down paths retained device-built M-major 128-row tasks.

Representative task speedups over serial:

| Family/point | Serial | M-major tasks | Speedup |
| --- | ---: | ---: | ---: |
| Q4_K B16 uniform | 40.270 ms | 35.821 ms | 1.12x |
| Q5_K B16 uniform | 37.421 ms | 34.074 ms | 1.10x |
| IQ2_S B16 uniform | 38.992 ms | 32.806 ms | 1.19x |

Rejected scheduling controls:
- N-major order nearly doubled B16 latency.
- Fixed 1,024-program traversal regressed Q4_K/IQ2_S and gave Q5_K 10 spills plus a 44-byte private segment.
- Runtime full/tail branching produced private segments and 2-4 spills.
- Split full/tail lists improved uniform B16 modestly but regressed every nonuniform B4 point by 6-14% because of the second launch.

Q5_K reused the Q4_K framework with width16 low/high decode and bounded packed prefetch. Prefetch improved the initial port by 12-24%. Q5_K S2 was rejected: 4.065 ms versus 3.913 ms S1 on sparse B1.

The project-owned IQ2_S decoder reconstructs sixteen aligned values from two grid entries, two sign bytes, one shared scale nibble, and one `d` factor. M256/N64 large down was rejected despite no spills because doubling N workgroups outweighed M reuse. Selective M128/N64 S2 was retained at 80-127 rows/group, improving B1 sparse from 5.537 to 3.618 ms.

Artifacts:

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step4_row_tasks_mmajor.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step4_row_tasks_nmajor.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step4_persistent1024.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_split_tasks.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step5_q5_prefetch.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_q5_sparse_s2_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step6_iq2.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step6_iq2_n64_reuse.json
```

### Qwen GB7 and final local controls: pair geometry, layouts, and tails

IQ2_S down retained swizzle16. IQ2_S pair retained swizzle4. Swizzle4 regressed down by 20-35% but improved pair by 15-25%, proving that pair and single layouts must remain separate.

IQ2_S and Q3_K large pairs retained M128/N64/K32. Reducing N relieved pair resource pressure and improved representative points by 1-7%. Width8 IQ2_S decode was rejected because it duplicated scale work and doubled loader groups. Pair B16 uniform regressed from 44.101 to 50.274 ms and down B16 uniform from 32.959 to 42.949 ms.

Q5_K swizzle8 was retained only for row tasks. A universal swizzle8 regressed M64/B1 by 5.6-22.5%, while sequential controls improved all row-task routes by `4.2-6.4%` at B4 and `5.2-6.6%` at B16. Row-task VGPR use fell from 256 to 233 before final inactive-M suppression.

Final inactive-M row-task controls retained Q4_K and Q5_K and rejected IQ2_S, as summarized in the latest Qwen results.

Artifacts:

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_swizzle4.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_pair_n64.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_q3_pair_n64.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_width8.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_q5_swizzle8_rowtask_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_q5_swizzle4_rowtask_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_rowtask_tail_predicate_control_false_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_rowtask_tail_predicate_control_true_25.json
```

### Qwen representation controls

Comparable Q4_K/Q5_K down deficits appear in historical dense shared-down controls as well as grouped kernels. IQ2_S diagnosis now remains within grouped kernels and independent dequantized references:

```text
Q4_K dense shared-down: 5.148 ms packed versus 4.223 ms BF16
Q5_K dense shared-down: 5.547 ms packed versus 4.210 ms BF16
```

The retained profiler controls are summarized in [Bottleneck attribution](#bottleneck-attribution). Profiler timing is intentionally not the latency source of record because collection perturbs execution. Event medians in the JSON matrices remain authoritative.

The transient BF16 and persistent shadow controls were rejected for the latency and memory reasons recorded under Remaining work.

Artifacts:

```text
~/tmp/torch-ggml-ops/mmq_bwd_grouped_final_dense_control.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_transient_bf16_floor.json
~/tmp/torch-ggml-ops/rocprof_grouped_bwd_final_down_q4_b4
~/tmp/torch-ggml-ops/rocprof_grouped_bwd_final_down_iq2_b4
```

## Rejected and closed experiments

### Approximate accumulators

The DeepSeek Q2_K M128/N64 controls kept FP32 and reduced-precision accumulation as separate generator problem types. The public default remained FP32 throughout.

| Candidate | Resources | Accuracy and timing | Decision |
| --- | --- | --- | --- |
| Direct paired BF16-C | 122 VGPR, 31 SGPR, 4,096-byte LDS, no private storage or spills | `0.86089` NRMSE at 513 rows | Rejected for accuracy. Unpaired BF16-C reproduced the error, ruling out `op_sel` packing and stores |
| Full-N K32/K64 FP32 slabs into BF16 | 256 VGPR, 647/2,069 spills, 1,568/5,248 private bytes | Not timed after the resource failure | Rejected by the production resource gate |
| Pair-serial K32/K64 slabs | 215/212 VGPR, no private storage or spills | `0.01343/0.00959` NRMSE and `253.624/343.433 ms` versus `187.805 ms` FP32 | Rejected for 35.0%/82.9% latency regressions and higher register use |
| Row-normalized paired FP16-C | 156 VGPR, 22 SGPR, 5,120-byte LDS, no private storage or spills | `0.00723-0.00727` NRMSE from input scale `1e-8` through `1e8`. `1,013.393 ms` with scaling and `250.303 ms` without the scale scan | Rejected. Even the no-scan lower bound was 33.3% slower |

A standalone all-ones K16 chain explained the BF16 failure: output increased by 16 through 32 instructions, reached 512, and then remained at 512 through 128 instructions. Native BF16-C loses small products as persistent magnitude grows, so software probes that round a completed K16 dot product understated hardware error. No reduced-precision body improved registers and latency under an acceptable numerical contract, and all experiment code was reverted.

### Earlier controls

| Family | Candidate | Decision and reason |
| --- | --- | --- |
| Qwen common | Universal S2 | Half-empty 64-row groups regressed uniform B1 |
| Qwen Q5_K | Selective S2 | 4.065 ms versus 3.913 ms S1 on sparse B1 |
| Qwen IQ2_S | M256/N64 | More state and twice the N workgroups regressed B4/B16 |
| Qwen row tasks | N-major order | Nearly doubled B16 latency |
| Qwen row tasks | Fixed 1,024-program traversal | Slower and spilled Q5_K |
| Qwen row tasks | Runtime full/tail branch | Private storage and 2-4 spills |
| Qwen row tasks | Split full/tail lists | Second launch regressed nonuniform B4 by 6-14% |
| Qwen IQ2_S | Width8 decode | Duplicated scale work and loader groups |
| Qwen IQ2_S | Shared pair/down swizzle | Opposite measured preferences. Keep layouts separate |
| Qwen Q5_K | Universal swizzle8 | Regressed B1. Retain only for row tasks |
| Qwen IQ2_S | Inactive-M row-task suppression | Mixed route movement and two regressions |
| DeepSeek IQ2_XXS | Early M128/N64 source form | 8-byte private segment failed the historical gate; superseded by the later resource-clean typed finalist |
| DeepSeek IQ2_XXS | M192/N64 | 32 private bytes, 7 VGPR spills, and scratch instructions failed the coefficient-only resource gate |
| DeepSeek IQ2_XXS | Width32 | Mixed route movement with no legal host separator |
| DeepSeek IQ2_XXS | Swizzle0 or swizzle16 | Swizzle4 was uniformly faster. Swizzle16 lost 24-39% |
| DeepSeek Q2_K | U4 | Lost to U2 by 1.2-4.3% |
| DeepSeek Q2_K | N128/U1 | 255-VGPR cliff and 3.06% geometric regression |
| DeepSeek Q2_K | Row tasks | Sufficient N-grid parallelism. No reduction in rounded tail work |
| DeepSeek fixed Q8_0 | M64/M128 | M256 remains selected at B1/B4; neither candidate met the promotion threshold there |
| DeepSeek fixed Q8_0 | Width32 or swizzle4 | Regressed all target batches |
| DeepSeek fixed Q8_0 | M512 | Accumulator growth would exceed the VGPR warning budget |
| Representation | Transient BF16 | Slower optimistic floor plus 512 MiB per projection |
| Representation | Persistent BF16 | 19-60 GiB residency and poor model-wide latency trade |

Do not restart wholesale direct-to-VGPR, two-LDS pipelines, GSU, split-K, grouped Stream-K, custom LDS barriers, ordinary K64, broad packed prefetch, or architecture-gated direct-global-to-LDS without new profiler evidence. Dense and grouped passes already found these neutral, slower, or resource-invalid.

Do not use compiler-managed local arrays as prefetch state or keep next-iteration packed fragments alive across current WMMA. Longer VGPR lifetimes lost to the retained bounded decode.

Do not replace fused pairs with two public outputs plus `torch.add`. That changes rounding and triples peak pair output allocation.

## Durable design rules

- Judge complete public latency and checkpoint-weighted latency, not isolated instruction counts.
- Use real nonzero cotangents and authoritative packed weights.
- Run GPU measurements sequentially.
- Preserve device-resident routing and inactive-expert sparsity.
- Match cooperative decode width to actual metadata sharing boundaries.
- Keep decode temporaries dead before long WMMA phases.
- Specialize and shorten state before reducing N. Never retain spills.
- Keep pair, single-down, and fixed layouts separate when semantics or measurements differ.
- Keep adjacent N workgroups for one row task contiguous in M-major order.
- Keep decode and barriers unconditional when suppressing only inactive consumer minitiles.
- Use normalized disassembly and source semantics. Raw ELF offsets and instruction ordering are not optimization evidence.
- Require a dense shared-primitive control when grouped and dense kernels share decode or arithmetic helpers.
- Use shape, quant type, token count, and rows/group for dispatch. Do not inspect device routes on the host.
- Treat the 256-VGPR range as a warning boundary even when the compiler reports no spills.

## Validation and packaging

Final validation after consolidation and the retained controls:
- `453 passed, 14 warnings` from the complete repository suite.
- `51 passed, 14 warnings` from the focused grouped-MMQ GPU and typed-config suite.
- Ruff and compileall pass.
- `git diff --check` passes.
- The in-tree extension builds successfully.
- Both new resource-gated kernels pass wave32, zero-private, zero-spill, scratch-free, call-free, and no-dynamic-stack checks.
- Two independent clean no-ccache builds pass `--verify-reproducible --jobs 4`.
- Bundle freshness reports 181 current kernels.
- Qwen 60-point and DeepSeek 27-point final acceptance matrices pass correctness.

Validation commands:

```bash
PYTHONPATH=. pytest -q
ruff check .
python -m compileall -q bench tools torch_ggml_ops tests
python tools/build_mmq_bundle.py --check
python tools/build_mmq_bundle.py --verify-reproducible --jobs 4
git diff --check
```

The concrete-wrapper conversion established byte-identical HSACOs for all 118 then-current symbolic-selector controls. Subsequent tail-predication changes are intentional HIP-semantic changes and remain reproducible and resource-gated.

Earlier apparent B1 movements above 1% were checked against rebuilt byte-identical Q4_K, Q5_K, and IQ2_S artifacts. Sequential warmed 25-repeat controls still ranged from `-2.21%` to `+0.61%`, confirming timing variance rather than an ISA regression. Byte-identical timing movement must not reopen a semantic optimization decision.

The current bundle has 34 grouped-backward entries:
- Seven generic singles.
- Seven generic pairs.
- One generic fixed Q8_0 entry.
- Twelve Qwen geometry-specific entries.
- Seven DeepSeek geometry-specific entries, including the two coefficient-only finalists.

`setup.py build_ext` generates the source-derived bundle and prunes stale artifacts. Local wheels may contain generated HSACOs. Git and source distributions do not.

## Artifact index

### Latest acceptance

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_deepseek_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped-bwd-finalists/deepseek_iq2xxs_m128_confirmation_25.json
~/tmp/torch-ggml-ops/grouped-bwd-finalists/deepseek_fixed_m192_b16_confirmation_25.json
~/tmp/torch-ggml-ops/grouped-bwd-production-final-correctness.json
~/tmp/torch-ggml-ops/grouped-bwd-production-final-resources.json
~/tmp/torch-ggml-ops/grouped-bwd-production-final-rebuild.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_rowtask_tail_predicate_control_false_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_rowtask_tail_predicate_control_true_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_iq2xxs_tail_predicate_baseline_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_iq2xxs_tail_predicate_candidate_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_tail_predicate_baseline_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_tail_predicate_candidate_25.json
```

### DeepSeek retained and rejected controls

```text
~/tmp/torch-ggml-ops/grouped_bwd_bf16_c_control_accuracy.json
~/tmp/torch-ggml-ops/grouped_bwd_bf16_c_candidate_accuracy.json
~/tmp/torch-ggml-ops/grouped_bwd_bf16_c_k32_9.json
~/tmp/torch-ggml-ops/grouped_bwd_bf16_c_k64_9.json
~/tmp/torch-ggml-ops/grouped_bwd_fp16_c_scaled_9.json
~/tmp/torch-ggml-ops/grouped_bwd_fp16_c_no_scale_floor_9.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_baseline_b1_b4.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m256_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_m128_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q80_generic_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_iq2xxs_swizzle4_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_iq2xxs_swizzle16_focus.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_u2_dispatch_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_u1_dispatch_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_n128_candidate_valid_b16_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_ds4_q2k_n64_bracket_b16_25.json
```

### Qwen historical and retained controls

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_baseline_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_final_full.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_one_expert.txt
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step1_q4_matrix.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step2_q3_pair_matrix.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step3_s1.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step4_row_tasks_mmajor.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step5_q5_prefetch.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step6_iq2.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_pair_n64.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_q3_pair_n64.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_q5_swizzle8_rowtask_control_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_q5_swizzle4_rowtask_control_25.json
~/tmp/torch-ggml-ops/mmq_bwd_grouped_final_dense_control.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_transient_bf16_floor.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_short_identical_control_a_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_qwen_short_identical_control_b_25.json
~/tmp/torch-ggml-ops/grouped_bwd_final_readobj.txt
~/tmp/torch-ggml-ops/grouped_bwd_final_disasm.txt
```

### Rejected Qwen controls

```text
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step3_s2.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_q5_sparse_s2_25.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step6_iq2_n64_reuse.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step4_row_tasks_nmajor.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step4_persistent1024.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_split_tasks.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_width8.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_swizzle0.json
~/tmp/torch-ggml-ops/grouped_mmq_bwd_step7_iq2_swizzle4.json
```

Related documents:
- `docs/mmq_bwd_optimization.md` for dense backward.
- `docs/grouped_mmq_fwd_optimization.md` for grouped forward.
- `docs/kernel_bundle.md` for source-only bundle generation and packaging.
