# Grouped MMQ forward optimization

## Status at a glance

This document is the source of truth for grouped MMQ forward kernel behavior, dispatch, benchmark evidence, and retuning decisions on gfx1151. It covers:
- `grouped_mmq_pair` for routed gate/up projections.
- `grouped_mmq` for routed down projections.
- `fixed_grouped_mmq` for the eight-group DeepSeek output-A projection.

Dense MMQ forward and backward are documented separately in `docs/mmq_fwd_optimization.md` and `docs/mmq_bwd_optimization.md`. The generic HSACO bundle and loader contract are documented in `docs/kernel_bundle.md`.

The local kernel pass and coefficient-only full HIP retune are complete for the current Qwen and DeepSeek packed representations. The retained dispatch is exact and resource-clean for all enforced production entries. The final campaign promoted bounded Q4_K mixed tails for Qwen B1/B4, the existing Q2_K J32/J16 body for DeepSeek B4, and a correctness repair for non-divisible J16/J80 activation loads. Backward reduced-precision controls closed the approximate-accumulator prerequisite before forward implementation. Read-only profiling attribution is complete at hardware-counter level for DeepSeek Q2_K B16, at selected-region kernel-trace level for Qwen IQ2_S down B16, and at campaign/source level for the other production families. Prepared-weight work remains outside this repository-local pass.

The final benchmark authorities are:

```text
~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_prior_retuned_final_9.json
~/tmp/torch-ggml-ops/grouped_mmq_fwd_deepseek_prior_retuned_final_9.json
```

They use warmup 3, 9 sequential repeats, correctness rows 256, the promoted exact-key AITER GMM table, and the forward-complete 179-kernel baseline. The current package contains 181 kernels after two backward-only additions; those additions do not change the forward benchmark evidence. The compact final TFLOPS and comparator ratios are reported in [Final benchmark results](#final-benchmark-results). Earlier matrices are experiment evidence rather than current performance authority.

## Prior-aware AITER comparator retune

The prior-aware screen covered all 24 Qwen/DeepSeek batch-shape targets. Stage 1 ranked the current config, one-field neighbors, cross-batch retained configs, and the best archived uniform-route configs on the coefficient-only learned/hash prior. Finalists were then measured over five weighted prior medoids per learned family and five per DeepSeek hash family (three distinct hash B16 profiles), with three warmups and nine alternating-order CUDA-event repeats. DeepSeek scoring used the `40/43` learned and `3/43` hash layer weights.

Before promotion, each selected key was replayed over captured route medoids and the mandatory `uniform`, `skewed`, `sparse`, and `boundary` controls. Captured Qwen base/checkpoint and DeepSeek initial/checkpoint learned routes were retained as separate control states. DeepSeek hash controls were exact token/`tid2eid` projections; the route-identical checkpoint copy was not counted as an independent sample. Every promoted key passed correctness and reversed-order 25-repeat controls on the prior, captured, and synthetic corpora. The full evidence is under:

```text
~/tmp/torch-ggml-ops/aiter_captured_control_profiles.json
~/tmp/torch-ggml-ops/aiter_synthetic_control_profiles.json
~/tmp/torch-ggml-ops/aiter-captured-confirmation/summary.json
~/tmp/torch-ggml-ops/aiter-synthetic-confirmation/summary.json
~/tmp/torch-ggml-ops/aiter-prior-config-correctness.json
```

The promoted forward entries are exact `(M,K,N,transposed_rhs)` keys. Config tuples are ordered `(BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, GROUP_SIZE, GRID_DIM, num_warps, num_stages)`:

| Key and workload | Previous tuple | Promoted tuple |
|---|---|---|
| Qwen B1 down `(16384,512,2048,True)` | `(64,64,32,2,80,4,1)` | `(128,64,32,2,80,4,2)` |
| Qwen B16 down `(262144,512,2048,True)` | `(64,64,128,1,40,8,2)` | `(128,64,64,1,80,8,1)` |
| DeepSeek B4 IQ2_XXS gate/up `(49152,4096,2048,True)` | `(128,64,128,1,160,8,3)` | `(128,64,128,1,40,8,3)` |

The six AITER replacements themselves are table-owned comparator changes only; the separately qualified HIP dispatch change is described below. No producer fusion, route metadata ABI, or model-owned layout changed. The GPU qualification used boundary-heavy nonuniform groups and found old/new bitwise-identical BF16 output, independent per-group BF16 reference NRMSE below `8.0e-5`, and mutation-sensitive output for all six entries.

For attribution, the retuned replay held the archived HIP medians fixed and substituted new AITER medians only at changed keys. Model-call-weighted HIP/AITER ratios therefore moved as follows:

| Model | B1 | B4 | B16 |
|---|---:|---:|---:|
| Qwen forward | `1.529 -> 1.516x` | `1.257 -> 1.257x` | `0.900 -> 0.881x` |
| DeepSeek routed forward, `40/43 + 3/43` | `1.894 -> 1.894x` | `1.284 -> 1.081x` | `1.223 -> 1.223x` |

The initial 12 pooled candidates did not all survive controls. DeepSeek B1 down regressed the synthetic controls by `2.3%`; DeepSeek B1 pair and Qwen B1 backward-down/forward-pair candidates had individual synthetic route failures; Qwen B4 pair/down candidates also regressed uniform controls. A bounded compromise screen of the remaining finalists found one robust Qwen B16 backward-pair configuration, but no replacement for the other six rejected targets. Thus the table does not encode a route-name or router-kind dispatch dimension, and synthetic controls remain part of every future comparator change.

The installed AITER `amd-aiter 0.1.20.dev46+gc4044aacd` `get_config` path was audited separately. Its `gmm.py` implementation accepts shape arguments but currently returns the architecture JSON `default` entry; the underlying source explicitly leaves shape lookup as a TODO. The installed package has no `gfx1151-GMM.json`, while benchmark calls pass explicit configs. Consequently `bench/aiter_gmm_heuristics.py:gmm_config` remains the benchmark-owned exact-key authority and unsupported shapes continue to fail closed. Exact table behavior is covered by `tests/test_aiter_gmm_heuristics.py`.

## Learned-B1 HIP ownership retune

The fitted and captured B1 profiles exposed a gap in the forward ownership predicate: total rows divided by active groups remained below 128 while individual experts reached `800-2048` rows. A focused screen therefore compared the existing production predicate with the existing 64-row device-task body. The isolated screen changed only this ownership choice; it added no kernel body, route input, host readback, or production environment switch.

The nine-repeat screen covered five prior medoids, five physical-ID captured medoids, and all four mandatory synthetic controls. Q3_K pair row tasks improved the prior by `1.1074x` but reached only `1.0179x` on captured routes, so Q3_K retained current dispatch. IQ2_S pair advanced with prior/captured/synthetic gains of `1.1122/1.0213/1.0024x`. Reversed-order 25-repeat confirmation measured `1.1177/1.0224/1.0038x`; the worst individual captured and synthetic profiles were `0.9964x` and `0.9973x`, both above the `0.99x` retention floor. Current and row-task outputs were bitwise identical on every profile.

Production now selects row tasks for the exact paired IQ2_S key `(rows,N,K)=(16384,512,2048)`. Other quant types, single forward calls, and other aggregate row counts retain the prior predicate except for the qualified Qwen IQ2_S B4 down key below. The condition uses only operator kind, quant type, exact geometry, and aggregate rows; it does not inspect `expert_offsets` or use a model/router label. The bundle remains unchanged at 179 kernels.

The post-change learned-route replay measured a `1.0847x` packed-key gain. Holding promoted AITER medians and all unaffected archived HIP medians fixed moved the model-call-weighted Qwen B1 forward ratio from `1.5162x` to `1.5627x`; B4/B16 remain `1.2569/0.8805x`. Evidence:

```text
~/tmp/torch-ggml-ops/hip-b1-rowtask-screen-9.json
~/tmp/torch-ggml-ops/hip-b1-rowtask-confirm-25.json
~/tmp/torch-ggml-ops/hip-b1-rowtask-retuned/qwen_learned_forward.json
~/tmp/torch-ggml-ops/hip-b1-rowtask-retuned/hip_only_analysis.json
```

## Qwen B4 IQ2_S mixed-tail dispatch retune

The first dispatch-only forward screen extended the existing `GroupedFwdSerialIQ2SN2048K512J64J32` body to the exact Qwen IQ2_S down key `(rows,N,K)=(65536,2048,512)`. It changed no device code, route ABI, producer behavior, or bundle artifact. The nine-repeat screen covered five fitted learned-prior medoids and four mandatory synthetic controls. The prior weighted speedup was `1.0321x`, every prior profile was at least `1.0097x`, and every synthetic control was at least `1.0006x`. Reversed-order 25-repeat confirmation measured `1.0228x` weighted over the prior medoids, with a `1.0114x` minimum prior profile and a `1.0021x` minimum synthetic control. All old/new outputs were bitwise identical.

The production selector now chooses the mixed-tail body when `rows == 65536` in addition to the existing small-group threshold. An independent FP32-accumulating reference on two full prior B4 routes and the boundary control reached maximum normalized RMSE `5.8323e-3`. Input and packed-weight mutations changed output on every qualification route. The bundle remains at 179 kernels.

Evidence:

```text
~/tmp/torch-ggml-ops/fwd-dispatch-screen/qwen_iq2s_j32_b4_9.json
~/tmp/torch-ggml-ops/fwd-dispatch-screen/qwen_iq2s_j32_b4_25.json
~/tmp/torch-ggml-ops/fwd-dispatch-screen/qwen_iq2s_b4_correctness.json
```

The first typed-geometry probe for this residual family compiled a standalone J32 wrapper as an isolated 180th HSACO. It passed zero-private-byte, zero-spill, zero-dynamic-stack, and wave32 checks, but its B4 screen reached only `0.8970x` weighted over the prior medoids and `0.8466x` on the worst synthetic control. The candidate remains rejected and the temporary 180-kernel build is not production authority. Its screen is `~/tmp/torch-ggml-ops/fwd-dispatch-screen/qwen_iq2s_pure_j32_b4_5.json`.

The other existing-body forward screens also remained rejected: Qwen Q5_K J64 and J32, Qwen IQ2_S down J64, DeepSeek IQ2_XXS J64/J80, DeepSeek Q2_K J32/J16 under learned and hash priors, and fixed Q8_0 bounded output-A. Their exact reports remain under `~/tmp/torch-ggml-ops/fwd-dispatch-screen/`; every measured alternate was bitwise identical but failed the greater-than-2% aggregate or `0.99x` control gate.

## Coefficient-only full HIP retuning campaign

Status: complete for every prior-addressable grouped-forward family and B in `{1,4,16}`. Search used the fitted coefficient-only routing distributions rather than captured route JSON. "All grouped-forward kernels" means every production family and batch key listed below; generic fallbacks and historical wrappers with no Qwen or DeepSeek workload mass remained correctness and resource controls.

The repository-owned generator `tools/tune_grouped_mmq_prior.py` produced byte-identical repeated corpora with disjoint 512-draw search and confirmation banks, five physical-ID medoids per family/component/batch, and the four mandatory controls. The isolated build added every missing legal J in `{16,32,64,80,128}`, linked row-task candidates, and fixed-Q8 candidates. The first 38 candidates yielded 14 resource-clean bodies and 24 pre-timing rejections. Adding the justified Q4_K J64/J32 mechanism made 39 candidates; after the J16/J80 safety repair, final inspection reported 17 pass and 22 fail.

The complete family decision is:

| Family | Final decision |
|---|---|
| Qwen Q3_K pair `(512,2048)` | Retain serial J64 at B1 and existing row-task J64 at B4/B16. No serial or row-task J alternative passed. |
| Qwen IQ2_S pair `(512,2048)` | Retain the qualified B1 row-task exception and existing B4/B16 ownership. No alternate J passed. |
| Qwen Q4_K down `(2048,512)` | Promote J64 with J32 tails only at aggregate rows 16,384 and 65,536. Keep pure J64 behavior at B16 and all other totals. |
| Qwen Q5_K down `(2048,512)` | Retain current J64/J32 ownership. Every missing geometry failed resources or timing. |
| Qwen IQ2_S down `(2048,512)` | Retain the existing small-route mixed body and qualified B4 exception. No new geometry passed. |
| DeepSeek IQ2_XXS pair `(2048,4096)` | Retain J64 at B1/B4 and repaired J80 at B16. J16 and J64-at-B16 were slower. |
| DeepSeek Q2_K down `(4096,2048)` | Promote the existing J32/J16 body at exact B4 rows 49,152; retain the previous B1 small-route rule and pure J32 at B16. |
| DeepSeek fixed Q8_0 `(1024,4096)` | Retain fixed-group J64 full/bounded. J16, J80, and bounded-dispatch alternatives failed timing. |

The Q4_K bounded body passed 25-repeat search/confirmation at B1 with `1.0591x/1.0828x` weighted prior speedup and at B4 with `1.0243x/1.0279x`. The minimum learned-prior speedups were `1.0163x` or better, and the minimum controls were `0.9934x` or better. B16 keeps the original J64 tail choice inside the same wrapper; its search/confirmation aggregate was `1.0008x/1.0124x`, with every prior and control above `0.9935x`. The typed wrapper serializes the active mechanism as `mixed_j32_tails=True` with `mixed_j32_rows=(16384,65536)` and rejects inactive or unsupported row bounds.

The DeepSeek Q2_K B4 mixed body passed the longer search bank at `1.0227x` and the disjoint confirmation bank at `1.0247x`. Minimum learned/hash prior speedups were `1.0169x/1.0173x`, minimum controls were `0.9998x/0.9996x`, and every output was bitwise equal. The earlier five-repeat `0.9881x` outlier did not reproduce under 25 repeats and was not waived.

The IQ2_XXS J80 screen exposed a correctness defect that aggregate synthetic controls did not find. `deepseek_hash_b16_search_medoid_312` reproducibly faulted because J80 loads `2880` activation integers and the last 128-thread iteration had 64 excess lanes. J16 has the same defect with `576` integers. The repair leaves complete cooperative iterations unchanged and guards only the final non-divisible iteration. Divisible J32/J64/J128 code objects remained byte-identical. The repaired J80 body completed the formerly faulting profile and remained approximately `1.096x` faster than J64 at B16.

Only two of the final 179 production HSACOs changed relative to the pre-promotion bundle: Q4_K N2048/K512 J64 and IQ2_XXS N2048/K4096 J80. The exact Q2_K mixed HSACO was already present; only its B4 selector changed. Clean A/B builds of all three finalists matched exactly in generated source, HSACO, disassembly, metadata, resources, and normalized command line. Production resources are Q4_K `242 VGPR / 46 SGPR`, IQ2_XXS J80 `208/54`, and Q2_K J32/J16 `213/43`, all with wave32, zero private bytes/spills/scratch/calls, and no dynamic stack.

Independent production-default qualification used sparse physical routes with inactive experts and non-aligned tails at every promoted key. The maximum sampled FP32-reference normalized RMSE was `0.01264`, minimum cosine was `0.999920`, all input and active packed-weight mutations changed their intended outputs, inactive-expert mutations were bitwise inert, and paired IQ2_XXS weight mutations changed only their corresponding output.

Evidence:

```text
~/tmp/torch-ggml-ops/grouped-fwd-prior-v1.json
~/tmp/torch-ggml-ops/grouped-fwd-candidate-resources.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/qwen_q4_down_bounded_mixed_search_25.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/qwen_q4_down_bounded_mixed_confirmation_25.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/deepseek_q2_down_mixed_b4_search_25.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/deepseek_q2_down_mixed_b4_confirmation_25.json
~/tmp/torch-ggml-ops/grouped-fwd-all-screen/deepseek_iq2xxs_pair_identity_guard_b16_search_5.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/deepseek_iq2xxs_production_b16_replay.json
~/tmp/torch-ggml-ops/grouped-fwd-all-confirm/production-finalist-correctness.json
~/tmp/torch-ggml-ops/grouped-fwd-production-final-rebuild.json
~/tmp/torch-ggml-ops/grouped-fwd-production-final-resources.json
```

| Production family | Current arithmetic choices | First retuning surface | Reopen only after that surface closes |
|---|---|---|---|
| Qwen Q3_K pair `(N,K)=(512,2048)` | serial J64, row-task J64 | serial/row-task ownership by exact aggregate rows; linked task-row/J geometry | decode/load schedule, output-store schedule |
| Qwen IQ2_S pair `(512,2048)` | serial J64, row-task J64 | ownership, J, and linked task-row geometry | IQ2_S decode schedule and metadata reuse |
| Qwen Q4_K down `(2048,512)` | serial J64 with bounded J32 tails at B1/B4 | J and tail policy | a new down row-task body or decode schedule |
| Qwen Q5_K down `(2048,512)` | serial J64 and small-route J32 | J32/J64 ownership and tail policy | Q5 high-bit decode/load schedule |
| Qwen IQ2_S down `(2048,512)` | serial J64 with optional J32 tail | main J, mixed-tail policy, and threshold | a new down row-task body or decode schedule |
| DeepSeek IQ2_XXS pair `(2048,4096)` | serial J64/J80 | J64/J80 ownership, bounded J neighbors, and tail policy | IQ2_XXS decode/load schedule |
| DeepSeek Q2_K down `(4096,2048)` | rolled J32 with J16 tails at small routes and exact B4 rows | main J, tail J, rolled reduction, and threshold | Q2_K scale/min decode schedule |
| DeepSeek fixed Q8_0 output A `(1024,4096)` | fixed-group J64 full/bounded | J and full/bounded geometry at token batches 1/4/16 | Q8_0 load and store schedule |

This was not only a `constexpr` sweep. The campaign kept three distinct levels, and results from one level were not attributed to another:
- Dispatch-only retuning selects among HSACOs already present in the bundle. It covers serial versus row-task ownership and existing J32/J64/J80, mixed-tail, and rolled-Q2 bodies. This level changes no device instructions and should run first.
- Typed geometry retuning generates a separate wrapper and HSACO for each legal linked configuration. Forward fields are main J, tail J, ownership mode, task-row size, fixed/full-tail mode, and the implemented rolled decode choice. Task-row size and row-task J are one linked field because the current row-task body consumes one J-sized task. Exact N, K, packed blocks per row, quant type, and pair/single ownership are part of the candidate key, not inferred source edits.
- Mechanism retuning changes decode ordering, prefetch, LDS layout, vector load width, accumulation ordering, or stores. It is a new HIP kernel implementation and is attempted only for a family that remains material after dispatch and geometry close. Each mechanism is an explicitly named typed HIP candidate rather than an incidental source rewrite.

`MMQ_I=64`, the 128-thread wave32 workgroup, Q8_1 producer layout, packed GGUF interpretation, and the public route ABI are not ordinary coordinate-descent fields. Varying any of them changes the core algorithm, producer contract, or shared dense behavior. In particular, the quantizer is shared with dense MMQ and is frozen for the arithmetic campaign. A later quantizer campaign would require dense and grouped controls, all three Q8_1 metadata layouts, complete-call timing, and the existing producer ABI qualification.

### Self-contained prior contract

The repository tool is `tools/tune_grouped_mmq_prior.py`. It embeds the coefficients, sampling logic, deterministic profile reduction, and validation. The shared mathematical authority is [Fitted model-level routing prior](ggtensile_plan.md#fitted-model-level-routing-prior), including the Qwen and DeepSeek learned residual laws, constrained rounding, and capture-free DeepSeek hash surrogate. The tuner copies that versioned state into its own source and report; it does not parse Markdown at runtime. This reference shares workload-prior mathematics only. The campaign built and tuned HIP kernels exclusively, and grouped GGTensile implementation remains deferred.

The only learned-prior workload input is `T = physical_batch * 2048`. DeepSeek learned and hash results are reported separately; the `40/43` learned plus `3/43` hash score may rank candidates but cannot hide a hash-component regression. The coefficient-only hash profile remains a benchmark surrogate rather than exact token/`tid2eid` projection. The script may require the Qwen and DeepSeek GGUF paths for packed weights, but it must not import `~/test_no_unsloth`, read files under `~/tmp`, consume captured route histograms, or condition on model layer, depth, checkpoint, adapter, training step, or route label.

For each family and B in `{1,4,16}`, the script deterministically generates a 512-draw search bank and a disjoint 512-draw confirmation bank, then reduces each bank to five physical-expert-ID medoids per router component. Search seeds are `8314159 + B * 104729 + i` for `i in [0,512)`; confirmation adds `1,000,003`, and DeepSeek hash adds `31,000` to the corresponding learned seed. K-medoids uses normalized L1 distance over the full physical 256-expert row vector, starts from source index zero, adds the farthest source at each initialization step, and uses the lowest source index for every tie. Cluster mass supplies the medoid weight. Every generated profile must serialize its seed, active expert IDs, group sizes, row sum, maximum group, coefficient version, bank, cluster membership, and medoid weight. The row sums are exactly Qwen `{16384,65536,262144}` and DeepSeek `{12288,49152,196608}`. Fixed output A instead uses token rows `{2048,8192,32768}` and has no synthetic routing.

`uniform`, `skewed`, `sparse`, and `boundary` remain mandatory deterministic controls generated inside the same script. They are retention controls, not prior mass. Captured routes may be replayed later as external evidence, but neither search nor promotion depends on a capture artifact. The resulting claim is therefore robustness over the fitted prior and deterministic controls, not training-wide production frequency.

### HIP compiler and candidate identity

Changing a shared constant in place and rebuilding the complete extension is not an acceptable search method. `hipcc` may produce a different register allocation or instruction schedule after a semantically neutral source change, and a global header edit can silently change unrelated wrappers. The campaign must instead follow these rules:
- Freeze the current generated wrapper source, transitive include hashes, HSACO, disassembly, metadata, resource counts, compiler identity, and exact command as the baseline for each production kernel.
- Extend the typed bundle configuration only where a real linked field is missing. Candidate source is emitted directly from typed state; the tuner does not regex-rewrite C++ or assembly.
- Compile one wrapper per candidate as an independent, content-addressed HSACO. Other bundle sources and the baseline artifact are not rebuilt while timing that candidate.
- Treat exact generated source bytes plus compiler/toolchain identity as part of the candidate key. A formatting-only or naming-only source variant is a different compiler experiment, not the same geometry.
- Compile every finalist twice in clean directories and require source, HSACO, disassembly, metadata, and resource identity before timing confirmation. A compiler upgrade invalidates that identity and requires requalification.
- Use a tool-owned candidate loader or temporary bundle namespace. Do not add an environment-variable selector, online tuner, or candidate path to production dispatch.
- Reject private bytes, VGPR or SGPR spills, scratch instructions, calls, and dynamic stack before GPU timing. Search artifacts do not enter the production 179-kernel bundle; a promoted new body changes the bundle only through the ordinary exact catalog/build path.

The prior generator is repository-owned. Candidate build/inspection and environment selectors remained isolated under `~/tmp/torch-ggml-ops/fwd-all-tuning/`; none entered production runtime behavior. Production gained only the typed bounded-tail fields and strict validation required by the promoted Q4_K wrapper. `GroupedBackwardConfig` still requires explicit geometry/mechanism fields before any future backward search. The public benchmark remains the final matrix harness and did not acquire a capture-file or production autotuning path.

### Executed forward campaign stages

| Stage | Work | Exit condition |
|---|---|---|
| F0 | Implement the embedded prior, independent deterministic medoid banks and controls; freeze current complete-call and arithmetic-only baselines for all eight families and three batches | Repeated profile generation is byte-identical and every baseline reproduces within the existing timing protocol |
| F1 | Exhaust legal dispatch among existing compatible HSACOs for each exact `(operator,quant,R,N,K)` key | No retained existing-body change remains above the promotion floor |
| F2 | Run bounded one-field-at-a-time typed geometry search, then a bounded cross of interacting linked fields such as `(J,tail_J)` and `(ownership,task_rows,J)` | Every finalist is resource-clean and independently reproducible |
| F3 | For still-material losses only, test one named HIP decode/load/LDS/accumulation/store mechanism at a time | Mechanism attribution is isolated from geometry and compiler-source noise |
| F4 | Confirm in reversed order, run independent references and mutation checks, replay all controls, rebuild twice, and update production dispatch/catalogs | Promotion gates below pass for the exact key |

Promotion is per exact key. It requires a greater-than-2% aggregate gain on the disjoint confirmation medoids, minimum individual search/confirmation profile and router-component speedup of at least `0.99x`, at least `0.99x` on every mandatory synthetic control, and reversed-order 25-repeat confirmation. Correctness requires the existing dense-packed or independently dequantized FP32-accumulating grouped reference, old/new comparison where applicable, input and packed-weight mutation sensitivity, inactive-expert coverage, non-tile-aligned tails, and paired-output independence. Complete public-call timing is authoritative; prequantized arithmetic timing is diagnostic.

Promotion must not add host reads of `expert_offsets`, route labels, model/checkpoint dispatch dimensions, hidden synchronization, prepared weights, or a changed Q8_1/GGUF ABI. Shared changes additionally rerun dense MMQ controls. The final report records rejected candidates as well as winners, recomputes model-call-weighted results without using those weights to hide a family regression, verifies `tools/build_mmq_bundle.py --check`, and records stable evidence paths and the final bundle count.

## Retained production dispatch

Dispatch is static and explainable. It depends on quant type, operator kind, exact production geometry, aggregate rows, and coarse total-row thresholds. It does not inspect device offsets on the host, use route names, use environment variables, or perform online autotuning.

Here `rows` is the total routed row count and `num_groups` is the number of route groups. Thresholds are launch hints. The device kernels still handle inactive experts, skew, and partial row tiles correctly.

| Workload | Exact geometry | Retained branch | Threshold |
|---|---|---|---|
| Qwen gate/up | `(N,K)=(512,2048)` | serial J64 for small groups. Device row-task J64 for large groups and exact paired IQ2_S rows 16,384 | row tasks when `rows >= 128 * num_groups`, or for paired IQ2_S when `rows == 16384` |
| Qwen down Q5_K | `(N,K)=(2048,512)` | standalone serial J32 for small groups. Serial J64 otherwise | J32 when `rows < 128 * num_groups` |
| Qwen down IQ2_S | `(N,K)=(2048,512)` | serial J64 with a J32 final-tail body for small groups. Serial J64 otherwise | mixed tail when `rows == 65536` or `rows < 128 * num_groups` |
| Qwen down Q4_K | `(N,K)=(2048,512)` | serial J64 with a J32 final-tail body only at qualified aggregate rows. Pure J64 tails otherwise | mixed tail when `rows` is 16,384 or 65,536 |
| DeepSeek output A | eight groups, `(N,K)=(1024,4096)` | fixed-group J64 | always |
| DeepSeek gate/up | `(N,K)=(2048,4096)` IQ2_XXS | serial J64 for small/medium groups. Serial J80 for large groups | J80 when `rows >= 512 * num_groups` |
| DeepSeek down | `(N,K)=(4096,2048)` Q2_K | serial J32 with factor-4 scale/min unrolling. Optional J16 tail | J16 tail when `rows == 49152` or `rows < 64 * num_groups` |
| Other supported shapes | runtime geometry | bounds-safe generic J128 fallback | none |

The common retained tile is `I=64`, with 128 threads and four wave32 waves. Qwen and DeepSeek exact branches specialize their production N/K and packed K-block counts at compile time. The fixed Q8_0 path uses a full-N J64 body when the output shape is complete and a bounded variant only for non-multiple output widths.

Gate/up row tasks are built once on the device and reused for the two projections. Each task stores an expert ID and a bounded row interval. B1 Q3_K retains serial ownership under the average-row predicate; the exact B1 IQ2_S pair uses the qualified row-task path. Forward down retains serial expert row ownership because its many output tiles already expose enough parallelism. The down descriptor experiment regressed.

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
  --output ~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_prior_retuned_final_9.json

PYTHONPATH=. python bench/benchmark_grouped_mmq_fwd.py \\
  --model ~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf \\
  --model-family deepseek --warmup 3 --repeats 9 --correctness-rows 256 \\
  --output ~/tmp/torch-ggml-ops/grouped_mmq_fwd_deepseek_prior_retuned_final_9.json
```

## Final benchmark results

The table reports the final uniform-route complete-operator result for every production quant type and full logical matrix shape. Logical shapes use `(M,N,K)`. Paired gate/up executes two matrices with the same routed M, while fixed output A executes eight independent group matrices. Logical TFLOPS includes all matrices in those calls. Complete timing includes Q8_1 activation quantization and workspace allocation.

Routed rows compare against the promoted exact-key BF16 AITER GMM table. Fixed Q8_0 output A is not an AITER GMM shape: it preserves eight independent fixed groups and therefore compares against BF16 `torch.bmm` with public-layout conversion. Ratios above `1.000x` favor packed HIP.

| Model/operator | GGUF type | B | Full logical shape `(M,N,K)` | Packed TFLOPS | Reference | HIP/reference |
|---|---|---:|---:|---:|---|---:|
| Qwen paired gate/up | Q3_K | 1 | `2 x (16384,512,2048)` | 18.04 | AITER GMM | `2.212x` |
| Qwen paired gate/up | Q3_K | 4 | `2 x (65536,512,2048)` | 19.14 | AITER GMM | `1.501x` |
| Qwen paired gate/up | Q3_K | 16 | `2 x (262144,512,2048)` | 19.24 | AITER GMM | `1.042x` |
| Qwen paired gate/up | IQ2_S | 1 | `2 x (16384,512,2048)` | 15.66 | AITER GMM | `1.937x` |
| Qwen paired gate/up | IQ2_S | 4 | `2 x (65536,512,2048)` | 16.20 | AITER GMM | `1.283x` |
| Qwen paired gate/up | IQ2_S | 16 | `2 x (262144,512,2048)` | 16.65 | AITER GMM | `0.913x` |
| Qwen down | IQ2_S | 1 | `(16384,2048,512)` | 15.32 | AITER GMM | `1.279x` |
| Qwen down | IQ2_S | 4 | `(65536,2048,512)` | 19.80 | AITER GMM | `0.862x` |
| Qwen down | IQ2_S | 16 | `(262144,2048,512)` | 20.44 | AITER GMM | `0.801x` |
| Qwen down | Q4_K | 1 | `(16384,2048,512)` | 21.60 | AITER GMM | `1.805x` |
| Qwen down | Q4_K | 4 | `(65536,2048,512)` | 24.04 | AITER GMM | `1.054x` |
| Qwen down | Q4_K | 16 | `(262144,2048,512)` | 24.16 | AITER GMM | `0.955x` |
| Qwen down | Q5_K | 1 | `(16384,2048,512)` | 19.56 | AITER GMM | `1.651x` |
| Qwen down | Q5_K | 4 | `(65536,2048,512)` | 23.65 | AITER GMM | `1.026x` |
| Qwen down | Q5_K | 16 | `(262144,2048,512)` | 24.33 | AITER GMM | `0.968x` |
| DeepSeek paired gate/up | IQ2_XXS | 1 | `2 x (12288,2048,4096)` | 12.44 | AITER GMM | `1.637x` |
| DeepSeek paired gate/up | IQ2_XXS | 4 | `2 x (49152,2048,4096)` | 20.99 | AITER GMM | `1.492x` |
| DeepSeek paired gate/up | IQ2_XXS | 16 | `2 x (196608,2048,4096)` | 22.32 | AITER GMM | `1.024x` |
| DeepSeek down | Q2_K | 1 | `(12288,4096,2048)` | 11.65 | AITER GMM | `1.464x` |
| DeepSeek down | Q2_K | 4 | `(49152,4096,2048)` | 12.75 | AITER GMM | `0.811x` |
| DeepSeek down | Q2_K | 16 | `(196608,4096,2048)` | 12.62 | AITER GMM | `0.540x` |
| DeepSeek fixed output A | Q8_0 | 1 | `8 x (2048,1024,4096)` | 12.57 | BF16 BMM | `0.750x` |
| DeepSeek fixed output A | Q8_0 | 4 | `8 x (8192,1024,4096)` | 12.57 | BF16 BMM | `0.727x` |
| DeepSeek fixed output A | Q8_0 | 16 | `8 x (32768,1024,4096)` | 12.58 | BF16 BMM | `0.716x` |

Across all four routed distributions, Qwen wins `37/60` comparisons and DeepSeek wins `14/24`; fixed Q8_0 wins `0/3` against BMM. The Qwen matrix contains 48 exact packed-dense Q3_K/Q4_K/Q5_K checks; IQ2_S uses grouped pair/single consistency and an independently dequantized BF16 reference. The DeepSeek routed matrix contains 24 exact packed-dense checks, while fixed Q8_0 has independent BMM-reference checks. Independent-reference NRMSE is `0.00597-0.01653` for Qwen and `0.00602-0.01140` for DeepSeek.

## Resource status

The current bundle contains 181 source-built HSACOs: the forward-complete 179-kernel baseline plus two qualified backward kernels. All enforced production resource gates pass with zero private bytes, zero VGPR/SGPR spills, and no dynamic stack. The broad all-artifact scan still reports known fallback spills. Those fallback artifacts are not part of the production zero-spill contract.

Representative retained arithmetic allocations from the retuning pass are:

| Branch | VGPR | SGPR | Private bytes | Spills | Dynamic stack |
|---|---:|---:|---:|---:|---|
| Fixed Q8_0 J64 | 213 | 48 | 0 | 0 | no |
| DeepSeek IQ2_XXS J64 | 229 | 77 | 0 | 0 | no |
| DeepSeek IQ2_XXS J80 | 208 | 54 | 0 | 0 | no |
| DeepSeek Q2_K J32 | 208 | 38 | 0 | 0 | no |
| DeepSeek Q2_K J32/J16 | 213 | 43 | 0 | 0 | no |
| Qwen Q4_K J64/J32 | 242 | 46 | 0 | 0 | no |
| Qwen Q5_K standalone J32 | 189 | 32 | 0 | 0 | no |

The Qwen row-task and Qwen down J64 entries are also enforced and spill-free. Exact per-entry resource metadata is generated with the bundle and is not a dispatch input.

## Bottleneck attribution

The profiling evidence has three levels and they must not be conflated:
- Counter-qualified: DeepSeek Q2_K down B16 has selected-region kernel traces plus issue, compute, memory-instruction, traffic, occupancy, and LDS-stall counters for HIP and AITER.
- Kernel-trace-qualified: Qwen IQ2_S down B16 has selected-region HIP body, Q8_1 quantizer, and AITER kernel traces with launch geometry and resource metadata.
- Campaign/source-attributed: Q3_K, Q4_K, Q5_K, IQ2_XXS, fixed Q8_0, and the paired IQ2_S path are explained by complete-call timing, resource-gated geometry/mechanism controls, and the retained source dataflow. Exact Q2_K counter ratios are not generalized to those families.

### DeepSeek Q2_K down B16

The full counter-qualified review and selected-region traces are retained at:

```text
~/tmp/torch-ggml-ops/profile-grouped-fwd-20260812/deepseek_q2_b16_profile_summary.md
~/tmp/torch-ggml-ops/profile-grouped-fwd-20260812/deepseek_q2k_down_b16_packed_v2/trace_results.db
~/tmp/torch-ggml-ops/profile-grouped-fwd-20260812/deepseek_q2k_down_b16_aiter/trace_results.db
```

Selected-region means over five calls were `252.011 ms` for the HIP J32 body, `5.423 ms` for BF16-to-Q8_1 quantization, and `149.439 ms` for AITER GMM. The packed body alone is `1.686x` AITER; traced HIP kernels including quantization are `1.723x`. Quantization is only `2.11%` of HIP traced kernel time and explains `5.02%` of the HIP-over-AITER excess.

The counter evidence identifies instruction and residency pressure rather than total memory traffic or an LDS-interface limit:

| Metric | HIP/AITER result |
|---|---:|
| Total instructions | `10.88x` |
| VALU instructions | `24.27x` |
| VALU issue cycles | `22.39x` |
| Instruction-fetch waits | `10.36x` |
| Mean occupancy per active CU | `6.25 / 10.67` waves |
| Video-memory fetch | `0.585x` AITER bytes |
| `ALUStalledByLDS` | `0.088% / 33.78%` |

HIP fetches less data than AITER while taking longer, and its ALUs are almost never blocked by a full or unready LDS queue. The retained source explains the excess: each expert/output-tile workgroup serially revisits J32 row tiles; every K256 block reloads packed Q2_K payloads, expands 2-bit values into WMMA-compatible LDS words, forms scale/min metadata, stages two Q8_1 activation planes, and applies scale/min/sum corrections around integer WMMA. The much larger instruction stream, larger code body, and lower residency prevent useful memory-level parallelism. This is not a quantizer-only, DRAM-volume, LDS-bank, or launch-count bottleneck.

### Qwen IQ2_S down B16

The selected-region trace separates the two costs without claiming a full counter diagnosis. Evidence is retained at:

```text
~/tmp/torch-ggml-ops/profile-grouped-fwd-20260812/qwen_iq2s_down_b16_packed/trace_results.db
~/tmp/torch-ggml-ops/profile-grouped-fwd-20260812/qwen_iq2s_down_b16_aiter/trace_results.db
```

| Component | Mean kernel time |
|---|---:|
| HIP IQ2_S J64 body | `25.703 ms` |
| HIP BF16-to-Q8_1 quantizer | `1.849 ms` |
| AITER GMM | `24.094 ms` |

The packed body is `1.067x` AITER and the traced HIP kernel total is `1.144x`. Quantization is `6.71%` of HIP traced kernel time but accounts for `53.5%` of this relatively small traced-kernel excess; the packed body accounts for the other `46.5%`. HIP uses `30,976` bytes of LDS and `232` reported VGPRs versus AITER's `16,384` bytes and `176` VGPRs.

This family therefore has two material costs. Reusing a prepared Q8_1 activation could remove the separate launch for same-input projections, but it would not remove the remaining IQ2_S grid/sign/scale decode and per-tile packed-weight staging. The paired path already shares one Q8_1 workspace, then launches each packed projection body separately, so pair reuse amortizes activation preparation but not weight decode.

### Other production families

| Family | Current attribution |
|---|---|
| Qwen Q3_K pair | It wins B1/B4 and is near parity at B16. Exact N/K specialization, full-row bodies, and serial/row-task ownership closed the local scheduling gap. Remaining work would be Q3 payload/scale decode or a reusable representation, not another ownership sweep. No dedicated counter collection was run because the retained family is not a material current loss. |
| Qwen IQ2_S pair | The B16 loss is consistent with the traced IQ2_S down costs. One activation quantization is shared, but both packed projections still perform their own IQ2_S decode, LDS staging, WMMA, and epilogue. This is source/campaign attribution, not a transfer of the down-kernel counter values. |
| Qwen Q4_K/Q5_K down | B16 is only modestly below AITER. J ownership, bounded tails, two-block K512 traversal, mixed-tail policies, and row-task alternatives are closed. The residual is packed scale/min and, for Q5_K, high-bit reconstruction plus resource pressure; no evidence supports a broad memory or LDS rewrite. |
| DeepSeek IQ2_XXS pair | The repaired J80 body reaches parity at B16 and wins earlier batches. Wider and smaller J choices were slower or resource-invalid. There is no current comparator deficit that justifies another profile campaign; the next possible mechanism would reduce IQ2_XXS lookup/sign decode or reuse decoded weights. |
| DeepSeek fixed Q8_0 | The fixed operator loses to a comparator that starts from BF16 weights. The packed path still quantizes activations and loads/scales int8 GGUF blocks before WMMA. J32/J128, scale staging, rolled loops, and workspace-layout changes failed, leaving a representation mismatch rather than an open geometry choice. |

## Remaining bottleneck and work boundary

No additional local schedule or accumulator change is justified for the current packed representations. The counter-qualified Q2_K result, the IQ2_S trace, and the family-specific control campaigns all point beyond launch tuning. Backward established that direct native BF16-C is numerically invalid for long reductions, K32/K64 FP32 slabs increase resources and latency, and normalized FP16-C remains 33.3% slower even when scale calculation is removed. Forward approximate accumulation, wider J tiles based on reclaimed registers, and static persistent scheduling are therefore closed at their prerequisite.

Reopen that sequence only after a new arithmetic mechanism demonstrates lower register use, acceptable real-weight and dynamic-range error, zero private storage and spills, and better public backward latency in an existing N64 body. Forward must then measure its existing geometry first, preserve integer MMQ and Q8_1 workspace semantics, and test only DeepSeek Q2_K J64, DeepSeek IQ2_XXS J128, or Qwen IQ2_S J128 where the latest B16 results justify it. Static grids from `{20,40,80,160,256}` come last.

The tuned reference exposes representation deficits in Qwen IQ2_S down, Qwen IQ2_S gate/up at B16, and DeepSeek Q2_K at B4/B16. DeepSeek IQ2_XXS is now approximately at parity at B16 rather than a material loss. Partial routed tiles repeat packed metadata interpretation, scale formation, packed loads, and decoded operand construction. The retained kernels have already removed unnecessary row decomposition and bounded-tail work, while AITER begins from predecoded BF16 weights.

DeepSeek fixed Q8_0 is also representation-limited. The complete public operator remains slower than BMM because the comparison begins from already dequantized BF16 weights. The two selected-region cases put packed multiplication at `93.29-97.89%` of absolute HIP traced kernel time; quantization is material to the small IQ2_S gap but does not explain the representation-level comparison.

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

## Notable optimization log

Earlier full timing matrices are intentionally omitted because the final artifacts above are the only performance authority. The retained historical evidence is limited to changes that explain the current structure or close a tempting alternative.

### Retained milestones

- Compile-time J64 and exact production N/K specialization removed the original spill cliff. Representative complete-call timing moved from `11.185` to `6.170 ms` for Qwen Q3_K B1 and from `6.874` to `2.842 ms` for Qwen Q4_K B1.
- Splitting exact full-row and bounded-tail bodies removed row decomposition from the common path. Q3_K B1 moved from `6.170` to `3.735 ms`, while Q4_K B1 moved from `2.842` to `1.599 ms`.
- Explicit two-block K512 traversal improved retained down bodies by roughly `1.02-1.21x`; affine pointer increments gave smaller durable gate/up gains.
- Device row-task descriptors improved large gate/up groups by roughly `1-2%`. The later learned-route screen promoted the existing row-task IQ2_S pair body at Qwen B1 after `1.1177x` prior confirmation; down projections retained serial ownership.
- The standalone gfx1151 bundle made source, resource, and code-object identity deterministic. Q5_K retained standalone J32 only for small nonuniform routes, where it improved approximately `14-16%` while the uniform route regressed.
- Exact DeepSeek IQ2_XXS J64 removed spills and moved B1 uniform pair timing from `77.549` to `33.581 ms`. J80 was retained at B16 and, after the non-divisible-load repair, remained approximately `1.096x` faster than J64.
- Exact DeepSeek Q2_K J32 plus factor-4 scale/min unrolling removed spills; the J32/J16 body was later promoted at exact B4 rows with `1.0247x` confirmation.
- Bounded Q4_K J32 tails were promoted only at Qwen B1/B4, with confirmation gains of `1.0828x` and `1.0279x`. Qwen IQ2_S down retained its exact B4 mixed-tail dispatch at `1.0228x` confirmation.

### Closed alternatives

- A complete decoded-weight LDS cache reduced effective speed to roughly `0.60-0.69x` because its LDS footprint reduced residency. The later IQ2_S-specific cache also regressed large batches by up to `13.8%`.
- J128, I128, and eight-wave ownership crossed resource limits, reduced residency, or violated output ownership. The four-wave I64/J64 workgroup remains the valid base geometry.
- Full/tail split launches, replicated invalid lanes, broad mixed-tail bodies, and compiler-managed packed-prefetch arrays either added launch work or crossed the spill cliff.
- Q5_K one-launch J64/J32 candidates traded a small uniform gain for `3.6-8.8%` nonuniform regressions, so the standalone dispatch remains.
- Fixed Q8_0 J32/J128, scale staging, rolled dot loops, and group-major Q8_1 workspace changes were invalid, resource-heavy, or neutral. Reopening this family requires a representation change rather than another local tile sweep.
- Approximate accumulators did not supply a lower-register prerequisite: direct BF16-C was numerically invalid, K32/K64 BF16-state slabs were slower, and normalized FP16-C remained `33.3%` slower.
- Q2_K metadata staging and broader IQ2_S loop/cache changes remained spill-free in some cases but regressed complete-call timing, so shorter decode lifetimes are retained.

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
- Cooperative full-tile activation loads require an explicit final-iteration guard whenever `J * MMQ_TILE_Y_K` is not divisible by 128. Aggregate-only synthetic controls are not sufficient correctness evidence for that condition.
- Activation quantization is family-dependent: it is `2.11%` of traced HIP kernel time for DeepSeek Q2_K B16 and `6.71%` for Qwen IQ2_S down B16. It is material to the small IQ2_S gap, but packed multiplication still accounts for more than `93%` of absolute traced HIP kernel time in both measured cases.
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

The complete final matrices are `~/tmp/torch-ggml-ops/grouped_mmq_fwd_qwen_prior_retuned_final_9.json` and `~/tmp/torch-ggml-ops/grouped_mmq_fwd_deepseek_prior_retuned_final_9.json`. They must be preserved as regression controls for future representation-level work. A future change must rerun both matrices, packed-reference exactness, independent-reference error checks, resource gates, and the relevant 25-repeat sequential controls.

## Final status

The grouped MMQ forward pass is complete for the current packed Qwen and DeepSeek representations.

The retained implementation uses compile-time exact shapes, J64/J32/J16/J80 only where measured, bounded Q4_K J32 tails at Qwen B1/B4, the existing Q2_K J16 tails at DeepSeek B4, guarded non-divisible full-tile activation loads, device row tasks only for qualified gate/up keys, two-block down unrolling, pointer-increment gate/up traversal, and deterministic standalone HSACO artifacts. Against the exact target-specific AITER table, Qwen wins 37/60 comparisons and DeepSeek wins 14/24 routed comparisons, with packed-reference and independent-reference checks passing throughout.

The remaining losses are representation-level comparisons against already decoded BF16 weights. Reduced-precision accumulation is also closed by the completed backward controls. No further local tile, accumulator, or scheduling sweep is planned until a compact reusable decoded representation, a transient dense stage, or a new profile-supported arithmetic mechanism can be evaluated end to end.
