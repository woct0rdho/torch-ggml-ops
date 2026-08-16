# GGTensile Design

## Purpose

GGTensile is the repository-local assembly kernel generator for packed GGUF matrix multiplication. It accepts one exact `ProblemType`, one exact `ProblemSize`, and one complete direction-specific solution, then either emits reproducible assembly or returns a structured rejection reason.

This document is authoritative for the generic design, implementation principles, supported scope, current implementation status, and completed writer-refactor contracts. Format-specific problem definitions, measurements, rejected mechanisms, artifact identities, and campaign chronology belong in the experiment records. The completed refactor is represented here by its durable semantic authorities, argument rules, ownership boundaries, source-stability policy, and qualification evidence; no separate migration plan is required.

GGTensile uses a deliberately small ROCISA surface inspired by TensileLite: structured modules and metadata, explicit register pools, and direct assembler/linker invocation. It does not import TensileLite's solution, problem-type, search, scheduling, allocation, or library-generation machinery. Existing HIP kernels remain the correctness control, performance control, and runtime fallback.

## Project Scope

The current backend targets gfx1151, code-object version 5, wave32, and WMMA V1. Support expands explicitly by operation, direction, quant format, exact shape family, activation layout, destination type, and ISA.

The project boundary is:
- exact positive problem sizes supplied at generation time.
- explicit complete solution identity with no silent repair or inferred tuning values.
- stable public forward and backward writer constructors plus `source()` and `write()` behavior.
- the exact 40-byte MMQ kernel ABI, code-object v5 metadata, and wave32 launch contract.
- authoritative packed GGUF weights, decoded inside the generated kernel unless a separate public representation contract is designed.
- shape dimensions exactly divisible by the selected ownership, vector widths, and reduction depth; edge tiles reject.
- exact-key runtime applicability and HIP fallback for every mismatch.
- formula-derived resource admission under the target VGPR, SGPR, LDS, and code-object constraints.
- zero private storage, spills, scratch instructions, calls, and dynamic stack.
- unchanged public HIP dispatch and the then-current 179-kernel bundle unless a separate integration review promotes exact GGTensile artifacts. A later grouped-backward review independently expanded the package to 181 kernels.
- no split reduction, atomics, persistent traversal, prepared-weight cache, hidden dense shadow, or external decode workspace unless the mechanism receives an explicit model, ABI, lifetime, validation, and dispatch contract.

Unsupported problems and solutions are normal validation results, not generator failures. Broader ISAs, edge handling, new operations, grouped ownership, prepared representations, and multi-kernel reductions are separate expansion projects rather than implicit capabilities.

## Required Format Inventory

The workload, not the set of types understood by generic GGUF code, defines required production coverage.

| Operation | Workload | Required formats |
| --- | --- | --- |
| MMQ | Qwen ordinary projections and language-model head | `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` |
| MMQ | DeepSeek ordinary projections and language-model head | `Q8_0` |
| Grouped MMQ | Qwen experts | `Q3_K`, `Q4_K`, `Q5_K`, `IQ2_S` |
| Grouped MMQ | DeepSeek experts | `Q2_K`, `IQ2_XXS`, fixed-group `Q8_0` |

The MMQ union is `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0`. The grouped union is `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `IQ2_XXS`, `IQ2_S`, and fixed-group `Q8_0`.

`Q6_K` has no current grouped production workload. `Q2_K`, `IQ2_XXS`, and `IQ2_S` have no current required ordinary MMQ campaign. Ordinary `IQ2_S` tests and production inventory are intentionally excluded. Fixed-group `Q8_0` is a distinct grouped contract and is not implied by ordinary `Q8_0` coverage.

### Initial routed forward campaign

The first grouped expansion is forward `grouped_mmq`: routed, non-fixed, and one projection per call. It covers `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `IQ2_XXS`, and `IQ2_S`. `Q8_0` is excluded because its required grouped workload is the separate fixed-group output-A contract. `Q6_K` is excluded because no current grouped production workload uses it.

The production disposition belongs to an exact format/shape target, not to a quant format in isolation:
- retained non-paired targets are down projections whose final production operator remains `grouped_mmq`.
- paired-successor precursors are gate/up projections qualified one projection at a time in the original grouped campaign. They establish the packed decoder, arithmetic, ownership, and artifact controls, but are not final production selections for those calls.
- the Qwen `IQ2_S` gate/up target has now completed a dedicated `grouped_mmq_pair` research campaign with shared Q8_1 workspace ownership, two packed weights, two output destinations, device row tasks, synchronization, resources, correctness, and complete-call timing. Public integration remains separate.

`IQ2_S` intentionally appears in both categories: `(N,K)=(2048,512)` is a retained single-routed down target, while `(N,K)=(512,2048)` is a paired-successor gate/up precursor.

For expert group `g`, the logical matrix operation is

```text
A_g[M_g,K] x W_g^T[K,N] -> C_g[M_g,N]
```

where `M_g` is the runtime row count for that active expert and `R = sum_g(M_g)` is the aggregate routed-row count. Every routed weight bank has 256 logical experts with shape `[256,N,K]`; the physical packed bank is `[256,N,packed_row_bytes]`.

Define the required aggregate row sets as `R_qwen = {16384, 65536, 262144}` and `R_deepseek = {12288, 49152, 196608}`. Each `(R_set,N,K)` entry below denotes three required operator-level `(R,N,K)` workload keys.

| Disposition | Workload | Quant | Expert `(N,K)` | Physical packed weights | Required aggregate `(R,N,K)` |
| --- | --- | --- | ---: | ---: | ---: |
| Retained non-paired | Qwen down outer main | `Q4_K` | `(2048,512)` | `[256,2048,288]` | `(R_qwen,2048,512)` |
| Retained non-paired | Qwen down outer edge | `Q5_K` | `(2048,512)` | `[256,2048,352]` | `(R_qwen,2048,512)` |
| Retained non-paired | Qwen down middle | `IQ2_S` | `(2048,512)` | `[256,2048,164]` | `(R_qwen,2048,512)` |
| Retained non-paired | DeepSeek down | `Q2_K` | `(4096,2048)` | `[256,4096,672]` | `(R_deepseek,4096,2048)` |
| Superseded by paired | Qwen gate/up outer | `Q3_K` | `(512,2048)` | `[256,512,880]` per projection | `(R_qwen,512,2048)` |
| Qualified paired research | Qwen gate/up middle | `IQ2_S` | `(512,2048)` | `[256,512,656]` per projection | `(R_qwen,512,2048)` |
| Superseded by paired | DeepSeek gate/up | `IQ2_XXS` | `(2048,4096)` | `[256,2048,1056]` per projection | `(R_deepseek,2048,4096)` |

The routed-row sets derive from sequence length 2,048 and the model top-k. Uniform routing gives the following concrete per-expert matrix heights:

| Model | Top-k | Physical batch | Aggregate `R` | Uniform `M_g` for 256 experts |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 8 | 1 | 16,384 | 64 |
| Qwen | 8 | 4 | 65,536 | 256 |
| Qwen | 8 | 16 | 262,144 | 1,024 |
| DeepSeek | 6 | 1 | 12,288 | 48 |
| DeepSeek | 6 | 4 | 49,152 | 192 |
| DeepSeek | 6 | 16 | 196,608 | 768 |

The initial inventory therefore contains 21 quant/aggregate-shape workload keys. Every key must be exercised under uniform, skewed, sparse, and boundary routing. Each paired-successor key must be checked independently with both the gate and up packed tensors even though they share one format/shape identity. Uniform, skewed, and boundary use all 256 experts; sparse uses 192, 224, and 240 active experts at physical batches 1, 4, and 16. Boundary routing includes `M_g` around 1, 16, 64, and 128, so a uniform `M_g` is benchmark context rather than an edge-free capability assumption.

`N` and `K` remain exact production dimensions, but `M_g` is metadata-driven and may not be a tile multiple. Grouped lowering must bounds-check row ownership and tails; this is an explicit grouped contract expansion and does not relax edge-tile rejection for the existing ordinary MMQ writers. Distribution names and model labels are evidence metadata, never candidate or dispatch inputs; the grouped problem contract must represent any required row and group bounds directly.

### Captured routing corpus and weighting

Static model inspection, aggregate `active_experts` counts, and one captured training batch are not sufficient to define a representative grouped-MMQ workload. They are sufficient to reject a single fixed synthetic distribution and to establish the required corpus schema, but performance promotion must use per-layer route histograms captured from real training states.

The current forward-only pilot captured the same 2,048-token dataset samples through the real model paths. Qwen used 16 shuffled B1 microbatches at seed `19260817` with the zero-adapter base and checkpoint `7400`; four of those samples were replayed with compact route selections to measure exact route drift. DeepSeek used 16 shuffled B1 microbatches at the same seed with zero-initialized rank-4 adapters, and the existing one-step audit supplies a same-batch before/after update comparison.

| Model/state | Router population | Captured B1 result | Aggregate result |
| --- | --- | --- | --- |
| Qwen checkpoint `7400` | 40 learned routers, top-8 | `M_g` maximum range `516-2047`, median `1675`; active experts `73-256`, median `192`; layer-max padding inflation median `18.22x`, maximum `29.15x` | B16 sum of 16 real B1 histograms: active experts `226-256`, median `253`; `M_g` maximum `10399-31956`, median `24928`; layer-max padding inflation median `24.00x` |
| DeepSeek zero-B initial state | 40 learned top-6 routers plus 3 hash routers | learned active experts `178-253`, median `235`; learned `M_g` maximum `477-1982`, median `1074`; learned layer-max padding inflation median `20.08x`, maximum `37.64x`; hash routers always use all 256 experts with `M_g` maximum `168-291` | B16 sum of 16 real B1 histograms: learned active experts `248-256`, median `255`; learned `M_g` maximum `10162-26070`, median `14969.5`; layer-max padding inflation median `19.34x`, maximum `33.95x` |

These measurements are workload evidence, not dispatch constants. The Qwen base-to-checkpoint replay changed `67.6%` of token-layer top-8 expert sets and `12.4%` of route memberships across four paired samples; ordered slots changed `53.0%`. The DeepSeek one-step audit changed route histograms in `40/43` layers, with a histogram-derived assignment lower bound of `1.4%` at the median and `2.1%` at the maximum. Repeated checkpoint replay without an update was histogram-identical, so the route changes are model-state effects rather than collector nondeterminism.

The benchmark corpus therefore has one record per `(model, checkpoint, physical batch, router kind, layer, sample, projection, quant-shape key)` containing the full `rows_per_expert[256]`, selected-token count, top-k, dataset/sample identity, and training-state metadata. Aggregate B4 and B16 histograms may be formed by summing independent B1 records when examples are independent; direct B4/B16 captures remain preferred whenever their physical batch behavior is itself under evaluation. Gate, up, and down projections in one layer reuse the layer's expert assignments and offsets, but retain separate timing records because their `(N,K)`, quantization, paired ownership, and output behavior differ.

Realistic ranking uses frequency-weighted histograms stratified by model, checkpoint or training phase, router kind, layer, physical batch, and projection/shape family. The layer and projection invocation frequencies must be explicit; a model-wide average histogram is not a substitute. A captured pilot is not training-wide evidence: the Qwen checkpoint `7400` and the DeepSeek zero-B initial corpus must be extended with multiple dataset batches at early, middle, and late checkpoints, plus same-batch before/after-update captures where router trainability is in question. Multiple data seeds are required before claiming dataset-wide frequency weights.

Uniform, skewed, sparse, and boundary distributions remain mandatory correctness and robustness controls for every exact workload key. They must not replace captured histograms for performance ranking. In particular, fixing every logical `M_g` to a layer maximum is unsupported: the observed distributions contain many small experts beside one or a few large experts, and maximum padding can issue an order of magnitude more rows than the useful route work. Grouped lowering and dispatch must consume the actual per-expert row metadata; distribution labels and model names remain evidence metadata only.

Pilot artifacts are retained outside the repository under `~/tmp/test_no_unsloth/qwen_routes_*`, `~/tmp/test_no_unsloth/deepseek_routes_*`, and the corresponding `*_route_capture_analysis.json` files. They are reproducible evidence and not runtime inputs or public dispatch policy.

### Fitted model-level routing prior

The model-level prior fit is stored outside the repository at `~/tmp/test_no_unsloth/model_level_route_prior_fit_v2.json` with the human-readable report at `~/tmp/test_no_unsloth/model_level_route_prior_fit_v2.md`. The generator and validation are `~/tmp/test_no_unsloth/fit_model_level_route_prior_v2.py`. The fit pools layers, depth, and base/checkpoint states; none is an input to the prior. The only physical-size input is `T = physical_batch * sequence_length`.

For Qwen and DeepSeek learned routers, sample active support and ranked skew as follows:

```text
x = log(T / 2048)
y_A = a0 + a_log_tokens*x + epsilon_A
A = round(257*sigmoid(y_A) - 0.5), clipped to [top_k, 256]
log(alpha) = b0 + b_log_tokens*x + b_active_residual*epsilon_A + epsilon_alpha
q_r = min(1, C*(r + shift)^(-alpha)), 1 <= r <= A
q_r = 0, r > A
sum(q_r) = top_k
```

For both non-hash priors, write `StudentT(nu,loc,scale)` for a location-scale Student-t variate and sample each residual as `clip(StudentT(nu,loc,scale),lo,hi)`. The complete residual laws are:

| Model | Residual | `nu` | `loc` | `scale` | Clip `[lo,hi]` |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen | `epsilon_A` | `8.3978226526` | `-0.0335118652` | `0.9830493404` | `[-2.5036492445,3.7809143778]` |
| Qwen | `epsilon_alpha` | `5.2038953632` | `0.0054814884` | `0.1025706842` | `[-0.4463245132,0.3833201702]` |
| DeepSeek learned | `epsilon_A` | `33.5987235964` | `-0.0013194870` | `0.8328268568` | `[-1.6829685257,2.7587218851]` |
| DeepSeek learned | `epsilon_alpha` | `5.1551932778` | `-0.0071110386` | `0.0923527684` | `[-0.3180690433,0.3604795507]` |

After solving for the unadjusted inclusion vector `q`, apply the fitted head multiplier `h` exactly as follows:

```text
q1_adjusted = min(1, h*q1)
q_r_adjusted = q_r*(top_k-q1_adjusted)/(top_k-q1), r = 2..A
```

This transfers mass to rank 1 while preserving `sum(q_r)=top_k`. Constrained largest-remainder rounding of `T*q_r` produces `M_g`, and a uniform random permutation assigns ranks to the 256 expert identifiers. The fitted Qwen parameters are `shift=11.5465232873`, `h=1.1255020052`, `a0=1.3568429238`, `a_log_tokens=1.1367804236`, `b0=0.7376182182`, `b_log_tokens=-0.0215993943`, and `b_active_residual=-0.1730074363`. The fitted DeepSeek learned parameters are `shift=6.3223820835`, `h=1.0643729189`, `a0=2.2468973539`, `a_log_tokens=0.8906164814`, `b0=0.4081577973`, `b_log_tokens=-0.0211298377`, and `b_active_residual=-0.0754167931`. Qwen uses top-k 8; DeepSeek learned routers use top-k 6.

The active-residual coefficient preserves the observed coupling between support and skew: layers with fewer active experts are also more concentrated. Its ablation leaves coarse physical-batch medians similar, but makes support and alpha conditionally independent. In the pooled fit, observed active/alpha correlations are `-0.778` for Qwen and `-0.616` for DeepSeek learned routers; removing the term reduces log-alpha regression `R^2` from `0.702` to `0.011` for Qwen and from `0.257` to `0.031` for DeepSeek. The coefficient therefore remains in the realistic prior. It may be omitted only in an explicitly lower-fidelity compact fallback after kernel-ranking sensitivity is checked.

DeepSeek hash routers are a separate `3/43` component and have a simpler structural formula than a ranked power law. Let `n_v` be the number of occurrences of physical token ID `v` among the `T=B*S` positions passed through the router, and let `H_l(v,j)` be slot `j` of hash layer `l`'s frozen `tid2eid[v]` row. Then the exact row count is

```text
M_l,e = sum_v n_v * sum_(j=1..6) 1[H_l(v,j) = e]
```

This equation explains the observed shape: one frequent token adds the same multiplicity to six experts, producing a correlated six-expert head that an independent rank or count law misses. When input IDs and `tid2eid` are available, use this exact projection rather than a fitted route prior. For an exchangeable benchmark with an unavailable hash table, sample a captured token-frequency histogram `n_v`, assign each token type one fixed uniform six-expert subset `H(v)`, reuse that mapping across all samples aggregated into the same synthetic layer, and randomly permute expert identities. Do not resample `H(v)` independently per microbatch; that would erase the persistent load from frequent token IDs.

The 16 captured S2048 samples contained a median `979` unique token IDs per sample, with 5th-95th percentile range `837-1137`; maximum token multiplicity had median `173.5` and range `123.5-215.8`. The exchangeable token-hash projection gave the following prior-predictive medians:

| Hash metric | B1 observed / projected | B4 observed / projected | B16 observed / projected |
| --- | ---: | ---: | ---: |
| Active experts | `256 / 256` | `256 / 256` | `256 / 256` |
| `max(M_g)/T` | `0.111 / 0.115` | `0.106 / 0.109` | `0.098 / 0.105` |
| Effective experts | `181.5 / 180.7` | `195.0 / 193.8` | `200.9 / 198.9` |
| Maximum-padding inflation | `4.73x / 4.90x` | `4.54x / 4.64x` | `4.20x / 4.46x` |

A one-parameter symmetric Dirichlet-multinomial and a one-parameter logistic-normal multinomial are not adequate replacements. Matching the Dirichlet model jointly to head and bulk metrics selected per-expert concentration `1.8119` but predicted only `165.6` effective experts at B16 versus `200.9` observed. Matching a logistic-normal body with `sigma=0.5005` predicted B16 `max(M_g)/T=0.0833` versus `0.0984` observed. A token-frequency Zipf fit, `p_v proportional to r^-0.6864`, had mean/median token-histogram TV `0.101/0.096` and does not retain special-token spikes by itself.

For future synthetic benchmark generation where input IDs and `tid2eid` are unavailable, use the following capture-free four-coefficient row prior. It models the row-load object observed by grouped MMQ; it is not an exact replacement for token hashing. Let `T >= 2048`, `B=T/2048`, `E=256`, and `k=6`:

```text
kappa_B = B*(kappa_rho + 1) - 1
rho ~ Beta(mu_rho*kappa_B, (1-mu_rho)*kappa_B)
a(T) = a0 * B^beta
u ~ Dirichlet(a(T) * 1_E)
H = a uniform k-subset of {0,..,E-1}
p_e = (1-rho)*u_e + (rho/k)*1[e in H]
M = bounded_largest_remainder( k*T*p, lower=1, upper=T )
```

The fitted coefficients are:

| Coefficient | Value |
| --- | ---: |
| `mu_rho` | `0.0765686035` |
| `kappa_rho` | `111.4009102046` |
| `a0` | `6.6600305083` |
| `beta` | `0.4771750368` |

`rho` represents the persistent mass of a dominant repeated token; its beta concentration makes that mass variation decrease as physical tokens aggregate. `H` represents the six experts selected by that token. The symmetric Dirichlet body absorbs the remaining token-frequency and hash-map collisions. Uniform expert identities are valid only under the current exchangeable grouped-MMQ contract; paired projections should reuse one generated `M` profile.

Prior-predictive medians from the independent initial capture, with the route-identical checkpoint retained only as an integrity check, are:

| `T` | Metric | Observed | Formula |
| ---: | --- | ---: | ---: |
| `2048` | `max(M_g)/T` | `0.11084` | `0.10742` |
| `2048` | effective experts | `181.53` | `188.69` |
| `8192` | `max(M_g)/T` | `0.10638` | `0.10583` |
| `8192` | effective experts | `195.04` | `195.78` |
| `32768` | `max(M_g)/T` | `0.09845` | `0.10373` |
| `32768` | effective experts | `200.87` | `200.44` |

The B1/B4 discrepancies are under `4%` for the two independent shape metrics; the B16 maximum is `5.4%` high with only three distinct hash-layer profiles, so it is not used to add another coefficient. The sampler always emits all 256 experts, `sum(M_g)=6T`, and `M_g<=T`. The fit and capture-free implementation are `~/tmp/test_no_unsloth/fit_hash_head_body_prior.py` and `~/tmp/test_no_unsloth/sample_hash_head_body_prior.py`, with coefficients in `~/tmp/test_no_unsloth/hash_head_body_prior_fit.json`.

### Hash-prior corpus sensitivity

The four-coefficient hash prior is corpus-sensitive. A controlled comparison kept the existing Chinese baseline at exactly its original 16 shuffled B1 samples, used a streaming WikiText-2 English sample stopped after exactly 16 blocks, and generated exactly 16 blocks of uniform random model token IDs. No full Chinese or English dataset iteration was used. All three corpora were projected through the same three frozen GGUF `tid2eid` tables; the 48 Chinese projections reproduced the recorded route rows exactly.

The English sample read 588 streaming records to obtain 32,768 tokens, used 99.85% predominantly-ASCII text, and used the same local DeepSeek tokenizer. The random control sampled uniformly from the complete valid model vocabulary `[0,129280)` with seed `20260812`. The refitted coefficients are:

| Corpus | `mu_rho` | `kappa_rho` | `a0` | `beta` |
| --- | ---: | ---: | ---: | ---: |
| Chinese baseline | `0.0765686` | `111.401` | `6.66003` | `0.477175` |
| English WikiText-2 | `0.0477295` | `289.340` | `2.68470` | `0.0833506` |
| Uniform random IDs | `0.0001831` | `2002.111` | `47.4263` | `0.650482` |

The English result is not just a smaller Chinese head. Its most frequent token is `the`, but space, punctuation, and other function-word tokens are also frequent, so the single persistent six-expert head in the compact formula absorbs several real heads into its body term. A 4,000-replicate conditional block bootstrap found that the English changes in `mu_rho`, `a0`, and `beta` excluded zero; the beta-concentration difference was too noisy to distinguish with only 16 blocks. The random control has no meaningful lexical head; its tiny fitted `mu_rho` is a finite-sample tie-break rather than a semantic feature.

All three corpora activated all 256 experts. Exact projected medians were:

| Corpus | `B` | `max(M_g)/T` | Effective experts | Maximum-padding inflation |
| --- | ---: | ---: | ---: | ---: |
| Chinese | 1 | `0.11084` | `181.53` | `4.73x` |
| Chinese | 4 | `0.10638` | `195.04` | `4.54x` |
| Chinese | 16 | `0.09845` | `200.87` | `4.20x` |
| English | 1 | `0.09692` | `178.07` | `4.14x` |
| English | 4 | `0.09094` | `183.57` | `3.88x` |
| English | 16 | `0.08813` | `185.92` | `3.76x` |
| Random | 1 | `0.03369` | `250.31` | `1.44x` |
| Random | 4 | `0.02887` | `254.14` | `1.23x` |
| Random | 16 | `0.02682` | `255.07` | `1.14x` |

Applying the Chinese prior unchanged to the English B16 shape predicted median `max(M_g)/T=0.10381` and `200.42` effective experts versus exact `0.08813` and `185.92`. Applied to random IDs it predicted `0.10373` and `200.52` versus exact `0.02682` and `255.07`. The refitted English and random priors recovered their respective B16 medians within about `2.5%` for both metrics, but remain exchangeable shape approximations rather than exact token-hash behavior.

The shape difference can affect grouped-MMQ comparator ranking. Exact medoids were replayed on gfx1151 through the retained DeepSeek `IQ2_XXS` gate/up pair and `Q2_K` down paths. The ratio is packed grouped-MMQ throughput divided by the BF16 AITER reference throughput, so values above `1` favor the packed path:

| DeepSeek B16 target | Chinese | English | Random |
| --- | ---: | ---: | ---: |
| `IQ2_XXS` pair, reversed-order 25-repeat confirmation | `0.942x` | `1.031x` | `0.854x` |
| `Q2_K` down, 9-repeat replay | `0.698x` | `0.747x` | `0.537x` |

The first row is a genuine ownership crossover: English makes the packed path faster while Chinese and random favor the AITER comparator. A hash-only B16 AITER configuration screen independently selected `BLOCK_SIZE_M=128`/`GROUP_SIZE=8` for Chinese and random, but `BLOCK_SIZE_M=64`/`GROUP_SIZE=8` for English on the down shape. This is comparator and ranking evidence only. Reweighting the existing five learned medoids at `40/43` and replacing only the hash component at `3/43` left the existing full-model down winner unchanged for all three corpora; the random-control margin was only about `1 ms` and is not promotion evidence.

Therefore exact token/table projection remains authoritative when token IDs are available. The Chinese four-coefficient fit remains a capture-free search prior for the current evidence set, not a language-universal production distribution. English and uniform-random profiles are mandatory sensitivity controls for future candidate ranking, while full captured histograms and multi-seed, multi-checkpoint natural-language captures remain required for any production frequency weighting or dispatch change. The comparison implementation, token archive, exact medoids, and GPU reports are recorded under `~/tmp/test_no_unsloth/hash_router_corpus_*`, `~/tmp/test_no_unsloth/benchmark_hash_*`, and `~/tmp/torch-ggml-ops/hash-router-corpus-replay/`.

The selected hierarchy is therefore exact token-hash projection when token IDs are known, this four-coefficient head-plus-body prior for capture-free candidate search, and the captured token/ranked profiles for ranking-sensitivity checks and final promotion. The formula must not be treated as proof of production frequency weighting until the planned multi-seed and multi-checkpoint corpus exists.

Prior-predictive medians for the fitted non-hash priors, reported as `(active experts, max(M_g)/T, effective experts, maximum-padding inflation)`, are:

| Component and `T` | Observed | Prior |
| --- | --- | --- |
| Qwen, B1/S2048 | `(200, 0.811, 26.4, 18.6x)` | `(202, 0.799, 27.6, 18.1x)` |
| Qwen, B4/S2048 | `(241, 0.758, 29.0, 21.8x)` | `(243, 0.752, 30.1, 22.0x)` |
| Qwen, B16/S2048 | `(254, 0.748, 29.5, 23.7x)` | `(254, 0.719, 32.0, 22.7x)` |
| DeepSeek learned, B1/S2048 | `(230, 0.537, 34.4, 19.9x)` | `(232, 0.503, 37.3, 18.9x)` |
| DeepSeek learned, B4/S2048 | `(249, 0.486, 38.2, 19.8x)` | `(249, 0.471, 40.9, 19.3x)` |
| DeepSeek learned, B16/S2048 | `(254, 0.479, 39.6, 20.0x)` | `(254, 0.446, 44.0, 18.8x)` |

The Qwen equal-total-token partition check used B1/S2048, B2/S1024, and B4/S512 in both base and checkpoint `7400`; the prior remained compatible with the observed active-count, head-load, and effective-expert distributions. DeepSeek sequence-length extrapolation remains untested because the audited attention path requires `S=2048`. The pooled prior is a benchmark model, not evidence that learned routing is training-step invariant: Qwen base versus checkpoint active-support KS was `p=0.99` and alpha-shape KS was `p=0.097`; DeepSeek learned alpha-shape KS was `p=0.097`, but active support differed at B16 (`p=0.014`). Therefore production frequency weighting still requires early, middle, and late checkpoints across multiple data seeds. Until those captures exist, use the pooled prior for candidate ranking and retain checkpoint-stratified histograms for final promotion.

### Grouped MMQ AITER comparator boundary

The grouped-MMQ learned-router prior is now connected to an offline AITER comparator screen, not to GGTensile generation or runtime dispatch. The screen covered all 24 Qwen/DeepSeek batch-shape targets with a bounded candidate domain: current exact config, one-field neighbors, cross-batch retained configs, and archived uniform-route candidates. Five weighted medoids per learned family and five per DeepSeek hash family compressed 512 coefficient-only samples while preserving physical expert identities. Captured learned medoids, exact DeepSeek token/hash projections, and `uniform`/`skewed`/`sparse`/`boundary` routes remained independent controls.

The screen initially produced 12 pooled candidates above 2% with component checks. After captured and synthetic 25-repeat controls, six exact GMM keys were promoted in `bench/aiter_gmm_heuristics.py`: DeepSeek B4 forward pair; Qwen B1 forward down and backward pair; and Qwen B16 forward down, backward down, and backward pair. The Qwen B16 backward-pair entry was selected by a bounded compromise screen after the first prior winner failed synthetic controls. All six passed boundary-heavy GPU correctness, old/new bitwise identity, mutation sensitivity, and reversed-order 25-repeat confirmation on prior, captured, and synthetic corpora. Six other initial targets retained current configs because their finalists regressed mandatory controls or had no robust compromise.

The exact-key table remains a comparator authority only. The installed AITER `aiter.ops.triton._triton_kernels.gmm.get_config` accepts shape parameters but returns an architecture-level JSON default; its source still marks shape lookup as a TODO, and the installed package has no `gfx1151-GMM.json`. Explicit benchmark configs therefore come from the repository table, whose layout-sensitive fail-closed behavior is tested by `tests/test_aiter_gmm_heuristics.py`.

Retuned learned/hash replays held archived HIP medians fixed when attributing ratio changes. The Qwen forward weighted ratios moved `1.529/1.257/0.900 -> 1.516/1.257/0.881x`, Qwen backward moved `1.143/1.302/1.153 -> 1.029/1.302/1.066x`, and DeepSeek routed forward moved `1.894/1.284/1.223 -> 1.894/1.081/1.223x` for B1/B4/B16. These are comparator-replay screening results with correctness rows 64, not final claims about HIP kernel ranking.

A subsequent existing-HIP ownership screen compared current dispatch with the already packaged row-task bodies over five prior, five captured, and four synthetic Qwen B1 profiles. Exact paired IQ2_S forward and single-down IQ2_S backward at aggregate rows 16,384 passed nine-repeat screening, reversed-order 25-repeat confirmation, every-profile `>0.99x` controls, and bitwise identity. Q3_K forward missed the captured `>1.02x` gate; Q4_K and Q5_K backward regressed synthetic controls. Production therefore adds only the two exact IQ2_S dispatch decisions, with no new kernel or host inspection of `expert_offsets`. Holding promoted AITER and unaffected archived HIP medians fixed, the B1 ratios move `1.5162 -> 1.5627x` forward and `1.0291 -> 1.0822x` backward; B4/B16 are unchanged. The focused existing-path review is complete. A new tail-aware device/hybrid mechanism remains deferred because current host dispatch cannot observe expert maxima without crossing the device-routing boundary.

## Design Principles

### Exact contracts, no repair

One complete problem and solution must predict one physical assembly stream. Unknown fields, missing fields, implicit numeric coercions, incompatible linked values, and unsupported policies reject before lowering. The generator never rewrites a request into a nearby valid solution.

Every accepted serialized field must affect canonical identity and lowering, or be explicitly fixed by the problem contract. Mutation tests enforce that accepted fields are projected or rejected rather than silently ignored.

### Deterministic, self-contained generation

Production generation does not search, benchmark, invoke HIP or LLVM code generation, consult measured winners, run an optimizing allocator or heuristic scheduler, repair register pressure, or fall back to source or insertion order. Formula-derived and deterministic lifetime-aware register assignment remains part of pure physical planning. Search, profiling, compiler experiments, and GPU timing are offline evidence only.

The same complete input must produce byte-identical source. Assembly, linking, and inspection then verify deterministic object and code-object translation.

### Capability, candidate, and selection are separate

Capability validation describes the formula-supported domain. Candidate manifests describe complete parameter points. Exact-key inventories and selected-solution catalogs describe measured production choices.

Winner maps never belong in capability validation. Candidate identity is parameter-only; exact-pair identity additionally includes the problem type and exact shape; artifact identity additionally includes generated source.

### Generate semantics, not stored schedules

The writer describes logical work, ownership, dependencies, and traversal. It must not become a repository of copied physical instruction streams. Shortening by moving a schedule into another module, template, JSON file, tuple, opcode table, issue-slot map, rank list, or compatibility writer is not simplification.

Line count is not the objective. A valid simplification deletes duplicated derivation or schedule-shaped implementation by replacing it with a typed invariant, formula, or genuinely shared semantic mechanism.

### Preserve direction-specific algorithms

Forward and backward share only mechanisms whose semantics and emitted text are genuinely direction-neutral. They retain separate solution contracts and lowering algorithms:
- forward consumes a previously produced Q8_1 activation workspace, expands integer weights, uses integer WMMA, and applies joint weight/activation correction.
- backward reads BF16 activations, dequantizes weights to BF16, and converges on a shared BF16-WMMA pipeline.
- packed-weight addressing, LDS ownership, output ownership, launch mapping, and epilogue traversal remain direction-specific where required.

GGTensile forward does not quantize activations. The Q8_1 producer is an upstream kernel with its own public workspace contract.

### Derive redundant state once

Geometry, ownership, grids, packed strides, activation strides, loop counts, accumulator counts, register roles, lifetimes, LDS offsets, waits, and resource usage derive from one authoritative contract/specification boundary. Exact shape names and selected resources are not derivation inputs.

Formula-derived capability may accept divisible shapes beyond the selected inventory. Catalog coverage and GPU evidence remain exact-key facts.

### Typed boundaries carry invariants

Components consume typed logical operands, register roles, memory roles, and dependency records rather than opaque `vN`, `sN`, half-register, address-expression, or instruction strings. Small helpers are retained only when they enforce a typed invariant, own a nontrivial formula, or form a reused abstraction boundary.

Contractual invalid states use explicit rejection reasons. Natural programming errors propagate. Python `try` blocks are reserved for releasing acquired resources or restoring process-global state before re-raising.

### Enum and boolean argument rules

Finite serialized domains are represented by enums, not open strings. Enum member names and serialized values must match exactly, including capitalization and underscores. The name used in Python, the value emitted in JSON, and the identity used by candidate hashing are one spelling; aliases, case folding, whitespace normalization, and compatibility spellings are not permitted. Existing JSON field names remain unchanged unless an explicit identity migration is approved.

Enum conversion is strict and occurs once at the canonical problem/specification boundary. Unknown values reject with a structured reason before physical planning or emission. Conversion must not infer a nearby policy, repair an invalid combination, or silently map multiple accepted values to one lowering. A serialized enum value is candidate identity, while its typed policy exposes only the semantic facts consumed by validation, physical planning, inspection, and lowering.

Serialized boolean fields remain strict JSON booleans and retain their existing field names for compatibility. At the derived-state boundary, a boolean that participates in a mode is converted to a named enum, capability set, or policy record. Boolean-heavy vectors and parallel flags are not semantic APIs: when several flags jointly select a mechanism, replace them with one typed policy that represents the valid combinations and rejects the rest. For example, packed-prefetch, Q5 metadata/shift, store priority, and LDS buffering are policy values even where their legacy solution representation contains booleans or integer switches.

A boolean is appropriate for a genuinely independent binary fact or a local predicate derived from a typed policy. A helper that genuinely needs an independent boolean receives it as a keyword-only argument; positional boolean arguments are prohibited because their meaning is not visible at the call site. Lowerers and physical planners do not repeatedly branch on raw serialized booleans. Public and serialized compatibility fields may remain boolean, but internal consumers use the derived typed authority.

### Completion requires recursive review

Every campaign and structural refactor ends with a fresh recursive review of the design contract, implementation, generated artifacts, selected and rejected evidence, target ISA, and related kernel work. The review is global across forward and backward directions, problem types, quant types, and exact shapes; a local completion statement is scoped to the contract and inventory it actually qualified, not a waiver for related work. Findings are classified as duplicate or closed, contract-incompatible, unsupported, deferred with an explicit prerequisite, or actionable.

An actionable finding must be implemented and qualified before the review is repeated. Completion is valid only when a fresh global pass finds no actionable in-contract mechanism. A result from one direction, quant type, or shape may transfer as evidence to another, but never as an automatic selection: the receiving semantics, ownership, lifetimes, synchronization, arithmetic order, resources, and exact-key gates must be re-derived and tested.

## Model and Identity

### Public records

`tools.ggtensile` exposes immutable, JSON-serializable records:
- `ProblemType`: operation, quant and activation types, destination and compute types, and transpose/layout semantics.
- `ProblemSize`: exact GEMM coordinates and operation-specific dimensions.
- `ForwardSolution` and `BackwardSolution`: complete direction-specific kernel choices.
- `SolutionKey`: the exact problem-type/problem-size/solution tuple with canonical JSON and a stable content hash.
- `RejectReason`: stable rule ID, diagnostic, involved parameters, and source.
- `KernelArtifact`: symbol, source/object/code-object paths, source identity, ABI, launch geometry, and inspected resources.

Named defaults are explicit and remain part of canonical identity. Parsing is strict and generation never mutates a solution.

### Public writer boundary

`ForwardKernelWriterAssembly` and `BackwardKernelWriterAssembly` are the only public assembly-writer entry points. Each preserves its established constructor, diagnostics, `source()`, and `write()` behavior. A facade may validate a key, construct canonical derived state, initialize ROCISA, emit the fixed code-object envelope and signature, dispatch to one validated lowerer, and translate errors at the public boundary. It must not own quant-format register constants, LDS offsets, decoder loops, WMMA loops, waits, barriers, epilogues, resource arithmetic, or a generic fallback body.

### Deployment catalogs

`tools/ggtensile/configs/` contains one `mmq_<direction>_<quant>_catalog.json` deployment file per supported direction and quant type. Each file has only three root fields: the canonical `ProblemType`, a deduplicated list of complete `Solutions`, and `ExactLogic` entries that map an exact `ProblemSize` to a solution index. Every listed solution must be referenced by at least one exact key. An absent key has no GGTensile deployment decision and falls back outside this catalog.

Deployment catalogs do not contain model names, tensor labels, family names, call counts, benchmark medians, experiment status, historical controls, candidate names, or rejected alternatives. Experiment chronology and current research status belong in the corresponding Markdown record. Immutable benchmark reports may contain their own workload and timing context, but they are evidence rather than deployment logic.

### Forward contract layers

The forward implementation separates fixed semantics, complete choices, and derived facts:

```text
ForwardProblemContract
  quant format and packed block
  activation layout and block
  arithmetic and signedness
  destination and BF16 rounding
  ISA, wavefront, ABI

ForwardKernelSpec
  geometry and ownership
  global memory and LDS representation
  decode and iteration policy
  dot and epilogue policy
  instruction policy
  resource limits

DerivedForwardState
  macro tile and workgroup ownership
  exact grid and K-block count
  packed and activation strides
  accumulator and loop state
  resource usage

QuantForwardSemantics
  packed payload planes
  scale/minimum or signed-scale fields
  high-bit reconstruction
  post-WMMA correction semantics

ForwardMechanismContract
  lowering and concrete physical-plan kind
  workitem-ID use and wave-M/wave-N ownership
  legacy serialized compatibility facts
  data-contract compatibility
  activation and weight block domains
  formula-derived reduction granularity

ForwardPhysicalPlan
  concrete mechanism layout and register roles
  deterministic assignments and lifetimes
  sole VGPR, SGPR, LDS, private, and spill usage
```

`ForwardResourceUsage` is the shared authority for writer metadata, capability admission, and artifact inspection. Resource limits are candidate constraints; VGPR, SGPR, LDS, private-segment, and spill outcomes are derived facts.

For gfx1151, admission and inspection account for the 64 KiB workgroup LDS ceiling and wave32's 24-VGPR allocation granularity. A logical high-water register index is not substituted for the allocated resource count reported in kernel metadata.

### Backward contract layers

Backward follows the same separation without sharing the forward algorithm:

```text
BackwardProblemContract
  quant format, packed block, BF16 inputs and output
  ISA, wavefront, ABI, and arithmetic contract

BackwardKernelSpec
  geometry, memory, decode, pipeline, and store policy
  strict Q3 pairing and linked mechanism capability

DerivedBackwardState
  exact grid, reduction trips, packed strides, and derived resources

BackwardMechanismContract
  decoder, LDS, schedule, lane-sharing, and prefetch capability

BackwardPhysicalPlan
  register, decoder, address, LDS, workgroup, and resource plans
  sole physical authority consumed by lowering and inspection
```

Quant-specific reader/decoder leaves produce the common BF16 weight representation. The common backward lowerer owns the BF16-WMMA pipeline and stores. Q4_K and Q5_K share packed metadata addressing and scale/minimum preparation while Q5_K high-bit reconstruction remains a distinct leaf.

### Canonical candidates

Complete candidates round-trip through normal serialized solution inputs. Canonical hashes exclude problem shape so a candidate may be tested on another formula-compatible shape; exact-pair manifests preserve the shape-specific evidence. Family-inactive legacy fields reject before hashing.

## Lowering Architecture

### Shared assembly infrastructure

`kernel_writer_assembly.py` owns the direction-neutral assembly module, ROCISA setup, source writing, pointer and address primitives, BF16 RNE emission, metadata/trailer emission, and deterministic register-pool mechanism. Direction writers own their algorithms and do not forward through compatibility writers.

### Responsibility map

| Module or module family | Sole responsibility |
| --- | --- |
| `kernel_writer_assembly_mmq_fwd.py` | Forward public facade, fixed source envelope, signature, and closed dispatch |
| `mmq_fwd_spec.py` | Forward problem/mechanism contracts, complete kernel specification, quant semantics, and typed policy state |
| `mmq_fwd_physical.py` | Pure closed forward physical-plan union, register/LDS roles, deterministic assignments, and resources |
| `mmq_fwd_lowering*.py` | Substantive mechanism orchestration and final instruction emission; no public writer or revalidation |
| `kernel_writer_assembly_mmq_bwd.py` | Backward public facade, fixed source envelope, signature, and closed dispatch |
| `mmq_bwd_spec.py` | Backward contracts, complete specification, derived state, and mechanism capabilities |
| `mmq_bwd_physical.py` | One pure backward register/decoder/address/LDS/resource authority |
| `mmq_bwd_lowering_quant.py` | Quant-specific packed readers and BF16 decoder leaves |
| `mmq_bwd_lowering.py` | Common backward BF16-WMMA pipeline, synchronization, and stores |
| `mmq_bwd_emission.py` | Typed bounded pending-zero VOPD formation at known eligible sites |
| `grouped_mmq_fwd_model.py` / `grouped_mmq_fwd_spec.py` | Grouped routed problem identity, typed mechanism policies, geometry, row dispatch, epilogue, and derived state |
| `grouped_mmq_fwd_physical.py` | Grouped activation/weight LDS layouts, register roles, and resource formulas |
| `grouped_mmq_fwd_route.py` | Complete grouped ABI route prologue, guards, and expert rebasing through one public emitter |
| `grouped_mmq_fwd_lowering*.py` | Explicit grouped mechanism orchestration, route composition, decode, correction, and output emission |
| `grouped_mmq_fwd_lowering_row_dispatch.py` | Shared typed activation/MMA/epilogue row-body threshold and label emission |
| `mmq_fwd_lowering_decoded_stage.py` | Typed common decoded-weight LDS staging and compatible WMMA/store emission |
| `inspection.py` | Artifact verification against plan-derived launch and resource facts; no allocation replay |

Specification and physical-planning modules are pure: they do not import ROCISA, invoke the toolchain, benchmark, or emit instructions. Lowerers consume canonical typed state and concrete plans; they do not read inventories, selected catalogs, tensor names, benchmark reports, or runtime winner maps.

### Completed writer-refactor boundaries

The completed refactor makes the following boundaries normative:
- Public ordinary and grouped facades validate one exact key, construct one typed derived state, initialize the fixed code-object envelope, dispatch to an explicit validated lowerer, and concatenate only generic sections plus typed mechanism-owned trailing sections. They do not own decoder loops, register allocation, LDS offsets, waits, WMMA bodies, epilogues, or resource formulas.
- Canonical specifications interpret serialized fields once. Grouped compatibility field-copy records are not used. `GroupedForwardProblemContract`, `GroupedForwardKernelSpec`, `GroupedRouteState`, typed activation/row/decode/epilogue policies, and the physical plan are the authorities shared by grouped validation, lowering, and inspection.
- `GroupedRouteEmitter` owns the complete routed ABI prologue: 64-byte kernarg loads, exact launch guards, int64 expert-ID loads, cumulative int32 offset loads, invalid-route inertness, and full-u64 expert-bank rebasing. It preserves the at-most-256 route-entry contract and final valid offset `R`. Mechanism lowerers compose its single public `emit()` operation and do not call sibling private methods.
- `GroupedActivationStagingPlan` owns exact versus ceil-sized bounds-masked staging, vector load coverage, and LDS write shape. `GroupedRowTileDispatchPolicy` and `GroupedRowTileDispatchEmitter` own the one-, two-, and three-body activation/MMA/epilogue topology derived from macro/tail geometry. `GroupedOutputStore` is validated against that topology rather than used as an implementation switch.
- `DecodedWeightLdsStageEmitter` is the typed common boundary for compatible Q4_K/Q5_K packed-weight staging, decoded LDS writes, scaled WMMA, and BF16 tile stores. Ordinary and grouped orchestration retain separate contexts and scalar plans; cross-context inheritance and casts are not semantic reuse. Q2_K remains a dedicated lowerer for F16_D2S6 staging, missing-sum groups, persistent all-ones operands, distributed production, and phased correction.
- IQ2_S codebook rodata is owned by the IQ2_S lowering result. The generic grouped facade knows only how to concatenate ordered result sections, preserving section order and source identity without embedding IQ2_S mechanism knowledge.
- Backward derives typed schedule, LDS-buffering, packed-prefetch, extraction, Q3 pairing, Q5 metadata/shift, store-priority, packed-row-address, decoder-capability, address-capability, and quant-register-shape policies. `BackwardSolution` JSON fields and their serialized identity remain unchanged. Physical planning and lowering consume the policies while `_emit_q3_k_*`, `_emit_q4_k_*`, `_emit_q5_k_*`, `_emit_q6_k_*`, and `_emit_q8_0_*` remain explicit arithmetic leaves.
- The public backward `registers` compatibility attribute remains because existing callers use it; unused facade state and confirmed one-hop aliases were removed only where public behavior and type narrowing were unaffected. `DeterministicRegisterPool` remains because it is production-used by physical planning.

These boundaries generalize capabilities, not algorithms. They do not introduce a generic scheduler, instruction IR, copied instruction-order table, callable dispatch registry, compatibility writer, or forward/backward lowering inheritance.

### Forward lowering

`ForwardKernelWriterAssembly` is a small public facade. It validates one key, constructs `DerivedForwardState`, emits the code-object envelope and ABI, and performs closed dispatch. It owns no decode, LDS, WMMA, wait, or epilogue body.

Pure planning in `mmq_fwd_physical.py` returns one concrete physical-plan type. Mechanism lowerers are organized by mechanism or data contract:
- packed-three-bit half-tile LDS lowering for the serialized Q3 control.
- direct packed scale/minimum lowering.
- decoded-weight LDS plus `F16_D4S4` activation lowering shared by compatible low-nibble and high-bit formats.
- structured signed-six-bit lowering with single-row or dual-row ownership.
- signed-int8 direct, register-tiled, wave-N LDS, exact small-M, and compact-depth32 physical mechanisms.

`F16D4S4ActivationMetadata` owns the shared activation group formula. Packed scale/minimum reconstruction is shared only by compatible correction paths, and one source-identical signed-int8 WMMA constructor is shared without merging ownership or post-WMMA arithmetic. Serialized operand-source names remain stable compatibility values even where internal classes use mechanism names.

All forward families consume Q8_1 bytes and metadata produced upstream. Weight decode, activation staging, integer dot, correction, waits, and output conversion remain separate semantic responsibilities when ownership or dependency order differs.

The concrete forward mechanism boundaries are:

| Mechanism family | Physical ownership retained by the mechanism |
| --- | --- |
| Packed three-bit half/full-weight LDS | Q3 payload planes, signed group scales, half/full decode ownership, 336-byte full rows, and four-barrier traversal |
| Direct packed scale/minimum | One-wave global reads, nibble decode, scale/minimum correction, and scalar output traversal |
| Decoded-weight LDS | Q4/Q5 decoded rows, F16_D4S4 activation staging, metadata schedule, clamp, and epilogue; Q5 high bits remain format-specific |
| Structured Q6 | Single/dual-row ownership, optional exact MT256 wave groups, near/far reads, signed decode, refill traversal, and typed delays |
| Signed-int8 direct/register/tiled | Direct/register ownership or ordinary, compact-depth32, and exact small-M LDS layouts with their own stage and scale-read policy |

There is intentionally no universal physical pipeline. Q3, Q6, and Q8 may share an F32_D4 activation contract while retaining different packed layouts, reduction depths, lane ownership, LDS rows, correction arithmetic, synchronization, and register lifetimes.

### Backward lowering

Backward retains one geometry-derived allocation, reduction pipeline, WMMA lowering, and store path. Quant-specific global-read and decoder leaves produce the common BF16 weight representation. This convergence point is why five backward formats can share a smaller common body without forcing the forward integer-WMMA algorithm into the same design.

### Shared component criteria

A semantic component is shared only when all of these are true:
- at least two mechanisms have the same typed input and output roles.
- producer, first-use, barrier, lifetime, and arithmetic-order requirements are equal.
- the abstraction owns a reusable formula or invariant and removes duplicated derivation.
- selected streams remain identical, or a deliberate stream change passes the full qualification gates.
- the component does not consult exact shapes, selected winners, timing evidence, or schedule tables.

Similar instruction spelling is not sufficient evidence. Q3 signed group-scale correction, Q4/Q5 scale/minimum correction, Q6 signed-scale/block-factor correction, and Q8 scale multiplication remain distinct correction leaves.

### Semantic stages and scheduling

There is no universal forward stage runner. Each mechanism owns orchestration and calls a shared component only when roles, dependencies, and arithmetic order are equal. Structured Q6 directly calls typed emitters corresponding to semantic phases such as:

```text
Setup
GlobalRead
Decode
LocalWrite
Barrier
LocalRead
Dot
ScaleAccumulate
LoopCommit
Epilogue
```

These names describe semantic ownership, not a stored stage graph or instruction schedule. Other mechanisms retain direct typed emitters where a stage abstraction would only repackage instruction order. Operations carry semantic coordinates such as row, payload plane, decode atom, K phase, output tile, LDS pair, and store batch. Dependencies are checked before emission.

The second level applies explicit deterministic policies for traversal, clustering, lookahead, local-write placement, local reads, dot grouping, epilogue scope, VOPD pairing, delays, clauses, and cache behavior. Policies are complete mechanism choices, not generic heuristic scheduling. Waits derive from typed producers and first-use boundaries; redundant weaker waits are suppressed monotonically.

### Register allocation and physical roles

Register roles declare width, alignment, lifetime, reuse class, ownership, and deterministic role order. `DeterministicRegisterPlan` uses lifetime-aware first fit with no optimization, repair, or insertion-order fallback. `DeterministicRegisterPool` supports explicit checkout/checkin reuse at known last-use boundaries.

Irregular selected assignments may remain pinned when they are measured ownership facts. Regular accumulators, outputs, pointers, decode values, LDS pairs, and scratch values derive from formulas or pool allocation. Lowering receives assignments; it does not discover pressure while emitting.

### Q6 semantic boundary

Q6 uses typed payload/address roles, 16 semantic decode atoms, explicit low/high extraction, signed-byte normalization, formula-derived LDS roles, producer-first-use VMEM waits, typed dependency delays, deterministic source-to-output register reuse, and shared dot/refill/epilogue orchestration.

The residual single-row/dual-row setup, near/far read, activation-read, and refill leaves preserve selected physical traversal. The completed Q3/Q8 convergence review found no equivalent ownership contract, so these remain bounded Q6 policies rather than duplicate full kernels.

### Prohibited production mechanisms

Neither direction may use:
- raw assembly templates or large assembly-string collections.
- sibling forwarding writers or compatibility aliases.
- external instruction data, opcode tables, rank tables, or absolute issue maps.
- learned rankers, opaque cost models, repair logic, or hidden ready-list priorities.
- source-order or insertion-order fallbacks.
- build-time HIP/LLVM scheduling or allocation.
- post-emission textual filtering or VOPD reconstruction.
- accepted fields that affect neither lowering nor validation.

Backward pending-zero pairing is a typed bounded operation invoked only at known eligible coordinates. Ordinary instruction formatting never parses or rewrites emitted text.

## Tuning and Search

Only values with implemented distinct lowerings may enter a candidate domain. Fixed arithmetic, ABI, ISA, activation layout, and format facts are contracts rather than genes.

The main linked tuning groups are:
- geometry and ownership.
- global-memory widths, assignment, clustering, prefetch, and cache policy.
- LDS representation, buffering, layout, swizzle, and plane placement.
- decode traversal, lane sharing, grouping, lookahead, and write deferral.
- iteration, local reads, dot grouping, and activation/weight overlap.
- epilogue traversal, dependency width, scope, priority, and stores.
- explicit VOPD, delay, clause, and cache policies.
- resource limits and desired occupancy.

Useful complete policy dimensions include workgroup wave count, macro tiles, row-tail ownership, WMMA grouping, activation layout and vector width, exact versus ceil/masked staging, direct versus decoded weight representation, LDS row/padding and payload width, metadata ownership and conversion schedule, decode overlap, epilogue dependency/store policy, grouped route ownership, backward schedule and buffering, lane sharing, and quant-specific extraction. Q6 may additionally expose traversal, stage clustering, latency, role-lifetime, producer-first-use, dependency-compatible pairing, physical carry, and epilogue-width policies through its existing typed schedule model.

A policy dimension is valid only when it changes a semantic or physical choice and has a complete lowering, strict validation, physical-plan formula, inspection expectation, and external search representation. Expert-ID width, cumulative-offset width, route-entry bounds, final valid offset, ABI, arithmetic, ISA, and fixed format facts are contracts, not tuning knobs. The writer never chooses a value from a benchmark winner, model name, shape label, or hidden default, and a new knob must not merely expose an instruction-order tuple.

Repository-owned search tooling lives outside the writer and operates on complete candidates:

```text
candidate_domains(quant_type, shape)
candidate_neighbors(seed, knob_group)
explain_invalid(candidate, shape)
canonical_candidate(candidate)
```

Search explores linked neighborhoods rather than a broad Cartesian product. Generation never benchmarks or selects. LLVM, HIP, TensileLite, EvoTensile, CK, and hipBLASLt may provide mechanism vocabulary or offline evidence, but cannot supply production instruction order, allocation, validity, or winner selection.

Exact-shape constants, fixed trip counts, peeled tails, affine-address reductions, register-lifetime shortening, and legal VOPD formation are derived lowering work rather than public knobs unless complete alternate mechanisms are implemented. Their value is judged by the same correctness, resource, and timing gates as larger policies.

### Candidate-domain evidence boundary

Automated domains are bounded deterministic enumerators of complete implemented policies. They are useful for identity, validation, and reproducible candidate construction, but they do not prove optimization exhaustion and need not reproduce every selected exact-key composition. A missing domain value is actionable performance work only when it identifies an implementable in-contract mechanism with an exact target, plausible gain path, and qualification gate. Otherwise it is tooling scope.

### Cross-campaign transfer policy

The same review rule applies when a finding crosses direction, format, or shape boundaries. Shared vocabulary such as affine address hoisting, dependency-derived waits, load clustering, or explicit register lifetimes may be reused only after the receiving lowering proves the corresponding semantics, physical roles, synchronization, arithmetic order, and resource envelope. A result may transfer as evidence, never as an automatic selection. Failed experiments remain recorded with their failure reasons even when a later changed premise justifies a bounded retest.

Current exact performance results, rejected alternatives, and cross-campaign review conclusions live in the ten MMQ experiment records. They are intentionally not duplicated in this design document.

### Diagnostic lower bounds and profiling

When a bottleneck is ambiguous, exact diagnostic kernels may isolate matrix/activation/LDS work from packed decode/LDS work while preserving launch geometry and declared resources. They pass the same symbol, ABI, resource, and forbidden-storage inspection as candidates. Hardware counters and normalized disassembly explain first-order limits but never select winners; unsupported or over-capacity counter requests remain recorded evidence rather than silently reduced measurements.

## Artifact Lifecycle

Generation, build, inspection, correctness, screening, and confirmation are separate immutable phases:
- `generate` validates one exact key, emits source or structured rejection, and records the source hash.
- `build` verifies an approved generation manifest and invokes the assembler and linker.
- `inspect` verifies symbol, ABI, ISA, metadata, resources, and forbidden storage or instructions.
- `correctness` runs only inspected artifacts against the operation control and independent references.
- `screen` ranks qualified candidates under warmed rotating controls.
- `confirmation` retimes finalists under a fresh, longer rotating protocol.

Each phase refuses to overwrite an existing artifact. Manifests are strict. Source is the primary generated content identity; object and code-object identities are deterministic translations verified by rebuild and inspection.

## Verification and Promotion

### Structural migrations

When behavior is intended to remain unchanged, compare generated source, executable text, metadata, symbols, resources, and code objects exactly. Preserve ABI, launch ownership, waits, VOPD pairings, LDS offsets, barriers, and resource envelopes.

Independent roots must reproduce byte-identical source, object, and code object. Executable identity discharges a new timing run for a purely structural move; a change that cannot preserve source and executable identity must be split from the migration and qualified as a deliberate stream change.

### Canonical identity migrations

Adding, removing, or canonicalizing a serialized field may change solution hashes and symbols while preserving the body. Such a migration must:
- record every old/new key and symbol mapping.
- compare source after symbol normalization and compare executable text, metadata, ABI, and resources exactly.
- update catalogs and manifests deliberately with no hidden compatibility alias.
- run direction-appropriate correctness, independent-reference, mutation, determinism, and forbidden-storage gates.

An identity migration may not conceal an instruction-stream change. If normalized source or executable behavior differs, the deliberate-stream rules apply.

### Deliberate stream changes

A deliberate instruction-stream change requires:
- finite output and exact control agreement when arithmetic order is unchanged.
- an independent dequantized or independently formulated reference.
- input, packed-weight, and activation-workspace mutation sensitivity.
- deterministic source and code-object rebuilds.
- strict ABI, ISA, VGPR, SGPR, LDS, spill, and private-storage inspection.
- representative and blind formula-compatible shapes where shared lowering changes.
- retained-parent comparisons and warmed GPU confirmation.

Static inspection never substitutes for execution correctness.

### Measurement and retention

GPU timing is serial, warmed, and rotating, with the same tensors and launch conditions. Reports preserve raw samples, medians, dispersion, paired comparisons, protocol identity, and normalized assembly around finalists.

The standing gates are:
- zero correctness failures.
- byte-identical independent generation and rebuild.
- zero private storage, spills, scratch, calls, or dynamic stack.
- repeated warmed evidence that every selected exact GGTensile multiply is faster than its exact HIP control.
- retained-parent comparisons for every deliberate stream change, with slower candidates rejected regardless of static resource or instruction reductions.
- representative checks on every exact key sharing an unconditional changed stream.

There is no fixed percentage threshold. Timing selects winners. Static issue counts, counters, code size, locality, and resources explain results but do not promote candidates by themselves. Weighted workload totals guide effort and reporting; they never authorize a slower exact key.

## Structural Design Invariants

### Ownership and dependency rules

- Every validated mechanism maps to exactly one substantial lowerer and one concrete physical plan; unknown mechanisms reject before emission.
- Contract/specification state, physical planning, mechanism emission, exact selection, and public dispatch have separate owners.
- Register assignment, LDS layout, decoder rows, address state, waits, launch metadata, and resources derive once and are consumed by validation, lowering, and inspection.
- Capability validation contains formulas and typed capability predicates, never production-shape lists, model/tensor names, or selected winners.
- Accepted fields project into canonical state and lowering or reject as inactive. No two accepted compatibility values may produce the same canonical lowering.
- Lowerers never parse or rewrite emitted instructions, choose instruction policy from exact problem size, or import selection and benchmark state.
- All current and future physical/lowering modules participate automatically in forbidden-dependency, structural-pattern, and executable-line coverage.

### Abstraction rule

Delete a helper when it only forwards one call, returns one nested attribute, copies a typed record field-for-field, or wraps one instruction at one call site. Retain it when it enforces a reusable invariant, owns a nontrivial formula, tracks dependencies or lifetimes, is a stable public operation, or removes substantive contract-equivalent duplication.

Do not add base lowerer classes, one-method sibling writers, callable registries, generic visitors, or an instruction IR merely to reduce visible repetition. A thin facade plus explicit mechanism lowerers is preferable to a generic pipeline that stores or reconstructs selected schedules.

### Required structural guards

Tests must enforce:
- no inventory, catalog, benchmark, or runtime imports from lowerers and physical planners.
- no ROCISA or toolchain dependency from pure specification and planning modules.
- no duplicated resource allocation in inspection.
- no raw assembly templates, opcode/issue tables, rank lists, source-order fallbacks, exact-shape instruction switches, or post-emission rewriting.
- no field-copy adapters at shared component boundaries and no unused or pass-through abstractions.
- complete writer-line coverage for every mechanism and preservation of both directions after shared assembly changes.

Repository gates include focused writer/spec/validation/search tests, complete GGTensile and repository suites, Ruff, formatting, `ty check`, compileall, pre-commit, bundle currency, deterministic artifact checks, and `git diff --check`.

## Non-Goals

GGTensile does not implicitly provide:
- deployment selection from broader formula capability.
- public runtime or bundle promotion from research qualification alone.
- retuning as part of a structural migration.
- one universal physical tile, LDS layout, stage graph, register plan, decoder, or epilogue.
- a generic scheduler, ready list, learned ranker, opaque cost model, repair path, hidden allocation policy, or online tuning.
- prepared weights, dense shadows, external decode workspaces, split-K, Stream-K, persistent/grouped traversal, producer fusion, hidden caches, or multi-kernel fixup without explicit public contracts.
- cosmetic file splitting that preserves duplicate derivation or facade-local physical knowledge.

## Completion Criteria

The writer architecture is complete when:
- both public writers are thin facades with no mechanism body, register allocator, decoder, pipeline implementation, or textual scheduling.
- each accepted mechanism has one concrete physical plan and one substantial explicit lowerer.
- shared mechanism facts have one typed authority across facade, validation, planning, inspection, and search.
- physical roles cross component boundaries as typed assignments or protocols; raw register numbers remain local to final emission where unavoidable.
- capability is formula-based and deployment applicability remains exact-catalog-owned.
- selected and frozen streams are byte-identical across structural changes, or every deliberate exception passes full correctness, artifact, determinism, and timing qualification.
- all selected forward and backward keys, currently 56 and 50 respectively, retain qualified behavior and public fallback remains intact.
- a fresh global recursive review finds no actionable in-contract structural mechanism.

## Current Implementation Status

### Coverage status

| Area | Current status |
| --- | --- |
| Shared generator, toolchain, inspection, runtime, and campaign infrastructure | Implemented for the current gfx1151 exact-key workflow |
| MMQ backward | Required `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0` campaigns complete; 50 exact inventory keys are selected |
| MMQ forward Q4_K | Twelve exact research identities qualified; the typed activation-base lifetime is accepted, `OneByEight` is rejected, and public wiring is deferred |
| MMQ forward Q5_K | Six exact inventory keys selected; the compact/high-bit body and shared Q4/Q5 activation-base lifetime are qualified, with public wiring deferred |
| MMQ forward Q6_K | Three exact language-model-head keys select the typed row/role wavefront; shared MT256 and `WideScalarCarryFrontier` are independently rejected, and public wiring is deferred |
| MMQ forward Q3_K | Dense 12-key inventory complete; all 12 exact keys select the typed full-weight research candidate, while public wiring remains deferred to the 179-kernel HIP bundle |
| MMQ forward Q8_0 | All 23 exact keys have final research decisions: 20 compact-depth32 selections, one ordinary HIP-shaped control, and two LM-head small-M controls; public wiring remains deferred |
| Grouped GGTensile forward | Isolated non-paired routed research kernels for Q2_K, Q4_K, Q5_K, and IQ2_S plus the paired Qwen IQ2_S row-task kernel are implemented and qualified across exact aggregate-row workloads; grouped ownership, search, inspection, deterministic rebuild, and resource gates are complete, while public wiring remains deferred |
| Grouped MMQ AITER comparator | Prior-aware bounded screen, captured/synthetic controls, six exact table replacements, and comparator-only replay complete |
| Grouped MMQ learned-B1 HIP ownership | Existing-path screen complete; exact IQ2_S forward-pair and backward-down row-task dispatches retained, while Q3_K/Q4_K/Q5_K controls are rejected |
| Public GGTensile runtime selection | Deferred; existing HIP bundle dispatch remains authoritative |

The ten direct MMQ forward and backward format campaigns are currently exhausted under the fixed exact-key contract: the separate post-Q6 physical-plan review found no actionable in-contract premise. The isolated grouped Q2_K, Q4_K, Q5_K, and IQ2_S non-paired scope and paired Qwen IQ2_S scope are qualified under their routed or row-task ABIs and exact workload contracts. Public dispatch, model-owned representations, and other explicitly separate integration scopes remain deferred. A genuinely changed premise still reopens the affected campaign under its exact correctness, resource, reproducibility, and timing gates.

### Implemented forward architecture

The completed forward-writer refactor established:
- a thin `ForwardKernelWriterAssembly` facade with validation, source envelope, output writing, and closed mechanism dispatch.
- strict `ForwardProblemContract`, complete `ForwardKernelSpec`, `ForwardMechanismContract`, `DerivedForwardState`, quant semantics, and formula-derived resource usage.
- a closed union of concrete physical plans as the sole register, LDS, and resource authority.
- formula-based divisible-shape capability separated from exact inventories and winners.
- canonical candidate and exact-pair manifests with linked external search neighborhoods.
- packed-three-bit, packed scale/minimum direct, decoded-weight LDS, structured-Q6, and signed-int8 lowerer modules.
- shared `F16D4S4ActivationMetadata`, packed scale/minimum reconstruction, and a source-identical signed-int8 WMMA component at proven semantic boundaries.
- read-only signed-int8 tiled-LDS register and scale-layout protocols at the shared stage, group, and store helper boundary; concrete physical plans cross that boundary without field-copy adapters.
- a shared decoded-LDS low-nibble/high-bit pipeline with a distinct high-bit reconstruction leaf.
- one structured Q6 orchestration with typed setup/read/decode/LDS/dot/refill/epilogue boundaries.
- 16 semantic Q6 decode atoms with deterministic lifetime-aware register reuse.
- typed address, payload, LDS-pair, accumulator, product, output, wait, and dependency-delay roles.
- deterministic allocation and formula-derived VGPR, SGPR, and LDS admission.
- removal of facade-local bodies, raw templates, forwarding writers, post-emission scheduling, inactive fields, and duplicate resource/decode authorities.
- structural tests for forbidden dependencies, ignored fields, raw operands, legacy reads, pass-through helpers, and complete lowerer/physical-plan line coverage.

The retained residual Q6 setup/read/refill traversal is intentionally mechanism-owned; the completed Q3/Q8 convergence review found no contract-equivalent replacement.

### Implemented backward architecture

The completed bidirectional writer refactor also established:
- a thin `BackwardKernelWriterAssembly` facade over `DerivedBackwardState`, `BackwardKernelSpec`, and one pure `BackwardPhysicalPlan`.
- one common BF16-WMMA/pipeline lowerer and one substantial packed-weight reader/decoder component rather than five forwarding writer classes.
- typed register, decoder, address, LDS, and resource plans as the sole lowering and inspection authority for the 50 selected backward keys.
- shared Q4/Q5 packed metadata addressing and scale/minimum preparation, with Q5 high-bit reconstruction retained as a distinct leaf.
- strict serialized Q3 `Inactive`/`Partial`/`Full` pairing policy with no exact-size instruction branch.
- formula-derived quant-block, tile, decoder-row, workgroup-mapping, WMMA, deep-pipeline geometry, LDS, and physical-capacity admission.
- typed quant mechanism capabilities for lane sharing, padded DepthU64, decoded-B pipelines, SIA3, and next-packed prefetch.
- exact typed policies for schedule interpretation, LDS buffering, packed-prefetch combinations, Q3 pairing, Q5 metadata/shift, extraction, store priority, packed row addressing, decoder capability, address capability, and quant-specific register shape; the legacy `BackwardSolution` JSON schema is unchanged.
- linked bounded decoder, complete-pipeline, and LDS-layout search neighborhoods filtered through normal validation.
- typed pending-zero VOPD formation at known coordinate sites; ordinary instruction formatting neither parses nor rewrites emitted text.
- complete structural and executable-line coverage for every current and future `mmq_bwd_*.py` physical/lowering module discovered by convention.

### Earlier catalog qualification snapshot

The earlier catalog qualification snapshot records:
- 352 GGTensile tests and 437 repository tests passing, with only the 14 existing Python 3.14 PyTorch deprecation warnings.
- all 56 selected forward and 50 selected backward artifacts regenerated from two independent roots and matching the reviewed post-migration baseline in source, object, code object, normalized disassembly, symbols, ABI, metadata, resources, waits, barriers, clauses, and VOPD counts.
- the canonical Q8 compact-M64 identity migration from `ggsol_13768a4d22953b59` to `ggsol_11f54a8999dc20e6`, with symbol-normalized source and executable identity.
- explicit Q3 pairing migration evidence for 50 changed backward identities and 56 unchanged forward identities, with normalized source, disassembly, and inspection identity.
- historical 516/447 source archives retained as pre-refreshed evidence; the 106 exact selected artifacts are the current identity authority.
- Ruff, formatting, `ty check`, `compileall`, pre-commit, bundle currency, `git diff --check`, structural guards, and complete writer-line coverage passing.
- the forward-complete gfx1151 MMQ baseline at 179 kernels, unchanged by this refactor; the current package has 181 kernels after two separately qualified grouped-backward additions.
- Q6 MT64 resources of 158 VGPRs, 27 SGPRs, and 28,928 bytes of LDS.
- Q6 MT128/M256 resources of 210 VGPRs, 27 SGPRs, and 38,400 bytes of LDS.
- zero private storage and zero VGPR/SGPR spills for the selected Q6 artifacts.
- exact HIP/public agreement, finite output, independent-reference qualification, mutation sensitivity, deterministic rebuilds, blind formula-compatible coverage, and warmed confirmation for the selected forward paths.
- six blind backward controls spanning both Q3 pairing policies and Q4_K/Q5_K/Q6_K/Q8_0, each bit-exact to installed HIP and independent dequantized BF16 matmul, deterministic, mutation-sensitive, code-object-v5, and spill/private-storage free.
- a resource-clean shared-MT256 Q6 control rejected at `1.0332x` selected-parent latency and `1.0289x` HIP latency.
- a resource-clean M64 `WideScalarCarryFrontier` Q6 control rejected at `1.00054x` selected-parent latency; its failed screen correctly prevented M128 derivation and 25-repeat confirmation.

### Post-Q6 physical-plan review

After those two Q6 experiments were rejected, a separate read-only pass classified every apparent opening across all ten experiment records, the strict catalogs, current forward/backward lowering and physical plans, bounded search domains, bundle state, and target exclusions. Remaining items are retained and measured, evidence-rejected, contract-incompatible or deferred, or stale chronology/tooling-only. No newly actionable in-contract mechanism has an exact target and credible qualification path. The strict 56-forward/50-backward catalog state remains unchanged. Its 179-kernel public-bundle baseline is preserved byte-for-byte inside the current 181-kernel package, which adds only two separately qualified grouped-backward kernels.

Exact source identities and normalized executable/code-object checks remain in artifact tests and experiment evidence rather than this generic design document.

## Writer Refactor Qualification

The completed source-preserving refactor was qualified against a baseline of 238 exact generated streams: 56 ordinary forward keys, 50 ordinary backward keys, and 132 grouped solution/production-row records. All 238 remained byte-identical, including comments, labels, whitespace, waits, section ordering, blank lines, and IQ2_S local rodata ordering.

The complete `tests/ggtensile` suite passes 422 tests covering model identity, strict validation, ordinary, grouped, and paired source generation, assembly and linking, artifact inspection, deterministic rebuilds, resource limits, route and invalid-route behavior, mutation sensitivity, typed policy branches, and executable-line coverage. The full repository suite passes 523 tests with the 14 existing Python 3.14 deprecation warnings. Ruff, formatting, compileall, `ty check`, pre-commit, and `git diff --check` also pass. Inspected retained artifacts remain code-object v5, gfx1151, wave32, ABI/resource compliant, and free of private storage, spills, scratch instructions, calls, and dynamic stack.

The recursive review covered both directions, every ordinary quantization type, grouped Q2_K/Q4_K/Q5_K/IQ2_S, exact production shapes, generated source, inspected artifacts, schedule modules, physical-plan modules, and public-boundary behavior. The subsequent paired review implemented and qualified the Qwen IQ2_S K128-interleaved row-task candidate. Public dispatch, generated bundle tables, extension registration, packaging, and HIP fallback remain unchanged; public grouped selection is deferred until a separate integration campaign qualifies those surfaces.

The source-preserving policy is the default for existing specifications. Any intentional source, executable, ABI, or rodata difference must be isolated as a deliberate stream change and pass independent correctness, mutation, deterministic-build, artifact/resource, and warmed timing gates. This qualification record is evidence of the current implementation, not a license to infer future coverage or promote research kernels through public dispatch.

## Experiment Records

Experiment records own exact problem scopes, timing tables, rejected mechanisms, debugging history, resource details, evidence paths, and campaign closure.

Completed MMQ backward records:
- `experiment_ggtensile_mmq_bwd_q3_k.md`
- `experiment_ggtensile_mmq_bwd_q4_k.md`
- `experiment_ggtensile_mmq_bwd_q5_k.md`
- `experiment_ggtensile_mmq_bwd_q6_k.md`
- `experiment_ggtensile_mmq_bwd_q8_0.md`

MMQ forward records:
- `experiment_ggtensile_mmq_fwd_q4_k.md`
- `experiment_ggtensile_mmq_fwd_q5_k.md`
- `experiment_ggtensile_mmq_fwd_q6_k.md`
- `experiment_ggtensile_mmq_fwd_q8_0.md`
- `experiment_ggtensile_mmq_fwd_q3_k.md`

Grouped MMQ forward records:
- `experiment_ggtensile_grouped_mmq_fwd_q2_k.md`
- `experiment_ggtensile_grouped_mmq_fwd_q4_k.md`
- `experiment_ggtensile_grouped_mmq_fwd_q5_k.md`
- `experiment_ggtensile_grouped_mmq_fwd_iq2_s.md`
- `experiment_ggtensile_grouped_mmq_fwd_pair_iq2_s.md`

These records retain campaign chronology and may describe historical premises that were later superseded. This document is authoritative for the current generic architecture and coverage status; selected catalogs and artifact tests are authoritative for current exact identities.

## Integration and Expansion

Each generated artifact owns one exact problem symbol. Runtime selection may use an artifact only when every problem type, exact size, ABI, architecture, solution, and source-identity assertion matches. Every mismatch falls back to the existing implementation.

Public GGTensile integration remains deferred until a useful production set is selected, packaging and identity are stable, exact dispatch engineering is complete, and end-to-end workloads pass correctness and weighted performance validation. Experimental force controls are not public policy.

Grouped GGTensile remains a separate deployment and integration scope. The isolated non-paired routed research writers for Q2_K, Q4_K, Q5_K, and IQ2_S and the paired Qwen IQ2_S row-task writer are implemented and qualified under the inventory above. Retained down targets stay non-paired; Q3_K and IQ2_XXS gate/up precursors still require paired successors. Public paired, single-routed, row-task, and fixed-group ownership still require explicit dispatch, packaging, memory, synchronization, resource, and fallback contracts. Ordinary coverage of an overlapping format does not count as grouped coverage.

Prepared weights, compact alternate public layouts, BF16 shadows, additional paired formats, persistent workgroups, split reduction, GSU, Stream-K, and multi-kernel fixup require model-visible ownership, lifetime, invalidation, memory accounting, ABI, and fallback design. They are not hidden extensions of the current exact single-kernel backend.

Bounded scan scripts may construct, cache, validate, and rank complete exact candidates outside the direction writers. A larger automated search system remains optional; deterministic `SolutionKey` identity, explainable rejection, immutable artifact phases, and reproducible evidence remain mandatory.
