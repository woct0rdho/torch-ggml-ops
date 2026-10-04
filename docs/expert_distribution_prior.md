# Expert Distribution Prior and Benchmark Policy

## Status and authority

This document is the authority for expert-route workload fitting and performance-benchmark input selection. It supplements `ggtensile_plan.md`: the plan defines generator architecture and exact kernel identity; this document defines the route distribution used to rank optimization candidates.

Historical experiment records preserve the distributions and timing procedures that were actually used at the time. They are evidence, not current instructions. In particular, references to five medoids, captured-route replay, corpus mixtures, or synthetic timing controls are superseded for a new optimization campaign by the single-prior route-bank policy below. Historical result documents are not rewritten when this protocol changes.

## Normative policy

For every workload family or explicitly declared problem type, select exactly one fitted expert prior law. The law is the optimization target. Do not rank a candidate against a different distribution, a second fitted law, a medoid bank, or a weighted mixture of fitted laws.

The fitting phase and the benchmark phase are separate:
- An offline fitting job may read captured route data and, where needed, token IDs and frozen router tables.
- The job emits one fit artifact containing the law, coefficients, fit corpus identity, fit seed, and validation statistics.
- Benchmark tooling consumes only that fitted law and the exact problem metadata. It does not read route captures, language corpora, checkpoint reports, or route-analysis JSON.
- For each exact problem key, the law is evaluated at its physical token/row size and produces a deterministic bank of complete expert-route vectors. The bank is generated before timing from the benchmark seed mixed with the exact problem key, and it holds exactly one vector per timing sample. Each vector is materialized in the production caller's contract: int64 expert ids for every physical expert, int32 group sizes over every expert with zero rows for inactive experts, and int32 cumulative offsets over every group. The active support is recorded as report provenance. Candidate and retained-parent timing use the same vector at each paired sample.

If the fitting inputs or method change, repeat qualification for the resulting fit. Do not mix profiles from different fits or silently add another benchmark distribution.

The workload family is defined by routing semantics and physical contract, not by quantization alone. A law may be shared by compatible Qwen shapes, for example, while a distinct learned-router or hash-router contract receives its own law. Learned and hash routing must not be combined into a `40/43` plus `3/43` timing mixture; either they are declared separate problem families with one law each, or one explicitly fitted law is selected for the combined problem type before timing begins.

Paired gate/up projections reuse each route vector for their shared routed rows. Their packed weights, outputs, arithmetic, and timing records remain separate where the exact operator contracts differ.

`uniform`, `skewed`, `sparse`, `repeated`, `boundary`, alternate language corpora, and exact captured profiles may still be used for non-timed correctness, malformed-route, tail, mutation, and ABI coverage. They are not optimization targets, ranking inputs, weights, fallback gates, or performance controls. A correctness matrix must not be reported as evidence for which candidate is faster.

No medoid reduction is part of the current policy. If the fitted law is stochastic, choose and serialize one deterministic seed and generate the route bank directly. Do not generate a search bank, reduce it with k-medoids, retain medoid weights, replay captured routes, or mix alternate route controls into the timed result.

## Distributional route-bank benchmark protocol

The route bank is the distributional measurement unit for grouped direct-kernel and public-API benchmarks. The benchmark allocates route tensors, activation workspaces, and any paired row-task workspaces before the timing loop. For paired IQ2_S direct kernels, the deployed J64 row-task descriptors are built once for every route vector before timing; activation quantization and row-task setup are outside the multiply-only timing surface. Both implementations receive that same production route representation: the direct kernels consume the ids and cumulative offsets, and the routed AITER baseline consumes the full-expert group sizes together with the dense BF16 expert weights.

The timing loop takes exactly one timing sample for each selected route vector, and the bank contains exactly that many vectors; the same bank and the same route tensors are reused by every block. A route is selected once, then the identical route tensors are passed to HIP and GGTensile. `launches_per_sample` may average multiple launches inside that one timed event; it does not select another route or create another sample. The sample loop runs `--repeats` samples per block over `--blocks` blocks (default `10` and `2`), and the starting implementation order flips between blocks, so both implementations occupy the first and second alternation position equally often and `--repeats` must be even. Each sample is timed inside a pre-submitted event pair: one unmeasured launch keeps the device busy while the measured window is enqueued, so host launch cost is not charged to the kernel. Warmup runs at least `--warmup` iterations per implementation and at least `--warmup-seconds` (default `0.5`) of device time before sampling. Numeric activation and weight values remain fixed across the bank; expert IDs may change packed-weight addresses, cache behavior, and work partitioning, which is intentional route variability.

The timed route count is the fixed `--repeats` sample count per block. The bank is generated once at that size before timing, so no route tensor or allocation is created inside the timed loop. Dense and fixed non-routed direct-kernel and public-API measurements use the same estimator with one ordinary timing sample per iteration rather than one route vector. Adaptive route-count stopping is not part of the current harness.

Each implementation records its per-route samples together with the median, mean, minimum, maximum, and population standard deviation, plus the measurement-stability fields: host launch cost per call and its device/host ratio with a host-bound flag, per-position sample counts, medians and gaps, and the per-block medians and their spread. The primary speedup is the ratio of median times, reported as `throughput_ratio.value`; the report also carries `throughput_ratio.paired`, the median of the paired per-sample ratios, which is immune to any residual alternation-position asymmetry. Per-route observations, the sample order, the complete protocol block, and the stability block remain in the report. This reports a paired distributional comparison without replacing the route bank with a single aggregate row count.

## Fit contract

The only route information needed after fitting is the fitted law. The law may depend on physical size `T = physical_batch * sequence_length` and on the declared routing family. Model names, layer labels, checkpoint labels, corpus names, and benchmark weights are fit provenance; they are not runtime dispatch inputs or candidate identity fields.

The fit may pool layers, depth, and base/checkpoint states when that is part of the declared model-level law. Those dimensions must not become hidden benchmark selectors. A fit report must state the pooling decision, the captured corpus, the route semantics, the physical-size range, and the reason the resulting law is valid for the target family.

The minimum fit artifact records:
- family/problem-type identifier and fit provenance;
- captured corpus and tokenizer/router-table identity, if applicable;
- physical sizes and top-k contract;
- fitted coefficients and residual laws;
- deterministic benchmark seed and profile-generation algorithm;
- goodness-of-fit and residual diagnostics;
- known limitations and the exact re-fit trigger.

After the artifact is accepted, a benchmark must be reproducible without access to the captured data or the source language corpus. The benchmark may load packed GGUF weights, but it must not import `~/test_no_unsloth`, read `~/tmp` route captures, inspect live model routes, or perform host-side route collection.

## Captured evidence migrated from the prior plan

The previous committed `ggtensile_plan.md` established why a route law is needed. Static model inspection, aggregate active-expert counts, and one captured batch do not describe grouped work. The pilot captured 16 shuffled B1 microbatches at sequence length 2,048 with seed `19260817`.

| Model/state | Router population | Captured B1 observation | B16 aggregate observation |
| --- | --- | --- | --- |
| Qwen checkpoint `7400` | 40 learned routers, top-8 | Maximum `M_g` `516-2047`, median `1675`; active experts `73-256`, median `192`; layer-max padding inflation median `18.22x`, maximum `29.15x` | Sum of 16 independent B1 histograms: active experts `226-256`, median `253`; maximum `M_g` `10399-31956`, median `24928`; padding inflation median `24.00x` |
| DeepSeek zero-B initial state | 40 learned top-6 routers plus 3 hash routers | Learned active experts `178-253`, median `235`; learned maximum `M_g` `477-1982`, median `1074`; learned padding inflation median `20.08x`, maximum `37.64x`; hash routers use all 256 experts with maximum `M_g` `168-291` | Sum of 16 independent B1 histograms: learned active experts `248-256`, median `255`; learned maximum `M_g` `10162-26070`, median `14969.5`; learned padding inflation median `19.34x` |
| Qwen3.8 released checkpoint (`qwen4exp`) | 48 learned top-10 routers, 512 experts, no hash router | Active experts `134-453`, median `355`; maximum `M_g` `606-2047`, median `1396`; layer-max padding inflation `11.79x-34.88x`, median `22.59x` | Sum of 16 independent B1 histograms: active experts `446-510`, median `496`; maximum `M_g` `9835-26741`, median `18207`; padding inflation `15.28x-38.58x`, median `27.61x` |

These observations are fit evidence, not dispatch constants. Qwen base-to-checkpoint replay changed `67.6%` of token-layer top-8 expert sets, `12.4%` of route memberships, and `53.0%` of ordered slots across four paired samples. The DeepSeek one-step audit changed route histograms in `40/43` layers. Repeated checkpoint replay without an update was histogram-identical, so the observed changes were model-state effects rather than collector nondeterminism.

The capture schema used one record per `(model, checkpoint, physical batch, router kind, layer, sample, projection, quant-shape key)`, including the complete `rows_per_expert[256]`, selected-token count, top-k, sample identity, and training-state metadata. This schema remains fit provenance. It is not a benchmark input schema.

## One fitted learned-router law

For the Qwen learned top-8, DeepSeek learned top-6 and Qwen3.8 learned top-10 families, the pooled law samples active support and ranked skew from physical size:

```text
x = log(T / 2048)
y_A = a0 + a_log_tokens*x + epsilon_A
A = round((E + 1)*sigmoid(y_A) - 0.5), clipped to [top_k, E]
log(alpha) = b0 + b_log_tokens*x + b_active_residual*epsilon_A + epsilon_alpha
q_r = min(1, C*(r + shift)^(-alpha)), 1 <= r <= A
q_r = 0, r > A
sum(q_r) = top_k
```

`E` is the model's physical expert count and is the only place the family's expert count enters: `E = 256` for the Qwen top-8 and DeepSeek top-6 families, `E = 512` for the Qwen3.8 top-10 family. For the two migrated families the generalized line is identically the `257*sigmoid(y_A) - 0.5` form they were fitted and validated with.

Each residual is a clipped location-scale Student-t variate. The fitted residual laws were:

| Family | Residual | `nu` | `loc` | `scale` | Clip `[lo,hi]` |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen learned | `epsilon_A` | `8.3978226526` | `-0.0335118652` | `0.9830493404` | `[-2.5036492445,3.7809143778]` |
| Qwen learned | `epsilon_alpha` | `5.2038953632` | `0.0054814884` | `0.1025706842` | `[-0.4463245132,0.3833201702]` |
| DeepSeek learned | `epsilon_A` | `33.5987235964` | `-0.0013194870` | `0.8328268568` | `[-1.6829685257,2.7587218851]` |
| DeepSeek learned | `epsilon_alpha` | `5.1551932778` | `-0.0071110386` | `0.0923527684` | `[-0.3180690433,0.3604795507]` |
| Qwen3.8 learned | `epsilon_A` | `7.4647474580` | `0.0130546014` | `0.4915605168` | `[-1.4935692978,1.9212931553]` |
| Qwen3.8 learned | `epsilon_alpha` | `8469463127.5627` | `0.0002737558` | `0.0870558989` (calibrated) | `[-0.2081819082,0.2166008918]` (calibrated) |

After solving for `q`, apply the fitted head multiplier:

```text
q1_adjusted = min(1, h*q1)
q_r_adjusted = q_r*(top_k-q1_adjusted)/(top_k-q1), r = 2..A
```

Constrained largest-remainder rounding of `T*q_r_adjusted` produces `M_g`; one deterministic permutation assigns ranks to physical expert IDs. The learned coefficients are:

| Family | `shift` | `h` | `a0` | `a_log_tokens` | `b0` | `b_log_tokens` | `b_active_residual` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen learned | `11.5465232873` | `1.1255020052` | `1.3568429238` | `1.1367804236` | `0.7376182182` | `-0.0215993943` | `-0.1730074363` |
| DeepSeek learned | `6.3223820835` | `1.0643729189` | `2.2468973539` | `0.8906164814` | `0.4081577973` | `-0.0211298377` | `-0.0754167931` |
| Qwen3.8 learned | `20.7163112483` | `1.1946652586` | `0.7076361999` | `0.9702739281` | `0.7505135414` | `-0.0229817472` | `-0.2302367069` |

The active-residual term is part of the single law because support and skew are coupled. The pooled captured correlations were `-0.778` for Qwen, `-0.616` for DeepSeek learned, and `-0.772` for Qwen3.8 learned routes. Removing the term reduced log-alpha regression `R^2` from `0.702` to `0.011` for Qwen, from `0.257` to `0.031` for DeepSeek, and from `0.451` to `0.017` for Qwen3.8. A compact ablation is not a second benchmark law; it is an invalid substitute unless the fit is explicitly changed and requalified.

### Qwen3.8 dispersion calibration

The Qwen3.8 `epsilon_alpha` law carries one deliberate deviation from "residual exactly as fitted": its scale and clip are multiplied by `0.6` (so the residual keeps its `nu` and location and its shape under scaling). The reason is that a per-profile `alpha` is obtained by minimizing the rank-curve TV, which trades the head against the body, so its residual spread over-states the layer-to-layer spread that the route metrics actually see. The uncalibrated law reproduces the mean route metrics but predicts `p05-p95` widths `1.2-1.75x` the observed ones for `max(M_g)/T`, effective experts and padding inflation at every captured size. With the calibration those widths match: effective-expert `p95` ratio against the captures `1.34 -> 1.17` at B4 and `1.44 -> 1.25` at B16, static-grid slots `1.14 -> 1.03` and `1.21 -> 1.09`, with the data-carrying tile count unchanged, and the median prior-predictive KS improves at B4 (`0.120 -> 0.076`) and B16 (`0.182 -> 0.135`) at a cost at B1 (`0.113 -> 0.135`). Calibrating the support residual instead, alone or together with the skew residual, makes every shape worse, so only the skew residual is calibrated. The two migrated families keep their residuals exactly as fitted.

## Qwen3.8 (Qwen4-Exp) learned-router fit

The third family is the Qwen3.8-Flash-Next (`qwen4exp`) MoE: 48 learned `Qwen4ExpTextTopKRouter` layers, top-10 over 512 experts, and no hash router, so one learned law covers the whole model. The fit uses 16 shuffled B1/S2048 microbatches of the Chinese-dialogue training corpus at seed `19260817` (the same corpus, seed and protocol as the two migrated families), plus B2/S1024 and B4/S512 partitions of the same token counts. The procedure is the pooled `v2` fit with the expert count generalized to 512; the coefficients are in the tables above, and the skew residual is calibrated as described there.

Three checks the migrated families did not receive were run on this fit.

### Prior-predictive and held-out quality

Observed against predicted medians (calibrated law, 10,000 draws per size):

| Size | active | `max(M_g)/T` | maximum `M_g` | padding inflation | median KS |
| --- | --- | --- | --- | --- | --- |
| B1/S2048 | `355 -> 345` | `0.682 -> 0.627` | `1396 -> 1284` | `22.6x -> 20.7x` | `0.135` |
| B4/S2048 | `454 -> 455` | `0.588 -> 0.583` | `4816 -> 4772` | `26.1x -> 26.2x` | `0.076` |
| B16/S2048 | `496 -> 496` | `0.556 -> 0.555` | `18207 -> 18191` | `27.6x -> 27.5x` | `0.135` |

Rank-curve TV (median over profiles) is `0.060 / 0.059 / 0.056` at B1/B4/B16, against `0.065 / 0.063 / 0.063` for Qwen and `0.072 / 0.071 / 0.069` for DeepSeek learned.

Held out by fitting on 8 of the 16 blocks and predicting the other 8, under the identical protocol for all three learned families:

| Family | held-out median KS, B1 / B4 | held-out errors (active, max share, effective experts, inflation) |
| --- | --- | --- |
| Qwen3.8 top-10 | `0.151 / 0.147` | `6.9% / 2.3% / 12.7% / 7.3%` at B1; `2.0% / 2.0% / 13.8% / 1.1%` at B4 |
| Qwen top-8 | `0.168 / 0.165` | `4.3% / 1.8% / 4.5% / 5.3%` at B1; `1.2% / 1.7% / 2.3% / 0.2%` at B4 |
| DeepSeek learned top-6 | `0.154 / 0.176` | `0.9% / 8.3% / 9.5% / 9.1%` at B1; `0.0% / 7.6% / 11.1% / 8.4%` at B4 |

The Qwen3.8 law generalizes at least as well as the two migrated learned laws (best median KS of the three, best maximum-share error), and its weakest metric is the effective-expert count, which is what the dispersion calibration above addresses. Fitting and predicting on the same half gives `0.107`/`0.117`, so the gap to `0.151`/`0.157` is the intrinsic dispersion misfit rather than parameter noise.

### Kernel-relevant sensitivity of the bank

Grouped cost is set by tile issue counts and group heights, not by the aggregate row count, so the fit error was translated into those units. Ratios of the bank to the captures: median, with the p95 in parentheses.

| Size | active | `max(M_g)` | tiles(64) | worst(64) | static slots = worst*active | tile-rounding padding |
| --- | --- | --- | --- | --- | --- | --- |
| B1/S2048 | `0.97 (1.03)` | `0.92 (1.10)` | `0.98 (1.02)` | `0.96 (1.10)` | `0.91 (0.99)` | `0.98 (1.02)` |
| B4/S2048 | `1.00 (1.02)` | `1.00 (1.19)` | `0.99 (1.01)` | `1.01 (1.19)` | `1.01 (1.14)` | `0.99 (1.01)` |
| B16/S2048 | `1.00 (1.00)` | `1.00 (1.17)` | `0.99 (1.00)` | `1.00 (1.16)` | `0.99 (1.21)` | `0.99 (1.01)` |

The multiply work is within `1-3%` at the median and within `2%` at the p95 at every captured size, while the concentration tail is over-stated: `worst(64)` and the static-grid slot count are `10-21%` high at the p95 before calibration and roughly half that after. A paired candidate/retained-parent comparison is unaffected because both see the same route tensors, absolute numbers are pessimistic by a few percent, and the decisions a heavier tail can influence are the ones named in the grouped records: small-route bodies, inactive-wave and inactive-M suppression, static-versus-row-task ownership, and split factors.

### Depth

From the same captures, 16 profiles per layer:

| Quantity | `epsilon_A` | `epsilon_alpha` | head share |
| --- | ---: | ---: | ---: |
| total scatter | `0.488` | `0.140` | - |
| per-layer-mean scatter | `0.395` | `0.125` | - |
| between-layer variance share (ICC) | `0.633` | `0.770` | `0.559` |

| Depth band | active | head share | effective experts | `epsilon_A` | `epsilon_alpha` |
| --- | ---: | ---: | ---: | ---: | ---: |
| early 0-15 | `376` | `0.622` | `57.6` | `+0.302` | `-0.039` |
| middle 16-31 | `364` | `0.646` | `51.1` | `+0.164` | `-0.014` |
| late 32-47 | `306` | `0.756` | `32.4` | `-0.368` | `+0.098` |

Routing becomes steadily more concentrated with depth, and early-versus-late KS is `0.53` for the skew residual and `0.63` for the support residual, so the layers are not exchangeable. The law nevertheless pools all layers, exactly as the two migrated families do, because the benchmark family is the *layer population* at a physical size: the exact keys carry no layer or depth input, every MoE layer repeats the same expert shapes, and the marginal over layers is therefore the target. The residual laws are what make that pooled law stochastic, and the depth extremes stay inside the bank rather than outside it: late layers sit at the bank's `10th` percentile of concentration and `79th` of head share, early layers at `71st` and `49th`. The law must not be read per-layer, and depth must not become a dispatch input: for late layers the pooled prediction under-states concentration by about `50%`.

### Corpus sensitivity

The same fit was run on three additional captures of 16 blocks each at B1/S2048: Chinese Wikipedia (`wikimedia/wikipedia 20231101.zh`), English WikiText-2 (`Salesforce/wikitext` streaming), and uniform random token IDs (seed `20260812`, the seed the hash-router study used).

| Corpus | active | effective experts | head share | padding inflation |
| --- | ---: | ---: | ---: | ---: |
| Chinese dialogue (declared fit corpus) | `355` | `48.3` | `0.682` | `22.6x` |
| Chinese Wikipedia | `373` | `67.6` | `0.577` | `20.1x` |
| English WikiText-2 | `405` | `78.1` | `0.541` | `21.3x` |
| Uniform random token IDs | `250` | `20.9` | `0.974` | `22.5x` |

Applying the declared Chinese-dialogue law to the other captures costs `28%` (Chinese Wikipedia) and `38%` (English) on the effective-expert median and `8%` / `15%` on the head share, while padding inflation stays within `2.6-8.4%`. The random arm costs `133%` on the effective-expert median and `36%` on the head share.

The direction for the two realistic corpora matches the hash-router study below: non-dialogue text is flatter than the Chinese baseline (`max(M_g)/T` `0.111` Chinese against `0.097` English there; effective experts `48.3` against `78.1` here), with a larger effect for the learned router. The random arm is the opposite of that study's and does not transfer: random token IDs saturate a learned router onto a small expert set (effective experts `21`, head share `0.97`), because the route is a function of the hidden state rather than of the token ID, whereas the hash law's random arm was nearly uniform. The tile issue count is again far less corpus-sensitive than the shape metrics: at B16 `tiles(64)` is `5402/5396/5386/5368` across the four corpora (`0.3%` spread) while static-grid slots range over `0.52x-1.25x`. The declared corpus remains the Chinese-dialogue training corpus.

### Implementation and consumption status

The law is implemented in `bench/workload_prior.py` (and in `~/test_no_unsloth/expert_distribution_prior.py`, which the AITER tooling imports) with a per-law expert count: `LearnedLaw.experts` is 512 for this law, the support transform is `A = round((E + 1)*sigmoid(y_A) - 0.5)` clipped to `[10, E]`, and `rows_per_expert` / `group_sizes` are 512 entries long. The two modules agree bitwise on the sampled rows and on `expert_prior_metadata()` for every law and seed. `tools/aiter_gmm_compat.py` resolves this family from the `Q2_0`, `IQ4_NL`, and `IQ4_XS` quantization types, and from the geometry alone when no type is given; a Qwen3.5-family quantization type never resolves a Qwen3.8 geometry. `tools/aiter_gmm_heuristics.py` carries the 54 measured `qwen3.8-learned` GMM/PTGMM keys over physical B1/B4/B16. The tuning campaign, its probe-screen protocol, and the confirmed gains are documented in `~/test_no_unsloth/docs/aiter_gmm_ptgmm_coefficient_prior_tuning.md`.

The law's alpha residual carries `nu = 8.47e9`; that is the t-fit diverging to "`nu = infinity`" on a near-Gaussian residual (the fit's KS statistic for the t and for a plain normal agree to `5e-6`, and the clipped tail mass differs from the normal by `1.5e-9`). The value is kept exactly as fitted because the tuned route banks are keyed to this sampler, and sampling it is numerically benign: a standard normal numerator over a chi-square denominator that concentrates on `nu` with relative spread `1/sqrt(nu) ~ 3e-6`, followed by the clip to `+/-0.21`, leaves the rank exponent within `[1.04, 2.52]` over the fitted size range.

## One fitted hash-router law

When token IDs and the frozen `tid2eid` table are available during fitting, the exact hash projection defines the law for the declared hash problem type:

```text
M_l,e = sum_v n_v * sum_(j=1..6) 1[H_l(v,j) = e]
```

Here `n_v` is the count of token ID `v` and `H_l(v,j)` is slot `j` of layer `l`'s table. A frequent token creates a correlated six-expert head; an independent rank law cannot represent that structure. After the fit is accepted, benchmark generation uses the declared law/profile and does not read token sequences, route captures, or live router tables.

For capture-free use, the prior plan fit one exchangeable head-plus-body law. It is one law, not a bank of alternatives:

```text
kappa_B = B*(kappa_rho + 1) - 1
rho ~ Beta(mu_rho*kappa_B, (1-mu_rho)*kappa_B)
a(T) = a0 * B^beta
u ~ Dirichlet(a(T) * 1_E)
H = a uniform k-subset of {0,..,E-1}
p_e = (1-rho)*u_e + (rho/k)*1[e in H]
M = bounded_largest_remainder(k*T*p, lower=1, upper=T)
```

The migrated coefficients are:

| Coefficient | Value |
| --- | ---: |
| `mu_rho` | `0.0765686035` |
| `kappa_rho` | `111.4009102046` |
| `a0` | `6.6600305083` |
| `beta` | `0.4771750368` |

The sampler emits all 256 experts, `sum(M_g)=6T`, and `M_g<=T`. The prior-predictive checks from the initial capture were:

| `T` | Metric | Observed | Formula |
| ---: | --- | ---: | ---: |
| `2048` | `max(M_g)/T` | `0.11084` | `0.10742` |
| `2048` | effective experts | `181.53` | `188.69` |
| `8192` | `max(M_g)/T` | `0.10638` | `0.10583` |
| `8192` | effective experts | `195.04` | `195.78` |
| `32768` | `max(M_g)/T` | `0.09845` | `0.10373` |
| `32768` | effective experts | `200.87` | `200.44` |

If the exact table projection is used, it replaces the capture-free surrogate for that hash problem type; the two must not be blended in one objective.

## Corpus and language sensitivity analysis

The previous plan compared three 16-block inputs projected through the same three frozen DeepSeek `tid2eid` tables: the original Chinese baseline, a streaming WikiText-2 English sample, and uniformly random model vocabulary IDs. This was fit sensitivity analysis, not authorization for three benchmark distributions.

| Corpus | `mu_rho` | `kappa_rho` | `a0` | `beta` |
| --- | ---: | ---: | ---: | ---: |
| Chinese baseline | `0.0765686` | `111.401` | `6.66003` | `0.477175` |
| English WikiText-2 | `0.0477295` | `289.340` | `2.68470` | `0.0833506` |
| Uniform random IDs | `0.0001831` | `2002.111` | `47.4263` | `0.650482` |

The English sample read 588 streaming records to obtain 32,768 tokens and was 99.85% predominantly ASCII. The random control sampled `[0,129280)` with seed `20260812`. The corpus comparison showed that a hash prior can be corpus-sensitive: the English changes in `mu_rho`, `a0`, and `beta` excluded zero in a 4,000-replicate conditional block bootstrap, while the beta-concentration difference was inconclusive at 16 blocks. The 16 captured S2048 samples had median `979` unique token IDs per sample (5th-95th percentile `837-1137`) and median maximum token multiplicity `173.5` (range `123.5-215.8`).

The exact projection's observed/projected shape metrics were:

| Hash metric | B1 observed / projected | B4 observed / projected | B16 observed / projected |
| --- | ---: | ---: | ---: |
| Active experts | `256 / 256` | `256 / 256` | `256 / 256` |
| `max(M_g)/T` | `0.111 / 0.115` | `0.106 / 0.109` | `0.098 / 0.105` |
| Effective experts | `181.5 / 180.7` | `195.0 / 193.8` | `200.9 / 198.9` |
| Maximum-padding inflation | `4.73x / 4.90x` | `4.54x / 4.64x` | `4.20x / 4.46x` |

A one-parameter symmetric Dirichlet-multinomial, a one-parameter logistic-normal multinomial, and a token-frequency Zipf fit were rejected as replacement laws: the Dirichlet fit predicted only `165.6` effective experts at B16 versus `200.9` observed, the logistic-normal fit predicted `max(M_g)/T=0.0833` versus `0.0984`, and the Zipf fit had mean/median token-histogram TV `0.101/0.096` while missing special-token spikes.

Exact projected medians were:

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

Applying the Chinese law unchanged to English B16 predicted `max(M_g)/T=0.10381` and `200.42` effective experts versus exact `0.08813` and `185.92`. Applied to random IDs it predicted `0.10373` and `200.52` versus exact `0.02682` and `255.07`. The comparison demonstrated why the fit corpus must be declared and recorded; it did not authorize English, random, or Chinese profiles as concurrent optimization targets.

The old plan also replayed these profiles through DeepSeek grouped paths. At B16, packed-throughput/AITER ratios were `0.942x/1.031x/0.854x` for the IQ2_XXS pair and `0.698x/0.747x/0.537x` for Q2_K down on Chinese/English/random inputs. Those crossovers are historical sensitivity evidence only. They must not be used to select a kernel under the single-prior policy.

## Fit limitations and required breadth

The DeepSeek one-step assignment audit changed route histograms in `40/43` layers. Its histogram-derived assignment lower bound was `1.4%` at the median and `2.1%` at the maximum, so the current corpus cannot support a claim that the pooled learned law is training-state invariant. The summed DeepSeek B16 learned captures also reached maximum-padding inflation of `33.95x`; the median alone understates the long-tail geometry that grouped kernels must handle.

The capture-free hash surrogate tracks the observed shape reasonably but is not exact token/table projection: the two independent shape metrics stayed under `4%` error at B1 and B4, while the B16 maximum-group metric was `5.4%` high. This is a fit diagnostic and a reason to record the exact hash projection when token IDs and `tid2eid` are available, not a reason to add another timed law.

Before claiming production-frequency weighting, extend the corpus with early, middle, and late checkpoints or training states, multiple data seeds, same-batch before/after-update captures where router trainability matters, and explicit layer and projection invocation frequencies. A pooled fit may remain the selected one-law model after that work, but the fit report must show why its pooling and weights represent the declared workload family. These breadth requirements trigger a new fit and requalification of the one-law route-bank benchmark procedure; they do not authorize concurrent laws, medoid banks, or profile mixtures.

The Qwen3.8 family adds three open items of its own. It was captured in one model state (the released checkpoint), so no checkpoint-invariance evidence exists for it. Its captures are at `S=2048`, and the routing is measurably flatter at the same token count with a shorter sequence (`B4/S512`: `+14%` active experts and `+25%` effective experts against the same-token `B1/S2048` prediction), so no other sequence length is in contract for this family; the fit corpus is the Chinese-dialogue training corpus, whose sensitivity is recorded above, and a change of training corpus triggers a refit rather than a second law. Depth is deliberately not an open item: it is pooled by design, its measured structure is recorded above, and depth must not become a dispatch input.

## Historical multi-medoid analysis

This section migrates the former medoid analysis so that old campaign reports remain intelligible. The procedure below was useful for studying route sensitivity, but it is superseded as a performance-benchmark input method. A medoid was a representative sampled route profile; it was never a second fitted law.

### Why multiple medoids were explored

A stochastic fitted law spans full physical row vectors, not just one scalar row count. At a fixed physical size `T`, draws can differ in active support, maximum `M_g`, concentration, long-tail group heights, and the physical expert IDs that own those rows. A single aggregate row count or a few marginal medians cannot preserve that geometry. The former campaigns therefore used several representative profiles to approximate the law's expected timing and to expose sensitivity to route shape:

```text
score(candidate) = sum_j cluster_mass_j * median_t latency(candidate, medoid_j)
```

This made the old objective sensitive to both the common route shapes and low-mass tall-expert tails. It also kept physical expert identities in the comparison instead of sorting away ownership information. The purpose was analysis of expected performance and robustness, not a claim that five profiles were the law itself.

### Former deterministic procedure

For each family, router component, physical batch, and bank, the old generator:
- Produced an independent 512-draw search bank and 512-draw confirmation bank from the fitted coefficient law. The search seeds were `8314159 + B * 104729 + i`; confirmation added `1,000,003`, and the DeepSeek hash component added `31,000`.
- Represented every draw as the complete physical `rows_per_expert[256]` vector. All rows in one bank had the same routed-row sum, so normalized-L1 distance compared physical load geometry rather than changing the total work.
- Reduced each bank to five physical-ID medoids with deterministic k-medoids. Initialization started at source index zero, added the farthest source at each step, selected the lowest source index for ties, and iterated medoid assignments until stable.
- Assigned every source draw to its nearest medoid. The medoid weight was its cluster mass, `number of assigned draws / 512`, rather than an equal five-way weight.
- Serialized the source seed, bank, active expert IDs, group sizes, full row vector, row sum, maximum group, cluster membership, and cluster mass so that the historical timing could be reproduced exactly.

The coefficient-only banks were generated without captured routes. Captured learned medoids, exact DeepSeek token/`tid2eid` projections, and `uniform`, `skewed`, `sparse`, and `boundary` profiles were subsequently replayed as additional historical controls. DeepSeek reports sometimes combined learned and hash results with `40/43` and `3/43` reporting weights. That was campaign reporting metadata, not permission to blend two laws in a current benchmark.

### What the analysis showed

The underlying route evidence justified looking beyond aggregate rows. Qwen B1 captures had maximum expert heights from `516` to `2047` and active support from `73` to `256`; DeepSeek learned B1 captures had maximum heights from `477` to `1982` and active support from `178` to `253`. At larger physical batches, summed histograms approached full support while retaining materially different maximum-group and padding behavior. The medoid bank retained those dimensions for timing rather than collapsing them into one average group size.

The profile weights were often highly uneven. In one historical B16 ownership screen, a `94.14%`-weight medoid regressed to `0.8802x` for the candidate while four low-weight long-tail medoids improved from `1.2166x` through `1.6098x`; the weighted result still rejected the candidate at `0.9049x`. In another screen, a single medoid with weight `0.90234375` made a geometry appear favorable while the other four profiles regressed. These results showed why medoids were useful sensitivity evidence: a weighted aggregate can hide which route shapes are driving an apparent win or loss.

Route shape and corpus also changed comparator ranking. The historical DeepSeek B16 replay measured packed grouped-MMQ versus the BF16 AITER reference at `0.942x`, `1.031x`, and `0.854x` for the IQ2_XXS pair on Chinese, English, and random inputs, respectively; the Q2_K down path measured `0.698x`, `0.747x`, and `0.537x`. The English case crossed ownership relative to the other two corpora. This was evidence that candidate ranking is route-sensitive, not evidence that every corpus or every medoid should become a timed optimization target.

### Current one-law route-bank rule

The fitted law remains the workload model. Each exact problem key now materializes a deterministic route bank from that law before timing and uses the same route tensors for the candidate and retained parent. The law is still distributional; the bank is a finite paired sample, not a claim that its finite median exhausts the residual support or is mathematically equivalent to integrating over the law.

Current benchmarks therefore do not generate five medoids, average or rotate over medoids, replay captures, or include synthetic route controls in performance ranking. They do generate a seeded bank of complete prior samples and measure one sample per selected route. Historical medoid tables remain useful for fit validation, route-sensitivity discussion, and non-timed correctness, ABI, mutation, malformed-route, and tail coverage. They cannot select a candidate, supply benchmark weights, or silently broaden the declared optimization target.

## Fit validation and benchmark handoff

The migrated non-hash prior-predictive medians were:

| Component and `T` | Observed | Prior |
| --- | --- | --- |
| Qwen, B1/S2048 | `(200, 0.811, 26.4, 18.6x)` | `(202, 0.799, 27.6, 18.1x)` |
| Qwen, B4/S2048 | `(241, 0.758, 29.0, 21.8x)` | `(243, 0.752, 30.1, 22.0x)` |
| Qwen, B16/S2048 | `(254, 0.748, 29.5, 23.7x)` | `(254, 0.719, 32.0, 22.7x)` |
| DeepSeek learned, B1/S2048 | `(230, 0.537, 34.4, 19.9x)` | `(232, 0.503, 37.3, 18.9x)` |
| DeepSeek learned, B4/S2048 | `(249, 0.486, 38.2, 19.8x)` | `(249, 0.471, 40.9, 19.3x)` |
| DeepSeek learned, B16/S2048 | `(254, 0.479, 39.6, 20.0x)` | `(254, 0.446, 44.0, 18.8x)` |
| Qwen3.8, B1/S2048 | `(355, 0.682, 1396, 22.6x)` | `(345, 0.627, 1284, 20.7x)` |
| Qwen3.8, B4/S2048 | `(454, 0.588, 4816, 26.1x)` | `(455, 0.583, 4772, 26.2x)` |
| Qwen3.8, B16/S2048 | `(496, 0.556, 18207, 27.6x)` | `(496, 0.555, 18191, 27.5x)` |

The prior remained compatible with the Qwen equal-total-token partition check at B1/S2048, B2/S1024, and B4/S512. DeepSeek extrapolation beyond S2048 was not established. These limitations belong in the fit artifact and do not justify changing the one-law route-bank benchmark procedure.

A benchmark report must record the fit provenance, exact problem key, physical size, deterministic route-bank seed, vector count, timing policy and sample metadata, route entries, and complete route tensors or a digest of them. It must state that no captured data, medoid bank, alternate corpus, or synthetic timing control was used. Candidate and retained-parent timings must use identical frozen route tensors and the same allocation/launch contract. For paired IQ2_S, it must also record that J64 row-task descriptors were prepared for every route before timing.

The current optimization claim is therefore narrow: performance is improved or rejected under one declared fitted prior law for one declared workload family/problem type. Correctness and malformed-route coverage can be broader, but no broader route result can be used to silently change the optimization target.
