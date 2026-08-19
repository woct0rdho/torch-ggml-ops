# GGTensile Grouped MMQ Backward Q5_K Experiment

## Purpose

Implement and optimize isolated gfx1151 grouped Q5_K backward kernels for the Qwen routed down projection. Public dispatch, generated bundle tables, registration, packaging, and HIP fallback remain unchanged pending a separate integration review.

The fitted Qwen routing-prior weighted sum of per-medoid median complete-call latency is the promotion objective. Uniform, skewed, sparse-ID, and boundary routes are correctness and diagnostic controls rather than ranking vetoes. Every retained artifact must remain exact against packed HIP, deterministic, independently reproducible, and free of private storage and spills.

## Exact Contract

For routed GEMM `g`:

```text
dY_g[M_g,2048] x W_g[2048,512] -> dX_g[M_g,512]
```

The exact aggregate-row keys are `R={16384,65536,262144}`. The physical Q5_K expert bank is `[256,2048,352]`: each 256-value block occupies 176 bytes, so each packed weight row is 352 bytes and each expert is 720,896 bytes. Inputs and outputs are contiguous BF16, accumulation is FP32 WMMA V1, and output conversion is BF16 RNE.

The grouped ABI remains the 56-byte Q4_K research ABI: `grad_output`, `packed_weight`, `grad_input`, `expert_indices`, `expert_offsets`, `num_experts`, `rows`, and `bytes_per_expert`. Route count and optional split ownership are launch geometry. No host route inspection, dense shadow, prepared bank, atomics, reduction workspace, or companion setup kernel is in scope.

The authoritative packed control is `~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf`, tensor `blk.0.ffn_down_exps.weight` (`Q5_K`, `[256,2048,352]`). The independent oracle dequantizes only selected routed experts to BF16 before matmul.

## Reuse Boundary

The grouped problem identity, route scalar plan, pointer rebasing, split-route traversal, tail predicates, launcher, inspection, and benchmark protocol transfer from Q4_K only after their quant type and packed stride become strict identity facts. The arithmetic body reuses the ordinary Q5_K leaf, including low-nibble payload, high-bit payload, scale/minimum reconstruction, decoded BF16 LDS, FP32 WMMA, and BF16 RNE stores.

Q4_K dependency batching is not a Q5_K knob. Q5_K candidates instead vary packed/scalar extraction, nibble-shift hoisting, metadata loading, geometry, schedule, LDS layout, and route ownership only when each choice changes emitted Q5_K ISA.

## Qualification

Correctness covers full exact rows, non-aligned route tails, first/non-first routes, sparse and repeated expert IDs, deterministic reruns, gradient and route mutation, active/inactive expert-weight mutation, invalid experts, malformed first/final offsets, and untouched sentinels. Candidate versus packed HIP must be BF16 bit-exact; the independent reference must remain finite with NRMSE below `0.01`.

Inspection requires gfx1151, wave32, code object v5, exact 56-byte metadata, bounded VGPR/SGPR indices, zero private bytes and spills, no scratch/calls/dynamic stack, and derived static WMMA/barrier counts. Two independent generate/build/inspect roots must produce byte-identical source and HSACO.

Timing uses warmed rotating GPU events and includes output allocation in both HIP and candidate complete-call paths. Search uses five fitted medoids; retained changes receive a disjoint confirmation-bank run with 5 warmups, 25 repeats, and reversed rotating order. TFLOPS is `2 * R * 512 * 2048 / (latency_ms * 1e9)`.

## Planned Search

- Generalize the strict grouped identity and benchmark from Q4_K-only to `{Q4_K,Q5_K}` without changing Q4 source hashes.
- Establish serial M64/N64, M128/N64, and M128/N128 Q5 controls and qualify full packed semantics.
- Screen Q5 packed extraction, nibble-shift hoisting, metadata loading, SIA/PGR/PLR, LDS padding/swizzle, and split-route factors against fitted medoids.
- Re-evaluate `Mixed128_64` only where the fitted prior contains enough routes at or below 64 rows.
- Confirm per-key winners on disjoint medoids, rebuild independently, and report final HIP/GGTensile TFLOPS and speedup.

## Completion Record

This section is updated after every coherent implementation or failed experiment.

### Campaign opened

The completed grouped Q4_K route shell, ordinary Q5_K backward record, production HIP Q5_K serial/row-task history, packed Q5_K tensor availability, and current generated Q5 decoder were reviewed. No Q4 timing result is assumed to transfer. The first implementation change will make quant type and packed expert stride strict grouped identity fields while preserving the Q4 source hash and 56-byte ABI.

### Strict Q5 identity and first controls

The grouped contract now accepts only Q4_K and Q5_K, derives the quant block geometry from the strict problem type, and rejects Q3_K/Q6_K identities. The shared benchmark validates the requested GGUF tensor type and `[256,K,row_bytes]` shape instead of assuming Q4_K's 288-byte row. Q4 source generation remains unchanged; Q5 symbols and hashes are distinct, and the Q5 guard requires `bytes_per_expert=720896`.

The Q5 pilot and initial SIA5 M128/N128, SIA5 M128/N64, SIA5 M64/N64, SIA4 double-LDS, and split-route artifacts all pass the partial 625-row route matrix. Packed HIP comparisons are BF16 bit-exact, deterministic reruns are bit-exact, and the independent dequantized BF16 oracle NRMSE is `6.85e-5`. Initial resource envelopes are 216/35/10 KiB for SIA5 M128/N128, 140/35/5 KiB for M128/N64, 100/35/4 KiB for M64/N64, and 220/35/16 KiB for double-LDS; all have zero private bytes and spills.

### First fitted-prior geometry screen

Nine-repeat search-bank weighted results are:

| Candidate | Key | Weighted HIP / candidate ms | Speedup vs HIP | Decision |
| --- | --- | ---: | ---: | --- |
| Pilot M128/N128 SIA2 | B1 | `3.7423 / 4.1997` | `0.8911x` | Rejected by timing |
| SIA5 M128/N128 | B1 | `3.7552 / 3.1414` | `1.1954x` | Loses to N64 |
| SIA5 M128/N64 | B1 | `3.8012 / 2.9121` | `1.3053x` | Provisional B1 parent |
| SIA5 M64/N64 | B1 | `3.8023 / 3.1782` | `1.1964x` | Rejected by timing |
| Split4 SIA5 M128/N128 | B4 | `9.3391 / 8.1098` | `1.1516x` | Provisional ownership control |
| Split4 double-LDS SIA4 M128/N128 | B4 | `9.4809 / 8.0422` | `1.1789x` | Provisional B4 parent; direct bracket required |
| Split8 SIA5 M128/N128 | B16 | `32.0196 / 25.4768` | `1.2568x` | Provisional B16 parent |

The pilot and M64 primary are closed. Q5 high-bit decode changes the B4 pipeline ranking relative to Q4, so double-LDS remains open at B4/B16. M128/N64 remains the B1 geometry premise; mixed tails and Q5 decode knobs will be evaluated against that parent rather than the slower M64 body.

### Q5 control qualification

Mixed M128/M64 tails, inline and hoisted nibble shifts, scalar extraction, lane sharing, and plain/padded/swizzled N64 layouts all assemble without private storage or spills. Double-LDS plus lane sharing is rejected by the typed pipeline contract before generation. The N64 vector-metadata artifact assembles at 140 VGPRs and removes three static VMEM instructions, but fails packed HIP, gradient/route/weight mutation, and independent-reference comparisons. N64 packed lane sharing fails the same gates. Both are rejected by correctness and are not timed. Vector metadata is exact on N128 double-LDS, proving the vector path itself is valid while those two N64 sharing combinations require strict rejection.

The valid-control fitted screen retains nibble-shift hoisting and the padded N64 layout. Inline shifts (`2.9642 ms`), scalar extraction (`2.9626 ms`), swizzle4 (`3.0641 ms`), swizzle8 (`3.0044 ms`), and plain LDS (`3.3103 ms`) all lose to the B1 parent at `2.9121 ms`. Mixed M128/M64 tails reduce weighted B1 candidate time to `2.6312 ms` and raise HIP-relative throughput from `1.3053x` to `1.4444x`; they advance to confirmation.

Route splitting is Q5-specific in the larger keys. At B4, split2 loses, split8 improves both pipeline controls, and split8 double-LDS SIA4 leads split8 single-LDS SIA5 by `7.8789` versus `8.0056 ms` in separate fitted screens. N128 vector metadata and inline shifts regress the double-LDS parent and are rejected. At B16, single-LDS SIA5 split8 leads double-LDS by `25.4768` versus `25.9548 ms`, while split4 regresses to `26.4956 ms`.

A 15-repeat same-process rotating bracket reverses the tiny B4 order: single-LDS SIA5 is `7.8958 ms` and double-LDS SIA4 is `7.9175 ms`, only a `0.27%` gap. The final B4 pipeline choice is deferred to disjoint confirmation. B16 remains stable in the bracket: SIA5 is `25.4031 ms` versus double-LDS at `25.8748 ms`, a `1.86%` lead. Q5 decode is heavier and the B16 routes are large, so split16 is opened as a changed-premise ownership control before closing the route search.

Split16 passes the full partial route matrix and advances decisively: SIA5 reaches `7.7649 ms` at B4 (`1.2160x` versus HIP) and `24.7998 ms` at B16 (`1.2892x`). Double-LDS split16 loses at both keys. Split32 is also correct, but regresses to `7.9267 ms` at B4 and `24.8375 ms` at B16. It helps low-weight high-skew profiles while adding inactive ownership at the dominant medoids, so split32 is removed and split16 is retained as the typed maximum.

The ordinary-Q5-inspired M256/N64 control is correct, but rises to 238 VGPRs and regresses to `8.6587 ms` at B4 and `31.3354 ms` at B16. Reduced packed decode does not overcome accumulator pressure and route masking. Geometry is therefore closed around mixed M128/N64 plus M64/N64 tails for B1 and M128/N128 split16 for B4/B16.

A 25-repeat same-process bracket on the disjoint confirmation medoids fixes the final identities. Mixed tails beat pure M128/N64 by `2.8353` versus `3.1084 ms` (`9.63%`). Split16 beats split8 by `7.8985` versus `8.0076 ms` at B4 (`1.38%`) and by `24.9272` versus `25.4524 ms` at B16 (`2.11%`). These confirmation-bank orders agree with the changed-premise mechanism decisions even where the search-bank B4 gap was noisy.

### Final qualification

The retained identities are:

| Key | Solution hash | Body and ownership |
| --- | --- | --- |
| B1 | `ggsol_98b9ce678d4cb09f` | SIA5 padded M128/N64 plus M64/N64 tail, `Mixed128_64`, serial routes |
| B4 | `ggsol_a45127a1814ae728` | SIA5 padded M128/N128, `SplitRoutes16` |
| B16 | `ggsol_c518830e3a59d061` | SIA5 padded M128/N128, `SplitRoutes16` |

| Key | VGPR / SGPR / LDS | WMMA / barrier / wait / VMEM / VALU issue |
| --- | ---: | ---: |
| B1 | `140 / 35 / 5120 B` | `24 / 4 / 27 / 120 / 709` |
| B4 | `216 / 35 / 10240 B` | `32 / 2 / 24 / 148 / 719` |
| B16 | `216 / 35 / 10240 B` | `32 / 2 / 24 / 148 / 719` |

All three artifacts report gfx1151, wave32, code object v5, 56-byte kernargs, zero private bytes, and zero VGPR/SGPR spills. Independent generate/build/inspect roots reproduce each complete assembly and HSACO byte-for-byte. The retained Q4 B1 source remains byte-identical.

Full-row qualification passes every packed HIP, independent oracle, determinism, mutation, malformed-route, and sentinel control:

| Key | Rows | HIP BF16 differences | Independent NRMSE | Deterministic differences / tail writes |
| --- | ---: | ---: | ---: | ---: |
| B1 | `16384` | `0` | `6.47e-5` | `0 / 0` |
| B4 | `65536` | `0` | `8.78e-5` | `0 / 0` |
| B16 | `262144` | `0` | `8.06e-5` | `0 / 0` |

ginal disjoint confirmation uses 5 warmups, 25 repeats, reversed rotating order, complete-call allocation in both paths, and fitted medoid weights:

| Key | HIP / GGTensile ms | HIP / GGTensile TFLOPS | Speedup vs HIP |
| --- | ---: | ---: | ---: |
| B1 | `3.9549 / 2.8721` | `8.6879 / 11.9633` | `1.3770x` |
| B4 | `9.2931 / 7.8143` | `14.7893 / 17.5881` | `1.1892x` |
| B16 | `31.7727 / 24.7802` | `17.3028 / 22.1853` | `1.2822x` |

The B1 weighted win is concentrated in the dominant confirmation medoid; its minimum individual-medoid throughput is `0.7859x`. B4 and B16 remain above HIP on every confirmation medoid, at minimum `1.1749x` and `1.2733x`. This follows the declared fitted weighted objective rather than introducing a post hoc per-medoid veto.

The isolated result does not modify public dispatch, generated bundle tables, registration, packaging, or HIP fallback. Integration remains a separate review.

### Final gates

## Reopened post-refactor optimization program

The current fixed split owners do not model the fitted long tail, and the generated tail still executes WMMAs for wholly inactive 16-row minitiles. The next work therefore mirrors the Q4_K program but retains Q5_K-specific decode and resource decisions.

1. Add wave-uniform inactive-M consumer suppression without diverging packed low/high decode or barriers. The HIP analog improved every Q5_K B4/B16 route by `1.05-6.34%`, or `3.68%` geometrically. Exact packed-HIP output and the existing nibble-shift-hoisted arithmetic are mandatory.
2. Add numeric device-built M-major J128 ownership using a separate task ABI and complete-call setup timing. The focused HIP B1 screen measured fitted-prior throughput at `1.2184x` and captured-profile throughput at `1.0558x`; synthetic timing alone caused the old rejection and is not a veto under this experiment's objective.
3. Test J64 or a single-launch mixed J128/J64 policy only after fixed J128 advances. Preserve Q5_K's task-local swizzle premise and reject any candidate with private storage or spills; the HIP row-task body already approaches `234` VGPRs.
4. Keep Q4_K `DependencyBatch4` out of Q5_K. Any decode follow-up must be a real Q5_K low/high reconstruction change that shortens live state. The prior vector-metadata and N64 lane-sharing failures remain correctness closures, and M256/N64, broad split factors, double LDS, and universal swizzle8 remain timing closures unless task ownership changes their exact premise.

Ranking uses only fitted medoids. Uniform, skewed, sparse-ID, and boundary routes remain exactness and diagnostic controls. Results and failures are appended here after each coherent experiment.

`pytest -q tests` passes with `711 passed` and 14 unrelated PyTorch/Python 3.14 deprecation warnings. `pre-commit run --all-files` passes pyupgrade, Ruff check, Ruff format, and ty. The worktree contains no production integration changes.

### Inactive-M suppression build and correctness checkpoint

The first identity-neutral implementation guards each wholly inactive wave and each inactive 16-row M minitile around the WMMA consumer, while leaving packed low/high decode, LDS traffic, barriers, and masked global access uniform. The manifest Q5_K serial control `ggsol_c638b6ec206c3b56` assembles at its prior `200 VGPR / 35 SGPR / 8192 B LDS` envelope with zero private bytes or spills and unchanged identity and static WMMA/barrier counts. The complete grouped build record is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-manifest-build/build.json`.

Its strict 625-row boundary matrix passes packed HIP, independent BF16, deterministic rerun, gradient/route/active/inactive-weight mutations, malformed expert/offset controls, and tail sentinels. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-correctness.json`. This is a build/correctness checkpoint only; selected-key fitted parent/candidate timing remains the promotion gate.

The selected B1 `ggsol_98b9ce678d4cb09f` source was built independently from the pre-change `c41913f` worktree and the suppression worktree. Both inspect at `140 VGPR / 35 SGPR / 5120 B LDS`, zero private bytes and spills, 24 static WMMAs, and four barriers. A same-process fitted search-bank bracket with allocation in both paths, three warmups, nine repeats, and rotating order improves weighted latency `2.6448 -> 2.5989 ms`, or `1.0177x` parent throughput. Outputs are bitwise equal on all five medoids. Four medoids, including the dominant profile, improve; the remaining low-weight profile is a `0.9926x` near tie. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b1-search9.json`. This advances to disjoint confirmation.

Disjoint confirmation with five warmups, 25 repeats, reversed base order, and the same complete-call allocation contract confirms `2.8357 -> 2.7996 ms`, or `1.0129x`. Every confirmation medoid is bitwise exact and the dominant profile improves. Two low-weight profiles measure `0.9743x` and `0.9954x`; they remain diagnostics under the declared weighted objective. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b1-confirm25.json`. Inactive-M suppression is retained for Q5_K B1 and advances to B4/B16 qualification.

The selected split16 B4 `ggsol_a45127a1814ae728` and B16 `ggsol_c518830e3a59d061` artifacts reproduce their parent envelopes at `216 VGPR / 35 SGPR / 10240 B LDS`, with zero private bytes or spills. Both pass the strict 625-row packed-HIP, independent BF16, deterministic, mutation, malformed-route, and sentinel matrix. Reports are `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b4-correctness.json` and `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b16-correctness.json`. Fitted parent/candidate timing remains outstanding.

The B4 fitted search-bank bracket improves `7.8476 -> 7.7645 ms`, or `1.0107x`, with every medoid bitwise exact and faster. The minimum ratio is `1.0091x`. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b4-search9.json`; the close weighted margin advances to a longer disjoint bracket rather than immediate retention.

The B16 fitted search-bank bracket improves `24.7676 -> 24.4248 ms`, or `1.0140x`, with bitwise equality throughout. The dominant medoid improves `1.0147x`; two low-weight medoids are near-tie regressions at `0.9974x` and `0.9959x`. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b16-search9.json`; B16 advances to disjoint confirmation under the weighted objective.

The disjoint B4 confirmation bracket measures `7.8109 -> 7.6883 ms`, or `1.0159x`, with every medoid bitwise exact and faster. The minimum ratio is `1.0131x`. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b4-confirm25.json`. Inactive-M suppression is retained for Q5_K B4.

The disjoint B16 confirmation bracket reverses the search-bank order and rejects suppression for that key: weighted latency regresses `24.9318 -> 25.0716 ms`, or `0.9944x` parent throughput, with four of five medoids slower despite bitwise equality. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b16-confirm25.json`. The retained implementation must leave selected Q5_K B16 on its parent path; the search-bank `1.0140x` result is not reproducible.

The EvoTensile median-log robust-scale analysis shows that all three Q5_K 25-repeat weighted intervals still cross zero: candidate-minus-parent is `[-2.557,0.031]%` at B1, `[-3.730,0.581]%` at B4, and `[-0.103,1.230]%` at B16. The aggregate analysis is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-confirm25-confidence.json`. B1/B4 retention and B16 rejection are therefore provisional pending adaptive sample top-ups rather than being treated as resolved from point estimates alone.

The 75-repeat B1 top-up resolves the comparison: `2.8741 -> 2.8215 ms`, or `1.0186x`, with a median-log robust 95% candidate-minus-parent interval of `[-2.743,-0.923]%`. The dominant medoid improves; three low-weight medoids vary around a tie and remain diagnostic. Reports are `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b1-confirm75.json` and `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b1-confirm75-confidence.json`. Inactive-M suppression is retained for Q5_K B1.

The 75-repeat B4 top-up resolves that comparison as well: `7.9329 -> 7.7282 ms`, or `1.0265x`, with every medoid faster and a robust 95% candidate-minus-parent interval of `[-3.835,-1.345]%`. Reports are `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b4-confirm75.json` and `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b4-confirm75-confidence.json`. Inactive-M suppression is retained for Q5_K B4.

The 75-repeat B16 top-up reverses the 25-repeat point estimate but remains unresolved: `25.0824 -> 25.0314 ms`, or `1.0020x`, with a robust 95% candidate-minus-parent interval of `[-0.597,0.192]%`. Reports are `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b16-confirm75.json` and `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b16-confirm75-confidence.json`. This is neither a supported win nor a supported loss; a 200-repeat top-up will test practical equivalence before the source path is finalized.

The 200-repeat B16 top-up resolves the ambiguity against suppression: weighted latency regresses `25.0288 -> 25.0906 ms`, or `0.9975x` parent throughput, with four of five medoids slower. The median-log robust 95% candidate slowdown interval is `[0.015,0.479]%`, entirely above zero. Reports are `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b16-confirm200.json` and `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-b16-confirm200-confidence.json`. Inactive-M suppression is retained for selected Q5_K B1/B4 and rejected for selected Q5_K B16.

The retained emitter now selects the parent consumer path for exact Q5_K B16 contracts without adding a solution field or changing the identity hash. A fresh selected rebuild of `ggsol_c518830e3a59d061` is byte-identical to the `c41913f` parent in both assembly (`sha256 4a0dd994382c48d7dfc20027fb6f4997884c34d53760196dd7981b5d25d3ad9f`) and HSACO (`75ebdd8adcbe9d9934069feb604191b7628367e17459086892fdb2c64f9981cc`). Q5_K B1/B4 still emit the measured guards. The selected build record is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-large-selected/build.json`.

Final selected full-row qualification passes at `16384`, `65536`, and `262144` rows with packed-HIP bit equality, deterministic reruns, all active/inactive mutation controls, malformed-route sentinels, and independent BF16 NRMSE below `0.01`. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-q5-full-correctness.json`.
