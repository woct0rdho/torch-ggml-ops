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

| Key | Solution hash | Body and ownership | Source / HSACO SHA-256 |
| --- | --- | --- | --- |
| B1 | `ggsol_98b9ce678d4cb09f` | SIA5 padded M128/N64 plus M64/N64 tail, `Mixed128_64`, serial routes | `7e360fd3861fc0d...` / `a9bc91f862ba5fd8...` |
| B4 | `ggsol_a45127a1814ae728` | SIA5 padded M128/N128, `SplitRoutes16` | `fc50750afe1880fc...` / `d4d6231101501f5...` |
| B16 | `ggsol_c518830e3a59d061` | SIA5 padded M128/N128, `SplitRoutes16` | `4a0dd994382c48d...` / `75ebdd8adcbe9d99...` |

| Key | VGPR / SGPR / LDS | WMMA / barrier / wait / VMEM / VALU issue |
| --- | ---: | ---: |
| B1 | `140 / 35 / 5120 B` | `24 / 4 / 27 / 120 / 709` |
| B4 | `216 / 35 / 10240 B` | `32 / 2 / 24 / 148 / 719` |
| B16 | `216 / 35 / 10240 B` | `32 / 2 / 24 / 148 / 719` |

All three artifacts report gfx1151, wave32, code object v5, 56-byte kernargs, zero private bytes, and zero VGPR/SGPR spills. Independent generate/build/inspect roots reproduce each complete assembly and HSACO byte-for-byte. The retained Q4 B1 source remains byte-identical at SHA-256 `06a0b9d459b98d7fef7b612ba5bf608080b00f5323aa5e1893422b1d44c654b8`.

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

`pytest -q tests` passes with `711 passed` and 14 unrelated PyTorch/Python 3.14 deprecation warnings. `pre-commit run --all-files` passes pyupgrade, Ruff check, Ruff format, and ty. The worktree contains no production integration changes.
