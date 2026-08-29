# GGTensile grouped MMQ forward paired IQ2_XXS experiment

## Scope

This record covers the isolated research kernel for DeepSeek's paired routed gate and up projections. The exact production contract is:
- 256 physical experts;
- two packed `IQ2_XXS` weight banks, one gate and one up projection;
- output features `N = 2048` and reduction features `K = 4096`;
- aggregate routed rows `R = 12,288`, `49,152`, or `196,608`, plus the `R = 35` boundary key;
- packed weight shape `[256,2048,1056]` per projection, with 16 256-value blocks and 66 bytes per block;
- one shared Q8_1 `F32_D4` activation workspace with shape `[32,R,144]`;
- two independent BF16 destinations with shape `[R,2048]`;
- device-resident int64 expert IDs and cumulative int32 route offsets; and
- at most 256 route entries, with a final valid offset equal to `R`, monotonic non-overlapping valid spans, and inert invalid experts.

The candidate computes both projections in one serial-route workgroup dataflow. Projection-indexed workgroups, two adjacent single-projection launches, and sequential full-projection phases that reload activation data are controls, not paired fusion.

Public selectors, generated bundles, extension registration, packaging, HIP fallback behavior, and prepared-weight representation changes remain outside this experiment. The retained identity is a research artifact and is not public dispatch.

## Authorities and arithmetic

`IQ2_XXS` is kept distinct from `IQ2_S`. Its packed block is one FP16 `d` followed by 64 bytes of four-byte grid indices and parity-expanded sign words. The 256-entry codebook is extracted at generation time from `csrc/vendor/llama_cpp/iq2_xxs_grid.cuh`; no second hand-maintained table is introduced.

The paired solution uses the selected-half K128 interleaved topology already qualified for the paired research family. One 128-thread wave32 workgroup owns a 64-row by 64-column tile for both projections. It stages each activation plane once, decodes a selected 128-value IQ2_XXS half into a reusable 64-row LDS image, computes the first projection, overwrites the weight image with the second projection, and computes the second projection while the activation plane remains resident.

The decoded weight image has 64 rows with a 160-byte stride: 128 payload bytes followed by two FP32 scale slots. IQ2_XXS has one scale per K32, so each pair of K16 WMMAs is accumulated in integer space before one FP32 weight-scale and activation-scale correction. This ordering is required for bitwise agreement with the installed controls.

## Failed candidates and corrections

The first coherent IQ2_XXS candidate produced non-finite output because its scale staging wrote to an incorrect LDS address. The scale address was corrected and the candidate became finite, but it then exposed two independent contract errors.

The first corrected body inherited the paired IQ2_S BF16 store shift of 10, which implies a 1,024-byte row stride. That is correct for `N = 512`, but IQ2_XXS requires a 4,096-byte row stride for `N = 2048`. The resulting cross-row workgroup collisions appeared as intermittent zero tiles. IQ2_XXS now emits the N2048-specific shift of 12, and the source test rejects the old shift.

The remaining R35 mismatch was one BF16 step in both projections. IQ2_XXS scales cover K32 while the initial body corrected each K16 fragment separately. The final body accumulates both integer fragments first and performs one FP32 correction per K32 scale. The corrected R35 candidate is bitwise identical to installed J64, installed J80, the public pair, and the installed serial control. Direct-to-LDS remains closed separately because the gfx1151 assembler rejected the required instruction forms.

No IQ2_XXS row-task candidate was retained. The installed workload has J64 and J80 serial controls but no installed paired IQ2_XXS row-task ABI/control; row-task ownership therefore remains a separate campaign.

## Artifact qualification

The retained R35 identity is the paired IQ2_XXS `R35/N2048/K4096` research artifact.

The durable build report is `~/tmp/torch-ggml-ops/ggtensile-grouped-iq2-xxs-pair-p0-build.json`. Independent first and second builds for `R = 35`, `12,288`, `49,152`, and `196,608` produced byte-identical source and code objects. The R35 object has 26,120 code bytes; each production object has 26,136 code bytes.

All retained artifacts are gfx1151 code-object v5, wave32, with an 80-byte paired kernarg segment, 148 VGPRs, 44 SGPRs, and 19,456 bytes of fixed LDS. Each contains 128 static WMMAs and eight barriers. Inspection found zero private storage, zero VGPR and SGPR spills, no scratch instructions, no calls, and no dynamic stack. GGTensile launches with zero dynamic LDS. The installed controls use separate validated dynamic-LDS contracts: 28,928 bytes for J64 and 31,552 bytes for J80.

The research ABI and physical plan enforce the exact geometry, two projections, paired selected-half decode, Q8_1 activation layout, output stride, and serial cumulative-offset ownership. Existing paired IQ2_S and Q3_K source identities remain byte-for-byte unchanged.

## Semantic qualification

The authoritative model is `~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf`, using `blk.0.ffn_gate_exps.weight` and `blk.0.ffn_up_exps.weight`. The aggregate semantic report is `~/tmp/torch-ggml-ops/ggtensile-grouped-iq2-xxs-pair-p0-production-semantic.json`; unlike the earlier per-invocation report, it contains all four row counts.

The valid route uses expert IDs `(0,7,7,12,255)` and monotonic cumulative offsets `(1,R/4,R/2,3R/4,R)`. Candidate gate and up outputs are bitwise equal to installed J64, installed J80, and `torch_ggml_ops.grouped_mmq_pair` at every row count. Reruns are bitwise deterministic and both destinations are finite.

An independent bounded reference reconstructs the authoritative Q8 workspace and uses `gguf.quants.dequantize(..., GGMLQuantizationType.IQ2_XXS)` for only the routed experts. The maximum absolute error is `0.015625` at R35 and `0.03125` at all production sizes. RMSE is bounded by `0.00013125` at R35 and is approximately `0.000116` to `0.000120` at production sizes.

The malformed route uses expert IDs `(0,256,12,255)` and cumulative offsets `(R/4,R/2,3R/4,R)`. The invalid expert span remains filled with the BF16 sentinel, candidate and J64 outputs agree bitwise, and the valid tail changes at every row count. Active-bank, activation, repeated-route, sparse, boundary, and invalid-route checks were also included in the qualification sequence; invalid route metadata is never used as a host-side descriptor.

## Complete-call performance

Timed calls include BF16 activation quantization, activation workspace allocation, both projections, device-resident route tensors, and CUDA/HIP event synchronization. Route metadata is not copied to the host, no CPU descriptors are built, and no `.item()` or implicit synchronization is used in the timed functions. The candidate is compared with two sequential installed serial launches and with the public pair.

The fitted-prior screen uses 512 deterministic draws, five weighted medoids, three warmups, and nine order-rotated repeats. Its durable report is `~/tmp/torch-ggml-ops/ggtensile-grouped-iq2-xxs-pair-p0-fitted9.json`.

| Batch | Rows | Candidate ms | J64 parent ms | J80 parent ms | Public pair ms | Public/candidate | Minimum public profile |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12,288 | 30.0722 | 33.9296 | 39.2354 | 33.8717 | 1.1263x | 1.0530x |
| 4 | 49,152 | 83.4574 | 92.4748 | 90.3963 | 92.3221 | 1.1062x | 1.0842x |
| 16 | 196,608 | 302.2398 | 335.3922 | 305.7019 | 305.1911 | 1.0098x | 0.9990x |

The one effectively neutral B16 fitted medoid triggered the required longer confirmation rather than being treated as universal dominance. The B16 25-repeat report is `~/tmp/torch-ggml-ops/ggtensile-grouped-iq2-xxs-pair-p0-b16-confirm25.json`.

| Batch | Rows | Candidate ms | J64 parent ms | J80 parent ms | Public pair ms | Public/candidate | Minimum public profile |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 196,608 | 304.0936 | 337.5326 | 307.8063 | 307.4068 | 1.0109x | 1.0068x |

The B1/B4 25-repeat transfer confirmation is `~/tmp/torch-ggml-ops/ggtensile-grouped-iq2-xxs-pair-p0-b1-b4-confirm25.json`.

| Batch | Rows | Candidate ms | J64 parent ms | J80 parent ms | Public pair ms | Public/candidate | Minimum public profile |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12,288 | 30.4352 | 35.1876 | 40.2553 | 34.5979 | 1.1368x | 1.0605x |
| 4 | 49,152 | 84.4732 | 93.5025 | 91.2841 | 93.4559 | 1.1063x | 1.0732x |

### Initial P0 throughput (historical)

This table records the original P0 selection before X1 selector fusion and the later typed J80 geometry. It is retained as selection context; the current retained speeds are reported at the end of this log. Effective TFLOPS is nominal dense-equivalent complete-call throughput, calculated as `4 * rows * N * K / seconds`: two FLOPs per FMA across both projections. B1 and B4 use the transfer confirmation above; B16 uses its dedicated 25-repeat confirmation.

| Batch | Rows | GGTensile complete | Public HIP complete | GGTensile effective TFLOPS | HIP effective TFLOPS | HIP/GGTensile speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12,288 | 30.4352 ms | 34.5979 ms | 13.55 | 11.92 | 1.1368x |
| 4 | 49,152 | 84.4732 ms | 93.4559 ms | 19.52 | 17.65 | 1.1063x |
| 16 | 196,608 | 304.0936 ms | 307.4068 ms | 21.69 | 21.46 | 1.0109x |

All 25-repeat outputs remained bitwise exact. The weakest B16 confirmation profile is still faster than public, while B1 and B4 transfer retain material margins.

## Synthetic route controls

The control-route bank uses four deterministic distributions at each production row count: uniform, skewed, sparse, and boundary. It uses device-resident int64 IDs and cumulative int32 offsets, three warmups, and nine order-rotated repeats. The durable report is `~/tmp/torch-ggml-ops/ggtensile-grouped-iq2-xxs-pair-p0-synthetic9.json`.

| Batch | Rows | Weighted public/candidate | Minimum profile public/candidate | All outputs exact |
| ---: | ---: | ---: | ---: | :---: |
| 1 | 12,288 | 1.3225x | 1.1764x | yes |
| 4 | 49,152 | 1.1170x | 1.1142x | yes |
| 16 | 196,608 | 1.0245x | 1.0147x | yes |

All 12 synthetic profiles matched J64, J80, and public output bitwise. The weakest route distribution remains positive at every production size.

## Repository qualification

The post-IQ2_XXS validation matrix completed with:
- `pytest -q tests/ggtensile`: 439 passed;
- `pytest -q`: 540 passed;
- `ruff check .`: passed;
- `ruff format --check .`: passed;
- `ty check`: passed;
- `python -m compileall -q tools tests`: passed;
- `git diff --check`: passed; and
- changed-file pre-commit hooks (`pyupgrade`, Ruff check, Ruff format, and `ty check`): passed.

The repository test run emitted only the 14 known Python 3.14 `torch.jit.script_method` deprecation warnings. No public dispatch, generated bundle table, package, registration, HIP fallback, or prepared-weight representation file changed in this campaign.

## Reopened source-level optimization review

The repaired IQ2_XXS paired result and its J64/J80 comparison remain historical results for their original bodies. The following probes change code generation or geometry and are therefore pending, independent identities rather than retroactive changes to the retained result.

### X1 paired sign-selector fusion

The paired IQ2_XXS decoder still uses the older sign-selector sequence built from `0x204081`, `0x01010101`, and `v_lshl_or_b32`. The common IQ2_S form uses `0x810204`, `0x04040404`, and an invariant selector base. Source-level enumeration of all 16 sign-nibble values found the two constructions equal modulo 32 bits. X1 replaces the older sequence only after materializing and validating the invariant selector SGPR under the IQ2_XXS physical plan.

The source estimate is one fewer VALU instruction per signed codebook dword, approximately 64 instructions per K256 block and approximately 1,024 per K4096 row tile. These static, per-block, and amortized counts are qualification inputs, not timing. X1 remains unqualified until the exact selector materialization, register ownership, assembly encoding, artifact disassembly, resources, bitwise output, mutation matrix, deterministic rebuilds, and warmed complete-call and prequantized timings pass independently.

### X2 paired zero-accumulator lifetime

The dedicated read-only `v124:v131` zero bank is rewritten before every projection even though it is consumed as the WMMA accumulator and is not identified as clobbered between projections. X2 initializes it once before the K4096 block loop. The repeated body loses 24 static initialization moves, and the amortized dynamic reduction is approximately 504 moves per K4096 row tile after retaining one eight-register setup. The static body, per-projection, and amortized counts remain separate evidence; X2 does not change nominal VGPR, SGPR, LDS, WMMA, or arithmetic requirements by assumption.

### X3 pre-scaled `d`

The decoder currently multiplies by `d` and then by `0.125` in each half. X3 pre-scales `d` once per half and uses that value for the local scale path. The source estimate is four fewer instructions per K256 block and approximately 64 fewer per K4096 row tile. A finite-FP16 source reassociation check over all finite FP16 values and odd scale values from 1 through 127 produced no differing FP32 bit patterns across 4,063,232 comparisons, but that result does not approve the device candidate. Exact output comparison remains mandatory.

### X4 paired epilogue scheduling

The two projections recompute equivalent column, row-mask, and vector-offset setup, but their destination pointers remain independent. X4 first tests shared column setup, then a separate address-materialization identity if dead `c` registers and store order permit it. X4b may interleave two independent exact `v_bfe_u32`/`v_add3_u32` BF16 RNE chains using proven-dead scratch VGPRs. No aliasing of output pointers, approximation, or relaxed rounding is allowed.

### X5 J80 geometry and X6 metadata

The current fused J64 candidate is only approximately 1.09% ahead of the public B16 result, while the installed serial J80 control is already close. X5 is a new serial-only IQ2_XXS geometry identity intended to test whether 25% more routed rows per workgroup amortizes decode, barriers, and route-loop overhead. It requires a new physical plan, output-tail proof, resource point, exact route matrix, deterministic artifacts, and complete-call and prequantized comparison against both fused J64 and the installed control. It must not be combined with `DeviceRowTasks64`; IQ2_XXS row-task ownership remains a separate future identity and is not inferred from IQ2_S or Q3_K.

The HIP control includes `.amdhsa_workgroup_processor_mode 1`, while the inspected GGTensile paired artifacts do not. X6 is an isolated metadata A/B with no assumed benefit. It must prove identical ABI, code-object, resources, correctness, and deterministic generation before any timing interpretation.

The current `s_clause 7`, waits, and barriers remain controls. Clause changes, compiler-produced `s_delay_alu`, and bank-valid GFX11 VOPD pairings are separate low-priority artifact screens. Direct-to-LDS remains closed because the required instruction form was rejected by the gfx1151 assembler, and no wait or barrier is removed by inference.

### Qualification and recursive review

X1-X6 are unqualified until each distinct typed identity and serialized policy passes exact gfx1151 code-object-v5 assembly and linking, ABI and metadata inspection, resource and disassembly inspection, independent exactness and mutation checks, deterministic rebuilds, and warmed complete-call and prequantized timing. Static instruction reductions do not substitute for device timing. Any composition of individually qualified probes receives a new identity and repeats every gate.

The historical retained classification remains scoped to the repaired J64/J80 bodies and the old comparator premise. Implement and qualify every actionable finding, then repeat the complete source, artifact, resource, correctness, determinism, and timing review from the changed premise. A fresh recursive pass must find no actionable in-contract mechanism before this record can close. Public dispatch, generated bundles, packaging, registration, HIP fallback, and prepared representations remain unchanged.

### X1 qualification result

X1 was materialized as an isolated candidate with the invariant selector SGPR and built independently twice for R35, B1, B4, and B16. The source body removed 64 selector VALU operations and one dependent scalar/vector construction from the old sequence; the linked artifact reported 2,152 VALU issue instructions versus 2,216 for the retained body, with unchanged 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, zero spills, and zero private bytes. All route, malformed-route, independent-reference, finite-output, mutation, and deterministic-build checks were exact.

The fitted complete-call medians improved from 30.0722/83.4574/302.2398 ms to 29.6455/81.8842/296.7120 ms at B1/B4/B16, or 1.42%/1.88%/1.83%. The independent 25-repeat B16 confirmation measured 298.2896 ms versus 304.0936 ms for the retained body, or 1.91%, with every medoid exact and a minimum public/candidate ratio of 1.0211x. X1 is rejected as a retained implementation because it remains below the two-percent advancement gate. Its exact decode equivalence and artifact evidence remain a scoped result; no source or public identity changed.

X1 is now closed under the changed X1 premise. The remaining X2-X6 probes stay independent and must not be composed with X1 without a new identity and a fresh recursive review.

### X2 qualification result

X2 removed the four repeated `v124:v131` zero-bank initializations from the statically emitted body and retained one eight-register setup before the block loop. Independent builds were byte-deterministic; the artifact changed from 2,216 to 2,192 VALU issue instructions while retaining 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, zero spills, and zero private bytes. All exact route, malformed-route, independent-reference, finite-output, mutation, and rerun checks passed.

The fitted complete-call medians were 30.4010/84.2791/303.0949 ms at B1/B4/B16 versus 30.0722/83.4574/302.2398 ms for the retained body, or +1.09%/+0.99%/+0.28%. The B16 screen included a profile below parity (`0.99897x` public/candidate). X2 is rejected as a retained implementation for neutral-to-regressive timing; no confirmation campaign or source integration follows.

X2 is closed under its changed premise. X3-X6 remain independent and must be evaluated without composing rejected X1 or X2 code.

### X3 qualification result

X3 moved the exact `* 0.125` factor from each local decoded scale into the four statically emitted FP32 `d` conversions, removing eight local multiplies and adding four invariant multiplies. Two builds were byte-deterministic. The artifact changed from 2,216 to 2,212 VALU issue instructions while retaining 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, zero spills, and zero private bytes. All exact route, malformed-route, independent-reference, finite-output, mutation, and rerun checks passed.

The fitted complete-call medians were 30.1428/83.1028/300.6703 ms at B1/B4/B16 versus 30.0722/83.4574/302.2398 ms for the retained body, or +0.24%/-0.42%/-0.52%. No point cleared the two-percent advancement gate. X3 is rejected as timing-neutral; no confirmation campaign or source integration follows.

X3 is closed under its changed premise. X4-X6 remain independent and must be evaluated without composing rejected X1-X3 code.

### X1 stability-policy reopening and typed retention

The fixed two-percent gate was subsequently withdrawn for candidates with stable evidence of a smaller gain. A direct same-session 25-repeat A/B against the instruction-identical retained P0 parent measured complete-call candidate/parent medians of 30.3345/30.9198 ms at B1, 83.0066/84.3965 ms at B4, and 299.2489/304.4231 ms at B16, or gains of 1.93%/1.67%/1.73%. Prequantized body gains were 1.97%/1.62%/1.91%. Every complete-call and body medoid was faster; minimum complete-call ratios were `1.01712x`, `1.01161x`, and `1.01453x`. Conservative 95% log-time intervals excluded parity for all fifteen complete-call medoids and fourteen of fifteen body medoids; the remaining body interval overlapped parity by 0.02% and did not favor the parent.

The retained typed identity serializes `TwoLaneSelectedHalfIQ2XXSFusedSelector` and admits it only for IQ2_XXS serial-route ownership. Its lowering materializes `0x03020100` once in the plan-owned scalar register and replaces each three-instruction selector construction with the algebraically equivalent `0x810204` multiply plus `v_and_or_b32`. The typed source reproduced the qualified probe instruction-for-instruction after canonical symbol normalization. Two typed builds were byte-deterministic and retained 2,152 VALU issue instructions, 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, zero spills, and zero private bytes. All bounded and production semantic checks passed exactly.

A fresh typed 25-repeat A/B measured complete-call gains of 1.88%/1.79%/1.86% and body gains of 2.00%/1.90%/1.74% at B1/B4/B16. Every medoid remained faster; minimum complete-call ratios were `1.01140x`, `1.00977x`, and `1.01801x`. Fourteen of fifteen complete-call and fourteen of fifteen body intervals excluded parity at 95% confidence, and neither remaining interval favored the parent. X1 is retained as `iq2_xxs_k128_interleaved_fused_selector()`.

This reopening does not retain X2 or X3 and does not compose them with X1. Public dispatch, generated bundle registration, packaging, prepared representations, and HIP fallback behavior remain unchanged. X4-X6 require independent identities and qualification against the retained X1 body where their mechanism is compatible.

### X6 processor-mode metadata result

X6 inserted `.amdhsa_workgroup_processor_mode 1` into the kernel descriptor of the retained typed X1 source for R35, B1, B4, and B16. The configured gfx1151 assembler accepted the directive. Two independent passes were deterministic, and each transformed source had the same object and HSACO bytes as the unmodified X1 parent at the corresponding row count. Inspection was consequently identical: 148 VGPRs, 44 SGPRs, 19,456 bytes of LDS, 128 WMMAs, eight barriers, zero private bytes, zero spills, the 80-byte ABI, and unchanged VALU/VMEM/LDS/wait/clause counts.

The metadata-only artifacts passed the full R35 and production route, malformed-route, independent-reference, rerun, active-bank, activation, and finiteness matrix against the retained X1 body and installed controls. A five-warmup, alternating 25-repeat parent/candidate measurement over the DeepSeek confirmation medoids was mixed and noise-scale: weighted complete-call parent/candidate ratios were `1.00107x`, `0.99714x`, and `0.99873x` at B1/B4/B16, while prequantized-body ratios were `1.00187x`, `0.99836x`, and `1.00009x`. Complete-call minimum ratios were `0.99660x`, `0.99658x`, and `0.99649x`; all robust 95% per-medoid intervals crossed parity.

X6 is closed as an assembler-accepted but executable-inert metadata spelling under this toolchain. No serialized policy, typed source identity, lowering change, composition, catalog entry, public integration, or source commit follows. The conclusion does not predict the behavior of processor mode under a different compiler artifact or target.

### X4 epilogue qualification result

The narrow shared-column probe removed six projection-1 setup instructions. Independent builds were deterministic and exact, and the linked body changed from 2,152 to 2,146 VALU issues with unchanged 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, waits, private storage, and spills. Its nine-repeat complete-call parent/candidate ratios were `0.99636x`, `1.00007x`, and `0.99875x` at B1/B4/B16; body ratios were `1.00049x`, `0.99835x`, and `0.99909x`. The direction is mixed and noise-scale.

A broader address-materialization probe reused dead accumulator registers to carry the common column and row terms. It removed 25 artifact VALU issues, from 2,152 to 2,127, without changing resources. The implementation preserved the IQ2_XXS N2048 row shift of 12 and passed the full route/reference/mutation matrix. Complete-call ratios were `1.00110x`, `0.99904x`, and `0.99927x`; body ratios were `0.99810x`, `0.99827x`, and `0.99969x`. It is also timing-neutral.

The exact width-two BF16 probe interleaved adjacent `v_bfe_u32`/`v_add3_u32` RNE chains using proven-dead scratch while preserving instruction counts and resources. It passed exact qualification, but complete-call ratios were `0.99651x`, `1.00139x`, and `1.00116x` while body ratios were `1.00342x`, `0.99987x`, and `1.00082x`. Complete/body disagreement closes it. These results exhaust X4's allowed shared-setup, materialized-address, and exact BF16-scheduling subprobes; no X4 source identity is retained.

### X1 plus X3 composition result

Because X1 became the canonical parent, X3 was rebuilt as a distinct X1+X3 composition rather than transferring its old result. Two builds at R35 and every production row count were deterministic. The candidate reduced VALU issues from 2,152 to 2,148 and retained all parent ABI, VGPR, SGPR, LDS, WMMA, barrier, wait, private-storage, and spill properties. The complete R35 and production valid-route, malformed-route, reference, rerun, and mutation matrices were exact.

The nine-repeat complete-call parent/candidate ratios were `1.00554x`, `1.00229x`, and `1.00153x` at B1/B4/B16, but body movement was approximately flat. A 25-repeat top-up measured complete ratios of `1.00166x`, `1.00126x`, and `1.00163x` and body ratios of `1.00071x`, `0.99980x`, and `1.00046x`. All 30 complete-call and body per-medoid 95% intervals crossed parity, and profile direction was mixed. X1+X3 is closed as timing-neutral; no typed composition or source retention follows.

### X5 J80 geometry qualification and B16 retention

The exploratory J80 plan increased the activation tile from 64 to 80 rows, the activation image from 9,216 to 11,520 bytes, total fixed LDS from 19,456 to 21,760 bytes, activation payload from 16 to 20 VGPRs, and M fragments from four to five. The resulting artifact uses 180 VGPRs, 44 SGPRs, 160 static WMMAs, eight barriers, zero private bytes, and zero spills. Its representative counts are 2,590 VALU issues, 3,166 VALU operations, 576 VOPD operations, 148 VMEM instructions, 348 LDS instructions, 52 waits, and ten clauses, versus X1 J64's 2,152/2,664/512/128/298/52/eight.

The initial monkeypatched artifacts intentionally reused the J64 symbol and key and relaxed only the inspector's expected static WMMA count from 128 to 160. They were therefore exploratory and non-retainable. Independent builds at R35, B1, B4, and B16 were nevertheless byte-deterministic, and `ggtensile-grouped-iq2-xxs-pair-j80-semantic.json` reports exact valid and malformed routes, independent references, reruns, both projection mutations, activation mutation, finite outputs, and inert invalid spans at every row count.

The sample-bearing 25-repeat probe was analyzed in median-log space with robust sigma equal to the maximum of standard deviation, scaled MAD, and scaled IQR. B1 was confidently slower on every complete and body medoid. B4 was distribution-dependent: the weighted result favored J80 only because medoid 234 has weight `0.90234375`; the other four complete-call medoids regressed, including two intervals wholly above parity. B16 was faster on all five complete and all five body medoids. These results reject J80 as a global J64 replacement and close the B1 and B4 keys.

J80 was then represented canonically by serialized `macro_tile = [80,64]`, restricted to IQ2_XXS fused-selector serial-route ownership, with a distinct physical allocation and symbol. The typed lowering reproduced the qualified probe instruction-for-instruction after symbol normalization. Two typed builds at all four row counts were byte-deterministic and retained the exact 180-VGPR, 44-SGPR, 21,760-byte-LDS resource point and artifact counts above. The typed semantic report `ggtensile-grouped-iq2-xxs-pair-j80-typed-semantic.json` again passed every exact route, malformed-route, reference, rerun, and mutation gate.

The final seven-warmup, alternating 25-repeat typed report is `ggtensile-grouped-iq2-xxs-pair-j80-typed-ab25.json`; robust intervals are in `ggtensile-grouped-iq2-xxs-pair-j80-typed-confidence.json`.

| Batch | Candidate complete | X1 J64 complete | Candidate body | X1 J64 body | Weighted complete gap 95% | Disposition |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 33.6193 ms | 30.5768 ms | 33.5219 ms | 30.0065 ms | +8.97% to +10.93% | rejected |
| 4 | 82.4434 ms | 83.8315 ms | 79.9258 ms | 81.1301 ms | -1.87% to -1.45% | rejected: profile-dependent |
| 16 | 294.6356 ms | 301.8988 ms | 282.7872 ms | 290.4282 ms | -2.55% to -2.26% | retained |

Every B16 complete-call interval was below parity, ranging from `[-3.17%,-2.55%]` to `[-2.46%,-1.98%]`; every body interval was also below parity. The weighted complete time reduction is 2.41% with a 95% interval of 2.26%-2.55%, and the weighted body reduction is 2.63% with a 2.40%-2.85% interval. The typed J80 candidate is `1.02465x` faster than X1 J64 complete and `1.02700x` faster in the body. It is also `1.05366x` faster complete than the installed serial J80 control.

Only the exact B16 research key, aggregate rows 196,608 with J80 fused-selector serial ownership, is retained. The implementation commit is `5c0a28e`; it changes no catalog, generated bundle, dispatch, registration, package, prepared representation, or HIP fallback. B1 and B4 remain explicit negative timing results, and the constructor's availability for research generation is not deployment evidence for those shapes.

### Post-J80 recursive final review

A separate read-only pass after typed J80 retention reread the paired model, strict spec, physical register/LDS ownership, IQ2_XXS and shared mechanics, inspection, generated J64/J80 artifacts, every X1-X6 result, the other paired formats, fixed and grouped records, and the exact gfx1151 wait/barrier/VOPD constraints. J80 is canonical only through serialized `[80,64]`, owns its 180-VGPR and 21,760-byte-LDS plan, and does not alter the J64 stream.

The changed geometry does not silently inherit or overturn X2-X4. B16 zero-bank hoisting was adverse on J64; X1+X3, shared-column setup, materialized addresses, and exact BF16 interleaving were timing-neutral or directionally incoherent. J80 provides no new dependency or resource transition that predicts a reversal. Its larger body dilutes fixed setup savings, while the one additional epilogue fragment does not change the dependencies behind the linearly repeated address or BF16 forms. This is not J80 timing evidence for an unbuilt composition: any future composition still requires a distinct typed key and full qualification. Rows above 80 and IQ2_XXS row-task ownership are separate geometry or ABI campaigns without a current exact workload premise; constructor availability is not selection evidence.

The eight barriers continue to protect projection-overwritten weight LDS and activation reuse, so no wait or barrier is removed without an artifact-specific hazard proof. No new bank-valid VOPD, delay, direct-to-LDS, exact BF16, clause, or direct-packed representation mechanism was found. The fresh pass therefore finds no actionable in-contract IQ2_XXS mechanism. Public integration and prepared representations remain deferred scope.

### Final retained throughput versus HIP

The final table reports prequantized multiply-only throughput. HIP and GGTensile consume the same activation workspace produced by the shared HIP quantizer, so activation quantization and other complete-call work are excluded. Nominal dense-equivalent throughput is `2 * aggregate rows * N * K / time`, doubled for paired two-projection kernels. GGTensile/HIP speedup is HIP body time divided by GGTensile body time.

| Aggregate rows | Public catalog hash | HIP TFLOPS | GGTensile TFLOPS | GGTensile/HIP speedup |
| ---: | :--- | ---: | ---: | ---: |
| 12,288 | `ggpair_f12ae9a800bd5ecb` | 11.851 | 13.741 | 1.1594x |
| 49,152 | `ggpair_6a7d51c21e083af7` | 17.985 | 20.329 | 1.1303x |
| 196,608 | `ggpair_c87599a2a8d4b1f3` | 22.054 | 23.329 | 1.0578x |

All timed outputs in the retained qualification were exact.

## Current One-Law Retuning and Reopening Flags

This annotation records the current paired IQ2_XXS direct-kernel benchmark only; it does not change X1/J80 retention, catalog, dispatch, or integration status. The primary reports are under `~/tmp/torch-ggml-ops/fresh-grouped-fwd-one-law-multiply-20260823/`, with a 75-repeat B1 top-up under `~/tmp/torch-ggml-ops/fresh-grouped-fwd-one-law-multiply-topup-20260823/`.

### Retune and re-evaluation flag

- IQ2_XXS B1, `ggpair_f12ae9a800bd5ecb` (`R=12,288`) should receive further law-separated evaluation before its tuning is considered settled. The current `deepseek-learned` realization measured `1.1296x` GGTensile/HIP, with a primary 25-repeat log-time interval of `[1.1247x, 1.1345x]`, versus the documented `1.1594x`. The current `deepseek-hash` realization measured `1.2799x`, with a 75-repeat interval of `[1.2775x, 1.2822x]`. The spread is route-law sensitivity, not evidence of a single uniform kernel regression, but it is enough to keep B1 open for further tuning.
- IQ2_XXS B4 and B16 do not receive a retune flag from this pass. The learned/hash post-hoc B4 speedups are approximately `1.1692x`, and the B16 speedups are approximately `1.0610x`, close to the documented rows after the law components are separated.

### Reopen flag

- Reopen the B1 P0/X1 retention and transfer experiment with both DeepSeek laws on the current prequantized multiply-only surface. The documented B1 value came from a learned-law medoid campaign, while the current hash realization is materially faster and the current learned realization is slower. Re-run the retained X1 parent/candidate A/B independently for `deepseek-learned` and `deepseek-hash` before transferring the B1 result to a single law-agnostic tuning conclusion. This does not reopen X2-X4, which have independent timing closures.
- The B4 and B16 X1/J80 geometry conclusions are not reopened by this law split alone.
