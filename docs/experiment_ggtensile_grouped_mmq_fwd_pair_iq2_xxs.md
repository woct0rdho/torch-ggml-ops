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

### Final retained throughput

The public pair is the installed HIP control. Effective TFLOPS is nominal dense-equivalent complete-call throughput, calculated as `4 * rows * N * K / seconds`: two FLOPs per FMA across both projections. B1 and B4 use the transfer confirmation above; B16 uses its dedicated 25-repeat confirmation.

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
