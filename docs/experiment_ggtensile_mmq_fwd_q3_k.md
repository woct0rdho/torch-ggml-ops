# GGTensile MMQ Forward Q3_K Experiment

## Purpose

This record covers the first isolated GGTensile MMQ forward Q3_K control on gfx1151. The control validates the Q3_K packed decoder, Q8_1 `F32_D4` workspace contract, wave32 WMMA mapping, and row-major BF16 output ownership on one exact ordinary shape. It is research evidence only. It does not add a Q3_K inventory, selected catalog, runtime dispatch entry, or public bundle kernel.

HIP remains the correctness and timing control. The existing public HIP path remains authoritative for production selection.

## Exact Scope

The operation is:

```text
output[M,N] = input[M,K] @ dequant(weight[N,K]).T
```

The first exact control is:

```text
(M,N,K) = (2048, 4096, 2048)
tensor = blk.4.attn_gate.weight
logical weight = (4096, 2048)
physical GGUF weight = (4096, 880)
```

Q3_K uses 110-byte blocks containing `hmask[32]`, `qs[64]`, `scales[12]`, and FP16 `d`. The activation producer is the fixed HIP Q8_1 `F32_D4` workspace with shape `(16, 2048, 144)`. No prepared weights, dense shadow, external decode storage, split-K reduction, persistent workgroup, grouped path, inventory coverage, or public integration is in scope.

The control uses workgroup `(32,4,1)`, a `128x64` macro tile, depth `16`, four wave32 waves, and 128 integer WMMA operations. It cooperatively stages activation rows and directly decodes signed Q3 payload rows into LDS before scale-corrected WMMA accumulation and BF16 RNE stores.

## Implementation

The Q3 backend adds typed semantic support for:
- Q3 block planes and physical byte sizing.
- Signed 3-bit reconstruction from the low 2-bit payload and high mask.
- Packed signed six-bit scale fields across all 16 scale groups.
- FP16 block factor `d` and FP32 scale correction.
- Q8_1 `F32_D4` activation addressing and scale use.
- Formula-derived Q3 LDS layout and register resources.
- An independent Q3 dequantization and matmul reference.

The isolated writer uses explicit Q3 register roles and deterministic lifetime-aware first-fit allocation. Its formula-derived candidate resources are 104 VGPRs, 16 SGPRs, and 28,672 bytes of LDS, with zero private storage and zero spills.

The initial device mismatch exposed two separate scale/control issues. First, encoded Q3 scale value `33` was lowered through signed six-bit extraction as `-31`; Q3 scale values are unsigned six-bit codes offset by 32, so the writer now subtracts 32 and converts the result as signed FP32. Second, the direct HIP test launcher initially allocated 38,400 bytes of LDS for Q3. The HIP Q3 shared layout requires 40,448 bytes from its 36-dword activation tile and 84-dword packed Q3 row stride. Correcting that control allocation removed the HIP-only tail-column corruption.

## Correctness and Resources

The final 25-repeat qualification run passed:
- Candidate versus HIP multiply: 0 differing BF16 elements out of 8,388,608; normalized RMSE `0.0`; maximum absolute error `0.0`.
- Candidate versus public complete path: 0 differing BF16 elements; normalized RMSE `0.0`.
- Candidate versus independent reference: normalized RMSE `0.006061011`, maximum absolute error `0.0390625`.
- Producer repeat: 0 differing bytes out of 4,718,592.
- Input mutation: 8,388,608 changed output elements.
- Packed-weight mutation: 2,007 changed output elements.
- Workspace mutation: 548 changed output elements.
- All outputs were finite.

Strict code-object inspection accepted code-object version 5, target gfx1151, wave32, the 40-byte kernarg ABI, 104 VGPRs, 16 SGPRs, 28,672 bytes of LDS, zero private bytes, and zero VGPR/SGPR spills. Static counts were 128 WMMAs, 4 barriers, 4,919 VALU issues, 88 VMEM operations, 434 LDS operations, 151 waits, and 8 clauses.

The accepted artifact is:

```text
kernel = torch_ggml_ops_ggtensile_gfx1151_v1_mmq_fwd_q3_k_m2048_n4096_k2048_31c9031538e012be
solution = ggsol_31c9031538e012be
source SHA256 = a761a25db73fea8b6d360f5550ca034a5e6089c270ae2598bc7c234df6e7c9eb
object SHA256 = 790dc13eb4b34dff38a1ad374b5da3a6de5be80935a3b311a6a19759e1e1992d
HSACO SHA256 = 571ac3eeaf7432c7ab910bf62188205fea7786f700713dbc6b2d2017eaaabf4c
```

Two independent CLI generate, assemble, build, and inspect roots reproduced the source, object, HSACO, and normalized inspection content byte-for-byte.

## Timing and Decision

The authoritative warmed rotating run used five warmup launches and 25 samples. Speedup is HIP median divided by GGTensile median; values below `1.0x` would favor GGTensile.

| Path | HIP median | GGTensile median | GGTensile/HIP time |
| --- | ---: | ---: | ---: |
| Multiply only | `1.417212 ms` | `2.594226 ms` | `1.8305x` |
| Complete call | `1.456055 ms` | `2.602045 ms` | `1.7871x` |

The control is correctness-qualified and resource-clean, but it is substantially slower than HIP. It is rejected for promotion and remains isolated measured research evidence. No additional Q3 shape, geometry, schedule variant, inventory entry, selected solution, dispatch rule, or public bundle entry is authorized by this result.

## Verification

The focused Q3/reference/spec/writer suite passes with 179 tests when run with the forward writer coverage suite. The full repository suite passes with 369 tests and the existing 14 warnings. Ruff, formatting, `ty`, compileall, pre-commit, and `git diff --check` pass. Regeneration of the frozen pre-Q8 catalog sources produces 447 expected and generated sources with zero missing, added, or changed files. The public gfx1151 bundle remains current at 179 kernels.

This is the first Q3 forward control, not a completed Q3 campaign. Expansion requires a new exact-shape premise and fresh correctness, resource, deterministic-build, and warmed timing evidence.
