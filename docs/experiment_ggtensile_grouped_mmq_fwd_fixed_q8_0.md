# GGTensile Fixed-Group MMQ Forward Q8_0 Experiment

## Purpose

Implement and optimize a GGTensile assembly kernel for the DeepSeek fixed-group Q8_0 output-A projection on gfx1151. The operation has eight independent fixed weight groups, a shared Q8_1 F32_D4 activation workspace, and BF16 output. This is a separate ABI and ownership problem from routed grouped MMQ and from dense MMQ: one workgroup owns an output-feature tile, a token tile, and one fixed group.

The campaign record is updated after each coherent implementation, correctness, resource, timing, rejection, or review result. Durable scripts, reports, and artifacts live below `~/tmp/torch-ggml-ops/`.

## Contract

Target only:
- gfx1151, code-object version 5, wave32, WMMA V1, and the six-argument fixed grouped ABI.
- Input BF16 with shape `[tokens, 8, 4096]`; upstream Q8_1 F32_D4 workspace with one quantized row per `(token, group)`.
- Eight packed Q8_0 weight groups, each with logical shape `[out_features, 4096]` and packed row stride `4352` bytes.
- BF16 output `[tokens, 8, out_features]` in the existing fixed output layout.
- Required token counts `2048`, `8192`, and `32768`, and the production output-A shape `out_features=1024, K=4096`.
- Exact arithmetic and BF16 rounding compatible with the authoritative HIP fixed-group kernel and the existing public operator.
- Deterministic source and artifact generation, strict ABI/resource inspection, independent packed-weight correctness, mutation checks, and warmed complete and prequantized timing.

The fixed kernel ABI is:

```text
packed_weight, activations, output, tokens, out_features, bytes_per_group
```

Its launch is `(ceil(out_features/64), ceil(tokens/64), 8)` with workgroup `(32,4,1)`. The upstream quantizer and complete-call allocation remain outside the assembly writer but inside complete-call timing.

Q8_0 stores one FP16 scale and 32 signed int8 values in 34 bytes. The dot is signed integer WMMA followed by the established FP32 correction:

```text
fp32_output += int32_dot * fp32(q8_0_scale) * fp32(q8_1_activation_scale)
```

No routing metadata, expert-index lookup, CPU descriptor, hidden copy, split-K, prepared weight, dense shadow, or public dispatch rewrite is part of the first candidate. Unsupported shapes must use the existing HIP path.

## Controls And Baseline

The authoritative performance control is `fixed_grouped_q8_0_mmq_bf16_body` in `csrc/mmq_core.cuh`, launched by `launch_fixed_grouped_forward` in `csrc/mmq_bundle.cpp`. It uses the same Q8_1 producer and packed GGUF weights as the GGTensile path. The public `fixed_grouped_mmq` operator is the complete-call control.

Before retaining a candidate, collect same-process HIP and public timings. Keep multiply-only and complete-call measurements separate. Complete-call timing includes activation quantization, workspace allocation, kernel launch setup, and the fixed grouped multiply. Never use `.item()`, host route/workspace metadata, implicit synchronization, or a CPU descriptor in a timed path.

Existing ordinary Q8_0, Q3_K, Q4_K, Q5_K, and Q6_K writer streams are a regression boundary. Changes to shared mechanisms require source and behavioral qualification; unrelated generated identities are not to be pinned in this document.

## Design Plan

- Add a typed fixed-group Q8_0 problem/solution contract with explicit `(tokens, N, K, groups)`, six-argument ABI, grid, packed group stride, and output addressing.
- Reuse signed-int8 arithmetic roles only where their data contract matches; keep fixed-group grid and addressing in a dedicated lowering.
- Add deterministic physical ownership and resource derivation for the fixed candidate. Retained gfx1151 artifacts require wave32, zero private storage, no spills, no scratch, no calls, and no dynamic stack.
- Build the smallest complete candidate, qualify exact output equality against HIP/public and an independent GGUF dequantized reference, then screen ownership and LDS variants.
- Promote only after two independent warmed confirmations on each required token count and no regression for a shape sharing the mechanism. Public dispatch, generated bundle registration, packaging, and fallback changes are deferred until the research artifact has final evidence.

The likely high-value dataflow is HIP-shaped cooperative LDS reuse: activation rows are shared across output columns, while each fixed group selects a disjoint packed-weight base through `z`. A candidate that merely launches one independent projection per workgroup or reloads activations for each output tile is not considered fused fixed-group work.

## Validation Gates

Every candidate must pass, in this order:
- strict fixed ABI and exact-shape validation;
- deterministic source generation and successful gfx1151 assembly/link;
- code-object v5, wave32, zero-spill resource inspection;
- bitwise equality with HIP and the public operator on all required token sizes;
- finite output, independent Q8_0 reference, input/weight/workspace mutation, and repeated-producer checks;
- complete-call and prequantized warmed timing against HIP and the retained parent;
- focused tests, full tests, lint/type/format/compile checks, and diff checks.

Any correctness, ABI, compiler, runtime, executable, or documentation finding that affects the contract restarts qualification. Failed candidates remain recorded with their artifact paths and are removed from active lowering unless a later premise explicitly reopens them.

## Completion Record

- Initial fixed-group contract boundary and ABI recorded.
- Added the typed fixed-group model, validation boundary, derived state, physical-plan wrapper, six-argument assembly writer, direct prequantized runtime, and artifact inspection in the fixed-group modules under `tools/ggtensile/`.
- Candidate P0 uses the existing signed-int8 Q8 small-M LDS mechanism with a dedicated fixed lowering. It emits code-object v5 metadata for a 40-byte six-argument kernarg segment, wave32, `(32,4,1)` workgroups, 144 VGPRs, 16 SGPRs, 28,672 fixed LDS bytes, 32 static signed-int8 WMMAs, two barriers, zero private storage, and zero register spills.
- P0 first assembled and linked successfully, but the first device attempt exposed three address-contract bugs. The output store initially used `s8` (the activation pointer low word) instead of `s13` (`out_features`); the lowering then loaded none of the scalar values at `s12:s15`; and the first flattened store used dense-row spacing rather than eight-group spacing. These failures are retained as review evidence, and the invalid artifact was not retained as a candidate.
- After loading the scalar kernargs and applying `((tile_token + dense_row) * 8 + group)` output addressing, P0 at `tokens=2048`, `N=1024`, and `K=4096` is bitwise identical to the public `fixed_grouped_mmq` control. The independent GGUF Q8_0 dequantized reference measured normalized RMSE `0.00605183` and maximum BF16 absolute error `0.0625`. Required-token qualification and timing remain open.
- Added focused model, rejection, source-dataflow, deterministic-build, and resource-inspection tests. The fixed-group test module passes all six tests; the artifact inspector independently confirms the 40-byte ABI, 32 WMMAs, two barriers, fixed LDS size, and prohibited-resource zeros.
- P0 passed bitwise public-control qualification at all required token counts: `2048` (`exact token count 2048`), `8192` (`8192`), and `32768` (`32768`), with `N=1024` and `K=4096`. Candidate and public outputs had equal shape, equal BF16 values, and zero maximum difference at each size.
- The `2048` qualification also passed input, packed-weight, and activation workspace mutation checks, and two independent Q8_1 F32_D4 producer launches produced identical workspaces. The authoritative packed source was `blk.0.attn_output_a.weight`; no synthetic quantized bytes were used.
- The first order-rotated timing screen used three warmups and nine repeats per size. P0 prequantized multiply medians were `4.046`, `17.562`, and `71.585 ms` at `2048`, `8192`, and `32768` tokens; the installed HIP control was `10.069`, `40.002`, and `159.427 ms`, giving candidate/control ratios `0.4019`, `0.4390`, and `0.4490`. Manual complete-call medians, including quantization and workspace allocation, were `4.982`, `21.428`, and `87.034 ms`; the public complete-call control was `11.045`, `43.676`, and `174.564 ms`, giving ratios `0.4511`, `0.4906`, and `0.4986`.
- Screen reports are durable under `~/tmp/torch-ggml-ops/fixed-grouped-q8-candidate/`. Confirmation repeats and the final retain/reject decision remain open.
- The first 25-repeat confirmation used seven warmups and seed `9012`. P0 prequantized medians were `4.145`, `17.694`, and `71.542 ms` versus HIP `10.188`, `40.073`, and `159.694 ms`, with ratios `0.4069`, `0.4415`, and `0.4480`. Complete-call medians were `5.004`, `21.397`, and `86.326 ms` versus public `11.054`, `43.737`, and `174.552 ms`, with ratios `0.4527`, `0.4892`, and `0.4946`.
- A second independent correctness run at seed `9013` remained bitwise exact at all three token counts and repeated the no-mutation and producer-repeat checks. Its second 25-repeat confirmation produced prequantized ratios `0.4045`, `0.4387`, and `0.4501`, and complete-call ratios `0.4514`, `0.4897`, and `0.4870` for `2048/8192/32768` tokens.
- P0 is retained as the qualified research candidate for this campaign. Public dispatch, generated bundle registration, packaging, fallback selection, and prepared-weight integration remain deliberately deferred.
- Repository verification passed: `546` tests, Ruff check and format, `ty check`, compileall, diff checks, and all pre-commit hooks.
