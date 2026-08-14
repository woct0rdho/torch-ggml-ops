# GGTensile Grouped MMQ Forward Q2_K Experiment

## Purpose

Build and evaluate isolated gfx1151 wave32 GGTensile grouped Q2_K forward kernels for the non-paired DeepSeek routed-down production shapes `(12288,4096,2048)`, `(49152,4096,2048)`, and `(196608,4096,2048)`. Paired projection kernels are outside this campaign. The fitted DeepSeek learned/hash route prior is the primary speed metric, with learned and hash components reported separately.

This work is research-only. It does not change public dispatch, generated bundle tables, extension registration, packaging, or HIP fallback behavior. The retained grouped Q4_K and Q5_K generated sources at commits `31bc7ab` and `1050724` are frozen regression contracts.

## Contract

The packed expert bank is `[256,4096,672]` bytes per logical output row. Q2_K stores 256 values in an 84-byte block: sixteen scale/minimum bytes, 64 two-bit payload bytes, and FP16 `d`/`dmin`. `K=2048` therefore uses eight blocks per output row.

The installed Q8_1 `F16_D2S6` producer emits `[16,R,144]` bytes. Each 128-value plane contains two FP16 scales, six FP16 sums, and 128 signed int8 payload values. The first six 16-value groups consume stored sums; the last two groups reconstruct their integer sums during MMA. Output is contiguous BF16 `[R,4096]`.

Route metadata remains device-resident: int64 physical expert IDs and cumulative int32 row offsets, with at most 256 entries and final valid offset `R`. The multiply ABI is the existing 64-byte grouped contract. No host route readback, route descriptor cache, prepared weight representation, or dense shadow is permitted.

Every retained artifact targets gfx1151 code-object version 5 and wave32. It must have zero private storage, spills, scratch instructions, calls, and dynamic stack and must rebuild byte-identically. Companion setup, workspace, reduction, atomics, SplitK, or persistent mechanisms must expose and time their complete contract.

## Installed Controls

The exact non-paired installed controls are:
- `torch_ggml_ops_mmq_gfx1151_v1_grouped_fwd_serial_q2_k_n4096_k2048_j32`
- `torch_ggml_ops_mmq_gfx1151_v1_grouped_fwd_serial_q2_k_n4096_k2048_j32_j16`

Both launch a `64 x G` grid with a 128-thread workgroup and 30,336 dynamic LDS bytes. Pure J32 uses 208 VGPRs and 38 SGPRs. The mixed J32/J16 body uses 213 VGPRs and 43 SGPRs. Both have zero private storage and spills.

Public dispatch selects the mixed body at exact rows 49,152 or when `R < 64 * G`; otherwise it selects pure J32. The research launcher reproduces this policy from aggregate rows and route-entry count without reading route contents. The exact `R=49,152` exception is covered by a repository test.

## Initial Premise

Q2_K cannot reuse the Q4/Q5 arithmetic by renaming the format. Its weight groups are 16 values rather than 32, scale/minimum metadata is nibble-packed, and activation metadata is `F16_D2S6`. The implementation therefore uses a dedicated `GroupedQ2KDecodedWeightLdsLowering` while reusing only semantically matching routed ownership, raw Q8_1 plane staging, integer WMMA, BF16 stores, register planning, ROCISA, and toolchain components.

The lowerer cooperatively decodes Q2_K two-bit payloads into padded LDS rows, converts the nibble scale/minimum metadata, extracts the dynamic activation scale/sum fields, and executes two K-half stages. Groups 0-5 use stored activation sums. Groups 6-7 issue all-ones integer WMMAs and reconstruct the missing activation sums without changing the producer contract. The static path emits the 16 Q2 group bodies directly; the dynamic path remains a looped control for comparison.

A true 32-row decoded-LDS parent matches installed J32 ownership. Larger 64-row and 128-row parents were added only as research controls to test decode reuse against residency. None of these identities changes public dispatch.

The retention gate is bitwise installed agreement. Matching the independent dequantized BF16 error envelope is useful for diagnosis, but it does not waive exact installed parity.

## Qualification

- Validate exact problem/solution mappings for all three production rows and reject Q4_K/Q5_K cross-format identities.
- Build every retained production key twice and compare source and code bytes.
- Inspect ABI, target, wave size, workgroup, VGPRs, SGPRs, LDS, barriers, waits, WMMAs, private bytes, spills, scratch, calls, and dynamic stack.
- Exercise full, partial, boundary, sparse-ID, repeated-ID, and invalid routes against the installed dispatch.
- Require active input, packed-weight, and workspace mutations to affect output while inactive experts remain inert.
- Compare against an independently dequantized BF16 grouped reference using bounded output-column slices; do not allocate the giant `[256,4096,2048]` reference after the earlier GPU reset coincided with that path.
- Regenerate every retained grouped Q4_K and Q5_K production identity and require byte-identical source.

Screens use deterministic 512-draw search and confirmation banks, five weighted medoids per DeepSeek learned/hash component, three warmups, and nine alternating-order CUDA-event repeats. Competitive candidates require reversed-order 25-repeat confirmation. Learned and hash components cannot hide one another. Captured, synthetic, sequential, and complete-call controls follow only after confirmation-prior competitiveness.

## Mechanism Order

1. Establish strict Q2_K modeling, `F16_D2S6` runtime support, installed J32/mixed controls, and a typed decoded-LDS correctness body.
2. Measure 32-row and 64-row parents against the exact public installed dispatch on learned/hash confirmation medoids.
3. Remove scalar group-loop and address overhead with static 32-, 64-, and 128-row group lowering while retaining the finite baseline as a comparison control.
4. Test arithmetic association changes only behind the exactness gate and only time a finite variant.
5. Stop the family when decode/correction volume and residency remain slower than the installed body; do not introduce flattened tasks, persistence, SplitK, or prepared decode without a changed utilization or model-owned lifetime prerequisite.

## Experiment Log

### Campaign opened

The production inventory, public selector, Q2_K packed layout, `F16_D2S6` quantizer, rolled installed arithmetic, prior contract, and non-paired DeepSeek model tensor were audited. The exact aggregate rows are `12,288`, `49,152`, and `196,608`; the packed bank/workspace/output shapes are `[256,4096,672]`, `[16,R,144]`, and `[R,4096]`. Paired kernels are excluded.

### Typed Q2_K control and exactness diagnostics

The first decoded-LDS control used a 32-row parent with dynamic group lowering. It built with `135 VGPRs`, `40 SGPRs`, `25,600` LDS bytes, 12 static WMMA instructions, and four barriers. A true 64-row dynamic parent used `159/40/30,208/24/4`. Both were finite and had zero private storage and spills.

The initial qualification timeout coincided with a gfxhub null-address page fault, SQC activity, queue-removal failure, and GPU reset. Subsequent bounded checks showed that invalid-route, one-tile, all-column, four-route, and full `(64,4)` launches retired correctly. The independent reference was changed to bounded output slices; no giant dequantized reference allocation is part of the retained procedure.

For the R35 route matrix, candidate and installed outputs were finite. The candidate matched the installed output on the first 64 columns in the bounded single-group comparison, while full output remained non-bitwise, generally by one BF16 step: the observed installed delta was `0.00390625` for a single group and up to `0.0078125` for mixed routes. The independent BF16 reference showed candidate maximum errors of about `0.0410` to `0.046875` and RMS errors around `0.0123` across uniform, boundary, skewed, repeated-ID, and sparse-ID routes. Active packed-weight, input, and workspace mutations changed output, an inactive expert remained bitwise inert, and the invalid route left a sentinel output untouched. These results qualify the body as finite and reference-consistent, not as bitwise-installed exact.

### Dynamic parents

The dynamic 32-row parent measured combined installed-over-candidate ratios of `0.7476x`, `0.7647x`, and `0.8227x` at B1, B4, and B16. The dynamic 64-row parent measured `0.6014x`, `0.7840x`, and `0.8949x`; the B16 aggregate included a large outlier, while regular profiles were about `0.883x`. Both parents were rejected because the installed body was faster on every production shape.

The instruction comparison explained the result. The installed pure J32 body has about 3,844 instructions, 126 VMEM instructions, 212 DS instructions, 2,748 non-WMMA VALU instructions, 48 WMMAs, and 150 waits. The dynamic Q2 body pays for two-bit decode, nibble correction, activation metadata handling, and group-loop work without recovering enough reuse.

### Static group lowering

Static group lowering removed the scalar Q2 group loop and selected direct group bodies through `Q2ScaleMinimumNibbleUnrolled`. It was corrected after the first attempt used `tile * 144` rather than `16 * tile * 144` for activation tile strides; the corrected path passed the bounded independent reference and route matrix.

The unrolled 32-row parent used `135/40/25,600/40/4` and measured `0.7998x`, `0.8229x`, and `0.8766x` at B1/B4/B16. Its instruction makeup was 2,255 total instructions, 62 VMEM, 236 DS, 1,720 non-WMMA VALU, 40 WMMAs, and 73 waits. It was the best B1-family candidate but remained about 20 percent slower at B1 and was not bitwise exact.

The unrolled 64-row parent used `159/40/30,208/80/4` and measured `0.6387x`, `0.8217x`, and `0.9306x`. Its instruction makeup was 3,895 total instructions, 114 VMEM, 340 DS, 3,090 non-WMMA VALU, 80 WMMAs, and 105 waits. It approached parity only at B16 and regressed badly at B1.

The unrolled 128-row parent used `239/40/39,424/160/4` and measured `0.3797x`, `0.7089x`, and `0.8798x`. The larger ownership reduces useful residency too sharply; it is rejected for all production selection purposes. The complete 128-row confirmation finished without leaving a benchmark process behind.

Static unrolling is therefore a real scalar-overhead reduction, but the dominant Q2 decode and correction instruction volume remains. Lowering the group loop alone does not overcome the installed control.

### Arithmetic association probe

A finite source-order mixed-FMA variant was tested after a previous diagnostic rewrite incorrectly fed full FP32 WMMA sums into FP16 mixed-FMA operands and produced `98,304` NaNs. The corrected probe passed the device finiteness and reference checks, but its confirmation ratios were only `0.7476x`, `0.7647x`, and `0.8229x`, versus the retained unrolled-32 baseline's `0.7998x`, `0.8229x`, and `0.8766x`. The source-order variant was reverted. Its rejection evidence is retained here rather than weakening the exactness gate.

### Deterministic final artifact set

The final research control set was rebuilt twice for all three production rows. The following resource identities were byte-identical across both builds, with zero private bytes and zero VGPR/SGPR spills:

| identity | VGPR | SGPR | LDS bytes | WMMAs | barriers | source/code identity |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| unrolled32 | 135 | 40 | 25,600 | 40 | 4 | identical for R12,288/R49,152/R196,608 on repeated builds |
| unrolled64 | 159 | 40 | 30,208 | 80 | 4 | identical for R12,288/R49,152/R196,608 on repeated builds |
| unrolled128 | 239 | 40 | 39,424 | 160 | 4 | identical for R12,288/R49,152/R196,608 on repeated builds |

The exact source and code SHA-256 values are recorded in `/tmp/ggtensile-grouped-q2-final-report.json`; the report also records the complete per-row resource inspection. The production artifact builds use the same problem identities as the three aggregate-row keys and do not alter public packaging.

## Outcome

No Q2_K candidate satisfies both retention requirements. The decoded-LDS family is finite, independently reference-consistent, route-safe for the exercised matrix, resource-clean, and deterministic, but every candidate is slower than the installed dispatch and full output is not bitwise identical to the installed body. The independent BF16 envelope is not a substitute for installed parity.

The Q2 model, dedicated lowerer, physical planning, inspection, validation, runtime launcher, and focused tests remain as research-only controls because they document the contract and provide reproducible rejection evidence. No Q2 selector or public bundle change is made. The unrolled 32-row and 64-row identities are useful retained controls for future work; the 128-row identity is retained only as a measured rejection control. Further optimization is deferred until a new mechanism can reduce Q2 decode/correction cost or establish a changed lifetime/utilization prerequisite without weakening bitwise parity.

The focused grouped forward file passes 36 tests, and the combined grouped-plus-dense generator suite passes 168 tests. Coverage includes the three exact production shapes, derived tensor/grid identities, F16_D2S6 and static-lowering source contracts, strict artifact inspection, deterministic rebuilds, exact installed dispatch boundaries, and cross-format rejection. The device route matrix covers uniform, boundary, skewed, repeated-ID, sparse-ID, and invalid routes with bounded independent-reference and active/inactive mutation checks.

## Final Classification

- Retained: grouped-only Q2_K format semantics, exact problem and solution identities, dedicated decoded-LDS lowering, F16_D2S6 research runtime support, static 32/64 controls, deterministic artifact inspection, bounded reference qualification, and the installed-selector control launchers.
- Rejected: dynamic 32/64 parents, unrolled 128-row ownership, source-order mixed-FMA association, and every public selector proposal from this decoded-LDS family. Each is slower than installed, non-bitwise, or both.
- Deferred behind a changed prerequisite: prepared decode requires a model-owned representation lifetime; persistence or flattened tasks require evidence of unresolved route/launch imbalance; SplitK requires an explicit reduction and complete-call win. TensileLite and Composable Kernel generator restrictions remain evidence about generator support, not gfx1151 hardware prohibitions.
- Contract-incompatible: paired-kernel scope, host route readback, public dispatch or package integration, hidden dense shadows, and unbounded full-bank dequantized references.
- Actionable after the first recursive pass: isolate Q2_K from the shared dense quant-format registry and rerun final active/inactive mutation gates. Both were completed, after which no additional in-contract mechanism had a measured premise strong enough to reopen implementation.

## Recursive Final Review

Before completion, reread this record, grouped Q4_K/Q5_K records, dense and grouped forward/backward histories, installed Q2_K source and normalized ISA, every generated artifact and report, relevant TensileLite/Composable Kernel mechanisms, and gfx1151 instruction constraints. Classify each remaining idea as retained, rejected, deferred behind a changed prerequisite, contract-incompatible, or actionable. Implement every actionable in-contract mechanism and repeat the review.
