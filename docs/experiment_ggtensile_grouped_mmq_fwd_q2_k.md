# GGTensile Grouped MMQ Forward Q2_K Experiment

## Purpose

Build and evaluate isolated gfx1151 wave32 GGTensile grouped Q2_K forward kernels for the non-paired DeepSeek routed-down production shapes `(12288,4096,2048)`, `(49152,4096,2048)`, and `(196608,4096,2048)`. Paired projection kernels are outside this campaign. The fitted DeepSeek learned/hash route prior is the primary speed metric, with learned and hash components reported separately.

This work is research-only. It does not change public dispatch, generated bundle tables, extension registration, packaging, or HIP fallback behavior. The retained grouped Q4_K and Q5_K generated sources are frozen regression contracts.

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

## Reopened Mechanism Order

- Treat the installed HIP J32 and mixed J32/J16 kernels as constructive implementations of the required performance, and derive their ownership, packed LDS layout, instruction families, loop bodies, arithmetic association, wait/barrier placement, and tail policy from source plus normalized gfx1151 ISA.
- Attack the largest gap first by replacing repeated fully decoded-weight staging with an installed-structured packed Q2_K path. Reuse the existing typed routed ABI and physical planning, but match the HIP kernel's bounded rolled dot-product topology and exact correction order closely enough to recover bitwise output.
- Build true J32 and mixed J32/J16 generated identities and reproduce the installed selector, including the exact `R=49,152` exception. Qualify each mechanism on R35 correctness and strict artifact inspection before any timing.
- Screen fitted-prior B1/B4/B16 multiply timing after every coherent structural change. Prefer mechanisms that improve the largest deficits; revert failed bodies after recording their rejection evidence.
- Once parity is reached, optimize packed loads, LDS layout, address generation, waits, barriers, instruction pairing, accumulator initialization, and route-sensitive ownership. Confirm competitive candidates with reversed-order 25-repeat, search, captured, synthetic, sequential, and complete-call controls.
- Continue until no actionable in-contract mechanism remains. Public dispatch and packaging stay unchanged unless separately requested; research selection may combine generated identities by exact production shape and route count.

## Experiment Log

### Campaign opened

The production inventory, public selector, Q2_K packed layout, `F16_D2S6` quantizer, rolled installed arithmetic, prior contract, and non-paired DeepSeek model tensor were audited. The exact aggregate rows are `12,288`, `49,152`, and `196,608`; the packed bank/workspace/output shapes are `[256,4096,672]`, `[16,R,144]`, and `[R,4096]`. Paired kernels are excluded.

### Typed Q2_K control and exactness diagnostics

The first decoded-LDS control used a 32-row parent with dynamic group lowering. It built with `135 VGPRs`, `40 SGPRs`, `25,600` LDS bytes, 12 static WMMA instructions, and four barriers. A true 64-row dynamic parent used `159/40/30,208/24/4`. Both were finite and had zero private storage and spills.

The initial qualification timeout coincided with a gfxhub null-address page fault, SQC activity, queue-removal failure, and GPU reset. Subsequent bounded checks showed that invalid-route, one-tile, all-column, four-route, and full `(64,4)` launches retired correctly. The independent reference was changed to bounded output slices; no giant dequantized reference allocation is part of the retained procedure.

For the R35 route matrix, candidate and installed outputs were finite. The candidate matched the installed output on the first 64 columns in the bounded single-group comparison, while full output remained non-bitwise, generally by one BF16 step: the observed installed delta was `0.00390625` for a single group and up to `0.0078125` for mixed routes. The independent BF16 reference showed candidate maximum errors of about `0.0410` to `0.046875` and RMS errors around `0.0123` across uniform, boundary, skewed, repeated-ID, and sparse-ID routes. Active packed-weight, input, and workspace mutations changed output, an inactive expert remained bitwise inert, and the invalid route left a sentinel output untouched. These results qualify the body as finite and reference-consistent, not as bitwise-installed exact.

### Installed arithmetic association probe

The first reopened probe added distinct 32- and 64-row HIP-association identities. Stored groups now compute `Cd*d`, apply the missing-sum term inside the temporary for groups 6-7, multiply the combined value by `dB`, and apply the `dmin*sB` correction in the same order as the installed rolled HIP body. The previous fully decoded identities remain unchanged.

The 32-row HIP-association body is bitwise exact against installed on the R35 single-group check and on uniform, boundary, skewed, repeated-ID, sparse-ID, mutation, and invalid-route checks. Its nine-repeat fitted-prior ratios were `0.7138x`, `0.7288x`, and `0.7866x` at B1/B4/B16, so arithmetic association alone is rejected as a speed mechanism but retained as the exactness control for structural work. No complete-call timing was run because it failed the multiply competitiveness gate.

The next large-margin target is the HIP loader topology: its 128 threads distribute one Q2 scale-pair conversion per output-row lane while the current generated body performs all 16 metadata conversions serially per producer lane. Its packed source stride is 400 bytes per decoded output row, with 256 payload bytes, 64 metadata bytes, and the same 64-row maximum ownership used by the installed J32 body. This provides a measured structural premise for the next implementation.

### HIP packed producer topology

A second reopened control reproduced the installed J32 LDS geometry and producer lane mapping: activation storage began at byte 128, the decoded 64-row weight tile began at byte 4,736, decoded rows used a 400-byte stride, and each lane converted one metadata group over eight output rows. The body used `135 VGPRs`, `40 SGPRs`, `30,336` LDS bytes, 40 static WMMAs, four barriers, zero private storage, and zero spills. It was bitwise exact across the complete R35 route and mutation matrix.

The explicit implementation measured only `0.6048x`, `0.6432x`, and `0.6763x` at B1/B4/B16. Its eight producer iterations each ended in `vmcnt(0)`, and each metadata lane reloaded the common `d/dmin` pair. This demonstrates that installed LDS geometry and lane mapping are not sufficient without the HIP compiler's multi-load software pipeline. The body is rejected and removed after recording this evidence. The next large-margin target is the installed body's four-group overlap: its roughly 208-VGPR allocation keeps multiple WMMA results, LDS reads, and corrections live, whereas the exact generated control uses only 135 VGPRs and resolves each group before issuing the next.

### Two-group arithmetic pipeline

An exact two-group control used the post-decode register lifetime to keep a second stored-sum group's weights, activations, metadata, and WMMA results live. It built at `175 VGPRs`, `40 SGPRs`, `25,600` LDS bytes, 40 WMMAs, four barriers, zero private storage, and zero spills, and passed the complete R35 route and mutation matrix bitwise.

Paired candidate/installed timing measured `0.7130x`, `0.7320x`, and `0.7792x` at B1/B4/B16. One B4 installed-control sample rose to 116.4 ms while adjacent controls remained near 66 ms; the fitted aggregate remained noncompetitive, and the unaffected B1/B16 results independently reject the mechanism. The extra live group did not hide the dominant work and slightly regressed B1/B16 relative to the exact 135-VGPR control. This body is rejected and removed. Partial LDS retirement, rather than additional group lifetime alone, is the next actionable installed-ISA mechanism.

### Partial LDS retirement

The local AMD LLVM wait-count implementation documents that register dependencies permit nonzero LDS waits when one event timeline is active. A dedicated exact schedule therefore stages activation scale/sum reads first, then WMMA payloads, then four `d/dmin` reads; `lgkmcnt(4)` releases the payload dependencies while those four correction reads remain pending. WMMA issuance and activation metadata extraction overlap their retirement, followed by `lgkmcnt(0)` immediately before the first dependent correction.

The body remained `135/40/25,600/40/4`, with zero private storage and spills, and passed the complete R35 route and mutation matrix bitwise. Paired candidate/installed timing improved to `0.7270x`, `0.7337x`, and `0.7922x` at B1/B4/B16, versus `0.7138x`, `0.7288x`, and `0.7866x` for the exact full-wait control. This is a real but insufficient latency reduction and is retained as the new exact structural parent. The next compiler-derived mechanism is bounded multi-group LDS issue with descending partial waits, matching the HIP source's `#pragma unroll 4` dependency graph without copying external assembly.

### Two-group partial-LDS dependency graph

Optimized HIP MIR showed a two-group unit with six payload reads, eight trailing correction reads, four WMMAs, and descending nonzero LDS waits. A typed generated control reproduced that dependency graph for stored-sum group pairs while preserving the exact mixed-FMA association. It used 40 additional transient VGPRs and built at `175/40/25,600/40/4`, with zero private storage and spills. The complete R35 route, mutation, invalid-row, and bounded-reference matrix remained bitwise exact against installed output.

The fitted-prior ratios were `0.7186x`, `0.7237x`, and `0.7780x` at B1/B4/B16. All three regress the single-group partial-LDS parent. The paired issue pattern therefore does not amortize its larger live state in this decoded-LDS body; it is rejected and removed. The compiler evidence remains useful, but additional group lifetime is now rejected both with full waits and with descending partial waits.

The same partial-retirement schedule was also screened with the existing 64-row ownership. It built at `159/40/30,208/80/4`, remained bitwise exact, and passed the full route/mutation matrix. Its fitted-prior ratios were `0.5861x`, `0.7288x`, and `0.8224x` at B1/B4/B16. The larger parent is substantially worse at B1 and remains below the earlier non-association 64-row B16 control (`0.9306x`), so partial waits do not rescue 64-row ownership. This identity is rejected; the exact 32-row partial-LDS body remains the structural parent.

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

Independent builds of the final production artifacts are byte-identical and retain complete per-row resource inspection. The production artifact builds use the same problem identities as the three aggregate-row keys and do not alter public packaging.

### Reopened tail and producer controls

The first true mixed J32/J16 generated identity exposed a fractional activation staging contract at 16 rows: `16 * 144` bytes is 4.5 dwords per 128-thread lane. The shared staging helper now rounds fractional Q2 tiles up and masks the final load/store; exact 32-row tiles retain their original full path. The mixed identity passed the complete R35 route and mutation matrix bitwise, including routes ending at 16 rows and shorter tails. It uses `135/40/25,600/60/4`, with zero private storage and spills. Its confirmation-prior medians were `23.1645/81.9734/315.7009 ms` at B1/B4/B16, versus installed `18.9023/64.0556/259.4599 ms`, or `0.8160x/0.7814x/0.8219x`. Mixed tail ownership is retained as an exact structural control, not as a competitive result.

The arithmetic and producer micro-optimizations were then composed in order. Pre-negating the packed `dmin` value once per output row removed repeated sign multiplies and remained exact. Pairing decoded payload stores with `ds_write2_b32`, then pairing the 16 scale/minimum metadata stores, reduced the generated producer to 2,161 instructions while preserving `135/40/25,600/40/4` resources and the full mutation matrix. The Meta2 mixed parent measured the timings above. A combined VOPD accumulator-initialization identity remained exact and resource-clean, but measured `23.2115/82.4876/315.7949 ms`; the small initialization change was neutral to regressive against the Meta2 parent and is rejected as an independent mechanism.

### Distributed installed-style producer

The installed Q2 loader maps one scale/minimum pair across each output-row lane and rolls eight rows through the producer. The generated decoded-LDS producer now queues eight payload, scale-byte, and `d/dmin` loads per lane across all four waves, retires them with descending `vmcnt` waits, converts each row once, and writes the same compact 320-byte LDS rows. A temporary register-state error initially clobbered the persistent all-ones operands for missing-sum groups; moving conversion temporaries out of `v28:v31` restored exactness. An independent LDS dump then matched the Meta2 producer byte-for-byte for all 64 decoded rows.

The distributed body passed the complete route, input, workspace, active-weight, inactive-weight, and invalid-output matrix at `135/40/25,600/60/4`. Its first paired screen measured `22.7202/80.7077/311.1496 ms` at B1/B4/B16, or `0.8316x/0.7923x/0.8290x` of installed. Pairing metadata rows with `ds_write2st64_b32` preserved exactness and gave a small follow-up screen of `22.6840/80.5222/311.0248 ms`; the controls moved between runs, but the B4/B16 improvement was directionally consistent. This distributed producer plus metadata pairing is retained as the current exact structural parent, although it remains materially slower than installed.

### LDS stride and compact prefetch screens

The installed 400-byte decoded-row stride was screened independently from its original serialized producer. It remained exact at `30,720 B` LDS but regressed to `23.5724/83.3991/321.4185 ms`. A 336-byte stride with the same metadata bank separation and lower `26,624 B` LDS was also exact and regressed to `23.3843/82.7590/318.8301 ms`. Both padding identities are rejected; the compact 320-byte layout remains faster in this generated consumer.

A compact consumer prefetch used the otherwise free `v98:v121` window to issue the next stored-sum group's LDS reads while correcting the current group, then reused the normal C fragments for the next WMMAs. It passed the complete exactness matrix at `135/40/25,600/60/4`. The confirmation-prior medians were `22.6712/80.7907/310.3967 ms`, versus installed `19.0170/64.3893/260.9008 ms`; it was neutral at B1, regressed at B4, and only marginally lower at B16 relative to the distributed parent. It is rejected as a broad mechanism, and its source identity has been removed after preserving this evidence.

The installed producer also uses explicit hard clauses around clustered VMEM loads. Adding `s_clause 2` to each contiguous payload/scale/`dmin` load triple preserved bitwise output and the `135/40/25,600/60/4` resource identity. Its confirmation-prior medians were `22.6958/80.9443/310.9456 ms`, with stable installed controls at `19.0673/65.1310/260.7494 ms`. The result was neutral at B1/B16 and about 0.5 percent slower at B4 than the unclausified distributed parent, so standalone producer clausing is rejected and removed.

The matching activation experiment added `s_clause 7` around each eight-load portion of the exact 32-row Q8_1 stage; the exec-masked 16-row tail was unchanged. It also remained exact and resource-identical, but measured `22.7880/80.8765/311.0833 ms`. B1/B4 regressed and B16 was neutral relative to the unclausified parent. Activation hard clauses are therefore rejected and removed independently of producer clauses.

The generated epilogue uses priority 2 while installed HIP emits no `s_setprio`. A priority-zero mixed identity remained exact and resource-identical but measured `22.7772/80.9481/310.9089 ms`. It regressed B1/B4 and changed B16 only within run noise, so removal of epilogue priority is rejected and the prior priority-2 schedule is restored.

Increasing BF16 conversion dependency width from two to four was also exact and resource-identical. It measured `22.7315/81.4743/311.8031 ms`, regressing B4/B16 and providing no credible B1 gain. Width four is rejected and the width-two epilogue is restored.

Reducing dependency width from two to one measured `22.7336/81.7928/311.7663 ms`. It was exact but likewise regressed B4/B16. Width one is rejected, leaving width two bracketed by both tested neighbors.

Converting both 16-row output tiles before stores (`epilogue_tiles_ahead=2`) measured `22.7679/81.3857/311.3760 ms`. It was exact but slower on all three shapes than one-tile-ahead scheduling. The retained J32/J16 epilogue point is therefore one tile ahead, dependency width two, priority two.

Pairing each of the four generated barriers with the installed body's `buffer_gl0_inv` remained exact and resource-identical but measured `22.7382/81.6651/311.7921 ms`. All three shapes regressed. Explicit L0 invalidation is rejected and removed; the generated barriers remain without cache invalidation.

### Eight-wide correction dependency schedule

The compiler delay analysis exposed the dominant generated scheduling defect: every output element completed its dependent `i32->f32`, scale product, activation-scale accumulation, and minimum correction chain before the next independent element began. The exact same per-element arithmetic was regrouped into eight-wide phases. Stored-sum groups reuse the post-WMMA activation registers for eight independent products. Missing-sum groups use the otherwise idle `v98:v105` window so the persistent all-ones results in `v76:v83` remain intact. The first draft incorrectly overlapped four temporaries with those all-ones registers and failed numerically; moving the temporaries restored bitwise parity.

The corrected body remains `135 VGPRs`, `40 SGPRs`, `25,600` LDS bytes, 60 static WMMAs, four barriers, zero private storage, and zero spills. It passed uniform, boundary, skewed, repeated-ID, sparse-ID, active/inactive weight, workspace, input, and invalid-output checks bitwise against installed dispatch. Its nine-repeat confirmation-prior medians were `15.7222/55.0022/215.3602 ms`, versus installed `19.3646/65.9245/263.0708 ms`, for installed-over-candidate speedups of `1.2317x/1.1986x/1.2215x` at B1/B4/B16. This is the first candidate to pass the multiply competitiveness gate and is retained for the full confirmation sequence.

Reversed-order 25-repeat confirmation measured `15.8260/56.1426/216.8549 ms`, versus installed `19.4388/67.2121/264.9736 ms`, or `1.2283x/1.1972x/1.2219x`. The minimum individual learned/hash medoid ratios were `1.1814x/1.1961x/1.2077x` at B1/B4/B16, and every output remained bitwise exact. The schedule therefore passes the competitive confirmation gate.

The same correction schedule was screened in the pure J32 identity to remove mixed-tail dispatch and code. It remained exact at `135/40/25,600/40/4`, but measured `17.1036/57.3835/217.4303 ms`, slower than mixed J32/J16 on every shape. The mixed identity benefits enough from true 16-row tails to outweigh its branch and code-size cost even at B16; pure J32 is rejected as the selected research identity.

The disjoint 512-draw fitted-prior search bank produced `16.3703/56.1158/217.7270 ms` versus installed `20.1156/67.3708/266.4235 ms`, or `1.2288x/1.2006x/1.2237x`. Minimum individual learned/hash ratios were `1.2075x/1.1763x/1.2049x`; all outputs were exact. The candidate advances to captured-route qualification.

Captured learned/hash route medoids measured `16.8008/56.4017/219.2802 ms` versus installed `20.4248/67.6938/267.1060 ms`, or `1.2157x/1.2002x/1.2181x`. Minimum captured-profile ratios were `1.1920x/1.1773x/1.1909x`, with bitwise output throughout. The candidate advances to synthetic controls.

Uniform, skewed, sparse, and boundary controls measured aggregate ratios of `1.2222x/1.2016x/1.2207x`. The minimum individual control ratios were `1.2008x/1.1970x/1.2196x` at B1/B4/B16, and all twelve outputs were bitwise exact. The gain is not dependent on the fitted prior.

Sequential 25-repeat controls ran each implementation's samples contiguously in both pass orders. Candidate-first/installed-first weighted ratios were `1.2209x/1.2091x` at B1, `1.1952x/1.1967x` at B4, and `1.2215x/1.2204x` at B16. Minimum profile ratios remained `1.1729x`, `1.1947x`, and `1.2075x`; all outputs were exact. Interleaving and cache order do not explain the gain.

Complete-call reversed-order 25-repeat timing included the fixed Q8_1 `F16_D2S6` quantizer and the allocated shared workspace before each multiply. Weighted candidate/installed times were `15.9617/19.6355 ms`, `57.5138/69.0594 ms`, and `224.5045/273.0164 ms`, for `1.2302x/1.2007x/1.2161x`. Minimum profile ratios were `1.1912x/1.2003x/1.2048x`, with exact output. The multiply gain survives the production quantization cost.

### Larger ownership after correction phasing

The correction schedule changed the prerequisite for the previously rejected 64-row parent. A composed 64-row body now uses the distributed producer, Meta2 paired stores, partial LDS retirement, and four-tile phased correction. Its missing-sum temporaries occupy the unused upper half of the 16-register all-ones allocation; an earlier decode-scratch placement was not exact. A 64/32 mixed-tail draft also failed bitwise parity and is rejected without timing.

The pure 64-row body passed the complete route/mutation matrix at `159/40/30,208/80/4`, with zero private storage and spills. Its nine-repeat confirmation-prior times were `21.2985/57.5899/206.2302 ms`, versus installed `19.3801/66.8774/264.5587 ms`, or `0.9099x/1.1613x/1.2828x`. It remains rejected for B1/B4, but improves over the confirmed 32/16 B16 candidate and is retained for B16-specific qualification.

Reversed-order 25-repeat B16 confirmation measured `206.6285 ms` versus installed `265.1847 ms`, or `1.2834x`. Every learned/hash medoid was exact and fell between `1.2821x` and `1.2855x`, so the B16 result advances to broad controls.

Search, captured, and synthetic B16 controls measured `1.2847x`, `1.2829x`, and `1.2915x`; sequential candidate-first/installed-first passes measured `1.2856x/1.2829x`. Complete-call timing with fixed Q8_1 quantization measured `213.4464/272.4457 ms`, or `1.2764x`. J64 is retained for B16.

J64 inherits the four-tile BF16 epilogue. One- and two-tile variants were exact at the same resources. Their nine-repeat ratios were `1.2845x` and `1.2851x`, but one-tile reversed-order 25-repeat confirmation measured `1.2836x`, indistinguishable from the qualified four-tile `1.2834x`; no premise exists for a change. Priority zero regressed to `1.2728x`. Dependency width one measured `1.2825x`; width four reached `1.2855x` at nine repeats but only `1.2844x` in reversed-order 25-repeat confirmation, a noise-scale movement of about 0.08 percent over width two. J64 therefore retains its four-tile epilogue, dependency width two, and priority two.

An exact J128 distributed/phased screen used `239/40/39,424/160/4` with zero spills and measured `239.1084 ms` versus `263.6021 ms`, or `1.1024x`. It is about 16% slower than J64 at B16, so the 128-row identity is rejected and removed.


## Research Selection

The retained research selection is shape-specific and does not alter public dispatch or packaging. Effective throughput uses the dense-equivalent operation count `2*R*4096*2048` divided by the qualified weighted complete-call median. It includes the fixed Q8_1 quantizer used by the retention audit and is therefore an end-to-end effective rate, not a WMMA-only rate.

| aggregate rows | generated identity | GGTensile effective TFLOPS | AITER effective TFLOPS | speedup vs AITER |
| ---: | --- | ---: | ---: | ---: |
| 12,288 (B1) | distributed phased J32/J16 | 12.92 | 10.50 | 1.2302x |
| 49,152 (B4) | distributed phased J32/J16 | 14.34 | 11.94 | 1.2007x |
| 196,608 (B16) | distributed phased J64 | 15.45 | 12.11 | 1.2764x |

The J32/J16 identity uses `135/40/25,600/60/4`; J64 uses `159/40/30,208/80/4`. Both have zero private bytes, spills, scratch instructions, calls, and dynamic stack. The research launcher retains the installed control selection independently for every comparison.

## Recursive Final Classification

- Retained: dedicated grouped Q2_K semantics and ABI; bounded `F16_D2S6` workspace production; exact partial-LDS retirement; pre-negated dmin; paired payload and metadata stores; all-wave distributed producer; compact 320-byte decoded LDS rows; J32/J16 one-tile-ahead and J64 four-tile BF16 epilogues, both at width two and priority two; eight-wide correction dependency phases; J32/J16 for B1/B4; and J64 for B16. The test suite locks both selected identities, strict resources, deterministic rebuilds, installed-control boundaries, and cross-format rejection.
- Rejected: dynamic parents; standalone association; HIP packed 400-byte layout; 336-byte layout; full and descending two-group pipelines; compact next-group prefetch; partial VMEM; VOPD initialization; explicit producer and activation hard clauses; cache invalidations; priority zero; epilogue dependency widths one/four; J32/J16 two-tile and J64 one/two-tile epilogues; pure J32 selection; J64 for B1/B4; bitwise-invalid J64/J32 mixed tails; and J128. The J128 phased screen is exact but resource-limited at `239/40/39,424/160/4` and is slower than J64.
- Deferred behind a changed prerequisite: prepared decode requires a model-owned packed representation and lifetime; persistence or flattened tasks require demonstrated route/launch imbalance; SplitK requires an explicit reduction contract and a complete-call win; non-power-of-two row ownership requires physical-plan and output-layout support. These are not current in-contract scheduling changes.
- Contract-incompatible: paired projection kernels, host route readback, route descriptor caches, hidden dense shadows, public dispatch/package changes, and full-bank dequantized reference allocation.
- Actionable: none found after the correction-phase revisit. The changed prerequisite was retested through J64 and J128 ownership. Further work requires one of the deferred contract changes above.

## Final Verification

Selected artifacts rebuilt byte-identically twice:

| aggregate rows | identity | VGPRs | LDS bytes | WMMAs | barriers |
| ---: | --- | ---: | ---: | ---: | ---: |
| 12,288 | distributed J32/J16 | 135 | 25,600 | 60 | 4 |
| 49,152 | distributed J32/J16 | 135 | 25,600 | 60 | 4 |
| 196,608 | distributed J64 | 159 | 30,208 | 80 | 4 |

Both selected identities again passed uniform, boundary, skewed, repeated-ID, sparse-ID, active/inactive weight, workspace, input, and invalid-output gates bitwise against installed controls. `tests/ggtensile/test_grouped_mmq_fwd.py` passes 40 tests. The broader grouped-plus-dense command passes 167 tests; its five pre-existing Q6 container reproducibility failures leave the asserted Q6 source, `.text`, and resource checks passing. The frozen Q4 source set and selected Q5 artifacts also rebuilt byte-identically.
