# GGTensile Grouped MMQ Backward IQ2_S Experiment

## Purpose

Implement and optimize isolated gfx1151 grouped non-paired IQ2_S backward kernels for the Qwen routed down projection. Public dispatch, generated bundles, extension registration, packaging, and HIP fallback remain unchanged pending a separate integration review.

The promotion objective is the Qwen learned routing-prior weighted sum of per-medoid median complete-call latency. Uniform, skewed, sparse-ID, and boundary routes remain correctness and diagnostic controls rather than ranking vetoes.

## Exact Contract

For routed GEMM `g`:

```text
dY_g[M_g,2048] x W_g[2048,512] -> dX_g[M_g,512]
```

The exact aggregate-row keys are `R={16384,65536,262144}`. The physical IQ2_S expert bank is `[256,2048,164]`: each 256-value block occupies 82 bytes, each packed row contains two blocks, and each expert occupies 335,872 bytes. Inputs and outputs are contiguous BF16, accumulation is FP32 WMMA V1, and output conversion is BF16 RNE.

The grouped ABI remains the 56-byte routed backward research ABI: `grad_output`, `packed_weight`, `grad_input`, `expert_indices`, `expert_offsets`, `num_experts`, `rows`, and `bytes_per_expert`. Route count and optional split ownership are launch geometry. The paired 512x2048 IQ2_S path, host route inspection, dense shadows, prepared banks, atomics, reduction workspaces, and companion setup kernels are out of scope.

The authoritative packed control is `~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf`, tensor `blk.10.ffn_down_exps.weight` (`IQ2_S`, logical `[256,2048,512]`, packed `[256,2048,164]`). The independent oracle dequantizes only selected routed experts to BF16 before matmul.

## Arithmetic Boundary

IQ2_S cannot reuse Q2_K arithmetic by relabeling the format. A block stores FP16 `d`, 32 low grid-index bytes, 32 sign bytes, eight packed high-index bytes, and eight packed scale bytes. Every aligned 16-value group uses two 1024-entry codebook lookups, two independent sign bytes, one four-bit scale, and `db = fp32(d) * (scale + 0.5) * 0.25`.

The first decoder maps one aligned 16-value group to each lane. It reconstructs two ten-bit codebook indices, loads two 64-bit grid entries from assembly-local read-only data, applies sign bits, scales in FP32, rounds decoded weights to BF16 in LDS, and reuses the established FP32-WMMA/BF16-RNE leaf. Codebook data is emitted into each isolated assembly artifact; it is not added to the ABI.

## Qualification

Correctness covers exact rows, non-aligned route tails, first/non-first routes, sparse and repeated expert IDs, deterministic reruns, gradient and route mutation, active/inactive expert-weight mutation, invalid experts, malformed first/final offsets, and untouched sentinels. Candidate versus packed HIP must be BF16 bit-exact; the independently dequantized reference must remain finite with NRMSE below `0.01`.

Inspection requires gfx1151, wave32, code object v5, exact 56-byte metadata, bounded VGPR/SGPR indices, zero private bytes and spills, no scratch/calls/dynamic stack, and derived static WMMA/barrier counts. Independent generate/build/inspect roots must produce byte-identical source and HSACO.

Timing uses warmed rotating GPU events and includes output allocation in both HIP and candidate complete-call paths. Search uses the five Qwen learned medoids for each physical batch. Retained changes receive disjoint confirmation with 5 warmups, 25 repeats, and reversed rotating order. TFLOPS is `2 * R * 512 * 2048 / (latency_ms * 1e9)`.

## Final Accepted Performance

The public bundle exports the three exact catalog identities below. Logical throughput is `2*R*N*K/(median_ms*1e9)`. Speedup is `HIP time / GGTensile time`, so values above `1.0x` favor GGTensile.

| Exact `(R,N,K)` | Public catalog hash | HIP ms | GGTensile ms | HIP TFLOPS | GGTensile TFLOPS | HIP time / GGTensile time |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `(16384,512,2048)` | `ggsol_67d58c3f849e005f` | `4.2894` | `2.7285` | `8.010` | `12.593` | `1.5721x` |
| `(65536,512,2048)` | `ggsol_d82241ad1072e2f0` | `9.9753` | `7.6761` | `13.778` | `17.905` | `1.2995x` |
| `(262144,512,2048)` | `ggsol_8684bd239b31ec32` | `34.2671` | `23.7922` | `16.043` | `23.107` | `1.4403x` |

The inactive-M reports measure retained parent versus final candidate, not a fresh three-way HIP bracket. To avoid mixing timing sessions, the HIP and retained-parent values use the earlier disjoint HIP confirmation, and the final latency/TFLOPS and speedup are normalized with the independently confirmed candidate/parent ratios. The latest raw brackets are `2.7853 -> 2.7241 ms` at B1, `7.8368 -> 7.6381 ms` at B4, and `24.0493 -> 23.7761 ms` at B16; reports are `~/tmp/torch-ggml-ops/ggtensile-inactive-m-iq2-s-b1-confirm25.json`, `...-b4-confirm25.json`, and `...-b16-confirm25.json`.

## Planned Search

- Add strict IQ2_S backward identity and exact Qwen shapes without changing retained Q2_K/Q4_K/Q5_K generated sources or the grouped ABI.
- Implement the authoritative width-16 codebook decoder and qualify a minimal single-LDS pilot against packed HIP and the independent oracle.
- Establish M64/N64, M128/N64, and M128/N128 controls, then inspect codebook latency, LDS layout, and decoded-B pipeline viability.
- Screen metadata sharing, decode scheduling, route tails, and split factors only where emitted ISA or ownership changes.
- Confirm per-key winners on disjoint Qwen medoids, qualify full rows, rebuild independently, and report HIP/GGTensile TFLOPS and speedup.

## Completion Record

This section is updated after every coherent implementation or failed experiment.

### Campaign opened

The completed grouped Q2_K/Q4_K/Q5_K shell, authoritative HIP `decode_iq2_s_group_16`, assembly-local 1024-entry codebook emitter, Qwen packed model tensor, and learned route prior were audited. The exact non-paired backward problem is `(R,512,2048)` for `R={16384,65536,262144}`, with two 82-byte blocks per packed row and a 335,872-byte expert stride. The first coherent change will add a strict IQ2_S identity, local codebook address state and rodata, and a direct width-16 BF16 decoded-LDS implementation. No production integration surface is in scope.

### Strict IQ2_S identity and decoded-LDS pilot

The backward-only format registry and grouped contract now admit IQ2_S for the three exact Qwen aggregate shapes. IQ2_S receives dedicated metadata registers and one aligned codebook SGPR pair; Q2_K/Q4_K/Q5_K register allocation and source emission remain on their existing branches. The assembly artifact carries the authoritative 8 KiB local codebook as read-only data and materializes its address with the same `s_getpc_b64` relocation contract used by the retained IQ2_S forward lowerers. Inspection permits that address materialization for IQ2_S while continuing to reject calls, `s_setpc_b64`, `s_swappc_b64`, scratch, spills, and dynamic stack use.

The first runtime pilot exposed three decoder-local defects before qualification: the second lookup reused a destructively masked high-index nibble, nonzero K rows added the packed-row base twice when addressing scales, and the generic N128 static coordinate used a Q4/Q5 half-tile convention rather than IQ2_S's full 128-value tile stride. All three were fixed without arithmetic or wait waivers. One-hot probes across K rows `0,1,2,7,8,15,16,31,32,127,128,2047` then reproduced every one of 512 independently dequantized BF16 weights exactly.

The M128/N128, DepthU32, SIA2, linear single-LDS pilot passes the 625-row boundary matrix bit-for-bit against packed HIP, including deterministic rerun, gradient mutation, route mutation, active/inactive weight mutation, malformed routes, and untouched-tail controls. The independent BF16 oracle differs in 223 of 320,000 values with maximum absolute error `0.0078125` and NRMSE `6.89e-5`. The inspected artifact uses 194 VGPRs, 38 SGPRs, 8 KiB LDS, 32 static WMMAs, and two barriers with zero private bytes or spills.

The five B1 Qwen learned search medoids put this correctness anchor at weighted HIP/GGTensile complete-call latency `4.1498/4.2616 ms`, or `0.9738x` HIP throughput. The dominant medoid is nearly tied at `0.995x`, but individual medoids range down to `0.6035x` as route sizes shrink. M128/N128 is retained as the arithmetic anchor, not as a timing parent; smaller geometry and explicit tail ownership are the next controls.

### First B1 geometry screen

SIA5/PGR2, eight-byte-padded single-LDS M64/N64, M128/N64, and M128/N128 controls all pass the full 625-row mutation matrix bit-for-bit against packed HIP. Their independent-oracle NRMSE remains `6.89e-5`. Inspected resources are `89/38/5 KiB`, `137/38/5 KiB`, and `210/38/10 KiB` respectively, with zero private bytes or spills.

The geometry screen retained M128/N64 as the B1 parent. M64/N64 was rejected for repeated decode work, and M128/N128 was rejected for its larger accumulator/register envelope; all three geometries remained exact and spill-free.

The result matches Q5 more closely than Q2: M64 reduces accumulator pressure but repeats the codebook decoder over twice as many M tiles, while N128 raises the body to 210 VGPRs. M128/N64 is the retained B1 premise for layout, schedule, decoded-B pipeline, and mixed-tail controls.

### First layout, schedule, tail, and pipeline qualification

Plain, pad16, swizzle4/8/16, SIA4, SIA2, and `Mixed128_64` single-LDS controls all pass the 625-row packed-HIP and mutation matrix. The M128/N128 SIA4/swizzle8 double-LDS control does not: candidate-versus-HIP, gradient, route, active-weight, and independent-oracle comparisons fail, while deterministic rerun remains stable. It is rejected before timing in this form. The failure is isolated to the decoded-B pipeline dispatch and will be inspected once for a local register-lifetime defect; no correctness waiver or timing result is accepted.

Inspection found that the shared decoded-B pipeline repurposes the first `quant_dm`/`quant_scale` registers as LDS read addresses after decode preparation. IQ2_S had left `db` and signs live in that alias set. Moving those two post-lookup values into the qh/scale slots, which are dead after codebook issue and outside the address pair, fixes the lifetime without adding registers or waits. Rebuilt single-LDS and double-LDS controls both pass every packed-HIP and mutation check. The repaired double-LDS mechanism is retained for timing; the original failing artifact remains rejected.

The B1 fitted screen closes the first mechanism set. Plain LDS, swizzles, SIA2, and corrected double LDS are closed; pad8/SIA5 `Mixed128_64` is the only material improvement (`2.9452 ms`, `1.4111x` HIP) and becomes the parent. Pad16 is statistically flat (`2.9378` versus `2.9452 ms`; same-process bracket `2.91844/2.91568 ms`) but uses 1 KiB more LDS, so pad8 is retained.

### Signed codebook preapply

The baseline decoded each magnitude and sign separately, then used XOR/subtract to form a signed integer. A retained lowering optimization now applies four sign nibbles to the four loaded codebook dwords with the forward-proven `v_perm_b32` selector construction. Each element then needs one signed-byte extraction. The IQ2_S-unused `input_half` SGPR holds `0x03020100`, so the change adds no resources and removes 48 static VALU issues from the mixed M128/M64 artifact (`797 -> 749`).

The candidate passes the complete packed-HIP matrix. A separate fitted run reaches weighted `2.8520 ms` (`1.4481x` HIP). More importantly, a 15-repeat same-process bracket against the preserved scalar-sign artifact improves every medoid and lowers weighted latency `2.9112 -> 2.8261 ms`, a `3.01%` gain. Signed-dword preapply is retained unconditionally for IQ2_S; the per-element sign path is closed.

### First B4/B16 ownership screen

M128/N64, M128/N128, split4/8/16/32, and repaired double-LDS controls pass the 625-row matrix at both larger aggregate keys. Serial ownership is poor on low-weight high-skew B4 medoids, while every split control is faster than HIP on every medoid.

The larger-key ownership screen rejected serial work and the smaller split factors. It retained B4 split16 and B16 split32 around the repaired double-LDS path; subsequent same-process brackets reopened the endpoint and led to the final split64 parents documented below.

B4 has a clear double-LDS premise; B16 has a noise-sized three-way order. Split, SIA, LDS layout, and buffering will be isolated around these parents before any final bracket.

Padded and swizzle16 double-LDS requests are rejected by the shared typed pipeline boundary before assembly: its two-buffer address toggle is implemented only for XOR-8. They are not timed. Single-LDS padded/swizzled controls and XOR-8 double-LDS SIA/split controls remain valid.

The isolation screen fixes B4 around SIA4/XOR-8 double LDS. SIA4 pad8 single reaches `8.6935 ms`; SIA4 swizzle8/16 single regress to `9.2166/9.4471 ms`; SIA5 XOR-8 double is `8.7720 ms`. Within SIA4 double LDS, split8/16/32 are `8.6233/8.5517/8.7360 ms`. B4 therefore retains split16.

B16 materially benefits from combining double LDS with split32: it reaches `29.2760 ms` (`1.1835x` HIP), versus `30.6796 ms` for SIA4 XOR-8 single LDS and roughly `29.80 ms` for the earlier split16 controls. Its dominant routes still leave two M tiles per split32 owner, so one IQ2_S-only split64 endpoint was opened. Split64 is exact and separately ties split32 (`29.2798` versus `29.2760 ms`) while improving the four low-weight medoids. A 5-warmup/25-repeat same-process bracket resolves the endpoint: split64 improves every medoid and lowers weighted latency `29.5511 -> 29.3369 ms`, or `0.73%`. Split64 is retained only for grouped IQ2_S; all other grouped quant types reject it.

### Workgroup-local codebook

The global-codebook decoder performs two random 64-bit VMEM reads per lane in every DepthU iteration. A retained mechanism now stages the full 8 KiB codebook contiguously into a disjoint LDS region once per active workgroup, then uses `ds_load_b64` for all lookups. The decoded-B buffer size and double-buffer toggle remain unchanged; total LDS grows by exactly 8 KiB and setup adds one barrier. B1/B4/B16 resources become `137/38/13 KiB`, `214/38/24 KiB`, and `214/38/24 KiB`; VGPR/SGPR counts and spill-free status are unchanged.

All three staged controls pass the packed-HIP matrix. Separate fitted results are B1 `2.6069 ms` (`1.6042x` HIP), B4 `8.1090 ms` (`1.2455x`), and B16 `26.7127 ms` (`1.2966x`). Same-process 15-repeat brackets against preserved global-codebook artifacts confirm every medoid and improve weighted latency `2.8282 -> 2.5811 ms` (`9.57%`) at B1, `8.5783 -> 8.0955 ms` (`5.96%`) at B4, and `29.2852 -> 26.8259 ms` (`9.17%`) at B16. Workgroup-local staging is retained unconditionally; repeated global codebook lookup is closed.

### Post-staging schedule and ownership reopening

Moving codebook traffic from VMEM to LDS changes the buffering premise, so schedule, layout, and ownership were reopened without revisiting closed arithmetic mechanisms. B1 remains M128/N64 SIA5/pad8 single LDS with `Mixed128_64`: SIA4/full, SIA5/swizzle8/mixed, and M128/N128 double-LDS controls reach `2.9787`, `2.6472`, and `3.5112 ms` versus the retained `2.6069 ms` fitted result.

At B4 and B16, pad8 single LDS now decisively beats XOR-8 double LDS. SIA2 remains slow. Fixed-ownership 15-repeat brackets choose SIA5 over SIA4 by `8.0677 -> 8.0170 ms` (`0.63%`) at B4 and `24.3643 -> 24.2527 ms` (`0.46%`) at B16. B4 split16 versus split64 then improves every medoid and weighted latency `8.0449 -> 7.9961 ms` (`0.61%`). A mixed tail helps the four skew medoids, but at fixed split64 it slows the 96.5%-weight medoid and loses weighted latency `8.0358 -> 8.0901 ms`; B4 therefore retains the full M128 path. B16 split64 improves every medoid over split32 in a 25-repeat bracket, narrowly lowering `24.2857 -> 24.2689 ms` (`0.07%`). Its mixed control also loses. The final post-staging parents are:

| Key | Geometry and ownership | VGPR / SGPR / LDS | Tail |
| --- | --- | ---: | --- |
| B1 | M128/N64, SIA5/PGR2, pad8 single LDS, serial routes | `137 / 38 / 13 KiB` | `Mixed128_64` |
| B4 | M128/N128, SIA5/PGR2, pad8 single LDS, split64 | `210 / 38 / 18 KiB` | Full M128 |
| B16 | M128/N128, SIA5/PGR2, pad8 single LDS, split64 | `210 / 38 / 18 KiB` | Full M128 |

### Decode extraction cleanup

The lane-half selector is now hoisted out of the per-row preparation loop, and equivalent shift/mask pairs use direct bitfield extracts. This removes six static VALU issues at N64 (`751 -> 745`) and eight at N128 (`771 -> 763`) without changing resources. All three keys pass the complete packed-HIP matrix. Same-process 15-repeat brackets improve weighted latency `2.5846 -> 2.5697 ms` (`0.58%`) at B1, `8.0378 -> 7.9967 ms` (`0.51%`) at B4, and `24.0933 -> 23.9533 ms` (`0.58%`) at B16.

An attempted FMA/output-modifier replacement for `(scale + 0.5) * 0.25` assembled and reduced one more issue per row, but failed candidate, mutation, and independent-oracle comparisons. The proven add/multiply sequence was restored; no arithmetic or correctness waiver is retained.

### Final qualification and confirmation

Each winner was rebuilt independently in two disjoint roots.

Inspection reports `137/38/13312` VGPR/SGPR/LDS bytes, 24 static WMMAs, five barriers, and 745 static VALU issues for B1. B4/B16 report `210/38/18432`, 32 WMMAs, three barriers, and 763 VALU issues. All have zero private bytes and zero VGPR/SGPR spills.

Full boundary-distribution qualification covers all `16384`, `65536`, and `262144` rows, or `8,388,608`, `33,554,432`, and `134,217,728` BF16 outputs. Every candidate output is bit-exact against packed HIP, deterministic reruns and all active mutation controls are bit-exact, malformed routes preserve their target sentinels, and no tail element changes. The independent BF16 oracle has maximum absolute error `0.0078125`; NRMSE is `6.03e-5`, `7.95e-5`, and `7.20e-5` for B1/B4/B16.

The earlier parent/HIP confirmation is represented in the normalized table above. The latest disjoint parent-to-candidate confirmations are:

| Key | Retained parent -> final candidate (ms) | Candidate / parent |
| --- | ---: | ---: |
| B1 | `2.7853 -> 2.7241` | `1.0225x` |
| B4 | `7.8368 -> 7.6381` | `1.0260x` |
| B16 | `24.0493 -> 23.7761` | `1.0115x` |

All three final candidates remain BF16 bit-exact on the confirmation medoids and improve their retained parent. Production dispatch, generated bundles, extension registration, packaging, and HIP fallback remain unchanged.

## Reopened post-refactor optimization program

IQ2_S already retains signed-dword preapplication, workgroup-local codebook staging, split64 ownership, and the selected layout/schedule interactions. The only immediate reopening is the fitted-objective interpretation of inactive-M suppression:
- Add the same wave-uniform wholly inactive 16-row consumer guard used by the Q4_K/Q5_K/Q2_K controls, leaving codebook staging, per-K decode, LDS traffic required by active siblings, and barriers uniform. The historical HIP candidate improved only `1.06%` geometrically and was rejected because B4 uniform and B16 boundary regressed. Those synthetic timings are diagnostic under the current objective, so run one narrow fitted B1 check and retain only a clear weighted gain.
- Device tasks remain secondary to the Q4_K/Q5_K implementation. HIP already retains IQ2_S row tasks at B1, but this GGTensile body is substantially faster and uses split64; no margin transfers automatically. Reuse a proven non-paired task ABI only if the first two quant types establish a resource-clean mechanism.
- Do not reopen global codebook lookup, width8 decode, M256/N64, shared pair/down swizzles, or broad decoder-wave designs. Direct controls already close those mechanisms. Any later decode work must reduce state below the current `194-210` VGPR envelope while preserving exact signed grid/scale reconstruction.

Fitted medoids alone rank timing. All route controls still gate exactness, deterministic coverage, malformed-route behavior, private storage, and spills. Each coherent result is appended here when completed.

### Inactive-M suppression build and correctness checkpoint

The first identity-neutral implementation guards each wholly inactive wave and each inactive 16-row M minitile around the WMMA consumer, while leaving codebook staging, packed decode, required LDS traffic, barriers, and masked global access uniform. Both IQ2_S manifest controls assemble at their prior `194 VGPR / 38 SGPR / 16384 B LDS` envelope with zero private bytes or spills and unchanged identity and static WMMA/barrier counts. The complete grouped build record is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-manifest-build/build.json`.

The strict 625-row boundary matrix for split64 `ggsol_2146fed1ffe03582` passes packed HIP, independent BF16, deterministic rerun, gradient/route/active/inactive-weight mutations, malformed expert/offset controls, and tail sentinels. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-iq2-s-split64-correctness.json`. This is a build/correctness checkpoint only; the narrow fitted B1 timing gate remains intentionally unresolved.

The selected B1 `ggsol_67d58c3f849e005f` source was built independently from the pre-change `c41913f` worktree and the suppression worktree. Both inspect at `137 VGPR / 38 SGPR / 13312 B LDS`, zero private bytes and spills, 24 static WMMAs, and five barriers. A same-process fitted search-bank bracket with allocation in both paths, three warmups, nine repeats, and rotating order improves weighted latency `2.6476 -> 2.5891 ms`, or `1.0226x` parent throughput. Outputs are bitwise equal and every medoid improves, with a minimum ratio of `1.0076x`. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-iq2-s-b1-search9.json`. This invalidates the old synthetic-timing rejection for the selected GGTensile parent and advances to disjoint confirmation.

Disjoint confirmation with five warmups, 25 repeats, reversed base order, and the same complete-call allocation contract confirms `2.7853 -> 2.7241 ms`, or `1.0225x`. Every medoid remains bitwise exact and improves individually, with a minimum ratio of `1.0182x`. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-iq2-s-b1-confirm25.json`. Inactive-M suppression is retained for IQ2_S B1 despite the older synthetic-control result.

The selected split64 B4 `ggsol_d82241ad1072e2f0` and B16 `ggsol_8684bd239b31ec32` artifacts reproduce their parent envelopes at `210 VGPR / 38 SGPR / 18432 B LDS`, with zero private bytes or spills. Both pass the strict 625-row packed-HIP, independent BF16, deterministic, mutation, malformed-route, and sentinel matrix, including uniform codebook staging and its three barriers. Reports are `~/tmp/torch-ggml-ops/ggtensile-inactive-m-iq2-s-b4-correctness.json` and `~/tmp/torch-ggml-ops/ggtensile-inactive-m-iq2-s-b16-correctness.json`. Fitted parent/candidate timing remains outstanding.

The B4 fitted search-bank bracket improves `7.9938 -> 7.7650 ms`, or `1.0295x`, with every medoid bitwise exact and faster. The minimum ratio is `1.0276x`. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-iq2-s-b4-search9.json`; B4 advances to disjoint confirmation.

The B16 fitted search-bank bracket improves `24.0904 -> 23.9032 ms`, or `1.0078x`, with every medoid bitwise exact and faster. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-iq2-s-b16-search9.json`. Because the weighted margin is below one percent, B16 remains provisional pending a longer disjoint bracket.

The disjoint B4 confirmation bracket measures `7.8368 -> 7.6381 ms`, or `1.0260x`, with every medoid bitwise exact and faster. The minimum ratio is `1.0239x`. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-iq2-s-b4-confirm25.json`. Inactive-M suppression is retained for IQ2_S B4.

The disjoint B16 confirmation bracket resolves the provisional result positively: `24.0493 -> 23.7761 ms`, or `1.0115x`, with every medoid bitwise exact and faster. The minimum ratio is `1.0074x`. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-iq2-s-b16-confirm25.json`. Inactive-M suppression is retained across all selected IQ2_S B1/B4/B16 keys.

The EvoTensile median-log robust-scale analysis marks all three weighted comparisons confidently faster at 95%: candidate-minus-parent intervals are `[-3.046,-1.344]%` at B1, `[-4.138,-0.955]%` at B4, and `[-1.629,-0.642]%` at B16. The aggregate analysis is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-confirm25-confidence.json`.

Final selected full-row qualification passes at `16384`, `65536`, and `262144` rows with packed-HIP bit equality, deterministic reruns, all active/inactive mutation controls, malformed-route sentinels, and independent BF16 NRMSE below `0.01`. The report is `~/tmp/torch-ggml-ops/ggtensile-inactive-m-iq2-s-full-correctness.json`.
