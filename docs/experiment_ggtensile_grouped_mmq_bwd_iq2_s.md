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

| Geometry | Weighted HIP / candidate ms | Speedup vs HIP | Minimum medoid | Decision |
| --- | ---: | ---: | ---: | --- |
| M64/N64 | `4.1594 / 3.5215` | `1.1812x` | `0.6065x` | Rejected by M128/N64 |
| M128/N64 | `4.1454 / 3.2794` | `1.2641x` | `0.7488x` | B1 parent |
| M128/N128 | `4.1693 / 3.4026` | `1.2253x` | `0.6376x` | Rejected by M128/N64 |

The result matches Q5 more closely than Q2: M64 reduces accumulator pressure but repeats the codebook decoder over twice as many M tiles, while N128 raises the body to 210 VGPRs. M128/N64 is the retained B1 premise for layout, schedule, decoded-B pipeline, and mixed-tail controls.

### First layout, schedule, tail, and pipeline qualification

Plain, pad16, swizzle4/8/16, SIA4, SIA2, and `Mixed128_64` single-LDS controls all pass the 625-row packed-HIP and mutation matrix. The M128/N128 SIA4/swizzle8 double-LDS control does not: candidate-versus-HIP, gradient, route, active-weight, and independent-oracle comparisons fail, while deterministic rerun remains stable. It is rejected before timing in this form. The failure is isolated to the decoded-B pipeline dispatch and will be inspected once for a local register-lifetime defect; no correctness waiver or timing result is accepted.

Inspection found that the shared decoded-B pipeline repurposes the first `quant_dm`/`quant_scale` registers as LDS read addresses after decode preparation. IQ2_S had left `db` and signs live in that alias set. Moving those two post-lookup values into the qh/scale slots, which are dead after codebook issue and outside the address pair, fixes the lifetime without adding registers or waits. Rebuilt single-LDS and double-LDS controls both pass every packed-HIP and mutation check. The repaired double-LDS mechanism is retained for timing; the original failing artifact remains rejected.

The B1 fitted screen closes the first mechanism set:

| Control | Weighted candidate ms | Speedup vs HIP | Minimum medoid | Decision |
| --- | ---: | ---: | ---: | --- |
| M128/N64 SIA5, plain | `3.5394` | `1.1762x` | `0.7250x` | Rejected |
| M128/N64 SIA5, pad16 | `3.3202` | `1.2508x` | `0.7710x` | Near parent; no tail gain |
| M128/N64 SIA5, swizzle4/8/16 | `3.4507 / 3.3300 / 3.5487` | `1.2067x / 1.2441x / 1.1716x` | `0.7195x / 0.7638x / 0.7501x` | Rejected |
| M128/N64 SIA4, pad8 | `3.3004` | `1.2650x` | `0.7679x` | Near SIA5 parent |
| M128/N64 SIA2, pad8 | `3.9366` | `1.0528x` | `0.6817x` | Rejected |
| M128/N128 SIA4 double-LDS, swizzle8 | `3.6801` | `1.1315x` | `0.6556x` | Rejected by timing |
| M128/N64 SIA5 pad8, `Mixed128_64` | `2.9452` | `1.4111x` | `0.7942x` | B1 parent |

The mixed tail is the only material improvement, reducing weighted latency by roughly ten percent while improving every medoid. Plain and swizzled LDS are closed. Double LDS is now correct but loses the single-LDS N64 parent decisively. Pad8/SIA5 mixed becomes the B1 parent; a direct bracket and mixed-layout controls remain before closure.
