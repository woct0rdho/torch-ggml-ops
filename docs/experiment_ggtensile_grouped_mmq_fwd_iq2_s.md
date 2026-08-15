# GGTensile grouped MMQ forward IQ2_S experiment

## Scope

This record covers isolated research kernels for the non-paired Qwen routed-down projection with packed `IQ2_S` weights. The exact production contract is:
- 256 physical experts;
- output features `N = 2048`;
- reduction features `K = 512`;
- aggregate routed rows `R = 16,384`, `65,536`, or `262,144`;
- device-resident int64 expert IDs and cumulative int32 route offsets;
- at most 256 route entries, with the final valid offset equal to `R`;
- fixed HIP `Q8_1` `F32_D4` activation quantization and its allocated workspace;
- BF16 output.

Paired projection kernels, public dispatch, generated bundle tables, extension registration, packaging, and the HIP fallback are out of scope.

## Format contract

One `IQ2_S` block represents 256 values in 82 bytes: one FP16 block scale, 64 `qs` bytes, eight `qh` bytes, and eight packed scale bytes. The first 32 `qs` bytes and two bits from `qh` select one of 1,024 eight-byte codebook entries. The upper 32 `qs` bytes provide one sign bit per decoded value. Two adjacent eight-value groups share a four-bit scale, with effective factor `d * (scale + 0.5) / 4`.

The required `K = 512` row therefore contains two packed blocks and occupies 164 bytes. The grouped launch grid is `(N / 64, G, 1)` with 128 workitems per workgroup for the first candidate family.

## Method

Correctness precedes timing. Candidates must pass independent bounded references, installed-kernel comparison, sequential-route controls, route mutation controls, workspace and input mutation checks, and invalid-route guards before performance can be retained.

Performance uses the fitted Qwen route prior: 512 draws reduced to five weighted medoids, three warmups, and nine alternating repeats. Competitive candidates receive a reversed-order 25-repeat confirmation. Every measurement has an adjacent installed HIP control, and throttled or noisy runs are discarded. Complete-call timings include fixed activation quantization and the allocated workspace.

Retained artifacts must use gfx1151 code-object v5 and wave32, report no private storage or spills, contain no scratch instructions, calls, or dynamic stack, and rebuild byte-identically.

The final review repeats correctness matrices, complete-call confirmation, deterministic rebuild and resource inspection, focused and broad tests, changed-file hooks, and repository-hygiene checks. A candidate is not final until that review is complete.

## Initial baseline and priority

Historical complete-body profiling identifies `R = 262,144` as the first target. The installed grouped HIP `IQ2_S` J64 body measured 25.703 ms, fixed quantization measured 1.849 ms, and the adjacent AITER control measured 24.094 ms. The installed body uses 232 VGPRs and 30,976 bytes of LDS. Exact fresh complete-call baselines for all three production shapes will be recorded with the first runnable research candidate.

The first implementation will model a Tensile-style 128-row by 64-column workgroup with full decoded-weight LDS reuse. Its producer distributes one weight row and one 128-value K half to each workitem, so all four waves cooperate on the 64-row weight tile without duplicate row decode. Tail rows remain route-bounds masked. This establishes the typed problem, solution, physical plan, inspection, runtime, and correctness path before schedule and tile-family screening.

## Experiment log

### Inventory and implementation plan

- Confirmed the three non-paired production shapes and their fixed grouped ABI.
- Confirmed that current research support has no `IQ2_S` problem or lowering.
- Selected the B16 shape as the first optimization target because it has the largest absolute body-time opportunity.
- Chose distributed full-weight decode into LDS as the first candidate, using the existing signed-int8 WMMA and `F32_D4` activation contracts.

No kernel candidate has yet been retained or rejected.

### Typed J64 distributed full-weight foundation

- Added the isolated `IQ2_S` problem and solution identity, exact production validation, packed-format semantics, compact J64 LDS layout, deterministic register plan, codebook emission, assembly lowering, artifact inspection, research runtime controls, and focused tests.
- The physical tile uses 64 activation rows, 64 output-feature rows, two 128-value decoded payload halves, and eight FP32 group factors per half. Its fixed resources are 116 VGPRs, 40 SGPRs, and 30,720 bytes of LDS.
- The producer maps `wave * 16 + lane[3:0]` to one output-feature row and `lane[4]` to one K half. It therefore decodes all 64 rows exactly once across the four waves rather than duplicating rows between waves.
- The first live launch exposed two ownership errors: scale addresses were based on payload lanes instead of WMMA output elements, and the second activation stage reused `v0` after that register had become an accumulator. Both were corrected by using the established wave/lane ownership formulas.
- A temporary constant-codebook launch proved that embedded codebook loads were not the source of the initial memory fault. The linked local table uses the same read-only code-object addressing model as the installed HIP kernel.
- The first completed output was globally sign-inverted because gfx11 `v_perm_b32` byte selectors chose the opposite listed data operand from the initial assumption. Swapping the positive and negative operands restored the intended per-byte sign mapping.
- A four-route, 35-row bounded launch then matched the installed mixed J64/J32 control bitwise for all 71,680 BF16 output elements. The artifact reports wave32, code-object v5, four barriers, 64 static WMMAs, no private storage, and no SGPR or VGPR spills.

This foundation is retained. Production route matrices and timing remain to be qualified before selecting or rejecting the J64 schedule.

### Fitted-prior production baseline

The retained J64 foundation was rebuilt for all three exact production row counts and compared adjacently with the installed dispatch on the five fitted Qwen medoids. All 15 candidate outputs matched the installed controls bitwise. The initial screen used three warmups and nine alternating repeats.

| Batch | Rows | Candidate body | HIP body | Candidate complete | HIP complete | Complete ratio | Candidate effective TFLOPS | HIP effective TFLOPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16,384 | 2.7231 ms | 3.1628 ms | 2.8288 ms | 3.2645 ms | 1.1540x | 12.15 | 10.53 |
| 4 | 65,536 | 8.1321 ms | 8.2610 ms | 8.4819 ms | 8.7318 ms | 1.0295x | 16.20 | 15.74 |
| 16 | 262,144 | 29.9013 ms | 26.9446 ms | 31.1351 ms | 28.2698 ms | 0.9080x | 17.66 | 19.45 |

Effective TFLOPS use dense-equivalent `2 * R * N * K` operations divided by complete-call time, including fixed Q8_1 quantization and the allocated workspace.

The B1 weighted result benefits from the 255-expert medoid; four lower-expert medoids remained 2-4% slower than HIP. B4 also had two medoids below `0.90x` despite a weighted win. B16 has 256 active experts in every fitted medoid and is consistently 9-10% slower than HIP, making it the next optimization target.

### Rejected J128 output-column ownership

An eight-wave, 256-thread variant doubled `MacroTile1` from 64 to 128 while retaining the 64-row route tile. Four waves staged the two halves of 64 Q8_1 rows and all eight waves decoded and computed separate 16-column IQ2_S weight tiles. This halved the x-grid and route/activation replication while preserving the same aggregate decode and WMMA work.

The candidate built at 116 VGPRs, 40 SGPRs, and 52,224 bytes of LDS with 64 static WMMAs, four barriers, zero private storage, and zero spills. A bounded 35-row route matched the installed mixed control bitwise for all 71,680 output elements. It nevertheless regressed B16: the five-medoid weighted body was 31.3072 ms versus 26.5758 ms for adjacent HIP, and complete-call time was 32.7224 ms versus 28.1217 ms (`0.8594x`). Every medoid was between `0.8564x` and `0.8620x`.

The one-workgroup 52,224-byte LDS allocation did not recover the benefit of the smaller grid. J128 is rejected and removed from the retained implementation.

### Coalesced activation staging

The retained J64 parent assigned each workitem one contiguous 72-byte half of an activation row. Although this minimized instructions, adjacent lanes were 144 bytes apart and produced poorly coalesced VMEM traffic. A new candidate linearly maps the 128 workitems over four 2,048-byte `b128` bands and one 1,024-byte `b64` band. It uses the existing 18 staging VGPRs and writes the identical 9,216-byte LDS image. Partial row tiles use a separate masked-load path.

The candidate remains at 116 VGPRs, 40 SGPRs, and 30,720 bytes of LDS with no spills or private storage. The 35-row route matched the installed mixed control bitwise. On the five B16 medoids, weighted body time improved from 29.9013 to 27.9572 ms and complete-call time improved from 31.1351 to 29.4779 ms. Adjacent HIP measured 26.4811 ms body and 28.0094 ms complete, leaving a complete-call ratio of `0.9502x`; individual medoids ranged from `0.9485x` to `0.9525x`.

The corresponding B1 complete call measured 2.7473 ms versus 3.2639 ms for HIP (`1.1881x`, 12.51 versus 10.53 effective TFLOPS), and B4 measured 8.1405 ms versus 8.6783 ms (`1.0661x`, 16.88 versus 15.84 effective TFLOPS). All ten outputs were bitwise exact. One B1 medoid was effectively tied at `0.9960x`; the two lowest-support B4 medoids remained at `0.9252x` and `0.9293x`, consistent with the B16 compute-core deficit.

Coalesced staging is the new provisional J64 parent because it removes more than half of the original B16 deficit without changing arithmetic or resources. It is not retained as final until the remaining HIP gap closes across the lower-support and B16 controls.

### 2026-04-14: rejected payload-first scale schedule

The coalesced parent was screened with a separate schedule that issued all five activation payload reads before weight-scale LDS reads, then delayed scale consumption until after four WMMAs and accumulator conversion. This matched the ordering visible in the installed HIP disassembly and kept the same 116 VGPR, 40 SGPR, 30,720-byte LDS, four-barrier, and 64-WMMA contract.

The 35-row bounded route remained bitwise exact, but the five-medoid B16 complete-call median was 29.5103 ms versus 29.4779 ms for coalesced staging and 28.0326 ms for adjacent HIP (`0.9499x`). The schedule is rejected as a near-neutral regression and removed.

### 2026-04-14: rejected packed index/sign vector loads

The weight producer was screened with packed loads for each 16-byte index and sign half. An initial revision unpacked asynchronous VMEM destinations before `vmcnt(0)` and faulted; restoring explicit VMEM retirement recovered bitwise correctness. Four unaligned `b32` reads per plane regressed B16 complete-call time to 29.6746 ms. One `b128` read per plane with codebook registers as temporary storage measured 29.4607 ms complete versus 29.4779 ms for the parent and 28.0633 ms for adjacent HIP (`0.9526x`).

The apparent `0.06%` parent improvement is below run-to-run resolution and adds unpacking complexity. Packed vector loads are rejected as neutral and removed.

### 2026-04-14: BFE decode extraction

A compact decode candidate replaces separate shift-and pairs with `v_bfe_u32` for QH two-bit fields, sign nibbles, and scale nibbles. This removes 56 VALU instructions per decoded weight block without changing any extracted integer value or the 116 VGPR, 40 SGPR, 30,720-byte LDS resource point.

The 35-row route remained bitwise exact. Across the five B16 medoids, weighted body time improved to 27.5855 ms and complete-call time to 28.9172 ms. Adjacent HIP measured 26.7819 ms body and 28.2080 ms complete, raising the complete-call ratio to `0.9755x`; individual medoids ranged from `0.9730x` to `0.9777x`.

B1 complete-call time improved to 2.6520 ms versus 3.2547 ms for HIP (`1.2273x`, 12.96 versus 10.56 effective TFLOPS), and every B1 medoid beat its control. B4 improved to 7.8400 ms versus 8.5931 ms (`1.0961x`, 17.53 versus 15.99 effective TFLOPS). Two lower-support B4 medoids remained below HIP at `0.9688x` and `0.9443x`, while the three high-support medoids were faster.

BFE extraction is the new provisional parent pending further work on the remaining lower-support and B16 gaps.
