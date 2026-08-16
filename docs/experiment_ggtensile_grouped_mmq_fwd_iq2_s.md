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

BFE extraction is the retained parent for subsequent decode scheduling work.

### 2026-04-14: combined sign-selector construction

The BFE parent still formed each byte-selector dword with multiply, mask, and shift-or. A dependent candidate shifts the spread constant into the multiply and uses `v_and_or_b32` with an invariant SGPR `0x03020100`, removing one instruction and one dependency edge from every signed codebook dword.

The bounded route remained bitwise exact and resources remained 116 VGPRs, 40 SGPRs, and 30,720 bytes LDS. B16 weighted body time fell to 26.7194 ms and complete-call time to 28.2488 ms versus 26.4635 ms and 27.9971 ms for adjacent HIP. The complete-call ratio reached `0.9911x`, with individual medoids between `0.9896x` and `0.9938x`.

B1 complete-call time was 2.6398 ms versus 3.2588 ms for HIP (`1.2345x`), and B4 was 7.8012 ms versus 8.6719 ms (`1.1116x`). Combined selector construction is retained as the decode parent.

### 2026-04-14: quarter-scale arithmetic and payload prefetch

Precomputing `d * 0.25` once and multiplying by `(scale + 0.5)` remained bitwise exact. Its B16 body result was near-neutral at 26.7029 ms, but complete-call time measured 28.1117 ms versus 28.0050 ms for HIP (`0.9962x`). The exact reassociation was retained as the arithmetic parent because it removes seven dependent FP operations per decoded block without increasing resources.

A dependent schedule then prefetches the next group's five activation/weight payload LDS reads while correcting the current WMMA results. Weight and activation scale reads remain after correction and overlap the following four WMMAs. The 35-row route remained bitwise exact with unchanged resources.

The five-medoid B16 screen measured 26.5156 ms body and 27.9807 ms complete versus 26.4687 ms and 28.0333 ms for adjacent HIP. Weighted complete-call performance reached `1.0019x`; individual medoids ranged from `0.9989x` to `1.0045x`.

### Final qualification and retention

The payload-prefetch identity received the required reversed-order 25-repeat confirmation across all three production shapes. Timings are weighted medians over the five fitted Qwen medoids and include fixed Q8_1 quantization and the allocated workspace.

| Batch | Rows | Candidate body | HIP body | Candidate complete | HIP complete | Complete ratio | Candidate effective TFLOPS | HIP effective TFLOPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16,384 | 2.4774 ms | 3.1532 ms | 2.5826 ms | 3.2557 ms | 1.2606x | 13.30 | 10.55 |
| 4 | 65,536 | 7.4912 ms | 8.4727 ms | 7.7919 ms | 8.8203 ms | 1.1320x | 17.64 | 15.58 |
| 16 | 262,144 | 27.0733 ms | 26.9418 ms | 28.5176 ms | 28.5591 ms | 1.0015x | 19.28 | 19.25 |

Every B1 and B16 medoid met or exceeded its adjacent complete-call control; the minimum B16 ratio was `1.0001x`. The weighted B4 result passed comfortably, while the lowest-support B4 medoid remained at `0.9682x`. B16 body time remained `0.9951x` of HIP, but the required complete-call metric passed in both benchmark orders.

Six bounded route profiles, including sequential, repeated, sparse, skewed, and boundary layouts, matched the installed control bitwise. Their independent 64-column references had maximum absolute error at most `0.005859375`. Synthetic controls at all three production row counts were finite and bitwise exact against the installed pure-J64 or mixed-J64/J32 dispatch selected for that shape. Active-weight, workspace, input, and route mutations changed output; an inactive-expert mutation was inert; invalid expert and out-of-range offset routes left sentinel output untouched.

All three production artifacts rebuilt byte-identically and passed strict inspection as gfx1151 code-object v5, wave32 kernels with 116 VGPRs, 40 SGPRs, 30,720 bytes of LDS, 64 static WMMAs, four barriers, zero private storage, zero spills, no scratch instructions, no calls, and no dynamic stack. The focused grouped-forward file passed 49 tests. The broader grouped-plus-dense run passed 176 tests and reproduced only the five pre-existing Q6 full-HSACO-container hash failures; the corresponding Q6 source, `.text`, and resource assertions passed.

The retained research identity is `iq2_s_serial_full_weight_lds_64_linear_payload_prefetch()`. Intermediate BFE, selector, and quarter-scale identities were folded into this single final identity. Public dispatch, generated bundle tables, extension registration, packaging, and HIP fallback remain unchanged.

### Reopened routed prologue and direct-to-LDS capability experiments

IQ2_S shares the R1 packed-kernarg, R2 paired cumulative-offset, and guarded high-stride portion of R3. Their timing transfer is gated by Q4_K B1 because both use K512 and 32 output-column workgroups per route, while IQ2_S performs strictly more decode work. IQ2_S output addressing already uses a power-of-two shift, so only a legal shift-add combine remains format-local.

The coalesced full 64-row activation stage is a linear 9,216-byte copy with matching global and LDS offsets, making activation-only direct-to-LDS the sole structurally plausible R4 target. Packed IQ2_S weights are excluded because codebook/sign/scale decode transforms their representation, and partial activation tails would retain the existing bounds-masked path even if full-tile support existed.

### Reopened experiment results

The shared artifact and device matrix included IQ2_S R35 and every production key. R1-R3 rebuilt byte-identically in two independent passes, retained 116 VGPRs, 40 SGPRs, 30,720 LDS bytes, 64 static WMMAs, four barriers, zero private storage, and zero spills, and matched the parent bitwise on first, odd/even non-first, repeated-ID, boundary, skewed, mutation, invalid-expert, and invalid-offset routes.

The Q4_K B1 transfer screen measured R1 at +0.030% candidate time, R2 at -0.086%, and R3 at -0.083% relative to its generated parent. None cleared the greater-than-two-percent gate. IQ2_S adds decode work around the same route setup, so no IQ2_S timing transfer was run and R1-R3 are closed as neutral shared micro-optimizations.

The local RDNA 3.5 XML inventory names direct global/buffer-to-LDS encodings, but the configured gfx1151 assembler does not expose a usable compute-kernel instruction. rocISA-style buffer `dword`, `b32`, `dwordx4`, and `b128` spellings with an LDS destination failed with invalid operands. LLVM's `global_load_dword ... lds` spelling rejected the LDS operand, and canonical `global_load_lds_dword` reported that the instruction is unsupported on gfx1151. The corresponding LLVM gfx11 assembler test also classifies `global_load_lds_dword` as unsupported. R4 is therefore unsupported by the current target/toolchain, not an implementable IQ2_S candidate.

The reopened pass is closed: R1-R3 are rejected by the shared timing gate and R4 is target/toolchain-unsupported. The retained payload-prefetch identity, selected source, public dispatch, generated bundles, packaging, and HIP fallback remain unchanged. Durable scripts, reports, probe source, and independent artifacts are under `~/tmp/torch-ggml-ops/`.
