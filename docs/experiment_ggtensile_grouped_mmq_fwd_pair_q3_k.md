# GGTensile grouped MMQ forward paired Q3_K experiment

## Scope

This record covers isolated GGTensile research kernels for Qwen's paired routed Q3_K gate and up projections. The exact production contract is:
- 256 physical experts;
- output features `N = 512` and reduction features `K = 2048` for both projections;
- aggregate routed rows `R = 16,384`, `65,536`, or `262,144`;
- two authoritative packed Q3_K banks with shape `[256,512,880]`;
- one fixed Q8_1 `F32_D4` activation workspace with shape `[16,R,144]` shared by both projections;
- two independent BF16 destinations with shape `[R,512]`;
- device-resident int64 expert IDs and cumulative int32 route offsets;
- at most 256 route entries, a final valid offset equal to `R`, and inert invalid routes.

A paired candidate must compute both projections through one workgroup arithmetic dataflow. Projection-indexed workgroups, adjacent single-projection launches, and sequential full-projection phases that reload activation data are controls, not fusion.

Public selectors, generated bundles, extension registration, packaging, and HIP fallback behavior remain outside this research campaign. Promotion requires a separate integration campaign after the research identity passes every gate below.

## Baseline and priority

The public `grouped_mmq_pair` path quantizes once and shares one activation allocation, then launches the existing single-projection grouped body once per packed bank and destination. At the prior fitted-route medoid, public complete-call time was `6.0770 ms`, `16.5897 ms`, and `58.2301 ms` for B1, B4, and B16. The corresponding BF16 AITER times were `10.8056 ms`, `25.1215 ms`, and `55.9767 ms`. B16 therefore has the only comparator deficit, while every shape still pays duplicate activation reads in the two packed bodies.

For one 64-row by 64-column output tile over full `K = 2048`, the Q8_1 workspace contributes `64 * 8 * 144 * 2 = 147,456` activation bytes across eight Q3_K blocks and two K128 halves per block. Each projection contributes `64 * 110 = 7,040` packed bytes per block before transaction alignment. A fused body can load each activation half once while retaining two independent accumulator banks. Q3 high-mask reconstruction, signed six-bit scales, both WMMA streams, both FP32 corrections, and both epilogues remain mandatory.

The largest performance gap is B16, so advancement starts there after bounded correctness. B1 and B4 transfer only after the B16 candidate beats its adjacent public and ownership-matched controls by more than two percent.

## Candidate Q1: selected-half K128 serial routes

Q1 uses one 128-thread wave32 workgroup to own 64 routed rows and 64 output columns in both projections. It keeps two independent 32-VGPR FP32 sum banks and one 9,216-byte activation image. For each K128 half it:
- decodes the selected Q3_K half from the first packed bank into one reusable 10,240-byte LDS weight image;
- stages the matching activation plane once;
- computes and corrects the first projection;
- overwrites only the weight image from the second packed bank;
- computes and corrects the second projection while the activation image remains resident.

Two producer lanes own each output row. They reconstruct the selected low two-bit payload with the matching high-mask bytes, decode the eight signed six-bit group scales with the FP16 block factor, and write one 128-byte int8 payload plus eight FP32 scales into a 160-byte padded LDS row. The compute order remains eight signed integer WMMAs per K128 half and M fragment, followed by the qualified FP32 correction and BF16 RNE store order.

Q1 uses the existing 80-byte cumulative-route paired ABI. It is a semantic and serial-ownership control because large or imbalanced routes remain resident in one workgroup per output tile.

## Candidate Q2: device 64-row tasks

Q2 preserves Q1's arithmetic, register plan, LDS plan, and output ownership. It consumes the installed device-built 64-row task arrays through the existing 96-byte paired row-task ABI. Grid Y is the bounded capacity `ceil(R/64) + route_entries`; every workgroup exits when its task index is not below the device task count. No timed path reads route values or task counts on the host.

The adjacent control is one installed task-setup launch followed by two installed Q3_K row-task bodies on the same activation workspace. The public pair is also required in every benchmark rotation.

## Qualification gates

Correctness precedes timing:
- Strict typed problem, solution, enum, serialization, validation, physical-plan, ABI, and writer tests.
- Byte-identical regeneration of both Q3_K candidates and byte-identical preservation of both existing IQ2_S paired sources.
- gfx1151 code-object v5, wave32, zero private storage, zero spills, no scratch instructions, calls, or dynamic stack, with emitted resources equal to the physical plan.
- Bitwise agreement for both destinations against adjacent installed controls and public dispatch on first, odd, even, last, repeated, sparse, skewed, boundary, and expert-255 routes.
- Active mutations of either packed bank affect only its destination; activation mutations affect both; inactive experts are inert; malformed routes preserve sentinels.
- Bounded independent dequantized references for both projections rather than a full model-sized dequantization.
- Complete-call timing includes fixed HIP Q8_1 quantization, allocated workspace, device task setup when applicable, and both projections.
- The fitted prior uses 512 deterministic draws, five weighted medoids, three warmups, nine order-rotated repeats, and reversed-order 25-repeat confirmation after advancement.
- Synthetic uniform, skewed, sparse, and boundary controls must preserve correctness and the retained performance direction at all three production row counts.

The inherited final-review rule applies. A fresh review must reread this record, the paired IQ2_S and dense Q3_K records, the retained source and disassembly, installed HIP controls, target ISA behavior, generated artifacts, benchmark evidence, and failed candidates. Any actionable in-contract finding is implemented and qualified before the review repeats; completion is valid only when a fresh pass finds none.

## Experiment log

### Initial implementation and artifact gate

The first implementation adds distinct Q3_K serial-route and device-row-task identities without changing the existing IQ2_S identities. Both Q3_K artifacts derive 148 VGPRs, 44 SGPRs, and 19,456 bytes fixed LDS. They contain 128 static integer WMMAs and eight barriers with zero private storage and zero register spills. The serial ABI is 80 bytes and the row-task ABI is 96 bytes.

The 35-row boundary probe used real `blk.0.ffn_gate_exps.weight` and `blk.0.ffn_up_exps.weight` tensors. Q1 and Q2 matched each other, their ownership-matched installed controls, and public `grouped_mmq_pair` bit-for-bit for both projections on routes containing experts 0, 7, 12, and 255. The device task builder produced four bounded tasks. Both outputs were finite.

The first direct-control probe incorrectly allocated 29,952 dynamic LDS bytes by counting 32 activation payload integers per row. The installed HIP tile owns 36 integers per row after Q8_1 metadata, so its authoritative J64 allocation is 30,976 bytes. That invalid setup differed in 356 BF16 values per projection. Correcting only the control launch allocation restored bitwise agreement; the GGTensile artifacts continue to launch with zero dynamic LDS because their 19,456 bytes are statically declared.

### Artifact qualification

The Q3_K serial-route and device-row-task identities were built independently twice for `R = 35`, `16,384`, `65,536`, and `262,144`. Each source and code object pair was byte-identical across the independent builds. The 35-row boundary artifact is 28,768 code bytes; each production artifact is 28,776 code bytes. The production build records contain 3,884 source instructions, 32 global loads, 16 scalar loads, 42 LDS writes, 128 static WMMAs, eight barriers, and two independent projection stores.

All four production artifacts are gfx1151 code-object v5, wave32, with a 96-byte row-task kernarg segment, 148 VGPRs, 44 SGPRs, 19,456 bytes of fixed LDS, zero private storage, zero VGPR or SGPR spills, no scratch instructions, no calls, and no dynamic stack. GGTensile artifacts launch with zero dynamic LDS. The installed HIP Q3_K J64 controls remain a separate ownership-matched control and require 30,976 dynamic LDS bytes because their tile owns 36 Q8 integers per row including Q8_1 metadata.

### Semantic qualification

The R35 semantic matrix used the authoritative `blk.0.ffn_gate_exps.weight` and `blk.0.ffn_up_exps.weight` tensors. Four exact routes containing experts 0, 7, 12, and 255 matched both installed Q3_K row-task controls and public `grouped_mmq_pair` bit-for-bit for both destinations. Reruns were exact. The bounded independent dequantized 64-column references measured normalized RMSE from `0.00586` through `0.00624`, with maximum absolute error `0.009765625`.

Mutating an active first-bank slice changed 2,882 values only in the first destination; mutating the corresponding second-bank slice changed 2,883 values only in the second destination. Mutating the activation changed 8,448 and 8,441 values in the two destinations. Expert 42 was inert. Invalid expert, offset-past-end, negative-previous-offset, and empty-route cases matched installed behavior and preserved every specified sentinel interval.

The production-row semantic runner then used repeated and boundary routes at all three retained row counts. Valid outputs matched installed controls, public dispatch, and deterministic reruns bit-for-bit, and both projections were finite. The malformed invalid-expert route preserved its inert interval at every shape while leaving the valid tail active. The valid task counts were 258, 1,026, and 4,098 for `R = 16,384`, `65,536`, and `262,144`; the malformed counts were 192, 768, and 3,072. No host route descriptor or task-count read is part of the timed path.

### Performance qualification

The B16 nine-repeat fitted-prior screen measured `54.5254 ms` complete for P2 versus `58.0474 ms` for public and `58.0001 ms` for the adjacent row-task parent. All five medoids were exact; the weakest ratios were `1.0598x` against public and `1.0597x` against the parent, clearing the more-than-two-percent advancement gate. B1 and B4 transfer screens measured `4.8264 ms` and `14.7843 ms` against public `5.7386 ms` and `16.1566 ms`, respectively.

Reversed-order 25-repeat confirmation passed at every production shape. Timings are weighted medians over 512 deterministic draws reduced to five weighted medoids, with three warmups and order rotation. Complete-call measurements include fixed HIP Q8_1 F32_D4 quantization, one allocated workspace, device task setup, and both projections. The row-task parent is one task setup followed by two adjacent installed Q3_K J64 row-task launches.

The public pair is the installed HIP control. This is the P2-era throughput table; the typed Q3-2 speeds below are the current retained result. Effective TFLOPS is nominal dense-equivalent complete-call throughput, calculated as `4 * rows * N * K / seconds`: two FLOPs per FMA across both projections.

| Batch | Rows | P2 body | Row-task parent body | P2 complete | Public HIP complete | Parent/P2 | HIP/P2 speedup | P2 effective TFLOPS | HIP effective TFLOPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16,384 | 4.4513 ms | 5.3532 ms | 4.8117 ms | 5.7575 ms | 1.1806x | 1.1966x | 14.28 | 11.94 |
| 4 | 65,536 | 12.7549 ms | 14.0354 ms | 14.6721 ms | 15.9899 ms | 1.0892x | 1.0898x | 18.73 | 17.19 |
| 16 | 262,144 | 48.1296 ms | 51.5795 ms | 54.8612 ms | 58.3158 ms | 1.0635x | 1.0630x | 20.04 | 18.85 |

The weakest per-medoid public/P2 ratios in the 25-repeat confirmations were `1.0838x`, `1.0641x`, and `1.0584x` at B1, B4, and B16. Thus the retained Q3 candidate beats both the public HIP pair and its ownership-matched adjacent parent at every fitted production shape.

Uniform, skewed, sparse, and boundary synthetic controls were exact at all three production row counts. Their weighted public/P2 ratios were `1.1667x`, `1.0800x`, and `1.0622x` for B1, B4, and B16. The minimum individual public ratios were `1.0817x`, `1.0619x`, and `1.0603x`; the minimum adjacent-parent ratios were `1.0587x`, `1.0599x`, and `1.0575x`. P2 retained the same direction for every synthetic route family.

### Failed and closed candidates

Dense Q3_K mechanical pairing was rejected before implementation: the existing 200-VGPR, 39,936-byte LDS dense point cannot accommodate a second 64-VGPR accumulator bank. The selected-half 64-row topology was the bounded alternative that preserved two output accumulators and remained at the qualified 148-VGPR, 19,456-byte point.

The first installed-control probe used 29,952 dynamic LDS bytes and differed in 356 BF16 values per projection. That control was invalid because the HIP J64 tile owns 36 Q8 integers per row; correcting the launch to 30,976 bytes restored bitwise agreement. Direct-to-LDS remains closed after the configured gfx1151 assembler rejected both the instruction form and the target capability. No further Q3 arithmetic or scheduling candidate was retained after P2 cleared the fitted, confirmation, transfer, and synthetic performance gates.

### Final repository qualification

The focused paired suite passes 22 tests, the complete GGTensile suite passes 432 tests, and the full repository suite passes 533 tests with only the 14 known Python 3.14 `torch.jit.script_method` deprecation warnings. Ruff, formatting, Python compilation, `ty check`, and `git diff --check` pass. The retained Q3 source and artifacts were rebuilt and inspected independently, and the fresh review found no actionable implementation, ABI, ISA, resource, runtime, benchmark, or documentation finding.

The retained research identities are `q3_k_k128_interleaved()` and `q3_k_k128_interleaved_row_tasks()`. Public selectors, generated bundle tables, extension registration, packaging, and HIP fallback behavior remain unchanged. Promotion remains a separate integration campaign with its own generated-artifact, registration, packaging, public-call, fallback, and deployment qualification.

## Reopened source-level optimization review

The P2 qualification and its conclusion that no candidate was actionable remain scoped to the previously emitted Q3_K paired body. A later source review found two exact local probes and several bounded artifact screens that were not part of P2. They are pending and have no new timing result.

### Q3-1 paired zero-accumulator lifetime

The paired physical plan assigns `v124:v131` as a read-only zero accumulator after row setup, but each projection rewrites it before WMMA. Q3-1 initializes the bank once before the K2048 block loop and preserves all waits, barriers, LDS reuse, scales, corrections, and output ownership. The repeated body loses 24 static initialization moves and the amortized dynamic reduction is approximately 248 moves per K2048 row tile after retaining one eight-register setup. These counts must be reported as static, per-projection, and amortized values, not as timing.

### Q3-2 variable-offset BFE scale extraction

Each Q3_K scale field currently uses `v_lshrrev_b32` followed by `v_and_b32`. Q3-2 replaces each pair with one variable-offset `v_bfe_u32`. Source semantics are exact for the unsigned field extraction, with an estimate of eight fewer instructions per half decode, 32 per K256 block, and approximately 256 per K2048 row tile. The variable-offset encoding, target support, register latency, disassembly, and generated artifact must be inspected; source equivalence alone is insufficient. `v_bfe_u32` is not treated as a generic VOPD operand.

### Q3-3 epilogue and Q3-4 metadata screens

The two projections recompute equivalent column, mask, and vector-offset setup while retaining independent output bases. Q3-3 may share only proven-identical setup first, then separately test exact width-two BF16 RNE chain interleaving if dead scratch VGPR lifetimes and store order permit it. The existing `v_bfe_u32`/`v_add3_u32` conversion remains the arithmetic contract; no relaxed rounding is allowed.

The HIP paired control carries `.amdhsa_workgroup_processor_mode 1`, absent from the inspected GGTensile paired artifacts. Q3-4 is an isolated metadata A/B with no presumed performance effect. The existing `s_clause 7`, waits, and barriers remain controls. Clause changes, compiler-produced `s_delay_alu`, and bank-valid GFX11 VOPD pairings are separate artifact screens, and LDS overwrite hazards prevent wait or barrier deletion by inference.

### Qualification and recursive review

Q3-1 through Q3-4 require distinct typed identities and serialized policies, exact gfx1151 code-object-v5 assembly and linking, ABI and metadata inspection, resource and disassembly inspection, independent exactness and mutation checks, deterministic rebuilds, and warmed prequantized and complete-call timings against the public and ownership-matched controls. Any combined identity repeats the full qualification order. Static or amortized instruction estimates are not timing claims.

The historical P2 final classification remains scoped to the old body and schedule. Implement and qualify every actionable finding, then repeat the complete source, artifact, resource, correctness, determinism, and timing review from the changed premise. Completion requires a fresh recursive pass with no actionable in-contract mechanism. Public selectors, generated bundles, packaging, registration, and HIP fallback remain outside this experiment.

### Q3-1 qualification result

Q3-1 moved the dedicated `v124:v131` initialization before the K2048 block loop and removed the other three statically emitted sets. The current typed parent was verified instruction-for-instruction against retained P2 after normalizing the already-qualified symbol-only identity change. Independent candidate builds were byte-deterministic. The artifact changed from 3,000 to 2,976 VALU issue instructions while retaining 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, zero spills, and zero private bytes.

The full R35 route, malformed-route, independent-reference, rerun, active/inactive weight mutation, projection-isolation, and activation-mutation matrix passed. Production-row route and malformed-route checks also passed at B1, B4, and B16. Direct same-session A/B timing against the rebuilt instruction-identical P2 parent matched both BF16 outputs at every fitted medoid. Weighted candidate/parent complete-call medians were 4.9995/4.9998 ms at B1, 14.5857/14.5849 ms at B4, and 54.5148/54.4689 ms at B16, equivalent to +0.01%/-0.01%/-0.08% candidate movement. Prequantized body movement was -0.28%/-0.07%/+0.16%.

Q3-1 is rejected as timing-neutral. Complete-call movement is effectively zero, body direction is mixed, and the largest observed movement is 0.28%, so the closure does not depend on the obsolete fixed two-percent threshold and no uncertainty top-up is warranted. No source integration or serial-ownership transfer follows. Q3-2 through Q3-4 remain independent and must not compose this rejected change.

### Q3-2 qualification result

Q3-2 replaced each of the 32 statically emitted Q3 scale-field `v_lshrrev_b32` plus `v_and_b32` pairs with one variable-offset `v_bfe_u32`. The configured gfx1151 assembler accepted the exact operand form and independent builds were byte-deterministic. The artifact changed from 3,000 to 2,968 VALU issue instructions while retaining 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, zero spills, and zero private bytes.

The full R35 route, malformed-route, independent-reference, rerun, active/inactive weight mutation, projection-isolation, and activation-mutation matrix passed. Production-row route and malformed-route checks also passed at B1, B4, and B16. Direct same-session A/B timing against the instruction-identical P2 parent matched both BF16 outputs at every fitted medoid. Weighted candidate/parent complete-call medians were 4.8530/4.9415 ms at B1, 14.6109/14.7620 ms at B4, and 54.4895/54.8696 ms at B16, or gains of 1.82%/1.03%/0.70%. Prequantized body gains were 1.33%/0.80%/1.36%. The weakest complete-call medoid ratios were `1.00073x`, `1.00364x`, and `1.00488x`.

Q3-2 is rejected because no shape cleared the strict two-percent advancement gate. The directional movement is retained as evidence, but no 25-repeat confirmation, source integration, or composition with Q3-1 follows. Q3-3 and Q3-4 remain independent.

### Q3-2 stability-policy reopening

The fixed two-percent gate was subsequently withdrawn for candidates with stable evidence of a smaller gain. A same-session 25-repeat confirmation with five warmups retained exact outputs at every medoid. Weighted candidate/parent complete-call medians were 4.9059/4.9525 ms at B1, 14.6099/14.7282 ms at B4, and 54.5834/55.0632 ms at B16, or gains of 0.95%/0.81%/0.88%. Every one of the fifteen medoids was faster; the minimum ratios were `1.00478x`, `1.00648x`, and `1.00819x`. Prequantized body gains were 0.91%/0.82%/1.05%, and every body medoid was also faster.

A conservative independent-sample 95% log-time analysis found nine of fifteen complete-call medoids and twelve of fifteen body medoids individually below parity; all remaining intervals overlapped parity narrowly and none favored the parent. The coherent direction across both timing layers, every shape, and every medoid is stable evidence of a small gain. Q3-2 is therefore reclassified as a retained research mechanism pending typed-source requalification. This reopening does not compose Q3-1 and does not change public dispatch, bundle registration, packaging, or HIP fallback behavior.

### Q3-2 typed retention

The retained source identity serializes `TwoLaneSelectedHalfQ3VariableBFE` as the decode schedule and admits it only for Q3_K device-row-task ownership. Baseline Q3_K, IQ2_S, and IQ2_XXS identities retain their prior schedules. The typed lowering reproduced the qualified probe instruction-for-instruction after canonical symbol normalization. Two typed builds were byte-deterministic and retained 2,968 VALU issue instructions, 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, zero spills, and zero private bytes.

The typed R35 route/reference/mutation matrix and all production-row route and malformed-route checks passed exactly. A fresh same-session 25-repeat typed A/B measured candidate/parent complete-call medians of 4.9006/4.9437 ms at B1, 14.5887/14.7411 ms at B4, and 54.6028/55.0552 ms at B16, or gains of 0.88%/1.04%/0.83%. Body gains were 0.98%/0.89%/1.02%. Every complete-call and body medoid was faster; minimum complete-call ratios were `1.00571x`, `1.00640x`, and `1.00738x`. Conservative 95% log-time intervals excluded parity for twelve of fifteen complete-call medoids and thirteen of fifteen body medoids; none of the remaining intervals favored the parent.

Q3-2 is retained as `q3_k_k128_interleaved_row_tasks_variable_bfe()`. It remains a research identity only: public dispatch, generated bundle registration, packaging, prepared representations, and HIP fallback behavior are unchanged. Any composition with Q3-1, Q3-3, or Q3-4 requires a new identity and full requalification.

### Q3-3 epilogue qualification result

Q3-3 was evaluated against retained typed Q3-2 as three independently materialized subprobes. Shared column setup removed six artifact VALU issues, from 2,968 to 2,962, with unchanged 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, waits, private storage, and spills. It passed the R35 and production route/reference/mutation matrices. Nine-repeat complete-call parent/candidate ratios were `0.99852x`, `1.00189x`, and `1.00059x` at B1/B4/B16; body ratios were `0.99946x`, `0.99988x`, and `1.00262x`. The mixed direction closes this narrow form.

The broader materialized-address form carried common column and row terms in registers proven dead after accumulation. It removed 25 VALU issues, from 2,968 to 2,943, without changing any resource or synchronization count. It preserved the Q3_K N512 row shift of 10 and passed the complete exact matrix. Complete-call ratios were `0.99765x`, `1.00515x`, and `1.00162x`; body ratios were `0.99921x`, `1.00031x`, and `0.99998x`. The apparent B4 complete gain does not transfer to the body or other shapes, so the candidate is timing-incoherent.

The exact width-two BF16 form interleaved adjacent `v_bfe_u32`/`v_add3_u32` RNE chains through a proven-dead scratch register. Builds and outputs were deterministic and exact, with unchanged instruction and resource counts. Complete-call ratios were `0.99711x`, `1.00218x`, and `1.00115x`; body ratios were `1.00394x`, `0.99797x`, and `1.00184x`. Complete/body disagreement again closes the candidate. These three results exhaust Q3-3's allowed shared-setup, address-materialization, and exact BF16-scheduling mechanisms; no source identity or composition is retained.

### Q3-4 processor-mode metadata result

Q3-4 added `.amdhsa_workgroup_processor_mode 1` to retained Q3-2 at R35, B1, B4, and B16. The configured gfx1151 assembler accepted the spelling, and two independent passes were deterministic. At every row count, the candidate object and linked HSACO were byte-identical to the parent. ABI, 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, 2,968 VALU issues, VMEM/LDS/wait/clause counts, private storage, and spills were necessarily unchanged.

Q3-4 is closed as executable-inert under this exact assembler, linker, target, and code-object-v5 artifact. No timing can distinguish byte-identical executables, and no serialized schedule, typed identity, source retention, or public integration follows. This does not predict behavior under another toolchain or target.

### Post-Q3-4 recursive final review

A separate read-only pass over retained Q3-2, its row-task physical plan, decode and epilogue lowerings, generated artifacts, Q3-1 through Q3-4, dense Q3, and the other paired formats found no new exact Q3_K mechanism. Zero-bank motion is timing-neutral, variable BFE is retained, all bounded epilogue forms are timing-incoherent, and processor mode is executable-inert. The reusable LDS image still requires its waits and barriers, and no bank-valid VOPD or exact BF16 replacement supplies a changed premise. No actionable in-contract paired Q3_K mechanism remains; public integration stays deferred.

### Final retained throughput versus HIP

The final table reports prequantized multiply-only throughput. HIP and GGTensile consume the same activation workspace produced by the shared HIP quantizer, so activation quantization and other complete-call work are excluded. Nominal dense-equivalent throughput is `2 * aggregate rows * N * K / time`, doubled for paired two-projection kernels. GGTensile/HIP speedup is HIP body time divided by GGTensile body time.

| Aggregate rows | Public catalog hash | HIP TFLOPS | GGTensile TFLOPS | GGTensile/HIP speedup |
| ---: | :--- | ---: | ---: | ---: |
| 16,384 | `ggpair_0aed3be2b92c0f07` | 12.810 | 15.525 | 1.2120x |
| 65,536 | `ggpair_5a0b32e2a6c0ec93` | 18.873 | 20.987 | 1.1120x |
| 262,144 | `ggpair_6d9aa5f305a16663` | 21.115 | 22.874 | 1.0833x |

All route, malformed-route, mutation, and finite-output checks were exact.
