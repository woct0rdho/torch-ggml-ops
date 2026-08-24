# GGTensile grouped MMQ forward paired IQ2_S experiment

## Scope

This record covers isolated research kernels for the paired Qwen routed gate and up projections with two authoritative packed `IQ2_S` weight banks. The exact production contract is:
- 256 physical experts;
- two projections with output features `N = 512` and reduction features `K = 2048`;
- aggregate routed rows `R = 16,384`, `65,536`, or `262,144`;
- one device-resident `Q8_1` `F32_D4` activation workspace shared by both projections;
- two independent packed-weight banks with shape `[256,512,656]`;
- two independent BF16 destinations with shape `[R,512]`;
- device-resident int64 expert IDs and cumulative int32 route offsets;
- at most 256 route entries, with the final valid offset equal to `R`; and
- inert invalid experts and invalid cumulative ranges.

The research kernel must compute both projections in one workgroup dataflow. A single launch containing projection-indexed independent workgroups, two adjacent single-projection launches, or sequential full-projection phases that reload the activation workspace is not a paired arithmetic kernel.

Public dispatch, generated bundle tables, extension registration, packaging, and the HIP fallback remain out of scope until the paired artifact independently passes every gate in this record.

## Baseline and opportunity

The public `grouped_mmq_pair` call already quantizes the BF16 input once, creates one `Q8_1` workspace, creates two outputs, and reuses one route or row-task description. It still invokes the grouped multiplication body once per packed weight and output pair.

At B16, the fitted-prior complete call measured `66.0312 ms` against `60.2659 ms` for the BF16 AITER control. Comparator parity therefore requires a `9.57%` throughput improvement, equivalent to an `8.73%` latency reduction. Launch and routed-prologue fusion alone cannot provide that reduction.

For one 64-row, 64-column output tile, the `K = 2048` workspace contributes `64 * 16 * 144 = 147,456` activation bytes and each projection contributes `64 * 8 * 82 = 41,984` packed-weight bytes. Two independent bodies explicitly request 378,880 activation-plus-weight bytes. A dataflow-fused body requests 231,424 bytes, a `38.9%` reduction before cache effects. Weight decode, both WMMA streams, both FP32 accumulation streams, and both BF16 epilogues remain mandatory.

The planning estimate is a `4-8%` complete-call improvement, with `10-12%` an optimistic ceiling. These values are hypotheses, not advancement evidence.

## Candidate P1: four-wave K128 interleaving

The first candidate uses one 128-thread wave32 workgroup to own the same J64 row tile and 64 output columns in both projections. It keeps two independent 32-VGPR FP32 sum banks and advances in K128 phases:
- Stage one 9,216-byte coalesced activation plane into LDS.
- Decode the matching K128 half from the first packed bank into one reusable half-weight LDS image.
- Execute the first projection's WMMAs and exact FP32 correction into its sum bank.
- Retire all LDS consumers, overwrite the half-weight image from the second packed bank, and execute the second projection into its sum bank while the activation plane remains resident.
- Advance to the next K128 activation plane without changing either projection's reduction order.
- Convert and store the two sum banks independently to their authoritative BF16 destinations.

The current full-weight decoder assigns one workitem to one output row and one K128 half. P1 instead assigns two producer workitems to one output row for the selected half and divides the eight codebook groups between them. The mapping must keep all four waves useful, decode each packed group exactly once, and preserve the retained codebook, sign, scale, quarter-scale, payload-prefetch, and arithmetic semantics.

The intended physical point is approximately 148-156 VGPRs, 44 SGPRs, and about 20 KiB LDS. Physical planning, emitted metadata, and final artifact inspection are authoritative; these estimates may not be repaired silently during lowering.

## Candidate P2: device 64-row task ownership

P1 retains one workgroup per route entry and output-column tile, then advances serially through that route's 64-row tiles. The public B16 pair instead builds bounded 64-row tasks on the device and distributes those tasks across independent workgroups. P2 keeps P1's fused K128 arithmetic, two accumulator banks, one activation image, and one reusable half-weight image, but consumes the installed device-built task count, expert IDs, row starts, and row ends.

The task builder remains a separate metadata launch. It reads the authoritative int64 expert IDs and cumulative int32 offsets on the device, rejects malformed routes, and writes compact int32 task descriptors. The fused arithmetic launch reads one task, guards its device task index against the device task count, rebases both packed banks with the same expert, and writes both destinations. It does not read route values on the host or create CPU descriptors.

## Closed precursor designs

- Projection-indexed workgroups share only a launch and cannot reuse LDS data.
- Sequential full-projection phases cannot retain the full `K = 2048` activation workspace and therefore reload every activation plane.
- Two complete decoded-weight LDS images reproduce the rejected 52,224-byte J128 resource point and are not the first candidate.
- Eight waves split between projections reproduce the rejected J128 ownership topology and its one-workgroup residency and synchronization costs.
- A compiler-managed HIP body with two accumulator arrays is a semantic probe, not a GGTensile performance gate; existing HIP J128 IQ2_S candidates reach the VGPR limit and spill.

## Semantic authorities and ABI

The paired problem and solution identities must explicitly own two packed weights, two outputs, one activation workspace, one route description, one exact shape, and one paired projection count. P1 uses an 80-byte cumulative-route ABI. P2 uses a separate 96-byte device-row-task ABI with four task pointers. Both emitters load and validate their complete ABI, apply the same expert stride to both weight banks, preserve independent output bases, and leave invalid routes inert in both destinations.

The physical plan owns both sum banks, transient decode/MMA reuse, half-weight and activation LDS offsets, pointer registers, output addresses, barriers, and declared resources. Lowering may reuse the qualified IQ2_S arithmetic leaves, but it may not infer a second projection from singular forward state or mutate the existing non-paired identity.

## Qualification plan

Correctness precedes timing. Each candidate advances through these gates in order:
- Strict model, serialization, validation, physical-plan, ABI, and structural writer tests.
- Two independent source/object builds with byte-identical source and code objects.
- gfx1151 code-object v5, wave32, zero private storage, zero VGPR/SGPR spills, no scratch instructions, calls, or dynamic stack, with declared registers and LDS matching the physical plan.
- Bounded device comparison against adjacent installed single-projection controls for both outputs, followed by sequential, repeated, sparse, skewed, boundary, invalid-expert, and invalid-offset routes.
- Input and activation mutations must change both outputs. Mutating either active packed bank must change only its destination. Mutating an inactive expert in either bank must be inert. Sentinel output in invalid routes must remain untouched in both destinations.
- Complete-call timing against the public installed pair and against two adjacent research single-projection launches sharing the same fixed quantization and route ownership. This separates arithmetic-core movement from fusion itself.
- The B16 fitted-prior screen uses 512 draws reduced to five weighted medoids, three warmups, and nine alternating repeats. A candidate must improve its adjacent paired parent by more than two percent before B1/B4 transfer or reversed-order 25-repeat confirmation.

The timed path must keep route metadata on the device. It must not call `.item()`, copy offsets to the host, build CPU descriptors, or introduce an implicit synchronization. Complete-call time includes fixed HIP Q8_1 quantization and allocated workspace.

The final review repeats correctness matrices, complete-call confirmation, deterministic rebuild and resource inspection, focused and broad tests, changed-file hooks, and repository-hygiene checks. A candidate is not final until that review is complete.

## Experiment log

### Design and implementation start

- Confirmed that the public pair shares quantization and route-task setup but launches one single-projection body for each packed weight and output.
- Confirmed that no paired grouped-forward writer, ABI, artifact inspector, or research runtime launcher exists.
- Rejected launch-only, sequential-reload, eight-wave, and double-full-weight-LDS designs before implementation because they cannot share the required dataflow or reproduce already rejected resource ownership.
- Selected P1, the four-wave K128-interleaved half-weight design, as the sole first implementation candidate.

### P1 implementation and corrected launch qualification

P1 introduced strict paired problem and solution identities, physical planning, validation, an 80-byte two-weight/two-output ABI, route emission, K128-interleaved IQ2_S lowering, artifact inspection, and a research-only runtime. The half-granular decoder divides each selected IQ2_S K128 half across two producer lanes while preserving codebook, sign, scale, quarter-scale, payload-prefetch, WMMA, correction, and BF16 store order.

The final P1 artifact uses 148 VGPRs, 44 SGPRs, and 19,456 bytes of fixed LDS. It contains 128 static WMMAs and eight barriers and has zero private storage, register spills, scratch instructions, calls, or dynamic stack. Independent builds for the 35-row key and all three production row counts produced byte-identical source and code objects. The pre-existing selected non-paired IQ2_S source also remained byte-identical.

Real Qwen gate and up weights matched adjacent installed controls and the public pair bitwise on bounded first, odd, even, last, repeated, and tile-boundary routes. Active weight mutations affected only their owning destination, inactive-expert mutations were inert, activation mutations affected both destinations, deterministic reruns were exact, and malformed routes left sentinel output untouched.

The first P1 timing run was invalid because the research launcher passed the artifact's 19,456 fixed LDS bytes again as dynamic LDS. The dispatched group segment was therefore 38,912 bytes. The launcher now passes zero dynamic LDS, matching the established GGTensile runtime contract; installed HIP controls continue to receive their required dynamic shared storage. The invalid report remains explicitly labeled as such and is not advancement evidence.

With the corrected launch, the B16 nine-repeat screen measured `65.0525 ms` complete versus `67.4435 ms` for the public pair, or `1.0368x`. All five medoids were exact and faster, but the weakest ratio was only `1.0034x`. P1 cleared the aggregate advancement gate, then P2 superseded it before confirmation.

### P2 device-row-task candidate

P2 preserves P1's arithmetic body and resources while replacing serial route ownership with the existing device-built 64-row task distribution. Its 96-byte ABI contains two packed weights, one activation workspace, two destinations, four task pointers, four exact u32 shape values, and one u64 expert stride. Grid Y is the bounded task capacity; each workgroup exits when its task index is not below the device task count.

The first P2 device launch found a scalar-lifetime defect: `row_end` shared `s29` with activation-plane stride and was overwritten before the row loop. Moving `row_end` to the now-dead `nrows_weight` shape-guard register preserved the 44-SGPR plan. The corrected bounded artifact matched installed row-task controls and the public pair bitwise.

P2 independently rebuilt byte-identically for rows 35, 16,384, 65,536, and 262,144. Strict inspection reports gfx1151 code-object v5, wave32, a 96-byte kernarg segment, 148 VGPRs, 44 SGPRs, 19,456 bytes fixed LDS, 128 WMMAs, eight barriers, zero private storage, zero spills, no scratch instructions, no calls, and no dynamic stack.

The bounded semantic matrix used real Qwen IQ2_S gate and up banks. Four route profiles matched both installed and public controls bitwise and deterministic reruns were exact. Independent dequantized 64-column references for both projections measured normalized RMSE from `0.00592` through `0.00648`, with maximum absolute error at most `0.015625`. Active first- and second-bank mutations changed 4,561 and 4,565 values only in their respective destinations. An inactive expert was inert, and activation mutation changed 17,785 and 17,801 values. Invalid expert, out-of-range offset, negative previous offset, and empty-route cases produced zero, one, or two valid device tasks as appropriate and preserved sentinel rows in both outputs.

The nine-repeat B16 screen measured `60.0484 ms` for P2 versus `67.8355 ms` for the public pair, or `1.1297x`; every medoid was at least `1.1255x`. P2 therefore advanced to reversed-order confirmation and independent B1/B4 transfer.

### Final qualification and retention

Reversed-order 25-repeat confirmation passed at every production shape. The final table reports prequantized multiply-only throughput. HIP and GGTensile consume the same activation workspace produced by the shared HIP quantizer, so activation quantization and other complete-call work are excluded. Nominal dense-equivalent throughput is `2 * aggregate rows * N * K / time`, doubled for paired two-projection kernels. GGTensile/HIP speedup is HIP body time divided by GGTensile body time.

| Aggregate rows | Public catalog hash | HIP TFLOPS | GGTensile TFLOPS | GGTensile/HIP speedup |
| ---: | :--- | ---: | ---: | ---: |
| 16,384 | `ggpair_a04b4d0379080838` | 11.420 | 13.786 | 1.2072x |
| 65,536 | `ggpair_52d8b2d6a398c27a` | 15.935 | 18.511 | 1.1616x |
| 262,144 | `ggpair_59e04f7993fc9621` | 17.783 | 20.347 | 1.1442x |

Every one of the fifteen fitted-prior outputs matched both adjacent controls bitwise. Minimum per-medoid public/P2 ratios were `1.1275x`, `1.1198x`, and `1.1242x` at B1, B4, and B16. The B16 candidate is within about one percent of the earlier `60.2659 ms` BF16 AITER comparator, rather than the original packed path's roughly nine-percent deficit.

The independent `uniform`, `skewed`, `sparse`, and `boundary` timing controls were also exact at all three production shapes. Equal-weight public/P2 ratios were `1.1906x`, `1.1400x`, and `1.1324x` at B1, B4, and B16. The minimum individual synthetic ratios were `1.1466x`, `1.1279x`, and `1.1267x`; no route control approached the retention floor.

The quantizer's authoritative K2048 F32_D4 tensor shape is `[16,R,144]`: sixteen K-plane blocks, aggregate rows, and 144 bytes per block. Both projections reuse that one allocation. P2 also uses one bounded task allocation whose capacity is `ceil(R/64) + route_entries`; neither timed research path reads its device task count on the host.

The focused paired module passes 12 tests. The complete GGTensile suite passes 422 tests, and the full repository suite passes 523 tests with 14 pre-existing Python 3.14 deprecation warnings. Ruff, formatting, Python compilation, `ty check`, deterministic source checks, and diff whitespace checks pass.

The retained research identity is `iq2_s_k128_interleaved_row_tasks()`. P1 remains a qualified serial-route precursor, not the selected production-shape candidate. Public selectors, generated bundle tables, extension registration, packaging, and the HIP fallback remain unchanged. Promotion now requires a separate integration campaign that preserves the public fallback and independently qualifies generated artifacts, package ownership, registration, complete public-call timing, and deployment behavior.

## Reopened source-level optimization review

The P1 and P2 qualification remains valid for the measured serial and row-task identities and their original schedules. A later audit of the paired emitter found bounded code-generation probes that were not included in those measurements. None below is implemented, timed, or retained.

### P3 paired zero-accumulator lifetime

All three paired families currently reinitialize the dedicated read-only `v124:v131` zero bank before each projection phase. The paired physical plan gives that bank a lifetime spanning row setup and compute, and `emit_signed_i8_wmma` consumes it as the WMMA accumulator without an identified intervening clobber. P3 would initialize it once before the K-block loop while preserving the existing waits, barriers, LDS image reuse, projection-specific scales, and output stores.

The current repeated body contains 24 initialization moves that can be removed from the repeated projection schedule while retaining one eight-register initialization. At K2048 the amortized dynamic reduction is approximately 248 moves per row tile after retaining that setup. The static repeated-body count, the per-projection count, and the amortized row-tile count must be reported independently; none is a timing claim. Serial-route and device-row-task ownership are separate P3 identities and require separate exact qualification.

### P4 paired epilogue sharing

The two projections recompute equivalent vector column extraction, wave/workgroup offsets, row masks, and vector offsets even though their output base pointers remain independent. A first probe may share only the six identical column-setup instructions. A broader probe may materialize per-fragment addresses in dead `c` registers, with an estimated source reduction of about 25 vector instructions, but it requires an explicit destination-aliasing and store-order contract. These are separate identities and must not silently merge the two output pointers.

P5 is an exact BF16 RNE scheduling probe. The existing `v_bfe_u32` plus `v_add3_u32` helper must remain unchanged semantically; two dead scratch VGPRs may interleave independent chains for the two projections if the physical plan proves their lifetimes. The work occurs once per output tile, so it is lower priority than P3. Approximate conversion, relaxed rounding, and changed accumulation are excluded.

### P6 isolated ISA and metadata screens

The paired control emits `s_clause 7` and explicit wait/barrier structure. Clause-boundary changes, compiler-produced `s_delay_alu`, and bank-valid GFX11 VOPD pairings may be screened only as independent artifact experiments. VOPD eligibility must use the exact gfx1151 instruction list and register-bank masks; `v_bfe_u32` is not assumed to be a legal generic VOPD operand. The reusable LDS image creates an overwrite hazard, so activation waits or barriers are not removed by inference from `BackOffBarrier` or compiler behavior.

The HIP paired control contains `.amdhsa_workgroup_processor_mode 1`, while the inspected GGTensile paired artifacts do not. A metadata-only A/B is a separate P6 identity. It has no assumed performance benefit and must first prove identical ABI, code-object, resource, correctness, and deterministic-build behavior before timing is considered.

### Qualification and recursive review

Each P3-P6 probe requires a distinct typed identity and serialized policy, exact gfx1151 code-object-v5 assembly and linking, ABI and metadata inspection, resource and disassembly inspection, independent exactness and mutation checks, two deterministic builds, and warmed prequantized and complete-call timing against both the installed public control and the ownership-matched parent. Any composed candidate receives a new identity and repeats the full gate.

The historical final classification is scoped to P1/P2 and their old premise. Implement and qualify every actionable finding, then repeat the complete source, artifact, resource, correctness, determinism, and timing review from the changed premise. Completion is valid only after a fresh recursive pass finds no actionable in-contract mechanism. Public selectors, generated bundles, packaging, registration, and HIP fallback remain outside this experiment.

### P3 qualification result

The row-task P3 probe moved the dedicated `v124:v131` initialization before the K2048 block loop and removed the other three statically emitted sets. The current typed parent was first verified instruction-for-instruction against retained P2 after normalizing the already-qualified symbol-only identity change. Independent candidate builds were byte-deterministic. The artifact changed from 3,296 to 3,272 VALU issue instructions while retaining 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, zero spills, and zero private bytes.

The full R35 route, malformed-route, independent-reference, rerun, active/inactive weight mutation, projection-isolation, and activation-mutation matrix passed exactly. Direct same-session A/B timing against the rebuilt instruction-identical P2 parent also matched both BF16 outputs at every fitted medoid. Weighted candidate/parent complete-call medians were 5.5457/5.5480 ms at B1, 16.1289/16.1252 ms at B4, and 59.9276/59.9965 ms at B16, equivalent to +0.04%/-0.02%/+0.11% candidate movement. Prequantized body movement was -0.12%/+0.05%/+0.17%.

P3 is rejected as timing-neutral. Its complete-call and body directions are mixed, and the observed movement is at most 0.17%, so the closure does not depend on the obsolete fixed two-percent threshold and no uncertainty top-up is warranted. No source integration or serial-ownership transfer follows.

### P4 and P5 qualification results

P4's shared-column form removed six artifact VALU issues, from 3,296 to 3,290, and its broader materialized-address form removed 25, to 3,271. Both rebuilt deterministically at R35 and every production row count, retained 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, 77 waits, zero private bytes, and zero spills, and passed the complete route, malformed-route, reference, rerun, projection-isolation, and mutation matrices.

Shared-column complete-call parent/candidate ratios were `0.99902x`, `0.99984x`, and `0.99881x` at B1/B4/B16; body ratios were `0.99594x`, `1.00236x`, and `1.00010x`. Materialized-address complete ratios were `0.99695x`, `0.99842x`, and `0.99987x`; body ratios were `0.98697x`, `0.99936x`, and `1.00133x`. The latter regresses the dominant B1 body by 1.30%, and neither form has coherent complete/body direction. Both P4 forms are closed without source retention.

P5 preserved the parent's instruction and resource counts while interleaving adjacent exact `v_bfe_u32`/`v_add3_u32` BF16 RNE chains through proven-dead scratch. It was deterministic and passed the same exact semantic matrix. Complete-call ratios were `1.00562x`, `1.00185x`, and `0.99830x`; body ratios were `0.99304x`, `1.00189x`, and `1.00139x`. The endpoint complete/body disagreement closes P5 as timing-incoherent. No approximate conversion or relaxed rounding was introduced.

### P6 target-level disposition

The processor-mode spelling was finally applied to current P2 row-task sources at R35, B1, B4, and B16. Two independent passes were deterministic, and every transformed object and HSACO was byte-identical to its parent. Inspection consequently remained at 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, zero private bytes, zero spills, and the exact 96-byte ABI. The durable report is `ggtensile-grouped-iq2-s-pair-p6-processor-mode-build.json`. Processor mode is executable-inert for this exact toolchain artifact, so no timing or typed identity follows.

No wait or barrier candidate is admitted: the eight barriers protect projection-overwritten weight LDS and activation reuse, and no exact gfx1151 hazard proof permits removal. The target review found no changed producer/consumer schedule requiring `s_delay_alu` and no independent bank-valid GFX11 VOPD pairing premise; `v_bfe_u32` remains ineligible. Clause spelling has no changed store or VMEM ordering premise. These are target/dependency exclusions rather than inferred timing claims.

P3-P6 are now resolved. A separate read-only pass over the restored P2 lowering, physical lifetimes, exact artifacts, and paired cross-family results found no new actionable IQ2_S mechanism. Public selectors, bundles, packaging, registration, and HIP fallback remain unchanged.
