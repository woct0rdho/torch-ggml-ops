# GGTensile Dense MMQ Forward Q4_K Plan

## Purpose

Build a strict gfx1151 wave32 GGTensile assembly campaign for the Qwen Q4_K dense-MMQ-forward production inventory. The result must consume the authoritative packed GGUF weight and the existing DS4 Q8_1 activation workspace, beat the existing HIP complete call on every retained exact key, and stop only after a recursive review finds no new valid in-contract optimization mechanism.

This is a dense-forward multiply campaign. The existing HIP DS4 activation quantizer remains unchanged and is treated as fixed producer infrastructure. It is launched before both the HIP and GGTensile multiplication controls. This campaign must not silently replace, rewrite, cache, fuse, or retune that quantizer unless a later review establishes a natural fused ownership contract and opens a separate measured project.

## Contract

Target only:
- gfx1151, wave32, WMMA V1, and BF16 input/output activations.
- Packed GGUF Q4_K weights with direct in-kernel packed decode; no prepared weights, dense shadows, external decode workspace, split-K, persistent workgroups, grouped MMQ, or online tuning.
- The fixed HIP DS4 Q8_1 activation producer. The GGTensile multiply consumes its 144-byte Q8_1 blocks directly.
- One exact forward `ProblemType` and one exact forward `ProblemSize` per GGTensile artifact.
- Strict rejection for unsupported solutions and HIP fallback outside selected exact keys.
- Zero private storage, spills, scratch instructions, calls, and dynamic stack.
- Serial warmed rotating-control timing. Builds and independent correctness jobs may run in parallel; timed GPU work never does.

Forward coordinates are:

```text
M = flattened activation rows
N = out_features
K = in_features

output[M,N] = input[M,K] @ dequant_q4_k(weight[N,K]).T
```

Q4_K has 256 logical values per 144-byte packed block. The fixed DS4 producer emits Q8_1 blocks with 128 signed int8 values plus four `(d,sum)` FP16 pairs. The producer allocates a padded row stride for the selected forward J tile but launches only real rows; the GGTensile multiply must use that exact stride and never consume an uninitialized padded row as a valid result.

## Exact Production Scope

Each ordinary projection runs at physical batches 1, 4, and 16, yielding `M={2048,8192,32768}`. The campaign has 12 exact keys:

| Family | `(N,K)` | Representative tensor | Calls | Keys |
| --- | ---: | --- | ---: | ---: |
| Narrow K/V/shared gate/up | `(512,2048)` | `blk.5.ffn_gate_shexp.weight` | 70 | 3 |
| Shared-expert down | `(2048,512)` | `blk.5.ffn_down_shexp.weight` | 30 | 3 |
| Attention output | `(2048,4096)` | `blk.3.attn_output.weight` | 10 | 3 |
| Attention query | `(8192,2048)` | `blk.39.attn_q.weight` | 1 | 3 |

The initial order is narrow M32768, attention-output M32768, shared-down M32768, and query M32768, then the lower-M counterparts. This follows production call weight and observed complete HIP latency, rather than an arbitrary geometry order.

## Fixed Activation Producer

The installed HIP `quantize_bf16_q8_1_ds4` HSACO is the fixed DS4 producer control. For each 32-value group it computes absolute maximum, `d=amax/127`, signed-int8 nearest quantization, and the input sum needed by Q4_K zero-point correction. It is outside the GGTensile solution identity in this campaign.

The forward benchmark has three explicit measurements:
- Fixed HIP quantizer plus HIP packed multiply: public HIP control.
- Fixed HIP quantizer plus GGTensile packed multiply: complete candidate.
- A prequantized-workspace bracket of HIP and GGTensile multiply bodies: diagnosis only.

The DS4 workspace produced for GGTensile must be byte-identical to the installed HIP producer. No alternate activation quantizer, clamp, rounding, reciprocal, reduction, metadata-layout, or workspace-lifetime change may be selected through this campaign.

## GGTensile Forward Design

Forward identity and writer/runtime/inspection support must be separate from existing backward identity so that forward ABI, Q8_1 workspace layout, signed-int8 WMMA, and output scaling cannot be confused with BF16 backward decode.

The Q4_K forward multiply owns:
- exact global addressing for packed `[N,K/256*144]` weights and DS4 workspace `[K/128, M_padded]` blocks;
- Q4_K payload, scale, and minimum extraction into LDS-facing int8 fragments;
- DS4 Q8_1 payload and `(d,sum)` loads;
- signed-int8 `v_wmma_i32_16x16x16_iu8` issue, integer accumulator lifetime, scale/min correction, FP32 accumulation, BF16 rounding, and stores;
- fixed exact M/N/K launch geometry, barriers, wait dependencies, LDS address layout, and output ownership;
- static resource accounting and full source coverage tests.

The initial body may reuse the backward project’s strict assembly lifecycle, register/resource gates, metadata parsing, exact-key catalogs, immutable phases, and Q4_K packed-load reasoning. It must not reuse a BF16-decode/WMMA body when that changes the Q8_1 integer-MMQ arithmetic. Reuse is valid only after operand layout and output ordering are proven equal.

Forward tuning knobs are introduced only after they have real alternate emitters and strict validation. Candidate categories are macro M/N/K ownership, int8-WMMA clamp form, packed payload/metadata vectorization, DS4 metadata lifetime, LDS padding/XOR layout, global/local prefetch, load-decode-WMMA scheduling, output correction scheduling, and exact output store traversal. The activation producer is not a knob.

## Correctness and Resource Gates

Before timing a candidate:
- inspect its symbol, exact forward ABI, gfx1151 metadata, LDS size, WMMA count, private segment, spills, scratch, calls, and dynamic stack;
- compare multiplication output bit-exactly with the exact HIP multiply when identical integer-WMMA and output ordering are retained; reordered semantic candidates may differ when candidate-to-HIP normalized RMSE is at most `5e-4`, maximum absolute error is at most `0.015625`, and the independent-reference normalized RMSE remains at most `0.04`;
- compare complete output against public HIP after input mutation and packed-weight mutation;
- compare with an independently GGUF-dequantized BF16 `torch.mm` reference using the established Q4_K forward error envelope;
- mutate a byte in the produced Q8_1 workspace for multiplication-only producer-handoff coverage;
- run reduced-K, one-hot activation, all-zero activation, positive/negative maximum, scale/sum, packed payload, metadata, block-boundary, tile-boundary, and output-row/column-boundary fixtures.

Any deliberate semantic experiment needs a separately named problem type, independent reference envelope, and complete-call acceptance. It cannot silently weaken bit-exact HIP comparison.

## Phases

### Phase 1: Strict forward infrastructure

- Add forward-specific problem identity, validation, assembly writer, direct HIP runtime modules, inspection, and immutable campaign phases without changing HIP code.
- Package the fixed installed DS4 quantizer as a benchmark-only producer control with its existing ABI and exact resource metadata.
- Add the 12-key Q4_K forward inventory and an open catalog.
- Add tests that cover every writer branch and every forward validation/ABI/resource rule.

### Phase 2: Correct int8 MMQ control

- Build a Q4_K K512 pilot that consumes DS4 workspace and matches HIP plus independent reference behavior.
- Extend to K2048 and K4096, then all four N families with fixed M128 ownership.
- Verify lane mapping, Q4_K scale/min correction, signedness, WMMA clamp behavior, output conversion, workspace stride, and physical packed-row accounting before screens.

### Phase 3: Large-margin search

Search high-weight M32768 keys first, then transfer only measured mechanisms:
- Exact M/N ownership and launch density.
- Q4_K global payload/metadata reads and DS4 global/LDS load layout.
- Int8 WMMA operand order, clamp necessity, integer accumulator placement, and scale/min correction scheduling.
- LDS padding/XOR layout, vector local reads, barriers, waits, and prefetch schedules.
- Bounded next-tile reads and overlap only when the prequantized lower bounds expose a real gap.
- Smaller M keys after large-M choices close.

Candidates pass correctness and inspection before nine-repeat serial screens. A new resource-bearing mechanism requires a stable gain above 2% in a 25-repeat rotating assembly-control confirmation. Unconditional instruction or resource reductions may remain when neutral or favorable.

### Phase 4: Final evidence

For every selected key, measure fixed-quantizer complete latency, prequantized multiply latency, int8-WMMA/DS4-Q4 decode floors where diagnosis is ambiguous, resource reports, independent rebuilds, producer/input/packed mutation correctness, and a 25-repeat serial confirmation. Selection requires every exact key to beat HIP; a weighted average never authorizes a slower key.

## Recursive Optimization-Exhaustion Review

Before declaring completion, reread this plan, the Q4_K HIP source and normalized ISA, all forward artifacts and timing reports, the dense-forward HIP record, completed GGTensile backward records, grouped histories, lower bounds, rejected candidates, `~/rdna35-isa-markdown/`, AMD LLVM definitions/tests, and relevant CK/TensileLite material.

Classify every remaining idea as retained and measured; rejected by correctness, resources, timing, or reproducibility; contract-incompatible or deferred with a prerequisite; or actionable with a target key and measurement gate. An actionable idea must be implemented and measured, then the full review repeated from the new premise. Completion is allowed only after a fresh review finds no actionable in-contract mechanism, every selected key beats HIP, and the residual bottleneck is quantified.

## Completion Record

- [x] Forward identity, fixed-quantizer control runtime, inspection, and exact 12-key inventory.
- [x] Correct Q4_K signed-int8 WMMA control for K512/K2048/K4096.
- [x] High-weight M32768 ownership, LDS, and scheduling search.
- [ ] Per-key catalog with all exact keys faster than HIP; two shared-down keys are selected and ten keys remain open.
- [x] Complete and prequantized lower bounds with residual bottleneck explanation.
- [x] Independent rebuild reproducibility, mutation coverage, and 25-repeat confirmation.
- [ ] Recursive optimization-exhaustion review with no actionable mechanism remaining.
- [x] Public dispatch and artifact packaging for the two selected exact keys; all other keys retain HIP fallback.

## Implementation Record

The initial forward infrastructure uses operation-specific writer modules: `kernel_writer_assembly_mmq_fwd.py` for forward and `kernel_writer_assembly_mmq_bwd.py` for backward. Inventories and solution catalogs live under `tools/ggtensile/configs/`. CLI generation selects the writer from the strict `ProblemType`; it does not infer the operation from dimensions or filenames.

The first control is deliberately diagnostic: one 32-thread wave owns a 16x16 output tile, reads Q4_K and Q8_1 DS4 operands directly from global memory, issues two signed/clamped integer WMMAs per 32-value group, applies per-output-column Q4_K scale/min metadata in FP16 followed by FP32 accumulation, and stores BF16 round-to-nearest-even output. It declares 88 VGPRs, 16 SGPRs, zero LDS, zero private bytes, and zero spills. The static loop body contains 16 integer-WMMA instructions and no barriers.

A standalone 16x16 signed-int8 WMMA mapping pilot matched all 256 CPU `B @ A.T` outputs. The first production control `(M,N,K)=(2048,512,2048)` then consumed the unchanged installed HIP DS4 workspace and matched the public HIP complete output bit-for-bit for all 1,048,576 BF16 elements. The initial mapping defect was an emitter register alias between the persistent packed-weight row and a metadata-address temporary; only output column zero was correct before that alias was removed. This failure mode is retained as a writer-coordinate test target rather than treated as a tuning result.

The ABI-aligned three-way diagnostic uses the same workspace for the installed HIP multiply and GGTensile multiply. On the first K2048 narrow key, the direct GGTensile body measured 0.494 ms versus 0.174 ms for the direct HIP body in a single-repeat check, and the complete candidate measured 0.519 ms versus 0.195 ms for HIP. These are diagnostic values, not confirmation evidence. Depending on the direct installed-HIP invocation, the candidate is either bit-identical or differs in one least-significant BF16 element with absolute error below `2^-20`; complete-call acceptance therefore uses public HIP and the independent reference.

The next control gives each 128-thread workgroup a `128x64` macro tile. Four flat wave32 waves each reuse one Q4_K `16x32` weight fragment across eight activation-row tiles, reducing repeated weight decode and metadata work without LDS. Its first nine-repeat rotating screen measured 0.294560 ms for the prequantized body versus 0.174040 ms for HIP, a `1.6925x` ratio, and 0.316359 ms complete versus 0.197112 ms for HIP, a `1.6050x` ratio. It preserved the one-element `2.98e-8` HIP difference and exactly preserved HIP's independent-reference envelope: normalized RMSE `0.0136406` and maximum absolute error `0.046875`.

Two activation-sharing controls were rejected. Literal padded-row LDS staging was correct and reduced static VMEM from 280 to 106, but 210 LDS instructions, serialized waits, barriers, and single-workgroup residency regressed the body to 0.405 ms. A coalesced 18,432-byte plane-at-a-time stage reduced that regression to 0.304 ms, and preloading all LDS fragments reached 0.292 ms at 236 VGPRs; neither beat the zero-LDS control. Preloading the same fragments directly from global memory regressed to 0.311 ms. These values are single-repeat mechanism diagnostics, not selection evidence.

The retained wave-reuse body keeps Q4_K and DS4 scale/min pairs packed as FP16 and uses `v_fma_mix_f32` to convert and multiply in the FP32 accumulator path. This removes 192 static VALU instructions while preserving the exact FP16 inputs and FP32 accumulation order. Its nine-repeat rotating screen measured 0.270722 ms for the body versus 0.173782 ms for HIP, a `1.5578x` ratio, and 0.293837 ms complete versus 0.195688 ms for HIP, a `1.5016x` ratio. Candidate output was bit-identical to HIP across baseline, input, packed-weight, and workspace mutations and retained independent-reference normalized RMSE `0.0136406` and maximum absolute error `0.046875`.

An eight-tile WMMA batch reduced waits and static VALU but was neutral at 0.269777 ms while consuming 246 VGPRs, so it was rejected. During its register-allocation diagnosis, one intermediate artifact overlapped a packed-scale register with a persistent metadata address and caused a process-scoped gfxhub/TCP read-permission fault. The process terminated, no stale `/dev/kfd` user remained, management queries were normal, and a fresh-process PyTorch allocation/add/synchronize completed without new kernel messages. The overlap was removed before any further timing and is covered by register-boundary source assertions.

Batching four activation tiles balances WMMA latency hiding against residency: it declares 194 VGPRs, issues 128 static WMMAs, reduces static VALU from 3,763 to 3,379, and reduces waits from 74 to 26. Its nine-repeat screen measured 0.257270 ms for the body versus 0.173956 ms for HIP, a `1.4789x` ratio, and 0.280581 ms complete versus 0.196384 ms for HIP, a `1.4287x` ratio. It remains bit-identical to HIP under all mutations and preserves the same independent-reference envelope.

The HIP-shaped staged control cooperatively loads the 8,192-byte raw Q4_K payload and two 9,216-byte DS4 activation planes into 26,624 bytes of LDS. Its final schedule declares 239 VGPRs, uses four barriers, issues 128 static WMMAs, and has zero private storage or spills. Bank-legal adjacent accumulator updates use 256 VOPD issues; one persistent zero fragment replaces repeated per-group zero initialization. The resulting body has 3,000 static VALU issues, 3,256 VALU operations, 148 VMEM instructions, and 284 LDS instructions. A nine-repeat screen measured 0.251980 ms for the body versus 0.174051 ms for HIP, a `1.4477x` ratio, and 0.275032 ms complete versus 0.196569 ms for HIP, a `1.3992x` ratio. Baseline and all input, packed-weight, and workspace mutations are bit-identical to HIP. This is the fastest handwritten assembly control so far, but remains non-selectable until it beats HIP.

A reassembled compiler-ISA control established the remaining implementation gap. Renaming the extracted gfx1151 HIP assembly, converting it to Code Object V5 with 38,400 bytes of static LDS, and preserving the compiler's `(32,4,1)` work-item geometry produced bit-identical output and a `0.9996x` single-repeat body ratio against the installed HIP artifact. The control confirms that code-object format and static versus dynamic LDS allocation do not explain the gap. The material differences are the compiler schedule's decoded-weight and packed-scale LDS layout, inner group loops, incremental LDS waits, clauses, and register allocation. Removing all `s_delay_alu` hints, the final deallocation message, GL0 invalidations, WMMA clamp, or memory clauses did not produce a stable material gain; removing clauses regressed by about 10%. Alternate AMDGPU scheduler strategies, row tiles 32 and 64, and aligned `b128` activation staging were exact but neutral or slower. These are assembly-oracle diagnostics only and do not modify the HIP source or authorize packaging the HIP artifact as a GGTensile solution.

The decoded staged control now reproduces the compiler's 38,400-byte LDS partition: a 512-byte prefix, one 18,432-byte DS4 plane, and 64 padded 304-byte weight rows holding decoded nibbles and packed FP16 scale/min pairs. Four vector Q loads are issued before descending VMEM waits, DS4 stores use stride-64 pairs under progressive waits, LDS reads overlap the first eight WMMAs, and two rolled four-group loops execute 128 dynamic WMMAs from 32 static instructions. The artifact declares 239 VGPRs and 16 SGPRs with no private storage or spills; it has 985 static VALU issues, 1,049 VALU operations, 64 VOPD issues, 141 VMEM instructions, 104 LDS instructions, four barriers, and 63 waits. A 15-repeat rotating comparison measured 0.204177 ms for the candidate body versus 0.181193 ms for HIP, a `1.1268x` ratio. Baseline, input, packed-weight, and workspace mutations remained bit-identical to HIP. This is the current handwritten control, but it is still non-selectable.

Full eight-tile arithmetic batching, `(32,4,1)` work-item geometry, a separate metadata LDS plane, broad scale-product overlap, and compiler-shaped scalar Q loads were exact but neutral or slower. A packed all-wave metadata decoder removed 30 static VALU and four static LDS issues but did not improve the rotating median and doubled metadata load participation, so it was rejected. An invalid experiment placed `s_clause` across Q loads separated by VALU address updates. The bounded launch timed out and the kernel log reported failed MES queue removal followed by a successful MODE2 reset and `device wedged, but recovered through reset`. No benchmark process remained, the device nodes and a fresh known-good launch were healthy, and the faulting clause code was removed before the campaign resumed.

The decoded-staged control was then generated and measured on all 12 production keys. Before the retained instruction schedule, its prequantized candidate/HIP ratios were approximately `1.08-1.16x` for the K2048/K4096 families and `1.47-1.50x` for K512. Direct global, wave-reuse, four-tile batching, and raw staged controls were all slower on K512. A temporary `128x128`/256-thread K512 geometry was both incorrect and slower, while corrected packed-work-item `(32,4,1)` geometry was exact but `0.1-0.3%` slower than the existing `(128,1,1)` workgroup across all three M values.

The retained decoded schedule combines only mechanisms that had already passed single-shape exactness and timing screens: selective weight and packed-metadata LDS-base hoists, short-lived `v_mad_u32_u24` activation addressing, direct `v_cvt_f16_u16_e32` conversion of bounded six-bit scale/min fields, eight contiguous clause-backed output-store runs, and one output-row address calculation followed by exact row increments. These mechanisms now have explicit `DenseForwardSolution` identity through `LdsAddressHoist`, `ActivationAddressing`, `MetadataConversion`, and `OutputStore`; the old decoded-staged control remains independently generatable. The retained writer reproduces the accepted experimental source byte-for-byte after normalizing the new strict symbol.

All 12 retained artifacts declare 239 VGPRs, 16 SGPRs, 38,400 bytes of LDS, zero private bytes, zero spills, 32 static WMMAs, four barriers, and eight clauses. Five-repeat transfer screens improved the old decoded writer by `6-31%`; the largest gains were the K512 keys, which improved by about `30%`. The generated-writer 25-repeat prequantized confirmations were bit-exact and produced candidate/HIP ratios from `0.9965x` to `1.0255x`, with an unweighted median of `1.0126x`. Three large-output keys were slightly faster than HIP. Representative complete-call ratios were `1.0140x` narrow, `1.0364x` shared-down, `1.0119x` attention-output, and `1.0024x` query; selection remains open because some exact keys do not yet beat HIP.

Strict correctness was repeated on every key. Baseline, public complete call, input mutation, packed-weight mutation, and DS4-workspace mutation were bit-identical to HIP. Every mutation changed output, all outputs were finite, and independent GGUF-dequantized BF16-reference normalized RMSE ranged from `0.01320` to `0.01372`, below the `0.04` limit. Independent generation and build trees produced byte-identical assembly and code objects for all 12 keys.

Supported K512 profiling rules out excess decode traffic, cache misses, and LDS conflicts as the residual cause. Retained GGTensile and HIP both report `11.739%` LDS-bank-conflict and approximately `80.5%` L2-hit rates. GGTensile executes 4,471 versus 5,211 VALU instructions per work item, 15% fewer total instructions, and 16% fewer flat loads, but reports about 70% more instruction-fetch waits and 14% more aggregate wait-any cycles. A bounded one-to-eight-NOP hot-loop phase scan was exact; its apparent `1.7%` short-screen gain collapsed to `0.3%` in 25-repeat rotation, so loop padding was rejected. The residual target is instruction issue/dependency scheduling, not additional memory reduction.

Three diagnostic lower bounds were generated and measured independently for every production key. Store-only work is approximately `2-7%` of retained latency for K2048/K4096 and `24-27%` for K512. Decode/stage/store work is `44-57%`, while memoryless math/control/store work is `86-88%`. These floors are not additive, but they close raw global/LDS traffic reduction as the dominant residual opportunity and explain why K512 is much more sensitive to conversion/store overlap.

Descriptor and mapping experiments did not generalize. gfx11 `.amdhsa_inst_pref_size` values 15, 31, 47, and 63 were exact; size 63 improved one K512 pilot by `4.2%`, but the all-key 25-repeat rotation was neutral on average and regressed shared-down. Flattened workgroup mapping and static M clusters 1, 2, 4, 8, and 16 were exact after correcting the launch contract. Cluster 1's initial `0.9787x` parent result became `0.9935x`, `0.9987x`, and `1.0015x` across K512 M2048/M8192/M32768, while larger clusters regressed sharply. Both mechanisms are rejected.

Output-width and synchronization alternatives are closed for gfx1151. Cross-lane B32, B64, and B128 gather/pack schedules preserve the 64 required BF16 stores only by adding DS exchange and packing work; the strongest B128 form was about `7%` slower. gfx1151 has no native packed BF16 conversion, and VOPD cannot pair the dynamic BF16 conversion, integer extraction, or `v_fma_mix_f32` operations. Split `s_barrier_signal`/`s_barrier_wait` is gfx12-only. Activation-load clauses, moving 32 or all 36 activation requests before metadata conversion, packed-weight clauses, and activation wait-ladder repacking were exact but neutral or slower.

Metadata extraction became the useful scheduling lever. Literal-bearing `v_perm_b32` reduced the static extraction count but regressed by about `0.5%`, consistent with measured instruction-fetch pressure. Width-8 dependency exposure improved the retained K512 family by approximately `0.6-1.1%`; fully exposing the 24 independent BFEs before eight combines was stronger on large M. Early placement after four packed-weight VMEM requests and partial-prefix variants improved the rotation-sensitive M2048 key but did not stably beat HIP. Upper-field byte prepacking removed seven instructions per K block, and B64/B128 metadata stores removed additional LDS issues, but their short wins did not displace independent extraction under 25-repeat confirmation. Scoped metadata, activation-stage, and epilogue priorities were retained only where exact-key timing justified them.

Epilogue conversion/store scheduling was searched over `TilesAhead={1,2,4,8}` and `DependencyWidth={1,2,4,8}` with exact BF16 round-to-nearest-even operations, unchanged store count, and unchanged 239-VGPR allocation. No universal schedule won. For shared-down M8192, independent extraction with `TilesAhead=1`, dependency width 4, and priority 2 measured `0.98712x` HIP in a 25-repeat head-to-head against retained and competing width-8 schedules. For M32768, independent extraction with `TilesAhead=1`, dependency width 2, and priority 2 measured `0.98832x` HIP in the same comparison. Fresh strict-writer confirmation measured aggregate candidate/HIP medians `0.99585x` and `0.98603x`, with paired medians `0.98933x` and `0.98560x`, respectively.

Those two shared-down keys are now the only selected forward entries. Their strict identity adds `MetadataSchedule`, `EpilogueTilesAhead`, `EpilogueDependencyWidth`, and `EpiloguePriority`; validation rejects independent extraction outside `(M,N,K)=(8192,2048,512)` and `(32768,2048,512)`. Generated instruction streams are identical to the measured experimental streams. Both artifacts declare 239 VGPRs, 16 SGPRs, 38,400 bytes LDS, zero private bytes/spills, 32 WMMAs, four barriers, and eight clauses. Full correctness is bit-identical to HIP for baseline and every mutation; independent-reference normalized RMSE is `0.0136748` and `0.0136804`. Fresh complete-call ratios are `0.99154x` and `0.98368x`, and independent generation/build trees are byte-identical. The other ten inventory keys remain open and use HIP fallback; transformed assembly is not packaged or selected.

Public packaging reads the selected entries and strict solution mappings directly from the campaign inventory and catalog. The builder serially generates both assembly sources before compilation, assembles and links them with the gfx1151 toolchain, and rejects any code object that violates the exact 40-byte ABI, 128-thread wave32 geometry, 239-VGPR/16-SGPR allocation, 38,400-byte LDS allocation, or zero private/spill/stack contract. Existing kernel IDs remain unchanged; the two GGTensile IDs are appended to the generated table. The packaged sources and code objects are byte-identical to the independently validated artifacts. Public dispatch selects them only for exact Q4_K shared-down M8192 and M32768; a fresh-process artifact-open trace confirmed that M2048 resolves the existing HIP K512 artifact while the two selected calls resolve their strict GGTensile artifacts. Public independent-reference NRMSE was `0.0138111`, `0.0136459`, and `0.0136657` for M2048/M8192/M32768, respectively.
