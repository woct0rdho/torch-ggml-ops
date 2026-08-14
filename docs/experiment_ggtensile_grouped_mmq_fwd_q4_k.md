# GGTensile Grouped MMQ Forward Q4_K Experiment

## Purpose

Implement and optimize the first routed GGTensile forward kernel on gfx1151. The initial format is Q4_K and the retained workload is the Qwen routed down projection. This is an isolated kernel research campaign: public dispatch, generated bundle tables, extension registration, packaging, and the existing HIP fallback remain unchanged.

Correctness comes first. The first artifact reuses the existing direct-global Q4_K arithmetic as a low-resource control for grouped workgroup mapping and row tails. Optimization begins only after that control is correct, deterministic, and resource-clean.

## Problem Identity and Terminology

For each routed GEMM `g`, compute

```text
A_g[M_g,512] x W_g^T[512,2048] -> C_g[M_g,2048]
```

`G` is the number of routed GEMMs, `gemmIndex` identifies an entry in the route metadata, and `expert_indices[gemmIndex]` selects a physical expert from the shared weight bank. `expert_offsets[gemmIndex]` is the cumulative row end, so `M_g = rowEnd_g - rowBegin_g` and `R = sum_g M_g`. A routed GEMM is not synonymous with a physical expert because routing metadata owns the association.

TensileLite calls its descriptor-backed mode `GroupedGemm`. This record uses grouped GEMM, GEMM index, workgroup-to-GEMM mapping, `MatrixInstruction`, `MacroTile0`, `MacroTile1`, and `DepthU` where those terms describe the same concepts. In the GGTensile forward convention, `MacroTile0` spans routed rows, `MacroTile1` spans output columns, and `DepthU` is the K-loop step. Q4_K decode, the Q8_1 `F16_D4S4` layout, cumulative route offsets, and expert-bank addressing remain GGTensile-specific mechanisms rather than being renamed as ordinary dense GEMM operands.

## Exact Scope

The authoritative packed Q4_K weight bank has physical shape `[256,2048,288]`. The exact aggregate-row generation keys are:

| Physical batch | Aggregate rows `R` | Exact `(R,N,K)` |
| ---: | ---: | ---: |
| 1 | 16,384 | `(16384,2048,512)` |
| 4 | 65,536 | `(65536,2048,512)` |
| 16 | 262,144 | `(262144,2048,512)` |

The bank has 256 physical experts. Runtime route metadata may contain up to 256 routed GEMMs and supplies the physical expert IDs and cumulative row ends. `M_g` is neither a generation constant nor necessarily a tile multiple. Uniform, skewed, sparse-ID, repeated-ID, and boundary routes are required controls. Development fixtures include row counts around 1, 15, 16, 17, 31, 32, 63, 64, 127, 128, and 129.

## Kernel Contract

The target is gfx1151 with wave32, WMMA V1, a BF16 destination, and code-object version 5. Packed GGUF Q4_K weights remain authoritative and are decoded by the generated kernel. The activation operand is the existing Q8_1 `F16_D4S4` workspace with physical shape `[K/128,R,144]`; there is no alternate producer, logical BF16 expert matrix, or padded-row interpretation.

Route metadata is device-resident and contiguous. `expert_indices` has int64 elements, `expert_offsets` has int32 elements, both have length `G`, and the final valid cumulative offset equals `R`.

The direct grouped control has the established 64-byte multiply kernarg ABI: five pointers named `weights`, `activations`, `dst`, `expert_indices`, and `expert_offsets`; four u32 values named `num_experts`, `nrows_weight`, `nrows_activation`, and `blocks_per_weight_row`; and one u64 `bytes_per_expert`. This is a fixed routed ABI, not TensileLite `SupportUserArgs` or a table of conventional per-GEMM descriptors. The baseline launch supplies `G` through grid geometry, as the isolated HIP grouped body does.

The first ownership mechanism is serial GEMM ownership. One workgroup owns one `(gemmIndex, MacroTile1)` output-column tile and iterates the GEMM's `MacroTile0` row tiles without host readback of route offsets. Invalid physical expert IDs, empty or reversed row ranges, and out-of-range cumulative offsets are inert. Valid partial row tiles use masked activation reads and masked BF16 stores.

Every retained kernel has zero private storage, spills, scratch instructions, calls, and dynamic stack unless a separately named mechanism explicitly introduces a scratch contract. Paired projection, fixed-group output-A, prepared weights, a dense shadow, an external decode workspace, producer fusion, public API wiring, and online tuning are outside this campaign. SplitK, persistent work distribution, and companion setup or reduction kernels are not baseline features, but they may be implemented in the isolated research path when a measured bottleneck gives them a concrete premise. Such a candidate must expose its additional ABI and workspace explicitly and include every launch in correctness and timing.

## TensileLite Grouped GEMM Audit

The audit covered TensileLite's grouped problem state, solution validation, gfx11 grouped configurations, kernel signature construction, assembly prologue, grouped workgroup mapping, and GSU components. The primary sources were `Tensile/SolutionStructs/Problem.py`, `Tensile/SolutionStructs/Solution.py`, `Tensile/Components/Signature.py`, `Tensile/KernelWriterAssembly.py`, `Tensile/Components/GSU.py`, and the gfx11 `grouped_gemm_gfx11.yaml` and `grouped_gemm_userargs_gfx11.yaml` tests in the local hipBLASLt TensileLite checkout. Composable Kernel's gfx11 `DeviceGroupedGemm_Wmma_Fixed_Nk` implementation was inspected as a secondary scheduling reference. No assembly body was copied into this design.

TensileLite `GroupedGemm` launches a flattened workgroup space over conventional GEMM descriptors. Its assembly prologue receives a GEMM count and an argument-buffer pointer, then linearly scans GEMMs by loading each `M`, `N`, and batch extent, computing the GEMM's tile count from `MacroTile0`, `MacroTile1`, batch count, and GSU, and accumulating workgroup ranges until it finds the owning GEMM. It rebases the flattened workgroup index to a GEMM-local tile index and loads that GEMM's arguments. This is a useful workgroup-to-GEMM mapping reference, especially for a future flattened or persistent scheduler, but it is not directly reusable with cumulative routed-row offsets and one shared expert bank.

`SupportUserArgs` enables an alternate packed per-GEMM argument structure in generated kernels, while the client-level `UseUserArgs` setting chooses that path. The routed 64-byte GGTensile ABI should not use either name: it has shared operand bases plus compact route metadata, not independent A/B/C/D pointers, sizes, strides, alpha, beta, and epilogue fields for every GEMM. `PreloadKernArgs`, by contrast, names a transferable optimization concept and may be tested as SGPR argument preloading if gfx1151 metadata and resource inspection support it.

The gfx11 grouped tests use wave32 WMMA `MatrixInstruction` configurations and exercise edge sizes such as 127 and 129. Their search fields include `DepthU`, `InnerUnroll`, `PrefetchGlobalRead`, `PrefetchLocalRead`, `ClusterLocalRead`, `GlobalReadVectorWidthA`, `GlobalReadVectorWidthB`, `LocalReadVectorWidth`, `ScheduleIterAlg`, `ExpandPointerSwap`, `TransposeLDS`, `LdsPadA`, `LdsPadB`, `LdsBlockSizePerPadA`, `LdsBlockSizePerPadB`, `1LDSBuffer`, `StaggerU`, `WaveSeparateGlobalReadB`, `GlobalReadPerMfma`, `LocalWritePerMfma`, `StoreVectorWidth`, `NumElementsPerBatchStore`, `SourceSwap`, `GlobalSplitU`, and `GlobalSplitUAlgorithm`. Generic grouped tests also exercise `PreloadKernArgs`.

TensileLite's current validator rejects `GroupedGemm` with `StreamK` and rejects grouped kernels when `BufferLoad=0`; disabled or architecture-specific test configurations also show grouped StreamK work under development. These are generator-state observations, not gfx1151 ISA restrictions and not GGTensile design prohibitions. `BufferLoad` in TensileLite selects SRD/MUBUF addressing and must not be confused with this record's direct-global arithmetic control, which describes the absence of decoded-weight LDS staging. GGTensile may implement a mechanism that TensileLite currently gates when ownership, synchronization, address safety, reduction behavior, and resources can be established independently.

TensileLite's gfx11 grouped tests explicitly include `GlobalSplitU` values 1 and 2 with the `MultipleBuffer` algorithm for supported layouts. SplitK is therefore a real grouped tuning family, not something to reject because another combination is gated. For this fixed `K=512` workload it remains premise-driven: a candidate must show that additional K parallelism pays for partial-output storage and reduction. The implementation may be native GGTensile and need not reproduce TensileLite's GSU code or restrictions.

## Tuning Vocabulary and Search Space

Geometry and ownership. `WorkGroup`, `WavefrontSize`, `MatrixInstruction`, `MacroTile0`, `MacroTile1`, `DepthU`, `InnerUnroll`, and `SourceSwap` describe workgroup shape, WMMA ownership, and K-loop granularity. Wavefront size remains fixed at 32 for gfx1151. The WMMA opcode and wave tile/group fields, row and column macro tiles, and number of waves are linked choices and must be represented as one validated solution identity rather than independently mutated integers.

Global reads and decode. `GlobalReadVectorWidthA/B`, `PrefetchGlobalRead`, `StaggerU`, `WaveSeparateGlobalReadB`, and `GlobalReadPerMfma` motivate separate packed-Q4_K and Q8_1 read widths, prefetch distance, wave ownership, and issue placement. Their numerical TensileLite values do not transfer automatically because one GGTensile operand is compressed and decoded in flight. GGTensile-specific fields must identify packed load width, decoder width or batch, reconstructed scale/minimum lifetime, and whether decoded weights go directly to WMMA operands or through LDS. `DirectToLds` and `DirectToVgpr` are mechanism names to investigate, not assumptions that GGUF bytes can bypass decode.

LDS and local reads. `PrefetchLocalRead`, `LocalReadVectorWidth`, `TransposeLDS`, `LdsPadA/B`, `LdsBlockSizePerPadA/B`, `1LDSBuffer`, and `ExpandPointerSwap` map to decoded-weight and activation LDS layout, bank-conflict control, buffer count, pointer movement, and local-read distance. They are inactive for the no-LDS direct control and become legal only in a lowering whose physical plan owns the corresponding buffers, barriers, and register lifetimes.

Instruction scheduling. `ScheduleIterAlg`, `ScheduleGlobalRead`, `ScheduleLocalWrite`, `ClusterLocalRead`, `GlobalReadPerMfma`, and `LocalWritePerMfma` provide useful names for ordering choices. GGTensile will use explicit schedule policies whose emitted dependency graph is inspectable; a TensileLite numeric `ScheduleIterAlg` value is not imported as a semantic promise. Each schedule change must preserve wait-count correctness and be measured separately from geometry.

Stores. `StoreVectorWidth`, `NumElementsPerBatchStore`, store priority, and a possible `SourceSwap`-compatible output layout define the epilogue search. Every store mechanism must remain bounds-safe for arbitrary `M_g`. Store remapping or cross-lane packing is eligible only when its added VGPRs and lane operations beat the simple BF16 path.

Grouped mapping and K partitioning. `WorkGroupMapping` motivates tile-order and cache-locality experiments after GEMM ownership is known. Serial GEMM ownership, a TensileLite-style flattened workgroup-to-GEMM mapping, row-task ownership, persistent queues, and work stealing are distinct scheduling mechanisms with distinct metadata costs. `GlobalSplitU`, `GlobalSplitUAlgorithm`, `LocalSplitU`, `WaveSplitK`, and StreamK-style K partitioning are also eligible when profiling supports them. Any cross-workgroup K partition needs an explicit FP32 partial-output and reduction design, deterministic correctness criteria, workspace accounting, and complete-call timing; a validation guard in another generator is not a rejection argument.

The search is staged rather than a blind Cartesian product. First establish geometry and arithmetic ownership, then packed reads and decode, then LDS and schedule, then stores and grouped mapping, and only then test cross-workgroup reduction or persistence if unresolved launch imbalance or K-loop latency remains material. Unknown, inactive, or semantically coupled fields are rejected by validation.

## Writer Reuse Decision

The ordinary `ForwardKernelWriterAssembly` facade cannot represent grouped ownership. Its fixed 40-byte ABI, rectangular exact-M grid, and edge-tile rejection are incompatible with runtime route metadata, workgroup-to-GEMM mapping, and partial routed rows. A distinct grouped writer, grouped problem identity, grouped solution identity, derived state, and physical register plan are required.

Reuse is permitted below the facade where semantics are identical. That includes `QuantForwardSemantics` for Q4_K packed planes and scale/minimum fields, `F16D4S4ActivationMetadata`, packed scale/minimum reconstruction, signed-int8 WMMA emission, BF16 RNE emission, ROCISA setup, deterministic register planning, and assembler/linker tooling. The direct-global Q4_K arithmetic schedule supplies the first control, with route selection, aggregate-row addressing, grouped workgroup mapping, and masked stores owned by a grouped lowerer.

The dense writer will not be invoked as a sibling writer, copied as generated assembly, or post-processed. A shared emitter is justified only when grouped and dense typed operands, lifetimes, dependencies, and emitted text are proven equal.

The G0 implementation now has standalone grouped problem, solution, validation, derived-state, physical-plan, lowering, writer, inspection, and runtime-launcher modules. The serial control uses `grid=(128,G,1)` for the fixed `N=2048` key, so `blockIdx.y` is the runtime GEMM index and `blockIdx.x` owns one 16-column tile. Its 88-VGPR Q4_K arithmetic plan is separate from the dense plan, while its scalar plan records the five pointer arguments, four u32 values, u64 expert stride, route bounds, expert rebasing temporaries, row loop, and saved EXEC mask. The emitted artifact declares a 64-byte kernarg segment, 32 SGPRs, 88 VGPRs, zero LDS, and zero private storage.

Route metadata is loaded in scalar memory before vector work begins. The lowerer validates the exact generated shape, rejects invalid expert IDs and cumulative ranges through an inert exit, computes the selected expert's 64-bit bank offset, and uses the aggregate row count for Q8_1 plane strides and output addressing. Each 16-row tile initializes activation registers to zero, masks its Q8_1 VMEM loads by `row < rowEnd`, restores the full wave for the two WMMA operations per Q4_K group, and masks the eight BF16 stores by the same row predicate.

## Implementation Phases

### G0: Contract and direct-global correctness control

Add strict grouped problem and solution identities, validation, derived state, deterministic physical register planning, a writer facade, artifact inspection, and a direct module launcher. Emit one-wave serial GEMM ownership with a 16-row `MacroTile0`, masked activation reads, and masked output stores for arbitrary `M_g`. Build independently twice and require byte-identical source and code objects. Qualify valid and invalid route metadata, full and partial tiles, input and packed-weight mutation sensitivity, finite output, exact agreement with the installed HIP packed path when arithmetic order is unchanged, and an independently dequantized grouped reference.

### G1: Production-shape correctness matrix

Qualify all three exact aggregate-row keys with uniform, skewed, sparse-ID, repeated-ID, and boundary routes. The timed launch path must not read route offsets on the host. Confirm expert-bank stride and output-row ownership with physical expert permutations and inactive-expert mutations.

### G2: Four-wave decoded-weight LDS implementation

Add a grouped physical plan for a four-wave body, initially `WorkGroup=[32,4,1]`, `MacroTile0=128`, `MacroTile1=64`, and `DepthU=32`, while treating those values as a measured starting point rather than a permanent restriction. Reuse only proven common Q4_K decode and WMMA components. Mask final activation staging and output stores without changing valid-lane arithmetic. Compare 128-, 64-, and 32-row ownership as explicit linked mechanisms. Inspect LDS, VGPR and SGPR allocation, barriers, waits, WMMA count, occupancy, and decoded-byte traffic before timing.

Within this phase, vary the relevant TensileLite-derived families in order: macro-tile and WMMA ownership, packed global-read width and prefetch, decoded LDS layout and buffer count, local-read width and distance, schedule policy, and store batching. Add each field to the grouped solution model only when the lowerer implements distinct behavior for it.

### G3: Grouped scheduling, K partitioning, and selection

Compare serial GEMM ownership with row-task and flattened workgroup-to-GEMM ownership only after the serial tiled body is correct. A flattened mapper must account for its scan or prefix cost at `G <= 256`; a row-task or persistent mechanism requires an explicit task ABI and setup-cost accounting. If profiling shows poor occupancy, route imbalance, or a dominant K loop after those changes, evaluate native GGTensile SplitK or StreamK-style mechanisms with explicit partial-output and reduction contracts rather than inheriting TensileLite's gates.

Compare complete prequantized grouped multiply calls against the installed HIP J64/J32 bodies on identical route profiles. Use warmed serial rotating timing, confirm a retained result in reversed order with longer repeats, and reject a candidate that regresses any exact key it would select. Exact aggregate-row selection remains in research records only; no public selector or generated bundle change is part of this campaign.

## Correctness and Resource Gates

Before timing, validate the exact grouped problem and solution identity and reject unknown or inactive fields. Inspect the symbol, kernarg ABI, gfx1151 target, wave32 metadata, workgroup, VGPRs, SGPRs, LDS, private segment, spills, scratch instructions, calls, dynamic stack, barriers, waits, and WMMA count. A mechanism with setup, scratch, atomics, partial outputs, or reduction must inspect and report every kernel and allocation in that mechanism.

Correctness qualification compares against both the installed grouped packed kernel and an independently dequantized BF16 grouped reference. It covers full tiles, non-aligned tails, sparse and repeated physical expert IDs, invalid route entries, aggregate-row boundaries, deterministic reruns, and independent rebuilds. Input, active packed-weight, and Q8_1 workspace mutations must change output, while inactive-expert mutations must remain inert. A changed accumulation order uses an explicit numerical envelope rather than claiming bitwise equivalence.

Timing is not promotion evidence until every correctness and resource gate passes. Complete mechanism time, including mapping setup, partial-buffer initialization, reduction, and synchronization, is authoritative. Static instruction reductions and lower resource counts explain results but do not select a kernel.

## Experiment Log

### Campaign opened

The generic GGTensile design, ordinary Q4_K forward record, completed grouped-HIP forward record, grouped HIP source, and benchmark infrastructure were reviewed. The exact Q4_K keys, packed bank shape, routed ABI, Q8_1 workspace contract, and mandatory runtime tails were confirmed. A separate grouped writer facade was selected. The existing direct-global Q4_K arithmetic is the first correctness control, while the dense 128x64 decoded-LDS body is only an optimization source at proven semantic component boundaries. The local gfx1151 GPU, ROCm assembler and linker tools, installed grouped HIP kernels, and Qwen GGUF models are available for device qualification.

### TensileLite grouped GEMM audit

TensileLite provides a genuine `GroupedGemm` assembly path, a packed per-GEMM `SupportUserArgs` variant, flattened workgroup-to-GEMM selection, gfx11 wave32 WMMA coverage, edge-size tests, and broad geometry, read, LDS, scheduling, store, mapping, and GSU knobs. Its current grouped mapper performs a linear scan over per-GEMM tile ranges. That mechanism is a scheduling reference but does not match the compact routed ABI or shared packed expert bank.

Generator rejections such as grouped StreamK or flat-address grouped loads are recorded as implementation state only. Conversely, the gfx11 grouped tests' `GlobalSplitU=2` coverage is evidence that SplitK belongs in the mechanism inventory. No external generator guard decides GGTensile feasibility: a mechanism is retained or rejected by its own typed contract, emitted dependencies, correctness, resources, reproducibility, and complete-call timing on gfx1151.

### G0 direct grouped control

The first grouped Q4_K artifact assembles and links for gfx1151 with the expected 64-byte ABI, 88 VGPRs, 32 SGPRs, zero LDS, zero private bytes, zero spills, no barriers, and 16 static WMMA instructions. Its source and code object are byte-identical across independent rebuilds. A device qualification using a real Qwen Q4_K routed down bank, the installed Q8_1 F16_D4S4 producer, and route lengths of 1, 15, 16, and 3 rows produced finite output and matched the installed grouped HIP packed path exactly. Against an independently dequantized grouped BF16 reference, the same run had maximum absolute error about `0.0124` and RMS error about `0.00292`, establishing the expected quantized arithmetic envelope without changing accumulation order.

## Recursive Final Review

Before declaring the Q4_K grouped campaign complete, reread this record, `docs/ggtensile_plan.md`, `docs/grouped_mmq_fwd_optimization.md`, the ordinary Q4_K forward record, grouped HIP source and normalized ISA, the TensileLite grouped writer and current grouped test configurations, relevant Composable Kernel fixed-NK scheduling code, every generated grouped artifact and timing report, rejected candidates, target ISA material, and relevant forward and backward mechanisms.

Classify every remaining idea as retained and measured; rejected by correctness, resources, timing, or reproducibility; contract-incompatible or deferred with an explicit prerequisite; or actionable with an exact target and qualification gate. Re-evaluate external generator guards as implementation facts rather than architectural conclusions. Implement and qualify every actionable finding, then repeat the complete review from the new premise. Completion is valid only when a fresh recursive pass finds no actionable in-contract mechanism and every selected exact key is correct, deterministic, resource-clean, and faster than its exact HIP control.

This rule is global across related GGTensile directions, formats, and shapes. Evidence may transfer, but ownership, lifetimes, synchronization, arithmetic order, resources, correctness, and exact-key timing must be re-derived here. Public API integration is explicitly outside this experiment and is not a completion criterion.
