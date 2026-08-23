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

### G0 production baseline

A research-only installed-control launcher now invokes the exact HIP Q4_K `N=2048`, `K=512`, `J64` artifact with the same packed bank and prequantized Q8_1 workspace as GGTensile. This keeps quantization, public dispatch, allocation, and host route handling outside the multiply comparison. Alternating-order event timing measured G0 at `5.15 ms` versus HIP at `1.64 ms` for `R=16384` uniform, `5.37 ms` versus `2.65 ms` for the boundary route, `20.93 ms` versus `5.36 ms` for `R=65536` uniform, `21.03 ms` versus `5.87 ms` for boundary, `99.42 ms` versus `20.33 ms` for `R=262144` uniform, and `93.09 ms` versus `21.03 ms` for boundary. G0 is therefore about 2.0 to 4.9 times slower, and the widening uniform-route gap establishes per-tile arithmetic and reuse as the immediate optimization premise rather than tail handling alone.

The production tiled HIP schedule is not bitwise identical to the one-wave direct control even though the small-tail control was exact. For `R=16384` uniform, the maximum absolute GGTensile-to-HIP difference was `3.05e-5` and RMS difference was `6.04e-9`. This is far below the independently dequantized reference envelope and is the explicit production schedule-order tolerance for subsequent tiled candidates.

### Four-wave direct-global geometry rejected

A geometry-only candidate grouped four WaveN owners into a 128-thread workgroup with `MacroTile0=16` and `MacroTile1=64` while retaining the direct-global arithmetic and 88-VGPR per-wave plan. It passed the 35-row boundary correctness control, matched the small-route HIP path exactly, and built for all production keys with 88 VGPRs, 32 SGPRs, zero LDS, and 16 static WMMA instructions. Relative to G0, its uniform median changed from `5.15` to `5.00 ms` at `R=16384`, `20.93` to `20.92 ms` at `R=65536`, and `99.42` to `108.09 ms` at `R=262144`; boundary medians were `5.36`, `21.53`, and `105.46 ms`. The minor smallest-key improvement did not transfer, both large-key routes regressed, and the candidate remained 2.1 to 5.3 times slower than HIP. The mechanism was reverted. Four-wave workgroup formation without decoded-weight or activation reuse is not an actionable optimization by itself.

### Fitted-prior decoded-LDS screen

The Qwen learned routing prior in `tools/tune_grouped_mmq_prior.py` is the primary speed metric for this campaign. Each screen uses the five weighted medoids from the deterministic 512-draw bank for the corresponding physical batch, a fixed real Q4_K bank, a shared prequantized Q8_1 F16_D4S4 workspace, three warmups, and nine alternating-order event repeats. The disjoint confirmation bank is the first ranking authority; the search bank and deterministic controls are independent evidence. This isolates the packed multiply body rather than assigning a routing, quantization, allocation, or public-dispatch difference to the candidate.

The first decoded-weight LDS body reused the typed dense Q4_K decode and scaled WMMA emitters under serial routed ownership. It stages a 128-by-64 output tile with 239 VGPRs, 40 SGPRs, 38,400 bytes of LDS, 32 static WMMAs, and four barriers. R35 routes containing 1-, 15-, 16-, and 3-row entries were bitwise equal to the installed packed HIP kernel and stayed within the established independent dequantized-reference envelope. All 15 production-prior comparisons were also bitwise equal to HIP. The initial serialized 128-row body reached only `0.734x`, `0.917x`, and `0.960x` weighted installed-over-candidate speed at B1, B4, and B16, so decoded reuse alone was necessary but insufficient.

Parameterizing the shared layout and register plan by row fragments produced a 64-by-64 grouped tile with 159 VGPRs, 40 SGPRs, 29,184 bytes of LDS, 16 static WMMAs, and four barriers. The same parameterization leaves the dense eight-fragment defaults unchanged. Independent scale/minimum extraction and metadata reads between low and high WMMAs reduced the 64-row low-half wait ladder to `7,5,3,1`. That schedule improved the weighted confirmation results to `1.060x`, `0.986x`, and `0.971x`, but did not yet close the large-batch gap.

The next retained mechanism is a typed mixed 64/32 tail body. It decodes weights once per K block, then branches only activation staging, local reads, WMMA correction, and masked stores when the final routed tile has at most 32 rows. The 32-row stage writes eight paired dwords and one final scalar LDS dword; both paths converge before the common barriers and activation-plane update. Its `a1d4-p2` conversion/store schedule uses independent groups of four BF16 roundings at priority two and explicitly restores priority before serial traversal continues. It remains at 159 VGPRs and 29,184 bytes of LDS, with 24 static WMMAs representing mutually exclusive 64- and 32-row bodies and four static barriers.

This mixed candidate is the material fitted-prior result for B1 and B4. In the reversed-order 25-repeat confirmation it measured `2.142 ms` versus `2.503 ms` for HIP at B1, a `1.168x` weighted speedup, and `6.073 ms` versus `6.202 ms` at B4, a `1.021x` weighted speedup. The independent search bank reproduced `1.205x` at B1 and `1.025x` at B4. The dominant medoids have roughly 250 active experts and many 32-row-or-smaller final remainders, which explains the transfer of the mixed-tail mechanism. Some low-weight large-group medoids remain slower, with confirmation minima of `0.931x` at B1 and `0.953x` at B4. The body is therefore retained as a strong weighted-prior research candidate, not represented as a universally dominant exact-key selector; it does not alter public dispatch or inspect routes on the host.

For B16, the scheduled 128-row `a1d2-p2` body is the better large-group control. Its nine-repeat confirmation was approximately neutral-to-positive at `1.001x` weighted, and the reversed-order 25-repeat confirmation reached `21.756 ms` versus `21.823 ms` for HIP, or `1.003x`, with every medoid at least `0.997x`. The `a1d4-p2` neighbor reached only `1.002x` in its initial screen. Those movements are below the normal two-percent promotion margin and should not be mistaken for a durable large-key win. The B16 mechanism remains a resource-clean correctness and schedule control while a new representation, ownership, or overlap premise is required for a meaningful improvement.

A separate reversed-order 25-repeat complete-call audit used the fixed HIP Q8_1 F16_D4S4 quantizer, allocated one shared workspace per HIP/GGTensile pair, and launched fresh quantization immediately before each timed multiply. Every confirmation medoid remained bitwise exact. Weighted installed-over-candidate complete-call ratios were `1.2167x` at B1, `1.0311x` at B4, and `1.0006x` at B16. Quantizer inclusion therefore preserved the mixed-tail margin and did not create a complete-call parity failure; the shared workspace allocation and producer contract remain fixed infrastructure rather than candidate-specific work.

### Final prequantized multiply-only performance

The final table reports prequantized multiply-only throughput. HIP and GGTensile consume the same activation workspace produced by the shared HIP quantizer, so activation quantization and other complete-call work are excluded. Nominal dense-equivalent throughput is `2 * aggregate rows * N * K / time`, doubled for paired two-projection kernels. GGTensile/HIP speedup is HIP body time divided by GGTensile body time.

| Aggregate rows | Public catalog hash | HIP TFLOPS | GGTensile TFLOPS | GGTensile/HIP speedup |
| ---: | :--- | ---: | ---: | ---: |
| 16,384 | `ggsol_9c98efab3bdeda2b` | 13.730 | 16.042 | 1.1684x |
| 65,536 | `ggsol_110dda8ec3bcfc8d` | 22.160 | 22.633 | 1.0213x |
| 262,144 | `ggsol_2aee91a9195cc10a` | 25.192 | 25.270 | 1.0031x |

The selected entries passed the retained route-correctness, resource, and deterministic-build checks.

### Reopened routed prologue and address micro-experiments

The final instruction review reopened four narrowly bounded mechanisms without reopening geometry, decode arithmetic, LDS scheduling, VOPD pairing, or public integration. R1 packs the aligned 64-byte kernarg block from eight `s_load_dwordx2` instructions into one `s_load_b256` and four `s_load_b64` instructions while preserving offsets, registers, the exact-shape guard, route loads, and pointer rebasing. R2 loads adjacent non-first `row_begin` and `row_end` cumulative `int32` offsets with one scalar 64-bit transaction, retaining the first-route special case and all invalid-route guards. R3 removes the multiply by the already-guarded zero high expert-stride word and strength-reduces the power-of-two BF16 output-row address, including a legal `v_lshl_add_u32` combine. R4 checks whether the target assembler exposes a usable direct-to-LDS instruction; it is an ISA/toolchain capability probe rather than a presumed kernel mechanism.

Q4_K B1 is the primary shared-mechanism screen because its K512 body gives route and address setup the largest plausible fraction of complete-call time. Every candidate must build independently twice, pass strict gfx1151 code-object v5 and resource inspection, and match the current generated parent bitwise on boundary, repeated-ID, sparse/skewed, first-route, unaligned route-index, invalid-expert, and invalid-offset controls. Timing uses the deterministic 512-draw Qwen fitted prior, five weighted medoids, three warmups, nine order-controlled repeats, the current generated parent, and an adjacent installed HIP control. Only a greater-than-two-percent candidate advances to reversed-order 25-repeat confirmation and cross-format transfer.

### Reopened route and address results

Two independent artifact passes built the current parent and R1-R3 for R35 plus every retained Q2_K, Q4_K, Q5_K, and IQ2_S production key, 16 problem keys in total. Corresponding source and code objects were byte-identical between passes. Strict inspection found no resource movement, private storage, VGPR spills, or SGPR spills. At Q4_K B1, every variant remained at 159 VGPRs, 40 SGPRs, 29,184 LDS bytes, 24 static WMMAs, and four barriers.

R1 reduced the generated scalar-load count from 11 to eight and reduced the Q4_K B1 code object by 24 bytes. R2 removed one route-prologue instruction and proved that a DWORD-aligned `s_load_b64` can read the adjacent cumulative offsets even when the dynamic byte offset is four modulo eight. The RDNA 3.5 scalar-memory specification states that `OFFSET` has no alignment restriction and that scalar addresses ignore only the low two bits. R3 removed one scalar multiply, replaced two output multiplies with two `v_lshl_add_u32` instructions, reduced VALU issue count by two, and reduced the Q4_K B1 code object by 24 bytes. None changed registers, LDS, barriers, waits, VOPD pairing, VMEM count, or arithmetic values.

The bounded device matrix covered all four formats. It exercised the first route, non-first route indices whose paired-offset addresses were zero, four, and eight bytes from the allocation, boundary tails, repeated physical experts, skewed routes, deterministic reruns, active-weight and activation mutations, inactive-expert mutation, invalid expert ID 256, and an out-of-range final offset. Every R1-R3 output was bitwise equal to its current generated parent, every active mutation changed output, every inactive mutation was inert, and both invalid cases left the destination untouched.

The prescribed Q4_K B1 confirmation-prior screen produced:

| mechanism | weighted candidate | adjacent installed | installed / candidate | candidate-time change from parent |
| --- | ---: | ---: | ---: | ---: |
| current generated parent | 2.1537 ms | 2.5113 ms | 1.1661x | control |
| R1 packed kernarg | 2.1543 ms | 2.5170 ms | 1.1684x | +0.030% |
| R2 paired route bounds | 2.1518 ms | 2.5223 ms | 1.1722x | -0.086% |
| R3 exact address | 2.1519 ms | 2.5177 ms | 1.1700x | -0.083% |

All 20 timed outputs were bitwise equal to the adjacent installed control. R1 was neutral-to-regressive, while the R2 and R3 movements were below one tenth of one percent. None approached the greater-than-two-percent gate, so no reversed-order 25-repeat confirmation or Q5_K/IQ2_S timing transfer was run.

R4 is unavailable in the configured gfx1151 assembler. The local RDNA 3.5 XML names `GLOBAL_LOAD_LDS_B32` and `BUFFER_LOAD_LDS_B32` encodings, but target exposure is narrower than the architecture inventory. The rocISA-style buffer `dword`, `b32`, `dwordx4`, and `b128` forms were rejected with invalid operands; LLVM's `global_load_dword ... lds` spelling rejected the LDS operand; and the canonical `global_load_lds_dword` form reported that the instruction is unsupported on gfx1151. No activation direct-to-LDS body can therefore be emitted by the current toolchain.

R1, R2, and R3 are rejected by timing under the current routed representation. R4 is unsupported by the current gfx1151 assembler target. Existing solution identities, selected source, public dispatch, generated bundles, packaging, and HIP fallback remain unchanged. The retained scripts, reports, and independent artifacts are under `~/tmp/torch-ggml-ops/` without embedding object identities in this record.

### Post-review disposition: zero-bank initialization

The later paired-source audit does not transfer a zero-bank hoisting candidate to Q4_K. The decoded-LDS Q4_K body already initializes its zero bank once per row tile, so it has no repeated per-projection lifetime matching the paired `v124:v131` finding. The existing route/address and direct-to-LDS closures remain scoped to their recorded Q4 body and gates; no new Q4 local accumulator experiment is pending.

## Recursive Final Review

A fresh recursive pass classifies the current state as follows. The direct-global body remains the routing and arithmetic control. The 64-row decoded-LDS schedule with mixed 32-row tails is retained for the B1/B4 research keys because it clears the complete-call gate with material weighted margin. The scheduled 128-row `a1d2-p2` body is retained only as the B16 control because its complete-call margin is approximately neutral. The 128-row initial body, serialized 64-row body without the metadata schedule, alternate local epilogue schedules, four-wave direct-global geometry, packed kernarg loads, paired route-bound loads, and exact address reductions are rejected or superseded by measured timing. Direct-to-LDS is unsupported by the configured gfx1151 assembler.

The remaining mapped mechanisms have no unmeasured first-order premise under the current contract. Flattened or persistent routing would add setup and synchronization to a workload with hundreds of independent output-column workgroups; SplitK would add partial-output and reduction storage to `K=512`; and a larger decoded tile already loses to LDS and VGPR residency. The residual B16 gap is below the promotion margin, while the slower low-weight large-group medoids in the B1/B4 mixed candidate require a new ownership or representation premise rather than another local wait or epilogue sweep. The recursive review therefore finds no additional actionable in-contract mechanism with material expected margin. Public dispatch and packaging remain intentionally untouched.

Before declaring the Q4_K grouped campaign complete, reread this record, `docs/ggtensile_plan.md`, `docs/grouped_mmq_fwd_optimization.md`, the ordinary Q4_K forward record, grouped HIP source and normalized ISA, the TensileLite grouped writer and current grouped test configurations, relevant Composable Kernel fixed-NK scheduling code, every generated grouped artifact and timing report, rejected candidates, target ISA material, and relevant forward and backward mechanisms.

Classify every remaining idea as retained and measured; rejected by correctness, resources, timing, or reproducibility; contract-incompatible or deferred with an explicit prerequisite; or actionable with an exact target and qualification gate. Re-evaluate external generator guards as implementation facts rather than architectural conclusions. Implement and qualify every actionable finding, then repeat the complete review from the new premise. Completion is valid only when a fresh recursive pass finds no actionable in-contract mechanism and every selected exact key is correct, deterministic, resource-clean, and faster than its exact HIP control.

This rule is global across related GGTensile directions, formats, and shapes. Evidence may transfer, but ownership, lifetimes, synchronization, arithmetic order, resources, correctness, and exact-key timing must be re-derived here. Public API integration is explicitly outside this experiment and is not a completion criterion.

## Post-Audit B16 Ownership Reopening

Status: G4 measured and rejected by exact kernel-body timing; G5 and G6 remain planned and unmeasured. The route-prologue, direct-to-LDS, local epilogue, and ordinary one-block ping-pong results remain closed. The G4 premise amortized decode across serial row tiles of one routed workgroup.

### G4: Route-persistent full-K decoded weights

The current row loop decodes both packed K512 blocks again for every 64- or 128-row tile. Add a B16-first identity that decodes the two blocks once, before the row loop, into two immutable LDS weight images. Keep activation LDS disjoint, then consume block zero and block one for every serial row tile without repeating packed-weight VMEM, nibble expansion, or metadata preparation. This remains direct packed consumption inside one kernel launch; it adds no prepared representation, external workspace, invalidation rule, or ABI field.

Derive the complete LDS size and occupancy before emission and reject if either image aliases activation storage or exceeds the target limit. Preserve the selected 128-row `a1d2-p2` arithmetic, output ownership, route guards, and exact BF16 conversion. The historical approximately 56-KiB dense ping-pong loss is context, not a rejection: that body sought same-row overlap, while G4 must account for decode work removed across every later routed row tile.

Screen the five fitted B16 medoids against the selected 128-row parent and adjacent HIP, reporting route tile counts, decode amortization, resources, body latency, and complete-call latency. Require a stable greater-than-two-percent complete-call gain before B4 or B1 transfer. Exact route, tail, malformed-route, active/inactive mutation, sentinel, deterministic-build, and independent-reference gates remain mandatory. G4 is exact and requires no model integration.

The implemented G4 body held two complete decoded Q4_K images in LDS. One image occupied `64 * 304 = 19,456` bytes; two images plus the existing 512-byte prefix and 128-row activation stage produced 57,856 bytes of LDS. The inspected gfx1151 artifact used 239 VGPRs, 40 SGPRs, 64 static WMMAs, eight barriers, zero private bytes, and zero spills. Existing one-image generated sources remained byte-identical.

A first multi-tile qualification exposed an operand-order error in the image-pointer restore: `v_sub_nc_u32 dst, 19456, dst` computed `19456 - dst`. Reversing the two source operands restored both weight and metadata pointers correctly. The repaired body was deterministic and bitwise equal to the selected 128-row parent for single-route lengths `1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 256` and for uniform, skewed, sparse, and boundary distributions at B1, B4, and B16. At `R=16,384` it also matched the installed packed control bitwise; its independently dequantized BF16 reference normalized RMSE was `0.0127600`.

The repaired candidate failed the performance discriminator before complete-call or fitted-prior confirmation was warranted:

| aggregate rows | distribution | full-weight body | 128-row parent | parent / full-weight |
| ---: | --- | ---: | ---: | ---: |
| 16,384 (B1) | uniform / skewed / sparse / boundary | 3.732 / 4.070 / 3.410 / 3.602 ms | 3.004 / 3.455 / 2.759 / 2.914 ms | 0.805x / 0.849x / 0.809x / 0.809x |
| 65,536 (B4) | uniform / skewed / sparse / boundary | 6.079 / 7.533 / 7.144 / 7.695 ms | 5.229 / 6.533 / 6.300 / 6.665 ms | 0.860x / 0.867x / 0.882x / 0.866x |
| 262,144 (B16) | uniform / skewed / sparse / boundary | 22.344 / 23.082 / 22.689 / 23.650 ms | 19.946 / 21.004 / 20.898 / 21.068 ms | 0.893x / 0.910x / 0.921x / 0.891x |

Even the B16 uniform route, with eight 128-row tiles per route and therefore the strongest decode-amortization premise, regressed by about 12%. The larger immutable LDS footprint and doubled static activation/MMA body outweighed the removed re-decodes. G4 is rejected by timing, its speculative solution identity and lowering were removed, and dispatch remains unchanged. Qualification and timing artifacts are retained under `~/tmp/torch-ggml-ops/`.

### G5: Generated synchronization and G6 output conversion

G5 adds typed VMEM/LDS events and row-loop LDS liveness to the current B16 parent, first reproducing its source byte-for-byte and then deriving counter-specific waits. Barrier removal is allowed only with a cross-wave proof for the exact current or G4 image lifetime. Standalone barrier deletion remains closed. G5 stays bit-exact and requires no model integration.

G6 separately compares final-output `RNEPreserveNaN`, `BiasRound`, and `Truncate`. Numerical tests use finite inputs, reject non-finite outputs, and report error distributions; the kernel emits no NaN/Inf branch or repair path. G6 remains outside exact selections and requires model integration. Do not compose G4, G5, and G6 before each independent discriminator is complete.

After G4's measured rejection, run the resource-neutral G6 screen before further G5 optimization. K512 makes output conversion a repeated low-arithmetic-intensity cost, so G6 has the larger remaining body-delta prior. G5 is still valuable for generator correctness and cross-format reuse, but recent fixed-map scheduling results place its expected performance movement in the low single digits. Neither assessment is a timing result.
