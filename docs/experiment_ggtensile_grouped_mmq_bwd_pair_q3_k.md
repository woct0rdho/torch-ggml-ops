# GGTensile Grouped MMQ Backward Pair Q3_K Experiment

## Purpose

Implement and optimize isolated gfx1151 grouped paired Q3_K backward kernels for the Qwen routed gate/up projections. Public dispatch, generated bundles, extension registration, packaging, and HIP fallback remain unchanged pending a separate integration review.

The promotion objective is the Qwen learned routing-prior weighted sum of per-medoid median complete-call latency. Fitted routes rank candidates. Uniform, skewed, sparse-ID, repeated-ID, and boundary routes remain correctness and diagnostic controls rather than substitutes for the fitted objective.

## Exact Contract

For routed GEMM `g`:

```text
dG_g[M_g,512] x W_gate_g[512,2048]
+ dU_g[M_g,512] x W_up_g[512,2048]
-> dX_g[M_g,2048]
```

The exact aggregate-row keys are `R={16384,65536,262144}`. Each physical Q3_K expert bank is `[256,512,880]`: each 256-value block occupies 110 bytes, each packed row contains eight blocks, and each expert occupies 450,560 bytes. Both gradient outputs and the shared gradient input are contiguous BF16. The arithmetic contract accumulates both projections into one FP32 WMMA accumulator set before one BF16 RNE conversion and store.

The isolated research ABI is the production-compatible 72-byte specialized pair ABI: `first_grad_output`, `second_grad_output`, `first_packed_weight`, `second_packed_weight`, `grad_input`, `expert_indices`, `expert_offsets`, `num_experts`, `rows`, and `bytes_per_expert`. Route count and typed ownership remain launch geometry. Independent projection workgroups, an intermediate destination, atomic accumulation, or reloading a stored first projection are outside the fused-pair contract.

The authoritative packed controls are the Qwen gate/up Q3_K expert tensors with logical shape `[256,512,2048]`. The independent oracle dequantizes only selected routed experts and sums the two routed matmuls in FP32 before BF16 conversion.

## Production Control

Installed HIP selects `GroupedBwdPairQ3KN512K2048M64N64` below `128 * num_groups` rows and `GroupedBwdPairQ3KN512K2048M128N64` otherwise. Both launch 128 threads as four wave32 waves, use N64 and K32, decode aligned width-16 Q3_K groups from both banks into separate padded LDS tiles, accumulate both projections into one FP32 register set, and perform one final BF16 store. The grid is `(2048 / 64, num_groups, 1)`.

The installed specialized ABI validates exact row count, physical bank geometry, common route geometry, expert count, and the 450,560-byte expert stride. Invalid experts, non-increasing offsets, negative starts, and offsets beyond `rows` make that route inert.

## Writer Reuse Boundary

The existing paired writer owns the strict fused ABI, route validation and rebasing, M64/M128 bounded ownership, activation addressing, WMMA issue, FP32 accumulator sharing, BF16 epilogue, inspection, and direct launcher. The ordinary backward Q3_K lowering already owns 110-byte block addressing, high-mask and low-payload loads, signed six-bit scale reconstruction, packed 3-bit extraction, BF16 decoded-weight LDS stores, and its register requirements. Those layers are reused through the shared paired facade only after strict Q3_K paired identity and resource derivation admit them.

No separate monolithic Q3_K assembly writer is planned. New lowering is limited to paired Q3_K scheduling that cannot be expressed by the quant-neutral pair shell: two disjoint padded LDS images, concurrent bank reads, direct second-projection pointers, or K32 overlap. Each such mechanism needs a typed identity, exact source path, resource proof, and measured premise.

## Qualification

Correctness covers exact rows, non-aligned route tails, first and non-first routes, sparse and repeated expert IDs, deterministic reruns, independent mutation of both gradient outputs and both active weight banks, inactive-expert mutations, malformed routes, and untouched sentinels. Candidate versus installed packed HIP must be BF16 bit-exact; the independently dequantized BF16 reference must remain finite with NRMSE below `0.01`.

Inspection requires gfx1151, wave32, code object v5, exact pair metadata, bounded register indices, zero private bytes and spills, no scratch, calls, or dynamic stack, and derived static WMMA/barrier counts. Independent generate/build/inspect roots must produce byte-identical source and HSACO.

Timing uses warmed rotating GPU events and equivalent output allocation in candidate and installed complete-call paths. Search uses five Qwen learned medoids per physical batch. Retained changes receive disjoint confirmation with at least 5 warmups and 25 repeats, reversed rotating order, independent and paired robust confidence intervals, and production-row validation. Pair TFLOPS is `4 * R * 512 * 2048 / (latency_ms * 1e9)`.

## Planned Search

- Add strict Q3_K paired problem, solution, contract, physical-plan, runtime-control, inspection, and serialization coverage without changing IQ2_S or IQ2_XXS source identities.
- Reuse the ordinary Q3_K packed reader/decoder under the simplest fused single-LDS pair schedule as the semantic anchor. Build M64/N64 and M128/N64 at bounded and exact production rows before timing.
- Prioritize the largest measured deficits. Compare geometry first, then dual padded LDS and concurrent packed-bank reads, because installed HIP proves that decoding both banks before a shared consumer phase is viable.
- Test activation prefetch, direct second pointers, Q3 packed extraction/pairing, SIA, and K32 pipelining only when the retained parent leaves a measured latency gap or resource-clean overlap opportunity.
- Confirm per-key winners on disjoint Qwen medoids, qualify full rows, rebuild independently, and report HIP/GGTensile TFLOPS and speedup.
- Leave public selection, generated bundle integration, packaging, and HIP fallback for a separate integration review.

## Recursive Final Review

After each optimization round, reread the contract, retained source and machine code, installed HIP control, fitted reports, correctness reports, resource inspection, and rejected experiments from first principles. Do not treat an earlier rejection as permanent when a retained mechanism changes its premise.

Classify every remaining idea as retained and measured; rejected by correctness, resources, timing, or reproducibility; contract-incompatible or deferred with an explicit prerequisite; or actionable with an exact target and qualification gate. Implement every actionable finding and repeat the review from the new premise. Completion requires a fresh recursive pass with no actionable in-contract mechanism and every selected production key correct, deterministic, resource-clean, independently reproducible, and faster than its exact HIP control.

## Completion Record

This section is updated after every coherent implementation, correctness, timing, or failed experiment.

### Campaign opened and implementation boundary audited

The production target is independently confirmed as the Qwen fused gate/up backward pair at `(R,N,K)=({16384,65536,262144},2048,512)`, with physical Q3_K banks `[256,512,880]` and a 450,560-byte expert stride. Installed HIP provides exact M64/N64 and M128/N64 fused controls under the same 72-byte ABI and dispatches by the `128 * num_groups` threshold.

The paired GGTensile shell already represents route ownership, pointer rebasing, bounded M tails, one shared FP32 accumulator set, projection order, one final BF16 store, direct launch, and strict inspection. The ordinary backward quant lowering already emits the complete Q3_K reader and decoder. The first coherent implementation will admit Q3_K through those existing ownership boundaries and use the serial single-LDS pair as a correctness anchor. A distinct writer would duplicate proven route and WMMA machinery and is not justified; quant-specific paired schedules will be added only from measured deficits.

### Strict Q3_K identity and single-LDS correctness anchor

The paired problem and contract now admit only the exact Qwen Q3_K geometry in addition to the existing IQ2 formats. Canonical M64/N64 and M128/N64 identities use pad8, packed extraction, explicit full Q3 VOPD pairing, K32, serial routes, and the existing interleaved projection schedule. The installed-control launcher binds the exact production M64/M128 symbols and independently checks the 450,560-byte expert stride and dispatch threshold. Q3_K reuses the ordinary quant lowerer through the paired facade; no separate Q3 assembly writer or copied HIP body was introduced.

Two independent roots generated, assembled, linked, and inspected both geometries at rows `35`, `16384`, `65536`, and `262144`. Source, object, and HSACO hashes match across roots for all eight artifacts. M64 reports `87 VGPR / 41 SGPR / 5120 B LDS`, 16 static WMMAs, and four barriers; M128 reports `127 / 41 / 5120 B`, 32 WMMAs, and four barriers. Both have bounded register indices, exact 72-byte metadata, zero private bytes, zero spills, no scratch, calls, or dynamic stack. The durable build record is `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-q3-k-anchor/build.json`.

The 35-row four-route matrix uses authoritative block-0 gate/up banks and exercises 1/15/16/3-row ownership, first and non-first routes, experts `0/17/255/42`, deterministic reruns, independent gradient and active-bank mutations, inactive-bank mutations, route mutation, invalid experts, negative starts, oversized final offsets, and destination sentinels. Both geometries are BF16 bit-exact to public HIP and the direct installed M64 control throughout. The independent FP32 pair reference differs in 3 of 71,680 BF16 values, with maximum absolute error `3.81e-6` and NRMSE `6.89e-8`. The report is `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-q3-k-anchor/correctness.json`. The anchors clear correctness, resource, and reproducibility gates; fitted geometry timing is next.

### Fitted geometry baseline

Three warmups and nine rotating repeats over all five Qwen learned search medoids include equivalent output allocation and pre-timing exactness checks. M64 is the B1 parent at `5.3885 ms`, effectively tied with public HIP at `5.3661 ms` (`0.9958x`). M128 loses B1 by `6.1%`, but becomes the large-key parent: it reaches `13.1599 ms` at B4 versus HIP `13.3911 ms` (`1.0176x`) and `52.6262 ms` at B16 versus HIP `48.6644 ms` (`0.9247x`). M64 is rejected at B4/B16, where it reaches only `0.8290x` and `0.7586x` HIP throughput.

The B16 deficit is the largest actionable margin. The single-LDS anchor decodes and consumes each bank serially and pays four dynamic barriers per K32 step. Installed HIP proves that two disjoint padded weight images fit and are competitive. The next mechanism is therefore dual padded LDS with both projections decoded before one consumer barrier and both WMMAs before one overwrite barrier. Local schedule tuning remains secondary until that synchronization deficit is removed. The durable timing report is `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-q3-k-anchor/geometry-search9.json`.

### Dual padded LDS

The first retained paired-specific schedule assigns the two decoded Q3_K projections to disjoint 5 KiB padded LDS images. It decodes both banks before one consumer barrier, performs both projection WMMA phases into the shared FP32 accumulators, and synchronizes once before overwrite. No quant arithmetic or HIP assembly body is copied. M128 remains `127 VGPR / 41 SGPR`, LDS rises to `10240 B`, barriers fall from four to two, and private bytes and spills remain zero. Two independent bounded and production builds are byte-identical; the build report is `~/tmp/torch-ggml-ops/ggtensile-bwd-pair-q3-k-dual-lds/build.json`.

The complete 35-row gradient, bank, route, malformed-route, deterministic, sentinel, and independent-oracle matrix remains exact. The independent reference retains the same 3 differing BF16 values and `6.89e-8` NRMSE as the anchor. At B16, the five-medoid rotating screen improves the source-preserved M128 parent from `52.8582` to `51.8399 ms`, or `1.0196x`. It remains behind contemporaneous HIP at `48.4986 ms` (`0.9355x`), so dual LDS is retained as the large-key optimization parent rather than a promotion candidate. The correctness and timing reports are `correctness.json` and `timing-b16-search9.json` under the same artifact root.
