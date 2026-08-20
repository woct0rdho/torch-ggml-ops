# GGTensile Grouped MMQ Backward Pair IQ2_XXS Experiment

## Purpose

Implement and optimize isolated gfx1151 grouped paired IQ2_XXS backward kernels for the DeepSeek routed gate/up projections. Public dispatch, generated bundles, extension registration, packaging, and HIP fallback remain unchanged pending a separate integration review.

The production objective is lower complete-call latency than the exact installed HIP control at every required aggregate-row key. Search profiles may rank intermediate candidates, but promotion requires exact production-row correctness, zero spills/private storage, deterministic artifacts, and formally confirmed timing at B1, B4, and B16.

## Exact Contract

For routed GEMM `g`:

```text
dG_g[M_g,2048] x W_gate_g[2048,4096]
+ dU_g[M_g,2048] x W_up_g[2048,4096]
-> dX_g[M_g,4096]
```

The exact aggregate-row keys are `R={12288,49152,196608}`. Each physical IQ2_XXS expert bank is `[256,2048,1056]`: each 256-value block occupies 66 bytes, each packed row contains sixteen blocks, and each expert occupies 2,162,688 bytes. Both gradient outputs and the shared gradient input are contiguous BF16. The arithmetic contract uses signed IQ2_XXS codebook values, the odd per-K32 group scale, FP32 scale arithmetic, one shared FP32 accumulator set for both projections, and one final BF16 RNE store.

The isolated research ABI is the production-compatible 72-byte specialized pair ABI: `first_grad_output`, `second_grad_output`, `first_packed_weight`, `second_packed_weight`, `grad_input`, `expert_indices`, `expert_offsets`, `num_experts`, `rows`, and `bytes_per_expert`. Route count and any typed split ownership remain launch geometry. Independent projection workgroups, an intermediate destination, atomic accumulation, or reloading a stored first projection are outside the fused-pair contract.

The authoritative codebook is extracted at generation time from `csrc/vendor/llama_cpp/iq2_xxs_grid.cuh`. The independent oracle dequantizes only selected routed experts and sums the two routed matmuls in FP32 before BF16 conversion.

## Production Control

Installed HIP dispatches `GroupedBwdPairIQ2XXSN2048K4096M64N64` at B1. Exact aggregate rows 49,152 and 196,608 dispatch `GroupedBwdTunedPairIQ2XXSN2048K4096M128N64`. Both use 128 threads as four wave32 waves, N64, K32, cooperative width-16 decode, swizzle4, separate LDS tiles for the two weight banks, inactive-M consumer suppression, and one final store. The grid is `(4096 / 64, num_groups, 1)`.

The specialized ABI validates exact row count, physical bank geometry, common route geometry, expert count, and the 2,162,688-byte expert stride. Invalid experts, non-increasing offsets, negative starts, and offsets beyond `rows` make that route inert.

## Qualification

Correctness covers exact rows, non-aligned route tails, first and non-first routes, sparse and repeated expert IDs, deterministic reruns, independent mutation of both gradient outputs and both active weight banks, inactive-expert mutations, malformed route controls, and untouched sentinels. Candidate versus installed packed HIP must be BF16 bit-exact. An independently dequantized bounded reference must remain finite and meet the established normalized-error gate.

Inspection requires gfx1151, wave32, code object v5, exact pair metadata, bounded register indices, zero private bytes and spills, no scratch, calls, or dynamic stack, and derived static WMMA/barrier counts. Independent generate/build/inspect roots must produce byte-identical source and HSACO.

Timing uses warmed rotating GPU events and includes equivalent output allocation in candidate and installed complete-call paths. Retained finalists receive disjoint confirmation with at least 5 warmups and 25 repeats, reversed rotating order, independent and paired robust confidence intervals, and production-row validation. Pair TFLOPS is `4 * R * 2048 * 4096 / (latency_ms * 1e9)`.

## Planned Search

- Reuse the ordinary backward tile, WMMA, route, ABI, store, runtime, and inspection machinery.
- Add typed IQ2_XXS width-16 packed reads and signed codebook reconstruction. Keep its 66-byte format and odd K32 scale distinct from IQ2_S.
- Establish a correct M64/N64 `InterleavedDepthU` fused anchor. It decodes and consumes each bank serially through one LDS tile and intentionally prioritizes semantic clarity over synchronization count.
- Compare M64 and M128 early. Then test only mechanisms justified by measured deficits: dual LDS, concurrent bank reads, activation and packed-weight pipelining, route ownership, decode scheduling, swizzle4, and exact geometry.
- Retain only mechanisms with explicit correctness, resource, timing, and identity records. Remove failed-only source identities before final review.

## Recursive Final Review

After each optimization round, reread the contract, retained source and machine code, installed HIP control, timing reports, correctness reports, resource inspection, and rejected experiments from first principles. Classify every remaining idea as retained and measured; rejected by correctness, resources, timing, or reproducibility; contract-incompatible or deferred with a prerequisite; or actionable with an exact target and gate. Implement every actionable finding and repeat the review.

Completion requires exact bounded and production correctness, zero spills/private storage, deterministic source and HSACO, complete production-row timing with confidence, focused and broad tests, documentation, cleanup of failed-only identities, and a final recursive pass with no actionable in-contract mechanism.

## Completion Record

### Campaign opened and implementation boundary audited

The target is the DeepSeek gate/up backward pair, not an ordinary backward kernel and not either projection in isolation. The physical bank shape is `[256,2048,1056]`, giving a 1,056-byte packed row and a 2,162,688-byte expert stride. B1 uses the installed M64 body; B4 and B16 use the separately qualified M128 body.

The existing paired GGTensile framework already owns the fused 72-byte ABI, route validation and rebasing, bounded M tails, a shared FP32 accumulator set, first-then-second projection accumulation, and one BF16 epilogue. Its simplest `InterleavedDepthU` schedule is quant-agnostic once packed reads and decode are supplied. Later dual-LDS and pipelined schedules currently contain IQ2_S-specific metadata helpers and are not admitted for the first IQ2_XXS identity.

The authoritative forward IQ2_XXS lowering and HIP `decode_backward_tile_group<GGML_TYPE_IQ2_XXS,16>` establish the packed semantics. The first retained source change will add a dedicated width-16 IQ2_XXS reader/decoder while reusing the generic backward tile and paired writer. No performance claim is recorded until the bounded artifact is bit-exact, resource-clean, and independently inspected.

### First fused IQ2_XXS anchor

The backward format contract now admits IQ2_XXS without conflating its 66-byte block with IQ2_S. The dedicated reader maps each of 128 threads to one aligned 16-value group in the 64-column by K32 decoded tile, loads one 64-bit index/sign word plus the shared FP16 block scale, reconstructs the two authoritative grid entries in staged LDS, expands each seven-bit sign group with odd parity, forms `d * (2*scale+1) / 8` in FP32, and stores BF16 RNE values through the reusable XOR-4 decoded-weight layout. The pair body reuses the routed ABI, bounded M ownership, BF16 WMMA, shared FP32 accumulators, projection pointer swaps, and one final store.

Canonical M64/N64 and M128/N64 single-LDS keys round-trip at bounded and production rows. Strict gfx1151 assembly inspection gives:

| Geometry | VGPR / SGPR / LDS | Static WMMA / barriers | Private bytes / spills |
| --- | ---: | ---: | ---: |
| M64/N64 | `87 / 41 / 6,144 B` | `16 / 5` | `0 / 0` |
| M128/N64 | `127 / 41 / 6,144 B` | `32 / 5` | `0 / 0` |

Both artifacts have bounded register indices, exact 72-byte metadata, wave32, code object v5, no scratch, calls, or dynamic stack. The M64 R35 candidate matches the installed specialized M64 control in all 143,360 BF16 outputs on real DeepSeek gate/up banks. An expanded R257 matrix also has zero differing elements for one-route full/tail ownership, a short first route followed by full tiles, three nonaligned routes, repeated experts, and mixed 64/65-row boundaries. Every result is finite and no destination sentinel remains.

The first lane mapping review caught and corrected an implementation error before this qualification: the lower/upper 16-value half of a K32 group is selected by lane bit zero. The retained source and test fix that mapping explicitly. The single-LDS body is now the correctness anchor. Full mutation, malformed-route, independent-oracle, production-row, reproducibility, and timing qualification remain outstanding.

### Coverage closure for the retained decoder

The writer coverage test now dispatches a normal IQ2_XXS emitter through the generic decode-prepare and decode-chunk helpers, and separately verifies that a physical plan without the staged codebook is rejected. The direct targeted test passes. A run containing only the writer-coverage and pair modules reports the expected shared-session coverage failure because other backward-writer paths are not imported/executed in that process; it is not a product failure. The complete `tests/ggtensile` session reached 679 passed before this closure and is being rerun after the two IQ2_XXS dispatch lines were covered.
