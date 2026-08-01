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

1. Fixed HIP quantizer plus HIP packed multiply: public HIP control.
2. Fixed HIP quantizer plus GGTensile packed multiply: complete candidate.
3. A prequantized-workspace bracket of HIP and GGTensile multiply bodies: diagnosis only.

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
- compare multiplication output bit-exactly with the exact HIP multiply when identical integer-WMMA and output ordering are retained;
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

1. Exact M/N ownership and launch density.
2. Q4_K global payload/metadata reads and DS4 global/LDS load layout.
3. Int8 WMMA operand order, clamp necessity, integer accumulator placement, and scale/min correction scheduling.
4. LDS padding/XOR layout, vector local reads, barriers, waits, and prefetch schedules.
5. Bounded next-tile reads and overlap only when the prequantized lower bounds expose a real gap.
6. Smaller M keys after large-M choices close.

Candidates pass correctness and inspection before nine-repeat serial screens. A new resource-bearing mechanism requires a stable gain above 2% in a 25-repeat rotating assembly-control confirmation. Unconditional instruction or resource reductions may remain when neutral or favorable.

### Phase 4: Final evidence

For every selected key, measure fixed-quantizer complete latency, prequantized multiply latency, int8-WMMA/DS4-Q4 decode floors where diagnosis is ambiguous, resource reports, independent rebuilds, producer/input/packed mutation correctness, and a 25-repeat serial confirmation. Selection requires every exact key to beat HIP; a weighted average never authorizes a slower key.

## Recursive Optimization-Exhaustion Review

Before declaring completion, reread this plan, the Q4_K HIP source and normalized ISA, all forward artifacts and timing reports, the dense-forward HIP record, completed GGTensile backward records, grouped histories, lower bounds, rejected candidates, `~/rdna35-isa-markdown/`, AMD LLVM definitions/tests, and relevant CK/TensileLite material.

Classify every remaining idea as retained and measured; rejected by correctness, resources, timing, or reproducibility; contract-incompatible or deferred with a prerequisite; or actionable with a target key and measurement gate. An actionable idea must be implemented and measured, then the full review repeated from the new premise. Completion is allowed only after a fresh review finds no actionable in-contract mechanism, every selected key beats HIP, and the residual bottleneck is quantified.

## Completion Record

- [x] Forward identity, fixed-quantizer control runtime, inspection, and exact 12-key inventory.
- [ ] Correct Q4_K signed-int8 WMMA control for K512/K2048/K4096.
- [ ] High-weight M32768 ownership, LDS, and scheduling search.
- [ ] Per-key catalog with all exact keys faster than HIP.
- [ ] Complete and prequantized lower bounds with residual bottleneck explanation.
- [ ] Independent rebuild reproducibility, mutation coverage, and 25-repeat confirmation.
- [ ] Recursive optimization-exhaustion review with no actionable mechanism remaining.
- [ ] Public dispatch and artifact packaging, deferred.

## Implementation Record

The initial forward infrastructure uses operation-specific writer modules: `kernel_writer_assembly_mmq_fwd.py` for forward and `kernel_writer_assembly_mmq_bwd.py` for backward. Inventories and solution catalogs live under `tools/ggtensile/configs/`. CLI generation selects the writer from the strict `ProblemType`; it does not infer the operation from dimensions or filenames.

The first control is deliberately diagnostic: one 32-thread wave owns a 16x16 output tile, reads Q4_K and Q8_1 DS4 operands directly from global memory, issues two signed/clamped integer WMMAs per 32-value group, applies per-output-column Q4_K scale/min metadata in FP16 followed by FP32 accumulation, and stores BF16 round-to-nearest-even output. It declares 88 VGPRs, 16 SGPRs, zero LDS, zero private bytes, and zero spills. The static loop body contains 16 integer-WMMA instructions and no barriers.

A standalone 16x16 signed-int8 WMMA mapping pilot matched all 256 CPU `B @ A.T` outputs. The first production control `(M,N,K)=(2048,512,2048)` then consumed the unchanged installed HIP DS4 workspace and matched the public HIP complete output bit-for-bit for all 1,048,576 BF16 elements. The initial mapping defect was an emitter register alias between the persistent packed-weight row and a metadata-address temporary; only output column zero was correct before that alias was removed. This failure mode is retained as a writer-coordinate test target rather than treated as a tuning result.
