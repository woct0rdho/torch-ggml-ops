# GGTensile Grouped MMQ Backward Q2_K Experiment

## Purpose

Implement and optimize isolated gfx1151 grouped Q2_K backward kernels for the DeepSeek routed down projection. Public dispatch, generated bundle tables, extension registration, packaging, and HIP fallback remain unchanged pending a separate integration review.

The promotion objective is the fitted DeepSeek learned/hash routing-prior weighted sum of per-medoid median complete-call latency. Learned and hash components retain the declared reporting weights `40/43` and `3/43`. Uniform, skewed, sparse-ID, and boundary routes remain correctness and diagnostic controls rather than post hoc ranking vetoes.

## Exact Contract

For routed GEMM `g`:

```text
dY_g[M_g,4096] x W_g[4096,2048] -> dX_g[M_g,2048]
```

The exact aggregate-row keys are `R={12288,49152,196608}`. The physical Q2_K expert bank is `[256,4096,672]`: each 256-value block occupies 84 bytes, each packed row contains eight blocks, and each expert occupies 2,752,512 bytes. Inputs and outputs are contiguous BF16, accumulation is FP32 WMMA V1, and output conversion is BF16 RNE.

The grouped ABI remains the 56-byte routed backward research ABI: `grad_output`, `packed_weight`, `grad_input`, `expert_indices`, `expert_offsets`, `num_experts`, `rows`, and `bytes_per_expert`. Route count and optional split ownership are launch geometry. No host route inspection, dense shadow, prepared bank, atomics, reduction workspace, or companion setup kernel is in scope.

The authoritative packed control is `/home/wd/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf`, tensor `blk.0.ffn_down_exps.weight` (`Q2_K`, `[256,4096,672]`). The independent oracle dequantizes only selected routed experts to BF16 before matmul.

## Arithmetic Boundary

Q2_K cannot reuse Q4_K/Q5_K arithmetic by renaming the format. A block stores sixteen scale/minimum bytes, 64 two-bit payload bytes, and FP16 `d`/`dmin`. Each decoder lane owns one aligned 16-value group, shares its payload shift and scale/minimum pair across those values, converts decoded weights to BF16 in LDS, and then reuses the established FP32-WMMA and BF16-RNE leaf.

The installed HIP baseline selects M64/N64/U1 below 128 rows per route, M128/N64/U2 from 128 through 511, and M128/N64/U1 above that. This is evidence for initial geometry and ownership controls, not assumed GGTensile ranking. Q2-specific packed reads, metadata lifetimes, LDS layout, schedule, and route splitting must be measured independently.

## Qualification

Correctness covers full exact rows, non-aligned route tails, first/non-first routes, sparse and repeated expert IDs, deterministic reruns, gradient and route mutation, active/inactive expert-weight mutation, invalid experts, malformed first/final offsets, and untouched sentinels. Candidate versus packed HIP must be BF16 bit-exact; the independently dequantized reference must remain finite with NRMSE below `0.01`.

Inspection requires gfx1151, wave32, code object v5, exact 56-byte metadata, bounded VGPR/SGPR indices, zero private bytes and spills, no scratch/calls/dynamic stack, and derived static WMMA/barrier counts. Independent generate/build/inspect roots must produce byte-identical source and HSACO.

Timing uses warmed rotating GPU events and includes output allocation in both HIP and candidate complete-call paths. Search uses five fitted medoids from each DeepSeek learned/hash component. Retained changes receive disjoint confirmation with 5 warmups, 25 repeats, and reversed rotating order. TFLOPS is `2 * R * 2048 * 4096 / (latency_ms * 1e9)`.

## Planned Search

- Add strict Q2_K backward identity and exact DeepSeek shapes without changing retained Q4_K/Q5_K source hashes or the grouped ABI.
- Implement the dedicated Q2_K packed reader and decoded-B LDS arithmetic, then qualify a minimal M128/N128 pilot against packed HIP and the independent oracle.
- Establish M64/N64, M128/N64, and M128/N128 controls with single-LDS and bounded pipeline schedules.
- Screen Q2 payload/metadata extraction, LDS padding/swizzle, route tails, and split factors only where emitted ISA or ownership changes.
- Confirm per-key winners on disjoint learned/hash medoids, qualify full rows, rebuild independently, and report HIP/GGTensile TFLOPS and speedup.

## Completion Record

This section is updated after every coherent implementation or failed experiment.

### Campaign opened

The completed grouped Q4_K/Q5_K shell, installed DeepSeek Q2_K backward bodies, grouped Q2_K forward lowerer, packed model tensor, and fitted learned/hash prior were audited. The exact backward problem is `(R,2048,4096)` for `R={12288,49152,196608}` with expert stride 2,752,512 bytes. The first coherent change will add a strict Q2 identity, dynamic grouped row-stride handling, generalized benchmark dimensions/prior selection, and a dedicated width-16 two-bit decoder while preserving Q4/Q5 generated source.

### Strict Q2 identity and decoded-LDS pilot

The backward-only format registry and grouped contract now admit Q2_K while ordinary forward remains unchanged. Q2 grouped identities require only the three exact DeepSeek shapes, emit a distinct symbol/hash, validate the `[256,4096,672]` packed tensor, and guard the full 2,752,512-byte expert stride. Grouped row-byte masking and the benchmark now derive dimensions from the key. DeepSeek timing combines ten learned/hash medoids with the declared `40/43` and `3/43` component weights.

The dedicated decoder maps each lane to one aligned 16-value group, loads one 128-bit payload segment plus its scale/minimum byte and FP16 `d`/`dmin`, performs the shared 2-bit shift, and writes BF16-RNE decoded values to the existing LDS/FP32-WMMA leaf. The M128/N128 SIA2 pilot passes the 625-row boundary matrix bit-for-bit against packed HIP, including deterministic, gradient, route, active/inactive weight, malformed-route, and sentinel controls. Independent-reference NRMSE is `3.78e-5`. The current artifact uses 190 VGPRs, 35 SGPRs, 8 KiB LDS, 32 static WMMAs, and two barriers with zero private bytes or spills.

The first M128/N64 double-LDS artifact failed packed HIP, determinism, mutations, and the independent oracle. The Q2 one-row metadata allocation exposed an existing pipeline lifetime requirement: the shared decoded-B pipeline uses a second scale-adjacent register for LDS read addressing while next-tile metadata is pending. Reserving that register removes the alias with the swizzled write-address bank. Rebuilt B1/B4/B16 double-LDS artifacts are exact and deterministic at 139 VGPRs; no wait or arithmetic waiver was used. All initial SIA5 geometry controls also pass. Current serial resources are 87 VGPR for M64/N64, 135 for M128/N64, and 206 for M128/N128, all at 35 SGPR with zero private bytes or spills.

Regenerated retained Q4 and Q5 B1 sources remain byte-identical at SHA-256 `06a0b9d459b98d7fef7b612ba5bf608080b00f5323aa5e1893422b1d44c654b8` and `7e360fd3861fc0daf6622d49f1d248918524ece1948cebb1473d5a46667762a1`.
