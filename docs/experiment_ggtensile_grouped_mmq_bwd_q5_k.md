# GGTensile Grouped MMQ Backward Q5_K Experiment

## Purpose

Implement and optimize isolated gfx1151 grouped Q5_K backward kernels for the Qwen routed down projection. Public dispatch, generated bundle tables, registration, packaging, and HIP fallback remain unchanged pending a separate integration review.

The fitted Qwen routing-prior weighted sum of per-medoid median complete-call latency is the promotion objective. Uniform, skewed, sparse-ID, and boundary routes are correctness and diagnostic controls rather than ranking vetoes. Every retained artifact must remain exact against packed HIP, deterministic, independently reproducible, and free of private storage and spills.

## Exact Contract

For routed GEMM `g`:

```text
dY_g[M_g,2048] x W_g[2048,512] -> dX_g[M_g,512]
```

The exact aggregate-row keys are `R={16384,65536,262144}`. The physical Q5_K expert bank is `[256,2048,352]`: each 256-value block occupies 176 bytes, so each packed weight row is 352 bytes and each expert is 720,896 bytes. Inputs and outputs are contiguous BF16, accumulation is FP32 WMMA V1, and output conversion is BF16 RNE.

The grouped ABI remains the 56-byte Q4_K research ABI: `grad_output`, `packed_weight`, `grad_input`, `expert_indices`, `expert_offsets`, `num_experts`, `rows`, and `bytes_per_expert`. Route count and optional split ownership are launch geometry. No host route inspection, dense shadow, prepared bank, atomics, reduction workspace, or companion setup kernel is in scope.

The authoritative packed control is `/home/wd/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf`, tensor `blk.0.ffn_down_exps.weight` (`Q5_K`, `[256,2048,352]`). The independent oracle dequantizes only selected routed experts to BF16 before matmul.

## Reuse Boundary

The grouped problem identity, route scalar plan, pointer rebasing, split-route traversal, tail predicates, launcher, inspection, and benchmark protocol transfer from Q4_K only after their quant type and packed stride become strict identity facts. The arithmetic body reuses the ordinary Q5_K leaf, including low-nibble payload, high-bit payload, scale/minimum reconstruction, decoded BF16 LDS, FP32 WMMA, and BF16 RNE stores.

Q4_K dependency batching is not a Q5_K knob. Q5_K candidates instead vary packed/scalar extraction, nibble-shift hoisting, metadata loading, geometry, schedule, LDS layout, and route ownership only when each choice changes emitted Q5_K ISA.

## Qualification

Correctness covers full exact rows, non-aligned route tails, first/non-first routes, sparse and repeated expert IDs, deterministic reruns, gradient and route mutation, active/inactive expert-weight mutation, invalid experts, malformed first/final offsets, and untouched sentinels. Candidate versus packed HIP must be BF16 bit-exact; the independent reference must remain finite with NRMSE below `0.01`.

Inspection requires gfx1151, wave32, code object v5, exact 56-byte metadata, bounded VGPR/SGPR indices, zero private bytes and spills, no scratch/calls/dynamic stack, and derived static WMMA/barrier counts. Two independent generate/build/inspect roots must produce byte-identical source and HSACO.

Timing uses warmed rotating GPU events and includes output allocation in both HIP and candidate complete-call paths. Search uses five fitted medoids; retained changes receive a disjoint confirmation-bank run with 5 warmups, 25 repeats, and reversed rotating order. TFLOPS is `2 * R * 512 * 2048 / (latency_ms * 1e9)`.

## Planned Search

- Generalize the strict grouped identity and benchmark from Q4_K-only to `{Q4_K,Q5_K}` without changing Q4 source hashes.
- Establish serial M64/N64, M128/N64, and M128/N128 Q5 controls and qualify full packed semantics.
- Screen Q5 packed extraction, nibble-shift hoisting, metadata loading, SIA/PGR/PLR, LDS padding/swizzle, and split-route factors against fitted medoids.
- Re-evaluate `Mixed128_64` only where the fitted prior contains enough routes at or below 64 rows.
- Confirm per-key winners on disjoint medoids, rebuild independently, and report final HIP/GGTensile TFLOPS and speedup.

## Completion Record

This section is updated after every coherent implementation or failed experiment.

### Campaign opened

The completed grouped Q4_K route shell, ordinary Q5_K backward record, production HIP Q5_K serial/row-task history, packed Q5_K tensor availability, and current generated Q5 decoder were reviewed. No Q4 timing result is assumed to transfer. The first implementation change will make quant type and packed expert stride strict grouped identity fields while preserving the Q4 source hash and 56-byte ABI.

### Strict Q5 identity and first controls

The grouped contract now accepts only Q4_K and Q5_K, derives the quant block geometry from the strict problem type, and rejects Q3_K/Q6_K identities. The shared benchmark validates the requested GGUF tensor type and `[256,K,row_bytes]` shape instead of assuming Q4_K's 288-byte row. Q4 source generation remains unchanged; Q5 symbols and hashes are distinct, and the Q5 guard requires `bytes_per_expert=720896`.

The Q5 pilot and initial SIA5 M128/N128, SIA5 M128/N64, SIA5 M64/N64, SIA4 double-LDS, and split-route artifacts all pass the partial 625-row route matrix. Packed HIP comparisons are BF16 bit-exact, deterministic reruns are bit-exact, and the independent dequantized BF16 oracle NRMSE is `6.85e-5`. Initial resource envelopes are 216/35/10 KiB for SIA5 M128/N128, 140/35/5 KiB for M128/N64, 100/35/4 KiB for M64/N64, and 220/35/16 KiB for double-LDS; all have zero private bytes and spills.
