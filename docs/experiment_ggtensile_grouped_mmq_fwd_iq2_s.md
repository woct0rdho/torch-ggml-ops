# GGTensile Grouped MMQ Forward IQ2_S Experiment

## Scope And Contract

This record covers the isolated gfx1151 grouped non-paired IQ2_S forward kernel for the Qwen routed down projection. For each routed expert, the operation is:

```text
X_g[M_g,512] @ W_g[512,2048] -> Y_g[M_g,2048]
```

The measured aggregate-row shapes are `R=16384`, `65536`, and `262144`. The packed expert bank is `[256,2048,164]`: each 256-value block occupies 82 bytes, each packed row contains two blocks, and each expert occupies 335,872 bytes. One IQ2_S block carries one FP16 block scale, 64 `qs` bytes, eight `qh` bytes, and eight packed scale bytes; two adjacent eight-value groups share a four-bit scale with effective factor `d * (scale + 0.5) / 4`. The full 1,024-entry codebook is embedded in the code object. Inputs and outputs are contiguous BF16, accumulation is FP32 WMMA V1, and stores use BF16 round-to-nearest-even.

The grouped research ABI is:

```text
input, packed_weight, output,
expert_indices, expert_offsets, num_experts, rows, bytes_per_expert
```

The retained body is a J64 distributed full-weight LDS kernel. The activation workspace is the fixed HIP Q8_1 `F32_D4` layout and activation quantization is outside multiply-only timing. Logical throughput is `2 * R * 2048 * 512 / (latency_ms * 1e9)`, and the speedup ratio is HIP time divided by GGTensile time.

## Final Benchmark Results

The table shows the fastest qualified kernel found for each measured aggregate-row shape.

| Matrix shape `(R,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(16384,2048,512)` | `ggsol_bacad5f35e7d3cb5` | `13.869` | `1.2728x` |
| `(65536,2048,512)` | `ggsol_ab0d8e160e3e389a` | `18.347` | `1.1310x` |
| `(262144,2048,512)` | `ggsol_da1baae36eda6c5f` | `20.306` | `0.9951x` |

All retained entries passed the fitted-prior qualification and the independent-reference and mutation checks. The retained identity is `iq2_s_serial_full_weight_lds_64_linear_payload_prefetch()`.

## Final Kernel Profiles

| Kernel hash | Geometry and ownership | Decode and schedule | VGPR / SGPR | LDS bytes | WMMAs / barriers |
| --- | --- | --- | ---: | ---: | ---: |
| `ggsol_bacad5f35e7d3cb5` | J64 serial full-weight LDS | BFE decode, combined selector, quarter-scale, payload prefetch | `116 / 40` | `30,720` | `64 / 4` |
| `ggsol_ab0d8e160e3e389a` | J64 serial full-weight LDS | BFE decode, combined selector, quarter-scale, payload prefetch | `116 / 40` | `30,720` | `64 / 4` |
| `ggsol_da1baae36eda6c5f` | J64 serial full-weight LDS | BFE decode, combined selector, quarter-scale, payload prefetch | `116 / 40` | `30,720` | `64 / 4` |

All three artifacts are gfx1151 code-object-v5 wave32 kernels with zero private storage, spills, scratch, calls, and dynamic stack.

## Accepted Kernel Experiments

### Dedicated IQ2_S decoder and identity

IQ2_S was given a strict grouped forward identity rather than reusing Q2_K or Q4_K arithmetic. The producer maps `wave * 16 + lane[3:0]` to one output-feature row and `lane[4]` to one K half, so all four waves decode the 64-row weight tile exactly once. The first live launches exposed three decoder-local defects: scale addresses derived from payload lanes instead of WMMA output elements, reuse of `v0` after it became an accumulator, and globally inverted signs caused by the `v_perm_b32` byte-selector operand order. After correction, a four-route 35-row bounded launch matched the installed mixed J64/J32 control bitwise for all 71,680 BF16 elements.

### Coalesced activation staging

The original J64 parent assigned each workitem one contiguous 72-byte half of an activation row, leaving adjacent lanes 144 bytes apart and the VMEM traffic poorly coalesced. The accepted staging linearly maps the 128 workitems over the 9,216-byte activation image using four `b128` bands and one `b64` band, keeping the 116-VGPR, 40-SGPR, and 30,720-byte-LDS profile. B16 weighted body time improved from `29.9013` to `27.9572 ms` and complete-call time from `31.1351` to `29.4779 ms`. B1 complete call improved to `2.7473 ms` versus `3.2639 ms` for HIP (`1.1881x`), and B4 to `8.1405 ms` versus `8.6783 ms` (`1.0661x`).

### BFE decode extraction and combined selector

`v_bfe_u32` replaces the separate shift-and pairs for QH two-bit fields, sign nibbles, and scale nibbles, removing 56 static VALU instructions per decoded weight block without changing any extracted value. The bounded route remained bitwise exact, and B16 complete-call time fell to `28.9172 ms` (`0.9755x` HIP), B1 to `2.6520 ms` (`1.2273x`), and B4 to `7.8400 ms` (`1.0961x`).

The sign-selector construction was then collapsed: the spread constant is shifted into the multiply and combined with `v_and_or_b32` against an invariant SGPR `0x03020100`, removing one instruction and one dependency edge from every signed codebook dword. B16 complete-call time reached `28.2488 ms` (`0.9911x` HIP), B1 `2.6398 ms` (`1.2345x`), and B4 `7.8012 ms` (`1.1116x`).

### Quarter-scale arithmetic and payload prefetch

Precomputing `d * 0.25` once and multiplying by `(scale + 0.5)` removed seven dependent FP operations per decoded block and stayed bitwise exact. A dependent schedule then prefetches the next group's five activation/weight payload LDS reads while correcting the current WMMA results, with weight and activation scale reads overlapping the following four WMMAs. The B16 five-medoid screen measured `27.9807 ms` complete versus `28.0333 ms` for adjacent HIP (`1.0019x`). These mechanisms complete the retained payload-prefetch identity.

## Rejected Kernel Experiments

### J128 output-column ownership

An eight-wave, 256-thread variant doubled the column tile to 128 while retaining the 64-row route tile. It built at 116 VGPRs, 40 SGPRs, 52,224 LDS bytes, 64 static WMMAs, and four barriers, and a bounded route stayed bitwise exact. It nevertheless regressed B16 to `31.3072 ms` body versus `26.5758 ms` for HIP and `32.7224 ms` complete (`0.8594x`), with every medoid between `0.8564x` and `0.8620x`. The larger one-workgroup LDS allocation could not repay the smaller grid.

### Payload-first scale schedule

A schedule issuing all five activation payload reads before weight-scale LDS reads and delaying scale consumption until after four WMMAs matched the installed HIP disassembly order but regressed B16 complete-call time to `29.5103 ms` versus `29.4779 ms` for the coalesced parent. It was removed as a near-neutral regression.

### Packed index/sign vector loads

Packed `b32` and `b128` loads for the 16-byte index and sign halves were screened. Four unaligned `b32` reads per plane regressed B16 complete-call time to `29.6746 ms`; one `b128` read per plane measured `29.4607 ms` versus `29.4779 ms` for the parent. The apparent 0.06% improvement is below run resolution and adds unpacking complexity, so the candidate was rejected and removed.

### Partial zero-bank hoisting

Moving only `v94:v99` before the row-tile decode loop while retaining the `v92:v93` repairs is timing-neutral: complete-call movement was `+0.15%`, `+0.20%`, and `+0.20%` at B1/B4/B16, body movement was `-0.01%`, `+0.20%`, and `+0.01%`, and per-medoid direction was mixed. The static move reduction is not a performance result.

### Shared routed-prologue and direct-to-LDS experiments

The shared packed-kernarg, paired cumulative-offset, and guarded high-stride/address reductions were exact but moved the Q4_K B1 transfer screen by `+0.030%`, `-0.086%`, and `-0.083%`, so no IQ2_S timing transfer was run. Direct-to-LDS cannot be emitted: the configured gfx1151 assembler rejects the buffer and LLVM spellings, and `global_load_lds_dword` is reported unsupported. A route-persistent two-block image remains unmeasured and was not pursued because it requires the same 52,224-byte one-workgroup LDS class that the rejected J128 body already showed to be occupancy-limited.

## Remaining Work

The current direct-kernel benchmark measured B1 at `1.1000x` GGTensile/HIP (75-repeat interval `[1.0865x, 1.1136x]`) and B4 at `1.0953x` (`[1.0874x, 1.1033x]`), below the documented `1.2728x` and `1.1310x`. B16 is effectively unchanged at `0.9966x`. The spread is short-row and route-law sensitivity rather than a correctness issue, so B1/B4 route-ownership and decode-reuse screening should be re-run on the current multiply-only protocol before any new persistent decoded-image or synchronization change is composed.

## Qualification Summary

The retained kernels passed exact packed-HIP comparison, independent BF16-reference checks, finite-output checks, deterministic reruns, route and weight mutations, inactive-expert inertness, invalid-route sentinels, and sequential, repeated, sparse, skewed, and boundary route profiles. The independent 64-column reference had a maximum absolute error of at most `0.005859375`. All three production artifacts rebuild byte-identically as gfx1151 code-object-v5 wave32 kernels with the profile above and zero private storage, spills, scratch, calls, and dynamic stack.
