# GGTensile Grouped MMQ Forward IQ2_S Experiment

## Scope

This record covers the routed gfx1151 non-paired IQ2_S down projection, one GEMM per routed expert:

```text
X_g[M_g,K] @ W_g[K,N] -> Y_g[M_g,N]
```

## Final Results

`TFLOPS = 2*R*N*K / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Each row is the median over the benchmark's sampled expert partitions, and both implementations are timed on the same partitions.

| Batch | Logical shape `(R,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `(16384,2048,512)` | 13.398 | 0.8895x | `grouped_mmq_fwd_iq2_s_r16384_n2048_k512_bacad5f35e7d3cb5` | `grouped_fwd_serial_iq2_s_n2048_k512_j64_j32_frag` |
| 4 | `(65536,2048,512)` | 18.385 | 0.9834x | `grouped_mmq_fwd_iq2_s_r65536_n2048_k512_ab0d8e160e3e389a` | `grouped_fwd_serial_iq2_s_n2048_k512_j64_j32_frag` |
| 16 | `(262144,2048,512)` | 19.351 | 0.9754x | `grouped_mmq_fwd_iq2_s_r262144_n2048_k512_da1baae36eda6c5f` | `grouped_fwd_serial_iq2_s_n2048_k512_j64` |

HIP is ahead on all 3 rows. GGTensile is at `0.8895x` to `0.9834x` (mean `0.9494x`).

These medians replace the earlier recorded values, which were paired against HIP bodies that have since been retuned.

## Accepted Experiments

### Dedicated IQ2_S decoder and identity

IQ2_S was given a strict grouped forward identity rather than reusing Q2_K or Q4_K arithmetic. The producer maps `wave * 16 + lane[3:0]` to one output-feature row and `lane[4]` to one K half, so all four waves decode the 64-row weight tile exactly once. The first live launches exposed three decoder-local defects: scale addresses derived from payload lanes instead of WMMA output elements, reuse of `v0` after it became an accumulator, and globally inverted signs caused by the `v_perm_b32` byte-selector operand order. After correction, a four-route 35-row bounded launch matched the installed mixed J64/J32 control bitwise for all 71,680 BF16 elements.

### Coalesced activation staging

The original J64 parent assigned each workitem one contiguous 72-byte half of an activation row, leaving adjacent lanes 144 bytes apart and the VMEM traffic poorly coalesced. The accepted staging linearly maps the 128 workitems over the 9,216-byte activation image using four `b128` bands and one `b64` band, keeping the 116-VGPR, 40-SGPR, and 30,720-byte-LDS profile. B16 weighted body time improved from `29.9013` to `27.9572 ms` and complete-call time from `31.1351` to `29.4779 ms`. B1 complete call improved to `2.7473 ms` versus `3.2639 ms` for HIP (`1.1881x`), and B4 to `8.1405 ms` versus `8.6783 ms` (`1.0661x`).

### BFE decode extraction and combined selector

`v_bfe_u32` replaces the separate shift-and pairs for QH two-bit fields, sign nibbles, and scale nibbles, removing 56 static VALU instructions per decoded weight block without changing any extracted value. The bounded route remained bitwise exact, and B16 complete-call time fell to `28.9172 ms` (`0.9755x` HIP), B1 to `2.6520 ms` (`1.2273x`), and B4 to `7.8400 ms` (`1.0961x`).

The sign-selector construction was then collapsed: the spread constant is shifted into the multiply and combined with `v_and_or_b32` against an invariant SGPR `0x03020100`, removing one instruction and one dependency edge from every signed codebook dword. B16 complete-call time reached `28.2488 ms` (`0.9911x` HIP), B1 `2.6398 ms` (`1.2345x`), and B4 `7.8012 ms` (`1.1116x`).

### Quarter-scale arithmetic and payload prefetch

Precomputing `d * 0.25` once and multiplying by `(scale + 0.5)` removed seven dependent FP operations per decoded block and stayed bitwise exact. A dependent schedule then prefetches the next group's five activation/weight payload LDS reads while correcting the current WMMA results, with weight and activation scale reads overlapping the following four WMMAs. The B16 five-medoid screen measured `27.9807 ms` complete versus `28.0333 ms` for adjacent HIP (`1.0019x`). These mechanisms complete the retained payload-prefetch identity.

## Rejected Experiments

### J128 output-column ownership

An eight-wave, 256-thread variant doubled the column tile to 128 while retaining the 64-row route tile. It built at 116 VGPRs, 40 SGPRs, 52,224 LDS bytes, 64 static WMMAs, and four barriers, and a bounded route stayed bitwise exact. It nevertheless regressed B16 to `31.3072 ms` body versus `26.5758 ms` for HIP and `32.7224 ms` complete (`0.8594x`), with every medoid between `0.8564x` and `0.8620x`. The larger one-workgroup LDS allocation could not repay the smaller grid.

### Payload-first scale schedule

A schedule issuing all five activation payload reads before weight-scale LDS reads and delaying scale consumption until after four WMMAs matched the installed HIP disassembly order but regressed B16 complete-call time to `29.5103 ms` versus `29.4779 ms` for the coalesced parent. It was removed as a near-neutral regression.

### Packed index/sign vector loads

Packed `b32` and `b128` loads for the 16-byte index and sign halves were screened. Four unaligned `b32` reads per plane regressed B16 complete-call time to `29.6746 ms`. One `b128` read per plane measured `29.4607 ms` versus `29.4779 ms` for the parent. The apparent 0.06% improvement is below run resolution and adds unpacking complexity, so the candidate was rejected and removed.

### Partial zero-bank hoisting

Moving only `v94:v99` before the row-tile decode loop while retaining the `v92:v93` repairs is timing-neutral: complete-call movement was `+0.15%`, `+0.20%`, and `+0.20%` at B1/B4/B16, body movement was `-0.01%`, `+0.20%`, and `+0.01%`, and per-medoid direction was mixed. The static move reduction is not a performance result.

### Shared routed-prologue and direct-to-LDS experiments

The shared packed-kernarg, paired cumulative-offset, and guarded high-stride/address reductions were exact but moved the Q4_K B1 transfer screen by `+0.030%`, `-0.086%`, and `-0.083%`, so no IQ2_S timing transfer was run. Direct-to-LDS cannot be emitted: the configured gfx1151 assembler rejects the buffer and LLVM spellings, and `global_load_lds_dword` is reported unsupported. A route-persistent two-block image remains unmeasured and was not pursued because it requires the same 52,224-byte one-workgroup LDS class that the rejected J128 body already showed to be occupancy-limited.

## Open Items

B1 and B4 route ownership and decode reuse should be re-screened on the current multiply-only protocol before any new persistent decoded-image or synchronization change is composed.

## Closure

The retained identity is `iq2_s_serial_full_weight_lds_64_linear_payload_prefetch()`: a J64 serial full-weight LDS body with BFE decode, combined selector, quarter-scale arithmetic, and payload prefetch.
