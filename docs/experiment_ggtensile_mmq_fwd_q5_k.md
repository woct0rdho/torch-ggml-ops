# GGTensile MMQ Forward Q5_K Results

## Scope

This record covers the dense Q5_K forward kernel keys used by the workload.

The retained kernels decode the packed representation directly into the common four-wave `128x64` LDS/WMMA body.

## Final Results

`TFLOPS = 2*M*N*K / (median_ms * 1e9)`, and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Medians are the repository benchmark's, re-measured in the current clock state. They replace the earlier recorded values, which came from a different clock state with the same artifacts.

| Family | `(M,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| --- | --- | ---: | ---: | --- | --- |
| Narrow K/V/gate/up | `(2048,512,2048)` | 26.476 | 1.0373x | `mmq_fwd_q5_k_m2048_n512_k2048_1b9a9abd77c5e8b6` | `dense_fwd_q5_k_k2048_j128_full` |
| Narrow K/V/gate/up | `(8192,512,2048)` | 27.943 | 1.0399x | `mmq_fwd_q5_k_m8192_n512_k2048_bf5eff47585aa8d6` | `dense_fwd_q5_k_k2048_j128_full` |
| Narrow K/V/gate/up | `(32768,512,2048)` | 28.136 | 1.0242x | `mmq_fwd_q5_k_m32768_n512_k2048_cb8efc54e9727b09` | `dense_fwd_q5_k_k2048_j128_full` |
| Shared down | `(2048,2048,512)` | 27.291 | 1.0723x | `mmq_fwd_q5_k_m2048_n2048_k512_d252a5127fdc4149` | `dense_fwd_q5_k_k512_j128_full` |
| Shared down | `(8192,2048,512)` | 27.229 | 1.0390x | `mmq_fwd_q5_k_m8192_n2048_k512_16caf42949a38b26` | `dense_fwd_q5_k_k512_j128_full` |
| Shared down | `(32768,2048,512)` | 27.449 | 1.0424x | `mmq_fwd_q5_k_m32768_n2048_k512_47871d72c8de59e5` | `dense_fwd_q5_k_k512_j128_full` |

GGTensile is ahead on all six entries, by `1.024x` to `1.072x` (mean `1.043x`).

## Multiply Profile

Per-workgroup counters of the narrow M32768 representative. Profiling perturbs timing, so they explain the remaining bottleneck rather than rank candidates.

| Counter per workgroup | HIP | GGTensile | Result |
| --- | ---: | ---: | --- |
| SQ instructions | `103,776` | `88,056` | GGTensile executes about 15% fewer |
| Branch instructions | `288` | `352` | Rolled four-group loops add 64 |
| Instruction-fetch waits | `2,467` | `2,817` | GGTensile has about 14% more |
| SQ busy cycles | `64,021` | `63,714` | Effectively tied |

The residual cost is Q5 high-bit insertion: 16 byte-wise merge chains remain on the critical path. A merge-free diagnostic reached about `0.98190x` of the final parent on narrow M32768, bounding the local opportunity near 1.8%. Tested alternatives either retained the same operation count, added cross-lane work, or regressed another exact key.

## Experiment Log

The log records kernel mechanisms only. A retained entry means the mechanism is part of the selected kernel or is a measured lowering used by it. A rejected entry remains closed unless a new compiler, ISA, hardware, ownership, or resource premise changes the evidence.

### Accepted

| Kernel mechanism | Evidence and disposition |
| --- | --- |
| Direct packed Q5_K decode into the retained `128x64` body | The common `239 VGPR / 38,400-byte LDS` body was retained. |
| Q5 payload-address contraction | Replaced separate shifts/adds with two `v_mad_u32_u24` address operations and immediate VMEM offsets. The combined form measured `0.99413x` its prior parent and `1.01972x` HIP on narrow M32768. The address-only form was neutral at `0.99913x`. Retained as resource-neutral instruction reduction. |
| Direct odd-plane mask/shift | Masks `0x02020202` and shifts by three directly, removing one shift per packed dword without changing arithmetic. Retained with the address contraction. |
| `qh`-first batched decode | Shifts the loaded high-bit words in place before low-nibble extraction and batches low/high preparations. Batching measured `0.99067x` its parent. Moving the four independent shifts to the front added a repeatable `0.99748x` reduction. Retained across exact keys. |
| Metadata extraction after low WMMA | Independent scale/minimum extraction is issued after the first low WMMA batch, preserving the dependency-safe metadata schedule. Narrow M2048 serialized and metadata-after-low controls were `1.04899x` and `1.01516x` HIP, while independent extraction reached `1.00708x`. |
| Four-wave LDS and WMMA ownership | The common `128x64`, four-wave layout remains the best valid resource-neutral body after unchanged `64x64`, `128x128`, `256x64`, and compact `128x32` alternatives lost. |
| Exact-key epilogues | The selected schedules are narrow M2048 `a1d8-p0` scalar, narrow M8192 `a8d1-p0` scalar, narrow M32768 `a7d3-p3` VOPD, shared-down M2048 `a1d2-p2` scalar, shared-down M8192 `a1d4-p2` VOPD, and shared-down M32768 `a1d2-p2` scalar. |
| VOPD accumulator initialization | Replaced 72 scalar accumulator clears/copies with 36 legal `v_dual_mov_b32` pairs. Retained only for narrow M32768 and shared-down M8192, where parent comparisons were repeatable and resource-neutral. |
| Persistent activation-base lifetime | The typed parent keeps the activation LDS base in `v236` and hoists invariant setup. The current-parent unhoisted control was exact but added one VALU issue and was slower. The persistent-base lowering remains selected. |

### Rejected: Decode And Data Movement

| Experiment | Evidence and disposition |
| --- | --- |
| High-plane lane sharing with `ds_bpermute_b32` | Correct after fixing the lane mask, but measured `1.00705x` the reduced-instruction parent and `1.02476x` HIP on narrow M32768. Rejected: cross-lane distribution does not repay global-load reduction. |
| DPP8 high-plane broadcast | Bit-exact, but measured `1.00756x` and `1.00714x` its parent on narrow M2048 and M32768. Rejected. |
| Alternate high-bit merge orders and `v_and_or_b32` forms | Even-first, odd-first, plane-grouped, shift-plus-`v_and_or_b32`, and in-place mask variants moved different M values in opposite directions or collapsed in a second rotation. Rejected. |
| Q5 payload and metadata transaction widths | Eight current-parent variants per M2048 target passed all correctness and resource gates. No width variant improved the parent in both rotations. Every variant remained slower than HIP. The best payload-global-8 result was `0.94854x/1.00121x` parent for narrow and `0.99980x/1.00616x` for shared-down. All four width axes are closed. |
| Padded decoded rows | `(LdsPadA,LdsPadB)=(4,0),(0,4),(4,16)` were rejected before lowering because the fixed B128 LDS emitter lacks a padding-safe cooperative address transform. No timing claim was made. |
| Compact LDS with consumer-side decode | A 30,720-byte pair-reuse body achieved four resident workgroups and removed payload LDS traffic, but ran `1.10934x` its parent on narrow M8192. Activation-read overlap was `1.10847x`. Consumer high-bit insertion adds too much serialized work immediately before WMMA. |
| Payload prefetch and larger buffers | Payload-only prefetch, larger activation buffers, and larger decoded-weight buffers did not remove the high-bit dependency chain and were neutral to slower. Rejected. |

### Rejected: Scheduling And Geometry

| Experiment | Evidence and disposition |
| --- | --- |
| Serialized metadata and alternate extraction timing | Slower than independent extraction. No stable schedule closed the remaining parent margin. Rejected. |
| Full-row, two-row, and group batching | Full-row and two-row preparation measured `1.00005x` and `1.00019x` parent. Full and two-way rolling/unrolling reached neutral to `1.00515x` slower and increased instruction-fetch pressure. Rejected. |
| Startup VOPD compositions and priority controls | Broad startup pairing and block/decode `s_setprio` controls were neutral or slower. Only the exact accumulator-initialization pairs with stable evidence were retained. |
| Changed forward geometries | Unchanged `64x64`, `128x128`, `256x64`, compact `128x32`, and related ownership variants lost without a Q5-specific resource or traffic gain. Rejected. |
| Activation-base unhoisting at M2048 | Current-parent scalar and merge reopenings were exact and mutation-sensitive. The two local screen leaders received 25-repeat confirmation: narrow `merge_odd_and_or` was `0.9906x/0.9967x` parent. Shared-down `scalar_a3d4_p1` was `0.9992x/1.0017x`. The unhoisted activation-base diagnostic was also slower and used `943` rather than `942` VALU issues. Rejected. |

### Rejected: ISA And Ownership Premises

| Experiment | Evidence and disposition |
| --- | --- |
| Fewer high-bit operations through VOPD or exact byte insertion | gfx1151 legal pairings do not combine the required two-plane shifts and masks into fewer exact operations. Tested substitutions preserved the operation count or introduced dependencies. Rejected. |
| Shift/shift and AND/AND VOPD pairings | Not legal or not useful under the gfx1151 instruction definitions and tests. Rejected by ISA constraints. |
| Direct-to-LDS and split-barrier staging | The required payload/metadata ownership and barrier ordering do not admit a correct lower-cost form under the current decoded contract. Rejected. |
| Q4/Q6/backward ownership transfers | Backward scalar extraction, padded transposed ownership, grouped J32/J64, and Q6 row compositions change the dataflow or workload shape. Existing forward geometry and high-bit-floor evidence provide no valid transfer premise. Rejected. |
| Software instruction prefetch | No useful gfx1151 mechanism was found. Prior instruction-fetch evidence supports schedule review but not a software-prefetch claim. Rejected. |

## Verification And Closure

The current-parent M2048 reopening rebuilt the previously leading scalar schedules and high-bit merge diagnostics against the typed writer. All 16 artifacts passed correctness and deterministic rebuild gates. No candidate produced stable parent improvement, so no exact key or kernel identity changed.

The remaining quantified bottleneck is the Q5 high-bit merge chain, not packed traffic, occupancy, or total SQ busy work. The accepted reductions and schedules have been exhausted under the current kernel ownership and instruction contract. Further work requires a genuinely different way to remove or hide consumer-side high-bit insertion. Repeating width, merge-order, epilogue, loop, or unchanged-geometry searches is not justified.

Artifacts for the final selected kernels and authoritative multiply confirmations are under `~/tmp/torch-ggml-ops/ggtensile-fwd-q5-k/retained-multiply-authoritative/`. The current-parent reopening artifacts are under `~/tmp/torch-ggml-ops/q5-m2048-reopen-v1/`. The data-movement screen is under `~/tmp/torch-ggml-ops/q5-data-movement-open-v1/`.
