# HIP MMQ Backward Q3_K Experiment

## Scope

This record covers gfx1151 HIP packed-MMQ input-gradient kernels for Q3_K weights.

Backward shapes are written `(M, in_features, out_features)`, matching the weight's `(N,K) = (out_features, in_features)`. Beyond the Qwen query and narrow projections measured first, the type carries the QSA attention query, key/value and shared-expert gate/up projections of the Qwen4-Exp checkpoint `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` (hidden size 2560) and the GatedDeltaNet `in_proj_qkv` and `in_proj_z` projections of both checkpoints in use, at the training token counts of a sequence length 2048 batch (B1/B4/B16).

GatedDeltaNet `out_proj` is deferred because wiring it needs the activation permutation. `token_embd.weight` `(248320,2560)` is an embedding gather rather than a multiply and stays on the GGUF embedding module.

## Final kernel result

| Family | `(M,K,N)` | HIP TFLOPS | HIP/torch.mm | Kernel |
| --- | ---: | ---: | ---: | --- |
| Query/query gate | `(2048,2048,8192)` | 22.649 | 1.329x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| Query/query gate | `(8192,2048,8192)` | 22.680 | 1.263x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| Query/query gate | `(32768,2048,8192)` | 23.165 | 1.255x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| Narrow key | `(2048,2048,512)` | 23.266 | 1.040x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| Narrow key | `(8192,2048,512)` | 24.272 | 1.003x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| Narrow key | `(32768,2048,512)` | 25.262 | 1.031x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| QSA query | `(2048,2560,12288)` | 22.329 | 0.946x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| QSA query | `(8192,2560,12288)` | 24.067 | 0.935x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| QSA query | `(32768,2560,12288)` | 23.431 | 0.932x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| QSA key/value | `(2048,2560,512)` | 24.308 | 1.152x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| QSA key/value | `(8192,2560,512)` | 26.035 | 0.976x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| QSA key/value | (32768,2560,512) | 30.200 | 1.125x | `dense_bwd_q3_k_mt128_nt128_ki32_full_pipea_nt4_ki64_mw4_sw16` |
| Shared-expert gate/up | (2048,2560,640) | 30.810 | 1.412x | `dense_bwd_q3_k_pipea_nt4_ki64_mw4_sw16_prefetch` |
| Shared-expert gate/up | `(8192,2560,640)` | 26.982 | 0.966x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| Shared-expert gate/up | (32768,2560,640) | 36.000 | 1.269x | `dense_bwd_q3_k_pipea_nt4_ki64_mw4_sw16_prefetch` |
| GatedDeltaNet QKV | `(2048,2560,10240)` | 22.735 | 0.962x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| GatedDeltaNet QKV | `(8192,2560,10240)` | 24.240 | 0.940x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| GatedDeltaNet QKV | `(32768,2560,10240)` | 23.644 | 0.929x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| GatedDeltaNet Z | `(2048,2560,6144)` | 23.646 | 1.006x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| GatedDeltaNet Z | `(8192,2560,6144)` | 24.755 | 0.959x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| GatedDeltaNet Z | `(32768,2560,6144)` | 23.878 | 0.950x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| GatedDeltaNet Z APEX-I-Mini | `(2048,2048,4096)` | 26.553 | 1.546x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| GatedDeltaNet Z APEX-I-Mini | `(8192,2048,4096)` | 22.360 | 1.218x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |
| GatedDeltaNet Z APEX-I-Mini | `(32768,2048,4096)` | 23.250 | 1.255x | `dense_bwd_q3_k_mt128_nt128_ki32_full_narrow` |

The values use the current packed/BF16 matrix and report the complete packed-gradient path represented there. The `Kernel` column names the deployed body for each exact key. It is the fastest built body whose output is bitwise equal to the reference body in the per-key candidate campaign.

## Kernel implementation

The build carries two tuned `mt128_nt128_ki32_full` variants: `..._full_narrow` (8-value LDS row padding, vector local loads) and `..._full_wide` (swizzle chunk 8, no LDS padding). Both were timed on both shapes and then on the eighteen Qwen4-Exp and GatedDeltaNet points. `full_narrow` won every one of them by `1.027-1.221x` and is the deployed body for all twenty-four keys, so the new shapes needed no new body. Exact shape and row-count specialization removes runtime bounds and address state from the common production shapes. Generic bounds-safe bodies remain available for unmatched shapes and are not competitive on these keys.

## Optimization log

### First tiled redesign

The original body decoded one packed value at a time into a 16x16 tile and sustained only about `0.05-0.13x` BF16 throughput. The retained redesign added four wave32 waves, LDS-staged decoded weights, multiple WMMA accumulator tiles, cooperative pair/quad/width-16 decode, and measured row/type-specific ownership.

Representative first redesign results included query Q3_K M32768 moving from `1,108.314 ms` to `162.884 ms`. The same four-wave structure became the Q3_K foundation for narrow and query shapes.

### Extraction, prefetch, and LDS layout

Wide Q3_K packed extraction, two-row prefetch, and the eight-BF16 XOR LDS layout were retained for the query body. Removing those controls regressed by `1.06-11.78%` depending on the measured point. Narrow Q3_K does not prefetch because its 110-byte block layout made the path slower.

K-loop unrolling, activation-half double buffering, generic padding, decoded-weight LDS caching, and broad local-load rules were rejected. The accepted Q3 path keeps packed state bounded to the active decode phase. Keeping the next iteration's packed fragments live across WMMA extended register lifetimes without a timing benefit.

### Qwen4-Exp and GatedDeltaNet shapes

The new shapes span contraction lengths from `512` to `12288` and output widths from `2048` to `2560`, all of which are multiples of the `ki32` step and the 128-wide tiles, so the deployed body serves them without a new specialization. Its rate over the eighteen new points is `22.3-27.5 TFLOPS`, inside the band of the six rows measured first (`22.6-25.3`), and the `full_narrow` candidate beats `full_wide` on every one of them.

The rate is not uniform in the contraction length: the short-contraction key, value and shared-expert shapes run at `24.3-27.5 TFLOPS` while the long-contraction query and QKV shapes run at `22.3-24.2`. The BF16 `torch.mm` baseline does not pay the packed decode, so the ratio against it crosses one inside this set: the narrow Qwen4 shapes are above it (`1.14-1.15x` at `M=2048`) and the longest contractions are below it (`0.93x`). That is the representation cost this record already names, and bringing it down needs a cheaper decode or a denser weight layout rather than another tiling sweep.

### Packaged-kernel controls

The source-built HSACO conversion produced a `+0.56%` initial geometric movement and a `+1.12%` embedded/bundle bracket movement, while embedded controls themselves drifted by `+1.04%`. Q3 query showed `2.9-7.0%` placement-sensitive movement without a device semantic change. Warm standalone modules, normalized ISA, and sequential controls are required before treating a timing change as a kernel result.

### Swizzle-only twin of the padded tile

The retained `_full_narrow` body carries `lds_padding = 8` and no LDS swizzle. Swizzle-only twins (8 KiB rather than 10 KiB) were screened on two families at `M=2048` and `32768`: padding wins, `1.08-1.29x` at `sw8` and `1.18-1.35x` at `sw16`, so this body keeps its padded tile. Evidence: `~/tmp/torch-ggml-ops/qwen4_fwd/bwd_swz_q3q4.txt`.

### Pipelined tile

The pipelined stage order was screened here on the same pattern as Q5_K and Q6_K: on the narrow-result shared-expert gate/up rows the 64-value-stage, four-row-tile body is `1.37x` and `1.20x` ahead of the deployed body at `M=2048` and `32768`, and on the QSA key/value rows it is `1.13x` ahead at `M=32768` while losing `0.77x` at `M=2048`, where its 256-row blocks leave too few workgroups. Those three rows take the pipelined body. The pipelined twins of the eight-column tile lose `1.06-1.13x` everywhere they were measured, so the rest of the family keeps the deployed padded tile.

### Prefetched decode on the pipelined tile

The pipelined projection rows used the width-16 group decode. The prefetched variant, which reads the payload planes as `uint4` before decoding, is `1.04x` and `1.07x` ahead of it on the shared-expert gate/up rows, so those two rows deploy it.

## Resources

The retained Q3_K query body uses `216 VGPR / 17 SGPR / 10 KiB LDS`, static and spill-free.

## Evidence

Current measurement evidence for the table above:

```text
~/tmp/torch-ggml-ops/qwen4_fwd/bwd_q3k_v1.txt
~/tmp/torch-ggml-ops/hip_vs_baseline/pass11_ordbwd_qwen.json
~/tmp/torch-ggml-ops/hip_selection/           (per-key candidate campaign)
tools/configs/hip_deployment.json             (deployed body per key)
```

The original campaign evidence is:

```text
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_narrow_q5_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb0_folded_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_selected_before_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_scalar_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_no_prefetch_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_no_swizzle_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_q3_selected_after_25.json
```

The source log's shared-body and packaging controls are:

```text
~/tmp/torch-ggml-ops/mmq_bwd_baseline_primary_sequential.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb0_folded_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_qb1_final_25.json
~/tmp/torch-ggml-ops/mmq_bwd_qwen_post_ds4_p1_control_9.json
~/tmp/torch-ggml-ops/mmq_bwd_pre_bundle.json
~/tmp/torch-ggml-ops/mmq_bwd_post_bundle.json
~/tmp/torch-ggml-ops/mmq_bwd_embedded_pre_control_25.json
~/tmp/torch-ggml-ops/mmq_bwd_bundle_control_25.json
~/tmp/torch-ggml-ops/mmq_bwd_embedded_post_control_25.json
```

The current Q3_K kernel record closes global J64, I128, alternate workgroup sizes, split-K, persistent workgroups, speculative prefetch, and broad swizzle changes for the existing packed representation.
