# GGTensile Grouped MMQ Forward Pair IQ2_S Experiment

## Scope And Contract

This record covers the isolated gfx1151 paired IQ2_S forward kernel for the Qwen routed gate and up projections. For each routed expert the kernel computes both projections in one workgroup dataflow:

```text
X_g[M_g,2048] @ W_gate_g[2048,512] -> Y_gate_g[M_g,512]
X_g[M_g,2048] @ W_up_g[2048,512]   -> Y_up_g[M_g,512]
```

The measured aggregate-row shapes are `R=16384`, `65536`, and `262144`. The two packed IQ2_S banks have shape `[256,512,656]`, one fixed Q8_1 `F32_D4` activation workspace is shared, and two independent BF16 destinations are produced. A launch that assigns one workgroup per projection, two adjacent single-projection launches, or sequential full-projection phases that reload activation data is not a paired arithmetic kernel.

The paired research ABI carries two packed weights, one activation workspace, two destinations, the device-resident route description, and the shape values. Activation quantization is outside multiply-only timing. Logical paired throughput is `4 * R * 512 * 2048 / (latency_ms * 1e9)`, and the speedup ratio is HIP time divided by GGTensile time.

## Final Benchmark Results

The table shows the fastest qualified kernel found for each measured aggregate-row shape.

| Matrix shape `(R,N,K)` | Kernel hash | GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: |
| `(16384,512,2048)` | `ggpair_a04b4d0379080838` | `13.786` | `1.2072x` |
| `(65536,512,2048)` | `ggpair_52d8b2d6a398c27a` | `18.511` | `1.1616x` |
| `(262144,512,2048)` | `ggpair_59e04f7993fc9621` | `20.347` | `1.1442x` |

The minimum per-medoid HIP/candidate ratios were `1.1275x`, `1.1198x`, and `1.1242x`. The retained identity is `iq2_s_k128_interleaved_row_tasks()`.

## Final Kernel Profiles

| Kernel hash | Geometry and ownership | Decode and schedule | VGPR / SGPR | LDS bytes | WMMAs / barriers |
| --- | --- | --- | ---: | ---: | ---: |
| `ggpair_a04b4d0379080838` | 64-row task, 64-column, two accumulator banks | K128 interleaved half-weight LDS, device row tasks | `148 / 44` | `19,456` | `128 / 8` |
| `ggpair_52d8b2d6a398c27a` | 64-row task, 64-column, two accumulator banks | K128 interleaved half-weight LDS, device row tasks | `148 / 44` | `19,456` | `128 / 8` |
| `ggpair_59e04f7993fc9621` | 64-row task, 64-column, two accumulator banks | K128 interleaved half-weight LDS, device row tasks | `148 / 44` | `19,456` | `128 / 8` |

All three artifacts are gfx1151 code-object-v5 wave32 kernels with zero private storage, spills, scratch, calls, and dynamic stack. The row-task ABI is 96 bytes.

## Accepted Kernel Experiments

### P1 four-wave K128 interleaving

The first candidate uses one 128-thread wave32 workgroup that owns a 64-row by 64-column tile in both projections and keeps two independent 32-VGPR FP32 sum banks. It stages each 9,216-byte activation plane once, decodes the matching K128 half from the first packed bank into a reusable half-weight LDS image, executes the first projection's WMMAs and FP32 correction, overwrites the half-weight image from the second packed bank, and executes the second projection while the activation plane remains resident. Two producer lanes share each output row so each packed group is decoded exactly once.

The final P1 artifact uses 148 VGPRs, 44 SGPRs, and 19,456 bytes of fixed LDS, with 128 static WMMAs, eight barriers, and the 80-byte cumulative-route ABI. The first timing run was invalid because the launcher passed the artifact's fixed LDS size again as dynamic LDS; after correcting the launch to zero dynamic LDS, the B16 nine-repeat screen measured `65.0525 ms` complete versus `67.4435 ms` for the installed pair (`1.0368x`), with all five medoids exact. P1 remained a qualified serial-route precursor and was superseded by P2 before confirmation.

### P2 device 64-row task ownership

P2 preserves P1's arithmetic body and resources while consuming device-built 64-row tasks through a separate 96-byte ABI. The first device launch had a scalar-lifetime defect: `row_end` shared `s29` with the activation-plane stride and was overwritten before the row loop; moving it to a dead shape-guard register preserved the 44-SGPR plan. The corrected B16 screen measured `60.0484 ms` versus `67.8355 ms` for the installed pair (`1.1297x`), with every medoid at least `1.1255x`, so P2 advanced to reversed-order confirmation.

The final confirmation produced the benchmark table above: `1.2072x`, `1.1616x`, and `1.1442x` against HIP, with minimum per-medoid ratios `1.1275x`, `1.1198x`, and `1.1242x`. Independent dequantized 64-column references for both projections measured normalized RMSE from `0.00592` through `0.00648` with maximum absolute error at most `0.015625`. Active first- and second-bank mutations changed only their own destination, an inactive expert was inert, and activation mutation changed both destinations.

## Rejected Kernel Experiments

### Closed precursor designs

Projection-indexed workgroups share only a launch. Sequential full-projection phases cannot retain the full K2048 activation workspace and reload every activation plane. Two complete decoded-weight LDS images reproduce the rejected 52,224-byte J128 resource point, and eight waves split between projections reproduce that ownership topology with its one-workgroup residency and synchronization cost. These designs were closed before implementation because they cannot share the required dataflow.

### P3 paired zero-accumulator lifetime

Initializing the dedicated read-only `v124:v131` zero bank once before the K2048 block loop removed 24 statically emitted VALU issue instructions while retaining 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, and zero spills. Outputs were exact, but weighted complete-call movement was `+0.04%`, `-0.02%`, and `+0.11%` at B1/B4/B16 and body movement was `-0.12%`, `+0.05%`, and `+0.17%`. The direction is mixed, so P3 is closed as timing-neutral.

### P4 and P5 epilogue probes

The shared-column setup form removed six VALU issues and the broader materialized-address form removed 25, without changing the resource class or synchronization. Both passed the exact route and mutation matrix, but complete-call ratios were directionally mixed and the materialized-address form regressed the dominant B1 body by 1.30%; neither form has coherent complete/body direction. Interleaving exact `v_bfe_u32`/`v_add3_u32` BF16 RNE chains through proven-dead scratch was also timing-incoherent, with complete-call and body movements disagreeing. All three forms are closed without source retention.

### P6 processor-mode metadata

Applying the processor-mode spelling to the retained row-task sources produced byte-identical objects and HSACOs at R35 and all production row counts, so it creates no distinct executable kernel and no timing result. No wait or barrier candidate was admitted: the eight barriers protect projection-overwritten weight LDS and activation reuse, and no exact gfx1151 hazard proof permits removal.

## Qualification Summary

The final paired kernels passed exact comparison against adjacent installed single-projection controls and the installed pair on bounded and production routes, both-projection and activation mutations, inactive-bank inertness, malformed-route sentinels, deterministic reruns, and independent dequantized references. Both projections reuse the one authoritative `[16,R,144]` activation workspace and the device task count is never read on the host. All production artifacts rebuild byte-identically as gfx1151 code-object-v5 wave32 kernels with the profile above and zero private storage, spills, scratch, calls, and dynamic stack.
