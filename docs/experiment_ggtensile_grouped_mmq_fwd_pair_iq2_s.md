# GGTensile Grouped MMQ Forward Pair IQ2_S Experiment

## Scope

This record covers the routed gfx1151 paired IQ2_S gate and up projections for the Qwen expert bank, computed in one workgroup dataflow.

## Final Results

`TFLOPS = 4*R*N*K / (median_ms * 1e9)` (both projections), and `Speedup vs HIP = HIP median time / GGTensile median time`, so a value above `1.0x` favors GGTensile. Each row is the median over the benchmark's sampled expert partitions, and both implementations are timed on the same partitions.

| Batch | Logical shape `(R,N,K)` | GGTensile TFLOPS | Speedup vs HIP | GGTensile kernel | HIP kernel |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `(16384,512,2048)` | 13.479 | 1.1076x | `grouped_mmq_fwd_pair_iq2_s_r16384_n512_k2048_0d821a0c4f9dafa0` | `grouped_fwd_row_task_iq2_s_n512_k2048_j64` |
| 4 | `(65536,512,2048)` | 18.956 | 1.0849x | `grouped_mmq_fwd_pair_iq2_s_r65536_n512_k2048_7bc3e427ffbd0b3a` | `grouped_fwd_row_task_iq2_s_n512_k2048_j64` |
| 16 | `(262144,512,2048)` | 19.823 | 1.0506x | `grouped_mmq_fwd_pair_iq2_s_r262144_n512_k2048_d161b5a8be7347b4` | `grouped_fwd_row_task_iq2_s_n512_k2048_j64` |

GGTensile is ahead on all 3 rows, with speedups from `1.0506x` to `1.1076x` (mean `1.0810x`).

These medians replace the earlier recorded values, which were paired against HIP bodies that have since been retuned.

## Accepted Experiments

### P1 four-wave K128 interleaving

The first candidate uses one 128-thread wave32 workgroup that owns a 64-row by 64-column tile in both projections and keeps two independent 32-VGPR FP32 sum banks. It stages each 9,216-byte activation plane once, decodes the matching K128 half from the first packed bank into a reusable half-weight LDS image, executes the first projection's WMMAs and FP32 correction, overwrites the half-weight image from the second packed bank, and executes the second projection while the activation plane remains resident. Two producer lanes share each output row so each packed group is decoded exactly once.

The final P1 artifact uses 148 VGPRs, 44 SGPRs, and 19,456 bytes of fixed LDS, with 128 static WMMAs, eight barriers, and the 80-byte cumulative-route ABI. The first timing run was invalid because the launcher passed the artifact's fixed LDS size again as dynamic LDS. After correcting the launch to zero dynamic LDS, the B16 nine-repeat screen measured `65.0525 ms` complete versus `67.4435 ms` for the installed pair (`1.0368x`), with all five medoids exact. P1 remained a qualified serial-route precursor and was superseded by P2 before confirmation.

### P2 device 64-row task ownership

P2 preserves P1's arithmetic body and resources while consuming device-built 64-row tasks through a separate 96-byte ABI. The first device launch had a scalar-lifetime defect: `row_end` shared `s29` with the activation-plane stride and was overwritten before the row loop. Moving it to a dead shape-guard register preserved the 44-SGPR plan. The corrected B16 screen measured `60.0484 ms` versus `67.8355 ms` for the installed pair (`1.1297x`), with every medoid at least `1.1255x`, so P2 advanced to reversed-order confirmation.

The final confirmation produced the recorded rows for this body: `1.2072x`, `1.1616x`, and `1.1442x` against HIP, with minimum per-medoid ratios `1.1275x`, `1.1198x`, and `1.1242x`.

## Rejected Experiments

### Closed precursor designs

Projection-indexed workgroups share only a launch. Sequential full-projection phases cannot retain the full K2048 activation workspace and reload every activation plane. Two complete decoded-weight LDS images reproduce the rejected 52,224-byte J128 resource point, and eight waves split between projections reproduce that ownership topology with its one-workgroup residency and synchronization cost. These designs were closed before implementation because they cannot share the required dataflow.

### P3 paired zero-accumulator lifetime

Initializing the dedicated read-only `v124:v131` zero bank once before the K2048 block loop removed 24 statically emitted VALU issue instructions while retaining 148 VGPRs, 44 SGPRs, 19,456 LDS bytes, 128 WMMAs, eight barriers, and zero spills. Outputs were exact, but weighted complete-call movement was `+0.04%`, `-0.02%`, and `+0.11%` at B1/B4/B16 and body movement was `-0.12%`, `+0.05%`, and `+0.17%`. The direction is mixed, so P3 is closed as timing-neutral.

### P4 and P5 epilogue probes

The shared-column setup form removed six VALU issues and the broader materialized-address form removed 25, without changing the resource class or synchronization. Both passed the exact route and mutation matrix, but complete-call ratios were directionally mixed and the materialized-address form regressed the dominant B1 body by 1.30%. Neither form has coherent complete/body direction. Interleaving exact `v_bfe_u32`/`v_add3_u32` BF16 RNE chains through proven-dead scratch was also timing-incoherent, with complete-call and body movements disagreeing. All three forms are closed without source retention.

### P6 processor-mode metadata

Applying the processor-mode spelling to the retained row-task sources produced byte-identical objects and HSACOs at R35 and all production row counts, so it creates no distinct executable kernel and no timing result. No wait or barrier candidate was admitted: the eight barriers protect projection-overwritten weight LDS and activation reuse, and no exact gfx1151 hazard proof permits removal.

## Closure

The retained identity is `iq2_s_k128_interleaved_row_tasks()`: 64-row device tasks with K128 interleaved half-weight LDS and two accumulator banks.
