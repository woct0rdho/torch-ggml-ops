# Grouped MMQ forward optimization

## Scope

This document covers routed grouped MMQ forward on gfx1151.

The production operators are:

- `grouped_mmq_pair` for gate and up;
- `grouped_mmq` for down.

Dense MMQ forward and backward are documented separately in `docs/mmq_fwd_optimization.md` and `docs/mmq_bwd_optimization.md`.

The grouped baseline is now measured. The next implementation pass should first remove the grouped kernel's register spills, then revisit grouped tile geometry and scheduling.

## Production contract

The public activation and output dtype is BF16.

Activations are quantized internally to Q8_1. Packed GGUF weights remain the authoritative representation.

`expert_indices` and `expert_offsets` remain device-resident. The optimized path must not inspect group sizes through `.item()`, a device-to-host copy, a CPU descriptor, or an implicit synchronization.

The metadata ABI is:

- `expert_indices`: contiguous CUDA `torch.int64`, shape `[G]`;
- `expert_offsets`: contiguous CUDA `torch.int32`, shape `[G]`;
- `expert_offsets[-1] = R`;
- `G <= 256`.

The sequence length is 2,048 and top-k is 8. The routed row count is therefore:

| Physical batch | Routed rows |
|---:|---:|
| 1 | 16,384 |
| 4 | 65,536 |
| 16 | 262,144 |

Batch 1 commonly has about 150-256 active experts per layer. The observed mean was about 198.5 active experts.

One representative batch-1 layer had 192 active experts and group sizes around 60-106 rows. Batch 4 and batch 16 usually activate all 256 experts, although the sizes remain skewed.

## Production checkpoint matrix

The benchmark uses real tensors from `Qwen3.6-35B-A3B-APEX-I-Mini.gguf`.

| Case | Logical expert shape | GGUF type | Layers | Operator |
|---|---:|---|---:|---|
| Gate/up outer | `512 x 2048` | Q3_K | 20 | `grouped_mmq_pair` |
| Gate/up middle | `512 x 2048` | IQ2_S | 20 | `grouped_mmq_pair` |
| Down outer edge | `2048 x 512` | Q5_K | 2 | `grouped_mmq` |
| Down outer main | `2048 x 512` | Q4_K | 18 | `grouped_mmq` |
| Down middle | `2048 x 512` | IQ2_S | 20 | `grouped_mmq` |

Gate and up use one shared Q8_1 activation workspace. The two packed projections still execute as two grouped multiplication launches.

## Benchmark infrastructure

`bench/benchmark_grouped_mmq_fwd.py` follows the dense benchmark conventions.

It records:

- complete public packed-operator latency;
- logical throughput;
- incremental peak allocation and reservation growth;
- AITER configuration;
- routing metadata and distribution statistics;
- checkpoint-weighted forward and optimizer-step estimates;
- grouped-versus-dense-MMQ exactness;
- grouped-versus-dequantized-BF16 error.

The packed timing includes Q8_1 quantization and grouped multiplication.

The BF16 performance reference is AITER Triton `gmm`, using `_gmm_config` from `~/test_no_unsloth/fast_moe_lora.py`. It is not `torch.matmul`.

AITER receives independently dequantized BF16 versions of the same logical GGUF experts. Dequantization and active-weight selection are setup costs and are not inside the timed GMM call.

The AITER reference uses the same production transposed weight metadata as `fast_moe_lora.py`. `work_stealing` remains disabled, matching the production call.

The full baseline command was:

```bash
python bench/benchmark_grouped_mmq_fwd.py \
  --warmup 2 \
  --repeats 5 \
  --correctness-rows 128 \
  --output /tmp/grouped_mmq_fwd_baseline_full.json
```

The full artifact is:

```text
/tmp/grouped_mmq_fwd_baseline_full.json
```

GPU benchmarks and profiler runs were sequential.

## Routing distributions

The benchmark covers four deterministic distributions.

`uniform` activates all 256 experts with equal group sizes.

`skewed` activates all 256 experts with non-multiple group sizes centered around the production mean.

`sparse` uses sparse active-expert IDs. It uses 192, 224, and 240 active experts for batches 1, 4, and 16.

`boundary` includes sizes 1, 15, 16, 17, 63, 64, 65, 127, 128, and 129 before filling the remaining groups. It exercises edge handling without making it the only performance distribution.

Representative summaries are:

| Batch | Distribution | Active experts | Minimum | Maximum | Mean |
|---:|---|---:|---:|---:|---:|
| 1 | uniform | 256 | 64 | 64 | 64.0 |
| 1 | skewed | 256 | 42 | 86 | 64.0 |
| 1 | sparse | 192 | 64 | 106 | 85.3 |
| 1 | boundary | 256 | 1 | 129 | 64.0 |
| 4 | uniform | 256 | 256 | 256 | 256.0 |
| 4 | skewed | 256 | 192 | 320 | 256.0 |
| 4 | sparse | 224 | 221 | 367 | 292.6 |
| 16 | uniform | 256 | 1,024 | 1,024 | 1,024.0 |
| 16 | skewed | 256 | 768 | 1,280 | 1,024.0 |
| 16 | sparse | 240 | 821 | 1,367 | 1,092.3 |

## Baseline results

The speedup column is `AITER time / packed MMQ time`. Values above 1.0 mean the packed path is faster.

### Uniform routing

| Case | Batch | Packed ms | AITER ms | Packed logical TFLOP/s | Speedup |
|---|---:|---:|---:|---:|---:|
| Gate/up Q3_K | 1 | 11.185 | 13.245 | 6.14 | 1.18x |
| Gate/up Q3_K | 4 | 32.572 | 33.679 | 8.44 | 1.03x |
| Gate/up Q3_K | 16 | 130.408 | 83.293 | 8.43 | 0.64x |
| Gate/up IQ2_S | 1 | 11.453 | 13.212 | 6.00 | 1.15x |
| Gate/up IQ2_S | 4 | 33.408 | 35.066 | 8.23 | 1.05x |
| Gate/up IQ2_S | 16 | 133.528 | 87.740 | 8.23 | 0.66x |
| Down IQ2_S | 1 | 6.326 | 3.578 | 5.43 | 0.57x |
| Down IQ2_S | 4 | 18.226 | 7.707 | 7.54 | 0.42x |
| Down IQ2_S | 16 | 69.996 | 44.702 | 7.85 | 0.64x |
| Down Q4_K | 1 | 6.874 | 3.524 | 5.00 | 0.51x |
| Down Q4_K | 4 | 21.977 | 7.730 | 6.25 | 0.35x |
| Down Q4_K | 16 | 83.663 | 44.266 | 6.57 | 0.53x |
| Down Q5_K | 1 | 6.953 | 3.575 | 4.94 | 0.51x |
| Down Q5_K | 4 | 21.990 | 7.777 | 6.25 | 0.35x |
| Down Q5_K | 16 | 84.674 | 43.794 | 6.49 | 0.52x |

### Sparse observed-style routing

| Case | Batch | Packed ms | AITER ms | Packed logical TFLOP/s | Speedup |
|---|---:|---:|---:|---:|---:|
| Gate/up Q3_K | 1 | 9.678 | 14.412 | 7.10 | 1.49x |
| Gate/up Q3_K | 4 | 35.091 | 36.875 | 7.83 | 1.05x |
| Gate/up Q3_K | 16 | 133.880 | 100.466 | 8.21 | 0.75x |
| Gate/up IQ2_S | 1 | 9.953 | 14.225 | 6.90 | 1.43x |
| Gate/up IQ2_S | 4 | 35.966 | 36.829 | 7.64 | 1.02x |
| Gate/up IQ2_S | 16 | 136.851 | 96.521 | 8.03 | 0.71x |
| Down IQ2_S | 1 | 5.544 | 2.839 | 6.20 | 0.51x |
| Down IQ2_S | 4 | 19.089 | 8.909 | 7.20 | 0.47x |
| Down IQ2_S | 16 | 71.064 | 44.474 | 7.74 | 0.63x |
| Down Q4_K | 1 | 6.342 | 2.821 | 5.42 | 0.44x |
| Down Q4_K | 4 | 22.196 | 8.817 | 6.19 | 0.40x |
| Down Q4_K | 16 | 84.262 | 44.430 | 6.52 | 0.53x |
| Down Q5_K | 1 | 6.394 | 2.842 | 5.37 | 0.44x |
| Down Q5_K | 4 | 22.310 | 8.805 | 6.16 | 0.39x |
| Down Q5_K | 16 | 84.894 | 43.764 | 6.48 | 0.52x |

The gate/up pair is already strong for batch 1. It remains approximately tied at batch 4 and loses at batch 16.

The down projection is the dominant deficit for every batch. Q4_K and Q5_K are especially weak at batch 4.

### Routing sensitivity

Across all four distributions, the average per-case speedups were:

| Case | Batch 1 | Batch 4 | Batch 16 |
|---|---:|---:|---:|
| Gate/up Q3_K | 1.35x | 1.04x | 0.71x |
| Gate/up IQ2_S | 1.31x | 1.03x | 0.69x |
| Down IQ2_S | 0.54x | 0.45x | 0.62x |
| Down Q4_K | 0.49x | 0.39x | 0.52x |
| Down Q5_K | 0.49x | 0.39x | 0.51x |

Packed gate/up benefits from sparse batch-1 routing because inactive experts do not launch grouped workgroups. AITER's fixed persistent grid still scans the active group list.

The current packed kernel is mildly sensitive to skew at batch 4. Each `(expert, output tile)` workgroup serially processes every 128-row chunk for its expert, so large groups create longer-lived workgroups.

### Checkpoint-weighted grouped base-projection estimate

The estimate applies the checkpoint layer counts and two executions per optimizer step under activation checkpointing. It covers grouped base projections only, not routing, activation functions, LoRA, or other model work.

| Batch | Distribution | Packed seconds | AITER seconds | Packed speedup |
|---:|---|---:|---:|---:|
| 1 | uniform | 1.434 | 1.343 | 0.94x |
| 1 | skewed | 1.430 | 1.500 | 1.05x |
| 1 | sparse | 1.261 | 1.372 | 1.09x |
| 1 | boundary | 1.439 | 1.501 | 1.04x |
| 4 | uniform | 4.247 | 3.367 | 0.79x |
| 4 | skewed | 4.546 | 3.754 | 0.83x |
| 4 | sparse | 4.494 | 3.657 | 0.81x |
| 4 | boundary | 4.626 | 3.643 | 0.79x |
| 16 | uniform | 16.708 | 10.398 | 0.62x |
| 16 | skewed | 17.022 | 11.405 | 0.67x |
| 16 | sparse | 17.045 | 11.433 | 0.67x |
| 16 | boundary | 17.015 | 10.861 | 0.64x |

Batch-1 observed-style routing is already favorable overall. Batch 4 and batch 16 require substantial kernel improvement.

AITER is the production BF16 reference. It is not a performance ceiling for a packed kernel with much smaller authoritative weights.

## Correctness

Every one of the 60 measured production points matched the concatenation of per-group dense MMQ outputs exactly in BF16.

This checks the grouped scheduler against the same Q8_1 activation semantics and the same packed GGUF decode path.

Against independently dequantized BF16 weights and BF16 AITER GMM, normalized RMSE ranges were:

| Type | Minimum NRMSE | Maximum NRMSE |
|---|---:|---:|
| Q3_K | 0.00599 | 0.00613 |
| IQ2_S | 0.00600 | 0.00611 |
| Q4_K | 0.01120 | 0.01381 |
| Q5_K | 0.01300 | 0.01650 |

These errors include the intended internal activation quantization.

## Incremental memory

The pair path uses one Q8_1 workspace for both outputs.

| Operator shape | Batch | Packed peak | AITER peak | Q8_1 workspace | Output bytes |
|---|---:|---:|---:|---:|---:|
| Pair `R x 2048 -> 2 x R x 512` | 1 | 68 MiB | 32 MiB | 36 MiB | 32 MiB |
| Pair `R x 2048 -> 2 x R x 512` | 4 | 272 MiB | 128 MiB | 144 MiB | 128 MiB |
| Pair `R x 2048 -> 2 x R x 512` | 16 | 1,088 MiB | 512 MiB | 576 MiB | 512 MiB |
| Down `R x 512 -> R x 2048` | 1 | 73 MiB | 64 MiB | 9 MiB | 64 MiB |
| Down `R x 512 -> R x 2048` | 4 | 292 MiB | 256 MiB | 36 MiB | 256 MiB |
| Down `R x 512 -> R x 2048` | 16 | 1,168 MiB | 1,024 MiB | 144 MiB | 1,024 MiB |

The measurements exclude resident packed or BF16 reference weights.

The pair workspace saving is material: two independent packed calls would need two quantization launches and two workspace lifetimes.

## Current packed kernel structure

The current grouped kernel uses:

- a `64 x 128` output-by-routed-row tile;
- four wave32 waves and 128 threads;
- one workgroup for each `(64-output tile, active expert)` pair;
- a serial loop over 128-row chunks within that expert;
- LDS staging for packed decoded weights and Q8_1 activations;
- int8 WMMA with FP32 accumulation;
- masked activation staging and BF16 stores for partial groups.

The dynamic LDS allocation is 40,448 bytes for Q3_K/IQ2_S and 38,400 bytes for Q4_K/Q5_K.

For gate/up, the launch has 8 output tiles per active expert. For down, it has 32 output tiles per active expert.

The paired operator quantizes once, then launches this grouped kernel twice. It does not maintain two accumulator sets in one kernel.

## Packed-kernel profiling

Kernel-trace artifacts are:

```text
/tmp/rocprof_grouped_gate_b1
/tmp/rocprof_grouped_gate_b16
/tmp/rocprof_grouped_down_q4_b4
```

### Gate/up Q3_K, batch 1, uniform

The profiled pair decomposed into:

- Q8_1 quantization: 0.829 ms;
- first grouped projection: 5.408 ms;
- second grouped projection: 5.359 ms.

The grouped projections account for about 93% of the packed operator time.

### Gate/up Q3_K, batch 16, uniform

The profiled pair decomposed into:

- Q8_1 quantization: 7.894 ms;
- two grouped projections: 122.445 ms total.

The grouped projections account for about 94% of the packed operator time.

### Down Q4_K, batch 4, uniform

The profiled single projection decomposed into:

- Q8_1 quantization: 0.916 ms;
- grouped projection: 21.964 ms.

The multiplication kernel is the first-order bottleneck. More quantizer work is not justified before fixing it.

## Packed code-object resources

The final extension code object was extracted from `.hip_fatbin` and inspected with `clang-offload-bundler`, `llvm-readobj`, `llvm-nm`, and `llvm-objdump`.

| Grouped kernel | VGPRs | SGPRs | Private bytes/thread | VGPR spills |
|---|---:|---:|---:|---:|
| Q3_K | 256 | 74 | 124 | 30 |
| Q4_K | 256 | 76 | 512 | 127 |
| Q5_K | 256 | 74 | 520 | 129 |
| Q6_K | 256 | 74 | 272 | 67 |
| IQ2_S | 255 | 78 | 132 | 32 |

The corresponding dense J=128 kernels have zero private segment and zero spills:

| Dense kernel | VGPRs | SGPRs | Private bytes/thread | VGPR spills |
|---|---:|---:|---:|---:|
| Q3_K | 216 | 29 | 0 | 0 |
| Q4_K | 254 | 30 | 0 | 0 |
| Q5_K | 230 | 29 | 0 | 0 |
| IQ2_S | 208 | 34 | 0 | 0 |

The packed arithmetic body is not inherently forced to spill. The grouped expert metadata and dynamic row-chunk loop push the inherited dense body over the register limit.

Static disassembly reinforces this conclusion. The grouped Q4_K function has 121 scratch-load and 76 scratch-store instruction sites, compared with 7 and 6 in the profiled AITER down kernel.

Q4_K and Q5_K spill roughly half a kilobyte per thread. This is the clearest explanation for the severe down-projection gap.

## AITER source, lowering, and profile

The inspected AITER source is:

```text
~/venv_torch/lib/python3.14/site-packages/aiter/ops/triton/gmm.py
~/venv_torch/lib/python3.14/site-packages/aiter/ops/triton/_triton_kernels/gmm.py
```

AITER uses a 256-program persistent grid.

Each program starts from its program ID and advances through logical GMM tiles by `GRID_DIM`. It walks the device-resident group-size array and never requires a host-side group descriptor.

The tile mapping uses XCD remapping. Edge tiles wrap input row and output-column load offsets with modulo arithmetic, then mask only the final store.

The K loop uses direct BF16 loads and `tl.dot`. Generated gfx1151 ISA contains `v_wmma_f32_16x16x16_bf16` instructions.

The production heuristic selects:

| Shape | M tile | N tile | K tile | Threads | Persistent programs |
|---|---:|---:|---:|---:|---:|
| Gate/up `K=2048, N=512` | 64 | 128 | 64 | 256 | 256 |
| Down `K=512, N=2048` | 128 | 128 | 64 | 256 | 256 |

The generated gate/up kernel uses 176 VGPRs, 58 SGPRs, no private segment, and no fixed LDS.

The generated down kernel uses 255 VGPRs, 57 SGPRs, and 48 private bytes per thread. It is near the register limit but its spill footprint is much smaller than packed Q4_K/Q5_K.

The gate/up AITER profile launched 256 workgroups of 256 threads. One profiled projection took 7.849 ms under kernel tracing.

The down AITER profile used the same persistent grid and took 8.156 ms under kernel tracing for the batch-4 uniform case.

Profiler time is perturbed and is used only for decomposition and resource comparison. CUDA-event medians remain the latency source of record.

Counter runs on down Q4_K batch 4 reported:

| Kernel | Occupancy | L2 hit rate |
|---|---:|---:|
| Packed grouped Q4_K | 18.44% | 69.73% |
| BF16 AITER GMM | 23.20% | 71.68% |

Packed `ALUStalledByLDS` was only 0.0154%. LDS bank stalls are therefore not the primary problem.

The similar L2 hit rates also argue against treating cache hit rate as the first optimization target. Register spills, tile shape, and the amount of scheduled work are stronger explanations.

`MemUnitBusy` was unavailable through dispatch-windowed counter collection on this gfx1151 profiler configuration and is not treated as a result.

## Lessons from dense MMQ forward and backward

The dense work provides several directly applicable rules.

First, inspect the final code object after every structural change. Source-level register intuition was repeatedly insufficient, and the grouped baseline is already a concrete example.

Second, use compile-time typed variants and measured dispatch. Q3_K, Q4_K, Q5_K, and IQ2_S have different decode and resource behavior.

Third, optimize full operator latency. The grouped quantizer is only about 4-8% of the profiled operator, so multiplication changes dominate the current opportunity.

Fourth, larger cooperative tiles can win when they amortize metadata, packed weight decode, and activation staging without spilling. The dense backward pass benefited from multiple WMMA tiles per wave and shape-specific layouts.

Fifth, LDS layout changes matter only after the dominant resource problem is under control. The dense backward pass gained from vector LDS loads, padding, and XOR swizzles, but the grouped Q4_K kernel currently has 127 VGPR spills.

Sixth, packed extraction and multi-value decode should be type-specific. Dense Q3_K, Q5_K, and Q6_K did not share one universal winner.

Finally, gfx1151 int8 WMMA has no decisive raw throughput advantage over BF16 WMMA. Packed grouped MMQ must win through smaller authoritative weights, decode quality, traffic reduction, reuse, and lower scheduling overhead.

## Bottleneck interpretation

### Register spilling is the first blocker

Q4_K and Q5_K reach 256 VGPRs and spill 127-129 VGPRs per thread.

The current dynamic grouped loop adds enough live state to turn a spill-free dense arithmetic body into a scratch-heavy grouped kernel.

Removing this spill footprint has higher priority than cache tuning, extra prefetching, or quantizer changes.

### The current output tile is narrower than AITER

Packed MMQ produces 64 output columns per workgroup. AITER uses 128.

For down, this means 32 packed output tiles per expert instead of 16 AITER output tiles. Packed decode and Q8 staging must compensate for that extra scheduling and tile overhead, but currently do not.

A future project-owned `I=128` tile is therefore a meaningful higher-ceiling direction. It must be paired with 256 threads or another accumulator layout that does not increase per-thread register pressure.

### Serial expert-chunk loops limit balancing

One packed workgroup owns one expert and output tile, then serially processes all 128-row chunks for that expert.

This is simple and avoids descriptor construction, but it couples workgroup lifetime to group size. A row-tile scheduler can expose more parallel work and compile a dense-like single-tile body.

### Quantization is not the dominant deficit

The quantizer contributes about 0.8-0.9 ms at the smaller profiled points and 7.9 ms for the batch-16 gate/up pair.

It is already shared by gate/up. Removing all quantization cost would not close the down or batch-16 gate/up gaps.

### Pair fusion is not yet justified

A fused two-weight compute kernel could load each Q8 activation tile once, but it would also need two accumulator sets or sequential projection phases.

Two live accumulator sets would worsen the current register problem. Pair compute fusion should be reconsidered only after a spill-free single-projection kernel exists.

## Optimization plan

### Phase 1: low-risk register and row-tile ablations

Generalize the grouped kernel over compile-time `J`, as dense forward already is.

Measure `J=64` and `J=128` for every production type and shape. `J=64` halves the accumulator footprint and activation LDS footprint.

Batch-1 gate/up is the strongest initial candidate because its uniform groups are exactly 64 rows and its observed-style groups are mostly below 128 rows.

Inspect the code object after each build. Reject variants that retain large private segments even if one short timing appears favorable.

Also test whether extracting one row-tile arithmetic body from the dynamic group loop reduces live state. A no-inline device helper is only a controlled ablation; it should not be retained if it creates dynamic stack use or device call overhead.

Required focused points are:

- gate/up Q3_K batch 1 sparse and uniform;
- gate/up IQ2_S batch 1 sparse;
- gate/up Q3_K batch 4 boundary;
- gate/up Q3_K batch 16 uniform;
- down Q4_K batch 4 uniform and boundary;
- down Q5_K batch 4 uniform;
- down IQ2_S batch 16 uniform.

### Phase 2: device-resident row-tile descriptors

If the dynamic loop still causes spills, move row-chunk scheduling out of the multiplication kernel.

A small GPU descriptor kernel can reserve `ceil(group_size / J)` entries per active expert with one device atomic, then write `(expert, row_start, row_count)` descriptors.

The descriptor storage upper bound is known without a device read:

```text
ceil(R / J) + G
```

A compute grid can launch to that upper bound and have excess workgroups return after reading a device-resident task count.

This preserves device metadata and avoids host synchronization. Descriptor storage is only tens of kilobytes for the production shapes.

The multiplication kernel then processes exactly one row tile. Its structure can closely reuse the already benchmarked dense-MMQ tile body and should recover the dense kernel's spill-free lowering.

`grouped_mmq_pair` should build descriptors once and reuse them for both packed projections.

The descriptor launch and empty upper-bound workgroups must be included in the public-operator timing.

### Phase 3: larger project-owned output tiles

After obtaining a spill-free row-tile body, measure larger output tiles.

The primary neighborhood is:

| Output tile `I` | Row tile `J` | Threads | Purpose |
|---:|---:|---:|---|
| 64 | 64 | 128 | minimum accumulator and LDS pressure |
| 64 | 128 | 128 | current arithmetic reuse without the grouped loop |
| 128 | 64 | 256 | wider output tile with 32 FP32 accumulators/thread |
| 128 | 128 | 256 | AITER-like logical tile with 64 FP32 accumulators/thread |

`I=128, J=128, 256 threads` keeps the current 64 accumulators per thread while halving output-tile count and Q8 activation reloads. Its estimated dynamic LDS is near the 64 KiB workgroup limit, so exact code-object and launch-resource checks are mandatory.

Keep this logic in project-owned files. Do not modify `csrc/vendor/llama_cpp/*`.

Use measured dispatch based on quant type, logical shape, total rows, and group count. `R / G` is available from tensor dimensions and does not require reading device metadata.

### Phase 4: type-specific decode and LDS tuning

Once the tile and scheduler are stable, apply the successful dense techniques:

- cooperative multi-value packed decode;
- vector global loads where alignment permits;
- vector LDS fragment loads;
- type-specific packed extraction;
- padding or XOR swizzles selected by measurement;
- exact full-tile specializations with bounded partial-tile paths.

Q4_K/Q5_K down is the priority. Q3_K/IQ2_S gate/up should preserve the existing batch-1 advantage while improving large batches.

Do not assume one decode layout will win across all four production types.

### Phase 5: pair-specific reuse

Only after the single-projection kernel is spill-free, test pair-specific changes.

Possible controlled experiments are:

- sharing row descriptors;
- sharing quantization, which already exists;
- interleaving the two projection launches for cache behavior;
- a fused activation-load path if register accounting proves it viable.

Do not retain a fused pair kernel that increases scratch traffic or loses the batch-1 sparse advantage.

### Phase 6: end-to-end validation

Re-run the full 60-point matrix after every retained dispatch change.

Report model-weighted grouped base-projection estimates for every routing distribution, not only one favorable case.

Then validate the real `fast_moe_lora.py` training path with current-stream correctness, activation checkpointing, and production routing metadata.

Run:

```bash
pytest -q tests/
python -m compileall -q bench

git diff --check
```

## Acceptance criteria

Every retained kernel must preserve exact grouped-versus-dense-MMQ BF16 output.

Q4_K and Q5_K grouped kernels should have no private segment or a small, measured, justified remainder. The current 512-520 bytes per thread is not acceptable as a final configuration.

Batch-1 sparse gate/up must retain a clear advantage over AITER.

Batch-4 down and batch-16 gate/up are the primary performance gates. A local improvement that regresses the model-weighted optimizer-step estimate should be rejected.

AITER remains the reference, not the ceiling. The final target is the best end-to-end packed grouped latency that preserves the production contract.

## Non-priorities and rejected directions

Do not replace AITER GMM with `torch.matmul` as the performance reference.

Do not link packed kernels against AITER or hipBLASLt.

Do not construct full logical BF16 expert matrices in the packed production path.

Do not introduce CPU expert descriptors or metadata synchronization.

Do not prioritize another quantizer rewrite before multiplication spills are removed.

Do not use runtime environment variables or online autotuning for dispatch.

Do not assume raw int8 WMMA throughput will overcome decode, LDS, scratch, or scheduling overhead.

Do not treat PC sampling on these short kernels as quantitative latency data.

## Immediate next step

Implement the compile-time `J=64` grouped variant and inspect its Q3_K, Q4_K, Q5_K, and IQ2_S code-object resources before broad benchmarking.

If Q4_K/Q5_K remain scratch-heavy, proceed directly to the device-resident row-tile descriptor scheduler and a dense-like single-tile multiplication kernel.
