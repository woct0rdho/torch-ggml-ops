# Exact MMQ kernel bundle and public launch design

## Purpose

This document defines how the selected GGTensile MMQ kernels and the selected HIP controls are turned into a deterministic gfx1151 bundle and launched by the public Python API. Kernel construction, physical planning, search, and performance promotion are documented in `ggtensile_plan.md` and the format-specific experiment records.

The public compute path is exact-key only. A problem whose key a GGTensile catalog carries is served by that kernel and every other deployed problem by its HIP control, and the resolution is baked into the generated table at build time, so a launch never picks a kernel from the shape at runtime. There is no nearby-shape selection, heuristic dispatch, online tuning, prepared-weight path, dense shadow, or implicit native allocation.

## Public API architecture

The allocating, autograd-capable Python functions are:

```python
torch_ggml_ops.mmq
torch_ggml_ops.grouped_mmq
torch_ggml_ops.grouped_mmq_pair
torch_ggml_ops.fixed_grouped_mmq
```

They retain PyTorch autograd for the input tensor. Packed weights and route metadata are nondifferentiable. Paired autograd always uses the fused paired input-gradient kernel, including when one output has no cotangent. Python supplies an explicit zero cotangent for the unused projection.

Two ordinary-MMQ functions expose allocation-free execution to enclosing custom autograd functions:

```python
torch_ggml_ops.mmq_inplace
torch_ggml_ops.mmq_grad_input_inplace
```

Both mutate caller-owned destination tensors and return `None`. `mmq_inplace` also mutates its caller-owned Q8_1 workspace. They do not install autograd edges. Tensor operands may be contiguous nonzero-storage-offset views when their effective data pointers satisfy the required alignment. Native validation remains authoritative.

There are deliberately no public dispatcher operators with these names. Direct legacy calls such as `torch.ops.torch_ggml_ops.mmq` are unsupported. The extension registers only private, allocation-free launch operators:
- `_mmq_launch` and `_mmq_grad_input_launch`.
- `_grouped_mmq_launch` and `_grouped_mmq_grad_input_launch`.
- `_grouped_mmq_pair_launch` and `_grouped_mmq_pair_grad_input_launch`.
- `_fixed_grouped_mmq_launch` and `_fixed_grouped_mmq_grad_input_launch`.

### Python allocation and autograd

`torch_ggml_ops/_mmq_cuda.py` derives output and temporary sizes directly from tensor dimensions, allocates every tensor explicitly, and calls one private launch operator. The native operators allocate no tensors.

Forward activation workspaces hold Q8_1 blocks and use:

```text
workspace_bytes = input.numel() / 128 * 144
```

for every valid exact key. Paired Q3_K forward allocates 64-row device tasks. Paired IQ2_S forward allocates 64-row tasks. Their explicit capacity is:

```text
ceil(aggregate_rows / task_rows) + route_entries
```

Serial paired routes receive zero-length task tensors. Output, workspace, task-count, task-expert, and task-range tensors are all visible to the private native launch.

`torch_ggml_ops/_mmq_autograd.py` owns the custom autograd functions. Backward passes autograd-owned cotangents directly to the native input-gradient kernels. It does not insert a copy or transpose. Cotangents may be aligned contiguous views, and native validation rejects invalid layouts rather than repairing them.

Python allocation is intentionally not a capability check. An unsupported request may allocate derived tensors first. The native launch validates the complete contract and exact deployment key before launching any GPU kernel.

### Native validation boundary

C++ is authoritative for:
- CUDA/HIP device placement and same-device relationships.
- BF16 activation/gradient/output dtypes.
- uint8 packed-weight and workspace dtypes.
- int64 expert indices and int32 offsets/tasks.
- exact ranks, physical packed shapes, logical output shapes, and element counts.
- contiguous layout. Storage offset itself is unrestricted.
- required effective-pointer alignment.
- positive and bounded dimensions.
- exactly 256 physical experts and 1-256 route entries for routed kernels.
- exact operation, quant type, `M`, `N`, and `K` deployment membership.
- exact row-task ownership, task size, and capacity.

Every failure occurs before quantization, task setup, or multiply launch. Native code never inserts a copy, creates a workspace, repairs a shape, or substitutes a kernel for the selected one.

## Selected inventory

All selected winners are loaded from the strict canonical catalogs in `tools/ggtensile/configs/mmq_*_catalog.json`.

The current public bundle contains 224 independently loadable artifacts:

| Artifact class | Count |
| --- | ---: |
| Q8_1 activation producers | 5 |
| Grouped row-task setup | 1 |
| Dense split-contraction reduction | 1 |
| Ordinary forward GGTensile | 50 |
| Ordinary backward GGTensile | 50 |
| Grouped forward GGTensile | 12 |
| Grouped paired forward GGTensile | 9 |
| Grouped backward GGTensile | 12 |
| Grouped paired backward GGTensile | 9 |
| Fixed grouped forward GGTensile | 3 |
| Fixed grouped backward GGTensile | 3 |
| HIP controls | 69 |

The 69 HIP controls serve 222 exact records: 207 ordinary dense keys, the 15 routed Qwen3.8 Q2_0 keys, and the three split-contraction Q5_K LM-head backward keys.

Resolution happens once, at generation time: a problem whose key is in a GGTensile catalog is served by that kernel, and a deployed problem without one is served by its HIP control. Both catalogs stay the single source of truth for their own winner, and the rule prefers GGTensile even where the HIP control is currently faster. A launch is therefore one host-table lookup with no runtime arbitration. Every routed and fixed-group problem of the two migrated families has a GGTensile kernel, so for them the fallback covers ordinary dense problems only. The Qwen3.8 Q2_0 routed family is HIP-only and is deployed through the routed rules of the same catalog.

A deployed split-contraction control is selected like any other HIP control. Its partial tiles and its reduction are two launches over buffers the Python layer allocates, which is what lets an enclosing compiled graph plan them. The public signatures are unchanged, and the C++ record carries the slice count, the reduction kernel index, and the slice width.

Two of the five Q8_1 producers are the grouped body that the F32_D4 and F16_D4S4 quantizers dispatch to while the activation row set is cache resident. The F16_D2S6 producer has no grouped body because its scale spans 64 values. The producer is part of the resolved record: a Q2_K problem takes F16_D2S6, a forward Q4_K or Q5_K problem takes F16_D4S4, and every other forward problem takes F32_D4. A backward problem consumes the BF16 gradient and quantizes nothing.

Each selected route stores one winner only. Artifact files use `<symbol>.hsaco`. Their names come from the canonical typed key. The deployment builder does not emit a manifest or record toolchain, source, object, code-object, or resource provenance. Benchmark medians, model names, rejected alternatives, and tuning heuristics are not deployment fields.

`csrc/generated/mmq_bundle_table.cuh` is generated directly from the typed inventory and contains:
- one ordered symbol array indexed by a numeric `MMQKernelIndex`.
- named indices for the seven setup artifacts.
- one exact deployment record per resolved problem, with the winning implementation and its activation producer.
- exact grid, workgroup, dynamic shared-memory request, ownership, row-task, and split-factor metadata, plus the split-contraction slice count, slice width, and reduction kernel.

`torch_ggml_ops/_deployment_records.py` is generated at the same time from the same resolution. It carries the device row-task tile and the split-contraction slice count of every record that has one, keyed by `(operation, quant_type, rows, out_features, in_features)`. The Python layer sizes its task banks and partial buffers from that table, so the allocation and the launch describe the same deployment. A missing key means the record is one launch over the route bank.

There is no generic `KernelNNN` runtime identity or tuning database.

## Build and package pipeline

`tools/mmq_deployment_bundle.py` is the build entry point. It is run as a module from the repository root, because `bench/` and `tools/` are repository scripts and are never packaged into the wheel. The builder:
- Loads every public canonical catalog from `tools/ggtensile/configs/` and the deployed HIP table from `tools/configs/hip_deployment.json`.
- Initializes every GGTensile writer serially and emits deterministic assembly.
- Assembles and links GGTensile sources for gfx1151, wave32, code-object v5.
- Compiles the five quantizers, row-task setup, the dense split-contraction reduction, and the selected HIP controls with HIP using deterministic compiler-unit settings.
- Verifies ELF target data and the single expected exported symbol.
- Generates operation/quant constants, the exact host table, and the Python record contract from the typed inventory.
- Installs the complete set transactionally and removes stale artifacts.

Temporary object files are deleted and never packaged. A failed build removes its staging directory. Every invocation regenerates every public kernel. There is no incremental bundle stamp or freshness check.

The build entry point is:

```bash
python -m tools.mmq_deployment_bundle --jobs 16
```

`setup.py build_ext`, wheel builds, and editable installs run the public bundle builder before compiling `_C.abi3.so`, then copy the exact public HSACO set into the wheel build tree. HIP controls remain outside the public package. Source distributions are not supported. HSACOs remain ignored by Git.

The HIP control build is a separate step. The HSACOs are ignored build outputs and are expected to be rebuilt on a new checkout or after a compiler/source change.

From the repository root, with the gfx1151 ROCm toolchain available:

```bash
python -m tools.build_mmq_hip_controls --jobs 16
python -m tools.build_mmq_hip_controls --check
```

Use `--force` when recovering from a stale or partially copied output directory:

```bash
python -m tools.build_mmq_hip_controls --force --jobs 16
```

`--verify-reproducible` compiles the complete retained inventory twice, compares the resulting bytes, and installs the first build only after the comparison succeeds:

```bash
python -m tools.build_mmq_hip_controls --verify-reproducible --jobs 16
```

The builder requires `hipcc` (or `--hipcc /path/to/hipcc`) and the matching `amdclang++`, `llvm-readelf`, `llvm-objdump`, and `llvm-objcopy` tools. `amdclang++` may be selected with `GGTENSILE_AMDCLANGXX`. The LLVM tools are normally found beside it or on `PATH`. The checked-in `csrc/mmq_core.cuh`, `csrc/ck/`, and `csrc/vendor/llama_cpp/` headers are the source inputs. No GPU is required to compile, although the device tests still require a compatible gfx1151 system and runtime.

Every header under `csrc/` is self-contained, so include order never matters and a translation unit includes exactly what it uses. `python -m tools.check_mmq_headers` compiles each header as its own translation unit and fails on a missing include, a missing include guard, or an unresolvable quoted include. Each HIP control includes only the family header that defines its body, so the generated translation units stay independent of each other. The three vendored `csrc/vendor/llama_cpp/mmq-*.cuh` templates are configuration fragments rather than headers: they expand against the `MMQ_*` settings and helpers that `mmq_core.cuh` defines before including them, and that header is their only include site.

`python -m tools.check_cpp_style` enforces the block convention that Composable Kernel and llama.cpp share on every checked-in C++/CUDA file: four spaces per block level, no tabs, no line shallower than its block level, and local includes before parent-directory includes. Line breaks, line width and vertical alignment are semantic choices and are not checked.

The inventory itself is data. `tools/configs/hip_deployment.json` names the body each exact key selects and lists the bodies the direct launchers use outside a key under `helpers`. `tools/configs/hip_control_catalog.json` holds the build parameters of exactly those bodies, one record per symbol. `tools/mmq_hip_control_spec.py` loads the two files and renders each wrapper. A candidate that no key selects is not built - the experiment records hold its screen - so adding a candidate means adding its record, and dropping a screen means deleting it.

A successful build atomically installs one bare-symbol file per control in that inventory under `build/mmq_hip_controls/gfx1151/` and writes a freshness stamp there. The current launchers expect names such as `grouped_fwd_serial_q2_k_n4096_k2048_j32.hsaco`. Older prefixed files such as `torch_ggml_ops_mmq_gfx1151_v1_<symbol>.hsaco` do not satisfy lookup and are replaced by a current rebuild. `--check` exits nonzero when the inventory, stamp, compiler, or source inputs are stale.

`tools/configs/hip_deployment.json` is the single source of truth for the fastest HIP kernel of every problem: it names the body each exact key selects. Direct-kernel benchmark runners accept `--hip-root` for the directory containing the built control set. Tests and runners otherwise use `GGTENSILE_HIP_CONTROL_ROOT` when set, followed by `build/mmq_hip_controls/gfx1151` when it is available. These controls are comparison artifacts only. Their presence does not change the 148-route public inventory.

## Runtime loading and launch

The installed layout is:

```text
torch_ggml_ops/
  _C.abi3.so
  kernels/gfx1151/
    <exact-symbol>.hsaco
    ...

# Optional HIP controls (outside the public package):
build/mmq_hip_controls/gfx1151/
  <control-symbol>.hsaco
  ...
```

`csrc/mmq_bundle_loader.cpp` locates `_C.abi3.so` with `dladdr` and resolves the kernel directory relative to the extension, independent of the process working directory. On first use of `(device, kernel index)`, it reads and retains the artifact bytes, loads the module, resolves the exact symbol, and caches the module/function. A mutex serializes first resolution.

`csrc/mmq_bundle.cpp` owns exact-record lookup and ABI argument packing, including the producer index and the dynamic shared-memory request (zero for a GGTensile record, whose artifact owns its LDS in the code object's group segment). Launch uses PyTorch's current stream and the grid/workgroup recorded for the exact route. Grouped serial routes replace the route-count grid dimension with the validated active route count. Static split ownership scales that same grid-Y route dimension by the exact split factor stored in the generated record. Paired packed-split kernels decode both route and split ownership from workgroup Y. Device row-task routes first launch the explicit setup artifact, then launch the exact multiply over bounded task slots. The task bank is a Python-owned buffer the caller passes in. A split-contraction dense backward writes FP32 partial tiles over the record's slice grid and a second launch reduces them into the BF16 gradient.

Artifact read, module load, symbol lookup, or launch failure is fatal and names the failing path or symbol. The loader never probes another artifact or invokes embedded arithmetic.

## Exact public compatibility

`M`, `N`, and `K` below are exact problem coordinates: `M` is the row count, `N` is the weight's `out_features` (its packed rows) and `K` is its `in_features` (the values per packed row), in both directions. Forward computes `[M,K] @ [N,K]^T -> [M,N]`. Backward computes the input gradient `[M,N] @ [N,K] -> [M,K]`. Both directions therefore list the same `(N,K)` for a quant type.

The table lists every GGTensile key. A deployed ordinary dense problem outside it, such as the Q2_0, Q4_0, Q5_0, IQ4_NL, and IQ4_XS matrices, the extra shapes of the shared quant types, and the chunked Q5_K LM-head keys, is served by the HIP control its key selects. The Qwen3.8 matrices widen the deployed set with `(N,K)` in `{(12288,2560),(6144,2560),(10240,2560),(640,2560),(2560,640),(512,2560)}` for Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q5_0, Q8_0, IQ4_NL and IQ4_XS, and with `M in {64,128,256}`, `(N,K)=(248320,2560)` for the split-contraction Q5_K LM head in both directions.

A forward key and its backward key are selected independently. If a supported forward is used with an input requiring gradients, its corresponding backward key must also appear below.

HIP control symbols spell the same letters, plus a tile geometry that follows the matrix instruction. A control that serves one shape spells a family key: a routed control `n<out_features>_k<in_features>`, the weight's own letters, identical for the forward and backward symbol of a family, and a fixed-group control `g<group count>_k<in_features>`. A dense control serves several shapes and carries only the trailing tags: `mt<row block>`, `nt<result columns, or the count of 16-column tiles>`, `ki<contraction stage, or its count>`, `j<token tile>`, `g<group count>`, `pad<LDS padding words>`, `sw<LDS swizzle chunk>`, `mw<m_tiles_per_wave>`, `s<split or stage count>` and `mb<min resident blocks>`.

The instruction computes `D[M,N] = A[M,K] * B[K,N]`, so its `N` is the dimension a tile writes and its `K` the contraction it walks: a backward control writes `in_features` and contracts over `out_features`, which makes its `nt`/`ki` the transpose of the family key's `n`/`k`, while a forward control's letters agree. The contraction stage is never written as a bare `k`, so `k<value>` in a symbol means `in_features`.

### Ordinary MMQ

| Direction | Quant type | Exact `(M,N,K)` support |
| --- | --- | --- |
| Forward | Q3_K | `M in {2048,8192,32768}`, `(N,K)` in `{(512,2048),(8192,2048)}` |
| Forward | Q4_K | `M in {2048,8192,32768}`, `(N,K)` in `{(512,2048),(2048,512),(2048,4096),(8192,2048)}` |
| Forward | Q5_K | `M in {2048,8192,32768}`, `(N,K)` in `{(512,2048),(2048,512)}` |
| Forward | Q6_K | `(64,248320,2048)`, `(128,248320,2048)`, `(256,248320,2048)` |
| Forward | Q8_0 | `M in {2048,8192,32768}` with `(N,K)` in `{(1024,4096),(32768,1024),(512,4096),(4096,8192),(2048,4096),(4096,2048)}`. Also `M in {32,64,128,256,512}`, `(N,K)=(129280,4096)` |
| Backward | Q3_K | `M in {2048,8192,32768}`, `(N,K)` in `{(512,2048),(8192,2048)}` |
| Backward | Q4_K | `M in {2048,8192,32768}`, `(N,K)` in `{(512,2048),(2048,512),(2048,4096),(8192,2048)}` |
| Backward | Q5_K | `M in {2048,8192,32768}`, `(N,K)` in `{(512,2048),(2048,512)}` |
| Backward | Q6_K | `(64,248320,2048)`, `(128,248320,2048)`, `(256,248320,2048)` |
| Backward | Q8_0 | `M in {2048,8192,32768}` with `(N,K)` in `{(512,4096),(1024,4096),(2048,4096),(4096,2048),(4096,8192),(32768,1024)}`. Also `M in {32,64,128,256,512}`, `(N,K)=(129280,4096)` |

Ordinary `Q2_K`, `IQ2_XXS`, and `IQ2_S` have no deployed key. Every ordinary shape not listed is unsupported.

### Routed grouped MMQ

Routed inputs have shape `[R,K]`. Packed weights have physical shape `[E,N,packed_row_bytes]` where `E` is the expert axis of the packed bank (at most 4096), and expert indices and cumulative offsets contain the same number of entries, from 1 through `E`. A routed family is keyed by `(operation, quant_type, R, N, K)`. The expert count is read from the bank, so a bank with fewer experts than the family's model serves the same record as long as the route entries fit it. The contraction width `K` must be a multiple of the 128-value activation block. The deployed Q2_0 bodies carry the remainder of a 256-value stage as a tail, which is what serves `K=640`.

| Direction | Quant type | Exact support | Ownership |
| --- | --- | --- | --- |
| Forward | Q4_K, Q5_K, IQ2_S | `R in {16384,65536,262144}`, `(N,K)=(2048,512)` | Serial routes |
| Forward | Q2_K | `R in {12288,49152,196608}`, `(N,K)=(4096,2048)` | Serial routes |
| Forward | Q2_0 | `R in {20480,81920,327680}`, `(N,K)=(2560,640)` | Serial routes |
| Forward | Q2_0 | `R in {20480,81920}`, `(N,K)=(640,2560)` | Serial routes |
| Forward | Q2_0 | `R=327680`, `(N,K)=(640,2560)` | Serial routes, wide J128 tile |
| Paired forward | IQ2_S | `R in {16384,65536,262144}`, `(N,K)=(512,2048)` | Device tasks, 64 rows/task |
| Paired forward | Q3_K | `R in {16384,65536,262144}`, `(N,K)=(512,2048)` | Device tasks, 64 rows/task |
| Paired forward | IQ2_XXS | `R in {12288,49152,196608}`, `(N,K)=(2048,4096)` | Serial routes |
| Paired forward | Q2_0 | `R in {20480,81920,327680}`, `(N,K)=(640,2560)` | Two single launches, one per packed bank |
| Backward | Q4_K, Q5_K, IQ2_S | `R in {16384,65536,262144}`, `(N,K)=(2048,512)` | Exact serial/static-split policy per key |
| Backward | Q2_K | `R in {12288,49152,196608}`, `(N,K)=(4096,2048)` | Exact serial/static-split policy per key |
| Backward | Q2_0 | `R=20480`, `(N,K)=(2560,640)` | Device tasks, 128 rows/task |
| Backward | Q2_0 | `R in {81920,327680}`, `(N,K)=(2560,640)` | Device tasks, 256 rows/task |
| Paired backward | Q3_K, IQ2_S | `R in {16384,65536,262144}`, `(N,K)=(512,2048)` | Exact pair policy per key |
| Paired backward | IQ2_XXS | `R in {12288,49152,196608}`, `(N,K)=(2048,4096)` | Exact pair policy per key |
| Paired backward | Q2_0 | `R in {20480,81920}`, `(N,K)=(640,2560)` | Device tasks, 128 rows/task |
| Paired backward | Q2_0 | `R=327680`, `(N,K)=(640,2560)` | Device tasks, 256 rows/task |

There is no standalone grouped forward for Q3_K or IQ2_XXS, no standalone grouped backward for Q3_K or IQ2_XXS, and no grouped Q6_K or routed Q8_0 deployment. A paired row without a fused body — the Q2_0 gate/up forward — launches the single-projection body once per packed bank over the same route bank and activation workspace.

### Fixed grouped Q8_0

Fixed grouped input has logical shape `[...,8,4096]`. Packed weight has physical shape `[8,1024,4352]`. Output is `[...,8,1024]`.

| Direction | Exact support |
| --- | --- |
| Forward | Tokens in `{2048,8192,32768}`, 8 groups, `(N,K)=(1024,4096)` |
| Backward | Tokens in `{2048,8192,32768}`, 8 groups, `(N,K)=(1024,4096)` |

Every other token count, group count, dimension, or quant type is unsupported.

## Failure and non-goals

Expected hard failures include:
- an unsupported exact deployment key.
- invalid dtype, rank, shape, element count, layout, effective-pointer alignment, or device.
- inconsistent paired problems or route metadata lengths.
- missing, unreadable, invalid, or wrong-symbol artifacts.
- HIP module or kernel-launch errors.
- use on an architecture other than the packaged gfx1151 target.

The bundle does not provide runtime compilation, architecture substitution, online benchmarking, environment-variable tuning, fallback arithmetic, host reads of route values, or a compatibility dispatcher for removed public native operators.

## Validation gates

Changes to selected keys, bundle generation, launch packing, or public wrappers require:
- strict catalog and deployment-inventory tests.
- a self-contained header layout (`python -m tools.check_mmq_headers`).
- the shared block indentation and include order (`python -m tools.check_cpp_style`).
- typed launch-metadata derivation checks.
- exact artifact-count and unique-symbol checks.
- one complete regeneration of the selected bundle from the typed inventory.
- editable or wheel build of the extension.
- native validation tests for dtype, device, contiguity, shape, element count, and effective-pointer alignment.
- packed-reference and independent numerical checks for affected forward/backward families.
- autograd and `torch.compile(fullgraph=True)` checks for ordinary, grouped paired, and fixed paths.
- the complete project test suite, pre-commit, and `git diff --check`.

A new selected kernel must first pass the generation and tuning process in `ggtensile_plan.md`. Integration then adds exactly one typed winner, rebuilds the generated table and artifacts, and extends the exact compatibility matrix in this document.
