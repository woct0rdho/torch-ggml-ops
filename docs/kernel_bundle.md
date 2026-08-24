# Exact MMQ kernel bundle and public launch design

## Purpose

This document defines how selected GGTensile MMQ kernels are turned into a deterministic gfx1151 bundle and launched by the public Python API. Kernel construction, physical planning, search, and performance promotion are documented in `ggtensile_plan.md` and the format-specific experiment records.

The public compute path is exact-key only. There is no HIP multiply fallback, nearby-shape selection, heuristic dispatch, online tuning, prepared-weight path, dense shadow, or implicit native allocation.

## Public API architecture

The exported Python functions are:

```python
torch_ggml_ops.mmq
torch_ggml_ops.grouped_mmq
torch_ggml_ops.grouped_mmq_pair
torch_ggml_ops.fixed_grouped_mmq
```

They retain PyTorch autograd for the input tensor. Packed weights and route metadata are nondifferentiable. Paired autograd always uses the fused paired input-gradient kernel, including when one output has no cotangent; Python supplies an explicit zero cotangent for the unused projection.

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

for every valid exact key. Paired Q3_K forward allocates 64-row device tasks; paired IQ2_S forward allocates 32-row tasks. Their explicit capacity is:

```text
ceil(aggregate_rows / task_rows) + route_entries
```

Serial paired routes receive zero-length task tensors. Output, workspace, task-count, task-expert, and task-range tensors are all visible to the private native launch.

`torch_ggml_ops/_mmq_autograd.py` owns the custom autograd functions. Backward makes only autograd-owned cotangents contiguous; public inputs and packed weights are never copied or repaired.

Python allocation is intentionally not a capability check. An unsupported request may allocate derived tensors first. The native launch validates the complete contract and exact deployment key before launching any GPU kernel.

### Native validation boundary

C++ is authoritative for:
- CUDA/HIP device placement and same-device relationships.
- BF16 activation/gradient/output dtypes.
- uint8 packed-weight and workspace dtypes.
- int64 expert indices and int32 offsets/tasks.
- exact ranks, physical packed shapes, logical output shapes, and element counts.
- contiguity and zero storage offsets.
- required pointer alignment.
- positive and bounded dimensions.
- exactly 256 physical experts and 1-256 route entries for routed kernels.
- exact operation, quant type, `M`, `N`, and `K` deployment membership.
- exact row-task ownership, task size, and capacity.

Every failure occurs before quantization, task setup, or multiply launch. Native code never inserts a copy, creates a workspace, repairs a shape, or chooses a substitute kernel.

## Selected inventory

Selected ordinary winners are loaded from the ten strict catalogs in `tools/ggtensile/configs/mmq_<direction>_<quant>_catalog.json`. Selected grouped, paired, and fixed winners are loaded from `tools/ggtensile/configs/mmq_deployment.json`.

The current public bundle contains 152 independently loadable artifacts:

| Artifact class | Count |
| --- | ---: |
| Q8_1 activation producers | 3 |
| Grouped row-task setup | 1 |
| Ordinary forward GGTensile | 50 |
| Ordinary backward GGTensile | 50 |
| Grouped forward GGTensile | 12 |
| Grouped paired forward GGTensile | 9 |
| Grouped backward GGTensile | 12 |
| Grouped paired backward GGTensile | 9 |
| Fixed grouped forward GGTensile | 3 |
| Fixed grouped backward GGTensile | 3 |

The four setup artifacts are the only HIP-compiled entries in the public bundle. All 148 public multiply artifacts come from typed GGTensile assembly writers. Historical HIP controls are built separately for research comparisons; they are never part of public dispatch or used as a fallback.

Each selected route stores one winner only. Inventory parsing reconstructs the typed exact key and verifies its hash, symbol, ABI, workgroup, grid, fixed LDS size, ownership, and row-task bounds against facts derived from that key. Benchmark medians, model names, rejected alternatives, and tuning heuristics are not deployment fields.

`csrc/generated/mmq_bundle_table.cuh` contains:
- one ordered symbol array indexed by a numeric `MMQKernelIndex`.
- named indices only for the four setup artifacts.
- one exact deployment record per multiply artifact.
- exact grid, workgroup, ownership, and row-task metadata.

There is no generic `KernelNNN` runtime identity or tuning database.

## Build and package pipeline

`tools/mmq_deployment_bundle.py` is the implementation; `tools/build_mmq_bundle.py` is the stable build entry point. The builder:
- Loads strict ordinary catalogs and the grouped deployment inventory.
- Initializes every GGTensile writer serially and emits deterministic assembly.
- Assembles and links GGTensile sources for gfx1151, wave32, code-object v5.
- Compiles only the three quantizers and row-task setup with HIP using deterministic compiler-unit settings.
- Verifies ELF target data and the single expected exported symbol.
- Generates the exact host table.
- Installs the complete set transactionally and removes stale artifacts.

Temporary object files are deleted and never packaged. A failed build removes its staging directory. The ignored `.mmq-build-input` stamp hashes the relevant toolchain identity, typed catalogs/inventory, writers, renderer, and device headers.

Useful gates are:

```bash
python tools/build_mmq_bundle.py --check
python tools/build_mmq_bundle.py --verify-reproducible --jobs 16
```

The reproducibility gate builds the complete bundle twice and requires byte-identical artifacts in inventory order. `--check` verifies the exact artifact set, generated header, and input stamp.

`setup.py build_ext`, wheel builds, and editable installs run the public bundle builder before compiling `_C.abi3.so`, then copy the exact public HSACO set into the wheel build tree. Historical controls remain outside the public package. Source distributions are not supported. HSACOs and local build stamps remain ignored by Git.

The historical control build is an explicit research-only step:

```bash
python tools/build_mmq_hip_controls.py --check
python tools/build_mmq_hip_controls.py --verify-reproducible --jobs 16
```

Direct-kernel benchmark runners accept `--hip-root` for the directory containing the historical-control set and otherwise use `build/mmq_hip_controls/gfx1151` when it is available. These controls are comparison artifacts only; their presence does not change the 148-route public inventory.

## Runtime loading and launch

The installed layout is:

```text
torch_ggml_ops/
  _C.abi3.so
  kernels/gfx1151/
    <exact-versioned-symbol>.hsaco
    ...

# Optional research-only controls (outside the public package):
build/mmq_hip_controls/gfx1151/
  <historical-control-symbol>.hsaco
  ...
```

`csrc/mmq_bundle_loader.cpp` locates `_C.abi3.so` with `dladdr` and resolves the kernel directory relative to the extension, independent of the process working directory. On first use of `(device, kernel index)`, it reads and retains the artifact bytes, loads the module, resolves the exact symbol, and caches the module/function. A mutex serializes first resolution.

`csrc/mmq_bundle.cpp` owns exact-record lookup and ABI argument packing. Launch uses PyTorch's current stream and the grid/workgroup recorded for the exact route. Grouped serial routes replace the route-count grid dimension with the validated active route count. Static split ownership scales that same grid-Y route dimension by the exact split factor stored in the generated record; paired packed-split kernels decode both route and split ownership from workgroup Y. Device row-task routes first launch the explicit setup artifact, then launch the exact paired multiply over bounded task slots.

Artifact read, module load, symbol lookup, or launch failure is fatal and names the failing path or symbol. The loader never probes another artifact or invokes embedded arithmetic.

## Exact public compatibility

`M`, `N`, and `K` below use the multiply-kernel convention. Forward computes `[M,K] @ [N,K]^T -> [M,N]`. Backward computes an input gradient with problem coordinates `[M,N] @ [K,N] -> [M,K]`; the table lists the backward kernel's `(M,N,K)` directly.

A forward key and its transposed backward key are selected independently. If a supported forward is used with an input requiring gradients, its corresponding backward key must also appear below.

### Ordinary MMQ

| Direction | Quant type | Exact `(M,N,K)` support |
| --- | --- | --- |
| Forward | Q3_K | `M in {2048,8192,32768}`, `(N,K)` in `{(512,2048),(8192,2048),(4096,2048),(2048,4096)}` |
| Forward | Q4_K | `M in {2048,8192,32768}`, `(N,K)` in `{(512,2048),(2048,512),(2048,4096),(8192,2048)}` |
| Forward | Q5_K | `M in {2048,8192,32768}`, `(N,K)` in `{(512,2048),(2048,512)}` |
| Forward | Q6_K | `(64,248320,2048)`, `(128,248320,2048)`, `(256,248320,2048)` |
| Forward | Q8_0 | `M in {2048,8192,32768}` with `(N,K)` in `{(1024,4096),(32768,1024),(512,4096),(4096,8192),(2048,4096),(4096,2048)}`; also `M in {32,64,128,256,512}`, `(N,K)=(129280,4096)` |
| Backward | Q3_K | `M in {2048,8192,32768}`, `(N,K)` in `{(2048,512),(2048,8192)}` |
| Backward | Q4_K | `M in {2048,8192,32768}`, `(N,K)` in `{(2048,512),(512,2048),(4096,2048),(2048,8192)}` |
| Backward | Q5_K | `M in {2048,8192,32768}`, `(N,K)` in `{(2048,512),(512,2048)}` |
| Backward | Q6_K | `(64,2048,248320)`, `(128,2048,248320)`, `(256,2048,248320)` |
| Backward | Q8_0 | `M in {2048,8192,32768}` with `(N,K)` in `{(4096,1024),(1024,32768),(4096,512),(8192,4096),(4096,2048),(2048,4096)}`; also `M in {32,64,128,256,512}`, `(N,K)=(4096,129280)` |

Ordinary `Q2_K`, `IQ2_XXS`, and `IQ2_S` have no deployed key. Every ordinary shape not listed is unsupported.

### Routed grouped MMQ

Routed inputs have shape `[R,K]`; packed weights have physical shape `[256,N,packed_row_bytes]`. Expert indices and cumulative offsets contain the same number of entries, from 1 through 256.

| Direction | Quant type | Exact support | Ownership |
| --- | --- | --- | --- |
| Forward | Q4_K, Q5_K, IQ2_S | `R in {16384,65536,262144}`, `(N,K)=(2048,512)` | Serial routes |
| Forward | Q2_K | `R in {12288,49152,196608}`, `(N,K)=(4096,2048)` | Serial routes |
| Paired forward | IQ2_S | `R in {16384,65536,262144}`, `(N,K)=(512,2048)` | Device tasks, 32 rows/task |
| Paired forward | Q3_K | `R in {16384,65536,262144}`, `(N,K)=(512,2048)` | Device tasks, 64 rows/task |
| Paired forward | IQ2_XXS | `R=196608`, `(N,K)=(2048,4096)` | Serial routes |
| Backward | Q4_K, Q5_K, IQ2_S | `R in {16384,65536,262144}`, `(N,K)=(512,2048)` | Exact serial/static-split policy per key |
| Backward | Q2_K | `R in {12288,49152,196608}`, `(N,K)=(2048,4096)` | Exact serial/static-split policy per key |
| Paired backward | Q3_K, IQ2_S | `R in {16384,65536,262144}`, `(N,K)=(2048,512)` | Exact pair policy per key |
| Paired backward | IQ2_XXS | `R in {12288,49152,196608}`, `(N,K)=(4096,2048)` | Exact pair policy per key |

There is no standalone grouped forward for Q3_K or IQ2_XXS, no standalone grouped backward for Q3_K or IQ2_XXS, and no grouped Q6_K or routed Q8_0 deployment. Formats absent from a paired row have no pair-via-two-singles behavior.

### Fixed grouped Q8_0

Fixed grouped input has logical shape `[...,8,4096]`; packed weight has physical shape `[8,1024,4352]`; output is `[...,8,1024]`.

| Direction | Exact support |
| --- | --- |
| Forward | Tokens in `{2048,8192,32768}`, 8 groups, `(N,K)=(1024,4096)` |
| Backward | Tokens in `{2048,8192,32768}`, 8 groups, `(N,K)=(4096,1024)` |

Every other token count, group count, dimension, or quant type is unsupported.

## Failure and non-goals

Expected hard failures include:
- an unsupported exact deployment key.
- invalid dtype, rank, shape, element count, layout, storage offset, alignment, or device.
- inconsistent paired problems or route metadata lengths.
- missing, unreadable, invalid, or wrong-symbol artifacts.
- HIP module or kernel-launch errors.
- use on an architecture other than the packaged gfx1151 target.

The bundle does not provide runtime compilation, architecture substitution, online benchmarking, environment-variable tuning, fallback arithmetic, host reads of route values, or a compatibility dispatcher for removed public native operators.

## Validation gates

Changes to selected keys, bundle generation, launch packing, or public wrappers require:
- strict catalog and deployment-inventory tests.
- typed launch-metadata derivation checks.
- exact artifact-count and unique-symbol checks.
- two-pass bundle reproducibility and `--check`.
- editable or wheel build of the extension.
- native validation tests for dtype, device, contiguity, storage offset, shape, element count, and alignment.
- packed-reference and independent numerical checks for affected forward/backward families.
- autograd and `torch.compile(fullgraph=True)` checks for ordinary, grouped paired, and fixed paths.
- the complete project test suite, pre-commit, and `git diff --check`.

A new selected kernel must first pass the generation and tuning process in `ggtensile_plan.md`. Integration then adds exactly one typed winner, rebuilds the generated table and artifacts, implements no fallback path, and extends the exact compatibility matrix in this document.
