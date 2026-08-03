# Architecture-specific kernel bundle design

## Purpose

This document defines the reusable build, artifact, loading, ABI, packaging, and validation design for architecture-specific device-kernel bundles.

It intentionally does not define which MMQ geometry, quantization layout, row threshold, tail policy, or scheduler is selected. Those are kernel-behavior decisions and belong in the corresponding optimization records:
- dense forward: `docs/mmq_fwd_optimization.md`.
- dense backward: `docs/mmq_bwd_optimization.md`.
- grouped forward: `docs/grouped_mmq_fwd_optimization.md`.
- grouped backward: `docs/grouped_mmq_bwd_optimization.md`.

The public PyTorch operators remain in `_C.abi3.so`. They continue to own validation, tensor and workspace allocation, stream selection, and host-visible static dispatch. Architecture-specific device entry points are independently compiled into standalone HSACO artifacts and launched through the HIP module API.

## Design goals

The bundle must provide:
- one reviewed source of truth for every packaged kernel entry.
- deterministic, reproducible code objects independent of temporary build paths.
- stable architecture- and ABI-versioned symbols.
- one independently replaceable artifact per exported kernel.
- lazy per-device loading without dependence on the process working directory.
- explicit failures for missing, invalid, or unsupported artifacts.
- build-time resource assertions for performance-sensitive entries.
- source-only version control and source distributions.
- locally generated wheel contents.
- no runtime compilation or online autotuning.

The bundle is an artifact and ownership boundary, not an optimization policy. A kernel may be added, removed, or retuned only through its optimization log and reviewed static dispatch. The bundle infrastructure should not infer performance policy from filenames, metadata, or compiler resources.

## Non-goals

The current design does not provide:
- runtime HIP compilation.
- runtime autotuning.
- content-addressed artifact selection.
- a silent arithmetic fallback when a selected artifact fails.
- cross-architecture substitution.
- host reads of device routing metadata.
- model-family names in the device ABI.
- launcher-overhead optimization.

Unsupported architectures, quant types, operator modes, or geometries remain explicit errors unless the owning operator documents another public fallback.

## Separation of concerns

The implementation has four layers.

### Reusable device bodies

Project-owned headers contain the arithmetic and scheduling bodies:
- `csrc/mmq_core.cuh`.
- `csrc/ck/mmq_backward.cuh`.
- `csrc/ck/grouped_mmq_backward.cuh`.
- `csrc/ck/grouped_mmq_backward_tiled.cuh`.

These bodies may use templates and compile-time parameters, but do not own package discovery or host dispatch.

### Generated concrete wrappers

`tools/mmq_bundle_wrapper_source.py` renders one small `.cu` translation unit per kernel specification. Each unit exposes one literal `extern "C" __global__` symbol, states its launch signature, and calls one selected device body with explicit template arguments.

MMQ forward, MMQ backward, and grouped MMQ backward entries remain separate when they have incompatible argument ABIs or launch-bound contracts. The generated units contain no family or geometry selector macros and no model-level dispatch policy. They are deterministic ignored build inputs under `build/`. Sdists contain the typed renderer and device sources rather than the duplicated generated files.

### Deterministic generator

`tools/build_mmq_bundle.py` owns:
- the ordered kernel inventory.
- public C++ kernel IDs.
- symbol and filename generation.
- typed wrapper configuration.
- concrete wrapper source generation.
- deterministic compiler invocation.
- artifact verification.
- optional resource gates.
- generated host load-table contents.
- stale-artifact pruning and atomic installation.

The inventory is implementation data, not a design constant. Kernel counts and selected specializations may change without changing this document.

### Host dispatcher and loader

`csrc/mmq_bundle.cpp` owns:
- architecture-specific package discovery.
- static kernel-ID selection requested by the operator.
- launch geometry and dynamic shared-memory calculation.
- retained artifact bytes.
- HIP module and function caching.
- `hipModuleLaunchKernel` calls.
- explicit error reporting.

`csrc/mmq_bundle.h` is the host launch API consumed by the extension operator layer. `csrc/mmq_hip.cu` owns public operator behavior and does not define or directly launch packaged MMQ `__global__` functions.

## Kernel specification contract

Each generator entry must define enough information to compile and load exactly one kernel:
- stable C++ kernel ID.
- stable exported symbol.
- stable artifact filename.
- typed wrapper family and configuration.
- explicit template values.
- whether the artifact is subject to the production resource gate.

The generator order must be deterministic and reviewed. The generated host table contains only loading identity, currently symbol and filename. Kernel behavior, launch geometry, and selection thresholds remain reviewed C++ policy rather than runtime artifact metadata.

A packaged entry must export exactly one expected public kernel symbol. Internal helper symbols are allowed only when they do not create another public launch ABI.

## Symbol and ABI versioning

Symbols use an architecture- and ABI-versioned prefix:

```text
torch_ggml_ops_mmq_<architecture>_v<abi>_
```

Suffixes describe reusable implementation traits such as operator family, execution mode, quant format, compile-time geometry, bounds mode, or workspace layout. They must not encode checkpoint names, model families, or benchmark case names.

Changing any of the following requires a new ABI version unless compatibility is deliberately preserved:
- argument order, type, or width.
- kernarg interpretation.
- workspace layout.
- output layout.
- dynamic shared-memory interpretation.
- launch-dimension interpretation.

Selection-only changes among ABI-compatible existing entries do not require renaming artifacts.

Architecture directory versioning and symbol ABI versioning are independent. A new GPU target gets a separate architecture directory and generated inventory. ABI versions may coexist during migration.

## Artifact layout

Installed wheels use an extension-relative layout:

```text
torch_ggml_ops/
  _C.abi3.so
  kernels/
    <architecture>/
      <versioned-symbol>.hsaco
      ...
```

Each HSACO is independently loadable and exports one expected public symbol. There is no release link step between entries.

Runtime JSON manifests, content-digest filenames, byte-count sidecars, and duplicate architecture metadata are intentionally unnecessary. The generated C++ table and HIP's code-object validation are the trusted loading boundary.

## Build pipeline

The generator performs the following transaction:
- Enumerate entries in a stable reviewed order.
- Hash all build inputs used for cache invalidation:
  - compiler identity.
  - architecture and compile options.
  - ordered kernel specifications.
  - the concrete-wrapper renderer.
  - included project-owned device headers.
- Compile every entry independently with `hipcc --genco -c` for the target architecture.
- Verify the artifact format, target architecture, and expected exported symbol.
- Read compiler resource metadata and apply the entry's resource gate when requested.
- Write artifacts, the non-source build-input stamp, and any changed generated host table to staging paths.
- Replace installed outputs atomically only after every entry succeeds.
- Remove stale artifacts that are no longer in the inventory.

The current gfx1151 build uses direct unbundled GPU code-object output, optimization level `-O3`, an explicit C++ language level, a source-prefix map, and a deterministic per-symbol Clang CUID. Changes to these options are build-input changes and invalidate the generated set.

The aggregate digest is stored in the ignored `.mmq-build-input` file beside locally generated HSACOs. It is not emitted into C++ source. The generated host header is rewritten only when its kernel IDs, symbols, or filenames change, so build-policy or kernel-body changes do not make the extension translation units stale.

## Compile cache

When `ccache` is available, the generator invokes `ccache hipcc` automatically. The explicit compile-only flag is required because ccache otherwise classifies `hipcc --genco` as a link operation and does not store it.

Generated wrappers use deterministic content-addressed source paths under `build/mmq_bundle_sources/<architecture>`. This keeps the ccache key stable across transactional artifact staging directories. The cache namespace includes the HIP compiler identity and common compiler options. Source contents and included header contents remain normal ccache inputs, so a wrapper-only change misses only that wrapper while a shared device-header change correctly misses every affected entry.

The aggregate bundle digest and atomic installation policy are unchanged. When that digest is stale, every entry is still invoked and every resulting artifact still passes ELF, symbol, architecture, and resource verification. Unchanged invocations can be ccache hits. A missing, disabled, evicted, or cold cache falls back to normal compilation.

`--no-ccache` or `TORCH_GGML_OPS_DISABLE_CCACHE=1` disables cache use. `--verify-reproducible` always bypasses ccache so its two builds are independent. `setup.py` also configures PyTorch's `PYTORCH_NVCC` hook, and the standard ccache compiler masquerade when available, for the extension translation units while preserving explicit user compiler settings.

## Reproducibility

Temporary directories, parallel build order, locale, timestamps, and default compiler-unit identifiers must not affect artifact bytes.

The generator therefore uses:
- stable locale settings.
- a stable source-date setting.
- source-prefix mapping.
- deterministic content-addressed wrapper source paths.
- ordered specifications.
- deterministic per-symbol CUIDs.
- staging directories outside installed output names.
- atomic final replacement.

`--verify-reproducible` builds the complete inventory twice in isolated directories and requires byte-for-byte equality in generator order. `--check` verifies that the installed artifact set and generated table correspond to the current build inputs.

Reproducibility is an artifact contract. It does not imply that semantically equivalent source arrangements, different compiler versions, or different kernel wrappers must produce identical ISA.

## Resource gates

A kernel specification may request a build-time production resource gate. A gated artifact must satisfy:

```text
private_segment_fixed_size = 0
vgpr_spill_count = 0
sgpr_spill_count = 0
uses_dynamic_stack = false
```

The gate is a compiler-resource assertion, not dispatch policy. It does not decide whether an entry is compatible, packaged, or selected. Compatibility entries may remain packaged without a production gate when their optimization log explicitly records that choice.

Optimization logs own the reason an entry is gated and the performance evidence for retaining it.

## Source-only repository and package policy

Generated HSACOs and the local build-input stamp are build outputs:
- `*.hsaco` and `.mmq-build-input` remain ignored by git.
- source distributions contain neither HSACOs nor the build-input stamp.
- source distributions include the generator, concrete-wrapper renderer, generated host table, and required headers.
- local wheel builds run the generator before extension compilation.
- generated artifacts are copied into the wheel build tree.
- stale artifacts in the wheel build tree are removed before packaging.

This keeps the repository and sdist source-only while allowing wheels to contain architecture-specific runtime artifacts.

## Runtime discovery and loading

The loader uses `dladdr` on an address in `_C.abi3.so` to locate the extension and then resolves `kernels/<architecture>` relative to it. Discovery is independent of the current working directory and Python import-path spelling.

Loading is lazy per `(device, kernel ID)`:
- Read the selected artifact into retained process memory.
- Call `hipModuleLoadData`.
- Resolve the expected symbol with `hipModuleGetFunction`.
- Cache retained bytes, module, and function.
- Launch on PyTorch's current stream with `hipModuleLaunchKernel`.

A mutex serializes first resolution. Cache hits do not reopen files. Modules and backing bytes remain owned for process lifetime to avoid device-context shutdown-order problems.

The loader does not use artifact loading as a tuning signal and does not fall back to a different kernel when loading or lookup fails.

## Dispatch boundary

Static dispatch may depend only on host-visible operator information that the owning optimization log allows, such as:
- target architecture.
- operator mode.
- quant type.
- exact logical geometry.
- public row count or padding.
- coarse host-visible routing buckets.

Dispatch must not depend on:
- a benchmark distribution name.
- online timing.
- environment-variable tuning.
- host reads of device offsets.
- compiler resource counts discovered at runtime.
- artifact filename ordering.

The concrete dispatch tables and thresholds are kernel behavior and are documented only in the four optimization logs linked at the top of this document.

## Failure model

Failures are explicit and identify the artifact path or symbol and HIP error where available. Expected hard failures include:
- missing artifact.
- unreadable artifact.
- module-load failure.
- missing exported symbol.
- unsupported architecture.
- unsupported operator mode, quant type, or geometry.
- build-time symbol, architecture, reproducibility, or resource-gate failure.

The runtime never silently substitutes a generic or embedded arithmetic implementation merely because a packaged artifact is unavailable.

## Validation contract

Bundle infrastructure changes require validation in four layers.

### Generator validation

- build-input cache invalidation.
- `--check`.
- two-pass byte reproducibility.
- exact expected artifact set.
- one expected public symbol per artifact.
- architecture metadata verification.
- all requested resource gates.

### Operator correctness

- complete project tests.
- exact packed-reference equality where required.
- independent numerical references retained by each operator family.
- unchanged public errors for unsupported paths.

### Packaging validation

- clean build from a tree with no generated HSACOs.
- wheel inspection for the generated inventory.
- sdist inspection for zero HSACOs.
- installed-wheel execution from outside the repository working directory.
- stale-output pruning.

### Performance validation

Performance belongs to the optimization logs, not to this design document. Any kernel-source, wrapper, inventory, or dispatch change must benchmark the exact packaged HSACO bytes through the production loader. Embedded fatbinary surrogates are not accepted performance artifacts.

Launcher overhead may be measured for regression safety, but arithmetic retuning should use warmed modules and kernel-focused timing when the optimization question concerns device behavior.

## Adding a kernel or architecture

A new entry should:
- Add or reuse a project-owned device body.
- Use the narrowest compatible generated wrapper ABI.
- Add one generator specification with a stable generic symbol.
- Add reviewed static selection in the owning host dispatcher.
- Document behavior, alternatives, and benchmark evidence in the owning optimization log.
- Add correctness coverage and any required resource gate.
- Run reproducibility, packaging, installed-wheel, and performance validation.

A new architecture should use a separate package directory and architecture-specific inventory. Shared source is allowed, but artifact identity, resource gates, and performance acceptance are architecture-specific.

## Optimization-record boundary

The bundle design should remain stable while kernel behavior evolves. The following information must stay out of this document and in the appropriate optimization log:
- exact kernel counts by family.
- selected tile sizes and K depths.
- quant-specific activation layouts.
- row-task thresholds and tail policies.
- model projection counts.
- benchmark artifacts and measured regressions.
- accepted and rejected tuning experiments.
- current retuning priorities.

This boundary keeps bundle maintenance generic and makes each kernel decision traceable to its workload, correctness contract, and measured behavior.
