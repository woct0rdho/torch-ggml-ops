# GGTensile assembly-writer refactor plan

## Objective

Make the MMQ forward and backward assembly writers easier to maintain and extend without merging their direction-specific algorithms. Share direction-neutral assembly infrastructure and quant-format facts, align the writers' outer structure, and preserve every generated production kernel exactly unless a source change is explicitly justified and revalidated.

## Design boundary

Keep these direction-specific:
- forward Q8_1 workspace consumption and integer-WMMA correction arithmetic.
- backward BF16 activation reads and BF16-WMMA data path.
- packed-weight address mapping.
- forward integer expansion and backward BF16 dequantization.
- LDS layouts and pipeline schedules.
- register-allocation policy.
- launch ABI, work-group mapping, and output-fragment mapping.
- `ForwardSolution` and `BackwardSolution` schemas and exact validation rules.

Share only mechanisms whose semantics and emitted instruction text are identical.

## Refactor steps

- Freeze generated assembly.
  - Generate every selected production forward and backward inventory key.
  - Generate every additional valid inventory-entry/catalog-solution combination from the pre-refactor committed tree.
  - Save each complete source and SHA-256 digest outside the repository under `/home/wd/tmp/torch-ggml-ops/`.
  - The selected production set is 18 forward keys and 50 backward keys. The broader catalog-valid set is compared independently.

- Add a direction-neutral assembly-writer module.
  - Move the common line emitter into `tools/ggtensile/kernel_writer_assembly.py`.
  - Share atomic source writing and digest calculation.
  - Share ROCISA initialization with guaranteed working-directory restoration.
  - Share identical ISA primitives: BF16 RNE preparation, multiply-or-shift address scaling, 64-bit pointer addition, three pointer-kernarg loads, and kernel trailer emission.
  - Keep the backward deferred-zero/VOPD combiner as a backward-specific subclass.

- Centralize quant-format geometry.
  - Add direction-neutral immutable format descriptions for Q3_K, Q4_K, Q5_K, Q6_K, and Q8_0.
  - Give the Q8_1 `F16_D4S4` activation block its own named constant even though its byte size equals a Q4_K block.
  - Replace duplicated block-size/value-count literals in GGTensile model, campaign, runtime, and writer code without changing serialized solution values or hashes.
  - Keep direction-specific load and decode instruction emitters separate.

- Align writer structure without merging algorithms.
  - Both writers continue to expose the same `write()` and `source()` interface.
  - Both use the shared prologue primitives and trailer.
  - Forward keeps its direct, wave-reuse, and HIP-staged bodies.
  - Backward keeps its schedule and quant dispatch structure.
  - Do not introduce a generic MMQ main loop or a shared solution base class.

- Enforce complete executable-line coverage.
  - Extend the existing tracing support to the shared assembly-writer module.
  - Continue requiring complete method-body line coverage for the forward and backward writers.
  - Require complete function and method-body line coverage for the shared module.
  - Add targeted shared-helper coverage only for branches not naturally exercised by the direction suites.

- Verify generated output and resources.
  - Compare all 68 selected production sources byte-for-byte with the frozen baseline.
  - Compare every catalog-valid forward and backward source against a detached pre-refactor worktree.
  - Run the existing retained Q4_K and Q5_K source/resource comparison scripts.
  - If any generated source differs, document the exact changed instructions/directives and run correctness, inspection, and performance confirmation before retaining it.
  - The intended result of this refactor is zero generated assembly differences.

- Repository validation.
  - Run focused GGTensile tests and the complete test suite.
  - Run Ruff formatting/linting, `ty check`, pre-commit, bundle-current checks, and `git diff --check`.

## Acceptance criteria

- Forward, backward, and shared writer executable method/function lines have complete coverage.
- All 18 forward and 50 backward selected production assembly sources are byte-identical to the pre-refactor baseline.
- All 87 forward and 292 backward catalog-valid assembly sources are byte-identical to the detached pre-refactor tree.
- Retained Q4_K and Q5_K instruction bodies and resource envelopes remain unchanged.
- Solution mappings, hashes, kernel names, ABIs, launch geometry, and strict validation behavior remain unchanged.
- No compatibility aliases, forwarding modules, duplicate writer implementations, or generic direction-conditional main loop are introduced.
