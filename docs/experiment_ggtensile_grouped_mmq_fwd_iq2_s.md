# GGTensile grouped MMQ forward IQ2_S experiment

## Scope

This record covers isolated research kernels for the non-paired Qwen routed-down
projection with packed `IQ2_S` weights. The exact production contract is:

- 256 physical experts;
- output features `N = 2048`;
- reduction features `K = 512`;
- aggregate routed rows `R = 16,384`, `65,536`, or `262,144`;
- device-resident int64 expert IDs and cumulative int32 route offsets;
- at most 256 route entries, with the final valid offset equal to `R`;
- fixed HIP `Q8_1` `F32_D4` activation quantization and its allocated workspace;
- BF16 output.

Paired projection kernels, public dispatch, generated bundle tables, extension
registration, packaging, and the HIP fallback are out of scope.

## Format contract

One `IQ2_S` block represents 256 values in 82 bytes: one FP16 block scale,
64 `qs` bytes, eight `qh` bytes, and eight packed scale bytes. The first 32
`qs` bytes and two bits from `qh` select one of 1,024 eight-byte codebook
entries. The upper 32 `qs` bytes provide one sign bit per decoded value. Two
adjacent eight-value groups share a four-bit scale, with effective factor
`d * (scale + 0.5) / 4`.

The required `K = 512` row therefore contains two packed blocks and occupies
164 bytes. The grouped launch grid is `(N / 64, G, 1)` with 128 workitems per
workgroup for the first candidate family.

## Method

Correctness precedes timing. Candidates must pass independent bounded
references, installed-kernel comparison, sequential-route controls, route
mutation controls, workspace and input mutation checks, and invalid-route
guards before performance can be retained.

Performance uses the fitted Qwen route prior: 512 draws reduced to five
weighted medoids, three warmups, and nine alternating repeats. Competitive
candidates receive a reversed-order 25-repeat confirmation. Every measurement
has an adjacent installed HIP control, and throttled or noisy runs are
discarded. Complete-call timings include fixed activation quantization and the
allocated workspace.

Retained artifacts must use gfx1151 code-object v5 and wave32, report no
private storage or spills, contain no scratch instructions, calls, or dynamic
stack, and rebuild byte-identically.

The final review repeats correctness matrices, complete-call confirmation,
deterministic rebuild and resource inspection, focused and broad tests,
changed-file hooks, and repository-hygiene checks. A candidate is not final
until that review is complete.

## Initial baseline and priority

Historical complete-body profiling identifies `R = 262,144` as the first
target. The installed grouped HIP `IQ2_S` J64 body measured 25.703 ms, fixed
quantization measured 1.849 ms, and the adjacent AITER control measured
24.094 ms. The installed body uses 232 VGPRs and 30,976 bytes of LDS. Exact
fresh complete-call baselines for all three production shapes will be recorded
with the first runnable research candidate.

The first implementation will model a Tensile-style 128-row by 64-column
workgroup with full decoded-weight LDS reuse. Its producer distributes one
weight row and one 128-value K half to each workitem, so all four waves
cooperate on the 64-row weight tile without duplicate row decode. Tail rows
remain route-bounds masked. This establishes the typed problem, solution,
physical plan, inspection, runtime, and correctness path before schedule and
tile-family screening.

## Experiment log

### 2026-04-14: inventory and implementation plan

- Confirmed the three non-paired production shapes and their fixed grouped ABI.
- Confirmed that current research support has no `IQ2_S` problem or lowering.
- Selected the B16 shape as the first optimization target because it has the
  largest absolute body-time opportunity.
- Chose distributed full-weight decode into LDS as the first candidate, using
  the existing signed-int8 WMMA and `F32_D4` activation contracts.

No kernel candidate has yet been retained or rejected.

### 2026-04-14: typed J64 distributed full-weight foundation

- Added the isolated `IQ2_S` problem and solution identity, exact production
  validation, packed-format semantics, compact J64 LDS layout, deterministic
  register plan, codebook emission, assembly lowering, artifact inspection,
  research runtime controls, and focused tests.
- The physical tile uses 64 activation rows, 64 output-feature rows, two
  128-value decoded payload halves, and eight FP32 group factors per half.
  Its fixed resources are 116 VGPRs, 40 SGPRs, and 30,720 bytes of LDS.
- The producer maps `wave * 16 + lane[3:0]` to one output-feature row and
  `lane[4]` to one K half. It therefore decodes all 64 rows exactly once across
  the four waves rather than duplicating rows between waves.
- The first live launch exposed two ownership errors: scale addresses were
  based on payload lanes instead of WMMA output elements, and the second
  activation stage reused `v0` after that register had become an accumulator.
  Both were corrected by using the established wave/lane ownership formulas.
- A temporary constant-codebook launch proved that embedded codebook loads
  were not the source of the initial memory fault. The linked local table uses
  the same read-only code-object addressing model as the installed HIP kernel.
- The first completed output was globally sign-inverted because gfx11
  `v_perm_b32` byte selectors chose the opposite listed data operand from the
  initial assumption. Swapping the positive and negative operands restored the
  intended per-byte sign mapping.
- A four-route, 35-row bounded launch then matched the installed mixed J64/J32
  control bitwise for all 71,680 BF16 output elements. The artifact reports
  wave32, code-object v5, four barriers, 64 static WMMAs, no private storage,
  and no SGPR or VGPR spills.

This foundation is retained. Production route matrices and timing remain to be
qualified before selecting or rejecting the J64 schedule.
