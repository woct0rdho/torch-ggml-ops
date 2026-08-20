# GGTensile Fixed-Grouped MMQ Backward Q8_0 Experiment

## Purpose

Build, qualify, and optimize a repo-owned gfx1151 assembly implementation of the fixed-group DeepSeek Q8_0 gradient multiply. The work is research-only until an exact artifact beats the installed HIP production dispatch without weakening correctness, determinism, inspection, or shape guards.

The campaign optimizes the largest production margin first. Every coherent mechanism that is retained or rejected is recorded below with its artifact, control, correctness result, timing protocol, and conclusion.

## Exact production contract

There are eight fixed, non-routed groups. For each group `g`:

```text
dY[:, g, :] [tokens, 1024] @ W[g] [1024, 4096]
    -> dX[:, g, :] [tokens, 4096]
```

Production token counts are exactly `2048`, `8192`, and `32768` (B1, B4, and B16). All floating tensors are contiguous BF16. The layouts are:

```text
grad_output   [tokens, 8, 1024]
packed_weight [8, 1024, 4352] uint8
grad_input    [tokens, 8, 4096]
```

Each Q8_0 row contains 4096 values packed as 128 blocks of 34 bytes, hence 4352 bytes per row and 4,456,448 bytes per group. Rows of `grad_output` and `grad_input` are token-major with the group axis interleaved. This is not routed grouped ownership and must not use synthetic route metadata.

The fixed six-argument ABI is, in order:

```text
grad_output, packed_weight, grad_input,
tokens (u32), out_features (u32), bytes_per_group (u64)
```

The research launcher accepts only the three exact token counts, eight groups, `out_features=1024`, `in_features=4096`, exact dtypes/shapes, contiguous zero-offset storage, and one HIP device.

## Production control

The exact control is `launch_fixed_grouped_backward` in `csrc/mmq_bundle.cpp`. It dispatches:
- B1/B4: `grouped_bwd_fixed_q8_0_g8_k4096_mt256_nt64`
- B16: `grouped_bwd_tuned_fixed_q8_0_g8_k4096_mt192_nt64`

Historical HIP timings are approximately 6.237 / 25.427 / 97.204 ms for B1/B4/B16. These numbers orient the work but are not acceptance measurements; candidate and installed control must be measured together in the current process and environment.

## Reuse and ownership boundary

The arithmetic authority is the ordinary MMQ backward Q8_0 pipeline:
- typed Q8_0 contract and kernel spec in `mmq_bwd_spec.py`
- physical resources in `mmq_bwd_physical.py`
- packed reads/decode in `mmq_bwd_lowering_quant.py`
- BF16 WMMA accumulation and store conversion in `mmq_bwd_lowering.py`

The initial compute identity is the mature ordinary Q8_0 catalog selection for `(M,N,K)=(tokens,4096,1024)`: M256/N64/K32, four waves, two prefetched A halves, SIA5, padded single-buffer LDS, packed Q8 extraction, and raised store priority.

The fixed experiment owns a dedicated model, contract/spec, validation, physical wrapper, lowering, writer, and direct launcher. Its address adapter owns group-Z grid mapping, one packed-weight bank per group, eight-way token-major activation/output row strides, and exact-shape launch guards. The routed grouped backward shell is deliberately not reused.

## Initial lowering (E0)

Launch grid is `(tokens / 256, 4096 / 64, 8)` and the workgroup is `(32,4,1)`. Grid X is the logical M tile, grid Y is the N tile, and grid Z is the fixed group. The prologue offsets each scalar base pointer to group Z. The composed ordinary tile emitter then uses row strides of `8*1024*2` for `grad_output` and `8*4096*2` for `grad_input`; packed Q8_0 rows remain contiguous within a group.

## Qualification and measurement

Each candidate is built independently and inspected for exact metadata, ABI, workgroup, LDS, VGPR/SGPR declarations, no spills, expected WMMA/barrier counts, and no undeclared register use.

Correctness is checked against both the installed fixed HIP kernel and a BF16 dequantized reference. Qualification covers all three production token counts, full-row writes, repeated-output determinism, poisoned output replacement, and mutations of every group in `grad_output` and `packed_weight`. A candidate is rejected on non-finite output, untouched rows, cross-group contamination, or a material error regression relative to the installed control.

Timing uses rotating candidate/control order on one stream, synchronized HIP events, at least three warmups and nine measured repetitions, with median and sample spread reported. Setup, allocation, dequantization, and reference work are outside kernel timing. B1, B4, and B16 are always reported together after a mechanism survives focused qualification.

## Staged plan

- Land the strict fixed identity, physical/address adapter, six-argument writer, direct launcher, source/resource tests, and independent build.
- Qualify the composed M256/N64/K32 anchor at B1/B4/B16.
- Measure against the exact installed dispatch and attack the largest relative or absolute margin first.
- Test one mechanism at a time, retaining only reproducible wins that preserve the full contract.
- Package any final selected identities only after recursive review closes.

## Recursive final-review rule

When a provisional winner exists, rerun inspection, full qualification, and the rotating three-shape benchmark from a clean independent build. Then review the disassembly and timing breakdown for the new largest remaining cost and form one concrete follow-up hypothesis. If that hypothesis is technically coherent, test and document it, then restart this final review on the resulting winner. Stop only when the review produces no justified follow-up mechanism or all justified mechanisms have been measured and rejected.

## Experiment log

### E0: compose mature ordinary Q8_0 arithmetic (retained design)

Status: retained as the correctness and arithmetic anchor; rejected as the performance selection.

The fixed layout changes ownership and affine addresses but not one group's `(M,N,K)=(tokens,4096,1024)` arithmetic. Reusing the selected ordinary M256/N64/K32 decoder/WMMA pipeline avoids creating a second Q8_0 arithmetic authority. The fixed layer will alter only group bases, row strides, grid mapping, ABI metadata, validation, and launch behavior.

Independent artifacts `ggsol_188223d5da093f20`, `ggsol_685187411ef21738`, and `ggsol_80bcab953d5a9461` passed inspection with 231 VGPRs, 17 SGPRs, 5120-byte LDS, no spills/scratch, 32 static WMMAs, and two barriers. At every production shape the candidate was bitwise identical to the installed HIP output and the 16-token dequantized BF16 reference sample. Every destination element was finite and overwritten, repeat differences were zero, and all eight gradient plus all eight packed-weight mutations changed only their selected group.

Rotating 3-warmup/9-repeat medians were:

| Tokens | Candidate (ms) | HIP control (ms) | Candidate / HIP |
|---:|---:|---:|---:|
| 2048 | 8.483 | 6.163 | 1.376x |
| 8192 | 64.137 | 24.929 | 2.573x |
| 32768 | 583.365 | 92.962 | 6.275x |

The nonlinear degradation makes B16 the largest absolute and relative gap. The anchor launch varied M tiles on grid X and N tiles on grid Y, unlike the installed HIP traversal. The arithmetic reuse decision remains sound, but this grid order is not retained.

### E1: N-major workgroup traversal

Status: retained for all production shapes.

The installed kernel places N tiles on grid X and M tiles on grid Y. Since grid X is the fastest workgroup coordinate, that order completes all 64 N tiles for one activation tile before advancing M, increasing reuse of the same `grad_output` rows. E1 swaps X/Y into the ordinary emitter's expected `s2=M`, `s3=N` coordinates while preserving group Z and every arithmetic instruction. This directly targets E0's shape-dependent collapse without confounding the result with a tile or decoder change.

E1 artifacts `ggsol_8c3281ed779f2b01`, `ggsol_0a368afb69590dbf`, and `ggsol_54c9cd4e43d342a7` retained the E0 inspection resources and passed the same full correctness, determinism, poison, reference-sample, and 16-mutation qualification. Rotating medians were:

| Tokens | E1 (ms) | HIP control (ms) | E1 / HIP | E1 speedup |
|---:|---:|---:|---:|---:|
| 2048 | 5.138 | 6.161 | 0.834x | 16.6% |
| 8192 | 20.879 | 24.886 | 0.839x | 16.1% |
| 32768 | 79.125 | 92.534 | 0.855x | 14.5% |

B16 improved 7.37x relative to E0. N-major traversal is therefore part of the strict retained solution identity, not a launcher-only policy.

### E2: M128/N128 square tile

Status: provisionally retained for all production shapes.

B16 remains the largest absolute time. Under N-major traversal, each M tile is read by one workgroup per N tile. Changing M256/N64 to M128/N128 keeps 128 accumulator VGPRs and exact production divisibility while halving the number of N workgroups and the repeated activation traffic per M tile. It also changes packed decode geometry and workgroup count, so it will be measured as a separate exact identity rather than inferred from ordinary-MMQ results.

The B1/B4/B16 exact artifacts use 202 VGPRs, 17 SGPRs, 10240-byte LDS, no spills/scratch, 32 static WMMAs, and two barriers. Their candidate-versus-HIP screens were bitwise identical. Rotating medians were:

| Tokens | E2 (ms) | E1 (ms) | E2 gain | HIP (ms) | E2 / HIP |
|---:|---:|---:|---:|---:|---:|
| 2048 | 4.979 | 5.138 | 3.1% | 6.203 | 0.803x |
| 8192 | 20.067 | 20.879 | 3.9% | 24.855 | 0.807x |
| 32768 | 75.699 | 79.125 | 4.3% | 92.903 | 0.815x |

E2 wins at every shape and becomes the provisional universal geometry. Full mutation/reference qualification remains required after recursive review.

### E2a: M64/N256 wider tile feasibility

Status: rejected at the representability gate; not benchmarked.

M64/N256 would preserve 128 accumulators and halve N workgroups again, but it requires 16 N repeats per wave. The shared backward arithmetic contract and validator intentionally support only 2/4/8 N repeats. Broadening that ordinary writer boundary is not justified while proven Q8 decode mechanisms remain untested, so no artifact was emitted and no timing claim is made.

### E3: packed-VOPD Q8 extraction on E2

Status: rejected.

The ordinary Q8_0 campaign measured about a 3.6% packed-VOPD decode gain on an M128 body. E2 also has an M128 macro tile and two decoder rows, so applying only `q8_0_extraction=packed_vopd` is a supported, isolated way to reduce decode issue count on the largest B16 path.

Artifact `ggsol_bc591301326900ed` was bitwise identical to HIP and reduced static VALU issues from 608 to 594, but increased VGPRs from 202 to 204. Its B16 median was 76.204 ms versus E2's 75.699 ms, a 0.67% regression, and one sample rose to 80.102 ms. The lower issue count does not offset the physical cost here; packed-VOPD is not retained.

### E4: M64/N128 occupancy tradeoff

Status: rejected.

Reducing only M repeats from two to one lowers the derived VGPR declaration from 202 to 122 while retaining N128 and 10240-byte LDS. This may cross a useful occupancy boundary on gfx1151. It doubles workgroups and total packed decode work, so B16 timing will decide whether added occupancy outweighs duplicated B work.

Artifact `ggsol_623857b32aed1720` was bitwise identical and used 122 VGPRs, but its B16 median was 107.295 ms: 41.7% slower than E2 and 15.5% slower than HIP. The duplicated packed decode/workgroup cost dominates occupancy. M64/N128 is not retained.

### E5: paired-row clause store

Status: rejected.

B16 writes 2 GiB of BF16 output. The shared backward emitter has a qualified unbounded store path that converts two physical fragment rows, issues one `s_clause`, and stores the pair through two addresses. E5 applies that epilogue policy to E2 while preserving geometry, decode, and N-major ownership. The store schedule is part of the strict identity.

Artifact `ggsol_e5c776f8240f13ee` was bitwise identical and added eight static clauses without changing resources. Its B16 median was 76.949 ms, 1.65% slower than E2. The epilogue is not the limiting stage at this geometry, so clause pairs are not retained.

### E6: DepthU64 on E2

Status: retained final selection for all production shapes.

Canonical DepthU64/PGR2/PLR1/SIA4 halves reduction-loop iterations, synchronization, and scalar loop work. It raises decoder rows from two to four and LDS from 10240 to 18432 bytes. The ordinary campaign retained DepthU64 only for a different Q-B shape, but the E2 N128 geometry changes the balance enough to justify one isolated B16 screen.

E6 artifacts use 216 VGPRs, 17 SGPRs, 18432-byte LDS, no spills/scratch, 64 static WMMAs, and two static barriers. Candidate-versus-HIP screens were bitwise identical. Rotating medians were:

| Tokens | E6 (ms) | E2 (ms) | E6 gain | HIP (ms) | E6 / HIP |
|---:|---:|---:|---:|---:|---:|
| 2048 | 4.624 | 4.979 | 7.1% | 6.210 | 0.745x |
| 8192 | 19.495 | 20.067 | 2.8% | 25.173 | 0.774x |
| 32768 | 73.510 | 75.699 | 2.9% | 92.632 | 0.794x |

DepthU64 wins at all three shapes and replaces E2 as the provisional universal selection. Full qualification is recorded below after recursive review.

### E7: packed-VOPD extraction on E6

Status: rejected.

E6 has four decoder rows, twice E3's decode work. Packed-VOPD may therefore save enough issue slots to offset its extra decode VGPRs even though it lost on the DepthU32 body. E7 changes only Q8 extraction on E6 and screens B16 first.

Artifact `ggsol_28f3a30c774f762b` was bitwise identical, but increased the declaration to 218 VGPRs. Its B16 median was 74.014 ms versus E6's 73.510 ms, a 0.69% regression. The extra decode packing does not pay on this body.

### E8: next packed-weight prefetch on E6

Status: rejected at the representability gate; not benchmarked.

E8 enables `CurrentAndNextTile` packed-weight prefetch while retaining E6's DepthU64/SIA4 geometry. It is the last direct overlap hypothesis for the remaining reduction-loop cost; the ordinary campaign's rejected next-prefetch results are relevant prior evidence but do not substitute for this fixed N128/DepthU64 measurement.

The ordinary Q8 decoder contract rejects this combination with `solution.depthu64.prefetchpacked.quant`: DepthU64 next-packed-tile prefetch is not implemented by the selected decoder. No artifact was emitted and no timing claim is made. Widening that decoder contract would violate the reuse boundary; the recursive review therefore has no remaining coherent overlap mechanism to test without starting a separate arithmetic implementation.

## Final qualification and selection

The selected solution is exposed as `FixedBackwardSolution.selected_q8_0()`: N-major M128/N128/DepthU64, PGR2, PLR1, SIA4, pad8 single-buffer LDS, packed Q8 extraction, raised-priority element-serial stores, and fixed group Z ownership. Fresh exact artifacts are:
- B1: `ggsol_e3f23cb5c4ad7aae`
- B4: `ggsol_8fb858b844ca8109`
- B16: `ggsol_d0bc5b94d26fe2b3`

Each independently inspected as 216 VGPRs, 17 SGPRs, 18432-byte LDS, 40-byte kernarg segment, 128 threads, 64 static WMMAs, two static barriers, no private segment, no spills, no scratch, and no undeclared register use.

The clean final run used seed 20260822, five warmups, and 25 rotating samples per candidate/control path:

| Tokens | Selected (ms) | HIP (ms) | Selected / HIP | Latency gain | TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 2048 | 4.847 | 6.289 | 0.771x | 22.9% | 28.35 |
| 8192 | 18.989 | 24.969 | 0.761x | 23.9% | 28.95 |
| 32768 | 75.021 | 93.240 | 0.805x | 19.5% | 29.31 |

For every shape, the complete output was bitwise identical to the exact installed HIP control and the 16-token dequantized BF16 reference sample. All values were finite, poison was fully replaced, repeat differences were zero, and each of eight gradient plus eight packed-weight mutations changed only its selected group.

The recursive final review rebuilt and reinspected all selected artifacts and retested the largest B16 cost. Packed-VOPD, lower-VGPR M64 ownership, clause stores, and next packed prefetch were respectively measured losses or rejected by the shared decoder contract. M64/N256 would require broadening the ordinary writer's supported geometry. No further mechanism is justified within this campaign's arithmetic-reuse boundary, so the review is closed.
