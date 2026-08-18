# GGTensile Grouped MMQ Backward Q2_K Experiment

## Purpose

Implement and optimize isolated gfx1151 grouped Q2_K backward kernels for the DeepSeek routed down projection. Public dispatch, generated bundle tables, extension registration, packaging, and HIP fallback remain unchanged pending a separate integration review.

The promotion objective is the fitted DeepSeek learned/hash routing-prior weighted sum of per-medoid median complete-call latency. Learned and hash components retain the declared reporting weights `40/43` and `3/43`. Uniform, skewed, sparse-ID, and boundary routes remain correctness and diagnostic controls rather than post hoc ranking vetoes.

## Exact Contract

For routed GEMM `g`:

```text
dY_g[M_g,4096] x W_g[4096,2048] -> dX_g[M_g,2048]
```

The exact aggregate-row keys are `R={12288,49152,196608}`. The physical Q2_K expert bank is `[256,4096,672]`: each 256-value block occupies 84 bytes, each packed row contains eight blocks, and each expert occupies 2,752,512 bytes. Inputs and outputs are contiguous BF16, accumulation is FP32 WMMA V1, and output conversion is BF16 RNE.

The grouped ABI remains the 56-byte routed backward research ABI: `grad_output`, `packed_weight`, `grad_input`, `expert_indices`, `expert_offsets`, `num_experts`, `rows`, and `bytes_per_expert`. Route count and optional split ownership are launch geometry. No host route inspection, dense shadow, prepared bank, atomics, reduction workspace, or companion setup kernel is in scope.

The authoritative packed control is `~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf`, tensor `blk.0.ffn_down_exps.weight` (`Q2_K`, `[256,4096,672]`). The independent oracle dequantizes only selected routed experts to BF16 before matmul.

## Arithmetic Boundary

Q2_K cannot reuse Q4_K/Q5_K arithmetic by renaming the format. A block stores sixteen scale/minimum bytes, 64 two-bit payload bytes, and FP16 `d`/`dmin`. Each decoder lane owns one aligned 16-value group, shares its payload shift and scale/minimum pair across those values, converts decoded weights to BF16 in LDS, and then reuses the established FP32-WMMA and BF16-RNE leaf.

The installed HIP baseline selects M64/N64/U1 below 128 rows per route, M128/N64/U2 from 128 through 511, and M128/N64/U1 above that. This is evidence for initial geometry and ownership controls, not assumed GGTensile ranking. Q2-specific packed reads, metadata lifetimes, LDS layout, schedule, and route splitting must be measured independently.

## Qualification

Correctness covers full exact rows, non-aligned route tails, first/non-first routes, sparse and repeated expert IDs, deterministic reruns, gradient and route mutation, active/inactive expert-weight mutation, invalid experts, malformed first/final offsets, and untouched sentinels. Candidate versus packed HIP must be BF16 bit-exact; the independently dequantized reference must remain finite with NRMSE below `0.01`.

Inspection requires gfx1151, wave32, code object v5, exact 56-byte metadata, bounded VGPR/SGPR indices, zero private bytes and spills, no scratch/calls/dynamic stack, and derived static WMMA/barrier counts. Independent generate/build/inspect roots must produce byte-identical source and HSACO.

Timing uses warmed rotating GPU events and includes output allocation in both HIP and candidate complete-call paths. Search uses five fitted medoids from each DeepSeek learned/hash component. Retained changes receive disjoint confirmation with 5 warmups, 25 repeats, and reversed rotating order. TFLOPS is `2 * R * 2048 * 4096 / (latency_ms * 1e9)`.

## Planned Search

- Add strict Q2_K backward identity and exact DeepSeek shapes without changing retained Q4_K/Q5_K source hashes or the grouped ABI.
- Implement the dedicated Q2_K packed reader and decoded-B LDS arithmetic, then qualify a minimal M128/N128 pilot against packed HIP and the independent oracle.
- Establish M64/N64, M128/N64, and M128/N128 controls with single-LDS and bounded pipeline schedules.
- Screen Q2 payload/metadata extraction, LDS padding/swizzle, route tails, and split factors only where emitted ISA or ownership changes.
- Confirm per-key winners on disjoint learned/hash medoids, qualify full rows, rebuild independently, and report HIP/GGTensile TFLOPS and speedup.

## Completion Record

This section is updated after every coherent implementation or failed experiment.

### Campaign opened

The completed grouped Q4_K/Q5_K shell, installed DeepSeek Q2_K backward bodies, grouped Q2_K forward lowerer, packed model tensor, and fitted learned/hash prior were audited. The exact backward problem is `(R,2048,4096)` for `R={12288,49152,196608}` with expert stride 2,752,512 bytes. The first coherent change will add a strict Q2 identity, dynamic grouped row-stride handling, generalized benchmark dimensions/prior selection, and a dedicated width-16 two-bit decoder while preserving Q4/Q5 generated source.

### Strict Q2 identity and decoded-LDS pilot

The backward-only format registry and grouped contract now admit Q2_K while ordinary forward remains unchanged. Q2 grouped identities require only the three exact DeepSeek shapes, emit a distinct symbol/hash, validate the `[256,4096,672]` packed tensor, and guard the full 2,752,512-byte expert stride. Grouped row-byte masking and the benchmark now derive dimensions from the key. DeepSeek timing combines ten learned/hash medoids with the declared `40/43` and `3/43` component weights.

The dedicated decoder maps each lane to one aligned 16-value group, loads one 128-bit payload segment plus its scale/minimum byte and FP16 `d`/`dmin`, performs the shared 2-bit shift, and writes BF16-RNE decoded values to the existing LDS/FP32-WMMA leaf. The M128/N128 SIA2 pilot passes the 625-row boundary matrix bit-for-bit against packed HIP, including deterministic, gradient, route, active/inactive weight, malformed-route, and sentinel controls. Independent-reference NRMSE is `3.78e-5`. The current artifact uses 190 VGPRs, 35 SGPRs, 8 KiB LDS, 32 static WMMAs, and two barriers with zero private bytes or spills.

The first M128/N64 double-LDS artifact failed packed HIP, determinism, mutations, and the independent oracle. The Q2 one-row metadata allocation exposed an existing pipeline lifetime requirement: the shared decoded-B pipeline uses a second scale-adjacent register for LDS read addressing while next-tile metadata is pending. Reserving that register removes the alias with the swizzled write-address bank. Rebuilt B1/B4/B16 double-LDS artifacts are exact and deterministic at 139 VGPRs; no wait or arithmetic waiver was used. All initial SIA5 geometry controls also pass. Current serial resources are 87 VGPR for M64/N64, 135 for M128/N64, and 206 for M128/N128, all at 35 SGPR with zero private bytes or spills.

Regenerated retained Q4 and Q5 B1 sources remain byte-identical.

### First fitted-prior geometry screen

Nine-repeat search-bank results combine the learned/hash components with their declared reporting weights:

| Candidate | Key | Weighted HIP / candidate ms | Speedup vs HIP | Decision |
| --- | --- | ---: | ---: | --- |
| Pilot M128/N128 SIA2 | B1 | `21.5154 / 27.3304` | `0.7872x` | Correctness anchor; rejected by timing |
| SIA5 M64/N64 | B1 | `22.4126 / 15.3135` | `1.4636x` | B1 parent |
| SIA5 M128/N64 | B1 | `22.0206 / 18.5717` | `1.1857x` | Loses M64 |
| SIA5 M128/N128 | B1 | `22.0305 / 20.4270` | `1.0785x` | Loses M64 |
| Double-LDS M128/N64 | B1 | `21.8743 / 19.4817` | `1.1228x` | Rejected by timing |
| SIA5 M128/N64 | B4 | `50.5124 / 37.4635` | `1.3483x` | Provisional B4 parent |
| SIA5 M128/N128 | B4 | `49.6191 / 37.7143` | `1.3157x` | Direct bracket required |
| Double-LDS M128/N64 | B4 | `50.3349 / 39.9505` | `1.2599x` | Rejected by timing |
| SIA5 M128/N64 | B16 | `190.4396 / 161.5854` | `1.1786x` | Rejected by N128 |
| SIA5 M128/N128 | B16 | `191.3111 / 132.1376` | `1.4478x` | B16 parent |
| Double-LDS M128/N64 | B16 | `190.9867 / 162.6025` | `1.1746x` | Rejected by timing |

The result differs from the installed HIP geometry because the assembly N128 body is 206 VGPR rather than the historical 255-VGPR cliff. B1 route tails still favor the 87-VGPR M64 body; B4 is within one percent between N64 and N128 and requires same-process resolution; B16 gains materially from N128 packed reuse. Double-LDS is closed unless a later changed mechanism removes work rather than only rescheduling it.

### Layout, schedule, and first tail controls

Every control remains packed-HIP exact. Eight-byte row padding is essential: plain LDS regresses the B1 M64 body from `15.3135` to `27.5238 ms` and the B4 M128/N64 body from `37.4635` to `44.3643 ms`. Swizzle4/swizzle8 regress B1 to `17.5561/17.6736 ms` and do not beat padding at B4. SIA2 plus plain LDS loses at `27.5723 ms`; SIA5/PGR1 loses at `15.5173 ms`. Padded SIA4 and SIA5 remain close at B1 (`15.1463/15.3135 ms`) and B4 (`37.4354/37.4635 ms`), requiring same-process brackets.

Existing `Mixed128_64` tails, which select M64 only through 64 rows, improve B4 modestly to `37.2586 ms`. The installed Q2 body uses M64 below 128 rows, providing a changed and format-specific premise for one `<128` tail-policy control. The first threshold-only artifact failed correctness because the established M64 branch emits one tile by construction; this was rejected before timing. A typed Q2-only policy with a proper M64 route loop then passed the complete mutation matrix, but regressed to `39.9595 ms` (`1.2283x` HIP) versus `37.2586 ms` for the <=64 policy. Rows 65-127 prefer one masked 135-VGPR M128 tile over two M64 tiles. The failed policy was removed, and no broader threshold is opened.

### B16 layout and ownership controls

The padded N128 body is ownership-bound on the skewed B16 prior. SIA5 serial is `132.1376 ms`; split2/4/8/16 improve monotonically to `128.827/128.088/127.692/127.524 ms`, with split16 at `1.5000x` HIP. Plain LDS collapses to `170.905 ms`, swizzle8 regresses to `134.621 ms`, and double-LDS reaches only `133.517 ms`. Padded SIA4 serial is slightly ahead of SIA5 at `131.487` versus `132.138 ms`.

Split32 passes all controls and improves SIA5 to `125.3000 ms` (`1.5186x` HIP), unlike the rejected Q5 endpoint. SIA4 split16 is `126.7151 ms` versus SIA5 split16 at `127.5237 ms`, but SIA4 split32 regresses to `126.0449 ms` versus SIA5 at `125.3000 ms`. The dominant learned medoid has routes up to 50 M128 tiles, so split64 was tested as the sole endpoint that removes the last two-tile loop. It is exact but a 15-repeat same-process bracket favors split32 at `126.1163` versus `126.4909 ms` (`0.30%`); split32 wins the dominant learned profile and every hash medoid. Split64 is removed and SplitRoutes32 is retained as the typed maximum.

The existing <=64 mixed tail materially changes B4 N128: it reaches `37.1417 ms` versus N64 mixed at `37.2586 ms`, while raising the minimum medoid speedup from `1.1639x` to `1.2277x`. At B16, mixed tails regress split32 from `125.3000` to `126.2204 ms`; the B16 parent remains masked.

Fifteen-repeat same-process brackets lock the macro parents. B1 padded M64/N64 SIA4 is `15.2892 ms` versus SIA5 at `15.4012 ms` (`0.73%`). B4 N64 mixed is `36.9186 ms` versus N128 mixed at `38.6898 ms` (`4.58%`), and N64 mixed SIA5 is `37.3181 ms` versus pure N64 SIA4 at `37.8544 ms` (`1.44%`). B16 SIA5 split32 is `126.9932 ms` versus SIA4 at `127.5897 ms` (`0.47%`). Geometry, tail ownership, SIA, LDS layout, and route splitting are therefore closed around these three parents.

### Q2 decode scheduling

A typed Q2 `DependencyBatch4` schedule reuses four dead `valu_b` register pairs and adds no resources. All three controls are exact. Same-process brackets show B1 improving from `15.1730` to `14.8558 ms` (`2.14%`) and B4 from `37.2895` to `35.0546 ms` (`6.38%`). B16 instead regresses from `127.0026` to `128.8607 ms` (`1.46%`) and loses every medoid, so B16 remains serial. A temporary dependency-width-two endpoint was also exact but lost B16 serial in a bracket (`126.3994` versus `127.0545 ms`, `0.51%`) and was removed. Final decoder choices are batch4 at B1/B4 and serial at B16; no other decode arithmetic is opened.

### Final confirmation

The final artifacts use five warmups, 25 repeats, reversed rotating order, and disjoint confirmation-bank learned/hash medoids. Timings are weighted by the declared `40/43` learned and `3/43` hash reporting weights; both HIP and candidate complete-call paths include output allocation.

| Key | Final identity | Weighted HIP / GGTensile ms | HIP / GGTensile TFLOPS | Speedup vs HIP |
| --- | --- | ---: | ---: | ---: |
| B1 (`R=12288`) | SIA4 M64/N64, padded, serial routes, Q2 batch4 | `22.1611 / 14.9114` | `9.3027 / 13.8256` | `1.4862x` |
| B4 (`R=49152`) | SIA5 M128/N64, padded, <=64 M64 mixed tail, serial routes, Q2 batch4 | `51.1721 / 35.4964` | `16.1149 / 23.2315` | `1.4416x` |
| B16 (`R=196608`) | SIA5 M128/N128, padded, masked tail, SplitRoutes32, serial Q2 decode | `191.5512 / 125.8210` | `17.2201 / 26.2161` | `1.5224x` |

Full-row correctness (`--correctness-rows 0`) is packed-HIP bit-exact at every final key: 12,288, 49,152, and 196,608 rows. Deterministic reruns, gradient mutation, route mutation, and active-weight mutation are exact with zero tail writes. Independent BF16-oracle NRMSE is `8.55e-6`, `5.60e-5`, and `9.81e-5` respectively; all remain below `0.01`.

Final inspected resources are B1 `87 VGPR / 35 SGPR / 5 KiB LDS`, B4 `135 / 35 / 5 KiB`, and B16 `206 / 35 / 10 KiB`, with zero private bytes and zero VGPR/SGPR spills. Two disjoint generate/build/inspect roots produce byte-identical artifacts.

The Q2 research route remains isolated: production dispatch, generated bundles, registration, packaging, extension integration, and HIP fallback were not changed. The final repository gate passes `pytest -q tests` with `718 passed` (14 external PyTorch Python 3.14 deprecation warnings) and `pre-commit run --all-files` with pyupgrade, Ruff check, Ruff format, and ty all passing.
