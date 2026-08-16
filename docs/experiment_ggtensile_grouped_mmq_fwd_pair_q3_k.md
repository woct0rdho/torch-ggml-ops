# GGTensile grouped MMQ forward paired Q3_K experiment

## Scope

This record covers isolated GGTensile research kernels for Qwen's paired routed Q3_K gate and up projections. The exact production contract is:

- 256 physical experts;
- output features `N = 512` and reduction features `K = 2048` for both projections;
- aggregate routed rows `R = 16,384`, `65,536`, or `262,144`;
- two authoritative packed Q3_K banks with shape `[256,512,880]`;
- one fixed Q8_1 `F32_D4` activation workspace with shape `[16,R,144]` shared by both projections;
- two independent BF16 destinations with shape `[R,512]`;
- device-resident int64 expert IDs and cumulative int32 route offsets;
- at most 256 route entries, a final valid offset equal to `R`, and inert invalid routes.

A paired candidate must compute both projections through one workgroup arithmetic dataflow. Projection-indexed workgroups, adjacent single-projection launches, and sequential full-projection phases that reload activation data are controls, not fusion.

Public selectors, generated bundles, extension registration, packaging, and HIP fallback behavior remain outside this research campaign. Promotion requires a separate integration campaign after the research identity passes every gate below.

## Baseline and priority

The public `grouped_mmq_pair` path quantizes once and shares one activation allocation, then launches the existing single-projection grouped body once per packed bank and destination. At the prior fitted-route medoid, public complete-call time was `6.0770 ms`, `16.5897 ms`, and `58.2301 ms` for B1, B4, and B16. The corresponding BF16 AITER times were `10.8056 ms`, `25.1215 ms`, and `55.9767 ms`. B16 therefore has the only comparator deficit, while every shape still pays duplicate activation reads in the two packed bodies.

For one 64-row by 64-column output tile over full `K = 2048`, the Q8_1 workspace contributes `64 * 8 * 144 * 2 = 147,456` activation bytes across eight Q3_K blocks and two K128 halves per block. Each projection contributes `64 * 110 = 7,040` packed bytes per block before transaction alignment. A fused body can load each activation half once while retaining two independent accumulator banks. Q3 high-mask reconstruction, signed six-bit scales, both WMMA streams, both FP32 corrections, and both epilogues remain mandatory.

The largest performance gap is B16, so advancement starts there after bounded correctness. B1 and B4 transfer only after the B16 candidate beats its adjacent public and ownership-matched controls by more than two percent.

## Candidate Q1: selected-half K128 serial routes

Q1 uses one 128-thread wave32 workgroup to own 64 routed rows and 64 output columns in both projections. It keeps two independent 32-VGPR FP32 sum banks and one 9,216-byte activation image. For each K128 half it:

- decodes the selected Q3_K half from the first packed bank into one reusable 10,240-byte LDS weight image;
- stages the matching activation plane once;
- computes and corrects the first projection;
- overwrites only the weight image from the second packed bank;
- computes and corrects the second projection while the activation image remains resident.

Two producer lanes own each output row. They reconstruct the selected low two-bit payload with the matching high-mask bytes, decode the eight signed six-bit group scales with the FP16 block factor, and write one 128-byte int8 payload plus eight FP32 scales into a 160-byte padded LDS row. The compute order remains eight signed integer WMMAs per K128 half and M fragment, followed by the qualified FP32 correction and BF16 RNE store order.

Q1 uses the existing 80-byte cumulative-route paired ABI. It is a semantic and serial-ownership control because large or imbalanced routes remain resident in one workgroup per output tile.

## Candidate Q2: device 64-row tasks

Q2 preserves Q1's arithmetic, register plan, LDS plan, and output ownership. It consumes the installed device-built 64-row task arrays through the existing 96-byte paired row-task ABI. Grid Y is the bounded capacity `ceil(R/64) + route_entries`; every workgroup exits when its task index is not below the device task count. No timed path reads route values or task counts on the host.

The adjacent control is one installed task-setup launch followed by two installed Q3_K row-task bodies on the same activation workspace. The public pair is also required in every benchmark rotation.

## Qualification gates

Correctness precedes timing:

- Strict typed problem, solution, enum, serialization, validation, physical-plan, ABI, and writer tests.
- Byte-identical regeneration of both Q3_K candidates and byte-identical preservation of both existing IQ2_S paired sources.
- gfx1151 code-object v5, wave32, zero private storage, zero spills, no scratch instructions, calls, or dynamic stack, with emitted resources equal to the physical plan.
- Bitwise agreement for both destinations against adjacent installed controls and public dispatch on first, odd, even, last, repeated, sparse, skewed, boundary, and expert-255 routes.
- Active mutations of either packed bank affect only its destination; activation mutations affect both; inactive experts are inert; malformed routes preserve sentinels.
- Bounded independent dequantized references for both projections rather than a full model-sized dequantization.
- Complete-call timing includes fixed HIP Q8_1 quantization, allocated workspace, device task setup when applicable, and both projections.
- The fitted prior uses 512 deterministic draws, five weighted medoids, three warmups, nine order-rotated repeats, and reversed-order 25-repeat confirmation after advancement.
- Synthetic uniform, skewed, sparse, and boundary controls must preserve correctness and the retained performance direction at all three production row counts.

The inherited final-review rule applies. A fresh review must reread this record, the paired IQ2_S and dense Q3_K records, the retained source and disassembly, installed HIP controls, target ISA behavior, generated artifacts, benchmark evidence, and failed candidates. Any actionable in-contract finding is implemented and qualified before the review repeats; completion is valid only when a fresh pass finds none.

## Experiment log

### Initial implementation and artifact gate

The first implementation adds distinct Q3_K serial-route and device-row-task identities without changing the existing IQ2_S identities. Both Q3_K artifacts derive 148 VGPRs, 44 SGPRs, and 19,456 bytes fixed LDS. They contain 128 static integer WMMAs and eight barriers with zero private storage and zero register spills. The serial ABI is 80 bytes and the row-task ABI is 96 bytes.

The 35-row boundary probe used real `blk.0.ffn_gate_exps.weight` and `blk.0.ffn_up_exps.weight` tensors. Q1 and Q2 matched each other, their ownership-matched installed controls, and public `grouped_mmq_pair` bit-for-bit for both projections on routes containing experts 0, 7, 12, and 255. The device task builder produced four bounded tasks. Both outputs were finite.

The first direct-control probe incorrectly allocated 29,952 dynamic LDS bytes by counting 32 activation payload integers per row. The installed HIP tile owns 36 integers per row after Q8_1 metadata, so its authoritative J64 allocation is 30,976 bytes. That invalid setup differed in 356 BF16 values per projection. Correcting only the control launch allocation restored bitwise agreement; the GGTensile artifacts continue to launch with zero dynamic LDS because their 19,456 bytes are statically declared.

Timing, mutation, malformed-route, bounded-reference, production-shape deterministic-build, and synthetic-route gates remain pending.
