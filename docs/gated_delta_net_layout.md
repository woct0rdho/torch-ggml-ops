# GatedDeltaNet packed layout after GGUF loading

## Purpose

This note records what layout the GatedDeltaNet weights are in once the forked `transformers` GGUF loader has loaded them, for the two checkpoints this project supports. It exists so that the HIP/GGTensile kernels for `in_proj_qkv` and `in_proj_z` can be written against the stored layout directly, and so the training code can call them without a permutation, a copy or a transposed copy.

| Checkpoint | Architecture | Layers | Hidden | Key heads | Value heads per key head | Key dim | Value dim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen3.6-35B-A3B-APEX-I-Mini.gguf` | `qwen35moe` | 40 (30 GatedDeltaNet) | 2048 | 16 | 2 | 2048 | 4096 |
| `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.gguf` | `qwen4exp` | 48 (36 GatedDeltaNet) | 2560 | 16 | 3 | 2048 | 6144 |

| Projection | Qwen3.6 logical `(N,K)` | Qwen3.8 logical `(N,K)` | Reorder after loading |
| --- | ---: | ---: | --- |
| `in_proj_qkv` | `(8192,2048)` | `(10240,2560)` | tiled -> grouped on the value rows only (rows), applied to the packed blocks |
| `in_proj_z` | `(4096,2048)` | `(6144,2560)` | tiled -> grouped on all rows, applied to the packed blocks |
| `out_proj` | `(2048,4096)` | `(2560,6144)` | tiled -> grouped on the *columns*. Deferred |

## Why a reorder exists

llama.cpp's converter writes the value heads *tiled* per key head (`v0k0 v0k1 v1k0 v1k1 ...`), while the model wants them *grouped* by key head. `transformers/integrations/gguf/gguf_conversion_mapping.py` undoes that in `_qwen35`'s `value_reorder`:
- `in_proj_qkv` gets `TiledToGroupedRows(..., offset=2*key_dim)`, so only the trailing value block is reordered and the q/k rows are kept.
- `in_proj_z` gets `TiledToGroupedRows(...)` over all of its rows.
- `out_proj` gets `TiledToGroupedInputs(...)`, a *column* reorder, and is the only one of the three that would cross quantization blocks.

`qwen4exp`'s mapping is `_qwen35_moe(config) + [...]`, and `_qwen35_moe` is `_qwen35 + MoE parts`, so both checkpoints are handled by exactly the same GatedDeltaNet code with the same operations.

The model's forward assumes the grouped layout (`z.reshape(batch_size, seq_len, -1, head_v_dim)` in `Qwen3_5MoeGatedDeltaNet.forward`). The reorder is therefore part of loading, not of the model.

## What the loader does with a packed weight

- `PermuteRows` declares `supports_packed = True` and permutes dim 0 of the payload, which for a GGUF weight is whole rows of whole blocks: no block is split, so no requantization is needed and the parameter stays a `GgufQuantizedParameter`.
- `PermuteInputFeatures` cannot permute packed columns, so it passes the packed tensor through untouched and hands the loader an `input_permutation` (the inverse of the column permutation). `get_gguf_plan` returns it, `quantizer_gguf.py` assigns it to `module.input_permutation`, and `GgufLinear.forward` applies it to its input instead.

So after loading:
- `in_proj_qkv` / `in_proj_z`: packed, rows already in the model's grouped order, `input_permutation = None`, `output_permutation = None`.
- `out_proj`: packed, rows exactly as the file stored them, `input_permutation` of 4096 (Qwen3.6) or 6144 (Qwen3.8) features at offset 0.

## Verified evidence

File-level, both checkpoints, running the loader's own plan and conversion chain on the file's bytes (`~/tmp/test_no_unsloth/qwen_gdn_plan.py`):

| Checkpoint | Projection | kept packed | input permutation | packed rows |
| --- | --- | --- | ---: | --- |
| Qwen3.6 | `in_proj_qkv` | yes (Q4_K, Q5_K) | none | 4096 value rows reordered, 4096 q/k rows kept, equals tiled -> grouped |
| Qwen3.6 | `in_proj_z` | yes (Q3_K, Q4_K) | none | all 4096 rows reordered, equals tiled -> grouped |
| Qwen3.6 | `out_proj` | yes | 4096 | byte-identical to the file |
| Qwen3.8 | `in_proj_qkv` | yes (Q2_0, Q3_K, Q4_K, IQ4_XS) | none | 6144 value rows reordered, 4096 q/k rows kept, equals tiled -> grouped |
| Qwen3.8 | `in_proj_z` | yes (Q2_0, Q3_K, Q4_K, IQ4_XS) | none | all 6144 rows reordered, equals tiled -> grouped |
| Qwen3.8 | `out_proj` | yes | 6144 | byte-identical to the file |

Loaded-model, Qwen3.6-35B-A3B-APEX-I-Mini, decoder layer 0, bf16 on the GPU (`~/tmp/test_no_unsloth/qwen_gdn_check.py`): the module is the fork's `GgufLinear` with `compute_dtype = bf16` and no permutations set, and:
- `in_proj_qkv`: cosine(module forward, plain matmul with the packed weight as loaded) = 0.999997, against the file's bytes 0.532 (value rows 0.067, q/k rows 0.999997).
- `in_proj_z`: 0.999997 against the loaded bytes, 0.092 against the file's.
- `out_proj`: 0.054 against the stored bytes. The module only reproduces with its input gathered.

## What this means for the kernels and for wiring

- The packed parameter's storage is `[out_features, row_bytes]`, row-major, with one logical row per stored row, and `row_bytes` is the kernel's weight stride (see the `Packed row` column of each HIP record).
- A kernel for `in_proj_qkv` or `in_proj_z` consumes that storage as it is: no dequantization, no row gathering, no transpose of the weights, and no permutation of the activation. The activation is the layer input exactly as the model computes it, and the input gradient the kernel returns is in that same feature order, so the backward pass needs no un-gather either.
- The only structural requirement is the ordinary one: the packed tensor's rows must be contiguous rows of whole blocks, which they are for every block width in these checkpoints (K2048 with the 256-wide types, K2560 with both the 256-wide types and the 64-wide Q2_0).
- `out_proj` is the exception and stays deferred: its forward needs the input gathered into the grouped order (one `index_select` of 4096 or 6144 features, the same gather the module performs), and its input gradient comes back in that gathered order.
