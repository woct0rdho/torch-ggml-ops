#pragma once

// Pipelined input-gradient body, reusable across quantized weight types.
//
// The shared dense body decodes one contraction stage into a single shared
// tile, barriers, multiplies from it, and barriers again, so every stage pays
// two barriers and the matrix units idle while the tile is filled. This body
// keeps the same tile geometry, decode and matrix work but holds two shared
// tiles: while the matrix units consume stage `s`, the loader writes stage
// `s + 1` into the other tile, and a single barrier per stage retires both.
// Setting `PIPELINE_TILES` to false keeps one tile and the two-barrier order,
// so a caller can separate the tile pipelining from the activation prefetch.
//
// The decoded tile is written one column per thread at a stride of
// `K_ITERATION` values, so every store of a thread lands in the same LDS bank
// unless the tile is padded or swizzled; both are template parameters the
// caller sets, as in the shared body.

#include "bf16_wmma.cuh"
#include "mmq_backward.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace torch_ggml_ops::ck {

// Decode sixteen consecutive input columns of one weight row, with the
// payload byte of a Q2_0 row read once for its four codes.
template <ggml_type TYPE>
static __device__ __forceinline__ void decode_pipelined_group(
        const char * __restrict__ packed_row,
        int block_index,
        int value_index,
        __hip_bfloat16 * values) {
    if constexpr (TYPE == GGML_TYPE_Q2_0) {
        const auto & block =
            reinterpret_cast<const block_q2_0 *>(packed_row)[block_index];
        const float d = fp16_to_fp32(block.d);
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            const int value = value_index + index;
            const int code = (block.qs[value >> 2] >> (2 * (value & 3))) & 0x03;
            values[index] = __float2bfloat16(d * static_cast<float>(code - 1));
        }
    } else {
        decode_backward_tile_group<TYPE, 16>(
            packed_row, block_index, value_index, values);
    }
}

template <ggml_type TYPE, bool PACK_Q5_QUANT_BYTES, bool PACK_Q6_QUANT_BYTES>
static __device__ __forceinline__ void decode_pipelined_group_prefetched(
        const char * __restrict__ packed_row,
        int block_index,
        int value_index,
        __hip_bfloat16 * values) {
    if constexpr (TYPE == GGML_TYPE_Q8_0) {
        const auto & block =
            reinterpret_cast<const block_q8_0 *>(packed_row)[block_index];
        const uint4 payload =
            load_uint4_unaligned(block.qs + value_index);
        // Q8_0 stores signed quants: reading the prefetched word as unsigned
        // bytes shifts every negative value by +256.
        const auto * quants = reinterpret_cast<const int8_t *>(&payload);
        const float d = fp16_to_fp32(block.d);
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            values[index] =
                __float2bfloat16(d * static_cast<float>(quants[index]));
        }
    } else if constexpr (TYPE == GGML_TYPE_Q4_0) {
        const auto & block =
            reinterpret_cast<const block_q4_0 *>(packed_row)[block_index];
        const uint4 payload = load_uint4_unaligned(block.qs);
        const auto * bytes = reinterpret_cast<const uint8_t *>(&payload);
        const int shift = 4 * ((value_index >> 4) & 1);
        const float d = fp16_to_fp32(block.d);
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            const int quant =
                (bytes[(value_index + index) & (QK4_0 / 2 - 1)] >> shift) & 0x0f;
            values[index] = __float2bfloat16(
                d * static_cast<float>(quant - 8));
        }
    } else if constexpr (TYPE == GGML_TYPE_Q5_0) {
        const auto & block =
            reinterpret_cast<const block_q5_0 *>(packed_row)[block_index];
        const uint4 payload = load_uint4_unaligned(block.qs);
        const auto * bytes = reinterpret_cast<const uint8_t *>(&payload);
        const uint32_t fifth = get_int_b4(block.qh, 0);
        const int shift = 4 * ((value_index >> 4) & 1);
        const float d = fp16_to_fp32(block.d);
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            const int value = value_index + index;
            const int low =
                (bytes[value & (QK5_0 / 2 - 1)] >> shift) & 0x0f;
            const int high = (fifth >> value) & 0x01;
            values[index] = __float2bfloat16(
                d * static_cast<float>((low | (high << 4)) - 16));
        }
    } else if constexpr (TYPE == GGML_TYPE_IQ4_NL) {
        const auto & block =
            reinterpret_cast<const block_iq4_nl *>(packed_row)[block_index];
        const int first_word = (value_index & (QK4_NL / 2 - 1)) / 4;
        const bool high_plane = ((value_index >> 4) & 1) != 0;
        const float d = fp16_to_fp32(block.d);
        // One 16-byte payload block holds both nibble planes, so a 32-bit word
        // feeds both a low-nibble and a high-nibble lookup at once and the
        // sixteen values need four permute pairs instead of sixteen gathers.
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            const int2 levels = iq4_table_lookup_16(
                get_int_b2(block.qs, first_word + word), kvalues_iq4nl);
            const int packed = high_plane ? levels.y : levels.x;
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                const auto level = static_cast<int8_t>(packed >> (8 * slot));
                values[4 * word + slot] =
                    __float2bfloat16(d * static_cast<float>(level));
            }
        }
    } else if constexpr (TYPE == GGML_TYPE_IQ4_XS) {
        const auto & block =
            reinterpret_cast<const block_iq4_xs *>(packed_row)[block_index];
        const int sub_block = value_index >> 5;
        const int plane = (value_index >> 4) & 1;
        const float scaled_d = fp16_to_fp32(block.d) *
            static_cast<float>(iq4_xs_scale(block, sub_block));
        const uint4 payload = load_uint4_unaligned(block.qs + 16 * sub_block);
        const auto * words = reinterpret_cast<const uint32_t *>(&payload);
        // The sixteen payload bytes hold both planes, so each word feeds one
        // permute pair and the plane selects the half.
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            const int2 nibbles =
                iq4_table_lookup_16(static_cast<int>(words[word]), kvalues_iq4nl);
            const int packed = plane ? nibbles.y : nibbles.x;
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                const auto level = static_cast<int8_t>(packed >> (8 * slot));
                values[4 * word + slot] =
                    __float2bfloat16(scaled_d * static_cast<float>(level));
            }
        }
    } else if constexpr (TYPE == GGML_TYPE_Q3_K) {
        const auto & block =
            reinterpret_cast<const block_q3_K *>(packed_row)[block_index];
        const int low_byte =
            (value_index >> 7) * 32 + (value_index & 31);
        const uint4 low = load_uint4_unaligned(block.qs + low_byte);
        const uint4 high =
            load_uint4_unaligned(block.hmask + (value_index & 31));
        decode_backward_tile_q3_preloaded(
            block, value_index, low, high, values);
    } else if constexpr (TYPE == GGML_TYPE_Q4_K) {
        const auto & block =
            reinterpret_cast<const block_q4_K *>(packed_row)[block_index];
        const int group = value_index >> 5;
        const int byte = (group >> 1) * 32 + (value_index & 31);
        const uint4 quants = load_uint4_unaligned(block.qs + byte);
        decode_backward_tile_q4_preloaded(
            block, value_index, quants, values);
    } else if constexpr (TYPE == GGML_TYPE_Q5_K) {
        const auto & block =
            reinterpret_cast<const block_q5_K *>(packed_row)[block_index];
        const int group = value_index >> 5;
        const int low_byte = (group >> 1) * 32 + (value_index & 31);
        const uint4 low = load_uint4_unaligned(block.qs + low_byte);
        const uint4 high =
            load_uint4_unaligned(block.qh + (value_index & 31));
        decode_backward_tile_q5_preloaded<PACK_Q5_QUANT_BYTES>(
            block, value_index, low, high, values);
    } else {
        decode_backward_tile_sixteen_q6<PACK_Q6_QUANT_BYTES>(
            packed_row, block_index, value_index, values);
    }
}

template <
    ggml_type TYPE,
    int EXACT_OUT_FEATURES,
    int EXACT_IN_FEATURES,
    int N_TILES,
    int K_ITERATION,
    int GROUP_M,
    int M_TILES_PER_WAVE,
    int ACTIVE_WAVES,
    int LDS_PADDING,
    int LDS_SWIZZLE_CHUNK,
    bool FULL_TILES,
    bool PREFETCH_LOCAL,
    bool VECTOR_LOCAL_LOAD,
    int DECODER_WIDTH,
    bool PREFETCH_A_FRAGMENTS,
    bool PIPELINE_TILES,
    bool PREFETCH_PACKED,
    bool PACK_Q5_QUANT_BYTES,
    bool PACK_Q6_QUANT_BYTES,
    bool PARTIAL_STORE>
static __device__ __forceinline__ void dense_mmq_pipelined_grad_input_body(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        float * __restrict__ partials,
        int rows,
        int out_features,
        int in_features,
        int blocks_per_weight_row,
        int split_begin,
        int split_end) {
    constexpr int N_PER_BLOCK = N_TILES * BACKWARD_N_PER_TILE;
    constexpr int WEIGHT_BLOCK_VALUES =
        TYPE == GGML_TYPE_Q8_0 ? QK8_0
                               : (TYPE == GGML_TYPE_Q2_0
                                      ? QK2_0
                                      : (TYPE == GGML_TYPE_Q4_0
                                             ? QK4_0
                                             : (TYPE == GGML_TYPE_Q5_0
                                                    ? QK5_0
                                                    : (TYPE == GGML_TYPE_IQ4_NL
                                                           ? QK4_NL
                                                           : QK_K))));
    static_assert(DECODER_WIDTH > 0 && N_PER_BLOCK % DECODER_WIDTH == 0);
    static_assert(
        TYPE == GGML_TYPE_Q2_0 || TYPE == GGML_TYPE_Q4_0 ||
        TYPE == GGML_TYPE_Q5_0 || TYPE == GGML_TYPE_IQ4_NL ||
        TYPE == GGML_TYPE_IQ4_XS || TYPE == GGML_TYPE_Q8_0 ||
        TYPE == GGML_TYPE_Q3_K || TYPE == GGML_TYPE_Q4_K ||
        TYPE == GGML_TYPE_Q5_K || TYPE == GGML_TYPE_Q6_K,
        "the pipelined body decodes one of the staged weight types");
    if constexpr (PARTIAL_STORE) {
        static_assert(
            FULL_TILES,
            "a split-contraction slice stages whole tiles, so it can only be "
            "built unguarded");
    }
    static_assert(K_ITERATION % 16 == 0);
    static_assert(ACTIVE_WAVES > 0 && ACTIVE_WAVES <= BACKWARD_WAVES);

    const int kernel_out_features =
        EXACT_OUT_FEATURES > 0 ? EXACT_OUT_FEATURES : out_features;
    const int kernel_in_features =
        EXACT_IN_FEATURES > 0 ? EXACT_IN_FEATURES : in_features;
    const int kernel_blocks_per_weight_row = EXACT_IN_FEATURES > 0
        ? EXACT_IN_FEATURES / WEIGHT_BLOCK_VALUES
        : blocks_per_weight_row;
    constexpr int M_PER_WAVE = M_TILES_PER_WAVE * BACKWARD_M_PER_TILE;
    constexpr int M_PER_BLOCK = M_PER_WAVE * ACTIVE_WAVES;
    constexpr int DECODE_GROUPS = N_PER_BLOCK / DECODER_WIDTH;

    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int m_block =
        GROUP_M > 0 ? blockIdx.z * GROUP_M + blockIdx.x : blockIdx.x;
    const int block_row_start = m_block * M_PER_BLOCK;
    const int wave_row_start = block_row_start + wave * M_PER_WAVE;
    const int input_column_start = blockIdx.y * N_PER_BLOCK;
    const int64_t packed_row_bytes =
        static_cast<int64_t>(kernel_blocks_per_weight_row) *
        gguf_block_bytes<TYPE>();

    __shared__ backward_shared_b_tile<
        N_PER_BLOCK, K_ITERATION, LDS_PADDING, LDS_SWIZZLE_CHUNK>
        shared_b[PIPELINE_TILES ? 2 : 1];
    f32_accumulator accumulators[M_TILES_PER_WAVE][N_TILES];

    // One contraction stage into the given tile: every thread decodes
    // `DECODER_WIDTH` consecutive input columns of one weight row per pass.
    const auto decode_stage = [&](auto & tile, int output_start, int stage_end) {
#pragma unroll
        for (int group_index = threadIdx.x;
             group_index < DECODE_GROUPS * K_ITERATION;
             group_index += BACKWARD_THREADS) {
            const int k = group_index / DECODE_GROUPS;
            const int local_input_column =
                DECODER_WIDTH * (group_index % DECODE_GROUPS);
            const int output_column = output_start + k;
            const int input_column = input_column_start + local_input_column;
            if constexpr (PARTIAL_STORE) {
#pragma unroll
                for (int index = 0; index < DECODER_WIDTH; ++index) {
                    tile[(local_input_column + index) * K_ITERATION + k] =
                        __float2bfloat16(0.0f);
                }
                if (output_column < stage_end) {
                    const char * packed_row = packed_weight +
                        static_cast<int64_t>(output_column) * packed_row_bytes;
                    __hip_bfloat16 values[DECODER_WIDTH];
                    if constexpr (PREFETCH_PACKED) {
                        decode_pipelined_group_prefetched<
                            TYPE, PACK_Q5_QUANT_BYTES, PACK_Q6_QUANT_BYTES>(
                            packed_row,
                            input_column / WEIGHT_BLOCK_VALUES,
                            input_column % WEIGHT_BLOCK_VALUES,
                            values);
                    } else {
                        decode_pipelined_group<TYPE>(
                            packed_row,
                            input_column / WEIGHT_BLOCK_VALUES,
                            input_column % WEIGHT_BLOCK_VALUES,
                            values);
                    }
#pragma unroll
                    for (int index = 0; index < DECODER_WIDTH; ++index) {
                        tile[(local_input_column + index) * K_ITERATION + k] =
                            values[index];
                    }
                }
            } else if constexpr (FULL_TILES) {
                const char * packed_row = packed_weight +
                    static_cast<int64_t>(output_column) * packed_row_bytes;
                __hip_bfloat16 values[DECODER_WIDTH];
                if constexpr (PREFETCH_PACKED) {
                    decode_pipelined_group_prefetched<
                        TYPE, PACK_Q5_QUANT_BYTES, PACK_Q6_QUANT_BYTES>(
                        packed_row,
                        input_column / WEIGHT_BLOCK_VALUES,
                        input_column % WEIGHT_BLOCK_VALUES,
                        values);
                } else {
                    decode_pipelined_group<TYPE>(
                        packed_row,
                        input_column / WEIGHT_BLOCK_VALUES,
                        input_column % WEIGHT_BLOCK_VALUES,
                        values);
                }
#pragma unroll
                for (int index = 0; index < DECODER_WIDTH; ++index) {
                    tile[(local_input_column + index) * K_ITERATION + k] =
                        values[index];
                }
            } else if (
                input_column + DECODER_WIDTH - 1 < kernel_in_features &&
                output_column < kernel_out_features) {
                const char * packed_row = packed_weight +
                    static_cast<int64_t>(output_column) * packed_row_bytes;
                __hip_bfloat16 values[DECODER_WIDTH];
                decode_pipelined_group<TYPE>(
                    packed_row,
                    input_column / WEIGHT_BLOCK_VALUES,
                    input_column % WEIGHT_BLOCK_VALUES,
                    values);
#pragma unroll
                for (int index = 0; index < DECODER_WIDTH; ++index) {
                    tile[(local_input_column + index) * K_ITERATION + k] =
                        values[index];
                }
            } else {
#pragma unroll
                for (int index = 0; index < DECODER_WIDTH; ++index) {
                    tile[(local_input_column + index) * K_ITERATION + k] =
                        __float2bfloat16(0.0f);
                }
            }
        }
    };

    // Load the activation fragment quad of one k tile; every lane reads
    // sixteen bf16 values, one per k, of its row of the cotangent.
    const auto load_a_fragments = [&](bf16_fragment (&a_fragments)
                                              [M_TILES_PER_WAVE],
                                      int output_start) {
#pragma unroll
        for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
            __hip_bfloat16 *a = fragment_data(a_fragments[m_tile]);
            const int a_row = wave_row_start + m_tile * BACKWARD_M_PER_TILE +
                c_row(lane);
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const int output_column = output_start + k;
                if constexpr (FULL_TILES) {
                    a[k] = grad_output[static_cast<int64_t>(a_row) *
                                           kernel_out_features +
                                       output_column];
                } else {
                    a[k] = a_row < rows && output_column < kernel_out_features
                        ? grad_output[static_cast<int64_t>(a_row) *
                                          kernel_out_features +
                                      output_column]
                        : __float2bfloat16(0.0f);
                }
            }
        }
    };

    const auto multiply_stage = [&](auto & tile, int output_start) {
        if (ACTIVE_WAVES == BACKWARD_WAVES || wave < ACTIVE_WAVES) {
#pragma unroll
            for (int k_tile = 0; k_tile < K_ITERATION; k_tile += 16) {
                bf16_fragment a_fragments[PREFETCH_A_FRAGMENTS ? 2 : 1]
                                          [M_TILES_PER_WAVE];
                if constexpr (PREFETCH_A_FRAGMENTS) {
                    load_a_fragments(
                        a_fragments[(k_tile / 16) & 1], output_start + k_tile);
                } else {
                    load_a_fragments(a_fragments[0], output_start + k_tile);
                }
                const int a_buffer = PREFETCH_A_FRAGMENTS ? (k_tile / 16) & 1 : 0;
                if constexpr (PREFETCH_LOCAL) {
#pragma unroll
                    for (int n_tile = 0; n_tile < N_TILES - 1; n_tile += 2) {
                        bf16_fragment b_first{};
                        bf16_fragment b_second{};
                        if constexpr (VECTOR_LOCAL_LOAD) {
                            tile.load_fragment_vector(
                                b_first,
                                n_tile * BACKWARD_N_PER_TILE + c_row(lane),
                                k_tile);
                            tile.load_fragment_vector(
                                b_second,
                                (n_tile + 1) * BACKWARD_N_PER_TILE + c_row(lane),
                                k_tile);
                        } else {
                            __hip_bfloat16 *first = fragment_data(b_first);
                            __hip_bfloat16 *second = fragment_data(b_second);
#pragma unroll
                            for (int k = 0; k < 16; ++k) {
                                first[k] =
                                    tile[(n_tile * BACKWARD_N_PER_TILE +
                                          c_row(lane)) *
                                             K_ITERATION +
                                         k_tile + k];
                                second[k] =
                                    tile[((n_tile + 1) * BACKWARD_N_PER_TILE +
                                          c_row(lane)) *
                                             K_ITERATION +
                                         k_tile + k];
                            }
                        }
#pragma unroll
                        for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
                            wmma_f32_16x16x16_bf16(
                                accumulators[m_tile][n_tile],
                                a_fragments[a_buffer][m_tile], b_first);
                        }
                        if constexpr (PREFETCH_A_FRAGMENTS) {
                            // The next tile's cotangent reads are issued one k
                            // tile early so their latency retires under this
                            // tile's matrix work instead of in front of it.
                            if (k_tile + 16 < K_ITERATION) {
                                load_a_fragments(
                                    a_fragments[((k_tile / 16) + 1) & 1],
                                    output_start + k_tile + 16);
                            }
                        }
#pragma unroll
                        for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
                            wmma_f32_16x16x16_bf16(
                                accumulators[m_tile][n_tile + 1],
                                a_fragments[a_buffer][m_tile], b_second);
                        }
                    }
                    if constexpr (N_TILES % 2 != 0) {
                        constexpr int n_tile = N_TILES - 1;
                        bf16_fragment b_fragment{};
                        if constexpr (VECTOR_LOCAL_LOAD) {
                            tile.load_fragment_vector(
                                b_fragment,
                                n_tile * BACKWARD_N_PER_TILE + c_row(lane),
                                k_tile);
                        } else {
                            __hip_bfloat16 *b = fragment_data(b_fragment);
#pragma unroll
                            for (int k = 0; k < 16; ++k) {
                                b[k] =
                                    tile[(n_tile * BACKWARD_N_PER_TILE +
                                          c_row(lane)) *
                                             K_ITERATION +
                                         k_tile + k];
                            }
                        }
#pragma unroll
                        for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
                            wmma_f32_16x16x16_bf16(
                                accumulators[m_tile][n_tile],
                                a_fragments[a_buffer][m_tile], b_fragment);
                        }
                    }
                } else {
#pragma unroll
                    for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
                        bf16_fragment b_fragment{};
                        if constexpr (VECTOR_LOCAL_LOAD) {
                            tile.load_fragment_vector(
                                b_fragment,
                                n_tile * BACKWARD_N_PER_TILE + c_row(lane),
                                k_tile);
                        } else {
                            __hip_bfloat16 *b = fragment_data(b_fragment);
#pragma unroll
                            for (int k = 0; k < 16; ++k) {
                                b[k] =
                                    tile[(n_tile * BACKWARD_N_PER_TILE +
                                          c_row(lane)) *
                                             K_ITERATION +
                                         k_tile + k];
                            }
                        }
#pragma unroll
                        for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
                            wmma_f32_16x16x16_bf16(
                                accumulators[m_tile][n_tile],
                                a_fragments[a_buffer][m_tile], b_fragment);
                        }
                    }
                }
            }
        }
    };

    const int contraction_begin = PARTIAL_STORE ? split_begin : 0;
    const int contraction_end =
        PARTIAL_STORE ? split_end : kernel_out_features;
    const int stage_count =
        (contraction_end - contraction_begin + K_ITERATION - 1) / K_ITERATION;
    const auto stage_end_of = [&](int stage) {
        return min(
            contraction_begin + (stage + 1) * K_ITERATION, contraction_end);
    };
    decode_stage(shared_b[0], contraction_begin, stage_end_of(0));
    __syncthreads();
    if constexpr (PIPELINE_TILES) {
        for (int stage = 0; stage < stage_count; ++stage) {
            if (stage + 1 < stage_count) {
                decode_stage(
                    shared_b[(stage + 1) & 1],
                    contraction_begin + (stage + 1) * K_ITERATION,
                    stage_end_of(stage + 1));
            }
            multiply_stage(
                shared_b[stage & 1], contraction_begin + stage * K_ITERATION);
            __syncthreads();
        }
    } else {
        multiply_stage(shared_b[0], contraction_begin);
        __syncthreads();
        for (int stage = 1; stage < stage_count; ++stage) {
            decode_stage(
                shared_b[0],
                contraction_begin + stage * K_ITERATION,
                stage_end_of(stage));
            __syncthreads();
            multiply_stage(
                shared_b[0], contraction_begin + stage * K_ITERATION);
            __syncthreads();
        }
    }

    if (ACTIVE_WAVES == BACKWARD_WAVES || wave < ACTIVE_WAVES) {
#pragma unroll
        for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
#pragma unroll
            for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
#pragma unroll
                for (int element = 0; element < 8; ++element) {
                    // gfx11's physical C fragment is J-major for this A/B
                    // layout: the I-major lane coordinates are transposed when
                    // written to row-major C.
                    const int output_row = wave_row_start +
                        m_tile * BACKWARD_M_PER_TILE +
                        c_column(lane, element);
                    const int output_column = input_column_start +
                        n_tile * BACKWARD_N_PER_TILE + c_row(lane);
                    const int64_t target =
                        static_cast<int64_t>(output_row) * kernel_in_features +
                        output_column;
                    if constexpr (PARTIAL_STORE) {
                        partials[target] =
                            accumulators[m_tile][n_tile].values[element];
                    } else if constexpr (FULL_TILES) {
                        grad_input[target] = __float2bfloat16(
                            accumulators[m_tile][n_tile].values[element]);
                    } else if (
                        output_row < rows && output_column < kernel_in_features) {
                        grad_input[target] = __float2bfloat16(
                            accumulators[m_tile][n_tile].values[element]);
                    }
                }
            }
        }
    }
}

// Prefetched variant: the payload planes covering the sixteen values are read
// as one or two `uint4` before decoding, which halves the loader's addressing
// work exactly as the deployed bodies' packed paths do. Every plane load is
// sixteen-byte aligned because the group is sixteen values wide and the
// per-type byte formulas are multiples of sixteen for such a group.
// Split-contraction variant of the pipelined body: the same stage order over
// one slice of the contraction, writing the per-slice FP32 partial tile that
// the reduction kernel consumes.
template <
    ggml_type TYPE,
    int EXACT_OUT_FEATURES,
    int EXACT_IN_FEATURES,
    int N_TILES,
    int K_ITERATION,
    int M_TILES_PER_WAVE,
    int ACTIVE_WAVES,
    int DECODER_WIDTH,
    int LDS_SWIZZLE_CHUNK,
    bool PREFETCH_LOCAL,
    bool VECTOR_LOCAL_LOAD,
    bool PREFETCH_PACKED,
    bool PACK_Q5_QUANT_BYTES,
    bool PACK_Q6_QUANT_BYTES>
static __device__ __forceinline__ void dense_mmq_pipelined_splitk_body(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        float * __restrict__ partials,
        int rows,
        int out_features,
        int in_features,
        int blocks_per_weight_row,
        int split_chunk) {
    constexpr int N_PER_BLOCK = N_TILES * BACKWARD_N_PER_TILE;
    const int kernel_in_features = EXACT_IN_FEATURES > 0
        ? EXACT_IN_FEATURES
        : in_features;
    const int split_begin = blockIdx.z * split_chunk;
    const int split_end = min(split_begin + split_chunk, out_features);
    dense_mmq_pipelined_grad_input_body<
        TYPE,
        EXACT_OUT_FEATURES,
        EXACT_IN_FEATURES,
        N_TILES,
        K_ITERATION,
        0,
        M_TILES_PER_WAVE,
        ACTIVE_WAVES,
        0,
        LDS_SWIZZLE_CHUNK,
        true,
        PREFETCH_LOCAL,
        VECTOR_LOCAL_LOAD,
        DECODER_WIDTH,
        false,
        true,
        PREFETCH_PACKED,
        PACK_Q5_QUANT_BYTES,
        PACK_Q6_QUANT_BYTES,
        true>(
            grad_output,
            packed_weight,
            nullptr,
            partials +
                static_cast<int64_t>(blockIdx.z) * rows * kernel_in_features,
            rows,
            out_features,
            in_features,
            blocks_per_weight_row,
            split_begin,
            split_end);
}

} // namespace torch_ggml_ops::ck
