#pragma once

// Pipelined Q2_0 input-gradient body.
//
// The shared dense body decodes one contraction stage into a single shared
// tile, barriers, multiplies from it, and barriers again, so every stage pays
// two barriers and the matrix units idle while the tile is filled. This body
// keeps the same tile geometry, decode and matrix work but holds two shared
// tiles: while the matrix units consume stage `s`, the loader writes stage
// `s + 1` into the other tile, and a single barrier per stage retires both.
//
// The decoded tile is written one column per thread at a stride of
// `K_ITERATION` values, so every store of a thread lands in the same LDS bank
// unless the tile is padded; the padding is therefore a template parameter the
// caller sets, as in the shared body.

#include "bf16_wmma.cuh"
#include "mmq_backward.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace torch_ggml_ops::ck {

// Decode sixteen consecutive input columns of one Q2_0 weight row.
static __device__ __forceinline__ void decode_q2_0_group(
        const char * __restrict__ packed_row,
        int block_index,
        int value_index,
        __hip_bfloat16 * values) {
    const auto & block =
        reinterpret_cast<const block_q2_0 *>(packed_row)[block_index];
    const float d = fp16_to_fp32(block.d);
#pragma unroll
    for (int index = 0; index < 16; ++index) {
        const int value = value_index + index;
        const int code = (block.qs[value >> 2] >> (2 * (value & 3))) & 0x03;
        values[index] = __float2bfloat16(d * static_cast<float>(code - 1));
    }
}

template <
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
    bool PREFETCH_A_FRAGMENTS>
static __device__ __forceinline__ void dense_mmq_q2_0_grad_input_body(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        int rows,
        int out_features,
        int in_features,
        int blocks_per_weight_row) {
    constexpr int N_PER_BLOCK = N_TILES * BACKWARD_N_PER_TILE;
    constexpr int WEIGHT_BLOCK_VALUES = QK2_0;
    static_assert(DECODER_WIDTH > 0 && N_PER_BLOCK % DECODER_WIDTH == 0);
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
        gguf_block_bytes<GGML_TYPE_Q2_0>();

    __shared__ backward_shared_b_tile<
        N_PER_BLOCK, K_ITERATION, LDS_PADDING, LDS_SWIZZLE_CHUNK> shared_b[2];
    f32_accumulator accumulators[M_TILES_PER_WAVE][N_TILES];

    // One contraction stage into the given tile: every thread decodes
    // `DECODER_WIDTH` consecutive input columns of one weight row per pass.
    const auto decode_stage = [&](auto & tile, int output_start) {
#pragma unroll
        for (int group_index = threadIdx.x;
             group_index < DECODE_GROUPS * K_ITERATION;
             group_index += BACKWARD_THREADS) {
            const int k = group_index / DECODE_GROUPS;
            const int local_input_column =
                DECODER_WIDTH * (group_index % DECODE_GROUPS);
            const int output_column = output_start + k;
            const int input_column = input_column_start + local_input_column;
            if constexpr (FULL_TILES) {
                const char * packed_row = packed_weight +
                    static_cast<int64_t>(output_column) * packed_row_bytes;
                __hip_bfloat16 values[DECODER_WIDTH];
                decode_q2_0_group(
                    packed_row,
                    input_column / WEIGHT_BLOCK_VALUES,
                    input_column % WEIGHT_BLOCK_VALUES,
                    values);
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
                decode_q2_0_group(
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

    const int stage_count =
        (kernel_out_features + K_ITERATION - 1) / K_ITERATION;
    decode_stage(shared_b[0], 0);
    __syncthreads();
    for (int stage = 0; stage < stage_count; ++stage) {
        if (stage + 1 < stage_count) {
            decode_stage(shared_b[(stage + 1) & 1], (stage + 1) * K_ITERATION);
        }
        multiply_stage(shared_b[stage & 1], stage * K_ITERATION);
        __syncthreads();
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
                    if constexpr (FULL_TILES) {
                        grad_input[static_cast<int64_t>(output_row) *
                                       kernel_in_features +
                                   output_column] =
                            __float2bfloat16(
                                accumulators[m_tile][n_tile].values[element]);
                    } else if (
                        output_row < rows && output_column < kernel_in_features) {
                        grad_input[static_cast<int64_t>(output_row) *
                                       kernel_in_features +
                                   output_column] =
                            __float2bfloat16(
                                accumulators[m_tile][n_tile].values[element]);
                    }
                }
            }
        }
    }
}

} // namespace torch_ggml_ops::ck
