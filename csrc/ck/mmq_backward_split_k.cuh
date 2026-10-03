#pragma once

#include "ck/mmq_backward.cuh"

namespace torch_ggml_ops::ck {

// Split-contraction dense input-gradient bodies.
//
// The dense backward kernels for the language-model-head shapes have a tiny
// output tile count and a very large contraction, so the ordinary grid leaves
// most of the machine idle. These bodies keep the deployed tile, decode and
// matrix work unchanged, but give each workgroup one contiguous slice of the
// contraction and write the per-slice FP32 partial tile instead of the
// output. A separate reduction kernel sums the slices in ascending order and
// rounds once to BF16, so the result is deterministic while the accumulation
// order inside a slice is the deployed one.
//
// The FP32 partial traffic is what keeps this mechanism off the routed
// families, whose output is large and whose contraction is short. Here the
// output is `rows x 2048/4096` and the contraction is `129,280` or `248,320`,
// so the partial round trip is a fraction of a percent of the multiply.

template <
    ggml_type type,
    int EXACT_OUT_FEATURES,
    int EXACT_IN_FEATURES,
    int N_TILES,
    int K_ITERATION,
    int M_TILES_PER_WAVE,
    int ACTIVE_WAVES,
    int DECODER_WIDTH,
    int LDS_SWIZZLE_CHUNK,
    bool FULL_TILES,
    bool VECTOR_LOCAL_LOAD,
    bool PACK_Q6_QUANT_BYTES>
static __device__ __forceinline__ void dense_mmq_grad_input_splitk_body(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        float * __restrict__ partials,
        int rows,
        int out_features,
        int in_features,
        int blocks_per_weight_row,
        int split_chunk) {
    constexpr int N_PER_BLOCK = N_TILES * BACKWARD_N_PER_TILE;
    constexpr int WEIGHT_BLOCK_VALUES =
        type == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
    const int kernel_out_features = EXACT_OUT_FEATURES > 0
        ? EXACT_OUT_FEATURES
        : out_features;
    const int kernel_in_features = EXACT_IN_FEATURES > 0
        ? EXACT_IN_FEATURES
        : in_features;
    const int kernel_blocks_per_weight_row = EXACT_IN_FEATURES > 0
        ? EXACT_IN_FEATURES / WEIGHT_BLOCK_VALUES
        : blocks_per_weight_row;
    constexpr int M_PER_WAVE = M_TILES_PER_WAVE * BACKWARD_M_PER_TILE;
    constexpr int M_PER_BLOCK = M_PER_WAVE * ACTIVE_WAVES;
    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int block_row_start = blockIdx.x * M_PER_BLOCK;
    const int wave_row_start = block_row_start + wave * M_PER_WAVE;
    const int input_column_start = blockIdx.y * N_PER_BLOCK;
    const int split_begin = blockIdx.z * split_chunk;
    const int split_end = min(split_begin + split_chunk, kernel_out_features);
    const int64_t packed_row_bytes =
        static_cast<int64_t>(kernel_blocks_per_weight_row) *
        gguf_block_bytes<type>();

    __shared__ backward_shared_b_tile<
        N_PER_BLOCK, K_ITERATION, 0, LDS_SWIZZLE_CHUNK> shared_b;
    f32_accumulator accumulators[M_TILES_PER_WAVE][N_TILES];

    for (int output_start = split_begin; output_start < split_end;
         output_start += K_ITERATION) {
        if constexpr (type == GGML_TYPE_Q6_K && N_TILES >= 2) {
            constexpr int groups_per_row = N_PER_BLOCK / 16;
#pragma unroll
            for (int group_index = threadIdx.x;
                 group_index < groups_per_row * K_ITERATION;
                 group_index += BACKWARD_THREADS) {
                const int k = group_index / groups_per_row;
                const int local_input_column = 16 * (group_index % groups_per_row);
                const int output_column = output_start + k;
                const int input_column = input_column_start + local_input_column;
                if (FULL_TILES ||
                    (input_column + 15 < kernel_in_features &&
                     output_column < kernel_out_features)) {
                    const char * packed_row = packed_weight +
                        static_cast<int64_t>(output_column) * packed_row_bytes;
                    __hip_bfloat16 values[16];
                    decode_backward_tile_sixteen_q6<PACK_Q6_QUANT_BYTES>(
                        packed_row,
                        input_column / WEIGHT_BLOCK_VALUES,
                        input_column % WEIGHT_BLOCK_VALUES,
                        values);
#pragma unroll
                    for (int index = 0; index < 16; ++index) {
                        shared_b[(local_input_column + index) * K_ITERATION + k] =
                            values[index];
                    }
                } else {
#pragma unroll
                    for (int index = 0; index < 16; ++index) {
                        shared_b[(local_input_column + index) * K_ITERATION + k] =
                            __float2bfloat16(0.0f);
                    }
                }
            }
        } else if constexpr (
            DECODER_WIDTH > 0 &&
            (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q3_K ||
             type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)
        ) {
            constexpr int groups_per_row = N_PER_BLOCK / DECODER_WIDTH;
#pragma unroll
            for (int group_index = threadIdx.x;
                 group_index < groups_per_row * K_ITERATION;
                 group_index += BACKWARD_THREADS) {
                const int k = group_index / groups_per_row;
                const int local_input_column =
                    DECODER_WIDTH * (group_index % groups_per_row);
                const int output_column = output_start + k;
                const int input_column = input_column_start + local_input_column;
                if (FULL_TILES ||
                    (input_column + DECODER_WIDTH - 1 < kernel_in_features &&
                     output_column < kernel_out_features)) {
                    const char * packed_row = packed_weight +
                        static_cast<int64_t>(output_column) * packed_row_bytes;
                    __hip_bfloat16 values[DECODER_WIDTH];
                    decode_backward_tile_group<type, DECODER_WIDTH>(
                        packed_row,
                        input_column / WEIGHT_BLOCK_VALUES,
                        input_column % WEIGHT_BLOCK_VALUES,
                        values);
#pragma unroll
                    for (int index = 0; index < DECODER_WIDTH; ++index) {
                        shared_b[(local_input_column + index) * K_ITERATION + k] =
                            values[index];
                    }
                } else {
#pragma unroll
                    for (int index = 0; index < DECODER_WIDTH; ++index) {
                        shared_b[(local_input_column + index) * K_ITERATION + k] =
                            __float2bfloat16(0.0f);
                    }
                }
            }
        }
        __syncthreads();

        if (ACTIVE_WAVES == BACKWARD_WAVES || wave < ACTIVE_WAVES) {
#pragma unroll
            for (int k_tile = 0; k_tile < K_ITERATION; k_tile += 16) {
                bf16_fragment a_fragments[M_TILES_PER_WAVE];
#pragma unroll
                for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
                    __hip_bfloat16 * a = fragment_data(a_fragments[m_tile]);
                    const int a_row =
                        wave_row_start + m_tile * BACKWARD_M_PER_TILE + c_row(lane);
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        const int output_column = output_start + k_tile + k;
                        if constexpr (FULL_TILES) {
                            a[k] = grad_output[
                                static_cast<int64_t>(a_row) * kernel_out_features +
                                output_column];
                        } else {
                            a[k] = a_row < rows && output_column < kernel_out_features
                                ? grad_output[
                                      static_cast<int64_t>(a_row) *
                                          kernel_out_features +
                                      output_column]
                                : __float2bfloat16(0.0f);
                        }
                    }
                }
#pragma unroll
                for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
                    bf16_fragment b_fragment{};
                    if constexpr (VECTOR_LOCAL_LOAD) {
                        shared_b.load_fragment_vector(
                            b_fragment,
                            n_tile * BACKWARD_N_PER_TILE + c_row(lane),
                            k_tile);
                    } else {
                        __hip_bfloat16 * b = fragment_data(b_fragment);
#pragma unroll
                        for (int k = 0; k < 16; ++k) {
                            b[k] = shared_b[
                                (n_tile * BACKWARD_N_PER_TILE + c_row(lane)) *
                                    K_ITERATION +
                                k_tile + k];
                        }
                    }
#pragma unroll
                    for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
                        wmma_f32_16x16x16_bf16(
                            accumulators[m_tile][n_tile],
                            a_fragments[m_tile],
                            b_fragment);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (ACTIVE_WAVES == BACKWARD_WAVES || wave < ACTIVE_WAVES) {
        const int64_t split_offset = static_cast<int64_t>(blockIdx.z) * rows *
            kernel_in_features;
#pragma unroll
        for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
#pragma unroll
            for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
#pragma unroll
                for (int element = 0; element < 8; ++element) {
                    const int output_row = wave_row_start +
                        m_tile * BACKWARD_M_PER_TILE + c_column(lane, element);
                    const int output_column = input_column_start +
                        n_tile * BACKWARD_N_PER_TILE + c_row(lane);
                    if constexpr (FULL_TILES) {
                        partials[split_offset +
                                 static_cast<int64_t>(output_row) *
                                     kernel_in_features +
                                 output_column] =
                            accumulators[m_tile][n_tile].values[element];
                    } else if (output_row < rows &&
                               output_column < kernel_in_features) {
                        partials[split_offset +
                                 static_cast<int64_t>(output_row) *
                                     kernel_in_features +
                                 output_column] =
                            accumulators[m_tile][n_tile].values[element];
                    }
                }
            }
        }
    }
}

// Deterministic reduction over the splits: ascending split order, one BF16
// rounding at the end.
static __device__ __forceinline__ void dense_mmq_grad_input_splitk_reduce_body(
        const float * __restrict__ partials,
        __hip_bfloat16 * __restrict__ grad_input,
        int rows,
        int in_features,
        int splits) {
    const int64_t count = static_cast<int64_t>(rows) * in_features;
    const int64_t split_stride = count;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < count;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        float total = 0.0f;
        for (int split = 0; split < splits; ++split) {
            total += partials[split * split_stride + index];
        }
        grad_input[index] = __float2bfloat16(total);
    }
}

} // namespace torch_ggml_ops::ck
