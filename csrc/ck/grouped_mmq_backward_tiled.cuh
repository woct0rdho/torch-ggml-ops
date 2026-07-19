#pragma once

#include "mmq_backward.cuh"

namespace torch_ggml_ops::ck {

static constexpr int GROUPED_BACKWARD_TILED_Q4_OUT_FEATURES = 2048;
static constexpr int GROUPED_BACKWARD_TILED_Q4_IN_FEATURES = 512;
static constexpr int GROUPED_BACKWARD_TILED_Q4_BLOCKS_PER_ROW = 2;
static constexpr int GROUPED_BACKWARD_TILED_N_TILES = 8;
static constexpr int GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE = 2;
static constexpr int GROUPED_BACKWARD_TILED_K = 32;
static constexpr int GROUPED_BACKWARD_TILED_N =
    GROUPED_BACKWARD_TILED_N_TILES * BACKWARD_N_PER_TILE;
static constexpr int GROUPED_BACKWARD_TILED_M =
    GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE * BACKWARD_M_PER_TILE *
    BACKWARD_WAVES;
static constexpr int GROUPED_BACKWARD_TILED_Q4_SWIZZLE = 16;

using grouped_backward_q4_shared_tile = backward_shared_b_tile<
    GROUPED_BACKWARD_TILED_N,
    GROUPED_BACKWARD_TILED_K,
    0,
    GROUPED_BACKWARD_TILED_Q4_SWIZZLE>;

template <bool FULL_ROWS>
static __device__ __forceinline__ void grouped_mmq_grad_input_q4_tile(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ expert_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        grouped_backward_q4_shared_tile & shared_b,
        int block_row_start,
        int row_end,
        int input_column_start) {
    constexpr int packed_row_bytes =
        GROUPED_BACKWARD_TILED_Q4_BLOCKS_PER_ROW * sizeof(block_q4_K);
    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int wave_row_start = block_row_start +
        wave * GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE * BACKWARD_M_PER_TILE;
    f32_accumulator accumulators
        [GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE]
        [GROUPED_BACKWARD_TILED_N_TILES];

#pragma unroll 1
    for (int output_start = 0;
         output_start < GROUPED_BACKWARD_TILED_Q4_OUT_FEATURES;
         output_start += GROUPED_BACKWARD_TILED_K) {
        const int local_input_column = 16 * (threadIdx.x & 7);
        const int first_k = threadIdx.x >> 3;
        const int second_k = first_k + 16;
        const int input_column = input_column_start + local_input_column;
        const int block_index = input_column / QK_K;
        const int value_index = input_column % QK_K;
        const int group = value_index >> 5;
        const int byte = (group >> 1) * 32 + (value_index & 31);
        const char * first_packed_row = expert_weight +
            static_cast<int64_t>(output_start + first_k) * packed_row_bytes;
        const char * second_packed_row = expert_weight +
            static_cast<int64_t>(output_start + second_k) * packed_row_bytes;
        const auto & first_block =
            reinterpret_cast<const block_q4_K *>(first_packed_row)[block_index];
        const auto & second_block =
            reinterpret_cast<const block_q4_K *>(second_packed_row)[block_index];
        const uint4 first_quants =
            *reinterpret_cast<const uint4 *>(first_block.qs + byte);
        const uint4 second_quants =
            *reinterpret_cast<const uint4 *>(second_block.qs + byte);
        __hip_bfloat16 values[16];
        decode_backward_tile_q4_preloaded(
            first_block, value_index, first_quants, values);
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            shared_b[(local_input_column + index) *
                GROUPED_BACKWARD_TILED_K + first_k] = values[index];
        }
        decode_backward_tile_q4_preloaded(
            second_block, value_index, second_quants, values);
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            shared_b[(local_input_column + index) *
                GROUPED_BACKWARD_TILED_K + second_k] = values[index];
        }
        __syncthreads();

#pragma unroll
        for (int k_tile = 0; k_tile < GROUPED_BACKWARD_TILED_K; k_tile += 16) {
            bf16_fragment a_fragments[GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE];
#pragma unroll
            for (int m_tile = 0;
                 m_tile < GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE;
                 ++m_tile) {
                __hip_bfloat16 * a = fragment_data(a_fragments[m_tile]);
                const int a_row = wave_row_start +
                    m_tile * BACKWARD_M_PER_TILE + c_row(lane);
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    if constexpr (FULL_ROWS) {
                        a[k] = grad_output[
                            static_cast<int64_t>(a_row) *
                                GROUPED_BACKWARD_TILED_Q4_OUT_FEATURES +
                            output_start + k_tile + k];
                    } else {
                        a[k] = a_row < row_end
                            ? grad_output[
                                static_cast<int64_t>(a_row) *
                                    GROUPED_BACKWARD_TILED_Q4_OUT_FEATURES +
                                output_start + k_tile + k]
                            : __float2bfloat16(0.0f);
                    }
                }
            }

#pragma unroll
            for (int n_tile = 0;
                 n_tile < GROUPED_BACKWARD_TILED_N_TILES;
                 n_tile += 2) {
                bf16_fragment b_first{};
                bf16_fragment b_second{};
                shared_b.load_fragment_vector(
                    b_first,
                    n_tile * BACKWARD_N_PER_TILE + c_row(lane),
                    k_tile);
                shared_b.load_fragment_vector(
                    b_second,
                    (n_tile + 1) * BACKWARD_N_PER_TILE + c_row(lane),
                    k_tile);
#pragma unroll
                for (int m_tile = 0;
                     m_tile < GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE;
                     ++m_tile) {
                    wmma_f32_16x16x16_bf16(
                        accumulators[m_tile][n_tile],
                        a_fragments[m_tile],
                        b_first);
                }
#pragma unroll
                for (int m_tile = 0;
                     m_tile < GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE;
                     ++m_tile) {
                    wmma_f32_16x16x16_bf16(
                        accumulators[m_tile][n_tile + 1],
                        a_fragments[m_tile],
                        b_second);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m_tile = 0;
         m_tile < GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE;
         ++m_tile) {
#pragma unroll
        for (int n_tile = 0;
             n_tile < GROUPED_BACKWARD_TILED_N_TILES;
             ++n_tile) {
#pragma unroll
            for (int element = 0; element < 8; ++element) {
                const int output_row = wave_row_start +
                    m_tile * BACKWARD_M_PER_TILE +
                    c_column(lane, element);
                const int output_column = input_column_start +
                    n_tile * BACKWARD_N_PER_TILE + c_row(lane);
                if constexpr (FULL_ROWS) {
                    grad_input[
                        static_cast<int64_t>(output_row) *
                            GROUPED_BACKWARD_TILED_Q4_IN_FEATURES +
                        output_column] = __float2bfloat16(
                            accumulators[m_tile][n_tile].values[element]);
                } else if (output_row < row_end) {
                    grad_input[
                        static_cast<int64_t>(output_row) *
                            GROUPED_BACKWARD_TILED_Q4_IN_FEATURES +
                        output_column] = __float2bfloat16(
                            accumulators[m_tile][n_tile].values[element]);
                }
            }
        }
    }
}

__launch_bounds__(BACKWARD_THREADS, 2)
static __global__ void grouped_mmq_grad_input_q4_tiled_kernel(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int rows,
        int64_t bytes_per_expert) {
    const int group = blockIdx.y;
    const int row_begin = group == 0 ? 0 : expert_offsets[group - 1];
    const int row_end = expert_offsets[group];
    const int64_t expert = expert_indices[group];
    if (expert < 0 || expert >= num_experts || row_begin < 0 ||
        row_end <= row_begin || row_end > rows) {
        return;
    }

    const int input_column_start = blockIdx.x * GROUPED_BACKWARD_TILED_N;
    const char * expert_weight =
        packed_weight + expert * bytes_per_expert;
    __shared__ grouped_backward_q4_shared_tile shared_b;

    int block_row_start = row_begin;
    for (; block_row_start + GROUPED_BACKWARD_TILED_M <= row_end;
         block_row_start += GROUPED_BACKWARD_TILED_M) {
        grouped_mmq_grad_input_q4_tile<true>(
            grad_output,
            expert_weight,
            grad_input,
            shared_b,
            block_row_start,
            row_end,
            input_column_start);
    }
    if (block_row_start < row_end) {
        grouped_mmq_grad_input_q4_tile<false>(
            grad_output,
            expert_weight,
            grad_input,
            shared_b,
            block_row_start,
            row_end,
            input_column_start);
    }
}

static inline void launch_grouped_mmq_grad_input_q4_tiled(
        const __hip_bfloat16 * grad_output,
        const char * packed_weight,
        __hip_bfloat16 * grad_input,
        const int64_t * expert_indices,
        const int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int64_t bytes_per_expert,
        hipStream_t stream) {
    const dim3 grid(
        GROUPED_BACKWARD_TILED_Q4_IN_FEATURES / GROUPED_BACKWARD_TILED_N,
        num_groups,
        1);
    const dim3 block(BACKWARD_THREADS, 1, 1);
    grouped_mmq_grad_input_q4_tiled_kernel<<<grid, block, 0, stream>>>(
        grad_output,
        packed_weight,
        grad_input,
        expert_indices,
        expert_offsets,
        num_experts,
        rows,
        bytes_per_expert);
}

static constexpr int GROUPED_BACKWARD_TILED_Q3_OUT_FEATURES = 512;
static constexpr int GROUPED_BACKWARD_TILED_Q3_IN_FEATURES = 2048;
static constexpr int GROUPED_BACKWARD_TILED_Q3_BLOCKS_PER_ROW = 8;
static constexpr int GROUPED_BACKWARD_TILED_Q3_PADDING = 8;

using grouped_backward_q3_shared_tile = backward_shared_b_tile<
    GROUPED_BACKWARD_TILED_N,
    GROUPED_BACKWARD_TILED_K,
    GROUPED_BACKWARD_TILED_Q3_PADDING,
    0>;

template <bool FULL_ROWS>
static __device__ __forceinline__ void grouped_mmq_pair_grad_input_q3_tile(
        const __hip_bfloat16 * __restrict__ first_grad_output,
        const __hip_bfloat16 * __restrict__ second_grad_output,
        const char * __restrict__ first_expert_weight,
        const char * __restrict__ second_expert_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        grouped_backward_q3_shared_tile & first_shared_b,
        grouped_backward_q3_shared_tile & second_shared_b,
        int block_row_start,
        int row_end,
        int input_column_start) {
    constexpr int packed_row_bytes =
        GROUPED_BACKWARD_TILED_Q3_BLOCKS_PER_ROW * sizeof(block_q3_K);
    constexpr int groups_per_row = GROUPED_BACKWARD_TILED_N / 16;
    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int wave_row_start = block_row_start +
        wave * GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE * BACKWARD_M_PER_TILE;
    f32_accumulator accumulators
        [GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE]
        [GROUPED_BACKWARD_TILED_N_TILES];

#pragma unroll 1
    for (int output_start = 0;
         output_start < GROUPED_BACKWARD_TILED_Q3_OUT_FEATURES;
         output_start += GROUPED_BACKWARD_TILED_K) {
#pragma unroll
        for (int group_index = threadIdx.x;
             group_index < groups_per_row * GROUPED_BACKWARD_TILED_K;
             group_index += BACKWARD_THREADS) {
            const int k = group_index / groups_per_row;
            const int local_input_column = 16 * (group_index % groups_per_row);
            const int input_column = input_column_start + local_input_column;
            const int output_column = output_start + k;
            const int block_index = input_column / QK_K;
            const int value_index = input_column % QK_K;
            const int64_t row_offset =
                static_cast<int64_t>(output_column) * packed_row_bytes;
            __hip_bfloat16 values[16];
            decode_backward_tile_group<GGML_TYPE_Q3_K, 16>(
                first_expert_weight + row_offset,
                block_index,
                value_index,
                values);
#pragma unroll
            for (int index = 0; index < 16; ++index) {
                first_shared_b[(local_input_column + index) *
                    GROUPED_BACKWARD_TILED_K + k] = values[index];
            }
            decode_backward_tile_group<GGML_TYPE_Q3_K, 16>(
                second_expert_weight + row_offset,
                block_index,
                value_index,
                values);
#pragma unroll
            for (int index = 0; index < 16; ++index) {
                second_shared_b[(local_input_column + index) *
                    GROUPED_BACKWARD_TILED_K + k] = values[index];
            }
        }
        __syncthreads();

#pragma unroll
        for (int k_tile = 0; k_tile < GROUPED_BACKWARD_TILED_K; k_tile += 16) {
            {
                bf16_fragment a_fragments
                    [GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE];
#pragma unroll
                for (int m_tile = 0;
                     m_tile < GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE;
                     ++m_tile) {
                    __hip_bfloat16 * a = fragment_data(a_fragments[m_tile]);
                    const int a_row = wave_row_start +
                        m_tile * BACKWARD_M_PER_TILE + c_row(lane);
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        if constexpr (FULL_ROWS) {
                            a[k] = first_grad_output[
                                static_cast<int64_t>(a_row) *
                                    GROUPED_BACKWARD_TILED_Q3_OUT_FEATURES +
                                output_start + k_tile + k];
                        } else {
                            a[k] = a_row < row_end
                                ? first_grad_output[
                                    static_cast<int64_t>(a_row) *
                                        GROUPED_BACKWARD_TILED_Q3_OUT_FEATURES +
                                    output_start + k_tile + k]
                                : __float2bfloat16(0.0f);
                        }
                    }
                }
#pragma unroll
                for (int n_tile = 0;
                     n_tile < GROUPED_BACKWARD_TILED_N_TILES;
                     n_tile += 2) {
                    bf16_fragment b_first{};
                    bf16_fragment b_second{};
                    first_shared_b.load_fragment_vector(
                        b_first,
                        n_tile * BACKWARD_N_PER_TILE + c_row(lane),
                        k_tile);
                    first_shared_b.load_fragment_vector(
                        b_second,
                        (n_tile + 1) * BACKWARD_N_PER_TILE + c_row(lane),
                        k_tile);
#pragma unroll
                    for (int m_tile = 0;
                         m_tile < GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE;
                         ++m_tile) {
                        wmma_f32_16x16x16_bf16(
                            accumulators[m_tile][n_tile],
                            a_fragments[m_tile],
                            b_first);
                    }
#pragma unroll
                    for (int m_tile = 0;
                         m_tile < GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE;
                         ++m_tile) {
                        wmma_f32_16x16x16_bf16(
                            accumulators[m_tile][n_tile + 1],
                            a_fragments[m_tile],
                            b_second);
                    }
                }
            }
            {
                bf16_fragment a_fragments
                    [GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE];
#pragma unroll
                for (int m_tile = 0;
                     m_tile < GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE;
                     ++m_tile) {
                    __hip_bfloat16 * a = fragment_data(a_fragments[m_tile]);
                    const int a_row = wave_row_start +
                        m_tile * BACKWARD_M_PER_TILE + c_row(lane);
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        if constexpr (FULL_ROWS) {
                            a[k] = second_grad_output[
                                static_cast<int64_t>(a_row) *
                                    GROUPED_BACKWARD_TILED_Q3_OUT_FEATURES +
                                output_start + k_tile + k];
                        } else {
                            a[k] = a_row < row_end
                                ? second_grad_output[
                                    static_cast<int64_t>(a_row) *
                                        GROUPED_BACKWARD_TILED_Q3_OUT_FEATURES +
                                    output_start + k_tile + k]
                                : __float2bfloat16(0.0f);
                        }
                    }
                }
#pragma unroll
                for (int n_tile = 0;
                     n_tile < GROUPED_BACKWARD_TILED_N_TILES;
                     n_tile += 2) {
                    bf16_fragment b_first{};
                    bf16_fragment b_second{};
                    second_shared_b.load_fragment_vector(
                        b_first,
                        n_tile * BACKWARD_N_PER_TILE + c_row(lane),
                        k_tile);
                    second_shared_b.load_fragment_vector(
                        b_second,
                        (n_tile + 1) * BACKWARD_N_PER_TILE + c_row(lane),
                        k_tile);
#pragma unroll
                    for (int m_tile = 0;
                         m_tile < GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE;
                         ++m_tile) {
                        wmma_f32_16x16x16_bf16(
                            accumulators[m_tile][n_tile],
                            a_fragments[m_tile],
                            b_first);
                    }
#pragma unroll
                    for (int m_tile = 0;
                         m_tile < GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE;
                         ++m_tile) {
                        wmma_f32_16x16x16_bf16(
                            accumulators[m_tile][n_tile + 1],
                            a_fragments[m_tile],
                            b_second);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m_tile = 0;
         m_tile < GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE;
         ++m_tile) {
#pragma unroll
        for (int n_tile = 0;
             n_tile < GROUPED_BACKWARD_TILED_N_TILES;
             ++n_tile) {
#pragma unroll
            for (int element = 0; element < 8; ++element) {
                const int output_row = wave_row_start +
                    m_tile * BACKWARD_M_PER_TILE +
                    c_column(lane, element);
                const int output_column = input_column_start +
                    n_tile * BACKWARD_N_PER_TILE + c_row(lane);
                if constexpr (FULL_ROWS) {
                    grad_input[
                        static_cast<int64_t>(output_row) *
                            GROUPED_BACKWARD_TILED_Q3_IN_FEATURES +
                        output_column] = __float2bfloat16(
                            accumulators[m_tile][n_tile].values[element]);
                } else if (output_row < row_end) {
                    grad_input[
                        static_cast<int64_t>(output_row) *
                            GROUPED_BACKWARD_TILED_Q3_IN_FEATURES +
                        output_column] = __float2bfloat16(
                            accumulators[m_tile][n_tile].values[element]);
                }
            }
        }
    }
}

__launch_bounds__(BACKWARD_THREADS, 1)
static __global__ void grouped_mmq_pair_grad_input_q3_tiled_kernel(
        const __hip_bfloat16 * __restrict__ first_grad_output,
        const __hip_bfloat16 * __restrict__ second_grad_output,
        const char * __restrict__ first_packed_weight,
        const char * __restrict__ second_packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int rows,
        int64_t bytes_per_expert) {
    const int group = blockIdx.y;
    const int row_begin = group == 0 ? 0 : expert_offsets[group - 1];
    const int row_end = expert_offsets[group];
    const int64_t expert = expert_indices[group];
    if (expert < 0 || expert >= num_experts || row_begin < 0 ||
        row_end <= row_begin || row_end > rows) {
        return;
    }

    const int input_column_start = blockIdx.x * GROUPED_BACKWARD_TILED_N;
    const char * first_expert_weight =
        first_packed_weight + expert * bytes_per_expert;
    const char * second_expert_weight =
        second_packed_weight + expert * bytes_per_expert;
    __shared__ grouped_backward_q3_shared_tile shared_b[2];

    int block_row_start = row_begin;
    for (; block_row_start + GROUPED_BACKWARD_TILED_M <= row_end;
         block_row_start += GROUPED_BACKWARD_TILED_M) {
        grouped_mmq_pair_grad_input_q3_tile<true>(
            first_grad_output,
            second_grad_output,
            first_expert_weight,
            second_expert_weight,
            grad_input,
            shared_b[0],
            shared_b[1],
            block_row_start,
            row_end,
            input_column_start);
    }
    if (block_row_start < row_end) {
        grouped_mmq_pair_grad_input_q3_tile<false>(
            first_grad_output,
            second_grad_output,
            first_expert_weight,
            second_expert_weight,
            grad_input,
            shared_b[0],
            shared_b[1],
            block_row_start,
            row_end,
            input_column_start);
    }
}

static inline void launch_grouped_mmq_pair_grad_input_q3_tiled(
        const __hip_bfloat16 * first_grad_output,
        const __hip_bfloat16 * second_grad_output,
        const char * first_packed_weight,
        const char * second_packed_weight,
        __hip_bfloat16 * grad_input,
        const int64_t * expert_indices,
        const int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int64_t bytes_per_expert,
        hipStream_t stream) {
    const dim3 grid(
        GROUPED_BACKWARD_TILED_Q3_IN_FEATURES / GROUPED_BACKWARD_TILED_N,
        num_groups,
        1);
    const dim3 block(BACKWARD_THREADS, 1, 1);
    grouped_mmq_pair_grad_input_q3_tiled_kernel<<<grid, block, 0, stream>>>(
        first_grad_output,
        second_grad_output,
        first_packed_weight,
        second_packed_weight,
        grad_input,
        expert_indices,
        expert_offsets,
        num_experts,
        rows,
        bytes_per_expert);
}

static constexpr int GROUPED_BACKWARD_SMALL_N_TILES = 4;
static constexpr int GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE = 1;
static constexpr int GROUPED_BACKWARD_SMALL_N =
    GROUPED_BACKWARD_SMALL_N_TILES * BACKWARD_N_PER_TILE;
static constexpr int GROUPED_BACKWARD_SMALL_M =
    GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE * BACKWARD_M_PER_TILE *
    BACKWARD_WAVES;

using grouped_backward_q4_small_shared_tile = backward_shared_b_tile<
    GROUPED_BACKWARD_SMALL_N,
    GROUPED_BACKWARD_TILED_K,
    0,
    GROUPED_BACKWARD_TILED_Q4_SWIZZLE>;
using grouped_backward_q3_small_shared_tile = backward_shared_b_tile<
    GROUPED_BACKWARD_SMALL_N,
    GROUPED_BACKWARD_TILED_K,
    GROUPED_BACKWARD_TILED_Q3_PADDING,
    0>;

template <
    int N_TILES,
    int M_TILES_PER_WAVE,
    int OUT_FEATURES,
    bool FULL_ROWS,
    typename SharedTile>
static __device__ __forceinline__ void grouped_backward_accumulate_projection(
        const __hip_bfloat16 * __restrict__ grad_output,
        const SharedTile & shared_b,
        f32_accumulator (&accumulators)[M_TILES_PER_WAVE][N_TILES],
        int wave_row_start,
        int row_end,
        int output_start,
        int lane) {
#pragma unroll
    for (int k_tile = 0; k_tile < GROUPED_BACKWARD_TILED_K; k_tile += 16) {
        bf16_fragment a_fragments[M_TILES_PER_WAVE];
#pragma unroll
        for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
            __hip_bfloat16 * a = fragment_data(a_fragments[m_tile]);
            const int a_row = wave_row_start +
                m_tile * BACKWARD_M_PER_TILE + c_row(lane);
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                if constexpr (FULL_ROWS) {
                    a[k] = grad_output[
                        static_cast<int64_t>(a_row) * OUT_FEATURES +
                        output_start + k_tile + k];
                } else {
                    a[k] = a_row < row_end
                        ? grad_output[
                            static_cast<int64_t>(a_row) * OUT_FEATURES +
                            output_start + k_tile + k]
                        : __float2bfloat16(0.0f);
                }
            }
        }
#pragma unroll
        for (int n_tile = 0; n_tile < N_TILES; n_tile += 2) {
            bf16_fragment b_first{};
            bf16_fragment b_second{};
            shared_b.load_fragment_vector(
                b_first,
                n_tile * BACKWARD_N_PER_TILE + c_row(lane),
                k_tile);
            shared_b.load_fragment_vector(
                b_second,
                (n_tile + 1) * BACKWARD_N_PER_TILE + c_row(lane),
                k_tile);
#pragma unroll
            for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
                wmma_f32_16x16x16_bf16(
                    accumulators[m_tile][n_tile],
                    a_fragments[m_tile],
                    b_first);
            }
#pragma unroll
            for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
                wmma_f32_16x16x16_bf16(
                    accumulators[m_tile][n_tile + 1],
                    a_fragments[m_tile],
                    b_second);
            }
        }
    }
}

template <
    int N_TILES,
    int M_TILES_PER_WAVE,
    int IN_FEATURES,
    bool FULL_ROWS>
static __device__ __forceinline__ void grouped_backward_store_tile(
        __hip_bfloat16 * __restrict__ grad_input,
        const f32_accumulator (&accumulators)[M_TILES_PER_WAVE][N_TILES],
        int wave_row_start,
        int row_end,
        int input_column_start,
        int lane) {
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
                if constexpr (FULL_ROWS) {
                    grad_input[
                        static_cast<int64_t>(output_row) * IN_FEATURES +
                        output_column] = __float2bfloat16(
                            accumulators[m_tile][n_tile].values[element]);
                } else if (output_row < row_end) {
                    grad_input[
                        static_cast<int64_t>(output_row) * IN_FEATURES +
                        output_column] = __float2bfloat16(
                            accumulators[m_tile][n_tile].values[element]);
                }
            }
        }
    }
}

template <int M_TILES_PER_WAVE, bool FULL_ROWS>
static __device__ __forceinline__ void grouped_mmq_grad_input_q4_small_tile(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ expert_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        grouped_backward_q4_small_shared_tile & shared_b,
        int block_row_start,
        int row_end,
        int input_column_start) {
    constexpr int packed_row_bytes =
        GROUPED_BACKWARD_TILED_Q4_BLOCKS_PER_ROW * sizeof(block_q4_K);
    constexpr int groups_per_row = GROUPED_BACKWARD_SMALL_N / 16;
    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int wave_row_start = block_row_start +
        wave * M_TILES_PER_WAVE * BACKWARD_M_PER_TILE;
    f32_accumulator accumulators
        [M_TILES_PER_WAVE]
        [GROUPED_BACKWARD_SMALL_N_TILES];

#pragma unroll 1
    for (int output_start = 0;
         output_start < GROUPED_BACKWARD_TILED_Q4_OUT_FEATURES;
         output_start += GROUPED_BACKWARD_TILED_K) {
        const int group_index = threadIdx.x;
        const int k = group_index / groups_per_row;
        const int local_input_column = 16 * (group_index % groups_per_row);
        const int input_column = input_column_start + local_input_column;
        const int block_index = input_column / QK_K;
        const int value_index = input_column % QK_K;
        const char * packed_row = expert_weight +
            static_cast<int64_t>(output_start + k) * packed_row_bytes;
        __hip_bfloat16 values[16];
        decode_backward_tile_group<GGML_TYPE_Q4_K, 16>(
            packed_row,
            block_index,
            value_index,
            values);
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            shared_b[(local_input_column + index) *
                GROUPED_BACKWARD_TILED_K + k] = values[index];
        }
        __syncthreads();
        grouped_backward_accumulate_projection<
            GROUPED_BACKWARD_SMALL_N_TILES,
            M_TILES_PER_WAVE,
            GROUPED_BACKWARD_TILED_Q4_OUT_FEATURES,
            FULL_ROWS>(
            grad_output,
            shared_b,
            accumulators,
            wave_row_start,
            row_end,
            output_start,
            lane);
        __syncthreads();
    }

    grouped_backward_store_tile<
        GROUPED_BACKWARD_SMALL_N_TILES,
        M_TILES_PER_WAVE,
        GROUPED_BACKWARD_TILED_Q4_IN_FEATURES,
        FULL_ROWS>(
        grad_input,
        accumulators,
        wave_row_start,
        row_end,
        input_column_start,
        lane);
}

__launch_bounds__(BACKWARD_THREADS, 2)
static __global__ void grouped_mmq_grad_input_q4_small_kernel(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int rows,
        int64_t bytes_per_expert) {
    const int group = blockIdx.y;
    const int row_begin = group == 0 ? 0 : expert_offsets[group - 1];
    const int row_end = expert_offsets[group];
    const int64_t expert = expert_indices[group];
    if (expert < 0 || expert >= num_experts || row_begin < 0 ||
        row_end <= row_begin || row_end > rows) {
        return;
    }

    const int input_column_start = blockIdx.x * GROUPED_BACKWARD_SMALL_N;
    const char * expert_weight = packed_weight + expert * bytes_per_expert;
    __shared__ grouped_backward_q4_small_shared_tile shared_b;

    int block_row_start = row_begin;
    for (; block_row_start + GROUPED_BACKWARD_SMALL_M <= row_end;
         block_row_start += GROUPED_BACKWARD_SMALL_M) {
        grouped_mmq_grad_input_q4_small_tile<
            GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE, true>(
            grad_output,
            expert_weight,
            grad_input,
            shared_b,
            block_row_start,
            row_end,
            input_column_start);
    }
    if (block_row_start < row_end) {
        grouped_mmq_grad_input_q4_small_tile<
            GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE, false>(
            grad_output,
            expert_weight,
            grad_input,
            shared_b,
            block_row_start,
            row_end,
            input_column_start);
    }
}

static inline void launch_grouped_mmq_grad_input_q4_small(
        const __hip_bfloat16 * grad_output,
        const char * packed_weight,
        __hip_bfloat16 * grad_input,
        const int64_t * expert_indices,
        const int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int64_t bytes_per_expert,
        hipStream_t stream) {
    const dim3 grid(
        GROUPED_BACKWARD_TILED_Q4_IN_FEATURES / GROUPED_BACKWARD_SMALL_N,
        num_groups,
        1);
    const dim3 block(BACKWARD_THREADS, 1, 1);
    grouped_mmq_grad_input_q4_small_kernel<<<grid, block, 0, stream>>>(
        grad_output,
        packed_weight,
        grad_input,
        expert_indices,
        expert_offsets,
        num_experts,
        rows,
        bytes_per_expert);
}

static constexpr int GROUPED_BACKWARD_SMALL_Q4_S2_M_TILES_PER_WAVE = 2;
static constexpr int GROUPED_BACKWARD_SMALL_Q4_S2_M =
    GROUPED_BACKWARD_SMALL_Q4_S2_M_TILES_PER_WAVE * BACKWARD_M_PER_TILE *
    BACKWARD_WAVES;

__launch_bounds__(BACKWARD_THREADS, 2)
static __global__ void grouped_mmq_grad_input_q4_small_s2_kernel(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int rows,
        int64_t bytes_per_expert) {
    const int group = blockIdx.y;
    const int row_begin = group == 0 ? 0 : expert_offsets[group - 1];
    const int row_end = expert_offsets[group];
    const int64_t expert = expert_indices[group];
    if (expert < 0 || expert >= num_experts || row_begin < 0 ||
        row_end <= row_begin || row_end > rows) {
        return;
    }

    const int input_column_start = blockIdx.x * GROUPED_BACKWARD_SMALL_N;
    const char * expert_weight = packed_weight + expert * bytes_per_expert;
    __shared__ grouped_backward_q4_small_shared_tile shared_b;

    int block_row_start = row_begin;
    for (; block_row_start + GROUPED_BACKWARD_SMALL_Q4_S2_M <= row_end;
         block_row_start += GROUPED_BACKWARD_SMALL_Q4_S2_M) {
        grouped_mmq_grad_input_q4_small_tile<
            GROUPED_BACKWARD_SMALL_Q4_S2_M_TILES_PER_WAVE, true>(
            grad_output,
            expert_weight,
            grad_input,
            shared_b,
            block_row_start,
            row_end,
            input_column_start);
    }
    if (block_row_start < row_end) {
        grouped_mmq_grad_input_q4_small_tile<
            GROUPED_BACKWARD_SMALL_Q4_S2_M_TILES_PER_WAVE, false>(
            grad_output,
            expert_weight,
            grad_input,
            shared_b,
            block_row_start,
            row_end,
            input_column_start);
    }
}

static inline void launch_grouped_mmq_grad_input_q4_small_s2(
        const __hip_bfloat16 * grad_output,
        const char * packed_weight,
        __hip_bfloat16 * grad_input,
        const int64_t * expert_indices,
        const int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int64_t bytes_per_expert,
        hipStream_t stream) {
    const dim3 grid(
        GROUPED_BACKWARD_TILED_Q4_IN_FEATURES / GROUPED_BACKWARD_SMALL_N,
        num_groups,
        1);
    const dim3 block(BACKWARD_THREADS, 1, 1);
    grouped_mmq_grad_input_q4_small_s2_kernel<<<grid, block, 0, stream>>>(
        grad_output,
        packed_weight,
        grad_input,
        expert_indices,
        expert_offsets,
        num_experts,
        rows,
        bytes_per_expert);
}

template <bool FULL_ROWS>
static __device__ __forceinline__ void grouped_mmq_pair_grad_input_q3_small_tile(
        const __hip_bfloat16 * __restrict__ first_grad_output,
        const __hip_bfloat16 * __restrict__ second_grad_output,
        const char * __restrict__ first_expert_weight,
        const char * __restrict__ second_expert_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        grouped_backward_q3_small_shared_tile & first_shared_b,
        grouped_backward_q3_small_shared_tile & second_shared_b,
        int block_row_start,
        int row_end,
        int input_column_start) {
    constexpr int packed_row_bytes =
        GROUPED_BACKWARD_TILED_Q3_BLOCKS_PER_ROW * sizeof(block_q3_K);
    constexpr int groups_per_row = GROUPED_BACKWARD_SMALL_N / 16;
    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int wave_row_start = block_row_start + wave * BACKWARD_M_PER_TILE;
    f32_accumulator accumulators
        [GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE]
        [GROUPED_BACKWARD_SMALL_N_TILES];

#pragma unroll 1
    for (int output_start = 0;
         output_start < GROUPED_BACKWARD_TILED_Q3_OUT_FEATURES;
         output_start += GROUPED_BACKWARD_TILED_K) {
        const int group_index = threadIdx.x;
        const int k = group_index / groups_per_row;
        const int local_input_column = 16 * (group_index % groups_per_row);
        const int input_column = input_column_start + local_input_column;
        const int block_index = input_column / QK_K;
        const int value_index = input_column % QK_K;
        const int64_t row_offset =
            static_cast<int64_t>(output_start + k) * packed_row_bytes;
        __hip_bfloat16 values[16];
        decode_backward_tile_group<GGML_TYPE_Q3_K, 16>(
            first_expert_weight + row_offset,
            block_index,
            value_index,
            values);
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            first_shared_b[(local_input_column + index) *
                GROUPED_BACKWARD_TILED_K + k] = values[index];
        }
        decode_backward_tile_group<GGML_TYPE_Q3_K, 16>(
            second_expert_weight + row_offset,
            block_index,
            value_index,
            values);
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            second_shared_b[(local_input_column + index) *
                GROUPED_BACKWARD_TILED_K + k] = values[index];
        }
        __syncthreads();
        grouped_backward_accumulate_projection<
            GROUPED_BACKWARD_SMALL_N_TILES,
            GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE,
            GROUPED_BACKWARD_TILED_Q3_OUT_FEATURES,
            FULL_ROWS>(
            first_grad_output,
            first_shared_b,
            accumulators,
            wave_row_start,
            row_end,
            output_start,
            lane);
        grouped_backward_accumulate_projection<
            GROUPED_BACKWARD_SMALL_N_TILES,
            GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE,
            GROUPED_BACKWARD_TILED_Q3_OUT_FEATURES,
            FULL_ROWS>(
            second_grad_output,
            second_shared_b,
            accumulators,
            wave_row_start,
            row_end,
            output_start,
            lane);
        __syncthreads();
    }

    grouped_backward_store_tile<
        GROUPED_BACKWARD_SMALL_N_TILES,
        GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE,
        GROUPED_BACKWARD_TILED_Q3_IN_FEATURES,
        FULL_ROWS>(
        grad_input,
        accumulators,
        wave_row_start,
        row_end,
        input_column_start,
        lane);
}

__launch_bounds__(BACKWARD_THREADS, 2)
static __global__ void grouped_mmq_pair_grad_input_q3_small_kernel(
        const __hip_bfloat16 * __restrict__ first_grad_output,
        const __hip_bfloat16 * __restrict__ second_grad_output,
        const char * __restrict__ first_packed_weight,
        const char * __restrict__ second_packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int rows,
        int64_t bytes_per_expert) {
    const int group = blockIdx.y;
    const int row_begin = group == 0 ? 0 : expert_offsets[group - 1];
    const int row_end = expert_offsets[group];
    const int64_t expert = expert_indices[group];
    if (expert < 0 || expert >= num_experts || row_begin < 0 ||
        row_end <= row_begin || row_end > rows) {
        return;
    }

    const int input_column_start = blockIdx.x * GROUPED_BACKWARD_SMALL_N;
    const char * first_expert_weight =
        first_packed_weight + expert * bytes_per_expert;
    const char * second_expert_weight =
        second_packed_weight + expert * bytes_per_expert;
    __shared__ grouped_backward_q3_small_shared_tile shared_b[2];

    int block_row_start = row_begin;
    for (; block_row_start + GROUPED_BACKWARD_SMALL_M <= row_end;
         block_row_start += GROUPED_BACKWARD_SMALL_M) {
        grouped_mmq_pair_grad_input_q3_small_tile<true>(
            first_grad_output,
            second_grad_output,
            first_expert_weight,
            second_expert_weight,
            grad_input,
            shared_b[0],
            shared_b[1],
            block_row_start,
            row_end,
            input_column_start);
    }
    if (block_row_start < row_end) {
        grouped_mmq_pair_grad_input_q3_small_tile<false>(
            first_grad_output,
            second_grad_output,
            first_expert_weight,
            second_expert_weight,
            grad_input,
            shared_b[0],
            shared_b[1],
            block_row_start,
            row_end,
            input_column_start);
    }
}

static inline void launch_grouped_mmq_pair_grad_input_q3_small(
        const __hip_bfloat16 * first_grad_output,
        const __hip_bfloat16 * second_grad_output,
        const char * first_packed_weight,
        const char * second_packed_weight,
        __hip_bfloat16 * grad_input,
        const int64_t * expert_indices,
        const int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int64_t bytes_per_expert,
        hipStream_t stream) {
    const dim3 grid(
        GROUPED_BACKWARD_TILED_Q3_IN_FEATURES / GROUPED_BACKWARD_SMALL_N,
        num_groups,
        1);
    const dim3 block(BACKWARD_THREADS, 1, 1);
    grouped_mmq_pair_grad_input_q3_small_kernel<<<grid, block, 0, stream>>>(
        first_grad_output,
        second_grad_output,
        first_packed_weight,
        second_packed_weight,
        grad_input,
        expert_indices,
        expert_offsets,
        num_experts,
        rows,
        bytes_per_expert);
}

static constexpr int GROUPED_BACKWARD_TILED_Q5_OUT_FEATURES = 2048;
static constexpr int GROUPED_BACKWARD_TILED_Q5_IN_FEATURES = 512;
static constexpr int GROUPED_BACKWARD_TILED_Q5_BLOCKS_PER_ROW = 2;
static constexpr int GROUPED_BACKWARD_TILED_Q5_SWIZZLE = 4;

using grouped_backward_q5_shared_tile = backward_shared_b_tile<
    GROUPED_BACKWARD_TILED_N,
    GROUPED_BACKWARD_TILED_K,
    0,
    GROUPED_BACKWARD_TILED_Q5_SWIZZLE>;
using grouped_backward_q5_small_shared_tile = backward_shared_b_tile<
    GROUPED_BACKWARD_SMALL_N,
    GROUPED_BACKWARD_TILED_K,
    0,
    GROUPED_BACKWARD_TILED_Q5_SWIZZLE>;

template <
    int N_TILES,
    int M_TILES_PER_WAVE,
    bool FULL_ROWS,
    typename SharedTile>
static __device__ __forceinline__ void grouped_mmq_grad_input_q5_tile(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ expert_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        SharedTile & shared_b,
        int block_row_start,
        int row_end,
        int input_column_start) {
    constexpr int packed_row_bytes =
        GROUPED_BACKWARD_TILED_Q5_BLOCKS_PER_ROW * sizeof(block_q5_K);
    constexpr int n_per_block = N_TILES * BACKWARD_N_PER_TILE;
    constexpr int groups_per_row = n_per_block / 16;
    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int wave_row_start = block_row_start +
        wave * M_TILES_PER_WAVE * BACKWARD_M_PER_TILE;
    f32_accumulator accumulators[M_TILES_PER_WAVE][N_TILES];

#pragma unroll 1
    for (int output_start = 0;
         output_start < GROUPED_BACKWARD_TILED_Q5_OUT_FEATURES;
         output_start += GROUPED_BACKWARD_TILED_K) {
        if constexpr (N_TILES == GROUPED_BACKWARD_TILED_N_TILES) {
            const int local_input_column = 16 * (threadIdx.x & 7);
            const int first_k = threadIdx.x >> 3;
            const int second_k = first_k + 16;
            const int input_column = input_column_start + local_input_column;
            const int block_index = input_column / QK_K;
            const int value_index = input_column % QK_K;
            const int group = value_index >> 5;
            const int low_byte =
                (group >> 1) * 32 + (value_index & 31);
            const int high_byte = value_index & 31;
            const char * first_packed_row = expert_weight +
                static_cast<int64_t>(output_start + first_k) * packed_row_bytes;
            const char * second_packed_row = expert_weight +
                static_cast<int64_t>(output_start + second_k) * packed_row_bytes;
            const auto & first_block =
                reinterpret_cast<const block_q5_K *>(first_packed_row)[block_index];
            const auto & second_block =
                reinterpret_cast<const block_q5_K *>(second_packed_row)[block_index];
            const uint4 first_low =
                *reinterpret_cast<const uint4 *>(first_block.qs + low_byte);
            const uint4 first_high =
                *reinterpret_cast<const uint4 *>(first_block.qh + high_byte);
            const uint4 second_low =
                *reinterpret_cast<const uint4 *>(second_block.qs + low_byte);
            const uint4 second_high =
                *reinterpret_cast<const uint4 *>(second_block.qh + high_byte);
            __hip_bfloat16 values[16];
            decode_backward_tile_q5_preloaded<false>(
                first_block,
                value_index,
                first_low,
                first_high,
                values);
#pragma unroll
            for (int index = 0; index < 16; ++index) {
                shared_b[(local_input_column + index) *
                    GROUPED_BACKWARD_TILED_K + first_k] = values[index];
            }
            decode_backward_tile_q5_preloaded<false>(
                second_block,
                value_index,
                second_low,
                second_high,
                values);
#pragma unroll
            for (int index = 0; index < 16; ++index) {
                shared_b[(local_input_column + index) *
                    GROUPED_BACKWARD_TILED_K + second_k] = values[index];
            }
        } else {
#pragma unroll
            for (int group_index = threadIdx.x;
                 group_index < groups_per_row * GROUPED_BACKWARD_TILED_K;
                 group_index += BACKWARD_THREADS) {
                const int k = group_index / groups_per_row;
                const int local_input_column =
                    16 * (group_index % groups_per_row);
                const int input_column = input_column_start + local_input_column;
                const int block_index = input_column / QK_K;
                const int value_index = input_column % QK_K;
                const char * packed_row = expert_weight +
                    static_cast<int64_t>(output_start + k) * packed_row_bytes;
                __hip_bfloat16 values[16];
                decode_backward_tile_group<GGML_TYPE_Q5_K, 16>(
                    packed_row,
                    block_index,
                    value_index,
                    values);
#pragma unroll
                for (int index = 0; index < 16; ++index) {
                    shared_b[(local_input_column + index) *
                        GROUPED_BACKWARD_TILED_K + k] = values[index];
                }
            }
        }
        __syncthreads();
        grouped_backward_accumulate_projection<
            N_TILES,
            M_TILES_PER_WAVE,
            GROUPED_BACKWARD_TILED_Q5_OUT_FEATURES,
            FULL_ROWS>(
            grad_output,
            shared_b,
            accumulators,
            wave_row_start,
            row_end,
            output_start,
            lane);
        __syncthreads();
    }

    grouped_backward_store_tile<
        N_TILES,
        M_TILES_PER_WAVE,
        GROUPED_BACKWARD_TILED_Q5_IN_FEATURES,
        FULL_ROWS>(
        grad_input,
        accumulators,
        wave_row_start,
        row_end,
        input_column_start,
        lane);
}

__launch_bounds__(BACKWARD_THREADS, 2)
static __global__ void grouped_mmq_grad_input_q5_tiled_kernel(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int rows,
        int64_t bytes_per_expert) {
    const int group = blockIdx.y;
    const int row_begin = group == 0 ? 0 : expert_offsets[group - 1];
    const int row_end = expert_offsets[group];
    const int64_t expert = expert_indices[group];
    if (expert < 0 || expert >= num_experts || row_begin < 0 ||
        row_end <= row_begin || row_end > rows) {
        return;
    }

    const int input_column_start = blockIdx.x * GROUPED_BACKWARD_TILED_N;
    const char * expert_weight = packed_weight + expert * bytes_per_expert;
    __shared__ grouped_backward_q5_shared_tile shared_b;

    int block_row_start = row_begin;
    for (; block_row_start + GROUPED_BACKWARD_TILED_M <= row_end;
         block_row_start += GROUPED_BACKWARD_TILED_M) {
        grouped_mmq_grad_input_q5_tile<
            GROUPED_BACKWARD_TILED_N_TILES,
            GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE,
            true>(
            grad_output,
            expert_weight,
            grad_input,
            shared_b,
            block_row_start,
            row_end,
            input_column_start);
    }
    if (block_row_start < row_end) {
        grouped_mmq_grad_input_q5_tile<
            GROUPED_BACKWARD_TILED_N_TILES,
            GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE,
            false>(
            grad_output,
            expert_weight,
            grad_input,
            shared_b,
            block_row_start,
            row_end,
            input_column_start);
    }
}

static inline void launch_grouped_mmq_grad_input_q5_tiled(
        const __hip_bfloat16 * grad_output,
        const char * packed_weight,
        __hip_bfloat16 * grad_input,
        const int64_t * expert_indices,
        const int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int64_t bytes_per_expert,
        hipStream_t stream) {
    const dim3 grid(
        GROUPED_BACKWARD_TILED_Q5_IN_FEATURES / GROUPED_BACKWARD_TILED_N,
        num_groups,
        1);
    const dim3 block(BACKWARD_THREADS, 1, 1);
    grouped_mmq_grad_input_q5_tiled_kernel<<<grid, block, 0, stream>>>(
        grad_output,
        packed_weight,
        grad_input,
        expert_indices,
        expert_offsets,
        num_experts,
        rows,
        bytes_per_expert);
}

__launch_bounds__(BACKWARD_THREADS, 2)
static __global__ void grouped_mmq_grad_input_q5_small_kernel(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int rows,
        int64_t bytes_per_expert) {
    const int group = blockIdx.y;
    const int row_begin = group == 0 ? 0 : expert_offsets[group - 1];
    const int row_end = expert_offsets[group];
    const int64_t expert = expert_indices[group];
    if (expert < 0 || expert >= num_experts || row_begin < 0 ||
        row_end <= row_begin || row_end > rows) {
        return;
    }

    const int input_column_start = blockIdx.x * GROUPED_BACKWARD_SMALL_N;
    const char * expert_weight = packed_weight + expert * bytes_per_expert;
    __shared__ grouped_backward_q5_small_shared_tile shared_b;

    int block_row_start = row_begin;
    for (; block_row_start + GROUPED_BACKWARD_SMALL_M <= row_end;
         block_row_start += GROUPED_BACKWARD_SMALL_M) {
        grouped_mmq_grad_input_q5_tile<
            GROUPED_BACKWARD_SMALL_N_TILES,
            GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE,
            true>(
            grad_output,
            expert_weight,
            grad_input,
            shared_b,
            block_row_start,
            row_end,
            input_column_start);
    }
    if (block_row_start < row_end) {
        grouped_mmq_grad_input_q5_tile<
            GROUPED_BACKWARD_SMALL_N_TILES,
            GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE,
            false>(
            grad_output,
            expert_weight,
            grad_input,
            shared_b,
            block_row_start,
            row_end,
            input_column_start);
    }
}

static inline void launch_grouped_mmq_grad_input_q5_small(
        const __hip_bfloat16 * grad_output,
        const char * packed_weight,
        __hip_bfloat16 * grad_input,
        const int64_t * expert_indices,
        const int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int64_t bytes_per_expert,
        hipStream_t stream) {
    const dim3 grid(
        GROUPED_BACKWARD_TILED_Q5_IN_FEATURES / GROUPED_BACKWARD_SMALL_N,
        num_groups,
        1);
    const dim3 block(BACKWARD_THREADS, 1, 1);
    grouped_mmq_grad_input_q5_small_kernel<<<grid, block, 0, stream>>>(
        grad_output,
        packed_weight,
        grad_input,
        expert_indices,
        expert_offsets,
        num_experts,
        rows,
        bytes_per_expert);
}


} // namespace torch_ggml_ops::ck
