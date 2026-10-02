#pragma once

#include "grouped_mmq_backward_tiled_common.cuh"

namespace torch_ggml_ops::ck {

static constexpr int GROUPED_BACKWARD_TILED_Q5_OUT_FEATURES = 2048;
static constexpr int GROUPED_BACKWARD_TILED_Q5_IN_FEATURES = 512;
static constexpr int GROUPED_BACKWARD_TILED_Q5_BLOCKS_PER_ROW = 2;
static constexpr int GROUPED_BACKWARD_TILED_Q5_SWIZZLE = 8;
static constexpr int GROUPED_BACKWARD_TILED_Q5_SMALL_SWIZZLE = 4;

using grouped_backward_q5_shared_tile = backward_shared_b_tile<
    GROUPED_BACKWARD_TILED_N,
    GROUPED_BACKWARD_TILED_K,
    0,
    GROUPED_BACKWARD_TILED_Q5_SWIZZLE>;
using grouped_backward_q5_small_shared_tile = backward_shared_b_tile<
    GROUPED_BACKWARD_SMALL_N,
    GROUPED_BACKWARD_TILED_K,
    0,
    GROUPED_BACKWARD_TILED_Q5_SMALL_SWIZZLE>;

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
        int input_column_start,
        bool skip_inactive_m_tiles = false) {
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
            lane,
            skip_inactive_m_tiles);
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
        lane,
        skip_inactive_m_tiles);
}

static __device__ __forceinline__ void grouped_mmq_grad_input_q5_small_body(
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

static __device__ __forceinline__ void grouped_mmq_grad_input_q5_row_task_body(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        const int32_t * __restrict__ task_count,
        const int32_t * __restrict__ task_experts,
        const int32_t * __restrict__ task_row_starts,
        const int32_t * __restrict__ task_row_ends,
        int64_t bytes_per_expert) {
    const int task = blockIdx.y;
    if (task >= task_count[0]) {
        return;
    }
    const int expert = task_experts[task];
    const int row_start = task_row_starts[task];
    const int row_end = task_row_ends[task];
    const int input_column_start = blockIdx.x * GROUPED_BACKWARD_TILED_N;
    const char * expert_weight = packed_weight +
        static_cast<int64_t>(expert) * bytes_per_expert;
    __shared__ grouped_backward_q5_shared_tile shared_b;
    grouped_mmq_grad_input_q5_tile<
        GROUPED_BACKWARD_TILED_N_TILES,
        GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE,
        false>(
        grad_output,
        expert_weight,
        grad_input,
        shared_b,
        row_start,
        row_end,
        input_column_start,
        true);
}

} // namespace torch_ggml_ops::ck
