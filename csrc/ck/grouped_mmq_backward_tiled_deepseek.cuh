#pragma once

#include "mmq_backward.cuh"

namespace torch_ggml_ops::ck {

using grouped_backward_deepseek_single_shared_tile = backward_shared_b_tile<
    GROUPED_BACKWARD_SMALL_N,
    GROUPED_BACKWARD_TILED_K,
    0,
    0>;
using grouped_backward_deepseek_pair_shared_tile = backward_shared_b_tile<
    GROUPED_BACKWARD_SMALL_N,
    GROUPED_BACKWARD_TILED_K,
    0,
    4>;

template <
    ggml_type type,
    int OUT_FEATURES,
    int IN_FEATURES,
    int BLOCKS_PER_WEIGHT_ROW,
    int M_TILES_PER_WAVE,
    int REDUCTION_UNROLL,
    bool SKIP_INACTIVE_M_TILES,
    bool FULL_ROWS>
static __device__ __forceinline__ void grouped_mmq_grad_input_deepseek_tile(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ expert_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        grouped_backward_deepseek_single_shared_tile & shared_b,
        int block_row_start,
        int row_end,
        int input_column_start) {
    constexpr int packed_row_bytes =
        BLOCKS_PER_WEIGHT_ROW * gguf_block_bytes<type>();
    constexpr int groups_per_row = GROUPED_BACKWARD_SMALL_N / 16;
    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int wave_row_start = block_row_start +
        wave * M_TILES_PER_WAVE * BACKWARD_M_PER_TILE;
    f32_accumulator accumulators
        [M_TILES_PER_WAVE]
        [GROUPED_BACKWARD_SMALL_N_TILES];

#pragma unroll REDUCTION_UNROLL
    for (int output_start = 0;
         output_start < OUT_FEATURES;
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
        decode_backward_tile_group<type, 16>(
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
            OUT_FEATURES,
            FULL_ROWS>(
            grad_output,
            shared_b,
            accumulators,
            wave_row_start,
            row_end,
            output_start,
            lane,
            SKIP_INACTIVE_M_TILES);
        __syncthreads();
    }

    grouped_backward_store_tile<
        GROUPED_BACKWARD_SMALL_N_TILES,
        M_TILES_PER_WAVE,
        IN_FEATURES,
        FULL_ROWS>(
        grad_input,
        accumulators,
        wave_row_start,
        row_end,
        input_column_start,
        lane,
        SKIP_INACTIVE_M_TILES);
}

template <
    ggml_type type,
    int OUT_FEATURES,
    int IN_FEATURES,
    int BLOCKS_PER_WEIGHT_ROW,
    int M_TILES_PER_WAVE,
    int REDUCTION_UNROLL,
    bool SKIP_INACTIVE_M_TILES>
static __device__ __forceinline__ void grouped_mmq_grad_input_deepseek_body(
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

    constexpr int m_per_block =
        M_TILES_PER_WAVE * BACKWARD_M_PER_TILE * BACKWARD_WAVES;
    const int input_column_start = blockIdx.x * GROUPED_BACKWARD_SMALL_N;
    const char * expert_weight = packed_weight + expert * bytes_per_expert;
    __shared__ grouped_backward_deepseek_single_shared_tile shared_b;

    int block_row_start = row_begin;
    for (; block_row_start + m_per_block <= row_end;
         block_row_start += m_per_block) {
        grouped_mmq_grad_input_deepseek_tile<
            type,
            OUT_FEATURES,
            IN_FEATURES,
            BLOCKS_PER_WEIGHT_ROW,
            M_TILES_PER_WAVE,
            REDUCTION_UNROLL,
            SKIP_INACTIVE_M_TILES,
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
        grouped_mmq_grad_input_deepseek_tile<
            type,
            OUT_FEATURES,
            IN_FEATURES,
            BLOCKS_PER_WEIGHT_ROW,
            M_TILES_PER_WAVE,
            REDUCTION_UNROLL,
            SKIP_INACTIVE_M_TILES,
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

template <
    ggml_type type,
    int OUT_FEATURES,
    int IN_FEATURES,
    int BLOCKS_PER_WEIGHT_ROW,
    int M_TILES_PER_WAVE,
    bool SKIP_INACTIVE_M_TILES,
    bool FULL_ROWS>
static __device__ __forceinline__ void grouped_mmq_pair_grad_input_deepseek_tile(
        const __hip_bfloat16 * __restrict__ first_grad_output,
        const __hip_bfloat16 * __restrict__ second_grad_output,
        const char * __restrict__ first_expert_weight,
        const char * __restrict__ second_expert_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        grouped_backward_deepseek_pair_shared_tile & first_shared_b,
        grouped_backward_deepseek_pair_shared_tile & second_shared_b,
        int block_row_start,
        int row_end,
        int input_column_start) {
    constexpr int packed_row_bytes =
        BLOCKS_PER_WEIGHT_ROW * gguf_block_bytes<type>();
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
         output_start < OUT_FEATURES;
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
        decode_backward_tile_group<type, 16>(
            first_expert_weight + row_offset,
            block_index,
            value_index,
            values);
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            first_shared_b[(local_input_column + index) *
                GROUPED_BACKWARD_TILED_K + k] = values[index];
        }
        decode_backward_tile_group<type, 16>(
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
            M_TILES_PER_WAVE,
            OUT_FEATURES,
            FULL_ROWS>(
            first_grad_output,
            first_shared_b,
            accumulators,
            wave_row_start,
            row_end,
            output_start,
            lane,
            SKIP_INACTIVE_M_TILES);
        grouped_backward_accumulate_projection<
            GROUPED_BACKWARD_SMALL_N_TILES,
            M_TILES_PER_WAVE,
            OUT_FEATURES,
            FULL_ROWS>(
            second_grad_output,
            second_shared_b,
            accumulators,
            wave_row_start,
            row_end,
            output_start,
            lane,
            SKIP_INACTIVE_M_TILES);
        __syncthreads();
    }

    grouped_backward_store_tile<
        GROUPED_BACKWARD_SMALL_N_TILES,
        M_TILES_PER_WAVE,
        IN_FEATURES,
        FULL_ROWS>(
        grad_input,
        accumulators,
        wave_row_start,
        row_end,
        input_column_start,
        lane,
        SKIP_INACTIVE_M_TILES);
}

template <
    ggml_type type,
    int OUT_FEATURES,
    int IN_FEATURES,
    int BLOCKS_PER_WEIGHT_ROW,
    int M_TILES_PER_WAVE,
    bool SKIP_INACTIVE_M_TILES>
static __device__ __forceinline__ void grouped_mmq_pair_grad_input_deepseek_body(
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

    constexpr int m_per_block =
        M_TILES_PER_WAVE * BACKWARD_M_PER_TILE * BACKWARD_WAVES;
    const int input_column_start = blockIdx.x * GROUPED_BACKWARD_SMALL_N;
    const char * first_expert_weight =
        first_packed_weight + expert * bytes_per_expert;
    const char * second_expert_weight =
        second_packed_weight + expert * bytes_per_expert;
    __shared__ grouped_backward_deepseek_pair_shared_tile shared_b[2];

    int block_row_start = row_begin;
    for (; block_row_start + m_per_block <= row_end;
         block_row_start += m_per_block) {
        grouped_mmq_pair_grad_input_deepseek_tile<
            type,
            OUT_FEATURES,
            IN_FEATURES,
            BLOCKS_PER_WEIGHT_ROW,
            M_TILES_PER_WAVE,
            SKIP_INACTIVE_M_TILES,
            true>(
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
        grouped_mmq_pair_grad_input_deepseek_tile<
            type,
            OUT_FEATURES,
            IN_FEATURES,
            BLOCKS_PER_WEIGHT_ROW,
            M_TILES_PER_WAVE,
            SKIP_INACTIVE_M_TILES,
            false>(
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

} // namespace torch_ggml_ops::ck
