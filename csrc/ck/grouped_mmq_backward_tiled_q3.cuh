#pragma once

#include "mmq_backward.cuh"

namespace torch_ggml_ops::ck {

static constexpr int GROUPED_BACKWARD_TILED_Q3_OUT_FEATURES = 512;
static constexpr int GROUPED_BACKWARD_TILED_Q3_IN_FEATURES = 2048;
static constexpr int GROUPED_BACKWARD_TILED_Q3_BLOCKS_PER_ROW = 8;
static constexpr int GROUPED_BACKWARD_TILED_Q3_PADDING = 8;

using grouped_backward_q3_small_shared_tile = backward_shared_b_tile<
    GROUPED_BACKWARD_SMALL_N,
    GROUPED_BACKWARD_TILED_K,
    GROUPED_BACKWARD_TILED_Q3_PADDING,
    0>;

template <int M_TILES_PER_WAVE, bool FULL_ROWS>
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
    const int wave_row_start = block_row_start +
        wave * M_TILES_PER_WAVE * BACKWARD_M_PER_TILE;
    f32_accumulator accumulators
        [M_TILES_PER_WAVE]
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
            M_TILES_PER_WAVE,
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
            M_TILES_PER_WAVE,
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
        M_TILES_PER_WAVE,
        GROUPED_BACKWARD_TILED_Q3_IN_FEATURES,
        FULL_ROWS>(
        grad_input,
        accumulators,
        wave_row_start,
        row_end,
        input_column_start,
        lane);
}

static __device__ __forceinline__ void grouped_mmq_pair_grad_input_q3_small_body(
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
        grouped_mmq_pair_grad_input_q3_small_tile<
            GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE, true>(
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
        grouped_mmq_pair_grad_input_q3_small_tile<
            GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE, false>(
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

static __device__ __forceinline__ void grouped_mmq_pair_grad_input_q3_n64_large_body(
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
    for (; block_row_start + GROUPED_BACKWARD_TILED_M <= row_end;
         block_row_start += GROUPED_BACKWARD_TILED_M) {
        grouped_mmq_pair_grad_input_q3_small_tile<
            GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE, true>(
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
        grouped_mmq_pair_grad_input_q3_small_tile<
            GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE, false>(
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
