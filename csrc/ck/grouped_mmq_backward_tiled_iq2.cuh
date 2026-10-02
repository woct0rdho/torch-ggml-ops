#pragma once

#include "grouped_mmq_backward_tiled_common.cuh"

namespace torch_ggml_ops::ck {

static constexpr int GROUPED_BACKWARD_TILED_IQ2_DOWN_OUT_FEATURES = 2048;
static constexpr int GROUPED_BACKWARD_TILED_IQ2_DOWN_IN_FEATURES = 512;
static constexpr int GROUPED_BACKWARD_TILED_IQ2_DOWN_BLOCKS_PER_ROW = 2;
static constexpr int GROUPED_BACKWARD_TILED_IQ2_PAIR_OUT_FEATURES = 512;
static constexpr int GROUPED_BACKWARD_TILED_IQ2_PAIR_IN_FEATURES = 2048;
static constexpr int GROUPED_BACKWARD_TILED_IQ2_PAIR_BLOCKS_PER_ROW = 8;
static constexpr int GROUPED_BACKWARD_TILED_IQ2_DOWN_SWIZZLE = 16;

using grouped_backward_iq2_down_shared_tile = backward_shared_b_tile<
    GROUPED_BACKWARD_TILED_N,
    GROUPED_BACKWARD_TILED_K,
    0,
    GROUPED_BACKWARD_TILED_IQ2_DOWN_SWIZZLE>;
using grouped_backward_iq2_down_small_shared_tile = backward_shared_b_tile<
    GROUPED_BACKWARD_SMALL_N,
    GROUPED_BACKWARD_TILED_K,
    0,
    GROUPED_BACKWARD_TILED_IQ2_DOWN_SWIZZLE>;
using grouped_backward_iq2_pair_shared_tile = backward_shared_b_tile<
    GROUPED_BACKWARD_TILED_N,
    GROUPED_BACKWARD_TILED_K,
    0,
    GROUPED_BACKWARD_TILED_IQ2_PAIR_SWIZZLE>;
using grouped_backward_iq2_pair_small_shared_tile = backward_shared_b_tile<
    GROUPED_BACKWARD_SMALL_N,
    GROUPED_BACKWARD_TILED_K,
    0,
    GROUPED_BACKWARD_TILED_IQ2_PAIR_SWIZZLE>;

template <
    int N_TILES,
    int M_TILES_PER_WAVE,
    bool FULL_ROWS,
    typename SharedTile>
static __device__ __forceinline__ void grouped_mmq_grad_input_iq2_tile(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ expert_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        SharedTile & shared_b,
        int block_row_start,
        int row_end,
        int input_column_start,
        bool skip_inactive_m_tiles = false) {
    constexpr int packed_row_bytes =
        GROUPED_BACKWARD_TILED_IQ2_DOWN_BLOCKS_PER_ROW * sizeof(block_iq2_s);
    constexpr int n_per_block = N_TILES * BACKWARD_N_PER_TILE;
    constexpr int groups_per_row = n_per_block / 16;
    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int wave_row_start = block_row_start +
        wave * M_TILES_PER_WAVE * BACKWARD_M_PER_TILE;
    f32_accumulator accumulators[M_TILES_PER_WAVE][N_TILES];

#pragma unroll 1
    for (int output_start = 0;
         output_start < GROUPED_BACKWARD_TILED_IQ2_DOWN_OUT_FEATURES;
         output_start += GROUPED_BACKWARD_TILED_K) {
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
            decode_iq2_s_group_16(
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
        __syncthreads();
        grouped_backward_accumulate_projection<
            N_TILES,
            M_TILES_PER_WAVE,
            GROUPED_BACKWARD_TILED_IQ2_DOWN_OUT_FEATURES,
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
        GROUPED_BACKWARD_TILED_IQ2_DOWN_IN_FEATURES,
        FULL_ROWS>(
        grad_input,
        accumulators,
        wave_row_start,
        row_end,
        input_column_start,
        lane,
        skip_inactive_m_tiles);
}

template <bool SMALL>
static __device__ __forceinline__ void grouped_mmq_grad_input_iq2_tiled_body(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int rows,
        int64_t bytes_per_expert) {
    constexpr int n_per_block =
        SMALL ? GROUPED_BACKWARD_SMALL_N : GROUPED_BACKWARD_TILED_N;
    constexpr int m_per_block =
        SMALL ? GROUPED_BACKWARD_SMALL_M : GROUPED_BACKWARD_TILED_M;
    const int group = blockIdx.y;
    const int row_begin = group == 0 ? 0 : expert_offsets[group - 1];
    const int row_end = expert_offsets[group];
    const int64_t expert = expert_indices[group];
    if (expert < 0 || expert >= num_experts || row_begin < 0 ||
        row_end <= row_begin || row_end > rows) {
        return;
    }

    const int input_column_start = blockIdx.x * n_per_block;
    const char * expert_weight = packed_weight + expert * bytes_per_expert;
    using shared_tile = std::conditional_t<
        SMALL,
        grouped_backward_iq2_down_small_shared_tile,
        grouped_backward_iq2_down_shared_tile>;
    __shared__ shared_tile shared_b;

    int block_row_start = row_begin;
    for (; block_row_start + m_per_block <= row_end;
         block_row_start += m_per_block) {
        if constexpr (SMALL) {
            grouped_mmq_grad_input_iq2_tile<
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
        } else {
            grouped_mmq_grad_input_iq2_tile<
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
    }
    if (block_row_start < row_end) {
        if constexpr (SMALL) {
            grouped_mmq_grad_input_iq2_tile<
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
        } else {
            grouped_mmq_grad_input_iq2_tile<
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
}

static constexpr int GROUPED_BACKWARD_IQ2_S2_M_TILES_PER_WAVE = 2;
static constexpr int GROUPED_BACKWARD_IQ2_S2_M =
    GROUPED_BACKWARD_IQ2_S2_M_TILES_PER_WAVE * BACKWARD_M_PER_TILE *
    BACKWARD_WAVES;

static __device__ __forceinline__ void grouped_mmq_grad_input_iq2_s2_body(
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
    __shared__ grouped_backward_iq2_down_small_shared_tile shared_b;

    int block_row_start = row_begin;
    for (; block_row_start + GROUPED_BACKWARD_IQ2_S2_M <= row_end;
         block_row_start += GROUPED_BACKWARD_IQ2_S2_M) {
        grouped_mmq_grad_input_iq2_tile<
            GROUPED_BACKWARD_SMALL_N_TILES,
            GROUPED_BACKWARD_IQ2_S2_M_TILES_PER_WAVE,
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
        grouped_mmq_grad_input_iq2_tile<
            GROUPED_BACKWARD_SMALL_N_TILES,
            GROUPED_BACKWARD_IQ2_S2_M_TILES_PER_WAVE,
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
    int N_TILES,
    int M_TILES_PER_WAVE,
    bool FULL_ROWS,
    typename SharedTile>
static __device__ __forceinline__ void grouped_mmq_pair_grad_input_iq2_tile(
        const __hip_bfloat16 * __restrict__ first_grad_output,
        const __hip_bfloat16 * __restrict__ second_grad_output,
        const char * __restrict__ first_expert_weight,
        const char * __restrict__ second_expert_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        SharedTile & first_shared_b,
        SharedTile & second_shared_b,
        int block_row_start,
        int row_end,
        int input_column_start) {
    constexpr int packed_row_bytes =
        GROUPED_BACKWARD_TILED_IQ2_PAIR_BLOCKS_PER_ROW * sizeof(block_iq2_s);
    constexpr int n_per_block = N_TILES * BACKWARD_N_PER_TILE;
    constexpr int groups_per_row = n_per_block / 16;
    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int wave_row_start = block_row_start +
        wave * M_TILES_PER_WAVE * BACKWARD_M_PER_TILE;
    f32_accumulator accumulators[M_TILES_PER_WAVE][N_TILES];

#pragma unroll 1
    for (int output_start = 0;
         output_start < GROUPED_BACKWARD_TILED_IQ2_PAIR_OUT_FEATURES;
         output_start += GROUPED_BACKWARD_TILED_K) {
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
            const int64_t row_offset =
                static_cast<int64_t>(output_start + k) * packed_row_bytes;
            __hip_bfloat16 values[16];
            decode_iq2_s_group_16(
                first_expert_weight + row_offset,
                block_index,
                value_index,
                values);
#pragma unroll
            for (int index = 0; index < 16; ++index) {
                first_shared_b[(local_input_column + index) *
                    GROUPED_BACKWARD_TILED_K + k] = values[index];
            }
            decode_iq2_s_group_16(
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
        grouped_backward_accumulate_projection<
            N_TILES,
            M_TILES_PER_WAVE,
            GROUPED_BACKWARD_TILED_IQ2_PAIR_OUT_FEATURES,
            FULL_ROWS>(
            first_grad_output,
            first_shared_b,
            accumulators,
            wave_row_start,
            row_end,
            output_start,
            lane);
        grouped_backward_accumulate_projection<
            N_TILES,
            M_TILES_PER_WAVE,
            GROUPED_BACKWARD_TILED_IQ2_PAIR_OUT_FEATURES,
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
        N_TILES,
        M_TILES_PER_WAVE,
        GROUPED_BACKWARD_TILED_IQ2_PAIR_IN_FEATURES,
        FULL_ROWS>(
        grad_input,
        accumulators,
        wave_row_start,
        row_end,
        input_column_start,
        lane);
}

template <bool SMALL>
static __device__ __forceinline__ void grouped_mmq_pair_grad_input_iq2_tiled_body(
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
    constexpr int n_per_block =
        SMALL ? GROUPED_BACKWARD_SMALL_N : GROUPED_BACKWARD_TILED_N;
    constexpr int m_per_block =
        SMALL ? GROUPED_BACKWARD_SMALL_M : GROUPED_BACKWARD_TILED_M;
    const int group = blockIdx.y;
    const int row_begin = group == 0 ? 0 : expert_offsets[group - 1];
    const int row_end = expert_offsets[group];
    const int64_t expert = expert_indices[group];
    if (expert < 0 || expert >= num_experts || row_begin < 0 ||
        row_end <= row_begin || row_end > rows) {
        return;
    }

    const int input_column_start = blockIdx.x * n_per_block;
    const char * first_expert_weight =
        first_packed_weight + expert * bytes_per_expert;
    const char * second_expert_weight =
        second_packed_weight + expert * bytes_per_expert;
    using shared_tile = std::conditional_t<
        SMALL,
        grouped_backward_iq2_pair_small_shared_tile,
        grouped_backward_iq2_pair_shared_tile>;
    __shared__ shared_tile shared_b[2];

    int block_row_start = row_begin;
    for (; block_row_start + m_per_block <= row_end;
         block_row_start += m_per_block) {
        if constexpr (SMALL) {
            grouped_mmq_pair_grad_input_iq2_tile<
                GROUPED_BACKWARD_SMALL_N_TILES,
                GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE,
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
        } else {
            grouped_mmq_pair_grad_input_iq2_tile<
                GROUPED_BACKWARD_TILED_N_TILES,
                GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE,
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
    }
    if (block_row_start < row_end) {
        if constexpr (SMALL) {
            grouped_mmq_pair_grad_input_iq2_tile<
                GROUPED_BACKWARD_SMALL_N_TILES,
                GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE,
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
        } else {
            grouped_mmq_pair_grad_input_iq2_tile<
                GROUPED_BACKWARD_TILED_N_TILES,
                GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE,
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
}

static __device__ __forceinline__ void grouped_mmq_pair_grad_input_iq2_n64_large_body(
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
    __shared__ grouped_backward_iq2_pair_small_shared_tile shared_b[2];

    int block_row_start = row_begin;
    for (; block_row_start + GROUPED_BACKWARD_TILED_M <= row_end;
         block_row_start += GROUPED_BACKWARD_TILED_M) {
        grouped_mmq_pair_grad_input_iq2_tile<
            GROUPED_BACKWARD_SMALL_N_TILES,
            GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE,
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
        grouped_mmq_pair_grad_input_iq2_tile<
            GROUPED_BACKWARD_SMALL_N_TILES,
            GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE,
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

static __device__ __forceinline__ void grouped_mmq_grad_input_iq2_row_task_body(
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
    __shared__ grouped_backward_iq2_down_shared_tile shared_b;
    grouped_mmq_grad_input_iq2_tile<
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
        false);
}

} // namespace torch_ggml_ops::ck
