#pragma once

#include "grouped_mmq_backward_tiled_common.cuh"

namespace torch_ggml_ops::ck {

static constexpr int FIXED_GROUPED_BACKWARD_GROUPS = 8;
static constexpr int FIXED_GROUPED_BACKWARD_OUT_FEATURES = 1024;
static constexpr int FIXED_GROUPED_BACKWARD_IN_FEATURES = 4096;
static constexpr int FIXED_GROUPED_BACKWARD_N_TILES = 4;
static constexpr int FIXED_GROUPED_BACKWARD_N =
    FIXED_GROUPED_BACKWARD_N_TILES * BACKWARD_N_PER_TILE;
static constexpr int FIXED_GROUPED_BACKWARD_BLOCKS_PER_ROW =
    FIXED_GROUPED_BACKWARD_IN_FEATURES / QK8_0;

template <int SWIZZLE>
using fixed_grouped_backward_shared_tile = backward_shared_b_tile<
    FIXED_GROUPED_BACKWARD_N,
    GROUPED_BACKWARD_TILED_K,
    0,
    SWIZZLE>;

template <
    int M_TILES_PER_WAVE,
    bool FULL_ROWS,
    int SWIZZLE>
static __device__ __forceinline__ void fixed_grouped_backward_accumulate(
        const __hip_bfloat16 * __restrict__ grad_output,
        const fixed_grouped_backward_shared_tile<SWIZZLE> & shared_b,
        f32_accumulator (&accumulators)
            [M_TILES_PER_WAVE][FIXED_GROUPED_BACKWARD_N_TILES],
        int wave_row_start,
        int row_end,
        int group,
        int output_start,
        int lane) {
#pragma unroll
    for (int k_tile = 0; k_tile < GROUPED_BACKWARD_TILED_K; k_tile += 16) {
        bf16_fragment a_fragments[M_TILES_PER_WAVE];
#pragma unroll
        for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
            __hip_bfloat16 * a = fragment_data(a_fragments[m_tile]);
            const int row = wave_row_start +
                m_tile * BACKWARD_M_PER_TILE + c_row(lane);
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                if constexpr (FULL_ROWS) {
                    a[k] = grad_output[
                        (static_cast<int64_t>(row) *
                            FIXED_GROUPED_BACKWARD_GROUPS + group) *
                            FIXED_GROUPED_BACKWARD_OUT_FEATURES +
                        output_start + k_tile + k];
                } else {
                    a[k] = row < row_end
                        ? grad_output[
                            (static_cast<int64_t>(row) *
                                FIXED_GROUPED_BACKWARD_GROUPS + group) *
                                FIXED_GROUPED_BACKWARD_OUT_FEATURES +
                            output_start + k_tile + k]
                        : __float2bfloat16(0.0f);
                }
            }
        }
#pragma unroll
        for (int n_tile = 0;
             n_tile < FIXED_GROUPED_BACKWARD_N_TILES;
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

template <int M_TILES_PER_WAVE, bool FULL_ROWS>
static __device__ __forceinline__ void fixed_grouped_backward_store(
        __hip_bfloat16 * __restrict__ grad_input,
        const f32_accumulator (&accumulators)
            [M_TILES_PER_WAVE][FIXED_GROUPED_BACKWARD_N_TILES],
        int wave_row_start,
        int row_end,
        int group,
        int input_column_start,
        int lane) {
#pragma unroll
    for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
#pragma unroll
        for (int n_tile = 0;
             n_tile < FIXED_GROUPED_BACKWARD_N_TILES;
             ++n_tile) {
#pragma unroll
            for (int element = 0; element < 8; ++element) {
                const int row = wave_row_start +
                    m_tile * BACKWARD_M_PER_TILE + c_column(lane, element);
                const int column = input_column_start +
                    n_tile * BACKWARD_N_PER_TILE + c_row(lane);
                if constexpr (FULL_ROWS) {
                    grad_input[
                        (static_cast<int64_t>(row) *
                            FIXED_GROUPED_BACKWARD_GROUPS + group) *
                            FIXED_GROUPED_BACKWARD_IN_FEATURES + column] =
                        __float2bfloat16(
                            accumulators[m_tile][n_tile].values[element]);
                } else if (row < row_end) {
                    grad_input[
                        (static_cast<int64_t>(row) *
                            FIXED_GROUPED_BACKWARD_GROUPS + group) *
                            FIXED_GROUPED_BACKWARD_IN_FEATURES + column] =
                        __float2bfloat16(
                            accumulators[m_tile][n_tile].values[element]);
                }
            }
        }
    }
}

template <
    int M_TILES_PER_WAVE,
    int DECODER_WIDTH,
    int SWIZZLE,
    bool FULL_ROWS>
static __device__ __forceinline__ void fixed_grouped_q8_0_grad_input_tile(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ group_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        fixed_grouped_backward_shared_tile<SWIZZLE> & shared_b,
        int block_row_start,
        int row_end,
        int group,
        int input_column_start) {
    constexpr int packed_row_bytes =
        FIXED_GROUPED_BACKWARD_BLOCKS_PER_ROW * sizeof(block_q8_0);
    constexpr int groups_per_row =
        FIXED_GROUPED_BACKWARD_N / DECODER_WIDTH;
    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int wave_row_start = block_row_start +
        wave * M_TILES_PER_WAVE * BACKWARD_M_PER_TILE;
    f32_accumulator accumulators
        [M_TILES_PER_WAVE]
        [FIXED_GROUPED_BACKWARD_N_TILES];

#pragma unroll 1
    for (int output_start = 0;
         output_start < FIXED_GROUPED_BACKWARD_OUT_FEATURES;
         output_start += GROUPED_BACKWARD_TILED_K) {
        const int decoder_group = threadIdx.x;
        if (decoder_group < GROUPED_BACKWARD_TILED_K * groups_per_row) {
            const int k = decoder_group / groups_per_row;
            const int local_input_column =
                DECODER_WIDTH * (decoder_group % groups_per_row);
            const int input_column = input_column_start + local_input_column;
            const char * packed_row = group_weight +
                static_cast<int64_t>(output_start + k) * packed_row_bytes;
            __hip_bfloat16 values[DECODER_WIDTH];
            decode_backward_tile_group<GGML_TYPE_Q8_0, DECODER_WIDTH>(
                packed_row,
                input_column / QK8_0,
                input_column % QK8_0,
                values);
#pragma unroll
            for (int index = 0; index < DECODER_WIDTH; ++index) {
                shared_b[(local_input_column + index) *
                    GROUPED_BACKWARD_TILED_K + k] = values[index];
            }
        }
        __syncthreads();
        fixed_grouped_backward_accumulate<
            M_TILES_PER_WAVE,
            FULL_ROWS,
            SWIZZLE>(
            grad_output,
            shared_b,
            accumulators,
            wave_row_start,
            row_end,
            group,
            output_start,
            lane);
        __syncthreads();
    }

    fixed_grouped_backward_store<M_TILES_PER_WAVE, FULL_ROWS>(
        grad_input,
        accumulators,
        wave_row_start,
        row_end,
        group,
        input_column_start,
        lane);
}

template <int M_TILES_PER_WAVE, int DECODER_WIDTH, int SWIZZLE>
static __device__ __forceinline__ void fixed_grouped_q8_0_grad_input_tiled_body(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        int tokens,
        int64_t bytes_per_group) {
    constexpr int m_per_block =
        M_TILES_PER_WAVE * BACKWARD_M_PER_TILE * BACKWARD_WAVES;
    const int block_row_start = blockIdx.y * m_per_block;
    const int group = blockIdx.z;
    const int input_column_start = blockIdx.x * FIXED_GROUPED_BACKWARD_N;
    const char * group_weight =
        packed_weight + static_cast<int64_t>(group) * bytes_per_group;
    __shared__ fixed_grouped_backward_shared_tile<SWIZZLE> shared_b;

    if (block_row_start + m_per_block <= tokens) {
        fixed_grouped_q8_0_grad_input_tile<
            M_TILES_PER_WAVE,
            DECODER_WIDTH,
            SWIZZLE,
            true>(
            grad_output,
            group_weight,
            grad_input,
            shared_b,
            block_row_start,
            tokens,
            group,
            input_column_start);
    } else if (block_row_start < tokens) {
        fixed_grouped_q8_0_grad_input_tile<
            M_TILES_PER_WAVE,
            DECODER_WIDTH,
            SWIZZLE,
            false>(
            grad_output,
            group_weight,
            grad_input,
            shared_b,
            block_row_start,
            tokens,
            group,
            input_column_start);
    }
}

} // namespace torch_ggml_ops::ck
