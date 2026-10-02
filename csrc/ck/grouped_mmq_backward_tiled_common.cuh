#pragma once

#include "mmq_backward.cuh"

namespace torch_ggml_ops::ck {

static constexpr int GROUPED_BACKWARD_TILED_N_TILES = 8;
static constexpr int GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE = 2;
static constexpr int GROUPED_BACKWARD_TILED_K = 32;
static constexpr int GROUPED_BACKWARD_TILED_N =
    GROUPED_BACKWARD_TILED_N_TILES * BACKWARD_N_PER_TILE;
static constexpr int GROUPED_BACKWARD_TILED_M =
    GROUPED_BACKWARD_TILED_M_TILES_PER_WAVE * BACKWARD_M_PER_TILE *
    BACKWARD_WAVES;
static constexpr int GROUPED_BACKWARD_SMALL_N_TILES = 4;
static constexpr int GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE = 1;
static constexpr int GROUPED_BACKWARD_SMALL_N =
    GROUPED_BACKWARD_SMALL_N_TILES * BACKWARD_N_PER_TILE;
static constexpr int GROUPED_BACKWARD_SMALL_M =
    GROUPED_BACKWARD_SMALL_M_TILES_PER_WAVE * BACKWARD_M_PER_TILE *
    BACKWARD_WAVES;

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
        int lane,
        bool skip_inactive_m_tiles = false) {
#pragma unroll
    for (int k_tile = 0; k_tile < GROUPED_BACKWARD_TILED_K; k_tile += 16) {
        bf16_fragment a_fragments[M_TILES_PER_WAVE];
#pragma unroll
        for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
            if constexpr (!FULL_ROWS) {
                if (skip_inactive_m_tiles &&
                    wave_row_start + m_tile * BACKWARD_M_PER_TILE >= row_end) {
                    continue;
                }
            }
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
                if constexpr (!FULL_ROWS) {
                    if (skip_inactive_m_tiles &&
                        wave_row_start + m_tile * BACKWARD_M_PER_TILE >=
                            row_end) {
                        continue;
                    }
                }
                wmma_f32_16x16x16_bf16(
                    accumulators[m_tile][n_tile],
                    a_fragments[m_tile],
                    b_first);
            }
#pragma unroll
            for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
                if constexpr (!FULL_ROWS) {
                    if (skip_inactive_m_tiles &&
                        wave_row_start + m_tile * BACKWARD_M_PER_TILE >=
                            row_end) {
                        continue;
                    }
                }
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
        int lane,
        bool skip_inactive_m_tiles = false) {
#pragma unroll
    for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
        if constexpr (!FULL_ROWS) {
            if (skip_inactive_m_tiles &&
                wave_row_start + m_tile * BACKWARD_M_PER_TILE >= row_end) {
                continue;
            }
        }
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

} // namespace torch_ggml_ops::ck
