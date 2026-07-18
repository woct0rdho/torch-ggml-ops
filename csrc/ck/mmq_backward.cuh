#pragma once

#include "bf16_wmma.cuh"
#include "gguf_decode.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace torch_ggml_ops::ck {

static constexpr int BACKWARD_WAVE_SIZE = 32;
static constexpr int BACKWARD_WAVES = 4;
static constexpr int BACKWARD_THREADS = BACKWARD_WAVE_SIZE * BACKWARD_WAVES;
static constexpr int BACKWARD_M_TILES_PER_WAVE = 1;
static constexpr int BACKWARD_M_PER_TILE = 16;
static constexpr int BACKWARD_M_PER_WAVE =
    BACKWARD_M_TILES_PER_WAVE * BACKWARD_M_PER_TILE;
static constexpr int BACKWARD_M_PER_BLOCK = BACKWARD_M_PER_WAVE * BACKWARD_WAVES;
static constexpr int BACKWARD_N_PER_TILE = 16;

template <ggml_type type>
static __device__ __forceinline__ __hip_bfloat16 decode_backward_tile_value(
        const char * packed_row,
        int block_index,
        int value_index,
        int local_input_column) {
    if constexpr (type == GGML_TYPE_Q6_K) {
        const auto & block =
            reinterpret_cast<const block_q6_K *>(packed_row)[block_index];
        float scaled_d = local_input_column == 0
            ? fp16_to_fp32(block.d) *
                static_cast<float>(block.scales[value_index >> 4])
            : 0.0f;
        scaled_d = __shfl_sync(
            0xffffffff, scaled_d, 0, BACKWARD_N_PER_TILE);

        const int chunk = value_index >> 7;
        const int remainder = value_index & 127;
        const int low_byte = chunk * 64 + (remainder & 63);
        const int low =
            (block.ql[low_byte] >> (4 * (remainder >> 6))) & 0x0f;
        const int high_byte = chunk * 32 + (value_index & 31);
        const int high =
            (block.qh[high_byte] >> (2 * ((remainder >> 5) & 3))) & 0x03;
        const int quant = (low | (high << 4)) - 32;
        return __float2bfloat16(scaled_d * static_cast<float>(quant));
    } else {
        return __float2bfloat16(decode_gguf_value<type>(
            packed_row, block_index, value_index));
    }
}

template <ggml_type type>
static __device__ __forceinline__ void decode_backward_tile_pair(
        const char * packed_row,
        int block_index,
        int value_index,
        __hip_bfloat16 & first,
        __hip_bfloat16 & second) {
    if constexpr (type == GGML_TYPE_Q3_K) {
        const auto & block =
            reinterpret_cast<const block_q3_K *>(packed_row)[block_index];
        const int scale_group = value_index >> 4;
        const int low_scale = scale_group < 8
            ? block.scales[scale_group]
            : block.scales[scale_group - 8] >> 4;
        const int high_scale =
            block.scales[8 + (scale_group & 3)] >> (2 * (scale_group >> 2));
        const int scale =
            ((low_scale & 0x0f) | ((high_scale & 0x03) << 4)) - 32;
        const float scaled_d = fp16_to_fp32(block.d) * static_cast<float>(scale);

        const int low_chunk = value_index >> 7;
        const int low_shift = 2 * ((value_index & 127) >> 5);
        const int low_byte = low_chunk * 32 + (value_index & 31);
        const int first_low = (block.qs[low_byte] >> low_shift) & 0x03;
        const int second_low = (block.qs[low_byte + 1] >> low_shift) & 0x03;
        const int high_shift = value_index >> 5;
        const int first_high =
            ((block.hmask[value_index & 31] >> high_shift) & 0x01) ^ 0x01;
        const int second_high =
            ((block.hmask[(value_index + 1) & 31] >> high_shift) & 0x01) ^ 0x01;
        first = __float2bfloat16(
            scaled_d * static_cast<float>(first_low - (first_high << 2)));
        second = __float2bfloat16(
            scaled_d * static_cast<float>(second_low - (second_high << 2)));
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        const auto & block =
            reinterpret_cast<const block_q4_K *>(packed_row)[block_index];
        const int group = value_index >> 5;
        const float d = fp16_to_fp32(block.d) *
            static_cast<float>(k_scale(block.scales, group));
        const float minimum = fp16_to_fp32(block.dmin) *
            static_cast<float>(k_min(block.scales, group));
        const int byte = (group >> 1) * 32 + (value_index & 31);
        const int shift = 4 * (group & 1);
        const int first_quant = (block.qs[byte] >> shift) & 0x0f;
        const int second_quant = (block.qs[byte + 1] >> shift) & 0x0f;
        first = __float2bfloat16(
            d * static_cast<float>(first_quant) - minimum);
        second = __float2bfloat16(
            d * static_cast<float>(second_quant) - minimum);
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        const auto & block =
            reinterpret_cast<const block_q5_K *>(packed_row)[block_index];
        const int group = value_index >> 5;
        const float d = fp16_to_fp32(block.d) *
            static_cast<float>(k_scale(block.scales, group));
        const float minimum = fp16_to_fp32(block.dmin) *
            static_cast<float>(k_min(block.scales, group));
        const int byte = (group >> 1) * 32 + (value_index & 31);
        const int shift = 4 * (group & 1);
        const int first_low = (block.qs[byte] >> shift) & 0x0f;
        const int second_low = (block.qs[byte + 1] >> shift) & 0x0f;
        const int first_high =
            (block.qh[value_index & 31] >> group) & 0x01;
        const int second_high =
            (block.qh[(value_index + 1) & 31] >> group) & 0x01;
        first = __float2bfloat16(
            d * static_cast<float>(first_low | (first_high << 4)) - minimum);
        second = __float2bfloat16(
            d * static_cast<float>(second_low | (second_high << 4)) - minimum);
    } else {
        first = __float2bfloat16(decode_gguf_value<type>(
            packed_row, block_index, value_index));
        second = __float2bfloat16(decode_gguf_value<type>(
            packed_row, block_index, value_index + 1));
    }
}

template <ggml_type type>
static __device__ __forceinline__ void decode_backward_tile_quad(
        const char * packed_row,
        int block_index,
        int value_index,
        __hip_bfloat16 * values) {
    if constexpr (type == GGML_TYPE_Q3_K) {
        const auto & block =
            reinterpret_cast<const block_q3_K *>(packed_row)[block_index];
        const int scale_group = value_index >> 4;
        const int low_scale = scale_group < 8
            ? block.scales[scale_group]
            : block.scales[scale_group - 8] >> 4;
        const int high_scale =
            block.scales[8 + (scale_group & 3)] >> (2 * (scale_group >> 2));
        const int scale =
            ((low_scale & 0x0f) | ((high_scale & 0x03) << 4)) - 32;
        const float scaled_d = fp16_to_fp32(block.d) * static_cast<float>(scale);
        const int low_chunk = value_index >> 7;
        const int low_shift = 2 * ((value_index & 127) >> 5);
        const int high_shift = value_index >> 5;
#pragma unroll
        for (int index = 0; index < 4; ++index) {
            const int value = value_index + index;
            const int low = (
                block.qs[low_chunk * 32 + (value & 31)] >> low_shift) & 0x03;
            const int high =
                ((block.hmask[value & 31] >> high_shift) & 0x01) ^ 0x01;
            values[index] = __float2bfloat16(
                scaled_d * static_cast<float>(low - (high << 2)));
        }
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        const auto & block =
            reinterpret_cast<const block_q6_K *>(packed_row)[block_index];
        const float scaled_d = fp16_to_fp32(block.d) *
            static_cast<float>(block.scales[value_index >> 4]);
        const int chunk = value_index >> 7;
        const int remainder = value_index & 127;
        const int low_byte = chunk * 64 + (remainder & 63);
        const int low_shift = 4 * (remainder >> 6);
        const int high_byte = chunk * 32 + (value_index & 31);
        const int high_shift = 2 * ((remainder >> 5) & 3);
#pragma unroll
        for (int index = 0; index < 4; ++index) {
            const int low = (block.ql[low_byte + index] >> low_shift) & 0x0f;
            const int high =
                (block.qh[high_byte + index] >> high_shift) & 0x03;
            values[index] = __float2bfloat16(
                scaled_d * static_cast<float>((low | (high << 4)) - 32));
        }
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        const auto & block =
            reinterpret_cast<const block_q4_K *>(packed_row)[block_index];
        const int group = value_index >> 5;
        const float d = fp16_to_fp32(block.d) *
            static_cast<float>(k_scale(block.scales, group));
        const float minimum = fp16_to_fp32(block.dmin) *
            static_cast<float>(k_min(block.scales, group));
        const int byte = (group >> 1) * 32 + (value_index & 31);
        const int shift = 4 * (group & 1);
#pragma unroll
        for (int index = 0; index < 4; ++index) {
            const int quant = (block.qs[byte + index] >> shift) & 0x0f;
            values[index] = __float2bfloat16(
                d * static_cast<float>(quant) - minimum);
        }
    } else {
        const auto & block =
            reinterpret_cast<const block_q5_K *>(packed_row)[block_index];
        const int group = value_index >> 5;
        const float d = fp16_to_fp32(block.d) *
            static_cast<float>(k_scale(block.scales, group));
        const float minimum = fp16_to_fp32(block.dmin) *
            static_cast<float>(k_min(block.scales, group));
        const int byte = (group >> 1) * 32 + (value_index & 31);
        const int shift = 4 * (group & 1);
#pragma unroll
        for (int index = 0; index < 4; ++index) {
            const int value = value_index + index;
            const int low = (block.qs[byte + index] >> shift) & 0x0f;
            const int high = (block.qh[value & 31] >> group) & 0x01;
            values[index] = __float2bfloat16(
                d * static_cast<float>(low | (high << 4)) - minimum);
        }
    }
}

static __device__ __forceinline__ void decode_backward_tile_sixteen_q6(
        const char * packed_row,
        int block_index,
        int value_index,
        __hip_bfloat16 * values) {
    const auto & block =
        reinterpret_cast<const block_q6_K *>(packed_row)[block_index];
    const float scaled_d = fp16_to_fp32(block.d) *
        static_cast<float>(block.scales[value_index >> 4]);
    const int chunk = value_index >> 7;
    const int remainder = value_index & 127;
    const int low_byte = chunk * 64 + (remainder & 63);
    const int low_shift = 4 * (remainder >> 6);
    const int high_byte = chunk * 32 + (value_index & 31);
    const int high_shift = 2 * ((remainder >> 5) & 3);
#pragma unroll
    for (int index = 0; index < 16; ++index) {
        const int low = (block.ql[low_byte + index] >> low_shift) & 0x0f;
        const int high = (block.qh[high_byte + index] >> high_shift) & 0x03;
        values[index] = __float2bfloat16(
            scaled_d * static_cast<float>((low | (high << 4)) - 32));
    }
}

template <ggml_type type, int N_TILES, int K_ITERATION>
__launch_bounds__(BACKWARD_THREADS, 2)
static __global__ void dense_mmq_grad_input_kernel(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        int rows,
        int out_features,
        int in_features,
        int blocks_per_weight_row) {
    constexpr int N_PER_BLOCK = N_TILES * BACKWARD_N_PER_TILE;
    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int block_row_start = blockIdx.x * BACKWARD_M_PER_BLOCK;
    const int wave_row_start = block_row_start + wave * BACKWARD_M_PER_WAVE;
    const int input_column_start = blockIdx.y * N_PER_BLOCK;
    const int64_t packed_row_bytes =
        static_cast<int64_t>(blocks_per_weight_row) * gguf_block_bytes<type>();

    __shared__ __hip_bfloat16 shared_b[N_PER_BLOCK * K_ITERATION];
    f32_accumulator accumulators[BACKWARD_M_TILES_PER_WAVE][N_TILES];

    for (int output_start = 0; output_start < out_features;
         output_start += K_ITERATION) {
        if constexpr (type == GGML_TYPE_Q6_K && N_TILES >= 2) {
            constexpr int groups_per_row = N_PER_BLOCK / 16;
#pragma unroll
            for (int group_index = threadIdx.x;
                 group_index < groups_per_row * K_ITERATION;
                 group_index += BACKWARD_THREADS) {
                const int k = group_index / groups_per_row;
                const int local_input_column =
                    16 * (group_index % groups_per_row);
                const int output_column = output_start + k;
                const int input_column = input_column_start + local_input_column;
                if (input_column + 15 < in_features && output_column < out_features) {
                    const char * packed_row = packed_weight +
                        static_cast<int64_t>(output_column) * packed_row_bytes;
                    __hip_bfloat16 values[16];
                    decode_backward_tile_sixteen_q6(
                        packed_row,
                        input_column / QK_K,
                        input_column % QK_K,
                        values);
#pragma unroll
                    for (int index = 0; index < 16; ++index) {
                        shared_b[
                            (local_input_column + index) * K_ITERATION + k] =
                            values[index];
                    }
                } else {
#pragma unroll
                    for (int index = 0; index < 16; ++index) {
                        shared_b[
                            (local_input_column + index) * K_ITERATION + k] =
                            __float2bfloat16(0.0f);
                    }
                }
            }
        } else if constexpr (
            (type == GGML_TYPE_Q6_K && N_TILES == 1) ||
            (K_ITERATION == 16 && type == GGML_TYPE_Q3_K && N_TILES == 4)
        ) {
            constexpr int quads_per_row = N_PER_BLOCK / 4;
#pragma unroll
            for (int quad_index = threadIdx.x;
                 quad_index < quads_per_row * K_ITERATION;
                 quad_index += BACKWARD_THREADS) {
                const int k = quad_index / quads_per_row;
                const int local_input_column = 4 * (quad_index % quads_per_row);
                const int output_column = output_start + k;
                const int input_column = input_column_start + local_input_column;
                if (input_column + 3 < in_features && output_column < out_features) {
                    const char * packed_row = packed_weight +
                        static_cast<int64_t>(output_column) * packed_row_bytes;
                    __hip_bfloat16 values[4];
                    decode_backward_tile_quad<type>(
                        packed_row,
                        input_column / QK_K,
                        input_column % QK_K,
                        values);
#pragma unroll
                    for (int index = 0; index < 4; ++index) {
                        shared_b[
                            (local_input_column + index) * K_ITERATION + k] =
                            values[index];
                    }
                } else {
#pragma unroll
                    for (int index = 0; index < 4; ++index) {
                        shared_b[
                            (local_input_column + index) * K_ITERATION + k] =
                            __float2bfloat16(0.0f);
                    }
                }
            }
        } else if constexpr (
            K_ITERATION == 16 &&
            (type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K ||
             type == GGML_TYPE_Q5_K)
        ) {
            constexpr int pairs_per_row = N_PER_BLOCK / 2;
#pragma unroll
            for (int pair_index = threadIdx.x;
                 pair_index < pairs_per_row * K_ITERATION;
                 pair_index += BACKWARD_THREADS) {
                const int k = pair_index / pairs_per_row;
                const int local_input_column = 2 * (pair_index % pairs_per_row);
                const int output_column = output_start + k;
                const int input_column = input_column_start + local_input_column;
                if (input_column + 1 < in_features && output_column < out_features) {
                    const char * packed_row = packed_weight +
                        static_cast<int64_t>(output_column) * packed_row_bytes;
                    decode_backward_tile_pair<type>(
                        packed_row,
                        input_column / QK_K,
                        input_column % QK_K,
                        shared_b[local_input_column * K_ITERATION + k],
                        shared_b[(local_input_column + 1) * K_ITERATION + k]);
                } else {
                    shared_b[local_input_column * K_ITERATION + k] =
                        __float2bfloat16(0.0f);
                    shared_b[(local_input_column + 1) * K_ITERATION + k] =
                        __float2bfloat16(0.0f);
                }
            }
        } else {
#pragma unroll
            for (int index = threadIdx.x; index < N_PER_BLOCK * K_ITERATION;
                 index += BACKWARD_THREADS) {
                const int k = index / N_PER_BLOCK;
                const int local_input_column = index % N_PER_BLOCK;
                const int output_column = output_start + k;
                const int input_column = input_column_start + local_input_column;
                if (input_column < in_features && output_column < out_features) {
                    const char * packed_row = packed_weight +
                        static_cast<int64_t>(output_column) * packed_row_bytes;
                    shared_b[local_input_column * K_ITERATION + k] =
                        decode_backward_tile_value<type>(
                            packed_row,
                            input_column / QK_K,
                            input_column % QK_K,
                            local_input_column % BACKWARD_N_PER_TILE);
                } else {
                    shared_b[local_input_column * K_ITERATION + k] =
                        __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int k_tile = 0; k_tile < K_ITERATION; k_tile += 16) {
            bf16_fragment a_fragments[BACKWARD_M_TILES_PER_WAVE];
#pragma unroll
            for (int m_tile = 0; m_tile < BACKWARD_M_TILES_PER_WAVE; ++m_tile) {
                __hip_bfloat16 * a = fragment_data(a_fragments[m_tile]);
                const int a_row =
                    wave_row_start + m_tile * BACKWARD_M_PER_TILE + c_row(lane);
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    const int output_column = output_start + k_tile + k;
                    a[k] = a_row < rows && output_column < out_features
                        ? grad_output[
                            static_cast<int64_t>(a_row) * out_features + output_column]
                        : __float2bfloat16(0.0f);
                }
            }

#pragma unroll
            for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
                bf16_fragment b_fragment{};
                __hip_bfloat16 * b = fragment_data(b_fragment);
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    b[k] = shared_b[
                        (n_tile * BACKWARD_N_PER_TILE + c_row(lane)) *
                            K_ITERATION +
                        k_tile + k];
                }
#pragma unroll
                for (int m_tile = 0; m_tile < BACKWARD_M_TILES_PER_WAVE; ++m_tile) {
                    wmma_f32_16x16x16_bf16(
                        accumulators[m_tile][n_tile],
                        a_fragments[m_tile],
                        b_fragment);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m_tile = 0; m_tile < BACKWARD_M_TILES_PER_WAVE; ++m_tile) {
#pragma unroll
        for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
#pragma unroll
            for (int element = 0; element < 8; ++element) {
                // gfx11's physical C fragment is J-major for this A/B layout: the
                // I-major lane coordinates are transposed when written to row-major C.
                const int output_row = wave_row_start +
                    m_tile * BACKWARD_M_PER_TILE + c_column(lane, element);
                const int output_column = input_column_start +
                    n_tile * BACKWARD_N_PER_TILE + c_row(lane);
                if (output_row < rows && output_column < in_features) {
                    grad_input[
                        static_cast<int64_t>(output_row) * in_features + output_column] =
                        __float2bfloat16(
                            accumulators[m_tile][n_tile].values[element]);
                }
            }
        }
    }
}

template <ggml_type type, int N_TILES, int K_ITERATION>
static inline void launch_dense_mmq_grad_input_tiled(
        const __hip_bfloat16 * grad_output,
        const char * packed_weight,
        __hip_bfloat16 * grad_input,
        int rows,
        int out_features,
        int in_features,
        hipStream_t stream) {
    constexpr int n_per_block = N_TILES * BACKWARD_N_PER_TILE;
    const dim3 grid(
        (rows + BACKWARD_M_PER_BLOCK - 1) / BACKWARD_M_PER_BLOCK,
        (in_features + n_per_block - 1) / n_per_block,
        1);
    const dim3 block(BACKWARD_THREADS, 1, 1);
    dense_mmq_grad_input_kernel<type, N_TILES, K_ITERATION>
        <<<grid, block, 0, stream>>>(
        grad_output,
        packed_weight,
        grad_input,
        rows,
        out_features,
        in_features,
        in_features / QK_K);
}

template <ggml_type type>
static inline void launch_dense_mmq_grad_input(
        const __hip_bfloat16 * grad_output,
        const char * packed_weight,
        __hip_bfloat16 * grad_input,
        int rows,
        int out_features,
        int in_features,
        hipStream_t stream) {
    if constexpr (type == GGML_TYPE_Q6_K) {
        if (rows <= 64) {
            launch_dense_mmq_grad_input_tiled<type, 1, 64>(
                grad_output, packed_weight, grad_input,
                rows, out_features, in_features, stream);
        } else if (rows <= 128) {
            launch_dense_mmq_grad_input_tiled<type, 2, 64>(
                grad_output, packed_weight, grad_input,
                rows, out_features, in_features, stream);
        } else if (rows <= 256) {
            launch_dense_mmq_grad_input_tiled<type, 4, 32>(
                grad_output, packed_weight, grad_input,
                rows, out_features, in_features, stream);
        } else if (rows <= 2048) {
            launch_dense_mmq_grad_input_tiled<type, 8, 16>(
                grad_output, packed_weight, grad_input,
                rows, out_features, in_features, stream);
        } else {
            launch_dense_mmq_grad_input_tiled<type, 16, 16>(
                grad_output, packed_weight, grad_input,
                rows, out_features, in_features, stream);
        }
    } else if (rows <= 128) {
        launch_dense_mmq_grad_input_tiled<type, 1, 16>(
            grad_output, packed_weight, grad_input,
            rows, out_features, in_features, stream);
    } else if (rows <= 256) {
        launch_dense_mmq_grad_input_tiled<type, 4, 16>(
            grad_output, packed_weight, grad_input,
            rows, out_features, in_features, stream);
    } else if (rows <= 2048) {
        if constexpr (type == GGML_TYPE_Q3_K) {
            if (out_features <= 512) {
                launch_dense_mmq_grad_input_tiled<type, 8, 16>(
                    grad_output, packed_weight, grad_input,
                    rows, out_features, in_features, stream);
            } else {
                launch_dense_mmq_grad_input_tiled<type, 4, 16>(
                    grad_output, packed_weight, grad_input,
                    rows, out_features, in_features, stream);
            }
        } else if constexpr (type == GGML_TYPE_IQ2_S) {
            launch_dense_mmq_grad_input_tiled<type, 4, 16>(
                grad_output, packed_weight, grad_input,
                rows, out_features, in_features, stream);
        } else if (in_features >= 4096) {
            launch_dense_mmq_grad_input_tiled<type, 16, 16>(
                grad_output, packed_weight, grad_input,
                rows, out_features, in_features, stream);
        } else {
            launch_dense_mmq_grad_input_tiled<type, 8, 16>(
                grad_output, packed_weight, grad_input,
                rows, out_features, in_features, stream);
        }
    } else if (rows <= 8192) {
        if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
            if (in_features == 2048) {
                launch_dense_mmq_grad_input_tiled<type, 16, 16>(
                    grad_output, packed_weight, grad_input,
                    rows, out_features, in_features, stream);
            } else {
                launch_dense_mmq_grad_input_tiled<type, 12, 16>(
                    grad_output, packed_weight, grad_input,
                    rows, out_features, in_features, stream);
            }
        } else {
            launch_dense_mmq_grad_input_tiled<type, 12, 16>(
                grad_output, packed_weight, grad_input,
                rows, out_features, in_features, stream);
        }
    } else {
        launch_dense_mmq_grad_input_tiled<type, 16, 16>(
            grad_output, packed_weight, grad_input,
            rows, out_features, in_features, stream);
    }
}

} // namespace torch_ggml_ops::ck
