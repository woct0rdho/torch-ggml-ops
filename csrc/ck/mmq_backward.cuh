#pragma once

#include "bf16_wmma.cuh"
#include "gguf_decode.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace torch_ggml_ops::ck {

static constexpr int BACKWARD_WAVE_SIZE = 32;
static constexpr int BACKWARD_WAVES = 4;
static constexpr int BACKWARD_THREADS = BACKWARD_WAVE_SIZE * BACKWARD_WAVES;
static constexpr int BACKWARD_M_PER_WAVE = 16;
static constexpr int BACKWARD_M_PER_BLOCK = BACKWARD_M_PER_WAVE * BACKWARD_WAVES;
static constexpr int BACKWARD_N_PER_BLOCK = 16;
static constexpr int BACKWARD_K_PER_ITERATION = 16;

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
            0xffffffff, scaled_d, 0, BACKWARD_N_PER_BLOCK);

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
__launch_bounds__(BACKWARD_THREADS, 2)
static __global__ void dense_mmq_grad_input_kernel(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        int rows,
        int out_features,
        int in_features,
        int blocks_per_weight_row) {
    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int block_row_start = blockIdx.x * BACKWARD_M_PER_BLOCK;
    const int wave_row_start = block_row_start + wave * BACKWARD_M_PER_WAVE;
    const int input_column_start = blockIdx.y * BACKWARD_N_PER_BLOCK;
    const int64_t packed_row_bytes =
        static_cast<int64_t>(blocks_per_weight_row) * gguf_block_bytes<type>();

    __shared__ __hip_bfloat16 shared_b[
        BACKWARD_N_PER_BLOCK * BACKWARD_K_PER_ITERATION];
    f32_accumulator accumulator;

    for (int output_start = 0; output_start < out_features;
         output_start += BACKWARD_K_PER_ITERATION) {
#pragma unroll
        for (int index = threadIdx.x; index < BACKWARD_N_PER_BLOCK * BACKWARD_K_PER_ITERATION;
             index += BACKWARD_THREADS) {
            const int k = index / BACKWARD_N_PER_BLOCK;
            const int local_input_column = index % BACKWARD_N_PER_BLOCK;
            const int output_column = output_start + k;
            const int input_column = input_column_start + local_input_column;
            if (input_column < in_features && output_column < out_features) {
                const char * packed_row =
                    packed_weight + static_cast<int64_t>(output_column) * packed_row_bytes;
                shared_b[local_input_column * BACKWARD_K_PER_ITERATION + k] =
                    decode_backward_tile_value<type>(
                        packed_row,
                        input_column / QK_K,
                        input_column % QK_K,
                        local_input_column);
            } else {
                shared_b[local_input_column * BACKWARD_K_PER_ITERATION + k] =
                    __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        bf16_fragment a_fragment{};
        bf16_fragment b_fragment{};
        __hip_bfloat16 * a = fragment_data(a_fragment);
        __hip_bfloat16 * b = fragment_data(b_fragment);
        const int a_row = wave_row_start + c_row(lane);
#pragma unroll
        for (int k = 0; k < BACKWARD_K_PER_ITERATION; ++k) {
            const int output_column = output_start + k;
            a[k] = a_row < rows && output_column < out_features
                ? grad_output[static_cast<int64_t>(a_row) * out_features + output_column]
                : __float2bfloat16(0.0f);
            b[k] = shared_b[c_row(lane) * BACKWARD_K_PER_ITERATION + k];
        }

        wmma_f32_16x16x16_bf16(accumulator, a_fragment, b_fragment);
        __syncthreads();
    }

#pragma unroll
    for (int element = 0; element < 8; ++element) {
        // gfx11's physical C fragment is J-major for this A/B layout: the
        // I-major lane coordinates are transposed when written to row-major C.
        const int output_row = wave_row_start + c_column(lane, element);
        const int output_column = input_column_start + c_row(lane);
        if (output_row < rows && output_column < in_features) {
            grad_input[static_cast<int64_t>(output_row) * in_features + output_column] =
                __float2bfloat16(accumulator.values[element]);
        }
    }
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
    const dim3 grid(
        (rows + BACKWARD_M_PER_BLOCK - 1) / BACKWARD_M_PER_BLOCK,
        (in_features + BACKWARD_N_PER_BLOCK - 1) / BACKWARD_N_PER_BLOCK,
        1);
    const dim3 block(BACKWARD_THREADS, 1, 1);
    dense_mmq_grad_input_kernel<type><<<grid, block, 0, stream>>>(
        grad_output,
        packed_weight,
        grad_input,
        rows,
        out_features,
        in_features,
        in_features / QK_K);
}

} // namespace torch_ggml_ops::ck
