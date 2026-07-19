#pragma once

#include "bf16_wmma.cuh"
#include "gguf_decode.cuh"
#include "grouped_mmq_backward_tiled.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace torch_ggml_ops::ck {

static constexpr int GROUPED_BACKWARD_WAVE_SIZE = 32;
static constexpr int GROUPED_BACKWARD_WAVES = 8;
static constexpr int GROUPED_BACKWARD_THREADS =
    GROUPED_BACKWARD_WAVE_SIZE * GROUPED_BACKWARD_WAVES;
static constexpr int GROUPED_BACKWARD_M_PER_WAVE = 16;
static constexpr int GROUPED_BACKWARD_M_PER_BLOCK =
    GROUPED_BACKWARD_M_PER_WAVE * GROUPED_BACKWARD_WAVES;
static constexpr int GROUPED_BACKWARD_N_PER_BLOCK = 16;
static constexpr int GROUPED_BACKWARD_K_PER_ITERATION = 16;
static_assert(
    GROUPED_BACKWARD_THREADS ==
        GROUPED_BACKWARD_N_PER_BLOCK * GROUPED_BACKWARD_K_PER_ITERATION,
    "one grouped backward thread must decode one shared BF16 weight value");

static __device__ __forceinline__ bool grouped_backward_metadata(
        const int64_t * expert_indices,
        const int32_t * expert_offsets,
        int group,
        int num_experts,
        int rows,
        int & row_begin,
        int & row_end,
        int64_t & expert) {
    row_begin = group == 0 ? 0 : expert_offsets[group - 1];
    row_end = expert_offsets[group];
    expert = expert_indices[group];
    return expert >= 0 && expert < num_experts && row_begin >= 0 &&
        row_end > row_begin && row_end <= rows;
}

template <ggml_type type>
__launch_bounds__(GROUPED_BACKWARD_THREADS, 1)
static __global__ void grouped_mmq_grad_input_kernel(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int rows,
        int out_features,
        int in_features,
        int blocks_per_weight_row,
        int64_t bytes_per_expert) {
    const int input_column_start =
        blockIdx.x * GROUPED_BACKWARD_N_PER_BLOCK;
    const int group = blockIdx.y;
    int row_begin;
    int row_end;
    int64_t expert;
    if (!grouped_backward_metadata(
            expert_indices,
            expert_offsets,
            group,
            num_experts,
            rows,
            row_begin,
            row_end,
            expert)) {
        return;
    }

    const int wave = threadIdx.x / GROUPED_BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % GROUPED_BACKWARD_WAVE_SIZE;
    const int64_t packed_row_bytes =
        static_cast<int64_t>(blocks_per_weight_row) * gguf_block_bytes<type>();
    const char * expert_weight = packed_weight + expert * bytes_per_expert;

    __shared__ __hip_bfloat16 shared_b[
        GROUPED_BACKWARD_N_PER_BLOCK * GROUPED_BACKWARD_K_PER_ITERATION];

    for (int block_row_start = row_begin; block_row_start < row_end;
         block_row_start += GROUPED_BACKWARD_M_PER_BLOCK) {
        const int wave_row_start =
            block_row_start + wave * GROUPED_BACKWARD_M_PER_WAVE;
        f32_accumulator accumulator;

        for (int output_start = 0; output_start < out_features;
             output_start += GROUPED_BACKWARD_K_PER_ITERATION) {
            const int index = threadIdx.x;
            const int k = index / GROUPED_BACKWARD_N_PER_BLOCK;
            const int local_input_column =
                index % GROUPED_BACKWARD_N_PER_BLOCK;
            const int output_column = output_start + k;
            const int input_column = input_column_start + local_input_column;
            if (input_column < in_features && output_column < out_features) {
                const char * packed_row =
                    expert_weight +
                    static_cast<int64_t>(output_column) * packed_row_bytes;
                shared_b[
                    local_input_column * GROUPED_BACKWARD_K_PER_ITERATION + k] =
                    __float2bfloat16(decode_gguf_value<type>(
                        packed_row,
                        input_column / QK_K,
                        input_column % QK_K));
            } else {
                shared_b[
                    local_input_column * GROUPED_BACKWARD_K_PER_ITERATION + k] =
                    __float2bfloat16(0.0f);
            }
            __syncthreads();

            bf16_fragment a_fragment{};
            bf16_fragment b_fragment{};
            __hip_bfloat16 * a = fragment_data(a_fragment);
            __hip_bfloat16 * b = fragment_data(b_fragment);
            const int a_row = wave_row_start + c_row(lane);
#pragma unroll
            for (int k_fragment = 0;
                 k_fragment < GROUPED_BACKWARD_K_PER_ITERATION;
                 ++k_fragment) {
                const int output_feature = output_start + k_fragment;
                a[k_fragment] =
                    a_row < row_end && output_feature < out_features
                    ? grad_output[
                          static_cast<int64_t>(a_row) * out_features +
                          output_feature]
                    : __float2bfloat16(0.0f);
                b[k_fragment] = shared_b[
                    c_row(lane) * GROUPED_BACKWARD_K_PER_ITERATION +
                    k_fragment];
            }

            wmma_f32_16x16x16_bf16(accumulator, a_fragment, b_fragment);
            __syncthreads();
        }

#pragma unroll
        for (int element = 0; element < 8; ++element) {
            const int output_row =
                wave_row_start + c_column(lane, element);
            const int output_column = input_column_start + c_row(lane);
            if (output_row < row_end && output_column < in_features) {
                grad_input[
                    static_cast<int64_t>(output_row) * in_features +
                    output_column] =
                    __float2bfloat16(accumulator.values[element]);
            }
        }
    }
}

template <ggml_type type>
static inline void launch_grouped_mmq_grad_input(
        const __hip_bfloat16 * grad_output,
        const char * packed_weight,
        __hip_bfloat16 * grad_input,
        const int64_t * expert_indices,
        const int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int out_features,
        int in_features,
        int64_t bytes_per_expert,
        hipStream_t stream) {
    if constexpr (type == GGML_TYPE_Q4_K) {
        if (out_features == GROUPED_BACKWARD_TILED_Q4_OUT_FEATURES &&
            in_features == GROUPED_BACKWARD_TILED_Q4_IN_FEATURES) {
            if (rows >= num_groups * GROUPED_BACKWARD_TILED_M) {
                launch_grouped_mmq_grad_input_q4_tiled(
                    grad_output,
                    packed_weight,
                    grad_input,
                    expert_indices,
                    expert_offsets,
                    num_experts,
                    num_groups,
                    rows,
                    bytes_per_expert,
                    stream);
            } else if (rows >= num_groups * 80) {
                launch_grouped_mmq_grad_input_q4_small_s2(
                    grad_output,
                    packed_weight,
                    grad_input,
                    expert_indices,
                    expert_offsets,
                    num_experts,
                    num_groups,
                    rows,
                    bytes_per_expert,
                    stream);
            } else {
                launch_grouped_mmq_grad_input_q4_small(
                    grad_output,
                    packed_weight,
                    grad_input,
                    expert_indices,
                    expert_offsets,
                    num_experts,
                    num_groups,
                    rows,
                    bytes_per_expert,
                    stream);
            }
            return;
        }
    }

    if constexpr (type == GGML_TYPE_Q5_K) {
        if (out_features == GROUPED_BACKWARD_TILED_Q5_OUT_FEATURES &&
            in_features == GROUPED_BACKWARD_TILED_Q5_IN_FEATURES) {
            if (rows >= num_groups * GROUPED_BACKWARD_TILED_M) {
                launch_grouped_mmq_grad_input_q5_tiled(
                    grad_output,
                    packed_weight,
                    grad_input,
                    expert_indices,
                    expert_offsets,
                    num_experts,
                    num_groups,
                    rows,
                    bytes_per_expert,
                    stream);
            } else {
                launch_grouped_mmq_grad_input_q5_small(
                    grad_output,
                    packed_weight,
                    grad_input,
                    expert_indices,
                    expert_offsets,
                    num_experts,
                    num_groups,
                    rows,
                    bytes_per_expert,
                    stream);
            }
            return;
        }
    }

    const dim3 grid(
        (in_features + GROUPED_BACKWARD_N_PER_BLOCK - 1) /
            GROUPED_BACKWARD_N_PER_BLOCK,
        num_groups,
        1);
    const dim3 block(GROUPED_BACKWARD_THREADS, 1, 1);
    grouped_mmq_grad_input_kernel<type><<<grid, block, 0, stream>>>(
        grad_output,
        packed_weight,
        grad_input,
        expert_indices,
        expert_offsets,
        num_experts,
        rows,
        out_features,
        in_features,
        in_features / QK_K,
        bytes_per_expert);
}

template <ggml_type type>
__launch_bounds__(GROUPED_BACKWARD_THREADS, 1)
static __global__ void grouped_mmq_pair_grad_input_kernel(
        const __hip_bfloat16 * __restrict__ first_grad_output,
        const __hip_bfloat16 * __restrict__ second_grad_output,
        const char * __restrict__ first_packed_weight,
        const char * __restrict__ second_packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int rows,
        int out_features,
        int in_features,
        int blocks_per_weight_row,
        int64_t bytes_per_expert) {
    const int input_column_start =
        blockIdx.x * GROUPED_BACKWARD_N_PER_BLOCK;
    const int group = blockIdx.y;
    int row_begin;
    int row_end;
    int64_t expert;
    if (!grouped_backward_metadata(
            expert_indices,
            expert_offsets,
            group,
            num_experts,
            rows,
            row_begin,
            row_end,
            expert)) {
        return;
    }

    const int wave = threadIdx.x / GROUPED_BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % GROUPED_BACKWARD_WAVE_SIZE;
    const int64_t packed_row_bytes =
        static_cast<int64_t>(blocks_per_weight_row) * gguf_block_bytes<type>();
    const char * first_expert_weight =
        first_packed_weight + expert * bytes_per_expert;
    const char * second_expert_weight =
        second_packed_weight + expert * bytes_per_expert;

    __shared__ __hip_bfloat16 shared_b[2][
        GROUPED_BACKWARD_N_PER_BLOCK * GROUPED_BACKWARD_K_PER_ITERATION];

    for (int block_row_start = row_begin; block_row_start < row_end;
         block_row_start += GROUPED_BACKWARD_M_PER_BLOCK) {
        const int wave_row_start =
            block_row_start + wave * GROUPED_BACKWARD_M_PER_WAVE;
        f32_accumulator accumulator;

        for (int output_start = 0; output_start < out_features;
             output_start += GROUPED_BACKWARD_K_PER_ITERATION) {
            const int index = threadIdx.x;
            const int k = index / GROUPED_BACKWARD_N_PER_BLOCK;
            const int local_input_column =
                index % GROUPED_BACKWARD_N_PER_BLOCK;
            const int output_column = output_start + k;
            const int input_column = input_column_start + local_input_column;
            if (input_column < in_features && output_column < out_features) {
                const int shared_index =
                    local_input_column * GROUPED_BACKWARD_K_PER_ITERATION + k;
                const int input_block = input_column / QK_K;
                const int input_offset = input_column % QK_K;
                const int64_t row_offset =
                    static_cast<int64_t>(output_column) * packed_row_bytes;
                shared_b[0][shared_index] = __float2bfloat16(
                    decode_gguf_value<type>(
                        first_expert_weight + row_offset,
                        input_block,
                        input_offset));
                shared_b[1][shared_index] = __float2bfloat16(
                    decode_gguf_value<type>(
                        second_expert_weight + row_offset,
                        input_block,
                        input_offset));
            } else {
                const int shared_index =
                    local_input_column * GROUPED_BACKWARD_K_PER_ITERATION + k;
                shared_b[0][shared_index] = __float2bfloat16(0.0f);
                shared_b[1][shared_index] = __float2bfloat16(0.0f);
            }
            __syncthreads();

            const int a_row = wave_row_start + c_row(lane);
            {
                bf16_fragment a_fragment{};
                bf16_fragment b_fragment{};
                __hip_bfloat16 * a = fragment_data(a_fragment);
                __hip_bfloat16 * b = fragment_data(b_fragment);
#pragma unroll
                for (int k_fragment = 0;
                     k_fragment < GROUPED_BACKWARD_K_PER_ITERATION;
                     ++k_fragment) {
                    const int output_feature = output_start + k_fragment;
                    const bool valid =
                        a_row < row_end && output_feature < out_features;
                    const int64_t grad_offset =
                        static_cast<int64_t>(a_row) * out_features + output_feature;
                    a[k_fragment] = valid
                        ? first_grad_output[grad_offset]
                        : __float2bfloat16(0.0f);
                    b[k_fragment] = shared_b[0][
                        c_row(lane) * GROUPED_BACKWARD_K_PER_ITERATION +
                        k_fragment];
                }
                wmma_f32_16x16x16_bf16(
                    accumulator, a_fragment, b_fragment);
            }
            {
                bf16_fragment a_fragment{};
                bf16_fragment b_fragment{};
                __hip_bfloat16 * a = fragment_data(a_fragment);
                __hip_bfloat16 * b = fragment_data(b_fragment);
#pragma unroll
                for (int k_fragment = 0;
                     k_fragment < GROUPED_BACKWARD_K_PER_ITERATION;
                     ++k_fragment) {
                    const int output_feature = output_start + k_fragment;
                    const bool valid =
                        a_row < row_end && output_feature < out_features;
                    const int64_t grad_offset =
                        static_cast<int64_t>(a_row) * out_features + output_feature;
                    a[k_fragment] = valid
                        ? second_grad_output[grad_offset]
                        : __float2bfloat16(0.0f);
                    b[k_fragment] = shared_b[1][
                        c_row(lane) * GROUPED_BACKWARD_K_PER_ITERATION +
                        k_fragment];
                }
                wmma_f32_16x16x16_bf16(
                    accumulator, a_fragment, b_fragment);
            }
            __syncthreads();
        }

#pragma unroll
        for (int element = 0; element < 8; ++element) {
            const int output_row =
                wave_row_start + c_column(lane, element);
            const int output_column = input_column_start + c_row(lane);
            if (output_row < row_end && output_column < in_features) {
                grad_input[
                    static_cast<int64_t>(output_row) * in_features +
                    output_column] =
                    __float2bfloat16(accumulator.values[element]);
            }
        }
    }
}

template <ggml_type type>
static inline void launch_grouped_mmq_pair_grad_input(
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
        int out_features,
        int in_features,
        int64_t bytes_per_expert,
        hipStream_t stream) {
    if constexpr (type == GGML_TYPE_Q3_K) {
        if (out_features == GROUPED_BACKWARD_TILED_Q3_OUT_FEATURES &&
            in_features == GROUPED_BACKWARD_TILED_Q3_IN_FEATURES) {
            if (rows >= num_groups * GROUPED_BACKWARD_TILED_M) {
                launch_grouped_mmq_pair_grad_input_q3_tiled(
                    first_grad_output,
                    second_grad_output,
                    first_packed_weight,
                    second_packed_weight,
                    grad_input,
                    expert_indices,
                    expert_offsets,
                    num_experts,
                    num_groups,
                    rows,
                    bytes_per_expert,
                    stream);
            } else {
                launch_grouped_mmq_pair_grad_input_q3_small(
                    first_grad_output,
                    second_grad_output,
                    first_packed_weight,
                    second_packed_weight,
                    grad_input,
                    expert_indices,
                    expert_offsets,
                    num_experts,
                    num_groups,
                    rows,
                    bytes_per_expert,
                    stream);
            }
            return;
        }
    }

    const dim3 grid(
        (in_features + GROUPED_BACKWARD_N_PER_BLOCK - 1) /
            GROUPED_BACKWARD_N_PER_BLOCK,
        num_groups,
        1);
    const dim3 block(GROUPED_BACKWARD_THREADS, 1, 1);
    grouped_mmq_pair_grad_input_kernel<type><<<grid, block, 0, stream>>>(
        first_grad_output,
        second_grad_output,
        first_packed_weight,
        second_packed_weight,
        grad_input,
        expert_indices,
        expert_offsets,
        num_experts,
        rows,
        out_features,
        in_features,
        in_features / QK_K,
        bytes_per_expert);
}

} // namespace torch_ggml_ops::ck
