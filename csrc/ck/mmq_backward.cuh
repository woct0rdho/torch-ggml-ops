#pragma once

#include "bf16_wmma.cuh"
#include "gguf_decode.cuh"
#include "../vendor/llama_cpp/iq2_xxs_grid.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace torch_ggml_ops::ck {

static constexpr int BACKWARD_WAVE_SIZE = 32;
static constexpr int BACKWARD_WAVES = 4;
static constexpr int BACKWARD_THREADS = BACKWARD_WAVE_SIZE * BACKWARD_WAVES;
static constexpr int BACKWARD_M_PER_TILE = 16;
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
    } else if constexpr (type == GGML_TYPE_Q2_0) {
        const auto & block =
            reinterpret_cast<const block_q2_0 *>(packed_row)[block_index];
        const float d = fp16_to_fp32(block.d);
        const int first_code =
            (block.qs[value_index >> 2] >> (2 * (value_index & 3))) & 0x03;
        const int second_value = value_index + 1;
        const int second_code =
            (block.qs[second_value >> 2] >> (2 * (second_value & 3))) & 0x03;
        first = __float2bfloat16(d * static_cast<float>(first_code - 1));
        second = __float2bfloat16(d * static_cast<float>(second_code - 1));
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
    } else if constexpr (type == GGML_TYPE_Q2_0) {
        const auto & block =
            reinterpret_cast<const block_q2_0 *>(packed_row)[block_index];
        const float d = fp16_to_fp32(block.d);
#pragma unroll
        for (int index = 0; index < 4; ++index) {
            const int value = value_index + index;
            const int code =
                (block.qs[value >> 2] >> (2 * (value & 3))) & 0x03;
            values[index] = __float2bfloat16(
                d * static_cast<float>(code - 1));
        }
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        const auto & block =
            reinterpret_cast<const block_q4_0 *>(packed_row)[block_index];
        const float d = fp16_to_fp32(block.d);
#pragma unroll
        for (int index = 0; index < 4; ++index) {
            const int value = value_index + index;
            const int quant = (
                block.qs[value & (QK4_0 / 2 - 1)] >>
                (4 * ((value >> 4) & 1))) & 0x0f;
            values[index] = __float2bfloat16(
                d * static_cast<float>(quant - 8));
        }
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        const auto & block =
            reinterpret_cast<const block_q5_0 *>(packed_row)[block_index];
        const float d = fp16_to_fp32(block.d);
        const uint32_t fifth = get_int_b4(block.qh, 0);
#pragma unroll
        for (int index = 0; index < 4; ++index) {
            const int value = value_index + index;
            const int low = (
                block.qs[value & (QK5_0 / 2 - 1)] >>
                (4 * ((value >> 4) & 1))) & 0x0f;
            const int high = (fifth >> value) & 0x01;
            values[index] = __float2bfloat16(
                d * static_cast<float>((low | (high << 4)) - 16));
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

template <ggml_type type, int WIDTH>
static __device__ __forceinline__ void decode_backward_tile_group(
        const char * packed_row,
        int block_index,
        int value_index,
        __hip_bfloat16 * values) {
    if constexpr (type == GGML_TYPE_Q8_0) {
        const auto & block =
            reinterpret_cast<const block_q8_0 *>(packed_row)[block_index];
        const float d = fp16_to_fp32(block.d);
#pragma unroll
        for (int index = 0; index < WIDTH; ++index) {
            values[index] = __float2bfloat16(
                d * static_cast<float>(block.qs[value_index + index]));
        }
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        const auto & block =
            reinterpret_cast<const block_q2_K *>(packed_row)[block_index];
        const int group = value_index >> 4;
        const int group_in_half = group & 7;
        const int half = group >> 3;
        const int shift = (group_in_half >> 1) * 2;
        const int byte = half * 32 + (group_in_half & 1) * 16;
        const uint8_t scale_min = block.scales[group];
        const float scale = fp16_to_fp32(block.d) *
            static_cast<float>(scale_min & 0x0f);
        const float minimum = fp16_to_fp32(block.dmin) *
            static_cast<float>(scale_min >> 4);
#pragma unroll
        for (int index = 0; index < WIDTH; ++index) {
            const int quant = (block.qs[byte + index] >> shift) & 0x03;
            values[index] = __float2bfloat16(
                scale * static_cast<float>(quant) - minimum);
        }
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        const auto & block =
            reinterpret_cast<const block_iq2_xxs *>(packed_row)[block_index];
        const int group = value_index >> 5;
        const int first_sub_group = (value_index & 31) >> 3;
        const int q_word = 2 * group;
        const uint32_t grid_data = static_cast<uint32_t>(
            get_int_b2(block.qs, q_word));
        const uint32_t sign_data = static_cast<uint32_t>(
            get_int_b2(block.qs, q_word + 1));
        const int first_grid_index =
            (grid_data >> (8 * first_sub_group)) & 0xff;
        const int second_grid_index =
            (grid_data >> (8 * (first_sub_group + 1))) & 0xff;
        const uint64_t first_grid = iq2xxs_grid[first_grid_index];
        const uint64_t second_grid = iq2xxs_grid[second_grid_index];
        const uint8_t first_signs = static_cast<uint8_t>(
            unpack_ksigns(sign_data >> (7 * first_sub_group)));
        const uint8_t second_signs = static_cast<uint8_t>(
            unpack_ksigns(sign_data >> (7 * (first_sub_group + 1))));
        const float db = fp16_to_fp32(block.d) *
            (0.5f + static_cast<float>(sign_data >> 28)) * 0.25f;
#pragma unroll
        for (int index = 0; index < WIDTH; ++index) {
            const int local = index & 7;
            const uint64_t grid = index < 8 ? first_grid : second_grid;
            const uint8_t signs = index < 8 ? first_signs : second_signs;
            const int magnitude =
                static_cast<int>((grid >> (8 * local)) & 0xff);
            const int sign = ((signs >> local) & 0x01) == 0 ? 1 : -1;
            values[index] = __float2bfloat16(
                db * static_cast<float>(magnitude * sign));
        }
    } else if constexpr (type == GGML_TYPE_Q3_K) {
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
        const int high_shift = value_index >> 5;
        const int high_byte = value_index & 31;
#pragma unroll
        for (int index = 0; index < WIDTH; ++index) {
            const int low = (block.qs[low_byte + index] >> low_shift) & 0x03;
            const int high =
                ((block.hmask[high_byte + index] >> high_shift) & 0x01) ^ 0x01;
            values[index] = __float2bfloat16(
                scaled_d * static_cast<float>(low - (high << 2)));
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
        for (int index = 0; index < WIDTH; ++index) {
            const int quant = (block.qs[byte + index] >> shift) & 0x0f;
            values[index] = __float2bfloat16(
                d * static_cast<float>(quant) - minimum);
        }
    } else if constexpr (type == GGML_TYPE_Q2_0) {
        const auto & block =
            reinterpret_cast<const block_q2_0 *>(packed_row)[block_index];
        const float d = fp16_to_fp32(block.d);
#pragma unroll
        for (int index = 0; index < WIDTH; ++index) {
            const int value = value_index + index;
            const int code =
                (block.qs[value >> 2] >> (2 * (value & 3))) & 0x03;
            values[index] = __float2bfloat16(
                d * static_cast<float>(code - 1));
        }
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        const auto & block =
            reinterpret_cast<const block_q4_0 *>(packed_row)[block_index];
        const float d = fp16_to_fp32(block.d);
#pragma unroll
        for (int index = 0; index < WIDTH; ++index) {
            const int value = value_index + index;
            const int quant = (
                block.qs[value & (QK4_0 / 2 - 1)] >>
                (4 * ((value >> 4) & 1))) & 0x0f;
            values[index] = __float2bfloat16(
                d * static_cast<float>(quant - 8));
        }
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        const auto & block =
            reinterpret_cast<const block_q5_0 *>(packed_row)[block_index];
        const float d = fp16_to_fp32(block.d);
        const uint32_t fifth = get_int_b4(block.qh, 0);
#pragma unroll
        for (int index = 0; index < WIDTH; ++index) {
            const int value = value_index + index;
            const int low = (
                block.qs[value & (QK5_0 / 2 - 1)] >>
                (4 * ((value >> 4) & 1))) & 0x0f;
            const int high = (fifth >> value) & 0x01;
            values[index] = __float2bfloat16(
                d * static_cast<float>((low | (high << 4)) - 16));
        }
    } else if constexpr (type == GGML_TYPE_Q6_K && WIDTH == 16) {
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
        for (int index = 0; index < WIDTH; ++index) {
            const int low = (block.ql[low_byte + index] >> low_shift) & 0x0f;
            const int high =
                (block.qh[high_byte + index] >> high_shift) & 0x03;
            values[index] = __float2bfloat16(
                scaled_d * static_cast<float>((low | (high << 4)) - 32));
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
        const int high_byte = value_index & 31;
#pragma unroll
        for (int index = 0; index < WIDTH; ++index) {
            const int low = (block.qs[byte + index] >> shift) & 0x0f;
            const int high = (block.qh[high_byte + index] >> group) & 0x01;
            values[index] = __float2bfloat16(
                d * static_cast<float>(low | (high << 4)) - minimum);
        }
    }
}

static __device__ __forceinline__ uint4 load_uint4_unaligned(
        const void * source) {
    uint4 value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

static __device__ __forceinline__ uint32_t load_uint32_unaligned(
        const void * source) {
    uint32_t value;
    __builtin_memcpy(&value, source, sizeof(value));
    return value;
}

static __device__ __forceinline__ void decode_backward_tile_q3_preloaded(
        const block_q3_K & block,
        int value_index,
        const uint4 & packed_low,
        const uint4 & packed_high,
        __hip_bfloat16 * values) {
    const int scale_group = value_index >> 4;
    const int low_scale = scale_group < 8
        ? block.scales[scale_group]
        : block.scales[scale_group - 8] >> 4;
    const int high_scale =
        block.scales[8 + (scale_group & 3)] >> (2 * (scale_group >> 2));
    const int scale =
        ((low_scale & 0x0f) | ((high_scale & 0x03) << 4)) - 32;
    const float scaled_d = fp16_to_fp32(block.d) * static_cast<float>(scale);
    const int low_shift = 2 * ((value_index & 127) >> 5);
    const int high_shift = value_index >> 5;
    const auto * low_words = reinterpret_cast<const uint32_t *>(&packed_low);
    const auto * high_words = reinterpret_cast<const uint32_t *>(&packed_high);
#pragma unroll
    for (int word = 0; word < 4; ++word) {
        const uint32_t quant_plus_four_bytes =
            ((low_words[word] >> low_shift) & 0x03030303U) |
            (((high_words[word] >> high_shift) & 0x01010101U) << 2);
#pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
            const int quant =
                static_cast<int>(
                    (quant_plus_four_bytes >> (8 * byte)) & 0x07U) - 4;
            values[4 * word + byte] = __float2bfloat16(
                scaled_d * static_cast<float>(quant));
        }
    }
}

static __device__ __forceinline__ void decode_backward_tile_q4_preloaded(
        const block_q4_K & block,
        int value_index,
        const uint4 & packed_quants,
        __hip_bfloat16 * values) {
    const int group = value_index >> 5;
    const float d = fp16_to_fp32(block.d) *
        static_cast<float>(k_scale(block.scales, group));
    const float minimum = fp16_to_fp32(block.dmin) *
        static_cast<float>(k_min(block.scales, group));
    const int shift = 4 * (group & 1);
    const auto * quant_bytes =
        reinterpret_cast<const uint8_t *>(&packed_quants);
#pragma unroll
    for (int index = 0; index < 16; ++index) {
        const int quant = (quant_bytes[index] >> shift) & 0x0f;
        values[index] = __float2bfloat16(
            d * static_cast<float>(quant) - minimum);
    }
}

static __device__ __forceinline__ void decode_backward_tile_q2_preloaded(
        const block_q2_0 & block,
        uint32_t packed_codes,
        __hip_bfloat16 * values) {
    const float d = fp16_to_fp32(block.d);
    const float negated = -d;
    const float doubled = d + d;
#pragma unroll
    for (int index = 0; index < 16; ++index) {
        const int code = (packed_codes >> (2 * index)) & 0x03;
        const float level = code == 0   ? negated
                            : code == 1 ? 0.0f
                            : code == 2 ? d
                                        : doubled;
        values[index] = __float2bfloat16(level);
    }
}

template <bool PACK_QUANT_BYTES>
static __device__ __forceinline__ void decode_backward_tile_q5_preloaded(
        const block_q5_K & block,
        int value_index,
        const uint4 & packed_low,
        const uint4 & packed_high,
        __hip_bfloat16 * values) {
    const int group = value_index >> 5;
    const float d = fp16_to_fp32(block.d) *
        static_cast<float>(k_scale(block.scales, group));
    const float minimum = fp16_to_fp32(block.dmin) *
        static_cast<float>(k_min(block.scales, group));
    const int shift = 4 * (group & 1);
    if constexpr (PACK_QUANT_BYTES) {
        const auto * low_words =
            reinterpret_cast<const uint32_t *>(&packed_low);
        const auto * high_words =
            reinterpret_cast<const uint32_t *>(&packed_high);
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            const uint32_t quant_bytes =
                ((low_words[word] >> shift) & 0x0f0f0f0fU) |
                (((high_words[word] >> group) & 0x01010101U) << 4);
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const int quant = (quant_bytes >> (8 * byte)) & 0x1f;
                values[4 * word + byte] = __float2bfloat16(
                    d * static_cast<float>(quant) - minimum);
            }
        }
    } else {
        const auto * low_bytes =
            reinterpret_cast<const uint8_t *>(&packed_low);
        const auto * high_bytes =
            reinterpret_cast<const uint8_t *>(&packed_high);
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            const int low = (low_bytes[index] >> shift) & 0x0f;
            const int high = (high_bytes[index] >> group) & 0x01;
            values[index] = __float2bfloat16(
                d * static_cast<float>(low | (high << 4)) - minimum);
        }
    }
}

template <bool PACK_QUANT_BYTES>
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
    if constexpr (PACK_QUANT_BYTES) {
        const uint4 packed_low = load_uint4_unaligned(block.ql + low_byte);
        const uint4 packed_high = load_uint4_unaligned(block.qh + high_byte);
        const auto * low_words =
            reinterpret_cast<const uint32_t *>(&packed_low);
        const auto * high_words =
            reinterpret_cast<const uint32_t *>(&packed_high);
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            const uint32_t quant_bytes =
                ((low_words[word] >> low_shift) & 0x0f0f0f0fU) |
                (((high_words[word] >> high_shift) & 0x03030303U) << 4);
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const int quant =
                    static_cast<int>(
                        (quant_bytes >> (8 * byte)) & 0x3fU) - 32;
                values[4 * word + byte] = __float2bfloat16(
                    scaled_d * static_cast<float>(quant));
            }
        }
    } else {
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            const int low =
                (block.ql[low_byte + index] >> low_shift) & 0x0f;
            const int high =
                (block.qh[high_byte + index] >> high_shift) & 0x03;
            values[index] = __float2bfloat16(
                scaled_d * static_cast<float>((low | (high << 4)) - 32));
        }
    }
}

template <int ROWS, int COLUMNS, int PADDING, int SWIZZLE_CHUNK>
struct backward_shared_b_tile {
    static_assert(COLUMNS % 16 == 0);
    static_assert(SWIZZLE_CHUNK == 0 || COLUMNS % SWIZZLE_CHUNK == 0);

    alignas(16) __hip_bfloat16 values[ROWS * (COLUMNS + PADDING)];

    static __device__ __forceinline__ int physical_index(int index) {
        const int row = index / COLUMNS;
        const int column = index % COLUMNS;
        if constexpr (SWIZZLE_CHUNK > 0) {
            constexpr int chunks_per_row = COLUMNS / SWIZZLE_CHUNK;
            static_assert((chunks_per_row & (chunks_per_row - 1)) == 0);
            const int chunk =
                (column / SWIZZLE_CHUNK) ^ (row & (chunks_per_row - 1));
            return row * (COLUMNS + PADDING) +
                chunk * SWIZZLE_CHUNK + column % SWIZZLE_CHUNK;
        } else {
            return row * (COLUMNS + PADDING) + column;
        }
    }

    __device__ __forceinline__ __hip_bfloat16 & operator[](int index) {
        return values[physical_index(index)];
    }

    __device__ __forceinline__ const __hip_bfloat16 & operator[](
            int index) const {
        return values[physical_index(index)];
    }

    __device__ __forceinline__ void load_fragment_vector(
            bf16_fragment & fragment,
            int row,
            int column) const {
        if constexpr (SWIZZLE_CHUNK == 4) {
            auto * destination = reinterpret_cast<uint2 *>(&fragment);
#pragma unroll
            for (int chunk = 0; chunk < 4; ++chunk) {
                const auto * source = reinterpret_cast<const uint2 *>(
                    values + physical_index(
                        row * COLUMNS + column + 4 * chunk));
                destination[chunk] = source[0];
            }
        } else {
            const auto * first = reinterpret_cast<const uint4 *>(
                values + physical_index(row * COLUMNS + column));
            const auto * second = reinterpret_cast<const uint4 *>(
                values + physical_index(row * COLUMNS + column + 8));
            auto * destination = reinterpret_cast<uint4 *>(&fragment);
            destination[0] = first[0];
            destination[1] = second[0];
        }
    }
};

template <
    ggml_type type,
    int EXACT_OUT_FEATURES,
    int EXACT_IN_FEATURES,
    int N_TILES,
    int K_ITERATION,
    int GROUP_M,
    int M_TILES_PER_WAVE,
    int ACTIVE_WAVES,
    int DECODER_WIDTH,
    bool PREFETCH_LOCAL,
    bool FULL_TILES,
    bool PREFETCH_PACKED,
    int LDS_PADDING,
    bool VECTOR_LOCAL_LOAD,
    int LDS_SWIZZLE_CHUNK,
    bool PACK_Q5_QUANT_BYTES,
    bool PACK_Q6_QUANT_BYTES>
static __device__ __forceinline__ void dense_mmq_grad_input_body(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        int rows,
        int out_features,
        int in_features,
        int blocks_per_weight_row) {
    constexpr int N_PER_BLOCK = N_TILES * BACKWARD_N_PER_TILE;
    constexpr int WEIGHT_BLOCK_VALUES =
        type == GGML_TYPE_Q8_0 ? QK8_0
                               : (type == GGML_TYPE_Q2_0 ? QK2_0 : QK_K);
    const int kernel_out_features = EXACT_OUT_FEATURES > 0
        ? EXACT_OUT_FEATURES
        : out_features;
    const int kernel_in_features = EXACT_IN_FEATURES > 0
        ? EXACT_IN_FEATURES
        : in_features;
    const int kernel_blocks_per_weight_row = EXACT_IN_FEATURES > 0
        ? EXACT_IN_FEATURES / WEIGHT_BLOCK_VALUES
        : blocks_per_weight_row;
    static_assert(ACTIVE_WAVES > 0 && ACTIVE_WAVES <= BACKWARD_WAVES);
    constexpr int M_PER_WAVE = M_TILES_PER_WAVE * BACKWARD_M_PER_TILE;
    constexpr int M_PER_BLOCK = M_PER_WAVE * ACTIVE_WAVES;
    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int m_block = GROUP_M > 0
        ? blockIdx.z * GROUP_M + blockIdx.x
        : blockIdx.x;
    const int block_row_start = m_block * M_PER_BLOCK;
    const int wave_row_start = block_row_start + wave * M_PER_WAVE;
    const int input_column_start = blockIdx.y * N_PER_BLOCK;
    const int64_t packed_row_bytes =
        static_cast<int64_t>(kernel_blocks_per_weight_row) *
        gguf_block_bytes<type>();

    __shared__ backward_shared_b_tile<
        N_PER_BLOCK, K_ITERATION, LDS_PADDING, LDS_SWIZZLE_CHUNK> shared_b;
    f32_accumulator accumulators[M_TILES_PER_WAVE][N_TILES];

    for (int output_start = 0; output_start < kernel_out_features;
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
                if (FULL_TILES ||
                    (input_column + 15 < kernel_in_features &&
                     output_column < kernel_out_features)
                ) {
                    const char * packed_row = packed_weight +
                        static_cast<int64_t>(output_column) * packed_row_bytes;
                    __hip_bfloat16 values[16];
                    decode_backward_tile_sixteen_q6<PACK_Q6_QUANT_BYTES>(
                        packed_row,
                        input_column / WEIGHT_BLOCK_VALUES,
                        input_column % WEIGHT_BLOCK_VALUES,
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
            PREFETCH_PACKED && type == GGML_TYPE_Q2_0 &&
            N_TILES == 8 && K_ITERATION == 32 && DECODER_WIDTH == 16
        ) {
            const int local_input_column = 16 * (threadIdx.x & 7);
            const int first_k = threadIdx.x >> 3;
            const int second_k = first_k + 16;
            const int input_column = input_column_start + local_input_column;
            const int block_index = input_column / WEIGHT_BLOCK_VALUES;
            const int value_index = input_column % WEIGHT_BLOCK_VALUES;
            const int byte = value_index >> 2;
            const char * first_packed_row = packed_weight +
                static_cast<int64_t>(output_start + first_k) * packed_row_bytes;
            const char * second_packed_row = packed_weight +
                static_cast<int64_t>(output_start + second_k) * packed_row_bytes;
            const auto & first_block =
                reinterpret_cast<const block_q2_0 *>(first_packed_row)[block_index];
            const auto & second_block =
                reinterpret_cast<const block_q2_0 *>(second_packed_row)[block_index];
            const uint32_t first_codes =
                load_uint32_unaligned(first_block.qs + byte);
            const uint32_t second_codes =
                load_uint32_unaligned(second_block.qs + byte);
            __hip_bfloat16 values[16];
            decode_backward_tile_q2_preloaded(first_block, first_codes, values);
#pragma unroll
            for (int index = 0; index < 16; ++index) {
                shared_b[
                    (local_input_column + index) * K_ITERATION + first_k] =
                    values[index];
            }
            decode_backward_tile_q2_preloaded(second_block, second_codes, values);
#pragma unroll
            for (int index = 0; index < 16; ++index) {
                shared_b[
                    (local_input_column + index) * K_ITERATION + second_k] =
                    values[index];
            }
        } else if constexpr (
            PREFETCH_PACKED && type == GGML_TYPE_Q4_K &&
            N_TILES == 8 && K_ITERATION == 32 && DECODER_WIDTH == 16
        ) {
            const int local_input_column = 16 * (threadIdx.x & 7);
            const int first_k = threadIdx.x >> 3;
            const int second_k = first_k + 16;
            const int input_column = input_column_start + local_input_column;
            const int block_index = input_column / WEIGHT_BLOCK_VALUES;
            const int value_index = input_column % WEIGHT_BLOCK_VALUES;
            const int group = value_index >> 5;
            const int byte = (group >> 1) * 32 + (value_index & 31);
            const char * first_packed_row = packed_weight +
                static_cast<int64_t>(output_start + first_k) * packed_row_bytes;
            const char * second_packed_row = packed_weight +
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
                shared_b[(local_input_column + index) * K_ITERATION + first_k] =
                    values[index];
            }
            decode_backward_tile_q4_preloaded(
                second_block, value_index, second_quants, values);
#pragma unroll
            for (int index = 0; index < 16; ++index) {
                shared_b[(local_input_column + index) * K_ITERATION + second_k] =
                    values[index];
            }
        } else if constexpr (
            PREFETCH_PACKED && type == GGML_TYPE_Q3_K &&
            N_TILES == 8 && K_ITERATION == 32 && DECODER_WIDTH == 16
        ) {
            const int local_input_column = 16 * (threadIdx.x & 7);
            const int first_k = threadIdx.x >> 3;
            const int second_k = first_k + 16;
            const int input_column = input_column_start + local_input_column;
            const int block_index = input_column / WEIGHT_BLOCK_VALUES;
            const int value_index = input_column % WEIGHT_BLOCK_VALUES;
            const int low_byte =
                (value_index >> 7) * 32 + (value_index & 31);
            const int high_byte = value_index & 31;
            const char * first_packed_row = packed_weight +
                static_cast<int64_t>(output_start + first_k) * packed_row_bytes;
            const char * second_packed_row = packed_weight +
                static_cast<int64_t>(output_start + second_k) * packed_row_bytes;
            const auto & first_block =
                reinterpret_cast<const block_q3_K *>(first_packed_row)[block_index];
            const auto & second_block =
                reinterpret_cast<const block_q3_K *>(second_packed_row)[block_index];
            const uint4 first_low =
                load_uint4_unaligned(first_block.qs + low_byte);
            const uint4 first_high =
                load_uint4_unaligned(first_block.hmask + high_byte);
            const uint4 second_low =
                load_uint4_unaligned(second_block.qs + low_byte);
            const uint4 second_high =
                load_uint4_unaligned(second_block.hmask + high_byte);
            __hip_bfloat16 values[16];
            decode_backward_tile_q3_preloaded(
                first_block, value_index, first_low, first_high, values);
#pragma unroll
            for (int index = 0; index < 16; ++index) {
                shared_b[(local_input_column + index) * K_ITERATION + first_k] =
                    values[index];
            }
            decode_backward_tile_q3_preloaded(
                second_block, value_index, second_low, second_high, values);
#pragma unroll
            for (int index = 0; index < 16; ++index) {
                shared_b[(local_input_column + index) * K_ITERATION + second_k] =
                    values[index];
            }
        } else if constexpr (
            PREFETCH_PACKED && type == GGML_TYPE_Q5_K &&
            N_TILES == 8 && K_ITERATION == 32 && DECODER_WIDTH == 16
        ) {
            const int local_input_column = 16 * (threadIdx.x & 7);
            const int first_k = threadIdx.x >> 3;
            const int second_k = first_k + 16;
            const int input_column = input_column_start + local_input_column;
            const int block_index = input_column / WEIGHT_BLOCK_VALUES;
            const int value_index = input_column % WEIGHT_BLOCK_VALUES;
            const int group = value_index >> 5;
            const int low_byte = (group >> 1) * 32 + (value_index & 31);
            const int high_byte = value_index & 31;
            const char * first_packed_row = packed_weight +
                static_cast<int64_t>(output_start + first_k) * packed_row_bytes;
            const char * second_packed_row = packed_weight +
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
            decode_backward_tile_q5_preloaded<PACK_Q5_QUANT_BYTES>(
                first_block, value_index, first_low, first_high, values);
#pragma unroll
            for (int index = 0; index < 16; ++index) {
                shared_b[(local_input_column + index) * K_ITERATION + first_k] =
                    values[index];
            }
            decode_backward_tile_q5_preloaded<PACK_Q5_QUANT_BYTES>(
                second_block, value_index, second_low, second_high, values);
#pragma unroll
            for (int index = 0; index < 16; ++index) {
                shared_b[(local_input_column + index) * K_ITERATION + second_k] =
                    values[index];
            }
        } else if constexpr (
            DECODER_WIDTH > 0 &&
            (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q2_0 ||
             type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K ||
             type == GGML_TYPE_Q5_K)
        ) {
            constexpr int groups_per_row = N_PER_BLOCK / DECODER_WIDTH;
#pragma unroll
            for (int group_index = threadIdx.x;
                 group_index < groups_per_row * K_ITERATION;
                 group_index += BACKWARD_THREADS) {
                const int k = group_index / groups_per_row;
                const int local_input_column =
                    DECODER_WIDTH * (group_index % groups_per_row);
                const int output_column = output_start + k;
                const int input_column = input_column_start + local_input_column;
                if constexpr (FULL_TILES) {
                    const char * packed_row = packed_weight +
                        static_cast<int64_t>(output_column) * packed_row_bytes;
                    __hip_bfloat16 values[DECODER_WIDTH];
                    decode_backward_tile_group<type, DECODER_WIDTH>(
                        packed_row,
                        input_column / WEIGHT_BLOCK_VALUES,
                        input_column % WEIGHT_BLOCK_VALUES,
                        values);
#pragma unroll
                    for (int index = 0; index < DECODER_WIDTH; ++index) {
                        shared_b[
                            (local_input_column + index) * K_ITERATION + k] =
                            values[index];
                    }
                } else if (
                    input_column + DECODER_WIDTH - 1 < kernel_in_features &&
                    output_column < kernel_out_features
                ) {
                    const char * packed_row = packed_weight +
                        static_cast<int64_t>(output_column) * packed_row_bytes;
                    __hip_bfloat16 values[DECODER_WIDTH];
                    decode_backward_tile_group<type, DECODER_WIDTH>(
                        packed_row,
                        input_column / WEIGHT_BLOCK_VALUES,
                        input_column % WEIGHT_BLOCK_VALUES,
                        values);
#pragma unroll
                    for (int index = 0; index < DECODER_WIDTH; ++index) {
                        shared_b[
                            (local_input_column + index) * K_ITERATION + k] =
                            values[index];
                    }
                } else {
#pragma unroll
                    for (int index = 0; index < DECODER_WIDTH; ++index) {
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
                if (input_column + 3 < kernel_in_features &&
                    output_column < kernel_out_features) {
                    const char * packed_row = packed_weight +
                        static_cast<int64_t>(output_column) * packed_row_bytes;
                    __hip_bfloat16 values[4];
                    decode_backward_tile_quad<type>(
                        packed_row,
                        input_column / WEIGHT_BLOCK_VALUES,
                        input_column % WEIGHT_BLOCK_VALUES,
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
                if (input_column + 1 < kernel_in_features &&
                    output_column < kernel_out_features) {
                    const char * packed_row = packed_weight +
                        static_cast<int64_t>(output_column) * packed_row_bytes;
                    decode_backward_tile_pair<type>(
                        packed_row,
                        input_column / WEIGHT_BLOCK_VALUES,
                        input_column % WEIGHT_BLOCK_VALUES,
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
                if (input_column < kernel_in_features &&
                    output_column < kernel_out_features) {
                    const char * packed_row = packed_weight +
                        static_cast<int64_t>(output_column) * packed_row_bytes;
                    shared_b[local_input_column * K_ITERATION + k] =
                        decode_backward_tile_value<type>(
                            packed_row,
                            input_column / WEIGHT_BLOCK_VALUES,
                            input_column % WEIGHT_BLOCK_VALUES,
                            local_input_column % BACKWARD_N_PER_TILE);
                } else {
                    shared_b[local_input_column * K_ITERATION + k] =
                        __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        if (ACTIVE_WAVES == BACKWARD_WAVES || wave < ACTIVE_WAVES) {
#pragma unroll
            for (int k_tile = 0; k_tile < K_ITERATION; k_tile += 16) {
                bf16_fragment a_fragments[M_TILES_PER_WAVE];
#pragma unroll
                for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
                    __hip_bfloat16 *a = fragment_data(a_fragments[m_tile]);
                    const int a_row =
                        wave_row_start + m_tile * BACKWARD_M_PER_TILE + c_row(lane);
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        const int output_column = output_start + k_tile + k;
                        if constexpr (FULL_TILES) {
                            a[k] = grad_output[static_cast<int64_t>(a_row) *
                                                   kernel_out_features +
                                               output_column];
                        } else {
                            a[k] = a_row < rows && output_column < kernel_out_features
                                       ? grad_output[static_cast<int64_t>(a_row) *
                                                         kernel_out_features +
                                                     output_column]
                                       : __float2bfloat16(0.0f);
                        }
                    }
                }

                if constexpr (PREFETCH_LOCAL) {
#pragma unroll
                    for (int n_tile = 0; n_tile < N_TILES - 1; n_tile += 2) {
                        bf16_fragment b_first{};
                        bf16_fragment b_second{};
                        if constexpr (VECTOR_LOCAL_LOAD) {
                            shared_b.load_fragment_vector(
                                b_first, n_tile * BACKWARD_N_PER_TILE + c_row(lane),
                                k_tile);
                            shared_b.load_fragment_vector(
                                b_second,
                                (n_tile + 1) * BACKWARD_N_PER_TILE + c_row(lane), k_tile);
                        } else {
                            __hip_bfloat16 *first = fragment_data(b_first);
                            __hip_bfloat16 *second = fragment_data(b_second);
#pragma unroll
                            for (int k = 0; k < 16; ++k) {
                                first[k] =
                                    shared_b[(n_tile * BACKWARD_N_PER_TILE + c_row(lane)) *
                                                 K_ITERATION +
                                             k_tile + k];
                                second[k] = shared_b[((n_tile + 1) * BACKWARD_N_PER_TILE +
                                                      c_row(lane)) *
                                                         K_ITERATION +
                                                     k_tile + k];
                            }
                        }
#pragma unroll
                        for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
                            wmma_f32_16x16x16_bf16(accumulators[m_tile][n_tile],
                                                   a_fragments[m_tile], b_first);
                        }
#pragma unroll
                        for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
                            wmma_f32_16x16x16_bf16(accumulators[m_tile][n_tile + 1],
                                                   a_fragments[m_tile], b_second);
                        }
                    }
                    if constexpr (N_TILES % 2 != 0) {
                        constexpr int n_tile = N_TILES - 1;
                        bf16_fragment b_fragment{};
                        if constexpr (VECTOR_LOCAL_LOAD) {
                            shared_b.load_fragment_vector(
                                b_fragment, n_tile * BACKWARD_N_PER_TILE + c_row(lane),
                                k_tile);
                        } else {
                            __hip_bfloat16 *b = fragment_data(b_fragment);
#pragma unroll
                            for (int k = 0; k < 16; ++k) {
                                b[k] =
                                    shared_b[(n_tile * BACKWARD_N_PER_TILE + c_row(lane)) *
                                                 K_ITERATION +
                                             k_tile + k];
                            }
                        }
#pragma unroll
                        for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
                            wmma_f32_16x16x16_bf16(accumulators[m_tile][n_tile],
                                                   a_fragments[m_tile], b_fragment);
                        }
                    }
                } else {
#pragma unroll
                    for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
                        bf16_fragment b_fragment{};
                        if constexpr (VECTOR_LOCAL_LOAD) {
                            shared_b.load_fragment_vector(
                                b_fragment, n_tile * BACKWARD_N_PER_TILE + c_row(lane),
                                k_tile);
                        } else {
                            __hip_bfloat16 *b = fragment_data(b_fragment);
#pragma unroll
                            for (int k = 0; k < 16; ++k) {
                                b[k] =
                                    shared_b[(n_tile * BACKWARD_N_PER_TILE + c_row(lane)) *
                                                 K_ITERATION +
                                             k_tile + k];
                            }
                        }
#pragma unroll
                        for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
                            wmma_f32_16x16x16_bf16(accumulators[m_tile][n_tile],
                                                   a_fragments[m_tile], b_fragment);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (ACTIVE_WAVES == BACKWARD_WAVES || wave < ACTIVE_WAVES) {
#pragma unroll
        for (int m_tile = 0; m_tile < M_TILES_PER_WAVE; ++m_tile) {
#pragma unroll
            for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
#pragma unroll
                for (int element = 0; element < 8; ++element) {
                    // gfx11's physical C fragment is J-major for this A/B layout:
                    // the I-major lane coordinates are transposed when written to
                    // row-major C.
                    const int output_row = wave_row_start +
                                           m_tile * BACKWARD_M_PER_TILE +
                                           c_column(lane, element);
                    const int output_column =
                        input_column_start + n_tile * BACKWARD_N_PER_TILE + c_row(lane);
                    if constexpr (FULL_TILES) {
                        grad_input[static_cast<int64_t>(output_row) * kernel_in_features +
                                   output_column] =
                            __float2bfloat16(
                                accumulators[m_tile][n_tile].values[element]);
                    } else if (output_row < rows &&
                               output_column < kernel_in_features) {
                        grad_input[static_cast<int64_t>(output_row) * kernel_in_features +
                                   output_column] =
                            __float2bfloat16(
                                accumulators[m_tile][n_tile].values[element]);
                    }
                }
            }
        }
    }
}

} // namespace torch_ggml_ops::ck
