#pragma once

#include "../vendor/llama_cpp/common.cuh"

#include <cstdint>

namespace torch_ggml_ops::ck {

static __device__ __forceinline__ float fp16_to_fp32(half value) {
    return __half2float(value);
}

static __device__ __forceinline__ int k_scale(
        const uint8_t * scales,
        int group) {
    if (group < 4) {
        return scales[group] & 0x3f;
    }
    const int index = group - 4;
    return (scales[8 + index] & 0x0f) | ((scales[index] >> 2) & 0x30);
}

static __device__ __forceinline__ int k_min(
        const uint8_t * scales,
        int group) {
    if (group < 4) {
        return scales[4 + group] & 0x3f;
    }
    const int index = group - 4;
    return (scales[8 + index] >> 4) | ((scales[4 + index] >> 2) & 0x30);
}

template <ggml_type type>
static constexpr __host__ __device__ int gguf_block_bytes() {
    if constexpr (type == GGML_TYPE_Q3_K) {
        return sizeof(block_q3_K);
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        return sizeof(block_q4_K);
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        return sizeof(block_q5_K);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        return sizeof(block_q6_K);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        return sizeof(block_iq2_s);
    }
}

template <ggml_type type>
static __device__ __forceinline__ float decode_gguf_value(
        const char * packed_row,
        int block_index,
        int value_index);

template <>
__device__ __forceinline__ float decode_gguf_value<GGML_TYPE_Q3_K>(
        const char * packed_row,
        int block_index,
        int value_index) {
    const auto & block = reinterpret_cast<const block_q3_K *>(packed_row)[block_index];
    const int scale_group = value_index >> 4;
    const int low_scale = scale_group < 8
        ? block.scales[scale_group]
        : block.scales[scale_group - 8] >> 4;
    const int high_scale = block.scales[8 + (scale_group & 3)] >> (2 * (scale_group >> 2));
    const int scale = ((low_scale & 0x0f) | ((high_scale & 0x03) << 4)) - 32;

    const int low_chunk = value_index >> 7;
    const int low_remainder = value_index & 127;
    const int low_shift = 2 * (low_remainder >> 5);
    const int low_byte = low_chunk * 32 + (low_remainder & 31);
    const int low = (block.qs[low_byte] >> low_shift) & 0x03;

    const int high_shift = value_index >> 5;
    const int high_byte = value_index & 31;
    const int high = ((block.hmask[high_byte] >> high_shift) & 0x01) ^ 0x01;
    const int quant = low - (high << 2);

    const float scaled_d = fp16_to_fp32(block.d) * static_cast<float>(scale);
    return scaled_d * static_cast<float>(quant);
}

template <>
__device__ __forceinline__ float decode_gguf_value<GGML_TYPE_Q4_K>(
        const char * packed_row,
        int block_index,
        int value_index) {
    const auto & block = reinterpret_cast<const block_q4_K *>(packed_row)[block_index];
    const int group = value_index >> 5;
    const int byte = (group >> 1) * 32 + (value_index & 31);
    const int quant = (block.qs[byte] >> (4 * (group & 1))) & 0x0f;
    const float d = fp16_to_fp32(block.d) * static_cast<float>(k_scale(block.scales, group));
    const float minimum = fp16_to_fp32(block.dmin) * static_cast<float>(k_min(block.scales, group));
    return d * static_cast<float>(quant) - minimum;
}

template <>
__device__ __forceinline__ float decode_gguf_value<GGML_TYPE_Q5_K>(
        const char * packed_row,
        int block_index,
        int value_index) {
    const auto & block = reinterpret_cast<const block_q5_K *>(packed_row)[block_index];
    const int group = value_index >> 5;
    const int byte = (group >> 1) * 32 + (value_index & 31);
    const int low = (block.qs[byte] >> (4 * (group & 1))) & 0x0f;
    const int high = (block.qh[value_index & 31] >> group) & 0x01;
    const int quant = low | (high << 4);
    const float d = fp16_to_fp32(block.d) * static_cast<float>(k_scale(block.scales, group));
    const float minimum = fp16_to_fp32(block.dmin) * static_cast<float>(k_min(block.scales, group));
    return d * static_cast<float>(quant) - minimum;
}

template <>
__device__ __forceinline__ float decode_gguf_value<GGML_TYPE_Q6_K>(
        const char * packed_row,
        int block_index,
        int value_index) {
    const auto & block = reinterpret_cast<const block_q6_K *>(packed_row)[block_index];
    const int chunk = value_index >> 7;
    const int remainder = value_index & 127;
    const int low_byte = chunk * 64 + (remainder & 63);
    const int low = (block.ql[low_byte] >> (4 * (remainder >> 6))) & 0x0f;
    const int high_byte = chunk * 32 + (value_index & 31);
    const int high = (block.qh[high_byte] >> (2 * ((remainder >> 5) & 3))) & 0x03;
    const int quant = (low | (high << 4)) - 32;
    const int scale = block.scales[value_index >> 4];
    const float d = fp16_to_fp32(block.d) * static_cast<float>(scale);
    return d * static_cast<float>(quant);
}

template <>
__device__ __forceinline__ float decode_gguf_value<GGML_TYPE_IQ2_S>(
        const char * packed_row,
        int block_index,
        int value_index) {
    const auto & block = reinterpret_cast<const block_iq2_s *>(packed_row)[block_index];
    const int grid_group = value_index >> 3;
    const int grid_element = value_index & 7;
    const int high = (block.qh[grid_group >> 2] >> (2 * (grid_group & 3))) & 0x03;
    const int grid_index = block.qs[grid_group] | (high << 8);
    const uint64_t grid = iq2s_grid[grid_index];
    const int magnitude = static_cast<int>((grid >> (8 * grid_element)) & 0xff);
    const int sign = ((block.qs[32 + grid_group] >> grid_element) & 0x01) == 0 ? 1 : -1;

    const int scale_group = value_index >> 4;
    const int scale = (block.scales[scale_group >> 1] >> (4 * (scale_group & 1))) & 0x0f;
    const float d = fp16_to_fp32(block.d) * (0.5f + static_cast<float>(scale));
    const float db = d * 0.25f;
    return db * static_cast<float>(magnitude * sign);
}

} // namespace torch_ggml_ops::ck
