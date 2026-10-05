#pragma once

#include "../vendor/llama_cpp/common.cuh"
#include "../vendor/llama_cpp/iq2_s_grid.cuh"
#include "../vendor/llama_cpp/iq2_xxs_grid.cuh"

#include <cstdint>

namespace torch_ggml_ops::ck {

static __device__ __forceinline__ float fp16_to_fp32(half value) {
    return __half2float(value);
}

// Look up eight 4-bit indices in a 16-byte table at once. The low three bits
// of each index select a byte of one half of the table with a byte permute and
// the fourth bit selects between the halves with a second permute, so no memory
// gather is needed.
static __device__ __forceinline__ int2 iq4_table_lookup_16(
        const int q4, const int8_t * table) {
    const uint32_t * values = (const uint32_t *) table;

    const uint32_t q_even = q4;
    const uint32_t q_odd = q4 >> 4;

    const uint32_t even_low = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    const uint32_t odd_low = __builtin_amdgcn_perm(values[1], values[0], q_odd & 0x07070707);
    const uint32_t even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    const uint32_t odd_high = __builtin_amdgcn_perm(values[3], values[2], q_odd & 0x07070707);

    const uint32_t even_mask = 0x03020100 | ((q_even & 0x08080808) >> 1);
    const uint32_t odd_mask = 0x03020100 | ((q_odd & 0x08080808) >> 1);

    return make_int2(
        __builtin_amdgcn_perm(even_high, even_low, even_mask),
        __builtin_amdgcn_perm(odd_high, odd_low, odd_mask));
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
    if constexpr (type == GGML_TYPE_Q8_0) {
        return sizeof(block_q8_0);
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        return sizeof(block_q2_K);
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        return sizeof(block_iq2_xxs);
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        return sizeof(block_q3_K);
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        return sizeof(block_q4_K);
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        return sizeof(block_q5_K);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        return sizeof(block_q6_K);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        return sizeof(block_iq2_s);
    } else if constexpr (type == GGML_TYPE_Q2_0) {
        return sizeof(block_q2_0);
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        return sizeof(block_q4_0);
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        return sizeof(block_q5_0);
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        return sizeof(block_iq4_nl);
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        return sizeof(block_iq4_xs);
    } else {
        return 0;
    }
}

template <ggml_type type>
static __device__ __forceinline__ float decode_gguf_value(
        const char * packed_row,
        int block_index,
        int value_index);

template <>
__device__ __forceinline__ float decode_gguf_value<GGML_TYPE_Q4_0>(
        const char * packed_row,
        int block_index,
        int value_index) {
    const auto & block = reinterpret_cast<const block_q4_0 *>(packed_row)[block_index];
    const int quant = (
        block.qs[value_index & (QK4_0 / 2 - 1)] >>
        (4 * ((value_index >> 4) & 1))) & 0x0f;
    return fp16_to_fp32(block.d) * static_cast<float>(quant - 8);
}

// Six-bit signed sub-block scale of an IQ4_XS block: a nibble from the low
// plane and two bits from the high plane, biased by 32.
static __device__ __forceinline__ int iq4_xs_scale(
        const block_iq4_xs & block, int sub_block) {
    const int low =
        (block.scales_l[sub_block >> 1] >> (4 * (sub_block & 1))) & 0x0f;
    const int high = (block.scales_h >> (2 * sub_block)) & 0x03;
    return (low | (high << 4)) - 32;
}

template <>
__device__ __forceinline__ float decode_gguf_value<GGML_TYPE_IQ4_XS>(
        const char * packed_row,
        int block_index,
        int value_index) {
    const auto & block =
        reinterpret_cast<const block_iq4_xs *>(packed_row)[block_index];
    // Within a sub-block the sixteen payload bytes carry both planes: byte `j`
    // holds value `j` in its low nibble and value `j + 16` in its high nibble.
    const int sub_block = value_index >> 5;
    const int plane = (value_index >> 4) & 1;
    const uint8_t byte = block.qs[16 * sub_block + (value_index & 15)];
    const int nibble = (byte >> (4 * plane)) & 0x0f;
    return fp16_to_fp32(block.d) * static_cast<float>(iq4_xs_scale(block, sub_block)) *
        static_cast<float>(kvalues_iq4nl[nibble]);
}

template <>
__device__ __forceinline__ float decode_gguf_value<GGML_TYPE_IQ4_NL>(
        const char * packed_row,
        int block_index,
        int value_index) {
    const auto & block =
        reinterpret_cast<const block_iq4_nl *>(packed_row)[block_index];
    const int nibble = (
        block.qs[value_index & (QK4_NL / 2 - 1)] >>
        (4 * ((value_index >> 4) & 1))) & 0x0f;
    return fp16_to_fp32(block.d) *
        static_cast<float>(kvalues_iq4nl[nibble]);
}

template <>
__device__ __forceinline__ float decode_gguf_value<GGML_TYPE_Q5_0>(
        const char * packed_row,
        int block_index,
        int value_index) {
    const auto & block = reinterpret_cast<const block_q5_0 *>(packed_row)[block_index];
    const uint32_t fifth = get_int_b4(block.qh, 0);
    const int low = (
        block.qs[value_index & (QK5_0 / 2 - 1)] >>
        (4 * ((value_index >> 4) & 1))) & 0x0f;
    const int high = (fifth >> value_index) & 0x01;
    return fp16_to_fp32(block.d) * static_cast<float>((low | (high << 4)) - 16);
}

template <>
__device__ __forceinline__ float decode_gguf_value<GGML_TYPE_Q2_0>(
        const char * packed_row,
        int block_index,
        int value_index) {
    const auto & block = reinterpret_cast<const block_q2_0 *>(packed_row)[block_index];
    const int quant = (block.qs[value_index >> 2] >> (2 * (value_index & 3))) & 0x03;
    return fp16_to_fp32(block.d) * static_cast<float>(quant - 1);
}

template <>
__device__ __forceinline__ float decode_gguf_value<GGML_TYPE_Q8_0>(
        const char * packed_row,
        int block_index,
        int value_index) {
    const auto & block = reinterpret_cast<const block_q8_0 *>(packed_row)[block_index];
    return fp16_to_fp32(block.d) * static_cast<float>(block.qs[value_index]);
}

template <>
__device__ __forceinline__ float decode_gguf_value<GGML_TYPE_Q2_K>(
        const char * packed_row,
        int block_index,
        int value_index) {
    const auto & block = reinterpret_cast<const block_q2_K *>(packed_row)[block_index];
    const int group = value_index >> 4;
    const int local = value_index & 15;
    const int group_in_half = group & 7;
    const int half = group >> 3;
    const int shift = (group_in_half >> 1) * 2;
    const int byte = half * 32 + (group_in_half & 1) * 16 + local;
    const uint8_t packed = block.qs[byte];
    const int quant = (packed >> shift) & 0x03;
    const uint8_t scale_min = block.scales[group];
    const float scale = fp16_to_fp32(block.d) * static_cast<float>(scale_min & 0x0f);
    const float minimum = fp16_to_fp32(block.dmin) * static_cast<float>(scale_min >> 4);
    return scale * static_cast<float>(quant) - minimum;
}

template <>
__device__ __forceinline__ float decode_gguf_value<GGML_TYPE_IQ2_XXS>(
        const char * packed_row,
        int block_index,
        int value_index) {
    const auto & block = reinterpret_cast<const block_iq2_xxs *>(packed_row)[block_index];
    const int group = value_index >> 5;
    const int local = value_index & 31;
    const int sub_group = local >> 3;
    const int element = local & 7;
    const int q_word = 2 * group;
    const int q2 = get_int_b2(block.qs, q_word);
    const uint8_t * grid_indices = reinterpret_cast<const uint8_t *>(&q2);
    const uint32_t sign_data = static_cast<uint32_t>(
        get_int_b2(block.qs, q_word + 1));
    const uint8_t signs = static_cast<uint8_t>(unpack_ksigns(
        sign_data >> (7 * sub_group)));
    const uint64_t grid = iq2xxs_grid[grid_indices[sub_group]];
    const int magnitude = static_cast<int>((grid >> (8 * element)) & 0xff);
    const int sign = (signs & (1 << element)) == 0 ? 1 : -1;
    const float db = fp16_to_fp32(block.d) *
        (0.5f + static_cast<float>(sign_data >> 28)) * 0.25f;
    return db * static_cast<float>(magnitude * sign);
}

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

static __device__ __forceinline__ void decode_iq2_s_group_8(
        const char * packed_row,
        int block_index,
        int value_index,
        __hip_bfloat16 * values) {
    const auto & block =
        reinterpret_cast<const block_iq2_s *>(packed_row)[block_index];
    const int grid_group = value_index >> 3;
    const int high =
        (block.qh[grid_group >> 2] >> (2 * (grid_group & 3))) & 0x03;
    const int grid_index = block.qs[grid_group] | (high << 8);
    const uint64_t grid = iq2s_grid[grid_index];
    const uint8_t signs = block.qs[32 + grid_group];
    const int scale_group = value_index >> 4;
    const int scale =
        (block.scales[scale_group >> 1] >> (4 * (scale_group & 1))) & 0x0f;
    const float db = fp16_to_fp32(block.d) *
        (0.5f + static_cast<float>(scale)) * 0.25f;
#pragma unroll
    for (int index = 0; index < 8; ++index) {
        const int magnitude =
            static_cast<int>((grid >> (8 * index)) & 0xff);
        const int sign = ((signs >> index) & 0x01) == 0 ? 1 : -1;
        values[index] = __float2bfloat16(
            db * static_cast<float>(magnitude * sign));
    }
}

static __device__ __forceinline__ void decode_iq2_s_group_16(
        const char * packed_row,
        int block_index,
        int value_index,
        __hip_bfloat16 * values) {
    const auto & block =
        reinterpret_cast<const block_iq2_s *>(packed_row)[block_index];
    const int first_grid_group = value_index >> 3;
    const int second_grid_group = first_grid_group + 1;
    const int first_high =
        (block.qh[first_grid_group >> 2] >>
         (2 * (first_grid_group & 3))) & 0x03;
    const int second_high =
        (block.qh[second_grid_group >> 2] >>
         (2 * (second_grid_group & 3))) & 0x03;
    const uint64_t first_grid = iq2s_grid[
        block.qs[first_grid_group] | (first_high << 8)];
    const uint64_t second_grid = iq2s_grid[
        block.qs[second_grid_group] | (second_high << 8)];
    const uint8_t first_signs = block.qs[32 + first_grid_group];
    const uint8_t second_signs = block.qs[32 + second_grid_group];
    const int scale_group = value_index >> 4;
    const int scale =
        (block.scales[scale_group >> 1] >> (4 * (scale_group & 1))) & 0x0f;
    const float db = fp16_to_fp32(block.d) *
        (0.5f + static_cast<float>(scale)) * 0.25f;
#pragma unroll
    for (int index = 0; index < 8; ++index) {
        const int first_magnitude =
            static_cast<int>((first_grid >> (8 * index)) & 0xff);
        const int second_magnitude =
            static_cast<int>((second_grid >> (8 * index)) & 0xff);
        const int first_sign =
            ((first_signs >> index) & 0x01) == 0 ? 1 : -1;
        const int second_sign =
            ((second_signs >> index) & 0x01) == 0 ? 1 : -1;
        values[index] = __float2bfloat16(
            db * static_cast<float>(first_magnitude * first_sign));
        values[8 + index] = __float2bfloat16(
            db * static_cast<float>(second_magnitude * second_sign));
    }
}

} // namespace torch_ggml_ops::ck
