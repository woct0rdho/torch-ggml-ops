#pragma once

#include <hip/hip_bf16.h>

namespace torch_ggml_ops::ck {

using bf16x16_t = __attribute__((ext_vector_type(16))) __bf16;
using floatx8_t = __attribute__((ext_vector_type(8))) float;

struct bf16_fragment {
    __hip_bfloat162 values[8];
};

struct f32_accumulator {
    float values[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
};

static __device__ __forceinline__ __hip_bfloat16 * fragment_data(bf16_fragment & fragment) {
    return reinterpret_cast<__hip_bfloat16 *>(fragment.values);
}

static __device__ __forceinline__ void wmma_f32_16x16x16_bf16(
        f32_accumulator & accumulator,
        const bf16_fragment & a,
        const bf16_fragment & b) {
    auto & acc = reinterpret_cast<floatx8_t &>(accumulator.values[0]);
    const auto & a_vec = reinterpret_cast<const bf16x16_t &>(a.values[0]);
    const auto & b_vec = reinterpret_cast<const bf16x16_t &>(b.values[0]);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_vec, b_vec, acc);
}

// gfx11 WMMA returns eight FP32 C values per lane. For a logical row-major
// 16x16 C tile, lane L owns row L%16 and columns 2*i + L/16, i in [0, 8).
static __device__ __forceinline__ int c_row(int lane) {
    return lane & 15;
}

static __device__ __forceinline__ int c_column(int lane, int element) {
    return 2 * element + (lane >> 4);
}

} // namespace torch_ggml_ops::ck
