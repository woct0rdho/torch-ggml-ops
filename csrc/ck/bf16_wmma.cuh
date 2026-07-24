#pragma once

#if defined(__HIP__)
#include <hip/hip_bf16.h>
#else
#include "port_cuda.cuh"
#include "vendor/llama_cpp/mma.cuh"
#include <cstdint>
#endif

namespace torch_ggml_ops::ck {

#if !defined(__HIP__)
// llama.cpp's tested tensor-core tile abstraction (mma.sync.m16n8k16), pulled in
// so the CUDA side of this seam can spell tile<>/mma() unqualified.
using namespace ggml_cuda_mma;
#endif

// ----------------------------------------------------------------------------
// 16x16x16 bf16 matrix-core seam.
//
// Every ck/ backward kernel reaches the matrix cores only through this header,
// so the two vendors' per-lane fragment layouts stay confined here and the
// kernel bodies remain single-source. The operation is
//
//     C[m][n] += sum_{k < 16} A[m][k] * B[n][k]
//
// with A and B both supplied row-major as [row][k] -- i.e. the N-side operand is
// transposed, which is what grad_input = grad_output @ W wants for a W stored
// [out_features][in_features].
//
//   bf16_fragment_a          one 16(M) x 16(K) operand tile
//   bf16_fragment_b          one 16(N) x 16(K) operand tile
//   f32_accumulator          one 16(M) x 16(N) fp32 tile
//   load_a_fragment<>()      fill A from a row-major matrix (global memory)
//   load_b_fragment()        fill B from a row-major shared-memory tile
//   wmma_f32_16x16x16_bf16() the multiply-accumulate itself
//   acc_m() / acc_n()        where lane `lane`'s accumulator element lands in C
//
// A lane holds ACCUMULATOR_ELEMENTS values of C. Kernels must not index the
// fragments or the accumulator by hand: the layouts have nothing in common.
//
// AMD (gfx11, wave32): one __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32. A lane
// owns row (lane & 15) of BOTH operands and holds all 16 K values of it; its C
// values are C[2*element + lane/16][lane & 15].
//
// NVIDIA (Ampere and later): two mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32,
// one per 8-wide half of N. A lane holds a quarter of each operand row-block.
// Note that the backward kernels launch a FLAT (BACKWARD_THREADS, 1) block, so
// threadIdx.x is NOT the warp lane -- everything here takes `lane` explicitly
// and never calls tile<>::get_i/get_j (which read threadIdx.x directly).
// ----------------------------------------------------------------------------

static constexpr int ACCUMULATOR_ELEMENTS = 8;

#if defined(__HIP__)

struct bf16_fragment {
    __hip_bfloat162 values[8];
};

using bf16_fragment_a = bf16_fragment;
using bf16_fragment_b = bf16_fragment;

struct f32_accumulator {
    float values[ACCUMULATOR_ELEMENTS] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    __device__ __forceinline__ float & value(int element) {
        return values[element];
    }
    __device__ __forceinline__ float value(int element) const {
        return values[element];
    }
};

static __device__ __forceinline__ __hip_bfloat16 * fragment_data(bf16_fragment & fragment) {
    return reinterpret_cast<__hip_bfloat16 *>(fragment.values);
}

// The operand row this lane owns, and the C position of its element-th value.
static __device__ __forceinline__ int fragment_row(int lane) {
    return lane & 15;
}

static __device__ __forceinline__ int acc_m(int lane, int element) {
    return 2 * element + (lane >> 4);
}

static __device__ __forceinline__ int acc_n(int lane, int element) {
    (void) element;
    return lane & 15;
}

using bf16x16_t = __attribute__((ext_vector_type(16))) __bf16;
using floatx8_t = __attribute__((ext_vector_type(8))) float;

static __device__ __forceinline__ void wmma_f32_16x16x16_bf16(
        f32_accumulator & accumulator,
        const bf16_fragment_a & a,
        const bf16_fragment_b & b) {
    auto & acc = reinterpret_cast<floatx8_t &>(accumulator.values[0]);
    const auto & a_vec = reinterpret_cast<const bf16x16_t &>(a.values[0]);
    const auto & b_vec = reinterpret_cast<const bf16x16_t &>(b.values[0]);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_vec, b_vec, acc);
}

// Fill the A operand from a row-major [*][source_stride] bf16 matrix, taking the
// 16x16 tile at (m_base, k_base). CHECK_M / CHECK_K zero-fill out-of-range rows
// / contraction columns; pass false where the caller knows the tile is full so
// no guard is emitted at all.
template <bool CHECK_M, bool CHECK_K>
static __device__ __forceinline__ void load_a_fragment(
        bf16_fragment_a & fragment,
        const __hip_bfloat16 * __restrict__ source,
        int64_t source_stride,
        int m_base,
        int k_base,
        int m_limit,
        int k_limit,
        int lane) {
    __hip_bfloat16 * a = fragment_data(fragment);
    const int m = m_base + fragment_row(lane);
    const bool row_in_range = !CHECK_M || m < m_limit;
    const int64_t row_offset = static_cast<int64_t>(m) * source_stride;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const bool in_range =
            row_in_range && (!CHECK_K || k_base + k < k_limit);
        a[k] = in_range
            ? source[row_offset + k_base + k]
            : __float2bfloat16(0.0f);
    }
}

// Fill the B operand from a row-major [*][source_stride] shared-memory tile,
// taking the 16(N) x 16(K) tile at (n_base, k_base). Callers zero-fill the
// shared tile for out-of-range columns, so no guard is needed here.
static __device__ __forceinline__ void load_b_fragment(
        bf16_fragment_b & fragment,
        const __hip_bfloat16 * source,
        int source_stride,
        int n_base,
        int k_base,
        int lane) {
    __hip_bfloat16 * b = fragment_data(fragment);
    const int row_offset = (n_base + fragment_row(lane)) * source_stride;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        b[k] = source[row_offset + k_base + k];
    }
}

#else // CUDA

// A: 16(M) x 8(K as bf16 pairs); B: 8(N) x 8(K as bf16 pairs), two per 16-wide N.
using bf16_fragment_a = tile<16, 8, nv_bfloat162>;

struct bf16_fragment_b {
    tile<8, 8, nv_bfloat162> halves[2]; // N in [0,8) and [8,16)
};

struct f32_accumulator {
    tile<16, 8, float> halves[2]; // N in [0,8) and [8,16)

    __device__ __forceinline__ float & value(int element) {
        return halves[element >> 2].x[element & 3];
    }
    __device__ __forceinline__ float value(int element) const {
        return halves[element >> 2].x[element & 3];
    }
};

static_assert(bf16_fragment_a::ne == 4, "unexpected m16n8k16 A fragment size");
static_assert(tile<8, 8, nv_bfloat162>::ne == 2, "unexpected m16n8k16 B fragment size");
static_assert(tile<16, 8, float>::ne == 4, "unexpected m16n8k16 C fragment size");
static_assert(2 * tile<16, 8, float>::ne == ACCUMULATOR_ELEMENTS, "");

// Per-lane m16n8k16 fragment layouts, spelled out rather than taken from
// tile<>::get_i/get_j because those read threadIdx.x, which is not the warp lane
// in the backward kernels' flat block. Validated standalone in
// docs/port-microtests/03_multiwarp_lane_formulas.cu.
//   A tile<16,8,bf162>: i = (l % 2) * 8 + lane / 4 , pair = (l / 2) * 4 + lane % 4
//   B tile<8,8,bf162> : i = lane / 4              , pair =  l      * 4 + lane % 4
//   C tile<16,8,float>: i = (l / 2) * 8 + lane / 4 , j    = (lane % 4) * 2 + l % 2
static __device__ __forceinline__ int acc_m(int lane, int element) {
    const int l = element & 3;
    return (l >> 1) * 8 + (lane >> 2);
}

static __device__ __forceinline__ int acc_n(int lane, int element) {
    const int l = element & 3;
    return (element >> 2) * 8 + (lane & 3) * 2 + (l & 1);
}

static __device__ __forceinline__ void wmma_f32_16x16x16_bf16(
        f32_accumulator & accumulator,
        const bf16_fragment_a & a,
        const bf16_fragment_b & b) {
    mma(accumulator.halves[0], a, b.halves[0]);
    mma(accumulator.halves[1], a, b.halves[1]);
}

template <bool CHECK_M, bool CHECK_K>
static __device__ __forceinline__ void load_a_fragment(
        bf16_fragment_a & fragment,
        const __nv_bfloat16 * __restrict__ source,
        int64_t source_stride,
        int m_base,
        int k_base,
        int m_limit,
        int k_limit,
        int lane) {
#pragma unroll
    for (int l = 0; l < bf16_fragment_a::ne; ++l) {
        const int m = m_base + (l & 1) * 8 + (lane >> 2);
        const int k = k_base + 2 * ((l >> 1) * 4 + (lane & 3));
        __nv_bfloat16 low = __float2bfloat16(0.0f);
        __nv_bfloat16 high = __float2bfloat16(0.0f);
        if (!CHECK_M || m < m_limit) {
            const int64_t row_offset = static_cast<int64_t>(m) * source_stride;
            if (!CHECK_K || k < k_limit) {
                low = source[row_offset + k];
            }
            if (!CHECK_K || k + 1 < k_limit) {
                high = source[row_offset + k + 1];
            }
        }
        fragment.x[l] = __halves2bfloat162(low, high);
    }
}

static __device__ __forceinline__ void load_b_fragment(
        bf16_fragment_b & fragment,
        const __nv_bfloat16 * source,
        int source_stride,
        int n_base,
        int k_base,
        int lane) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int row_offset =
            (n_base + half * 8 + (lane >> 2)) * source_stride;
#pragma unroll
        for (int l = 0; l < tile<8, 8, nv_bfloat162>::ne; ++l) {
            const int k = k_base + 2 * (l * 4 + (lane & 3));
            fragment.halves[half].x[l] = __halves2bfloat162(
                source[row_offset + k], source[row_offset + k + 1]);
        }
    }
}

#endif // __HIP__

} // namespace torch_ggml_ops::ck
