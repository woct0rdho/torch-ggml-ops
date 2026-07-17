#pragma once

#include "vendor/llama_cpp/common.cuh"
#include "vendor/llama_cpp/mma.cuh"

#include <cstdint>

// Narrow dense-MMQ configuration for gfx1151/RDNA3.5. These are the inherited
// llama.cpp settings for J=128 with fallback row bounds enabled.
#define MMQ_ITER_K 256
#define MMQ_TILE_NE_K 32
#define MMQ_TILE_Y_K (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)

static constexpr int MMQ_I = 64;
static constexpr int MMQ_J = 128;
static constexpr int MMQ_NTHREADS = 128;
static constexpr int MMQ_NWARPS = MMQ_NTHREADS / WARP_SIZE;

struct block_q8_1_mmq {
    union {
        float d4[4];
        half2 ds4[4];
        half d2s6[8];
    };
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "unexpected MMQ Q8_1 block size");

enum ggml_cuda_mmq_sram_layout {
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K,
};

static constexpr __host__ __device__ ggml_cuda_mmq_sram_layout mmq_sram_layout(ggml_type type) {
    return type == GGML_TYPE_Q3_K || type == GGML_TYPE_IQ2_S
        ? GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K
        : type == GGML_TYPE_Q6_K
            ? GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K
            : GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1;
}

static constexpr __host__ __device__ int mmq_sram_stride(ggml_type type) {
    switch (mmq_sram_layout(type)) {
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1:
            return 2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / QI8_1 + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K:
            return 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / 2 + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K:
            return 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI6_K + MMQ_TILE_NE_K / 8 + 7;
    }
    return -1;
}

template <ggml_type type, int J, bool fallback>
static constexpr __host__ __device__ int ggml_cuda_mmq_get_nthreads() {
    static_assert(J == MMQ_J && fallback);
    return MMQ_NTHREADS;
}

template <ggml_type type, int J, bool fallback>
static constexpr __host__ __device__ int ggml_cuda_mmq_get_I() {
    static_assert(J == MMQ_J && fallback);
    return MMQ_I;
}

template <ggml_type type, int J, bool fallback>
static constexpr __host__ __device__ int ggml_cuda_mmq_get_sram_stride() {
    static_assert(J == MMQ_J && fallback);
    return mmq_sram_stride(type);
}

template <ggml_type type, int J, bool fallback>
static constexpr __host__ __device__ int ggml_cuda_mmq_get_rows_per_warp() {
    static_assert(J == MMQ_J && fallback);
    return 16;
}

// Compatibility overloads used by the selectively vendored templates.
static constexpr __device__ int ggml_cuda_mmq_get_nthreads(ggml_type, int, bool) { return MMQ_NTHREADS; }
static constexpr __device__ int ggml_cuda_mmq_get_I(ggml_type, int, bool) { return MMQ_I; }
static constexpr __device__ int ggml_cuda_mmq_get_sram_stride(ggml_type type, int, bool) { return mmq_sram_stride(type); }
static constexpr __device__ int ggml_cuda_mmq_get_rows_per_warp(ggml_type, int, bool) { return 16; }

#include "vendor/llama_cpp/mmq-load-targets.cuh"
#include "vendor/llama_cpp/mmq-vec-dot-targets.cuh"

template <ggml_type type>
static __device__ __forceinline__ void mmq_load_target(
        const char * x, int * tile, int block_offset, int i_max, int row_stride) {
    if constexpr (type == GGML_TYPE_Q3_K) {
        ggml_cuda_mmq_load_tiles_q3_K<type, MMQ_J, true>(x, tile, block_offset, i_max, row_stride);
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        ggml_cuda_mmq_load_tiles_q4_K<type, MMQ_J, true>(x, tile, block_offset, i_max, row_stride);
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        ggml_cuda_mmq_load_tiles_q5_K<type, MMQ_J, true>(x, tile, block_offset, i_max, row_stride);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        ggml_cuda_mmq_load_tiles_q6_K<type, MMQ_J, true>(x, tile, block_offset, i_max, row_stride);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        ggml_cuda_mmq_load_tiles_iq2_s<type, MMQ_J, true>(x, tile, block_offset, i_max, row_stride);
    }
}

template <ggml_type type>
static __device__ __forceinline__ void mmq_vec_dot_target(
        const int * x, const int * y, float * sum, int k00) {
    if constexpr (type == GGML_TYPE_Q3_K || type == GGML_TYPE_IQ2_S) {
        ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, MMQ_J, true>(x, y, sum, k00);
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, MMQ_J, true>(x, y, sum, k00);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma<type, MMQ_J, true>(x, y, sum, k00);
    }
}

template <ggml_type type>
static __device__ __forceinline__ void mmq_write_back_bf16(
        const float * sum, __hip_bfloat16 * dst, int stride, int i_max, int j_max) {
    using namespace ggml_cuda_mma;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int ntx = 16 / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * (ntx * tile_C::I);

#pragma unroll
    for (int j0 = 0; j0 < MMQ_J; j0 += ntx * tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if (j <= j_max && i <= i_max) {
                    dst[j * stride + i] = __float2bfloat16(sum[(j0 / tile_C::J + n) * tile_C::ne + l]);
                }
            }
        }
    }
}

enum mmq_q8_layout {
    MMQ_Q8_D4,
    MMQ_Q8_DS4,
};

template <ggml_type type>
static constexpr __host__ __device__ mmq_q8_layout mmq_activation_layout() {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? MMQ_Q8_DS4 : MMQ_Q8_D4;
}

template <ggml_type type>
__launch_bounds__(64, 2)
static __global__ void quantize_bf16_mmq_q8_1(
        const __hip_bfloat16 * __restrict__ x,
        block_q8_1_mmq * __restrict__ y,
        int64_t rows,
        int64_t rows_padded,
        int64_t k) {
    constexpr mmq_q8_layout layout = mmq_activation_layout<type>();
    const int64_t row = blockIdx.x;
    const int64_t i0 = (static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x) * 4;
    if (i0 >= k) {
        return;
    }

    float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (row < rows) {
        const __hip_bfloat16 * src = x + row * k + i0;
        xi.x = __bfloat162float(src[0]);
        xi.y = __bfloat162float(src[1]);
        xi.z = __bfloat162float(src[2]);
        xi.w = __bfloat162float(src[3]);
    }

    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, WARP_SIZE));
    }

    float sum = xi.x + xi.y + xi.z + xi.w;
    if constexpr (layout == MMQ_Q8_DS4) {
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset, WARP_SIZE);
        }
    }

    const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
    const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;
    char4 q;
    q.x = static_cast<int8_t>(roundf(xi.x * d_inv));
    q.y = static_cast<int8_t>(roundf(xi.y * d_inv));
    q.z = static_cast<int8_t>(roundf(xi.z * d_inv));
    q.w = static_cast<int8_t>(roundf(xi.w * d_inv));

    const int64_t block_k = i0 / (4 * QK8_1);
    const int iqs = i0 % (4 * QK8_1);
    block_q8_1_mmq & out = y[block_k * rows_padded + row];
    reinterpret_cast<char4 *>(out.qs)[iqs / 4] = q;

    if (iqs % 32 == 0) {
        if constexpr (layout == MMQ_Q8_DS4) {
            out.ds4[iqs / 32] = make_half2(d, sum);
        } else {
            out.d4[iqs / 32] = d;
        }
    }
}

template <ggml_type type>
__launch_bounds__(MMQ_NTHREADS, 2)
static __global__ void dense_mmq_bf16_kernel(
        const char * __restrict__ weights,
        const int * __restrict__ activations,
        __hip_bfloat16 * __restrict__ dst,
        int nrows_weight,
        int nrows_activation,
        int nrows_activation_padded,
        int blocks_per_weight_row) {
    const int tile_i = blockIdx.x;
    const int tile_j = blockIdx.y;
    const int i_max = nrows_weight - tile_i * MMQ_I - 1;
    const int j_max = nrows_activation - tile_j * MMQ_J - 1;

    extern __shared__ int shared[];
    int * tile_y = shared + MMQ_J;
    int * tile_x = tile_y + GGML_PAD(MMQ_J * MMQ_TILE_Y_K, MMQ_NTHREADS);

    float sum[MMQ_J * MMQ_I / MMQ_NTHREADS] = {0.0f};
    constexpr int q8_block_ints = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb = 0; kb < blocks_per_weight_row; ++kb) {
        const int weight_block_offset = tile_i * MMQ_I * blocks_per_weight_row + kb;
        mmq_load_target<type>(weights, tile_x, weight_block_offset, i_max, blocks_per_weight_row);

#pragma unroll
        for (int l0 = 0; l0 < MMQ_J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const int src = ((2 * kb) * nrows_activation_padded + tile_j * MMQ_J) * q8_block_ints + l;
            tile_y[l] = activations[src];
        }
        __syncthreads();
        mmq_vec_dot_target<type>(tile_x, tile_y, sum, 0);
        __syncthreads();

#pragma unroll
        for (int l0 = 0; l0 < MMQ_J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const int src = ((2 * kb + 1) * nrows_activation_padded + tile_j * MMQ_J) * q8_block_ints + l;
            tile_y[l] = activations[src];
        }
        __syncthreads();
        mmq_vec_dot_target<type>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }

    mmq_write_back_bf16<type>(
        sum,
        dst + tile_j * MMQ_J * nrows_weight + tile_i * MMQ_I,
        nrows_weight,
        i_max,
        j_max);
}

template <ggml_type type>
__launch_bounds__(MMQ_NTHREADS, 2)
static __global__ void grouped_mmq_bf16_kernel(
        const char * __restrict__ weights,
        const int * __restrict__ activations,
        __hip_bfloat16 * __restrict__ dst,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int nrows_weight,
        int nrows_activation,
        int blocks_per_weight_row,
        int64_t bytes_per_expert) {
    const int tile_i = blockIdx.x;
    const int group = blockIdx.y;
    const int row_begin = group == 0 ? 0 : expert_offsets[group - 1];
    const int row_end = expert_offsets[group];
    const int64_t expert = expert_indices[group];
    if (
        expert < 0 || expert >= num_experts || row_begin < 0 ||
        row_end <= row_begin || row_end > nrows_activation
    ) {
        return;
    }

    const int i_max = nrows_weight - tile_i * MMQ_I - 1;
    const char * expert_weights = weights + expert * bytes_per_expert;

    extern __shared__ int shared[];
    int * tile_y = shared + MMQ_J;
    int * tile_x = tile_y + GGML_PAD(MMQ_J * MMQ_TILE_Y_K, MMQ_NTHREADS);

    constexpr int q8_block_ints = sizeof(block_q8_1_mmq) / sizeof(int);
    static_assert(MMQ_TILE_Y_K == q8_block_ints, "unexpected grouped Q8 tile layout");

    for (int row_start = row_begin; row_start < row_end; row_start += MMQ_J) {
        const int j_max = min(MMQ_J, row_end - row_start) - 1;
        float sum[MMQ_J * MMQ_I / MMQ_NTHREADS] = {0.0f};

        for (int kb = 0; kb < blocks_per_weight_row; ++kb) {
            const int weight_block_offset = tile_i * MMQ_I * blocks_per_weight_row + kb;
            mmq_load_target<type>(
                expert_weights,
                tile_x,
                weight_block_offset,
                i_max,
                blocks_per_weight_row);

#pragma unroll
            for (int l0 = 0; l0 < MMQ_J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                const int local_row = l / q8_block_ints;
                const int q8_int = l % q8_block_ints;
                if (local_row <= j_max) {
                    const int src = (
                        (2 * kb) * nrows_activation + row_start + local_row
                    ) * q8_block_ints + q8_int;
                    tile_y[l] = activations[src];
                } else {
                    tile_y[l] = 0;
                }
            }
            __syncthreads();
            mmq_vec_dot_target<type>(tile_x, tile_y, sum, 0);
            __syncthreads();

#pragma unroll
            for (int l0 = 0; l0 < MMQ_J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                const int local_row = l / q8_block_ints;
                const int q8_int = l % q8_block_ints;
                if (local_row <= j_max) {
                    const int src = (
                        (2 * kb + 1) * nrows_activation + row_start + local_row
                    ) * q8_block_ints + q8_int;
                    tile_y[l] = activations[src];
                } else {
                    tile_y[l] = 0;
                }
            }
            __syncthreads();
            mmq_vec_dot_target<type>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();
        }

        mmq_write_back_bf16<type>(
            sum,
            dst + row_start * nrows_weight + tile_i * MMQ_I,
            nrows_weight,
            i_max,
            j_max);
        __syncthreads();
    }
}
