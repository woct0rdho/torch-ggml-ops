#pragma once

#include "vendor/llama_cpp/common.cuh"
#include "vendor/llama_cpp/mma.cuh"

#include <cstdint>

// Narrow dense-MMQ configuration for gfx1151/RDNA3.5. These are the inherited
// llama.cpp settings for J=128 with fallback row bounds enabled.
static constexpr int MMQ_ITER_K = 256;
static constexpr int MMQ_TILE_NE_K = 32;
static constexpr int MMQ_TILE_Y_K = MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1;

static constexpr int MMQ_I = 64;
static constexpr int MMQ_J = 128;
static constexpr int MMQ_J_MEDIUM = 80;
static constexpr int MMQ_J_SMALL = 64;
static constexpr int MMQ_J_TINY = 32;
static constexpr int MMQ_J_MIN = 16;
static constexpr int MMQ_NTHREADS = 128;
static constexpr int MMQ_NWARPS = MMQ_NTHREADS / WARP_SIZE;

struct block_q8_1_mmq {
    union {
        float scales_f32[4];
        half2 scale_sum_pairs_f16[4];
        struct {
            half scales[2];
            half sums[6];
        } scales2_sums6_f16;
    };
    int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "unexpected MMQ Q8_1 block size");

enum ggml_cuda_mmq_sram_layout {
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K,
};

static constexpr __host__ __device__ ggml_cuda_mmq_sram_layout mmq_sram_layout(ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ2_XXS
        ? GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0
        : type == GGML_TYPE_Q2_K
            ? GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K
            : type == GGML_TYPE_Q3_K || type == GGML_TYPE_IQ2_S
                ? GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K
                : type == GGML_TYPE_Q6_K
                    ? GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K
                    : GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1;
}

static constexpr __host__ __device__ int mmq_sram_stride(ggml_type type) {
    switch (mmq_sram_layout(type)) {
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0:
            return 2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / QI8_0 + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1:
            return 2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / QI8_1 + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K:
            return 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K:
            return 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / 2 + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K:
            return 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI6_K + MMQ_TILE_NE_K / 8 + 7;
    }
    return -1;
}

template <ggml_type type, int J, bool fallback>
static constexpr __host__ __device__ int ggml_cuda_mmq_get_nthreads() {
    static_assert(
        J == MMQ_J || J == MMQ_J_MEDIUM || J == MMQ_J_SMALL ||
        J == MMQ_J_TINY || J == MMQ_J_MIN);
    return MMQ_NTHREADS;
}

template <ggml_type type, int J, bool fallback>
static constexpr __host__ __device__ int ggml_cuda_mmq_get_I() {
    static_assert(
        J == MMQ_J || J == MMQ_J_MEDIUM || J == MMQ_J_SMALL ||
        J == MMQ_J_TINY || J == MMQ_J_MIN);
    return MMQ_I;
}

template <ggml_type type, int J, bool fallback>
static constexpr __host__ __device__ int ggml_cuda_mmq_get_sram_stride() {
    static_assert(
        J == MMQ_J || J == MMQ_J_MEDIUM || J == MMQ_J_SMALL ||
        J == MMQ_J_TINY || J == MMQ_J_MIN);
    return mmq_sram_stride(type);
}

template <ggml_type type, int J, bool fallback>
static constexpr __host__ __device__ int ggml_cuda_mmq_get_rows_per_warp() {
    static_assert(
        J == MMQ_J || J == MMQ_J_MEDIUM || J == MMQ_J_SMALL ||
        J == MMQ_J_TINY || J == MMQ_J_MIN);
    return 16;
}

// Compatibility overloads used by the selectively vendored templates.
static constexpr __device__ int ggml_cuda_mmq_get_nthreads(ggml_type, int, bool) { return MMQ_NTHREADS; }
static constexpr __device__ int ggml_cuda_mmq_get_I(ggml_type, int, bool) { return MMQ_I; }
static constexpr __device__ int ggml_cuda_mmq_get_sram_stride(ggml_type type, int, bool) { return mmq_sram_stride(type); }
static constexpr __device__ int ggml_cuda_mmq_get_rows_per_warp(ggml_type, int, bool) { return 16; }

enum mmq_q8_1_metadata_layout {
    MMQ_Q8_1_METADATA_F32_D4,
    MMQ_Q8_1_METADATA_F16_D4S4,
    MMQ_Q8_1_METADATA_F16_D2S6,
};

#include "vendor/llama_cpp/mmq-load-targets.cuh"
#include "vendor/llama_cpp/mmq-vec-dot-targets.cuh"
#include "vendor/llama_cpp/mmq-vec-dot-q2-k-rolled.cuh"

template <ggml_type type, int J, bool fallback = true>
static __device__ __forceinline__ void mmq_load_target(
        const char * x, int * tile, int block_offset, int i_max, int row_stride) {
    if constexpr (type == GGML_TYPE_Q8_0) {
        constexpr int blocks_per_iteration = MMQ_ITER_K / QK8_0;
        ggml_cuda_mmq_load_tiles_q8_0<type, J, fallback>(
            x,
            tile,
            block_offset * blocks_per_iteration,
            i_max,
            row_stride * blocks_per_iteration);
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        ggml_cuda_mmq_load_tiles_q2_K<type, J, fallback>(x, tile, block_offset, i_max, row_stride);
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        ggml_cuda_mmq_load_tiles_q3_K<type, J, fallback>(x, tile, block_offset, i_max, row_stride);
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        ggml_cuda_mmq_load_tiles_q4_K<type, J, fallback>(x, tile, block_offset, i_max, row_stride);
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        ggml_cuda_mmq_load_tiles_q5_K<type, J, fallback>(x, tile, block_offset, i_max, row_stride);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        ggml_cuda_mmq_load_tiles_q6_K<type, J, fallback>(x, tile, block_offset, i_max, row_stride);
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        ggml_cuda_mmq_load_tiles_iq2_xxs<type, J, fallback>(x, tile, block_offset, i_max, row_stride);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        ggml_cuda_mmq_load_tiles_iq2_s<type, J, fallback>(x, tile, block_offset, i_max, row_stride);
    }
}

template <ggml_type type, int J, bool fallback = true, bool rolled_q2_k = false>
static __device__ __forceinline__ void mmq_vec_dot_target(
        const int * x, const int * y, float * sum, int k00) {
    if constexpr (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ2_XXS) {
        ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<
            type, J, fallback, MMQ_Q8_1_METADATA_F32_D4>(x, y, sum, k00);
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        if constexpr (rolled_q2_k) {
            ggml_cuda_mmq_vec_dot_q2_K_q8_1_mma_rolled<type, J, fallback>(
                x, y, sum, k00);
        } else {
            ggml_cuda_mmq_vec_dot_q2_K_q8_1_mma<type, J, fallback>(
                x, y, sum, k00);
        }
    } else if constexpr (type == GGML_TYPE_Q3_K || type == GGML_TYPE_IQ2_S) {
        ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback>(x, y, sum, k00);
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma<type, J, fallback>(x, y, sum, k00);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma<type, J, fallback>(x, y, sum, k00);
    }
}

template <ggml_type type, int J, bool full_i = false, bool full_j = false>
static __device__ __forceinline__ void mmq_write_back_bf16(
        const float * sum, __hip_bfloat16 * dst, int stride, int i_max, int j_max) {
    using namespace ggml_cuda_mma;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int ntx = 16 / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * (ntx * tile_C::I);

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
                const int i = i0 + n * tile_C::I + tile_C::get_i(l);
                if ((full_j || j <= j_max) && (full_i || i <= i_max)) {
                    dst[j * stride + i] = __float2bfloat16(sum[(j0 / tile_C::J + n) * tile_C::ne + l]);
                }
            }
        }
    }
}

template <ggml_type type>
static constexpr __host__ __device__ mmq_q8_1_metadata_layout
mmq_activation_metadata_layout() {
    return type == GGML_TYPE_Q2_K
        ? MMQ_Q8_1_METADATA_F16_D2S6
        : type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K
            ? MMQ_Q8_1_METADATA_F16_D4S4
            : MMQ_Q8_1_METADATA_F32_D4;
}

template <ggml_type type>
static __device__ __forceinline__ void quantize_bf16_mmq_q8_1_body(
        const __hip_bfloat16 * __restrict__ x,
        block_q8_1_mmq * __restrict__ y,
        int64_t rows,
        int64_t rows_padded,
        int64_t k) {
    constexpr mmq_q8_1_metadata_layout metadata_layout =
        mmq_activation_metadata_layout<type>();
    constexpr int values_per_scale =
        metadata_layout == MMQ_Q8_1_METADATA_F16_D2S6 ? 64 : 32;
    constexpr int values_per_sum =
        metadata_layout == MMQ_Q8_1_METADATA_F16_D2S6 ? 16 : 32;
    const int64_t row = blockIdx.x;

    for (int64_t i0 = static_cast<int64_t>(threadIdx.x) * 4;
         i0 < k;
         i0 += static_cast<int64_t>(blockDim.x) * 4) {
        const __hip_bfloat16 * src = x + row * k + i0;
        const float4 xi = make_float4(
            __bfloat162float(src[0]),
            __bfloat162float(src[1]),
            __bfloat162float(src[2]),
            __bfloat162float(src[3]));

        float amax = fmaxf(
            fmaxf(fabsf(xi.x), fabsf(xi.y)),
            fmaxf(fabsf(xi.z), fabsf(xi.w)));
#pragma unroll
        for (int offset = values_per_scale / 8; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, WARP_SIZE));
        }

        float sum = xi.x + xi.y + xi.z + xi.w;
        if constexpr (metadata_layout != MMQ_Q8_1_METADATA_F32_D4) {
#pragma unroll
            for (int offset = values_per_sum / 8; offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset, WARP_SIZE);
            }
        }

        const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
        const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;
        const char4 q = make_char4(
            static_cast<int8_t>(roundf(xi.x * d_inv)),
            static_cast<int8_t>(roundf(xi.y * d_inv)),
            static_cast<int8_t>(roundf(xi.z * d_inv)),
            static_cast<int8_t>(roundf(xi.w * d_inv)));

        const int64_t block_k = i0 / (4 * QK8_1);
        const int iqs = i0 % (4 * QK8_1);
        block_q8_1_mmq & out = y[block_k * rows_padded + row];
        reinterpret_cast<char4 *>(out.qs)[iqs / 4] = q;

        if constexpr (metadata_layout == MMQ_Q8_1_METADATA_F16_D2S6) {
            if (iqs % 16 == 0 && iqs < 96) {
                out.scales2_sums6_f16.sums[iqs / 16] = sum;
                if (iqs % 64 == 0) {
                    out.scales2_sums6_f16.scales[iqs / 64] = d;
                }
            }
        } else if (iqs % 32 == 0) {
            if constexpr (metadata_layout == MMQ_Q8_1_METADATA_F16_D4S4) {
                out.scale_sum_pairs_f16[iqs / 32] = make_half2(d, sum);
            } else {
                out.scales_f32[iqs / 32] = d;
            }
        }
    }
}

template <
    ggml_type type,
    int J,
    int fixed_blocks_per_weight_row = 0,
    bool full_i = false,
    bool full_j = false>
static __device__ __forceinline__ void dense_mmq_bf16_body(
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
    const int j_max = nrows_activation - tile_j * J - 1;

    extern __shared__ int shared[];
    int * tile_y = shared + J;
    int * tile_x = tile_y + GGML_PAD(J * MMQ_TILE_Y_K, MMQ_NTHREADS);

    float sum[J * MMQ_I / MMQ_NTHREADS] = {0.0f};
    constexpr int q8_block_ints = sizeof(block_q8_1_mmq) / sizeof(int);
    const int kernel_blocks_per_weight_row = fixed_blocks_per_weight_row > 0
        ? fixed_blocks_per_weight_row
        : blocks_per_weight_row;

    for (int kb = 0; kb < kernel_blocks_per_weight_row; ++kb) {
        const int weight_block_offset =
            tile_i * MMQ_I * kernel_blocks_per_weight_row + kb;
        mmq_load_target<type, J, !full_i>(
            weights,
            tile_x,
            weight_block_offset,
            i_max,
            kernel_blocks_per_weight_row);

#pragma unroll
        for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const int src =
                ((2 * kb) * nrows_activation_padded + tile_j * J) * q8_block_ints + l;
            tile_y[l] = activations[src];
        }
        __syncthreads();
        mmq_vec_dot_target<type, J>(tile_x, tile_y, sum, 0);
        __syncthreads();

#pragma unroll
        for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const int src =
                ((2 * kb + 1) * nrows_activation_padded + tile_j * J) * q8_block_ints + l;
            tile_y[l] = activations[src];
        }
        __syncthreads();
        mmq_vec_dot_target<type, J>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }

    mmq_write_back_bf16<type, J, full_i, full_j>(
        sum,
        dst + tile_j * J * nrows_weight + tile_i * MMQ_I,
        nrows_weight,
        i_max,
        j_max);
}

template <int J, int groups, int blocks_per_weight_row, bool fallback>
static __device__ __forceinline__ void fixed_grouped_q8_0_mmq_bf16_body(
        const char * __restrict__ weights,
        const int * __restrict__ activations,
        __hip_bfloat16 * __restrict__ dst,
        int tokens,
        int nrows_weight,
        int64_t bytes_per_group) {
    constexpr ggml_type type = GGML_TYPE_Q8_0;
    constexpr int q8_block_ints = sizeof(block_q8_1_mmq) / sizeof(int);
    static_assert(MMQ_TILE_Y_K == q8_block_ints, "unexpected fixed-group Q8 tile layout");

    const int tile_i = blockIdx.x;
    const int token_start = blockIdx.y * J;
    const int group = blockIdx.z;
    const int i_max = min(MMQ_I, nrows_weight - tile_i * MMQ_I) - 1;
    const int j_max = min(J, tokens - token_start) - 1;
    if (j_max < 0) {
        return;
    }

    const int total_activation_rows = tokens * groups;
    const char * group_weights = weights + static_cast<int64_t>(group) * bytes_per_group;
    extern __shared__ int shared[];
    int * tile_y = shared + J;
    int * tile_x = tile_y + GGML_PAD(J * MMQ_TILE_Y_K, MMQ_NTHREADS);
    float sum[J * MMQ_I / MMQ_NTHREADS] = {0.0f};

#pragma unroll 1
    for (int kb = 0; kb < blocks_per_weight_row; ++kb) {
        const int weight_block_offset =
            tile_i * MMQ_I * blocks_per_weight_row + kb;
        mmq_load_target<type, J, fallback>(
            group_weights,
            tile_x,
            weight_block_offset,
            i_max,
            blocks_per_weight_row);

#pragma unroll
        for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const int local_token = l / q8_block_ints;
            const int q8_int = l % q8_block_ints;
            if (local_token <= j_max) {
                const int activation_row =
                    (token_start + local_token) * groups + group;
                tile_y[l] = activations[
                    ((2 * kb) * total_activation_rows + activation_row) *
                        q8_block_ints +
                    q8_int];
            } else {
                tile_y[l] = 0;
            }
        }
        __syncthreads();
        mmq_vec_dot_target<type, J, fallback>(tile_x, tile_y, sum, 0);
        __syncthreads();

#pragma unroll
        for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const int local_token = l / q8_block_ints;
            const int q8_int = l % q8_block_ints;
            if (local_token <= j_max) {
                const int activation_row =
                    (token_start + local_token) * groups + group;
                tile_y[l] = activations[
                    ((2 * kb + 1) * total_activation_rows + activation_row) *
                        q8_block_ints +
                    q8_int];
            } else {
                tile_y[l] = 0;
            }
        }
        __syncthreads();
        mmq_vec_dot_target<type, J, fallback>(
            tile_x, tile_y, sum, MMQ_TILE_NE_K);
        __syncthreads();
    }

    mmq_write_back_bf16<type, J, !fallback, false>(
        sum,
        dst + (token_start * groups + group) * nrows_weight + tile_i * MMQ_I,
        groups * nrows_weight,
        i_max,
        j_max);
}

template <int J>
static __device__ __forceinline__ void
load_grouped_nonaligned_full_activation_tile(
        const int * __restrict__ activation,
        int * __restrict__ tile_y) {
    constexpr int tile_ints = J * MMQ_TILE_Y_K;
    constexpr int complete_ints = tile_ints / MMQ_NTHREADS * MMQ_NTHREADS;
    static_assert(tile_ints % MMQ_NTHREADS != 0);
    const int lane = threadIdx.y * WARP_SIZE + threadIdx.x;

#pragma unroll
    for (int l0 = 0; l0 < complete_ints; l0 += MMQ_NTHREADS) {
        const int l = l0 + lane;
        tile_y[l] = activation[l];
    }
    const int l = complete_ints + lane;
    tile_y[l] = l < tile_ints ? activation[l] : 0;
}

template <ggml_type type, int J, bool fixed_shape, bool full_j>
static __device__ __forceinline__ void grouped_mmq_k_block(
        const char * __restrict__ expert_weights,
        const int * __restrict__ activations,
        int * __restrict__ tile_x,
        int * __restrict__ tile_y,
        float * __restrict__ sum,
        int tile_i,
        int row_start,
        int j_max,
        int nrows_activation,
        int kernel_blocks_per_weight_row,
        int i_max,
        int kb) {
    constexpr int q8_block_ints = sizeof(block_q8_1_mmq) / sizeof(int);
    const int activation_plane_stride = nrows_activation * q8_block_ints;
    const int valid_activation_ints = (j_max + 1) * q8_block_ints;
    const int * activation_k = activations
        + ((2 * kb) * nrows_activation + row_start) * q8_block_ints;
    const int weight_block_offset =
        tile_i * MMQ_I * kernel_blocks_per_weight_row + kb;
    mmq_load_target<type, J, !fixed_shape>(
        expert_weights,
        tile_x,
        weight_block_offset,
        i_max,
        kernel_blocks_per_weight_row);

    if constexpr (full_j && J * MMQ_TILE_Y_K % MMQ_NTHREADS != 0) {
        load_grouped_nonaligned_full_activation_tile<J>(activation_k, tile_y);
    } else {
#pragma unroll
        for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            if constexpr (full_j) {
                tile_y[l] = activation_k[l];
            } else if (l < valid_activation_ints) {
                tile_y[l] = activation_k[l];
            } else {
                tile_y[l] = 0;
            }
        }
    }
    __syncthreads();
    mmq_vec_dot_target<type, J, !fixed_shape>(tile_x, tile_y, sum, 0);
    __syncthreads();

    if constexpr (full_j && J * MMQ_TILE_Y_K % MMQ_NTHREADS != 0) {
        load_grouped_nonaligned_full_activation_tile<J>(
            activation_k + activation_plane_stride, tile_y);
    } else {
#pragma unroll
        for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            if constexpr (full_j) {
                tile_y[l] = activation_k[activation_plane_stride + l];
            } else if (l < valid_activation_ints) {
                tile_y[l] = activation_k[activation_plane_stride + l];
            } else {
                tile_y[l] = 0;
            }
        }
    }
    __syncthreads();
    mmq_vec_dot_target<type, J, !fixed_shape>(
        tile_x, tile_y, sum, MMQ_TILE_NE_K);
    __syncthreads();
}

template <
    ggml_type type,
    int J,
    int fixed_nrows_weight,
    int fixed_blocks_per_weight_row,
    bool full_j,
    bool rolled_q2_k = false>
static __device__ __forceinline__ void grouped_mmq_row_tile(
        const char * __restrict__ expert_weights,
        const int * __restrict__ activations,
        __hip_bfloat16 * __restrict__ dst,
        int * __restrict__ tile_x,
        int * __restrict__ tile_y,
        int tile_i,
        int row_start,
        int row_end,
        int nrows_weight,
        int nrows_activation,
        int blocks_per_weight_row) {
    constexpr bool fixed_shape = fixed_nrows_weight > 0 && fixed_blocks_per_weight_row > 0;
    const int kernel_nrows_weight = fixed_shape ? fixed_nrows_weight : nrows_weight;
    const int kernel_blocks_per_weight_row =
        fixed_shape ? fixed_blocks_per_weight_row : blocks_per_weight_row;
    const int i_max = fixed_shape ? MMQ_I - 1 : kernel_nrows_weight - tile_i * MMQ_I - 1;
    const int j_max = full_j ? J - 1 : row_end - row_start - 1;
    float sum[J * MMQ_I / MMQ_NTHREADS] = {0.0f};

    if constexpr (fixed_blocks_per_weight_row == 2) {
        grouped_mmq_k_block<type, J, fixed_shape, full_j>(
            expert_weights,
            activations,
            tile_x,
            tile_y,
            sum,
            tile_i,
            row_start,
            j_max,
            nrows_activation,
            kernel_blocks_per_weight_row,
            i_max,
            0);
        grouped_mmq_k_block<type, J, fixed_shape, full_j>(
            expert_weights,
            activations,
            tile_x,
            tile_y,
            sum,
            tile_i,
            row_start,
            j_max,
            nrows_activation,
            kernel_blocks_per_weight_row,
            i_max,
            1);
    } else {
        constexpr int q8_block_ints = sizeof(block_q8_1_mmq) / sizeof(int);
        const int activation_half_stride = nrows_activation * q8_block_ints;
        const int * activation_k = activations + row_start * q8_block_ints;
        int weight_block_offset = tile_i * MMQ_I * kernel_blocks_per_weight_row;
#pragma unroll 1
        for (int kb = 0; kb < kernel_blocks_per_weight_row; ++kb) {
            mmq_load_target<type, J, !fixed_shape>(
                expert_weights,
                tile_x,
                weight_block_offset,
                i_max,
                kernel_blocks_per_weight_row);

            if constexpr (full_j && J * MMQ_TILE_Y_K % MMQ_NTHREADS != 0) {
                load_grouped_nonaligned_full_activation_tile<J>(
                    activation_k, tile_y);
            } else {
#pragma unroll
                for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
                    const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                    if constexpr (full_j) {
                        tile_y[l] = activation_k[l];
                    } else {
                        const int local_row = l / q8_block_ints;
                        const int q8_int = l % q8_block_ints;
                        if (local_row <= j_max) {
                            tile_y[l] = activation_k[
                                local_row * q8_block_ints + q8_int];
                        } else {
                            tile_y[l] = 0;
                        }
                    }
                }
            }
            __syncthreads();
            mmq_vec_dot_target<type, J, !fixed_shape, rolled_q2_k>(
                tile_x, tile_y, sum, 0);
            __syncthreads();

            if constexpr (full_j && J * MMQ_TILE_Y_K % MMQ_NTHREADS != 0) {
                load_grouped_nonaligned_full_activation_tile<J>(
                    activation_k + activation_half_stride, tile_y);
            } else {
#pragma unroll
                for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
                    const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                    if constexpr (full_j) {
                        tile_y[l] = activation_k[activation_half_stride + l];
                    } else {
                        const int local_row = l / q8_block_ints;
                        const int q8_int = l % q8_block_ints;
                        if (local_row <= j_max) {
                            tile_y[l] = activation_k[
                                activation_half_stride +
                                local_row * q8_block_ints + q8_int];
                        } else {
                            tile_y[l] = 0;
                        }
                    }
                }
            }
            __syncthreads();
            mmq_vec_dot_target<type, J, !fixed_shape, rolled_q2_k>(
                tile_x, tile_y, sum, MMQ_TILE_NE_K);
            __syncthreads();

            ++weight_block_offset;
            activation_k += 2 * activation_half_stride;
        }
    }

    mmq_write_back_bf16<type, J, fixed_shape, full_j>(
        sum,
        dst + row_start * kernel_nrows_weight + tile_i * MMQ_I,
        kernel_nrows_weight,
        i_max,
        j_max);
    __syncthreads();
}

template <
    ggml_type type,
    int J,
    int fixed_nrows_weight,
    int fixed_blocks_per_weight_row,
    bool mixed_j32_tails = false,
    bool mixed_q2_k_tails = false,
    bool rolled_q2_k = false,
    int mixed_j32_rows_a = 0,
    int mixed_j32_rows_b = 0>
static __device__ __forceinline__ void grouped_mmq_tail_tile(
        const char * __restrict__ expert_weights,
        const int * __restrict__ activations,
        __hip_bfloat16 * __restrict__ dst,
        int * __restrict__ tile_x,
        int * __restrict__ tile_y,
        int tile_i,
        int row_start,
        int row_end,
        int nrows_weight,
        int nrows_activation,
        int blocks_per_weight_row) {
    if constexpr (
        mixed_j32_tails &&
        (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_Q4_K) &&
        J == MMQ_J_SMALL
    ) {
        constexpr bool qualified_rows_are_bounded =
            mixed_j32_rows_a > 0 || mixed_j32_rows_b > 0;
        const bool qualified_rows =
            !qualified_rows_are_bounded ||
            nrows_activation == mixed_j32_rows_a ||
            nrows_activation == mixed_j32_rows_b;
        const int tail_rows = row_end - row_start;
        if (qualified_rows && tail_rows <= MMQ_J_TINY) {
            grouped_mmq_row_tile<
                type, MMQ_J_TINY, fixed_nrows_weight, fixed_blocks_per_weight_row,
                false, rolled_q2_k>(
                    expert_weights, activations, dst, tile_x, tile_y, tile_i,
                    row_start, row_end, nrows_weight, nrows_activation,
                    blocks_per_weight_row);
        } else {
            grouped_mmq_row_tile<
                type, J, fixed_nrows_weight, fixed_blocks_per_weight_row,
                false, rolled_q2_k>(
                    expert_weights, activations, dst, tile_x, tile_y, tile_i,
                    row_start, row_end, nrows_weight, nrows_activation,
                    blocks_per_weight_row);
        }
    } else if constexpr (
        mixed_q2_k_tails && type == GGML_TYPE_Q2_K && J == MMQ_J_TINY
    ) {
        const int tail_rows = row_end - row_start;
        if (tail_rows <= MMQ_J_MIN) {
            grouped_mmq_row_tile<
                type, MMQ_J_MIN, fixed_nrows_weight, fixed_blocks_per_weight_row,
                false, rolled_q2_k>(
                    expert_weights, activations, dst, tile_x, tile_y, tile_i,
                    row_start, row_end, nrows_weight, nrows_activation,
                    blocks_per_weight_row);
        } else {
            grouped_mmq_row_tile<
                type, J, fixed_nrows_weight, fixed_blocks_per_weight_row,
                false, rolled_q2_k>(
                    expert_weights, activations, dst, tile_x, tile_y, tile_i,
                    row_start, row_end, nrows_weight, nrows_activation,
                    blocks_per_weight_row);
        }
    } else {
        grouped_mmq_row_tile<
            type, J, fixed_nrows_weight, fixed_blocks_per_weight_row,
            false, rolled_q2_k>(
                expert_weights, activations, dst, tile_x, tile_y, tile_i,
                row_start, row_end, nrows_weight, nrows_activation,
                blocks_per_weight_row);
    }
}

template <
    ggml_type type,
    int J,
    int fixed_nrows_weight,
    int fixed_blocks_per_weight_row,
    bool mixed_j32_tails = false,
    bool mixed_q2_k_tails = false,
    bool rolled_q2_k = false,
    int mixed_j32_rows_a = 0,
    int mixed_j32_rows_b = 0>
static __device__ __forceinline__ void grouped_mmq_bf16_body(
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
    constexpr int q8_block_ints = sizeof(block_q8_1_mmq) / sizeof(int);
    static_assert(MMQ_TILE_Y_K == q8_block_ints, "unexpected grouped Q8 tile layout");

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

    const char * expert_weights = weights + expert * bytes_per_expert;
    extern __shared__ int shared[];
    int * tile_y = shared + J;
    int * tile_x = tile_y + GGML_PAD(J * MMQ_TILE_Y_K, MMQ_NTHREADS);

    int row_start = row_begin;
    for (; row_start + J <= row_end; row_start += J) {
        grouped_mmq_row_tile<
            type, J, fixed_nrows_weight, fixed_blocks_per_weight_row,
            true, rolled_q2_k>(
                expert_weights,
                activations,
                dst,
                tile_x,
                tile_y,
                tile_i,
                row_start,
                row_end,
                nrows_weight,
                nrows_activation,
                blocks_per_weight_row);
    }
    if (row_start < row_end) {
        grouped_mmq_tail_tile<
            type, J, fixed_nrows_weight, fixed_blocks_per_weight_row,
            mixed_j32_tails, mixed_q2_k_tails, rolled_q2_k,
            mixed_j32_rows_a, mixed_j32_rows_b>(
                expert_weights,
                activations,
                dst,
                tile_x,
                tile_y,
                tile_i,
                row_start,
                row_end,
                nrows_weight,
                nrows_activation,
                blocks_per_weight_row);
    }
}

static __device__ __forceinline__ void grouped_mmq_build_row_tasks_body(
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int32_t * __restrict__ task_count,
        int32_t * __restrict__ task_experts,
        int32_t * __restrict__ task_row_starts,
        int32_t * __restrict__ task_row_ends,
        int num_experts,
        int num_groups,
        int nrows_activation,
        int row_tile) {
    __shared__ int cumulative_tasks[256];
    const int group = threadIdx.x;

    int row_begin = 0;
    int row_end = 0;
    int expert = -1;
    int group_tasks = 0;
    if (group < num_groups) {
        row_begin = group == 0 ? 0 : expert_offsets[group - 1];
        row_end = expert_offsets[group];
        const int64_t expert64 = expert_indices[group];
        if (
            expert64 >= 0 && expert64 < num_experts && row_begin >= 0 &&
            row_end > row_begin && row_end <= nrows_activation
        ) {
            expert = static_cast<int>(expert64);
            group_tasks = (row_end - row_begin + row_tile - 1) / row_tile;
        }
    }
    cumulative_tasks[group] = group_tasks;
    __syncthreads();

#pragma unroll
    for (int offset = 1; offset < 256; offset <<= 1) {
        const int add = group >= offset ? cumulative_tasks[group - offset] : 0;
        __syncthreads();
        cumulative_tasks[group] += add;
        __syncthreads();
    }

    if (group < num_groups) {
        const int task_begin = group == 0 ? 0 : cumulative_tasks[group - 1];
        for (int task = 0; task < group_tasks; ++task) {
            const int task_index = task_begin + task;
            const int task_row_start = row_begin + task * row_tile;
            task_experts[task_index] = expert;
            task_row_starts[task_index] = task_row_start;
            task_row_ends[task_index] = min(task_row_start + row_tile, row_end);
        }
        if (group == num_groups - 1) {
            task_count[0] = cumulative_tasks[group];
        }
    }
}

template <ggml_type type, int J, int fixed_nrows_weight, int fixed_blocks_per_weight_row>
static __device__ __forceinline__ void grouped_mmq_row_task_body(
        const char * __restrict__ weights,
        const int * __restrict__ activations,
        __hip_bfloat16 * __restrict__ dst,
        const int32_t * __restrict__ task_count,
        const int32_t * __restrict__ task_experts,
        const int32_t * __restrict__ task_row_starts,
        const int32_t * __restrict__ task_row_ends,
        int nrows_activation,
        int64_t bytes_per_expert) {
    constexpr int q8_block_ints = sizeof(block_q8_1_mmq) / sizeof(int);
    static_assert(MMQ_TILE_Y_K == q8_block_ints, "unexpected grouped Q8 tile layout");
    static_assert(
        fixed_nrows_weight > 0 && fixed_blocks_per_weight_row > 0,
        "row-task grouped MMQ requires a fixed shape");

    const int task = blockIdx.y;
    if (task >= task_count[0]) {
        return;
    }

    const int tile_i = blockIdx.x;
    const int expert = task_experts[task];
    const int row_start = task_row_starts[task];
    const int row_end = task_row_ends[task];
    const char * expert_weights = weights + static_cast<int64_t>(expert) * bytes_per_expert;

    extern __shared__ int shared[];
    int * tile_y = shared + J;
    int * tile_x = tile_y + GGML_PAD(J * MMQ_TILE_Y_K, MMQ_NTHREADS);

    if (row_end - row_start == J) {
        grouped_mmq_row_tile<
            type, J, fixed_nrows_weight, fixed_blocks_per_weight_row, true>(
                expert_weights,
                activations,
                dst,
                tile_x,
                tile_y,
                tile_i,
                row_start,
                row_end,
                fixed_nrows_weight,
                nrows_activation,
                fixed_blocks_per_weight_row);
    } else {
        grouped_mmq_row_tile<
            type, J, fixed_nrows_weight, fixed_blocks_per_weight_row, false>(
                expert_weights,
                activations,
                dst,
                tile_x,
                tile_y,
                tile_i,
                row_start,
                row_end,
                fixed_nrows_weight,
                nrows_activation,
                fixed_blocks_per_weight_row);
    }
}
