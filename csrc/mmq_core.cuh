#pragma once

#include "ck/gguf_decode.cuh"
#include "vendor/llama_cpp/common.cuh"
#include "vendor/llama_cpp/mma.cuh"

#include <cstdint>

// Narrow dense-MMQ configuration for gfx1151/RDNA3.5. These are the inherited
// llama.cpp settings for J=128 with fallback row bounds enabled.
static constexpr int MMQ_ITER_K = 256;
static constexpr int MMQ_TILE_NE_K = 32;
static constexpr int MMQ_TILE_Y_K = MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1;

// A control can stage half of MMQ_ITER_K at a time. The weight tile then holds
// one 128-value half, which halves its LDS footprint and lifts the body to four
// workgroups per WGP, while the bytes loaded and the barriers per 256 values are
// unchanged because a stage's weight and activation loads already share one
// barrier. Only the level layouts that store one byte per value are covered.
// A control that reads activation fragments from global memory keeps only the
// four per-group scales of each token in LDS, one sixteen-byte row per token.
#if defined(MMQ_FRAGMENT_ACT)
static constexpr int MMQ_META_STRIDE = 4;
#else
static constexpr int MMQ_META_STRIDE = MMQ_TILE_Y_K;
#endif

// Planar activation layout, used by controls that read their fragments straight
// from the workspace. Inside every (128-value block, 16-token chunk) the sixteen
// metadata headers gather into one 256-byte plane and each of the four 32-value
// groups forms its own 512-byte plane of sixteen 32-byte chunks, so a warp's
// fragment reads become one contiguous span per group instead of sixteen groups
// scattered at the per-token cell stride. The total per token and block is
// unchanged: sixteen bytes of scales plus 128 bytes of values.

#if defined(MMQ_HALF_STAGE)
static constexpr int MMQ_STAGE_K = MMQ_ITER_K / 2;
static constexpr int MMQ_QUANT_INTS = MMQ_TILE_NE_K;
#else
static constexpr int MMQ_STAGE_K = MMQ_ITER_K;
static constexpr int MMQ_QUANT_INTS = 2 * MMQ_TILE_NE_K;
#endif

// The row tile and the workgroup width are overridable by a control that stages
// a wider weight tile: `I` must equal the warp count times the sixteen rows the
// vector dot gives each warp, so a wider `I` also wants a wider workgroup. A
// control that defines `MMQ_I` or `MMQ_NTHREADS` before including this header
// gets that configuration, and every other control keeps the inherited one.
#if !defined(MMQ_I)
static constexpr int MMQ_I = 64;
#endif
static constexpr int MMQ_J = 128;
static constexpr int MMQ_J_MEDIUM = 80;
static constexpr int MMQ_J_SMALL = 64;
static constexpr int MMQ_J_TINY = 32;
static constexpr int MMQ_J_MIN = 16;
#if !defined(MMQ_NTHREADS)
static constexpr int MMQ_NTHREADS = 128;
#endif
static constexpr int MMQ_NWARPS = MMQ_NTHREADS / WARP_SIZE;
// Compact single-stage weight tile: 32 quant ints for 128 values plus that
// stage's scale/min pairs, so the two stages of a k block share one footprint.
// Only the controls that measured a gain enable it.
#ifndef MMQ_COMPACT_QUANT_INTS
#define MMQ_COMPACT_QUANT_INTS 32
#endif
#ifndef MMQ_COMPACT_STRIDE
#define MMQ_COMPACT_STRIDE 36
#endif
#ifndef MMQ_COMPACT_STRIDE_Q2_K
#define MMQ_COMPACT_STRIDE_Q2_K 40
#endif

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
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0 ||
        type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_Q2_0 || type == GGML_TYPE_IQ4_NL ||
        type == GGML_TYPE_IQ4_XS
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
#if defined(MMQ_HALF_STAGE)
    return MMQ_QUANT_INTS + MMQ_QUANT_INTS / QI8_0 + 4;
#elif defined(MMQ_COMPACT_TILE)
    return type == GGML_TYPE_Q2_K
        ? MMQ_COMPACT_STRIDE_Q2_K
        : (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)
        ? MMQ_COMPACT_STRIDE
        : mmq_sram_layout(type) == GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0
            ? 2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / QI8_0 + 4
            : mmq_sram_layout(type) == GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K
                ? 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K + 4
                : mmq_sram_layout(type) == GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K
                    ? 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / 2 + 4
                    : mmq_sram_layout(type) == GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K
                        ? 2 * MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI6_K + MMQ_TILE_NE_K / 8 + 7
                        : 2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / QI8_1 + 4;
#else
    switch (mmq_sram_layout(type)) {
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0:
#if defined(MMQ_HALF_STAGE)
            return MMQ_QUANT_INTS + MMQ_QUANT_INTS / QI8_0 + 4;
#else
            return 2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / QI8_0 + 4;
#endif
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
#endif
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

// Vendored llama.cpp templates. These three files are fragments rather than
// headers: they expand against the configuration and helpers defined above,
// so they are included here and nowhere else.
// Fragment-order activation reads. The int8 B fragment of one vector dot needs,
// for each of sixteen tokens, the thirty-two contiguous bytes of one 32-value
// group: `load_ldmatrix` fetches exactly that in two 16-byte chunks at
// `xs0 + (lane % 16)*stride` and `+16 B`. A Q8_1 workspace block already stores
// its four 32-byte groups back to back behind a 16-byte metadata header, so the
// same 32 bytes are contiguous in global memory too and the LDS tile in between
// buys nothing but a copy. This loader reads them straight from global.
template <typename T, ggml_cuda_mma::data_layout dl>
static __device__ __forceinline__ void load_fragment_global(
        ggml_cuda_mma::tile<16, 8, T, dl> & t,
        const T * __restrict__ xs0,
        const int stride) {
    static_assert(
        dl == ggml_cuda_mma::DATA_LAYOUT_I_MAJOR_MIRRORED, "bad data layout");
    static_assert(sizeof(t.x) == 32, "bad ne");
    const T * base = xs0 + t.get_i(0) * stride;
    __builtin_memcpy(t.x + 0, base + 0, 16);
    __builtin_memcpy(t.x + 4, base + 4, 16);
}

#include "vendor/llama_cpp/mmq-load-targets.cuh"
#include "vendor/llama_cpp/mmq-vec-dot-targets.cuh"
#include "vendor/llama_cpp/mmq-vec-dot-q2-k-rolled.cuh"

// Repository-local vec-dot targets that hoist the activation metadata out
// of the column loop. Selected per control through the MMQ_EPILOGUE_HOISTED
// macro so both forms can be measured against each other. They mirror the
// vendored targets of the same shape and only move the metadata loads.
template <ggml_type type, int J, bool fallback, mmq_q8_1_metadata_layout metadata_layout>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_q8_1_hoisted(
        const int * __restrict__ x,
        const int * __restrict__ y,
        float * __restrict__ sum,
        const int k00) {
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16, 8, int, input_layout> tile_A;
    typedef tile<16, 8, int, input_layout> tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx = rows_per_warp / tile_C::I;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int * x_qs = (const int *) x;
    const float * x_df = (const float *) x_qs + 2 * MMQ_TILE_NE_K;
    const int * y_qs = (const int *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(
                A[n], x_qs + (i0 + n * tile_A::I) * sram_stride + k0, sram_stride);
        }
        float dA[ntx][tile_C::ne];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                dA[n][l] = x_df[i * sram_stride + k0 / QI8_0];
            }
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            float dB;
            const int j = j0 + tile_C::get_j(0);
            if (metadata_layout == MMQ_Q8_1_METADATA_F32_D4) {
                dB = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];
            } else {
                dB = __low2float(y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int row = (j0 / tile_C::J + n) * tile_C::ne + l;
                    sum[row] += C.x[l] * dA[n][l] * dB;
                }
            }
        }
    }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_1_q8_1_hoisted(
        const int * __restrict__ x,
        const int * __restrict__ y,
        float * __restrict__ sum,
        const int k00) {
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16, 8, int, input_layout> tile_A;
    typedef tile<16, 8, int, input_layout> tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx = rows_per_warp / tile_C::I;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int * x_qs = (const int *) x;
#if defined(MMQ_COMPACT_TILE)
    const half2 * x_dm = (const half2 *) x_qs + MMQ_COMPACT_QUANT_INTS;
#else
    const half2 * x_dm = (const half2 *) x_qs + 2 * MMQ_TILE_NE_K;
#endif
    const int * y_qs = (const int *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(
                A[n], x_qs + (i0 + n * tile_A::I) * sram_stride + k0, sram_stride);
        }
        float2 dmA[ntx][tile_C::ne];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                dmA[n][l] = __half22float2(x_dm[i * sram_stride + k0 / QI8_1]);
            }
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float2 dsB = __half22float2(y_dm[j * MMQ_TILE_Y_K + k01 / QI8_1]);

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int row = (j0 / tile_C::J + n) * tile_C::ne + l;
                    sum[row] += dmA[n][l].x * dsB.x * C.x[l];
                    sum[row] += dmA[n][l].y * dsB.y;
                }
            }
        }
    }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_hoisted(
        const int * __restrict__ x,
        const int * __restrict__ y,
        float * __restrict__ sum,
        const int k00) {
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16, 4, int, input_layout> tile_A;
    typedef tile<16, 4, int, input_layout> tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx = rows_per_warp / tile_C::I;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int * x_qs = (const int *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K * 2;
    const int * y_qs = (const int *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(
                A[n], x_qs + (i0 + n * tile_A::I) * sram_stride + k0, sram_stride);
        }
        float dA[ntx][tile_C::ne];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                dA[n][l] = x_df[i * sram_stride + k0 / 4];
            }
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int row = (j0 / tile_C::J + n) * tile_C::ne + l;
                    sum[row] += C.x[l] * dA[n][l] * dB;
                }
            }
        }
    }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q6_K_q8_1_hoisted(
        const int * __restrict__ x,
        const int * __restrict__ y,
        float * __restrict__ sum,
        const int k00) {
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16, 4, int, input_layout> tile_A;
    typedef tile<16, 4, int, input_layout> tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx = rows_per_warp / tile_C::I;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int * x_qs = (const int *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K * 2;
    const int * x_sc = (const int *) x_df + MMQ_TILE_NE_K / QI6_K;
    const int * y_qs = (const int *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(
                A[n], x_qs + (i0 + n * tile_A::I) * sram_stride + k0, sram_stride);
        }
        float dA[ntx][tile_C::ne];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = i0 + n * tile_A::I + tile_C::get_i(l);
                const int8_t * sc =
                    (const int8_t *) (x_sc + i * sram_stride + k00 / 16);
                dA[n][l] = x_df[i * sram_stride] * sc[k01 / 4];
            }
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int row = (j0 / tile_C::J + n) * tile_C::ne + l;
                    sum[row] += C.x[l] * dA[n][l] * dB;
                }
            }
        }
    }
}

template <ggml_type type, int J, bool fallback, bool rolled_q2_k>
static __device__ __forceinline__ void mmq_vec_dot_target_hoisted(
        const int * x, const int * y, float * sum, int k00) {
    if constexpr (type == GGML_TYPE_Q8_0) {
        ggml_cuda_mmq_vec_dot_q8_0_q8_1_hoisted<
            type, J, fallback, MMQ_Q8_1_METADATA_F32_D4>(x, y, sum, k00);
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_hoisted<type, J, fallback>(
            x, y, sum, k00);
    } else if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        ggml_cuda_mmq_vec_dot_q8_1_q8_1_hoisted<type, J, fallback>(
            x, y, sum, k00);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        ggml_cuda_mmq_vec_dot_q6_K_q8_1_hoisted<type, J, fallback>(
            x, y, sum, k00);
    }
}

// Q2_0 loader: 64 weights in an 18-byte block, one fp16 scale and four
// consecutive 2-bit codes per payload byte whose level is `code - 1`. A stage
// therefore holds four blocks. The expanded int8 values fill the same LDS slots
// the Q8_0 loader produces, and each block scale is repeated over the two
// 32-value scale entries it covers, so the Q8_0 vector dot consumes the tile
// unchanged.
// Expand one payload byte into its four levels. The table variant reads a
// precomputed LDS table instead of repeating the spread and the level offset.
static __device__ __forceinline__ int q2_0_spread_levels(const int byte) {
    const int spread =
        (byte & 0x03) | ((byte & 0x0C) << 6) | ((byte & 0x30) << 12) | ((byte & 0xC0) << 18);
    return __vsubss4(spread, 0x01010101);
}

template <ggml_type type, int J, bool fallback, bool table_decode = false>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q2_0(
        const char * __restrict__ x,
        int * __restrict__ x_tile,
        const int kbx0,
        const int i_max,
        const int stride) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int * x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + 2 * MMQ_TILE_NE_K);

    constexpr int blocks_per_iteration = MMQ_STAGE_K / QK2_0;
    constexpr int threads_per_row = blocks_per_iteration * QI2_0;
    constexpr int nrows = warp_size / threads_per_row;
    constexpr int scale_entries_per_block = QK2_0 / QK8_1;
    constexpr int scale_entries_per_row = blocks_per_iteration * scale_entries_per_block;

    const int txi = threadIdx.x % threads_per_row;
    const int kbx = txi / QI2_0;
    const int kqsx = txi % QI2_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q2_0 * bxi = (const block_q2_0 *) x + kbx0 + i * stride + kbx;
        const uint16_t * qxi = (const uint16_t *) bxi->qs + 4 * kqsx;
        const int dst = kbx * (scale_entries_per_block * QI8_0) + kqsx * QI8_0;

        if constexpr (table_decode) {
            const int * levels = x_tile + I * sram_stride;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int word = qxi[j];
                x_qs[i * sram_stride + dst + 2 * j + 0] = levels[word & 0xFF];
                x_qs[i * sram_stride + dst + 2 * j + 1] = levels[(word >> 8) & 0xFF];
            }
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int word = qxi[j];
#pragma unroll
                for (int half_word = 0; half_word < 2; ++half_word) {
                    const int byte = (word >> (8 * half_word)) & 0xFF;
                    x_qs[i * sram_stride + dst + 2 * j + half_word] =
                        q2_0_spread_levels(byte);
                }
            }
        }
    }

    const int ksx = threadIdx.x % scale_entries_per_row;
    const int scale_block = ksx / scale_entries_per_block;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q2_0 * bxi = (const block_q2_0 *) x + kbx0 + i * stride + scale_block;
        x_df[i * sram_stride + ksx] = __half2float(bxi->d);
    }
}

// Q4_0 loader: 32 weights in an 18-byte block, one fp16 scale and a
// nibble-plane pair whose low nibble is weight j and high nibble weight j + 16,
// with the level `nibble - 8`. A stage holds eight blocks, whose expanded int8
// values fill the same LDS slots the Q8_0 loader produces, and the 32-wide
// block means one scale per 32-value group, so no metadata is replicated.
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q4_0(
        const char * __restrict__ x,
        int * __restrict__ x_tile,
        const int kbx0,
        const int i_max,
        const int stride) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int * x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + 2 * MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_0);
    constexpr int nrows = warp_size / threads_per_row;
    constexpr int blocks_per_iteration = MMQ_STAGE_K / QK4_0;

    const int txi = threadIdx.x % threads_per_row;
    const int kbx = txi / QI4_0;
    const int kqsx = txi % QI4_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + threadIdx.x / threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i * stride + kbx;
        const int qs0 = get_int_b2(bxi->qs, kqsx);

        x_qs[i * sram_stride + kbx * (2 * QI4_0) + kqsx] =
            __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
        x_qs[i * sram_stride + kbx * (2 * QI4_0) + kqsx + QI4_0] =
            __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
    }

    constexpr int rows_per_warp = warp_size / blocks_per_iteration;
    const int kbxd = threadIdx.x % blocks_per_iteration;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_iteration;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i * stride + kbxd;
        x_df[i * sram_stride + kbxd] = __half2float(bxi->d);
    }
}

// Q5_0 loader: 32 weights in a 22-byte block, one fp16 scale, a little-endian
// fifth-bit word and Q4_0's nibble planes, with the level `nibble | 16*bit - 16`.
// A stage holds eight blocks, whose expanded int8 values fill the same LDS slots
// the Q8_0 loader produces, and the 32-wide block means one scale per 32-value
// group.
// Place the four fifth bits of one group at bit 4 of four consecutive bytes.
static __device__ __forceinline__ int q5_0_spread_bits(const int bits) {
    return ((bits & 0x01) << 4) | ((bits & 0x02) << 11) | ((bits & 0x04) << 18) |
        ((bits & 0x08) << 25);
}

template <ggml_type type, int J, bool fallback, bool table_decode = false>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q5_0(
        const char * __restrict__ x,
        int * __restrict__ x_tile,
        const int kbx0,
        const int i_max,
        const int stride) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int * x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + 2 * MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_0);
    constexpr int nrows = warp_size / threads_per_row;
    constexpr int blocks_per_iteration = MMQ_STAGE_K / QK5_0;
    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI5_0;

    const int txi = threadIdx.x % threads_per_row;
    const int kbx = txi / QI5_0;
    const int kqsx = txi % QI5_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + threadIdx.x / threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *) x + kbx0 + i * stride + kbx;

        const int ql = get_int_b2(bxi->qs, kqsx);
        const int qh = get_int_b2(bxi->qh, 0) >> (4 * kqsx);

        if constexpr (table_decode) {
            // The nibble plane is already one nibble per byte, so only the
            // fifth-bit spread is read from the table.
            const int * bit_table = x_tile + I * sram_stride;
            const int bits_lo = bit_table[qh & 0xF];
            const int bits_hi = bit_table[(qh >> 16) & 0xF];
            x_qs[i * sram_stride + kbx * (2 * QI5_0) + kqsx] = __vsubss4(
                ((ql >> 0) & 0x0F0F0F0F) | bits_lo, 0x10101010);
            x_qs[i * sram_stride + kbx * (2 * QI5_0) + kqsx + QI5_0] = __vsubss4(
                ((ql >> 4) & 0x0F0F0F0F) | bits_hi, 0x10101010);
            continue;
        }

        int qs0 = (ql >> 0) & 0x0F0F0F0F;
        qs0 |= (qh << 4) & 0x00000010;
        qs0 |= (qh << 11) & 0x00001000;
        qs0 |= (qh << 18) & 0x00100000;
        qs0 |= (qh << 25) & 0x10000000;
        x_qs[i * sram_stride + kbx * (2 * QI5_0) + kqsx] =
            __vsubss4(qs0, 0x10101010);

        int qs1 = (ql >> 4) & 0x0F0F0F0F;
        qs1 |= (qh >> 12) & 0x00000010;
        qs1 |= (qh >> 5) & 0x00001000;
        qs1 |= (qh << 2) & 0x00100000;
        qs1 |= (qh << 9) & 0x10000000;
        x_qs[i * sram_stride + kbx * (2 * QI5_0) + kqsx + QI5_0] =
            __vsubss4(qs1, 0x10101010);
    }

    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *) x + kbx0 + i * stride + kbxd;
        x_df[i * sram_stride + kbxd] = __half2float(bxi->d);
    }
}

// IQ4_NL loader: 32 weights in an 18-byte block, one fp16 scale and Q4_0's
// nibble planes, whose nibbles index the sixteen-level codebook. A stage holds
// eight blocks, whose levels fill the same LDS slots the Q8_0 loader produces,
// with one scale per 32-value group.
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_nl(
        const char * __restrict__ x,
        int * __restrict__ x_tile,
        const int kbx0,
        const int i_max,
        const int stride) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int * x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + 2 * MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_NL);
    constexpr int nrows = warp_size / threads_per_row;
    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_NL;

    const int txi = threadIdx.x % threads_per_row;
    const int kbx = txi / QI4_NL;
    const int kqsx = txi % QI4_NL;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + threadIdx.x / threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i * stride + kbx;
        const int2 v = torch_ggml_ops::ck::iq4_table_lookup_16(get_int_b2(bxi->qs, kqsx), kvalues_iq4nl);

        x_qs[i * sram_stride + kbx * (2 * QI4_NL) + kqsx] = v.x;
        x_qs[i * sram_stride + kbx * (2 * QI4_NL) + kqsx + QI4_NL] = v.y;
    }

    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i * stride + kbxd;
        x_df[i * sram_stride + kbxd] = __half2float(bxi->d);
    }
}

// IQ4_XS loader: 256 weights in a 136-byte block, one fp16 base scale, eight
// 6-bit signed sub-block scales and a nibble payload that indexes the same
// sixteen-level codebook as IQ4_NL. A stage is exactly one block, and its levels
// fill the Q8_0 LDS slots with one f32 scale per 32-value group.
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_xs(
        const char * __restrict__ x,
        int * __restrict__ x_tile,
        const int kbx0,
        const int i_max,
        const int stride) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int * x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + 2 * MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_XS);
    constexpr int nrows = warp_size / threads_per_row;

    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + threadIdx.x / threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i * stride;
        const int2 v = torch_ggml_ops::ck::iq4_table_lookup_16(get_int_b4(bxi->qs, kqsx), kvalues_iq4nl);
        const int k0 = 8 * (kqsx / 4) + kqsx % 4;

        x_qs[i * sram_stride + k0 + 0] = v.x;
        x_qs[i * sram_stride + k0 + 4] = v.y;
    }

    constexpr int rows_per_warp = warp_size / 8;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / (MMQ_TILE_NE_K / 4);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i * stride;
        const int scale_index = threadIdx.x % 8;
        const int ls =
            ((bxi->scales_l[scale_index / 2] >> (4 * (scale_index % 2))) & 0x0F) |
            (((bxi->scales_h >> (2 * scale_index)) & 0x03) << 4);
        x_df[i * sram_stride + scale_index] = __half2float(bxi->d) * (ls - 32);
    }
}

template <ggml_type type, int J, bool fallback = true, bool table_decode = false>
static __device__ __forceinline__ void mmq_load_target(
        const char * x, int * tile, int block_offset, int i_max, int row_stride,
        int stage = 0) {
#if defined(MMQ_COMPACT_TILE)
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        ggml_cuda_mmq_load_tiles_q45_K_compact<type, J, fallback>(
            x, tile, block_offset, i_max, row_stride, stage);
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        ggml_cuda_mmq_load_tiles_q2_K_compact<type, J, fallback>(
            x, tile, block_offset, i_max, row_stride, stage);
    } else
#endif
    if constexpr (type == GGML_TYPE_Q8_0) {
        constexpr int blocks_per_iteration = MMQ_STAGE_K / QK8_0;
        ggml_cuda_mmq_load_tiles_q8_0<type, J, fallback>(
            x,
            tile,
            block_offset * blocks_per_iteration,
            i_max,
            row_stride * blocks_per_iteration);
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        constexpr int blocks_per_iteration = MMQ_STAGE_K / QK4_0;
        ggml_cuda_mmq_load_tiles_q4_0<type, J, fallback>(
            x,
            tile,
            block_offset * blocks_per_iteration,
            i_max,
            row_stride * blocks_per_iteration);
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        ggml_cuda_mmq_load_tiles_iq4_xs<type, J, fallback>(
            x, tile, block_offset, i_max, row_stride);
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        constexpr int blocks_per_iteration = MMQ_STAGE_K / QK4_NL;
        ggml_cuda_mmq_load_tiles_iq4_nl<type, J, fallback>(
            x,
            tile,
            block_offset * blocks_per_iteration,
            i_max,
            row_stride * blocks_per_iteration);
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        constexpr int blocks_per_iteration = MMQ_STAGE_K / QK5_0;
        ggml_cuda_mmq_load_tiles_q5_0<type, J, fallback, table_decode>(
            x,
            tile,
            block_offset * blocks_per_iteration,
            i_max,
            row_stride * blocks_per_iteration);
    } else if constexpr (type == GGML_TYPE_Q2_0) {
        constexpr int blocks_per_iteration = MMQ_STAGE_K / QK2_0;
        ggml_cuda_mmq_load_tiles_q2_0<type, J, fallback, table_decode>(
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
        const int * x, const int * y, float * sum, int k00, const int * meta = nullptr) {
#if defined(MMQ_EPILOGUE_HOISTED)
    if constexpr (
        type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_Q3_K || type == GGML_TYPE_IQ2_S ||
        type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
        type == GGML_TYPE_Q6_K) {
        mmq_vec_dot_target_hoisted<type, J, fallback, rolled_q2_k>(
            x, y, sum, k00);
        return;
    }
#endif
    if constexpr (
        type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_Q2_0 || type == GGML_TYPE_Q4_0 ||
        type == GGML_TYPE_Q5_0 || type == GGML_TYPE_IQ4_NL ||
        type == GGML_TYPE_IQ4_XS) {
        ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<
            type, J, fallback, MMQ_Q8_1_METADATA_F32_D4>(
            x, y, sum, k00, meta == nullptr ? y : meta);
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

// Staged write-back. The gfx11 C fragment hands a thread every other column of
// one output row, so the direct store is two bytes wide, scatters sixteen rows
// across a warp and has to round each value in software: disassembled, that
// write-back is 657 instructions per thread. Staging the tile in LDS first lets
// the read side take eight contiguous columns per lane, so the store becomes one
// 16-byte vector and the rounding a packed instruction.
//
// Every exact-tile control has a fixed shape, so this path has no boundary work
// at all: `full_i` and `full_j` must both hold and the row stride must be a
// whole number of 16-byte vectors, which the dense forward launcher already
// requires by tiling the weight rows in 64-row groups.
template <ggml_type type, int J, bool full_i, bool full_j>
static __device__ __forceinline__ void mmq_write_back_bf16_staged(
        const float * sum,
        __hip_bfloat16 * dst,
        int stride,
        int i_max,
        int j_max,
        float * stage) {
    using namespace ggml_cuda_mma;
    using tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int ne = tile_C::ne;
    constexpr int lanes = MMQ_NTHREADS;
    constexpr int lanes_per_row = MMQ_I / 8;
    constexpr int rows_per_pass = lanes / lanes_per_row;
    // One staging pass covers whole 16-row C fragments. The row stride is
    // padded so the two half-warps of a fragment land in different banks.
    constexpr int chunk = J < 64 ? J : 64;
    // A row stride of MMQ_I + 2 words spreads the sixteen rows of a store
    // across sixteen banks, and keeps the staging reads 8-byte aligned.
    constexpr int row_words = MMQ_I + 2;
    static_assert(full_i && full_j, "the staged write-back is exact-tile only");
    static_assert(J % 16 == 0, "the staged write-back walks 16-row C fragments");
    static_assert(MMQ_I % 8 == 0, "eight columns per lane per store");
    static_assert(chunk % 16 == 0 && rows_per_pass % 16 == 0);
    static_assert(chunk % rows_per_pass == 0, "a pass covers whole rows");
    (void)i_max;
    (void)j_max;

    const int lane = threadIdx.y * WARP_SIZE + threadIdx.x;
    // The C fragment is indexed by the warp-local lane, not by the linear
    // thread id: warp W covers columns [W*16, W*16+16) of the staged tile.
    const int fragment_row = threadIdx.x % 16;
    const int fragment_column = threadIdx.x / 16;
    const int i0 = threadIdx.y * tile_C::I;

    for (int base = 0; base < J; base += chunk) {
#pragma unroll
        for (int j0 = 0; j0 < chunk; j0 += 16) {
#pragma unroll
            for (int l = 0; l < ne; ++l) {
                stage[(j0 + fragment_row) * row_words + i0 + 2 * l + fragment_column] =
                    sum[(base / 16 + j0 / 16) * ne + l];
            }
        }
        __syncthreads();
#pragma unroll
        for (int pass = 0; pass < chunk / rows_per_pass; ++pass) {
            const int row = pass * rows_per_pass + lane / lanes_per_row;
            const int column = (lane % lanes_per_row) * 8;
            const float * source = stage + row * row_words + column;
            union {
                __hip_bfloat162 pairs[4];
                uint4 vector;
            } packed;
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                packed.pairs[pair] = __float22bfloat162_rn(
                    make_float2(source[2 * pair], source[2 * pair + 1]));
            }
            // The dense forward launcher tiles the weight rows in 64-row groups,
            // so a row starts on a 128-byte boundary and this is aligned.
            *reinterpret_cast<uint4 *>(
                dst + static_cast<int64_t>(base + row) * stride + column) =
                packed.vector;
        }
        __syncthreads();
    }
}

// Write-back selection: the staged body needs a scratch tile and the plain body
// does not, so the dense forward body picks between them at compile time.
#if defined(MMQ_EPILOGUE_STAGED)
template <ggml_type type, int J, bool full_i, bool full_j>
static __device__ __forceinline__ void mmq_write_back_bf16_impl(
        const float * sum,
        __hip_bfloat16 * dst,
        int stride,
        int i_max,
        int j_max,
        int * scratch) {
    mmq_write_back_bf16_staged<type, J, full_i, full_j>(
        sum, dst, stride, i_max, j_max, reinterpret_cast<float *>(scratch));
}
#else
template <ggml_type type, int J, bool full_i, bool full_j>
static __device__ __forceinline__ void mmq_write_back_bf16_impl(
        const float * sum,
        __hip_bfloat16 * dst,
        int stride,
        int i_max,
        int j_max,
        int *) {
    mmq_write_back_bf16<type, J, full_i, full_j>(sum, dst, stride, i_max, j_max);
}
#endif

template <ggml_type type>
static constexpr __host__ __device__ mmq_q8_1_metadata_layout
mmq_activation_metadata_layout() {
    return type == GGML_TYPE_Q2_K
        ? MMQ_Q8_1_METADATA_F16_D2S6
        : type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K
            ? MMQ_Q8_1_METADATA_F16_D4S4
            : MMQ_Q8_1_METADATA_F32_D4;
}

// Group producer for the two 32-value metadata layouts. The row producer above
// gives a thread four values, folds every 32-value group through three LDS
// permutes, divides twice per group and rounds with roundf. That is ~39 SASS
// instructions per value and leaves the warp waiting on LDS latency it cannot
// hide. This body gives a thread half a group, so the group's amax and sum need
// one lane exchange, the loads and stores are 16-byte vectors and the index
// math happens once. It is the same arithmetic - the two IEEE divisions and the
// roundf stay - so the workspace is bitwise identical, and it is faster
// whenever the activation row set is cache resident, which is where the row
// producer runs at a third of its instruction rate. Past cache it is a few
// percent behind, so the dispatch keeps the row producer for large row sets.
template <ggml_type type>
static __device__ __forceinline__ void quantize_bf16_mmq_q8_1_grouped_body(
        const __hip_bfloat16 * __restrict__ x,
        block_q8_1_mmq * __restrict__ y,
        int64_t rows,
        int64_t rows_padded,
        int64_t k) {
    constexpr mmq_q8_1_metadata_layout metadata_layout =
        mmq_activation_metadata_layout<type>();
    static_assert(
        metadata_layout == MMQ_Q8_1_METADATA_F32_D4 ||
            metadata_layout == MMQ_Q8_1_METADATA_F16_D4S4,
        "the grouped producer serves the two 32-value layouts");
    constexpr int values_per_thread = 16;
    constexpr int values_per_group = 32;
    constexpr int threads_per_group = values_per_group / values_per_thread;
    constexpr int groups_per_block = 4;
    const int64_t row = blockIdx.x;
    const int groups = static_cast<int>(k / values_per_group);
    // Both partners of a group stay active so the exchange is well defined.
    const int slots = (groups * threads_per_group + 1) & ~1;
    for (int slot = threadIdx.x; slot < slots; slot += blockDim.x) {
        const bool active = slot < groups * threads_per_group;
        const int group = slot / threads_per_group;
        const int part = slot % threads_per_group;
        const int64_t offset = active
            ? row * k + static_cast<int64_t>(group) * values_per_group +
                part * values_per_thread
            : 0;

        float values[values_per_thread];
#pragma unroll
        for (int i = 0; i < values_per_thread / 8; ++i) {
            const uint4 quad = reinterpret_cast<const uint4 *>(x + offset)[i];
            const uint32_t raw[4] = {quad.x, quad.y, quad.z, quad.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                // BF16 is the upper half of an FP32, so the conversion is a shift.
                values[8 * i + 2 * j] = __uint_as_float(raw[j] << 16);
                values[8 * i + 2 * j + 1] = __uint_as_float(raw[j] & 0xffff0000u);
            }
        }

        float amax = fabsf(values[0]);
        float sum = values[0];
#pragma unroll
        for (int i = 1; i < values_per_thread; ++i) {
            amax = fmaxf(amax, fabsf(values[i]));
            if constexpr (metadata_layout == MMQ_Q8_1_METADATA_F16_D4S4) {
                sum += values[i];
            }
        }
#pragma unroll
        for (int step = threads_per_group / 2; step > 0; step >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, step, WARP_SIZE));
            if constexpr (metadata_layout == MMQ_Q8_1_METADATA_F16_D4S4) {
                sum += __shfl_xor_sync(0xffffffff, sum, step, WARP_SIZE);
            }
        }
        const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
        const float d_inv = amax == 0.0f ? 0.0f : 127.0f / amax;

        int8_t bytes[values_per_thread];
#pragma unroll
        for (int i = 0; i < values_per_thread; ++i) {
            const int rounded = static_cast<int>(roundf(values[i] * d_inv));
            bytes[i] = static_cast<int8_t>(
                rounded < -128 ? -128 : (rounded > 127 ? 127 : rounded));
        }
        if (!active) {
            continue;
        }
        const int64_t block_k = group / groups_per_block;
        const int lane = (group % groups_per_block) * values_per_group +
            part * values_per_thread;
        block_q8_1_mmq & out = y[block_k * rows_padded + row];
        __builtin_memcpy(out.qs + lane, bytes, values_per_thread);
        if (part == 0) {
            if constexpr (metadata_layout == MMQ_Q8_1_METADATA_F16_D4S4) {
                out.scale_sum_pairs_f16[group % groups_per_block] =
                    make_half2(d, sum);
            } else {
                out.scales_f32[group % groups_per_block] = d;
            }
        }
    }
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

// Window load for a packed row whose contraction is not a whole number of
// MMQ_ITER_K stages. Offsets and lengths are counted in quantized values. Only
// the loaders that address rows in blocks small enough to express a half stage
// have a window form.
template <ggml_type type, int J, bool fallback, bool table_decode = false>
static __device__ __forceinline__ void mmq_load_window(
        const char * x, int * tile, int value_offset, int value_end, int i_max) {
    if constexpr (type == GGML_TYPE_Q8_0) {
        ggml_cuda_mmq_load_tiles_q8_0<type, J, fallback>(
            x, tile, value_offset / QK8_0, i_max, value_end / QK8_0);
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        ggml_cuda_mmq_load_tiles_q4_0<type, J, fallback>(
            x, tile, value_offset / QK4_0, i_max, value_end / QK4_0);
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        ggml_cuda_mmq_load_tiles_q5_0<type, J, fallback, table_decode>(
            x, tile, value_offset / QK5_0, i_max, value_end / QK5_0);
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        ggml_cuda_mmq_load_tiles_iq4_nl<type, J, fallback>(
            x, tile, value_offset / QK4_NL, i_max, value_end / QK4_NL);
    } else {
        static_assert(type == GGML_TYPE_Q2_0, "window loads need a fine-grained loader");
        ggml_cuda_mmq_load_tiles_q2_0<type, J, fallback>(
            x, tile, value_offset / QK2_0, i_max, value_end / QK2_0);
    }
}

template <
    ggml_type type,
    int J,
    int fixed_blocks_per_weight_row = 0,
    bool full_i = false,
    bool full_j = false,
    int tail_values = 0,
    bool table_decode = false>
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

    // A contraction that is not a whole number of MMQ_ITER_K stages ends in a
    // stage that holds `tail_values` values (128, i.e. one vector dot call). Its
    // packed window is loaded from the end of the row, so the load stays inside
    // the row, and only that window's upper half is consumed.
    static_assert(
        tail_values == 0 || tail_values == 128,
        "a tail stage holds one 128-value vector dot call");
    static_assert(
        tail_values == 0 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0 ||
            type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q2_0 ||
            type == GGML_TYPE_IQ4_NL,
        "a tail stage needs a loader that can address a half stage");
    static_assert(
        tail_values == 0 || fixed_blocks_per_weight_row > 0,
        "a tail stage requires the exact contraction length");
    constexpr int tail_window_offset =
        tail_values == 0 ? 0 : (fixed_blocks_per_weight_row - 2) * MMQ_ITER_K + tail_values;
    constexpr int row_values = tail_values == 0
        ? 0
        : (fixed_blocks_per_weight_row - 1) * MMQ_ITER_K + tail_values;
    extern __shared__ int shared[];
#if defined(MMQ_FRAGMENT_ACT)
    // Only the four per-group scales of each token stay in LDS. The quants are
    // read as fragments straight from the workspace.
    static_assert(tail_values == 0, "the fragment path owns the whole stage");
    constexpr int activation_ints = J * MMQ_META_STRIDE;
#else
    constexpr int activation_ints = GGML_PAD(J * MMQ_TILE_Y_K, MMQ_NTHREADS);
#endif
    int * tile_y = shared + J;
    int * tile_x = tile_y + activation_ints;

    float sum[J * MMQ_I / MMQ_NTHREADS] = {0.0f};
    constexpr int q8_block_ints = sizeof(block_q8_1_mmq) / sizeof(int);

    // The table decode keeps the 256 expansions of one payload byte in LDS
    // directly behind the weight tile, so the loader replaces the spread and
    // the level offset with one lookup.
    if constexpr (table_decode) {
        static_assert(
            type == GGML_TYPE_Q2_0 || type == GGML_TYPE_Q5_0,
            "the level table serves the two-bit and the nibble-plus-bit decodes");
        int * levels = tile_x + MMQ_I * ggml_cuda_mmq_get_sram_stride<type, J, false>();
        if constexpr (type == GGML_TYPE_Q2_0) {
            for (int entry = threadIdx.y * WARP_SIZE + threadIdx.x; entry < 256;
                 entry += MMQ_NTHREADS) {
                levels[entry] = q2_0_spread_levels(entry);
            }
        } else {
            // Sixteen fifth-bit spreads, indexed by the four bits of a group.
            for (int entry = threadIdx.y * WARP_SIZE + threadIdx.x; entry < 16;
                 entry += MMQ_NTHREADS) {
                levels[entry] = q5_0_spread_bits(entry);
            }
        }
        __syncthreads();
    }
    const int kernel_blocks_per_weight_row = fixed_blocks_per_weight_row > 0
        ? fixed_blocks_per_weight_row
        : blocks_per_weight_row;

#if defined(MMQ_HALF_STAGE)
    static_assert(
        mmq_sram_layout(type) == GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0,
        "the half stage covers the one-byte-per-value level layouts");
    static_assert(tail_values == 0, "a half stage does not implement the tail window");
    const int stage_count = 2 * kernel_blocks_per_weight_row;
    for (int stage = 0; stage < stage_count; ++stage) {
        // Half a weight block per stage: the loader's row layout already holds
        // the two 128-value halves side by side, so a stage is one of them.
        mmq_load_target<type, J, !full_i, table_decode>(
            weights,
            tile_x,
            2 * (tile_i * MMQ_I * kernel_blocks_per_weight_row + (stage >> 1)) +
                (stage & 1),
            i_max,
            2 * kernel_blocks_per_weight_row);
#pragma unroll
        for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const int src =
                (stage * nrows_activation_padded + tile_j * J) * q8_block_ints + l;
            tile_y[l] = activations[src];
        }
        __syncthreads();
        mmq_vec_dot_target<type, J>(tile_x, tile_y, sum, 0);
        __syncthreads();
    }
#else
    for (int kb = 0; kb < kernel_blocks_per_weight_row; ++kb) {
        const bool tail = tail_values > 0 && kb + 1 == kernel_blocks_per_weight_row;
        if constexpr (tail_values > 0) {
            // Every stage of a tailed row is addressed by its value offset, so
            // the stride is the true row length rather than the stage count.
            mmq_load_window<type, J, !full_i, table_decode>(
                weights,
                tile_x,
                tile_i * MMQ_I * row_values +
                    (tail ? tail_window_offset : kb * MMQ_ITER_K),
                row_values,
                i_max);
        } else {
            const int weight_block_offset =
                tile_i * MMQ_I * kernel_blocks_per_weight_row + kb;
            mmq_load_target<type, J, !full_i, table_decode>(
                weights,
                tile_x,
                weight_block_offset,
                i_max,
                kernel_blocks_per_weight_row);
        }

#if defined(MMQ_FRAGMENT_ACT)
        // One stage's activation is one Q8_1 block per token, and the workspace
        // lays those blocks out exactly as the tile did, so the dot addresses its
        // fragments in global memory with the same arithmetic. Each token's
        // sixteen-byte metadata header is copied onto the LDS strip in one
        // sixteen-byte load ahead of the barrier that publishes it.
        const int * activation_stage =
            activations + ((2 * kb) * nrows_activation_padded + tile_j * J) * q8_block_ints;
        if (threadIdx.y * WARP_SIZE + threadIdx.x < J) {
            const int token = threadIdx.y * WARP_SIZE + threadIdx.x;
            __builtin_memcpy(
                tile_y + token * MMQ_META_STRIDE,
                activation_stage + token * q8_block_ints,
                16);
        }
#else
#pragma unroll
        for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
            const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
            const int src =
                ((2 * kb) * nrows_activation_padded + tile_j * J) * q8_block_ints + l;
            tile_y[l] = activations[src];
        }
        const int * activation_stage = tile_y;
#endif
        __syncthreads();
        if constexpr (tail_values > 0) {
            if (!tail) {
                mmq_vec_dot_target<type, J>(
                    tile_x, activation_stage, sum, 0, tile_y);
            }
        } else {
            mmq_vec_dot_target<type, J>(tile_x, activation_stage, sum, 0, tile_y);
        }
        __syncthreads();

        if constexpr (tail_values > 0) {
            if (!tail) {
#pragma unroll
                for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
                    const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                    const int src =
                        ((2 * kb + 1) * nrows_activation_padded + tile_j * J) *
                            q8_block_ints +
                        l;
                    tile_y[l] = activations[src];
                }
                __syncthreads();
            }
        } else {
#if defined(MMQ_FRAGMENT_ACT)
            // Advance to the second half of the stage and refresh the metadata
            // strip before the barrier that publishes it to the whole block.
            activation_stage += nrows_activation_padded * q8_block_ints;
            if (threadIdx.y * WARP_SIZE + threadIdx.x < J) {
                const int token = threadIdx.y * WARP_SIZE + threadIdx.x;
                __builtin_memcpy(
                    tile_y + token * MMQ_META_STRIDE,
                    activation_stage + token * q8_block_ints,
                    16);
            }
#else
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
                const int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;
                const int src =
                    ((2 * kb + 1) * nrows_activation_padded + tile_j * J) * q8_block_ints + l;
                tile_y[l] = activations[src];
            }
#endif
            __syncthreads();
        }
        mmq_vec_dot_target<type, J>(
            tile_x, activation_stage, sum, MMQ_TILE_NE_K, tile_y);
        __syncthreads();
    }
#endif

    mmq_write_back_bf16_impl<type, J, full_i, full_j>(
        sum,
        dst + tile_j * J * nrows_weight + tile_i * MMQ_I,
        nrows_weight,
        i_max,
        j_max,
        tile_y);
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

#if defined(MMQ_PREFETCH_ACT)
// Register staging for the second activation plane: the global loads are issued
// before the first dot and only stored to LDS after it, which overlaps their
// latency with the WMMA work of the same k block.
template <int J>
static constexpr int grouped_activation_plane_regs() {
    return (J * MMQ_TILE_Y_K + MMQ_NTHREADS - 1) / MMQ_NTHREADS;
}

template <int J, int N>
static __device__ __forceinline__ void load_grouped_activation_plane_regs(
        const int * __restrict__ activation,
        int (&regs)[N],
        const int lane,
        const int valid_ints) {
    constexpr int tile_ints = J * MMQ_TILE_Y_K;
#pragma unroll
    for (int index = 0; index < N; ++index) {
        const int l = lane + index * MMQ_NTHREADS;
        // never read past the quantized rows of the bank
        regs[index] = l < tile_ints && l < valid_ints ? activation[l] : 0;
    }
}

template <int J, int N>
static __device__ __forceinline__ void store_grouped_activation_plane_regs(
        int * __restrict__ tile_y,
        const int (&regs)[N],
        const int lane) {
    constexpr int tile_ints = J * MMQ_TILE_Y_K;
#pragma unroll
    for (int index = 0; index < N; ++index) {
        const int l = lane + index * MMQ_NTHREADS;
        if (l < tile_ints) {
            tile_y[l] = regs[index];
        }
    }
}
#endif // MMQ_PREFETCH_ACT

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

// One activation plane of a grouped stage. A control with `MMQ_FRAGMENT_ACT`
// keeps only the sixteen-byte metadata header of each token in LDS and lets the
// vector dot read its B fragments straight from the workspace: the header is the
// first sixteen bytes of the token's row record and the 128 payload bytes the
// fragment loader addresses sit behind it.
template <int J>
static __device__ __forceinline__ void grouped_mmq_load_metadata_strip(
        const int * __restrict__ activation_k,
        int * __restrict__ tile_y) {
    constexpr int q8_block_ints = sizeof(block_q8_1_mmq) / sizeof(int);
    const int lane = threadIdx.y * WARP_SIZE + threadIdx.x;
    if (lane < J) {
        __builtin_memcpy(
            tile_y + lane * MMQ_META_STRIDE,
            activation_k + lane * q8_block_ints,
            16);
    }
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
#if !defined(MMQ_COMPACT_TILE)
    mmq_load_target<type, J, !fixed_shape>(
        expert_weights,
        tile_x,
        weight_block_offset,
        i_max,
        kernel_blocks_per_weight_row);
#endif
#if defined(MMQ_PREFETCH_ACT) && !defined(MMQ_FRAGMENT_ACT)
    constexpr int plane_regs = grouped_activation_plane_regs<J>();
    int plane_buffer[plane_regs];
    {
        const int lane = threadIdx.y * WARP_SIZE + threadIdx.x;
        load_grouped_activation_plane_regs<J>(
            activation_k + activation_plane_stride,
            plane_buffer,
            lane,
            valid_activation_ints);
    }
#endif

#if defined(MMQ_FRAGMENT_ACT)
    grouped_mmq_load_metadata_strip<J>(activation_k, tile_y);
#else
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
#endif
#if defined(MMQ_COMPACT_TILE)
    mmq_load_target<type, J, !fixed_shape>(
        expert_weights, tile_x, weight_block_offset, i_max,
        kernel_blocks_per_weight_row, 0);
#endif
    __syncthreads();
#if defined(MMQ_FRAGMENT_ACT)
    mmq_vec_dot_target<type, J, !fixed_shape>(
        tile_x, activation_k, sum, 0, tile_y);
#else
    mmq_vec_dot_target<type, J, !fixed_shape>(tile_x, tile_y, sum, 0);
#endif
    __syncthreads();

#if defined(MMQ_FRAGMENT_ACT)
    grouped_mmq_load_metadata_strip<J>(
        activation_k + activation_plane_stride, tile_y);
#elif defined(MMQ_PREFETCH_ACT)
    {
        const int lane = threadIdx.y * WARP_SIZE + threadIdx.x;
        store_grouped_activation_plane_regs<J>(tile_y, plane_buffer, lane);
    }
#else
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
#endif
#if defined(MMQ_COMPACT_TILE)
    mmq_load_target<type, J, !fixed_shape>(
        expert_weights, tile_x, weight_block_offset, i_max,
        kernel_blocks_per_weight_row, 1);
#endif
    __syncthreads();
#if defined(MMQ_COMPACT_TILE)
    mmq_vec_dot_target<type, J, !fixed_shape>(tile_x, tile_y, sum, 0);
#elif defined(MMQ_FRAGMENT_ACT)
    mmq_vec_dot_target<type, J, !fixed_shape>(
        tile_x, activation_k + activation_plane_stride, sum, MMQ_TILE_NE_K, tile_y);
#else
    mmq_vec_dot_target<type, J, !fixed_shape>(
        tile_x, tile_y, sum, MMQ_TILE_NE_K);
#endif
    __syncthreads();
}

template <
    ggml_type type,
    int J,
    int fixed_nrows_weight,
    int fixed_blocks_per_weight_row,
    bool full_j,
    bool rolled_q2_k = false,
    int k_tail_values = 0,
    bool table_decode = false>
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
    // A contraction that is not a whole number of MMQ_ITER_K stages makes every
    // stage of such a row address its packed window by value offset, and gives
    // the row a 128-value tail stage that only consumes its upper half.
    constexpr int k_row_values = k_tail_values == 0
        ? 0
        : (fixed_blocks_per_weight_row - 1) * MMQ_ITER_K + k_tail_values;
    constexpr int k_tail_window = k_tail_values == 0
        ? 0
        : (fixed_blocks_per_weight_row - 2) * MMQ_ITER_K + k_tail_values;
    static_assert(
        k_tail_values == 0 || fixed_shape,
        "a contraction tail needs the exact weights geometry");
    float sum[J * MMQ_I / MMQ_NTHREADS] = {0.0f};

    if constexpr (fixed_blocks_per_weight_row == 2 && k_tail_values == 0) {
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
#if defined(MMQ_WEIGHT_PIPELINE)
        int weight_block_offset = tile_i * MMQ_I * kernel_blocks_per_weight_row;
        // Register pipeline over the k blocks of one row tile: the packed
        // payload of the block `MMQ_WEIGHT_PIPELINE` ahead is read into
        // registers while the current block's decode, shared stores, barriers
        // and matrix work run, so that block's load latency overlaps several
        // stages of work instead of one. The decode, the shared stores and the
        // summation order are unchanged, and the stage index is a constant
        // inside the unrolled inner loop, so the payload array stays in
        // registers.
        constexpr int pipeline_stages = MMQ_WEIGHT_PIPELINE;
        constexpr int payload_slots = mmq_iq2_xxs_payload_slots<type, J, !fixed_shape>();
        iq2_xxs_stage_payload payload[pipeline_stages * payload_slots];
#pragma unroll
        for (int stage = 0; stage < pipeline_stages; ++stage) {
            if (stage < kernel_blocks_per_weight_row) {
                ggml_cuda_mmq_load_payload_iq2_xxs<type, J, !fixed_shape>(
                    expert_weights,
                    weight_block_offset + stage,
                    i_max,
                    kernel_blocks_per_weight_row,
                    payload + stage * payload_slots);
            }
        }
        // The first dot needs its tile, so block zero is staged before the loop.
        ggml_cuda_mmq_store_payload_iq2_xxs<type, J, !fixed_shape>(tile_x, payload);
#pragma unroll 1
        for (int kb_base = 0; kb_base < kernel_blocks_per_weight_row; kb_base += pipeline_stages) {
#pragma unroll
            for (int stage = 0; stage < pipeline_stages; ++stage) {
                const int kb = kb_base + stage;
                if (kb >= kernel_blocks_per_weight_row) {
                    break;
                }
                if (kb + pipeline_stages < kernel_blocks_per_weight_row) {
                    ggml_cuda_mmq_load_payload_iq2_xxs<type, J, !fixed_shape>(
                        expert_weights,
                        weight_block_offset + pipeline_stages,
                        i_max,
                        kernel_blocks_per_weight_row,
                        payload + stage * payload_slots);
                }

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
    #if defined(MMQ_COMPACT_TILE)
                mmq_load_target<type, J, !fixed_shape>(
                    expert_weights,
                    tile_x,
                    weight_block_offset,
                    i_max,
                    kernel_blocks_per_weight_row,
                    1);
    #endif
                __syncthreads();
    #if defined(MMQ_COMPACT_TILE)
                mmq_vec_dot_target<type, J, !fixed_shape, rolled_q2_k>(
                    tile_x, tile_y, sum, 0);
    #else
                mmq_vec_dot_target<type, J, !fixed_shape, rolled_q2_k>(
                    tile_x, tile_y, sum, MMQ_TILE_NE_K);
    #endif
                __syncthreads();

                if (kb + 1 < kernel_blocks_per_weight_row) {
                    ggml_cuda_mmq_store_payload_iq2_xxs<type, J, !fixed_shape>(
                        tile_x, payload + ((stage + 1) % pipeline_stages) * payload_slots);
                }
                ++weight_block_offset;
                activation_k += 2 * activation_half_stride;
            }
            }
#else
        int weight_block_offset = tile_i * MMQ_I * kernel_blocks_per_weight_row;
        const int full_stages = k_tail_values == 0
            ? kernel_blocks_per_weight_row
            : kernel_blocks_per_weight_row - 1;
#pragma unroll 1
        for (int kb = 0; kb < full_stages; ++kb) {
            if constexpr (k_tail_values > 0) {
                mmq_load_window<type, J, !fixed_shape, table_decode>(
                    expert_weights,
                    tile_x,
                    tile_i * MMQ_I * k_row_values + kb * MMQ_ITER_K,
                    k_row_values,
                    i_max);
            } else {
                mmq_load_target<type, J, !fixed_shape, table_decode>(
                    expert_weights,
                    tile_x,
                    weight_block_offset,
                    i_max,
                    kernel_blocks_per_weight_row
#if defined(MMQ_COMPACT_TILE)
                    ,
                    0
#endif
                );
            }

#if defined(MMQ_FRAGMENT_ACT)
            grouped_mmq_load_metadata_strip<J>(activation_k, tile_y);
#else
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
#endif
            __syncthreads();
#if defined(MMQ_FRAGMENT_ACT)
            mmq_vec_dot_target<type, J, !fixed_shape, rolled_q2_k>(
                tile_x, activation_k, sum, 0, tile_y);
#else
            mmq_vec_dot_target<type, J, !fixed_shape, rolled_q2_k>(
                tile_x, tile_y, sum, 0);
#endif
            __syncthreads();

#if defined(MMQ_FRAGMENT_ACT)
            grouped_mmq_load_metadata_strip<J>(
                activation_k + activation_half_stride, tile_y);
#else
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
#endif
#if defined(MMQ_COMPACT_TILE)
            mmq_load_target<type, J, !fixed_shape, table_decode>(
                expert_weights,
                tile_x,
                weight_block_offset,
                i_max,
                kernel_blocks_per_weight_row,
                1);
#endif
            __syncthreads();
#if defined(MMQ_COMPACT_TILE)
            mmq_vec_dot_target<type, J, !fixed_shape, rolled_q2_k>(
                tile_x, tile_y, sum, 0);
#elif defined(MMQ_FRAGMENT_ACT)
            mmq_vec_dot_target<type, J, !fixed_shape, rolled_q2_k>(
                tile_x, activation_k + activation_half_stride, sum,
                MMQ_TILE_NE_K, tile_y);
#else
            mmq_vec_dot_target<type, J, !fixed_shape, rolled_q2_k>(
                tile_x, tile_y, sum, MMQ_TILE_NE_K);
#endif
            __syncthreads();

            ++weight_block_offset;
            activation_k += 2 * activation_half_stride;
        }
#endif
        if constexpr (k_tail_values > 0) {
            if (kernel_blocks_per_weight_row > 0) {
                // The tail stage: the packed window sits at the end of the row
                // and holds the last 128 values in its upper half, which is the
                // half the MMQ_TILE_NE_K vector dot reads.
                mmq_load_window<type, J, !fixed_shape, table_decode>(
                    expert_weights,
                    tile_x,
                    tile_i * MMQ_I * k_row_values + k_tail_window,
                    k_row_values,
                    i_max);
#if defined(MMQ_FRAGMENT_ACT)
                grouped_mmq_load_metadata_strip<J>(activation_k, tile_y);
#else
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
#endif
                __syncthreads();
#if defined(MMQ_FRAGMENT_ACT)
                mmq_vec_dot_target<type, J, !fixed_shape, rolled_q2_k>(
                    tile_x, activation_k, sum, MMQ_TILE_NE_K, tile_y);
#else
                mmq_vec_dot_target<type, J, !fixed_shape, rolled_q2_k>(
                    tile_x, tile_y, sum, MMQ_TILE_NE_K);
#endif
                __syncthreads();
            }
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
    int mixed_j32_rows_b = 0,
    int k_tail_values = 0,
    bool mixed_j16_tails = false,
    bool table_decode = false>
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
        (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_Q4_K ||
         type == GGML_TYPE_Q2_0) &&
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
            if constexpr (mixed_j16_tails) {
                if (tail_rows <= MMQ_J_MIN) {
                    grouped_mmq_row_tile<
                        type, MMQ_J_MIN, fixed_nrows_weight,
                        fixed_blocks_per_weight_row, false, rolled_q2_k,
                        k_tail_values, table_decode>(
                            expert_weights, activations, dst, tile_x, tile_y,
                            tile_i, row_start, row_end, nrows_weight,
                            nrows_activation, blocks_per_weight_row);
                    return;
                }
            }
            grouped_mmq_row_tile<
                type, MMQ_J_TINY, fixed_nrows_weight, fixed_blocks_per_weight_row,
                false, rolled_q2_k, k_tail_values, table_decode>(
                    expert_weights, activations, dst, tile_x, tile_y, tile_i,
                    row_start, row_end, nrows_weight, nrows_activation,
                    blocks_per_weight_row);
        } else {
            grouped_mmq_row_tile<
                type, J, fixed_nrows_weight, fixed_blocks_per_weight_row,
                false, rolled_q2_k, k_tail_values, table_decode>(
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
                false, rolled_q2_k, k_tail_values, table_decode>(
                    expert_weights, activations, dst, tile_x, tile_y, tile_i,
                    row_start, row_end, nrows_weight, nrows_activation,
                    blocks_per_weight_row);
        } else {
            grouped_mmq_row_tile<
                type, J, fixed_nrows_weight, fixed_blocks_per_weight_row,
                false, rolled_q2_k, k_tail_values, table_decode>(
                    expert_weights, activations, dst, tile_x, tile_y, tile_i,
                    row_start, row_end, nrows_weight, nrows_activation,
                    blocks_per_weight_row);
        }
    } else {
        grouped_mmq_row_tile<
            type, J, fixed_nrows_weight, fixed_blocks_per_weight_row,
            false, rolled_q2_k, k_tail_values, table_decode>(
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
    int mixed_j32_rows_b = 0,
    int k_tail_values = 0,
    bool mixed_j16_tails = false,
    bool table_decode = false>
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
#if defined(MMQ_FRAGMENT_ACT)
    // Only the four per-group scales of each token stay in LDS, so the weight
    // tile starts directly behind the metadata strip.
    constexpr int activation_ints = J * MMQ_META_STRIDE;
#else
    constexpr int activation_ints = GGML_PAD(J * MMQ_TILE_Y_K, MMQ_NTHREADS);
#endif
    int * tile_y = shared + J;
    int * tile_x = tile_y + activation_ints;

    // The table decode keeps the 256 expansions of one payload byte in LDS
    // directly behind the weight tile, so the loader replaces the spread and
    // the level offset with one lookup.
    if constexpr (table_decode) {
        static_assert(
            type == GGML_TYPE_Q2_0,
            "the grouped level table serves the two-bit decode");
        int * levels = tile_x + MMQ_I * ggml_cuda_mmq_get_sram_stride<type, J, false>();
        for (int entry = threadIdx.y * WARP_SIZE + threadIdx.x; entry < 256;
             entry += MMQ_NTHREADS) {
            levels[entry] = q2_0_spread_levels(entry);
        }
        __syncthreads();
    }

    int row_start = row_begin;
    for (; row_start + J <= row_end; row_start += J) {
        grouped_mmq_row_tile<
            type, J, fixed_nrows_weight, fixed_blocks_per_weight_row,
            true, rolled_q2_k, k_tail_values, table_decode>(
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
            mixed_j32_rows_a, mixed_j32_rows_b, k_tail_values,
            mixed_j16_tails, table_decode>(
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
    // The cumulative scan covers 256 groups per pass, and a family with 512
    // physical experts can route to more entries than that, so the pass runs
    // once per chunk of 256 entries with the finished count as its base.
    __shared__ int cumulative_tasks[256];
    const int slot = threadIdx.x;
    int task_base = 0;

    for (int chunk = 0; chunk < num_groups; chunk += 256) {
        const int group = chunk + slot;
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
        cumulative_tasks[slot] = group_tasks;
        __syncthreads();

#pragma unroll
        for (int offset = 1; offset < 256; offset <<= 1) {
            const int add = slot >= offset ? cumulative_tasks[slot - offset] : 0;
            __syncthreads();
            cumulative_tasks[slot] += add;
            __syncthreads();
        }

        if (group < num_groups) {
            const int task_begin = task_base +
                (slot == 0 ? 0 : cumulative_tasks[slot - 1]);
            for (int task = 0; task < group_tasks; ++task) {
                const int task_index = task_begin + task;
                const int task_row_start = row_begin + task * row_tile;
                task_experts[task_index] = expert;
                task_row_starts[task_index] = task_row_start;
                task_row_ends[task_index] =
                    min(task_row_start + row_tile, row_end);
            }
        }
        task_base += cumulative_tasks[255];
        __syncthreads();
    }
    if (slot == 0) {
        task_count[0] = task_base;
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
