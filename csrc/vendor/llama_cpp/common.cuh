// Minimal llama.cpp compatibility surface for the gfx1151 MMQ operators.
// Derived from ggml/src/ggml-common.h, ggml/src/ggml-cuda/common.cuh, and ggml/src/ggml-cuda/vendors/hip.h.
// Only what ROCm on RDNA3.5 needs is kept: the quant block layouts, the index helpers used by the
// tile loaders, and the warp/shuffle surface they rely on. The architecture switches, the CUDA
// compatibility aliases and the unused helper functions were removed.

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#define WARP_SIZE 32

#define __shfl_sync(mask, var, lane, width) __shfl((var), (lane), (width))
#define __shfl_xor_sync(mask, var, lane_mask, width) __shfl_xor((var), (lane_mask), (width))

#define NO_DEVICE_CODE __builtin_trap()
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

// GGML quantization identifiers used by the checkpoint.
enum ggml_type : int32_t {
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_S = 22,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_Q2_0 = 42,
};

// Sixteen-level codebook of IQ4_NL, in the order its nibbles index it.
static const __device__ int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

#define QK_K 256
#define K_SCALE_SIZE 12
#define QK4_0 32
#define QK4_NL 32
#define QK5_0 32
#define QK8_0 32
#define QK8_1 32
#define QK2_0 64

#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
#define QR4_NL 2
#define QI4_NL (QK4_NL / (4 * QR4_NL))
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
// Q2_0 keeps four consecutive codes per payload byte, so a 32-value group is
// addressed by two payload words rather than by four.
#define QI2_0 (QK2_0 / 32)
#define QR2_K 4
#define QR3_K 4
#define QI3_K (QK_K / (4 * QR3_K))
#define QR4_K 2
#define QR5_K 2
#define QI5_K (QK_K / (4 * QR5_K))
#define QR6_K 2
#define QI6_K (QK_K / (4 * QR6_K))
#define QR2_XXS 4
#define QR2_S 4

struct block_q4_0 {
    half d;
    uint8_t qs[QK4_0 / 2];
};
static_assert(sizeof(block_q4_0) == 18, "wrong q4_0 block size");

struct block_iq4_nl {
    half d;
    uint8_t qs[QK4_NL / 2];
};
static_assert(sizeof(block_iq4_nl) == 18, "wrong iq4_nl block size");

struct block_q5_0 {
    half d;
    uint8_t qh[QK5_0 / 8];
    uint8_t qs[QK5_0 / 2];
};
static_assert(sizeof(block_q5_0) == 22, "wrong q5_0 block size");

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};
static_assert(sizeof(block_q8_0) == 34, "wrong q8_0 block size");

typedef struct {
    half d;
    uint8_t qs[QK2_0 / 4];
} block_q2_0;
static_assert(sizeof(block_q2_0) == 18, "wrong q2_0 block size");

struct block_q8_1 {
    half2 ds;
    int8_t qs[QK8_1];
};
static_assert(sizeof(block_q8_1) == 36, "wrong q8_1 block size");

struct block_q2_K {
    uint8_t scales[QK_K / 16];
    uint8_t qs[QK_K / 4];
    union {
        struct {
            half d;
            half dmin;
        };
        half2 dm;
    };
};
static_assert(sizeof(block_q2_K) == 84, "wrong q2_K block size");

struct block_q3_K {
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    uint8_t scales[12];
    half d;
};
static_assert(sizeof(block_q3_K) == 110, "wrong q3_K block size");

struct block_q4_K {
    union {
        struct {
            half d;
            half dmin;
        };
        half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 144, "wrong q4_K block size");

struct block_q5_K {
    union {
        struct {
            half d;
            half dmin;
        };
        half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q5_K) == 176, "wrong q5_K block size");

struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    half d;
};
static_assert(sizeof(block_q6_K) == 210, "wrong q6_K block size");

struct block_iq2_xxs {
    half d;
    uint16_t qs[QK_K / 8];
};
static_assert(sizeof(block_iq2_xxs) == 66, "wrong iq2_xxs block size");

struct block_iq2_s {
    half d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t scales[QK_K / 32];
};
static_assert(sizeof(block_iq2_s) == 82, "wrong iq2_s block size");

static constexpr __host__ __device__ int ggml_cuda_get_physical_warp_size() {
    return 32;
}

template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(
        void * __restrict__ dst, const void * __restrict__ src) {
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;
    static_assert(alignment == 0 || nbytes % alignment == 0, "bad alignment");
#pragma unroll
    for (int i = 0; i < nbytes / nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            static_cast<char *>(dst)[i] = static_cast<const char *>(src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            static_cast<short *>(dst)[i] = static_cast<const short *>(src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            static_cast<int *>(dst)[i] = static_cast<const int *>(src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            static_cast<int2 *>(dst)[i] = static_cast<const int2 *>(src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            static_cast<int4 *>(dst)[i] = static_cast<const int4 *>(src)[i];
        }
    }
}

using int8x4_t = int8_t __attribute__((ext_vector_type(4)));
using uint8x4_t = uint8_t __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ int __vsubss4(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t &>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t &>(b);
    const int8x4_t vc = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int &>(vc);
}

static __device__ __forceinline__ int __vsub4(const int a, const int b) {
    return __vsubss4(a, b);
}

static __device__ __forceinline__ unsigned int __vcmpne4(unsigned int a, unsigned int b) {
    const uint8x4_t & va = reinterpret_cast<const uint8x4_t &>(a);
    const uint8x4_t & vb = reinterpret_cast<const uint8x4_t &>(b);
    unsigned int c;
    uint8x4_t & vc = reinterpret_cast<uint8x4_t &>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0x00 : 0xff;
    }
    return c;
}

static __device__ __forceinline__ int get_int_b2(const void * x, const int i32) {
    const uint16_t * x16 = static_cast<const uint16_t *>(x);
    int32_t tmp;
    __builtin_memcpy(&tmp, x16 + 2 * i32, sizeof(tmp));
    return tmp;
}

static __device__ __forceinline__ int get_int_b4(const void * x, const int i32) {
    int32_t tmp;
    __builtin_memcpy(&tmp, static_cast<const int32_t *>(x) + i32, sizeof(tmp));
    return tmp;
}

static __device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t value) {
    const uint32_t parity = __popc(value) & 1;
    const uint32_t signs = value ^ (parity << 7);
    return signs * 0x01010101U;
}
