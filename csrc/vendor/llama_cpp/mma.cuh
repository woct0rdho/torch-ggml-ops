// Only the int8 WMMA paths used by the gfx1151 MMQ forward kernels are kept. The
// fp16/bf16/fp4 variants and every non-ROCm architecture were removed. The
// remaining code is exercised by the HIP control bundle.

#pragma once

// A, B, and C are represented using the same fundamental data type: a row-major matrix with I rows and J columns.
// Note that J is measured in physical 32 bit elements instead of logical elements.
// The methods get_i and get_j can be used to get the physical 32 bit index of the lth element of a thread within a tile.
// All matrix tiles have ne physical 32 bit elements per warp.
//
// All pointers for load_ldmatrix must be to shared memory and aligned to 16 bytes.

#include "common.cuh"

namespace ggml_cuda_mma {

    // Some architectures perform multiple matrix multiplications per warp in parallel, effectively splitting the
    // warp into subgroups that each perform a single mma instruction. In those cases the data can be split in
    // different ways across the warp. RDNA3 uses the mirrored layout for the A and B matrices.
    enum data_layout {
        // By default the data uses the I direction as its major dimension and the J direction as its minor dimension.
        // For the A/C matrices this means I major == row major, J major == column major.
        // For the B matrix this means I major == column major, J major == row major.
        // MIRRORED == Each data value is held exactly once per thread subgroup.
        DATA_LAYOUT_I_MAJOR           =  0, // int and float matrix A&B
        DATA_LAYOUT_J_MAJOR           = 10, // int and float matrix C for RDNA3
        DATA_LAYOUT_I_MAJOR_MIRRORED  = 20, // matrix A&B for RDNA3
    };

    static constexpr __device__ data_layout get_input_data_layout() {
        return DATA_LAYOUT_I_MAJOR_MIRRORED;
    }

    template <int I_, int J_, typename T, data_layout ds_=DATA_LAYOUT_I_MAJOR>
    struct tile {};

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I == 16 && J == 16) return true;
            if (I == 16 && J == 8) return true;
            if (I == 16 && J == 4) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (supported()) {
                return threadIdx.x % 16;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 16 && J == 16) {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>) {
                    // matrix C
                    return 2 * l + (threadIdx.x / 16);
                } else {
                    // matrix A&B
                    return l;
                }
            } else if constexpr (I == 16 && J == 8) {
                return ne * (threadIdx.x / 16) + l;
            } else if constexpr (I == 16 && J == 4) {
                return ne * (threadIdx.x / 16) + l;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
    };

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_J_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_J_MAJOR;

        static constexpr int ne = tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::ne;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_j(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_i(l);
        }
    };

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;

        // RDNA3
        static constexpr int         ne = I * J / 32 * 2;

        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I == 16 && J == 16) return true;
            if (I == 16 && J == 8)  return true;
            if (I == 16 && J == 4)  return true;
            if (I == 32 && J == 8)  return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 16) {
                return threadIdx.x % 16;
            } else if constexpr (I == 32) {
                return (threadIdx.x % 16) * 2 + l / (ne/2);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 16) {
                return l;
            } else if constexpr (I == 32) {
                return l % (ne/2);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
    };

    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 4, T, dl> & t, const T * __restrict__ xs0, const int stride) {
        static_assert(dl == DATA_LAYOUT_I_MAJOR_MIRRORED, "bad data layout");
        static_assert(sizeof(t.x) == 16, "bad ne");
        ggml_cuda_memcpy_1<8>(t.x + 0, xs0 + t.get_i(0)*stride + 0);
        ggml_cuda_memcpy_1<8>(t.x + 2, xs0 + t.get_i(0)*stride + 2);
    }

    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T, dl> & t, const T * __restrict__ xs0, const int stride) {
        static_assert(dl == DATA_LAYOUT_I_MAJOR_MIRRORED, "bad data layout");
        static_assert(sizeof(t.x) == 32, "bad ne");
        ggml_cuda_memcpy_1<16>(t.x + 0, xs0 + t.get_i(0)*stride + 0);
        ggml_cuda_memcpy_1<16>(t.x + 4, xs0 + t.get_i(0)*stride + 4);
    }

    template <data_layout dl_d, data_layout dl_ab>
    static __device__ __forceinline__ void mma(
            tile<16, 16, int, dl_d> & D, const tile<16, 8, int, dl_ab> & A, const tile<16, 8, int, dl_ab> & B) {
        using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
        int32x8_t * acc = (int32x8_t *) D.x;
        using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
        int32x4_t * a_vec = (int32x4_t *) A.x;
        int32x4_t * b_vec = (int32x4_t *) B.x;
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[0], true, b_vec[0], acc[0], true);
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[1], true, b_vec[1], acc[0], true);
    }

    template <data_layout dl_d, data_layout dl_ab>
    static __device__ __forceinline__ void mma(
            tile<16, 16, int, dl_d> & D, const tile<16, 4, int, dl_ab> & A, const tile<16, 4, int, dl_ab> & B) {
        using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
        int32x8_t * acc = (int32x8_t *) D.x;
        using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
        int32x4_t * a_vec = (int32x4_t *) A.x;
        int32x4_t * b_vec = (int32x4_t *) B.x;
        acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[0], true, b_vec[0], acc[0], false);
    }
}
