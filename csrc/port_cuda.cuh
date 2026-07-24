#pragma once
//
// CUDA compatibility shim for the NVIDIA (CuTe/PTX) port of torch-ggml-ops.
//
// The upstream sources target ROCm/HIP (gfx1151). This header lets the SAME
// sources compile with nvcc by mapping the small HIP name surface onto CUDA.
// It is included by common.cuh / the ck/ backward headers / mmq_hip.cu ONLY on
// the CUDA path (see the `#if defined(__HIP__)` guards there); under hipcc the
// original HIP headers are used unchanged, so the AMD build is preserved.
//
// Ported for RTX 4090 Laptop (Ada, sm_89), CUDA 13.3 toolchain. 2026-07.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// --- bfloat16 type names -------------------------------------------------
// HIP calls them __hip_bfloat16 / __hip_bfloat162; CUDA provides the __nv_
// variants with an identical scalar layout and the same __float2bfloat16 /
// __bfloat162float device helpers.
using __hip_bfloat16  = __nv_bfloat16;
using __hip_bfloat162 = __nv_bfloat162;

// --- runtime handle / error names ---------------------------------------
using hipStream_t = cudaStream_t;
using hipError_t  = cudaError_t;
static const cudaError_t hipSuccess = cudaSuccess;

static inline const char * hipGetErrorString(cudaError_t e) {
    return cudaGetErrorString(e);
}
static inline cudaError_t hipGetLastError() {
    return cudaGetLastError();
}
