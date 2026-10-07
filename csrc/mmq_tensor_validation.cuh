#pragma once

#include "mmq_quant_layout.cuh"

#include <hip/hip_runtime.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstdint>

namespace {

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

constexpr int64_t kQuantWorkspaceBlockBytes = 144;
constexpr int64_t kQuantWorkspaceBlockValues = 4 * QK8_1;

hipStream_t current_stream(const Tensor & tensor) {
    void * stream_pointer = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream_pointer));
    return static_cast<hipStream_t>(stream_pointer);
}

void validate_explicit_buffer(
        const Tensor & buffer,
        const Tensor & reference,
        ScalarType dtype,
        int64_t elements,
        const char * name) {
    STD_TORCH_CHECK(buffer.is_cuda(), name, " must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(
        buffer.get_device_index() == reference.get_device_index(),
        name,
        " must be on the input device");
    STD_TORCH_CHECK(buffer.scalar_type() == dtype, name, " has an invalid dtype");
    STD_TORCH_CHECK(buffer.is_contiguous(), name, " must be contiguous");
    STD_TORCH_CHECK(buffer.numel() == elements, name, " has an invalid element count");
    STD_TORCH_CHECK(
        reinterpret_cast<uintptr_t>(buffer.const_data_ptr()) % 16 == 0,
        name,
        " data pointer must be 16-byte aligned");
}

void validate_explicit_vector(
        const Tensor & buffer,
        const Tensor & reference,
        ScalarType dtype,
        int64_t elements,
        const char * name) {
    validate_explicit_buffer(buffer, reference, dtype, elements, name);
    STD_TORCH_CHECK(
        buffer.dim() == 1 && buffer.size(0) == elements,
        name,
        " must have the exact one-dimensional shape");
}

void validate_replaced_final_dimension(
        const Tensor & output,
        const Tensor & reference,
        int64_t final_dimension,
        const char * name) {
    STD_TORCH_CHECK(output.dim() == reference.dim(), name, " has an invalid rank");
    for (int64_t dimension = 0; dimension + 1 < reference.dim(); ++dimension) {
        STD_TORCH_CHECK(
            output.size(dimension) == reference.size(dimension),
            name,
            " has an invalid shape");
    }
    STD_TORCH_CHECK(
        output.size(output.dim() - 1) == final_dimension,
        name,
        " has an invalid final dimension");
}
} // namespace
