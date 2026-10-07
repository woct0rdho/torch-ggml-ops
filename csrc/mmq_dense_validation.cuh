#pragma once

#include "mmq_bundle.h"
#include "mmq_tensor_validation.cuh"

#include <limits>

namespace {

struct DenseMMQShape {
    int rows;
    int out_features;
    int in_features;
    int64_t workspace_bytes;
};

DenseMMQShape validate_dense_mmq(
        const Tensor & input,
        const Tensor & packed_weight,
        int64_t quant_type,
        int64_t out_features) {
    STD_TORCH_CHECK(input.is_cuda(), "input must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(
        input.get_device_index() == packed_weight.get_device_index(),
        "input and packed_weight must be on the same device");
    STD_TORCH_CHECK(input.scalar_type() == ScalarType::BFloat16, "input must be BF16");
    STD_TORCH_CHECK(packed_weight.scalar_type() == ScalarType::Byte, "packed_weight must be uint8");
    STD_TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    STD_TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous");
    STD_TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    STD_TORCH_CHECK(
        packed_weight.dim() == 2,
        "packed_weight must have shape [out_features, row_bytes]");
    const int64_t in_features = input.size(input.dim() - 1);
    STD_TORCH_CHECK(
        in_features > 0 && in_features % QK_K == 0,
        "input width must be a positive multiple of 256");
    STD_TORCH_CHECK(out_features > 0, "out_features must be positive");
    const int64_t rows = input.numel() / in_features;
    STD_TORCH_CHECK(rows > 0, "zero-row inputs are not supported");
    STD_TORCH_CHECK(
        rows <= std::numeric_limits<int>::max() &&
            in_features <= std::numeric_limits<int>::max() &&
            out_features <= std::numeric_limits<int>::max(),
        "dense MMQ dimensions exceed the kernel limit");
    const int64_t row_bytes = packed_row_bytes(quant_type, in_features);
    STD_TORCH_CHECK(
        packed_weight.size(0) == out_features &&
            packed_weight.size(1) == row_bytes,
        "packed_weight shape does not match the exact problem");
    STD_TORCH_CHECK(
        packed_weight.numel() == out_features * row_bytes,
        "packed_weight byte count does not match the exact problem");
    STD_TORCH_CHECK(
        reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 16 == 0,
        "input data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(
        reinterpret_cast<uintptr_t>(packed_weight.const_data_ptr()) % 16 == 0,
        "packed_weight data pointer must be 16-byte aligned");
    torch_ggml_ops::mmq_bundle::require_exact_deployment(
        torch_ggml_ops::mmq_bundle::kOrdinaryForward,
        static_cast<int32_t>(quant_type),
        static_cast<int>(rows),
        static_cast<int>(out_features),
        static_cast<int>(in_features));
    return {
        static_cast<int>(rows),
        static_cast<int>(out_features),
        static_cast<int>(in_features),
        rows * (in_features / kQuantWorkspaceBlockValues) *
            kQuantWorkspaceBlockBytes,
    };
}

DenseMMQShape validate_dense_mmq_backward(
        const Tensor & grad_output,
        const Tensor & packed_weight,
        int64_t quant_type,
        int64_t in_features) {
    STD_TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be a CUDA/HIP tensor");
    STD_TORCH_CHECK(
        grad_output.get_device_index() == packed_weight.get_device_index(),
        "grad_output and packed_weight must be on the same device");
    STD_TORCH_CHECK(grad_output.scalar_type() == ScalarType::BFloat16, "grad_output must be BF16");
    STD_TORCH_CHECK(packed_weight.scalar_type() == ScalarType::Byte, "packed_weight must be uint8");
    STD_TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    STD_TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous");
    STD_TORCH_CHECK(grad_output.dim() >= 1, "grad_output must have at least one dimension");
    STD_TORCH_CHECK(
        packed_weight.dim() == 2,
        "packed_weight must have shape [out_features, row_bytes]");
    STD_TORCH_CHECK(
        in_features > 0 && in_features % QK_K == 0,
        "in_features must be a positive multiple of 256");
    const int64_t out_features = grad_output.size(grad_output.dim() - 1);
    STD_TORCH_CHECK(out_features > 0, "gradient width must be positive");
    const int64_t rows = grad_output.numel() / out_features;
    STD_TORCH_CHECK(rows > 0, "zero-row gradients are not supported");
    STD_TORCH_CHECK(
        rows <= std::numeric_limits<int>::max() &&
            in_features <= std::numeric_limits<int>::max() &&
            out_features <= std::numeric_limits<int>::max(),
        "dense MMQ dimensions exceed the kernel limit");
    const int64_t row_bytes = packed_row_bytes(quant_type, in_features);
    STD_TORCH_CHECK(
        packed_weight.size(0) == out_features &&
            packed_weight.size(1) == row_bytes,
        "packed_weight shape does not match the exact backward problem");
    STD_TORCH_CHECK(
        packed_weight.numel() == out_features * row_bytes,
        "packed_weight byte count does not match the exact backward problem");
    STD_TORCH_CHECK(
        reinterpret_cast<uintptr_t>(grad_output.const_data_ptr()) % 16 == 0,
        "grad_output data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(
        reinterpret_cast<uintptr_t>(packed_weight.const_data_ptr()) % 16 == 0,
        "packed_weight data pointer must be 16-byte aligned");
    torch_ggml_ops::mmq_bundle::require_exact_deployment(
        torch_ggml_ops::mmq_bundle::kOrdinaryBackward,
        static_cast<int32_t>(quant_type),
        static_cast<int>(rows),
        static_cast<int>(out_features),
        static_cast<int>(in_features));
    return {
        static_cast<int>(rows),
        static_cast<int>(out_features),
        static_cast<int>(in_features),
        0,
    };
}

} // namespace
