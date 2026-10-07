#pragma once

#include "mmq_bundle.h"
#include "mmq_dense_validation.cuh"

#include <torch/csrc/stable/accelerator.h>

namespace {

// The split-contraction dense backward writes one FP32 partial tile per slice
// and a second launch reduces them into the BF16 gradient. Both buffers are
// allocated by the Python caller so an enclosing compiled graph can plan them.
void mmq_grad_input_split_launch_cuda(
        const Tensor & grad_output,
        const Tensor & packed_weight,
        int64_t quant_type,
        int64_t in_features,
        Tensor partials) {
    const DenseMMQShape shape = validate_dense_mmq_backward(
        grad_output, packed_weight, quant_type, in_features);
    const int slices = torch_ggml_ops::mmq_bundle::exact_deployment_split_slices(
        torch_ggml_ops::mmq_bundle::kOrdinaryBackward,
        static_cast<std::int32_t>(quant_type),
        shape.rows,
        shape.out_features,
        shape.in_features);
    STD_TORCH_CHECK(
        slices > 0,
        "the deployed dense backward for this key is not a split-contraction control");
    validate_explicit_buffer(
        partials,
        grad_output,
        ScalarType::Float,
        static_cast<int64_t>(slices) * shape.rows * shape.in_features,
        "partials");
    torch::stable::accelerator::DeviceGuard guard(grad_output.get_device_index());
    torch_ggml_ops::mmq_bundle::launch_dense_backward_split(
        static_cast<std::int32_t>(quant_type),
        grad_output.const_data_ptr(),
        static_cast<const char *>(packed_weight.const_data_ptr()),
        partials.mutable_data_ptr(),
        shape.rows,
        shape.out_features,
        shape.in_features,
        shape.weight_block_values,
        current_stream(grad_output));
}

void mmq_grad_input_split_reduce_launch_cuda(
        const Tensor & partials,
        Tensor grad_input,
        int64_t rows,
        int64_t in_features,
        int64_t slices) {
    STD_TORCH_CHECK(rows > 0 && in_features > 0, "the reduction needs a positive shape");
    STD_TORCH_CHECK(slices > 0, "the reduction needs at least one slice");
    validate_explicit_buffer(
        partials,
        partials,
        ScalarType::Float,
        slices * rows * in_features,
        "partials");
    validate_explicit_buffer(
        grad_input,
        partials,
        ScalarType::BFloat16,
        rows * in_features,
        "grad_input");
    torch::stable::accelerator::DeviceGuard guard(partials.get_device_index());
    torch_ggml_ops::mmq_bundle::launch_dense_backward_split_reduce(
        partials.const_data_ptr(),
        grad_input.mutable_data_ptr(),
        static_cast<int>(rows),
        static_cast<int>(in_features),
        static_cast<int>(slices),
        current_stream(partials));
}

} // namespace
