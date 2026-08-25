void grouped_mmq_launch_cuda(
        const Tensor & input,
        const Tensor & packed_weight,
        const Tensor & expert_indices,
        const Tensor & expert_offsets,
        int64_t quant_type,
        int64_t out_features,
        Tensor output,
        Tensor workspace) {
    const GroupedMMQShape shape = validate_grouped_mmq(
        input,
        packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        out_features);
    torch_ggml_ops::mmq_bundle::require_exact_deployment(
        torch_ggml_ops::mmq_bundle::kGroupedForward,
        static_cast<int32_t>(quant_type), shape.rows,
        shape.out_features, shape.in_features);
    const int64_t workspace_bytes = static_cast<int64_t>(shape.rows) *
        (shape.in_features / kQuantWorkspaceBlockValues) *
        kQuantWorkspaceBlockBytes;
    validate_explicit_buffer(
        output,
        input,
        ScalarType::BFloat16,
        static_cast<int64_t>(shape.rows) * shape.out_features,
        "output");
    STD_TORCH_CHECK(
        output.dim() == 2 && output.size(0) == shape.rows &&
            output.size(1) == shape.out_features,
        "output has an invalid grouped shape");
    validate_explicit_vector(
        workspace, input, ScalarType::Byte, workspace_bytes, "workspace");
    torch::stable::accelerator::DeviceGuard guard(input.get_device_index());
    hipStream_t stream = current_stream(input);
    torch_ggml_ops::mmq_bundle::launch_quantize(
        static_cast<int32_t>(quant_type),
        static_cast<const __hip_bfloat16 *>(input.const_data_ptr()),
        workspace.mutable_data_ptr(),
        shape.rows,
        shape.rows,
        shape.in_features,
        stream);
    torch_ggml_ops::mmq_bundle::launch_grouped_forward(
        static_cast<int32_t>(quant_type),
        static_cast<const char *>(packed_weight.const_data_ptr()),
        static_cast<const int *>(workspace.const_data_ptr()),
        static_cast<__hip_bfloat16 *>(output.mutable_data_ptr()),
        static_cast<const int64_t *>(expert_indices.const_data_ptr()),
        static_cast<const int32_t *>(expert_offsets.const_data_ptr()),
        shape.num_experts,
        shape.num_groups,
        shape.rows,
        shape.in_features,
        shape.out_features,
        shape.bytes_per_expert,
        stream);
}

void grouped_mmq_grad_input_launch_cuda(
        const Tensor & grad_output,
        const Tensor & packed_weight,
        const Tensor & expert_indices,
        const Tensor & expert_offsets,
        int64_t quant_type,
        int64_t in_features,
        Tensor grad_input) {
    const GroupedMMQShape shape = validate_grouped_mmq_grad_input(
        grad_output,
        packed_weight,
        expert_indices,
        expert_offsets,
        quant_type,
        in_features);
    torch_ggml_ops::mmq_bundle::require_exact_deployment(
        torch_ggml_ops::mmq_bundle::kGroupedBackward,
        static_cast<int32_t>(quant_type), shape.rows,
        shape.in_features, shape.out_features);
    validate_explicit_buffer(
        grad_input,
        grad_output,
        ScalarType::BFloat16,
        static_cast<int64_t>(shape.rows) * shape.in_features,
        "grad_input");
    STD_TORCH_CHECK(
        grad_input.dim() == 2 && grad_input.size(0) == shape.rows &&
            grad_input.size(1) == shape.in_features,
        "grad_input has an invalid grouped shape");
    torch::stable::accelerator::DeviceGuard guard(grad_output.get_device_index());
    torch_ggml_ops::mmq_bundle::launch_grouped_backward(
        static_cast<int32_t>(quant_type),
        static_cast<const __hip_bfloat16 *>(grad_output.const_data_ptr()),
        static_cast<const char *>(packed_weight.const_data_ptr()),
        static_cast<__hip_bfloat16 *>(grad_input.mutable_data_ptr()),
        static_cast<const int64_t *>(expert_indices.const_data_ptr()),
        static_cast<const int32_t *>(expert_offsets.const_data_ptr()),
        shape.num_experts,
        shape.num_groups,
        shape.rows,
        shape.out_features,
        shape.in_features,
        shape.bytes_per_expert,
        current_stream(grad_output));
}

