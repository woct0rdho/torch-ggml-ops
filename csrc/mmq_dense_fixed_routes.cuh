void mmq_launch_cuda(
        const Tensor & input,
        const Tensor & packed_weight,
        int64_t quant_type,
        int64_t out_features,
        Tensor output,
        Tensor workspace) {
    const DenseMMQShape shape =
        validate_dense_mmq(input, packed_weight, quant_type, out_features);
    validate_explicit_buffer(
        output,
        input,
        ScalarType::BFloat16,
        static_cast<int64_t>(shape.rows) * shape.out_features,
        "output");
    validate_replaced_final_dimension(output, input, shape.out_features, "output");
    validate_explicit_vector(
        workspace, input, ScalarType::Byte, shape.workspace_bytes, "workspace");
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
    torch_ggml_ops::mmq_bundle::launch_dense_forward(
        static_cast<int32_t>(quant_type),
        static_cast<const char *>(packed_weight.const_data_ptr()),
        static_cast<const int *>(workspace.const_data_ptr()),
        static_cast<__hip_bfloat16 *>(output.mutable_data_ptr()),
        shape.rows,
        shape.rows,
        shape.in_features,
        shape.out_features,
        stream);
}

void mmq_grad_input_launch_cuda(
        const Tensor & grad_output,
        const Tensor & packed_weight,
        int64_t quant_type,
        int64_t in_features,
        Tensor grad_input) {
    const DenseMMQShape shape = validate_dense_mmq_backward(
        grad_output, packed_weight, quant_type, in_features);
    validate_explicit_buffer(
        grad_input,
        grad_output,
        ScalarType::BFloat16,
        static_cast<int64_t>(shape.rows) * shape.in_features,
        "grad_input");
    validate_replaced_final_dimension(
        grad_input, grad_output, shape.in_features, "grad_input");
    torch::stable::accelerator::DeviceGuard guard(grad_output.get_device_index());
    torch_ggml_ops::mmq_bundle::launch_dense_backward(
        static_cast<int32_t>(quant_type),
        static_cast<const __hip_bfloat16 *>(grad_output.const_data_ptr()),
        static_cast<const char *>(packed_weight.const_data_ptr()),
        static_cast<__hip_bfloat16 *>(grad_input.mutable_data_ptr()),
        shape.rows,
        shape.out_features,
        shape.in_features,
        current_stream(grad_output));
}

void fixed_grouped_mmq_launch_cuda(
        const Tensor & input,
        const Tensor & packed_weight,
        Tensor output,
        Tensor workspace) {
    const FixedMMQShape shape = validate_fixed_mmq(input, packed_weight, false);
    const int64_t workspace_bytes = static_cast<int64_t>(shape.total_rows) *
        (4096 / kQuantWorkspaceBlockValues) * kQuantWorkspaceBlockBytes;
    validate_explicit_buffer(
        output,
        input,
        ScalarType::BFloat16,
        static_cast<int64_t>(shape.total_rows) * shape.out_features,
        "output");
    validate_replaced_final_dimension(output, input, shape.out_features, "output");
    validate_explicit_vector(
        workspace, input, ScalarType::Byte, workspace_bytes, "workspace");
    torch::stable::accelerator::DeviceGuard guard(input.get_device_index());
    hipStream_t stream = current_stream(input);
    torch_ggml_ops::mmq_bundle::launch_quantize(
        GGML_TYPE_Q8_0,
        static_cast<const __hip_bfloat16 *>(input.const_data_ptr()),
        workspace.mutable_data_ptr(),
        shape.total_rows,
        shape.total_rows,
        4096,
        stream);
    torch_ggml_ops::mmq_bundle::launch_fixed_grouped_forward(
        static_cast<const char *>(packed_weight.const_data_ptr()),
        static_cast<const int *>(workspace.const_data_ptr()),
        static_cast<__hip_bfloat16 *>(output.mutable_data_ptr()),
        shape.tokens,
        shape.out_features,
        shape.bytes_per_group,
        stream);
}

void fixed_grouped_mmq_grad_input_launch_cuda(
        const Tensor & grad_output,
        const Tensor & packed_weight,
        Tensor grad_input) {
    const FixedMMQShape shape =
        validate_fixed_mmq(grad_output, packed_weight, true);
    validate_explicit_buffer(
        grad_input,
        grad_output,
        ScalarType::BFloat16,
        static_cast<int64_t>(shape.total_rows) * 4096,
        "grad_input");
    validate_replaced_final_dimension(grad_input, grad_output, 4096, "grad_input");
    torch::stable::accelerator::DeviceGuard guard(grad_output.get_device_index());
    torch_ggml_ops::mmq_bundle::launch_fixed_grouped_backward(
        grad_output.const_data_ptr(),
        static_cast<const char *>(packed_weight.const_data_ptr()),
        grad_input.mutable_data_ptr(),
        shape.tokens,
        shape.out_features,
        shape.bytes_per_group,
        current_stream(grad_output));
}

